#!/usr/bin/env python3
"""
Tape 1 Pipeline - Complete end-to-end execution.

Implements CAI-PM-A1-RECORDING-ALIGNMENT-0001:
- UI-native discovery with __NEXT_DATA__ persistence
- Raw token price recording with provenance
- Canonical 1-second grid normalization (NO silent filling)
- Health and alignment reporting
- Proxy probability derivation

Supports multiple families (BTC_5M_UPDOWN, BTC_15M_UPDOWN) with independent
artifact sets per family.

USAGE:
    python -m src.polymarket.tape1.run_tape1_pipeline --family BTC_15M_UPDOWN --duration-minutes 5
    python -m src.polymarket.tape1.run_tape1_pipeline --family BTC_5M_UPDOWN --duration-minutes 5

EXIT CODES:
    0 = Success
    1 = Failure (empty/flat/truncated/stale tape, or gate fail)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import ProbabilityTapeConfig
from .families import get_family_spec, list_families, validate_family
from .ui_discovery import UIDiscovery, DiscoveryEvidence
from .recorder_raw import Tape1Recorder
from .normalize_1s import ProbabilityTapeNormalizer
from .health_reports import (
    generate_run_metadata,
    generate_missingness_report,
    generate_staleness_report,
    generate_flatline_report,
)
from .alignment_preview import generate_joinability_report
from .gate_v2 import HealthGateV2


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_run_directory(base_dir: Path, family_id: str) -> Path:
    """Create timestamped run directory with family-scoped subdirs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    
    # Create family-scoped subdirectories
    family_dir = run_dir / f"family={family_id}"
    (family_dir / "discovery").mkdir(parents=True, exist_ok=True)
    (family_dir / "tape1_raw").mkdir(parents=True, exist_ok=True)
    (family_dir / "tape1_1s").mkdir(parents=True, exist_ok=True)
    (family_dir / "health").mkdir(parents=True, exist_ok=True)
    (family_dir / "alignment_preview").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created run directory: {family_dir}")
    
    return family_dir


def run_discovery(run_dir: Path, family_id: str) -> tuple[Optional[dict], dict]:
    """
    Run UI discovery and persist results.
    
    Args:
        run_dir: Family-scoped run directory
        family_id: Family identifier (e.g., "BTC_5M_UPDOWN")
    
    Returns:
        (snapshot, token_mapping)
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 1: UI DISCOVERY ({family_id})")
    logger.info("=" * 80)
    
    discovery_dir = run_dir / "discovery"
    
    # Discover market for this family
    discovery = UIDiscovery(timeout_seconds=10.0)
    snapshot, debug_info = discovery.discover_by_family(family_id=family_id)
    
    if snapshot is None:
        logger.error("Discovery failed - no active market found")
        
        # Save debug info
        debug_file = discovery_dir / "discovery_debug.json"
        with open(debug_file, 'w') as f:
            f.write(debug_info.to_json())
        
        return None, {}
    
    logger.info(f"✓ Discovered market: {snapshot.slug}")
    logger.info(f"  Family: {family_id}")
    logger.info(f"  Condition ID: {snapshot.condition_id}")
    logger.info(f"  Token YES: {snapshot.token_yes_id}")
    logger.info(f"  Token NO: {snapshot.token_no_id}")
    
    # Save discovery snapshot
    snapshot_file = discovery_dir / "discovery_snapshot.json"
    with open(snapshot_file, 'w') as f:
        f.write(snapshot.to_json())
    
    # REQUIREMENT: Create discovery evidence artifact (v1)
    # Evidence is ephemeral + lossy, persisting only sufficient extract to justify token IDs
    evidence = discovery.create_discovery_evidence(
        snapshot=snapshot,
        extracted_market_slice=debug_info.extracted_market_slice,
        discovery_method="ui_probe_v1",
    )
    
    evidence_file = discovery_dir / "discovery_evidence.json"
    with open(evidence_file, 'w') as f:
        f.write(evidence.to_json())
    
    logger.info(f"✓ Saved discovery_evidence.json (v1)")
    logger.info(f"  Evidence SHA256: {evidence.to_sha256()}")
    
    # REQUIREMENT: Save raw __NEXT_DATA__ JSON (fetch it again to get full payload)
    # For now, we save the debug info which contains the extracted market slice
    # In production, we would save the full __NEXT_DATA__ payload
    next_data_file = discovery_dir / "next_data_raw.json"
    with open(next_data_file, 'w') as f:
        json.dump(debug_info.extracted_market_slice, f, indent=2)
    
    logger.info(f"✓ Saved next_data_raw.json")
    
    # REQUIREMENT: Create token_mapping.json with YES/NO semantic resolution
    # Map based on outcome meaning (UP = YES, DOWN = NO for BTC markets)
    # NOTE: Semantic binding is PER DISCOVERY SNAPSHOT, not global invariant
    token_mapping = {
        "token_yes_id": snapshot.token_yes_id,
        "token_no_id": snapshot.token_no_id,
        "token_mapping": {
            snapshot.token_yes_id: {
                "outcome": "UP",
                "semantic_label": "YES",
                "description": "BTC price increases",
                "binding_source": "discovery_snapshot"
            },
            snapshot.token_no_id: {
                "outcome": "DOWN",
                "semantic_label": "NO",
                "description": "BTC price decreases or stays same",
                "binding_source": "discovery_snapshot"
            }
        },
        "discovery_method": "ui_html_next_data",
        "discovered_at_utc": snapshot.resolved_at_utc,
        "slug": snapshot.slug,
        "condition_id": snapshot.condition_id,
        "discovery_evidence_sha256": evidence.to_sha256(),
    }
    
    token_mapping_file = discovery_dir / "token_mapping.json"
    with open(token_mapping_file, 'w') as f:
        json.dump(token_mapping, f, indent=2)
    
    logger.info(f"✓ Saved token_mapping.json")
    
    return snapshot.to_dict(), token_mapping


def run_recording(
    run_dir: Path,
    snapshot: dict,
    duration_seconds: float,
    family_id: str,
) -> Optional[dict]:
    """
    Run raw recording phase.
    
    Args:
        run_dir: Family-scoped run directory
        snapshot: Discovery snapshot
        duration_seconds: Recording duration
        family_id: Family identifier
    
    Returns:
        Recording metadata
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 2: RAW RECORDING ({family_id})")
    logger.info("=" * 80)
    
    tape1_raw_dir = run_dir / "tape1_raw"
    
    # Create config with proper output directory
    config = ProbabilityTapeConfig(
        workspace_root=Path.cwd(),
        raw_output_dir=tape1_raw_dir,
        poll_interval_seconds=1.0,
        request_timeout_seconds=5.0,
    )
    
    # Create recorder with family_id
    recorder = Tape1Recorder(
        config=config,
        token_yes_id=snapshot['token_yes_id'],
        token_no_id=snapshot['token_no_id'],
        slug=snapshot['slug'],
        condition_id=snapshot['condition_id'],
        family_id=family_id,
        expiry_ts_utc=snapshot.get('expiry_ts_utc'),
    )
    
    try:
        metadata = recorder.record(duration_seconds)
        logger.info(f"✓ Recording completed")
        return metadata
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return None


def run_normalization(
    run_dir: Path,
    snapshot: dict,
    token_mapping: dict,
    recording_metadata: dict,
    family_id: str,
) -> Optional[Path]:
    """
    Run 1-second grid normalization.
    
    Args:
        run_dir: Family-scoped run directory
        snapshot: Discovery snapshot
        token_mapping: Token mapping
        recording_metadata: Recording metadata
        family_id: Family identifier
    
    Returns:
        Path to normalized parquet file
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 3: 1-SECOND NORMALIZATION ({family_id})")
    logger.info("=" * 80)
    
    tape1_1s_dir = run_dir / "tape1_1s"
    tape1_raw_dir = run_dir / "tape1_raw"
    
    # Find the raw run directory (recorder creates a run_* subdirectory)
    raw_run_dirs = list(tape1_raw_dir.glob("run_*"))
    if not raw_run_dirs:
        logger.error("No raw run directory found")
        return None
    
    raw_run_dir = sorted(raw_run_dirs)[-1]  # Latest
    logger.info(f"Using raw data from: {raw_run_dir}")
    
    # Create config
    config = ProbabilityTapeConfig(
        workspace_root=Path.cwd(),
        raw_output_dir=raw_run_dir,  # Point to specific run
        normalized_output_dir=tape1_1s_dir,
    )
    
    # Create normalizer with family_id
    normalizer = ProbabilityTapeNormalizer(
        config=config,
        token_up_id=snapshot['token_yes_id'],
        token_down_id=snapshot['token_no_id'],
        family_id=family_id,
        resolution_id=snapshot['condition_id'],
        expiry_ts_utc=snapshot.get('expiry_ts_utc'),
    )
    
    try:
        output_file = normalizer.normalize()
        logger.info(f"✓ Normalization completed")
        
        # Rename to standard name
        standard_name = tape1_1s_dir / "tape1_1s.parquet"
        if output_file != standard_name:
            output_file.rename(standard_name)
            output_file = standard_name
        
        return output_file
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return None


def run_health_reports(
    run_dir: Path,
    snapshot: dict,
    recording_metadata: dict,
    normalized_file: Path,
) -> bool:
    """
    Generate health reports.
    
    Returns:
        True if all reports generated successfully
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: HEALTH REPORTS")
    logger.info("=" * 80)
    
    health_dir = run_dir / "health"
    
    try:
        # Run metadata
        run_metadata = generate_run_metadata(
            snapshot=snapshot,
            recording_metadata=recording_metadata,
            normalized_file=normalized_file,
        )
        
        run_metadata_file = health_dir / "run_metadata.json"
        with open(run_metadata_file, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        logger.info(f"✓ Saved run_metadata.json")
        
        # Missingness report
        missingness = generate_missingness_report(normalized_file)
        missingness_file = health_dir / "missingness_report.json"
        with open(missingness_file, 'w') as f:
            json.dump(missingness, f, indent=2)
        logger.info(f"✓ Saved missingness_report.json")
        
        # Staleness report
        staleness = generate_staleness_report(normalized_file)
        staleness_file = health_dir / "staleness_report.json"
        with open(staleness_file, 'w') as f:
            json.dump(staleness, f, indent=2)
        logger.info(f"✓ Saved staleness_report.json")
        
        # Flatline report
        flatline = generate_flatline_report(normalized_file)
        flatline_file = health_dir / "flatline_report.json"
        with open(flatline_file, 'w') as f:
            json.dump(flatline, f, indent=2)
        logger.info(f"✓ Saved flatline_report.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Health report generation failed: {e}")
        return False


def run_alignment_preview(
    run_dir: Path,
    normalized_file: Path,
) -> bool:
    """
    Generate alignment preview reports.
    
    Returns:
        True if all reports generated successfully
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: ALIGNMENT PREVIEW")
    logger.info("=" * 80)
    
    alignment_dir = run_dir / "alignment_preview"
    
    try:
        # Joinability report
        joinability = generate_joinability_report(normalized_file)
        joinability_file = alignment_dir / "joinability_report.json"
        with open(joinability_file, 'w') as f:
            json.dump(joinability, f, indent=2)
        logger.info(f"✓ Saved joinability_report.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Alignment preview generation failed: {e}")
        return False


def run_health_gate(
    run_dir: Path,
    normalized_file: Path,
    snapshot: dict,
    family_id: str,
) -> tuple[bool, dict]:
    """
    Run health gate evaluation.
    
    Args:
        run_dir: Family-scoped run directory
        normalized_file: Path to normalized tape
        snapshot: Discovery snapshot
        family_id: Family identifier
    
    Returns:
        (gate_passed, verdict_dict)
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 6: HEALTH GATE ({family_id})")
    logger.info("=" * 80)
    
    health_dir = run_dir / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    
    config = ProbabilityTapeConfig(workspace_root=Path.cwd())
    gate = HealthGateV2(
        config=config,
        normalized_file=normalized_file,
        family_id=family_id,
        resolution_id=snapshot.get('condition_id', 'unknown'),
    )
    
    try:
        verdict = gate.evaluate()
        
        # Write gate verdict
        verdict_file = health_dir / "gate_verdict.json"
        with open(verdict_file, 'w') as f:
            json.dump(verdict, f, indent=2)
        
        logger.info(f"✓ Saved gate_verdict.json")
        logger.info(f"Gate verdict: {verdict['verdict']}")
        
        gate_passed = verdict['verdict'] == 'PASS'
        
        if not gate_passed:
            logger.warning(f"Gate FAILED. Reasons:")
            for reason in verdict.get('failure_reasons', []):
                logger.warning(f"  - {reason}")
        
        return gate_passed, verdict
        
    except Exception as e:
        logger.error(f"Gate evaluation failed: {e}")
        return False, {'verdict': 'FAIL', 'error': str(e)}


def print_health_summary(run_dir: Path) -> None:
    """Print key health metrics to stdout."""
    logger.info("=" * 80)
    logger.info("HEALTH SUMMARY")
    logger.info("=" * 80)
    
    health_dir = run_dir / "health"
    
    # Load reports
    run_metadata_file = health_dir / "run_metadata.json"
    missingness_file = health_dir / "missingness_report.json"
    staleness_file = health_dir / "staleness_report.json"
    
    if run_metadata_file.exists():
        with open(run_metadata_file) as f:
            run_meta = json.load(f)
        
        recording = run_meta.get('recording', {})
        normalization = run_meta.get('normalization', {})
        
        logger.info(f"Duration: {recording.get('duration_actual_seconds', 'N/A')}s (requested: {recording.get('duration_requested_seconds', 'N/A')}s)")
        logger.info(f"Total polls: {recording.get('total_polls', 'N/A')}")
        logger.info(f"Success rate: {recording.get('success_rate_percent', 'N/A')}%")
        logger.info(f"Normalized seconds: {normalization.get('normalized_seconds', 'N/A')}")
    
    if missingness_file.exists():
        with open(missingness_file) as f:
            miss = json.load(f)
        
        logger.info(f"Coverage (both tokens): {miss.get('percent_both_tokens', 'N/A')}%")
        logger.info(f"Missing seconds: {miss.get('seconds_with_neither', 'N/A')}")
    
    if staleness_file.exists():
        with open(staleness_file) as f:
            stale = json.load(f)
        
        logger.info(f"Max staleness: YES={stale.get('max_yes_staleness_seconds', 'N/A')}s, NO={stale.get('max_no_staleness_seconds', 'N/A')}s")
    
    logger.info("=" * 80)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Tape 1 Pipeline - Complete end-to-end execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--family",
        type=str,
        default="BTC_15M_UPDOWN",
        choices=list_families(),
        help=f"Family ID to record. Options: {', '.join(list_families())}",
    )
    
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=5.0,
        help="Recording duration in minutes",
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/polymarket/tape1_runs",
        help="Base output directory for runs",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Validate family
    if not validate_family(args.family):
        logger.error(f"Invalid family: {args.family}")
        logger.error(f"Supported families: {', '.join(list_families())}")
        return 1
    
    family_spec = get_family_spec(args.family)
    
    setup_logging(args.verbose)
    
    duration_seconds = args.duration_minutes * 60
    base_dir = Path(args.outdir)
    
    logger.info("=" * 80)
    logger.info("POLYMARKET TAPE 1 PIPELINE")
    logger.info("CAI-PM-A1-RECORDING-ALIGNMENT-0001")
    logger.info("=" * 80)
    logger.info(f"Family: {args.family}")
    logger.info(f"  Window: {family_spec.window_seconds}s")
    logger.info(f"  Gate min duration: {family_spec.gate_min_run_duration_seconds}s")
    logger.info(f"Duration: {args.duration_minutes} minutes ({duration_seconds}s)")
    logger.info(f"Output: {base_dir}")
    logger.info("=" * 80)
    
    # Create run directory (family-scoped)
    run_dir = create_run_directory(base_dir, args.family)
    
    # Phase 1: Discovery
    snapshot, token_mapping = run_discovery(run_dir, args.family)
    if snapshot is None:
        logger.error("❌ Pipeline failed at discovery")
        return 1
    
    # Phase 2: Recording
    recording_metadata = run_recording(run_dir, snapshot, duration_seconds, args.family)
    if recording_metadata is None:
        logger.error("❌ Pipeline failed at recording")
        return 1
    
    # Phase 3: Normalization
    normalized_file = run_normalization(run_dir, snapshot, token_mapping, recording_metadata, args.family)
    if normalized_file is None:
        logger.error("❌ Pipeline failed at normalization")
        return 1
    
    # Phase 4: Health reports
    health_ok = run_health_reports(run_dir, snapshot, recording_metadata, normalized_file)
    if not health_ok:
        logger.error("❌ Pipeline failed at health reporting")
        return 1
    
    # Phase 5: Alignment preview
    alignment_ok = run_alignment_preview(run_dir, normalized_file)
    if not alignment_ok:
        logger.error("❌ Pipeline failed at alignment preview")
        return 1
    
    # Phase 6: Health gate
    gate_passed, gate_verdict = run_health_gate(run_dir, normalized_file, snapshot, args.family)
    if not gate_passed:
        logger.warning(f"⚠️  Pipeline completed but gate FAILED")
        # Don't return 1 here - we still want to produce outputs
        # But mark in exit code
    
    # Print summary
    print_health_summary(run_dir)
    
    logger.info("=" * 80)
    if gate_passed:
        logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY (Gate: PASS)")
    else:
        logger.info(f"⚠️  PIPELINE COMPLETED (Gate: FAIL)")
    logger.info(f"✓ All deliverables written to: {run_dir}")
    logger.info("=" * 80)
    
    # Return non-zero if gate failed
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
