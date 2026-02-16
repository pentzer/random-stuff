"""
Single entrypoint for Polymarket Tape 1 live recording (LEGACY).

DEPRECATED: Use src.polymarket.tape1.run_tape1_pipeline instead.

This module provides backward compatibility for --family BTC_15M_UPDOWN,
but lacks normalization and gate integration. For full CAI compliance with
health reporting and gate verification, use run_tape1_pipeline.

Usage (legacy, not recommended):
    python -m src.polymarket.run_live --family BTC_15M_UPDOWN --duration-seconds 300

Recommended instead:
    python -m src.polymarket.tape1.run_tape1_pipeline --family BTC_15M_UPDOWN --duration-minutes 5
    python -m src.polymarket.run_dual_live --duration-minutes 5  # For dual 5m+15m
"""

import argparse
import logging
import sys
from pathlib import Path

from .tape1.config import ProbabilityTapeConfig
from .tape1.families import list_families, validate_family, get_family_spec
from .tape1.ui_discovery import UIDiscovery
from .tape1.recorder_raw import Tape1Recorder


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for Tape 1 pipeline (legacy)."""
    parser = argparse.ArgumentParser(
        description="Polymarket Tape 1: UI discovery + REST recording (LEGACY - missing normalization + gate)",
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
        "--duration-seconds",
        type=float,
        default=300,
        help="Recording duration in seconds",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="data/polymarket/tape1",
        help="Output directory (will create subdirs for discovery/raw/normalized/gate)",
    )

    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with non-zero status if gate FAILS (not implemented in legacy mode)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("⚠️  DEPRECATED: Polymarket Tape 1 - Legacy Recording Only")
    logger.info("=" * 80)
    logger.info("This entry point does NOT include normalization + gate.")
    logger.info("For full CAI-compliant pipeline, use:")
    logger.info("  python -m src.polymarket.tape1.run_tape1_pipeline ...")
    logger.info("=" * 80)
    logger.info(f"Family: {args.family}")
    logger.info(f"Duration: {args.duration_seconds}s")
    logger.info(f"Output dir: {args.outdir}")
    logger.info(f"Require pass: {args.require_pass}")
    logger.info("=" * 80)

    # Validate family
    if not validate_family(args.family):
        logger.error(f"Unsupported family: {args.family}")
        logger.error(f"Supported families: {', '.join(list_families())}")
        return 1

    family_spec = get_family_spec(args.family)

    # Create config
    workspace_root = Path.cwd()
    config = ProbabilityTapeConfig(
        workspace_root=workspace_root,
        poll_interval_seconds=1.0,
        request_timeout_seconds=5.0,
    )

    # Adjust output dirs if custom outdir specified
    if args.outdir != "data/polymarket/tape1":
        base_dir = Path(args.outdir)
        config.raw_output_dir = base_dir / "raw"
        config.normalized_output_dir = base_dir / "normalized"
        config.gate_output_dir = base_dir / "gate"
        config.discovery_output_dir = base_dir / "discovery"
        config.views_output_dir = base_dir / "views"

    # ========================================================================
    # PHASE 1: UI DISCOVERY
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PHASE 1: UI DISCOVERY ({args.family})")
    logger.info("=" * 80)

    discovery_client = UIDiscovery(timeout_seconds=10.0)

    try:
        snapshot, debug_info = discovery_client.discover_by_family(
            family_id=args.family,
            max_windows_ahead=family_spec.max_windows_ahead_default,
        )
    except Exception as e:
        logger.error(f"Discovery failed with exception: {e}")
        return 1

    if snapshot is None:
        logger.error("Discovery FAILED: No valid market found")
        logger.error(f"Attempted slugs: {debug_info.attempted_slugs}")
        logger.error(f"Error: {debug_info.error}")
        return 1

    logger.info(f"✓ Discovery SUCCESS")
    logger.info(f"  Slug: {snapshot.slug}")
    logger.info(f"  Condition ID: {snapshot.condition_id}")
    logger.info(f"  Token YES: {snapshot.token_yes_id}")
    logger.info(f"  Token NO: {snapshot.token_no_id}")
    if snapshot.expiry_ts_utc:
        logger.info(f"  Expiry: {snapshot.expiry_ts_utc}")
    if snapshot.question:
        logger.info(f"  Question: {snapshot.question}")

    # Write discovery outputs
    config.discovery_output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    from datetime import datetime, timezone
    
    run_id_base = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    resolution_file = config.discovery_output_dir / f"discovery_resolution_{run_id_base}.json"
    with open(resolution_file, 'w') as f:
        f.write(snapshot.to_json())
    logger.info(f"  Wrote: {resolution_file}")

    debug_file = config.discovery_output_dir / f"discovery_debug_{run_id_base}.json"
    with open(debug_file, 'w') as f:
        f.write(debug_info.to_json())
    logger.info(f"  Wrote: {debug_file}")

    # ========================================================================
    # PHASE 2: RAW RECORDING
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: RAW RECORDING (TAPE 1)")
    logger.info("=" * 80)

    recorder = Tape1Recorder(
        config=config,
        token_yes_id=snapshot.token_yes_id,
        token_no_id=snapshot.token_no_id,
        slug=snapshot.slug,
        condition_id=snapshot.condition_id,
        family_id=args.family,
        expiry_ts_utc=snapshot.expiry_ts_utc,
    )

    try:
        recorder_metadata = recorder.record(duration_seconds=args.duration_seconds)
    except Exception as e:
        logger.error(f"Recording FAILED with exception: {e}")
        return 1

    logger.info(f"✓ Recording SUCCESS")
    logger.info(f"  Run ID: {recorder_metadata['run_id']}")
    logger.info(f"  Actual duration: {recorder_metadata['duration_actual_seconds']:.1f}s")
    logger.info(f"  Total polls: {recorder_metadata['total_polls']}")
    logger.info(f"  Success rate: {recorder_metadata['success_rate_percent']:.1f}%")

    # ========================================================================
    # PHASE 3: NORMALIZATION (TODO - using existing module for now)
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3: NORMALIZATION (1-second grid)")
    logger.info("=" * 80)
    logger.info("⚠ Normalization module integration pending - skipping for now")
    logger.info("  Raw data is available for manual normalization")
    logger.info("  Use src.polymarket.tape1.run_tape1_pipeline for full pipeline")
    
    # TODO: Integrate normalizer when ready
    # normalizer = Tape1Normalizer(...)
    # normalized_path = normalizer.normalize(recorder_metadata['run_id'])

    # ========================================================================
    # PHASE 4: HEALTH GATE (TODO)
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 4: HEALTH GATE")
    logger.info("=" * 80)
    logger.info("⚠ Gate module integration pending - skipping for now")
    logger.info("  Use src.polymarket.tape1.run_tape1_pipeline for full pipeline")
    
    # TODO: Integrate gate when ready
    # gate = Tape1Gate(...)
    # verdict = gate.evaluate()
    # if args.require_pass and verdict.verdict == "FAIL":
    #     logger.error("Gate FAILED and --require-pass set")
    #     return 1

    # ========================================================================
    # COMPLETE
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ LEGACY PIPELINE COMPLETE (discovery + recording only)")
    logger.info("=" * 80)
    logger.info(f"Discovery: {resolution_file}")
    logger.info(f"Raw data: {recorder_metadata['output_dir']}")
    logger.info("")
    logger.info("For complete CAI-PM-A1 compliance, use full pipeline:")
    logger.info(f"  python -m src.polymarket.tape1.run_tape1_pipeline --family {args.family}")
    logger.info("")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
