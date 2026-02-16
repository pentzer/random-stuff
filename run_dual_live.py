"""
Dual recording orchestrator for Polymarket Tape 1.

Spawns independent processes to record both BTC_5M_UPDOWN and BTC_15M_UPDOWN
families in parallel, writing to separate family-scoped directories within
a shared session directory.

This implements the "Option A (recommended)" approach: supervisor spawns two
independent processes instead of blending into one tape.

Benefits:
- No shared state / event loop contention
- Clean audit surface: each family's artifacts are isolated
- Prevents accidental complementarity / fill behavior
- Easier to debug and reason about

Usage:
    python -m src.polymarket.run_dual_live --duration-minutes 5

Exit codes:
    0 = Both families succeeded
    1 = At least one family failed
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_session_directory(base_dir: Path) -> Path:
    """
    Create timestamped session directory.
    
    Each session contains separate family-scoped subdirectories.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created session directory: {session_dir}")
    
    return session_dir


def spawn_family_recorder(
    family_id: str,
    session_dir: Path,
    duration_minutes: float,
    verbose: bool = False,
) -> subprocess.Popen:
    """
    Spawn a recorder process for a single family.
    
    Args:
        family_id: Family identifier (e.g., "BTC_5M_UPDOWN")
        session_dir: Session directory for output
        duration_minutes: Recording duration in minutes
        verbose: Enable verbose logging
    
    Returns:
        Popen object for the spawned process
    """
    cmd = [
        sys.executable,
        "-m",
        "src.polymarket.tape1.run_tape1_pipeline",
        "--family",
        family_id,
        "--duration-minutes",
        str(duration_minutes),
        "--outdir",
        str(session_dir),
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    logger.info(f"Spawning recorder for {family_id}")
    logger.info(f"  Command: {' '.join(cmd)}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    
    return proc


def stream_output(proc: subprocess.Popen, family_id: str) -> None:
    """
    Stream output from a process with family ID prefix.
    
    Args:
        proc: Subprocess to stream from
        family_id: Family ID for logging prefix
    """
    if proc.stdout is None:
        return
    
    prefix = f"[{family_id}]"
    for line in proc.stdout:
        line = line.rstrip('\n')
        logger.info(f"{prefix} {line}")


def run_dual_recording(
    base_dir: Path,
    duration_minutes: float,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Run dual recording for BTC_5M_UPDOWN and BTC_15M_UPDOWN.
    
    Args:
        base_dir: Base output directory
        duration_minutes: Recording duration in minutes
        verbose: Enable verbose logging
    
    Returns:
        Dictionary mapping family_id -> exit_code
    """
    logger.info("=" * 80)
    logger.info("DUAL RECORDING ORCHESTRATOR")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration_minutes} minutes")
    logger.info(f"Base output dir: {base_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    # Create session directory
    session_dir = create_session_directory(base_dir)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("SPAWNING RECORDERS")
    logger.info("=" * 80)
    
    families = ["BTC_5M_UPDOWN", "BTC_15M_UPDOWN"]
    processes = {}
    
    # Spawn both recorders
    for family_id in families:
        proc = spawn_family_recorder(
            family_id=family_id,
            session_dir=session_dir,
            duration_minutes=duration_minutes,
            verbose=verbose,
        )
        processes[family_id] = proc
        logger.info(f"✓ Spawned {family_id} (PID: {proc.pid})")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("MONITORING RECORDERS")
    logger.info("=" * 80)
    
    # Wait for all processes to complete and stream output
    exit_codes = {}
    
    for family_id, proc in processes.items():
        logger.info(f"Waiting for {family_id}...")
        
        # Stream output with prefix
        stream_output(proc, family_id)
        
        # Wait and collect exit code
        exit_code = proc.wait()
        exit_codes[family_id] = exit_code
        
        if exit_code == 0:
            logger.info(f"✓ {family_id} completed successfully (exit code: 0)")
        else:
            logger.error(f"✗ {family_id} failed (exit code: {exit_code})")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DUAL RECORDING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Session directory: {session_dir}")
    
    for family_id, exit_code in exit_codes.items():
        status = "✓ PASS" if exit_code == 0 else "✗ FAIL"
        logger.info(f"  {family_id}: {status}")
    
    logger.info("=" * 80)
    
    return exit_codes


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Tape 1 Dual Recording - Record both 5m and 15m families in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Base output directory for sessions",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    base_dir = Path(args.outdir)
    
    # Run dual recording
    exit_codes = run_dual_recording(
        base_dir=base_dir,
        duration_minutes=args.duration_minutes,
        verbose=args.verbose,
    )
    
    # Return non-zero if any family failed
    if any(code != 0 for code in exit_codes.values()):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
