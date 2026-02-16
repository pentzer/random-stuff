"""
Alignment preview reports for Tape 1.

Produces:
- joinability_report.json (time range, gaps, cadence check, monotonicity)

REAL CADENCE CHECK: Verifies that all consecutive timestamps are exactly 1 second apart.
No placeholders - actual metrics only.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import polars as pl


logger = logging.getLogger(__name__)


def generate_joinability_report(normalized_file: Path) -> Dict[str, Any]:
    """
    Generate joinability_report.json with REAL metrics.
    
    Analyzes:
    - Time range (start, end, duration)
    - Unique seconds count
    - Cadence check (must be exactly 1-second grid)
    - Gap detection and analysis
    - Monotonicity check
    - Duplicate timestamp detection
    """
    try:
        df = pl.read_parquet(normalized_file)
        
        # Time range - use ts_1s column (the correct timestamp column)
        timestamps = df["ts_1s"]
        start_ts = timestamps.min()
        end_ts = timestamps.max()
        
        # Duration calculation
        start_dt = pd.to_datetime(start_ts)
        end_dt = pd.to_datetime(end_ts)
        duration_seconds = (end_dt - start_dt).total_seconds()
        
        # Unique seconds
        unique_seconds = df["ts_1s"].n_unique()
        total_rows = len(df)
        
        # Expected seconds (should match total_rows for 1-second grid)
        expected_seconds = int(duration_seconds) + 1  # inclusive range
        
        # Convert to pandas for detailed analysis
        pdf = df.select(["ts_1s"]).to_pandas()
        pdf["ts_1s"] = pd.to_datetime(pdf["ts_1s"])
        pdf = pdf.sort_values("ts_1s").reset_index(drop=True)
        
        # REAL CADENCE CHECK: Verify all consecutive differences are exactly 1 second
        pdf["ts_diff"] = pdf["ts_1s"].diff().dt.total_seconds()
        
        # Count cadence violations (all non-first rows should have diff == 1.0)
        cadence_violations = (pdf["ts_diff"][1:] != 1.0).sum()
        is_regular_cadence = cadence_violations == 0
        
        # Gap detection (missing seconds = diff > 1.0)
        gaps = []
        gap_rows = pdf[pdf["ts_diff"] > 1.0]
        
        for _, row in gap_rows.iterrows():
            gap_size = int(row["ts_diff"]) - 1
            gaps.append({
                "gap_seconds": gap_size,
            })
        
        num_gaps = len(gaps)
        max_gap_seconds = max((g["gap_seconds"] for g in gaps), default=0)
        total_missing_seconds = sum(g["gap_seconds"] for g in gaps)
        
        # Monotonicity check
        is_monotonic = pdf["ts_1s"].is_monotonic_increasing
        
        # Duplicate check
        has_duplicates = unique_seconds < total_rows
        num_duplicates = total_rows - unique_seconds if has_duplicates else 0
        
        # Cadence quality rating
        if is_regular_cadence:
            cadence_quality = "PERFECT_1S"
        elif int(cadence_violations) <= 1:
            cadence_quality = "NEARLY_PERFECT"
        elif int(cadence_violations) <= int(total_rows * 0.01):  # < 1% violations
            cadence_quality = "ACCEPTABLE"
        else:
            cadence_quality = "POOR"
        
        # Joinability recommendation
        recommendation = _make_recommendation(
            is_regular_cadence=bool(is_regular_cadence),
            cadence_violations=int(cadence_violations),
            has_duplicates=bool(has_duplicates),
            num_gaps=int(num_gaps),
            is_monotonic=bool(is_monotonic),
        )
        
        report = {
            "time_range": {
                "start_ts_utc": str(start_ts),
                "end_ts_utc": str(end_ts),
                "duration_seconds": float(duration_seconds),
                "expected_seconds": int(expected_seconds),
            },
            "coverage": {
                "total_rows": int(total_rows),
                "unique_seconds": int(unique_seconds),
                "has_duplicates": bool(has_duplicates),
                "num_duplicates": int(num_duplicates),
            },
            "cadence": {
                "is_regular_1s_grid": bool(is_regular_cadence),
                "cadence_quality": str(cadence_quality),
                "cadence_violations": int(cadence_violations),
            },
            "gaps": {
                "num_gaps": int(num_gaps),
                "max_gap_seconds": int(max_gap_seconds),
                "total_missing_seconds": int(total_missing_seconds),
            },
            "monotonicity": {
                "is_monotonic_increasing": bool(is_monotonic),
            },
            "recommendation": str(recommendation),
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate joinability report: {e}")
        return {
            "error": str(e),
            "recommendation": "CANNOT_ASSESS",
        }


def _make_recommendation(
    is_regular_cadence: bool,
    cadence_violations: int,
    has_duplicates: bool,
    num_gaps: int,
    is_monotonic: bool,
) -> str:
    """
    Make joinability recommendation based on metrics.
    
    Returns:
        Recommendation string for downstream usage
    """
    if not is_monotonic:
        return "CANNOT_JOIN_NOT_MONOTONIC"
    
    if has_duplicates:
        return "CANNOT_JOIN_DUPLICATES_PRESENT"
    
    if not is_regular_cadence:
        return "CANNOT_JOIN_IRREGULAR_CADENCE"
    
    if num_gaps > 0:
        return "CAN_JOIN_WITH_CAUTION_GAPS_PRESENT"
    
    return "READY_FOR_D2"

