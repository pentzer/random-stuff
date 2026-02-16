"""
Health report generators for Tape 1.

Produces:
- run_metadata.json
- missingness_report.json
- staleness_report.json
- flatline_report.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import polars as pl


logger = logging.getLogger(__name__)


def generate_run_metadata(
    snapshot: dict,
    recording_metadata: dict,
    normalized_file: Path,
) -> Dict[str, Any]:
    """
    Generate run_metadata.json.
    
    Includes:
    - Duration requested vs actual
    - Poll counts
    - Exit reason
    """
    # Load normalized data to get final stats
    try:
        df = pd.read_parquet(normalized_file)
        normalized_seconds = len(df)
    except Exception as e:
        logger.warning(f"Could not read normalized file: {e}")
        normalized_seconds = 0
    
    metadata = {
        "pipeline": "tape1_polymarket_btc_15m_updown",
        "version": "0001",
        "discovery": {
            "slug": snapshot.get("slug"),
            "condition_id": snapshot.get("condition_id"),
            "token_yes_id": snapshot.get("token_yes_id"),
            "token_no_id": snapshot.get("token_no_id"),
            "resolved_at_utc": snapshot.get("resolved_at_utc"),
        },
        "recording": {
            "start_ts_utc": recording_metadata.get("start_ts_utc"),
            "end_ts_utc": recording_metadata.get("end_ts_utc"),
            "duration_requested_seconds": recording_metadata.get("duration_requested_seconds"),
            "duration_actual_seconds": recording_metadata.get("duration_actual_seconds"),
            "total_polls": recording_metadata.get("total_polls"),
            "successful_polls": recording_metadata.get("successful_polls"),
            "failed_polls": recording_metadata.get("failed_polls"),
            "success_rate_percent": recording_metadata.get("success_rate_percent"),
            "exit_reason": recording_metadata.get("exit_reason", "completed_successfully"),
        },
        "normalization": {
            "normalized_seconds": normalized_seconds,
        },
    }
    
    return metadata


def generate_missingness_report(normalized_file: Path) -> Dict[str, Any]:
    """
    Generate missingness_report.json.
    
    Analyzes:
    - Seconds with both tokens
    - Seconds with only YES
    - Seconds with only NO
    - Seconds with neither
    - Gap analysis
    """
    try:
        # Use polars for efficient processing
        df = pl.read_parquet(normalized_file)
        
        total_seconds = len(df)
        
        # Count coverage patterns
        both = df.filter(
            (pl.col("yes_proxy_prob").is_not_null()) & (pl.col("no_proxy_prob").is_not_null())
        )
        yes_only = df.filter(
            (pl.col("yes_proxy_prob").is_not_null()) & (pl.col("no_proxy_prob").is_null())
        )
        no_only = df.filter(
            (pl.col("yes_proxy_prob").is_null()) & (pl.col("no_proxy_prob").is_not_null())
        )
        neither = df.filter(
            (pl.col("yes_proxy_prob").is_null()) & (pl.col("no_proxy_prob").is_null())
        )
        
        seconds_with_both = len(both)
        seconds_with_yes_only = len(yes_only)
        seconds_with_no_only = len(no_only)
        seconds_with_neither = len(neither)
        
        # Percentages
        percent_both = (seconds_with_both / total_seconds * 100) if total_seconds > 0 else 0
        percent_yes_only = (seconds_with_yes_only / total_seconds * 100) if total_seconds > 0 else 0
        percent_no_only = (seconds_with_no_only / total_seconds * 100) if total_seconds > 0 else 0
        percent_neither = (seconds_with_neither / total_seconds * 100) if total_seconds > 0 else 0
        
        # Gap analysis (consecutive seconds with missing data)
        missing_mask = (df["yes_proxy_prob"].is_null()) | (df["no_proxy_prob"].is_null())
        gaps = []
        gap_start = None
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and gap_start is None:
                gap_start = i
            elif not is_missing and gap_start is not None:
                gap_length = i - gap_start
                gaps.append(gap_length)
                gap_start = None
        
        # If last row is missing, close the gap
        if gap_start is not None:
            gaps.append(len(df) - gap_start)
        
        max_gap_seconds = max(gaps) if gaps else 0
        num_gaps = len(gaps)
        
        report = {
            "total_seconds": total_seconds,
            "seconds_with_both_tokens": seconds_with_both,
            "seconds_with_yes_only": seconds_with_yes_only,
            "seconds_with_no_only": seconds_with_no_only,
            "seconds_with_neither": seconds_with_neither,
            "percent_both_tokens": round(percent_both, 2),
            "percent_yes_only": round(percent_yes_only, 2),
            "percent_no_only": round(percent_no_only, 2),
            "percent_neither": round(percent_neither, 2),
            "gap_analysis": {
                "num_gaps": num_gaps,
                "max_gap_seconds": max_gap_seconds,
                "total_gap_seconds": sum(gaps),
            },
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate missingness report: {e}")
        return {"error": str(e)}


def generate_staleness_report(normalized_file: Path) -> Dict[str, Any]:
    """
    Generate staleness_report.json.
    
    Analyzes:
    - Max staleness for YES and NO tokens
    - Distribution of staleness values
    - Carry-forward frequency
    """
    try:
        df = pl.read_parquet(normalized_file)
        
        # Staleness stats for YES (up)
        up_staleness = df.filter(pl.col("yes_staleness_seconds").is_not_null())["yes_staleness_seconds"]
        max_yes_staleness = up_staleness.max() if len(up_staleness) > 0 else 0
        mean_yes_staleness = up_staleness.mean() if len(up_staleness) > 0 else 0
        median_yes_staleness = up_staleness.median() if len(up_staleness) > 0 else 0
        
        # Staleness stats for NO (down)
        down_staleness = df.filter(pl.col("no_staleness_seconds").is_not_null())["no_staleness_seconds"]
        max_no_staleness = down_staleness.max() if len(down_staleness) > 0 else 0
        mean_no_staleness = down_staleness.mean() if len(down_staleness) > 0 else 0
        median_no_staleness = down_staleness.median() if len(down_staleness) > 0 else 0
        
        # Count carry-forward usage
        carry_yes = df.filter(pl.col("yes_source") == "carry")
        carry_no = df.filter(pl.col("no_source") == "carry")
        
        report = {
            "yes_token": {
                "max_staleness_seconds": float(max_yes_staleness) if max_yes_staleness else 0,
                "mean_staleness_seconds": float(mean_yes_staleness) if mean_yes_staleness else 0,
                "median_staleness_seconds": float(median_yes_staleness) if median_yes_staleness else 0,
                "carry_forward_count": len(carry_yes),
            },
            "no_token": {
                "max_staleness_seconds": float(max_no_staleness) if max_no_staleness else 0,
                "mean_staleness_seconds": float(mean_no_staleness) if mean_no_staleness else 0,
                "median_staleness_seconds": float(median_no_staleness) if median_no_staleness else 0,
                "carry_forward_count": len(carry_no),
            },
            "max_yes_staleness_seconds": float(max_yes_staleness) if max_yes_staleness else 0,
            "max_no_staleness_seconds": float(max_no_staleness) if max_no_staleness else 0,
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate staleness report: {e}")
        return {"error": str(e)}


def generate_flatline_report(normalized_file: Path) -> Dict[str, Any]:
    """
    Generate flatline_report.json.
    
    Detects:
    - Consecutive seconds with identical values (potential stale data)
    - Longest flatline sequences
    """
    try:
        df = pl.read_parquet(normalized_file)
        
        # Detect flatlines in YES token
        yes_flatlines = _detect_flatlines(df["yes_proxy_prob"].to_list())
        
        # Detect flatlines in NO token
        no_flatlines = _detect_flatlines(df["no_proxy_prob"].to_list())
        
        report = {
            "yes_token": {
                "max_flatline_seconds": yes_flatlines["max_length"],
                "num_flatlines_gt_5s": yes_flatlines["count_gt_5"],
                "num_flatlines_gt_10s": yes_flatlines["count_gt_10"],
                "total_flatline_seconds": yes_flatlines["total_length"],
            },
            "no_token": {
                "max_flatline_seconds": no_flatlines["max_length"],
                "num_flatlines_gt_5s": no_flatlines["count_gt_5"],
                "num_flatlines_gt_10s": no_flatlines["count_gt_10"],
                "total_flatline_seconds": no_flatlines["total_length"],
            },
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate flatline report: {e}")
        return {"error": str(e)}


def _detect_flatlines(values: list) -> Dict[str, int]:
    """Detect consecutive identical values (flatlines)."""
    if not values:
        return {"max_length": 0, "count_gt_5": 0, "count_gt_10": 0, "total_length": 0}
    
    flatlines = []
    current_value = None
    current_length = 0
    
    for value in values:
        if value is None:
            # Reset on None
            if current_length >= 2:
                flatlines.append(current_length)
            current_value = None
            current_length = 0
        elif value == current_value:
            current_length += 1
        else:
            if current_length >= 2:
                flatlines.append(current_length)
            current_value = value
            current_length = 1
    
    # Close last flatline
    if current_length >= 2:
        flatlines.append(current_length)
    
    max_length = max(flatlines) if flatlines else 0
    count_gt_5 = sum(1 for f in flatlines if f > 5)
    count_gt_10 = sum(1 for f in flatlines if f > 10)
    total_length = sum(flatlines)
    
    return {
        "max_length": max_length,
        "count_gt_5": count_gt_5,
        "count_gt_10": count_gt_10,
        "total_length": total_length,
    }
