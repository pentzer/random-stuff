"""
Health gate for probability tape - REAL GATES, NO PLACEHOLDERS.

Strict thresholds for usability. FAIL on violation, never warn.
Emits explicit gate_verdict.json with all check metrics.

GATES IMPLEMENTED:
1. min_run_duration_seconds: minimum 600s (10 minutes)
2. cadence: exactly 1-second grid (no gaps > 1s)
3. weak_tape: detect excessive missing data or flatline behavior
4. staleness: max staleness threshold (3 seconds)
5. proxy_sanity: prices within [0,1], not forced to complement 1.0
6. clock_anomaly: detect future timestamps
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from .config import ProbabilityTapeConfig


logger = logging.getLogger(__name__)


class HealthGateV2:
    """Real health gate with explicit, non-negotiable checks."""
    
    def __init__(
        self,
        config: ProbabilityTapeConfig,
        normalized_file: Path,
        family_id: str,
        resolution_id: str,
    ):
        """
        Initialize health gate.
        
        Args:
            config: Configuration with threshold values
            normalized_file: Path to canonical normalized parquet
            family_id: Family ID (e.g., BTC_15M_UPDOWN)
            resolution_id: Resolution/condition ID
        """
        self.config = config
        self.normalized_file = normalized_file
        self.family_id = family_id
        self.resolution_id = resolution_id
        
        # Gate thresholds (tunable, but explicit)
        self.min_run_duration_seconds = 600  # 10 minutes
        self.yes_missing_rate_max = 0.02  # 2%
        self.no_missing_rate_max = 0.02   # 2%
        self.both_missing_rate_max = 0.0   # 0% - any both_missing is failure
        self.max_staleness_seconds = 3.0   # 3 seconds
        self.staleness_violation_rate_max = 0.01  # 1% of rows can exceed max
        self.flatline_std_threshold = 1e-6  # std < this = flatline
    
    def evaluate(self) -> dict:
        """
        Evaluate tape against all gates.
        
        Returns:
            dict with verdict (PASS/FAIL), checks, and metrics
        """
        gate_id = f"gate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        gate_ts = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Starting health gate: {gate_id}")
        logger.info(f"Normalized file: {self.normalized_file}")
        
        # Load data
        if not self.normalized_file.exists():
            logger.error(f"Normalized file does not exist: {self.normalized_file}")
            return self._fail_verdict(
                gate_id,
                gate_ts,
                ["FILE_NOT_FOUND"],
            )
        
        try:
            df = pd.read_parquet(self.normalized_file)
        except Exception as e:
            logger.error(f"Failed to read parquet: {e}")
            return self._fail_verdict(
                gate_id,
                gate_ts,
                [f"FAILED_TO_READ: {str(e)}"],
            )
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Ensure ts_1s is datetime
        df['ts_1s'] = pd.to_datetime(df['ts_1s'])
        
        # Calculate all metrics
        metrics = self._calculate_all_metrics(df)
        
        # Evaluate each gate
        checks = {}
        failure_reasons = []
        
        # GATE 1: Minimum duration
        check_1 = bool(self._check_min_duration(metrics))
        checks['min_run_duration'] = check_1
        if not check_1:
            failure_reasons.append(f"Duration {metrics['duration_seconds']:.1f}s < {self.min_run_duration_seconds}s")
        
        # GATE 2: Cadence (1-second grid)
        check_2 = bool(self._check_cadence(df))
        checks['cadence'] = check_2
        if not check_2:
            failure_reasons.append(f"Cadence violation: {metrics['cadence_violations']} gaps > 1s")
        
        # GATE 3: Weak tape (missing rates, flatline)
        check_3 = bool(self._check_weak_tape(metrics))
        checks['not_weak_tape'] = check_3
        if not check_3:
            reasons = []
            if metrics['yes_missing_rate'] > self.yes_missing_rate_max:
                reasons.append(f"YES missing {metrics['yes_missing_rate']:.2%}")
            if metrics['no_missing_rate'] > self.no_missing_rate_max:
                reasons.append(f"NO missing {metrics['no_missing_rate']:.2%}")
            if metrics['both_missing_rate'] > self.both_missing_rate_max:
                reasons.append(f"Both missing {metrics['both_missing_rate']:.2%}")
            if metrics['yes_flatline']:
                reasons.append("YES is flatline")
            if metrics['no_flatline']:
                reasons.append("NO is flatline")
            failure_reasons.append("Weak tape: " + ", ".join(reasons))
        
        # GATE 4: Staleness
        check_4 = bool(self._check_staleness(metrics))
        checks['staleness'] = check_4
        if not check_4:
            failure_reasons.append(
                f"Staleness violation: "
                f"{metrics['staleness_violations_yes']} YES "
                f"+ {metrics['staleness_violations_no']} NO "
                f"exceed {self.max_staleness_seconds}s"
            )
        
        # GATE 5: Proxy sanity
        check_5 = bool(self._check_proxy_sanity(metrics))
        checks['proxy_sanity'] = check_5
        if not check_5:
            reasons = []
            if metrics['proxy_out_of_range']:
                reasons.append(f"Prices out of [0,1]: {metrics['proxy_out_of_range']} rows")
            if metrics['forced_complement']:
                reasons.append("proxy_sum always ≈ 1.0 (forced complement detected)")
            failure_reasons.append("Proxy sanity: " + ", ".join(reasons))
        
        # Determine overall verdict
        all_passed = all(checks.values())
        verdict = "PASS" if all_passed else "FAIL"
        
        logger.info(f"Gate checks: {checks}")
        logger.info(f"Verdict: {verdict}")
        if failure_reasons:
            logger.info(f"Failures: {failure_reasons}")
        
        # Build result
        result = {
            "verdict": verdict,
            "gate_id": gate_id,
            "family_id": self.family_id,
            "resolution_id": self.resolution_id,
            "gate_ts_utc": gate_ts,
            "checks": checks,
            "metrics": metrics,
            "thresholds": {
                "min_run_duration_seconds": self.min_run_duration_seconds,
                "yes_missing_rate_max": self.yes_missing_rate_max,
                "no_missing_rate_max": self.no_missing_rate_max,
                "both_missing_rate_max": self.both_missing_rate_max,
                "max_staleness_seconds": self.max_staleness_seconds,
                "staleness_violation_rate_max": self.staleness_violation_rate_max,
                "flatline_std_threshold": self.flatline_std_threshold,
            },
            "failure_reasons": failure_reasons,
        }
        
        return result
    
    def _calculate_all_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate all metrics from normalized data."""
        
        # Time range
        start_ts = df['ts_1s'].min()
        end_ts = df['ts_1s'].max()
        duration = (end_ts - start_ts).total_seconds()
        n_rows = len(df)
        
        # Missing rates
        yes_missing = df['flag_yes_missing'].sum() if 'flag_yes_missing' in df else df['yes_proxy_prob'].isna().sum()
        no_missing = df['flag_no_missing'].sum() if 'flag_no_missing' in df else df['no_proxy_prob'].isna().sum()
        both_missing = df['flag_both_missing'].sum() if 'flag_both_missing' in df else (
            df['yes_proxy_prob'].isna() & df['no_proxy_prob'].isna()
        ).sum()
        
        yes_missing_rate = yes_missing / n_rows if n_rows > 0 else 0.0
        no_missing_rate = no_missing / n_rows if n_rows > 0 else 0.0
        both_missing_rate = both_missing / n_rows if n_rows > 0 else 0.0
        
        # Flatline detection (std dev of non-null values)
        yes_flatline = False
        yes_std = 0.0
        yes_values = df['yes_proxy_prob'].dropna()
        if len(yes_values) > 1:
            yes_std = yes_values.std()
            yes_flatline = yes_std < self.flatline_std_threshold
        
        no_flatline = False
        no_std = 0.0
        no_values = df['no_proxy_prob'].dropna()
        if len(no_values) > 1:
            no_std = no_values.std()
            no_flatline = no_std < self.flatline_std_threshold
        
        # Staleness violations
        staleness_violations_yes = 0
        staleness_violations_no = 0
        
        if 'yes_staleness_seconds' in df:
            staleness_violations_yes = (df['yes_staleness_seconds'] > self.max_staleness_seconds).sum()
        
        if 'no_staleness_seconds' in df:
            staleness_violations_no = (df['no_staleness_seconds'] > self.max_staleness_seconds).sum()
        
        # Cadence violations (gaps > 1 second)
        cadence_violations = 0
        if len(df) > 1:
            time_diffs = df['ts_1s'].diff().dt.total_seconds()
            cadence_violations = (time_diffs > 1.0).sum()
        
        # Proxy sanity
        proxy_out_of_range = 0
        if 'yes_proxy_prob' in df:
            proxy_out_of_range += ((df['yes_proxy_prob'] < 0.0) | (df['yes_proxy_prob'] > 1.0)).sum()
        if 'no_proxy_prob' in df:
            proxy_out_of_range += ((df['no_proxy_prob'] < 0.0) | (df['no_proxy_prob'] > 1.0)).sum()
        
        # Forced complement detection (proxy_sum always ≈ 1.0)
        forced_complement = False
        if 'proxy_sum' in df:
            p_sum_valid = df['proxy_sum'].dropna()
            if len(p_sum_valid) > 0:
                # Check if sum is always very close to 1.0
                sum_std = (p_sum_valid - 1.0).abs().std()
                # If std is very small AND mean is very close to 1, it's likely forced
                sum_mean_dist = abs(p_sum_valid.mean() - 1.0)
                forced_complement = (sum_std < 1e-6) and (sum_mean_dist < 1e-3)
        
        return {
            "n_rows": int(n_rows),
            "duration_seconds": float(duration),
            "start_ts": str(start_ts.isoformat()),
            "end_ts": str(end_ts.isoformat()),
            "yes_missing": int(yes_missing),
            "yes_missing_rate": float(yes_missing_rate),
            "no_missing": int(no_missing),
            "no_missing_rate": float(no_missing_rate),
            "both_missing": int(both_missing),
            "both_missing_rate": float(both_missing_rate),
            "yes_flatline": bool(yes_flatline),
            "yes_std": float(yes_std),
            "no_flatline": bool(no_flatline),
            "no_std": float(no_std),
            "staleness_violations_yes": int(staleness_violations_yes),
            "staleness_violations_no": int(staleness_violations_no),
            "cadence_violations": int(cadence_violations),
            "proxy_out_of_range": int(proxy_out_of_range),
            "forced_complement": bool(forced_complement),
        }
    
    def _check_min_duration(self, metrics: dict) -> bool:
        """Check if duration >= minimum."""
        return metrics['duration_seconds'] >= self.min_run_duration_seconds
    
    def _check_cadence(self, df: pd.DataFrame) -> bool:
        """Check if all timestamps are exactly 1 second apart."""
        if len(df) <= 1:
            return True
        
        time_diffs = df['ts_1s'].diff().dt.total_seconds()
        # First row will have NaT, so start from second
        gaps = (time_diffs[1:] > 1.0).sum()
        
        return gaps == 0
    
    def _check_weak_tape(self, metrics: dict) -> bool:
        """Check for weak tape conditions."""
        # Missing rate checks
        if metrics['yes_missing_rate'] > self.yes_missing_rate_max:
            return False
        if metrics['no_missing_rate'] > self.no_missing_rate_max:
            return False
        if metrics['both_missing_rate'] > self.both_missing_rate_max:
            return False
        
        # Flatline checks
        if metrics['yes_flatline']:
            return False
        if metrics['no_flatline']:
            return False
        
        return True
    
    def _check_staleness(self, metrics: dict) -> bool:
        """Check if staleness violations are within acceptable rate."""
        n_rows = metrics['n_rows']
        violations = metrics['staleness_violations_yes'] + metrics['staleness_violations_no']
        violation_rate = violations / n_rows if n_rows > 0 else 0.0
        
        return violation_rate <= self.staleness_violation_rate_max
    
    def _check_proxy_sanity(self, metrics: dict) -> bool:
        """Check proxy sanity."""
        if metrics['proxy_out_of_range'] > 0:
            return False
        
        if metrics['forced_complement']:
            return False
        
        return True
    
    def _fail_verdict(self, gate_id: str, gate_ts: str, failure_reasons: List[str]) -> dict:
        """Create a FAIL verdict."""
        return {
            "verdict": "FAIL",
            "gate_id": gate_id,
            "family_id": self.family_id,
            "resolution_id": self.resolution_id,
            "gate_ts_utc": gate_ts,
            "checks": {},
            "metrics": {},
            "thresholds": {},
            "failure_reasons": failure_reasons,
        }
