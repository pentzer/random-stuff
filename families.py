"""
Family registry for Polymarket Tape 1 recording.

Defines family specifications (window size, slug format, thresholds) as a single
source of truth used by discovery, orchestration, and downstream processing.

A "family" represents a distinct recording contract:
- Distinct market discovery window (e.g., 5m vs 15m)
- Separate artifacts per family (no mixing/blending)
- Family-scoped thresholds and gate expectations
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FamilySpec:
    """Specification for a Tape 1 recording family."""
    
    # Identity
    family_id: str
    """Unique family identifier (e.g., BTC_5M_UPDOWN, BTC_15M_UPDOWN)"""
    
    # Market discovery
    window_seconds: int
    """Window duration in seconds (300 for 5m, 900 for 15m)"""
    
    slug_prefix: str
    """Slug prefix for candidate market discovery (e.g., btc-updown-5m-)"""
    
    # Recording contract
    record_duration_seconds: int
    """Recommended recording duration (e.g., 260s for 5m, 860s for 15m)"""
    
    # Discovery behavior
    max_windows_ahead_default: int = 8
    """Default number of windows ahead to probe during discovery"""
    
    # Gate thresholds (per-family overrides)
    gate_min_run_duration_seconds: int = 300
    """Minimum successful recording duration to pass gate"""
    
    gate_min_percent_seconds_with_both_tokens: float = 95.0
    """Minimum coverage percentage (both YES and NO tokens must be present)"""
    
    gate_max_staleness_seconds: int = 10
    """Maximum allowed staleness for any token in normalized tape"""
    
    gate_min_midpoint_success_rate: float = 80.0
    """Minimum success rate for midpoint polling"""
    
    # Metadata
    description: str = ""
    """Human-readable description"""


# ============================================================================
# FAMILY REGISTRY (single source of truth)
# ============================================================================

BTC_5M_UPDOWN = FamilySpec(
    family_id="BTC_5M_UPDOWN",
    window_seconds=300,
    slug_prefix="btc-updown-5m-",
    max_windows_ahead_default=8,
    record_duration_seconds=260,  # Full window minus ~40s warmup
    gate_min_run_duration_seconds=240,  # 5m window - stricter than 15m
    gate_min_percent_seconds_with_both_tokens=95.0,
    gate_max_staleness_seconds=10,
    gate_min_midpoint_success_rate=80.0,
    description="BTC price Up/Down rolling market (5-minute window)",
)

BTC_15M_UPDOWN = FamilySpec(
    family_id="BTC_15M_UPDOWN",
    window_seconds=900,
    slug_prefix="btc-updown-15m-",
    max_windows_ahead_default=8,
    record_duration_seconds=260,  # Conservative: ~5 min sample only (existing default)
                                    # TODO: Decide if we want "true 15m" (~860s) instead
    gate_min_run_duration_seconds=300,  # Existing gate threshold
    gate_min_percent_seconds_with_both_tokens=95.0,
    gate_max_staleness_seconds=10,
    gate_min_midpoint_success_rate=80.0,
    description="BTC price Up/Down rolling market (15-minute window)",
)

# Registry mapping
FAMILY_REGISTRY = {
    "BTC_5M_UPDOWN": BTC_5M_UPDOWN,
    "BTC_15M_UPDOWN": BTC_15M_UPDOWN,
}


# ============================================================================
# PUBLIC API
# ============================================================================

def get_family_spec(family_id: str) -> Optional[FamilySpec]:
    """
    Retrieve family specification by ID.
    
    Args:
        family_id: Family identifier (e.g., "BTC_5M_UPDOWN")
    
    Returns:
        FamilySpec if family is known, None otherwise
    """
    return FAMILY_REGISTRY.get(family_id)


def list_families() -> list[str]:
    """
    List all known family IDs.
    
    Returns:
        List of family identifiers in registry
    """
    return sorted(FAMILY_REGISTRY.keys())


def validate_family(family_id: str) -> bool:
    """
    Check if family is registered.
    
    Args:
        family_id: Family identifier to validate
    
    Returns:
        True if family is in registry
    """
    return family_id in FAMILY_REGISTRY


# ============================================================================
# INVARIANTS (enforce auditability)
# ============================================================================

"""
Contract: Each family produces an independent artifact set:

discovery/
  - discovery_snapshot.json (what market we found)
  - discovery_evidence.json (proof of discovery)
  - next_data_raw.json (raw __NEXT_DATA__ slice)
  - token_mapping.json (semantic binding YES/NO to tokens)

tape1_raw/
  - run_*/probability_tape_raw.jsonl (append-only raw polls)
  - run_*/metadata.json (recording run metadata)

tape1_1s/
  - tape1_1s.parquet (canonical 1s grid normalization)

health/
  - run_metadata.json
  - missingness_report.json
  - staleness_report.json
  - flatline_report.json
  - gate_verdict.json

alignment_preview/
  - joinability_report.json

NO SHARED FILES between families.
Only a shared parent "session" directory (session_YYYYMMDD_HHMMSS).

When running both families:
  session_YYYYMMDD_HHMMSS/
    family=BTC_5M_UPDOWN/
      discovery/...
      tape1_raw/...
      tape1_1s/...
      health/...
      alignment_preview/...
    family=BTC_15M_UPDOWN/
      discovery/...
      tape1_raw/...
      tape1_1s/...
      health/...
      alignment_preview/...

This prevents collisions and enforces auditability.
"""
