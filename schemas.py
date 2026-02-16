"""
Strict schemas for Polymarket Probability Tape.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Literal
import json


@dataclass
class RawPriceRecord:
    """Single raw price poll from REST endpoint."""
    
    local_receive_ts_utc: str  # ISO format
    endpoint: Literal["midpoint", "price", "last"]
    token_id: str
    http_status: int
    response_ms: float
    raw_payload: dict  # verbatim JSON response
    error: Optional[str] = None
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(asdict(self))
    
    def extract_value(self) -> Optional[float]:
        """Extract numeric value from raw_payload."""
        if self.error is not None or self.http_status != 200:
            return None
        
        # Try common response formats
        payload = self.raw_payload
        
        # Direct value key
        if isinstance(payload, dict):
            for key in ['midpoint', 'price', 'last', 'value']:
                if key in payload:
                    try:
                        return float(payload[key])
                    except (ValueError, TypeError):
                        continue
        
        # Direct numeric payload
        try:
            return float(payload)
        except (ValueError, TypeError):
            pass
        
        return None


@dataclass
class NormalizedProbabilityRow:
    """Single row of normalized probability tape (1-second resolution)."""
    
    ts_1s: str  # ISO format, second-level (required name: ts_1s)
    family_id: str
    resolution_id: str
    
    # Token IDs
    token_yes_id: str
    token_no_id: str
    
    # Raw midpoint/price values (as observed from API)
    yes_mid_price: Optional[float]  # Raw YES token price
    no_mid_price: Optional[float]   # Raw NO token price
    
    # DERIVED proxy probabilities (explicitly labeled)
    # These are DERIVED from observed prices, NOT direct measurements
    yes_proxy_prob: Optional[float]  # Derived from yes_mid_price
    no_proxy_prob: Optional[float]   # Derived from no_mid_price
    
    # Sum (diagnostic, NOT enforced to equal 1, NO forced complementarity)
    proxy_sum: Optional[float]
    
    # Normalized probabilities (optional, only if user wants complementarity)
    yes_norm_prob: Optional[float] = None
    no_norm_prob: Optional[float] = None
    
    # Metadata
    yes_source: Literal["midpoint", "price", "carry", "none"] = "none"
    no_source: Literal["midpoint", "price", "carry", "none"] = "none"
    yes_staleness_seconds: Optional[float] = None
    no_staleness_seconds: Optional[float] = None
    
    # Flags (missingness indicators)
    flag_yes_missing: bool = False
    flag_no_missing: bool = False
    flag_both_missing: bool = False
    
    # Contract metadata
    expiry_ts_utc: Optional[str] = None
    time_to_expiry_seconds: Optional[float] = None


@dataclass
class RecorderRunMetadata:
    """Metadata for a probability tape recording run."""
    
    run_id: str
    family_id: str
    resolution_id: str
    start_ts_utc: str
    end_ts_utc: str
    duration_seconds: float
    total_polls: int
    successful_polls: int
    failed_polls: int
    token_up_id: str
    token_down_id: str
    endpoints_used: list[str]
    config: dict
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)


@dataclass
class NormalizationRunMetadata:
    """Metadata for normalization run."""
    
    normalization_id: str
    input_resolution_id: str
    family_id: str
    start_ts_utc: str
    end_ts_utc: str
    duration_seconds: float
    raw_records_processed: int
    normalized_records_output: int
    seconds_with_both_tokens: int
    seconds_with_up_only: int
    seconds_with_down_only: int
    seconds_with_neither: int
    max_up_staleness_seconds: float
    max_down_staleness_seconds: float
    config: dict
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)


@dataclass
class GateVerdict:
    """Health gate verdict for probability tape."""
    
    verdict: Literal["PASS", "FAIL"]
    gate_id: str
    family_id: str
    resolution_id: str
    normalization_id: str
    gate_ts_utc: str
    
    # Metrics
    run_duration_seconds: float
    total_seconds: int
    seconds_with_both_tokens: int
    percent_seconds_with_both_tokens: float
    max_up_staleness_seconds: float
    max_down_staleness_seconds: float
    midpoint_success_rate_up: float
    midpoint_success_rate_down: float
    
    # Threshold checks
    checks: dict[str, bool]
    failure_reasons: list[str]
    
    # Thresholds used
    thresholds: dict
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        # Convert booleans to strings if needed for JSON compatibility
        # (JSON serializer handles bool fine in Python 3.10+, but be explicit)
        return json.dumps(data, indent=2, default=str)


# Parquet schema for normalized tape
NORMALIZED_TAPE_SCHEMA = {
    "ts_utc": "datetime64[ns]",
    "family_id": "string",
    "resolution_id": "string",
    "token_up_id": "string",
    "token_down_id": "string",
    "p_up_raw": "float64",
    "p_down_raw": "float64",
    "p_up_proxy": "float64",
    "p_down_proxy": "float64",
    "p_sum": "float64",
    "p_up_norm": "float64",
    "p_down_norm": "float64",
    "up_source": "string",
    "down_source": "string",
    "up_staleness_seconds": "float64",
    "down_staleness_seconds": "float64",
    "expiry_ts_utc": "datetime64[ns]",
    "time_to_expiry_seconds": "float64",
}
