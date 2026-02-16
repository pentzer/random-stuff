"""
Configuration for Polymarket Probability Tape.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProbabilityTapeConfig:
    """Configuration for probability tape recording and normalization."""
    
    # Polling configuration
    poll_interval_seconds: float = 1.0
    request_timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_delay_seconds: float = 0.5
    
    # Output directories
    workspace_root: Path = Path(".")
    raw_output_dir: Path = Path("data/polymarket/tape1/raw")
    normalized_output_dir: Path = Path("data/polymarket/tape1/normalized")
    gate_output_dir: Path = Path("data/polymarket/tape1/gate")
    discovery_output_dir: Path = Path("data/polymarket/tape1/discovery")
    views_output_dir: Path = Path("data/polymarket/tape1/views")
    
    # Normalization configuration
    carry_forward_max_seconds: int = 0  # NO carry-forward (NO silent filling) - preserve missingness
    initial_warmup_seconds: int = 5  # allow NaN during warmup
    
    # Value constraints
    min_probability: float = 0.0
    max_probability: float = 1.0
    eps: float = 1e-9  # epsilon for division safety
    
    # Gate thresholds (strict)
    gate_min_run_duration_seconds: int = 300
    gate_min_percent_seconds_with_both_tokens: float = 95.0
    gate_max_staleness_seconds: int = 10
    gate_min_midpoint_success_rate: float = 80.0
    
    def __post_init__(self):
        """Resolve paths relative to workspace root."""
        self.raw_output_dir = self.workspace_root / self.raw_output_dir
        self.normalized_output_dir = self.workspace_root / self.normalized_output_dir
        self.gate_output_dir = self.workspace_root / self.gate_output_dir
        self.discovery_output_dir = self.workspace_root / self.discovery_output_dir
        self.views_output_dir = self.workspace_root / self.views_output_dir


# Default configuration instance
DEFAULT_CONFIG = ProbabilityTapeConfig()
