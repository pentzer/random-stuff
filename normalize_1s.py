"""
Deterministic 1-second grid normalization for probability tape.
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import ProbabilityTapeConfig
from .schemas import NormalizedProbabilityRow, NormalizationRunMetadata


logger = logging.getLogger(__name__)


class ProbabilityTapeNormalizer:
    """Normalizes raw JSONL data to 1-second probability grid."""
    
    def __init__(
        self,
        config: ProbabilityTapeConfig,
        token_up_id: str,
        token_down_id: str,
        family_id: str,
        resolution_id: str,
        expiry_ts_utc: Optional[str] = None,
    ):
        """
        Initialize normalizer.
        
        Args:
            config: Configuration
            token_up_id: UP token ID
            token_down_id: DOWN token ID
            family_id: Family ID
            resolution_id: Resolution ID from discovery
            expiry_ts_utc: Optional expiry timestamp
        """
        self.config = config
        self.token_up_id = token_up_id
        self.token_down_id = token_down_id
        self.family_id = family_id
        self.resolution_id = resolution_id
        self.expiry_ts_utc = expiry_ts_utc
    
    def normalize(self) -> Path:
        """
        Normalize raw data to 1-second grid.
        
        Returns:
            Path to output parquet file (canonical tape, WITHOUT complement normalization)
        """
        normalization_id = f"norm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting normalization: {normalization_id}")
        logger.info(f"Resolution: {self.resolution_id}")
        
        # Read raw data
        up_records = self._read_raw_records(self.token_up_id)
        down_records = self._read_raw_records(self.token_down_id)
        
        logger.info(f"Read {len(up_records)} UP records, {len(down_records)} DOWN records")
        
        if len(up_records) == 0 or len(down_records) == 0:
            raise ValueError("No raw records found for one or both tokens")
        
        # Create 1-second grid
        grid_data = self._create_second_grid(up_records, down_records)
        
        logger.info(f"Created grid with {len(grid_data)} seconds")
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(row) for row in grid_data])
        
        # Convert timestamps
        df['ts_1s'] = pd.to_datetime(df['ts_1s'])
        if self.expiry_ts_utc:
            df['expiry_ts_utc'] = pd.to_datetime(df['expiry_ts_utc'])
        
        # Write to parquet
        self.config.normalized_output_dir.mkdir(parents=True, exist_ok=True)
        
        # CANONICAL TAPE: Remove complement normalization, keep only proxy + raw prices
        # Columns to keep in canonical:
        canonical_columns = [
            'ts_1s',
            'family_id',
            'resolution_id',
            'token_yes_id',
            'token_no_id',
            'yes_mid_price',
            'no_mid_price',
            'yes_proxy_prob',
            'no_proxy_prob',
            'proxy_sum',
            'yes_source',
            'no_source',
            'yes_staleness_seconds',
            'no_staleness_seconds',
            'flag_yes_missing',
            'flag_no_missing',
            'flag_both_missing',
            'expiry_ts_utc',
            'time_to_expiry_seconds',
        ]
        
        # Filter to canonical columns (in case there are extras)
        df_canonical = df[[col for col in canonical_columns if col in df.columns]]
        
        # Write canonical tape
        output_file = self.config.normalized_output_dir / f"probability_tape_{self.family_id}.parquet"
        df_canonical.to_parquet(output_file, index=False)
        logger.info(f"Wrote CANONICAL normalized tape to {output_file}")
        logger.info(f"  Canonical columns: {list(df_canonical.columns)}")
        
        # DIAGNOSTICS TAPE: Write complement normalization (yes_norm_prob, no_norm_prob)
        # as separate file for inspection/analysis only
        df_diagnostics = df[['ts_1s', 'yes_norm_prob', 'no_norm_prob', 'proxy_sum'] + 
                            ['yes_proxy_prob', 'no_proxy_prob']].copy()
        
        # Compute complement error = |yes_norm_prob + no_norm_prob - 1|
        df_diagnostics['complement_error'] = (
            (df_diagnostics['yes_norm_prob'] + df_diagnostics['no_norm_prob'] - 1.0).abs()
        )
        
        diagnostics_file = self.config.normalized_output_dir / f"tape1_diagnostics.parquet"
        df_diagnostics.to_parquet(diagnostics_file, index=False)
        logger.info(f"Wrote DIAGNOSTICS tape (complement normalization) to {diagnostics_file}")
        
        # Write schema to tape1_schema.json (canonical schema only)
        schema_file = self.config.normalized_output_dir / "tape1_schema.json"
        with open(schema_file, "w") as f:
            json.dump({
                "columns": list(df_canonical.columns),
                "dtypes": df_canonical.dtypes.astype(str).to_dict(),
                "description": "Polymarket Tape 1 - 1-second grid with proxy probabilities (canonical, NO forced complementarity)",
                "version": "0002",
                "note": "Complement normalization (yes_norm_prob, no_norm_prob) removed from canonical. See tape1_diagnostics.parquet for optional normalized view.",
            }, f, indent=2)
        logger.info(f"Wrote schema to {schema_file}")
        
        # Write metadata
        metadata = self._create_metadata(normalization_id, len(up_records) + len(down_records), len(df), df_canonical)
        metadata_file = self.config.normalized_output_dir / f"{normalization_id}_metadata.json"
        with open(metadata_file, "w") as f:
            f.write(metadata.to_json())
        
        logger.info(f"Wrote normalization metadata to {metadata_file}")
        
        return output_file
    
    def _read_raw_records(self, token_id: str) -> list[dict]:
        """Read all raw JSONL records for a token."""
        # The recorder writes token_<id>.jsonl files directly in the run directory
        token_file = self.config.raw_output_dir / f"token_{token_id}.jsonl"
        
        if not token_file.exists():
            logger.warning(f"Token file does not exist: {token_file}")
            return []
        
        records = []
        
        with open(token_file) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to parse line in {token_file}: {e}")
        
        return records
    
    def _extract_price(self, record: dict) -> Optional[float]:
        """Extract price from raw payload."""
        if record.get("error") is not None or record.get("http_status") != 200:
            return None
        
        payload = record.get("raw_payload", {})
        
        # Try to extract price (depends on endpoint)
        # Midpoint endpoint can return: float, {"mid": float}, {"price": float}, {"midpoint": float}
        # Price endpoint returns {"price": float}
        
        if isinstance(payload, (int, float)):
            return float(payload)
        
        if isinstance(payload, dict):
            # Try common keys in order of preference
            for key in ["mid", "midpoint", "price", "value"]:
                if key in payload:
                    try:
                        return float(payload[key])
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _create_second_grid(
        self,
        up_records: list[dict],
        down_records: list[dict],
    ) -> list[NormalizedProbabilityRow]:
        """Create 1-second grid from raw records."""
        
        # Build time-indexed dictionaries
        up_by_second = {}
        down_by_second = {}
        
        for record in up_records:
            ts_str = record.get("local_receive_ts_utc")
            if not ts_str:
                continue
            
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                second = ts.replace(microsecond=0)
                
                price = self._extract_price(record)
                endpoint = record.get("endpoint", "unknown")
                
                # Keep latest record per second
                if second not in up_by_second or ts > up_by_second[second]["ts"]:
                    up_by_second[second] = {
                        "ts": ts,
                        "price": price,
                        "endpoint": endpoint,
                    }
            except Exception as e:
                logger.debug(f"Failed to process UP record: {e}")
        
        for record in down_records:
            ts_str = record.get("local_receive_ts_utc")
            if not ts_str:
                continue
            
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                second = ts.replace(microsecond=0)
                
                price = self._extract_price(record)
                endpoint = record.get("endpoint", "unknown")
                
                # Keep latest record per second
                if second not in down_by_second or ts > down_by_second[second]["ts"]:
                    down_by_second[second] = {
                        "ts": ts,
                        "price": price,
                        "endpoint": endpoint,
                    }
            except Exception as e:
                logger.debug(f"Failed to process DOWN record: {e}")
        
        # Determine time range
        all_seconds = set(up_by_second.keys()) | set(down_by_second.keys())
        
        if not all_seconds:
            raise ValueError("No valid timestamps found in records")
        
        min_second = min(all_seconds)
        max_second = max(all_seconds)
        
        logger.info(f"Time range: {min_second} to {max_second}")
        
        # Generate complete second grid
        current = min_second
        grid = []
        
        last_up_price = None
        last_up_endpoint = "none"
        last_up_ts = None
        
        last_down_price = None
        last_down_endpoint = "none"
        last_down_ts = None
        
        expiry_dt = None
        if self.expiry_ts_utc:
            try:
                expiry_dt = datetime.fromisoformat(self.expiry_ts_utc.replace('Z', '+00:00'))
            except:
                pass
        
        while current <= max_second:
            # Get UP value
            up_source = "none"
            up_staleness = None
            
            if current in up_by_second:
                last_up_price = up_by_second[current]["price"]
                last_up_endpoint = up_by_second[current]["endpoint"]
                last_up_ts = current
                up_source = last_up_endpoint
                up_staleness = 0.0
            elif last_up_price is not None and last_up_ts is not None:
                # Carry forward
                staleness = (current - last_up_ts).total_seconds()
                if staleness <= self.config.carry_forward_max_seconds:
                    up_source = "carry"
                    up_staleness = staleness
                else:
                    # Too stale, reset
                    last_up_price = None
                    up_source = "none"
            
            # Get DOWN value
            down_source = "none"
            down_staleness = None
            
            if current in down_by_second:
                last_down_price = down_by_second[current]["price"]
                last_down_endpoint = down_by_second[current]["endpoint"]
                last_down_ts = current
                down_source = last_down_endpoint
                down_staleness = 0.0
            elif last_down_price is not None and last_down_ts is not None:
                # Carry forward
                staleness = (current - last_down_ts).total_seconds()
                if staleness <= self.config.carry_forward_max_seconds:
                    down_source = "carry"
                    down_staleness = staleness
                else:
                    # Too stale, reset
                    last_down_price = None
                    down_source = "none"
            
            # Clamp and compute probabilities
            p_up_proxy = None
            p_down_proxy = None
            p_sum = None
            p_up_norm = None
            p_down_norm = None
            
            if last_up_price is not None:
                p_up_proxy = max(self.config.min_probability, min(self.config.max_probability, last_up_price))
            
            if last_down_price is not None:
                p_down_proxy = max(self.config.min_probability, min(self.config.max_probability, last_down_price))
            
            if p_up_proxy is not None and p_down_proxy is not None:
                p_sum = p_up_proxy + p_down_proxy
                
                # Normalized probabilities (optional)
                if p_sum > self.config.eps:
                    p_up_norm = p_up_proxy / p_sum
                    p_down_norm = p_down_proxy / p_sum
            
            # Time to expiry
            time_to_expiry = None
            if expiry_dt:
                time_to_expiry = (expiry_dt - current).total_seconds()
            
            # Flags for missingness
            flag_yes_missing = last_up_price is None
            flag_no_missing = last_down_price is None
            flag_both_missing = flag_yes_missing and flag_no_missing
            
            row = NormalizedProbabilityRow(
                ts_1s=current.isoformat(),
                family_id=self.family_id,
                resolution_id=self.resolution_id,
                token_yes_id=self.token_up_id,
                token_no_id=self.token_down_id,
                yes_mid_price=last_up_price,
                no_mid_price=last_down_price,
                yes_proxy_prob=p_up_proxy,
                no_proxy_prob=p_down_proxy,
                proxy_sum=p_sum,
                yes_norm_prob=p_up_norm,
                no_norm_prob=p_down_norm,
                yes_source=up_source,
                no_source=down_source,
                yes_staleness_seconds=up_staleness,
                no_staleness_seconds=down_staleness,
                flag_yes_missing=flag_yes_missing,
                flag_no_missing=flag_no_missing,
                flag_both_missing=flag_both_missing,
                expiry_ts_utc=self.expiry_ts_utc,
                time_to_expiry_seconds=time_to_expiry,
            )
            
            grid.append(row)
            
            current += timedelta(seconds=1)
        
        return grid
    
    def _create_metadata(
        self,
        normalization_id: str,
        raw_count: int,
        normalized_count: int,
        df: pd.DataFrame,
    ) -> NormalizationRunMetadata:
        """Create normalization metadata."""
        
        # Count seconds with different coverage
        both = df[(df['yes_proxy_prob'].notna()) & (df['no_proxy_prob'].notna())]
        up_only = df[(df['yes_proxy_prob'].notna()) & (df['no_proxy_prob'].isna())]
        down_only = df[(df['yes_proxy_prob'].isna()) & (df['no_proxy_prob'].notna())]
        neither = df[(df['yes_proxy_prob'].isna()) & (df['no_proxy_prob'].isna())]
        
        # Max staleness
        max_up_staleness = df['yes_staleness_seconds'].max() if df['yes_staleness_seconds'].notna().any() else 0.0
        max_down_staleness = df['no_staleness_seconds'].max() if df['no_staleness_seconds'].notna().any() else 0.0
        
        start_ts = df['ts_1s'].min()
        end_ts = df['ts_1s'].max()
        duration = (end_ts - start_ts).total_seconds()
        
        # Convert config to JSON-serializable format
        config_dict = {}
        for key, value in vars(self.config).items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        return NormalizationRunMetadata(
            normalization_id=normalization_id,
            input_resolution_id=self.resolution_id,
            family_id=self.family_id,
            start_ts_utc=start_ts.isoformat(),
            end_ts_utc=end_ts.isoformat(),
            duration_seconds=duration,
            raw_records_processed=raw_count,
            normalized_records_output=normalized_count,
            seconds_with_both_tokens=len(both),
            seconds_with_up_only=len(up_only),
            seconds_with_down_only=len(down_only),
            seconds_with_neither=len(neither),
            max_up_staleness_seconds=max_up_staleness,
            max_down_staleness_seconds=max_down_staleness,
            config=config_dict,
        )
