"""
Append-only raw recorder for Tape 1 (implied probability proxy).

Records per-second REST polling of /midpoint (preferred) and /price (fallback) for two tokens.
Outputs JSONL with full provenance.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .config import ProbabilityTapeConfig
from .clob_rest import PolymarketRESTClient
from .schemas import RawPriceRecord


logger = logging.getLogger(__name__)


class Tape1Recorder:
    """Raw recorder for Tape 1 implied probability proxy."""

    def __init__(
        self,
        config: ProbabilityTapeConfig,
        token_yes_id: str,
        token_no_id: str,
        slug: str,
        condition_id: str,
        family_id: str = "BTC_15M_UPDOWN",
        expiry_ts_utc: Optional[str] = None,
    ):
        """
        Initialize recorder.
        
        Args:
            config: Configuration
            token_yes_id: Yes/Up token ID
            token_no_id: No/Down token ID
            slug: Market slug
            condition_id: Condition ID from discovery
            family_id: Family ID
            expiry_ts_utc: Contract expiry (ISO format, optional)
        """
        self.config = config
        self.token_yes_id = token_yes_id
        self.token_no_id = token_no_id
        self.slug = slug
        self.condition_id = condition_id
        self.family_id = family_id
        self.expiry_ts_utc = expiry_ts_utc

        self.client = PolymarketRESTClient(config)
        
        # Stats
        self.total_polls = 0
        self.successful_polls = 0
        self.failed_polls = 0
        self.records_by_token = {token_yes_id: 0, token_no_id: 0}
        
        # 404 storm detection
        self.consecutive_404 = {token_yes_id: 0, token_no_id: 0}
        
    def record(self, duration_seconds: float) -> dict:
        """
        Record for specified duration.
        
        Args:
            duration_seconds: Recording duration in seconds
        
        Returns:
            Run metadata dict
        
        Raises:
            RuntimeError: On critical failures (404 storm, no records, etc.)
        """
        # Generate run ID
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        
        # Create run directory
        run_dir = self.config.raw_output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        token_yes_file = run_dir / f"token_{self.token_yes_id}.jsonl"
        token_no_file = run_dir / f"token_{self.token_no_id}.jsonl"
        events_file = run_dir / "recorder_events.jsonl"
        metadata_file = run_dir / "run_metadata.json"
        
        token_files = {
            self.token_yes_id: token_yes_file,
            self.token_no_id: token_no_file,
        }
        
        # Parse expiry if available
        expiry_dt = None
        if self.expiry_ts_utc:
            try:
                expiry_dt = datetime.fromisoformat(
                    self.expiry_ts_utc.replace('Z', '+00:00')
                )
            except Exception as e:
                logger.warning(f"Failed to parse expiry: {e}")
        
        # Log start
        start_ts = datetime.now(timezone.utc)
        logger.info("=" * 80)
        logger.info(f"Starting Tape 1 recorder: {run_id}")
        logger.info(f"  Family: {self.family_id}")
        logger.info(f"  Slug: {self.slug}")
        logger.info(f"  Condition ID: {self.condition_id}")
        logger.info(f"  Token YES: {self.token_yes_id}")
        logger.info(f"  Token NO: {self.token_no_id}")
        logger.info(f"  Duration: {duration_seconds}s")
        logger.info(f"  Poll interval: {self.config.poll_interval_seconds}s")
        if expiry_dt:
            logger.info(f"  Expiry: {self.expiry_ts_utc}")
        logger.info("=" * 80)
        
        # Write start event
        self._write_event(
            events_file,
            {
                "event": "recorder_start",
                "ts_utc": start_ts.isoformat(),
                "run_id": run_id,
                "family_id": self.family_id,
                "slug": self.slug,
                "condition_id": self.condition_id,
                "token_yes_id": self.token_yes_id,
                "token_no_id": self.token_no_id,
                "duration_seconds": duration_seconds,
            }
        )
        
        # Recording loop
        start_time = time.monotonic()
        stop_time = start_time + duration_seconds
        poll_interval = self.config.poll_interval_seconds
        run_start_ts_utc = datetime.now(timezone.utc)
        record_seq = 0  # Monotonic counter
        
        # Schedule next poll for each token (stagger slightly to avoid bursts)
        next_poll = {
            self.token_yes_id: start_time,
            self.token_no_id: start_time + 0.01,  # 10ms stagger
        }
        
        # Clock anomaly tracking
        clock_anomaly_detected = False
        clock_anomaly_threshold_seconds = 120  # Future-time threshold
        
        try:
            while time.monotonic() < stop_time:
                # Check expiry
                if expiry_dt and datetime.now(timezone.utc) >= expiry_dt:
                    logger.info("Contract expiry reached, stopping recorder")
                    self._write_event(
                        events_file,
                        {
                            "event": "expiry_reached",
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    break
                
                now = time.monotonic()
                poll_scheduled_ts_utc = datetime.now(timezone.utc)
                
                # Poll each token when due
                for token_id in [self.token_yes_id, self.token_no_id]:
                    if now >= next_poll[token_id]:
                        # Poll with fallback
                        raw_record = self.client.poll_midpoint(
                            token_id,
                            fallback_to_price=True,
                        )
                        
                        # AUGMENT: Add timestamp provenance + sequence info to raw record
                        # Convert raw_record to dict, add fields, convert back
                        record_dict = {
                            "local_receive_ts_utc": raw_record.local_receive_ts_utc,
                            "endpoint": raw_record.endpoint,
                            "token_id": raw_record.token_id,
                            "http_status": raw_record.http_status,
                            "response_ms": raw_record.response_ms,
                            "raw_payload": raw_record.raw_payload,
                            "error": raw_record.error,
                            # NEW PROVENANCE FIELDS
                            "run_start_ts_utc": run_start_ts_utc.isoformat(),
                            "record_seq": record_seq,
                            "poll_scheduled_ts_utc": poll_scheduled_ts_utc.isoformat(),
                        }
                        
                        # Write to token's JSONL
                        self._write_record_dict(token_files[token_id], record_dict)
                        
                        # CHECK: Future-time anomaly (hard gate)
                        try:
                            record_ts = datetime.fromisoformat(
                                raw_record.local_receive_ts_utc.replace('Z', '+00:00')
                            )
                            now_utc = datetime.now(timezone.utc)
                            delta = (record_ts - now_utc).total_seconds()
                            if delta > clock_anomaly_threshold_seconds:
                                clock_anomaly_detected = True
                                logger.error(
                                    f"CLOCK ANOMALY: recorded timestamp {delta:.1f}s in future! "
                                    f"record_ts={record_ts.isoformat()}, now_utc={now_utc.isoformat()}"
                                )
                                self._write_event(
                                    events_file,
                                    {
                                        "event": "clock_anomaly_detected",
                                        "ts_utc": now_utc.isoformat(),
                                        "record_ts": record_ts.isoformat(),
                                        "delta_seconds": delta,
                                        "token_id": token_id,
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Failed to check clock anomaly: {e}")
                        
                        record_seq += 1
                        
                        # Update stats
                        self.total_polls += 1
                        self.records_by_token[token_id] += 1
                        
                        if raw_record.error is None and raw_record.http_status == 200:
                            self.successful_polls += 1
                            self.consecutive_404[token_id] = 0
                        else:
                            self.failed_polls += 1
                            
                            # Track 404s
                            if raw_record.http_status == 404:
                                self.consecutive_404[token_id] += 1
                        
                        # Schedule next poll (avoid drift)
                        next_poll[token_id] = max(
                            next_poll[token_id] + poll_interval,
                            now + poll_interval
                        )
                
                # Check for 404 storm (both tokens failing repeatedly)
                if self._both_tokens_404(threshold=5):
                    error_msg = (
                        f"404 storm detected: both tokens returned 404 "
                        f"at least 5 times consecutively"
                    )
                    logger.error(error_msg)
                    self._write_event(
                        events_file,
                        {
                            "event": "404_storm",
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "consecutive_404_yes": self.consecutive_404[self.token_yes_id],
                            "consecutive_404_no": self.consecutive_404[self.token_no_id],
                        }
                    )
                    raise RuntimeError(error_msg)
                
                # Sleep briefly to avoid busy loop
                time.sleep(0.01)
        
        finally:
            # Close client
            self.client.close()
        
        # End time
        end_ts = datetime.now(timezone.utc)
        actual_duration = (end_ts - start_ts).total_seconds()
        
        # Write end event
        self._write_event(
            events_file,
            {
                "event": "recorder_end",
                "ts_utc": end_ts.isoformat(),
                "actual_duration_seconds": actual_duration,
                "total_polls": self.total_polls,
                "successful_polls": self.successful_polls,
                "failed_polls": self.failed_polls,
            }
        )
        
        # Validate outputs
        if self.total_polls == 0:
            raise RuntimeError("No polls completed during run")
        
        if self.successful_polls == 0:
            raise RuntimeError("No successful polls during run")
        
        # HARD GATE: Clock anomaly
        if clock_anomaly_detected:
            # Write clock anomaly report
            anomaly_file = run_dir / "health" / "clock_anomaly.json"
            anomaly_file.parent.mkdir(parents=True, exist_ok=True)
            with open(anomaly_file, 'w') as f:
                json.dump({
                    "verdict": "FAIL",
                    "reason": "Clock anomaly detected - recorded timestamps in future",
                    "threshold_seconds": clock_anomaly_threshold_seconds,
                    "ts_check_utc": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
            raise RuntimeError("Clock anomaly detected - recorded timestamps are in the future!")
        
        for token_id, file_path in token_files.items():
            if not file_path.exists() or file_path.stat().st_size == 0:
                raise RuntimeError(f"No data written for token {token_id}")
        
        # Build metadata
        metadata = {
            "run_id": run_id,
            "family_id": self.family_id,
            "slug": self.slug,
            "condition_id": self.condition_id,
            "token_yes_id": self.token_yes_id,
            "token_no_id": self.token_no_id,
            "expiry_ts_utc": self.expiry_ts_utc,
            "start_ts_utc": start_ts.isoformat(),
            "end_ts_utc": end_ts.isoformat(),
            "run_start_ts_utc": run_start_ts_utc.isoformat(),
            "duration_requested_seconds": duration_seconds,
            "duration_actual_seconds": actual_duration,
            "total_polls": self.total_polls,
            "successful_polls": self.successful_polls,
            "failed_polls": self.failed_polls,
            "total_records": record_seq,
            "success_rate_percent": (self.successful_polls / self.total_polls * 100) if self.total_polls > 0 else 0,
            "records_by_token": self.records_by_token,
            "output_dir": str(run_dir),
            "config": {
                "poll_interval_seconds": self.config.poll_interval_seconds,
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "max_retries": self.config.max_retries,
            },
        }
        
        # Write metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log completion
        logger.info("=" * 80)
        logger.info(f"Recorder completed: {run_id}")
        logger.info(f"  Actual duration: {actual_duration:.1f}s")
        logger.info(f"  Total polls: {self.total_polls}")
        logger.info(f"  Successful: {self.successful_polls}")
        logger.info(f"  Failed: {self.failed_polls}")
        logger.info(f"  Success rate: {metadata['success_rate_percent']:.1f}%")
        logger.info(f"  Output: {run_dir}")
        logger.info("=" * 80)
        
        return metadata
    
    def _write_record(self, file_path: Path, record: RawPriceRecord) -> None:
        """Write a raw price record to JSONL file."""
        with open(file_path, 'a') as f:
            f.write(record.to_jsonl() + '\n')
    
    def _write_record_dict(self, file_path: Path, record_dict: dict) -> None:
        """Write a raw price record dict to JSONL file."""
        with open(file_path, 'a') as f:
            json.dump(record_dict, f)
            f.write('\n')
    
    def _write_event(self, file_path: Path, event: dict) -> None:
        """Write an event to events JSONL file."""
        with open(file_path, 'a') as f:
            json.dump(event, f)
            f.write('\n')
    
    def _both_tokens_404(self, threshold: int) -> bool:
        """Check if both tokens have consecutive 404s above threshold."""
        return (
            self.consecutive_404[self.token_yes_id] >= threshold
            and self.consecutive_404[self.token_no_id] >= threshold
        )
