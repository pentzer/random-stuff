"""
Robust REST client for Polymarket probability tape.

Polls /midpoint (preferred) and /price (fallback) endpoints.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Literal
import httpx

from .config import ProbabilityTapeConfig
from .schemas import RawPriceRecord


logger = logging.getLogger(__name__)


class PolymarketRESTClient:
    """REST client for Polymarket CLOB API price endpoints."""
    
    # Base URL for CLOB API
    BASE_URL = "https://clob.polymarket.com"
    
    def __init__(self, config: ProbabilityTapeConfig):
        """Initialize client."""
        self.config = config
        self.client = httpx.Client(
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PolymarketTapeRecorder/1.0)"},
        )
    
    def close(self):
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def poll_midpoint(
        self,
        token_id: str,
        fallback_to_price: bool = True,
    ) -> RawPriceRecord:
        """
        Poll midpoint price for a token.
        
        Args:
            token_id: Token ID to poll
            fallback_to_price: If True, fallback to /price if /midpoint fails
        
        Returns:
            RawPriceRecord with result
        """
        # Try midpoint first
        record = self._poll_endpoint(token_id, "midpoint")
        
        if record.error is None and record.http_status == 200:
            return record
        
        # Fallback to price if enabled
        if fallback_to_price:
            logger.warning(
                f"Midpoint failed for {token_id} (status={record.http_status}), "
                f"falling back to /price"
            )
            return self._poll_endpoint(token_id, "price")
        
        return record
    
    def _poll_endpoint(
        self,
        token_id: str,
        endpoint: Literal["midpoint", "price", "last"],
    ) -> RawPriceRecord:
        """
        Poll a specific endpoint with retries.
        
        Args:
            token_id: Token ID to poll
            endpoint: Endpoint name (midpoint, price, last)
        
        Returns:
            RawPriceRecord
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params = {"token_id": token_id}
        
        for attempt in range(self.config.max_retries):
            start_time = time.perf_counter()
            local_receive_ts = datetime.now(timezone.utc).isoformat()
            
            try:
                response = self.client.get(url, params=params)
                response_ms = (time.perf_counter() - start_time) * 1000
                
                # Parse response
                try:
                    payload = response.json()
                except Exception as json_err:
                    payload = {"raw_text": response.text}
                    logger.warning(f"Failed to parse JSON from {endpoint}: {json_err}")
                
                record = RawPriceRecord(
                    local_receive_ts_utc=local_receive_ts,
                    endpoint=endpoint,
                    token_id=token_id,
                    http_status=response.status_code,
                    response_ms=response_ms,
                    raw_payload=payload,
                    error=None if response.status_code == 200 else f"HTTP {response.status_code}",
                )
                
                if response.status_code == 200:
                    logger.debug(
                        f"Successfully polled {endpoint} for {token_id}: "
                        f"{response_ms:.1f}ms"
                    )
                else:
                    logger.warning(
                        f"Non-200 response from {endpoint} for {token_id}: "
                        f"{response.status_code}"
                    )
                
                return record
                
            except httpx.TimeoutException as e:
                response_ms = (time.perf_counter() - start_time) * 1000
                error_msg = f"Timeout after {response_ms:.0f}ms"
                
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"{error_msg} for {endpoint}/{token_id}, "
                        f"retry {attempt + 1}/{self.config.max_retries}"
                    )
                    time.sleep(self.config.retry_delay_seconds)
                    continue
                else:
                    logger.error(
                        f"{error_msg} for {endpoint}/{token_id}, "
                        f"all retries exhausted"
                    )
                    return RawPriceRecord(
                        local_receive_ts_utc=local_receive_ts,
                        endpoint=endpoint,
                        token_id=token_id,
                        http_status=0,
                        response_ms=response_ms,
                        raw_payload={},
                        error=error_msg,
                    )
            
            except Exception as e:
                response_ms = (time.perf_counter() - start_time) * 1000
                error_msg = f"Exception: {type(e).__name__}: {e}"
                
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"{error_msg} for {endpoint}/{token_id}, "
                        f"retry {attempt + 1}/{self.config.max_retries}"
                    )
                    time.sleep(self.config.retry_delay_seconds)
                    continue
                else:
                    logger.error(
                        f"{error_msg} for {endpoint}/{token_id}, "
                        f"all retries exhausted"
                    )
                    return RawPriceRecord(
                        local_receive_ts_utc=local_receive_ts,
                        endpoint=endpoint,
                        token_id=token_id,
                        http_status=0,
                        response_ms=response_ms,
                        raw_payload={},
                        error=error_msg,
                    )
        
        # Should not reach here
        raise RuntimeError("Retry logic error")
