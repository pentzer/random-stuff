"""
UI-based discovery for BTC Up/Down rolling markets (5m and 15m).

Discovery path:
1. Compute candidate slugs from timestamp windows
2. Fetch polymarket.com/event/<slug> with browser User-Agent
3. Parse __NEXT_DATA__ JSON from HTML
4. Extract market object, clobTokenIds, conditionId, expiry
5. Persist discovery evidence snapshot
6. Return discovery snapshot

This is the AUTHORITATIVE discovery path for rolling BTC markets.
Gamma API is NOT used.

Evidence persistence: Snapshots are ephemeral + lossy, persisting only sufficient
evidence to justify token ID selection.

Supports multiple families (5m, 15m) with distinct window sizes and slug formats.
"""

import json
import logging
import re
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, asdict

import httpx
from bs4 import BeautifulSoup

from .families import get_family_spec


logger = logging.getLogger(__name__)


@dataclass
class DiscoverySnapshot:
    """Discovery result for a BTC 15m Up/Down market."""
    
    resolved_at_utc: str  # ISO format
    slug: str
    condition_id: str
    token_yes_id: str
    token_no_id: str
    expiry_ts_utc: Optional[str] = None
    question: Optional[str] = None
    title: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DiscoveryEvidence:
    """Ephemeral + lossy evidence of discovery (AUTHORITATIVE format)."""
    
    discovery_ts_utc: str  # ISO8601, UTC timestamp of discovery
    discovery_method: str  # "hardcoded_slug" | "ui_probe_v1"
    page_url: str  # URL of page fetched
    resolved_slug: str  # The slug that was successfully resolved
    condition_id: str  # Condition ID from market
    yes_token_id: str  # Token ID for YES/UP
    no_token_id: str  # Token ID for NO/DOWN
    evidence: dict  # Object with type/extract/sha256
    notes: str  # Semantic binding / metadata
    version: str = "discovery_snapshot_v1"
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "discovery_ts_utc": self.discovery_ts_utc,
            "discovery_method": self.discovery_method,
            "page_url": self.page_url,
            "resolved_slug": self.resolved_slug,
            "condition_id": self.condition_id,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "evidence": self.evidence,
            "notes": self.notes,
            "version": self.version,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_sha256(self) -> str:
        """Compute SHA256 hash of normalized JSON representation."""
        normalized = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


@dataclass
class DiscoveryDebugInfo:
    """Debug info from discovery attempt."""
    
    attempted_slugs: list[str]
    successful_slug: Optional[str]
    http_status: Optional[int]
    next_data_found: bool
    market_object_found: bool
    extracted_market_slice: Optional[dict] = None  # minimal slice, not full page
    error: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)


class UIDiscovery:
    """UI-based discovery for BTC 15m Up/Down markets."""
    
    BASE_URL = "https://polymarket.com/event"
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    MAX_EVIDENCE_SIZE_BYTES = 65536  # 64KB limit on evidence extract
    
    def __init__(self, timeout_seconds: float = 10.0):
        """Initialize discovery client."""
        self.timeout_seconds = timeout_seconds
    
    def create_discovery_evidence(
        self,
        snapshot: DiscoverySnapshot,
        extracted_market_slice: dict,
        discovery_method: str = "ui_probe_v1",
    ) -> DiscoveryEvidence:
        """
        Create discovery evidence from snapshot and extracted data.
        
        Args:
            snapshot: DiscoverySnapshot from discovery
            extracted_market_slice: Minimal market extract from __NEXT_DATA__
            discovery_method: "hardcoded_slug" | "ui_probe_v1"
        
        Returns:
            DiscoveryEvidence object
        """
        # Create evidence extract (bounded JSON)
        evidence_extract = json.dumps(extracted_market_slice, indent=2)
        
        # Truncate if necessary
        if len(evidence_extract) > self.MAX_EVIDENCE_SIZE_BYTES:
            logger.warning(
                f"Evidence extract truncated from {len(evidence_extract)} to {self.MAX_EVIDENCE_SIZE_BYTES} bytes"
            )
            evidence_extract = evidence_extract[:self.MAX_EVIDENCE_SIZE_BYTES]
        
        # Compute hash
        evidence_sha256 = hashlib.sha256(evidence_extract.encode('utf-8')).hexdigest()
        
        # Create evidence object
        evidence = {
            "evidence_type": "html_substring",
            "evidence_extract": evidence_extract,
            "evidence_sha256": evidence_sha256,
        }
        
        # Create notes on semantic binding
        notes = f"BTC Up/Down market - Family: {snapshot.slug.split('-')[2]}. Slug: {snapshot.slug}. YES token (up): {snapshot.token_yes_id}, NO token (down): {snapshot.token_no_id}. Semantic binding is per-discovery snapshot; UP=YES, DOWN=NO are outcomes, not global invariants."
        
        # Create DiscoveryEvidence
        discovery_evidence = DiscoveryEvidence(
            discovery_ts_utc=snapshot.resolved_at_utc,
            discovery_method=discovery_method,
            page_url=f"{self.BASE_URL}/{snapshot.slug}",
            resolved_slug=snapshot.slug,
            condition_id=snapshot.condition_id,
            yes_token_id=snapshot.token_yes_id,
            no_token_id=snapshot.token_no_id,
            evidence=evidence,
            notes=notes,
        )
        
        return discovery_evidence
    
    def discover_by_family(
        self,
        family_id: str,
        current_time: Optional[datetime] = None,
        max_windows_ahead: Optional[int] = None,
    ) -> tuple[Optional[DiscoverySnapshot], DiscoveryDebugInfo]:
        """
        Discover market for a specific family.
        
        Args:
            family_id: Family identifier (e.g., "BTC_5M_UPDOWN", "BTC_15M_UPDOWN")
            current_time: Reference time (default: now UTC)
            max_windows_ahead: Number of windows to try ahead (default from family spec)
        
        Returns:
            (snapshot, debug_info)
            snapshot is None if not found
        """
        family_spec = get_family_spec(family_id)
        if family_spec is None:
            raise ValueError(f"Unknown family: {family_id}")
        
        if max_windows_ahead is None:
            max_windows_ahead = family_spec.max_windows_ahead_default
        
        return self._discover_generic(
            window_seconds=family_spec.window_seconds,
            slug_prefix=family_spec.slug_prefix,
            current_time=current_time,
            max_windows_ahead=max_windows_ahead,
        )
    
    def _discover_generic(
        self,
        window_seconds: int,
        slug_prefix: str,
        current_time: Optional[datetime] = None,
        max_windows_ahead: int = 8,
    ) -> tuple[Optional[DiscoverySnapshot], DiscoveryDebugInfo]:
        """
        Generic discovery for any window size and slug prefix.
        
        Args:
            window_seconds: Market window duration (300 for 5m, 900 for 15m)
            slug_prefix: Slug prefix (e.g., "btc-updown-5m-", "btc-updown-15m-")
            current_time: Reference time (default: now UTC)
            max_windows_ahead: Number of windows to try ahead
        
        Returns:
            (snapshot, debug_info)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Generate candidate slugs
        candidate_slugs = self._generate_candidate_slugs_generic(
            window_seconds, slug_prefix, current_time, max_windows_ahead
        )
        
        debug_info = DiscoveryDebugInfo(
            attempted_slugs=candidate_slugs,
            successful_slug=None,
            http_status=None,
            next_data_found=False,
            market_object_found=False,
        )
        
        # Try each candidate
        for slug in candidate_slugs:
            logger.info(f"Trying slug: {slug}")
            
            snapshot = self._try_slug(slug, debug_info)
            if snapshot is not None:
                snapshot.resolved_at_utc = current_time.isoformat()
                debug_info.successful_slug = slug
                logger.info(f"✓ Successfully resolved slug: {slug}")
                logger.info(f"  conditionId: {snapshot.condition_id}")
                logger.info(f"  token_yes: {snapshot.token_yes_id}")
                logger.info(f"  token_no: {snapshot.token_no_id}")
                return snapshot, debug_info
        
        # No slug worked
        debug_info.error = f"No valid market found in {len(candidate_slugs)} candidate slugs"
        logger.error(debug_info.error)
        return None, debug_info
    
    def discover_btc_15m_updown(
        self,
        current_time: Optional[datetime] = None,
        max_windows_ahead: int = 8,
    ) -> tuple[Optional[DiscoverySnapshot], DiscoveryDebugInfo]:
        """
        Discover current BTC 15m Up/Down market.
        
        DEPRECATED: Use discover_by_family("BTC_15M_UPDOWN", ...) instead.
        Kept for backward compatibility.
        
        Args:
            current_time: Reference time (default: now UTC)
            max_windows_ahead: Number of 15m windows to try ahead
        
        Returns:
            (snapshot, debug_info)
            snapshot is None if not found
        """
        return self.discover_by_family(
            family_id="BTC_15M_UPDOWN",
            current_time=current_time,
            max_windows_ahead=max_windows_ahead,
        )
    
    def discover_btc_5m_updown(
        self,
        current_time: Optional[datetime] = None,
        max_windows_ahead: int = 8,
    ) -> tuple[Optional[DiscoverySnapshot], DiscoveryDebugInfo]:
        """
        Discover current BTC 5m Up/Down market.
        
        Args:
            current_time: Reference time (default: now UTC)
            max_windows_ahead: Number of 5m windows to try ahead
        
        Returns:
            (snapshot, debug_info)
            snapshot is None if not found
        """
        return self.discover_by_family(
            family_id="BTC_5M_UPDOWN",
            current_time=current_time,
            max_windows_ahead=max_windows_ahead,
        )
    
    def _generate_candidate_slugs_generic(
        self,
        window_seconds: int,
        slug_prefix: str,
        current_time: datetime,
        max_windows_ahead: int,
    ) -> list[str]:
        """
        Generate candidate slugs for any window size and prefix.
        
        Format: {slug_prefix}{ts_rounded_to_window_seconds}
        
        Args:
            window_seconds: Window duration (300 for 5m, 900 for 15m)
            slug_prefix: Prefix for slug (e.g., "btc-updown-5m-", "btc-updown-15m-")
            current_time: Reference time
            max_windows_ahead: Number of windows to try
        
        Returns:
            List of candidate slugs (current + future windows)
        """
        slugs = []
        
        for i in range(max_windows_ahead + 1):
            offset_time = current_time + timedelta(seconds=i * window_seconds)
            ts_unix = int(offset_time.timestamp())
            ts_rounded = (ts_unix // window_seconds) * window_seconds
            slug = f"{slug_prefix}{ts_rounded}"
            slugs.append(slug)
        
        return slugs
    
    def _generate_candidate_slugs(
        self,
        current_time: datetime,
        max_windows_ahead: int,
    ) -> list[str]:
        """
        Generate candidate slugs for BTC 15m Up/Down markets.
        
        DEPRECATED: Use _generate_candidate_slugs_generic() instead.
        Kept for backward compatibility.
        
        Format: btc-updown-15m-{ts_rounded_to_900s}
        
        Args:
            current_time: Reference time
            max_windows_ahead: Number of 15m windows to try
        
        Returns:
            List of candidate slugs (current + future windows)
        """
        return self._generate_candidate_slugs_generic(
            window_seconds=900,
            slug_prefix="btc-updown-15m-",
            current_time=current_time,
            max_windows_ahead=max_windows_ahead,
        )
    
    def _try_slug(
        self,
        slug: str,
        debug_info: DiscoveryDebugInfo,
    ) -> Optional[DiscoverySnapshot]:
        """
        Try to resolve a single slug.
        
        Args:
            slug: Market slug to try
            debug_info: Debug info object to populate
        
        Returns:
            DiscoverySnapshot if successful, None otherwise
        """
        url = f"{self.BASE_URL}/{slug}"
        
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                    follow_redirects=True,
                )
                debug_info.http_status = response.status_code
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for slug: {slug}")
                    return None
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find __NEXT_DATA__ script tag
                next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})
                if next_data_script is None:
                    logger.warning(f"__NEXT_DATA__ not found for slug: {slug}")
                    debug_info.next_data_found = False
                    return None
                
                debug_info.next_data_found = True
                
                # Parse JSON
                next_data = json.loads(next_data_script.string)
                
                # Extract market data
                # Navigate: pageProps -> page -> data or similar
                # The structure varies, so try common paths
                market = self._extract_market_from_next_data(next_data, slug)
                
                if market is None:
                    logger.warning(f"Market not found in __NEXT_DATA__ for slug: {slug}")
                    debug_info.market_object_found = False
                    return None
                
                debug_info.market_object_found = True
                
                # Extract required fields
                condition_id = market.get('conditionId')
                clob_token_ids = market.get('clobTokenIds')
                
                if not condition_id:
                    logger.warning(f"conditionId missing for slug: {slug}")
                    return None
                
                if not clob_token_ids or len(clob_token_ids) != 2:
                    logger.warning(
                        f"clobTokenIds invalid for slug: {slug} "
                        f"(expected 2, got {len(clob_token_ids) if clob_token_ids else 0})"
                    )
                    return None
                
                # Extract optional fields
                question = market.get('question') or market.get('title')
                title = market.get('title')
                
                # Extract expiry
                expiry_ts_utc = None
                if 'endDate' in market:
                    expiry_ts_utc = market['endDate']
                elif 'endTime' in market:
                    expiry_ts_utc = market['endTime']
                elif 'end_date_iso' in market:
                    expiry_ts_utc = market['end_date_iso']
                
                # Store minimal debug slice
                debug_info.extracted_market_slice = {
                    'conditionId': condition_id,
                    'clobTokenIds': clob_token_ids,
                    'question': question,
                    'title': title,
                    'endDate': market.get('endDate'),
                    'endTime': market.get('endTime'),
                }
                
                # Create snapshot
                snapshot = DiscoverySnapshot(
                    resolved_at_utc="",  # Will be set by caller
                    slug=slug,
                    condition_id=condition_id,
                    token_yes_id=clob_token_ids[0],
                    token_no_id=clob_token_ids[1],
                    expiry_ts_utc=expiry_ts_utc,
                    question=question,
                    title=title,
                )
                
                return snapshot
                
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching slug: {slug}")
            debug_info.error = "HTTP timeout"
            return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP error for slug {slug}: {e}")
            debug_info.error = f"HTTP error: {e}"
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for slug {slug}: {e}")
            debug_info.error = f"JSON decode error: {e}"
            return None
        except Exception as e:
            logger.error(f"Unexpected error for slug {slug}: {e}")
            debug_info.error = f"Unexpected error: {e}"
            return None
    
    def _extract_market_from_next_data(
        self,
        next_data: dict,
        slug: str,
    ) -> Optional[dict]:
        """
        Extract market object from __NEXT_DATA__ payload.
        
        Tries common paths in Next.js data structure.
        
        Args:
            next_data: Parsed __NEXT_DATA__ JSON
            slug: Target slug for matching
        
        Returns:
            Market dict if found, None otherwise
        """
        # Try common paths
        paths_to_try = [
            ['props', 'pageProps', 'page', 'data'],
            ['props', 'pageProps', 'market'],
            ['props', 'pageProps', 'data'],
            ['props', 'pageProps', 'dehydratedState', 'queries', 0, 'state', 'data'],
        ]
        
        for path in paths_to_try:
            obj = next_data
            for key in path:
                if isinstance(obj, dict):
                    obj = obj.get(key)
                elif isinstance(obj, list) and isinstance(key, int):
                    if len(obj) > key:
                        obj = obj[key]
                    else:
                        obj = None
                else:
                    obj = None
                
                if obj is None:
                    break
            
            if obj is not None and isinstance(obj, dict):
                # Check if this looks like a market object
                if 'conditionId' in obj and 'clobTokenIds' in obj:
                    return obj
                
                # Check if slug matches
                if obj.get('slug') == slug:
                    return obj
        
        # Fallback: search recursively for market with matching slug
        market = self._recursive_search_market(next_data, slug)
        if market is not None:
            return market
        
        logger.warning(f"Could not extract market from __NEXT_DATA__ for slug: {slug}")
        return None
    
    def _recursive_search_market(
        self,
        obj: any,
        slug: str,
        max_depth: int = 10,
        current_depth: int = 0,
    ) -> Optional[dict]:
        """
        Recursively search for market object matching slug.
        
        Args:
            obj: Object to search
            slug: Target slug
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
        
        Returns:
            Market dict if found, None otherwise
        """
        if current_depth > max_depth:
            return None
        
        if isinstance(obj, dict):
            # Check if this is a market
            if obj.get('slug') == slug and 'conditionId' in obj and 'clobTokenIds' in obj:
                return obj
            
            # Recurse into values
            for value in obj.values():
                result = self._recursive_search_market(
                    value, slug, max_depth, current_depth + 1
                )
                if result is not None:
                    return result
        
        elif isinstance(obj, list):
            # Recurse into list items
            for item in obj:
                result = self._recursive_search_market(
                    item, slug, max_depth, current_depth + 1
                )
                if result is not None:
                    return result
        
        return None
