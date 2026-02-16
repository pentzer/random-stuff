# Polymarket Probability Tape (Tape 1 Only)

**Artifact:** CAI-POLYMARKET-PROBABILITY-TAPE-CLEAN-0001  
**Status:** FROZEN  
**Parent:** CAI-POLYMARKET-DISCOVERY-CLEAN-0001

## Overview

This module produces a single, clean Polymarket data product: a **per-second implied probability series** for BTC 15-minute UP/DOWN contract families.

**Key characteristics:**
- REST polling only (NO WebSocket, NO CLOB book microstructure)
- Midpoint-derived probability proxy (explicitly non-executable)
- Deterministic 1-second normalization grid
- Append-only raw capture + reproducible normalization
- Strict health gate (NO soft landings)

## What This Is (Tape 1)

A time series of implied probabilities derived from token prices:

- `p_up_proxy(t)`: Clamped midpoint price for UP token at time `t`
- `p_down_proxy(t)`: Clamped midpoint price for DOWN token at time `t`
- `p_sum(t) = p_up_proxy(t) + p_down_proxy(t)` (diagnostic, NOT enforced = 1)
- `p_up_norm(t)`, `p_down_norm(t)`: Optional normalized probabilities (sum = 1)

**NOT EXECUTABLE:** These are midpoint-derived proxies, not bid/ask quotes.

## What Was Removed (Tape 2)

All CLOB microstructure systems were deleted:
- `src/polymarket/recorder.py` (WebSocket + /book recorder)
- `src/polymarket/clob_health_gate.py` (CLOB health modules)
- Book schemas, WS message parsing
- Normalization modules assuming bid/ask state

**Reason:** Single coherent data product for modeling. CLOB microstructure (bid/ask spreads, book depth) is not needed for implied probability modeling.

## Data Product Definition

### Raw Data (Append-Only JSONL)

Location: `data/polymarket/probability_tape/raw/`

Structure:
```
raw/
  token_<tokenId>/
    midpoint_YYYYMMDD_HH.jsonl
    price_YYYYMMDD_HH.jsonl
  events/
    tape_events.jsonl
  recorder_*_metadata.json
```

Each JSONL line contains:
- `local_receive_ts_utc`: Local receipt timestamp (UTC ISO)
- `endpoint`: "midpoint" | "price" | "last"
- `token_id`: Token ID polled
- `http_status`: HTTP status code
- `response_ms`: Response time in milliseconds
- `raw_payload`: Verbatim JSON response
- `error`: Error message (nullable)

### Normalized Data (1-Second Grid Parquet)

Location: `data/polymarket/probability_tape/normalized/`

File: `probability_tape_BTC_15M_UPDOWN.parquet`

Schema:
- `ts_utc`: Timestamp (second-level)
- `family_id`: Family ID (e.g., "BTC_15M_UPDOWN")
- `resolution_id`: Resolution ID from discovery
- `token_up_id`: UP token ID
- `token_down_id`: DOWN token ID
- `p_up_raw`: Raw price from endpoint
- `p_down_raw`: Raw price from endpoint
- `p_up_proxy`: Clamped probability [0, 1]
- `p_down_proxy`: Clamped probability [0, 1]
- `p_sum`: Sum of proxies (diagnostic)
- `p_up_norm`: Normalized probability (optional)
- `p_down_norm`: Normalized probability (optional)
- `up_source`: "midpoint" | "price" | "carry" | "none"
- `down_source`: "midpoint" | "price" | "carry" | "none"
- `up_staleness_seconds`: Seconds since last update
- `down_staleness_seconds`: Seconds since last update
- `expiry_ts_utc`: Contract expiry timestamp
- `time_to_expiry_seconds`: Computed time to expiry

**Normalization logic:**
- For each second, use latest successfully received value
- If no update in that second, carry forward last known value
- Max carry-forward: 60 seconds (configurable)
- Clamp all probabilities to [0, 1]
- Track staleness explicitly

### Gate Output (Health Validation)

Location: `data/polymarket/probability_tape/gate/`

Files:
- `probability_tape_gate_report.json`: Full gate verdict
- `probability_tape_gate_summary.md`: Human-readable summary
- `gate_*_metadata.json`: Gate run metadata

**Verdict:** PASS | FAIL

## Endpoints Used

### Primary: `/midpoint/{token_id}`

**Preferred endpoint.** Returns midpoint price (non-executable proxy).

### Fallback: `/price/{token_id}`

Used only if `/midpoint` returns non-200 or is unavailable.

**Base URL:** `https://clob.polymarket.com`

## Health Gate Thresholds

Default thresholds (strict, configurable):

| Threshold | Default | Description |
|-----------|---------|-------------|
| `gate_min_run_duration_seconds` | 300 | Minimum recording duration |
| `gate_min_percent_seconds_with_both_tokens` | 95.0 | Min % of seconds with both UP and DOWN values |
| `gate_max_staleness_seconds` | 10 | Max staleness allowed (after warmup) |
| `gate_min_midpoint_success_rate` | 80.0 | Min % of polls using midpoint (vs fallback) |

**NO SOFT LANDINGS:** If any threshold fails, verdict = FAIL.

Warmup period (5s) allows initial NaNs.

## Usage

### Run End-to-End

```bash
python -m src.polymarket.probability_tape.cli --duration-seconds 300 --require-pass
```

This will:
1. Check discovery (require FOUND status for BTC_15M_UPDOWN)
2. Record raw data for 300 seconds
3. Normalize to 1-second grid
4. Run health gate
5. Exit 0 only if verdict = PASS

### Optional Arguments

```bash
--family BTC_15M_UPDOWN          # Family to record (default)
--duration-seconds 300            # Recording duration
--require-pass                    # Exit non-zero if FAIL
--verbose                         # Debug logging
```

### Discovery Dependency

**MUST run discovery first:**

```bash
python -m src.polymarket.discovery.cli --family BTC_15M_UPDOWN --require-found
```

Discovery must produce:
- `resolution_status == FOUND`
- Exactly 2 primary token IDs (UP and DOWN)
- Resolution file in `data/polymarket/discovery/resolution/`

If discovery FAILS, probability tape CANNOT proceed.

## Configuration

See `src/polymarket/probability_tape/config.py`

Key settings:
- `poll_interval_seconds`: 1.0 (default)
- `request_timeout_seconds`: 5.0
- `max_retries`: 3
- `carry_forward_max_seconds`: 60
- Gate thresholds (above)

## Example Output

### Successful Run

```
Polymarket Probability Tape CLI
================================================================

Step 1: Loading discovery
------------------------------------------------------------
Family: BTC_15M_UPDOWN
Resolution: resolution_BTC_15M_UPDOWN_20260202_063125
UP token: 123456
DOWN token: 789012
Expiry: 2026-02-02T07:00:00Z

Step 2: Recording raw data
------------------------------------------------------------
Starting probability tape recorder: recorder_20260202_120000
Duration: 300s, Poll interval: 1.0s
...
Recorder finished: recorder_20260202_120000
Total polls: 600
Successful: 598, Failed: 2

Step 3: Normalizing to 1-second grid
------------------------------------------------------------
Read 299 UP records, 299 DOWN records
Created grid with 300 seconds
Wrote normalized tape to data/polymarket/probability_tape/normalized/probability_tape_BTC_15M_UPDOWN.parquet

Step 4: Running health gate
------------------------------------------------------------
Loaded 300 rows
Metrics:
  run_duration_seconds: 299.0
  total_seconds: 300
  seconds_with_both_tokens: 298
  percent_seconds_with_both_tokens: 99.3
  max_up_staleness_seconds: 1.0
  max_down_staleness_seconds: 1.0
  midpoint_success_rate_up: 99.7
  midpoint_success_rate_down: 99.7

Threshold checks:
  run_duration_sufficient: PASS
  percent_seconds_with_both_sufficient: PASS
  up_staleness_acceptable: PASS
  down_staleness_acceptable: PASS
  up_midpoint_success_rate_sufficient: PASS
  down_midpoint_success_rate_sufficient: PASS

================================================================
FINAL VERDICT: PASS
================================================================
✓ Probability tape PASSED health gate
  Coverage: 99.3%
  Max staleness: UP=1.0s, DOWN=1.0s
  Output: data/polymarket/probability_tape/normalized/probability_tape_BTC_15M_UPDOWN.parquet
```

## Failure Modes

### Discovery Not FOUND

If BTC 15M UP/DOWN markets are not currently available on Polymarket:

```
Resolution status is FAILED, not FOUND
Discovery failed. Cannot proceed with probability tape.
```

**Resolution:** Wait for markets to become available, or use a different family.

### Endpoint Fragility

If `/midpoint` and `/price` both fail frequently:

```
✗ Probability tape FAILED health gate
Failure reasons:
  - up_midpoint_success_rate_insufficient
  - percent_seconds_with_both_sufficient
```

**This is a HARD FAILURE.** Do not weaken thresholds silently.

### Staleness Violation

If updates become too stale (> 10s):

```
Failure reasons:
  - up_staleness_acceptable
  - down_staleness_acceptable
```

## Architecture

```
probability_tape/
├── __init__.py          # Module package
├── config.py            # Configuration and thresholds
├── schemas.py           # Dataclass schemas for all data structures
├── rest_client.py       # Robust REST client with retries
├── recorder.py          # Append-only raw data recorder
├── normalization.py     # Deterministic 1-second grid
├── gate.py              # Health gate evaluation
├── cli.py               # End-to-end CLI entrypoint
└── README.md            # This file
```

## Invariants

1. **Raw data is append-only.** Never modify or delete raw JSONL.
2. **Normalization is deterministic.** Same raw data → same normalized output.
3. **Gate thresholds are explicit.** No hidden soft landings.
4. **Discovery is authoritative.** Token IDs come from discovery resolution only.
5. **Probability proxy is non-executable.** Midpoint ≠ tradeable price.
6. **No book data.** This is Tape 1 (probability proxy), not Tape 2 (CLOB microstructure).

## Migration from Old System

**Old CLOB Pipeline (Deleted):**
- `src/polymarket/recorder.py` → REMOVED
- `src/polymarket/clob_health_gate.py` → REMOVED
- WS book streaming → REMOVED
- Bid/ask normalization → REMOVED

**New Probability Tape:**
- REST polling only
- Midpoint-derived proxy
- Single coherent data product

**Historical data:** Old CLOB data directories are NOT deleted automatically. They remain for reference but are no longer written to.

## Frozen Decisions

- Tape 1 ONLY (probability proxy). Tape 2 (CLOB book) is OUT OF SCOPE.
- Midpoint endpoint preferred over price.
- 1-second normalization grid (not tick-by-tick).
- Strict gate thresholds (95% coverage, 10s max staleness).
- No probabilistic smoothing beyond carry-forward.

## Contact

For questions about this artifact, refer to parent artifact CAI-POLYMARKET-DISCOVERY-CLEAN-0001 and project documentation in `project_context/`.
