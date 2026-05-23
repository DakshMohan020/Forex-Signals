"""
fetch_data.py  —  Forex Swing Trading System v4.4
Downloads and refreshes H4 OHLC data for all configured pairs.

Run manually or call from scheduler.py before each signal check.

Usage:
    python fetch_data.py              # update all pairs
    python fetch_data.py --pair GBPUSD  # update one pair only
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# yfinance ticker symbols
TICKERS = {
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    # "EURUSD": "EURUSD=X",
    # "AUDUSD": "AUDUSD=X",
}

HISTORY_YEARS = 1    # how many years of history to fetch on first run
H4_BARS_MIN   = 500   # minimum bars needed for indicator warmup


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_pair(pair: str, ticker: str, out_path: Path) -> bool:
    """
    Download H4 data via yfinance.
    If file already exists, only fetches the missing tail (incremental update).
    Returns True on success.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run:  pip install yfinance")
        return False

    # Determine date range
    start_date = (datetime.now(timezone.utc) - timedelta(days=365 * HISTORY_YEARS)
                  ).strftime("%Y-%m-%d")

    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["Date"], index_col="Date")
        existing.index = existing.index.tz_localize(None)
        last_date  = existing.index[-1]
        # Only fetch bars after last known date
        start_date = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"  [{pair}] Existing data to {last_date.date()}. Fetching from {start_date}...")
    else:
        existing = pd.DataFrame()
        print(f"  [{pair}] No data found. Fetching {HISTORY_YEARS} years from {start_date}...")

    # yfinance: download 1h bars (max 730 days per call), then resample to 4h
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1h",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        print(f"  [{pair}] No data returned from yfinance.")
        return False

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Resample 1h → 4h
    raw.index = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    h4 = raw.resample("4h").agg({
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume":"sum",
    }).dropna()

    h4.index.name = "Date"
    h4.columns    = ["open", "high", "low", "close", "tick_volume"]

    # Merge with existing and deduplicate
    if not existing.empty:
        existing.columns = [c.lower() for c in existing.columns]
        combined = pd.concat([existing, h4])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = h4

    if len(combined) < H4_BARS_MIN:
        print(f"  [{pair}] Only {len(combined)} bars — need at least {H4_BARS_MIN}. "
              f"Try increasing HISTORY_YEARS.")
        return False

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(out_path)
    print(f"  [{pair}] Saved {len(combined)} bars → {out_path}")
    return True


def fetch_all(pairs: list[str] | None = None) -> bool:
    """Fetch/update all (or specified) pairs. Returns True if all succeeded."""
    targets = pairs or list(TICKERS.keys())
    results = {}
    for pair in targets:
        if pair not in TICKERS:
            print(f"  [{pair}] Unknown pair — add it to TICKERS in fetch_data.py")
            results[pair] = False
            continue
        ticker   = TICKERS[pair]
        out_path = DATA_DIR / f"{pair}h4.csv"
        results[pair] = fetch_pair(pair, ticker, out_path)

    # Summary
    ok  = [p for p, v in results.items() if v]
    bad = [p for p, v in results.items() if not v]
    print(f"\n  Data fetch complete: {len(ok)} OK  {len(bad)} failed")
    if bad:
        print(f"  Failed pairs: {bad}")
    return len(bad) == 0


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch/update H4 forex data")
    parser.add_argument("--pair", nargs="+", help="Specific pair(s) to update, e.g. GBPUSD")
    args = parser.parse_args()

    print(f"\nForex Data Fetcher  —  v4.4")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    success = fetch_all(args.pair)
    sys.exit(0 if success else 1)
