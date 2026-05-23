"""
signals.py  —  Forex Swing Trading System v4.4
Live signal engine. Import this module in scanner.py and scheduler.py.

Usage:
    from signals import check_signals
    alerts = check_signals()
    for a in alerts:
        print(a)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# ── Config — edit these to match your setup ──────────────────────────────────
DATA_DIR = Path("data")          # folder where your H4 CSV files live

PAIRS = {
    "GBPUSD": {"file": DATA_DIR / "GBPUSDh4.csv", "divisor": 1.0},
    "USDJPY": {"file": DATA_DIR / "USDJPYh4.csv", "divisor": 1.0},
    # Add more pairs here:
    # "EURUSD": {"file": DATA_DIR / "EURUSDh4.csv", "divisor": 1.0},
    # "AUDUSD": {"file": DATA_DIR / "AUDUSDh4.csv", "divisor": 1.0},
}

ACCOUNT_EQUITY  = 10_000.0   # update to your live account balance
RISK_PCT        = 0.02        # 2% risk per trade
CORR_RISK_MULT  = 0.5         # 50% risk if another pair already in trade

# Indicator parameters — do not change without re-running backtest
ATR_PERIOD      = 14
ATR_STOP_MULT   = 2.0
TP1_R           = 1.0
TP2_R           = 3.0
EMA_FAST        = 20
EMA_SLOW        = 50
EMA_TREND       = 200
RSI_PERIOD      = 14
ADX_PERIOD      = 14
ADX_THRESHOLD   = 30
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
SESSION_START   = 12           # UTC hour
SESSION_END     = 20           # UTC hour


# ── Data loading ─────────────────────────────────────────────────────────────
def load_pair(path: Path, divisor: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] / divisor
    return df.sort_index().dropna()


# ── Indicators ────────────────────────────────────────────────────────────────
def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def _adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up   = h - h.shift()
    down = l.shift() - l
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    tr   = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1 / n, min_periods=n).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, min_periods=n).mean() / atr14
    ndi   = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/n, min_periods=n).mean() / atr14
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    adx   = dx.ewm(alpha=1 / n, min_periods=n).mean()
    return adx, pdi, ndi


def _rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"]        = _atr(df, ATR_PERIOD)
    df["ema_fast"]   = df["close"].ewm(span=EMA_FAST,  min_periods=EMA_FAST).mean()
    df["ema_slow"]   = df["close"].ewm(span=EMA_SLOW,  min_periods=EMA_SLOW).mean()
    df["ema_trend"]  = df["close"].ewm(span=EMA_TREND, min_periods=EMA_TREND).mean()
    df["rsi"]        = _rsi(df["close"], RSI_PERIOD)
    adx, pdi, ndi    = _adx(df, ADX_PERIOD)
    df["adx"]        = adx
    df["pdi"]        = pdi
    df["ndi"]        = ndi
    exp1             = df["close"].ewm(span=MACD_FAST, min_periods=MACD_FAST).mean()
    exp2             = df["close"].ewm(span=MACD_SLOW, min_periods=MACD_SLOW).mean()
    df["macd"]       = exp1 - exp2
    df["macd_sig"]   = df["macd"].ewm(span=MACD_SIGNAL, min_periods=MACD_SIGNAL).mean()
    df["macd_hist"]  = df["macd"] - df["macd_sig"]
    df["d1_trend"]   = df["close"].rolling(6).mean()
    df["w1_trend"]   = df["close"].rolling(42).mean()
    return df.dropna()


# ── Signal logic ──────────────────────────────────────────────────────────────
def _is_long(bar, prev_hist):
    return (
        bar["close"]    > bar["ema_trend"]   and
        bar["ema_fast"] > bar["ema_slow"]    and
        bar["close"]    > bar["d1_trend"]    and
        bar["close"]    > bar["w1_trend"]    and
        bar["adx"]      > ADX_THRESHOLD      and
        bar["pdi"]      > bar["ndi"]         and
        55 < bar["rsi"] < 75                 and
        bar["macd_hist"] > 0                 and
        bar["macd_hist"] > prev_hist         and
        SESSION_START <= bar.name.hour < SESSION_END
    )


def _is_short(bar, prev_hist):
    return (
        bar["close"]    < bar["ema_trend"]   and
        bar["ema_fast"] < bar["ema_slow"]    and
        bar["close"]    < bar["d1_trend"]    and
        bar["close"]    < bar["w1_trend"]    and
        bar["adx"]      > ADX_THRESHOLD      and
        bar["ndi"]      > bar["pdi"]         and
        25 < bar["rsi"] < 45                 and
        bar["macd_hist"] < 0                 and
        bar["macd_hist"] < prev_hist         and
        SESSION_START <= bar.name.hour < SESSION_END
    )


# ── Trade plan calculator ─────────────────────────────────────────────────────
def _trade_plan(signal_bar, direction: int, equity: float,
                correlated_open: bool) -> dict:
    """
    Build the full trade plan from the signal bar.
    direction: +1 = long, -1 = short
    Entry is at the OPEN of the NEXT bar (next-bar confirmation).
    We approximate entry as signal_bar.close (worst case = signal bar close).
    """
    entry     = signal_bar["close"]
    atr       = signal_bar["atr"]
    stop_dist = ATR_STOP_MULT * atr

    stop  = entry - direction * stop_dist
    tp1   = entry + direction * TP1_R * stop_dist
    tp2   = entry + direction * TP2_R * stop_dist

    risk_mult  = CORR_RISK_MULT if correlated_open else 1.0
    risk_usd   = equity * RISK_PCT * risk_mult
    # pip value approximation: risk_usd / stop_dist_in_price_units
    size_units = risk_usd / stop_dist if stop_dist > 0 else 0

    return {
        "direction":      "LONG" if direction == 1 else "SHORT",
        "signal_bar":     signal_bar.name.strftime("%Y-%m-%d %H:%M UTC"),
        "entry_approx":   round(entry, 5),
        "stop_loss":      round(stop,  5),
        "tp1":            round(tp1,   5),
        "tp2":            round(tp2,   5),
        "stop_dist_pips": round(stop_dist * 10000, 1),   # approx pips for 4-dp pairs
        "risk_usd":       round(risk_usd, 2),
        "size_units":     round(size_units, 0),
        "corr_adjusted":  correlated_open,
        "adx":            round(signal_bar["adx"], 1),
        "rsi":            round(signal_bar["rsi"], 1),
        "atr":            round(atr, 5),
    }


# ── Main entry point ──────────────────────────────────────────────────────────
def check_signals(equity: float = ACCOUNT_EQUITY,
                  open_positions: list[str] | None = None) -> list[dict]:
    """
    Load data for all pairs, run indicators, check the last completed bar
    for a signal, and return a list of alert dicts.

    Parameters
    ----------
    equity          : current account equity
    open_positions  : list of pair names with an open trade, e.g. ["GBPUSD"]

    Returns
    -------
    List of dicts, one per signal found. Empty list = no signals.
    """
    if open_positions is None:
        open_positions = []

    alerts = []
    now_utc = datetime.now(timezone.utc)

    for pair, cfg in PAIRS.items():
        path = cfg["file"]
        if not path.exists():
            print(f"  [{pair}] Data file not found: {path}")
            continue

        try:
            raw = load_pair(path, cfg["divisor"])
            df  = add_indicators(raw)
        except Exception as e:
            print(f"  [{pair}] Error loading data: {e}")
            continue

        if len(df) < 2:
            print(f"  [{pair}] Insufficient data (need at least 2 bars)")
            continue

        signal_bar = df.iloc[-1]   # last completed H4 bar
        prev_bar   = df.iloc[-2]

        correlated = any(p != pair and p in open_positions for p in PAIRS)

        direction = None
        if _is_long(signal_bar, prev_bar["macd_hist"]):
            direction = 1
        elif _is_short(signal_bar, prev_bar["macd_hist"]):
            direction = -1

        if direction is not None:
            plan = _trade_plan(signal_bar, direction, equity, correlated)
            plan["pair"]       = pair
            plan["checked_at"] = now_utc.strftime("%Y-%m-%d %H:%M UTC")
            alerts.append(plan)

    return alerts


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_signal(alert: dict):
    div = "=" * 52
    print(f"\n{div}")
    print(f"  🔔  SIGNAL: {alert['pair']}  {alert['direction']}")
    print(f"{div}")
    print(f"  Signal bar : {alert['signal_bar']}")
    print(f"  Checked at : {alert['checked_at']}")
    print(f"  ── Entry ──────────────────────────────────────")
    print(f"  Entry (approx) : {alert['entry_approx']}  (next bar open)")
    print(f"  Stop loss      : {alert['stop_loss']}  ({alert['stop_dist_pips']} pips)")
    print(f"  TP1 (1R, 50%)  : {alert['tp1']}")
    print(f"  TP2 (3R, 50%)  : {alert['tp2']}")
    print(f"  ── Sizing ─────────────────────────────────────")
    print(f"  Risk           : ${alert['risk_usd']:.2f}"
          + ("  [corr-adjusted 50%]" if alert["corr_adjusted"] else ""))
    print(f"  Position size  : {alert['size_units']:,.0f} units")
    print(f"  ── Indicators ─────────────────────────────────")
    print(f"  ADX : {alert['adx']}   RSI : {alert['rsi']}   ATR : {alert['atr']}")
    print(f"{div}\n")


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nForex Signal Scanner  —  v4.4")
    print(f"Running at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    signals = check_signals()

    if not signals:
        print("  No signals on current bar.\n")
    else:
        for s in signals:
            print_signal(s)
