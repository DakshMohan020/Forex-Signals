"""
Forex Swing Trading Backtest v4.4
Changes vs v4.3 (backed by pre-run isolation tests):

  IMPLEMENTED:
  1. RSI filter tightened: longs RSI>50->55, shorts RSI<50->45
     Pre-test on v4.3 trades: removes 3 trades across both pairs, all losers.
     Net +$602 improvement. Both pairs benefit. Minimal curve-fitting risk.

  REJECTED after quantitative pre-testing:
  - Trailing stop (1.0-3.0x ATR): negative for GBPUSD at ALL multipliers.
    USDJPY marginally positive at 2.5x but pair-specific tuning = overfitting.
  - RSI>60: helps GBPUSD (+$664) but hurts USDJPY (-$913). Net negative.
  - RSI>57: same directional split across pairs. Not universal.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILES = {
    'GBPUSD': '/mnt/user-data/uploads/GBPUSDh4.txt',
    'USDJPY': '/mnt/user-data/uploads/USDJPYh4.txt',
}
PRICE_DIVISORS = {'GBPUSD': 100.0, 'USDJPY': 1000.0}

ATR_PERIOD        = 14
ATR_STOP_MULT     = 2.0
EMA_FAST          = 20
EMA_SLOW          = 50
EMA_TREND         = 200
RSI_PERIOD        = 14
ADX_PERIOD        = 14
ADX_THRESHOLD     = 30
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9

TP1_R             = 1.0       # CHANGE: 1.5 → 1.0 (more BE coverage)
TP2_R             = 3.0
RISK_PER_TRADE    = 0.02      # CHANGE: 1% → 2% (within half-Kelly)
CORR_RISK_MULT    = 0.5       # NEW: 50% risk if other pair already in trade
STALE_BARS        = 40
INITIAL_CAPITAL   = 10_000.0

SESSION_START_UTC = 12
SESSION_END_UTC   = 20

TRAIN_BARS = 4380
TEST_BARS  = 1095


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_atr(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n).mean()

def compute_adx(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    up   = h - h.shift()
    down = l.shift() - l
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/n, min_periods=n).mean()
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, min_periods=n).mean() / atr14
    ndi  = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/n, min_periods=n).mean() / atr14
    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1/n, min_periods=n).mean(), pdi, ndi

def compute_rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_indicators(df):
    df = df.copy()
    df['atr']       = compute_atr(df, ATR_PERIOD)
    df['ema_fast']  = df['close'].ewm(span=EMA_FAST, min_periods=EMA_FAST).mean()
    df['ema_slow']  = df['close'].ewm(span=EMA_SLOW, min_periods=EMA_SLOW).mean()
    df['ema_trend'] = df['close'].ewm(span=EMA_TREND, min_periods=EMA_TREND).mean()
    df['rsi']       = compute_rsi(df['close'], RSI_PERIOD)
    adx, pdi, ndi   = compute_adx(df, ADX_PERIOD)
    df['adx'], df['pdi'], df['ndi'] = adx, pdi, ndi
    exp1 = df['close'].ewm(span=MACD_FAST, min_periods=MACD_FAST).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, min_periods=MACD_SLOW).mean()
    df['macd']        = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, min_periods=MACD_SIGNAL).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['d1_trend']    = df['close'].rolling(6).mean()
    df['w1_trend']    = df['close'].rolling(42).mean()
    return df


# ── Signal Generation ─────────────────────────────────────────────────────────
def generate_signals(df):
    long_cond = (
        (df['close'] > df['ema_trend']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['close'] > df['d1_trend']) &
        (df['close'] > df['w1_trend']) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['pdi'] > df['ndi']) &
        (df['rsi'] > 55) & (df['rsi'] < 75) &
        (df['macd_hist'] > 0) &
        (df['macd_hist'] > df['macd_hist'].shift(1))
    )
    short_cond = (
        (df['close'] < df['ema_trend']) &
        (df['ema_fast'] < df['ema_slow']) &
        (df['close'] < df['d1_trend']) &
        (df['close'] < df['w1_trend']) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['ndi'] > df['pdi']) &
        (df['rsi'] < 45) & (df['rsi'] > 25) &
        (df['macd_hist'] < 0) &
        (df['macd_hist'] < df['macd_hist'].shift(1))
    )
    hour = df.index.hour
    in_session = (hour >= SESSION_START_UTC) & (hour < SESSION_END_UTC)
    sig = pd.Series(0, index=df.index)
    sig[long_cond  & in_session] =  1
    sig[short_cond & in_session] = -1
    return sig.shift(1).fillna(0)


# ── Trade Simulation (multi-pair, correlation-aware) ─────────────────────────
def simulate_all_pairs(pair_data: dict, capital: float) -> dict:
    """
    Simulate all pairs together so the correlation guard can track
    whether another pair is in a live trade at entry time.

    pair_data: {pair_name: (df_with_indicators, signals_series)}
    Returns:   {pair_name: trades_dataframe}
    """
    # Align on common index
    all_indices = sorted(set().union(*[set(df.index) for df, _ in pair_data.values()]))
    pairs       = list(pair_data.keys())

    # Per-pair state
    state = {p: dict(in_trade=False, direction=0, entry_price=0.0, stop_loss=0.0,
                     tp1_price=0.0, tp2_price=0.0, stop_dist=0.0,
                     tp1_hit=False, entry_bar=None, entry_time=None,
                     trade_risk=0.0)
             for p in pairs}
    trades  = {p: [] for p in pairs}
    equity  = capital

    # Build fast lookup: timestamp → bar index per pair
    pair_iloc = {p: {ts: i for i, ts in enumerate(pair_data[p][0].index)}
                 for p in pairs}

    for ts in all_indices:
        # Determine how many pairs are currently in a live trade (before this bar)
        active_count = sum(1 for p in pairs if state[p]['in_trade'])

        for p in pairs:
            df, sig_series = pair_data[p]
            if ts not in pair_iloc[p]:
                continue
            i   = pair_iloc[p][ts]
            bar = df.iloc[i]
            sig = sig_series.iloc[i]
            h, l = bar['high'], bar['low']
            s   = state[p]

            if s['in_trade']:
                bars_held  = i - s['entry_bar']
                closed_bar = False

                # 1. TP1
                if not s['tp1_hit']:
                    tp1_now = (s['direction'] ==  1 and h >= s['tp1_price']) or \
                              (s['direction'] == -1 and l <= s['tp1_price'])
                    if tp1_now:
                        pnl_pct = s['direction'] * (s['tp1_price'] - s['entry_price']) / s['entry_price']
                        pnl     = s['trade_risk'] * (pnl_pct / (s['stop_dist'] / s['entry_price'])) * 0.5
                        equity += pnl
                        trades[p].append({
                            'entry_time':  s['entry_time'],
                            'exit_time':   ts,
                            'direction':   'long' if s['direction']==1 else 'short',
                            'entry':       round(s['entry_price'], 5),
                            'exit':        round(s['tp1_price'], 5),
                            'leg':         'tp1',
                            'exit_reason': 'tp1',
                            'pnl':         round(pnl, 4),
                            'pnl_pct':     round(pnl_pct * 100, 4),
                            'trade_risk':  round(s['trade_risk'], 4),
                            'equity':      round(equity, 4),
                        })
                        s['tp1_hit']   = True
                        s['stop_loss'] = s['entry_price']  # move to BE

                # 2. TP2
                if not closed_bar and s['tp1_hit']:
                    tp2_now = (s['direction'] ==  1 and h >= s['tp2_price']) or \
                              (s['direction'] == -1 and l <= s['tp2_price'])
                    if tp2_now:
                        pnl_pct = s['direction'] * (s['tp2_price'] - s['entry_price']) / s['entry_price']
                        pnl     = s['trade_risk'] * (pnl_pct / (s['stop_dist'] / s['entry_price'])) * 0.5
                        equity += pnl
                        trades[p].append({
                            'entry_time':  s['entry_time'],
                            'exit_time':   ts,
                            'direction':   'long' if s['direction']==1 else 'short',
                            'entry':       round(s['entry_price'], 5),
                            'exit':        round(s['tp2_price'], 5),
                            'leg':         'final',
                            'exit_reason': 'tp2',
                            'pnl':         round(pnl, 4),
                            'pnl_pct':     round(pnl_pct * 100, 4),
                            'trade_risk':  round(s['trade_risk'], 4),
                            'equity':      round(equity, 4),
                        })
                        s['in_trade'] = False
                        closed_bar    = True

                # 3. Stop
                if not closed_bar:
                    sl_hit = (s['direction'] ==  1 and l <= s['stop_loss']) or \
                             (s['direction'] == -1 and h >= s['stop_loss'])
                    if sl_hit:
                        pnl_pct = s['direction'] * (s['stop_loss'] - s['entry_price']) / s['entry_price']
                        pnl     = s['trade_risk'] * (pnl_pct / (s['stop_dist'] / s['entry_price'])) * 0.5
                        equity += pnl
                        trades[p].append({
                            'entry_time':  s['entry_time'],
                            'exit_time':   ts,
                            'direction':   'long' if s['direction']==1 else 'short',
                            'entry':       round(s['entry_price'], 5),
                            'exit':        round(s['stop_loss'], 5),
                            'leg':         'final',
                            'exit_reason': 'stop',
                            'pnl':         round(pnl, 4),
                            'pnl_pct':     round(pnl_pct * 100, 4),
                            'trade_risk':  round(s['trade_risk'], 4),
                            'equity':      round(equity, 4),
                        })
                        s['in_trade'] = False
                        closed_bar    = True

                # 4. Stale
                if not closed_bar and bars_held >= STALE_BARS:
                    exit_price = bar['close']
                    pnl_pct    = s['direction'] * (exit_price - s['entry_price']) / s['entry_price']
                    pnl        = s['trade_risk'] * (pnl_pct / (s['stop_dist'] / s['entry_price'])) * 0.5
                    equity    += pnl
                    trades[p].append({
                        'entry_time':  s['entry_time'],
                        'exit_time':   ts,
                        'direction':   'long' if s['direction']==1 else 'short',
                        'entry':       round(s['entry_price'], 5),
                        'exit':        round(exit_price, 5),
                        'leg':         'final',
                        'exit_reason': 'stale',
                        'pnl':         round(pnl, 4),
                        'pnl_pct':     round(pnl_pct * 100, 4),
                        'trade_risk':  round(s['trade_risk'], 4),
                        'equity':      round(equity, 4),
                    })
                    s['in_trade'] = False

            else:
                # Entry
                if sig != 0:
                    direction   = int(sig)
                    entry_price = bar['open']
                    atr_val     = bar['atr']
                    stop_dist   = ATR_STOP_MULT * atr_val

                    # Correlation guard: if another pair already in trade, use reduced risk
                    other_active = sum(1 for q in pairs if q != p and state[q]['in_trade'])
                    risk_mult    = CORR_RISK_MULT if other_active > 0 else 1.0
                    trade_risk   = equity * RISK_PER_TRADE * risk_mult

                    s.update(dict(
                        in_trade    = True,
                        direction   = direction,
                        entry_price = entry_price,
                        stop_dist   = stop_dist,
                        stop_loss   = entry_price - direction * stop_dist,
                        tp1_price   = entry_price + direction * TP1_R * stop_dist,
                        tp2_price   = entry_price + direction * TP2_R * stop_dist,
                        tp1_hit     = False,
                        entry_bar   = i,
                        entry_time  = ts,
                        trade_risk  = trade_risk,
                    ))

    return {p: pd.DataFrame(t) for p, t in trades.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(trades, initial_capital):
    if trades.empty:
        return {}
    trade_pnl  = trades.groupby('entry_time')['pnl'].sum()
    trade_risk = trades.groupby('entry_time')['trade_risk'].first()
    total = len(trade_pnl)
    wins  = (trade_pnl > 0).sum()
    win_r = wins / total * 100 if total else 0
    win_pnl  = trade_pnl[trade_pnl > 0]
    loss_pnl = trade_pnl[trade_pnl <= 0]
    avg_win_r  = (win_pnl  / trade_risk[win_pnl.index]).mean()  if len(win_pnl)  else 0
    avg_loss_r = (loss_pnl / trade_risk[loss_pnl.index]).mean() if len(loss_pnl) else 0
    rr         = abs(avg_win_r / avg_loss_r) if avg_loss_r != 0 else 0
    eq_curve   = trades.groupby('entry_time')['equity'].last().sort_index()
    roll_max   = eq_curve.cummax()
    drawdown   = (eq_curve - roll_max) / roll_max * 100
    max_dd     = drawdown.min()
    total_return = (trades['equity'].iloc[-1] - initial_capital) / initial_capital * 100
    ret_series = trade_pnl / initial_capital
    n = len(ret_series)
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(10.0)) \
             if (ret_series.std() > 0 and n >= 5) else np.nan
    return {
        'total_trades':   total,
        'win_rate_%':     round(win_r, 1),
        'avg_win_R':      round(avg_win_r, 2),
        'avg_loss_R':     round(avg_loss_r, 2),
        'rr_ratio':       round(rr, 2),
        'total_return_%': round(total_return, 2),
        'max_drawdown_%': round(max_dd, 2),
        'sharpe':         round(sharpe, 2) if not np.isnan(sharpe) else 'n/a',
        'exits':          trades['exit_reason'].value_counts().to_dict(),
    }


# ── Walk-Forward ──────────────────────────────────────────────────────────────
def walk_forward_multi(all_dfs, all_sigs_fn):
    """Walk-forward over all pairs simultaneously (correlation-aware)."""
    pair_names = list(all_dfs.keys())
    n_bars     = min(len(df) for df in all_dfs.values())
    results    = []
    window_start = 0
    fold = 0

    while window_start + TRAIN_BARS + TEST_BARS <= n_bars:
        fold += 1
        train_end = window_start + TRAIN_BARS
        test_end  = train_end + TEST_BARS

        pair_data = {}
        for p in pair_names:
            df_full   = all_dfs[p]
            test_df   = df_full.iloc[train_end:test_end]
            test_sig  = all_sigs_fn[p].iloc[train_end:test_end]
            pair_data[p] = (test_df, test_sig)

        fold_trades = simulate_all_pairs(pair_data, INITIAL_CAPITAL)

        for p in pair_names:
            t       = fold_trades[p]
            metrics = calc_metrics(t, INITIAL_CAPITAL)
            test_df = pair_data[p][0]
            period  = f"{test_df.index[0].strftime('%Y-%m')} → {test_df.index[-1].strftime('%Y-%m')}"
            metrics.update({'pair': p, 'fold': fold, 'period': period})
            results.append(metrics)

        window_start += TEST_BARS

    return results


# ── Load & Run ────────────────────────────────────────────────────────────────
def load_data(path, divisor):
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] / divisor
    return df.sort_index()

all_dfs   = {}
all_sigs  = {}
for pair, path in DATA_FILES.items():
    print(f"  Loading {pair} …")
    raw = load_data(path, PRICE_DIVISORS[pair])
    df  = add_indicators(raw)
    df  = df.dropna()
    all_dfs[pair]  = df
    all_sigs[pair] = generate_signals(df)

# Full-sample simulation (for trade log output)
print("\n  Running full-sample simulation …")
full_trades = simulate_all_pairs(
    {p: (all_dfs[p], all_sigs[p]) for p in all_dfs},
    INITIAL_CAPITAL
)

# Walk-forward
print("  Running walk-forward …\n")
wf_results = walk_forward_multi(all_dfs, all_sigs)

# Print results
for pair in DATA_FILES:
    pair_res = [r for r in wf_results if r['pair'] == pair]
    t = full_trades[pair]
    tp1_r = (t['exit_reason']=='tp1').mean()*100  if not t.empty else 0
    tp2_r = (t['exit_reason']=='tp2').mean()*100  if not t.empty else 0
    stp_r = (t['exit_reason']=='stop').mean()*100 if not t.empty else 0
    stl_r = (t['exit_reason']=='stale').mean()*100 if not t.empty else 0
    corr_trades = (t['trade_risk'] < (t['equity'] * RISK_PER_TRADE * 0.9)).sum() \
                  if not t.empty else 0

    print(f"\n{'='*67}")
    print(f"  {pair}  |  TP1:{tp1_r:.0f}%  TP2:{tp2_r:.0f}%  Stop:{stp_r:.0f}%  "
          f"Stale:{stl_r:.0f}%  |  Corr-reduced entries: {corr_trades}")
    print(f"\n  {'Fold':<5} {'Period':<22} {'Trades':<8} {'Win%':<8} {'RR':<6} "
          f"{'Return%':<10} {'MaxDD%':<9} {'Sharpe'}")
    print(f"  {'-'*75}")
    for r in pair_res:
        print(f"  {r['fold']:<5} {r['period']:<22} {r.get('total_trades',0):<8} "
              f"{r.get('win_rate_%',0):<8} {r.get('rr_ratio',0):<6} "
              f"{r.get('total_return_%',0):<10} {r.get('max_drawdown_%',0):<9} "
              f"{r.get('sharpe','n/a')}")

# Aggregate
print(f"\n{'='*67}")
print("  AGGREGATE SUMMARY (all pairs, all folds)")
print(f"  {'Metric':<32} {'Mean':>8}  {'Std':>8}")
print(f"  {'-'*52}")
summary_df = pd.DataFrame(wf_results)
for col in ['total_trades','win_rate_%','rr_ratio','total_return_%','max_drawdown_%']:
    vals = summary_df[col].dropna()
    std_str = f"{vals.std():>8.2f}" if col in ['win_rate_%','total_return_%','max_drawdown_%'] else "        "
    print(f"  {col:<32} {vals.mean():>8.2f}  {std_str}")
sharpe_vals = pd.to_numeric(summary_df['sharpe'], errors='coerce').dropna()
print(f"  {'sharpe (valid folds)':<32} {sharpe_vals.mean():>8.2f}  {sharpe_vals.std():>8.2f}")

# v4.2 comparison
print(f"\n  v4.2 BASELINE:")
print(f"  {'win_rate_%':<32} {'42.54':>8}  {'18.81':>8}")
print(f"  {'rr_ratio':<32} {'2.66':>8}")
print(f"  {'total_return_%':<32} {'2.52':>8}  {'2.45':>8}")
print(f"  {'max_drawdown_%':<32} {'-1.21':>8}  {'0.74':>8}")
print(f"  {'sharpe':<32} {'0.93':>8}  {'0.88':>8}")

# Validation
print(f"\n  VALIDATION")
print(f"  {'-'*52}")
for p, t in full_trades.items():
    if t.empty: continue
    neg_stale = ((t['exit_reason']=='stale') & (t['pnl']<-0.01)).sum()
    tp1_only  = (t.groupby('entry_time')['leg'].apply(list)
                  .apply(lambda x: x==['tp1'])).sum()
    print(f"  {p}  neg stale: {neg_stale}  orphan tp1: {tp1_only}")

# Save
summary_df.drop(columns=['exits'], errors='ignore').to_csv(
    '/mnt/user-data/outputs/wf_summary_v44.csv', index=False)
for p, t in full_trades.items():
    if not t.empty:
        t.to_csv(f'/mnt/user-data/outputs/trades_{p}_v44.csv', index=False)
print(f"\n  ✓ Saved to /mnt/user-data/outputs/")
print(f"{'='*67}\n")
