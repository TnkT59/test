"""Technical indicators. Every value at bar t uses bars <= t only.

Smoothing conventions:
  * Wilder smoothing (ATR, RSI, ADX): x_t = (x_{t-1} * (n - 1) + v_t) / n,
    seeded with the simple average of the first n values. This is the Turtle
    "N" formula, (19 * PDN + TR) / 20, for n = 20.
  * EMA: alpha = 2 / (n + 1), seeded with the simple average of the first n
    values.
  * Standard deviation for Bollinger Bands and z-scores: population (ddof=0),
    as Bollinger specifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def _seeded_recursive(x: pd.Series, n: int, alpha: float) -> pd.Series:
    """Recursive smoothing seeded with the SMA of the first n valid values."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    # seed at the end of the first run of n consecutive valid values
    run, seed_end = 0, -1
    for i, val in enumerate(v):
        run = 0 if np.isnan(val) else run + 1
        if run == n:
            seed_end = i
            break
    if seed_end < 0:
        return pd.Series(out, index=x.index)
    out[seed_end] = v[seed_end - n + 1:seed_end + 1].mean()
    prev = out[seed_end]
    for i in range(seed_end + 1, len(v)):
        if np.isnan(v[i]):
            out[i] = prev
            continue
        prev = prev + alpha * (v[i] - prev)
        out[i] = prev
    return pd.Series(out, index=x.index)


def ema(x: pd.Series, n: int) -> pd.Series:
    return _seeded_recursive(x, n, 2.0 / (n + 1))


def wilder(x: pd.Series, n: int) -> pd.Series:
    return _seeded_recursive(x, n, 1.0 / n)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev_close).abs(),
                    (df.low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df.high.iloc[0] - df.low.iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR."""
    return wilder(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. 100 when there were gains and no losses; undefined (NaN)
    when price did not move at all."""
    d = close.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder((-d).clip(lower=0), n)
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's Directional Movement: +DI, -DI and ADX."""
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr = true_range(df)
    tr.iloc[0] = np.nan
    tr_s = wilder(tr, n)
    plus_di = 100 * wilder(plus_dm, n) / tr_s
    minus_di = 100 * wilder(minus_dm, n) / tr_s
    total = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / total.where(total > 0)
    adx = wilder(dx, n)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df.close, n)
    a = atr(df, n)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    mean = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - mean) / sd.where(sd > 0)


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    """Highest high of the last n bars including the current one."""
    return high.rolling(n, min_periods=n).max()


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    return low.rolling(n, min_periods=n).min()


def linreg_value(x: pd.Series, n: int) -> pd.Series:
    """Value at the latest bar of the least-squares line through the last n
    values (Pine Script's linreg(x, n, 0))."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    t = np.arange(n, dtype=float)
    tm = t.mean()
    denom = ((t - tm) ** 2).sum()
    for i in range(n - 1, len(v)):
        w = v[i - n + 1:i + 1]
        if np.isnan(w).any():
            continue
        slope = ((t - tm) * (w - w.mean())).sum() / denom
        out[i] = w.mean() + slope * (n - 1 - tm)
    return pd.Series(out, index=x.index)


def squeeze_momentum(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Momentum histogram of the TTM Squeeze, as in LazyBear's open-source
    "Squeeze Momentum Indicator": linreg(close - avg(avg(highest(high, n),
    lowest(low, n)), sma(close, n)), n, 0). Carter's own formula is not public."""
    hh = df.high.rolling(n, min_periods=n).max()
    ll = df.low.rolling(n, min_periods=n).min()
    basis = ((hh + ll) / 2 + sma(df.close, n)) / 2
    return linreg_value(df.close - basis, n)
