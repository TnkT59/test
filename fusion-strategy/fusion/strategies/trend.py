"""Trend family: T1 MA cross, T2 Turtle System 1, T3 Turtle System 2, T4 ADX trend.

Turtle rules are from "The Original Turtle Trading Rules" (Faith et al.,
originalturtles.org, 2003), checked against the primary text:
  * N = (19 * PDN + TR) / 20, seeded with a 20-day simple average (Wilder ATR 20).
  * System 1: enter when price exceeds the 20-day high or low. Skip the signal
    if the last breakout, taken or not, "would have resulted in a winning
    trade"; "the direction of the last breakout was irrelevant". A breakout is
    a loser if price "moved 2N against the position before a profitable 10-day
    exit". When a System 1 signal is skipped, enter at the 55-day breakout
    instead (the "Failsafe Breakout").
  * System 2: 55-day breakout, no skip rule.
  * Exits: 10-day opposite extreme (System 1), 20-day (System 2). Stop 2N.
  * Pyramiding (adding units every 1/2 N) is omitted, as PLAN.md states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Signals, crossed_above, crossed_below, mid_frame


def ma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200, atr_n: int = 14,
             stop_atr: float = 2.0) -> Signals:
    """T1: long when SMA(fast) crosses above SMA(slow), short on the cross
    below, stop-and-reverse. Protective stop 2 x ATR(14) is a deviation."""
    df = mid_frame(df)
    f, s = ind.sma(df.close, fast), ind.sma(df.close, slow)
    up, dn = crossed_above(f, s), crossed_below(f, s)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=up, short_mkt=dn, exit_long_mkt=dn, exit_short_mkt=up,
                   stop_long_dist=stop, stop_short_dist=stop)


def _turtle_last_breakout_won(df: pd.DataFrame, entry_n: int, exit_n: int,
                              n_vals: np.ndarray) -> np.ndarray:
    """For each bar t: True if the most recent completed System 1 breakout (as
    a hypothetical trade, whether taken or not) was a winner, as known at the
    close of t. A hypothetical trade enters at the breakout level (or the open
    if price gapped through it), is a loser if price moves 2N against it, and
    otherwise ends at the 10-day exit; a winner if that exit is above the
    entry (long) or below it (short).

    Bar-level conventions: if both breakout levels are crossed in the same bar,
    the one nearer the bar's open is taken as first; the 2N stop may be hit on
    the breakout bar itself; on the bar a trade exits, a break of the opposite
    20-bar level (which lies beyond the 10-bar exit) starts a new one."""
    o, h, l, c = df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    hi = ind.donchian_high(df.high, entry_n).to_numpy()
    lo = ind.donchian_low(df.low, entry_n).to_numpy()
    xhi = ind.donchian_high(df.high, exit_n).to_numpy()
    xlo = ind.donchian_low(df.low, exit_n).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    won = False
    pos, entry, stop = 0, 0.0, 0.0

    def start(direction: int, level: float, j: int, n: float, after_exit: bool) -> None:
        nonlocal pos, entry, stop, won
        pos = direction
        if after_exit:
            # price came through the exit first, so the bar's other extreme
            # was before this entry and cannot be its stop
            entry = level
        else:
            entry = max(level, o[j]) if direction == 1 else min(level, o[j])
        stop = entry - 2 * n if direction == 1 else entry + 2 * n
        if after_exit:
            # only the path from the bar's far extreme to its close is known
            # to come after this entry
            hit = c[j] <= stop if pos == 1 else c[j] >= stop
        else:
            hit = l[j] <= stop if pos == 1 else h[j] >= stop
        if hit:
            won, pos = False, 0

    for j in range(1, len(df)):
        H, L, N = hi[j - 1], lo[j - 1], n_vals[j - 1]
        levels_ok = np.isfinite(H) and np.isfinite(L) and np.isfinite(N)
        if pos == 0:
            if levels_ok:
                up, dn = h[j] > H, l[j] < L
                if up and dn:
                    up = abs(o[j] - H) <= abs(o[j] - L)
                    dn = not up
                if up:
                    start(1, H, j, N, after_exit=False)
                elif dn:
                    start(-1, L, j, N, after_exit=False)
        else:
            X = xlo[j - 1] if pos == 1 else xhi[j - 1]
            was = pos
            if pos == 1:
                if l[j] <= stop:
                    won, pos = False, 0
                elif np.isfinite(X) and l[j] <= X:
                    won, pos = min(X, o[j]) > entry, 0
            else:
                if h[j] >= stop:
                    won, pos = False, 0
                elif np.isfinite(X) and h[j] >= X:
                    won, pos = max(X, o[j]) < entry, 0
            if pos == 0 and levels_ok:
                if was == 1 and l[j] < L:
                    start(-1, L, j, N, after_exit=True)
                elif was == -1 and h[j] > H:
                    start(1, H, j, N, after_exit=True)
        out[j] = won
    return out


def turtle(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10, atr_n: int = 20,
           stop_n: float = 2.0, skip_rule: bool = True, failsafe_n: int = 55) -> Signals:
    """T2 (defaults) and T3 (entry_n=55, exit_n=20, skip_rule=False).

    Orders for bar t+1, decided at the close of t: buy stop at the highest
    high of bars t-entry_n+1..t (the preceding entry_n bars as seen from t+1),
    sell stop at the lowest low; exit stops at the exit_n extremes; stop 2N
    from the fill, N = Wilder ATR(20) at t."""
    df = mid_frame(df)
    n_vals = ind.atr(df, atr_n).to_numpy()
    hi = ind.donchian_high(df.high, entry_n).to_numpy()
    lo = ind.donchian_low(df.low, entry_n).to_numpy()
    if skip_rule:
        skip = _turtle_last_breakout_won(df, entry_n, exit_n, n_vals)
        fhi = ind.donchian_high(df.high, failsafe_n).to_numpy()
        flo = ind.donchian_low(df.low, failsafe_n).to_numpy()
        hi = np.where(skip, fhi, hi)
        lo = np.where(skip, flo, lo)
    else:
        skip = np.zeros(len(df), dtype=bool)
    valid = np.isfinite(n_vals)
    hi, lo = np.where(valid, hi, np.nan), np.where(valid, lo, np.nan)
    stop = stop_n * n_vals
    sig = Signals(df.index, long_stop=hi, short_stop=lo,
                  stop_long_dist=stop, stop_short_dist=stop,
                  exit_long_stop=ind.donchian_low(df.low, exit_n).to_numpy(),
                  exit_short_stop=ind.donchian_high(df.high, exit_n).to_numpy())
    sig.notes["skip_active"] = skip
    return sig


def turtle_s2(df: pd.DataFrame, entry_n: int = 55, exit_n: int = 20, atr_n: int = 20,
              stop_n: float = 2.0) -> Signals:
    return turtle(df, entry_n=entry_n, exit_n=exit_n, atr_n=atr_n, stop_n=stop_n,
                  skip_rule=False)


def adx_trend(df: pd.DataFrame, n: int = 14, threshold: float = 25.0, atr_n: int = 14,
              stop_atr: float = 2.0) -> Signals:
    """T4: long when +DI crosses above -DI with ADX > 25, short the mirror
    image; exit on the opposite DI cross (whatever the ADX). Protective stop
    2 x ATR(14) is a deviation (Wilder's extreme-point stop is not modelled)."""
    df = mid_frame(df)
    d = ind.dmi(df, n)
    up, dn = crossed_above(d.plus_di, d.minus_di), crossed_below(d.plus_di, d.minus_di)
    strong = (d.adx > threshold).to_numpy()
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=up & strong, short_mkt=dn & strong,
                   exit_long_mkt=dn, exit_short_mkt=up,
                   stop_long_dist=stop, stop_short_dist=stop)
