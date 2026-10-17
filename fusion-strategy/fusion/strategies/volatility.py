"""Volatility family: V1 TTM-style squeeze, V2 volatility breakout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ._daily import daily_context
from .base import Signals, mid_frame


def squeeze(df: pd.DataFrame, n: int = 20, bb_k: float = 2.0, kc_mult: float = 1.5,
            atr_n: int = 14, stop_atr: float = 2.0) -> Signals:
    """V1 (Carter, TTM Squeeze): the squeeze is on while BB(20, 2) is inside
    Keltner(20, 1.5 x ATR(20), EMA midline). It fires on the first bar after
    being on where it is off. Enter at the next open in the direction of the
    momentum histogram; exit at the next open once the histogram has the
    opposite sign. Stop 2 x ATR(14). The histogram formula is LazyBear's
    open-source version (Carter's own is not public)."""
    df = mid_frame(df)
    bb = ind.bollinger(df.close, n, bb_k)
    kc = ind.keltner(df, n, kc_mult)
    valid = (bb.lower.notna() & kc.lower.notna()).to_numpy()
    on = ((bb.lower > kc.lower) & (bb.upper < kc.upper)).to_numpy() & valid
    prev_on = np.concatenate([[False], on[:-1]])
    prev_valid = np.concatenate([[False], valid[:-1]])
    fire = prev_on & prev_valid & valid & ~on
    mom = ind.squeeze_momentum(df, n).to_numpy()
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    with np.errstate(invalid="ignore"):
        long_ = fire & (mom > 0)
        short = fire & (mom < 0)
        exit_long, exit_short = mom < 0, mom > 0
    sig = Signals(df.index, long_mkt=long_, short_mkt=short,
                  exit_long_mkt=exit_long, exit_short_mkt=exit_short,
                  stop_long_dist=stop, stop_short_dist=stop)
    sig.notes["squeeze_on"] = on
    return sig


def vol_breakout(df: pd.DataFrame, k: float = 0.5, atr_n: int = 14,
                 stop_atr: float = 2.0) -> Signals:
    """V2 (Larry Williams-style volatility breakout, k = 0.5 by convention):
    buy stop at today's open + k x the prior day's range, sell stop at today's
    open - k x that range; the first one hit wins, one trade per day. Exit at
    the day's close (17:00 New York). Runs on M15/H1. Levels come from the
    D1 trading day; the stop is 2 x the prior day's daily ATR(14).

    Orders exist from the day's first bar onward (so the earliest entry is in
    the second bar, once the day's open is known) and never on the day-end
    bar, whose orders would belong to the next day."""
    df = mid_frame(df)
    c = daily_context(df, atr_n)
    rng = (c.prev_high - c.prev_low).to_numpy()
    ok = np.isfinite(rng) & np.isfinite(c.prev_atr.to_numpy()) & ~c.day_end.to_numpy()
    long_lvl = np.where(ok, c.day_open.to_numpy() + k * rng, np.nan)
    short_lvl = np.where(ok, c.day_open.to_numpy() - k * rng, np.nan)
    stop = stop_atr * c.prev_atr.to_numpy()
    return Signals(df.index, long_stop=long_lvl, short_stop=short_lvl,
                   stop_long_dist=stop, stop_short_dist=stop,
                   flat_at_close=c.day_end.to_numpy(), group=c.day_id.to_numpy(),
                   max_entries_per_group=1,
                   order_expiry=c.day_close.to_numpy(), exit_deadline=c.day_close.to_numpy())
