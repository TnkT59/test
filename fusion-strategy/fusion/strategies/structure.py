"""Structure family: P1 floor-pivot reversion, P2 prior-day high/low breakout.

Both use the prior completed trading day (17:00 New York roll) and are flat
at the day's close (17:00 New York), one trade per side per day. The trading
rules are conventions; the floor-pivot formulas are the standard ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._daily import daily_context
from .base import Signals, mid_frame


def pivot_reversion(df: pd.DataFrame) -> Signals:
    """P1: P = (H + L + C) / 3, R1 = 2P - L, S1 = 2P - H, R2 = P + (H - L),
    S2 = P - (H - L) from the prior day. Sell limit at R1 (stop R2, target P),
    buy limit at S1 (stop S2, target P). Exit at the day's close: PLAN.md did
    not state a time exit; holding past the day the pivots belong to is not
    part of the idea, so it is added and flagged as a convention."""
    df = mid_frame(df)
    c = daily_context(df)
    H, L, C = c.prev_high.to_numpy(), c.prev_low.to_numpy(), c.prev_close.to_numpy()
    P = (H + L + C) / 3
    R1, S1 = 2 * P - L, 2 * P - H
    R2, S2 = P + (H - L), P - (H - L)
    ok = np.isfinite(P) & ~c.day_end.to_numpy() & (H > L)
    f = lambda x: np.where(ok, x, np.nan)
    return Signals(df.index, short_limit=f(R1), long_limit=f(S1),
                   stop_short_px=f(R2), stop_long_px=f(S2),
                   target_short_px=f(P), target_long_px=f(P),
                   flat_at_close=c.day_end.to_numpy(), group=c.day_id.to_numpy(),
                   max_entries_per_group=1, per_side=True,
                   order_expiry=c.day_close.to_numpy(), exit_deadline=c.day_close.to_numpy())


def prior_day_breakout(df: pd.DataFrame) -> Signals:
    """P2: buy stop at the prior-day high, sell stop at the prior-day low, one
    trade per side per day, stop at the prior-day midpoint, exit at 17:00 NY."""
    df = mid_frame(df)
    c = daily_context(df)
    H, L = c.prev_high.to_numpy(), c.prev_low.to_numpy()
    ok = np.isfinite(H) & np.isfinite(L) & (H > L) & ~c.day_end.to_numpy()
    mid = (H + L) / 2
    f = lambda x: np.where(ok, x, np.nan)
    return Signals(df.index, long_stop=f(H), short_stop=f(L),
                   stop_long_px=f(mid), stop_short_px=f(mid),
                   flat_at_close=c.day_end.to_numpy(), group=c.day_id.to_numpy(),
                   max_entries_per_group=1, per_side=True,
                   order_expiry=c.day_close.to_numpy(), exit_deadline=c.day_close.to_numpy())
