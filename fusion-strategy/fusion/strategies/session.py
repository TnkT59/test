"""Session family: S1 Asian range -> London breakout, S2 New York opening range.

Both are the same pattern: a range from fixed local clock times, then stop
orders at its high and low inside an entry window, one trade per day, stop at
the opposite side, target 1 x range, flat at the session exit. Times are
local and DST-aware. These rule sets are conventions (no canonical source).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import (bar_length, closes_at_or_after, hhmm, local_day, local_minutes,
                        local_time_on_bar_date)
from .base import Signals, mid_frame


def session_breakout(df: pd.DataFrame, tz: str, range_start: str, range_end: str,
                     entry_end: str, exit: str, target_mult: float = 1.0,
                     min_range_share: float = 0.8) -> Signals:
    """Orders decided at the close of bar t are meant for bar t+1 when, with
    no gap, t+1 would open inside [range_end, entry_end) local time; they
    expire at entry_end, so after a gap they are not honoured. The range uses
    bars that open inside [range_start, range_end) on the same local date; all
    have closed by then. Days with fewer than min_range_share of the expected
    range bars get no orders. Positions must be flat by the session exit."""
    df = mid_frame(df)
    blen = int(bar_length(df.index) / pd.Timedelta(minutes=1))
    r0, r1, e1 = hhmm(range_start), hhmm(range_end), hhmm(entry_end)
    if (r1 - r0) % blen:
        raise ValueError(f"range {range_start}-{range_end} is not a multiple of the bar size")
    mins = local_minutes(df.index, tz)
    day = local_day(df.index, tz)
    in_range = (mins >= r0) & (mins < r1)
    frame = pd.DataFrame({"day": day, "h": np.where(in_range, df.high, np.nan),
                          "l": np.where(in_range, df.low, np.nan), "n": in_range})
    agg = frame.groupby("day").agg(h=("h", "max"), l=("l", "min"), n=("n", "sum"))
    need = int(np.ceil(min_range_share * (r1 - r0) / blen))
    agg.loc[agg.n < need, ["h", "l"]] = np.nan
    hi = agg.h.reindex(day).to_numpy()
    lo = agg.l.reindex(day).to_numpy()
    nxt = mins + blen                     # open time of the next bar if contiguous
    live = (nxt >= r1) & (nxt < e1) & np.isfinite(hi) & np.isfinite(lo) & (hi > lo)
    rng = hi - lo
    return Signals(df.index,
                   long_stop=np.where(live, hi, np.nan), short_stop=np.where(live, lo, np.nan),
                   stop_long_px=np.where(live, lo, np.nan), stop_short_px=np.where(live, hi, np.nan),
                   target_long_px=np.where(live, hi + target_mult * rng, np.nan),
                   target_short_px=np.where(live, lo - target_mult * rng, np.nan),
                   flat_at_close=closes_at_or_after(df.index, tz, exit),
                   order_expiry=local_time_on_bar_date(df.index, tz, entry_end),
                   exit_deadline=local_time_on_bar_date(df.index, tz, exit),
                   group=day, max_entries_per_group=1)


def asian_london(df: pd.DataFrame) -> Signals:
    """S1: Asian range 00:00-07:00 London; entries 07:00-12:00; exit 16:00."""
    return session_breakout(df, "Europe/London", "00:00", "07:00", "12:00", "16:00")


def ny_orb(df: pd.DataFrame) -> Signals:
    """S2: range 08:00-08:30 New York; entries 08:30-11:00; exit 16:00."""
    return session_breakout(df, "America/New_York", "08:00", "08:30", "11:00", "16:00")
