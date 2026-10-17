"""Clock helpers. All bar indexes are tz-aware UTC and labelled by bar open time.

The FX trading day rolls at 17:00 America/New_York (PLAN.md section 2): a bar
belongs to trading day D if it opens in [D-1 17:00, D 17:00) New York time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
ROLL_HOUR = 17


def bar_length(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing of the index (the bar size)."""
    if len(index) < 2:
        raise ValueError("need at least two bars to infer the bar size")
    return pd.Timedelta(np.median(np.diff(index.as_unit("ns").asi8)), unit="ns")


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading day (naive date at midnight) of each bar, rolling at 17:00 NY."""
    ny = index.tz_convert(NY)
    shifted = ny + pd.Timedelta(hours=24 - ROLL_HOUR)
    return shifted.tz_localize(None).normalize()


def local_minutes(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Minutes since local midnight of each bar's open time."""
    loc = index.tz_convert(tz)
    return np.asarray(loc.hour * 60 + loc.minute)


def local_day(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Local calendar date of each bar's open time, as int64 day numbers."""
    loc = index.tz_convert(tz).tz_localize(None).normalize()
    return loc.as_unit("ns").asi8 // 86_400_000_000_000


def hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def closes_at_or_after(index: pd.DatetimeIndex, tz: str, end: str) -> np.ndarray:
    """True for the bar that contains local clock time `end` as its close, or
    straddles it: bar opens before `end` and closes at or after it. Clock-only,
    so it is known in advance and leaks nothing."""
    blen = int(bar_length(index) / pd.Timedelta(minutes=1))
    start = local_minutes(index, tz)
    e = hhmm(end)
    return (start < e) & (start + blen >= e)


def local_time_on_bar_date(index: pd.DatetimeIndex, tz: str, at: str) -> np.ndarray:
    """UTC ns of local clock time `at` on each bar's local calendar date
    (wall-clock, DST-aware)."""
    midnight = index.tz_convert(tz).tz_localize(None).normalize()
    wall = midnight + pd.Timedelta(minutes=hhmm(at))
    return wall.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward") \
        .tz_convert("UTC").as_unit("ns").asi8


def trading_day_end(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC ns of 17:00 New York at the end of each bar's trading day."""
    end = trading_day(index) + pd.Timedelta(hours=ROLL_HOUR)
    return end.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward") \
        .tz_convert("UTC").as_unit("ns").asi8
