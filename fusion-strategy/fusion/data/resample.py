"""Resample intraday bars (PLAN.md section 2).

Bars are labelled by open time (left-labelled, left-closed). M15 and H1 use
plain UTC clock buckets. H4 and D1 are anchored to the 17:00 New York roll,
DST-aware: D1 is one bar per trading day (five per week, no Sunday stub) and
H4 buckets start at 17:00, 21:00, 01:00, 05:00, 09:00 and 13:00 New York.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import NY, ROLL_HOUR, trading_day

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def _agg(df: pd.DataFrame, key) -> pd.DataFrame:
    spec = {c: f for c, f in AGG.items()}
    for extra, f in (("spread", "max"), ("tick_volume", "sum")):
        if extra in df.columns:
            spec[extra] = f
    return df.groupby(key).agg(spec)


def _roll_start(day: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC start of each trading day: the previous calendar day at 17:00 NY."""
    start = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=ROLL_HOUR))
    return start.tz_localize(NY, ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    tf = tf.upper()
    if tf in ("M15", "H1"):
        rule = {"M15": "15min", "H1": "1h"}[tf]
        out = _agg(df, pd.Grouper(freq=rule, label="left", closed="left"))
        return out.dropna(subset=["open"])
    day = trading_day(df.index)
    if tf == "D1":
        out = _agg(df, day)
        out.index = _roll_start(pd.DatetimeIndex(out.index))
        return out
    if tf == "H4":
        ny = df.index.tz_convert(NY)
        mins_since_roll = ((ny.hour * 60 + ny.minute) - ROLL_HOUR * 60) % 1440
        bucket = np.asarray(mins_since_roll // 240)
        out = _agg(df, [day.rename("day"), pd.Index(bucket, name="bucket")])
        days = pd.DatetimeIndex(out.index.get_level_values(0))
        k = out.index.get_level_values(1).to_numpy()
        start_ny = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=ROLL_HOUR)
                    + pd.to_timedelta(k * 4, unit="h"))
        out.index = start_ny.tz_localize(NY, ambiguous="raise",
                                         nonexistent="shift_forward").tz_convert("UTC")
        return out
    raise ValueError(f"unknown timeframe {tf}")
