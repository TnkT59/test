"""Daily context for intraday bars, using completed trading days only.

The trading day rolls at 17:00 New York. For a bar in trading day d, the
"prior day" values come from day d-1, which closed before the bar opened.
The current day's open is the open of day d's first bar, known from that bar
onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..timeutil import NY, bar_length, closes_at_or_after, trading_day, trading_day_end


def daily_context(df: pd.DataFrame, atr_n: int = 14) -> pd.DataFrame:
    if bar_length(df.index) >= pd.Timedelta(hours=20):
        raise ValueError("daily context needs intraday bars")
    day = trading_day(df.index)
    g = df.groupby(day)
    daily = pd.DataFrame({"open": g.open.first(), "high": g.high.max(),
                          "low": g.low.min(), "close": g.close.last()})
    daily["atr"] = ind.atr(daily, atr_n)
    prev = daily.shift(1)
    ctx = pd.DataFrame({
        "day": day,
        "prev_high": prev.high.reindex(day).to_numpy(),
        "prev_low": prev.low.reindex(day).to_numpy(),
        "prev_close": prev.close.reindex(day).to_numpy(),
        "prev_atr": prev.atr.reindex(day).to_numpy(),
        "day_open": daily.open.reindex(day).to_numpy(),
    }, index=df.index)
    # day-end bar: the one whose close reaches 17:00 New York
    ctx["day_end"] = closes_at_or_after(df.index, NY, "17:00")
    ctx["day_id"] = (day.as_unit("ns").asi8 // 86_400_000_000_000).astype(np.int64)
    ctx["day_close"] = trading_day_end(df.index)
    return ctx
