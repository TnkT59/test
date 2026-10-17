"""Momentum family: M1 time-series momentum, M2 MACD."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..timeutil import bar_length, trading_day
from .base import Signals, crossed_above, crossed_below, mid_frame


def _is_month_end(days: pd.DatetimeIndex) -> np.ndarray:
    """True on a bar whose trading day is the last weekday of its month.
    Calendar only (holidays ignored), so it does not depend on later bars."""
    nxt = days + pd.offsets.BDay(1)
    return np.asarray(nxt.month != days.month)


def tsmom(df: pd.DataFrame, lookback_months: int = 12, atr_n: int = 14,
          stop_atr: float = 2.0) -> Signals:
    """M1 (Moskowitz, Ooi & Pedersen 2012): at each month end go long if the
    trailing 12-month return is positive, short if negative, and hold for the
    month. D1 bars only. A position that keeps its sign is held, not closed
    and reopened. Protective stop 2 x ATR(14) is a deviation (the paper
    scales positions to 40% annualised volatility instead)."""
    df = mid_frame(df)
    if bar_length(df.index) < pd.Timedelta(hours=20):
        raise ValueError("TSMOM is defined on D1 bars only")
    days = trading_day(df.index)
    month_end = _is_month_end(days)
    close = df.close.to_numpy()
    # close of the last bar on or before the same date lookback_months earlier
    past_dates = days - pd.DateOffset(months=lookback_months)
    pos = np.searchsorted(days.as_unit("ns").asi8, past_dates.as_unit("ns").asi8, side="right") - 1
    past_close = np.where(pos >= 0, close[np.clip(pos, 0, None)], np.nan)
    ret = close / past_close - 1
    up = month_end & (ret > 0)
    dn = month_end & (ret < 0)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=up, short_mkt=dn, exit_long_mkt=dn, exit_short_mkt=up,
                   stop_long_dist=stop, stop_short_dist=stop)


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
               atr_n: int = 14, stop_atr: float = 2.0) -> Signals:
    """M2 (Appel, 12/26/9): long when MACD crosses above its signal line, short
    on the cross below, stop-and-reverse. Protective stop 2 x ATR(14) is a
    deviation."""
    df = mid_frame(df)
    m = ind.macd(df.close, fast, slow, signal)
    up, dn = crossed_above(m.macd, m.signal), crossed_below(m.macd, m.signal)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=up, short_mkt=dn, exit_long_mkt=dn, exit_short_mkt=up,
                   stop_long_dist=stop, stop_short_dist=stop)
