"""Momentum family: M1 time-series momentum, M2 MACD."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..timeutil import bar_length, trading_day
from .base import Signals, crossed_above, crossed_below, mid_frame


def _rebalance_bars(days: pd.DatetimeIndex) -> np.ndarray:
    """True on the bar where a month's decision is made: the bar whose trading
    day is the last weekday of its month, or, if that day has no bar (e.g.
    Good Friday), the first bar of the next month (one day late). Uses the
    calendar and bars up to t only."""
    last_weekday = np.asarray((days + pd.offsets.BDay(1)).month != days.month)
    month = np.asarray(days.year * 12 + days.month)
    new_month = np.concatenate([[False], month[1:] != month[:-1]])
    prev_had_end = np.concatenate([[True], last_weekday[:-1]])
    return last_weekday | (new_month & ~prev_had_end)


def tsmom(df: pd.DataFrame, lookback_months: int = 12, atr_n: int = 14,
          stop_atr: float = 2.0) -> Signals:
    """M1 (Moskowitz, Ooi & Pedersen 2012): at each month end go long if the
    trailing 12-month return is positive, short if negative, and hold for one
    month. D1 bars only. Every month the position is closed and reopened at
    the next open, with a new size and a new 2 x ATR(14) stop, so each month
    is one trade (a fund would only adjust its size; closing and reopening
    costs one extra round trip a month). Protective stop 2 x ATR(14) is a
    deviation (the paper scales positions to 40% annualised volatility)."""
    df = mid_frame(df)
    if bar_length(df.index) < pd.Timedelta(hours=20):
        raise ValueError("TSMOM is defined on D1 bars only")
    days = trading_day(df.index)
    month_end = _rebalance_bars(days)
    close = df.close.to_numpy()
    # close of the last bar on or before the same date lookback_months earlier
    past_dates = days - pd.DateOffset(months=lookback_months)
    pos = np.searchsorted(days.as_unit("ns").asi8, past_dates.as_unit("ns").asi8, side="right") - 1
    past_close = np.where(pos >= 0, close[np.clip(pos, 0, None)], np.nan)
    ret = close / past_close - 1
    with np.errstate(invalid="ignore"):
        up = month_end & (ret > 0)
        dn = month_end & (ret < 0)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    # close every open position at each rebalance, then reopen per the sign
    return Signals(df.index, long_mkt=up, short_mkt=dn,
                   exit_long_mkt=month_end, exit_short_mkt=month_end,
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
