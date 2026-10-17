"""Mean-reversion family: R1 Bollinger + RSI, R2 z-score."""
from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from .base import Signals, mid_frame


def bb_rsi(df: pd.DataFrame, bb_n: int = 20, bb_k: float = 2.0, rsi_n: int = 14,
           rsi_low: float = 30.0, rsi_high: float = 70.0, atr_n: int = 14,
           stop_atr: float = 2.0) -> Signals:
    """R1: long when the close is below the lower band and RSI < 30, short when
    above the upper band and RSI > 70. Exit at the middle band: a limit at the
    SMA(20) value of bar t, live during bar t+1. Stop 2 x ATR(14)."""
    df = mid_frame(df)
    bb = ind.bollinger(df.close, bb_n, bb_k)
    r = ind.rsi(df.close, rsi_n)
    long_ = (df.close < bb.lower) & (r < rsi_low)
    short = (df.close > bb.upper) & (r > rsi_high)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=long_.to_numpy(), short_mkt=short.to_numpy(),
                   exit_long_limit=bb.mid.to_numpy(), exit_short_limit=bb.mid.to_numpy(),
                   stop_long_dist=stop, stop_short_dist=stop)


def zscore_revert(df: pd.DataFrame, n: int = 20, entry_z: float = 2.0, atr_n: int = 14,
                  stop_atr: float = 2.0) -> Signals:
    """R2: z = (close - SMA(20)) / stdev(20). Long when z < -2, short when
    z > +2, exit at the next open once z has crossed back to 0. Stop 2 x ATR(14)."""
    df = mid_frame(df)
    z = ind.zscore(df.close, n)
    stop = stop_atr * ind.atr(df, atr_n).to_numpy()
    return Signals(df.index, long_mkt=(z < -entry_z).to_numpy(),
                   short_mkt=(z > entry_z).to_numpy(),
                   exit_long_mkt=(z >= 0).to_numpy(), exit_short_mkt=(z <= 0).to_numpy(),
                   stop_long_dist=stop, stop_short_dist=stop)
