"""Synthetic market data for tests. Not market data; results on it mean nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fx_hours(start: str, days: int, freq: str = "15min") -> pd.DatetimeIndex:
    """Bar open times (UTC) inside FX trading hours: Sunday 17:00 to Friday
    17:00 New York."""
    idx = pd.date_range(start, periods=int(days * 86400 / pd.Timedelta(freq).total_seconds()),
                        freq=freq, tz="UTC")
    ny = idx.tz_convert("America/New_York")
    wd, hr = ny.dayofweek, ny.hour
    open_mkt = (wd < 4) | ((wd == 4) & (hr < 17)) | ((wd == 6) & (hr >= 17))
    return idx[open_mkt]


def random_walk(index: pd.DatetimeIndex, seed: int = 0, price: float = 1.10,
                vol: float = 0.0006, regime_len: int = 400, drift: float = 0.00025,
                digits: int = 5) -> pd.DataFrame:
    """OHLC random walk with alternating trend regimes, so trend, breakout and
    mean-reversion rules all fire. vol is per bar."""
    rng = np.random.default_rng(seed)
    n = len(index)
    regime = rng.choice([-1.0, 0.0, 1.0], size=n // regime_len + 1)
    mu = np.repeat(regime, regime_len)[:n] * drift
    # calm and volatile phases, so volatility-compression rules can fire
    vol_mult = np.repeat(rng.choice([0.35, 1.0, 1.6], size=n // regime_len + 1), regime_len)[:n]
    steps = mu + rng.normal(0, vol, n) * vol_mult
    close = price * np.exp(np.cumsum(steps))
    opn = np.concatenate([[price], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.7, (n, 2))) * (close * vol_mult)[:, None]
    high = np.maximum(opn, close) + wick[:, 0]
    low = np.minimum(opn, close) - wick[:, 1]
    r = lambda x: np.round(x, digits)
    return pd.DataFrame({"open": r(opn), "high": r(high), "low": r(low), "close": r(close)},
                        index=index)


_CACHE = {}


def datasets() -> dict:
    """Shared synthetic test data per timeframe (built once)."""
    if not _CACHE:
        from fusion.data.resample import resample
        m15 = random_walk(fx_hours("2023-01-01", 160), seed=11)
        _CACHE.update({
            "M15": m15, "H1": resample(m15, "H1"), "H4": resample(m15, "H4"),
            "D1": resample(random_walk(fx_hours("2021-01-01", 1100, "1h"), seed=12, vol=0.0025,
                                       regime_len=1200, drift=0.0006), "D1"),
        })
    return _CACHE
