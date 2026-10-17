"""Strategy registry: one entry per PLAN.md section 4 rule set.

`STRATEGIES[id].run(df)` returns validated Signals (every entry has a stop).
`df` holds mid-price OHLC bars indexed by tz-aware UTC open time.
"""
from __future__ import annotations

from .base import Signals, StrategySpec
from .meanrev import bb_rsi, zscore_revert
from .momentum import macd_cross, tsmom
from .session import asian_london, ny_orb
from .structure import pivot_reversion, prior_day_breakout
from .trend import adx_trend, ma_cross, turtle, turtle_s2
from .volatility import squeeze, vol_breakout

ALL_TF = ("M15", "H1", "H4", "D1")
INTRADAY = ("M15", "H1")
ATR_STOP = "protective stop 2 x ATR(14) added (hard rule; the original has no stop)"
RANGE_RULE = "no trade on a day with fewer than 80% of the range bars (data gaps)"

_SPECS = (
    StrategySpec("T1", "trend", "MA cross 50/200", ALL_TF,
                 "Golden/death cross (convention)", (ATR_STOP,),
                 dict(fast=50, slow=200), ma_cross),
    StrategySpec("T2", "trend", "Turtle System 1", ALL_TF,
                 "The Original Turtle Trading Rules (Faith et al., 2003)",
                 ("no pyramiding (units at 1/2 N)",
                  "55-day failsafe entry when a System 1 signal is skipped: in the original "
                  "rules, missing from the PLAN.md T2 row",
                  "a losing breakout is judged by the sign of the hypothetical trade's P/L "
                  "at its 2N stop or 10-day exit",
                  "the skip tracker resolves hypothetical trades on the strategy's own bars "
                  "(stop first; nearer level first when both are crossed in one bar), so it "
                  "can disagree with the engine's finer fills"),
                 dict(entry_n=20, exit_n=10, skip_rule=True, failsafe_n=55), turtle),
    StrategySpec("T3", "trend", "Turtle System 2", ALL_TF,
                 "The Original Turtle Trading Rules (Faith et al., 2003)",
                 ("no pyramiding (units at 1/2 N)",),
                 dict(entry_n=55, exit_n=20), turtle_s2),
    StrategySpec("T4", "trend", "ADX / DI cross", ALL_TF,
                 "Wilder (1978); ADX > 25 threshold (convention)",
                 (ATR_STOP + "; Wilder's extreme-point stop not modelled",),
                 dict(n=14, threshold=25.0), adx_trend),
    StrategySpec("M1", "momentum", "Time-series momentum 12m", ("D1",),
                 "Moskowitz, Ooi & Pedersen (2012), JFE",
                 (ATR_STOP, "sized by 0.5% risk, not 40% target volatility",
                  "closed and reopened every month (one round trip a month)",
                  "if the last weekday of a month has no bar, the decision is made on the "
                  "first bar of the next month"),
                 dict(lookback_months=12), tsmom),
    StrategySpec("M2", "momentum", "MACD 12/26/9", ALL_TF, "Appel; default 12/26/9",
                 (ATR_STOP,), dict(fast=12, slow=26, signal=9), macd_cross),
    StrategySpec("R1", "meanrev", "Bollinger 20/2 + RSI 14", ALL_TF,
                 "Bollinger (20, 2); Wilder RSI (14, 30/70); combination (convention)",
                 (), dict(bb_n=20, bb_k=2.0, rsi_n=14), bb_rsi),
    StrategySpec("R2", "meanrev", "Z-score 20, |z| > 2", ALL_TF, "(convention)",
                 (), dict(n=20, entry_z=2.0), zscore_revert),
    StrategySpec("V1", "volatility", "Squeeze BB(20,2) in KC(20,1.5)", ALL_TF,
                 "Carter, Mastering the Trade (TTM Squeeze)",
                 ("momentum histogram is LazyBear's open-source formula; Carter's is not public",),
                 dict(n=20, bb_k=2.0, kc_mult=1.5), squeeze),
    StrategySpec("V2", "volatility", "Volatility breakout open +/- 0.5 x prior range", INTRADAY,
                 "Larry Williams-style; k = 0.5 (convention)",
                 ("stop 2 x prior-day daily ATR(14)",
                  "earliest entry on the day's second bar (the day's open must be known)"),
                 dict(k=0.5), vol_breakout),
    StrategySpec("S1", "session", "Asian range -> London breakout", INTRADAY, "(convention)",
                 (RANGE_RULE,), {}, asian_london),
    StrategySpec("S2", "session", "New York opening range 08:00-08:30", ("M15",),
                 "(convention)", (RANGE_RULE,), {}, ny_orb),
    StrategySpec("P1", "structure", "Floor-pivot reversion R1/S1", INTRADAY,
                 "Floor-trader pivots; trading rule (convention)",
                 ("flat at 17:00 New York (time exit not in PLAN.md)",
                  "one trade per side per day (not in PLAN.md)"), {}, pivot_reversion),
    StrategySpec("P2", "structure", "Prior-day high/low breakout", INTRADAY, "(convention)",
                 (), {}, prior_day_breakout),
)

STRATEGIES: dict[str, StrategySpec] = {s.id: s for s in _SPECS}

__all__ = ["STRATEGIES", "Signals", "StrategySpec"]
