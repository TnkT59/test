"""Common order interface for every strategy in the library.

A strategy reads closed bars only and returns, for every bar t, the orders
decided at the close of bar t. Those orders are live only during bar t+1
(entries at its open, or stop/limit levels inside it). Nothing at index t may
depend on bars after t; tests/test_no_lookahead.py checks this.

Prices are levels in the same price series the strategy was given (mid by
default). The Phase 3 engine turns them into bid/ask fills with costs.

Engine contract (Phase 3). Bar t+1 means the next bar that actually exists,
which after a gap (weekend, holiday, early close) can open much later:
  * Expiry: orders from bar t are honoured in bar t+1 only if t+1 opens before
    `order_expiry[t]` (UTC ns). Session and day strategies set it to the end
    of their entry window or trading day; others leave it unlimited.
  * Deadline: a position opened from bar t's orders must be flat by
    `exit_deadline[t]` (UTC ns). It is closed at the close of the bar whose
    close reaches the deadline (`flat_at_close`), or, if that bar is missing,
    at the open of the first bar that opens at or after the deadline.
  * At the open of bar t+1, in this order: flat/exit market orders from t,
    then market entries from t. An exit and an opposite entry on the same
    open is a reversal.
  * Inside bar t+1: for an open position, its protective stop, target and exit
    stop/limit levels are resolved first. A new entry from a stop or limit
    level can then fill in the same bar only if its level lies beyond the exit
    price in the direction price was travelling. If several outcomes are
    possible in one bar, the worst one for the trade is assumed.
  * Fills: an entry whose fill price is already beyond its own stop is
    cancelled; one whose fill is already beyond its target gets no target.
  * At most one position per strategy and symbol. An entry signal in the
    direction already held is ignored; one in the opposite direction while a
    position is still open (no exit fired) is ignored.
  * When `max_entries_per_group` is set, the engine allows at most that many
    entries per `group` id (per side when `per_side` is True).

Hard rule (PLAN.md): every entry must carry a protective stop, either an
absolute level (`stop_*_px`) or a distance from the fill (`stop_*_dist`).
`Signals.validate()` enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

FLOAT_FIELDS = (
    "long_stop", "short_stop", "long_limit", "short_limit",
    "stop_long_px", "stop_short_px", "stop_long_dist", "stop_short_dist",
    "target_long_px", "target_short_px",
    "exit_long_stop", "exit_short_stop", "exit_long_limit", "exit_short_limit",
)
BOOL_FIELDS = ("long_mkt", "short_mkt", "exit_long_mkt", "exit_short_mkt", "flat_at_close")
TIME_FIELDS = ("order_expiry", "exit_deadline")
NO_LIMIT = np.iinfo(np.int64).max


@dataclass
class Signals:
    index: pd.DatetimeIndex
    # entries for bar t+1
    long_mkt: np.ndarray = None       # buy at next open
    short_mkt: np.ndarray = None      # sell at next open
    long_stop: np.ndarray = None      # buy-stop level during next bar (nan = none)
    short_stop: np.ndarray = None     # sell-stop level
    long_limit: np.ndarray = None     # buy-limit level
    short_limit: np.ndarray = None    # sell-limit level
    # protective stop and target attached to an entry made from bar t's orders
    stop_long_px: np.ndarray = None
    stop_short_px: np.ndarray = None
    stop_long_dist: np.ndarray = None   # distance below the long fill
    stop_short_dist: np.ndarray = None  # distance above the short fill
    target_long_px: np.ndarray = None
    target_short_px: np.ndarray = None
    # exits for an open position, decided at the close of bar t
    exit_long_mkt: np.ndarray = None
    exit_short_mkt: np.ndarray = None
    exit_long_stop: np.ndarray = None   # sell-stop exit level (e.g. Turtle 10-day low)
    exit_short_stop: np.ndarray = None  # buy-stop exit level
    exit_long_limit: np.ndarray = None  # sell-limit exit level (e.g. middle band)
    exit_short_limit: np.ndarray = None
    flat_at_close: np.ndarray = None
    order_expiry: np.ndarray = None     # UTC ns; orders from t need t+1 to open before it
    exit_deadline: np.ndarray = None    # UTC ns; a position from t's orders is flat by then
    # engine rules
    group: np.ndarray = None
    max_entries_per_group: int | None = None
    per_side: bool = False
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.index)
        for name in FLOAT_FIELDS:
            v = getattr(self, name)
            setattr(self, name, np.full(n, np.nan) if v is None else np.asarray(v, dtype=float))
        for name in BOOL_FIELDS:
            v = getattr(self, name)
            setattr(self, name, np.zeros(n, dtype=bool) if v is None
                    else np.nan_to_num(np.asarray(v, dtype=float)).astype(bool))
        for name in TIME_FIELDS:
            v = getattr(self, name)
            setattr(self, name, np.full(n, NO_LIMIT, dtype=np.int64) if v is None
                    else np.asarray(v, dtype=np.int64))
        if self.group is None:
            self.group = np.zeros(n, dtype=np.int64)
        for name in FLOAT_FIELDS + BOOL_FIELDS + TIME_FIELDS + ("group",):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")

    # ------------------------------------------------------------------
    def entries(self, side: str) -> np.ndarray:
        if side == "long":
            return self.long_mkt | np.isfinite(self.long_stop) | np.isfinite(self.long_limit)
        return self.short_mkt | np.isfinite(self.short_stop) | np.isfinite(self.short_limit)

    def validate(self) -> None:
        """Hard rule: no entry without a protective stop."""
        for side, px, dist in (("long", self.stop_long_px, self.stop_long_dist),
                               ("short", self.stop_short_px, self.stop_short_dist)):
            e = self.entries(side)
            has_stop = np.isfinite(px) | (np.isfinite(dist) & (dist > 0))
            bad = e & ~has_stop
            if bad.any():
                first = self.index[np.flatnonzero(bad)[0]]
                raise ValueError(f"{side} entry without a stop at {first} "
                                 f"({int(bad.sum())} bars)")

    def as_frame(self) -> pd.DataFrame:
        data = {name: getattr(self, name) for name in FLOAT_FIELDS + BOOL_FIELDS + TIME_FIELDS
                + ("group",)}
        return pd.DataFrame(data, index=self.index)


@dataclass(frozen=True)
class StrategySpec:
    id: str
    family: str
    name: str
    timeframes: tuple[str, ...]
    source: str
    deviations: tuple[str, ...]
    params: dict
    fn: object  # callable(df, **params) -> Signals

    def run(self, df: pd.DataFrame, **overrides) -> Signals:
        sig = self.fn(df, **{**self.params, **overrides})
        sig.validate()
        return sig


def crossed_above(a: pd.Series, b: pd.Series) -> np.ndarray:
    """a crosses above b on bar t: a[t] > b[t] and a[t-1] <= b[t-1]."""
    prev_a, prev_b = a.shift(1), b.shift(1)
    return ((a > b) & (prev_a <= prev_b)).to_numpy()


def crossed_below(a: pd.Series, b: pd.Series) -> np.ndarray:
    prev_a, prev_b = a.shift(1), b.shift(1)
    return ((a < b) & (prev_a >= prev_b)).to_numpy()


def mid_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The OHLC columns a strategy reads. Callers pass mid prices (PLAN.md 4)."""
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex (UTC)")
    return df


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(Signals) if f.name in FLOAT_FIELDS + BOOL_FIELDS)
