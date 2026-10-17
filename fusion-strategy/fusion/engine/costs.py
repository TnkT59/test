"""Per-symbol cost model, read from config/costs.toml."""
from __future__ import annotations

import math
import pathlib
import tomllib
from dataclasses import dataclass

CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config" / "costs.toml"
COMMISSION_PER_LOT_RT = 7.0
COMMISSION_PER_100K_NOTIONAL = 7.0
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class CostSpec:
    symbol: str
    lot: float
    pip: float
    slip: float
    max_lev: float
    commission: str
    swap_long: float
    swap_short: float
    triple_swap_day: int      # 0 = Monday
    point: float = float("nan")

    @property
    def usd_quote(self) -> bool:
        """P/L arrives in USD (EURUSD, GBPUSD, XAUUSD); False for USDJPY."""
        return not self.symbol.startswith("USD")

    def to_usd(self, amount_quote: float, price: float) -> float:
        return amount_quote if self.usd_quote else amount_quote / price

    def notional_usd(self, units: float, price: float) -> float:
        return units * price if self.usd_quote else units

    def commission_usd(self, units: float, price: float) -> float:
        if self.commission == "per_lot":
            return COMMISSION_PER_LOT_RT * units / self.lot
        if self.commission == "per_notional":
            return COMMISSION_PER_100K_NOTIONAL * self.notional_usd(units, price) / 100_000
        raise ValueError(f"unknown commission model {self.commission!r}")

    def swap_known(self) -> bool:
        return not (math.isnan(self.swap_long) or math.isnan(self.swap_short)
                    or math.isnan(self.point))

    def swap_usd(self, direction: int, units: float, nights: int, price: float) -> float:
        """Swap for `nights` charged rolls (triple days already counted three
        times); negative = cost. Swap is in points per lot per night."""
        pts = self.swap_long if direction == 1 else self.swap_short
        quote = pts * self.point * units * nights
        return self.to_usd(quote, price)


def load_costs(path: pathlib.Path = CONFIG, point: dict[str, float] | None = None) -> dict[str, CostSpec]:
    raw = tomllib.loads(path.read_text())
    out = {}
    for sym, v in raw.items():
        out[sym] = CostSpec(symbol=sym, lot=float(v["lot"]), pip=float(v["pip"]),
                            slip=float(v["slip"]), max_lev=float(v["max_lev"]),
                            commission=v["commission"], swap_long=float(v["swap_long"]),
                            swap_short=float(v["swap_short"]),
                            triple_swap_day=WEEKDAYS.index(v["triple_swap_day"]),
                            point=(point or {}).get(sym, float("nan")))
    return out
