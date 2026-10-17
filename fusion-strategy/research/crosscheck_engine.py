"""Cross-check the Phase 3 engine against the independently reviewed ORB
research engine (research/orb_sessions.py) on the same real data.

Both run the New York opening range (S2 in the library; the "NewYork" session
for XAUUSD and USDJPY in the research script) on bid bars with the same
conservative spread, slippage, commission and 0.5% sizing, 2023-2024 only
(nothing from the 2025+ holdout). Known rule differences, reported separately:
  * both range sides crossed in one bar: research takes the long first, the
    engine the level nearer the bar's open;
  * a fill already beyond its stop: research keeps it, the engine cancels it;
  * an early close before 16:00 New York: research exits at the last bar's
    close, the engine at the first bar after the deadline.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import orb_sessions as orb                                   # noqa: E402
from fusion.data import mt5                                  # noqa: E402
from fusion.data.splits import cap_before_holdout            # noqa: E402
from fusion.engine.backtest import run_backtest              # noqa: E402
from fusion.engine.costs import CostSpec                     # noqa: E402
from fusion.strategies import STRATEGIES                     # noqa: E402

DATA = HERE.parent / "data" / "mt5"
END = pd.Timestamp("2025-01-01", tz="UTC")


def research_trades(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    bars = orb.to_bars(df)
    sess = orb.sessions_for(sym)[2]
    days = orb.trading_days(df, sess)
    tr, _ = orb.simulate_session(bars, sym, sess, days)
    return orb.size_and_score(tr, sym)


def engine_trades(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = orb.SPEC[sym]
    cost = CostSpec(sym, spec["lot"], spec["pip"], spec["slip"], spec["max_lev"],
                    "per_notional" if sym == "XAUUSD" else "per_lot",
                    float("nan"), float("nan"), 2, df.attrs.get("point", float("nan")))
    sig = STRATEGIES["S2"].run(df[["open", "high", "low", "close"]])
    res = run_backtest(sig, df, cost, allow_missing_swap=True)
    return res.trades


def compare(sym: str) -> dict:
    files = mt5.files_by_symbol(DATA)
    raw = {s: mt5.load_raw(f) for s, f in files.items()}
    tz, _ = mt5.resolve_server_tz(raw)
    df = cap_before_holdout(mt5.to_utc_bars(raw[sym], tz))
    r = research_trades(sym, df)
    e = engine_trades(sym, df)
    day = lambda ns: pd.to_datetime(ns, utc=True).tz_convert("America/New_York").date()
    r = r.assign(d=[day(x) for x in r.t_in]).set_index("d")
    e = e.assign(d=[day(x) for x in e.t_in]).set_index("d")
    both = r.index.intersection(e.index)
    same = [d for d in both if r.loc[d, "dir"] == e.loc[d, "dir"]
            and np.isclose(r.loc[d, "entry"], e.loc[d, "entry"])
            and np.isclose(r.loc[d, "exit"], e.loc[d, "exit"])
            and np.isclose(r.loc[d, "R"], e.loc[d, "R"], atol=1e-6)]
    diff = sorted((set(both) - set(same)) | (set(r.index) ^ set(e.index)))
    return dict(symbol=sym, research=len(r), engine=len(e), identical=len(same),
                differ=len(diff), sample=[str(d) for d in diff[:8]],
                avgR_research=r.R.mean(), avgR_engine=e.R.mean(), r=r, e=e, diff=diff)


def categorize(out: dict) -> dict:
    """Why each differing day differs."""
    r, e = out["r"], out["e"]
    cats: dict = {}
    for d in out["diff"]:
        if d not in r.index or d not in e.index:
            k = "only in " + ("research" if d in r.index else "engine")
        elif r.loc[d, "dir"] != e.loc[d, "dir"]:
            k = "direction: both range sides crossed in the entry bar"
        elif not np.isclose(r.loc[d, "entry"], e.loc[d, "entry"]):
            k = "entry price"
        elif not np.isclose(r.loc[d, "exit"], e.loc[d, "exit"]):
            k = f"exit: research {r.loc[d, 'reason']} vs engine {e.loc[d, 'reason_out']}"
        else:
            k = "same prices, size differs"
        cats[k] = cats.get(k, 0) + 1
    return cats


if __name__ == "__main__":
    for sym in ("XAUUSD", "USDJPY"):
        out = compare(sym)
        print({k: v for k, v in out.items() if k not in ("r", "e", "diff", "sample")})
        print("  differences:", categorize(out))
