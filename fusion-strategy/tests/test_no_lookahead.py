"""PLAN.md section 3: orders at bar t may depend on bars <= t only.

For every strategy and every timeframe it runs on, at 50 random cut points k:
  1. Truncation: orders computed on bars [0, k) must equal the full-data
     orders on [0, k).
  2. Perturbation: replace bars [k, n) with unrelated prices, once 30%
     above and once 30% below the last close; orders on [0, k) must not
     change.
  3. Negative control: a deliberately leaky strategy (reads close[t+1]) must
     fail both checks, which shows the checks can catch lookahead.
"""
import unittest
import zlib

import numpy as np
import pandas as pd

from fusion import indicators as ind
from fusion.strategies import STRATEGIES, Signals
from tests.helpers import datasets, random_walk

CUTS = 50
# fields that are orders (as opposed to the stop/target attached to an entry)
ORDER_FIELDS = ("long_mkt", "short_mkt", "long_stop", "short_stop", "long_limit", "short_limit",
                "exit_long_mkt", "exit_short_mkt", "exit_long_stop", "exit_short_stop",
                "exit_long_limit", "exit_short_limit", "flat_at_close")


def frame_of(sig: Signals) -> pd.DataFrame:
    f = sig.as_frame()
    for k, v in sig.notes.items():
        f[f"note_{k}"] = v
    return f


def perturbed(df: pd.DataFrame, k: int, seed: int, level: float) -> pd.DataFrame:
    """Bars [k, n) replaced by an unrelated random walk starting at `level`
    times the last close before the cut."""
    noise = random_walk(df.index, seed=seed + 1000, price=float(df.close.iloc[k - 1]) * level)
    out = df.copy()
    cols = ["open", "high", "low", "close"]
    out.iloc[k:, [out.columns.get_loc(c) for c in cols]] = noise[cols].iloc[k:].to_numpy()
    return out


def cut_points(full: pd.DataFrame, n: int, cuts: int, rng) -> np.ndarray:
    """Half random cuts; half targeted so that truncation ends exactly on a bar
    where a leak would show: the bar where any single order field switches on,
    and the first two bars of each group (day or session). A leak that only
    affects such bars, like a range or a day's open read from the next bar,
    is invisible to random cuts most of the time."""
    lo = max(30, n // 10)
    cand = set()
    for c in ORDER_FIELDS:
        v = full[c].to_numpy()
        live = np.isfinite(v) if v.dtype.kind == "f" else v.astype(bool)
        cand.update(np.flatnonzero(live & ~np.concatenate([[False], live[:-1]])).tolist())
    g = full["group"].to_numpy()
    if (g != g[0]).any():
        first = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
        cand.update(first.tolist())
        cand.update((first + 1).tolist())
    cand = np.array(sorted(i for i in cand if lo <= i < n - 2), dtype=int)
    k = min(cuts // 2, len(cand))
    targeted = rng.choice(cand, size=k, replace=False) + 1 if k else np.array([], dtype=int)
    rand = rng.integers(lo, n - 1, size=cuts - k)
    return np.concatenate([rand, targeted.astype(int)])


def check(fn, df: pd.DataFrame, cuts: int = CUTS, seed: int = 0):
    """Return (truncation failures, perturbation failures, cuts checked)."""
    full = frame_of(fn(df))
    rng = np.random.default_rng(seed)
    ks = cut_points(full, len(df), cuts, rng)
    bad_t = bad_p = 0
    for k in ks:
        want = full.iloc[:k]
        got_t = frame_of(fn(df.iloc[:k])).iloc[:k]
        got_up = frame_of(fn(perturbed(df, k, int(k), 1.3))).iloc[:k]
        got_dn = frame_of(fn(perturbed(df, k, int(k), 0.7))).iloc[:k]
        bad_t += not want.equals(got_t)
        bad_p += not (want.equals(got_up) and want.equals(got_dn))
    return bad_t, bad_p, len(ks)


def leaky(df: pd.DataFrame) -> Signals:
    """Negative control: buys when the NEXT bar closes higher."""
    nxt = df.close.shift(-1)
    stop = 2 * ind.atr(df, 14).to_numpy()
    return Signals(df.index, long_mkt=(nxt > df.close).to_numpy(), stop_long_dist=stop)


class TestNoLookahead(unittest.TestCase):
    def test_every_strategy_on_every_timeframe(self):
        data = datasets()
        for sid, spec in STRATEGIES.items():
            for tf in spec.timeframes:
                with self.subTest(sid=sid, tf=tf):
                    bad_t, bad_p, n = check(spec.run, data[tf], seed=zlib.crc32(f"{sid}{tf}".encode()))
                    self.assertEqual(n, CUTS)
                    self.assertEqual(bad_t, 0, f"{sid} {tf}: truncation changed past orders")
                    self.assertEqual(bad_p, 0, f"{sid} {tf}: perturbation changed past orders")

    def test_negative_control_is_caught(self):
        df = datasets()["H1"]
        bad_t, bad_p, n = check(leaky, df)
        # the leak shows whenever the next close is above the current one,
        # so roughly half of the cuts must catch it
        self.assertGreaterEqual(bad_t, 0.25 * n, "truncation check missed a leaky strategy")
        self.assertGreaterEqual(bad_p, 0.25 * n, "perturbation check missed a leaky strategy")


if __name__ == "__main__":
    unittest.main()
