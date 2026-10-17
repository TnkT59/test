"""PLAN.md section 3: orders at bar t may depend on bars <= t only.

For every strategy and every timeframe it runs on, at 50 random cut points k:
  1. Truncation: orders computed on bars [0, k) must equal the full-data
     orders on [0, k).
  2. Perturbation: replace bars [k, n) with unrelated prices; orders on
     [0, k) must not change.
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


def perturbed(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Bars [k, n) replaced by an unrelated random walk at a different level."""
    noise = random_walk(df.index, seed=seed + 1000, price=float(df.close.iloc[k - 1]) * 1.3)
    out = df.copy()
    cols = ["open", "high", "low", "close"]
    out.iloc[k:, [out.columns.get_loc(c) for c in cols]] = noise[cols].iloc[k:].to_numpy()
    return out


def cut_points(full: pd.DataFrame, n: int, cuts: int, rng) -> np.ndarray:
    """Half random cuts; half targeted right after bars where an order switches
    on (truncation then ends on that bar). A leak that only affects the bar
    where an order first goes live, such as a range that reads the next bar,
    is invisible to random cuts most of the time."""
    lo = max(30, n // 10)
    live = np.zeros(n, dtype=bool)
    for c in ORDER_FIELDS:
        v = full[c].to_numpy()
        live |= np.isfinite(v) if v.dtype.kind == "f" else v.astype(bool)
    starts = np.flatnonzero(live & ~np.concatenate([[False], live[:-1]]))
    starts = starts[(starts >= lo) & (starts < n - 2)]
    k = min(cuts // 2, len(starts))
    targeted = rng.choice(starts, size=k, replace=False) + 1 if k else np.array([], dtype=int)
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
        got_p = frame_of(fn(perturbed(df, k, int(k)))).iloc[:k]
        bad_t += not want.equals(got_t)
        bad_p += not want.equals(got_p)
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
