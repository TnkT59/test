"""End-to-end test of scripts/phase3.py on synthetic MT5 exports."""
import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.helpers import fx_hours, random_walk

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase3", ROOT / "scripts" / "phase3.py")
phase3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase3)

COSTS = """
[EURUSD]
lot = 100000
pip = 0.0001
slip = 0.0001
max_lev = 30.0
commission = "per_lot"
swap_long = -6.0
swap_short = 1.5
triple_swap_day = "Wednesday"
"""


def write_export(path: pathlib.Path, start: str, days: int, seed: int, drop_share: float = 0.0):
    df = random_walk(fx_hours(start, days), seed=seed)
    if drop_share:
        rng = np.random.default_rng(seed)
        hours = df.index.floor("h").unique()
        gone = rng.choice(hours, size=int(drop_share * len(hours)), replace=False)
        df = df[~df.index.floor("h").isin(gone)]
    server = df.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    for t, r in zip(server, df.itertuples()):
        lines.append(f"{t:%Y.%m.%d}\t{t:%H:%M:%S}\t{r.open:.5f}\t{r.high:.5f}\t{r.low:.5f}"
                     f"\t{r.close:.5f}\t10\t0\t8")
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-16"))


def run(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            phase3.main(args)
            code = 0
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue()


class TestPhase3Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        (self.d / "data").mkdir()
        (self.d / "costs.toml").write_text(COSTS)

    def tearDown(self):
        self.tmp.cleanup()

    def base(self, *extra):
        return ["--data", str(self.d / "data"), "--out", str(self.d / "out"), "--symbols", "EURUSD",
                "--costs", str(self.d / "costs.toml"), *extra]

    def test_runs_and_writes_survivor_table(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 210, seed=1)
        code, out = run(self.base("--strategies", "T1,S2,P2"))
        self.assertEqual(code, 0, out)
        for f in ("phase1_quality.md", "phase3_metrics.csv", "phase3_trades.csv.gz",
                  "phase3_pooled.csv", "survivors.md"):
            self.assertTrue((self.d / "out" / f).exists(), f)
        m = pd.read_csv(self.d / "out" / "phase3_metrics.csv")
        self.assertEqual(set(m.period), {"IS", "OOS"})
        self.assertEqual(len(m), 2 * (4 + 1 + 2))          # T1: 4 TFs, S2: 1, P2: 2
        self.assertIn("✅ Phase 3", out)
        tr = pd.read_csv(self.d / "out" / "phase3_trades.csv.gz")
        self.assertTrue((tr.t_in < pd.Timestamp("2025-01-01", tz="UTC").value).all())

    def record(self, sid="T1"):
        """Swap in a strategy that records what the runner feeds it."""
        from fusion.strategies.base import StrategySpec
        orig = phase3.STRATEGIES
        spec = orig[sid]
        seen = []

        def fn(df, **kw):
            seen.append(df.copy())
            return spec.fn(df, **kw)

        phase3.STRATEGIES = {**orig, sid: StrategySpec(spec.id, spec.family, spec.name, spec.timeframes,
                                                         spec.source, spec.deviations, spec.params, fn)}
        self.addCleanup(setattr, phase3, "STRATEGIES", orig)
        return seen

    def test_wiring_mid_bars_holdout_windows_and_gate(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 1260, seed=6)   # into 2025
        seen = self.record("T1")
        code, out = run(self.base("--strategies", "T1"))
        self.assertEqual(code, 0, out)
        m15 = [d for d in seen if (d.index[1] - d.index[0]) == pd.Timedelta(minutes=15)][0]
        self.assertLess(m15.index.max(), pd.Timestamp("2025-01-01", tz="UTC"))      # holdout cut
        from fusion.data import mt5
        raw = mt5.load_raw([self.d / "data" / "EURUSD_M15.csv"])
        bars = mt5.to_utc_bars(raw, "NY+7")
        mid = mt5.mid_bars(bars).loc[m15.index]
        self.assertTrue(np.allclose(m15.close, mid.close))                            # mid, not bid
        self.assertFalse(np.allclose(m15.close, bars.close.loc[m15.index]))
        m = pd.read_csv(self.d / "out" / "phase3_metrics.csv")
        self.assertTrue(m[m.period == "IS"].survives.isna().all())
        tr = pd.read_csv(self.d / "out" / "phase3_trades.csv.gz")
        oos = tr[tr.period == "OOS"]
        self.assertTrue((oos.t_in >= pd.Timestamp("2022-01-01", tz="UTC").value).all())
        self.assertTrue((oos.t_out <= pd.Timestamp("2025-01-01", tz="UTC").value).all())
        ins = tr[tr.period == "IS"]
        self.assertTrue((ins.t_in < pd.Timestamp("2022-01-01", tz="UTC").value).all())

    def test_m5_export_still_gives_m15_cells(self):
        df = random_walk(fx_hours("2021-09-01", 120, "5min"), seed=7)
        server = df.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
        lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
        lines += [f"{t:%Y.%m.%d}\t{t:%H:%M:%S}\t{r.open:.5f}\t{r.high:.5f}\t{r.low:.5f}\t{r.close:.5f}\t1\t0\t8"
                  for t, r in zip(server, df.itertuples())]
        (self.d / "data" / "EURUSD_M5.csv").write_text("\n".join(lines) + "\n")
        seen = self.record("S2")
        code, out = run(self.base("--strategies", "S2"))
        self.assertEqual(code, 0, out)
        self.assertEqual(seen[0].index[1] - seen[0].index[0], pd.Timedelta(minutes=15))

    def test_spread_floor_spikes_and_duplicates(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 210, seed=8)
        code, out = run(self.base("--strategies", "V2"))
        self.assertEqual(code, 0, out)
        import json
        st = json.loads((self.d / "out" / "settings.json").read_text())
        self.assertAlmostEqual(st["floors_in_pips"]["EURUSD"], 0.8)          # spread 8 points
        m = pd.read_csv(self.d / "out" / "phase3_metrics.csv").fillna({"duplicate_of": ""})
        h1 = m[(m.tf == "H1") & (m.period == "OOS")].iloc[0]
        self.assertEqual(h1.duplicate_of, "V2 M15")                          # daily levels: same trades
        code, out = run(self.base("--strategies", "V2", "--drop-spikes", "--spread-floor", "none",
                                  "--out", str(self.d / "out2")))
        self.assertEqual(code, 0, out)
        st = json.loads((self.d / "out2" / "settings.json").read_text())
        self.assertTrue(st["drop_spikes"])
        self.assertEqual(st["spread_floor"], "none")

    def test_prepare_bars_floor_uses_in_sample_only_and_drops_spikes(self):
        idx = pd.date_range("2023-06-01", periods=400, freq="15min", tz="UTC")
        close = 1.1 + np.cumsum(np.full(400, 0.00001))
        close[200] += 0.01                                       # one-bar spike that reverts
        b = pd.DataFrame({"open": close, "high": close + 0.0001, "low": close - 0.0001, "close": close,
                          "spread_px": 0.0, "spread_cons": 0.0}, index=idx)
        b.loc[idx[:100:10], "spread_px"] = 0.00004               # in-sample non-zero spreads
        b.loc[idx[300:], "spread_px"] = 0.0009                   # later, outside the window
        ins = (idx[0], idx[150])
        out, floor = phase3.prepare_bars(b, ins, "is_median", False)
        self.assertAlmostEqual(floor, 0.00004)
        self.assertTrue((out.spread_cons >= 0.00004 - 1e-15).all())
        same, _ = phase3.prepare_bars(b, ins, "none", False)
        self.assertTrue((same.spread_cons == 0).all())
        dropped, _ = phase3.prepare_bars(b, ins, "none", True)
        self.assertNotIn(idx[200], dropped.index)
        self.assertEqual(len(dropped), len(b) - 1)

    def test_stops_without_swap(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 60, seed=2)
        args = self.base("--strategies", "T1")
        args[args.index("--costs") + 1] = str(ROOT / "config" / "costs.toml")   # swaps unset
        code, out = run(args)
        self.assertEqual(code, 2)
        self.assertIn("swap not set", out)

    def test_swap_excluded_marks_overnight_cells_provisional(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 210, seed=5)
        args = self.base("--strategies", "T2,S2", "--swap-missing", "exclude")
        args[args.index("--costs") + 1] = str(ROOT / "config" / "costs.toml")   # swaps unset
        code, out = run(args)
        self.assertEqual(code, 0, out)
        m = pd.read_csv(self.d / "out" / "phase3_metrics.csv")
        self.assertFalse(m.swap_included.any())
        t2 = m[(m.strategy == "T2") & (m.tf == "H1")]
        self.assertTrue((t2.swap_nights > 0).all() and t2.provisional.all())
        s2 = m[m.strategy == "S2"]
        self.assertTrue((s2.swap_nights == 0).all() and not s2.provisional.any())
        self.assertIn("provisional", (self.d / "out" / "survivors.md").read_text())

    def test_stops_without_in_sample_data(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2022-02-01", 60, seed=3)
        code, out = run(self.base("--strategies", "T1"))
        self.assertEqual(code, 2)
        self.assertIn("no in-sample data", out)

    def test_stops_on_missing_hours(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 60, seed=4, drop_share=0.03)
        code, out = run(self.base("--strategies", "T1"))
        self.assertEqual(code, 2)
        self.assertIn("more than 1% of trading hours missing", out)


if __name__ == "__main__":
    unittest.main()
