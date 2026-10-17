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

    def test_stops_without_swap(self):
        write_export(self.d / "data" / "EURUSD_M15.csv", "2021-09-01", 60, seed=2)
        args = self.base("--strategies", "T1")
        args[args.index("--costs") + 1] = str(ROOT / "config" / "costs.toml")   # swaps unset
        code, out = run(args)
        self.assertEqual(code, 2)
        self.assertIn("swap not set", out)

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
