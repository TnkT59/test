"""Tests for orb_sessions.py. Run: python -m unittest test_orb_sessions -v"""
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import orb_sessions as orb


def random_walk_utc(start="2023-01-02", weeks=30, seed=0, price=1.1, vol=0.00008,
                    spread_pts=8, digits=5):
    """M1 bid bars Sun 17:00 -> Fri 17:00 New York, indexed in UTC."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=weeks * 7 * 1440, freq="min", tz="UTC")
    ny = idx.tz_convert("America/New_York")
    wd, hr = ny.dayofweek, ny.hour
    open_mkt = ((wd < 4) | ((wd == 4) & (hr < 17)) | ((wd == 6) & (hr >= 17)))
    idx = idx[open_mkt]
    steps = rng.normal(0, vol, (len(idx), 4))
    close = price + np.cumsum(steps[:, 0])
    opn = np.concatenate([[price], close[:-1]])
    hi = np.maximum(opn, close) + np.abs(steps[:, 1])
    lo = np.minimum(opn, close) - np.abs(steps[:, 2])
    r = lambda x: np.round(x, digits)
    return pd.DataFrame({"open": r(opn), "high": r(hi), "low": r(lo), "close": r(close),
                         "spread": spread_pts}, index=idx)


def to_server(df_utc, scheme):
    if scheme == "NY+7":
        loc = df_utc.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    else:
        loc = df_utc.index.tz_convert(scheme).tz_localize(None)
    out = df_utc.copy()
    out.index = loc
    return out


def write_mt5(df_server, path, utf16=True):
    lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    for t, r in df_server.iterrows():
        lines.append(f"{t:%Y.%m.%d}\t{t:%H:%M:%S}\t{r.open:.5f}\t{r.high:.5f}\t{r.low:.5f}"
                     f"\t{r.close:.5f}\t10\t0\t{int(r.spread)}")
    text = "\r\n".join(lines) + "\r\n"
    path.write_bytes(text.encode("utf-16") if utf16 else text.encode("utf-8"))


def bars_from(df_utc, point=1e-5):
    d = df_utc.copy()
    d["spread_px"] = d.spread * point
    return orb.to_bars(d)


LONDON = orb.SESSIONS[1]


def london_day(day="2024-03-05", base=1.1000, rng_half=0.0010, spread_pts=10):
    """Flat London morning: range 08:00-08:30 is base +/- rng_half."""
    t0 = pd.Timestamp(f"{day} 07:00", tz="Europe/London")
    idx = pd.date_range(t0, pd.Timestamp(f"{day} 17:00", tz="Europe/London"),
                        freq="min", inclusive="left").tz_convert("UTC")
    df = pd.DataFrame({"open": base, "high": base + 0.0001, "low": base - 0.0001,
                       "close": base, "spread": spread_pts}, index=idx)
    local = idx.tz_convert("Europe/London")
    in_rng = (local.hour == 8) & (local.minute < 30)
    df.loc[in_rng & (local.minute == 10), "high"] = base + rng_half
    df.loc[in_rng & (local.minute == 20), "low"] = base - rng_half
    return df, local


def at(local, hh, mm):
    return (local.hour == hh) & (local.minute == mm)


class TestLoader(unittest.TestCase):
    def test_reads_utf16_tab_export_and_infers_point(self):
        df = random_walk_utc(weeks=1).iloc[:500]
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "EURUSD_M1_2023.csv"
            write_mt5(to_server(df, "NY+7"), p, utf16=True)
            got = orb.read_mt5_csv(p)
        self.assertEqual(len(got), 500)
        self.assertEqual(list(got.columns), ["open", "high", "low", "close", "spread"])
        self.assertAlmostEqual(orb.infer_point(got.close.to_numpy()), 1e-5)
        self.assertEqual(orb.symbol_of(p), "EURUSD")
        self.assertEqual(orb.symbol_of(pathlib.Path("GOLD.m_2024.csv")), "XAUUSD")


class TestServerTimezone(unittest.TestCase):
    def test_detects_ny_plus_7(self):
        df = random_walk_utc(start="2023-01-01", weeks=60)
        tz, score = orb.detect_server_tz(to_server(df, "NY+7").index)
        self.assertEqual(tz, "NY+7")
        self.assertGreater(score, 0.95)

    def test_detects_fixed_utc_plus_2(self):
        df = random_walk_utc(start="2023-01-01", weeks=60)
        tz, score = orb.detect_server_tz(to_server(df, "Etc/GMT-2").index)
        self.assertEqual(tz, "Etc/GMT-2")

    def test_round_trip_to_utc(self):
        df = random_walk_utc(start="2023-03-01", weeks=10)
        back = orb.server_to_utc(to_server(df, "NY+7").index, "NY+7")
        self.assertTrue((back == df.index).all())


class TestTradeRules(unittest.TestCase):
    def test_long_breakout_hits_target(self):
        df, local = london_day()
        df.loc[at(local, 8, 40), "high"] = 1.1012          # ask = 1.1012 + 0.0001 >= 1.1010
        df.loc[at(local, 9, 0), "high"] = 1.1031            # target = 1.1010 + 0.0020
        trades, skipped = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                               [pd.Timestamp("2024-03-05").date()])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["dir"], 1)
        self.assertAlmostEqual(t["entry"], 1.1010 + 0.0001)    # level + slippage (open+spread < level)
        self.assertEqual(t["reason"], "target")
        self.assertAlmostEqual(t["exit"], 1.1030 - 0.0001)
        self.assertAlmostEqual(t["stop"], 1.0990)

    def test_both_sides_in_one_bar_is_a_loss(self):
        df, local = london_day()
        df.loc[at(local, 8, 40), ["high", "low"]] = [1.1015, 1.0985]
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                         [pd.Timestamp("2024-03-05").date()])
        self.assertEqual(trades[0]["reason"], "stop")
        self.assertLess(trades[0]["exit"], trades[0]["entry"])

    def test_target_not_allowed_on_entry_bar(self):
        df, local = london_day()
        df.loc[at(local, 8, 40), "high"] = 1.1040
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                         [pd.Timestamp("2024-03-05").date()])
        self.assertNotEqual(trades[0]["reason"], "target")
        self.assertNotEqual(trades[0]["t_out"], trades[0]["t_in"])

    def test_short_breakout_then_time_exit(self):
        df, local = london_day()
        df.loc[at(local, 9, 0), "low"] = 1.0988
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                         [pd.Timestamp("2024-03-05").date()])
        t = trades[0]
        self.assertEqual((t["dir"], t["reason"]), (-1, "time"))
        self.assertAlmostEqual(t["entry"], 1.0990 - 0.0001)
        self.assertAlmostEqual(t["exit"], 1.1000 + 0.0001 + 0.0001)   # close + spread + slip
        exit_local = pd.Timestamp(t["t_out"], tz="UTC").tz_convert("Europe/London")
        self.assertEqual((exit_local.hour, exit_local.minute), (15, 59))

    def test_no_entry_after_entry_window(self):
        df, local = london_day()
        df.loc[at(local, 11, 5), "high"] = 1.1020
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                         [pd.Timestamp("2024-03-05").date()])
        self.assertEqual(trades, [])

    def test_missing_range_bars_skips_day(self):
        df, local = london_day()
        df = df[~((local.hour == 8) & (local.minute < 10))]
        trades, skipped = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                               [pd.Timestamp("2024-03-05").date()])
        self.assertEqual((trades, skipped), ([], 1))

    def test_sizing_and_costs(self):
        df, local = london_day()
        df.loc[at(local, 8, 40), "high"] = 1.1012
        df.loc[at(local, 9, 0), "high"] = 1.1031
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON,
                                         [pd.Timestamp("2024-03-05").date()])
        s = orb.size_and_score(trades, "EURUSD").iloc[0]
        dist = 1.1011 - 1.0990
        self.assertAlmostEqual(s.risk_usd, s.units * dist, places=6)
        self.assertLessEqual(s.risk_usd, 500.0 + 1e-6)
        self.assertGreater(s.risk_usd, 500.0 - 1000 * dist)   # within one 0.01-lot step
        self.assertAlmostEqual(s.commission, 7.0 * s.units / 100_000)
        gross = (1.1029 - 1.1011) * s.units
        self.assertAlmostEqual(s.pnl_usd, gross - s.commission, places=6)


class TestNoLookahead(unittest.TestCase):
    """Trades that closed before time T must not change when data after T is
    removed (truncation) or replaced with unrelated prices (perturbation)."""

    def setUp(self):
        self.df = random_walk_utc(start="2024-01-01", weeks=12, seed=7)
        self.days = orb.trading_days(self.df, LONDON)

    def _closed_before(self, df, cut):
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON, self.days)
        return [t for t in trades if t["t_out"] < cut.value]

    def test_truncation_and_perturbation(self):
        rng = np.random.default_rng(1)
        full = self._closed_before(self.df, self.df.index[-1] + pd.Timedelta(minutes=1))
        self.assertGreater(len(full), 30)
        for k in rng.integers(len(self.df) // 5, len(self.df) - 1, size=25):
            cut = self.df.index[k]
            want = [t for t in full if t["t_out"] < cut.value]
            truncated = self._closed_before(self.df[self.df.index < cut], cut)
            noisy = self.df.copy()
            after = noisy.index >= cut
            noise = random_walk_utc(start="2024-01-01", weeks=12, seed=int(k))
            noisy.loc[after, ["open", "high", "low", "close"]] = \
                noise.loc[noisy.index[after], ["open", "high", "low", "close"]].to_numpy() + 0.05
            perturbed = self._closed_before(noisy, cut)
            self.assertEqual(want, truncated, f"truncation changed trades at {cut}")
            self.assertEqual(want, perturbed, f"perturbation changed trades at {cut}")


class TestRandomWalkSanity(unittest.TestCase):
    """On a driftless random walk ORB has no edge: gross mean R must be
    indistinguishable from 0 and costs must make it worse."""

    def test_no_edge_on_random_walk(self):
        df = random_walk_utc(start="2023-01-02", weeks=104, seed=3)
        bars = bars_from(df)
        for sess in orb.SESSIONS:
            days = orb.trading_days(df, sess)
            g, _ = orb.simulate_session(bars, "EURUSD", sess, days, costs=False)
            n, _ = orb.simulate_session(bars, "EURUSD", sess, days)
            rg = orb.size_and_score(g, "EURUSD", costs=False).R.to_numpy()
            rn = orb.size_and_score(n, "EURUSD").R.to_numpy()
            t = rg.mean() / (rg.std(ddof=1) / np.sqrt(len(rg)))
            self.assertLess(abs(t), 3.0, f"{sess.name}: gross t={t:.2f} on random data")
            self.assertLess(rn.mean(), rg.mean(), sess.name)


class TestEndToEnd(unittest.TestCase):
    def test_run_writes_outputs(self):
        df = random_walk_utc(start="2023-01-01", weeks=20, seed=5)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "data").mkdir()
            write_mt5(to_server(df, "NY+7"), d / "data" / "EURUSD_2023.csv")
            res = orb.run(d / "data", d / "out", "2023-01-01", None, None)
            self.assertEqual(set(res.session), {"Tokyo", "London", "NewYork"})
            self.assertEqual(set(res.case), {"net", "net_2x_spread", "gross"})
            for f in ("orb_metrics.csv", "orb_trades_net.csv", "orb_by_year.csv",
                      "orb_pooled_by_session.csv", "orb_cum_R.png", "data_notes.txt"):
                self.assertTrue((d / "out" / f).exists(), f)
            self.assertIn("broker server timezone: NY+7", (d / "out" / "data_notes.txt").read_text())


if __name__ == "__main__":
    unittest.main()
