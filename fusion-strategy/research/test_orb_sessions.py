"""Tests for orb_sessions.py. Run: python -m unittest test_orb_sessions -v"""
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import orb_sessions as orb

MIN = 60_000_000_000
LONDON = orb.SESSIONS[1]
DAY = pd.Timestamp("2024-03-05").date()


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
    """Bars that use the raw bar spread, so expected prices are exact."""
    d = df_utc.copy()
    d["spread_px"] = d.spread * point
    return orb.to_bars(d, col="spread_px")


def london_day(day="2024-03-05", base=1.1000, rng_half=0.0010, spread_pts=10):
    """Flat London day (bars base +/- 0.0001, spread 1 pip). The 08:00-08:30
    range is base +/- rng_half: high 1.1010, low 1.0990, range 0.0020."""
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


def set_bar(df, local, hh, mm, **ohlc):
    for k, v in ohlc.items():
        df.loc[at(local, hh, mm), k] = v


def sim(df, sym="EURUSD", **kw):
    return orb.simulate_session(bars_from(df), sym, LONDON, [DAY], **kw)


def local_hm(t_ns):
    t = pd.Timestamp(t_ns, tz="UTC").tz_convert("Europe/London")
    return t.hour, t.minute


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

    def test_rejects_unsupported_bar_size(self):
        df = random_walk_utc(weeks=1).iloc[::7].iloc[:300]   # 7-minute bars
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "EURUSD_M7.csv"
            write_mt5(to_server(df, "NY+7"), p, utf16=False)
            with self.assertRaises(ValueError):
                orb.read_mt5_csv(p)

    def test_bad_rows_are_dropped_and_counted(self):
        df = random_walk_utc(weeks=1).iloc[:500]
        raw = to_server(df, "NY+7")
        raw.iloc[10, raw.columns.get_loc("spread")] = np.nan
        raw.iloc[20, raw.columns.get_loc("high")] = raw.iloc[20]["low"] - 0.001
        out, point, dropped, n_bad = orb.load_symbol(raw, "NY+7")
        self.assertEqual(n_bad, 2)
        self.assertEqual(len(out), 498)
        self.assertFalse(out[["open", "high", "low", "close", "spread_cons"]].isna().any().any())


class TestIndicesAndDigits(unittest.TestCase):
    def test_digits_come_from_text_not_float(self):
        col = pd.Series(["33361.50", "33368.50", "33264.00"])
        self.assertEqual(orb.text_digits(col), 2)               # floats alone suggest 1
        self.assertAlmostEqual(orb.infer_point(col.astype(float).to_numpy()), 0.1)
        self.assertEqual(orb.text_digits(pd.Series(["130.924"])), 3)

    def test_loaded_index_spread_uses_text_digits(self):
        idx = pd.date_range("2024-01-02 01:00", periods=200, freq="15min")
        lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
        lines += [f"{t:%Y.%m.%d}\t{t:%H:%M:%S}\t33361.50\t33368.50\t33264.50\t33271.50\t1\t0\t150"
                  for t in idx]
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "US30_M15.csv"
            f.write_text("\n".join(lines) + "\n")
            raw = orb.load_raw([f])
        self.assertEqual(raw.attrs["digits"], 2)
        out, point, _, _ = orb.load_symbol(raw, "NY+7")
        self.assertAlmostEqual(point, 0.01)
        self.assertAlmostEqual(out.spread_px.iloc[0], 1.50)      # 150 points = 1.5 index points

    def test_symbols_and_sessions(self):
        self.assertEqual(orb.symbol_of(pathlib.Path("USTEC_M15_2023.csv")), "USTEC")
        self.assertEqual(orb.symbol_of(pathlib.Path("US30_M15_2023.csv")), "US30")
        ny = orb.sessions_for("USTEC")[2]
        self.assertEqual((ny.range_start, ny.range_end), ("09:30", "10:00"))
        self.assertEqual(orb.sessions_for("XAUUSD")[2].range_start, "08:00")

    def test_index_commission_is_per_notional(self):
        t = dict(day=DAY, session="NewYork", symbol="US30", dir=1, t_in=0, t_out=MIN,
                 mins_after_range=0, reason="time", level=40000.0, entry=40002.0,
                 exit=40050.0, stop=39950.0, range=50.0, plan_dist=54.0)
        s = orb.size_and_score([t], "US30").iloc[0]
        self.assertAlmostEqual(s.units, 9.25)                   # floor(500 / 54, 0.01)
        self.assertAlmostEqual(s.commission, 7.0 * 9.25 * 40002.0 / 100_000)


class TestServerTimezone(unittest.TestCase):
    def test_detects_ny_plus_7(self):
        df = random_walk_utc(start="2023-01-01", weeks=60)
        tz, score = orb.detect_server_tz(to_server(df, "NY+7").index)
        self.assertEqual(tz, "NY+7")
        self.assertGreater(score, 0.95)

    def test_detects_fixed_utc_plus_2(self):
        df = random_walk_utc(start="2023-01-01", weeks=60)
        tz, _ = orb.detect_server_tz(to_server(df, "Etc/GMT-2").index)
        self.assertEqual(tz, "Etc/GMT-2")

    def test_round_trip_to_utc(self):
        df = random_walk_utc(start="2023-03-01", weeks=10)
        back = orb.server_to_utc(to_server(df, "NY+7").index, "NY+7")
        self.assertTrue((back == df.index).all())

    def test_session_windows_in_dst_mismatch_weeks(self):
        # 2024-03-12: US on summer time, UK not yet. 2024-10-29: UK back, US not.
        cases = {"2024-03-12": (8, 12, 0), "2024-10-29": (8, 12, 0), "2024-07-02": (7, 12, 0)}
        for day, (lon, ny, tok) in cases.items():
            d = pd.Timestamp(day).date()
            got = [pd.Timestamp(orb._local_bound(d, s.range_start, s.tz), tz="UTC").hour
                   for s in orb.SESSIONS]
            self.assertEqual(got, [tok, lon, ny], day)


class TestSpread(unittest.TestCase):
    def test_conservative_spread_is_causal_and_never_below_raw(self):
        df = random_walk_utc(start="2024-01-01", weeks=8, seed=2)
        rng = np.random.default_rng(0)
        raw = pd.Series(rng.exponential(0.00005, len(df)), index=df.index)
        cons = orb.conservative_spread(raw)
        self.assertTrue((cons >= raw - 1e-15).all())
        cut = df.index[len(df) // 2]
        changed = raw.copy()
        changed[changed.index >= cut] *= 50
        cons2 = orb.conservative_spread(changed)
        before = cons.index < cut
        self.assertTrue(np.allclose(cons[before], cons2[before]))
        self.assertGreater((cons > raw).mean(), 0.3)


class TestTradeRules(unittest.TestCase):
    def test_long_breakout_hits_target(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)            # ask 1.1013 >= 1.1010
        set_bar(df, local, 9, 0, high=1.1031)             # target 1.1030
        trades, stats = sim(df)
        self.assertEqual(stats, dict(skipped_missing=0, incomplete=0))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["dir"], t["reason"]), (1, "target"))
        self.assertAlmostEqual(t["entry"], 1.1011)
        self.assertAlmostEqual(t["exit"], 1.1029)
        self.assertAlmostEqual(t["stop"], 1.0990)
        self.assertAlmostEqual(t["plan_dist"], 0.0022)

    def test_short_stop_is_judged_on_ask(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)              # short at 1.0990 - slip
        set_bar(df, local, 9, 30, high=1.1009)            # bid below stop, ask 1.1010 hits it
        (t,), _ = sim(df)
        self.assertEqual((t["dir"], t["reason"]), (-1, "stop"))
        self.assertAlmostEqual(t["entry"], 1.0989)
        self.assertAlmostEqual(t["exit"], 1.1011)
        self.assertEqual(local_hm(t["t_out"]), (9, 30))

    def test_short_target_is_judged_on_ask(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)
        set_bar(df, local, 9, 30, low=1.0970)             # bid at target, ask 1.0971 is not
        set_bar(df, local, 9, 40, low=1.0969)             # ask 1.0970 reaches target
        (t,), _ = sim(df)
        self.assertEqual(t["reason"], "target")
        self.assertEqual(local_hm(t["t_out"]), (9, 40))
        self.assertAlmostEqual(t["exit"], 1.0971)

    def test_gap_entry_and_gap_stop_fill_at_open(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, open=1.1025, high=1.1026, low=1.1024, close=1.1025)
        set_bar(df, local, 9, 0, open=1.0980, high=1.0981, low=1.0975, close=1.0980)
        (t,), _ = sim(df)
        self.assertAlmostEqual(t["entry"], 1.1027)        # ask open 1.1026 + slip
        self.assertEqual(t["reason"], "stop")
        self.assertAlmostEqual(t["exit"], 1.0979)         # bid open 1.0980 - slip
        s = orb.size_and_score([t], "EURUSD").iloc[0]
        self.assertLess(s.R, -2.0)                        # sized on the level, gap loses > 1R

    def test_short_gap_entry_fills_at_open(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, open=1.0975, high=1.0976, low=1.0974, close=1.0975)
        (t,), _ = sim(df)
        self.assertEqual(t["dir"], -1)
        self.assertAlmostEqual(t["entry"], 1.0974)        # bid open 1.0975 - slip

    def test_short_gap_stop_fills_at_ask_open(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)
        set_bar(df, local, 9, 30, open=1.1020, high=1.1021, low=1.1019, close=1.1020)
        (t,), _ = sim(df)
        self.assertAlmostEqual(t["exit"], 1.1022)

    def test_range_excludes_the_0830_bar(self):
        df, local = london_day()
        set_bar(df, local, 8, 30, high=1.1050)
        (t,), _ = sim(df)
        self.assertAlmostEqual(t["level"], 1.1010)
        self.assertAlmostEqual(t["range"], 0.0020)
        self.assertEqual(local_hm(t["t_in"]), (8, 30))

    def test_no_entry_on_last_range_bar(self):
        df, local = london_day()
        set_bar(df, local, 8, 29, high=1.1012)            # part of the range: high becomes 1.1012
        trades, _ = sim(df)
        self.assertEqual(trades, [])

    def test_target_multiple(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)
        set_bar(df, local, 9, 0, high=1.1031)             # 1x target 1.1030
        set_bar(df, local, 10, 0, high=1.1051)            # 2x target 1.1050
        (one,), _ = sim(df)
        (two,), _ = sim(df, target_mult=2.0)
        (half,), _ = sim(df, target_mult=0.5)
        self.assertEqual(local_hm(one["t_out"]), (9, 0))
        self.assertAlmostEqual(one["exit"], 1.1029)
        self.assertEqual(local_hm(two["t_out"]), (10, 0))
        self.assertAlmostEqual(two["exit"], 1.1049)
        self.assertEqual(half["reason"], "target")
        self.assertEqual(local_hm(half["t_out"]), (9, 0))    # 1.1020 first reached at 09:00
        self.assertAlmostEqual(half["exit"], 1.1019)
        set_bar(df, local, 8, 50, high=1.1021)
        (half,), _ = sim(df, target_mult=0.5)
        self.assertEqual(local_hm(half["t_out"]), (8, 50))
        self.assertAlmostEqual(half["plan_dist"], one["plan_dist"])

    def test_partial_take_profit(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)            # long entry 1.1011
        set_bar(df, local, 9, 0, high=1.1021)             # 0.5x target 1.1020
        set_bar(df, local, 10, 0, high=1.1051)            # 2x target 1.1050
        legs = [sim(df, target_mult=m)[0] for m in (0.5, 2.0)]
        (t,), dropped = orb.merge_legs(legs, (0.5, 0.5))
        self.assertEqual((t["reason"], dropped), ("target+target", 0))
        self.assertEqual(local_hm(t["t_out"]), (10, 0))
        s = orb.size_and_score([t], "EURUSD").iloc[0]
        self.assertEqual(s.units, 226_000)                 # two legs of 113,000
        want = 113_000 * (1.1019 - 1.1011) + 113_000 * (1.1049 - 1.1011) - 7.0 * 2.26
        self.assertAlmostEqual(s.pnl_usd, want, places=6)
        # first leg banks, second is stopped
        set_bar(df, local, 10, 0, high=1.1001, low=1.0985)
        legs = [sim(df, target_mult=m)[0] for m in (0.5, 2.0)]
        (t,), _ = orb.merge_legs(legs, (0.5, 0.5))
        self.assertEqual(t["reason"], "target+stop")
        s = orb.size_and_score([t], "EURUSD").iloc[0]
        want = 113_000 * (1.1019 - 1.1011) + 113_000 * (1.0989 - 1.1011) - 7.0 * 2.26
        self.assertAlmostEqual(s.pnl_usd, want, places=6)

    def test_single_full_leg_equals_plain_target(self):
        df = random_walk_utc(start="2024-01-01", weeks=8, seed=9)
        days = orb.trading_days(df, LONDON)
        plain, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON, days, target_mult=2.0)
        merged, _ = orb.merge_legs([plain], (1.0,))
        a = orb.size_and_score(plain, "EURUSD").R.to_numpy()
        b = orb.size_and_score(merged, "EURUSD").R.to_numpy()
        self.assertTrue(np.allclose(a, b))

    def test_without_stop_trade_rides_to_target_or_time(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)
        set_bar(df, local, 9, 0, low=1.0950)              # far through the stop
        set_bar(df, local, 10, 0, high=1.1031)            # later reaches the target
        (with_stop,), _ = sim(df)
        (no_stop,), _ = sim(df, use_stop=False)
        self.assertEqual(with_stop["reason"], "stop")
        self.assertEqual((no_stop["reason"], local_hm(no_stop["t_out"])), ("target", (10, 0)))
        self.assertAlmostEqual(no_stop["plan_dist"], with_stop["plan_dist"])
        set_bar(df, local, 10, 0, high=1.1001)            # never reaches the target
        (no_stop,), _ = sim(df, use_stop=False)
        self.assertEqual(no_stop["reason"], "time")

    def test_stop_wins_when_stop_and_target_share_a_later_bar(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)
        set_bar(df, local, 9, 0, high=1.1031, low=1.0989)
        (t,), _ = sim(df)
        self.assertEqual(t["reason"], "stop")
        self.assertAlmostEqual(t["exit"], 1.0989)

    def test_both_sides_in_one_bar_is_a_loss(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1015, low=1.0985)
        (t,), _ = sim(df)
        self.assertEqual(t["reason"], "stop")
        self.assertLess(t["exit"], t["entry"])

    def test_target_not_allowed_on_entry_bar(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1040)
        (t,), _ = sim(df)
        self.assertNotEqual(t["reason"], "target")
        self.assertNotEqual(t["t_out"], t["t_in"])

    def test_entry_past_target_sets_no_target(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, open=1.1035, high=1.1036, low=1.1034, close=1.1035)
        set_bar(df, local, 9, 0, high=1.1040)
        (t,), _ = sim(df)
        self.assertEqual(t["reason"], "time")

    def test_short_time_exit_pays_spread(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)
        (t,), _ = sim(df)
        self.assertEqual((t["dir"], t["reason"]), (-1, "time"))
        self.assertAlmostEqual(t["exit"], 1.1000 + 0.0001 + 0.0001)
        self.assertEqual(local_hm(t["t_out"]), (15, 59))

    def test_stress_adds_extra_spread(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)
        (t,), _ = sim(df, stress=True)
        self.assertAlmostEqual(t["exit"], 1.1000 + 0.0002 + 0.0001)

    def test_gross_has_no_costs(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)
        (t,), _ = sim(df, costs=False)
        self.assertAlmostEqual(t["entry"], 1.0990)
        self.assertAlmostEqual(t["exit"], 1.1000)

    def test_no_entry_after_entry_window(self):
        df, local = london_day()
        set_bar(df, local, 11, 5, high=1.1020)
        trades, _ = sim(df)
        self.assertEqual(trades, [])

    def test_missing_range_bars_skips_day(self):
        df, local = london_day()
        df = df[~((local.hour == 8) & (local.minute < 10))]
        trades, stats = sim(df)
        self.assertEqual((trades, stats["skipped_missing"]), ([], 1))

    def test_open_position_at_end_of_data_is_dropped(self):
        df, local = london_day()
        set_bar(df, local, 9, 0, low=1.0988)              # short, would exit at 16:00
        df = df[local.hour < 12]
        trades, stats = sim(df)
        self.assertEqual((trades, stats["incomplete"]), ([], 1))

    def test_trade_closed_before_end_of_data_is_kept(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)
        set_bar(df, local, 9, 0, high=1.1031)
        df = df[local.hour < 12]
        (t,), stats = sim(df)
        self.assertEqual((t["reason"], stats["incomplete"]), ("target", 0))


def to_m15(df_utc):
    g = df_utc.resample("15min", label="left", closed="left")
    out = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                        "close": g.close.last(), "spread": g.spread.min()}).dropna()
    return out


class TestM15(unittest.TestCase):
    def test_loader_accepts_m15_and_rejects_mixed(self):
        df = random_walk_utc(weeks=2)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            write_mt5(to_server(to_m15(df), "NY+7"), d / "EURUSD_2023.csv", utf16=False)
            self.assertEqual(orb.bar_minutes(orb.read_mt5_csv(d / "EURUSD_2023.csv").index), 15)
            write_mt5(to_server(df.iloc[:3000], "NY+7"), d / "EURUSD_2024.csv", utf16=False)
            with self.assertRaises(ValueError):
                orb.load_raw([d / "EURUSD_2023.csv", d / "EURUSD_2024.csv"])

    def test_m15_breakout_and_range_bars(self):
        df, local = london_day()
        set_bar(df, local, 8, 40, high=1.1012)
        set_bar(df, local, 9, 0, high=1.1031)
        m15 = to_m15(df)
        b = bars_from(m15)
        self.assertEqual(b.bar_ns, 15 * MIN)
        (t,), stats = orb.simulate_session(b, "EURUSD", LONDON, [DAY])
        self.assertEqual((t["dir"], t["reason"]), (1, "target"))
        self.assertAlmostEqual(t["level"], 1.1010)
        self.assertEqual(local_hm(t["t_in"]), (8, 30))     # the 08:30-08:45 bar
        self.assertEqual(local_hm(t["t_out"]), (9, 0))
        one_missing = m15[~at(m15.index.tz_convert("Europe/London"), 8, 15)]
        trades, stats = orb.simulate_session(bars_from(one_missing), "EURUSD", LONDON, [DAY])
        self.assertEqual((trades, stats["skipped_missing"]), ([], 1))


class TestSizing(unittest.TestCase):
    @staticmethod
    def trade(**kw):
        base = dict(day=DAY, session="London", symbol="X", dir=1, t_in=0, t_out=MIN,
                    mins_after_range=0, reason="time")
        return {**base, **kw}

    def test_eurusd_sized_on_planned_loss(self):
        t = self.trade(level=1.1010, entry=1.1011, exit=1.1029, stop=1.0990, range=0.0020,
                       plan_dist=0.0022)
        s = orb.size_and_score([t], "EURUSD").iloc[0]
        self.assertEqual(s.units, 227_000)                # floor(500 / 0.0022) to 1,000
        self.assertAlmostEqual(s.commission, 7.0 * 2.27)
        self.assertAlmostEqual(s.pnl_usd, 0.0018 * 227_000 - 7.0 * 2.27, places=6)
        self.assertAlmostEqual(s.risk_usd, 227_000 * 0.0022)

    def test_usdjpy_converted_to_usd(self):
        t = self.trade(level=150.00, entry=150.01, exit=150.50, stop=149.80, range=0.20,
                       plan_dist=0.22)
        s = orb.size_and_score([t], "USDJPY").iloc[0]
        self.assertEqual(s.units, 340_000)
        self.assertAlmostEqual(s.commission, 7.0 * 3.4)
        self.assertAlmostEqual(s.pnl_usd, 0.49 * 340_000 / 150.50 - 7.0 * 3.4, places=6)
        self.assertAlmostEqual(s.risk_usd, 340_000 * 0.22 / 150.00, places=6)

    def test_xau_commission_per_100k_notional(self):
        t = self.trade(level=2000.00, entry=2000.20, exit=2010.00, stop=1990.00, range=10.0,
                       plan_dist=10.4)
        s = orb.size_and_score([t], "XAUUSD").iloc[0]
        self.assertEqual(s.units, 48)
        self.assertAlmostEqual(s.commission, 7.0 * 48 * 2000.20 / 100_000)
        self.assertAlmostEqual(s.pnl_usd, 9.8 * 48 - s.commission, places=6)

    def test_risk_setting_scales_size_not_r(self):
        t = self.trade(level=1.1010, entry=1.1011, exit=1.0989, stop=1.0990, range=0.0020,
                       plan_dist=0.0022)
        lo = orb.size_and_score([t, t], "EURUSD", risk_pct=0.005)
        hi = orb.size_and_score([t, t], "EURUSD", risk_pct=0.03)
        self.assertAlmostEqual(lo.R.iloc[0], hi.R.iloc[0], places=2)
        self.assertGreater(hi.units.iloc[0], 5.9 * lo.units.iloc[0])
        self.assertLess(hi.equity.iloc[-1], lo.equity.iloc[-1])
        self.assertAlmostEqual(hi.risk_pct_actual.iloc[0], 0.03, places=3)

    def test_leverage_cap(self):
        t = self.trade(level=1.3000, entry=1.3001, exit=1.3000, stop=1.3000, range=0.0,
                       plan_dist=0.0002)
        s = orb.size_and_score([t], "GBPUSD").iloc[0]
        self.assertTrue(s.capped)
        self.assertEqual(s.units, 2_307_000)               # 30 x 100k / 1.30, floored


class TestMetrics(unittest.TestCase):
    def test_hand_calculation(self):
        pnl = [1000.0, -500.0, -1000.0, 2000.0]
        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-10"])
        eq = 100_000 + np.cumsum(pnl)
        t_out = (days + pd.Timedelta(hours=15)).tz_localize("UTC").as_unit("ns").asi8
        df = pd.DataFrame(dict(pnl_usd=pnl, equity=eq, t_out=t_out, R=[2.0, -1.0, -2.0, 4.0],
                               dir=[1, -1, 1, -1], capped=False))
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-12")
        m = orb.metrics(df, start, end)
        self.assertAlmostEqual(m["max_dd"], 1500 / 101_000)
        self.assertAlmostEqual(m["profit_factor"], 2.0)
        self.assertAlmostEqual(m["net_profit"], 1500.0)
        self.assertAlmostEqual(m["cagr"], 1.015 ** (365.25 / 11) - 1)
        self.assertAlmostEqual(m["avg_R_long"], 0.0)
        self.assertAlmostEqual(m["avg_R_short"], 1.5)
        # independent Sharpe: 10 business days, returns on the trade days only
        bd = pd.bdate_range(start, end)
        r = pd.Series(0.0, index=bd)
        prev = 100_000.0
        for d, p in zip(days, pnl):
            r[d] = p / prev
            prev += p
        self.assertAlmostEqual(m["sharpe"], r.mean() / r.std() * np.sqrt(252))

    def test_pooled_t_stat_clusters_by_day(self):
        rng = np.random.default_rng(4)
        days = pd.bdate_range("2024-01-01", periods=200).date
        r = rng.normal(0.05, 1.0, len(days))
        one = pd.DataFrame(dict(session="London", day=days, R=r))
        four = pd.concat([one.assign(symbol=s) for s in ("XAUUSD", "USDJPY", "USTEC", "US30")])
        p = orb.pooled(four).iloc[0]
        self.assertEqual(p.trades, 800)
        self.assertAlmostEqual(p.t_stat_daily_clustered, orb._t(r))


class TestNoLookahead(unittest.TestCase):
    """Trades that closed before time T must not change when data after T is
    removed (truncation) or replaced with unrelated prices (perturbation)."""

    def setUp(self):
        self.df = random_walk_utc(start="2024-01-01", weeks=12, seed=7)
        self.days = orb.trading_days(self.df, LONDON)

    def _closed_before(self, df, cut):
        trades, _ = orb.simulate_session(bars_from(df), "EURUSD", LONDON, self.days)
        return [t for t in trades if t["t_out"] < cut.as_unit("ns").value]

    def test_truncation_and_perturbation(self):
        rng = np.random.default_rng(1)
        full = self._closed_before(self.df, self.df.index[-1] + pd.Timedelta(minutes=1))
        self.assertGreater(len(full), 30)
        for k in rng.integers(len(self.df) // 5, len(self.df) - 1, size=25):
            cut = self.df.index[k]
            want = [t for t in full if t["t_out"] < cut.as_unit("ns").value]
            truncated = self._closed_before(self.df[self.df.index < cut], cut)
            noisy = self.df.copy()
            after = noisy.index >= cut
            noise = random_walk_utc(start="2024-01-01", weeks=12, seed=int(k))
            noisy.loc[after, ["open", "high", "low", "close"]] = \
                noise.loc[noisy.index[after], ["open", "high", "low", "close"]].to_numpy() + 0.05
            perturbed = self._closed_before(noisy, cut)
            self.assertEqual(want, truncated, f"truncation changed trades at {cut}")
            self.assertEqual(want, perturbed, f"perturbation changed trades at {cut}")


class TestTickGroundTruth(unittest.TestCase):
    """Simulate on a tick path (30 ticks/min, constant spread), aggregate to M1
    bid bars, and check the bar simulator reproduces the tick-level entry
    minute, exit minute and exit reason for every trade. Catches off-by-one
    windows and boundary lookahead that truncation tests cannot see."""

    TPM = 30

    def make_day(self, day, rng, spread=0.00005, vol=0.00008, p0=1.1):
        t0 = pd.Timestamp(f"{day} 07:00", tz="Europe/London").tz_convert("UTC")
        nmin = 10 * 60
        bid = np.round(p0 + np.cumsum(rng.normal(0, vol / np.sqrt(self.TPM), nmin * self.TPM)), 5)
        ask = bid + spread
        tt = t0.as_unit("ns").value + np.arange(nmin * self.TPM) * (MIN // self.TPM)
        b = bid.reshape(nmin, self.TPM)
        bars = pd.DataFrame({"open": b[:, 0], "high": b.max(1), "low": b.min(1),
                             "close": b[:, -1], "spread_px": spread},
                            index=pd.date_range(t0, periods=nmin, freq="min"))
        return tt, bid, ask, bars

    @staticmethod
    def truth(day, tt, bid, ask):
        s = LONDON
        r0, r1, e1, x1 = (orb._local_bound(day, x, s.tz)
                          for x in (s.range_start, s.range_end, s.entry_end, s.exit))
        i0, i1, ie, ix = np.searchsorted(tt, [r0, r1, e1, x1])
        hi, lo = bid[i0:i1].max(), bid[i0:i1].min()
        R = hi - lo
        big = 1 << 60
        L = np.flatnonzero(ask[i1:ie] >= hi)
        S = np.flatnonzero(bid[i1:ie] <= lo)
        jl = L[0] if len(L) else big
        js = S[0] if len(S) else big
        if jl == js == big:
            return None
        if jl <= js:
            k = i1 + jl
            st, tg = np.flatnonzero(bid[k:ix] <= lo), np.flatnonzero(bid[k:ix] >= hi + R)
        else:
            k = i1 + js
            st, tg = np.flatnonzero(ask[k:ix] >= hi), np.flatnonzero(ask[k:ix] <= lo - R)
        a = st[0] if len(st) else big
        b = tg[0] if len(tg) else big
        if a == b == big:
            kx, why = ix - 1, "time"
        elif a <= b:
            kx, why = k + a, "stop"
        else:
            kx, why = k + b, "target"
        return (1 if jl <= js else -1), tt[k] // MIN * MIN, tt[kx] // MIN * MIN, why

    def test_bars_match_ticks(self):
        rng = np.random.default_rng(3)
        days = [d.date() for d in pd.bdate_range("2023-01-02", periods=150)]
        truth, frames = {}, []
        for day in days:
            tt, bid, ask, bars = self.make_day(day, rng)
            truth[day] = self.truth(day, tt, bid, ask)
            frames.append(bars)
        trades, _ = orb.simulate_session(orb.to_bars(pd.concat(frames), col="spread_px"),
                                         "EURUSD", LONDON, days)
        self.assertGreater(len(trades), 100)
        self.assertEqual(len(trades), sum(v is not None for v in truth.values()))
        for t in trades:
            d, t_in, t_out, why = truth[t["day"]]
            self.assertEqual((t["dir"], t["t_in"], t["t_out"]), (d, t_in, t_out), t["day"])
            # same-bar stop/target ambiguity resolves to stop in the bar model
            if why != t["reason"]:
                self.assertEqual((t["reason"], why), ("stop", "target"), t["day"])


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
            self.assertLess(abs(orb._t(rg)), 3.0, f"{sess.name}: gross t on random data")
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
            self.assertEqual(set(res.case), {"net", "net_stress", "gross"})
            self.assertTrue(res[res.case != "net"].survives_gate.isna().all())
            for f in ("orb_metrics.csv", "orb_trades_net.csv", "orb_by_year.csv",
                      "orb_pooled_by_session.csv", "orb_entry_timing.csv", "orb_cum_R.png",
                      "data_notes.txt"):
                self.assertTrue((d / "out" / f).exists(), f)
            notes = (d / "out" / "data_notes.txt").read_text()
            self.assertIn("broker server timezone: NY+7", notes)
            self.assertIn("conservative", notes)


if __name__ == "__main__":
    unittest.main()
