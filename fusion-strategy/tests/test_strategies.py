import unittest

import numpy as np
import pandas as pd

from fusion import indicators as ind
from fusion.data.resample import resample
from fusion.strategies import STRATEGIES, Signals
from fusion.strategies.trend import turtle, turtle_s2
from fusion.timeutil import trading_day
from tests.helpers import datasets, fx_hours, random_walk

TF = datasets()
M15, H1, D1_LONG = TF["M15"], TF["H1"], TF["D1"]


def path(closes, half=0.5, start="2020-01-01"):
    """Daily-like bars from a close path: open = previous close, high/low =
    max/min(open, close) +/- half."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range(start, periods=len(c), freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + half,
                         "low": np.minimum(o, c) - half, "close": c}, index=idx)


class TestRegistry(unittest.TestCase):
    def test_ids_and_timeframes_match_plan(self):
        want = {"T1": 4, "T2": 4, "T3": 4, "T4": 4, "M1": 1, "M2": 4, "R1": 4, "R2": 4,
                "V1": 4, "V2": 2, "S1": 2, "S2": 1, "P1": 2, "P2": 2}
        self.assertEqual({k: len(s.timeframes) for k, s in STRATEGIES.items()}, want)
        self.assertEqual(sum(want.values()), 42)            # PLAN.md 4.8

    def test_every_strategy_runs_validates_and_trades(self):
        for sid, spec in STRATEGIES.items():
            for tf in spec.timeframes:
                with self.subTest(sid=sid, tf=tf):
                    sig = spec.run(TF[tf])                    # run() validates the stop rule
                    n = int(sig.entries("long").sum() + sig.entries("short").sum())
                    self.assertGreater(n, 0, "no entry orders on synthetic data")

    def test_validate_rejects_entry_without_stop(self):
        idx = M15.index[:5]
        sig = Signals(idx, long_mkt=[False, True, False, False, False])
        with self.assertRaises(ValueError):
            sig.validate()


class TestTrend(unittest.TestCase):
    def test_ma_cross(self):
        df = H1
        sig = STRATEGIES["T1"].run(df)
        f, s = ind.sma(df.close, 50), ind.sma(df.close, 200)
        up = (f > s) & (f.shift() <= s.shift())
        self.assertTrue(np.array_equal(sig.long_mkt, up.to_numpy()))
        self.assertTrue(np.array_equal(sig.exit_short_mkt, up.to_numpy()))
        self.assertTrue(np.allclose(sig.stop_long_dist[300:], 2 * ind.atr(df, 14).to_numpy()[300:]))

    def test_turtle_levels_without_skip(self):
        df = H1
        sig = turtle_s2(df)
        t = 400
        self.assertAlmostEqual(sig.long_stop[t], df.high.iloc[t - 54:t + 1].max())
        self.assertAlmostEqual(sig.short_stop[t], df.low.iloc[t - 54:t + 1].min())
        self.assertAlmostEqual(sig.exit_long_stop[t], df.low.iloc[t - 19:t + 1].min())
        self.assertAlmostEqual(sig.stop_long_dist[t], 2 * ind.atr(df, 20).iloc[t])
        self.assertFalse(sig.notes["skip_active"].any())

    def test_turtle_winning_breakout_turns_on_skip_and_failsafe(self):
        closes = [100.0] * 60 + list(np.arange(101.0, 117.0)) + [105.0] * 12
        df = path(closes)
        sig = turtle(df)
        skip = sig.notes["skip_active"]
        # breakout at bar 60 (entry 100.5, 2N stop 98.5); 10-day exit on bar 76 at 106.5
        self.assertFalse(skip[75])
        self.assertTrue(skip[76])
        t = 80
        self.assertAlmostEqual(sig.long_stop[t], df.high.iloc[t - 54:t + 1].max())   # failsafe 55
        self.assertAlmostEqual(sig.short_stop[t], df.low.iloc[t - 54:t + 1].min())

    def test_turtle_losing_breakout_keeps_20_day_entry(self):
        # the long loses on bar 61, which also breaks the 20-bar low; that
        # short is tracked and exits as a winner on bar 72
        closes = [100.0] * 60 + [101.0] + [97.0] * 12
        df = path(closes)
        sig = turtle(df)
        skip = sig.notes["skip_active"]
        self.assertFalse(skip[:72].any())
        self.assertTrue(skip[72])
        t = 66
        self.assertAlmostEqual(sig.long_stop[t], df.high.iloc[t - 19:t + 1].max())

    def test_adx_gate(self):
        df = H1
        sig = STRATEGIES["T4"].run(df)
        d = ind.dmi(df, 14)
        up = ((d.plus_di > d.minus_di) & (d.plus_di.shift() <= d.minus_di.shift())).to_numpy()
        self.assertTrue(np.array_equal(sig.long_mkt, up & (d.adx > 25).to_numpy()))
        self.assertTrue(np.array_equal(sig.exit_short_mkt, up))


class TestMomentum(unittest.TestCase):
    def test_tsmom_month_end_sign(self):
        df = D1_LONG
        sig = STRATEGIES["M1"].run(df)
        days = trading_day(df.index)
        idx = np.flatnonzero(sig.long_mkt | sig.short_mkt)
        self.assertGreater(len(idx), 12)
        for i in idx:
            d = days[i]
            self.assertNotEqual((d + pd.offsets.BDay(1)).month, d.month)   # last weekday
            past = df.close[days <= d - pd.DateOffset(months=12)]
            self.assertTrue(len(past) > 0)
            r = df.close.iloc[i] / past.iloc[-1] - 1
            self.assertEqual(bool(sig.long_mkt[i]), r > 0)
            self.assertTrue(sig.exit_long_mkt[i] and sig.exit_short_mkt[i])   # monthly reopen
        first = days[0] + pd.DateOffset(months=12)
        self.assertFalse((sig.long_mkt | sig.short_mkt)[days < first].any())

    def test_tsmom_rejects_intraday(self):
        with self.assertRaises(ValueError):
            STRATEGIES["M1"].run(H1)

    def test_macd_cross(self):
        df = H1
        sig = STRATEGIES["M2"].run(df)
        m = ind.macd(df.close)
        dn = ((m.macd < m.signal) & (m.macd.shift() >= m.signal.shift())).to_numpy()
        self.assertTrue(np.array_equal(sig.short_mkt, dn))
        self.assertTrue(np.array_equal(sig.exit_long_mkt, dn))


class TestMeanReversion(unittest.TestCase):
    def test_bb_rsi(self):
        df = M15
        sig = STRATEGIES["R1"].run(df)
        bb, r = ind.bollinger(df.close), ind.rsi(df.close)
        want = ((df.close < bb.lower) & (r < 30)).to_numpy()
        self.assertTrue(np.array_equal(sig.long_mkt, want))
        self.assertTrue(np.allclose(sig.exit_long_limit[50:], bb.mid.to_numpy()[50:]))

    def test_zscore(self):
        df = M15
        sig = STRATEGIES["R2"].run(df)
        z = ind.zscore(df.close, 20).to_numpy()
        with np.errstate(invalid="ignore"):
            self.assertTrue(np.array_equal(sig.short_mkt, z > 2))
            self.assertTrue(np.array_equal(sig.exit_short_mkt, z <= 0))


class TestVolatility(unittest.TestCase):
    def test_squeeze_fires_on_release_only(self):
        df = H1
        sig = STRATEGIES["V1"].run(df)
        on = sig.notes["squeeze_on"]
        fire = np.flatnonzero(sig.long_mkt | sig.short_mkt)
        self.assertGreater(len(fire), 5)
        for i in fire:
            self.assertTrue(on[i - 1] and not on[i])
        mom = ind.squeeze_momentum(df).to_numpy()
        self.assertTrue((mom[np.flatnonzero(sig.long_mkt)] > 0).all())

    def test_vol_breakout_levels_and_day_end(self):
        df = M15
        sig = STRATEGIES["V2"].run(df)
        day = trading_day(df.index)
        daily = df.groupby(day).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"))
        d = daily.index[20]                     # the 14-day ATR stop needs 14 completed days
        prev = daily.iloc[19]
        bars = np.flatnonzero(day == d)
        i = bars[5]
        self.assertAlmostEqual(sig.long_stop[i], daily.o.iloc[20] + 0.5 * (prev.h - prev.l))
        self.assertAlmostEqual(sig.short_stop[i], daily.o.iloc[20] - 0.5 * (prev.h - prev.l))
        end = bars[-1]
        ny = df.index[end].tz_convert("America/New_York")
        self.assertEqual((ny.hour, ny.minute), (16, 45))
        self.assertTrue(sig.flat_at_close[end])
        self.assertTrue(np.isnan(sig.long_stop[end]))
        first_day = np.flatnonzero(day == daily.index[0])
        self.assertTrue(np.isnan(sig.long_stop[first_day]).all())
        self.assertEqual(sig.max_entries_per_group, 1)


class TestSession(unittest.TestCase):
    def _check(self, sid, df, tz, r0, r1, e1, exit_):
        sig = STRATEGIES[sid].run(df)
        loc = df.index.tz_convert(tz)
        mins = loc.hour * 60 + loc.minute
        blen = int((df.index[1] - df.index[0]).total_seconds() // 60)
        h = lambda s: int(s[:2]) * 60 + int(s[3:])
        live = np.isfinite(sig.long_stop)
        nxt = mins + blen
        self.assertTrue(((nxt[live] >= h(r1)) & (nxt[live] < h(e1))).all())
        i = np.flatnonzero(live)[3]
        date = loc[i].date()
        rng = (loc.date == date) & (mins >= h(r0)) & (mins < h(r1))
        hi, lo = df.high[rng].max(), df.low[rng].min()
        self.assertAlmostEqual(sig.long_stop[i], hi)
        self.assertAlmostEqual(sig.stop_long_px[i], lo)
        self.assertAlmostEqual(sig.target_long_px[i], hi + (hi - lo))
        self.assertAlmostEqual(sig.target_short_px[i], lo - (hi - lo))
        flat = np.flatnonzero(sig.flat_at_close)
        self.assertTrue(((mins[flat] < h(exit_)) & (mins[flat] + blen >= h(exit_))).all())
        return sig

    def test_asian_london_m15_and_h1(self):
        for df in (M15, H1):
            self._check("S1", df, "Europe/London", "00:00", "07:00", "12:00", "16:00")

    def test_ny_orb(self):
        self._check("S2", M15, "America/New_York", "08:00", "08:30", "11:00", "16:00")

    def test_london_range_follows_dst(self):
        sig = STRATEGIES["S1"].run(M15)
        utc_hours = pd.Series(M15.index.hour[np.isfinite(sig.long_stop)],
                              index=M15.index[np.isfinite(sig.long_stop)])
        summer = utc_hours[utc_hours.index.month == 5]
        winter = utc_hours[utc_hours.index.month == 1]
        self.assertEqual(summer.min(), 5)    # 06:00 London summer = 05:00 UTC (bar before)
        self.assertEqual(winter.min(), 6)


class TestStructure(unittest.TestCase):
    def test_pivots_and_prior_day_levels(self):
        df = M15
        p1, p2 = STRATEGIES["P1"].run(df), STRATEGIES["P2"].run(df)
        day = trading_day(df.index)
        daily = df.groupby(day).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
        prev = daily.iloc[19]
        i = np.flatnonzero(day == daily.index[20])[10]
        P = (prev.h + prev.l + prev.c) / 3
        self.assertAlmostEqual(p1.short_limit[i], 2 * P - prev.l)
        self.assertAlmostEqual(p1.long_limit[i], 2 * P - prev.h)
        self.assertAlmostEqual(p1.stop_short_px[i], P + (prev.h - prev.l))
        self.assertAlmostEqual(p1.stop_long_px[i], P - (prev.h - prev.l))
        self.assertAlmostEqual(p1.target_long_px[i], P)
        self.assertAlmostEqual(p2.long_stop[i], prev.h)
        self.assertAlmostEqual(p2.stop_long_px[i], (prev.h + prev.l) / 2)
        self.assertTrue(p1.per_side and p2.per_side)


class TestResample(unittest.TestCase):
    def test_d1_has_five_bars_a_week_starting_17_ny(self):
        d1 = resample(M15, "D1")
        ny = d1.index.tz_convert("America/New_York")
        self.assertTrue(((ny.hour == 17) & (ny.minute == 0)).all())
        self.assertNotIn(5, set(ny.dayofweek))           # no Saturday-start bar
        self.assertFalse((pd.Series(trading_day(d1.index)).dt.dayofweek >= 5).any())

    def test_h4_buckets_anchor_to_roll(self):
        h4 = resample(M15, "H4")
        ny = h4.index.tz_convert("America/New_York")
        self.assertTrue(set(ny.hour) <= {17, 21, 1, 5, 9, 13})

    def test_every_source_bar_sits_inside_its_resampled_bar(self):
        for tf in ("H1", "H4", "D1"):
            out = resample(M15, tf)
            lab = out.index.as_unit("ns").asi8
            src = M15.index.as_unit("ns").asi8
            k = np.searchsorted(lab, src, side="right") - 1
            self.assertTrue((k >= 0).all(), tf)
            if tf == "D1":
                same_day = trading_day(M15.index) == trading_day(out.index[k])
                self.assertTrue(np.asarray(same_day).all(), tf)
            else:
                span = {"H1": 3600, "H4": 4 * 3600}[tf] * 10**9
                self.assertTrue((src < lab[k] + span).all(), tf)
            grp = pd.Series(k, index=M15.index)
            agg = M15.groupby(grp.to_numpy()).agg(open=("open", "first"), high=("high", "max"),
                                                  low=("low", "min"), close=("close", "last"))
            got = out.iloc[agg.index][["open", "high", "low", "close"]].to_numpy()
            self.assertTrue(np.allclose(got, agg.to_numpy()), tf)

    def test_h1_aggregation(self):
        t0 = H1.index[50]
        part = M15[(M15.index >= t0) & (M15.index < t0 + pd.Timedelta(hours=1))]
        self.assertEqual(H1.loc[t0, "open"], part.open.iloc[0])
        self.assertEqual(H1.loc[t0, "high"], part.high.max())
        self.assertEqual(H1.loc[t0, "close"], part.close.iloc[-1])


if __name__ == "__main__":
    unittest.main()
