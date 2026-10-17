"""Spec tests: every order field of every strategy, recomputed independently
from the PLAN.md rule, must match; fields a strategy does not use must stay
empty. Plus scenario tests for the Turtle skip rule, TSMOM rebalancing and
holiday early closes."""
import unittest

import numpy as np
import pandas as pd

from fusion import indicators as ind
from fusion.strategies import STRATEGIES
from fusion.strategies.base import BOOL_FIELDS, FLOAT_FIELDS, NO_LIMIT, TIME_FIELDS
from fusion.strategies.momentum import _rebalance_bars
from fusion.strategies.trend import _turtle_last_breakout_won, turtle
from fusion.timeutil import trading_day
from tests.helpers import EARLY_CLOSE_DAYS, datasets

DATA = datasets()
NAN = np.nan


def cross_up(a, b):
    return ((a > b) & (a.shift() <= b.shift())).to_numpy()


def cross_dn(a, b):
    return ((a < b) & (a.shift() >= b.shift())).to_numpy()


def utc_ns(naive_local, tz):
    return pd.DatetimeIndex(naive_local).tz_localize(tz).tz_convert("UTC").as_unit("ns").asi8


class SpecCase(unittest.TestCase):
    def assert_fields(self, sig, want: dict):
        for name in FLOAT_FIELDS + BOOL_FIELDS + TIME_FIELDS:
            got = getattr(sig, name)
            if name in want:
                exp = np.asarray(want[name])
                if name in FLOAT_FIELDS:
                    exp = exp.astype(float)
                    ok = np.array_equal(np.isnan(got), np.isnan(exp)) and \
                        np.allclose(got[~np.isnan(got)], exp[~np.isnan(exp)])
                else:
                    ok = np.array_equal(got, exp.astype(got.dtype))
                self.assertTrue(ok, f"{name} differs from the rule")
            elif name in FLOAT_FIELDS:
                self.assertTrue(np.isnan(got).all(), f"{name} should be empty")
            elif name in BOOL_FIELDS:
                self.assertFalse(got.any(), f"{name} should be empty")
            else:
                self.assertTrue((got == NO_LIMIT).all(), f"{name} should be unlimited")


class TestTrendSpec(SpecCase):
    def test_t1(self):
        df = DATA["H1"]
        f, s = ind.sma(df.close, 50), ind.sma(df.close, 200)
        stop = 2 * ind.atr(df, 14).to_numpy()
        up, dn = cross_up(f, s), cross_dn(f, s)
        self.assert_fields(STRATEGIES["T1"].run(df), dict(
            long_mkt=up, short_mkt=dn, exit_long_mkt=dn, exit_short_mkt=up,
            stop_long_dist=stop, stop_short_dist=stop))

    def _turtle(self, sid, entry_n, exit_n, skip):
        for tf in ("H1", "D1"):
            df = DATA[tf]
            n = ind.atr(df, 20).to_numpy()
            hi = df.high.rolling(entry_n).max().to_numpy()
            lo = df.low.rolling(entry_n).min().to_numpy()
            if skip:
                won = _turtle_last_breakout_won(df, entry_n, exit_n, n)
                hi = np.where(won, df.high.rolling(55).max(), hi)
                lo = np.where(won, df.low.rolling(55).min(), lo)
            ok = np.isfinite(n)
            self.assert_fields(STRATEGIES[sid].run(df), dict(
                long_stop=np.where(ok, hi, NAN), short_stop=np.where(ok, lo, NAN),
                stop_long_dist=2 * n, stop_short_dist=2 * n,
                exit_long_stop=df.low.rolling(exit_n).min(),
                exit_short_stop=df.high.rolling(exit_n).max()))

    def test_t2(self):
        self._turtle("T2", 20, 10, skip=True)

    def test_t3(self):
        self._turtle("T3", 55, 20, skip=False)

    def test_t4(self):
        df = DATA["H1"]
        d = ind.dmi(df, 14)
        up, dn = cross_up(d.plus_di, d.minus_di), cross_dn(d.plus_di, d.minus_di)
        strong = (d.adx > 25).to_numpy()
        stop = 2 * ind.atr(df, 14).to_numpy()
        self.assert_fields(STRATEGIES["T4"].run(df), dict(
            long_mkt=up & strong, short_mkt=dn & strong, exit_long_mkt=dn, exit_short_mkt=up,
            stop_long_dist=stop, stop_short_dist=stop))


class TestMomentumMeanRevSpec(SpecCase):
    def test_m1(self):
        df = DATA["D1"]
        days = trading_day(df.index)
        reb = _rebalance_bars(days)
        ret = np.full(len(df), np.nan)
        for i in range(len(df)):
            past = df.close[days <= days[i] - pd.DateOffset(months=12)]
            if len(past):
                ret[i] = df.close.iloc[i] / past.iloc[-1] - 1
        stop = 2 * ind.atr(df, 14).to_numpy()
        with np.errstate(invalid="ignore"):
            self.assert_fields(STRATEGIES["M1"].run(df), dict(
                long_mkt=reb & (ret > 0), short_mkt=reb & (ret < 0),
                exit_long_mkt=reb, exit_short_mkt=reb,
                stop_long_dist=stop, stop_short_dist=stop))

    def test_m2(self):
        df = DATA["H1"]
        m = ind.macd(df.close, 12, 26, 9)
        up, dn = cross_up(m.macd, m.signal), cross_dn(m.macd, m.signal)
        stop = 2 * ind.atr(df, 14).to_numpy()
        self.assert_fields(STRATEGIES["M2"].run(df), dict(
            long_mkt=up, short_mkt=dn, exit_long_mkt=dn, exit_short_mkt=up,
            stop_long_dist=stop, stop_short_dist=stop))

    def test_r1(self):
        df = DATA["M15"]
        bb, r = ind.bollinger(df.close, 20, 2.0), ind.rsi(df.close, 14)
        stop = 2 * ind.atr(df, 14).to_numpy()
        self.assert_fields(STRATEGIES["R1"].run(df), dict(
            long_mkt=((df.close < bb.lower) & (r < 30)).to_numpy(),
            short_mkt=((df.close > bb.upper) & (r > 70)).to_numpy(),
            exit_long_limit=bb.mid, exit_short_limit=bb.mid,
            stop_long_dist=stop, stop_short_dist=stop))

    def test_r2(self):
        df = DATA["M15"]
        mean, sd = df.close.rolling(20).mean(), df.close.rolling(20).std(ddof=0)
        z = ((df.close - mean) / sd).to_numpy()
        stop = 2 * ind.atr(df, 14).to_numpy()
        with np.errstate(invalid="ignore"):
            self.assert_fields(STRATEGIES["R2"].run(df), dict(
                long_mkt=z < -2, short_mkt=z > 2, exit_long_mkt=z >= 0, exit_short_mkt=z <= 0,
                stop_long_dist=stop, stop_short_dist=stop))


class TestVolatilitySpec(SpecCase):
    def test_v1(self):
        df = DATA["H1"]
        mid = df.close.rolling(20).mean()
        sd = df.close.rolling(20).std(ddof=0)
        ema = ind.ema(df.close, 20)
        a20 = ind.atr(df, 20)
        on = ((mid - 2 * sd > ema - 1.5 * a20) & (mid + 2 * sd < ema + 1.5 * a20)).to_numpy()
        fire = np.concatenate([[False], on[:-1]]) & ~on & np.isfinite(a20.to_numpy()) \
            & np.isfinite(mid.to_numpy())
        hh, ll = df.high.rolling(20).max(), df.low.rolling(20).min()
        x = (df.close - ((hh + ll) / 2 + mid) / 2).to_numpy()
        mom = np.full(len(df), np.nan)
        for i in range(38, len(df)):
            sl, ic = np.polyfit(np.arange(20), x[i - 19:i + 1], 1)
            mom[i] = ic + sl * 19
        stop = 2 * ind.atr(df, 14).to_numpy()
        with np.errstate(invalid="ignore"):
            self.assert_fields(STRATEGIES["V1"].run(df), dict(
                long_mkt=fire & (mom > 0), short_mkt=fire & (mom < 0),
                exit_long_mkt=mom < 0, exit_short_mkt=mom > 0,
                stop_long_dist=stop, stop_short_dist=stop))

    def test_v2(self):
        df = DATA["M15"]
        day = trading_day(df.index)
        g = df.groupby(day)
        daily = pd.DataFrame({"open": g.open.first(), "high": g.high.max(),
                              "low": g.low.min(), "close": g.close.last()})
        daily["atr"] = ind.atr(daily, 14)
        prev = daily.shift(1).reindex(day)
        rng = (prev.high - prev.low).to_numpy()
        ny = df.index.tz_convert("America/New_York")
        day_end = ((ny.hour == 16) & (ny.minute == 45))
        ok = np.isfinite(rng) & np.isfinite(prev.atr.to_numpy()) & ~day_end
        dopen = daily.open.reindex(day).to_numpy()
        close_utc = utc_ns(day + pd.Timedelta(hours=17), "America/New_York")
        self.assert_fields(STRATEGIES["V2"].run(df), dict(
            long_stop=np.where(ok, dopen + 0.5 * rng, NAN),
            short_stop=np.where(ok, dopen - 0.5 * rng, NAN),
            stop_long_dist=2 * prev.atr.to_numpy(), stop_short_dist=2 * prev.atr.to_numpy(),
            flat_at_close=day_end, order_expiry=close_utc, exit_deadline=close_utc))


class TestSessionStructureSpec(SpecCase):
    def _session(self, sid, df, tz, r0, r1, e1, x):
        loc = df.index.tz_convert(tz)
        mins = np.asarray(loc.hour * 60 + loc.minute)
        blen = 15 if (df.index[1] - df.index[0]) == pd.Timedelta(minutes=15) else 60
        m = lambda s_: int(s_[:2]) * 60 + int(s_[3:])
        date = loc.tz_localize(None).normalize()
        rb = (mins >= m(r0)) & (mins < m(r1))
        sub = pd.DataFrame({"d": date[rb], "h": df.high[rb].to_numpy(), "l": df.low[rb].to_numpy()})
        agg = sub.groupby("d").agg(h=("h", "max"), l=("l", "min"), n=("h", "size"))
        need = int(np.ceil(0.8 * (m(r1) - m(r0)) / blen))
        agg = agg[agg.n >= need]
        hi, lo = agg.h.reindex(date).to_numpy(), agg.l.reindex(date).to_numpy()
        live = (mins + blen >= m(r1)) & (mins + blen < m(e1)) & np.isfinite(hi) & (hi > lo)
        w = lambda v: np.where(live, v, NAN)
        flat = (mins < m(x)) & (mins + blen >= m(x))
        sig = STRATEGIES[sid].run(df)
        self.assertTrue(flat.any())
        self.assert_fields(sig, dict(
            long_stop=w(hi), short_stop=w(lo), stop_long_px=w(lo), stop_short_px=w(hi),
            target_long_px=w(2 * hi - lo), target_short_px=w(2 * lo - hi), flat_at_close=flat,
            order_expiry=utc_ns(date + pd.Timedelta(minutes=m(e1)), tz),
            exit_deadline=utc_ns(date + pd.Timedelta(minutes=m(x)), tz)))
        self.assertEqual(sig.max_entries_per_group, 1)

    def test_s1(self):
        for tf in ("M15", "H1"):
            self._session("S1", DATA[tf], "Europe/London", "00:00", "07:00", "12:00", "16:00")

    def test_s2(self):
        self._session("S2", DATA["M15"], "America/New_York", "08:00", "08:30", "11:00", "16:00")

    def test_s2_needs_both_range_bars(self):
        df = DATA["M15"]
        loc = df.index.tz_convert("America/New_York")
        target_day = loc[5000].date()
        cut = df[~((loc.date == target_day) & (loc.hour == 8) & (loc.minute == 15))]
        sig = STRATEGIES["S2"].run(cut)
        cl = cut.index.tz_convert("America/New_York")
        self.assertFalse(np.isfinite(sig.long_stop[cl.date == target_day]).any())

    def _daily_prev(self, df):
        day = trading_day(df.index)
        g = df.groupby(day)
        daily = pd.DataFrame({"h": g.high.max(), "l": g.low.min(), "c": g.close.last()})
        prev = daily.shift(1).reindex(day)
        ny = df.index.tz_convert("America/New_York")
        day_end = (ny.hour == 16) & (ny.minute == 45)
        close_utc = utc_ns(day + pd.Timedelta(hours=17), "America/New_York")
        return prev, day_end, close_utc

    def test_p1(self):
        df = DATA["M15"]
        prev, day_end, close_utc = self._daily_prev(df)
        H, L, C = prev.h.to_numpy(), prev.l.to_numpy(), prev.c.to_numpy()
        P = (H + L + C) / 3
        ok = np.isfinite(P) & ~day_end & (H > L)
        w = lambda v: np.where(ok, v, NAN)
        sig = STRATEGIES["P1"].run(df)
        self.assert_fields(sig, dict(
            short_limit=w(2 * P - L), long_limit=w(2 * P - H),
            stop_short_px=w(P + H - L), stop_long_px=w(P - (H - L)),
            target_short_px=w(P), target_long_px=w(P), flat_at_close=day_end,
            order_expiry=close_utc, exit_deadline=close_utc))
        self.assertTrue(sig.per_side)

    def test_p2(self):
        df = DATA["M15"]
        prev, day_end, close_utc = self._daily_prev(df)
        H, L = prev.h.to_numpy(), prev.l.to_numpy()
        ok = np.isfinite(H) & (H > L) & ~day_end
        w = lambda v: np.where(ok, v, NAN)
        sig = STRATEGIES["P2"].run(df)
        self.assert_fields(sig, dict(
            long_stop=w(H), short_stop=w(L), stop_long_px=w((H + L) / 2),
            stop_short_px=w((H + L) / 2), flat_at_close=day_end,
            order_expiry=close_utc, exit_deadline=close_utc))
        self.assertTrue(sig.per_side)


class TestEarlyClose(unittest.TestCase):
    """On a holiday early close the day-end bar is missing. Orders from the
    last bar must expire before the next bar opens, and positions must have a
    deadline the next bar's open reaches."""

    def test_orders_expire_and_deadline_passes(self):
        df = DATA["M15"]
        ny = df.index.tz_convert("America/New_York")
        t = df.index.as_unit("ns").asi8
        for sid in ("V2", "P1", "P2", "S1", "S2"):
            sig = STRATEGIES[sid].run(df)
            for d in EARLY_CLOSE_DAYS:
                last = np.flatnonzero((ny.date == pd.Timestamp(d).date()) & (ny.hour < 17))
                if not len(last):
                    continue
                i = last[-1]
                with self.subTest(sid=sid, day=d):
                    self.assertFalse(sig.flat_at_close[i])            # no 16:45 bar that day
                    self.assertGreaterEqual(t[i + 1], sig.order_expiry[i])
                    self.assertGreaterEqual(t[i + 1], sig.exit_deadline[i])


class TestTurtleScenarios(unittest.TestCase):
    def path(self, closes, half=0.5):
        c = np.asarray(closes, dtype=float)
        o = np.concatenate([[c[0]], c[:-1]])
        idx = pd.date_range("2020-01-01", periods=len(c), freq="D", tz="UTC")
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) + half,
                             "low": np.minimum(o, c) - half, "close": c}, index=idx)

    def won(self, closes):
        df = self.path(closes)
        return _turtle_last_breakout_won(df, 20, 10, ind.atr(df, 20).to_numpy())

    def frame(self, rows):
        o, h, l, c = map(np.array, zip(*rows))
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                            index=pd.date_range("2020-01-01", periods=len(rows), freq="D", tz="UTC"))

    def test_2n_stop_inside_the_exit_channel(self):
        # A deep wick five bars before the breakout keeps the 10-bar low at
        # 96.0, below the 2N stop (~98.2). Bar 61 hits the stop but not the
        # 10-bar exit; price then holds above the entry, so ignoring the 2N
        # rule would turn this into a winner.
        flat = (100.0, 100.5, 99.5, 100.0)
        rows = [flat] * 55 + [(100.0, 100.5, 96.0, 100.0)] + [flat] * 4
        rows += [(100.0, 101.0, 99.9, 100.8), (100.8, 100.9, 98.0, 99.0)]
        rows += [(100.9, 100.95, 100.85, 100.9)] * 19 + [(100.9, 100.95, 99.0, 99.2)] * 3
        df = self.frame(rows)
        self.assertFalse(_turtle_last_breakout_won(df, 20, 10, ind.atr(df, 20).to_numpy()).any())

    def test_short_opened_on_exit_bar_is_not_stopped_by_that_bars_high(self):
        # Bar 65 falls from 105 through the long's exit and the 20-bar low.
        # Its high came before the short's entry, so it must not stop it; the
        # short then exits as a winner on bar 76.
        rows = [(100.0, 100.5, 99.5, 100.0)] * 60
        c = 100.0
        for _ in range(5):
            rows.append((c, c + 1.5, c - 0.5, c + 1.0))
            c += 1.0
        rows.append((105.0, 105.5, 92.5, 93.0))
        rows += [(94.0, 94.5, 93.5, 94.0)] * 14
        df = self.frame(rows)
        w = _turtle_last_breakout_won(df, 20, 10, ind.atr(df, 20).to_numpy())
        self.assertFalse(w[:76].any())
        self.assertTrue(w[76:].all())

    def test_2n_adverse_is_a_loser_even_if_it_recovers(self):
        # Flat bars give N = 1. Bar 60 breaks out long (entry 100.5, 2N stop
        # 98.5); bar 61 dips to 98.0 (a loser). Price then sits above the
        # entry without a new 20-bar breakout, and the 10-bar exit on bar 81
        # would be profitable if the 2N rule were ignored.
        rows = [(100.0, 100.5, 99.5, 100.0)] * 60 + [(100.0, 101.0, 99.9, 100.8),
                                                      (100.8, 100.9, 98.0, 99.0)]
        rows += [(100.9, 100.95, 100.85, 100.9)] * 19 + [(100.9, 100.95, 100.0, 100.2)] * 3
        o, h, l, c = map(np.array, zip(*rows))
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                          index=pd.date_range("2020-01-01", periods=len(rows), freq="D", tz="UTC"))
        w = _turtle_last_breakout_won(df, 20, 10, ind.atr(df, 20).to_numpy())
        self.assertFalse(w.any())

    def test_ten_day_exit_at_a_loss_is_a_loser(self):
        # breakout at 100.5, drifts to 99.8 (less than 2N) and exits below entry
        w = self.won([100.0] * 60 + [101.0] + [100.2] * 3 + [99.6] * 12)
        self.assertFalse(w.any())

    def test_short_winner_turns_on_skip_and_short_failsafe(self):
        closes = [100.0] * 60 + list(np.arange(99.0, 84.0, -1.0)) + [95.0] * 12
        df = self.path(closes)
        sig = turtle(df)
        skip = sig.notes["skip_active"]
        i = int(np.argmax(skip))
        self.assertTrue(skip[i:].all())
        self.assertAlmostEqual(sig.short_stop[i + 2], df.low.iloc[i + 2 - 54:i + 3].min())
        self.assertAlmostEqual(sig.long_stop[i + 2], df.high.iloc[i + 2 - 54:i + 3].max())

    def test_opposite_breakout_on_the_exit_bar_is_tracked(self):
        # long loses on a bar that also breaks the 20-bar low; that short wins
        closes = [100.0] * 60 + list(np.arange(101.0, 106.0)) + [93.0] \
            + list(np.arange(92.0, 80.0, -1.0)) + [90.0] * 12
        w = self.won(closes)
        self.assertFalse(w[:78].any())
        self.assertTrue(w[79:].all())


class TestTsmomCalendar(unittest.TestCase):
    def test_holiday_month_end_falls_back_to_next_month(self):
        days = pd.DatetimeIndex(["2024-03-26", "2024-03-27", "2024-03-28",   # 29th missing
                                 "2024-04-01", "2024-04-02", "2024-04-30", "2024-05-01"])
        reb = _rebalance_bars(days)
        self.assertEqual(list(reb), [False, False, False, True, False, True, False])

    def test_every_rebalance_closes_and_reopens(self):
        sig = STRATEGIES["M1"].run(DATA["D1"])
        reb = sig.exit_long_mkt
        self.assertTrue(np.array_equal(reb, sig.exit_short_mkt))
        self.assertTrue(((sig.long_mkt | sig.short_mkt) <= reb).all())
        self.assertGreater(int((sig.long_mkt | sig.short_mkt).sum()), 12)


if __name__ == "__main__":
    unittest.main()
