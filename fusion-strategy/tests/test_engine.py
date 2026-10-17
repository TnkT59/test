"""Engine tests: every fill rule on hand-built bars with exact expected prices."""
import math
import unittest

import numpy as np
import pandas as pd

from fusion.engine.backtest import SwapUnknown, run_backtest
from fusion.engine.costs import CostSpec
from fusion.engine.metrics import summarize
from fusion.strategies import STRATEGIES
from fusion.strategies.base import Signals
from tests.helpers import datasets

SLIP, SP = 0.0001, 0.0001
EUR = CostSpec("EURUSD", lot=100_000, pip=0.0001, slip=SLIP, max_lev=30.0, commission="per_lot",
               swap_long=-5.0, swap_short=1.0, triple_swap_day=2, point=0.00001)
EUR_NOSWAP = CostSpec("EURUSD", 100_000, 0.0001, SLIP, 30.0, "per_lot", math.nan, math.nan, 2, 0.00001)
START = "2024-03-05 00:00"        # Tuesday


def bars(rows, start=START, freq="15min", sp=SP):
    """rows of (open, high, low, close) bid prices; flat filler for None."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    o, h, l, c = map(np.array, zip(*rows))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "spread_cons": sp}, index=idx)


FLAT = (1.1000, 1.1001, 1.0999, 1.1000)


def signals(index, **at):
    """at: field=(position, value) or field=[(pos, value), ...]."""
    n = len(index)
    kw = {}
    for name, spec in at.items():
        if name in ("max_entries_per_group", "per_side", "group"):
            kw[name] = spec
            continue
        items = spec if isinstance(spec, list) else [spec]
        if name in ("long_mkt", "short_mkt", "exit_long_mkt", "exit_short_mkt", "flat_at_close"):
            arr = np.zeros(n, dtype=bool)
        elif name in ("order_expiry", "exit_deadline"):
            arr = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        else:
            arr = np.full(n, np.nan)
        for pos, val in items:
            arr[pos] = val
        kw[name] = arr
    return Signals(index, **kw)


def run(sig, fine, cost=EUR, **kw):
    return run_backtest(sig, fine, cost, **kw)


class TestFills(unittest.TestCase):
    def test_market_entry_and_signal_exit(self):
        fine = bars([FLAT, FLAT, FLAT, (1.1010, 1.1011, 1.1009, 1.1010)])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0020),
                      exit_long_mkt=(2, True))
        t = run(sig, fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.1000 + SP + SLIP)
        self.assertAlmostEqual(t.stop, t.entry - 0.0020)
        self.assertAlmostEqual(t.exit, 1.1010 - SLIP)
        self.assertEqual((t.reason_in, t.reason_out), ("market", "signal"))
        self.assertEqual(t.units, 227_000)                      # 500 / (0.0020 + 2 slip)
        self.assertAlmostEqual(t.commission, 7 * 2.27)
        self.assertAlmostEqual(t.pnl_usd, (t.exit - t.entry) * 227_000 - 7 * 2.27)

    def test_buy_stop_level_and_gap(self):
        fine = bars([FLAT, (1.1000, 1.1012, 1.0999, 1.1010), FLAT])
        sig = signals(fine.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.0030))
        t = run(sig, fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.1010 + SLIP)          # ask 1.1013 >= level
        gap = bars([FLAT, (1.1020, 1.1022, 1.1019, 1.1020), FLAT])
        t = run(signals(gap.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.0030)), gap).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.1020 + SP + SLIP)     # gapped: ask open

    def test_sell_stop_and_limits(self):
        fine = bars([FLAT, (1.1000, 1.1001, 1.0985, 1.0990), FLAT])
        t = run(signals(fine.index, short_stop=(0, 1.0990), stop_short_dist=(0, 0.0030)), fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.0990 - SLIP)
        t = run(signals(fine.index, long_limit=(0, 1.0990), stop_long_dist=(0, 0.0030)), fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.0990 + SLIP)          # ask low 1.0986 <= 1.0990
        up = bars([FLAT, (1.1020, 1.1022, 1.1018, 1.1020), FLAT])
        t = run(signals(up.index, short_limit=(0, 1.1010), stop_short_dist=(0, 0.0030)), up).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.1020 - SLIP)          # opened above the limit

    def test_protective_stop_and_gap_through_it(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1001, 1.0975, 1.0980), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_px=(0, 1.0985))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "stop")
        self.assertAlmostEqual(t.exit, 1.0985 - SLIP)
        gap = bars([FLAT, FLAT, (1.0970, 1.0972, 1.0965, 1.0970), FLAT])
        t = run(signals(gap.index, long_mkt=(0, True), stop_long_px=(0, 1.0985)), gap).trades.iloc[0]
        self.assertAlmostEqual(t.exit, 1.0970 - SLIP)

    def test_target_and_stop_first_in_same_bar(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1031, 1.0999, 1.1020), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_px=(0, 1.0985),
                      target_long_px=(0, 1.1030))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.reason_out, round(t.exit, 5)), ("target", round(1.1030 - SLIP, 5)))
        both = bars([FLAT, FLAT, (1.1000, 1.1031, 1.0980, 1.1020), FLAT])
        t = run(signals(both.index, long_mkt=(0, True), stop_long_px=(0, 1.0985),
                        target_long_px=(0, 1.1030)), both).trades.iloc[0]
        self.assertEqual(t.reason_out, "stop")

    def test_short_exits_use_ask(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1009, 1.0990, 1.1000), (1.1000, 1.1001, 1.0979, 1.0985), FLAT])
        sig = signals(fine.index, short_mkt=(0, True), stop_short_px=(0, 1.1010),
                      target_short_px=(0, 1.0980))
        t = run(sig, fine).trades.iloc[0]
        # bar 2: ask high 1.1010 reaches the stop although the bid high is 1.1009
        self.assertEqual(t.reason_out, "stop")
        self.assertAlmostEqual(t.exit, 1.1010 + SLIP)
        fine2 = bars([FLAT, FLAT, (1.1000, 1.1001, 1.0979, 1.0985), (1.1000, 1.1001, 1.0978, 1.0985), FLAT])
        t = run(signals(fine2.index, short_mkt=(0, True), stop_short_px=(0, 1.1010),
                        target_short_px=(0, 1.0980)), fine2).trades.iloc[0]
        # bar 2: ask low 1.0980 just reaches the target
        self.assertEqual((t.reason_out, round(t.exit, 5)), ("target", round(1.0980 + SLIP, 5)))

    def test_intrabar_entry_cannot_take_profit_in_its_bar(self):
        fine = bars([FLAT, (1.1000, 1.1040, 1.0999, 1.1035), (1.1035, 1.1036, 1.1034, 1.1035)])
        sig = signals(fine.index, long_stop=[(0, 1.1010), (1, 1.1010)], stop_long_dist=[(0, .003), (1, .003)],
                      target_long_px=[(0, 1.1030), (1, 1.1030)])
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "target")
        self.assertEqual(t.t_out, fine.index[2].value + 1)       # next bar, not the entry bar

    def test_target_fills_at_level_not_at_a_better_open(self):
        # entry bar runs through the target (not allowed to take profit there);
        # the next bar opens far above: the resting target fills at its level
        fine = bars([FLAT, (1.1000, 1.1040, 1.0999, 1.1038), (1.1050, 1.1052, 1.1049, 1.1050), FLAT])
        sig = signals(fine.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.003),
                      target_long_px=(0, 1.1030))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "target")
        self.assertAlmostEqual(t.exit, 1.1030 - SLIP)

    def test_intrabar_entry_can_be_stopped_in_its_bar(self):
        fine = bars([FLAT, (1.1000, 1.1012, 1.0960, 1.0990), FLAT])
        sig = signals(fine.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.0030))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "stop")
        self.assertAlmostEqual(t.exit, t.stop - SLIP)

    def test_exit_stop_and_exit_limit(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1001, 1.0990, 1.0992), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0050),
                      exit_long_stop=(1, 1.0994))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.reason_out, round(t.exit, 5)), ("exit_stop", round(1.0994 - SLIP, 5)))
        fine = bars([FLAT, FLAT, (1.1000, 1.1006, 1.0999, 1.1004), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0050),
                      exit_long_limit=(1, 1.1005))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.reason_out, round(t.exit, 5)), ("exit_limit", round(1.1005 - SLIP, 5)))


class TestTimeRules(unittest.TestCase):
    def test_flat_at_close(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1003, 1.0999, 1.1002), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0050), flat_at_close=(2, True))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "flat")
        self.assertAlmostEqual(t.exit, 1.1002 - SLIP)
        self.assertEqual(t.t_out, fine.index[2].value + 15 * 60 * 10**9)

    def test_deadline_on_missing_bar(self):
        fine = bars([FLAT] * 6)
        fine = fine.drop(fine.index[3:5])                       # bars 3 and 4 missing
        deadline = (fine.index[0] + pd.Timedelta(minutes=50)).value
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0050),
                      exit_deadline=(0, deadline))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "deadline")
        self.assertEqual(t.t_out, fine.index[3].value)          # first bar at/after the deadline

    def test_expired_orders_are_not_honoured(self):
        fine = bars([FLAT] * 6)
        fine = fine.drop(fine.index[1:4])                        # gap after bar 0
        expiry = (fine.index[0] + pd.Timedelta(minutes=30)).value
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      order_expiry=(0, expiry))
        res = run(sig, fine)
        self.assertEqual(len(res.trades), 0)
        self.assertEqual(res.counts["expired"], 1)

    def test_one_entry_per_group_and_per_side(self):
        rows = [FLAT, (1.1000, 1.1012, 1.0999, 1.1001), (1.1000, 1.1001, 1.0990, 1.0999),
                (1.1000, 1.1012, 1.0999, 1.1001), FLAT]
        fine = bars(rows)
        lv = dict(long_stop=[(i, 1.1010) for i in range(4)], short_stop=[(i, 1.0992) for i in range(4)],
                  stop_long_dist=[(i, 0.0005) for i in range(4)], stop_short_dist=[(i, 0.0005) for i in range(4)],
                  exit_long_limit=[(i, 1.1000) for i in range(4)], exit_short_limit=[(i, 1.1000) for i in range(4)])
        one = run(signals(fine.index, group=np.zeros(5, dtype=np.int64), max_entries_per_group=1, **lv), fine)
        self.assertEqual(len(one.trades), 1)
        per = run(signals(fine.index, group=np.zeros(5, dtype=np.int64), max_entries_per_group=1,
                          per_side=True, **lv), fine)
        self.assertEqual(sorted(per.trades.dir.tolist()), [-1, 1])


class TestSameBarRules(unittest.TestCase):
    def test_nearest_level_to_open_wins(self):
        fine = bars([FLAT, (1.1008, 1.1012, 1.0985, 1.1000), FLAT])
        sig = signals(fine.index, long_stop=(0, 1.1010), short_stop=(0, 1.0990),
                      stop_long_dist=(0, 0.01), stop_short_dist=(0, 0.01))
        self.assertEqual(run(sig, fine).trades.iloc[0].dir, 1)

    def test_fill_beyond_stop_cancels_and_beyond_target_drops_target(self):
        gap = bars([FLAT, (1.0970, 1.0972, 1.0969, 1.0970), FLAT])
        sig = signals(gap.index, long_limit=(0, 1.0990), stop_long_px=(0, 1.0980))
        res = run(sig, gap)
        self.assertEqual((len(res.trades), res.counts["cancelled_beyond_stop"]), (0, 1))
        up = bars([FLAT, (1.1040, 1.1042, 1.1039, 1.1040), FLAT, FLAT])
        sig = signals(up.index, long_stop=(0, 1.1010), stop_long_px=(0, 1.0990),
                      target_long_px=(0, 1.1030))
        t = run(sig, up).trades.iloc[0]
        self.assertTrue(np.isnan(t.target))
        self.assertEqual(t.reason_out, "period_end")

    def test_opposite_stop_entry_after_exit_in_same_bar(self):
        # long from bar 0's order; on bar 2 price falls through the 10-bar exit
        # (1.0994) and then the 20-bar low (1.0990): exit, then short
        rows = [FLAT, FLAT, (1.1000, 1.1001, 1.0980, 1.0985), FLAT]
        fine = bars(rows)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_long_stop=(1, 1.0994), short_stop=(1, 1.0990),
                      stop_short_dist=(1, 0.0030))
        tr = run(sig, fine).trades
        self.assertEqual(list(tr.dir), [1, -1])
        self.assertAlmostEqual(tr.iloc[1].entry, 1.0990 - SLIP)
        # the bar's high came before the short: not a same-bar stop
        self.assertNotEqual(tr.iloc[1].reason_out, "stop")

    def test_no_opposite_entry_if_level_not_beyond_exit(self):
        rows = [FLAT, FLAT, (1.1000, 1.1001, 1.0980, 1.0985), FLAT]
        fine = bars(rows)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_long_stop=(1, 1.0990), short_stop=(1, 1.0994),
                      stop_short_dist=(1, 0.0030))
        self.assertEqual(list(run(sig, fine).trades.dir), [1])

    def test_same_direction_signal_is_ignored_while_in_position(self):
        fine = bars([FLAT] * 5)
        sig = signals(fine.index, long_mkt=[(0, True), (1, True), (2, True)],
                      stop_long_dist=[(0, 0.005), (1, 0.005), (2, 0.005)])
        self.assertEqual(len(run(sig, fine).trades), 1)


class TestSizingAndCosts(unittest.TestCase):
    def test_leverage_cap_and_too_small(self):
        # planned loss 0.1 + 2 x 1 pip = 2.1 pips -> 2.38M units, 2.62M notional;
        # a 10:1 cap allows 1M of notional
        lev10 = CostSpec("EURUSD", 100_000, 0.0001, SLIP, 10.0, "per_lot", -5.0, 1.0, 2, 0.00001)
        fine = bars([FLAT, FLAT, FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.00001))
        t = run(sig, fine, cost=lev10).trades.iloc[0]
        self.assertTrue(t.capped)
        self.assertLessEqual(t.units * t.entry, 10 * 100_000 + 1e-6)
        self.assertGreater(t.units * t.entry, 10 * 100_000 - 1_000 * t.entry - 1e-6)
        res = run(sig, fine, start_equity=10.0)
        self.assertEqual((len(res.trades), res.counts["too_small"]), (0, 1))

    def test_usdjpy_converts_to_usd(self):
        jpy = CostSpec("USDJPY", 100_000, 0.01, 0.01, 30.0, "per_lot", -5.0, 1.0, 2, 0.001)
        fine = bars([(150.0, 150.01, 149.99, 150.0)] * 2 + [(150.5, 150.51, 150.49, 150.5)] * 2, sp=0.01)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.20), exit_long_mkt=(2, True))
        t = run(sig, fine, cost=jpy).trades.iloc[0]
        plan_usd = (0.20 + 0.02) / t.entry
        self.assertEqual(t.units, math.floor(500 / plan_usd / 1000) * 1000)
        self.assertAlmostEqual(t.pnl_usd, (t.exit - t.entry) * t.units / t.exit - 7 * t.lots)

    def test_swap_nights_and_unknown_swap(self):
        # long from Tuesday 00:15 UTC to Thursday 00:15 UTC: rolls on Tuesday
        # and Wednesday 17:00 New York; Wednesday counts three times -> 4 nights
        idx = pd.date_range(START, "2024-03-07 00:30", freq="15min", tz="UTC")
        fine = bars([FLAT] * len(idx))
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_long_mkt=(len(idx) - 3, True))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.nights, 4)
        self.assertAlmostEqual(t.swap, -5.0 * 0.00001 * t.units * 4)
        with self.assertRaises(SwapUnknown):
            run(sig, fine, cost=EUR_NOSWAP)

    def test_closing_at_the_roll_pays_no_swap(self):
        # 17:00 New York on 2024-03-05 = 22:00 UTC (EST); flat at the close of
        # the 21:45 UTC bar, so the position is out at the roll
        start = pd.Timestamp("2024-03-05 21:00", tz="UTC")
        fine = bars([FLAT] * 8, start=str(start.tz_localize(None)))
        i = list(fine.index).index(pd.Timestamp("2024-03-05 21:45", tz="UTC"))
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005), flat_at_close=(i, True))
        t = run(sig, fine, cost=EUR_NOSWAP).trades.iloc[0]
        self.assertEqual((t.nights, t.swap), (0, 0.0))

    def test_mark_to_market_drawdown_counts_open_losses(self):
        rows = [FLAT, FLAT, (1.1000, 1.1001, 1.0950, 1.0960), (1.0960, 1.1010, 1.0959, 1.1005), FLAT]
        fine = bars(rows)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0100),
                      exit_long_mkt=(3, True))
        res = run(sig, fine)
        t = res.trades.iloc[0]
        trough = 100_000 + (1.0960 - t.entry) * t.units
        self.assertAlmostEqual(res.equity.min(), trough)
        m = summarize(res.trades, res.equity, 100_000, (fine.index[0], fine.index[-1]), res.exposure)
        self.assertAlmostEqual(m["max_dd"], (100_000 - trough) / 100_000)


class TestCoarseOnFine(unittest.TestCase):
    """H1 signals resolved on the M15 bars inside each hour."""

    def test_order_of_events_inside_the_hour(self):
        m15 = [FLAT] * 4 + [(1.1000, 1.1031, 1.0999, 1.1030), (1.1030, 1.1031, 1.0980, 1.0985)] + [FLAT] * 2
        fine = bars(m15)
        h1 = fine.resample("1h").first().index
        sig = signals(h1, long_mkt=(0, True), stop_long_px=(0, 1.0985), target_long_px=(0, 1.1030))
        self.assertEqual(run(sig, fine).trades.iloc[0].reason_out, "target")   # 2nd M15 bar first
        m15b = [FLAT] * 4 + [(1.1000, 1.1001, 1.0980, 1.0985), (1.0985, 1.1031, 1.0984, 1.1030)] + [FLAT] * 2
        fine_b = bars(m15b)
        self.assertEqual(run(sig, fine_b).trades.iloc[0].reason_out, "stop")


class TestPeriodAndCausality(unittest.TestCase):
    def test_period_bounds(self):
        fine = bars([FLAT] * 8)
        sig = signals(fine.index, long_mkt=[(0, True), (4, True)], stop_long_dist=[(0, .005), (4, .005)])
        res = run(sig, fine, period=(fine.index[3], fine.index[7]))
        t = res.trades
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].t_in, fine.index[5].value)
        self.assertEqual(t.iloc[0].reason_out, "period_end")
        self.assertEqual(t.iloc[0].t_out, fine.index[6].value + 15 * 60 * 10**9)

    def test_trades_before_a_cut_do_not_depend_on_later_data(self):
        df = datasets()["H1"]
        fine = datasets()["M15"].assign(spread_cons=0.00008)
        spec = STRATEGIES["T1"]
        full = run(spec.run(df), fine, allow_missing_swap=True).trades
        for k in (len(df) // 2, (3 * len(df)) // 4):
            cut = df.index[k]
            part = run(spec.run(df[df.index < cut]), fine[fine.index < cut],
                       allow_missing_swap=True).trades
            done = full[full.t_out < cut.value].reset_index(drop=True)
            part = part[part.reason_out != "period_end"].reset_index(drop=True)
            part = part[part.t_out < cut.value].reset_index(drop=True)
            pd.testing.assert_frame_equal(done, part)

    def test_every_registry_strategy_runs_with_stops_and_risk(self):
        data = datasets()
        fine = data["M15"].assign(spread_cons=0.00008)
        for sid, spec in STRATEGIES.items():
            tf = "D1" if sid == "M1" else ("M15" if "M15" in spec.timeframes and sid in ("S2",) else
                                           ("H1" if "H1" in spec.timeframes else spec.timeframes[0]))
            if tf == "D1":
                continue        # D1 synthetic data is a different series from the M15 one
            with self.subTest(sid=sid, tf=tf):
                res = run(spec.run(data[tf]), fine, allow_missing_swap=True)
                tr = res.trades
                self.assertGreater(len(tr), 0)
                self.assertTrue(np.isfinite(tr.stop).all())
                # risk never above 0.5% of equity at entry (sized before the fill)
                eq_before = 100_000 + tr.pnl_usd.cumsum().shift(fill_value=0)
                self.assertTrue((tr.risk_usd <= 0.005 * eq_before + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
