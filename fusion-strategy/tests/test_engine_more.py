"""More engine tests: short-side prices, bid/ask trigger sides, gaps, swap,
commission, timing edges, metrics and the survival gate. Each test pins a
rule that a plausible bug would break."""
import math
import unittest

import numpy as np
import pandas as pd

from fusion.engine.backtest import run_backtest
from fusion.engine.costs import CostSpec
from fusion.engine.metrics import summarize, survives
from tests.test_engine import EUR, FLAT, SLIP, SP, START, bars, run, signals

NS15 = 15 * 60 * 10**9


class TestTriggerSides(unittest.TestCase):
    def test_buy_stop_triggers_on_ask_high(self):
        fine = bars([FLAT, (1.1000, 1.1011, 1.0999, 1.1005), FLAT])       # ask high 1.1012
        sig = signals(fine.index, long_stop=(0, 1.1012), stop_long_dist=(0, 0.003))
        self.assertEqual(len(run(sig, fine).trades), 1)

    def test_buy_limit_needs_the_ask(self):
        fine = bars([FLAT, (1.1000, 1.1001, 1.0990, 1.0995), FLAT])       # ask low 1.0991
        sig = signals(fine.index, long_limit=(0, 1.0990), stop_long_dist=(0, 0.003))
        self.assertEqual(len(run(sig, fine).trades), 0)

    def test_sell_limit_needs_the_bid(self):
        fine = bars([FLAT, (1.1000, 1.1009, 1.0999, 1.1005), FLAT])       # ask high 1.1010
        sig = signals(fine.index, short_limit=(0, 1.1010), stop_short_dist=(0, 0.003))
        self.assertEqual(len(run(sig, fine).trades), 0)

    def test_short_target_needs_the_ask(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1001, 1.0979, 1.0985), FLAT])  # ask low 1.0980
        sig = signals(fine.index, short_mkt=(0, True), stop_short_px=(0, 1.1050),
                      target_short_px=(0, 1.0979))
        self.assertNotEqual(run(sig, fine).trades.iloc[0].reason_out, "target")


class TestShortPrices(unittest.TestCase):
    def test_short_market_entry_at_bid_minus_slippage(self):
        fine = bars([FLAT, FLAT, FLAT])
        t = run(signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005)), fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.1000 - SLIP)

    def test_sell_stop_gap_fills_at_open(self):
        fine = bars([FLAT, (1.0980, 1.0981, 1.0975, 1.0978), FLAT])
        t = run(signals(fine.index, short_stop=(0, 1.0990), stop_short_dist=(0, 0.005)), fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.0980 - SLIP)

    def test_buy_limit_better_open(self):
        fine = bars([FLAT, (1.0980, 1.0981, 1.0975, 1.0978), FLAT])
        t = run(signals(fine.index, long_limit=(0, 1.0990), stop_long_dist=(0, 0.005)), fine).trades.iloc[0]
        self.assertAlmostEqual(t.entry, 1.0980 + SP + SLIP)

    def test_short_stop_gap_fills_at_ask_open(self):
        fine = bars([FLAT, FLAT, (1.1020, 1.1022, 1.1019, 1.1021), FLAT])
        sig = signals(fine.index, short_mkt=(0, True), stop_short_px=(0, 1.1010))
        t = run(sig, fine).trades.iloc[0]
        self.assertAlmostEqual(t.exit, 1.1020 + SP + SLIP)

    def test_short_signal_flat_deadline_and_period_end_exits_pay_the_ask(self):
        fine = bars([FLAT, FLAT, FLAT, (1.1010, 1.1011, 1.1009, 1.1010), FLAT])
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005), exit_short_mkt=(2, True))
        self.assertAlmostEqual(run(sig, fine).trades.iloc[0].exit, 1.1010 + SP + SLIP)
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005), flat_at_close=(3, True))
        self.assertAlmostEqual(run(sig, fine).trades.iloc[0].exit, 1.1010 + SP + SLIP)
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005),
                      exit_deadline=(0, fine.index[3].value))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "deadline")
        self.assertAlmostEqual(t.exit, 1.1010 + SP + SLIP)
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005))
        t = run(sig, fine, period=(fine.index[0], fine.index[4])).trades.iloc[0]
        self.assertEqual(t.reason_out, "period_end")
        self.assertAlmostEqual(t.exit, 1.1010 + SP + SLIP)

    def test_short_mark_to_market_at_the_ask(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1011, 1.0999, 1.1010), FLAT])
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.005), exit_short_mkt=(2, True))
        res = run(sig, fine)
        t = res.trades.iloc[0]
        self.assertAlmostEqual(res.equity.iloc[2], 100_000 + (t.entry - (1.1010 + SP)) * t.units)

    def test_short_same_bar_stop_uses_the_ask_and_is_checked(self):
        # short entry at 1.0990; bid high 1.1004 -> ask 1.1005 reaches the 15-pip stop
        fine = bars([FLAT, (1.1000, 1.1004, 1.0985, 1.0990), FLAT])
        t = run(signals(fine.index, short_stop=(0, 1.0990), stop_short_dist=(0, 0.0016)), fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "stop")
        self.assertEqual(t.t_out, fine.index[1].value + 1)


class TestSameBarAndExits(unittest.TestCase):
    def test_long_takes_the_nearer_of_target_and_exit_limit(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1040, 1.0999, 1.1030), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      target_long_px=(0, 1.1035), exit_long_limit=(1, 1.1020))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.reason_out, round(t.exit, 5)), ("exit_limit", round(1.1020 - SLIP, 5)))

    def test_reversal_needs_a_level_beyond_the_exit(self):
        # short stopped on the way up at 1.1010; a long stop at 1.1005 lies
        # behind the exit, so it cannot fill after it in this bar
        fine = bars([FLAT, FLAT, (1.1000, 1.1020, 1.0999, 1.1015), FLAT])
        sig = signals(fine.index, short_mkt=(0, True), stop_short_px=(0, 1.1010),
                      long_stop=(1, 1.1005), stop_long_dist=(1, 0.003))
        self.assertEqual(list(run(sig, fine).trades.dir), [-1])

    def test_no_limit_entry_after_an_exit_in_the_same_bar(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1015, 1.0980, 1.0985), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_px=(0, 1.0990),
                      short_limit=(1, 1.1010), stop_short_dist=(1, 0.003))
        self.assertEqual(list(run(sig, fine).trades.dir), [1])

    def test_no_limit_entry_even_further_along_the_move(self):
        # long stopped on the way down at 1.0990; a sell limit at 1.0985 lies
        # further along the move, but limit entries never fill after an exit
        fine = bars([FLAT, FLAT, (1.1000, 1.1015, 1.0980, 1.0985), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_px=(0, 1.0990),
                      short_limit=(1, 1.0985), stop_short_dist=(1, 0.003))
        self.assertEqual(list(run(sig, fine).trades.dir), [1])

    def test_no_new_entry_after_profit_taking_in_the_same_bar(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1031, 1.0999, 1.1000), FLAT])
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=[(0, 0.005), (1, 0.005)],
                      target_long_px=(0, 1.1030), long_limit=(1, 1.1000))
        self.assertEqual(len(run(sig, fine).trades), 1)

    def test_exit_stop_is_checked_on_the_entry_bar(self):
        fine = bars([FLAT, (1.1000, 1.1012, 1.0990, 1.0995), FLAT])
        sig = signals(fine.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.0100),
                      exit_long_stop=(0, 1.0994))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "exit_stop")
        self.assertAlmostEqual(t.exit, 1.0994 - SLIP)

    def test_same_bar_stop_fill_stays_inside_the_bar(self):
        wide = bars([FLAT, (1.1000, 1.1012, 1.0990, 1.0995), FLAT], sp=0.0030)
        # ask 1.1030+ at entry; a 1-pip stop lies above the bar's bid high
        sig = signals(wide.index, long_stop=(0, 1.1010), stop_long_dist=(0, 0.0001))
        t = run(sig, wide).trades
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].reason_out, "stop")
        self.assertLessEqual(t.iloc[0].exit, 1.1012 - SLIP + 1e-12)


class TestTiming(unittest.TestCase):
    def test_deadline_at_a_bar_open_exits_at_that_open(self):
        fine = bars([FLAT] * 6)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_deadline=(0, fine.index[3].value))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.reason_out, t.t_out), ("deadline", fine.index[3].value))

    def test_short_deadline_inside_a_coarse_bar_pays_the_ask(self):
        rows = [FLAT] * 6 + [(1.1010, 1.1011, 1.1009, 1.1010)] + [FLAT] * 5
        fine = bars(rows)
        h1 = fine.resample("1h").first().index
        sig = signals(h1, short_mkt=(0, True), stop_short_dist=(0, 0.005),
                      exit_deadline=(0, (h1[1] + pd.Timedelta(minutes=30)).value))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.reason_out, "deadline")
        self.assertAlmostEqual(t.exit, 1.1010 + SP + SLIP)

    def test_deadline_inside_a_coarse_bar(self):
        fine = bars([FLAT] * 12)
        h1 = fine.resample("1h").first().index
        sig = signals(h1, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_deadline=(0, (h1[1] + pd.Timedelta(minutes=30)).value))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual(t.t_out, (h1[1] + pd.Timedelta(minutes=30)).value)

    def test_expiry_equal_to_the_next_open_is_expired(self):
        fine = bars([FLAT] * 4)
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      order_expiry=(0, fine.index[1].value))
        self.assertEqual(len(run(sig, fine).trades), 0)

    def test_period_end_inside_a_daily_bar(self):
        fine = bars([FLAT] * 200, start="2024-03-04 22:00")
        d1 = pd.DatetimeIndex(["2024-03-04 22:00", "2024-03-05 22:00", "2024-03-06 22:00"], tz="UTC")
        sig = signals(d1, long_mkt=(0, True), stop_long_dist=(0, 0.005))
        end = pd.Timestamp("2024-03-06 12:00", tz="UTC")
        res = run(sig, fine, period=(d1[0], end), cost=EUR)
        t = res.trades.iloc[0]
        self.assertEqual(t.reason_out, "period_end")
        self.assertEqual(t.t_out, end.value)
        self.assertAlmostEqual(res.equity.iloc[-1] - 100_000, t.pnl_usd)


class TestSizingCostsSwap(unittest.TestCase):
    def test_lot_size_rounds_down(self):
        fine = bars([FLAT] * 3)
        t = run(signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.0026)), fine).trades.iloc[0]
        self.assertEqual(t.units, 178_000)                       # 500 / 0.0028 = 178,571

    def test_sizing_compounds_on_current_equity(self):
        fine = bars([FLAT, FLAT, (1.1000, 1.1001, 1.0950, 1.0960), FLAT, FLAT, FLAT])
        sig = signals(fine.index, long_mkt=[(0, True), (3, True)],
                      stop_long_px=[(0, 1.0960), (3, 1.0960)])
        tr = run(sig, fine).trades
        eq = 100_000 + tr.pnl_usd.iloc[0]
        self.assertLess(tr.risk_usd.iloc[1], 0.005 * eq + 1e-6)
        self.assertGreater(tr.risk_usd.iloc[1], 0.005 * eq - 1_000 * 0.0044)

    def test_xau_commission_per_notional(self):
        xau = CostSpec("XAUUSD", 100, 0.01, 0.20, 20.0, "per_notional", -30.0, -20.0, 2, 0.01)
        fine = bars([(2000.0, 2000.5, 1999.5, 2000.0)] * 4, sp=0.30)
        t = run(signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 10.0),
                        exit_long_mkt=(1, True)), fine, cost=xau).trades.iloc[0]
        self.assertAlmostEqual(t.commission, 7.0 * t.units * t.entry / 100_000)

    def test_usdjpy_leverage_cap_counts_usd_units(self):
        jpy = CostSpec("USDJPY", 100_000, 0.01, 0.01, 30.0, "per_lot", -5.0, 1.0, 2, 0.001)
        fine = bars([(150.0, 150.01, 149.99, 150.0)] * 3, sp=0.01)
        t = run(signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.001)), fine, cost=jpy).trades.iloc[0]
        self.assertTrue(t.capped)
        self.assertEqual(t.units, 3_000_000)                    # 30 x 100k USD, not / price

    def test_short_swap_and_usdjpy_conversion(self):
        jpy = CostSpec("USDJPY", 100_000, 0.01, 0.01, 30.0, "per_lot", -5.0, 2.0, 2, 0.001)
        idx = pd.date_range(START, "2024-03-06 00:30", freq="15min", tz="UTC")
        fine = bars([(150.0, 150.01, 149.99, 150.0)] * len(idx), sp=0.01)
        sig = signals(fine.index, short_mkt=(0, True), stop_short_dist=(0, 0.5),
                      exit_short_mkt=(len(idx) - 3, True))
        t = run(sig, fine, cost=jpy).trades.iloc[0]
        self.assertEqual(t.nights, 1)
        self.assertAlmostEqual(t.swap, 2.0 * 0.001 * t.units / t.exit)

    def test_entry_at_the_roll_does_not_pay_it_but_exit_at_the_roll_open_does(self):
        # 2024-03-05 17:00 New York = 22:00 UTC
        fine = bars([FLAT] * 12, start="2024-03-05 21:30")
        i = list(fine.index).index(pd.Timestamp("2024-03-05 22:00", tz="UTC"))
        sig = signals(fine.index, long_mkt=(i - 1, True), stop_long_dist=(i - 1, 0.005),
                      exit_long_mkt=(i + 3, True))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.t_in, t.nights), (fine.index[i].value, 0))
        sig = signals(fine.index, long_mkt=(0, True), stop_long_dist=(0, 0.005),
                      exit_long_mkt=(i - 1, True))
        t = run(sig, fine).trades.iloc[0]
        self.assertEqual((t.t_out, t.nights), (fine.index[i].value, 1))


class TestMetricsAndGate(unittest.TestCase):
    def frame(self, pnl, comm=0.0):
        days = pd.date_range("2024-01-02", periods=len(pnl), freq="B", tz="UTC") + pd.Timedelta(hours=15)
        tr = pd.DataFrame(dict(pnl_usd=pnl, commission=comm, swap=0.0, R=np.sign(pnl), dir=1))
        eq = pd.Series(100_000 + np.cumsum(pnl), index=days)
        return tr, eq

    def test_drawdown_from_the_starting_equity(self):
        tr, eq = self.frame([-1000.0, -1000.0, 500.0])
        m = summarize(tr, eq, 100_000, (eq.index[0], eq.index[-1]), 0.5)
        self.assertAlmostEqual(m["max_dd"], 0.02)

    def test_sharpe_cagr_and_net_profit_factor(self):
        tr, eq = self.frame([1000.0, -500.0, 2000.0, -500.0], comm=50.0)
        per = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC"))
        m = summarize(tr, eq, 100_000, per, 0.5)
        r = pd.Series([0.01, -500 / 101_000, 2000 / 100_500, -500 / 102_500])
        self.assertAlmostEqual(m["sharpe"], r.mean() / r.std() * np.sqrt(252))
        self.assertAlmostEqual(m["cagr"], 1.02 ** (365.25 / 365) - 1)
        self.assertAlmostEqual(m["profit_factor"], 3000 / 1000)          # P/L already net

    def test_gate_boundaries(self):
        base = dict(trades=100, profit_factor=1.16, max_dd=0.20)
        self.assertTrue(survives(base))
        self.assertFalse(survives({**base, "profit_factor": 1.15}))
        self.assertFalse(survives({**base, "profit_factor": 1.12}))
        self.assertFalse(survives({**base, "trades": 99}))
        self.assertFalse(survives({**base, "max_dd": 0.25}))
        self.assertFalse(survives({**base, "max_dd": 0.27}))


if __name__ == "__main__":
    unittest.main()
