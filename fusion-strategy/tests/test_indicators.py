import unittest

import numpy as np
import pandas as pd

from fusion import indicators as ind
from tests.helpers import fx_hours, random_walk


def frame(n=300, seed=1):
    return random_walk(fx_hours("2024-01-01", 10)[:n], seed=seed)


class TestSmoothing(unittest.TestCase):
    def test_wilder_is_turtle_n_formula(self):
        df = frame()
        tr = ind.true_range(df).to_numpy()
        n = ind.atr(df, 20).to_numpy()
        self.assertTrue(np.isnan(n[18]))
        self.assertAlmostEqual(n[19], tr[:20].mean())
        for t in range(20, 60):
            self.assertAlmostEqual(n[t], (19 * n[t - 1] + tr[t]) / 20)

    def test_ema_seed_and_alpha(self):
        x = pd.Series(np.arange(1.0, 31.0))
        e = ind.ema(x, 10).to_numpy()
        self.assertAlmostEqual(e[9], np.mean(np.arange(1.0, 11.0)))
        a = 2 / 11
        self.assertAlmostEqual(e[10], e[9] + a * (11.0 - e[9]))

    def test_seed_skips_leading_gaps(self):
        x = pd.Series([np.nan, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
        w = ind.wilder(x, 3).to_numpy()
        self.assertAlmostEqual(w[5], 3.0)
        self.assertTrue(np.isnan(w[4]))


class TestOscillators(unittest.TestCase):
    def test_rsi_matches_naive_wilder(self):
        close = frame().close
        d = close.diff().to_numpy()
        g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
        ag, al = g[1:15].mean(), l[1:15].mean()
        want = [100 - 100 / (1 + ag / al)]
        for t in range(15, 40):
            ag = (13 * ag + g[t]) / 14
            al = (13 * al + l[t]) / 14
            want.append(100 - 100 / (1 + ag / al))
        got = ind.rsi(close, 14).to_numpy()[14:40]
        self.assertTrue(np.allclose(got, want))

    def test_rsi_all_gains_is_100(self):
        r = ind.rsi(pd.Series(np.arange(1.0, 40.0)), 14)
        self.assertTrue((r.dropna() == 100).all())

    def test_rsi_flat_is_undefined(self):
        r = ind.rsi(pd.Series(np.full(40, 1.2345)), 14)
        self.assertTrue(r.isna().all())

    def test_adx_matches_wilder_running_sums(self):
        self._adx_vs_running_sums(frame(400, seed=3))

    def test_adx_equal_moves_give_no_directional_movement(self):
        # bars whose high rises exactly as much as the low falls: Wilder
        # counts neither +DM nor -DM
        df = frame(400, seed=4).copy()
        h, l = df.high.to_numpy().copy(), df.low.to_numpy().copy()
        for t in range(30, 400, 5):
            h[t], l[t] = h[t - 1] + 0.0004, l[t - 1] - 0.0004
            h[t] = max(h[t], df.open.iloc[t], df.close.iloc[t])
            l[t] = min(l[t], df.open.iloc[t], df.close.iloc[t])
            if h[t] - h[t - 1] != l[t - 1] - l[t]:
                d = max(h[t] - h[t - 1], l[t - 1] - l[t])
                h[t], l[t] = h[t - 1] + d, l[t - 1] - d
        df["high"], df["low"] = h, l
        up, dn = np.diff(h), -np.diff(l)
        self.assertGreater(int(((up == dn) & (up > 0)).sum()), 50)
        self._adx_vs_running_sums(df)

    def _adx_vs_running_sums(self, df):
        h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
        n = 14
        tr = np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])
        up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr14, p14, m14 = tr[:n].sum(), pdm[:n].sum(), mdm[:n].sum()
        dxs, pdis = [], []
        for t in range(n, len(tr) + 1):
            if t > n:
                tr14 = tr14 - tr14 / n + tr[t - 1]
                p14 = p14 - p14 / n + pdm[t - 1]
                m14 = m14 - m14 / n + mdm[t - 1]
            pdi, mdi = 100 * p14 / tr14, 100 * m14 / tr14
            pdis.append(pdi)
            dxs.append(100 * abs(pdi - mdi) / (pdi + mdi))
        adx = [np.mean(dxs[:n])]
        for x in dxs[n:]:
            adx.append((adx[-1] * (n - 1) + x) / n)
        got = ind.dmi(df, n)
        self.assertTrue(np.allclose(got.plus_di.to_numpy()[n:n + 50], pdis[:50]))
        self.assertTrue(np.allclose(got.adx.to_numpy()[2 * n - 1:2 * n + 49], adx[:50]))

    def test_macd_lines(self):
        close = frame().close
        m = ind.macd(close)
        self.assertTrue(np.allclose((ind.ema(close, 12) - ind.ema(close, 26)).dropna(),
                                    m.macd.dropna()))
        self.assertTrue(np.allclose(m["hist"].dropna(), (m.macd - m.signal).dropna()))


class TestBands(unittest.TestCase):
    def test_bollinger_uses_population_sd(self):
        close = frame().close
        bb = ind.bollinger(close, 20, 2.0)
        w = close.iloc[30:50].to_numpy()
        self.assertAlmostEqual(bb.upper.iloc[49], w.mean() + 2 * w.std(ddof=0))

    def test_keltner(self):
        df = frame()
        kc = ind.keltner(df, 20, 1.5)
        self.assertTrue(np.allclose((kc.upper - kc.mid).dropna(), 1.5 * ind.atr(df, 20).dropna()
                                    .reindex((kc.upper - kc.mid).dropna().index)))

    def test_linreg_value_matches_polyfit(self):
        x = frame().close
        got = ind.linreg_value(x, 20).to_numpy()
        for i in (19, 57, 120):
            w = x.to_numpy()[i - 19:i + 1]
            slope, icpt = np.polyfit(np.arange(20), w, 1)
            self.assertAlmostEqual(got[i], icpt + slope * 19)

    def test_donchian_includes_current_bar(self):
        h = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        self.assertEqual(list(ind.donchian_high(h, 3).dropna()), [3.0, 5.0, 5.0])

    def test_zscore(self):
        close = frame().close
        z = ind.zscore(close, 20)
        w = close.iloc[80:100].to_numpy()
        self.assertAlmostEqual(z.iloc[99], (w[-1] - w.mean()) / w.std(ddof=0))


if __name__ == "__main__":
    unittest.main()
