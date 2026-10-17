# ORB by session: results

This is an exploratory test, run 2026-10-17 at your request, and it sits outside Phases 1–7. The code is in `research/orb_sessions.py`, the per-trade output is in `orb_trades_net.csv`, and every number below comes from the CSVs in this folder.

## Verdict

**There is no tradable edge here.** After costs, 11 of 12 symbol/session combinations lose money, and none passes the survival gate (profit factor > 1.15, ≥ 100 trades, max drawdown < 25%).
- **USTEC New York:** the only positive one, at +0.027R per trade with t = 0.96. That is indistinguishable from zero.
- **Before costs:** the best combinations make about +0.07R per trade (USDJPY Tokyo, USTEC New York, XAUUSD Tokyo). None of them clears the multiple-testing bar of |t| > 2.87 for 12 tests.
- **Costs:** they take 0.04–0.24R per trade, depending on the combination. That is more than any gross edge.

**Confidence:**
- **High** that this ORB, with these costs, loses money on this data. The pooled t-stats are −10, −10 and −5, and the losses show up in every year from 2023 to 2026.
- **Moderate** that there is no gross edge at all, because USTEC New York before costs reaches t = 2.48 on its own.

This matches the published replication cited earlier, which found NQ ORB at +0.13R before costs and about zero after.

## Setup

- **Data:** your MT5 M15 exports for XAUUSD, USDJPY, USTEC and US30, from 2023-01-02 to 2026-10-16, run as one block. **That includes the 2025+ holdout** (see PLAN.md §11).
- **Rules:** fixed, nothing tuned.
  - **Range:** the first 30 minutes of each session.
  - **Orders:** a buy stop at the range high and a sell stop at the range low. One trade per session per day.
  - **Exits:** the stop is the opposite side of the range, the target is 1× the range, and anything still open is closed at the session exit.
  - **Session times:**
    - Tokyo: 09:00–09:30 JST, exit 15:00.
    - London: 08:00–08:30, exit 16:00.
    - New York: 08:00–08:30 ET for XAUUSD and USDJPY, and 09:30–10:00 ET (the cash open) for USTEC and US30. Exit 16:00.
- **Costs:**
  - **Spread:** the ask is estimated with a conservative spread: the higher of the bar's spread and the 90th-percentile spread for that minute of the day over the previous 20 days.
  - **Slippage:** per fill, 1 pip FX, $0.20 XAU, and 1 point USTEC / 2 points US30.
  - **Commission:** $7 per lot for FX, and $7 per $100k notional for XAU and the indices.
  - **Swap:** none, because every trade closes before the 17:00 New York rollover.
- **Sizing:** 0.5% risk, sized before the fill on the planned loss. Gaps can therefore lose more than 1R.
- **Cases:**
  - **net** is the model above.
  - **stress** adds one more slippage-sized amount to the spread.
  - **gross** has no costs.

## Results per symbol and session (net unless stated)

| Symbol | Session | Trades | Win rate | Profit factor | Avg R (net) | t-stat | Max DD | CAGR | Avg R gross | Avg R stress | Passes gate |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| US30 | Tokyo | 967 | 45.1% | 0.50 | -0.237 | -8.98 | 69.4% | -26.3% | -0.001 | -0.284 | no |
| US30 | London | 965 | 46.0% | 0.64 | -0.209 | -7.07 | 65.4% | -23.6% | -0.062 | -0.240 | no |
| US30 | NewYork | 928 | 49.9% | 0.91 | -0.036 | -1.28 | 18.4% | -4.5% | +0.016 | -0.050 | no |
| USDJPY | Tokyo | 977 | 52.0% | 0.75 | -0.107 | -4.01 | 41.7% | -13.0% | +0.075 | -0.145 | no |
| USDJPY | London | 966 | 47.9% | 0.62 | -0.202 | -7.12 | 63.1% | -22.8% | -0.036 | -0.256 | no |
| USDJPY | NewYork | 976 | 49.7% | 0.66 | -0.202 | -7.08 | 64.0% | -23.0% | -0.010 | -0.241 | no |
| USTEC | Tokyo | 966 | 47.7% | 0.68 | -0.146 | -5.22 | 51.9% | -17.2% | +0.045 | -0.175 | no |
| USTEC | London | 961 | 47.9% | 0.73 | -0.138 | -4.55 | 49.4% | -16.3% | +0.001 | -0.156 | no |
| USTEC | NewYork | 957 | 53.3% | 1.06 | +0.027 | +0.96 | 10.3% | +3.2% | +0.071 | +0.015 | no |
| XAUUSD | Tokyo | 972 | 51.9% | 0.72 | -0.119 | -4.20 | 47.8% | -14.4% | +0.072 | -0.160 | no |
| XAUUSD | London | 965 | 51.5% | 0.81 | -0.085 | -2.88 | 34.6% | -10.5% | +0.043 | -0.120 | no |
| XAUUSD | NewYork | 978 | 49.2% | 0.77 | -0.106 | -3.58 | 41.6% | -13.0% | -0.013 | -0.132 | no |

## What-if: different risk per trade (your request; your plan's limit is 0.5%)

This is a backtest only. Risk size changes the equity curve but not the edge: every trade's R, the win rate and the profit factor are identical at every setting. Starting balance is $100,000, with compounding. At high settings the 30:1 (FX) and 20:1 (gold, indices) leverage caps limit the position, so the risk actually taken is lower than the setting. Without the caps, the high settings would lose more.

**Total return over the period (max drawdown in brackets)**

| Symbol | Session | 0.5% | 1% | 3% | 5% | 7% | 9% | 15% |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| US30 | Tokyo | -69% (69%) | -90% (90%) | -97% (98%) | -98% (98%) | -98% (98%) | -98% (98%) | -98% (98%) |
| US30 | London | -64% (65%) | -87% (88%) | -99% (99%) | -100% (100%) | -100% (100%) | -99% (100%) | -99% (100%) |
| US30 | NewYork | -16% (18%) | -31% (34%) | -72% (77%) | -91% (93%) | -97% (98%) | -99% (99%) | -100% (100%) |
| USDJPY | Tokyo | -41% (42%) | -66% (67%) | -94% (95%) | -98% (98%) | -99% (99%) | -99% (99%) | -99% (99%) |
| USDJPY | London | -63% (63%) | -86% (87%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USDJPY | NewYork | -63% (64%) | -86% (87%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USTEC | Tokyo | -51% (52%) | -76% (77%) | -94% (95%) | -96% (96%) | -96% (97%) | -96% (97%) | -97% (97%) |
| USTEC | London | -49% (49%) | -75% (75%) | -98% (98%) | -99% (99%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USTEC | NewYork | +13% (10%) | +25% (20%) | +57% (49%) | +63% (67%) | +19% (79%) | -42% (88%) | -86% (96%) |
| XAUUSD | Tokyo | -44% (48%) | -70% (73%) | -93% (95%) | -94% (96%) | -94% (96%) | -92% (96%) | -86% (95%) |
| XAUUSD | London | -34% (35%) | -58% (58%) | -92% (92%) | -97% (97%) | -98% (98%) | -98% (98%) | -98% (98%) |
| XAUUSD | NewYork | -41% (42%) | -66% (67%) | -96% (97%) | -99% (99%) | -99% (99%) | -100% (100%) | -100% (100%) |

**How often the leverage cap limited the position, and the median risk actually taken (all 12 combinations)**

| Risk setting | 0.5% | 1% | 3% | 5% | 7% | 9% | 15% |
|---|--:|--:|--:|--:|--:|--:|--:|
| Trades limited by the cap | 0% | 1% | 33% | 62% | 78% | 87% | 97% |
| Median risk actually taken | 0.5% | 1.0% | 3.0% | 3.9% | 3.8% | 3.8% | 3.8% |

**Reading it:**
- **Eleven combinations lose money at every risk setting.** At 3% risk and above, they lose 72% to almost 100% of the account.
- **USTEC New York** is the only one that grows: +13% at 0.5%, a peak of +63% at 5% (with a 67% drawdown), then −42% at 9% and −86% at 15%.
  - That rise-then-collapse is what happens when a tiny edge is bet too hard.
  - The edge itself (+0.027R, t = 0.96) is not statistically real, and picking "5%" after seeing these numbers would be fitting to the past.

## All symbols pooled per session (net)

The t-stat is clustered by day, because trades on the same day across symbols are correlated.

| Session | Trades | Days | Avg R (net) | t-stat (clustered by day) | Win rate |
|---|--:|--:|--:|--:|--:|
| Tokyo | 3882 | 984 | -0.152 | -10.02 | 49.2% |
| London | 3857 | 984 | -0.159 | -9.97 | 48.3% |
| NewYork | 3839 | 984 | -0.080 | -5.15 | 50.5% |

## Where the cost goes (info only; this breaks your cost rule)

These are the same trades with spread and commission but **zero slippage**. Even in this case nothing is significant: USTEC New York is +0.05R with t = 1.75.

| Symbol | Session | Avg R | t-stat | Profit factor |
|---|---|--:|--:|--:|
| US30 | Tokyo | -0.148 | -4.97 | 0.69 |
| US30 | London | -0.144 | -4.51 | 0.75 |
| US30 | NewYork | -0.009 | -0.32 | 0.97 |
| USDJPY | Tokyo | +0.002 | +0.08 | 1.00 |
| USDJPY | London | -0.106 | -3.33 | 0.79 |
| USDJPY | NewYork | -0.089 | -2.74 | 0.85 |
| USTEC | Tokyo | -0.080 | -2.66 | 0.82 |
| USTEC | London | -0.087 | -2.70 | 0.83 |
| USTEC | NewYork | +0.050 | +1.75 | 1.12 |
| XAUUSD | Tokyo | -0.011 | -0.34 | 0.97 |
| XAUUSD | London | -0.007 | -0.20 | 0.98 |
| XAUUSD | NewYork | -0.046 | -1.44 | 0.90 |

## Checks on the method

- **M15 bars versus ticks:** on simulated random-walk tick data with no spread, the M15 bar model understates the true average R by 0.006–0.008R per trade, and the M1 model by about 0.001R. Coarse bars are therefore slightly pessimistic, but the effect is tiny next to the cost drag.
- **Tests:** 43 unit tests pass on the final code. They cover exact fill prices, gaps, range boundaries, sizing, metrics, no-lookahead (truncation and perturbation), tick ground truth and M15. All 25 deliberately planted errors were caught.
- **First-bar entries:** entries on the first bar after the range do worse than later ones in 11 of 12 combinations (see `orb_entry_timing.csv`).

## Caveats

- **Raw-spread account:** your account looks like one. USDJPY shows a zero spread on 83% of M15 bars, because MT5 records each bar's minimum spread. Spread cost is therefore mostly the slippage assumption, which is shown separately above.
- **Index contract specs:** for USTEC and US30 these were not provided. R, profit factor and win rate don't depend on them. The assumed commission costs on average 0.035R per trade on USTEC and 0.053R on US30. If your broker charges none, net results improve by about that much. Even so, USTEC New York can't exceed its gross result of +0.071R (t = 2.48), which is not significant after multiple testing.
- **Scope:** one broker, one bar size and 3.8 years of data.

## Data notes

- broker server timezone: NY+7 (detected; share of FX weekly opens at Sunday 17:xx NY: USDJPY 97.0%)
- US30: 1 file(s), 89,499 bars, 2023-01-02 23:00:00+00:00 -> 2026-10-16 20:45:00+00:00, point=0.01, 0 bad rows dropped (NaN, high<low, negative spread), 0 bars dropped at DST switches, bar size M15; spread p50/p90/p99 in units of 1 price: raw 1.20/1.50/2.00, conservative 1.20/1.50/2.50; bars with zero spread 0.8%; Tokyo window raw p50/p90/p99 1.20/1.50/2.50; London window raw p50/p90/p99 1.00/1.20/2.00; NewYork window raw p50/p90/p99 0.70/1.00/2.00
- USDJPY: 1 file(s), 94,433 bars, 2023-01-01 22:00:00+00:00 -> 2026-10-16 20:45:00+00:00, point=0.001, 0 bad rows dropped (NaN, high<low, negative spread), 0 bars dropped at DST switches, bar size M15; spread p50/p90/p99 in units of 0.01 price: raw 0.00/0.10/6.60, conservative 0.00/0.10/8.76; bars with zero spread 83.1%; Tokyo window raw p50/p90/p99 0.00/0.10/0.10; London window raw p50/p90/p99 0.00/0.10/0.10; NewYork window raw p50/p90/p99 0.00/0.10/0.10
- USTEC: 1 file(s), 89,522 bars, 2023-01-02 23:00:00+00:00 -> 2026-10-16 20:45:00+00:00, point=0.01, 0 bad rows dropped (NaN, high<low, negative spread), 0 bars dropped at DST switches, bar size M15; spread p50/p90/p99 in units of 1 price: raw 1.00/2.00/2.00, conservative 1.00/2.00/2.00; bars with zero spread 0.6%; Tokyo window raw p50/p90/p99 1.00/2.00/2.00; London window raw p50/p90/p99 1.00/2.00/2.00; NewYork window raw p50/p90/p99 0.90/1.00/1.05
- XAUUSD: 1 file(s), 89,705 bars, 2023-01-02 23:00:00+00:00 -> 2026-10-16 20:45:00+00:00, point=0.01, 0 bad rows dropped (NaN, high<low, negative spread), 0 bars dropped at DST switches, bar size M15; spread p50/p90/p99 in units of 0.01 price: raw 5.00/7.00/13.00, conservative 5.00/8.10/16.00; bars with zero spread 18.1%; Tokyo window raw p50/p90/p99 5.00/7.00/13.00; London window raw p50/p90/p99 5.00/5.00/9.00; NewYork window raw p50/p90/p99 5.00/7.00/10.00
