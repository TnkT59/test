# ORB by session without a stop loss: what-if

**This breaks your own hard rule** ("NEVER … any position without a stop loss"), so it's a backtest what-if only. It can't go into the fusion strategy, the EA or the demo runner unless you change that rule.

## Setup

The trades are identical to `../orb_sessions/`: same data, entries, target, session exit, costs and position size. The only change is that **the stop is never triggered**. Position size is still set as if the stop were at the other side of the range, so R is measured on that distance and a trade can lose far more than 1R.

## Verdict

**Removing the stop doesn't create an edge. It swaps many small losses for a few very large ones.**
- **The gate:** all 12 combinations still fail. 11 lose money after costs.
- **USTEC New York:** the only positive one, at +0.047R with t = 1.30, which is not significant.
- **Win rates:** they rise sharply, for example from 46% to 77% on US30 London.
- **Losses:** 19.3% of trades now lose more than 1R, 10.2% more than 2R and 5.6% more than 3R. The worst trade loses 40R, which is 20% of the account at 0.5% risk.
- **Before costs:** the best is XAUUSD Tokyo at +0.154R (t = 2.91). With 24 variants tested (12 with a stop, 12 without), a result needs |t| > 3.08, so it doesn't count. After costs it is −0.033R.

## Per symbol and session (net; arrows show with stop → without stop)

| Symbol | Session | Win rate (stop → none) | Profit factor | Avg R net | t-stat (no stop) | Worst trade | Max DD | Avg R gross (no stop) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| US30 | Tokyo | 45% → 54% | 0.50 → 0.55 | -0.237 → -0.215 | -6.07 | -1.3R → -5.4R | 69% → 66% | +0.032 (t +0.82) |
| US30 | London | 46% → 77% | 0.64 → 0.72 | -0.209 → -0.215 | -2.98 | -2.1R → -14.2R | 65% → 70% | -0.064 (t -0.81) |
| US30 | NewYork | 50% → 56% | 0.91 → 0.91 | -0.036 → -0.036 | -0.98 | -1.1R → -9.1R | 18% → 26% | +0.016 (t +0.43) |
| USDJPY | Tokyo | 52% → 61% | 0.75 → 0.79 | -0.107 → -0.098 | -2.57 | -1.2R → -7.9R | 42% → 41% | +0.086 (t +1.96) |
| USDJPY | London | 48% → 68% | 0.62 → 0.60 | -0.202 → -0.324 | -4.45 | -1.3R → -40.4R | 63% → 82% | -0.185 (t -2.15) |
| USDJPY | NewYork | 50% → 73% | 0.66 → 0.82 | -0.202 → -0.113 | -2.05 | -4.8R → -17.3R | 64% → 53% | +0.088 (t +1.35) |
| USTEC | Tokyo | 48% → 57% | 0.68 → 0.72 | -0.146 → -0.143 | -3.55 | -1.3R → -8.9R | 52% → 52% | +0.044 (t +1.03) |
| USTEC | London | 48% → 78% | 0.73 → 0.88 | -0.138 → -0.072 | -1.08 | -1.2R → -15.5R | 49% → 44% | +0.076 (t +1.12) |
| USTEC | NewYork | 53% → 60% | 1.06 → 1.11 | +0.027 → +0.047 | +1.30 | -1.1R → -8.8R | 10% → 12% | +0.090 (t +2.43) |
| XAUUSD | Tokyo | 52% → 72% | 0.72 → 0.93 | -0.119 → -0.033 | -0.72 | -1.1R → -10.5R | 48% → 23% | +0.154 (t +2.91) |
| XAUUSD | London | 52% → 77% | 0.81 → 0.95 | -0.085 → -0.019 | -0.34 | -1.1R → -10.9R | 35% → 32% | +0.123 (t +1.99) |
| XAUUSD | NewYork | 49% → 70% | 0.77 → 0.80 | -0.106 → -0.110 | -1.90 | -2.3R → -14.7R | 42% → 48% | -0.019 (t -0.30) |

## All symbols pooled per session (net, no stop)

| Session | Trades | Avg R net | t-stat (clustered by day) | Win rate |
|---|--:|--:|--:|--:|
| Tokyo | 3882 | -0.122 | -5.64 | 61.3% |
| London | 3857 | -0.158 | -4.29 | 74.8% |
| NewYork | 3839 | -0.054 | -2.12 | 64.8% |

## The tail

Distribution of net R per trade without a stop:
- median +0.66R;
- 5th percentile -3.21R;
- 1st percentile -6.66R;
- 0.1th percentile -12.34R.

**Worst trade:** USDJPY London on 2026-07-30. The opening range was 9.2 pips. It went long at 163.66, and USDJPY then fell about 4.4 yen over six consecutive M15 bars from 14:30 London time. The trade exited at 159.27 at the session end, for −40.4R, or 20% of the account at 0.5% risk. With the stop, the same trade was stopped out at 163.55 for −1.1R. I checked the bars, and the move runs continuously across consecutive bars, so it is not a single bad tick. I can't confirm the news behind it.

## Risk what-if without a stop: total return (max drawdown)

| Symbol | Session | 0.5% | 1% | 3% | 5% | 7% | 9% | 15% |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| US30 | Tokyo | -65% (66%) | -88% (88%) | -97% (97%) | -97% (98%) | -97% (98%) | -97% (98%) | -98% (98%) |
| US30 | London | -67% (70%) | -90% (92%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| US30 | NewYork | -17% (26%) | -33% (46%) | -79% (88%) | -97% (98%) | -99% (100%) | -100% (100%) | -100% (100%) |
| USDJPY | Tokyo | -39% (41%) | -64% (66%) | -95% (96%) | -98% (98%) | -99% (99%) | -99% (99%) | -99% (99%) |
| USDJPY | London | -80% (82%) | -97% (97%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USDJPY | NewYork | -44% (53%) | -71% (79%) | -98% (99%) | -99% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USTEC | Tokyo | -51% (52%) | -77% (78%) | -96% (96%) | -97% (98%) | -98% (98%) | -98% (99%) | -99% (99%) |
| USTEC | London | -33% (44%) | -59% (71%) | -97% (99%) | -99% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |
| USTEC | NewYork | +23% (12%) | +48% (23%) | +118% (58%) | +102% (80%) | -2% (90%) | -76% (96%) | -99% (99%) |
| XAUUSD | Tokyo | -17% (23%) | -35% (42%) | -72% (78%) | -84% (88%) | -90% (93%) | -90% (93%) | -85% (93%) |
| XAUUSD | London | -13% (32%) | -29% (55%) | -88% (95%) | -98% (99%) | -98% (99%) | -99% (99%) | -99% (100%) |
| XAUUSD | NewYork | -44% (48%) | -71% (76%) | -98% (99%) | -100% (100%) | -100% (100%) | -100% (100%) | -100% (100%) |

As with the stop, picking the best risk setting from this table would be fitting to the past. The only combination that grows (USTEC New York) has an edge that isn't statistically real.
