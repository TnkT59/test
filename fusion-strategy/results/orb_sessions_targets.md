# ORB by session: take-profit variants

These are the same trades as `orb_sessions/`, with the stop at the opposite side of the range. Only the take-profit changes. Columns read **0.5× / 1× / 2× / partial**, where partial means half the position closes at 0.5× the range and the other half runs to 2×. Each variant has its own folder: `orb_sessions_target_0.5x/`, `orb_sessions/`, `orb_sessions_target_2x/` and `orb_sessions_partial_0.5x_2x/`. Every number below comes from those CSVs.

## Verdict

**No take-profit variant produces an edge, and no combination passes the survival gate with any of them.**
- **0.5×:** worse than 1× in 11 of 12 combinations after costs. A trade that risks 1 to win 0.5 needs about 67% wins to break even before costs.
- **2×:** better than 1× in 10 of 12 combinations, but still negative in 11. USTEC New York is +0.046R (t = 1.36).
- **Partial (½ at 0.5×, ½ at 2×):**
  - Better than 1× in 5 of 12 combinations and worse in 7.
  - Worse than plain 2× in 12 of 12 and better than plain 0.5× in 12 of 12.
  - Its result is the average of the 0.5× and 2× results to within 0.001R per combination. That is expected: with the same entry and stop, a 50/50 split is just half of each trade. So it can't beat the better of the two.
  - Best case: USTEC New York at +0.012R (t = 0.48).
- **Multiple testing:** there are now 60 ORB variants (1×, no stop, 0.5×, 2× and partial, each on 12 combinations). A result needs |t| > 3.34 to count as real, and none comes close.

## All symbols pooled per session (net; t-stat clustered by day)

| Target | Tokyo avg R (t) | London avg R (t) | New York avg R (t) |
|---|--:|--:|--:|
| 0.5× | -0.192 (-16.8) | -0.179 (-15.6) | -0.151 (-12.7) |
| 1× | -0.152 (-10.0) | -0.159 (-10.0) | -0.080 (-5.1) |
| 2× | -0.142 (-7.3) | -0.128 (-5.8) | -0.055 (-2.7) |
| ½ at 0.5× + ½ at 2× | -0.167 (-11.8) | -0.154 (-10.2) | -0.103 (-6.9) |

## Per symbol and session (0.5× / 1× / 2× / partial)

| Symbol | Session | Avg R net | t-stat (partial) | Win rate | Profit factor | Max DD | Avg R gross |
|---|---|--:|--:|--:|--:|--:|--:|
| US30 | Tokyo | -0.249 / -0.237 / -0.214 / -0.232 | -9.48 | 58% / 45% / 37% / 41% | 0.34 / 0.50 / 0.60 / 0.47 | 70% / 69% / 67% / 68% | -0.034 / -0.001 / +0.024 / -0.005 |
| US30 | London | -0.204 / -0.209 / -0.184 / -0.194 | -6.93 | 61% / 46% / 32% / 33% | 0.48 / 0.64 / 0.74 / 0.62 | 63% / 65% / 63% / 62% | -0.055 / -0.062 / -0.038 / -0.047 |
| US30 | NewYork | -0.074 / -0.036 / -0.026 / -0.050 | -1.97 | 61% / 50% / 43% / 50% | 0.77 / 0.91 / 0.94 / 0.86 | 30% / 18% / 16% / 22% | -0.023 / +0.016 / +0.024 / +0.000 |
| USDJPY | Tokyo | -0.160 / -0.107 / -0.112 / -0.136 | -5.61 | 63% / 52% / 41% / 45% | 0.53 / 0.75 / 0.77 / 0.65 | 55% / 42% / 44% / 49% | +0.012 / +0.075 / +0.070 / +0.041 |
| USDJPY | London | -0.211 / -0.202 / -0.149 / -0.180 | -6.74 | 61% / 48% / 36% / 38% | 0.45 / 0.62 / 0.75 / 0.61 | 64% / 63% / 53% / 58% | -0.045 / -0.036 / +0.026 / -0.009 |
| USDJPY | NewYork | -0.279 / -0.202 / -0.154 / -0.217 | -7.68 | 56% / 50% / 37% / 38% | 0.39 / 0.66 / 0.79 / 0.62 | 75% / 64% / 56% / 66% | -0.098 / -0.010 / +0.045 / -0.027 |
| USTEC | Tokyo | -0.163 / -0.146 / -0.120 / -0.141 | -5.55 | 62% / 48% / 39% / 43% | 0.52 / 0.68 / 0.75 / 0.64 | 55% / 52% / 49% / 50% | +0.019 / +0.045 / +0.074 / +0.046 |
| USTEC | London | -0.163 / -0.138 / -0.107 / -0.135 | -4.67 | 62% / 48% / 34% / 34% | 0.57 / 0.73 / 0.83 / 0.71 | 55% / 49% / 47% / 50% | -0.032 / +0.001 / +0.035 / +0.002 |
| USTEC | NewYork | -0.022 / +0.027 / +0.046 / +0.012 | +0.48 | 65% / 53% / 47% / 53% | 0.92 / 1.06 / 1.09 / 1.03 | 14% / 10% / 13% / 11% | +0.025 / +0.071 / +0.091 / +0.058 |
| XAUUSD | Tokyo | -0.195 / -0.119 / -0.120 / -0.159 | -5.98 | 61% / 52% / 38% / 40% | 0.42 / 0.72 / 0.78 / 0.62 | 62% / 48% / 50% / 55% | -0.018 / +0.072 / +0.072 / +0.027 |
| XAUUSD | London | -0.138 / -0.085 / -0.075 / -0.106 | -3.79 | 65% / 52% / 36% / 36% | 0.60 / 0.81 / 0.87 / 0.76 | 49% / 35% / 34% / 40% | -0.009 / +0.043 / +0.055 / +0.023 |
| XAUUSD | NewYork | -0.223 / -0.106 / -0.082 / -0.152 | -5.32 | 57% / 49% / 36% / 38% | 0.46 / 0.77 / 0.86 / 0.68 | 66% / 42% / 38% / 53% | -0.134 / -0.013 / +0.020 / -0.057 |

Note: in the partial run, one XAUUSD Tokyo trade (2026-02-02) was skipped. That day's range was $143.87, so the 0.5% risk came to 1 oz in total, which can't be split into two halves at the 1 oz minimum.
