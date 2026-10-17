# ORB by session: target 0.5× vs 1× vs 2× the range

These are the same trades as `orb_sessions/` with the stop at the opposite side of the range. Only the take-profit distance changes: 0.5×, 1× (the original rule) or 2× the range. Results for 0.5× are in `orb_sessions_target_0.5x/` and for 2× in `orb_sessions_target_2x/`. Every number below comes from those CSVs.

## Verdict

**Neither target produces an edge.** No combination passes the survival gate with either target.
- **0.5× target:** worse than 1× in 11 of 12 combinations after costs.
  - The win rate rises to 56–65%, but before costs a trade that risks 1 and wins 0.5 needs about 67% wins just to break even.
  - The costs are the same per trade while the win is halved, so costs take twice the share.
- **2× target:** better than 1× in 10 of 12 combinations after costs, but still negative in 11.
  - The only positive one is USTEC New York, at +0.046R (t = 1.36).
  - The best result before costs is also USTEC New York, at +0.091R (t = 2.60).
- **Multiple testing:** there are now 48 ORB variants (1×, no stop, 0.5× and 2×, each on 12 combinations). A result needs |t| > 3.28 to count as real, and none comes close.

## All symbols pooled per session (net; t-stat clustered by day)

| Target | Tokyo avg R (t) | London avg R (t) | New York avg R (t) |
|---|--:|--:|--:|
| 0.5× | -0.192 (-16.8) | -0.179 (-15.6) | -0.151 (-12.7) |
| 1× | -0.152 (-10.0) | -0.159 (-10.0) | -0.080 (-5.1) |
| 2× | -0.142 (-7.3) | -0.128 (-5.8) | -0.055 (-2.7) |

## Per symbol and session

| Symbol | Session | Avg R net 0.5× / 1× / 2× | t-stat 2× | Win rate 0.5× / 1× / 2× | Profit factor 0.5× / 1× / 2× | Max DD 0.5× / 1× / 2× | Avg R gross 0.5× / 1× / 2× |
|---|---|--:|--:|--:|--:|--:|--:|
| US30 | Tokyo | -0.249 / -0.237 / -0.214 | -6.55 | 58% / 45% / 37% | 0.34 / 0.50 / 0.60 | 70% / 69% / 67% | -0.034 / -0.001 / +0.024 |
| US30 | London | -0.204 / -0.209 / -0.184 | -4.48 | 61% / 46% / 32% | 0.48 / 0.64 / 0.74 | 63% / 65% / 63% | -0.055 / -0.062 / -0.038 |
| US30 | NewYork | -0.074 / -0.036 / -0.026 | -0.78 | 61% / 50% / 43% | 0.77 / 0.91 / 0.94 | 30% / 18% / 16% | -0.023 / +0.016 / +0.024 |
| USDJPY | Tokyo | -0.160 / -0.107 / -0.112 | -3.33 | 63% / 52% / 41% | 0.53 / 0.75 / 0.77 | 55% / 42% / 44% | +0.012 / +0.075 / +0.070 |
| USDJPY | London | -0.211 / -0.202 / -0.149 | -3.86 | 61% / 48% / 36% | 0.45 / 0.62 / 0.75 | 64% / 63% / 53% | -0.045 / -0.036 / +0.026 |
| USDJPY | NewYork | -0.279 / -0.202 / -0.154 | -3.98 | 56% / 50% / 37% | 0.39 / 0.66 / 0.79 | 75% / 64% / 56% | -0.098 / -0.010 / +0.045 |
| USTEC | Tokyo | -0.163 / -0.146 / -0.120 | -3.44 | 62% / 48% / 39% | 0.52 / 0.68 / 0.75 | 55% / 52% / 49% | +0.019 / +0.045 / +0.074 |
| USTEC | London | -0.163 / -0.138 / -0.107 | -2.52 | 62% / 48% / 34% | 0.57 / 0.73 / 0.83 | 55% / 49% / 47% | -0.032 / +0.001 / +0.035 |
| USTEC | NewYork | -0.022 / +0.027 / +0.046 | +1.36 | 65% / 53% / 47% | 0.92 / 1.06 / 1.09 | 14% / 10% / 13% | +0.025 / +0.071 / +0.091 |
| XAUUSD | Tokyo | -0.195 / -0.119 / -0.120 | -3.17 | 61% / 52% / 38% | 0.42 / 0.72 / 0.78 | 62% / 48% / 50% | -0.018 / +0.072 / +0.072 |
| XAUUSD | London | -0.138 / -0.085 / -0.075 | -1.80 | 65% / 52% / 36% | 0.60 / 0.81 / 0.87 | 49% / 35% / 34% | -0.009 / +0.043 / +0.055 |
| XAUUSD | NewYork | -0.223 / -0.106 / -0.082 | -2.04 | 57% / 49% / 36% | 0.46 / 0.77 / 0.86 | 66% / 42% / 38% | -0.134 / -0.013 / +0.020 |
