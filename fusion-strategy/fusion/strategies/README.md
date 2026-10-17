# Strategy library (Phase 2)

This file is generated from `fusion/strategies/__init__.py`, and the exact rules are in each module's docstrings. Every strategy places its orders at the close of a bar for the next bar only. Every entry carries a stop.

| ID | Strategy | Timeframes | Source | Deviations from the original / PLAN.md |
|---|---|---|---|---|
| T1 | MA cross 50/200 | M15, H1, H4, D1 | Golden/death cross (convention) | • protective stop 2 x ATR(14) added (hard rule; the original has no stop) |
| T2 | Turtle System 1 | M15, H1, H4, D1 | The Original Turtle Trading Rules (Faith et al., 2003) | • no pyramiding (units at 1/2 N)<br>• 55-day failsafe entry when a System 1 signal is skipped: in the original rules, missing from the PLAN.md T2 row<br>• a losing breakout is judged by the sign of the hypothetical trade's P/L at its 2N stop or 10-day exit<br>• the skip tracker resolves hypothetical trades on the strategy's own bars (stop first; nearer level first when both are crossed in one bar), so it can disagree with the engine's finer fills |
| T3 | Turtle System 2 | M15, H1, H4, D1 | The Original Turtle Trading Rules (Faith et al., 2003) | • no pyramiding (units at 1/2 N) |
| T4 | ADX / DI cross | M15, H1, H4, D1 | Wilder (1978); ADX > 25 threshold (convention) | • protective stop 2 x ATR(14) added (hard rule; the original has no stop); Wilder's extreme-point stop not modelled |
| M1 | Time-series momentum 12m | D1 | Moskowitz, Ooi & Pedersen (2012), JFE | • protective stop 2 x ATR(14) added (hard rule; the original has no stop)<br>• sized by 0.5% risk, not 40% target volatility<br>• closed and reopened every month (one round trip a month)<br>• if the last weekday of a month has no bar, the decision is made on the first bar of the next month |
| M2 | MACD 12/26/9 | M15, H1, H4, D1 | Appel; default 12/26/9 | • protective stop 2 x ATR(14) added (hard rule; the original has no stop) |
| R1 | Bollinger 20/2 + RSI 14 | M15, H1, H4, D1 | Bollinger (20, 2); Wilder RSI (14, 30/70); combination (convention) | none |
| R2 | Z-score 20, |z| > 2 | M15, H1, H4, D1 | (convention) | none |
| V1 | Squeeze BB(20,2) in KC(20,1.5) | M15, H1, H4, D1 | Carter, Mastering the Trade (TTM Squeeze) | • momentum histogram is LazyBear's open-source formula; Carter's is not public |
| V2 | Volatility breakout open +/- 0.5 x prior range | M15, H1 | Larry Williams-style; k = 0.5 (convention) | • stop 2 x prior-day daily ATR(14)<br>• earliest entry on the day's second bar (the day's open must be known) |
| S1 | Asian range -> London breakout | M15, H1 | (convention) | • no trade on a day with fewer than 80% of the range bars (data gaps) |
| S2 | New York opening range 08:00-08:30 | M15 | (convention) | • no trade on a day with fewer than 80% of the range bars (data gaps) |
| P1 | Floor-pivot reversion R1/S1 | M15, H1 | Floor-trader pivots; trading rule (convention) | • flat at 17:00 New York (time exit not in PLAN.md)<br>• one trade per side per day (not in PLAN.md) |
| P2 | Prior-day high/low breakout | M15, H1 | (convention) | none |

Strategy-timeframe cells: 42. Excluded: ICT / smart-money concepts (see PLAN.md §4.7).
