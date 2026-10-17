# Fusion Strategy: Plan (Phase 0)

Status: **Approved 2026-10-14 with all defaults (Q1–Q7).** Phase 1 has not run: Dukascopy blocks this container, and the paid S3 route was declined. See §11.

Instruments: XAUUSD, EURUSD, GBPUSD, USDJPY. Timeframes: M15, H1, H4, D1.
Execution targets: Python research and backtest first, then the MQL5 EA, then a demo runner. **No live account, ever.**

---

## 0. Open questions (please answer when you approve)

Items marked **[DEFAULT]** are what I will do if you just say "approved".

| # | Question | Why it matters | Default |
|---|---|---|---|
| Q1 | **Where does the code run?** This session is a Linux cloud container, so `C:\Trading\fusion-strategy\` does not exist here. The `MetaTrader5` pip package is Windows-only (pip finds no distribution here), and the MT5 Strategy Tester needs Windows. Dukascopy's datafeed answered **HTTP 429 "Too Many Requests"** to this container, so Phase 1 cannot download from here today. | Phases 1, 6 and 7 need a Windows machine with MT5, and probably your network for Dukascopy. | **[DEFAULT]** I write all the code in this repo under `fusion-strategy/`, a mirror of the target layout. You clone it to `C:\Trading\fusion-strategy\` and run the phase scripts there. I only report numbers you send back to me, or numbers produced by runs I can execute myself. |
| Q2 | Is the slippage (1 pip FX, $0.20 XAU) charged **per side** or **per round turn**? | Per side doubles the cost. | **[DEFAULT]** Per side, which is the more conservative reading. |
| Q3 | What is the XAU commission "equivalent"? | It's ambiguous: either $7 per 100-oz lot, or $7 per 100k USD of notional (~$14–$40/lot at 2015–2025 prices). | **[DEFAULT]** $7 per 100k USD notional per round turn, the same rate as FX, so it scales with price. |
| Q4 | **Swap source.** Dukascopy candles carry no swap rates, and historical broker swap tables are not in any allowed package. | Swap is required on every overnight hold. | **[DEFAULT]** A fixed swap table per symbol (long/short, in points per lot per night, triple on Wednesday), entered by you from your demo broker's current Contract Specification and stored in `config/costs.yaml`. **Limitation:** today's swap applied to 2015–2024 is wrong in level, because rates changed a lot over the period. This will be stated in REPORT.md. |
| Q5 | **Trade-count gate vs slow strategies.** On D1, Turtle/Donchian, MA crossover and TSMOM (monthly rebalance = 36 decisions in 2022–2024) cannot plausibly reach 100 OOS trades on one symbol. | As written, they fail by construction, not on merit. | **[DEFAULT]** Apply the gate exactly as written (per strategy × symbol × timeframe). Also report a *pooled* row (4 symbols combined) for information only. It does **not** change survival unless you say so. |
| Q6 | What starting balance should the backtests use? | It affects lot rounding (0.01-lot minimum) but not % results. | **[DEFAULT]** $100,000, so a 0.5% risk can be sized accurately on XAU with wide stops. |
| Q7 | ICT / "smart money" ideas. | See §4.7. | **[DEFAULT]** Excluded. |

---

## 1. Folder layout

```
fusion-strategy/
├── PLAN.md                  # this file
├── REPORT.md                # written in Phases 3–6
├── requirements.txt         # pinned: pandas, numpy, numba, backtesting.py OR vectorbt (one only, see §6), matplotlib, pyarrow, dukascopy-python, MetaTrader5 (Windows only)
├── config/
│   ├── symbols.yaml         # pip size, contract size, min lot, slippage, commission
│   ├── costs.yaml           # swap table (Q4)
│   └── splits.yaml          # IS 2015-01-01..2021-12-31, OOS 2022-01-01..2024-12-31, HOLDOUT 2025-01-01..today
├── data/
│   ├── raw/                 # Dukascopy M1 bid + ask, parquet, one file per symbol-year
│   └── bars/                # resampled M15/H1/H4/D1 bid+ask parquet
├── fusion/
│   ├── data/download.py     # Phase 1
│   ├── data/quality.py      # Phase 1 gap/weekend/bad-tick/spread report
│   ├── data/resample.py
│   ├── indicators.py        # SMA/EMA, Wilder ATR/ADX/RSI, BB, KC, MACD, z-score (numba where useful)
│   ├── strategies/          # one module per family (§4)
│   │   ├── trend.py  momentum.py  meanrev.py  volatility.py  session.py  structure.py
│   ├── engine/
│   │   ├── fills.py         # bid/ask fills, slippage, commission, swap
│   │   ├── backtest.py      # event loop over closed bars; intrabar stop/target resolved on M1
│   │   ├── risk.py          # sizing, daily loss limit, kill switch
│   │   └── metrics.py       # §5 metrics, Monte Carlo, deflated Sharpe
│   ├── combine/             # Phase 4: regime filter, vote, walk-forward
│   └── holdout.py           # Phase 5: single run, writes a lock file
├── tests/
│   ├── test_no_lookahead.py # §3
│   ├── test_costs.py
│   └── test_holdout_lock.py
├── scripts/                 # phase1_data.py … phase5_holdout.py, entry points
├── results/                 # CSV/parquet trade lists, metric tables, PNG charts
├── mql5/FusionEA.mq5        # Phase 6
└── run_demo.py              # Phase 7
```

---

## 2. Data

- **Source.** Dukascopy M1 BID and ASK candles, 2015-01-01 → today, for the 4 symbols. Times are stored in UTC.
- **Resampling.** M15/H1/H4 are left-labeled and left-closed (each bar is labeled by its open time). **D1 uses the FX trading day, which rolls at 17:00 America/New_York** (DST-aware). That gives 5 daily bars per week and no Sunday stub. H4 is anchored to the same 17:00 NY roll. Bid and ask are resampled separately. Mid = (bid+ask)/2 is used only for indicators.
- **Quality report (Phase 1),** per symbol:
  - *Expected trading minutes:* Sun 17:00 NY → Fri 17:00 NY, minus the daily XAU break (17:00–18:00 NY). Exchange holidays (25 Dec, 1 Jan) are listed separately and not counted as gaps.
  - *Missing-hours %* = expected hours with zero M1 bars ÷ expected hours. **STOP if > 1% for any symbol.**
  - *Weekend bars:* bars between Fri 17:00 and Sun 17:00 NY. They are counted, then dropped.
  - *Bad ticks:* ask < bid; or an M1 bar with |log return| > 10 × rolling median absolute M1 return (window 1,440) that reverts by more than 50% within 5 bars; or high < low. Flagged bars are counted, then dropped.
  - *Spread distribution:* close-ask − close-bid, in pips (in $ for XAU). Report p50/p90/p99/max overall, by hour of day, and by year.

## 3. No-lookahead guarantee

- A signal for bar *t* uses only bars ≤ *t* (closed). The order is placed at the **open of the first M1 bar after bar *t* closes**: ask for buys, bid for sells, plus slippage.
- Higher-timeframe inputs (for example, D1 pivots used on M15) use the **last completed** higher-TF bar only.
- `tests/test_no_lookahead.py`:
  1. **Truncation test:** signals computed on the full data must equal signals computed on data truncated at a random *t* (for all bars ≤ *t*), over 50 random cut points per strategy.
  2. **Shift/perturbation test:** replace every bar after *t* with random noise. All signals ≤ *t* must be unchanged.
  3. **Negative control:** a deliberately leaky strategy (uses bar *t+1*'s close) must **fail** both tests, which proves the tests can catch lookahead.

## 4. Strategy library: exact rules

Shared conventions for all strategies:
- Indicators use mid prices. ATR, RSI and ADX use **Wilder smoothing**.
- One position per strategy per symbol at a time.
- Opposite signal = exit (and reverse, if the strategy is a stop-and-reverse type).
- **Mandatory stop (hard rule).** If the original rules have no stop, I add a protective stop at **2 × ATR(14)** from entry and **flag it as a deviation** from the original. 2×ATR mirrors the Turtle 2N stop.
- **Phase 3 sizing:** risk 0.5% of equity per trade, on the distance from entry to the strategy's stop. There are no daily loss limit or kill switch in Phase 3; those belong to Phase 4. Everything is long and short on all symbols.
- "Original/default" means the parameters from the cited source. Where no single canonical source exists, the row says **(convention)**. Those parameters are the common textbook defaults, not something I tuned.

### 4.1 Trend
| ID | Rules | Source |
|---|---|---|
| T1 MA cross | Long when SMA(50) crosses above SMA(200). Short on the cross below. Stop-and-reverse. Protective stop 2×ATR(14) (deviation). | "Golden/death cross" **(convention)**. There is no single originating paper. |
| T2 Turtle S1 | Long on a break above the 20-bar high, short on a break below the 20-bar low (intrabar, stop-entry order). Exit on the opposite 10-bar extreme. Stop 2N, where N = Wilder ATR(20). **Skip rule:** skip a S1 signal if the previous S1 breakout trade (taken or hypothetical) was a winner. No pyramiding (the original adds up to 4 units at ½N; omitted because it is averaging *up*, not down — **ask if you want it included**). | Original Turtle Trading Rules (Faith, *Way of the Turtle*, 2007). |
| T3 Turtle S2 | 55-bar breakout entry, 20-bar exit, 2N stop, no skip rule. | Same. |
| T4 ADX trend | Long when +DI(14) crosses above −DI(14) **and** ADX(14) > 25. Short is the mirror image. Exit on the opposite DI cross. Protective stop 2×ATR(14) (deviation; Wilder used an "extreme point" rule, which I am not modelling). | Wilder, *New Concepts in Technical Trading Systems* (1978). The 25 threshold is **(convention)**. |

### 4.2 Momentum
| ID | Rules | Source |
|---|---|---|
| M1 TSMOM | At each month end, go long if the trailing 12-month return is > 0, short if < 0. Hold 1 month. D1 only. Size: 0.5% risk on a 2×ATR(14) stop (deviation: the paper scales to 40% annualised vol). | Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE. |
| M2 MACD | MACD = EMA(12) − EMA(26), signal = EMA(9) of MACD. Long when MACD crosses above signal, short on the cross below. Stop-and-reverse. Protective stop 2×ATR(14) (deviation). | Appel (late 1970s), default 12/26/9. |

### 4.3 Mean reversion
| ID | Rules | Source |
|---|---|---|
| R1 BB + RSI | Long when close < lower BB(20, 2σ) **and** RSI(14) < 30. Short when close > upper band **and** RSI > 70. Exit at the middle band (SMA 20). Stop 2×ATR(14). | BB: Bollinger (default 20/2). RSI: Wilder (14, 30/70). The combination is **(convention)**. |
| R2 Z-score | z = (close − SMA(20)) / stdev(20). Long when z < −2, short when z > +2. Exit when z crosses 0. Stop 2×ATR(14). | **(convention)** |

### 4.4 Volatility
| ID | Rules | Source |
|---|---|---|
| V1 Squeeze | The squeeze is "on" when BB(20, 2) is inside Keltner(20, 1.5 × ATR(20), EMA midline). It "fires" on the first bar after it was on for ≥ 1 bar and then turns off. Direction = sign of the momentum histogram (linear-regression value of close − avg(avg(highest 20, lowest 20), SMA 20) over 20 bars). Exit when the histogram changes sign. Stop 2×ATR(14). | Carter, *Mastering the Trade* (TTM Squeeze). |
| V2 ATR expansion / volatility breakout | Buy-stop at today's open + 0.5 × the prior day's range, sell-stop at open − 0.5 × the prior day's range. The first one hit wins. Exit at the day's close (17:00 NY). Stop 2×ATR(14). Runs on M15/H1, with levels from D1. | Larry Williams-style volatility breakout. The 0.5 multiplier is **(convention)**. |

### 4.5 Session (times DST-aware, in London / New York local time)
| ID | Rules | Source |
|---|---|---|
| S1 Asian-range → London | Asian range = high/low from 00:00 to 07:00 London time. From 07:00 to 12:00 London: buy-stop at range high, sell-stop at range low, one trade per day. Stop at the opposite side of the range. Target 1 × range. Exit at 16:00 London if still open. M15/H1 only. | **(convention)**. There is no canonical published version. |
| S2 NY open ORB | Opening range = 08:00–08:30 New York time. From 08:30 to 11:00 NY: break of the range high/low, one trade per day. Stop at the opposite side. Target 1 × range. Exit 16:00 NY. M15 only (H1 is too coarse for a 30-min range). | **(convention)** |

### 4.6 Structure
| ID | Rules | Source |
|---|---|---|
| P1 Pivot reversion | Floor pivots from the prior D1 bar: P = (H+L+C)/3, R1 = 2P − L, S1 = 2P − H, R2 = P + (H−L), S2 = P − (H−L). Short at R1 touch, long at S1 touch. Target P. Stop at R2/S2. M15/H1. | Floor-trader pivots. The trading rule is **(convention)**. |
| P2 Prior-day H/L breakout | Buy-stop at the prior-day high, sell-stop at the prior-day low, one trade per side per day. Stop at the prior-day midpoint. Exit at 17:00 NY. M15/H1. | **(convention)** |

### 4.7 Excluded
- **ICT / smart money (order blocks, liquidity sweeps, market-structure shift, OTE).** The *patterns* can be coded (for example, a fair-value gap is a 3-bar gap where bar₁.high < bar₃.low). However, the trading model (which gap, which session, entry, stop and target) is taught in videos, not published as one fixed rule set. Any rules I wrote would be **my** parameters, not "original" ones, which breaks the Phase 2 requirement. **Excluded**, unless you want a clearly labelled FVG-retrace variant counted as an extra trial.

### 4.8 Test grid and trial count
Each strategy runs only on the timeframes where it is defined:

| Strategy | Timeframes |
|---|---|
| T1–T4, M2, R1, R2, V1 | M15, H1, H4, D1 |
| M1 | D1 |
| V2, S1, P1, P2 | M15, H1 |
| S2 | M15 |

Count: 8 × 4 + 1 + 4 × 2 + 1 = **42 strategy-timeframe cells × 4 symbols = 168 Phase 3 trials.** All 168 count toward the deflated Sharpe in Phase 4.

---

## 5. Costs, fills and metrics

- **Fills:** buy at ask, sell at bid (from the M1 data), so spread is built in. Slippage is added against you on every fill (Q2): FX 1 pip (0.0001; 0.01 for USDJPY), XAU $0.20.
- **Stops/targets:** resolved on M1 bars inside the signal bar. If both are touched in the same M1 bar, the **stop is assumed hit first**, which is the conservative choice.
- **Commission:** $7 per round-turn lot for FX. XAU per Q3.
- **Swap:** applied at 17:00 NY for each position held over the roll, triple on Wednesday (Q4).
- **Metrics** (IS and OOS separately): net profit, CAGR, max drawdown (% from equity peak, mark-to-market on bar close), Sharpe (daily returns, √252, rf = 0), profit factor, win rate, average R (P/L ÷ initial risk), trade count. Also exposure %.
- **Phase 3 survival:** OOS PF > 1.15 **and** OOS trades ≥ 100 **and** OOS max DD < 25%. IS parameters stay frozen in OOS (they are the defaults, so nothing is tuned in Phase 3).

---

## 6. Phase 4: combination design (proposal, finalised after the Phase 3 survivors are known)

- **Engine:** a custom event loop (numba). I'll use `backtesting.py` only as a cross-check on one strategy. Neither library handles bid/ask fills, M1 intrabar resolution, swap and portfolio-level daily limits out of the box. Only one of backtesting.py or vectorbt will be installed.
- **(a) Regime filter:** ADX(14) on H4 > θ_trend means trend regime, so only trend/momentum/volatility-breakout survivors count. ADX < θ_trend **and** ATR(14) percentile (rolling 1 year) < p_vol means range regime, so only mean-reversion/pivot survivors count. Anything else counts as neutral, and only session survivors count.
- **(b) Vote:** enter when ≥ k eligible survivors agree on direction on the same closed bar, and none disagree.
- **(c) Risk:** 0.5% risk per trade, stop = m × ATR(14), exit at R-multiple target or opposite vote. Daily loss limit 3% (realised + open, measured from the 17:00 NY equity) flattens and blocks new trades until the next day. Max-drawdown kill switch at 10% from peak flattens and stops trading permanently.
- **Tunable parameters (max 6):** θ_trend, p_vol, k, m, target R, max holding bars. Grid ≤ 3 values each (≤ 729 combinations). Every combination counts toward the DSR trial count.
- **Anchored walk-forward on IS:** train 2015–2017/test 2018, train 2015–2018/test 2019, train 2015–2019/test 2020, train 2015–2020/test 2021. The parameter set with the best Sharpe on the 4 stitched test years is frozen, then run once on OOS 2022–2024.
- **Robustness:**
  - ±20% on each tuned parameter, one at a time and jointly at the corners. "Cliff" = OOS PF falls below 1.0 or Sharpe falls by > 50%.
  - 1,000 reshuffles of the OOS trade order, with p95 max DD reported.
  - Deflated Sharpe (Bailey & López de Prado, 2014), with N = all Phase 3 + Phase 4 trials and the observed variance of trial Sharpes.
- **Benchmarks:**
  - Buy-and-hold per symbol (long, with swap, no leverage beyond 1×).
  - Random entry: coin-flip direction at the same bar frequency as the strategy, with the same risk model. 1,000 seeds, and the strategy's percentile among them is reported.

## 7. Phase 5: holdout
`holdout.py` refuses to run unless a frozen config hash from Phase 4 exists. It writes `results/HOLDOUT_LOCK` on its first run and refuses to run again. The result goes into REPORT.md whatever it is. **If the holdout loses money: STOP and report.**

## 8. Phase 6: MQL5 EA
- `FusionEA.mq5` re-implements the same indicators (Wilder smoothing, identical formulas) and evaluates **only on new closed bars**.
- Inputs: symbol, risk %, daily loss %, max DD %, `SignalsOnly` (log and alert, no orders), and the frozen strategy parameters.
- The EA refuses to trade when `ACCOUNT_TRADE_MODE` is not demo.
- Tester: "Every tick based on real ticks", 2022–2024. Match test: share of Python OOS trades with an EA trade on the same day in the same direction must be ≥ 90%.
- Known mismatch risk: MT5's real ticks come from **your broker**, not Dukascopy, so spreads, quotes and the session/day boundary differ. If the match is < 90%: STOP and report the breakdown by cause.

## 9. Phase 7: demo runner
`run_demo.py` reads `MT5_LOGIN`, `MT5_PASSWORD` and `MT5_SERVER` from the environment. Missing variables abort the run. It calls `account_info()`. **If `trade_mode != ACCOUNT_TRADE_MODE_DEMO`, it exits with an error before any other call.** On each closed bar it computes signals with the same Python code, appends them to `logs/signals.csv`, and, unless `--signals-only` is set, places orders with SL attached on the demo account. The 3% daily and 10% kill-switch limits are enforced.

## 10. Stop points
- After Phase 0 (this plan).
- Phase 1, if any symbol has > 1% missing hours.
- Phase 3, to show you the survivor table, or if nothing survives.
- Phase 5, if the holdout loses money.
- Phase 6, if the trade match is < 90%.
- Any phase that fails twice.
- Before: new dependencies, deleting files, changing splits, risk > 0.5%, or anything involving a live account.

## 11. Change log

- **2026-10-17. Exploratory ORB test, outside Phases 1–7.** On your request, `research/orb_sessions.py` tests a 30-minute opening-range breakout on the Tokyo, London and New York sessions for all 4 symbols.
  - **Data:** M1 bars exported from your MT5 demo terminal, instead of Dukascopy. MT5 bars are bid-only and carry one spread value per bar, reportedly the bar's *minimum* spread, so a 2× spread stress case is reported alongside.
  - **Period: 2023-01-01 to the end of the data, as one block, on your instruction.** This overlaps the 2025+ holdout. **The holdout is no longer clean for any strategy that includes an ORB or session-breakout component (S1, S2 or this test).** Phase 5 must state this if such a component reaches the final strategy.
  - **Sessions:** these are not the same as S1/S2 in §4.5. Each session uses its own first 30 minutes as the range: Tokyo 09:00–09:30 JST, London 08:00–08:30, New York 08:00–08:30 local time. Exits are at 15:00 JST, 16:00 London and 16:00 NY.
- **2026-10-17. ORB run on your data.**
  - **Data:** you supplied M15 exports for XAUUSD, USDJPY, USTEC and US30. EURUSD and GBPUSD were swapped for the two US index CFDs.
  - **Index assumptions:**
    - Contract size: 1 unit per lot.
    - Slippage: 1 point USTEC and 2 points US30 per fill.
    - Commission: $7 per $100k notional.
    - New York range: 09:30–10:00 ET.
    - These were proposed and have not been confirmed by you.
  - **Result:** nothing passes the survival gate. See `results/orb_sessions/README.md`.
  - **Risk what-if:** a sweep of 0.5–15% risk per trade was run as a backtest at your request. The plan's limit stays at 0.5%.
- **2026-10-17. No-stop what-if, at your request.**
  - It breaks hard rule "no position without a stop loss", so it is a backtest only. It is excluded from Phases 2–7 unless you change the rule.
  - The same ORB trades were run with the stop never triggered. All 12 combinations still fail the gate. See `results/orb_sessions_no_stop/README.md`.
  - Trial count for the deflated Sharpe is now 24 ORB variants on top of the Phase 3 grid.

