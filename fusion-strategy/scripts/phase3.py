"""Phase 3: every strategy on every symbol and timeframe, in-sample then
out-of-sample with the default parameters unchanged (PLAN.md sections 4-5).

    python scripts/phase3.py --data data/mt5 --out results/phase3

Stops (exit code 2) before any backtest when:
  * a symbol is missing more than 1% of trading hours (Phase 1 rule);
  * there is no in-sample data (nothing before 2022);
  * swap is not set in config/costs.toml (overnight holds need it);
Data from 2025 on is dropped at load (holdout).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.data import mt5, quality                              # noqa: E402
from fusion.data.resample import resample                         # noqa: E402
from fusion.data.splits import IN_SAMPLE, OUT_OF_SAMPLE, cap_before_holdout  # noqa: E402
from fusion.engine.backtest import SwapUnknown, run_backtest      # noqa: E402
from fusion.engine.costs import CONFIG, load_costs                # noqa: E402
from fusion.engine.metrics import summarize, survives             # noqa: E402
from fusion.strategies import STRATEGIES                          # noqa: E402

PLAN_SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")
START_EQUITY = 100_000.0


def stop(msg: str) -> None:
    print(f"STOP: {msg}")
    sys.exit(2)


def load(data: pathlib.Path, symbols, server_tz=None):
    files = mt5.files_by_symbol(data)
    missing = [s for s in symbols if s not in files]
    if missing:
        stop(f"no data files for {missing} in {data}")
    raw = {s: mt5.load_raw(files[s]) for s in symbols}
    tz, found = mt5.resolve_server_tz(raw, server_tz)
    bars = {s: cap_before_holdout(mt5.to_utc_bars(raw[s], tz)) for s in symbols}
    return bars, tz, found


def pooled(rows: pd.DataFrame, trades: dict) -> pd.DataFrame:
    """Per strategy and timeframe, all symbols together (information only;
    survival is per symbol, PLAN.md Q5)."""
    out = []
    for (sid, tf, period), g in rows.groupby(["strategy", "tf", "period"]):
        tr = [trades[(sid, tf, s, period)] for s in g.symbol if (sid, tf, s, period) in trades]
        tr = pd.concat([t for t in tr if len(t)], ignore_index=True) if any(len(t) for t in tr) else None
        if tr is None:
            out.append(dict(strategy=sid, tf=tf, period=period, trades=0))
            continue
        p = tr.pnl_usd
        losses = -p[p < 0].sum()
        out.append(dict(strategy=sid, tf=tf, period=period, trades=len(tr),
                        profit_factor=float(p[p > 0].sum() / losses) if losses > 0 else float("inf"),
                        avg_R=float(tr.R.mean()), win_rate=float((p > 0).mean())))
    return pd.DataFrame(out)


def survivor_table(rows: pd.DataFrame) -> str:
    oos = rows[rows.period == "OOS"].copy()
    ins = rows[rows.period == "IS"].set_index(["strategy", "tf", "symbol"])
    lines = ["| Strategy | TF | Symbol | IS PF | IS trades | OOS net profit | OOS CAGR | OOS max DD | "
             "OOS Sharpe | OOS PF | OOS win rate | OOS avg R | OOS trades | Survives |",
             "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for _, r in oos.sort_values(["strategy", "tf", "symbol"]).iterrows():
        i = ins.loc[(r.strategy, r.tf, r.symbol)]
        lines.append(
            f"| {r.strategy} | {r.tf} | {r.symbol} | {i.profit_factor:.2f} | {int(i.trades)} | "
            f"{r.net_profit:,.0f} | {r.cagr:+.1%} | {r.max_dd:.1%} | {r.sharpe:.2f} | "
            f"{r.profit_factor:.2f} | {r.win_rate:.0%} | {r.avg_R:+.3f} | {int(r.trades)} | "
            f"{'**yes**' if r.survives else 'no'} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=ROOT / "data" / "mt5")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "phase3")
    ap.add_argument("--symbols", default=",".join(PLAN_SYMBOLS))
    ap.add_argument("--strategies", default="", help="comma-separated ids (default: all)")
    ap.add_argument("--costs", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--server-tz", default=None)
    a = ap.parse_args(argv)
    symbols = tuple(s.strip() for s in a.symbols.split(",") if s.strip())
    sids = [s.strip() for s in a.strategies.split(",") if s.strip()] or list(STRATEGIES)
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1 checks
    bars, tz, found = load(a.data, symbols, a.server_tz)
    costs = load_costs(a.costs, point={s: b.attrs["point"] for s, b in bars.items()})
    reps = [quality.report(s, bars[s], costs[s].pip) for s in symbols]
    (a.out / "phase1_quality.md").write_text(
        f"Broker server timezone: {tz} {found}\n\n" + quality.to_markdown(reps) + "\n")
    (a.out / "phase1_quality.json").write_text(json.dumps(reps, indent=1, default=str))
    bad = [r["symbol"] for r in reps if r["stop"]]
    if bad:
        stop(f"more than 1% of trading hours missing for {bad} (see phase1_quality.md)")
    first = {s: b.index[0] for s, b in bars.items()}
    if any(f >= IN_SAMPLE[1] for f in first.values()):
        stop(f"no in-sample data: first bars {first}; in-sample is 2015-2021")
    unknown = [s for s in symbols if not costs[s].swap_known()]
    if unknown:
        stop(f"swap not set in {a.costs} for {unknown}: copy swap long/short (points) and the "
             "3-day swap weekday from MT5 Symbol -> Specification")

    # ---- Phase 3 grid
    mids = {s: mt5.mid_bars(b) for s, b in bars.items()}
    rows, trades, t0 = [], {}, time.time()
    for sid in sids:
        spec = STRATEGIES[sid]
        for tf in spec.timeframes:
            for sym in symbols:
                coarse = mids[sym] if tf == "M15" else resample(mids[sym], tf)
                sig = spec.run(coarse)
                for name, per in (("IS", IN_SAMPLE), ("OOS", OUT_OF_SAMPLE)):
                    res = run_backtest(sig, bars[sym], costs[sym], period=per)
                    start = max(per[0], first[sym].floor("D"))
                    m = summarize(res.trades, res.equity, START_EQUITY, (start, per[1]), res.exposure)
                    m.update(strategy=sid, tf=tf, symbol=sym, period=name, period_start=str(start),
                             survives=survives(m) if name == "OOS" else None, **res.counts)
                    rows.append(m)
                    trades[(sid, tf, sym, name)] = res.trades.assign(
                        strategy=sid, tf=tf, symbol=sym, period=name)
                print(f"{sid} {tf} {sym} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "phase3_metrics.csv", index=False)
    pd.concat([t for t in trades.values() if len(t)], ignore_index=True) \
        .to_csv(a.out / "phase3_trades.csv.gz", index=False)
    pooled(df, trades).to_csv(a.out / "phase3_pooled.csv", index=False)
    table = survivor_table(df)
    (a.out / "survivors.md").write_text(table + "\n")
    n_surv = int(df[(df.period == "OOS")].survives.sum())
    cells = len(df[df.period == "OOS"])
    print(table)
    print(f"\n✅ Phase 3 — {cells} strategy x timeframe x symbol cells run in-sample and "
          f"out-of-sample — {n_surv} survive (OOS PF > 1.15, >= 100 trades, max DD < 25%) — "
          f"{'STOP: nothing survived' if n_surv == 0 else 'next: show you the survivor table'}")


if __name__ == "__main__":
    try:
        main()
    except SwapUnknown as e:
        stop(str(e))
