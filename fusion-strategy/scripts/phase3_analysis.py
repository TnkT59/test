"""Phase 3 follow-up tables for REPORT.md, computed from the runner's outputs
(scripts/phase3.py). Nothing here changes a verdict: the gate is applied by
the runner. This adds what is needed to read the survivor table honestly.

    python scripts/phase3_analysis.py --main results/phase3 \
        --sens results/phase3_sens_no_floor results/phase3_sens_drop_spikes

  * survivors with long/short split, t-statistic of R and their verdicts in
    the sensitivity runs;
  * overlap between survivors of the same strategy and symbol (share of
    out-of-sample trades with the same direction, entry and exit price);
  * multiple testing: Bonferroni threshold for the number of distinct cells;
  * provisional cells (swap left out): the overnight financing rate, charged
    on both directions as a share of notional per year (360-day basis, at
    entry price), at which out-of-sample PF falls to the 1.15 gate, and the
    swap cost per 1% a year. A break-even, not an estimate of the broker's swap;
  * near misses: PF and drawdown pass but fewer than 100 trades;
  * buy-and-hold change of each symbol's mid close in each period (context).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.data import mt5                                       # noqa: E402
from fusion.data.splits import cap_before_holdout, split          # noqa: E402
from fusion.engine.costs import CONFIG, load_costs                # noqa: E402

GATE_PF = 1.15
ALPHA = 0.05


def load_run(d: pathlib.Path):
    m = pd.read_csv(d / "phase3_metrics.csv").fillna({"duplicate_of": ""})
    t = pd.read_csv(d / "phase3_trades.csv.gz")
    return m, t


def cell_trades(t: pd.DataFrame, sid, tf, sym, period="OOS") -> pd.DataFrame:
    return t[(t.strategy == sid) & (t.tf == tf) & (t.symbol == sym) & (t.period == period)]


def pf(p: np.ndarray) -> float:
    loss = -p[p < 0].sum()
    return float(p[p > 0].sum() / loss) if loss > 0 else float("inf")


def breakeven_rate(tr: pd.DataFrame, cost) -> tuple[float, float]:
    """(annual financing rate at which PF falls to GATE_PF, USD cost per 1%
    a year). Both directions pay; nights already count triple days."""
    notional = np.array([cost.notional_usd(u, px) for u, px in zip(tr.units, tr.entry)])
    per_rate = notional * tr.nights.to_numpy() / 360.0        # USD per 1.0 (=100%) a year
    p = tr.pnl_usd.to_numpy()
    if pf(p) <= GATE_PF:
        return 0.0, float(per_rate.sum() * 0.01)
    lo, hi = 0.0, 1.0
    if pf(p - hi * per_rate) > GATE_PF:
        return float("inf"), float(per_rate.sum() * 0.01)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pf(p - mid * per_rate) > GATE_PF else (lo, mid)
    return lo, float(per_rate.sum() * 0.01)


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    k = lambda t: set(zip(t.dir, t.entry.round(8), t.exit.round(8)))
    ka, kb = k(a), k(b)
    return len(ka & kb) / max(len(ka), len(kb)) if ka or kb else 1.0


def buy_and_hold(data: pathlib.Path, symbols, periods) -> pd.DataFrame:
    files = mt5.files_by_symbol(data)
    raw = {s: mt5.load_raw(files[s]) for s in symbols}
    tz, _ = mt5.resolve_server_tz(raw)
    rows = []
    for s in symbols:
        mid = mt5.mid_bars(cap_before_holdout(mt5.to_utc_bars(raw[s], tz))).close
        for name, (a, b) in periods:
            w = mid[(mid.index >= a) & (mid.index < b)]
            rows.append(dict(symbol=s, period=name, first=str(w.index[0])[:10], last=str(w.index[-1])[:10],
                             change=float(w.iloc[-1] / w.iloc[0] - 1)))
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=pathlib.Path, default=ROOT / "results" / "phase3")
    ap.add_argument("--sens", type=pathlib.Path, nargs="*", default=[])
    ap.add_argument("--data", type=pathlib.Path, default=ROOT / "data" / "mt5")
    ap.add_argument("--costs", type=pathlib.Path, default=CONFIG)
    a = ap.parse_args(argv)
    settings = json.loads((a.main / "settings.json").read_text())
    m, t = load_run(a.main)
    costs = load_costs(a.costs)
    sens = {d.name: load_run(d)[0] for d in a.sens}
    oos = m[m.period == "OOS"].astype({"survives": bool})
    ins = m[m.period == "IS"].set_index(["strategy", "tf", "symbol"])
    distinct = oos[oos.duplicate_of == ""]
    n_cells = len(distinct)
    z = NormalDist().inv_cdf(1 - ALPHA / (2 * n_cells))
    surv = oos[oos.survives].copy()
    f = lambda x, s: "n/a" if x != x else format(x, s)

    # ---- survivors
    rows = []
    for _, r in surv.sort_values(["strategy", "tf", "symbol"]).iterrows():
        tr = cell_trades(t, r.strategy, r.tf, r.symbol)
        i = ins.loc[(r.strategy, r.tf, r.symbol)]
        row = {"Cell": f"{r.strategy} {r.tf} {r.symbol}",
               "IS PF / trades": f"{f(i.profit_factor, '.2f')} / {int(i.trades)}",
               "OOS net $": f"{r.net_profit:,.0f}", "CAGR": f"{r.cagr:+.1%}", "Max DD": f"{r.max_dd:.1%}",
               "Sharpe": f"{r.sharpe:.2f}", "PF": f"{r.profit_factor:.3f}", "Win rate": f"{r.win_rate:.0%}",
               "Trades": int(r.trades),
               "Avg R": f"{r.avg_R:+.3f}", "t(R)": f"{r.t_stat_R:.2f}",
               "Long / short trades": f"{int(r.n_long)} / {int(r.n_short)}",
               "Long / short net $": f"{tr[tr.dir == 1].pnl_usd.sum():,.0f} / "
                                     f"{tr[tr.dir == -1].pnl_usd.sum():,.0f}",
               "Swap": "left out (provisional)" if r.provisional else "no overnight holds",
               "Duplicate of": r.duplicate_of or "-"}
        for name, sm in sens.items():
            s = sm[(sm.period == "OOS") & (sm.strategy == r.strategy) & (sm.tf == r.tf) & (sm.symbol == r.symbol)]
            row[name] = ("survives" if bool(s.survives.iloc[0]) else
                         f"fails (PF {s.profit_factor.iloc[0]:.3f})") if len(s) else "n/a"
        rows.append(row)
    surv_tab = pd.DataFrame(rows)

    # ---- sensitivity: cells that survive in a sensitivity run but not the main run
    extra = []
    for name, sm in sens.items():
        so = sm[(sm.period == "OOS") & (sm.survives == True)]  # noqa: E712
        for _, r in so.iterrows():
            if not ((surv.strategy == r.strategy) & (surv.tf == r.tf) & (surv.symbol == r.symbol)).any():
                mm = oos[(oos.strategy == r.strategy) & (oos.tf == r.tf) & (oos.symbol == r.symbol)].iloc[0]
                extra.append({"Run": name, "Cell": f"{r.strategy} {r.tf} {r.symbol}",
                              "PF there": f"{r.profit_factor:.3f}", "Trades there": int(r.trades),
                              "PF in main run": f"{mm.profit_factor:.3f}",
                              "Max DD in main run": f"{mm.max_dd:.1%}", "Trades in main run": int(mm.trades)})
    extra_tab = pd.DataFrame(extra)

    # ---- overlap between survivors of the same strategy and symbol
    ov = []
    for (sid, sym), g in surv.groupby(["strategy", "symbol"]):
        tfs = list(g.tf)
        for x in range(len(tfs)):
            for y in range(x + 1, len(tfs)):
                ov.append({"Strategy": sid, "Symbol": sym, "Cells": f"{tfs[x]} vs {tfs[y]}",
                           "Same OOS trades": f"{overlap(cell_trades(t, sid, tfs[x], sym), cell_trades(t, sid, tfs[y], sym)):.1%}"})
    ov_tab = pd.DataFrame(ov)

    # ---- provisional survivors: swap break-even
    sw = []
    for _, r in surv[surv.provisional].sort_values(["strategy", "tf", "symbol"]).iterrows():
        tr = cell_trades(t, r.strategy, r.tf, r.symbol)
        rate, per1 = breakeven_rate(tr, costs[r.symbol])
        sw.append({"Cell": f"{r.strategy} {r.tf} {r.symbol}", "Overnight trades": int(r.overnight_trades),
                   "Swap nights": int(r.swap_nights), "OOS net $ (no swap)": f"{r.net_profit:,.0f}",
                   "Swap cost per 1%/yr": f"{per1:,.0f}",
                   "PF reaches 1.15 at": "never below 100%/yr" if rate == float("inf") else f"{rate:.2%}/yr"})
    sw_tab = pd.DataFrame(sw)

    # ---- near misses
    nm = oos[(oos.duplicate_of == "") & (oos.trades < 100) & (oos.profit_factor > GATE_PF) & (oos.max_dd < 0.25)]
    nm_tab = pd.DataFrame([{"Cell": f"{r.strategy} {r.tf} {r.symbol}", "PF": f"{r.profit_factor:.2f}",
                            "Trades": int(r.trades), "Avg R": f"{r.avg_R:+.3f}", "t(R)": f"{r.t_stat_R:.2f}"}
                           for _, r in nm.sort_values("t_stat_R", ascending=False).iterrows()])

    # ---- overall counts
    by = distinct.assign(fail=np.select(
        [distinct.survives, distinct.trades < 100, distinct.profit_factor <= GATE_PF],
        ["survives", "fewer than 100 trades", "PF <= 1.15"], "max DD >= 25%"))
    counts = by.fail.value_counts().to_dict()
    pos = int((distinct.net_profit > 0).sum())
    is_m = m[(m.period == "IS")]
    is_m = is_m.merge(distinct[["strategy", "tf", "symbol"]], on=["strategy", "tf", "symbol"])

    periods = [("IS", tuple(pd.Timestamp(x) for x in settings["in_sample"])),
               ("OOS", tuple(pd.Timestamp(x) for x in settings["out_of_sample"]))]
    bh = buy_and_hold(a.data, sorted(set(m.symbol)), periods)
    bh["change"] = bh.change.map(lambda x: f"{x:+.1%}")

    sig = surv[(surv.duplicate_of == "") & (surv.t_stat_R >= z)]
    tt = distinct[distinct.trades >= 2].t_stat_R
    t_counts = ", ".join(f"t >= {c}: {int((tt >= c).sum())} (chance alone, independent cells: "
                         f"{len(tt) * (1 - NormalDist().cdf(c)):.1f})" for c in (1.645, 1.96, 2.576))
    p = pd.read_csv(a.main / "phase3_pooled.csv")
    p = p[(p.period == "OOS")].merge(surv[["strategy", "tf"]].drop_duplicates(), on=["strategy", "tf"])
    pooled_tab = pd.DataFrame([{"Strategy": r.strategy, "TF": r.tf, "Trades (4 symbols)": int(r.trades),
                                "PF": f"{r.profit_factor:.3f}", "Avg R": f"{r.avg_R:+.3f}",
                                "Win rate": f"{r.win_rate:.0%}"} for r in p.itertuples()])
    lines = [
        "## Phase 3 analysis (generated by scripts/phase3_analysis.py)", "",
        f"Run: {a.main.name}; in-sample {settings['in_sample'][0][:10]} to {settings['in_sample'][1][:10]}, "
        f"out-of-sample {settings['out_of_sample'][0][:10]} to {settings['out_of_sample'][1][:10]}; "
        f"spread floor {settings['spread_floor']}; spikes dropped: {settings['drop_spikes']}; "
        f"swap left out for {', '.join(settings['swap_unknown'])}.", "",
        f"Distinct cells (exact duplicates removed): {n_cells}. Out-of-sample verdicts: "
        + ", ".join(f"{k}: {v}" for k, v in counts.items())
        + f". Cells with positive out-of-sample net profit: {pos} of {n_cells}. "
        f"In-sample cells with positive net profit: {int((is_m.net_profit > 0).sum())} of {len(is_m)} "
        f"({int((is_m.trades == 0).sum())} with no in-sample trades).", "",
        "### Survivors", "", md_table(surv_tab), "",
        "### Cells that survive only in a sensitivity run", "",
        md_table(extra_tab) if len(extra_tab) else "None.", "",
        "### Overlap between survivors of the same strategy and symbol", "",
        md_table(ov_tab) if len(ov_tab) else "None.", "",
        "### Multiple testing", "",
        f"{n_cells} distinct cells were tested. A Bonferroni bound at {ALPHA:.0%} family-wise error "
        f"(two-sided) needs t(R) >= {z:.2f}. Survivors that clear it: "
        + (", ".join(f"{r.strategy} {r.tf} {r.symbol} (t = {r.t_stat_R:.2f})" for _, r in sig.iterrows())
           if len(sig) else "none")
        + f". Highest t(R) among survivors: {surv.t_stat_R.max():.2f}." if len(surv) else "No survivors.", "",
        f"Out-of-sample t(R) across the {len(tt)} distinct cells with at least 2 trades: {t_counts}. "
        "Cells share strategies and symbols, so they are not independent; the chance counts are a "
        "reference, not a test.", "",
        "### Pooled across the 4 symbols (information only), for strategy x timeframe with a survivor", "",
        md_table(pooled_tab) if len(pooled_tab) else "None.", "",
        "### Provisional survivors: swap break-even", "",
        "Financing charged on both directions as a share of notional per year (360-day basis, notional "
        "at entry price). This is the rate at which the cell would fail the PF gate, not an estimate of "
        "the broker's swap. Drawdown is not recomputed.", "",
        md_table(sw_tab) if len(sw_tab) else "None.", "",
        "### Near misses (PF > 1.15 and max DD < 25%, fewer than 100 trades)", "",
        md_table(nm_tab) if len(nm_tab) else "None.", "",
        "### Buy-and-hold change of the mid close (context)", "", md_table(bh), ""]
    text = "\n".join(lines)
    (a.main / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
