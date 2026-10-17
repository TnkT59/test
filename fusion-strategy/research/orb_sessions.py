"""Opening-range breakout (ORB) on the Tokyo, London and New York sessions.

Exploratory test requested 2026-10-17, outside the Phase 1-7 pipeline.
Data: M1 bars exported from a MetaTrader 5 terminal (Symbols -> Bars -> Export).
Period: 2023-01-01 .. end of data, as one block (this uses the 2025+ holdout;
see PLAN.md).

Rules (fixed, nothing tuned):
  * Range  = high/low of the bid M1 bars in the first 30 minutes of the session.
  * Entry  = buy stop at range high (fills when ask reaches it), sell stop at
             range low (fills when bid reaches it). First trigger wins; one
             trade per session per day; entries only inside the entry window.
  * Stop   = opposite side of the range.  Target = 1 x range from the level.
  * Exit   = session exit time if neither stop nor target was hit.

Fill model (conservative):
  * Buy at ask = bid + bar spread, sell at bid. Slippage against us on every
    fill: 1 pip FX, $0.20 XAU.
  * Stop and target touched in the same M1 bar -> stop first. On the entry bar
    the stop can be hit but the target cannot.
  * Commission: $7 per round-turn lot FX; XAU $7 per $100k notional.
  * Swap: none, because every trade closes before the 17:00 New York roll
    (asserted in code).
  * Sizing: 0.5% of equity at risk on the entry-to-stop distance, rounded
    down to 0.01 lot, capped at 30:1 leverage FX / 20:1 XAU (EU retail limits).

Known limitation: MT5 stores one spread value per bar, reportedly the minimum
spread in that bar, so the base case may understate costs. The report adds a
"2x spread" stress case for that reason.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")
ALIASES = {"GOLD": "XAUUSD"}

SPEC = {
    # units per lot, pip size, slippage per fill (price units), max leverage
    "EURUSD": dict(lot=100_000, pip=0.0001, slip=0.0001, max_lev=30.0),
    "GBPUSD": dict(lot=100_000, pip=0.0001, slip=0.0001, max_lev=30.0),
    "USDJPY": dict(lot=100_000, pip=0.01, slip=0.01, max_lev=30.0),
    "XAUUSD": dict(lot=100, pip=0.01, slip=0.20, max_lev=20.0),
}
COMMISSION_PER_LOT_RT = 7.0          # FX, USD per round-turn lot
COMMISSION_PER_100K_NOTIONAL = 7.0   # XAU, USD per round turn per $100k
RISK_PCT = 0.005
START_EQUITY = 100_000.0
NY_ROLL_HOUR = 17


@dataclass(frozen=True)
class Session:
    name: str
    tz: str
    range_start: str
    range_end: str
    entry_end: str
    exit: str


SESSIONS = (
    Session("Tokyo", "Asia/Tokyo", "09:00", "09:30", "12:00", "15:00"),
    Session("London", "Europe/London", "08:00", "08:30", "11:00", "16:00"),
    Session("NewYork", "America/New_York", "08:00", "08:30", "11:00", "16:00"),
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) > 1 and raw[1:2] == b"\x00":
        return raw.decode("utf-16-le")
    return raw.decode("utf-8-sig")


def read_mt5_csv(path: pathlib.Path) -> pd.DataFrame:
    """Read an MT5 bar export. Returns server-time-indexed open/high/low/close/spread."""
    from io import StringIO

    text = _decode(path.read_bytes())
    head = text.split("\n", 1)[0]
    sep = "\t" if "\t" in head else (";" if ";" in head else ",")
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    df.columns = [re.sub(r"[<>\s]", "", c).lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        stamp = df["date"].str.strip() + " " + df["time"].str.strip()
    elif "time" in df.columns:
        stamp = df["time"].str.strip()
    elif "datetime" in df.columns:
        stamp = df["datetime"].str.strip()
    else:
        raise ValueError(f"{path.name}: no date/time column in {list(df.columns)}")
    stamp = stamp.str.replace(".", "-", regex=False)
    idx = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            idx = pd.to_datetime(stamp, format=fmt)
            break
        except ValueError:
            pass
    if idx is None:
        idx = pd.to_datetime(stamp, format="mixed")
    need = ["open", "high", "low", "close", "spread"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; got {list(df.columns)}")
    out = pd.DataFrame({c: pd.to_numeric(df[c]) for c in need})
    out.index = pd.DatetimeIndex(idx, name="server_time")
    return out


def symbol_of(path: pathlib.Path) -> str | None:
    name = path.stem.upper()
    for s in SYMBOLS + tuple(ALIASES):
        if s in name:
            return ALIASES.get(s, s)
    return None


def infer_point(prices: np.ndarray) -> float:
    sample = prices[: min(len(prices), 20_000)]
    for d in range(0, 7):
        scaled = sample * 10**d
        if np.all(np.abs(scaled - np.round(scaled)) < 1e-6):
            return 10.0**-d
    return 1e-6


# --------------------------------------------------------------------------
# Server time -> UTC
# --------------------------------------------------------------------------

CANDIDATE_TZ = ("NY+7", "Europe/Helsinki", "Europe/London",
                "Europe/Berlin", "UTC", "Etc/GMT-1", "Etc/GMT-2", "Etc/GMT-3")


def server_to_utc(idx: pd.DatetimeIndex, server_tz: str) -> pd.DatetimeIndex:
    """Convert server wall-clock times to UTC. Times that do not exist or are
    ambiguous in the server zone (DST switch, market closed) become NaT."""
    if server_tz == "NY+7":
        local, zone = idx - pd.Timedelta(hours=7), "America/New_York"
    else:
        local, zone = idx, server_tz
    return local.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def weekly_opens(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    gaps = idx[1:] - idx[:-1]
    return idx[1:][gaps > pd.Timedelta(hours=24)]


FX_OPEN_NY_HOUR = 17   # spot FX reopens Sunday 17:00 New York


def detect_server_tz(idx: pd.DatetimeIndex) -> tuple[str, float]:
    """Pick the server timezone under which an FX symbol's weekly open lands on
    Sunday 17:xx New York in the most weeks. Use FX symbols only: gold's weekly
    open differs between brokers (17:00 or 18:00 NY). Fixed offsets that differ
    by a whole hour are only told apart by this anchor."""
    opens = weekly_opens(idx)
    if len(opens) < 8:
        raise ValueError("not enough weekly opens to detect the server timezone")
    best = ("", -1.0)
    for tz in CANDIDATE_TZ:
        ny = server_to_utc(pd.DatetimeIndex(opens), tz).tz_convert("America/New_York")
        ok = (ny.dayofweek == 6) & (ny.hour == FX_OPEN_NY_HOUR)
        score = float(np.asarray(ok, dtype=bool).sum() / len(opens))
        if score > best[1]:
            best = (tz, score)
    return best


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class Bars:
    t: np.ndarray       # int64 ns UTC
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    sp: np.ndarray      # spread in price units


def _local_bound(day, hhmm: str, tz: str) -> int:
    ts = pd.Timestamp(f"{day} {hhmm}").tz_localize(tz).tz_convert("UTC")
    return ts.value


def _ny_roll_between(t0: int, t1: int) -> bool:
    a = pd.Timestamp(t0, tz="UTC").tz_convert("America/New_York")
    b = pd.Timestamp(t1, tz="UTC").tz_convert("America/New_York")
    roll = a.normalize() + pd.Timedelta(hours=NY_ROLL_HOUR)
    if roll <= a:
        roll += pd.Timedelta(days=1)
    return roll < b


def simulate_session(bars: Bars, sym: str, sess: Session, days,
                     spread_mult: float = 1.0, costs: bool = True):
    """Return (trades, skipped_days). Trades are dicts in chronological order;
    sizing is applied later. A day is skipped when < 24 of the 30 range
    minutes have bars."""
    spec = SPEC[sym]
    slip = spec["slip"] if costs else 0.0
    trades = []
    skipped = 0
    for day in days:
        r0 = _local_bound(day, sess.range_start, sess.tz)
        r1 = _local_bound(day, sess.range_end, sess.tz)
        e1 = _local_bound(day, sess.entry_end, sess.tz)
        x1 = _local_bound(day, sess.exit, sess.tz)
        i0, i1 = np.searchsorted(bars.t, [r0, r1])
        if i1 - i0 < 24:          # need >= 80% of the 30 range minutes
            skipped += 1
            continue
        hi = bars.h[i0:i1].max()
        lo = bars.l[i0:i1].min()
        rng = hi - lo
        if rng <= 0:
            continue
        j_end_entry, j_exit = np.searchsorted(bars.t, [e1, x1])
        pos = 0
        entry = stop = target = 0.0
        entry_i = -1
        for j in range(i1, j_exit):
            sp = bars.sp[j] * spread_mult if costs else 0.0
            o, h, l, c = bars.o[j], bars.h[j], bars.l[j], bars.c[j]
            if pos == 0:
                if j >= j_end_entry:
                    break
                long_hit = h + sp >= hi
                short_hit = l <= lo
                if long_hit:          # both in one bar -> long first, then stop check
                    pos, entry = 1, max(hi, o + sp) + slip
                    stop, target = lo, hi + rng
                elif short_hit:
                    pos, entry = -1, min(lo, o) - slip
                    stop, target = hi, lo - rng
                else:
                    continue
                entry_i = j
            # position open: stop first, target not on the entry bar
            if pos == 1:
                if l <= stop:
                    trades.append(_trade(day, sess, sym, pos, entry, min(stop, o) - slip,
                                         stop, entry_i, j, bars, "stop", hi, lo))
                    pos = 0
                    break
                if j > entry_i and h >= target:
                    trades.append(_trade(day, sess, sym, pos, entry, target - slip,
                                         stop, entry_i, j, bars, "target", hi, lo))
                    pos = 0
                    break
            else:
                if h + sp >= stop:
                    trades.append(_trade(day, sess, sym, pos, entry, max(stop, o + sp) + slip,
                                         stop, entry_i, j, bars, "stop", hi, lo))
                    pos = 0
                    break
                if j > entry_i and l + sp <= target:
                    trades.append(_trade(day, sess, sym, pos, entry, target + slip,
                                         stop, entry_i, j, bars, "target", hi, lo))
                    pos = 0
                    break
        if pos != 0:               # time exit at the close of the last bar before exit time
            k = j_exit - 1
            sp = bars.sp[k] * spread_mult if costs else 0.0
            px = bars.c[k] - slip if pos == 1 else bars.c[k] + sp + slip
            trades.append(_trade(day, sess, sym, pos, entry, px, stop, entry_i, k,
                                 bars, "time", hi, lo))
    return trades, skipped


def _trade(day, sess, sym, pos, entry, exit_px, stop, i_in, i_out, bars, why, hi, lo):
    t_in, t_out = int(bars.t[i_in]), int(bars.t[i_out])
    assert not _ny_roll_between(t_in, t_out), "trade crosses the NY roll: swap would apply"
    return dict(day=day, session=sess.name, symbol=sym, dir=pos, entry=entry,
                exit=exit_px, stop=stop, t_in=t_in, t_out=t_out, reason=why,
                range_hi=hi, range_lo=lo)


def size_and_score(trades: list[dict], sym: str, costs: bool = True) -> pd.DataFrame:
    """Apply 0.5% risk sizing with compounding; return trades with USD P/L and R."""
    spec = SPEC[sym]
    eq = START_EQUITY
    rows = []
    for tr in trades:
        dist = abs(tr["entry"] - tr["stop"])
        to_usd = (1.0 / tr["entry"]) if sym == "USDJPY" else 1.0
        dist_usd = dist * to_usd
        if dist_usd <= 0:
            continue
        units = RISK_PCT * eq / dist_usd
        notional_usd = units if sym == "USDJPY" else units * tr["entry"]
        capped = notional_usd > spec["max_lev"] * eq
        if capped:
            units = spec["max_lev"] * eq / (1.0 if sym == "USDJPY" else tr["entry"])
        step = spec["lot"] * 0.01
        units = math.floor(units / step) * step
        if units <= 0:
            continue
        pnl = tr["dir"] * (tr["exit"] - tr["entry"]) * units
        pnl_usd = pnl / tr["exit"] if sym == "USDJPY" else pnl
        if costs:
            if sym == "XAUUSD":
                comm = COMMISSION_PER_100K_NOTIONAL * units * tr["entry"] / 100_000
            else:
                comm = COMMISSION_PER_LOT_RT * units / spec["lot"]
        else:
            comm = 0.0
        net = pnl_usd - comm
        risk_usd = units * dist_usd
        eq_before = eq
        eq += net
        rows.append({**tr, "units": units, "lots": units / spec["lot"], "capped": capped,
                     "commission": comm, "pnl_usd": net, "risk_usd": risk_usd,
                     "R": net / risk_usd, "equity_before": eq_before, "equity": eq})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if df.empty:
        return dict(trades=0)
    wins = df.loc[df.pnl_usd > 0, "pnl_usd"].sum()
    losses = -df.loc[df.pnl_usd < 0, "pnl_usd"].sum()
    eq = np.concatenate([[START_EQUITY], df.equity.to_numpy()])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max())
    years = (end - start).days / 365.25
    final = float(eq[-1])
    cagr = (final / START_EQUITY) ** (1 / years) - 1 if final > 0 else -1.0
    out_day = (pd.to_datetime(df.t_out, utc=True).dt.tz_convert("America/New_York")
               .dt.tz_localize(None).dt.normalize())
    daily = df.groupby(out_day)["pnl_usd"].sum()
    bdays = pd.bdate_range(start.normalize(), end.normalize())
    daily = daily.reindex(bdays.union(daily.index), fill_value=0.0)
    eq_daily = START_EQUITY + daily.cumsum()
    rets = daily / eq_daily.shift(1).fillna(START_EQUITY)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    r = df.R.to_numpy()
    t_stat = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 1 else 0.0
    return dict(trades=len(df), win_rate=float((df.pnl_usd > 0).mean()),
                profit_factor=float(wins / losses) if losses > 0 else float("inf"),
                avg_R=float(r.mean()), t_stat_R=t_stat, net_profit=final - START_EQUITY,
                cagr=cagr, max_dd=max_dd, sharpe=sharpe,
                capped=int(df.capped.sum()))


def survives(m: dict) -> bool:
    return m.get("trades", 0) >= 100 and m["profit_factor"] > 1.15 and m["max_dd"] < 0.25


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_raw(files) -> pd.DataFrame:
    df = pd.concat([read_mt5_csv(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def resolve_server_tz(raw: dict[str, pd.DataFrame], override: str | None):
    """One timezone for the whole broker, detected on the FX symbols."""
    if override:
        return override, {}
    fx = [s for s in raw if s != "XAUUSD"]
    if not fx:
        sys.exit("no FX symbol to detect the broker timezone from; pass --server-tz")
    found = {s: detect_server_tz(raw[s].index) for s in fx}
    tzs = {tz for tz, _ in found.values()}
    if len(tzs) != 1 or min(sc for _, sc in found.values()) < 0.9:
        sys.exit(f"server timezone unclear: {found}; pass --server-tz")
    return tzs.pop(), found


def load_symbol(df: pd.DataFrame, tz: str):
    point = infer_point(np.concatenate([df.open.to_numpy(), df.close.to_numpy()]))
    df = df.copy()
    df.index = server_to_utc(df.index, tz)
    dropped = int(df.index.isna().sum())
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["spread_px"] = df.spread * point
    return df, point, dropped


def to_bars(df: pd.DataFrame) -> Bars:
    return Bars(t=df.index.as_unit("ns").asi8, o=df.open.to_numpy(float),
                h=df.high.to_numpy(float), l=df.low.to_numpy(float),
                c=df.close.to_numpy(float), sp=df.spread_px.to_numpy(float))


def trading_days(df: pd.DataFrame, sess: Session):
    local = df.index.tz_convert(sess.tz)
    days = pd.Index(local.normalize().date).unique()
    return [d for d in days if pd.Timestamp(d).dayofweek < 5]


def run(data_dir: pathlib.Path, out_dir: pathlib.Path, start: str, end: str | None,
        server_tz: str | None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[pathlib.Path]] = {}
    for f in sorted(data_dir.glob("*.csv")):
        s = symbol_of(f)
        if s:
            files.setdefault(s, []).append(f)
    if not files:
        sys.exit(f"no MT5 CSV files for {SYMBOLS} found in {data_dir}")
    rows, all_trades, notes = [], [], []
    raw = {sym: load_raw(fl) for sym, fl in files.items()}
    tz, found = resolve_server_tz(raw, server_tz)
    notes.append(f"broker server timezone: {tz} "
                 + (f"(detected; share of FX weekly opens at Sunday 17:xx NY: "
                    + ", ".join(f"{s} {sc:.1%}" for s, (_, sc) in found.items()) + ")"
                    if found else "(given with --server-tz)"))
    for sym, fl in files.items():
        df, point, dropped = load_symbol(raw.pop(sym), tz)
        lo_t = pd.Timestamp(start, tz="UTC")
        hi_t = pd.Timestamp(end, tz="UTC") if end else df.index.max()
        df = df[(df.index >= lo_t) & (df.index <= hi_t)]
        notes.append(f"{sym}: {len(fl)} file(s), {len(df):,} M1 bars, "
                     f"{df.index.min()} -> {df.index.max()}, point={point:g}, "
                     f"{dropped} bars dropped at DST switches, "
                     f"median spread={df.spread_px.median() / SPEC[sym]['pip']:.2f} pips")
        bars = to_bars(df)
        span = (df.index.min(), df.index.max())
        for sess in SESSIONS:
            days = trading_days(df, sess)
            for label, mult, costs in (("net", 1.0, True), ("net_2x_spread", 2.0, True),
                                       ("gross", 1.0, False)):
                tr, skipped = simulate_session(bars, sym, sess, days, spread_mult=mult,
                                               costs=costs)
                scored = size_and_score(tr, sym, costs=costs)
                m = metrics(scored, span[0].tz_localize(None), span[1].tz_localize(None))
                rows.append(dict(symbol=sym, session=sess.name, case=label,
                                 days=len(days), skipped_days=skipped, **m,
                                 survives_gate=survives(m) if m.get("trades") else False))
                if label == "net" and not scored.empty:
                    all_trades.append(scored)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "orb_metrics.csv", index=False)
    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.to_csv(out_dir / "orb_trades_net.csv", index=False)
        _yearly(trades).to_csv(out_dir / "orb_by_year.csv", index=False)
        _pooled(trades).to_csv(out_dir / "orb_pooled_by_session.csv", index=False)
        _plot(trades, out_dir / "orb_cum_R.png")
    (out_dir / "data_notes.txt").write_text("\n".join(notes) + "\n")
    return res


def _yearly(trades: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_datetime(trades.t_in, utc=True).dt.year
    g = trades.groupby([trades.symbol, trades.session, y.rename("year")])["R"]
    return g.agg(trades="count", avg_R="mean", sum_R="sum").reset_index()


def _pooled(trades: pd.DataFrame) -> pd.DataFrame:
    """Per session, all symbols together, in R. Bonferroni: 12 symbol x session
    cells were tested, so a cell needs |t| > 2.87 for 5% family-wise two-sided."""
    rows = []
    for sess, g in trades.groupby("session"):
        r = g.R.to_numpy()
        rows.append(dict(session=sess, trades=len(r), avg_R=r.mean(),
                         t_stat_R=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                         win_rate=(r > 0).mean()))
    return pd.DataFrame(rows)


def _plot(trades: pd.DataFrame, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, sess in zip(axes, SESSIONS):
        sub = trades[trades.session == sess.name]
        for sym, g in sub.groupby("symbol"):
            x = pd.to_datetime(g.t_out, utc=True)
            ax.plot(x, g.R.cumsum().to_numpy(), label=sym, linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{sess.name} ORB, net of costs")
        ax.set_ylabel("cumulative R")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/mt5"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/orb_sessions"))
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--server-tz", default=None,
                   help="override detection, e.g. NY+7, Europe/Helsinki, Etc/GMT-2")
    a = p.parse_args(argv)
    res = run(a.data, a.out, a.start, a.end, a.server_tz)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
