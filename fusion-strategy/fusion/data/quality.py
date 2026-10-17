"""Phase 1 data-quality report (PLAN.md section 2) for UTC bars from fusion.data.mt5.

  * Expected trading time: Sunday 17:00 to Friday 17:00 New York, minus gold's
    daily 17:00-18:00 New York break, minus 25 December and 1 January (listed
    separately, not counted as gaps).
  * Missing hours: expected clock hours with no bar at all, as a share of
    expected hours. PLAN.md: STOP if above 1% for any symbol.
  * Weekend bars: bars opening between Friday 17:00 and Sunday 17:00 New York.
  * Bad bars: NaN, high < low or negative spread (dropped by the loader), plus
    spikes: |log return| above 10 x the rolling median |log return| of the
    previous day, that reverses by more than half within 5 bars.
  * Spreads: raw MT5 bar spread (reportedly the bar minimum) in pips, overall,
    by New York hour and by year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import NY

MAX_MISSING_SHARE = 0.01
HOLIDAYS = ((12, 25), (1, 1))


def expected_hours(start: pd.Timestamp, end: pd.Timestamp, gold: bool) -> tuple[pd.DatetimeIndex, int]:
    """Clock hours (UTC) the market should be open, and holiday hours removed."""
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    ny = hours.tz_convert(NY)
    wd, hr = ny.dayofweek, ny.hour
    open_ = (wd < 4) | ((wd == 4) & (hr < 17)) | ((wd == 6) & (hr >= 17))
    if gold:
        open_ &= ~((hr == 17) & (wd != 4) & (wd != 5))
    hol = np.zeros(len(hours), dtype=bool)
    for m, d in HOLIDAYS:
        hol |= (ny.month == m) & (ny.day == d)
    return hours[open_ & ~hol], int((open_ & hol).sum())


def spikes(bars: pd.DataFrame, window: int = 96, mult: float = 10.0, revert: int = 5) -> int:
    lr = np.log(bars.close).diff()
    med = lr.abs().rolling(window, min_periods=window // 2).median().shift(1)
    big = lr.abs() > mult * med
    fut = np.log(bars.close).shift(-revert) - np.log(bars.close)
    reverts = (np.sign(fut) == -np.sign(lr)) & (fut.abs() > 0.5 * lr.abs())
    return int((big & reverts).sum())


def report(sym: str, bars: pd.DataFrame, pip: float) -> dict:
    gold = sym == "XAUUSD"
    exp, hol_hours = expected_hours(bars.index[0], bars.index[-1], gold)
    have = pd.DatetimeIndex(bars.index.floor("h").unique())
    missing = exp.difference(have)
    ny = bars.index.tz_convert(NY)
    wd, hr = ny.dayofweek, ny.hour
    weekend = ((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17))
    sp = bars.spread_px / pip
    q = lambda s: {f"p{int(x * 100)}": round(float(s.quantile(x)), 3) for x in (0.5, 0.9, 0.99)} \
        | {"max": round(float(s.max()), 3)}
    miss_share = len(missing) / len(exp) if len(exp) else 1.0
    gaps = pd.Series(missing).diff().ne(pd.Timedelta(hours=1)).cumsum() if len(missing) else pd.Series()
    longest = int(pd.Series(missing).groupby(gaps).size().max()) if len(missing) else 0
    return dict(
        symbol=sym, first=str(bars.index[0]), last=str(bars.index[-1]), bars=len(bars),
        expected_hours=len(exp), missing_hours=len(missing), missing_share=miss_share,
        longest_gap_hours=longest, holiday_hours_excluded=hol_hours,
        weekend_bars=int(weekend.sum()), dropped_bad_rows=int(bars.attrs.get("dropped_bad", 0)),
        spikes=spikes(bars), spread_pips=q(sp),
        spread_by_ny_hour=sp.groupby(hr).median().round(3).to_dict(),
        spread_by_year=sp.groupby(bars.index.year).median().round(3).to_dict(),
        zero_spread_share=float((bars.spread_px == 0).mean()),
        stop=miss_share > MAX_MISSING_SHARE)


def to_markdown(reports: list[dict]) -> str:
    lines = ["| Symbol | First bar | Last bar | Bars | Missing hours | Share | Longest gap (h) | "
             "Weekend bars | Bad rows | Spikes | Spread p50/p90/p99 (pips) | Zero-spread bars | STOP |",
             "|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|--:|:-:|"]
    for r in reports:
        s = r["spread_pips"]
        lines.append(f"| {r['symbol']} | {r['first'][:10]} | {r['last'][:10]} | {r['bars']:,} | "
                     f"{r['missing_hours']:,} of {r['expected_hours']:,} | {r['missing_share']:.2%} | "
                     f"{r['longest_gap_hours']} | {r['weekend_bars']} | {r['dropped_bad_rows']} | "
                     f"{r['spikes']} | {s['p50']}/{s['p90']}/{s['p99']} | {r['zero_spread_share']:.0%} | "
                     f"{'**yes**' if r['stop'] else 'no'} |")
    return "\n".join(lines)
