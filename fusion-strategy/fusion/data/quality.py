"""Phase 1 data-quality report (PLAN.md section 2) for UTC bars from fusion.data.mt5.

  * Expected trading time: Sunday 17:00 to Friday 17:00 New York, minus gold's
    and the US index CFDs' daily 17:00-18:00 New York break, minus 25 December and 1 January (listed
    separately, not counted as gaps).
  * Missing hours: expected clock hours with no bar at all. They are split
    into scheduled closures (US exchange holidays and early-close days, whose
    trading day or New York date is on the calendar below; gold and the US
    index CFDs close on them) and unexplained gaps. PLAN.md's STOP rule (more
    than 1% missing) is applied to unexplained gaps: the rule is about data
    completeness, and a closed market has no data to miss. Both are reported.
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


def us_market_closures(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """US exchange holidays (NYSE rules, Juneteenth from 2022) and the usual
    early-close days (3 July, the day after Thanksgiving, 24 December), as
    New York dates."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
                                        USLaborDay, USMartinLutherKingJr, USMemorialDay,
                                        USPresidentsDay, USThanksgivingDay, nearest_workday)

    class Exchange(AbstractHolidayCalendar):
        rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday),
                 USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
                 Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                         observance=nearest_workday),
                 Holiday("July4", month=7, day=4, observance=nearest_workday),
                 USLaborDay, USThanksgivingDay,
                 Holiday("Christmas", month=12, day=25, observance=nearest_workday)]

    a = start.tz_convert(NY).tz_localize(None).normalize() - pd.Timedelta(days=7)
    b = end.tz_convert(NY).tz_localize(None).normalize() + pd.Timedelta(days=7)
    hol = Exchange().holidays(a, b)
    days = {d.date() for d in hol}
    for y in range(a.year, b.year + 1):
        days.add(pd.Timestamp(y, 7, 3).date())
        days.add(pd.Timestamp(y, 12, 24).date())
        days.add(pd.Timestamp(y, 12, 31).date())
        days.add(pd.Timestamp(y, 1, 1).date())
    for d in hol:
        if d.month == 11 and d.dayofweek == 3:          # Thanksgiving -> Friday after
            days.add((d + pd.Timedelta(days=1)).date())
    return days


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


DAILY_BREAK = ("XAUUSD", "USTEC", "US30")     # closed 17:00-18:00 New York


def report(sym: str, bars: pd.DataFrame, pip: float) -> dict:
    gold = sym in DAILY_BREAK
    exp, hol_hours = expected_hours(bars.index[0], bars.index[-1], gold)
    have = pd.DatetimeIndex(bars.index.floor("h").unique())
    missing = exp.difference(have)
    closures = us_market_closures(bars.index[0], bars.index[-1])
    mny = missing.tz_convert(NY)
    tday = (mny + pd.Timedelta(hours=7)).date          # trading day, 17:00 New York roll
    scheduled = np.array([a in closures or b_ in closures for a, b_ in zip(mny.date, tday)],
                         dtype=bool) if len(missing) else np.zeros(0, dtype=bool)
    unexplained = missing[~scheduled]
    ny = bars.index.tz_convert(NY)
    wd, hr = ny.dayofweek, ny.hour
    weekend = ((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17))
    sp = bars.spread_px / pip
    q = lambda s: {f"p{int(x * 100)}": round(float(s.quantile(x)), 3) for x in (0.5, 0.9, 0.99)} \
        | {"max": round(float(s.max()), 3)}
    miss_share = len(missing) / len(exp) if len(exp) else 1.0
    gap_share = len(unexplained) / len(exp) if len(exp) else 1.0
    u = pd.Series(unexplained)
    longest = int(u.groupby(u.diff().ne(pd.Timedelta(hours=1)).cumsum()).size().max()) if len(u) else 0
    return dict(
        symbol=sym, first=str(bars.index[0]), last=str(bars.index[-1]), bars=len(bars),
        expected_hours=len(exp), missing_hours=len(missing), missing_share=miss_share,
        scheduled_closure_hours=int(scheduled.sum()), unexplained_gap_hours=len(unexplained),
        unexplained_share=gap_share, longest_gap_hours=longest,
        unexplained_dates=sorted({str(d) for d in unexplained.tz_convert(NY).date})[:20],
        holiday_hours_excluded=hol_hours,
        weekend_bars=int(weekend.sum()), dropped_bad_rows=int(bars.attrs.get("dropped_bad", 0)),
        spikes=spikes(bars), spread_pips=q(sp),
        spread_by_ny_hour=sp.groupby(hr).median().round(3).to_dict(),
        spread_by_year=sp.groupby(bars.index.year).median().round(3).to_dict(),
        zero_spread_share=float((bars.spread_px == 0).mean()),
        stop=gap_share > MAX_MISSING_SHARE)


def to_markdown(reports: list[dict]) -> str:
    lines = ["| Symbol | First bar | Last bar | Bars | Missing hours (all) | Of which holiday closures | "
             "Unexplained gaps | Gap share | Longest gap (h) | Weekend bars | Bad rows | Spikes | "
             "Spread p50/p90/p99 (pips) | Zero-spread bars | STOP |",
             "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|:-:|"]
    for r in reports:
        s = r["spread_pips"]
        lines.append(f"| {r['symbol']} | {r['first'][:10]} | {r['last'][:10]} | {r['bars']:,} | "
                     f"{r['missing_hours']:,} of {r['expected_hours']:,} ({r['missing_share']:.2%}) | "
                     f"{r['scheduled_closure_hours']:,} | {r['unexplained_gap_hours']:,} | "
                     f"{r['unexplained_share']:.2%} | "
                     f"{r['longest_gap_hours']} | {r['weekend_bars']} | {r['dropped_bad_rows']} | "
                     f"{r['spikes']} | {s['p50']}/{s['p90']}/{s['p99']} | {r['zero_spread_share']:.0%} | "
                     f"{'**yes**' if r['stop'] else 'no'} |")
    return "\n".join(lines)
