"""MT5 bar exports -> UTC bars with bid OHLC and spreads.

The functions below are a maintained copy of the loader in
research/orb_sessions.py, which was reviewed and tested there (that script is
kept unchanged so the published ORB results stay reproducible). MT5 exports
are bid bars in broker server time with one spread value per bar, reportedly
the bar's minimum, in points of 10**-digits.
"""
from __future__ import annotations

import pathlib
import re
import sys
from io import StringIO

import numpy as np
import pandas as pd

SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USTEC", "US30")
ALIASES = {"GOLD": "XAUUSD"}
FX = ("EURUSD", "GBPUSD", "USDJPY")
BAR_MINUTES_ALLOWED = (1, 5, 15)
CANDIDATE_TZ = ("NY+7", "Europe/Helsinki", "Europe/London",
                "Europe/Berlin", "UTC", "Etc/GMT-1", "Etc/GMT-2", "Etc/GMT-3")
FX_OPEN_NY_HOUR = 17   # spot FX reopens Sunday 17:00 New York
SPREAD_LOOKBACK_DAYS = 20
SPREAD_QUANTILE = 0.9


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) > 1 and raw[1:2] == b"\x00":
        return raw.decode("utf-16-le")
    return raw.decode("utf-8-sig")


def read_mt5_csv(path: pathlib.Path) -> pd.DataFrame:
    """Read an MT5 bar export. Returns server-time-indexed open/high/low/close/spread."""
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
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in need})
    out.index = pd.DatetimeIndex(idx, name="server_time")
    bar_minutes(out.index, path.name)
    out.attrs["digits"] = text_digits(df["close"])
    return out


def text_digits(col: pd.Series) -> int:
    """Price digits as MT5 printed them (e.g. '33361.50' -> 2). MT5 writes
    every price with the symbol's digits, and the <SPREAD> column is in units
    of 10**-digits, so this must come from the text, not the float values."""
    frac = col.dropna().str.strip().str.partition(".")[2]
    return int(frac.str.len().max()) if len(frac) else 0


def bar_minutes(idx: pd.DatetimeIndex, name: str = "") -> int:
    spacing = idx.to_series().diff().dropna()
    m = spacing.median() / pd.Timedelta(minutes=1) if len(spacing) else float("nan")
    if m not in BAR_MINUTES_ALLOWED:
        raise ValueError(f"{name}: median bar spacing is {spacing.median()}, "
                         f"expected one of M{BAR_MINUTES_ALLOWED}")
    return int(m)


def symbol_of(path: pathlib.Path) -> str | None:
    name = path.stem.upper()
    for s in SYMBOLS + tuple(ALIASES):
        if s in name:
            return ALIASES.get(s, s)
    return None


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


def conservative_spread(spread_px: pd.Series) -> pd.Series:
    """max(bar spread, 90th percentile of the spread at the same New York
    minute of day over the previous 20 observations). Past bars only."""
    ny = spread_px.index.tz_convert("America/New_York")
    key = ny.hour * 60 + ny.minute
    hist = (spread_px.groupby(key)
            .transform(lambda s: s.rolling(SPREAD_LOOKBACK_DAYS, min_periods=5)
                       .quantile(SPREAD_QUANTILE).shift(1)))
    return np.maximum(spread_px, hist.fillna(spread_px))


def load_raw(files) -> pd.DataFrame:
    parts = [read_mt5_csv(f) for f in files]
    sizes = {bar_minutes(p.index, f.name) for p, f in zip(parts, files)}
    if len(sizes) != 1:
        raise ValueError(f"mixed bar sizes {sorted(sizes)} in {[f.name for f in files]}")
    digits = {p.attrs["digits"] for p in parts}
    if len(digits) != 1:
        raise ValueError(f"mixed price digits {sorted(digits)} in {[f.name for f in files]}")
    df = pd.concat(parts).sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="first")].copy()
    df.attrs["digits"] = digits.pop()
    return df


def resolve_server_tz(raw: dict[str, pd.DataFrame], override: str | None = None):
    """One timezone for the whole broker, detected on the FX symbols."""
    if override:
        return override, {}
    fx = [s for s in raw if s in FX]
    if not fx:
        raise ValueError("no FX symbol to detect the broker timezone from; pass a server tz")
    found = {s: detect_server_tz(raw[s].index) for s in fx}
    tzs = {tz for tz, _ in found.values()}
    if len(tzs) != 1 or min(sc for _, sc in found.values()) < 0.9:
        raise ValueError(f"server timezone unclear: {found}; pass a server tz")
    return tzs.pop(), found


def to_utc_bars(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Server-time bid bars -> UTC bars with columns open/high/low/close (bid),
    spread_px (raw), spread_cons (conservative ask - bid estimate) and the
    mid OHLC the strategies read (mid = bid + raw spread / 2)."""
    digits = df.attrs["digits"]
    bad = df[["open", "high", "low", "close", "spread"]].isna().any(axis=1) \
        | (df.high < df.low) | (df.spread < 0)
    out = df[~bad].copy()
    out.index = server_to_utc(out.index, tz)
    out = out[~out.index.isna()].sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="first")]
    out["spread_px"] = out.spread * 10.0 ** -digits
    out["spread_cons"] = conservative_spread(out.spread_px)
    out.attrs.update(digits=digits, point=10.0 ** -digits, dropped_bad=int(bad.sum()))
    return out


def mid_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Mid OHLC for indicators: bid + half the raw bar spread."""
    half = bars.spread_px / 2
    return pd.DataFrame({c: bars[c] + half for c in ("open", "high", "low", "close")},
                        index=bars.index)


def files_by_symbol(data_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    out: dict[str, list[pathlib.Path]] = {}
    for f in sorted(data_dir.glob("*.csv")):
        s = symbol_of(f)
        if s:
            out.setdefault(s, []).append(f)
    return out
