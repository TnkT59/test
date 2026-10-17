"""Event-driven backtest of one strategy on one symbol (PLAN.md sections 5-6).

Inputs:
  * Signals on the strategy's own bars ("coarse" bars, e.g. H1), built from
    the contract in fusion/strategies/base.py.
  * Fine bars (M15 here) with bid OHLC and an ask-minus-bid estimate
    (`spread_cons`). Each coarse bar is resolved on the fine bars inside it.

Fills, against the trader on every one:
  * buys at ask (bid + spread), sells at bid, plus slippage per fill;
  * stop orders (entries and exits) fill at the level, or at the open if price
    gapped through it; limit entries fill at the level, or at the open if it
    was already better;
  * profit-taking exits (targets, exit limits) fill at their level, never
    better: a resting order would have filled at the level as soon as price
    reached it, and price may have passed it inside a bar where this model
    does not allow taking profit (the entry bar);
  * inside one fine bar, adverse exits (protective stop, exit stops) beat
    favourable ones (target, exit limits); a position opened inside a fine bar
    may be stopped in that bar (worst case) but not take profit in it;
  * when several entry levels trigger in one fine bar, the one nearest the
    bar's open is taken as first (a path assumption, not the worst case);
  * after an adverse exit, only an entry stop further along the same move may
    fill in that bar, and its stop is judged from the bar's far extreme to its
    close (that extreme came before the entry); after a profit-taking exit, no
    new entry in that bar;
  * an entry whose fill is beyond its own stop is cancelled; one beyond its
    target gets no target.

Sizing: risk `risk_pct` of current equity on the planned loss, decided before
the fill: |entry level - stop| + 2 x slippage for stop and limit orders, the
stop distance + 2 x slippage for market orders. Rounded down to 0.01 lot and
capped at `max_lev` x equity in notional.

Costs: commission per CostSpec at exit; swap for every 17:00 New York roll the
position is held through (three times on the triple-swap weekday). Swap must
be known in config/costs.toml before any position may be held overnight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import NO_LIMIT, Signals
from ..timeutil import NY, ROLL_HOUR
from .costs import CostSpec

NS_MIN = 60_000_000_000


class SwapUnknown(RuntimeError):
    pass


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series            # mark-to-market equity at each fine-bar close
    exposure: float              # share of fine bars with an open position
    counts: dict = field(default_factory=dict)


def roll_times(start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """UTC ns of every weekday 17:00 New York roll from the day before start to
    the day after end, and each roll's weekday (0 = Monday)."""
    days = pd.bdate_range(start.tz_convert(NY).normalize().tz_localize(None) - pd.Timedelta(days=1),
                          end.tz_convert(NY).normalize().tz_localize(None) + pd.Timedelta(days=1))
    rolls = (days + pd.Timedelta(hours=ROLL_HOUR)).tz_localize(NY).tz_convert("UTC")
    return rolls.as_unit("ns").asi8, np.asarray(days.dayofweek)


def run_backtest(sig: Signals, fine: pd.DataFrame, cost: CostSpec, *, risk_pct: float = 0.005,
                 start_equity: float = 100_000.0, period: tuple | None = None,
                 with_costs: bool = True, allow_missing_swap: bool = False) -> Result:
    co = sig.index.as_unit("ns").asi8
    ft = fine.index.as_unit("ns").asi8
    if len(co) < 2 or len(ft) < 1:
        raise ValueError("not enough bars")
    flen = int(np.median(np.diff(ft))) if len(ft) > 1 else 15 * NS_MIN
    cid = np.searchsorted(co, ft, side="right") - 1
    if (cid < 0).any():
        fine, ft, cid = fine[cid >= 0], ft[cid >= 0], cid[cid >= 0]
    f0 = np.searchsorted(cid, np.arange(len(co)), side="left")
    f1 = np.searchsorted(cid, np.arange(len(co)), side="right")

    o, h, l, c = (fine[k].to_numpy(float).tolist() for k in ("open", "high", "low", "close"))
    sp = (fine["spread_cons"].to_numpy(float) if with_costs else np.zeros(len(ft))).tolist()
    ft_l = ft.tolist()
    slip = cost.slip if with_costs else 0.0
    p_from = period[0].as_unit("ns").value if period else -(2**63)
    p_to = period[1].as_unit("ns").value if period else 2**63 - 1

    S = {k: getattr(sig, k).tolist() for k in (
        "long_mkt", "short_mkt", "long_stop", "short_stop", "long_limit", "short_limit",
        "stop_long_px", "stop_short_px", "stop_long_dist", "stop_short_dist",
        "target_long_px", "target_short_px", "exit_long_mkt", "exit_short_mkt",
        "exit_long_stop", "exit_short_stop", "exit_long_limit", "exit_short_limit",
        "flat_at_close", "order_expiry", "exit_deadline", "group")}
    rolls, roll_dow = roll_times(fine.index[0], fine.index[-1])

    cash = start_equity
    equity = np.full(len(ft), np.nan)
    in_pos = np.zeros(len(ft), dtype=bool)
    trades = []
    counts = dict(cancelled_beyond_stop=0, too_small=0, capped=0, expired=0, group_cap_blocks=0)
    used: dict = {}
    pos = None   # dict while a position is open

    fin = lambda x: x == x and x is not None        # not NaN

    def close_pos(j: int, px: float, reason: str, at_open: bool, at_close: bool = False):
        nonlocal cash, pos
        d, units = pos["dir"], pos["units"]
        t_exit = ft_l[j] if at_open else (ft_l[j] + flen if at_close else ft_l[j] + 1)
        pnl = cost.to_usd(d * (px - pos["entry"]) * units, px)
        comm = cost.commission_usd(units, pos["entry"]) if with_costs else 0.0
        # rolls after the entry and up to the exit: a position closed at the
        # close of the 16:45 bar (17:00) is out before the roll; one closed at
        # the open of the bar that starts at 17:00 was held through it
        a = np.searchsorted(rolls, pos["t_in"], side="right")
        b = np.searchsorted(rolls, t_exit, side="right" if at_open else "left")
        held = roll_dow[a:b] if b > a else np.array([], dtype=int)
        nights = int(len(held) + 2 * int((held == cost.triple_swap_day).sum()))
        swap = 0.0
        if nights and with_costs:
            if not cost.swap_known():
                if not allow_missing_swap:
                    raise SwapUnknown(f"{cost.symbol}: swap not set in config/costs.toml "
                                      f"(position held over {nights} roll(s))")
            else:
                swap = cost.swap_usd(d, units, nights, px)
        net = pnl - comm + swap
        cash += net
        trades.append(dict(dir=d, t_in=pos["t_in"], t_out=t_exit, entry=pos["entry"], exit=px,
                           stop=pos["stop"], target=pos["target"], units=units,
                           lots=units / cost.lot, risk_usd=pos["risk_usd"], commission=comm,
                           swap=swap, nights=nights, pnl_usd=net, R=net / pos["risk_usd"],
                           reason_in=pos["reason"], reason_out=reason, group=pos["group"],
                           capped=pos["capped"]))
        pos = None

    def open_pos(j: int, d: int, fill: float, ref: float, stop_px: float, stop_dist: float,
                 target: float, reason: str, t: int, intrabar: bool) -> bool:
        nonlocal pos
        if fin(stop_px):
            stop = stop_px
            plan = abs(ref - stop) + 2 * slip
        else:
            stop = fill - d * stop_dist
            plan = stop_dist + 2 * slip
        if (d == 1 and fill <= stop) or (d == -1 and fill >= stop):
            counts["cancelled_beyond_stop"] += 1
            return False
        if fin(target) and ((d == 1 and fill >= target) or (d == -1 and fill <= target)):
            target = float("nan")
        dist_usd = cost.to_usd(plan, ref)
        if not dist_usd > 0:
            return False
        units = risk_pct * cash / dist_usd
        capped = cost.notional_usd(units, ref) > cost.max_lev * cash
        if capped:
            units = cost.max_lev * cash / cost.notional_usd(1.0, ref)
        step = cost.lot * 0.01
        units = math.floor(units / step + 1e-9) * step
        if units <= 0:
            counts["too_small"] += 1
            return False
        counts["capped"] += capped
        g = S["group"][t]
        key = (g, d) if sig.per_side else g
        used[key] = used.get(key, 0) + 1
        pos = dict(dir=d, entry=fill, stop=stop, target=target, units=units,
                   risk_usd=units * dist_usd, t_in=ft_l[j] + (1 if intrabar else 0),
                   deadline=S["exit_deadline"][t], entry_j=j, intrabar=intrabar,
                   reason=reason, group=g, capped=capped)
        return True

    def cap_ok(t: int, d: int) -> bool:
        if sig.max_entries_per_group is None:
            return True
        g = S["group"][t]
        key = (g, d) if sig.per_side else g
        if used.get(key, 0) >= sig.max_entries_per_group:
            counts["group_cap_blocks"] += 1
            return False
        return True

    def resolve_exits(j: int, t: int):
        """Exit an open position inside fine bar j. Returns (direction of the
        move that caused an adverse exit, exit price), "favourable" after a
        profit-taking exit, or None. (An intrabar entry is resolved for its
        own bar in try_entries, so profit is never taken in that bar.)"""
        d = pos["dir"]
        ask_h, ask_l, ask_o = h[j] + sp[j], l[j] + sp[j], o[j] + sp[j]
        if d == 1:
            adv = [x for x in (pos["stop"], S["exit_long_stop"][t]) if fin(x) and l[j] <= x]
            if adv:
                px = min(max(adv), o[j]) - slip
                close_pos(j, px, "stop" if max(adv) == pos["stop"] else "exit_stop", False)
                return (-1, px + slip)
            fav = [x for x in (pos["target"], S["exit_long_limit"][t]) if fin(x) and h[j] >= x]
            if fav:
                px = min(fav) - slip          # at the level, never better (see module doc)
                close_pos(j, px, "target" if min(fav) == pos["target"] else "exit_limit", False)
                return "favourable"
        else:
            adv = [x for x in (pos["stop"], S["exit_short_stop"][t]) if fin(x) and ask_h >= x]
            if adv:
                px = max(min(adv), ask_o) + slip
                close_pos(j, px, "stop" if min(adv) == pos["stop"] else "exit_stop", False)
                return (1, px - slip)
            fav = [x for x in (pos["target"], S["exit_short_limit"][t]) if fin(x) and ask_l <= x]
            if fav:
                px = max(fav) + slip
                close_pos(j, px, "target" if max(fav) == pos["target"] else "exit_limit", False)
                return "favourable"
        return None

    def try_entries(j: int, t: int, after_exit: tuple | None):
        ask_h, ask_l, ask_o = h[j] + sp[j], l[j] + sp[j], o[j] + sp[j]
        cand = []
        L = S["long_stop"][t]
        if fin(L) and ask_h >= L and cap_ok(t, 1):
            cand.append((abs(L - ask_o) if ask_o < L else 0.0, 1, max(L, ask_o) + slip, L, "stop"))
        L = S["short_stop"][t]
        if fin(L) and l[j] <= L and cap_ok(t, -1):
            cand.append((abs(o[j] - L) if o[j] > L else 0.0, -1, min(L, o[j]) - slip, L, "stop"))
        if after_exit is None:
            L = S["long_limit"][t]
            if fin(L) and ask_l <= L and cap_ok(t, 1):
                cand.append((abs(ask_o - L) if ask_o > L else 0.0, 1, min(L, ask_o) + slip, L, "limit"))
            L = S["short_limit"][t]
            if fin(L) and h[j] >= L and cap_ok(t, -1):
                cand.append((abs(L - o[j]) if o[j] < L else 0.0, -1, max(L, o[j]) - slip, L, "limit"))
        else:
            move, exit_px = after_exit
            # only an entry stop further along the move that caused the exit
            cand = [x for x in cand if x[1] == move and x[4] == "stop"
                    and (x[3] <= exit_px if move == -1 else x[3] >= exit_px)]
        if not cand:
            return
        cand.sort(key=lambda x: x[0])
        _, d, fill, level, kind = cand[0]
        sfx = "long" if d == 1 else "short"
        if not open_pos(j, d, fill, level, S[f"stop_{sfx}_px"][t], S[f"stop_{sfx}_dist"][t],
                        S[f"target_{sfx}_px"][t], kind, t, intrabar=True):
            return
        # same fine bar: the stop (or an exit stop) may already be hit. After
        # an exit in this bar, price was travelling through the exit towards
        # this entry, so the bar's far extreme came before it and only the
        # move from there to the close counts; otherwise assume the worst.
        # The fill is kept inside the prices the bar traded.
        xs = S["exit_long_stop"][t] if d == 1 else S["exit_short_stop"][t]
        levels = [x for x in (pos["stop"], xs) if fin(x)]
        if d == 1:
            lvl = max(levels)                 # the first adverse level price meets
            low = c[j] if after_exit is not None else l[j]
            if low <= lvl:
                close_pos(j, min(lvl, h[j]) - slip, "stop" if lvl == pos["stop"] else "exit_stop", False)
        else:
            lvl = min(levels)
            high = c[j] + sp[j] if after_exit is not None else h[j] + sp[j]
            if high >= lvl:
                close_pos(j, max(lvl, l[j] + sp[j]) + slip, "stop" if lvl == pos["stop"] else "exit_stop",
                          False)

    n = len(co)
    for b in range(1, n):
        t = b - 1
        j0, j1 = int(f0[b]), int(f1[b])
        if j0 >= j1:
            continue
        in_period = p_from <= co[b] < p_to
        honoured = in_period and co[b] < S["order_expiry"][t]
        if in_period and not honoured and (S["long_mkt"][t] or S["short_mkt"][t] or any(
                fin(S[k][t]) for k in ("long_stop", "short_stop", "long_limit", "short_limit"))):
            counts["expired"] += 1
        # --- at the open of the first fine bar
        if pos is not None and ft_l[j0] >= pos["deadline"]:
            close_pos(j0, o[j0] - slip if pos["dir"] == 1 else o[j0] + sp[j0] + slip,
                      "deadline", True)
        if pos is not None and ((pos["dir"] == 1 and S["exit_long_mkt"][t]) or
                                (pos["dir"] == -1 and S["exit_short_mkt"][t])):
            close_pos(j0, o[j0] - slip if pos["dir"] == 1 else o[j0] + sp[j0] + slip,
                      "signal", True)
        if pos is None and honoured:
            if S["long_mkt"][t] and cap_ok(t, 1):
                fill = o[j0] + sp[j0] + slip
                if open_pos(j0, 1, fill, fill, S["stop_long_px"][t], S["stop_long_dist"][t],
                            S["target_long_px"][t], "market", t, intrabar=False):
                    pass
            elif S["short_mkt"][t] and cap_ok(t, -1):
                fill = o[j0] - slip
                open_pos(j0, -1, fill, fill, S["stop_short_px"][t], S["stop_short_dist"][t],
                         S["target_short_px"][t], "market", t, intrabar=False)
        # --- inside the fine bars
        live_entries = honoured and any(fin(S[k][t]) for k in
                                        ("long_stop", "short_stop", "long_limit", "short_limit"))
        for j in range(j0, j1):
            if ft_l[j] >= p_to:
                break
            if pos is not None and j > j0 and ft_l[j] >= pos["deadline"]:
                close_pos(j, o[j] - slip if pos["dir"] == 1 else o[j] + sp[j] + slip,
                          "deadline", True)
            after = None
            if pos is not None:
                after = resolve_exits(j, t)
            if pos is None and live_entries and after != "favourable":
                try_entries(j, t, after)
            if pos is not None and (j + 1 >= len(ft_l) or ft_l[j + 1] >= p_to):
                # the period ends inside this strategy bar: close at the last
                # fine bar of the period
                close_pos(j, c[j] - slip if pos["dir"] == 1 else c[j] + sp[j] + slip,
                          "period_end", False, at_close=True)
            if pos is not None:
                in_pos[j] = True
                mark = c[j] if pos["dir"] == 1 else c[j] + sp[j]
                equity[j] = cash + cost.to_usd(pos["dir"] * (mark - pos["entry"]) * pos["units"], mark)
            else:
                equity[j] = cash
        # --- at the close of the last fine bar
        jl = j1 - 1
        if pos is not None and S["flat_at_close"][b] and ft_l[jl] < p_to:
            px = c[jl] - slip if pos["dir"] == 1 else c[jl] + sp[jl] + slip
            close_pos(jl, px, "flat", False, at_close=True)
            equity[jl] = cash

    eq = pd.Series(equity, index=fine.index[:len(equity)])
    if period:
        eq = eq[(eq.index >= period[0]) & (eq.index < period[1])]
    eq = eq.ffill().fillna(start_equity)
    tr = pd.DataFrame(trades)
    inside = (ft >= p_from) & (ft < p_to)
    exposure = float(in_pos[inside].mean()) if inside.any() else 0.0
    return Result(trades=tr, equity=eq, exposure=exposure, counts=counts)
