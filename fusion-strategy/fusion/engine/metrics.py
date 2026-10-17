"""Performance metrics (PLAN.md section 5), from trades and the
mark-to-market equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import trading_day


def _t(r: np.ndarray) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def summarize(trades: pd.DataFrame, equity: pd.Series, start_equity: float,
              period: tuple[pd.Timestamp, pd.Timestamp], exposure: float) -> dict:
    years = (period[1] - period[0]).days / 365.25
    eq = equity.to_numpy()
    final = float(eq[-1]) if len(eq) else start_equity
    peak = np.maximum.accumulate(np.concatenate([[start_equity], eq]))
    dd = float(((peak - np.concatenate([[start_equity], eq])) / peak).max())
    daily = equity.groupby(trading_day(equity.index)).last()
    rets = daily.pct_change().fillna(daily.iloc[0] / start_equity - 1) if len(daily) else daily
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    out = dict(trades=int(len(trades)), net_profit=final - start_equity,
               cagr=(final / start_equity) ** (1 / years) - 1 if final > 0 and years > 0 else -1.0,
               max_dd=dd, sharpe=sharpe, exposure=exposure)
    if len(trades):
        p = trades.pnl_usd
        wins, losses = p[p > 0].sum(), -p[p < 0].sum()
        r = trades.R.to_numpy()
        out.update(profit_factor=float(wins / losses) if losses > 0 else float("inf"),
                   win_rate=float((p > 0).mean()), avg_R=float(r.mean()), t_stat_R=_t(r),
                   n_long=int((trades.dir == 1).sum()), n_short=int((trades.dir == -1).sum()),
                   swap_usd=float(trades.swap.sum()), commission_usd=float(trades.commission.sum()))
    else:
        out.update(profit_factor=float("nan"), win_rate=float("nan"), avg_R=float("nan"),
                   t_stat_R=0.0, n_long=0, n_short=0, swap_usd=0.0, commission_usd=0.0)
    return out


def survives(m: dict) -> bool:
    """Phase 3 gate (PLAN.md): out-of-sample PF > 1.15, >= 100 trades, max DD < 25%."""
    return m["trades"] >= 100 and m["profit_factor"] > 1.15 and m["max_dd"] < 0.25
