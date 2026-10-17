"""The fixed data split (PLAN.md, hard rule). Never tuned after the holdout.

In-sample 2015-2021, out-of-sample 2022-2024, holdout 2025 onward. Phase 3
and 4 code loads data through `cap_before_holdout`, which drops every bar from
2025-01-01 on, so the holdout cannot be read by accident. Only Phase 5 may
open it, once, through `open_holdout`, which writes a lock file and refuses a
second opening.
"""
from __future__ import annotations

import pathlib

import pandas as pd

IN_SAMPLE = (pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"))
OUT_OF_SAMPLE = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
LOCK = pathlib.Path(__file__).resolve().parents[2] / "results" / "HOLDOUT_LOCK"


def cap_before_holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index < HOLDOUT_START]


def period(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return {"IS": IN_SAMPLE, "OOS": OUT_OF_SAMPLE}[name]


def open_holdout(frozen_config_hash: str) -> None:
    """Phase 5 only. Records the frozen config and refuses a second opening."""
    if LOCK.exists():
        raise PermissionError(f"holdout already opened: {LOCK.read_text().strip()}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"opened with config {frozen_config_hash} at {pd.Timestamp.now(tz='UTC')}\n")
