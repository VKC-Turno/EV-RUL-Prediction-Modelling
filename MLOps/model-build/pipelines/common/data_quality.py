"""Data-quality gates shared by every OEM preprocessing step.

Two jobs:
  1. `clip_sentinels` — bound each raw channel to physical range (garbage -> NaN). Same bounds as the
     research repo README §2. Run this in the compaction step too, so garbage never reaches SoH code.
  2. `apply_quality` — drop data-thin (vin, month) rows the models can't trust. Mirrors src/data_quality.py.

These bounds also seed the SageMaker Model Monitor data-quality baseline (the rejection rate is a drift signal).
"""
import numpy as np
import pandas as pd

# physical bounds — anything outside -> NaN
BOUNDS = {
    "soc": (0.0, 100.0),
    "current": (-400.0, 400.0),          # |I| <= 400 A
    "batteryVoltage": (20.0, 120.0),
    "batteryPackVoltage": (20.0, 120.0),
    "resCapacity": (1.0, 500.0),
    "batteryRemainingCapacity": (1.0, 500.0),
}

MIN_ROWS_PER_MONTH = 5     # a (vin, month) needs at least this many clean rows to be usable


def clip_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Null out-of-range values in place-ish (returns a copy). Unknown columns pass through untouched."""
    out = df.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce")
            out[col] = v.where((v >= lo) & (v <= hi))
    return out


def apply_quality(m: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    """Keep vehicles with enough history; drop data-thin vin-months. `m` is the monthly feature frame.

    Conservative on purpose: we drop data-thin vehicles, NOT flat ones (a flat SoH is a real signal).
    """
    if "n_sessions" in m.columns:
        m = m[pd.to_numeric(m["n_sessions"], errors="coerce").fillna(0) > 0]
    counts = m.groupby("vin")["month"].transform("size")
    m = m[counts >= min_months]
    return m.sort_values(["vin", "month"]).reset_index(drop=True)
