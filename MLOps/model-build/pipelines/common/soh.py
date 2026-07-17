"""Per-feed SoH target — branch on what the OEM's telemetry carries.

    signed current + SoC ............ coulomb()        (Mahindra via intellicar, Piaggio via intellicar)
    remaining-capacity, no current .. bms_capacity()   (Euler, Montra)
    reported SoH only ............... reported()        (Bajaj)

Every method returns a per-(vin, month) frame [vin, month, soh_raw, soh] where `soh` is the
monotone-decreasing envelope of `soh_raw`, anchored <= 100. The forecasters train on `soh`.

Reference (canonical) implementations in the research repo:
    coulomb       -> src/soh.py  (coulomb_capacity_monthly -> capacity_to_soh)
    bms_capacity  -> src/euler_bms_soh.py / src/montra_features.py
    reported      -> src/bajaj_model.py path
This module is the pipeline-facing, dependency-light version (pandas + scikit-learn only).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# knobs (mirror src/soh.py)
MIN_DSOC = 2.0          # % — ignore charge/discharge segments smaller than this
CAP_BOUNDS = (40.0, 400.0)
HIGH_SOC = (95.0, 100.0)   # BMS-capacity is read near full charge
MAX_DROP_PP = 6.0          # cap implausible month-to-month SoH jumps


def _isotonic_envelope(age_months, soh_raw):
    """Non-increasing fit of soh_raw vs age, clipped to <= 100. NaN-safe."""
    a = np.asarray(age_months, float)
    y = np.asarray(soh_raw, float)
    ok = np.isfinite(a) & np.isfinite(y)
    out = np.full(len(y), np.nan)
    if ok.sum() >= 2:
        iso = IsotonicRegression(increasing=False, y_max=100.0, out_of_bounds="clip")
        out[ok] = np.minimum(iso.fit_transform(a[ok], y[ok]), 100.0)
    return out


def _monthly(df, value_col, age_col="age_months"):
    """month-median of value_col -> [vin, month, age_months, soh_raw] then isotonic per vin."""
    g = (df.dropna(subset=[value_col])
           .groupby(["vin", "month"])
           .agg(soh_raw=(value_col, "median"), age_months=(age_col, "max"))
           .reset_index())
    parts = []
    for vin, sub in g.groupby("vin"):
        sub = sub.sort_values("month").copy()
        sub["soh"] = _isotonic_envelope(sub["age_months"], sub["soh_raw"])
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else g.assign(soh=np.nan)


# ── method 1: coulomb counting ───────────────────────────────────────────────────────────
def coulomb(df, cap0=None):
    """ΔSoC-weighted pooled capacity: Σ|∫I·dt| / Σ(|ΔSoC|/100) per (vin, month) -> normalise to cap0 -> SoH.

    `df` needs signed `current` (A) + `soc` (%) + `t` (datetime) + `age_months`.
    """
    d = df.dropna(subset=["current", "soc", "t"]).sort_values(["vin", "t"]).copy()
    d["dt_h"] = d.groupby("vin")["t"].diff().dt.total_seconds() / 3600.0
    d["dsoc"] = d.groupby("vin")["soc"].diff()
    d["dah"] = (d["current"].abs() * d["dt_h"]).where(d["dt_h"].between(0, 1.0))  # ignore long gaps
    d = d[d["dsoc"].abs() >= MIN_DSOC]
    d["month"] = d["t"].values.astype("datetime64[M]")
    agg = (d.groupby(["vin", "month"])
             .agg(ah=("dah", "sum"), dsoc=("dsoc", lambda s: s.abs().sum() / 100.0),
                  age_months=("age_months", "max")).reset_index())
    agg = agg[agg["dsoc"] > 0]
    agg["cap_ah"] = (agg["ah"] / agg["dsoc"]).clip(*CAP_BOUNDS)
    cap0 = cap0 or agg.groupby("vin")["cap_ah"].transform("max")
    agg["soh_raw"] = (agg["cap_ah"] / cap0 * 100.0).clip(upper=100.0)
    return _finalise(agg)


# ── method 2: BMS remaining-capacity ─────────────────────────────────────────────────────
def bms_capacity(df, cap_col=None, cap0=None):
    """resCapacity / (SoC/100) read near full charge -> full-capacity -> normalise to cap0 -> SoH."""
    cap_col = cap_col or next((c for c in ("resCapacity", "batteryRemainingCapacity") if c in df.columns), None)
    if cap_col is None:
        raise ValueError("bms_capacity needs a remaining-capacity column (resCapacity / batteryRemainingCapacity)")
    d = df.dropna(subset=[cap_col, "soc"]).copy()
    d = d[d["soc"].between(*HIGH_SOC)]
    d["cap_full"] = (d[cap_col] / (d["soc"] / 100.0)).clip(*CAP_BOUNDS)
    d["month"] = pd.to_datetime(d["t"]).values.astype("datetime64[M]") if "t" in d.columns else d["month"]
    agg = (d.groupby(["vin", "month"])
             .agg(cap_full=("cap_full", "median"), age_months=("age_months", "max")).reset_index())
    cap0 = cap0 or agg.groupby("vin")["cap_full"].transform("max")
    agg["soh_raw"] = (agg["cap_full"] / cap0 * 100.0).clip(upper=100.0)
    return _finalise(agg)


# ── method 3: reported SoH ───────────────────────────────────────────────────────────────
def reported(df, soh_col="batterySoh"):
    """Monthly median of the BMS-reported SoH, kept non-increasing (isotonic)."""
    d = df.rename(columns={soh_col: "rep"}).dropna(subset=["rep"]).copy()
    d = d[d["rep"].between(30, 100)]          # drop the 0.0/garbage the reported field carries
    if "month" not in d.columns:
        d["month"] = pd.to_datetime(d["t"]).values.astype("datetime64[M]")
    agg = (d.groupby(["vin", "month"])
             .agg(soh_raw=("rep", "median"), age_months=("age_months", "max")).reset_index())
    return _finalise(agg)


def _finalise(agg):
    parts = []
    for vin, sub in agg.groupby("vin"):
        sub = sub.sort_values("month").copy()
        sub["soh"] = _isotonic_envelope(sub["age_months"], sub["soh_raw"])
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out[["vin", "month", "age_months", "soh_raw", "soh"]]


METHODS = {"coulomb": coulomb, "bms_capacity": bms_capacity, "reported": reported}


def compute(method: str, df, **kw):
    if method not in METHODS:
        raise ValueError(f"unknown SoH method '{method}'. Known: {sorted(METHODS)}")
    return METHODS[method](df, **kw)
