#!/usr/bin/env python3
"""v2 full-charge SoH ported to EULER (the fix the flatness investigation surfaced).

Euler's production SoH = BMS remaining-capacity → normalize → clip → isotonic envelope, which (like
Mahindra) collapses noisy/trendless capacity into flat or artifact curves (30% flat, 24% at 100-clip).
BUT — contrary to the config note — Euler's DENSE feed carries signed current (88%) and pack voltage
(100%). So we can do the same thing v2 did for Mahindra: coulomb-count capacity on FULL charge events.

Euler has two voltage variants (~60 V and ~82 V) and 60 V packs barely move in voltage, so the full-cycle
gate is **SoC-based** here (ΔSoC and end-SoC), not voltage endpoints. Coulomb capacity = ∫I·dt / (ΔSoC/100).
Loads all dense files once (231 files, ~19 M rows) and groups in memory — no per-vin scan, so it's a single
fast pass (no checkpoint/resume needed, unlike the Mahindra intellicar tiny-files feed).

Outputs (parallel to the Mahindra artifacts, so the v2 dashboard can read them):
  data/euler/full_charge_events.parquet · full_charge_soh.parquet · full_charge_summary.parquet · full_charge_report.json
Run: .venv/bin/python src/euler_full_charge_soh.py
"""
import os, json, time, glob
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

DENSE = "data/euler/dense/*.parquet"
FEAT = "data/euler/features/feature_table.parquet"
OUT_EV = "data/euler/full_charge_events.parquet"
OUT_SOH = "data/euler/full_charge_soh.parquet"
OUT_SUM = "data/euler/full_charge_summary.parquet"
OUT_REP = "data/euler/full_charge_report.json"

SOC_LO, SOC_HI = 0.0, 100.0
CUR_MAX = 400.0                  # drop current sentinels (feed has ±1e5 A glitches)
V_LO, V_HI = 20.0, 120.0         # valid pack voltage (60 V and 82 V variants)
CHG_MIN = 2.0
GAP_S = 600.0
MIN_ROWS = 5
FULL_DSOC = 50.0                 # a FULL charge spans >= this much SoC ...
FULL_END = 88.0                  # ... and ends >= this SoC (near-full)
CAP_LO, CAP_HI = 40.0, 320.0     # plausible per-charge capacity (Ah); Euler packs ~130-180 Ah
BASE_AGE_LO, BASE_AGE_HI = 0.5, 12.0


def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def _cv(s):
    s = pd.Series(s).dropna()
    return float(s.std() / s.mean() * 100.0) if (len(s) > 1 and s.mean()) else np.nan


def charge_events(g):
    """One row per charge event for one vehicle: soc0/soc1, ΔSoC, Ah, capacity, start/end voltage, is_full."""
    g = g.sort_values("t")
    dt = g["t"].diff().dt.total_seconds().fillna(0.0)
    sd = g["soc"].diff()
    med = g.loc[sd > 0, "current"].median()
    sign = np.sign(med) if (np.isfinite(med) and med != 0) else 1.0
    chg = (np.sign(g["current"]) == sign) & (g["current"].abs() > CHG_MIN)
    start = chg & ~(chg.shift(1, fill_value=False) & (dt <= GAP_S))
    g2 = g.assign(ev=start.cumsum())[chg]
    rows = []
    for _, s in g2.groupby("ev"):
        if len(s) < MIN_ROWS:
            continue
        dsoc = float(s["soc"].iloc[-1] - s["soc"].iloc[0])
        if dsoc <= 1.0:
            continue
        secs = (s["t"] - s["t"].iloc[0]).dt.total_seconds().to_numpy()
        ah = abs(_trapz(s["current"].abs().to_numpy(), secs / 3600.0))
        cap = ah / (dsoc / 100.0)
        vv = s.loc[s["voltage"].between(V_LO, V_HI), "voltage"]
        v0 = float(vv.iloc[:5].median()) if len(vv) >= 3 else np.nan
        v1 = float(vv.iloc[-5:].median()) if len(vv) >= 3 else np.nan
        rows.append(dict(t0=s["t"].iloc[0], soc0=float(s["soc"].iloc[0]), soc1=float(s["soc"].iloc[-1]),
                         dsoc=dsoc, ah=ah, cap=cap, v0=v0, v1=v1, n=int(len(s))))
    e = pd.DataFrame(rows)
    if len(e):
        e = e[e["cap"].between(CAP_LO, CAP_HI)].reset_index(drop=True)
        e["is_full"] = (e["dsoc"] >= FULL_DSOC) & (e["soc1"] >= FULL_END)   # FULL = deep by SoC (variant-agnostic)
    return e


def main():
    t0 = time.time()
    fs = glob.glob(DENSE)
    cols = ["vin", "eventAt", "batterySoc", "batteryVoltage", "batteryCurrent"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    d["vin"] = d["vin"].astype(str)
    d = d.rename(columns={"batterySoc": "soc", "batteryVoltage": "voltage", "batteryCurrent": "current"})
    for c in ("soc", "voltage", "current"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(pd.to_numeric(d["eventAt"], errors="coerce"), unit="ms")
    d = d.dropna(subset=["t", "soc", "current"])
    d = d[(d["soc"].between(SOC_LO, SOC_HI)) & (d["current"].abs() <= CUR_MAX)]
    print(f"loaded {len(fs)} files, {len(d):,} clean rows, {d['vin'].nunique()} vehicles ({time.time()-t0:.0f}s)", flush=True)

    fe = pd.read_parquet(FEAT); fe["vin"] = fe["vin"].astype(str); fe["month"] = pd.to_datetime(fe["month"])
    reg = {}
    for vin, g in fe.groupby("vin"):
        g = g.sort_values("month")
        reg[vin] = g["month"].iloc[0] - pd.DateOffset(months=int(round(float(g["age_months"].iloc[0]))))
    # OLD-method (BMS-capacity) SoH fade + raw capacity noise, per vehicle, for the comparison
    bms = {}
    for vin, g in fe.groupby("vin"):
        so = pd.to_numeric(g.sort_values("month")["soh"], errors="coerce").dropna()
        rc = pd.to_numeric(g["capacity_ah"], errors="coerce") if "capacity_ah" in g.columns else pd.Series(dtype=float)
        bms[vin] = (float(so.iloc[0] - so.iloc[-1]) if len(so) > 1 else np.nan, _cv(rc))

    ev_all, soh_all, summ = [], [], []
    for i, (vin, g) in enumerate(d.groupby("vin"), 1):
        if len(g) < 200:
            continue
        e = charge_events(g)
        if e.empty:
            continue
        r = reg.get(vin)
        e["vin"] = vin
        e["age_months"] = ((e["t0"] - r).dt.days / 30.4) if r is not None else np.nan
        full = e[e["is_full"]].copy()
        base = full[full["age_months"].between(BASE_AGE_LO, BASE_AGE_HI)]["cap"]
        cap0 = float(base.median()) if len(base) >= 2 else (
            float(full.sort_values("age_months")["cap"].head(5).median()) if len(full) else np.nan)
        if np.isfinite(cap0) and cap0 > 0 and len(full):
            full["cap0"] = cap0
            full["soh_full"] = np.clip(100.0 * full["cap"] / cap0, None, 100.0)
            soh_all.append(full[["vin", "t0", "age_months", "cap", "cap0", "soh_full", "dsoc", "soc0", "soc1"]])
        ev_all.append(e[["vin", "t0", "age_months", "soc0", "soc1", "dsoc", "ah", "cap", "v0", "v1", "is_full"]])
        bd, bcv = bms.get(vin, (np.nan, np.nan))
        summ.append(dict(vin=vin, n_events=int(len(e)), n_full=int(len(full)), cap0=cap0,
                         cv_full=_cv(full["cap"]), cv_session=bcv, bms_drop=bd,
                         soh_last=(float(full.sort_values("age_months")["soh_full"].iloc[-1])
                                   if (len(full) and np.isfinite(cap0)) else np.nan),
                         age_last=float(e["age_months"].max()) if e["age_months"].notna().any() else np.nan))
        if i % 40 == 0:
            print(f"  [{i}] {time.time()-t0:.0f}s", flush=True)

    EV = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
    SOH = pd.concat(soh_all, ignore_index=True) if soh_all else pd.DataFrame()
    SUM = pd.DataFrame(summ)
    EV.to_parquet(OUT_EV, index=False); SOH.to_parquet(OUT_SOH, index=False); SUM.to_parquet(OUT_SUM, index=False)
    good = SUM[SUM["n_full"] >= 3]
    report = dict(oem="euler", complete=True, vehicles_with_events=int(len(SUM)),
                  vehicles_ge3_full=int(len(good)), total_charge_events=int(len(EV)),
                  total_full_charges=int(EV["is_full"].sum()) if len(EV) else 0,
                  median_full_per_vehicle=float(SUM["n_full"].median()) if len(SUM) else 0.0,
                  cv_bms_capacity_median=round(float(SUM["cv_session"].median()), 1) if SUM["cv_session"].notna().any() else None,
                  cv_full_median=round(float(good["cv_full"].median()), 1) if len(good) else None,
                  full_dsoc=FULL_DSOC, full_end=FULL_END, elapsed_s=round(time.time() - t0, 0))
    json.dump(report, open(OUT_REP, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_EV}, {OUT_SOH}, {OUT_SUM}, {OUT_REP}")


if __name__ == "__main__":
    main()
