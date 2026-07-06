#!/usr/bin/env python3
"""Euler SoH from BMS full-capacity — the real fix (coulomb was NOISIER; see euler_full_charge_soh.py negative test).

Euler's BMS reports batteryRemainingCapacity directly, and at near-full SoC that reading (≈5.7% raw CV) is
cleaner than anything we can coulomb-count. Production SoH already uses it, but pushes it through a heavy
isotonic-envelope + 100-clip that collapses it into flat / cliff / stuck-floor artifacts (30% flat, 24% clipped).

This exposes the SAME signal raw and monthly, so the dashboard can apply the light monotone-decreasing fit +
outlier-greying (the treatment we built for Mahindra) instead of the flattening envelope.

full_cap = batteryRemainingCapacity / (SoC/100) at SoC 95-100%, monthly median per vehicle. One fast pass over
the 231 dense files.  Output: data/euler/bms_soh.parquet (+ _summary.parquet, _report.json).
Run: .venv/bin/python src/euler_bms_soh.py
"""
import os, json, glob
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
DENSE = "data/euler/dense/*.parquet"
FEAT = "data/euler/features/feature_table.parquet"
OUT, OUT_SUM, OUT_REP = "data/euler/bms_soh.parquet", "data/euler/bms_soh_summary.parquet", "data/euler/bms_soh_report.json"
SOC_LO, SOC_HI = 95.0, 100.0
RC_LO, RC_HI = 1.0, 500.0
MIN_MON_N = 20                        # a month needs >= this many near-full readings
BASE_LO, BASE_HI = 0.5, 12.0          # baseline cap0 window (months)


def _cv(s):
    s = pd.Series(s).dropna()
    return float(s.std() / s.mean() * 100.0) if (len(s) > 1 and s.mean()) else np.nan


def main():
    fs = glob.glob(DENSE)
    d = pd.concat([pd.read_parquet(f, columns=["vin", "eventAt", "batterySoc", "batteryRemainingCapacity"])
                   for f in fs], ignore_index=True)
    d["vin"] = d["vin"].astype(str)
    for c in ("batterySoc", "batteryRemainingCapacity"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["batterySoc"].between(SOC_LO, SOC_HI)) & (d["batteryRemainingCapacity"].between(RC_LO, RC_HI))].copy()
    d["full_cap"] = d["batteryRemainingCapacity"] / (d["batterySoc"] / 100.0)
    d["t"] = pd.to_datetime(pd.to_numeric(d["eventAt"], errors="coerce"), unit="ms")
    d["month"] = d["t"].dt.to_period("M").dt.to_timestamp()
    mon = d.groupby(["vin", "month"]).agg(full_cap=("full_cap", "median"), n=("full_cap", "size"),
                                          raw_cv=("full_cap", _cv)).reset_index()
    mon = mon[mon["n"] >= MIN_MON_N]

    fe = pd.read_parquet(FEAT); fe["vin"] = fe["vin"].astype(str); fe["month"] = pd.to_datetime(fe["month"])
    reg = {}
    for vin, g in fe.groupby("vin"):
        g = g.sort_values("month")
        reg[vin] = g["month"].iloc[0] - pd.DateOffset(months=int(round(float(g["age_months"].iloc[0]))))
    mon["age_months"] = [((m - reg[v]).days / 30.4 if v in reg else np.nan) for v, m in zip(mon["vin"], mon["month"])]
    mon = mon.merge(fe[["vin", "month", "soh"]].rename(columns={"soh": "soh_prod"}), on=["vin", "month"], how="left")

    parts, summ = [], []
    for vin, g in mon.groupby("vin"):
        g = g.sort_values("month")
        base = g[g["age_months"].between(BASE_LO, BASE_HI)]["full_cap"]
        cap0 = float(base.median()) if len(base) >= 2 else float(g["full_cap"].head(3).median())
        if not (np.isfinite(cap0) and cap0 > 0):
            continue
        g = g.assign(cap0=cap0, soh_full=np.clip(100.0 * g["full_cap"] / cap0, None, 102.0))
        parts.append(g)
        sp = pd.to_numeric(fe[fe["vin"] == vin].sort_values("month")["soh"], errors="coerce").dropna()
        summ.append(dict(vin=vin, n_months=int(len(g)), cap0=cap0, cv_full=_cv(g["full_cap"]),
                         raw_cv_median=float(g["raw_cv"].median()),
                         prod_drop=float(sp.iloc[0] - sp.iloc[-1]) if len(sp) > 1 else np.nan))
    M = pd.concat(parts, ignore_index=True); S = pd.DataFrame(summ)
    M.to_parquet(OUT, index=False); S.to_parquet(OUT_SUM, index=False)
    rep = dict(oem="euler", vehicles=int(S["vin"].nunique()), vehicles_ge4_months=int((S["n_months"] >= 4).sum()),
               median_months=float(S["n_months"].median()),
               cv_monthly_median=round(float(S["cv_full"].median()), 1),
               cv_raw_reading_median=round(float(S["raw_cv_median"].median()), 1))
    json.dump(rep, open(OUT_REP, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {OUT}, {OUT_SUM}, {OUT_REP}")


if __name__ == "__main__":
    main()
