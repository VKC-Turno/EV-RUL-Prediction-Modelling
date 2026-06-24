#!/usr/bin/env python3
"""A/B the Mahindra SoH envelope: greedy cumulative-min vs robust isotonic (src/soh.py).

Recomputes SoH from the SAME monthly coulomb capacity under both envelopes, then judges each as a
forecasting target via LOVO actual-vs-predicted, scored against an ENVELOPE-AGNOSTIC physical truth
(R = centered rolling-median of the pre-envelope normalized capacity) so neither method is trivially
favoured. Also reports a flatness/staircase metric (the artifact the user flagged). Writes the
isotonic feature table to a sidecar; does NOT overwrite the live table.

Run: .venv/bin/python src/mahindra_soh_ab.py [--write]
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, "src")
import soh as S
import model as M

FT = "data/mahindra/features/feature_table.parquet"
ft = pd.read_parquet(FT)
ft["month"] = pd.to_datetime(ft["month"])

# --- registration dates (true calendar-age anchor) ---
reg_raw = pd.read_csv("Mh_Regd_Date.csv")
vin_col = next((c for c in reg_raw.columns if c.lower() == "vin"), None) or \
          next(c for c in reg_raw.columns if "vin" in c.lower())
date_col = "vehicle_registration_date" if "vehicle_registration_date" in reg_raw.columns else \
           next(c for c in reg_raw.columns if "reg" in c.lower() and "date" in c.lower())
reg_raw["reg"] = pd.to_datetime(reg_raw[date_col], errors="coerce", dayfirst=True)
REG = dict(zip(reg_raw[vin_col], reg_raw["reg"]))
print(f"reg file: vin col '{vin_col}', date col '{date_col}', {reg_raw['reg'].notna().sum()} dated VINs")

cap_month = ft[["vin", "month", "capacity_ah"]].copy()
new = S.capacity_to_soh(cap_month, reg=REG, method="isotonic")     # has soh + soh_raw + age_months
old = S.capacity_to_soh(cap_month, reg=REG, method="greedy")

# --- sanity: my greedy recompute should reproduce the live table's soh; age must match ---
a = ft[["vin", "month", "age_months", "soh"]].merge(
    new[["vin", "month", "age_months"]], on=["vin", "month"], suffixes=("_ft", "_iso"))
print(f"age_months max|diff| recompute-vs-live: {(a['age_months_ft'] - a['age_months_iso']).abs().max():.3f}")
b = ft[["vin", "month", "soh"]].merge(old[["vin", "month", "soh"]], on=["vin", "month"], suffixes=("_ft", "_grd"))
print(f"greedy recompute vs live soh  max|diff|: {(b['soh_ft'] - b['soh_grd']).abs().max():.3f}  "
      f"(near 0 => live table = current soh.py greedy)")

# --- build per-method modelling frames sharing all stress cols; only soh differs ---
stress_cols = [c for c in ft.columns if c not in ("soh",)]
base = ft[stress_cols].copy()
m_iso = base.merge(new[["vin", "month", "soh"]], on=["vin", "month"])
m_grd = base.merge(old[["vin", "month", "soh"]], on=["vin", "month"])

# --- envelope-agnostic ground truth R per vin (denoised physical SoH, no monotone bias) ---
R_by_vin = {}
for v, g in new.sort_values(["vin", "month"]).groupby("vin"):
    R_by_vin[v] = g["soh_raw"].rolling(5, center=True, min_periods=1).median().to_numpy()


def lovo(m, label):
    tr = M.build_transitions(m)
    rows = []
    for v in m["vin"].unique():
        g = m[m["vin"] == v].sort_values("month").reset_index(drop=True)
        if len(g) < 3:
            continue
        t = tr[tr["vin"] != v]
        mod = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500,
                                learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                verbose=-1).fit(t[M.FEATS].to_numpy(), t["loss"].to_numpy(),
                                                sample_weight=t["w"].to_numpy())
        pred = M.free_run_observed(g, mod)
        R = R_by_vin[v][:len(g)]
        mae = float(np.mean(np.abs(pred - R)))
        drop = float(R[0] - R[-1])
        rows.append((v, mae, drop))
    d = pd.DataFrame(rows, columns=["vin", "mae", "drop"])
    deg = d[d["drop"] >= 3.0]; flat = d[d["drop"] < 3.0]
    print(f"\n[{label}]  n={len(d)}  overall MAE {d['mae'].mean():.3f} | "
          f"degraders(n={len(deg)}) {deg['mae'].mean():.3f} | flat(n={len(flat)}) {flat['mae'].mean():.3f}")
    return d


def flatness(frame, label):
    """Fraction of consecutive months with ~zero SoH change (the staircase/freeze artifact)."""
    fr = []
    for v, g in frame.sort_values(["vin", "month"]).groupby("vin"):
        s = g["soh"].to_numpy()
        if len(s) < 3:
            continue
        d = np.abs(np.diff(s))
        fr.append((d < 0.05).mean())
    fr = np.array(fr)
    print(f"[{label}] flat-month fraction: mean {fr.mean():.2f}  median {np.median(fr):.2f}  "
          f"vehicles >80% flat: {(fr > 0.8).sum()}/{len(fr)}")


print("\n=== flatness (staircase artifact) ===")
flatness(m_grd, "greedy ")
flatness(m_iso, "isotonic")

print("\n=== LOVO actual-vs-predicted, scored vs envelope-agnostic physical truth ===")
dg = lovo(m_grd, "greedy ")
di = lovo(m_iso, "isotonic")
print(f"\nVERDICT  overall isotonic {di['mae'].mean():.3f} vs greedy {dg['mae'].mean():.3f}  "
      f"(delta {di['mae'].mean() - dg['mae'].mean():+.3f}; negative = isotonic better)")

if "--write" in sys.argv:
    out = base.merge(new[["vin", "month", "soh", "soh_raw"]], on=["vin", "month"])
    out = out[ft.columns.tolist() + (["soh_raw"] if "soh_raw" not in ft.columns else [])]
    Path(FT).rename(FT.replace(".parquet", "_greedy_backup.parquet"))
    out.to_parquet(FT, index=False)
    print(f"\nWROTE isotonic SoH -> {FT} (greedy backed up to *_greedy_backup.parquet)")
else:
    sidecar = FT.replace(".parquet", "_isotonic.parquet")
    base.merge(new[["vin", "month", "soh", "soh_raw"]], on=["vin", "month"]).to_parquet(sidecar, index=False)
    print(f"\nwrote sidecar (not live): {sidecar}  — rerun with --write to adopt")
