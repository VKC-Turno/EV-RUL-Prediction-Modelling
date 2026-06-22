#!/usr/bin/env python3
"""Compute & cross-validate three SoH methods for a dense Euler vehicle.

  1. coulomb        — ∫I·dt / (ΔSoC/100), via soh.coulomb_capacity_monthly (gold standard).
  2. bms_capacity   — BMS-reported full capacity = batteryRemainingCapacity / (SoC/100) -> SoH.
  3. reported       — BMS-reported batterySoh (coarse), monthly median.

Writes data/euler/soh/<VIN>_methods.{csv,json} and prints an agreement summary.
Usage: euler_soh_methods.py <VIN>
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, "src")
import soh

VIN = sys.argv[1] if len(sys.argv) > 1 else "MD9EMHDL23A217086"
df = pd.read_parquet(f"data/euler/dense/{VIN}.parquet").sort_values("t").reset_index(drop=True)

# registration (for coulomb anchoring)
rdf = pd.read_csv("data/euler/Euler_Regd_Details.csv")
rdf["reg"] = pd.to_datetime(rdf["regd_date"], format="%d/%m/%y", errors="coerce")
REG = {VIN: rdf.set_index("vin")["reg"].get(VIN)}

# ---- 1. Coulomb counting (reuse the vectorized pipeline; it's sign-agnostic: |∫I·dt|/|ΔSoC|) ----
cdf = df.rename(columns={"batteryCurrent": "current", "batterySoc": "soc"})[["vin", "t", "soc", "current"]].dropna()
cap_month, _ = soh.coulomb_capacity_monthly(cdf, backend="cpu")
coulomb = soh.capacity_to_soh(cap_month, reg=REG)[["vin", "month", "soh", "capacity_ah", "age_months"]]
coulomb = coulomb.rename(columns={"soh": "soh_coulomb", "capacity_ah": "cap_coulomb"})

# ---- 2. BMS remaining-capacity -> full capacity -> SoH ----
b = df[(df["batterySoc"] > 20) & (df["batteryRemainingCapacity"] > 0)].copy()
b["full_cap"] = b["batteryRemainingCapacity"] / (b["batterySoc"] / 100.0)
b["month"] = b["t"].dt.to_period("M").dt.to_timestamp()
bms = b.groupby("month")["full_cap"].median().reset_index()
nominal = bms["full_cap"].iloc[:6].quantile(0.90)        # robust early full capacity ~ pack rating
bms["soh_bms"] = (100.0 * bms["full_cap"] / nominal).clip(upper=100)
bms["soh_bms"] = bms["soh_bms"].cummin()                 # monotonic envelope

# ---- 3. Reported BMS SoH ----
r = df[(df["batterySoh"] > 0) & (df["batterySoh"] <= 100)].copy()
r["month"] = r["t"].dt.to_period("M").dt.to_timestamp()
rep = r.groupby("month")["batterySoh"].median().reset_index().rename(columns={"batterySoh": "soh_reported"})
rep["soh_reported"] = rep["soh_reported"].cummin()

# ---- align & compare ----
M = coulomb.merge(bms[["month", "soh_bms", "full_cap"]], on="month", how="outer") \
           .merge(rep, on="month", how="outer").sort_values("month").reset_index(drop=True)
out = Path("data/euler/soh"); out.mkdir(parents=True, exist_ok=True)
M.to_csv(out / f"{VIN}_methods.csv", index=False)

both = M.dropna(subset=["soh_coulomb", "soh_bms"])
agree = {}
if len(both) >= 3:
    agree["coulomb_vs_bms_rmse"] = float(np.sqrt(np.mean((both["soh_coulomb"] - both["soh_bms"]) ** 2)))
    agree["coulomb_vs_bms_corr"] = float(np.corrcoef(both["soh_coulomb"], both["soh_bms"])[0, 1])
b2 = M.dropna(subset=["soh_coulomb", "soh_reported"])
if len(b2) >= 3:
    agree["coulomb_vs_reported_rmse"] = float(np.sqrt(np.mean((b2["soh_coulomb"] - b2["soh_reported"]) ** 2)))

summary = dict(
    vin=VIN, n_months=int(M["month"].nunique()),
    nominal_full_cap_ah=round(float(nominal), 1),
    coulomb_first=round(float(coulomb["soh_coulomb"].iloc[0]), 1) if len(coulomb) else None,
    coulomb_last=round(float(coulomb["soh_coulomb"].iloc[-1]), 1) if len(coulomb) else None,
    bms_first=round(float(bms["soh_bms"].iloc[0]), 1) if len(bms) else None,
    bms_last=round(float(bms["soh_bms"].iloc[-1]), 1) if len(bms) else None,
    reported_first=round(float(rep["soh_reported"].iloc[0]), 1) if len(rep) else None,
    reported_last=round(float(rep["soh_reported"].iloc[-1]), 1) if len(rep) else None,
    agreement=agree,
)
(out / f"{VIN}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nmonthly (head/tail):")
cols = ["month", "soh_coulomb", "soh_bms", "soh_reported", "cap_coulomb", "full_cap"]
print(M[cols].head(6).to_string(index=False))
print("...")
print(M[cols].tail(6).to_string(index=False))
