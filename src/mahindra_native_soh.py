#!/usr/bin/env python3
"""Native-only Mahindra SoH via the distance-per-SoC RANGE PROXY (Tier D), on the whole-fleet MONTHLY sample.

The native feed carries no current/voltage/reported-SoH, so the only SoH-like signal is km-per-%SoC while
driving (odometer-up & soc-down segments) — a range/capacity proxy. This builds it per (vin, month) from
data/mahindra/native100/ and NORMALISES each vehicle to its own first-6-month baseline.

Deliberately NO monotone envelope: forcing monotonicity on a noisy proxy would ratchet noise dips into fake
"degraders" (the iso-floor artifact). We show the RAW normalised proxy so the flat/noisy reality is honest.
This is SEPARATE from the intellicar coulomb SoH — it exists to SHOW what native monthly data yields.

-> data/mahindra/native_monthly_soh.parquet  [vin, month, age_months, soh, cap_range]
Run: .venv/bin/python src/mahindra_native_soh.py
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

fs = sorted(glob.glob("data/mahindra/native100/*.parquet"))
cols = ["vin", "eventAt", "soc", "odometer"]
df = pd.concat([pd.read_parquet(f)[[c for c in cols if c in pd.read_parquet(f, columns=None).columns]]
                for f in fs], ignore_index=True)
df["t"] = pd.to_datetime(pd.to_numeric(df["eventAt"], errors="coerce"), unit="ms")
for c in ["soc", "odometer"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["vin"] = df["vin"].astype(str)
df = df[df.soc.between(0, 100) & df.odometer.between(0, 300000)].dropna(subset=["t"]).sort_values(["vin", "t"])

# driving/discharge segments = consecutive rows with odometer UP and soc DOWN (no label needed)
df["d_odo"] = df.groupby("vin")["odometer"].diff()
df["d_soc"] = -df.groupby("vin")["soc"].diff()
df["dtm"] = df.groupby("vin")["t"].diff().dt.total_seconds() / 60
seg = df[df.d_odo.between(0.1, 80) & df.d_soc.between(0.5, 40) & df.dtm.between(0.1, 180)].copy()
seg["month"] = seg["t"].dt.to_period("M").dt.to_timestamp()
vm = seg.groupby(["vin", "month"]).agg(odo=("d_odo", "sum"), soc=("d_soc", "sum"), n=("d_odo", "size")).reset_index()
vm = vm[vm.n >= 3].copy()
vm["cap_range"] = 100 * vm.odo / vm.soc                 # km per full SoC = range/capacity proxy
vm = vm[vm.cap_range.between(20, 400)].sort_values(["vin", "month"])

parts = []
for vin, g in vm.groupby("vin"):
    g = g.sort_values("month").copy()
    base = g["cap_range"].head(6).median()              # baseline = this vehicle's first ~6 months' range
    if not np.isfinite(base) or base <= 0:
        continue
    g["age_months"] = (g["month"] - g["month"].iloc[0]).dt.days / 30.4
    g["soh"] = (100.0 * g["cap_range"] / base).clip(40, 110)   # raw normalised proxy (NOT monotone)
    parts.append(g[["vin", "month", "age_months", "soh", "cap_range"]])
sohdf = pd.concat(parts, ignore_index=True)
keep = sohdf.groupby("vin").size()
sohdf = sohdf[sohdf.vin.isin(keep[keep >= 4].index)].reset_index(drop=True)

out = "data/mahindra/native_monthly_soh.parquet"
Path(out).parent.mkdir(parents=True, exist_ok=True)
sohdf.to_parquet(out, index=False)
d = sohdf.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
print(f"wrote {out}: {sohdf.vin.nunique()} vins, {len(sohdf):,} vin-months, median {int(sohdf.groupby('vin').size().median())} mo")
print(f"net drop>=2pp: {(d >= 2).sum()} | net rise (proxy went UP): {(d <= -2).sum()} | flat: {(d.abs() < 2).sum()}")
