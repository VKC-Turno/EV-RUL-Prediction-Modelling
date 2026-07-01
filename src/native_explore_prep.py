#!/usr/bin/env python3
"""Precompute lightweight summaries for the Mahindra native-feed EXPLORATION dashboard
(dashboard/native_explorer.py), so the app never loads the 23M raw rows at runtime.

Reads data/mahindra/native_monthly/*.parquet (the whole-fleet monthly sample) ->
  - data/mahindra/native_vehicle_summary.parquet   (one row per vehicle)
  - data/mahindra/native_vehicle_monthly.parquet    (one row per vehicle-month, for time-series drill-down)
Run: .venv/bin/python src/native_explore_prep.py
"""
import glob
import numpy as np, pandas as pd

fs = sorted(glob.glob("data/mahindra/native_monthly/*.parquet"))
print(f"reading {len(fs)} monthly files…", flush=True)


def load(f):
    d = pd.read_parquet(f)
    keep = ["vin", "eventAt", "soc", "odometer", "distanceToEmpty"]
    d["vehicleStatus"] = d["vehicleStatus"] if "vehicleStatus" in d.columns else np.nan
    return d[keep + ["vehicleStatus"]]


df = pd.concat([load(f) for f in fs], ignore_index=True)
df["t"] = pd.to_datetime(pd.to_numeric(df["eventAt"], errors="coerce"), unit="ms")
for c in ["soc", "odometer", "distanceToEmpty"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["vin"] = df["vin"].astype(str)
df = df[df.soc.between(0, 100) & df.odometer.between(0, 300000)].dropna(subset=["t"])
df["month"] = df["t"].dt.to_period("M").dt.to_timestamp()
df["driving"] = (df["vehicleStatus"] == "DRIVING").astype(float)
df["charging"] = (df["vehicleStatus"] == "CHARGING").astype(float)
print(f"clean rows: {len(df):,} | vins: {df.vin.nunique():,}", flush=True)

g = df.groupby("vin")
summ = pd.DataFrame({
    "n_rows": g.size(), "n_months": g["month"].nunique(),
    "first": g["t"].min(), "last": g["t"].max(),
    "odo_max": g["odometer"].max(), "km_window": g["odometer"].max() - g["odometer"].min(),
    "soc_min": g["soc"].min(), "soc_max": g["soc"].max(), "soc_mean": g["soc"].mean(),
    "dte_med": g["distanceToEmpty"].median(), "frac_driving": g["driving"].mean(),
    "has_status": g["driving"].count() > 0,
}).reset_index()
summ["span_mo"] = ((summ["last"] - summ["first"]).dt.days / 30.4).round(1)
summ.to_parquet("data/mahindra/native_vehicle_summary.parquet", index=False)

gm = df.groupby(["vin", "month"])
monthly = pd.DataFrame({
    "soc_mean": gm["soc"].mean(), "soc_min": gm["soc"].min(), "soc_max": gm["soc"].max(),
    "odo_max": gm["odometer"].max(), "dte_med": gm["distanceToEmpty"].median(),
    "n": gm.size(), "frac_driving": gm["driving"].mean(), "frac_charging": gm["charging"].mean(),
}).reset_index()
monthly.to_parquet("data/mahindra/native_vehicle_monthly.parquet", index=False)
print(f"wrote summary ({len(summ):,} vins) + monthly ({len(monthly):,} vin-months)")
