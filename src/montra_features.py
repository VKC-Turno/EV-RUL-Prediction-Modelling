#!/usr/bin/env python3
"""Build the Montra SoH target + monthly feature table (10-vehicle POC) from the sampled intellicar-style feed.

Montra (new OEM, VIN prefix P60…) carries soc + current + batteryPackVoltage + resCapacity + PACK temperature at
sub-minute cadence, so SoH is coulomb-counted with src/soh.py — the same ΔSoC-weighted pooled method + robust-
isotonic envelope as Mahindra/Piaggio. Input = data/montra/sample_raw.parquet (from src/montra_sample.py). Output
schema matches the other OEMs' featengg so model.py + oem_train + the dashboards consume it unchanged.
Caveats: 10-vehicle POC over ~4 months (2024-07..10) only — short window, little decline expected; no per-vin reg
file -> first-telemetry anchoring (used_reg=False); Montra has REAL pack temp (better than Piaggio's motor proxy).
  -> data/montra/features/feature_table.parquet  and  data/redshift/montra_featengg.parquet
Run: .venv/bin/python src/montra_features.py   (after src/montra_sample.py)
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")
import soh as soh_mod
from features import electrical_features

RAW = "data/montra/sample_raw.parquet"
OUT = "data/montra/features/feature_table.parquet"
STORE = "data/redshift/montra_featengg.parquet"
BOUNDS = dict(soc=(0., 100.), current=(-400., 400.), batteryVoltage=(20., 120.),
              odometer=(0., 300000.), temperature=(-20., 130.))
SCHEMA = ["vin", "ymd", "capacity_ah", "n_sessions", "tot_dsoc", "age_months", "used_reg", "soh_raw", "soh",
          "ah_throughput", "cur_abs_mean", "cur_dis_mean", "cur_chg_mean", "soc_mean", "frac_soc_high",
          "frac_soc_low", "volt_mean", "volt_min", "volt_max", "n_rows_ic", "cur_abs_p95", "dod_mean",
          "temp_mean", "temp_max", "dte_mean", "odo_max", "km_month", "cum_ah", "cum_km",
          "inv_sqrt_age", "soh_deficit"]

df = pd.read_parquet(RAW).rename(columns={"batteryPackVoltage": "batteryVoltage"})
df["vin"] = df["vin"].astype(str)
if "t" not in df.columns:
    df["t"] = pd.to_datetime(pd.to_numeric(df["eventAt"], errors="coerce"), unit="ms")
for c, (lo, hi) in BOUNDS.items():
    if c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce"); df[c] = v.where(v.between(lo, hi))
ic = df.dropna(subset=["t", "soc", "current"])
print(f"montra: {len(ic):,} rows, {ic['vin'].nunique()} vins", flush=True)

# Montra current is UNSIGNED (no negative = no coulomb charge-event detection), but it carries resCapacity, so
# SoH = BMS remaining-capacity (Euler-style): resCapacity/(SoC/100) at near-full SoC -> full_cap -> cap0-norm ->
# isotonic. First-telemetry age anchor (no reg file). (POC fleet is ~new -> SoH sits near 100.)
from sklearn.isotonic import IsotonicRegression
ic["resCapacity"] = pd.to_numeric(ic.get("resCapacity"), errors="coerce")
nf = ic[(ic["soc"].between(95, 100)) & (ic["resCapacity"].between(1, 500))].copy()
nf["month"] = nf["t"].dt.to_period("M").dt.to_timestamp()
nf["full_cap"] = nf["resCapacity"] / (nf["soc"] / 100.0)
mon = (nf.groupby(["vin", "month"], observed=True)
         .agg(capacity_ah=("full_cap", "median"), n_rows_ic=("full_cap", "size")).reset_index())
mon = mon[mon["n_rows_ic"] >= 5]
parts = []
for vin, g in mon.groupby("vin"):
    g = g.sort_values("month").copy()
    g["age_months"] = ((g["month"] - g["month"].iloc[0]).dt.days / 30.4)     # first-telemetry anchor
    cap0 = float(g["capacity_ah"].head(3).median())
    g["soh_raw"] = np.clip(100.0 * g["capacity_ah"] / cap0, None, 100.0) if cap0 > 0 else 100.0
    iso = IsotonicRegression(increasing=False, y_max=100.0, out_of_bounds="clip").fit(g["age_months"], g["soh_raw"])
    g["soh"] = np.clip(iso.predict(g["age_months"]), None, 100.0)
    g["used_reg"] = False
    parts.append(g)
sohdf = pd.concat(parts, ignore_index=True)
print(f"  BMS-capacity SoH: {sohdf['vin'].nunique()} vins, {len(sohdf):,} vin-months", flush=True)
elec, _ = electrical_features(ic[["vin", "t", "soc", "current", "batteryVoltage"]], backend="cpu")
ic["month"] = ic["t"].dt.to_period("M").dt.to_timestamp()
um = (ic.groupby(["vin", "month"], observed=True)
        .agg(odo_max=("odometer", "max"), temp_mean=("temperature", "mean"),
             temp_max=("temperature", "max")).reset_index())

m = (sohdf.merge(elec, on=["vin", "month"], how="left")
          .merge(um, on=["vin", "month"], how="left")
          .sort_values(["vin", "month"]).reset_index(drop=True))
m["ah_throughput"] = m["ah_throughput"].fillna(0.0)
m["cum_ah"] = m.groupby("vin")["ah_throughput"].cumsum()
m["km_month"] = m.groupby("vin")["odo_max"].diff().clip(lower=0).fillna(0.0)
m["cum_km"] = m.groupby("vin")["km_month"].cumsum()
m["inv_sqrt_age"] = 1.0 / np.sqrt(np.maximum(m["age_months"], 0.0) + 1.0)
m["soh_deficit"] = 100.0 - m["soh"]
m["ymd"] = m["month"].dt.date
for c in SCHEMA:
    if c not in m.columns:
        m[c] = np.nan
store = m[SCHEMA]
local = m[["vin", "month"] + [c for c in SCHEMA if c not in ("vin", "ymd")]]

Path(OUT).parent.mkdir(parents=True, exist_ok=True); Path(STORE).parent.mkdir(parents=True, exist_ok=True)
local.to_parquet(OUT, index=False); store.to_parquet(STORE, index=False)
deg = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
print(f"wrote {OUT} + {STORE}: {store['vin'].nunique()} vins, {len(store):,} vin-months")
print(f"SoH start median {m.groupby('vin')['soh'].first().median():.1f} -> end median {m.groupby('vin')['soh'].last().median():.1f} "
      f"| degraders(>=2pp) {(deg >= 2).sum()}/{m['vin'].nunique()} | median months {int(m.groupby('vin').size().median())}")
print("DONE")
