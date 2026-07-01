#!/usr/bin/env python3
"""Build the Piaggio SoH target + monthly feature table — COULOMB (Tier A), via the intellicar feed.

Piaggio (Ape E-Xtra FX, VIN prefix MBX) is a Mahindra-twin: the shared **intellicar** feed carries signed
pack **current + voltage + SoC** at sub-minute cadence, so SoH is coulomb-counted with src/soh.py (the same
ΔSoC-weighted-pooled method + robust-isotonic envelope as Mahindra). The native `parquet/piaggio/` feed has
no voltage / no reported SoH, so it only supplements a distance-to-empty feature (and is a Tier-D distance
cross-check, not used for the target here).

Output columns match the other OEMs' featengg store tables, so the shared rate model (src/model.py) and both
dashboards consume Piaggio unchanged.

  ->  data/piaggio/features/feature_table.parquet   (local; the dashboard's load_cohort reads the store copy)
  ->  data/redshift/piaggio_featengg.parquet        (store copy, ymd as date objects like the other OEMs)
Run: .venv/bin/python src/piaggio_features.py   (after import_cohort.py intellicar piaggio + piaggio piaggio)

Caveats (documented for the playbook): (1) no per-vin registration file yet -> first-telemetry anchoring
(used_reg=False), same as Mahindra; aged vehicles that entered telemetry already-worn are anchored ~100% at
first sight, so early SoH is optimistic. (2) No pack temperature in either feed -> temp_* are a MOTOR-temp
proxy, flagged. Refine both when a reg source / pack-temp signal appears.
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")
import soh as soh_mod
from features import electrical_features

IC_DIR, NATIVE_DIR = "data/piaggio/intellicar", "data/piaggio/feed"
OUT = "data/piaggio/features/feature_table.parquet"
STORE = "data/redshift/piaggio_featengg.parquet"

# Physical bounds — clip sentinels BEFORE any math (playbook A10: SoC=79903, current=-22220 etc. pass notnull).
BOUNDS = dict(soc=(0.0, 100.0), current=(-400.0, 400.0), batteryVoltage=(30.0, 90.0),
              odometer=(0.0, 300000.0), chargeCycle=(0.0, 20000.0), motorTemperature=(-20.0, 130.0))

# Final schema (order matches data/redshift/mahindra_featengg.parquet exactly).
SCHEMA = ["vin", "ymd", "capacity_ah", "n_sessions", "tot_dsoc", "age_months", "used_reg", "soh_raw", "soh",
          "ah_throughput", "cur_abs_mean", "cur_dis_mean", "cur_chg_mean", "soc_mean", "frac_soc_high",
          "frac_soc_low", "volt_mean", "volt_min", "volt_max", "n_rows_ic", "cur_abs_p95", "dod_mean",
          "temp_mean", "temp_max", "dte_mean", "odo_max", "km_month", "cum_ah", "cum_km",
          "inv_sqrt_age", "soh_deficit"]


def _load(dir_, numeric):
    files = sorted(glob.glob(f"{dir_}/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet in {dir_} — run import_cohort.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev = "eventAt" if "eventAt" in df.columns else "eventat"
    df["t"] = pd.to_datetime(pd.to_numeric(df[ev], errors="coerce"), unit="ms")
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, (lo, hi) in BOUNDS.items():            # sentinel clip -> out-of-band becomes NaN
        if c in df.columns:
            df[c] = df[c].where(df[c].between(lo, hi))
    return df.dropna(subset=["t", "vin"])


print("loading intellicar (coulomb source)…", flush=True)
ic = _load(IC_DIR, ["soc", "current", "batteryVoltage", "odometer", "chargeCycle", "motorTemperature"])
ic = ic[ic["vin"].astype(str).str.startswith("MBX")].dropna(subset=["soc", "current"])
print(f"  {len(ic):,} rows, {ic['vin'].nunique()} vins", flush=True)

# 1. Coulomb SoH (vin, month, capacity_ah, n_sessions, tot_dsoc, age_months, used_reg, soh_raw, soh)
capm, _ = soh_mod.coulomb_capacity_monthly(ic[["vin", "t", "soc", "current"]])
sohdf = soh_mod.capacity_to_soh(capm, reg={}, method="isotonic")     # first-telemetry anchor (no reg file yet)
print(f"  coulomb SoH: {sohdf['vin'].nunique()} vins, {len(sohdf)} vin-months", flush=True)

# 2. Electrical STRESS features (ah_throughput, cur_*, soc_*, volt_*, dod_mean, cur_abs_p95, n_rows_ic)
elec, _ = electrical_features(ic[["vin", "t", "soc", "current", "batteryVoltage"]])

# 3. Monthly usage / thermal from intellicar (odometer, cycles, motor-temp proxy)
ic["month"] = ic["t"].dt.to_period("M").dt.to_timestamp()
um = (ic.groupby(["vin", "month"])
        .agg(odo_max=("odometer", "max"), temp_mean=("motorTemperature", "mean"),
             temp_max=("motorTemperature", "max")).reset_index())

# 4. Native distanceTillEmpty -> dte_mean (optional supplement; feed may lag)
try:
    nat = _load(NATIVE_DIR, ["distanceTillEmpty"])
    nat["month"] = nat["t"].dt.to_period("M").dt.to_timestamp()
    dte = (nat.groupby(["vin", "month"])["distanceTillEmpty"].mean().reset_index()
             .rename(columns={"distanceTillEmpty": "dte_mean"}))
except SystemExit:
    dte = pd.DataFrame(columns=["vin", "month", "dte_mean"])

# 5. Merge on (vin, month) + cumulative / curvature features
m = (sohdf.merge(elec, on=["vin", "month"], how="left")
          .merge(um, on=["vin", "month"], how="left")
          .merge(dte, on=["vin", "month"], how="left")
          .sort_values(["vin", "month"]).reset_index(drop=True))
m["ah_throughput"] = m["ah_throughput"].fillna(0.0)
m["cum_ah"] = m.groupby("vin")["ah_throughput"].cumsum()
m["km_month"] = m.groupby("vin")["odo_max"].diff().clip(lower=0).fillna(0.0)
m["cum_km"] = m.groupby("vin")["km_month"].cumsum()
m["inv_sqrt_age"] = 1.0 / np.sqrt(np.maximum(m["age_months"], 0.0) + 1.0)
m["soh_deficit"] = 100.0 - m["soh"]
m["ymd"] = m["month"].dt.date                                         # date objects, like the other OEMs
for c in SCHEMA:
    if c not in m.columns:
        m[c] = np.nan
m = m[SCHEMA]

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(STORE).parent.mkdir(parents=True, exist_ok=True)
m.to_parquet(OUT, index=False)
m.to_parquet(STORE, index=False)
print(f"\nwrote {OUT} and {STORE}: {m['vin'].nunique()} vins, {len(m):,} vin-months")
deg = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
print(f"SoH: start median {m.groupby('vin')['soh'].first().median():.1f} | "
      f"end median {m.groupby('vin')['soh'].last().median():.1f} | "
      f"degraders (drop>=2pp): {(deg >= 2).sum()} / {m['vin'].nunique()}")
print(f"median months/vin: {int(m.groupby('vin').size().median())}")
