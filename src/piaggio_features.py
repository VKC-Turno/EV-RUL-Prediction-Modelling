#!/usr/bin/env python3
"""Build the Piaggio SoH target + monthly feature table — COULOMB (Tier A), from the intellicar feed.

Piaggio (Ape E-Xtra FX, VIN prefix MBX) is a Mahindra-twin: the shared intellicar feed carries signed pack
current + voltage + SoC at sub-minute cadence, so SoH is coulomb-counted with src/soh.py (same ΔSoC-weighted
pooled method + robust-isotonic envelope as Mahindra). Native supplements distance-to-empty only.

SCALE: the extraction is ~287M rows across ~308k tiny files, so we read in threaded BATCHES and shrink `vin`
to a category to fit the working set in RAM; then run the (already per-(vin,month)) coulomb/feature
aggregation over the whole frame at once — chunking by ingest-partition would be WRONG (events scatter across
partitions). Output schema matches the other OEMs' featengg so src/model.py + dashboards consume it unchanged.

Caveats: no per-vin reg file yet -> first-telemetry anchoring (used_reg=False), like Mahindra; temp_* are a
MOTOR-temp proxy (no pack temp in the feed).
  -> data/piaggio/features/feature_table.parquet  and  data/redshift/piaggio_featengg.parquet
Run: .venv/bin/python src/piaggio_features.py   (after import_cohort.py intellicar piaggio + piaggio piaggio)
"""
import os, sys, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")
import soh as soh_mod
from features import electrical_features

IC_DIR, NATIVE_DIR = "data/piaggio/intellicar", "data/piaggio/feed"
OUT = "data/piaggio/features/feature_table.parquet"
STORE = "data/redshift/piaggio_featengg.parquet"
BOUNDS = dict(soc=(0., 100.), current=(-400., 400.), batteryVoltage=(30., 90.),
              odometer=(0., 300000.), chargeCycle=(0., 20000.), motorTemperature=(-20., 130.))
SCHEMA = ["vin", "ymd", "capacity_ah", "n_sessions", "tot_dsoc", "age_months", "used_reg", "soh_raw", "soh",
          "ah_throughput", "cur_abs_mean", "cur_dis_mean", "cur_chg_mean", "soc_mean", "frac_soc_high",
          "frac_soc_low", "volt_mean", "volt_min", "volt_max", "n_rows_ic", "cur_abs_p95", "dod_mean",
          "temp_mean", "temp_max", "dte_mean", "odo_max", "km_month", "cum_ah", "cum_km",
          "inv_sqrt_age", "soh_deficit"]


def _read_all(directory, cols, batch=20000, workers=32):
    """Threaded batched read of a directory of tiny parquets -> one frame with `vin` as category (compact)."""
    files = glob.glob(f"{directory}/*.parquet")
    if not files:
        return pd.DataFrame(columns=cols)
    def rd(f):
        try:
            return pd.read_parquet(f, columns=cols)
        except Exception:
            return None
    batches = []
    for s in range(0, len(files), batch):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = [d for d in pool.map(rd, files[s:s + batch]) if d is not None and len(d)]
        if parts:
            b = pd.concat(parts, ignore_index=True); b["vin"] = b["vin"].astype("category")
            for c in b.columns:                        # halve numeric memory for the 287M-row working set
                if b[c].dtype == "float64":
                    b[c] = b[c].astype("float32")
            batches.append(b)
        print(f"    read {min(s + batch, len(files)):,}/{len(files):,} files", flush=True)
    df = pd.concat(batches, ignore_index=True)
    del batches
    return df


def _prep(df):
    df["t"] = pd.to_datetime(pd.to_numeric(df["eventAt"], errors="coerce"), unit="ms")
    for c, (lo, hi) in BOUNDS.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.where(v.between(lo, hi))
    df["vin"] = df["vin"].astype(str)
    return df[df["vin"].str.startswith("MBX")].dropna(subset=["t"])


print("loading intellicar (coulomb source) — batched…", flush=True)
ic = _prep(_read_all(IC_DIR, ["vin", "eventAt", "soc", "current", "batteryVoltage", "odometer",
                              "chargeCycle", "motorTemperature"]))
ic = ic.dropna(subset=["soc", "current"])
print(f"  intellicar: {len(ic):,} rows, {ic['vin'].nunique()} vins", flush=True)

# backend="cpu": 287M rows won't fit on the GPU (cuDF auto-backend OOMs); 50 GB RAM handles it on CPU.
cap_month, _ = soh_mod.coulomb_capacity_monthly(ic[["vin", "t", "soc", "current"]], backend="cpu")
sohdf = soh_mod.capacity_to_soh(cap_month, reg={}, method="isotonic")
print(f"  coulomb SoH: {sohdf['vin'].nunique()} vins, {len(sohdf):,} vin-months", flush=True)
elec, _ = electrical_features(ic[["vin", "t", "soc", "current", "batteryVoltage"]], backend="cpu")
ic["month"] = ic["t"].dt.to_period("M").dt.to_timestamp()
um = (ic.groupby(["vin", "month"], observed=True)
        .agg(odo_max=("odometer", "max"), temp_mean=("motorTemperature", "mean"),
             temp_max=("motorTemperature", "max")).reset_index())
del ic

print("loading native (dte supplement) — batched…", flush=True)
nat = _read_all(NATIVE_DIR, ["vin", "eventat", "distanceTillEmpty"])
if len(nat):
    nat["t"] = pd.to_datetime(pd.to_numeric(nat["eventat"], errors="coerce"), unit="ms")
    nat["vin"] = nat["vin"].astype(str); nat["month"] = nat["t"].dt.to_period("M").dt.to_timestamp()
    nat["distanceTillEmpty"] = pd.to_numeric(nat["distanceTillEmpty"], errors="coerce")
    dte = (nat.dropna(subset=["t"]).groupby(["vin", "month"])["distanceTillEmpty"].mean()
              .reset_index().rename(columns={"distanceTillEmpty": "dte_mean"}))
else:
    dte = pd.DataFrame(columns=["vin", "month", "dte_mean"])
del nat

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
m["ymd"] = m["month"].dt.date
for c in SCHEMA:
    if c not in m.columns:
        m[c] = np.nan
store = m[SCHEMA]                                                         # store schema uses `ymd` (like the others)
local = m[["vin", "month"] + [c for c in SCHEMA if c not in ("vin", "ymd")]]  # local feature-table schema uses `month`

Path(OUT).parent.mkdir(parents=True, exist_ok=True); Path(STORE).parent.mkdir(parents=True, exist_ok=True)
local.to_parquet(OUT, index=False); store.to_parquet(STORE, index=False)
m = store
deg = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
print(f"\nwrote {OUT} + {STORE}: {m['vin'].nunique()} vins, {len(m):,} vin-months")
print(f"SoH start median {m.groupby('vin')['soh'].first().median():.1f} | end median {m.groupby('vin')['soh'].last().median():.1f} "
      f"| degraders(>=2pp) {(deg >= 2).sum()}/{m['vin'].nunique()} | median months {int(m.groupby('vin').size().median())}")
