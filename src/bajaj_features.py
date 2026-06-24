#!/usr/bin/env python3
"""Build the Bajaj condition-aware feature table from the dense cohort parquets.

Target SoH = the BMS-**reported** SoH `essBmsSohcEstPercValue` (per-month median, smoothed, monotone
non-increasing). This is the only feasible target for Bajaj: the feed carries NO current, NO voltage,
and NO remaining-capacity, so coulomb counting and Euler's BMS-capacity method are both impossible.
The reported value is clean and well-behaved (one quantized value/month, declines smoothly with cycling
and odometer — see validation in onboarding notes), so unlike Euler's coarse batterySoh it needs only
light cleaning. We do NOT renormalize to 100% at t0: reported SoH is a BMS absolute, and aged vehicles
legitimately enter the feed already below 100%.

Features from charge-cycle count / SoC / pack & ambient temperature / drive-efficiency / odometer.
Garbage/sentinel rows are clipped via per-signal physical bounds.

-> data/bajaj/features/feature_table.parquet  (one row per vin-month)
"""
import os, glob
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

# physical bounds for the Bajaj-native value columns (clip out sentinels / garbage)
BOUNDS = {
    "essBmsSocEstPercValue": (0, 100),
    "essBmsSohcEstPercValue": (40, 100),          # reported SoH; <40 is implausible/garbage
    "essBmsChgcycleActCountValue": (0, 20000),
    "essBmsTemperatureActDegcValue": (-20, 80),
    "etsVcuAmbienttempActDegcValue": (-20, 60),
    "etsVcuDriveeffEstWhpkmValue": (0, 1000),
}


def load_clean(fp):
    df = pd.read_parquet(fp)
    df["t"] = pd.to_datetime(df["t"]) if "t" in df.columns else \
        pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c, (lo, hi) in BOUNDS.items():
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.where((s >= lo) & (s <= hi))
    # odometer: metres -> km; clip nonsense (0 / negative / absurd)
    odo_km = pd.to_numeric(df.get("hmiIclOdoActMValue"), errors="coerce") / 1000.0
    df["odo_km"] = odo_km.where((odo_km > 0) & (odo_km < 1_000_000))
    df["month"] = df["t"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("t").reset_index(drop=True)


def reported_soh_monthly(df):
    """Per-month median reported SoH, smoothed then forced monotone non-increasing.

    Smooth (3-month rolling median) BEFORE the cummin envelope so one noisy month can't carve a
    permanent fake cliff (per PROCEDURE.md rule 4). Require >=15 obs/month and >=4 months."""
    d = df[df["essBmsSohcEstPercValue"].notna()]
    mon = d.groupby("month").agg(soh_raw=("essBmsSohcEstPercValue", "median"),
                                 n_soh=("essBmsSohcEstPercValue", "size")).reset_index()
    mon = mon[mon["n_soh"] >= 15].sort_values("month")
    if len(mon) < 4:
        return None
    sm = mon["soh_raw"].rolling(3, min_periods=1, center=True).median()
    mon["soh"] = np.minimum.accumulate(sm.to_numpy())        # monotone non-increasing
    return mon[["month", "soh"]]


def monthly_features(df):
    g = df.groupby("month")
    feat = g.agg(
        soc_mean=("essBmsSocEstPercValue", "mean"),
        cyc_max=("essBmsChgcycleActCountValue", "max"),
        temp_mean=("essBmsTemperatureActDegcValue", "mean"),
        temp_max=("essBmsTemperatureActDegcValue", "max"),
        amb_temp_mean=("etsVcuAmbienttempActDegcValue", "mean"),
        driveeff_mean=("etsVcuDriveeffEstWhpkmValue", "mean"),
        odo_max=("odo_km", "max"),
        n_rows=("essBmsSocEstPercValue", "size"),
    )
    # SoC dwell (calendar-aging proxies)
    feat["frac_soc_high"] = df.assign(h=(df["essBmsSocEstPercValue"] > 90).astype(float)) \
        .groupby("month")["h"].mean()
    feat["frac_soc_low"] = df.assign(l=(df["essBmsSocEstPercValue"] < 20).astype(float)) \
        .groupby("month")["l"].mean()
    feat["temp_p95"] = g["essBmsTemperatureActDegcValue"].quantile(0.95)
    return feat.reset_index()


def main():
    # registration dates -> TRUE calendar age (months since registration), like Euler/Mahindra.
    # The Bajaj feed only spans ~2025-09+, but vehicles registered earlier; without this, age would be
    # measured from first telemetry (understating real age). regd_date is ISO (YYYY-MM-DD).
    REG = {}
    for rp in ("Bajaj_Regd_Details.csv", "data/bajaj/Bajaj_Regd_Details.csv"):
        if Path(rp).exists():
            r = pd.read_csv(rp); r["reg"] = pd.to_datetime(r["regd_date"], errors="coerce")
            REG = dict(zip(r["vin"], r["reg"]))
            print(f"  using registration dates from {rp} ({r['reg'].notna().sum()} dated VINs)")
            break
    if not REG:
        print("  WARNING: no Bajaj_Regd_Details.csv found — age falls back to first-telemetry month")
    parts = []
    for fp in sorted(glob.glob("data/bajaj/dense/*.parquet")):
        vin = Path(fp).stem
        df = load_clean(fp)
        soh = reported_soh_monthly(df)
        if soh is None:
            print(f"  skip {vin}: insufficient reported-SoH months")
            continue
        feat = monthly_features(df)
        m = soh.merge(feat, on="month", how="inner").sort_values("month")
        # calendar age from registration where available (reg must predate first telemetry), else first month
        r = REG.get(vin)
        base = r if (r is not None and pd.notna(r) and r <= m["month"].iloc[0]) else m["month"].iloc[0]
        m["age_months"] = ((m["month"] - base).dt.days / 30.4).round(1)
        # monthly km from odometer (charge-cycle delta is a parallel usage signal)
        m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
        m["cyc_month"] = m["cyc_max"].diff().clip(lower=0).fillna(0.0)
        m["cum_km"] = m["km_month"].cumsum()
        m["cum_cycles"] = m["cyc_month"].cumsum()
        m["vin"] = vin
        parts.append(m)
        print(f"  {vin}: {len(m)} months, SoH {m['soh'].iloc[0]:.0f}->{m['soh'].iloc[-1]:.0f}, "
              f"cycles {m['cyc_max'].iloc[0]:.0f}->{m['cyc_max'].iloc[-1]:.0f}, "
              f"odo {m['odo_max'].iloc[0]:.0f}->{m['odo_max'].iloc[-1]:.0f} km")
    if not parts:
        print("NO vehicles yielded usable reported SoH — aborting (nothing written).")
        return
    out = pd.concat(parts, ignore_index=True)
    Path("data/bajaj/features").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/bajaj/features/feature_table.parquet", index=False)
    print(f"\nwrote data/bajaj/features/feature_table.parquet: {len(out)} rows, "
          f"{out['vin'].nunique()} vehicles, {out.shape[1]} cols")
    print("cols:", list(out.columns))


if __name__ == "__main__":
    main()
