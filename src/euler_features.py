#!/usr/bin/env python3
"""Build the Euler condition-aware feature table from the dense cohort parquets.

Target SoH = validated BMS remaining-capacity method (high-SoC band, isotonic monotone fit — see the
cross-validation in dashboard/crossval_workflow.js). Features from current / SoC / temperature /
voltage / cell-imbalance / odometer (richer than Mahindra: real temp, voltage, imbalance).
Garbage rows are sanitized first (the data carries SoC=79903, current=-22220 sentinels).

-> data/euler/features/feature_table.parquet  (one row per vin-month)
"""
import os, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

os.chdir(Path(__file__).resolve().parent.parent)
GAP_S = 300.0
NOMINAL = 133.0          # rated pack Ah (for C-rate); SoH baseline is per-vehicle early full_cap

BOUNDS = {"batteryCurrent": (-200, 200), "batterySoc": (0, 100), "batterySoh": (0, 100),
          "batteryRemainingCapacity": (0, 200), "batteryVoltage": (40, 120), "batteryTemperature": (-20, 80)}


def load_clean(fp):
    df = pd.read_parquet(fp)
    df["t"] = pd.to_datetime(df["t"]) if "t" in df.columns else pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c, (lo, hi) in BOUNDS.items():
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.where((s >= lo) & (s <= hi))
    df["odometer"] = pd.to_numeric(df.get("odometer"), errors="coerce")
    if "cellImbalance" in df:
        df["cellImbalance"] = pd.to_numeric(df["cellImbalance"], errors="coerce").clip(0, 5000)
    df["month"] = df["t"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("t").reset_index(drop=True)


def bms_soh_monthly(df):
    """Validated BMS-capacity SoH: high-SoC full_cap, isotonic decreasing, /per-vehicle nominal."""
    d = df[(df["batterySoc"].between(95, 100)) & (df["batteryRemainingCapacity"].between(0, 500))].copy()
    d["full_cap"] = d["batteryRemainingCapacity"] / (d["batterySoc"] / 100.0)
    med = d["full_cap"].median()
    if not np.isfinite(med) or med < 40:                 # broken / zero remaining-capacity for this vehicle
        return None
    d = d[d["full_cap"].between(0.6 * med, 1.4 * med)]    # adaptive window around THIS vehicle's pack (Hiload ≠ ~133Ah cohort)
    mon = d.groupby("month").agg(full_cap=("full_cap", "median"), n=("full_cap", "size")).reset_index()
    mon = mon[mon["n"] >= 15]
    if len(mon) < 6:
        return None
    nominal = float(mon["full_cap"].iloc[:6].quantile(0.90))
    fit = IsotonicRegression(increasing=False).fit_transform(np.arange(len(mon)), mon["full_cap"].to_numpy())
    mon["soh"] = np.clip(100.0 * fit / nominal, None, 100.0)
    return mon[["month", "soh"]]


def monthly_features(df):
    gap = df["t"].diff().dt.total_seconds()
    brk = gap.isna() | (gap > GAP_S) | (gap <= 0)
    df = df.assign(sid=brk.cumsum().astype("int64"), dt=gap.where(~brk, 0).fillna(0))
    cur_prev = df.groupby("sid")["batteryCurrent"].shift()
    df = df.assign(dQ=((df["batteryCurrent"] + cur_prev) / 2 * df["dt"] / 3600).abs().fillna(0),
                   absc=df["batteryCurrent"].abs())
    g = df.groupby("month")
    feat = g.agg(ah_throughput=("dQ", "sum"), cur_abs_mean=("absc", "mean"),
                 soc_mean=("batterySoc", "mean"), volt_mean=("batteryVoltage", "mean"),
                 volt_min=("batteryVoltage", "min"), volt_max=("batteryVoltage", "max"),
                 temp_mean=("batteryTemperature", "mean"), temp_max=("batteryTemperature", "max"),
                 odo_max=("odometer", "max"), n_rows=("absc", "size"))
    feat["cur_abs_p95"] = g["absc"].quantile(0.95)
    feat["cur_chg_mean"] = df[df["batteryCurrent"] > 0].groupby("month")["batteryCurrent"].mean()
    feat["cur_dis_mean"] = df[df["batteryCurrent"] < 0].groupby("month")["batteryCurrent"].mean()
    feat["frac_soc_high"] = df.assign(h=(df["batterySoc"] > 90).astype(float)).groupby("month")["h"].mean()
    feat["frac_soc_low"] = df.assign(l=(df["batterySoc"] < 20).astype(float)).groupby("month")["l"].mean()
    if "cellImbalance" in df:
        feat["imbalance_mean"] = g["cellImbalance"].mean()
    # depth-of-discharge: mean SoC drop over discharge sessions
    fr = df.groupby("sid").agg(month=("month", "first"), s0=("batterySoc", "first"), s1=("batterySoc", "last"))
    fr["dod"] = fr["s0"] - fr["s1"]
    feat["dod_mean"] = fr[fr["dod"] >= 3].groupby("month")["dod"].mean()
    feat["crate_p95"] = feat["cur_abs_p95"] / NOMINAL
    return feat.reset_index()


def main():
    reg = pd.read_csv("data/euler/Euler_Regd_Details.csv")
    reg["reg"] = pd.to_datetime(reg["regd_date"], format="%d/%m/%y", errors="coerce")
    REG = dict(zip(reg["vin"], reg["reg"]))
    parts = []
    for fp in sorted(glob.glob("data/euler/dense/*.parquet")):
        vin = Path(fp).stem
        df = load_clean(fp)
        soh = bms_soh_monthly(df)
        if soh is None:
            print(f"  skip {vin}: insufficient high-SoC capacity data"); continue
        feat = monthly_features(df)
        m = soh.merge(feat, on="month", how="inner").sort_values("month")
        r = REG.get(vin)
        base = r if (r is not None and pd.notna(r) and r <= m["month"].iloc[0]) else m["month"].iloc[0]
        m["age_months"] = ((m["month"] - base).dt.days / 30.4).round(1)
        m["cur_chg_mean"] = m["cur_chg_mean"].fillna(0.0)
        m["cur_dis_mean"] = m["cur_dis_mean"].fillna(0.0)
        m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
        m["cum_ah"] = m["ah_throughput"].cumsum()
        m["cum_km"] = m["km_month"].cumsum()
        m["vin"] = vin
        parts.append(m)
        print(f"  {vin}: {len(m)} months, SoH {m['soh'].iloc[0]:.1f}->{m['soh'].iloc[-1]:.1f}, "
              f"imbalance {'yes' if 'imbalance_mean' in m else 'no'}")
    out = pd.concat(parts, ignore_index=True)
    Path("data/euler/features").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/euler/features/feature_table.parquet", index=False)
    print(f"\nwrote data/euler/features/feature_table.parquet: {len(out)} rows, "
          f"{out['vin'].nunique()} vehicles, {out.shape[1]} cols")
    print("cols:", list(out.columns))


if __name__ == "__main__":
    main()
