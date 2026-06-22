"""Vectorized electrical feature engineering (cuDF/GPU-capable) — no per-group Python apply.

Same backend pattern as soh.py: the heavy groupby/cumsum/shift math is vectorized, so it runs
on a cuDF (GPU) frame or pandas (CPU). Produces per-(vin, month) electrical/cycling features.
"""
import numpy as np
import pandas as pd
from soh import GAP_S

MIN_DOD = 3.0   # min per-session SoC drop (%) counted as a discharge event for DoD


def electrical_features(df, backend="auto"):
    """df columns: vin, t(datetime64), soc, current, batteryVoltage.
    Returns (pandas DataFrame [vin, month, <features>], used_gpu)."""
    use_gpu = False
    if backend in ("auto", "gpu"):
        try:
            import cudf
            df = df if str(type(df)).startswith("<class 'cudf") else cudf.from_pandas(df)
            use_gpu = True
        except Exception:
            if backend == "gpu":
                raise

    df = df.sort_values(["vin", "t"]).reset_index(drop=True)
    dt_ns = df.groupby("vin")["t"].diff()
    dt = dt_ns.dt.total_seconds() if hasattr(dt_ns.dt, "total_seconds") else dt_ns.astype("int64") / 1e9
    brk = dt.isna() | (dt > GAP_S) | (dt <= 0)
    df = df.assign(dt=dt.fillna(0.0), sid=brk.astype("int32").cumsum())
    cur_prev = df.groupby("sid")["current"].shift()
    # helper columns so every reduction is a single-func groupby (avoids MultiIndex on cuDF)
    df = df.assign(
        dQ=((df["current"] + cur_prev) / 2.0 * df["dt"] / 3600.0).abs().fillna(0.0),
        absc=df["current"].abs(),
        cur_neg=df["current"].where(df["current"] < 0),
        cur_pos=df["current"].where(df["current"] > 0),
        socH=(df["soc"] > 90).astype("float32"),
        socL=(df["soc"] < 20).astype("float32"),
        volt_mn=df["batteryVoltage"], volt_mx=df["batteryVoltage"],
        month=df["t"].dt.year * 100 + df["t"].dt.month,
        cnt=1.0,
    )
    g = df.groupby(["vin", "month"])
    feat = g.agg({"dQ": "sum", "absc": "mean", "cur_neg": "mean", "cur_pos": "mean",
                  "soc": "mean", "socH": "mean", "socL": "mean", "batteryVoltage": "mean",
                  "volt_mn": "min", "volt_mx": "max", "cnt": "sum"})
    p95 = g["absc"].quantile(0.95).rename("cur_abs_p95")
    feat = feat.join(p95)
    # --- Depth of Discharge: mean per-session SoC drop over discharge events ---
    fr = df.groupby("sid")[["vin", "month", "soc"]].first()
    fr["soc_last"] = df.groupby("sid")["soc"].last()
    fr["dod"] = fr["soc"] - fr["soc_last"]                  # positive = discharge depth
    fr = fr[fr["dod"] >= MIN_DOD]
    dod = fr.groupby(["vin", "month"])["dod"].mean().reset_index()
    dod.columns = ["vin", "month", "dod_mean"]
    if use_gpu:
        feat = feat.to_pandas(); dod = dod.to_pandas()
    feat = feat.reset_index().rename(columns={
        "dQ": "ah_throughput", "absc": "cur_abs_mean", "cur_neg": "cur_dis_mean",
        "cur_pos": "cur_chg_mean", "soc": "soc_mean", "socH": "frac_soc_high",
        "socL": "frac_soc_low", "batteryVoltage": "volt_mean", "volt_mn": "volt_min",
        "volt_mx": "volt_max", "cnt": "n_rows_ic"})
    feat[["cur_dis_mean", "cur_chg_mean"]] = feat[["cur_dis_mean", "cur_chg_mean"]].fillna(0.0)
    feat = feat.merge(dod, on=["vin", "month"], how="left")
    feat["month"] = pd.to_datetime(feat["month"].astype("int64").astype(str), format="%Y%m")
    return feat, use_gpu
