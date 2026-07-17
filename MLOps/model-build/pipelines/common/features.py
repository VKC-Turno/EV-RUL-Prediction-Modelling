"""Monthly stress features + assembly into the `<oem>_featengg` schema.

One row per (vin, month). Mirrors src/features.py (electrical_features) + the assembly in src/*_features.py.
Missing-per-OEM columns stay NaN (the tree models are NaN-tolerant) — never rename or drop columns, the
whole downstream (forecasters + dashboards) reads this schema positionally-by-name.
"""
import numpy as np
import pandas as pd

# the featengg contract — keep identical across OEMs (research repo README §5)
SCHEMA = [
    "vin", "ymd", "capacity_ah", "n_sessions", "tot_dsoc", "age_months", "used_reg", "soh_raw", "soh",
    "ah_throughput", "cur_abs_mean", "cur_dis_mean", "cur_chg_mean", "soc_mean", "frac_soc_high",
    "frac_soc_low", "volt_mean", "volt_min", "volt_max", "n_rows_ic", "cur_abs_p95", "dod_mean",
    "temp_mean", "temp_max", "dte_mean", "odo_max", "km_month", "cum_ah", "cum_km", "inv_sqrt_age",
    "soh_deficit",
]


def _num(sub: pd.DataFrame, *names) -> pd.Series:
    """First present column among `names` as a numeric Series; an all-NaN Series if none present."""
    for n in names:
        if n in sub.columns:
            return pd.to_numeric(sub[n], errors="coerce")
    return pd.Series(np.nan, index=sub.index, dtype="float64")


def electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (vin, month): current/SoC/voltage aggregates. Robust to absent channels (-> NaN columns)."""
    d = df.copy()
    if "month" not in d.columns:
        d["month"] = pd.to_datetime(d["t"]).values.astype("datetime64[M]")
    if "t" in d.columns:
        d["dt_h"] = d.groupby("vin")["t"].diff().dt.total_seconds().div(3600.0).clip(0, 1.0)
    else:
        d["dt_h"] = np.nan

    def agg(sub):
        cur = _num(sub, "current")
        soc = _num(sub, "soc")
        volt = _num(sub, "batteryVoltage", "batteryPackVoltage")
        temp = _num(sub, "temperature", "batteryTemperature")
        dt_h = _num(sub, "dt_h")
        return pd.Series({
            "n_rows_ic": len(sub),
            "ah_throughput": float(np.nansum((cur.abs() * dt_h).to_numpy())),
            "cur_abs_mean": cur.abs().mean(), "cur_abs_p95": cur.abs().quantile(0.95),
            "cur_chg_mean": cur[cur > 0].mean(), "cur_dis_mean": cur[cur < 0].abs().mean(),
            "soc_mean": soc.mean(),
            "frac_soc_high": float((soc >= 80).mean()), "frac_soc_low": float((soc <= 20).mean()),
            "dod_mean": float(soc.max() - soc.min()) if soc.notna().any() else np.nan,
            "volt_mean": volt.mean(), "volt_min": volt.min(), "volt_max": volt.max(),
            "temp_mean": temp.mean(), "temp_max": temp.max(),
        })

    return d.groupby(["vin", "month"], group_keys=True).apply(agg, include_groups=False).reset_index()


def assemble(soh_df: pd.DataFrame, feat_df: pd.DataFrame, usage_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join SoH + electrical features + usage (odometer/km) -> the featengg schema."""
    m = soh_df.merge(feat_df, on=["vin", "month"], how="left")
    if usage_df is not None:
        m = m.merge(usage_df, on=["vin", "month"], how="left")
    m = m.sort_values(["vin", "month"])
    # derived
    m["km_month"] = m.groupby("vin")["odo_max"].diff().clip(lower=0) if "odo_max" in m else np.nan
    m["cum_ah"] = m.groupby("vin")["ah_throughput"].cumsum()
    m["cum_km"] = m.groupby("vin")["km_month"].cumsum() if "km_month" in m else np.nan
    m["inv_sqrt_age"] = 1.0 / np.sqrt(m["age_months"].astype(float) + 1.0)
    m["soh_deficit"] = 100.0 - m["soh"]
    m["ymd"] = pd.to_datetime(m["month"]).dt.strftime("%Y-%m-%d")
    m["capacity_ah"] = m.get("cap_full", m.get("cap_ah", np.nan))
    # ensure every schema column exists
    for c in SCHEMA:
        if c not in m.columns:
            m[c] = np.nan
    return m[SCHEMA].reset_index(drop=True)
