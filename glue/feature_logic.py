"""Per-VIN feature extraction, shared by the Glue Spark job (one vehicle at a time).

Spark hands each VIN's raw telemetry rows to `vin_features(pdf, oem, reg_map)` as a pandas DataFrame
(via applyInPandas); this returns that vehicle's per-(vin, month) feature rows — the SAME feature_table
schema produced locally by src/<oem>_features.py / the Mahindra coulomb pipeline. It reuses the repo's
df-level functions so Glue and local stay byte-identical.

Source feed per OEM (see src/config.py):
  bajaj    -> its own native feed   (reported SoH; no current / no voltage)
  euler    -> its own native feed   (BMS remaining-capacity SoH; current/voltage on 2023+)
  mahindra -> the INTELLICAR feed   (only source with `current`, needed for coulomb SoH). Thermal / GPS /
              distance-to-empty features additionally come from Mahindra's NATIVE feed and are merged on
              (vin, month) in the notebook — see the TODO in `mahindra()` for the dual-source join.
"""
import numpy as np
import pandas as pd

# repo src modules — packaged via --extra-py-files src.zip (they chdir on import; harmless in Glue)
import bajaj_features as bf
import euler_features as ef
import soh as soh_mod
import features as feat_mod


def _age_from_reg(month_series, reg):
    """(age_months since registration, reg_known). Falls back to first observed month if reg unknown/late."""
    first = month_series.iloc[0]
    known = reg is not None and pd.notna(reg) and reg <= first
    base = reg if known else first
    return ((month_series - base).dt.days / 30.4).round(1), known


def bajaj(pdf, reg):
    df = bf.load_clean(pdf)                       # eventAt->t, bounds-clip, odo(m)->km, month
    soh = bf.reported_soh_monthly(df)
    if soh is None:
        return None
    m = soh.merge(bf.monthly_features(df), on="month", how="inner").sort_values("month")
    m["age_months"], m["reg_known"] = _age_from_reg(m["month"], reg)
    m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
    m["cyc_month"] = m["cyc_max"].diff().clip(lower=0).fillna(0.0)
    m["cum_km"] = m["km_month"].cumsum()
    m["cum_cycles"] = m["cyc_month"].cumsum()
    return m


def euler(pdf, reg):
    df = ef.load_clean(pdf)
    soh = ef.bms_soh_monthly(df)                  # high-SoC capacity -> isotonic SoH; None if pack broken
    if soh is None:
        return None
    m = soh.merge(ef.monthly_features(df), on="month", how="inner").sort_values("month")
    m["age_months"], m["reg_known"] = _age_from_reg(m["month"], reg)
    m["cur_chg_mean"] = m["cur_chg_mean"].fillna(0.0)
    m["cur_dis_mean"] = m["cur_dis_mean"].fillna(0.0)
    m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
    m["cum_ah"] = m["ah_throughput"].cumsum()
    m["cum_km"] = m["km_month"].cumsum()
    return m


def mahindra(pdf, reg):
    # INTELLICAR rows: vin, eventAt(ms) or t, soc, current, batteryVoltage, odometer
    df = pdf.copy()
    df["t"] = pd.to_datetime(df["t"]) if "t" in df.columns else \
        pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c in ("soc", "current", "batteryVoltage", "odometer"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["month"] = df["t"].dt.to_period("M").dt.to_timestamp()
    vin = str(df["vin"].iloc[0])
    cap, _ = soh_mod.coulomb_capacity_monthly(df[["vin", "t", "soc", "current"]])
    if cap.empty:
        return None
    sohdf = soh_mod.capacity_to_soh(cap, reg={vin: reg} if reg is not None else None)  # soh, age_months, capacity_ah
    feat, _ = feat_mod.electrical_features(df[["vin", "t", "soc", "current", "batteryVoltage"]])
    m = sohdf.merge(feat, on=["vin", "month"], how="inner").sort_values("month")
    odo = df.groupby("month")["odometer"].max().rename("odo_max").reset_index()
    m = m.merge(odo, on="month", how="left")
    m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
    m["cum_ah"] = m["ah_throughput"].cumsum()
    m["cum_km"] = m["km_month"].cumsum()
    # TODO (dual-source): temp_mean/temp_max, lat_mean/lon_mean, dte_mean come from Mahindra's NATIVE feed
    #   (battery-oem-data/parquet/mahindra/vehicle-data/) and are merged on (vin, month) in
    #   notebooks/02_features_model/mahindra_degradation_model.ipynb. Add that native read + left-merge here
    #   if those columns are in your trained model's FEATS (src/model.py STRESS). Until then this yields the
    #   coulomb-SoH + electrical/usage subset.
    return m


BUILDERS = {"bajaj": bajaj, "euler": euler, "mahindra": mahindra}


def vin_features(pdf, oem, reg_map):
    """Spark-UDF entry point. pdf = one VIN's rows; reg_map = {vin: registration Timestamp}.
    Returns that VIN's feature rows, or an EMPTY frame (skip) on insufficient data / any error —
    one bad vehicle must never fail the whole fleet job."""
    vin = str(pdf["vin"].iloc[0])
    try:
        m = BUILDERS[oem.lower()](pdf, reg_map.get(vin))
    except Exception:
        return pd.DataFrame()
    if m is None or len(m) == 0:
        return pd.DataFrame()
    m["vin"] = vin
    return m
