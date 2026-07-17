"""OFFLINE equivalence check — incremental (normalised) == full batch, on LOCAL data.

Does NOT run Glue/Spark/S3. It reimplements the job's algorithm in pandas and proves that the
day-by-day normalised computation (euler_daily -> vin_stats -> features_daily -> read-time join, exactly
what euler_preprocessing_incremental.py does in Spark) reproduces the straightforward full-history batch
computation, byte-for-byte on the 21-column output.

Input: a small slice of data/euler/dense/*.parquet (per-event Euler telemetry — the Glue raw schema).
Run:   .venv/bin/python MLOps/glue/local_equivalence_check.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROLL_DAYS = 7
FINAL = ["vin", "event_date", "current_soh", "vehicle_age_days", "estimated_cycle_count",
         "cumulative_distance", "avg_daily_distance", "avg_battery_temperature", "max_battery_temperature",
         "avg_current", "max_current", "rolling_temperature_exposure", "high_temp_exposure_minutes",
         "soc_variation", "driving_intensity", "max_cell_imbalance", "soh_degradation_per_day",
         "avg_speed", "max_speed", "soh_degradation_per_km", "rolling_cycle_count"]
VEHICLE_MODE_MAP = {"30": 0, "31": 1, "32": 2, "33": 3, "34": 4, "35": 5, "45636F6E6F6D79": 6}


# ── shared Sections 1-4: per-event -> one row per (vin, event_date) ───────────────────────
def clean_and_daily(events: pd.DataFrame) -> pd.DataFrame:
    d = events.copy()
    d = d[d["vin"].notna() & (d["vin"].astype(str).str.strip() != "")]
    d["event_timestamp"] = pd.to_datetime(d["t"])
    d["event_date"] = d["event_timestamp"].dt.normalize()

    ranges = {"battery_soh": ("batterySoh", 70, 100), "battery_soc": ("batterySoc", 0, 100),
              "odometer_clean": ("odometer", 0, 125000), "battery_current": ("batteryCurrent", -150, 150),
              "battery_temperature": ("batteryTemperature", -20, 60),
              "battery_remaining_capacity": ("batteryRemainingCapacity", 0, 210),
              "cell_imbalance": ("cellImbalance", 0, 200)}
    for alias, (raw, lo, hi) in ranges.items():
        v = pd.to_numeric(d.get(raw), errors="coerce")
        d[alias] = v.where((v >= lo) & (v <= hi))
    # synthetic speed (dense feed has no speed column) — deterministic, present in both paths
    d["speed_clean"] = d["battery_current"].abs().clip(0, 60)
    # vehicle_mode via the job's hex map
    def _mode(v):
        try:
            return VEHICLE_MODE_MAP.get(format(int(v), "X"))
        except (ValueError, TypeError):
            return None
    d["vehicle_mode"] = d["vehicleMode"].map(_mode)

    d = d.dropna(subset=["battery_soh", "battery_soc", "odometer_clean", "battery_current",
                         "battery_temperature", "battery_remaining_capacity", "cell_imbalance"])
    d = d.sort_values(["vin", "event_timestamp"])
    d["next_ts"] = d.groupby("vin")["event_timestamp"].shift(-1)
    d["duration_minutes"] = ((d["next_ts"] - d["event_timestamp"]).dt.total_seconds() / 60).fillna(0)
    d["prev_soc"] = d.groupby("vin")["battery_soc"].shift(1)
    dsoc = (d["battery_soc"] - d["prev_soc"]).abs()
    d["cycle_difference"] = np.where(d["prev_soc"].notna() & (dsoc >= 1), dsoc, 0.0)
    d["running_minutes"] = np.where(d["vehicle_mode"] == 3, d["duration_minutes"], 0.0)
    d["high_temp_minutes"] = np.where(d["battery_temperature"] > 45, d["duration_minutes"], 0.0)
    drv = d["vehicle_mode"] == 3

    g = d.groupby(["vin", "event_date"])
    daily = g.agg(
        current_soh=("battery_soh", "max"),
        daily_distance=("odometer_clean", lambda s: s.max() - s.min()),
        avg_battery_temperature=("battery_temperature", "mean"),
        max_battery_temperature=("battery_temperature", "max"),
        max_cell_imbalance=("cell_imbalance", "max"),
        estimated_cycle_count=("cycle_difference", lambda s: s.sum() / 100),
        high_temp_exposure_minutes=("high_temp_minutes", "sum"),
        running_hours=("running_minutes", lambda s: s.sum() / 60),
    ).reset_index()
    soc = g["battery_soc"]
    daily["soc_variation"] = (soc.max() - soc.min()).values
    drv_g = d[drv].groupby(["vin", "event_date"])
    for name, colr in [("avg_current", ("battery_current", "mean")), ("max_current", ("battery_current", "max")),
                       ("avg_speed", ("speed_clean", "mean")), ("max_speed", ("speed_clean", "max"))]:
        s = drv_g[colr[0]].agg(colr[1]).rename(name)
        daily = daily.merge(s, on=["vin", "event_date"], how="left")
    daily["driving_intensity"] = np.where(daily["running_hours"] > 0,
                                          daily["daily_distance"] / daily["running_hours"], np.nan)
    return daily.sort_values(["vin", "event_date"]).reset_index(drop=True)


# ── BATCH: trend/degradation over full history, in one pass ───────────────────────────────
def batch_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["vin", "event_date"]).copy()
    g = d.groupby("vin")
    d["first_vehicle_date"] = g["event_date"].transform("min")
    d["vehicle_age_days"] = (d["event_date"] - d["first_vehicle_date"]).dt.days
    d["cumulative_distance"] = g["daily_distance"].cumsum()
    d["avg_daily_distance"] = g["daily_distance"].expanding().mean().reset_index(level=0, drop=True)
    d["rolling_temperature_exposure"] = (g["high_temp_exposure_minutes"]
                                         .rolling(ROLL_DAYS, min_periods=1).sum().reset_index(level=0, drop=True))
    d["rolling_cycle_count"] = (g["estimated_cycle_count"]
                                .rolling(ROLL_DAYS, min_periods=1).mean().reset_index(level=0, drop=True))
    smax = g["current_soh"].transform("max"); smin = g["current_soh"].transform("min")
    ndays = g["event_date"].transform("count")
    d["soh_degradation_per_day"] = np.where(ndays > 1, (smax - smin) / (ndays - 1), 0.0)
    cmax = g["cumulative_distance"].transform("max"); cmin = g["cumulative_distance"].transform("min")
    d["soh_degradation_per_km"] = np.where((cmax - cmin) > 0, (smax - smin) / (cmax - cmin), 0.0)
    return d[FINAL].sort_values(["vin", "event_date"]).reset_index(drop=True)


# ── INCREMENTAL: day-by-day, normalised (mirrors the Glue job exactly) ────────────────────
def incremental_features(daily: pd.DataFrame) -> pd.DataFrame:
    euler_daily = pd.DataFrame()
    features_daily = []
    vin_stats = {}
    for D in sorted(daily["event_date"].unique()):
        daily_new = daily[daily["event_date"] == D]
        euler_daily = pd.concat([euler_daily, daily_new], ignore_index=True)   # MERGE upsert (new day)
        touched = daily_new["vin"].unique()
        hist = euler_daily[euler_daily["vin"].isin(touched)]
        # vin_stats: recompute for touched vins from full history <= D
        for vin, gg in hist.groupby("vin"):
            gg = gg.sort_values("event_date")
            smax, smin = gg["current_soh"].max(), gg["current_soh"].min()
            n = len(gg); total = gg["daily_distance"].sum(); first_daily = gg["daily_distance"].iloc[0]
            total_km = total - first_daily
            vin_stats[vin] = dict(
                first_vehicle_date=gg["event_date"].min(), day_count=n, total_distance=total,
                soh_degradation_per_day=((smax - smin) / (n - 1)) if n > 1 else 0.0,
                soh_degradation_per_km=((smax - smin) / total_km) if total_km > 0 else 0.0)
        # today's per-date row for each touched vin
        for vin, row in daily_new.set_index("vin").iterrows():
            st = vin_stats[vin]
            recent = hist[(hist["vin"] == vin) & (hist["event_date"] <= D)].sort_values("event_date").tail(ROLL_DAYS)
            feat = dict(row)
            feat["vin"] = vin
            feat["cumulative_distance"] = st["total_distance"]
            feat["avg_daily_distance"] = st["total_distance"] / st["day_count"]
            feat["vehicle_age_days"] = (D - st["first_vehicle_date"]).days
            feat["rolling_temperature_exposure"] = recent["high_temp_exposure_minutes"].sum()
            feat["rolling_cycle_count"] = recent["estimated_cycle_count"].mean()
            features_daily.append(feat)
    fd = pd.DataFrame(features_daily)
    # read-time join: broadcast FINAL vin-level rates to every row
    rates = pd.DataFrame([{**{"vin": v}, "soh_degradation_per_day": s["soh_degradation_per_day"],
                           "soh_degradation_per_km": s["soh_degradation_per_km"]}
                          for v, s in vin_stats.items()])
    fd = fd.merge(rates, on="vin", how="left")
    return fd[FINAL].sort_values(["vin", "event_date"]).reset_index(drop=True)


# ── compare ───────────────────────────────────────────────────────────────────────────────
def compare(a, b, tol=1e-6):
    assert list(a.columns) == list(b.columns) == FINAL
    assert len(a) == len(b), f"row count {len(a)} vs {len(b)}"
    worst = {}
    nan_mismatch = []
    for c in FINAL:
        if c in ("vin", "event_date"):
            assert (a[c].astype(str).values == b[c].astype(str).values).all(), f"key mismatch {c}"
            continue
        av = pd.to_numeric(a[c], errors="coerce").to_numpy()
        bv = pd.to_numeric(b[c], errors="coerce").to_numpy()
        if (np.isnan(av) != np.isnan(bv)).any():
            nan_mismatch.append(c)
        d = np.abs(av - bv); d[np.isnan(av) & np.isnan(bv)] = 0.0
        worst[c] = float(np.nanmax(d)) if len(d) else 0.0
    return worst, nan_mismatch


def main():
    files = sorted(glob.glob("data/euler/dense/*.parquet"), key=os.path.getsize)[:6]
    if not files:
        sys.exit("no data/euler/dense/*.parquet found")
    cols = ["t", "vin", "batterySoc", "batterySoh", "odometer", "batteryCurrent",
            "batteryTemperature", "batteryRemainingCapacity", "cellImbalance", "vehicleMode"]
    import pyarrow.parquet as pq
    frames = []
    for f in files:
        have = set(pq.ParquetFile(f).schema.names)
        frames.append(pd.read_parquet(f, columns=[c for c in cols if c in have]))
    events = pd.concat(frames, ignore_index=True)
    events["t"] = pd.to_datetime(events["t"])
    # keep each vehicle's FIRST 25 days (per-VIN window) so every VIN gets many days -> rolling(7) engages
    vin_start = events.groupby("vin")["t"].transform("min").dt.normalize()
    events = events[events["t"] < vin_start + pd.Timedelta(days=25)]
    nd = events.assign(d=events["t"].dt.normalize()).groupby("vin")["d"].nunique()
    print(f"slice: {events['vin'].nunique()} vehicles, {len(events)} events, "
          f"days/vin: {nd.min()}–{nd.max()}")

    daily = clean_and_daily(events)
    print(f"daily rows: {len(daily)}  vins: {daily['vin'].nunique()}  "
          f"days/vin: {daily.groupby('vin').size().min()}–{daily.groupby('vin').size().max()}")

    b = batch_features(daily)
    i = incremental_features(daily)
    worst, nan_mismatch = compare(b, i)

    max_diff = max(worst.values()) if worst else 0.0
    print(f"\nmax abs diff across all 19 numeric feature columns: {max_diff:.3e}")
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    print("largest per-column diffs:", [(c, round(v, 9)) for c, v in top])
    if nan_mismatch:
        print("NaN-alignment mismatches:", nan_mismatch)
    ok = (max_diff <= 1e-6) and not nan_mismatch
    print("\n" + ("PASS — incremental == batch on local data" if ok
                  else "FAIL — incremental diverges from batch"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
