#!/usr/bin/env python3
"""EXPERIMENT: coulomb SoH from FULL CHARGE EVENTS only.

The production SoH pipeline (src/soh.py) estimates monthly capacity from *every* continuous-logging
session — charge, discharge, or a mix — via capacity = Σ|∫I·dt| / Σ(|ΔSoC|/100). Because most sessions
are short, partial and sometimes direction-reversing, that estimate scatters ~±15% month-to-month, and
the robust envelope then collapses an unresolvable trend into a flat 100% line (the "flat-line artifact").

This experiment restricts the capacity measurement to **full charge events**: a single-direction charge
segment that spans a large SoC range and ends near the top (ΔSoC >= FULL_DSOC AND soc_end >= FULL_END).
Those are the cleanest coulomb windows available — prototype CV drops from ~13% (session) to ~1% (full
charge) on the noisiest vehicles — so a few-percent real fade becomes *resolvable* instead of buried.

Inputs : data/mahindra/intellicar (raw vin,eventAt,soc,current) + data/redshift/mahindra_featengg.parquet
Outputs: data/mahindra/full_charge_events.parquet   (every charge event, is_full flag)
         data/mahindra/full_charge_soh.parquet      (per full-charge SoH point: cap, cap0, soh_full)
         data/mahindra/full_charge_summary.parquet   (per-vehicle: n_full, CV session vs full, soh_last)
         data/mahindra/full_charge_report.json       (fleet noise-reduction headline)
Run    : .venv/bin/python src/full_charge_soh.py   (~15 min; scans 95 vehicles' raw telemetry)
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc

os.chdir(Path(__file__).resolve().parent.parent)

RAW = "data/mahindra/intellicar"
FEATENGG = "data/redshift/mahindra_featengg.parquet"
OUT_EV = "data/mahindra/full_charge_events.parquet"
OUT_SOH = "data/mahindra/full_charge_soh.parquet"
OUT_SUM = "data/mahindra/full_charge_summary.parquet"
OUT_DCIR = "data/mahindra/full_charge_dcir.parquet"
OUT_REP = "data/mahindra/full_charge_report.json"

# --- cleaning / detection knobs (validated in prototype) -----------------------------------
SOC_LO, SOC_HI = 0.0, 100.0     # drop SoC glitches (raw feed has values up to 837)
CUR_MAX = 400.0                  # drop current sentinels (raw feed has spikes to 65,279 A)
CHG_MIN = 2.0                    # |current| above this = actively charging (A)
GAP_S = 600.0                    # >10 min gap ends a charge event
MIN_ROWS = 5                     # min samples in a charge event
FULL_DSOC = 50.0                 # a FULL charge spans at least this much SoC (%)
FULL_END = 85.0                  # ...and ends at least this high (%)
CAP_LO, CAP_HI = 40.0, 400.0     # plausible per-event capacity (Ah)
BASE_AGE_LO, BASE_AGE_HI = 0.5, 10.0   # baseline cap0 = median full-charge cap in this age window
# --- DCIR (internal-resistance growth) knobs -----------------------------------------------
V_LO, V_HI = 20.0, 60.0          # valid pack-voltage window (feed has 0.0 V and >1000 V glitches)
DCIR_DI_MIN = 30.0               # min current step |ΔI| (A) to read dV/dI
DCIR_DT_MAX = 2.0                # ...across at most this many seconds
DCIR_R_LO, DCIR_R_HI = 1.0, 50.0  # plausible pack resistance (mΩ)
DCIR_BAND = (40.0, 70.0)         # SoC band the age-trend is measured in (control for SoC confound)


def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def load_clean(dset, vin):
    t = dset.to_table(columns=["vin", "eventAt", "soc", "current", "batteryVoltage"],
                      filter=pc.field("vin") == pc.scalar(vin)).to_pandas()
    t["t"] = pd.to_datetime(pd.to_numeric(t["eventAt"], errors="coerce"), unit="ms")
    for c in ("soc", "current", "batteryVoltage"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["t", "soc", "current"])
    n_raw = len(t)
    t = t[(t["soc"] >= SOC_LO) & (t["soc"] <= SOC_HI) & (t["current"].abs() <= CUR_MAX)]
    t = t.rename(columns={"batteryVoltage": "voltage"})     # voltage kept raw; DCIR gates it per-step
    return t.sort_values("t").reset_index(drop=True), n_raw


def dcir_events(df, di_min=DCIR_DI_MIN, dt_max=DCIR_DT_MAX):
    """Internal-resistance proxy R = |dV/dI| (mΩ) at fast current step-changes (charge start/stop,
    load transients). Voltage is gated to a physical window first (the feed reads 0.0 V on 20-37% of
    rows and has kV spikes). Returns one row per usable step with its SoC (for SoC-conditioning)."""
    d = df[df["voltage"].between(V_LO, V_HI)].sort_values("t")
    if len(d) < 20:
        return pd.DataFrame(columns=["t", "soc", "r", "dI"])
    dt = d["t"].diff().dt.total_seconds()
    dI = d["current"].diff()
    dV = d["voltage"].diff()
    step = (dI.abs() >= di_min) & (dt <= dt_max) & (dt > 0)
    r = (dV / dI).abs() * 1000.0
    e = pd.DataFrame({"t": d["t"].to_numpy(), "soc": d["soc"].to_numpy(),
                      "r": r.to_numpy(), "dI": dI.abs().to_numpy()})[step.to_numpy()]
    return e[e["r"].between(DCIR_R_LO, DCIR_R_HI)].reset_index(drop=True)


def charge_events(df):
    """One row per charge event: soc0/soc1, ΔSoC, Ah delivered, capacity, is_full."""
    d = df
    dt = d["t"].diff().dt.total_seconds().fillna(0.0)
    sd = d["soc"].diff()
    med = d.loc[sd > 0, "current"].median()                 # sign of charging current for THIS vehicle
    sign = np.sign(med) if (np.isfinite(med) and med != 0) else 1.0
    chg = (np.sign(d["current"]) == sign) & (d["current"].abs() > CHG_MIN)
    start = chg & ~(chg.shift(1, fill_value=False) & (dt <= GAP_S))
    d2 = d.assign(ev=start.cumsum())[chg]
    rows = []
    for _, g in d2.groupby("ev"):
        if len(g) < MIN_ROWS:
            continue
        dsoc = float(g["soc"].iloc[-1] - g["soc"].iloc[0])
        if dsoc <= 1.0:
            continue
        secs = (g["t"] - g["t"].iloc[0]).dt.total_seconds().to_numpy()
        ah = abs(_trapz(g["current"].abs().to_numpy(), secs / 3600.0))
        cap = ah / (dsoc / 100.0)
        rows.append(dict(t0=g["t"].iloc[0], soc0=float(g["soc"].iloc[0]), soc1=float(g["soc"].iloc[-1]),
                         dsoc=dsoc, ah=ah, cap=cap, n=int(len(g))))
    e = pd.DataFrame(rows)
    if len(e):
        e = e[e["cap"].between(CAP_LO, CAP_HI)].reset_index(drop=True)
        e["is_full"] = (e["dsoc"] >= FULL_DSOC) & (e["soc1"] >= FULL_END)
    return e


def reg_dates(fe):
    out = {}
    for vin, g in fe.groupby("vin"):
        g = g.sort_values("age_months")
        m0 = pd.to_datetime(g["ymd"].iloc[0]); a0 = float(g["age_months"].iloc[0])
        out[vin] = m0 - pd.DateOffset(months=int(round(a0)))
    return out


def _cv(s):
    s = pd.Series(s).dropna()
    return float(s.std() / s.mean() * 100.0) if (len(s) > 1 and s.mean()) else np.nan


def _write_artifacts(ev_all, soh_all, dcir_all, summ, n_cohort, t_start, final=False):
    """Write all four artifacts + the report. Called periodically (checkpoint) and at the end, so a
    kill mid-run (e.g. the ~10-min background cap) leaves durable, resumable partial results."""
    EV = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
    SOH = pd.concat(soh_all, ignore_index=True) if soh_all else pd.DataFrame()
    DC = pd.concat(dcir_all, ignore_index=True) if dcir_all else pd.DataFrame()
    SUM = pd.DataFrame(summ)
    EV.to_parquet(OUT_EV, index=False)
    SOH.to_parquet(OUT_SOH, index=False)
    DC.to_parquet(OUT_DCIR, index=False)
    SUM.to_parquet(OUT_SUM, index=False)
    good = SUM[SUM["n_full"] >= 3] if len(SUM) else SUM
    dtr = SUM.dropna(subset=["dcir_slope"]) if ("dcir_slope" in SUM.columns) else pd.DataFrame()
    report = dict(
        cohort=n_cohort, vehicles_with_events=int(len(SUM)), complete=bool(final),
        vehicles_ge3_full=int(len(good)), total_charge_events=int(len(EV)),
        total_full_charges=int(EV["is_full"].sum()) if len(EV) else 0,
        median_full_per_vehicle=float(SUM["n_full"].median()) if len(SUM) else 0.0,
        cv_session_median=round(float(SUM["cv_session"].median()), 1) if len(SUM) else None,
        cv_full_median=round(float(good["cv_full"].median()), 1) if len(good) else None,
        cv_reduction_pp=(round(float(SUM["cv_session"].median() - good["cv_full"].median()), 1)
                         if (len(good) and SUM["cv_session"].notna().any()) else None),
        dcir_vehicles=int((SUM["n_dcir"] > 0).sum()) if "n_dcir" in SUM.columns else 0,
        dcir_median_mohm=round(float(SUM["dcir_med"].median()), 1) if len(dtr) else None,
        dcir_slope_median_mohm_yr=round(float(dtr["dcir_slope"].median()), 2) if len(dtr) else None,
        dcir_frac_trending=round(float((dtr["dcir_r2"] > 0.3).mean()), 2) if len(dtr) else None,
        dcir_band=list(DCIR_BAND),
        full_dsoc=FULL_DSOC, full_end=FULL_END, elapsed_s=round(time.time() - t_start, 0))
    json.dump(report, open(OUT_REP, "w"), indent=2)
    return report


def main(limit=None):
    fe = pd.read_parquet(FEATENGG); fe["vin"] = fe["vin"].astype(str)
    dset = ds.dataset(RAW, format="parquet")
    raw_vins = set(map(str, pc.unique(dset.to_table(columns=["vin"])["vin"]).to_pylist()))
    cohort = sorted(set(fe["vin"].unique()) & raw_vins)
    if limit:
        cohort = cohort[:limit]
    reg = reg_dates(fe)
    sess_cv = {v: _cv(pd.to_numeric(g["capacity_ah"], errors="coerce"))
               for v, g in fe.groupby("vin")}

    ev_all, soh_all, dcir_all, summ = [], [], [], []
    done = set()
    # RESUME from a prior DCIR-schema checkpoint (survives the ~10-min background kill) ---------
    if os.path.exists(OUT_SUM):
        try:
            prev = pd.read_parquet(OUT_SUM)
            if "n_dcir" in prev.columns and len(prev) > 1:           # a real partial, not the smoke test
                done = set(prev["vin"].astype(str)); summ = prev.to_dict("records")
                ev_all = [pd.read_parquet(OUT_EV)] if os.path.exists(OUT_EV) else []
                soh_all = [pd.read_parquet(OUT_SOH)] if os.path.exists(OUT_SOH) else []
                dcir_all = [pd.read_parquet(OUT_DCIR)] if os.path.exists(OUT_DCIR) else []
                print(f"resuming: {len(done)} vehicles already done", flush=True)
        except Exception as ex:
            print(f"resume failed ({ex}); starting fresh", flush=True); done = set()
    todo = [v for v in cohort if v not in done]
    if limit:
        todo = todo[:limit]
    t_start = time.time()
    for i, vin in enumerate(todo, 1):
        try:
            df, n_raw = load_clean(dset, vin)
            if len(df) < 50:
                continue
            r = reg.get(vin)
            # --- DCIR: resistance-growth signal (independent of charge events) --------------
            de = dcir_events(df)
            dcir_med = dcir_slope = dcir_r2 = np.nan
            if len(de):
                de["vin"] = vin
                de["age_months"] = ((de["t"] - r).dt.days / 30.4) if r is not None else np.nan
                if len(de) > 2000:
                    de = de.sample(2000, random_state=0).sort_values("t")
                dcir_all.append(de[["vin", "t", "age_months", "soc", "r", "dI"]])
                dcir_med = float(de["r"].median())
                band = de[de["soc"].between(*DCIR_BAND)].dropna(subset=["age_months"])
                if len(band) >= 8:                              # SoC-conditioned monthly trend
                    mm = band.assign(mo=band["age_months"].round()).groupby("mo")["r"].median()
                    if len(mm) >= 4:
                        b = np.polyfit(mm.index.values, mm.values, 1)
                        dcir_slope = float(b[0] * 12.0)         # mΩ per year, within the SoC band
                        pred = np.polyval(b, mm.index.values); ss = float(((mm.values - mm.values.mean()) ** 2).sum())
                        dcir_r2 = float(1 - ((mm.values - pred) ** 2).sum() / ss) if ss > 0 else np.nan
            # --- capacity from full charge events ------------------------------------------
            e = charge_events(df)
            if e.empty:
                continue
            e["vin"] = vin
            e["age_months"] = ((e["t0"] - r).dt.days / 30.4) if r is not None else np.nan
            full = e[e["is_full"]].copy()
            # baseline capacity: median of early full charges (clean, so the median is stable)
            base = full[full["age_months"].between(BASE_AGE_LO, BASE_AGE_HI)]["cap"]
            cap0 = float(base.median()) if len(base) >= 2 else (
                float(full.sort_values("age_months")["cap"].head(5).median()) if len(full) else np.nan)
            if np.isfinite(cap0) and cap0 > 0 and len(full):
                full["cap0"] = cap0
                full["soh_full"] = np.clip(100.0 * full["cap"] / cap0, None, 100.0)
                soh_all.append(full[["vin", "t0", "age_months", "cap", "cap0", "soh_full", "dsoc", "soc0", "soc1"]])
            ev_all.append(e[["vin", "t0", "age_months", "soc0", "soc1", "dsoc", "ah", "cap", "is_full"]])
            fcv = _cv(full["cap"])
            summ.append(dict(vin=vin, n_events=int(len(e)), n_full=int(len(full)),
                             cap0=cap0, cv_full=fcv, cv_session=sess_cv.get(vin, np.nan),
                             soh_last=(float(full.sort_values("age_months")["soh_full"].iloc[-1])
                                       if (len(full) and np.isfinite(cap0)) else np.nan),
                             age_last=float(e["age_months"].max()) if e["age_months"].notna().any() else np.nan,
                             n_dcir=int(len(de)), dcir_med=dcir_med, dcir_slope=dcir_slope, dcir_r2=dcir_r2))
            if i % 6 == 0:                                    # checkpoint: durable partial results
                _write_artifacts(ev_all, soh_all, dcir_all, summ, len(cohort), t_start)
                print(f"  [{len(done)+i}/{len(cohort)}] {time.time()-t_start:.0f}s · checkpointed", flush=True)
        except Exception as ex:
            print(f"  !! {vin}: {ex}", flush=True)

    report = _write_artifacts(ev_all, soh_all, dcir_all, summ, len(cohort), t_start, final=True)
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_EV}, {OUT_SOH}, {OUT_DCIR}, {OUT_SUM}, {OUT_REP}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
