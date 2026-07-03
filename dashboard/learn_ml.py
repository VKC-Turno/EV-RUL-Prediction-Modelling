"""Learn ML — a guided, from-scratch walkthrough of how our battery State-of-Health models are built.

A teaching dashboard for someone with NO machine-learning background. It walks through every step of
the real pipeline — problem, data, target, features, train/validation/test split, training, feature
importance, errors & overfitting, leave-one-vehicle-out validation, forecasting with uncertainty, and
limits — showing **all three fleets (Euler · Mahindra · Bajaj) side by side** on every page, so you can
compare which features each OEM's feed provides and which ones each model actually relies on.

Run:  .venv/bin/streamlit run dashboard/learn_ml.py
"""
import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import data_quality                                     # shared data-quality gate (drops data-thin vehicles)
import rul_km                                            # SoH forecast -> remaining kilometres to end-of-life
import soh_audit                                         # SoH-signal artifact audit (cliff / stuck-floor / iso-floor)
import training_curation                                 # robust √t-smoothed training-set buckets
import config                                            # single source for warranty terms (FLEET_WARRANTY)

_ico = ROOT / "turno_logo.png"
st.set_page_config(page_title="Turno · Battery SoH Prediction Pipeline", layout="wide",
                   page_icon=str(_ico) if _ico.exists() else "🔋")

TEAL, AMBER, RED, GREEN, GREY = "#1f9e8f", "#e0922b", "#d4504e", "#2ec16b", "#9fb3c8"
AX = dict(gridcolor="#1c2738", zerolinecolor="#1c2738", color="#8aa0b6", linecolor="#27374e")

# ── per-OEM config: each OEM differs in how SoH is measured and which model module it uses ──
OEMS = {
    "Euler": dict(
        ft="data/euler/features/feature_table.parquet", module="euler_model",
        soh_method="BMS remaining-capacity",
        soh_explain="The battery's own management system reports how much charge it can still hold "
                    "(in amp-hours). We divide that by its original capacity to get SoH %.",
        model_desc="**gradient-boosted decision trees** (XGBoost for monthly loss, plus a LightGBM "
                   "'trajectory' model that adds uncertainty bands)",
        lovo=dict(overall=3.50, model=5.03, persist=5.77, trend=5.63, band=0.80),
        label="Euler electric-3-wheelers"),
    "Mahindra": dict(
        ft="data/mahindra/features/feature_table.parquet", module="model",
        soh_method="Coulomb counting",
        soh_explain="We measure the electric **current** flowing in and out of the pack and add it up "
                    "over time (∫ current · time) to track how much capacity remains — the gold-standard "
                    "way to measure SoH.",
        model_desc="**gradient-boosted decision trees** (LightGBM, predicting monthly SoH loss at "
                   "several confidence levels)",
        lovo=dict(overall=3.15, model=4.62, persist=5.37, trend=4.85, band=None),
        label="Mahindra Treo / Zor electric-3-wheelers"),
    "Bajaj": dict(
        ft="data/bajaj/features/feature_table.parquet", module="bajaj_model",
        soh_method="BMS-reported SoH",
        soh_explain="Bajaj's battery management system reports its own SoH estimate directly — the feed "
                    "has no current or voltage, so we can't measure capacity ourselves. We just clean the "
                    "reported value (monthly median, kept non-increasing) and trust it.",
        model_desc="**gradient-boosted decision trees** (LightGBM rate model). With no current/voltage, "
                   "it leans on age, temperature, charge habits and mileage",
        lovo=dict(overall=1.14, model=1.04, persist=3.08, trend=2.06, band=None),
        label="Bajaj RE / cargo electric-3-wheelers"),
    "Piaggio": dict(
        ft="data/piaggio/features/feature_table.parquet", module="model",
        soh_method="Coulomb counting",
        soh_explain="Like Mahindra, we integrate the pack **current** from the intellicar feed "
                    "(∫ current · time) to track remaining capacity — coulomb counting, the gold-standard method.",
        model_desc="**gradient-boosted decision trees** (LightGBM, predicting monthly SoH loss at several "
                   "confidence levels) — the same model as Mahindra",
        lovo=dict(overall=2.83, model=2.96, persist=2.70, trend=3.13, band=None),
        label="Piaggio Ape E-Xtra FX electric-3-wheelers"),
}


def lay(**kw):
    b = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#cdd9e8", size=12), margin=dict(l=50, r=20, t=30, b=40),
             legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)), height=380)
    b.update(kw); return b


@st.cache_data(show_spinner=False)
def load_ft(path):
    return pd.read_parquet(path).sort_values(["vin", "month"])


def deg_filter(m, mode):
    """Restrict the frame to a training population: 'deg' = degraders (≥2pp SoH drop), 'safe' = flat/near-new
    (<2pp drop), 'all'/False = full dataset. (Callers filter only the TRAIN split; val/test stay fixed.)"""
    if mode in (None, False, "all"):
        return m
    d = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
    keep = d[d >= 2].index if mode == "deg" else d[d < 2].index
    return m[m["vin"].isin(keep)]


def _train_vins(vins, drop, mode):
    """Filter a vin set to the chosen training population ('deg' ≥2pp / 'safe' <2pp / else all)."""
    if mode == "deg":
        return {v for v in vins if drop[v] >= 2}
    if mode == "safe":
        return {v for v in vins if drop[v] < 2}
    return set(vins)


def _split(vins, drop, seed=0, force_train=frozenset()):
    """By-vehicle 60/20/20 split, stratified so each split keeps degraders and flat vehicles.
    `force_train` vehicles (the completely-aged / reached-EoL ones) are ALL placed in train — they are
    far too scarce to 'spend' on val/test, and a model can only *learn* end-of-life behaviour from
    vehicles that actually reached it. (Honest aged-vehicle accuracy then comes from Step-9 LOVO, which
    holds out each vehicle one at a time, not from the held-out test set.)"""
    rng = np.random.RandomState(seed)
    ft = set(force_train) & set(vins)
    out = [set(ft), set(), set()]
    pool = [v for v in vins if v not in ft]
    for grp in (sorted(v for v in pool if drop[v] >= 2), sorted(v for v in pool if drop[v] < 2)):
        grp = list(grp); rng.shuffle(grp); n = len(grp)
        ntr, nva = int(n * 0.6), int(n * 0.2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out


@st.cache_data(show_spinner="Training the model…")
def diagnostics(oem_key, deg_only=False):
    """Train one model on the TRAIN split; report per-transition RMSE on each split + feature importance."""
    cfg = OEMS[oem_key]
    m = data_quality.apply_quality(load_cohort(oem_key), oem_key)   # drop data-thin; split FIXED regardless of deg_only
    mod = importlib.import_module(cfg["module"])
    FEATS = mod.FEATS
    g = m.groupby("vin")
    drop = (g["soh"].first() - g["soh"].last())
    smin = g["soh"].min(); aged = set(smin[smin <= EOL_PCT[oem_key]].index)   # completely-aged -> always train
    TR, VA, TE = _split(list(m["vin"].unique()), drop, force_train=aged)
    tr = _train_vins(TR, drop, deg_only)                       # toggle filters TRAIN only (val/test fixed)
    t_tr = mod.build_transitions(m[m["vin"].isin(tr)])
    if hasattr(mod, "train"):                       # Euler rate model (XGBoost regression)
        reg = mod.train(t_tr); bias = float(getattr(reg, "_cal_bias", 0.0))
    else:
        # Mahindra: use a plain regression model (the kind build_mahindra uses), NOT the quantile-0.5
        # model — the loss target is zero-inflated, so the median quantile degenerates to predicting 0
        # everywhere and never splits (all-zero feature importance). Squared-error regression splits fine.
        import xgboost as xgb
        reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=8, verbosity=0).fit(
            t_tr[FEATS].to_numpy(), t_tr["loss"].to_numpy(), sample_weight=t_tr["w"].to_numpy())
        bias = 0.0

    def rmse(vs):
        t = mod.build_transitions(m[m["vin"].isin(vs)])
        if not len(t):
            return None
        pred = np.clip(reg.predict(t[FEATS].to_numpy()) + bias, 0.0, None)
        return round(float(np.sqrt(np.mean((t["loss"].to_numpy() - pred) ** 2))), 4)

    fi = sorted(zip(FEATS, [float(x) for x in reg.feature_importances_]), key=lambda x: -x[1])
    return {"sizes": {"train": len(tr), "validation": len(VA), "test": len(TE)},
            "errors": {"train": rmse(tr), "validation": rmse(VA), "test": rmse(TE)},
            "fi": fi,
            "splits": {"train": sorted(tr), "validation": sorted(VA), "test": sorted(TE)}}


@st.cache_resource(show_spinner=False)
def forecaster(oem_key, deg_only=False):
    cfg = OEMS[oem_key]
    mod = importlib.import_module(cfg["module"])
    m = deg_filter(data_quality.apply_quality(load_cohort(oem_key), oem_key), deg_only)
    if oem_key == "Euler":
        if deg_only in (False, "all"):                    # use the persisted all-vehicle model
            import euler_train
            b = euler_train.load_latest()
            if b and b.get("traj_model"):
                return mod, b["traj_model"]
        return mod, mod.train_traj(mod.build_traj_samples(m))
    return mod, mod.train_quantiles(mod.build_transitions(m))


def forecast_demo(oem_key, m, deg_only=False):
    """Pick a clear teaching example (decent history, real decline, sensible non-cliff forecast)."""
    mod, fmodel = forecaster(oem_key, deg_only)          # model trained on degraders only when toggled
    H = 120                                               # forecast far ahead; _forecast_fig truncates at EoL
    m = data_quality.apply_quality(m, oem_key)           # never forecast a data-thin vehicle
    grp = m.groupby("vin")
    o = pd.DataFrame({"months": grp.size(), "s0": grp.soh.first(), "s1": grp.soh.last()})
    o["dropp"] = o.s0 - o.s1
    min_mo = min(15, int(o.months.median()))
    cand = o[(o.months >= min_mo) & (o.dropp >= 2)].sort_values("months", ascending=False)
    order = list(cand.index) or list(o.sort_values("months", ascending=False).index)

    def run(vin):
        gg = m[m["vin"] == vin].sort_values("month").reset_index(drop=True)
        if oem_key == "Euler":
            fc = mod.forecast(gg, fmodel, H); return gg, fc[0.1], fc[0.5], fc[0.9]
        sim = mod.simulate(gg, fmodel, H)
        return gg, sim["q10"].to_numpy(), sim["q50"].to_numpy(), sim["q90"].to_numpy()

    for vin in order:
        res = run(vin)
        if res[2][min(29, len(res[2]) - 1)] >= 50:       # P50 still sensible at ~2.5 yr (not an instant cliff)
            return res
    return run(order[0])


# Per-OEM representative warranty — SINGLE SOURCE = config.FLEET_WARRANTY (so config and dashboard never drift).
# Euler HiLoad 3yr/80k (provisional); Mahindra Treo 3yr; Bajaj RE battery 5yr/120k (km usually binds first).
WARRANTY_YR = {o: config.FLEET_WARRANTY[o.lower()][0] for o in ("Euler", "Mahindra", "Bajaj", "Piaggio")}
WARRANTY_KM = {o: config.FLEET_WARRANTY[o.lower()][1] for o in ("Euler", "Mahindra", "Bajaj", "Piaggio")}
ODO_OK = {"Euler": False, "Mahindra": False, "Bajaj": True, "Piaggio": False}   # odometer reliable enough for a km-bound deadline?
EOL_PCT = {"Euler": 80, "Mahindra": 80, "Bajaj": 70, "Piaggio": 80}   # end-of-life SoH threshold per OEM (Bajaj = 70%)


def eff_warr_months(oem, g, wyr):
    """Effective warranty deadline (age-months) = min(time term, time to hit the km limit at recent pace) —
    'whichever of time / km comes first'. Falls back to time-only where the odometer is unreliable (Euler
    noisy / Mahindra sparse) so a bad km projection is never trusted."""
    time_m = wyr * 12.0
    if not ODO_OK.get(oem) or "km_month" not in g.columns:
        return time_m
    kmcol = "cum_km" if "cum_km" in g.columns else ("odo_max" if "odo_max" in g.columns else None)
    if kmcol is None:
        return time_m
    cur_km = float(g[kmcol].iloc[-1]); cur_age = float(g["age_months"].iloc[-1])
    kmpm = float(pd.Series(g["km_month"]).tail(6).median())
    if not (kmpm > 50 and 0 < cur_km < WARRANTY_KM[oem]):      # need a sane pace, not already over the limit
        return time_m
    return float(min(time_m, cur_age + (WARRANTY_KM[oem] - cur_km) / kmpm))
# Euler/Mahindra SoH is renormalised to 100% at registration (so we anchor the curve at 100 there); Bajaj
# uses the ABSOLUTE BMS-reported SoH, so it legitimately starts below 100 — no 100% anchor (no fake drop).
RENORM100 = {"Euler": True, "Mahindra": True, "Bajaj": False, "Piaggio": True}

# vin -> registration date (= age 0 on the SoH curves), per OEM's registration file
REG_FILES = {
    "Euler": ("data/euler/Euler_Regd_Details.csv", "regd_date", "%d/%m/%y"),
    "Mahindra": ("Mh_Regd_Date.csv", "vehicle_registration_date", None),
    "Bajaj": ("Bajaj_Regd_Details.csv", "regd_date", None),
}


@st.cache_data(show_spinner=False)
def reg_dates(oem_key):
    if oem_key not in REG_FILES:                 # Piaggio has no reg file -> first-telemetry anchoring
        return {}
    f, col, fmt = REG_FILES[oem_key]
    if not Path(f).exists():
        return {}
    r = pd.read_csv(f)
    d = pd.to_datetime(r[col], format=fmt, errors="coerce") if fmt else pd.to_datetime(r[col], errors="coerce")
    return dict(zip(r["vin"], d))


@st.cache_data(show_spinner=False)
def warranty_map(oem_key):
    """(default_years, {vin: years}). Warranty term per vehicle — Mahindra varies by model
    (Treo 3 yr / Zor Grand 5 yr); Euler/Bajaj use a single term. The warranty *date* = registration +
    these years, which on the registration-anchored age axis is simply `years × 12 months`."""
    import config
    default = WARRANTY_YR[oem_key]
    vmp = "data/manifests/mahindra_vin_model.csv"
    if oem_key == "Mahindra" and Path(vmp).exists():
        vm = dict(pd.read_csv(vmp).values)
        return default, {v: config.warranty_for("mahindra", m)[0] for v, m in vm.items()}
    return default, {}


@st.cache_data(show_spinner="Forecasting every vehicle…")
def all_split_predictions(oem_key, deg_only=False):
    """Train ONCE on the TRAIN split, then forecast every vehicle in each split from its latest data out to
    its own warranty deadline. Returns {'train':[...], 'validation':[...], 'test':[...]}.
    NOTE: 'train' vehicles were used to fit the model (in-sample); validation & test are held out."""
    cfg = OEMS[oem_key]
    m = data_quality.apply_quality(load_cohort(oem_key), oem_key)   # drop data-thin; splits FIXED regardless of deg_only
    mod = importlib.import_module(cfg["module"])
    g = m.groupby("vin")
    drop = g["soh"].first() - g["soh"].last()
    smin = g["soh"].min(); aged = set(smin[smin <= EOL_PCT[oem_key]].index)   # completely-aged -> always train
    TR, VA, TE = _split(list(m["vin"].unique()), drop, force_train=aged)
    tr_vins = _train_vins(TR, drop, deg_only)                                # fit on the TRAIN split only
    train = m[m["vin"].isin(tr_vins)]
    euler = oem_key == "Euler"
    fmodel = (mod.train_traj(mod.build_traj_samples(train)) if euler
              else mod.train_quantiles(mod.build_transitions(train)))
    reg = reg_dates(oem_key); wdef, wmap = warranty_map(oem_key)

    def forecast_vin(vin):
        gg = m[m["vin"] == vin].sort_values("month").reset_index(drop=True); n = len(gg)
        if n < 6:
            return None
        # Forecast from the LATEST (present) data point using the vehicle's FULL observed history, so the
        # forecast begins exactly where the measured line ends.
        hist = gg; cut_age = float(gg["age_months"].iloc[-1])
        warr_age = eff_warr_months(oem_key, gg, wmap.get(vin, wdef))   # km-bound: min(time term, time-to-120k-km)
        H_MAX = 120                                             # cap (months) to avoid absurd extrapolation
        # Only roll forward as far as the display needs — to the warranty deadline + a buffer to catch EoL
        # crossings. Vehicles that survive well past warranty are still classified 'safe' without the full
        # 120-month roll, so this cuts Step-10 latency ~2x without changing any outcome.
        H = int(np.clip(round(warr_age - cut_age) + 30, 12, H_MAX))
        if euler:
            fc = mod.forecast(hist, fmodel, H); p10, p50, p90 = fc[0.1], fc[0.5], fc[0.9]
        else:
            sim = mod.simulate(hist, fmodel, H)
            p10, p50, p90 = sim["q10"].to_numpy(), sim["q50"].to_numpy(), sim["q90"].to_numpy()
        # extend until the P50 forecast reaches EoL (even past warranty); else stop a little past the deadline.
        hit = np.where(np.asarray(p50) <= EOL_PCT[oem_key])[0]
        end = (int(hit[0]) + 4) if len(hit) else int(round(warr_age - cut_age)) + 6
        end = int(np.clip(max(end, round(warr_age - cut_age) + 2), 3, H))
        last = float(gg["soh"].iloc[-1])                        # anchor the forecast to the present SoH
        fage = np.concatenate([[cut_age], cut_age + np.arange(1, end + 1)])
        p10 = np.concatenate([[last], p10[:end]]); p50 = np.concatenate([[last], p50[:end]])
        p90 = np.concatenate([[last], p90[:end]])
        rd = reg.get(vin)
        return dict(vin=vin[-6:], reg=(rd.strftime("%b '%y") if pd.notna(rd) else "?"),
                    warr_age=warr_age, age=gg["age_months"].to_numpy().tolist(),
                    soh=gg["soh"].to_numpy().tolist(), fage=fage.tolist(),
                    p10=p10.tolist(), p50=p50.tolist(), p90=p90.tolist())

    out = {}
    for name, vins in (("train", TR), ("validation", VA), ("test", TE)):
        out[name] = [p for p in (forecast_vin(v) for v in sorted(vins)) if p]
    return out


def concept(t): st.info("💡 **Why it matters** — " + t)
def takeaway(t): st.success("✅ **Takeaway** — " + t)


# ───────────────────────────── sidebar / navigation ─────────────────────────────
_logo = next((p for p in (ROOT / "turno_logo.png", ROOT / "turno.gif", ROOT / "image.png",
                           ROOT / "dashboard" / "image.png", ROOT / "assets" / "image.png") if p.exists()), None)
if _logo:
    st.sidebar.image(str(_logo), width=110)
st.sidebar.title("Battery SoH · Prediction Pipeline")
st.sidebar.caption("How our battery-health models are built — explained from scratch, **comparing all "
                   "three fleets side by side.**")
SMOOTH = st.sidebar.checkbox("Smooth SoH curves", value=True,
                             help="Round the staircase from the monotonic SoH envelope into a curve "
                                  "(display only — the model still uses the raw monthly SoH).")
# DEG_ONLY holds the training-population MODE: "all" | "deg" (degraders, ≥2pp drop) | "safe" (flat/near-new,
# <2pp). Only the TRAIN split is filtered — validation/test stay the SAME, so it's a fair comparison.
_TRAIN_ON = st.sidebar.radio("Train on", ["All vehicles", "Degraders only", "Safe only"], index=0,
                             help="Which vehicles the model is FIT on. Degraders lost ≥2% SoH; Safe lost <2% "
                                  "(flat / near-new). Validation & test vehicles stay identical, so the warranty-"
                                  "risk counts are a fair comparison. Affects Steps 5/7/8/10. 'All vehicles' is default.")
DEG_ONLY = {"All vehicles": "all", "Degraders only": "deg", "Safe only": "safe"}[_TRAIN_ON]
if DEG_ONLY != "all":
    st.sidebar.warning(f"⚠️ Training on **{_TRAIN_ON}** (validation/test held fixed).")


def smooth(s, win=5):
    return s.rolling(win, center=True, min_periods=1).mean() if SMOOTH else s


STEPS = ["📋 Overview", "1 · The problem", "2 · The data", "3 · The target (SoH)", "4 · Data quality",
         "5 · Features", "6 · Train / Validation / Test", "7 · Training the model",
         "8 · Which clues matter?", "9 · Errors & overfitting", "10 · A tougher test (LOVO)",
         "11 · Predicting the future", "12 · Range & km left", "13 · Limits & retraining",
         "14 · Validation & data needs", "15 · Native-only Mahindra SoH"]
step = st.sidebar.radio("Steps", STEPS, label_visibility="collapsed")
st.sidebar.markdown("---")

OEM_KEYS = list(OEMS.keys())


@st.cache_data(show_spinner=False)
def _store_cohort(oem, _mtime):
    """Raw (ungated) Redshift store feature table, renamed to the local schema (ymd→month, str vin)."""
    r = pd.read_parquet(f"data/redshift/{oem.lower()}_featengg.parquet").rename(columns={"ymd": "month"})
    r["month"] = pd.to_datetime(r["month"].astype(str)); r["vin"] = r["vin"].astype(str)
    return r


def load_cohort(oem):
    """The cohort the WHOLE dashboard runs on: prefer the full Redshift STORE table (much larger), fall
    back to the local feature table when the store is absent or not bigger. Cache-busts on parquet mtime."""
    p = f"data/redshift/{oem.lower()}_featengg.parquet"
    if os.path.exists(p):
        store = _store_cohort(oem, os.path.getmtime(p))
        if store["vin"].nunique() > load_ft(OEMS[oem]["ft"])["vin"].nunique():
            return store
    return load_ft(OEMS[oem]["ft"])


FEATS_BY = {o: load_cohort(o) for o in OEM_KEYS}                          # store cohort where available


def _fleet_coverage():
    """Authoritative registered-fleet size per OEM (Redshift soh_etl.oem_fleet_kpis, cached to
    data/redshift/fleet_coverage.csv), keyed Title-case. Includes OEMs we don't model yet (Piaggio, Montra).
    Empty dict if the file is absent so the dashboard still runs offline (callers fall back to constants)."""
    p = "data/redshift/fleet_coverage.csv"
    if not os.path.exists(p):
        return {}
    fc = pd.read_csv(p)
    return {str(r.oem).title(): dict(fleet=int(r.fleet),
            modeled=(int(r.modeled) if pd.notna(r.modeled) else None),
            coverage=(float(r.coverage_pct) if pd.notna(r.coverage_pct) else None)) for r in fc.itertuples()}


FLEET_COV = _fleet_coverage()                                            # registered fleet vs modeled, from Redshift

# "Mahindra Native" — a SEPARATE OEM: the ~12k native-only Mahindra fleet, SoH via the distance-per-SoC
# proxy on native data ALONE (built by src/mahindra_native_soh.py, zero intellicar). Never combined with the
# intellicar coulomb "Mahindra"; shown as its own Step-3 column to expose the native feed's lack of SoH signal.
_NAT_SOH_PATH = "data/mahindra/native_monthly_soh.parquet"
NATIVE_SOH = pd.read_parquet(_NAT_SOH_PATH) if os.path.exists(_NAT_SOH_PATH) else None

# ── Native-only Mahindra BAYESIAN probable-SoH model (predicted, no measured SoH). It is NOT routed through the
# supervised machinery (no feature table / trained model); it appears as an extra COMPARISON column on the pages
# where that's a like-for-like comparison (target, features, forecasting, validation, limits), clearly labelled
# 'predicted'. Full model lives in Step 15. ──
import json as _json


def _load_json(p):
    return _json.load(open(p)) if os.path.exists(p) else None


NAT_BAYES = pd.read_parquet("data/mahindra/native_behaviour_soh.parquet") if os.path.exists("data/mahindra/native_behaviour_soh.parquet") else None
NAT_REPORT = _load_json("data/mahindra/behaviour_soh_report.json")
NAT_VAL_CUR = pd.read_parquet("data/mahindra/behaviour_validation_curves.parquet") if os.path.exists("data/mahindra/behaviour_validation_curves.parquet") else None
NAT_VAL_ACT = pd.read_parquet("data/mahindra/behaviour_validation_actual.parquet") if os.path.exists("data/mahindra/behaviour_validation_actual.parquet") else None
NAT_VAL_REP = _load_json("data/mahindra/behaviour_validation_report.json")
NAT_FLEET = max(int((FLEET_COV.get("Mahindra") or {}).get("fleet") or 10933) - FEATS_BY["Mahindra"].vin.nunique(), 0)


def _native_bayes_fig(height=300):
    """Fleet probable-SoH band from the Bayesian behaviour model (predicted; no measured SoH). Amber = Mahindra."""
    if NAT_BAYES is None:
        return None
    ag = NAT_BAYES.groupby("age_months"); med, p10, p90 = ag.soh_p50.mean(), ag.soh_p10.mean(), ag.soh_p90.mean(); x = med.index
    f = go.Figure()
    f.add_scatter(x=x, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip")
    f.add_scatter(x=x, y=p10, fill="tonexty", fillcolor="rgba(224,146,43,0.18)", line=dict(width=0), name="p10–p90 band")
    f.add_scatter(x=x, y=med, line=dict(color=AMBER, width=2.4), name="predicted median")
    f.add_hline(y=80, line=dict(color=RED, dash="dot"))
    f.update_xaxes(title_text="age (months)", **AX); f.update_yaxes(title_text="SoH %", range=[70, 101], **AX)
    f.update_layout(**lay(height=height, margin=dict(l=44, r=14, t=18, b=36)))
    return f
_tv = sum(F.vin.nunique() for F in FEATS_BY.values()); _tm = sum(len(F) for F in FEATS_BY.values())
st.sidebar.caption(f"Running example: **{_tv} vehicles** across Euler · Mahindra · Bajaj, "
                   f"{_tm:,} vehicle-months.")
st.sidebar.progress(STEPS.index(step) / (len(STEPS) - 1), text=f"Step {STEPS.index(step)} of {len(STEPS)-1}")


def ov(F):
    g = F.groupby("vin")
    return pd.DataFrame({"months": g.size(), "s0": g.soh.first(), "s1": g.soh.last(),
                         "smin": g.soh.min(), "age": g.age_months.last()})


# ── feature-availability matrix: which physical signals each fleet's feed actually carries ──
SIGNALS = [
    ("Calendar age", ["age_months"]),
    ("Pack temperature", ["temp_mean", "temp_max", "temp_p95"]),
    ("Ambient temperature", ["amb_temp_mean"]),
    ("State of charge / dwell", ["soc_mean", "frac_soc_high", "frac_soc_low", "dod_mean"]),
    ("Pack current (→ Ah)", ["ah_throughput", "cur_abs_mean", "cur_abs_p95", "cur_chg_mean", "cur_dis_mean"]),
    ("Pack voltage", ["volt_mean", "volt_min", "volt_max"]),
    ("Mileage / odometer", ["odo_max", "km_month", "cum_km"]),
    ("Charge-cycle count", ["cyc_max", "cyc_month", "cum_cycles"]),
    ("Drive efficiency", ["driveeff_mean", "wh_per_km", "dte_mean"]),
    ("Coulomb capacity (Ah)", ["capacity_ah"]),
    ("Cell imbalance", ["imbalance_mean", "cell_imbalance", "imbalance"]),
]


def availability_df():
    # ✅ = MODELLED feature (present in that OEM's feature table). ◦ (EXTRA rows) = present in the raw feed but NOT
    # modelled (weak/flaky for SoH — speed, GPS, drive-state). Native Mahindra carries only age/SoC/odometer as
    # usable features. Cell imbalance is a real degradation signal but only Euler's feed reports it.
    NATIVE_SIG = {"Calendar age", "State of charge / dwell", "Mileage / odometer"}
    EXTRA = [
        ("Vehicle speed",               {"Euler": "—", "Mahindra · intellicar": "—", "Mahindra · native": "◦", "Bajaj": "—", "Piaggio": "◦"}),
        ("GPS / location",              {"Euler": "—", "Mahindra · intellicar": "—", "Mahindra · native": "◦", "Bajaj": "—", "Piaggio": "◦"}),
        ("Operating state (drive/chg)", {"Euler": "◦", "Mahindra · intellicar": "—", "Mahindra · native": "◦", "Bajaj": "—", "Piaggio": "◦"}),
    ]
    rows = []
    for name, cands in SIGNALS:
        row = {"Signal": name}
        for o in OEM_KEYS:
            cols = set(FEATS_BY[o].columns)
            row[o] = "✅" if any(c in cols for c in cands) else "—"
        row["Mahindra Native"] = "✅" if name in NATIVE_SIG else "—"
        rows.append(row)
    df = pd.DataFrame(rows).rename(columns={"Mahindra": "Mahindra · intellicar", "Mahindra Native": "Mahindra · native"})
    for name, vals in EXTRA:
        df = pd.concat([df, pd.DataFrame([{"Signal": name, **vals}])], ignore_index=True)
    order = ["Signal", "Euler", "Mahindra · intellicar", "Mahindra · native", "Bajaj", "Piaggio"]
    return df[[c for c in order if c in df.columns]].fillna("—")


# Per-OEM RAW-field audit (from docs/oem_fields_one_pager.md). ✅ usable · ⚠️ weak/caveats · ❌ not usable.
_FCOLS = ["field", "what it is", "use for SoH / RUL", "data quality"]
OEM_FIELD_AUDIT = {
    "Euler — dense 2023+ feed (BMS remaining-capacity + current/voltage; strongest)": [
        ("batteryRemainingCapacity", "Ah remaining", "✅ SoH target (BMS-capacity)", "94%"),
        ("batteryCurrent", "pack current (signed)", "✅ Ah throughput / C-rate", "94%"),
        ("batteryVoltage", "pack voltage", "✅ voltage-stress feature", "100%"),
        ("batterySoh", "BMS reported SoH", "✅ SoH cross-check", "100% (coarse)"),
        ("batterySoc", "state of charge %", "✅ cycling / DoD / dwell", "100%"),
        ("batteryTemperature", "pack temp °C", "✅ thermal stress", "100%"),
        ("cellImbalance", "cell imbalance", "✅ degradation signal", "68%"),
        ("vehicleMode", "drive / charge mode", "⚠️ usage", "94%"),
        ("odometer", "distance", "⚠️ km/RUL — noisy outliers", "100% present"),
        ("eventAt, vin", "time, id", "✅ keys", "100%"),
    ],
    "Mahindra · Intellicar feed (has current — only ~2% of fleet, ~224 of ~11,000)": [
        ("current", "pack current", "✅ coulomb SoH (only source w/ current)", "100%"),
        ("soc", "state of charge %", "✅ ΔSoC for every method", "100%"),
        ("batteryVoltage", "pack voltage", "✅ energy / health cross-check", "100%"),
        ("odometer", "distance", "✅ distance-per-SoC", "93%"),
        ("dte", "distance-to-empty", "⚠️ range-retention proxy", "100%"),
        ("make, model", "OEM / variant", "⚠️ capacity context", "100%"),
    ],
    "Mahindra · Native OEM feed (~98% of fleet — NO current → SoH not measurable)": [
        ("soc", "state of charge %", "⚠️ distance-per-SoC proxy only", "100% (garbage to clip)"),
        ("odometer", "distance", "⚠️ distance-per-SoC", "100% (0 garbage)"),
        ("distanceToEmpty", "range left", "⚠️ range proxy (fails)", "100%"),
        ("batteryTemp", "pack temp °C", "✅ thermal feature", "~30% (−50/2001 outliers)"),
        ("state", "DRIVE / CHARGE / IDLE", "✅ segmentation feature", "~30%"),
        ("latitude, longitude", "GPS", "✅ climate / season proxy", "100%"),
        ("gearPosition, vehicleModel", "usage / context", "⚠️ low value", "—"),
        ("kwh", "instantaneous power", "❌ not cumulative → not integrable", "25%"),
    ],
    "Bajaj — verbose BMS feed (~10-month history; no current/voltage)": [
        ("essBmsSohcEstPercValue", "reported SoH %", "✅ SoH target (clean, monotone)", "good"),
        ("essBmsChgcycleActCountValue", "charge-cycle count", "✅ direct aging driver", "good"),
        ("essBmsSocEstPercValue", "SoC %", "✅ cycling / dwell", "good"),
        ("essBmsTemperatureActDegcValue", "pack temp °C", "✅ thermal", "good"),
        ("etsVcuAmbienttempActDegcValue", "ambient temp °C", "✅ climate proxy", "good"),
        ("etsVcuDriveeffEstWhpkmValue", "drive efficiency Wh/km", "⚠️ range-fade proxy", "ok"),
        ("hmiIclOdoActMValue", "odometer (metres ÷1000)", "✅ km/RUL — clean", "good"),
        ("evcChgInputenergycountActKwhValue", "charge input energy kWh", "❌ not cumulative", "weak"),
    ],
    "Piaggio · Intellicar feed (has current → coulomb SoH; only ~250 of 1,913 pulled so far)": [
        ("current", "pack current", "✅ coulomb SoH (like Mahindra-IC)", "100%"),
        ("soc", "state of charge %", "✅ ΔSoC for every method", "100%"),
        ("batteryVoltage", "pack voltage", "✅ energy / health cross-check", "100% (IC; native null)"),
        ("chargeCycle", "charge-cycle count", "✅ direct aging driver", "100%"),
        ("odometer", "distance", "✅ km / distance-per-SoC", "good"),
        ("controllerTemperature, motorTemperature", "controller / motor temp", "⚠️ thermal PROXY — no pack temp", "100%"),
        ("distanceTillEmpty", "range left (native feed)", "⚠️ range proxy", "100%"),
        ("make, model", "OEM / variant", "⚠️ capacity context", "100%"),
    ],
}


# ── small per-OEM plot panels (compact, sized to sit three-across) ──
def _soh_fig(oem, h=300, which="all"):
    F = FEATS_BY[oem]; eol = EOL_PCT[oem]; anch = RENORM100[oem]
    fig = go.Figure()
    for vin, g in F.groupby("vin"):
        deg = (g.soh.iloc[0] - g.soh.iloc[-1]) >= 2
        if (which == "deg" and not deg) or (which == "flat" and deg):
            continue
        ax = ([0.0] if anch else []) + (g.age_months / 12).tolist()
        sy = ([100.0] if anch else []) + smooth(g.soh).tolist()
        fig.add_scatter(x=ax, y=sy, mode="lines", line=dict(color=RED if deg else GREY, width=1),
                        opacity=0.45, showlegend=False)
    fig.add_hline(y=eol, line=dict(color=AMBER, dash="dash"))
    fig.update_xaxes(title="age (years)", dtick=1, **AX)
    fig.update_yaxes(range=[min(eol - 5, 55), 101], **AX)
    fig.update_layout(**lay(height=h, margin=dict(l=42, r=8, t=22, b=34)))
    return fig


def _native_soh_fig(which="all", h=300):
    """Native-only Mahindra SoH from the distance-per-SoC proxy (monthly sample). Same layout as _soh_fig
    but on NATIVE_SOH, with RELATIVE observed-age (no reg anchor) and NO monotone envelope (so noise is honest)."""
    fig = go.Figure()
    for vin, g in NATIVE_SOH.groupby("vin"):
        deg = (g.soh.iloc[0] - g.soh.iloc[-1]) >= 2
        if (which == "deg" and not deg) or (which == "flat" and deg):
            continue
        fig.add_scatter(x=(g.age_months / 12).tolist(), y=smooth(g.soh).tolist(), mode="lines",
                        line=dict(color=RED if deg else GREY, width=1), opacity=0.4, showlegend=False)
    fig.add_hline(y=80, line=dict(color=AMBER, dash="dash"))
    fig.update_xaxes(title="months observed (yr)", dtick=1, **AX)
    fig.update_yaxes(range=[55, 111], **AX)
    fig.update_layout(**lay(height=h, margin=dict(l=42, r=8, t=22, b=34)))
    return fig


def _fi_fig(oem, h=330):
    fi = pd.DataFrame(diagnostics(oem, DEG_ONLY)["fi"], columns=["feature", "importance"]).head(10)
    fig = go.Figure(go.Bar(x=fi.importance[::-1], y=fi.feature[::-1], orientation="h", marker_color=TEAL))
    fig.update_xaxes(**AX); fig.update_yaxes(**AX)
    fig.update_layout(**lay(height=h, margin=dict(l=8, r=8, t=20, b=28)))
    return fig


def _err_fig(oem, h=300):
    e = diagnostics(oem, DEG_ONLY)["errors"]
    fig = go.Figure(go.Bar(x=["Train", "Val", "Test"], y=[e["train"], e["validation"], e["test"]],
                           marker_color=[GREEN, AMBER, RED],
                           text=[f"{e['train']:.2f}", f"{e['validation']:.2f}", f"{e['test']:.2f}"],
                           textposition="outside"))
    fig.update_yaxes(**AX); fig.update_xaxes(**AX)
    fig.update_layout(**lay(height=h, margin=dict(l=36, r=8, t=22, b=28)))
    return fig


def _lovo_fig(oem, h=300):
    L = OEMS[oem]["lovo"]
    fig = go.Figure(go.Bar(x=["Model", "Persist", "Trend"], y=[L["model"], L["persist"], L["trend"]],
                           marker_color=[GREEN, GREY, AMBER],
                           text=[L["model"], L["persist"], L["trend"]], textposition="outside"))
    fig.update_yaxes(title="forecast error · RMSE pp (↓ better)", **AX); fig.update_xaxes(**AX)
    fig.update_layout(**lay(height=h, margin=dict(l=36, r=8, t=22, b=28)))
    return fig


def _forecast_fig(oem, h=330):
    F = FEATS_BY[oem]
    g, p10, p50, p90 = forecast_demo(oem, F, DEG_ONLY)
    eol = EOL_PCT[oem]; a0 = g.age_months.iloc[-1]
    wdef, wmap = warranty_map(oem); wyr = wmap.get(g.vin.iloc[0], wdef)
    # show the prediction until P50 reaches EoL (+ a few months); if it never does, stop ~a year past warranty
    hit = np.where(np.asarray(p50) <= eol)[0]
    keep = int(np.clip((int(hit[0]) + 4) if len(hit) else int(round(wyr * 12 - a0)) + 12, 6, len(p50)))
    p10, p50, p90 = p10[:keep], p50[:keep], p90[:keep]
    sm = smooth(g.soh); fa = np.arange(a0 + 1, a0 + len(p50) + 1)
    xc = np.concatenate([[a0], fa]) / 12.0                      # months -> years for the age axis
    c10 = np.concatenate([[sm.iloc[-1]], p10]); c50 = np.concatenate([[sm.iloc[-1]], p50])
    c90 = np.concatenate([[sm.iloc[-1]], p90])
    fig = go.Figure()
    if RENORM100[oem]:
        fig.add_scatter(x=[0, g.age_months.iloc[0] / 12], y=[100, sm.iloc[0]], mode="lines",
                        line=dict(color=TEAL, width=1.2, dash="dot"), showlegend=False)
    else:                                              # Bajaj: telemetry starts months after registration
        a1 = g.age_months.iloc[0] / 12
        fig.add_vrect(x0=0, x1=a1, fillcolor="rgba(159,179,200,.06)", line_width=0)
        fig.add_scatter(x=[0, a1], y=[sm.iloc[0], sm.iloc[0]], mode="lines",
                        line=dict(color=GREY, width=1, dash="dot"), showlegend=False)
        fig.add_annotation(x=0, y=sm.iloc[0], text="reg", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=GREY, size=9))
    fig.add_scatter(x=g.age_months / 12, y=sm, mode="markers+lines", line=dict(color=TEAL, width=2),
                    marker=dict(size=3), showlegend=False)
    fig.add_scatter(x=xc, y=c90, mode="lines", line=dict(width=0, color=GREY), showlegend=False)
    fig.add_scatter(x=xc, y=c10, mode="lines", fill="tonexty", fillcolor="rgba(46,193,107,.18)",
                    line=dict(width=0, color=GREY), showlegend=False)
    fig.add_scatter(x=xc, y=c50, mode="lines", line=dict(color=GREEN, width=2.5, dash="dash"),
                    showlegend=False)
    fig.add_hline(y=EOL_PCT[oem], line=dict(color=AMBER, dash="dash"))
    fig.add_vline(x=wyr, line=dict(color="#9aa7b6", dash="dashdot"))
    fig.update_xaxes(title="age (years)", dtick=1, **AX)
    fig.update_yaxes(range=[60, 101], **AX)            # common 60–100% scale across all fleets
    fig.update_layout(**lay(height=h, margin=dict(l=40, r=8, t=22, b=34)))
    return fig, g.vin.iloc[0], len(p50), wyr


# ── SoH → kilometres: range now (rated × SoH) + remaining km to EoL (× usage rate), mirrors src/rul_km.py ──
RATED_KM = {"Euler": 120, "Mahindra": 80, "Bajaj": 178, "Piaggio": 110}   # rated full-charge range, km (Piaggio=PROVISIONAL, Ape spec unverified)
RATED_KM_SRC = {                                          # OEM source for the promised range (per OEM_Model_Specs.csv)
    "Euler": "https://eulermotors.com/en/hiload",
    "Mahindra": "https://www.mahindralastmilemobility.com/treo-zor-dv",
    "Bajaj": "https://www.bajajauto.com",
    "Piaggio": "https://www.piaggio-cv.co.in",   # PROVISIONAL — Ape E-Xtra FX rated range unverified
}


@st.cache_data(show_spinner=False)
def _euler_rated():
    p = Path("data/manifests/euler_variant_map.csv")
    if not p.exists():
        return {}
    v = pd.read_csv(p)
    return {k: float(x) for k, x in zip(v["vin"], pd.to_numeric(v["rated_km"], errors="coerce")) if pd.notna(x)}


@st.cache_data(show_spinner=False)
def _rul_order(oem):
    """Ordered candidate vins for the warranty cards: most-degraded that REACHED warranty first; else oldest.
    Returns (vins, reached_flag)."""
    m = data_quality.apply_quality(FEATS_BY[oem], oem)
    wyr, wkm = WARRANTY_YR[oem], WARRANTY_KM[oem]
    a = m.groupby("vin").agg(n=("soh", "size"), cur=("soh", "last"),
                             age=("age_months", "last"), odo=("odo_max", "max"))
    a = a[a["n"] >= 8]; a["age_yr"] = a["age"] / 12; a["odoc"] = a["odo"].where(a["odo"] < 3e5)
    reached = a[(a["age_yr"] >= wyr) | (a["odoc"] >= wkm)]             # reached warranty by time OR distance
    if len(reached):
        return list(reached.sort_values("cur").index), True           # worst SoH first
    if len(a):
        return list(a.sort_values("age_yr", ascending=False).index), False   # oldest first
    return list(pd.unique(m["vin"])), False


# Manually-pinned Step-11 pairs (full VINs) — [gentler/green, harder/purple]: same distance, different usage.
_RUL_PAIR_OVERRIDE = {"Euler": ["MD9EMHDL24C217135", "MD9EMHDL24C217054"]}


@st.cache_data(show_spinner=False)
def _rul_pair(oem):
    """Two REAL degraders that started ~100%, drove a SIMILAR distance, but degraded at DIFFERENT rates —
    ending at different SoH (=> different range & remaining life). Both are declining (neither is a flat
    artifact), so it's a fair rate-vs-rate comparison: the point that the same kilometres age batteries very
    differently (usage intensity + heat). Returns [gentler_vin, harder_vin]; falls back to worst-SoH pair."""
    m = data_quality.apply_quality(FEATS_BY[oem], oem)
    ov = _RUL_PAIR_OVERRIDE.get(oem)
    if ov and set(ov).issubset(set(m["vin"].unique())):
        return list(ov)                                            # user-pinned pair (same distance, different usage)
    a = m.groupby("vin").agg(n=("soh", "size"), first=("soh", "first"), last=("soh", "last"),
                             odo=("odo_max", "max"))
    a["drp"] = a["first"] - a["last"]
    for start_hi, min_drop in ((97, 5.0), (90, 4.0), (0, 3.0)):     # prefer 'started ~100 & a clear decline'
        d = a[(a["n"] >= 8) & (a["odo"] < 3e5) & (a["odo"] > 5000)
              & (a["first"] >= start_hi) & (a["drp"] >= min_drop)].sort_values("odo")
        if len(d) < 2:
            continue
        v = d.index.to_numpy(); od = d["odo"].to_numpy(); ls = d["last"].to_numpy()
        tol = max(0.10 * float(np.median(od)), 5000.0)             # "same distance" window
        best = None
        for i in range(len(d)):
            j = i + 1
            while j < len(d) and od[j] - od[i] <= tol:
                gap = abs(ls[i] - ls[j]); score = gap + (od[i] + od[j]) / 2 / 1e5   # different endpoints, tie-break higher km
                if gap >= 5 and (best is None or score > best[0]):
                    best = (score, i, j)
                j += 1
        if best is not None:
            _, i, j = best
            hi, lo = (i, j) if ls[i] >= ls[j] else (j, i)
            return [v[hi], v[lo]]                                   # gentler degrader (green), harder degrader (purple)
    order, _ = _rul_order(oem)
    return order[:2]


@st.cache_data(show_spinner=False)
def _rul_demo(oem, deg_only, vin):
    m = data_quality.apply_quality(FEATS_BY[oem], oem)
    eol, wyr, wkm = EOL_PCT[oem], WARRANTY_YR[oem], WARRANTY_KM[oem]
    g = m[m.vin == vin].sort_values("month").reset_index(drop=True)
    cur = float(g.soh.iloc[-1]); a0m = float(g["age_months"].iloc[-1])
    has_odo = "odo_max" in g.columns and bool((g["odo_max"] < 3e5).any())
    odo_now = float(g["odo_max"][g["odo_max"] < 3e5].max()) if has_odo else None
    reached_flag = (a0m / 12.0 >= wyr) or (odo_now is not None and odo_now >= wkm)   # did THIS vehicle reach warranty?
    reach_by = "km" if (odo_now is not None and odo_now >= wkm) else "age"
    # near-term degradation rate from the model, then a LINEAR projection to EoL — avoids the rate model's
    # asymptotic flattening (its predicted monthly loss decays to ~0 over a long horizon).
    mod, fmodel = forecaster(oem, deg_only)
    try:
        p50 = (np.asarray(mod.forecast(g, fmodel, 18)[0.5], dtype="float64") if oem == "Euler"
               else mod.simulate(g, fmodel, 18)["q50"].to_numpy())
    except Exception:
        p50 = np.array([cur])
    k = min(12, len(p50))
    model_rate = max((cur - float(p50[k - 1])) / k, 0.0) if k >= 1 else 0.0   # model near-term pp/month
    span_m = max(a0m - float(g["age_months"].iloc[0]), 1.0)                   # this vehicle's observed life (months)
    hist_rate = max((float(g["soh"].iloc[0]) - cur) / span_m, 0.0)           # ...and its observed degradation pace
    rate = max(model_rate, hist_rate)                                        # faster of the two -> healthy ones still project
    if rate > 0.01 and cur > eol:
        mte = (cur - eol) / rate
        npts = int(min(np.ceil((cur - (eol - 3)) / rate), 24))              # to ~3pp past EoL, capped at 2 yr
        f_soh = cur - rate * np.arange(1, npts + 1)
    else:
        mte, f_soh = None, np.array([])                                     # truly flat -> no projection
    kmpm = rul_km.km_per_month(g["age_months"], g["odo_max"]) if "odo_max" in g.columns else None
    if kmpm is not None and (kmpm <= 0 or kmpm > 8000):               # gate dirty/sparse odometers
        kmpm = None
    rem_km = int(round(kmpm * mte)) if (kmpm and mte) else None
    rated = float(RATED_KM[oem])                                       # one flat OEM rated -> range strictly = rated×SoH
    ntr = len(f_soh)
    f_odo = ((np.full(ntr, kmpm).cumsum() + odo_now).tolist() if (kmpm and odo_now and ntr) else None)
    gc = g[g["odo_max"] < 3e5] if "odo_max" in g.columns else g.iloc[0:0]   # clean-odo trajectory (km-based plot)
    return dict(vin=vin, cur=cur, eol=eol, mte=mte, kmpm=(int(round(kmpm)) if kmpm else None),
                rem_km=rem_km, rated=rated, range_now=rated * cur / 100, range_eol=rated * eol / 100,
                reached=reached_flag, reach_by=reach_by, age_yr=a0m / 12.0, odo=odo_now, wyr=wyr, wkm=wkm,
                a_hist=(g["age_months"].to_numpy() / 12.0).tolist(), soh_hist=smooth(g["soh"]).to_numpy().tolist(),
                f_age=((a0m + np.arange(1, ntr + 1)) / 12.0).tolist(), f_p50=f_soh.tolist(),
                odo_hist=gc["odo_max"].to_numpy().tolist(), odo_soh=smooth(gc["soh"]).to_numpy().tolist(), f_odo=f_odo)


_PAIRCOL = [TEAL, "#c792ff"]                                          # two-vehicle comparison colours


def _range_fig(oem, infos, h=230):
    rated, eol = infos[0]["rated"], infos[0]["eol"]
    xs = np.linspace(100, eol - 3, 40)
    fig = go.Figure()
    # OEM promised (rated) range — the headline number, constant; actual range = rated × SoH falls below it
    fig.add_hline(y=rated, line=dict(color=GREY, width=1.5, dash="dot"),
                  annotation_text=f"OEM promised {rated:.0f} km", annotation_position="top right",
                  annotation=dict(font=dict(color=GREY, size=10)))
    fig.add_scatter(x=xs, y=rated * xs / 100, mode="lines", line=dict(color="#3a6b7e", width=2), showlegend=False)
    fig.add_vline(x=eol, line=dict(color=AMBER, dash="dash"))
    for j, info in enumerate(infos):                                  # each vehicle's "range now" on the rated line
        fig.add_scatter(x=[info["cur"]], y=[info["range_now"]], mode="markers",
                        marker=dict(color=_PAIRCOL[j % 2], size=12), name=f"…{info['vin'][-6:]}")
    fig.update_xaxes(title="SoH %", autorange="reversed", **AX)        # 100% (new) on the left, aging to the right
    fig.update_yaxes(title="full-charge range (km)", range=[0, rated * 1.12], **AX)
    fig.update_layout(**lay(height=h, margin=dict(l=46, r=12, t=14, b=34),
                            legend=dict(orientation="h", y=-0.28, font=dict(size=10))))
    return fig


def _rul_soh_fig(oem, infos, h=230, on_km=None):
    """SoH for two vehicles overlaid: each measured (solid) + forecast (dashed) to EoL, with the warranty limit
    marked. X-axis: ODOMETER (km) when on_km (the same-distance comparison), else AGE / auto."""
    info0 = infos[0]; eol = info0["eol"]
    auto_km = info0.get("reach_by") == "km" and info0.get("odo_hist") and len(info0["odo_hist"]) >= 2
    use_km = auto_km if on_km is None else on_km
    if use_km and not (info0.get("odo_hist") and len(info0["odo_hist"]) >= 2):
        use_km = False                                                # no odo history -> fall back to age
    fig = go.Figure(); hi = 0.0
    for j, info in enumerate(infos):
        c = _PAIRCOL[j % 2]
        if use_km:
            x, y, fx, fy = info["odo_hist"], info["odo_soh"], info.get("f_odo"), info.get("f_p50")
        else:
            x, y, fx, fy = info["a_hist"], info["soh_hist"], info.get("f_age"), info.get("f_p50")
        if not x:
            continue
        fig.add_scatter(x=x, y=y, mode="lines+markers", line=dict(color=c, width=2), marker=dict(size=3),
                        name=f"…{info['vin'][-6:]}")
        if fx and fy and len(fx) == len(fy):                          # bridge from the last measured point
            fig.add_scatter(x=[x[-1]] + list(fx), y=[y[-1]] + list(fy), mode="lines",
                            line=dict(color=c, width=2, dash="dash"), showlegend=False)
        hi = max(hi, x[-1], (fx[-1] if fx else 0))
    wline = info0["wkm"] if use_km else info0["wyr"]
    wlabel = f"{info0['wkm']//1000}k-km warranty" if use_km else f"{info0['wyr']}-yr warranty"
    xtitle = "odometer (km)" if use_km else "age (years)"
    hi = max(hi, wline)
    fig.add_hline(y=eol, line=dict(color=AMBER, dash="dash"), annotation_text=f"{eol}% EoL",
                  annotation_position="bottom left")
    fig.add_vline(x=wline, line=dict(color=GREY, width=1.5, dash="dashdot"), annotation_text=wlabel,
                  annotation_position="top right", annotation=dict(font=dict(color=GREY, size=10)))
    xa = dict(title=xtitle, range=[0, hi * 1.05 if use_km else hi + 0.3], **AX)
    if not use_km:
        xa["dtick"] = 1
    fig.update_xaxes(**xa)
    fig.update_yaxes(title="SoH %", range=[eol - 6, 101], **AX)        # bound to the data band (not down to 55)
    fig.update_layout(**lay(height=h, margin=dict(l=44, r=12, t=10, b=34),
                            legend=dict(orientation="h", y=-0.28, font=dict(size=10))))
    return fig




# Four outcome groups for held-out test vehicles (used by _testgrid_fig).  Order = panel order.
TESTCAT = [("At-risk", RED), ("Safe", GREEN), ("Genuinely flat", TEAL), ("Flat (unproven)", AMBER)]
FLAT_DROP = 3.0      # < this much observed decline = "flat"
PROVEN_SPAN = 18.0   # ... and observed for >= this many months = "genuinely" flat (else "unproven")


def _bucketise(preds, eol):
    """Sort each test vehicle into one of the four outcome groups."""
    cats = {k: [] for k, _ in TESTCAT}
    for p in preds:
        soh = np.asarray(p["soh"], float); age = np.asarray(p["age"], float)
        drop = float(soh[0] - soh.min())                  # observed decline so far
        span = float(age[-1] - age[0])                    # how long we've actually watched it
        if _fc_at_warranty(p, "p50") < eol:
            cats["At-risk"].append(p)                     # forecast crosses EoL by warranty
        elif drop >= FLAT_DROP:
            cats["Safe"].append(p)                        # really declining, but projects to survive
        elif span >= PROVEN_SPAN:
            cats["Genuinely flat"].append(p)              # flat AND watched long enough to trust it
        else:
            cats["Flat (unproven)"].append(p)             # flat but young/short — could still decline
    return cats


def _testgrid_fig(oem, which="test"):
    """2x2 panel: one split's vehicles grouped by outcome (one panel per group, vehicles overlaid)."""
    preds = all_split_predictions(oem, DEG_ONLY).get(which, [])
    if not preds:
        return None, {}
    eol = EOL_PCT[oem]; a100 = RENORM100[oem]
    cats = _bucketise(preds, eol)
    titles = [f"{k} — {len(cats[k])}" for k, _ in TESTCAT]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.07)
    pos = [(1, 1), (1, 2), (2, 1), (2, 2)]; ymin = 101.0
    for idx, (k, color) in enumerate(TESTCAT):
        r, c = pos[idx]
        wx, wy = [], []                                                          # each vehicle's OWN deadline point
        for p in cats[k]:
            age = np.asarray(p["age"]) / 12.0; fage = np.asarray(p["fage"]) / 12.0
            sm = smooth(pd.Series(p["soh"]))
            ax = ([0.0] + age.tolist()) if a100 else age.tolist()
            sy = ([100.0] + sm.tolist()) if a100 else sm.tolist()
            ymin = min(ymin, min(sy), min(p["p50"]))
            fig.add_scatter(x=ax, y=sy, mode="lines", line=dict(color=color, width=0.8),
                            opacity=0.4, row=r, col=c, showlegend=False)          # measured
            fig.add_scatter(x=fage, y=p["p50"], mode="lines", line=dict(color=color, width=0.8, dash="dot"),
                            opacity=0.35, row=r, col=c, showlegend=False)          # forecast P50
            wx.append(p["warr_age"] / 12.0); wy.append(_fc_at_warranty(p, "p50"))  # this vehicle's warranty end
        if wx:                                                                    # ◆ = each vehicle's own deadline
            fig.add_scatter(x=wx, y=wy, mode="markers", row=r, col=c, showlegend=False, opacity=0.85,
                            marker=dict(symbol="diamond", size=5, color="#e8edf2",
                                        line=dict(color="#0e1726", width=0.6)))
        fig.add_hline(y=eol, line=dict(color=AMBER, width=1, dash="dot"), row=r, col=c)
    fig.update_yaxes(range=[max(min(ymin - 3, eol - 8), 40), 101], title_text="SoH %", **AX)
    fig.update_xaxes(title_text="age (years)", dtick=1, **AX)
    fig.update_annotations(font_size=12)
    fig.update_layout(**lay(height=640, showlegend=False, margin=dict(l=44, r=16, t=46, b=40)))
    return fig, {k: len(cats[k]) for k, _ in TESTCAT}


@st.cache_data(show_spinner=False)
def atrisk_compare(oem):
    """At-risk (n, count) per split for each training population (all / degraders / safe); splits held fixed."""
    eol = EOL_PCT[oem]; res = {}
    for mode in ("all", "deg", "safe"):
        preds = all_split_predictions(oem, mode); d = {}; tn = ta = 0
        for which in ("test", "validation", "train"):
            c = _bucketise(preds[which], eol)
            n = sum(len(v) for v in c.values()); a = len(c["At-risk"])
            d[which] = (n, a); tn += n; ta += a
        d["fleet"] = (tn, ta); res[mode] = d
    return res


def _atrisk_fig(oem):
    """Grouped bar: at-risk % per split for each training population (all / degraders / safe); splits fixed."""
    res = atrisk_compare(oem)
    order = [("test", "Test"), ("validation", "Validation"), ("train", "Train"), ("fleet", "Fleet (all)")]
    x = [lbl for _, lbl in order]
    pct = lambda md: [(100 * res[md][k][1] / res[md][k][0]) if res[md][k][0] else 0.0 for k, _ in order]
    txt = lambda md: [f"{res[md][k][1]}/{res[md][k][0]}<br>{100 * res[md][k][1] / res[md][k][0]:.0f}%"
                      if res[md][k][0] else "—" for k, _ in order]
    fig = go.Figure()
    for md, name, color in (("all", "All vehicles", TEAL), ("deg", "Degraders-only", AMBER),
                            ("safe", "Safe-only", GREEN)):
        fig.add_bar(x=x, y=pct(md), name=name, marker_color=color,
                    text=txt(md), textposition="outside", textfont=dict(size=10), cliponaxis=False)
    ymax = (max(pct("all") + pct("deg") + pct("safe")) or 1) * 1.3
    fig.update_layout(barmode="group", **lay(height=360, margin=dict(l=46, r=16, t=40, b=30)))
    fig.update_yaxes(title_text="At-risk %", range=[0, ymax], **AX)
    fig.update_xaxes(**AX)
    return fig


def _feature_grid_fig(oem):
    """Small-multiples: every engineered feature (plus SoH first) vs age, for the most-degraded vehicle."""
    F = FEATS_BY[oem]
    vin = ov(F).sort_values("s1").index[0]
    g = F[F.vin == vin].sort_values("age_months")
    ageyr = (g["age_months"] / 12).to_numpy()
    exclude = {"vin", "month", "soh", "soh_raw", "reg_known", "age_months"}
    feats = [c for c in g.columns if c not in exclude and pd.api.types.is_numeric_dtype(g[c])]
    panels = ["soh"] + feats
    ncols = 4; nrows = int(np.ceil(len(panels) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=panels,
                        vertical_spacing=max(0.04, 0.30 / nrows), horizontal_spacing=0.05)
    for i, name in enumerate(panels):
        r, c = i // ncols + 1, i % ncols + 1
        y = smooth(g["soh"]) if name == "soh" else g[name]
        fig.add_scatter(x=ageyr, y=y, mode="lines",
                        line=dict(color=TEAL if name == "soh" else GREY, width=1.6),
                        row=r, col=c, showlegend=False)
    fig.update_xaxes(dtick=1, **AX); fig.update_yaxes(**AX)
    for cc in range(1, ncols + 1):
        fig.update_xaxes(title_text="age (yr)", row=nrows, col=cc)
    fig.update_annotations(font_size=10)
    fig.update_layout(**lay(height=max(nrows * 150, 300), showlegend=False,
                            margin=dict(l=34, r=10, t=26, b=30)))
    return fig, vin, len(feats)


def _fc_at_warranty(p, q="p50"):
    """Forecast quantile value at (the point closest to) the vehicle's warranty deadline."""
    fage = np.asarray(p["fage"])
    return float(np.asarray(p[q])[int(np.argmin(np.abs(fage - p["warr_age"])))])


# ── validation-page analyses: held-out backtests + the data-sufficiency learning curve ──
@st.cache_data(show_spinner="Backtesting held-out vehicles…")
def _backtest_eval(oem_key, deg_only=False):
    """Train on the non-test vehicles; for each held-out TEST vehicle anchor at the MIDPOINT of its history
    and forecast the rest, comparing P50 to the actual later SoH. Returns per-vehicle records (measured +
    forecast + MAE + usage + decline) for the actual-vs-predicted and usage views."""
    cfg = OEMS[oem_key]; mod = importlib.import_module(cfg["module"]); euler = oem_key == "Euler"
    m = data_quality.apply_quality(load_cohort(oem_key), oem_key)
    gb = m.groupby("vin"); drop = gb["soh"].first() - gb["soh"].last()
    smin = gb["soh"].min(); aged = set(smin[smin <= EOL_PCT[oem_key]].index)
    TR, VA, TE = _split(list(m["vin"].unique()), drop, force_train=aged)
    tr_vins = _train_vins(TR | VA, drop, deg_only)
    model = (mod.train_traj(mod.build_traj_samples(m[m["vin"].isin(tr_vins)])) if euler
             else mod.train_quantiles(mod.build_transitions(m[m["vin"].isin(tr_vins)])))
    eol = EOL_PCT[oem_key]; recs = []
    te = sorted(TE)                                              # subsample for speed — a mean MAE / usage
    if len(te) > 120:                                           # distribution is reliable from ~120 vehicles
        te = [te[i] for i in np.random.RandomState(0).choice(len(te), 120, replace=False)]
    for vin in te:
        g = m[m["vin"] == vin].sort_values("month").reset_index(drop=True); n = len(g)
        if n < 6:
            continue
        cut = n // 2; hist = g.iloc[:cut + 1]; aa = float(hist["age_months"].iloc[-1])
        H = int(round(g["age_months"].iloc[-1] - aa))
        if H < 1:
            continue
        if euler:
            fc = mod.forecast(hist, model, H); p50 = np.asarray(fc[0.5])
        else:
            p50 = mod.simulate(hist, model, H)["q50"].to_numpy()
        errs = [abs(p50[d - 1] - float(r["soh"])) for _, r in g.iloc[cut + 1:].iterrows()
                for d in [int(round(r["age_months"] - aa))] if 1 <= d <= len(p50)]
        if not errs:
            continue
        last = float(hist["soh"].iloc[-1]); span = float(g["age_months"].iloc[-1] - g["age_months"].iloc[0])
        usage = float(pd.to_numeric(g["km_month"], errors="coerce").clip(upper=15000).median()) if "km_month" in g else np.nan
        recs.append(dict(vin=vin[-6:], n=n, span=span, cut_age=aa,
                         age=g["age_months"].tolist(), soh=g["soh"].tolist(),
                         fage=(aa + np.arange(0, H + 1)).tolist(), p50=[last] + p50[:H].tolist(),
                         mae=float(np.mean(errs)), usage=usage,
                         decline=float((g["soh"].iloc[0] - g["soh"].min()) / max(span, 1)),
                         reached_eol=bool(g["soh"].min() <= eol)))
    return recs


@st.cache_data(show_spinner="Measuring how much history the model needs…")
def _learning_curve(oem_key, deg_only=False):
    """Forecast accuracy vs months of history. FIXED eval set (same vehicles for every anchor length so the
    curve isn't confounded): for each anchor N (first N months) forecast the rest and average the MAE."""
    cfg = OEMS[oem_key]; mod = importlib.import_module(cfg["module"]); euler = oem_key == "Euler"
    m = data_quality.apply_quality(load_cohort(oem_key), oem_key)
    G = {v: g.sort_values("month").reset_index(drop=True) for v, g in m.groupby("vin")}
    rows = {v: len(G[v]) for v in G}
    rmax = int(np.percentile(list(rows.values()), 90))
    anchors = [a for a in (3, 4, 6, 8, 10, 12, 15, 18) if a <= rmax - 3]
    ev = [v for v in G if rows[v] >= (anchors[-1] + 3)] if anchors else []
    tr = [v for v in G if v not in set(ev)]
    if not anchors or not ev or len(tr) < 10:
        return [], [], 0
    n_ev = len(ev)
    if len(ev) > 80:                                            # subsample the eval set — a mean curve is stable
        ev = [ev[i] for i in np.random.RandomState(1).choice(len(ev), 80, replace=False)]
    model = (mod.train_traj(mod.build_traj_samples(m[m["vin"].isin(tr)])) if euler
             else mod.train_quantiles(mod.build_transitions(m[m["vin"].isin(tr)])))
    ns, maes = [], []
    for N in anchors:
        es = []
        for v in ev:
            g = G[v]; anc = g.iloc[:N]; aa = float(anc["age_months"].iloc[-1])
            H = int(round(g["age_months"].iloc[-1] - aa))
            if H < 1:
                continue
            p = (np.asarray(mod.forecast(anc, model, H)[0.5]) if euler else mod.simulate(anc, model, H)["q50"].to_numpy())
            for j in range(N, len(g)):
                d = int(round(g["age_months"].iloc[j] - aa))
                if 1 <= d <= len(p):
                    es.append(abs(p[d - 1] - float(g["soh"].iloc[j])))
        ns.append(N); maes.append(float(np.mean(es)) if es else float("nan"))
    return ns, maes, n_ev


@st.cache_data(show_spinner="Backtesting example vehicles…")
def _validation_demos(oem_key, deg_only=False):
    """Pick a few COMPLETE-history vehicles (≤2 at-risk + ≤2 safe), train a model that EXCLUDES them, then
    backtest each (anchor at the midpoint, forecast the rest) — an honest actual-vs-predicted on real journeys
    (incl. aged ones, which the held-out test set never has because aged vehicles are forced into training)."""
    cfg = OEMS[oem_key]; mod = importlib.import_module(cfg["module"]); euler = oem_key == "Euler"
    m = data_quality.apply_quality(load_cohort(oem_key), oem_key); eol = EOL_PCT[oem_key]
    o = m.groupby("vin").agg(nrow=("soh", "size"), smin=("soh", "min"), s0=("soh", "first"),
                             a0=("age_months", "first"), a1=("age_months", "last"))
    o["span"] = o["a1"] - o["a0"]; o["drop"] = o["s0"] - o["smin"]
    o = o[o["nrow"] >= 6]
    atr = list(o[(o["smin"] <= eol) | (o["drop"] >= 8)].sort_values(["span", "drop"], ascending=False).index[:2])
    safe = list(o[(o["smin"] > eol) & (o["drop"] < 5)].sort_values("span", ascending=False).index[:2])
    demos = atr + safe; labs = ["at-risk"] * len(atr) + ["safe"] * len(safe)
    if not demos:
        return []
    train = m[~m["vin"].isin(set(demos))]
    model = (mod.train_traj(mod.build_traj_samples(train)) if euler
             else mod.train_quantiles(mod.build_transitions(train)))
    recs = []
    for vin, lab in zip(demos, labs):
        g = m[m["vin"] == vin].sort_values("month").reset_index(drop=True); n = len(g)
        cut = n // 2; hist = g.iloc[:cut + 1]; aa = float(hist["age_months"].iloc[-1])
        H = int(round(g["age_months"].iloc[-1] - aa))
        if H < 1:
            continue
        p50 = np.asarray(mod.forecast(hist, model, H)[0.5]) if euler else mod.simulate(hist, model, H)["q50"].to_numpy()
        errs = [abs(p50[d - 1] - float(r["soh"])) for _, r in g.iloc[cut + 1:].iterrows()
                for d in [int(round(r["age_months"] - aa))] if 1 <= d <= len(p50)]
        last = float(hist["soh"].iloc[-1])
        recs.append(dict(vin=vin[-6:], lab=lab, cut_age=aa, age=g["age_months"].tolist(), soh=g["soh"].tolist(),
                         fage=(aa + np.arange(0, H + 1)).tolist(), p50=[last] + p50[:H].tolist(),
                         mae=float(np.mean(errs)) if errs else float("nan")))
    return recs


# ═════════════════════════════════ STEP 0 ═════════════════════════════════
if step == STEPS[0]:
    st.title("🔧 The battery-health prediction pipeline we built")
    st.markdown(
        "This walks through the **State-of-Health (SoH) prediction pipeline we built at Turno** — end to end, "
        "on a real problem: forecasting the **health of EV batteries** across **three fleets at once "
        "(Euler · Mahindra · Bajaj)**. Every stage shows all three side by side, so you can see how the *same* "
        "pipeline adapts to each OEM's data — **which sensors each feed provides, and what each model relies on.**")
    concept("Rather than hand-coding rules for how a battery ages, we **let the model learn the patterns from "
            "the fleet itself** — thousands of batteries aging over time — and use them to forecast how any "
            "given battery will age. That engine drives everything that follows.")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        F = FEATS_BY[oem]
        col.markdown(f"### {oem}")
        col.caption(OEMS[oem]["label"])
        col.metric("Vehicles tracked", F.vin.nunique())
        col.markdown(f"**SoH method:** {OEMS[oem]['soh_method']}")
    takeaway("Walk the stages left-to-right to follow the whole pipeline — from raw telemetry to a "
             "warranty-risk call. Where the three fleets differ is exactly where the interesting engineering "
             "decisions live.")

# ═════════════════════════════════ STEP 1 ═════════════════════════════════
elif step == STEPS[1]:
    st.title("1 · The problem — what are we predicting?")
    st.markdown("Every EV battery slowly **wears out**. We measure that wear as **State of Health (SoH)** — "
                "the percentage of the battery's *original* capacity that remains. Each fleet has its own "
                "**end-of-life threshold** we forecast it to cross:")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        col.markdown(f"### {oem}")
        col.metric("End of life", f"{EOL_PCT[oem]}% SoH")
        col.metric("Warranty term", f"{WARRANTY_YR[oem]} yr")
        col.caption(f"{FEATS_BY[oem].vin.nunique()} vehicles tracked")
    concept("Months until a vehicle crosses its end-of-life SoH = its **Remaining Useful Life (RUL)**. Note "
            "**Bajaj's line is 70%**, Euler/Mahindra **80%** — different chemistry and warranty expectations.")
    takeaway("We predict a number — **future SoH** — for each vehicle. Predicting a number is a "
             "**regression** problem. Same problem, three fleets, slightly different finish lines.")

# ═════════════════════════════════ STEP 2 ═════════════════════════════════
elif step == STEPS[2]:
    st.title("2 · The data — what the model sees")
    st.markdown("Each vehicle streams battery telemetry, summarised to **one row per vehicle per month**. "
                "But the three feeds **don't carry the same sensors** — this table is the heart of the "
                "comparison:")

    # ── fleet at a glance: the TRUE registered fleet (Redshift) vs how much of it we can actually model ──
    st.markdown("#### Fleet at a glance — the true fleet, and how much of it we actually model")
    _REGDEF = {"Euler": 2125, "Mahindra": 10933, "Bajaj": 1781, "Piaggio": 1913, "Montra": 481}  # offline fallback
    REG = {o: int(FLEET_COV.get(o, {}).get("fleet") or _REGDEF[o]) for o in _REGDEF}
    MODELED = {o: FEATS_BY[o].vin.nunique() for o in OEM_KEYS}          # the cohort the pipeline runs on
    OEMCOL = {"Euler": TEAL, "Mahindra": AMBER, "Bajaj": "#6f7fd6", "Piaggio": "#46566e", "Montra": "#333f4e"}
    _tot_reg = sum(REG.values()); _tot_mod = sum(MODELED[o] for o in OEM_KEYS); _covered = _tot_mod + NAT_FLEET
    mm = st.columns(4)
    mm[0].metric("Actual registered fleet", f"{_tot_reg:,}", "5 OEMs (Redshift)", delta_color="off")
    mm[1].metric("Modelled — measured SoH", f"{_tot_mod:,}", f"~{100*_tot_mod/_tot_reg:.0f}% · the working cohort", delta_color="off")
    mm[2].metric("+ Native Mahindra (predicted)", f"~{NAT_FLEET:,}", "behaviour prior — no measured SoH", delta_color="off")
    mm[3].metric("Total covered", f"~{_covered:,}", f"~{100*_covered/_tot_reg:.0f}% · measured + predicted", delta_color="off")
    st.caption(f"**Actual fleet {_tot_reg:,} vehicles** · we compute a **measured** SoH for **{_tot_mod:,}** of them "
               f"(the cohort every other page runs on), and add a **predicted** curve for the ~{NAT_FLEET:,} native-only "
               f"Mahindra that carry no SoH signal — **~{100*_covered/_tot_reg:.0f}% covered** one way or the other.")
    fc1, fc2 = st.columns([0.82, 1.18])                # narrower pie, wider (busier) reason-breakdown bar
    # pie — every registered vehicle across all 5 OEMs; the 3 we model are saturated, the 2 we don't are muted
    _po = list(_REGDEF)
    pie = go.Figure(go.Pie(labels=_po, values=[REG[o] for o in _po], hole=0.45, sort=False,
                           marker=dict(colors=[OEMCOL[o] for o in _po], line=dict(color="#0e1726", width=2)),
                           textinfo="label+percent", textfont_size=12))
    pie.update_layout(**lay(height=340, showlegend=False, margin=dict(l=10, r=10, t=46, b=10),
                            title=dict(text=f"Registered fleet — {sum(REG.values()):,} vehicles, 5 OEMs", font=dict(size=14))))
    fc1.plotly_chart(pie, use_container_width=True)
    fc1.caption("Source: Redshift `oem_fleet_kpis`. **We model 3 of the 5 OEMs** (coloured). **Mahindra alone is "
                "~64% of the fleet** — and it's where SoH is hardest to measure (next chart). **Piaggio (1,913)** "
                "and **Montra (481)** aren't modelled yet — Piaggio is well-instrumented and the strongest next add.")
    # bar — WHY each fleet is / isn't modelled, coloured by the DOMINANT reason. Mahindra is shown as its two
    # real feeds (intellicar has current → measurable; native has none → not measurable). The grey is split into
    # three honest reasons, not one: not-measurable (no signal) vs not-downloaded vs SoH-not-computed (thin data).
    MAH_IC = MODELED["Mahindra"]                        # intellicar-coulomb cohort (has current) = modelled Mahindra
    MAH_NAT = max(REG["Mahindra"] - MAH_IC, 0)          # the rest: native-only (no current) → not measurable, ever
    NAVY = "#5a6b8c"
    BARS = [("Euler",                 REG["Euler"],   {"mod": MODELED["Euler"],   "navy": REG["Euler"] - MODELED["Euler"]}),
            ("Mahindra · intellicar", MAH_IC,         {"mod": MAH_IC}),
            ("Mahindra · native",     MAH_NAT,        {"grey": MAH_NAT}),
            ("Bajaj",                 REG["Bajaj"],   {"mod": MODELED["Bajaj"],   "navy": REG["Bajaj"] - MODELED["Bajaj"]}),
            ("Piaggio",               REG["Piaggio"], {"mod": MODELED["Piaggio"], "amber": REG["Piaggio"] - MODELED["Piaggio"]})]
    CATS = [("mod",   "modelled (usable SoH)", TEAL),
            ("amber", "not downloaded yet (signal in S3)", AMBER),
            ("navy",  "SoH not computed (thin / older data)", NAVY),
            ("grey",  "not measurable (no current/voltage)", GREY)]
    xlab = [f"{n}<br>{t:,}" for n, t, _ in BARS]
    bar = go.Figure()
    for key, name, color in CATS:
        ys = [d.get(key, 0) for _, _, d in BARS]
        txt = [f"{100*y/t:.0f}%" if y else "" for y, (_, t, _) in zip(ys, BARS)]
        bar.add_bar(name=name, x=xlab, y=ys, marker_color=color, text=txt, textposition="inside", insidetextanchor="middle")
    bar.update_layout(barmode="stack", barnorm="percent", **lay(height=400, showlegend=True,
                            margin=dict(l=10, r=10, t=46, b=104),
                            legend=dict(orientation="h", y=-0.30, x=0.5, xanchor="center", font=dict(size=10), traceorder="normal"),
                            title=dict(text="Why each fleet is / isn't modelled (Mahindra split by feed)", font=dict(size=14))))
    bar.update_yaxes(title_text="% of fleet", ticksuffix="%", **AX); bar.update_xaxes(**AX)
    fc2.plotly_chart(bar, use_container_width=True)
    fc2.caption("Each bar = that fleet, split by **why** it is or isn't modelled. **Mahindra is shown as its two "
                "real fleets:** ~233 **intellicar** (have current → coulomb SoH, all modelled) and ~10,700 "
                "**native-only** (no current → *not measurable, ever*). Teal = usable SoH · **amber = the signal "
                "exists upstream, just not downloaded yet — Piaggio (~250 of 1,913 pulled)** · **navy = telemetry too "
                "thin / older-batch — Euler, Bajaj (matures over time)** · **grey = no electrical signal at all "
                "(native Mahindra)**. Coloured by the *dominant* reason per fleet.")
    concept("**The grey isn't one thing — it's three, and only one is permanent.** "
            "**① Not measurable (grey):** the ~10,700 **native-only Mahindra** have no current/voltage in their feed "
            "→ no SoH is possible *ever* without a richer signal — a **structural** limit, fixed only by new "
            "hardware/telemetry. **② Not yet downloaded (amber):** **Piaggio's** coulomb signal already exists in S3 "
            "(well-instrumented intellicar), but we've only pulled ~250 of 1,913 — a **pull** limit, fixed by "
            "downloading more. **③ SoH not computed (navy):** the remaining **Euler / Bajaj** have telemetry that's "
            "too thin, or an older batch lacking the right signal quality — a **data-maturity** limit that shrinks as "
            "the fleet ages. Only ① needs a new feed; ② and ③ are ours to close.")

    st.markdown("#### Which signals each fleet's feed provides")
    st.dataframe(availability_df(), hide_index=True, use_container_width=True)
    st.caption("**✅** = used as a model feature · **◦** = present in the raw feed but *not modelled* (weak / flaky for "
               "SoH — e.g. speed, GPS, drive-state) · **—** = absent. **Mahindra is split into its two feeds:** "
               "**intellicar** carries current/voltage → coulomb SoH (~233 vehicles); the **native** feed (~10,700) "
               "gives only age, SoC & odometer as usable features — it *also* streams speed / GPS / status, but those "
               "are too weak or flaky to model. **Cell imbalance** is a genuine degradation signal, but **only Euler's "
               "feed reports it**. **Bajaj** has no current/voltage (it reports SoH directly) and is the only feed with "
               "charge-cycle count & ambient temp.")
    concept("Two kinds of columns everywhere: **the target** (`soh`, the answer we predict) and "
            "**features** (the clues). What changes per fleet is *which clues exist*.")
    st.markdown("#### When does each fleet's data start — and who was captured *from new*?")
    st.markdown("A vehicle only truly starts at **~100% SoH** if telemetry begins near its registration. If the "
                "feed turned on *after* the vehicle was already in service, its first reading is already aged.")
    wcols = st.columns(len(OEM_KEYS))
    for col, oem in zip(wcols, OEM_KEYS):
        F = FEATS_BY[oem]; m = pd.to_datetime(F["month"]); fa = F.groupby("vin")["age_months"].min()
        n = len(fa); new = int((fa <= 3).sum())
        col.markdown(f"**{oem}** · data {m.min():%b %Y} → {m.max():%b %Y}")
        col.metric("Captured ~new (start ≈100% SoH)", f"{new} / {n}",
                   f"{100*new/n:.0f}% · median start age {fa.median():.0f} mo", delta_color="off")
    st.caption("⚠️ **Bajaj** data only starts **Sep 2025**, so every Bajaj vehicle was already ~16 months old at "
               "first telemetry — **0% captured new**. That's why Bajaj can't be anchored to 100% and we use its "
               "**absolute** BMS-reported SoH. Euler (from Oct 2023) and Mahindra (from Mar 2023) capture far more "
               "vehicles near-new, so their SoH is anchored to 100% at registration.")
    st.markdown("#### Every raw field each OEM sends — and whether it's usable")

    def _shortfeed(feed):
        if feed.startswith("Euler"):
            return "Euler"
        if feed.startswith("Mahindra") and "Intellicar" in feed:
            return "Mahindra·IC"
        if "Native" in feed:
            return "Mahindra·native"
        if feed.startswith("Bajaj"):
            return "Bajaj"
        if feed.startswith("Piaggio"):
            return "Piaggio"
        return feed[:12]
    _flat = []
    for feed, fields in OEM_FIELD_AUDIT.items():
        fs = _shortfeed(feed)
        for fld, what, use, dq in fields:
            u = "✅ usable" if use.startswith("✅") else ("⚠️ weak" if use.startswith("⚠️") else "❌ not usable")
            _flat.append({"Feed": fs, "Field": fld, "What it is": what, "Use for SoH / RUL": use, "Data quality": dq, "_u": u})
    _fdf = pd.DataFrame(_flat); _order = list(dict.fromkeys(_fdf.Feed))
    ufig = go.Figure()
    for lab, clr in [("✅ usable", GREEN), ("⚠️ weak", AMBER), ("❌ not usable", RED)]:
        ys = [int(((_fdf.Feed == f) & (_fdf._u == lab)).sum()) for f in _order]
        ufig.add_bar(name=lab, x=_order, y=ys, marker_color=clr, text=[y or "" for y in ys], textposition="inside")
    ufig.update_layout(barmode="stack", **lay(height=300, margin=dict(l=10, r=10, t=40, b=64),
                       legend=dict(orientation="h", y=-0.26, x=0.5, xanchor="center", font=dict(size=10)),
                       title=dict(text="Raw fields per feed, by usability", font=dict(size=13))))
    ufig.update_yaxes(title_text="# raw fields", **AX); ufig.update_xaxes(**AX)
    st.plotly_chart(ufig, use_container_width=True)
    st.caption("Every raw field each feed sends, counted by whether we can use it (**✅ usable · ⚠️ weak/caveats · "
               "❌ not usable**) — Euler & the intellicar feeds are field-rich; native Mahindra & Bajaj are leaner. "
               "Per-field detail below (also `docs/oem_fields_one_pager.md`):")
    _pick = st.selectbox("Filter by feed", ["All feeds"] + _order, key="fieldaudit")
    _show = _fdf if _pick == "All feeds" else _fdf[_fdf.Feed == _pick]
    st.dataframe(_show.drop(columns="_u").reset_index(drop=True), hide_index=True, use_container_width=True)
    st.caption("**Native Mahindra proxy-SoH coverage (95 vehicles)** — *computable* but **none trustworthy**: "
               "distanceToEmpty **84** (corr ≈ −0.20) · distance-per-SoC **25** (≈ +0.22) · charge-energy/`kwh` **8** "
               "(≈ +0.32) → the native tier falls back to an **age prior**, not a proxy.")
    takeaway("ML needs **examples** — one per row. A model can only use clues its feed actually carries, so "
             "the three models necessarily lean on different features (Step 8 shows how differently).")

# ═════════════════════════════════ STEP 3 ═════════════════════════════════
elif step == STEPS[3]:
    st.title("3 · The target — State of Health over time")
    st.markdown("Each line is one vehicle's SoH as it ages. We split the two populations the model must "
                "handle into separate rows: **degraders** (lost ≥2% — real aging) and **still-near-new** "
                "(flat) vehicles.")
    st.markdown("##### 🔴 Degraders — what real aging looks like")
    _ncol = len(OEM_KEYS) + (1 if NATIVE_SOH is not None else 0)
    cols = st.columns(_ncol)
    for col, oem in zip(cols[:len(OEM_KEYS)], OEM_KEYS):
        o = ov(FEATS_BY[oem]); eol = EOL_PCT[oem]
        col.markdown(f"**{oem}** · _{OEMS[oem]['soh_method']}_ · {int((o.s0-o.s1>=2).sum())} degraders")
        col.plotly_chart(_soh_fig(oem, which="deg"), use_container_width=True)
        col.caption(f"reached {eol}%: {int((o.smin<=eol).sum())} · median history {int(o.months.median())} mo")
    if NATIVE_SOH is not None:
        col = cols[len(OEM_KEYS)]; o = ov(NATIVE_SOH); rise = int((o.s0 - o.s1 <= -2).sum())
        _clean = 0                                        # how many are a CLEAN monotonic decline (no up-bump)?
        for _v, _g in NATIVE_SOH.sort_values("month").groupby("vin"):
            _s = _g["soh"].values
            if len(_s) >= 6 and (_s[0] - _s[-1]) >= 2 and np.diff(_s).max() <= 1.0:
                _clean += 1
        col.markdown(f"**Mahindra Native** · _distance-per-SoC (native-only)_ · {int((o.s0-o.s1>=2).sum())} 'degraders'")
        col.plotly_chart(_native_soh_fig(which="deg"), use_container_width=True)
        col.caption(f"**Separate OEM — native feed only, never mixed with intellicar.** ⚠️ **Noise, not aging:** "
                    f"**{rise}** vehicles' proxy went *up* ≥2pp (~coin-flip), and **{_clean} of {NATIVE_SOH.vin.nunique()}** "
                    "is a clean monotonic decline (no up-bumps or cliffs) — every curve wobbles both ways. No current → "
                    "no true SoH.")
    st.markdown("##### ⚪ Still near-new — flat (lost <2%)")
    cols2 = st.columns(_ncol)
    for col, oem in zip(cols2[:len(OEM_KEYS)], OEM_KEYS):
        o = ov(FEATS_BY[oem])
        col.markdown(f"**{oem}** · {int((o.s0-o.s1<2).sum())} flat vehicles")
        col.plotly_chart(_soh_fig(oem, which="flat"), use_container_width=True)
    if NATIVE_SOH is not None:
        col = cols2[len(OEM_KEYS)]; o = ov(NATIVE_SOH)
        col.markdown(f"**Mahindra Native** · {int((o.s0-o.s1<2).sum())} flat")
        col.plotly_chart(_native_soh_fig(which="flat"), use_container_width=True)
        col.caption("**Its own OEM** — the ~12k native-only Mahindra fleet, distance-per-SoC on native data, "
                    "**never combined with the intellicar coulomb Mahindra**. Wobbles both ways: no clean aging signal.")
    concept("How SoH is *measured* differs per fleet — Euler reads BMS remaining-capacity, Mahindra "
            "coulomb-counts current, Bajaj trusts the BMS-reported value. (Euler/Mahindra are anchored to "
            "100% at registration; Bajaj uses the absolute reported value, so its lines start below 100%.)")
    takeaway("Most vehicles sit in the **grey (flat) row** — still healthy. The **degraders** are fewer but "
             "the most valuable examples: they show the model what real aging looks like.")

# ═════════════════════════════════ STEP 4 ═════════════════════════════════
elif step == STEPS[5]:
    st.title("5 · Features — turning raw data into clues")
    st.markdown("A **feature** is one clue we give the model. We *engineer* features that capture known "
                "battery-aging factors — but **a fleet can only offer features its feed supports.**")
    groups = {
        "Euler": "✅ age · heat · **current / Ah throughput** · **voltage** · SoC habits · `inv_sqrt_age` "
                 "curvature\n\n_rich electrical signals; no ambient temp / cycle count_",
        "Mahindra": "✅ age · heat · **current / Ah** · **voltage** · SoC & DoD · mileage · GPS · curvature"
                    "\n\n_the richest feed (coulomb + location)_",
        "Bajaj": "✅ age · heat · **ambient temp** · SoC habits · **charge cycles** · mileage · curvature"
                 "\n\n❌ **no current, no voltage** — must age-predict without electrical stress signals",
        "Piaggio": "✅ age · **current / Ah throughput** · **voltage** · SoC habits · charge cycles · mileage · "
                   "curvature\n\n_coulomb (intellicar), like Mahindra; motor-temp proxy only, no pack temp_",
        "Mahindra Native": "✅ age · **km / month** · SoC habits (mean, time-at-low) · odometer\n\n❌ **no current, "
                           "no voltage, no SoH** — behaviour features only. SoH is *predicted* from these (Step 15), "
                           "not measured; **km/month is the one credible ageing driver.**",
    }
    _fk = OEM_KEYS + (["Mahindra Native"] if NAT_BAYES is not None else [])
    cols = st.columns(len(_fk))
    for col, oem in zip(cols, _fk):
        col.markdown(f"### {oem}"); col.markdown(groups[oem])
    concept("**Feature engineering** = turning raw data into meaningful clues. `inv_sqrt_age` encodes the "
            "known fact that batteries fade fast early, then level off. The key cross-fleet point: Bajaj "
            "must predict aging **without** the electrical signals Euler/Mahindra rely on.")
    st.markdown("#### Every feature vs age — for the most-degraded vehicle in each fleet")
    st.caption("Each small panel is one clue the model sees, over the vehicle's life (SoH first, in teal). "
               "Pick a fleet's tab — the *set* of panels differs because the feeds differ.")
    tabs = st.tabs(OEM_KEYS)
    for tab, oem in zip(tabs, OEM_KEYS):
        with tab:
            fig, vin, nf = _feature_grid_fig(oem)
            st.caption(f"**{oem}** · vehicle {vin[-6:]} · {nf} features")
            st.plotly_chart(fig, use_container_width=True)
    takeaway("Each model receives the features its feed allows and learns which combinations predict SoH "
             "loss. Step 8 reveals *which* it actually relied on — and it differs per fleet.")

# ═════════════════════════════════ STEP 5 ═════════════════════════════════
elif step == STEPS[6]:
    st.title("6 · Splitting the data — train / validation / test")
    st.markdown("A rule we hold to: **never judge a model on data it learned from.** For each fleet we "
                "split the *vehicles* (never rows) into three groups — 🟢 train · 🟡 validation · 🔴 test — "
                "stratified so each keeps degraders and flat vehicles. **Columns = fleets, rows = the three "
                "groups** (each shown on its own so you can see who's in it):")
    cols = st.columns(len(OEM_KEYS))
    aged_rows = []; risk_by_oem = {}
    for col, oem in zip(cols, OEM_KEYS):
        d = diagnostics(oem, DEG_ONLY); s = d["sizes"]; tr = d["splits"]["train"]
        col.markdown(f"**{oem}** — 🟢 {s['train']} · 🟡 {s['validation']} · 🔴 {s['test']}")
        F = FEATS_BY[oem]; eol = EOL_PCT[oem]; wm = WARRANTY_YR[oem] * 12
        o = ov(F); aged_vins = [v for v in tr if v in o.index and o.loc[v, "smin"] <= eol]
        risk = []                                    # crossed EoL BEFORE its warranty deadline = early failure
        for v in aged_vins:
            below = F[(F.vin == v) & (F.soh <= eol)].sort_values("age_months")
            if len(below) and float(below.age_months.iloc[0]) < wm:
                risk.append(v)
        risk_by_oem[oem] = risk
        aged_rows.append({"Fleet": oem, "Train vehicles": s["train"],
                          "Completely-aged": len(aged_vins),
                          "↳ At-risk (failed in warranty)": len(risk),
                          "↳ Graceful (post-warranty)": len(aged_vins) - len(risk)})
    splitnames = ["train", "validation", "test"]
    rowlab = {"train": "🟢 Training", "validation": "🟡 Validation", "test": "🔴 Test"}
    colmap = {"train": GREEN, "validation": AMBER, "test": RED}
    fig = make_subplots(rows=3, cols=len(OEM_KEYS), column_titles=list(OEM_KEYS),
                        row_titles=[rowlab[k] for k in splitnames],
                        vertical_spacing=0.05, horizontal_spacing=0.05)
    for ci, oem in enumerate(OEM_KEYS, start=1):
        sp = diagnostics(oem, DEG_ONLY)["splits"]; F = FEATS_BY[oem]
        for ri, key in enumerate(splitnames, start=1):
            for vin in sp[key]:
                gg = F[F.vin == vin].sort_values("age_months")
                a100 = RENORM100[oem]
                ax = ([0.0] if a100 else []) + (gg.age_months / 12).tolist()
                sy = ([100.0] if a100 else []) + smooth(gg.soh).tolist()
                fig.add_scatter(x=ax, y=sy, mode="lines", line=dict(color=colmap[key], width=1),
                                opacity=0.5, row=ri, col=ci, showlegend=False)
            fig.add_hline(y=EOL_PCT[oem], line=dict(color=AMBER, width=1, dash="dot"), row=ri, col=ci)
    fig.update_yaxes(range=[55, 101], **AX); fig.update_xaxes(dtick=1, **AX)
    for ci in range(1, len(OEM_KEYS) + 1):
        fig.update_xaxes(title_text="age (years)", row=3, col=ci)
    fig.update_annotations(font_size=12)
    fig.update_layout(**lay(height=640, showlegend=False, margin=dict(l=42, r=44, t=44, b=40)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### 🔵 Completely-aged vehicles are forced into training — but what *kind* of aging is it?")
    tot = {"Fleet": "**All fleets**",
           "Train vehicles": sum(r["Train vehicles"] for r in aged_rows),
           "Completely-aged": sum(r["Completely-aged"] for r in aged_rows),
           "↳ At-risk (failed in warranty)": sum(r["↳ At-risk (failed in warranty)"] for r in aged_rows),
           "↳ Graceful (post-warranty)": sum(r["↳ Graceful (post-warranty)"] for r in aged_rows)}
    st.table(pd.DataFrame(aged_rows + [tot]).set_index("Fleet"))
    st.error("⚠️ **All aged examples are early failures — there are 0 graceful-aging examples.** Every "
             "completely-aged vehicle crossed EoL *inside* its warranty window (Euler from 11 mo, Mahindra "
             "from 15 mo), so the only end-of-life behaviour the model ever sees is the **failure tail** — "
             "biasing it toward over-predicting risk. **Bajaj has 0 aged at all**, so its long-horizon "
             "forecast is pure *extrapolation* until real aged Bajaj exist. We also can't validate the "
             "'98% survive' assumption from data — no vehicle has aged gracefully to EoL. Honest aged "
             "accuracy now comes from **Step 10 (LOVO)**, not the held-out test set.")

    # ── every vehicle per fleet: the healthy mass (grey) vs the completely-aged early-failures (red) ──
    st.markdown("##### 🔎 *Every* vehicle's SoH trajectory — the healthy fleet (grey) vs the early-failures (red)")
    aged_set = {o: set(risk_by_oem.get(o, [])) for o in OEM_KEYS}
    gated = {o: data_quality.apply_quality(FEATS_BY[o], o) for o in OEM_KEYS}
    ymin = 101.0
    fig2 = make_subplots(rows=1, cols=len(OEM_KEYS), horizontal_spacing=0.06, subplot_titles=[
        f"{o} — {gated[o].vin.nunique()} vehicles · {len(aged_set[o])} reached EoL" for o in OEM_KEYS])
    for ci, oem in enumerate(OEM_KEYS, start=1):
        Fg = gated[oem]; eol = EOL_PCT[oem]; a100 = RENORM100[oem]
        for is_aged, vins in ((False, [v for v in Fg.vin.unique() if v not in aged_set[oem]]),
                              (True, list(aged_set[oem]))):       # grey first, red on top
            for v in vins:
                gg = Fg[Fg.vin == v].sort_values("age_months")
                ax = ([0.0] if a100 else []) + (gg.age_months / 12).tolist()
                sy = ([100.0] if a100 else []) + gg.soh.tolist()
                ymin = min(ymin, min(sy))
                fig2.add_scatter(x=ax, y=sy, mode="lines", row=1, col=ci, showlegend=False,
                                 line=dict(color=(RED if is_aged else GREY), width=(1.4 if is_aged else 0.7)),
                                 opacity=(0.8 if is_aged else 0.16))
        fig2.add_hline(y=eol, line=dict(color=AMBER, dash="dash"), row=1, col=ci,
                       annotation_text=f"{eol}% EoL", annotation_font_size=10)
        fig2.add_vline(x=WARRANTY_YR[oem], line=dict(color=GREEN, dash="dot"), row=1, col=ci,
                       annotation_text=f"{WARRANTY_YR[oem]}-yr warr", annotation_font_size=10)
        fig2.update_xaxes(title_text="age (years)", row=1, col=ci, **AX)
    fig2.update_yaxes(range=[max(ymin - 3, 55), 101], title_text="SoH %", **AX)
    fig2.update_annotations(font_size=12)
    fig2.update_layout(**lay(height=390, showlegend=False, margin=dict(l=46, r=16, t=48, b=42)))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Every line is one vehicle in the fleet cohort. **Grey = the healthy majority** — they ride "
               "near the top, mostly above the amber EoL line. **Red = the completely-aged early-failures** — "
               "they dive below EoL well to the *left* of the green warranty line. Note **Bajaj's entire "
               "fleet stays above its 70% EoL (0 red)**: nothing has reached end-of-life yet, so its "
               "long-horizon forecast is pure extrapolation. (Some red dives are also partly SoH-pipeline "
               "artifacts — frozen values / recalibration cliffs — see the data-quality note.)")

    # ── curated training set: robust √t-smoothed buckets (the "good training data" selection to confirm) ──
    st.markdown("---")
    st.markdown("##### 🎯 Curated training set — robust √t-smoothed buckets *(confirm before retraining)*")
    st.caption("Each vehicle's SoH is robustly **√t-smoothed** (Theil–Sen, ignores cliffs/stuck values) and "
               "**projected to its warranty deadline**. 🟢 **graceful-aged** = aged/near-warranty that projects "
               "to *survive*; ⚪ **genuine-flat**; 🟦 **probable out-of-risk** (young, projects safe); 🔴 "
               "**at-risk** (projects < EoL even after cleaning); 🚫 **thin** (too little data to fit a trend).")
    WARR_MO = {o: WARRANTY_YR[o] * 12 for o in OEM_KEYS}
    cur = {o: training_curation.curate(data_quality.apply_quality(FEATS_BY[o], o), EOL_PCT[o], WARR_MO[o])
           for o in OEM_KEYS}
    BMAP = [("GRACEFUL", "🟢 Graceful-aged", GREEN), ("FLAT", "⚪ Genuine-flat", GREY),
            ("PROBABLE_OOR", "🟦 Probable-OOR", TEAL), ("AT_RISK", "🔴 At-risk", RED),
            ("EXCLUDED", "🚫 Thin", None)]
    crows = []
    for o in OEM_KEYS:
        vc = cur[o].bucket.value_counts()
        crows.append({"Fleet": o, **{lbl: int(vc.get(k, 0)) for k, lbl, _ in BMAP},
                      "✅ Good total": int(cur[o].bucket.isin(training_curation.GOOD).sum())})
    st.table(pd.DataFrame(crows).set_index("Fleet"))
    cmin = 101.0
    cfig = make_subplots(rows=1, cols=len(OEM_KEYS), horizontal_spacing=0.06, subplot_titles=list(OEM_KEYS))
    for ci, o in enumerate(OEM_KEYS, start=1):
        Fg = data_quality.apply_quality(FEATS_BY[o], o); eol = EOL_PCT[o]; a100 = RENORM100[o]; warr = WARR_MO[o]
        bk = cur[o].set_index("vin")
        for k, _, color in BMAP:
            if color is None:
                continue
            for v in bk[bk.bucket == k].index:
                gg = Fg[Fg.vin == v].sort_values("age_months")
                ax = ([0.0] if a100 else []) + (gg.age_months / 12).tolist()
                sy = ([100.0] if a100 else []) + gg.soh.tolist()
                cmin = min(cmin, min(sy))
                cfig.add_scatter(x=ax, y=sy, mode="lines", line=dict(color=color, width=1.0),
                                 opacity=0.5, row=1, col=ci, showlegend=False)
                proj, smn = bk.loc[v, "proj"], bk.loc[v, "sm_now"]   # √t projection from smoothed-now, FORWARD only
                if pd.notna(proj) and warr / 12 > ax[-1] + 1e-9:     # skip vehicles already past warranty age
                    cmin = min(cmin, float(proj))
                    cfig.add_scatter(x=[ax[-1], warr / 12], y=[float(smn), float(proj)], mode="lines",
                                     line=dict(color=color, width=0.7, dash="dot"), opacity=0.3,
                                     row=1, col=ci, showlegend=False)
        cfig.add_hline(y=eol, line=dict(color=AMBER, dash="dash"), row=1, col=ci,
                       annotation_text=f"{eol}% EoL", annotation_font_size=10)
        cfig.add_vline(x=warr / 12, line=dict(color="#7f8ea3", dash="dot"), row=1, col=ci,
                       annotation_text=f"{WARRANTY_YR[o]}-yr warr", annotation_font_size=10)
        cfig.update_xaxes(title_text="age (years)", row=1, col=ci, **AX)
    cfig.update_yaxes(range=[max(cmin - 3, 55), 101], title_text="SoH %", **AX)
    cfig.update_annotations(font_size=12)
    cfig.update_layout(**lay(height=400, showlegend=False, margin=dict(l=46, r=16, t=46, b=42)))
    st.plotly_chart(cfig, use_container_width=True)
    st.caption("Solid = observed, dotted = √t projection to warranty. Confirm: 🟢 glide above EoL near the "
               "warranty line, 🔴 dip below, ⚪ stay flat. Rules in `src/training_curation.py` "
               "(graceful = age ≥60% warranty or reached EoL · projects ≥EoL · ≥3pp decline). "
               "✅ **Euler now uses the corrected 3-yr / 36-mo warranty** (the old 5-yr was borrowed from the "
               "passenger HiCity/NEO model; our cohort is 100% HiLoad cargo) — this cut Euler at-risk **22→9** "
               "and lifted graceful-aged **21→39**. Still PROVISIONAL: unverified pending an official Euler doc.")
    concept("**Train** = the vehicles the model learns from · **validation** = a held-out set we use to tune "
            "it · **test** = vehicles it has never seen, scored only at the end. Only the test number is an "
            "honest estimate of real-world accuracy.")
    st.warning("⚠️ **We split by *whole vehicle*, never by row.** If two months of the *same* battery were "
               "in both train and test, the model could 'peek' at that battery's future — cheating, called "
               "**data leakage**.")
    takeaway("Train to learn, validate to tune, test to judge — on *different* vehicles each, for every "
             "fleet. That's how we get an *honest* accuracy estimate.")

# ═════════════════════════════════ STEP 6 ═════════════════════════════════
elif step == STEPS[7]:
    st.title("7 · Training the model — how it learns")
    st.markdown("All three fleets use the same model family — **gradient-boosted decision trees** — but one "
                "model trained per fleet. A **single decision tree** asks yes/no questions to reach a guess:")
    st.code("if temp_max > 38°C:\n    if age_months > 24:  ->  predict 'loses 0.4% this month'\n"
            "    else:                ->  predict 'loses 0.2% this month'\nelse:                    ->  "
            "predict 'loses 0.1% this month'", language="text")
    concept("**Gradient boosting** = build *hundreds* of small trees in sequence. Each new tree focuses on "
            "fixing the **mistakes** of the ones before it. Added together, they capture subtle, combined "
            "patterns no single rule could.")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        col.markdown(f"**{oem}**"); col.markdown(OEMS[oem]["model_desc"])
    st.markdown("Why trees and not a neural network / LSTM? We have only a **few hundred** batteries with "
                "short histories — *small* data. Gradient-boosted trees are the most reliable choice there; "
                "we even tested foundation time-series models (Chronos, TimesFM) and **this won.**")
    takeaway("'Training' = the computer tunes these trees until its predictions on the **training set** "
             "match reality as closely as possible. Next: did it learn real patterns, or just memorise?")

# ═════════════════════════════════ STEP 7 ═════════════════════════════════
elif step == STEPS[8]:
    st.title("8 · Which clues mattered most? — Feature importance")
    st.markdown("After training, each model reports **how much it leaned on each feature** (bigger bar = "
                "used more). Compare the three — they're **not the same**, because their feeds differ. The "
                "line under each chart flags **thinly-populated features** — a clue can rank low simply "
                "because it's mostly *missing data*, not because it doesn't matter.")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        col.markdown(f"**{oem}**"); col.plotly_chart(_fi_fig(oem), use_container_width=True)
        # data coverage of the model's features — surfaces e.g. Mahindra temp_max being ~7% populated
        F = FEATS_BY[oem]; mfeats = importlib.import_module(OEMS[oem]["module"]).FEATS
        cov = {f: float(F[f].notna().mean()) for f in mfeats if f in F.columns}
        thin = sorted(((f, c) for f, c in cov.items() if c < 0.5), key=lambda x: x[1])
        if thin:
            col.caption("📉 **thin features** (mostly-missing data → bar is unreliable): "
                        + " · ".join(f"`{f}` {c:.0%}" for f, c in thin))
        else:
            col.caption(f"📊 all {len(cov)} model features ≥50% populated "
                        f"(median {100 * np.median(list(cov.values())):.0f}%).")
    concept("**Feature importance** opens the 'black box': which inputs drove predictions. Euler/Mahindra "
            "can lean on electrical & age signals; Bajaj — lacking current/voltage — leans on age, "
            "temperature, charge cycles and mileage. The *available* features shape what each model can use.")
    st.warning("⚠️ **Important ≠ causes aging.** Usage clues often rank high but point the *wrong* way — "
               "heavily-degraded batteries are also older. Heat and calendar age are the trustworthy real "
               "drivers.")
    takeaway("This page is the payoff of the comparison: feed availability (Step 2) → which clues each "
             "model can actually use here. Pair importance with domain knowledge before claiming 'X causes "
             "aging'.")

# ═════════════════════════════════ STEP 8 ═════════════════════════════════
elif step == STEPS[9]:
    st.title("9 · Is it any good? — Errors & overfitting")
    st.markdown("We measure error as **RMSE** — roughly the typical size of the model's mistake (in "
                "percentage-points of monthly SoH loss). **Lower is better.** Checked on all three splits, "
                "per fleet — the Test ÷ Train gap reveals overfitting:")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        e = diagnostics(oem, DEG_ONLY)["errors"]; gap = e["test"] / max(e["train"], 1e-9)
        col.markdown(f"**{oem}** — test **{e['test']:.2f}** · gap {gap:.1f}×")
        col.plotly_chart(_err_fig(oem), use_container_width=True)
    concept("**Training error is always optimistically low** — the model has seen that data. The **test "
            "error** is the honest one. A tiny train error with a large test error means the model "
            "**memorised** the training vehicles instead of learning general patterns — what we call "
            "**overfitting**.")
    takeaway("A modest train→test gap is normal across all three fleets. Always quote the "
             "**test/validation** number, never the training one.")

# ═════════════════════════════════ STEP 9 ═════════════════════════════════
elif step == STEPS[10]:
    st.title("10 · A tougher, fairer test — Leave-One-Vehicle-Out")
    st.markdown("A single split can be lucky or unlucky. So we use **Leave-One-Vehicle-Out (LOVO)**: hold "
                "out one whole vehicle, train on the rest, forecast it — repeat for *every* vehicle. We "
                "compare each fleet's model against two lazy baselines:")
    st.markdown("- **Persistence:** 'assume SoH stays exactly where it is.'\n"
                "- **Trend line:** 'fit a simple curve and extend it.'\n\n"
                "Each bar below is **forecast error** — RMSE in SoH percentage-points, on the batteries that "
                "actually decline. **Lower is better.**")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        L = OEMS[oem]["lovo"]
        col.markdown(f"**{oem}** — model **{L['model']}** vs persist {L['persist']}")
        col.plotly_chart(_lovo_fig(oem), use_container_width=True)
        col.caption(f"{(1-L['model']/L['persist'])*100:.0f}% better than persistence on decliners")
    concept("LOVO asks the real question: given a battery's *early* history, can we forecast its *future* "
            "SoH? **Bajaj** shows the biggest win over persistence (steady, fast, clean decline); "
            "Euler/Mahindra are harder (noisier, slower aging) but still beat the baselines on decliners.")
    takeaway("Each model beats both lazy baselines on the batteries that actually decline — the ones that "
             "matter for warranty & RUL. That's the evidence it learned something real.")

# ═════════════════════════════════ STEP 10 ═════════════════════════════════
elif step == STEPS[11]:
    st.title("11 · Predicting the future — with honest uncertainty")
    st.markdown("The payoff: feed a battery's history to the trained model and it projects SoH **forward** — "
                "not as one over-confident number, but as a **range**. One clear example per fleet:")
    _fc_nc = 1 if NAT_BAYES is not None else 0
    cols = st.columns(len(OEM_KEYS) + _fc_nc)
    for col, oem in zip(cols[:len(OEM_KEYS)], OEM_KEYS):
        try:
            fig, vin, hmon, wyr = _forecast_fig(oem)
            col.markdown(f"**{oem}** · {vin[-6:]}")
            col.plotly_chart(fig, use_container_width=True)
            col.caption(f"predicted to end-of-life · band ≈ 80% range · amber {EOL_PCT[oem]}% EoL · "
                        f"dash-dot {wyr}-yr warranty")
        except Exception as ex:
            col.warning(f"{oem}: demo unavailable ({ex}).")
    if _fc_nc:
        col = cols[len(OEM_KEYS)]; nr = NAT_REPORT["native"]
        col.markdown("**Mahindra Native** · _predicted_")
        col.plotly_chart(_native_bayes_fig(), use_container_width=True)
        col.caption(f"**No measured SoH** — a Bayesian *behaviour* forecast for the ~{NAT_FLEET:,} native-only fleet "
                    f"(the other 4 forecast from a measured history; this one has none). Median ≈"
                    f"{nr['soh50_at_36mo_median']:.0f}% @3yr, ±{nr['band_width_at_36mo_median']/2:.0f}pp band · full model → Step 15.")
    concept("The dashed line is the **best single estimate**; the shaded band says 'we're ~80% sure the "
            "truth lands in here'. Honest uncertainty is essential — a confident-but-wrong forecast is "
            "dangerous for warranty decisions.")
    takeaway("Where each forecast crosses its end-of-life line gives that vehicle its **Remaining Useful "
             "Life**. This is exactly what powers the main SoH dashboard.")

    st.markdown("---")
    st.markdown("### 📋 Forecast from today — by split, grouped by outcome")
    st.caption("For **train**, **validation** and **test** vehicles we forecast SoH from each vehicle's latest "
               "data point to its warranty deadline, then sort it into **four outcome groups**. Each panel "
               "overlays that group's vehicles — **solid = measured, dotted = forecast P50**; amber = EoL line; "
               "each **◆ marks that vehicle's OWN warranty deadline** (whichever of its time term or km limit it "
               "reaches first). ◆ above EoL = safe, below = at-risk. Validation & test are held out; train is the "
               "in-sample set the model learned from.")
    SPLIT_SECTIONS = [("test", "🔴 Test vehicles — held out (final check)", True),
                      ("validation", "🟡 Validation vehicles — held out (tuning)", False),
                      ("train", "🟢 Train vehicles — in-sample (the model learned from these)", False)]
    tabs = st.tabs(OEM_KEYS)
    for tab, oem in zip(tabs, OEM_KEYS):
        with tab:
            eol = EOL_PCT[oem]
            st.markdown("**At-risk vehicles — effect of the *training population*** "
                        "(held-out vehicles are identical across all bars; only the training set changes):")
            try:
                st.plotly_chart(_atrisk_fig(oem), use_container_width=True, key=f"arcmp_{oem}")
            except Exception as ex:
                st.caption(f"(at-risk comparison unavailable: {ex})")
            for which, label, is_open in SPLIT_SECTIONS:
                with st.expander(label, expanded=is_open):
                    try:
                        fig, counts = _testgrid_fig(oem, which)
                        if not fig:
                            st.info(f"No {which} vehicles with enough history to forecast.")
                            continue
                        n = sum(counts.values())
                        cc = st.columns(5)
                        cc[0].metric(f"{oem} {which} vehicles", n)
                        cc[1].metric("🔴 At-risk", counts["At-risk"])
                        cc[2].metric("🟢 Safe", counts["Safe"])
                        cc[3].metric("🟦 Genuinely flat", counts["Genuinely flat"])
                        cc[4].metric("🟡 Flat (unproven)", counts["Flat (unproven)"])
                        st.plotly_chart(fig, use_container_width=True, key=f"grid_{oem}_{which}")
                        if which == "train":
                            st.caption("⚠️ In-sample — these vehicles trained the model, so their fit flatters it; "
                                       "shown for completeness, not as evidence of accuracy.")
                    except Exception as ex:
                        st.warning(f"{which.title()} grid unavailable ({ex}).")
            st.caption(f"**At-risk** = P50 forecast below the {eol}% EoL line at warranty · **Safe** = really "
                       "declining (≥3pp) but projects to survive · **Genuinely flat** = <3pp decline, observed "
                       "≥18 months (trustworthy) · **Flat (unproven)** = <3pp decline but observed <18 months "
                       "(could still decline — just young). **Warranty deadline = whichever of time *or* the "
                       "120k-km limit comes first** — each vehicle's **◆** marker. **'Safe' = won't trigger a "
                       "warranty *claim***, not that the battery is necessarily healthy. Switch *Train on* in the "
                       "sidebar to watch the split move.")

# ═════════════════════════════════ STEP 11 ═════════════════════════════════
elif step == STEPS[12]:
    st.title("12 · From SoH to kilometres — range & km left")
    st.markdown("Forecasting gives **SoH over time**. Two conversions turn that into the numbers a warranty needs:")
    st.markdown("- **Range now ≈ rated full-charge range × SoH** — a 90%-SoH pack goes ~90% as far.\n"
                "- **Remaining km to end-of-life = km/month × months-until-SoH-hits-EoL** (read off the forecast). "
                "A high-utilisation vehicle therefore delivers *more* km before the same calendar-driven EoL.")
    st.caption("For each fleet we pick **two vehicles that both started ~100%, drove a similar distance, but aged "
               "at different rates** — so the *same* kilometres leave them at very different SoH, range and "
               "remaining life. Distance alone doesn't set health: usage intensity and heat do.")
    cols = st.columns(len(OEM_KEYS))
    for col, oem in zip(cols, OEM_KEYS):
        pair = _rul_pair(oem)
        infos = [_rul_demo(oem, DEG_ONLY, v) for v in pair]
        started = [(i["soh_hist"][0] if i.get("soh_hist") else i["cur"]) for i in infos]
        dist = np.mean([i["odo"] for i in infos if i.get("odo")]) if any(i.get("odo") for i in infos) else None
        rgap = abs(infos[0]["range_now"] - infos[1]["range_now"]) if len(infos) == 2 else 0.0
        col.markdown(f"**{oem}** · both from ~100%, ~**{dist/1000:.0f}k km** — aged differently"
                     if dist else f"**{oem}**")
        for j, info in enumerate(infos):
            dot = "🟢" if j == 0 else "🟣"
            d = f"{info['odo']/1000:.0f}k km" if info.get("odo") else f"{info['age_yr']:.1f} yr"
            col.caption(f"{dot} …{info['vin'][-6:]}: **{started[j]:.0f}% → {info['cur']:.0f}%** "
                        f"(−{started[j]-info['cur']:.0f}pp) · **{d}** · range **{info['range_now']:.0f} km**")
        if len(infos) == 2:
            drops = sorted(started[k] - infos[k]["cur"] for k in (0, 1))
            col.caption(f"↳ Same distance — one lost **{drops[0]:.0f}pp**, the other **{drops[1]:.0f}pp** → "
                        f"**{rgap:.0f} km** apart in range.")
        col.plotly_chart(_rul_soh_fig(oem, infos, on_km=True), use_container_width=True)
        col.caption("SoH vs **distance** — both from ~100%, same km, aged differently. Measured → forecast (dashed); "
                    "grey dash-dot = warranty km.")
        col.plotly_chart(_range_fig(oem, infos), use_container_width=True)
        col.caption(f"…converts to range. Grey dotted = OEM promised **{infos[0]['rated']:.0f} km** "
                    f"(ARAI · [source]({RATED_KM_SRC[oem]})); blue line = rated × SoH.")
    concept("This is exactly what `src/rul_km.py` computes (and the SoH/RUL decision dashboard shows). "
            "Remaining-km needs a **clean odometer**: Bajaj has one; Mahindra's is too sparse; Euler's field "
            "odometer is noisy — shown as *n/a* rather than a confidently wrong number.")
    takeaway("**SoH → range now** (rated × SoH) and **SoH forecast → km-to-EoL** (× usage rate) are the two "
             "numbers a data-backed warranty is built on. ARAI ranges are optimistic (real-world ~30–40% lower); "
             "what we add is the *fade* and the *remaining distance*.")

elif step == STEPS[4]:
    st.title("4 · Data quality — which vehicles can we trust?")
    st.markdown("Not every vehicle has **enough data to prove** how it's aging. A battery with only a few "
                "valid months, or a short observation window, can look 'flat' just because we haven't "
                "watched it long enough — training on it would teach the model a trend that isn't really "
                "there. So we **document each vehicle's data quality** and train only on the trustworthy ones.")
    concept("A vehicle is **trainable** only with enough valid SoH months AND a long enough age span to "
            "confirm its trend; otherwise it's **data-thin** — excluded from training until it accrues more "
            "history (or a longer-observed vehicle takes its place).")
    dq = Path("data/manifests/vehicle_data_quality.csv")
    if dq.exists():
        q = pd.read_csv(dq)
        cols = st.columns(len(OEM_KEYS))
        for col, oem in zip(cols, OEM_KEYS):
            d = q[q["oem"] == oem]; thin = d[d["quality"] == "thin"]
            col.markdown(f"### {oem}")
            col.metric("Trainable vehicles", int((d["quality"] == "trainable").sum()))
            col.metric("🚫 Data-thin (excluded)", int(len(thin)))
            col.caption(f"{int((thin['vehicle_class']=='degrader').sum())} degraders + "
                        f"{int((thin['vehicle_class']=='flat').sum())} flat too thin to trust")
            _F = FEATS_BY[oem]                          # data frequency = raw telemetry rows / vehicle-month
            _dc = "n_rows" if "n_rows" in _F.columns else ("n_rows_ic" if "n_rows_ic" in _F.columns else None)
            if _dc:
                _rpm = int(_F[_dc].median()); _hz = _rpm / 2_626_560                # 30.4 days × 86,400 s/month
                _iv = 2_626_560 / max(_rpm, 1)
                _ivs = f"{_iv:.0f} s" if _iv < 120 else (f"{_iv/60:.0f} min" if _iv < 7200 else f"{_iv/3600:.1f} h")
                col.metric("📶 Data frequency", f"{_rpm:,}/mo", f"≈ {_hz:.4f} Hz · ~1 sample / {_ivs}",
                           delta_color="off",
                           help="median raw telemetry rows per vehicle-month · Hz = rows ÷ 2.63 M s/month (avg sampling rate)")
        st.caption("**Data frequency** = median raw telemetry rows per vehicle-month — how densely each feed samples. "
                   "**In Hz** it's rows ÷ the ~2.63 M seconds in a month (average sampling rate): Euler's native BMS "
                   "feed is sparse (~750/mo ≈ **0.0003 Hz**, ~1 sample/hour); Bajaj and the intellicar-based Mahindra & "
                   "Piaggio are dense (15k–38k/mo ≈ **0.006–0.015 Hz**, ~1 sample/min). Denser sampling → more reliable "
                   "monthly SoH.")
        st.markdown("#### The data-thin vehicles — excluded from training until better observed")
        st.info("**What 'data-thin' means, exactly** (`src/build_data_quality.py`). A vehicle is flagged *thin* — and "
                "held out of **training** (it still gets a forecast at inference) — when we can't yet trust the *shape* "
                "of its SoH trend. The **`reasons`** column below is one or both of:\n"
                "- **`<6_valid_months`** — fewer than **6** months with a valid SoH reading → too few points to fit a trend;\n"
                "- **`<9mo_span`** — its readings cover **less than 9 months** of battery age → too short a window to tell "
                "real fade from noise. *(Bajaj has **no span bar** — its whole feed is ~9 months, so Bajaj is judged on the "
                "6-month rule alone.)*\n\n"
                "**Override — always trainable:** a vehicle that has already **dropped ≥5pp** or **reached end-of-life** "
                "(≤80% SoH; Bajaj ≤70%) is kept no matter how short its window — it has *proven* a real trend, which is all "
                "the span rule was there to confirm.")
        cc = ["oem", "vin", "model", "months", "span_months", "current_age_mo", "soh_drop",
              "vehicle_class", "reasons"]
        st.dataframe(q[q["quality"] == "thin"][cc].reset_index(drop=True), hide_index=True,
                     use_container_width=True)
        st.caption(f"Manifest `{dq}` — {int((q['quality']=='trainable').sum())} trainable / "
                   f"{int((q['quality']=='thin').sum())} thin of {len(q)}; built by "
                   "`src/build_data_quality.py`.")
    else:
        st.info("Run `python src/build_data_quality.py` to generate the data-quality manifest.")
    st.warning("⚠️ This is a **data** decision, not a model one: we keep well-observed *flat* vehicles "
               "(valuable negative examples) and only drop vehicles whose trend is **unprovable**. A "
               "backtest showed this beats both 'use everything' and 'drop all flat vehicles'.")

    # ── SoH-signal artifact audit: is the SoH we DO have real degradation or pipeline noise? ──
    st.markdown("---")
    st.markdown("#### 🔬 Is the SoH *signal itself* trustworthy? — measurement-artifact audit")
    st.info("**What 'artifact' means here:** a move in the *reported* SoH that comes from **how it's measured or "
            "processed** — the battery-management system re-estimating its own capacity, or our smoothing / monotone-"
            "envelope logic — **not from the battery actually ageing.** The number changed; the real health didn't. "
            "They matter because training on a fake −24pp 'drop', or a value frozen for 18 months, teaches the model "
            "degradation that never physically happened.")
    st.markdown("Data-thin (above) is about *how much* data; this is about whether the SoH numbers we **do** "
                "have are real degradation or **pipeline artifacts**. Three patterns — verified by hand on the "
                "completely-aged vehicles — corrupt the signal:")
    st.markdown("- **Cliff** — a single-month SoH drop ≥6pp: physically impossible for Li-ion, a BMS capacity "
                "*re-estimation jump* (e.g. Euler …217380 sat flat at 90.6 then dropped −24.6pp in one month). "
                "Corrupts the monthly-loss target the rate model trains on.\n"
                "- **Stuck-floor** — SoH frozen at its minimum for ≥5 months after a real drop: a held/stale "
                "value (Euler …217158 reported *exactly* 79.0 for 18 straight months).\n"
                "- **Iso-floor** *(Mahindra only)* — the monotone envelope pinned ≥2pp *below* a raw coulomb "
                "signal that had recovered (H48636: raw climbed back to 81% but the envelope froze at 78.5%).")
    arows = []; AIMP = {}
    for oem in OEM_KEYS:
        s = soh_audit.summary(data_quality.apply_quality(FEATS_BY[oem], oem), EOL_PCT[oem])
        AIMP[oem] = (int(s["aged"]), int(s["aged_tainted"]))         # (reached-EoL, of which artifact-tainted)
        arows.append({"Fleet": oem, "Vehicles": s["n"], "✅ Clean": s["clean"], "⚠️ Tainted": s["tainted"],
                      "· Cliff": s["cliff"], "· Stuck": s["stuck"], "· Iso-floor": s["iso"],
                      "Aged tainted": f"{s['aged_tainted']} / {s['aged']}"})
    asum = lambda k: sum(r[k] for r in arows)
    aged_t = sum(int(r["Aged tainted"].split(" / ")[0]) for r in arows)
    aged_n = sum(int(r["Aged tainted"].split(" / ")[1]) for r in arows)
    arows.append({"Fleet": "**All fleets**", "Vehicles": asum("Vehicles"), "✅ Clean": asum("✅ Clean"),
                  "⚠️ Tainted": asum("⚠️ Tainted"), "· Cliff": asum("· Cliff"), "· Stuck": asum("· Stuck"),
                  "· Iso-floor": asum("· Iso-floor"), "Aged tainted": f"{aged_t} / {aged_n}"})
    st.table(pd.DataFrame(arows).set_index("Fleet"))
    st.error(f"🔬 **{aged_t} of the {aged_n} completely-aged vehicles carry an artifact** — the model's end-of-life "
             "signal is almost entirely contaminated.\n"
             f"- Only **{aged_n - aged_t}** of the completely-aged cohort show clean, gradual aging all the way to EoL — the rest are artifacts.\n"
             "- **Bajaj ~98% clean** (smooth reported SoH, no envelope) — its only problem is that *nothing has aged yet*.\n"
             "- **Euler & Mahindra (~⅓ tainted)** carry the envelope + BMS-recalibration artifacts.\n"
             "- **Implication — fix the SoH pipeline before trusting long-horizon at-risk numbers:** a cliff / stuck / "
             "iso-floor doesn't just add noise, it **biases the rare end-of-life examples the model leans on**.\n"
             "- Detector: `src/soh_audit.py`.")
    st.info("🔬 **Can we get a *cleaner* SoH instead?** We tested two alternatives for Euler (LFP fleet) — "
            "**neither is a drop-in replacement:**\n"
            "- **Segment coulomb-counting** (re-derive SoH from the dense current/voltage): physically sound but **far "
            "too noisy** (9–28pp month-to-month), segment-starved on the worst vehicles — confirms coulomb is broken on "
            "this field data.\n"
            "- **The BMS's native `batterySoh` field:** garbage-laden (values of 0 and >70,000), and where clean it "
            "reads ~10–20pp *higher* than our remaining-capacity SoH.\n"
            "- **Still a useful signal:** the native estimate brackets the truth from *above* and remaining-capacity "
            "from *below* → some 'aged' Euler vehicles are likely **less degraded than the stuck-floor SoH suggests**.\n"
            "- Scripts in scratchpad; see `src/soh_audit.py`.")

    # ── artifact IMPACT: the end-of-life / at-risk signal WITH vs WITHOUT the tainted endpoints ──
    st.markdown("##### 📉 The bias, quantified — the end-of-life signal *with* vs *without* the artifacts")
    _raw_eol = sum(a for a, _ in AIMP.values()); _taint_eol = sum(t for _, t in AIMP.values())
    _clean_eol = _raw_eol - _taint_eol; _pct = 100 * _taint_eol / max(_raw_eol, 1)
    im = st.columns(3)
    im[0].metric("Reported at/below EoL (raw SoH)", _raw_eol, "what training currently sees", delta_color="off")
    im[1].metric("…of which are SoH artifacts", _taint_eol, f"~{_pct:.0f}% fake", delta_color="off")
    im[2].metric("Trustworthy end-of-life", _clean_eol, "artifact-free", delta_color="off")
    st.markdown(f"The raw SoH says **{_raw_eol}** vehicles have reached end-of-life, but **{_taint_eol}** got there via a "
                f"cliff / stuck-floor / iso-floor — only **{_clean_eol}** are real. So the model's end-of-life examples "
                f"are **~{_pct:.0f}% fake**, and any long-horizon at-risk % learned from them is **biased upward**. "
                "Excluding the tainted endpoints from training is the lever — the split per fleet:")
    ef = go.Figure(); _oems = list(AIMP)
    ef.add_bar(name="trustworthy (real EoL)", x=_oems, y=[AIMP[o][0] - AIMP[o][1] for o in _oems], marker_color=TEAL,
               text=[(AIMP[o][0] - AIMP[o][1]) or "" for o in _oems], textposition="inside")
    ef.add_bar(name="artifact-suspect (fake EoL)", x=_oems, y=[AIMP[o][1] for o in _oems], marker_color=RED,
               text=[AIMP[o][1] or "" for o in _oems], textposition="inside")
    ef.update_layout(barmode="stack", **lay(height=320, margin=dict(l=44, r=16, t=34, b=40),
                     title=dict(text="Vehicles observed at/below EoL — real vs artifact, per fleet", font=dict(size=13))))
    ef.update_yaxes(title_text="vehicles reached EoL", **AX); ef.update_xaxes(**AX)
    st.plotly_chart(ef, use_container_width=True)
    st.caption("This is the **observed** end-of-life cohort — the rare, high-value vehicles the rate model leans on to "
               "learn what aging-to-EoL looks like. With most of them artifacts, that behaviour (and the long-horizon "
               "at-risk %) is learned mostly from **fake** degradation. Real fix: a per-vehicle **endpoint SoH "
               "recompute**, or excluding these endpoints from training — both scoped off `src/soh_audit.py`.")

    # ── coverage: are we doing this for the WHOLE fleet, or just the downloaded subset? ──
    st.markdown("---")
    st.markdown("#### 📡 Coverage — are we doing this for the *whole* fleet?")
    st.markdown("**This whole dashboard runs on the full Redshift store cohort** — *not* a local subset. Below: the "
                "true registered fleet (Redshift `oem_fleet_kpis`) → how many we can model → the training-quality "
                "gate → curation + artifact audit at that scale:")
    _REGDEF = {"Euler": 2125, "Mahindra": 10933, "Bajaj": 1781, "Piaggio": 1913}
    REG = {o: int(FLEET_COV.get(o, {}).get("fleet") or _REGDEF[o]) for o in OEM_KEYS}
    cov = []
    for oem in OEM_KEYS:
        active = data_quality.apply_quality(FEATS_BY[oem], oem)
        cur = training_curation.curate(active, EOL_PCT[oem], WARRANTY_YR[oem] * 12)
        au = soh_audit.summary(active, EOL_PCT[oem])
        modeled = FEATS_BY[oem].vin.nunique()
        cov.append({"OEM": oem, "Registered fleet": REG[oem], "Modelled (store)": modeled,
                    "Coverage": f"{100*modeled/REG[oem]:.1f}%", "Trained-on (gated)": active["vin"].nunique(),
                    "✅ Good": int(cur.bucket.isin(training_curation.GOOD).sum()),
                    "🔴 At-risk": int((cur.bucket == "AT_RISK").sum()), "⚠️ Tainted": au["tainted"]})
    st.table(pd.DataFrame(cov).set_index("OEM"))
    st.warning("⚠️ **We model the full Redshift store — 729 Euler / 233 Mahindra / 1,025 Bajaj**, which is **100% of "
               "the SoH Redshift has computed** (the un-imported remainder is 0 for Mahindra/Bajaj and just 3 for "
               "Euler — verified directly against the store). **The gap to the registered fleet is physical, not a "
               "missing download:** **Mahindra** — only ~233 of 10,933 (≈2%) are SoH-measurable at all (the rest "
               "are native-only: no current → no SoH, ever, without a new signal); **Euler** — only the 2023+ "
               "dense feed carries the signal (~34%); **Bajaj** — needs the BMS-reported SoH field (~58%). Growing "
               "coverage needs **more raw signal upstream**, not another pull. (Fleet-wide view: Step 2.)")

    takeaway("Documenting data quality stops us training on unprovable vehicles by mistake — and tells us "
             "exactly which vehicles need longer observation before we can trust their trend. "
             "The artifact audit goes further: it flags where the SoH *values themselves* need fixing.")

elif step == STEPS[13]:
    st.title("13 · Limits, honesty & retraining")
    st.markdown("Good ML is **honest about what it doesn't know.** Each fleet's real data limits:")
    _lk = OEM_KEYS + (["Mahindra Native"] if NAT_BAYES is not None else [])
    cols = st.columns(len(_lk))
    for col, oem in zip(cols[:len(OEM_KEYS)], OEM_KEYS):
        o = ov(FEATS_BY[oem]); eol = EOL_PCT[oem]
        col.markdown(f"### {oem}")
        col.metric(f"Reached {eol}%", f"{int((o.smin<=eol).sum())} of {len(o)}")
        col.metric("Full journeys seen", f"{int(((o.s0>=99)&(o.smin<=eol)).sum())}")
        col.metric("Median history", f"{int(o.months.median())} mo")
    if NAT_BAYES is not None:
        col = cols[len(OEM_KEYS)]; nr = NAT_REPORT["native"]
        col.markdown("### Mahindra Native")
        col.metric("Native-only fleet", f"~{NAT_FLEET:,}")
        col.metric("Measured SoH", "none — predicted")
        col.metric("Predicted @3yr", f"{nr['soh50_at_36mo_median']:.0f}% ±{nr['band_width_at_36mo_median']/2:.0f}pp")
    st.markdown("- **Few complete journeys:** very few batteries have aged all the way to end-of-life yet, "
                "so far-future predictions are *extrapolation* — educated guesses beyond what we've seen.\n"
                "- **Young fleets:** most vehicles are still near-new (Bajaj especially — telemetry only "
                "since ~2025), limiting 'end-game' examples.\n"
                "- **Mahindra Native (~98% of the Mahindra fleet):** no current/voltage → **no measured SoH at all**; "
                "its curve is a *predicted* behaviour prior with wide bands (Step 15), sharpening only if the feed "
                "gains current/voltage — or as the fleet ages.\n"
                "- This is a **data** limit, not a model flaw — it improves only as batteries age.")
    concept("ML models are **never 'done'.** As more batteries age and more data arrives, we **retrain** to "
            "keep them accurate — on a schedule, with a versioned **registry** of every model.")
    reg = Path("models/euler/registry.json")
    if reg.exists():
        import json
        st.markdown("#### Euler model registry — every retrain is tracked")
        st.dataframe(pd.DataFrame(json.load(open(reg))), use_container_width=True)
    takeaway("That's the full pipeline across three fleets: problem → data → target → features → split → "
             "train → inspect → validate → forecast → retrain — raw telemetry to a warranty-risk call, "
             "end to end.")

# ═════════════════════════════════ STEP 14 ═════════════════════════════════
elif step == STEPS[14]:
    st.title("14 · Validation — does it work, and how much data does it need?")
    st.markdown("Three checks: **(a)** real held-out vehicles' measured SoH vs what the model predicted from "
                "only **half** their history · **(b)** how accuracy improves as we feed it more months · "
                "**(c)** whether accuracy holds across usage levels.")
    OEMCOL = {"Euler": TEAL, "Mahindra": AMBER, "Bajaj": "#6f7fd6", "Piaggio": "#c792ff"}

    # ---- (a) actual vs predicted ----
    st.markdown("#### (a) Actual vs predicted — forecast from half a vehicle's history")
    st.caption("Each vehicle was **never seen in training**. We anchor at the midpoint of its measured history "
               "(teal), forecast the rest (dashed), and compare to what actually happened — a couple heading "
               "toward end-of-life (**at-risk**) and a couple that stay healthy (**safe**), picked for the "
               "longest histories.")
    _vk = OEM_KEYS + (["Mahindra Native"] if NAT_VAL_CUR is not None else [])
    vtabs = st.tabs(_vk)
    for vt, oem in zip(vtabs[:len(OEM_KEYS)], OEM_KEYS):
        with vt:
            try:
                recs = _validation_demos(oem, DEG_ONLY); eol = EOL_PCT[oem]; a100 = RENORM100[oem]
                if not recs:
                    st.info("Not enough complete-history vehicles to backtest.")
                else:
                    vfig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                                         subplot_titles=[f"…{d['vin']} · {'🔴 at-risk' if d['lab']=='at-risk' else '🟢 safe'} · err {d['mae']:.1f}pp" for d in recs])
                    for i, d in enumerate(recs):
                        r, c = i // 2 + 1, i % 2 + 1; clr = RED if d["lab"] == "at-risk" else GREEN
                        age = np.asarray(d["age"]) / 12.0; fage = np.asarray(d["fage"]) / 12.0
                        sm = smooth(pd.Series(d["soh"]))
                        ax = ([0.0] + age.tolist()) if a100 else age.tolist()
                        sy = ([100.0] + sm.tolist()) if a100 else sm.tolist()
                        vfig.add_scatter(x=ax, y=sy, mode="lines", line=dict(color=TEAL, width=1.6), row=r, col=c, showlegend=False)
                        vfig.add_scatter(x=fage, y=d["p50"], mode="lines", line=dict(color=clr, width=1.6, dash="dash"), row=r, col=c, showlegend=False)
                        vfig.add_vline(x=d["cut_age"] / 12.0, line=dict(color="#7f8ea3", width=1, dash="dot"), row=r, col=c)
                        vfig.add_hline(y=eol, line=dict(color=AMBER, width=1, dash="dot"), row=r, col=c)
                    vfig.update_yaxes(range=[max(eol - 12, 45), 101], **AX); vfig.update_xaxes(title_text="age (years)", **AX)
                    vfig.update_annotations(font_size=11)
                    vfig.update_layout(**lay(height=540, showlegend=False, margin=dict(l=40, r=14, t=46, b=36)))
                    st.plotly_chart(vfig, use_container_width=True)
                    st.caption("Teal = measured · dashed = forecast P50 from the dotted anchor onward · amber = "
                               "EoL · **err** = mean abs error (pp) over the forecast window. The model was "
                               "trained **without** these vehicles and only saw the data left of the dotted line.")
            except Exception as ex:
                st.warning(f"Backtest unavailable ({ex}).")
    if NAT_VAL_CUR is not None:
        with vtabs[len(OEM_KEYS)]:
            st.caption("Native has **no measured SoH** — so we validate on the Mahindra vehicles that *do* (intellicar "
                       "coulomb): **hold them out**, predict their SoH from **behaviour alone**, and check against the "
                       "real measured curve. This is the ground truth behind the predicted native fleet curve.")
            demos = NAT_VAL_REP["demos"]
            nvf = make_subplots(rows=1, cols=len(demos), shared_yaxes=True, horizontal_spacing=0.03,
                                subplot_titles=[f"{d['label']} · {d['km_month']} km/mo · {d['actual_first']:.0f}→{d['actual_last']:.0f}% · band covers {d['band_coverage']*100:.0f}%"
                                                for d in demos])
            for i, d in enumerate(demos, start=1):
                cv = NAT_VAL_CUR[NAT_VAL_CUR.vin == d["vin"]].sort_values("age")
                ac = NAT_VAL_ACT[NAT_VAL_ACT.vin == d["vin"]].sort_values("age")
                nvf.add_scatter(x=cv.age, y=cv.p90, line=dict(width=0), row=1, col=i, showlegend=False, hoverinfo="skip")
                nvf.add_scatter(x=cv.age, y=cv.p10, fill="tonexty", fillcolor="rgba(224,146,43,0.18)", line=dict(width=0),
                                row=1, col=i, showlegend=(i == 1), name="predicted from behaviour (p10–p90)")
                nvf.add_scatter(x=cv.age, y=cv.p50, line=dict(color=AMBER, width=2), row=1, col=i, showlegend=(i == 1), name="predicted median")
                nvf.add_scatter(x=ac.age, y=ac.actual_soh, mode="lines+markers", line=dict(color=RED, width=1.6),
                                marker=dict(size=4), row=1, col=i, showlegend=(i == 1), name="ACTUAL measured SoH")
                nvf.add_hline(y=80, line=dict(color="#888", dash="dot"), row=1, col=i)
            nvf.update_yaxes(range=[55, 103], **AX); nvf.update_xaxes(title_text="age (months)", **AX)
            nvf.update_annotations(font_size=10); nvf.update_layout(**lay(height=380, margin=dict(l=44, r=14, t=54, b=36)))
            st.plotly_chart(nvf, use_container_width=True)
            sy = NAT_VAL_REP["systematic"]
            st.caption(f"Behaviour predicts the real curve — the light user stays flat at the top of its band, the "
                       f"heavy users decline into the lower band (76–100% of points covered). And it's systematic: "
                       f"across **{sy['vehicles']} Mahindra vehicles** more km/month means faster fade "
                       f"(r = {sy['km_vs_rate_corr']:+.2f}). The predicted native-fleet curve rides this same model → Step 15.")

    # ---- (b) data sufficiency ----
    st.markdown("---")
    st.markdown("#### (b) How much history does the model need?")
    st.caption("Same held-out vehicles each time; we give the model only the first **N months** of each and "
               "forecast the rest. Error drops as N grows, then flattens — the **knee is the minimum history** "
               "for a near-best forecast.")
    lcfig = go.Figure(); knees = {}
    for oem in OEM_KEYS:
        ns, maes, nev = _learning_curve(oem, DEG_ONLY)
        if not ns:
            continue
        lcfig.add_scatter(x=ns, y=maes, mode="lines+markers", name=f"{oem} (n={nev})", line=dict(color=OEMCOL[oem], width=2))
        best = np.nanmin(maes); knees[oem] = next((n for n, mae in zip(ns, maes) if mae <= best + 0.3), ns[-1])
    lcfig.update_xaxes(title_text="months of history given to the model", **AX)
    lcfig.update_yaxes(title_text="forecast error (MAE, pp)", **AX)
    lcfig.update_layout(**lay(height=380, margin=dict(l=52, r=20, t=30, b=46)))
    st.plotly_chart(lcfig, use_container_width=True)
    if knees:
        st.caption("**Minimum useful history (knee):** " + " · ".join(f"{o} ≈ {k} mo" for o, k in knees.items())
                   + ". Below it the forecast is materially less reliable; beyond it more history barely helps. "
                   "(Young fleets like Bajaj cap how far the curve can go.)")

    # ---- (c) usage-stratified ----
    st.markdown("---")
    st.markdown("#### (c) Does accuracy hold across usage levels? (and a defect/artifact check)")
    st.caption("Held-out vehicles split into low / medium / high **km-per-month** bands — to check the model "
               "isn't biased for any usage regime. The scatter flags **fast-decline + low-usage** vehicles: "
               "the likeliest **SoH artifacts or defective packs**, since real wear needs real use.")
    ucols = st.columns(len(OEM_KEYS))
    for col, oem in zip(ucols, OEM_KEYS):
        ru = [r for r in _backtest_eval(oem, DEG_ONLY) if r["usage"] == r["usage"] and r["usage"] > 0]
        col.markdown(f"**{oem}**")
        if len(ru) < 6:
            col.caption("not enough usage data"); continue
        u = np.array([r["usage"] for r in ru]); mae = np.array([r["mae"] for r in ru])
        t1, t2 = np.percentile(u, [33, 66])
        for b, sel in (("low", u <= t1), ("med", (u > t1) & (u <= t2)), ("high", u > t2)):
            col.caption(f"{b} use: MAE {np.mean(mae[sel]):.2f}pp (n={int(sel.sum())})")
    scfig = go.Figure()
    for oem in OEM_KEYS:
        ru = [r for r in _backtest_eval(oem, DEG_ONLY) if r["usage"] == r["usage"] and r["usage"] > 0]
        if ru:
            scfig.add_scatter(x=[r["usage"] for r in ru], y=[r["decline"] for r in ru], mode="markers",
                              name=oem, marker=dict(color=OEMCOL[oem], size=6, opacity=0.6))
    scfig.add_hline(y=0.5, line=dict(color=RED, width=1, dash="dot"), annotation_text="fast decline", annotation_font_size=10)
    scfig.update_xaxes(title_text="usage (km / month)", **AX); scfig.update_yaxes(title_text="observed decline (pp / month)", **AX)
    scfig.update_layout(**lay(height=360, margin=dict(l=52, r=20, t=30, b=46)))
    st.plotly_chart(scfig, use_container_width=True)
    st.caption("**Top-left = fast decline despite low usage = likely a SoH artifact or a defective pack** (real "
               "wear needs real use). Bottom-right = high-usage vehicles wearing as expected.")
    takeaway("The model tracks held-out vehicles to within a few pp, needs roughly the knee-many months to be "
             "reliable, and the usage view separates *real wear* from *suspect artifacts* — a built-in sanity check.")

# ═════════════════════════════════ STEP 15 ═════════════════════════════════
elif step == STEPS[15]:
    import json
    st.title("15 · The hardest fleet — probable SoH for native-only Mahindra")
    st.markdown("About **98% of the Mahindra fleet** runs on the *native* feed: it streams state-of-charge, "
                "odometer and time — but **no current, no voltage, no reported SoH**. Steps 1–14 could *measure* "
                "SoH; here we cannot. And every shortcut we tried to squeeze SoH out of the native feed "
                "(distance-per-%SoC, charge-rate, distance-to-empty) **failed** — none of them tracks the true "
                "coulomb SoH, even on the handful of vehicles where we can check it directly.")
    concept("So we change the question. We can't see this battery's *health*, but we can see its **habits** — "
            "how many km it does a month, how deep it runs down, how it charges. **Vehicles that charge and drive "
            "alike, age alike.** We learn that habit→ageing link on the fleets that *do* have real SoH (Euler, "
            "Bajaj, Piaggio, and the Mahindras that carry current), then read off a **probable** SoH curve for "
            "each native vehicle — with honest uncertainty.")

    _bp = "data/mahindra/native_behaviour_soh.parquet"; _br = "data/mahindra/behaviour_soh_report.json"
    P = pd.read_parquet(_bp) if os.path.exists(_bp) else None
    R = json.load(open(_br)) if os.path.exists(_br) else None
    if P is None or R is None:
        st.info("Run `python src/behaviour_soh_experiment.py` to build the native probable-SoH curves.")
    else:
        nat = R["native"]; kms = R["behaviour_slopes"]["km_month"]; hd = R["heldout_rate_mae"]; bd = R["band_decomposition"]
        st.markdown("#### A **Bayesian** model gives a *range*, not a single number")
        st.caption("Instead of one SoH line, it returns a **band of plausible curves**: a median (our best guess) "
                   "and a shaded region we're ~80% sure the truth sits inside. **Age** sets the backbone; **usage** "
                   "tilts how fast it falls. Trained on "
                   f"**{R['n_good_vehicles']} vehicles / {R['n_obs']:,} vehicle-months** from the four measured fleets.")
        m = st.columns(4)
        m[0].metric("Median SoH at 3 years", f"{nat['soh50_at_36mo_median']:.0f}%")
        m[1].metric("Uncertainty band (3 yr)", f"±{nat['band_width_at_36mo_median']/2:.0f} pp", help="half of the p10–p90 spread")
        m[2].metric("Mileage effect", f"{kms['abs_sensitivity']:+.3f} %/mo", help="per +1,000 km/month — the only usage habit that credibly changes the ageing rate (more km → faster fade)")
        m[3].metric("Habits beat OEM-average by", f"{hd['behaviour_improvement_pct']:+.1f}%", help="held-out improvement over just using the OEM's average ageing rate")

        c1, c2 = st.columns(2)
        ag = P.groupby("age_months"); med, p10, p90 = ag.soh_p50.mean(), ag.soh_p10.mean(), ag.soh_p90.mean(); xg = med.index
        c1.markdown("**Probable SoH for the whole native fleet**")
        f1 = go.Figure()
        f1.add_scatter(x=xg, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip")
        f1.add_scatter(x=xg, y=p10, fill="tonexty", fillcolor="rgba(31,158,143,0.18)", line=dict(width=0), name="plausible range (p10–p90)")
        f1.add_scatter(x=xg, y=med, line=dict(color=TEAL, width=3), name="most-likely SoH (median)")
        for v in list(P.vin.unique())[:15]:
            d = P[P.vin == v]
            f1.add_scatter(x=d.age_months, y=d.soh_p50, line=dict(color=GREY, width=0.6), opacity=0.35, showlegend=False, hoverinfo="skip")
        f1.add_hline(y=80, line=dict(color=RED, dash="dash"), annotation_text="80% end-of-life", annotation_font_size=10)
        f1.update_xaxes(title_text="battery age (months)", **AX); f1.update_yaxes(title_text="SoH %", range=[70, 101], **AX)
        f1.update_layout(**lay(height=400))
        c1.plotly_chart(f1, use_container_width=True)

        c2.markdown("**How habits tilt it — heavy vs light use**")
        kmv = P.groupby("vin").km_month.first()
        hi = set(kmv[kmv > kmv.quantile(.75)].index); lo = set(kmv[kmv < kmv.quantile(.25)].index)
        f2 = go.Figure()
        for grp, col, fill, lab in [(hi, RED, "rgba(212,80,78,0.12)", "heavy use (most km/month)"),
                                    (lo, GREEN, "rgba(46,193,107,0.12)", "light use (fewest km/month)")]:
            s = P[P.vin.isin(grp)].groupby("age_months"); xx = s.soh_p50.mean().index
            f2.add_scatter(x=xx, y=s.soh_p90.mean(), line=dict(width=0), showlegend=False, hoverinfo="skip")
            f2.add_scatter(x=xx, y=s.soh_p10.mean(), fill="tonexty", fillcolor=fill, line=dict(width=0), showlegend=False, hoverinfo="skip")
            f2.add_scatter(x=xx, y=s.soh_p50.mean(), line=dict(color=col, width=3), name=lab)
        f2.add_hline(y=80, line=dict(color=GREY, dash="dash"))
        f2.update_xaxes(title_text="battery age (months)", **AX); f2.update_yaxes(title_text="SoH %", range=[70, 101], **AX)
        f2.update_layout(**lay(height=400))
        c2.plotly_chart(f2, use_container_width=True)

        psk = R.get("per_source_km_rate", {})
        psk_txt = ", ".join(f"{s} {v['rho']:+.2f}" for s, v in psk.items())
        ratio = bd["heterogeneity_sd"] / bd["parameter_sd"] if bd["parameter_sd"] else float("nan")
        st.markdown("**How much to trust it — the honest part**")
        st.markdown(
            f"- **Only mileage (km/month) credibly matters.** More km → faster ageing, and this holds across "
            f"**3 of the 4 measured fleets** (per-fleet correlation: {psk_txt}), *including Mahindra's own*. "
            f"How full or empty a vehicle is kept doesn't move the needle credibly.\n"
            f"- **The band is wide on purpose.** Two vehicles with identical habits still age quite differently — "
            f"that vehicle-to-vehicle spread (σ≈{bd['heterogeneity_sd']:.2f} %/mo) is **~{ratio:.0f}× larger** than "
            f"our uncertainty about the model itself. Without current/voltage the native feed simply **cannot** "
            f"explain the rest; more data won't shrink it, a richer feed would.\n"
            f"- So this is a **fleet-level** tool (warranty exposure, cohort risk) — **not** a per-vehicle SoH gauge.")
        # ---- proof of validation on Mahindra vehicles that DO have measured SoH ----
        st.markdown("---")
        st.markdown("#### Does it actually work? Proof on the vehicles we *can* measure")
        _vc, _va = "data/mahindra/behaviour_validation_curves.parquet", "data/mahindra/behaviour_validation_actual.parquet"
        _vs, _vr = "data/mahindra/behaviour_validation_scatter.parquet", "data/mahindra/behaviour_validation_report.json"
        if all(os.path.exists(p) for p in (_vc, _va, _vs, _vr)):
            VC = pd.read_parquet(_vc); VA = pd.read_parquet(_va); VS = pd.read_parquet(_vs); VR = json.load(open(_vr))
            st.caption("The native fleet has no ground truth — but the Mahindra vehicles on the **intellicar** feed carry "
                       "both a behaviour fingerprint AND a **real, coulomb-measured** SoH. So we **hold a few out of "
                       "training**, predict their SoH curve from behaviour *alone*, and check it against reality.")
            vfig = make_subplots(rows=1, cols=len(VR["demos"]), shared_yaxes=True, horizontal_spacing=0.03,
                                 subplot_titles=[f"{d['label']} · {d['km_month']} km/mo · {d['actual_first']:.0f}→{d['actual_last']:.0f}% · band covers {d['band_coverage']*100:.0f}%"
                                                 for d in VR["demos"]])
            for i, d in enumerate(VR["demos"], start=1):
                cv = VC[VC.vin == d["vin"]].sort_values("age"); ac = VA[VA.vin == d["vin"]].sort_values("age")
                vfig.add_scatter(x=cv.age, y=cv.p90, line=dict(width=0), row=1, col=i, showlegend=False, hoverinfo="skip")
                vfig.add_scatter(x=cv.age, y=cv.p10, fill="tonexty", fillcolor="rgba(31,158,143,0.18)", line=dict(width=0),
                                 row=1, col=i, showlegend=(i == 1), name="predicted from behaviour (p10–p90)")
                vfig.add_scatter(x=cv.age, y=cv.p50, line=dict(color=TEAL, width=2), row=1, col=i, showlegend=(i == 1), name="predicted median")
                vfig.add_scatter(x=ac.age, y=ac.actual_soh, mode="lines+markers", line=dict(color=RED, width=1.6),
                                 marker=dict(size=4), row=1, col=i, showlegend=(i == 1), name="ACTUAL measured SoH")
                vfig.add_hline(y=80, line=dict(color="#888", dash="dot"), row=1, col=i)
            vfig.update_yaxes(range=[55, 103], **AX); vfig.update_xaxes(title_text="age (months)", **AX)
            vfig.update_annotations(font_size=10)
            vfig.update_layout(**lay(height=380, margin=dict(l=44, r=14, t=54, b=38)))
            st.plotly_chart(vfig, use_container_width=True)
            st.caption("Teal band = SoH predicted from **behaviour only**, with that vehicle **held out** of training · red = "
                       "its **actual** coulomb-measured SoH. The light user stays flat at the top of its band; the heavier "
                       "users decline into the lower band — behaviour predicts *where* each vehicle lands, and the band "
                       "covers 76–100% of the real measurements.")
            sy = VR["systematic"]; mult = sy["heavy_median_rate"] / sy["light_median_rate"] if sy["light_median_rate"] else float("nan")
            st.markdown(f"**And it's systematic, not cherry-picked.** Across **{sy['vehicles']} Mahindra vehicles** with a "
                        f"measured SoH, more km/month means faster fade (correlation **{sy['km_vs_rate_corr']:+.2f}**): the "
                        f"lightest third lose **{abs(sy['light_median_rate']):.2f}%/month**, the heaviest third "
                        f"**{abs(sy['heavy_median_rate']):.2f}%/month — about {mult:.1f}× faster**.")
            reg = VS[~VS.is_demo]; dem = VS[VS.is_demo]
            sfig = go.Figure()
            sfig.add_scatter(x=reg.km_month, y=reg.actual_rate, mode="markers",
                             marker=dict(color=GREY, size=6, opacity=0.55), name="Mahindra vehicles (measured)")
            b = np.polyfit(VS.km_month, VS.actual_rate, 1); xs = np.array([VS.km_month.min(), VS.km_month.max()])
            sfig.add_scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=TEAL, dash="dash", width=2), name="trend")
            sfig.add_scatter(x=dem.km_month, y=dem.actual_rate, mode="markers",
                             marker=dict(color=RED, size=13, symbol="star", line=dict(color="#fff", width=0.8)), name="the 3 shown above")
            sfig.update_xaxes(title_text="driving behaviour — km per month", **AX)
            sfig.update_yaxes(title_text="actual SoH decline (% per month)", **AX)
            sfig.update_layout(**lay(height=360, margin=dict(l=54, r=20, t=30, b=46)))
            st.plotly_chart(sfig, use_container_width=True)
        else:
            st.info("Run `python src/behaviour_soh_validation.py` to build the Mahindra validation exhibit.")

        takeaway("For the fleet we can't measure, the honest answer is a **probability, not a number**: a Mahindra "
                 "age curve tilted by mileage, with a wide but calibrated band — and on the vehicles we *can* measure, "
                 "behaviour predicts the real SoH curve (light users flat, heavy users declining, ~76–100% band "
                 "coverage). It plans warranty risk today, and sharpens as the native feed carries current/voltage.")
