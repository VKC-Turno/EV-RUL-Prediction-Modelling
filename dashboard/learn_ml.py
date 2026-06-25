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

_ico = ROOT / "turno_logo.png"
st.set_page_config(page_title="Turno · EV SoH Prediction Modeling", layout="wide",
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
}


def lay(**kw):
    b = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#cdd9e8", size=12), margin=dict(l=50, r=20, t=30, b=40),
             legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)), height=380)
    b.update(kw); return b


@st.cache_data(show_spinner=False)
def load_ft(path):
    return pd.read_parquet(path).sort_values(["vin", "month"])


def deg_filter(m, on):
    """When `on`, keep only DEGRADER vehicles (>=2 pp total SoH drop), excluding flat/near-new ones from
    the train/validation/test sets. Off (default) uses the full dataset."""
    if not on:
        return m
    d = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
    return m[m["vin"].isin(d[d >= 2].index)]


def _split(vins, drop, seed=0):
    """By-vehicle 60/20/20 split, stratified so each split keeps degraders and flat vehicles."""
    rng = np.random.RandomState(seed)
    out = [set(), set(), set()]
    for grp in (sorted(v for v in vins if drop[v] >= 2), sorted(v for v in vins if drop[v] < 2)):
        grp = list(grp); rng.shuffle(grp); n = len(grp)
        ntr, nva = int(n * 0.6), int(n * 0.2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out


@st.cache_data(show_spinner="Training the model for this lesson…")
def diagnostics(oem_key, deg_only=False):
    """Train one model on the TRAIN split; report per-transition RMSE on each split + feature importance."""
    cfg = OEMS[oem_key]
    m = data_quality.apply_quality(load_ft(cfg["ft"]), oem_key)   # drop data-thin; split FIXED regardless of deg_only
    mod = importlib.import_module(cfg["module"])
    FEATS = mod.FEATS
    g = m.groupby("vin")
    drop = (g["soh"].first() - g["soh"].last())
    TR, VA, TE = _split(list(m["vin"].unique()), drop)
    tr = {v for v in TR if drop[v] >= 2} if deg_only else TR   # toggle filters TRAIN only (val/test fixed)
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
    m = deg_filter(data_quality.apply_quality(load_ft(cfg["ft"]), oem_key), deg_only)
    if oem_key == "Euler":
        if not deg_only:                                  # use the persisted all-vehicle model
            import euler_train
            b = euler_train.load_latest()
            if b and b.get("traj_model"):
                return mod, b["traj_model"]
        return mod, mod.train_traj(mod.build_traj_samples(m))
    return mod, mod.train_quantiles(mod.build_transitions(m))


def forecast_demo(oem_key, m, deg_only=False):
    """Pick a clear teaching example (decent history, real decline, sensible non-cliff forecast)."""
    mod, fmodel = forecaster(oem_key, deg_only)          # model trained on degraders only when toggled
    H = 18 if oem_key == "Bajaj" else 30                  # Bajaj: short history + fast, steady decline
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
        if res[2][-1] >= 50:                             # P50 doesn't cliff to ~0 — a clean teaching arc
            return res
    return run(order[0])


# Representative warranty term (years) per OEM — drawn as the warranty deadline on the prediction plots.
# (Euler 5 yr; Mahindra Treo 3 yr = the cohort majority; Bajaj ~3 yr.)
WARRANTY_YR = {"Euler": 5, "Mahindra": 3, "Bajaj": 5}   # Bajaj RE/Maxima = 5 yr/120k km (verified spec sheet)
EOL_PCT = {"Euler": 80, "Mahindra": 80, "Bajaj": 70}   # end-of-life SoH threshold per OEM (Bajaj = 70%)
# Euler/Mahindra SoH is renormalised to 100% at registration (so we anchor the curve at 100 there); Bajaj
# uses the ABSOLUTE BMS-reported SoH, so it legitimately starts below 100 — no 100% anchor (no fake drop).
RENORM100 = {"Euler": True, "Mahindra": True, "Bajaj": False}

# vin -> registration date (= age 0 on the SoH curves), per OEM's registration file
REG_FILES = {
    "Euler": ("data/euler/Euler_Regd_Details.csv", "regd_date", "%d/%m/%y"),
    "Mahindra": ("Mh_Regd_Date.csv", "vehicle_registration_date", None),
    "Bajaj": ("Bajaj_Regd_Details.csv", "regd_date", None),
}


@st.cache_data(show_spinner=False)
def reg_dates(oem_key):
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


@st.cache_data(show_spinner="Forecasting every test vehicle…")
def test_predictions(oem_key, deg_only=False):
    """Train on the NON-test vehicles, then for each TEST vehicle forecast from its latest data out to
    its own warranty deadline (= registration + warranty term). Returns per-vehicle actual + forecast."""
    cfg = OEMS[oem_key]
    m = data_quality.apply_quality(load_ft(cfg["ft"]), oem_key)   # drop data-thin; TEST set FIXED regardless of deg_only
    mod = importlib.import_module(cfg["module"])
    g = m.groupby("vin")
    drop = g["soh"].first() - g["soh"].last()
    TR, VA, TE = _split(list(m["vin"].unique()), drop)
    tr_vins = {v for v in (TR | VA) if drop[v] >= 2} if deg_only else (TR | VA)  # filter TRAIN only
    train = m[m["vin"].isin(tr_vins)]
    euler = oem_key == "Euler"
    fmodel = (mod.train_traj(mod.build_traj_samples(train)) if euler
              else mod.train_quantiles(mod.build_transitions(train)))
    reg = reg_dates(oem_key); wdef, wmap = warranty_map(oem_key)
    out = []
    for vin in sorted(TE):
        gg = m[m["vin"] == vin].sort_values("month").reset_index(drop=True); n = len(gg)
        if n < 6:
            continue
        # Forecast from the LATEST (present) data point using the vehicle's FULL observed history — not
        # from a 60% cut. Operationally we want "where is it now and where is it heading", so the green
        # forecast begins exactly where the measured (teal) line ends.
        hist = gg; cut_age = float(gg["age_months"].iloc[-1])
        warr_age = wmap.get(vin, wdef) * 12                     # registration + warranty term, on the age axis
        H_MAX = 120                                             # cap (months) to avoid absurd extrapolation
        if euler:
            fc = mod.forecast(hist, fmodel, H_MAX); p10, p50, p90 = fc[0.1], fc[0.5], fc[0.9]
        else:
            sim = mod.simulate(hist, fmodel, H_MAX)
            p10, p50, p90 = sim["q10"].to_numpy(), sim["q50"].to_numpy(), sim["q90"].to_numpy()
        # extend the x-axis until the P50 forecast reaches the end-of-life line (even past warranty); if it
        # never does, stop a little past the warranty deadline. Always show through the warranty line.
        hit = np.where(np.asarray(p50) <= EOL_PCT[oem_key])[0]
        end = (int(hit[0]) + 4) if len(hit) else int(round(warr_age - cut_age)) + 6
        end = int(np.clip(max(end, round(warr_age - cut_age) + 2), 3, H_MAX))
        last = float(gg["soh"].iloc[-1])                        # anchor the forecast to the present SoH
        fage = np.concatenate([[cut_age], cut_age + np.arange(1, end + 1)])
        p10 = np.concatenate([[last], p10[:end]]); p50 = np.concatenate([[last], p50[:end]])
        p90 = np.concatenate([[last], p90[:end]])
        rd = reg.get(vin)
        out.append(dict(vin=vin[-6:], reg=(rd.strftime("%b '%y") if pd.notna(rd) else "?"),
                        warr_age=warr_age, age=gg["age_months"].to_numpy().tolist(),
                        soh=gg["soh"].to_numpy().tolist(), fage=fage.tolist(),
                        p10=p10.tolist(), p50=p50.tolist(), p90=p90.tolist()))
    return out


def concept(t): st.info("💡 **Concept** — " + t)
def takeaway(t): st.success("✅ **Takeaway** — " + t)


# ───────────────────────────── sidebar / navigation ─────────────────────────────
_logo = next((p for p in (ROOT / "turno_logo.png", ROOT / "turno.gif", ROOT / "image.png",
                           ROOT / "dashboard" / "image.png", ROOT / "assets" / "image.png") if p.exists()), None)
if _logo:
    st.sidebar.image(str(_logo), width=110)
st.sidebar.title("EV SoH Prediction Modeling")
st.sidebar.caption("How our battery-health models are built — explained from scratch, **comparing all "
                   "three fleets side by side.**")
SMOOTH = st.sidebar.checkbox("Smooth SoH curves", value=True,
                             help="Round the staircase from the monotonic SoH envelope into a curve "
                                  "(display only — the model still uses the raw monthly SoH).")
DEG_ONLY = st.sidebar.checkbox("Train on degraders only", value=False,
                               help="Exclude flat/near-new vehicles (lost <2% SoH) from the TRAINING set "
                                    "only — the validation/test vehicles stay the SAME, so the warranty-risk "
                                    "counts are a fair with-vs-without-flat comparison on identical "
                                    "vehicles. Affects Steps 5/7/8/10. OFF is the default.")
if DEG_ONLY:
    st.sidebar.warning("⚠️ Degraders-only TRAINING (validation/test vehicles held fixed).")


def smooth(s, win=5):
    return s.rolling(win, center=True, min_periods=1).mean() if SMOOTH else s


STEPS = ["👋 Start here", "1 · The problem", "2 · The data", "3 · The target (SoH)", "4 · Features",
         "5 · Train / Validation / Test", "6 · Training the model", "7 · Which clues matter?",
         "8 · Errors & overfitting", "9 · A tougher test (LOVO)", "10 · Predicting the future",
         "11 · Data quality", "12 · Limits & retraining"]
step = st.sidebar.radio("Steps", STEPS, label_visibility="collapsed")
st.sidebar.markdown("---")

OEM_KEYS = list(OEMS.keys())
FEATS_BY = {o: load_ft(OEMS[o]["ft"]) for o in OEM_KEYS}
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
]


def availability_df():
    rows = []
    for name, cands in SIGNALS:
        row = {"Signal": name}
        for o in OEM_KEYS:
            cols = set(FEATS_BY[o].columns)
            row[o] = "✅" if any(c in cols for c in cands) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


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
    sm = smooth(g.soh); a0 = g.age_months.iloc[-1]; fa = np.arange(a0 + 1, a0 + len(p50) + 1)
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
    wdef, wmap = warranty_map(oem); wyr = wmap.get(g.vin.iloc[0], wdef)
    fig.add_vline(x=wyr, line=dict(color="#9aa7b6", dash="dashdot"))
    fig.update_xaxes(title="age (years)", dtick=1, **AX)
    fig.update_yaxes(range=[60, 101], **AX)            # common 60–100% scale across all fleets
    fig.update_layout(**lay(height=h, margin=dict(l=40, r=8, t=22, b=34)))
    return fig, g.vin.iloc[0], len(p50), wyr


def _testgrid_fig(oem):
    preds = test_predictions(oem, DEG_ONLY)
    if not preds:
        return None, 0
    ncols = 4; nrows = int(np.ceil(len(preds) / ncols))
    titles = [f"{p['vin']} · reg {p['reg']}" for p in preds]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        vertical_spacing=0.06, horizontal_spacing=0.04)
    eol = EOL_PCT[oem]; risk = []
    for i, p in enumerate(preds):
        r, c = i // ncols + 1, i % ncols + 1
        atrisk = _fc_at_warranty(p, "p50") < eol; risk.append(atrisk)   # below EoL at warranty = at-risk
        clr = RED if atrisk else GREEN
        fillc = "rgba(212,80,78,.16)" if atrisk else "rgba(46,193,107,.15)"
        sm = smooth(pd.Series(p["soh"]))
        age = np.array(p["age"]) / 12.0; fage = np.array(p["fage"]) / 12.0   # months -> years
        if RENORM100[oem]:
            fig.add_scatter(x=[0, age[0]], y=[100, sm.iloc[0]], mode="lines",
                            line=dict(color=TEAL, width=1, dash="dot"), row=r, col=c, showlegend=False)
        else:                                          # Bajaj: telemetry starts months after registration
            fig.add_scatter(x=[0, age[0]], y=[sm.iloc[0], sm.iloc[0]], mode="lines",
                            line=dict(color=GREY, width=1, dash="dot"), row=r, col=c, showlegend=False)
        fig.add_scatter(x=age, y=sm.tolist(), mode="lines",
                        line=dict(color=TEAL, width=1.4), row=r, col=c, showlegend=False)
        fig.add_scatter(x=fage, y=p["p90"], mode="lines", line=dict(width=0, color=GREY),
                        row=r, col=c, showlegend=False)
        fig.add_scatter(x=fage, y=p["p10"], mode="lines", fill="tonexty",
                        fillcolor=fillc, line=dict(width=0), row=r, col=c, showlegend=False)
        fig.add_scatter(x=fage, y=p["p50"], mode="lines",
                        line=dict(color=clr, width=1.8, dash="dash"), row=r, col=c, showlegend=False)
        fig.add_vline(x=p["warr_age"] / 12, line=dict(color="#9aa7b6", width=1, dash="dashdot"), row=r, col=c)
    fig.add_hline(y=eol, line=dict(color=AMBER, width=1, dash="dot"), row="all", col="all")
    if not RENORM100[oem]:                                 # shade reg→first-telemetry gap (after row="all" hline)
        for i, p in enumerate(preds):
            r, c = i // ncols + 1, i % ncols + 1
            fig.add_vrect(x0=0, x1=p["age"][0] / 12.0, fillcolor="rgba(159,179,200,.06)",
                          line_width=0, layer="below", row=r, col=c)
    fig.update_yaxes(range=[min(eol - 10, 45), 101], **AX); fig.update_xaxes(dtick=1, **AX)
    fig.update_annotations(font_size=10)
    for i, flag in enumerate(risk):                       # red subplot title for at-risk vehicles
        if flag and i < len(fig.layout.annotations):
            fig.layout.annotations[i].font.color = RED
    fig.update_layout(**lay(height=max(nrows * 175, 300), showlegend=False,
                            margin=dict(l=30, r=12, t=26, b=24)))
    return fig, len(preds)


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


# ═════════════════════════════════ STEP 0 ═════════════════════════════════
if step == STEPS[0]:
    st.title("🎓 How a Machine-Learning model is built")
    st.markdown(
        "Welcome! This teaches **machine learning (ML) from zero**, using a real project: predicting the "
        "**health of EV batteries** — across **three fleets at once (Euler · Mahindra · Bajaj)**. Every "
        "page shows all three side by side, so you can see how the *same* pipeline adapts to each OEM's "
        "data — especially **which sensors each feed provides and which clues each model relies on.**")
    concept("**Machine learning** = instead of writing rules by hand, we show a computer many examples and "
            "let it *find the patterns itself*. Here: we show it many batteries aging over time, and it "
            "learns to predict how a new battery will age.")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        F = FEATS_BY[oem]
        col.markdown(f"### {oem}")
        col.caption(OEMS[oem]["label"])
        col.metric("Vehicles tracked", F.vin.nunique())
        col.markdown(f"**SoH method:** {OEMS[oem]['soh_method']}")
    takeaway("Work through the steps in order. By the end you'll understand the whole pipeline — and where "
             "the three fleets differ is exactly where the interesting comparison lives.")

# ═════════════════════════════════ STEP 1 ═════════════════════════════════
elif step == STEPS[1]:
    st.title("1 · The problem — what are we predicting?")
    st.markdown("Every EV battery slowly **wears out**. We measure that wear as **State of Health (SoH)** — "
                "the percentage of the battery's *original* capacity that remains. Each fleet has its own "
                "**end-of-life threshold** we forecast it to cross:")
    cols = st.columns(3)
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
    st.markdown("#### Which signals each fleet's feed provides")
    st.dataframe(availability_df(), hide_index=True, use_container_width=True)
    st.caption("✅ = present in that OEM's feed. **Bajaj has no pack current or voltage**, so we can't "
               "coulomb-count or measure capacity ourselves; **only Bajaj reports charge-cycle count & "
               "ambient temp**. Euler & Mahindra are the electrically-rich pair (Mahindra even has GPS).")
    concept("Two kinds of columns everywhere: **the target** (`soh`, the answer we predict) and "
            "**features** (the clues). What changes per fleet is *which clues exist*.")
    st.markdown("#### A real slice of each fleet's rows")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        F = FEATS_BY[oem]
        col.markdown(f"**{oem}** — {F.vin.nunique()} veh · {len(F):,} rows · {F.shape[1]} cols")
        show = [c for c in ["month", "soh", "age_months", "temp_max", "soc_mean", "odo_max"] if c in F]
        col.dataframe(F[show].head(6).reset_index(drop=True), hide_index=True, use_container_width=True)
    takeaway("ML needs **examples** — one per row. A model can only use clues its feed actually carries, so "
             "the three models necessarily lean on different features (Step 7 shows how differently).")

# ═════════════════════════════════ STEP 3 ═════════════════════════════════
elif step == STEPS[3]:
    st.title("3 · The target — State of Health over time")
    st.markdown("Each line is one vehicle's SoH as it ages. We split the two populations the model must "
                "handle into separate rows: **degraders** (lost ≥2% — real aging) and **still-near-new** "
                "(flat) vehicles.")
    st.markdown("##### 🔴 Degraders — what real aging looks like")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        o = ov(FEATS_BY[oem]); eol = EOL_PCT[oem]
        col.markdown(f"**{oem}** · _{OEMS[oem]['soh_method']}_ · {int((o.s0-o.s1>=2).sum())} degraders")
        col.plotly_chart(_soh_fig(oem, which="deg"), use_container_width=True)
        col.caption(f"reached {eol}%: {int((o.smin<=eol).sum())} · median history {int(o.months.median())} mo")
    st.markdown("##### ⚪ Still near-new — flat (lost <2%)")
    cols2 = st.columns(3)
    for col, oem in zip(cols2, OEM_KEYS):
        o = ov(FEATS_BY[oem])
        col.markdown(f"**{oem}** · {int((o.s0-o.s1<2).sum())} flat vehicles")
        col.plotly_chart(_soh_fig(oem, which="flat"), use_container_width=True)
    concept("How SoH is *measured* differs per fleet — Euler reads BMS remaining-capacity, Mahindra "
            "coulomb-counts current, Bajaj trusts the BMS-reported value. (Euler/Mahindra are anchored to "
            "100% at registration; Bajaj uses the absolute reported value, so its lines start below 100%.)")
    takeaway("Most vehicles sit in the **grey (flat) row** — still healthy. The **degraders** are fewer but "
             "the most valuable examples: they show the model what real aging looks like.")

# ═════════════════════════════════ STEP 4 ═════════════════════════════════
elif step == STEPS[4]:
    st.title("4 · Features — turning raw data into clues")
    st.markdown("A **feature** is one clue we give the model. We *engineer* features that capture known "
                "battery-aging factors — but **a fleet can only offer features its feed supports.**")
    groups = {
        "Euler": "✅ age · heat · **current / Ah throughput** · **voltage** · SoC habits · `inv_sqrt_age` "
                 "curvature\n\n_rich electrical signals; no ambient temp / cycle count_",
        "Mahindra": "✅ age · heat · **current / Ah** · **voltage** · SoC & DoD · mileage · GPS · curvature"
                    "\n\n_the richest feed (coulomb + location)_",
        "Bajaj": "✅ age · heat · **ambient temp** · SoC habits · **charge cycles** · mileage · curvature"
                 "\n\n❌ **no current, no voltage** — must age-predict without electrical stress signals",
    }
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
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
             "loss. Step 7 reveals *which* it actually relied on — and it differs per fleet.")

# ═════════════════════════════════ STEP 5 ═════════════════════════════════
elif step == STEPS[5]:
    st.title("5 · Splitting the data — train / validation / test")
    st.markdown("The golden rule of ML: **never judge a model on data it learned from.** For each fleet we "
                "split the *vehicles* (never rows) into three groups — 🟢 train · 🟡 validation · 🔴 test — "
                "stratified so each keeps degraders and flat vehicles. **Columns = fleets, rows = the three "
                "groups** (each shown on its own so you can see who's in it):")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        s = diagnostics(oem, DEG_ONLY)["sizes"]
        col.markdown(f"**{oem}** — 🟢 {s['train']} · 🟡 {s['validation']} · 🔴 {s['test']}")
    splitnames = ["train", "validation", "test"]
    rowlab = {"train": "🟢 Training", "validation": "🟡 Validation", "test": "🔴 Test"}
    colmap = {"train": GREEN, "validation": AMBER, "test": RED}
    fig = make_subplots(rows=3, cols=3, column_titles=list(OEM_KEYS),
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
    for ci in range(1, 4):
        fig.update_xaxes(title_text="age (years)", row=3, col=ci)
    fig.update_annotations(font_size=12)
    fig.update_layout(**lay(height=640, showlegend=False, margin=dict(l=42, r=44, t=44, b=40)))
    st.plotly_chart(fig, use_container_width=True)
    concept("Like studying for an exam: **train** = the textbook you study, **validation** = practice "
            "papers to adjust your approach, **test** = the *real* exam (questions you've never seen). Only "
            "the test score is honest.")
    st.warning("⚠️ **We split by *whole vehicle*, never by row.** If two months of the *same* battery were "
               "in both train and test, the model could 'peek' at that battery's future — cheating, called "
               "**data leakage**.")
    takeaway("Train to learn, validate to tune, test to judge — on *different* vehicles each, for every "
             "fleet. That's how we get an *honest* accuracy estimate.")

# ═════════════════════════════════ STEP 6 ═════════════════════════════════
elif step == STEPS[6]:
    st.title("6 · Training the model — how it learns")
    st.markdown("All three fleets use the same model family — **gradient-boosted decision trees** — but one "
                "model trained per fleet. A **single decision tree** asks yes/no questions to reach a guess:")
    st.code("if temp_max > 38°C:\n    if age_months > 24:  ->  predict 'loses 0.4% this month'\n"
            "    else:                ->  predict 'loses 0.2% this month'\nelse:                    ->  "
            "predict 'loses 0.1% this month'", language="text")
    concept("**Gradient boosting** = build *hundreds* of small trees in sequence. Each new tree focuses on "
            "fixing the **mistakes** of the ones before it. Added together, they capture subtle, combined "
            "patterns no single rule could.")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        col.markdown(f"**{oem}**"); col.markdown(OEMS[oem]["model_desc"])
    st.markdown("Why trees and not a neural network / LSTM? We have only a **few hundred** batteries with "
                "short histories — *small* data. Gradient-boosted trees are the most reliable choice there; "
                "we even tested foundation time-series models (Chronos, TimesFM) and **this won.**")
    takeaway("'Training' = the computer tunes these trees until its predictions on the **training set** "
             "match reality as closely as possible. Next: did it learn real patterns, or just memorise?")

# ═════════════════════════════════ STEP 7 ═════════════════════════════════
elif step == STEPS[7]:
    st.title("7 · Which clues mattered most? — Feature importance")
    st.markdown("After training, each model reports **how much it leaned on each feature** (bigger bar = "
                "used more). Compare the three — they're **not the same**, because their feeds differ. The "
                "line under each chart flags **thinly-populated features** — a clue can rank low simply "
                "because it's mostly *missing data*, not because it doesn't matter.")
    cols = st.columns(3)
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
elif step == STEPS[8]:
    st.title("8 · Is it any good? — Errors & overfitting")
    st.markdown("We measure error as **RMSE** — roughly the typical size of the model's mistake (in "
                "percentage-points of monthly SoH loss). **Lower is better.** Checked on all three splits, "
                "per fleet — the Test ÷ Train gap reveals overfitting:")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        e = diagnostics(oem, DEG_ONLY)["errors"]; gap = e["test"] / max(e["train"], 1e-9)
        col.markdown(f"**{oem}** — test **{e['test']:.2f}** · gap {gap:.1f}×")
        col.plotly_chart(_err_fig(oem), use_container_width=True)
    concept("**Training error is always optimistically low** — the model has seen that data. The **test "
            "error** is the honest one. Tiny train + large test = the model **memorised** instead of "
            "learning general patterns (**overfitting**: acing practice, failing the real exam).")
    takeaway("A modest train→test gap is normal across all three fleets. Always quote the "
             "**test/validation** number, never the training one.")

# ═════════════════════════════════ STEP 9 ═════════════════════════════════
elif step == STEPS[9]:
    st.title("9 · A tougher, fairer test — Leave-One-Vehicle-Out")
    st.markdown("A single split can be lucky or unlucky. So we use **Leave-One-Vehicle-Out (LOVO)**: hold "
                "out one whole vehicle, train on the rest, forecast it — repeat for *every* vehicle. We "
                "compare each fleet's model against two lazy baselines:")
    st.markdown("- **Persistence:** 'assume SoH stays exactly where it is.'\n"
                "- **Trend line:** 'fit a simple curve and extend it.'\n\n"
                "Each bar below is **forecast error** — RMSE in SoH percentage-points, on the batteries that "
                "actually decline. **Lower is better.**")
    cols = st.columns(3)
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
elif step == STEPS[10]:
    st.title("10 · Predicting the future — with honest uncertainty")
    st.markdown("The payoff: feed a battery's history to the trained model and it projects SoH **forward** — "
                "not as one over-confident number, but as a **range**. One clear example per fleet:")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        try:
            fig, vin, hmon, wyr = _forecast_fig(oem)
            col.markdown(f"**{oem}** · {vin[-6:]}")
            col.plotly_chart(fig, use_container_width=True)
            col.caption(f"{hmon}-mo forecast · band ≈ 80% range · amber {EOL_PCT[oem]}% EoL · dash-dot "
                        f"{wyr}-yr warranty")
        except Exception as ex:
            col.warning(f"{oem}: demo unavailable ({ex}).")
    concept("The dashed line is the **best single estimate**; the shaded band says 'we're ~80% sure the "
            "truth lands in here'. Honest uncertainty is essential — a confident-but-wrong forecast is "
            "dangerous for warranty decisions.")
    takeaway("Where each forecast crosses its end-of-life line gives that vehicle its **Remaining Useful "
             "Life**. This is exactly what powers the main SoH dashboard.")

    st.markdown("---")
    st.markdown("### 📋 Forecast from today — every held-out **test** vehicle")
    st.caption("Pick a fleet's tab. For each test vehicle (never seen in training): its **full measured "
               "history** (teal — Euler/Mahindra anchored to 100% at registration; Bajaj shows the absolute "
               "reported SoH, so it starts below 100) plus the model's forecast **from its latest (present) "
               "data point onward** (green dashed + band), out to the warranty deadline. Title = "
               "registration date · age in years.")
    tabs = st.tabs(OEM_KEYS)
    for tab, oem in zip(tabs, OEM_KEYS):
        with tab:
            try:
                preds = test_predictions(oem, DEG_ONLY); eol = EOL_PCT[oem]
                fig, n = _testgrid_fig(oem)
                if not fig:
                    st.info("No test vehicles with enough history to forecast.")
                else:
                    ar50 = sum(1 for p in preds if _fc_at_warranty(p, "p50") < eol)
                    ar10 = sum(1 for p in preds if _fc_at_warranty(p, "p10") < eol)
                    cc = st.columns(3)
                    cc[0].metric(f"{oem} test vehicles", n)
                    cc[1].metric(f"🔴 At-risk by warranty (P50 < {eol}%)", f"{ar50} / {n}")
                    cc[2].metric("Worst-case at-risk (P10)", f"{ar10} / {n}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"At-risk = forecast below the amber {eol}% line at the warranty deadline. "
                               f"Flip **Train on degraders only** in the sidebar to see how the count moves.")
            except Exception as ex:
                st.warning(f"Test-vehicle grid unavailable ({ex}).")

# ═════════════════════════════════ STEP 11 ═════════════════════════════════
elif step == STEPS[11]:
    st.title("11 · Data quality — which vehicles can we trust?")
    st.markdown("Not every vehicle has **enough data to prove** how it's aging. A battery with only a few "
                "valid months, or a short observation window, can look 'flat' just because we haven't "
                "watched it long enough — training on it would teach the model a trend that isn't really "
                "there. So we **document each vehicle's data quality** and train only on the trustworthy ones.")
    concept("A vehicle is **trainable** only with enough valid SoH months AND a long enough age span to "
            "confirm its trend; otherwise it's **data-thin** — excluded from training, and a candidate to "
            "delete and re-download as a better-observed vehicle.")
    dq = Path("data/manifests/vehicle_data_quality.csv")
    if dq.exists():
        q = pd.read_csv(dq)
        cols = st.columns(3)
        for col, oem in zip(cols, OEM_KEYS):
            d = q[q["oem"] == oem]; thin = d[d["quality"] == "thin"]
            col.markdown(f"### {oem}")
            col.metric("Trainable vehicles", int((d["quality"] == "trainable").sum()))
            col.metric("🚫 Data-thin (excluded)", int(len(thin)))
            col.caption(f"{int((thin['vehicle_class']=='degrader').sum())} degraders + "
                        f"{int((thin['vehicle_class']=='flat').sum())} flat too thin to trust")
        st.markdown("#### The data-thin vehicles — free this space and re-import better-observed ones")
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
    takeaway("Documenting data quality stops us training on unprovable vehicles by mistake — and tells us "
             "exactly which dense files to delete and replace with more useful, longer-history vehicles.")

elif step == STEPS[12]:
    st.title("12 · Limits, honesty & retraining")
    st.markdown("Good ML is **honest about what it doesn't know.** Each fleet's real data limits:")
    cols = st.columns(3)
    for col, oem in zip(cols, OEM_KEYS):
        o = ov(FEATS_BY[oem]); eol = EOL_PCT[oem]
        col.markdown(f"### {oem}")
        col.metric(f"Reached {eol}%", f"{int((o.smin<=eol).sum())} of {len(o)}")
        col.metric("Full journeys seen", f"{int(((o.s0>=99)&(o.smin<=eol)).sum())}")
        col.metric("Median history", f"{int(o.months.median())} mo")
    st.markdown("- **Few complete journeys:** very few batteries have aged all the way to end-of-life yet, "
                "so far-future predictions are *extrapolation* — educated guesses beyond what we've seen.\n"
                "- **Young fleets:** most vehicles are still near-new (Bajaj especially — telemetry only "
                "since ~2025), limiting 'end-game' examples.\n"
                "- This is a **data** limit, not a model flaw — it improves only as batteries age.")
    concept("ML models are **never 'done'.** As more batteries age and more data arrives, we **retrain** to "
            "keep them accurate — on a schedule, with a versioned **registry** of every model.")
    reg = Path("models/euler/registry.json")
    if reg.exists():
        import json
        st.markdown("#### Euler model registry — every retrain is tracked")
        st.dataframe(pd.DataFrame(json.load(open(reg))), use_container_width=True)
    takeaway("You now understand the full pipeline across three fleets: problem → data → target → features "
             "→ split → train → inspect → validate → forecast → retrain. That's machine learning, end to "
             "end. 🎓")
