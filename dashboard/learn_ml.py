"""Learn ML — a guided, from-scratch walkthrough of how our battery State-of-Health models are built.

A teaching dashboard for someone with NO machine-learning background. Pick an OEM (Euler or Mahindra)
and walk through every step of the real pipeline — problem, data, target, features,
train/validation/test split, training, feature importance, errors & overfitting, leave-one-vehicle-out
validation, forecasting with uncertainty, and limits — using that OEM's actual data and model.

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

st.set_page_config(page_title="Learn ML — Battery SoH", layout="wide", page_icon="🎓")

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
def diagnostics(oem_key):
    """Train one model on the TRAIN split; report per-transition RMSE on each split + feature importance."""
    cfg = OEMS[oem_key]
    m = load_ft(cfg["ft"])
    mod = importlib.import_module(cfg["module"])
    FEATS = mod.FEATS
    g = m.groupby("vin")
    drop = (g["soh"].first() - g["soh"].last())
    TR, VA, TE = _split(list(m["vin"].unique()), drop)
    t_tr = mod.build_transitions(m[m["vin"].isin(TR)])
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
    return {"sizes": {"train": len(TR), "validation": len(VA), "test": len(TE)},
            "errors": {"train": rmse(TR), "validation": rmse(VA), "test": rmse(TE)},
            "fi": fi}


@st.cache_resource(show_spinner=False)
def forecaster(oem_key):
    cfg = OEMS[oem_key]
    mod = importlib.import_module(cfg["module"])
    m = load_ft(cfg["ft"])
    if oem_key == "Euler":
        import euler_train
        b = euler_train.load_latest()
        return mod, b["traj_model"] if (b and b.get("traj_model")) else mod.train_traj(mod.build_traj_samples(m))
    return mod, mod.train_quantiles(mod.build_transitions(m))


def forecast_demo(oem_key, m):
    """Pick a clear teaching example (decent history, real decline, sensible non-cliff forecast)."""
    mod, fmodel = forecaster(oem_key)
    H = 18 if oem_key == "Bajaj" else 30                  # Bajaj: short history + fast, steady decline
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
WARRANTY_YR = {"Euler": 5, "Mahindra": 3, "Bajaj": 3}

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
def test_predictions(oem_key):
    """Train on the NON-test vehicles, then for each TEST vehicle forecast from its 60% history out to
    its own warranty deadline (= registration + warranty term). Returns per-vehicle actual + forecast."""
    cfg = OEMS[oem_key]
    m = load_ft(cfg["ft"])
    mod = importlib.import_module(cfg["module"])
    g = m.groupby("vin")
    drop = g["soh"].first() - g["soh"].last()
    TR, VA, TE = _split(list(m["vin"].unique()), drop)
    train = m[m["vin"].isin(TR | VA)]
    euler = oem_key == "Euler"
    fmodel = (mod.train_traj(mod.build_traj_samples(train)) if euler
              else mod.train_quantiles(mod.build_transitions(train)))
    reg = reg_dates(oem_key); wdef, wmap = warranty_map(oem_key)
    out = []
    for vin in sorted(TE):
        gg = m[m["vin"] == vin].sort_values("month").reset_index(drop=True); n = len(gg)
        if n < 6:
            continue
        cut = n - max(1, min(int(round(n * 0.4)), n - 4))
        hist = gg.iloc[:cut]; cut_age = float(gg["age_months"].iloc[cut - 1])
        warr_age = wmap.get(vin, wdef) * 12                     # registration + warranty term, on the age axis
        H = int(np.clip(round(warr_age - cut_age), 3, 40))      # forecast out to the warranty deadline (capped)
        if euler:
            fc = mod.forecast(hist, fmodel, H); p10, p50, p90 = fc[0.1], fc[0.5], fc[0.9]
        else:
            sim = mod.simulate(hist, fmodel, H)
            p10, p50, p90 = sim["q10"].to_numpy(), sim["q50"].to_numpy(), sim["q90"].to_numpy()
        fage = cut_age + np.arange(1, H + 1)
        rd = reg.get(vin)
        out.append(dict(vin=vin[-6:], reg=(rd.strftime("%b '%y") if pd.notna(rd) else "?"),
                        warr_age=warr_age, age=gg["age_months"].to_numpy().tolist(),
                        soh=gg["soh"].to_numpy().tolist(), fage=fage.tolist(),
                        p10=p10.tolist(), p50=p50.tolist(), p90=p90.tolist()))
    return out


def concept(t): st.info("💡 **Concept** — " + t)
def takeaway(t): st.success("✅ **Takeaway** — " + t)


# ───────────────────────────── sidebar / navigation ─────────────────────────────
st.sidebar.title("🎓 Learn ML")
oem = st.sidebar.selectbox("Which fleet's model?", list(OEMS.keys()))
CFG = OEMS[oem]
FEAT = load_ft(CFG["ft"])
SMOOTH = st.sidebar.checkbox("Smooth SoH curves", value=True,
                             help="Round the staircase from the monotonic SoH envelope into a curve "
                                  "(display only — the model still uses the raw monthly SoH).")


def smooth(s, win=5):
    return s.rolling(win, center=True, min_periods=1).mean() if SMOOTH else s
st.sidebar.caption(f"How our **{oem}** battery-health model is built — explained from scratch.")
STEPS = ["👋 Start here", "1 · The problem", "2 · The data", "3 · The target (SoH)", "4 · Features",
         "5 · Train / Validation / Test", "6 · Training the model", "7 · Which clues matter?",
         "8 · Errors & overfitting", "9 · A tougher test (LOVO)", "10 · Predicting the future",
         "11 · Limits & retraining"]
step = st.sidebar.radio("Steps", STEPS, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Running example: **{FEAT.vin.nunique()} {CFG['label']}**, {len(FEAT):,} vehicle-months.")
st.sidebar.progress(STEPS.index(step) / (len(STEPS) - 1), text=f"Step {STEPS.index(step)} of {len(STEPS)-1}")


def ov():
    g = FEAT.groupby("vin")
    return pd.DataFrame({"months": g.size(), "s0": g.soh.first(), "s1": g.soh.last(),
                         "smin": g.soh.min(), "age": g.age_months.last()})


# ═════════════════════════════════ STEP 0 ═════════════════════════════════
if step == STEPS[0]:
    st.title("🎓 How a Machine-Learning model is built")
    st.markdown(
        f"Welcome! This teaches **machine learning (ML) from zero**, using a real project: predicting "
        f"the **health of {oem} EV batteries**. No maths needed — every step is explained plainly, with "
        f"the actual data and model we use. *(Switch fleets with the dropdown on the left.)*")
    concept("**Machine learning** = instead of writing rules by hand, we show a computer many examples "
            "and let it *find the patterns itself*. Here: we show it many batteries aging over time, and "
            "it learns to predict how a new battery will age.")
    c = st.columns(2)
    c[0].markdown("1. **The problem** — what we predict, and why\n2. **The data** — what we collect\n"
                  "3. **The target** — the 'answer' to predict\n4. **Features** — clues for the model\n"
                  "5. **Train/Validation/Test** — splitting fairly\n6. **Training** — how it learns")
    c[1].markdown("7. **Feature importance** — which clues mattered\n8. **Errors & overfitting** — is it good?\n"
                  "9. **A tougher test** — proving it generalises\n10. **Predicting the future** — with uncertainty\n"
                  "11. **Limits & retraining** — ML is never 'done'")
    takeaway("Work through the steps in order. By the end you'll understand the whole pipeline behind our "
             "forecasts.")

# ═════════════════════════════════ STEP 1 ═════════════════════════════════
elif step == STEPS[1]:
    st.title("1 · The problem — what are we predicting?")
    st.markdown("Every EV battery slowly **wears out**. We measure that wear as **State of Health (SoH)** — "
                "the percentage of the battery's *original* capacity that remains.")
    c = st.columns(3)
    c[0].metric("Brand-new", "100% SoH", "full range")
    c[1].metric("End of first life", "80% SoH", "range noticeably down")
    c[2].metric("End of life", "~60% SoH", "needs replacing")
    concept("A battery at **80% SoH** delivers ~80% of its original range. **80%** is the industry line for "
            "'end of first life'. Our job: predict **when** each vehicle crosses it — its **Remaining "
            "Useful Life (RUL)**.")
    st.markdown("**Why it matters:** warranty (will it drop below 80% in time?), resale value, and "
                "planning replacements / second-life use.")
    takeaway("We predict a number — **future SoH** — for each vehicle. Predicting a number is a "
             "**regression** problem in ML.")

# ═════════════════════════════════ STEP 2 ═════════════════════════════════
elif step == STEPS[2]:
    st.title("2 · The data — what the model sees")
    st.markdown("Each vehicle streams battery telemetry, which we summarise to **one row per vehicle per "
                "month**. A real slice:")
    show = [c for c in ["vin", "month", "soh", "age_months", "temp_max", "soc_mean", "odo_max"] if c in FEAT]
    sample = FEAT[show].head(8).copy()
    _rd = reg_dates(oem)
    sample.insert(1, "registered", sample["vin"].map(lambda v: (_rd.get(v).date() if pd.notna(_rd.get(v)) else None)))
    st.dataframe(sample.reset_index(drop=True), use_container_width=True)
    concept("Two kinds of columns:\n\n• **The target** (`soh`) — the 'answer' we predict.\n\n"
            "• **Features** (everything else) — the *clues* the model uses.")
    c = st.columns(3)
    c[0].metric("Vehicles", f"{FEAT.vin.nunique()}")
    c[1].metric("Vehicle-months (rows)", f"{len(FEAT):,}")
    c[2].metric("Columns", f"{FEAT.shape[1]}")
    takeaway("ML needs **examples**. Each row is one: 'under these conditions, health was X'. The model "
             "learns from thousands of them.")

# ═════════════════════════════════ STEP 3 ═════════════════════════════════
elif step == STEPS[3]:
    st.title("3 · The target — State of Health over time")
    st.caption(f"How we measure SoH for {oem}: **{CFG['soh_method']}**. {CFG['soh_explain']}")
    st.markdown("Each line below is one vehicle's SoH as it ages. **Red** = clearly degraded; **grey** = "
                "still near-new. *This is what the model learns to reproduce and extend.*")
    fig = go.Figure()
    for vin, g in FEAT.groupby("vin"):
        deg = (g.soh.iloc[0] - g.soh.iloc[-1]) >= 2
        ax = [0] + g.age_months.tolist(); sy = [100.0] + smooth(g.soh).tolist()  # anchor 100% at registration
        fig.add_scatter(x=ax, y=sy, mode="lines",
                        line=dict(color=RED if deg else GREY, width=1), opacity=0.5, showlegend=False)
    fig.add_hline(y=80, line=dict(color=AMBER, dash="dash"), annotation_text="80% — end of first life")
    fig.update_xaxes(title="age (months since registration)", **AX)
    fig.update_yaxes(title="SoH (%)", range=[55, 101], **AX)
    fig.update_layout(**lay(height=420)); st.plotly_chart(fig, use_container_width=True)
    o = ov(); c = st.columns(3)
    c[0].metric("Degraders (lost ≥2%)", f"{int((o.s0-o.s1>=2).sum())} of {len(o)}")
    c[1].metric("Reached 80%", f"{int((o.smin<=80).sum())}")
    c[2].metric("Median history", f"{int(o.months.median())} months")
    takeaway("Most lines are still **above 80%** — most batteries haven't worn out yet. The few that have "
             "are the most valuable examples, because they show the model what real aging looks like.")

# ═════════════════════════════════ STEP 4 ═════════════════════════════════
elif step == STEPS[4]:
    st.title("4 · Features — turning raw data into clues")
    st.markdown("A **feature** is one clue we give the model. Raw sensor streams aren't directly useful, so "
                "we *engineer* features that capture known battery-aging factors:")
    st.markdown(
        "| Clue | Why it might matter |\n|---|---|\n"
        "| **Age** (`age_months`) | batteries age over calendar time |\n"
        "| **Heat** (`temp_mean`, `temp_max`) | heat is the #1 ager of lithium cells |\n"
        "| **Usage** (`ah_throughput`, `km_month`) | how hard the battery is worked |\n"
        "| **Charge habits** (`frac_soc_high`, `dod_mean`) | sitting full / deep discharges stress cells |\n"
        "| **Electrical stress** (`volt_mean`, `cur_abs_p95`) | voltage & current extremes |\n"
        "| **Curvature** (`inv_sqrt_age`) | batteries fade fast early, then slow down |")
    concept("**Feature engineering** = turning raw data into meaningful clues. A good feature makes the "
            "pattern *easier* for the model to see. `inv_sqrt_age`, for instance, encodes the known fact "
            "that batteries fade quickly at first, then level off.")
    vin = ov().sort_values("s1").index[0]; g = FEAT[FEAT.vin == vin]
    sm4 = smooth(g.soh)
    fig = go.Figure()
    fig.add_scatter(x=[0, g.age_months.iloc[0]], y=[100, sm4.iloc[0]], mode="lines",
                    line=dict(color=TEAL, width=1.5, dash="dot"), showlegend=False)  # 100% at registration
    fig.add_scatter(x=g.age_months, y=sm4, name="SoH (target)", line=dict(color=TEAL, width=3))
    if "temp_max" in g:
        fig.add_scatter(x=g.age_months, y=g.temp_max, name="temperature (a feature)", yaxis="y2",
                        line=dict(color=RED, width=2, dash="dot"))
        fig.update_layout(yaxis2=dict(title="temp °C", overlaying="y", side="right", **AX))
    fig.update_layout(**lay(height=340)); fig.update_xaxes(title="age (months)", **AX)
    fig.update_yaxes(title="SoH %", **AX)
    st.caption(f"One vehicle ({vin[-6:]}): the target (SoH) and one feature (temperature) over time.")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("The model receives ~20 features per row and learns which combinations predict SoH loss. "
             "Step 7 shows *which* it actually relied on.")

# ═════════════════════════════════ STEP 5 ═════════════════════════════════
elif step == STEPS[5]:
    st.title("5 · Splitting the data — train / validation / test")
    st.markdown("The golden rule of ML: **never judge a model on data it learned from.** The real question "
                "is whether it works on *new* batteries. So we split our vehicles into three groups:")
    d = diagnostics(oem); s = d["sizes"]
    c = st.columns(3)
    c[0].metric("🟢 Training set", f"{s['train']} vehicles", "the model LEARNS from these")
    c[1].metric("🟡 Validation set", f"{s['validation']} vehicles", "used to TUNE the model")
    c[2].metric("🔴 Test set", f"{s['test']} vehicles", "final UNSEEN exam")
    concept("Like studying for an exam:\n\n• **Training set** = the textbook you study.\n"
            "• **Validation set** = practice papers to adjust your approach.\n"
            "• **Test set** = the *real* exam — questions you've never seen. Only this score is honest.")
    st.warning("⚠️ **We split by *whole vehicle*, never by row.** If two months of the *same* battery were "
               "in both train and test, the model could 'peek' at that battery's future — cheating. That "
               "kind of cheating is called **data leakage**.")
    takeaway("Train to learn, validate to tune, test to judge — on *different* vehicles each. That's how we "
             "get an *honest* accuracy estimate.")

# ═════════════════════════════════ STEP 6 ═════════════════════════════════
elif step == STEPS[6]:
    st.title("6 · Training the model — how it learns")
    st.markdown(f"Our {oem} model is {CFG['model_desc']}. That sounds scary; the idea is simple.")
    st.markdown("**A single decision tree** asks yes/no questions to reach a guess:")
    st.code("if temp_max > 38°C:\n    if age_months > 24:  ->  predict 'loses 0.4% this month'\n"
            "    else:                ->  predict 'loses 0.2% this month'\nelse:                    ->  "
            "predict 'loses 0.1% this month'", language="text")
    concept("**Gradient boosting** = build *hundreds* of small trees in sequence. Each new tree focuses on "
            "fixing the **mistakes** of the ones before it. Added together, they capture subtle, combined "
            "patterns no single rule could.")
    st.markdown("Why trees and not a neural network / LSTM? We have only a **few hundred** batteries with "
                "short histories — *small* data. Gradient-boosted trees are the most reliable choice there; "
                "we even tested foundation time-series models (Chronos, TimesFM) and **this won.**")
    takeaway("'Training' = the computer tunes these trees until its predictions on the **training set** "
             "match reality as closely as possible. Next: did it learn real patterns, or just memorise?")

# ═════════════════════════════════ STEP 7 ═════════════════════════════════
elif step == STEPS[7]:
    st.title("7 · Which clues mattered most? — Feature importance")
    st.markdown("After training, the model can tell us **how much it leaned on each feature**. Bigger bar = "
                "used more.")
    fi = pd.DataFrame(diagnostics(oem)["fi"], columns=["feature", "importance"]).head(12)
    fig = go.Figure(go.Bar(x=fi.importance[::-1], y=fi.feature[::-1], orientation="h", marker_color=TEAL))
    fig.update_xaxes(title="how much the model relied on it", **AX); fig.update_yaxes(**AX)
    fig.update_layout(**lay(height=420)); st.plotly_chart(fig, use_container_width=True)
    concept("**Feature importance** opens the 'black box' a little: it shows which inputs drove predictions. "
            "It builds trust and can reveal data issues.")
    st.warning("⚠️ **Important ≠ causes aging.** Importance shows what the model *used*, not what physically "
               "wears the battery. In our data, usage clues often rank high but point the *wrong* way — "
               "because heavily-degraded batteries are also older. Heat and calendar age are the more "
               "trustworthy real drivers.")
    takeaway("Use feature importance to understand and sanity-check the model — but pair it with domain "
             "knowledge before claiming 'X causes aging'.")

# ═════════════════════════════════ STEP 8 ═════════════════════════════════
elif step == STEPS[8]:
    st.title("8 · Is it any good? — Errors & overfitting")
    st.markdown("We measure error as **RMSE** — roughly the typical size of the model's mistake (here, in "
                "percentage-points of monthly SoH loss). **Lower is better.** Checked on all three splits:")
    e = diagnostics(oem)["errors"]
    fig = go.Figure(go.Bar(x=["Training", "Validation", "Test"], y=[e["train"], e["validation"], e["test"]],
                           marker_color=[GREEN, AMBER, RED],
                           text=[f"{e['train']:.2f}", f"{e['validation']:.2f}", f"{e['test']:.2f}"],
                           textposition="outside"))
    fig.update_yaxes(title="error (RMSE, pp/month — lower is better)", **AX); fig.update_xaxes(**AX)
    fig.update_layout(**lay(height=360)); st.plotly_chart(fig, use_container_width=True)
    gap = e["test"] / max(e["train"], 1e-9)
    c = st.columns(3)
    c[0].metric("Training error", f"{e['train']:.2f}")
    c[1].metric("Test error (honest)", f"{e['test']:.2f}")
    c[2].metric("Test ÷ Train", f"{gap:.1f}×", "gap = some overfitting")
    concept("**Training error is always optimistically low** — the model has seen that data. The **test "
            "error** is the honest one. If training error is *tiny* but test error is *large*, the model "
            "**memorised** instead of learning general patterns — that's **overfitting** (acing practice "
            "answers, failing the real exam).")
    takeaway("A modest train→test gap is normal. A huge gap means overfitting — fix it with more data or a "
             "simpler model. Always quote the **test/validation** number, never the training one.")

# ═════════════════════════════════ STEP 9 ═════════════════════════════════
elif step == STEPS[9]:
    st.title("9 · A tougher, fairer test — Leave-One-Vehicle-Out")
    st.markdown("A single split can be lucky or unlucky. So we use **Leave-One-Vehicle-Out (LOVO)**: hold "
                "out **one whole vehicle**, train on the rest, forecast the held-out one — repeat for "
                "*every* vehicle. The fairest test we can run.")
    concept("LOVO asks the real question: given a battery's *early* history, can we forecast its *future* "
            "SoH? We give the model the first 60% of each vehicle's life and ask it to predict the last 40%.")
    st.markdown("#### The bar to beat: lazy baselines")
    st.markdown("- **Persistence:** 'assume SoH stays exactly where it is.'\n"
                "- **Trend line:** 'fit a simple curve and extend it.'")
    L = CFG["lovo"]
    fig = go.Figure(go.Bar(x=["Our model", "Persistence", "Trend line"], y=[L["model"], L["persist"], L["trend"]],
                           marker_color=[GREEN, GREY, AMBER],
                           text=[f"{L['model']}", f"{L['persist']}", f"{L['trend']}"], textposition="outside"))
    fig.update_yaxes(title="forecast error on declining batteries (RMSE, lower better)", **AX)
    fig.update_xaxes(**AX); fig.update_layout(**lay(height=340)); st.plotly_chart(fig, use_container_width=True)
    c = st.columns(3)
    c[0].metric("Overall LOVO error", f"{L['overall']}")
    c[1].metric("On declining batteries", f"{L['model']}")
    c[2].metric("vs persistence", f"{(1-L['model']/L['persist'])*100:.0f}% better")
    takeaway("Our model beats both lazy baselines on the batteries that actually decline — the ones that "
             "matter for warranty & RUL. That's the evidence it learned something real.")

# ═════════════════════════════════ STEP 10 ═════════════════════════════════
elif step == STEPS[10]:
    st.title("10 · Predicting the future — with honest uncertainty")
    st.markdown("The payoff: feed a battery's history to the trained model and it projects SoH **forward** — "
                "and not as one over-confident number, but as a **range**.")
    try:
        g, p10, p50, p90 = forecast_demo(oem, FEAT)
        sm = smooth(g.soh)
        a0 = g.age_months.iloc[-1]; fa = np.arange(a0 + 1, a0 + len(p50) + 1)
        # connect the forecast to the (smoothed) end of the measured line so there's no visual jump
        xc = np.concatenate([[a0], fa])
        c10 = np.concatenate([[sm.iloc[-1]], p10]); c50 = np.concatenate([[sm.iloc[-1]], p50])
        c90 = np.concatenate([[sm.iloc[-1]], p90])
        fig = go.Figure()
        fig.add_scatter(x=[0, g.age_months.iloc[0]], y=[100, sm.iloc[0]], mode="lines",
                        name="100% at registration", line=dict(color=TEAL, width=1.4, dash="dot"))
        fig.add_scatter(x=g.age_months, y=sm, name="measured so far", mode="markers+lines",
                        line=dict(color=TEAL, width=2), marker=dict(size=4))
        fig.add_scatter(x=xc, y=c90, name="optimistic", line=dict(color=GREY, width=0), showlegend=False)
        fig.add_scatter(x=xc, y=c10, name="uncertainty range", fill="tonexty",
                        fillcolor="rgba(46,193,107,.18)", line=dict(color=GREY, width=0))
        fig.add_scatter(x=xc, y=c50, name="best estimate", line=dict(color=GREEN, width=3, dash="dash"))
        fig.add_hline(y=80, line=dict(color=AMBER, dash="dash"), annotation_text="80% EoFL")
        _wdef, _wmap = warranty_map(oem); _wyr = _wmap.get(g.vin.iloc[0], _wdef)
        fig.add_vline(x=_wyr * 12, line=dict(color="#9aa7b6", dash="dashdot"),
                      annotation_text=f"{_wyr}-yr warranty", annotation_position="top")
        fig.update_xaxes(title="age (months since registration)", **AX); fig.update_yaxes(title="SoH %", **AX)
        fig.update_layout(**lay(height=420))
        _rd = reg_dates(oem).get(g.vin.iloc[0])
        _reg = f" · registered {_rd:%b %Y}" if pd.notna(_rd) else ""
        st.caption(f"Vehicle {g.vin.iloc[0][-6:]}{_reg}: measured history (solid) + {len(p50)}-month forecast "
                   f"(dashed) with its band. Warranty line = registration + {_wyr} years.")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as ex:
        st.warning(f"Forecast demo unavailable ({ex}).")
    concept("The dashed line is the **best single estimate**; the shaded band says 'we're ~80% sure the "
            "truth lands in here'. Honest uncertainty is essential — a confident-but-wrong forecast is "
            "dangerous for warranty decisions.")
    takeaway("Where the forecast crosses 80% gives each vehicle its **Remaining Useful Life**. This is "
             "exactly what powers the main SoH dashboard.")

    st.markdown("---")
    st.markdown("### 📋 Prediction vs actual — every held-out **test** vehicle")
    st.caption("For each test vehicle (never seen in training): measured SoH (teal, anchored to **100% at "
               "registration**) vs the model's forecast from 60% of its history (green dashed + band), out "
               "to its warranty deadline. Title = **registration date**. Amber dotted = 80% EoFL · grey "
               "dash-dot = warranty (registration + term).")
    try:
        preds = test_predictions(oem)
        if not preds:
            st.info("No test vehicles with enough history to forecast.")
        else:
            ncols = 4; nrows = int(np.ceil(len(preds) / ncols))
            titles = [f"{p['vin']} · reg {p['reg']}" for p in preds]
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                                vertical_spacing=0.06, horizontal_spacing=0.04)
            for i, p in enumerate(preds):
                r, c = i // ncols + 1, i % ncols + 1
                sm = smooth(pd.Series(p["soh"]))
                fig.add_scatter(x=[0, p["age"][0]], y=[100, sm.iloc[0]], mode="lines",       # 100% at registration
                                line=dict(color=TEAL, width=1, dash="dot"), row=r, col=c, showlegend=False)
                fig.add_scatter(x=p["age"], y=sm.tolist(), mode="lines",
                                line=dict(color=TEAL, width=1.4), row=r, col=c, showlegend=False)
                fig.add_scatter(x=p["fage"], y=p["p90"], mode="lines", line=dict(width=0, color=GREY),
                                row=r, col=c, showlegend=False)
                fig.add_scatter(x=p["fage"], y=p["p10"], mode="lines", fill="tonexty",
                                fillcolor="rgba(46,193,107,.15)", line=dict(width=0), row=r, col=c, showlegend=False)
                fig.add_scatter(x=p["fage"], y=p["p50"], mode="lines",
                                line=dict(color=GREEN, width=1.6, dash="dash"), row=r, col=c, showlegend=False)
                fig.add_vline(x=p["warr_age"], line=dict(color="#9aa7b6", width=1, dash="dashdot"), row=r, col=c)
            fig.add_hline(y=80, line=dict(color=AMBER, width=1, dash="dot"), row="all", col="all")
            fig.update_yaxes(range=[55, 101], **AX); fig.update_xaxes(**AX)
            fig.update_annotations(font_size=10)
            fig.update_layout(**lay(height=max(nrows * 175, 320), showlegend=False,
                                    margin=dict(l=30, r=12, t=26, b=24)))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(preds)} test vehicles. Where green (forecast) stays above the amber 80% line "
                       f"at the warranty deadline = predicted to survive warranty; where it dips below = at-risk.")
    except Exception as ex:
        st.warning(f"Test-vehicle grid unavailable ({ex}).")

# ═════════════════════════════════ STEP 11 ═════════════════════════════════
elif step == STEPS[11]:
    st.title("11 · Limits, honesty & retraining")
    st.markdown("Good ML is **honest about what it doesn't know.** Our model's real limitations:")
    o = ov(); c = st.columns(3)
    c[0].metric("Batteries that reached 80%", f"{int((o.smin<=80).sum())} of {len(o)}")
    c[1].metric("Full 100→80 journeys seen", f"{int(((o.s0>=99)&(o.smin<=80)).sum())}")
    c[2].metric("Median history", f"{int(o.months.median())} months")
    st.markdown("- **Few complete journeys:** very few batteries have aged all the way to 80% yet, so "
                "far-future predictions are *extrapolation* — educated guesses beyond what we've seen.\n"
                "- **Young fleet:** most vehicles are still near-new, limiting 'end-game' examples.\n"
                "- This is a **data** limit, not a model flaw — it improves only as batteries age.")
    concept("ML models are **never 'done'.** As more batteries age and more data arrives, we **retrain** to "
            "keep them accurate — on a schedule, with a versioned **registry** of every model.")
    reg = Path("models/euler/registry.json")
    if oem == "Euler" and reg.exists():
        import json
        st.markdown("#### Model registry — every retrain is tracked")
        st.dataframe(pd.DataFrame(json.load(open(reg))), use_container_width=True)
    takeaway("You now understand the full pipeline: problem → data → target → features → split → train → "
             "inspect → validate → forecast → retrain. That's machine learning, end to end. 🎓")
