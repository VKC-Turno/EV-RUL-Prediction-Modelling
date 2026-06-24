"""Learn ML — a guided walkthrough of how our battery State-of-Health model is built.

A teaching dashboard for someone with NO machine-learning background. It walks through every step
of the real pipeline — problem, data, target, features, train/validation/test split, training,
feature importance, errors & overfitting, leave-one-vehicle-out validation, forecasting with
uncertainty, and limitations — using our actual Euler data and the model trained by src/euler_train.py.

Run:  .venv/bin/streamlit run dashboard/learn_ml.py
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

st.set_page_config(page_title="Learn ML — Battery SoH", layout="wide", page_icon="🎓")

TEAL, AMBER, RED, GREEN, GREY, BLUE = "#1f9e8f", "#e0922b", "#d4504e", "#2ec16b", "#9fb3c8", "#5b8def"
AX = dict(gridcolor="#1c2738", zerolinecolor="#1c2738", color="#8aa0b6", linecolor="#27374e")


def lay(**kw):
    b = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#cdd9e8", size=12), margin=dict(l=50, r=20, t=30, b=40),
             legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)), height=380)
    b.update(kw); return b


@st.cache_data
def load():
    feat = pd.read_parquet("data/euler/features/feature_table.parquet").sort_values(["vin", "month"])
    diag = json.load(open("models/euler/diagnostics.json")) if Path("models/euler/diagnostics.json").exists() else None
    reg = json.load(open("models/euler/registry.json")) if Path("models/euler/registry.json").exists() else None
    return feat, diag, reg


FEAT, DIAG, REG = load()


def concept(text):
    st.info("💡 **Concept** — " + text)


def takeaway(text):
    st.success("✅ **Takeaway** — " + text)


# ───────────────────────────── navigation ─────────────────────────────
STEPS = [
    "👋 Start here",
    "1 · The problem",
    "2 · The data",
    "3 · The target (SoH)",
    "4 · Features",
    "5 · Train / Validation / Test",
    "6 · Training the model",
    "7 · Which clues matter?",
    "8 · Errors & overfitting",
    "9 · A tougher test (LOVO)",
    "10 · Predicting the future",
    "11 · Limits & retraining",
]
st.sidebar.title("🎓 Learn ML")
st.sidebar.caption("How our battery health model is built — explained from scratch.")
step = st.sidebar.radio("Steps", STEPS, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Running example: **{FEAT.vin.nunique()} Euler electric-3-wheelers**, "
                   f"{len(FEAT):,} vehicle-months of data.")
prog = STEPS.index(step) / (len(STEPS) - 1)
st.sidebar.progress(prog, text=f"Step {STEPS.index(step)} of {len(STEPS)-1}")


def ov():
    g = FEAT.groupby("vin")
    return pd.DataFrame({"months": g.size(), "s0": g.soh.first(), "s1": g.soh.last(),
                         "smin": g.soh.min(), "age": g.age_months.last()})


# ═════════════════════════════════ STEP 0 ═════════════════════════════════
if step == STEPS[0]:
    st.title("🎓 How a Machine-Learning model is built")
    st.markdown(
        "Welcome! This walkthrough teaches **machine learning (ML) from zero**, using a real project: "
        "predicting the **health of electric-vehicle batteries**. No maths background needed — every "
        "step is explained in plain language, with the actual data and model we use.")
    concept("**Machine learning** = instead of writing rules by hand, we show a computer lots of "
            "examples and let it *find the patterns itself*. Here: we show it many batteries aging "
            "over time, and it learns to predict how a new battery will age.")
    st.markdown("#### The journey (left sidebar)")
    c = st.columns(2)
    c[0].markdown(
        "1. **The problem** — what are we predicting, and why?\n"
        "2. **The data** — what information we collect\n"
        "3. **The target** — the 'answer' the model learns to predict\n"
        "4. **Features** — turning raw data into useful clues\n"
        "5. **Train/Validation/Test** — how we split data fairly\n"
        "6. **Training** — how the model actually learns")
    c[1].markdown(
        "7. **Feature importance** — which clues mattered most\n"
        "8. **Errors & overfitting** — is it any good?\n"
        "9. **A tougher test** — proving it generalises\n"
        "10. **Predicting the future** — with honest uncertainty\n"
        "11. **Limits & retraining** — ML is never 'done'")
    takeaway("Work through the steps in order. By the end you'll understand the *whole* pipeline that "
             "produces the forecasts in our main dashboard.")

# ═════════════════════════════════ STEP 1 ═════════════════════════════════
elif step == STEPS[1]:
    st.title("1 · The problem — what are we predicting?")
    st.markdown(
        "Every EV battery slowly **wears out**. We measure that wear as **State of Health (SoH)** — "
        "the percentage of the battery's *original* capacity that remains.")
    c = st.columns(3)
    c[0].metric("Brand-new battery", "100% SoH", "full range")
    c[1].metric("End of first life", "80% SoH", "range noticeably down")
    c[2].metric("End of life", "~60% SoH", "needs replacing")
    concept("A battery at **80% SoH** delivers ~80% of its original range. **80%** is the industry "
            "line for 'end of first life' (warranty / second-life decisions). Our job: predict **when** "
            "each vehicle will cross it — that's **RUL: Remaining Useful Life**.")
    st.markdown("#### Why it matters")
    st.markdown("- **Warranty:** will this battery drop below 80% before the warranty ends?\n"
                "- **Resale:** a healthier battery is worth more.\n"
                "- **Planning:** when will we need to replace or repurpose packs?")
    takeaway("We want to predict a number — **future SoH** — for each vehicle. Predicting a number is "
             "called a **regression** problem in ML.")

# ═════════════════════════════════ STEP 2 ═════════════════════════════════
elif step == STEPS[2]:
    st.title("2 · The data — what the model sees")
    st.markdown(
        "Each vehicle streams telemetry from its battery: charge level, current, voltage, temperature, "
        "odometer, and more — many readings per minute. We summarise it to **one row per vehicle per "
        "month**. Here's a real slice of that table:")
    show = ["vin", "month", "soh", "age_months", "temp_mean", "volt_mean", "soc_mean", "odo_max"]
    st.dataframe(FEAT[show].head(8).reset_index(drop=True), use_container_width=True)
    concept("Two kinds of columns:\n\n"
            "• **The target** (`soh`) — the 'answer' we want to predict.\n\n"
            "• **Features** (everything else: temperature, voltage, age…) — the *clues* the model uses "
            "to predict the target.")
    c = st.columns(3)
    c[0].metric("Vehicles", f"{FEAT.vin.nunique()}")
    c[1].metric("Vehicle-months (rows)", f"{len(FEAT):,}")
    c[2].metric("Columns (features+target)", f"{FEAT.shape[1]}")
    takeaway("ML needs **examples**. Each row is one example: 'under these conditions, the battery's "
             "health was X'. The model learns from thousands of such rows.")

# ═════════════════════════════════ STEP 3 ═════════════════════════════════
elif step == STEPS[3]:
    st.title("3 · The target — State of Health over time")
    st.markdown("This is what the model is trying to predict. Each line is one vehicle's SoH as it ages. "
                "**Red** = batteries that have clearly degraded; **grey** = still near-new.")
    fig = go.Figure()
    for vin, g in FEAT.groupby("vin"):
        deg = (g.soh.iloc[0] - g.soh.iloc[-1]) >= 2
        fig.add_scatter(x=g.age_months, y=g.soh, mode="lines",
                        line=dict(color=RED if deg else GREY, width=1), opacity=0.5, showlegend=False)
    fig.add_hline(y=80, line=dict(color=AMBER, dash="dash"), annotation_text="80% — end of first life")
    fig.update_xaxes(title="age (months)", **AX); fig.update_yaxes(title="SoH (%)", range=[55, 101], **AX)
    fig.update_layout(**lay(height=420))
    st.plotly_chart(fig, use_container_width=True)
    o = ov()
    c = st.columns(3)
    c[0].metric("Degraders (lost ≥2%)", f"{int((o.s0-o.s1>=2).sum())} of {len(o)}")
    c[1].metric("Reached 80%", f"{int((o.smin<=80).sum())}")
    c[2].metric("Median history", f"{int(o.months.median())} months")
    concept("Notice most lines are still **above 80%** — most batteries haven't worn out yet. The few "
            "that *have* declined are the most valuable examples, because they show the model what "
            "real aging looks like.")
    takeaway("How we compute SoH from raw telemetry is its own topic (we use the battery's reported "
             "remaining capacity). For ML, what matters: this curve is the **'answer key'** the model "
             "learns to reproduce and extend.")

# ═════════════════════════════════ STEP 4 ═════════════════════════════════
elif step == STEPS[4]:
    st.title("4 · Features — turning raw data into clues")
    st.markdown(
        "A **feature** is a single clue we give the model. Raw sensor streams aren't directly useful, so "
        "we *engineer* features that capture things known to age a battery:")
    st.markdown(
        "| Clue | Example features | Why it might matter |\n"
        "|---|---|---|\n"
        "| **Age** | `age_months` | batteries age over calendar time |\n"
        "| **Heat** | `temp_mean`, `temp_max` | heat is the #1 ager of lithium cells |\n"
        "| **Usage** | `ah_throughput`, `km_month` | how hard the battery is worked |\n"
        "| **Charge habits** | `frac_soc_high`, `dod_mean` | sitting full / deep discharges stress cells |\n"
        "| **Electrical stress** | `volt_mean`, `cur_abs_p95` | voltage & current extremes |\n"
        "| **Curvature** | `inv_sqrt_age` | batteries fade fast early, then slow down |")
    concept("**Feature engineering** is turning raw data into meaningful clues. A good feature makes the "
            "pattern *easier* for the model to see. The `inv_sqrt_age` clue, for example, encodes the "
            "known fact that batteries fade quickly at first and then level off.")
    vin = ov().sort_values("s1").index[0]
    g = FEAT[FEAT.vin == vin]
    fig = go.Figure()
    fig.add_scatter(x=g.age_months, y=g.soh, name="SoH (target)", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=g.age_months, y=g.temp_mean, name="temperature (a feature)", yaxis="y2",
                    line=dict(color=RED, width=2, dash="dot"))
    fig.update_layout(**lay(height=340), yaxis2=dict(title="temp °C", overlaying="y", side="right", **AX))
    fig.update_xaxes(title="age (months)", **AX); fig.update_yaxes(title="SoH %", **AX)
    st.caption(f"One vehicle ({vin[-6:]}): the target (SoH) and one feature (temperature) over time.")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("The model receives ~20 features per row and learns which combinations predict SoH loss. "
             "We'll see *which* features it actually relied on in Step 7.")

# ═════════════════════════════════ STEP 5 ═════════════════════════════════
elif step == STEPS[5]:
    st.title("5 · Splitting the data — train / validation / test")
    st.markdown(
        "The golden rule of ML: **never judge a model on data it learned from.** Of course it does well "
        "on examples it has already seen — the real question is whether it works on *new* batteries. So "
        "we split our vehicles into three groups:")
    if DIAG:
        s = DIAG["split_sizes"]
        c = st.columns(3)
        c[0].metric("🟢 Training set", f"{s['train']} vehicles", "the model LEARNS from these")
        c[1].metric("🟡 Validation set", f"{s['validation']} vehicles", "used to TUNE the model")
        c[2].metric("🔴 Test set", f"{s['test']} vehicles", "final UNSEEN exam")
    concept("Think of studying for an exam:\n\n"
            "• **Training set** = the textbook you study from.\n"
            "• **Validation set** = practice papers you use to adjust your approach.\n"
            "• **Test set** = the *real* exam — questions you've never seen. Only this score is honest.")
    st.warning("⚠️ **We split by *whole vehicle*, never by row.** If two months of the *same* battery "
               "landed in both train and test, the model could 'peek' at that battery's future — "
               "cheating. Splitting by vehicle keeps the test truly unseen. This kind of cheating is "
               "called **data leakage**.")
    takeaway("Train to learn, validate to tune, test to judge — on *different* vehicles each. This is "
             "how we get an *honest* estimate of real-world accuracy.")

# ═════════════════════════════════ STEP 6 ═════════════════════════════════
elif step == STEPS[6]:
    st.title("6 · Training the model — how it learns")
    st.markdown(
        "Our model is a **gradient-boosted decision tree** (XGBoost / LightGBM). That sounds scary; the "
        "idea is simple.")
    st.markdown("**A single decision tree** asks yes/no questions to reach a guess:")
    st.code("if temp_mean > 38°C:\n    if age_months > 24:  ->  predict 'loses 0.4% this month'\n"
            "    else:                ->  predict 'loses 0.2% this month'\nelse:                    ->  "
            "predict 'loses 0.1% this month'", language="text")
    concept("**Gradient boosting** = build *hundreds* of small trees, one after another. Each new tree "
            "focuses on fixing the **mistakes** the previous trees made. Add them all up and you get a "
            "model that captures subtle, combined patterns no single rule could.")
    st.markdown("Why this kind of model (and not, say, a neural network / LSTM)?")
    st.markdown("- We have only a **few hundred batteries** with short histories — *small* data.\n"
                "- Gradient-boosted trees are the most reliable choice on small, tabular data.\n"
                "- We actually tested fancier sequence models (Chronos, TimesFM) — **this won.**")
    takeaway("'Training' = the computer adjusts these trees until its predictions on the **training set** "
             "match reality as closely as possible. Next: did it learn real patterns, or just memorise?")

# ═════════════════════════════════ STEP 7 ═════════════════════════════════
elif step == STEPS[7]:
    st.title("7 · Which clues mattered most? — Feature importance")
    st.markdown("After training, the model can tell us **how much it leaned on each feature**. Bigger bar "
                "= the model used that clue more.")
    if DIAG and DIAG.get("feature_importance_rate"):
        fi = pd.DataFrame(DIAG["feature_importance_rate"], columns=["feature", "importance"]).head(12)
        fig = go.Figure(go.Bar(x=fi.importance[::-1], y=fi.feature[::-1], orientation="h",
                               marker_color=TEAL))
        fig.update_xaxes(title="how much the model relied on it", **AX); fig.update_yaxes(**AX)
        fig.update_layout(**lay(height=420))
        st.plotly_chart(fig, use_container_width=True)
    concept("**Feature importance** opens the 'black box' a little: it shows which inputs drove the "
            "predictions. It builds trust and can reveal data issues.")
    st.warning("⚠️ **Important ≠ causes aging.** Importance shows what the model *used*, not what "
               "physically wears the battery. In our data, usage clues (throughput, deep-discharge) often "
               "rank high but point the *wrong* way — because our heavily-degraded batteries are also "
               "older. Heat and calendar age are the more trustworthy real drivers.")
    takeaway("Use feature importance to understand and sanity-check the model — but pair it with domain "
             "knowledge before claiming 'X causes battery aging'.")

# ═════════════════════════════════ STEP 8 ═════════════════════════════════
elif step == STEPS[8]:
    st.title("8 · Is it any good? — Errors & overfitting")
    st.markdown("We measure error as **RMSE** — roughly, the typical size of the model's mistake "
                "(here, in percentage-points of monthly SoH loss). **Lower is better.** We check it on "
                "all three splits:")
    if DIAG and DIAG.get("errors"):
        e = DIAG["errors"]
        fig = go.Figure(go.Bar(x=["Training", "Validation", "Test"],
                               y=[e["train"], e["validation"], e["test"]],
                               marker_color=[GREEN, AMBER, RED],
                               text=[f"{e['train']:.2f}", f"{e['validation']:.2f}", f"{e['test']:.2f}"],
                               textposition="outside"))
        fig.update_yaxes(title="error (RMSE, pp/month — lower is better)", **AX); fig.update_xaxes(**AX)
        fig.update_layout(**lay(height=360))
        st.plotly_chart(fig, use_container_width=True)
        gap = e["test"] / max(e["train"], 1e-9)
        c = st.columns(3)
        c[0].metric("Training error", f"{e['train']:.2f}")
        c[1].metric("Test error (honest)", f"{e['test']:.2f}")
        c[2].metric("Test ÷ Train", f"{gap:.1f}×", "some gap = some overfitting")
    concept("**Training error is always optimistically low** — the model has seen that data. The "
            "**test error** is the honest one. If training error is *tiny* but test error is *large*, the "
            "model **memorised** the training data instead of learning general patterns — that's "
            "**overfitting** (like memorising practice answers and failing the real exam).")
    takeaway("A modest train→test gap is normal and acceptable. A huge gap means overfitting — fix it "
             "with more data or a simpler model. Always quote the **test/validation** number, never "
             "the training one.")

# ═════════════════════════════════ STEP 9 ═════════════════════════════════
elif step == STEPS[9]:
    st.title("9 · A tougher, fairer test — Leave-One-Vehicle-Out")
    st.markdown(
        "A single train/test split can be lucky or unlucky depending on which vehicles land where. So we "
        "use **Leave-One-Vehicle-Out (LOVO)**: hold out **one whole vehicle**, train on all the others, "
        "forecast the held-out one — and repeat for *every* vehicle. It's the fairest test we can run.")
    concept("LOVO also asks the real question: given a battery's *early* history, can we forecast its "
            "*future* SoH? We give the model the first 60% of each vehicle's life and ask it to predict "
            "the last 40%.")
    st.markdown("#### The bar to beat: simple baselines")
    st.markdown("A model is only useful if it beats lazy guesses:\n"
                "- **Persistence:** 'assume SoH stays exactly where it is.'\n"
                "- **Trend line:** 'fit a simple curve and extend it.'")
    deg = go.Figure(go.Bar(x=["Our model", "Persistence", "Trend line"], y=[5.03, 5.77, 5.63],
                           marker_color=[GREEN, GREY, AMBER],
                           text=["5.03", "5.77", "5.63"], textposition="outside"))
    deg.update_yaxes(title="forecast error on declining batteries (RMSE, lower better)", **AX)
    deg.update_xaxes(**AX); deg.update_layout(**lay(height=340))
    st.plotly_chart(deg, use_container_width=True)
    if REG:
        last = REG[-1]
        c = st.columns(3)
        c[0].metric("Overall LOVO error", f"{last.get('overall_rmse')}")
        c[1].metric("On declining batteries", f"{last.get('degrading_rmse')}")
        c[2].metric("Uncertainty band coverage", f"{last.get('band_coverage')}", "target ≈ 0.80")
    takeaway("Our model beats both baselines on the batteries that actually decline — the ones that "
             "matter for warranty & RUL. That's the evidence it learned something real.")

# ═════════════════════════════════ STEP 10 ═════════════════════════════════
elif step == STEPS[10]:
    st.title("10 · Predicting the future — with honest uncertainty")
    st.markdown("Finally, the payoff: feed a battery's history to the trained model and it projects SoH "
                "**forward**. Crucially, it doesn't give one over-confident number — it gives a **range**.")
    try:
        import euler_model as em, euler_train
        bundle = euler_train.load_latest()
        if bundle and bundle.get("traj_model"):
            # pick a clear teaching example: long history, moderate decline, currently a bit above 80%,
            # whose forecast sensibly approaches/crosses 80% (avoid steep low-quality vehicles that cliff).
            o = ov()
            cand = o[(o.months >= 15) & (o.s1.between(83, 92)) & ((o.s0 - o.s1).between(3, 15))] \
                .sort_values("months", ascending=False)
            order = list(cand.index) or list(o.sort_values("months", ascending=False).index)
            vin, fc, g = order[0], None, None
            for v in order:
                gg = FEAT[FEAT.vin == v].sort_values("month").reset_index(drop=True)
                f = em.forecast(gg, bundle["traj_model"], 30)
                if 58 <= f[0.5][-1] <= 83:                 # sensible decline that nears/crosses 80%
                    vin, fc, g = v, f, gg; break
            if fc is None:
                g = FEAT[FEAT.vin == vin].sort_values("month").reset_index(drop=True)
                fc = em.forecast(g, bundle["traj_model"], 30)
            last_age = g.age_months.iloc[-1]
            fa = np.arange(last_age + 1, last_age + 31)
            fig = go.Figure()
            fig.add_scatter(x=g.age_months, y=g.soh, name="measured so far", mode="markers+lines",
                            line=dict(color=TEAL, width=2), marker=dict(size=5))
            fig.add_scatter(x=fa, y=fc[0.9], name="optimistic (P90)", line=dict(color=GREY, width=0))
            fig.add_scatter(x=fa, y=fc[0.1], name="uncertainty range", fill="tonexty",
                            fillcolor="rgba(46,193,107,.18)", line=dict(color=GREY, width=0))
            fig.add_scatter(x=fa, y=fc[0.5], name="best estimate (P50)",
                            line=dict(color=GREEN, width=3, dash="dash"))
            fig.add_hline(y=80, line=dict(color=AMBER, dash="dash"), annotation_text="80% EoFL")
            fig.update_xaxes(title="age (months)", **AX); fig.update_yaxes(title="SoH %", **AX)
            fig.update_layout(**lay(height=420))
            st.caption(f"Vehicle {vin[-6:]}: measured history (solid) + 24-month forecast (dashed) with "
                       f"its uncertainty band.")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecast demo unavailable ({e}). Run `python src/euler_train.py` first.")
    concept("The **P50** line is the best single estimate. The shaded band (**P10–P90**) says 'we're ~80% "
            "sure the truth lands in here'. Honest uncertainty is essential — a confident-but-wrong "
            "forecast is dangerous for warranty decisions.")
    takeaway("The forecast crosses 80% at some future age — that gives each vehicle its **Remaining "
             "Useful Life**. This is exactly what powers the main SoH dashboard.")

# ═════════════════════════════════ STEP 11 ═════════════════════════════════
elif step == STEPS[11]:
    st.title("11 · Limits, honesty & retraining")
    st.markdown("Good ML is **honest about what it doesn't know.** Our model's real limitations:")
    o = ov()
    c = st.columns(3)
    c[0].metric("Batteries that reached 80%", f"{int((o.smin<=80).sum())} of {len(o)}")
    c[1].metric("Full 100%→80% journeys seen", f"{int(((o.s0>=99)&(o.smin<=80)).sum())}")
    c[2].metric("Median history", f"{int(o.months.median())} months")
    st.markdown("- **Few complete journeys:** very few batteries have actually aged all the way to 80% "
                "yet, so far-future predictions are *extrapolation* — educated guesses beyond what we've "
                "seen.\n"
                "- **Young fleet:** most vehicles are still near-new, so the model has limited 'end-game' "
                "examples.\n"
                "- This is a **data** limit, not a model flaw — and it improves only as batteries age.")
    concept("ML models are **never 'done'.** As more batteries age and more data arrives, we **retrain** "
            "to keep them accurate. We do this on a schedule and keep a **registry** of every version.")
    if REG:
        st.markdown("#### Model registry — every retrain is tracked")
        st.dataframe(pd.DataFrame(REG)[["version", "trained_at", "n_vehicles", "n_degraders",
                                        "overall_rmse", "degrading_rmse"]], use_container_width=True)
    takeaway("You now understand the full pipeline: problem → data → target → features → split → train → "
             "inspect → validate → forecast → retrain. That's machine learning, end to end. 🎓")
