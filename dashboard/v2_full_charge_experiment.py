#!/usr/bin/env python3
"""EXPERIMENT dashboard — SoH from FULL CHARGE EVENTS only (Mahindra intellicar).

Story: the production SoH pipeline estimates monthly capacity from every continuous-logging session
(charge / discharge / mixed). That scatters ~±15% and the robust envelope collapses it to a flat line.
This experiment measures capacity ONLY on full charge events (big single-direction charge ending near
100%) — the cleanest coulomb window — and asks: does the noise collapse, and does real fade emerge?

Reads the artifacts from src/full_charge_soh.py. Because that script stores EVERY charge event with its
ΔSoC / soc_end / capacity, the "what counts as a full charge" thresholds are tunable live here.

Run: .venv/bin/streamlit run dashboard/v2_full_charge_experiment.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.chdir(Path(__file__).resolve().parent.parent)

EV_P = "data/mahindra/full_charge_events.parquet"
SUM_P = "data/mahindra/full_charge_summary.parquet"
DCIR_P = "data/mahindra/full_charge_dcir.parquet"
REP_P = "data/mahindra/full_charge_report.json"
FE_P = "data/redshift/mahindra_featengg.parquet"

# ---- theme ---------------------------------------------------------------------------------
PANEL, PANEL2 = "#12151c", "#1a1f2b"
GREEN, AMBER, BLUE, RED = "#28c76f", "#f0a020", "#5aa9f7", "#ff6b6b"
TEXT, MUTE, LINE = "#e6edf3", "#8b949e", "rgba(255,255,255,0.09)"
CAP_LO, CAP_HI = 40.0, 400.0

st.set_page_config(page_title="V2 · Full-charge SoH experiment", layout="wide", page_icon="🔋")
st.markdown(f"<style>.stApp{{background:{PANEL};}} .block-container{{padding-top:2.2rem;max-width:1400px;}}</style>",
            unsafe_allow_html=True)


def style(fig, height=340, legend=True):
    fig.update_layout(height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(color=MUTE, size=12), margin=dict(l=20, r=24, t=30, b=40),
                      hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, bordercolor=LINE),
                      showlegend=legend, legend=dict(orientation="h", x=0, y=1.06, yanchor="bottom",
                                                     bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, color=MUTE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, color=MUTE)
    return fig


def card(fig, height=340, legend=True):
    with st.container(border=True):
        st.plotly_chart(style(fig, height, legend), use_container_width=True, config={"displayModeBar": False})


def cv(s):
    s = pd.Series(s).dropna()
    return float(s.std() / s.mean() * 100.0) if (len(s) > 1 and s.mean()) else np.nan


@st.cache_data(show_spinner=False)
def load(cache_key=0):                                   # cache_key = artifact mtime -> refreshes as checkpoints land
    if not os.path.exists(EV_P):
        return None, None, None, None, None
    ev = pd.read_parquet(EV_P); ev["vin"] = ev["vin"].astype(str)
    summ = pd.read_parquet(SUM_P) if os.path.exists(SUM_P) else pd.DataFrame()
    if len(summ):
        summ["vin"] = summ["vin"].astype(str)
    dcir = pd.read_parquet(DCIR_P) if os.path.exists(DCIR_P) else pd.DataFrame()
    if len(dcir):
        dcir["vin"] = dcir["vin"].astype(str)
    rep = json.load(open(REP_P)) if os.path.exists(REP_P) else {}
    fe = pd.read_parquet(FE_P, columns=["vin", "ymd", "age_months", "capacity_ah", "soh"])
    fe["vin"] = fe["vin"].astype(str)
    return ev, summ, dcir, rep, fe


def apply_full(ev, dsoc_min, end_min):
    f = ev[(ev["dsoc"] >= dsoc_min) & (ev["soc1"] >= end_min) & ev["cap"].between(CAP_LO, CAP_HI)].copy()
    return f


def veh_soh(full_v):
    """Add soh_full to one vehicle's full charges: 100*cap/cap0, cap0 = median early full-charge cap."""
    full_v = full_v.sort_values("age_months").copy()
    base = full_v[full_v["age_months"].between(0.5, 10.0)]["cap"]
    cap0 = float(base.median()) if len(base) >= 2 else (
        float(full_v["cap"].head(5).median()) if len(full_v) else np.nan)
    if np.isfinite(cap0) and cap0 > 0:
        full_v["soh_full"] = np.clip(100.0 * full_v["cap"] / cap0, None, 100.0)
    else:
        full_v["soh_full"] = np.nan
    return full_v, cap0


# ============================================================================================
_ck = os.path.getmtime(EV_P) if os.path.exists(EV_P) else 0     # bust the cache when a checkpoint rewrites artifacts
ev, summ, dcir, rep, fe = load(_ck)
st.title("🔋 V2 · Full-charge SoH experiment — Mahindra intellicar")
st.caption("**V2 of the SoH pipeline** (Mahindra intellicar only). Instead of estimating capacity from every "
           "logging session (v1), measure it only on **full charge events** — deep, single-direction charges "
           "ending near 100% — and watch the coulomb noise collapse.")

if ev is None or ev.empty:
    st.warning("Artifacts not found yet. Run `.venv/bin/python src/full_charge_soh.py` "
               "(≈15 min; scans 95 vehicles' raw telemetry). This page reads its output.")
    st.stop()

# ---- sidebar controls: the "full charge" definition is tunable -----------------------------
st.sidebar.header("What counts as a *full* charge?")
dsoc_min = st.sidebar.slider("Min SoC span ΔSoC (%)", 20, 90, 50, 5,
                             help="How much state-of-charge the event must cover. Bigger span = cleaner capacity, fewer events.")
end_min = st.sidebar.slider("Must end above (%)", 60, 100, 85, 5,
                            help="The charge must finish at least this high — a proper 'fill up', not a partial top-up.")
min_full = st.sidebar.slider("Fleet view: min full charges/vehicle", 1, 15, 3, 1)
st.sidebar.caption(f"Loaded **{ev['vin'].nunique()}** vehicles · **{len(ev):,}** charge events. "
                   "Every event is stored with its ΔSoC & end-SoC, so these thresholds re-filter live.")

FULL = apply_full(ev, dsoc_min, end_min)
# per-vehicle full-charge SoH (recomputed at current thresholds)
soh_parts, cvrows = [], []
for vin, gv in FULL.groupby("vin"):
    sv, cap0 = veh_soh(gv)
    soh_parts.append(sv)
    cvrows.append(dict(vin=vin, n_full=len(gv), cv_full=cv(gv["cap"]), cap0=cap0))
SOH = pd.concat(soh_parts, ignore_index=True) if soh_parts else pd.DataFrame(columns=FULL.columns.tolist() + ["soh_full"])
CVF = pd.DataFrame(cvrows)
# session CV per vehicle from featengg
sess_cv = fe.groupby("vin")["capacity_ah"].apply(lambda s: cv(pd.to_numeric(s, errors="coerce"))).rename("cv_session")
CVF = CVF.merge(sess_cv, on="vin", how="left")
fleet = CVF[CVF["n_full"] >= min_full]

# ---- headline metrics ----------------------------------------------------------------------
st.subheader("The headline")
c1, c2, c3, c4 = st.columns(4)
cvs = float(fleet["cv_session"].median()) if len(fleet) else np.nan
cvf = float(fleet["cv_full"].median()) if len(fleet) else np.nan
c1.metric("Vehicles (≥ threshold)", f"{len(fleet)}", help=f"with ≥{min_full} full charges")
c2.metric("Median capacity noise — sessions (now)", f"{cvs:.0f}%" if np.isfinite(cvs) else "—")
c3.metric("Median capacity noise — full charges", f"{cvf:.0f}%" if np.isfinite(cvf) else "—",
          delta=(f"−{cvs-cvf:.0f} pp" if (np.isfinite(cvs) and np.isfinite(cvf)) else None), delta_color="inverse")
c4.metric("Full charges found", f"{len(FULL):,}", help=f"of {len(ev):,} total charge events")
if np.isfinite(cvs) and np.isfinite(cvf) and cvf > 0:
    st.success(f"✅ Restricting to full charges cuts the month-to-month capacity noise about "
               f"**{cvs/cvf:.0f}×** (median CV {cvs:.0f}% → {cvf:.0f}%). Real fade of a few percent becomes "
               f"resolvable instead of buried under measurement scatter.")

with st.expander("🔬 What this feed can actually measure — feasibility audit (read before adding features)"):
    st.markdown("""
| Signal / method | Verdict | Why (measured on the raw feed) |
|---|---|---|
| **Full-charge coulomb capacity** | ✅ **usable** | CV ~1% vs ~13% for sessions — the target of this experiment |
| **SoC exposure · throughput · C-rate** | ✅ from `current` | discharge C-rate (median ~0.25C, peaks 1.6C), DoD, mean-SoC, Ah-throughput |
| **DCIR (internal resistance)** | ⚠️ **measurable, weak as age signal** | 6–11 mΩ, thousands of steps — but SoC-controlled age-trend ≈ 0 (§④) |
| **Temperature / thermal (Arrhenius)** | ❌ **absent** | 0 of 233 vehicles have any temperature; `temp_*` columns are 100% null |
| **ICA / dQ·dV (incremental capacity)** | ❌ **infeasible** | LFP flat plateau + *pack-level* voltage + 0.25C → one smeared peak, no fingerprint |
| **Rest-OCV SoC cross-check** | ❌ **dead 20–80%** | plateau moves <0.17 V per 10% SoC vs ~2 V scatter → can't police BMS-SoC drift |
| **Speed / GPS / accel–braking** | ❌ **not in feed** | raw feed has 9 columns, none kinematic |

**Coverage reality:** ~8–20 full charges per vehicle over 3.5 yr; only ~17% of calendar days have any data → a **quarterly**, not monthly, SoH cadence is realistic.
**Data hygiene (applied everywhere):** SoC ∈ [0,100] (feed has values to 837) · |current| ≤ 400 A (sentinels to 65,279) · voltage ∈ [20,60] V (reads 0.0 V on 20–37% of rows).
""")

# ============================================================================================
st.subheader("① Fleet — does the noise actually drop?")
fc1, fc2 = st.columns([0.55, 0.45])
with fc1:
    fig = go.Figure()
    mx = float(np.nanmax([fleet["cv_session"].max(), fleet["cv_full"].max(), 5])) if len(fleet) else 30
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTE, dash="dash", width=1),
                             name="equal noise", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fleet["cv_session"], y=fleet["cv_full"], mode="markers",
                             marker=dict(size=8, color=GREEN, opacity=0.7, line=dict(width=0)),
                             text=fleet["vin"], name="vehicle",
                             hovertemplate="%{text}<br>session CV %{x:.0f}%<br>full-charge CV %{y:.0f}%<extra></extra>"))
    fig.update_layout(xaxis_title="capacity noise — SESSIONS (CV %)",
                      yaxis_title="capacity noise — FULL charges (CV %)")
    card(fig, 380)
    st.caption("Each dot is a vehicle. **Below the dashed line = full charges are cleaner.** Almost every "
               "vehicle sits well below it — the further below, the noisier its session estimate was.")
with fc2:
    d = fleet.dropna(subset=["cv_session", "cv_full"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d["cv_session"], name="sessions (now)", marker_color=AMBER, opacity=0.6, nbinsx=24))
    fig.add_trace(go.Histogram(x=d["cv_full"], name="full charges", marker_color=GREEN, opacity=0.7, nbinsx=24))
    fig.update_layout(barmode="overlay", xaxis_title="per-vehicle capacity noise (CV %)", yaxis_title="vehicles")
    card(fig, 380)
    st.caption("The whole distribution shifts left: full-charge capacity clusters near a few-percent CV, "
               "while session capacity has a long noisy tail.")

# ============================================================================================
st.subheader("② Vehicle explorer — capacity & SoH, session vs full charge")
order = (CVF.assign(gap=CVF["cv_session"].fillna(0) - CVF["cv_full"].fillna(0))
         .sort_values(["n_full", "gap"], ascending=False)) if len(CVF) else CVF
choices = order[order["n_full"] >= 1]["vin"].tolist()
if not choices:
    st.info("No vehicle has a full charge at these thresholds — loosen the sliders.")
    st.stop()
vin = st.selectbox("Vehicle", choices,
                   format_func=lambda v: f"{v} · {int(order.loc[order.vin==v,'n_full'].iloc[0])} full charges "
                                         f"· session CV {order.loc[order.vin==v,'cv_session'].iloc[0]:.0f}%")
gv = SOH[SOH["vin"] == vin].sort_values("age_months")
fev = fe[fe["vin"] == vin].sort_values("age_months")
fev_cap = pd.to_numeric(fev["capacity_ah"], errors="coerce")
fev_age = pd.to_numeric(fev["age_months"], errors="coerce") / 12

ec1, ec2 = st.columns(2)
with ec1:
    fig = go.Figure()
    allv = apply_full(ev[ev["vin"] == vin], 0, 0)                 # every charge event for this vin
    fig.add_trace(go.Scatter(x=allv["age_months"] / 12, y=allv["cap"], mode="markers", name="all charge events",
                             marker=dict(size=5, color=MUTE, opacity=0.45)))
    fig.add_trace(go.Scatter(x=fev_age, y=fev_cap, mode="lines+markers", name="session capacity (now)",
                             line=dict(color=AMBER, width=1.6), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=gv["age_months"] / 12, y=gv["cap"], mode="markers", name="FULL charges",
                             marker=dict(size=9, color=GREEN, line=dict(width=1, color="#0c3"))))
    fig.update_layout(xaxis_title="battery age (years)", yaxis_title="capacity (Ah)")
    card(fig, 360)
    st.caption("Grey = every charge event (partial charges scatter wildly). **Green = full charges** — they "
               "collapse onto a tight, slowly-declining band. Amber = today's session estimate.")
with ec2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_numeric(fev["age_months"], errors="coerce") / 12,
                             y=pd.to_numeric(fev["soh"], errors="coerce"), mode="lines+markers",
                             name="production SoH (envelope)", line=dict(color=MUTE, width=2.4), marker=dict(size=4)))
    if gv["soh_full"].notna().any():
        fig.add_trace(go.Scatter(x=gv["age_months"] / 12, y=gv["soh_full"], mode="markers",
                                 name="full-charge SoH", marker=dict(size=9, color=GREEN)))
        # simple robust trend line through full-charge SoH
        m = gv.dropna(subset=["age_months", "soh_full"])
        if len(m) >= 3:
            b = np.polyfit(m["age_months"], m["soh_full"], 1)
            xs = np.array([m["age_months"].min(), m["age_months"].max()])
            fig.add_trace(go.Scatter(x=xs / 12, y=np.polyval(b, xs), mode="lines",
                                     line=dict(color=GREEN, width=1.4, dash="dash"),
                                     name=f"full-charge trend ({b[0]*12:+.1f} pp/yr)"))
    fig.update_layout(xaxis_title="battery age (years)", yaxis_title="SoH (%)")
    card(fig, 360)
    st.caption("Production SoH (grey) is the robust envelope of the noisy session capacity — often pinned flat. "
               "**Full-charge SoH (green)** is measured directly from clean full charges; its slope is a much "
               "more honest read on real fade.")

# ============================================================================================
st.subheader("③ Threshold sensitivity — cleaner vs how many charges you keep")
sweep = []
for dm in [20, 30, 40, 50, 60, 70, 80]:
    f = apply_full(ev, dm, end_min)
    per = f.groupby("vin").agg(n=("cap", "size"), c=("cap", cv))
    per = per[per["n"] >= min_full]
    sweep.append(dict(dsoc=dm, median_cv=float(per["c"].median()) if len(per) else np.nan,
                      median_n=float(per["n"].median()) if len(per) else np.nan,
                      vehicles=int(len(per))))
sw = pd.DataFrame(sweep)
tc1, tc2 = st.columns(2)
with tc1:
    fig = go.Figure(go.Scatter(x=sw["dsoc"], y=sw["median_cv"], mode="lines+markers",
                               line=dict(color=GREEN, width=2.4), marker=dict(size=7)))
    fig.add_vline(x=dsoc_min, line=dict(color=BLUE, dash="dot"))
    fig.update_layout(xaxis_title="min ΔSoC threshold (%)", yaxis_title="median full-charge noise (CV %)")
    card(fig, 320, legend=False)
    st.caption("Stricter (deeper) charges → cleaner capacity. Diminishing returns past ~50%.")
with tc2:
    fig = go.Figure(go.Scatter(x=sw["dsoc"], y=sw["median_n"], mode="lines+markers",
                               line=dict(color=AMBER, width=2.4), marker=dict(size=7)))
    fig.add_vline(x=dsoc_min, line=dict(color=BLUE, dash="dot"))
    fig.update_layout(xaxis_title="min ΔSoC threshold (%)", yaxis_title="median full charges / vehicle")
    card(fig, 320, legend=False)
    st.caption("...but stricter also means fewer charges to average. The blue line marks your current setting.")

# ============================================================================================
st.subheader("④ DCIR — resistance growth, and why it's the honest negative")
if dcir is None or dcir.empty:
    st.info("DCIR artifact not computed yet — it lands with the full `src/full_charge_soh.py` run.")
else:
    dsl = summ.dropna(subset=["dcir_slope"]) if ("dcir_slope" in summ.columns) else pd.DataFrame()
    m1, m2, m3 = st.columns(3)
    m1.metric("Vehicles with DCIR reads", int((summ["n_dcir"] > 0).sum()) if "n_dcir" in summ.columns else 0)
    m2.metric("Median resistance",
              f"{summ['dcir_med'].median():.1f} mΩ" if ("dcir_med" in summ.columns and summ['dcir_med'].notna().any()) else "—")
    m3.metric("Median age-slope (SoC-controlled)", f"{dsl['dcir_slope'].median():+.2f} mΩ/yr" if len(dsl) else "—",
              help="≈ 0 ⇒ no robust resistance growth once SoC is held fixed")
    dvv = dcir[dcir["vin"] == vin]
    gc1, gc2 = st.columns(2)
    with gc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dvv["age_months"] / 12, y=dvv["r"], mode="markers", name="all steps",
                                 marker=dict(size=3, color=MUTE, opacity=0.22)))
        band = dvv[dvv["soc"].between(40, 70)]
        fig.add_trace(go.Scatter(x=band["age_months"] / 12, y=band["r"], mode="markers", name="SoC 40–70%",
                                 marker=dict(size=4, color=BLUE, opacity=0.5)))
        if len(band) > 20:
            mm = band.assign(mo=band["age_months"].round()).groupby("mo")["r"].median().reset_index()
            fig.add_trace(go.Scatter(x=mm["mo"] / 12, y=mm["r"], mode="lines+markers", name="monthly median (band)",
                                     line=dict(color=GREEN, width=2.6)))
        fig.update_layout(xaxis_title="battery age (years)", yaxis_title="DCIR |dV/dI| (mΩ)")
        card(fig, 340)
        st.caption("Resistance is very measurable — thousands of dV/dI steps at charge start/stop. But held to a "
                   "fixed SoC band (blue), the monthly median (green) shows little age trend for this vehicle.")
    with gc2:
        socbin = dvv.assign(sb=(dvv["soc"] // 5 * 5)).groupby("sb")["r"].median().reset_index()
        fig = go.Figure(go.Scatter(x=socbin["sb"], y=socbin["r"], mode="lines+markers", line=dict(color=AMBER, width=2.6)))
        fig.update_layout(xaxis_title="state of charge (%)", yaxis_title="median DCIR (mΩ)")
        card(fig, 340, legend=False)
        st.caption("**Why we condition on SoC:** resistance depends strongly on SoC, so an uncontrolled monthly "
                   "'rise' is usually a shift in *which SoCs were sampled*, not real growth.")
    if len(dsl):
        fig = go.Figure(go.Histogram(x=dsl["dcir_slope"], marker_color=GREEN, nbinsx=30))
        fig.add_vline(x=0, line=dict(color=MUTE, dash="dash"))
        fig.update_layout(xaxis_title="per-vehicle DCIR age-slope, SoC-controlled (mΩ/yr)", yaxis_title="vehicles")
        card(fig, 300, legend=False)
        frac_up = float((dsl["dcir_slope"] > 0.5).mean()) * 100
        st.caption(f"Fleet DCIR age-slopes cluster around **zero** — only **{frac_up:.0f}%** of vehicles show a "
                   f"meaningful rise (> 0.5 mΩ/yr). **Honest result:** after controlling for SoC, resistance growth "
                   f"is *not* a robust field degradation signal on this feed — capacity fade (§②) is the usable one.")

# ============================================================================================
with st.expander("⑤ Method & caveats"):
    st.markdown(f"""
**How a full charge is measured** (from `src/full_charge_soh.py`):
1. **Clean the raw feed** — drop SoC outside [0,100] (the feed has values to 837) and |current| > {CAP_HI:.0f} A
   (sentinels to 65,279 A). The production session method survives these only because its (40,400) Ah bound
   silently discards glitch-derived sessions.
2. **Detect charge events** — contiguous runs where current is charging the pack (sign auto-detected per
   vehicle) with < 10-min gaps; capacity = ∫|I|·dt ÷ (ΔSoC/100).
3. **Keep the *full* ones** — ΔSoC ≥ **{dsoc_min:.0f}%** ending ≥ **{end_min:.0f}%** (tunable, left). These are
   deep, single-direction windows — no direction-reversal noise, minimal SoC-resolution error.
4. **SoH** = 100 × cap ÷ cap0, where cap0 = median early full-charge capacity. No heavy envelope needed —
   the signal is already clean.

**Caveats**
- Coverage: not every vehicle charges deeply often — vehicles with few full charges (raise the fleet
  threshold to see) give sparse SoH series.
- SoC is still **BMS-reported** on an LFP pack; if the BMS recalibrates its SoC scale, even a full charge's
  ΔSoC can drift — full charges reduce *scatter*, they don't fully escape BMS-SoC bias.
- This is Mahindra-intellicar only (needs battery current). Euler/Bajaj use different SoH sources.
""")
    if rep:
        st.json(rep)
