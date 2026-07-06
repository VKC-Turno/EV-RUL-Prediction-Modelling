#!/usr/bin/env python3
"""Flatness investigation & fix — Euler · Bajaj · Piaggio (one page for all three).

Investigates WHY each OEM's SoH comes out flat, and whether it's fixable. Grounded in the finding that
Mahindra/Piaggio flatness = noisy coulomb → normalize-to-cap0 → clip-at-100 → isotonic envelope collapse,
Euler = same envelope machinery on BMS remaining-capacity, and Bajaj = the BMS *reports* SoH already flat
in coarse 1-pp integer steps (a source-resolution limit, not our pipeline).

Reads each OEM's data/<oem>/features/feature_table.parquet and src/soh_audit.py.
Run: .venv/bin/streamlit run dashboard/flatness_investigation.py
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, "src")
import soh_audit

# ---- theme ---------------------------------------------------------------------------------
PANEL, PANEL2 = "#12151c", "#1a1f2b"
GREEN, AMBER, BLUE, RED, PURPLE = "#28c76f", "#f0a020", "#5aa9f7", "#ff6b6b", "#c792ff"
TEXT, MUTE, LINE = "#e6edf3", "#8b949e", "rgba(255,255,255,0.09)"

st.set_page_config(page_title="Flatness investigation — Euler/Bajaj/Piaggio", layout="wide", page_icon="🩺")
st.markdown(f"<style>.stApp{{background:{PANEL};}} .block-container{{padding-top:2.2rem;max-width:1400px;}}</style>",
            unsafe_allow_html=True)

# ---- per-OEM metadata (SoH method, cause, fixability) — from the cross-OEM flatness audit ---
OEMS = {
    "Euler": dict(key="euler", eol=80, chem="LFP",
                  method="BMS remaining-capacity (high-SoC band)",
                  cause="SAME family as Mahindra", cause_color=AMBER,
                  mech="full_cap = remCap/(SoC/100) on 95–100% SoC → normalize to early cap0 → clip ≤100 → "
                       "isotonic monotone envelope. Trendless/noisy BMS-capacity gets flattened; early months pin at 100.",
                  v2="✅ APPLIES", v2_color=GREEN,
                  fix="Port v2 full-charge here: the **dense** Euler feed carries signed current (~79%) and pack "
                      "voltage (~100%, 57–61 V) — contradicting the config note that says voltage is NULL. Voltage-"
                      "bounded full-charge capacity should cut the same noise it did on Mahindra."),
    "Bajaj": dict(key="bajaj", eol=70, chem="LFP",
                  method="BMS-reported SoH (essBmsSohcEstPercValue)", cause="FUNDAMENTALLY DIFFERENT", cause_color=RED,
                  mech="The BMS reports SoH already-flat, integer-quantized in 1-pp steps (100% integer-valued). "
                       "A vehicle looks flat only because the firmware's coarse counter hasn't ticked down in the "
                       "~10-month window. No coulomb, no cap0, essentially no 100-clip.",
                  v2="❌ IMPOSSIBLE", v2_color=RED,
                  fix="No current AND no voltage → coulomb / voltage-window both impossible. Flatness is a **source "
                      "resolution limit**, not our pipeline — only finer BMS SoH logging upstream would help."),
    "Piaggio": dict(key="piaggio", eol=80, chem="LFP",
                    method="Coulomb — literally Mahindra's src/soh.py", cause="IDENTICAL to Mahindra", cause_color=RED,
                    mech="Same ΔSoC-weighted pooled coulomb capacity + robust-isotonic envelope + cap0 normalize + "
                         "100-clip. Only OEM besides Mahindra to trip ISO_FLOOR — the envelope-below-raw signature.",
                    v2="❌ NO VOLTAGE", v2_color=RED,
                    fix="Its intellicar feed has signed current (100%) but **batteryVoltage is 100% NULL**, so the "
                        "voltage-endpoint full-cycle test can't run. A SoC-window variant is possible but reintroduces "
                        "BMS-SoC drift. Stuck with the noisy every-session coulomb unless voltage gets logged."),
}


@st.cache_data(show_spinner=False)
def load_oem(key):
    p = f"data/{key}/features/feature_table.parquet"
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p); d["vin"] = d["vin"].astype(str)
    mc = "month" if "month" in d.columns else ("ymd" if "ymd" in d.columns else None)
    if mc:
        d["month"] = pd.to_datetime(d[mc].astype(str), errors="coerce")
    return d


@st.cache_data(show_spinner=False)
def diagnose(key, eol):
    F = load_oem(key)
    if F is None or "soh" not in F.columns:
        return None
    per = []
    for vin, g in F.groupby("vin"):
        g = g.sort_values("month") if "month" in g.columns else g.sort_values("age_months")
        s = pd.to_numeric(g["soh"], errors="coerce").dropna()
        if len(s) < 2:
            continue
        per.append(dict(vin=vin, drop=float(s.iloc[0] - s.iloc[-1]), pin=bool((s >= 99.5).all()),
                        months=len(s), last=float(s.iloc[-1])))
    P = pd.DataFrame(per)
    clip_pct = float((pd.to_numeric(F["soh"], errors="coerce") >= 99.95).mean()) * 100
    try:
        summ = soh_audit.summary(F, eol)
    except Exception:
        summ = dict(n=len(P), clean=np.nan, tainted=np.nan, cliff=np.nan, stuck=np.nan, iso=np.nan)
    return dict(n=int(len(P)), flat=int((P["drop"] < 1.5).sum()),
                flat_pct=round((P["drop"] < 1.5).mean() * 100) if len(P) else 0,
                pinned=int(P["pin"].sum()), clip_pct=round(clip_pct, 1), summ=summ, P=P)


def card_chart(fig, height=340, legend=False):
    fig.update_layout(height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTE, size=12),
                      margin=dict(l=20, r=20, t=28, b=38), showlegend=legend,
                      legend=dict(orientation="h", x=0, y=1.05, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, bordercolor=LINE))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, color=MUTE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, color=MUTE)
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================================
st.title("🩺 Flatness — investigate & fix (Euler · Bajaj · Piaggio)")
st.caption("Why does each OEM's State-of-Health come out flat, is it the same cause as Mahindra, and can it be "
           "fixed? One page for all three. Baseline: Mahindra/Piaggio flatness = noisy coulomb → normalize → "
           "clip-at-100 → isotonic-envelope collapse.")

DIAG = {name: diagnose(m["key"], m["eol"]) for name, m in OEMS.items()}

# ---- 1) diagnosis cards --------------------------------------------------------------------
st.subheader("① Diagnosis — same cause as Mahindra, or different?")
cols = st.columns(3)
for col, (name, m) in zip(cols, OEMS.items()):
    d = DIAG[name]
    with col:
        with st.container(border=True):
            st.markdown(f"### {name}")
            st.markdown(f"<span class='pill'></span>**SoH method:** {m['method']}", unsafe_allow_html=True)
            st.markdown(f"<div style='margin:6px 0;'><span style='background:{m['cause_color']}26;color:{m['cause_color']};"
                        f"padding:2px 8px;border-radius:8px;font-size:0.85rem;'>● {m['cause']}</span></div>",
                        unsafe_allow_html=True)
            if d:
                a, b = st.columns(2)
                a.metric("Flat vehicles", f"{d['flat']}/{d['n']}", f"{d['flat_pct']:.0f}%")
                b.metric("Rows at 100% clip", f"{d['clip_pct']:.1f}%")
                s = d["summ"]
                st.caption(f"Artifacts (soh_audit): CLIFF **{s.get('cliff','–')}** · STUCK_FLOOR "
                           f"**{s.get('stuck','–')}** · ISO_FLOOR **{s.get('iso','–')}**  ·  pinned-at-100: {d['pinned']}")
            st.markdown(f"<div style='color:{MUTE};font-size:0.9rem;margin-top:6px;'>{m['mech']}</div>",
                        unsafe_allow_html=True)

# ---- 2) artifact + flat comparison ---------------------------------------------------------
st.subheader("② How the flatness splits — artifacts vs genuine")
b1, b2 = st.columns(2)
with b1:
    fig = go.Figure()
    names = list(OEMS)
    for lab, key, color in [("clean", "clean", GREEN), ("CLIFF", "cliff", RED),
                            ("STUCK_FLOOR", "stuck", AMBER), ("ISO_FLOOR", "iso", PURPLE)]:
        fig.add_trace(go.Bar(name=lab, x=names, y=[(DIAG[n]["summ"].get(key) or 0) if DIAG[n] else 0 for n in names],
                             marker_color=color))
    fig.update_layout(barmode="stack", yaxis_title="vehicles")
    card_chart(fig, 360, legend=True)
    st.caption("Artifact taint per OEM. **ISO_FLOOR** (envelope pinned below the recovered raw) needs `soh_raw`, "
               "persisted only for Piaggio (and Mahindra) — Piaggio trips it 61×, the Mahindra signature. Euler shows "
               "CLIFF + STUCK_FLOOR (BMS re-estimation jumps). Bajaj is near-clean — its flatness isn't an artifact.")
with b2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="flat %", x=list(OEMS), y=[DIAG[n]["flat_pct"] if DIAG[n] else 0 for n in OEMS],
                         marker_color=BLUE))
    fig.add_trace(go.Bar(name="rows at 100% clip", x=list(OEMS), y=[DIAG[n]["clip_pct"] if DIAG[n] else 0 for n in OEMS],
                         marker_color=AMBER))
    fig.update_layout(barmode="group", yaxis_title="%")
    card_chart(fig, 360, legend=True)
    st.caption("Euler & Piaggio: high flat-% and ~18–24% of rows pinned at exactly 100 — the clip+envelope signature. "
               "Bajaj: low flat-% and ~0% clip — flatness there is the BMS's coarse 1-pp quantization, not our pipeline.")

# ---- 3) flat-curve explorer ----------------------------------------------------------------
st.subheader("③ See it — a vehicle's SoH curve (and what's underneath)")
ce1, ce2 = st.columns([0.24, 0.76])
with ce1:
    oem = st.radio("OEM", list(OEMS), index=0)
d = DIAG[oem]; F = load_oem(OEMS[oem]["key"])
if d is None or F is None:
    st.info("No feature table for this OEM.")
else:
    P = d["P"].copy()
    P["kind"] = np.where(P["pin"], "pinned at 100", np.where(P["drop"] < 1.5, "flat (low)", "declining"))
    order = P.sort_values(["drop"])           # flattest / most-artifact first
    with ce2:
        vin = st.selectbox("Vehicle", order["vin"].tolist(),
                           format_func=lambda v: f"{v} · {order.loc[order.vin==v,'kind'].iloc[0]} "
                                                 f"· drop {order.loc[order.vin==v,'drop'].iloc[0]:.1f}pp")
    g = F[F["vin"] == vin].sort_values("month" if "month" in F.columns else "age_months")
    ageyr = pd.to_numeric(g["age_months"], errors="coerce") / 12
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ageyr, y=pd.to_numeric(g["soh"], errors="coerce"), mode="lines+markers",
                             name="SoH (pipeline)", line=dict(color=GREEN, width=2.6), marker=dict(size=5)))
    if "soh_raw" in g.columns and pd.to_numeric(g["soh_raw"], errors="coerce").notna().any():
        fig.add_trace(go.Scatter(x=ageyr, y=pd.to_numeric(g["soh_raw"], errors="coerce"), mode="markers",
                                 name="soh_raw (pre-envelope)", marker=dict(size=6, color=AMBER, opacity=0.7)))
    fig.add_hline(y=OEMS[oem]["eol"], line=dict(color=RED, dash="dot"),
                  annotation_text=f"EoL {OEMS[oem]['eol']}%", annotation_font_color=RED)
    lay = dict(xaxis_title="battery age (years)", yaxis=dict(title="SoH (%)"))
    if "capacity_ah" in g.columns and pd.to_numeric(g["capacity_ah"], errors="coerce").notna().sum() >= 3:
        fig.add_trace(go.Scatter(x=ageyr, y=pd.to_numeric(g["capacity_ah"], errors="coerce"), mode="lines+markers",
                                 name="raw capacity (Ah)", line=dict(color=MUTE, width=1.4, dash="dot"),
                                 marker=dict(size=3), yaxis="y2"))
        lay["xaxis"] = dict(title="battery age (years)", domain=[0, 0.92])
        lay["yaxis2"] = dict(title="capacity (Ah)", color=MUTE, overlaying="y", side="right", showgrid=False)
    fig.update_layout(**lay)
    card_chart(fig, 380, legend=True)
    _has_raw = "soh_raw" in g.columns
    st.caption(("**Green = pipeline SoH · amber = soh_raw (pre-envelope) · grey = raw coulomb capacity.** Where the "
                "amber/grey scatter is noisy but green is flat, the isotonic envelope has collapsed noise into a flat "
                "line — the artifact." if _has_raw else
                "Green = pipeline SoH. This OEM doesn't persist `soh_raw`, so the pre-envelope signal can't be overlaid "
                "here — but the flat/clip stats above still show the envelope+clip signature (Euler) or the BMS's "
                "reported quantization (Bajaj)."))

# ---- 4) fixability / the fix ---------------------------------------------------------------
st.subheader("④ Can it be fixed? — and how")
for name, m in OEMS.items():
    with st.container(border=True):
        c1, c2 = st.columns([0.16, 0.84])
        c1.markdown(f"**{name}**")
        c1.markdown(f"<span style='background:{m['v2_color']}26;color:{m['v2_color']};padding:2px 8px;"
                    f"border-radius:8px;font-size:0.82rem;'>v2 full-charge: {m['v2']}</span>", unsafe_allow_html=True)
        c2.markdown(m["fix"])
st.info("**Bottom line.** Euler & Piaggio share Mahindra's coulomb-noise + envelope-collapse cause; Bajaj's flatness "
        "is the BMS's own coarse quantization (nothing to fix in our pipeline). The one actionable fix is **Euler**: "
        "its dense feed has the signed current + voltage that v2's full-charge method needs, so porting the "
        "voltage-windowed capacity there is the clear next build.")
