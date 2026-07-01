#!/usr/bin/env python3
"""Mahindra NATIVE-feed data-exploration dashboard — understand the feed BEFORE modeling.

The native feed carries no pack current / voltage / reported-SoH, so it can't be coulomb-counted. This app
explores what it DOES carry (soc, odometer, distanceToEmpty, vehicleStatus, ...), how much data each vehicle
has, and whether the only SoH-like signal — a distance-per-SoC range proxy — shows any real degradation.

Runs off precomputed summaries (src/native_explore_prep.py) + the proxy SoH (src/mahindra_native_soh.py),
so there is no raw 23M-row load at runtime.
Run: streamlit run dashboard/native_explorer.py --server.port 8503
"""
import os
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Mahindra Native Feed Explorer", layout="wide")
TEAL, AMBER, RED, GREEN, GREY, BLUE = "#1f9e8f", "#e0922b", "#d4504e", "#2ec16b", "#9fb3c8", "#5aa9f7"
AX = dict(gridcolor="#1c2738", zerolinecolor="#1c2738", color="#8aa0b6", linecolor="#27374e")


def lay(**k):
    return dict(paper_bgcolor="#0e1726", plot_bgcolor="#0e1726", font=dict(color="#c7d2e0", size=12),
                margin=dict(l=52, r=18, t=40, b=38), **k)


D = "data/mahindra"


@st.cache_data
def load(name):
    p = f"{D}/{name}"
    return pd.read_parquet(p) if os.path.exists(p) else None


summ, monthly, proxy = load("native_vehicle_summary.parquet"), load("native_vehicle_monthly.parquet"), load("native_monthly_soh.parquet")

st.title("🔍 Mahindra Native Feed — Data Exploration")
st.caption("Understand the native-only fleet **before** modeling. This feed has **no pack current, voltage, or "
           "reported SoH** — so SoH could only ever be a distance-per-SoC *range proxy*. The question this app "
           "answers: **is there any usable degradation signal in here at all?**")

if summ is None:
    st.error("Run `python src/native_explore_prep.py` first to build the summaries.")
    st.stop()

# ── 1. Overview ──
st.header("1 · Overview")
c = st.columns(5)
c[0].metric("Vehicles", f"{summ.vin.nunique():,}")
c[1].metric("≥6 months", f"{(summ.n_months >= 6).sum():,}")
c[2].metric("≥12 months", f"{(summ.n_months >= 12).sum():,}")
c[3].metric("Median mo/veh", int(summ.n_months.median()))
c[4].metric("Median odo", f"{summ.odo_max.median():,.0f} km")
st.info("**Dense feed** (~hundreds of rows/file) covering ~the whole fleet — a big change from the old "
        "'70M tiny files' picture. But **no electrical SoH signal**: no current → no coulomb, no voltage → no "
        "OCV, no reported SoH. Usable history is **2025-01 → 2026-07** (~18 mo).")

# ── 2. Fields / schema migration ──
st.header("2 · What the feed carries — and the 2024→2025 schema migration")
FBY = pd.DataFrame([
    ("soc", "✅ 100%", "✅ 100%", "SoC % — usable"),
    ("odometer", "✅ 100%", "✅ 100%", "km — usable"),
    ("distanceToEmpty", "✅ 100%", "✅ 100%", "BMS's own range estimate"),
    ("vehicleStatus", "❌ 0%", "✅ 100%", "NEW: CHARGING / IDLE / DRIVING"),
    ("vehicleMode / vehicleSpeed", "❌ 0%", "✅ 100%", "NEW"),
    ("batteryTemp", "✅ 100%", "❌ 0%", "LOST after 2024 (thermal proxy gone)"),
    ("kwh (charge energy)", "⚠️ 52%", "❌ 0%", "LOST after 2024 (charge-energy proxy gone)"),
    ("current / voltage / batterySoh", "❌", "❌", "never present → no coulomb / OCV / reported SoH"),
], columns=["field", "2024", "2025–26", "note"])
st.table(FBY.set_index("field"))
st.warning("The feed **migrated**: it dropped `batteryTemp` & `kwh` (old thermal + charge-energy proxies) and "
           "gained `vehicleStatus`. Since 2024 is only ~2 months / a handful of vehicles, the practical signals "
           "are **soc + odometer + distanceToEmpty + vehicleStatus** from 2025 on.")

# ── 3. Data availability ──
st.header("3 · Data availability per vehicle")
c1, c2 = st.columns(2)
f = go.Figure(go.Histogram(x=summ.n_months, marker_color=TEAL, nbinsx=20))
f.update_xaxes(title="months of data per vehicle", **AX); f.update_yaxes(title="vehicles", **AX)
f.update_layout(**lay(height=320, title="Coverage — months per vehicle"))
c1.plotly_chart(f, use_container_width=True)
f2 = go.Figure(go.Histogram(x=summ.odo_max.clip(0, 150000), marker_color=BLUE, nbinsx=30))
f2.update_xaxes(title="odometer (km)", **AX); f2.update_yaxes(title="vehicles", **AX)
f2.update_layout(**lay(height=320, title="Fleet odometer spread"))
c2.plotly_chart(f2, use_container_width=True)
st.caption(f"**{(summ.n_months >= 6).sum():,} of {len(summ):,}** vehicles have ≥6 months — the modelable subset "
           f"*if* a signal existed. Most carry real mileage (median {summ.odo_max.median():,.0f} km).")

# ── 4. The signal question ──
st.header("4 · Is there a degradation signal? — the distance-per-SoC proxy")
if proxy is not None:
    ch = proxy.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
    fig = go.Figure(go.Histogram(x=ch.clip(-30, 30), marker_color=GREY, nbinsx=40))
    fig.add_vline(x=0, line=dict(color=AMBER, dash="dash"))
    fig.update_xaxes(title="net proxy change over window (pp) — negative = 'degraded', positive = 'improved'", **AX)
    fig.update_yaxes(title="vehicles", **AX)
    fig.update_layout(**lay(height=360, title="Per-vehicle net change in the range proxy"))
    st.plotly_chart(fig, use_container_width=True)
    down, up, flat = int((ch >= 2).sum()), int((ch <= -2).sum()), int((ch.abs() < 2).sum())
    st.error(f"**Symmetric = noise, not aging.** {down} vehicles' proxy fell ≥2pp — but **{up} *rose* ≥2pp** and "
             f"{flat} are flat. That's a near coin-flip centred on 0. A real SoH signal would be strongly skewed "
             "left (most vehicles declining). **The native distance proxy has no usable degradation signal** — "
             "driving/seasonal efficiency noise (±10–20%) swamps the ~0.1–0.3 %/mo of real fade. Native-only "
             "vehicles should stay on an **age prior**, not be modeled as SoH.")
else:
    st.info("Proxy not built yet — run `python src/mahindra_native_soh.py`.")

# ── 5. Vehicle drill-down ──
st.header("5 · Vehicle drill-down")
top = summ.sort_values("n_rows", ascending=False).head(400).copy()
lbl = {r.vin: f"{r.vin} · {int(r.n_months)} mo · {r.odo_max:,.0f} km · SoC {r.soc_min:.0f}–{r.soc_max:.0f}%"
       for r in top.itertuples()}
vin = st.selectbox("Pick a vehicle (top 400 by data volume)", top.vin.tolist(), format_func=lambda v: lbl[v])
g = monthly[monthly.vin == vin].sort_values("month")
row = summ[summ.vin == vin].iloc[0]
m = st.columns(4)
m[0].metric("Months", int(row.n_months)); m[1].metric("Rows", f"{int(row.n_rows):,}")
m[2].metric("Odometer", f"{row.odo_max:,.0f} km"); m[3].metric("Driving %", f"{100*row.frac_driving:.0f}%")
fig = go.Figure()
fig.add_scatter(x=g.month, y=g.soc_mean, name="SoC mean %", line=dict(color=TEAL))
fig.add_scatter(x=g.month, y=g.dte_med, name="DTE median (km)", line=dict(color=BLUE), yaxis="y2")
fig.update_layout(**lay(height=360, title=f"{vin} — SoC & BMS range estimate (DTE) over time",
                        yaxis=dict(title="SoC %", **AX),
                        yaxis2=dict(title="DTE km", overlaying="y", side="right", **AX),
                        xaxis=dict(**AX), legend=dict(orientation="h", y=1.14)))
st.plotly_chart(fig, use_container_width=True)
if proxy is not None and vin in set(proxy.vin):
    pv = proxy[proxy.vin == vin].sort_values("month")
    pf = go.Figure(go.Scatter(x=pv.month, y=pv.soh, line=dict(color=RED)))
    pf.update_yaxes(title="range proxy (norm %)", range=[55, 111], **AX); pf.update_xaxes(**AX)
    pf.update_layout(**lay(height=280, title=f"{vin} — distance-per-SoC proxy (noisy, NOT a real SoH)"))
    st.plotly_chart(pf, use_container_width=True)
st.caption("Driving segments = odometer-up & soc-down. The proxy divides km by %SoC used while driving; it "
           "wobbles with driving efficiency and season — not battery health.")
