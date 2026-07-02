#!/usr/bin/env python3
"""Mahindra NATIVE-feed data-exploration dashboard — understand the feed BEFORE modeling.

The native feed carries no pack current / voltage / reported-SoH, so it can't be coulomb-counted. This app
explores what it DOES carry (soc, odometer, distanceToEmpty, vehicleStatus, ...), how much data each vehicle
has, and whether the only SoH-like signal — a distance-per-SoC range proxy — shows any real degradation.

Runs off precomputed summaries (src/native_explore_prep.py) + the proxy SoH (src/mahindra_native_soh.py),
so there is no raw 23M-row load at runtime.
Run: streamlit run dashboard/native_explorer.py --server.port 8503
"""
import os, json
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
st.caption("A **deep-dive on the 100 longest-availability vehicles** with **COMPLETE data** (all fields, no "
           "file cap — ~95 driving-segments/vehicle/month vs ~11 in the thin sample). The native feed has **no "
           "pack current, voltage, or reported SoH**, so SoH could only ever be a distance-per-SoC *range "
           "proxy*. **Verdict: even at complete resolution, there is no usable degradation signal.**")

# ── Executive summary (verdict up top) ──
with st.container(border=True):
    st.markdown("### 📋 Findings summary — can the native feed give us a Mahindra SoH?")
    st.error("**No — native-only Mahindra SoH is not measurable or validatable from the current feed.** Two real "
             "levers: **(a)** get **current/voltage** into the native stream (direct measurement), or **(b)** let "
             "the fleet **age** and re-run the **charge-rate proxy (§12)** — the cleanest native signal. Nothing "
             "else moves the needle.")
    st.markdown(
        "- **No electrical signal** — the feed carries SoC, odometer, distance-to-empty, status; **no current, "
        "voltage, or reported SoH**. So coulomb / OCV / reported-SoH are all impossible (§1–2).\n"
        "- **Distance-per-SoC proxy = noise** — even on *complete* data (~95 driving-segments/month), the range "
        "proxy is a coin-flip (**40 down / 44 up**) and drifts *up* — not aging (§4, §10).\n"
        "- **DTE is redundant with SoC** — the BMS range estimate ≈ SoC × a fixed constant (**r = 0.92**), so it "
        "carries zero capacity information (§6).\n"
        "- **No significant degradation** — the first-principles fixed-window rate is stable ~1.25 km/%SoC but its "
        "trend's **95% CI spans zero**; searching for a 'better' window is p-hacking (§10).\n"
        "- **The noise can't be controlled** — speed + season explain only **~5%** of the variance; the dominant "
        "confound is unobservable **cargo payload** (§10).\n"
        "- **Can't validate vs ground truth** — against the intellicar **coulomb SoH** the proxy shows **no "
        "significant correlation (r = +0.07)**; even the one vehicle with rich native *and* a 9-month coulomb "
        "overlap (a real 100→89% degrader) has the proxy tracking the **wrong way** (r = −0.60, spurious). The "
        "other four both-feeds vehicles barely co-occur at all (coulomb 2023–24, native 2025–26) (§11).\n"
        "- **Charge-rate is the best remaining lever** — a charger's ~constant current means **%SoC/hr rises as "
        "capacity fades**, and its confound (charger type) is *controllable*: it cuts within-vehicle noise from "
        "**~28% to ~8%** (vs discharge's uncontrollable payload). Still flat on this young fleet, but this is the "
        "**test to re-run as it ages** (§12).")
    st.caption("~12k native-only Mahindra vehicles ride on this feed. This dashboard is the evidence behind the verdict — "
               "the sections below walk through it. **Re-run §12 (charge-rate) as the fleet ages** — it's the cleanest "
               "method; the data isn't there *yet*.")

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
st.info("These 100 are the **richest-history vehicles** in the fleet — **complete data** (no file cap, "
        "~1,700 rows/vehicle/month, ~20 months each). Still **no electrical SoH signal**: no current → no "
        "coulomb, no voltage → no OCV, no reported SoH. The tight, complete monthly estimates below are exactly "
        "what makes the 'no signal' verdict **conclusive** — it's not a sampling artifact.")

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
st.caption(f"All **{len(summ)}** vehicles are high-availability by design (~20 months each — the fleet's "
           f"longest histories). Odometer {summ.odo_max.min():,.0f}–{summ.odo_max.max():,.0f} km "
           f"(median {summ.odo_max.median():,.0f}) — real usage, so if degradation were measurable, it would show here.")

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
    st.error(f"**Symmetric = noise, not aging — and this is COMPLETE data (~95 driving-segments/month).** "
             f"{down} vehicles' proxy fell ≥2pp, but **{up} *rose* ≥2pp** and {flat} are flat: a near coin-flip "
             "centred on 0 (it even drifts slightly *up* — physically impossible for real SoH). The 8× tighter "
             "estimates from complete data **did not** surface a hidden trend, so this is **not a sampling "
             "artifact**. **The native feed cannot measure SoH degradation** — driving/seasonal efficiency noise "
             "(±10–20%) swamps the ~0.1–0.3 %/mo of real fade. Native-only vehicles stay on an **age prior**; the "
             "fix is *signal* (current/voltage in the feed), not more data.")
    st.markdown("**Two more angles, same verdict:**")
    st.markdown("- **The BMS's own full-charge range** (`distanceToEmpty` at high SoC) is also **flat** — median "
                "0.0% change over the window, 55 of 100 vehicles flat. Even the vehicle's own range estimate "
                "shows no degradation.\n"
                "- **Finer time bins don't help.** Binning the *same* complete data **weekly** is noisier "
                "(residual 10.2% of level) than **monthly** (7.7%), with no extra trend — higher-frequency data "
                "would *hurt*, not help.")
    # the 5 highest-availability vehicles, one panel each — every one wobbles, none trends down
    top5 = pd.read_csv("data/manifests/mahindra_native_top100.csv")["vin"].astype(str).head(5).tolist()
    p5 = proxy[proxy.vin.isin(top5)]
    if len(p5):
        f5 = make_subplots(rows=1, cols=len(top5), shared_yaxes=True, horizontal_spacing=0.015,
                           subplot_titles=[f"…{v[-6:]}" for v in top5])
        for i, v in enumerate(top5, start=1):
            pv = p5[p5.vin == v].sort_values("month")
            f5.add_scatter(x=pv.month, y=pv.soh, line=dict(color=RED, width=1.6), row=1, col=i, showlegend=False)
            f5.add_hline(y=100, line=dict(color=GREY, dash="dot"), row=1, col=i)
        f5.update_yaxes(range=[60, 115], **AX); f5.update_xaxes(showticklabels=False, **AX)
        f5.update_layout(**lay(height=260,
                         title="The 5 highest-availability vehicles — distance-per-SoC proxy (each wobbles ±10%, none trends down)"))
        st.plotly_chart(f5, use_container_width=True)
        st.caption("Same story per vehicle as in aggregate: the proxy drifts up and down with driving/season, "
                   "never settling into a decline — there's no capacity signal underneath it.")
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

# ── 6. Raw signal explorer — browse the actual telemetry, full resolution ──
st.header("6 · Raw signal explorer — what the feed actually streams")
st.caption("Browse a single vehicle-day at full resolution. This is the **raw** native feed — SoC, odometer, "
           "the BMS range estimate (DTE), speed, status. Note how coarse (often ~2-min cadence) and sometimes "
           "unreliable it is (`vehicleStatus`/`vehicleSpeed` can be flaky). No current/voltage = no capacity.")


@st.cache_data(show_spinner="Loading raw signals…")
def load_raw():
    import glob
    fs = sorted(glob.glob("data/mahindra/native100/*.parquet"))
    keep = ["vin", "eventAt", "soc", "odometer", "distanceToEmpty", "vehicleSpeed", "vehicleStatus", "vehicleMode"]
    # union columns across files — 2024 files predate vehicleStatus/vehicleMode, so read each file's own subset
    d = pd.concat([pd.read_parquet(f)[[c for c in keep if c in pd.read_parquet(f, columns=None).columns]]
                   for f in fs], ignore_index=True)
    d["t"] = pd.to_datetime(pd.to_numeric(d["eventAt"], errors="coerce"), unit="ms")
    for c in ["soc", "odometer", "distanceToEmpty", "vehicleSpeed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["vin"] = d["vin"].astype(str)
    d = d.dropna(subset=["t"]); d["day"] = d["t"].dt.date
    return d


raw = load_raw()
rc = st.columns(2)
rvin = rc[0].selectbox("Vehicle", sorted(raw.vin.unique()), format_func=lambda v: v[-8:], key="rawvin")
gv = raw[raw.vin == rvin]
days = sorted(gv["day"].unique())
di = rc[1].select_slider("Day", options=list(range(len(days))), value=len(days) // 2,
                         format_func=lambda i: str(days[i])) if len(days) > 1 else 0
gd = gv[gv["day"] == days[di]].sort_values("t")
if len(gd) < 2:
    st.info("No rows that day.")
else:
    cad = gd["t"].diff().dt.total_seconds().median()
    s = st.columns(4)
    s[0].metric("Rows", len(gd)); s[1].metric("SoC swing", f"{gd.soc.min():.0f}→{gd.soc.max():.0f}%")
    s[2].metric("Distance", f"+{gd.odometer.max()-gd.odometer.min():.1f} km"); s[3].metric("Cadence", f"~{cad:.0f}s")
    fig = go.Figure()
    fig.add_scatter(x=gd.t, y=gd.soc, name="SoC %", line=dict(color=TEAL), mode="lines+markers", marker=dict(size=3))
    fig.add_scatter(x=gd.t, y=gd.distanceToEmpty, name="DTE (km)", line=dict(color=BLUE), yaxis="y2")
    if "vehicleSpeed" in gd:
        fig.add_scatter(x=gd.t, y=gd.vehicleSpeed, name="speed", line=dict(color=AMBER), yaxis="y2", opacity=0.55)
    fig.update_layout(**lay(height=380, title=f"…{rvin[-8:]} — raw signals on {days[di]}",
                            yaxis=dict(title="SoC %", **AX),
                            yaxis2=dict(title="DTE km / speed", overlaying="y", side="right", **AX),
                            xaxis=dict(**AX), legend=dict(orientation="h", y=1.12)))
    st.plotly_chart(fig, use_container_width=True)
    if "vehicleStatus" in gd:
        st.caption(f"vehicleStatus mix: {gd.vehicleStatus.value_counts().to_dict()}")


# ── 7. SoC during charging vs discharging events ──
st.header("7 · SoC — charging vs discharging events (all vehicles)")


@st.cache_data(show_spinner=False)
def _charge_discharge():
    r = load_raw().sort_values(["vin", "t"])
    r["dsoc"] = r.groupby("vin")["soc"].diff()
    return r.loc[r.dsoc > 0, "soc"].to_numpy(), r.loc[r.dsoc < 0, "soc"].to_numpy()


chg, dis = _charge_discharge()
f7 = go.Figure()
f7.add_histogram(x=chg, name=f"charging (SoC rising) · {len(chg):,}", marker_color=GREEN, opacity=0.6, nbinsx=50)
f7.add_histogram(x=dis, name=f"discharging (SoC falling) · {len(dis):,}", marker_color=RED, opacity=0.6, nbinsx=50)
f7.update_layout(barmode="overlay", **lay(height=360, title="SoC distribution — charging vs discharging",
                                          legend=dict(orientation="h", y=1.1)))
f7.update_xaxes(title="SoC %", **AX); f7.update_yaxes(title="observations", **AX)
st.plotly_chart(f7, use_container_width=True)
st.caption("Charging = consecutive readings where SoC **rises**; discharging = SoC **falls** (more robust than the "
           "flaky `vehicleStatus`). Shows the SoC band each mode operates over — how deep vehicles run down while "
           "driving, and from what level they charge back up.")

# ── 8. Individual SoC vs time — full timeline ──
st.header("8 · Individual SoC vs time — full timeline (charge green / discharge red)")


@st.cache_data(show_spinner=False)
def _rich_vins(n=6):
    return load_raw().groupby("vin").size().sort_values(ascending=False).head(n).index.astype(str).tolist()


all_vins = sorted(raw.vin.unique())
sel = st.multiselect("Pick vehicles", all_vins, default=_rich_vins(6), format_func=lambda v: v[-8:], max_selections=9)
if not sel:
    st.info("Select at least one vehicle above.")
else:
    nrow = len(sel)
    grid = make_subplots(rows=nrow, cols=2,
                         specs=[[{"secondary_y": True}, {"secondary_y": True}] for _ in range(nrow)],
                         column_titles=["🟢 Charging (SoC↑) + odometer", "🔴 Discharging (SoC↓) + odometer"],
                         row_titles=[f"…{v[-6:]}" for v in sel],
                         vertical_spacing=min(0.06, 1.0 / max(nrow, 2)), horizontal_spacing=0.07)
    for i, v in enumerate(sel):
        g = raw[raw.vin == v].sort_values("t").assign(dsoc=lambda x: x.soc.diff())
        up = g[g.dsoc > 0].iloc[::2]; dn = g[g.dsoc < 0].iloc[::2]; od = g.iloc[::4]
        for col, pts, clr in ((1, up, GREEN), (2, dn, RED)):
            grid.add_scattergl(x=pts.t, y=pts.soc, mode="markers", marker=dict(color=clr, size=2),
                               row=i + 1, col=col, secondary_y=False, showlegend=False)
            grid.add_scattergl(x=od.t, y=od.odometer, mode="lines", line=dict(color=BLUE, width=1),
                               row=i + 1, col=col, secondary_y=True, showlegend=False)
    grid.update_yaxes(range=[0, 105], secondary_y=False, **AX)
    grid.update_yaxes(secondary_y=True, **AX)
    grid.update_xaxes(**AX)
    grid.update_layout(**lay(height=max(320, 210 * nrow),
                       title="Charging (left) vs Discharging (right) — SoC markers + odometer (blue, right axis)"))
    st.plotly_chart(grid, use_container_width=True)
st.caption("One row per vehicle: **charging in column 1, discharging in column 2**. SoC markers (green/red, left "
           "axis) + **odometer** (blue line, right axis) over the full ~20-month timeline. Odometer climbs "
           "steadily while the SoC band stays flat — no degradation signal.")

# ── 9. SoC vs time by operating mode (vehicleStatus) ──
st.header("9 · SoC vs time — one panel per operating mode")
mv = st.selectbox("Vehicle", sorted(raw.vin.unique()), format_func=lambda v: v[-8:], key="modevin")
gm = raw[raw.vin == mv].sort_values("t")
MODES = ["CHARGING", "DRIVING", "IDLE", "DISCONNECTED"]
MCOL = {"CHARGING": GREEN, "DRIVING": RED, "IDLE": GREY, "DISCONNECTED": AMBER}
mgrid = make_subplots(rows=2, cols=2, subplot_titles=[f"{m} · {int((gm.vehicleStatus == m).sum()):,} pts" for m in MODES],
                      vertical_spacing=0.13, horizontal_spacing=0.06)
for i, m in enumerate(MODES):
    r, c = i // 2 + 1, i % 2 + 1
    pts = gm[gm.vehicleStatus == m].iloc[::2]
    mgrid.add_scattergl(x=pts.t, y=pts.soc, mode="markers", marker=dict(color=MCOL[m], size=2), row=r, col=c, showlegend=False)
mgrid.update_yaxes(range=[0, 105], **AX); mgrid.update_xaxes(**AX)
mgrid.update_layout(**lay(height=460, title=f"…{mv[-6:]} — SoC over the full timeline, split by vehicleStatus"))
st.plotly_chart(mgrid, use_container_width=True)
st.caption("SoC over the full timeline, one panel per operating mode. (`vehicleMode` is constant = ECO, so "
           "'mode' = `vehicleStatus`.) CHARGING sits high (topping up), DRIVING spans the discharge range, IDLE "
           "is scattered, DISCONNECTED is the flaky/gappy state flagged earlier.")

# ── 10. First-principles: fixed-SoC-window discharge rate over time ──
st.header("10 · First-principles — is there a degradation signal? (honest verdict)")


@st.cache_data(show_spinner="Extracting fixed-window discharge events…")
def _fixed_window_rate(HI, LO):
    d = load_raw()[["vin", "t", "soc", "odometer"]].dropna().sort_values(["vin", "t"])
    ev = []
    for v, g in d.groupby("vin"):
        soc = g.soc.values; odo = g.odometer.values; t = g.t.values
        if len(soc) < 10:
            continue
        sh = np.empty_like(soc); sh[0] = soc[0]; sh[1:] = soc[:-1]
        dhi = np.where((soc <= HI) & (sh > HI))[0]
        dlo = np.where((soc <= LO) & (sh > LO))[0]
        up = np.where(soc > sh + 3)[0]
        for hi in dhi:
            ua = up[up > hi]; nu = ua[0] if len(ua) else len(soc) + 1
            la = dlo[(dlo > hi) & (dlo < nu)]
            if len(la):
                km = odo[la[0]] - odo[hi]
                if 0 < km < 200:
                    ev.append((v, t[hi], km / (HI - LO)))
    return pd.DataFrame(ev, columns=["vin", "t", "rate"])


E = _fixed_window_rate(90, 20)          # fixed WIDE window (max ΔSoC = least relative noise); NOT tuned to the result
sl = []
for v, g in E.groupby("vin"):
    if len(g) >= 6:
        g = g.sort_values("t")
        x = ((g.t.astype("int64") - g.t.astype("int64").min()) / 8.64e13 / 30.4).values
        sl.append(np.polyfit(x, g.rate.values, 1)[0] / g.rate.mean() * 100)   # per-vehicle %/mo
sl = np.array(sl)
rng = np.random.default_rng(0)
boot = np.array([np.median(rng.choice(sl, len(sl), replace=True)) for _ in range(3000)]) if len(sl) else np.array([0.0])
clo, chi = np.percentile(boot, [2.5, 97.5]); med = float(np.median(sl)) if len(sl) else 0.0
verdict = (f"Fleet degradation slope **{med:+.2f}%/mo**, 95% CI **[{clo:+.2f}, {chi:+.2f}]** · "
           f"{100*np.mean(sl < 0):.0f}% of {len(sl)} vehicles decline.")
if chi < 0:
    st.success("✅ **Significant degradation detected.** " + verdict)
elif clo > 0:
    st.warning("⚠️ Proxy trends *up* (impossible for real SoH) — pure noise. " + verdict)
else:
    st.error("🚫 **No significant degradation signal yet.** " + verdict + " The CI **spans zero** — this fleet's "
             "real ~3–6% fade over 20 months is below the driving-efficiency noise floor.")
# the 5 richest vehicles, individually
top5 = E["vin"].value_counts().head(5).index.tolist()
meta = {}; titles = []
for v in top5:
    ev = E[E.vin == v].sort_values("t")
    x = ((ev.t.astype("int64") - ev.t.astype("int64").min()) / 8.64e13 / 30.4).values
    b = np.polyfit(x, ev.rate.values, 1)
    meta[v] = (ev, b, x); titles.append(f"…{v[-6:]} · {len(ev)} events · trend {b[0] / ev.rate.mean() * 100:+.2f}%/mo")
grid = make_subplots(rows=len(top5), cols=1, subplot_titles=titles, vertical_spacing=0.06)
for i, v in enumerate(top5, start=1):
    ev, b, x = meta[v]
    grid.add_scattergl(x=ev.t, y=ev.rate, mode="markers", marker=dict(color=GREY, size=4, opacity=0.5), row=i, col=1, showlegend=False)
    mm = ev.assign(mon=ev.t.dt.to_period("M").dt.to_timestamp()).groupby("mon")["rate"].median().reset_index()
    grid.add_scatter(x=mm.mon, y=mm.rate, mode="lines+markers", line=dict(color=TEAL, width=2), row=i, col=1, showlegend=False)
    grid.add_scatter(x=[ev.t.min(), ev.t.max()], y=[np.polyval(b, x.min()), np.polyval(b, x.max())],
                     mode="lines", line=dict(color=RED, dash="dash", width=1.5), row=i, col=1, showlegend=False)
grid.update_yaxes(range=[0.6, 2.0], **AX); grid.update_xaxes(**AX)
grid.update_layout(**lay(height=max(320, 230 * len(top5)),
                   title="Fixed 90→20% window · km per %SoC · the 5 richest vehicles, individually"))
st.plotly_chart(grid, use_container_width=True)
st.caption("**Method (no tuning):** fixed *wide* 90→20% window (max ΔSoC), per-vehicle-normalized slope, bootstrap "
           "95% CI. Auto-searching for the 'best' window is **p-hacking** — the slope swings −0.04 to −0.44%/mo "
           "across windows, all inside noise. I also tested **controlling for speed + season**: they explain only "
           "~5% of the variance (the dominant confound is unobservable **cargo payload**), so it doesn't help. "
           "**So: no measurable degradation on this young fleet — re-run this exact test as it ages** (fade will "
           "clear the noise), or add current/voltage to the feed.")

# ── 11. Both-feed coulomb cross-validation ──
st.header("11 · Cross-validation — does the native proxy track REAL (coulomb) SoH?")


@st.cache_data
def load_crossval():
    P = pd.read_parquet("data/mahindra/crossval_pairs.parquet") if os.path.exists("data/mahindra/crossval_pairs.parquet") else None
    C = pd.read_csv("data/mahindra/crossval_coverage.csv", index_col=0) if os.path.exists("data/mahindra/crossval_coverage.csv") else None
    return P, C


P, C = load_crossval()
if P is None:
    st.info("Run `python src/crossval_prep.py` (needs the both-feeds native pull).")
else:
    m = P.nz.notna() & P.cz.notna() & np.isfinite(P.nz) & np.isfinite(P.cz)
    x = P.nz[m].values; y = P.cz[m].values; r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(0)
    boot = [np.corrcoef(x[(i := rng.integers(0, len(x), len(x)))], y[i])[0, 1] for _ in range(2000)]
    clo, chi = np.percentile(boot, [2.5, 97.5])
    a, b, cc = st.columns(3)
    a.metric("Overlap tested", f"{P.vin.nunique()} veh · {len(P)} mo")
    b.metric("Correlation r", f"{r:+.2f}"); cc.metric("95% CI", f"[{clo:+.2f}, {chi:+.2f}]")
    if clo > 0:
        st.success("✅ **Native proxy tracks the coulomb SoH** — the distance proxy is validated for native-only vehicles.")
    elif chi < 0:
        st.warning("⚠️ Native proxy *anti*-correlates with coulomb — not usable.")
    else:
        st.error("🚫 **No significant tracking** (CI spans 0). The native distance-per-SoC proxy does **not** reflect "
                 "the real coulomb SoH on the vehicles that have both.")
    g1, g2 = st.columns(2)
    fig = go.Figure(go.Scattergl(x=P.nz[m], y=P.cz[m], mode="markers", marker=dict(color=GREY, size=5, opacity=0.5)))
    fig.update_xaxes(title="native distance proxy (norm.)", **AX); fig.update_yaxes(title="coulomb SoH (norm.)", **AX)
    fig.update_layout(**lay(height=340, title="If the proxy worked, points would trend ↗"))
    g1.plotly_chart(fig, use_container_width=True)
    if C is not None:
        f2 = go.Figure()
        f2.add_bar(x=C.index, y=C["coulomb"], name="intellicar coulomb", marker_color=TEAL)
        f2.add_bar(x=C.index, y=C["native"], name="native", marker_color=AMBER)
        f2.update_layout(barmode="group", **lay(height=340, title="Why overlap is thin — feeds cover different periods",
                                                legend=dict(orientation="h", y=1.12)))
        f2.update_xaxes(**AX); f2.update_yaxes(title="vehicle-months", **AX)
        g2.plotly_chart(f2, use_container_width=True)
    st.caption("The native distance-per-SoC proxy shows **no significant correlation** with the intellicar coulomb "
               "SoH (ground truth) on the ~23 vehicles that have both. It's barely testable: most of the ~220 "
               "both-feeds vehicles migrated intellicar→native (coulomb 2023–24, native 2025–26) and share **no** "
               "months; only **23 overlap ≥3 months** (median 4), and within those the coulomb is near-flat "
               "(**median 2 distinct SoH values**) — too little dynamic range to correlate. **Bottom line: "
               "native-only Mahindra SoH can't be validated or measured from the current feed — it needs "
               "current/voltage, or fleet aging.**")


# ── 11b. Both-feeds case study — the one vehicle with rich native AND overlapping coulomb ──
st.subheader("Case study — the one vehicle with rich native data AND overlapping coulomb")


@st.cache_data(show_spinner="Extracting charge events…")
def _charge_window_rate(LO, HI):
    """Charge-time-per-%SoC over a fixed CC-phase window: crossing UP through LO then UP through HI inside one
    charge (no discharge between). rate = %SoC per hour. `consistent` flags each vehicle's own usual-charger
    events (within ±25–33% of its median rate) — the charger-type confound is *controllable*, unlike payload."""
    d = load_raw()[["vin", "t", "soc"]].dropna().sort_values(["vin", "t"])
    ev = []
    for v, g in d.groupby("vin"):
        soc = g.soc.values; ts = g.t.values.astype("datetime64[s]").astype("int64"); tt = g.t.values
        if len(soc) < 10:
            continue
        sh = np.empty_like(soc); sh[0] = soc[0]; sh[1:] = soc[:-1]
        ulo = np.where((soc >= LO) & (sh < LO))[0]     # cross UP through LO
        uhi = np.where((soc >= HI) & (sh < HI))[0]     # cross UP through HI
        dn = np.where(soc < sh - 3)[0]                 # a discharge resets the charge
        for lo in ulo:
            da = dn[dn > lo]; nd = da[0] if len(da) else len(soc) + 1
            ha = uhi[(uhi > lo) & (uhi < nd)]
            if len(ha):
                hrs = (ts[ha[0]] - ts[lo]) / 3600.0
                if 0.15 < hrs < 10:
                    ev.append((v, tt[lo], (HI - LO) / hrs))
    E = pd.DataFrame(ev, columns=["vin", "t", "rate"])
    if len(E):
        E["vmed"] = E.groupby("vin")["rate"].transform("median")
        E["consistent"] = (E.rate / E.vmed).between(0.75, 1.33)
    return E


@st.cache_data(show_spinner="Building both-feeds case study…")
def load_casestudy():
    fe, tp = "data/redshift/mahindra_featengg.parquet", "data/manifests/mahindra_native_top100.csv"
    if not (os.path.exists(fe) and os.path.exists(tp)):
        return None
    cc = pd.read_parquet(fe); cc["vin"] = cc["vin"].astype(str); cc["month"] = pd.to_datetime(cc["ymd"].astype(str))
    top100 = set(pd.read_csv(tp)["vin"].astype(str))
    vins = sorted(top100 & set(cc.vin.unique()))
    if not vins:
        return None
    raw = load_raw(); chg = _charge_window_rate(30, 70)
    rows = []
    for v in vins:
        cs = cc[cc.vin == v][["month", "soh"]].rename(columns={"soh": "coulomb"})
        g = raw[raw.vin == v].sort_values("t").copy()
        g["do"] = g.odometer.diff(); g["ds"] = -g.soc.diff(); g["dm"] = g.t.diff().dt.total_seconds() / 60
        seg = g[g.do.between(0.1, 80) & g.ds.between(0.5, 40) & g.dm.between(0.1, 180)].copy()
        seg["month"] = seg.t.dt.to_period("M").dt.to_timestamp()
        dps = seg.groupby("month").agg(o=("do", "sum"), sc=("ds", "sum"), n=("do", "size"))
        dps = dps[dps.n >= 3].copy(); dps["native_km_soc"] = 100 * dps.o / dps.sc
        dps = dps[dps.native_km_soc.between(20, 400)][["native_km_soc"]]
        cr = chg[(chg.vin == v) & chg.consistent] if len(chg) else chg
        if len(cr):
            crm = cr.assign(month=cr.t.dt.to_period("M").dt.to_timestamp()).groupby("month")["rate"].median().rename("native_charge_rate")
        else:
            crm = pd.Series(dtype=float, name="native_charge_rate")
        merged = cs.set_index("month").join(dps, how="outer").join(crm, how="outer").reset_index()
        merged["vin"] = v
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    for col in ["coulomb", "native_km_soc", "native_charge_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")   # guard object-dtype from all-NaN outer joins
    return out


CS = load_casestudy()
if CS is None or not len(CS):
    st.info("Case study needs `data/redshift/mahindra_featengg.parquet` + the native100 pull.")
else:
    def _novlp(v):
        d = CS[CS.vin == v]
        return int((d.coulomb.notna() & d.native_km_soc.notna()).sum())
    vins = [v for v in sorted(CS.vin.unique(), key=_novlp, reverse=True) if CS[CS.vin == v].coulomb.notna().any()]
    star = vins[0]
    ds = CS[CS.vin == star]
    ov = ds[ds.coulomb.notna() & ds.native_km_soc.notna()]
    rr = float(ov.coulomb.corr(ov.native_km_soc)) if len(ov) >= 3 else float("nan")
    tag = "n/a" if np.isnan(rr) else ("wrong sign" if rr < 0 else ("too weak" if rr < 0.3 else "tracks"))
    desc = ("**can't be computed** (too few overlap months)" if np.isnan(rr) else
            ("the **wrong sign** and spurious" if rr < 0 else
             ("**too weak to be usable**" if rr < 0.3 else "positive — worth a closer look")))
    cser = ds.dropna(subset=["coulomb"]).sort_values("month")
    k1, k2, k3 = st.columns(3)
    k1.metric(f"…{star[-8:]} coulomb", f"{cser.coulomb.iloc[0]:.0f}→{cser.coulomb.iloc[-1]:.0f}%",
              help="A real degrader per the intellicar coulomb ground truth")
    k2.metric("Overlap window", f"{_novlp(star)} months")
    k3.metric("Proxy vs truth (r)", f"{rr:+.2f}", delta=tag,
              delta_color="inverse" if (np.isnan(rr) or rr < 0.3) else "normal")
    st.error(f"Of the **{len(vins)} vehicles** with both rich native data and a coulomb SoH, only "
             f"**…{star[-8:]}** overlaps in time — and it's a genuine degrader (coulomb "
             f"{cser.coulomb.iloc[0]:.0f}→{cser.coulomb.iloc[-1]:.0f}%). Yet across the {_novlp(star)}-month overlap "
             f"the native distance proxy correlates **r = {rr:+.2f}** with the truth — {desc} "
             "(coulomb steps only ~2.5pp in-window, buried under proxy noise). The other "
             f"{len(vins)-1} vehicles have **zero** temporal overlap: their coulomb is 2023–24, native 2025–26.")
    titles = []
    for v in vins:
        d0 = CS[CS.vin == v].dropna(subset=["coulomb"])
        titles.append(f"…{v[-8:]} · coulomb {d0.coulomb.iloc[0]:.0f}→{d0.coulomb.iloc[-1]:.0f}% · overlap {_novlp(v)} mo")
    cg = make_subplots(rows=len(vins), cols=1, specs=[[{"secondary_y": True}] for _ in vins],
                       subplot_titles=titles, vertical_spacing=0.08)
    for i, v in enumerate(vins, start=1):
        d = CS[CS.vin == v].sort_values("month")
        cl = d.dropna(subset=["coulomb"]); km = d.dropna(subset=["native_km_soc"]); ch = d.dropna(subset=["native_charge_rate"])
        cg.add_scatter(x=cl.month, y=cl.coulomb, mode="lines+markers", line=dict(color=BLUE, width=2),
                       marker=dict(size=5), row=i, col=1, secondary_y=False, showlegend=(i == 1), name="coulomb SoH (truth)")
        cg.add_scatter(x=km.month, y=km.native_km_soc, mode="lines+markers", line=dict(color=AMBER, dash="dot"),
                       marker=dict(size=4), row=i, col=1, secondary_y=True, showlegend=(i == 1), name="native km/%SoC")
        cg.add_scatter(x=ch.month, y=ch.native_charge_rate, mode="lines+markers", line=dict(color=GREEN, dash="dot"),
                       marker=dict(size=4), row=i, col=1, secondary_y=True, showlegend=(i == 1), name="native %SoC/hr")
        ovd = d[d.coulomb.notna() & (d.native_km_soc.notna() | d.native_charge_rate.notna())]
        if len(ovd):
            cg.add_vrect(x0=ovd.month.min(), x1=ovd.month.max(), fillcolor=AMBER, opacity=0.10, line_width=0, row=i, col=1)
    cg.update_yaxes(title_text="coulomb %", secondary_y=False, **AX)
    cg.update_yaxes(title_text="native", secondary_y=True, **AX)
    cg.update_xaxes(**AX)
    cg.update_layout(**lay(height=max(360, 200 * len(vins)),
                     title="Coulomb SoH (truth, blue) vs native proxies — feeds barely co-occur; only the top vehicle overlaps",
                     legend=dict(orientation="h", y=1.06)))
    st.plotly_chart(cg, use_container_width=True)
    st.caption("Blue = intellicar coulomb SoH (ground truth, left axis); amber/green = native proxies (right axis); "
               "shaded = the both-feeds overlap. The cohort migrated intellicar→native around end-2024, so coulomb "
               "lives in 2023–24 and native in 2025–26 — **they don't line up**. The single overlap case has the "
               "native proxies **not tracking** the real decline. This is the visual proof that native SoH can't be "
               "*validated* against coulomb with today's data — it needs the feeds to overlap, or the fleet to age.")


# ── 12. Charge-rate capacity proxy — the cleanest native signal ──
st.header("12 · Charge-rate capacity proxy — the cleanest native signal (re-run as the fleet ages)")
st.caption("The discharge proxy (§10) is killed by **unobservable payload**. Charging is different: a charger "
           "delivers ~constant current, so **%SoC-per-hour rises as capacity fades** (constant current fills a "
           "smaller pack faster) — and its one big confound, **charger type, is controllable**: filter to each "
           "vehicle's own usual charger and the noise collapses. This is the most promising native method.")

CE = _charge_window_rate(30, 70)
if not len(CE):
    st.info("No charge events extracted — need the native100 pull.")
else:
    CEc = CE[CE.consistent]
    raw_cv = float(CE.groupby("vin")["rate"].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan).median() * 100)
    con_cv = float(CEc.groupby("vin")["rate"].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan).median() * 100)
    slc = []
    for v, g in CEc.groupby("vin"):
        if len(g) >= 6:
            g = g.sort_values("t")
            x = ((g.t.astype("int64") - g.t.astype("int64").min()) / 8.64e13 / 30.4).values
            slc.append(np.polyfit(x, g.rate.values, 1)[0] / g.rate.mean() * 100)
    slc = np.array(slc); rng = np.random.default_rng(0)
    bootc = np.array([np.median(rng.choice(slc, len(slc), replace=True)) for _ in range(3000)]) if len(slc) else np.array([0.0])
    lo2, hi2 = np.percentile(bootc, [2.5, 97.5]); med2 = float(np.median(slc)) if len(slc) else 0.0
    q1, q2, q3 = st.columns(3)
    q1.metric("Charge events (30→70%)", f"{len(CE):,}", help=f"{CE.vin.nunique()} vehicles")
    q2.metric("Noise: raw → consistent-charger", f"{raw_cv:.0f}% → {con_cv:.0f}%",
              delta=f"{con_cv-raw_cv:.0f} pp", delta_color="inverse",
              help="Median within-vehicle coefficient of variation. Controlling for charger type collapses it.")
    q3.metric("Fleet slope 95% CI (%/mo)", f"[{lo2:+.2f}, {hi2:+.2f}]", help=f"median {med2:+.2f}%/mo · n={len(slc)}")
    if lo2 > 0:
        st.success(f"✅ **Significant charge-rate rise = capacity fade detected** ({med2:+.2f}%/mo). The native "
                   "charge proxy now measures degradation — validate against coulomb and roll out.")
    elif hi2 < 0:
        st.warning(f"⚠️ Charge rate *falls* over time ({med2:+.2f}%/mo) — unexpected; investigate before use.")
    else:
        st.info(f"🟡 **Cleanest native proxy we have, but no fade signal *yet*.** Controlling for charger type cuts "
                f"within-vehicle noise to **~{con_cv:.0f}%** (vs discharge's uncontrollable ±10–20%), yet the trend "
                f"is flat (**{med2:+.2f}%/mo**, CI [{lo2:+.2f}, {hi2:+.2f}] — spans 0). At ~{con_cv:.0f}% noise we're "
                "right at the detection threshold: this young fleet's ~3–6% fade hasn't cleared it. **This is the "
                "test most likely to fire as the fleet ages — re-run it in ~6–12 months.**")
    top5c = CEc["vin"].value_counts().head(5).index.tolist()
    metac = {}; titlesc = []
    for v in top5c:
        ev = CEc[CEc.vin == v].sort_values("t")
        x = ((ev.t.astype("int64") - ev.t.astype("int64").min()) / 8.64e13 / 30.4).values
        bb = np.polyfit(x, ev.rate.values, 1)
        metac[v] = (ev, bb, x); titlesc.append(f"…{v[-6:]} · {len(ev)} charges · trend {bb[0]/ev.rate.mean()*100:+.2f}%/mo")
    gridc = make_subplots(rows=len(top5c), cols=1, subplot_titles=titlesc, vertical_spacing=0.06)
    for i, v in enumerate(top5c, start=1):
        ev, bb, x = metac[v]
        gridc.add_scattergl(x=ev.t, y=ev.rate, mode="markers", marker=dict(color=GREY, size=4, opacity=0.5), row=i, col=1, showlegend=False)
        mm = ev.assign(mon=ev.t.dt.to_period("M").dt.to_timestamp()).groupby("mon")["rate"].median().reset_index()
        gridc.add_scatter(x=mm.mon, y=mm.rate, mode="lines+markers", line=dict(color=GREEN, width=2), row=i, col=1, showlegend=False)
        gridc.add_scatter(x=[ev.t.min(), ev.t.max()], y=[np.polyval(bb, x.min()), np.polyval(bb, x.max())],
                          mode="lines", line=dict(color=RED, dash="dash", width=1.5), row=i, col=1, showlegend=False)
    gridc.update_yaxes(title_text="%SoC / hr", **AX); gridc.update_xaxes(**AX)
    gridc.update_layout(**lay(height=max(320, 220 * len(top5c)),
                        title="Consistent-charger %SoC/hr over time · the 5 most-charged vehicles (flat = no fade yet)"))
    st.plotly_chart(gridc, use_container_width=True)
    st.caption("**Method:** charge events crossing up through a fixed 30→70% window (CC phase), rate = %SoC/hr, "
               "keep only each vehicle's consistent-charger sessions (within ±25–33% of its median rate), "
               "per-vehicle-normalized slope, bootstrap 95% CI — same rigor as §10. Grey = per-charge; green = "
               "monthly median; red = fitted trend. **Why it beats discharge:** charge current is set by the "
               "charger (controllable), not by unobservable cargo/terrain — so as the fleet ages, this is the "
               "cleanest shot at a native SoH.")


# ── 13. Probable SoH curves — Bayesian, behaviour-conditioned ──
st.header("13 · Probable SoH curves — Bayesian, behaviour-conditioned (the constructive answer)")
st.caption("The proxies don't *measure* SoH — but usage still *drives* degradation. So instead of sensing SoH, we "
           "model the trajectory: a hierarchical **Bayesian** model learns SoH-vs-age on the feeds that HAVE a real "
           "SoH (Euler, Bajaj, Piaggio, Mahindra-intellicar), lets a **native-computable behaviour fingerprint** "
           "(km/month, SoC habits) tilt the degradation *rate*, and posterior-predicts a **probable SoH curve with "
           "credible bands** for every native vehicle — anchored on the Mahindra baseline. Honest by construction: "
           "weak behaviour → wide bands.")


@st.cache_data
def load_bayes():
    pp, rp = "data/mahindra/native_behaviour_soh.parquet", "data/mahindra/behaviour_soh_report.json"
    P = pd.read_parquet(pp) if os.path.exists(pp) else None
    R = json.load(open(rp)) if os.path.exists(rp) else None
    return P, R


BP, BR = load_bayes()
if BP is None or BR is None:
    st.info("Run `python src/behaviour_soh_experiment.py` to build the Bayesian behaviour-conditioned curves.")
else:
    nat = BR["native"]; kms = BR["behaviour_slopes"]["km_month"]; hd = BR["heldout_rate_mae"]; bd = BR["band_decomposition"]
    mc = st.columns(4)
    mc[0].metric("Native median SoH @36mo", f"{nat['soh50_at_36mo_median']:.0f}%")
    mc[1].metric("Credible band @36mo", f"±{nat['band_width_at_36mo_median']/2:.0f} pp", help="half of the p10–p90 width")
    mc[2].metric("km/month effect (+1 SD)", f"{kms['mean']:+.3f} SoH/mo",
                 delta="credible" if kms["credible"] else "n.s.", delta_color="normal" if kms["credible"] else "off")
    mc[3].metric("Behaviour vs OEM-avg (held-out)", f"{hd['behaviour_improvement_pct']:+.1f}%",
                 help="reduction in held-out degradation-rate MAE from adding behaviour over the OEM baseline")

    g1, g2 = st.columns(2)
    ag = BP.groupby("age_months")
    med, p10, p90 = ag.soh_p50.mean(), ag.soh_p10.mean(), ag.soh_p90.mean(); xg = med.index
    f13 = go.Figure()
    f13.add_scatter(x=xg, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip")
    f13.add_scatter(x=xg, y=p10, fill="tonexty", fillcolor="rgba(90,169,247,0.18)", line=dict(width=0), name="fleet p10–p90 band")
    f13.add_scatter(x=xg, y=med, line=dict(color=BLUE, width=3), name="fleet median (p50)")
    for v in list(BP.vin.unique())[:15]:
        d = BP[BP.vin == v]
        f13.add_scatter(x=d.age_months, y=d.soh_p50, line=dict(color=GREY, width=0.6), opacity=0.4, showlegend=False, hoverinfo="skip")
    f13.add_hline(y=80, line=dict(color=RED, dash="dash"))
    f13.update_xaxes(title="age (months)", **AX); f13.update_yaxes(title="SoH %", range=[70, 101], **AX)
    f13.update_layout(**lay(height=380, title="Native probable SoH — fleet median + credible band", legend=dict(orientation="h", y=1.13)))
    g1.plotly_chart(f13, use_container_width=True)

    kmv = BP.groupby("vin").km_month.first()
    hi = set(kmv[kmv > kmv.quantile(.75)].index); lo = set(kmv[kmv < kmv.quantile(.25)].index)
    f14 = go.Figure()
    for grp, col, fill, lab in [(hi, RED, "rgba(212,80,78,0.12)", "heavy usage (top-quartile km/mo)"),
                                (lo, GREEN, "rgba(46,193,107,0.12)", "light usage (bottom-quartile km/mo)")]:
        s = BP[BP.vin.isin(grp)].groupby("age_months"); xx = s.soh_p50.mean().index
        f14.add_scatter(x=xx, y=s.soh_p90.mean(), line=dict(width=0), showlegend=False, hoverinfo="skip")
        f14.add_scatter(x=xx, y=s.soh_p10.mean(), fill="tonexty", fillcolor=fill, line=dict(width=0), showlegend=False, hoverinfo="skip")
        f14.add_scatter(x=xx, y=s.soh_p50.mean(), line=dict(color=col, width=3), name=lab)
    f14.add_hline(y=80, line=dict(color=GREY, dash="dash"))
    f14.update_xaxes(title="age (months)", **AX); f14.update_yaxes(title="SoH %", range=[70, 101], **AX)
    f14.update_layout(**lay(height=380, title="Behaviour tilt — heavy vs light usage (km/month)", legend=dict(orientation="h", y=1.13)))
    g2.plotly_chart(f14, use_container_width=True)

    st.markdown("**What drives the curve — and how much to trust it**")
    cta, ctb = st.columns(2)
    sb = BR["source_baseline_rate"]; ssb = BR.get("source_sigma_b", {})
    cta.caption("Per-OEM baseline rate + between-vehicle spread σ_b (per-source calibrated):")
    cta.table(pd.DataFrame({"baseline (SoH/mo)": {k: round(v, 3) for k, v in sb.items()},
                            "σ_b (SoH/mo)": {k: round(ssb.get(k, float('nan')), 3) for k in sb}}))
    ctb.caption("Behaviour slopes on the degradation rate (per +1 global SD) — only km/month is credible:")
    ctb.table(pd.DataFrame([{"behaviour": k, "slope/+1SD": round(v["mean"], 3),
                             "95% CI": f"[{v['lo']:+.3f}, {v['hi']:+.3f}]", "credible": "✅" if v["credible"] else "—"}
                            for k, v in BR["behaviour_slopes"].items()]).set_index("behaviour"))

    psk = BR.get("per_source_km_rate", {})
    psk_txt = ", ".join(f"**{s} {v['rho']:+.2f}**" for s, v in psk.items())
    st.warning(f"**Honesty on the one real driver (km/month) and the bands.** After fixing the odometer definition, the "
               f"km/month→faster-fade effect is **credible and consistent across 3 of 4 OEMs** (per-source km→rate: "
               f"{psk_txt}) — including **Mahindra-intellicar's own −0.24**, so the native tilt has *same-OEM* support "
               f"rather than being borrowed (Euler is the lone null — its raw odometer is glitchy; Bajaj is strongest "
               f"but its rates are late-window local slopes). **The band is irreducible:** Mahindra between-vehicle "
               f"heterogeneity **σ_b={bd['heterogeneity_sd']:.2f} SoH/mo dominates parameter uncertainty "
               f"{bd['parameter_sd']:.3f}** — vehicles that charge & drive alike still fade differently, and the native "
               f"feed (no temperature/current) can't resolve why.")
    st.success(f"✅ **Constructive takeaway:** every native vehicle gets a **principled probabilistic SoH curve** — the "
               f"Mahindra baseline tilted by its (credible) mileage effect, with a per-source-calibrated "
               f"**±{nat['band_width_at_36mo_median']/2:.0f}pp** band. This native cohort is the *longest-availability "
               f"(highest-usage)* subset (median **{nat.get('km_month_median', 0):.0f} km/mo**), so its median lands at "
               f"a steeper **~{nat['soh50_at_36mo_median']:.0f}% at 36 months**. Behaviour beats the OEM-average by "
               f"**{hd['behaviour_improvement_pct']:+.1f}%** out-of-sample — good for **fleet-level warranty/risk**, not "
               f"a per-vehicle sensor. Best the native feed supports without current/voltage.")
