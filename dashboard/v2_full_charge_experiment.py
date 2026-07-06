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
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def mpl_scatter(x, y, color, xlabel, ylabel, clabel="cycle #"):
    """Dark-themed matplotlib scatter (PNG) — much lighter than plotly for dense per-sample charge traces."""
    fig_m, ax = plt.subplots(figsize=(5.4, 4.0), dpi=110)
    sca = ax.scatter(x, y, c=color, cmap="viridis", s=6, alpha=0.5, linewidths=0)
    cb = fig_m.colorbar(sca, ax=ax, pad=0.02); cb.set_label(clabel, color=MUTE)
    cb.ax.yaxis.set_tick_params(color=MUTE); plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=MUTE)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig_m.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTE); ax.xaxis.label.set_color(MUTE); ax.yaxis.label.set_color(MUTE)
    ax.grid(True, color="#2a2f3a", linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color("#2a2f3a")
    fig_m.tight_layout()
    return fig_m


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


def apply_full(ev, vstart_max, vend_min):
    """FULL cycle = starts at/below vstart_max V and ends at/above vend_min V (voltage endpoints)."""
    f = ev[(ev["v0"] <= vstart_max) & (ev["v1"] >= vend_min) & ev["cap"].between(CAP_LO, CAP_HI)].copy()
    return f


@st.cache_data(show_spinner="Loading charge voltage–capacity curves…")
def full_charge_curves(vin, vstart_max, vend_min):
    """Per-sample (voltage, cumulative-Ah) inside each FULL charge for one vehicle, from the raw feed.
    FULL = starts ≤ vstart_max V and ends ≥ vend_min V. Returns long df [cycle, voltage, cap_ah, soc] or None."""
    import pyarrow.dataset as pds, pyarrow.compute as pc
    RAW = "data/mahindra/intellicar"
    if not os.path.isdir(RAW):
        return None
    try:
        t = pds.dataset(RAW, format="parquet").to_table(
            columns=["vin", "eventAt", "soc", "current", "batteryVoltage"],
            filter=pc.field("vin") == pc.scalar(str(vin))).to_pandas()
    except Exception:
        return None
    t["t"] = pd.to_datetime(pd.to_numeric(t["eventAt"], errors="coerce"), unit="ms")
    for c in ("soc", "current", "batteryVoltage"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["t", "soc", "current"])
    t = t[t["soc"].between(0, 100) & (t["current"].abs() <= 400)].sort_values("t").reset_index(drop=True)
    if len(t) < 50:
        return None
    dt = t["t"].diff().dt.total_seconds().fillna(0.0); sd = t["soc"].diff()
    med = t.loc[sd > 0, "current"].median(); sign = np.sign(med) if (np.isfinite(med) and med != 0) else 1.0
    chg = (np.sign(t["current"]) == sign) & (t["current"].abs() > 2.0)
    start = chg & ~(chg.shift(1, fill_value=False) & (dt <= 600))
    t["ev"] = start.cumsum()
    rows, cyc = [], 0
    for _, g in t[chg].groupby("ev"):
        if len(g) < 5:
            continue
        gvv = g.loc[g["batteryVoltage"].between(20, 60), "batteryVoltage"]
        if len(gvv) < 3:
            continue
        if not (float(gvv.iloc[:5].median()) <= vstart_max and float(gvv.iloc[-5:].median()) >= vend_min):
            continue                                                    # FULL charges by voltage endpoints
        cyc += 1
        secs = (g["t"] - g["t"].iloc[0]).dt.total_seconds().to_numpy()
        cur = g["current"].abs().to_numpy()
        cah = np.concatenate([[0.0], np.cumsum((cur[1:] + cur[:-1]) / 2 * np.diff(secs) / 3600.0)])
        sub = pd.DataFrame({"cycle": cyc, "voltage": g["batteryVoltage"].to_numpy(),
                            "cap_ah": cah, "soc": g["soc"].to_numpy()})
        sub = sub[sub["voltage"].between(20, 60)]                       # valid pack voltage only
        if len(sub) >= 5:
            rows.append(sub)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    if len(out) > 8000:
        out = out.iloc[:: max(1, len(out) // 8000)]
    return out


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


def mono_decreasing_fit(x, y, deg=3):
    """Least-squares polynomial of degree `deg`, constrained monotone NON-INCREASING on the data range.
    Parametrised y = a0 - Σ_{k≥1} a_k (x-xmin)^k with every a_k ≥ 0, so dy/dx ≤ 0 for x ≥ xmin (the whole
    fitted range). Solved as a bounded linear least-squares. Returns (grid_x, grid_y, coefs) or (None,)*3."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.max() <= x.min():
        return None, None, None
    from scipy.optimize import lsq_linear
    deg = int(min(deg, max(1, len(x) // 4)))                 # fewer points -> lower degree (guard overfit)
    x0 = x - x.min()
    D = np.column_stack([np.ones_like(x0)] + [-(x0 ** k) for k in range(1, deg + 1)])
    lb = np.array([-np.inf] + [0.0] * deg); ub = np.full(deg + 1, np.inf)
    try:
        c = lsq_linear(D, y, bounds=(lb, ub)).x
    except Exception:
        return None, None, None
    gx = np.linspace(x.min(), x.max(), 60); g0 = gx - x.min()
    gy = c[0] - sum(c[k] * (g0 ** k) for k in range(1, deg + 1))
    return gx, gy, c


def fit_with_greying(age, soh, flat_thr=1.5):
    """Ceiling-clip + MAD-residual outlier greying, then a monotone-decreasing fit — the §② treatment,
    reusable. Returns dict(a, s, inl, gx, gy, drop) on the sorted finite points, or None."""
    a = np.asarray(age, float); s = np.asarray(soh, float)
    m = np.isfinite(a) & np.isfinite(s); a, s = a[m], s[m]
    o = np.argsort(a); a, s = a[o], s[o]
    if len(a) < 3:
        return None
    excl = s >= 99.95
    if excl.any():
        excl[int(np.where(excl)[0][0])] = False                 # keep the first ceiling point as the anchor
    idx = np.where(~excl)[0]
    if len(idx) >= 4:
        x, y = a[idx], s[idx]
        sls = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
        sl = float(np.median(sls)) if sls else 0.0
        resid = y - (sl * x + np.median(y - sl * x)); rmad = float(np.median(np.abs(resid - np.median(resid))))
        bad = np.abs(resid - np.median(resid)) > max(3.5 * 1.4826 * rmad, 3.0)
        excl[idx[bad]] = True
    inl = ~excl
    gx = gy = None; drop = np.nan
    if inl.sum() >= 3:
        gx, gy, _ = mono_decreasing_fit(a[inl], s[inl], deg=3)
        if gx is not None:
            drop = float(gy[0] - gy[-1])
    return dict(a=a, s=s, inl=inl, gx=gx, gy=gy, drop=drop)


def render_euler():
    """Euler view: BMS full-capacity SoH (the real fix — coulomb was noisier) with monotone fit + outlier greying,
    versus the flattening production envelope. Reads src/euler_bms_soh.py artifacts."""
    P, SP, RP = "data/euler/bms_soh.parquet", "data/euler/bms_soh_summary.parquet", "data/euler/bms_soh_report.json"
    if not os.path.exists(P):
        st.warning("Euler artifacts not found — run `.venv/bin/python src/euler_bms_soh.py` first.")
        return
    M = pd.read_parquet(P); M["vin"] = M["vin"].astype(str)
    _rep = json.load(open(RP)) if os.path.exists(RP) else {}
    st.caption("**Euler — BMS full-capacity SoH.** We tested v2's coulomb full-charge here and it was *noisier* "
               "than Euler's own BMS reading (10.8% vs 5.7% raw). So the fix isn't coulomb — it's to take "
               "`batteryRemainingCapacity` at near-full SoC and apply the light **monotone fit + outlier-greying** "
               "instead of the heavy isotonic envelope + 100-clip that flattens it into cliffs and stuck floors.")
    FLAT = 1.5
    comp, fits = [], {}
    for vin, g in M.groupby("vin"):
        g = g.sort_values("age_months")
        f = fit_with_greying(g["age_months"].values, g["soh_full"].values)
        if f is None or f["gx"] is None:
            continue
        fits[vin] = (g, f)
        sp = pd.to_numeric(g["soh_prod"], errors="coerce").dropna()
        dprod = float(sp.iloc[0] - sp.iloc[-1]) if len(sp) > 1 else np.nan
        comp.append(dict(vin=vin, drop_prod=dprod, drop_new=f["drop"],
                         flat_prod=bool(np.isfinite(dprod) and dprod < FLAT), flat_new=bool(f["drop"] < FLAT)))
    C = pd.DataFrame(comp)
    h = st.columns(4)
    h[0].metric("vehicles", f"{_rep.get('vehicles', len(C))}")
    h[1].metric("within-month reading noise", f"{_rep.get('cv_raw_reading_median', '–')}%",
                help="near-full BMS capacity is very reliable within a month")
    h[2].metric("flat — production envelope", f"{int(C['flat_prod'].sum())}/{len(C)}" if len(C) else "–")
    h[3].metric("flat — BMS-cap monotone fit", f"{int(C['flat_new'].sum())}/{len(C)}" if len(C) else "–")
    if len(C):
        st.subheader("① Fleet — production envelope vs BMS-capacity monotone fit")
        mx = float(max(C["drop_prod"].max(), C["drop_new"].max(), 4))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTE, dash="dash", width=1),
                                 name="agree", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=C["drop_prod"], y=C["drop_new"], mode="markers", text=C["vin"], name="vehicle",
                                 marker=dict(size=7, color=GREEN, opacity=0.7),
                                 hovertemplate="%{text}<br>prod %{x:.1f}pp · fit %{y:.1f}pp<extra></extra>"))
        fig.add_hline(y=FLAT, line=dict(color=AMBER, dash="dot")); fig.add_vline(x=FLAT, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(xaxis_title="fade — production envelope (pp)", yaxis_title="fade — BMS-cap monotone fit (pp)")
        card(fig, 340)
        st.caption("Below the diagonal = the production envelope reports *more* fade than the clean BMS-capacity fit "
                   "(its noise-driven cliffs). The monotone fit on the reliable BMS reading is the honest curve.")
    st.subheader("② Vehicle explorer — the fix on one battery")
    order = (C.assign(g=(C["drop_prod"] - C["drop_new"]).abs()).sort_values("g", ascending=False)
             if len(C) else pd.DataFrame({"vin": list(fits)}))
    vin = st.selectbox("Euler vehicle", order["vin"].tolist(),
                       format_func=lambda v: (f"{v} · prod {order.loc[order.vin==v,'drop_prod'].iloc[0]:.1f}pp · "
                                              f"fit {order.loc[order.vin==v,'drop_new'].iloc[0]:.1f}pp") if "drop_prod" in order else v)
    g, f = fits[vin]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["age_months"] / 12, y=pd.to_numeric(g["soh_prod"], errors="coerce"),
                             mode="lines+markers", name="production SoH (envelope)",
                             line=dict(color=MUTE, width=2.4), marker=dict(size=4)))
    a, s, inl = f["a"], f["s"], f["inl"]
    fig.add_trace(go.Scatter(x=a[inl] / 12, y=s[inl], mode="markers", name="BMS full-capacity SoH",
                             marker=dict(size=8, color=GREEN)))
    if (~inl).any():
        fig.add_trace(go.Scatter(x=a[~inl] / 12, y=s[~inl], mode="markers", name="excluded (cliff/outlier)",
                                 marker=dict(size=8, color="#4a5160", symbol="x", opacity=0.85)))
    if f["gx"] is not None:
        _yrs = (f["gx"][-1] - f["gx"][0]) / 12.0; _rate = (f["gy"][0] - f["gy"][-1]) / _yrs if _yrs > 0 else 0.0
        fig.add_trace(go.Scatter(x=f["gx"] / 12, y=f["gy"], mode="lines", name=f"monotone fit (−{_rate:.1f} pp/yr)",
                                 line=dict(color=GREEN, width=2.4, dash="dash")))
    fig.update_layout(xaxis_title="battery age (years)", yaxis=dict(title="SoH (%)"))
    card(fig, 380, legend=True)
    st.caption("Grey = Euler's production envelope (prone to cliffs + stuck floors). Green = the BMS full-capacity "
               "reading with cliff/outlier points greyed and a monotone-decreasing fit — a cleaner, artifact-free "
               "SoH from the *same* underlying signal. Same fix, different signal source than Mahindra.")


# ============================================================================================
st.title("🔋 V2 · Full-charge SoH experiment")
_oem = st.sidebar.radio("OEM", ["Mahindra · coulomb full-charge", "Euler · BMS full-capacity"], index=0)
if _oem.startswith("Euler"):
    render_euler()
    st.stop()
# -------------------------------------- Mahindra (intellicar) --------------------------------------
_ck = os.path.getmtime(EV_P) if os.path.exists(EV_P) else 0     # bust the cache when a checkpoint rewrites artifacts
ev, summ, dcir, rep, fe = load(_ck)
st.caption("**V2 of the SoH pipeline** (Mahindra intellicar). Instead of estimating capacity from every "
           "logging session (v1), measure it only on **full charge events** — deep, single-direction charges "
           "ending near 100% — and watch the coulomb noise collapse.")

if ev is None or ev.empty:
    st.warning("Artifacts not found yet. Run `.venv/bin/python src/full_charge_soh.py` "
               "(≈15 min; scans 95 vehicles' raw telemetry). This page reads its output.")
    st.stop()

# ---- sidebar controls: the "full charge" definition is tunable -----------------------------
st.sidebar.header("What counts as a *full* charge?")
st.sidebar.caption("Full cycles are defined by **pack-voltage endpoints** — drift-free, unlike BMS SoC.")
vstart_max = st.sidebar.slider("Start voltage ≤ (V)", 51.0, 54.5, 53.0, 0.1,
                               help="A full cycle must START at/below this pack voltage — a deep, low-SoC start.")
vend_min = st.sidebar.slider("End voltage ≥ (V)", 54.0, 56.5, 55.0, 0.1,
                             help="…and END at/above this — into the CV knee, near full. Higher = cleaner, fewer.")
min_full = st.sidebar.slider("Fleet view: min full charges/vehicle", 1, 15, 3, 1)
st.sidebar.caption(f"Loaded **{ev['vin'].nunique()}** vehicles · **{len(ev):,}** charge events. "
                   "Every event stores its start/end voltage, so these thresholds re-filter live.")
if "v0" not in ev.columns:
    st.warning("The loaded artifacts predate voltage-based full cycles (no `v0`/`v1`). Re-run "
               "`.venv/bin/python src/full_charge_soh.py` to regenerate them.")
    st.stop()

FULL = apply_full(ev, vstart_max, vend_min)
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
    allv = ev[ev["vin"] == vin]                                   # every charge event for this vin
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
        m = gv.dropna(subset=["age_months", "soh_full"]).sort_values("age_months").copy()
        # (1) ceiling clip: soh_full is capped at 100, so those points are censored — keep only the FIRST
        #     100% point (the age-0 anchor) and exclude the rest so they don't flatten the trend.
        ceil = (m["soh_full"] >= 99.95).to_numpy()
        m["excluded"] = ceil.copy()
        if ceil.any():
            m.iloc[np.where(ceil)[0][0], m.columns.get_loc("excluded")] = False    # keep first 100%
        # (2) MAD-on-residuals outliers among the remaining (non-ceiling) points
        pool = m[~m["excluded"]]
        if len(pool) >= 4:
            x = pool["age_months"].to_numpy(); y = pool["soh_full"].to_numpy()
            sls = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
            sl = float(np.median(sls)) if sls else 0.0
            resid = y - (sl * x + np.median(y - sl * x))
            rmad = float(np.median(np.abs(resid - np.median(resid))))
            thr = max(3.5 * 1.4826 * rmad, 3.0)
            m.loc[pool.index[np.abs(resid - np.median(resid)) > thr], "excluded"] = True
        inl = m[~m["excluded"]]; out = m[m["excluded"]]
        fig.add_trace(go.Scatter(x=inl["age_months"] / 12, y=inl["soh_full"], mode="markers",
                                 name="full-charge SoH", marker=dict(size=9, color=GREEN)))
        if len(out):
            fig.add_trace(go.Scatter(x=out["age_months"] / 12, y=out["soh_full"], mode="markers",
                                     name=f"excluded · ceiling/outlier ({len(out)})",
                                     marker=dict(size=9, color="#4a5160", symbol="x", opacity=0.85)))
        if len(inl) >= 3:                                             # optimised monotone-decreasing polynomial
            gx, gy, _ = mono_decreasing_fit(inl["age_months"].to_numpy(), inl["soh_full"].to_numpy(), deg=3)
            if gx is not None:
                yrs = (gx[-1] - gx[0]) / 12.0
                rate = (gy[0] - gy[-1]) / yrs if yrs > 0 else 0.0
                fig.add_trace(go.Scatter(x=gx / 12, y=gy, mode="lines",
                                         line=dict(color=GREEN, width=2.4, dash="dash"),
                                         name=f"monotone fit (−{rate:.1f} pp/yr avg)"))
    fig.update_layout(xaxis_title="battery age (years)", yaxis_title="SoH (%)")
    card(fig, 360)
    st.caption("Production SoH (grey) is the robust envelope of the noisy session capacity — often pinned flat. "
               "**Full-charge SoH (green)** is measured from clean full charges. **Grey ✕ = excluded**: the "
               "100%-clipped points (only the first is kept, as the anchor) plus MAD residual outliers. The dashed "
               "line is an **optimised monotone-decreasing polynomial** fit on the inliers.")

# per-full-charge signatures: voltage vs capacity + voltage vs SoC, coloured by cycle --------
st.markdown("**Charge signatures per full charge** — pack voltage vs charged capacity, and voltage vs SoC, "
            "coloured by cycle number (dark = early → bright = recent). As the pack ages the curves drift.")
cc = full_charge_curves(vin, vstart_max, vend_min)
if cc is None or cc.empty:
    st.info("Raw within-charge telemetry couldn't be loaded for this vehicle, or it has no full charges at the "
            "current thresholds.")
else:
    vc1, vc2 = st.columns(2)
    with vc1:
        with st.container(border=True):
            f1 = mpl_scatter(cc["cap_ah"], cc["voltage"], cc["cycle"], "charged capacity (Ah)", "pack voltage (V)")
            st.pyplot(f1, use_container_width=True); plt.close(f1)
        st.caption(f"**Voltage vs charged capacity** · {int(cc['cycle'].nunique())} full charges. At a given voltage, "
                   "later cycles get there with fewer Ah as capacity fades (ICA-adjacent).")
    with vc2:
        with st.container(border=True):
            f2 = mpl_scatter(cc["soc"], cc["voltage"], cc["cycle"], "state of charge (%)", "pack voltage (V)")
            st.pyplot(f2, use_container_width=True); plt.close(f2)
        st.caption("**Voltage vs SoC** — the pseudo-OCV shape; drift across cycles reflects rising internal "
                   "resistance (LFP's flat plateau limits the detail). First load ~10 s, then cached.")

# ============================================================================================
st.subheader("③ Threshold sensitivity — cleaner vs how many charges you keep")
sweep = []
for ve in [54.0, 54.5, 55.0, 55.5, 56.0]:
    f = apply_full(ev, vstart_max, ve)
    per = f.groupby("vin").agg(n=("cap", "size"), c=("cap", cv))
    per = per[per["n"] >= min_full]
    sweep.append(dict(vend=ve, median_cv=float(per["c"].median()) if len(per) else np.nan,
                      median_n=float(per["n"].median()) if len(per) else np.nan,
                      vehicles=int(len(per))))
sw = pd.DataFrame(sweep)
tc1, tc2 = st.columns(2)
with tc1:
    fig = go.Figure(go.Scatter(x=sw["vend"], y=sw["median_cv"], mode="lines+markers",
                               line=dict(color=GREEN, width=2.4), marker=dict(size=7)))
    fig.add_vline(x=vend_min, line=dict(color=BLUE, dash="dot"))
    fig.update_layout(xaxis_title="end-voltage threshold ≥ (V)", yaxis_title="median full-charge noise (CV %)")
    card(fig, 320, legend=False)
    st.caption("A higher end-voltage (deeper into the CV knee) → cleaner capacity.")
with tc2:
    fig = go.Figure(go.Scatter(x=sw["vend"], y=sw["median_n"], mode="lines+markers",
                               line=dict(color=AMBER, width=2.4), marker=dict(size=7)))
    fig.add_vline(x=vend_min, line=dict(color=BLUE, dash="dot"))
    fig.update_layout(xaxis_title="end-voltage threshold ≥ (V)", yaxis_title="median full charges / vehicle")
    card(fig, 320, legend=False)
    st.caption("...but higher also means fewer charges qualify. The blue line marks your current setting.")

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
st.subheader("⑤ SoH curves grouped by battery age")
BANDS = [(0.0, 1.5, "~1 yr"), (1.5, 2.5, "~2 yr"), (2.5, 3.5, "~3 yr"), (3.5, 4.5, "~4 yr"), (4.5, 99.0, "~5 yr")]
FLAT_DROP = 1.5     # monotone-fit total fade (pp) below this over the vehicle's span => "flat"
curves = []
for vin_i, gvi in SOH.groupby("vin"):
    mi = gvi.dropna(subset=["age_months", "soh_full"]).sort_values("age_months")
    if len(mi) < 4:
        continue
    ceil = (mi["soh_full"] >= 99.95).to_numpy(); keep = ~ceil
    if ceil.any():
        keep[int(np.where(ceil)[0][0])] = True                          # keep the first 100% as the anchor
    fitm = mi[keep] if keep.sum() >= 3 else mi
    gx, gy, _ = mono_decreasing_fit(fitm["age_months"].to_numpy(), fitm["soh_full"].to_numpy(), deg=3)
    if gx is None:
        continue
    curves.append(dict(vin=vin_i, age_last=float(mi["age_months"].max()) / 12.0,
                       gx=gx / 12.0, gy=gy, drop=float(gy[0] - gy[-1])))
if not curves:
    st.info("Not enough per-vehicle full-charge SoH curves at the current thresholds.")
else:
    bandc = {i: [] for i in range(5)}; flatc = []
    for c in curves:
        if c["drop"] < FLAT_DROP:
            flatc.append(c); continue
        for i, (lo, hi, _lbl) in enumerate(BANDS):
            if lo <= c["age_last"] < hi:
                bandc[i].append(c); break
    titles = [f"{BANDS[i][2]} · {len(bandc[i])} veh" for i in range(5)] + [f"Flat · {len(flatc)} veh"]
    ymin = min(float(np.min(c["gy"])) for c in curves)
    yrng = [max(80.0, ymin - 2), 100.6]
    sp = make_subplots(rows=2, cols=3, subplot_titles=titles, vertical_spacing=0.14, horizontal_spacing=0.055)
    pos = {0: (1, 1), 1: (1, 2), 2: (1, 3), 3: (2, 1), 4: (2, 2)}
    for i in range(5):
        r, cpos = pos[i]
        for c in bandc[i]:
            sp.add_trace(go.Scatter(x=c["gx"], y=c["gy"], mode="lines", line=dict(color=GREEN, width=1.3),
                                    opacity=0.55, showlegend=False, hoverinfo="skip"), row=r, col=cpos)
    for c in flatc:
        sp.add_trace(go.Scatter(x=c["gx"], y=c["gy"], mode="lines", line=dict(color="#8b949e", width=1.2),
                                opacity=0.5, showlegend=False, hoverinfo="skip"), row=2, col=3)
    sp.update_yaxes(range=yrng, gridcolor=LINE, color=MUTE)
    sp.update_xaxes(gridcolor=LINE, color=MUTE)
    sp.update_xaxes(title_text="battery age (years)", row=2)
    sp.update_layout(height=540, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTE, size=11),
                     margin=dict(l=24, r=16, t=42, b=36))
    sp.for_each_annotation(lambda a: a.update(font=dict(color=MUTE, size=12)))
    with st.container(border=True):
        st.plotly_chart(sp, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Each line is one vehicle's **monotone-decreasing full-charge SoH fit**, grouped by how much "
               f"battery-age history it has. Degrading vehicles fill the age panels; **flat vehicles** (modelled "
               f"fade < {FLAT_DROP:.1f} pp over their span — **{len(flatc)}** of them) show no measurable "
               "degradation from full charges and are drawn separately. Empty panels (e.g. ~4–5 yr) = no vehicles "
               "that old in the intellicar cohort.")

# ============================================================================================
st.subheader("⑥ Method comparison — does full charge fix flatness & degradation?")
FLAT2 = 1.5
comp = []
for v, gvi in SOH.groupby("vin"):
    mi = gvi.dropna(subset=["age_months", "soh_full"]).sort_values("age_months")
    if len(mi) < 3:
        continue
    ceil = (mi["soh_full"] >= 99.95).to_numpy(); keep = ~ceil
    if ceil.any():
        keep[int(np.where(ceil)[0][0])] = True
    fm = mi[keep] if keep.sum() >= 3 else mi
    gx, gy, _ = mono_decreasing_fit(fm["age_months"].to_numpy(), fm["soh_full"].to_numpy(), deg=3)
    if gx is None:
        continue
    drop_new = float(gy[0] - gy[-1])
    fo = fe[fe["vin"] == v].sort_values("age_months")
    so = pd.to_numeric(fo["soh"], errors="coerce").dropna()
    if len(so) < 3:
        continue
    drop_old = float(so.iloc[0] - so.iloc[-1])                       # old enveloped-SoH fade
    capo = pd.to_numeric(fo["capacity_ah"], errors="coerce")
    co = capo.rolling(3, min_periods=1, center=True).median()
    b = co[fo["age_months"].between(0.5, 10)].median(); b = b if (np.isfinite(b) and b > 0) else co.median()
    dr = mono_decreasing_fit(fo["age_months"].to_numpy(), np.clip(100 * co / b, None, 100).to_numpy(), deg=3)
    drop_oldrefit = float(dr[1][0] - dr[1][-1]) if dr[0] is not None else np.nan
    comp.append(dict(vin=v, drop_old=drop_old, drop_new=drop_new, drop_oldrefit=drop_oldrefit,
                     flat_old=drop_old < FLAT2, flat_new=drop_new < FLAT2))
C = pd.DataFrame(comp)
if not len(C):
    st.info("Not enough vehicles with both an old SoH history and full charges at these thresholds.")
else:
    cvs2 = float(fleet["cv_session"].median()); cvf2 = float(fleet["cv_full"].median())
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Capacity noise — session → full", f"{cvs2:.0f}% → {cvf2:.0f}%",
               help="Lower = more accurate measurement")
    mc2.metric("Flat — OLD enveloped SoH", f"{int(C['flat_old'].sum())}/{len(C)}")
    mc3.metric("Flat — full-charge SoH", f"{int(C['flat_new'].sum())}/{len(C)}")
    mc4.metric("Flat→declining rescued by full charge", f"{int((C['flat_old'] & ~C['flat_new']).sum())}")
    cc1, cc2 = st.columns([0.55, 0.45])
    with cc1:
        mx = float(max(C["drop_old"].max(), C["drop_new"].max(), 4))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTE, dash="dash", width=1),
                                 name="agree", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=C["drop_old"], y=C["drop_new"], mode="markers", text=C["vin"], name="vehicle",
                                 marker=dict(size=8, color=GREEN, opacity=0.7),
                                 hovertemplate="%{text}<br>old %{x:.1f}pp · full-charge %{y:.1f}pp<extra></extra>"))
        fig.add_hline(y=FLAT2, line=dict(color=AMBER, dash="dot"))
        fig.add_vline(x=FLAT2, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(xaxis_title="degradation seen by OLD method (pp)",
                          yaxis_title="degradation seen by FULL CHARGE (pp)")
        card(fig, 360)
        st.caption("Each dot a vehicle. Points **below the diagonal** = the old method reports *more* fade than the "
                   "clean full-charge measurement — usually **noise-driven fake degradation** (envelope step-downs). "
                   "Amber lines = the 1.5 pp flat threshold.")
    with cc2:
        r = C["drop_oldrefit"].corr(C["drop_new"])
        resc = int((C["flat_old"] & (C["drop_oldrefit"] < FLAT2)).sum()); nfo = int(C["flat_old"].sum())
        st.markdown(f"""
**What the comparison says**

- **Accuracy ✅** — capacity noise **{cvs2:.0f}% → {cvf2:.0f}%** (~{cvs2/max(cvf2,0.1):.0f}× cleaner). The gain is in the
  *measurement*, not the fit.
- **Flatness — reframed.** Full charge shows **more** flat vehicles ({int(C['flat_new'].sum())} vs {int(C['flat_old'].sum())}),
  not fewer. The old method wasn't over-flattening — it was inventing **fake degradation** from noise. Most of this
  young fleet is genuinely near-flat.
- **Can't fix it by re-fitting the old data** — a better monotone fit on the *old* capacity leaves **{nfo - resc}/{nfo}**
  flats still flat, and its per-vehicle fade barely tracks the truth (**r ≈ {r:+.2f}**). The old capacity is simply too
  noisy for any fit to recover a ~1–3 pp fade.
""")

    st.markdown("**Fixed window vs per-driver adaptive** — pick each driver's *widest repeatable* voltage window "
                "(≥ 4 of their charges span it) and measure the clean whole-charge capacity in it.")

    def _pick(gg):
        s = gg["v0"].values; e = gg["v1"].values; best = None
        for qa in (0.5, 0.7, 0.9):
            for qb in (0.1, 0.3, 0.5):
                Va = float(np.quantile(s, qa)); Vb = float(np.quantile(e, qb))
                if Vb - Va < 1.0:
                    continue
                nn = int(((s <= Va) & (e >= Vb)).sum())
                if nn < 4:
                    continue
                sc = (Vb - Va) * np.sqrt(nn)
                if best is None or sc > best[0]:
                    best = (sc, Va, Vb)
        return best

    def _drop(gg):
        gg = gg.sort_values("age_months"); base = gg[gg["age_months"].between(0.5, 10)]["cap"]
        c0 = base.median() if len(base) >= 2 else gg["cap"].head(5).median()
        if not (np.isfinite(c0) and c0 > 0):
            return np.nan
        f = fit_with_greying(gg["age_months"].values, np.clip(100 * gg["cap"] / c0, None, 100).values)
        return f["drop"] if (f and f["gx"] is not None) else np.nan

    fxm = adm = fxd = add = rec = 0
    for _v, _g in ev.groupby("vin"):
        if len(_g) < 5:
            continue
        _fx = _g[(_g["v0"] <= vstart_max) & (_g["v1"] >= vend_min)]; _p = _pick(_g)
        _ad = _g[(_g["v0"] <= _p[1]) & (_g["v1"] >= _p[2])] if _p else _g.iloc[0:0]
        _df = _drop(_fx) if len(_fx) >= 3 else np.nan
        _da = _drop(_ad) if len(_ad) >= 3 else np.nan
        if np.isfinite(_df):
            fxm += 1; fxd += int(_df >= 1.5)
        if np.isfinite(_da):
            adm += 1; add += int(_da >= 1.5)
        if np.isfinite(_da) and _da >= 1.5 and (not np.isfinite(_df) or _df < 1.5):
            rec += 1
    a1, a2, a3 = st.columns(3)
    a1.metric(f"declining — fixed ({vstart_max:.0f}/{vend_min:.0f} V)", f"{fxd} / {fxm} meas")
    a2.metric("declining — per-driver adaptive", f"{add} / {adm} meas")
    a3.metric("recovered from flatness", f"{rec}", help="fixed read flat/unmeasurable → adaptive shows real decline")
    st.caption(f"The per-driver window measures more vehicles ({adm} vs {fxm}) and detects real decline in **{add}** of "
               f"them vs {fxd} for the fixed window — **{rec} vehicles recovered from flatness**. The win is the "
               "*selection* (each driver's deepest repeatable window → enough comparable charges to resolve fade), "
               "paired with the clean whole-charge capacity. The fixed-voltage *windowed-Ah* itself was too noisy on "
               "LFP's flat plateau (CV 17–41%), so selection — not a narrower measurement — is what recovers the signal.")

# ============================================================================================
with st.expander("⑦ Method & caveats"):
    st.markdown(f"""
**How a full charge is measured** (from `src/full_charge_soh.py`):
1. **Clean the raw feed** — drop SoC outside [0,100] (the feed has values to 837) and |current| > {CAP_HI:.0f} A
   (sentinels to 65,279 A). The production session method survives these only because its (40,400) Ah bound
   silently discards glitch-derived sessions.
2. **Detect charge events** — contiguous runs where current is charging the pack (sign auto-detected per
   vehicle) with < 10-min gaps; capacity = ∫|I|·dt ÷ (ΔSoC/100).
3. **Keep the *full* ones** — by **voltage endpoints**: start ≤ **{vstart_max:.1f} V** and end ≥ **{vend_min:.1f} V**
   (tunable, left). Voltage endpoints are drift-free (unlike BMS SoC, which recalibrates), so "full" means the
   same physical charge window over the battery's life — the key robustness win of this definition.
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
