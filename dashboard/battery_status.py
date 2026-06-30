"""Turno — Battery Health (consumer-facing sample app).

A clean, friendly, owner-facing view of an EV battery's health, degradation and
remaining life — styled after premium battery-status apps (dark UI, big numbers,
monospace section labels), but populated with OUR LFP three-wheeler fleet's real
data and LFP-correct explanations. Plain language, no ML jargon. Completely
separate from the internal/technical dashboards.

Run:
    .venv/bin/streamlit run dashboard/battery_status.py --server.port 8502

Data: data/redshift/{euler,mahindra,bajaj}_featengg.parquet (real monthly data).
Projection: a simple robust √t fit per vehicle, clamped so health never rises with age.

Fleet is LFP across all three OEMs. Unlike NMC/NCA cells, LFP tolerates sitting at
high state-of-charge well — for LFP the dominant calendar-aging stressor is HEAT
(high pack temperature, and high SoC × high temperature together). The "why it
matters" copy below reflects that, and does NOT repeat NMC-style "high SoC is bad".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Repo root as cwd so data/ and src/ resolve regardless of launch dir.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) != os.getcwd():
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        pass
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Per-OEM constants (LFP chemistry across the fleet).
# ---------------------------------------------------------------------------
OEMS = ["Euler", "Mahindra", "Bajaj"]
OEM_KEY = {"Euler": "euler", "Mahindra": "mahindra", "Bajaj": "bajaj"}
EOL = {"Euler": 80.0, "Mahindra": 80.0, "Bajaj": 70.0}
RATED_KM = {"Euler": 120.0, "Mahindra": 80.0, "Bajaj": 178.0}

_FALLBACK_WARR = {"euler": (3, 80000), "mahindra": (3, 120000), "bajaj": (5, 120000)}
try:
    import config as _cfg  # noqa: E402

    FLEET_WARRANTY = dict(getattr(_cfg, "FLEET_WARRANTY", _FALLBACK_WARR))
except Exception:
    FLEET_WARRANTY = _FALLBACK_WARR

# "Pack ran hot" threshold for LFP calendar aging (°C, monthly mean pack temp).
HOT_C = 35.0

# ---------------------------------------------------------------------------
# Dark "premium" palette.
# ---------------------------------------------------------------------------
BG = "#0e1116"
PANEL = "#171b22"
PANEL2 = "#1d222b"
LINE = "#262c36"
TEXT = "#e8edf4"
MUTE = "#8a95a5"
FAINT = "#5b6675"
GREEN = "#34d17f"
AMBER = "#f3b14e"
RED = "#ef5d63"
BLUE = "#5aa9f7"

st.set_page_config(page_title="Battery Health", layout="wide", page_icon="🔋")


# ===========================================================================
# Data loading
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_oem(oem: str) -> pd.DataFrame:
    """Load + tidy one OEM's monthly battery data, gated for data-thin vehicles."""
    key = OEM_KEY[oem]
    df = pd.read_parquet(f"data/redshift/{key}_featengg.parquet")
    df = df.rename(columns={"ymd": "month"})
    df["vin"] = df["vin"].astype(str)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    for c in ["soh", "age_months", "soc_mean", "frac_soc_high", "frac_soc_low",
              "temp_mean", "temp_max", "temp_p95", "amb_temp_mean", "driveeff_mean",
              "odo_max", "cum_km", "km_month", "cyc_month", "cum_cycles", "cum_ah"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # km_month carries odometer-rollover garbage (values in the millions); cap to
    # a sane three-wheeler monthly ceiling so usage charts/pace aren't blown up.
    if "km_month" in df.columns:
        df.loc[(df["km_month"] < 0) | (df["km_month"] > 15000), "km_month"] = np.nan

    df = df.sort_values(["vin", "month"]).reset_index(drop=True)

    try:
        import data_quality  # noqa: E402

        gate_cols = [c for c in ["vin", "month", "soh", "age_months"] if c in df.columns]
        kept = data_quality.apply_quality(df[gate_cols].copy(), oem)
        df = df[df["vin"].isin(kept["vin"].unique())].copy()
    except Exception:
        pass

    return df


@st.cache_data(show_spinner=False)
def vehicle_index(oem: str) -> pd.DataFrame:
    """One row per vehicle with stats used to rank demo-worthy vehicles."""
    df = load_oem(oem)
    rows = []
    for vin, g in df.groupby("vin"):
        g = g.dropna(subset=["soh"])
        n = len(g)
        if n == 0:
            continue
        drop = float(g["soh"].head(3).mean() - g["soh"].tail(3).mean())
        rows.append(dict(vin=vin, n_months=n, last_soh=float(g["soh"].iloc[-1]),
                         drop=drop if np.isfinite(drop) else 0.0,
                         age=float(g["age_months"].max()) if "age_months" in g else np.nan))
    idx = pd.DataFrame(rows)
    if idx.empty:
        return idx
    idx["demo_score"] = (idx["n_months"].clip(upper=24)
                         + idx["drop"].clip(lower=0, upper=25) * 1.5)
    return idx.sort_values("demo_score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fleet_stat(oem: str, col: str):
    """Fleet median + 10/90 pct for a column (for 'vs fleet' comparisons)."""
    df = load_oem(oem)
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 10:
        return None
    return dict(med=float(s.median()), lo=float(s.quantile(0.10)),
                hi=float(s.quantile(0.90)), values=s.to_numpy())


# ===========================================================================
# √t projection (robust, clamped so health can't rise with age)
# ===========================================================================
def _theilsen(x: np.ndarray, y: np.ndarray):
    n = len(x)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(n)
              for j in range(i + 1, n) if x[j] != x[i]]
    if not slopes:
        return 0.0, float(np.median(y))
    s = float(np.median(slopes))
    return s, float(np.median(y - s * x))


def project(age_months, soh, eol: float) -> dict | None:
    """soh ≈ a − b·√age, clamped non-increasing, projected forward to EoL."""
    age = np.asarray(age_months, dtype=float)
    soh = np.asarray(soh, dtype=float)
    m = np.isfinite(age) & np.isfinite(soh)
    age, soh = age[m], soh[m]
    if len(age) < 3:
        return None
    x = np.sqrt(np.clip(age, 0, None))
    slope, inter = _theilsen(x, soh)
    if slope > 0:
        slope, inter = 0.0, float(np.median(soh))

    def fit(t):
        t = np.asarray(t, dtype=float)
        return inter + slope * np.sqrt(np.clip(t, 0, None))

    now_age = float(age.max())
    if slope < -1e-9:
        root = (eol - inter) / slope
        t_eol = root * root if root > 0 else 0.0
        months_to_eol = max(0.0, t_eol - now_age)
    else:
        months_to_eol = None
    return dict(fit=fit, slope=slope, inter=inter, now_age=now_age,
                now_soh=float(fit(now_age)), months_to_eol=months_to_eol)


# ===========================================================================
# Formatting + UI helpers
# ===========================================================================
def health_status(soh: float, eol: float):
    margin = soh - eol
    if margin <= 2:
        return "Needs attention", RED, "●"
    if margin <= 8:
        return "Aging normally", AMBER, "●"
    return "Healthy", GREEN, "●"


def fmt_age(months: float) -> str:
    if not np.isfinite(months):
        return "—"
    yrs, mo = int(months // 12), int(round(months - (months // 12) * 12))
    if mo == 12:
        yrs, mo = yrs + 1, 0
    if yrs and mo:
        return f"{yrs} yr {mo} mo"
    return f"{yrs} yr" if yrs else f"{mo} mo"


def fmt_months_human(months):
    if months is None or not np.isfinite(months):
        return "—"
    return "now" if months <= 0 else fmt_age(months)


def section(label: str, title: str):
    """Tesla-style: small uppercase monospace label + big title."""
    st.markdown(
        f"<div style='margin:34px 0 4px 0;'>"
        f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        f"font-size:0.72rem;letter-spacing:0.16em;color:{FAINT};"
        f"text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{TEXT};margin-top:2px;'>"
        f"{title}</div></div>",
        unsafe_allow_html=True,
    )


def big_number(value: str, unit: str = "", color: str = TEXT):
    return (f"<span style='font-size:2.6rem;font-weight:700;color:{color};'>{value}</span>"
            f"<span style='font-size:1.0rem;color:{MUTE};margin-left:6px;'>{unit}</span>")


def split_bar(frac: float, c_left: str, c_right: str, h: int = 14):
    """Two-tone horizontal split bar (left fraction filled c_left)."""
    p = float(np.clip(frac, 0, 1)) * 100
    return (
        f"<div style='display:flex;height:{h}px;border-radius:7px;overflow:hidden;"
        f"background:{c_right};'>"
        f"<div style='width:{p:.1f}%;background:{c_left};'></div></div>"
    )


def legend_row(color: str, label: str, value: str):
    return (
        f"<div style='display:flex;align-items:center;gap:10px;margin-top:8px;'>"
        f"<span style='width:11px;height:11px;border-radius:3px;background:{color};"
        f"display:inline-block;'></span>"
        f"<span style='color:{MUTE};flex:1;'>{label}</span>"
        f"<span style='color:{TEXT};font-weight:600;'>{value}</span></div>"
    )


def why(text: str, sources: str | None = None):
    s = (f"<div style='color:{FAINT};font-size:0.78rem;margin-top:8px;'>Sources: {sources}</div>"
         if sources else "")
    st.markdown(
        f"<div style='color:{MUTE};font-size:0.95rem;line-height:1.6;margin-top:10px;'>"
        f"{text}</div>{s}", unsafe_allow_html=True,
    )


def hist_with_zone(values, marker, threshold, lower_good=True, unit=""):
    """Distribution histogram with a coloured 'aging zone' past the threshold +
    a dotted marker at this vehicle's value."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return None
    lo, hi = np.percentile(values, 1), np.percentile(values, 99)
    values = values[(values >= lo) & (values <= hi)]
    fig = go.Figure()
    # good side
    fig.add_trace(go.Histogram(
        x=values, nbinsx=28, marker_color=GREEN, opacity=0.55,
        hoverinfo="skip", showlegend=False,
    ))
    # shade the aging zone
    if lower_good:
        fig.add_vrect(x0=threshold, x1=max(hi, marker if marker else hi),
                      fillcolor=AMBER, opacity=0.13, line_width=0)
    else:
        fig.add_vrect(x0=min(lo, marker if marker else lo), x1=threshold,
                      fillcolor=AMBER, opacity=0.13, line_width=0)
    fig.add_vline(x=threshold, line=dict(color=AMBER, width=1.5, dash="dot"))
    if marker is not None and np.isfinite(marker):
        fig.add_vline(x=marker, line=dict(color=TEXT, width=2.5, dash="dot"),
                      annotation_text=f"  you: {marker:.0f}{unit}",
                      annotation_position="top",
                      annotation_font=dict(color=TEXT, size=12))
    fig.update_layout(
        height=190, paper_bgcolor=PANEL, plot_bgcolor=PANEL, bargap=0.06,
        margin=dict(l=8, r=8, t=18, b=8),
        xaxis=dict(gridcolor=LINE, zeroline=False, color=MUTE),
        yaxis=dict(visible=False), showlegend=False,
    )
    return fig


def gauge(soh: float, eol: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(soh, 1),
        number={"suffix": " %", "font": {"size": 54, "color": TEXT}},
        gauge={
            "axis": {"range": [eol - 20, 100], "tickwidth": 1, "tickcolor": FAINT,
                     "tickfont": {"size": 11, "color": MUTE}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": PANEL, "borderwidth": 0,
            "steps": [
                {"range": [eol - 20, eol], "color": "rgba(239,93,99,0.18)"},
                {"range": [eol, eol + 8], "color": "rgba(243,177,78,0.18)"},
                {"range": [eol + 8, 100], "color": "rgba(52,209,127,0.16)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 4}, "thickness": 0.8,
                          "value": eol},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=18, r=18, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


# ===========================================================================
# Global dark styling
# ===========================================================================
st.markdown(
    f"""
    <style>
      .stApp {{ background:{BG}; }}
      .block-container {{ padding-top: 1.8rem; max-width: 1120px; }}
      [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      h1,h2,h3,h4,p,span,label,div {{ color:{TEXT}; }}
      div[data-testid="stMetric"] {{
          background:{PANEL}; border:1px solid {LINE}; border-radius:14px;
          padding:14px 16px;
      }}
      div[data-testid="stMetricLabel"] p {{ color:{MUTE}; font-weight:600; }}
      div[data-testid="stMetricValue"] {{ color:{TEXT}; }}
      .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px;
                padding:20px 22px; }}
      /* st.container(border=True) -> dark card to match .panel */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
          background:{PANEL}; border:1px solid {LINE} !important;
          border-radius:16px; padding:18px 20px;
      }}
      /* metric cards already carry their own panel; flatten them inside a card */
      div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMetric"] {{
          background:transparent; border:none; padding:4px 0;
      }}
      .pill {{ display:inline-block; padding:4px 14px; border-radius:999px;
               font-weight:700; font-size:0.9rem; }}
      div[data-baseweb="select"] > div {{ background:{PANEL2}; border-color:{LINE}; }}
      .stProgress > div > div > div {{ background:{GREEN}; }}
      .stSelectbox label {{ color:{MUTE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Header + picker
# ===========================================================================
hcol1, hcol2 = st.columns([0.72, 0.28])
with hcol1:
    st.markdown(f"<h1 style='margin-bottom:0;'>🔋 Your Battery Health</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTE};margin-top:4px;'>A simple, friendly check-up "
                f"for your electric three-wheeler — by Turno.</p>", unsafe_allow_html=True)
with hcol2:
    oem = st.selectbox("Vehicle brand", OEMS, index=0)

idx = vehicle_index(oem)
if idx.empty:
    st.error("No vehicle data available for this brand right now.")
    st.stop()

demo_vins = idx["vin"].head(8).tolist()
ordered = demo_vins + [v for v in idx["vin"].tolist() if v not in demo_vins]


def _vin_label(v: str) -> str:
    row = idx[idx["vin"] == v]
    star = " ⭐" if v in demo_vins else ""
    if row.empty:
        return v + star
    return f"{v}  ·  {int(row.iloc[0]['n_months'])} mo history{star}"


vcol1, _ = st.columns([0.6, 0.4])
with vcol1:
    vin = st.selectbox("Vehicle (VIN)", ordered, index=0, format_func=_vin_label,
                       help="⭐ = demo vehicles with long history and visible decline.")

df = load_oem(oem)
g = df[df["vin"] == vin].dropna(subset=["soh"]).sort_values("age_months")
if g.empty:
    st.warning("This vehicle doesn't have any battery readings yet.")
    st.stop()

eol, rated = EOL[oem], RATED_KM[oem]
warr_years, warr_km = FLEET_WARRANTY[OEM_KEY[oem]]
warr_months = warr_years * 12


def last_val(frame, col):
    if col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[-1])
    return None


def mean_val(frame, col):
    if col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s):
            return float(s.mean())
    return None


latest = g.iloc[-1]
soh_now = float(latest["soh"])
age_now = float(latest["age_months"]) if np.isfinite(latest.get("age_months", np.nan)) else np.nan
odo = last_val(g, "odo_max") or last_val(g, "cum_km")
cycles = last_val(g, "cum_cycles")
km_month = mean_val(g, "km_month")
if (km_month is None or km_month <= 0) and odo and np.isfinite(age_now) and age_now > 0:
    km_month = odo / age_now

range_now = rated * soh_now / 100.0
status_label, status_color, _ = health_status(soh_now, eol)
proj = project(g["age_months"].to_numpy(), g["soh"].to_numpy(), eol)


# ===========================================================================
# 1) HERO — Battery health gauge
# ===========================================================================
section("BATTERY HEALTH", "How your battery is doing")
hero_l, hero_r = st.columns([0.42, 0.58])
with hero_l:
    st.plotly_chart(gauge(soh_now, eol, status_color), use_container_width=True,
                    config={"displayModeBar": False})
with hero_r:
    st.write("")
    st.markdown(f"<span class='pill' style='background:{status_color}26;color:{status_color};'>"
                f"● {status_label}</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:10px;'>{big_number(f'{soh_now:.0f}', '%', status_color)}"
                f"<span style='color:{MUTE};font-size:1.05rem;margin-left:6px;'>battery health</span></div>",
                unsafe_allow_html=True)
    if proj and proj["months_to_eol"] is not None and soh_now > eol:
        life = f"about <b style='color:{TEXT}'>{fmt_months_human(proj['months_to_eol'])}</b> of healthy life left"
    elif soh_now <= eol:
        life = "this battery has reached its <b>end-of-life</b> health line"
    else:
        life = "holding steady — <b>no meaningful decline</b> detected yet"
    st.markdown(f"<p style='color:{MUTE};font-size:1.1rem;margin-top:14px;'>"
                f"Your {oem} battery is at {soh_now:.0f}% of its original capacity — {life}.</p>",
                unsafe_allow_html=True)

# Metric cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Battery health", f"{soh_now:.0f}%")
c2.metric("Range now", f"{range_now:.0f} km", delta=f"{range_now - rated:.0f} km vs new",
          delta_color="inverse")
c3.metric("Battery age", fmt_age(age_now))
c4.metric("Distance driven", f"{odo:,.0f} km" if odo and odo > 0 else "—")
c5.metric("Charge cycles", f"{cycles:,.0f}" if cycles and np.isfinite(cycles) and cycles > 0 else "—",
          help=None if (cycles and cycles > 0) else "Not reported for this vehicle.")


# ===========================================================================
# 2) STATE OF CHARGE  (LFP-correct: high SoC is tolerated; heat is the stressor)
# ===========================================================================
soc_high = mean_val(g, "frac_soc_high")
if soc_high is not None:
    section("STATE OF CHARGE", "Where you keep the charge")
    sl, sr = st.columns([0.55, 0.45])
    with sl:
        with st.container(border=True):
            pct_high = soc_high * 100
            st.markdown(f"{big_number(f'{pct_high:.0f}', '%', GREEN)}"
                        f"<span style='color:{MUTE};margin-left:8px;'>of the time at a high charge</span>",
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown(split_bar(soc_high, GREEN, PANEL2), unsafe_allow_html=True)
            st.markdown(legend_row(GREEN, "Time at high charge", f"{pct_high:.0f}%")
                        + legend_row(PANEL2, "Time at lower charge", f"{100 - pct_high:.0f}%"),
                        unsafe_allow_html=True)
    with sr:
        why(
            "Your fleet runs <b>LFP (lithium iron phosphate)</b> batteries. Unlike the "
            "nickel-based cells in many cars, LFP is <b>very tolerant of sitting at a high "
            "state of charge</b> — keeping it topped up does little harm, so charge to full "
            "whenever it's convenient. For LFP the bigger ageing driver is <b>heat</b> "
            "(see Temperature below), especially a hot pack <i>while</i> sitting full. "
            "So: charge freely, but try to park and charge in the shade.",
            sources="Turno LFP fleet analysis",
        )


# ===========================================================================
# 3) TEMPERATURE  (the real LFP stressor — give it prominence)
# ===========================================================================
temp_mean_v = mean_val(g, "temp_mean")
if temp_mean_v is not None:
    section("TEMPERATURE", "How hot your pack runs")
    tl, tr = st.columns([0.55, 0.45])
    with tl:
        # % of months the pack ran hot (mean monthly temp >= HOT_C)
        tser = pd.to_numeric(g["temp_mean"], errors="coerce").dropna()
        frac_hot = float((tser >= HOT_C).mean()) if len(tser) else 0.0
        peak = mean_val(g, "temp_p95")
        if peak is None:
            peak = last_val(g, "temp_max")
        hot_color = RED if frac_hot > 0.4 else (AMBER if frac_hot > 0.15 else GREEN)
        with st.container(border=True):
            st.markdown(f"{big_number(f'{temp_mean_v:.0f}', '°C', TEXT)}"
                        f"<span style='color:{MUTE};margin-left:8px;'>typical pack temperature</span>",
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            st.markdown(split_bar(frac_hot, hot_color, PANEL2), unsafe_allow_html=True)
            st.markdown(legend_row(hot_color, f"Time running hot (≥{HOT_C:.0f}°C)", f"{frac_hot * 100:.0f}%")
                        + legend_row(PANEL2, "Time in a comfortable range", f"{(1 - frac_hot) * 100:.0f}%"),
                        unsafe_allow_html=True)
            if peak is not None:
                st.markdown(legend_row(AMBER, "Hottest the pack got", f"{peak:.0f}°C"),
                            unsafe_allow_html=True)
    with tr:
        why(
            "Heat is the <b>number-one ageing stressor for LFP</b> batteries. A pack that "
            "regularly runs hot loses capacity faster — and the effect compounds when it's "
            "hot <i>and</i> sitting at a high charge. Indian operating temperatures make this "
            "the metric to watch. <b>What helps:</b> park and charge in shade, avoid charging "
            "right after a long hot run, and give the pack a few minutes to cool.",
            sources="Turno LFP fleet analysis",
        )
        # distribution vs fleet (where this vehicle sits)
        fs = fleet_stat(oem, "temp_mean")
        if fs is not None:
            fig = hist_with_zone(fs["values"], temp_mean_v, HOT_C, lower_good=True, unit="°C")
            if fig is not None:
                st.markdown(f"<div style='color:{FAINT};font-size:0.8rem;margin-top:6px;'>"
                            f"Your pack temperature vs the whole fleet (amber = hot zone):</div>",
                            unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# 4) USAGE  (monthly km + cycles trend — no time-of-day data, honest substitute)
# ===========================================================================
section("USAGE", "How much you drive and charge")
gm = g.dropna(subset=["age_months"]).copy()
has_km = "km_month" in gm.columns and gm["km_month"].notna().any()
has_cyc = "cyc_month" in gm.columns and gm["cyc_month"].notna().any()

ul, ur = st.columns([0.55, 0.45])
with ul:
    if has_km or has_cyc:
        fig = go.Figure()
        if has_km:
            fig.add_trace(go.Bar(x=gm["age_months"], y=gm["km_month"], name="km / month",
                                 marker_color=BLUE, opacity=0.85))
        if has_cyc:
            fig.add_trace(go.Scatter(x=gm["age_months"], y=gm["cyc_month"],
                                     name="charge cycles / month", yaxis="y2",
                                     mode="lines+markers", line=dict(color=AMBER, width=2)))
        layout = dict(
            height=300, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Battery age (months)", gridcolor=LINE, color=MUTE, zeroline=False),
            yaxis=dict(title="km / month", gridcolor=LINE, color=MUTE, zeroline=False),
            legend=dict(orientation="h", y=1.02, x=0, font=dict(color=MUTE)),
            bargap=0.3,
        )
        if has_cyc:
            layout["yaxis2"] = dict(title="cycles / mo", overlaying="y", side="right",
                                    color=MUTE, showgrid=False)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(f"<div class='panel' style='color:{MUTE};'>Monthly usage trend "
                    f"isn't available for this vehicle yet.</div>", unsafe_allow_html=True)
with ur:
    km_avg = float(gm["km_month"].dropna().mean()) if has_km else None
    cyc_avg = float(gm["cyc_month"].dropna().mean()) if has_cyc else None
    with st.container(border=True):
        if km_avg is not None:
            st.metric("Typical distance / month", f"{km_avg:,.0f} km")
        if odo and odo > 0:
            st.metric("Lifetime distance", f"{odo:,.0f} km")
        if cyc_avg is not None:
            st.metric("Typical charges / month", f"{cyc_avg:,.0f}")
        elif cycles and cycles > 0:
            st.metric("Total charge cycles", f"{cycles:,.0f}")
    note = ("Higher mileage and more frequent charging both add wear over time, but on LFP "
            "they matter less than heat. ")
    if not has_cyc:
        note += "Charge-frequency isn't measured on this vehicle, so we show distance only."
    why(note)


# ===========================================================================
# 5) EFFICIENCY  (only where driveeff_mean exists — Bajaj)
# ===========================================================================
if "driveeff_mean" in g.columns and g["driveeff_mean"].notna().any():
    section("EFFICIENCY", "Your efficiency")
    el, er = st.columns([0.55, 0.45])
    ge = g.dropna(subset=["driveeff_mean", "age_months"])
    eff_now = float(ge["driveeff_mean"].iloc[-1])
    with el:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ge["age_months"], y=ge["driveeff_mean"],
                                 mode="lines+markers", line=dict(color=GREEN, width=2.5),
                                 name="Your efficiency"))
        fs = fleet_stat(oem, "driveeff_mean")
        if fs is not None:
            fig.add_hline(y=fs["med"], line=dict(color=MUTE, width=1.2, dash="dot"),
                          annotation_text="fleet typical",
                          annotation_position="bottom right",
                          annotation_font=dict(color=MUTE, size=11))
        fig.update_layout(height=270, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Battery age (months)", gridcolor=LINE, color=MUTE,
                                     zeroline=False),
                          yaxis=dict(title="efficiency score", gridcolor=LINE, color=MUTE,
                                     zeroline=False),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with er:
        with st.container(border=True):
            st.markdown(big_number(f"{eff_now:.0f}", "", GREEN), unsafe_allow_html=True)
            st.markdown(f"<span style='color:{MUTE};'>current efficiency score</span>",
                        unsafe_allow_html=True)
        why("How efficiently your vehicle turns battery energy into distance. Smoother, "
            "steadier driving keeps this high; lots of hard acceleration and stop-start "
            "traffic pulls it down. It doesn't damage the battery directly, but a higher "
            "score means more range from the same charge.")


# ===========================================================================
# 6) DEGRADATION + PROJECTION CHART
# ===========================================================================
section("LIFESPAN", "How your battery is aging")
chart_end = max(warr_months, age_now if np.isfinite(age_now) else 0,
                (proj["months_to_eol"] + proj["now_age"]) if (proj and proj["months_to_eol"]) else 0)
chart_end = float(min(chart_end + 3, max(warr_months + 6, 90)))

fig = go.Figure()
try:
    fl = df.dropna(subset=["soh", "age_months"]).copy()
    fl["bin"] = (fl["age_months"] / 3).round() * 3
    band = fl.groupby("bin")["soh"].agg(
        lo=lambda s: s.quantile(0.10), hi=lambda s: s.quantile(0.90), n="size").reset_index()
    band = band[band["n"] >= 5].sort_values("bin")
    if len(band) >= 3:
        fig.add_trace(go.Scatter(
            x=list(band["bin"]) + list(band["bin"][::-1]),
            y=list(band["hi"]) + list(band["lo"][::-1]),
            fill="toself", fillcolor="rgba(90,169,247,0.10)", line=dict(width=0),
            hoverinfo="skip", name="Typical for the fleet"))
except Exception:
    pass

fig.add_trace(go.Scatter(x=g["age_months"], y=g["soh"], mode="lines+markers",
                         line=dict(color=status_color, width=3), marker=dict(size=6),
                         name="Your battery"))
if proj is not None:
    xp = np.linspace(proj["now_age"], chart_end, 40)
    fig.add_trace(go.Scatter(x=xp, y=proj["fit"](xp), mode="lines",
                             line=dict(color=status_color, width=2, dash="dash"),
                             name="Projected"))
fig.add_hline(y=eol, line=dict(color=AMBER, width=2, dash="dot"),
              annotation_text=f"End-of-life ({eol:.0f}%)", annotation_position="bottom right",
              annotation_font=dict(color=AMBER, size=12))
fig.add_vline(x=warr_months, line=dict(color=MUTE, width=1.5, dash="dash"),
              annotation_text=f"Warranty ({warr_years} yr)", annotation_position="top left",
              annotation_font=dict(color=MUTE, size=11))
fig.update_layout(height=380, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                  margin=dict(l=10, r=10, t=42, b=10),
                  xaxis=dict(title="Battery age (months)", gridcolor=LINE, color=MUTE, zeroline=False),
                  yaxis=dict(title="Battery health (%)", gridcolor=LINE, color=MUTE,
                             range=[eol - 18, 102]),
                  legend=dict(orientation="h", x=1, xanchor="right", y=1.06, yanchor="bottom",
                              bgcolor="rgba(0,0,0,0)", font=dict(color=MUTE, size=11)),
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# 7) WARRANTY CARD + verdict
# ===========================================================================
section("WARRANTY", "What's next")
time_used_frac = float(np.clip(age_now / warr_months, 0, 1)) if np.isfinite(age_now) else 0.0
km_used_frac = months_to_km_limit = None
if km_month and km_month > 0 and odo is not None:
    km_used_frac = float(np.clip(odo / warr_km, 0, 1))
    months_to_km_limit = max(0.0, (warr_km - odo) / km_month)

time_remaining = max(0.0, warr_months - age_now) if np.isfinite(age_now) else warr_months
if months_to_km_limit is not None:
    eff_remaining = min(time_remaining, months_to_km_limit)
    eff_bound = "distance" if months_to_km_limit < time_remaining else "time"
else:
    eff_remaining, eff_bound = time_remaining, "time"
eff_deadline_age = (age_now if np.isfinite(age_now) else 0.0) + eff_remaining
proj_at_deadline = float(proj["fit"](eff_deadline_age)) if proj is not None else soh_now
survives = proj_at_deadline >= eol

wl, wr = st.columns([0.5, 0.5])
with wl:
    with st.container(border=True):
        st.markdown(f"**Warranty term** — {warr_years} years or {warr_km:,.0f} km "
                    f"(whichever comes first)")
        st.write("")
        st.markdown(f"Time used — **{fmt_age(age_now)}** of {warr_years} years")
        st.progress(time_used_frac)
        if km_used_frac is not None:
            st.markdown(f"Distance used — **{odo:,.0f} km** of {warr_km:,.0f} km")
            st.progress(km_used_frac)
        else:
            st.caption("Distance usage not available for this vehicle.")
        st.write("")
        if eff_remaining > 0:
            st.markdown(f"⏳ About **{fmt_months_human(eff_remaining)}** of warranty left "
                        f"(reaches its {eff_bound} limit first).")
        else:
            st.markdown("⏳ This vehicle is **past its warranty window**.")
with wr:
    with st.container(border=True):
        vcolor = GREEN if survives else AMBER
        verdict = "On track to stay healthy through warranty" if survives else "Worth watching"
        st.markdown(f"<span class='pill' style='background:{vcolor}26;color:{vcolor};'>"
                    f"{'✅' if survives else '👀'} {verdict}</span>", unsafe_allow_html=True)
        st.write("")
        st.metric("Projected health at warranty end", f"{proj_at_deadline:.0f}%",
                  delta=f"{proj_at_deadline - eol:+.0f} pts vs end-of-life",
                  delta_color="normal" if survives else "inverse")
        if survives:
            st.markdown(f"At its current ageing pace, your battery should still be around "
                        f"**{proj_at_deadline:.0f}%** when the warranty ends — comfortably above "
                        f"the {eol:.0f}% end-of-life line.")
        else:
            st.markdown(f"At its current ageing pace, your battery may approach the "
                        f"**{eol:.0f}%** end-of-life line near the end of warranty. Keep an eye on "
                        f"range and book a check-up if it drops noticeably.")


# ===========================================================================
# FRIENDLY ONE-SENTENCE SUMMARY
# ===========================================================================
if soh_now <= eol:
    summary = (f"Your battery has reached its end-of-life health line ({soh_now:.0f}%). It still "
               f"runs, but expect noticeably shorter range — a battery check-up is recommended.")
    bg = RED
elif status_label == "Healthy" and survives:
    summary = (f"Your battery is healthy and ageing normally — at {soh_now:.0f}% health and "
               f"projected to stay above the warranty threshold. Nothing to worry about. 🎉")
    bg = GREEN
elif survives:
    summary = (f"Your battery is ageing at a normal pace ({soh_now:.0f}% health) and is projected "
               f"to stay healthy through the warranty. Keep charging and driving as usual.")
    bg = GREEN
else:
    summary = (f"Your battery is at {soh_now:.0f}% health and ageing a little faster than typical. "
               f"It's worth watching as you approach the end of warranty.")
    bg = AMBER

st.write("")
st.markdown(f"<div class='panel' style='border-left:5px solid {bg};font-size:1.15rem;"
            f"line-height:1.55;'>💬 {summary}</div>", unsafe_allow_html=True)
st.write("")
st.markdown(f"<div style='color:{FAINT};font-size:0.82rem;'>Turno · Sample customer "
            f"battery-health view. Health, range and projections are estimates from your "
            f"vehicle's monthly battery data and may vary with real-world use.</div>",
            unsafe_allow_html=True)
