"""Turno — Battery Health (consumer-facing sample app).

A clean, friendly, owner-facing view of an EV battery's health, degradation and
remaining life — styled after premium battery-status apps (dark UI, big numbers,
monospace section labels), but populated with OUR LFP three-wheeler fleet's real
data and LFP-correct explanations. Plain language, no ML jargon. Completely
separate from the internal/technical dashboards.

Run:
    .venv/bin/streamlit run dashboard/battery_status.py --server.port 8502

Data: data/redshift/{euler,mahindra,bajaj}_featengg.parquet (real monthly data).
Projection: the same conditioned pipeline models as the internal dashboard (Euler trajectory ·
Mahindra expected-loss quantiles · Bajaj quantiles), anchored at the present SoH; falls back to a
robust √t fit for vehicles with too little history for the model.

Fleet is LFP across all three OEMs. Unlike NMC/NCA cells, LFP tolerates sitting at
high state-of-charge well — for LFP the dominant calendar-aging stressor is HEAT
(high pack temperature, and high SoC × high temperature together). The "why it
matters" copy below reflects that, and does NOT repeat NMC-style "high SoC is bad".

Layout: one uniform design system — full-width container, a single stat-card
component, full-bleed charts (no mismatched side-by-side columns), and one chart
styler so every section looks identical and premium.
"""
from __future__ import annotations

import importlib
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
OEMS = ["Euler", "Mahindra", "Bajaj", "Piaggio", "Mahindra Native"]
OEM_KEY = {"Euler": "euler", "Mahindra": "mahindra", "Bajaj": "bajaj",
           "Piaggio": "piaggio", "Mahindra Native": "mahindra_native"}
EOL = {"Euler": 80.0, "Mahindra": 80.0, "Bajaj": 70.0, "Piaggio": 80.0, "Mahindra Native": 80.0}
RATED_KM = {"Euler": 120.0, "Mahindra": 80.0, "Bajaj": 178.0, "Piaggio": 110.0, "Mahindra Native": 120.0}

_FALLBACK_WARR = {"euler": (3, 80000), "mahindra": (3, 120000), "bajaj": (5, 120000),
                  "piaggio": (3, 100000), "mahindra_native": (3, 120000)}
try:
    import config as _cfg  # noqa: E402

    FLEET_WARRANTY = {**_FALLBACK_WARR, **dict(getattr(_cfg, "FLEET_WARRANTY", {}) or {})}
except Exception:
    FLEET_WARRANTY = _FALLBACK_WARR

# "Pack ran hot" threshold for LFP calendar aging (°C, monthly mean pack temp).
HOT_C = 35.0

# ---------------------------------------------------------------------------
# Dark "premium" palette.
# ---------------------------------------------------------------------------
BG = "#0c0f14"
PANEL = "#161a21"
PANEL2 = "#1d222b"
LINE = "#242b36"
TEXT = "#eef2f8"
MUTE = "#8a95a5"
FAINT = "#5b6675"
GREEN = "#34d17f"
AMBER = "#f3b14e"
RED = "#ef5d63"
BLUE = "#5aa9f7"
GREEN_FILL = "rgba(52,209,127,.16)"

st.set_page_config(page_title="Battery Health", layout="wide", page_icon="🔋")


# ===========================================================================
# Data loading
# ===========================================================================
@st.cache_data(show_spinner=False)
def _load_native() -> pd.DataFrame:
    """Mahindra NATIVE fleet — no on-board SoH sensor. Build a per-vehicle ESTIMATED SoH series from the Bayesian
    behaviour model's median (p50) curve (data/mahindra/native_behaviour_soh.parquet), truncated at each vehicle's
    current age. No electrical/behaviour columns, so the SoC/temperature/habit sections auto-skip. soh = predicted p50."""
    p = "data/mahindra/native_behaviour_soh.parquet"
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_parquet(p); d["vin"] = d["vin"].astype(str)
    reg = {}
    try:
        r = pd.read_csv("Mh_Regd_Date.csv"); rv = next(c for c in r.columns if c.lower() == "vin")
        r["rd"] = pd.to_datetime(r["vehicle_registration_date"], errors="coerce")
        reg = dict(zip(r[rv].astype(str), r["rd"]))
    except Exception:
        pass
    rows = []
    for vin, g in d.groupby("vin"):
        g = g.sort_values("age_months")
        la = float(g["last_age"].iloc[0]) if pd.notna(g["last_age"].iloc[0]) else float(g["age_months"].max())
        gg = g[g["age_months"] <= la + 0.1]
        if len(gg) < 2:
            gg = g.head(3)
        rd = reg.get(vin)
        for _, rr in gg.iterrows():
            mo = (rd + pd.DateOffset(months=int(rr["age_months"]))) if (rd is not None and pd.notna(rd)) else pd.NaT
            rows.append(dict(vin=vin, month=mo, soh=float(rr["soh_p50"]), age_months=float(rr["age_months"]),
                             km_month=(float(rr["km_month"]) if pd.notna(rr["km_month"]) else np.nan),
                             soh_p10=float(rr["soh_p10"]), soh_p90=float(rr["soh_p90"])))
    return pd.DataFrame(rows).sort_values(["vin", "age_months"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_oem(oem: str) -> pd.DataFrame:
    """Load + tidy one OEM's monthly battery data, gated for data-thin vehicles."""
    if oem == "Mahindra Native":
        return _load_native()
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
    """One row per vehicle with stats + a quick safe / at-risk estimate (a fast √t fit vs the warranty
    deadline — the detailed view uses the full model). Sorted oldest-first."""
    df = load_oem(oem)
    eol = EOL[oem]; warr_yr, warr_km = FLEET_WARRANTY[OEM_KEY[oem]]; warr_mo = warr_yr * 12
    rows = []
    for vin, g in df.groupby("vin"):
        g = g.dropna(subset=["soh"]).sort_values("age_months")
        n = len(g)
        if n == 0:
            continue
        soh = g["soh"].to_numpy(dtype=float); age = g["age_months"].to_numpy(dtype=float)
        now_soh = float(soh[-1]); now_age = float(age[-1]) if np.isfinite(age[-1]) else np.nan
        # effective warranty deadline (age-months): km-bound where the odometer is usable, else the time term
        odo = None
        for c in ("odo_max", "cum_km"):
            s = pd.to_numeric(g[c], errors="coerce").dropna() if c in g.columns else None
            if s is not None and len(s):
                odo = float(s.iloc[-1]); break
        km_s = pd.to_numeric(g["km_month"], errors="coerce").dropna() if "km_month" in g.columns else None
        kmpm = float(km_s.mean()) if (km_s is not None and len(km_s)) else None
        eff_age = warr_mo
        if odo and kmpm and kmpm > 0 and 0 < odo < warr_km and np.isfinite(now_age):
            eff_age = now_age + min(max(warr_mo - now_age, 0.0), (warr_km - odo) / kmpm)
        if now_soh <= eol:
            status = "⚠️ at-risk"
        else:
            pj = project(age, soh, eol)
            at_dl = (float(pj["fit"](eff_age)) if (pj and np.isfinite(now_age) and now_age < eff_age)
                     else now_soh)
            status = "⚠️ at-risk" if at_dl < eol else "✅ safe"
        drop = float(g["soh"].head(3).mean() - g["soh"].tail(3).mean())
        rows.append(dict(vin=vin, n_months=n, last_soh=now_soh, soh_start=float(soh[0]),
                         drop=drop if np.isfinite(drop) else 0.0,
                         flat=("flat" if (not np.isfinite(drop) or drop < 2) else "declining"),
                         age=now_age if np.isfinite(now_age) else 0.0, status=status))
    idx = pd.DataFrame(rows)
    if idx.empty:
        return idx
    idx["demo_score"] = (idx["n_months"].clip(upper=24)
                         + idx["drop"].clip(lower=0, upper=25) * 1.5)
    return idx.sort_values("age", ascending=False).reset_index(drop=True)   # oldest first


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


@st.cache_data(show_spinner=False)
def _vin_model_map():
    try:
        r = pd.read_csv("Vin_Model_Details.csv"); r["vin"] = r["vin"].astype(str)
        col = "model" if "model" in r.columns else ("name" if "name" in r.columns else None)
        return dict(zip(r["vin"], r[col].fillna(""))) if col else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _spec_sheet():
    try:
        return pd.read_csv("OEM_Model_Specs.csv")
    except Exception:
        return pd.DataFrame()


def vehicle_spec(oem: str, vin: str):
    """Best-effort (model_name, spec_row|None) from Vin_Model_Details + OEM_Model_Specs (fuzzy model match)."""
    model = (_vin_model_map().get(vin, "") or "").strip()
    sp = _spec_sheet()
    if sp.empty:
        return model, None
    base = "mahindra" if oem.startswith("Mahindra") else OEM_KEY.get(oem, oem.lower())
    cand = sp[sp["oem"].astype(str).str.lower() == base]
    if cand.empty:
        return model, None
    row = None
    if model:
        ml = model.lower()
        for _, r in cand.iterrows():
            k = str(r["model"]).lower()
            if k and (k in ml or (k.split() and ml.startswith(k.split()[0]))):
                row = r; break
    return model, (row if row is not None else cand.iloc[0])


def real_range(soh: float, spec, rated: float, driveeff):
    """Physics-based range = usable energy ÷ consumption, instead of the optimistic ARAI-rated × SoH.
    usable energy = battery_kWh × SoH; consumption = the vehicle's MEASURED Wh/km where we have it (e.g. Bajaj's
    drive-efficiency), else the model's rated (ARAI) Wh/km. Falls back to rated × SoH when battery kWh is unknown
    (e.g. Piaggio, no spec sheet). Returns (range_km, basis_text)."""
    kwh = None
    if spec is not None:
        try:
            v = float(spec.get("battery_capacity_kWh")); kwh = v if 3 < v < 30 else None
        except Exception:
            kwh = None
        try:                                              # prefer the model's own ARAI range (matches the spec card)
            rv = float(spec.get("rated_range_km")); rated = rv if 20 < rv < 400 else rated
        except Exception:
            pass
    if kwh is None or not rated:
        return rated * soh / 100.0, "ARAI range × health"
    if driveeff is not None and 10 < float(driveeff) < 250:
        whkm, basis = float(driveeff), "your measured Wh/km"
    else:
        whkm, basis = kwh * 1000.0 / rated, "rated (ARAI) Wh/km"
    return (kwh * soh / 100.0) * 1000.0 / whkm, basis


@st.cache_data(show_spinner="Loading raw charge/voltage telemetry…")
def raw_trace(oem: str, vin: str):
    """A vehicle's raw SoC / voltage / current over time from the local dense or intellicar feed (where downloaded),
    resampled to 5-min. Euler/Mahindra/Piaggio carry pack voltage; Bajaj does not. Returns df[t,soc,voltage,current]
    or None. Only a subset of vehicles have local raw telemetry."""
    import glob
    frames, cmap = [], None
    try:
        if oem == "Euler":
            cmap = {"soc": "batterySoc", "voltage": "batteryVoltage", "current": "batteryCurrent"}
            for f in glob.glob("data/euler/dense/*.parquet"):
                d = pd.read_parquet(f); d = d[d["vin"].astype(str) == vin]
                if len(d):
                    frames.append(d)
        elif oem == "Bajaj":
            cmap = {"soc": "essBmsSocEstPercValue", "voltage": None, "current": None}
            for f in glob.glob("data/bajaj/dense/*.parquet"):
                d = pd.read_parquet(f); d = d[d["vin"].astype(str) == vin]
                if len(d):
                    frames.append(d)
        elif oem in ("Mahindra", "Piaggio"):
            import pyarrow.dataset as ds, pyarrow.compute as pc
            path = "data/mahindra/intellicar" if oem == "Mahindra" else "data/piaggio/intellicar"
            cmap = {"soc": "soc", "voltage": "batteryVoltage", "current": "current"}
            if os.path.isdir(path):
                tbl = ds.dataset(path, format="parquet").to_table(
                    columns=["vin", "eventAt", "soc", "batteryVoltage", "current"], filter=pc.field("vin") == vin)
                if tbl.num_rows:
                    frames.append(tbl.to_pandas())
    except Exception:
        return None
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame()
    out["t"] = pd.to_datetime(pd.to_numeric(d["eventAt"], errors="coerce"), unit="ms")
    out["soc"] = pd.to_numeric(d[cmap["soc"]], errors="coerce")
    out["voltage"] = pd.to_numeric(d[cmap["voltage"]], errors="coerce") if cmap["voltage"] in d.columns else np.nan
    out["current"] = pd.to_numeric(d[cmap["current"]], errors="coerce") if cmap["current"] in d.columns else np.nan
    out = out[out["soc"].between(0, 100)].dropna(subset=["t"]).sort_values("t")
    if out.empty:
        return None
    out = (out.set_index("t").resample("5min").agg({"soc": "last", "voltage": "mean", "current": "mean"})
              .dropna(subset=["soc"]).reset_index())
    return out


# ===========================================================================
# Personalisation — per-vehicle behaviour vs the fleet (age-matched, LFP-tuned)
# ===========================================================================
# For every metric below, a HIGHER value = harsher on the pack (gentler = lower).
# High-SoC dwell is deliberately EXCLUDED — LFP tolerates a full pack; heat and
# hard, high-current use are the real stressors. Currents are stored as abs().
_BEH_COLS = {"cur_chg_mean": "charge_i", "cur_dis_mean": "drive_i", "cur_abs_p95": "peak_i",
             "crate_p95": "crate", "ah_throughput": "throughput", "dod_mean": "dod",
             "frac_soc_low": "lowdwell", "temp_mean": "temp", "dte_mean": "range",
             "capacity_ah": "cap", "km_month": "km", "cyc_month": "cyc_month"}
_CARE_METRICS = ["charge_i", "drive_i", "peak_i", "crate", "throughput", "dod", "lowdwell", "temp"]
_LEVER_LABEL = {"temp": ("running hot", "park and charge in the shade"),
                "peak_i": ("hard peak pulls", "smoother starts and lighter peak loads"),
                "crate": ("hard peak pulls", "smoother starts and lighter peak loads"),
                "drive_i": ("hard driving", "smoother acceleration"),
                "throughput": ("heavy throughput", "spreading out very heavy days where you can"),
                "dod": ("deep discharges", "topping up before it runs low"),
                "lowdwell": ("deep discharges", "recharging before it runs near-empty")}


@st.cache_data(show_spinner=False)
def behaviour_table(oem: str) -> pd.DataFrame:
    """One row per vehicle: lifetime-mean of each behaviour metric (abs for currents) + age + SoH now.
    Only columns that exist for the OEM are filled. Used for fleet percentile ranking."""
    df = load_oem(oem)
    rows = []
    for vin, g in df.groupby("vin"):
        soh = pd.to_numeric(g["soh"], errors="coerce").dropna()
        if soh.empty:
            continue
        rec = {"vin": vin, "soh_now": float(soh.iloc[-1]),
               "age": float(pd.to_numeric(g.get("age_months"), errors="coerce").max())}
        for src, key in _BEH_COLS.items():
            if src in g.columns:
                s = pd.to_numeric(g[src], errors="coerce")
                if src == "km_month":
                    s = s.where((s >= 0) & (s <= 15000))
                s = s.dropna()
                if len(s):
                    rec[key] = abs(float(s.mean())) if src in ("cur_chg_mean", "cur_dis_mean") else float(s.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def vehicle_behaviour(frame) -> dict:
    """This vehicle's lifetime behaviour values, keyed like behaviour_table (mean_val resolves at call time)."""
    out = {}
    for src, key in _BEH_COLS.items():
        v = mean_val(frame, src)
        if v is not None:
            out[key] = abs(v) if src in ("cur_chg_mean", "cur_dis_mean") else v
    return out


def _col(bt, name):
    return bt[name] if (name in getattr(bt, "columns", [])) else pd.Series(dtype=float)


def gentler_than(series, value):
    """% of the fleet HARSHER (higher) than this value — i.e. 'you are gentler than X%'. Needs >=8 peers."""
    s = series.dropna()
    if value is None or not np.isfinite(value) or len(s) < 8:
        return None
    return round(float((s > value).mean() * 100), 0)


def harsher_than(series, value):
    """% of the fleet BELOW this value — for 'better than X%' on a higher-is-better metric (range/SoH)."""
    s = series.dropna()
    if value is None or not np.isfinite(value) or len(s) < 8:
        return None
    return round(float((s < value).mean() * 100), 0)


def peer_soh_rank(bt, age, soh, window=6):
    """Age-matched: % of similar-age vehicles with LOWER SoH than this one (falls back to whole fleet)."""
    if bt.empty or soh is None or not np.isfinite(soh):
        return None, 0
    band = bt[(bt["age"] >= age - window) & (bt["age"] <= age + window)] if np.isfinite(age) else bt
    s = band["soh_now"].dropna()
    if len(s) < 10:
        s = bt["soh_now"].dropna()
    if len(s) < 10:
        return None, len(s)
    return round(float((s < soh).mean() * 100), 0), len(s)


def care_score(bt, rv):
    g = [gentler_than(_col(bt, k), rv[k]) for k in _CARE_METRICS if k in rv]
    g = [x for x in g if x is not None]
    return round(float(np.mean(g)), 0) if g else None


def care_grade(score):
    if score is None:
        return ("—", MUTE)
    if score >= 72:
        return ("Gentle", GREEN)
    if score >= 50:
        return ("Moderate", GREEN)
    if score >= 35:
        return ("Firm", AMBER)
    return ("Hard", RED)


def style_word(gpct, words=("Firm", "Typical", "Gentle")):
    if gpct is None:
        return "—"
    return words[2] if gpct >= 66 else (words[1] if gpct >= 33 else words[0])


def top_levers(bt, rv, n=2):
    seen, scored = set(), []
    for k, (label, action) in _LEVER_LABEL.items():
        if k in rv and label not in seen:
            h = harsher_than(_col(bt, k), rv[k])
            if h is not None and h >= 60:
                seen.add(label)
                scored.append((h, label, action))
    scored.sort(reverse=True)
    return scored[:n]


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
    """Robust √t decline (slope ≤ 0), but ANCHORED at the present measured SoH so the projection is
    continuous with the measured line — it starts exactly where the battery is now, not at a smoothed
    fit value (which, for cliff-laden SoH like Mahindra's, would float above the last reading)."""
    age = np.asarray(age_months, dtype=float)
    soh = np.asarray(soh, dtype=float)
    m = np.isfinite(age) & np.isfinite(soh)
    age, soh = age[m], soh[m]
    if len(age) < 3:
        return None
    order = np.argsort(age); age, soh = age[order], soh[order]
    slope, _ = _theilsen(np.sqrt(np.clip(age, 0, None)), soh)   # robust √t slope (deceleration-aware)
    slope = min(slope, 0.0)                                     # health can't rise with age
    now_age = float(age[-1]); now_soh = float(soh[-1])          # anchor at the LATEST measured point
    inter = now_soh - slope * np.sqrt(max(now_age, 0.0))        # so the line passes through (now_age, now_soh)

    def fit(t):
        t = np.asarray(t, dtype=float)
        return inter + slope * np.sqrt(np.clip(t, 0, None))

    if slope < -1e-9 and now_soh > eol:
        root = (eol - inter) / slope
        t_eol = root * root if root > 0 else now_age
        months_to_eol = max(0.0, t_eol - now_age)
    else:
        months_to_eol = None                                   # flat, or already at/below EoL
    return dict(fit=fit, slope=slope, inter=inter, now_age=now_age,
                now_soh=now_soh, months_to_eol=months_to_eol)


# ---------------------------------------------------------------------------
# Model-based projection — the SAME conditioned models the internal pipeline uses
# (euler_model trajectory · Mahindra expected-loss quantiles · Bajaj quantiles),
# so the customer forecast matches the pipeline. Falls back to the √t fit above
# for vehicles with too little history for the model.
# ---------------------------------------------------------------------------
_MODEL_MODULE = {"Euler": "euler_model", "Mahindra": "model", "Bajaj": "bajaj_model", "Piaggio": "model"}


# Training populations compared in the Lifespan chart. safe = never observed at/below EoL.
POPULATIONS = [("all", "All vehicles", GREEN), ("deg", "Degraders only", AMBER),
               ("safe", "Safe (incl. flat)", BLUE), ("safe_deg", "Safe degraders", "#c792ff")]


@st.cache_resource(show_spinner=False)
def customer_forecaster(oem: str, pop: str = "all"):
    """Train (cached) the pipeline model for an OEM on a chosen training population:
    all · deg (drop≥2) · safe (never reached EoL) · safe_deg (safe AND drop≥2)."""
    df = load_oem(oem); eol = EOL[oem]
    gg = df.groupby("vin"); drop = gg["soh"].first() - gg["soh"].last(); smin = gg["soh"].min()
    if pop == "deg":
        keep = set(drop[drop >= 2].index)
    elif pop == "safe":
        keep = set(smin[smin > eol].index)
    elif pop == "safe_deg":
        keep = set(drop[drop >= 2].index) & set(smin[smin > eol].index)
    else:
        keep = set(df["vin"].unique())
    df = df[df["vin"].isin(keep)]
    mod = importlib.import_module(_MODEL_MODULE[oem])
    if oem == "Euler":
        return mod, mod.train_traj(mod.build_traj_samples(df))
    return mod, mod.train_quantiles(mod.build_transitions(df))


def model_project(oem: str, g, eol: float, horizon: int = 120, pop: str = "all") -> dict | None:
    """Forecast this vehicle with the pipeline model (trained on population `pop`), exposing the same
    interface as project(): fit(age)->SoH, months_to_eol, now_age, now_soh. Anchored at the present SoH."""
    if oem not in _MODEL_MODULE:                                  # e.g. Mahindra Native -> no supervised model
        return None
    gg = g.dropna(subset=["soh", "age_months"]).sort_values("month")
    if len(gg) < 4:
        return None
    now_age = float(gg["age_months"].iloc[-1]); now_soh = float(gg["soh"].iloc[-1])
    try:
        mod, fmodel = customer_forecaster(oem, pop)
        if oem == "Euler":
            p50 = np.asarray(mod.forecast(gg, fmodel, horizon)[0.5], dtype=float)
        else:
            p50 = mod.simulate(gg, fmodel, horizon)["q50"].to_numpy()
    except Exception:
        return None
    if p50.size == 0:
        return None
    ages = now_age + np.arange(0, len(p50) + 1)                    # now, now+1, … now+H
    central = np.concatenate([[now_soh], p50])                     # anchor at the present measured SoH
    below = np.where(central <= eol)[0]
    months_to_eol = float(ages[below[0]] - now_age) if len(below) else None
    tail_slope = min(float(central[-1] - central[-2]), 0.0) if len(central) >= 2 else 0.0

    def fit(t):
        t = np.asarray(t, dtype=float)
        y = np.interp(t, ages, central)                           # flat before now, model curve through horizon
        return np.where(t > ages[-1], central[-1] + tail_slope * (t - ages[-1]), y)   # extrapolate past horizon

    return dict(fit=fit, slope=tail_slope, now_age=now_age, now_soh=now_soh,
                months_to_eol=months_to_eol)


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


def section(label: str, title: str, sub: str | None = None):
    """Small uppercase monospace label + big title + optional one-line subtitle."""
    sub_html = (f"<div style='color:{MUTE};font-size:0.98rem;margin-top:4px;'>{sub}</div>"
                if sub else "")
    st.markdown(
        f"<div style='margin:42px 0 18px 0;'>"
        f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        f"font-size:0.72rem;letter-spacing:0.18em;color:{FAINT};"
        f"text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:1.62rem;font-weight:700;color:{TEXT};margin-top:3px;"
        f"letter-spacing:-0.01em;'>{title}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def big_number(value: str, unit: str = "", color: str = TEXT):
    return (f"<span style='font-size:2.6rem;font-weight:700;color:{color};"
            f"letter-spacing:-0.02em;'>{value}</span>"
            f"<span style='font-size:1.0rem;color:{MUTE};margin-left:6px;'>{unit}</span>")


def card(label: str, value: str, sub: str | None = None, color: str = TEXT) -> str:
    """Uniform stat-card markup. Equal min-height so a row of them always lines up."""
    sub_html = (f"<div style='color:{FAINT};font-size:0.78rem;margin-top:5px;'>{sub}</div>"
                if sub else "")
    return (
        f"<div style='background:{PANEL};border:1px solid {LINE};border-radius:16px;"
        f"padding:18px 20px;min-height:108px;'>"
        f"<div style='color:{MUTE};font-size:0.74rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>{label}</div>"
        f"<div style='color:{color};font-size:1.85rem;font-weight:700;margin-top:8px;"
        f"line-height:1.05;letter-spacing:-0.02em;'>{value}</div>{sub_html}</div>"
    )


def stat_strip(items):
    """Render a row of equal-width uniform stat cards. items = (label, value[, sub[, color]])."""
    cols = st.columns(len(items), gap="medium")
    for col, it in zip(cols, items):
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else None
        color = it[3] if len(it) > 3 else TEXT
        with col:
            st.markdown(card(label, value, sub, color), unsafe_allow_html=True)


def why(text: str, sources: str | None = None):
    s = (f"<div style='color:{FAINT};font-size:0.78rem;margin-top:10px;'>Sources: {sources}</div>"
         if sources else "")
    st.markdown(
        f"<div style='color:{MUTE};font-size:0.96rem;line-height:1.65;margin-top:16px;"
        f"max-width:980px;'>{text}</div>{s}", unsafe_allow_html=True,
    )


def _polish(fig, height: int = 320, legend: bool = False):
    """One styler for every chart -> uniform bg, grid, font, margins. Merges with each
    chart's own axis titles (plotly update_layout recurses into sub-objects)."""
    fig.update_layout(
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=MUTE, size=12), margin=dict(l=18, r=18, t=20, b=16),
        xaxis=dict(gridcolor=LINE, color=MUTE, zeroline=False),
        yaxis=dict(gridcolor=LINE, color=MUTE, zeroline=False),
        showlegend=legend, hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, bordercolor=LINE),
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom",
                                      bgcolor="rgba(0,0,0,0)", font=dict(color=MUTE, size=11)))
    return fig


def chart(fig, height: int = 320, legend: bool = False):
    """Full-width, uniformly-styled chart wrapped in a premium card."""
    with st.container(border=True):
        st.plotly_chart(_polish(fig, height, legend), use_container_width=True,
                        config={"displayModeBar": False})


def _soc_curve(soc_mean, frac_low, frac_high):
    fl = max(float(frac_low or 0.0), 0.0); fh = max(float(frac_high or 0.0), 0.0)
    fm = max(1.0 - fl - fh, 0.05); mid = float(np.clip(soc_mean if soc_mean else 55.0, 35.0, 78.0))
    x = np.linspace(0, 100, 240)
    y = (fl * np.exp(-0.5 * ((x - 16) / 9) ** 2) + fm * np.exp(-0.5 * ((x - mid) / 13) ** 2)
         + fh * np.exp(-0.5 * ((x - 88) / 8) ** 2))
    return x, y / (y.max() or 1.0)


def soc_density_fig(soc_mean, frac_low, frac_high, overlays=None):
    """Approximate SoC distribution from the summary stats we track. `overlays` = list of
    (label, soc_mean, frac_low, frac_high, colour) drawn as comparison lines (e.g. best / worst vehicle)."""
    x, y = _soc_curve(soc_mean, frac_low, frac_high)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name="You",
                               line=dict(color=GREEN, width=2.4), fillcolor=GREEN_FILL))
    if soc_mean:
        fig.add_vline(x=float(soc_mean), line=dict(color=MUTE, width=1.2, dash="dot"),
                      annotation_text=f"you {soc_mean:.0f}%", annotation_font_size=11,
                      annotation_font_color=MUTE)
    for label, sm, flo, fhi, color in (overlays or []):
        xo, yo = _soc_curve(sm, flo, fhi)
        fig.add_scatter(x=xo, y=yo, mode="lines", name=label, line=dict(color=color, width=2, dash="dash"))
    fig.update_layout(xaxis=dict(title="state of charge (%)", range=[0, 100]),
                      yaxis=dict(visible=False))
    return fig


@st.cache_data(show_spinner=False)
def best_worst_soc(oem: str):
    """Healthiest + most-degraded vehicle (by current SoH) among REAL decliners with enough history
    (≥15 months, ≥2pp drop — never flat / too new) for the SoC comparison overlay. Returns dict or None."""
    df = load_oem(oem)
    rows = []
    for vin, g in df.groupby("vin"):
        gg = g.dropna(subset=["soh"]).sort_values("age_months")
        if len(gg) < 15:                                   # not too new
            continue
        if float(gg["soh"].iloc[0] - gg["soh"].iloc[-1]) < 2:   # not flat
            continue
        sm = mean_val(g, "soc_mean"); fh = mean_val(g, "frac_soc_high"); fl = mean_val(g, "frac_soc_low")
        if sm is None or fh is None:
            continue
        rows.append((vin, float(gg["soh"].iloc[-1]), sm, fl, fh))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[1])
    return {"worst": rows[0], "best": rows[-1]}


def temp_dist_fig(values, marker):
    """Pack-temperature distribution with the LFP 'hot zone' shaded + this vehicle's marker."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return None
    lo, hi = np.percentile(values, 1), np.percentile(values, 99)
    values = values[(values >= lo) & (values <= hi)]
    fig = go.Figure(go.Histogram(x=values, nbinsx=30, marker_color=BLUE, opacity=0.55,
                                 hoverinfo="skip", showlegend=False))
    fig.add_vrect(x0=HOT_C, x1=max(hi, marker if marker else hi),
                  fillcolor=AMBER, opacity=0.12, line_width=0)
    fig.add_vline(x=HOT_C, line=dict(color=AMBER, width=1.5, dash="dot"))
    if marker is not None and np.isfinite(marker):
        fig.add_vline(x=marker, line=dict(color=TEXT, width=2.5, dash="dot"),
                      annotation_text=f"  you: {marker:.0f}°C", annotation_position="top",
                      annotation_font=dict(color=TEXT, size=12))
    fig.update_layout(bargap=0.06, xaxis=dict(title="monthly pack temperature (°C)"),
                      yaxis=dict(visible=False))
    return fig


def gauge(soh: float, eol: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(soh, 1),
        number={"suffix": " %", "font": {"size": 52, "color": TEXT}},
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
    fig.update_layout(height=240, margin=dict(l=18, r=18, t=10, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


def range_gauge(range_now: float, rated: float, eol: float, color: str):
    """Range shown the same way as SoH — a gauge from 0 to the rated full-charge range, with the
    end-of-life range (rated × EoL%) marked as the threshold and the same health-coloured zones."""
    eol_range = rated * eol / 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(range_now),
        number={"suffix": " km", "font": {"size": 44, "color": TEXT}},
        gauge={
            "axis": {"range": [0, rated], "tickwidth": 1, "tickcolor": FAINT,
                     "tickfont": {"size": 11, "color": MUTE}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": PANEL, "borderwidth": 0,
            "steps": [
                {"range": [0, eol_range], "color": "rgba(239,93,99,0.18)"},
                {"range": [eol_range, eol_range + rated * 0.08], "color": "rgba(243,177,78,0.18)"},
                {"range": [eol_range + rated * 0.08, rated], "color": "rgba(52,209,127,0.16)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 4}, "thickness": 0.8,
                          "value": eol_range},
        },
    ))
    fig.update_layout(height=240, margin=dict(l=18, r=18, t=10, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


# ===========================================================================
# Global dark styling — full-width premium shell
# ===========================================================================
st.markdown(
    f"""
    <style>
      .stApp {{ background:{BG}; }}
      .block-container {{ padding: 1.6rem 2.6rem 4rem; max-width: 1500px; }}
      [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      h1,h2,h3,h4,p,span,label,div {{ color:{TEXT}; }}
      * {{ -webkit-font-smoothing:antialiased; }}
      /* Uniform card surface for every bordered container, metric + custom card */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
          background:{PANEL}; border:1px solid {LINE} !important;
          border-radius:16px; padding:18px 20px;
      }}
      div[data-testid="stMetric"] {{
          background:transparent; border:none; padding:2px 0;
      }}
      div[data-testid="stMetricLabel"] p {{ color:{MUTE}; font-weight:600;
          text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em; }}
      div[data-testid="stMetricValue"] {{ color:{TEXT}; font-weight:700; }}
      .pill {{ display:inline-block; padding:5px 16px; border-radius:999px;
               font-weight:700; font-size:0.9rem; }}
      div[data-baseweb="select"] > div {{ background:{PANEL2}; border-color:{LINE}; }}
      .stProgress > div > div > div {{ background:{GREEN}; }}
      .stSelectbox label {{ color:{MUTE}; font-weight:600; text-transform:uppercase;
          font-size:0.72rem; letter-spacing:0.05em; }}
      hr {{ border-color:{LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Header + picker
# ===========================================================================
_gif = REPO_ROOT / "turno.gif"
_hc1, _hc2 = st.columns([0.62, 0.38], gap="large")
with _hc1:
    if _gif.exists():
        st.image(str(_gif), width=120)
    st.markdown("<h1 style='margin-bottom:0;font-weight:800;letter-spacing:-0.02em;'>"
                "🔋 Your Battery Health</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTE};margin-top:4px;font-size:1.0rem;'>A simple, friendly "
                f"check-up for your electric three-wheeler — by Turno.</p>", unsafe_allow_html=True)

# One compact row: brand · VIN · specifications
cb, cv, cs = st.columns([0.16, 0.34, 0.50], gap="medium")
with cb:
    oem = st.selectbox("Vehicle brand", OEMS, index=0)

idx = vehicle_index(oem)
if idx.empty:
    st.error("No vehicle data available for this brand right now.")
    st.stop()
demo_vins = idx.nlargest(8, "demo_score")["vin"].tolist()     # ⭐ = long history + visible decline
ordered = idx["vin"].tolist()                                 # already sorted oldest-first


def _vin_label(v: str) -> str:
    row = idx[idx["vin"] == v]
    star = " ⭐" if v in demo_vins else ""
    if row.empty:
        return v + star
    r0 = row.iloc[0]
    trend = "➖ flat" if r0["flat"] == "flat" else "📉 degrading"
    return f"{v} · {int(r0['n_months'])} mo · {r0['status']} · {trend}{star}"


with cv:
    vin = st.selectbox("Vehicle (VIN)", ordered, index=0, format_func=_vin_label,
                       help="⭐ = demo vehicles with long history and visible decline.")

df = load_oem(oem)
g = df[df["vin"] == vin].dropna(subset=["soh"]).sort_values("age_months")
if g.empty:
    st.warning("This vehicle doesn't have any battery readings yet.")
    st.stop()

_model, _spec = vehicle_spec(oem, vin)


def _spv(colname, suffix=""):
    if _spec is None:
        return None
    v = _spec.get(colname)
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "unverified", "none (not disclosed)")):
        return None
    try:
        return f"{float(v):g}{suffix}"
    except Exception:
        return str(v)


with cs:
    _wy, _wk = FLEET_WARRANTY[OEM_KEY[oem]]
    _body = (str(_spec.get("body_type")) if (_spec is not None and pd.notna(_spec.get("body_type"))
             and str(_spec.get("body_type")).lower() != "unverified") else "")
    _chips = [_model or f"{oem} 3-wheeler", _spv("battery_capacity_kWh", " kWh"),
              ("LFP" if _spv("chemistry") is None else _spv("chemistry")),
              _spv("rated_range_km", " km"), _spv("motor_power_kW", " kW"),
              f"warranty {_wy}yr / {_wk / 1000:.0f}k km"]
    _chip_html = " &nbsp;·&nbsp; ".join(f"<b style='color:{TEXT};'>{c}</b>" for c in _chips if c)
    st.markdown(f"<div style='color:{FAINT};font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:0.05em;'>Specifications{(' · ' + _body) if _body else ''}</div>"
                f"<div style='color:{MUTE};font-size:0.95rem;line-height:1.75;margin-top:3px;'>{_chip_html}</div>",
                unsafe_allow_html=True)

if oem == "Mahindra Native":
    st.warning("🔎 **Estimated health — no on-board sensor.** This vehicle streams charge, distance and time but "
               "**no current or voltage**, so its battery health can't be measured directly. The figures below are a "
               "**statistical estimate** from its **age and mileage** (our behaviour model, validated against "
               "sensor-equipped Mahindras to ~1.5% error) — a guide with a wide margin, not a precise reading. "
               "Sensor-based sections (charge habits, temperature) are unavailable.")

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

_driveeff = mean_val(g, "driveeff_mean")                                    # _model/_spec computed with the picker row
range_now, range_basis = real_range(soh_now, _spec, rated, _driveeff)       # energy ÷ efficiency, not ARAI × SoH
range_full, _ = real_range(100.0, _spec, rated, _driveeff)                  # full-health range → range-gauge scale
status_label, status_color, _ = health_status(soh_now, eol)
proj = (model_project(oem, g, eol)                             # the pipeline model (matches the internal dashboard)
        or project(g["age_months"].to_numpy(), g["soh"].to_numpy(), eol))   # √t fallback for thin history


# ===========================================================================
# 1) HERO — Battery health gauge
# ===========================================================================
section("BATTERY HEALTH", "How your battery is doing")
cH, cR, cS = st.columns([0.24, 0.24, 0.52], gap="medium")      # gauges + status card all in ONE row
with cH:
    with st.container(border=True):
        st.plotly_chart(gauge(soh_now, eol, status_color), use_container_width=True,
                        config={"displayModeBar": False}, key="g_soh")
        st.markdown(f"<div style='text-align:center;color:{MUTE};font-size:0.85rem;margin-top:-6px;'>"
                    f"Battery health</div>", unsafe_allow_html=True)
with cR:
    with st.container(border=True):
        st.plotly_chart(range_gauge(range_now, range_full, eol, status_color), use_container_width=True,
                        config={"displayModeBar": False}, key="g_range")
        st.markdown(f"<div style='text-align:center;color:{MUTE};font-size:0.85rem;margin-top:-6px;'>"
                    f"Range · {range_full:.0f} km at full · "
                    f"<span style='color:{FAINT};'>{range_basis}</span></div>", unsafe_allow_html=True)
with cS:
    with st.container(border=True):
        st.markdown(f"<span class='pill' style='background:{status_color}26;color:{status_color};'>"
                    f"● {status_label}</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:12px;'>{big_number(f'{soh_now:.0f}', '%', status_color)}"
                    f"<span style='color:{MUTE};font-size:1.05rem;margin-left:6px;'>battery health</span></div>",
                    unsafe_allow_html=True)
        if proj and proj["months_to_eol"] is not None and soh_now > eol:
            life = f"about <b style='color:{TEXT}'>{fmt_months_human(proj['months_to_eol'])}</b> of healthy life left"
        elif soh_now <= eol:
            life = "this battery has reached its <b>end-of-life</b> health line"
        else:
            life = "holding steady — <b>no meaningful decline</b> detected yet"
        st.markdown(f"<p style='color:{MUTE};font-size:1.05rem;margin-top:12px;line-height:1.5;'>"
                    f"Your {oem} battery is at {soh_now:.0f}% of its original capacity — {life}.</p>",
                    unsafe_allow_html=True)
        # age · distance · cycles folded in as chips (was a separate second row of stat cards)
        _age_s = fmt_age(age_now)
        _dist_s = f"{odo:,.0f} km" if odo and odo > 0 else "—"
        _cyc_s = f"{cycles:,.0f}" if cycles and np.isfinite(cycles) and cycles > 0 else "—"
        st.markdown(
            "<div style='margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;'>"
            + "".join(f"<span class='pill' style='background:{FAINT}1f;color:{TEXT};'>{lbl} <b>{val}</b></span>"
                      for lbl, val in [("Age", _age_s), ("Driven", _dist_s), ("Charge cycles", _cyc_s)])
            + "</div>", unsafe_allow_html=True)
        _prank, _pn = peer_soh_rank(behaviour_table(oem), age_now, soh_now)
        if _prank is not None:
            _pc = GREEN if _prank >= 50 else AMBER
            st.markdown(f"<div style='margin-top:8px;'><span class='pill' style='background:{_pc}26;color:{_pc};'>"
                        f"🏅 Healthier than ~{_prank:.0f}% of similar-age {oem}s</span></div>",
                        unsafe_allow_html=True)


# ===========================================================================
# 6) LIFESPAN — degradation history + projection (the centrepiece chart)
# ===========================================================================
section("LIFESPAN", "How your battery is aging")
life_left = (fmt_months_human(proj["months_to_eol"])
             if proj and proj["months_to_eol"] is not None
             else ("reached end-of-life" if soh_now <= eol else "no decline yet"))
stat_strip([
    ("Health today", f"{soh_now:.0f}%", None, status_color),
    ("Healthy life left", life_left),
    ("End-of-life line", f"{eol:.0f}%"),
    ("Warranty term", f"{warr_years} yr / {warr_km / 1000:.0f}k km"),
])
st.write("")

# Extend the x-axis all the way to where the projection crosses end-of-life, so the dashed line
# actually reaches the 80% line. Cap at ~25 yr so a near-flat vehicle can't produce a runaway axis.
# Forecast the vehicle under each training population (overlaid below); hero/warranty use the 'all' proj.
pop_projs = {pop: model_project(oem, g, eol, pop=pop) for pop, _, _ in POPULATIONS}
eol_age = (proj["now_age"] + proj["months_to_eol"]) if (proj and proj["months_to_eol"] is not None) else None
_cross = [(p["now_age"] + p["months_to_eol"]) for p in pop_projs.values() if p and p["months_to_eol"] is not None]
_latest = max(_cross) if _cross else (eol_age or 0)
_base = max(warr_months + 6, (age_now + 12) if np.isfinite(age_now) else 60, 60)
chart_end = float(min(max(_base, _latest + 6), 300))

fig = go.Figure()
try:
    fl = df.dropna(subset=["soh", "age_months"]).copy()
    fl["bin"] = (fl["age_months"] / 3).round() * 3
    band = fl.groupby("bin")["soh"].agg(
        lo=lambda s: s.quantile(0.10), hi=lambda s: s.quantile(0.90), n="size").reset_index()
    band = band[band["n"] >= 5].sort_values("bin")
    if len(band) >= 3:
        fig.add_trace(go.Scatter(
            x=[b / 12 for b in band["bin"]] + [b / 12 for b in band["bin"][::-1]],
            y=list(band["hi"]) + list(band["lo"][::-1]),
            fill="toself", fillcolor="rgba(90,169,247,0.10)", line=dict(width=0),
            hoverinfo="skip", name="Typical for the fleet"))
except Exception:
    pass

fig.add_trace(go.Scatter(x=g["age_months"] / 12, y=g["soh"], mode="lines+markers",
                         line=dict(color=status_color, width=3), marker=dict(size=6),
                         name="Your battery"))
for pop, label, color in POPULATIONS:                          # one predicted line per training population
    pj = pop_projs.get(pop)
    if pj is None:
        continue
    xp = np.linspace(pj["now_age"], chart_end, 40)
    fig.add_trace(go.Scatter(x=xp / 12, y=pj["fit"](xp), mode="lines",
                             line=dict(color=color, width=2, dash="dash"), name=f"Predicted · {label}"))
fig.add_hline(y=eol, line=dict(color=AMBER, width=2, dash="dot"),
              annotation_text=f"End-of-life ({eol:.0f}%)", annotation_position="bottom right",
              annotation_font=dict(color=AMBER, size=12))
fig.add_vline(x=warr_years, line=dict(color=MUTE, width=1.5, dash="dash"),
              annotation_text=f"Warranty ({warr_years} yr)", annotation_position="top left",
              annotation_font=dict(color=MUTE, size=11))
if eol_age is not None and eol_age <= chart_end + 1:
    fig.add_vline(x=eol_age / 12, line=dict(color=status_color, width=1.4, dash="dot"),
                  annotation_text=f"Est. end-of-life · {fmt_months_human(proj['months_to_eol'])} left",
                  annotation_position="top right",
                  annotation_font=dict(color=status_color, size=11))
# Registration date = age 0. If recoverable (first timestamp − first age), show a dual x-axis
# (calendar date on the bottom, battery age on top) + a vertical line at registration.
reg = None
try:
    _m0 = pd.Timestamp(g["month"].iloc[0])
    if pd.notna(_m0):
        reg = (_m0 - pd.DateOffset(months=int(round(float(g["age_months"].iloc[0]))))).normalize()
except Exception:
    reg = None

if reg is not None and pd.notna(reg):
    _reglbl = reg.strftime("%b '%y")
    fig.add_vline(x=0, line=dict(color=FAINT, width=1.5, dash="dot"),
                  annotation_text=f"Registered {_reglbl}", annotation_position="bottom right",
                  annotation_font=dict(color=FAINT, size=10))
    _yrmax = max(1, int(np.ceil(chart_end / 12)))
    _yrs = list(range(0, _yrmax + 1))
    _dates = [(reg + pd.DateOffset(years=y)).strftime("%b '%y") for y in _yrs]
    fig.add_trace(go.Scatter(x=[0, _yrmax], y=[eol, eol], mode="lines", line=dict(width=0),
                             xaxis="x2", showlegend=False, hoverinfo="skip"))     # anchors the top age axis
    fig.update_layout(
        height=440, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTE, size=12),
        margin=dict(l=44, r=16, t=52, b=40), hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        xaxis=dict(title="Date", range=[0, _yrmax], gridcolor=LINE, color=MUTE, zeroline=False,
                   tickmode="array", tickvals=_yrs, ticktext=_dates),
        xaxis2=dict(title="Battery age (years)", overlaying="x", side="top", range=[0, _yrmax],
                    color=MUTE, showgrid=False, zeroline=False, tickmode="array",
                    tickvals=_yrs, ticktext=[str(y) for y in _yrs]),
        yaxis=dict(title="Battery health (%)", range=[eol - 18, 102], gridcolor=LINE, color=MUTE, zeroline=False))
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    fig.update_layout(xaxis=dict(title="Battery age (years)"),
                      yaxis=dict(title="Battery health (%)", range=[eol - 18, 102]),
                      hovermode="x unified")
    chart(fig, height=400, legend=True)
st.caption("Each dashed line is the *same* forecast model trained on a different **population** of vehicles — "
           "All · Degraders-only (lost ≥2%) · Safe (never reached end-of-life) · Safe-degraders (safe **and** "
           "declining). Where they diverge shows how much the training set shapes the prediction. The "
           "headline health & warranty figures above use the **All-vehicles** model.")


# ===========================================================================
# 6b) USAGE + CHARGE-CURVE EVOLUTION — odometer over time; SoC↔voltage aging
# ===========================================================================
section("USAGE", "Capacity, health and distance over the battery's life")

# --- overlay: coulomb-counted capacity + SoH + odometer on a shared age axis ---------------
age_yr = pd.to_numeric(g["age_months"], errors="coerce") / 12
_soh_s = pd.to_numeric(g["soh"], errors="coerce")
_cap_s = pd.to_numeric(g["capacity_ah"], errors="coerce") if "capacity_ah" in g.columns else pd.Series(index=g.index, dtype=float)
_od_s = pd.to_numeric(g["odo_max"], errors="coerce") if "odo_max" in g.columns else pd.Series(index=g.index, dtype=float)
_od_s = _od_s.where(_od_s < 3e5)                                  # drop odometer sentinels
has_cap = _cap_s.notna().sum() >= 3
has_od = _od_s.notna().sum() >= 2 and bool((_od_s.fillna(0) > 0).any())

ov = go.Figure()
ov.add_trace(go.Scatter(x=age_yr, y=_soh_s, name="SoH (%)", mode="lines+markers",
                        line=dict(color=GREEN, width=2.6), marker=dict(size=5), yaxis="y"))
if has_cap:
    ov.add_trace(go.Scatter(x=age_yr, y=_cap_s, name="capacity (Ah)", mode="lines+markers",
                            line=dict(color=AMBER, width=1.6), marker=dict(size=3), yaxis="y2"))
if has_od:
    ov.add_trace(go.Scatter(x=age_yr, y=_od_s, name="odometer (km)", mode="lines",
                            line=dict(color=BLUE, width=1.8, dash="dot"), yaxis="y3"))

n_right = int(has_cap) + int(has_od)
dom_r = 0.82 if n_right == 2 else (0.90 if n_right == 1 else 1.0)
lay = dict(height=390, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTE, size=12),
           margin=dict(l=20, r=(98 if n_right == 2 else 60), t=28, b=44), hovermode="x unified",
           hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, bordercolor=LINE),
           legend=dict(orientation="h", x=0, y=1.07, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                       font=dict(color=MUTE, size=11)),
           xaxis=dict(title="battery age (years)", domain=[0.0, dom_r], gridcolor=LINE, color=MUTE, zeroline=False),
           yaxis=dict(title="SoH (%)", color=GREEN, range=[max(eol - 12, 40), 102], gridcolor=LINE, zeroline=False))
if has_cap:
    lay["yaxis2"] = dict(title="capacity (Ah)", color=AMBER, overlaying="y", side="right",
                         anchor="x", showgrid=False, zeroline=False)
if has_od:
    a3 = dict(title="odometer (km)", color=BLUE, overlaying="y", side="right", showgrid=False, zeroline=False)
    if has_cap:
        a3["anchor"] = "free"; a3["position"] = 0.93
    else:
        a3["anchor"] = "x"
    lay["yaxis3"] = a3
ov.update_layout(**lay)
with st.container(border=True):
    st.plotly_chart(ov, use_container_width=True, config={"displayModeBar": False})
_cap_note = (" The **coulomb-counted capacity** (amber) bounces month-to-month — that scatter is measurement noise, "
             "not real fade; the robust **SoH** envelope (green) stays smooth despite it." if has_cap else "")
st.caption(f"**Health, measured capacity and distance on one age axis.**{_cap_note}"
           + (" Blue (dotted) = odometer." if has_od else ""))

# --- opt-in raw charge curve: SoC↔voltage coloured by charge # -----------------------------
if st.checkbox("🔬 Also show the raw charge curve (SoC–voltage, coloured by charge #) — loads raw telemetry",
               value=False, key="cc_toggle"):
    tr = raw_trace(oem, vin)
    if tr is None or tr.empty:
        st.info("Raw SoC/voltage telemetry isn't downloaded for this vehicle (available for a subset of "
                "Euler, Mahindra and Piaggio).")
    else:
        tr = tr.copy(); tr["dsoc"] = tr["soc"].diff().fillna(0.0)
        _use_i = tr["current"].notna().sum() > 20
        chg = (tr["current"] > 1.0) if _use_i else (tr["dsoc"] > 0.3)   # charging = current in, or SoC rising
        start = chg & ~chg.shift(1, fill_value=False)
        tr["event"] = start.cumsum()
        ch = tr[chg & (tr["event"] > 0)]
        n_ev = int(ch["event"].nunique()) if len(ch) else 0
        if n_ev < 2:
            st.info("Not enough distinct charging events in the downloaded window.")
        else:
            has_v = ch["voltage"].notna().sum() > 20 and ch["voltage"].between(20, 120).mean() > 0.4
            if has_v:
                ch = ch[ch["voltage"].between(20, 120)]                 # drop sensor-glitch V (0 / >1000)
                chp = ch.iloc[:: max(1, len(ch) // 6000)]               # cap plotted points
                fig = go.Figure(go.Scattergl(x=chp["soc"], y=chp["voltage"], mode="markers",
                                marker=dict(size=4, color=chp["event"], colorscale="Viridis", opacity=0.55,
                                            colorbar=dict(title="charge #", thickness=12))))
                fig.update_layout(xaxis=dict(title="state of charge (%)", range=[0, 100]),
                                  yaxis=dict(title="pack voltage (V)"))
                _cc_cap = (f"SoC↔voltage across **{n_ev} charges** (dark = early → yellow = recent). As the pack "
                           f"ages the curve shifts — a fingerprint of fade + rising internal resistance.")
            else:
                chp = ch.iloc[:: max(1, len(ch) // 6000)]               # cap plotted points
                fig = go.Figure(go.Scattergl(x=chp["t"], y=chp["soc"], mode="markers",
                                marker=dict(size=4, color=chp["event"], colorscale="Viridis", opacity=0.55,
                                            colorbar=dict(title="charge #", thickness=12))))
                fig.update_layout(xaxis=dict(title="time"), yaxis=dict(title="state of charge (%)", range=[0, 100]))
                _cc_cap = (f"No pack voltage in this feed → **SoC over time** across **{n_ev} charges**.")
            chart(fig, height=340)
            st.caption(_cc_cap)


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

wl, wr = st.columns([0.5, 0.5], gap="large")
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
        st.metric("Predicted health at warranty end", f"{proj_at_deadline:.0f}%",
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
# 2) STATE OF CHARGE  (LFP-correct: high SoC is tolerated; heat is the stressor)
# ===========================================================================
soc_high = mean_val(g, "frac_soc_high")
if soc_high is not None:
    sm = mean_val(g, "soc_mean"); fl = mean_val(g, "frac_soc_low"); pct_high = soc_high * 100
    section("STATE OF CHARGE", "Where you keep the charge")
    stat_strip([
        ("Typical charge", f"{sm:.0f}%" if sm is not None else "—"),
        ("Time at high charge", f"{pct_high:.0f}%"),
        ("Time at low charge", f"{fl * 100:.0f}%" if fl is not None else "—"),
    ])
    st.write("")
    chart(soc_density_fig(sm, fl, soc_high), height=300)
    st.caption("Approximate shape of your charging habit — how much time your battery spends at low / typical / high "
               "charge. **For LFP this barely affects health** (heat is the real driver — see Temperature); it's just "
               "a picture of *how* you charge, not a health score.")
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
    tser = pd.to_numeric(g["temp_mean"], errors="coerce").dropna()
    frac_hot = float((tser >= HOT_C).mean()) if len(tser) else 0.0
    peak = mean_val(g, "temp_p95")
    if peak is None:
        peak = last_val(g, "temp_max")
    hot_color = RED if frac_hot > 0.4 else (AMBER if frac_hot > 0.15 else GREEN)
    section("TEMPERATURE", "How hot your pack runs")
    stat_strip([
        ("Typical pack temp", f"{temp_mean_v:.0f}°C"),
        ("Time running hot", f"{frac_hot * 100:.0f}%", f"at or above {HOT_C:.0f}°C", hot_color),
        ("Hottest it got", f"{peak:.0f}°C" if peak is not None else "—"),
    ])
    fs = fleet_stat(oem, "temp_mean")
    fig = temp_dist_fig(fs["values"], temp_mean_v) if fs is not None else None
    if fig is not None:
        st.write("")
        chart(fig, height=300)
        st.caption("Your pack temperature against the whole fleet — amber marks the LFP hot zone.")
    why(
        "Heat is the <b>number-one ageing stressor for LFP</b> batteries. A pack that "
        "regularly runs hot loses capacity faster — and the effect compounds when it's "
        "hot <i>and</i> sitting at a high charge. Indian operating temperatures make this "
        "the metric to watch. <b>What helps:</b> park and charge in shade, avoid charging "
        "right after a long hot run, and give the pack a few minutes to cool.",
        sources="Turno LFP fleet analysis",
    )


# ===========================================================================
# 4) USAGE  (monthly km + charging trend — full-width, no mismatched columns)
# ===========================================================================
gm = g.dropna(subset=["age_months"]).sort_values("age_months").copy()
has_km = "km_month" in gm.columns and gm["km_month"].notna().any()
km_avg = float(gm["km_month"].dropna().mean()) if has_km else None
# Charging signal: Bajaj reports real charge cycles; Euler/Mahindra don't, but DO report cumulative Ah
# throughput (cum_ah) — and since a battery is charged ≈ as much as it's driven, the monthly Ah that
# flows through the pack is an honest charging-intensity proxy.
chg_y = chg_name = chg_y2 = chg_lbl = chg_val = chg_tot_lbl = chg_tot_val = None
if "cyc_month" in gm.columns and gm["cyc_month"].notna().any():
    chg_y, chg_name, chg_y2 = gm["cyc_month"], "charge cycles / month", "cycles / mo"
    chg_lbl, chg_val = "Typical charges / month", f"{float(gm['cyc_month'].dropna().mean()):,.0f}"
    if cycles and cycles > 0:
        chg_tot_lbl, chg_tot_val = "Total charge cycles", f"{cycles:,.0f}"
elif "cum_ah" in gm.columns and gm["cum_ah"].notna().sum() >= 2:
    thru = gm["cum_ah"].diff().clip(lower=0)
    if thru.notna().any():
        chg_y, chg_name, chg_y2 = thru, "energy through pack (Ah / mo)", "Ah / mo"
        chg_lbl, chg_val = "Energy through pack / month", f"{float(thru.dropna().mean()):,.0f} Ah"
        chg_tot_lbl, chg_tot_val = "Lifetime energy throughput", f"{float(gm['cum_ah'].dropna().iloc[-1]):,.0f} Ah"
has_chg = chg_y is not None

section("USAGE", "How much you drive and charge" if has_chg else "How much you drive")
usage_items = []
if km_avg is not None:
    usage_items.append(("Distance / month", f"{km_avg:,.0f} km"))
if odo and odo > 0:
    usage_items.append(("Lifetime distance", f"{odo:,.0f} km"))
if chg_lbl:
    usage_items.append((chg_lbl, chg_val))
if chg_tot_lbl:
    usage_items.append((chg_tot_lbl, chg_tot_val))
if usage_items:
    stat_strip(usage_items)
    st.write("")

if has_km or has_chg:
    fig = go.Figure()
    if has_km:
        fig.add_trace(go.Bar(x=gm["age_months"] / 12, y=gm["km_month"], name="km / month",
                             marker_color=BLUE, opacity=0.85))
    if has_chg:
        fig.add_trace(go.Scatter(x=gm["age_months"] / 12, y=chg_y, name=chg_name, yaxis="y2",
                                 mode="lines+markers", line=dict(color=AMBER, width=2.4),
                                 marker=dict(size=5)))
    fig.update_layout(
        bargap=0.32,
        xaxis=dict(title="Battery age (years)"),
        yaxis=dict(title="km / month"),
    )
    if has_chg:
        fig.update_layout(yaxis2=dict(title=chg_y2, overlaying="y", side="right",
                                      color=MUTE, showgrid=False))
    chart(fig, height=340, legend=True)
else:
    st.info("Monthly usage trend isn't available for this vehicle yet.")

note = "Higher mileage and more charging both add wear over time, but on LFP they matter less than heat. "
if chg_name and "cycles" in chg_name:
    note += "The amber line is your actual charge-cycle pattern."
elif has_chg:
    note += ("This fleet doesn't report charge-session counts, so the amber line shows the energy flowing "
             "through the pack each month — a charging-intensity proxy (you charge ≈ as much as you drive).")
else:
    note += "Charge data isn't reported for this vehicle, so we show distance only."
why(note)


# ===========================================================================
# 4b) HOW YOU TREAT YOUR BATTERY — personalised, fleet-relative, LFP-tuned
# ===========================================================================
bt = behaviour_table(oem)
rv = vehicle_behaviour(g)
has_current = "charge_i" in rv and "drive_i" in rv          # Euler / Mahindra (intellicar currents)
has_cycles = ("cyc_month" in rv) or ("cyc_max" in g.columns) or ("cum_cycles" in g.columns)

if has_current or has_cycles or "temp" in rv:
    section("HOW YOU TREAT YOUR BATTERY", "Your habits vs the fleet")

    if has_current:
        cs = care_score(bt, rv)
        grade, gcol = care_grade(cs)
        cl, cr = st.columns([0.4, 0.6], gap="large")
        with cl:
            with st.container(border=True):
                st.markdown(big_number(f"{cs:.0f}" if cs is not None else "–", "/100", gcol),
                            unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:10px;'><span class='pill' "
                            f"style='background:{gcol}26;color:{gcol};'>{grade} usage</span></div>"
                            f"<div style='color:{MUTE};margin-top:10px;'>Battery-care score — higher means "
                            f"gentler on the pack than similar {oem}s.</div>", unsafe_allow_html=True)
        with cr:
            why("This blends how gently you charge, drive, cycle and work the pack — each compared with the rest "
                "of the fleet. It deliberately ignores how full you keep the battery: for your <b>LFP</b> pack a "
                "full charge is fine. <b>Charge to full whenever you like;</b> the real levers are smoother pulls "
                "and keeping the pack cool.")

        # behaviour fingerprint
        chg = gentler_than(_col(bt, "charge_i"), rv.get("charge_i"))
        drv = gentler_than(_col(bt, "drive_i"), rv.get("drive_i"))
        depk = "dod" if ("dod" in bt.columns and "dod" in rv) else ("lowdwell" if "lowdwell" in rv else None)
        dep = gentler_than(_col(bt, depk), rv.get(depk)) if depk else None
        items = [("Charging style", style_word(chg),
                  f"gentler than {chg:.0f}% of fleet" if chg is not None else "—",
                  GREEN if (chg or 0) >= 50 else AMBER),
                 ("Driving style", style_word(drv),
                  f"gentler than {drv:.0f}% of fleet" if drv is not None else "—",
                  GREEN if (drv or 0) >= 50 else AMBER)]
        if dep is not None:
            items.append(("Cycling depth", style_word(dep, ("Deep", "Typical", "Shallow")),
                          f"shallower than {dep:.0f}% of fleet", GREEN if dep >= 50 else AMBER))
        st.write("")
        stat_strip(items)

        # Mahindra: tangible numbers (real capacity, range, cycles) the OEM signals support
        if oem == "Mahindra":
            tang = []
            caps = pd.to_numeric(g["capacity_ah"], errors="coerce").dropna() if "capacity_ah" in g.columns else pd.Series(dtype=float)
            cap_new = float(caps.head(3).mean()) if len(caps) >= 2 else None
            if cap_new and len(caps):
                cap_now = float(caps.iloc[-1]); pctcap = 100 * cap_now / cap_new if cap_new else None
                tang.append(("Pack capacity now", f"{cap_now:.0f} Ah",
                             f"≈{pctcap:.0f}% of ~{cap_new:.0f} Ah when newer" if pctcap else None))
            if "range" in rv:
                rrank = harsher_than(_col(bt, "range"), rv["range"])
                tang.append(("Typical range", f"{rv['range']:.0f} km",
                             f"better than {rrank:.0f}% of similar vehicles" if rrank is not None else "the vehicle's own distance-to-empty"))
            if "throughput" in rv and cap_new:
                tang.append(("Full cycles / month", f"{rv['throughput'] / cap_new:.1f}",
                             "energy through the pack ÷ its capacity"))
            if tang:
                st.write("")
                stat_strip(tang)

        # personalised "what's aging your battery most"
        levers = top_levers(bt, rv)
        if levers:
            parts = " and ".join(f"<b>{lab}</b> (higher than {h:.0f}% of similar vehicles)" for h, lab, _ in levers)
            actions = "; ".join(act for _, _, act in levers)
            why(f"The habits adding the most wear right now are {parts}. Easing these — {actions} — is where you'd "
                f"gain the most extra life. Charging stays simple: charge to full whenever you like; heat is the "
                f"main thing to manage.")
        else:
            why("No single habit stands out as harsh — your pack is mostly seeing normal calendar and cycle ageing. "
                "The biggest lever is heat: keep it parked and charged in the shade.")

    else:  # Bajaj — no current signals; lean on heat rank + cycle count
        items = []
        if "temp" in rv:
            cool = gentler_than(_col(bt, "temp"), rv["temp"])
            if cool is not None:
                items.append(("Cool-running rank", f"cooler than {cool:.0f}%",
                              "heat is the #1 LFP stressor", GREEN if cool >= 50 else AMBER))
        cycmax = last_val(g, "cyc_max") or last_val(g, "cum_cycles")
        if cycmax and cycmax > 0:
            items.append(("Charge-cycles lived", f"{cycmax:,.0f}", "LFP packs last many thousands"))
        cm = mean_val(g, "cyc_month")
        if cm is not None:
            items.append(("Cycles / month", f"{cm:,.0f}", "your recent charging pace"))
        if items:
            stat_strip(items)
            why("Your battery is <b>LFP</b>, built for a long cycle life, so a steady monthly pace is comfortably "
                "normal. The main thing that ages it is heat — charge to full whenever you like, just park in the "
                "shade.")


# ===========================================================================
# 5) EFFICIENCY  (only where driveeff_mean exists — Bajaj)
# ===========================================================================
if "driveeff_mean" in g.columns and g["driveeff_mean"].notna().any():
    ge = g.dropna(subset=["driveeff_mean", "age_months"])
    eff_now = float(ge["driveeff_mean"].iloc[-1])
    fs = fleet_stat(oem, "driveeff_mean")
    section("EFFICIENCY", "How efficiently you drive")
    stat_strip([
        ("Current efficiency", f"{eff_now:.0f}"),
        ("Fleet typical", f"{fs['med']:.0f}" if fs else "—"),
    ])
    st.write("")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ge["age_months"] / 12, y=ge["driveeff_mean"],
                             mode="lines+markers", line=dict(color=GREEN, width=2.6),
                             marker=dict(size=5), name="Your efficiency"))
    if fs is not None:
        fig.add_hline(y=fs["med"], line=dict(color=MUTE, width=1.2, dash="dot"),
                      annotation_text="fleet typical", annotation_position="bottom right",
                      annotation_font=dict(color=MUTE, size=11))
    fig.update_layout(xaxis=dict(title="Battery age (years)"),
                      yaxis=dict(title="efficiency score"))
    chart(fig, height=300)
    why("How efficiently your vehicle turns battery energy into distance. Smoother, "
        "steadier driving keeps this high; lots of hard acceleration and stop-start "
        "traffic pulls it down. It doesn't damage the battery directly, but a higher "
        "score means more range from the same charge.")


# ===========================================================================
# FRIENDLY ONE-SENTENCE SUMMARY
# ===========================================================================
if soh_now <= eol:
    summary = (f"Your battery has reached its end-of-life health line ({soh_now:.0f}%). It still "
               f"runs, but expect noticeably shorter range — a battery check-up is recommended.")
    bg = RED
elif status_label == "Healthy" and survives:
    summary = (f"Your battery is healthy and ageing normally — at {soh_now:.0f}% health and "
               f"predicted to stay above the warranty threshold. Nothing to worry about. 🎉")
    bg = GREEN
elif survives:
    summary = (f"Your battery is ageing at a normal pace ({soh_now:.0f}% health) and is predicted "
               f"to stay healthy through the warranty. Keep charging and driving as usual.")
    bg = GREEN
else:
    summary = (f"Your battery is at {soh_now:.0f}% health and ageing a little faster than typical. "
               f"It's worth watching as you approach the end of warranty.")
    bg = AMBER

st.write("")
st.markdown(f"<div style='background:{PANEL};border:1px solid {LINE};border-left:5px solid {bg};"
            f"border-radius:16px;padding:20px 24px;font-size:1.15rem;line-height:1.55;'>"
            f"💬 {summary}</div>", unsafe_allow_html=True)
st.write("")
st.markdown(f"<div style='color:{FAINT};font-size:0.82rem;'>Turno · Sample customer "
            f"battery-health view. Health, range and predictions are estimates from your "
            f"vehicle's monthly battery data and may vary with real-world use.</div>",
            unsafe_allow_html=True)
