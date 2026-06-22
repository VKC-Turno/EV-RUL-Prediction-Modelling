"""Vectorized SoH computation (coulomb counting) — no Python per-session loops.

The heavy session math is pure vectorized groupby/cumsum/shift, so it runs unchanged on a
cuDF (GPU) frame or a pandas (CPU) frame. Pass backend='gpu' to use RAPIDS cuDF if installed.
"""
import numpy as np
import pandas as pd

GAP_S = 300.0          # >5 min gap starts a new continuous-logging session
MIN_ROWS = 5           # min samples per session
MIN_DSOC = 2.0         # min |ΔSoC| per session (%)
CAP_BOUNDS = (40.0, 400.0)   # plausible per-session capacity (Ah)
MIN_MONTH_SOC = 30.0   # a month's sessions must collectively span >=30% SoC (pooled coverage)
SMOOTH_WIN = 5         # rolling-median window (months) for the capacity series
BASE_AGE_LO, BASE_AGE_HI = 1.0, 12.0   # baseline = median capacity in this age window (skip settling)
MAX_DROP_PER_MONTH = 6.0   # cap physical SoH loss (pp/month); larger jumps are artifacts


def _session_caps(df):
    """Per-session capacity (Ah) from columns [vin, t, soc, current]. Works on pandas or cuDF."""
    df = df.sort_values(["vin", "t"]).reset_index(drop=True)
    # seconds between consecutive samples within a VIN (NaN at each VIN's first row)
    dt_ns = df.groupby("vin")["t"].diff()
    dt = dt_ns.dt.total_seconds() if hasattr(dt_ns.dt, "total_seconds") else dt_ns.astype("int64") / 1e9
    brk = dt.isna() | (dt > GAP_S) | (dt <= 0)
    df = df.assign(dt=dt.fillna(0.0), sid=brk.astype("int32").cumsum())
    cur_prev = df.groupby("sid")["current"].shift()
    df = df.assign(dQ=((df["current"] + cur_prev) / 2.0 * df["dt"] / 3600.0).fillna(0.0))
    g = df.groupby("sid").agg(vin=("vin", "first"), t0=("t", "first"),
                              soc0=("soc", "first"), soc1=("soc", "last"),
                              dQ=("dQ", "sum"), n=("soc", "size"))
    return g


def coulomb_capacity_monthly(df, backend="auto"):
    """Return (pandas DataFrame [vin, month, capacity_ah, n_sessions, tot_dsoc], used_gpu).

    Monthly capacity is the **ΔSoC-weighted pooled** estimate: capacity = Σ|∫I·dt| / Σ(|ΔSoC|/100)
    over the month's sessions. This weights by SoC swing, so large reliable charge/discharge
    segments dominate and the many tiny-ΔSoC (noisy, CV~30%) sessions barely affect the ratio —
    far more stable than a per-session median/percentile."""
    use_gpu = False
    if backend in ("auto", "gpu"):
        try:
            import cudf
            df = df if str(type(df)).startswith("<class 'cudf") else cudf.from_pandas(df)
            use_gpu = True
        except Exception:
            if backend == "gpu":
                raise
    g = _session_caps(df)
    if use_gpu:
        g = g.to_pandas()
    g["dSoC"] = (g["soc1"] - g["soc0"]).abs()
    g = g[(g["n"] >= MIN_ROWS) & (g["dSoC"] >= MIN_DSOC)].copy()
    g["aQ"] = g["dQ"].abs()
    g["cap_sess"] = g["aQ"] / (g["dSoC"] / 100.0)
    g = g[g["cap_sess"].between(*CAP_BOUNDS)]          # drop per-session outliers
    g["month"] = g["t0"].dt.to_period("M").dt.to_timestamp()
    out = (g.groupby(["vin", "month"])
             .agg(aQ=("aQ", "sum"), tot_dsoc=("dSoC", "sum"), n_sessions=("aQ", "size"))
             .reset_index())
    out["capacity_ah"] = out["aQ"] / (out["tot_dsoc"] / 100.0)   # ΔSoC-weighted pooled capacity
    out = out[(out["tot_dsoc"] >= MIN_MONTH_SOC) & out["capacity_ah"].between(*CAP_BOUNDS)]
    return out[["vin", "month", "capacity_ah", "n_sessions", "tot_dsoc"]].reset_index(drop=True), use_gpu


def vehicle_anchor(cap_month, reg):
    """vin -> anchor date for age 0 / SoH 100: the registration date when available and not later
    than first telemetry; otherwise the first observed month (with a `used_reg` flag)."""
    out = {}
    for vin, g in cap_month.groupby("vin"):
        first = g["month"].min(); r = reg.get(vin) if reg else None
        use = r is not None and pd.notna(r) and r <= first
        out[vin] = (r if use else first, use)
    return out


def capacity_to_soh(cap_month, reg=None, max_drop=MAX_DROP_PER_MONTH):
    """Per-VIN SoH curve with **SoH = 100% anchored at the registration date** (age 0).

    `reg`: dict vin -> registration Timestamp. Age is measured from registration (true calendar
    age); when a vehicle's telemetry starts after registration, the full-capacity reference is
    back-extrapolated along the fade trend to age 0 — so the first observed point sits below 100%
    by the amount it aged before telemetry began. Falls back to first-telemetry anchoring when the
    registration date is missing or later than first telemetry. Output adds `age_months` (true age)
    and `used_reg`."""
    reg = reg or {}
    anchor = vehicle_anchor(cap_month, reg)
    parts = []
    for vin, g in cap_month.groupby("vin"):
        g = g.sort_values("month").copy()
        base_t, used = anchor[vin]
        age = ((g["month"] - base_t).dt.days / 30.4).to_numpy()
        g["age_months"] = age; g["used_reg"] = used
        cap = g["capacity_ah"].rolling(SMOOTH_WIN, min_periods=1, center=True).median().to_numpy()
        amin = age.min()                                   # months from anchor to first telemetry (gap)
        early = cap[(age >= amin) & (age <= amin + (BASE_AGE_HI - BASE_AGE_LO))]
        early_base = np.median(early) if len(early) else cap[0]
        # full-capacity reference at the anchor = early observed capacity grossed up by the fade
        # rate over ONLY the registration->telemetry gap (so SoH starts ~100% at the anchor and
        # sits below 100 only in proportion to a real gap; no global-fit overshoot).
        if len(age) >= 4 and age.max() > age.min():
            slope = np.polyfit(age, cap, 1)[0]
            rate = min(max(-slope / max(early_base, 1e-9), 0.0), 0.01)   # fractional fade/month (0..1%)
        else:
            rate = 0.0
        cap0 = early_base * (1.0 + rate * max(amin, 0.0))
        raw = np.clip(100.0 * cap / cap0, None, 100.0)
        gap = (g["month"].diff().dt.days / 30.4).fillna(1.0).to_numpy()
        soh = np.empty(len(raw)); soh[0] = min(raw[0], 100.0)
        for i in range(1, len(raw)):
            target = min(raw[i], soh[i - 1])                       # monotonic non-increasing
            floor = soh[i - 1] - max_drop * max(gap[i], 1e-9)      # cap the drop rate
            soh[i] = max(target, floor)
        g["soh"] = soh
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
