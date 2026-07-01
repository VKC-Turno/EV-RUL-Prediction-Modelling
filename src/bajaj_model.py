"""Condition-aware degradation rate model for Bajaj (reported-SoH cohort).

Same shape as src/model.py (predict monthly SoH loss from conditions + curvature, roll forward), but
with Bajaj's feature set: NO current/voltage (the feed lacks them), so it leans on thermal, SoC dwell,
usage (km/cycles) and the √t curvature. Target is the BMS-reported SoH (already monotone-cleaned in the
feature table; not renormalised to 100, so aged vehicles legitimately start below 100%).

Note: charge cycles were shown NOT to add predictive value for SoH on this fleet (degradation is
calendar-driven), but they're kept as low-weight features here — XGBoost/LightGBM tolerate them.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

STATE = ["soh", "age_months", "cum_km", "cum_cycles", "odo_max"]
STRESS = ["temp_mean", "temp_max", "temp_p95", "amb_temp_mean", "soc_mean", "frac_soc_high",
          "frac_soc_low", "driveeff_mean", "km_month", "cyc_month"]
CURV = ["inv_sqrt_age", "soh_deficit"]
FEATS = STATE + STRESS + CURV


def _curv(age_months, soh):
    return 1.0 / np.sqrt(np.maximum(age_months, 0) + 1.0), 100.0 - soh


def build_transitions(m, max_gap=3.0):
    """feature_table -> per-month transition rows with target monthly `loss`, `gap`, weight `w`."""
    parts = []
    for vin, g in m.groupby("vin"):
        g = g.sort_values("month").reset_index(drop=True)
        gap = g["month"].diff().shift(-1).dt.days / 30.4
        loss = ((g["soh"] - g["soh"].shift(-1)) / gap).clip(lower=0)
        r = g[STATE + STRESS].copy()
        r["inv_sqrt_age"], r["soh_deficit"] = _curv(g["age_months"].to_numpy(), g["soh"].to_numpy())
        r["vin"] = vin; r["loss"] = loss.values; r["gap"] = gap.values
        parts.append(r)
    t = pd.concat(parts, ignore_index=True)
    t = t[(t["gap"] <= max_gap) & t["loss"].notna()].copy()
    t["w"] = 1.0 + t["loss"].clip(0, 5)          # up-weight real-decline months
    return t


def train_quantiles(t, alphas=(0.1, 0.5, 0.9)):
    X, y, w = t[FEATS].to_numpy(), t["loss"].to_numpy(), t["w"].to_numpy()
    common = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, verbose=-1)
    models = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **common).fit(X, y, sample_weight=w)
              for a in alphas}
    # Expected (mean) monthly loss for the CENTRAL line: ~46% of Bajaj months lose nothing, so the conditional
    # median loss is ~0 for gentle vehicles and a q50 forecast stays flat; the mean bends it toward EoL.
    models["mean"] = lgb.LGBMRegressor(objective="regression", **common).fit(X, y, sample_weight=w)
    return models


def _row(state, stress):
    isa, dfc = _curv(state["age_months"], state["soh"])
    return pd.DataFrame([{**{s: state[s] for s in STATE}, **stress,
                          "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()


def simulate(g, models, horizon, recent_k=6):
    """Roll forward assuming recent stress persists. Central (q50) uses EXPECTED (mean) loss so a gentle
    vehicle whose conditional MEDIAN loss is ~0 still bends toward EoL; q10/q90 stay genuine quantile bands.
    Per-step loss is capped so the knee feedback can't run away. Returns q10/q50/q90 SoH columns."""
    g = g.sort_values("month"); last = g.iloc[-1]
    stress = g.iloc[-recent_k:][STRESS].median().to_dict()
    st = {s: float(last[s]) for s in STATE}
    lo_m, hi_m = models[0.1], models[0.9]
    mid_m = models.get("mean", models[0.5])          # expected loss (falls back to median if absent)
    MAX_STEP = 1.2
    lo = mid = hi = float(last["soh"]); out = []
    for _ in range(max(int(horizon), 1)):
        x = _row(st, stress)
        lo = max(lo - min(max(lo_m.predict(x)[0], 0.0), MAX_STEP), 0.0)   # 0.1-loss quantile -> upper band
        mid = max(mid - min(max(mid_m.predict(x)[0], 0.0), MAX_STEP), 0.0)  # expected loss -> central
        hi = max(hi - min(max(hi_m.predict(x)[0], 0.0), MAX_STEP), 0.0)   # 0.9-loss quantile -> lower band
        # Freeze cum_km / cum_cycles: growing them past the (young-fleet) training range makes the tree
        # extrapolate to ~0 loss and the forecast flatlines. Age + soh_deficit still evolve the prediction.
        st.update(soh=mid, age_months=st["age_months"] + 1)
        out.append([lo, mid, hi])
    return pd.DataFrame(out, columns=["q10", "q50", "q90"])
