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
    return {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.03,
                                 num_leaves=15, min_child_samples=20, verbose=-1).fit(X, y, sample_weight=w)
            for a in alphas}


def _row(state, stress):
    isa, dfc = _curv(state["age_months"], state["soh"])
    return pd.DataFrame([{**{s: state[s] for s in STATE}, **stress,
                          "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()


def simulate(g, models, horizon, recent_k=6):
    """Roll forward assuming recent-median stress persists. Returns quantile SoH columns (q10/q50/q90)."""
    g = g.sort_values("month"); last = g.iloc[-1]
    stress = g.iloc[-recent_k:][STRESS].median().to_dict()
    st = {s: float(last[s]) for s in STATE}
    sh = {q: float(last["soh"]) for q in models}
    qs = sorted(models); out = []
    for _ in range(max(int(horizon), 1)):
        x = _row(st, stress)
        for q in qs:
            sh[q] = sh[q] - max(models[q].predict(x)[0], 0)
        st.update(soh=sh[qs[len(qs) // 2]], age_months=st["age_months"] + 1,
                  cum_km=st["cum_km"] + stress.get("km_month", 0.0),
                  cum_cycles=st["cum_cycles"] + stress.get("cyc_month", 0.0))
        out.append([sh[q] for q in qs])
    return pd.DataFrame(out, columns=[f"q{int(q*100)}" for q in qs])
