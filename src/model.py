"""Condition-aware degradation rate model (shared by all forecasting notebooks).

Predicts monthly SoH loss from operating conditions + curvature features, then rolls forward.
Upgrades over the first version:
  - curvature features: `inv_sqrt_age` (1/√age — lets the rate be high early / low late, i.e. the
    √t-fade shape) and `soh_deficit` (100−SoH — lets loss accelerate near the low-SoH knee);
  - sample weighting by loss so the slow plateau doesn't drown out real degradation months.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

STATE = ["soh", "age_months", "cum_ah", "cum_km", "odo_max"]
# Pruned 2026-06: dropped volt_min (const), temp_mean & frac_drive (sparse/noisy for Mahindra).
# NOTE: dod_mean, wh_per_km, lat_mean, lon_mean (climate) ARE computed in the feature table but
# left OUT of the model — validation showed they hurt the degrading-vehicle backtest (overfit on
# small regional data). Re-add here if more/diverse data later makes them helpful.
STRESS = ["ah_throughput", "cur_abs_mean", "cur_abs_p95", "cur_dis_mean", "cur_chg_mean", "soc_mean",
          "frac_soc_high", "frac_soc_low", "volt_mean", "volt_max", "temp_max", "km_month", "dte_mean"]
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
    t["w"] = 1.0 + t["loss"].clip(0, 5)          # down-weight plateau (low-loss) months
    return t


def train_quantiles(t, alphas=(0.1, 0.5, 0.9)):
    X, y, w = t[FEATS].to_numpy(), t["loss"].to_numpy(), t["w"].to_numpy()
    common = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, verbose=-1)
    models = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **common).fit(X, y, sample_weight=w)
              for a in alphas}
    # Expected (mean) monthly loss for the CENTRAL trajectory. SoH here is a staircase — most months lose
    # nothing — so the MEDIAN monthly loss is ~0 and a q50 forecast stays flat; the mean (lifted above zero
    # by the occasional step-downs) is what actually bends the central forecast toward end-of-life.
    models["mean"] = lgb.LGBMRegressor(objective="regression", **common).fit(X, y, sample_weight=w)
    return models


def _row(state, stress):
    isa, dfc = _curv(state["age_months"], state["soh"])
    return pd.DataFrame([{**{s: state[s] for s in STATE}, **stress,
                          "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()


def simulate(g, models, horizon, recent_k=6):
    """Roll the rate model forward assuming recent-median stress persists. The CENTRAL line (q50) uses the
    EXPECTED (mean) monthly loss — for a staircase SoH the median monthly loss is ~0, so a true-median
    forecast never declines; the mean bends it toward EoL. q10/q90 stay genuine loss-quantile bands.
    Returns a DataFrame with q10/q50/q90 SoH columns."""
    g = g.sort_values("month"); last = g.iloc[-1]
    stress = g.iloc[-recent_k:][STRESS].median().to_dict()
    st = {s: last[s] for s in STATE}
    lo_m, hi_m = models[0.1], models[0.9]
    mid_m = models.get("mean", models[0.5])          # expected loss (falls back to median if a model lacks it)
    # Cap per-step loss so the soh_deficit feedback (loss accelerates near the knee) can't extrapolate a
    # runaway crash past the reliably-observed monthly loss (~p90 of training loss). Prevents flat AND runaway.
    MAX_STEP = 1.2
    lo = mid = hi = float(last["soh"]); out = []
    for _ in range(max(int(horizon), 1)):
        x = _row(st, stress)
        lo = max(lo - min(max(lo_m.predict(x)[0], 0.0), MAX_STEP), 0.0)   # 0.1-loss quantile -> upper band
        mid = max(mid - min(max(mid_m.predict(x)[0], 0.0), MAX_STEP), 0.0)  # expected loss -> central
        hi = max(hi - min(max(hi_m.predict(x)[0], 0.0), MAX_STEP), 0.0)   # 0.9-loss quantile -> lower band
        # Freeze cum_ah: growing it past the training range makes the tree extrapolate to ~0 loss and
        # flatline the forecast (see bajaj_model). Age + soh_deficit still evolve the prediction.
        st.update(soh=mid, age_months=st["age_months"] + 1)
        out.append([lo, mid, hi])
    return pd.DataFrame(out, columns=["q10", "q50", "q90"])


def free_run_observed(g, model_med):
    """Free-run predicted SoH over a vehicle's OBSERVED months using its actual per-month stress
    (predicted-SoH state feeds back). For actual-vs-predicted validation."""
    g = g.sort_values("month").reset_index(drop=True)
    pred = [g["soh"].iloc[0]]
    gap = g["month"].diff().dt.days / 30.4
    for i in range(1, len(g)):
        row = g.iloc[i - 1]
        state = {**{s: (pred[-1] if s == "soh" else row[s]) for s in STATE}}
        x = _row(state, {s: row[s] for s in STRESS})
        step = max(model_med.predict(x)[0], 0) * (gap.iloc[i] if gap.iloc[i] > 0 else 1)
        pred.append(pred[-1] - step)
    return np.array(pred)
