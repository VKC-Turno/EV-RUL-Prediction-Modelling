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
    return {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=500, learning_rate=0.03,
                                 num_leaves=15, min_child_samples=20, verbose=-1).fit(X, y, sample_weight=w)
            for a in alphas}


def _row(state, stress):
    isa, dfc = _curv(state["age_months"], state["soh"])
    return pd.DataFrame([{**{s: state[s] for s in STATE}, **stress,
                          "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()


def simulate(g, models, horizon, recent_k=6):
    """Roll the rate model forward assuming recent-median stress persists. Returns DataFrame of
    quantile SoH columns (q10/q50/q90)."""
    g = g.sort_values("month"); last = g.iloc[-1]
    stress = g.iloc[-recent_k:][STRESS].median().to_dict()
    st = {s: last[s] for s in STATE}
    sh = {q: last["soh"] for q in models}
    qs = sorted(models); out = []
    for _ in range(max(int(horizon), 1)):
        x = _row(st, stress)
        for q in qs:
            sh[q] = sh[q] - max(models[q].predict(x)[0], 0)
        st.update(soh=sh[qs[len(qs) // 2]], age_months=st["age_months"] + 1,
                  cum_ah=st["cum_ah"] + stress.get("ah_throughput", 0))
        out.append([sh[q] for q in qs])
    return pd.DataFrame(out, columns=[f"q{int(q*100)}" for q in qs])


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
