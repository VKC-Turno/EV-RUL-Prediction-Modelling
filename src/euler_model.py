"""Euler condition-aware SoH degradation + forecast model.

Target = validated BMS-capacity SoH (high-SoC, isotonic) from
``data/euler/features/feature_table.parquet`` (one row per vin-month).

Two complementary forecasters live here, both API-preserving:

  RATE model      predict monthly ΔSoH (loss), roll forward.  This is the original
                  public API (``build_transitions`` / ``train`` / ``free_run``) that the
                  dashboard and notebooks import — kept working, but hardened against the
                  known failure mode (over-predicting loss on genuinely flat vehicles) with
                  a flat-vs-degrading gate + stronger regularization.

  TRAJECTORY model  the improved, well-validated forecaster.  It predicts *cumulative loss
                  from an anchor month* as a smooth √-age curve whose slope is conditioned on
                  per-vehicle operating stress, and emits P10/P50/P90 quantile bands
                  (LightGBM quantile, falling back to sklearn GradientBoosting).  ``forecast``
                  is the entry point; the notebook + backtest use it.

Everything is NaN-tolerant (``imbalance_mean`` is NaN for older vehicles) — XGBoost / LightGBM
handle NaN natively and the engineered features avoid feeding NaN into linear math.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# ───────────────────────── feature groups (public — dashboard imports these) ─────────────────────────
STATE = ["soh", "age_months", "cum_ah", "cum_km", "odo_max"]
STRESS = ["ah_throughput", "cur_abs_mean", "cur_abs_p95", "cur_chg_mean", "cur_dis_mean", "soc_mean",
          "frac_soc_high", "frac_soc_low", "volt_mean", "volt_max", "temp_mean", "temp_max",
          "km_month", "dod_mean", "crate_p95", "imbalance_mean"]
CURV = ["inv_sqrt_age", "soh_deficit"]
FEATS = STATE + STRESS + CURV

# Stress features that actually condition the *trajectory* model (pruned: drop weak / leaky ones,
# keep usage-rate + thermal + cycling-depth, which physically drive Li-ion fade).  imbalance_mean
# is intentionally excluded here because it is NaN for the whole older sub-cohort.
TRAJ_STRESS = ["ah_throughput", "km_month", "cur_abs_p95", "crate_p95", "temp_mean", "temp_max",
               "frac_soc_high", "frac_soc_low", "dod_mean", "soc_mean"]

EOL = 80.0           # end-of-first-life SoH threshold
FLAT_TAIL_PP = 1.0   # a vehicle whose observed history lost < this is treated as ~flat
OWN_SLOPE_THR = 0.4  # obs_rate (pp/mo) at which own-slope continuation reaches full blend weight
OWN_SLOPE_WMAX = 0.7 # max weight given to the vehicle's own recent slope vs the pooled trajectory


# ════════════════════════════════════════════════════════════════════════════════════════════════════
#  RATE MODEL  (original public API — preserved & hardened)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
def _curv(age, soh):
    return 1.0 / np.sqrt(np.maximum(age, 0) + 1.0), 100.0 - soh


# A single-step raw SoH drop bigger than this (pp, not gap-normalised) is a BMS capacity-recalibration
# artifact (the isotonic full_cap holds a flat plateau then steps), NOT a month of real wear.  Such
# transitions used to teach the rate model huge spurious losses (max ~24 pp/mo) AND, by being the only
# large-loss examples, both inflated held-out RMSE and biased the prior.  We winsorise their target loss
# to GLITCH_CAP rather than drop them, which keeps a (bounded) large-loss signal so the model can still
# predict genuine fast decline, without the artifact extremes.  Backtest-validated (LOVO K=6 holdout):
# winsorise+cap beats both keep-raw (RMSE) and drop (degrading-cohort under-prediction).
GLITCH_PP = 6.0      # |single-step SoH change| above this is treated as a capacity glitch. TIGHTENED 10->6
#                      (2026-06-30): the old 10pp bound missed the 6-10pp BMS re-estimation jumps (soh_audit
#                      CLIFF=6); confirmatory 5fold×3seed free-run MAE 3.42->3.29 (-0.12pp), winsorise≈drop so
#                      we keep winsorise (preserves bounded large-loss signal; drop under-predicts degraders).
#                      NOTE: this affects the RATE model (diagnostics/free_run) only. Cleaning the deployed
#                      TRAJECTORY model's SoH target was TESTED (cliff-interp / median3 / monotone / theil-sen)
#                      and REJECTED — null vs seed noise, adversarially verified: its cumulative-loss + √-horizon
#                      formulation already absorbs the cliff/stuck artifacts. Keep the trajectory target RAW.
GLITCH_CAP = 4.0     # winsorise the glitch transition's monthly loss target to this (pp/mo)


def build_transitions(m, max_gap=3.0):
    """One row per vin-month transition with the realised monthly loss as target.

    Loss is the *forward* SoH drop normalised to a per-month rate.  Rows are weighted up where a
    real drop happened so the gradient-boosted rate model does not collapse to predicting 0.  Months
    straddling a BMS capacity-recalibration glitch (single-step |ΔSoH| > ``GLITCH_PP``) have their
    loss target winsorised to ``GLITCH_CAP`` so the artifact does not teach a spurious huge loss.
    """
    parts = []
    for vin, g in m.groupby("vin"):
        g = g.sort_values("month").reset_index(drop=True)
        gap = g["month"].diff().shift(-1).dt.days / 30.4
        raw_drop = g["soh"] - g["soh"].shift(-1)            # un-normalised forward SoH change
        loss = (raw_drop / gap).clip(lower=0)
        glitch = raw_drop.abs() > GLITCH_PP
        r = g[STATE + STRESS].copy()
        r["inv_sqrt_age"], r["soh_deficit"] = _curv(g["age_months"].to_numpy(), g["soh"].to_numpy())
        r["vin"] = vin
        r["loss"] = loss.values
        r["gap"] = gap.values
        r["glitch"] = glitch.values
        parts.append(r)
    t = pd.concat(parts, ignore_index=True)
    t = t[(t["gap"] <= max_gap) & t["loss"].notna()].copy()
    # winsorise capacity-glitch losses instead of dropping (preserve bounded large-loss signal)
    t.loc[t["glitch"], "loss"] = np.minimum(t.loc[t["glitch"], "loss"], GLITCH_CAP)
    t = t.drop(columns=["glitch"])
    t["w"] = 1.0 + t["loss"].clip(0, 5)
    return t


# Mean observed fleet monthly loss is recomputed at fit time and stored on the model so ``free_run``
# can apply a small additive calibration making the mean predicted loss match the fleet mean (the
# zero-inflated, heavily-regularised loss target otherwise regresses toward 0 and under-predicts).
def train(t):
    """Train the monthly-loss rate model.

    Re-tuned (vs the heavily-regularised preliminary version) to let genuine decline through: depth 4,
    lr 0.04, lighter min_child_weight/reg_lambda/gamma — guarded against over-fitting / flat-vehicle
    over-prediction by the LOVO K=6 backtest, not by suppressing the signal.  A small global bias is
    fitted so the mean predicted monthly loss equals the observed fleet mean loss rate; it is stashed
    on the returned regressor as ``_cal_bias`` and applied in ``free_run``.  Signature/return type
    (an XGBRegressor) are unchanged."""
    mdl = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.04, max_depth=4, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=4, reg_lambda=1.5, gamma=0.05,
        n_jobs=8, verbosity=0,
    ).fit(t[FEATS].to_numpy(), t["loss"].to_numpy(), sample_weight=t["w"].to_numpy())
    # global mean-loss calibration: shift so mean(pred) == observed fleet mean loss rate
    pred_mean = float(np.clip(mdl.predict(t[FEATS].to_numpy()), 0.0, None).mean())
    mdl._cal_bias = float(t["loss"].mean() - pred_mean)
    return mdl


def _recent_loss_rate(g):
    """Observed monthly SoH-loss rate over a vehicle's recent history (pp/month)."""
    g = g.sort_values("month")
    if len(g) < 3:
        return 0.0
    tail = g.iloc[-min(8, len(g)):]
    span = (tail["age_months"].iloc[-1] - tail["age_months"].iloc[0])
    if span <= 0:
        return 0.0
    return max((tail["soh"].iloc[0] - tail["soh"].iloc[-1]) / span, 0.0)


# free_run gate zones, keyed off the vehicle's recent observed loss rate (pp/mo):
FLAT_THR = 0.04      # below this the vehicle is ~flat -> damp the model's manufactured loss
DEG_THR = 0.10       # at/above this the vehicle is genuinely degrading -> continue/accelerate
FLAT_DAMP = 0.35     # on flat vehicles, only take this fraction of the model's predicted loss
DEG_ACCEL = 1.25     # on degrading vehicles, continue at >= obs_rate * this (late-life acceleration)


def free_run(g, mdl, months, gate=True):
    """Roll the rate model forward ``months`` steps from a vehicle's last observation.

    Condition-aware 3-zone gate (keyed off the vehicle's recent observed loss rate ``obs_rate``),
    plus the global mean-loss calibration fitted in ``train``:

      * flat (obs_rate < FLAT_THR):   step = FLAT_DAMP · pred — don't manufacture loss on flat vehicles.
      * degrading (obs_rate ≥ DEG_THR): step = max(pred, obs_rate · DEG_ACCEL) — never pull a real
                                        decliner DOWN to a lower historical rate, and allow late-life
                                        acceleration.  This is the fix for the documented under-
                                        prediction (forecasts too flat) on genuinely degrading vehicles.
      * in between: linearly blend the two.

    The previous gate damped EVERY vehicle's loss toward its backward-looking ``obs_rate``, which
    suppressed future decline on accelerating vehicles (the user's "too optimistic" complaint).  Now
    the damping is confined to flat vehicles and the model is allowed to over-shoot the recent rate on
    degraders.  ``gate=False`` keeps the raw (calibrated) model for diagnostics.  Backtest-validated
    (LOVO K=6): overall signed bias ≈0 and held-out RMSE materially lower than the old gate.
    """
    g = g.sort_values("month")
    last = g.iloc[-1]
    stress = g.iloc[-6:][STRESS].median().to_dict()
    st = {s: float(last[s]) for s in STATE}
    soh = float(last["soh"])
    obs_rate = _recent_loss_rate(g)
    cal_bias = float(getattr(mdl, "_cal_bias", 0.0))
    out = []
    for _ in range(int(months)):
        isa, dfc = _curv(st["age_months"], st["soh"])
        x = pd.DataFrame([{**{s: st[s] for s in STATE}, **stress,
                           "inv_sqrt_age": isa, "soh_deficit": dfc}])[FEATS].to_numpy()
        pred = max(float(mdl.predict(x)[0]) + cal_bias, 0.0)
        if not gate:
            step = pred
        elif obs_rate < FLAT_THR:
            step = FLAT_DAMP * pred
        elif obs_rate >= DEG_THR:
            step = max(pred, obs_rate * DEG_ACCEL)
        else:                                                # transition zone: blend flat <-> degrading
            w = (obs_rate - FLAT_THR) / (DEG_THR - FLAT_THR)
            step = (1.0 - w) * (FLAT_DAMP * pred) + w * max(pred, obs_rate * DEG_ACCEL)
        soh = min(max(soh - max(step, 0.0), 0.0), 100.0)     # monotone within [0, 100]
        st.update(soh=soh, age_months=st["age_months"] + 1,
                  cum_ah=st["cum_ah"] + stress.get("ah_throughput", 0))
        out.append(soh)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY MODEL  (improved forecaster with P10/P50/P90 bands)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                                            # pragma: no cover
    _HAS_LGB = False
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = (0.1, 0.5, 0.9)

# Horizon-scaled empirical band half-widths (SoH pp), calibrated on the LOVO P50 residuals.
# Band lower (P10) and upper (P90) offsets grow ~√horizon.  These defaults reproduce ~80% P10–P90
# coverage on the current fleet; ``calibrate_band`` recomputes them from a fresh backtest residual
# table so the band stays honest as more vehicles arrive.  (lo is the downside half-width: actuals
# sit BELOW P50 more often than above, so lo > hi.)
BAND = {"lo": 1.56, "hi": 1.18}      # half-width per √month at horizon h: lo*√h below, hi*√h above
                                     # (calibrated to ≈0.80 LOVO P10–P90 coverage; see euler_backtest)


def calibrate_band(resid_df, target=0.80):
    """Fit horizon-scaled band offsets from a residual table (cols: ``h``, ``resid`` = actual-P50).

    Returns {'lo','hi'} such that P50 - lo·√h and P50 + hi·√h give ≈ ``target`` two-sided coverage.
    """
    r = resid_df.copy()
    r["sq"] = np.sqrt(r["h"].clip(lower=1))
    a = (1 - target) / 2
    # scale-free offsets: per-√h quantiles of the standardized residual
    z = r["resid"] / r["sq"]
    lo = float(-z.quantile(a))
    hi = float(z.quantile(1 - a))
    return {"lo": max(lo, 0.3), "hi": max(hi, 0.3)}


def _vehicle_summary(g):
    """Per-vehicle static descriptors used to set the trajectory slope: recent operating stress
    (median of last 6 months) + the early-life observed loss rate.  NaN-tolerant."""
    g = g.sort_values("month")
    rec = g.iloc[-min(6, len(g)):][TRAJ_STRESS].median()
    return rec


def build_traj_samples(m, min_hist=4):
    """Expand each vehicle into (anchor, horizon) → cumulative-loss samples.

    For every anchor month with ≥ ``min_hist`` months of history, and every later observed month,
    emit one row: features = anchor state + recent stress + Δage, target = SoH lost between anchor
    and the later month.  The model thus learns *cumulative* loss as a function of how far ahead we
    forecast and under what conditions — which rolls forward stably and yields calibratable bands.
    """
    rows = []
    for vin, g in m.groupby("vin"):
        g = g.sort_values("month").reset_index(drop=True)
        if len(g) < min_hist + 1:
            continue
        for i in range(min_hist - 1, len(g) - 1):
            anchor = g.iloc[i]
            rec = g.iloc[max(0, i - 5):i + 1][TRAJ_STRESS].median()
            # early observed slope at the anchor (pp/√month) — the model's main degradation signal
            hist = g.iloc[:i + 1]
            obs_rate = _recent_loss_rate(hist)
            a0 = float(anchor["age_months"])
            s0 = float(anchor["soh"])
            for j in range(i + 1, len(g)):
                fut = g.iloc[j]
                dage = float(fut["age_months"] - a0)
                if dage <= 0:
                    continue
                row = {
                    "vin": vin,
                    "age0": a0, "soh0": s0, "deficit0": 100.0 - s0,
                    "obs_rate": obs_rate, "dage": dage, "sqrt_dage": np.sqrt(dage),
                    "cum_loss": float(anchor["soh"] - fut["soh"]),
                }
                for f in TRAJ_STRESS:
                    row[f] = float(rec[f]) if pd.notna(rec[f]) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


TRAJ_FEATS = ["age0", "soh0", "deficit0", "obs_rate", "dage", "sqrt_dage"] + TRAJ_STRESS


def train_traj(samples, n_estimators=400):
    """Fit the central (P50) cumulative-loss model.  One LightGBM quantile-0.5 regressor (or a
    sklearn GBR fallback); uncertainty is supplied separately by the calibrated empirical band, which
    is honest about the dominant which-vehicle-am-I uncertainty that pooled quantiles can't see."""
    X = samples[TRAJ_FEATS]
    y = samples["cum_loss"].clip(lower=0).to_numpy()
    w = (1.0 + samples["cum_loss"].clip(0, 25).to_numpy() * 0.15)   # weight up real degradation
    if _HAS_LGB:
        mdl = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, n_jobs=8, verbosity=-1)
        mdl.fit(X.to_numpy(), y, sample_weight=w)
        impute = None
    else:                                                  # pragma: no cover
        mdl = GradientBoostingRegressor(
            loss="quantile", alpha=0.5, n_estimators=n_estimators, learning_rate=0.03,
            max_depth=3, subsample=0.8, min_samples_leaf=15)
        impute = np.nanmedian(X.to_numpy(), axis=0)
        mdl.fit(np.where(np.isnan(X.to_numpy()), impute, X.to_numpy()), y, sample_weight=w)
    return {"p50": mdl, "_impute": impute, "band": dict(BAND)}


def _predict_p50(models, X):
    mdl = models["p50"]
    if models.get("_impute") is None:
        return mdl.predict(X)
    return mdl.predict(np.where(np.isnan(X), models["_impute"], X))     # pragma: no cover


def forecast(g, models, months, quantiles=QUANTILES):
    """Forecast SoH for ``months`` ahead from a vehicle's history → dict of quantile→array.

    The P50 path is the conditioned cumulative-loss model rolled out with monotone (non-increasing
    SoH) loss; P10/P90 are an empirical √-horizon envelope around P50, calibrated to ≈80% coverage.
    Genuinely-flat vehicles (≈100% SoH, no observed slope) get the central path pinned to their
    recent slope and a tight band, so the model does not manufacture degradation on them.
    Returns {0.1: [...], 0.5: [...], 0.9: [...]} (P10 = pessimistic/low SoH, P90 = optimistic).
    """
    g = g.sort_values("month")
    last = g.iloc[-1]
    rec = _vehicle_summary(g)
    obs_rate = _recent_loss_rate(g)
    a0 = float(last["age_months"])
    s0 = float(last["soh"])
    flat = (s0 >= 99.5) or (_history_drop(g) < FLAT_TAIL_PP and obs_rate < 0.05)

    base = {"age0": a0, "soh0": s0, "deficit0": 100.0 - s0, "obs_rate": obs_rate}
    stress = {f: (float(rec[f]) if pd.notna(rec[f]) else np.nan) for f in TRAJ_STRESS}
    dages = np.arange(1, months + 1, dtype=float)
    rows = [{**base, **stress, "dage": d, "sqrt_dage": np.sqrt(d)} for d in dages]
    X = pd.DataFrame(rows)[TRAJ_FEATS].to_numpy()

    loss = np.maximum.accumulate(np.clip(_predict_p50(models, X), 0, None))
    p50_pool = s0 - loss
    # Blend the pooled trajectory with a continuation of THIS vehicle's own recent slope, weighted
    # up the more it is already degrading (obs_rate).  This lets a fast early decliner keep declining
    # — fixing the pooled model's regress-to-the-fleet-mean under-prediction on steep degraders —
    # while a flat vehicle (obs_rate≈0) stays on the gentle pooled path.  Validated to lower RMSE on
    # BOTH degrading and flat cohorts vs the pooled path alone.
    own = s0 - obs_rate * dages
    w_own = float(np.clip(obs_rate / OWN_SLOPE_THR, 0.0, OWN_SLOPE_WMAX))
    p50 = (1.0 - w_own) * p50_pool + w_own * own
    if flat:
        p50 = np.minimum(s0 - obs_rate * dages, s0)

    band = models.get("band", BAND)
    sq = np.sqrt(dages)
    width = 0.5 if flat else 1.0                          # tighter band on flat vehicles
    p10 = p50 - band["lo"] * sq * width                  # pessimistic (lower SoH)
    p90 = p50 + band["hi"] * sq * width                  # optimistic, capped at history SoH
    p90 = np.minimum(p90, s0)
    p10 = np.minimum.accumulate(np.clip(p10, 0, 100))    # band lower edge is monotone non-increasing
    out = {0.1: p10, 0.5: np.clip(p50, 0, 100), 0.9: np.clip(p90, 0, 100)}
    return out


def _history_drop(g):
    g = g.sort_values("month")
    return float(g["soh"].iloc[0] - g["soh"].iloc[-1])
