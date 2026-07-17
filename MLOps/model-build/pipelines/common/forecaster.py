"""Reference quantile SoH-trajectory forecaster (the pipeline-facing model interface).

Contract every OEM forecaster satisfies:
    fit(featengg_frame)  -> model
    model.simulate(hist, H) -> DataFrame[q10, q50, q90]   # H months ahead of hist's last row

This module is a compact, working baseline (monotone trend + empirical residual band) so the pipeline
is runnable end-to-end. In production, config.model_module selects the richer families ported from the
research repo:
    euler_model   -> rate + trajectory models with recalibrated P10/P90 band
    model         -> build_transitions -> train_quantiles -> simulate (Mahindra / Piaggio / Montra)
    bajaj_model   -> same API, conditioned on cycles/temp/efficiency (no current/voltage)
Port those into pipelines/common/forecasters/ and switch on cfg.model_module in fit(); the interface
below is what train.py / backtest_lib.py depend on, so keep it stable.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class TrajectoryModel:
    slope_ppm: float          # mean SoH loss per month (pp), >= 0
    resid_q: dict             # {0.1: .., 0.5: .., 0.9: ..} residual quantiles

    def simulate(self, hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
        last = float(pd.to_numeric(hist["soh"], errors="coerce").iloc[-1])
        steps = np.arange(1, horizon + 1)
        centre = last - self.slope_ppm * steps
        return pd.DataFrame({
            "h": steps,
            "q10": np.clip(centre + self.resid_q[0.1], 0, 100),
            "q50": np.clip(centre + self.resid_q[0.5], 0, 100),
            "q90": np.clip(centre + self.resid_q[0.9], 0, 100),
        })


def fit(m: pd.DataFrame, model_module: str = "model") -> TrajectoryModel:
    """Pooled per-vehicle degradation rate + residual quantiles. `m` is the featengg frame."""
    m = m.copy()
    m["soh"] = pd.to_numeric(m["soh"], errors="coerce")
    m["age_months"] = pd.to_numeric(m["age_months"], errors="coerce")
    rates = []
    for _, g in m.groupby("vin"):
        g = g.dropna(subset=["soh", "age_months"]).sort_values("age_months")
        if len(g) >= 3 and g["age_months"].iloc[-1] > g["age_months"].iloc[0]:
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            fit_soh = iso.fit_transform(g["age_months"], g["soh"])
            span = g["age_months"].iloc[-1] - g["age_months"].iloc[0]
            rates.append(max(0.0, (fit_soh[0] - fit_soh[-1]) / span))
    slope = float(np.median(rates)) if rates else 0.0
    # residual band from one-step errors around the pooled slope
    resid = []
    for _, g in m.groupby("vin"):
        g = g.dropna(subset=["soh"]).sort_values("age_months")
        if len(g) >= 2:
            pred = g["soh"].iloc[0] - slope * (g["age_months"] - g["age_months"].iloc[0])
            resid.extend((g["soh"] - pred).tolist())
    resid = np.asarray(resid) if resid else np.array([0.0])
    q = {p: float(np.quantile(resid, p)) for p in (0.1, 0.5, 0.9)}
    return TrajectoryModel(slope_ppm=slope, resid_q=q)
