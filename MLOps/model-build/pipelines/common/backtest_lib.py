"""Shared evaluation driver for the backtest Processing step.

By-vehicle held-out backtest: for each held-out test vehicle, forecast its last ~40% from its first ~60%
and compare q50 to the actual SoH; report overall + degrading-cohort RMSE and a persistence baseline.
Mirrors src/oem_train.py::backtest and the Euler LOVO in src/euler_train.py.

Writes /opt/ml/processing/evaluation/evaluation.json — the PropertyFile the ConditionStep + RegisterModel
consume. The degrading-vs-persistence numbers are what decide whether a fleet's model is worth deploying.
"""
import json
import pathlib
import numpy as np
import pandas as pd

DEGRADE_PP = 2.0
HOLDOUT = 0.40
MIN_HIST = 4


def _rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else None


def evaluate(model, m: pd.DataFrame, test_vins) -> dict:
    """`model` exposes simulate(hist, H)->DataFrame[q50]; `m` is the featengg frame."""
    errs, perr = [], []
    for vin in test_vins:
        g = m[m["vin"] == vin].sort_values("ymd").reset_index(drop=True)
        n = len(g)
        if n < MIN_HIST + 2:
            continue
        k = max(1, min(int(round(n * HOLDOUT)), n - MIN_HIST))
        hist, fut = g.iloc[:n - k], g.iloc[n - k:]
        try:
            q50 = np.asarray(model.simulate(hist, len(fut))["q50"], float)
        except Exception:
            continue
        actual = fut["soh"].to_numpy(float)
        last = float(hist["soh"].iloc[-1])
        degrader = (last - actual[-1]) >= DEGRADE_PP
        for j in range(len(fut)):
            errs.append((q50[j] - actual[j], degrader))
            perr.append(last - actual[j])
    if not errs:
        return {"n_forecasts": 0}
    e = np.array([x[0] for x in errs]); dmask = np.array([x[1] for x in errs])
    return {
        "overall_rmse": round(_rmse(e), 3),
        "degrading_rmse": round(_rmse(e[dmask]), 3) if dmask.any() else None,
        "persistence_rmse": round(_rmse(perr), 3),
        "n_test_vehicles": int(len(set(test_vins))),
        "n_forecasts": int(len(e)),
    }


def write_report(metrics: dict, output_dir="/opt/ml/processing/evaluation") -> str:
    out = pathlib.Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    report = {
        "metric": "held-out per-vehicle SoH forecast RMSE (pp)",
        "regression_metrics": {k: {"value": v} for k, v in metrics.items() if v is not None},
        **metrics,
    }
    path = out / "evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}: {metrics}")
    return str(path)


def _load_model(model_dir="/opt/ml/processing/model"):
    """SageMaker mounts the training ModelArtifacts as model.tar.gz — extract and unpickle."""
    import glob, pickle, tarfile
    for tar in glob.glob(str(pathlib.Path(model_dir) / "*.tar.gz")):
        with tarfile.open(tar) as t:
            t.extractall(model_dir)
    with open(pathlib.Path(model_dir) / "model.pkl", "rb") as f:
        return pickle.load(f)["model"]


def run(oem: str, model_dir="/opt/ml/processing/model", feat_dir="/opt/ml/processing/featengg"):
    """Evaluate-step entry: load model + featengg, held-out backtest the last 20% of vehicles, write report."""
    import glob
    model = _load_model(model_dir)
    files = glob.glob(str(pathlib.Path(feat_dir) / "**" / "*.parquet"), recursive=True)
    m = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    m["vin"] = m["vin"].astype(str)
    vins = sorted(m["vin"].unique())
    test_vins = vins[int(len(vins) * 0.8):]            # deterministic held-out tail
    metrics = evaluate(model, m, test_vins)
    return write_report(metrics)
