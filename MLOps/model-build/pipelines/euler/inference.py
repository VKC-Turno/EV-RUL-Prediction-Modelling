"""SageMaker inference handlers for the Euler trajectory model."""
import json
import os
import pickle

import numpy as np
import pandas as pd

import euler_model


def model_fn(model_dir):
    """Load the model bundle produced by ``pipelines.euler.train``."""
    with open(os.path.join(model_dir, "model.pkl"), "rb") as model_file:
        return pickle.load(model_file)


def input_fn(request_body, content_type):
    """Decode a JSON request containing one or more vehicle histories."""
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body) if isinstance(request_body, str) else request_body
    if not isinstance(payload, (dict, list)):
        raise ValueError("request must be a JSON object or array")
    return payload


def _instances(payload):
    if isinstance(payload, list):
        return payload
    instances = payload.get("instances", [payload])
    if not isinstance(instances, list) or not instances:
        raise ValueError("instances must be a non-empty array")
    return instances


def _history(instance):
    if not isinstance(instance, dict):
        raise ValueError("each instance must be a JSON object")
    rows = instance.get("history")
    if rows is None:
        rows = [{key: value for key, value in instance.items() if key not in {"horizon", "history"}}]
    if not isinstance(rows, list) or not rows:
        raise ValueError("history must be a non-empty array")
    frame = pd.DataFrame(rows)
    missing = {"soh", "age_months"} - set(frame.columns)
    if missing:
        raise ValueError(f"history is missing required fields: {sorted(missing)}")
    frame["soh"] = pd.to_numeric(frame["soh"], errors="raise")
    frame["age_months"] = pd.to_numeric(frame["age_months"], errors="raise")
    if "month" not in frame:
        if "ymd" in frame:
            frame["month"] = pd.to_datetime(frame["ymd"], errors="raise")
        else:
            origin = pd.Timestamp("2000-01-01")
            frame["month"] = [origin + pd.DateOffset(months=int(age)) for age in frame["age_months"]]
    else:
        frame["month"] = pd.to_datetime(frame["month"], errors="raise")
    for feature in euler_model.TRAJ_STRESS:
        if feature not in frame:
            frame[feature] = np.nan
        else:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame.sort_values("month").reset_index(drop=True)


def _prediction(instance, model_bundle):
    horizon = int(instance.get("horizon", 1))
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    trajectory = euler_model.forecast(_history(instance), model_bundle["traj_model"], horizon)
    rows = []
    for index in range(horizon):
        rows.append({
            "month": index + 1,
            "q10": float(trajectory[0.1][index]),
            "q50": float(trajectory[0.5][index]),
            "q90": float(trajectory[0.9][index]),
        })
    return {"horizon": horizon, **{key: rows[-1][key] for key in ("q10", "q50", "q90")},
            "trajectory": rows}


def predict_fn(payload, model_bundle):
    """Forecast each instance and return final-horizon values plus full trajectories."""
    if not isinstance(model_bundle, dict) or "traj_model" not in model_bundle:
        raise ValueError("model artifact does not contain traj_model")
    predictions = [_prediction(instance, model_bundle) for instance in _instances(payload)]
    response = {"predictions": predictions}
    if len(predictions) == 1:
        response.update({key: predictions[0][key] for key in ("q10", "q50", "q90")})
    return response


def output_fn(prediction, accept):
    """Serialize predictions as JSON."""
    if accept and accept.split(";", 1)[0].strip().lower() not in {"application/json", "*/*"}:
        raise ValueError(f"unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
