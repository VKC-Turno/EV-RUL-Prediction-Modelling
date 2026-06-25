"""Glue Python Shell prediction job — ONE OEM.

Loads the feature_table + a persisted model from S3, forecasts every (quality-gated) vehicle forward
from its LATEST data to its warranty deadline, and writes one prediction row per vehicle (current SoH,
SoH forecast quantiles, RUL in months, at-risk-by-warranty) to S3. Feature tables are small, so a single
Python-Shell node is plenty — no Spark needed.

Job parameters:
  --OEM          euler | mahindra | bajaj
  --FEATURE_S3   s3://.../features/oem=<oem>/      (output of feature_extraction_job)
  --MODEL_S3     s3://.../models/<oem>/latest.pkl  (bundle written by persist_models.py)
  --OUT_S3       s3://.../predictions/oem=<oem>/

Packaging: --extra-py-files s3://.../src.zip
Deps:      --additional-python-modules pandas,pyarrow,xgboost,lightgbm,scikit-learn,s3fs
"""
import sys
import json
import pickle
import importlib
from awsglue.utils import getResolvedOptions
import numpy as np
import pandas as pd
import s3fs

import data_quality

args = getResolvedOptions(sys.argv, ["OEM", "FEATURE_S3", "MODEL_S3", "OUT_S3"])
OEM = args["OEM"].lower()
MODULE = {"euler": "euler_model", "mahindra": "model", "bajaj": "bajaj_model"}[OEM]
EOL = {"euler": 80, "mahindra": 80, "bajaj": 70}[OEM]
WARR_YR = {"euler": 5, "mahindra": 3, "bajaj": 5}[OEM]
H_MAX = 120
fs = s3fs.S3FileSystem()

mod = importlib.import_module(MODULE)
with fs.open(args["MODEL_S3"], "rb") as f:
    bundle = pickle.load(f)                       # {"traj_model": ...} (euler) | {"models": {...}} (mah/baj)

m = data_quality.apply_quality(pd.read_parquet(args["FEATURE_S3"]), OEM)   # never predict on data-thin
m["month"] = pd.to_datetime(m["month"])


def forecast(g):
    if OEM == "euler":
        fc = mod.forecast(g, bundle["traj_model"], H_MAX)
        return np.asarray(fc[0.1]), np.asarray(fc[0.5]), np.asarray(fc[0.9])
    sim = mod.simulate(g, bundle["models"], H_MAX)
    return sim["q10"].to_numpy(), sim["q50"].to_numpy(), sim["q90"].to_numpy()


rows = []
for vin, g in m.groupby("vin"):
    g = g.sort_values("month").reset_index(drop=True)
    if len(g) < 4:
        continue
    p10, p50, p90 = forecast(g)
    cur_age = float(g["age_months"].iloc[-1]); cur_soh = float(g["soh"].iloc[-1])
    warr_age = WARR_YR * 12
    hit = np.where(p50 <= EOL)[0]
    rul_months = int(hit[0] + 1) if len(hit) else None        # months from now until P50 crosses EoL
    months_to_warr = max(int(round(warr_age - cur_age)), 1)
    at_risk_p50 = bool((p50[:months_to_warr] <= EOL).any())   # expected to cross EoL by warranty
    at_risk_p10 = bool((p10[:months_to_warr] <= EOL).any())   # worst-case
    rows.append(dict(
        oem=OEM, vin=vin, asof_age_months=round(cur_age, 1), current_soh=round(cur_soh, 1),
        eol_pct=EOL, rul_months=rul_months, warranty_age_months=warr_age,
        at_risk_by_warranty=at_risk_p50, at_risk_worstcase=at_risk_p10,
        forecast_p50=json.dumps([round(float(x), 2) for x in p50]),
        forecast_p10=json.dumps([round(float(x), 2) for x in p10]),
        forecast_p90=json.dumps([round(float(x), 2) for x in p90])))

out = pd.DataFrame(rows)
dest = args["OUT_S3"].rstrip("/") + f"/predictions_{OEM}.parquet"
with fs.open(dest, "wb") as f:
    out.to_parquet(f, index=False)
print(f"[prediction] OEM={OEM}: {len(out)} vehicles "
      f"({int(out['at_risk_by_warranty'].sum())} at-risk by warranty) -> {dest}")
