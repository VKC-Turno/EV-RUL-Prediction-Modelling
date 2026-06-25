"""Train + persist the per-OEM models to S3 (run periodically, e.g. monthly, or after a fleet refresh).

The prediction job loads these bundles. Models are trained on the quality-gated feature tables (data-thin
vehicles excluded), versioned by date, and registered. Bundle shape per OEM matches what prediction_job
expects:
  euler            -> {"traj_model": <euler_model.train_traj output>}
  mahindra / bajaj -> {"models":     <(.)_model.train_quantiles output>}

Run as a Glue Python Shell job (or locally with AWS creds):
  --FEATURE_ROOT_S3   s3://.../features         (expects oem=<oem>/ subfolders)
  --MODEL_ROOT_S3     s3://.../models           (writes <oem>/<oem>_<YYYYMMDD>.pkl + latest.pkl + registry.json)
  --OEMS              euler,mahindra,bajaj       (optional; default all three)
  --DATE              YYYYMMDD                    (required — Glue has no wall clock by default; pass it in)

Packaging: --extra-py-files s3://.../src.zip
Deps:      --additional-python-modules pandas,pyarrow,xgboost,lightgbm,scikit-learn,s3fs
"""
import sys
import json
import pickle
import importlib
from awsglue.utils import getResolvedOptions
import pandas as pd
import s3fs

import data_quality

args = getResolvedOptions(sys.argv, ["FEATURE_ROOT_S3", "MODEL_ROOT_S3", "DATE"])
OEMS = ["euler", "mahindra", "bajaj"]
if "--OEMS" in sys.argv:
    OEMS = [o.strip().lower() for o in getResolvedOptions(sys.argv, ["OEMS"])["OEMS"].split(",")]
MODULE = {"euler": "euler_model", "mahindra": "model", "bajaj": "bajaj_model"}
fs = s3fs.S3FileSystem()


def train_bundle(oem, m):
    mod = importlib.import_module(MODULE[oem])
    if oem == "euler":
        return {"traj_model": mod.train_traj(mod.build_traj_samples(m))}
    return {"models": mod.train_quantiles(mod.build_transitions(m))}


for oem in OEMS:
    feat = f"{args['FEATURE_ROOT_S3'].rstrip('/')}/oem={oem}/"
    m = data_quality.apply_quality(pd.read_parquet(feat), oem)
    m["month"] = pd.to_datetime(m["month"])
    g = m.groupby("vin")
    n_veh = int(m["vin"].nunique())
    n_deg = int((g["soh"].first() - g["soh"].last() >= 2).sum())

    bundle = train_bundle(oem, m)
    bundle.update(oem=oem, version=f"{oem}_{args['DATE']}", n_vehicles=n_veh, n_degraders=n_deg)

    root = f"{args['MODEL_ROOT_S3'].rstrip('/')}/{oem}"
    for key in (f"{root}/{oem}_{args['DATE']}.pkl", f"{root}/latest.pkl"):
        with fs.open(key, "wb") as f:
            pickle.dump(bundle, f)

    # append a registry row
    reg_key = f"{root}/registry.json"
    registry = json.load(fs.open(reg_key)) if fs.exists(reg_key) else []
    registry.append({"version": bundle["version"], "date": args["DATE"],
                     "n_vehicles": n_veh, "n_degraders": n_deg})
    with fs.open(reg_key, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[persist] {oem}: trained on {n_veh} vehicles ({n_deg} degraders) -> {root}/latest.pkl")
