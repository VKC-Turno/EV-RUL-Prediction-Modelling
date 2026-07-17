"""SageMaker Training entry point (script mode) — shared by every OEM.

Reads the featengg table from the training channel, trains the OEM's forecaster (family selected by the
registry), and writes a pickled bundle {model, meta} to SM_MODEL_DIR (-> model.tar.gz). Mirrors
src/oem_train.py / src/euler_train.py, minus the local registry.json (the SageMaker Model Registry owns
versioning now).

    python -m pipelines.common.train --oem euler
"""
import argparse
import glob
import json
import os
import pickle
import pandas as pd

from . import config, forecaster


def _load(channel):
    files = glob.glob(os.path.join(channel, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no featengg parquet in {channel}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oem", required=True)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    a = p.parse_args()

    cfg = config.get(a.oem)
    m = _load(a.train)
    m["vin"] = m["vin"].astype(str)
    n_veh = m["vin"].nunique()
    drop = m.groupby("vin")["soh"].first() - m.groupby("vin")["soh"].last()
    n_deg = int((drop >= 2.0).sum())

    model = forecaster.fit(m, model_module=cfg.model_module)
    meta = dict(oem=a.oem, soh_method=cfg.soh_method, model_module=cfg.model_module,
                eol_pct=cfg.eol_pct, warr_years=cfg.warr_years, warr_km=cfg.warr_km,
                has_gate=cfg.has_gate, maturity=cfg.maturity,
                n_vehicles=int(n_veh), n_degraders=n_deg, n_vin_months=int(len(m)))

    os.makedirs(a.model_dir, exist_ok=True)
    with open(os.path.join(a.model_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)
    with open(os.path.join(a.model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{a.oem}] trained {cfg.model_module} on {n_veh} vehicles ({n_deg} degraders) -> {a.model_dir}")


if __name__ == "__main__":
    main()
