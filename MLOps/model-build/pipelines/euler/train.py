"""SageMaker Training entry point — Euler SoH forecaster, OUR euler_model, end to end.

Reads the `euler_featengg` feature store (produced by the Glue job), does the POINT-IN-TIME cohort selection
(keep labelled rows + the data-quality gate), trains our RATE + TRAJECTORY quantile forecaster
(`euler_model`), runs the LOVO backtest to recalibrate the P10/P90 band (`euler_backtest`), computes the
stratified train/val/test diagnostics (`euler_train`), and writes the model bundle + `evaluation.json`.

All logic is ours — this only wires our functions into the SageMaker channel/model-dir contract. The src
modules (euler_model / euler_backtest / euler_train / data_quality) are shipped via the estimator's
`dependencies=[...]`; run locally with `PYTHONPATH=src`.

    python train.py --oem euler [--fast] [--label]
Channels: SM_CHANNEL_TRAIN = euler_featengg parquet; optional SM_CHANNEL_LABEL = bms_soh parquet (soh_target).
"""
import argparse
import glob
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import euler_model as em
import euler_backtest as eb
import euler_train as et
import data_quality


def _read_parquet_dir(channel):
    files = glob.glob(os.path.join(channel, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no parquet under {channel}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def _load_featengg(channel):
    m = _read_parquet_dir(channel)
    if "month" not in m.columns and "ymd" in m.columns:      # Glue store keys on ymd; euler_model wants month
        m["month"] = pd.to_datetime(m["ymd"])
    m["month"] = pd.to_datetime(m["month"])
    m["vin"] = m["vin"].astype(str)
    return m.sort_values(["vin", "month"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oem", default="euler")
    p.add_argument("--fast", action="store_true", help="skip the slow LOVO backtest")
    p.add_argument("--label", action="store_true", help="train on the hybrid soh_target (needs LABEL channel)")
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--label-channel", default=os.environ.get("SM_CHANNEL_LABEL", ""))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    a = p.parse_args()

    m = _load_featengg(a.train)

    # POINT-IN-TIME cohort selection (this is the SELECTION the feature store deliberately does NOT bake in):
    #   * featengg is all-vehicles / all-months with soh nullable -> training needs a label, keep soh-present rows;
    #   * drop data-thin vehicles via the manifest gate (no-op if the manifest isn't shipped);
    #   * (in-service filtering, if any, would also go here — from a fleet-status input, at run time).
    m = m[m["soh"].notna()].copy()
    before = m["vin"].nunique()
    m = data_quality.apply_quality(m, a.oem.capitalize())
    if m["vin"].nunique() < before:
        print(f"  data-quality gate dropped {before - m['vin'].nunique()} thin vehicle(s)")

    target = "soh_production"
    if a.label and a.label_channel:
        lab_files = glob.glob(os.path.join(a.label_channel, "**", "*.parquet"), recursive=True)
        m, cov = et.apply_label(m, path=lab_files[0])          # hybrid: clean label on flats, prod on decliners
        target = "soh_hybrid"
        print(f"  target=soh_hybrid: {cov['vehicles']}/{cov['vehicles_before']} vehicles, "
              f"{cov['rows']}/{cov['rows_before']} vin-months")
    m = m.sort_values(["vin", "month"]).reset_index(drop=True)

    g = m.groupby("vin")
    n_veh = int(m["vin"].nunique())
    n_deg = int((g["soh"].first() - g["soh"].last() >= 2).sum())
    print(f"training euler on {n_veh} vehicles ({n_deg} degraders, {len(m)} vin-months, target={target})…")

    rate = em.train(em.build_transitions(m))                  # rate model (our XGBoost monthly-loss)
    traj = em.train_traj(em.build_traj_samples(m))            # trajectory model (our LightGBM P10/P50/P90)

    lovo = {}
    if not a.fast:
        print("  LOVO backtest (recalibrating P10/P90 band)…")
        per = eb.run_backtest(m)
        per_cal, band = eb.recalibrate(per, m)
        traj["band"] = band
        summ, calib = eb.summarize(per_cal), eb.calibration(per_cal)

        def rmse_of(c):
            r = summ[summ["cohort"] == c]
            return float(r["trajectory_RMSE"].iloc[0]) if len(r) else None
        lovo = dict(overall_rmse=rmse_of("overall"), degrading_rmse=rmse_of("degrading"),
                    flat_rmse=rmse_of("flat"), band_coverage=round(float(calib.get("overall", np.nan)), 3))

    print("  diagnostics (train/val/test split + feature importance)…")
    diag = et.diagnostics(m)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    version = f"euler_{stamp}"
    meta = dict(version=version, trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                oem=a.oem, target=target, n_vehicles=n_veh, n_degraders=n_deg, n_vin_months=int(len(m)), **lovo)

    os.makedirs(a.model_dir, exist_ok=True)
    with open(os.path.join(a.model_dir, "model.pkl"), "wb") as f:
        pickle.dump({"rate_model": rate, "traj_model": traj, "band": traj.get("band"), "meta": meta}, f)
    report = dict(metric="LOVO trajectory RMSE (pp)", **meta, diagnostics=diag,
                  regression_metrics={k: {"value": meta[k]} for k in ("overall_rmse", "degrading_rmse",
                                                                       "flat_rmse") if meta.get(k) is not None})
    with open(os.path.join(a.model_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"saved model + evaluation.json | LOVO overall={meta.get('overall_rmse')} "
          f"degrading={meta.get('degrading_rmse')} band={meta.get('band_coverage')}")


if __name__ == "__main__":
    main()
