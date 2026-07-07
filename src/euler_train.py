#!/usr/bin/env python3
"""Train, validate, and PERSIST the Euler SoH forecaster — versioned model registry + diagnostics.

Single training pipeline that emits everything downstream needs:
  • trains the RATE model (the dashboard's forecaster) and the TRAJECTORY model (P50 + P10/P90);
  • LOVO backtest -> honest generalization RMSE + recalibrated P10/P90 band;
  • a by-vehicle 60/20/20 train/validation/test split -> the three split errors + feature importance;
  • saves a versioned, loadable bundle + a human-readable registry + a diagnostics file.

Artifacts (models/euler/):
  euler_<YYYYMMDD>.pkl   pickled bundle {rate_model, traj_model, band, meta}   (gitignored — binary)
  latest.pkl            -> copy of the newest bundle; what the dashboard loads
  registry.json          one entry per training run (date, n_veh, n_deg, LOVO RMSE) — committed
  diagnostics.json       feature importance + train/val/test errors for the diagnostics dashboard

Designed for a QUARTERLY retrain (scripts/retrain_euler.sh): as more vehicles age toward 80%,
re-import -> rebuild features -> retrain here -> recalibrate band -> new version. The registry lets
you see whether each retrain actually improves accuracy.

Run:  .venv/bin/python src/euler_train.py [--fast]   (--fast skips the slow LOVO backtest)
"""
import os, sys, json, pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))      # so `import euler_model` works standalone
import euler_model as em

FT = "data/euler/features/feature_table.parquet"
BMS_SOH = "data/euler/bms_soh.parquet"        # recovery-aware clean soh_label (src/euler_bms_soh.py)
MDIR = Path("models/euler")
LATEST = MDIR / "latest.pkl"
REGISTRY = MDIR / "registry.json"
DIAG = MDIR / "diagnostics.json"


def apply_label(m, path=BMS_SOH):
    """Swap the training target `soh` for the recovery-aware clean `soh_label` (monotone, <=100, artifact-free;
    see src/euler_bms_soh.py). Joins on (vin, month), KEEPS only months that carry a finite label (unanchored /
    sparse-near-full months have none), and overwrites `soh`. Returns (m_labeled, coverage_dict). The rest of the
    pipeline is untouched — same features, same model — so this is a pure target swap."""
    lab = pd.read_parquet(path)[["vin", "month", "soh_label"]].copy()
    lab["vin"] = lab["vin"].astype(str); lab["month"] = pd.to_datetime(lab["month"])
    m = m.copy(); m["vin"] = m["vin"].astype(str); m["month"] = pd.to_datetime(m["month"])
    before_rows, before_veh = len(m), m["vin"].nunique()
    m = m.merge(lab, on=["vin", "month"], how="left")
    m = m[m["soh_label"].notna()].copy()
    m["soh"] = m["soh_label"].to_numpy()
    m = m.drop(columns=["soh_label"])
    # a vehicle needs enough labeled months to build transitions/trajectories
    keep = m.groupby("vin")["month"].transform("size") >= 3
    m = m[keep].copy()
    cov = dict(target="soh_label", rows=len(m), rows_before=before_rows,
               vehicles=int(m["vin"].nunique()), vehicles_before=int(before_veh))
    return m, cov


# ───────────────────────── train / validation / test split diagnostics ─────────────────────────
def _split(vins, drop, seed=0):
    """By-vehicle 60/20/20 split, stratified so each split keeps degraders and flat vehicles."""
    rng = np.random.RandomState(seed)
    deg = sorted(v for v in vins if drop[v] >= 2)
    flat = sorted(v for v in vins if drop[v] < 2)
    out = ([set(), set(), set()])
    for grp in (deg, flat):
        grp = list(grp); rng.shuffle(grp); n = len(grp)
        ntr, nva = int(n * 0.6), int(n * 0.2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out  # (train, val, test) vin-sets


def diagnostics(m, seed=0):
    """Train on the TRAIN split only; report per-transition monthly-loss RMSE on each split +
    feature importance (the model's native objective is monthly SoH loss, so that's the error)."""
    g = m.groupby("vin")
    drop = (g["soh"].first() - g["soh"].last())
    TR, VA, TE = _split(list(m["vin"].unique()), drop, seed)
    rate = em.train(em.build_transitions(m[m["vin"].isin(TR)]))
    bias = float(getattr(rate, "_cal_bias", 0.0))

    def rmse(vinset):
        t = em.build_transitions(m[m["vin"].isin(vinset)])
        if not len(t):
            return None
        pred = np.clip(rate.predict(t[em.FEATS].to_numpy()) + bias, 0.0, None)
        return round(float(np.sqrt(np.mean((t["loss"].to_numpy() - pred) ** 2))), 4)

    fi_rate = sorted(zip(em.FEATS, [float(x) for x in rate.feature_importances_]),
                     key=lambda x: -x[1])
    # trajectory-model importance (LightGBM P50), trained on the full data for the importance read
    traj = em.train_traj(em.build_traj_samples(m))
    try:
        fi_traj = sorted(zip(em.TRAJ_FEATS, [float(x) for x in traj["p50"].feature_importances_]),
                         key=lambda x: -x[1])
    except Exception:
        fi_traj = []
    return {
        "metric": "per-transition monthly SoH-loss RMSE (pp/month)",
        "split_sizes": {"train": len(TR), "validation": len(VA), "test": len(TE)},
        "errors": {"train": rmse(TR), "validation": rmse(VA), "test": rmse(TE)},
        "feature_importance_rate": fi_rate,
        "feature_importance_trajectory": fi_traj,
    }


# ───────────────────────── full train + persist ─────────────────────────
def main(fast=False, label=False):
    os.chdir(Path(__file__).resolve().parent.parent)
    m = pd.read_parquet(FT)
    m["month"] = pd.to_datetime(m["month"])
    m = m.sort_values(["vin", "month"])
    import data_quality                                 # manifest-driven gate: never train on data-thin vehicles
    before = m["vin"].nunique(); m = data_quality.apply_quality(m, "Euler")
    if m["vin"].nunique() < before:
        print(f"  data-quality gate: dropped {before - m['vin'].nunique()} thin vehicle(s)")
    target = "soh_production"
    if label:
        m, cov = apply_label(m); target = "soh_label"
        m = m.sort_values(["vin", "month"])
        print(f"  target = soh_label (recovery-aware clean): {cov['vehicles']}/{cov['vehicles_before']} vehicles, "
              f"{cov['rows']}/{cov['rows_before']} vin-months carry a label")
    g = m.groupby("vin")
    n_veh = int(m["vin"].nunique())
    n_deg = int((g["soh"].first() - g["soh"].last() >= 2).sum())
    print(f"training Euler forecaster on {n_veh} vehicles ({n_deg} degraders, {len(m)} vin-months)…")

    rate = em.train(em.build_transitions(m))
    traj = em.train_traj(em.build_traj_samples(m))

    lovo = {}
    if not fast:
        print("  LOVO backtest (recalibrating P10/P90 band)…")
        import euler_backtest as eb
        per = eb.run_backtest(m)
        per_cal, band = eb.recalibrate(per, m)
        traj["band"] = band
        summ = eb.summarize(per_cal)
        calib = eb.calibration(per_cal)

        def rmse_of(c):
            r = summ[summ["cohort"] == c]
            return float(r["trajectory_RMSE"].iloc[0]) if len(r) else None
        lovo = {"overall_rmse": rmse_of("overall"), "degrading_rmse": rmse_of("degrading"),
                "flat_rmse": rmse_of("flat"), "band_coverage": round(float(calib.get("overall", np.nan)), 3),
                "band": band}

    print("  diagnostics (train/val/test split + feature importance)…")
    diag = diagnostics(m)

    stamp = datetime.now().strftime("%Y%m%d")
    version = f"euler_{stamp}"
    meta = {"version": version, "trained_at": datetime.now().isoformat(timespec="seconds"),
            "target": target, "n_vehicles": n_veh, "n_degraders": n_deg, "n_vin_months": len(m),
            "feature_table": FT, **lovo}
    bundle = {"rate_model": rate, "traj_model": traj, "band": traj.get("band"), "meta": meta}

    MDIR.mkdir(parents=True, exist_ok=True)
    with open(MDIR / f"{version}.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(LATEST, "wb") as f:
        pickle.dump(bundle, f)

    reg = json.load(open(REGISTRY)) if REGISTRY.exists() else []
    reg.append({k: meta.get(k) for k in ["version", "trained_at", "target", "n_vehicles", "n_degraders",
                                          "overall_rmse", "degrading_rmse", "flat_rmse", "band_coverage"]})
    json.dump(reg, open(REGISTRY, "w"), indent=2)
    diag["version"] = version
    diag["trained_at"] = meta["trained_at"]
    json.dump(diag, open(DIAG, "w"), indent=2)

    print(f"\nsaved models/euler/{version}.pkl (+ latest.pkl)")
    print(f"  LOVO RMSE: overall {meta.get('overall_rmse')} | degrading {meta.get('degrading_rmse')} | "
          f"band {meta.get('band_coverage')}")
    print(f"  split RMSE (pp/mo): {diag['errors']}  | registry now {len(reg)} run(s)")


def load_latest():
    """Return the persisted model bundle, or None if no saved model (callers fall back to on-demand)."""
    p = Path("models/euler/latest.pkl")
    if p.exists():
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            return None
    return None


if __name__ == "__main__":
    main(fast="--fast" in sys.argv, label="--label" in sys.argv)
