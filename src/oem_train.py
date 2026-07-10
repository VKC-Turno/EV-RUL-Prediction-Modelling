#!/usr/bin/env python3
"""Train, backtest, and PERSIST a deployed SoH forecaster for the NON-Euler OEMs (Mahindra / Bajaj / Piaggio).

Euler has src/euler_train.py (rate + trajectory models). The other three share one model API
(model.py for Mahindra & Piaggio, bajaj_model.py for Bajaj): build_transitions -> train_quantiles -> simulate.
This is the equivalent persist/registry pipeline for them, so the dashboards can load_latest() a versioned
deployed model instead of retraining on every run.

Artifacts (models/<oem>/):
  <oem>_<YYYYMMDD>.pkl   pickled bundle {model, meta}          (gitignored — binary)
  latest.pkl            -> the newest bundle; what the dashboards load
  registry.json          one entry per training run (committed)
  diagnostics.json       split sizes + holdout RMSE (committed)

Run:  .venv/bin/python src/oem_train.py <mahindra|bajaj|piaggio> [--target soh]
Importable: from oem_train import load_latest; b = load_latest("mahindra")
"""
import os, sys, json, pickle, importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_quality

CFG = {
    "mahindra": dict(module="model", eol=80.0, warr_yr=3),
    "bajaj": dict(module="bajaj_model", eol=70.0, warr_yr=5),
    "piaggio": dict(module="model", eol=80.0, warr_yr=3),
    "montra": dict(module="model", eol=80.0, warr_yr=3),   # new OEM, 10-veh POC (~4mo, new fleet -> flat SoH)
}
STORE = "data/redshift/{oem}_featengg.parquet"
DEGRADE_PP = 2.0


def load_cohort(oem):
    """The deployed cohort: the Redshift feature-engineering store, gated for data-thin vehicles."""
    d = pd.read_parquet(STORE.format(oem=oem)).rename(columns={"ymd": "month"})
    d["vin"] = d["vin"].astype(str); d["month"] = pd.to_datetime(d["month"], errors="coerce")
    for c in ("soh", "age_months"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["vin", "month"]).reset_index(drop=True)
    return data_quality.apply_quality(d, oem.capitalize())


def _split(vins, drop, seed=0):
    """By-vehicle 60/20/20 split, stratified so each split keeps degraders and flat vehicles."""
    rng = np.random.RandomState(seed)
    out = [set(), set(), set()]
    for grp in (sorted(v for v in vins if drop.get(v, 0) >= DEGRADE_PP),
                sorted(v for v in vins if drop.get(v, 0) < DEGRADE_PP)):
        grp = list(grp); rng.shuffle(grp); n = len(grp); ntr, nva = int(n * 0.6), int(n * 0.2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out  # (train, val, test)


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def backtest(mod, m, test_vins, holdout=0.40, min_hist=4):
    """Train on non-test vehicles; for each held-out TEST vehicle forecast its last `holdout` from the first
    ~60% and compare the q50 to the actual SoH. Returns overall + degrading-cohort RMSE and a persistence baseline."""
    train = m[~m["vin"].isin(test_vins)]
    fmodel = mod.train_quantiles(mod.build_transitions(train))
    errs, perr, deg = [], [], []
    for vin in test_vins:
        g = m[m["vin"] == vin].sort_values("month").reset_index(drop=True)
        n = len(g)
        if n < min_hist + 2:
            continue
        k = max(1, min(int(round(n * holdout)), n - min_hist)); cut = n - k
        hist, fut = g.iloc[:cut], g.iloc[cut:]
        H = len(fut)
        try:
            q50 = mod.simulate(hist, fmodel, H)["q50"].to_numpy()
        except Exception:
            continue
        actual = fut["soh"].to_numpy(float)
        last = float(hist["soh"].iloc[-1])
        for j in range(H):
            errs.append((q50[j] - actual[j], bool((last - actual[-1]) >= DEGRADE_PP)))
            perr.append(last - actual[j])                                   # persistence baseline
    if not errs:
        return {}
    e = np.array([x[0] for x in errs]); dmask = np.array([x[1] for x in errs])
    return dict(overall_rmse=round(_rmse(e, 0), 3),
                degrading_rmse=round(_rmse(e[dmask], 0), 3) if dmask.any() else None,
                persistence_rmse=round(_rmse(perr, 0), 3),
                n_test_vehicles=int(len(set(test_vins) & set(m["vin"]))), n_forecasts=int(len(e)))


def train(oem, target="soh"):
    os.chdir(Path(__file__).resolve().parent.parent)
    oem = oem.lower(); cfg = CFG[oem]
    mod = importlib.import_module(cfg["module"])
    m = load_cohort(oem)
    if target != "soh":                                    # Part B: swap in a cleaned target column
        m = m[m[target].notna()].copy(); m["soh"] = pd.to_numeric(m[target], errors="coerce")
    g = m.groupby("vin"); drop = (g["soh"].first() - g["soh"].last())
    n_veh = int(m["vin"].nunique()); n_deg = int((drop >= DEGRADE_PP).sum())
    print(f"training {oem} forecaster on {n_veh} vehicles ({n_deg} degraders, {len(m)} vin-months, target={target})…")

    model = mod.train_quantiles(mod.build_transitions(m))  # DEPLOYED model = trained on ALL gated vehicles
    TR, VA, TE = _split(list(m["vin"].unique()), drop.to_dict())
    print(f"  backtest on {len(TE)} held-out test vehicles…")
    bt = backtest(mod, m, TE)

    stamp = datetime.now().strftime("%Y%m%d"); version = f"{oem}_{stamp}"
    meta = dict(version=version, trained_at=datetime.now().isoformat(timespec="seconds"), oem=oem,
                target=target, module=cfg["module"], eol=cfg["eol"], warr_yr=cfg["warr_yr"],
                n_vehicles=n_veh, n_degraders=n_deg, n_vin_months=int(len(m)), **bt)
    bundle = {"model": model, "meta": meta}

    mdir = Path("models") / oem; mdir.mkdir(parents=True, exist_ok=True)
    pickle.dump(bundle, open(mdir / f"{version}.pkl", "wb"))
    pickle.dump(bundle, open(mdir / "latest.pkl", "wb"))
    reg_p = mdir / "registry.json"
    reg = json.load(open(reg_p)) if reg_p.exists() else []
    reg.append({k: meta.get(k) for k in ("version", "trained_at", "target", "n_vehicles", "n_degraders",
                                         "overall_rmse", "degrading_rmse", "persistence_rmse")})
    json.dump(reg, open(reg_p, "w"), indent=2)
    json.dump(dict(version=version, trained_at=meta["trained_at"],
                   split_sizes=dict(train=len(TR), validation=len(VA), test=len(TE)), **bt),
              open(mdir / "diagnostics.json", "w"), indent=2)
    print(f"saved models/{oem}/{version}.pkl (+ latest.pkl) | overall RMSE {bt.get('overall_rmse')} "
          f"| degrading {bt.get('degrading_rmse')} vs persist {bt.get('persistence_rmse')} | registry {len(reg)} run(s)")
    return bundle


def load_latest(oem):
    """Return the persisted bundle {model, meta} for `oem`, or None (callers fall back to on-demand training)."""
    p = Path("models") / oem.lower() / "latest.pkl"
    if p.exists():
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            return None
    return None


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tgt = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--target"), "soh")
    oems = args or ["mahindra", "bajaj", "piaggio"]
    for o in oems:
        train(o, target=tgt)
