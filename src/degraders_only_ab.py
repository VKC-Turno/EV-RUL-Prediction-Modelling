#!/usr/bin/env python3
"""A/B: does IGNORING the flat / near-new vehicles (train on degraders only) help?

For each OEM we hold a FIXED test set, then train two models on the SAME train split:
  - ALL    : every training vehicle (degraders + flat)
  - DEG-ONLY: only the degrading training vehicles (>= DEG pp total drop)
…and score monthly-loss RMSE on the test DEGRADERS and (separately) the test FLAT vehicles. This shows
whether dropping flat vehicles sharpens degrader predictions, and what it costs on the flat ones (which
still exist in production and must not be over-predicted into false decline).

A common squared-error XGBoost is used for every OEM so the only thing that varies is the TRAIN SUBSET.
Run: .venv/bin/python src/degraders_only_ab.py
"""
import os, sys, importlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")

OEMS = {"Euler": ("data/euler/features/feature_table.parquet", "euler_model"),
        "Mahindra": ("data/mahindra/features/feature_table.parquet", "model"),
        "Bajaj": ("data/bajaj/features/feature_table.parquet", "bajaj_model")}
DEG = 2.0   # total SoH drop (pp) to count as a degrader


def split(vins, drop, seed=0):
    rng = np.random.RandomState(seed); out = [set(), set(), set()]
    for grp in (sorted(v for v in vins if drop[v] >= 2), sorted(v for v in vins if drop[v] < 2)):
        grp = list(grp); rng.shuffle(grp); n = len(grp); ntr, nva = int(n * .6), int(n * .2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out


def fit(t, feats):
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=8, verbosity=0).fit(
        t[feats].to_numpy(), t["loss"].to_numpy(), sample_weight=t["w"].to_numpy())


def rmse(model, t, feats):
    if not len(t):
        return float("nan")
    p = np.clip(model.predict(t[feats].to_numpy()), 0, None)
    return float(np.sqrt(np.mean((t["loss"].to_numpy() - p) ** 2)))


for oem, (ft, modname) in OEMS.items():
    M = importlib.import_module(modname)
    m = pd.read_parquet(ft); feats = M.FEATS
    g = m.groupby("vin"); drop = (g.soh.first() - g.soh.last())
    vins = list(m.vin.unique()); TR, VA, TE = split(vins, drop)
    degset = {v for v in vins if drop[v] >= DEG}
    tr_all = M.build_transitions(m[m.vin.isin(TR)])
    tr_deg = M.build_transitions(m[m.vin.isin(TR & degset)])
    te_deg = M.build_transitions(m[m.vin.isin(TE & degset)])
    te_flat = M.build_transitions(m[m.vin.isin(TE - degset)])
    m_all, m_deg = fit(tr_all, feats), fit(tr_deg, feats)
    print(f"\n{oem}: train {len(TR)} ({len(TR & degset)} deg / {len(TR - degset)} flat) | "
          f"test {len(TE)} ({len(TE & degset)} deg / {len(TE - degset)} flat)")
    print(f"  test-DEGRADER loss-RMSE:  all-trained {rmse(m_all, te_deg, feats):.3f}   "
          f"deg-only {rmse(m_deg, te_deg, feats):.3f}")
    print(f"  test-FLAT     loss-RMSE:  all-trained {rmse(m_all, te_flat, feats):.3f}   "
          f"deg-only {rmse(m_deg, te_flat, feats):.3f}")
