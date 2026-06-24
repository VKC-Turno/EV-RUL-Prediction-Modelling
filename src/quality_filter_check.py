#!/usr/bin/env python3
"""Does excluding DATA-THIN vehicles (too few valid months / too short a span to trust their trend) from
training beat (a) keeping everything and (b) dropping all flat vehicles?

A "thin" vehicle has < MIN_MONTHS valid SoH months OR < MIN_SPAN months of age span — its flat-or-not
trend is unprovable. We filter thin vehicles (BOTH flat and degraders) from the TRAINING set only, keep the
SAME held-out test degraders, and backtest: forecast each test degrader's last ~40% from its first ~60%,
reporting signed error (pred-actual; >0 = over-predicts life) and MAE. Three training sets compared:
  ALL       - every training vehicle (current production behaviour)
  QUALITY   - thin vehicles removed (both classes)
  DEG-only  - only degraders (the blunt version, for reference)

Run: .venv/bin/python src/quality_filter_check.py
"""
import os, sys, importlib, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")

SPEC = {"Euler": ("data/euler/features/feature_table.parquet", "euler_model"),
        "Mahindra": ("data/mahindra/features/feature_table.parquet", "model"),
        "Bajaj": ("data/bajaj/features/feature_table.parquet", "bajaj_model")}
MIN_MONTHS, MIN_SPAN = 6, 9.0


def split(vins, drop, seed=0):
    rng = np.random.RandomState(seed); out = [set(), set(), set()]
    for grp in (sorted(v for v in vins if drop[v] >= 2), sorted(v for v in vins if drop[v] < 2)):
        grp = list(grp); rng.shuffle(grp); n = len(grp); ntr, nva = int(n * .6), int(n * .2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out


print(f"{'OEM':9} {'train set':10} {'#train':>6} {'n_test':>6} {'signed err':>11} {'MAE':>7}")
for oem, (ft, modname) in SPEC.items():
    M = importlib.import_module(modname); m = pd.read_parquet(ft); m["month"] = pd.to_datetime(m["month"])
    g = m.groupby("vin")
    drop = g["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1])
    months = g.size(); span = g["age_months"].max() - g["age_months"].min()
    degv = set(drop[drop >= 2].index)
    thin = {v for v in m.vin.unique() if months[v] < MIN_MONTHS or span[v] < MIN_SPAN}
    TR, VA, TE = split(list(m.vin.unique()), drop)
    pool = TR | VA
    sets = {"ALL": pool, "QUALITY": pool - thin, "DEG-only": {v for v in pool if v in degv}}

    def train(vs):
        f = m[m.vin.isin(vs)]
        return (M.train_traj(M.build_traj_samples(f)) if oem == "Euler"
                else M.train_quantiles(M.build_transitions(f)))

    def p50(hist, fm, H):
        if oem == "Euler":
            return np.asarray(M.forecast(hist, fm, H)[0.5])
        return M.simulate(hist, fm, H)["q50"].to_numpy()

    test_deg = [v for v in TE if v in degv and months[v] >= 8]
    for label, vs in sets.items():
        fm = train(vs); errs = []
        for v in test_deg:
            gg = m[m.vin == v].sort_values("month").reset_index(drop=True); n = len(gg)
            cut = max(4, int(round(n * 0.6))); hist = gg.iloc[:cut]; tail = gg.iloc[cut:]
            if len(tail) < 2:
                continue
            a0 = float(gg.age_months.iloc[cut - 1]); H = int(np.ceil(gg.age_months.iloc[-1] - a0)) + 2
            pf = p50(hist, fm, max(H, 1))
            for _, row in tail.iterrows():
                idx = int(round(row.age_months - a0)) - 1
                if 0 <= idx < len(pf):
                    errs.append(pf[idx] - row.soh)
        errs = np.array(errs)
        print(f"{oem:9} {label:10} {len(vs):>6} {len(test_deg):>6} {errs.mean():>11.2f} {np.abs(errs).mean():>7.2f}")
    print(f"          (thin vehicles dropped by QUALITY filter from train pool: {len(pool & thin)} of {len(pool)})")
