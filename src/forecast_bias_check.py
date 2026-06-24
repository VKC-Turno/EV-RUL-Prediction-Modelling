#!/usr/bin/env python3
"""Is the forecast biased HIGH (over-predicting battery life) on active decliners — and does training on
degraders-only fix it? Backtests held-out DEGRADER vehicles: forecast their last ~40% from the first ~60%
and measure the SIGNED error (predicted SoH - actual SoH) at the held-out tail.

  signed error > 0  => forecast ABOVE actual = over-predicting life (too optimistic)  <- the user's claim
  signed error ~ 0  => well-calibrated
MAE is reported alongside so we see accuracy, not just bias. Test vehicles are NOT in either training set
(clean out-of-sample), and the SAME test vehicles are scored both ways.

Run: .venv/bin/python src/forecast_bias_check.py
"""
import os, sys, importlib, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")

SPEC = {"Euler": ("data/euler/features/feature_table.parquet", "euler_model"),
        "Mahindra": ("data/mahindra/features/feature_table.parquet", "model"),
        "Bajaj": ("data/bajaj/features/feature_table.parquet", "bajaj_model")}


def split(vins, drop, seed=0):
    rng = np.random.RandomState(seed); out = [set(), set(), set()]
    for grp in (sorted(v for v in vins if drop[v] >= 2), sorted(v for v in vins if drop[v] < 2)):
        grp = list(grp); rng.shuffle(grp); n = len(grp); ntr, nva = int(n * .6), int(n * .2)
        for i, s in enumerate((grp[:ntr], grp[ntr:ntr + nva], grp[ntr + nva:])):
            out[i] |= set(s)
    return out


print(f"{'OEM':9} {'mode':9} {'n':>3} {'signed err (bias)':>18} {'MAE':>7}   (signed>0 = over-predicts life)")
for oem, (ft, modname) in SPEC.items():
    M = importlib.import_module(modname); m = pd.read_parquet(ft); m["month"] = pd.to_datetime(m["month"])
    drop = m.groupby("vin")["soh"].agg(lambda s: s.iloc[0] - s.iloc[-1]); degv = set(drop[drop >= 2].index)
    TR, VA, TE = split(list(m.vin.unique()), drop)
    def train(vs):
        f = m[m.vin.isin(vs)]
        return (M.train_traj(M.build_traj_samples(f)) if oem == "Euler"
                else M.train_quantiles(M.build_transitions(f)))
    fm_all = train(TR | VA)
    fm_deg = train({v for v in (TR | VA) if v in degv})
    def p50(hist, fm, H):
        if oem == "Euler":
            return np.asarray(M.forecast(hist, fm, H)[0.5])
        return M.simulate(hist, fm, H)["q50"].to_numpy()
    test_deg = [v for v in TE if v in degv and len(m[m.vin == v]) >= 8]
    for label, fm in [("ALL", fm_all), ("DEG-only", fm_deg)]:
        errs = []
        for v in test_deg:
            g = m[m.vin == v].sort_values("month").reset_index(drop=True); n = len(g)
            cut = max(4, int(round(n * 0.6))); hist = g.iloc[:cut]; tail = g.iloc[cut:]
            if len(tail) < 2:
                continue
            a0 = float(g.age_months.iloc[cut - 1]); H = int(np.ceil(g.age_months.iloc[-1] - a0)) + 2
            pf = p50(hist, fm, max(H, 1))
            for _, row in tail.iterrows():
                idx = int(round(row.age_months - a0)) - 1
                if 0 <= idx < len(pf):
                    errs.append(pf[idx] - row.soh)             # pred - actual
        errs = np.array(errs)
        print(f"{oem:9} {label:9} {len(test_deg):>3} {errs.mean():>18.2f} {np.abs(errs).mean():>7.2f}")
