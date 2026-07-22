#!/usr/bin/env python3
"""Coulomb-yardstick ACCEPTANCE GATE for Euler SoH target changes — the standing safeguard.

Why this exists: a target change (e.g. -> the recovery-aware clean soh_label) can lower the model's *self-target*
LOVO RMSE simply by being smoother, while actually forecasting real capacity WORSE on the vehicles that matter.
The first soh_label retrain did exactly that — it looked +30% better against soh_full, but soh_full is NOT
independent of the target (both come from the same near-full batteryRemainingCapacity family), and against a
genuinely independent yardstick the deployment-critical DEGRADING cohort regressed.

This gate scores a CANDIDATE target's forecasts against the PHYSICALLY INDEPENDENT coulomb full-charge SoH
(computed from measured current in euler_full_charge_soh.py, not from BMS capacity bookkeeping). Coulomb is noisy
per-event, but the noise CANCELS in a paired candidate-vs-production comparison, so it is a sound arbiter where a
coulomb-derived *target* would not be. A candidate PASSES only if, on the coulomb-confirmed DECLINER cohort, it
does not regress vs the incumbent production target in either RMSE or optimism-bias.

Usage:
  .venv/bin/python src/euler_accept_gate.py [--candidate soh_target] [--holdout 0.4] [--ckpt PATH]
Importable:
  from euler_accept_gate import run_gate; res = run_gate("soh_target")  # -> dict with ["verdict"] == "PASS"/"FAIL"
Exit code is nonzero on FAIL so a promotion script can gate on it.
"""
import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import theilslopes

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # chdir moved into main() so this imports cleanly
import euler_model as em
import data_quality

FEAT = "data/euler/features/feature_table.parquet"
BMS = "data/euler/bms_soh.parquet"
COUL = "data/euler/full_charge_soh.parquet"
HOLDOUT, MIN_HIST, DECL_PPY = 0.40, 4, 3.0
RMSE_TOL, BIAS_TOL = 0.25, 0.5            # allowed candidate slack vs production on the decliner cohort


def _slope_ppy(age, y, minn=4, minspan=4.0):
    a = np.asarray(age, float); v = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(v); a, v = a[m], v[m]
    if len(a) < minn or (a.max() - a.min()) < minspan:
        return np.nan
    return float(-theilslopes(v, a)[0] * 12.0)


def _forecast_at(train_m, hist, ages_abs, anchor_age, tgt):
    """Train the trajectory model on train_m[tgt], forecast, and return predicted SoH at each absolute age."""
    mm = train_m.copy(); mm["soh"] = mm[tgt].to_numpy()
    h = hist.copy(); h["soh"] = h[tgt].to_numpy()
    dmax = max(1, int(np.ceil(ages_abs.max() - anchor_age)) + 1)
    mdl = em.train_traj(em.build_traj_samples(mm))
    p50 = np.asarray(em.forecast(h, mdl, dmax)[0.5], float)
    xs = np.arange(1, dmax + 1)
    return np.interp(np.clip(ages_abs - anchor_age, xs[0], xs[-1]), xs, p50)


def _frame(candidate, feat=FEAT, bms=BMS):
    m = pd.read_parquet(feat)
    m["vin"] = m["vin"].astype(str); m["month"] = pd.to_datetime(m["month"])
    m = data_quality.apply_quality(m, "Euler")
    lab = pd.read_parquet(bms)[list(dict.fromkeys(["vin", "month", "soh_target", candidate]))].copy()
    lab["vin"] = lab["vin"].astype(str); lab["month"] = pd.to_datetime(lab["month"])
    m = m.merge(lab, on=["vin", "month"], how="inner")
    m = m[m[candidate].notna()].sort_values(["vin", "month"]).reset_index(drop=True)
    m["soh_prod"] = m["soh"].astype(float)
    return m


def run_gate(candidate="soh_target", holdout=HOLDOUT, ckpt=None, log=print, feat=FEAT, bms=BMS, coul=COUL):
    """LOVO-score `candidate` vs production against the independent coulomb yardstick; return a verdict dict.
    Paths default to the local data/euler/ layout; SageMaker passes them as processing-input channels."""
    m = _frame(candidate, feat, bms)
    C = pd.read_parquet(coul)[["vin", "age_months", "soh_full"]].copy()
    C["vin"] = C["vin"].astype(str)
    C["coul"] = np.clip(pd.to_numeric(C["soh_full"], errors="coerce"), None, 100.0)
    C = C.dropna(subset=["age_months", "coul"])
    decl = {v: _slope_ppy(c["age_months"], c["coul"]) for v, c in C.groupby("vin")}

    vins = sorted(v for v, g in m.groupby("vin") if len(g) >= MIN_HIST + 2 and v in set(C["vin"]))
    done = set(pd.read_parquet(ckpt)["vin"].astype(str)) if (ckpt and Path(ckpt).exists()) else set()
    rows, todo = [], [v for v in vins if v not in done]
    log(f"acceptance gate: {len(vins)} scoreable vehicles ({len(done)} cached, {len(todo)} to run)")
    for i, vin in enumerate(todo, 1):
        g = m[m["vin"] == vin].sort_values("month").reset_index(drop=True)
        n = len(g); k = max(1, min(int(round(n * holdout)), n - MIN_HIST)); cut = n - k
        hist = g.iloc[:cut]; anchor = float(hist["age_months"].iloc[-1])
        ce = C[(C["vin"] == vin) & (C["age_months"] > anchor)]
        if not len(ce):
            rows.append(dict(vin=vin, n_pts=0)); continue
        ages, truth = ce["age_months"].to_numpy(), ce["coul"].to_numpy()
        tm = m[m["vin"] != vin]
        rec = dict(vin=vin, n_pts=int(len(ce)), decl_ppy=float(decl.get(vin, np.nan)))
        for tgt in ("soh_prod", candidate):
            pred = _forecast_at(tm, hist, ages, anchor, tgt)
            rec[f"rmse_{tgt}"] = float(np.sqrt(np.mean((pred - truth) ** 2)))
            rec[f"err_{tgt}"] = float(np.mean(pred - truth))
        rows.append(rec)
        if ckpt and (i % 5 == 0 or i == len(todo)):
            acc = pd.read_parquet(ckpt) if Path(ckpt).exists() else pd.DataFrame()
            pd.concat([acc, pd.DataFrame(rows)], ignore_index=True).to_parquet(ckpt, index=False); rows = []
            log(f"  [{i}/{len(todo)}] {vin}")
    if ckpt:
        acc = pd.read_parquet(ckpt) if Path(ckpt).exists() else pd.DataFrame()
        R = pd.concat([acc, pd.DataFrame(rows)], ignore_index=True) if rows else acc
    else:
        R = pd.DataFrame(rows)
    R = R[R["vin"].isin(vins) & (R["n_pts"] > 0)].copy()

    cand_rmse, prod_rmse = f"rmse_{candidate}", "rmse_soh_prod"
    cand_err, prod_err = f"err_{candidate}", "err_soh_prod"
    cohorts = {}
    for name, sub in [("overall", R), ("decliner", R[R["decl_ppy"] >= DECL_PPY]), ("flat", R[R["decl_ppy"] < DECL_PPY])]:
        if not len(sub):
            continue
        cohorts[name] = dict(n=int(len(sub)), n_pts=int(sub["n_pts"].sum()),
                             rmse_prod=round(sub[prod_rmse].mean(), 3), rmse_cand=round(sub[cand_rmse].mean(), 3),
                             bias_prod=round(sub[prod_err].mean(), 2), bias_cand=round(sub[cand_err].mean(), 2),
                             cand_win_pct=round(float((sub[cand_rmse] < sub[prod_rmse]).mean() * 100), 0))
    d = cohorts.get("decliner", {})
    passed = bool(d and d["rmse_cand"] <= d["rmse_prod"] + RMSE_TOL and d["bias_cand"] <= d["bias_prod"] + BIAS_TOL)
    return dict(candidate=candidate, yardstick="independent coulomb full-charge SoH", decl_ppy=DECL_PPY,
                cohorts=cohorts, verdict="PASS" if passed else "FAIL")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)       # CLI: resolve the default data/euler/ paths
    cand = "soh_target"; holdout = HOLDOUT; ckpt = None
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == "--candidate" and i + 1 < len(a): cand = a[i + 1]
        elif x == "--holdout" and i + 1 < len(a): holdout = float(a[i + 1])
        elif x == "--ckpt" and i + 1 < len(a): ckpt = a[i + 1]
    res = run_gate(cand, holdout=holdout, ckpt=ckpt)
    print(json.dumps(res, indent=2))
    print(f"\nGATE VERDICT [{cand}]: {res['verdict']}")
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
