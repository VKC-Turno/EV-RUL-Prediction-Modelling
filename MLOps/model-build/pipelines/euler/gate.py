"""Euler ACCEPTANCE GATE entry point (Processing step) — the standing safeguard.

Scores a CANDIDATE SoH target's forecasts against a PHYSICALLY INDEPENDENT yardstick (coulomb full-charge
SoH) on the coulomb-confirmed DECLINER cohort. PASS only if the candidate does not regress vs the incumbent
production target in either RMSE or optimism-bias. Emits gate.json -> {"verdict": "PASS"|"FAIL", ...}, which
the ConditionStep reads to decide auto-approve vs hold-for-review.

Why independent: a smoother target lowers the model's *self*-target error without being more accurate; the
first soh_label retrain looked +30% better against a non-independent yardstick and was correctly rejected.
Canonical implementation: src/euler_accept_gate.py (this is the pipeline-facing port).

Inputs (Processing):
    /opt/ml/processing/featengg   euler featengg (candidate target = `soh`, incumbent = `soh_prod` if present)
    /opt/ml/processing/yardstick  coulomb full-charge SoH [vin, age_months, soh_full]  (optional)
"""
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

RMSE_TOL, BIAS_TOL, DECL_PPY = 0.25, 0.5, 3.0     # candidate slack vs production on the decliner cohort
OUT = "/opt/ml/processing/gate"


def _read(dir_):
    files = glob.glob(os.path.join(dir_, "**", "*.parquet"), recursive=True)
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else None


def run():
    m = _read("/opt/ml/processing/featengg")
    yard = _read("/opt/ml/processing/yardstick")
    cohorts, verdict = {}, "PASS"

    if m is not None and yard is not None and {"vin", "age_months", "soh_full"} <= set(yard.columns):
        yard = yard.dropna(subset=["age_months", "soh_full"]).copy()
        # decliner cohort = coulomb slope >= DECL_PPY
        decl = set()
        for vin, g in yard.groupby("vin"):
            g = g.sort_values("age_months")
            span = g["age_months"].iloc[-1] - g["age_months"].iloc[0]
            if span >= 4 and (g["soh_full"].iloc[0] - g["soh_full"].iloc[-1]) / span * 12 >= DECL_PPY:
                decl.add(str(vin))
        cand = "soh"
        incumbent = "soh_prod" if "soh_prod" in m.columns else "soh"
        merged = m.merge(yard[["vin", "age_months", "soh_full"]], on=["vin", "age_months"], how="inner")
        sub = merged[merged["vin"].astype(str).isin(decl)]
        if len(sub):
            def rmse(col): return float(np.sqrt(np.mean((sub[col] - sub["soh_full"]) ** 2)))
            def bias(col): return float(np.mean(sub[col] - sub["soh_full"]))
            cohorts["decliner"] = dict(n=int(sub["vin"].nunique()),
                                       rmse_cand=round(rmse(cand), 3), rmse_prod=round(rmse(incumbent), 3),
                                       bias_cand=round(bias(cand), 2), bias_prod=round(bias(incumbent), 2))
            d = cohorts["decliner"]
            verdict = "PASS" if (d["rmse_cand"] <= d["rmse_prod"] + RMSE_TOL and
                                 d["bias_cand"] <= d["bias_prod"] + BIAS_TOL) else "FAIL"
    else:
        # no independent yardstick available in this run -> do not auto-approve a target change
        verdict = "PASS" if (m is not None and "soh_prod" not in m.columns) else "FAIL"
        cohorts["note"] = "no coulomb yardstick input; see src/euler_accept_gate.py for the full check"

    report = dict(candidate="soh", yardstick="independent coulomb full-charge SoH",
                  decl_ppy=DECL_PPY, rmse_tol=RMSE_TOL, bias_tol=BIAS_TOL,
                  cohorts=cohorts, verdict=verdict)
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "gate.json").write_text(json.dumps(report, indent=2))
    print(f"GATE VERDICT [euler]: {verdict}  {cohorts}")
    sys.exit(0)     # verdict travels via gate.json; never fail the step itself


if __name__ == "__main__":
    run()
