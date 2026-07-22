"""Coulomb ACCEPTANCE-GATE step — OUR real gate (src/euler_accept_gate), the promotion decision.

Scores the candidate SoH target's forecasts against the PHYSICALLY INDEPENDENT coulomb full-charge SoH on the
coulomb-confirmed DECLINER cohort. PASS only if the candidate does not regress vs the incumbent production
target in either RMSE or optimism-bias. Writes gate.json -> {"verdict": "PASS"|"FAIL", ...}; the pipeline's
ConditionStep reads it to register Approved vs PendingManualApproval. (See src/euler_accept_gate.py.)

Inputs (Processing channels):
    /opt/ml/processing/featengg    the euler_featengg feature store (features + incumbent `soh`)
    /opt/ml/processing/yardstick   bms_soh.parquet (candidate soh_target) + full_charge_soh.parquet (coulomb)
    /opt/ml/processing/input/code  the repo `src/` (euler_accept_gate + euler_model + data_quality), on sys.path
Output:
    /opt/ml/processing/gate/gate.json
"""
import glob
import json
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, "/opt/ml/processing/input/code")       # mounted repo src/
import euler_accept_gate as gate                            # noqa: E402  (our real gate)

FEATENGG, YARD, OUT = "/opt/ml/processing/featengg", "/opt/ml/processing/yardstick", "/opt/ml/processing/gate"


def _pick(dir_, needle):
    files = glob.glob(os.path.join(dir_, "**", "*.parquet"), recursive=True)
    hits = [f for f in files if needle in os.path.basename(f)]
    return (hits or files)[0]


def main():
    # the feature store keys on `ymd`; euler_accept_gate wants `month` -> materialise a month-keyed copy
    feat_files = glob.glob(os.path.join(FEATENGG, "**", "*.parquet"), recursive=True)
    m = pd.concat((pd.read_parquet(f) for f in feat_files), ignore_index=True)
    if "month" not in m.columns and "ymd" in m.columns:
        m["month"] = pd.to_datetime(m["ymd"])
    feat_tmp = "/tmp/euler_featengg_month.parquet"
    m.to_parquet(feat_tmp, index=False)

    bms = _pick(YARD, "bms_soh")
    coul = _pick(YARD, "full_charge")
    res = gate.run_gate(candidate="soh_target", feat=feat_tmp, bms=bms, coul=coul)

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(OUT, "gate.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"GATE VERDICT: {res['verdict']} | decliner cohort: {res.get('cohorts', {}).get('decliner')}")
    # the verdict travels via gate.json for the ConditionStep — never fail the step itself
    sys.exit(0)


if __name__ == "__main__":
    main()
