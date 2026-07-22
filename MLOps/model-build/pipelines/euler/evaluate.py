"""Evaluate step — surface the model's evaluation.json as a STANDALONE artifact for ModelMetrics.

train.py writes evaluation.json (LOVO RMSE + stratified train/val/test diagnostics + feature importance)
INTO model.tar.gz. This step extracts it to a ProcessingOutput so the pipeline can (a) attach it to the
registered model as ModelMetrics and (b) read it via a PropertyFile. Splitting evaluate from train also lets a
future version recompute metrics independently of the training run.

    input   /opt/ml/processing/model/model.tar.gz   (from Train's ModelArtifacts)
    output  /opt/ml/processing/evaluation/evaluation.json
"""
import glob
import json
import os
import pathlib
import tarfile

MODEL_DIR = "/opt/ml/processing/model"
OUT = "/opt/ml/processing/evaluation"


def main():
    report = None
    for tar in glob.glob(os.path.join(MODEL_DIR, "*.tar.gz")):
        with tarfile.open(tar) as tf:
            member = next((n for n in tf.getnames() if n.endswith("evaluation.json")), None)
            if member is not None:
                report = json.load(tf.extractfile(member))
                break
    if report is None:                                    # model dir may already be extracted
        loose = glob.glob(os.path.join(MODEL_DIR, "**", "evaluation.json"), recursive=True)
        report = json.load(open(loose[0])) if loose else {"error": "evaluation.json not found"}

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(OUT, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("surfaced evaluation.json:",
          {k: report.get(k) for k in ("overall_rmse", "degrading_rmse", "flat_rmse", "band_coverage")})


if __name__ == "__main__":
    main()
