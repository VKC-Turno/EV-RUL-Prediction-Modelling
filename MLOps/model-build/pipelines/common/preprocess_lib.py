"""Shared preprocessing driver for the SoH feature-generation Processing step.

Each pipelines/<oem>/preprocess.py is a 3-line entry point that calls `run(oem, ...)`. The per-feed SoH
method is selected from the OEM registry (common.config) — that is the ONLY branch. Output is the
`<oem>_featengg` table (common.features.SCHEMA), written for both the Feature Store ingest and training.

SageMaker Processing contract:
    input  : /opt/ml/processing/input   (curated/compacted telemetry parquet for this OEM)
    output : /opt/ml/processing/output/featengg/<oem>.parquet
"""
import argparse
import pathlib
import pandas as pd

from . import config, data_quality, soh, features

IN_DEFAULT = "/opt/ml/processing/input"
OUT_DEFAULT = "/opt/ml/processing/output"


def _read_telemetry(input_dir: str) -> pd.DataFrame:
    """Read the compacted per-OEM telemetry. (Compaction upstream solves the tiny-files problem.)"""
    files = sorted(pathlib.Path(input_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {input_dir}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if "eventAt" in df.columns and "t" not in df.columns:
        df["t"] = pd.to_datetime(df["eventAt"], unit="ms", errors="coerce")
    df["vin"] = df["vin"].astype("category")
    return df


def run(oem: str, input_dir: str = IN_DEFAULT, output_dir: str = OUT_DEFAULT) -> str:
    cfg = config.get(oem)
    df = _read_telemetry(input_dir)
    df = data_quality.clip_sentinels(df)

    # 1) SoH target — the one per-feed branch
    soh_df = soh.compute(cfg.soh_method, df)

    # 2) monthly stress features
    feat_df = features.electrical_features(df)

    # 3) usage (odometer -> km/month), if present
    usage_df = None
    if "odometer" in df.columns:
        df["month"] = pd.to_datetime(df["t"]).values.astype("datetime64[M]")
        usage_df = (df.groupby(["vin", "month"])["odometer"].max()
                      .reset_index().rename(columns={"odometer": "odo_max"}))

    m = features.assemble(soh_df, feat_df, usage_df)
    m = data_quality.apply_quality(m.assign(month=pd.to_datetime(m["ymd"])))
    m = m.drop(columns=["month"], errors="ignore")

    out = pathlib.Path(output_dir) / "featengg"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{oem}.parquet"
    m.to_parquet(path, index=False)
    print(f"[{oem}] wrote {len(m)} vin-months -> {path} (soh_method={cfg.soh_method}, gate={cfg.has_gate})")
    return str(path)


def cli(oem: str):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=IN_DEFAULT)
    p.add_argument("--output", default=OUT_DEFAULT)
    a = p.parse_args()
    run(oem, a.input, a.output)
