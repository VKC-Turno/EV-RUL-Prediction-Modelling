# SoH/RUL — Model Build (SageMaker Pipelines)

Per-OEM SageMaker **model-build** pipelines for battery State-of-Health / RUL forecasting. Mirrors the AWS
SageMaker MLOps *ModelBuild* template (`pipelines/abalone`), generalised to **one folder per OEM + a shared
`common/` package**. Sibling of the `model-deploy` repo.

```
model-build/
├── buildspec.yml                 # CodeBuild: upsert + start one pipeline per OEM
├── setup.py / setup.cfg / tox.ini
├── pipelines/
│   ├── get_pipeline_definition.py   # CLI: print a pipeline's definition JSON
│   ├── run_pipeline.py              # CLI: upsert + start a pipeline
│   ├── _utils.py  __version__.py
│   ├── common/                      # ← shared functionality (the science + the DAG factory)
│   │   ├── config.py                #   OEM registry (soh_method · model_module · eol · warranty · has_gate)
│   │   ├── data_quality.py          #   sentinel clipping + data-thin gate
│   │   ├── soh.py                   #   per-feed SoH: coulomb / bms_capacity / reported
│   │   ├── features.py              #   electrical features + the featengg SCHEMA
│   │   ├── forecaster.py            #   quantile trajectory model interface (+ reference impl)
│   │   ├── train.py                 #   SageMaker Training entry point (script mode)
│   │   ├── backtest_lib.py          #   held-out backtest -> evaluation.json
│   │   └── pipeline_factory.py      #   builds the DAG from an OEM's registry entry
│   ├── euler/                       # ← one folder per OEM
│   │   ├── pipeline.py              #   thin: get_pipeline() -> factory
│   │   ├── preprocess.py            #   Processing entry -> featengg
│   │   ├── evaluate.py              #   Processing entry -> backtest
│   │   └── gate.py                  #   ACCEPTANCE GATE (Euler only — has an independent yardstick)
│   ├── mahindra/  bajaj/  piaggio/  montra/   # {pipeline,preprocess,evaluate}.py
└── tests/test_pipelines.py          # unit tests for common/ (no AWS needed)
```

## The DAG (per OEM)

`preprocess (Processing)` → `train (Training)` → `evaluate (Processing)` →
**`[acceptance gate → ConditionStep]`** → `RegisterModel`

The gate + `ConditionStep` are added **only when `config.<oem>.has_gate`** is true (today: Euler). On the
Euler DAG the gate scores the candidate SoH target against a physically-independent coulomb yardstick on the
decliner cohort and registers the model `Approved` only on PASS; every other OEM registers with the
`ModelApprovalStatus` parameter (manual approval in Studio).

## Onboarding a new OEM

1. Audit the S3 feed → pick `soh_method` (coulomb / bms_capacity / reported).
2. Add one entry to `pipelines/common/config.py`.
3. Create `pipelines/<oem>/` with the three thin entry points (copy an existing OEM; add `gate.py` only if
   an independent yardstick exists).
4. Add the OEM to `buildspec.yml`'s `OEMS`. Done — no factory changes.

## Run locally

```bash
pip install -e .[test]
pytest                                   # exercises common/ (no AWS)
get-pipeline-definition --module-name pipelines.euler.pipeline \
    --kwargs '{"region":"ap-south-1","role":"<role-arn>","default_bucket":"oem-data-iot"}'
```

## Packaging note (shared `common/`)

Unlike the single-file abalone template, the per-OEM Processing entry points import the shared `common/`
package. The processors must therefore upload the whole `pipelines/` tree (via `source_dir` / a
`FrameworkProcessor`, or by adding `common/` to the code bundle) so `from common import ...` resolves in the
job container. The entry scripts add `pipelines/` to `sys.path` to make this work both locally and in-job.

## Relationship to the research repo

`common/` is the pipeline-facing port of the canonical implementations in the research repo `src/`
(`soh.py`, `euler_bms_soh.py`, `features.py`, `oem_train.py`, `euler_accept_gate.py`). Keep the OEM registry
here in sync with `src/config.py` + `src/oem_train.py::CFG` when the science changes. See
`docs/MLOPS_SAGEMAKER.md` for the full architecture (compaction, Feature Store, Model Registry, Monitor).
