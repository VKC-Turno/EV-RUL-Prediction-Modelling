# MLOps — Battery SoH / RUL

A clean, per-OEM MLOps scaffold for the battery State-of-Health / Remaining-Useful-Life platform, laid out
like the two Turno reference repos — a **model-build** side and a **model-deploy** side — with **one folder
per OEM** and a shared **`common/`** package on each side.

```
MLOps/
├── model-build/     # SageMaker Pipelines: preprocess -> train -> evaluate -> [gate] -> register
│   └── pipelines/{common, euler, mahindra, bajaj, piaggio, montra}/
└── model-deploy/    # fetch Approved package -> CloudFormation endpoint (staging -> test -> prod)
    └── {common, test}/
```

## Design in one line

**One parameterised pipeline, five parameter sets.** The steps are identical for every OEM; what changes is
the *feed audit → SoH method → whether the acceptance gate applies → whether the fleet is mature enough to
auto-approve*. All of that lives in `model-build/pipelines/common/config.py` (the OEM registry) — onboarding
an OEM is *add one entry*, not new pipeline code.

| OEM | SoH method | Model family | Gate | Fleet |
|---|---|---|---|---|
| **Euler** | BMS remaining-capacity → hybrid | `euler_model` | **yes** | mature |
| **Mahindra** | coulomb (+ behaviour model for native-only) | `model` | no | mature |
| **Bajaj** | reported BMS SoH | `bajaj_model` | no | young |
| **Piaggio** | coulomb (via intellicar) | `model` | no | mature |
| **Montra** | BMS remaining-capacity | `model` | no | placeholder |

## Where things live

- **Science + DAG factory** → `model-build/pipelines/common/` (`config`, `soh`, `features`, `forecaster`,
  `data_quality`, `train`, `backtest_lib`, `pipeline_factory`).
- **Per-OEM entry points** → `model-build/pipelines/<oem>/` (`pipeline.py`, `preprocess.py`, `evaluate.py`;
  Euler also `gate.py`).
- **Deployment** → `model-deploy/` (approved-package fetch, CFN endpoint template, stage configs, smoke test).

## Verify locally

```bash
cd model-build && pip install -e .[test] && pytest        # exercises common/ (no AWS)
```

## References

- Structure follows the AWS SageMaker MLOps *ModelBuild* / *ModelDeploy* templates
  (`pipelines/abalone`) and the Turno `iot-model-build-mlops` / `iot-model-deploy-mlops` repos
  (**left unmodified**).
- Architecture, migration plan, and per-OEM flows: `../docs/MLOPS_SAGEMAKER.md`.
- Canonical science: the research repo `../src/` (this `common/` is its pipeline-facing port).

> This scaffold is self-contained under `MLOps/`; it does not modify the research pipeline or the original
> MLOps repos.
