# SoH/RUL — Model Deploy

Per-OEM **model-deploy** side, mirroring the AWS SageMaker MLOps *ModelDeploy* template. Takes the latest
**Approved** model package for each OEM from the Model Registry (produced by the `model-build` repo) and
deploys it — staging → smoke-test → prod.

```
model-deploy/
├── buildspec.yml               # CodeBuild: fetch approved packages -> render CFN params
├── build.py                    # per-OEM: get latest Approved package -> <oem>-{staging,prod}-config-export.json
├── endpoint-config-template.yml# CloudFormation: Model + EndpointConfig + Endpoint (+ DataCapture for Monitor)
├── staging-config.json         # staging stage params (instance type/count, capture path)
├── prod-config.json            # prod stage params
├── common/build_helpers.py     # get_approved_package(group) from the registry
└── test/
    ├── buildspec.yml
    └── test.py                 # smoke-test each live staging endpoint (q50 in [0,100])
```

## Deploy path

1. `build.py` lists each `soh-forecaster-<oem>` group and pulls the newest **Approved** package. OEMs with
   no approved package (e.g. the **Montra placeholder**, which stays `PendingManualApproval`) are skipped.
2. It renders `<oem>-staging-config-export.json` / `<oem>-prod-config-export.json` from the base stage
   configs + the package ARN.
3. The wrapping CodePipeline deploys `endpoint-config-template.yml` to **staging**, runs `test/test.py`, and
   on success promotes to **prod**.

## Batch first, endpoints second

The product is **monthly, fleet-wide** scoring, so the primary path is **Batch Transform** over the current
month's features (wire it as a scheduled step; it needs no always-on endpoint). The real-time endpoints here
exist for the **customer app**'s on-demand per-VIN lookups. `DataCaptureConfig` on every endpoint feeds
SageMaker **Model Monitor** (data-quality + model-quality drift). Remember the model-quality signal is
**trailing** (ground truth arrives months later) and that **placeholder fleets** should have drift alarms
suppressed. See `docs/MLOPS_SAGEMAKER.md`.
