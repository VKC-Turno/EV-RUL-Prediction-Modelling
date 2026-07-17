"""Render CloudFormation parameter files for each OEM's staging + prod deployment.

For every OEM with an Approved model package, emit `<oem>-staging-config-export.json` and
`<oem>-prod-config-export.json` (the parameter files the CFN template in endpoint-config-template.yml
consumes). OEMs with no Approved package (e.g. the Montra placeholder) are skipped.

Env (injected by SageMaker Projects): SAGEMAKER_PROJECT_NAME, SAGEMAKER_PROJECT_ID,
SAGEMAKER_PIPELINE_ROLE_ARN (model execution role), AWS_REGION. OEMS overrides the fleet list.
"""
import argparse
import json
import os

from common.build_helpers import get_approved_package

OEMS = os.environ.get("OEMS", "euler mahindra bajaj piaggio montra").split()


def _extend(base_path, extra):
    with open(base_path) as f:
        cfg = json.load(f)
    cfg["Parameters"].update(extra)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=os.environ.get("AWS_REGION"))
    p.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_PIPELINE_ROLE_ARN"))
    a = p.parse_args()

    project = os.environ.get("SAGEMAKER_PROJECT_NAME", "soh-rul")
    project_id = os.environ.get("SAGEMAKER_PROJECT_ID", "")

    for oem in OEMS:
        arn = get_approved_package(f"soh-forecaster-{oem}", a.region)
        if arn is None:
            continue
        common = {
            "SageMakerProjectName": project,
            "SageMakerProjectId": project_id,
            "ModelExecutionRoleArn": a.role_arn,
            "ModelPackageName": arn,
            "OEM": oem,
        }
        staging = _extend("staging-config.json", {**common, "EndpointName": f"soh-{oem}-staging"})
        prod = _extend("prod-config.json", {**common, "EndpointName": f"soh-{oem}-prod"})
        with open(f"{oem}-staging-config-export.json", "w") as f:
            json.dump(staging, f, indent=2)
        with open(f"{oem}-prod-config-export.json", "w") as f:
            json.dump(prod, f, indent=2)
        print(f"[{oem}] wrote staging + prod CFN parameter files")


if __name__ == "__main__":
    main()
