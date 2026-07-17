"""Shared model-deploy helpers — fetch the latest APPROVED model package per OEM from the registry.

The build side registers per-OEM Model Package Groups (`soh-forecaster-<oem>`). Deploy only ever ships a
package whose approval status is Approved — for Euler that is the output of the acceptance-gate
ConditionStep; for the others it is a manual approval in Studio. Placeholder fleets (Montra) should stay
PendingManualApproval, so this returns nothing for them and the deploy is skipped.
"""
import boto3


def get_approved_package(model_package_group_name: str, region: str = None):
    """Return the ARN of the most-recent Approved model package in the group, or None."""
    sm = boto3.client("sagemaker", region_name=region) if region else boto3.client("sagemaker")
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print(f"[{model_package_group_name}] no Approved package — skipping deploy")
        return None
    arn = pkgs[0]["ModelPackageArn"]
    print(f"[{model_package_group_name}] latest approved: {arn}")
    return arn
