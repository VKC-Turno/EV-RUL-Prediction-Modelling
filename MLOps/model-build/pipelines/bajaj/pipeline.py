"""Bajaj SoH-forecaster build pipeline.

Thin wrapper over common.pipeline_factory. Bajaj has no independent yardstick, so the factory builds the
DAG without the acceptance gate (config.bajaj.has_gate is False): preprocess -> train -> evaluate ->
RegisterModel (ModelApprovalStatus parameter).
"""
from pipelines.common import pipeline_factory

OEM = "bajaj"


def get_pipeline(region=None, role=None, default_bucket=None,
                 pipeline_name=None, model_package_group_name=None, **kwargs):
    return pipeline_factory.get_pipeline(
        oem=OEM, region=region, role=role, default_bucket=default_bucket,
        pipeline_name=pipeline_name, model_package_group_name=model_package_group_name)
