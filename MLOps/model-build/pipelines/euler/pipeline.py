"""Euler SoH-forecaster build pipeline.

Thin wrapper over common.pipeline_factory. Euler is the only OEM with a physically-independent yardstick
(coulomb full-charge SoH), so its DAG additionally runs the ACCEPTANCE GATE + ConditionStep before
RegisterModel (added automatically by the factory because config.euler.has_gate is True).
"""
from pipelines.common import pipeline_factory

OEM = "euler"


def get_pipeline(region=None, role=None, default_bucket=None,
                 pipeline_name=None, model_package_group_name=None, **kwargs):
    return pipeline_factory.get_pipeline(
        oem=OEM, region=region, role=role, default_bucket=default_bucket,
        pipeline_name=pipeline_name, model_package_group_name=model_package_group_name)
