"""Shared SageMaker Pipeline factory — builds one OEM's build DAG from its registry entry.

    preprocess (Processing) -> train (Training) -> evaluate (Processing)
        -> [acceptance gate (Processing) -> ConditionStep]   # only when cfg.has_gate
        -> RegisterModel

Every pipelines/<oem>/pipeline.py just calls get_pipeline(oem=..., ...). The gate + ConditionStep are
added only for OEMs with a physically-independent yardstick (today: Euler). This is the AWS SageMaker
MLOps "ModelBuild" template (pipelines/abalone) generalised to per-OEM + a shared common package.
"""
import os

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource

from . import config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pipelines
FRAMEWORK_VERSION = "1.2-1"


def _session(region, default_bucket):
    boto = sagemaker.Session().boto_session if region is None else None
    return PipelineSession(default_bucket=default_bucket)


def get_pipeline(
    oem: str,
    region: str = None,
    role: str = None,
    default_bucket: str = None,
    pipeline_name: str = None,
    model_package_group_name: str = None,
) -> Pipeline:
    cfg = config.get(oem)
    pipeline_name = pipeline_name or f"soh-{oem}-build"
    model_package_group_name = model_package_group_name or f"soh-forecaster-{oem}"
    session = _session(region, default_bucket)
    role = role or sagemaker.get_execution_role()

    # ── parameters ───────────────────────────────────────────────────────────────────────
    proc_instance = ParameterString("ProcessingInstanceType", default_value="ml.m5.2xlarge")
    train_instance = ParameterString("TrainingInstanceType", default_value="ml.m5.xlarge")
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    input_data = ParameterString(
        "InputDataUrl",
        default_value=f"s3://{default_bucket}/curated/{oem}/")   # compacted telemetry
    min_hist = ParameterInteger("MinMonths", default_value=3)

    sk_proc = SKLearnProcessor(framework_version=FRAMEWORK_VERSION, role=role,
                               instance_type=proc_instance, instance_count=1,
                               base_job_name=f"{oem}-preprocess", sagemaker_session=session)

    # ── 1) preprocess: telemetry -> featengg ──────────────────────────────────────────────
    step_pre = ProcessingStep(
        name=f"{oem.capitalize()}Preprocess",
        processor=sk_proc,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(output_name="featengg", source="/opt/ml/processing/output/featengg")],
        code=os.path.join(HERE, oem, "preprocess.py"),
    )

    # ── 2) train ──────────────────────────────────────────────────────────────────────────
    est = SKLearn(entry_point="train.py", source_dir=os.path.join(HERE, "common"),
                  framework_version=FRAMEWORK_VERSION, role=role, instance_type=train_instance,
                  instance_count=1, base_job_name=f"{oem}-train",
                  hyperparameters={"oem": oem}, sagemaker_session=session)
    step_train = TrainingStep(
        name=f"{oem.capitalize()}Train",
        estimator=est,
        inputs={"train": TrainingInput(
            s3_data=step_pre.properties.ProcessingOutputConfig.Outputs["featengg"].S3Output.S3Uri,
            content_type="application/x-parquet")},
    )

    # ── 3) evaluate (held-out backtest) ───────────────────────────────────────────────────
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name=f"{oem.capitalize()}Evaluate",
        processor=sk_proc,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                            destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_pre.properties.ProcessingOutputConfig.Outputs["featengg"].S3Output.S3Uri,
                            destination="/opt/ml/processing/featengg"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(HERE, oem, "evaluate.py"),
        property_files=[eval_report],
    )

    model_metrics = ModelMetrics(model_statistics=MetricsSource(
        s3_uri=step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/evaluation.json",
        content_type="application/json"))

    def _register(status):
        return RegisterModel(
            name=f"{oem.capitalize()}Register{status.replace(' ', '')}",
            estimator=est,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/json"], response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=model_package_group_name,
            approval_status=status, model_metrics=model_metrics)

    steps = [step_pre, step_train, step_eval]

    # ── 4) acceptance gate (gated OEMs only) ──────────────────────────────────────────────
    if cfg.has_gate:
        gate_report = PropertyFile(name="GateReport", output_name="gate", path="gate.json")
        step_gate = ProcessingStep(
            name=f"{oem.capitalize()}AcceptanceGate",
            processor=sk_proc,
            inputs=[ProcessingInput(
                source=step_pre.properties.ProcessingOutputConfig.Outputs["featengg"].S3Output.S3Uri,
                destination="/opt/ml/processing/featengg")],
            outputs=[ProcessingOutput(output_name="gate", source="/opt/ml/processing/gate")],
            code=os.path.join(HERE, oem, "gate.py"),
            property_files=[gate_report],
        )
        cond = ConditionEquals(
            left=JsonGet(step_name=step_gate.name, property_file=gate_report, json_path="verdict"),
            right="PASS")
        step_cond = ConditionStep(
            name=f"{oem.capitalize()}GateCondition",
            conditions=[cond],
            if_steps=[_register("Approved")],           # PASS -> auto-approve
            else_steps=[_register("PendingManualApproval")],  # FAIL -> hold for manual review
        )
        steps += [step_gate, step_cond]
    else:
        steps += [_register(approval)]

    return Pipeline(
        name=pipeline_name,
        parameters=[proc_instance, train_instance, approval, input_data, min_hist],
        steps=steps,
        sagemaker_session=session,
    )
