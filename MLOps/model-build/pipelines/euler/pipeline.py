"""Euler SoH-forecaster model-build pipeline — END TO END, our euler_model + our coulomb gate.

Preprocessing lives in the Glue job (-> the `euler_featengg` feature store), so this pipeline is the
CONSUMING side:

    LoadFeatengg -> Train -> Evaluate
                         -> Gate -> Condition( verdict == PASS )
                                                 ├─ PASS -> RegisterModel(Approved)
                                                 └─ FAIL -> RegisterModel(PendingManualApproval)

"All logic is ours": train.py calls src/euler_model / euler_backtest / euler_train; gate.py calls
src/euler_accept_gate (the coulomb-yardstick acceptance gate). Both ship the repo `src/` — train via the
estimator's dependencies, gate via a mounted code input. Euler is the gated OEM; the gate needs the
independent coulomb full-charge SoH + the candidate target (bms_soh) as the `yardstick` input channel.
"""
import os

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource

HERE = os.path.dirname(os.path.abspath(__file__))                        # .../pipelines/euler
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
FW = "1.2-1"
ICEBERG_CONF = [{"Classification": "spark-defaults", "Properties": {
    "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    "spark.sql.catalog.glue_catalog": "org.apache.iceberg.spark.SparkCatalog",
    "spark.sql.catalog.glue_catalog.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
    "spark.sql.catalog.glue_catalog.io-impl": "org.apache.iceberg.aws.s3.S3FileIO"}}]


def get_pipeline(region=None, role=None, default_bucket=None,
                 pipeline_name="soh-euler-build", model_package_group_name="soh-forecaster-euler",
                 featengg_table="glue_catalog.turno_ml.euler_featengg", **kwargs):
    session = PipelineSession(default_bucket=default_bucket)
    role = role or sagemaker.get_execution_role()

    proc_type = ParameterString("ProcessingInstanceType", default_value="ml.m5.xlarge")
    train_type = ParameterString("TrainingInstanceType", default_value="ml.m5.2xlarge")
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    # gate inputs: bms_soh (candidate soh_target) + full_charge_soh (coulomb yardstick); and the repo src/
    yardstick_url = ParameterString("YardstickDataUrl", default_value=f"s3://{default_bucket}/euler/yardstick/")
    src_code_url = ParameterString("SrcCodeUrl", default_value=f"s3://{default_bucket}/euler/src/")

    # ── 1) unload euler_featengg -> parquet ───────────────────────────────────────────────
    spark_proc = PySparkProcessor(framework_version="3.3", role=role, instance_type=proc_type,
                                  instance_count=1, base_job_name="euler-load-featengg", sagemaker_session=session)
    step_load = ProcessingStep(name="EulerLoadFeatengg", step_args=spark_proc.run(
        submit_app=os.path.join(HERE, "load_featengg.py"),
        arguments=["--table", featengg_table, "--output", "/opt/ml/processing/output"],
        configuration=ICEBERG_CONF,
        outputs=[ProcessingOutput(output_name="featengg", source="/opt/ml/processing/output")]))
    feat_uri = step_load.properties.ProcessingOutputConfig.Outputs["featengg"].S3Output.S3Uri

    # ── 2) train OUR forecaster ───────────────────────────────────────────────────────────
    est = SKLearn(entry_point="train.py", source_dir=HERE, dependencies=[SRC], framework_version=FW,
                  role=role, instance_type=train_type, instance_count=1, base_job_name="euler-train",
                  hyperparameters={"oem": "euler"}, sagemaker_session=session)
    step_train = TrainingStep(name="EulerTrain", estimator=est,
                              inputs={"train": TrainingInput(s3_data=feat_uri,
                                                             content_type="application/x-parquet")})
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts

    sk_proc = SKLearnProcessor(framework_version=FW, role=role, instance_type=proc_type, instance_count=1,
                               base_job_name="euler-proc", sagemaker_session=session)

    # ── 3) Evaluate -> surface evaluation.json as a standalone artifact (ModelMetrics) ────
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="EulerEvaluate", processor=sk_proc, code=os.path.join(HERE, "evaluate.py"),
        inputs=[ProcessingInput(source=model_data, destination="/opt/ml/processing/model")],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[eval_report])
    eval_uri = step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    model_metrics = ModelMetrics(model_statistics=MetricsSource(
        s3_uri=Join(on="/", values=[eval_uri, "evaluation.json"]), content_type="application/json"))

    # ── 4) Gate -> the coulomb-yardstick verdict (our euler_accept_gate) ──────────────────
    gate_report = PropertyFile(name="GateReport", output_name="gate", path="gate.json")
    step_gate = ProcessingStep(
        name="EulerGate", processor=sk_proc, code=os.path.join(HERE, "gate.py"),
        inputs=[ProcessingInput(source=feat_uri, destination="/opt/ml/processing/featengg"),
                ProcessingInput(source=yardstick_url, destination="/opt/ml/processing/yardstick"),
                ProcessingInput(source=src_code_url, destination="/opt/ml/processing/input/code")],
        outputs=[ProcessingOutput(output_name="gate", source="/opt/ml/processing/gate")],
        property_files=[gate_report])

    def _register(status, name):
        return RegisterModel(
            name=name, estimator=est, model_data=model_data,
            content_types=["application/json"], response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"], transform_instances=["ml.m5.xlarge"],
            model_package_group_name=model_package_group_name, approval_status=status,
            model_metrics=model_metrics)

    # ── 5) ConditionStep: register Approved only if the gate PASSES ───────────────────────
    cond = ConditionEquals(
        left=JsonGet(step_name=step_gate.name, property_file=gate_report, json_path="verdict"), right="PASS")
    step_cond = ConditionStep(
        name="EulerGateCondition", conditions=[cond],
        if_steps=[_register("Approved", "EulerRegisterApproved")],
        else_steps=[_register("PendingManualApproval", "EulerRegisterPending")])

    return Pipeline(name=pipeline_name,
                    parameters=[proc_type, train_type, approval, yardstick_url, src_code_url],
                    steps=[step_load, step_train, step_eval, step_gate, step_cond],
                    sagemaker_session=session)
