"""Euler SoH-forecaster model-build pipeline — END TO END, our euler_model.

Preprocessing now lives in the Glue job (`MLOps/glue/euler_featengg_incremental.py` -> the `euler_featengg`
feature store), so this pipeline is the CONSUMING side:

    LoadFeatengg (unload euler_featengg -> parquet)  ->  Train (our rate + trajectory forecaster + LOVO)
      ->  RegisterModel

"All logic is ours": `train.py` calls `src/euler_model` / `euler_backtest` / `euler_train` (shipped via the
estimator's `dependencies=[src]`); the point-in-time cohort selection + train/val/test split happen there.
Euler is the gated OEM — the coulomb acceptance gate (`gate.py`) is wired as an OPTIONAL ConditionStep (it
needs the independent coulomb full-charge yardstick as a separate input; off by default).
"""
import os

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.processing import ProcessingOutput
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../pipelines/euler
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))  # repo src (our real modules)
FRAMEWORK_VERSION = "1.2-1"
ICEBERG_CONF = [
    {"Classification": "spark-defaults", "Properties": {
        "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        "spark.sql.catalog.glue_catalog": "org.apache.iceberg.spark.SparkCatalog",
        "spark.sql.catalog.glue_catalog.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
        "spark.sql.catalog.glue_catalog.io-impl": "org.apache.iceberg.aws.s3.S3FileIO"}}]


def get_pipeline(region=None, role=None, default_bucket=None,
                 pipeline_name="soh-euler-build", model_package_group_name="soh-forecaster-euler",
                 featengg_table="glue_catalog.turno_ml.euler_featengg", use_hybrid_label=False, **kwargs):
    session = PipelineSession(default_bucket=default_bucket)
    role = role or sagemaker.get_execution_role()

    proc_type = ParameterString("ProcessingInstanceType", default_value="ml.m5.xlarge")
    train_type = ParameterString("TrainingInstanceType", default_value="ml.m5.2xlarge")
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    # 1) unload the euler_featengg Iceberg feature store -> parquet (the training channel)
    spark_proc = PySparkProcessor(framework_version="3.3", role=role, instance_type=proc_type,
                                  instance_count=1, base_job_name="euler-load-featengg",
                                  sagemaker_session=session)
    step_load = ProcessingStep(
        name="EulerLoadFeatengg",
        step_args=spark_proc.run(
            submit_app=os.path.join(HERE, "load_featengg.py"),
            arguments=["--table", featengg_table, "--output", "/opt/ml/processing/output"],
            configuration=ICEBERG_CONF,
            outputs=[ProcessingOutput(output_name="featengg", source="/opt/ml/processing/output")]))

    # 2) train OUR forecaster (rate + trajectory + LOVO band recalibration + stratified diagnostics)
    est = SKLearn(entry_point="train.py", source_dir=HERE, dependencies=[SRC],
                  framework_version=FRAMEWORK_VERSION, role=role, instance_type=train_type, instance_count=1,
                  base_job_name="euler-train",
                  hyperparameters={"oem": "euler", **({"label": ""} if use_hybrid_label else {})},
                  sagemaker_session=session)
    feat_uri = step_load.properties.ProcessingOutputConfig.Outputs["featengg"].S3Output.S3Uri
    step_train = TrainingStep(
        name="EulerTrain", estimator=est,
        inputs={"train": TrainingInput(s3_data=feat_uri, content_type="application/x-parquet")})

    # 3) register the trained bundle (evaluation.json travels inside model.tar.gz; a production setup adds a
    #    separate Evaluate step emitting it as a ModelMetrics source, and the gate.py ConditionStep for Euler)
    step_register = RegisterModel(
        name="EulerRegister", estimator=est,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"], response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"], transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name, approval_status=approval)

    return Pipeline(name=pipeline_name, parameters=[proc_type, train_type, approval],
                    steps=[step_load, step_train, step_register], sagemaker_session=session)
