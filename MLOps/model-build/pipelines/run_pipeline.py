"""Upsert a pipeline and start an execution (used by the CodeBuild buildspec).

    python pipelines/run_pipeline.py \\
        --module-name pipelines.euler.pipeline \\
        --role-arn <sagemaker-execution-role> \\
        --tags '[{"Key":"oem","Value":"euler"}]' \\
        --kwargs '{"region":"ap-south-1","default_bucket":"oem-data-iot"}'
"""
import argparse
import json

from pipelines._utils import get_pipeline_driver, get_pipeline_custom_tags


def main():
    p = argparse.ArgumentParser("Creates/updates and starts a pipeline for the given module.")
    p.add_argument("-n", "--module-name", required=True)
    p.add_argument("-role-arn", "--role-arn", required=True)
    p.add_argument("-tags", "--tags", default=None)
    p.add_argument("-kwargs", "--kwargs", default=None)
    p.add_argument("-d", "--description", default=None)
    args = p.parse_args()

    tags = json.loads(args.tags) if args.tags else []
    pipeline = get_pipeline_driver(args.module_name, args.kwargs)

    print("###### Upserting pipeline with definition:")
    print(pipeline.definition())
    custom_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)
    pipeline.upsert(role_arn=args.role_arn, description=args.description, tags=custom_tags)

    execution = pipeline.start()
    print(f"###### Started execution: {execution.arn}")
    execution.wait()
    print("###### Execution completed. Step results:")
    print(execution.list_steps())


if __name__ == "__main__":
    main()
