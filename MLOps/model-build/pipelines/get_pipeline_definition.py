"""Print a pipeline's definition JSON (used by CI to inspect/lint before upsert).

    python pipelines/get_pipeline_definition.py \\
        --module-name pipelines.euler.pipeline \\
        --kwargs '{"region":"ap-south-1","role":"<role-arn>","default_bucket":"oem-data-iot"}'
"""
import argparse

from pipelines._utils import get_pipeline_driver


def main():
    p = argparse.ArgumentParser("Gets the pipeline definition for the given module.")
    p.add_argument("-n", "--module-name", required=True,
                   help="e.g. pipelines.euler.pipeline / pipelines.mahindra.pipeline")
    p.add_argument("-kwargs", "--kwargs", default=None,
                   help="dict string of keyword args for get_pipeline")
    args = p.parse_args()
    pipeline = get_pipeline_driver(args.module_name, args.kwargs)
    print(pipeline.definition())


if __name__ == "__main__":
    main()
