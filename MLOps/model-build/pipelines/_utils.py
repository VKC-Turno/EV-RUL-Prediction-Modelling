"""Small helpers to load a pipeline module dynamically (mirrors the AWS MLOps template _utils)."""
import ast
import importlib


def get_pipeline_driver(module_name, passed_args=None):
    """Import `module_name` (e.g. pipelines.euler.pipeline) and call its get_pipeline(**kwargs)."""
    _imports = importlib.import_module(module_name)
    kwargs = convert_struct(passed_args)
    return _imports.get_pipeline(**kwargs)


def convert_struct(args_str=None):
    return ast.literal_eval(args_str) if args_str else {}


def get_pipeline_custom_tags(module_name, args, tags):
    """Hook to append project/OEM tags (SageMaker Projects fills these in). No-op by default."""
    return tags
