"""Packaging for the SoH/RUL model-build pipelines."""
import setuptools

about = {}
with open("pipelines/__version__.py") as f:
    exec(f.read(), about)

with open("README.md") as f:
    readme = f.read()

setuptools.setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "sagemaker>=2.150.0",
        "boto3>=1.34.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
    ],
    extras_require={"test": ["pytest", "pytest-cov", "black", "flake8"]},
    entry_points={
        "console_scripts": [
            "get-pipeline-definition=pipelines.get_pipeline_definition:main",
            "run-pipeline=pipelines.run_pipeline:main",
        ]
    },
)
