import re

from setuptools import setup
from setuptools_rust import Binding, RustExtension


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
_deps = [
    "black==22.3",  # after updating to black 2023, also update Python version in pyproject.toml to 3.7
    "click==8.0.4",
    "flake8>=3.8.3",
    "flax>=0.6.3",
    "h5py>=3.7.0",
    "huggingface_hub>=0.12.1",
    "isort>=5.5.4",
    "jax>=0.4.1",
    "numpy>=1.24.2",
    "setuptools_rust>=1.5.2",
    "pytest>=7.2.0",
    "pytest-benchmark>=4.0.0",
    "tensorflow>=2.12.0",
    "torch>=1.10",
    "paddlepaddle>=2.4.1",
]


deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["torch"] = deps_list("torch")
extras["numpy"] = deps_list("numpy")
extras["tensorflow"] = deps_list("tensorflow")
extras["jax"] = deps_list("jax", "flax")
extras["paddlepaddle"] = deps_list("paddlepaddle")
extras["quality"] = deps_list("black", "isort", "flake8", "click")
extras["testing"] = (
    deps_list("setuptools_rust", "huggingface_hub", "pytest", "pytest-benchmark", "h5py") + extras["numpy"]
)
extras["all"] = (
    extras["torch"]
    + extras["numpy"]
    + extras["tensorflow"]
    + extras["jax"]
    + extras["paddlepaddle"]
    + extras["quality"]
    + extras["testing"]
)
extras["dev"] = extras["all"]

with open("py_src/safetensors/__init__.py", "r") as f:
    version = f.readline().split("=")[-1].strip().strip('"')

setup(
    name="safetensors",
    version=version,
    description="Fast and Safe Tensor serialization",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",
    author="",
    author_email="",
    url="https://github.com/huggingface/safetensors",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("safetensors._safetensors_rust", binding=Binding.PyO3, debug=False)],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "py_src"},
    packages=[
        "safetensors",
    ],
    package_data={},
    zip_safe=False,
)
