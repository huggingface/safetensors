from setuptools import setup
from setuptools_rust import Binding, RustExtension


with open("requirements-dev.txt", "r") as f:
    testing = f.read().splitlines()

extras = {}
extras["testing"] = testing

setup(
    name="safetensors",
    version="0.2.3.dev0",
    description="Fast and Safe Tensor serialization",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",
    author="",
    author_email="",
    url="https://github.com/huggingface/safetensors",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("safetensors.safetensors_rust", binding=Binding.PyO3, debug=False)],
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
