import os
from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    requirements = []

setup(
    name="scFates",
    version_format="{tag}",
    setup_requires=["setuptools-git-version"],
    description="scanpy compatible python suite for fast tree inference and advanced pseudotime downstream analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisFaure/scFates",
    author="Louis Faure",
    author_email="",
    packages=find_packages(),
    package_dir={"scFates": "scFates"},
    install_requires=requirements,
    include_package_data=True,
    package_data={"": ["datasets/*.h5ad"]},
    zip_safe=False,
)
