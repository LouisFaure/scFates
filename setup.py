import os
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
try:
    from pypandoc import convert

    read_md = lambda f: convert(f, "rst")
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, "r").read()

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    requirements = []

setup(
    name="scFates",
    version="0.2.0",
    description="scanpy compatible python suite for fast tree inference and advanced pseudotime downstream analysis",
    long_description=read_md("README.md"),
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
