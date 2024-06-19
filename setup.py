import os
from os import path
from setuptools import setup, find_packages

dir = path.abspath(path.dirname(__file__))
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(path.join(dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    requirements = []


def set_version():
    head = open(path.join(dir, ".git", "HEAD")).readline()
    if head.find("dev") != -1:
        return {
            "template": "{tag}.dev{ccount}",
            "dev_template": "{tag}.dev{ccount}",
            "dirty_template": "{tag}.dev{ccount}",
        }
    else:
        return {"template": "{tag}", "dev_template": "{tag}", "dirty_template": "{tag}"}


setup(
    name="scFates",
    version_config=set_version(),
    setup_requires=["setuptools-git-versioning>=2.0,<3"],
    setuptools_git_versioning={
        "enabled": True,
    },
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
    package_data={"": ["datasets/*.h5ad", "*.R"]},
    zip_safe=False,
)
