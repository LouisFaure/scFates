[![PyPI](https://img.shields.io/pypi/v/scFates.svg)](https://pypi.python.org/pypi/scFates/)
[![Documentation Status](https://readthedocs.org/projects/scfates/badge/?version=latest)](https://scfates.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/LouisFaure/scFates.svg?style=shield)](https://circleci.com/gh/LouisFaure/scFates)
[![TravisCI](https://api.travis-ci.com/LouisFaure/scFates.svg?branch=master)](https://travis-ci.com/github/LouisFaure/scFates)
[![codecov](https://codecov.io/gh/LouisFaure/scFates/branch/master/graph/badge.svg)](https://codecov.io/gh/LouisFaure/scFates)
[![GitHub license](https://img.shields.io/github/license/LouisFaure/scFates)](https://github.com/LouisFaure/scFates/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Description
===========

This package provides a scalable Python suite for fast tree inference and advanced pseudotime downstream analysis, with a focus on fate biasing. This package is compatible with anndata object format used in scanpy or scvelo pipelines. A complete documentation of this package is available [here](https://scfates.readthedocs.io/en/latest).

Tree inference algorithms
=========================

The user have the choice between two algorithm for tree inference:

## ElPiGraph

For scTree, the [python](https://github.com/j-bac/elpigraph-python/) implementation of the ElPiGraph algorithm is used, which include GPU accelerated principal tree inference. A self-contained description of the algorithm is available [here](https://github.com/auranic/Elastic-principal-graphs/blob/master/ElPiGraph_Methods.pdf) or in the related [paper](https://www.mdpi.com/1099-4300/22/3/296)

A [R implementation](https://github.com/Albluca/ElPiGraph.R) of this algorithm is also available, coded by [Luca Albergante](https://github.com/Albluca)

A native MATLAB implementation of the algorithm (coded by [Andrei
Zinovyev](https://github.com/auranic/) and [Evgeny
Mirkes](https://github.com/Mirkes)) is also
[available](https://github.com/auranic/Elastic-principal-graphs)

## Simple PPT

A [simple PPT](https://www.acsu.buffalo.edu/~yijunsun/lab/Paper/simplePPT.pdf) inspired approach, translated from the [crestree R package](https://github.com/hms-dbmi/crestree), code has been also adapted to run on GPU for accelerated tree inference.

Citations
=========

Code for PPT inference and most of downstream pseudotime analysis was initially written in a [R package](https://github.com/hms-dbmi/crestree) by Ruslan Soldatov for the following paper:

Soldatov, R., Kaucka, M., Kastriti, M. E., Petersen, J., Chontorotzea, T., Englmaier, L., … Adameyko, I. (2019). Spatiotemporal structure of cell fate decisions in murine neural crest. Science, 364(6444).

if you are using ElPiGraph, please cite :

Albergante, L., Mirkes, E. M., Chen, H., Martin, A., Faure, L., Barillot, E., … Zinovyev, A. (2020). Robust And Scalable Learning Of Complex Dataset Topologies Via Elpigraph. Entropy, 22(3), 296.


Installation
============

scFates 0.2 is now available on pypi, you can install it using:

    pip install scFates

or the latest development version can be installed from GitHub:

    pip install git+https://github.com/LouisFaure/scFates

## Python dependencies

scFates gives the choice of between SimplePPT and ElPiGraph for learning a principal graph from the data.
Elpigraph needs to be installed from its github repository with the following command:

	pip install git+https://github.com/j-bac/elpigraph-python.git

## R dependencies

scFates rely on the R package *mgcv* to perform testing and fitting of the features on the peudotime
tree. Package is installed in an R session with the following command:

    install.packages('mgcv')

## GPU dependencies (optional)

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups in some functions, for that you will need
[Rapids 0.17](https://rapids.ai/) installed.

## Docker container

scFates can be run on a [Docker container](https://hub.docker.com/repository/docker/louisfaure/scfates) based on Rapids 0.17 container,
which provide a gpu enabled environment with Jupyter Lab. Use the following command:

    docker run --rm -it --gpus all -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        louisfaure/scfates:version-0.2
