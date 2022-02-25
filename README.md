[![PyPI](https://img.shields.io/pypi/v/scFates.svg)](https://pypi.python.org/pypi/scFates/)
[![Documentation Status](https://readthedocs.org/projects/scfates/badge/?version=latest)](https://scfates.readthedocs.io/en/latest/?badge=latest)
[![Build and Test](https://github.com/LouisFaure/scFates/actions/workflows/load_test_upload.yml/badge.svg)](https://github.com/LouisFaure/scFates/actions/workflows/load_test_upload.yml)
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

For scFates, the [python](https://github.com/j-bac/elpigraph-python/) implementation of the ElPiGraph algorithm is used, which include GPU accelerated principal tree inference. A self-contained description of the algorithm is available [here](https://github.com/auranic/Elastic-principal-graphs/blob/master/ElPiGraph_Methods.pdf) or in the related [paper](https://www.mdpi.com/1099-4300/22/3/296)

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

    Soldatov, R., Kaucka, M., Kastriti, M. E., Petersen, J., Chontorotzea, T., Englmaier, L., … Adameyko, I. (2019).
    Spatiotemporal structure of cell fate decisions in murine neural crest.
    Science, 364(6444).

if you are using ElPiGraph, please cite:

    Albergante, L., Mirkes, E. M., Chen, H., Martin, A., Faure, L., Barillot, E., … Zinovyev, A. (2020).
    Robust And Scalable Learning Of Complex Dataset Topologies Via Elpigraph.
    Entropy, 22(3), 296.

Code for preprocessing has been translated from R package pagoda2, if you use any of these functions (`scf.pp.batch_correct` & `scf.pp.find_overdispersed`), please cite:

    Nikolas Barkas, Viktor Petukhov, Peter Kharchenko and Evan
    Biederstedt (2021). pagoda2: Single Cell Analysis and Differential
    Expression. R package version 1.0.2.

Palantir python tool provides a great dimensionality reduction method, which usually lead to consitent trees with scFates, if use `scf.pp.diffusion`, please cite:

    Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er (2019)
    Characterization of cell fate probabilities in single-cell data with Palantir.
    Nature Biotechnology

Installation
============

scFates is available on pypi, you can install it using:

    pip install -U scFates

or the latest development version can be installed from GitHub:

    pip install git+https://github.com/LouisFaure/scFates

Dependencies
------------

scFates installed via pip gives a base package that can perform tree fitting on adata objects, here is a list of dependencies needed for each related functions:

-`tl.tree`: ElPiGraph can be also used for learning a principal graph from the data (`method="epg"`). Elpigraph can be installed from its github repository with the following command:

    pip install git+https://github.com/j-bac/elpigraph-python.git

-`tl.cellrank_to_tree`: Requires cellrank to be installed in order to function::

    pip install cellrank

-`pp.find_overdispersed`, `tl.test_association`, `tl.fit`, `tl.test_fork`, `tl.activation`, `tl.test_association_covariate`, `tl.test_covariate`: Require R package mgcv interfaced via python package rpy2::

    pip install rpy2

    install.packages('mgcv') # run in R session


GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups for the following functions:

`pp.filter_cells`, `pp.batch_correct`, `pp.diffusion`, `tl.tree`, `tl.cluster`

The latest version of rapids framework is required (at least 0.17) it is recommanded to create a new conda environment:

    conda create -n scFates-gpu -c rapidsai -c nvidia -c conda-forge -c defaults \
        cuml=21.12 cugraph=21.12 python=3.8 cudatoolkit=11.0 -y
    conda activate scFates-gpu
    pip install git+https://github.com/j-bac/elpigraph-python.git
    pip install scFates

## Docker container

scFates can be run on a [Docker container](https://hub.docker.com/repository/docker/louisfaure/scfates) based on Rapids 0.18 container,
which provide a gpu enabled environment with Jupyter Lab. Use the following command:

    docker run --rm -it --gpus all -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        louisfaure/scfates:latest
