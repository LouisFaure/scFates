[![Documentation Status](https://readthedocs.org/projects/scfates/badge/?version=latest)](https://scfates.readthedocs.io/en/latest/?badge=latest)

Description
===========

This package provides an Python tool suite for fast tree inference and advanced pseudotime downstream analysis, with a focus on fate biasing. This package is compatible with anndata object format used in scanpy or scvelo pipelines. A complete documentation of this package is available [here](https://scfates.readthedocs.io/en/latest).

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

Currently, scFates can only be installed from GitHub_ using::

    pip install git+https://github.com/LouisFaure/scFates

or::

    git clone https://github.com/LouisFaure/scFates
    pip install -e scFates
    
    
## R dependencies

scFates rely on the R package *mgcv* to perform testing and fitting of the features on the peudotime
tree. Package is installed in an R session with the following command::

    install.packages('mgcv')

## GPU dependencies (optional)

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups in some functions, 
the following dependencies are required::

    pip install cupy cudf grapheno
    
    
## Docker container

scFates can be run on a [Docker container](https://hub.docker.com/repository/docker/louisfaure/scfates) based on Rapids container, which provide a gpu enabled environment with Jupyter Lab. Use the following command::

    docker run --rm -it --gpus all -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        louisfaure/scfates:tagname        
