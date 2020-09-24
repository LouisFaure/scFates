[![Documentation Status](https://readthedocs.org/projects/scfates/badge/?version=latest)](https://scfates.readthedocs.io/en/latest/?badge=latest)

Description
===========

This package provides an Python tool suite for fast tree inference and advanced pseudotime downstream analysis, with a focus on fate biasing. This package is compatible with anndata object format generated via scanpy or scvelo pipelines.

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

## PPT

A [simple PPT](https://www.acsu.buffalo.edu/~yijunsun/lab/Paper/simplePPT.pdf) inspired approach, translated from the [crestree R package](https://github.com/hms-dbmi/crestree), code has been also adapted to run on GPU for accelerated tree inference.

Citations
=========

Code for PPT inference and most of downstream pseudotime analysis was initially written in a [R package](https://github.com/hms-dbmi/crestree) by Ruslan Soldatov for the following paper:

Soldatov, R., Kaucka, M., Kastriti, M. E., Petersen, J., Chontorotzea, T., Englmaier, L., … Adameyko, I. (2019). Spatiotemporal structure of cell fate decisions in murine neural crest. Science, 364(6444).

if you are using ElPiGraph, please cite :

Albergante, L., Mirkes, E. M., Chen, H., Martin, A., Faure, L., Barillot, E., … Zinovyev, A. (2020). Robust And Scalable Learning Of Complex Dataset Topologies Via Elpigraph. Entropy, 22(3), 296.


Requirements
============

This code was tested with Python 3.6, elpigraph needs to be installed via its github repo:

```bash
pip install git+https://github.com/j-bac/elpigraph-python.git
```

if you have a CUDA ready GPU you have to install:
```bash
pip install cupy==8.0.0b3
```

Installation & Usage
====================

To install that package, clone this git, open a terminal on the root of the git folder and type:
```bash
pip install .
```

Or, without cloning, simply run the following command
```bash
pip install git+https://github.com/LouisFaure/scTree.git
```

Example of usage is [described here](documentation/Basic_usage_example_ElPiGraph.ipynb)
