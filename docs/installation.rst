Installation
============

scFates is continuously tested with python 3.7 and 3.8, it is recommended to use a Miniconda_ environment.

PyPI
----

scFates is available on pypi, you can install it using::

    pip install scFates

or the latest development version can be installed from GitHub_ using::

    pip install git+https://github.com/LouisFaure/scFates


Python dependencies
------------------

scFates gives the choice of between SimplePPT and ElPiGraph for learning a principal graph from the data.
Elpigraph needs to be installed from its github repository with the following command::

    pip install git+https://github.com/j-bac/elpigraph-python.git


R dependencies
--------------

scFates rely on the R package *mgcv* to perform testing and fitting of the features on the peudotime
tree. Package is installed in an R session with the following command::

    install.packages('mgcv')

GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups in some functions,
the latest version of rapids framework is required (at least 0.17) it is recommanded to create a new conda environment::

    conda create -n scFates-gpu -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.19 python=3.8 cudatoolkit=11.0 -y
    conda activate scFates-gpu
    pip install git+https://github.com/j-bac/elpigraph-python.git
    pip install scFates --ignore-installed

Docker container
----------------

scFates can be run on a `Docker container`_ based on Rapids container, which provides a gpu enabled environment with Jupyter Lab. Use the following command::

    docker run --rm -it --gpus all -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        louisfaure/scfates:latest

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
.. _`Docker container`: https://hub.docker.com/repository/docker/louisfaure/scfates
