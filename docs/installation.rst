Installation
============

scFates has been tested with python 3.7, it is recommended to use a Miniconda_ environment.

PyPI
----

Currently, scFates can only be installed from GitHub_ using::

    pip install git+https://github.com/LouisFaure/scFates

or::

    git clone https://github.com/LouisFaure/scFates
    pip install -e scFates
    
    
R dependencies
--------------

scFates rely on the R package *mgcv* to perform testing and fitting of the features on the peudotime
tree. Package is installed in an R session with the following command::

    install.packages('mgcv')

GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups in some functions, 
the latest version of cupy is required (at least 8.0)::

    pip install cupy-cuda11

Modify the cuda version to match yours, for feature clustering using GPU, the following dependency is required::

    pip install grapheno


Docker container
----------------

scFates can be run on a `Docker container`_ based on Rapids container, which provide a gpu enabled environment with Jupyter Lab. Use the following command::

    docker run --rm -it --gpus all -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        louisfaure/scfates:tagname        
        
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
.. _`Docker container`: https://hub.docker.com/repository/docker/louisfaure/scfates