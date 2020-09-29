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
the following dependencies are required::

    pip install cupy cudf grapheno



.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
