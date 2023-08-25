Installation
============

scFates is continuously tested with python 3.7 and 3.8, it is recommended to use a Miniconda_ environment.

PyPI
----

scFates is available on pypi, you can install it using::

    pip install scFates

or the latest development version can be installed from GitHub_ using::

    pip install git+https://github.com/LouisFaure/scFates


With all dependencies
---------------------

- :func:`scFates.pp.find_overdispersed`, :func:`scFates.tl.test_association`, :func:`scFates.tl.fit`, :func:`scFates.tl.test_fork`, :func:`scFates.tl.activation`: Require R package mgcv interfaced via python package rpy2::

    conda create -n scFates -c conda-forge -c r python=3.8 r-mgcv rpy2 -y
    conda activate scFates
    pip install scFates

to avoid any possible crashes due to rpy2 not finding the R install on conda, run the following import command:

.. code-block:: python

    import os, sys
    os.environ['R_HOME'] = sys.exec_prefix+"/lib/R/"
    import scFates

- :func:`scFates.tl.cellrank_to_tree`: Requires cellrank to be installed in order to function::

    pip install cellrank



On Apple Silicon
----------------

Installing mgcv using conda/mamba on Apple Silicon lead to the package not being able to find some dynamic libraries (BLAS).
In that case it is recommended to install it separately:

.. code-block:: bash

    mamba create -n scFates -c conda-forge -c bioconda -c defaults python numpy=1.24.4 "libblas=*=*accelerate" rpy2 -y
    mamba activate scFates
    Rscript -e 'install.packages("mgcv",repos = "http://cran.us.r-project.org")'


GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups for the following functions:

:func:`scFates.pp.filter_cells`, :func:`scFates.pp.batch_correct`, :func:`scFates.pp.diffusion`, :func:`scFates.tl.tree`, :func:`scFates.tl.cluster`

The latest version of rapids framework is required (at least 0.17) it is recommanded to create a new conda environment::

    conda create -n scFates-gpu -c rapidsai -c nvidia -c conda-forge -c defaults cuml=21.12 cugraph=21.12 python=3.8 cudatoolkit=11.0 -y
    conda activate scFates-gpu
    pip install git+https://github.com/j-bac/elpigraph-python.git
    pip install scFates


.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
