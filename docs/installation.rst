Installation
============

scFates is continuously tested with python 3.11, it is recommended to use a Miniconda_ environment.

PyPI
----

scFates is available on pypi, you can install it using::

    pip install scFates

or the latest development version can be installed from GitHub_ using::

    pip install git+https://github.com/LouisFaure/scFates


With all dependencies
---------------------

- :func:`scFates.pp.find_overdispersed`, :func:`scFates.tl.test_association`, :func:`scFates.tl.fit`, :func:`scFates.tl.test_fork`, :func:`scFates.tl.activation`: Require R package mgcv interfaced via python package rpy2::

    conda create -n scFates -c conda-forge -c r python=3.11 r-mgcv rpy2 -y
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

With an NVIDIA GPU, the following functions can run on the GPU by passing ``device="gpu"``:

:func:`scFates.pp.filter_cells`, :func:`scFates.pp.batch_correct`, :func:`scFates.pp.diffusion`, :func:`scFates.tl.tree`, :func:`scFates.tl.curve`, :func:`scFates.tl.circle`, :func:`scFates.tl.cluster`

This requires `rapids-singlecell <https://rapids-singlecell.readthedocs.io>`_ together with CuPy and the RAPIDS libraries (cuDF, cuML, cuGraph), built for your CUDA version. With CUDA 12, its ``rapids`` extra installs all of them::

    pip install --extra-index-url=https://pypi.nvidia.com scFates "rapids-singlecell-cu12[rapids]"

See the `rapids-singlecell installation guide <https://rapids-singlecell.readthedocs.io/en/latest/installation.html>`_ for conda environments and other CUDA versions.


.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
