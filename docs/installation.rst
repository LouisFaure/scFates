Installation
============

scFates has been tested with python 3.7, it is recommended to use a Miniconda_ environment.

PyPI
----

Currently, scFates can only be installed from GitHub_ using:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=pip%2520install%2520git%252Bhttps%253A%252F%252Fgithub.com%252FLouisFaure%252FscFates"
  frameborder="0" scrolling="no" style="width: 100%; height:150px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>  

    
R dependencies
--------------

scFates rely on the R package *mgcv* to perform testing and fitting of the features on the peudotime
tree. Package is installed in an R session with the following command::

    install.packages('mgcv')

GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, scFates can leverage CUDA computations for speedups in some functions, 
the latest version of cupy is required (at least 8.0, modify the cuda version to match yours), as well as grapheno for feature clustering:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=pip%2520install%2520cupy-cuda11%2520grapheno"
  frameborder="0" scrolling="no" style="width: 100%; height:150px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>  

Docker container
----------------

scFates can be run on a `Docker container`_ based on Rapids container, which provides a gpu enabled environment with Jupyter Lab. Use the following command:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=docker%2520run%2520--rm%2520-it%2520--gpus%2520all%2520-p%25208888%253A8888%2520-p%25208787%253A8787%2520-p%25208786%253A8786%2520%255C%250A%2520%2520%2520%2520louisfaure%252Fscfates%253Atagname"
  frameborder="0" scrolling="no" style="width: 100%; height:250px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>    
        
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
.. _`Docker container`: https://hub.docker.com/repository/docker/louisfaure/scfates