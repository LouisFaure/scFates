Installation
============

scFates has been tested with python 3.7, it is recommended to use a Miniconda_ environment.

PyPI
----

scFates 0.2 is now available on pypi, you can install it using:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=pip%2520install%2520scFates"
  frameborder="0" scrolling="no" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px';" style="width: 100%; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>


or the latest development version can be installed from GitHub_ using:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=pip%2520install%2520git%252Bhttps%253A%252F%252Fgithub.com%252FLouisFaure%252FscFates"
  frameborder="0" scrolling="no" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px';" style="width: 100%; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>


Python dependencies
--------------

scFates gives the choice of between SimplePPT and ElPiGraph for learning a principal graph from the data.
Elpigraph needs to be installed from its github repository with the following command:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=pip%2520install%2520git%252Bhttps%253A%252F%252Fgithub.com%252Fj-bac%252Felpigraph-python.git"
  frameborder="0" scrolling="no" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px';" style="width: 100%; border:0; transform: scale(1); overflow:hidden;"
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
the latest version of rapids framework is required (at least 8.17) it is recommanded to create a new conda environment:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=conda%2520create%2520-n%2520rapids-0.17%2520-c%2520rapidsai%2520-c%2520nvidia%2520-c%2520conda-forge%2520-c%2520defaults%2520rapids%253D0.17%2520python%253D3.7%2520cudatoolkit%253D11.0"
  frameborder="0" scrolling="no" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px';" style="width: 100%; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>



Docker container
----------------

scFates can be run on a `Docker container`_ based on Rapids container, which provides a gpu enabled environment with Jupyter Lab. Use the following command:

.. raw:: html

    <iframe
  src="_static/carbon.now.sh/index.html?bg=rgba%28252%2C252%2C252%2C1%29&t=seti&wt=none&l=application%2Fx-sh&ds=true&dsyoff=5px&dsblur=12px&wc=true&wa=true&pv=20px&ph=56px&ln=false&fl=1&fm=Hack&fs=15px&lh=133%25&si=false&es=2x&wm=false&code=docker%2520run%2520--rm%2520-it%2520--gpus%2520all%2520-p%25208888%253A8888%2520-p%25208787%253A8787%2520-p%25208786%253A8786%2520%255C%250A%2520%2520%2520%2520louisfaure%252Fscfates%253Aversion-0.2"
  frameborder="0" scrolling="no" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px';" style="width: 100%; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
  </iframe>

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Github: https://github.com/LouisFaure/scFates
.. _`Docker container`: https://hub.docker.com/repository/docker/louisfaure/scfates
