|PyPi| |Doc Badge| |Build Badge| |codecov Badge| |License| |black|

.. |PyPi| image:: https://img.shields.io/pypi/v/scFates.svg
    :target: https://pypi.python.org/pypi/scFates/
.. |Doc Badge| image:: https://readthedocs.org/projects/scfates/badge/?version=latest
    :target: https://scfates.readthedocs.io/en/latest/
.. |Build Badge| image:: https://github.com/LouisFaure/scFates/actions/workflows/load_test_upload.yml/badge.svg
    :target: https://github.com/LouisFaure/scFates/actions/workflows/load_test_upload.yml
.. |codecov Badge| image:: https://codecov.io/gh/LouisFaure/scFates/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/LouisFaure/scFates
.. |License| image:: https://img.shields.io/github/license/LouisFaure/scFates
    :target: https://github.com/LouisFaure/scFates/blob/master/LICENSE
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


scFates - a python package for advanced pseudotime and bifurcation analysis
===========================================================================

**scFates** is a scalable Python suite for tree inference and advanced pseudotime analysis.



The related work is now available in Bioinformatics_::

    Louis Faure, Ruslan Soldatov, Peter V. Kharchenko, Igor Adameyko
    scFates: a scalable python package for advanced pseudotime and bifurcation analysis from single cell data
    Bioinformatics, btac746; doi: https://doi.org/10.1093/bioinformatics/btac746


It initially is a translation from crestree_, a R package developed for the analysis of neural crest fates during
murine embryonic development (`Soldatov et al., Science, 2019 <https://doi.org/10.1126/science.aas9536>`_), and used
in another study of neural crest derived sensory neurons (`Faure et al., Nature Communications, 2020 <https://doi.org/10.1038/s41467-020-17929-4>`_).

The initial R version included a tree inference approach inspired from SimplePPT, this version now adds the choice of using ElPiGraph_,
another method for principal graph learning (`Albergante et al., Entropy, 2020 <https://doi.org/10.3390/e22030296>`_).
scFates is fully compatible with scanpy_, and contains GPU-accelerated functions for faster and scalable inference.


Analysis key steps
^^^^^^^^^^^^^^^^^^
- learn a principal tree on the space of your choice (gene expression, PCA, diffusion, ...).
- obtain a pseudotime value upon selection of a root (or two roots).
- test and fit features significantly changing along the tree, cluster them.
- perform differential expression between branches.
- identify branch-specific early activating features and probe their correlations prior to bifurcation.

.. toctree::
   :caption: Main
   :maxdepth: 2
   :hidden:

   installation
   api
   release_notes
   references

.. toctree::
   :caption: Walkthroughs
   :maxdepth: 2
   :hidden:

   Basic_Curved_trajectory_analysis
   Tree_Analysis_Bone_marrow_fates
   Tree_operations
   Explore_sigma
   Conversion_from_CellRank_pipeline
   Beyond_scRNAseq

.. _scanpy: https://scanpy.readthedocs.io/
.. _crestree: https://github.com/hms-dbmi/crestree
.. _ElPiGraph: https://github.com/j-bac/elpigraph-python/
.. _Bioinformatics: https://doi.org/10.1093/bioinformatics/btac746
