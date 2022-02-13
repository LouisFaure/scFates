.. module:: scFates
.. automodule:: scFates
   :noindex:


API
===

Import scFates as::

   import scFates as scf

Some convenient preprocessing functions translated from pagoda2 have been included:

Pre-processing
--------------

.. autosummary::
    :toctree: .

    pp.filter_cells
    pp.batch_correct
    pp.find_overdispersed
    pp.diffusion


Tree inference
--------------

.. autosummary::
    :toctree: .

    tl.tree
    tl.curve
    tl.circle
    tl.cellrank_to_tree

Tree operations
---------------

.. autosummary::
    :toctree: .

    tl.cleanup
    tl.subset_tree
    tl.attach_tree
    tl.extend_tips
    tl.simplify

Pseudotime analysis
-------------------

.. autosummary::
    :toctree: .

    tl.root
    tl.roots
    tl.pseudotime
    tl.dendrogram
    tl.test_association
    tl.test_association_covariate
    tl.fit
    tl.cluster
    tl.test_covariate
    tl.linearity_deviation
    tl.get_path

Bifurcation analysis
--------------------

**Branch specific feature extraction and classification**

.. autosummary::
    :toctree: .

    tl.test_fork
    tl.branch_specific
    tl.activation
    tl.activation_lm

**Correlation analysis**

.. autosummary::
    :toctree: .

    tl.module_inclusion
    tl.slide_cells
    tl.slide_cors
    tl.synchro_path
    tl.synchro_path_multi

Plot
----

**Trajectory**

.. autosummary::
    :toctree: .

    pl.graph
    pl.trajectory
    pl.trajectory_3d
    pl.dendrogram
    pl.milestones

**Pseudotime features**

.. autosummary::
    :toctree: .

    pl.test_association
    pl.single_trend
    pl.trends
    pl.matrix
    pl.linearity_deviation
    pl.binned_pseudotime_meta

**Bifurcation & correlation analysis**

.. autosummary::
    :toctree: .

    pl.modules
    pl.test_fork
    pl.slide_cors
    pl.synchro_path
    pl.module_inclusion


Getting analysed data
---------------------

.. autosummary::
    :toctree: .

    get.fork_stats
    get.modules
    get.slide_cors
