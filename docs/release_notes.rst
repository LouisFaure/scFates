.. role:: small
.. role:: smaller

Release Notes
=============

Version 0.2.7 :small:`September 23, 2021`
-------------------------------------

.. rubric:: Additions

- :func:`scFates.tl.circle`, to fit a principal circle on high dimensions!
- :func:`scFates.tl.dendrogram` and `pl.dendrogram`, for generating and plotting a dendrogram URD style single-cell embedding for better interpretability
- :func:`scFates.tl.extend_tips` (replaces `tl.refine_pseudotime` ) to avoid the compression of cells at the tips.
- :func:`scFates.pl.binned_pseudotime_meta`, a dotplot showing the proportion of cells for a given category, along binned pseudotime intervals.

.. rubric:: New walkthroughs

- `Tree operation walkthrough <https://scfates.readthedocs.io/en/latest/Tree_operations.html>`_, for tree subsetting, attachment and extension.
- `Basic trajectory walkthrough <https://scfates.readthedocs.io/en/latest/Basic_pseudotime_analysis.html>`_, for simple developmental transition.
- `Going beyond scRNAseq <https://scfates.readthedocs.io/en/latest/Beyond_scRNAseq.html>`_, one can also apply scFates to other dynamical systems, such as neuronal recordings.

.. rubric:: Improvements

- :func:`scFates.tl.attach_tree`: Allow to attach trees without milestones (using vertiex id instead).
- :func:`scFates.tl.subset_tree`: Better handling of tree subsetting when different root is used. Previosu milestones are saved.
- :func:`scFates.pl.trends` now respects embedding aspect ratio, can now save figure.

.. rubric:: Changes

- any graph fitting functions relying in elpigraph now removes automatically non-assigned nodes, and reattach the separated tree at the level of removals in case the tree is broken into pieces.
- :func:`scFates.pl.milestones` default layout to dendrogram view (similar to `tl.dendrogram` layout).
- :func:`scFates.tl.subset_tree` default mode is "extract".
- :func:`scFates.pl.linearity_deviation` has a font parameter, with a default value.

Version 0.2.6 :small:`August 29, 2021`
-------------------------------------

.. rubric:: Additions

- added :func:`scFates.tl.subset_tree` and :func:`scFates.tl.attach_tree`, functions that allow to perform linkage or cutting operations on tree or set of two trees.

.. rubric:: Improvements

- Added possibility to show any metadata on top of :func:`scFates.pl.trends`
- :func:`scFates.pl.trajectory` can now color segments with nice gradients of milestone colors following pseudotime.
- Added check for sparsity in :func:`scFates.pp.find_overdispersed`, as it is a crucial parameter for finding overdispersed features.
- :func:`scFates.tl.root` can now automatically select a tip, and with a minimum value instead of a max.
- :func:`scFates.pl.single_trend` can now plot raw and fitted mean module along pseudotime, plots with embedding can now be saved as image.

Version 0.2.5 :small:`July 09, 2021`
------------------------------------

.. rubric:: Addition/Changes

- code for SimplePPT algorithm has been moved to a standalone python package `simpelppt <https://github.com/LouisFaure/simpleppt/>`_.
- :func:`scFates.tl.activation_lm`, a more robust version of tl.activation, as it uses linear model to identify activation of feature prior to bifurcation.
- :func:`scFates.tl.root` can now automatically select root from any feature expression.


Version 0.2.4 :small:`May 31, 2021`
-----------------------------------

As mentioned in the following `issue <https://github.com/LouisFaure/scFates/issues/3>`_, this release removes the need to install the following dependencies: Palantir, cellrank and rpy2.
This allows for a faster installation of a base scFates package and avoid any possible issues caused by rpy2 and R conflicts.


.. rubric:: Modifications/Improvements

- :func:`scFates.pl.modules`: added `smooth` parameter for knn smoothing of the plotted values.
- :func:`scFates.pl.trajectory`: better segment and fork coloring, now uses averaging weigthed by the soft assignment matrix R to generate values.

Version 0.2.3 :small:`May 17, 2021`
-----------------------------------

.. rubric:: Additions

- :func:`scFates.tl.module_inclusion` and its plotting counterpart, estimate the pseudotime of inclusion of a feature whitin its own module.
- :func:`scFates.tl.linearity_deviation` and its plotting counterpart, a test to assess whether a given bride could be the result of doublets or not.
- :func:`scFates.tl.synchro_path_multi`, called with more than two terminal states. This wrapper will call :func:`scFates.tl.synchro_path` on all pair combination theses endpoints.
- :func:`scFates.tl.root` can now automatically identify the root node of the tree, by projecting on it differentiation measurments such as CytoTRACE.

.. rubric:: Modifications/Improvements

- More precise cell projection of critical transition index values via loess fit.


Version 0.2.2 :small:`Apr 27, 2021`
-----------------------------------

.. rubric:: Additions for conversion and downstream analysis

- :func:`scFates.tl.critical_transition`, with its plotting counterpart, calculate the critical transition index along the trajectory.
- :func:`scFates.tl.criticality_drivers`, identifies genes correlated with the projected critical transition index value on the cells.
- :func:`scFates.pl.test_fork`, plotting counterpart of :func:`scFates.tl.test_fork`, for better selection of threshold A.
- :func:`scFates.tl.cellrank_to_tree`, wrapper that convert results from CellRank analysis into a principal tree that can be subsequently analysed.

.. rubric:: Additions for preprocessing

- :func:`scFates.pp.diffusion`, wrapper that performs Palantir.
- :func:`scFates.pp.filter_cells` a molecule by genes filter translated from pagoda2 R package.
- :func:`scFates.pp.batch_correct` a simple batch correction method translated from pagoda2 R package.
- :func:`scFates.pp.find_overdispersed`, translated from pagoda2 R package.

Version 0.2.0 :small:`Feb 25, 2021`
------------------------------------

.. rubric:: Additons

- :func:`scFates.tl.curve` function, a wrapper of computeElasticPrincipalCurve from ElPiGraph, is now added to fit simple curved trajectories.
- Following this addition and for clarity, plotting functions :func:`scFates.pl.tree` and :func:`scFates.pl.tree_3d` have been respectively renamed :func:`scFates.pl.graph` and :func:`scFates.pl.trajectory_3d`.

.. rubric:: Modifications on :func:`scFates.tl.tree` when simplePPT is used

- euclidean distance function is replaced by :func:`sklearn.metrics.pairwise_distances` for cpu and :func:`cuml.metrics.pairwise_distances.pairwise_distances` for gpu, leading to speedups. Non-euclidean metrics can now be used for distance calculations.
- Several steps of computation are now performed via numba functions, leading to speedups for both cpu and gpu.
- Thanks to rapids 0.17 release, :func:`scipy.sparse.csgraph.minimum_spanning_tree` is replaced by :func:`cugraph.tree.minimum_spanning_tree.minimum_spanning_tree` on gpu, providing great speed improvements when learning a graph with very high number of nodes.

.. rubric:: :func:`scFates.tl.test_fork` modifications

- includes now a parameter that rescale the pseudotime length of the two post-bifurcation branches to 1. This allows for comparison between all cells, instead of only keeping cells with a pseudotime up to the maximum pseudotime of the shortest branch. This is useful especially when the two branches present highly different pseudotime length.
- can now perform DE on more than two branches (such in case of trifurcation).

.. rubric:: Other modifications on crestree related downstream analysis functions

- tl.activation now uses a distance based (pseudotime) sliding window instead of cells, leading to a more robust identification of activation pseudotime.
- include a fully working :func:`scFates.tl.refine_pseudotime` function, which applies Palantir separately on each segment of the fitted tree in order to mitigate the compressed pseudotime of cells at the tips.
- :func:`scFates.tl.slide_cors` can be performed using user defined group of genes, as well as on a single segment of the trajectory.


Version 0.1 :small:`Nov 16, 2020`
--------------------------------------

Version with downstream analysis functions closely related to the initial R package crestree. Includes ElPiGraph as an option to infer a principal graph.
