.. role:: small
.. role:: smaller

Release Notes
=============

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
