.. role:: small
.. role:: smaller

Release Notes
=============

Version 0.2.2 :small:`Apr 27, 2021`
-----------------------------------

Additions for conversion and downstream analysis:

- `tl.critical_transition`, with its plotting counterpart, calculate the critical transition index along the trajectory.
- `tl.criticality_drivers`, identifies genes correlated with the projected critical transition index value on the cells.
- `pl.test_fork`, plotting counterpart of `tl.test_fork`, for better selection of threshold A.
- `tl.cellrank_to_tree`, wrapper that convert results from CellRank analysis into a principal tree that can be subsequently analysed.

Additions for preprocessing:

- `pp.diffusion`, wrapper that performs Palantir.
- `pp.filter_cells` a molecule by genes filter translated from pagoda2 R package.
- `pp.batch_correct` a simple batch correction method translated from pagoda2 R package.
- `pp.find_overdispersed`, translated from pagoda2 R package.

Version 0.2.0 :small:`Feb 25, 2021`
------------------------------------

Additons:

- `tl.curve` function, a wrapper of computeElasticPrincipalCurve from ElPiGraph, is now added to fit simple curved trajectories.
- Following this addition and for clarity, plotting functions `pl.tree` and `pl.tree_3d` have been respectively renamed `pl.graph` and `pl.trajectory_3d`.

Modifications on `tl.tree` when simplePPT is used:

- euclidean distance function is replaced by sklearn.metrics.pairwise_distances for cpu and cuml.metrics.pairwise_distances for gpu, leading to speedups. Non-euclidean metrics can now be used for distance calculations.
- Several steps of computation are now performed via numba functions, leading to speedups for both cpu and gpu.
- Thanks to rapids 0.17 release, scipy.sparse.csgraph.minimum_spanning_tree is replaced by cugraph.tree.minimum_spanning_tree on gpu, providing great speed improvements when learning a graph with very high number of nodes.

`tl.test_fork` modifications:

- includes now a parameter that rescale the pseudotime length of the two post-bifurcation branches to 1. This allows for comparison between all cells, instead of only keeping cells with a pseudotime up to the maximum pseudotime of the shortest branch. This is useful especially when the two branches present highly different pseudotime length.
- can now perform DE on more than two branches (such in case of trifurcation).

Other modifications on crestree related downstream analysis functions:

- tl.activation now uses a distance based (pseudotime) sliding window instead of cells, leading to a more robust identification of activation pseudotime.
- include a fully working `tl.refine_pseudotime` function, which applies Palantir separately on each segment of the fitted tree in order to mitigate the compressed pseudotime of cells at the tips.
- `tl.slide_cors` can be performed using user defined group of genes, as well as on a single segment of the trajectory.


Version 0.1 :small:`Nov 16, 2020`
--------------------------------------

Version with downstream analysis functions closely related to the initial R package crestree. Includes ElPiGraph as an option to infer a principal graph.
