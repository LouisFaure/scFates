.. role:: small
.. role:: smaller

Release Notes
=============

Version 1.1.0 :small:`March 30, 2567/2025`
--------------------------------------------
- Allow to generate vector graphics plots for pl.trajectory and pl.graph
- Fix github actions


Version 1.0.9 :small:`October 31, 2566/2024`
--------------------------------------------
- Future proofing scFates by allowing pandas>2.0
- Fixed most Deprecation warnings.
- Fix issue with branch_specific, added check when all genes are filtered out

Version 1.0.8 :small:`June 19, 2566/2024`
--------------------------------------------
- Compatibility with cellrank2
- various dependency fixes

Version 1.0.7 :small:`February 15, 2566/2024`
--------------------------------------------
Fixed adjustedText version to avoid error, changed tests to python 3.11

Version 1.0.6 :small:`August 26, 2566/2023`
-------------------------------------------
Fixed exception in :func:`scFates.tl.slide_cells`

Version 1.0.5 :small:`August 25, 2566/2023`
-------------------------------------------
Fix int and bool check when ordering segments in :func:`scFates.pl.trends`.

Version 1.0.4 :small:`August 13, 2566/2023`
------------------------------------------
- updated notebooks.
- relaxed mutli-mapping rule when using elpigraph.
- better handling of milestone renaming.
- prevent grid on module_inclusion plot
- correcting messages displayed by functions.

Version 1.0.3 :small:`July 23, 2566/2023`
------------------------------------------
Updated elpigraph version to fix error with networkx, fixed typo.

Version 1.0.2 :small:`April 28, 2566/2023`
------------------------------------------
Constrained pandas version requirement (<2.0) to avoid broken functions.

Version 1.0.1 :small:`March 10, 2566/2023`
------------------------------------------
Minor fixes to make scFates compatible with the newer versions of matplotlib (3.5+).
Constrained networkx requirement to avoid error happening in the last version 3.0.


Version 1.0.0 :small:`November 29, 2022`
-------------------------------------
The tool is now `published <https://doi.org/10.1093/bioinformatics/btac746>`_, it is considered stable enough to be released as v1.0.0

- :func:`scFates.pl.trends` displays an error message if no feature is plotted.
- :func:`scFates.tl.tree` now accept parameters transfer to elpigraph-python.

Version 0.9.1 :small:`August 28, 2022`
-------------------------------------

- Switched to ElPiGraph approach to calculate pseudotime when using that algorithm, leading to more accurate pseudotime measurement.
- Added parameter `epg_extend_leaves` to call :func:`elpigraph.ExtendLeaves` during graph learning using ElPiGraph.
- Working function for :func:`scFates.tl.test_association_monocle3` (R file was missing from package).
- Fixed output from :func:`scFates.tl.test_association_covariate`.
- Allow no legend for :func:`scFates.pl.covariate`.


Version 0.9.0 :small:`August 18, 2022`
--------------------------------------

Major release:

This release has several improvements from 0.8

Major changes:

- As discussed on issue `#7 <https://github.com/LouisFaure/scFates/issues/7>`_, pseudotime calculation has been fixed when using elpigraph. The previous change introduced the issue of cells being assigned the pseudotime of their closest node only. Now the cells are assigned to their closest edge and have a pseudotime value according to their distance between the two nodes composing that edge.
- Added :func:`scFates.tl.explore_sigma`, a tool for SimplePPT that explore ranges of sigma parameters to avoid the ones which collapse the tree (see the `related notebook <https://scfates.readthedocs.io/en/latest/Explore_sigma.html>`_) for more info).
- New approach to analyses circles, upon removal of edge linked to the root node,  the graph is considered as two converging segments toward the furthest node. This allow to perform mulitple mapping without having cells being assigned either the lowest or the furthest pseudotime, leading to wrong assignement when taking the mean of all mappings. The circle can be further unrolled with :func:`scFates.tl.unroll_circle` to assign a unique pseudotime value to all cells (for more info see the `related notebook <https://scfates.readthedocs.io/en/latest/Beyond_scRNAseq.html>`_).
- added :func:`scFates.tl.test_association_monocle3`, to test whether features are significantly changing along the tree, using monocle3 approach (requires the package). This can be handy for large dataset where test_association is too slow (does not generate A parameter).
- Reworked :func:`scFates.tl.cluster`, now uses scanpy and leiden as backend, leading to faster gene module calculations.


Version 0.8.1 :small:`July 18, 2022`
------------------------------------

Minor release:
- `pl.milestones_graph` has been removed, simplifying the dependency requirements
- :func:`scFates.tl.rename_milestones` now accepts dictionaries
- minor plot fixes


Version 0.8.0 :small:`June 29, 2022`
------------------------------------

This release is stable and ready for journal submission, it is meant to be ready to use and in line with all methods described in the manuscript.

Major changes:

- **breaking change!** pseudotime calculation is now deterministic, which differs from the previous implementation derived from crestree package. In the previous implementation, cells were assigned to a random position between a node and its closest neighbor. Now cells are assigned a pseudotime according to their soft assignment value between between the node and its closest neighbor.
- When calculating pseudotime over several mappings, the mean of all pseudotimes is saved in .obs, instead of taking the first mapping. Cell are assigned to their most assigned segment among all mappings, with corrections for cases were the pseudotime is over or under the limit of the segment.

Other changes:

- :func:`scFates.pl.milestones` has been converted into a embedding plot which colors the cells as a gradient following milestones. This plot will be called in any other plotting functions which as a coloring of cell paramter set to 'milestones'.
- Added :func:`scFates.tl.convert_to_soft` to convert ElPiGraph hard assignment R matrix output into a soft one, allowing for probabilistic mapping of cells.
- For plot with embeddings, the basis parameter is now automatically guessed if none is mentionned.
- Improved flexibility and consistency when plotting sub-trajectories
- Default parameters for :func:`scFates.tl.module_inclusion` have been modified, to focus more on already identified early genes. Inclusion of single gene can now be plotted.


Version 0.4.2 :small:`May 16, 2022`
---------------------------------------

Minor release:

- Updated to latest elpigraph version available on pypi, induced slightly changes in principal graph results.
- Added cmap parameter to :func:`scFates.pl.matrix`, more responsive plotting.
- Fix presence of NAs as repulsion scores in :func:`scFates.pl.slide_cors`.

Version 0.4.1 :small:`March 25, 2022`
---------------------------------------

Minor release focused mainly in plotting improvements:

- Better handling of cases between plot module trends and feature trends for :func:`scFates.pl.single_trend`.
- Added colorbar and normalization parameter to :func:`scFates.pl.matrix`.
- Ordering cells according to pseudotime in :func:`scFates.pl.dendrogram` when coloring by milestone gradients.
- Rasterize segments in :func:`scFates.pl.trajectory`.
- Fixed auto root selection for :func:`scFates.tl.cellrank_to_tree`

Version 0.4.0 :small:`February 25, 2022`
---------------------------------------

.. rubric:: Additions

- :func:`scFates.tl.test_association_covariate`, to separately test for associated features for each covariates on the same trajectory path.
- :func:`scFates.tl.test_covariate`, to test for branch differential gene expression between two covariates on the same trajectory path.

.. rubric:: Improvements

- :func:`scFates.tl.fit` can be called for any features.
- :func:`scFates.tl.test_association` has now spline.df parameter.
- :func:`scFates.pl.graph` : Segments and nodes are now rasterized in pl.graph for lighter plotting.
- :func:`scFates.pl.matrix` can now return related dataset.
- :func:`scFates.pl.slide_cors` : Absolute repulsion score is now shown.


Version 0.3.2 :small:`February 12, 2022`
---------------------------------------

.. rubric:: Additions

- :module:`scFates.get` to easily extract data generated by various analyses. (:func:`scFates.get.fork_stats`, :func:`scFates.get.modules`, :func:`scFates.get.slide_cors`)
- :func:`scFates.tl.simplify`, subset a tree by cutting of any nodes and cells having a higher pseudotime value than a threshold.
- `scf.settings.set_figure_pubready()` to set publication ready figures (PDF/Arial output, needs Arial installed on the system)

.. rubric:: Improvements/Fix

- **_!Affected results!_**: Effect calculation only consider compared cells when rescale=False in :func:`scFates.tl.test_fork`
- Merged :func:`scFates.tl.limit_pseudotime` with :func:`scFates.tl.subset`, can now cutoff before a set pseudotime (`t_min` parameter).
- :func:`scFates.pl.slide_cors` : Allow to focus on one window and annotate most repuslive genes. Fixed inverted colors for the gene modules when bifuraction analysis was applied.
- Flexibility improvements for :func:`scFates.pl.matrix`, :func:`scFates.pl.single_trend`, :func:`scFates.pl.graph`, :func:`scFates.pl.synchro_path`, :func:`scFates.pl.modules`



Version 0.3.1 :small:`January 4, 2022`
---------------------------------------

.. rubric:: Additions

- :func:`scFates.pl.matrix` a new and compact way for plotting features over a subset or the whole tree.
- :func:`scFates.tl.limit_pseudotime`, subset a tree by cutting of any nodes and cells having a higher pseudotime value than a threshold.
- `scf.settings.set_figure_pubready()` to set publication ready figures (PDF/Arial output, needs Arial installed on the system)

.. rubric:: Improvements/Fix

- Solved :func:`scFates.tl.dendrogram` breaking down when version of seaborn is higher than v0.11.1
- :func:`scFates.tl.cluster`: Output more information.
- Better parallel handling of :func:`tl.test_association` for multiple mapping.
- Flexibility improvements for :func:`scFates.pl.trends`, :func:`scFates.pl.single_trend`, :func:`scFates.pl.synchro_path`, :func:`scFates.pl.modules`.


Version 0.3 :small:`November 11, 2021`
---------------------------------------

.. rubric:: Changes

- **_!Breaking change!_** R soft assignment matrix now is moved to `.obsm` for better flexibility (notably when subsetting). If using an older dataset: refit the tree (with the same parameters) to update to the new data organisation.
- Removal of LOESS for :func:`scFates.tl.synchro_path` (too slow). Using GAM instead, and only when calling :func:`scFates.pl.synchro_path`.
- Removal of critical transition related functions.

.. rubric:: Improvements

- :func:`scFates.pp.batch_correct` Faster matrix saving.
- :func:`scFates.tl.circle`: Allow to use weights for graph fitting with simpleppt.
- :func:`scFates.tl.subset_tree`: Transfer segment colors to new tree when subsetting.
- :func:`scFates.tl.circle`: Better parallelism when doing on multiple mappings.
- :func:`scFates.pl.binned_pseudotime_meta`: More responsive plot.
- Better handling of R dependencies related errors.

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
