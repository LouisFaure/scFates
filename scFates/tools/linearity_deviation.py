from scFates import logging as logg
from scFates import settings
from scFates.tools.utils import getpath, get_X, ProgressParallel

import igraph
from joblib import delayed
import sys

import numpy as np
import statsmodels.formula.api as smf
from anndata import AnnData
import pandas as pd


def linearity_deviation(
    adata: AnnData,
    start_milestone,
    end_milestone,
    percentiles=[20, 80],
    n_jobs: int = 1,
    n_map: int = 1,
    plot: bool = False,
    basis: str = "X_umap",
    copy: bool = False,
):
    """\
    Identifies genes that specifically characterize a given transition but not the progenitors neither the progenies.

    This approach has been developped in the following study [Kameneva21]_.
    Designed to test whether a bridge/transition is the result of a doublet population only, this test checks if a gene expression occuring in the transition/bridge could be explained by a linear mixture of expressions of that gene in progenitors and progenies. The gene expression profile of each cell of the bridge is modeled as a linear combination of mean gene expression profiles in progenitors and progenies.

    For each gene in each cell in bridge is calculated the magnitude of the residuals not explained by the model. The mean residuals across all cells in the transition/bridge is then normalized to the standard deviation of the expression of a given gene. The obtained normalized mean residual values is used to prioritize the genes with distinctive expression patterns in the bridge population.

    Parameters
    ----------
    adata
        Annotated data matrix.
    start_milestone
        tip defining progenitor branch.
    end_milestone
        tips defining the progeny branch.
    percentiles
        pseudotime percentiles to define the progenitor and progeny populations
    n_jobs
        number of cpu processes used to perform the test.
    n_map
        number of cell mappings from which to do the test.
    plot
        plot the cells selection according to percentiles.
    basis
        basis to use in case of plotting
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.var['A->B_rss']`
            pearson residuals of the linear fit.
        `.obs['A->B_lindev_sel']`
            cell selections used for the test.

    """

    logg.info("Estimation of deviation from linearity", reset=True)

    adata = adata.copy() if copy else adata

    name = start_milestone + "->" + end_milestone

    graph = adata.uns["graph"]

    def lindev_map(m):
        df = adata.uns["pseudotime_list"][str(m)].copy()
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        cells = getpath(
            img,
            graph["milestones"][start_milestone],
            graph["tips"],
            graph["milestones"][end_milestone],
            graph,
            df,
        )

        t_perc = [cells.t.quantile(p / 100) for p in percentiles]

        df[name + "_lindev_sel"] = 'none'
        df.loc[
            adata.obs_names.isin(cells.index), name + "_lindev_sel"
        ] = "putative bridge"
        df.loc[
            (adata.obs_names.isin(cells.index)) & (adata.obs.t < t_perc[0]),
            name + "_lindev_sel",
        ] = "putative progenitors"
        df.loc[
            (adata.obs_names.isin(cells.index)) & (adata.obs.t > t_perc[1]),
            name + "_lindev_sel",
        ] = "putative progenies"

        if m == 0:
            adata.obs[name + "_lindev_sel"] = df[name + "_lindev_sel"]

        progenitors = adata.obs_names[
            df[name + "_lindev_sel"] == "putative progenitors"
        ]
        progenies = adata.obs_names[df[name + "_lindev_sel"] == "putative progenies"]
        bridge = adata.obs_names[df[name + "_lindev_sel"] == "putative bridge"]

        X_all = get_X(adata, cells.index, adata.var_names, layer=None)
        X_progenitors = get_X(adata, progenitors, adata.var_names, layer=None)
        X_progenies = get_X(adata, progenies, adata.var_names, layer=None)
        X_bridge = get_X(adata, bridge, adata.var_names, layer=None)

        X = X_bridge.tolist()

        A = X_progenitors.mean(axis=0)
        B = X_progenies.mean(axis=0)

        def get_resid(x):
            model = smf.ols(
                formula="x ~ A + B - 1", data=pd.DataFrame({"x": x, "A": A, "B": B})
            )
            results = model.fit()
            return results.resid_pearson

        rs = ProgressParallel(
            total=len(X),
            n_jobs=n_jobs,
            use_tqdm=n_map == 1,
            desc="    cells on the bridge",
            file=sys.stdout,
        )(delayed(get_resid)(x) for x in X)

        rs = np.vstack(rs).T
        return rs.mean(axis=1) / X_all.std(axis=0)

    n_jobs_map = 1
    if n_map > 1:
        n_jobs_map = n_jobs
        n_jobs = 1

    rss = ProgressParallel(
        total=n_map,
        n_jobs=n_jobs_map,
        use_tqdm=n_map > 1,
        desc="    multi mapping",
        file=sys.stdout,
    )(delayed(lindev_map)(m) for m in range(n_map))

    if plot:
        import scanpy as sc

        sc.pl.embedding(adata, basis=basis, color=name + "_lindev_sel")

    adata.var[name + "_rss"] = np.nanmean(np.vstack(rss), axis=0)

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .var['" + name + "_rss'], pearson residuals of the linear fit.\n"
        "    .obs['" + name + "_lindev_sel'], cell selections used for the test."
    )

    return adata if copy else None
