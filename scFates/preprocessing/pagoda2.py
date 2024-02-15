"""
The following code has been translated from R package pagoda2, if you use any of these functions please cite:
Nikolas Barkas, Viktor Petukhov, Peter Kharchenko and Evan
Biederstedt (2021). pagoda2: Single Cell Analysis and Differential
Expression. R package version 1.0.2.
"""

from anndata import AnnData

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


import scipy.sparse as sp
import multiprocessing
import numpy as np


from .. import logging as logg
from .. import settings
from ..tools.utils import get_SE, bh_adjust, importeR


def filter_cells(
    adata: AnnData,
    p_level=None,
    min_cell_size=500,
    max_cell_size=5e4,
    subset=True,
    plot=False,
    copy=False,
):

    """\
    Filter cells using on gene/molecule relationship.

    Code has been translated from pagoda2 R function gene.vs.molecule.cell.filter.


    Parameters
    ----------
    adata
        Annotated data matrix.
    p_level
        Statistical confidence level for deviation from the main trend, used for cell filtering (default=min(1e-3,1/adata.shape[0]))
    min_cell_size
        Minimum number of transcripts per cells
    max_cell_size
        Maximum number of transcripts per cells
    subset
        if False, add a column `outlier` in adata.obs, otherwise subset the adata.
    plot
        Plot the molecule distribution and the gene/molecule dependency fit.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------

    adata : anndata.AnnData
        if `copy=True` and `subset=True` it returns subsetted (removing outliers) or else add fields to `adata`:

        `.obs['outlier']`
            whether a cell is an outlier.

    """

    adata = adata.copy() if copy else adata

    Rpy2, R, rstats, rMASS, Formula = importeR("filter cells", "MASS")

    logg.info("Filtering cells", reset=True)
    X = adata.X.copy()

    logg.info("    obtaining gene and molecule counts")

    df = pd.DataFrame(
        {
            "molecules": np.array(adata.X.sum(axis=1)).ravel(),
            "genes": np.array((adata.X > 0).sum(axis=1)).ravel(),
        },
        index=adata.obs_names,
    )

    df = df.loc[df.molecules >= min_cell_size, :]
    df = np.log10(df)

    logg.info("    fitting RLM")

    m = rMASS.rlm(Formula("genes~molecules"), data=df)
    p_level = min(1e-3, 1 / adata.shape[0]) if p_level is None else p_level

    pr = rstats.predict(m, interval="prediction", level=1 - p_level, type="response")
    pr = pd.DataFrame(pr, index=df.index)

    outlier = (df.genes < pr[1]) | (df.genes > pr[2])

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")

    if plot:
        fig, ax = plt.subplots()
        idx = df.sort_values("molecules").index
        ax.fill_between(
            df.molecules[[idx[0], idx[-1]]],
            pr[1][[idx[0], idx[-1]]],
            pr[2][[idx[0], idx[-1]]],
            color="yellow",
            alpha=0.3,
        )

        ax.scatter(
            df.loc[~outlier].molecules, df.loc[~outlier].genes, s=5, alpha=0.2, c="k"
        )
        ax.scatter(
            df.loc[outlier].molecules, df.loc[outlier].genes, s=5, alpha=0.5, c="r"
        )
        ax.axvline(x=np.log10(min_cell_size), c="r", linestyle="--")
        ax.axvline(x=np.log10(max_cell_size), c="r", linestyle="--")
        ax.set_ylabel("log10(genes)")
        ax.set_xlabel("log10(molecules)")

    df = df.loc[df.molecules > np.log10(min_cell_size), :]
    df = df.loc[df.molecules < np.log10(max_cell_size), :]
    df = df.loc[~outlier]
    if subset:
        adata._inplace_subset_obs(df.index)
        logg.hint("subsetted adata.")

    else:
        adata.obs["outlier"] = True
        adata.obs[df.index, "outlier"] = False
        logg.hint("added \n" "    .obs['outlier'], boolean column indicating outliers.")

    return adata if copy else None


def batch_correct(
    adata, batch_key, layer="X", depth_scale=1e3, device="cpu", inplace=True
):

    """\
    batch correction of the count matrix.

    Code has been translated from pagoda2 R function setCountMatrix (plain model).

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column name to use for batch.
    layer
        Which layer to correct, if layer doesn't exist, then correct X and save to layer
    depth_scale
        Depth scale.
    device
        Run method on either `cpu` or on `gpu`.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `inplace=False` it returns the corrected matrix, else it update field to `adata`:

        `.X`
            batch-corrected count matrix.

    """
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if layer in adata.layers:
        X = adata.layers[layer].copy()
    else:
        X = adata.X.copy()

    logg.info("Performing pagoda2 batch correction", reset=True)
    if adata.obs[batch_key].dtype.name != "category":
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    batches = adata.obs[batch_key].cat.categories
    nbatches = len(batches)

    if device == "cpu":
        depth = X.sum(axis=1)
        depth = np.array(depth).ravel()

        gene_av = (np.array(X.sum(axis=0)).ravel() + len(batches)) / (
            depth.sum() + len(batches)
        )
        tc = np.vstack([X[adata.obs[batch_key] == b, :].sum(axis=0) for b in batches])
        tc = np.log(tc + 1) - np.log(
            np.array([depth[adata.obs[batch_key].values == b].sum() for b in batches])
            + 1
        ).reshape(-1, 1)
        bc = np.exp(tc - np.log(gene_av.astype(np.float64)))
        bc = pd.DataFrame(np.transpose(bc), columns=batches)
        X = sp.csr_matrix(X.transpose())

        batch = adata.obs[batch_key].cat.rename_categories(range(nbatches))
        count_gene = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        acc = np.transpose(np.vstack([count_gene, batch[X.indices].values]))
        X.data = X.data / bc.values[acc[:, 0], acc[:, 1]]
        logg.info("    depth scaling")
        X = X.transpose()
        d = depth / depth_scale
        X = X.multiply(1.0 / d[None, :].T)

    elif device == "gpu":
        try:
            import cupy as cp
            import cudf
            from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )

        X = csr_matrix_gpu(X)
        depth = X.sum(axis=1)
        depth = cp.array(depth).ravel()

        gene_av = (cp.array(X.sum(axis=0)).ravel() + len(batches)) / (
            depth.sum() + len(batches)
        )
        tc = cp.vstack([X[adata.obs[batch_key] == b, :].sum(axis=0) for b in batches])
        tc = cp.log(tc + 1) - cp.log(
            cp.array([depth[adata.obs[batch_key].values == b].sum() for b in batches])
            + 1
        ).reshape(-1, 1)
        bc = cp.exp(tc - np.log(gene_av.astype(cp.float64)))
        bc = cudf.DataFrame(np.transpose(bc.get()), columns=batches)
        X = csr_matrix_gpu(X.transpose())

        batch = adata.obs[batch_key].cat.rename_categories(range(nbatches))
        count_gene = cp.repeat(cp.arange(X.shape[0]), cp.diff(X.indptr).get().tolist())
        batch_to_stack = cp.array(batch.values[X.indices.get()])
        acc = cp.transpose(cp.vstack([count_gene, batch_to_stack]))
        X.data = X.data / bc.values[acc[:, 0], acc[:, 1]]
        X = X.transpose()
        logg.info("    depth scaling")
        d = depth / depth_scale
        X = X.multiply(1.0 / d[None, :].T)
        X = X.get()

    X = sp.csr_matrix(X)
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")

    if inplace:
        if layer == "X":
            adata.X = X
            logg.hint("updated \n" "    .X, batch corrected matrix.")
        else:
            adata.layers[layer] = X
            logg.hint(
                "updated \n" "    .layer['" + layer + "'], batch corrected matrix."
            )

    else:
        return sp.csr_matrix(X)


def find_overdispersed(
    adata,
    gam_k: int = 5,
    alpha: float = 5e-2,
    layer: str = "X",
    plot: bool = False,
    copy: bool = False,
):

    """\
    Find overdispersed gene, using pagoda2 strategy.

    Code has been translated from pagoda2 R function adjustVariance.

    Parameters
    ----------
    adata
        Annotated data matrix.
    gam_k
        The k used for the generalized additive model.
    alpha
        The criterion used to measure statistical significance.
    layer
        Which layer to use.
    plot
        Plot selected genes.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

         .var['res']
             residuals of GAM fit.
         .var['lp']
             p-value.
         .var['lpa']
             BH adjusted p-value.
         .var['qv']
             percentile of qui-squared distribution.
         .var['highly_variable']
             feature is over-dispersed.

    """

    Rpy2, R, rstats, rmgcv, Formula = importeR("finding overdispersed features")

    if any(np.array(adata.X.sum(axis=0)).ravel() == 0):
        raise Exception(
            "Cannot find overdispersed features if some are not expressed in any cell!"
        )

    if not sp.issparse(adata.X):
        raise Exception("Overdispersion requires sparse matrix!")

    logg.info("Finding overdispersed features", reset=True)

    adata = adata.copy() if copy else adata

    if layer == "X":
        X = adata.X.copy()
    else:
        X = adata.layers[layer].copy()

    logg.info("    computing mean and variances")

    m = np.log(np.array(X.mean(axis=0)).ravel())
    v = np.log(StandardScaler(with_mean=False).fit(X).var_)
    X.data = np.ones_like(X.data)
    n_obs = np.array(X.sum(axis=0)).ravel()

    df = pd.DataFrame({"m": m, "v": v, "n_obs": n_obs}, index=adata.var_names)

    logg.info("    gam fitting")
    m = rmgcv.gam(Formula("v~s(m,k=" + str(gam_k) + ")"), data=df)
    df["res"] = rstats.residuals(m, type="response")
    n_obs = df.n_obs

    df["lp"] = rstats.pf(np.exp(df.res), n_obs, n_obs, lower_tail=False, log_p=True)
    df["lpa"] = bh_adjust(df["lp"], log=True)

    n_cells = adata.shape[0]
    df["qv"] = (
        rstats.qchisq(df["lp"], n_cells - 1, lower_tail=False, log_p=True) / n_cells
    )

    ods = df["lpa"] < np.log(alpha)
    df["highly_variable"] = ods.values

    adata.var[df.columns] = df

    logg.info(
        "    found " + str(sum(df["highly_variable"])) + " over-dispersed features",
        time=True,
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "added \n"
        "    .var['res'], residuals of GAM fit.\n"
        "    .var['lp'], p-value.\n"
        "    .var['lpa'], BH adjusted p-value.\n"
        "    .var['qv'], percentile of qui-squared distribution.\n"
        "    .var['highly_variable'], feature is over-dispersed.\n"
    )

    if plot:
        fig, ax = plt.subplots()
        df.loc[~df["highly_variable"]].plot.scatter(
            x="m", y="v", ax=ax, c="lightgrey", s=1
        )
        df.loc[df["highly_variable"]].plot.scatter(x="m", y="v", ax=ax, s=1, c="k")
        ax.set_xlabel("log10(magnitude)")
        ax.set_ylabel("log10(variance)")

    return adata if copy else None


def neighbors(
    adata,
    n_neighbors=50,
    n_pcs=None,
    use_rep="X_pca",
    center=True,
    metric="angular",
    n_jobs=-1,
):

    """\
    Make the nearest neighbor graph, using N2 algorithm, similar to pagoda2.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        Number of neighbors.
    n_pcs
        Number of dimension to use.
    use_rep
        Which reduction to use to calculate graph.
    center
        Center the reduction before calulating graph.
    metric
        metric to use, either 'angular' (cosine) or 'l2' (euclidean)
    n_jobs
        Number of cpu cores to use.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

         .var['res']
             residuals of GAM fit.
         .var['lp']
             p-value.
         .var['lpa']
             BH adjusted p-value.
         .var['qv']
             percentile of qui-squared distribution.
         .var['highly_variable']
             feature is over-dispersed.

    """

    try:
        from n2 import HnswIndex
    except ImportError:
        raise ImportError("Please install the N2 algorithm: `pip install n2`.")

    logg.info("Generating NN graph", reset=True)
    if center:
        x = adata.obsm[use_rep] - adata.obsm[use_rep].mean(axis=1).reshape(-1, 1)
    else:
        x = adata.obsm[use_rep]
    if n_pcs is None:
        n_pcs = x.shape[1]

    logg.info("    building graph")
    u = HnswIndex(n_pcs, metric)
    for i in range(x.shape[0]):
        u.add_data(x[i, :])

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1

    u.build(m=n_pcs + 1, n_threads=n_jobs)

    logg.info("    generating adjacency")
    adj = u.batch_search_by_ids(np.arange(x.shape[0]), n_pcs + 1, num_threads=n_jobs)
    adj = [a[1:] for a in adj]
    idx = np.repeat(range(x.shape[0]), n_pcs)

    adj = np.concatenate(adj)
    adj = sp.coo_matrix((np.ones_like(idx), (idx, adj)), shape=(x.shape[0], x.shape[0]))

    adata.obsp["connectivities"] = sp.csr_matrix(adj)

    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": n_neighbors, "method": "n2", "metric": metric},
    }

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    .obsp['connectivities'], adjacency matrix.")
