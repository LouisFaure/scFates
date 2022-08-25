import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd
from functools import partial
from anndata import AnnData
import shutil
import sys
import copy
from statsmodels.stats.multitest import multipletests
import igraph
import warnings


from joblib import delayed
from tqdm import tqdm

from .. import logging as logg
from .. import settings
from ..plot.test_association import test_association as plot_test_association
from .utils import getpath, get_X, ProgressParallel, importeR

Rpy2, R, rstats, rmgcv, Formula = importeR("testing feature association to the tree")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]


def test_association(
    adata: AnnData,
    n_map: int = 1,
    n_jobs: int = 1,
    spline_df: int = 5,
    fdr_cut: float = 1e-4,
    A_cut: int = 1,
    st_cut: float = 0.8,
    reapply_filters: bool = False,
    plot: bool = False,
    copy: bool = False,
    layer: Optional[str] = None,
):

    """\
    Determine a set of genes significantly associated with the trajectory.


    Feature expression is modeled as a function of pseudotime in a branch-specific manner,
    using cubic spline regression :math:`g_{i} \\sim\ t_{i}` for each branch independently.
    This tree-dependent model is then compared with an unconstrained model :math:`g_{i} \\sim\ 1`
    using F-test.

    The models are fit using *mgcv* R package.

    Benjamini-Hochberg correction is used to adjust for multiple hypothesis testing.


    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        adata layer to use for the test.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    spline_df
        dimension of the basis used to represent the smooth term.
    fdr_cut
        FDR (Benjamini-Hochberg adjustment) cutoff on significance; significance if FDR < fdr_cut.
    A_cut
        amplitude is max of predicted value minus min of predicted value by GAM. significance if A > A_cut.
    st_cut
        cutoff on stability (fraction of mappings with significant (fdr,A) pair) of association; significance, significance if st > st_cut.
    reapply_filters
        avoid recomputation and reapply fitlers.
    plot
        call scf.pl.test_association after the test.
    root
        restrain the test to a subset of the tree (in combination with leaves).
    leaves
        restrain the test to a subset of the tree (in combination with root).
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.var['p_val']`
            p-values from statistical test.
        `.var['fdr']`
            corrected values from multiple testing.
        `.var['st']`
            proportion of mapping in which feature is significant.
        `.var['A']`
            amplitue of change of tested feature.
        '.var['signi']`
            feature is significantly changing along pseuodtime
        `.uns['stat_assoc_list']`
            list of fitted features on the tree for all mappings.
    """

    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, rmgcv, Formula])[idx])

    adata = adata.copy() if copy else adata

    if "t" not in adata.obs:
        raise ValueError(
            "You need to run `tl.pseudotime` before testing for association."
            + "Or add a precomputed pseudotime at adata.obs['t'] for single segment."
        )

    if reapply_filters & ("stat_assoc_list" in adata.uns):
        stat_assoc_l = list(adata.uns["stat_assoc_list"].values())
        # stat_assoc_l = list(map(lambda x: pd.DataFrame(x,index=x["features"]),stat_assoc_l))
        adata = apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut)

        logg.info(
            "reapplied filters, "
            + str(sum(adata.var["signi"]))
            + " significant features"
        )

        if plot:
            plot_test_association(adata)

        return adata if copy else None

    Xgenes = get_X(adata, adata.obs_names, adata.var_names, layer, togenelist=True)

    logg.info("test features for association with the trajectory", reset=True, end="\n")

    stat_assoc_l = list()

    def test_assoc_map(m):
        if n_map == 1:
            df = adata.obs.loc[:, ["t", "seg"]]
        else:
            df = adata.uns["pseudotime_list"][str(m)]
        data = list(zip([df] * len(Xgenes), Xgenes))

        stat = ProgressParallel(
            n_jobs=n_jobs if n_map == 1 else 1,
            total=len(data),
            file=sys.stdout,
            use_tqdm=n_map == 1,
            desc="    single mapping ",
        )(delayed(test_assoc)(data[d], spline_df) for d in range(len(data)))
        stat = pd.DataFrame(stat, index=adata.var_names, columns=["p_val", "A"])
        stat["fdr"] = multipletests(stat.p_val, method="bonferroni")[1]
        return stat

    stat_assoc_l = ProgressParallel(
        n_jobs=1 if n_map == 1 else n_jobs,
        total=n_map,
        file=sys.stdout,
        use_tqdm=n_map > 1,
        desc="    multi mapping ",
    )(delayed(test_assoc_map)(m) for m in range(n_map))

    adata = apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut)

    logg.info(
        "    found " + str(sum(adata.var["signi"])) + " significant features",
        time=True,
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "added\n"
        "    .var['p_val'] values from statistical test.\n"
        "    .var['fdr'] corrected values from multiple testing.\n"
        "    .var['st'] proportion of mapping in which feature is significant.\n"
        "    .var['A'] amplitue of change of tested feature.\n"
        "    .var['signi'] feature is significantly changing along pseudotime.\n"
        "    .uns['stat_assoc_list'] list of fitted features on the graph for all mappings."
    )

    if plot:
        plot_test_association(adata)

    return adata if copy else None


def test_association_monocle3(
    adata: AnnData,
    qval_cut: float = 1e-4,
    n_jobs: bool = 1,
    copy: bool = False,
    **kwargs,
):
    """\
    Determine a set of genes significantly associated with the trajectory.

    the function `graph_test` *monocle3* R package is called,
    using `neighbor_graph = 'principal_graph'` parameter. The statistic tells you
    whether cells at nearby positions on a trajectory will have similar (or dissimilar)
    expression levels for the gene being tested.


    Parameters
    ----------
    adata
        Annotated data matrix.
    qval_cut
        cutoff on the corrected p-values to consider a feature as significant.
    n_jobs
        number of cpu processes used to perform the test.
    spline_df
        dimension of the basis used to represent the smooth term.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        `.var['status']`
            The feature could be tested.
        `.var['p_value']`
            p-values from statistical test.
        `.var['morans_test_statistic']`
            Statistics of the Moran test
        `.var['morans_I']`
            Moran’s I is a measure of multi-directional and multi-dimensional spatial autocorrelation.
        `.var['q_value']`
            corrected values from multiple testing.
        '.var['signi']`
            feature has q_value below cutoff.

    """

    Rpy2, R, rstats, monocle3, Formula = importeR(
        "testing feature association to the tree", "monocle3"
    )
    check = [type(imp) == str for imp in [Rpy2, R, rstats, monocle3, Formula]]
    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, monocle3, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, monocle3, Formula])[idx])

    logg.info(
        "test features for association with the trajectory, monocle3 way",
        reset=True,
        end="\n",
    )

    adata = adata.copy() if copy else adata

    import scipy.sparse as sp
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import py2rpy
    from scFates.tools.graph_fitting import get_data
    import anndata2ri

    anndata2ri.activate()

    from . import __path__

    Rfun = os.path.join(__path__[0], "_test_monocle3.R")
    rsource = ro.r["source"]
    rsource(Rfun)
    _test_monocle3 = ro.globalenv["test_monocle3"]

    # prepare data
    B = sp.csc_matrix(adata.uns["graph"]["B"]).astype(float)
    F = pd.DataFrame(adata.uns["graph"]["F"])
    F.columns = ["Y_" + str(i) for i in np.arange(adata.obsm["X_R"].shape[1]) + 1]
    pr_graph_cell_proj_closest_vertex = pd.DataFrame(
        {"1": np.argmax(adata.obsm["X_R"], axis=1) + 1}, index=adata.obs_names
    )
    cells = adata.obs_names
    genes = adata.var_names
    X = sp.csc_matrix(adata.X)
    UMAP, use_rep = get_data(
        adata, adata.uns["graph"]["use_rep"], adata.uns["graph"]["ndims_rep"]
    )
    UMAP = UMAP.values

    # prevent double log transformation
    if ("log1p" in adata.uns) & ("expression_family" not in kwargs):
        kwargs["expression_family"] = "uninormal"

    # Avoiding rBind error for large datasets
    # see: https://github.com/cole-trapnell-lab/monocle3/issues/509
    if ("neighbor_graph" not in kwargs) & (adata.shape[0] <= 10000):
        kwargs["neighbor_graph"] = "principal_graph"
    elif ("neighbor_graph" not in kwargs) & (adata.shape[0] > 10000):
        kwargs["neighbor_graph"] = "knn"

    logg.info(
        f"    neighbor_graph set to {kwargs['neighbor_graph']}",
        end=" " if settings.verbosity > 2 else "\n",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr_graph_test_res = _test_monocle3(
            X,
            genes,
            cells,
            UMAP,
            pr_graph_cell_proj_closest_vertex,
            F,
            B,
            n_jobs,
            **kwargs,
        )
    for c in pr_graph_test_res.columns:
        adata.var[c] = pr_graph_test_res[c]

    adata.var["signi"] = pr_graph_test_res.q_value < qval_cut
    logg.info(
        "    found " + str(sum(adata.var["signi"])) + " significant features",
        time=True,
        end=" " if settings.verbosity > 2 else "\n",
    )

    logg.hint(
        "added\n"
        "    .var['status'] the feature could be tested.\n"
        "    .var['p_value'] p-values from statistical test.\n"
        "    .var['morans_test_statistic'] Statistics of the Moran test.\n"
        "    .var['morans_I'] Moran’s I is a measure of multi-directional and multi-dimensional spatial autocorrelation.\n"
        "    .var['q_value'] corrected values from multiple testing.\n"
        "    .var['signi'] feature has q_value below cutoff."
    )

    return adata if copy else None


def test_assoc(data, spline_df):
    sdf = data[0]
    sdf["exp"] = data[1]

    global rmgcv
    global rstats

    def gamfit(s):
        m = rmgcv.gam(
            Formula(f"exp~s(t,k={spline_df})"), data=sdf.loc[sdf["seg"] == s, :]
        )
        return dict({"d": m[5][0], "df": m[42][0], "p": rmgcv.predict_gam(m)})

    mdl = list(map(gamfit, sdf.seg.unique()))
    mdf = pd.concat(list(map(lambda x: pd.DataFrame([x["d"], x["df"]]), mdl)), axis=1).T
    mdf.columns = ["d", "df"]

    odf = sum(mdf["df"]) - mdf.shape[0]
    m0 = rmgcv.gam(Formula("exp~1"), data=sdf)
    if sum(mdf["d"]) == 0:
        fstat = 0
    else:
        fstat = (m0[5][0] - sum(mdf["d"])) / (m0[42][0] - odf) / (sum(mdf["d"]) / odf)

    df_res0 = m0[42][0]
    df_res_odf = df_res0 - odf
    pval = rstats.pf(fstat, df_res_odf, odf, lower_tail=False)[0]
    pr = np.concatenate(list(map(lambda x: x["p"], mdl)))

    return [pval, max(pr) - min(pr)]


def apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut, prefix=""):
    n_map = len(stat_assoc_l)
    if n_map > 1:
        stat_assoc = pd.DataFrame(
            {
                prefix
                + "p_val": pd.concat(
                    list(map(lambda x: x[prefix + "p_val"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "A": pd.concat(
                    list(map(lambda x: x[prefix + "A"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "fdr": pd.concat(
                    list(map(lambda x: x[prefix + "fdr"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "st": pd.concat(
                    list(
                        map(
                            lambda x: (x[prefix + "fdr"] < fdr_cut)
                            & (x[prefix + "A"] > A_cut),
                            stat_assoc_l,
                        )
                    ),
                    axis=1,
                ).sum(axis=1)
                / n_map,
            }
        )
    else:
        stat_assoc = stat_assoc_l[0]
        stat_assoc[prefix + "st"] = (
            (stat_assoc[prefix + "fdr"] < fdr_cut) & (stat_assoc[prefix + "A"] > A_cut)
        ) * 1

    # saving results
    stat_assoc[prefix + "signi"] = stat_assoc[prefix + "st"] > st_cut
    for c in stat_assoc.columns:
        adata.var[c] = stat_assoc[c]

    names = np.arange(len(stat_assoc_l)).astype(str).tolist()

    dictionary = dict(zip(names, stat_assoc_l))
    adata.uns[prefix + "stat_assoc_list"] = dictionary

    return adata
