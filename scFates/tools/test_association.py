import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Union, Optional, Tuple, Collection, Sequence, Iterable

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


from joblib import delayed, Parallel
from tqdm import tqdm
from scipy import sparse

from .. import logging as logg
from .. import settings
from ..plot.test_association import test_association as plot_test_association
from .utils import getpath


try:
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr
    import rpy2.rinterface

    pandas2ri.activate()
except Exception as e:
    warnings.warn(
        'Cannot compute gene expression trends without installing rpy2. \
        \nPlease use "pip3 install rpy2" to install rpy2'
    )
    print(e.__doc__)
    print(e.message)


if not shutil.which("R"):
    warnings.warn(
        "R installation is necessary for computing gene expression trends. \
        \nPlease install R and try again"
    )

try:
    rstats = importr("stats")
except Exception as e:
    warnings.warn(
        "R installation is necessary for computing gene expression trends. \
        \nPlease install R and try again"
    )
    print(e.__doc__)
    print(e.message)

try:
    rmgcv = importr("mgcv")
except Exception as e:
    warnings.warn(
        'R package "mgcv" is necessary for computing gene expression trends. \
        \nPlease install gam from https://cran.r-project.org/web/packages/gam/ and try again'
    )
    print(e.__doc__)
    print(e.message)


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
    root=None,
    leaves=None,
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
        amplitude cutoff on significance; significance if A > A_cut.
    st_cut
        cutoff on stability (fraction of mappings with significant (fdr,A) pair) of association; significance, significance if st > st_cut.
    reapply_filters
        avoid recpmputation and reapply fitlers.
    plot
        call scf.pl.test_associationa after the test.
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

    adata = adata.copy() if copy else adata

    if "pseudotime_list" not in adata.uns:
        raise ValueError(
            "You need to run `tl.pseudotime` before testing for association."
        )

    graph = adata.uns["graph"]

    mlsc_temp = None
    if leaves is not None:
        # weird hack to keep milestones colors saved
        if "milestones_colors" in adata.uns:
            mlsc = adata.uns["milestones_colors"].copy()
            mlsc_temp = mlsc.copy()

        dct = graph["milestones"]
        keys = np.array(list(dct.keys()))
        vals = np.array(list(dct.values()))

        leaves = list(map(lambda leave: dct[leave], leaves))
        root = dct[root]

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

    genes = adata.var_names
    if root is None:
        cells = graph["cells_fitted"]
    else:
        df = adata.obs.copy()
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        cells = np.unique(
            np.concatenate(
                list(
                    map(
                        lambda leave: getpath(
                            img, root, graph["tips"], leave, graph, df
                        ).index,
                        leaves,
                    )
                )
            )
        )

    if layer is None:
        if sparse.issparse(adata.X):
            Xgenes = adata[cells, genes].X.A.T.tolist()
        else:
            Xgenes = adata[cells, genes].X.T.tolist()
    else:
        if sparse.issparse(adata.layers[layer]):
            Xgenes = adata[cells, genes].layers[layer].A.T.tolist()
        else:
            Xgenes = adata[cells, genes].layers[layer].T.tolist()

    logg.info("test features for association with the trajectory", reset=True, end="\n")

    stat_assoc_l = list()

    for m in range(n_map):
        data = list(
            zip(
                [adata.uns["pseudotime_list"][str(m)].loc[cells, :]] * len(Xgenes),
                Xgenes,
            )
        )

        stat = Parallel(n_jobs=n_jobs)(
            delayed(gt_fun)(data[d])
            for d in tqdm(
                range(len(data)), file=sys.stdout, desc="    mapping " + str(m)
            )
        )
        stat = pd.DataFrame(stat, index=genes, columns=["p_val", "A"])
        stat["fdr"] = multipletests(stat.p_val, method="bonferroni")[1]
        stat_assoc_l = stat_assoc_l + [stat]

    adata = apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut)

    if mlsc_temp is not None:
        adata.uns["milestones_colors"] = mlsc_temp

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


def gt_fun(data):
    sdf = data[0]
    sdf["exp"] = data[1]

    global rmgcv
    global rstats

    def gamfit(s):
        m = rmgcv.gam(Formula("exp~s(t,k=5)"), data=sdf.loc[sdf["seg"] == s, :])
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


def apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut):
    n_map = len(stat_assoc_l)
    if n_map > 1:
        stat_assoc = pd.DataFrame(
            {
                "p_val": pd.concat(
                    list(map(lambda x: x["p_val"], stat_assoc_l)), axis=1
                ).median(axis=1),
                "A": pd.concat(
                    list(map(lambda x: x["A"], stat_assoc_l)), axis=1
                ).median(axis=1),
                "fdr": pd.concat(
                    list(map(lambda x: x["fdr"], stat_assoc_l)), axis=1
                ).median(axis=1),
                "st": pd.concat(
                    list(
                        map(lambda x: (x.fdr < fdr_cut) & (x.A > A_cut), stat_assoc_l)
                    ),
                    axis=1,
                ).sum(axis=1)
                / n_map,
            }
        )
    else:
        stat_assoc = stat_assoc_l[0]
        stat_assoc["st"] = ((stat_assoc.fdr < fdr_cut) & (stat_assoc.A > A_cut)) * 1

    # saving results
    stat_assoc["signi"] = stat_assoc["st"] > st_cut

    if set(stat_assoc.columns.tolist()).issubset(adata.var.columns):
        adata.var[stat_assoc.columns] = stat_assoc
    else:
        adata.var = pd.concat([adata.var, stat_assoc], axis=1)

    # save all tests for each mapping
    names = np.arange(len(stat_assoc_l)).astype(str).tolist()
    # todict=list(map(lambda x: x.to_dict(),stat_assoc_l))

    # todict=list(map(lambda x: dict(zip(["features"]+x.columns.tolist(),
    #                                   [x.index.tolist()]+x.to_numpy().T.tolist())),
    #                stat_assoc_l))

    dictionary = dict(zip(names, stat_assoc_l))
    adata.uns["stat_assoc_list"] = dictionary

    return adata
