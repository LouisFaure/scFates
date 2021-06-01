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
import igraph
import warnings
from functools import reduce
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as sm

from joblib import delayed
from tqdm import tqdm

from .. import logging as logg
from .. import settings
from .utils import getpath, ProgressParallel, get_X, importeR

Rpy2, R, rstats, rmgcv, Formula = importeR("performing bifurcation analysis")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]


def test_fork(
    adata: AnnData,
    root_milestone,
    milestones,
    features: Union[None, str] = None,
    rescale: bool = False,
    layer: Optional[str] = None,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False,
):

    """\
    Test for branch differential gene expression and differential upregulation from progenitor to terminal state.

    First, differential gene expression between two branches is performed. This is performed by
    comparing the full model :math:`g_{i} \\sim\ pseudotime*Branch` to the reduced one
    :math:`g_{i} \\sim\ pseudotime` using F-test.

    Then, each feature is tested for its upregulation along the path from progenitor to terminal state,
    using the linear model :math:`g_{i} \sim\ pseudotime`.


    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    features
        Which features to test (all by default).
    rescale
        By default, analysis restrict to only cells having a pseudotime lower than the shortest branch maximum pseudotime, this can be avoided by rescaling the post bifurcation pseudotime of both branches to 1.
    layer
        layer to use for the test
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['fork']`
            DataFrame with fork test results.

    """

    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, rmgcv, Formula])[idx])

    adata = adata.copy() if copy else adata

    genes = adata.var_names if features is None else features

    logg.info("testing fork", reset=True)

    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    dct = graph["milestones"]
    dct_rev = dict(zip(dct.values(), dct.keys()))

    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    g = igraph.Graph.Adjacency((graph["B"] > 0).tolist(), mode="undirected")
    # Add edge weights and node labels.
    g.es["weight"] = graph["B"][graph["B"].nonzero()]

    vpath = g.get_shortest_paths(root, leaves)
    interPP = list(set(vpath[0]) & set(vpath[1]))
    vpath = g.get_shortest_paths(graph["pp_info"].loc[interPP, :].time.idxmax(), leaves)

    fork_stat = list()
    upreg_stat = list()

    for m in tqdm(
        range(n_map), disable=n_map == 1, file=sys.stdout, desc="    multi mapping "
    ):

        if n_map == 1:
            logg.info("    single mapping")
        ## Diff expr between forks

        df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def get_branches(m):
            leave = adata.uns["graph"]["milestones"][m]
            ddf = getpath(img, root, adata.uns["graph"]["tips"], leave, graph, df)
            ddf["i"] = adata.uns["graph"]["milestones"][m]
            return ddf

        brcells = pd.concat([get_branches(m) for m in milestones])
        brcells = brcells.loc[brcells.index.drop_duplicates(False)]

        matw = None
        if matw is None:
            brcells["w"] = 1
        else:
            brcells["w"] = matw[gene, :][:, graph["cells_fitted"]]

        brcells.drop(["seg", "edge"], axis=1, inplace=True)

        if rescale:
            for i in brcells.i.unique():
                brcells.loc[brcells.i == i, "t"] = (
                    brcells.loc[brcells.i == i].t - brcells.loc[brcells.i == i].t.min()
                ) / (
                    brcells.loc[brcells.i == i].t.max()
                    - brcells.loc[brcells.i == i].t.min()
                )

        Xgenes = get_X(adata, brcells.index, genes, layer, togenelist=True)

        data = list(zip([brcells] * len(Xgenes), Xgenes))

        stat = ProgressParallel(
            n_jobs=n_jobs,
            use_tqdm=n_map == 1,
            total=len(data),
            file=sys.stdout,
            desc="    Differential expression",
        )(delayed(gt_fun)(data[d]) for d in range(len(data)))

        stat = pd.concat(stat, axis=1).T
        stat.index = genes
        stat.columns = [dct_rev[c] for c in stat.columns[: len(milestones)]] + ["de_p"]

        fork_stat = fork_stat + [stat]

        topleave = fork_stat[m].iloc[:, :-1].idxmax(axis=1).apply(lambda mil: dct[mil])

        ## test for upregulation
        logg.info("    test for upregulation for each leave vs root")
        leaves_stat = list()
        for leave in leaves:
            subtree = getpath(img, root, graph["tips"], leave, graph, df).sort_values(
                "t"
            )

            topgenes = topleave[topleave == leave].index

            Xgenes = get_X(adata, subtree.index, topgenes, layer, togenelist=True)

            data = list(zip([subtree] * len(Xgenes), Xgenes))

            stat = ProgressParallel(
                n_jobs=n_jobs,
                use_tqdm=n_map == 1,
                total=len(data),
                file=sys.stdout,
                desc="    upreg " + str(keys[vals == leave][0]),
            )(delayed(test_upreg)(data[d]) for d in range(len(data)))

            stat = pd.DataFrame(stat, index=topgenes, columns=["up_A", "up_p"])
            leaves_stat = leaves_stat + [stat]

        upreg_stat = upreg_stat + [pd.concat(leaves_stat).loc[genes]]

    # summarize fork statistics
    # fork_stat=list(map(lambda x: pd.DataFrame(x,index=genes,columns=["effect","p_val"]),fork_stat))

    fdr_l = list(
        map(
            lambda x: pd.Series(
                multipletests(x.de_p, method="bonferroni")[1], index=x.index, name="fdr"
            ),
            fork_stat,
        )
    )

    st_l = list(
        map(
            lambda x: pd.Series(
                (x.de_p < 5e-2).values * 1, index=x.index, name="signi_p"
            ),
            fork_stat,
        )
    )
    stf_l = list(
        map(
            lambda x: pd.Series((x < 5e-2).values * 1, index=x.index, name="signi_fdr"),
            fdr_l,
        )
    )

    fork_stat = list(
        map(
            lambda w, x, y, z: pd.concat([w, x, y, z], axis=1),
            fork_stat,
            fdr_l,
            st_l,
            stf_l,
        )
    )

    effect = list(
        map(
            lambda i: pd.concat(
                list(map(lambda x: x.iloc[:, i], fork_stat)), axis=1
            ).median(axis=1),
            range(len(milestones)),
        )
    )
    p_val = pd.concat(list(map(lambda x: x.de_p, fork_stat)), axis=1).median(axis=1)
    fdr = pd.concat(list(map(lambda x: x.fdr, fork_stat)), axis=1).median(axis=1)
    signi_p = pd.concat(list(map(lambda x: x.signi_p, fork_stat)), axis=1).mean(axis=1)
    signi_fdr = pd.concat(list(map(lambda x: x.signi_fdr, fork_stat)), axis=1).mean(
        axis=1
    )

    colnames = fork_stat[0].columns
    fork_stat = pd.concat(effect + [p_val, fdr, signi_p, signi_fdr], axis=1)
    fork_stat.columns = colnames

    # summarize upregulation stats
    colnames = upreg_stat[0].columns
    upreg_stat = pd.concat(
        list(
            map(
                lambda i: pd.concat(
                    list(map(lambda x: x.iloc[:, i], upreg_stat)), axis=1
                ).median(axis=1),
                range(2),
            )
        ),
        axis=1,
    )
    upreg_stat.columns = colnames

    summary_stat = pd.concat([fork_stat, upreg_stat], axis=1)
    adata.uns = uns_temp

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    name = root_milestone + "->" + "<>".join(milestones)

    # adata.uns[name]["fork"] = summary_stat

    adata.uns[name] = {"fork": summary_stat}
    logg.hint(
        "added \n" "    .uns['" + name + "']['fork'], DataFrame with fork test results."
    )

    return adata if copy else None


def gt_fun(data):

    sdf = data[0]
    sdf["exp"] = data[1]

    ## add weighted matrix part
    #
    #

    global rmgcv

    m = rmgcv.gam(
        Formula("exp ~ s(t)+s(t,by=as.factor(i))+as.factor(i)"),
        data=sdf,
        weights=sdf["w"],
    )

    Amps = sdf.groupby("i").apply(lambda x: np.mean(x.exp))
    res = Amps - Amps.max()
    res["de_p"] = rmgcv.summary_gam(m)[3][1]

    return res


def test_upreg(data):

    sdf = data[0]
    sdf["exp"] = data[1]

    result = sm.ols(formula="exp ~ t", data=sdf).fit()
    return [result.params["t"], result.pvalues["t"]]


def branch_specific(
    adata: AnnData,
    root_milestone,
    milestones,
    effect: float = None,
    stf_cut: float = 0.7,
    up_A: float = 0,
    up_p: float = 5e-2,
    copy: bool = False,
):

    """\
    Assign genes differentially expressed between two post-bifurcation branches.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    effect
        minimum expression differences to call gene as differentially upregulated.
    stf_cut
        fraction of projections when gene passed fdr < 0.05.
    up_A
        minimum expression increase at derivative compared to progenitor branches to call gene as branch-specific.
    up_p
        p-value of expression changes of derivative compared to progenitor branches to call gene as branch-specific.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['branch']`
            assigned branch.

    """

    adata = adata.copy() if copy else adata

    uns_temp = adata.uns.copy()

    name = root_milestone + "->" + "<>".join(milestones)

    df = adata.uns[name]["fork"]

    df = df[(df.up_A > up_A) & (df.up_p < up_p) & (df.signi_fdr > stf_cut)]
    df = df[((df.iloc[:, : len(milestones)] + effect) > 0).sum(axis=1) == 1]
    df["branch"] = df.iloc[:, : len(milestones)].idxmax(axis=1)

    logg.info(
        "    "
        + "branch specific features: "
        + ", ".join(
            [
                ": ".join(st)
                for st in list(
                    zip(
                        df.value_counts("branch").index,
                        df.value_counts("branch").values.astype(str),
                    )
                )
            ]
        )
    )

    adata.uns = uns_temp

    adata.uns[name]["fork"] = df

    logg.info("    finished", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    .uns['"
        + name
        + "']['fork'], DataFrame updated with additionnal 'branch' column."
    )

    return adata if copy else None
