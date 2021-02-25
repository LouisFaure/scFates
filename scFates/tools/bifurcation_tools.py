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

from scipy import sparse

from joblib import delayed, Parallel
from tqdm import tqdm

from .. import logging as logg
from .. import settings
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
    warnings.warn(e.__doc__)
    warnings.warn(e.message)


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
    Test for branch differential gene expression, and differential upregulation after bifurcation point.

    First, differential gene expression between two branches is performed. Then,
    the feature are tested to intify the ones with higher average expression
    in one of the derivative branches compared to the progenitor branch.


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

    adata = adata.copy() if copy else adata

    genes = adata.var_names if features is None else features

    logg.info("testing fork", reset=True)

    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    dct = graph["milestones"]

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

    for m in range(n_map):
        logg.info("    mapping: " + str(m))
        ## Diff expr between forks

        df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def get_branches(i):
            x = vpath[i][1:]
            segs = graph["pp_info"].loc[x, :].seg.unique()
            df_sub = df.loc[df.seg.isin(segs), :].copy(deep=True)
            df_sub.loc[:, "i"] = i
            return df_sub

        brcells = pd.concat(
            list(map(get_branches, range(len(vpath)))), axis=0, sort=False
        )
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

        if layer is None:
            if sparse.issparse(adata.X):
                Xgenes = adata[brcells.index, genes].X.A.T.tolist()
            else:
                Xgenes = adata[brcells.index, genes].X.T.tolist()
        else:
            if sparse.issparse(adata.layers[layer]):
                Xgenes = adata[brcells.index, genes].layers[layer].A.T.tolist()
            else:
                Xgenes = adata[brcells.index, genes].layers[layer].T.tolist()

        data = list(zip([brcells] * len(Xgenes), Xgenes))

        stat = Parallel(n_jobs=n_jobs)(
            delayed(gt_fun)(data[d])
            for d in tqdm(
                range(len(data)), file=sys.stdout, desc="    differential expression"
            )
        )

        fork_stat = fork_stat + [
            pd.DataFrame(stat, index=genes, columns=milestones + ["de_p"])
        ]

        topleave = fork_stat[m].iloc[:, :-1].idxmax(axis=1).apply(lambda mil: dct[mil])

        ## test for upregulation
        logg.info("    test for upregulation for each leave vs root")
        leaves_stat = list()
        for leave in leaves:
            subtree = getpath(img, root, graph["tips"], leave, graph, df).sort_values(
                "t"
            )

            topgenes = topleave[topleave == leave].index

            if layer is None:
                if sparse.issparse(adata.X):
                    Xgenes = adata[subtree.index, topgenes].X.A.T.tolist()
                else:
                    Xgenes = adata[subtree.index, topgenes].X.T.tolist()
            else:
                if sparse.issparse(adata.layers[layer]):
                    Xgenes = adata[subtree.index, topgenes].layers[layer].A.T.tolist()
                else:
                    Xgenes = adata[subtree.index, topgenes].layers[layer].T.tolist()

            data = list(zip([subtree] * len(Xgenes), Xgenes))

            stat = Parallel(n_jobs=n_jobs)(
                delayed(test_upreg)(data[d])
                for d in tqdm(
                    range(len(data)),
                    file=sys.stdout,
                    desc="    leave " + str(keys[vals == leave][0]),
                )
            )
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
    global rstats

    def gamfit(sdf):
        m = rmgcv.gam(
            Formula("exp ~ s(t)+s(t,by=as.factor(i))+as.factor(i)"),
            data=sdf,
            weights=sdf["w"],
        )
        return rmgcv.summary_gam(m)[3][1]

    Amps = sdf.groupby("i").apply(lambda x: np.mean(x.exp))
    return (Amps - Amps.max()).tolist() + [gamfit(sdf)]


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


def activation(
    adata: AnnData,
    root_milestone,
    milestones,
    deriv_cut: float = 0.15,
    pseudotime_offset: float = 0,
    nwin: int = 20,
    steps: int = 5,
    n_map: int = 1,
    copy: bool = False,
    n_jobs=-1,
    layer: Optional[str] = None,
):

    """\
    Identify pseudotime of activation of branc-specififc features.

    This aims in classifying the genes according to their their activation timing
    compared to the pseudotime of the bifurcation. Any feature activated before the
    bifurcation is considered as 'early', others are considered 'late'.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    deriv_cut
        a first passage of derivative at this cutoff (in proportion to the full dynamic range of the fitted feature) is considered as activation timing
    pseudotime_offset
        consider a feature as early if it gets activated before: pseudotime at bifurcation-pseudotime_offset.
    nwin
        windows of pseudotime to use for assessing activation timimg
    steps
        number of steps dividing a window for that will slide along the pseudotime
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

        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['module']`
            classify feature as 'early' or 'late'.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['activation']`
            pseudotime of activationh.

    """

    graph = adata.uns["graph"]

    logg.info("testing activation", reset=True)

    uns_temp = adata.uns.copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)
    stats = adata.uns[name]["fork"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    allact = []

    for m in range(n_map):

        df = adata.uns["pseudotime_list"][str(m)]
        acti = pd.Series(0, index=stats.index)

        for leave in leaves:
            subtree = getpath(img, root, graph["tips"], leave, graph, df).sort_values(
                "t"
            )
            del subtree["branch"]
            subtree["deriv_cut"] = deriv_cut
            subtree["nwin"] = nwin
            subtree["steps"] = steps

            genes = stats.index[stats["branch"] == str(keys[vals == leave][0])]

            wf = warnings.filters.copy()
            warnings.filterwarnings("ignore")
            if layer is None:
                if sparse.issparse(adata.X):
                    Xgenes = adata[subtree.index, genes].X.A.T.tolist()
                else:
                    Xgenes = adata[subtree.index, genes].X.T.tolist()
            else:
                if sparse.issparse(adata.layers[layer]):
                    Xgenes = adata[subtree.index, genes].layers[layer].A.T.tolist()
                else:
                    Xgenes = adata[subtree.index, genes].layers[layer].T.tolist()
            warnings.filters = wf

            data = list(zip([subtree] * len(Xgenes), Xgenes))

            acti.loc[genes] = Parallel(n_jobs=n_jobs)(
                delayed(get_activation)(data[d])
                for d in tqdm(
                    range(len(data)),
                    file=sys.stdout,
                    desc="    leave " + str(keys[vals == leave][0]),
                )
            )

        allact = allact + [acti]

    stats["activation"] = pd.concat(allact, axis=1).median(axis=1).values

    common_seg = list(
        set.intersection(
            *list(
                map(lambda l: set(img.get_shortest_paths(str(root), str(l))[0]), leaves)
            )
        )
    )
    common_seg = np.array(img.vs["name"], dtype=int)[common_seg]
    fork_t = (
        adata.uns["graph"]["pp_info"].loc[common_seg, "time"].max() - pseudotime_offset
    )

    logg.info("    threshold pseudotime is: " + str(fork_t))

    stats["module"] = "early"
    stats.loc[stats["activation"] > fork_t, "module"] = "late"

    adata.uns = uns_temp

    adata.uns[name]["fork"] = stats

    for l in leaves:
        c_early = np.sum(
            (stats.branch == str(keys[vals == l][0])) & (stats.module == "early")
        )
        c_late = np.sum(
            (stats.branch == str(keys[vals == l][0])) & (stats.module == "late")
        )
        logg.info(
            "    "
            + str(c_early)
            + " early and "
            + str(c_late)
            + " late features specific to leave "
            + str(keys[vals == l][0])
        )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    .uns['"
        + name
        + "']['fork'], DataFrame updated with additionnal 'activation' and 'module' columns."
    )

    return adata if copy else None


def get_activation(data):
    global rmgcv
    subtree = data[0]
    subtree["exp"] = data[1]
    # load parameters
    deriv_cut = subtree["deriv_cut"][0]
    nwin = subtree["nwin"][0]
    steps = subtree["steps"][0]

    wf = warnings.filters.copy()
    warnings.filterwarnings("ignore")

    def gamfit(sdf):
        m = rmgcv.gam(Formula("exp ~ s(t)"), data=sdf, gamma=1)
        return rmgcv.predict_gam(m)

    subtree["fitted"] = gamfit(subtree)

    window = (subtree.t.max() - subtree.t.min()) / nwin
    step = window / steps

    wins = pd.DataFrame(
        {
            "start": [step * i for i in range(nwin * steps)] + subtree.t.min(),
            "end": [step * i for i in range(nwin * steps)] + subtree.t.min() + window,
        }
    )
    wins = wins.loc[wins.end < subtree.t.max()]

    df_t = subtree.t.sort_values()

    fitted = subtree.fitted
    fitted = (fitted - fitted.min()) / (fitted.max() - fitted.min())
    fitted = fitted[df_t.index]
    changes = wins.apply(
        lambda x: (fitted.loc[df_t[(df_t > x[0]) & (df_t < x[1])].index]).diff().sum(),
        axis=1,
    )

    if sum(changes > deriv_cut) == 0:
        act = subtree.t.max() + 1
    else:
        act = wins.mean(axis=1)[changes > deriv_cut].min()
    warnings.filters = wf

    return act
