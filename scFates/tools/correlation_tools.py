from typing import Union, Optional, List, Iterable
from typing_extensions import Literal

import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
from scipy import sparse
import itertools

from joblib import delayed
from functools import partial
from statsmodels.stats.weightstats import DescrStatsW
from skmisc.loess import loess
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import warnings
from .. import logging as logg
from .. import settings
from .utils import getpath, ProgressParallel, get_X

import sys

sys.setrecursionlimit(10000)

from scFates.tools.utils import importeR

Rpy2, R, rstats, rmgcv, Formula = importeR("Fitting syncho path on cells")


def synchro_path(
    adata: AnnData,
    root_milestone,
    milestones,
    genesetA: Optional[Iterable] = None,
    genesetB: Optional[Iterable] = None,
    n_map=1,
    n_jobs=None,
    layer: Optional[str] = None,
    perm=True,
    w=200,
    step=30,
    winp=10,
    knots=10,
    copy: bool = False,
):
    """\
    Estimates pseudotime trends of local intra- and inter-module correlations of fates-specific modules.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    n_map
        number of probabilistic cells projection to use for estimates.
    n_jobs
        number of cpu processes to perform estimates (per mapping).
    layer
        adata layer to use for estimates.
    perm
        estimate control trends for local permutations instead of real expression matrix.
    w
        local window, in number of cells, to estimate correlations.
    step
        steps, in number of cells, between local windows.
    winp
        window of permutation in cells.
    knots
        number of knots for GAM fit of corAB on cells pre-fork
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['synchro']`
            Dataframe containing mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules.
        `.obs['intercor root_milestone->milestoneA<>milestoneB']`
            loess fit of inter-module mean local gene-gene correlations prior to bifurcation

    """

    adata = adata.copy() if copy else adata

    logg.info("computing local correlations", reset=True)

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    uns_temp = adata.uns.copy()

    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    if genesetA is None:
        bif = adata.uns[name]["fork"]
        genesetA = bif.index[(bif.module == "early") & (bif.branch == milestones[0])]
        genesetB = bif.index[(bif.module == "early") & (bif.branch == milestones[1])]

    genesets = np.concatenate([genesetA, genesetB])

    if n_map == 1:
        logg.info("    single mapping")

    def synchro_map(m):
        if n_map == 1:
            df = adata.obs.loc[:, ["t", "seg"]]
        else:
            df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def synchro_milestone(leave):
            cells = getpath(img, root, graph["tips"], leave, graph, df)
            cells = cells.sort_values("t").index

            X = get_X(adata, cells, genesets, layer)
            mat = pd.DataFrame(X, index=cells, columns=genesets)

            if permut == True:
                winperm = np.min([winp, mat.shape[0]])
                for i in np.arange(0, mat.shape[0] - winperm, winperm):
                    mat.iloc[i : (i + winperm), :] = (
                        mat.iloc[i : (i + winperm), :]
                        .apply(np.random.permutation, axis=0)
                        .values
                    )

            def slide_path(i):
                cls = mat.index[i : (i + w)]
                cor = mat.loc[cls, :].corr(method="spearman")
                corA = cor.loc[:, genesetA].mean(axis=1)
                corB = cor.loc[:, genesetB].mean(axis=1)
                corA[genesetA] = (
                    (corA[genesetA] - 1 / len(genesetA))
                    * len(genesetA)
                    / (len(genesetA) - 1)
                )
                corB[genesetB] = (
                    (corB[genesetB] - 1 / len(genesetB))
                    * len(genesetB)
                    / (len(genesetB) - 1)
                )

                return pd.Series(
                    {
                        "t": adata.obs.t[cls].mean(),
                        "dist": (corA[genesetA].mean() - corA[genesetB].mean()) ** 2
                        + (corB[genesetA].mean() - corB[genesetB].mean()) ** 2,
                        "corAA": corA[genesetA].mean(),
                        "corBB": corB[genesetB].mean(),
                        "corAB": corA[genesetB].mean(),
                        "n_map": m,
                    }
                )

            ww = np.arange(0, mat.shape[0] - w, step)

            res = ProgressParallel(
                n_jobs=n_jobs,
                total=len(ww),
                use_tqdm=n_map == 1,
                file=sys.stdout,
                desc="    to " + str(keys[vals == leave][0]),
            )(delayed(slide_path)(i) for i in ww)

            return pd.concat(res, axis=1).T

        return pd.concat(list(map(synchro_milestone, leaves)), keys=milestones)

    if n_map > 1:
        permut = False
        stats = ProgressParallel(
            n_jobs=n_jobs, total=n_map, file=sys.stdout, desc="    multi mapping"
        )(delayed(synchro_map)(i) for i in range(n_map))
        allcor_r = pd.concat(stats)
        if perm:
            permut = True

            stats = ProgressParallel(
                n_jobs=n_jobs,
                total=n_map,
                file=sys.stdout,
                desc="    multi mapping permutations",
            )(delayed(synchro_map)(i) for i in range(n_map))
            allcor_p = pd.concat(stats)
            allcor = pd.concat([allcor_r, allcor_p], keys=["real", "permuted"])
        else:
            allcor = pd.concat([allcor_r], keys=["real"])
    else:
        permut = False
        allcor_r = pd.concat(list(map(synchro_map, range(n_map))))

        if perm:
            permut = True
            allcor_p = pd.concat(list(map(synchro_map, range(n_map))))
            allcor = pd.concat([allcor_r, allcor_p], keys=["real", "permuted"])
        else:
            allcor = pd.concat([allcor_r], keys=["real"])

    runs = pd.DataFrame(allcor.to_records())["level_0"].unique()

    dct_cormil = dict(
        zip(
            ["corAA", "corBB", "corAB"],
            [m + "\nintra-module" for m in milestones]
            + [" vs ".join(milestones)[0] + "\ninter-module"],
        )
    )
    if len(milestones) > 1:
        fork = list(
            set(img.get_shortest_paths(str(root), str(leaves[0]))[0]).intersection(
                img.get_shortest_paths(str(root), str(leaves[1]))[0]
            )
        )
        fork = np.array(img.vs["name"], dtype=int)[fork]
        fork_t = adata.uns["graph"]["pp_info"].loc[fork, "time"].max()
        res = allcor.loc[allcor.t < fork_t, :]
        res = res[~res.t.duplicated()]
    else:
        res = allcor
        fork_t = res.t.max()

    m = rmgcv.gam(
        Formula("corAB ~ s(t, bs = 'cs',k=%s)" % knots),
        data=res,
    )
    pred = rmgcv.predict_gam(m)

    tval = adata.obs.t.copy()
    tval[tval > fork_t] = np.nan

    def inter_values(tv):
        if ~np.isnan(tv):
            return pred[np.argmin(np.abs(res.t.values - tv))]
        else:
            return tv

    adata.obs["inter_cor " + name] = list(map(inter_values, tval))

    if n_map == 1:
        df = adata.obs.loc[:, ["t", "seg"]]
    else:
        df = adata.uns["pseudotime_list"][str(0)]
    cells = np.concatenate(
        [getpath(img, root, graph["tips"], l, graph, df).index for l in leaves]
    )

    adata.obs.loc[~adata.obs_names.isin(cells), "inter_cor " + name] = np.nan

    adata.uns = uns_temp

    allcor = dict(
        zip(
            allcor.index.levels[0],
            [
                dict(
                    zip(
                        allcor.loc[l1].index.levels[0],
                        [
                            allcor.loc[l1].loc[l2]
                            for l2 in allcor.loc[l1].index.levels[0]
                        ],
                    )
                )
                for l1 in allcor.index.levels[0]
            ],
        )
    )

    if name in adata.uns:
        adata.uns[name]["synchro"] = allcor
    else:
        adata.uns[name] = {"synchro": allcor}

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['synchro'], mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules.\n"
        "    .obs['inter_cor "
        + name
        + "'], GAM fit of inter-module mean local gene-gene correlations prior to bifurcation."
    )

    return adata if copy else None


def synchro_path_multi(
    adata: AnnData, root_milestone, milestones, copy=False, **kwargs
):
    """\
    Wrappers that call `tl.synchro_path` on the pairwise combination of all selected branches.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    kwargs
        arguments to pass to tl.synchro_path.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['synchro']`
            Dataframe containing mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules.
        `.obs['intercor root_milestone->milestoneA<>milestoneB']`
            loess fit of inter-module mean local gene-gene correlations prior to bifurcation

    """

    adata = adata.copy() if copy else adata

    logg.info("computing local correlations", reset=True)

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    uns_temp = adata.uns.copy()

    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    bif = adata.uns[name]["fork"]

    genesets = dict(
        zip(
            milestones,
            [
                bif.index[(bif.module == "early") & (bif.branch == m)]
                for m in milestones
            ],
        )
    )

    pairs = list(itertools.combinations(milestones, 2))

    for m_pair in pairs:
        synchro_path(
            adata,
            root_milestone,
            m_pair,
            genesetA=genesets[m_pair[0]],
            genesetB=genesets[m_pair[1]],
            **kwargs
        )


def module_inclusion(
    adata,
    root_milestone,
    milestones,
    w: int = 300,
    step: int = 30,
    pseudotime_offset: Union["all", float] = 0,
    module: Literal["all", "early"] = "early",
    n_perm: int = 10,
    n_map: int = 1,
    map_cutoff: float = 0.8,
    n_jobs: int = 1,
    alp: int = 10,
    autocor_cut: float = 0.95,
    iterations: int = 15,
    parallel_mode: Literal["window", "mappings"] = "window",
    identify_early_features: bool = False,
    layer=None,
    perm: bool = False,
    winp: int = 10,
    copy: bool = False,
):
    """\
    Estimates the pseudotime onset of a feature within its fate-specific module.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    w
        local window, in number of cells, to estimate correlations.
    step
        steps, in number of cells, between local windows.
    pseudotime_offset
        restrict the cell selection up to a pseudotime offset after the fork
    module
        restrict the gene selection to already classified early genes.
    n_perm
        number of permutations used to estimate the background local correlations.
    n_map
        number of probabilistic cells projection to use for estimates.
    map_cutoff
        proportion of mapping in which inclusion pseudotimne was found for a given feature to keep it.
    n_jobs
        number of cpu processes to perform estimates.
    alp
        parameter regulating stringency of inclusion event.
    autocor_cut
        cutoff on correlation of inclusion times between sequential iterations of the algorithm to stop it.
    iterations
        maximum number of iterations of the algorithm.
    parallel_mode
        whether to run in parallel over the windows of cells or the mappings.
    identify_early_features
        classify a feature as early if its inclusion pseudotime is before the bifurcation
    layer
        adata layer to use for estimates.
    perm
        do local estimates for locally permuted expression matrix.
    winp
        window of permutation in cells.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['module_inclusion']`
            Dataframes ontaining inclusion timing for each gene (rows) in each probabilistic cells projection (columns).
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']`
            Updated with 'inclusion' pseudotime column and 'module column if `identify_early_features=True`'
    """

    adata = adata.copy() if copy else adata

    logg.info("Calculating onset of features within their own module", reset=True)

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    uns_temp = adata.uns.copy()

    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    common_seg = list(
        set.intersection(
            *list(
                map(
                    lambda l: set(img.get_shortest_paths(str(root), str(l))[0]),
                    leaves,
                )
            )
        )
    )
    common_seg = np.array(img.vs["name"], dtype=int)[common_seg]
    fork_t = adata.uns["graph"]["pp_info"].loc[common_seg, "time"].max()

    def onset_map(m):
        if n_map == 1:
            df = adata.obs.loc[:, ["t", "seg"]]
        else:
            df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def onset_milestone(milestone):
            sel = adata.uns[name]["fork"].branch == milestone
            sel = (
                sel & (adata.uns[name]["fork"].module == "early")
                if module == "early"
                else sel
            )
            geneset = adata.uns[name]["fork"].index[sel]
            cells = getpath(
                img,
                root,
                graph["tips"],
                adata.uns["graph"]["milestones"][milestone],
                graph,
                df,
            )

            if pseudotime_offset != "all":
                cells = cells.loc[cells.t < (fork_t + pseudotime_offset)]

            cells = cells.sort_values("t").index

            X = get_X(adata, cells, geneset, layer)
            mat = pd.DataFrame(X, index=cells, columns=geneset)

            if perm:
                winperm = np.min([winp, mat.shape[0]])
                for i in np.arange(0, mat.shape[0] - winperm, winperm):
                    mat.iloc[i : (i + winperm), :] = (
                        mat.iloc[i : (i + winperm), :]
                        .apply(np.random.permutation, axis=0)
                        .values
                    )

            mat = mat.T
            ww = np.arange(0, len(cells) - w, step)
            logW = pd.DataFrame(1, index=geneset, columns=range(len(ww)))

            def slide_cor(i):
                cls0 = mat.columns[ww[i] : min(ww[i] + w - 1, len(cells))]
                mat_1 = mat[cls0]
                mat_2 = mat_1.copy()
                mat_2[mat_2 != 0] = 1
                mat_2 = mat_2.dot(mat_2.T)
                mat_2[mat_2 < 10] = 0
                mat_2[mat_2 >= 10] = 1
                cor = mat_1.T.corr(method="spearman") * mat_2

                def perm_mat():
                    mat_perm = mat[cls0].apply(np.random.permutation, axis=0).values
                    mat_perm = pd.DataFrame(mat_perm, columns=cls0)
                    mat_perm_2 = mat_perm.copy()
                    mat_perm_2[mat_perm_2 != 0] = 1
                    mat_perm_2 = mat_perm_2.dot(mat_perm_2.T)
                    mat_perm_2[mat_perm_2 < 10] = 0
                    mat_perm_2[mat_perm_2 >= 10] = 1
                    return mat_perm.T.corr(method="spearman") * mat_perm_2

                allperm = [perm_mat() for i in range(n_perm)]

                return cor, allperm

            res = ProgressParallel(
                n_jobs=n_jobs,
                total=len(ww),
                use_tqdm=n_map == 1,
                file=sys.stdout,
                desc="    to " + milestone,
            )(delayed(slide_cor)(i) for i in range(len(ww)))

            cors = [r[0] for r in res]
            cors_p = [r[1] for r in res]

            def corEvol(cors, cors_p, logW):
                cor_Ps = []
                for i in range(len(cors)):
                    cor, cor_p = cors[i], cors_p[i]
                    corTrue = (cor.T * logW.iloc[:, i]).mean(axis=0)
                    cor_control = [
                        (cor_p.T.values * logW.iloc[:, i].values).mean(axis=0)
                        for cor_p in cor_p
                    ]
                    cor_Ps = cor_Ps + [
                        np.array(
                            [
                                sum(
                                    cor_c + np.random.uniform(0, 0.01, len(cor_c))
                                    >= corTrue
                                    for cor_c in cor_control
                                )
                            ]
                        )
                        / len(cor_control)
                    ]
                return cor_Ps

            logList = []
            autocor = 0

            def switch_point(r, alp):
                def logL(j):
                    v = 0
                    if j >= 2:
                        v = v
                    if j <= len(r):
                        v = v + sum(
                            -alp * r[j : len(r)] + np.log(alp / (1 - np.exp(-alp)))
                        )

                    return v

                switch_point = np.argmax([logL(j) for j in range(len(r) + 1)])
                n = len([logL(j) for j in range(len(r) + 1)])
                return np.concatenate(
                    [np.repeat(0, switch_point), np.repeat(1, n - switch_point - 1)]
                )

            i = 0
            auto_cor = 0
            while (autocor < autocor_cut) & (i <= iterations):
                pMat = np.vstack(corEvol(cors, cors_p, logW)).T
                sws = [switch_point(pMat[i, :], alp) for i in range(pMat.shape[0])]
                sws = np.vstack(sws)
                logW = pd.DataFrame(sws, index=geneset)
                logList = logList + [logW]

                if i > 0:
                    autocor = np.corrcoef(
                        logList[i - 1].idxmax(axis=1), logList[i].idxmax(axis=1)
                    )[1][0]

                i = i + 1

            incl_t = pd.Series(np.nan, index=logW.index)
            logW = logW.loc[logW.sum(axis=1) != 0]

            incl_t[logW.index] = [
                adata.obs.t[cells[(pos * (step - 1)) : (pos * (step - 1) + w)]].mean()
                for pos in logW.idxmax(axis=1)
            ]
            return incl_t

        return [onset_milestone(milestone) for milestone in milestones]

    n_jobs_map = 1
    if parallel_mode == "mappings":
        n_jobs_map = n_jobs
        n_jobs = 1

    stats = ProgressParallel(
        n_jobs=n_jobs_map,
        total=n_map,
        use_tqdm=n_map > 1,
        file=sys.stdout,
        desc="    multi mapping",
    )(delayed(onset_map)(i) for i in range(n_map))

    matSwitch = dict(
        zip(
            milestones,
            [
                pd.concat(
                    [s[i] for s in stats],
                    axis=1,
                    keys=np.arange(len([s[i] for s in stats])).astype(str),
                )
                for i in range(len(leaves))
            ],
        )
    )

    perm_str = "_perm" if perm else ""
    if perm:
        identify_early_features = False

    for m in milestones:
        props = 1 - np.isnan(matSwitch[m]).sum(axis=1) / n_map
        matSwitch[m] = matSwitch[m].loc[props > map_cutoff]

    adata.uns[name]["fork"]["props_incl"] = np.nan
    adata.uns[name]["fork"].loc[props.index, "props_incl"] = props.values

    adata.uns[name]["module_inclusion" + perm_str] = matSwitch

    updated = "."
    if perm == False:
        common_seg = list(
            set.intersection(
                *list(
                    map(
                        lambda l: set(img.get_shortest_paths(str(root), str(l))[0]),
                        leaves,
                    )
                )
            )
        )
        common_seg = np.array(img.vs["name"], dtype=int)[common_seg]
        fork_t = adata.uns["graph"]["pp_info"].loc[common_seg, "time"].max()

        g_early = []
        dfs = []
        for i in range(len(leaves)):
            included = pd.concat([s[i] for s in stats], axis=1).mean(axis=1)
            df = included[~np.isnan(included)].sort_values()
            adata.uns[name]["fork"].loc[df.index, "inclusion"] = df.values
            g_early = g_early + [df.index[df < fork_t]]
            dfs = dfs + [df]

        if identify_early_features:
            updated = " and 'module'."
            adata.uns[name]["fork"]["module"] = np.nan
            for df, g_e in zip(dfs, g_early):
                adata.uns[name]["fork"].loc[df.index, "module"] = "late"
                adata.uns[name]["fork"].loc[g_e, "module"] = "early"

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['module_inclusion"
        + perm_str
        + "'], milestone specific dataframes containing inclusion timing for each gene in each probabilistic cells projection.\n"
        + "    .uns['"
        + name
        + "']['fork'] has been updated with the column 'inclusion'"
        + updated
    )

    return adata if copy else None
