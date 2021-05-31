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
    loess_span=0.2,
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
    loess_span
        fraction of points to take in account for loess fit
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
            [milestones[0] + "\nintra-module", milestones[1] + "\nintra-module"]
            + [milestones[0] + " vs " + milestones[1] + "\ninter-module"],
        )
    )
    logg.info(" done, computing LOESS fit")
    for cc in ["corAA", "corBB", "corAB"]:
        allcor[cc + "_lowess"] = 0
        allcor[cc + "_ll"] = 0
        allcor[cc + "_ul"] = 0
        for r in range(len(runs)):
            for mil in milestones:
                res = allcor.loc[runs[r]].loc[mil]
                l = loess(res.t, res[cc], span=loess_span)
                l.fit()
                pred = l.predict(res.t, stderror=True)
                conf = pred.confidence()

                allcor.loc[(runs[r], mil), cc + "_lowess"] = pred.values
                allcor.loc[(runs[r], mil), cc + "_ll"] = conf.lower
                allcor.loc[(runs[r], mil), cc + "_ul"] = conf.upper

    fork = list(
        set(img.get_shortest_paths(str(root), str(leaves[0]))[0]).intersection(
            img.get_shortest_paths(str(root), str(leaves[1]))[0]
        )
    )
    fork = np.array(img.vs["name"], dtype=int)[fork]
    fork_t = adata.uns["graph"]["pp_info"].loc[fork, "time"].max()
    res = allcor.loc[allcor.t < fork_t, :]
    res = res[~res.t.duplicated()]
    l = loess(res.t, res["corAB"], span=loess_span)
    l.fit()
    pred = l.predict(res.t, stderror=True)

    tval = adata.obs.t.copy()
    tval[tval > fork_t] = np.nan

    def inter_values(tv):
        if ~np.isnan(tv):
            return pred.values[np.argmin(np.abs(res.t.values - tv))]
        else:
            return tv

    adata.obs["inter_cor " + name] = list(map(inter_values, tval))

    df = adata.uns["pseudotime_list"][str(0)]
    cells = np.concatenate(
        [
            getpath(img, root, graph["tips"], leaves[0], graph, df).index,
            getpath(img, root, graph["tips"], leaves[1], graph, df).index,
        ]
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
        + "'], loess fit of inter-module mean local gene-gene correlations prior to bifurcation."
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
    pseudotime_offset: Union["all", float] = "all",
    module: Literal["all", "early"] = "all",
    n_perm: int = 10,
    n_map: int = 1,
    map_cutoff: float = 0.8,
    n_jobs: int = 1,
    alp: int = 10,
    autocor_cut: float = 0.95,
    iterations: int = 15,
    parallel_mode: Literal["window", "mappings"] = "window",
    identify_early_features: bool = True,
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
            geneset = adata.uns[name]["fork"].index[
                adata.uns[name]["fork"].branch == milestone
            ]
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


def critical_transition(
    adata: AnnData,
    root_milestone,
    milestones,
    n_map=1,
    n_jobs=None,
    layer: Optional[str] = None,
    w=100,
    step=30,
    loess_span=0.4,
    gamma=1.5,
    n_points=200,
    copy: bool = False,
):
    """\
    Estimates local critical transition index along the trajectory.

    Based from the concept of pre-bifurcation struture from [Bargaje17]_.
    This study proposes the idea that a signature indicating the flattening
    of the quasi-potential landscape can be detected prior to bifurcation.

    To detect this signal, this function estimates local critical transition
    index along the trajectory, by calculating along a moving window of cell
    the following:

    .. math::
        \\frac{<{\\left | R(g_i,g_j) \\right |>}}{<\\left | R(c_k,c_l) \\right |>}

    Which is the ratio between the mean of the absolute gene by gene correlations
    and the mean of the absolute cell by cell correlations.

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
    w
        local window, in number of cells, to estimate correlations.
    step
        steps, in number of cells, between local windows.
    loess_span
        fraction of points to take in account for loess fit
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata` for a bifurcation:

        `.uns['root_milestone->milestoneA<>milestoneB']['critical transition']`
            containing local critical transition index per window of cells.
        `.obs['root_milestone->milestoneA<>milestoneB pre-fork CI lowess']`
            local critical transition index loess fitted onto cells prior to bifurcation.

    For a linear trajectory:

        `.uns['root_milestone->milestoneA']['critical transition']`
            containing local critical transition index per window of cells.
        `.obs['root_milestone->milestoneA CI lowess']`
            local critical transition index loess fitted onto cells along the path.

    """

    adata = adata.copy() if copy else adata

    logg.info("Calculating local critical transition index", reset=True)

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

    def critical_map(m, gamma, loess_span):
        df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def critical_milestone(leave):
            cells = getpath(img, root, graph["tips"], leave, graph, df).index

            X = get_X(adata, cells, adata.var_names, layer)
            mat = pd.DataFrame(X, index=cells, columns=adata.var_names)

            mat = mat.iloc[adata.obs.t[mat.index].argsort().values, :]

            def slide_path(i):
                cls = mat.index[i : (i + w)]
                cor_gene = mat.loc[cls, :].corr(method="pearson").values
                cor_cell = mat.loc[cls, :].T.corr(method="pearson").values
                R_gene = np.nanmean(
                    np.abs(cor_gene[np.triu_indices(cor_gene.shape[0], k=1)])
                )
                R_cell = np.nanmean(
                    np.abs(cor_cell[np.triu_indices(cor_cell.shape[0], k=1)])
                )
                return [adata.obs.t[cls].mean(), R_gene / R_cell, cls]

            wins = np.arange(0, mat.shape[0] - w, step)

            stats = ProgressParallel(
                n_jobs=n_jobs,
                total=len(wins),
                use_tqdm=n_map == 1,
                file=sys.stdout,
                desc="    to " + str(keys[vals == leave][0]),
            )(delayed(slide_path)(i) for i in wins)

            cells_l = [s[2] for s in stats]
            stats = pd.DataFrame([[s[0], s[1]] for s in stats], columns=("t", "ci"))

            l = loess(stats.t, stats.ci, span=loess_span)
            l.fit()
            pred = l.predict(stats.t, stderror=True)
            conf = pred.confidence()

            stats["lowess"] = pred.values
            stats["ll"] = conf.lower
            stats["ul"] = conf.upper

            cell_stats = [
                pd.DataFrame(
                    np.repeat(stats.ci[i].reshape(-1, 1), len(cells_l[i])),
                    index=cells_l[i],
                    columns=["ci"],
                )
                for i in range(stats.shape[0])
            ]

            cell_stats = pd.concat(cell_stats, axis=1)
            cell_stats = cell_stats.T.groupby(level=0).mean().T
            cell_stats["t"] = adata.obs.loc[cell_stats.index, "t"]

            l = loess(cell_stats.t, cell_stats.ci, span=loess_span)
            pred = l.predict(cell_stats.t, stderror=True)

            cell_stats["fit"] = pred.values

            lspaced_stats = pd.DataFrame(
                {
                    "t": np.linspace(
                        cell_stats["t"].min(), cell_stats["t"].max(), n_points
                    )
                }
            )
            pred = l.predict(lspaced_stats.t, stderror=True)
            lspaced_stats["fit"] = pred.values

            del cell_stats["t"]
            return stats, cell_stats, lspaced_stats

        res = list(map(critical_milestone, leaves))

        cell_stats = pd.concat([r[1] for r in res]).groupby(level=0).mean()

        res_slide = dict(zip(milestones, [r[0] for r in res]))

        res_lspaced = dict(zip(milestones, [r[2] for r in res]))

        return cell_stats, res_slide, res_lspaced

    if n_map == 1:
        df, res_slide, res_lspaced = critical_map(0, gamma, loess_span)
    else:
        # TODO: adapt multimapping
        stats = Parallel(n_jobs=n_jobs)(
            delayed(critical_map)(i)
            for i in tqdm(range(n_map), file=sys.stdout, desc="    multi mapping ")
        )
        res_slides = pd.concat(stats)

    if name in adata.uns:
        adata.uns[name]["critical transition"] = {
            "LOESS": res_slide,
            "eLOESS": res_lspaced,
        }
    else:
        adata.uns[name] = {
            "critical transition": {"LOESS": res_slide, "eLOESS": res_lspaced}
        }

    adata.obs.loc[df.index, name + " CI"] = df.ci.values

    adata.obs.loc[df.index, name + " CI fitted"] = df.fit.values

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['critical transition'], df containing local critical transition index per window of cells.\n"
        "    .obs['"
        + name
        + " CI'], local critical transition index projected onto cells.\n"
        "    .obs['"
        + name
        + " CI fitted'], GAM fit of local critical transition index projected onto cells."
    )

    return adata if copy else None


def criticality_drivers(
    adata: AnnData,
    root_milestone,
    milestones,
    t_span=None,
    confidence_level: float = 0.95,
    layer: Optional[str] = None,
    device="cpu",
    copy: bool = False,
):

    """\
    Calculates correlations between genes and local critical transition index along trajectory.

    Fisher test for the correlations comes from CellRank function `cr.tl.lineages_drivers`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    t_span
        restrict correlations to a window of pseudotime
    confidence_level
        correlation confidence interval.
    layer
        adata layer to use for estimates.
    device
        whether to run the correlation matrix computation on a cpu or gpu.
    loess_span
        fraction of points to take in account for loess fit
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['criticality drivers'].
            a df containing gene correlation with critical index transition.


    """

    adata = adata.copy() if copy else adata

    logg.info("Calculating gene to critical transition index correlations", reset=True)

    name = root_milestone + "->" + "<>".join(milestones)
    obs_name = name + " CI fitted"

    if t_span is None:
        cells = adata.obs_names[~np.isnan(adata.obs[obs_name])]
    else:
        cells = adata.obs_names[
            (~np.isnan(adata.obs[obs_name]))
            & (adata.obs.t > t_span[0])
            & (adata.obs.t < t_span[1])
        ]

    CI = adata[cells].obs[obs_name].values

    if layer is None:
        X = adata[cells].X
    else:
        X = adata[cells].layers[layer]

    if device == "cpu":
        from .utils import cor_mat_cpu

        X = X.A if sparse.issparse(X) else X
        corr = cor_mat_cpu(X, CI.reshape(-1, 1)).ravel()
    else:
        from .utils import cor_mat_gpu
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as csr_gpu

        X = csr_gpu(X) if sparse.issparse(X) else cp.array(X)
        corr = cor_mat_gpu(X, cp.array(CI).reshape(-1, 1)).ravel().get()

    ### Fisher testing of correlations, CellRank implementation
    ### https://github.com/theislab/cellrank/blob/b6345d5e6dd148317782ffc9a9f96793ad98ead9/cellrank/tl/_utils.py#L488
    ### Copyright (c) 2019, Theis Lab

    confidence_level = 0.95
    n = adata.shape[0]
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
    z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

    z = norm.ppf(qh)
    corr_ci_low = np.tanh(mean - z * se)
    corr_ci_high = np.tanh(mean + z * se)
    pvals = 2 * norm.cdf(-np.abs(z_score))

    ###

    res = pd.DataFrame(
        {"corr": corr, "pval": pvals, "ci_low": corr_ci_low, "ci_high": corr_ci_high},
        index=adata.var_names,
    )

    res["q_val"] = np.nan
    res.loc[~np.isnan(pvals), "q_val"] = multipletests(
        res[~np.isnan(pvals)].pval.values, alpha=0.05, method="fdr_bh"
    )[1]

    adata.uns[name]["criticality drivers"] = res.sort_values(
        "corr", ascending=False
    ).dropna()

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['criticality drivers'], df containing gene correlation with critical index transition."
    )

    return adata if copy else None
