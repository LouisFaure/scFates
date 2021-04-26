from typing import Union, Optional, List

import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
from scipy import sparse

from joblib import delayed, Parallel
from tqdm import tqdm
from functools import partial
from statsmodels.stats.weightstats import DescrStatsW
from skmisc.loess import loess
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import warnings
from .. import logging as logg
from .. import settings
from .utils import getpath

import sys

sys.setrecursionlimit(10000)

from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
import rpy2.rinterface
pandas2ri.activate()

rmgcv = importr("mgcv")

def slide_cells(
    adata: AnnData,
    root_milestone,
    milestones,
    win: int = 50,
    mapping: bool = True,
    copy: bool = False,
    ext: bool = False,
):

    """\
    Assign cells in a probabilistic manner to non-intersecting windows along pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    win
        number of cell per local pseudotime window.
    mapping
        project cells onto tree pseudotime in a probabilistic manner.
    copy
        Return a copy instead of writing to adata.
    ext
        Output the list externally instead of writting to anndata

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['cell_freq']`
            List of np.array containing probability assignment of cells on non intersecting windows.

    """

    adata = adata.copy() if copy else adata

    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    # weird fix to avoid loss of milestone colors...
    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    def getsegs(g, root, leave, graph):
        path = np.array(g.vs[:]["name"])[
            np.array(g.get_shortest_paths(str(root), str(leave)))
        ][0]
        segs = list()
        for i in range(len(path) - 1):
            segs = segs + [
                np.argwhere(
                    (
                        graph["pp_seg"][["from", "to"]]
                        .astype(str)
                        .apply(lambda x: all(x.values == path[[i, i + 1]]), axis=1)
                    ).to_numpy()
                )[0][0]
            ]
        segs = graph["pp_seg"].index[segs].tolist()
        return segs

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    paths = list(map(lambda l: getsegs(img, root, l, graph), leaves))
    pp_probs = graph["R"].sum(axis=0)

    if len(milestones) == 2:
        seg_progenies = list(set.intersection(*[set(path) for path in paths]))
        seg_branch1 = list(set.difference(set(paths[0]), set(seg_progenies)))
        seg_branch2 = list(set.difference(set(paths[1]), set(seg_progenies)))

        pps = (
            graph["pp_info"]
            .PP[
                graph["pp_info"].seg.isin(
                    np.array(seg_progenies + seg_branch1 + seg_branch2).astype(str)
                )
            ]
            .index
        )

        seg_branch1 = [str(seg) for seg in seg_branch1]
        seg_branch2 = [str(seg) for seg in seg_branch2]
        seg_progenies = [str(seg) for seg in seg_progenies]
        segs_cur = np.unique(
            np.array(seg_progenies + seg_branch1 + seg_branch2).flatten().astype(str)
        )

    elif len(milestones) == 1:
        paths = [str(p[0]) for p in paths]

        pps = graph["pp_info"].seg.isin(paths).index

        seg_progenies = list(set.intersection(*[set(path) for path in paths]))
        seg_progenies = [str(seg) for seg in seg_progenies]
        segs_cur = seg_progenies

    def region_extract(pt_cur, segs_cur, nbranch):
        freq = list()

        pp_next = pps[
            (graph["pp_info"].loc[pps, "time"].values >= pt_cur)
            & graph["pp_info"].loc[pps, "seg"].isin(segs_cur).values
        ]

        cmsm = np.cumsum(
            pp_probs[pp_next][np.argsort(graph["pp_info"].loc[pp_next, "time"].values)]
        )
        inds = np.argwhere(cmsm > win).flatten()

        if len(inds) == 0:
            if cmsm.max() > win / 2:
                if mapping:
                    cell_probs = graph["R"][:, pp_next].sum(axis=1)
                else:
                    cell_probs = (
                        np.isin(
                            np.apply_along_axis(
                                lambda x: np.argmax(x), axis=1, arr=graph["R"]
                            ),
                            pp_next,
                        )
                        * 1
                    )
                freq = freq + [cell_probs]
            return freq
        else:
            pps_region = pp_next[
                np.argsort(graph["pp_info"].loc[pp_next, "time"].values)
            ][: inds[0]]
            if mapping:
                cell_probs = graph["R"][:, pps_region].sum(axis=1)
            else:
                cell_probs = (
                    np.isin(
                        np.apply_along_axis(
                            lambda x: np.argmax(x), axis=1, arr=graph["R"]
                        ),
                        pps_region,
                    )
                    * 1
                )

            freq = freq + [cell_probs]
            pt_cur = graph["pp_info"].loc[pps_region, "time"].max()

            if nbranch == 1:
                if (
                    sum(~graph["pp_info"].loc[pps_region, :].seg.isin(seg_progenies))
                    == 0
                ):
                    res = region_extract(pt_cur, segs_cur, nbranch)
                    return freq + res

            if nbranch == 2:
                if (
                    sum(~graph["pp_info"].loc[pps_region, :].seg.isin(seg_progenies))
                    == 0
                ):
                    res = region_extract(pt_cur, segs_cur, nbranch)
                    return freq + res

                elif (
                    sum(~graph["pp_info"].loc[pps_region, :].seg.isin(seg_branch1)) == 0
                ):
                    res = region_extract(pt_cur, segs_cur, nbranch)
                    return freq + res

                elif (
                    sum(~graph["pp_info"].loc[pps_region, :].seg.isin(seg_branch2)) == 0
                ):

                    res = region_extract(pt_cur, segs_cur, nbranch)
                    return freq + res

                elif ~(
                    sum(
                        ~graph["pp_info"]
                        .loc[pps_region, :]
                        .seg.isin([str(seg) for seg in seg_progenies])
                    )
                    == 0
                ):
                    pt_cur1 = (
                        graph["pp_info"]
                        .loc[pps_region, "time"][
                            graph["pp_info"]
                            .loc[pps_region, "seg"]
                            .isin([str(seg) for seg in seg_branch1])
                        ]
                        .max()
                    )
                    segs_cur1 = seg_branch1
                    pt_cur2 = (
                        graph["pp_info"]
                        .loc[pps_region, "time"][
                            graph["pp_info"]
                            .loc[pps_region, "seg"]
                            .isin([str(seg) for seg in seg_branch2])
                        ]
                        .max()
                    )
                    segs_cur2 = seg_branch2
                    res1 = region_extract(pt_cur1, segs_cur1, nbranch)
                    res2 = region_extract(pt_cur2, segs_cur2, nbranch)
                    return freq + res1 + res2

    pt_cur = graph["pp_info"].loc[pps, "time"].min()

    freq = region_extract(pt_cur, segs_cur, len(milestones))
    name = root_milestone + "->" + "<>".join(milestones)

    adata.uns = uns_temp

    freqs = list(
        map(lambda f: pd.Series(f, index=adata.uns["graph"]["cells_fitted"]), freq)
    )

    if ext is False:
        if name in adata.uns:
            adata.uns[name]["cell_freq"] = freqs
        else:
            adata.uns[name] = {"cell_freq": freqs}
        logg.hint(
            "added \n"
            "    .uns['"
            + name
            + "']['cell_freq'], probability assignment of cells on "
            + str(len(freq))
            + " non intersecting windows."
        )

    if copy:
        return adata
    elif ext:
        return freqs
    else:
        None


def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones: List,
    genesetA=None,
    genesetB=None,
    layer: Optional[str] = None,
    copy: bool = False,
):
    """\
    Obtain gene module correlations in the non-intersecting windows along pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    layer
        adata layer from which to compute the correlations.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['corAB']`
            Dataframe containing gene-gene correlation modules.

    """

    adata = adata.copy() if copy else adata

    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    # weird fix to avoid loss of milestone colors...
    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    if (genesetA is None or genesetB is None) and len(milestones) == 1:
        raise ValueError(
            "You need two list of genes when a non-bifurcating trajectory is analysed!"
        )

    if genesetA is None or genesetB is None:
        bif = adata.uns[name]["fork"]
    freqs = adata.uns[name]["cell_freq"]
    nwin = len(freqs)

    if genesetA is None:
        genesetA = bif.index[
            (bif["branch"] == milestones[0]).values & (bif["module"] == "early").values
        ]
    if genesetB is None:
        genesetB = bif.index[
            (bif["branch"] == milestones[1]).values & (bif["module"] == "early").values
        ]
    genesets = np.concatenate([genesetA, genesetB])

    if layer is None:
        if sparse.issparse(adata.X):
            X = adata[:, genesets].X.A
        else:
            X = adata[:, genesets].X
    else:
        if sparse.issparse(adata.layers[layer]):
            X = adata[:, genesets].layers[layer].A
        else:
            X = adata[:, genesets].layers[layer]

    X = pd.DataFrame(X, index=adata.obs_names, columns=genesets)
    X_r = X.rank(axis=0)

    def gather_cor(i, geneset):
        freq = freqs[i][adata.obs_names]
        cormat = pd.DataFrame(
            DescrStatsW(X_r.values, weights=freq).corrcoef,
            index=genesets,
            columns=genesets,
        )
        np.fill_diagonal(cormat.values, np.nan)
        return cormat.loc[:, geneset].mean(axis=1)

    gather = partial(gather_cor, geneset=genesetA)
    corA = pd.concat(list(map(gather, range(nwin))), axis=1)

    gather = partial(gather_cor, geneset=genesetB)
    corB = pd.concat(list(map(gather, range(nwin))), axis=1)

    groups = ["A", "B"] if len(milestones) == 1 else milestones

    corAB = pd.concat([corA, corB], keys=groups)
    corAB.columns = [str(c) for c in corAB.columns]

    adata.uns = uns_temp
    adata.uns[name]["corAB"] = corAB

    logg.hint(
        "added \n" "    .uns['" + name + "']['corAB'], gene-gene correlation modules."
    )

    return adata if copy else None


def synchro_path(
    adata: AnnData,
    root_milestone,
    milestones,
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

    bif = adata.uns[name]["fork"]

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

        genesetA = bif.index[(bif.module == "early") & (bif.branch == milestones[0])]
        genesetB = bif.index[(bif.module == "early") & (bif.branch == milestones[1])]
        genesets = np.concatenate([genesetA, genesetB])

        def synchro_milestone(leave):
            cells = getpath(img, root, graph["tips"], leave, graph, df).index

            if layer is None:
                if sparse.issparse(adata.X):
                    mat = pd.DataFrame(
                        adata[cells, genesets].X.A, index=cells, columns=genesets
                    )
                else:
                    mat = pd.DataFrame(
                        adata[cells, genesets].X, index=cells, columns=genesets
                    )
            else:
                if sparse.issparse(adata.layers[layer]):
                    mat = pd.DataFrame(
                        adata[cells, genesets].layers[layer].A,
                        index=cells,
                        columns=genesets,
                    )
                else:
                    mat = pd.DataFrame(
                        adata[cells, genesets].layers[layer],
                        index=cells,
                        columns=genesets,
                    )

            mat = mat.iloc[adata.obs.t[mat.index].argsort().values, :]

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

            return pd.concat(
                list(
                    map(
                        slide_path,
                        tqdm(
                            np.arange(0, mat.shape[0] - w, step),
                            disable=n_map > 1,
                            desc="    leave " + str(keys[vals == leave][0]),
                        ),
                    )
                ),
                axis=1,
            ).T

        return pd.concat(list(map(synchro_milestone, leaves)), keys=milestones)

    if n_map > 1:
        permut = False
        stats = Parallel(n_jobs=n_jobs)(
            delayed(synchro_map)(i)
            for i in tqdm(range(n_map), desc="    multi mapping ")
        )
        allcor_r = pd.concat(stats)
        if perm:
            permut = True
            stats = Parallel(n_jobs=n_jobs)(
                delayed(synchro_map)(i)
                for i in tqdm(range(n_map), desc="    multi mapping permutations")
            )
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

    adata.uns[name]["synchro"] = allcor

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['syncho'], mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules.\n"
        "    .obs['inter_cor "
        + name
        + "'], loess fit of inter-module mean local gene-gene correlations prior to bifurcation."
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
    gamma = 1.5,
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
    def critical_map(m,gamma,loess_span):
        df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        def critical_milestone(leave):
            cells = getpath(img, root, graph["tips"], leave, graph, df).index

            if layer is None:
                if sparse.issparse(adata.X):
                    mat = pd.DataFrame(
                        adata[cells].X.A, index=cells, columns=adata.var_names
                    )
                else:
                    mat = pd.DataFrame(
                        adata[cells].X, index=cells, columns=adata.var_names
                    )
            else:
                if sparse.issparse(adata.layers[layer]):
                    mat = pd.DataFrame(
                        adata[cells].layers[layer].A,
                        index=cells,
                        columns=adata.var_names,
                    )
                else:
                    mat = pd.DataFrame(
                        adata[cells].layers[layer],
                        index=cells,
                        columns=adata.var_names,
                    )

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
                return [adata.obs.t[cls].max(), R_gene / R_cell,cls]

            stats = Parallel(n_jobs=n_jobs)(
                delayed(slide_path)(i)
                for i in tqdm(
                    np.arange(0, mat.shape[0] - w, step),
                    disable=n_map > 1,
                    file=sys.stdout,
                    desc="    leave " + str(keys[vals == leave][0]),
                )
            )

            cells_l = [s[2] for s in stats]
            stats = pd.DataFrame([[s[0],s[1]] for s in stats], columns=("t", "ci"))

            l = loess(stats.t, stats.ci, span=loess_span)
            l.fit()
            pred = l.predict(stats.t, stderror=True)
            conf = pred.confidence()

            stats["lowess"] = pred.values
            stats["ll"] = conf.lower
            stats["ul"] = conf.upper

            cell_stats=[pd.DataFrame(np.repeat(stats.ci[i].reshape(-1, 1),
                                               len(cells_l[i])),
                                     index=cells_l[i],
                                     columns=["ci"]) for i in range(stats.shape[0])]

            cell_stats = pd.concat(cell_stats, axis=1)
            cell_stats = cell_stats.T.groupby(level=0).mean().T
            cell_stats["t"]=adata.obs.loc[cell_stats.index,"t"]
            global rmgcv
            m = rmgcv.gam(
                Formula("ci~s(t,bs='ts')"),
                data=cell_stats,
                gamma=gamma,
            )

            cell_stats["fit"]=rmgcv.predict_gam(m)
            del cell_stats["t"]
            return stats,cell_stats

        res = list(map(critical_milestone, leaves))

        cell_stats = pd.concat([r[1] for r in res]).groupby(level=0).mean()

        res_slide = dict(zip(milestones,[r[0] for r in res]))

        return cell_stats, res_slide

    if n_map == 1:
        df,res_slide = critical_map(0,gamma,loess_span)
    else:
        #TODO: adapt multimapping
        stats = Parallel(n_jobs=n_jobs)(
            delayed(critical_map)(i)
            for i in tqdm(range(n_map), file=sys.stdout, desc="    multi mapping ")
        )
        res_slides = pd.concat(stats)


    if name in adata.uns:
        adata.uns[name]["critical transition"] = res_slide
    else:
        adata.uns[name] = {"critical transition": res_slide}

    adata.obs.loc[df.index, name + " CI"] = df.ci.values

    adata.obs.loc[df.index, name + " CI fitted"] = df.fit.values

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['critical transition'], df containing local critical transition index per window of cells.\n"
        "    .obs['"+ name+ " CI'], local critical transition index projected onto cells.\n"
        "    .obs['"+ name+ " CI fitted'], GAM fit of local critical transition index projected onto cells."
    )

    return adata if copy else None


def criticality_drivers(
    adata: AnnData,
    root_milestone,
    milestones,
    layer: Optional[str] = None,
    device="cpu",
    copy: bool = False,
):

    """\
    Calculates correlations between genes and local critical transition index along trajectory.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
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

    X = adata[~np.isnan(adata.obs[obs_name])].X
    CI = adata[~np.isnan(adata.obs[obs_name])].obs[obs_name].values

    if layer is None:
        X = adata[~np.isnan(adata.obs[obs_name])].X
    else:
        X = adata[~np.isnan(adata.obs[obs_name])].layers[layer]

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

    # fisher testing of correlations, code borrowed from CellRank :)
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

    res = pd.DataFrame(
        {"corr": corr, "pval": pvals, "ci_low": corr_ci_low, "ci_high": corr_ci_high},
        index=adata.var_names,
    )

    res["q_val"] = np.nan
    res.loc[~np.isnan(pvals), "q_val"] = multipletests(
        res[~np.isnan(pvals)].pval.values, alpha=0.05, method="fdr_bh"
    )[1]

    adata.uns[name]["criticality drivers"] = res.sort_values("corr", ascending=False).dropna()

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['criticality drivers'], df containing gene correlation with critical index transition."
    )

    return adata if copy else None
