from typing import Optional, List
import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
from functools import partial
from statsmodels.stats.weightstats import DescrStatsW

from .utils import get_X
from .. import logging as logg


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
    pp_probs = adata.obsm["X_R"].sum(axis=0)

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
        pps = (
            graph["pp_info"]
            .PP[graph["pp_info"].seg.isin(np.array(paths[0]).astype(str))]
            .index
        )
        seg_progenies = [str(seg) for seg in paths[0]]
        segs_cur = np.array(seg_progenies)

    def region_extract(pt_cur, segs_cur, nbranch):
        freq = list()

        pp_next = pps[
            (graph["pp_info"].loc[pps, "time"].values >= pt_cur)
            & graph["pp_info"].loc[pps, "seg"].isin(segs_cur).values
        ]
        if len(pp_next) == 0:
            raise Exception("win parameter too small, increase number of cells")
        cmsm = np.cumsum(
            pp_probs[pp_next][np.argsort(graph["pp_info"].loc[pp_next, "time"].values)]
        )
        inds = np.argwhere(cmsm > win).flatten()

        if len(inds) == 0:
            if cmsm.max() > win / 2:
                if mapping:
                    cell_probs = adata.obsm["X_R"][:, pp_next].sum(axis=1)
                else:
                    cell_probs = (
                        np.isin(
                            np.apply_along_axis(
                                lambda x: np.argmax(x), axis=1, arr=adata.obsm["X_R"]
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
                cell_probs = adata.obsm["X_R"][:, pps_region].sum(axis=1)
            else:
                cell_probs = (
                    np.isin(
                        np.apply_along_axis(
                            lambda x: np.argmax(x), axis=1, arr=adata.obsm["X_R"]
                        ),
                        pps_region,
                    )
                    * 1
                )

            freq = freq + [cell_probs]
            pt_cur = graph["pp_info"].loc[pps_region, "time"].max()

            # ignore the last node, to avoid infinite recursion
            if len(pps_region) == 1:
                return freq

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

    freqs = list(map(lambda f: pd.Series(f, index=adata.obs_names), freq))

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
    perm=False,
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

    genes = adata.var_names
    if perm:
        genesetA = np.random.choice(genes[~genes.isin(genesetA)], len(genesetA))
        genesetB = np.random.choice(genes[~genes.isin(genesetB)], len(genesetB))

    genesets = np.concatenate([genesetA, genesetB])

    X = get_X(adata, adata.obs_names, genesets, layer)

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
    corA.columns = [str(c) for c in corA.columns]
    corA = dict(
        zip(
            ["genesetA", "genesetB"],
            [corA.loc[geneset] for geneset in [genesetA, genesetB]],
        )
    )

    gather = partial(gather_cor, geneset=genesetB)
    corB = pd.concat(list(map(gather, range(nwin))), axis=1)
    corB.columns = [str(c) for c in corB.columns]
    corB = dict(
        zip(
            ["genesetA", "genesetB"],
            [corB.loc[geneset] for geneset in [genesetA, genesetB]],
        )
    )

    groups = ["A", "B"] if len(milestones) == 1 else milestones

    adata.uns = uns_temp
    adata.uns[name]["corAB"] = dict(zip(groups, [corA, corB]))

    logg.hint(
        "added \n" "    .uns['" + name + "']['corAB'], gene-gene correlation modules."
    )

    return adata if copy else None
