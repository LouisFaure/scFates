from typing import Optional, Union
from typing_extensions import Literal
from anndata import AnnData
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import shortest_path
import igraph
import warnings

from .. import logging as logg
from .. import settings
from .utils import process_R_cpu, norm_R_cpu, cor_mat_cpu
from .root import root
from .pseudotime import pseudotime, rename_milestones
from sklearn.metrics import pairwise_distances


def cleanup(
    adata: AnnData,
    minbranchlength: int = 3,
    leaves: Optional[int] = None,
    copy: bool = False,
):
    """\
    Remove spurious branches from the tree.

    Parameters
    ----------
    adata
        Annotated data matrix.
    minbranchlength
        Branches having less than the defined amount of nodes are discarded
    leaves
        Manually select branch tips to remove
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['B']`
            subsetted adjacency matrix of the principal points.
        `.uns['graph']['R']`
            subsetted updated soft assignment of cells to principal point in representation space.
        `.uns['graph']['F']`
            subsetted coordinates of principal points in representation space.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before cleaning it"
        )
    graph = adata.uns["graph"]

    B = graph["B"]
    R = graph["R"]
    F = graph["F"]
    init_num = B.shape[0]
    init_pp = np.arange(B.shape[0])
    if leaves is not None:
        g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
        tips = np.argwhere(np.array(g.degree()) == 1).flatten()
        branches = np.argwhere(np.array(g.degree()) > 2).flatten()
        idxmin = list(
            map(
                lambda l: np.argmin(
                    list(map(len, g.get_all_shortest_paths(l, branches)))
                ),
                leaves,
            )
        )
        torem_manual = np.concatenate(
            list(
                map(
                    lambda i: np.array(
                        g.get_shortest_paths(leaves[i], branches[idxmin[i]])[0][:-1]
                    ),
                    range(len(leaves)),
                )
            )
        )
        B = np.delete(B, torem_manual, axis=0)
        B = np.delete(B, torem_manual, axis=1)
        R = np.delete(R, torem_manual, axis=1)
        F = np.delete(F, torem_manual, axis=1)

    while True:
        torem = []
        g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
        tips = np.argwhere(np.array(g.degree()) == 1).flatten()
        branches = np.argwhere(np.array(g.degree()) > 2).flatten()

        if len(branches) == 0:
            break

        dist = np.array(
            list(
                map(
                    lambda t: np.min(
                        list(map(len, g.get_all_shortest_paths(t, branches)))
                    ),
                    tips,
                )
            )
        )

        if np.min(dist) > minbranchlength:
            break

        tip_torem = tips[np.argmin(dist)].T.flatten()
        B = np.delete(B, tip_torem, axis=0)
        B = np.delete(B, tip_torem, axis=1)
        R = np.delete(R, tip_torem, axis=1)
        F = np.delete(F, tip_torem, axis=1)
    R = (R.T / R.sum(axis=1)).T
    graph["R"] = R
    graph["B"] = B
    graph["F"] = F
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    graph["tips"] = np.argwhere(np.array(g.degree()) == 1).flatten()
    graph["forks"] = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.uns["graph"] = graph

    logg.info(
        "    graph cleaned", time=False, end=" " if settings.verbosity > 2 else "\n"
    )
    logg.hint("removed " + str(init_num - B.shape[0]) + " principal points")

    return adata if copy else None


def subset_tree(
    adata: AnnData,
    root_milestone,
    milestones,
    mode: Literal["extract", "substract"] = "extract",
    copy: bool = False,
):

    """\
    Subset the fitted tree.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    mode
        whether to substract or extract the mentionned path.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        subsetted dataset if `copy=True` it returns or else subsets these fields to `adata`:

        `.uns['graph']['B']`
            subsetted adjacency matrix of the principal points.
        `.uns['graph']['R']`
            subsetted updated soft assignment of cells to principal point in representation space.
        `.uns['graph']['F']`
            subsetted coordinates of principal points in representation space.
        `.obs['old_milestones']`
            previous milestones from initial tree.
    """

    logg.info("subsetting tree", reset=True)

    adata = adata.copy() if copy else adata

    adata.obs["old_milestones"] = adata.obs.milestones.copy()

    if "milestones_colors" in adata.uns:
        adata.uns["old_milestones_colors"] = adata.uns["milestones_colors"].copy()
        old_dct = dict(
            zip(
                adata.obs["old_milestones"].cat.categories,
                adata.uns["old_milestones_colors"],
            )
        )
    else:
        old_dct = None

    graph = adata.uns["graph"].copy()
    B = graph["B"].copy()
    R = graph["R"].copy()
    F = graph["F"].copy()

    dct = adata.uns["graph"]["milestones"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    dct = graph["milestones"]
    dct_rev = dict(zip(dct.values(), dct.keys()))
    oldmil = adata.obs.milestones.cat.categories.copy()

    if "milestones_colors" in adata.uns:
        oldmilcol = np.array(adata.uns["milestones_colors"])

    milpath = img.get_all_shortest_paths(
        str(dct[root_milestone]), [str(dct[m]) for m in milestones]
    )
    milpath = np.unique(np.array(milpath).ravel())
    milsel = np.array(img.vs["name"])[milpath].astype(int)
    milsel = np.array([dct_rev[m] for m in milsel])
    if mode == "substract":
        milsel = np.array(list(dct.keys()))[~np.isin(list(dct.keys()), milsel)]

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    leaves = list(map(lambda leave: dct[leave], milestones))
    r = dct[root_milestone]
    sub_nodes = np.unique(np.concatenate(g.get_all_shortest_paths(r, leaves)))

    cells = getpath(adata, root_milestone, milestones).index

    if mode == "substract":
        sub_nodes = sub_nodes[sub_nodes != r]
        sub_nodes = ~np.isin(range(B.shape[0]), sub_nodes)
        sub_cells = ~np.isin(graph["cells_fitted"], cells)

    if mode == "extract":
        sub_nodes = sub_nodes[sub_nodes != r] if r != graph["root"] else sub_nodes
        sub_nodes = np.isin(range(B.shape[0]), sub_nodes)
        sub_cells = np.isin(graph["cells_fitted"], cells)
        if r != graph["root"]:
            newroot = np.array(g.neighborhood(r)[1:])[sub_nodes[g.neighborhood(r)[1:]]][
                0
            ]
            dct[root_milestone] = newroot
            dct_rev = dict(zip(dct.values(), dct.keys()))

    R = R[sub_cells, :][:, sub_nodes]
    B = B[sub_nodes, :][:, sub_nodes]
    F = F[:, sub_nodes]

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.uns["graph"]["tips"] = tips
    adata.uns["graph"]["forks"] = forks
    adata.uns["graph"]["cells_fitted"] = np.array(graph["cells_fitted"])[sub_cells]
    adata.uns["graph"]["R"] = R
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F

    adata._inplace_subset_obs(adata.uns["graph"]["cells_fitted"])

    rmil = dct_rev[graph["root"]]
    rmil = root_milestone if ~np.isin(rmil, milsel) else rmil
    nodes = pd.Series(sub_nodes, index=np.arange(len(sub_nodes)))
    nodes.loc[nodes] = np.arange(nodes.sum())

    del adata.uns["graph"]["milestones"]
    del adata.obs["milestones"]
    root(adata, nodes[dct[rmil]])
    pseudotime(adata)

    nodes[[n is False for n in nodes]] = np.nan  # handle zero
    newmil = [
        dct_rev[nodes.index[nodes == int(m)][0]]
        for m in adata.obs.milestones.cat.categories
    ]
    rename_milestones(adata, newmil)
    milsel = newmil

    if "milestones_colors" in adata.uns:
        newcols = [oldmilcol[oldmil == m][0] for m in milsel]
    else:
        newcols = None

    adata.uns["milestones_colors"] = newcols
    if old_dct is not None:
        adata.uns["old_milestons_colors"] = [
            old_dct[m] for m in adata.obs.old_milestones.cat.categories
        ]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    if mode == "substract":
        logg.hint("tree subsetted")
    else:
        logg.hint("tree extracted")

    logg.hint(
        "added \n" "    .obs['old_milestones'], previous milestones from intial tree"
    )

    return adata if copy else None


def attach_tree(
    adata: AnnData,
    adata_branch: AnnData,
    linkage: Union[None, tuple] = None,
):
    """\
    Attach a tree to another!

    Given that the datasets were initially processed together.

    Parameters
    ----------
    adata
        Annotated data matrix.
    adata_branch
        Annotated data matrix containing cells and tree to attach to the adata.
    linkage
        Force the attachment of the two tree between two respective milestones (main tree, branch),
        the adjacency matrix will not be updated.

    Returns
    -------
    adata : anndata.AnnData
        combined dataset with the following merged tree fields:

        `.uns['graph']['B']`
            merged adjacency matrix of the principal points.
        `.uns['graph']['R']`
            merged updated soft assignment of cells to principal point in representation space.
        `.uns['graph']['F']`
            merged coordinates of principal points in representation space.
    """

    logg.info("attaching tree", reset=True)

    if "ppt" not in adata.uns:
        raise Exception("tree attachment can only run on graph computed with ppt!")

    graph = adata.uns["graph"]
    B = graph["B"].copy()
    R = graph["R"].copy()
    F = graph["F"].copy()

    R2 = adata_branch.uns["graph"]["R"].copy()

    logg.info("    merging")
    R, R2 = np.concatenate((R, np.zeros((R2.shape[0], R.shape[1])))), np.concatenate(
        (np.zeros((R.shape[0], R2.shape[1])), R2)
    )
    R = np.concatenate((R, R2), axis=1)

    if linkage is not None:
        n_init = B.shape[0]
        B2 = adata_branch.uns["graph"]["B"].copy()
        B, B2 = (
            np.concatenate((B, np.zeros((B2.shape[0], B.shape[1])))),
            np.concatenate((np.zeros((B.shape[0], B2.shape[1])), B2)),
        )
        B = np.concatenate((B, B2), axis=1)
        if "milestones" in adata.uns["graph"]:
            i = adata.uns["graph"]["milestones"][linkage[0]]
            ib = adata_branch.uns["graph"]["milestones"][linkage[1]]
        else:
            i, ib = int(linkage[0]), int(linkage[1])
        B[i, n_init + ib] = 1
        B[n_init + ib, i] = 1
    else:
        B = None

    newcells = np.concatenate(
        [np.array(graph["cells_fitted"]), adata_branch.uns["graph"]["cells_fitted"]]
    )

    adata = adata.concatenate(
        adata_branch, batch_key=None, index_unique=None, uns_merge="first", join="outer"
    )

    logg.info("    tree refitting")
    use_rep = graph["use_rep"]
    ndims_rep = graph["ndims_rep"] if "ndims_rep" in graph else None
    ndims_rep = (
        adata[newcells].obsm[use_rep].shape[1] if ndims_rep is None else ndims_rep
    )
    F = np.dot(adata[newcells].obsm[use_rep].T, R) / R.sum(axis=0)

    def run_ppt(F, adata, R, B):
        d = pairwise_distances(F.T, metric=adata.uns["ppt"]["metric"])

        W = np.empty_like(adata.obsm[use_rep].T)
        W.fill(1)

        csr = csr_matrix(np.triu(d, k=-1))
        Tcsr = minimum_spanning_tree(csr)
        mat = Tcsr.toarray()
        mat = mat + mat.T - np.diag(np.diag(mat))
        if linkage is None:
            B = (mat > 0).astype(int)

        D = (np.identity(B.shape[0])) * np.array(B.sum(axis=0))
        L = D - B
        M = L * adata.uns["ppt"]["lam"] + np.identity(R.shape[1]) * np.array(
            R.sum(axis=0)
        )
        old_F = F
        F = np.linalg.solve(M.T, (np.dot(adata[newcells].obsm[use_rep].T * W, R)).T).T
        return F, B, R

    F, B, R = run_ppt(F, adata, R, B)

    if linkage is None:
        R = pairwise_distances(
            adata[newcells].obsm[use_rep], F.T, metric=adata.uns["ppt"]["metric"]
        )
        process_R_cpu(R, adata.uns["ppt"]["sigma"])
        Rsum = R.sum(axis=1)
        norm_R_cpu(R, Rsum)

        F, B, R = run_ppt(F, adata, R, B)

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.uns["graph"]["tips"] = tips
    adata.uns["graph"]["forks"] = forks
    adata.uns["graph"]["cells_fitted"] = newcells
    adata.uns["graph"]["R"] = R
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F

    if "milestones" in adata.uns["graph"]:
        del adata.uns["graph"]["milestones"]
        del adata.obs["milestones"]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("datasets combined")

    return adata


def extend_tip(adata, tip, use_rep, restrict_seg):

    graph = adata.uns["graph"].copy()
    selecta = graph["R"].argmax(axis=1) == tip
    adata_sub = adata[selecta]

    early_cell = (
        adata_sub.obs.t.idxmin() if tip == graph["root"] else adata_sub.obs.t.idxmax()
    )

    cellsel = (
        adata.obs_names[adata.obs.seg == adata.obs.seg[early_cell]]
        if restrict_seg
        else adata.obs_names
    )

    ms_data = adata[cellsel].obsm[use_rep]
    from sklearn import preprocessing

    data = pd.DataFrame(
        preprocessing.minmax_scale(ms_data),
        index=adata[cellsel].obs_names,
    )

    #  Start cell as the nearest diffusion map boundary
    dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
    dists = pairwise_distances(
        data.loc[dm_boundaries, :], data.loc[early_cell, :].values.reshape(1, -1)
    )
    tip_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()

    adata_sub = adata[selecta]
    graph = adata.uns["graph"].copy()
    B = graph["B"]
    F = graph["F"]
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")

    add = np.zeros(B.shape[0]).reshape(1, -1)
    add[:, tip] = 1

    B = np.concatenate([B, add])

    add = np.zeros(B.shape[0]).reshape(-1, 1)
    add[tip] = 1
    B = np.concatenate([B, add], axis=1)

    F = np.concatenate([F, adata[tip_cell].obsm[use_rep].reshape(-1, 1)], axis=1)

    tip_nodes = np.array(g.neighborhood(tip, order=0, mindist=0) + [B.shape[0] - 1])
    R = graph["R"]
    cnei = np.isin(np.argmax(R, axis=1), tip_nodes)

    R[cnei, :] = 0
    R[:, tip_nodes[:-1]] = 0

    R_tip = pairwise_distances(
        adata[cnei].obsm[use_rep], F[:, tip_nodes].T, metric=adata.uns["ppt"]["metric"]
    )
    process_R_cpu(R_tip, adata.uns["ppt"]["sigma"] / 100)
    Rsum = R_tip.sum(axis=1)
    norm_R_cpu(R_tip, Rsum)

    R = np.concatenate([R, np.zeros(R.shape[0]).reshape(1, -1).T], axis=1)
    Rsum = R.sum(axis=1)
    norm_R_cpu(R, Rsum)

    for i, c in enumerate(
        np.argwhere(adata.obs_names.isin(adata_sub.obs_names)).ravel()
    ):
        for j, n in enumerate(tip_nodes):
            R[c, n] = R_tip[i, j]

    Rsum = R.sum(axis=1)
    norm_R_cpu(R, Rsum)

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.uns["graph"]["tips"] = tips
    adata.uns["graph"]["forks"] = forks
    adata.uns["graph"]["R"] = R
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F

    return adata


def extend_tips(adata: AnnData, restrict_seg: bool = False, copy: bool = False):

    """\
    Extend the tips of the tree to avoid compression of cells on pseudotime axis.

    Employ boundary detection as used by Palantir function `identify_terminal_states`.
    the boundary cells is then converted to a prinicpal point linked to the nearest tip.

    Parameters
    ----------
    adata
        Annotated data matrix.
    restrict_seg
        Whether to detect boundaries per segment or not.
    milestones
        Endpoint(s) of the path.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        Dataset with extended tree.

    """

    logg.info("subsetting tree", reset=True)

    if "t" not in adata.obs.columns:
        raise ValueError("You need to compute pseudotime before extending tips.")

    adata = adata.copy() if copy else adata

    for t in adata.uns["graph"]["tips"]:
        adata = extend_tip(adata, t, adata.uns["graph"]["use_rep"], restrict_seg)

    if any(adata.uns["graph"]["R"].sum(axis=0) == 0) & (restrict_seg == False):
        raise ValueError(
            "One tip has been assigned to no cell, rerun with restrict_seg=True."
        )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("tips extended")

    return adata if copy else None


def getpath(adata, root_milestone, milestones, include_root=False):
    """\
    Obtain dataframe of cell of a given path.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        Starting point of the path.
    milestones
        Endpoint(s) of the path.
    include_root
        When the root milestone is not the root, cells for the start tip are missing.

    Returns
    -------
    df : DataFrame
        pandas dataframe containing the cell selection from the path and relevant tree and pseudotime informations.

    """

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    g = igraph.Graph()
    g.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    g.add_edges(edges)

    uns_temp = adata.uns.copy()

    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = adata.uns["graph"]["milestones"]
    dct = dict(zip(dct.keys(), dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    df = adata.obs.copy()
    wf = warnings.filters.copy()
    warnings.filterwarnings("ignore")
    # for tip in leaves:
    def gatherpath(tip):
        try:
            path = np.array(g.vs[:]["name"])[
                np.array(g.get_shortest_paths(str(root), str(tip)))
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
            segs = graph["pp_seg"].index[segs]
            pth = df.loc[df.seg.astype(int).isin(segs), :].copy(deep=True)
            pth["branch"] = str(root) + "_" + str(tip)
            # warnings.filterwarnings("default")
            warnings.filters = wf
            return pth
        except IndexError:
            pass

    res = pd.concat(list(map(gatherpath, leaves)), axis=0)

    if include_root:
        sel = df.edge.str.startswith(str(root) + "|") | df.edge.str.contains(
            "\|" + str(root)
        )
        df = df.loc[sel]
        res = pd.concat([res, df], axis=0)

    return res
