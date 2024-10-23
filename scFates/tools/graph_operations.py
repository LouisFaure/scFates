from typing import Optional, Union, Iterable
from typing_extensions import Literal
from anndata import AnnData
import anndata as ad
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
    R = adata.obsm["X_R"]
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
    adata.obsm["X_R"] = R
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
    root_milestone: Optional[str] = None,
    milestones: Optional[Iterable] = None,
    mode: Literal["extract", "substract", "pseudotime"] = "extract",
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    copy: bool = False,
):

    """\
    Subset the fitted tree.

    if pseudotime parameter used, cutoff tree by removing cells/nodes after or before defined pseudotime.

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

    if ((t_min is not None) | (t_max is not None)) & (mode != "pseudotime"):
        logg.warn("setting mode to `pseudotime`")
        mode = "pseudotime"
    if mode == "pseudotime":
        _subset_t(adata, t_min, t_max)
        logg.info(
            "    finished", time=True, end=" " if settings.verbosity > 2 else "\n"
        )
        logg.hint("tree subsetted")

        return adata if copy else None

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

    adata.obs["old_seg"] = adata.obs.seg.copy()
    if "seg_colors" in adata.uns:
        adata.uns["old_seg_colors"] = adata.uns["seg_colors"].copy()
        oldseg = adata.obs.seg.cat.categories.copy()
        oldseg_col = np.array(adata.uns["seg_colors"].copy())
    else:
        oldseg = None

    graph = adata.uns["graph"].copy()
    B = graph["B"].copy()
    R = adata.obsm["X_R"].copy()
    F = graph["F"].copy()

    dct = graph["milestones"]
    dct_rev = dict(zip(dct.values(), dct.keys()))
    oldmil = adata.obs.milestones.cat.categories.copy()

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    if "milestones_colors" in adata.uns:
        oldmilcol = np.array(adata.uns["milestones_colors"])
    else:
        oldmilcol = None

    milpath = img.get_all_shortest_paths(
        str(dct[root_milestone]), [str(dct[m]) for m in milestones]
    )
    milpath = np.unique([item for sublist in milpath for item in sublist])
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
        sub_cells = ~np.isin(adata.obs_names, cells)

    if mode == "extract":
        sub_nodes = sub_nodes[sub_nodes != r] if r != graph["root"] else sub_nodes
        sub_nodes = np.isin(range(B.shape[0]), sub_nodes)
        sub_cells = np.isin(adata.obs_names, cells)
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

    Rsum = R.sum(axis=1)
    norm_R_cpu(R, Rsum)

    adata.uns["graph"]["tips"] = tips
    adata.uns["graph"]["forks"] = forks
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F

    adata._inplace_subset_obs(adata.obs_names[sub_cells])
    adata.obsm["X_R"] = R

    rmil = dct_rev[graph["root"]]
    rmil = root_milestone if ~np.isin(rmil, milsel) else rmil
    nodes = pd.Series(sub_nodes, index=np.arange(len(sub_nodes)))
    nodes.loc[nodes] = np.arange(nodes.sum())

    del adata.uns["graph"]["milestones"]
    del adata.obs["milestones"]
    del adata.uns["graph"]["root"]
    if "root2" in adata.uns["graph"]:
        del adata.uns["graph"]["root2"]
    root(adata, nodes[dct[rmil]])
    pseudotime(adata)

    nodes[[n is False for n in nodes]] = np.nan  # handle zero
    newmil = [
        dct_rev[nodes.index[nodes == int(m)][0]]
        for m in adata.obs.milestones.cat.categories
    ]
    rename_milestones(adata, newmil)
    milsel = newmil

    if ("milestones_colors" in adata.uns) & (oldmilcol is not None):
        newcols = [oldmilcol[oldmil == m][0] for m in milsel]
    else:
        newcols = None

    adata.uns["milestones_colors"] = newcols
    if old_dct is not None:
        adata.uns["old_milestons_colors"] = [
            old_dct[m] for m in adata.obs.old_milestones.cat.categories
        ]

    if oldseg is not None:
        adata.uns["seg_colors"] = [
            oldseg_col[oldseg == s][0] for s in adata.obs.seg.cat.categories
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


def _subset_t(adata: AnnData, t_min=None, t_max=None):
    t_min = adata.obs.t.min() - 0.1 if t_min is None else t_min
    t_max = adata.obs.t.max() + 0.1 if t_max is None else t_max
    pp_info = adata.uns["graph"]["pp_info"]
    keep = pp_info.index[(pp_info.time > t_min) & (pp_info.time < t_max)]
    pp_info = pp_info.loc[keep]
    adata.uns["graph"]["B"] = adata.uns["graph"]["B"][keep, :][:, keep]
    adata.uns["graph"]["F"] = adata.uns["graph"]["F"][:, keep]
    R = adata.obsm["X_R"][:, keep]
    Rsum = R.sum(axis=1)
    norm_R_cpu(R, Rsum)
    adata.obsm["X_R"] = R
    g = igraph.Graph.Adjacency(
        (adata.uns["graph"]["B"] > 0).tolist(), mode="undirected"
    )
    adata.uns["graph"]["tips"] = np.argwhere(np.array(g.degree()) == 1).ravel()
    adata.uns["graph"]["forks"] = np.argwhere(np.array(g.degree()) > 2).ravel()
    adata.uns["graph"]["pp_info"] = pp_info.reset_index(drop=True)
    root(adata, adata.uns["graph"]["pp_info"].time.idxmin())
    adata._inplace_subset_obs(
        adata.obs_names[(adata.obs.t > t_min) & (adata.obs.t < t_max)]
    )
    pseudotime(adata)


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
    R = adata.obsm["X_R"].copy()
    F = graph["F"].copy()

    R2 = adata_branch.obsm["X_R"].copy()

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

    newcells = np.concatenate([adata.obs_names, adata_branch.obs_names])

    adata = ad.concat([adata,adata_branch],uns_merge="first", join="outer")

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
    adata.obsm["X_R"] = R
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F

    if "milestones" in adata.uns["graph"]:
        del adata.uns["graph"]["milestones"]
        del adata.obs["milestones"]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("datasets combined")

    return adata


def simplify(adata: AnnData, n_nodes: int = 10, copy: bool = False):

    """\
    While keeping nodes defining forks and tips (milestones), reduce the number of nodes composing the segments.

    This can be helpful to simplify visualisations.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_nodes
        Number of nodes to keep per segments in between milestone nodes.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        Dataset with simplified graph.

    """

    logg.info("simplifying graph", reset=True)

    adata = adata.copy() if copy else adata

    use_rep = adata.uns["graph"]["use_rep"]
    metric = adata.uns["graph"]["metrics"]
    res = []
    miles = {}
    mils = np.array(list(adata.uns["graph"]["milestones"].values()))
    R = adata.obsm["X_R"]
    pp_info = adata.uns["graph"]["pp_info"]
    for s in pp_info.seg.unique():
        pp_info = adata.uns["graph"]["pp_info"]
        pp_info = pp_info.loc[pp_info.seg == s]
        pp_info.sort_values("time", inplace=True)

        tokeep = pp_info.index[np.isin(pp_info.index, mils)]
        tosimplify = pp_info.index[~np.isin(pp_info.index, mils)]
        for tk in tokeep:
            res.append(R[:, tk])
            miles[tk] = len(res) - 1
        idxs = np.array_split(tosimplify, n_nodes)
        for idx in idxs:
            res.append(R[:, idx].mean(axis=1))

    newR = np.vstack(res).T
    newR = newR / newR.sum(axis=1).reshape(-1, 1)

    rep = adata.obsm[use_rep]
    F_mat = np.dot(rep.T, newR) / newR.sum(axis=0)

    d = pairwise_distances(F_mat.T, metric=metric)

    csr = csr_matrix(np.triu(d, k=-1))
    Tcsr = minimum_spanning_tree(csr)
    mat = Tcsr.toarray()
    mat = mat + mat.T - np.diag(np.diag(mat))
    B = (mat > 0).astype(int)

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.obsm["X_R"] = newR
    adata.uns["graph"]["B"] = B
    adata.uns["graph"]["F"] = F_mat
    adata.uns["graph"]["tips"] = tips
    adata.uns["graph"]["forks"] = forks

    newmil = {}
    for m in adata.uns["graph"]["milestones"].keys():
        newmil[m] = miles[adata.uns["graph"]["milestones"][m]]

    milroot = (
        pd.Series(adata.uns["graph"]["milestones"]) == adata.uns["graph"]["root"]
    ).idxmax()
    newroot = newmil[milroot]

    root(adata, newroot)
    pseudotime(adata)

    adata.uns["graph"]["milestones"] = newmil

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("graph simplified")

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


def convert_to_soft(
    adata, sigma: float, lam: int, n_steps: int = 1, copy: bool = False
):
    """\
    Convert an hard assignment matrix to a soft one, allowing for probabilistic mapping.

    Parameters
    ----------
    adata
        Annotated data matrix.
    sigma
        Sigma parameter from SimplePPT.
    lam
        Lambda parameter from SimplePPT.
    n_steps
        Number of steps to run the solving of R and F
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        dataset if `copy=True` it returns or else updated these fields to `adata`:

        `.uns['graph']['R']`
            converted from hard to soft assignment matrix.
        `.uns['graph']['F']`
            solved from newly converted soft assignment matrix.

    """

    logg.info("Converting R into soft assignment matrix", reset=True)

    adata = adata.copy() if copy else adata

    graph = adata.uns["graph"]
    F = graph["F"]
    B = graph["B"]
    X = adata.obsm[graph["use_rep"]][:, : graph["ndims_rep"]]

    for i in range(n_steps):
        R = pairwise_distances(X, F.T, metric="euclidean")
        process_R_cpu(R, sigma)
        Rsum = R.sum(axis=1)
        norm_R_cpu(R, Rsum)
        D = (np.identity(B.shape[0])) * np.array(B.sum(axis=0))
        L = D - B
        M = L * lam + np.identity(R.shape[1]) * np.array(R.sum(axis=0))
        F = np.linalg.solve(M.T, (np.dot(X.T, R)).T).T

    adata.obsm["X_R"] = R
    adata.uns["graph"]["F"] = F
    adata.uns["graph"]["method"] = "ppt"

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    .obsm['X_R'] converted soft assignment of cells to principal points.\n"
        "    .uns['graph']['F'] coordinates of principal points in representation space."
    )

    return adata if copy else None
