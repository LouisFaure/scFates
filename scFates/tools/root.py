from typing import Optional, Union
from anndata import AnnData
import numpy as np
import pandas as pd
import igraph

import itertools
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .utils import get_X

from .. import logging as logg
from .. import settings


def root(
    adata: AnnData,
    root: Union[int, str],
    tips_only: bool = False,
    min_val: bool = False,
    layer: Optional = None,
    copy: bool = False,
):
    """\
    Define the root of the trajectory.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root
        Either an Id (int) of the tip of the fork to be considered as a root. Or a key (str) from obs/X (such as CytoTRACE) for automatic selection.
    tips_only
        Perform automatic assignment on tips only.
    min_val
        Perform automatic assignment using minimum value instead.
    layer
        If key is in X, choose which layer to use for the averaging.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['root']`
            selected root.
        `.uns['graph']['pp_info']`
            for each PP, its distance vs root and segment assignment.
        `.uns['graph']['pp_seg']`
            segments network information.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` or `tl.curve` first to compute a princal graph before choosing a root."
        )

    graph = adata.uns["graph"]
    circle = len(graph["tips"]) == 0
    if type(root) == str:
        if root in adata.obs:
            root_val = adata.obs[root]
        elif root in adata.var_names:
            root_val = get_X(adata, adata.obs_names, root, layer).ravel()
        else:
            raise ValueError(f"{root} not present in adata.var_names or adata.obs")
        logg.info("automatic root selection using " + root + " values", time=False)
        nodes = np.arange(adata.obsm["X_R"].shape[1])
        avgs = pd.Series(np.nan, index=nodes)
        # handle empty nodes
        unassigned = np.array([adata.obsm["X_R"][:, n].sum() for n in nodes]) > 0
        nodes = nodes[unassigned]
        avgs_temp = list(
            map(
                lambda n: np.average(root_val, weights=adata.obsm["X_R"][:, n]),
                nodes,
            )
        )
        avgs.loc[nodes] = avgs_temp
        if tips_only:
            mask = np.ones(avgs.shape, bool)
            mask[adata.uns["graph"]["tips"]] = False
            avgs[mask] = 0
        if min_val:
            if tips_only:
                avgs[mask] = avgs.max()
            root = np.argmin(avgs)
        else:
            root = np.argmax(avgs)

    if circle:
        d = 1e-6 + pairwise_distances(
            graph["F"].T, graph["F"].T, metric=graph["metrics"]
        )
        to_g = graph["B"] * d
        csr = csr_matrix(to_g)

        g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
        g.es["weight"] = to_g[to_g.nonzero()]
        root_dist_matrix = shortest_path(csr, directed=False, indices=root)
        pp_info = pd.DataFrame(
            {
                "PP": g.vs.indices,
                "time": root_dist_matrix,
                "seg": np.zeros(csr.shape[0]),
            }
        )

        furthest = pp_info.time.idxmax()
        pp_info.loc[
            g.get_shortest_paths(v=root, to=furthest, weights="weight")[0][:-1], "seg"
        ] = 1

        s = pp_info.seg.copy()
        s[furthest] = 0
        g.vs["group"] = s
        g.vs["label"] = s.index

        sub_g = g.vs.select(group=0).subgraph()
        a, b = np.argwhere(np.array(sub_g.degree()) == 1).ravel()
        dst_0 = sub_g.distances(a, b, weights="weight")[0][0]
        a, b = sub_g.vs["label"][a], sub_g.vs["label"][b]
        dst_1 = g.distances(root, furthest, weights="weight")[0][0]
        pp_seg = pd.concat(
            [
                pd.Series([0, a, b, dst_0], index=["n", "from", "to", "d"]),
                pd.Series([1, root, furthest, dst_1], index=["n", "from", "to", "d"]),
            ],
            axis=1,
        ).T

        pp_seg.n = pp_seg.n.astype(int).astype(str)
        pp_seg["from"] = pp_seg["from"].astype(int)
        pp_seg.to = pp_seg.to.astype(int)
        pp_info.seg = pp_info.seg.astype(int).astype(str)

        adata.uns["graph"]["root"] = root
        adata.uns["graph"]["root2"] = pp_seg["from"][0]
        adata.uns["graph"]["pp_info"] = pp_info
        adata.uns["graph"]["pp_seg"] = pp_seg

        adata.uns["graph"]["forks"] = np.array([furthest])
        adata.uns["graph"]["tips"] = np.array([furthest, root, pp_seg["from"][0]])

    else:
        d = 1e-6 + pairwise_distances(
            graph["F"].T, graph["F"].T, metric=graph["metrics"]
        )

        to_g = graph["B"] * d

        csr = csr_matrix(to_g)

        g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
        g.es["weight"] = to_g[to_g.nonzero()]

        root_dist_matrix = shortest_path(csr, directed=False, indices=root)
        pp_info = pd.DataFrame(
            {
                "PP": g.vs.indices,
                "time": root_dist_matrix,
                "seg": np.zeros(csr.shape[0]),
            }
        )

        nodes = np.argwhere(
            np.apply_along_axis(arr=(csr > 0).todense(), axis=0, func1d=np.sum) != 2
        ).flatten()
        nodes = np.unique(np.append(nodes, root))

        pp_seg = list()
        for node1, node2 in itertools.combinations(nodes, 2):
            paths12 = g.get_shortest_paths(node1, node2)
            paths12 = np.array([val for sublist in paths12 for val in sublist])

            if np.sum(np.isin(nodes, paths12)) == 2:
                fromto = np.array([node1, node2])
                path_root = root_dist_matrix[[node1, node2]]
                fro = fromto[np.argmin(path_root)]
                to = fromto[np.argmax(path_root)]
                pp_info.loc[paths12, "seg"] = len(pp_seg) + 1
                pp_seg.append(
                    pd.DataFrame(
                        {
                            "n": len(pp_seg) + 1,
                            "from": fro,
                            "to": to,
                            "d": shortest_path(csr, directed=False, indices=fro)[to],
                        },
                        index=[len(pp_seg) + 1],
                    )
                )

        pp_seg = pd.concat(pp_seg, axis=0)
        pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

        pp_seg["from"] = pp_seg["from"].astype(int)
        pp_seg["to"] = pp_seg["to"].astype(int)

        pp_info["seg"] = pp_info["seg"].astype(int).astype(str)

        graph["pp_info"] = pp_info
        graph["pp_seg"] = pp_seg
        graph["root"] = root
        adata.uns["graph"] = graph

    logg.info(
        "node " + str(root) + " selected as a root",
        time=False,
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "added\n"
        "    .uns['graph']['root'] selected root.\n"
        "    .uns['graph']['pp_info'] for each PP, its distance vs root and segment assignment.\n"
        "    .uns['graph']['pp_seg'] segments network information."
    )

    return adata if copy else None


def roots(adata: AnnData, roots, meeting, copy: bool = False):

    """\
    Define 2 roots of the tree.

    Parameters
    ----------
    adata
        Annotated data matrix.
    roots
        list of tips or forks to be considered a roots.
    meeting
         node ID of the meeting point fo the two converging paths.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['root']`
            farthest root selected.
        `.uns['graph']['root2']`
            2nd root selected.
        `.uns['graph']['meeting']`
            meeting point on the tree.
        `.uns['graph']['pp_info']`
            for each PP, its distance vs root and segment assignment).
        `.uns['graph']['pp_seg']`
            segments network information.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before choosing two roots."
        )

    graph = adata.uns["graph"]

    d = 1e-6 + pairwise_distances(graph["F"].T, graph["F"].T, metric=graph["metrics"])

    to_g = graph["B"] * d

    csr = csr_matrix(to_g)

    g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
    g.es["weight"] = to_g[to_g.nonzero()]

    root = roots[
        np.argmax(shortest_path(csr, directed=False, indices=roots)[:, meeting])
    ]
    root2 = roots[
        np.argmin(shortest_path(csr, directed=False, indices=roots)[:, meeting])
    ]

    root_dist_matrix = shortest_path(csr, directed=False, indices=root)
    pp_info = pd.DataFrame(
        {"PP": g.vs.indices, "time": root_dist_matrix, "seg": np.zeros(csr.shape[0])}
    )

    nodes = np.argwhere(
        np.apply_along_axis(arr=(csr > 0).todense(), axis=0, func1d=np.sum) != 2
    ).flatten()
    pp_seg = []
    for node1, node2 in itertools.combinations(nodes, 2):
        paths12 = g.get_shortest_paths(node1, node2)
        paths12 = np.array([val for sublist in paths12 for val in sublist])

        if np.sum(np.isin(nodes, paths12)) == 2:
            fromto = np.array([node1, node2])
            path_root = root_dist_matrix[[node1, node2]]
            fro = fromto[np.argmin(path_root)]
            to = fromto[np.argmax(path_root)]
            pp_info.loc[paths12, "seg"] = len(pp_seg) + 1
            pp_seg.append(
                pd.DataFrame(
                    {
                        "n": len(pp_seg) + 1,
                        "from": fro,
                        "to": to,
                        "d": shortest_path(csr, directed=False, indices=fro)[to],
                    },
                    index=[len(pp_seg) + 1],
                )
            )
    pp_seg = pd.concat(pp_seg, axis=0)
    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)
    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)
    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)

    tips = graph["tips"]
    tips = tips[~np.isin(tips, roots)]

    edges = pp_seg[["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(pp_seg[["from", "to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    root2paths = pd.Series(
        shortest_path(csr, directed=False, indices=root2)[tips.tolist() + [meeting]],
        index=tips.tolist() + [meeting],
    )

    toinvert = root2paths.index[(root2paths <= root2paths[meeting])]

    for toinv in toinvert:
        pathtorev = np.array(img.vs[:]["name"])[
            np.array(img.get_shortest_paths(str(root2), str(toinv)))
        ][0]
        for i in range((len(pathtorev) - 1)):
            segtorev = pp_seg.index[
                pp_seg[["from", "to"]]
                .astype(str)
                .apply(lambda x: all(x.values == pathtorev[[i + 1, i]]), axis=1)
            ]

            pp_seg.loc[segtorev, ["from", "to"]] = pp_seg.loc[segtorev][
                ["to", "from"]
            ].values
            pp_seg["from"] = pp_seg["from"].astype(int)
            pp_seg["to"] = pp_seg["to"].astype(int)

    pptoinvert = np.unique(np.concatenate(g.get_shortest_paths(root2, toinvert)))
    reverted_dist = (
        shortest_path(csr, directed=False, indices=root2)
        + np.abs(
            np.diff(shortest_path(csr, directed=False, indices=roots)[:, meeting])
        )[0]
    )
    pp_info.loc[pptoinvert, "time"] = reverted_dist[pptoinvert]

    graph["pp_info"] = pp_info
    graph["pp_seg"] = pp_seg
    graph["root"] = root
    graph["root2"] = root2
    graph["meeting"] = meeting

    adata.uns["graph"] = graph

    logg.info("root selected", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    " + str(root) + " is the farthest root.\n"
        "    .uns['graph']['root'] farthest root selected.\n"
        "    .uns['graph']['root2'] 2nd root selected.\n"
        "    .uns['graph']['meeting'] meeting point on the tree.\n"
        "    .uns['graph']['pp_info'] for each PP, its distance vs root and segment assignment.\n"
        "    .uns['graph']['pp_seg'] segments network information."
    )

    return adata if copy else None
