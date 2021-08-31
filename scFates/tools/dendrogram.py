import networkx as nx
import random
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt

from .. import logging as logg
from .. import settings


def dendrogram(adata, crowdedness=1):

    logg.info("Generating dendrogram of tree", reset=True)

    roots = None

    graph = adata.uns["graph"]

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph(directed=True)
    img.add_vertices(vals.astype(str))
    img.add_edges(edges)

    img.vs["label"] = keys

    pos = hierarchy_pos(img.to_networkx(), vert_gap=0.1)

    layout = pd.DataFrame(pos).T
    layout.index = np.array(list(graph["milestones"].values()), int)[
        np.array(list(pos.keys()))
    ]
    layout = layout[0]
    newseg = layout.loc[graph["pp_seg"].to]
    newseg.index = graph["pp_seg"].n

    df = adata.obs.copy()

    df["seg_pos"] = df.seg.astype(str)
    df["t"] = -df.t

    for i, x in enumerate(newseg):
        df.loc[df.seg == newseg.index[i], "seg_pos"] = x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax = sns.swarmplot(x="seg_pos", y="t", data=df, size=crowdedness)

    plt.close()
    segments = list()
    for i in range(len(ax.collections)):
        newseg = newseg.sort_values()
        t_min, t_max = (
            ax.collections[i].get_offsets()[:, 1].max(),
            ax.collections[i].get_offsets()[:, 1].min(),
        )
        tip = graph["pp_seg"].loc[int(newseg.index[i])]["to"]

        segments = segments + [[newseg.index[i], [(i, i), (t_min, t_max)]]]

        for l in graph["pp_seg"].n[graph["pp_seg"]["from"] == tip]:
            link = np.argwhere(newseg.index == str(l))[0][0]
            segments = segments + [[l, [(i, link), (t_max, t_max)]]]

    axc_dct = zip(np.sort(df.seg_pos.unique()), ax.collections)
    df = df.loc[df.t.sort_values().index]
    new_emb = [
        pd.DataFrame(axc.get_offsets(), index=df.loc[df.seg_pos == s].index)
        for s, axc in axc_dct
    ]
    new_emb = pd.concat(new_emb).loc[adata.obs_names].values

    adata.obsm["X_dendro"] = new_emb

    df = pd.DataFrame(segments)
    segments = df[0].unique()

    adata.uns["dendro_segments"] = dict(
        zip(segments, [df.loc[df[0] == s, 1].tolist() for s in segments])
    )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .obs['X_dendro'], new embedding generated.\n"
        "    .uns['dendro_segments'] tree segments used for plotting."
    )


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
