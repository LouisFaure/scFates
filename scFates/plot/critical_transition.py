import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Union, Optional
import igraph
import numpy as np


def critical_transition(
    adata: AnnData,
    root_milestone,
    milestones,
    path: Union["all", "pre-fork", str] = "all",
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot results generated from tl.critical_transition.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    name = root_milestone + "->" + "<>".join(milestones)
    mlsc = np.array(adata.uns["milestones_colors"].copy())

    if len(milestones) > 1:
        graph = adata.uns["graph"]

        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        dct = graph["milestones"]
        leaves = list(map(lambda leave: dct[leave], milestones))
        root = dct[root_milestone]

        fork = list(
            set(img.get_shortest_paths(str(root), str(leaves[0]))[0]).intersection(
                img.get_shortest_paths(str(root), str(leaves[1]))[0]
            )
        )
        fork = np.array(img.vs["name"], dtype=int)[fork]
        fork_t = adata.uns["graph"]["pp_info"].loc[fork, "time"].max()

    if len(milestones) == 1:
        dfs = [adata.uns[name]["critical transition"]]
        dfs = dict(zip(milestones, dfs))
    elif (path == "all") & (len(milestones) > 1):
        dfs = [adata.uns[name]["critical transition"][m] for m in milestones]
        dfs = dict(zip(milestones, dfs))
    elif path != "all":
        dfs = [adata.uns[name]["critical transition"][path]]
        dfs = dict(zip([path], dfs))

    for p, df in dfs.items():
        col = (
            "grey"
            if p == "pre-fork"
            else mlsc[adata.obs.milestones.cat.categories == p][0]
        )
        plt.scatter(df.t, df.ic, c=col)
        plt.plot(df.t, df.lowess, c=col)
        plt.fill_between(
            df.t.values.tolist(),
            df.ll.tolist(),
            df.ul.tolist(),
            alpha=0.33,
            edgecolor=col,
            facecolor=col,
        )
        plt.xlabel("pseudotime")
        plt.ylabel("critical index")
        if (len(milestones) > 1) & (path != "pre-fork"):
            plt.axvline(fork_t, color="black")
