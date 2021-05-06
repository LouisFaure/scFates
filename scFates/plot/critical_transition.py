import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Union, Optional
import igraph
import numpy as np
from scanpy.plotting._utils import savefig_or_show


def critical_transition(
    adata: AnnData,
    root_milestone,
    milestones,
    col: Union[str, None] = None,
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

    fig, ax = plt.subplots()
    for p, df in adata.uns[name]["critical transition"]["LOESS"].items():
        col = mlsc[adata.obs.milestones.cat.categories == p][0] if col is None else col
        ax.plot(df.t, df.ci, "+", c=col, zorder=10, alpha=0.3)
        ax.plot(df.t, df.lowess, c=col)
        ax.fill_between(
            df.t.values.tolist(),
            df.ll.tolist(),
            df.ul.tolist(),
            alpha=0.33,
            edgecolor=col,
            facecolor=col,
        )
        ax.set_xlabel("pseudotime")
        ax.set_ylabel("critical index")
        if len(milestones) > 1:
            ax.axvline(fork_t, color="black")

    savefig_or_show("critical_transition", show=show, save=save)

    if show == False:
        return ax
