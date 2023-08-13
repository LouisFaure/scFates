from typing import Union, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hex2color
import seaborn as sns
import igraph
import numpy as np

from scanpy.plotting._utils import savefig_or_show
from typing_extensions import Literal


def module_inclusion(
    adata,
    root_milestone,
    milestones,
    bins: int,
    branch: str,
    feature: Optional[str] = None,
    figsize: tuple = (6, 5),
    max_t: Literal["fork", "max"] = "max",
    perm: bool = False,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot results generated from tl.module_inclusion.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    bins
        Number of bins to separate the pseudotime path.
    branch
        Which endpoint to focus on.
    feature
        Plot a specific feature.
    figsize
        Size of the figure.
    max_t
        Until which pseudotime to limit the plot and binning.
    perm
        Show permutation results.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

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

    perm_str = "_perm" if perm else ""
    matSwitch = adata.uns[name]["module_inclusion" + perm_str]

    fork = list(
        set(img.get_shortest_paths(str(root), str(leaves[0]))[0]).intersection(
            img.get_shortest_paths(str(root), str(leaves[1]))[0]
        )
    )
    fork = np.array(img.vs["name"], dtype=int)[fork]
    fork_t = adata.uns["graph"]["pp_info"].loc[fork, "time"].max()

    maxt = fork_t if max_t == "fork" else matSwitch[branch].max().max()
    sg = np.linspace(0, maxt, bins)
    sort = matSwitch[branch].mean(axis=1).sort_values()
    sort = sort[sort < maxt] if max_t == "fork" else sort
    sort = sort[~np.isnan(sort)].index
    matSwitch[branch] = matSwitch[branch].loc[sort]
    hm = np.vstack(
        matSwitch[branch]
        .apply(lambda x: np.array([sum(s > x) / len(x) for s in sg]), axis=1)
        .values
    )

    c_mil = np.array(mlsc)[np.argwhere(adata.obs.milestones.cat.categories == branch)][
        0
    ][0]
    gg = LinearSegmentedColormap.from_list("", ["lightgrey", c_mil])

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    if feature is not None:
        dat = matSwitch[branch].loc[feature]

        ax.plot(range(bins), [(s > dat).sum() / 100 for s in sg], color="k")
        plt.grid(False)
        ax.axhline(0.5, linestyle="--", color="grey")
        ax.axhline(0, linestyle="--", color="grey")
        ax.axhline(1, linestyle="--", color="grey")
        ax.set_title(feature)
        ax.set_ylabel("inclusion probability")
        ax.set_xlabel("pseudotime")
    else:
        sns.heatmap(
            hm,
            yticklabels=sort,
            xticklabels=np.round(sg, 3),
            cmap=gg,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.3, "anchor": (0, 0)},
        )
        ax.set_xlabel("pseudotime")

        ax.axhline(y=0, color="k", linewidth=2)
        ax.axhline(y=hm.shape[0], color="k", linewidth=2)
        ax.axvline(x=0, color="k", linewidth=2)
        ax.axvline(x=hm.shape[1], color="k", linewidth=2)

        if max_t == "max":
            ax.axvline(
                fork_t / (matSwitch[branch].max().max() / bins),
                color="k",
                linestyle="dashed",
            )
        ax.grid(False)
    if show == False:
        return ax

    savefig_or_show("module_inclusion", show=show, save=save)
