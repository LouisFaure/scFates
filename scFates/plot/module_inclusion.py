import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hex2color
import seaborn as sns
import igraph
import numpy as np


def module_inclusion(
    adata, root_milestone, milestones, bins, branch, figsize=(6, 5), perm=False
):

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

    sg = np.linspace(0, matSwitch[branch].max().max(), bins)
    sort = matSwitch[branch].mean(axis=1).sort_values()
    sort = sort[~np.isnan(sort)].index
    matSwitch[branch] = matSwitch[branch].loc[sort]
    hm = np.vstack(
        matSwitch[branch]
        .apply(lambda x: np.array([sum(s > x) / len(x) for s in sg]), axis=1)
        .values
    )

    gg = LinearSegmentedColormap.from_list("", ["lightgrey", "#787F3A"])

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        hm,
        yticklabels=sort,
        xticklabels=np.round(sg, 2),
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

    ax.axvline(
        fork_t / (matSwitch[branch].max().max() / bins),
        color="k",
        linestyle="dashed",
    )
