from typing import Union, Iterable
import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
from scFates.tools.utils import get_X
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def matrix(
    adata: sc.AnnData,
    features: Iterable,
    nbins: int = 5,
    layer: Union[None, str] = None,
    do_annot: bool = False,
    annot_top: bool = True,
    **kwargs
):

    adata = adata[:, features].copy()

    X = get_X(adata, adata.obs_names, adata.var_names, layer=layer)

    adata.X = X / X.max(axis=0).ravel()

    graph = adata.uns["graph"]

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph(directed=True)
    img.add_vertices(vals.astype(str))
    img.add_edges(edges)

    allpaths = img.get_all_shortest_paths(
        str(graph["root"]), to=graph["tips"].astype(str)
    )

    allpaths = np.array(allpaths, dtype=object)[
        np.argsort(np.array([len(p) for p in allpaths]))
    ]

    order = allpaths[0]
    for i in range(1, len(allpaths)):
        order = order + np.array(allpaths[i])[~np.isin(allpaths[i], order)].tolist()

    order = np.array(order)[1:]

    order = pd.Series(graph["milestones"].keys(), index=graph["milestones"].values())[
        np.array(img.vs["name"])[order].astype(int)
    ]
    order = pd.Series(
        range(len(adata.obs.seg.cat.categories)), index=graph["pp_seg"]["to"]
    )[order.index].values

    vs2mils = pd.Series(dct.keys(), index=dct.values())

    cellsel = [
        adata.obs.milestones[adata.obs.seg == s]
        for s in adata.obs.seg.cat.categories[order]
    ]

    fig, axs = plt.subplots(
        1,
        len(order) + 1 * do_annot,
        constrained_layout=True,
        sharey=True,
        figsize=(
            2 * len(order) / 4 + 2 + 2 * do_annot,
            (len(features) + 1 * annot_top) / 5 + 1 / 3,
        ),
    )

    pos = np.arange(len(order), 0, -1)
    for i, s in enumerate(order):
        adata_sub = adata[adata.obs.seg == adata.obs.seg.cat.categories[s]].copy()
        adata_sub.obs["split"] = pd.cut(adata_sub.obs.t, bins=nbins)

        ss = int(adata.obs.seg.cat.categories[s])
        sel = (
            adata.obs.milestones.cat.categories
            == vs2mils[graph["pp_seg"].loc[ss]["from"]]
        )
        start = np.array(adata.uns["milestones_colors"])[sel][0]
        sel = (
            adata.obs.milestones.cat.categories
            == vs2mils[graph["pp_seg"].loc[ss]["to"]]
        )
        end = np.array(adata.uns["milestones_colors"])[sel][0]

        from matplotlib.colors import LinearSegmentedColormap

        my_cm = LinearSegmentedColormap.from_list("aspect", [start, end])

        if "use_raw" not in kwargs:
            kwargs["use_raw"] = False

        M = sc.pl.MatrixPlot(adata_sub, features, "split", vmin=0, vmax=1, **kwargs)
        M.swap_axes()
        M._mainplot(axs[i])
        axs[i].set_xticklabels("")
        plt.margins(y=10)
        plt.setp(axs[i].get_yticklabels(), style="italic")
        if annot_top:
            divider = make_axes_locatable(axs[i])
            cax = divider.new_vertical(size=0.2, pad=0.05, pack_start=False)
            mappable = cm.ScalarMappable(cmap=my_cm)

            fig.add_axes(cax)
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cbar.set_ticks([])
            cbar.outline.set_linewidth(1.5)

    if do_annot:
        Amps = adata.var.loc[features, "A"]
        axs[i + 1].barh(
            np.arange(len(features)) + 0.5,
            Amps,
            color="salmon",
            height=0.65,
            left=0,
            edgecolor="black",
            label="A",
        )
        axs[i + 1].invert_yaxis()
        # axs[i+1].axis("off")
        axs[i + 1].spines["top"].set_visible(False)
        axs[i + 1].spines["left"].set_visible(False)
        axs[i + 1].spines["right"].set_visible(False)

        axs[i + 1].get_xaxis().tick_bottom()
        axs[i + 1].tick_params(left=False)
        axs[i + 1].set_xlim([0, 4])
        axs[i + 1].set_xticks([0, 2])
        axs[i + 1].set_xticklabels([0, 2])

        axs[i + 1].grid(False)
        if annot_top:
            divider = make_axes_locatable(axs[i + 1])
            cax = divider.new_vertical(size=0.2, pad=0.05, pack_start=False)
            fig.add_axes(cax)
            cax.annotate("Amplitude", (0, 0), va="bottom", ha="left", size=12)
            cax.axis("off")
