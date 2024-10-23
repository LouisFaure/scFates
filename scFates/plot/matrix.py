from typing import Union, Sequence, Optional, Iterable
from typing_extensions import Literal
import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
from scFates.tools.utils import get_X
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scanpy.plotting._utils import savefig_or_show
from anndata import AnnData
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import logging as logg
from ..tools.graph_operations import subset_tree
from .utils import subset_cells


def matrix(
    adata: AnnData,
    features: Sequence,
    nbins: int = 5,
    layer: str = "fitted",
    norm: Literal["max", "minmax", "none"] = "max",
    annot_var: bool = False,
    annot_top: bool = True,
    link_seg: bool = True,
    root_milestone: Optional[str] = None,
    milestones: Optional[Iterable] = None,
    feature_style: str = "normal",
    feature_spacing: float = 1,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    colorbar_title: Optional[str] = None,
    figsize: Union[None, tuple] = None,
    return_data: bool = False,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):
    """\
    Plot a set of features as per-segment matrix plots of binned pseudotimes.

    Parameters
    ----------
    adata
        Annotated data matrix.
    features
        Name of the features.
    nbins
        Number of pseudotime bins per segment.
    layer
        Layer to use for the expression to display.
    norm
        How to normalize the expression.
    annot_var
        Annotate overall tree amplitude of expression of the marker (from .var['A']).
    annot_top
        Display milestones gradient for each segment on top of plots.
    link_seg
        Link the segment together to keep track of the the tree progression.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    feature_style
        Font style of the feature labels.
    feature_spacing
        When figsize is None, controls the the height of each rows.
    cmap
        colormap to use, by default is plt.rcParams["image.cmap"].
    colorbar
        Show the colorbar.
    colorbar_title
        Set a custom colorbar title.
    figsize
        Custom figure size.
    show
        show the plot.
    save
        save the plot.
    save_genes
        save list of genes following the order displayed on the heatmap.
    **kwargs
        arguments passed to :class:`scanpy.pl.MatrixPlot`

    Returns
    -------
    If `show==False` an array of :class:`~matplotlib.axes.Axes`

    """

    adata = adata[:, features].copy()

    if milestones is not None:
        adata = subset_cells(adata, root_milestone, milestones)

    cmap = plt.rcParams["image.cmap"] if cmap is None else cmap

    if (layer == "fitted") & ("fitted" not in adata.layers):
        print("Features not fitted, using X expression matrix instead")
        layer = None

    X = get_X(adata, adata.obs_names, adata.var_names, layer=layer)
    if norm == "max":
        adata.X = X / X.max(axis=0).ravel()
    elif norm == "minmax":
        adata.X = (X - X.min(axis=0).ravel()) / (
            X.max(axis=0).ravel() - X.min(axis=0).ravel()
        )

    maxval = max(X.max(axis=0)) if norm == "none" else 1

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
    to_mil = graph["pp_seg"].loc[adata.obs.seg.cat.categories.astype(int), "to"]
    seg_order = pd.Series(range(len(adata.obs.seg.cat.categories)), index=to_mil)[
        order[order.index.isin(to_mil)].index
    ].values

    vs2mils = pd.Series(dct.keys(), index=dct.values())

    cellsel = [
        adata.obs.milestones[adata.obs.seg == s]
        for s in adata.obs.seg.cat.categories[seg_order]
    ]

    fig, axs = plt.subplots(
        1,
        len(seg_order) + 1 * annot_var,
        constrained_layout=False,
        sharey=True,
        figsize=(
            len(seg_order) * 0.85 + 0.85 + annot_var * 1,
            (len(features)) / (5 - feature_spacing) + 0.4 * annot_top,
        )
        if figsize is None
        else figsize,
    )

    if (len(seg_order) + 1 * annot_var) == 1:
        axs = [axs]

    pos = np.arange(len(seg_order), 0, -1)
    caxs = []
    if return_data:
        datas = dict()
    for i, s in enumerate(seg_order):
        adata_sub = adata[adata.obs.seg == adata.obs.seg.cat.categories[s]].copy()
        adata_sub.obs["split"] = pd.cut(adata_sub.obs.t, bins=nbins)

        # remove unsed categories
        counts = adata_sub.obs.split.value_counts()
        counts = np.array(counts[counts == 0].index.values)
        empty = np.argwhere(
            np.isin(np.array(adata_sub.obs.split.cat.categories), counts)
        )
        if len(empty) > 0:
            logg.warn(f"removed {len(empty)} empty interval")
            adata_sub.obs.split = adata_sub.obs.split.cat.remove_unused_categories()

        if annot_top:
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

            my_cm = LinearSegmentedColormap.from_list("aspect", [start, end])

        if "use_raw" not in kwargs:
            kwargs["use_raw"] = False

        M = sc.pl.MatrixPlot(
            adata_sub,
            features,
            "split",
            vmin=0,
            vmax=maxval,
            cmap=cmap,
            antialiaseds=True,
            linewidth=0.0,
            snap=True,
            edgecolors="none",
            **kwargs,
        )
        M.swap_axes()
        M._mainplot(axs[i])
        axs[i].set_xticklabels("")
        axs[i].set_xticks([])
        plt.margins(y=10)
        plt.setp(axs[i].get_yticklabels(), style=feature_style)
        if annot_top:
            divider = make_axes_locatable(axs[i])
            cax = divider.new_vertical(size=0.2, pad=0.05, pack_start=False)
            caxs.append(cax)
            mappable = cm.ScalarMappable(cmap=my_cm)

            fig.add_axes(cax)
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cbar.set_ticks([])
            cbar.outline.set_linewidth(1.5)

        if return_data:
            data = pd.DataFrame(
                0.0, index=features, columns=adata_sub.obs.split.cat.categories
            )
            allidx = data.stack().index
            for group in data.columns:
                data.loc[data.index, group] = np.array(
                    adata_sub[adata_sub.obs.split == group, data.index].X.mean(axis=0)
                ).ravel()

            k = "->".join(
                vs2mils[
                    graph["pp_seg"].loc[int(adata.obs.seg.cat.categories[s])][
                        ["from", "to"]
                    ]
                ]
            )
            datas[k] = data

    if annot_var:
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
        axs[i + 1].spines["top"].set_visible(False)
        axs[i + 1].spines["left"].set_visible(False)
        axs[i + 1].spines["right"].set_visible(False)

        axs[i + 1].get_xaxis().tick_bottom()
        axs[i + 1].tick_params(left=False)
        axs[i + 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i + 1].grid(False)

        if annot_top:
            divider = make_axes_locatable(axs[i + 1])
            cax = divider.new_vertical(size=0.2, pad=0.05, pack_start=False)
            fig.add_axes(cax)
            cax.annotate("Amplitude", (0, 0), va="bottom", ha="left", size=12)
            cax.axis("off")

    if link_seg:
        caxs_dct = dict(zip([adata.obs.seg.cat.categories[o] for o in seg_order], caxs))

        kw = dict(
            arrowprops=dict(
                arrowstyle="<|-",
                facecolor="k",
                connectionstyle="bar,fraction=.2",
                shrinkA=0.1,
            ),
            zorder=0,
            va="center",
            xycoords="axes fraction",
            annotation_clip=False,
        )

        kwclose = dict(
            arrowprops=dict(
                arrowstyle="<|-",
                facecolor="k",
                connectionstyle="bar,fraction=1",
                shrinkA=0.1,
            ),
            zorder=0,
            va="center",
            xycoords="axes fraction",
            annotation_clip=False,
        )

        pp_seg = adata.uns["graph"]["pp_seg"]

        dsts = pd.Series(range(len(caxs_dct)), index=caxs_dct.keys())
        for s, cax in caxs_dct.items():
            fro = pp_seg.loc[int(s), "from"]
            to = pp_seg.loc[int(s), "to"]
            for n in pp_seg.index[pp_seg["from"] == to]:
                if str(n) not in dsts.index:
                    pass
                elif dsts[str(n)] - dsts[s] > 1:
                    cax.annotate("", xy=[1, 1], xytext=[2.4, 1], **kw)
                else:
                    cax.annotate("", xy=[1, 1], xytext=[1.2, 1], **kwclose)

    if colorbar:
        ax = axs[i + 1 * annot_var]
        position = inset_axes(
            ax,
            width=1,
            height=0.15,
            bbox_to_anchor=(1.04, 0),
            bbox_transform=ax.transAxes,
            loc=3,
            borderpad=2,
        )

        mappable = cm.ScalarMappable(cmap=plt.get_cmap(cmap))
        cbar = fig.colorbar(mappable, cax=position, orientation="horizontal", aspect=50)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(
            [
                "min" if norm == "minmax" else "0",
                "%.2f" % maxval if norm == "none" else "max",
            ]
        )
        title = "expression"
        if colorbar_title is not None:
            title = colorbar_title
        elif norm == "minmax":
            title = "minmax\nnormalized\nexpression"
        elif norm == "max":
            title = "max\nnormalized\nexpression"
        cbar.ax.set_title(title, loc="center")

    savefig_or_show("matrix", show=show, save=save)

    if show == False:
        return axs
    if return_data:
        return datas
