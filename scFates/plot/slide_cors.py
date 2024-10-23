import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex
from matplotlib.patches import ConnectionPatch

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from scFates.plot.dendrogram import dendrogram
from adjustText import adjust_text
from matplotlib import gridspec
from sklearn.metrics import pairwise_distances

from .. import get


def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones,
    col: Union[None, list] = None,
    basis: str = "umap",
    win_keep: Union[None, list] = None,
    frame_emb: bool = True,
    focus=None,
    top_focus=4,
    labels: Union[None, tuple] = None,
    fig_height: float = 6,
    fontsize: int = 16,
    fontsize_focus: int = 18,
    point_size: int = 20,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    kwargs_text: dict = {},
    kwargs_con: dict = {},
    kwargs_adjust: dict = {},
    **kwargs,
):

    """\
    Plot results generated from tl.slide_cors.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    genesetA
        plot correlation with custom geneset.
    genesetB
        plot correlation with custom geneset.
    col
        specify color for the two modules, by default according to their respective milestones.
    basis
        Name of the `obsm` basis to use.
    win_keep
        plot only a subset of windows.
    frame_emb
        add frame around emb plot.
    focus
        add on the right side a scatter focusing one defined window.
    top_focus
        highlight n top markers for each module, having the greatest distance to 0,0 coordinates.
    labels
        labels defining the two modules, named after the milestones if None, or 'A' and 'B' if less than two milestones is used.
    fig_height
        figure height.
    fontsize
        repulsion score font size.
    fontsize_focus
        fontsize of x and y labels in focus plot.
    point_size
        correlation plot point size.
    show
        show the plot.
    save
        save the plot.
    kwargs_text
        parameters for the text annotation of the labels.
    kwargs_con
        parameters passed on the ConnectionPatch linking the focused plot to the rest.
    kwargs_adjust
        parameters passed to adjust_text.
    **kwargs
        if `basis=dendro`, arguments passed to :func:`scFates.pl.dendrogram`

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    if "milestones_colors" not in adata.uns or len(adata.uns["milestones_colors"]) == 1:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")

    mlsc = np.array(adata.uns["milestones_colors"].copy())
    if mlsc.dtype == "float":
        mlsc = list(map(rgb2hex, mlsc))

    name = root_milestone + "->" + "<>".join(milestones)
    freqs = adata.uns[name]["cell_freq"]
    nwin = len(freqs)

    corAB = adata.uns[name]["corAB"].copy()

    if len(milestones) > 1:
        genesetA = corAB[milestones[0]]["genesetA"].index
        genesetB = corAB[milestones[0]]["genesetB"].index
        corA = pd.concat(adata.uns[name]["corAB"][milestones[0]])
        corB = pd.concat(adata.uns[name]["corAB"][milestones[1]])
        if labels is None:
            labelA, labelB = milestones if labels is None else labels
    else:
        genesetA = corAB["A"]["genesetA"].index
        genesetB = corAB["A"]["genesetB"].index
        corA = pd.concat(adata.uns[name]["corAB"]["A"])
        corB = pd.concat(adata.uns[name]["corAB"]["B"])
        labelA, labelB = ("A", "B") if labels is None else labels

    groupsA = np.ones(corA.shape[0])
    groupsA[: len(genesetA)] = 2
    groupsB = np.ones(corA.shape[0])
    groupsB[len(genesetA) :] = 2

    gr = LinearSegmentedColormap.from_list("greyreds", ["lightgrey", "black"])

    if win_keep is not None:
        freqs = [freqs[i] for i in win_keep]
        corA = corA.iloc[:, win_keep]
        corB = corB.iloc[:, win_keep]
        corA.columns = np.arange(len(win_keep)).astype(str)
        corB.columns = np.arange(len(win_keep)).astype(str)

    maxlim = (
        np.max(
            [
                corB.max().max(),
                np.abs(corB.min().min()),
                corA.max().max(),
                np.abs(corA.min().min()),
            ]
        )
        + 0.01
    )

    nwin = len(freqs)

    focus_true = False if focus is None else True

    fig = plt.figure(figsize=((nwin + focus_true * 2) * fig_height / 2, fig_height))

    gs0 = gridspec.GridSpec(
        1,
        1 + focus_true * 1,
        figure=fig,
        width_ratios=[nwin, 2] if focus_true else None,
        wspace=0.05,
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, nwin, subplot_spec=gs0[0], hspace=0.05, wspace=0.05
    )

    emb = adata.obsm["X_" + basis]

    for i in range(nwin):
        ax_emb = fig.add_subplot(gs00[0, i])
        freq = freqs[i]
        if basis == "dendro":
            if "s" in kwargs:
                s = kwargs["s"]
                del kwargs["s"]
            else:
                s = 10
            dendrogram(
                adata,
                s=s,
                cmap=gr,
                show=False,
                ax=ax_emb,
                show_info=False,
                title="",
                alpha=0,
                **kwargs,
            )
            ax_emb.set_xlabel("")
            ax_emb.set_ylabel("")

        ax_emb.scatter(
            emb[np.argsort(freq), 0],
            emb[np.argsort(freq), 1],
            s=10,
            c=freq.iloc[np.argsort(freq)],
            cmap=gr,
            rasterized=True,
        )
        ax_emb.grid(None)
        ax_emb.set_xticks([])
        ax_emb.set_yticks([])
        ma = np.max(plt.rcParams["figure.figsize"])
        mi = np.min(plt.rcParams["figure.figsize"])
        toreduce = plt.rcParams["figure.figsize"] / ma
        if len(np.argwhere(toreduce != 1)) == 0:
            pass
        elif np.argwhere(toreduce != 1)[0][0] == 0:
            span = ax_emb.get_xlim()[1] - ax_emb.get_xlim()[0]
            midpoint = ax_emb.get_xlim()[0] + span / 2
            enlargment = (ax_emb.get_xlim()[1] - ax_emb.get_xlim()[0]) * (ma / mi) / 2
            ax_emb.set_xlim([midpoint - enlargment, midpoint + enlargment])
        elif np.argwhere(toreduce != 1)[0][0] == 1:
            span = ax_emb.get_ylim()[1] - ax_emb.get_ylim()[0]
            midpoint = ax_emb.get_ylim()[0] + span / 2
            enlargment = (ax_emb.get_ylim()[1] - ax_emb.get_ylim()[0]) * (ma / mi) / 2
            ax_emb.set_ylim([midpoint - enlargment, midpoint + enlargment])
        if frame_emb == False:
            ax_emb.axis("off")

    c_mil = (
        [
            np.array(mlsc)[adata.obs.milestones.cat.categories == miles]
            for miles in milestones
        ]
        if len(milestones) == 2
        else ["tab:blue", "tab:red"]
    )
    c_mil = col if col is not None else c_mil
    genesets = [genesetA, genesetB]
    for i in range(nwin):
        ax_scat = fig.add_subplot(gs00[1, i])
        for j in range(2):
            ax_scat.scatter(
                corA.loc[groupsB == (j + 1), str(i)],
                corB.loc[groupsB == (j + 1), str(i)],
                color=c_mil[j],
                alpha=0.5,
                rasterized=True,
                s=point_size,
            )
        tokeep = ~corA.iloc[:, i].isna()
        rep = (
            np.corrcoef(groupsA[tokeep], corA.loc[tokeep].iloc[:, i])[0][1]
            + np.corrcoef(groupsB[tokeep], corB.loc[tokeep].iloc[:, i])[0][1]
        ) / 2
        ax_scat.annotate(
            str(round(np.abs(rep), 2)),
            xy=(0.7, 0.88),
            xycoords="axes fraction",
            fontsize=fontsize,
        )
        ax_scat.grid(None)
        ax_scat.axvline(0, linestyle="dashed", color="grey", zorder=0)
        ax_scat.axhline(0, linestyle="dashed", color="grey", zorder=0)
        ax_scat.set_xlim([-maxlim, maxlim])
        ax_scat.set_ylim([-maxlim, maxlim])
        ax_scat.set_xticks([])
        ax_scat.set_yticks([])
        if i == focus:
            ax_focus = ax_scat
            ax_focus.tick_params(color="red", labelcolor="red")
            for spine in ax_focus.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(spine.get_linewidth() * 2)

    if focus_true:
        ax_scat = fig.add_subplot(gs0[1])
        for j in range(2):
            ax_scat.scatter(
                corA.loc[groupsB == (j + 1), str(focus)],
                corB.loc[groupsB == (j + 1), str(focus)],
                color=c_mil[j],
                alpha=0.5,
                rasterized=True,
                s=point_size * 2,
            )
        ax_scat.grid(None)
        ax_scat.axvline(0, linestyle="dashed", color="grey", zorder=0)
        ax_scat.axhline(0, linestyle="dashed", color="grey", zorder=0)
        ax_scat.set_xlim([-maxlim, maxlim])
        ax_scat.set_ylim([-maxlim, maxlim])
        ax_scat.set_xticks([])
        ax_scat.set_yticks([])
        ax_scat.set_xlabel("correlation with %s" % labelA, fontsize=fontsize_focus)
        ax_scat.set_ylabel("correlation with %s" % labelB, fontsize=fontsize_focus)

        fra = 0.1 / (len(freqs) - focus)

        con = ConnectionPatch(
            xyA=(maxlim * 0.75, -maxlim),
            coordsA="data",
            xyB=(-maxlim * 0.75, -maxlim),
            coordsB="data",
            axesA=ax_focus,
            axesB=ax_scat,
            arrowstyle="->",
            connectionstyle=f"bar,fraction={fra}",
            **kwargs_con,
        )
        ax_scat.add_artist(con)
        con.set_linewidth(2)
        con.set_color("red")

        fA = pd.concat(
            [corA.loc["genesetA"][str(focus)], corB.loc["genesetA"][str(focus)]], axis=1
        )
        fA = fA.loc[
            (fA.iloc[:, 0].values > 0) & (fA.iloc[:, 1].values < 0),
        ]
        fB = pd.concat(
            [corA.loc["genesetB"][str(focus)], corB.loc["genesetB"][str(focus)]], axis=1
        )
        fB = fB.loc[
            (fB.iloc[:, 1].values > 0) & (fB.iloc[:, 0].values < 0),
        ]
        if top_focus > 0:
            topA = np.flip(
                pairwise_distances(np.array([0, 0]).reshape(1, -1), fA).argsort()
            )[0][:top_focus]
            topB = np.flip(
                pairwise_distances(np.array([0, 0]).reshape(1, -1), fB).argsort()
            )[0][:top_focus]
            texts_A = [
                ax_scat.text(fA.loc[t].iloc[0], fA.loc[t].iloc[1], t, **kwargs_text)
                for t in fA.index[topA]
            ]
            texts_B = [
                ax_scat.text(fB.loc[t].iloc[0], fB.loc[t].iloc[1], t, **kwargs_text)
                for t in fB.index[topB]
            ]
            texts = texts_A + texts_B
            adjust_text(texts, **kwargs_adjust)

    savefig_or_show("slide_cors", show=show, save=save)
