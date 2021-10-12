import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from .dendrogram import dendrogram


def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones,
    col: Union[None, list] = None,
    basis: str = "umap",
    win_keep: Union[None, list] = None,
    frame_emb: bool = True,
    fig_height: float = 6,
    fontsize: int = 16,
    point_size: int = 20,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
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
    fig_height
        figure height.
    fontsize
        repulsion score font size.
    point_size
        correlation plot point size.
    show
        show the plot.
    save
        save the plot.
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
    else:
        genesetA = corAB["A"]["genesetA"].index
        genesetB = corAB["A"]["genesetB"].index
        corA = pd.concat(adata.uns[name]["corAB"]["A"])
        corB = pd.concat(adata.uns[name]["corAB"]["B"])

    groupsA = np.ones(corA.shape[0])
    groupsA[len(genesetA) :] = 2
    groupsB = np.ones(corA.shape[0])
    groupsB[: len(genesetA)] = 2

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
    fig, axs = plt.subplots(2, nwin, figsize=(nwin * fig_height / 2, fig_height))

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    emb = adata.obsm["X_" + basis]

    for i in range(nwin):
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
                ax=axs[0, i],
                show_info=False,
                title="",
                alpha=0,
                **kwargs,
            )
            axs[0, i].set_xlabel("")
            axs[0, i].set_ylabel("")

        axs[0, i].scatter(
            emb[np.argsort(freq), 0],
            emb[np.argsort(freq), 1],
            s=10,
            c=freq[np.argsort(freq)],
            cmap=gr,
            rasterized=True,
        )
        axs[0, i].grid(b=None)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        ratio = plt.rcParams["figure.figsize"][0] / plt.rcParams["figure.figsize"][1]
        axs[0, i].set_aspect(ratio)
        if frame_emb == False:
            axs[0, i].axis("off")

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
        for j in range(2):
            axs[1, i].scatter(
                corA.loc[groupsB == (j + 1), str(i)],
                corB.loc[groupsB == (j + 1), str(i)],
                color=c_mil[j],
                alpha=0.5,
                rasterized=True,
                s=point_size,
            )
        rep = (
            np.corrcoef(groupsA, corA.iloc[:, i])[0][1]
            + np.corrcoef(groupsB, corB.iloc[:, i])[0][1]
        ) / 2
        axs[1, i].annotate(
            str(round(rep, 2)),
            xy=(0.7, 0.88),
            xycoords="axes fraction",
            fontsize=fontsize,
        )
        axs[1, i].grid(b=None)
        axs[1, i].axvline(0, linestyle="dashed", color="grey", zorder=0)
        axs[1, i].axhline(0, linestyle="dashed", color="grey", zorder=0)
        axs[1, i].set_xlim([-maxlim, maxlim])
        axs[1, i].set_ylim([-maxlim, maxlim])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    if show == False:
        return axs

    savefig_or_show("slide_cors", show=show, save=save)
