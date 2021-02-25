import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from scipy import stats
from adjustText import adjust_text
from matplotlib import patches
from scipy import sparse

from typing import Union, Optional, List
from scanpy.plotting._utils import savefig_or_show

from .. import logging as logg
from ..tools.utils import getpath
from .trajectory import trajectory


def trends(
    adata: AnnData,
    features=None,
    highlight_features: Union[List, "A", "fdr"] = "A",
    n_features: int = 10,
    root_milestone: Union[None, str] = None,
    milestones: Union[None, str] = None,
    module: Union[None, "early", "late"] = None,
    branch: Union[None, str] = None,
    annot: Union[None, "seg", "milestones"] = None,
    title: str = "",
    feature_cmap: str = "RdBu_r",
    pseudo_cmap: str = "viridis",
    plot_emb: bool = True,
    basis: str = "umap",
    heatmap_space: float = 0.5,
    offset_names: float = 0.15,
    fontsize: float = 9,
    ordering: Union["pearson", "spearman", "quantile", "max", None] = "pearson",
    ord_thre=0.7,
    filter_complex: bool = False,
    complex_thre: float = 0.7,
    complex_z: float = 3,
    fig_heigth: float = 4,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    save_genes: Optional[bool] = None,
    **kwargs,
):

    """\
    Plot a set of fitted features over pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    features
        Name of the fitted features.
    highlight_features
        which features will be annotated on the heatmap, by default, features with highest amplitude are shown.
    n_features
        number of top features to show if no list are provided.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    module
        if bifurcation analysis as been performed, subset features to a specific module.
    branch
        if bifurcation analysis as been performed, subset features to a specific milestone.
    annot
        adds an annotation row o ntop of the heatmap.
    title
        add a title.
    feature_cmap
        colormap for features.
    pseudo_cmap
        colormap for pseudotime.
    plot_emb
        call pl.trajectory on the left side.
    basis
        Name of the `obsm` basis to use if plot_emb is True.
    heatmap_space
        how much space does the heatmap take, in proportion of the whole plot space.
    offset_names
        how far on the right the annotated features should be displayed, in proportion of the heatmap space.
    fontsize
        font size of the feature annotations.
    ordering
        strategy to order the features on heatmap, quantile takes the mean pseudotime of the choosen value.
    ord_thre
        for 'max': proportion of maximum of fitted value to consider to compute the mean pseudotime.
        for 'quantile': quantile to consider to compute the mean pseudotime.
        for 'pearson'/'spearman': proportion of max value to assign starting cell.
    filter_complex
        filter out complex features
    complex_thre
        quantile to consider for complex filtering
    complex_z
        standard deviation threshold on the quantile subsetted feature.
    fig_heigth
        figure height.
    show
        show the plot.
    save
        save the plot.
    save_genes
        save list of genes following the order displayed o nthe heatmap.

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    offset_heatmap = 1 - heatmap_space

    if plot_emb:
        adata_temp = adata.copy()

    graph = adata.uns["graph"]

    if root_milestone is not None:
        adata = adata.copy()
        dct = graph["milestones"]

        leaves = list(map(lambda leave: dct[leave], milestones))
        root = dct[root_milestone]
        df = adata.obs.copy(deep=True)
        edges = (
            adata.uns["graph"]["pp_seg"][["from", "to"]]
            .astype(str)
            .apply(tuple, axis=1)
            .values
        )
        img = igraph.Graph()
        img.add_vertices(
            np.unique(
                adata.uns["graph"]["pp_seg"][["from", "to"]]
                .values.flatten()
                .astype(str)
            )
        )
        img.add_edges(edges)
        cells = np.unique(
            np.concatenate(
                list(
                    map(
                        lambda leave: getpath(
                            img,
                            root,
                            adata.uns["graph"]["tips"],
                            leave,
                            adata.uns["graph"],
                            df,
                        ).index,
                        leaves,
                    )
                )
            )
        )

        adata = adata[cells]

    if features is None:
        features = adata.var_names
    if module is not None and branch is not None:
        name = root_milestone + "->" + "<>".join(milestones)
        df = adata.uns[name]["fork"]
        features = df.loc[(df.branch == branch) & (df.module == module), :].index

    fitted = pd.DataFrame(
        adata.layers["fitted"], index=adata.obs_names, columns=adata.var_names
    ).T.copy(deep=True)
    g = adata.obs.groupby("seg")
    seg_order = g.apply(lambda x: np.mean(x.t)).sort_values().index.tolist()
    cell_order = np.concatenate(
        list(
            map(
                lambda x: adata.obs.t[adata.obs.seg == x].sort_values().index, seg_order
            )
        )
    )
    fitted = fitted.loc[:, cell_order]
    # fitted=fitted.apply(lambda x: (x-x.mean())/x.std(),axis=1)

    fitted = fitted.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    if filter_complex:
        # remove complex features using quantiles
        varia = list(
            map(
                lambda x: adata.obs.t[cell_order][
                    fitted.loc[x, :].values
                    > np.quantile(fitted.loc[x, :].values, q=complex_thre)
                ].var(),
                features,
            )
        )
        z = np.abs(stats.zscore(varia))
        torem = np.argwhere(z > complex_z).flatten()

        if len(torem) > 0:
            logg.info("found " + str(len(torem)) + " complex fitted features")
            logg.hint("added\n" + "    'complex' column in (adata.var)")
            adata.var["complex"] = False
            # adata.var.iloc[torem,"complex"]=True
            adata.var.loc[fitted.index[torem], "complex"] = True
            features = adata.var_names[~adata.var["complex"]]

    fitted = fitted.loc[features, :]

    if ordering == "quantile":
        feature_order = (
            fitted.apply(
                lambda x: adata.obs.t[fitted.columns][
                    x > np.quantile(x, q=ord_thre)
                ].mean(),
                axis=1,
            )
            .sort_values()
            .index
        )
    elif ordering == "max":
        feature_order = (
            fitted.apply(
                lambda x: adata.obs.t[fitted.columns][
                    (x - x.min()) / (x.max() - x.min()) > ord_thre
                ].mean(),
                axis=1,
            )
            .sort_values()
            .index
        )
    elif ordering in ("pearson", "spearman"):
        start_feature = (
            fitted.apply(
                lambda x: adata.obs.t[fitted.columns][
                    (x - x.min()) / (x.max() - x.min()) > ord_thre
                ].mean(),
                axis=1,
            )
            .sort_values()
            .index[0]
        )
        feature_order = (
            fitted.T.corr(method=ordering)
            .loc[start_feature, :]
            .sort_values(ascending=False)
            .index
        )

    else:
        feature_order = fitted.index

    fitted_sorted = fitted.loc[feature_order, :]

    if annot == "milestones":
        color_key = "milestones_colors"
        if color_key not in adata.uns or len(adata.uns[color_key]) == 1:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")

        def milestones_prog(s):
            cfrom = adata.obs.t[adata.obs.seg == s].idxmin()
            cto = adata.obs.t[adata.obs.seg == s].idxmax()
            mfrom = adata.obs.milestones[cfrom]
            mto = adata.obs.milestones[cto]
            import numpy as np

            mfrom_c = adata.uns["milestones_colors"][
                np.argwhere(adata.obs.milestones.cat.categories == mfrom)[0][0]
            ]
            mto_c = adata.uns["milestones_colors"][
                np.argwhere(adata.obs.milestones.cat.categories == mto)[0][0]
            ]

            from matplotlib.colors import LinearSegmentedColormap

            cm = LinearSegmentedColormap.from_list("test", [mfrom_c, mto_c], N=1000)
            pst = (
                adata.obs.t[adata.obs.seg == s] - adata.obs.t[adata.obs.seg == s].min()
            ) / (
                adata.obs.t[adata.obs.seg == s].max()
                - adata.obs.t[adata.obs.seg == s].min()
            )
            return pd.Series(list(map(to_hex, cm(pst))), index=pst.index)

        annot_cmap = pd.concat(list(map(milestones_prog, seg_order)))

    if annot == "seg":
        color_key = "seg_colors"
        if color_key not in adata.uns or len(adata.uns[color_key]) == 1:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, "seg")

        annot_cmap = pd.Series(
            list(
                map(
                    lambda s: adata.uns["seg_colors"][
                        adata.obs.seg.cat.categories == s
                    ][0],
                    adata.obs.seg[fitted_sorted.columns].values,
                )
            ),
            index=fitted_sorted.columns,
        )

    fig, f_axs = plt.subplots(
        ncols=2,
        nrows=20,
        figsize=(fig_heigth * (1 + 1 * plot_emb), fig_heigth),
        gridspec_kw={"width_ratios": [1 * plot_emb, 1]},
    )
    gs = f_axs[2, 1].get_gridspec()

    # remove the underlying axes
    start = 2 if annot is not None else 1
    for ax in f_axs[start:, -1]:
        ax.remove()
    axheatmap = fig.add_subplot(gs[start:, -1])

    gs = f_axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in f_axs[:, 0]:
        ax.remove()

    if annot is not None:
        axannot = f_axs[0, 1]
        sns.heatmap(
            pd.DataFrame(range(fitted_sorted.shape[1])).T,
            robust=False,
            rasterized=True,
            cmap=annot_cmap[fitted_sorted.columns].values.tolist(),
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            ax=axannot,
        )
        axpsdt = f_axs[1, 1]

    else:
        axpsdt = f_axs[0, 1]

    sns.heatmap(
        pd.DataFrame(adata.obs.t[fitted_sorted.columns].values).T,
        robust=True,
        rasterized=True,
        cmap=pseudo_cmap,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=axpsdt,
        vmax=adata.obs.t.max(),
    )

    sns.heatmap(
        fitted_sorted,
        robust=True,
        cmap=feature_cmap,
        rasterized=True,
        xticklabels=False,
        yticklabels=False,
        ax=axheatmap,
        cbar=False,
    )

    def add_frames(axis, vert):
        rect = patches.Rectangle(
            (0, 0),
            len(fitted_sorted.columns),
            vert,
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )
        # Add the patch to the Axes
        axis.add_patch(rect)
        offset = 0
        for s in seg_order[:-1]:
            prev_offset = offset
            offset = offset + (adata.obs.seg == s).sum()
            rect = patches.Rectangle(
                (prev_offset, 0),
                (adata.obs.seg == s).sum(),
                vert,
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            axis.add_patch(rect)
        return axis

    axpsdt = add_frames(axpsdt, 1)

    if annot is not None:
        axannot = add_frames(axannot, 1)

    axheatmap = add_frames(axheatmap, fitted_sorted.shape[0])

    if highlight_features is "A":
        highlight_features = (
            adata.var.A[features].sort_values(ascending=False)[:n_features].index
        )
    elif highlight_features is "fdr":
        highlight_features = (
            adata.var.fdr[features].sort_values(ascending=True)[:n_features].index
        )
    xs = np.repeat(fitted_sorted.shape[1], len(highlight_features))
    ys = (
        np.array(
            list(
                map(
                    lambda g: np.argwhere(fitted_sorted.index == g)[0][0],
                    highlight_features,
                )
            )
        )
        + 0.5
    )

    texts = []
    for x, y, s in zip(xs, ys, highlight_features):
        texts.append(axheatmap.text(x, y, s, fontsize=fontsize))

    patch = patches.Rectangle(
        (0, 0),
        fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_names,
        fitted_sorted.shape[0],
        alpha=0,
    )  # We add a rectangle to make sure the labels don't move to the right
    axpsdt.set_xlim(
        (0, fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_heatmap)
    )
    if annot is not None:
        axannot.set_xlim(
            (0, fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_heatmap)
        )
    axheatmap.set_xlim(
        (0, fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_heatmap)
    )
    axheatmap.add_patch(patch)
    axheatmap.hlines(
        fitted_sorted.shape[0], 0, fitted_sorted.shape[1], color="k", clip_on=True
    )

    adjust_text(
        texts,
        ax=axheatmap,
        add_objects=[patch],
        va="center",
        ha="left",
        autoalign=False,
        expand_text=(1.05, 1.2),
        lim=5000,
        only_move={"text": "y", "objects": "x"},
        precision=0.1,
        expand_points=(1.2, 1.05),
    )

    for i in range(len(xs)):
        xx = [xs[i] + 1, fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_names]
        yy = [ys[i], texts[i].get_position()[1]]
        axheatmap.plot(xx, yy, color="k", linewidth=0.75)

    plt.tight_layout(h_pad=0, w_pad=0)

    if plot_emb:
        axemb = fig.add_subplot(gs[:, 0])
        adata_temp.obs["mean_trajectory"] = np.nan
        adata_temp.obs.loc[
            fitted_sorted.columns, "mean_trajectory"
        ] = fitted_sorted.mean(axis=0).values
        trajectory(
            adata=adata_temp,
            basis=basis,
            color_seg="mean_trajectory",
            cmap_seg=feature_cmap,
            color_cells=annot,
            show_info=False,
            ax=axemb,
            title=title,
            root_milestone=root_milestone,
            milestones=milestones,
            **kwargs,
        )

    if save_genes is not None:
        with open(save_genes, "w") as f:
            for item in fitted_sorted.index:
                f.write("%s\n" % item)

    if show == False:
        return f_axs

    savefig_or_show("trends", show=show, save=save)


def single_trend(
    adata: AnnData,
    feature: str,
    basis: str = "umap",
    ylab: str = "expression",
    color_exp=None,
    alpha_expr: float = 0.3,
    size_expr: float = 2,
    fitted_linewidth: float = 2,
    layer: Union[str, None] = None,
    cmap_seg: str = "RdBu_r",
    cmap_cells: str = "RdBu_r",
    figsize: tuple = (8, 4),
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):

    """\
    Plot a single feature fit over pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    feature
        Name of the fitted feature
    basis
        Name of the `obsm` basis to use.
    ylab
        ylabel of right plot.
    colo_rexp
        color of raw datapoints on right plot.
    alpha_expr
        alpha of raw datapoints on right plot.
    size_expr
        size of raw datapoints on right plot.
    fitted_linewidth
        linewidth of fitted line on right plot.
    layer
        layer to plot for the raw datapoints.
    cmap_seg
        colormap for trajectory segments on left plot.
    cmap_cells
        colormap for cells on left plot.
    figsize
        figure size in inches
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a tuple of two :class:`~matplotlib.axes.Axes`

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    color_key = "seg_colors"
    if color_key not in adata.uns:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, "seg")

    if layer is None:
        if sparse.issparse(adata.X):
            Xfeature = adata[:, feature].X.A.T.flatten()
        else:
            Xfeature = adata[:, feature].X.T.flatten()
    else:
        if sparse.issparse(adata.layers[layer]):
            Xfeature = adata[:, feature].layers[layer].A.T.flatten()
        else:
            Xfeature = adata[:, feature].layers[layer].T.flatten()

    df = pd.DataFrame(
        {
            "t": adata.obs.t,
            "fitted": adata[:, feature].layers["fitted"].flatten(),
            "expression": Xfeature,
            "seg": adata.obs.seg,
        }
    ).sort_values("t")

    for s in df.seg.unique():
        if color_exp is None:
            ax2.scatter(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "expression"],
                alpha=alpha_expr,
                s=size_expr,
                c=adata.uns["seg_colors"][
                    np.argwhere(adata.obs.seg.cat.categories == s)[0][0]
                ],
            )
            ax2.plot(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "fitted"],
                c=adata.uns["seg_colors"][
                    np.argwhere(adata.obs.seg.cat.categories == s)[0][0]
                ],
                linewidth=fitted_linewidth,
            )
            tolink = adata.uns["graph"]["pp_seg"].loc[int(s), "to"]
            for next_s in adata.uns["graph"]["pp_seg"].n.iloc[
                np.argwhere(
                    adata.uns["graph"]["pp_seg"].loc[:, "from"].isin([tolink]).values
                ).flatten()
            ]:
                ax2.plot(
                    [
                        df.loc[df.seg == s, "t"].iloc[-1],
                        df.loc[df.seg == next_s, "t"].iloc[0],
                    ],
                    [
                        df.loc[df.seg == s, "fitted"].iloc[-1],
                        df.loc[df.seg == next_s, "fitted"].iloc[0],
                    ],
                    c=adata.uns["seg_colors"][
                        np.argwhere(adata.obs.seg.cat.categories == next_s)[0][0]
                    ],
                    linewidth=fitted_linewidth,
                )
        else:
            ax2.scatter(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "expression"],
                c=color_exp,
                alpha=alpha_expr,
                s=size_expr,
            )
            ax2.plot(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "fitted"],
                c=color_exp,
                linewidth=fitted_linewidth,
            )
            tolink = adata.uns["graph"]["pp_seg"].loc[int(s), "to"]
            for next_s in adata.uns["graph"]["pp_seg"].n.iloc[
                np.argwhere(
                    adata.uns["graph"]["pp_seg"].loc[:, "from"].isin([tolink]).values
                ).flatten()
            ]:
                ax2.plot(
                    [
                        df.loc[df.seg == s, "t"].iloc[-1],
                        df.loc[df.seg == next_s, "t"].iloc[0],
                    ],
                    [
                        df.loc[df.seg == s, "fitted"].iloc[-1],
                        df.loc[df.seg == next_s, "fitted"].iloc[0],
                    ],
                    c=color_exp,
                    linewidth=fitted_linewidth,
                )

    ax2.set_ylabel(ylab)
    ax2.set_xlabel("pseudotime")
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    trajectory(
        adata,
        basis=basis,
        color_seg=feature,
        cmap_seg=cmap_seg,
        color_cells=feature,
        cmap=cmap_cells,
        show_info=False,
        ax=ax1,
        title=feature,
        layer="fitted",
        show=False,
        save=False,
        **kwargs,
    )

    savefig_or_show("single_trend", show=show, save=save)

    if show is False:
        return (ax1, ax2)
