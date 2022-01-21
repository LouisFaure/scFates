import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text
from matplotlib import patches
from scipy import sparse
import matplotlib.gridspec as gridspec

from typing import Union, Optional, List
from typing_extensions import Literal
from scanpy.plotting._utils import savefig_or_show
import scanpy as sc

from .. import logging as logg
from ..tools.utils import getpath, importeR
from .trajectory import trajectory
from .dendrogram import dendrogram

Rpy2, R, rstats, rmgcv, Formula = importeR("fitting associated features")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]

from ..get import modules as get_modules
from .trajectory import remove_info
from .utils import gen_milestones_gradients
from .. import logging as logg


def trends(
    adata: AnnData,
    features=None,
    cluster=None,
    highlight_features: Union[List, Literal["A", "fdr"]] = "A",
    n_features: int = 10,
    root_milestone: Union[None, str] = None,
    milestones: Union[None, str] = None,
    module: Union[None, Literal["early", "late"]] = None,
    branch: Union[None, str] = None,
    annot: Union[None, str] = None,
    title: str = "",
    feature_cmap: str = "RdBu_r",
    pseudo_cmap: str = "viridis",
    plot_emb: bool = True,
    plot_heatmap: bool = True,
    wspace: Union[None, float] = None,
    show_segs: bool = True,
    basis: str = "umap",
    heatmap_space: float = 0.5,
    offset_names: float = 0.15,
    fontsize: int = 9,
    style: Literal["normal", "italic", "oblique"] = "normal",
    ordering: Union[
        None, Literal["pearson", "spearman", "quantile", "max"]
    ] = "pearson",
    ord_thre=0.7,
    filter_complex: bool = False,
    complex_thre: float = 0.7,
    complex_z: float = 3,
    figsize: Union[None, tuple] = None,
    axemb=None,
    show: Optional[bool] = None,
    output_mean: bool = False,
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
        adds an annotation row on top of the heatmap.
    title
        add a title.
    feature_cmap
        colormap for features.
    pseudo_cmap
        colormap for pseudotime.
    plot_emb
        call pl.trajectory on the left side.
    plot_heatmap
        show heatmap on the right side.
    wspace
        width space between emb and heatmap.
    show_segs
        display segments on emb.
    basis
        Name of the `obsm` basis to use if plot_emb is True.
    heatmap_space
        how much space does the heatmap take, in proportion of the whole plot space.
    offset_names
        how far on the right the annotated features should be displayed, in proportion of the heatmap space.
    fontsize
        font size of the feature annotations.
    style
        font style.
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
    figsize
        figure size.
    axemb
        existing ax for plotting emb
    output_mean
        output mean fitted values to adata.
    show
        show the plot.
    save
        save the plot.
    save_genes
        save list of genes following the order displayed on the heatmap.
    **kwargs
        if `plot_emb=True`, arguments passed to :func:`scFates.pl.trajectory` or :func:`scFates.pl.dendrogram` if `basis="dendro"`

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    offset_heatmap = 1 - heatmap_space

    if plot_emb:
        adata_temp = adata.copy()

    graph = adata.uns["graph"]

    if milestones is not None:
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

    if (features is None) & (cluster is None):
        features = adata.var_names
    if cluster is not None:
        features = adata.var_names[adata.var.fit_clusters == cluster]

    if branch is not None:
        name = root_milestone + "->" + "<>".join(milestones)
        df = adata.uns[name]["fork"]
        if module is not None:
            sel = (df.branch == branch) & (df.module == module)
        else:
            sel = df.branch == branch
        features = df.loc[sel, :].index

    fitted = pd.DataFrame(
        adata.layers["fitted"], index=adata.obs_names, columns=adata.var_names
    ).T.copy(deep=True)
    g = adata.obs.groupby("seg")
    seg_order = g.apply(lambda x: np.min(x.t)).sort_values().index.tolist()
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
        annot_cmap = gen_milestones_gradients(adata, seg_order)

    elif type(annot) == str:
        if len(adata.obs[annot].unique()) > 1:
            color_key = annot + "_colors"
            if color_key not in adata.uns or len(adata.uns[color_key]) == 1:
                from . import palette_tools

                palette_tools._set_default_colors_for_categorical_obs(adata, annot)
            annot_cmap = pd.Series(
                list(
                    map(
                        lambda c: adata.uns[color_key][
                            adata.obs[annot].cat.categories == c
                        ][0],
                        adata.obs[annot][fitted_sorted.columns].values,
                    )
                ),
                index=fitted_sorted.columns,
            )
    else:
        annot = None

    if axemb is None:
        ratio = plt.rcParams["figure.figsize"][0] / plt.rcParams["figure.figsize"][1]
        figsize = (
            ((4 * plot_emb * ratio) + 4 * plot_heatmap, 4)
            if figsize is None
            else figsize
        )
        fig = plt.figure(figsize=figsize)
        gsubs = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1 * plot_emb * ratio, 1 * plot_heatmap],
            wspace=wspace,
        )
        axs = []
    if plot_heatmap:
        gs_ht = gridspec.GridSpecFromSubplotSpec(
            2 + (annot is not None),
            1,
            height_ratios=(1, 1, 18) if (annot is not None) else (1, 19),
            subplot_spec=gsubs[1],
            hspace=0,
        )
        i = 0
        if annot is not None:
            axannot = plt.subplot(gs_ht[i])
            axs = axs + [axannot]
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
            i = i + 1

        axpsdt = plt.subplot(gs_ht[i])
        axs = axs + [axpsdt]
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

        axheatmap = plt.subplot(gs_ht[i + 1])
        axs = axs + [axheatmap]
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

        if highlight_features == "A":
            highlight_features = (
                adata.var.A[features].sort_values(ascending=False)[:n_features].index
            )
        elif highlight_features == "fdr":
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
            texts.append(axheatmap.text(x, y, s, fontsize=fontsize, style=style))

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
            xx = [
                xs[i] + 1,
                fitted_sorted.shape[1] + fitted_sorted.shape[1] * offset_names,
            ]
            yy = [ys[i], texts[i].get_position()[1]]
            axheatmap.plot(xx, yy, color="k", linewidth=0.75)

    if plot_emb:
        if axemb is None:
            gs_emb = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsubs[0])
            axemb = fig.add_subplot(gs_emb[0])
            axs = axs + [axemb]
        adata_temp.obs["mean_trajectory"] = 0
        adata_temp.obs.loc[
            fitted_sorted.columns, "mean_trajectory"
        ] = fitted_sorted.mean(axis=0).values
        if output_mean:
            adata.obs["mean_trajectory"] = adata_temp.obs["mean_trajectory"]
        if "color" in kwargs:
            color = kwargs["color"]
            del kwargs["color"]
        else:
            color = "mean_trajectory"

        if "cmap" in kwargs:
            feature_cmap = kwargs["cmap"]
            del kwargs["cmap"]

        if basis == "dendro":
            dendrogram(
                adata_temp,
                color=color,
                cmap=feature_cmap,
                ax=axemb,
                title=title,
                root_milestone=root_milestone,
                milestones=milestones,
                show_info=False,
                **kwargs,
            )
        elif show_segs == False:
            sc.pl.embedding(
                adata=adata_temp,
                basis=basis,
                color=color,
                cmap=feature_cmap,
                ax=axemb,
                title=title,
                show=False,
                **kwargs,
            )
            remove_info(adata_temp, axemb, color)
        else:
            trajectory(
                adata=adata_temp,
                basis=basis,
                color_seg=color,
                color_cells=color,
                cmap_seg=feature_cmap,
                cmap=feature_cmap,
                show_info=False,
                ax=axemb,
                title=title,
                root_milestone=root_milestone,
                milestones=milestones,
                show=False,
                **kwargs,
            )

    if save_genes is not None:
        with open(save_genes, "w") as f:
            for item in fitted_sorted.index:
                f.write("%s\n" % item)

    if show == False:
        return axs if plot_heatmap else axemb
    if save is not None:
        savefile = "figures/trends" + save
        logg.warn("saving figure to file " + savefile)
        fig.savefig(savefile, bbox_inches="tight")


def single_trend(
    adata: AnnData,
    feature: Union[str, None] = None,
    root_milestone: Union[None, str] = None,
    milestones: Union[None, str] = None,
    module: Union[None, Literal["early", "late"]] = None,
    branch: Union[None, str] = None,
    basis: str = "umap",
    ylab: str = "expression",
    color_exp=None,
    alpha_expr: float = 0.3,
    size_expr: float = 2,
    fitted_linewidth: float = 2,
    layer: Union[str, None] = None,
    cmap_seg: str = "RdBu_r",
    cmap_cells: str = "RdBu_r",
    plot_emb: bool = True,
    wspace: Union[None, float] = None,
    figsize: tuple = (8, 4),
    ax_trend=None,
    ax_emb=None,
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
        Name of the fitted feature.
    root_milestone
        if plotting module instead of feature, tip defining progenitor branch.
    milestones
        if plotting module instead of feature, tips defining the progenies branches.
    module
        if plotting module instead of feature, whether to plot early or late modules.
    branch
        if plotting module instead of feature, plot fitted milestone-specific module.
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
    plot_emb
        plot the emb on the left side.
    wspace
        width space between emb and heatmap.
    figsize
        figure size in inches.
    ax_trend
        existing ax for trends, only works when emb plot is disabled.
    ax_emb
        existing ax for embedding plot.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a tuple of two :class:`~matplotlib.axes.Axes`

    """
    if root_milestone is None:
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
    else:
        if plot_emb:
            logg.warn("trajectory plotting is not yet implemented for module plot!")
            plot_emb = False
        graph = adata.uns["graph"]
        miles_cat = adata.obs.milestones.cat.categories
        mil_col = np.array(adata.uns["milestones_colors"])

        X_modules = get_modules(adata, root_milestone, milestones, layer)
        X_early, X_late = X_modules.iloc[:, :2], X_modules.iloc[:, 2:]

        X = pd.concat([X_early, X_late], axis=1)
        X["t"] = adata.obs.loc[X.index, "t"].values

        def gamfit(f):
            m = rmgcv.gam(
                Formula(f),
                data=X,
                gamma=1.5,
            )
            return pd.Series(rmgcv.predict_gam(m), index=X.index)

        predictions = gamfit("{}_{}~s(t,bs='ts')".format(module, branch))

        df = pd.DataFrame(
            {
                "t": adata[X.index].obs.t,
                "fitted": predictions.values,
                "expression": X["_".join([module, branch])].values,
                "seg": adata[X.index].obs.seg,
            }
        ).sort_values("t")

        color_exp = mil_col[miles_cat == branch][0]
    ratio = plt.rcParams["figure.figsize"][0] / plt.rcParams["figure.figsize"][1]
    if (ax_emb is None) & (ax_trend is None):
        if plot_emb:
            fig, (ax_emb, ax_trend) = plt.subplots(
                1,
                2,
                figsize=figsize,
                gridspec_kw=dict(width_ratios=[1 * ratio, 1], wspace=wspace),
            )
        else:
            fig, ax_trend = plt.subplots(1, 1, figsize=figsize)
            # axs = ["empty", axs]

    for s in df.seg.unique():
        if color_exp is None:
            ax_trend.scatter(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "expression"],
                alpha=alpha_expr,
                s=size_expr,
                c=adata.uns["seg_colors"][
                    np.argwhere(adata.obs.seg.cat.categories == s)[0][0]
                ],
            )
            ax_trend.plot(
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
                ax_trend.plot(
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
            ax_trend.scatter(
                df.loc[df.seg == s, "t"],
                df.loc[df.seg == s, "expression"],
                c=color_exp,
                alpha=alpha_expr,
                s=size_expr,
            )
            ax_trend.plot(
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
                ax_trend.plot(
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

    ax_trend.set_ylabel(ylab)
    ax_trend.set_xlabel("pseudotime")
    x0, x1 = ax_trend.get_xlim()
    y0, y1 = ax_trend.get_ylim()
    if plot_emb:
        ax_trend.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    if plot_emb:
        if basis == "dendro":
            dendrogram(
                adata,
                color=feature,
                cmap=cmap_cells,
                ax=ax_emb,
                title=feature,
                show_info=False,
                layer="fitted",
                show=False,
                save=False,
                **kwargs,
            )
        else:
            trajectory(
                adata,
                basis=basis,
                color_seg=feature,
                cmap_seg=cmap_seg,
                color_cells=feature,
                cmap=cmap_cells,
                show_info=False,
                ax=ax_emb,
                title=feature,
                layer="fitted",
                show=False,
                save=False,
                **kwargs,
            )

    savefig_or_show("single_trend", show=show, save=save)

    if show is False:
        return (ax_emb, ax_trend) if plot_emb else ax_trend
