import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import matplotlib.pyplot as plt
from scipy import sparse

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from ..tools.utils import getpath


def modules(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    mode: str = "2d",
    marker_size: int = 20,
    highlight: bool = False,
    incl_3d: int = 30,
    rot_3d: int = 315,
    alpha: float = 1,
    cmap_pseudotime="viridis",
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    layer: Optional[str] = None,
):

    plt.rcParams["axes.grid"] = False
    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    dct = graph["milestones"]

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    stats = adata.uns[name]["fork"]

    if "milestones_colors" not in adata.uns or len(adata.uns["milestones_colors"]) == 1:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")

    mlsc = adata.uns["milestones_colors"].copy()
    mls = adata.obs.milestones.cat.categories.tolist()
    dct = dict(zip(mls, mlsc))
    df = adata.obs.copy(deep=True)
    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    cells = np.unique(
        np.concatenate(
            [
                getpath(
                    img, root, adata.uns["graph"]["tips"], leaves[0], graph, df
                ).index,
                getpath(
                    img, root, adata.uns["graph"]["tips"], leaves[1], graph, df
                ).index,
            ]
        )
    )

    cols = adata.uns[color + "_colors"].copy()
    obscol = adata.obs[color].cat.categories.tolist()
    dct_c = dict(zip(obscol, cols))

    if layer is None:
        if sparse.issparse(adata.X):
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].X.A),
                index=cells,
                columns=stats.index,
            )
        else:
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].X), index=cells, columns=stats.index
            )
    else:
        if sparse.issparse(adata.layers[layer]):
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].layers[layer].A),
                index=cells,
                columns=stats.index,
            )
        else:
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].layers[layer]),
                index=cells,
                columns=stats.index,
            )

    miles = adata.obs.loc[X.index, color].astype(str)

    early_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "early")
    late_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "late")

    early_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "early")
    late_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "late")

    if mode == "2d":
        fig, axs = plt.subplots(2, 2)

        if highlight:
            axs[0, 0].scatter(
                X.loc[:, early_1].mean(axis=1),
                X.loc[:, early_2].mean(axis=1),
                s=marker_size * 2,
                c="k",
            )
            axs[1, 0].scatter(
                X.loc[:, late_1].mean(axis=1),
                X.loc[:, late_2].mean(axis=1),
                s=marker_size * 2,
                c="k",
            )
            axs[0, 1].scatter(
                X.loc[:, early_1].mean(axis=1),
                X.loc[:, early_2].mean(axis=1),
                s=marker_size * 2,
                c="k",
            )
            axs[1, 1].scatter(
                X.loc[:, late_1].mean(axis=1),
                X.loc[:, late_2].mean(axis=1),
                s=marker_size * 2,
                c="k",
            )

        for m in obscol:
            axs[0, 0].scatter(
                X.loc[miles.index[miles == m], early_1].mean(axis=1),
                X.loc[miles.index[miles == m], early_2].mean(axis=1),
                s=marker_size,
                c=dct_c[m],
                alpha=alpha,
            )
        axs[0, 0].set_aspect(1.0 / axs[0, 0].get_data_ratio(), adjustable="box")
        axs[0, 0].set_xlabel("early " + milestones[0])
        axs[0, 0].set_ylabel("early " + milestones[1])

        for m in obscol:
            axs[1, 0].scatter(
                X.loc[miles.index[miles == m], late_1].mean(axis=1),
                X.loc[miles.index[miles == m], late_2].mean(axis=1),
                s=marker_size,
                c=dct_c[m],
                alpha=alpha,
            )
        axs[1, 0].set_aspect(1.0 / axs[1, 0].get_data_ratio(), adjustable="box")
        axs[1, 0].set_xlabel("late " + milestones[0])
        axs[1, 0].set_ylabel("late " + milestones[1])

        axs[0, 1].scatter(
            X.loc[:, early_1].mean(axis=1),
            X.loc[:, early_2].mean(axis=1),
            c=adata.obs.t[X.index],
            s=marker_size,
            alpha=alpha,
            cmap=cmap_pseudotime,
        )
        axs[0, 1].set_aspect(1.0 / axs[0, 1].get_data_ratio(), adjustable="box")
        axs[0, 1].set_xlabel("early " + milestones[0])
        axs[0, 1].set_ylabel("early " + milestones[1])

        axs[1, 1].scatter(
            X.loc[:, late_1].mean(axis=1),
            X.loc[:, late_2].mean(axis=1),
            c=adata.obs.t[X.index],
            s=marker_size,
            alpha=alpha,
            cmap=cmap_pseudotime,
        )
        axs[1, 1].set_aspect(1.0 / axs[1, 1].get_data_ratio(), adjustable="box")
        axs[1, 1].set_xlabel("late " + milestones[0])
        axs[1, 1].set_ylabel("late " + milestones[1])
        plt.tight_layout()

        fig.set_figheight(10)
        fig.set_figwidth(10)

    if mode == "3d":
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for m in obscol:
            ax.scatter(
                xs=X.loc[miles.index[miles == m], early_1].mean(axis=1),
                ys=X.loc[miles.index[miles == m], early_2].mean(axis=1),
                zs=adata.obs.t[miles.index[miles == m]],
                c=dct_c[m],
                alpha=alpha,
                s=10,
            )

        ax.invert_xaxis()
        ax.view_init(incl_3d, rot_3d)
        plt.xlabel("early " + milestones[0])
        plt.ylabel("early " + milestones[1])
        ax.set_zlabel("pseudotime")

        ax = fig.add_subplot(1, 2, 2, projection="3d")

        for m in obscol:
            ax.scatter(
                xs=X.loc[miles.index[miles == m], late_1].mean(axis=1),
                ys=X.loc[miles.index[miles == m], late_2].mean(axis=1),
                zs=adata.obs.t[miles.index[miles == m]],
                c=dct_c[m],
                alpha=alpha,
                s=10,
            )

        ax.invert_xaxis()
        ax.view_init(incl_3d, rot_3d)
        plt.xlabel("late " + milestones[0])
        plt.ylabel("late " + milestones[1])
        ax.set_zlabel("pseudotime")

    adata.uns = uns_temp

    savefig_or_show("modules", show=show, save=save)
