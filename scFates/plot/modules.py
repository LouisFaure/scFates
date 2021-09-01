import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import matplotlib.pyplot as plt
from scipy import sparse

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from ..tools.utils import getpath
from .trajectory import trajectory as plot_trajectory
from .utils import setup_axes
from ..tools.graph_operations import subset_tree
from .. import settings

import scanpy as sc


def modules(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    show_traj: bool = False,
    layer: Optional[str] = None,
    smooth: bool = False,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):
    """\
    Plot the mean expression of the early and late modules.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    color
        color the cells with variable from adata.obs.
    show_traj
        show trajectory on the early module plot.
    layer
        layer to use to compute mean of module.
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to :func:`scFates.pl.trajectory` if `show_traj=True`, else to :func:`scanpy.pl.embedding`

    Returns
    -------
    If `show==False` a tuple of :class:`~matplotlib.axes.Axes`

    """

    plt.rcParams["axes.grid"] = False

    X_early, X_late = get_modules(adata, root_milestone, milestones, layer)

    cells = X_early.index
    verb = settings.verbosity
    settings.verbosity = 1
    adata_c = subset_tree(adata, root_milestone, milestones, copy=True)
    settings.verbosity = verb
    adata_c.obsm["X_early"] = X_early.loc[adata_c.obs_names].values
    adata_c.obsm["X_late"] = X_late.loc[adata_c.obs_names].values

    if smooth:
        adata_c.obsm["X_early"] = adata_c.obsp["connectivities"].dot(
            adata_c.obsm["X_early"]
        )
        adata_c.obsm["X_late"] = adata_c.obsp["connectivities"].dot(
            adata_c.obsm["X_late"]
        )

    axs, _, _, _ = setup_axes(panels=[0, 1])

    if show_traj:
        plot_trajectory(
            adata_c,
            basis="early",
            root_milestone=root_milestone,
            milestones=milestones,
            color_cells=color,
            show=False,
            title="",
            legend_loc="none",
            ax=axs[0],
            **kwargs,
        )
    else:
        sc.pl.embedding(
            adata_c[cells],
            basis="early",
            color=color,
            legend_loc="none",
            title="",
            show=False,
            ax=axs[0],
            **kwargs,
        )

    sc.pl.embedding(
        adata_c[cells],
        basis="late",
        color=color,
        legend_loc="none",
        show=False,
        title="",
        ax=axs[1],
        **kwargs,
    )

    axs[0].set_xlabel("early " + milestones[0])
    axs[0].set_ylabel("early " + milestones[1])
    axs[1].set_xlabel("late " + milestones[0])
    axs[1].set_ylabel("late " + milestones[1])

    savefig_or_show("modules", show=show, save=save)


def get_modules(adata, root_milestone, milestones, layer):
    graph = adata.uns["graph"]

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

    if layer is None:
        if sparse.issparse(adata.X):
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].X.A),
                index=cells,
                columns=stats.index,
            )
        else:
            X = pd.DataFrame(
                np.array(adata[cells, stats.index].X),
                index=cells,
                columns=stats.index,
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
                np.array(adata[:, stats.index].layers[layer]),
                index=cells,
                columns=stats.index,
            )

    early_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "early")
    late_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "late")

    early_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "early")
    late_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "late")

    X_early = pd.DataFrame(
        {
            "early_" + milestones[0]: X.loc[:, early_1].mean(axis=1),
            "early_" + milestones[1]: X.loc[:, early_2].mean(axis=1),
        },
        index=X.index,
    )

    X_late = pd.DataFrame(
        {
            "late_" + milestones[0]: X.loc[:, late_1].mean(axis=1),
            "late_" + milestones[1]: X.loc[:, late_2].mean(axis=1),
        },
        index=X.index,
    )

    return X_early, X_late
