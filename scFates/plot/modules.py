import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import matplotlib.pyplot as plt
from scipy import sparse

from typing import Union, Optional
from typing_extensions import Literal
from scanpy.plotting._utils import savefig_or_show
from ..tools.utils import getpath
from .trajectory import trajectory as plot_trajectory
from .utils import setup_axes
from ..tools.graph_operations import subset_tree
from .milestones import milestones as milestones_plot
from ..get import modules as get_modules
from .. import settings

import scanpy as sc


def modules(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    module: Literal["early", "late", "all"] = "all",
    show_traj: bool = False,
    layer: Optional[str] = None,
    smooth: bool = False,
    ax_early=None,
    ax_late=None,
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
    module
        whether to show early, late or both modules.
    show_traj
        show trajectory on the early module plot.
    layer
        layer to use to compute mean of module.
    smooth
        whether to smooth the data using knn graph.
    ax_early
        existing axes for early module.
    ax_late
        existing axes for late module.
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

    X_modules = get_modules(adata, root_milestone, milestones, layer)

    cells = X_modules.index
    X_early, X_late = X_modules.iloc[:, :2], X_modules.iloc[:, 2:]
    verb = settings.verbosity
    settings.verbosity = 1
    nmil = len(adata.uns["graph"]["milestones"])
    if nmil > 4:
        adata_c = subset_tree(adata, root_milestone, milestones, copy=True)
        adata_c.obsm["X_early"] = X_early.loc[adata_c.obs_names].values
        adata_c.obsm["X_late"] = X_late.loc[adata_c.obs_names].values
        adata_c.uns["seg_colors"] = [
            np.array(adata_c.uns["milestones_colors"])[
                pd.Series(adata_c.uns["graph"]["milestones"]) == t
            ][0]
            for t in adata_c.uns["graph"]["pp_seg"].to
        ]
    else:
        adata_c = AnnData(
            X_early.values,
            obs=adata[cells].obs,
            uns=adata.uns,
            obsm={
                "X_early": X_early.values,
                "X_late": X_late.values,
                "X_R": adata[cells].obsm["X_R"],
            },
            obsp=adata.obsp,
        )
    settings.verbosity = verb

    if smooth:
        adata_c.obsm["X_early"] = adata_c.obsp["connectivities"].dot(
            adata_c.obsm["X_early"]
        )
        adata_c.obsm["X_late"] = adata_c.obsp["connectivities"].dot(
            adata_c.obsm["X_late"]
        )

    if module == "all":
        if (ax_early is None) & (ax_late is None):
            axs, _, _, _ = setup_axes(panels=[0, 1])
            ax_early, ax_late = axs
    elif module == "early":
        if ax_early is None:
            axs, _, _, _ = setup_axes(panels=[0])
            ax_early = axs[0]
    elif module == "late":
        if ax_late is None:
            axs, _, _, _ = setup_axes(panels=[0])
            ax_late = axs[0]

    if (color == "milestones") & ("old_milestones" in adata_c.obs):
        color = "old_milestones"

    if (color == "seg") & ("old_seg" in adata_c.obs):
        color = "old_seg"

    if (module == "early") | (module == "all"):
        if (color == "old_milestones") | (color == "milestones"):
            milestones_plot(
                adata_c,
                basis="early",
                subset=cells,
                title="",
                show=False,
                ax=ax_early,
            )
        else:
            sc.pl.embedding(
                adata_c[cells],
                basis="early",
                color=color,
                legend_loc="none",
                title="",
                show=False,
                ax=ax_early,
                **kwargs,
            )
        if show_traj:
            if color != "milestones":
                kwargs["legend_loc"] = "none"
            if "alpha" in kwargs:
                kwargs.pop("alpha")
            plot_trajectory(
                adata_c,
                basis="early",
                root_milestone=root_milestone,
                milestones=milestones,
                alpha=0,
                show=False,
                title="",
                ax=ax_early,
                **kwargs,
            )
        ax_early.set_xlabel("early " + milestones[0])
        ax_early.set_ylabel("early " + milestones[1])

    if (module == "late") | (module == "all"):
        if (color == "old_milestones") | (color == "milestones"):
            milestones_plot(
                adata_c,
                basis="late",
                subset=cells,
                title="",
                show=False,
                ax=ax_late,
            )
        else:
            sc.pl.embedding(
                adata_c[cells],
                basis="late",
                color=color,
                legend_loc="none",
                show=False,
                title="",
                ax=ax_late,
                **kwargs,
            )
        ax_late.set_xlabel("late " + milestones[0])
        ax_late.set_ylabel("late " + milestones[1])

    savefig_or_show("modules", show=show, save=save)

    if show == False:
        if module == "all":
            return ax_early, ax_late
        elif module == "early":
            return ax_early
        elif module == "late":
            return ax_late
