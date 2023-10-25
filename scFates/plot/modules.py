import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import sparse

from typing import Union, Optional
from typing_extensions import Literal
from scanpy.plotting._utils import savefig_or_show
from ..tools.utils import getpath
from .trajectory import trajectory as plot_trajectory
from .utils import setup_axes
from ..tools.graph_operations import subset_tree
from ..tools.fit import fit
from ..tools.onset_stat import co_activation_test
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
    If `show==False` a single or a tuple of :class:`~matplotlib.axes.Axes`

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


def modules_fit(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    module: Literal["early", "late", "all"] = "all",
    title=None,
    show_coact: bool = False,
    layer: Optional[str] = None,
    fitted_linewidth=3,
    n_jobs: int = 1,
    ax_early=None,
    ax_late=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot the GAM fit of mean expression of the early and late modules.

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
    title
        automatically set the title to fitted {module} gene modules
    layer
        layer to use to compute mean of module.
    fitted_linewidth
        linewidth of GAM fit.
    n_jobs
        number of jobs to fit the modules, maximum 4.
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
    If `show==False` a single or a list of :class:`~matplotlib.axes.Axes`

    """

    df = get_modules(adata, root_milestone, milestones, layer, module)
    adata_s = adata[df.index]
    adata_s = sc.AnnData(df, obsm=adata_s.obsm, obs=adata_s.obs, uns=adata_s.uns)
    if settings.verbosity > 2:
        temp_verb = settings.verbosity
        settings.verbosity = 1
        reset = True
    fit(adata_s, adata_s.var_names, n_jobs=n_jobs)
    if reset:
        settings.verbosity = temp_verb

    if module == "all":
        modules = ["early", "late"]
        if (ax_early is None) & (ax_late is None):
            axs, _, _, _ = setup_axes(panels=[0, 1])
            axs = axs
        else:
            axs = [ax_early, ax_late]
    elif module == "early":
        modules = [module]
        if ax_early is None:
            axs, _, _, _ = setup_axes(panels=[0])
            axs = [axs[0]]
        else:
            axs = [ax_early]
    elif module == "late":
        modules = [module]
        if ax_late is None:
            axs, _, _, _ = setup_axes(panels=[0])
            axs = [axs[0]]
        else:
            axs = [ax_late]

    for module, ax in zip(modules, axs):
        sc.pl.scatter(
            adata_s,
            x=f"{module}_{milestones[0]}",
            y=f"early_{milestones[1]}",
            layers="fitted",
            color="seg",
            title=f"fitted {module} gene modules" if title is None else title,
            show=False,
            ax=ax,
            alpha=0.1,
            legend_loc="none",
        )
        df = pd.DataFrame(
            adata_s.layers["fitted"], columns=adata_s.var_names, index=adata_s.obs_names
        )
        df = pd.concat([adata_s.obs, df], axis=1).sort_values("t")
        adata_s = adata_s[adata_s.obs.sort_values("t").index]
        for s in adata.obs.seg.cat.categories:
            color_exp = adata_s.uns["seg_colors"][adata.obs.seg.cat.categories == s][0]
            ax.plot(
                df.loc[df.seg == s, f"{module}_{milestones[0]}"],
                df.loc[df.seg == s, f"{module}_{milestones[1]}"],
                c=color_exp,
                linewidth=fitted_linewidth,
            )
            tolink = adata.uns["graph"]["pp_seg"].loc[int(s), "to"]
            for next_s in adata.uns["graph"]["pp_seg"].n.iloc[
                np.argwhere(
                    adata.uns["graph"]["pp_seg"].loc[:, "from"].isin([tolink]).values
                ).flatten()
            ]:
                ax.plot(
                    [
                        df.loc[df.seg == s, f"{module}_{milestones[0]}"].iloc[-1],
                        df.loc[df.seg == next_s, f"{module}_{milestones[0]}"].iloc[0],
                    ],
                    [
                        df.loc[df.seg == s, f"{module}_{milestones[1]}"].iloc[-1],
                        df.loc[df.seg == next_s, f"{module}_{milestones[1]}"].iloc[0],
                    ],
                    c=adata_s.uns["seg_colors"][adata.obs.seg.cat.categories == next_s][
                        0
                    ],
                    linewidth=fitted_linewidth,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{module} " + milestones[0])
        ax.set_ylabel(f"{module} " + milestones[1])

    if show_coact:
        ax = axs[0] if module == "all" else ax
        name = f'{root_milestone}->{"<>".join(milestones)}'
        if "co_activation" not in adata.uns[name]:
            co_activation_test(adata, root_milestone, milestones)
        pvals = np.array(adata.uns[name]["co_activation"]["pvals"])

        def get_signi(pval):
            if pval <= 0.001:
                txt = "**** "
            elif pval <= 0.001:
                txt = "***  "
            elif pval <= 0.01:
                txt = "**   "
            elif pval <= 0.05:
                txt = "*    "
            elif pval > 0.05:
                txt = "ns   "
            return txt

        s1, s2 = [get_signi(p) for p in pvals]
        empty_patch1 = mpatches.Patch(color="none", label=s1 + milestones[0])
        empty_patch2 = mpatches.Patch(color="none", label=s2 + milestones[1])
        ax.legend(
            title="co-activation test in\nprogenitor branch",
            handles=[empty_patch1, empty_patch2],
        )

    savefig_or_show("modules_fit", show=show, save=save)

    if show == False:
        if module == "all":
            return axs
        else:
            return axs[0]
