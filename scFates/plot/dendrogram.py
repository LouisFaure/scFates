from typing import Union, Optional
import scanpy as sc
import numpy as np
from scanpy.plotting._utils import savefig_or_show
from .. import settings
from ..tools.graph_operations import subset_tree
from .utils import is_categorical, gen_milestones_gradients


def dendrogram(
    adata,
    root_milestone=None,
    milestones=None,
    color_milestones: bool = False,
    color_seg: str = "k",
    linewidth_seg: float = 3,
    alpha_seg: float = 0.3,
    tree_behind: bool = False,
    show_info: bool = True,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs
):
    """\
    Plot the single-cell dendrogram embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    color_milestones
        color the cells with gradients combining pseudotime and milestones.
    color_seg
        color the segments, either a color, or 'seg' colors from obs.seg.
    linewidth_seg
        linewidth of the segments.
    alpha_seg
        alpha of the segments.
    tree_behind
        whether to plot the segment in front or behind the cells.
    show_info
        display the colorbar or not.
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to :func:`scanpy.pl.embedding`.

    Returns
    -------
    If `show==False` an object of :class:`~matplotlib.axes.Axes`

    """

    if root_milestone is not None:
        verb = settings.verbosity
        settings.verbosity = 1
        adata = subset_tree(adata, root_milestone, milestones, copy=True)
        settings.verbosity = verb

    if color_milestones:
        if "sort_order" not in kwargs:
            order = adata.obs.t.sort_values().index
        else:
            if kwargs["sort_order"]:
                order = adata.obs.t.sort_values().index
            else:
                order = adata.obs_names

        ax = sc.pl.embedding(
            adata[order], basis="dendro", alpha=0, show=False, **kwargs
        )
        ax.scatter(
            adata[order].obsm["X_dendro"][:, 0],
            adata[order].obsm["X_dendro"][:, 1],
            c=gen_milestones_gradients(adata)[order].values,
            s=120000 / adata.shape[0] if "s" not in kwargs else kwargs["s"],
            marker=".",
            rasterized=True,
        )
    else:
        ax = sc.pl.embedding(adata, basis="dendro", show=False, **kwargs)

    for key, value in adata.uns["dendro_segments"].items():
        if "seg_colors" not in adata.uns:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, "seg")
        for s in value:
            ax.plot(
                s[0],
                s[1],
                linewidth=linewidth_seg,
                alpha=alpha_seg,
                c=adata.uns["seg_colors"][int(key) - 1]
                if color_seg == "seg"
                else color_seg,
                zorder=0 if tree_behind else None,
            )

    if show_info == False and "color" in kwargs:
        if is_categorical(adata, kwargs["color"]):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            # ax.set_box_aspect(aspect=1)
            fig = ax.get_gridspec().figure
            cbar = np.argwhere(
                ["colorbar" in a.get_label() for a in fig.get_axes()]
            ).ravel()
            if len(cbar) > 0:
                fig.get_axes()[cbar[0]].remove()

    savefig_or_show("dendrogram", show=show, save=save)

    if show == False:
        return ax
