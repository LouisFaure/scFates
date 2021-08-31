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
    color_milestones=False,
    color_seg="k",
    linewidth_seg: float = 3,
    alpha_seg: float = 0.5,
    tree_behind: bool = False,
    show_info: bool = True,
    show=None,
    save=None,
    **kwargs
):

    if root_milestone is not None:
        verb = settings.verbosity
        settings.verbosity = 1
        adata = subset_tree(adata, root_milestone, milestones, copy=True)
        settings.verbosity = verb

    if color_milestones:
        ax = sc.pl.embedding(adata, basis="dendro", show=False, **kwargs)
        ax.scatter(
            adata.obsm["X_dendro"][:, 0],
            adata.obsm["X_dendro"][:, 1],
            c=gen_milestones_gradients(adata)[adata.obs_names].values,
            s=120000 / adata.shape[0] if "s" not in kwargs else kwargs["s"],
            marker=".",
        )
    else:
        ax = sc.pl.embedding(adata, basis="dendro", show=False, **kwargs)

    for key, value in adata.uns["dendro_segments"].items():
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
            ax.set_box_aspect(aspect=1)
            fig = ax.get_gridspec().figure
            cbar = np.argwhere(
                ["colorbar" in a.get_label() for a in fig.get_axes()]
            ).ravel()
            if len(cbar) > 0:
                fig.get_axes()[cbar[0]].remove()

    savefig_or_show("dendrogram", show=show, save=save)

    if show == False:
        return ax
