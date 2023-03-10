from typing import Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scanpy.plotting._utils import savefig_or_show
from scanpy import AnnData


def binned_pseudotime_meta(
    adata,
    key,
    nbins: int = 20,
    rotation: int = 0,
    show_colorbar: bool = False,
    rev=False,
    cmap="viridis",
    ax=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):

    """\
    Plot a dot plot of proportion of cells from a given category over binned sections of pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    key
        category to study.
    nbins
        Number of pseudotime bin to generate.
    rotation
        rotaton of the category labels.
    show_colorbar
        display pseudotime colorbar.
    cmap
        colormap of the pseudotime cbar.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a tuple of :class:`~matplotlib.axes.Axes`

    """

    ncats = len(adata.obs[key].cat.categories)
    intervals = pd.cut(adata.obs.t, bins=nbins)
    counts = pd.concat(
        [
            adata.obs[key][intervals[intervals == c].index].value_counts()
            for c in intervals.cat.categories
        ],
        axis=1,
    )
    prop = counts / counts.sum(axis=0)

    xs = np.arange(prop.shape[0])
    ys = np.arange(prop.shape[1])
    ys = [c.mid for c in intervals.cat.categories]
    ys = -np.array(ys) if rev else ys
    X, Y = np.meshgrid(xs, ys)

    specs = (
        {"width_ratios": [2.5 * ncats / 6, 0.25 * show_colorbar]}
        if show_colorbar
        else None
    )
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1 + 1 * show_colorbar,
            figsize=(2.5 * ncats / 6 + 0.25 * show_colorbar, (nbins / 4) + 1),
            gridspec_kw=specs,
        )
        ax2 = ax[1] if show_colorbar else None
        ax = ax[0] if show_colorbar else ax

        if show_colorbar:
            cmap = eval("mpl.cm." + cmap)
            timebins = np.array([i.mid for i in intervals.cat.categories])
            norm = mpl.colors.Normalize(vmin=timebins.min(), vmax=timebins.max())
            norm = mpl.colors.BoundaryNorm(timebins, cmap.N)
            cbar = mpl.colorbar.ColorbarBase(
                ax2,
                cmap=cmap,
                norm=norm,
                spacing="proportional",
                orientation="vertical",
            )
            cbar.set_ticks([])

    if key + "_colors" not in adata.uns:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, key)

    for i, d in enumerate(adata.obs[key].cat.categories):
        ax.scatter(
            X[:, i], Y[:, i], s=prop.values[i, :] * 200, c=adata.uns[key + "_colors"][i]
        )

    ax.grid(None)
    ax.set_xticks(range(len(adata.obs[key].cat.categories)))
    ax.set_xticklabels(adata.obs[key].cat.categories, rotation=rotation)
    ax.set_yticks([])
    ax.set_xlim([-0.5, ncats - 0.5])

    plt.tight_layout(pad=0.5)

    savefig_or_show("binned_pseudotime_meta", show=show, save=save)

    if show == False:
        return ax
