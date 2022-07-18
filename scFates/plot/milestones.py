from typing_extensions import Literal
import igraph
import numpy as np
import pandas as pd
from typing import Union, Optional, Iterable
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from ..tools.dendrogram import hierarchy_pos
from .. import logging as logg
from .utils import gen_milestones_gradients, get_basis, setup_axes
from scanpy.plotting._utils import savefig_or_show
import scanpy as sc


def milestones(
    adata,
    basis: Union[None, str] = None,
    annotate: bool = False,
    title: str = "milestones",
    subset: Optional[Iterable] = None,
    ax=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):
    """\
    Display the milestone graph in PAGA style.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Reduction to use for plotting.
    annotate
        Display milestone labels on the plot.
    title
        Plot title to display.
    subset
        Subset cells.
    ax
        Add plot to existing ax.
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    If `show==False` an object of :class:`~matplotlib.axes.Axes`

    """

    basis = get_basis(adata, basis)
    emb = adata.obsm[f"X_{basis}"]

    if "sort_order" not in kwargs:
        order = adata.obs.t.sort_values().index
    else:
        if kwargs["sort_order"]:
            order = adata.obs.t.sort_values().index
        else:
            order = adata.obs_names
    order = order[order.isin(subset)] if subset is not None else order
    if "color" in kwargs:
        kwargs.pop("color")
    # if ax is None:
    #    ax = sc.pl.embedding(adata[order], basis=basis, alpha=0, show=False, **kwargs)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if "edgecolor" not in kwargs:
        kwargs["edgecolor"] = "none"
    if "s" not in kwargs:
        kwargs["s"] = 120000 / adata.shape[0]
    ax.scatter(
        adata[order].obsm[f"X_{basis}"][:, 0],
        adata[order].obsm[f"X_{basis}"][:, 1],
        c=gen_milestones_gradients(adata)[order].values,
        marker=".",
        rasterized=True,
        plotnonfinite=True,
        **kwargs,
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.autoscale_view()
    if annotate:
        R = adata.obsm["X_R"]
        proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

        X = proj[list(adata.uns["graph"]["milestones"].values()), :]
        adata_m = sc.AnnData(
            X,
            dtype=X.dtype,
            obs=dict(mil=list(adata.uns["graph"]["milestones"].keys())),
            obsm={basis: X},
        )
        adata_m.obs["mil"] = adata_m.obs["mil"].astype("category")

        sc.pl.embedding(
            adata_m,
            basis,
            color="mil",
            title=title,
            legend_loc="on data",
            ax=ax,
            alpha=0,
            legend_fontoutline=True,
            show=False,
        )

    savefig_or_show("milestones", show=show, save=save)

    if show == False:
        return ax
