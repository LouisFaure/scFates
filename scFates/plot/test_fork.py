from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from typing import Union, Optional
import numpy as np
from scanpy.plotting._utils import savefig_or_show


def test_fork(
    adata: AnnData,
    root_milestone,
    milestones,
    col: Union[None, list] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot results generated from tl.test_fork.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    col
        color of the two sets of genes, by default to the color of the end milestones.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    if "milestones_colors" not in adata.uns or len(adata.uns["milestones_colors"]) == 1:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")

    mlsc = np.array(adata.uns["milestones_colors"].copy())
    if mlsc.dtype == "float":
        mlsc = list(map(rgb2hex, mlsc))

    name = root_milestone + "->" + "<>".join(milestones)
    df = adata.uns[name]["fork"]
    df = df.loc[df.fdr < 0.05]

    c_mil = (
        np.array(mlsc)[
            np.argwhere(adata.obs.milestones.cat.categories.isin(milestones))
        ].flatten()
        if col is None
        else col
    )

    A = df.iloc[:, [0, 1]].values

    fig, ax = plt.subplots()

    ax.scatter(A[A[:, 0] != 0, 0], df.loc[A[:, 0] != 0].fdr.values, color=c_mil[0])
    ax.scatter(
        np.abs(A[A[:, 1] != 0, 1]), df.loc[A[:, 1] != 0].fdr.values, color=c_mil[1]
    )

    ax.set_xlabel(" < A > ".join(milestones))
    ax.set_ylabel("FDR")

    left, right = ax.get_xlim()
    bounds = np.abs(A.ravel()).max()
    bounds = bounds + bounds * 0.1
    ax.set_xlim([-bounds, bounds])
    xticks = ax.get_xticks()
    xticks = [str(int(xt)) if xt.is_integer() else str(xt) for xt in np.abs(xticks)]
    ax.set_xticklabels(xticks)

    if show == False:
        return ax

    savefig_or_show("test_fork", show=show, save=save)
