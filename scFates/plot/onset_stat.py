from anndata import AnnData
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scanpy.plotting._utils import savefig_or_show


def onset_stat(
    adata: AnnData,
    root_milestone,
    milestones,
    color_synchro="grey",
    alpha_synchro: float = 0.01,
    figsize: tuple = (5, 2),
    whiskers: Union[tuple, None] = (5, 95),
    ax: Optional = None,
    plot_logref: bool = True,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot a set of features as per-segment matrix plots of binned pseudotimes.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    color_synchro
        coloration of all inter-module correlation curves plotted.
    alpha_synchro
        alpha of all inter-module correlation curves plotted.
    figsize
        size of the figure
    whiskers
        set a confidence interval for the whiskers of the box plot
    ax
        existing :class:`~matplotlib.axes.Axes` to update
    show
        show the plot, otherwise return :class:`~matplotlib.axes.Axes`
    save
        save figure

    Returns
    -------
    If `show==False` an object of :class:`~matplotlib.axes.Axes`

    """

    name = f'{root_milestone}->{"<>".join(milestones)}'
    df = adata.uns[name]["synchro"]["real"][milestones[0]]
    onsets = adata.uns[name]["onset"]["data"]
    n_map = adata.uns[name]["onset"]["n_map"]
    fork_t = adata.uns[name]["onset"]["fork_t"]

    nrows = 2 if plot_logref else 1
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2 * nrows), nrows=nrows, sharex=plot_logref)

    if plot_logref:
        ax, ax2 = ax
    for i in range(n_map):
        ax.plot(
            df.loc[df.n_map == i, "t"],
            df.loc[df.n_map == i, "corAB"],
            c=color_synchro,
            alpha=alpha_synchro,
        )
    ax.axvline(fork_t, c="k")
    ax.axhline(0, c="grey", linestyle="--")
    y1, y2 = ax.get_ylim()
    y = y2 - y1
    plt.grid(False)
    ax.boxplot(onsets, vert=False, positions=[0], widths=[y / 4], whis=whiskers)
    ax.set_xlim([0, fork_t])
    ax.set_ylim([y1, y2])
    ax.set_ylabel("inter-module\ncorrelation")

    if plot_logref:
        df = adata.uns[name]["onset"]["logreg"]
        ax2.plot(df.t, df.loss, color="k", alpha=1, linewidth=2)
        ax.set_xticks([0, fork_t], ["", ""])
        ax2.set_xticks([0, fork_t], ["progenitor", "bifurcation"])
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("Logistic\nregression")
        xTick_objects = ax2.xaxis.get_major_ticks()
        xTick_objects[0].label1.set_horizontalalignment("left")
        ax2.fill_between(df.t, df.loss, color="pink")
        # Create a patch for the legend
        pink_patch = mpatches.Patch(color="pink", label="biasing")

        # Add the patch to the legend
        plt.legend(handles=[pink_patch])
    else:
        ax.set_xticks([0, fork_t], ["progenitor", "bifurcation"])
        xTick_objects = ax.xaxis.get_major_ticks()
        xTick_objects[0].label1.set_horizontalalignment("left")

    savefig_or_show("onset_stat", show=show, save=save)

    if show == False:
        if plot_logref:
            return (ax, ax2)
        else:
            return ax
