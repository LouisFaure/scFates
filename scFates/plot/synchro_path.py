import numpy as np
import pandas as pd
from anndata import AnnData
import igraph

import matplotlib.pyplot as plt
from skmisc.loess import loess
from matplotlib import lines
from typing import Union, Optional

from scanpy.plotting._utils import savefig_or_show

from ..tools.utils import importeR

Rpy2, R, rstats, rmgcv, Formula = importeR("Plotting syncho path fit")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]


def synchro_path(
    adata: AnnData,
    root_milestone,
    milestones,
    knots: int = 10,
    alpha_multi: float = 0.1,
    knots_multi: Union[int, None] = None,
    max_t: Union[float, None] = None,
    figsize: Union[tuple, None] = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
):
    """\
    Plot results generated from tl.synchro_path.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    knots
        number of knots for gamfit.
    alpha_multi
        alpha multiple mapping.
    knots_multi
        knots for overall mappings (plot disabled if not set).
    max_t
        set max pseudotime to show.
    figsize
        figure size.
    show
        show the plot.
    save
        save the plot.

    Returns
    -------
    If `show==False` a matrix of :class:`~matplotlib.axes.Axes`

    """

    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, rmgcv, Formula])[idx])

    plt.rcParams["axes.grid"] = False

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    mlsc = np.array(adata.uns["milestones_colors"].copy())
    if isinstance(mlsc, (list)):
        mlsc = np.array(mlsc)
    # mlsc_temp = mlsc.copy()

    dct = graph["milestones"]
    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    fork = list(
        set(img.get_shortest_paths(str(root), str(leaves[0]))[0]).intersection(
            img.get_shortest_paths(str(root), str(leaves[1]))[0]
        )
    )
    fork = np.array(img.vs["name"], dtype=int)[fork]
    fork_t = adata.uns["graph"]["pp_info"].loc[fork, "time"].max()

    allcor = adata.uns[name]["synchro"]
    runs = ["real", "permuted"] if len(list(allcor.keys())) == 2 else ["real"]

    fig, axs = plt.subplots(
        3, len(runs), figsize=(len(runs) * 6, 6) if figsize is None else figsize
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.025, top=0.95)
    i = 0

    dct_cormil = dict(
        zip(
            ["corAA", "corBB", "corAB"],
            [milestones[0] + "\nintra-module", milestones[1] + "\nintra-module"]
            + [milestones[0] + " vs " + milestones[1] + "\ninter-module"],
        )
    )

    axs = axs.ravel(order="F")

    for run in runs:
        for cc in ["corAA", "corBB", "corAB"]:
            for mil in milestones:
                res = allcor[run][mil]
                nmaps = res.n_map.unique()
                col = mlsc[adata.obs.milestones.cat.categories == mil][0]

                if len(nmaps) == 1:
                    m = rmgcv.gam(
                        Formula("%s ~ s(t, bs = 'cs',k=%s)" % (cc, knots)), data=res
                    )
                    pred, se = rmgcv.predict_gam(m, se_fit=True)

                    fit = pred
                    ll = pred - se
                    ul = pred + se

                    axs[i].plot(
                        res.t.values, res[cc], "+", c=col, alpha=0.5, rasterized=True
                    )
                    axs[i].plot(res.t.values, fit, c=col, rasterized=True)
                    axs[i].fill_between(
                        res.t.values.tolist(),
                        ll.tolist(),
                        ul.tolist(),
                        alpha=0.33,
                        edgecolor=col,
                        facecolor=col,
                        rasterized=True,
                    )
                else:
                    for mp in nmaps:
                        m = rmgcv.gam(
                            Formula("%s ~ s(t, bs = 'cs',k=%s)" % (cc, knots)),
                            data=res.loc[res.n_map == mp],
                        )
                        pred = rmgcv.predict_gam(m)
                        axs[i].plot(
                            res.loc[res.n_map == mp].t,
                            res.loc[res.n_map == mp][cc],
                            c=col,
                            alpha=0.01,
                            rasterized=True,
                        )
                        axs[i].plot(
                            res.loc[res.n_map == mp].t,
                            pred,
                            c=col,
                            alpha=alpha_multi,
                            rasterized=True,
                        )

                axs[i].axvline(fork_t, color="black")
                axs[i].axhline(0, linestyle="dashed", color="grey", zorder=0)

                if i == 0:
                    vertical_line = lines.Line2D(
                        [],
                        [],
                        color="black",
                        marker="|",
                        linestyle="None",
                        markersize=10,
                        markeredgewidth=1.5,
                        label="bifurcation",
                    )

                    axs[i].legend(handles=[vertical_line])

                if (i < 2) | ((i > 2) & (i < 5)):
                    axs[i].set_xticks([])
                else:
                    axs[i].set_xlabel("pseudotime")

                if i <= 2:
                    axs[i].set_ylabel(dct_cormil[cc])

                if i > 2:
                    axs[i].set_ylim(axs[i - 3].get_ylim())
                    axs[i].set_yticks([])

                if i == 0:
                    axs[i].set_title("real")

                if i == 3:
                    axs[i].set_title("permuted")

                axs[i].grid(None)

            if max_t is not None:
                axs[i].set_xlim([0, max_t])
            i = i + 1

    if show == False:
        return axs

    savefig_or_show("synchro_path", show=show, save=save)
