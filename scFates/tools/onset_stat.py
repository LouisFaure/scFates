from typing import Optional, Iterable
from .. import settings
from .. import logging as logg
from joblib import delayed
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData

import igraph
import numpy as np
import scipy.stats as st

from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import statsmodels.formula.api as sm
from . import getpath
from ..get import modules as get_modules


def onset_stat(
    adata: AnnData, root_milestone, milestones, pseudotime_offset=0, n_map=None
):
    """\
    Estimate the onset of biasing over probabilistic mappings. Simply inter-module correlation progression is
    taken in reverse starting from the bifurcation point, onset pseudotime is considered once the inter-module
    correlation switch from negative to positive as the window of cells rewind towards progenitors

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    pseudotime_offset
        pseudotime offest.
    n_map
        number of mappings to consider.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns and adds fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['onset']['mean']`
            The mean onset over all mappings.
        `.uns['root_milestone->milestoneA<>milestoneB']['onset']['CI']`
            95% confidence intervals over mappings.
        `.uns['root_milestone->milestoneA<>milestoneB']['onset']['data']`
            all detected onsets.
        `.uns['root_milestone->milestoneA<>milestoneB']['onset']['logreg_score']`
            logistic regression score.
        `.uns['root_milestone->milestoneA<>milestoneB']['onset']['logreg']`
            logistic regression loss.
    """

    name = f'{root_milestone}->{"<>".join(milestones)}'
    if "synchro" not in adata.uns[name]:
        raise ValueError("You need to run `tl.synchro_path` before estimating onset.")
    df = adata.uns[name]["synchro"]["real"][milestones[0]]

    n_map = int(df.n_map.max() + 1) if n_map is None else n_map

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
    fork_t = fork_t - pseudotime_offset
    df = df.loc[df.t < fork_t]

    onsets = []
    dfs = []
    for n in range(n_map):
        bol = df.loc[df.n_map == n, "corAB"] > 0
        if all(bol == False):
            onsets.append(0)
        else:
            idx = bol[::-1].index[np.argwhere(bol[::-1].values)[0][0]]
            onsets.append(df.loc[df.n_map == n, "t"][idx])

            bol[:idx] = True
        bol = ~bol
        dfs.append(
            pd.DataFrame(
                {
                    "t": df.loc[(df.n_map == n), "t"].values,
                    "class": bol.values * 1,
                    "n_map": n,
                }
            )
        )

    # onset mean
    onsets = np.array(onsets)
    mean = np.mean(onsets)
    ci = st.t.interval(0.95, len(onsets) - 1, loc=np.mean(onsets), scale=st.sem(onsets))

    # onset logreg
    df = pd.concat(dfs)
    df = df.sort_values("t")
    X = df.loc[:, "t"].values.reshape(-1, 1)
    y = df.loc[:, "class"].values
    clf = LogisticRegression(random_state=0, penalty="none").fit(X, y)
    loss = expit(X * clf.coef_ + clf.intercept_).ravel()
    score = clf.score(X, y)
    df["loss"] = loss

    # results
    adata.uns[name]["onset"] = {
        "mean": mean,
        "CI": ci,
        "data": onsets,
        "logreg_score": score,
        "logreg": df,
        "n_map": n_map,
        "fork_t": fork_t,
    }

    logg.info(
        f"    found onset at pseudotime: {mean:.4f}",
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "added \n"
        "    .uns['" + name + "']['onset']['mean'], the mean onset over all mappings.\n"
        "    .uns['"
        + name
        + "']['onset']['CI'], 95% confidence intervals over mappings.\n"
        "    .uns['" + name + "']['onset']['data'], all detected onsets.\n"
        "    .uns['"
        + name
        + "']['onset']['logreg_score'], logistic regression score.\n"
        "    .uns['" + name + "']['onset']['logreg'], logistic regression loss."
    )


def co_activation_test(
    adata: AnnData, root_milestone, milestones, coact_p_cutoff: float = 5e-2
):
    """\
    Test for co-activation of early gene modules.
    Mean gene expression of early genes in progenitor branch are fitted using the following linear model:
     :math:`\mu_{i} \sim\ pseudotime`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    coact_p_cutoff
        p-value cutoff for appled to the result of both tests.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns and adds fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['co_activation']['pvals']`
            tuple of the two p-values from the linear model.
        `.uns['root_milestone->milestoneA<>milestoneB']['co_activation']['co-activating']`
            boolean set to True if both tests p-values are below the defined cutoff.
        `.uns['root_milestone->milestoneA<>milestoneB']['co_activation']['coact_p_cutoff']`
            cutoff used.

    """
    logg.info("testing for co-activation of early gene modules", reset=True)
    name = f'{root_milestone}->{"<>".join(milestones)}'

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
    mil = pd.Series(adata.uns["graph"]["milestones"])
    b = mil.index[mil == adata.uns["graph"]["pp_info"].loc[fork, "time"].idxmax()][0]
    progenitor_cells = getpath(adata, root_milestone, [b]).index

    df = get_modules(adata, root_milestone, milestones).loc[progenitor_cells]
    df["t"] = adata.obs.loc[progenitor_cells, "t"]

    pvals = []
    for m in milestones:
        result = sm.ols(formula=f"early_{m} ~ t", data=df).fit()
        pvals.append(result.pvalues["t"])

    res = all(np.array(pvals) < coact_p_cutoff)
    adata.uns[name]["co_activation"] = {
        "pvals": pvals,
        "co-activating": res,
        "coact_p_cutoff": coact_p_cutoff,
    }

    res = (
        "    significant co-activation "
        if res
        else "    non-significant co-activation "
    )
    res += f"with cutoff {coact_p_cutoff}"
    res += f"\n    p-values: {milestones[0]} = {pvals[0]:.4}, {milestones[1]} = {pvals[1]:.4}"

    logg.info(
        res,
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "added \n"
        "    .uns['"
        + name
        + "']['co_activation']['pvals'], tuple of the two p-values from the linear model.\n"
        "    .uns['"
        + name
        + "']['co_activation']['co-activating'], boolean set to True if both tests p-values are below the defined cutoff.\n"
        "    .uns['" + name + "']['co_activation']['coact_p_cutoff'], cutoff used."
    )
