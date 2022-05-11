from typing import Optional, Iterable
from typing_extensions import Literal
from anndata import AnnData
import numpy as np
import pandas as pd
import igraph
from ..tools.utils import get_X, getpath


def fork_stats(
    adata: AnnData,
    root_milestone: str,
    milestones: Iterable,
    module: Optional[str] = None,
    branch: Optional[str] = None,
):
    """\
    Extract statistics from the fork analysis.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    module
        subset features to a specific module.
    branch
        subset features to a specific milestone.

    Returns
    -------
    a :class:`pandas.DataFrame` extracted from adata.uns.

    """

    key = root_milestone + "->" + "<>".join(milestones)
    df = adata.uns[key]["fork"]

    if module is not None:
        df = df.loc[df.module == module]
    if branch is not None:
        df = df.loc[df.branch == branch]

    return df


def slide_cors(
    adata: AnnData,
    root_milestone: str,
    milestones: Iterable,
    branch: str,
    geneset_branch: str,
):
    """\
    Extract statistics from the sliding window correlation analysis.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    branch
        subset features to a specific milestone.
    geneset_branch
        which geneset to show correlations.

    Returns
    -------
    a :class:`pandas.DataFrame` extracted from adata.uns.

    """

    key = root_milestone + "->" + "<>".join(milestones)

    dct = dict(zip(milestones, ["genesetA", "genesetB"]))
    geneset_branch = dct[geneset_branch]
    corAB = pd.DataFrame(adata.uns[key]["corAB"])

    return corAB.loc[geneset_branch].loc[branch]


def modules(
    adata: AnnData,
    root_milestone: str,
    milestones: Iterable,
    layer: Optional[str] = None,
    module: Literal["early", "late", "all"] = "all",
):
    """\
    Extract mean expression of identified early and late modules.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    layer
        layer to use to calculate the mean.
    module
        extract either the early, late or both modules.

    Returns
    -------
    a :class:`pandas.DataFrame`.

    """

    graph = adata.uns["graph"]

    dct = graph["milestones"]

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)

    stats = adata.uns[name]["fork"]

    if "milestones_colors" not in adata.uns or len(adata.uns["milestones_colors"]) == 1:
        from . import palette_tools

        palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")

    df = adata.obs.copy(deep=True)
    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    cells = np.unique(
        np.concatenate(
            [
                getpath(
                    img, root, adata.uns["graph"]["tips"], leaves[0], graph, df
                ).index,
                getpath(
                    img, root, adata.uns["graph"]["tips"], leaves[1], graph, df
                ).index,
            ]
        )
    )
    X = pd.DataFrame(
        get_X(adata, cells, stats.index, layer), index=cells, columns=stats.index
    )

    early_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "early")
    late_1 = (stats.branch.values == milestones[0]) & (stats.module.values == "late")

    early_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "early")
    late_2 = (stats.branch.values == milestones[1]) & (stats.module.values == "late")

    X_early = pd.DataFrame(
        {
            "early_" + milestones[0]: X.loc[:, early_1].mean(axis=1),
            "early_" + milestones[1]: X.loc[:, early_2].mean(axis=1),
        },
        index=X.index,
    )

    X_late = pd.DataFrame(
        {
            "late_" + milestones[0]: X.loc[:, late_1].mean(axis=1),
            "late_" + milestones[1]: X.loc[:, late_2].mean(axis=1),
        },
        index=X.index,
    )

    if module == "all":
        return pd.concat([X_early, X_late], axis=1)
    elif module == "early":
        return X_early
    else:
        return X_late
