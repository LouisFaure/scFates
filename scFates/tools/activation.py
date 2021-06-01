import numpy as np
import pandas as pd
import igraph
from scanpy import AnnData
from typing import Optional
from tqdm import tqdm
import sys
from joblib import delayed
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as sm

from .utils import get_X, ProgressParallel, getpath
from .. import logging as logg
from .. import settings


def activation(
    adata: AnnData,
    root_milestone,
    milestones,
    fdr_cut: float = 0.05,
    stf_cut: float = 0.8,
    pseudotime_offset: float = 0,
    n_map: int = 1,
    copy: bool = False,
    n_jobs=-1,
    layer: Optional[str] = None,
):

    """\
    Classify whether a feature is being activated prior to the bifurcation or not.

    The common path between the two fates is retained for analysis, each feature is
    tested for its upregulation along the path from progenitor to the fork,
    using the linear model :math:`g_{i} \sim\ pseudotime`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    stf_cut
        fraction of projections when gene passed fdr < 0.05.
    pseudotime_offset
        consider cells to retain up to the pseudotime_fork-pseudotime_offset.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    copy
        Return a copy instead of writing to adata.
    layer
        layer to use for the test

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['module']`
            classify feature as 'early' or 'late'.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['slope']`
            slope calculated by the linear model.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['pval']`
            pval resulting from linear model.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['fdr']`
            corrected fdr value.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['prefork_signi']`
            proportion of projections where fdr<0.05.

    """

    graph = adata.uns["graph"]

    uns_temp = adata.uns.copy()

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    name = root_milestone + "->" + "<>".join(milestones)
    stats = adata.uns[name]["fork"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    img.add_edges(edges)

    common_seg = list(
        set.intersection(
            *list(
                map(lambda l: set(img.get_shortest_paths(str(root), str(l))[0]), leaves)
            )
        )
    )
    common_seg = np.array(img.vs["name"], dtype=int)[common_seg]
    fork_t = adata.uns["graph"]["pp_info"].loc[common_seg, "time"].max()

    fork_id = adata.uns["graph"]["pp_info"].loc[common_seg, "time"].idxmax()

    prefork_stat = list()
    for m in tqdm(
        range(n_map), disable=n_map == 1, file=sys.stdout, desc="    multi mapping "
    ):

        if n_map == 1:
            logg.info("    single mapping")
        df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        ddf = getpath(img, root, adata.uns["graph"]["tips"], fork_id, graph, df)

        ddf = ddf.loc[np.array(ddf.t.values) < (ddf.t.max() - pseudotime_offset)]

        df = adata.uns[name]["fork"]
        Xgenes = get_X(adata, ddf.index, df.index, layer, togenelist=True)
        data = list(zip([ddf] * len(Xgenes), Xgenes))

        allact = ProgressParallel(
            n_jobs=n_jobs,
            use_tqdm=n_map == 1,
            total=len(data),
            file=sys.stdout,
            desc="    prefork activation",
        )(delayed(test_prefork)(data[d]) for d in range(len(data)))

        allact = pd.DataFrame(
            np.array(allact), columns=["slope", "pval"], index=df.index
        )
        allact["fdr"] = multipletests(allact.pval, method="bonferroni")[1]
        allact["prefork_signi"] = (allact.fdr < fdr_cut) * 1

        prefork_stat = prefork_stat + [allact]

    df = adata.uns[name]["fork"]
    df_res = pd.DataFrame(
        {
            "slope": np.median([t.slope for t in prefork_stat], axis=0),
            "pval": np.median([t.pval for t in prefork_stat], axis=0),
            "fdr": np.median([t.fdr for t in prefork_stat], axis=0),
            "prefork_signi": np.mean([t.prefork_signi for t in prefork_stat], axis=0),
        },
        index=df.index,
    )
    df[["slope", "pval", "fdr", "prefork_signi"]] = df_res[
        ["slope", "pval", "fdr", "prefork_signi"]
    ]
    df["module"] = "late"
    df.loc[(df.slope > 0) & (df.prefork_signi > stf_cut), "module"] = "early"
    adata.uns[name]["fork"] = df

    for l in leaves:
        c_early = np.sum(
            (df.branch == str(keys[vals == l][0])) & (df.module == "early")
        )
        c_late = np.sum((df.branch == str(keys[vals == l][0])) & (df.module == "late"))
        logg.info(
            "    "
            + str(c_early)
            + " early and "
            + str(c_late)
            + " late features specific to leave "
            + str(keys[vals == l][0])
        )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    .uns['"
        + name
        + "']['fork'], DataFrame updated with additionnal 'slope','pval','fdr','prefork_signi' and 'module' columns."
    )

    return adata if copy else None


def test_prefork(dat):
    sdf = dat[0]
    sdf["exp"] = dat[1]
    result = sm.ols(formula="exp ~ t", data=sdf).fit()
    return [result.params["t"], result.pvalues["t"]]
