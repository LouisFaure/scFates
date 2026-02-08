import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd
from functools import partial, reduce
from anndata import AnnData
import shutil
import sys
import copy
import igraph
import warnings

from joblib import delayed
from tqdm import tqdm

from .. import logging as logg
from .. import settings
from .utils import getpath, get_X, importeR, ProgressParallel


Rpy2, R, rstats, rmgcv, Formula = importeR("fitting associated features")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]


def fit(
    adata: AnnData,
    features: Optional[Iterable] = None,
    layer: Optional[str] = None,
    n_map: int = 1,
    n_jobs: int = 1,
    gamma: float = 1.5,
    knots: int = -1,
    save_raw: bool = True,
    copy: bool = False,
):

    """\
    Model feature expression levels as a function of tree positions.

    The models are fit using *mgcv* R package. Note that since adata can currently only keep the
    same dimensions for each of its layers. While the dataset is subsetted to keep only significant
    feratures, the unsubsetted dataset is kept in adata.raw (save_raw parameter).


    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        adata layer to use for the fitting.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    gamma
        stringency of penalty.
    knots
        number of knots for the GAM fit.
    save_raw
        save the unsubsetted anndata to adata.raw
    copy
        Return a copy instead of writing to adata.
    Returns
    -------

    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.layers['fitted']`
            fitted features on the trajectory for all mappings.

    """

    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, rmgcv, Formula])[idx])

    adata = adata.copy() if copy else adata

    if features is None:
        if "signi" not in adata.var.columns:
            raise ValueError(
                "You need to run `tl.test_association` before fitting features."
            )

        features = adata.var_names[adata.var.signi]

    if len(features) == 0:
        logg.info("    no features to fit")
        return adata if copy else None

    graph = adata.uns["graph"]
    tips = graph["tips"]
    root = graph["root"]
    tips = tips[~np.isin(tips, root)]
    root2 = None
    if "root2" in graph:
        root2 = graph["root2"]
        tips = tips[~np.isin(tips, graph["root2"])]

    logg.info("fit features associated with the trajectory", reset=True, end="\n")

    stat_assoc = list()

    for m in tqdm(
        range(n_map), disable=n_map == 1, file=sys.stdout, desc="    multi mapping "
    ):
        if ("t_old" in adata.obs.columns) | (n_map == 1):
            df = adata.obs.loc[:, ["t", "seg"]]
        else:
            df = adata.uns["pseudotime_list"][str(m)]
        edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
        img = igraph.Graph()
        img.add_vertices(
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
        )
        img.add_edges(edges)

        subtree = pd.concat(
            list(
                filter(
                    lambda x: x is not None, 
                    map(lambda tip: getpath(img, root, tips, tip, graph, df), tips)
                    )),
            axis=0,
        )
        if root2 is not None:
            subtree = pd.concat(
                [
                    subtree,
                    pd.concat(
                        list(
                            map(
                                lambda tip: getpath(img, root2, tips, tip, graph, df),
                                tips,
                            )
                        ),
                        axis=0,
                    ),
                ]
            )
        subtree = subtree[["t", "branch"]]
        subtree["gamma"] = gamma
        subtree["knots"] = knots

        Xgenes = get_X(adata, subtree.index, features, layer, togenelist=True)

        data = list(zip([subtree] * len(Xgenes), Xgenes))

        stat = ProgressParallel(
            n_jobs=n_jobs,
            total=len(data),
            use_tqdm=(n_map == 1) & (settings.verbosity > 1),
            file=sys.stdout,
            desc="    single mapping ",
        )(delayed(gt_fun)(data[d]) for d in range(len(data)))

        stat_assoc = stat_assoc + [stat]

    for i in range(len(stat_assoc)):
        stat_assoc[i] = pd.concat(stat_assoc[i], axis=1)
        stat_assoc[i].columns = features

    names = np.arange(len(stat_assoc)).astype(str).tolist()
    dictionary = dict(zip(names, stat_assoc))

    if n_map == 1:
        fitted = dictionary["0"]
    else:
        dfs = list(dictionary.values())
        fitted = reduce(lambda x, y: x.add(y, fill_value=0), dfs) / n_map

    if save_raw:
        adata.raw = adata

    adata._inplace_subset_var(features)

    adata.layers["fitted"] = fitted.loc[adata.obs_names, :]

    logg.info(
        "    finished (adata subsetted to keep only fitted features!)",
        time=True,
        end=" " if settings.verbosity > 2 else "\n",
    )

    if save_raw:
        logg.hint(
            "added\n"
            "    .layers['fitted'], fitted features on the trajectory for all mappings.\n"
            "    .raw, unfiltered data."
        )
    else:
        logg.hint(
            "added\n"
            "    .layers['fitted'], fitted features on the trajectory for all mappings."
        )

    return adata if copy else None


def gt_fun(data):
    sdf = data[0]
    sdf["exp"] = data[1]
    gamma = sdf["gamma"].iloc[0]
    knots = sdf["knots"].iloc[0]

    global rmgcv
    global rstats

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri as p2ri
    from rpy2.robjects.conversion import localconverter
    context = localconverter(ro.default_converter + p2ri.converter)


    def gamfit(b):
        with context as cv:
            dat = cv.py2rpy(sdf.loc[sdf["branch"] == b, :])

        m = rmgcv.gam(
            Formula(f"exp~s(t,bs='ts',k={knots})"),
            data=dat,
            gamma= ro.FloatVector([gamma]),
        )
        return pd.Series(
            rmgcv.predict_gam(m), index=sdf.loc[sdf["branch"] == b, :].index
        )

    mdl = list(map(gamfit, sdf.branch.unique()))

    return pd.concat(mdl, axis=1, sort=False).apply(np.nanmean, axis=1)
