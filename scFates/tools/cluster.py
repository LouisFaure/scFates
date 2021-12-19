from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp

from .. import logging as logg
from .. import settings

import phenograph


def cluster(
    adata: AnnData,
    knn: int = 10,
    metric: str = "euclidean",
    device: str = "cpu",
    copy: bool = False,
    n_jobs: int = 1,
):

    """\
    Cluster feature trends.

    The models are fit using *mgcv* R package. Note that since adata can currently only keep the
    same dimensions for each of its layers, the dataset is subsetted to keep only significant
    feratures.


    Parameters
    ----------
    adata
        Annotated data matrix.
    knn
        Number of neighbors.
    metric
        distance metric to use for clustering.
    device
        run the analysis on 'cpu' with phenograph, or on 'gpu' with grapheno.
    leaves
        restrain the fit to a subset of the tree (in combination with root).
    copy
        Return a copy instead of writing to adata.
    Returns
    -------

    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:

        `.var['fit_clusters']`
            cluster assignments for features.

    """

    adata = data.copy() if copy else adata

    if device == "gpu":
        from . import grapheno_modified

        logg.info("    clustering using grapheno")
        communities, G, Q = grapheno_modified.cluster(
            adata.layers["fitted"].T, metric=metric, n_neighbors=knn
        )
        communities = communities.get()
        G = sp.csr_matrix(G.to_numpy_array())

    elif device == "cpu":
        logg.info("    clustering using phenograph")
        communities, G, Q = phenograph.cluster(
            adata.layers["fitted"].T, primary_metric=metric, k=knn, n_jobs=n_jobs
        )
        G = sp.csr_matrix(G)

    adata.varp["similarity"] = G
    adata.var["fit_clusters"] = communities

    adata.uns["fit_cluster"] = dict(
        parameters=dict(knn=knn, metric=metric, device=device, n_jobs=n_jobs),
        modularity=Q,
    )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n"
        "    .obsp['similarity'], pairwise similarity graph.\n"
        "    .var['fit_clusters'], cluster assignments for features.\n"
        "    .uns['fit_clusters'], parameters and modularity of the clustering."
    )

    return adata if copy else None
