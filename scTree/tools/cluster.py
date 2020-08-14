import numpy as np
import pandas as pd
from anndata import AnnData

from .. import logging as logg
from .. import settings

import phenograph
from sklearn.preprocessing import StandardScaler


def cluster(
    adata: AnnData,
    knn: int = 10,
    metric="euclidean",
    copy: bool = False):

    adata = data.copy() if copy else adata
    
    if "fitted" not in adata.layers:
        raise ValueError(
            "You need to run `tl.fit` first to fit the features before clustering them."
        )
    
    clusters = cluster_trends(pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names).T,k=knn,metric=metric)
    
    adata.uns["fit_clusters"] = clusters.to_dict()
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'fit_clusters', cluster assignments for features (adata.uns)"
    )
    
    return adata if copy else None
    
def cluster_trends(trends, k=10, n_jobs=-1,metric="euclidean"):
    """Function to cluster gene trends, thank you palantir devs :)
    :param trends: Matrix of gene expression trends
    :param k: K for nearest neighbor construction
    :param n_jobs: Number of jobs for parallel processing
    :return: Clustering of gene trends
    """

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    # Cluster
    clusters, _, _ = phenograph.cluster(trends, k=k,
                                        primary_metric = metric,
                                        n_jobs=n_jobs)
    clusters = pd.Series(clusters, index=trends.index)
    return clusters