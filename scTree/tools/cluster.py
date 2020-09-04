#from types import MappingProxyType
from typing import Optional, Tuple, Sequence, Type, Mapping, Any
from packaging import version

import numpy as np
import pandas as pd
from anndata import AnnData

from .. import logging as logg
from .. import settings

import igraph
import phenograph
#from scanpy.neighbors import compute_neighbors_umap, _compute_connectivities_umap

    
def cluster(
    adata: AnnData,
    knn: int = 10,
    metric: str = "euclidean",
    device: str = "cpu",
    copy: bool = False,
    n_jobs: int = 1): 
    
    adata = data.copy() if copy else adata
    
    if device == "gpu":
        import grapheno
        import cudf
        logg.info('    clustering using grapheno')
        clusters = grapheno.cluster(cudf.DataFrame(adata.layers["fitted"],
                                        index=adata.obs_names,
                                        columns=adata.var_names).T,
                         metric=metric,n_neighbors=knn)[0]
        
    elif device =="cpu":
        logg.info('    clustering using phenograph')
        clusters = phenograph.cluster(adata.layers["fitted"].T,
                         primary_metric=metric,k=knn,n_jobs=n_jobs)[0]
        
    adata.var["fit_clusters"] = clusters.get()
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'fit_clusters', cluster assignments for features (adata.var)"
    )
    
    return adata if copy else None
    