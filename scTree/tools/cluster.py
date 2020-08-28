from types import MappingProxyType
from typing import Optional, Tuple, Sequence, Type, Mapping, Any
from packaging import version

import numpy as np
import pandas as pd
from anndata import AnnData

from .. import logging as logg
from .. import settings

import igraph
from sklearn.preprocessing import StandardScaler
from scanpy.neighbors import _get_sparse_matrix_from_indices_distances_umap, compute_neighbors_umap, _compute_connectivities_umap


try:
    from louvain.VertexPartition import MutableVertexPartition
except ImportError:
    class MutableVertexPartition: pass
    MutableVertexPartition.__module__ = 'louvain.VertexPartition'

def cluster(
    adata: AnnData,
    knn: int = 10,
    metric: str = "euclidean",
    device: str = "cpu",
    resolution: float = 1,
    use_weights: bool = False,
    copy: bool = False,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    random_state = 0,
    n_jobs: int = 1,
    n_iterations: int = -1):

    adata = data.copy() if copy else adata
    
    partition_kwargs = dict(partition_kwargs)
    
    if "fitted" not in adata.layers:
        raise ValueError(
            "You need to run `tl.fit` first to fit the features before clustering them."
        )
    
    start=logg.info("feature trends clustering using leiden",reset=True)
    
    if device == "gpu":
        logg.info('    computing connectivities using "neighbors" from cuGraph')
        knn_indices, knn_dists = compute_neighbors_gpu(X=adata.layers["fitted"].T,
                                                       n_neighbors=knn,metric=metric)
        
        distances, connectivities=_compute_connectivities_umap(knn_indices, 
                                                               knn_dists,
                                                               knn_dists.shape[0], 
                                                               knn_dists.shape[1],
                                                               set_op_mix_ratio=1.0,
                                                               local_connectivity=1.0)
        
        logg.info('    detecting communities using "leiden" from cuGraph')
        import cudf
        import cugraph
        offsets = cudf.Series(connectivities.indptr)
        indices = cudf.Series(connectivities.indices)
        if use_weights:
            sources, targets = connectivities.nonzero()
            weights = connectivities[sources, targets]
            if isinstance(weights, np.matrix):
                weights = weights.A1
            weights = cudf.Series(weights)
        else:
            weights = None
        g = cugraph.Graph()
        g.add_adj_list(offsets, indices, weights)
        leiden_parts, _ = cugraph.leiden(g,resolution=resolution,max_iter=1000)
        groups = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
        
        
    elif device =="cpu":
        logg.info('    computing connectivities using "neighbors" from sklearn')
        knn_indices, knn_dists = compute_neighbors_cpu(X=adata.layers["fitted"].T,
                                                       n_neighbors=knn,metric=metric)
        
        distances, connectivities=_compute_connectivities_umap(knn_indices, 
                                                               knn_dists,
                                                               knn_dists.shape[0], 
                                                               knn_dists.shape[1],
                                                               set_op_mix_ratio=1.0,
                                                               local_connectivity=1.0)
        sources, targets = connectivities.nonzero()
        weights = connectivities[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = igraph.Graph(directed=None)
        g.add_vertices(connectivities.shape[0])
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except:
            pass
        
        logg.info('    detecting communities using "leiden" from leidenalg')
        import leidenalg
        if partition_type is None:
            partition_type = leidenalg.RBConfigurationVertexPartition
        # Prepare find_partition arguments as a dictionary,
        # appending to whatever the user provided. It needs to be this way
        # as this allows for the accounting of a None resolution
        # (in the case of a partition variant that doesn't take it on input)
        if use_weights:
            partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
        partition_kwargs['n_iterations'] = n_iterations
        partition_kwargs['seed'] = random_state
        partition_kwargs['resolution_parameter'] = resolution
        # clustering proper
        part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
        # store output into adata.obs
        groups = np.array(part.membership)
    
    
    adata.var["fit_clusters"] = groups
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'fit_clusters', cluster assignments for features (adata.var)"
    )
    
    return adata if copy else None
    


def compute_neighbors_gpu(
    X: np.ndarray,
    n_neighbors: int,
    metric: str
):
    """Compute nearest neighbors using RAPIDS cuml.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to compute nearest neighbors for.
    n_neighbors
        The number of neighbors to use.
        Returns
    -------
    **knn_indices**, **knn_dists** : np.arrays of shape (n_observations, n_neighbors)
    """
    from cuml.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors,metric=metric)
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    knn_dist, knn_indices = nn.kneighbors(X_contiguous)
    return knn_indices, knn_dist 

def compute_neighbors_cpu(
    X: np.ndarray,
    n_neighbors: int,
    metric: str,
    n_jobs: int =1
):
    """Compute nearest neighbors using RAPIDS cuml.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to compute nearest neighbors for.
    n_neighbors
        The number of neighbors to use.
        Returns
    -------
    **knn_indices**, **knn_dists** : np.arrays of shape (n_observations, n_neighbors)
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors,metric=metric,n_jobs=n_jobs)
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    knn_dist, knn_indices = nn.kneighbors(X_contiguous)
    
    return knn_indices, knn_dist
