# Grapheno was modified to accept non cudf entries and is working with rapids-0.17

import time
import cuml
import cudf
import cugraph
import cupy as cp
import numpy as np
from cupy.sparse import csr_matrix


def find_neighbors(X, n_neighbors, metric, algorithm, distributed):
    """
    Returns the indices of the k-nearest neighbors for each cell
    in a cell-by-feature dataframe.

    Parameters
    ----------
    X : cudf.DataFrame
        Input cell-by-feature dataframe.
    n_neighbors : int
        Number of neighbors for kNN.
    metric: string
        Distance metric to use for kNN.
        Currently, only 'euclidean' is supported.
    algorithm: string
        The query algorithm to use.
        Currently, only 'brute' is supported.
    distributed: bool
        If True, use a multi-GPU dask cluster for kNN search.
    Returns
    -------
    idx : cudf.DataFrame
        The indices of the kNN for each cell in X.
    """

    print(
        f"Finding {n_neighbors} nearest neighbors using "
        f"{metric} metric and {algorithm} algorithm...",
        flush=True,
    )

    if distributed:
        print("Running distributed kNN...")
        import dask_cudf
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        from cuml.dask.neighbors import NearestNeighbors

        cluster = LocalCUDACluster()
        client = Client(cluster)
        npartitions = len(cluster.cuda_visible_devices)

        X_dask = dask_cudf.from_cudf(X, npartitions=npartitions)

        # Use n_neighbors + 1 to account for self index
        model = cuml.dask.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, client=client
        )

        model.fit(X_dask)

        idx = model.kneighbors(X_dask, return_distance=False).compute()

        client.shutdown()

    else:
        # Use n_neighbors + 1 to account for self index
        model = cuml.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, metric=metric, algorithm=algorithm
        )

        model.fit(X)

        idx = model.kneighbors(X, return_distance=False)

    # Drop self index
    # this replace original grephano implementation that just remove the first index,assuming it is self index
    # which is not always the case!
    idx = np.vstack(list(map(lambda i: idx[i, idx[i, :] != i], range(idx.shape[0]))))

    return idx


def kneighbors_graph(idx, n_neighbors, n_fit):
    """
    Returns k-neighbors graph built using k-nearest neighbors indices.

    Parameters
    ----------
    idx : cudf.DataFrame
        The indices of the kNN for each cell in X.
    n_neighbors : int
        Number of neighbors for kNN.
    n_fit: int
        Distance metric to use for kNN.
        Currently, only 'euclidean' is supported.
    Returns
    -------
    G : cugraph.Graph
        k-neighbors graph.
    """
    n_nonzero = n_neighbors * n_fit
    weight = cp.ones(n_nonzero)
    indptr = cp.arange(0, n_nonzero + 1, n_neighbors)
    graph = csr_matrix((weight, cp.array(idx.ravel()), indptr), shape=(n_fit, n_fit))

    offsets = cudf.Series(graph.indptr)
    indices = cudf.Series(graph.indices)

    G = cugraph.Graph()
    G.from_cudf_adjlist(offsets, indices, None)

    return G


def sort_by_size(clusters, min_size):
    """
    Relabel clustering in order of descending cluster size.
    New labels are consecutive integers beginning at 0
    Clusters that are smaller than min_size are assigned to -1.
    Copied from https://github.com/jacoblevine/PhenoGraph.

    Parameters
    ----------
    clusters: numpy array
        Either numpy or cupy array of cluster labels.
    min_size: int
        Minimum cluster size.
    Returns
    -------
    relabeled: cupy array
        Array of cluster labels re-labeled by size.

    """
    relabeled = cp.zeros(clusters.shape, dtype=cp.int)
    sizes = cp.array([cp.sum(clusters == x) for x in cp.unique(clusters)])
    o = cp.argsort(sizes)[::-1]
    for i, c in enumerate(o):
        if sizes[c] > min_size:
            relabeled[clusters == c] = i
        else:
            relabeled[clusters == c] = -1
    return relabeled


def cluster(
    X,
    n_neighbors=30,
    community="louvain",
    metric="euclidean",
    algorithm="brute",
    similarity="jaccard",
    min_size=10,
    distributed=False,
):
    """
    Clusters

    Parameters
    ----------
    X : cudf.DataFrame
        Input cell-by-feature dataframe.
    n_neighbors : int
        Number of neighbors for kNN.
    community: string
        Community detection algorithm to use.
        Deault is 'louvain'.
    metric: string
        Distance metric to use for kNN.
        Currently, only 'euclidean' is supported.
    algorithm: string
        The query algorithm to use.
        Currently, only 'brute' is supported.
    similarity: string
        Similarity metric to use for neighbor edge refinement.
        Default is 'jaccard'.
    min_size: int
        Minimum cluster size.
    distributed: bool
        If True, use a multi-GPU dask cluster for kNN search.
    Returns
    -------
    communities: cudf.DataFrame
        Community labels.
    G: cugraph.Graph
        k-neighbors graph.
    Q: float
        Modularity score for detected communities.
        Q is not returned if community='ecg' is used.
    """

    tic = time.time()
    # Go!

    idx = find_neighbors(X, n_neighbors, metric, algorithm, distributed)

    print(f"Neighbors computed in {time.time() - tic} seconds...")

    subtic = time.time()

    G = kneighbors_graph(idx, n_neighbors, X.shape[0])

    if similarity == "overlap":
        print("Computing overlap similarity...", flush=True)
        G = cugraph.overlap(G)

    else:
        similarity = "jaccard"
        print("Computing Jaccard similarity...", flush=True)
        G = cugraph.jaccard(G)

    print(
        f"{similarity} graph constructed in {time.time() - subtic} seconds...",
        flush=True,
    )

    g = cugraph.symmetrize_df(G, "source", "destination")
    G = cugraph.Graph()
    G.from_cudf_edgelist(g, edge_attr=f"{similarity}_coeff")
    del g

    if community == "louvain":

        print("Running Louvain modularity optimization...", flush=True)

        parts, Q = cugraph.louvain(G, max_iter=1000)

        communities = sort_by_size(
            cp.asarray(parts.sort_values(by="vertex").partition), min_size
        )

        n_parts = cp.unique(communities).shape[0]

        print(f"grapheno completed in {time.time() - tic} seconds...", flush=True)
        print(f"Communities detected: {n_parts}", flush=True)
        print(f"Modularity: {Q}", flush=True)

        return communities, G, Q

    elif community == "leiden":

        print("Running Leiden modularity optimization...", flush=True)

        parts, Q = cugraph.leiden(G, max_iter=1000)

        communities = sort_by_size(
            cp.asarray(parts.sort_values(by="vertex").partition), min_size
        )

        n_parts = cp.unique(communities).shape[0]

        print(f"grapheno completed in {time.time() - tic} seconds...", flush=True)
        print(f"Communities detected: {n_parts}", flush=True)
        print(f"Modularity: {Q}", flush=True)

        return communities, G, Q

    elif community == "ecg":

        print("Running ECG...", flush=True)
        parts = cugraph.ecg(G)
        communities = sort_by_size(
            cp.asarray(parts.sort_values(by="vertex").partition), min_size
        )

        n_parts = cp.unique(communities).shape[0]

        print(f"grapheno completed in {time.time() - tic} seconds...", flush=True)
        print(f"Communities detected: {n_parts}", flush=True)

        return communities, G, None

    # Insert any community/clustering method...
    elif community == "your favorite method":
        pass
