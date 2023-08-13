import scanpy as sc
from anndata import AnnData
from .. import logging as logg
from .. import settings
from .graph_fitting import get_data


def cluster(
    adata: AnnData,
    layer="fitted",
    n_neighbors: int = 20,
    n_pcs: int = 50,
    metric: str = "cosine",
    resolution: float = 1,
    device: str = "cpu",
    copy: bool = False,
):
    """\
    Cluster features. Uses scanpy backend when using cpu, and cuml when using gpu.
    Dataset is transposed, PCA is calulcated and a nearest neighbor graph is generated
    from PC space. Leiden algorithm is used for community detection.


    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer of feature to calculate clusters, by default fitted features
    n_neighbors
        Number of neighbors.
    n_pcs
        Number of PC to keep for PCA.
    metric
        distance metric to use for clustering.
    resolution
        Resolution parameter for leiden algorithm.
    device
        run the analysis on 'cpu' with phenograph, or on 'gpu' with grapheno.

    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:
        `.var['cluters']`
            module assignments for features.
    """

    layer = "X" if layer not in adata.layers else layer

    logg.info(
        f"Clustering features using {layer} layer",
        reset=True,
        end="\n",
    )

    X, use_rep = get_data(adata, layer, None)

    adata_s = adata.copy().T
    if device == "gpu":
        from cuml.decomposition import PCA

        pca = PCA(n_components=n_pcs)
        adata_s.obsm["X_pca"] = pca.fit_transform(adata_s.X)
    else:
        sc.pp.pca(adata_s, n_comps=n_pcs)

    sc.pp.neighbors(
        adata_s,
        n_neighbors=n_neighbors,
        metric=metric,
        method="rapids" if device == "gpu" else "umap",
    )

    if device == "gpu":
        from scanpy.tools import _utils

        adjacency = _utils._choose_graph(adata_s, None, None)
        import cudf
        import cugraph
        from natsort import natsorted
        import numpy as np
        import pandas as pd

        offsets = cudf.Series(adjacency.indptr)
        indices = cudf.Series(adjacency.indices)

        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        weights = cudf.Series(weights)

        g = cugraph.Graph()

        if hasattr(g, "add_adj_list"):
            g.add_adj_list(offsets, indices, weights)
        else:
            g.from_cudf_adjlist(offsets, indices, weights)

        leiden_parts, _ = cugraph.leiden(g, resolution=resolution)
        groups = (
            leiden_parts.to_pandas()
            .sort_values("vertex")[["partition"]]
            .to_numpy()
            .ravel()
        )
        adata_s.obs["leiden"] = pd.Categorical(
            values=groups.astype("U"),
            categories=natsorted(map(str, np.unique(groups))),
        )
    else:
        sc.tl.leiden(adata_s, resolution=resolution)

    adata.var["clusters"] = adata_s.obs.leiden

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    .var['clusters'] identified modules.")
