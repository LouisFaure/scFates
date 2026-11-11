import pandas as pd
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
    Cluster features. Uses scanpy when using cpu, and rapids-singlecell when using gpu.
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
        Run the analysis on 'cpu' (scanpy) or on 'gpu' (rapids-singlecell).
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.var['clusters']`
            module assignments for features.
    """

    adata = adata.copy() if copy else adata

    layer = "X" if layer not in adata.layers else layer

    logg.info(
        f"Clustering features using {layer} layer",
        reset=True,
        end="\n",
    )

    X, _ = get_data(adata, layer, None)
    adata_s = AnnData(
        X.values.T,
        obs=pd.DataFrame(index=adata.var_names),
        var=pd.DataFrame(index=adata.obs_names),
    )

    if device == "gpu":
        try:
            import rapids_singlecell as rsc
        except ModuleNotFoundError:
            raise Exception(
                "rapids-singlecell is required for device='gpu', use device='cpu' instead!"
            )

        rsc.pp.pca(adata_s, n_comps=n_pcs)
        rsc.pp.neighbors(adata_s, n_neighbors=n_neighbors, metric=metric)
        rsc.tl.leiden(adata_s, resolution=resolution)
    else:
        sc.pp.pca(adata_s, n_comps=n_pcs)
        sc.pp.neighbors(adata_s, n_neighbors=n_neighbors, metric=metric)
        sc.tl.leiden(adata_s, resolution=resolution, flavor="igraph", n_iterations=2)

    adata.var["clusters"] = adata_s.obs.leiden.values

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    .var['clusters'] identified modules.")

    return adata if copy else None
