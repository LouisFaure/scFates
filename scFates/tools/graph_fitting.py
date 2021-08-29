from typing import Optional, Union
from anndata import AnnData
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import shortest_path
import igraph
from tqdm import tqdm
import sys
import igraph
import warnings
import itertools
import math
from scipy import sparse
import simpleppt

from ..plot.trajectory import graph as plot_graph
from .. import logging as logg
from .. import settings
from .utils import get_X


def curve(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    epg_lambda: Optional[Union[float, int]] = 0.01,
    epg_mu: Optional[Union[float, int]] = 0.1,
    epg_trimmingradius: Optional = np.inf,
    epg_initnodes: Optional[int] = 2,
    epg_verbose: bool = False,
    device: str = "cpu",
    plot: bool = False,
    basis: Optional[str] = "umap",
    seed: Optional[int] = None,
    copy: bool = False,
):
    """\
    Generate a principal curve.

    Learn a curved representation on any space, composed of nodes, approximating the
    position of the cells on a given space such as gene expression, pca, diffusion maps, ...
    Uses ElpiGraph algorithm.

    Parameters
    ----------
    adata
        Annotated data matrix.
    Nodes
        Number of nodes composing the principial tree, use a range of 10 to 100 for
        ElPiGraph approach and 100 to 2000 for PPT approach.
    use_rep
        Choose the space to be learned by the principal tree.
    ndims_rep
        Number of dimensions to use for the inference.
    init
        Initialise the point positions.
    epg_lambda
        Parameter for ElPiGraph, coefficient of ‘stretching’ elasticity [Albergante20]_.
    epg_mu
        Parameter for ElPiGraph, coefficient of ‘bending’ elasticity [Albergante20]_.
    epg_trimmingradius
        Parameter for ElPiGraph, trimming radius for MSE-based data approximation term [Albergante20]_.
    epg_initnodes
        numerical 2D matrix, the k-by-m matrix with k m-dimensional positions of the nodes
        in the initial step
    epg_verbose
        show verbose output of epg algorithm
    device
        Run method on either `cpu` or on `gpu`
    plot
        Plot the resulting tree.
    basis
        Basis onto which the resulting tree should be projected.
    seed
        A numpy random seed.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['epg']`
            dictionnary containing information from elastic principal curve
        `.uns['graph']['B']`
            adjacency matrix of the principal points
        `.uns['graph']['R']`
            soft assignment of cells to principal point in representation space
        `.uns['graph']['F']`
            coordinates of principal points in representation space
    """

    logg.info(
        "inferring a principal curve",
        reset=True,
        end=" " if settings.verbosity > 2 else "\n",
    )

    adata = adata.copy() if copy else adata

    if Nodes is None:
        if adata.shape[0] * 2 > 100:
            Nodes = 100
        else:
            Nodes = int(adata.shape[0] / 2)

    logg.hint(
        "parameters used \n"
        "    "
        + str(Nodes)
        + " principal points, mu = "
        + str(epg_mu)
        + ", lambda = "
        + str(epg_lambda)
    )
    curve_epg(
        adata,
        Nodes,
        use_rep,
        ndims_rep,
        init,
        epg_lambda,
        epg_mu,
        epg_trimmingradius,
        epg_initnodes,
        device,
        seed,
        epg_verbose,
    )

    if plot:
        plot_graph(adata, basis)

    return adata if copy else None


def tree(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    method: str = None,
    init: Optional[DataFrame] = None,
    ppt_sigma: Optional[Union[float, int]] = 0.1,
    ppt_lambda: Optional[Union[float, int]] = 1,
    ppt_metric: str = "euclidean",
    ppt_nsteps: int = 50,
    ppt_err_cut: float = 5e-3,
    ppt_gpu_tpb: int = 16,
    epg_lambda: Optional[Union[float, int]] = 0.01,
    epg_mu: Optional[Union[float, int]] = 0.1,
    epg_trimmingradius: Optional = np.inf,
    epg_initnodes: Optional[int] = 2,
    epg_verbose: bool = False,
    device: str = "cpu",
    plot: bool = False,
    basis: Optional[str] = "umap",
    seed: Optional[int] = None,
    copy: bool = False,
):
    """\
    Generate a principal tree.

    Learn a simplified representation on any space, compsed of nodes, approximating the
    position of the cells on a given space such as gene expression, pca, diffusion maps, ...
    If `method=='ppt'`, uses simpleppt implementation from [Soldatov19]_.
    If `method=='epg'`, uses Elastic Principal Graph approach from [Albergante20]_.

    Parameters
    ----------
    adata
        Annotated data matrix.
    Nodes
        Number of nodes composing the principial tree, use a range of 10 to 100 for
        ElPiGraph approach and 100 to 2000 for PPT approach.
    use_rep
        Choose the space to be learned by the principal tree.
    ndims_rep
        Number of dimensions to use for the inference.
    method
        If `ppt`, uses simpleppt approach, `ppt_lambda` and `ppt_sigma` are the
        parameters controlling the algorithm. If `epg`, uses ComputeElasticPrincipalTree
        function from elpigraph python package, `epg_lambda` `epg_mu` and `epg_trimmingradius`
        are the parameters controlling the algorithm.
    init
        Initialise the point positions.
    ppt_sigma
        Regularization parameter for simpleppt [Mao15]_.
    ppt_lambda
        Parameter for simpleppt, penalty for the tree length [Mao15]_.
    ppt_metric
        The metric to use to compute distances in high dimensional space.
        For compatible metrics, check the documentation of
        sklearn.metrics.pairwise_distances if using cpu or
        cuml.metrics.pairwise_distances if using gpu.
    ppt_nsteps
        Number of steps for the optimisation process of simpleppt.
    ppt_err_cut
        Stop simpleppt algorithm if proximity of principal points between iterations less than defiend value.
    ppt_gpu_tpb
        Threads per block parameter for cuda computations.
    epg_lambda
        Parameter for ElPiGraph, coefficient of ‘stretching’ elasticity [Albergante20]_.
    epg_mu
        Parameter for ElPiGraph, coefficient of ‘bending’ elasticity [Albergante20]_.
    epg_trimmingradius
        Parameter for ElPiGraph, trimming radius for MSE-based data approximation term [Albergante20]_.
    epg_initnodes
        numerical 2D matrix, the k-by-m matrix with k m-dimensional positions of the nodes
        in the initial step
    epg_verbose
        show verbose output of epg algorithm
    device
        Run either mehtod on `cpu` or on `gpu`
    plot
        Plot the resulting tree.
    basis
        Basis onto which the resulting tree should be projected.
    seed
        A numpy random seed.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['ppt']`
            dictionnary containing information from simpelppt tree if method='ppt'
        `.uns['epg']`
            dictionnary containing information from elastic principal tree if method='epg'
        `.uns['graph']['B']`
            adjacency matrix of the principal points
        `.uns['graph']['R']`
            soft assignment of cells to principal point in representation space
        `.uns['graph']['F']`
            coordinates of principal points in representation space
    """

    logg.info(
        "inferring a principal tree",
        reset=True,
        end=" " if settings.verbosity > 2 else "\n",
    )

    adata = adata.copy() if copy else adata

    X, use_rep = get_data(adata, use_rep, ndims_rep)

    if Nodes is None:
        if adata.shape[0] * 2 > 2000:
            Nodes = 2000
        else:
            Nodes = int(adata.shape[0] / 2)

    if method == "ppt":
        ppt = simpleppt.ppt(
            X,
            Nodes=Nodes,
            init=init,
            sigma=ppt_sigma,
            lam=ppt_lambda,
            metric=ppt_metric,
            nsteps=ppt_nsteps,
            err_cut=ppt_err_cut,
            device=device,
            gpu_tbp=ppt_gpu_tpb,
            seed=seed,
        )

        ppt = vars(ppt)

        graph = {
            "B": ppt["B"],
            "R": ppt["R"],
            "F": ppt["F"],
            "tips": ppt["tips"],
            "forks": ppt["forks"],
            "cells_fitted": X.index.tolist(),
            "metrics": ppt["metric"],
            "use_rep": use_rep,
            "ndims_rep": ndims_rep,
        }

        adata.uns["graph"] = graph
        adata.uns["ppt"] = ppt

    elif method == "epg":
        graph, epg = tree_epg(
            X,
            Nodes,
            init,
            epg_lambda,
            epg_mu,
            epg_trimmingradius,
            epg_initnodes,
            device,
            seed,
            epg_verbose,
        )
        graph["use_rep"] = use_rep
        graph["ndims_rep"] = ndims_rep
        adata.uns["graph"] = graph
        adata.uns["epg"] = epg

    if plot:
        plot_graph(adata, basis)

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['" + method + "'], dictionnary containing inferred tree.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['R'] soft assignment of cells to principal point in representation space.\n"
        "    .uns['graph']['F'] coordinates of principal points in representation space."
    )

    return adata if copy else None


def tree_epg(
    X,
    Nodes: int = None,
    init: Optional[DataFrame] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    initnodes: int = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
):

    try:
        import elpigraph

    except Exception as e:
        warnings.warn(
            'ElPiGraph package is not installed \
            \nPlease use "pip install git+https://github.com/j-bac/elpigraph-python.git" to install it'
        )
    logg.hint(
        "parameters used \n"
        "    "
        + str(Nodes)
        + " principal points, mu = "
        + str(mu)
        + ", lambda = "
        + str(lam)
    )

    if seed is not None:
        np.random.seed(seed)

    if device == "gpu":
        import cupy as cp
        from cuml.metrics import pairwise_distances
        from .utils import cor_mat_gpu

        Tree = elpigraph.computeElasticPrincipalTree(
            X.values.astype(np.float64),
            NumNodes=Nodes,
            Do_PCA=False,
            InitNodes=initnodes,
            Lambda=lam,
            Mu=mu,
            TrimmingRadius=trimmingradius,
            GPU=True,
            verbose=verbose,
        )

        R = pairwise_distances(
            cp.asarray(X.values), cp.asarray(Tree[0]["NodePositions"])
        )

        R = cp.asnumpy(R)
        # Hard assigment
        R = sparse.csr_matrix(
            (np.repeat(1, R.shape[0]), (range(R.shape[0]), R.argmin(axis=1))), R.shape
        ).A

    else:
        from .utils import cor_mat_cpu
        from sklearn.metrics import pairwise_distances

        Tree = elpigraph.computeElasticPrincipalTree(
            X.values.astype(np.float64),
            NumNodes=Nodes,
            Do_PCA=False,
            InitNodes=initnodes,
            Lambda=lam,
            Mu=mu,
            TrimmingRadius=trimmingradius,
            verbose=verbose,
        )

        R = pairwise_distances(X.values, Tree[0]["NodePositions"])
        # Hard assigment
        R = sparse.csr_matrix(
            (np.repeat(1, R.shape[0]), (range(R.shape[0]), R.argmin(axis=1))), R.shape
        ).A

    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(Tree[0]["Edges"][0].flatten().astype(int)))
    g.add_edges(
        pd.DataFrame(Tree[0]["Edges"][0]).astype(int).apply(tuple, axis=1).values
    )

    # mat = np.asarray(g.get_adjacency().data)
    # mat = mat + mat.T - np.diag(np.diag(mat))
    # B=((mat>0).astype(int))

    B = np.asarray(g.get_adjacency().data)

    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    graph = {
        "B": B,
        "R": R,
        "F": Tree[0]["NodePositions"].T,
        "tips": tips,
        "forks": forks,
        "cells_fitted": X.index.tolist(),
        "metrics": "euclidean",
    }

    Tree[0]["Edges"] = list(Tree[0]["Edges"])

    return graph, Tree[0]


def curve_epg(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    initnodes: int = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
):
    try:
        import elpigraph

    except Exception as e:
        warnings.warn(
            'ElPiGraph package is not installed \
            \nPlease use "pip install git+https://github.com/j-bac/elpigraph-python.git" to install it'
        )

    X, use_rep = get_data(adata, use_rep, ndims_rep)

    if seed is not None:
        np.random.seed(seed)

    if device == "gpu":
        import cupy as cp
        from .utils import cor_mat_gpu
        from cuml.metrics import pairwise_distances

        Curve = elpigraph.computeElasticPrincipalCurve(
            X.values.astype(np.float64),
            NumNodes=Nodes,
            Do_PCA=False,
            InitNodes=initnodes,
            Lambda=lam,
            Mu=mu,
            TrimmingRadius=trimmingradius,
            GPU=True,
            verbose=verbose,
        )

        R = pairwise_distances(
            cp.asarray(X.values), cp.asarray(Curve[0]["NodePositions"])
        )

        R = cp.asnumpy(R)
        # Hard assigment
        R = sparse.csr_matrix(
            (np.repeat(1, R.shape[0]), (range(R.shape[0]), R.argmin(axis=1))), R.shape
        ).A

    else:
        from .utils import cor_mat_cpu
        from sklearn.metrics import pairwise_distances

        Curve = elpigraph.computeElasticPrincipalCurve(
            X.values.astype(np.float64),
            NumNodes=Nodes,
            Do_PCA=False,
            InitNodes=initnodes,
            Lambda=lam,
            Mu=mu,
            TrimmingRadius=trimmingradius,
            verbose=verbose,
        )

        R = pairwise_distances(X.values, Curve[0]["NodePositions"])
        # Hard assigment
        R = sparse.csr_matrix(
            (np.repeat(1, R.shape[0]), (range(R.shape[0]), R.argmin(axis=1))), R.shape
        ).A

    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(Curve[0]["Edges"][0].flatten().astype(int)))
    g.add_edges(
        pd.DataFrame(Curve[0]["Edges"][0]).astype(int).apply(tuple, axis=1).values
    )

    # mat = np.asarray(g.get_adjacency().data)
    # mat = mat + mat.T - np.diag(np.diag(mat))
    # B=((mat>0).astype(int))

    B = np.asarray(g.get_adjacency().data)

    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    graph = {
        "B": B,
        "R": R,
        "F": Curve[0]["NodePositions"].T,
        "tips": tips,
        "forks": forks,
        "cells_fitted": X.index.tolist(),
        "metrics": "euclidean",
        "use_rep": use_rep,
        "ndims_rep": ndims_rep,
    }

    Curve[0]["Edges"] = list(Curve[0]["Edges"])

    adata.uns["graph"] = graph
    adata.uns["epg"] = Curve[0]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['epg'] dictionnary containing inferred elastic curve generated from elpigraph.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['R'] hard assignment of cells to principal point in representation space.\n"
        "    .uns['graph']['F'], coordinates of principal points in representation space."
    )

    return adata


def get_data(adata, use_rep, ndims_rep):

    if use_rep not in adata.obsm.keys() and f"X_{use_rep}" in adata.obsm.keys():
        use_rep = f"X_{use_rep}"

    if (
        (use_rep not in adata.layers.keys())
        & (use_rep not in adata.obsm.keys())
        & (use_rep != "X")
    ):
        use_rep = "X" if adata.n_vars < 50 or n_pcs == 0 else "X_pca"
        n_pcs = None if use_rep == "X" else n_pcs

    if use_rep == "X":
        ndims_rep = None
        if sparse.issparse(adata.X):
            X = DataFrame(adata.X.A, index=adata.obs_names)
        else:
            X = DataFrame(adata.X, index=adata.obs_names)
    elif use_rep in adata.layers.keys():
        if sparse.issparse(adata.layers[use_rep]):
            X = DataFrame(adata.layers[use_rep].A, index=adata.obs_names)
        else:
            X = DataFrame(adata.layers[use_rep], index=adata.obs_names)
    elif use_rep in adata.obsm.keys():
        X = DataFrame(adata.obsm[use_rep], index=adata.obs_names)

    if ndims_rep is not None:
        X = X.iloc[:, :ndims_rep]

    return X, use_rep
