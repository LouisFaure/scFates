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

from ..plot.trajectory import graph as plot_graph
from .. import logging as logg
from .. import settings

try:
    import elpigraph

except Exception as e:
    warnings.warn(
        'ElPiGraph package is not installed \
        \nPlease use "pip install git+https://github.com/j-bac/elpigraph-python.git" to install it'
    )


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
        "inferring a principal tree",
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

    if Nodes is None:
        if adata.shape[0] * 2 > 2000:
            Nodes = 2000
        else:
            Nodes = int(adata.shape[0] / 2)

    if method == "ppt":
        logg.hint(
            "parameters used \n"
            "    "
            + str(Nodes)
            + " principal points, sigma = "
            + str(ppt_sigma)
            + ", lambda = "
            + str(ppt_lambda)
            + ", metric = "
            + ppt_metric
        )
        tree_ppt(
            adata,
            Nodes=Nodes,
            use_rep=use_rep,
            ndims_rep=ndims_rep,
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

    elif method == "epg":
        logg.hint(
            "parameters used \n"
            "    "
            + str(Nodes)
            + " principal points, mu = "
            + str(epg_mu)
            + ", lambda = "
            + str(epg_lambda)
        )
        tree_epg(
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


def tree_ppt(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    sigma: Optional[Union[float, int]] = 0.1,
    lam: Optional[Union[float, int]] = 1,
    metric: str = "euclidean",
    nsteps: int = 50,
    err_cut: float = 5e-3,
    device: str = "cpu",
    gpu_tbp: int = 16,
    seed: Optional[int] = None,
):

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
    X_t = X.values.T

    # if seed is not None:
    #    np.random.seed(seed)

    if device == "gpu":
        import rmm

        rmm.reinitialize(managed_memory=True)
        assert rmm.is_initialized()
        import cupy as cp
        from cuml.metrics import pairwise_distances
        from .utils import process_R_gpu, norm_R_gpu, cor_mat_gpu, mst_gpu, matmul

        X_gpu = cp.asarray(X_t, dtype=np.float64)
        W = cp.empty_like(X_gpu)
        W.fill(1)

        if init is None:
            if seed is not None:
                np.random.seed(seed)
            F_mat_gpu = X_gpu[
                :, np.random.choice(X.shape[0], size=Nodes, replace=False)
            ]
        else:
            F_mat_gpu = cp.asarray(init.T)
            M = init.T.shape[0]

        iterator = tqdm(range(nsteps), file=sys.stdout, desc="    fitting")
        for i in iterator:
            R = pairwise_distances(X_gpu.T, F_mat_gpu.T, metric=metric)

            threadsperblock = (gpu_tbp, gpu_tbp)
            blockspergrid_x = math.ceil(R.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(R.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            process_R_gpu[blockspergrid, threadsperblock](R, sigma)
            Rsum = R.sum(axis=1)
            norm_R_gpu[blockspergrid, threadsperblock](R, Rsum)

            d = pairwise_distances(F_mat_gpu.T, metric=metric)
            mst = mst_gpu(d)
            mat = mst + mst.T - cp.diag(cp.diag(mst.A))
            B = (mat > 0).astype(int)

            D = cp.identity(B.shape[0]) * B.sum(axis=0)
            L = D - B
            M = L * lam + cp.identity(R.shape[1]) * R.sum(axis=0)
            old_F = F_mat_gpu

            dotprod = cp.zeros((X_gpu.shape[0], R.shape[1]))
            TPB = 16
            threadsperblock = (gpu_tbp, gpu_tbp)
            blockspergrid_x = math.ceil(dotprod.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(dotprod.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            matmul[blockspergrid, threadsperblock]((X_gpu * W), R, dotprod)

            F_mat_gpu = cp.linalg.solve(M.T, dotprod.T).T

            err = cp.max(
                cp.sqrt((F_mat_gpu - old_F).sum(axis=0) ** 2)
                / cp.sqrt((F_mat_gpu ** 2).sum(axis=0))
            )
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break

        if i == (nsteps - 1):
            logg.info("    inference not converged (error: " + str(err) + ")")

        score = cp.array(
            [
                cp.sum((1 - cor_mat_gpu(F_mat_gpu, X_gpu)) * R) / R.shape[0],
                sigma / R.shape[0] * cp.sum(R * cp.log(R)),
                lam / 2 * cp.sum(d * B),
            ]
        )

        ppt = [
            X.index.tolist(),
            cp.asnumpy(score),
            cp.asnumpy(F_mat_gpu),
            cp.asnumpy(R),
            cp.asnumpy(B),
            cp.asnumpy(L),
            cp.asnumpy(d),
            lam,
            sigma,
            nsteps,
            metric,
        ]
    else:
        from sklearn.metrics import pairwise_distances
        from .utils import process_R_cpu, norm_R_cpu, cor_mat_cpu

        X_cpu = np.asarray(X_t, dtype=np.float64)
        W = np.empty_like(X_cpu)
        W.fill(1)

        if init is None:
            if seed is not None:
                np.random.seed(seed)
            F_mat_cpu = X_cpu[
                :, np.random.choice(X.shape[0], size=Nodes, replace=False)
            ]
        else:
            F_mat_cpu = np.asarray(init.T)
            Nodes = init.T.shape[0]

        j = 1
        err = 100

        # while ((j <= nsteps) & (err > err_cut)):
        iterator = tqdm(range(nsteps), file=sys.stdout, desc="    fitting")
        for i in iterator:
            R = pairwise_distances(X_cpu.T, F_mat_cpu.T, metric=metric)

            process_R_cpu(R, sigma)
            Rsum = R.sum(axis=1)
            norm_R_cpu(R, Rsum)

            d = pairwise_distances(F_mat_cpu.T, metric=metric)

            csr = csr_matrix(np.triu(d, k=-1))
            Tcsr = minimum_spanning_tree(csr)
            mat = Tcsr.toarray()
            mat = mat + mat.T - np.diag(np.diag(mat))
            B = (mat > 0).astype(int)

            D = (np.identity(B.shape[0])) * np.array(B.sum(axis=0))
            L = D - B
            M = L * lam + np.identity(R.shape[1]) * np.array(R.sum(axis=0))
            old_F = F_mat_cpu

            F_mat_cpu = np.linalg.solve(M.T, (np.dot(X_cpu * W, R)).T).T

            err = np.max(
                np.sqrt((F_mat_cpu - old_F).sum(axis=0) ** 2)
                / np.sqrt((F_mat_cpu ** 2).sum(axis=0))
            )

            err = err.item()
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break

        if i == (nsteps - 1):
            logg.info("    not converged (error: " + str(err) + ")")

        score = [
            np.sum((1 - cor_mat_cpu(F_mat_cpu, X_cpu)) * R) / R.shape[0],
            sigma / R.shape[0] * np.sum(R * np.log(R)),
            lam / 2 * np.sum(d * B),
        ]

        ppt = [
            X.index.tolist(),
            score,
            F_mat_cpu,
            R,
            B,
            L,
            d,
            lam,
            sigma,
            nsteps,
            metric,
        ]

    names = [
        "cells_fitted",
        "score",
        "F",
        "R",
        "B",
        "L",
        "d",
        "lambda",
        "sigma",
        "nsteps",
        "metric",
    ]
    ppt = dict(zip(names, ppt))

    g = igraph.Graph.Adjacency((ppt["B"] > 0).tolist(), mode="undirected")

    # remvoe lonely nodes
    co_nodes = np.argwhere(np.array(g.degree()) > 0).ravel()
    ppt["R"] = ppt["R"][:, co_nodes]
    ppt["F"] = ppt["F"][:, co_nodes]
    ppt["B"] = ppt["B"][co_nodes, :][:, co_nodes]
    ppt["L"] = ppt["L"][co_nodes, :][:, co_nodes]
    ppt["d"] = ppt["d"][co_nodes, :][:, co_nodes]

    if len(co_nodes) < Nodes:
        logg.info("    " + str(Nodes - len(co_nodes)) + " lonely nodes removed")

    g = igraph.Graph.Adjacency((ppt["B"] > 0).tolist(), mode="undirected")
    ppt["tips"] = np.argwhere(np.array(g.degree()) == 1).flatten()
    ppt["forks"] = np.argwhere(np.array(g.degree()) > 2).flatten()

    if len(ppt["tips"]) > 30:
        logg.info("    more than 30 tips detected!")

    graph = {
        "B": ppt["B"],
        "R": ppt["R"],
        "F": ppt["F"],
        "tips": ppt["tips"],
        "forks": ppt["forks"],
        "cells_fitted": X.index.tolist(),
        "metrics": ppt["metric"],
    }

    adata.uns["graph"] = graph

    adata.uns["ppt"] = ppt

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['ppt'], dictionnary containing inferred tree.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['R'] soft assignment of cells to principal point in representation space.\n"
        "    .uns['graph']['F'] coordinates of principal points in representation space."
    )

    return adata


def tree_epg(
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

    adata.uns["graph"] = graph
    adata.uns["epg"] = Tree[0]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['epg'] dictionnary containing inferred elastic tree generated from elpigraph.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['R'] hard assignment of cells to principal point in representation space.\n"
        "    .uns['graph']['F'] coordinates of principal points in representation space."
    )

    return adata


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


def cleanup(
    adata: AnnData,
    minbranchlength: int = 3,
    leaves: Optional[int] = None,
    copy: bool = False,
):
    """\
    Remove spurious branches from the tree.

    Parameters
    ----------
    adata
        Annotated data matrix.
    minbranchlength
        Branches having less than the defined amount of nodes are discarded
    leaves
        Manually select branch tips to remove
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['B']`
            subsetted adjacency matrix of the principal points.
        `.uns['graph']['R']`
            subsetted updated soft assignment of cells to principal point in representation space.
        `.uns['graph']['F']`
            subsetted coordinates of principal points in representation space.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before cleaning it"
        )
    graph = adata.uns["graph"]

    B = graph["B"]
    R = graph["R"]
    F = graph["F"]
    init_num = B.shape[0]
    init_pp = np.arange(B.shape[0])
    if leaves is not None:
        g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
        tips = np.argwhere(np.array(g.degree()) == 1).flatten()
        branches = np.argwhere(np.array(g.degree()) > 2).flatten()
        idxmin = list(
            map(
                lambda l: np.argmin(
                    list(map(len, g.get_all_shortest_paths(l, branches)))
                ),
                leaves,
            )
        )
        torem_manual = np.concatenate(
            list(
                map(
                    lambda i: np.array(
                        g.get_shortest_paths(leaves[i], branches[idxmin[i]])[0][:-1]
                    ),
                    range(len(leaves)),
                )
            )
        )
        B = np.delete(B, torem_manual, axis=0)
        B = np.delete(B, torem_manual, axis=1)
        R = np.delete(R, torem_manual, axis=1)
        F = np.delete(F, torem_manual, axis=1)

    while True:
        torem = []
        g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
        tips = np.argwhere(np.array(g.degree()) == 1).flatten()
        branches = np.argwhere(np.array(g.degree()) > 2).flatten()

        if len(branches) == 0:
            break

        dist = np.array(
            list(
                map(
                    lambda t: np.min(
                        list(map(len, g.get_all_shortest_paths(t, branches)))
                    ),
                    tips,
                )
            )
        )

        if np.min(dist) > minbranchlength:
            break

        tip_torem = tips[np.argmin(dist)].T.flatten()
        B = np.delete(B, tip_torem, axis=0)
        B = np.delete(B, tip_torem, axis=1)
        R = np.delete(R, tip_torem, axis=1)
        F = np.delete(F, tip_torem, axis=1)
    R = (R.T / R.sum(axis=1)).T
    graph["R"] = R
    graph["B"] = B
    graph["F"] = F
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    graph["tips"] = np.argwhere(np.array(g.degree()) == 1).flatten()
    graph["forks"] = np.argwhere(np.array(g.degree()) > 2).flatten()

    adata.uns["graph"] = graph

    logg.info(
        "    graph cleaned", time=False, end=" " if settings.verbosity > 2 else "\n"
    )
    logg.hint("removed " + str(init_num - B.shape[0]) + " principal points")

    return adata if copy else None


def root(adata: AnnData, root: int, copy: bool = False):
    """\
    Define the root of the trajectory.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root
        Id of the tip of the fork to be considered as a root.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['root']`
            selected root.
        `.uns['graph']['pp_info']`
            for each PP, its distance vs root and segment assignment.
        `.uns['graph']['pp_seg']`
            segments network information.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` or `tl.curve` first to compute a princal graph before choosing a root."
        )

    graph = adata.uns["graph"]

    from sklearn.metrics import pairwise_distances

    d = 1e-6 + pairwise_distances(graph["F"].T, graph["F"].T, metric=graph["metrics"])

    to_g = graph["B"] * d

    csr = csr_matrix(to_g)

    g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
    g.es["weight"] = to_g[to_g.nonzero()]

    root_dist_matrix = shortest_path(csr, directed=False, indices=root)
    pp_info = pd.DataFrame(
        {"PP": g.vs.indices, "time": root_dist_matrix, "seg": np.zeros(csr.shape[0])}
    )

    nodes = np.argwhere(
        np.apply_along_axis(arr=(csr > 0).todense(), axis=0, func1d=np.sum) != 2
    ).flatten()
    nodes = np.unique(np.append(nodes, root))

    pp_seg = pd.DataFrame(columns=["n", "from", "to", "d"])
    for node1, node2 in itertools.combinations(nodes, 2):
        paths12 = g.get_shortest_paths(node1, node2)
        paths12 = np.array([val for sublist in paths12 for val in sublist])

        if np.sum(np.isin(nodes, paths12)) == 2:
            fromto = np.array([node1, node2])
            path_root = root_dist_matrix[[node1, node2]]
            fro = fromto[np.argmin(path_root)]
            to = fromto[np.argmax(path_root)]
            pp_info.loc[paths12, "seg"] = pp_seg.shape[0] + 1
            pp_seg = pp_seg.append(
                pd.DataFrame(
                    {
                        "n": pp_seg.shape[0] + 1,
                        "from": fro,
                        "to": to,
                        "d": shortest_path(csr, directed=False, indices=fro)[to],
                    },
                    index=[pp_seg.shape[0] + 1],
                )
            )

    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)
    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

    pp_seg["from"] = pp_seg["from"].astype(int)
    pp_seg["to"] = pp_seg["to"].astype(int)

    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)
    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)

    graph["pp_info"] = pp_info
    graph["pp_seg"] = pp_seg
    graph["root"] = root

    adata.uns["graph"] = graph

    logg.info("root selected", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n"
        "    .uns['graph']['root'] selected root.\n"
        "    .uns['graph']['pp_info'] for each PP, its distance vs root and segment assignment.\n"
        "    .uns['graph']['pp_seg'] segments network information."
    )

    return adata if copy else None


def roots(adata: AnnData, roots, meeting, copy: bool = False):

    """\
    Define 2 roots of the tree.

    Parameters
    ----------
    adata
        Annotated data matrix.
    roots
        list of tips or forks to be considered a roots.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['graph']['root']`
            farthest root selected.
        `.uns['graph']['root2']`
            2nd root selected.
        `.uns['graph']['meeting']`
            meeting point on the tree.
        `.uns['graph']['pp_info']`
            for each PP, its distance vs root and segment assignment).
        `.uns['graph']['pp_seg']`
            segments network information.
    """

    adata = adata.copy() if copy else adata

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before choosing two roots."
        )

    graph = adata.uns["graph"]

    from sklearn.metrics import pairwise_distances

    d = 1e-6 + pairwise_distances(graph["F"].T, graph["F"].T, metric=graph["metrics"])

    to_g = graph["B"] * d

    csr = csr_matrix(to_g)

    g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
    g.es["weight"] = to_g[to_g.nonzero()]

    root = roots[
        np.argmax(shortest_path(csr, directed=False, indices=roots)[:, meeting])
    ]
    root2 = roots[
        np.argmin(shortest_path(csr, directed=False, indices=roots)[:, meeting])
    ]

    root_dist_matrix = shortest_path(csr, directed=False, indices=root)
    pp_info = pd.DataFrame(
        {"PP": g.vs.indices, "time": root_dist_matrix, "seg": np.zeros(csr.shape[0])}
    )

    nodes = np.argwhere(
        np.apply_along_axis(arr=(csr > 0).todense(), axis=0, func1d=np.sum) != 2
    ).flatten()
    pp_seg = pd.DataFrame(columns=["n", "from", "to", "d"])
    for node1, node2 in itertools.combinations(nodes, 2):
        paths12 = g.get_shortest_paths(node1, node2)
        paths12 = np.array([val for sublist in paths12 for val in sublist])

        if np.sum(np.isin(nodes, paths12)) == 2:
            fromto = np.array([node1, node2])
            path_root = root_dist_matrix[[node1, node2]]
            fro = fromto[np.argmin(path_root)]
            to = fromto[np.argmax(path_root)]
            pp_info.loc[paths12, "seg"] = pp_seg.shape[0] + 1
            pp_seg = pp_seg.append(
                pd.DataFrame(
                    {
                        "n": pp_seg.shape[0] + 1,
                        "from": fro,
                        "to": to,
                        "d": shortest_path(csr, directed=False, indices=fro)[to],
                    },
                    index=[pp_seg.shape[0] + 1],
                )
            )

    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)
    pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)
    pp_info["seg"] = pp_info["seg"].astype(int).astype(str)

    tips = graph["tips"]
    tips = tips[~np.isin(tips, roots)]

    edges = pp_seg[["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(pp_seg[["from", "to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    root2paths = pd.Series(
        shortest_path(csr, directed=False, indices=root2)[tips.tolist() + [meeting]],
        index=tips.tolist() + [meeting],
    )

    toinvert = root2paths.index[(root2paths <= root2paths[meeting])]

    for toinv in toinvert:
        pathtorev = np.array(img.vs[:]["name"])[
            np.array(img.get_shortest_paths(str(root2), str(toinv)))
        ][0]
        for i in range((len(pathtorev) - 1)):
            segtorev = pp_seg.index[
                pp_seg[["from", "to"]]
                .astype(str)
                .apply(lambda x: all(x.values == pathtorev[[i + 1, i]]), axis=1)
            ]

            pp_seg.loc[segtorev, ["from", "to"]] = pp_seg.loc[segtorev][
                ["to", "from"]
            ].values
            pp_seg["from"] = pp_seg["from"].astype(int)
            pp_seg["to"] = pp_seg["to"].astype(int)

    pptoinvert = np.unique(np.concatenate(g.get_shortest_paths(root2, toinvert)))
    reverted_dist = (
        shortest_path(csr, directed=False, indices=root2)
        + np.abs(
            np.diff(shortest_path(csr, directed=False, indices=roots)[:, meeting])
        )[0]
    )
    pp_info.loc[pptoinvert, "time"] = reverted_dist[pptoinvert]

    graph["pp_info"] = pp_info
    graph["pp_seg"] = pp_seg
    graph["root"] = root
    graph["root2"] = root2
    graph["meeting"] = meeting

    adata.uns["graph"] = graph

    logg.info("root selected", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    " + str(root) + " is the farthest root.\n"
        "    .uns['graph']['root'] farthest root selected.\n"
        "    .uns['graph']['root2'] 2nd root selected.\n"
        "    .uns['graph']['meeting'] meeting point on the tree.\n"
        "    .uns['graph']['pp_info'] for each PP, its distance vs root and segment assignment.\n"
        "    .uns['graph']['pp_seg'] segments network information."
    )

    return adata if copy else None


def getpath(adata, root_milestone, milestones):

    graph = adata.uns["graph"]

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    g = igraph.Graph()
    g.add_vertices(
        np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(str))
    )
    g.add_edges(edges)

    uns_temp = adata.uns.copy()

    if "milestones_colors" in adata.uns:
        mlsc = adata.uns["milestones_colors"].copy()

    dct = dict(
        zip(
            adata.obs.milestones.cat.categories.tolist(),
            np.unique(graph["pp_seg"][["from", "to"]].values.flatten().astype(int)),
        )
    )
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves = list(map(lambda leave: dct[leave], milestones))
    root = dct[root_milestone]

    df = adata.obs.copy()
    wf = warnings.filters.copy()
    warnings.filterwarnings("ignore")
    # for tip in leaves:
    def gatherpath(tip):
        try:
            path = np.array(g.vs[:]["name"])[
                np.array(g.get_shortest_paths(str(root), str(tip)))
            ][0]
            segs = list()
            for i in range(len(path) - 1):
                segs = segs + [
                    np.argwhere(
                        (
                            graph["pp_seg"][["from", "to"]]
                            .astype(str)
                            .apply(lambda x: all(x.values == path[[i, i + 1]]), axis=1)
                        ).to_numpy()
                    )[0][0]
                ]
            segs = graph["pp_seg"].index[segs]
            pth = df.loc[df.seg.astype(int).isin(segs), :].copy(deep=True)
            pth["branch"] = str(root) + "_" + str(tip)
            # warnings.filterwarnings("default")
            warnings.filters = wf
            return pth
        except IndexError:
            pass

    return pd.concat(list(map(gatherpath, leaves)), axis=0)


def round_base_10(x):
    if x < 0:
        return 0
    elif x == 0:
        return 10
    return 10 ** np.ceil(np.log10(x))
