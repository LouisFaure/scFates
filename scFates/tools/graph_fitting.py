from typing import Optional, Union
from typing_extensions import Literal
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
    epg_extend_leaves: bool = False,
    epg_verbose: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
    plot: bool = False,
    basis: Optional[str] = "umap",
    seed: Optional[int] = None,
    copy: bool = False,
    **kwargs,
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
    epg_lambda
        Parameter for ElPiGraph, coefficient of ‘stretching’ elasticity [Albergante20]_.
    epg_mu
        Parameter for ElPiGraph, coefficient of ‘bending’ elasticity [Albergante20]_.
    epg_trimmingradius
        Parameter for ElPiGraph, trimming radius for MSE-based data approximation term [Albergante20]_.
    epg_extend_leaves
        Parameter for ElPiGraph, calls :func:`elpigraph.ExtendLeaves` after graph learning.
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
    **kwargs
        Arguments passsed to :func:`elpigraph.computeElasticPrincipalCurve`
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['epg']`
            dictionnary containing information from elastic principal curve
        `.obsm['X_R']`
            soft assignment of cells to principal points
        `.uns['graph']['B']`
            adjacency matrix of the principal points
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
        epg_lambda,
        epg_mu,
        epg_trimmingradius,
        epg_extend_leaves,
        device,
        seed,
        epg_verbose,
        **kwargs,
    )

    if plot:
        plot_graph(adata, basis)

    return adata if copy else None


def tree(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    weight_rep: str = None,
    method: Literal["ppt", "epg"] = "ppt",
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
    epg_extend_leaves: bool = False,
    epg_verbose: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
    plot: bool = False,
    basis: Optional[str] = "umap",
    seed: Optional[int] = None,
    copy: bool = False,
    **kwargs,
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
    weight_rep
        If `ppt`, use a weight matrix for learning the tree.
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
    epg_extend_leaves
        Parameter for ElPiGraph, calls :func:`elpigraph.ExtendLeaves` after graph learning.
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
    **kwargs
        Arguments passsed to :func:`elpigraph.computeElasticPrincipalTree`
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['ppt']`
            dictionnary containing information from simpelppt tree if method='ppt'
        `.uns['epg']`
            dictionnary containing information from elastic principal tree if method='epg'
        `.obsm['R']`
            soft assignment of cells to principal points
        `.uns['graph']['B']`
            adjacency matrix of the principal points
        `.uns['graph']['F']`
            coordinates of principal points in representation space
    """

    adata = adata.copy() if copy else adata

    X, use_rep = get_data(adata, use_rep, ndims_rep)
    X = X.values

    W = get_data(adata, weight_rep, ndims_rep)[0] if weight_rep is not None else None

    if Nodes is None:
        if adata.shape[0] * 2 > 2000:
            Nodes = 2000
        else:
            Nodes = int(adata.shape[0] / 2)

    if method == "ppt":
        simpleppt.settings.verbosity = settings.verbosity
        ppt = simpleppt.ppt(
            X,
            W,
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
            progress=settings.verbosity > 1,
        )

        ppt = vars(ppt)

        graph = {
            "B": ppt["B"],
            "F": ppt["F"],
            "tips": ppt["tips"],
            "forks": ppt["forks"],
            "metrics": ppt["metric"],
            "use_rep": use_rep,
            "ndims_rep": ndims_rep,
            "method": "ppt",
        }

        adata.uns["graph"] = graph
        adata.uns["ppt"] = ppt
        adata.obsm["X_R"] = ppt["R"]

    elif method == "epg":
        logg.info(
            "inferring a principal tree",
            reset=True,
            end=" " if settings.verbosity > 2 else "\n",
        )
        graph, R, EPG = tree_epg(
            X,
            Nodes,
            use_rep,
            ndims_rep,
            epg_lambda,
            epg_mu,
            epg_trimmingradius,
            epg_extend_leaves,
            device,
            seed,
            epg_verbose,
            **kwargs,
        )
        adata.uns["graph"] = graph
        adata.uns["epg"] = EPG
        adata.obsm["X_R"] = R

    if plot:
        plot_graph(adata, basis)

    logg.hint(
        "added \n"
        "    .uns['" + method + "'], dictionnary containing inferred tree.\n"
        "    .obsm['X_R'] soft assignment of cells to principal points.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['F'] coordinates of principal points in representation space."
    )

    return adata if copy else None


def circle(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    epg_lambda: Optional[Union[float, int]] = 0.01,
    epg_mu: Optional[Union[float, int]] = 0.1,
    epg_trimmingradius: Optional = np.inf,
    epg_verbose: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
    plot: bool = False,
    basis: Optional[str] = "umap",
    seed: Optional[int] = None,
    copy: bool = False,
    **kwargs,
):
    """\
    Generate a principal circle.

    Learn a circled representation on any space, composed of nodes, approximating the
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
    epg_lambda
        Parameter for ElPiGraph, coefficient of ‘stretching’ elasticity [Albergante20]_.
    epg_mu
        Parameter for ElPiGraph, coefficient of ‘bending’ elasticity [Albergante20]_.
    epg_trimmingradius
        Parameter for ElPiGraph, trimming radius for MSE-based data approximation term [Albergante20]_.
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
    **kwargs
        Arguments passsed to :func:`elpigraph.computeElasticPrincipalCircle`
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.uns['epg']`
            dictionnary containing information from elastic principal curve
        `.obsm['X_R']`
            soft assignment of cells to principal points
        `.uns['graph']['B']`
            adjacency matrix of the principal points
        `.uns['graph']['F']`
            coordinates of principal points in representation space
    """

    logg.info(
        "inferring a principal circle",
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
    circle_epg(
        adata,
        Nodes,
        use_rep,
        ndims_rep,
        epg_lambda,
        epg_mu,
        epg_trimmingradius,
        device,
        seed,
        epg_verbose,
    )

    if plot:
        plot_graph(adata, basis)

    return adata if copy else None


def tree_epg(
    X,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    extend_leaves: bool = False,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
):

    import elpigraph

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
        try:
            import cupy as cp
            from cuml.metrics import pairwise_distances
            from .utils import cor_mat_gpu
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )

    EPG = elpigraph.computeElasticPrincipalTree(
        X,
        NumNodes=Nodes,
        Do_PCA=False,
        Lambda=lam,
        Mu=mu,
        TrimmingRadius=trimmingradius,
        GPU=device == "gpu",
        verbose=verbose,
        **kwargs,
    )[0]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")

    graph, R, EPG = epg_to_graph(
        EPG, X, Nodes, use_rep, ndims_rep, extend_leaves, device
    )

    EPG["Edges"] = list(EPG["Edges"])[0]

    return graph, R, EPG


def curve_epg(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    extend_leaves: bool = False,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
):
    import elpigraph

    X, use_rep = get_data(adata, use_rep, ndims_rep)
    X = X.values

    if seed is not None:
        np.random.seed(seed)

    if device == "gpu":
        try:
            import cupy as cp
            from .utils import cor_mat_gpu, norm_R_cpu
            from cuml.metrics import pairwise_distances
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )

    EPG = elpigraph.computeElasticPrincipalCurve(
        X,
        NumNodes=Nodes,
        Do_PCA=False,
        Lambda=lam,
        Mu=mu,
        TrimmingRadius=trimmingradius,
        GPU=device == "gpu",
        verbose=verbose,
        **kwargs,
    )[0]

    graph, R, EPG = epg_to_graph(
        EPG, X, Nodes, use_rep, ndims_rep, extend_leaves, device
    )

    EPG["Edges"] = list(EPG["Edges"])

    adata.uns["graph"] = graph
    adata.uns["epg"] = EPG
    adata.obsm["X_R"] = R

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['epg'] dictionnary containing inferred elastic curve generated from elpigraph.\n"
        "    .obsm['X_R'] soft assignment of cells to principal points.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['F'], coordinates of principal points in representation space."
    )

    return adata


def circle_epg(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
):
    import elpigraph

    X, use_rep = get_data(adata, use_rep, ndims_rep)
    X = X.values

    if seed is not None:
        np.random.seed(seed)

    if device == "gpu":
        try:
            import cupy as cp
            from .utils import cor_mat_gpu
            from cuml.metrics import pairwise_distances
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )

    EPG = elpigraph.computeElasticPrincipalCircle(
        X,
        NumNodes=Nodes,
        Do_PCA=False,
        Lambda=lam,
        Mu=mu,
        TrimmingRadius=trimmingradius,
        GPU=device == "gpu",
        verbose=verbose,
    )[0]

    graph, R, EPG = epg_to_graph(EPG, X, Nodes, use_rep, ndims_rep, False, device)

    EPG["Edges"] = list(EPG["Edges"])[0]

    adata.uns["graph"] = graph
    adata.uns["epg"] = EPG
    adata.obsm["X_R"] = R

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .uns['epg'] dictionnary containing inferred elastic circle generated from elpigraph.\n"
        "    .obsm['X_R'] soft assignment of cells to principal points.\n"
        "    .uns['graph']['B'] adjacency matrix of the principal points.\n"
        "    .uns['graph']['F'], coordinates of principal points in representation space."
    )

    return adata


def epg_to_graph(EPG, X, Nodes, use_rep, ndims_rep, extend_leaves, device):
    import elpigraph
    from .utils import norm_R_cpu

    if device == "gpu":
        from cuml.metrics import pairwise_distances
    else:
        from sklearn.metrics import pairwise_distances

    if extend_leaves:
        EPG = elpigraph.ExtendLeaves(X, EPG, Mode="WeightedCentroid")
        Nodes = EPG["NodePositions"].shape[0]

    F = EPG["NodePositions"].T

    # assign to edge and obtain R
    R = pairwise_distances(X, EPG["NodePositions"])
    idx_nodes = np.arange(Nodes)
    elpigraph.utils.getProjection(X, EPG)
    mask = [
        np.isin(idx_nodes, EPG["Edges"][0][EPG["projection"]["edge_id"][i]]) * 1
        for i in range(X.shape[0])
    ]

    R = R * np.vstack(mask)
    Rsum = R.sum(axis=1)
    norm_R_cpu(R, Rsum)
    R = np.abs(R - 1)
    R[R == 1] = 0

    # obtain B
    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(EPG["Edges"][0].flatten().astype(int)))
    g.add_edges(pd.DataFrame(EPG["Edges"][0]).astype(int).apply(tuple, axis=1).values)

    B = np.asarray(g.get_adjacency().data)

    emptynodes = np.argwhere(R.max(axis=0) == 0).ravel()

    if len(emptynodes) > 0:
        logg.info("    there are %d non assigned nodes" % (len(emptynodes)))

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    graph = {
        "B": B,
        "F": F,
        "tips": tips,
        "forks": forks,
        "metrics": "euclidean",
        "use_rep": use_rep,
        "ndims_rep": ndims_rep,
        "method": "epg",
    }

    return graph, R, EPG


def explore_sigma(
    adata,
    Nodes,
    use_rep=None,
    ndims_rep=None,
    sigmas=[1000, 100, 10, 1, 0.1, 0.01],
    nsteps=1,
    metric="euclidean",
    seed=None,
    plot=False,
    second_round=False,
    **kwargs,
):
    """\
    Explore varisou sigma parameters for best tree fitting. Given that high sigma tend
    to collapse the principal points into the middle of the whole data (meaning taking
    in account all the datapoints regardless their locality), it is possible to explore
    which sigma is best by detecting at which level the tree stops collapsing.

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
    sigmas
        Range of sigma parameters to test.
    device
        Run method on either `cpu` or on `gpu`.
    nsteps
        Number of SimplePPT iteration, usually 1 is enough.
    metric
        Distance metric to use.
    seed
        A numpy random seed.
    plot
        Plot the resulting tree.
    second_round
        Perform a second exploration, on a restricted sigma parameters based on the first estimated sigma.
    **kwargs
        Arguments passsed to :func:`elpigraph.computeElasticPrincipalCircle`
    Returns
    -------
    sigma : float
        suggested sigma value
    """
    import copy
    from sklearn.metrics import pairwise_distances

    X, use_rep = get_data(adata, use_rep, ndims_rep)

    mindist = list()
    verb = copy.deepcopy(settings.verbosity)
    settings.verbosity = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sigma in sigmas:
            tree(
                adata,
                Nodes=Nodes,
                use_rep=use_rep,
                method="ppt",
                ppt_nsteps=nsteps,
                ppt_sigma=sigma,
                ppt_metric=metric,
                seed=seed,
                **kwargs,
            )
            mindist.append(
                pairwise_distances(adata.uns["graph"]["F"].T, X, metric=metric)
                .min(axis=0)
                .mean()
            )

    def point_on_line(a, b, p):
        ap = p - a
        ab = b - a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    a = np.array([0, mindist[0]])
    b = np.array([len(sigmas) - 1, mindist[len(sigmas) - 1]])
    projected = list()

    for i in range(len(sigmas)):
        p = np.array([i, mindist[i]])

        projected.append(point_on_line(a, b, p))

    curve = np.vstack([np.arange(len(sigmas)), mindist]).T
    proj = np.vstack(projected)
    dists = np.array(
        [np.linalg.norm(curve[i, :] - proj[i, :]) for i in range(len(sigmas))]
    )
    dists[(proj[:, 0] - curve[:, 0]) < 0] = 0
    selected = np.argmax(dists)

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        ax1.plot(range(len(sigmas)), mindist, color="k")
        ax1.set_xticks(range(len(sigmas)))
        ax1.set_xticklabels(sigmas)
        ax1.scatter(selected, mindist[selected], color="r", zorder=100)
        ax1.set_xlabel("sigma parameter")
        ax1.set_ylabel(f"mean minimum {metric} distance")
        ax1.grid(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree(
                adata,
                Nodes=Nodes,
                use_rep=use_rep,
                method="ppt",
                ppt_nsteps=3,
                ppt_sigma=sigmas[selected],
                ppt_lambda=1,
                seed=seed,
            )
        plot_graph(
            adata,
            show=False,
            size_nodes=0.1,
            title=f"iteration {nsteps}",
            ax=ax2,
            forks=False,
            tips=False,
        )

    settings.verbosity = verb
    sigma = sigmas[selected]
    if second_round:
        sigmas = np.arange(sigma / 5, sigma, sigma / 5)[::-1]
        sigma = explore_sigma(
            adata,
            Nodes,
            use_rep,
            ndims_rep,
            sigmas,
            nsteps,
            metric,
            seed,
            plot,
            second_round=False,
            **kwargs,
        )
    return sigma


def get_data(adata, use_rep, ndims_rep):

    if use_rep not in adata.obsm.keys() and f"X_{use_rep}" in adata.obsm.keys():
        use_rep = f"X_{use_rep}"

    if (
        (use_rep not in adata.layers.keys())
        & (use_rep not in adata.obsm.keys())
        & (use_rep != "X")
    ):
        use_rep = "X" if adata.n_vars < 50 or ndims_rep is None else "X_pca"
        ndims_rep = None if use_rep == "X" else ndims_rep

    if use_rep == "X":
        ndims_rep = None
        if sparse.issparse(adata.X):
            X = DataFrame(adata.X.A, index=adata.obs_names, columns=adata.var_names)
        else:
            X = DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    elif use_rep in adata.layers.keys():
        if sparse.issparse(adata.layers[use_rep]):
            X = DataFrame(
                adata.layers[use_rep].A, index=adata.obs_names, columns=adata.var_names
            )
        else:
            X = DataFrame(
                adata.layers[use_rep], index=adata.obs_names, columns=adata.var_names
            )
    elif use_rep in adata.obsm.keys():
        X = DataFrame(adata.obsm[use_rep], index=adata.obs_names)

    if ndims_rep is not None:
        X = X.iloc[:, :ndims_rep]

    return X, use_rep
