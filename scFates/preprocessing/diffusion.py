from anndata import AnnData
from typing import Optional
from typing_extensions import Literal
from scipy.sparse import csr_matrix, find, issparse
import pandas as pd
import numpy as np

from .. import logging as logg
from .. import settings


def diffusion(
    adata: AnnData,
    n_components: int = 10,
    knn: int = 30,
    alpha: float = 0,
    multiscale: bool = True,
    n_eigs: int = None,
    device: Literal["cpu", "gpu"] = "cpu",
    n_pcs: int = 50,
    save_uns: bool = False,
    copy: bool = False,
):
    """\
    Wrapper to generate diffusion maps using Palantir.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of diffusion components.
    knn
        Number of nearest neighbors for graph construction.
    alpha
        Normalization parameter for the diffusion operator.
    multiscale
        Whether to get mutliscale diffusion space
        (calls palantir.utils.determine_multiscale_space).
    n_eigs
        if multiscale is True, how much components to retain.
    device
        Run method on either `cpu` or on `gpu`.
    do_PCA
        Whether to perform PCA or not.
    n_pcs
        Number of PC components.
    seed
        Get reproducible results for the GPU implementation.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns AnnData, else it update field to `adata`:

        `.obsm['X_diffusion']`
            if `multiscale = False`, diffusion space.
        `.obsm['X_multiscale_diffusion']`
            if `multiscale = True`, multiscale diffusion space.
        `.uns['diffusion']`
            dict containing results from Palantir.
    """

    logg.info("Running Diffusion maps ", reset=True)

    data_df = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)

    if device == "cpu":
        from palantir.utils import run_diffusion_maps

        res = run_diffusion_maps(
            data_df, n_components=n_components, knn=knn, alpha=alpha
        )
    # code converted in GPU, not reproducible!
    elif device == "gpu":
        logg.warn(
            "GPU implementation uses eigsh from cupy.sparse, which is not currently reproducible and can give unstable results!"
        )
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
            from cupyx.scipy.sparse.linalg import eigsh
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )

        # Determine the kernel
        N = data_df.shape[0]
        if not issparse(data_df):
            from cuml.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=knn, metric="euclidean")
            X_contiguous = np.ascontiguousarray(data_df.values)
            nn.fit(X_contiguous)

            kNN = nn.kneighbors_graph(X_contiguous, mode="distance")
            kNN.setdiag(0)
            kNN.eliminate_zeros()

            # Adaptive k
            adaptive_k = int(np.floor(knn / 3))
            adaptive_std = np.zeros(N)

            for i in np.arange(len(adaptive_std)):
                adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
                    adaptive_k - 1
                ]

            # Kernel
            x, y, dists = find(kNN)

            # X, y specific stds
            dists = dists / adaptive_std[x]
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

            # Diffusion components
            kernel = W + W.T
        else:
            kernel = data_df

        # Markov
        D = np.ravel(kernel.sum(axis=1))

        if alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-alpha)
            mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
            kernel = mat.dot(kernel).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]
        kernel = csr_matrix_gpu(kernel)
        D = csr_matrix_gpu((cp.array(D), (cp.arange(N), cp.arange(N))), shape=(N, N))
        T = D.dot(kernel)
        # Eigen value dcomposition
        D, V = eigsh(T, n_components, tol=1e-4, maxiter=1000)
        D, V = D.get(), V.get()

        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Normalize
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create are results dictionary
        res = {"T": T.get(), "EigenVectors": V, "EigenValues": D}
        res["EigenVectors"] = pd.DataFrame(res["EigenVectors"])
        if not issparse(data_df):
            res["EigenVectors"].index = data_df.index
        res["EigenValues"] = pd.Series(res["EigenValues"])
        res["kernel"] = kernel.get()

    if multiscale:
        logg.info("    determining multiscale diffusion space")
        from palantir.utils import determine_multiscale_space

        adata.obsm["X_diffusion_multiscale"] = determine_multiscale_space(
            res, n_eigs=n_eigs
        ).values
        logstr = "    .obsm['X_diffusion_multiscale'], multiscale diffusion space."
    else:
        adata.obsm["X_diffusion"] = res["EigenVectors"].iloc[:, 1:].values
        logstr = "    .obsm['X_diffusion'], diffusion space."

    if save_uns:
        adata.uns["diffusion"] = res
        uns_str = "\n    .uns['diffusion'] dict containing diffusion maps results."
    else:
        uns_str = ""

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" + logstr + uns_str)
