from numba import cuda, njit, prange
import math
from tqdm import tqdm
from joblib import Parallel
from scipy import sparse
import numpy as np
import shutil
import sys, os
from .. import logging as logg


def get_X(adata, cells, genes, layer, togenelist=False):
    if layer is None:
        if sparse.issparse(adata.X):
            X = adata[cells, genes].X.toarray()
        else:
            X = adata[cells, genes].X
    else:
        if sparse.issparse(adata.layers[layer]):
            X = adata[cells, genes].layers[layer].toarray()
        else:
            X = adata[cells, genes].layers[layer]

    if togenelist:
        return X.T.tolist()
    else:
        return X


class ProgressParallel(Parallel):
    def __init__(
        self, use_tqdm=True, total=None, file=None, desc=None, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._file = file
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            desc=self._desc,
            file=self._file,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@cuda.jit
def process_R_gpu(R, sigma):
    x, y = cuda.grid(2)
    if x < R.shape[0] and y < R.shape[1]:
        R[x, y] = math.exp(-R[x, y] / sigma)


@cuda.jit
def norm_R_gpu(R, Rsum):
    x, y = cuda.grid(2)
    if x < R.shape[0] and y < R.shape[1]:
        R[x, y] = R[x, y] / Rsum[x]
        if math.isnan(R[x, y]):
            R[x, y] = 0


@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


@njit(parallel=True)
def process_R_cpu(R, sigma):
    Rshape = R.shape
    R = R.ravel()
    for x in prange(len(R)):
        R[x] = math.exp(-R[x] / sigma)
    R.reshape(Rshape)


@njit(parallel=True)
def norm_R_cpu(R, Rsum):
    for x in prange(len(Rsum)):
        for y in range(R.shape[1]):
            R[x, y] = R[x, y] / Rsum[x]
            if math.isnan(R[x, y]):
                R[x, y] = 0


def cor_mat_cpu(A, B):
    import numpy as np

    A1 = A - A.mean(axis=0)
    B1 = B - B.mean(axis=0)
    res = (B1.T.dot(A1)).T / np.sqrt(
        (A1 ** 2).sum(axis=0).reshape(A1.shape[1], 1)
        @ (B1 ** 2).sum(axis=0).reshape(1, B1.shape[1])
    )
    return res.T


def cor_mat_gpu(A, B):
    import cupy as cp

    A1 = A - A.mean(axis=0)
    B1 = B - B.mean(axis=0)
    res = (B1.T.dot(A1)).T / cp.sqrt(
        (A1 ** 2).sum(axis=0).reshape(A1.shape[1], 1)
        @ (B1 ** 2).sum(axis=0).reshape(1, B1.shape[1])
    )
    return res.T


def mst_gpu(d):
    import numpy as np
    import cugraph
    import cudf
    import cupy as cp
    from cupyx.scipy.sparse.csr import csr_matrix as csr_cupy
    from cupyx.scipy.sparse.coo import coo_matrix
    from cugraph.tree.minimum_spanning_tree_wrapper import mst_double, mst_float
    import scipy

    csr_gpu = csr_cupy(d)
    offsets = cudf.Series(csr_gpu.indptr)
    indices = cudf.Series(csr_gpu.indices)

    num_verts = csr_gpu.shape[0]
    num_edges = len(csr_gpu.indices)
    weights = cudf.Series(csr_gpu.data)

    if weights.dtype == np.float32:
        mst = mst_float(num_verts, num_edges, offsets, indices, weights)

    else:
        mst = mst_double(num_verts, num_edges, offsets, indices, weights)

    mst = csr_cupy(
        coo_matrix(
            (mst.weight.values, (mst.src.values, mst.dst.values)),
            shape=(num_verts, num_verts),
        )
    ).get()
    return csr_cupy(scipy.sparse.triu(mst))


def getpath(g, root, tips, tip, tree, df):
    import warnings
    import numpy as np

    wf = warnings.filters.copy()
    warnings.filterwarnings("ignore")
    try:
        path = np.array(g.vs[:]["name"])[
            np.array(g.get_shortest_paths(str(root), str(tip)))
        ][0]
        segs = list()
        for i in range(len(path) - 1):
            segs = segs + [
                np.argwhere(
                    (
                        tree["pp_seg"][["from", "to"]]
                        .astype(str)
                        .apply(lambda x: all(x.values == path[[i, i + 1]]), axis=1)
                    ).to_numpy()
                )[0][0]
            ]
        segs = tree["pp_seg"].index[segs]
        pth = df.loc[df.seg.astype(int).isin(segs), :].copy(deep=True)
        pth["branch"] = str(root) + "_" + str(tip)
        warnings.filters = wf
        return pth
    except IndexError:
        pass


def palantir_on_seg(adata, seg, ms_data):
    import palantir

    adata_sub = adata[
        adata.obs.seg == seg,
    ]
    pr = palantir.core.run_palantir(
        ms_data.loc[adata_sub.obs_names, :], adata_sub.obs.t.idxmin()
    )
    return pr.pseudotime


@njit(parallel=True)
def get_SE(MSE, x, se):
    N = len(x)
    xmean = x.mean()
    xxmean = np.sum((x - xmean) ** 2)
    for i in range(N):
        se[i] = math.sqrt(MSE) * math.sqrt(1 + 1 / N + (x[i] - xmean) ** 2 / xxmean)


def bh_adjust(x, log=False):
    x.sort_values(ascending=True)
    if log:
        q = x.sort_values(ascending=True) + np.log(len(x) / (np.arange(len(x)) + 1))
    else:
        q = x.sort_values(ascending=True) * len(x) / (np.arange(len(x)) + 1)
    return (q.reindex(index=q.index[::-1]).cummin())[x.index]


def importeR(task, module="mgcv"):
    try:
        from rpy2.robjects import pandas2ri, Formula
        from rpy2.robjects.packages import PackageNotInstalledError, importr
        import rpy2.rinterface

        pandas2ri.activate()
        Rpy2 = True
    except ModuleNotFoundError as e:
        Rpy2 = (
            "rpy2 installation is necessary for "
            + task
            + '. \
            \nPlease use "pip3 install rpy2" to install rpy2'
        )
        Formula = False

    whichR = shutil.which("R")

    if not ((whichR is not None) | (os.path.isfile(sys.exec_prefix+"/lib/R/bin/R"))):
        R = (
            "R installation is necessary for "
            + task
            + ". \
            \nPlease install R and try again"
        )
    else:
        R = True

    try:
        rstats = importr("stats")
    except Exception as e:
        rstats = (
            "R installation is necessary for "
            + task
            + ". \
            \nPlease install R and try again"
        )

    try:
        rmodule = importr(module)
    except Exception as e:
        rmodule = (
            f'R package "{module}" is necessary for {task}'
            + "\nPlease install it and try again"
        )

    return Rpy2, R, rstats, rmodule, Formula
