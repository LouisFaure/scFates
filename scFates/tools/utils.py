from numba import cuda, njit, prange
import math


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

    A1 = A - np.mean(A, axis=0)
    B1 = B - np.mean(B, axis=0)
    res = (B1.T.dot(A1)).T / np.sqrt(
        (A1 ** 2).sum(axis=0).reshape(A1.shape[1], 1)
        @ (B1 ** 2).sum(axis=0).reshape(1, B1.shape[1])
    )
    return res.T


def cor_mat_gpu(A, B):
    import cupy as cp

    A1 = A - cp.mean(A, axis=0)
    B1 = B - cp.mean(B, axis=0)
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
