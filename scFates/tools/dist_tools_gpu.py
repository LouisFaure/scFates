import cupy as cp
def euclidean_mat_gpu(A,B):
    x=cp.repeat((A**2).sum(axis=0),B.shape[1]).reshape(A.shape[1],B.shape[1])
    y=cp.repeat((B**2).sum(axis=0),A.shape[1]).reshape(B.shape[1],A.shape[1]).T
    res=cp.sqrt(x + y - 2*(B.T.dot(A)).T)
    res[cp.isnan(res)]=0
    return res

def cor_mat_gpu(A,B):
    A1 = (A-cp.mean(A,axis=0))
    B1 = (B-cp.mean(B,axis=0))
    res = (B1.T.dot(A1)).T/cp.sqrt((A1**2).sum(axis=0).reshape(A1.shape[1],1) @ (B1**2).sum(axis=0).reshape(1,B1.shape[1]))
    return res.T