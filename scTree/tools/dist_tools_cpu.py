import numpy as np

def euclidean_mat_cpu(A,B):
    np.seterr(invalid='ignore')
    x=np.repeat((A**2).sum(axis=0),B.shape[1]).reshape(A.shape[1],B.shape[1])
    y=np.repeat((B**2).sum(axis=0),A.shape[1]).reshape(B.shape[1],A.shape[1]).T
    res=np.sqrt(x + y - 2*(B.T.dot(A)).T)
    res[np.isnan(res)]=0
    return res

def cor_mat_cpu(A,B):
    A1 = (A-np.mean(A,axis=0))
    B1 = (B-np.mean(B,axis=0))
    res = (B1.T.dot(A1)).T/np.sqrt((A1**2).sum(axis=0).reshape(A1.shape[1],1) @ (B1**2).sum(axis=0).reshape(1,B1.shape[1]))
    return res.T
