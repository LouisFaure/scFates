
def cor_mat_cpu(A,B):
    import numpy as np
    A1 = (A-np.mean(A,axis=0))
    B1 = (B-np.mean(B,axis=0))
    res = (B1.T.dot(A1)).T/np.sqrt((A1**2).sum(axis=0).reshape(A1.shape[1],1) @ (B1**2).sum(axis=0).reshape(1,B1.shape[1]))
    return res.T

def cor_mat_gpu(A,B):
    import cupy as cp
    A1 = (A-cp.mean(A,axis=0))
    B1 = (B-cp.mean(B,axis=0))
    res = (B1.T.dot(A1)).T/cp.sqrt((A1**2).sum(axis=0).reshape(A1.shape[1],1) @ (B1**2).sum(axis=0).reshape(1,B1.shape[1]))
    return res.T


def getpath(g,root,tips,tip,tree,df):
    import warnings
    import numpy as np
    wf=warnings.filters.copy()
    warnings.filterwarnings("ignore")
    try:
        path=np.array(g.vs[:]["name"])[np.array(g.get_shortest_paths(str(root),str(tip)))][0]
        segs = list()
        for i in range(len(path)-1):
            segs= segs + [np.argwhere((tree["pp_seg"][["from","to"]].astype(str).apply(lambda x: 
                                                                                    all(x.values == path[[i,i+1]]),axis=1)).to_numpy())[0][0]]
        segs=tree["pp_seg"].index[segs]
        pth=df.loc[df.seg.astype(int).isin(segs),:].copy(deep=True)
        pth["branch"]=str(root)+"_"+str(tip)
        warnings.filters=wf
        return(pth)
    except IndexError:
        pass
