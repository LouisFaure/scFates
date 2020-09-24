import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Union, Optional, Tuple, Collection, Sequence, Iterable

import numpy as np
import pandas as pd
from functools import partial
from anndata import AnnData
import shutil
import sys
import copy
import igraph
import warnings
from functools import reduce

from joblib import delayed, Parallel
from tqdm import tqdm
from scipy import sparse

from .. import logging as logg
from .. import settings


try:
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr
    import rpy2.rinterface
    pandas2ri.activate()  
except Exception as e:
    warnings.warn(
        'Cannot compute gene expression trends without installing rpy2. \
        \nPlease use "pip3 install rpy2" to install rpy2'
    )
    print(e.__doc__)
    print(e.message)

        
if not shutil.which("R"):
    warnings.warn(
        "R installation is necessary for computing gene expression trends. \
        \nPlease install R and try again"
    )

try:
    rstats = importr("stats")    
except Exception as e:
    warnings.warn(
        "R installation is necessary for computing gene expression trends. \
        \nPlease install R and try again"
    )
    print(e.__doc__)
    print(e.message)
    
try:
    rmgcv = importr("mgcv")  
except Exception as e:
    warnings.warn(
        'R package "mgcv" is necessary for computing gene expression trends. \
        \nPlease install gam from https://cran.r-project.org/web/packages/gam/ and try again'
    )
    print(e.__doc__)
    print(e.message)


def fit(
    adata: AnnData,
    root = None,
    leaves = None,
    n_map: int = 1,
    n_jobs: int = 1,
    spline_df: int = 5,
    fdr_cut: float = 1e-4,
    A_cut: int = 1,
    st_cut: float = 0.8,
    gamma: float = 1.5,
    copy: bool = False,
    layer: Optional[str] = None):
    
    """\
    Model feature expression levels as a function of tree positions.
    
    The models are fit using *mgcv* R package. Note that since adata can currently only keep the 
    same dimensions for each of its layers, the dataset is subsetted to keep only significant
    feratures.
    

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        adata layer to use for the fitting.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    gamma
        stringency of penalty.
    root
        restrain the fit to a subset of the tree (in combination with leaves).
    leaves
        restrain the fit to a subset of the tree (in combination with root).
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:
        
        `.layers['fitted']` 
            fitted features on the tree for all mappings.
    
    """
    
    adata = adata.copy() if copy else adata
    
    if "signi" not in adata.var.columns:
        raise ValueError(
            "You need to run `tl.test_association` before fitting features."
        )
    
    genes = adata.var_names[adata.var.signi]
    
    
    tree = adata.uns["tree"]
    tips = tree["tips"]
    
    if leaves is not None:
        mlsc = adata.uns["milestones_colors"].copy()
        mlsc_temp = mlsc.copy()
        dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                       np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
        keys = np.array(list(dct.keys()))
        vals = np.array(list(dct.values()))

        leaves=list(map(lambda leave: dct[leave],leaves))
        root=dct[root]
    else:
        mlsc_temp=None
        
    if root is None:
        root = tree["root"]
        tips = tips[~np.isin(tips,root)] 
    root2=None
    if "root2" in tree:
        root2 = tree["root2"]
        tips = tips[~np.isin(tips,tree["root2"])] 
    
    if leaves is not None:
        tips=leaves
    
    logg.info("fit features associated with the tree", reset=True, end="\n")
       
    stat_assoc=list()
    
    for m in range(n_map):
        if "t_old" in adata.obs.columns:
            df=adata.obs.copy()
        else:
            df=tree["pseudotime_list"][str(m)]
        edges=tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
        img = igraph.Graph()
        img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
        img.add_edges(edges)
        
        temp = pd.concat(list(map(lambda tip: getpath(img,root,tips,tip,tree,df), tips)),axis=0)
        if root2 is not None:
            temp = pd.concat([temp,pd.concat(list(map(lambda tip: getpath(img,root2,tips,tip,tree,df), tips)),axis=0)])
        temp.drop(['edge','seg'],axis=1,inplace=True)
        temp.columns = ['t', 'branch']
        temp["gamma"] = gamma
        #temp = temp[~temp.index.duplicated(keep='first')]
        if layer is None:
            if sparse.issparse(adata.X):
                Xgenes = adata[temp.index,genes].X.A.T.tolist()
            else:
                Xgenes = adata[temp.index,genes].X.T.tolist()
        else:
            if sparse.issparse(adata.layers[layer]):
                Xgenes = adata[temp.index,genes].layers[layer].A.T.tolist()
            else:
                Xgenes = adata[temp.index,genes].layers[layer].T.tolist()
        
        data = list(zip([temp]*len(Xgenes),Xgenes))
        
        stat = Parallel(n_jobs=n_jobs)(
            delayed(gt_fun)(
                data[d]
            )
            for d in tqdm(range(len(data)),file=sys.stdout,desc="    mapping "+str(m))
        )
                        
        stat_assoc = stat_assoc + [stat] 
    
    
    for i in range(len(stat_assoc)):
        stat_assoc[i]=pd.concat(stat_assoc[i],axis=1)
        stat_assoc[i].columns=adata.var_names[adata.var.signi]
        
    
    
    names = np.arange(len(stat_assoc)).astype(str).tolist()
    dictionary = dict(zip(names, stat_assoc))
    
    if n_map==1:
        fitted = dictionary["0"]
    else:
        dfs = list(dictionary.values)
        fitted = reduce(lambda x, y: x.add(y, fill_value=0), dfs)/n_map
    
    adata._inplace_subset_obs(np.unique(dictionary["0"].index))
    adata._inplace_subset_var(genes)
    
    adata.layers["fitted"] = fitted.loc[adata.obs_names,:]
    
    if mlsc_temp is not None:
        adata.uns["milestones_colors"]=mlsc_temp
    
    logg.info("    finished (adata subsetted to keep only fitted features!)", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'fitted', fitted features on the tree for all mappings (adata.layers)"
    )
    
    
    return adata if copy else None

def getpath(g,root,tips,tip,tree,df):
    wf=warnings.filters.copy()
    warnings.filterwarnings("ignore")
    try:
        path=np.array(g.vs[:]["name"])[np.array(g.get_shortest_paths(str(root),str(tip)))][0]
        segs = list()
        for i in range(len(path)-1):
            segs= segs + [np.argwhere((tree["pp_seg"][["from","to"]].astype(str).apply(lambda x: 
                                                                                    all(x.values == path[[i,i+1]]),axis=1)).to_numpy())[0][0]]
        segs=tree["pp_seg"].index[segs]
        pth=df.loc[df.seg.astype(int).isin(segs),["t","seg","edge"]].copy()
        pth["branch"]=str(root)+"_"+str(tip)
        warnings.filters=wf
        return(pth)
    except IndexError:
        pass

def gt_fun(data):     
    sdf = data[0]
    sdf["exp"] = data[1]
    gamma=sdf["gamma"][0]
    
    global rmgcv
    global rstats
    
    def gamfit(b):
        m = rmgcv.gam(Formula("exp~s(t,bs='ts')"),data=sdf.loc[sdf["branch"]==b,:],gamma=gamma)
        return pd.Series(rmgcv.predict_gam(m),index=sdf.loc[sdf["branch"]==b,:].index)

    mdl=list(map(gamfit,sdf.branch.unique()))
          
    return pd.concat(mdl,axis=1,sort=False).apply(np.nanmean,axis=1)