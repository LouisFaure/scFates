import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from functools import partial
from anndata import AnnData
import shutil
import sys
import copy
from statsmodels.stats.multitest import multipletests

from joblib import delayed, Parallel
from tqdm import tqdm
from scipy import sparse

from .. import logging as logg
from .. import settings
from ..plot.test_association import test_association as plot_test_association


try:
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr
    import rpy2.rinterface
    pandas2ri.activate()
    
except ImportError:
    raise RuntimeError(
        'Cannot compute gene expression trends without installing rpy2. \
        \nPlease use "pip3 install rpy2" to install rpy2'
    )

        
if not shutil.which("R"):
    raise RuntimeError(
        "R installation is necessary for computing gene expression trends. \
        \nPlease install R and try again"
    )

try:
    rmgcv = importr("mgcv")
except embedded.RRuntimeError:
    raise RuntimeError(
        'R package "mgcv" is necessary for computing gene expression trends. \
        \nPlease install gam from https://cran.r-project.org/web/packages/gam/ and try again'
    )
rmgcv = importr("mgcv")
rstats = importr("stats")





def test_association(
    adata: AnnData,
    n_map: int = 1,
    n_jobs: int = 1,
    spline_df: int = 5,
    fdr_cut: float = 1e-4,
    A_cut: int = 1,
    st_cut: float = 0.8,
    reapply_filters: bool = False,
    plot: bool = False,
    copy: bool = False):
    
    adata = data.copy() if copy else adata
    
    if "pseudotime_list" not in adata.uns["tree"]:
        raise ValueError(
            "You need to run `tl.pseudotime` before testing for association."
        )
    
    r = adata.uns["tree"]
    
    if reapply_filters & ("stat_assoc_list" in adata.uns["tree"]):
        stat_assoc_l = list(adata.uns["tree"]["stat_assoc_list"].values())
        #stat_assoc_l = list(map(lambda x: pd.DataFrame(x,index=x["features"]),stat_assoc_l))
        adata = apply_filters(adata,stat_assoc_l,fdr_cut,A_cut,st_cut)
        
        logg.info("reapplied filters, "+str(sum(adata.var["signi"]))+ " significant features")
        
        if plot:
            plot_test_association(adata)
            
        return adata if copy else None
    
    genes = adata.var_names
    cells = r["cells_fitted"]
    
    if sparse.issparse(adata.X):
        Xgenes = adata[cells,genes].X.A.T.tolist()
    else:
        Xgenes = adata[cells,genes].X.T.tolist()
     
    logg.info("test features for association with the tree", time=False, end="\n")
       
    stat_assoc_l=list()
    
    for m in range(n_map):
        data = list(zip([r["pseudotime_list"][str(m)]]*len(Xgenes),Xgenes))
        
        stat = Parallel(n_jobs=n_jobs)(
            delayed(gt_fun)(
                data[d]
            )
            for d in tqdm(range(len(data)),file=sys.stdout,desc="    mapping "+str(m))
        )
        stat = pd.DataFrame(stat,index=genes,columns=["p_val","A"])
        stat["fdr"] = multipletests(stat.p_val,method="bonferroni")[1]                  
        stat_assoc_l = stat_assoc_l + [stat]
        
    adata = apply_filters(adata,stat_assoc_l,fdr_cut,A_cut,st_cut)

    logg.info("    found "+str(sum(adata.var["signi"]))+ " significant features",
              time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'p_val' values from statistical test (adata.var)\n"
        "    'fdr' corrected values from multiple testing (adata.var)\n"
        "    'st' proportion of mapping in which feature is significant (adata.var)\n"
        "    'A' amplitue of change of tested feature (adata.var)\n"
        "    'tree/stat_assoc_list', list of fitted features on the tree for all mappings (adata.uns)"
    )
    
    if plot:
        plot_test_association(adata)
    
    return adata if copy else None

def gt_fun(data):     
    sdf = data[0]
    sdf["exp"] = data[1]
    
    global rmgcv
    global rstats
    
    def gamfit(s):
        m = rmgcv.gam(Formula("exp~s(t,k=5)"),data=sdf.loc[sdf["seg"]==s,:])
        return dict({"d":m[5][0],"df":m[42][0],"p":rmgcv.predict_gam(m)})

    mdl=list(map(gamfit,sdf.seg.unique()))
    mdf=pd.concat(list(map(lambda x: pd.DataFrame([x["d"],x["df"]]),mdl)),axis=1).T
    mdf.columns=["d","df"]

    odf = sum(mdf["df"])-mdf.shape[0]
    m0 = rmgcv.gam(Formula("exp~1"),data=sdf)
    if sum(mdf["d"])==0:
        fstat = 0
    else:
        fstat = (m0[5][0] - sum(mdf["d"])) / (m0[42][0]-odf) / (sum(mdf["d"])/odf)

    df_res0 = m0[42][0]
    df_res_odf = df_res0-odf
    pval = rstats.pf(fstat,df_res_odf,odf,lower_tail=False)[0]
    pr = np.concatenate(list(map(lambda x: x["p"],mdl)))
    
    return [pval,max(pr)-min(pr)]


def apply_filters(adata,stat_assoc_l,fdr_cut,A_cut,st_cut):
    n_map = len(stat_assoc_l)
    if n_map>1:
        stat_assoc = pd.DataFrame({"p_val":pd.concat(list(map(lambda x: 
                                                    x["p_val"],stat_assoc_l)),axis=1).median(axis=1),
                      "A":pd.concat(list(map(lambda x: x["A"],stat_assoc_l)),axis=1).median(axis=1),
                      "fdr":pd.concat(list(map(lambda x: x["fdr"],stat_assoc_l)),axis=1).median(axis=1),
                     "st": pd.concat(list(map(lambda x: (x.fdr<fdr_cut) & (x.A>A_cut),
                                              stat_assoc_l)),axis=1).sum(axis=1)/n_map})
    else:
        stat_assoc=stat_assoc_l[0]
        stat_assoc["st"] = ((stat_assoc.fdr<fdr_cut) & (stat_assoc.A>A_cut))*1
     
    
    # saving results 
    stat_assoc["signi"]=stat_assoc["st"]>st_cut   
      
    if set(stat_assoc.columns.tolist()).issubset(adata.var.columns):
        adata.var[stat_assoc.columns] = stat_assoc
    else:
        adata.var = pd.concat([adata.var,stat_assoc],axis=1)
    
    # save all tests for each mapping
    names = np.arange(len(stat_assoc_l)).astype(str).tolist()
    #todict=list(map(lambda x: x.to_dict(),stat_assoc_l))
    
    #todict=list(map(lambda x: dict(zip(["features"]+x.columns.tolist(),
    #                                   [x.index.tolist()]+x.to_numpy().T.tolist())),
    #                stat_assoc_l))
    
    dictionary = dict(zip(names, stat_assoc_l))
    adata.uns["tree"]["stat_assoc_list"]=dictionary
    
    return adata