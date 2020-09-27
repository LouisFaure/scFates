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
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as sm

from scipy import sparse

from joblib import delayed, Parallel
from tqdm import tqdm

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
    warnings.warn(e.__doc__)
    warnings.warn(e.message)

        
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


def test_fork(
    adata: AnnData,
    root_milestone,
    milestones,
    layer: Optional[str] = None,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False):
    
    """\
    Test for branch differential gene expression, and differential upregulation after bifurcation point.
    
    First, differential gene expression between two branches is performed. Then,
    the feature are tested to intify the ones with higher average expression 
    in one of the derivative branches compared to the progenitor branch. 


    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    layer
        layer to use for the test
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']`
            DataFrame with fork test results.

    """
    
    adata = adata.copy() if copy else adata
    
    logg.info("testing fork", reset=True)

    genes = adata.var_names[adata.var.signi]
    
    tree = adata.uns["tree"]
    
    uns_temp = adata.uns.copy()
    
    zmlsc = adata.uns["milestones_colors"].copy()
    
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    g = igraph.Graph.Adjacency((tree["B"]>0).tolist(),mode="undirected")
    # Add edge weights and node labels.
    g.es['weight'] = tree["B"][tree["B"].nonzero()]

    vpath = g.get_shortest_paths(root,leaves)
    interPP = list(set(vpath[0]) & set(vpath[1])) 
    vpath = g.get_shortest_paths(tree["pp_info"].loc[interPP,:].time.idxmax(),leaves)

    fork_stat=list()
    upreg_stat=list()
    
    for m in range(n_map):
        logg.info("    mapping: "+str(m))
        ## Diff expr between forks
        
        df = tree["pseudotime_list"][str(m)]
        def get_branches(i):
            x = vpath[i][1:]
            segs=tree["pp_info"].loc[x,:].seg.unique()
            df_sub=df.loc[df.seg.isin(segs),:].copy(deep=True)
            df_sub.loc[:,"i"]=i
            return(df_sub)

        brcells = pd.concat(list(map(get_branches,range(len(vpath)))),axis=0,sort=False)
        matw=None
        if matw is None:
            brcells["w"]=1
        else:
            brcells["w"] = matw[gene,:][:,tree["cells_fitted"]]

        brcells.drop(["seg","edge"],axis=1,inplace=True)

        if layer is None:
            if sparse.issparse(adata.X):
                Xgenes = adata[brcells.index,genes].X.A.T.tolist()
            else:
                Xgenes = adata[brcells.index,genes].X.T.tolist()
        else:
            if sparse.issparse(adata.layers[layer]):
                Xgenes = adata[brcells.index,genes].layers[layer].A.T.tolist()
            else:
                Xgenes = adata[brcells.index,genes].layers[layer].T.tolist()

        data = list(zip([brcells]*len(Xgenes),Xgenes))

        stat = Parallel(n_jobs=n_jobs)(
                delayed(gt_fun)(
                    data[d]
                )
                for d in tqdm(range(len(data)),file=sys.stdout,desc="    differential expression")
            )

        fork_stat = fork_stat + [stat]
        
        ## test for upregulation
        logg.info("    test for upregulation for each leave vs root")
        leaves_stat=list()
        for leave in leaves:
            vpath = g.get_shortest_paths(root,leave)
            totest = get_branches(0)

            if layer is None:
                if sparse.issparse(adata.X):
                    Xgenes = adata[totest.index,genes].X.A.T.tolist()
                else:
                    Xgenes = adata[totest.index,genes].X.T.tolist()
            else:
                if sparse.issparse(adata.layers[layer]):
                    Xgenes = adata[totest.index,genes].layers[layer].A.T.tolist()
                else:
                    Xgenes = adata[totest.index,genes].layers[layer].T.tolist()

            data = list(zip([totest]*len(Xgenes),Xgenes))

            stat = Parallel(n_jobs=n_jobs)(
                    delayed(test_upreg)(
                        data[d]
                    )
                    for d in tqdm(range(len(data)),file=sys.stdout,desc="    leave "+str(keys[vals==leave][0]))
                )
            stat=pd.DataFrame(stat,index=genes,columns=[str(keys[vals==leave][0])+"_A",str(keys[vals==leave][0])+"_p"])
            leaves_stat=leaves_stat+[stat]
            
        upreg_stat=upreg_stat+[pd.concat(leaves_stat,axis=1)]
    
    
    # summarize fork statistics
    fork_stat=list(map(lambda x: pd.DataFrame(x,index=genes,columns=["effect","p_val"]),fork_stat))

    fdr_l=list(map(lambda x: pd.Series(multipletests(x.p_val,method="bonferroni")[1],
                                       index=x.index,name="fdr"),fork_stat))

    st_l=list(map(lambda x: pd.Series((x.p_val<5e-2).values*1,
                  index=x.index,name="signi_p"),fork_stat))
    stf_l=list(map(lambda x: pd.Series((x<5e-2).values*1,
                  index=x.index,name="signi_fdr"),fdr_l))

    fork_stat=list(map(lambda w,x,y,z: pd.concat([w,x,y,z],axis=1),fork_stat,fdr_l,st_l,stf_l))

    effect=pd.concat(list(map(lambda x: x.effect,fork_stat)),axis=1).median(axis=1)
    p_val=pd.concat(list(map(lambda x: x.p_val,fork_stat)),axis=1).median(axis=1)
    fdr=pd.concat(list(map(lambda x: x.fdr,fork_stat)),axis=1).median(axis=1)
    signi_p=pd.concat(list(map(lambda x: x.signi_p,fork_stat)),axis=1).mean(axis=1)
    signi_fdr=pd.concat(list(map(lambda x: x.signi_fdr,fork_stat)),axis=1).mean(axis=1)

    
    colnames=fork_stat[0].columns
    fork_stat=pd.concat([effect,p_val,fdr,signi_p,signi_fdr],axis=1)
    fork_stat.columns=colnames
    
    # summarize upregulation stats
    colnames=upreg_stat[0].columns
    upreg_stat=pd.concat(list(map(lambda i: 
                                  pd.concat(list(map(lambda x: 
                                                     x.iloc[:,i]
                                                     ,upreg_stat)),axis=1).median(axis=1),
                                  range(4))),axis=1)
    upreg_stat.columns=colnames
    
    
    summary_stat=pd.concat([fork_stat,upreg_stat],axis=1)
    adata.uns=uns_temp
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    #adata.uns[name]["fork"] = summary_stat
    
    adata.uns[name] = {"fork":summary_stat}
    logg.hint(
        "added \n"
        "    '"+name+"/fork', DataFrame with fork test results (adata.uns)")

    return adata if copy else None


def gt_fun(data):

    sdf = data[0]
    sdf["exp"] = data[1]

    ## add weighted matrix part
    #
    #

    global rmgcv
    global rstats


    def gamfit(sdf):
        m = rmgcv.gam(Formula("exp ~ s(t)+s(t,by=as.factor(i))+as.factor(i)"),
                      data=sdf,weights=sdf["w"])
        return rmgcv.summary_gam(m)[3][1]

    g = sdf.groupby("i")
    return [g.apply(lambda x: np.mean(x.exp)).diff(periods=-1)[0],gamfit(sdf)]


def test_upreg(data):

    sdf = data[0]
    sdf["exp"] = data[1]

    result = sm.ols(formula="exp ~ t", data=sdf).fit()
    return([result.params["t"],result.pvalues["t"]])



def branch_specific(
    adata: AnnData,
    root_milestone,
    milestones,
    effect_b1: float = None,
    effect_b2: float = None,
    stf_cut: float = 0.7,
    pd_a: float = 0,
    pd_p: float = 5e-2,
    copy: bool = False,
    ):
    
    """\
    Assign genes differentially expressed between two post-bifurcation branches.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    effect_b1
        expression differences to call gene as differentially upregulated at branch 1.
    effect_b2
        expression differences to call gene as differentially upregulated at branch 2.
    stf_cut
        fraction of projections when gene passed fdr < 0.05.
    pd_a
        minimum expression increase at derivative compared to progenitor branches to call gene as branch-specific.
    pd_p
        p-value of expression changes of derivative compared to progenitor branches to call gene as branch-specific.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['branch']`
            assigned branch.

    """
    
    adata = adata.copy() if copy else adata
    
    tree=adata.uns["tree"]
    
    uns_temp = adata.uns.copy()
    
    mlsc = adata.uns["milestones_colors"].copy()
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    stats = adata.uns[name]["fork"]
    
    stats["branch"] = "none" 
    
    idx_a=stats.apply(lambda x: (x.effect > effect_b1) &
                                        (x[5]>pd_a) & (x[6]<pd_p) & (x.signi_fdr >stf_cut),
                                         axis=1)
    
    idx_a=idx_a[idx_a].index
    
    logg.info("    "+str(len(idx_a))+" features found to be specific to leave "+str(keys[vals==leaves[0]][0]))
    
    

    idx_b=stats.apply(lambda x: (x.effect < -effect_b2) &
                                            (x[7]>pd_a) & (x[8]<pd_p) & (x.signi_fdr >stf_cut),
                                             axis=1)
    
    idx_b=idx_b[idx_b].index
    
    logg.info("    "+str(len(idx_b))+" features found to be specific to leave "+str(keys[vals==leaves[1]][0]))

    stats.loc[idx_a,"branch"] = str(keys[vals==leaves[0]][0])
    stats.loc[idx_b,"branch"] = str(keys[vals==leaves[1]][0])
    
    stats = stats.loc[stats.branch!="none",:]
    
    adata.uns=uns_temp
    
    adata.uns[name]["fork"] = stats
    
    logg.info("    finished", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    '"+name+"/fork', DataFrame updated with additionnal 'branch' column (adata.uns)")

    return adata if copy else None


def activation(adata: AnnData,
    root_milestone,
    milestones,
    deriv_cut: float = 0.015,
    pseudotime_offset: float = 0,
    n_map: int =1,
    copy: bool = False,
    n_jobs=-1,
    layer: Optional[str] = None):
    
    """\
    Identify pseudotime of activation of branc-specififc features.
    
    This aims in classifying the genes according to their their activation timing
    compared to the pseudotime of the bifurcation. Any feature activated before the 
    bifurcation is considered as 'early', others are considered 'late'.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    deriv_cut
        a first passage of derivative at this cutoff is considered as activation timing
    pseudotime_offset
        consider a feature as early if it gets activated before: pseudotime at bifurcation-pseudotime_offset.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['module']`
            classify feature as 'early' or 'late'.
        `.uns['root_milestone->milestoneA<>milestoneB']['fork']['activation']`
            pseudotime of activationh.

    """
    
    tree = adata.uns["tree"]
    
    logg.info("testing activation", reset=True)
    
    uns_temp = adata.uns.copy()
    
    mlsc = adata.uns["milestones_colors"].copy()
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    stats = adata.uns[name]["fork"]
    
    for m in range(n_map):
        df = tree["pseudotime_list"][str(m)]
        edges = tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
        img = igraph.Graph()
        img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
        img.add_edges(edges)  

        def get_df(feature):
            global rmgcv
            subtree=getpath(img,root,tree["tips"],leave,tree,df).sort_values("t")
            del subtree["branch"]
            subtree["deriv_cut"]=deriv_cut
            wf=warnings.filters.copy()
            warnings.filterwarnings("ignore")
            if layer is None:
                if sparse.issparse(adata.X):
                    subtree["exp"] = np.array(adata[subtree.index,feature].X.A)
                else:
                    subtree["exp"] = np.array(adata[subtree.index,feature].X)
            else:
                if sparse.issparse(adata.layers[layer]):
                    subtree["exp"] = np.array(adata[subtree.index,feature].layers[layer].A)
                else:
                    subtree["exp"] = np.array(adata[subtree.index,feature].layers[layer])
            warnings.filters=wf
            return subtree
            

        genes1 = stats.index[stats["branch"]==str(keys[vals==leaves[0]][0])]
        leave = leaves[0]
        dfs=list(map(get_df,genes1))
        
        acts1 = Parallel(n_jobs=n_jobs)(
            delayed(get_activation)(
                dfs[d]
            )
            for d in tqdm(range(len(dfs)),file=sys.stdout,desc="    leave "+str(keys[vals==leave][0]))
        )

        genes2=stats.index[stats["branch"]==str(keys[vals==leaves[1]][0])]
        leave=leaves[1]
        dfs=list(map(get_df,genes2))
        
        acts2 = Parallel(n_jobs=n_jobs)(
            delayed(get_activation)(
                dfs[d]
            )
            for d in tqdm(range(len(dfs)),file=sys.stdout,desc="    leave "+str(keys[vals==leave][0]))
        )

    stats["activation"] = 0
    stats.loc[genes1,"activation"] = acts1
    stats.loc[genes2,"activation"] = acts2

    fork = list(set(img.get_shortest_paths(str(root),str(leaves[0]))[0]).intersection(img.get_shortest_paths(str(root),str(leaves[1]))[0]))
    fork = np.array(img.vs["name"],dtype=int)[fork]
    fork_t = adata.uns["tree"]["pp_info"].loc[fork,"time"].max()-pseudotime_offset
    
    stats["module"] = "early"
    stats.loc[stats["activation"]>fork_t,"module"]="late"
    
    adata.uns=uns_temp
    
    adata.uns[name]["fork"] = stats
    
    c_early = np.sum((stats.branch==str(keys[vals==leaves[0]][0])) & (stats.module=="early"))
    c_late = np.sum((stats.branch==str(keys[vals==leaves[0]][0])) & (stats.module=="late"))
    logg.info("    "+str(c_early)+" early and "+str(c_late)+" late features specific to leave "+str(keys[vals==leaves[0]][0]))
   
    c_early = np.sum((stats.branch==str(keys[vals==leaves[1]][0])) & (stats.module=="early"))
    c_late = np.sum((stats.branch==str(keys[vals==leaves[1]][0])) & (stats.module=="late"))
    logg.info("    "+str(c_early)+" early and "+str(c_late)+" late features specific to leave "+str(keys[vals==leaves[1]][0]))
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated \n"
        "    '"+name+"/fork', DataFrame updated with additionnal 'activation' and 'module' columns (adata.uns)")
    
    return adata if copy else None


def get_activation(subtree):
    global rmgcv 
    deriv_cut=subtree["deriv_cut"][0]
    wf=warnings.filters.copy()
    warnings.filterwarnings("ignore")
    def gamfit(sdf):
        m = rmgcv.gam(Formula("exp ~ s(t)"),data=sdf,gamma=1)
        return rmgcv.predict_gam(m)

    subtree["fitted"]=gamfit(subtree)
    deriv_d = subtree.fitted.max()-subtree.fitted.min()
    deriv_n=subtree.fitted.diff()
    deriv = deriv_n/deriv_d
    if sum((deriv>deriv_cut).values)==0:
        act=subtree.t.max()+1
    else:
        act=np.min(subtree.t[(deriv>deriv_cut).values])
    warnings.filters=wf
    return np.min([act,subtree.t.max()])

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
        pth=df.loc[df.seg.astype(int).isin(segs),:].copy(deep=True)
        pth["branch"]=str(root)+"_"+str(tip)
        warnings.filters=wf
        return(pth)
    except IndexError:
        pass