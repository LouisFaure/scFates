from typing import Union, Optional, Tuple, Collection, Sequence, Iterable

import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
from scipy import sparse

from functools import partial
from statsmodels.stats.weightstats import DescrStatsW
from skmisc.loess import loess

import warnings
from .. import logging as logg
from .. import settings

import sys
sys.setrecursionlimit(10000)

def slide_cells(
    adata: AnnData,
    root_milestone,
    milestones,
    win: int = 50,
    mapping: bool = True,
    copy: bool = False,
    ):
    
    """\
    Assign cells in a probabilistic manner to non-intersecting windows along pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    win
        number of cell per local pseudotime window.
    mapping
        project cells onto tree pseudotime in a probabilistic manner.
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['cell_freq']`
            List of np.array containing probability assignment of cells on non intersecting windows.
    
    """
    
    adata = adata.copy() if copy else adata
    
    tree = adata.uns["tree"]
    
    uns_temp = adata.uns.copy()
    
    mlsc = adata.uns["milestones_colors"].copy()
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    def getsegs(g,root,leave,tree):
        path=np.array(g.vs[:]["name"])[np.array(g.get_shortest_paths(str(root),str(leave)))][0]
        segs = list()
        for i in range(len(path)-1):
            segs= segs + [np.argwhere((tree["pp_seg"][["from","to"]].astype(str).apply(lambda x: 
                                                                                    all(x.values == path[[i,i+1]]),axis=1)).to_numpy())[0][0]]
        segs=tree["pp_seg"].index[segs].tolist()
        return(segs)


    edges=tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    paths = list(map(lambda l: getsegs(img,root,l,tree),leaves))
    
    
    seg_progenies = list(set.intersection(*[set(path) for path in paths]))
    seg_branch1 = list(set.difference(set(paths[0]),set(seg_progenies)))
    seg_branch2 = list(set.difference(set(paths[1]),set(seg_progenies)))
    pp_probs = tree["R"].sum(axis=0)
    pps = tree["pp_info"].PP[tree["pp_info"].seg.isin(np.array(seg_progenies+seg_branch1+seg_branch2).astype(str))].index

    seg_branch1 = [str(seg) for seg in seg_branch1]
    seg_branch2 = [str(seg) for seg in seg_branch2]
    seg_progenies = [str(seg) for seg in seg_progenies]

    def region_extract(pt_cur,segs_cur):
        freq = list()

        pp_next = pps[(tree["pp_info"].loc[pps,"time"].values >= pt_cur) & 
                      tree["pp_info"].loc[pps,"seg"].isin(segs_cur).values]


        cmsm = np.cumsum(pp_probs[pp_next][np.argsort(tree["pp_info"].loc[pp_next,"time"].values)])
        inds=np.argwhere(cmsm > win).flatten()

        if len(inds)==0:
            if (cmsm.max() > win/2):
                if mapping:
                    cell_probs = tree["R"][:,pp_next].sum(axis=1)
                else:
                    cell_probs = np.isin(np.apply_along_axis(lambda x: np.argmax(x),axis=1,arr=tree["R"]),pp_next)*1
                freq = freq+[cell_probs]
            return freq
        else: 
            pps_region = pp_next[np.argsort(tree["pp_info"].loc[pp_next,"time"].values)][:inds[0]]
            if mapping:
                cell_probs = tree["R"][:,pps_region].sum(axis=1)
            else:
                cell_probs = np.isin(np.apply_along_axis(lambda x: np.argmax(x),axis=1,arr=tree["R"]),pps_region)*1

            freq = freq+[cell_probs]
            pt_cur = tree["pp_info"].loc[pps_region,"time"].max()



            if (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_progenies))==0):
                res = region_extract(pt_cur,segs_cur)
                return freq+res

            elif (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_branch1))==0):
                res = region_extract(pt_cur,segs_cur)
                return freq+res

            elif (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_branch2))==0):

                res = region_extract(pt_cur,segs_cur)
                return freq+res


            elif (~(sum(~tree["pp_info"].loc[pps_region,:].seg.isin([str(seg) for seg in seg_progenies]))==0)):
                pt_cur1 = tree["pp_info"].loc[pps_region,"time"][tree["pp_info"].loc[pps_region,"seg"].isin([str(seg) for seg in seg_branch1])].max()
                segs_cur1 = seg_branch1
                pt_cur2 = tree["pp_info"].loc[pps_region,"time"][tree["pp_info"].loc[pps_region,"seg"].isin([str(seg) for seg in seg_branch2])].max()
                segs_cur2 =seg_branch2
                res1 = region_extract(pt_cur1,segs_cur1)
                res2 = region_extract(pt_cur2,segs_cur2)
                return freq+res1+res2
            
    pt_cur = tree["pp_info"].loc[pps,"time"].min()
    segs_cur=np.unique(np.array(seg_progenies+seg_branch1+seg_branch2).flatten().astype(str))
    
    freq=region_extract(pt_cur,segs_cur)
    name=root_milestone+"->"+milestones[0]+"<>"+milestones[1]
    
    adata.uns=uns_temp
    
    
    
    adata.uns[name]["cell_freq"]=list(map(lambda f: 
                                          pd.Series(f,index=adata.uns["tree"]["cells_fitted"]),
                                          freq))
    
    logg.hint(
        "added \n"
        "    '"+name+"/cell_freq', probability assignment of cells on "+str(len(freq))+" non intersecting windows (adata.uns)")
    
    return adata if copy else None



def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones,
    layer: Optional[str] = None,
    copy: bool = False
    ):
    """\
    Obtain gene module correlations in the non-intersecting windows along pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    layer
        adata layer from which to compute the correlations.
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['corAB']`
            Dataframe containing gene-gene correlation modules.
    
    """
    
    adata = adata.copy() if copy else adata
    
    tree = adata.uns["tree"]    
    
    uns_temp = adata.uns.copy()
    
    mlsc = adata.uns["milestones_colors"].copy()
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=root_milestone+"->"+milestones[0]+"<>"+milestones[1]
    
    bif = adata.uns[name]["fork"]
    freqs= adata.uns[name]["cell_freq"]
    nwin = len(freqs)
    
    genesetA = bif.index[(bif["branch"]==milestones[0]).values & (bif["module"]=="early").values]
    genesetB = bif.index[(bif["branch"]==milestones[1]).values & (bif["module"]=="early").values]
    genesets = np.concatenate([genesetA,genesetB])
    
    if layer is None:
        if sparse.issparse(adata.X):
            X = adata[:,genesets].X.A
        else:
            X = adata[:,genesets].X
    else:
        if sparse.issparse(adata.layers[layer]):
            X = adata[:,genesets].layers[layer].A
        else:
            X = adata[:,genesets].layers[layer]
    
    X=pd.DataFrame(X,index=adata.obs_names,columns=genesets)
    X_r=X.rank(axis=0)
    def gather_cor(i,geneset):
        freq=freqs[i][adata.obs_names]
        cormat = pd.DataFrame(DescrStatsW(X_r.values,weights=freq).corrcoef,
                              index=genesets,columns=genesets)
        np.fill_diagonal(cormat.values, np.nan)
        return cormat.loc[:,geneset].mean(axis=1)

    
    gather = partial(gather_cor, geneset=genesetA)
    corA=pd.concat(list(map(gather,range(nwin))),axis=1)

    gather = partial(gather_cor, geneset=genesetB)
    corB=pd.concat(list(map(gather,range(nwin))),axis=1)
    
    corAB=pd.concat([corA,corB], keys=milestones) 
    corAB.columns=[str(c) for c in corAB.columns]
    
    adata.uns=uns_temp
    adata.uns[name]["corAB"]=corAB
    
    logg.hint(
        "added \n"
        "    '"+name+"/corAB', gene-gene correlation modules (adata.uns)")
    
    return adata if copy else None



def synchro_path(
    adata: AnnData,
    root_milestone,
    milestones,
    n_map=1,
    n_jobs=None,
    layer: Optional[str] = None,
    perm=True,
    w=200,
    step=30,
    winp=10,
    loess_span=.2,
    copy: bool = False
    ):
    """\
    Estimates pseudotime trends of local intra- and inter-module correlations of fates-specific modules.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    n_map
        number of probabilistic cells projection to use for estimates.
    n_jobs
        number of cpu processes to perform estimates (per mapping).
    layer
        adata layer to use for estimates.
    perm
        estimate control trends for local permutations instead of real expression matrix.
    w
        local window, in number of cells, to estimate correlations.
    step
        steps, in number of cells, between local windows.
    winp
        window of permutation in cells.
    loess_span
        fraction of points to take in account for loess fit
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns subsetted or else subset (keeping only
        significant features) and add fields to `adata`:
        
        `.uns['root_milestone->milestoneA<>milestoneB']['synchro']`
            Dataframe containing mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules.
        `.obs['intercor root_milestone->milestoneA<>milestoneB']`
            loess fit of inter-module mean local gene-gene correlations prior to bifurcation
    
    """
    
    adata = adata.copy() if copy else adata
       
    logg.info("computing local correlations", reset=True)
    
    tree = adata.uns["tree"]    
    
    edges = tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)  
    
    uns_temp = adata.uns.copy()
    
    mlsc = adata.uns["milestones_colors"].copy()
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=root_milestone+"->"+milestones[0]+"<>"+milestones[1]
    
    bif = adata.uns[name]["fork"]
    
    def synchro_map(m):
        df = tree["pseudotime_list"][str(m)]
        edges = tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
        img = igraph.Graph()
        img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
        img.add_edges(edges)  

        genesetA=bif.index[(bif.module=="early") & (bif.branch==milestones[0])]
        genesetB=bif.index[(bif.module=="early") & (bif.branch==milestones[1])]
        genesets = np.concatenate([genesetA,genesetB])
        
        def synchro_milestone(leave):
            cells=getpath(img,root,tree["tips"],leave,tree,df).index
            
            if layer is None:
                if sparse.issparse(adata.X):
                    mat = pd.DataFrame(adata[cells,genesets].X.A,
                        index=cells,columns=genesets)
                else:
                    mat = pd.DataFrame(adata[cells,genesets].X,
                        index=cells,columns=genesets)
            else:
                if sparse.issparse(adata.layers[layer]):
                    mat = pd.DataFrame(adata[cells,genesets].layers[layer].A,
                        index=cells,columns=genesets)
                else:
                    mat = pd.DataFrame(adata[cells,genesets].layers[layer],
                        index=cells,columns=genesets)
            
            mat=mat.iloc[adata.obs.t[mat.index].argsort().values,:]

            if permut==True:
                winperm=np.min([winp,mat.shape[0]])
                for i in np.arange(0,mat.shape[0]-winperm,winperm):
                    mat.iloc[i:(i+winperm),:]=mat.iloc[i:(i+winperm),:].apply(np.random.permutation,axis=0).values

            def slide_path(i):
                cls=mat.index[i:(i+w)]
                cor=mat.loc[cls,:].corr(method="spearman")
                corA=cor.loc[:,genesetA].mean(axis=1)
                corB=cor.loc[:,genesetB].mean(axis=1)
                corA[genesetA] = (corA[genesetA] - 1/len(genesetA))*len(genesetA)/(len(genesetA)-1)
                corB[genesetB] = (corB[genesetB] - 1/len(genesetB))*len(genesetB)/(len(genesetB)-1)

                return pd.Series({"t":adata.obs.t[cls].mean(),
                             "dist":(corA[genesetA].mean()-corA[genesetB].mean())**2+(corB[genesetA].mean()-corB[genesetB].mean())**2,
                             "corAA":corA[genesetA].mean(),"corBB":corB[genesetB].mean(),"corAB":corA[genesetB].mean(),
                                 "n_map":m})


            return pd.concat(list(map(slide_path,np.arange(0,mat.shape[0]-w,step))),axis=1).T

        return pd.concat(list(map(synchro_milestone,leaves)),keys=milestones)
    
    
    if n_map>1:
        permut = False
        stats = Parallel(n_jobs=n_jobs)(delayed(synchro_map)(i) for i in tqdm(range(n_map)))
        allcor_r = pd.concat(stat)
        if perm:
            permut=True
            stats = Parallel(n_jobs=n_jobs)(delayed(synchro_map)(i) for i in tqdm(range(n_map)))
            allcor_p=pd.concat(stats)
            allcor=pd.concat([allcor_r,allcor_p],keys=["real","permuted"])
        else:
            allcor=pd.concat([allcor_r], keys=['real'])
    else:
        permut=False
        allcor_r=pd.concat(list(map(synchro_map,range(n_map))))

        if perm:
            permut=True
            allcor_p=pd.concat(list(map(synchro_map,range(n_map))))
            allcor=pd.concat([allcor_r,allcor_p],keys=["real","permuted"])
        else:
            allcor=pd.concat([allcor_r], keys=['real'])
    
    
    
    
    runs=pd.DataFrame(allcor.to_records())["level_0"].unique()
    

    dct_cormil=dict(zip(["corAA","corBB","corAB"],
                        [milestones[0]+"\nintra-module",
                         milestones[1]+"\nintra-module"]+[milestones[0]+" vs "+milestones[1]+"\ninter-module"]))

    for cc in ["corAA","corBB","corAB"]:
        allcor[cc+"_lowess"]=0
        allcor[cc+"_ll"]=0
        allcor[cc+"_ul"]=0
        for r in range(len(runs)):
            for mil in milestones:
                res=allcor.loc[runs[r]].loc[mil]
                l = loess(res.t, res[cc],span=loess_span)
                l.fit()
                pred = l.predict(res.t, stderror=True)
                conf = pred.confidence()

                allcor.loc[(runs[r],mil),cc+"_lowess"] = pred.values
                allcor.loc[(runs[r],mil),cc+"_ll"] = conf.lower
                allcor.loc[(runs[r],mil),cc+"_ul"] = conf.upper
       
    
    fork=list(set(img.get_shortest_paths(str(root),str(leaves[0]))[0]).intersection(img.get_shortest_paths(str(root),str(leaves[1]))[0]))
    fork=np.array(img.vs["name"],dtype=int)[fork]
    fork_t=adata.uns["tree"]["pp_info"].loc[fork,"time"].max()
    res=allcor.loc[allcor.t<fork_t,:]
    res=res[~res.t.duplicated()]
    l = loess(res.t, res["corAB"],span=loess_span)
    l.fit()
    pred = l.predict(res.t, stderror=True)

    tval=adata.obs.t.copy()
    tval[tval>fork_t]=np.nan

    def inter_values(tv):
        if ~np.isnan(tv):
            return pred.values[np.argmin(np.abs(res.t.values-tv))]
        else:
            return tv
    adata.obs["inter_cor "+name]=list(map(inter_values,tval))
    
    df = tree["pseudotime_list"][str(0)]
    cells=np.concatenate([getpath(img,root,tree["tips"],leaves[0],tree,df).index,
                          getpath(img,root,tree["tips"],leaves[1],tree,df).index])
    
    adata.obs["inter_cor "+name][~adata.obs_names.isin(cells)]=np.nan              
    
    adata.uns=uns_temp
    
    adata.uns[name]["synchro"]=allcor
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    '"+name+"/synchro', mean local gene-gene correlations of all possible gene pairs inside one module, or between the two modules (adata.uns)\n"
        "    'inter_cor "+name+"', loess fit of inter-module mean local gene-gene correlations prior to bifurcation (adata.obs)")
    
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
        pth=df.loc[df.seg.astype(int).isin(segs),:].copy(deep=True)
        pth["branch"]=str(root)+"_"+str(tip)
        warnings.filters=wf
        return(pth)
    except IndexError:
        pass