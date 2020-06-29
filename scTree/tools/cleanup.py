from typing import Optional
from anndata import AnnData
import numpy as np
import igraph

from .. import logging as logg
from .. import settings

def cleanup(
    adata: AnnData,
    minbranchlength: int = 3,
    leaves: Optional[int] = None,
    copy: bool = False):
    
    adata = adata.copy() if copy else adata
    
    if "tree" not in adata.uns:
        raise ValueError(
            "You need to run `tl.ppt_tree` first to compute a princal tree before cleaning it"
        )
    r = adata.uns["tree"]
    
    B=r["B"]
    R=r["R"]
    F=r["F"]
    init_num=B.shape[0]
    
    if leaves is not None:
        g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        branches = np.argwhere(np.array(g.degree())>2).flatten()
        idxmin=np.array(list(map(lambda t: np.argmin(list(map(len,g.get_all_shortest_paths(t,branches)))),leaves))).item()
        torem_manual=list(map(lambda t: g.get_all_shortest_paths(t,branches),leaves))[0][idxmin]
        B=np.delete(B,torem_manual,axis=0)
        B=np.delete(B,torem_manual,axis=1)
        R=np.delete(R,torem_manual,axis=1)
        F=np.delete(F,torem_manual,axis=1)
    
    while True:
        torem=[]
        g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        tips = np.argwhere(np.array(g.degree())==1).flatten()
        branches = np.argwhere(np.array(g.degree())>2).flatten()
        dist=np.array(list(map(lambda t: np.min(list(map(len,g.get_all_shortest_paths(t,branches)))),tips)))

        tips_torem=tips[np.argwhere(dist<minbranchlength)].T.flatten()
        if len(tips_torem)==0:
            break
        B=np.delete(B,tips_torem,axis=0)
        B=np.delete(B,tips_torem,axis=1)
        R=np.delete(R,tips_torem,axis=1)
        F=np.delete(F,tips_torem,axis=1)
    R = (R.T/R.sum(axis=1)).T
    r["R"]=R
    r["B"]=B
    r["F"]=F
    g = igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
    r["tips"] = np.argwhere(np.array(g.degree())==1).flatten()
    r["forks"] = np.argwhere(np.array(g.degree())>2).flatten()
    
    adata.uns["tree"] = r
    
    logg.info("    tree cleaned", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "removed "+str(init_num-B.shape[0])+" principal points"
    )
    
    return adata if copy else None