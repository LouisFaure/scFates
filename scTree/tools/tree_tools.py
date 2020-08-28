from typing import Optional, Union
from anndata import AnnData
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import shortest_path
import igraph
from tqdm import tqdm
import sys
import igraph
import warnings
import itertools
from copy import deepcopy

from ..tools.dist_tools_cpu import euclidean_mat_cpu, cor_mat_cpu
from ..plot.tree import tree as plot_tree
from .. import logging as logg
from .. import settings

def tree(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    method: str = None,
    basis: Optional[str] = "umap",
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    ppt_sigma: Optional[Union[float, int]] = 0.1,
    ppt_lambda: Optional[Union[float, int]] = 1,
    epg_lambda: Optional[Union[float, int]] = 0.01,
    epg_mu: Optional[Union[float, int]] = 0.1,
    epg_trimmingradius: Optional = np.inf,
    epg_initnodes: Optional[int] = 2,
    ppt_nsteps: int = 50,
    ppt_err_cut: float = 5e-3,
    device: str = "cpu",
    plot: bool = False,
    seed: Optional[int] = None,
    copy: bool = False):
    
    
    logg.info("inferring a principal tree", reset=True, end=" " if settings.verbosity > 2 else "\n")
    
    adata = adata.copy() if copy else adata
    
    if method == "ppt":
        logg.hint(
        "parameters used \n"
        "    "+str(Nodes)+ " principal points, sigma = "+str(ppt_sigma)+", lambda = "+str(ppt_lambda)
        )
        tree_ppt(adata,M=Nodes,use_rep=use_rep,ndims_rep=ndims_rep,
                 init=init,sigma=ppt_sigma,lam=ppt_lambda,
                 nsteps=ppt_nsteps,err_cut=ppt_err_cut,
                 device=device,seed=seed)
                 
    elif method == "epg":
        logg.hint(
        "parameters used \n"
        "    "+str(Nodes)+ " principal points, mu = "+str(epg_mu)+", lambda = "+str(epg_lambda)
        )
        tree_epg(adata,Nodes,use_rep,ndims_rep,init,
                 epg_lambda,epg_mu,epg_trimmingradius,epg_initnodes,
                 device,seed)
    
    if plot:
        plot_tree(adata,basis)   
    
    return adata if copy else None



def tree_ppt(
    adata: AnnData,
    M: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    sigma: Optional[Union[float, int]] = 0.1,
    lam: Optional[Union[float, int]] = 1,
    nsteps: int = 50,
    err_cut: float = 5e-3,
    device: str = "cpu",
    seed: Optional[int] = None):
    
    
    
    if use_rep is None:
        use_rep = "X" if adata.n_vars < 50 or n_pcs == 0 else "X_pca"
        n_pcs = None if use_rep == "X" else n_pcs
    elif use_rep not in adata.obsm.keys() and f"X_{use_rep}" in adata.obsm.keys():
        use_rep = f"X_{use_rep}"
    
    X=DataFrame(adata.obsm[use_rep],index=adata.obs_names)
    
    X_t=X.values.T
    
    if seed is not None:
        np.random.seed(seed)
    
    if device=="gpu":
        import cupy as cp
        from .dist_tools_gpu import euclidean_mat_gpu, cor_mat_gpu
        X_gpu=cp.asarray(X_t)
        W=cp.empty_like(X_gpu)
        W.fill(1)


        if init is None:
            F_mat_gpu=X_gpu[:,np.random.choice(X.shape[0], size=M, replace=False)]
        else:
            F_mat_gpu=cp.asarray(init.T)
            M=init.T.shape[0]


        iterator = tqdm(range(nsteps),file=sys.stdout,desc="    fitting")
        for i in iterator:
            R = euclidean_mat_gpu(X_gpu,F_mat_gpu)
            R = (cp.exp(-R/sigma))
            R = (R.T/R.sum(axis=1)).T
            R[cp.isnan(R)]=0
            d = euclidean_mat_gpu(F_mat_gpu,F_mat_gpu)

            csr = csr_matrix(np.triu(cp.asnumpy(d),k=-1))
            Tcsr = minimum_spanning_tree(csr)
            mat=Tcsr.toarray()
            mat = mat + mat.T - np.diag(np.diag(mat))
            B=cp.asarray((mat>0).astype(int))

            D = cp.identity(B.shape[0])*B.sum(axis=0)
            L = D-B
            M = L*lam + cp.identity(R.shape[1])*R.sum(axis=0)
            old_F = F_mat_gpu

            F_mat_gpu=cp.linalg.solve(M.T,(cp.dot(X_gpu*W,R)).T).T

            err = cp.max(cp.sqrt((F_mat_gpu-old_F).sum(axis=0)**2)/cp.sqrt((F_mat_gpu**2).sum(axis=0)))
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break
        
        if i == (nsteps-1):
             logg.info("    inference not converged (error: "+str(err)+")")
            

        score=cp.array([cp.sum((1-cor_mat_gpu(F_mat_gpu,X_gpu))*R)/R.shape[0],
           sigma/R.shape[0]*cp.sum(R*cp.log(R)),
           lam/2*cp.sum(d*B)])

        B = cp.asnumpy(B)
        g = igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        tips = np.argwhere(np.array(g.degree())==1).flatten()
        forks = np.argwhere(np.array(g.degree())>2).flatten()

        r = [X.index.tolist(),cp.asnumpy(score),cp.asnumpy(F_mat_gpu),cp.asnumpy(R),
             (B),cp.asnumpy(L),cp.asnumpy(d),lam,sigma,nsteps,tips,forks,
            "euclidean"]
    else:  
        from .dist_tools_cpu import euclidean_mat_cpu, cor_mat_cpu
        X_cpu=np.asarray(X_t)
        W=np.empty_like(X_cpu)
        W.fill(1)

        if init is None:
            F_mat_cpu=X_cpu[:,np.random.choice(X.shape[0], size=M, replace=False)]
        else:
            F_mat_cpu=np.asarray(init.T)
            M=init.T.shape[0]

        j=1 
        err=100

        #while ((j <= nsteps) & (err > err_cut)):
        iterator = tqdm(range(nsteps),file=sys.stdout,desc="    ")
        for i in iterator:
            R = euclidean_mat_cpu(X_cpu,F_mat_cpu)
            R = (np.exp(-R/sigma))
            R = (R.T/R.sum(axis=1)).T
            R[np.isnan(R)]=0
            d = euclidean_mat_cpu(F_mat_cpu,F_mat_cpu)

            csr = csr_matrix(np.triu(d,k=-1))
            Tcsr = minimum_spanning_tree(csr)
            mat=Tcsr.toarray()
            mat = mat + mat.T - np.diag(np.diag(mat))
            B=((mat>0).astype(int))


            D = (np.identity(B.shape[0]))*np.array(B.sum(axis=0))
            L = D-B
            M = L*lam + np.identity(R.shape[1])*np.array(R.sum(axis=0))
            old_F = F_mat_cpu

            F_mat_cpu=np.linalg.solve(M.T,(np.dot(X_cpu*W,R)).T).T

            err = np.max(np.sqrt((F_mat_cpu-old_F).sum(axis=0)**2)/np.sqrt((F_mat_cpu**2).sum(axis=0)))

            err=err.item()
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break
        
        if i == (nsteps-1):
             logg.info("    not converged (error: "+str(err)+")")
        
        score=[np.sum((1-cor_mat_cpu(F_mat_cpu,X_cpu))*R)/R.shape[0],
           sigma/R.shape[0]*np.sum(R*np.log(R)),
           lam/2*np.sum(d*B)]

        g = igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        tips = np.argwhere(np.array(g.degree())==1).flatten()
        forks = np.argwhere(np.array(g.degree())>2).flatten()

        r = [X.index.tolist(),score,F_mat_cpu,R,B,L,d,lam,sigma,nsteps,tips,forks,"euclidean",use_rep]
    
    names = ['cells_fitted','score','F', 'R', 'B', 'L', 'd','lambda',
             'sigma','nsteps','tips','forks','metrics','rep_used']
    r = dict(zip(names, r))
    
    tree = {"B":r["B"],"R":r["R"],"F":r["F"],"tips":tips,"forks":forks,
            "cells_fitted":X.index.tolist(),"metrics":"euclidean"}
    
    adata.uns["tree"] = tree
    
    adata.uns["ppt"] = r
        
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    'ppt', dictionnary containing inferred tree (adata.uns)\n"
        "    'tree/B', adjacency matrix of the principal points (adata.uns)\n"
        "    'tree/R', soft assignment of cells to principal point in representation space (adata.uns)\n"
        "    'tree/F', coordinates of principal points in representation space (adata.uns)"
    )
    
    return adata



def tree_epg(
    adata: AnnData,
    Nodes: int = None,
    use_rep: str = None,
    ndims_rep: Optional[int] = None,
    init: Optional[DataFrame] = None,
    lam: Optional[Union[float, int]] = 0.01,
    mu: Optional[Union[float, int]] = 0.1,
    trimmingradius: Optional = np.inf,
    initnodes: int = None,
    device: str = "cpu",
    seed: Optional[int] = None):
    

    if use_rep is None:
        use_rep = "X" if adata.n_vars < 50 or n_pcs == 0 else "X_pca"
        n_pcs = None if use_rep == "X" else n_pcs
    elif use_rep not in adata.obsm.keys() and f"X_{use_rep}" in adata.obsm.keys():
        use_rep = f"X_{use_rep}"
    
    X=DataFrame(adata.obsm[use_rep],index=adata.obs_names)
    
    X_t=X.values.T
    
    if seed is not None:
        np.random.seed(seed)
    
    if device=="gpu":
        import elpigraphgpu
        import cupy as cp
        from .dist_tools_gpu import euclidean_mat_gpu, cor_mat_gpu
        
        Tree=elpigraphgpu.computeElasticPrincipalTree(X_t.T,NumNodes=Nodes,
                                                      Do_PCA=False,InitNodes=initnodes,
                                                      Lambda=lam,Mu=mu,
                                                      TrimmingRadius=trimmingradius,
                                                      drawAccuracyComplexity=False, 
                                                      drawPCAView=False, drawEnergy=False)
        
        
        R = euclidean_mat_gpu(cp.asarray(X_t),cp.asarray(Tree[0]["NodePositions"].T))
        
        R = (cp.exp(-R/.001))
        R = (R.T/R.sum(axis=1)).T
        R[cp.isnan(R)]=0
        
        R=cp.asnumpy(R)
        
        
    else:  
        import elpigraph
        from .dist_tools_cpu import euclidean_mat_cpu, cor_mat_cpu
        
        Tree = elpigraph.computeElasticPrincipalTree(X_t.T,NumNodes=Nodes,Do_PCA=False,
                                                      Lambda=epg_lambda,Mu=epg_mu,
                                                      drawAccuracyComplexity=False, 
                                                      drawPCAView=False, drawEnergy=False)
        
        R = euclidean_mat_cpu(X_t,Tree[0]["NodePositions"].T)
        
        # Force soft assigment to assign with confidence cells to their closest node
        R = (np.exp(-R/.0001))
        R = (R.T/R.sum(axis=1)).T
        R[np.isnan(R)]=0
    
    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(Tree[0]["Edges"][0].flatten().astype(int)))
    g.add_edges(pd.DataFrame(Tree[0]["Edges"][0]).astype(int).apply(tuple,axis=1).values)
    
    #mat = np.asarray(g.get_adjacency().data)
    #mat = mat + mat.T - np.diag(np.diag(mat))
    #B=((mat>0).astype(int))
    
    B = np.asarray(g.get_adjacency().data)

    tips = np.argwhere(np.array(g.degree())==1).flatten()
    forks = np.argwhere(np.array(g.degree())>2).flatten()
    
    tree = {"B":B,"R":R,"F":Tree[0]["NodePositions"].T,"tips":tips,"forks":forks,
            "cells_fitted":X.index.tolist(),"metrics":"euclidean"}
    
    Tree[0]["Edges"] = list(Tree[0]["Edges"])
    
    adata.uns["tree"] = tree
    adata.uns["epg"] = Tree[0]
    
        
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    'epg', dictionnary containing inferred elastic tree generated from elpigraph (adata.uns)\n"
        "    'tree/B', adjacency matrix of the principal points (adata.uns)\n"
        "    'tree/R', soft assignment (sigma=0.0001) of cells to principal point in representation space (adata.uns)\n"
        "    'tree/F', coordinates of principal points in representation space (adata.uns)"
    )
    
    return adata


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
    init_pp=np.arange(B.shape[0])
    if leaves is not None:
        g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        tips = np.argwhere(np.array(g.degree())==1).flatten()
        branches = np.argwhere(np.array(g.degree())>2).flatten()
        idxmin=list(map(lambda l: np.argmin(list(map(len,g.get_all_shortest_paths(l,branches)))),leaves))
        torem_manual=np.concatenate(list(map(lambda i: np.array(g.get_shortest_paths(leaves[i],branches[idxmin[i]])[0][:-1]),range(len(leaves)))))
        B=np.delete(B,torem_manual,axis=0)
        B=np.delete(B,torem_manual,axis=1)
        R=np.delete(R,torem_manual,axis=1)
        F=np.delete(F,torem_manual,axis=1)
    
    while True:
        torem=[]
        g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
        tips = np.argwhere(np.array(g.degree())==1).flatten()
        branches = np.argwhere(np.array(g.degree())>2).flatten()
        
        if len(branches)==0:
            break
        
        dist=np.array(list(map(lambda t: np.min(list(map(len,g.get_all_shortest_paths(t,branches)))),tips)))

        if np.argmin(dist)>minbranchlength:
            break
        
        tip_torem=tips[np.argmin(dist)].T.flatten()
        B=np.delete(B,tip_torem,axis=0)
        B=np.delete(B,tip_torem,axis=1)
        R=np.delete(R,tip_torem,axis=1)
        F=np.delete(F,tip_torem,axis=1)
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


def root(
    adata: AnnData,
    root: int,
    copy: bool = False):
    
    adata = adata.copy() if copy else adata
    
    if "tree" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before choosing a root."
        )
        
    r = adata.uns["tree"]
    
    if (r["metrics"]=="euclidean"):
        d = 1e-6 + euclidean_mat_cpu(r["F"],r["F"])
    
    to_g = r["B"]*d
    
    csr = csr_matrix(to_g)
    
    g = igraph.Graph.Adjacency((to_g>0).tolist(),mode="undirected")
    g.es['weight'] = to_g[to_g.nonzero()]
    
    root_dist_matrix = shortest_path(csr,directed=False, indices=root)
    pp_info=pd.DataFrame({"PP":g.vs.indices,
                          "time":root_dist_matrix,
                          "seg":np.zeros(csr.shape[0])})
    
    nodes = np.argwhere(np.apply_along_axis(arr=(csr>0).todense(),axis=0,func1d=np.sum)!=2).flatten()
    pp_seg = pd.DataFrame(columns = ["n","from","to","d"])
    for node1,node2 in itertools.combinations(nodes,2):
        paths12 = g.get_shortest_paths(node1,node2)
        paths12 = np.array([val for sublist in paths12 for val in sublist])

        if np.sum(np.isin(nodes,paths12))==2:
            fromto = np.array([node1,node2])
            path_root = root_dist_matrix[[node1,node2]]
            fro = fromto[np.argmin(path_root)]
            to = fromto[np.argmax(path_root)]
            pp_info.loc[paths12,"seg"]=pp_seg.shape[0]+1
            pp_seg=pp_seg.append(pd.DataFrame({"n":pp_seg.shape[0]+1,
                              "from":fro,"to":to,
                              "d":shortest_path(csr,directed=False, indices=fro)[to]},
                             index=[pp_seg.shape[0]+1]))
      
    pp_seg["n"]=pp_seg["n"].astype(int).astype(str)
    pp_seg["n"]=pp_seg["n"].astype(int).astype(str)
    
    pp_info["seg"]=pp_info["seg"].astype(int).astype(str)
    pp_info["seg"]=pp_info["seg"].astype(int).astype(str)
    
    r["pp_info"]=pp_info
    r["pp_seg"]=pp_seg
    r["root"]=root
    
    adata.uns["tree"] = r
    
    logg.info("root selected", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'tree/root', selected root (adata.uns)\n"
        "    'tree/pp_info', for each PP, its distance vs root and segment assignment (adata.uns)\n"
        "    'tree/pp_seg', segments network information (adata.uns)"
    )
    
    return adata if copy else None


def roots(
    adata: AnnData,
    roots,
    meeting,
    copy: bool = False):
    
    adata = adata.copy() if copy else adata
    
    if "tree" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before choosing two roots."
        )
        
    r = adata.uns["tree"]
    
    if (r["metrics"]=="euclidean"):
        d = 1e-6 + euclidean_mat_cpu(r["F"],r["F"])

    to_g = r["B"]*d

    csr = csr_matrix(to_g)

    g = igraph.Graph.Adjacency((to_g>0).tolist(),mode="undirected")
    g.es['weight'] = to_g[to_g.nonzero()]


    root=roots[np.argmax(shortest_path(csr,directed=False, indices=roots)[:,meeting])]
    root2=roots[np.argmin(shortest_path(csr,directed=False, indices=roots)[:,meeting])]

    root_dist_matrix = shortest_path(csr,directed=False, indices=root)
    pp_info=pd.DataFrame({"PP":g.vs.indices,
                          "time":root_dist_matrix,
                          "seg":np.zeros(csr.shape[0])})

    nodes = np.argwhere(np.apply_along_axis(arr=(csr>0).todense(),axis=0,func1d=np.sum)!=2).flatten()
    pp_seg = pd.DataFrame(columns = ["n","from","to","d"])
    for node1,node2 in itertools.combinations(nodes,2):
        paths12 = g.get_shortest_paths(node1,node2)
        paths12 = np.array([val for sublist in paths12 for val in sublist])

        if np.sum(np.isin(nodes,paths12))==2:
            fromto = np.array([node1,node2])
            path_root = root_dist_matrix[[node1,node2]]
            fro = fromto[np.argmin(path_root)]
            to = fromto[np.argmax(path_root)]
            pp_info.loc[paths12,"seg"]=pp_seg.shape[0]+1
            pp_seg=pp_seg.append(pd.DataFrame({"n":pp_seg.shape[0]+1,
                              "from":fro,"to":to,
                              "d":shortest_path(csr,directed=False, indices=fro)[to]},
                             index=[pp_seg.shape[0]+1]))

    pp_seg["n"]=pp_seg["n"].astype(int).astype(str)
    pp_seg["n"]=pp_seg["n"].astype(int).astype(str)

    pp_info["seg"]=pp_info["seg"].astype(int).astype(str)
    pp_info["seg"]=pp_info["seg"].astype(int).astype(str)


    tips=r["tips"]
    tips=tips[~np.isin(tips,roots)]


    edges=pp_seg[["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(pp_seg[["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)


    root2paths=pd.Series(shortest_path(csr,directed=False, indices=root2)[tips.tolist()+[meeting]],
              index=tips.tolist()+[meeting])

    toinvert=root2paths.index[(root2paths<=root2paths[meeting])]

    for toinv in toinvert:
        pathtorev=(np.array(img.vs[:]["name"])[np.array(img.get_shortest_paths(str(root2),str(toinv)))][0])
        for i in range((len(pathtorev)-1)):
            segtorev=pp_seg.index[pp_seg[["from","to"]].astype(str).apply(lambda x: 
                                                                          all(x.values == pathtorev[[i+1,i]]),axis=1)]
        
            pp_seg.loc[segtorev,["from","to"]]=pp_seg.loc[segtorev][["to","from"]].values
            pp_seg["from"]=pp_seg["from"].astype(int).astype(str)
            pp_seg["to"]=pp_seg["to"].astype(int).astype(str)

    pptoinvert=np.unique(np.concatenate(g.get_shortest_paths(root2,toinvert)))
    reverted_dist=shortest_path(csr,directed=False, indices=root2)+np.abs(np.diff(shortest_path(csr,directed=False, indices=roots)[:,meeting]))[0]
    pp_info.loc[pptoinvert,"time"] = reverted_dist[pptoinvert]
    
    
    
    r["pp_info"]=pp_info
    r["pp_seg"]=pp_seg
    r["root"]=root
    r["root2"]=root2
    r["meeting"]=meeting
    
    adata.uns["tree"] = r
    
    logg.info("root selected", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    "+str(root)+" is the farthest root\n"
        "    'tree/root', farthest root selected (adata.uns)\n"
        "    'tree/root2', 2nd root selected (adata.uns)\n"
        "    'tree/meeting', meeting point on the three (adata.uns)\n"
        "    'tree/pp_info', for each PP, its distance vs root and segment assignment (adata.uns)\n"
        "    'tree/pp_seg', segments network information (adata.uns)"
    )
    
    return adata if copy else None


def getpath(adata,
            root_milestone,
            milestones):
    
    tree = adata.uns["tree"]    
    
    edges = tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    g = igraph.Graph()
    g.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    g.add_edges(edges)  
    
    uns_temp = deepcopy(adata.uns)
    
    mlsc = deepcopy(adata.uns["milestones_colors"])
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves = list(map(lambda leave: dct[leave],milestones))
    root = dct[root_milestone]
    
    df = adata.obs.copy()
    
    warnings.filterwarnings("ignore")
    #for tip in leaves:
    def gatherpath(tip):    
        try:
            path = np.array(g.vs[:]["name"])[np.array(g.get_shortest_paths(str(root),str(tip)))][0]
            segs = list()
            for i in range(len(path)-1):
                segs= segs + [np.argwhere((tree["pp_seg"][["from","to"]].astype(str).apply(lambda x: 
                                                                                        all(x.values == path[[i,i+1]]),axis=1)).to_numpy())[0][0]]
            segs=tree["pp_seg"].index[segs]
            pth=df.loc[df.seg.astype(int).isin(segs),:].copy(deep=True)
            pth["branch"]=str(root)+"_"+str(tip)
            warnings.filterwarnings("default")
            return(pth)
        except IndexError:
            pass
    
    return pd.concat(list(map(gatherpath,leaves)),axis=1)