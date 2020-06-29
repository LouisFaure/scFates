import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import itertools

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ..tools.dist_tools_cpu import euclidean_mat_cpu, cor_mat_cpu
from .. import logging as logg
from .. import settings

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