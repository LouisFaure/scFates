from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from joblib import delayed, Parallel
from tqdm import tqdm
import sys

from .. import logging as logg
from .. import settings

def pseudotime(
    adata: AnnData,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False):
    
    if "root" not in adata.uns["tree"]:
        raise ValueError(
            "You need to run `tl.root` or `tl.roots` before projecting cells."
        )
        
        
    r = adata.uns["tree"]

    logg.info("projecting cells onto the principal tree")
    
    
    if n_map == 1:
        df_l = [map_cells(r,multi=False)]
    else:
        df_l = Parallel(n_jobs=n_jobs)(
            delayed(map_cells)(
                r=r,multi=True
            )
            for m in tqdm(range(n_map),file=sys.stdout,desc="    mappings")
        )

    
    # formatting cell projection data
    
    df_summary = df_l[0]

    df_summary["seg"]=df_summary["seg"].astype("category")
    df_summary["edge"]=df_summary["edge"].astype("category")
    
    # remove pre-existing palette to avoid errors with plotting
    if "seg_colors" in adata.uns:
        del adata.uns["seg_colors"]
    
    
    if set(df_summary.columns.tolist()).issubset(adata.obs.columns):
        adata.obs[df_summary.columns] = df_summary
    else:
        adata.obs = pd.concat([adata.obs,df_summary],axis=1)
    
    
    #list(map(lambda x: x.column))
    
    #todict=list(map(lambda x: dict(zip(["cells"]+["_"+s for s in x.columns.tolist()],
    #                                   [x.index.tolist()]+x.to_numpy().T.tolist())),df_l))
    names = np.arange(len(df_l)).astype(str).tolist()
    #vals = todict
    dictionary = dict(zip(names, df_l))
    adata.uns["tree"]["pseudotime_list"]=dictionary
    
    if n_map > 1:
        adata.obs["t_sd"]=pd.concat(list(map(lambda x: pd.Series(x["_t"]),
                           list(adata.uns["tree"]["pseudotime_list"].values()))),axis=1).apply(np.std,axis=1).values

   
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'edge', assigned edge (adata.obs)\n"
        "    't', pseudotime value (adata.obs)\n"
        "    'seg', segment of the tree where the cell is assigned to (adata.obs)\n"
        "    'tree/pseudotime_list', list of cell projection from all mappings (adata.uns)"
    )
    
    return adata if copy else None

def map_cells(r,multi=False):
    import igraph
    g = igraph.Graph.Adjacency((r["B"]>0).tolist(),mode="undirected")
    # Add edge weights and node labels.
    g.es['weight'] = r["B"][r["B"].nonzero()]
    if multi: 
        rrm = (np.apply_along_axis(lambda x: np.random.choice(np.arange(len(x)),size=1,p=x),axis=1,arr=r["R"])).T.flatten()
    else:
        rrm = np.apply_along_axis(np.argmax,axis=1,arr=r["R"])
    
    def map_on_edges(v):
        vcells=np.argwhere(rrm==v)

        if vcells.shape[0]>0:
            nv = np.array(g.neighborhood(v,order=1))
            nvd = np.array(g.shortest_paths(v,nv)[0])

            spi = np.apply_along_axis(np.argmax,axis=1,arr=r["R"][vcells,nv[1:]])
            ndf = pd.DataFrame({"cell":vcells.flatten(),"v0":v,"v1":nv[1:][spi],"d":nvd[1:][spi]})

            p0 = r["R"][vcells,v].flatten()
            p1 = np.array(list(map(lambda x: r["R"][vcells[x],ndf.v1[x]],range(len(vcells))))).flatten()

            alpha = np.random.uniform(size=len(vcells))
            f = np.abs( (np.sqrt(alpha*p1**2+(1-alpha)*p0**2)-p0)/(p1-p0) )
            ndf["t"] = r["pp_info"].loc[ndf.v0,"time"].values+(r["pp_info"].loc[ndf.v1,"time"].values-r["pp_info"].loc[ndf.v0,"time"].values)*alpha
            ndf["seg"] = 0
            isinfork = (r["pp_info"].loc[ndf.v0,"PP"].isin(r["forks"])).values
            ndf.loc[isinfork,"seg"] = r["pp_info"].loc[ndf.loc[isinfork,"v1"],"seg"].values
            ndf.loc[~isinfork,"seg"] = r["pp_info"].loc[ndf.loc[~isinfork,"v0"],"seg"].values
            
            return ndf
        else:
            return None
    
    
    df = list(map(map_on_edges,range(r["B"].shape[1])))
    df = pd.concat(df)
    df.sort_values("cell",inplace=True)
    df.index=r["cells_fitted"]
    
    df["edge"]=df.apply(lambda x: str(int(x[1]))+"|"+str(int(x[2])),axis=1)

    df.drop(["cell","v0","v1","d"],axis=1,inplace=True)

    return df
