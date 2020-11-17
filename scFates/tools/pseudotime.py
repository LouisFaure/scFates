from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from joblib import delayed, Parallel
from tqdm import tqdm
import sys
import igraph

from .. import logging as logg
from .. import settings

def pseudotime(
    adata: AnnData,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False):
    """\
    Compute pseudotime.
    
    Projects cells onto the tree, and uses distance from the root as a pseudotime value.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_jobs
        Number of cpu processes to use in case of performing multiple mapping.
    n_map
        number of probabilistic mapping of cells onto the tree to use. If n_map=1 then likelihood cell mapping is used.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        
        `.obs['edge']`
            assigned edge.
        `.obs['t']`
            assigned pseudotime value.
        `.obs['seg']`
            assigned segment of the tree.
        `.obs['milestone']`
            assigned region surrounding forks and tips.
        `.uns['pseudotime_list']`
            list of cell projection from all mappings.
    """
    
    if "root" not in adata.uns["tree"]:
        raise ValueError(
            "You need to run `tl.root` or `tl.roots` before projecting cells."
        )
    
    adata = adata.copy() if copy else adata
    
    tree = adata.uns["tree"]

    logg.info("projecting cells onto the principal tree", reset=True)
    
    
    if n_map == 1:
        df_l = [map_cells(tree,multi=False)]
    else:
        df_l = Parallel(n_jobs=n_jobs)(
            delayed(map_cells)(
                tree=tree,multi=True
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
    adata.uns["pseudotime_list"]=dictionary
    
    if n_map > 1:
        adata.obs["t_sd"]=pd.concat(list(map(lambda x: pd.Series(x["_t"]),
                           list(adata.uns["pseudotime_list"].values()))),axis=1).apply(np.std,axis=1).values

   
    milestones=pd.Series(index=adata.obs_names)
    for seg in tree["pp_seg"].n:
        cell_seg=adata.obs.loc[adata.obs["seg"]==seg,"t"]
        milestones[cell_seg.index[(cell_seg-min(cell_seg)-(max(cell_seg-min(cell_seg))/2)<0)]]=tree["pp_seg"].loc[int(seg),"from"]
        milestones[cell_seg.index[(cell_seg-min(cell_seg)-(max(cell_seg-min(cell_seg))/2)>0)]]=tree["pp_seg"].loc[int(seg),"to"]
    adata.obs["milestones"]=milestones
    adata.obs.milestones=adata.obs.milestones.astype("category")
    
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n" + "    'edge', assigned edge (adata.obs)\n"
        "    't', pseudotime value (adata.obs)\n"
        "    'seg', segment of the tree where the cell is assigned to (adata.obs)\n"
        "    'milestones', milestones assigned to (adata.obs)\n"
        "    'pseudotime_list', list of cell projection from all mappings (adata.uns)"
    )
    
    return adata if copy else None

def map_cells(tree,multi=False):
    import igraph
    g = igraph.Graph.Adjacency((tree["B"]>0).tolist(),mode="undirected")
    # Add edge weights and node labels.
    g.es['weight'] = tree["B"][tree["B"].nonzero()]
    if multi: 
        rrm = (np.apply_along_axis(lambda x: np.random.choice(np.arange(len(x)),size=1,p=x),axis=1,arr=tree["R"])).T.flatten()
    else:
        rrm = np.apply_along_axis(np.argmax,axis=1,arr=tree["R"])
    
    def map_on_edges(v):
        vcells=np.argwhere(rrm==v)

        if vcells.shape[0]>0:
            nv = np.array(g.neighborhood(v,order=1))
            nvd = np.array(g.shortest_paths(v,nv)[0])

            spi = np.apply_along_axis(np.argmax,axis=1,arr=tree["R"][vcells,nv[1:]])
            ndf = pd.DataFrame({"cell":vcells.flatten(),"v0":v,"v1":nv[1:][spi],"d":nvd[1:][spi]})

            p0 = tree["R"][vcells,v].flatten()
            p1 = np.array(list(map(lambda x: tree["R"][vcells[x],ndf.v1[x]],range(len(vcells))))).flatten()

            alpha = np.random.uniform(size=len(vcells))
            f = np.abs( (np.sqrt(alpha*p1**2+(1-alpha)*p0**2)-p0)/(p1-p0) )
            ndf["t"] = tree["pp_info"].loc[ndf.v0,"time"].values+(tree["pp_info"].loc[ndf.v1,"time"].values-tree["pp_info"].loc[ndf.v0,"time"].values)*alpha
            ndf["seg"] = 0
            isinfork = (tree["pp_info"].loc[ndf.v0,"PP"].isin(tree["forks"])).values
            ndf.loc[isinfork,"seg"] = tree["pp_info"].loc[ndf.loc[isinfork,"v1"],"seg"].values
            ndf.loc[~isinfork,"seg"] = tree["pp_info"].loc[ndf.loc[~isinfork,"v0"],"seg"].values
            
            return ndf
        else:
            return None
    
    
    df = list(map(map_on_edges,range(tree["B"].shape[1])))
    df = pd.concat(df)
    df.sort_values("cell",inplace=True)
    df.index=tree["cells_fitted"]
    
    df["edge"]=df.apply(lambda x: str(int(x[1]))+"|"+str(int(x[2])),axis=1)

    df.drop(["cell","v0","v1","d"],axis=1,inplace=True)

    return df

def refine_pseudotime(
    adata: AnnData,
    n_jobs: int = 1,
    ms_data = None,
    use_rep = None,
    copy: bool = False):
    
    """\
    Refine computed pseudotime.
    
    Projection using principal tree can lead to compressed pseudotimes for the cells localised 
    near the tips. To counteract this, diffusion based pseudotime is performed using Palantir [Setty19]_ on each
    segment separately.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_jobs
        Number of cpu processes (max is the number of segments).
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:
        
        `.obs['t']`
            updated assigned pseudotimes value.
        `.obs['t_old']`
            previously assigned pseudotime.
    """
    
    adata = adata.copy() if copy else adata
    
    adata.obs["t_old"]=adata.obs.t.copy()
    
    logg.info("refining pseudotime using palantir on each segment of the tree", reset=True)
    
    def palantir_on_seg(seg,ms_data=ms_data):
        import palantir
        adata_sub=adata[adata.obs.seg==seg,]
        
        if use_rep is not None:
            dm_res=palantir.utils.run_diffusion_maps(pd.DataFrame(adata_sub.obsm["X_"+use_rep],
                                                                  index=adata_sub.obs_names))
            ms_data=palantir.utils.determine_multiscale_space(dm_res)
        elif ms_data is not None:
            ms_data=pd.DataFrame(adata_sub.obsm["X_"+ms_data],index=adata_sub.obs_names)
        else:
            dm_res=palantir.utils.run_diffusion_maps(pd.DataFrame(adata_sub.X,
                                                                  index=adata_sub.obs_names))
            ms_data=palantir.utils.determine_multiscale_space(dm_res)
        pr=palantir.core.run_palantir(ms_data,adata_sub.obs.t.idxmin())
        return pr.pseudotime

    pseudotimes = Parallel(n_jobs=n_jobs)(
        delayed(palantir_on_seg)(
            s
        )
        for s in tqdm(adata.uns["tree"]["pp_seg"].n.values.astype(str),file=sys.stdout)
    )

    g=igraph.Graph(directed=True)

    g.add_vertices(np.unique(adata.uns["tree"]["pp_seg"].loc[:,["from","to"]].values.flatten().astype(str)))
    g.add_edges(adata.uns["tree"]["pp_seg"].loc[:,["from","to"]].values.astype(str))

    allpth=g.get_shortest_paths(str(adata.uns["tree"]["root"]),[tip for tip in g.vs["name"] if tip!=str(adata.uns["tree"]["root"])])

    for p in allpth:
        pth=np.array(g.vs["name"],dtype=int)[p]
        dt=0
        for i in range(len(pth)-1):
            sel=adata.uns["tree"]["pp_seg"].loc[:,["from","to"]].apply(lambda x: np.all(x==pth[i:i+2]),axis=1).values
            adata.obs.loc[adata.obs.seg==adata.uns["tree"]["pp_seg"].loc[sel,"n"].values[0],"t"]=(pseudotimes[np.argwhere(sel)[0][0]]*adata.uns["tree"]["pp_seg"].loc[sel,"d"].values[0]).values+dt
            dt=dt+adata.uns["tree"]["pp_seg"].loc[sel,"d"].values[0]
            
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated\n" + "    't' with palantir pseudotime values (adata.obs)\n"
        "added\n" +"    't_old', previous pseudotime data (adata.obs)"
    )
    
    return adata if copy else None