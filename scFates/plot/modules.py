import numpy as np
import pandas as pd
from anndata import AnnData
import warnings

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from scipy import sparse 
import warnings

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from ..tools.utils import getpath

def modules(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    basis: str = "umap",
    mode: str = "2d",
    marker_size: int = 20,
    highlight: bool = False,
    incl_3d: int = 30,
    rot_3d: int = 315,
    alpha: float = 1,
    cmap_pseudotime = "viridis",
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    layer: Optional[str] = None):
    
    plt.rcParams["axes.grid"] = False
    graph=adata.uns["graph"]
    
    uns_temp=adata.uns.copy()
    
    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    
    stats = adata.uns[name]["fork"]
    mlsc = adata.uns["milestones_colors"].copy()
    mls = adata.obs.milestones.cat.categories.tolist()
    dct = dict(zip(mls,mlsc))
    df = adata.obs.copy(deep=True)
    edges=graph["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(graph["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    
    cells=np.unique(np.concatenate([getpath(img,root,adata.uns["graph"]["tips"],leaves[0],graph,df).index,
                   getpath(img,root,adata.uns["graph"]["tips"],leaves[1],graph,df).index]))

    
    cols = adata.uns[color+"_colors"].copy()
    obscol = adata.obs[color].cat.categories.tolist()
    dct_c = dict(zip(obscol,cols))
           
        
    if layer is None:
        if sparse.issparse(adata.X):
            X=pd.DataFrame(np.array(adata[cells,stats.index].X.A),index=cells,columns=stats.index)
        else:
            X=pd.DataFrame(np.array(adata[cells,stats.index].X),index=cells,columns=stats.index)
    else:
        if sparse.issparse(adata.layers[layer]):
            X=pd.DataFrame(np.array(adata[cells,stats.index].layers[layer].A),index=cells,columns=stats.index)
        else:
            X=pd.DataFrame(np.array(adata[cells,stats.index].layers[layer]),index=cells,columns=stats.index)    
    
    miles=adata.obs.loc[X.index,color].astype(str)

    early_1=(stats.branch.values==str(keys[vals==leaves[0]][0])) & (stats.module.values=="early")
    late_1=(stats.branch.values==str(keys[vals==leaves[0]][0])) & (stats.module.values=="late")

    early_2=(stats.branch.values==str(keys[vals==leaves[1]][0])) & (stats.module.values=="early")
    late_2=(stats.branch.values==str(keys[vals==leaves[1]][0])) & (stats.module.values=="late")
    
    
    if mode=="2d":
        fig, axs = plt.subplots(2,2)
        
        if highlight:
            axs[0,0].scatter(X.loc[:,early_1].mean(axis=1),
                        X.loc[:,early_2].mean(axis=1),s=marker_size*2,c="k")
            axs[1,0].scatter(X.loc[:,late_1].mean(axis=1),
                        X.loc[:,late_2].mean(axis=1),s=marker_size*2,c="k")
            axs[0,1].scatter(X.loc[:,early_1].mean(axis=1),
                    X.loc[:,early_2].mean(axis=1),s=marker_size*2,c="k")
            axs[1,1].scatter(X.loc[:,late_1].mean(axis=1),
                    X.loc[:,late_2].mean(axis=1),s=marker_size*2,c="k")
        
        for m in obscol:
            axs[0,0].scatter(X.loc[miles.index[miles==m],early_1].mean(axis=1),
                        X.loc[miles.index[miles==m],early_2].mean(axis=1),
                             s=marker_size,c=dct_c[m],alpha=alpha)
        axs[0,0].set_aspect(1.0/axs[0,0].get_data_ratio(), adjustable='box')
        axs[0,0].set_xlabel("early "+str(keys[vals==leaves[0]][0]))
        axs[0,0].set_ylabel("early "+str(keys[vals==leaves[1]][0]))

        for m in obscol:
            axs[1,0].scatter(X.loc[miles.index[miles==m],late_1].mean(axis=1),
                        X.loc[miles.index[miles==m],late_2].mean(axis=1),
                             s=marker_size, c=dct_c[m],alpha=alpha)
        axs[1,0].set_aspect(1.0/axs[1,0].get_data_ratio(), adjustable='box')
        axs[1,0].set_xlabel("late "+str(keys[vals==leaves[0]][0]))
        axs[1,0].set_ylabel("late "+str(keys[vals==leaves[1]][0]))

        axs[0,1].scatter(X.loc[:,early_1].mean(axis=1),
                    X.loc[:,early_2].mean(axis=1),c=adata.obs.t[X.index],
                         s=marker_size,alpha=alpha,cmap=cmap_pseudotime)
        axs[0,1].set_aspect(1.0/axs[0,1].get_data_ratio(), adjustable='box')
        axs[0,1].set_xlabel("early "+str(keys[vals==leaves[0]][0]))
        axs[0,1].set_ylabel("early "+str(keys[vals==leaves[1]][0]))


        axs[1,1].scatter(X.loc[:,late_1].mean(axis=1),
                    X.loc[:,late_2].mean(axis=1),c=adata.obs.t[X.index],
                         s=marker_size,alpha=alpha,cmap=cmap_pseudotime)
        axs[1,1].set_aspect(1.0/axs[1,1].get_data_ratio(), adjustable='box')
        axs[1,1].set_xlabel("late "+str(keys[vals==leaves[0]][0]))
        axs[1,1].set_ylabel("late "+str(keys[vals==leaves[1]][0]))
        plt.tight_layout()

        fig.set_figheight(10)
        fig.set_figwidth(10)
        
    if mode=="3d":
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for m in obscol:
            ax.scatter(xs=X.loc[miles.index[miles==m],early_1].mean(axis=1),
                                ys=X.loc[miles.index[miles==m],early_2].mean(axis=1),
                        zs=adata.obs.t[miles.index[miles==m]],
                        c=dct_c[m],alpha=alpha,s=10)

        ax.invert_xaxis() 
        ax.view_init(incl_3d,rot_3d)
        plt.xlabel("early "+str(keys[vals==leaves[0]][0]))
        plt.ylabel("early "+str(keys[vals==leaves[1]][0]))
        ax.set_zlabel("pseudotime")

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for m in obscol:
            ax.scatter(xs=X.loc[miles.index[miles==m],late_1].mean(axis=1),
                                ys=X.loc[miles.index[miles==m],late_2].mean(axis=1),
                        zs=adata.obs.t[miles.index[miles==m]],
                        c=dct_c[m],alpha=alpha,s=10)

        ax.invert_xaxis() 
        ax.view_init(incl_3d, rot_3d)
        plt.xlabel("late "+str(keys[vals==leaves[0]][0]))
        plt.ylabel("late "+str(keys[vals==leaves[1]][0]))
        ax.set_zlabel("pseudotime")
    
    adata.uns=uns_temp
    
    savefig_or_show('modules', show=show, save=save)
