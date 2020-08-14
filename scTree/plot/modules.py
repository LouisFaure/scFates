import numpy as np
import pandas as pd
from anndata import AnnData
import warnings

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from scipy import sparse 
import warnings

from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show
from . import palette_tools

def modules(
    adata: AnnData,
    root_milestone,
    milestones,
    color: str = "milestones",
    basis: str = "umap",
    alpha: float = 1,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None):
    
    plt.rcParams["axes.grid"] = False
    tree=adata.uns["tree"]
    
    uns_temp=deepcopy(adata.uns)
    
    dct = dict(zip(adata.copy().obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))
                   
    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    
    stats = adata.uns[name]["fork"]
    mlsc = deepcopy(adata.uns["milestones_colors"])
    mls = adata.obs.milestones.cat.categories.tolist()
    dct = dict(zip(mls,mlsc))
    df = adata.obs.copy(deep=True)
    edges=tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    
    cells=np.unique(np.concatenate([getpath(img,root,adata.uns["tree"]["tips"],leaves[0],tree,df).index,
                   getpath(img,root,adata.uns["tree"]["tips"],leaves[1],tree,df).index]))

    
    cols = deepcopy(adata.uns[color+"_colors"])
    obscol = adata.obs[color].cat.categories.tolist()
    dct_c = dict(zip(obscol,cols))
    
    if sparse.issparse(adata.X):
        X=pd.DataFrame(np.array(adata[cells,stats.index].X.A),index=cells,columns=stats.index)
    else:
        X=pd.DataFrame(np.array(adata[cells,stats.index].X),index=cells,columns=stats.index)
    miles=adata.obs.loc[X.index,color].astype(str)

    early_1=(stats.branch.values==str(keys[vals==leaves[0]][0])) & (stats.module.values=="early")
    late_1=(stats.branch.values==str(keys[vals==leaves[0]][0])) & (stats.module.values=="late")

    early_2=(stats.branch.values==str(keys[vals==leaves[1]][0])) & (stats.module.values=="early")
    late_2=(stats.branch.values==str(keys[vals==leaves[1]][0])) & (stats.module.values=="late")

    fig, axs = plt.subplots(2,2)

    for m in obscol:
        axs[0,0].scatter(X.loc[miles.index[miles==m],early_1].mean(axis=1),
                    X.loc[miles.index[miles==m],early_2].mean(axis=1),c=dct_c[m],alpha=alpha)
    axs[0,0].set_aspect(1.0/axs[0,0].get_data_ratio(), adjustable='box')
    axs[0,0].set_xlabel("early "+str(keys[vals==leaves[0]][0]))
    axs[0,0].set_ylabel("early "+str(keys[vals==leaves[1]][0]))

    for m in obscol:
        axs[1,0].scatter(X.loc[miles.index[miles==m],late_1].mean(axis=1),
                    X.loc[miles.index[miles==m],late_2].mean(axis=1),c=dct_c[m],alpha=alpha)
    axs[1,0].set_aspect(1.0/axs[1,0].get_data_ratio(), adjustable='box')
    axs[1,0].set_xlabel("late "+str(keys[vals==leaves[0]][0]))
    axs[1,0].set_ylabel("late "+str(keys[vals==leaves[1]][0]))

    axs[0,1].scatter(X.loc[:,early_1].mean(axis=1),
                X.loc[:,early_2].mean(axis=1),c=adata.obs.t[X.index],alpha=alpha)
    axs[0,1].set_aspect(1.0/axs[0,1].get_data_ratio(), adjustable='box')
    axs[0,1].set_xlabel("early "+str(keys[vals==leaves[0]][0]))
    axs[0,1].set_ylabel("early "+str(keys[vals==leaves[1]][0]))


    axs[1,1].scatter(X.loc[:,late_1].mean(axis=1),
                X.loc[:,late_2].mean(axis=1),c=adata.obs.t[X.index],alpha=alpha)
    axs[1,1].set_aspect(1.0/axs[1,1].get_data_ratio(), adjustable='box')
    axs[1,1].set_xlabel("late "+str(keys[vals==leaves[0]][0]))
    axs[1,1].set_ylabel("late "+str(keys[vals==leaves[1]][0]))
    plt.tight_layout()

    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    adata.uns=uns_temp
    
    savefig_or_show('tree', show=show, save=save)


    
def getpath(g,root,tips,tip,tree,df):
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
        warnings.filterwarnings("default")
        return(pth)
    except IndexError:
        pass