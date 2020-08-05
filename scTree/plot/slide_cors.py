import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from copy import deepcopy

def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones,
    basis: str = "umap"):
    
    tree = adata.uns["tree"]  
    
    mlsc = deepcopy(adata.uns["milestones_colors"])
    mlsc_temp = deepcopy(mlsc)
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    bif = adata.uns["tree"][name]
    freq = adata.uns["tree"][name+"-cell_freq"]
    nwin = len(adata.uns["tree"][name+"-cell_freq"])
    genesetA=bif.index[(bif["branch"]==milestones[0]).values & (bif["module"]=="early").values]
    genesetB=bif.index[(bif["branch"]==milestones[1]).values & (bif["module"]=="early").values]
    corA=adata.uns["tree"][name+"-corAB"].loc[milestones[0]].copy(deep=True)
    corB=adata.uns["tree"][name+"-corAB"].loc[milestones[1]].copy(deep=True)
    
    
    gr=LinearSegmentedColormap.from_list("greyreds",["lightgrey","black"])

    maxlim=np.max([corB.max().max(),np.abs(corB.min().min()),corA.max().max(),np.abs(corA.min().min())])+0.01

    fig, axs = plt.subplots(2,nwin,figsize=(nwin*3, 6))

    fig.subplots_adjust(hspace = .05, wspace=.05)
    emb=adata[adata.uns["tree"]["cells_fitted"],:].obsm["X_"+basis]
    for i in range(nwin):
        freq=adata.uns["tree"][name+"-cell_freq"][i]
        axs[0,i].scatter(emb[np.argsort(freq),0],emb[np.argsort(freq),1],
                         s=10,c=freq[np.argsort(freq)],cmap=gr)
        axs[0,i].grid(b=None)
        axs[0,i].set_xticks([]) 
        axs[0,i].set_yticks([]) 

    c_mil=adata.uns["milestones_colors"][np.argwhere(adata.obs.milestones.cat.categories.isin(milestones))].flatten()    
    genesets=[genesetA,genesetB]
    for i in range(nwin):
        for j in range(2):
            axs[1,i].scatter(corA.loc[genesets[j],i],corB.loc[genesets[j],i],
                             color=c_mil[j],alpha=.5)
        axs[1,i].grid(b=None)
        axs[1,i].axvline(0,linestyle="dashed",color="grey",zorder=0)
        axs[1,i].axhline(0,linestyle="dashed",color="grey",zorder=0)
        axs[1,i].set_xlim([-maxlim,maxlim])
        axs[1,i].set_ylim([-maxlim,maxlim])
        axs[1,i].set_xticks([]) 
        axs[1,i].set_yticks([]) 