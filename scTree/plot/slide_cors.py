import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex

from copy import deepcopy

def slide_cors(
    adata: AnnData,
    root_milestone,
    milestones,
    basis: str = "umap"):
    
    tree = adata.uns["tree"]  
    
    uns_temp=deepcopy(adata.uns)
    
    mlsc = deepcopy(adata.uns["milestones_colors"])
    if mlsc.dtype == "float":
        mlsc=list(map(rgb2hex,mlsc))
        
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    bif = adata.uns[name]["fork"]
    freqs = adata.uns[name]["cell_freq"]
    nwin = len(freqs)
    genesetA=bif.index[(bif["branch"]==milestones[0]).values & (bif["module"]=="early").values]
    genesetB=bif.index[(bif["branch"]==milestones[1]).values & (bif["module"]=="early").values]
    corA=adata.uns[name]["corAB"].loc[milestones[0]].copy(deep=True)
    corB=adata.uns[name]["corAB"].loc[milestones[1]].copy(deep=True)
    groupsA=np.ones(corA.shape[0])
    groupsA[corA.index.isin(genesetB)]=2  
    groupsB=np.ones(corA.shape[0])
    groupsB[corA.index.isin(genesetA)]=2
    
    gr=LinearSegmentedColormap.from_list("greyreds",["lightgrey","black"])

    maxlim=np.max([corB.max().max(),np.abs(corB.min().min()),corA.max().max(),np.abs(corA.min().min())])+0.01

    fig, axs = plt.subplots(2,nwin,figsize=(nwin*3, 6))

    fig.subplots_adjust(hspace = .05, wspace=.05)
    emb=adata[adata.uns["tree"]["cells_fitted"],:].obsm["X_"+basis]
    for i in range(nwin):
        freq=freqs[i]
        axs[0,i].scatter(emb[np.argsort(freq),0],emb[np.argsort(freq),1],
                         s=10,c=freq[np.argsort(freq)],cmap=gr)
        axs[0,i].grid(b=None)
        axs[0,i].set_xticks([]) 
        axs[0,i].set_yticks([]) 

    c_mil=np.array(mlsc)[np.argwhere(adata.obs.milestones.cat.categories.isin(milestones))].flatten()    
    genesets=[genesetA,genesetB]
    for i in range(nwin):
        for j in range(2):
            axs[1,i].scatter(corA.loc[genesets[j],str(i)],corB.loc[genesets[j],str(i)],
                             color=c_mil[j],alpha=.5)
        rep=(np.corrcoef(groupsA,corA.iloc[:,i])[0][1]+np.corrcoef(groupsB,corB.iloc[:,i])[0][1])/2
        axs[1,i].annotate(str(round(rep,2)), xy=(0.7,0.88),xycoords='axes fraction',fontsize=16)
        axs[1,i].grid(b=None)
        axs[1,i].axvline(0,linestyle="dashed",color="grey",zorder=0)
        axs[1,i].axhline(0,linestyle="dashed",color="grey",zorder=0)
        axs[1,i].set_xlim([-maxlim,maxlim])
        axs[1,i].set_ylim([-maxlim,maxlim])
        axs[1,i].set_xticks([]) 
        axs[1,i].set_yticks([]) 