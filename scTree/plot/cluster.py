import numpy as np
import pandas as pd
from anndata import AnnData

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec

from . import palette_tools

def cluster(
    adata: AnnData,
    clu: int,
    figsize: tuple = (20,12),
    basis: str = "umap"):
    
    
    
    clusters = pd.Series(adata.uns["tree"]["fit_clusters"])
    
    fitted = adata.uns["tree"]["fit_summary"].T.copy(deep=True)
    g = adata.obs.groupby('seg')
    seg_order=g.apply(lambda x: np.mean(x.t)).sort_values().index.tolist()
    cell_order=np.concatenate(list(map(lambda x: adata.obs.t[adata.obs.seg==x].sort_values().index,seg_order)))
    fitted=adata.uns["tree"]["fit_list"]["0"].T.copy(deep=True)
    fitted=fitted.loc[:,cell_order]
    fitted=fitted.apply(lambda x: (x-x.mean())/x.std(),axis=1)

    segs=adata.obs["seg"].copy(deep=True)
    
    
    color_key = "seg_colors"
    if color_key not in adata.uns or len(adata.uns[color_key]):
        palette_tools._set_default_colors_for_categorical_obs(adata,"seg")
    pal=dict(zip(adata.obs["seg"].cat.categories,adata.uns[color_key]))

    #segments = segs.map(pal)
    segments = segs.astype(str).map(pal)
    segments.name = "segments"
    #segments = segments.astype(str)


    # Get the color map by name:
    cm = plt.get_cmap('viridis')

    pseudotime = cm(adata.obs.t[cell_order]/adata.obs.t[cell_order].max())

    pseudotime=list(map(to_hex,pseudotime.tolist()))

    col_colors=pd.concat([segments,pd.Series(pseudotime,name="pseudotime",index=cell_order)],axis=1,sort=False)

    fitted=fitted.loc[clusters.index[clusters==clu],:]


    hm=sns.clustermap(fitted,figsize=figsize,dendrogram_ratio=0, colors_ratio=0.03,robust=True,
                row_cluster=False,col_cluster=False,col_colors=col_colors,cbar_pos=None,xticklabels=False)
    hm.gs.update(left=0.526)
    gs2 = GridSpec(1,1, left=0.05,right=0.50)
    ax2 = hm.fig.add_subplot(gs2[0])
    ax2.scatter(adata.obsm["X_"+basis][:,0],adata.obsm["X_"+basis][:,1],
                c=fitted.mean(axis=0)[adata.obs_names],cmap="magma")
    ax2.grid(False)
    x0,x1 = ax2.get_xlim()
    y0,y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax2.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False) # labels along the bottom edge are off
    ax2.set_xlabel(basis+"1",fontsize=18)
    ax2.set_ylabel(basis+"2",fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)