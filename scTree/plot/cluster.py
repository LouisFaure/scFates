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
    clu = None,
    combi=True,
    figsize: tuple = (20,12),
    basis: str = "umap",
    colormap: str = "magma"):
    
    
    clusters = pd.Series(adata.uns["tree"]["fit_clusters"])
    
    fitted = pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names).T.copy(deep=True)
    g = adata.obs.groupby('milestones')
    mil_order=g.apply(lambda x: np.mean(x.t)).sort_values().index.tolist()
    cell_order=np.concatenate(list(map(lambda x: adata.obs.t[adata.obs.milestones==x].sort_values().index,mil_order)))
    #fitted=adata.uns["tree"]["fit_list"]["0"].T.copy(deep=True)
    fitted=fitted.loc[:,cell_order]
    fitted=fitted.apply(lambda x: (x-x.mean())/x.std(),axis=1)

    mils=adata.obs["milestones"].copy(deep=True)
    
    
    color_key = "milestones_colors"
    if color_key not in adata.uns or len(adata.uns[color_key]):
        palette_tools._set_default_colors_for_categorical_obs(adata,"milestones")
    pal=dict(zip(adata.obs["milestones"].cat.categories,adata.uns[color_key]))

    #milestones = milestones.map(pal)
    milestones = mils.astype(str).map(pal)
    milestones.name = "milestones"
    #milestones = milestones.astype(str)


    # Get the color map by name:
    cm = plt.get_cmap('viridis')

    pseudotime = cm(adata.obs.t[cell_order]/adata.obs.t[cell_order].max())

    pseudotime = list(map(to_hex,pseudotime.tolist()))

    col_colors=pd.concat([milestones,pd.Series(pseudotime,name="pseudotime",index=cell_order)],axis=1,sort=False)
    
    
    def get_clus_order(c):
        genes_clu=adata.var_names[pd.Series(adata.uns["tree"]["fit_clusters"])==pd.Series(adata.uns["tree"]["fit_clusters"]).unique()[c]]
        fitted_clu=fitted.loc[genes_clu,:]
        mid=(fitted_clu.mean(axis=0).min()+(fitted_clu.mean(axis=0).max()-fitted_clu.mean(axis=0).min())/2)
        return adata.obs.t[(fitted_clu.mean(axis=0)[fitted_clu.mean(axis=0)>mid]-fitted_clu.mean(axis=0)[fitted_clu.mean(axis=0)>mid].median()).abs().idxmin()]
    
    def get_in_clus_order(c):
        test=fitted.loc[clusters.index[clusters==c],:]
        start_cell=adata.obs_names[adata.obs.t==adata.obs.loc[test.idxmax(axis=1).values,"t"].sort_values().iloc[0]]
        early_gene=test.index[test.idxmax(axis=1).isin(start_cell)][0]

        ix = test.T.corr(method="spearman").sort_values(early_gene, ascending=False).index
        return ix
        


    if clu is not None:
        fitted=fitted.loc[clusters.index[clusters==clu],:]
        fitted_sorted = fitted.loc[get_in_clus_order(clu), :]
    else:
        #ordr=list(map(get_clus_order,pd.Series(adata.uns["tree"]["fit_clusters"]).unique()))
        #idxs=np.concatenate(list(map(get_in_clus_order,pd.Series(adata.uns["tree"]["fit_clusters"]).unique()[np.argsort(ordr)])))
        start_cell=adata.obs_names[adata.obs.t==adata.obs.loc[fitted.idxmax(axis=1).values,"t"].sort_values().iloc[0]]
        early_gene=fitted.index[fitted.idxmax(axis=1).isin(start_cell)][0]

        ix = fitted.T.corr(method="spearman").sort_values(early_gene, ascending=False).index
        fitted_sorted = fitted.loc[ix, :]
        
        df_f=pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names)
        #df_f=df_f.apply(lambda x: (x-x.mean())/x.std(),axis=1)
        fitted_sorted=df_f.iloc[adata.obs.t.argsort().values,adata.obs.t[df_f.idxmax(axis=0).values].argsort().values].T
        fitted_sorted=fitted_sorted.apply(lambda x: (x-x.mean())/x.std(),axis=1)
        combi=False
    
    
    
    
    #pd.Series(adata.uns["tree"]["fit_clusters"]).unique()[np.argsort(ordr)]

    hm=sns.clustermap(fitted_sorted,figsize=figsize,dendrogram_ratio=0, colors_ratio=0.03,robust=True,cmap=colormap,
                row_cluster=False,col_cluster=False,col_colors=col_colors,cbar_pos=None,xticklabels=False)
    if combi:
        hm.gs.update(left=0.526)
        gs2 = GridSpec(1,1, left=0.05,right=0.50)
        ax2 = hm.fig.add_subplot(gs2[0])
        ax2.scatter(adata.obsm["X_"+basis][:,0],adata.obsm["X_"+basis][:,1],
                    c=fitted.mean(axis=0)[adata.obs_names],cmap=colormap)
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