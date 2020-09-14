import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec

import warnings
from . import palette_tools
from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show

def cluster(
    adata: AnnData,
    clu = None,
    genes = None,
    combi=True,
    root_milestone = None,
    milestones = None,
    cell_size=20,
    thre_ord=.7,
    figsize: tuple = (20,12),
    basis: str = "umap",
    colormap: str = "magma",
    emb_back = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None):
    
    
    #clusters = pd.Series(adata.uns["tree"]["fit_clusters"])
    
    
    fitted = pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names).T.copy(deep=True)
    g = adata.obs.groupby('seg')
    seg_order=g.apply(lambda x: np.mean(x.t)).sort_values().index.tolist()
    cell_order=np.concatenate(list(map(lambda x: adata.obs.t[adata.obs.seg==x].sort_values().index,seg_order)))
    fitted=fitted.loc[:,cell_order]
    fitted=fitted.apply(lambda x: (x-x.mean())/x.std(),axis=1)

    seg=adata.obs["seg"].copy(deep=True)
    
    
    color_key = "seg_colors"
    if color_key not in adata.uns or len(adata.uns[color_key]):
        palette_tools._set_default_colors_for_categorical_obs(adata,"seg")
    pal=dict(zip(adata.obs["seg"].cat.categories,adata.uns[color_key]))

    segs = seg.astype(str).map(pal)
    segs.name = "segs"


    # Get the color map by name:
    cm = plt.get_cmap('viridis')

    pseudotime = cm(adata.obs.t[cell_order]/adata.obs.t[cell_order].max())

    pseudotime = list(map(to_hex,pseudotime.tolist()))

    col_colors=pd.concat([segs,pd.Series(pseudotime,name="pseudotime",index=cell_order)],axis=1,sort=False)
    
    
    def get_in_clus_order(c):
        test=fitted.loc[clusters.index[clusters==c],:]
        start_cell=adata.obs_names[adata.obs.t==adata.obs.loc[test.idxmax(axis=1).values,"t"].sort_values().iloc[0]]
        early_gene=test.index[test.idxmax(axis=1).isin(start_cell)][0]

        ix = test.T.corr(method="pearson").sort_values(early_gene, ascending=False).index
        return ix
        


    if clu is not None:
        clusters = adata.var["fit_clusters"]
        fitted=fitted.loc[clusters.index[clusters==clu],:]
        fitted_sorted = fitted.loc[get_in_clus_order(clu), :]
    else:
        fitted=fitted.loc[genes,:]
        
        #start_cell=adata.obs_names[adata.obs.t==adata.obs.loc[fitted.idxmax(axis=1).values,"t"].sort_values().iloc[0]]
        #early_gene=fitted.index[fitted.idxmax(axis=1).isin(start_cell)][0]
        
        fitted_norm=fitted.apply(lambda x: (x-np.min(x))/np.max(x-np.min(x)),axis=1)
        early_gene=fitted_norm.index[fitted_norm.apply(lambda x: adata.obs.t[x>.5].mean(),axis=1).argmin()]
        late_gene=fitted_norm.index[fitted_norm.apply(lambda x: adata.obs.t[x>.5].mean(),axis=1).argmax()]
        
        ix = fitted.T.corr(method="pearson").sort_values(early_gene, ascending=False).index
        
        ix = fitted_norm.apply(lambda x: adata.obs.t[x>thre_ord].mean(),axis=1).sort_values().index
        
        fitted_sorted = fitted.loc[ix, :]
        
#         df_f=pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names)
#         #df_f=df_f.apply(lambda x: (x-x.mean())/x.std(),axis=1)
#         fitted_sorted=df_f.iloc[adata.obs.t.argsort().values,adata.obs.t[df_f.idxmax(axis=0).values].argsort().values].T
#         fitted_sorted=fitted_sorted.apply(lambda x: (x-x.mean())/x.std(),axis=1)
        #combi=False
    
    
    if root_milestone is not None:
        dct = dict(zip(adata.copy().obs.milestones.cat.categories.tolist(),
                   np.unique(adata.uns["tree"]["pp_seg"][["from","to"]].values.flatten().astype(int))))
        keys = np.array(list(dct.keys()))
        vals = np.array(list(dct.values()))

        leaves = list(map(lambda leave: dct[leave],milestones))
        root = dct[root_milestone]
        df = adata.obs.copy(deep=True)
        edges=adata.uns["tree"]["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
        img = igraph.Graph()
        img.add_vertices(np.unique(adata.uns["tree"]["pp_seg"][["from","to"]].values.flatten().astype(str)))
        img.add_edges(edges)
        cells=np.unique(np.concatenate([getpath(img,root,adata.uns["tree"]["tips"],leaves[0],adata.uns["tree"],df).index,
                       getpath(img,root,adata.uns["tree"]["tips"],leaves[1],adata.uns["tree"],df).index]))

        col_colors=col_colors[col_colors.index.isin(cells)]
        fitted_sorted=fitted_sorted.loc[:,fitted_sorted.columns.isin(cells)]
        
    else:
        cells=None
    
    

    hm=sns.clustermap(fitted_sorted,figsize=figsize,dendrogram_ratio=0, colors_ratio=0.03,robust=True,cmap=colormap,
                row_cluster=False,col_cluster=False,col_colors=col_colors,cbar_pos=None,xticklabels=False)
    if combi:
        hm.gs.update(left=0.526)
        gs2 = GridSpec(1,1, left=0.05,right=0.50)
        ax2 = hm.fig.add_subplot(gs2[0])
        
        if emb_back is not None:
            ax2.scatter(emb_back[:,0],emb_back[:,1],s=cell_size,color="lightgrey")
        
        if cells is not None:
            ax2.scatter(adata.obsm["X_"+basis][~adata.obs_names.isin(cells),0],
                        adata.obsm["X_"+basis][~adata.obs_names.isin(cells),1],
                        c="lightgrey",s=cell_size)
            ax2.scatter(adata.obsm["X_"+basis][adata.obs_names.isin(cells),0],
                        adata.obsm["X_"+basis][adata.obs_names.isin(cells),1],
                        c="black",s=cell_size*2)
            ax2.scatter(adata.obsm["X_"+basis][adata.obs_names.isin(cells),0],
                        adata.obsm["X_"+basis][adata.obs_names.isin(cells),1],
                        s=cell_size,
                        c=fitted.mean(axis=0)[adata[adata.obs_names.isin(cells),:].obs_names],cmap=colormap)
        else:
            cells = adata.obs_names
            ax2.scatter(adata.obsm["X_"+basis][:,0],
                        adata.obsm["X_"+basis][:,1],
                        s=cell_size,
                        c=fitted.mean(axis=0)[adata[adata.obs_names.isin(cells),:].obs_names],cmap=colormap)
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
    
    savefig_or_show('cluster', show=show, save=save)
            
            
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