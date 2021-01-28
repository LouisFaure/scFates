import numpy as np
import pandas as pd
from anndata import AnnData

import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from scipy import stats
from adjustText import adjust_text
from matplotlib import patches
from scipy import sparse
import scanpy as sc

import warnings
from typing import Union, Optional
from scanpy.plotting._utils import savefig_or_show

from .. import logging as logg
from ..tools.utils import getpath
    
def trends(
    adata: AnnData,
    features = None,
    highlight_features = None,
    n_features: int = 10,
    root_milestone = None,
    milestones = None,
    show_milestones: bool = True,
    plot_emb: bool = True,
    cell_size = 20,
    cells = None,
    highlight = True,
    fontsize = 11,
    order = True,
    ordering = "pearson",
    ord_thre = .7,
    filter_complex=False,
    complex_thre = .7,
    complex_z = 3,
    fig_heigth = 4,
    frame: Union[float,None] = 2,
    basis: str = "umap",
    colormap: str = "RdBu_r",
    pseudo_colormap: str = "viridis",
    emb_back = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    save_genes: Optional[bool] = None):
    
    graph = adata.uns["graph"]
    
    if root_milestone is not None:
        adata = adata.copy()
        dct = graph["milestones"]
        keys = np.array(list(dct.keys()))
        vals = np.array(list(dct.values()))

        leaves = list(map(lambda leave: dct[leave],milestones))
        root = dct[root_milestone]
        df = adata.obs.copy(deep=True)
        edges=adata.uns["graph"]["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
        img = igraph.Graph()
        img.add_vertices(np.unique(adata.uns["graph"]["pp_seg"][["from","to"]].values.flatten().astype(str)))
        img.add_edges(edges)
        cells=np.unique(np.concatenate(list(map(lambda leave: 
                                                getpath(img,root,
                                                        adata.uns["graph"]["tips"],
                                                        leave,adata.uns["graph"],df).index,
                                                leaves))))

        adata=adata[cells]
        
    
    
    if features is None:
        features = adata.var_names
    
    fitted = pd.DataFrame(adata.layers["fitted"],index=adata.obs_names,columns=adata.var_names).T.copy(deep=True)
    g = adata.obs.groupby('seg')
    seg_order=g.apply(lambda x: np.mean(x.t)).sort_values().index.tolist()
    cell_order=np.concatenate(list(map(lambda x: adata.obs.t[adata.obs.seg==x].sort_values().index,seg_order)))
    fitted=fitted.loc[:,cell_order]
    #fitted=fitted.apply(lambda x: (x-x.mean())/x.std(),axis=1)
    
    fitted=fitted.apply(lambda x: (x-x.min())/(x.max()-x.min()),axis=1)

    if filter_complex:
        # remove complex features using quantiles
        varia = list(map(lambda x: adata.obs.t[cell_order][fitted.loc[x,:].values>np.quantile(fitted.loc[x,:].values,q=complex_thre)].var(),features))
        z = np.abs(stats.zscore(varia))
        torem = np.argwhere(z > complex_z).flatten()

        if len(torem)>0:
            logg.info("found "+str(len(torem))+" complex fitted features")
            logg.hint( "added\n" + "    'complex' column in (adata.var)")
            adata.var["complex"]=False
            #adata.var.iloc[torem,"complex"]=True
            adata.var.loc[fitted.index[torem],"complex"]=True
            features=adata.var_names[~adata.var["complex"]]

    fitted = fitted.loc[features,:]
    
    if order:
        if ordering=="quantile":
            feature_order = fitted.apply(lambda x: adata.obs.t[fitted.columns][x>np.quantile(x,q=ord_thre)].mean(),axis=1).sort_values().index
        elif ordering=="max":
            feature_order = fitted.apply(lambda x: adata.obs.t[fitted.columns][(x-x.min())/(x.max()-x.min())>ord_thre].mean(),axis=1).sort_values().index
        elif ordering in ("pearson", "spearman"):
            start_feature = fitted.apply(lambda x: adata.obs.t[fitted.columns][(x-x.min())/(x.max()-x.min())>ord_thre].mean(),axis=1).sort_values().index[0]
            feature_order = fitted.T.corr(method=ordering).loc[start_feature,:].sort_values(ascending=False).index

        fitted_sorted = fitted.loc[feature_order, :]
    else:
        fitted_sorted = fitted
    
    
    if show_milestones:
        color_key = "milestones_colors"
        if color_key not in adata.uns or len(adata.uns[color_key])==1:
            from . import palette_tools
            palette_tools._set_default_colors_for_categorical_obs(adata,"milestones")

        def milestones_prog(s):
            cfrom=adata.obs.t[adata.obs.seg==s].idxmin()
            cto=adata.obs.t[adata.obs.seg==s].idxmax()
            mfrom=adata.obs.milestones[cfrom]
            mto=adata.obs.milestones[cto]
            import numpy as np
            mfrom_c=adata.uns["milestones_colors"][np.argwhere(adata.obs.milestones.cat.categories==mfrom)[0][0]]
            mto_c=adata.uns["milestones_colors"][np.argwhere(adata.obs.milestones.cat.categories==mto)[0][0]]

            from matplotlib.colors import LinearSegmentedColormap

            cm=LinearSegmentedColormap.from_list("test",[mfrom_c,mto_c],N=1000)
            pst=(adata.obs.t[adata.obs.seg==s]-adata.obs.t[adata.obs.seg==s].min())/(adata.obs.t[adata.obs.seg==s].max()-adata.obs.t[adata.obs.seg==s].min())
            return pd.Series(list(map(to_hex,cm(pst))),index=pst.index)

        mil_cmap=pd.concat(list(map(milestones_prog,seg_order)))
        

    
    fig, f_axs = plt.subplots(ncols=2, nrows=20,figsize=(fig_heigth*(1+1*plot_emb),fig_heigth),
                         gridspec_kw={'width_ratios':[1*plot_emb,1]})
    gs = f_axs[2, 1].get_gridspec()

    # remove the underlying axes
    start = 2 if show_milestones else 1
    for ax in f_axs[start:, -1]:
        ax.remove()
    axheatmap = fig.add_subplot(gs[start:, -1])


    gs = f_axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in f_axs[:, 0]:
        ax.remove()

    if show_milestones:
        axmil = f_axs[0,1]
        sns.heatmap(pd.DataFrame(range(fitted_sorted.shape[1])).T,robust=False,rasterized=True,
                     cmap=mil_cmap[fitted_sorted.columns].values.tolist(),
                     xticklabels=False,yticklabels=False,cbar=False,ax=axmil)
        axpsdt = f_axs[1,1]

    else:
        axpsdt = f_axs[0,1]


    sns.heatmap(pd.DataFrame(adata.obs.t[fitted_sorted.columns].values).T,robust=True,rasterized=True,
                    cmap=pseudo_colormap,xticklabels=False,yticklabels=False,cbar=False,ax=axpsdt,vmax=adata.obs.t.max())

    sns.heatmap(fitted_sorted,robust=True,cmap=colormap,rasterized=True,
                    xticklabels=False,yticklabels=False,ax=axheatmap,cbar=False)

    def add_frames(axis,vert):
        rect = patches.Rectangle((0,0),len(fitted_sorted.columns),vert,linewidth=1,edgecolor='k',facecolor='none')
        # Add the patch to the Axes
        axis.add_patch(rect)
        offset=0
        for s in seg_order[:-1]:
            prev_offset=offset
            offset=offset+(adata.obs.seg==s).sum()
            rect = patches.Rectangle((prev_offset,0),(adata.obs.seg==s).sum(),vert,linewidth=1,edgecolor='k',facecolor='none')
            axis.add_patch(rect)
        return axis

    axpsdt = add_frames(axpsdt,1)

    if show_milestones:
        axmil = add_frames(axmil,1)

    axheatmap=add_frames(axheatmap,fitted_sorted.shape[0])


    if highlight_features is None:
        highlight_features = adata.var.A[features].sort_values(ascending=False)[:n_features].index
    xs=np.repeat(fitted_sorted.shape[1],len(highlight_features))  
    ys=np.array(list(map(lambda g: np.argwhere(fitted_sorted.index==g)[0][0],highlight_features)))+0.5

    texts = []
    for x, y, s in zip(xs, ys, highlight_features):
        texts.append(axheatmap.text(x, y, s, fontsize=fontsize))

    patch = patches.Rectangle((0, 0), fitted_sorted.shape[1]+fitted_sorted.shape[1]/6, fitted_sorted.shape[0], alpha=0) # We add a rectangle to make sure the labels don't move to the right    
    axpsdt.set_xlim((0,fitted_sorted.shape[1]+fitted_sorted.shape[1]/3))
    if show_milestones:
        axmil.set_xlim((0,fitted_sorted.shape[1]+fitted_sorted.shape[1]/3))
    axheatmap.set_xlim((0,fitted_sorted.shape[1]+fitted_sorted.shape[1]/3))
    axheatmap.add_patch(patch)
    axheatmap.hlines(fitted_sorted.shape[0],0,fitted_sorted.shape[1], color="k", clip_on=True) 

    plt.tight_layout(h_pad=0,w_pad=0.05)

    adjust_text(texts,ax=axheatmap,add_objects=[patch],va="center",ha="left",autoalign=False,
        expand_text =(1.05,1.15), lim=5000,only_move={"points":"x", "text":"y", "objects":"x"},
            precision=0.001, expand_points=(1.01, 1.05))

    for i in range(len(xs)):
        xx=[xs[i]+1,fitted_sorted.shape[1]+fitted_sorted.shape[1]/6]
        yy=[ys[i],texts[i].get_position()[1]]
        axheatmap.plot(xx, yy,color="k",linewidth=0.75)


    if plot_emb:
        axemb = fig.add_subplot(gs[:, 0])
        if emb_back is not None:
            axemb.scatter(emb_back[:,0],emb_back[:,1],s=cell_size,color="lightgrey",edgecolor="none")

        if cells is not None:
            axemb.scatter(adata.obsm["X_"+basis][~adata.obs_names.isin(cells),0],
                        adata.obsm["X_"+basis][~adata.obs_names.isin(cells),1],
                        c="lightgrey",s=cell_size,edgecolor="none")
            if highlight:
                axemb.scatter(adata.obsm["X_"+basis][adata.obs_names.isin(cells),0],
                            adata.obsm["X_"+basis][adata.obs_names.isin(cells),1],
                            c="black",s=cell_size*2,edgecolor="none")
            axemb.scatter(adata.obsm["X_"+basis][adata.obs_names.isin(cells),0],
                        adata.obsm["X_"+basis][adata.obs_names.isin(cells),1],
                        s=cell_size,edgecolor="none",
                        c=fitted.mean(axis=0)[adata[adata.obs_names.isin(cells),:].obs_names],cmap=colormap)
        else:
            cells = adata.obs_names
            if highlight:
                axemb.scatter(adata.obsm["X_"+basis][:,0],
                            adata.obsm["X_"+basis][:,1],c="k",s=cell_size*2,edgecolor="none")
            axemb.scatter(adata.obsm["X_"+basis][:,0],
                        adata.obsm["X_"+basis][:,1],
                        s=cell_size,edgecolor="none",
                        c=fitted.mean(axis=0)[adata[adata.obs_names.isin(cells),:].obs_names],cmap=colormap)
        axemb.grid(False)
        x0,x1 = axemb.get_xlim()
        y0,y1 = axemb.get_ylim()
        axemb.set_aspect(abs(x1-x0)/abs(y1-y0))
        axemb.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False) # labels along the bottom edge are off
        axemb.set_xlabel(basis+"1",fontsize=18)
        axemb.set_ylabel(basis+"2",fontsize=18)
        for axis in ['top','bottom','left','right']:
            axemb.spines[axis].set_linewidth(1)
    
    if save_genes is not None:
        with open(save_genes, 'w') as f:
            for item in fitted_sorted.index:
                f.write("%s\n" % item)
    
    savefig_or_show('trends', show=show, save=save)    
    
    
def single_trend(
    adata: AnnData,
    feature: str,
    basis: str = "umap",
    ylab = "expression",
    layer = None,
    colormap: str = "RdBu_r",
    colorexp = None,
    figsize = (10,5.5),
    show_title = True,
    emb_back = None,
    size_cells = None,
    highlight = False,
    alpha_expr = 0.3,
    size_expr = 2,
    fitted_linewidth = 2,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None):
  
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize,constrained_layout=True)
    if show_title:
        fig.suptitle(feature)
    
    color_key = "seg_colors"
    if color_key not in adata.uns:
        from . import palette_tools
        palette_tools._set_default_colors_for_categorical_obs(adata,"seg")
    pal=dict(zip(adata.obs["seg"].cat.categories,adata.uns[color_key]))

    ncells = adata.shape[0]
    
    if size_cells is None:
        size_cells = 30000 / ncells

    if emb_back is not None:
        ncells = emb_back.shape[0]
        if size_cells is None:
            size_cells = 30000 / ncells
        ax1.scatter(emb_back[:,0],emb_back[:,1],s=size_cells,color="lightgrey",edgecolor="none")
        
    if layer is None:
        if sparse.issparse(adata.X):
            Xfeature = adata[:,feature].X.A.T.flatten()
        else:
            Xfeature = adata[:,feature].X.T.flatten()
    else:
        if sparse.issparse(adata.layers[layer]):
            Xfeature = adata[:,feature].layers[layer].A.T.flatten()
        else:
            Xfeature = adata[:,feature].layers[layer].T.flatten()

    df=pd.DataFrame({"t":adata.obs.t,"fitted":adata[:,feature].layers["fitted"].flatten(),"expression":Xfeature,"seg":adata.obs.seg}).sort_values("t")

    if highlight:
        ax1.scatter(adata.obsm["X_"+basis][:,0],adata.obsm["X_"+basis][:,1],c="k",s=size_cells*2,edgecolor="none")

    ax1.scatter(adata.obsm["X_"+basis][:,0],adata.obsm["X_"+basis][:,1],c=df.fitted[adata.obs_names],s=size_cells,cmap=colormap,edgecolor="none")
    ax1.grid(b=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(basis+"1")
    ax1.set_ylabel(basis+"2")
    x0,x1 = ax1.get_xlim()
    y0,y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
    for s in df.seg.unique():
        if colorexp is None:
            plt.scatter(df.loc[df.seg==s,"t"],df.loc[df.seg==s,"expression"],alpha=alpha_expr,s=size_expr,
                        c=adata.uns['seg_colors'][np.argwhere(adata.obs.seg.cat.categories==s)[0][0]])
            plt.plot(df.loc[df.seg==s,"t"],df.loc[df.seg==s,"fitted"],
                     c=adata.uns['seg_colors'][np.argwhere(adata.obs.seg.cat.categories==s)[0][0]],
                     linewidth=fitted_linewidth)
            tolink=adata.uns['graph']['pp_seg'].loc[int(s),"to"]
            for next_s in adata.uns['graph']['pp_seg'].n.iloc[np.argwhere(adata.uns['graph']['pp_seg'].loc[:,'from'].isin([tolink]).values).flatten()]:
                plt.plot([df.loc[df.seg==s,"t"].iloc[-1],df.loc[df.seg==next_s,"t"].iloc[0]],
                     [df.loc[df.seg==s,"fitted"].iloc[-1],df.loc[df.seg==next_s,"fitted"].iloc[0]],
                     c=adata.uns['seg_colors'][np.argwhere(adata.obs.seg.cat.categories==next_s)[0][0]],
                             linewidth=fitted_linewidth)
        else:
            plt.scatter(df.loc[df.seg==s,"t"],df.loc[df.seg==s,"expression"],c=colorexp,alpha=alpha_expr,s=size_expr)
            plt.plot(df.loc[df.seg==s,"t"],df.loc[df.seg==s,"fitted"],
                     c=colorexp,
                     linewidth=fitted_linewidth)
            tolink=adata.uns['graph']['pp_seg'].loc[int(s),"to"]
            for next_s in adata.uns['graph']['pp_seg'].n.iloc[np.argwhere(adata.uns['graph']['pp_seg'].loc[:,'from'].isin([tolink]).values).flatten()]:
                plt.plot([df.loc[df.seg==s,"t"].iloc[-1],df.loc[df.seg==next_s,"t"].iloc[0]],
                     [df.loc[df.seg==s,"fitted"].iloc[-1],df.loc[df.seg==next_s,"fitted"].iloc[0]],
                     c=colorexp,linewidth=fitted_linewidth)


    ax2.set_ylabel(ylab)
    ax2.set_xlabel("pseudotime")
    x0,x1 = ax2.get_xlim()
    y0,y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1-x0)/abs(y1-y0))
    plt.tight_layout()
    
    savefig_or_show('single_trend', show=show, save=save)    
    
    if show is False:
        return (ax1,ax2)
