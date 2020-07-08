import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.collections

import plotly.express as px
import plotly.graph_objects as go

def tree(
    adata: AnnData,
    basis: str = "umap",
    cex_tree: float = None,
    col_tree: bool = False,
    tips: bool = True,
    forks: bool = True):
    
    if "tree" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` first to compute a princal tree before choosing a root."
        )
       
    r = adata.uns["tree"]
    
    emb = adata.obsm[f"X_{basis}"]
    emb_f = adata[r["cells_fitted"],:].obsm[f"X_{basis}"]
    
    R=r["R"]
    
    proj=(np.dot(emb_f.T,R)/R.sum(axis=0)).T
    
    B=r["B"]
    
    fig = plt.figure() 
    ax = plt.subplot()
    ax.scatter(emb[:,0],emb[:,1],s=2,color="grey")
    
    ax.grid(False)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False) # labels along the bottom edge are off
    
    ax.set_xlabel(basis+"1")
    ax.set_ylabel(basis+"2")
    
    al = np.array(igraph.Graph.Adjacency((B>0).tolist(),mode="undirected").get_edgelist())
    segs=al.tolist()
    vertices=proj.tolist()
    lines = [[tuple(vertices[j]) for j in i]for i in segs]
    lc = matplotlib.collections.LineCollection(lines,colors="k",linewidths=2)
    ax.add_collection(lc)
    
    ax.scatter(proj[:,0],proj[:,1],s=cex_tree,c="k")
    
    if col_tree==True:
        for seg in np.unique(r["pp_info"]["seg"]):
            subproj=proj[r["pp_info"]["seg"]==seg,:]
            ax.scatter(subproj[:,0],subproj[:,1],zorder=2)
            
    bbox = dict(facecolor='white', alpha=0.6, edgecolor="white", pad=0.1)
    
    if tips:
        for tip in r["tips"]:
            ax.annotate(tip, (proj[tip,0], proj[tip,1]), ha="center", va="center",
                       xytext=(-8, 8), textcoords='offset points',bbox=bbox)
    if forks:    
        for fork in r["forks"]:
            ax.annotate(fork, (proj[fork,0], proj[fork,1]), ha="center", va="center",
                       xytext=(-8, 8), textcoords='offset points',bbox=bbox)      
        
        
        
def scatter3d(emb,col,cell_cex,nm):
    return go.Scatter3d(x=emb[:,0], y=emb[:,1], z=emb[:,2], mode='markers',
                 marker=dict(size=cell_cex,color=col,opacity=0.9),name=nm)

def tree_3d(
    adata: AnnData,
    basis: str = "umap3d",
    color: str = None,
    tree_cex: int = 5,
    cell_cex: int = 2,
    figsize: tuple = (900,900),
    cmap = None,
    palette = None):
    
    r = adata.uns["tree"]
    
    emb = adata.obsm[f"X_{basis}"]
    if emb.shape[1]>3:
        raise ValueError(
            "Embedding is not three dimensional."
        )
    
    emb_f = adata[r["cells_fitted"],:].obsm[f"X_{basis}"]
    
    
    
    R=r["R"]
    proj=(np.dot(emb_f.T,R)/R.sum(axis=0)).T

    B=r["B"]

    al = np.array(igraph.Graph.Adjacency((B>0).tolist(),mode="undirected").get_edgelist())
    segs=al.tolist()
    vertices=proj.tolist()
    vertices=np.array(vertices)
    segs=np.array(segs)

    x_lines = list()
    y_lines = list()
    z_lines = list()

    x=vertices[:,0]
    y=vertices[:,1]
    z=vertices[:,2]
    

    for i in range(segs.shape[0]):
        p = segs[i,:]
        for i in range(2):
            x_lines.append(x[p[i]])
            y_lines.append(y[p[i]])
            z_lines.append(z[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    
    
    if color is not None:
        if adata.obs[color].dtype.name == "str":
            adata.obs[color]=adata.obs[color].astype("category")
        if adata.obs[color].dtype.name == "category":
            if adata.uns[color+"_colors"] is not None:
                palette = adata.uns[color+"_colors"]
                pal_dict=dict(zip(adata.obs["leiden"].cat.categories,adata.uns["leiden_colors"]))
                trace1=list(map(lambda x: scatter3d(emb_f[adata.obs[color]==x,:],pal_dict[x],cell_cex,x), 
                                list(pal_dict.keys())))
                
            else:
                trace1=list(map(lambda x: scatter3d(emb_f[adata.obs[color]==x,:],None,cell_cex,x), 
                                np.unique(adata.obs[color]).tolist()))
            
        else:
            if cmap is None:
                cmap="Viridis"
            trace1 = [go.Scatter3d(
                    x=emb_f[:,0],
                    y=emb_f[:,1],
                    z=emb_f[:,2],
                    mode='markers',
                    marker=dict(size=cell_cex,
                                color=adata.obs[color],
                                colorscale=cmap,
                                opacity=0.9
                    ))]
                
    
    else:
        trace1 = [go.Scatter3d(
                    x=emb_f[:,0],
                    y=emb_f[:,1],
                    z=emb_f[:,2],
                    mode='markers',
                    marker=dict(size=cell_cex,
                                color="grey",
                                opacity=0.9
                    ))]

    trace2 = [go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='lines',
        line=dict(width=tree_cex)
    )]

    fig = go.Figure(data=trace1+trace2)
    fig.update_layout(scene = dict(xaxis = dict(visible=False),
                                   yaxis = dict(visible=False),
                                   zaxis = dict(visible=False)),
                      width=figsize[0],height=figsize[0],
                     margin=dict(l=5, r=5, t=5, b=5))
    fig.show()        
        