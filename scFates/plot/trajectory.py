import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.collections
from typing import Union, Optional
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc

from scanpy.plotting._utils import savefig_or_show
import types
from matplotlib.backend_bases import GraphicsContextBase, RendererBase

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numba import njit
import math

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()

def trajectory(
    adata: AnnData,
    basis: str = "umap",
    emb_back : Union[np.ndarray,None] = None,
    size_nodes: float = None,
    col_traj: bool = False,
    color_cells: Union[str, None] = None,
    alpha_cells: float = 1,
    tips: bool = True,
    forks: bool = True,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None):
    
    """\
    Project trajectory onto embedding.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Name of the `obsm` basis to use.
    emb_back
        Other cells to show in background.
    size_nodes
        size of the projected prinicpal points.
    col_traj
        color trajectory by segments.
    color_cells
        cells color
    alpha_cells
        cells alpha
    tips
        display tip ids.
    forks
        display fork ids.
    show
        show the plot.
    save
        save the plot.
    
    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
    
    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` or `tl.curve` first to compute a princal graph before plotting."
        )
       
    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
    
    graph = adata.uns["graph"]
    
    emb = adata.obsm[f"X_{basis}"]
    emb_f = adata[graph["cells_fitted"],:].obsm[f"X_{basis}"]
    
    R=graph["R"]
    
    proj=(np.dot(emb_f.T,R)/R.sum(axis=0)).T
    
    B=graph["B"]
    
    fig = plt.figure() 
    ax = plt.subplot()
    
    if emb_back is not None:
        ax.scatter(emb_back[:,0],emb_back[:,1],s=2,color="lightgrey",alpha=alpha_cells)
    
    
    ax.scatter(emb[:,0],emb[:,1],s=2,color="grey",alpha=alpha_cells)
    
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
    
    ax.scatter(proj[:,0],proj[:,1],s=size_nodes,c="k")
    
            
    bbox = dict(facecolor='white', alpha=0.6, edgecolor="white", pad=0.1)
    
    if tips:
        for tip in graph["tips"]:
            ax.annotate(tip, (proj[tip,0], proj[tip,1]), ha="center", va="center",
                       xytext=(-8, 8), textcoords='offset points',bbox=bbox)
    if forks:    
        for fork in graph["forks"]:
            ax.annotate(fork, (proj[fork,0], proj[fork,1]), ha="center", va="center",
                       xytext=(-8, 8), textcoords='offset points',bbox=bbox)
            
    savefig_or_show('trajectory', show=show, save=save)
    
    
def trajectory_pseudotime(
    adata: AnnData,
    basis: str = "umap",
    emb_back = None,
    colormap: str = "viridis",
    color_cells = "grey",
    size_cells = None,
    alpha_cells: float = 1,
    scale_path: float = 1,
    arrows: bool = False,
    arrow_offset: int = 10,
    ax = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None):
    
    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.pseudotime` first before plotting."
        )
       
    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
    
    graph = adata.uns["graph"]
    
    emb = adata.obsm[f"X_{basis}"]
    emb_f = adata[graph["cells_fitted"],:].obsm[f"X_{basis}"]
    
    R=graph["R"]
    
    proj=(np.dot(emb_f.T,R)/R.sum(axis=0)).T
    
    B=graph["B"]

    if ax is None:
        fig = plt.figure() 
        ax = plt.subplot()
    
    ncells = adata.shape[0]
    if size_cells is None:
        size_cells = 30000 / ncells

    if emb_back is not None:
        ncells = emb_back.shape[0]
        if size_cells is None:
            size_cells = 30000 / ncells
        ax.scatter(emb_back[:,0],emb_back[:,1],s=size_cells,color="lightgrey",alpha=alpha_cells,edgecolor="none")


    ax.scatter(emb[:,0],emb[:,1],s=size_cells,color=color_cells,alpha=alpha_cells,edgecolor="none")

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
    all_t = pd.Series(list(map(lambda s: graph["pp_info"].time[graph["pp_info"].index.isin(s)].mean().mean(),segs)))


    sm = ScalarMappable(norm=Normalize(vmin=all_t.min(), vmax=all_t.max()), cmap=colormap)
    lc = matplotlib.collections.LineCollection(lines,colors="k",linewidths=7.5*scale_path,zorder=100)
    ax.add_collection(lc)      

    g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
    paths=g.get_shortest_paths(graph["root"],graph["tips"])
    seg=graph["pp_seg"].loc[:,["from","to"]].values.tolist()
    for s in seg:
        if arrows:
            path=np.array(g.get_shortest_paths(int(s[0]),int(s[1]))[0])        
            coord=proj[path,]
            out=np.empty(len(path)-1)
            cdist_numba(coord,out)
            mid=np.argmin(np.abs(out.cumsum()-out.sum()/2))
            if mid+arrow_offset > (len(path)-1):
                arrow_offset = len(path)-1-mid
            ax.quiver(proj[path[mid],0],proj[path[mid],1],
                      proj[path[mid+arrow_offset],0]-proj[path[mid],0],
                      proj[path[mid+arrow_offset],1]-proj[path[mid],1],headwidth=15*scale_path,headaxislength=10*scale_path,headlength=10*scale_path,units="dots",zorder=101)

            ax.quiver(proj[path[mid],0],proj[path[mid],1],
                      proj[path[mid+arrow_offset],0]-proj[path[mid],0],
                      proj[path[mid+arrow_offset],1]-proj[path[mid],1],headwidth=12*scale_path,headaxislength=10*scale_path,headlength=10*scale_path,units="dots",
                      color=sm.to_rgba(graph["pp_info"].loc[path,:].time.iloc[mid]),zorder=102)


    lc = matplotlib.collections.LineCollection(lines,colors=[sm.to_rgba(t) for t in all_t],linewidths=5*scale_path,zorder=104)
    ax.scatter(proj[graph["tips"],0],proj[graph["tips"],1],zorder=103,c="k",s=200*scale_path)
    ax.add_collection(lc)

    ax.scatter(proj[graph["tips"],0],proj[graph["tips"],1],zorder=105,
               c=sm.to_rgba(graph["pp_info"].time.loc[graph["tips"]]),s=140*scale_path)
            
    savefig_or_show('pseudotime', show=show, save=save)
    
@njit()
def cdist_numba(coords,out):
    for i in range(0,coords.shape[0]-1):
        out[i] = math.sqrt((coords[i,0] - coords[i+1,0])**2+(coords[i,1] - coords[i+1,1])**2)
        
        
def scatter3d(emb,col,cell_cex,nm):
    return go.Scatter3d(x=emb[:,0], y=emb[:,1], z=emb[:,2], mode='markers',
                 marker=dict(size=cell_cex,color=col,opacity=0.9),name=nm)

def trajectory_3d(
    adata: AnnData,
    basis: str = "umap3d",
    color: str = None,
    traj_cex: int = 5,
    cell_cex: int = 2,
    figsize: tuple = (900,900),
    cmap = None,
    palette = None):
    
    r = adata.uns["graph"]
    
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
        
        if color+"_colors" not in adata.uns:
            from . import palette_tools
            palette_tools._set_default_colors_for_categorical_obs(adata,color)
            
        if adata.obs[color].dtype.name == "category":
            palette = adata.uns[color+"_colors"]
            pal_dict=dict(zip(adata.obs[color].cat.categories,adata.uns[color+"_colors"]))
            trace1=list(map(lambda x: scatter3d(emb_f[adata.obs[color]==x,:],pal_dict[x],cell_cex,x), 
                            list(pal_dict.keys())))
                
            
            
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
        line=dict(width=traj_cex)
    )]

    fig = go.Figure(data=trace1+trace2)
    fig.update_layout(scene = dict(xaxis = dict(visible=False),
                                   yaxis = dict(visible=False),
                                   zaxis = dict(visible=False)),
                      width=figsize[0],height=figsize[0],
                     margin=dict(l=5, r=5, t=5, b=5))
    fig.show()        
        