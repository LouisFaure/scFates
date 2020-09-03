import igraph
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def milestones(adata,
               color=None,
               cmap=None,
               roots=None):

    tree = adata.uns["tree"]
        
    B=tree["B"]
    R=tree["R"]
    F=tree["F"]
    g=igraph.Graph.Adjacency((B>0).tolist(),mode="undirected")
    tips = np.argwhere(np.array(g.degree())==1).flatten()
    branches = np.argwhere(np.array(g.degree())>2).flatten()
    
    edges=tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph(directed=True)
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)
    
    img.vs["label"] = adata.obs.milestones.cat.categories.tolist()
    
    dct=dict(zip(img.vs["name"],img.vs["label"]))
    if roots is None:
        if "root2" not in adata.uns["tree"]:
            roots=[dct[str(adata.uns["tree"]["root"])]]
        else:
            roots=[dct[str(adata.uns["tree"]["root"])],
                   dct[str(adata.uns["tree"]["root2"])]]
    
    layout=img.layout_reingold_tilford(root=list(map(lambda root: np.argwhere(np.array(img.vs["label"])==root)[0][0],roots)))
    
    if color is None:
        img.vs["color"] = adata.uns["milestones_colors"]
    else:
        if cmap is None:
            cmap="viridis"
        g=adata.obs.groupby("milestones")
        val_milestones=g.apply(lambda x: np.mean(x[color]))
        norm = matplotlib.colors.Normalize(vmin=min(val_milestones), vmax=max(val_milestones), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=eval("cm."+cmap))
        c_mil=list(map(lambda m:mcolors.to_hex(mapper.to_rgba(m)),val_milestones.values))
        img.vs["color"] = c_mil
    
    return igraph.plot(img,bbox=(500,500),layout=layout,margin = 50)