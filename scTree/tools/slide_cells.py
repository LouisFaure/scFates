import numpy as np
import pandas as pd
import igraph
from anndata import AnnData

from .. import logging as logg
from .. import settings

def slide_cells(
    adata: AnnData,
    root,
    leaves,
    win: int = 50,
    mapping: bool = True,
    copy: bool = False,
    ):
    
    adata = adata.copy() if copy else adata
    
    tree = adata.uns["tree"]
    
    def getsegs(g,root,leave,tree):
        path=np.array(g.vs[:]["name"])[np.array(g.get_shortest_paths(str(root),str(leave)))][0]
        segs = list()
        for i in range(len(path)-1):
            segs= segs + [np.argwhere((tree["pp_seg"][["from","to"]].astype(str).apply(lambda x: 
                                                                                    all(x.values == path[[i,i+1]]),axis=1)).to_numpy())[0][0]]
        segs=tree["pp_seg"].index[segs].tolist()
        return(segs)


    edges=tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)

    paths = list(map(lambda l: getsegs(img,root,l,tree),leaves))
    
    
    seg_progenies = list(set.intersection(*[set(path) for path in paths]))
    seg_branch1 = list(set.difference(set(paths[0]),set(seg_progenies)))
    seg_branch2 = list(set.difference(set(paths[1]),set(seg_progenies)))
    pp_probs = tree["R"].sum(axis=0)
    pps = tree["pp_info"].PP[tree["pp_info"].seg.isin(np.array(paths).flatten().astype(str))].index

    seg_branch1 = [str(seg) for seg in seg_branch1]
    seg_branch2 = [str(seg) for seg in seg_branch2]
    seg_progenies = [str(seg) for seg in seg_progenies]

    
    def region_extract(pt_cur,segs_cur):
        freq = list()

        pp_next = pps[(tree["pp_info"].loc[pps,"time"].values >= pt_cur) & 
                      tree["pp_info"].loc[pps,"seg"].isin(segs_cur).values]


        cmsm = np.cumsum(pp_probs[pp_next][np.argsort(tree["pp_info"].loc[pp_next,"time"].values)])
        inds=np.argwhere(cmsm > win).flatten()

        if len(inds)==0:
            if (cmsm.max() > win/2):
                if mapping:
                    cell_probs = tree["R"][:,pp_next].sum(axis=1)
                else:
                    cell_probs = np.isin(np.apply_along_axis(lambda x: np.argmax(x),axis=1,arr=tree["R"]),pp_next)*1
                freq = freq+[cell_probs]
            return freq
        else: 
            pps_region = pp_next[np.argsort(tree["pp_info"].loc[pp_next,"time"].values)][:inds[0]]
            if mapping:
                cell_probs = tree["R"][:,pps_region].sum(axis=1)
            else:
                cell_probs = np.isin(np.apply_along_axis(lambda x: np.argmax(x),axis=1,arr=tree["R"]),pps_region)*1

            freq = freq+[cell_probs]
            pt_cur = tree["pp_info"].loc[pps_region,"time"].max()



            if (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_progenies))==0):
                res = region_extract(pt_cur,segs_cur)
                return freq+res

            elif (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_branch1))==0):
                res = region_extract(pt_cur,segs_cur)
                return freq+res

            elif (sum(~tree["pp_info"].loc[pps_region,:].seg.isin(seg_branch2))==0):

                res = region_extract(pt_cur,segs_cur)
                return freq+res


            elif (~(sum(~tree["pp_info"].loc[pps_region,:].seg.isin([str(seg) for seg in seg_progenies]))==0)):
                pt_cur1 = tree["pp_info"].loc[pps_region,"time"][tree["pp_info"].loc[pps_region,"seg"].isin([str(seg) for seg in seg_branch1])].max()
                segs_cur1 = seg_branch1
                pt_cur2 = tree["pp_info"].loc[pps_region,"time"][tree["pp_info"].loc[pps_region,"seg"].isin([str(seg) for seg in seg_branch2])].max()
                segs_cur2 =seg_branch2
                res1 = region_extract(pt_cur1,segs_cur1)
                res2 = region_extract(pt_cur2,segs_cur2)
                return freq+res1+res2
            
    pt_cur = tree["pp_info"].loc[pps,"time"].min()
    segs_cur=np.unique(np.array(paths).flatten().astype(str))
    
    freq=region_extract(pt_cur,segs_cur)
    name=str(root)+"to"+str(leaves[0])+"vs"+str(leaves[1])
    adata.uns["tree"][name+"_cell_freq"]=freq
    
    logg.hint(
        "added \n"
        "    'tree/"+name+"_cell_freq', probability assignment of cells on non intersecting windows (adata.uns)")
    
    return adata if copy else None