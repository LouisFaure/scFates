import numpy as np
import pandas as pd
from anndata import AnnData
import igraph

import matplotlib.pyplot as plt
from skmisc.loess import loess
from matplotlib import lines


def synchro_path(
    adata: AnnData,
    root_milestone,
    milestones):
    
    plt.rcParams["axes.grid"] = False
    
    tree = adata.uns["tree"]
    
    edges = tree["pp_seg"][["from","to"]].astype(str).apply(tuple,axis=1).values
    img = igraph.Graph()
    img.add_vertices(np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(str)))
    img.add_edges(edges)  
    
    mlsc = adata.uns["milestones_colors"].copy()
    mlsc_temp = mlsc.copy()
    dct = dict(zip(adata.obs.milestones.cat.categories.tolist(),
                   np.unique(tree["pp_seg"][["from","to"]].values.flatten().astype(int))))
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    leaves=list(map(lambda leave: dct[leave],milestones))
    root=dct[root_milestone]
    
    name=str(keys[vals==root][0])+"->"+str(keys[vals==leaves[0]][0])+"<>"+str(keys[vals==leaves[1]][0])
    
    
    fork=list(set(img.get_shortest_paths(str(root),str(leaves[0]))[0]).intersection(img.get_shortest_paths(str(root),str(leaves[1]))[0]))
    fork=np.array(img.vs["name"],dtype=int)[fork]
    fork_t=adata.uns["tree"]["pp_info"].loc[fork,"time"].max()

    allcor=adata.uns[name]["synchro"]
    runs=pd.DataFrame(allcor.to_records())["level_0"].unique()
    
    fig, axs = plt.subplots(3,len(runs),figsize=(len(runs)*6, 6))
    fig.subplots_adjust(hspace = .05,wspace = .025, top=0.95)
    i=0

    dct_cormil=dict(zip(["corAA","corBB","corAB"],
                        [milestones[0]+"\nintra-module",
                         milestones[1]+"\nintra-module"]+[milestones[0]+" vs "+milestones[1]+"\ninter-module"]))

    axs=axs.ravel(order="F")

    for r in range(len(runs)):
        for cc in ["corAA","corBB","corAB"]:
            for mil in milestones:
                res=allcor.loc[runs[r]].loc[mil]
                l = loess(res.t, res[cc])
                l.fit()
                pred = l.predict(res.t, stderror=True)
                conf = pred.confidence()

                lowess = pred.values
                ll = conf.lower
                ul = conf.upper

                axs[i].plot(res.t.values, res[cc], '+',c=mlsc[adata.obs.milestones.cat.categories==mil][0],alpha=.5)
                axs[i].plot(res.t.values, lowess,c=mlsc[adata.obs.milestones.cat.categories==mil][0])
                axs[i].fill_between(res.t.values.tolist(),ll.tolist(),ul.tolist(),alpha=.33)
                axs[i].axvline(fork_t,color="black")
                axs[i].axhline(0,linestyle="dashed",color="grey",zorder=0)


                if i==0:
                    vertical_line = lines.Line2D([], [], color='black', marker='|', linestyle='None',
                                      markersize=10, markeredgewidth=1.5, label='bifurcation')

                    axs[i].legend(handles = [vertical_line])

                if (i<2) | ((i>2)&(i<5)):
                    axs[i].set_xticks([])
                else:
                    axs[i].set_xlabel("pseudotime")

                if i<=2:
                    axs[i].set_ylabel(dct_cormil[cc])

                if i>2:
                    axs[i].set_ylim(axs[i-3].get_ylim())
                    axs[i].set_yticks([])

                if i==0:
                    axs[i].set_title("real")

                if i==3:
                    axs[i].set_title("permuted")
                    
                axs[i].grid(b=None)


            i=i+1
