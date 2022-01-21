from typing_extensions import Literal
import igraph
import numpy as np
import pandas as pd
from typing import Union
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..tools.dendrogram import hierarchy_pos
from .. import logging as logg


def milestones(
    adata,
    color=None,
    cmap=None,
    roots=None,
    layout: Literal["dendro", "reingold_tilford"] = "dendro",
    figsize=(500, 500),
):
    """\
    Display the milestone graph in PAGA style.

    Parameters
    ----------
    adata
        Annotated data matrix.
    color
        color the milestones with variable from adata.obs.
    cmap
        colormap to use for the node coloring.
    roots
        select milestones to position on top fo the plot.
    dendro
        generate layout following dendrogram representation.
    figsize
        figure size in pixels

    Returns
    -------
    igraph.plot

    """

    graph = adata.uns["graph"]

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph(directed=True)
    img.add_vertices(vals.astype(str))
    img.add_edges(edges)

    img.vs["label"] = keys

    dct = dict(zip(img.vs["name"], img.vs["label"]))
    if roots is None:
        if "root2" not in adata.uns["graph"]:
            roots = [dct[str(adata.uns["graph"]["root"])]]
        else:
            roots = [
                dct[str(adata.uns["graph"]["root"])],
                dct[str(adata.uns["graph"]["root2"])],
            ]
            if layout == "dendro":
                logg.warn("two roots detected, reverting to reingold_tilford layout")
                layout = "reingold_tilford"

    if color is None:
        if "milestones_colors" not in adata.uns:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")
        img.vs["color"] = [
            np.array(adata.uns["milestones_colors"])[
                adata.obs.milestones.cat.categories == dct[m]
            ][0]
            for m in img.vs["name"]
        ]
    else:
        if cmap is None:
            cmap = "viridis"
        g = adata.obs.groupby("milestones")
        val_milestones = g.apply(lambda x: np.mean(x[color]))
        norm = matplotlib.colors.Normalize(
            vmin=min(val_milestones), vmax=max(val_milestones), clip=True
        )
        mapper = cm.ScalarMappable(norm=norm, cmap=eval("cm." + cmap))
        c_mil = list(
            map(lambda m: mcolors.to_hex(mapper.to_rgba(m)), val_milestones.values)
        )
        img.vs["color"] = c_mil

    if layout == "dendro":
        pos = hierarchy_pos(img.to_networkx(), vert_gap=0.1)
        pos = pd.DataFrame([pos[s] for s in np.sort(np.array(list(pos.keys())))])
        pos.iloc[:, 1] = -pos.iloc[:, 1]
        layout = list(pos.to_records(index=False))
    else:
        layout = img.layout_reingold_tilford(
            root=list(
                map(
                    lambda root: np.argwhere(np.array(img.vs["label"]) == root)[0][0],
                    roots,
                )
            )
        )

    return igraph.plot(img, bbox=figsize, layout=layout, margin=50)
