import igraph
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def milestones(adata, color=None, cmap=None, roots=None, figsize=(500, 500)):
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
        select milestones to position on top fo the plot
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

    img.vs["label"] = adata.obs.milestones.cat.categories.tolist()

    dct = dict(zip(img.vs["name"], img.vs["label"]))
    if roots is None:
        if "root2" not in adata.uns["graph"]:
            roots = [dct[str(adata.uns["graph"]["root"])]]
        else:
            roots = [
                dct[str(adata.uns["graph"]["root"])],
                dct[str(adata.uns["graph"]["root2"])],
            ]

    layout = img.layout_reingold_tilford(
        root=list(
            map(
                lambda root: np.argwhere(np.array(img.vs["label"]) == root)[0][0], roots
            )
        )
    )

    if color is None:
        if "milestones_colors" not in adata.uns:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")
        img.vs["color"] = adata.uns["milestones_colors"]
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

    return igraph.plot(img, bbox=figsize, layout=layout, margin=50)
