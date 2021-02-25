import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.collections
from typing import Union, Optional, Sequence, Tuple
import plotly.graph_objects as go
import scanpy as sc
from cycler import Cycler

from pandas.api.types import is_categorical_dtype
from scanpy.plotting._utils import savefig_or_show
import types

from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numba import njit
import math


def graph(
    adata: AnnData,
    basis: str = "umap",
    size_nodes: float = None,
    color_cells: Union[str, None] = None,
    tips: bool = True,
    forks: bool = True,
    ax=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):

    """\
    Project principal graph onto embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Name of the `obsm` basis to use.
    size_nodes
        size of the projected prinicpal points.
    color_cells
        cells color
    tips
        display tip ids.
    forks
        display fork ids.
    ax
        Add plot to existing ax
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to scanpy functions pl.embedding

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`

    """

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` or `tl.curve` first to compute a princal graph before plotting."
        )

    graph = adata.uns["graph"]

    emb = adata.obsm[f"X_{basis}"]
    emb_f = adata[graph["cells_fitted"], :].obsm[f"X_{basis}"]

    if "components" in kwargs:
        cmp = np.array(kwargs["components"]) - 1
        emb_f = emb_f[:, cmp]

    else:
        emb_f = emb_f[:, :2]

    R = graph["R"]

    proj = (np.dot(emb_f.T, R) / R.sum(axis=0)).T

    B = graph["B"]

    if ax is None:
        ax = sc.pl.embedding(
            adata, color=color_cells, basis=basis, show=False, **kwargs
        )
    else:
        sc.pl.embedding(
            adata, color=color_cells, basis=basis, ax=ax, show=False, **kwargs
        )

    al = np.array(
        igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected").get_edgelist()
    )
    segs = al.tolist()
    vertices = proj.tolist()
    lines = [[tuple(vertices[j]) for j in i] for i in segs]
    lc = matplotlib.collections.LineCollection(lines, colors="k", linewidths=2)
    ax.add_collection(lc)

    ax.scatter(proj[:, 0], proj[:, 1], s=size_nodes, c="k")

    bbox = dict(facecolor="white", alpha=0.6, edgecolor="white", pad=0.1)

    if tips:
        for tip in graph["tips"]:
            ax.annotate(
                tip,
                (proj[tip, 0], proj[tip, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )
    if forks:
        for fork in graph["forks"]:
            ax.annotate(
                fork,
                (proj[fork, 0], proj[fork, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )

    savefig_or_show("graph", show=show, save=save)


def trajectory(
    adata: AnnData,
    basis: str = "umap",
    root_milestone: Union[str, None] = None,
    milestones: Union[str, None] = None,
    color_seg: str = "t",
    cmap_seg: str = "viridis",
    layer_seg: Union[str, None] = None,
    color_cells: Union[str, None] = None,
    scale_path: float = 1,
    arrows: bool = False,
    arrow_offset: int = 10,
    show_info: bool = True,
    ax=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs,
):

    """\
    Project trajectory onto embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Name of the `obsm` basis to use.
    root_milestone
        tip defining progenitor branch.
    milestones
        tips defining the progenies branches.
    col_seg
        color trajectory segments.
    layer_seg
        layer to use when coloring seg with a feature.
    color_cells
        cells color.
    scale_path
        changes the width of the path
    arrows
        display arrows on segments (positioned at half pseudotime distance).
    arrow_offset
        arrow offset in number of nodes used to obtain its direction.
    show_info
        display legend/colorbar.
    ax
        Add plot to existing ax
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to scanpy functions pl.scatter/pl.tsne/pl.umap

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`

    """

    class GC(GraphicsContextBase):
        def __init__(self):
            super().__init__()
            self._capstyle = "round"

    def custom_new_gc(self):
        return GC()

    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

    if "graph" not in adata.uns:
        raise ValueError("You need to run `tl.pseudotime` first before plotting.")

    graph = adata.uns["graph"]

    emb = adata.obsm[f"X_{basis}"]
    emb_f = adata[graph["cells_fitted"], :].obsm[f"X_{basis}"]

    if "components" in kwargs:
        cmp = np.array(kwargs["components"]) - 1
        emb_f = emb_f[:, cmp]

    else:
        emb_f = emb_f[:, :2]

    R = graph["R"]

    nodes = graph["pp_info"].index
    proj = pd.DataFrame((np.dot(emb_f.T, R) / R.sum(axis=0)).T, index=nodes)

    B = graph["B"]
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    g.vs[:]["name"] = [v.index for v in g.vs]

    tips = graph["tips"]

    if root_milestone is not None:
        dct = graph["milestones"]
        nodes = g.get_all_shortest_paths(
            dct[root_milestone], [dct[m] for m in milestones]
        )
        nodes = np.unique(np.concatenate(nodes))
        tips = graph["tips"][np.isin(graph["tips"], nodes)]
        proj = proj.loc[nodes, :]
        g.delete_vertices(graph["pp_info"].index[~graph["pp_info"].index.isin(nodes)])
        if ax is None:
            ax = sc.pl.embedding(adata, show=False, basis=basis, **kwargs)
        else:
            sc.pl.embedding(adata, show=False, ax=ax, basis=basis, **kwargs)

    c_edges = np.array([e.split("|") for e in adata.obs.edge], dtype=int)
    cells = [any(np.isin(c_e, nodes)) for c_e in c_edges]

    import logging

    anndata_logger = logging.getLogger("anndata")
    prelog = anndata_logger.level
    anndata_logger.level = 40
    adata_c = adata[cells, :]

    if is_categorical(adata, color_cells):
        if color_cells not in adata.uns or len(adata.uns[color_cells * "_colors"]) == 1:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, color_cells)

        adata_c.uns[color_cells + "_colors"] = [
            adata.uns[color_cells + "_colors"][
                np.argwhere(adata.obs[color_cells].cat.categories == c)[0][0]
            ]
            for c in adata_c.obs[color_cells].cat.categories
        ]

    if ax is None:
        ax = sc.pl.embedding(
            adata, color=color_cells, basis=basis, show=False, **kwargs
        )
    else:
        sc.pl.embedding(
            adata, color=color_cells, basis=basis, ax=ax, show=False, **kwargs
        )

    anndata_logger.level = prelog
    if show_info == False and color_cells is not None:
        if is_categorical(adata, color_cells):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            ax.set_box_aspect(aspect=1)
            fig = ax.get_gridspec().figure
            cbar = np.argwhere(
                ["colorbar" in a.get_label() for a in fig.get_axes()]
            ).ravel()
            if len(cbar) > 0:
                fig.get_axes()[cbar[0]].remove()

    al = np.array(g.get_edgelist())

    edges = [g.vs[e.tolist()]["name"] for e in al]

    lines = [[tuple(proj.loc[j]) for j in i] for i in edges]

    vals = pd.Series(
        _get_color_values(adata, color_seg, layer=layer_seg)[0], index=adata.obs_names
    )

    sorted_edges = np.sort(
        np.array([e.split("|") for e in adata.obs.edge], dtype=int), axis=1
    )

    seg_val = pd.Series(
        [
            vals[adata[(sorted_edges == e).sum(axis=1) == 2].obs_names].values.mean()
            for e in edges
        ]
    )

    emptyedges = seg_val.index[[np.isnan(sv) for sv in seg_val]]
    nodes = np.array([edges[i] for i in emptyedges]).ravel()
    empty = graph["pp_info"].loc[nodes, :]

    gg = igraph.Graph()
    gg.add_vertices(np.unique(np.array(edges).ravel()).astype(str))
    gg.add_edges(np.array(edges).astype(str))
    s_tips = np.argwhere(np.array(gg.degree()) == 1).ravel()

    atip = s_tips[0]
    s_tips = s_tips[1:]

    for t in s_tips:
        path = np.array(
            gg.get_shortest_paths(
                atip,
                t,
            )[0]
        )

        path = np.array(gg.vs[:]["name"])[path].astype(int)
        path = [path[i : i + 2] for i in range(len(path) - 1)]
        path = [np.sort(p) for p in path]
        holes = np.array([all(np.isin(e, nodes)) for e in path])

        val1 = 0
        idx = []
        toreplace = []
        for i in range(len(holes)):
            if (~holes[i]) and (len(idx) == 0):
                val1 = seg_val[
                    np.argwhere((np.array(edges) == np.sort(path[i])).sum(axis=1) == 2)[
                        0
                    ][0]
                ]
            if (~holes[i]) and (len(idx) > 0):
                val2 = seg_val[
                    np.argwhere((np.array(edges) == np.sort(path[i])).sum(axis=1) == 2)[
                        0
                    ][0]
                ]
                toreplace = toreplace + [[idx, np.mean([val1, val2])]]
                idx = []
            if holes[i]:
                idx = idx + [i]

        if holes[len(holes) - 1]:
            toreplace = toreplace + [[idx, val1]]

        for idx, val in toreplace:
            for i in idx:
                seg_val[
                    np.argwhere((np.array(edges) == np.sort(path[i])).sum(axis=1) == 2)[
                        0
                    ][0]
                ] = val

    sm = ScalarMappable(
        norm=Normalize(vmin=seg_val.min(), vmax=seg_val.max()), cmap=cmap_seg
    )
    lc = matplotlib.collections.LineCollection(
        lines, colors="k", linewidths=7.5 * scale_path, zorder=100
    )
    ax.add_collection(lc)

    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    seg = graph["pp_seg"].loc[:, ["from", "to"]].values.tolist()

    if arrows:
        for s in seg:
            path = np.array(g.get_shortest_paths(s[0], s[1])[0])
            coord = proj.loc[
                path,
            ].values
            out = np.empty(len(path) - 1)
            cdist_numba(coord, out)
            mid = np.argmin(np.abs(out.cumsum() - out.sum() / 2))
            if mid + arrow_offset > (len(path) - 1):
                arrow_offset = len(path) - 1 - mid
            ax.quiver(
                proj.loc[path[mid], 0],
                proj.loc[path[mid], 1],
                proj.loc[path[mid + arrow_offset], 0] - proj.loc[path[mid], 0],
                proj.loc[path[mid + arrow_offset], 1] - proj.loc[path[mid], 1],
                headwidth=15 * scale_path,
                headaxislength=10 * scale_path,
                headlength=10 * scale_path,
                units="dots",
                zorder=101,
            )
            c_arrow = vals[
                (
                    np.array([e.split("|") for e in adata.obs.edge], dtype=int)
                    == path[mid]
                ).sum(axis=1)
                == 1
            ].mean()
            ax.quiver(
                proj.loc[path[mid], 0],
                proj.loc[path[mid], 1],
                proj.loc[path[mid + arrow_offset], 0] - proj.loc[path[mid], 0],
                proj.loc[path[mid + arrow_offset], 1] - proj.loc[path[mid], 1],
                headwidth=12 * scale_path,
                headaxislength=10 * scale_path,
                headlength=10 * scale_path,
                units="dots",
                color=sm.to_rgba(c_arrow),
                zorder=102,
            )

    lc = matplotlib.collections.LineCollection(
        lines,
        colors=[sm.to_rgba(sv) for sv in seg_val],
        linewidths=5 * scale_path,
        zorder=104,
    )
    ax.scatter(
        proj.loc[tips, 0],
        proj.loc[tips, 1],
        zorder=103,
        c="k",
        s=200 * scale_path,
    )
    ax.add_collection(lc)

    tip_val = []
    for tip in tips:
        sel = (
            np.array(list(map(lambda e: e.split("|"), adata.obs.edge)), dtype=int)
            == tip
        ).sum(axis=1) == 1
        tip_val = tip_val + [vals.loc[adata.obs_names[sel]].mean()]

    ax.scatter(
        proj.loc[tips, 0],
        proj.loc[tips, 1],
        zorder=105,
        c=sm.to_rgba(tip_val),
        s=140 * scale_path,
    )

    if show == False:
        return ax

    savefig_or_show("trajectory", show=show, save=save)


@njit()
def cdist_numba(coords, out):
    for i in range(0, coords.shape[0] - 1):
        out[i] = math.sqrt(
            (coords[i, 0] - coords[i + 1, 0]) ** 2
            + (coords[i, 1] - coords[i + 1, 1]) ** 2
        )


def scatter3d(emb, col, cell_size, nm):
    return go.Scatter3d(
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        mode="markers",
        marker=dict(size=cell_size, color=col, opacity=0.9),
        name=nm,
    )


def trajectory_3d(
    adata: AnnData,
    basis: str = "umap3d",
    color: str = None,
    traj_width: int = 5,
    cell_size: int = 2,
    figsize: tuple = (900, 900),
    cmap=None,
):

    """\
    Project trajectory onto 3d embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Name of the `obsm` basis to use.
    color
        cells color.
    traj_width
        segments width.
    cell_size
        cell size.
    figsize
        figure size in pixels.
    cmap
        colormap of the cells.

    Returns
    -------
    an interactive plotly graph figure.

    """

    r = adata.uns["graph"]

    emb = adata.obsm[f"X_{basis}"]
    if emb.shape[1] > 3:
        raise ValueError("Embedding is not three dimensional.")

    emb_f = adata[r["cells_fitted"], :].obsm[f"X_{basis}"]

    R = r["R"]
    proj = (np.dot(emb_f.T, R) / R.sum(axis=0)).T

    B = r["B"]

    al = np.array(
        igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected").get_edgelist()
    )
    segs = al.tolist()
    vertices = proj.tolist()
    vertices = np.array(vertices)
    segs = np.array(segs)

    x_lines = list()
    y_lines = list()
    z_lines = list()

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    for i in range(segs.shape[0]):
        p = segs[i, :]
        for i in range(2):
            x_lines.append(x[p[i]])
            y_lines.append(y[p[i]])
            z_lines.append(z[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    if color is not None:
        if adata.obs[color].dtype.name == "str":
            adata.obs[color] = adata.obs[color].astype("category")

        if color + "_colors" not in adata.uns:
            from . import palette_tools

            palette_tools._set_default_colors_for_categorical_obs(adata, color)

        if adata.obs[color].dtype.name == "category":
            pal_dict = dict(
                zip(adata.obs[color].cat.categories, adata.uns[color + "_colors"])
            )
            trace1 = list(
                map(
                    lambda x: scatter3d(
                        emb_f[adata.obs[color] == x, :], pal_dict[x], cell_size, x
                    ),
                    list(pal_dict.keys()),
                )
            )

        else:
            if cmap is None:
                cmap = "Viridis"
            trace1 = [
                go.Scatter3d(
                    x=emb_f[:, 0],
                    y=emb_f[:, 1],
                    z=emb_f[:, 2],
                    mode="markers",
                    marker=dict(
                        size=cell_size,
                        color=adata.obs[color],
                        colorscale=cmap,
                        opacity=0.9,
                    ),
                )
            ]

    else:
        trace1 = [
            go.Scatter3d(
                x=emb_f[:, 0],
                y=emb_f[:, 1],
                z=emb_f[:, 2],
                mode="markers",
                marker=dict(size=cell_size, color="grey", opacity=0.9),
            )
        ]

    trace2 = [
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            name="lines",
            line=dict(width=traj_width),
        )
    ]

    fig = go.Figure(data=trace1 + trace2)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=figsize[0],
        height=figsize[0],
        margin=dict(l=5, r=5, t=5, b=5),
    )
    fig.show()


def _get_color_values(
    adata,
    value_to_plot,
    groups=None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    use_raw=False,
    gene_symbols=None,
    layer=None,
) -> Tuple[Union[np.ndarray, str], bool]:
    """
    Returns the value or color associated to each data point.
    For categorical data, the return value is list of colors taken
    from the category palette or from the given `palette` value.
    For non-categorical data, the values are returned
    Returns
    -------
    values
        Values to plot
    is_categorical
        Are the values categorical?
    """
    if value_to_plot is None:
        return "lightgray", False
    if (
        gene_symbols is not None
        and value_to_plot not in adata.obs.columns
        and value_to_plot not in adata.var_names
    ):
        # We should probably just make an index for this, and share it over runs
        value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][
            0
        ]  # TODO: Throw helpful error if this doesn't work
    if use_raw and value_to_plot not in adata.obs.columns:
        values = adata.raw.obs_vector(value_to_plot)
    else:
        values = adata.obs_vector(value_to_plot, layer=layer)

    ###
    # when plotting, the color of the dots is determined for each plot
    # the data is either categorical or continuous and the data could be in
    # 'obs' or in 'var'
    if not is_categorical_dtype(values):
        return values, False
    else:  # is_categorical_dtype(values)
        color_key = f"{value_to_plot}_colors"
        if palette:
            _utils._set_colors_for_categorical_obs(adata, value_to_plot, palette)
        elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
            values.categories
        ):
            #  set a default palette in case that no colors or few colors are found
            _utils._set_default_colors_for_categorical_obs(adata, value_to_plot)
        else:
            _utils._validate_palette(adata, value_to_plot)

        color_vector = np.asarray(adata.uns[color_key])[values.codes]

        # Handle groups
        if groups:
            color_vector = np.fromiter(
                map(colors.to_hex, color_vector), "<U15", len(color_vector)
            )
            # set color to 'light gray' for all values
            # that are not in the groups
            color_vector[~adata.obs[value_to_plot].isin(groups)] = "lightgray"
        return color_vector, True


def is_categorical(data, c=None):
    from pandas.api.types import is_categorical_dtype as cat

    if c is None:
        return cat(data)  # if data is categorical/array
    if not is_view(data):  # if data is anndata view
        strings_to_categoricals(data)
    return isinstance(c, str) and c in data.obs.keys() and cat(data.obs[c])


def is_view(adata):
    return (
        adata.is_view
        if hasattr(adata, "is_view")
        else adata.isview
        if hasattr(adata, "isview")
        else adata._isview
        if hasattr(adata, "_isview")
        else True
    )


def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas.api.types import is_string_dtype, is_integer_dtype, is_bool_dtype
    from pandas import Categorical

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c
