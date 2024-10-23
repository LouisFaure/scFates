import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.collections
from typing import Union, Optional, Sequence, Tuple, List
import plotly.graph_objects as go
import scanpy as sc
from cycler import Cycler

from pandas.api.types import is_categorical_dtype
from scanpy.plotting._utils import savefig_or_show

import matplotlib.patheffects as path_effects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, hex2color, rgb2hex
from numba import njit
import math

from .utils import is_categorical, get_basis, subset_cells
from . import palette_tools
from .milestones import milestones as milestones_plot
from ..tools.graph_operations import subset_tree
from .. import settings


def graph(
    adata: AnnData,
    basis: Union[None, str] = None,
    size_nodes: float = None,
    alpha_nodes: float = 1,
    linewidth: float = 2,
    alpha_seg: float = 1,
    color_cells: Union[str, None] = None,
    tips: bool = True,
    forks: bool = True,
    nodes: Optional[List] = [],
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
        Size of the projected prinicpal points.
    alpha_nodes
        Alpha of nodes.
    linewidth
        Line width of the segments.
    alpha_seg
        Alpha of segments.
    color_cells
        cells color
    tips
        display tip ids.
    forks
        display fork ids.
    nodes
        display any node id.
    ax
        Add plot to existing ax
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to :func:`scanpy.pl.embedding`

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`

    """

    if "graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.tree` or `tl.curve` first to compute a princal graph before plotting."
        )

    graph = adata.uns["graph"]

    basis = get_basis(adata, basis)

    emb = adata.obsm[f"X_{basis}"]

    if "components" in kwargs:
        cmp = np.array(kwargs["components"])
        emb = emb[:, cmp]
    elif "dimensions" in kwargs:
        cmp = np.array(kwargs["dimensions"])
        emb = emb[:, cmp]
    else:
        emb = emb[:, :2]

    R = adata.obsm["X_R"]

    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

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
    lc = matplotlib.collections.LineCollection(
        lines, colors="k", linewidths=linewidth, alpha=alpha_seg, rasterized=True
    )
    ax.add_collection(lc)

    ax.scatter(
        proj[:, 0], proj[:, 1], s=size_nodes, c="k", alpha=alpha_nodes, rasterized=True
    )

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
    if nodes:
        for node in nodes:
            ax.annotate(
                node,
                (proj[node, 0], proj[node, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )
    if show == False:
        return ax

    savefig_or_show("graph", show=show, save=save)


def trajectory(
    adata: AnnData,
    basis: Union[None, str] = None,
    root_milestone: Union[str, None] = None,
    milestones: Union[str, None] = None,
    color_seg: str = "t",
    cmap_seg: str = "viridis",
    layer_seg: Union[str, None] = "fitted",
    perc_seg: Union[List, None] = None,
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
    perc_seg
        percentile cutoffs for segments.
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
        arguments to pass to :func:`scanpy.pl.embedding`.

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`

    """

    if "graph" not in adata.uns:
        raise ValueError("You need to run `tl.pseudotime` first before plotting.")

    if (root_milestone is not None) & (color_seg == "milestones"):
        if ax is None:
            ax = sc.pl.embedding(adata, show=False, basis=basis, **kwargs)
        else:
            sc.pl.embedding(adata, show=False, ax=ax, basis=basis, **kwargs)
        verb_temp = settings.verbosity
        settings.verbosity = 1
        adata = subset_tree(
            adata,
            root_milestone=root_milestone,
            milestones=milestones,
            mode="extract",
            copy=True,
        )
        settings.verbosity = verb_temp

    graph = adata.uns["graph"]

    basis = get_basis(adata, basis) if basis is None else basis

    emb = adata.obsm[f"X_{basis}"]

    if "components" in kwargs:
        cmp = np.array(kwargs["components"])
        emb = emb[:, cmp]
    elif "dimensions" in kwargs:
        cmp = np.array(kwargs["dimensions"])
        emb = emb[:, cmp]
    else:
        emb = emb[:, :2]

    R = adata.obsm["X_R"]

    nodes = graph["pp_info"].index
    proj = pd.DataFrame((np.dot(emb.T, R) / R.sum(axis=0)).T, index=nodes)

    B = graph["B"]
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    g.vs[:]["name"] = [v.index for v in g.vs]

    miles_ids = np.concatenate([graph["tips"], graph["forks"]])

    if (root_milestone is not None) & (color_seg != "milestones"):
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

    import logging

    anndata_logger = logging.getLogger("anndata")
    prelog = anndata_logger.level
    anndata_logger.level = 40
    adata_s = (
        subset_cells(adata, root_milestone, milestones)
        if milestones is not None
        else adata.copy()
    )

    if is_categorical(adata, color_cells):
        if (
            color_cells + "_colors" not in adata.uns
            or len(adata.uns[color_cells + "_colors"]) == 1
        ):

            palette_tools._set_default_colors_for_categorical_obs(adata, color_cells)

        adata_s.uns[color_cells + "_colors"] = [
            adata.uns[color_cells + "_colors"][
                np.argwhere(adata.obs[color_cells].cat.categories == c)[0][0]
            ]
            for c in adata_s.obs[color_cells].cat.categories
        ]
    if is_categorical(adata, color_seg):
        if (
            color_seg + "_colors" not in adata.uns
            or len(adata.uns[color_seg + "_colors"]) == 1
        ):

            palette_tools._set_default_colors_for_categorical_obs(adata, color_seg)

        adata_s.uns[color_seg + "_colors"] = [
            adata.uns[color_seg + "_colors"][
                np.argwhere(adata.obs[color_seg].cat.categories == c)[0][0]
            ]
            for c in adata_s.obs[color_seg].cat.categories
        ]

    plotter = milestones_plot if color_cells == "milestones" else sc.pl.embedding
    if ("s" not in kwargs) & (milestones is not None):
        kwargs["s"] = 120000 / adata.shape[0]
    if ax is None:
        ax = plotter(adata_s, color=color_cells, basis=basis, show=False, **kwargs)
    else:
        plotter(adata_s, color=color_cells, basis=basis, ax=ax, show=False, **kwargs)

    anndata_logger.level = prelog
    if show_info == False and color_cells is not None:
        remove_info(adata, ax, color_cells)

    al = np.array(g.get_edgelist())

    edges = [g.vs[e.tolist()]["name"] for e in al]

    lines = [[tuple(proj.loc[j]) for j in i] for i in edges]

    miles_ids = miles_ids[np.isin(miles_ids, proj.index)]

    if color_seg == "milestones":
        from matplotlib.colors import LinearSegmentedColormap

        rev_dict = dict(zip(graph["milestones"].values(), graph["milestones"].keys()))
        miles_cat = adata.obs.milestones.cat.categories
        mil_col = np.array(adata.uns["milestones_colors"])

        def get_milestones_gradients(i):
            start = graph["pp_seg"].iloc[i, 1]
            end = graph["pp_seg"].iloc[i, 2]
            mil_path = graph["pp_info"].time[g.get_all_shortest_paths(start, end)[0]]
            mil_path = (mil_path - mil_path.min()) / (mil_path.max() - mil_path.min())
            start_col = mil_col[miles_cat == rev_dict[start]][0]
            end_col = mil_col[miles_cat == rev_dict[end]][0]

            edges_mil = pd.Series(
                [
                    mil_path[[first, second]].mean()
                    for first, second in zip(mil_path.index, mil_path.index[1:])
                ],
                index=[
                    (first, second)
                    for first, second in zip(mil_path.index, mil_path.index[1:])
                ],
            )

            cmap = LinearSegmentedColormap.from_list("mil", [start_col, end_col])
            return pd.Series(
                [rgb2hex(c) for c in cmap(edges_mil)], index=edges_mil.index
            )

        edge_colors = pd.concat(
            [get_milestones_gradients(i) for i in range(graph["pp_seg"].shape[0])]
        )

        edges_tuples = [tuple(e) for e in edges]
        edge_colors.index = [
            e if any((np.array(edges_tuples) == e).sum(axis=1) == 2) else e[::-1]
            for e in edge_colors.index
        ]
        edge_colors = edge_colors[edges_tuples]

        color_segs = [hex2color(c) for c in edge_colors]
        color_mils = [
            mil_col[miles_cat == mm][0] for mm in [rev_dict[m] for m in miles_ids]
        ]
    elif color_seg == "seg":
        seg_edges = (
            graph["pp_info"]
            .loc[np.array(edges).ravel(), "seg"]
            .values.reshape(-1, 2)[:, 0]
        )
        seg_col = pd.Series(adata.uns["seg_colors"], index=adata.obs.seg.cat.categories)
        color_segs = [hex2color(seg_col.loc[s]) for s in seg_edges]
        color_mils = [
            hex2color(seg_col.loc[graph["pp_info"].loc[m].seg])
            for m in graph["milestones"].values()
        ]
    else:
        vals = pd.Series(
            _get_color_values(adata, color_seg, layer=layer_seg)[0],
            index=adata.obs_names,
        )
        if not np.issubdtype(vals.dtype, np.number):
            raise Exception("can only color numerical values")
        R = pd.DataFrame(adata.obsm["X_R"], index=adata.obs_names)
        R = R.loc[adata.obs_names]
        vals = vals[~np.isnan(vals)]
        R = R.loc[vals.index]

        def get_nval(i):
            return np.average(vals, weights=R.loc[:, i])

        node_vals = np.array(list(map(get_nval, range(R.shape[1]))))
        seg_val = node_vals[np.array(edges)].mean(axis=1)

        if perc_seg is not None:
            min_v, max_v = np.percentile(seg_val, perc_seg)
            seg_val[seg_val < min_v] = min_v
            seg_val[seg_val > max_v] = max_v

        sm = ScalarMappable(
            norm=Normalize(vmin=seg_val.min(), vmax=seg_val.max()), cmap=cmap_seg
        )

        lines = [lines[i] for i in np.argsort(seg_val)]
        seg_val = seg_val[np.argsort(seg_val)]

        color_segs = [sm.to_rgba(sv) for sv in seg_val]
        color_mils = sm.to_rgba(node_vals[miles_ids])

    lc = matplotlib.collections.LineCollection(
        lines,
        colors="k",
        linewidths=7.5 * scale_path,
        zorder=100,
        path_effects=[path_effects.Stroke(capstyle="round")],
        rasterized=True,
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
                rasterized=True,
            )
            closest = (
                np.array([e.split("|") for e in adata.obs.edge], dtype=int) == path[mid]
            ).sum(axis=1) == 1
            if color_seg == "seg":
                c_arrow = seg_col.loc[adata.obs.seg.loc[closest]]
            elif color_seg == "milestones":
                mid_edge = np.argwhere((np.array(edges) == path[mid]).sum(axis=1))[0][0]
                c_arrow = np.array(color_segs)[mid_edge]
            else:
                c_arrow = sm.to_rgba(vals[closest].mean())
            ax.quiver(
                proj.loc[path[mid], 0],
                proj.loc[path[mid], 1],
                proj.loc[path[mid + arrow_offset], 0] - proj.loc[path[mid], 0],
                proj.loc[path[mid + arrow_offset], 1] - proj.loc[path[mid], 1],
                headwidth=12 * scale_path,
                headaxislength=10 * scale_path,
                headlength=10 * scale_path,
                units="dots",
                color=c_arrow,
                zorder=102,
                rasterized=True,
            )

    lc = matplotlib.collections.LineCollection(
        lines,
        colors=color_segs,
        linewidths=5 * scale_path,
        zorder=104,
        path_effects=[path_effects.Stroke(capstyle="round")],
        rasterized=True,
    )

    ax.scatter(
        proj.loc[miles_ids, 0],
        proj.loc[miles_ids, 1],
        zorder=103,
        c="k",
        s=200 * scale_path,
        rasterized=True,
    )
    ax.add_collection(lc)

    ax.scatter(
        proj.loc[miles_ids, 0],
        proj.loc[miles_ids, 1],
        zorder=105,
        c=color_mils,
        s=140 * scale_path,
        rasterized=True,
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

    R = adata.obsm["X_R"]
    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

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
            palette_tools._set_default_colors_for_categorical_obs(adata, color)

        if adata.obs[color].dtype.name == "category":
            pal_dict = dict(
                zip(adata.obs[color].cat.categories, adata.uns[color + "_colors"])
            )
            trace1 = list(
                map(
                    lambda x: scatter3d(
                        emb[adata.obs[color] == x, :], pal_dict[x], cell_size, x
                    ),
                    list(pal_dict.keys()),
                )
            )

        else:
            if cmap is None:
                cmap = "Viridis"
            trace1 = [
                go.Scatter3d(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    z=emb[:, 2],
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
                x=emb[:, 0],
                y=emb[:, 1],
                z=emb[:, 2],
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
    if not isinstance(values, pd.CategoricalDtype):
        return values, False
    else:  # is_categorical_dtype(values)
        color_key = f"{value_to_plot}_colors"
        if palette:
            palette_tools._set_colors_for_categorical_obs(adata, value_to_plot, palette)
        elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
            values.categories
        ):
            #  set a default palette in case that no colors or few colors are found
            palette_tools._set_default_colors_for_categorical_obs(adata, value_to_plot)
        else:
            palette_tools._validate_palette(adata, value_to_plot)

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


def remove_info(adata, ax, color):
    if is_categorical(adata, color):
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
