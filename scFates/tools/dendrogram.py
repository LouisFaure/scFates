import networkx as nx
import random
import warnings
import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
import sys

from .utils import ProgressParallel
from joblib import delayed

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

__all__ = ["swarmplot"]


from .. import logging as logg
from .. import settings

import anndata


def dendrogram(adata: anndata.AnnData, crowdedness: float = 1, n_jobs: int = 1):

    """\
    Generate a  single-cell dendrogram embedding.

    This representation aims in simplifying and abstracting the view of the tree, it follows URD style represenation of the cells.

    Parameters
    ----------
    adata
        Annotated data matrix.
    crowdedness
        will influence the repartition of the cells along the segments.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.obs['X_dendro']`
           new embedding generated.
        `.uns['dendro_segments']`
            tree segments used for plotting.

    """

    logg.info("Generating dendrogram of tree", reset=True)

    roots = None

    graph = adata.uns["graph"]

    dct = graph["milestones"]
    keys = np.array(list(dct.keys()))
    vals = np.array(list(dct.values()))

    edges = graph["pp_seg"][["from", "to"]].astype(str).apply(tuple, axis=1).values
    img = igraph.Graph(directed=True)
    img.add_vertices(vals.astype(str))
    img.add_edges(edges)

    img.vs["label"] = keys

    pos = hierarchy_pos(img.to_networkx(), vert_gap=0.1)

    layout = pd.DataFrame(pos).T
    layout.index = np.array(list(graph["milestones"].values()), int)[
        np.array(list(pos.keys()))
    ]
    layout = layout[0]
    newseg = layout.loc[graph["pp_seg"].to]
    newseg.index = graph["pp_seg"].n
    df = adata.obs.copy()

    df["seg_pos"] = df.seg.astype(str)
    df["t"] = -df.t

    for i, x in enumerate(newseg):
        df.loc[df.seg == newseg.index[i], "seg_pos"] = x

    df.seg_pos = df.seg_pos.astype("category").cat.rename_categories(
        range(len(df.seg_pos.unique()))
    )
    df.seg_pos = df.seg_pos.astype(int).astype("category")

    dend = swarmplot(
        x="seg_pos", y="t", data=df, orient="v", size=crowdedness, n_jobs=n_jobs
    )
    adata.obsm["X_dendro"] = dend.loc[adata.obs_names].values

    # generate segments
    newseg = newseg.iloc[newseg.argsort()]

    segments = []
    for i, n in enumerate(newseg.index):
        fro = graph["pp_seg"].loc[int(n), "from"]
        to = graph["pp_seg"].loc[int(n), "to"]
        t_min = -graph["pp_info"].loc[fro].time
        t_max = -graph["pp_info"].loc[to].time

        segments = segments + [[n, [(i, i), (t_min, t_max)]]]

        for l in graph["pp_seg"].n[graph["pp_seg"]["from"] == to]:
            link = np.argwhere(newseg.index == str(l))[0][0]
            segments = segments + [[l, [(i, link), (t_max, t_max)]]]

    df = pd.DataFrame(segments)
    segments = df[0].unique()

    adata.uns["dendro_segments"] = dict(
        zip(segments, [df.loc[df[0] == s, 1].tolist() for s in segments])
    )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .obsm['X_dendro'], new embedding generated.\n"
        "    .uns['dendro_segments'] tree segments used for plotting."
    )


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


### All functions bellow have been adapted from seaborn v0.11.1
### https://github.com/mwaskom/seaborn/blob/v0.11.1/seaborn/categorical.py


class _CategoricalPlotter(object):

    width = 0.8
    default_palette = "light"
    require_numeric = True

    def establish_variables(
        self,
        x=None,
        y=None,
        hue=None,
        data=None,
        orient=None,
        order=None,
        hue_order=None,
        units=None,
    ):
        """Convert input specification into a common representation."""

        x = data.get(x, x)
        y = data.get(y, y)
        hue = data.get(hue, hue)
        units = data.get(units, units)

        # Figure out the plotting orientation
        vals, groups = y, x
        # Put them into the common representation
        plot_data = [np.asarray(vals)]

        # Get a label for the value axis
        value_label = vals.name

        # Get the categorical axis label
        group_label = None
        if hasattr(groups, "name"):
            group_label = groups.name

        # Get the order on the categorical axis
        group_names = list(data.seg_pos.cat.categories)

        plot_data, value_label = self._group_longform(vals, groups, group_names)

        plot_hues = None
        hue_title = None
        plot_units = None

        self.orient = orient
        self.plot_data = plot_data
        self.group_label = group_label
        self.value_label = value_label
        self.group_names = group_names
        self.plot_hues = plot_hues
        self.hue_title = hue_title

    def _group_longform(self, vals, grouper, order):
        """Group a long-form variable by another with correct order."""

        # Group the val data
        grouped_vals = vals.groupby(grouper,observed=False)
        out_data = []
        for g in order:
            try:
                g_vals = grouped_vals.get_group(g)
            except KeyError:
                g_vals = np.array([])
            out_data.append(g_vals)

        # Get the vals axis label
        label = vals.name

        return out_data, label


class _CategoricalScatterPlotter(_CategoricalPlotter):

    default_palette = "dark"
    require_numeric = False


class _SwarmPlotter(_CategoricalScatterPlotter):
    def __init__(
        self, x, y, hue, data, order, hue_order, dodge, orient, color, palette, n_jobs
    ):
        """Initialize the plotter."""
        self.establish_variables(x, y, hue, data, orient, order, hue_order)

        # Set object attributes
        self.dodge = dodge
        self.width = 0.8
        self.n_jobs = n_jobs

    def could_overlap(self, xy_i, swarm, d):
        """Return a list of all swarm points that could overlap with target.

        Assumes that swarm is a sorted list of all points below xy_i.
        """
        _, y_i = xy_i
        neighbors = []
        for xy_j in reversed(swarm):
            _, y_j = xy_j
            if (y_i - y_j) < d:
                neighbors.append(xy_j)
            else:
                break
        return np.array(list(reversed(neighbors)))

    def position_candidates(self, xy_i, neighbors, d):
        """Return a list of (x, y) coordinates that might be valid."""
        candidates = [xy_i]
        x_i, y_i = xy_i
        left_first = True
        for x_j, y_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max(d ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i), (x_j + dx, y_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors, d):
        """Remove candidates from the list if they overlap with the swarm."""

        # IF we have no neighbours, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]

        d_square = d ** 2

        for xy_i in candidates:
            x_i, y_i = xy_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i

            sq_distances = np.power(dx, 2.0) + np.power(dy, 2.0)

            # good candidate does not overlap any of neighbors
            # which means that squared distance between candidate
            # and any of the neighbours has to be at least
            # square of the diameter
            good_candidate = np.all(sq_distances >= d_square)

            if good_candidate:
                return xy_i

        # If `position_candidates` works well
        # this should never happen
        raise Exception(
            "No non-overlapping candidates found. " "This should not happen."
        )

    def beeswarm(self, orig_xy, d):
        """Adjust x position of points to avoid overlaps."""
        # In this method, ``x`` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xy[0, 0]

        # Start the swarm with the first point
        swarm = [orig_xy[0]]

        # Loop over the remaining points
        for xy_i in orig_xy[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xy_i, swarm, d)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xy_i, neighbors, d)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbours
            new_xy_i = self.first_non_overlapping_candidate(candidates, neighbors, d)

            # Place it into the swarm
            swarm.append(new_xy_i)

        return np.array(swarm)

    def add_gutters(self, points, center, width):
        """Stop points from extending beyond their territory."""
        half_width = width / 2
        low_gutter = center - half_width
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        high_gutter = center + half_width
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter
        return points

    def swarm_points(self, ax, points, center, width, s, **kws):
        """Find new positions on the categorical axis for each point."""
        # Convert from point size (area) to diameter
        default_lw = mpl.rcParams["patch.linewidth"]
        lw = kws.get("linewidth", kws.get("lw", default_lw))
        dpi = ax.figure.dpi
        d = (np.sqrt(s) + lw) * (dpi / 72)
        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_xy = ax.transData.transform(points.get_offsets())

        # Order the variables so that x is the categorical axis

        # Do the beeswarm in point coordinates
        new_xy = self.beeswarm(orig_xy, d)

        # Transform the point coordinates back to data coordinates

        new_x, new_y = ax.transData.inverted().transform(new_xy).T

        # Add gutters

        self.add_gutters(new_x, center, width)

        # Reposition the points so they do not overlap
        return np.c_[new_x, new_y]

    def draw_swarmplot(self, ax, kws):
        """Plot the data."""
        s = kws.pop("s")

        centers = []
        swarms = []

        # Set the categorical axes limits here for the swarm math

        ax.set_xlim(-0.5, len(self.plot_data) - 0.5)

        # Plot each swarm
        cells = []
        for i, group_data in enumerate(self.plot_data):
            width = self.width
            swarm_data = np.asarray(group_data)

            # Sort the points for the beeswarm algorithm
            sorter = np.argsort(swarm_data)
            swarm_data = swarm_data[sorter]
            cell = group_data.index[sorter]

            # Plot the points in centered positions
            cat_pos = np.ones(swarm_data.size) * i
            points = ax.scatter(cat_pos, swarm_data, s=s, **kws)

            cells.append(cell)
            centers.append(i)
            swarms.append(points)

        # Autoscale the valus axis to set the data/axes transforms properly
        ax.autoscale_view(scalex=self.orient == "h", scaley=self.orient == "v")

        xys = ProgressParallel(
            n_jobs=self.n_jobs,
            total=len(self.plot_data),
            file=sys.stdout,
            desc="    segment ",
        )(
            delayed(self.swarm_points)(ax, swarm, center, 0.8, s, **kws)
            for center, swarm in zip(centers, swarms)
        )
        plt.close()
        dend = pd.concat(
            [pd.DataFrame(xy, index=cell) for cell, xy in zip(cells, xys)], axis=0
        )
        return dend

    def plot(self, ax, kws):
        """Make the full plot."""
        return self.draw_swarmplot(ax, kws)


def swarmplot(
    *,
    x=None,
    y=None,
    hue=None,
    data=None,
    order=None,
    hue_order=None,
    dodge=False,
    orient=None,
    color=None,
    palette=None,
    size=5,
    edgecolor="gray",
    linewidth=0,
    ax=None,
    n_jobs=1,
    **kwargs
):

    plotter = _SwarmPlotter(
        x, y, hue, data, order, hue_order, dodge, orient, color, palette, n_jobs
    )
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        edgecolor = "gray"
    kwargs.update(dict(s=size ** 2, edgecolor=edgecolor, linewidth=linewidth))

    dend = plotter.plot(ax, kwargs)
    return dend
