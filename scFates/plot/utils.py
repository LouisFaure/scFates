# Extracted from scanpy, thanks!
from functools import lru_cache
from typing import Union, Sequence
from typing_extensions import Literal

import numpy as np
from matplotlib import pyplot as pl
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import SubplotParams as sppars


def setup_axes(
    ax: Union[Axes, Sequence[Axes]] = None,
    panels="blue",
    colorbars=(False,),
    right_margin=None,
    left_margin=None,
    projection: Literal["2d", "3d"] = "2d",
    show_ticks=False,
):
    """Grid of axes for plotting, legends and colorbars."""
    make_projection_available(projection)
    if left_margin is not None:
        raise NotImplementedError("We currently don’t support `left_margin`.")
    if np.any(colorbars) and right_margin is None:
        right_margin = 1 - rcParams["figure.subplot.right"] + 0.21  # 0.25
    elif right_margin is None:
        right_margin = 1 - rcParams["figure.subplot.right"] + 0.06  # 0.10
    # make a list of right margins for each panel
    if not isinstance(right_margin, list):
        right_margin_list = [right_margin for i in range(len(panels))]
    else:
        right_margin_list = right_margin

    # make a figure with len(panels) panels in a row side by side
    top_offset = 1 - rcParams["figure.subplot.top"]
    bottom_offset = 0.15 if show_ticks else 0.08
    left_offset = 1 if show_ticks else 0.3  # in units of base_height
    base_height = rcParams["figure.figsize"][1]
    height = base_height
    base_width = rcParams["figure.figsize"][0]
    if show_ticks:
        base_width *= 1.1

    draw_region_width = (
        base_width - left_offset - top_offset - 0.5
    )  # this is kept constant throughout

    right_margin_factor = sum([1 + right_margin for right_margin in right_margin_list])
    width_without_offsets = (
        right_margin_factor * draw_region_width
    )  # this is the total width that keeps draw_region_width

    right_offset = (len(panels) - 1) * left_offset
    figure_width = width_without_offsets + left_offset + right_offset
    draw_region_width_frac = draw_region_width / figure_width
    left_offset_frac = left_offset / figure_width
    right_offset_frac = 1 - (len(panels) - 1) * left_offset_frac

    if ax is None:
        pl.figure(
            figsize=(figure_width, height),
            subplotpars=sppars(left=0, right=1, bottom=bottom_offset),
        )
    left_positions = [left_offset_frac, left_offset_frac + draw_region_width_frac]
    for i in range(1, len(panels)):
        right_margin = right_margin_list[i - 1]
        left_positions.append(
            left_positions[-1] + right_margin * draw_region_width_frac
        )
        left_positions.append(left_positions[-1] + draw_region_width_frac)
    panel_pos = [[bottom_offset], [1 - top_offset], left_positions]

    axs = []
    if ax is None:
        for icolor, color in enumerate(panels):
            left = panel_pos[2][2 * icolor]
            bottom = panel_pos[0][0]
            width = draw_region_width / figure_width
            height = panel_pos[1][0] - bottom
            if projection == "2d":
                ax = pl.axes([left, bottom, width, height])
            elif projection == "3d":
                ax = pl.axes([left, bottom, width, height], projection="3d")
            axs.append(ax)
    else:
        axs = ax if isinstance(ax, cabc.Sequence) else [ax]

    return axs, panel_pos, draw_region_width, figure_width


@lru_cache(None)
def make_projection_available(projection):
    avail_projections = {"2d", "3d"}
    if projection not in avail_projections:
        raise ValueError(f"choose projection from {avail_projections}")
    if projection == "2d":
        return

    from io import BytesIO
    from matplotlib import __version__ as mpl_version
    from mpl_toolkits.mplot3d import Axes3D

    fig = Figure()
    ax = Axes3D(fig)

    circles = PatchCollection([Circle((5, 1)), Circle((2, 2))])
    ax.add_collection3d(circles, zs=[1, 2])

    buf = BytesIO()
    try:
        fig.savefig(buf)
    except ValueError as e:
        if not "operands could not be broadcast together" in str(e):
            raise e
        raise ValueError(
            "There is a known error with matplotlib 3d plotting, "
            f"and your version ({mpl_version}) seems to be affected. "
            "Please install matplotlib==3.0.2 or wait for "
            "https://github.com/matplotlib/matplotlib/issues/14298"
        )
