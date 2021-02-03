from typing import Union, Sequence
import numpy as np
from matplotlib import pyplot as pl
from cycler import Cycler, cycler
from matplotlib import rcParams
from . import palettes
from matplotlib.colors import to_hex


def _set_colors_for_categorical_obs(
    adata,
    value_to_plot,
    palette: Union[str, Sequence[str], Cycler],
):
    """
    Sets the adata.uns[value_to_plot + '_colors'] according to the given palette
    Parameters
    ----------
    adata
        annData object
    value_to_plot
        name of a valid categorical observation
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'
    Returns
    -------
    None
    """

    categories = adata.obs[value_to_plot].cat.categories
    # check is palette is a valid matplotlib colormap
    if isinstance(palette, str) and palette in pl.colormaps():
        # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
        cmap = pl.get_cmap(palette)
        colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, len(categories)))]

    else:
        # check if palette is a list and convert it to a cycler, thus
        # it doesnt matter if the list is shorter than the categories length:
        if isinstance(palette, cabc.Sequence):
            if len(palette) < len(categories):
                logg.warning(
                    "Length of palette colors is smaller than the number of "
                    f"categories (palette length: {len(palette)}, "
                    f"categories length: {len(categories)}. "
                    "Some categories will have the same color."
                )
            # check that colors are valid
            _color_list = []
            for color in palette:
                if not is_color_like(color):
                    # check if the color is a valid R color and translate it
                    # to a valid hex color value
                    if color in additional_colors:
                        color = additional_colors[color]
                    else:
                        raise ValueError(
                            "The following color value of the given palette "
                            f"is not valid: {color}"
                        )
                _color_list.append(color)

            palette = cycler(color=_color_list)
        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that the value of 'palette' is a valid "
                "matplotlib colormap string (eg. Set2), a  list of color names "
                "or a cycler with a 'color' key."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"]) for x in range(len(categories))]

    adata.uns[value_to_plot + "_colors"] = colors_list


def _set_default_colors_for_categorical_obs(adata, value_to_plot):
    """
    Sets the adata.uns[value_to_plot + '_colors'] using default color palettes
    Parameters
    ----------
    adata
        AnnData object
    value_to_plot
        Name of a valid categorical observation
    Returns
    -------
    None
    """

    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]
        palette = [to_hex(x) for x in palette]
    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            logg.info(
                f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
                "'grey' color will be used for all categories."
            )

    adata.uns[value_to_plot + "_colors"] = palette[:length]


def add_colors_for_categorical_sample_annotation(
    adata, key, palette=None, force_update_colors=False
):

    color_key = f"{key}_colors"
    colors_needed = len(adata.obs[key].cat.categories)
    if palette and force_update_colors:
        _set_colors_for_categorical_obs(adata, key, palette)
    elif color_key in adata.uns and len(adata.uns[color_key]) <= colors_needed:
        _validate_palette(adata, key)
    else:
        _set_default_colors_for_categorical_obs(adata, key)
