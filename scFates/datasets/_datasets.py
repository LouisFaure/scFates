from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

HERE = Path(__file__).parent

from scanpy import read


url_datadir = "https://github.com/LouisFaure/scFates_notebooks/raw/main/"


def morarach20() -> AnnData:
    """\
    10X sequencing from the developping (E15.5) enteric nervous system, it
    includes Schwann Cell precursors and two neuronal population generated
    via a bifurcation.

    Returns
    -------
    Annotated data matrix.
    """

    filename = "data/morarach20.h5ad"
    url = f"{url_datadir}data/morarach20.h5ad"
    return read(filename, backup_url=url, sparse=True, cache=True)


def neucrest19() -> AnnData:
    """\
    SS2 of neural crest cells at E8.5 from `Soldatov et al. (2019) <https://doi.org/10.1126/science.aas9536>`_.
    contains a multifucating trajectory.

    Returns
    -------
    Annotated data matrix.
    """

    filename = "data/neucrest.h5ad"
    url = f"{url_datadir}data/neucrest.h5ad"
    return read(filename, backup_url=url, sparse=True, cache=True)


def pancreas() -> AnnData:
    """\
    Data from `Bastidas-Ponce et al. (2019) <https://doi.org/10.1242/dev.173849>`_.
    Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition
    with transcriptome profiles sampled from embryonic day 15.5.

    This datasets has been processed via CellRank and then converted into a scFate tree.

    Returns
    -------
    Annotated data matrix.
    """

    filename = "data/pancreas_quad.h5ad"
    url = f"{url_datadir}data/pancreas_quad.h5ad"
    return read(filename, backup_url=url, sparse=True, cache=True)


def test_adata(plot=False) -> AnnData:
    """\
    10X sequencing from the developping (E15.5) enteric nervous system, it
    includes Schwann Cell precursors and two neuronal population generated
    via a bifurcation. If plot is set to True, processed bone marrow data
    will be loaded.

    Returns
    -------
    Annotated data matrix.
    """
    if plot:
        filename = "data/empty_tree.h5ad"
        url = f"{url_datadir}data/empty_tree.h5ad"
        return read(filename, backup_url=url, sparse=True, cache=True)
    else:
        filename = HERE / "test.h5ad"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
            return read(filename)
