from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

HERE = Path(__file__).parent

from scanpy import read


def morarach20() -> AnnData:
    """\
    10X sequencing from the developping (E15.5) enteric nervous system, it
    includes Schwann Cell precursors and two neuronal population generated
    via a bifurcation.

    Returns
    -------
    Annotated data matrix.
    """

    filename = HERE / "morarach20.h5ad"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
        return read(filename)


def test_adata() -> AnnData:
    """\
    10X sequencing from the developping (E15.5) enteric nervous system, it
    includes Schwann Cell precursors and two neuronal population generated
    via a bifurcation.

    Returns
    -------
    Annotated data matrix.
    """

    filename = HERE / "test.h5ad"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
        return read(filename)
