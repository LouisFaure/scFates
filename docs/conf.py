import sys
import os
import inspect
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Mapping
import mock

MOCK_MODULES = [
    "scanpy",
    "joblib",
    "tqdm",
    "scikit-misc",
    "numba",
    "seaborn",
    "statsmodels",
    "plotly",
    "adjustText",
    "statsmodels.stats.multitest",
    "statsmodels.formula.api",
    "statsmodels.api",
    "igraph",
    "statsmodels.stats.weightstats",
    "skmisc.loess",
    "statsmodels.stats.multitest",
    "scanpy.plotting._utils",
    "scanpy.plotting._tools.scatterplots",
    "plotly.express",
    "plotly.graph_objects",
    "phenograph",
    "sklearn.preprocessing",
    "tqdm",
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

from sphinx.application import Sphinx
from sphinx.ext import autosummary

import matplotlib  # noqa

matplotlib.use("agg")

HERE = Path(__file__).parent
sys.path.insert(0, f"{HERE.parent}")


import scFates


logger = logging.getLogger(__name__)


# -- General configuration ------------------------------------------------

needs_sphinx = "2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinxext.opengraph",
    "sphinx_autodoc_typehints",
    "readthedocs_ext.readthedocs",
    "sphinx_copybutton",
    "nbsphinx",
    "scanpydoc",
    "docutils",
]

ogp_site_url = "https://scfates.readthedocs.io/"
ogp_image = "https://scfates.readthedocs.io/en/latest/_images/scFates_logo_dark.png"

# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
    scanpy=("https://scanpy.readthedocs.io/en/latest/", None),
    cuml=("https://docs.rapids.ai/api/cuml/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    cugraph=("https://docs.rapids.ai/api/cugraph/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
)

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"

# General information about the project.
project = "scFates"
author = "Louis Faure"
title = "Tree learning on scRNAseq"

version = scFates.__version__.replace(".dirty", "")
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False


# -- Retrieve notebooks ------------------------------------------------

from urllib.request import urlretrieve

notebooks_url = "https://github.com/LouisFaure/scFates_notebooks/raw/main/"
notebooks = [
    "Basic_pseudotime_analysis.ipynb",
    "Advanced_bifurcation_analysis.ipynb",
    "Conversion_from_CellRank_pipeline.ipynb",
    "Critical_Transition.ipynb",
]
for nb in notebooks:
    try:
        urlretrieve(notebooks_url + nb, nb)
    except:
        pass


# -- Options for HTML output ----------------------------------------------

html_theme = "scanpydoc"
html_theme_options = {
    "titles_only": True,
    "logo_only": True,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="LouisFaure",  # Username
    github_repo="scFates",  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)

html_show_sphinx = False
html_logo = "_static/scFates_Logo.svg"
html_static_path = ["_static"]
html_extra_path = ["_extra"]


def setup(app):
    app.add_css_file("custom.css")
