import sys
import os
import inspect
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Mapping

from sphinx.application import Sphinx
from sphinx.ext import autosummary

import matplotlib  # noqa

matplotlib.use("agg")

HERE = Path(__file__).parent
sys.path.insert(0, f"{HERE.parent.parent}")
sys.path.insert(0, os.path.abspath("_ext"))
import scFates

logger = logging.getLogger(__name__)


# -- General configuration ------------------------------------------------

needs_sphinx = "1.7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    'readthedocs_ext.readthedocs'
]


# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_custom_sections = [("Params", "Parameters")]

intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
    scanpy=("https://scanpy.readthedocs.io/en/latest/", None),
)

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"

# General information about the project.
project = "scFates"
author = "Louis Faure"
title = "Tree learning on scRNAseq"

version = "0.1"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False



# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = dict(navigation_depth=1, titles_only=True)
html_static_path = ["_static"]


def setup(app):
    app.add_stylesheet("custom.css")


# -- Prettier Param docs --------------------------------------------

from typing import Dict, List, Tuple
from docutils import nodes
from sphinx import addnodes
from sphinx.domains.python import PyTypedField, PyObject
from sphinx.environment import BuildEnvironment


class PrettyTypedField(PyTypedField):
    list_type = nodes.definition_list

    def make_field(
        self,
        types: Dict[str, List[nodes.Node]],
        domain: str,
        items: Tuple[str, List[nodes.inline]],
        env: BuildEnvironment = None,
    ) -> nodes.field:
        def makerefs(rolename, name, node):
            return self.make_xrefs(rolename, domain, name, node, env=env)

        def handle_item(
            fieldarg: str, content: List[nodes.inline]
        ) -> nodes.definition_list_item:
            head = nodes.term()
            head += makerefs(self.rolename, fieldarg, addnodes.literal_strong)
            fieldtype = types.pop(fieldarg, None)
            if fieldtype is not None:
                head += nodes.Text(" : ")
                if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                    (text_node,) = fieldtype  # type: nodes.Text
                    head += makerefs(
                        self.typerolename, text_node.astext(), addnodes.literal_emphasis
                    )
                else:
                    head += fieldtype

            body_content = nodes.paragraph("", "", *content)
            body = nodes.definition("", body_content)

            return nodes.definition_list_item("", head, body)

        fieldname = nodes.field_name("", self.label)
        if len(items) == 1 and self.can_collapse:
            fieldarg, content = items[0]
            bodynode = handle_item(fieldarg, content)
        else:
            bodynode = self.list_type()
            for fieldarg, content in items:
                bodynode += handle_item(fieldarg, content)
        fieldbody = nodes.field_body("", bodynode)
        return nodes.field("", fieldname, fieldbody)


# replace matching field types with ours
PyObject.doc_field_types = [
    PrettyTypedField(
        ft.name,
        names=ft.names,
        typenames=ft.typenames,
        label=ft.label,
        rolename=ft.rolename,
        typerolename=ft.typerolename,
        can_collapse=ft.can_collapse,
    )
    if isinstance(ft, PyTypedField)
    else ft
    for ft in PyObject.doc_field_types
]