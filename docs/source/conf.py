# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from sphinxawesome_theme.postprocess import Icons
sys.path.insert(0, os.path.abspath('../../src'))

project = 'topic3d'
copyright = '2026, Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann'
author = 'Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann'
release = '0.0.0-dev.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # supports NumPy / Google docstrings
    "sphinx.ext.viewcode",          # Markdown support
    "sphinx_autodoc_typehints",
    "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_permalinks_icon = Icons.permalinks_icon
# html_theme_options = {
#     "logo_light": "path/to/light/logo",
#     "logo_dark": "path/to/dark/logo"
# }