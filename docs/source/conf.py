# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import pygments
import a11y_pygments
sys.path.insert(0, os.path.abspath("../../src/"))

project = "topic3d"
copyright = "2026, Nollmann Lab"
author = "Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann"
release = "0.0.0-dev.11"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "myst_nb"
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "breeze"
html_theme_options = {"default_mode": "auto",
                      "external_links": ["https://github.com/ggautham07/topic3d",
                                         ],
                      "header_tabs": False}
html_static_path = ["_static"]
html_title = "topic3d documentation"
html_logo = "_static/logo-icon.png"
pygments_light_style = "a11y-high-contrast-light"
pygments_dark_style = "github-dark-high-contrast"