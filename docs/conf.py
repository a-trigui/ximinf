# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

project = 'ximinf'
copyright = '2025, Adam Trigui'
author = 'Adam Trigui'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "myst_parser",                # for Markdown
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
html_theme = "sphinx_rtd_theme"

# Optional for RTD theme customization
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
