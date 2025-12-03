# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
import importlib.metadata

project = 'ximinf'
copyright = '2025, Adam Trigui'
author = 'Adam Trigui'

# conf.py
try:
    # Replace 'your_package_name' with your actual package name
    release = importlib.metadata.version("ximinf")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.3"

# Short X.Y version for display
version = ".".join(release.split(".")[:2])


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
