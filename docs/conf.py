# Configuration file for Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath("../src"))  # Make your package importable

import importlib.metadata

# -- Project information -----------------------------------------------------
project = 'ximinf'
copyright = '2025, Adam Trigui'
author = 'Adam Trigui'

try:
    release = importlib.metadata.version("ximinf")  # Get version from package
except importlib.metadata.PackageNotFoundError:
    release = "0.0.50" # Fallback version if package not found

# Short X.Y version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",         # core autodoc
    "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    "sphinx.ext.viewcode",        # links to source code
    "myst_parser",                # Markdown support
    "sphinx_autodoc_typehints",   # type hints
]

# Autosummary generates summary tables
autosummary_generate = True

# Templates path
templates_path = ["_templates"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
