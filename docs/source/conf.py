# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../src/house_value_prediction'))
project = 'house_value_prediction'
copyright = '2024, Farhana Mohamed Asgar'
author = 'Farhana Mohamed Asgar'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['modules.rst']
extensions = [
    'sphinx.ext.autodoc',   # This extension enables the autodoc directive
    'sphinx.ext.napoleon',  # Optional: If you're using Google or NumPy style docstrings
    'sphinx.ext.viewcode',  # Optional: To include links to your source code in the docs
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_static_path = ['_static']
