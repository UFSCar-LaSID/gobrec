# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'GOBRec'
copyright = '2026, LaSID'
author = 'Gregorio'
release = '0.4.0'
html_title = "GOBRec"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc"
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

templates_path = ['_templates']
exclude_patterns = []
html_show_sourcelink = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
}
html_baseurl = "https://UFSCar-LaSID.github.io/gobrec/"
html_static_path = ['_static']
html_css_files = ['custom.css']

html_context = {
    "display_github": True,
    "github_user": "UFSCar-LaSID",
    "github_repo": "gobrec",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


# Adds .nojekyll file to the output to avoid GitHub Pages ignoring files and folders that start with an underscore

def setup(app):
    app.connect("build-finished", create_nojekyll)

def create_nojekyll(app, exception):
    if app.builder.name == "html":
        nojekyll_path = os.path.join(app.outdir, ".nojekyll")
        with open(nojekyll_path, "w") as f:
            f.write("")