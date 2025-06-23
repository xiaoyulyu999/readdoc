# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AI随想'
copyright = '2025, Xiaoyu Lyu'
author = 'Xiaoyu Lyu'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx_code_tabs',
    'sphinx.ext.mathjax',
    'nbsphinx',
]



templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# Add this line
html_static_path = ['_static']

# Add this to inject your CSS
def setup(app):
    app.add_css_file('custom.css')  # for Sphinx >= 1.8

html_theme_options = {
    'collapse_navigation': False,  # <--- This is the key!
    'navigation_depth': 4,         # Optional: how deep to show nested items
}

