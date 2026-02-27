# Configuration file for the Sphinx documentation builder.

project = "KGLite"
copyright = "2024, Kristian dF Kollsgård"
author = "Kristian dF Kollsgård"

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# -- MyST (Markdown) settings ------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# -- Sphinx-AutoAPI settings --------------------------------------------------
# Parses .pyi stubs directly — no need to import the Rust extension module.

autoapi_dirs = ["../kglite"]
autoapi_type = "python"
autoapi_file_patterns = ["*.pyi"]
autoapi_ignore = ["**/code_tree/**"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_python_class_content = "both"  # show class docstring + __init__ docstring
autoapi_member_order = "groupwise"

# Suppress warnings about not being able to import the Rust extension
# and cross-reference warnings from included markdown files (CYPHER.md, FLUENT.md)
# whose relative links target GitHub paths that don't exist in the Sphinx tree.
suppress_warnings = ["autoapi.python_import_resolution", "myst.xref_missing"]

# -- General settings ---------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = "KGLite"
html_theme_options = {
    "source_repository": "https://github.com/kkollsga/kglite",
    "source_branch": "main",
    "source_directory": "docs/",
}
