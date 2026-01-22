project = "HLAVO"
author = "Jan Březina, Martin Špetlík, Pavel Exner, Jan Stebel"

extensions = [
    "myst_parser",
]

# umožní .md jako zdroj
source_suffix = {
    ".md": "markdown",
}

# hlavní dokument (root)
root_doc = "index"

# téma – lokálně i na RTD ok
html_theme = "sphinx_rtd_theme"

# (volitelné) zapnutí vybraných MyST rozšíření
myst_enable_extensions = [
    "colon_fence",   # ::: pro bloky (admonitions apod.)
    "deflist",
    "tasklist",
]

# (volitelné) upozornění na špatné odkazy apod.
nitpicky = False
