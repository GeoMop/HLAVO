project = "HLAVO"
author = "Jan Březina, Martin Špetlík, Pavel Exner, Jan Stebel"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",  # <-- enable MathJax rendering
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
    "dollarmath",  # <-- parse $...$ and $$...$$ in MyST
    "amsmath",      # enables \[...\] and \(...\) and environments like align
]

# (volitelné) upozornění na špatné odkazy apod.
nitpicky = False


# Optional but often helpful: allow AMS math environments
myst_dmath_allow_labels = True
myst_amsmath_enable = True

# Optional: MathJax config tweaks (usually not required)
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    }
}