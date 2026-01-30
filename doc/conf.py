# conf.py

project = "HLAVO"
author = "Jan Březina, Martin Špetlík, Pavel Exner, Jan Stebel"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",   # HTML math rendering
    "sphinx.ext.imgconverter",  # optional alternative backend plugin if you use rsvg-convert
    #"sphinxcontrib_svg2pdfconverter",
]

# allow .md sources
source_suffix = {
    ".md": "markdown",
}

# main document (root)
root_doc = "index"

# ---------- HTML ----------
html_theme = "sphinx_rtd_theme"

# ---------- MyST ----------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "dollarmath",
    "amsmath",
]

nitpicky = False

myst_dmath_allow_labels = True
myst_amsmath_enable = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    }
}

# ---------- LaTeX / PDF ----------
# Use xelatex for Unicode (č, ř, ž, …) and generally fewer encoding headaches.
latex_engine = "lualatex"

# This defines a single PDF "manual" (book-like) generated from root_doc.
latex_documents = [
    (root_doc, "hlavo.tex", 
     f"{project} Documentation", author, 
     howto)         # article style (without chapters)
    #"manual"),     # book style (with chapters)
]

# Basic LaTeX settings that work well for technical docs.
latex_elements = {
    # Paper size and font size:
    "papersize": "a4paper",
    "pointsize": "11pt",

    # If you want a slightly roomier layout, uncomment:
    # "geometry": r"\usepackage[a4paper,margin=2.5cm]{geometry}",

    # Helpful packages for code/listings formatting
    # (Sphinx already includes a lot, but this doesn't hurt.)
    "preamble": r"""
\usepackage{unicode-math}
""",
}

# Optional: show deeper ToC in the PDF sidebar/ToC
latex_show_urls = "footnote"

# If you have a logo, you can enable this (path relative to doc dir):
latex_logo = "graphics/logo_TACR_zakl.pdf"
