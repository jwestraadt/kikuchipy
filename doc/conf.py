# Configuration file for the Sphinx documentation app.

# This file only contains a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import inspect
import os
from os.path import relpath, dirname
import re
import sys

from numpydoc.docscrape_sphinx import SphinxDocString

from kikuchipy import release as kp_release
import kikuchipy


# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append("../")

# Project information
project = "kikuchipy"
copyright = f"2019-{datetime.now().year}, {kp_release.author}"
author = kp_release.author
version = kp_release.version
release = kp_release.version

master_doc = "index"

if "dev" in version:
    release_version = "develop"
else:
    release_version = "v" + version

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    #    "nbsphinx",
    "notfound.extension",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
]

# Create links to references within kikuchipy's documentation to these
# packages
intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/stable", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "diffsims": ("https://diffsims.readthedocs.io/en/latest", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyvista": ("https://docs.pyvista.org", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This image also
# affects html_static_path and html_extra_path.
exclude_patterns = ["build", "_static/logo/*.ipynb"]

# The theme to use for HTML and HTML Help pages. See the documentation
# for a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "kikuchipy",
    },
    "show_prev_next": False,
    "github_url": "https://github.com/pyxem/kikuchipy",
}
html_context = {
    "github_user": "pyxem",
    "github_repo": "kikuchipy",
    "github_version": release_version,
    "doc_path": "doc",
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
html_static_path = ["_static"]

# Syntax highlighting
pygments_style = "friendly"

# Logo
html_logo = "_static/logo/plasma_logo.svg"
html_favicon = "_static/logo/plasma_favicon.png"

# -- nbsphinx configuration
# Taken from nbsphinx' own nbsphinx configuration file, with slight
# modifications to point nbviewer and Binder to the GitHub develop
# branch links when the documentation is launched from a kikuchipy
# version with "dev" in the version

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = (
    r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <style>a:hover { text-decoration: underline; }</style>

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="""
    + f"'https://github.com/pyxem/kikuchipy/blob/{release_version}"
    + r"""/{{ docname|e }}'>{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="""
    + f"'https://mybinder.org/v2/gh/pyxem/kikuchipy/{release_version}"
    + r"""?filepath={{ docname|e }}'><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
)
# https://nbsphinx.readthedocs.io/en/0.8.0/never-execute.html
nbsphinx_execute = "auto"  # always, auto, never
nbsphinx_allow_errors = True
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]

# -- sphinxcontrib-bibtex configuration
bibtex_bibfiles = ["bibliography.bib"]


# -- Relevant for the PDF build with LaTeX
latex_elements = {
    # pdflatex doesn't like some Unicode characters, so a replacement
    # for one of them is made here
    "preamble": r"\DeclareUnicodeCharacter{2588}{-}"
}


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.
    This is taken from SciPy's conf.py:
    https://github.com/scipy/scipy/blob/develop/doc/source/conf.py.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(kikuchipy.__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("kikuchipy/"):
        m = re.match(r"^.*dev0\+([a-f0-9]+)$", kikuchipy.__version__)
        pre_link = "https://github.com/pyxem/kikuchipy/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in kikuchipy.__version__:
            return pre_link + "develop/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (kikuchipy.__version__, fn, linespec)
    else:
        return None


# -- Custom 404 page
notfound_context = {
    "body": (
        "<h1>Page not found.</h1>\n\nPerhaps try the "
        "<a href='http://kikuchipy.org/user_guide/index.html'>user guide page</a>."
    ),
}
notfound_no_urls_prefix = True


# -- Copy button customization (taken from PyVista)
# Exclude traditional Python prompts from the copied code
copybutton_prompt_text = r">>> ?|\.\.\. "
copybutton_prompt_is_regexp = True


# -- sphinx.ext.autodoc
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"


# -- numpydoc
numpydoc_show_class_members = False
numpydoc_use_plots = True
numpydoc_xref_param_type = True
numpydoc_validate = True
numpydoc_validation_checks = {
    "all",  # All but the following:
    "ES01",  # Not all docstrings need an extend summary.
    "EX01",  # Examples: Will eventually enforce
    "GL01",  # Contradicts numpydoc examples
    "PR04",  # Doesn't seem to work with type hints?
    "SA01",  # Not all docstrings need a "See Also"
    "SA04",  # "See Also" section does not need descriptions
    "SS06",  # Not possible to make all summaries one line
    "YD01",  # Yields: No plan to enforce
}


# -- matplotlib.sphinxext.plot_directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False


def _str_examples(self):
    examples_str = "\n".join(self["Examples"])
    if (
        self.use_plots
        and re.search(r"\b(.plot)\b", examples_str)
        and "plot::" not in examples_str
    ):
        out = []
        out += self._str_header("Examples")
        out += [".. plot::", ""]
        out += self._str_indent(self["Examples"])
        out += [""]
        return out
    else:
        return self._str_section("Examples")


SphinxDocString._str_examples = _str_examples


# -- CSS and other things
def setup(app):
    app.add_css_file("no_search_highlight.css")
