# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
import warnings

sys.path.append('..')

# sphinx_autodoc_typehints itself calls a Sphinx API removed in Sphinx 10; the deprecation comes from
# the extension, not from this project, so silence it to keep the build output clean.
warnings.filterwarnings('ignore', message=r'.*set_application.*is deprecated.*')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sfsutils'
year = datetime.datetime.now().year
copyright = f'{year}, Janek Sendrowski'
author = 'Janek Sendrowski'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'autodocsumm',  # per-class method-summary table at the top of each class
    'myst_nb',
    'sphinxcontrib.bibtex',
    'sphinx_book_theme'
]

# Page-level ``.. autosummary::`` blocks render an inline class table linking
# to the autoclass docs on the same page; no stub pages need generating.
autosummary_generate = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

bibtex_bibfiles = ['refs.bib']

typehints_use_signature = True
typehints_fully_qualified = False

# The plotting methods return the lazily-imported ``plt.Axes`` (matplotlib is imported inside the
# methods to avoid selecting a backend at import time), and many signatures reference the optional
# ``cyvcf2.Variant`` backend. Neither name is importable at documentation time, so sphinx_autodoc_typehints
# cannot resolve these forward references; the warning is cosmetic, so suppress that subtype.
suppress_warnings = ['sphinx_autodoc_typehints.forward_reference']

# Warn about cross-references that do not resolve. An unqualified role (``:class:`Spectra```) resolves
# inside the owning module's page but not in contexts without a module scope, such as the autosummary
# summary tables, where it silently degrades to plain text. Nitpicky mode turns that into a build
# warning so it cannot go unnoticed; refs to names outside the package are listed below.
nitpicky = True

nitpick_ignore_regex = [
    # standard library and typing constructs autodoc emits from annotations
    (r'py:.*', r'(typing\..*|Union|Optional|Any|Callable|Iterable|Iterator|Literal|Sequence|Mapping)'),
    # optional third-party backends, not importable at documentation time
    (r'py:.*', r'(np|pd|plt|sns)\..*'),
    (r'py:.*', r'(cyvcf2|tskit|zarr|Bio|tqdm|matplotlib|numpy|pandas|seaborn|scipy)(\..*)?'),
    (r'py:.*', r'(SeqRecord|FastaIterator|DictReader|TextIOWrapper)'),
    # private helpers that are deliberately undocumented
    (r'py:.*', r'.*\b_[A-Za-z_]+'),
    # the file-handler hierarchy is internal plumbing and is not documented; it surfaces only in the
    # ``Bases:`` line of the classes built on it, and documenting it would in turn dangle its own bases
    (r'py:class', r'sfsutils\.io_handlers\.(MultiHandler|FASTAHandler|VCFHandler|GFFHandler|FileHandler)'),
]

pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

# enable dollar-delimited and AMS math in MyST markdown (notebooks and .md)
myst_enable_extensions = ['dollarmath', 'amsmath']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'reference/Python/resources']

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    # autodocsumm: prepend a compact summary table (names + one-line descriptions)
    # before the full docs -- a class list after each module docstring and a method
    # list after each class docstring. Sections are ;;-separated; restricting to
    # Classes and Methods skips the Attributes table (it duplicates the per-attribute
    # docs below). Signatures are dropped to keep the tables to one line per entry.
    'autosummary': True,
    'autosummary-sections': 'Classes;;Methods',
    'autosummary-nosignatures': True
}

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'search_bar_text': 'Search...',
    'repository_url': 'https://github.com/Sendrowski/SFSUtils',
    'repository_branch': 'master',
    'use_repository_button': True,
    'use_edit_page_button': False,
    'use_issues_button': False
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "logo.png"
html_favicon = "favicon.ico"
