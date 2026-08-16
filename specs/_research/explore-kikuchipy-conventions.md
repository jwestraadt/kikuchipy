# kikuchipy conventions for a new indexing feature — exhaustive extraction

Repo root: `c:/Users/westraadt.1/Repos/kikuchipy` (branch `develop`, version `0.13.dev0`). All paths below are absolute unless clearly repo-relative inside a quoted config.

---

## 1. Code style rules

### 1.1 Formatters / linters (authoritative config)

`c:/Users/westraadt.1/Repos/kikuchipy/pyproject.toml`
- L195–215 `[tool.ruff.lint]`:
  - `exclude = ["*.ipynb"]` (L197)
  - `select = ["F", "E", "W", "I"]` (pyflakes, pycodestyle, isort) (L198–203)
  - `extend-ignore = ["E501", "E402"]` (L206–209) → **no hard line-length lint**, but ruff-format/black default 88 applies to code.
  - `[tool.ruff.lint.isort] force-sort-within-sections = true` (L211–212) → within each import block, `import x` and `from x import y` are sorted together alphabetically by module name (see `ebsd.py` L22–105 for the canonical example).
  - `[tool.ruff.format] exclude = ["*.ipynb"]` (L214–215)

`c:/Users/westraadt.1/Repos/kikuchipy/.pre-commit-config.yaml`
- L22–26: `ruff` + `ruff-format` (rev `v0.15.14`)
- L28–33: `black-jupyter` (rev `26.5.1`) applied **only to `\.ipynb`** with `args: [--line-length=77]` → notebook code cells are black-formatted at 77 chars.
- L34–38: `licenseheaders` hook with `-t .license.tmpl -cy -n kikuchipy -f` (currently in `ci.skip`, L45).
- L41–45: pre-commit.ci: `autofix_prs: false`, `autoupdate_schedule: monthly`.

Install requirement: `pre-commit install` (doc/dev/code_style.rst L11–13).

### 1.2 Style rules from `c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/code_style.rst`
- L6–10: PEP 8 + Black code style, run via ruff in pre-commit.
- L15–17: docstrings follow **numpydoc** (`:doc:`numpydoc <numpydoc:format>``), validated at doc build time.
- L19–20: **comment and docstring lines preferably ≤ 72 characters (including leading whitespace)**.
- L22–24: imports in **three blocks separated by blank lines**: stdlib → third-party → kikuchipy.
- L26–43: **type hints in the signature, no type duplication in the docstring**. Canonical form:
  ```
  def my_function(a: int, b: bool | None = None) -> tuple[float, np.ndarray]:
      """This is a new function.

      Parameters
      ----------
      a
          Explanation of ``a``.
      b
          Explanation of flag ``b``. Default is ``None``.

      Returns
      -------
      values
          Explanation of returned values.
      """
  ```
- L45–46: **lazy module imports via PEP 562** (`lazy_loader`).

Exceptions actually seen in code: types *are* written in docstrings for (a) properties (`value : EBSDDetector`, ebsd.py L208–211), (b) class `Parameters` documenting `**kwargs`-set attributes (ebsd.py L126–140), (c) private helpers whose parameter is not annotated (`metric : SimilarityMetric`, `_dictionary_indexing.py` L215).

### 1.3 License header (mandatory in every file)
Template `c:/Users/westraadt.1/Repos/kikuchipy/.license.tmpl` (GPL-3.0-or-later, `Copyright 2019-${years} the ${projectname} developers`).
Two header variants coexist:
- New style (with leading/trailing bare `#` lines, lowercase "the"): `src/kikuchipy/indexing/_dictionary_indexing.py` L1–18, `src/kikuchipy/_utils/deprecated.py` L1–18, `conftest.py` L1–18.
- Old style: `src/kikuchipy/indexing/__init__.py` L1–16 (`Copyright 2019-2024 The kikuchipy developers`). Use the **new** style for new files.
YAML files use `##`-prefixed headers (`.github/workflows/tests.yml` L1–18).

### 1.4 Public API exposure — `lazy_loader.attach_stub` pattern
Every package `__init__.py` is a 4-line stub; the real API lives in the sibling `__init__.pyi`:
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/__init__.py` L20–43: `import lazy_loader`; `credits = [...]` (L23–37); `__version__ = "0.13.dev0"` (L39); `__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)` (L41); `del lazy_loader` (L43).
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/__init__.pyi` L18–48: explicit `from . import (...)` + `from .io._io import load` + `from .logging import set_log_level`, then `__all__` grouped `# Functions` / `# Modules`.
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/__init__.py` L18–30: module docstring + stub.
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/__init__.pyi` L18–42: imports from **private** modules, `__all__` **alphabetically sorted**:
  ```
  from ._hough_indexing import xmap_from_hough_indexing_data
  from ._merge_crystal_maps import merge_crystal_maps
  from ._orientation_similarity_map import orientation_similarity_map
  from ._refinement._refinement import (compute_refine_orientation_projection_center_results, compute_refine_orientation_results, compute_refine_projection_center_results)
  from .similarity_metrics._normalized_cross_correlation import NormalizedCrossCorrelationMetric
  from .similarity_metrics._normalized_dot_product import NormalizedDotProductMetric
  from .similarity_metrics._similarity_metric import SimilarityMetric
  __all__ = [... 9 names, sorted, classes first by capitalization ...]
  ```
- Other stubs: `src/kikuchipy/pattern/__init__.pyi` L18–44, `src/kikuchipy/data/__init__.pyi` L18–37, `src/kikuchipy/signals/util/__init__.pyi`, `src/kikuchipy/_utils/__init__.pyi` L18–23.
- `src/kikuchipy/signals/__init__.pyi` additionally carries a module docstring with an embedded `.. autosummary:: :template: custom-module-template.rst` rubric (L18–30).

**Private module naming**: leading underscore for every non-public implementation module (`_dictionary_indexing.py`, `_hough_indexing.py`, `_merge_crystal_maps.py`, `_orientation_similarity_map.py`, `_refinement/_refinement.py`, `_solvers.py`, `_objective_functions.py`, `similarity_metrics/_similarity_metric.py`, `detectors/_ebsd_detector.py`, `signals/util/_crystal_map.py`, `_dask.py`, `_master_pattern.py`). Private *packages* also carry an explicit warning docstring: `src/kikuchipy/indexing/_refinement/__init__.py` L18–29 (“This module and documentation is only relevant for kikuchipy developers… .. warning: … for internal use only”).

`similarity_metrics/__init__.py` is header-only (16 lines, no stub) — sub-subpackages that expose nothing need no `.pyi`.

### 1.5 HyperSpy extension registration
`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/hyperspy_extension.yaml` L18–75 registers only *signal classes* (EBSD, LazyEBSD, EBSDMasterPattern, …). Entry point in pyproject L117–118 (`[project.entry-points."hyperspy.extensions"] kikuchipy = "kikuchipy"`). **A new `EBSD` method requires no change here**; a new signal class would.

### 1.6 Logging
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/logging.py` L21–53: public `set_log_level(level: int | str)`, exported in root `.pyi`.
- Module loggers: `_logger = logging.getLogger(__name__)` at `src/kikuchipy/signals/ebsd.py` L115 and `src/kikuchipy/signals/_kikuchi_master_pattern.py` L31.
- User-visible progress/info is printed with `print(...)`, not logged: `_dictionary_indexing.py` L77–85, L136–139; `_hough_indexing.py` L234, L242; `_refinement.py` L111, L117 (`file=sys.stdout`), L427.

### 1.7 Warnings & errors
- Plain `warnings.warn("...")` for soft user warnings — `ebsd.py` L260, L262 (static background setter).
- `ValueError` with an f-string naming both offending and expected values is the dominant validation style: `ebsd.py` L1933–1964, L2942–2946, L3068–3072; `_hough_indexing.py` L87, L93, L303–305, L312–315, L377–401, L457–483.
- Validation helpers take `raise_if_not: bool = False` and return `bool` (`_indexer_is_compatible_with_kikuchipy` `_hough_indexing.py` L339–406; `_phase_lists_are_compatible` L409–490; `_detector_is_compatible_with_signal`, `_xmap_is_compatible_with_signal`).
- Custom exceptions live in `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/_utils/exceptions.py` (`UnknownHemisphereError`, `UnknownProjectionError`, both `ValueError` subclasses that build the message from `given`).
- Missing optional dependency → `ImportError` via `verify_dependency_or_raise` (§7).
- `# pragma: no cover` on unreachable/optional branches: `_constants.py` L46, L63, L73; `ebsd.py` L107, L1675, L1697, L1771, L1803, L2871; `_similarity_metric.py` L228, L235, L242.

### 1.8 Deprecations
Utility module is **`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/_utils/deprecated.py`** (not `_deprecated.py`), exporting two decorator **classes** via `_utils/__init__.pyi` L18–23:
- `class deprecated` (L30–111): `__init__(since, alternative=None, alternative_is_function=True, removal=None)`; emits `VisibleDeprecationWarning` via `warnings.warn_explicit` and rewrites `__doc__` with a `.. deprecated:: <since>` directive under `Notes`.
- `class deprecated_argument` (L114–158): `__init__(name, since, removal, alternative=None)`; warns only if the argument appears in `kwargs`.
- `VisibleDeprecationWarning` is imported from `kikuchipy._constants` (`_constants.py` L79–86, NumPy-version shim).

`c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/handling_deprecations.rst`
- L4–9: semantic versioning; one minor-release heads-up before removal.
- L11–32: decorator goes **immediately above the signature**, below `@property` when both apply. **NB: the doc uses the names `deprecate` / `deprecate_argument`, which do not exist in the code — the real names are `deprecated` / `deprecated_argument`.**
- Changelog convention: a `Deprecated` section stating the version at which the warning becomes an error (CHANGELOG.rst L91–101).

---

## 2. Test conventions

### 2.1 Layout
- `c:/Users/westraadt.1/Repos/kikuchipy/tests/` with one subpackage per source module: `test_data`, `test_detectors`, `test_draw`, `test_filters`, `test_imaging`, `test_indexing`, `test_io`, `test_pattern`, `test_signals`, `test_simulations`, `test_utils`. Every dir has an `__init__.py` carrying the license header only (`tests/test_indexing/__init__.py`).
- Indexing tests: `tests/test_indexing/{test_dictionary_indexing.py (180 L), test_ebsd_refinement.py (1343 L), test_merge_crystal_maps.py (686 L), test_orientation_similarity_map.py (64 L), test_similarity_metrics.py (54 L)}`. Hough indexing lives with the signal: `tests/test_signals/test_ebsd_hough_indexing.py` (389 L).
- Tests are shipped in the wheel: pyproject L140–142 `[tool.hatch.build.targets.wheel.force-include] "tests/" = "kikuchipy/tests"`, `"conftest.py" = "kikuchipy/conftest.py"`.
- Test classes are `class TestXxx:` grouping by feature; methods `def test_...(self, fixture)`. Class-level setup via `def setup_method(self)` (`test_ebsd_hough_indexing.py` L39–45) or a shared base class holding class attributes (`class EBSDRefineTestSetup` → `class TestEBSDRefine(EBSDRefineTestSetup)`, `test_ebsd_refinement.py` L17–43).

### 2.2 pytest configuration — `pyproject.toml` L152–193
```
addopts = ["-ra", "--import-mode=importlib", "--strict-markers", "--benchmark-skip",
           "--ignore=doc/_static/image/doc_reference_frames.py",
           "--ignore=examples/*/*.py",
           "--ignore=src/kikuchipy/data/emsoft_ebsd/create_dummy_emsoft_ebsd_file.py",
           "--ignore=src/kikuchipy/data/oxford_h5ebsd/create_oxford_h5ebsd_file.py",
           "--ignore=src/kikuchipy/data/oxford_binary/create_oxford_binary_file.py",
           "--ignore-glob=src/kikuchipy/data/emsoft_ebsd_master_pattern/*.py"]
doctest_optionflags = "NORMALIZE_WHITESPACE"
filterwarnings = [...11 ignores, L171-189...]
markers = ["weekly: mark test as running only weekly on CI"]
xfail_strict = true
```
- `--strict-markers` ⇒ any new marker must be declared in `markers`.
- Coverage: L144–150 `[tool.coverage.report] precision = 2`; `[tool.coverage.run] branch = false`, `source = ["src/kikuchipy"]`, `relative_files = true`.

### 2.3 conftest.py — `c:/Users/westraadt.1/Repos/kikuchipy/conftest.py` (873 L)
- L20–26: **must stay in the repo top directory** so `pytest --doctest-modules src` discovers it (issue #744).
- L64–83: `pytest_addoption` adds `--weekly`; `MARKERS = ["weekly"]`; `pytest_runtest_setup` skips marked tests unless the flag is given.
- L89–99: PyVista guarded by `dependency_version["pyvista"] is not None`; `pv.OFF_SCREEN = True`; fixture `skipif_no_vtk_support` (L96–99) using `system_supports_plotting()`.
- L105–107: `pytest_sessionstart` **downloads `nickel_ebsd_large(allow_download=True)`** and sets `matplotlib` backend `agg`.
- L113–129: autouse `doctest_setup_teardown` (plt.ioff, HyperSpy progressbar off, chdir into a `TemporaryDirectory`).
- L131–133: autouse `import_to_namespace` injecting `DATA_PATH` into the doctest namespace.
- Fixture index (name @ line): `assert_dictionary_func` 139; **`dummy_signal` 161** (EBSD `<3,3|3,3>`, hard-coded uint8 data, `xmap` with 2 phases, per-point `pc`, `detector`; docstring warns the data must not change); `dummy_background` 217; **`ebsd_with_axes_and_random_data` 227** (parametrized `[(nav_shape, sig_shape, lazy, dtype)]`, used `indirect=True`); `nickel_structure` 265; `nickel_phase` 274; `pc1` 279; **`detector` 285** (parametrized `[[(1,), (60, 60)]]`, TSL convention); `rotations` 302; **`get_single_phase_xmap` 307** (factory returning a `CrystalMap` with `scores`/`simulation_indices` props); `save_path_hdf5` 341; `save_path_nordif` 350; `ni_small_axes_manager` 355; `ebsd_directory` 377; `kikuchipy_h5ebsd_path` 403; `nickel_ebsd_large_h5ebsd_renamed` 408; `edax_binary_path` 419; `edax_binary_file` 424; `edax_h5ebsd_path` 481; `oxford_binary_path` 489; `oxford_binary_file` 494; `oxford_h5ebsd_file` 577; `emsoft_ebsd_master_pattern_file` 594; `emsoft_ebsd_path` 599; `emsoft_ebsd_file` 604; `emsoft_ebsd_master_pattern_metadata` 609; `emsoft_ebsd_master_pattern_axes_manager` 618; `emsoft_ecp_master_pattern_file` 661; `emsoft_tkd_master_pattern_file` 742; `nordif_path` 823; `nordif_renamed_calibration_pattern` 828; `bruker_path` 842; `bruker_h5ebsd_file` 847; `bruker_h5ebsd_roi_file` 855; `bruker_h5ebsd_nonrectangular_roi_file` 865.
- **There is no `nickel_ebsd_small` fixture** — tests call `kp.data.nickel_ebsd_small()` directly (e.g. `tests/test_signals/test_ebsd_hough_indexing.py` L40, `benchmarks/indexing/test_dictionary_indexing.py` L30).
- Test files **cannot import from conftest** (L25–26).

### 2.4 Optional-dependency skipping
Canonical pattern (module-level constant or inline decorator) using `kikuchipy._constants.dependency_version`:
- `tests/test_signals/test_ebsd_hough_indexing.py` L35–37: `@pytest.mark.skipif(dependency_version["pyebsdindex"] is None, reason="pyebsdindex is not installed")` on the whole class; also the inverse at L371 for the “not installed” error path.
- `tests/test_detectors/test_ebsd_detector.py` L32–36: named module constants `skipif_pyebsdindex_installed` / `skipif_pyebsdindex_not_installed`.
- NLopt: `tests/test_indexing/test_ebsd_refinement.py` L216, 240, 452, 581, 608, 873, 1042, 1156 (`dependency_version["nlopt"] is None/is not None`).
- PyVista: `tests/test_signals/test_ebsd_master_pattern.py` L543, L567.
- `pytest.importorskip` for module-wide skips: `tests/test_detectors/test_ebsd_detector_signals.py` L22 (`psygnal`), `tests/test_draw/test_ebsd_detector_plots_widgets.py` L22 (`ipywidgets`).

### 2.5 Numba testing — `doc/dev/running_writing_tests.rst` L70–81
- A numba function is only *covered* if called as `numba_func.py_func()`.
- **Always test both** `numba_func(...)` and `numba_func.py_func(...)` (machine code may differ per OS; issue #496).
- Examples: `tests/test_indexing/test_similarity_metrics.py` L43–53 (`_ncc_single_patterns_1d_float32_exp_centered` both ways, `np.isclose(r1, r2)`); `tests/test_indexing/test_ebsd_refinement.py` L45–60 (`_prepare_pattern` and `_prepare_pattern.py_func`).

### 2.6 multiprocessing / xdist — `doc/dev/running_writing_tests.rst` L83–93
Tests calling `multiprocessing` (e.g. `pyebsdindex.pcopt.optimize_pso`) must only run the multiprocessing part when the xdist `worker_id` fixture equals `"master"`. Implementation: `tests/test_signals/test_ebsd_hough_indexing.py` L327, L337 (`def test_optimize_pc_pso(self, worker_id): ... if worker_id == "master":`).

### 2.7 Benchmarks — `doc/dev/improving_performance.rst`
- L10–14: “To check whether a change is an improvement or a regression, a benchmark should be written… stored in the top directory `kikuchipy/benchmarks`… run with `pytest --benchmark-only`”.
- Framework: **pytest-benchmark** (pyproject L99). Skipped by default via `--benchmark-skip` (pyproject L157).
- Files: `c:/Users/westraadt.1/Repos/kikuchipy/benchmarks/indexing/test_dictionary_indexing.py` (63 L) and `.../test_refinement.py` (115 L). Naming is `test_<feature>.py` with plain functions `def test_dictionary_indexing(benchmark):` (L24) that call `benchmark(s.dictionary_indexing, dictionary=..., signal_mask=..., keep_n=1)` (L54–59) and then assert a relaxed numerical result (`np.isclose(xmap.scores.mean(), 0.1887, atol=1e-4)`, L63). Shared setup as a module-level helper function (`ebsd_refinement_benchmark_setup()`, test_refinement.py L25–54).
- No ASV configuration exists.

### 2.8 Exact commands
From `doc/dev/running_writing_tests.rst`, `building_writing_documentation.rst`, `setting_up_development_installation.rst`, `improving_performance.rst`:
```
pre-commit install                       # code style hooks
pip install --editable ".[dev]"          # everything (doc+tests+coverage+ruff+black+isort+hatch)
pip install -e ".[tests,coverage]"       # tests only
pytest --cov                             # run tests + terminal coverage
coverage html                            # htmlcov/index.html
pytest -n 4                              # xdist, 4 CPUs
pytest -k TestEBSD                       # single class/function
pytest --reruns 2                        # pytest-rerunfailures for flaky tests
pytest --doctest-modules src             # docstring examples (run from top dir)
pytest --benchmark-only                  # benchmarks
pytest --weekly                          # weekly-marked tests
./doc/tutorials/run_nbval.sh             # nbval check of stored-output notebooks
pip install --editable ".[doc]" ; cd doc ; make html      # docs
cd doc ; make clean ; make linkcheck                      # cleanup / link check
```
CI (`.github/workflows/tests.yml`): `PYTEST_ARGS: --cov-branch --cov-report=xml --reruns 2 -n 4 --cov=kikuchipy` (L39); doctests `xvfb-run pytest src --doctest-modules --doctest-continue-on-failure` (L107, `continue-on-error: true`); `coverage report --show-missing` (L123); Codecov upload (L125–129); `timeout-minutes: 15` (L36). Matrix (L43–52): ubuntu/windows/macOS × Python 3.13/3.14, plus ubuntu+3.10 “oldest” pinned set (`dask==2021.8.1 diffsims==0.5.2 hyperspy==2.2 matplotlib==3.6 numba==0.57 numpy==1.23.0 orix==0.12.1 pooch==1.3.0 pyebsdindex==0.3.9.2 scikit-image==0.21.0`) and ubuntu+3.14 “minimum_requirement” (no optional deps). Optional deps installed only on Linux/Windows (`pip install -e ".[all]" ; pip install pyopencl`, L72–77); macOS skips nlopt (L79–84).
Weekly (`.github/workflows/weekly.yml`): notebooks via nbval (L54–56) and `pytest --weekly --reruns 2 -n 4` (L86–88), cron `15 6 * * 1`.
CI skip: put `"[skip ci]"` in the commit message (`doc/dev/continuous_integration.rst` L9–10).

---

## 3. Documentation conventions

### 3.1 Structure & framework — `doc/dev/building_writing_documentation.rst`
- L4–8: three doc categories — **examples**, **tutorials**, **reference** — following the **Diátaxis** framework; new documents must fit one.
- L40–42: Sphinx-Gallery builds `examples/` (top-level dir) into `doc/examples/`.
- L44–46: nbsphinx converts notebooks into tutorials; **notebook code must be black-formatted**.
- L119–126: “Writing API reference” — inherited attributes/methods are not listed unless explicitly coded in the inheriting class (see `EBSDMasterPattern` inheriting from private `KikuchiMasterPattern`, `src/kikuchipy/signals/_kikuchi_master_pattern.py` L34–38 and the re-declared thin wrappers in `ebsd_master_pattern.py` L388–500 and `ebsd.py` L3111–3183).

### 3.2 Tutorial notebook rules (`building_writing_documentation.rst` L48–117)
- L53–57: first cell is a **Markdown cell** with the text “This notebook is part of the kikuchipy documentation https://kikuchipy.org. Links to the documentation won't work from the notebook.” and cell metadata `{"nbsphinx": "hidden"}`. Verified: `doc/tutorials/hough_indexing.ipynb` cell 0 metadata `{'nbsphinx': 'hidden'}`.
- L58–59: silence matplotlib output with `_ = ax[0].imshow(...)`.
- L60–62: link API as `[fft_filter()](../reference/generated/kikuchipy.signals.EBSD.fft_filter.rst)` (parentheses required for callables).
- L63–64: cross-notebook links `[image quality](feature_maps.ipynb#image-quality)`.
- L65–66: external APIs as plain Markdown URLs.
- L67–70: **thumbnail** set by adding the `nbsphinx-thumbnail` tag to a code cell with image output, and the notebook **must be added to the appropriate topic in `doc/tutorials/index.rst`**. Verified format (`hough_indexing.ipynb` cell 51): `metadata = {"nbsphinx-thumbnail": {"tooltip": "Hough indexing using PyEBSDIndex"}, "tags": ["nbsphinx-thumbnail"]}`.
- L71–76: must be readable in light *and* dark `pydata_sphinx_theme`; print axes managers explicitly (`print(s.axes_manager)`), white figure backgrounds.
- **L77–83 (output policy)**: nbsphinx only executes notebooks *without* stored cell output; **notebooks should be stored without cell output**; store output only when too heavy for RTD (15 min / 3 GB limit).
  Measured state: `hough_indexing.ipynb` 63 cells, **0 code cells with outputs**; `feature_maps.ipynb` 16 code cells, 0 with outputs; `pattern_matching.ipynb` 48 code cells, **40 with outputs**; `hybrid_indexing.ipynb` 66 code cells, 17 with outputs; `pc_fit_plane.ipynb` 37/30.
  Notebook metadata is minimal: only `kernelspec` (`Python 3 (ipykernel)` / `python3`) and `language_info`; `nbformat 4.5`.
- L84–93: black also formats notebook cells; opt out with `# fmt: off` / `# fmt: on`.
- L94–100: PyVista Jupyter backend set with `pyvista.set_jupyter_backend("static")`.
- L102–112: notebooks with stored output are re-validated **weekly with nbval**.
- L114–117: Binder uses the root `environment.yml` (`--editable .[doc,all]`).

### 3.3 nbsphinx / nbval settings
`c:/Users/westraadt.1/Repos/kikuchipy/doc/conf.py`
- L55–71 `extensions`: `matplotlib.sphinxext.plot_directive`, `nbsphinx`, `sphinxcontrib.bibtex`, autodoc, autosummary, doctest, imgconverter, intersphinx, linkcode, mathjax, `numpydoc`, `sphinx_codeautolink`, `sphinx_copybutton`, `sphinx_design`, `sphinx_gallery.gen_gallery`.
- L75–114 `intersphinx_mapping` (add an entry when you reference a new library’s docs, e.g. a SHT package).
- L133–163: `pydata_sphinx_theme`, `use_edit_page_button`, `github_version: "develop"`.
- L165–198: `nbsphinx_execute = "auto"`, `nbsphinx_allow_errors = True`, `nbsphinx_execute_arguments = ["--InlineBackend.rc=figure.facecolor='w'"]`, `nbsphinx_prolog` injecting the Binder/GitHub admonition (branch `develop` when version contains “dev”).
- L203–207: `bibtex_bibfiles = ["user/bibliography.bib"]`, `bibtex_reference_style = "author_year"`.
- L300–308: `autosummary_ignore_module_all = False`, `autosummary_imported_members = True`, `autodoc_typehints_format = "short"`, `autodoc_default_options = {"show-inheritance": True}`.
- L310–334 **numpydoc validation**: `numpydoc_show_class_members = False`, `numpydoc_use_plots = True`, `numpydoc_xref_param_type = True`, and `numpydoc_validation_checks = {"all", "ES01","EX01","GL01","GL02","GL07","GL08","PR01","PR02","PR04","RT01","SA01","SA04","SS06","YD01"}` (i.e. *all* checks except those listed). Practical implications: summary must be present and correctly punctuated, `Returns` names required, parameter order/description enforced.
- L336–365: `plot_directive` config + monkeypatch of `SphinxDocString._str_examples` that auto-wraps `Examples` containing `.plot`/`.imshow` in `.. plot::`.
- L367–380 Sphinx-Gallery: `examples_dirs = "../examples"`, `gallery_dirs = "examples"`, `backreferences_dir = "reference/generated"`, `doc_module = ("kikuchipy",)`, `filename_pattern = "^((?!sgskip).)*$"`, `show_memory = True`.
- L383–393: `custom_setup()` downloads `nickel_ebsd_large` and `si_ebsd_moving_screen(0)` at build time.

nbval: `c:/Users/westraadt.1/Repos/kikuchipy/doc/tutorials/run_nbval.sh` L8–15 hard-codes the notebook list (`hough_indexing.ipynb`, `hybrid_indexing.ipynb`, `mandm2021_sunday_short_course.ipynb`, `pattern_matching.ipynb`, `pc_extrapolate_plane.ipynb`, `pc_fit_plane.ipynb`) and runs
`pytest -v --nbval <notebooks> --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg` (L23).
`c:/Users/westraadt.1/Repos/kikuchipy/doc/tutorials/tutorials_sanitize.cfg` (7 regexes) normalizes: dask “Completed | <time>” → TIME; `N patterns/s`; `N comparisons/s`; tqdm `100% … it/s]` → TQDM_PROGRESSBAR; `Figure size`; `Refining <N>`; `Matching M/N`. **Any new indexing method that prints a speed/progress line must either match one of these regexes or add a new one.**

### 3.4 Registering a tutorial
`c:/Users/westraadt.1/Repos/kikuchipy/doc/tutorials/index.rst` — topic sections each containing an `.. nbgallery::` with `:caption:` and bare notebook stems. The **Indexing** section is L36–48:
```
Indexing
========

.. nbgallery::
    :caption: Indexing

    hough_indexing
    pattern_matching
    hybrid_indexing
    pc_orientation_dependence
    pc_fit_plane
    pc_extrapolate_plane
    pc_calibration_moving_screen_technique
```
Tutorials index is pulled into `doc/user/index.rst` L13–18 (`../tutorials/index.rst`, `../examples/index.rst`).

### 3.5 API reference generation
`c:/Users/westraadt.1/Repos/kikuchipy/doc/reference/index.rst`
- L1 label `.. _api:`; L7–9 `**Release**: |version|` / `**Date**: |today|`; L15–18 caution about breaking changes.
- L24–30: `.. autolink-skip::` + a `pycon` block showing `import kikuchipy as kp`.
- L34–41: `.. rubric:: Functions` + `autosummary :toctree: generated` (`load`, `set_log_level`).
- L42–57: `.. rubric:: Modules` + `autosummary :toctree: generated :template: custom-module-template.rst` listing `data, detectors, draw, filters, imaging, indexing, io, pattern, signals, simulations`.
⇒ **A new public object appears in the API reference solely by being exported in the module’s `__init__.pyi` `__all__`.** Templates: `doc/_templates/custom-{module,class,function,method,attribute}-template.rst` (module template recurses over functions/classes with the custom sub-templates).

### 3.6 Citations
- Bibliography file: `c:/Users/westraadt.1/Repos/kikuchipy/doc/user/bibliography.bib` (32 entries). Key convention: `firstauthorlastnameYYYYfirstsignificantword` (e.g. `chen2015dictionary`, `jackson2019dictionary`, `callahan2013dynamical`, `singh2016orientation`, `singh2017application`, `marquardt2017quantitative`, `pang2020global`, `hjelen1991electron`, `aanes2019electron`). Fields typically `author, doi, journal, pages, title ({{Braced Title}}), volume, year`.
- Rendered by `doc/user/bibliography.rst` L7–8 (`.. bibliography:: bibliography.bib` `:all:`).
- **In docstrings** use `:cite:`key`` — e.g. `ebsd.py` L1840 (`:cite:`chen2015dictionary,jackson2019dictionary``), L2550 (`:cite:`pang2020global``), `_orientation_similarity_map.py` L25, `_normalized_cross_correlation.py` L28, `ebsd_master_pattern.py` get_patterns (`:cite:`callahan2013dynamical``), `data/_data.py` L50/84/133/…, `detectors/_ebsd_detector.py` many.
- **In notebooks**, `:cite:` is *not* used — plain Markdown links to DOIs/papers instead (grep for `cite:` across `doc/tutorials/*.ipynb` returns nothing).
- `doc/user/open_datasets.rst` L13–20 is a pure `:cite:` list.

### 3.7 Changelog
`c:/Users/westraadt.1/Repos/kikuchipy/CHANGELOG.rst`
- L9–14: **Keep a Changelog 1.1.0** format + SemVer; entries in descending chronological order; “Contributors to each release were listed in alphabetical order by first name until version 0.7.0” (⇒ per-release `Contributors` sections exist only up to 0.6.x, e.g. L674–677, L687–690; **no longer used**).
- L17–33 the live `Unreleased` skeleton with section order **Added / Changed / Removed / Fixed / Deprecated** (underlined with `-`, release headers underlined with `=`).
- Entry style (L41–101): imperative/descriptive sentence, one sentence per line where possible, ending with a PR link on its own line:
  ```
  - Two new ``EBSDDetector`` methods to convert between detector pixel and gnomonic
    coordinates.
    (`#793 <https://github.com/pyxem/kikuchipy/pull/793>`_)
  ```
  Note the single-underscore anonymous link form `` `_ `` for PR links; API objects in double backticks. Some entries carry no PR link (L41, L67).
- Release headers: `0.12.0 (2026-05-24)` + `===================` (L36–37).
- No contributor initials are used anywhere in current entries.
- Mirrored into the docs by `doc/changelog.rst` (`:tocdepth: 2` + `.. include:: ../CHANGELOG.rst`).

### 3.8 Credits / contributors
`c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/maintaining_package_credits.rst` L4–12: with consent, every new contributor is added to **two** sources — “`kikuchipy/__init__.py`: List of contributors `__credits__`” and “`.zenodo.json`: Zenodo entry”; “the initial commiter is listed first, with the others sorted by line contributions.”
Reality check: the variable in `src/kikuchipy/__init__.py` L22–37 is **`credits`** (comment L22: “Initial committer first, then sorted by line contributions”), *not* `__credits__`. `.zenodo.json` holds `{"creators": [{"name", "orcid", "affiliation"}, …]}` in the same order (13 entries).
PR template also requires it (`.github/pull_request_template.md` L25).

---

## 4. Data module conventions

`c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/adding_to_data_module.rst`
- L6–12: datasets used in docs/tests are handled with **pooch**, listed in a file registry `kikuchipy.data._registry.py` with **MD5 hash** (`md5sum <file>`) and location — “the latter potentially not within the package but from the `kikuchipy-data <https://github.com/pyxem/kikuchipy-data>`__ repository or elsewhere, **since some files are considered too large to include in the package**”.
- L14–23: files not shipped are downloaded when the user passes `allow_download=True`; cache location `pooch.os_cache("kikuchipy")`, overridable by env var **`KIKUCHIPY_DATA_DIR`**; updating the hash triggers a re-download.
- L25–27: each kikuchipy version gets its own cache subdirectory; old ones are never auto-deleted.

Implementation:
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/_registry.py`
  - L18–20 comment: “All hashes are MD5 hashes and can be checked locally with e.g. md5sum. All file paths are relative to the cache directory `kikuchipy/<version>/data/`”.
  - L22 `# fmt: off` … end `# fmt: on` (aligned dict literals, exempt from black).
  - L23–72 `_registry_hashes` grouped by comment into `# In package (relative to the kikuchipy/data directory)` (3 files), `# From GitHub`, `# From Zenodo`.
  - L75 `KP_DATA_REPO_URL = "https://raw.githubusercontent.com/pyxem/kikuchipy-data/"`; L73–74 comment mandating **permalinks (commit SHA)**, e.g. `KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/nickel_ebsd_large/patterns_v2.h5"` (L78).
  - Zenodo URLs for large master patterns and gain series (L83–…).
  - Tail: both dicts are re-keyed with a `"data/"` prefix into `registry_hashes` / `registry_urls`.
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/_data.py`
  - L32–41 `marshall = pooch.create(path=pooch.os_cache("kikuchipy"), base_url="", version=__version__.replace(".dev", "+"), version_dev="develop", env="KIKUCHIPY_DATA_DIR", registry=registry_hashes, urls=registry_urls, retry_if_failed=5)`.
  - Public loaders (all exported in `data/__init__.pyi`): `nickel_ebsd_small` (L47), `nickel_ebsd_large` (L79), `ni_gain` (L125), `ni_gain_calibration` (L191), `si_ebsd_moving_screen` (L259), `si_wafer` (L326), `nickel_ebsd_master_pattern_small` (L385), `ebsd_master_pattern` (L447).
  - Loader signature convention: `def f(..., allow_download: bool = False, show_progressbar: bool | None = None, **kwargs) -> EBSD:` with docstring sections Parameters / Returns (`ebsd_signal`) / **Notes** (hosting repo + license, e.g. “The dataset carries a CC BY 4.0 license.”) / **Examples** (doctest showing the repr). Body: `dset = Dataset("<relpath>"); file_path = dset.fetch_file_path(allow_download, show_progressbar); return load(file_path, **kwargs)`.
  - `class Dataset` (L529–677): properties `is_in_package`, `is_in_cache`, `has_correct_hash`, `url`, `md5_hash`; `fetch_file_path_from_collection` (zip collections via `pooch.Unzip`); `fetch_file_path` raises `AttributeError` on hash mismatch inside the package and `ValueError` telling the user to pass `allow_download=True` otherwise.
- Package-shipped data dirs: `src/kikuchipy/data/{kikuchipy_h5ebsd, emsoft_ebsd, emsoft_ebsd_master_pattern, edax_binary, edax_h5ebsd, nordif, oxford_binary, _dummy_files}`. Dummy-file creation scripts inside these dirs are `--ignore`d by pytest (pyproject L165–168).
- Tests download ~15 MB (`doc/dev/running_writing_tests.rst` L15–21) ⇒ **tests require an internet connection**; `tests/test_data/test_data.py` L241 uses `@pytest.mark.weekly` for heavy downloads.

---

## 5. Public indexing API pattern (to mirror for `EBSD.spherical_indexing` / `kikuchipy.indexing.SphericalIndexer`)

### 5.1 Exact signatures (`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/ebsd.py`)

**`hough_indexing`** (L1600–1719):
```python
def hough_indexing(
    self,
    phase_list: PhaseList,
    indexer: "EBSDIndexer",
    chunksize: int = 528,
    verbose: int = 1,
    return_index_data: bool = False,
    return_band_data: bool = False,
) -> (CrystalMap | tuple[CrystalMap, np.ndarray] | tuple[CrystalMap, np.ndarray, np.ndarray]):
```
Body order (L1673–1719): `verify_dependency_or_raise("pyebsdindex", "Hough indexing")` → lazy+no-PyOpenCL guard → derive `am`, `nav_shape = am.navigation_shape[::-1]`, `nav_size`, `sig_shape = am.signal_shape[::-1]`, `step_sizes = tuple([a.scale for a in am.navigation_axes[::-1]])` → validate with `_indexer_is_compatible_with_kikuchipy(..., raise_if_not=True)` and `_phase_lists_are_compatible(..., raise_if_not=True)` → `chunksize = min(chunksize, max(am.navigation_size, 1))`; `patterns = self.data.reshape((-1,) + sig_shape)`; if `self._lazy`: `patterns.rechunk({0: chunksize, 1: -1, 2: -1})` → call private `_hough_indexing(...)` → `xmap.scan_unit = _get_navigation_axes_unit(am)` → return tuple variants.

**`hough_indexing_optimize_pc`** (L1721–1825):
```python
def hough_indexing_optimize_pc(
    self,
    pc0: np.ndarray | list | tuple,
    indexer: "EBSDIndexer",
    batch: bool = False,
    method: str = "Nelder-Mead",
    **kwargs,
) -> "EBSDDetector":
```
Validates `pc0.size == 3` (L1779–1781) and `method.lower() in ["nelder-mead", "pso"]` (L1783–1789); returns a **new** `EBSDDetector(shape=sig_shape, pc=pc, sample_tilt=indexer.sampleTilt, tilt=indexer.camElev)` (L1818–1823).

**`dictionary_indexing`** (L1827–1984):
```python
def dictionary_indexing(
    self,
    dictionary: EBSD,
    metric: SimilarityMetric | str = "ncc",
    keep_n: int = 20,
    n_per_iteration: int | None = None,
    navigation_mask: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
    rechunk: bool = False,
    dtype: str | np.dtype | type | None = None,
) -> CrystalMap:
```
Body (L1921–1984): default `n_per_iteration` from `dictionary.data.chunksize[0]` if lazy else `dict_size` → validate `navigation_mask` (shape == nav shape, not all-True, must be `np.ndarray`) and `signal_mask` (`np.ndarray`) → signal shapes must match → dictionary must have 1D `xmap` of size `dict_size` → `metric = self._prepare_metric(...)` → `with dask.config.set(**{"array.slicing.split_large_chunks": False}):` call `_dictionary_indexing(...)` → `xmap.scan_unit = _get_navigation_axes_unit(am_exp)`.
Returned `CrystalMap`: `keep_n` rotations/point, `prop = {"scores", "simulation_indices"}` (docstring L1898–1905; construction `_dictionary_indexing.py` L141–167).
`See Also` block (L1907–1919) lists sibling methods and `kikuchipy.indexing.{merge_crystal_maps, orientation_similarity_map, SimilarityMetric, …}` with one-line descriptions.

**`refine_orientation`** (L1986–2185):
```python
def refine_orientation(
    self,
    xmap: CrystalMap,
    detector: EBSDDetector,
    master_pattern: "EBSDMasterPattern",
    energy: int | float,
    navigation_mask: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
    pseudo_symmetry_ops: Rotation | None = None,
    method: str | None = "minimize",
    method_kwargs: dict | None = None,
    trust_region: tuple | list | np.ndarray | None = None,
    initial_step: float | None = None,
    rtol: float = 1e-4,
    maxeval: int | None = None,
    compute: bool = True,
    rechunk: bool = True,
    chunk_kwargs: dict | None = None,
) -> CrystalMap | da.Array:
```
Body (L2155–2185): `points_to_refine = self._check_refinement_parameters(...)` → `patterns, signal_mask = self._prepare_patterns_for_refinement(...)` → `return _refine_orientation(...)`.

**`refine_projection_center`** (L2187–2374): same parameter list minus `pseudo_symmetry_ops`; returns `tuple[np.ndarray, EBSDDetector, np.ndarray] | da.Array`.
**`refine_orientation_projection_center`** (L2376–2592): adds `pseudo_symmetry_ops`, `initial_step: tuple | list | np.ndarray | None`, `rtol: float | None = 1e-4`; returns `tuple[CrystalMap, EBSDDetector] | da.Array`.

Docstring template used by all three refinement methods: r-string summary → paragraph on what is optimized/fixed → bulleted list of supported local/global optimizers with `:func:` roles → `Parameters` → `Returns` (`out`) → `See Also` → `Notes` (optional-dependency caveat, `:ref:`dependencies``, physics caveats with `:cite:`).

### 5.2 Private helper names

`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/_dictionary_indexing.py`
- `_dictionary_indexing(experimental, experimental_nav_shape, dictionary, step_sizes, dictionary_xmap, metric, keep_n, n_per_iteration) -> CrystalMap` (L36–169)
- `_match_chunk(experimental, simulated, keep_n, metric) -> tuple[da.Array, da.Array]` (L172–203)
- `_dictionary_indexing_info_message(metric, n_experimental_all, dictionary_size, phase_name, n_experimental=None) -> str` (L206–237)

Notable mechanics: chunked iteration with `tqdm(zip(chunk_starts, chunk_ends), total=n_iterations)` (L105); single-shot path wrapped in `with ProgressBar():` + `da.compute` (L92–93); running top-k merge via `np.argsort(negative_sign * all_scores, axis=1)[:, :keep_n]` + `np.take_along_axis` (L120–128); timing → `print(f"  Indexing speed: {patterns_per_second:.5f} patterns/s, {comparisons_per_second:.5f} comparisons/s")` after `sleep(0.2)` to avoid tqdm bleed-through (L130–139); crystal map built from `create_coordinate_arrays(experimental_nav_shape, step_sizes)` with `is_in_data` when a navigation mask is used (L141–167); `phase_list=dictionary_xmap.phases_in_data`.
Info message format (L228–236): `"Dictionary indexing information:\n  Phase name: …\n  Matching N experimental pattern(s) to M dictionary pattern(s)\n  <metric repr>"`.

`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/_hough_indexing.py`
- public `xmap_from_hough_indexing_data(data, phase_list, data_index=-1, navigation_shape=None, step_sizes=None, scan_unit="px") -> CrystalMap` (L43–118)
- `_get_indexer_from_detector(phase_list, shape, pc, sample_tilt, tilt, reflectors=None, **kwargs) -> "EBSDIndexer"` (L121–184)
- `_hough_indexing(patterns, phase_list, nav_shape, step_sizes, indexer, chunksize, verbose) -> tuple[CrystalMap, np.ndarray, np.ndarray]` (L187–251)
- `_get_pyebsdindex_phaselist(phase_list, reflectors=None) -> list["BandIndexer"]` (L254–336)
- `_indexer_is_compatible_with_kikuchipy(indexer, sig_shape, nav_size=None, check_pc=True, raise_if_not=False) -> bool` (L339–406)
- `_phase_lists_are_compatible(phase_list, indexer, raise_if_not=False) -> bool` (L409–490)
- `_get_info_message(nav_size, chunksize, indexer) -> str` (L493–510) → `"Hough indexing with PyEBSDIndex information:\n  PyOpenCL: …\n  Projection center (Bruker[, mean]): (x, y, z)\n  Indexing N pattern(s) in M chunk(s)"`
- `_optimize_pc(pc0, patterns, indexer, batch, method, **kwargs) -> np.ndarray` (L513–526)

`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/_refinement/_refinement.py` (1320 L)
- Public compute helpers: `compute_refine_orientation_results` (L58), `compute_refine_projection_center_results` (L133), `compute_refine_orientation_projection_center_results` (L199).
- `_get_crystal_map_parameters` (L302), `_refine_orientation` (L340), `_refine_orientation_chunk_scipy` (L441) / `_chunk_nlopt` (L504), `_refine_pc` (L577) + chunks (L642, L670), `_refine_orientation_pc` (L705) + chunks (L779, L811), `class _RefinementSetup` (~L840–1287), `_get_master_pattern_data` (L1288).
- `_RefinementSetup` is the reusable “setup object” pattern: class-level annotations (L900–919: `mode`, `data_shape`, `nav_size`, `rotations_array`, `chunk_func`, `map_blocks_kwargs`, `solver_kwargs`, …), `set_optimization_parameters` (L1053), `set_fixed_parameters` (L1141), `get_bound_constraints` (L1178), `get_info_message` (L1244), plus `n_control_variables` / `bounds_chunks` properties (L1040–1051).
- Dask execution: `da.map_blocks(ref.chunk_func, patterns, ref.rotations_array, lower_bounds, upper_bounds, ..., **ref.map_blocks_kwargs)` (L391–407, L413–424, L617, L752) with `map_blocks_kwargs = {"drop_axis": …, "new_axis": (1,), "dtype": np.float64}` (L1016–1018) and chunks `(patterns.chunksize[0], -1, -1)` (L960).
- Compute path: `print(f"Refining {n} orientation(s):", file=sys.stdout)` → `with ProgressBar(): res = results.compute()` → `print(f"Refinement speed: {…:.5f} patterns/s")` (L111–117).

### 5.3 Signal-side private helpers to reuse (`ebsd.py`)
- `_check_refinement_parameters(xmap, detector, master_pattern, navigation_mask=None, signal_mask=None) -> np.ndarray` (L2880–2970) — “No checks of the parameters should be necessary after this function runs successfully.” Calls `_detector_is_compatible_with_signal`, `_xmap_is_compatible_with_signal`, `_get_indexed_points_in_data_in_xmap`, `master_pattern._is_suitable_for_projection(raise_if_not=True)`, `_equal_phase`.
- `_prepare_patterns_for_refinement(points_to_refine, signal_mask, rechunk, chunk_kwargs=None) -> tuple[da.Array, np.ndarray]` (L2972–3047) — `get_dask_array(signal=self)` → `da.atleast_3d` → reshape `(nav_size, sig_size)` → boolean masking → `get_chunking(data_shape=..., nav_dim=1, sig_dim=1, dtype="float32", **chunk_kwargs)` with fallback `chunk_shape = 64` → `patterns[:, np.newaxis, :]`.
- `_prepare_metric(metric, navigation_mask, signal_mask, dtype, rechunk, n_dictionary_patterns) -> SimilarityMetric` (L3049–3088) — `metrics = {"ncc": NormalizedCrossCorrelationMetric, "ndp": NormalizedDotProductMetric}`; string→class lookup; `ValueError` naming the allowed keys; sets `n_experimental_patterns`, `n_dictionary_patterns`, masks, dtype; `metric.raise_error_if_invalid()`.
- `_get_sum_signal(signal, out_signal_axes=None) -> hs.signals.Signal2D` (`@staticmethod`, L3090–3105) — `nansum` over signal axes, `set_signal_type("")`, `transpose(out_signal_axes)`.
- `_get_navigation_axes_unit(axes_manager) -> str` (module-level, L3380–3387) — returns `"px"` unless HyperSpy units are defined.
- Detector/PC validation entry point for users: `EBSDDetector.get_indexer(phase_list, reflectors=None, **kwargs)` (`src/kikuchipy/detectors/_ebsd_detector.py` L1607–1667) — warns via `_warn_if_angles_ignored` then delegates to `_get_indexer_from_detector` with `pc=self.pc_flattened.squeeze()`.

### 5.4 SimilarityMetric extension point (template for a new metric/indexer class)
`src/kikuchipy/indexing/similarity_metrics/_similarity_metric.py` L23–253: `class SimilarityMetric(abc.ABC)` with class attributes `_allowed_dtypes: list[type] = []`, `_sign: int | None = None`; `__init__(n_experimental_patterns=None, n_dictionary_patterns=None, navigation_mask=None, signal_mask=None, dtype="float32", rechunk=False)`; `__repr__` producing `"<Class>: float32, greater is better, rechunk: False, navigation mask: False, signal mask: False"` (L88–95, asserted verbatim in `tests/test_indexing/test_similarity_metrics.py` L36–39); properties with docstring `Parameters/value` blocks; abstract `prepare_dictionary`, `prepare_experimental`, `match` (L223–242, each `return NotImplemented  # pragma: no cover`); `raise_error_if_invalid()` (L244–253).
Concrete example `NormalizedCrossCorrelationMetric` (`_normalized_cross_correlation.py` L26–241): `_allowed_dtypes = [np.float32, np.float64]`, `_sign = 1`, math in the class docstring via `.. math::` + `:cite:`gonzalez2017digital``, `__call__`, numbered preparation steps in the method docstrings, `da.einsum("ik,mk->im", ..., optimize=True, dtype=self.dtype)` for matching, module-level numba kernel at L200. `NormalizedDotProductMetric` mirrors it (`_normalized_dot_product.py` L25/46/47).

### 5.5 Master-pattern side (dictionary generation)
`src/kikuchipy/signals/ebsd_master_pattern.py` L97–…: `get_patterns(rotations: Rotation, detector: EBSDDetector, energy: int | float | None = None, dtype_out: str | np.dtype | type = "float32", compute: bool = False, show_progressbar: bool | None = None, **kwargs) -> EBSD | LazyEBSD`, `:cite:`callahan2013dynamical``; `_is_suitable_for_projection(raise_if_not=False)` at L331. Base private class `KikuchiMasterPattern` in `src/kikuchipy/signals/_kikuchi_master_pattern.py` L34 with `_custom_attributes = ["hemisphere", "phase", "projection"]` (L57).
`EBSD` class contract: `_signal_type = "EBSD"`, `_alias_signal_types`, `_custom_attributes = ["detector", "static_background", "xmap"]` (ebsd.py L184–186); custom attributes are properties with validating setters (L203–263); `class LazyEBSD(LazyKikuchipySignal2D, EBSD)` at L3186.

---

## 6. Git / branch / PR conventions

`c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/using_git.rst`
- L7–8: **new feature → branch off `develop`; bug fix → branch off `main`**.
- L10–12: `git switch -c your-awesome-feature-name upstream/develop`.
- L18–26: `git commit -s -m "An explanatory commit message"` — **`-s` (sign-off) is required**; confirmed in history (`Signed-off-by: Håkon Wiik Ånes <hwaanes@gmail.com>` on every non-merge commit).
- L28–45: keep up to date by **merging** `develop` (or `main`) into the branch (not rebasing).
- L47–57: `git push -u origin <branch>`; PR to `develop` for features, `main` for bug fixes.
- Branch naming observed: descriptive kebab-case (`further-detector-changes`, `prepare-0.12.0-release`, `main-into-develop-after-0.12.0`).
- Commit subject style: imperative/descriptive sentence case, no prefix/scope, no trailing period (e.g. “Catch test warning about unused non-zero detector angles”, “Set version to 0.13.dev0”).

`c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/setting_up_development_installation.rst` L6–35: fork → clone → `git remote add upstream https://github.com/pyxem/kikuchipy.git` → conda env → `pip install --editable ".[dev]"`.

`c:/Users/westraadt.1/Repos/kikuchipy/.github/pull_request_template.md` (verbatim checklist):
```
#### Description of the change
<!-- Remember to branch off the develop branch for new features and the main branch for patches. -->

#### Progress of the PR
- [ ] Docstrings for all functions (numpydoc link)
- [ ] Unit tests with pytest for all lines
- [ ] Clean code style by running black via pre-commit

#### Minimal example of the bug fix or new feature
```python
>>> import kikuchipy as kp
>>> s = kp.data.nickel_ebsd_small()
>>> s
<EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
>>> # Your new feature...
```

#### For reviewers
- [ ] The PR title is short, concise, and will make sense 1 year later.
- [ ] New functions are imported in corresponding `__init__.py`.
- [ ] New features, API changes, and deprecations are mentioned in the unreleased section in `CHANGELOG.rst`.
- [ ] New contributors are added to `kikuchipy/__init__.py` and `.zenodo.json`.
```
Community entry points (`doc/dev/index.rst` L25–30): GitHub Discussions, Issues, PRs, Gitter. Code of Conduct is binding (L22–23, `CODE_OF_CONDUCT.rst`, enforcement manual `doc/dev/report_handling_manual.rst`, `:orphan:`).
Maintainer/reviewer: `pyproject.toml` L10 `maintainers = [{name = "Håkon Wiik Ånes", email = "hwaanes@gmail.com"}]`; all merges in recent history are by him.

Release process — `c:/Users/westraadt.1/Repos/kikuchipy/RELEASE.rst`: Gitflow-like (L5–6); minor release branch off `develop`, patch off `main` (L17–23); run all notebooks + `make linkcheck` (L25–28); verify `__credits__`/`.zenodo.json` (L30–32); bump `__version__` in `kikuchipy/__init__.py` and tidy `CHANGELOG.rst` per Keep a Changelog (L34–35); PR release branch → `main` (L37–40); a push to `main` touching `src/kikuchipy/__init__.py` triggers `.github/workflows/perhaps_make_tagged_release_draft.yml` which auto-drafts release `kikuchipy X.Y.Z` / tag `vX.Y.Z` with the body extracted from the changelog by `perhaps_make_tagged_release_draft.py` and converted rst→md via pandoc; `publish.yml` publishes to TestPyPI then PyPI on release publish (trusted publishing, `id-token: write`); post-release: RTD build, Zenodo, Binder, merge `main` back into `develop`, close milestone, conda-forge feedstock PR (L75–102).
Dependabot: monthly GitHub Actions updates targeting `develop` (`.github/dependabot.yml` L27–35).

---

## 7. Optional dependency handling pattern

Central registry — `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/_constants.py`:
- L27–41 `deps_for_version_check` list (required: hyperspy, matplotlib, numpy, rosettasciio, scikit-image; optional: IPython, ipywidgets, nlopt, psygnal, pyvista, pyebsdindex).
- L42–48 `dependency_version: dict[str, Version | None]` built with `importlib.metadata.version` inside `try/except ImportError` (`# pragma: no cover`).
- L51–56:
  ```python
  def verify_dependency_or_raise(package: str, reason: str) -> None:
      """Raise an informative ImportError if a *package* required for some *reason* is not installed."""
      if dependency_version[package] is None:
          raise ImportError(f"{reason} requires that {package!r} is installed")
  ```
- L59–76: PyOpenCL context probe → module flag `pyopencl_context_available` (bare `except Exception`, `# pragma: no cover`), used to gate lazy Hough indexing.

Usage pattern (three layers):
1. **Type-checking-only import**, guarded twice: `if TYPE_CHECKING:  # pragma: no cover` + `if dependency_version["pyebsdindex"] is not None:` then `from pyebsdindex.ebsd_index import EBSDIndexer` (`_hough_indexing.py` L37–40); annotations use the **string form** `"EBSDIndexer"`. Same for nlopt in `_solvers.py` L43–47 and `_refinement.py` L50–55; `ebsd.py` L107–113 wraps it in `try/except ImportError: pass`.
2. **Runtime guard then deferred import inside the function**: `verify_dependency_or_raise("pyebsdindex", "Getting an indexer")` then `from pyebsdindex.ebsd_index import EBSDIndexer` (`_hough_indexing.py` L168–170); `from pyebsdindex.tripletvote import addphase` (L292); `from pyebsdindex.pcopt import optimize_pso as optimize_func` (L521–524); `import nlopt` inside the chunk functions (`_refinement.py` L529, L681, L824, L1099). Public method entry points call `verify_dependency_or_raise` first (`ebsd.py` L1673, L1767–1769; also used for PyVista in `_kikuchi_master_pattern.py`).
3. **Docstring Notes** stating the requirement + `:ref:`dependencies`` link (`ebsd.py` L1663–1666, L1762–1765, L2147–2153; `_hough_indexing.py` L164–166).

Packaging (`pyproject.toml` L67–115): extras `all` (ipywidgets, IPython, nlopt, psygnal, `pyebsdindex >= 0.3.9.2`, pyvista), `doc`, `tests`, `coverage`, `dev` (which includes `kikuchipy[doc,tests,coverage]`).
Docs (`doc/user/installation.rst` L127–143): bulleted “Some functionality requires optional dependencies” list with a one-line purpose and recommended install channel per package; note that `[all]` does not install pyopencl.
Version-conditional imports for *required* deps use `packaging.version.Version` comparisons in a small shim module — `src/kikuchipy/_utils/hyperspy_utils.py` L26–33 and `src/kikuchipy/_utils/rosettasciio_utils.py` L32–45 (both re-export via `__all__`).

**⇒ To add e.g. `pyshtools`:** add to `deps_for_version_check` in `_constants.py`; add to `[project.optional-dependencies] all` (and `doc` if a tutorial needs it) in pyproject; guard all imports as above with `verify_dependency_or_raise("pyshtools", "Spherical indexing")`; add an intersphinx entry in `doc/conf.py` if you want `:doc:` links; document it in `doc/user/installation.rst`; add `skipif` tests both ways; extend the CI “Install optional dependencies” step if it is not covered by `[all]`; mention it in `CHANGELOG.rst` under `Added`.

---

## 8. Numba conventions

Baseline decorator (overwhelming majority):
```python
@njit(cache=True, nogil=True, fastmath=True)
```
- `src/kikuchipy/indexing/_refinement/_solvers.py` L50 (`_prepare_pattern`) — imported as `from numba import njit` (L27).
- `src/kikuchipy/pattern/_pattern.py` L96, 198, 348, 392, 415, 484, 498, 763, 794 (`cache=True, fastmath=True, nogil=True` — keyword order varies, all three present).
- `src/kikuchipy/signals/util/_master_pattern.py` L311, 386, 461, 694.

**Explicit eager signatures** are used where the types are stable (gives compile-time checking and avoids dispatch overhead):
- `src/kikuchipy/indexing/similarity_metrics/_normalized_cross_correlation.py` L200:
  `@njit("float64(float32[:], float32[:], float32)", cache=True, nogil=True, fastmath=True)`
- `src/kikuchipy/pattern/_pattern.py` L114 `"float32[:, :](float32[:, :])"`, L125 `"float32[:](float32[:])"`, L136 `"Tuple((float32[:], float32))(float32[:])"`, L776 `"float32[:, :](float32[:, :], int64)"`.
- `src/kikuchipy/signals/util/_master_pattern.py` L127–132 `("float64[:, :](float64[:], float64, int64, int64, float64[:, ::1], bool_[:])")` (note the **C-contiguous `::1`** layout marker), L215–221, L584–590 (multi-line signature strings), L724–727.
- `src/kikuchipy/_utils/numba.py` L26, L43, L59–61, L87: `@nb.njit("float64[:](float64, float64, float64)", cache=True, fastmath=True, nogil=True)` — this module imports `import numba as nb` and holds cross-module kernels (`rotation_from_rodrigues`, `rotation_from_euler`, `rotate_vector`, `vec_dot`), each with a `# ---- Section ---- #` banner comment.

**Parallel** (`parallel=True` + `nb.prange`) is rare and only where the loop is embarrassingly parallel:
- `src/kikuchipy/simulations/kikuchi_pattern_simulator.py` L680–693: `@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:, :], nb.float64[:, :], nb.float64[:]), cache=True, parallel=True, fastmath=True, nogil=True)` with `for i in nb.prange(m):` — note the typed-object signature form (`nb.float64[:]`) rather than a string, and the section banner `# ------------------- Numba-accelerated functions -------------------- #`.
- `src/kikuchipy/signals/util/_master_pattern.py` L200, 281, 294, 566 use `nb.prange` **without** `parallel=True` (prange then degrades to range) — existing inconsistency; prefer explicit `parallel=True` only when benchmarked.

Other conventions:
- Numba kernels get full numpydoc docstrings including the exact dtype/shape contract of each argument (`_normalized_cross_correlation.py` L204–221: “1D array of shape (n_pixels,) and data type 32-bit floats already centered”).
- Numba functions are **module-private** (`_`-prefixed) except the cross-module helpers in `_utils/numba.py`.
- Objective functions passed to SciPy/NLopt are **plain Python** functions taking `(x, *args)` with the `*args` contents enumerated by index in the docstring (`src/kikuchipy/indexing/_refinement/_objective_functions.py` L36–74, L77–132, L135–190); they call numba kernels internally.
- Compilation artifacts (`*.nbi`, `*.nbc`) appear in `__pycache__` from `cache=True`.
- Recent perf guidance in history: commit `b480e6a1` “Follow Numba performance warning suggestion and make arrays contiguous”.
- Performance philosophy — `doc/dev/improving_performance.rst` L4–8: “(1) get the correct result, (2) don't fill up memory, (3) … doesn't take too long. To keep memory in check, we should use Dask wherever possible. To speed up computations, we should use Numba wherever possible.”

---

## 9. Concrete checklist for adding `EBSD.spherical_indexing` / `kikuchipy.indexing.SphericalIndexer`

| Artefact | Path | Convention source |
|---|---|---|
| Private implementation | `src/kikuchipy/indexing/_spherical_indexing.py` (module docstring “Private tools for …”; new-style license header) | mirrors `_dictionary_indexing.py` L1–22 |
| Public names | add imports + sorted `__all__` entries to `src/kikuchipy/indexing/__init__.pyi` | L18–42 |
| Signal method | `src/kikuchipy/signals/ebsd.py`, placed with the other indexing methods (after L1984 / before `refine_orientation` at L1986); import the private function in the kikuchipy import block (L50–61) | ebsd.py L1600–2592 |
| Validation | reuse `_detector_is_compatible_with_signal`, `_xmap_is_compatible_with_signal`, add `_*_is_compatible_with_kikuchipy(..., raise_if_not=False) -> bool` helpers | `_hough_indexing.py` L339, L409 |
| Returned map | `CrystalMap` from `create_coordinate_arrays(nav_shape, step_sizes)`; `prop` dict with `scores` (+ e.g. `simulation_indices`); `xmap.scan_unit = _get_navigation_axes_unit(am)` | `_dictionary_indexing.py` L141–167; ebsd.py L1982 |
| Progress/timing | `print(info_message)` before, `tqdm`/`with ProgressBar():` during, `print(f"  Indexing speed: {x:.5f} patterns/s")` after | `_dictionary_indexing.py` L77–139 |
| Dask | `self.data.reshape((-1,) + sig_shape)`, `rechunk({0: chunksize, 1: -1, 2: -1})`, `get_dask_array`/`get_chunking` from `kikuchipy.signals.util._dask`, `da.map_blocks(..., **map_blocks_kwargs)` | ebsd.py L1694–1698, L3009–3042; `_refinement.py` L391–424 |
| Unit tests | `tests/test_indexing/test_spherical_indexing.py` (+ `tests/test_signals/test_ebsd_spherical_indexing.py` if the method is signal-heavy, following the Hough precedent); `class TestSphericalIndexing:` using `dummy_signal`, `get_single_phase_xmap`, `detector`, `ebsd_with_axes_and_random_data`; `.py_func` tests for any numba kernel; `skipif(dependency_version["<dep>"] is None)` for optional libs | §2 |
| Benchmark | `benchmarks/indexing/test_spherical_indexing.py` with `def test_spherical_indexing(benchmark):` | `benchmarks/indexing/test_dictionary_indexing.py` |
| Tutorial | `doc/tutorials/spherical_indexing.ipynb` (hidden first cell, thumbnail tag, black@77, outputs stripped unless heavy) + entry in the **Indexing** `nbgallery` of `doc/tutorials/index.rst` (L36–48) + optionally `doc/tutorials/run_nbval.sh` list if outputs are stored (then check `tutorials_sanitize.cfg` covers the printed timing lines) | §3 |
| Bibliography | add EMSphInx/SHT refs to `doc/user/bibliography.bib` (`singh2016orientation`, `singh2017application` already present) and cite with `:cite:` in the docstring | §3.6 |
| Changelog | `CHANGELOG.rst` → `Unreleased` → `Added`, entry + `` (`#NNN <https://github.com/pyxem/kikuchipy/pull/NNN>`_) `` | §3.7 |
| Optional dep | `pyproject.toml` extras + `_constants.deps_for_version_check` + `verify_dependency_or_raise` + `doc/user/installation.rst` + `doc/conf.py` intersphinx | §7 |
| Credits | `src/kikuchipy/__init__.py` `credits` + `.zenodo.json` if a new contributor | §3.8 |
| Branch/PR | branch off `upstream/develop`, `git commit -s`, PR into `develop`, fill the reviewer checklist | §6 |

### Discrepancies worth knowing (docs vs. code)
1. `doc/dev/handling_deprecations.rst` L13/L26 names the decorators `deprecate` / `deprecate_argument`; the actual classes in `src/kikuchipy/_utils/deprecated.py` are `deprecated` / `deprecated_argument`.
2. `doc/dev/maintaining_package_credits.rst` L8 says `__credits__`; the actual name in `src/kikuchipy/__init__.py` L23 is `credits`. `RELEASE.rst` L31 also says `__credits__`.
3. `doc/dev/running_writing_tests.rst` L4–5 says “The tests reside in a `tests/` directory within each module”; in reality all tests live in the single top-level `tests/` package.
4. `doc/dev/building_writing_documentation.rst` L117 refers to `setup.py`; the project uses `pyproject.toml` + hatchling.
