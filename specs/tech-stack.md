# Tech stack and conventions (constitution)

Everything below is binding for every feature branch of the spherical indexing
project. Where kikuchipy's own developer guide (`doc/dev/*.rst`) says more, it
also applies.

## Repositories, branches, git

- Work happens on the fork `jwestraadt/kikuchipy`; upstream `pyxem/kikuchipy` is the `upstream` remote. `develop` on the fork tracks `upstream/develop` (synced 2026-08-16 to 0.14.dev0).
- One feature = one branch off `develop`, named as in `specs/roadmap.md`; PR into the fork's `develop`. Merge (never rebase) `develop` into a feature branch to update it. Upstream PR series only after the maintainers have answered the licence/patent issue.
- Commits are signed off: `git commit -s`. Commit only the feature's own file list; the user's uncommitted notebook edits (`doc/tutorials/hybrid_indexing.ipynb`, `load_save_data.ipynb`) are never swept into a commit.
- `specs/` lives on the fork only (excluded from pre-commit via `.pre-commit-config.yaml` so the `licenseheaders` hook does not stamp the Markdown); strip `specs/` and that exclude line from any upstream PR.

## Runtime dependencies

- Anything in `[project.dependencies]` of `pyproject.toml` (0.14.dev0: numpy, scipy >= 1.7, numba >= 0.57, dask[array], orix >= 0.12.1, h5py, hyperspy, matplotlib, tqdm, diffpy.structure, diffsims, scikit-image, scikit-learn, rosettasciio, imageio, pyyaml, lazy_loader, packaging, typing-extensions). `pooch` is **optional** (`tests`/`doc`/`all` extras): never import it at module scope in `src/`; download-backed real-data tests require the `tests` extra and skip cleanly without it (as `kikuchipy.data` already does).
- **No** FFTW/pyfftw, shtns, pyshtools, rocket-fft, sympy. Test oracles that need newer libraries (`scipy.special.sph_harm_y` >= 1.15) are version-gated with `pytest.importorskip`/`packaging.version` so the CI "oldest" job (py3.10, numba 0.57, numpy 1.23, orix 0.12.1) stays green; do not use `Orientation.reduce()` in tests (use `angle_with`). The oldest matrix is verified on CI only, so every gate must be an `importorskip`/version check, never an untested branch; local recipe if needed: `uv run --isolated --python 3.10 --with "numpy==1.23.0" --with "numba==0.57" --with "orix==0.12.1" pytest tests/test_indexing -k spherical`.

## Numerics

- float64/complex128 throughout (EMSphInx `Real = double`); a float32 fast path is a later, measured optimisation.
- FFTs: `scipy.fft` with `norm="forward"` on inverse transforms to reproduce FFTW's unnormalised `c2r`. DCT: `scipy.fft.dctn(type=2)` / `dctn(type=3)` == FFTW REDFT10/REDFT01 (unnormalised) -- **not** `idctn(type=3)`.
- `fast_size` is a verbatim port of EMSphInx `fastSize` ({2,3,5,7,11,13}-smooth, `util/fft.hpp:438-491`); never `scipy.fft.next_fast_len(real=True)`. Warn when `slP` has a factor 13.
- Grids: odd side length `dim`; Legendre layout for indexing (`dim = bw+2` if `bw` odd else `bw+3`), Lambert layout for master patterns (`bw = (dim-1)/2`). Ring quadrature weights from the ported `computeWeightsSkip` (Sneeuw system), asserting `|sum(w_hat) - 1| <= cbrt(eps)/64`.
- SH coefficients `alm[m, l]` m-major complex128, `l < m` zero, `a_00 = sqrt(4 pi)` for f == 1; the Condon-Shortley composition is *determined* by a test against `sph_harm_y` and then frozen in the `SphericalHarmonicTransform` docstring.
- SHT dual path: numba DFT-matrix kernel for indexing bandwidths (<= 128), `scipy.fft.rfft` per ring above; both agree to 1e-12.
- Symmetry flags (`fNf`, `fMr`) from all **38** orix point groups keyed on `Symmetry.name`, cross-checked against `get_axis_orders`; `.sht` `zRot`/`cmpFlg` come from the space-group LUTs and are kept separate.
- Euler: grid -> ZYZ `alpha = 2 pi m/slP - pi/2, beta = 2 pi k/slP - pi, gamma = 2 pi n/slP - pi/2`; the single source of truth for ZYZ -> quaternion is `_euler.zyz_to_quaternion` (port of `zyz2qu`); Bunge equivalent `(alpha + pi/2, beta, gamma - pi/2)`; the result orientation is sample->crystal (`~Rotation`), pinned by a forward-projection test through `EBSDMasterPattern.get_patterns` at known rotations *before* any EMSphInx reference is generated.
- Back-projection is a **gather** (iterate sphere normals -> detector pixels); `_get_direction_cosines_from_detector` is only an oracle. v1 supports one PC per call (`ValueError` naming `detector.pc_average` otherwise) and raises `ValueError` naming `detector.azimuthal`/`detector.twist` when either is non-zero (EMSphInx `detector.hpp:337` refuses omega tilt too); `tilt`/`sample_tilt` are supported.
- Bandwidth vs source resolution: `get_spherical_harmonics`/`from_master_pattern` warn when `bandwidth > (source_dim - 1)/2` (a 401-px Lambert master carries information up to bw ~200; bw 384 fixtures exist only for mp2sht parity, never for accuracy claims). Default bandwidth 384 (EMSphInx mp2sht default). **The Lambert-layout SHT itself is numerically usable only for `dim <~ 275`** (the Sneeuw ring-weight system is a Chebyshev-Vandermonde system whose precision guard trips around `dim 277-301`; measured in Phase 1), so master patterns (401/1001 px) are always regridded to a Legendre grid before analysis (Phase 2 `toLegendre`, exactly as EMSphInx does).
- Symmetry LUTs keyed on `Symmetry.name` are validated at import/test time against `orix.quaternion.symmetry._groups` (38 names on orix 0.13/0.14; confirm on the 0.12.1 floor before pinning) with a clear error for unknown names.
- Result contract: failed patterns -> `phase_id = -1`, identity rotation, score 0, `is_indexed == False` (EMSphInx semantics); `navigation_mask` follows kikuchipy polarity (only `False` entries are indexed); the `scores` prop docstring states the metric is un-normalised (not a bounded NCC), comparable only within a fixed geometry, and unusable with `orientation_similarity_map`.
- PatternRepack contract (`write_emsphinx_patterns`): root dataset `Manufacturer` (string accepted by `PatternFile::GetVendor`, verified against `IndexEBSD.exe`), `/patterns` `(n, h, w)` contiguous with early allocation so EMSphInx can memory-map it, optional vertical flip (EMSphInx hard-codes it), binning as in-dtype average (`binAvg`) or float32 sum (`binFloat`); EMsoft raw `.data` input is out of scope.
- EMSphInx quirks that affect results (2x master mean over `totW`, quartered corner weights, `interpPeak` `x[2]` bounds bug, Gaussian-fit off-by-one, metric not norm-divided) and its preprocessing (mosaic AHE `n_regions`, 2-D Gaussian background, DCT image quality, circular mask) are ported **faithfully behind explicit per-call keywords** (e.g. `emsphinx_compatible=True`, `n_regions=10`) -- never a module global (numba freezes globals). Each quirk is listed in one docs table with its default.
- Misorientation tolerances vs kikuchipy's stored `xmap`s are derived from a measured mean-PC error floor (re-run `refine_orientation` with `pc_average` on `nickel_ebsd_small/large`, record the spread in Phase 5/6 `validation.md`) rather than asserted a priori.
- Performance numbers are recorded baselines in `validation.md`, not merge gates, except the hard floor >= 2 patterns/s/core at bw 68 (60x60) which triggers the fallback list (float32, `m % fNf` plane skipping, coarse-only default). Memory per thread is measured (`tracemalloc`) for bw in {63, 68, 88, 113} and a warning helper lives on `SphericalIndexer`.

## Code layout and style

- Private package `src/kikuchipy/indexing/_spherical/` (leading underscore, warning docstring like `indexing/_refinement/__init__.py`); public names only through `src/kikuchipy/indexing/__init__.pyi` (`lazy_loader.attach_stub`, sorted `__all__`). Signal methods go in `src/kikuchipy/signals/ebsd.py` next to `dictionary_indexing`; the `.sht` reader is an io plugin `src/kikuchipy/io/plugins/emsphinx_master_pattern/` (producer_content naming like `ebsdsim_master_pattern/`, which it is modelled on -- but with the GPL header, since it imports the GPL harmonics code).
- ruff + ruff-format via `pre-commit run --files <changed files>` (never `--all-files`: the `licenseheaders` hook would rewrite the repo). numpydoc; type hints in signatures only; comment/docstring lines <= 72 chars; three import blocks; `@njit(cache=True, nogil=True)` (add `fastmath=True` only after tolerances pass); no `parallel=True` -- parallelism is `dask.array.map_blocks` over pattern chunks with the threaded scheduler; every `scipy.fft` call passes `workers=1` (also in the SHT ring transforms) so dask threads never oversubscribe. Numba caches (`cache=True`) are written by one process: run a phase's tests once with `-n 0` before `-n 4`.
- Licence headers: kikuchipy GPL header from `.license.tmpl` + a delimited third-party block enumerating the derived functions, copying `src/kikuchipy/signals/util/_master_pattern.py:20-57` (GPL header, blank line, `# The following copyright notice ...` rationale, then the verbatim third-party notice between `# ####` rules). EMSphInx-derived files carry the CMU/Lenthe notice verbatim (incl. commercial-licence contact), "GPL-2.0-or-later, conveyed under GPL-3.0-or-later", and the modification notice required by GPLv2 s.2(a) / GPLv3 s.5(a) ("changed by <name>, <date>"). `_sht_file.py` carries the SHTfile BSD-3 notice verbatim (repo `https://github.com/EMsoft-org/SHTfile` @ `e49ad6b`).

## Tests, docs, data

- pytest; `tests/test_indexing/test_spherical_*.py`, `tests/test_signals/test_ebsd_spherical_indexing.py`, `tests/test_io/test_emsphinx_master_pattern.py`; every numba kernel also tested via `.py_func`; heavy tests `@pytest.mark.weekly`; benchmarks in `benchmarks/indexing/test_spherical_indexing.py` (pytest-benchmark).
- Real data: `kp.data.nickel_ebsd_small()` + `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")` (shipped) in the default suite; `nickel_ebsd_large(allow_download=True)` subset in the default suite, full map weekly; `ebsd_master_pattern("ni")` (cached locally) weekly. EMSphInx references are generated once with the built binaries (`nthread=1, batchsize=1`) and shipped in-package under `src/kikuchipy/data/emsphinx/` (each < 100 kB, md5 in `_registry.py`, no URL); large/local-only inputs stay behind env vars with `pytest.skip`.
- Assertions on scores are measured-then-pinned (`pytest.approx(measured, rel=0.05)`) or scale-free (ordering); misorientations via `Orientation.angle_with(..., degrees=True)`.
- Docs: the numpydoc reference is generated from `__all__`; `:cite:` keys are added to `doc/user/bibliography.bib`; tutorial notebook rules: hidden first cell, thumbnail tag, black at 77, registered in `doc/tutorials/index.rst`; because indexing cells are expensive, outputs are stored (as `pattern_matching.ipynb` does), the notebook is added to the `NOTEBOOKS` array in `doc/tutorials/run_nbval.sh` and non-deterministic output (timings) is covered by `doc/tutorials/tutorials_sanitize.cfg` (`doc/dev/building_writing_documentation.rst`).
- CHANGELOG: `Unreleased -> Added/Changed` entries, one per feature; on the fork link the fork PR (`(`#N <https://github.com/jwestraadt/kikuchipy/pull/N>`_)`) and rewrite to the `pyxem/kikuchipy` PR number when the upstream series is opened. Upstream-submission checklist: strip `specs/`, drop the `specs/` pre-commit exclude, rewrite CHANGELOG links.

## Commands (venv is uv-managed; no pip)

Run from Git Bash (PowerShell does not expand globs for native commands); every command is shell-independent as written.

```
uv run pytest tests/test_indexing tests/test_signals tests/test_io -k spherical -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py
uv run pytest --weekly -k spherical
uv run pre-commit run --files <explicit list of changed files, never specs/>
uv run sphinx-build -b html doc doc/_build/html          # `make html` in doc/ on Linux/CI
uv run sphinx-build -b linkcheck doc doc/_build/linkcheck
```

## Process per feature (spec-driven)

Gates are stated in artefacts; the model/effort notes in parentheses are the project's working practice (agreed with the user), not something a PR reviewer verifies.

1. `plan` -- `specs/<date>-<name>/plan.md` (task groups) drafted (Fable 5, effort xhigh, ultracode) -- **the user approves it**.
2. `spec` -- `requirements.md` (scope, decisions, context) and `validation.md` (automated + manual checks, definition of done) recorded on the branch.
3. `tests` -- failing tests committed first (real data where possible) (Opus 5, xhigh, ultracode).
4. `implementation` (Opus 5, xhigh, ultracode).
5. `adversarial review` -- independent reviewers try to refute correctness against the EMSphInx source and the spec (`/code-review high` + the EMSphInx-gotcha checklist from `specs/_research/explore-emsphinx-core-algorithm.md` section 8) (Opus 5).
6. `fix` -> `pre-commit` -> signed commit -> PR into fork `develop` using `.github/PULL_REQUEST_TEMPLATE.md`, stating the GPL-only licensing up front.

Documentation-only phases (Phase 0) skip the failing-tests and CHANGELOG gates.
