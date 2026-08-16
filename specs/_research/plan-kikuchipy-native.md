# Implementation plan: EMSphInx spherical indexing (CPU) → kikuchipy

Perspective: kikuchipy-native integration and maintainability. Reuse orix/kikuchipy abstractions, add the smallest possible public surface, no new hard dependencies, dask + numba execution, numpydoc, spec-driven branches.

Repo facts verified during planning (`c:/Users/westraadt.1/Repos/kikuchipy`, branch `develop`, 0.13.dev0): `origin` = `jwestraadt/kikuchipy` fork, **no `upstream` remote**; local clone lacks `doc/dev/licensing_considerations.rst` (exists upstream) → first action is `git remote add upstream https://github.com/pyxem/kikuchipy.git && git fetch upstream && git merge upstream/develop`. Stray untracked `IndexEBSD.nml` in repo root (created by an earlier `IndexEBSD.exe -t`) must be deleted before any commit. `src/kikuchipy/indexing/__init__.pyi` currently exports 9 names (verified). `_get_direction_cosines_from_detector(detector, signal_mask=None) -> np.ndarray` in `src/kikuchipy/signals/util/_master_pattern.py:83` (verified) — this is the sphere-direction source for back-projection. kikuchipy's `oxford_binary` plugin already reads `.ebsp` incl. beam x/y footers and derives the navigation shape (`src/kikuchipy/io/plugins/oxford_binary/_api.py:282-322`) → **EBSPDims and PatternRepack need no new code** (see F8 docs).

---

## A. Constitution (`specs/`)

### `specs/mission.md` (one paragraph)
> Bring EMSphInx's spherical-harmonic (SHT) EBSD indexing (Lenthe, Singh & De Graef, Ultramicroscopy 207 (2019) 112841) into kikuchipy as a first-class, CPU-only, notebook-driven indexing method — `EBSD.spherical_indexing()` backed by `kikuchipy.indexing.SphericalIndexer` and `kikuchipy.indexing.MasterPatternHarmonics` — plus the supporting EMSphInx workflows (master pattern → harmonics / `.sht` I/O, harmonics → image, pseudo-symmetry prediction), while reusing kikuchipy/orix abstractions (`EBSDDetector` direction cosines, `EBSDMasterPattern` Lambert grids, `CrystalMap` results, existing preprocessing, `get_indexer`-like construction), adding no new hard dependencies, following kikuchipy's contribution guidelines (numpydoc, ruff/black, dask+numba, pytest with real data, Keep-a-Changelog), respecting the GPL-2.0-or-later → GPL-3.0-or-later relicensing obligations (CMU/Lenthe copyright notices retained), and delivering a `spherical_indexing.ipynb` tutorial comparable to the Hough and dictionary indexing tutorials.

### `specs/tech-stack.md` (bullets)
- **Runtime deps (all already required by kikuchipy, none added):** numpy, scipy (`scipy.fft` for ring rFFTs, 3-D `irfftn`, `dctn/idctn`; `numpy.polynomial.legendre.leggauss` for Legendre nodes/weights; `scipy.linalg.cho_factor/cho_solve`), numba (`@njit(cache=True, nogil=True)`; `fastmath=False` for SHT/Wigner recursions until benchmarked), dask (`da.map_blocks` over pattern chunks), orix (`Rotation`, `Symmetry`, `Phase/PhaseList`, `CrystalMap`, `create_coordinate_arrays`, `IPFColorKeyTSL`), h5py (EMsoft `accum_e`), matplotlib (tutorial only).
- **Explicitly not used:** FFTW/pyfftw (GPL, no wheels), shtns (CeCILL, no wheels), rocket-fft (no cp314 wheels), pyshtools (BSD-3 but wrong grids; see §C.2 — not adopted, may become an optional test oracle later), sympy (test oracle replaced by hard-coded reference values ported from `EMSphInx/test/sht/wigner.cpp` and `scipy.special.sph_harm_y`).
- **Layout:** private package `src/kikuchipy/indexing/_spherical/` (modules `_square_grid.py`, `_sht.py`, `_symmetry.py`, `_harmonics.py`, `_sht_file.py`, `_wigner.py`, `_correlator.py`, `_back_projection.py`, `_indexer.py`, `_pseudo_symmetry.py`); public names via `src/kikuchipy/indexing/__init__.pyi` (`lazy_loader.attach_stub`), sorted `__all__`.
- **License headers:** kikuchipy GPL-3.0-or-later header (new style, `Copyright 2019-2026 the kikuchipy developers`) **plus** in every ported file: `# Portions derived from EMSphInx (https://github.com/EMsoft-org/EMSphInx), Copyright (c) 2019 De Graef Group, Carnegie Mellon University, author William C. Lenthe, GPL-2.0-or-later; translated to Python and modified by <name>, 2026, and relicensed under GPL-3.0-or-later per the "or any later version" clause.` Never place this code in a BSD-3 area of kikuchipy (upstream `doc/dev/licensing_considerations.rst`).
- **Style:** PEP 8/Black via `pre-commit run --all-files` (ruff, ruff-format, black-jupyter@77 for notebooks); numpydoc, types in signatures only, comment/docstring lines ≤ 72 chars; three import blocks; `_`-prefixed private modules; `print()` for user progress, `warnings.warn` soft warnings, `ValueError` with offending vs expected values.
- **Rotation conventions:** orix `Rotation.from_euler(..., direction="lab2crystal")` ≡ EMSphInx `eu2qu` (verified bit-identical); correlator ZYZ `(α,β,γ)` → Bunge `(α−π/2, β, γ+π/2)`; final orientation = `~Rotation.from_euler(bunge)` (conjugate; `northPoleQuat` is identity in EMSphInx and is not ported).
- **Test/doc commands:** `pytest tests/test_indexing -n 4 --cov`; `pytest --doctest-modules src/kikuchipy/indexing/_spherical`; numba kernels tested both as `f(...)` and `f.py_func(...)` (small `bw` for py_func); `pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py`; `pytest --weekly` for the EMSphInx regression; `cd doc && make html` (numpydoc validation runs at build); `./doc/tutorials/run_nbval.sh` if the notebook stores outputs.
- **Real data policy:** unit/integration tests use in-package `kp.data.nickel_ebsd_small()` + `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)`; synthetic-but-physical tests use `EBSDMasterPattern.get_patterns()` at known orientations; heavier tests use `nickel_ebsd_large` (downloaded at pytest session start) and are `@pytest.mark.weekly` if > ~30 s; EMSphInx golden files via `kikuchipy-data` + pooch registry.
- **Git:** branch off `upstream/develop`, kebab-case branch names, `git commit -s`, merge (not rebase) develop, PR into `develop`, PR template checklist, `CHANGELOG.rst` Unreleased → Added, credits (`src/kikuchipy/__init__.py::credits`, `.zenodo.json`).
- **Workflow per feature:** plan → approve → spec recorded in `specs/<date>-<name>/{plan.md,requirements.md,validation.md,tasks.md}` → tests (real data first) → implementation → adversarial review (`/code-review high` on the branch diff + manual checklist in `validation.md`) → fix → commit → PR.

### `specs/roadmap.md` (ordered phases = features; each a checkbox list)
```
Phase 0  Bootstrap (branch spherical-indexing-specs)
  [ ] add upstream remote, merge upstream/develop, delete stray IndexEBSD.nml
  [ ] commit specs/mission.md, tech-stack.md, roadmap.md
  [ ] open pyxem/kikuchipy issue "Spherical indexing (EMSphInx port)" referencing EMSphInx#7; e-mail pyxem.team@gmail.com re GPL2+→GPL3+ and CMU patent status
Phase 1  Square Legendre grid + discrete SHT (F1)
  [ ] _square_grid.py: legendre normals, ring index tables, ring solid angles, Lambert<->Legendre regrid, DCT resampler
  [ ] _sht.py: SquareLegendreSHT.analyze/synthesize (numba)
  [ ] tests: f=1 -> a00=sqrt(4pi); synth->analyze identity to 5e-3 up to bw=384; sph_harm_y oracle; py_func
Phase 2  Master pattern harmonics + .sht I/O (F2)
  [ ] _symmetry.py: z-rotation order / equatorial mirror per orix Symmetry
  [ ] _harmonics.py: MasterPatternHarmonics (from_master_pattern, from_file, to_file, to_master_pattern, resize)
  [ ] EBSDMasterPattern.get_spherical_harmonics; EMsoft reader exposes MC energy weights
  [ ] _sht_file.py: v1.1 reader/writer, CRC-32C, pack/unpack, SG LUTs; tests vs EMSphInx/data/Ni {20kV 75.7deg}.sht
Phase 3  Wigner d functions (F3)
  [ ] _wigner.py: d(pi/2) tables, d(beta) tables (+pre-built e/w/b), 1st/2nd derivatives, rotate_harmonics
  [ ] tests vs Mathematica tables ported from EMSphInx test/sht/wigner.cpp; MasterPatternHarmonics.rotate
Phase 4  SO(3) cross-correlation + refinement (F4)
  [ ] _correlator.py: compute (symmetry-reduced), irfftn, peak, tri-quadratic interp, Newton refine, normalized correlator
  [ ] tests: rotate-and-recover < 1e-4 deg (bw 53..158), normalized variant, point groups {2,m,2/m,3,4,4/m,6,6/m,m-3m}
Phase 5  Indexer + EBSD API (F5)
  [ ] _back_projection.py: detector -> sphere LUT from _get_direction_cosines_from_detector
  [ ] _indexer.py: SphericalIndexer, EBSDDetector.get_spherical_indexer, EBSD.spherical_indexing (dask map_blocks)
  [ ] tests: simulated patterns (get_patterns) < 0.5 deg; nickel_ebsd_small vs bundled xmap; benchmark; CHANGELOG; bib
Phase 6  Pseudo-symmetry (F6)
  [ ] _pseudo_symmetry.py: find_pseudo_symmetry (MasterXcorr); indexer pseudo_symmetry_ops
Phase 7  EMSphInx reference regression (F7)
  [ ] generate golden .sht/.npz with build/Release exes; host in kikuchipy-data; weekly test
Phase 8  Tutorial + docs (F8)
  [ ] doc/tutorials/spherical_indexing.ipynb, tutorials/index.rst, related_projects.rst, installation notes, run_nbval.sh
  [ ] final PR to pyxem/kikuchipy develop (specs/ excluded or agreed with maintainers)
```

---

## B. Features / branches

Common to every feature: spec folder with `plan.md` (approach, decisions), `requirements.md` (numbered REQ-n with acceptance), `validation.md` (commands, tolerances, adversarial-review checklist), `tasks.md` (checkboxes). PR into `develop` (fork first, upstream at the end), `git commit -s`, PR template filled, pre-commit clean.

### F0 — `spherical-indexing-specs` (folder: `specs/` root files only)
Files: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`. No code. Validation: `pre-commit run --all-files` passes; `git status` clean of `IndexEBSD.nml`.

### F1 — Square Legendre grid and discrete SHT
- **Branch:** `spherical-square-legendre-sht`; **spec:** `specs/2026-08-16-square-legendre-sht/`
- **New files:** `src/kikuchipy/indexing/_spherical/__init__.py` (header-only, plus the "internal use only" docstring pattern of `indexing/_refinement/__init__.py`), `src/kikuchipy/indexing/_spherical/_square_grid.py`, `src/kikuchipy/indexing/_spherical/_sht.py`, `tests/test_indexing/test_spherical_grid.py`, `tests/test_indexing/test_spherical_sht.py`.
- **API (private):**
  ```python
  # _square_grid.py
  def _grid_dim_from_bandwidth(bandwidth: int) -> int            # bw + (2 if bw % 2 else 3)
  def _legendre_ring_cosines(dim: int) -> np.ndarray             # [1, leggauss(dim-2) positive roots desc], size (dim+1)//2
  def _legendre_ring_weights(dim: int) -> np.ndarray             # 4*pi*w_hat_y/max(1,8y); pole 0; equator halved (see §C)
  def _legendre_normals(dim: int) -> np.ndarray                  # (dim*dim, 3) north hemisphere, row-major (j*dim+i)
  def _ring_number(dim: int) -> np.ndarray                       # (dim*dim,) Chebyshev ring index
  def _ring_indices(dim: int) -> tuple[np.ndarray, np.ndarray]   # flat CCW index table (offsets, indices) for readRing/writeRing
  def _ring_solid_angles(dim: int) -> np.ndarray                 # per ring, relative to mean pixel
  def _lambert_to_legendre(nh: np.ndarray, sh: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]  # bilinear via _vector2lambert
  def _resample_dct(image: np.ndarray, shape: tuple[int, int], zero_mean: bool = False) -> np.ndarray  # scipy.fft.dctn/idctn (image::Rescaler)
  # _sht.py
  class _SquareLegendreSHT:
      def __init__(self, dim: int, bandwidth: int | None = None): ...   # bandwidth <= dim - 2
      def analyze(self, north: np.ndarray, south: np.ndarray, out: np.ndarray | None = None) -> np.ndarray   # (bw, bw) complex128 [m, l]
      def synthesize(self, alm: np.ndarray) -> tuple[np.ndarray, np.ndarray]                                # (dim, dim) x 2
  ```
  Numba kernels: `_analyze_rings(...)`, `_synthesize_rings(...)`, `_alf_coefficients(bw)` (amn/bmn tables). Ring FFTs with `scipy.fft.rfft/irfft` on stacked per-ring buffers (one Python call per ring or padded batch), recursion in numba.
- **Reused:** `kikuchipy.signals.util._master_pattern._vector2lambert`, `_lambert2vector`, `_get_lambert_interpolation_parameters` (bilinear scheme), `numpy.polynomial.legendre.leggauss`.
- **Tests:** synthetic — `f = 1` → `a[0,0] == sqrt(4π)` (rtol 1e-12), all others < 1e-12; deterministic `np.random.default_rng(0)` band-limited coefficients → `synthesize → analyze` identity (max abs err ≤ 5e-3, mean ≤ 5e-5) for bw ∈ {4, 8, 16, 53, 68, 88, 113, 158, 384} (bw ≥ 158 marked slow but < 10 s); comparison with `scipy.special.sph_harm_y` evaluated on the Legendre grid for a single `Y_l^m` (l ≤ 6) — analyze recovers `δ_lm` to 1e-8; Legendre weights equal `leggauss` weights (equator halved) to 1e-13; `_ring_indices` covers every pixel exactly once and matches azimuth ordering; py_func variants at bw = 8. Real data — `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")` regridded to dim = 71 then analyzed at bw = 68 and synthesized: correlation with the regridded input ≥ 0.99 on the north hemisphere.
- **Acceptance:** all above; runtime of analyze at bw = 68 < 20 ms per sphere after JIT.
- **Validation commands:** `pytest tests/test_indexing/test_spherical_grid.py tests/test_indexing/test_spherical_sht.py -n 4 --cov=kikuchipy.indexing._spherical`; `pytest --doctest-modules src/kikuchipy/indexing/_spherical`; `pre-commit run --all-files`.
- **CHANGELOG:** none yet (private).

### F2 — Master pattern harmonics, symmetry tables, `.sht` I/O (mp2sht + sht2png equivalents)
- **Branch:** `spherical-master-pattern-harmonics`; **spec:** `specs/2026-08-18-master-pattern-harmonics/`
- **New/changed files:** `_spherical/_symmetry.py`, `_spherical/_harmonics.py`, `_spherical/_sht_file.py`; `src/kikuchipy/indexing/__init__.pyi` (+`MasterPatternHarmonics`); `src/kikuchipy/signals/ebsd_master_pattern.py` (+`get_spherical_harmonics`); `src/kikuchipy/io/plugins/_emsoft_master_pattern.py` (store normalized `sum(EMData/MCOpenCL/accum_e, axis=(0,1))` as `original_metadata.EMData.MCOpenCL.energy_weights` — small, backwards compatible); tests `tests/test_indexing/test_master_pattern_harmonics.py`, `tests/test_indexing/test_sht_file.py`, `tests/test_indexing/test_spherical_symmetry.py`, additions to `tests/test_signals/test_ebsd_master_pattern.py`, `tests/test_io/test_emsoft_ebsd_master_pattern.py`.
- **Public API:**
  ```python
  class MasterPatternHarmonics:  # kikuchipy.indexing
      def __init__(self, coefficients: np.ndarray, phase: Phase, sample_tilt: float = 70.0, energy: float | None = None, name: str | None = None): ...
      coefficients: np.ndarray        # (bw, bw) complex128, a_l^m at [m, l], zeros for l < m, m >= 0 only
      bandwidth: int
      phase: Phase                     # orix; point_group drives symmetry flags
      z_rotation_order: int            # EMSphInx nFold / fNf
      has_equatorial_mirror: bool      # EMSphInx mirror / fMr
      sample_tilt: float; energy: float | None
      @classmethod
      def from_master_pattern(cls, master_pattern: "EBSDMasterPattern", bandwidth: int = 384, energy: int | float | None = None, energy_weights: np.ndarray | None = None, normalize: bool = True) -> "MasterPatternHarmonics"
      @classmethod
      def from_file(cls, filename: str | Path) -> "MasterPatternHarmonics"        # EMSphInx .sht v1.1
      def to_file(self, filename: str | Path, notes: str = "", doi: str = "", overwrite: bool = False) -> None
      def to_master_pattern(self, npx: int | None = None, projection: str = "lambert", hemisphere: str = "both") -> "EBSDMasterPattern"   # sht2png equivalent; synthesize on Legendre grid, regrid to Lambert (and stereographic via existing plotting)
      def resize(self, bandwidth: int) -> "MasterPatternHarmonics"
      def remove_dc(self) -> "MasterPatternHarmonics"
      def __repr__(self) -> str    # "<MasterPatternHarmonics: ni (m-3m), bandwidth 384, 20 keV>"
  # EBSDMasterPattern
  def get_spherical_harmonics(self, bandwidth: int = 384, energy: int | float | None = None, energy_weights: np.ndarray | None = None, normalize: bool = True) -> MasterPatternHarmonics
  ```
  `_symmetry.py`: `_z_rotation_order(symmetry: Symmetry) -> int`, `_has_equatorial_mirror(symmetry) -> bool` (32-entry dicts keyed on `Symmetry.name`, cross-checked in tests against `Symmetry.get_axis_orders()` and improper-diad normals), `_space_group_z_rotation(sg)`/`_space_group_compression(sg)` LUTs copied verbatim from `sht_file.in.hpp:1838-1869`.
  `_sht_file.py`: `_read_sht(path) -> dict`, `_write_sht(path, ...)`, `_crc32c(bytes)` (EMSphInx variant with the LUT copied verbatim), `_pack_harmonics/_unpack_harmonics(alm, bw, z_rot, flags)`.
- **Behavioural spec (from EMSphInx `master.hpp:550-595`, `mp2sht.cpp`):** require `projection == "lambert"`, `hemisphere == "both"` (raise `ValueError` otherwise, using `_is_suitable_for_projection`-style checks); energy-weighted average over the energy axis with weights = MC `accum_e` sums (from `original_metadata`) or uniform with a `warnings.warn`; if source dim < √2·(bw+2) DCT-upsample; regrid to Legendre `dim = bw + 2 (+1 if even)`; solid-angle-weighted mean/std over both hemispheres — **fix EMSphInx bugs 8.7/8.8** (mean divided by `2*totW`; corners halved once) and document; analyze at bw. Symmetry flags from `phase.point_group`. Reading `.sht`: kV, primary angle → `sample_tilt`, `sgEff` → `Phase(space_group=sgEff)` (point group via orix), lattice if `numXtal == 1`.
- **Reused:** `EBSDMasterPattern` axes (`hemisphere`, `energy`, `phase`), `_lambert2vector`, `kikuchipy.io.plugins._emsoft_master_pattern` reader, orix `Phase`, `get_point_group`.
- **Tests:** real — `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)` → `get_spherical_harmonics(68)`: `z_rotation_order == 4`, `has_equatorial_mirror`, systematic zeros (`m % 4 != 0` rows and `(l+m)` odd) below 1e-6 relative; round trip `to_file → from_file` bit-equal coefficients, CRC valid; `EMSphInx/data/Ni {20kV 75.7deg}.sht` (copied into `tests`? no — see F7 hosting; for F2 read it from an env-var path and skip if absent, plus a synthetic `.sht` written by our writer): header fields exactly as in the exploration report (bw 384, zRot 4, cmpFlg 7, doubCnt 9312, CRC 0xf2af93ef); `to_master_pattern()` returns an `EBSDMasterPattern` whose north hemisphere correlates ≥ 0.98 with the input Lambert master (bw 384 vs 401 px); `resize` up/down round trip; energy weighting: for `ebsd_master_pattern("ni")` (cached locally, marked `weekly`) weights sum to 1 and match h5py-read `accum_e`; symmetry tables for all 32 point groups vs orix-derived checks; error paths (stereographic input, single hemisphere).
- **Acceptance:** above; `from_master_pattern` at bw = 384 from a 1001² MP < 15 s.
- **Validation:** `pytest tests/test_indexing/test_master_pattern_harmonics.py tests/test_indexing/test_sht_file.py tests/test_indexing/test_spherical_symmetry.py tests/test_signals/test_ebsd_master_pattern.py -n 4`; doc build for numpydoc validation of the new class.
- **CHANGELOG (Added):** "New class ``kikuchipy.indexing.MasterPatternHarmonics`` and method ``EBSDMasterPattern.get_spherical_harmonics()`` computing spherical harmonic coefficients of a master pattern, with reading/writing of EMSphInx ``.sht`` files. (`#NNN <...>`_)"; (Changed) "EMsoft master pattern reader stores Monte Carlo energy weights in ``original_metadata``."

### F3 — Wigner d functions and harmonic rotation
- **Branch:** `spherical-wigner-d`; **spec:** `specs/2026-08-20-wigner-d/`
- **Files:** `_spherical/_wigner.py`; `tests/test_indexing/test_spherical_wigner.py`; `MasterPatternHarmonics.rotate(rotation: Rotation) -> MasterPatternHarmonics` added in `_harmonics.py`.
- **API (private numba):** `_wigner_d_half_pi(bw) -> np.ndarray` shape `(bw, bw, bw)` with `d[k, m, j] = d^j_{k,m}(π/2)` (and the transposed layout used by the correlator via array transposition, no second table); `_wigner_d_tables(bw, t, negative_beta, e_table, w_table, b_table) -> np.ndarray` shape `(bw, bw, bw, 2)` = `d^j_{k,m}(β)`, `d^j_{k,m}(π−β)`; `_wigner_pre_tables(bw) -> (e, w, b)`; `_wigner_d(j, k, m, t, negative_beta) -> float` scalar reference; `_rotate_harmonics(alm, zyz) -> np.ndarray`; helper `_zyz_from_rotation(rot) -> np.ndarray` and `_rotation_from_zyz(zyz) -> Rotation` (`eu = (α−π/2, β, γ+π/2)`, `Rotation.from_euler`).
- **Tests:** port the j ≤ 4, |k|,|m| ≤ 4 Mathematica tables from `EMSphInx/test/sht/wigner.cpp` for β ∈ {π/2, π/3, 2π/3, −π/3, −2π/3} as numpy constants (tolerance 2·eps for scalars, tables at β = 0.9708055194 vs scalar, derivatives 24·eps); `_wigner_d_half_pi` orthogonality `Σ_k d^j_{k,m} d^j_{k,n} = δ_mn`; `rotate_harmonics` vs brute force: synthesize random band-limited f, rotate the Legendre grid directions with orix `Rotation`, re-interpolate and analyze — coefficients agree to 1e-2 for bw = 16 (interpolation-limited) and exactly (1e-10) for a Y_l^m analytic check with `sph_harm_y`; identity rotation is a no-op; composition `rotate(a).rotate(b) == rotate(b*a)`; py_func at bw = 8. Real data: `nickel_ebsd_master_pattern_small` harmonics rotated by a cubic symmetry element → coefficients unchanged (rel. 1e-6) — a strong convention check.
- **Validation:** `pytest tests/test_indexing/test_spherical_wigner.py -n 4`.

### F4 — SO(3) cross-correlation, peak interpolation, Newton refinement, normalized correlator
- **Branch:** `spherical-cross-correlation`; **spec:** `specs/2026-08-22-spherical-cross-correlation/`
- **Files:** `_spherical/_correlator.py`; `tests/test_indexing/test_spherical_correlator.py`.
- **API (private):**
  ```python
  class _SphericalCorrelator:
      def __init__(self, bandwidth: int, workers: int = 1): ...          # precomputes d(pi/2), pre-tables, slP=next_fast_len(2bw-1), buffers
      def compute(self, flm, gln, mirror: bool, z_rot: int) -> np.ndarray   # xc[k, n, m], shape (bwP, slP, slP)
      def find_peak(self, xc, weights=None) -> int
      def interpolate_peak(self, xc, index) -> tuple[np.ndarray, float]   # zyz, value
      def refine_peak(self, flm, gln, mirror, z_rot, zyz, eps=0.01) -> tuple[np.ndarray, float]
      def correlate(self, flm, gln, mirror, z_rot, refine=True) -> tuple[np.ndarray, float]
  class _NormalizedCorrelator(_SphericalCorrelator):
      def __init__(self, bandwidth, flm, flm2, mirror, z_rot, mlm, workers=1)   # rDen from Huhle et al.
      def correlate(self, gln, refine=True) -> tuple[np.ndarray, float]
  ```
  numba kernels: `_fill_spectrum(flm, gln, d_half, mirror, z_rot, fxc)` (k, n, m, j loops with systemic-zero skipping; `parallel=False`, nogil), `_extract_neighbourhood`, `_interpolate_maxima` (tri-quadratic + Newton, fix the `x[2]` bounds bug), `_derivatives(...)` (xc, Jacobian, Hessian), Newton loop with `scipy.linalg.cho_factor` fallback logic (2×2 / 1×1 degeneracies) in Python or numba `np.linalg`.
- **Reused:** `scipy.fft.next_fast_len(2*bw-1)`, `scipy.fft.irfftn(fxc, s=(slP,)*3, norm="forward", workers=...)`.
- **Tests (all synthetic, deterministic `default_rng(0)`, mirroring `EMSphInx/test/sht/sht_xcorr.cpp`):** random band-limited sphere with optional z-mirror / n-fold (grid symmetrisation helpers in `_square_grid.py`), random orix `Rotation`, `_rotate_harmonics` → `correlate(refine=True)` recovers the rotation: misorientation < 1e-4° for bw ∈ {53, 68, 88, 113, 123, 158} and padded sizes {54..64}; without refinement < 360/(2·slP)·1.5; normalized correlator with a wedge mask (as in `testNCorr`) < 1e-3°; point groups {112, 11m, 112/m, 3, 4, 4/m, 6, 6/m, m-3m} at bw 53..63 → misorientation < 0.012° after symmetry reduction via `orix.Misorientation(...).reduce()`; `find_peak`/`interpolate_peak` index↔Euler mapping round trip (`eulerIndex`), including the glide `R(a,b,g) == R(a+π,−b,g+π)`; Newton fails gracefully at β = 0/π (returns interpolated); py_func at bw = 12. Real data: Ni harmonics (bw 68) autocorrelation → maxima at the 24 proper cubic operators (`orix.O`) each within 0.05°.
- **Acceptance:** above; `compute` at bw = 68 ≤ 40 ms, refine ≤ 25 ms per call after JIT (target; record in spec).
- **Validation:** `pytest tests/test_indexing/test_spherical_correlator.py -n 4`.

### F5 — Back-projection, `SphericalIndexer`, `EBSDDetector.get_spherical_indexer`, `EBSD.spherical_indexing`
- **Branch:** `spherical-indexer`; **spec:** `specs/2026-08-25-spherical-indexer/`
- **New/changed files:** `_spherical/_back_projection.py`, `_spherical/_indexer.py`; `src/kikuchipy/indexing/__init__.pyi` (+`SphericalIndexer`); `src/kikuchipy/detectors/_ebsd_detector.py` (+`get_spherical_indexer`); `src/kikuchipy/signals/ebsd.py` (+`spherical_indexing`, placed after `dictionary_indexing` L1984); `benchmarks/indexing/test_spherical_indexing.py`; `doc/user/bibliography.bib` (+`lenthe2019spherical`, `lenthe2019pseudo`, `huhle2009` if cited); `CHANGELOG.rst`; tests `tests/test_indexing/test_spherical_back_projection.py`, `tests/test_indexing/test_spherical_indexer.py`, `tests/test_signals/test_ebsd_spherical_indexing.py`, `tests/test_detectors/test_ebsd_detector.py` (+get_spherical_indexer).
- **Public API:**
  ```python
  class SphericalIndexer:  # kikuchipy.indexing
      def __init__(self, harmonics: MasterPatternHarmonics | list[MasterPatternHarmonics], detector: EBSDDetector,
                   bandwidth: int = 68, normalize: bool = True, refine: bool = True,
                   signal_mask: np.ndarray | None = None, resample: bool = True,
                   pseudo_symmetry_ops: Rotation | None = None): ...
      bandwidth: int; phase_list: PhaseList; detector: EBSDDetector; normalize: bool; refine: bool
      grid_dim: int; correlation_shape: tuple[int, int, int]
      def index_patterns(self, patterns: np.ndarray, n_best: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]
          # patterns (n, nrows, ncols) -> rotations quats (n, n_best, 4), scores (n, n_best), phase index (n, n_best)
      def refine_orientations(self, patterns: np.ndarray, rotations: Rotation, phase_index: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]
      def get_correlation(self, pattern: np.ndarray, phase_index: int = 0) -> np.ndarray   # full xc cube (diagnostics/plots)
      def project_to_sphere(self, pattern: np.ndarray) -> np.ndarray   # (2, dim, dim) back-projected sphere (diagnostics/tutorial)
      def get_info_message(self, nav_size: int, chunksize: int) -> str
      def __repr__(self)
  # EBSDDetector
  def get_spherical_indexer(self, harmonics, bandwidth: int = 68, **kwargs) -> SphericalIndexer   # mirrors get_indexer; warns like _warn_if_angles_ignored? (twist/azimuthal ARE supported here via direction cosines)
  # EBSD
  def spherical_indexing(self, indexer: SphericalIndexer, navigation_mask: np.ndarray | None = None,
                         n_best: int = 1, chunk_kwargs: dict | None = None, rechunk: bool = True) -> CrystalMap
  ```
  Returned `CrystalMap`: `rotations` (n [, n_best]), `phase_id` (best), `prop = {"scores": (n, n_best), "phase_index": (n, n_best) only if multi-phase, "pseudo_symmetry_index" if ops given}`, `scan_unit = _get_navigation_axes_unit(am)`, `is_in_data` from `navigation_mask`. Prints `Spherical indexing information:` block (bandwidth, side length, phases with z-rot/mirror, PC (Bruker, mean), n patterns / chunks) then `Indexing speed: X patterns/s` (matches `tutorials_sanitize.cfg`).
- **Back-projection design:** for `dim = bw + 2/3`, build the sphere→detector LUT once per detector PC (support per-point PC arrays lazily: recompute LUT when PC changes; v1 requires a single mean PC and warns/raises otherwise, matching Hough); resample detector to `(round(w·s), round(h·s))` with `s = √2·scale_factor(dim)` via `_resample_dct`; direction of every Legendre grid point → detector fractional coords via `detector.sample_to_detector` + gnomonic bounds (`_convert_detector_coordinates.convert_gnomonic_to_pixel_coords`); keep points inside frame ∧ `signal_mask==False` (kikuchipy convention: True = ignore, inverted internally) ∧ `z_det>0`; store as `scipy.sparse.csr_matrix` (n_sphere_pts × n_det_px) of bilinear weights + per-point ring solid angles; `unproject(pattern)`: DCT resample (zero-mean) → sparse matvec → weighted zero-mean/unit-std → scatter into `(2, dim, dim)` (south handled correctly with `dim*dim + i`, fixing EMSphInx bug 8.9); binary window mask → `mlm`.
- **Reused:** `_get_direction_cosines_from_detector` (used to *validate* the inverse mapping and to compute the window in tests), `EBSDDetector.sample_to_detector`, `gnomonic_bounds`, `pc_flattened`, `_detector_is_compatible_with_signal`, `_get_indexed_points_in_data_in_xmap`, `_get_navigation_axes_unit`, `get_dask_array`, `get_chunking`, `create_coordinate_arrays`, `dask.diagnostics.ProgressBar`, `_get_sum_signal`-style helpers not needed.
- **Threading:** dask `map_blocks` over pattern chunks (`chunk_kwargs`, default `chunk_shape` 64 like refinement); each chunk runs the numba nogil kernels serially per pattern with a per-thread `_SphericalCorrelator` (thread-local cache keyed by thread id, since buffers/tables ≈ 50–290 MB — see §C.5); `scipy.fft` calls with `workers=1`.
- **Tests:** synthetic-physical — `nickel_ebsd_master_pattern_small` + `EBSDDetector(shape=(60,60), pc=(0.42,0.21,0.50), sample_tilt=70)`: `mp.get_patterns(R, det, energy=20)` for 12 random `R` (incl. β≈0 and near-symmetric ones) → `SphericalIndexer(h, det, bandwidth=68).index_patterns` → `Rotation.angle_with` after `Orientation.reduce()`: all < 0.5° with refine, < 3° without; scores of true match > any pseudo-match. Real — `kp.data.nickel_ebsd_small()` (backgrounds removed) with `det = s.detector` (`pc = pc_average`): 9 patterns vs bundled `s.xmap`: `np.all(angles < 2°)` after refine (fallback assertion `mean(angles < 2°) ≥ 8/9` if flaky), scores > 0.3; multi-phase list `[ni, ni_rotated_by_5deg]` picks phase 0; `navigation_mask`; lazy signal; `n_best=3`; error paths (bandwidth out of [16, 512], detector shape mismatch, harmonics bandwidth < indexer bandwidth, stereographic MP). `nickel_ebsd_large` (4125 patterns, bw 53, refine) vs bundled xmap: median < 1°, `mean(angles < 2°) > 0.95` — `@pytest.mark.weekly` if > 30 s locally, else regular. Detector: `get_spherical_indexer` returns indexer with `detector is det`; per-point PC → `ValueError` (v1). numba `.py_func` for `_fill_spectrum` already covered in F4; back-projection kernel py_func on a 16×16 detector.
- **Benchmark:** `benchmarks/indexing/test_spherical_indexing.py::test_spherical_indexing(benchmark)` on `nickel_ebsd_small` (bw 68) asserting `xmap.scores.mean() > 0.3`.
- **Docs:** numpydoc on all; `:cite:\`lenthe2019spherical\`` in `EBSD.spherical_indexing` and `SphericalIndexer` docstrings; `See Also` cross-links to `dictionary_indexing`, `hough_indexing`, `refine_orientation`.
- **CHANGELOG (Added):** "Spherical indexing of EBSD patterns via the spherical harmonic transform, ported from EMSphInx (CPU only): ``EBSD.spherical_indexing()``, ``kikuchipy.indexing.SphericalIndexer`` and ``EBSDDetector.get_spherical_indexer()``. (`#NNN <...>`_)".
- **Validation:** `pytest tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py tests/test_detectors/test_ebsd_detector.py -k spherical -n 4 --cov`; `pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py`; `cd doc && make html`.

### F6 — Pseudo-symmetry prediction (MasterXcorr equivalent)
- **Branch:** `spherical-pseudo-symmetry`; **spec:** `specs/2026-08-29-pseudo-symmetry/`
- **Files:** `_spherical/_pseudo_symmetry.py`; `indexing/__init__.pyi` (+`find_pseudo_symmetry`); `tests/test_indexing/test_pseudo_symmetry.py`; indexer support finished (`pseudo_symmetry_ops`: after the best peak, refine from `q0 * op` for each op, keep top-n, record `pseudo_symmetry_index` as in `_refinement.py`).
- **Public API:**
  ```python
  def find_pseudo_symmetry(harmonics: MasterPatternHarmonics, other: MasterPatternHarmonics | None = None,
                           bandwidth: int = 88, cutoff: float = 0.7, refine: bool = True, min_separation: float = 2.0,
                           return_correlation: bool = False) -> tuple[Rotation, np.ndarray] | tuple[Rotation, np.ndarray, np.ndarray]
      # returns candidate operators (true point-group operators removed via orix Symmetry, sorted by score desc), relative scores, optional xc volume (bwP, slP, slP)
  ```
  Also `MasterPatternHarmonics.plot_correlation`? — no; tutorial shows the volume slice with matplotlib and operators on `orix.plot.StereographicPlot`.
- **Tests:** synthetic random sphere with imposed 4-fold: `find_pseudo_symmetry(other=None)` finds no candidates above 0.7 besides removed true ops; a sphere built as `f + 0.9·rotate(f, 60° about z)` yields the 60° operator with score ≈ 0.9; real: `nickel_ebsd_master_pattern_small` bw 88 → known Ni pseudo-symmetries absent (empty above 0.7) — and Mg/Ti hcp masters (local only, skip if absent) as an exploratory notebook check, not CI. Indexer with `pseudo_symmetry_ops` on simulated patterns → `pseudo_symmetry_index` 0 for all.
- **CHANGELOG (Added):** "``kikuchipy.indexing.find_pseudo_symmetry()`` predicting pseudo-symmetry operators from master pattern spherical cross-correlation (EMSphInx MasterXcorr equivalent); ``SphericalIndexer`` accepts ``pseudo_symmetry_ops``."

### F7 — EMSphInx reference regression (real data, weekly)
- **Branch:** `spherical-indexing-emsphinx-reference`; **spec:** `specs/2026-08-31-emsphinx-reference-regression/` (also holds the generation scripts + `.nml` under `specs/.../scripts/`, not shipped).
- **Files:** `src/kikuchipy/data/_registry.py` (+ hashes/URLs for `emsphinx/ni_20kv_bw384.sht`, `emsphinx/nickel_ebsd_small_bw68_ref.npz`, `emsphinx/nickel_ebsd_large_bw53_ref.npz`), `src/kikuchipy/data/_data.py` (+ private helper `_emsphinx_reference(name, allow_download)` or public `emsphinx_reference_indexing()` — recommend private + used only in tests), `tests/test_indexing/test_spherical_emsphinx_reference.py` (`@pytest.mark.weekly`), `tests/test_data/test_data.py` availability ping.
- **Generation (see §D):** `mp2sht.exe`, `IndexEBSD.exe` with the nml in §D; convert `.ang` → `.npz` (phi1, Phi, phi2, metric, phase) with orix.
- **Tolerances:** see §D.
- **CHANGELOG:** none (tests/data only), unless the `.sht` is exposed publicly.

### F8 — Tutorial notebook and documentation
- **Branch:** `spherical-indexing-tutorial`; **spec:** `specs/2026-09-02-spherical-indexing-tutorial/`
- **Files:** `doc/tutorials/spherical_indexing.ipynb`, `doc/tutorials/index.rst` (Indexing nbgallery + `spherical_indexing`), `doc/tutorials/run_nbval.sh` (only if outputs are stored), `doc/tutorials/tutorials_sanitize.cfg` (if a new progress line format), `doc/user/related_projects.rst` (+EMSphInx), `doc/user/installation.rst` (note: no extra deps), `README.rst`/`doc/index.rst` feature list if present, `CHANGELOG.rst` (Added: tutorial).
- **Notebook outline** (hidden first MD cell; black@77; outputs stripped unless RTD too slow; `nbsphinx-thumbnail` on the IPF map cell):
  1. `# Spherical indexing` — what SHT indexing is (Lenthe et al. 2019 DOI link, pseudo-symmetry paper), CPU-only, license note.
  2. Imports; `s = kp.data.nickel_ebsd_large(allow_download=True)`.
  3. `## Pre-processing` — `remove_static_background`, `remove_dynamic_background`, IQ map (`get_image_quality`, kikuchipy's replacement for EMSphInx's DCT IQ), optional `adaptive_histogram_equalization` (replacement for `nregions`).
  4. `## Master pattern on the sphere` — `mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)`; `mp.plot_spherical()`; `h = mp.get_spherical_harmonics(bandwidth=384)`; bandwidth discussion with the fast-size table (53, 63, 68, 74, 88, 95, 113, 123, 158, …); `h.resize(68).to_master_pattern().plot()` truncation demo (sht2png equivalent); `h.to_file("ni.sht")` / `MasterPatternHarmonics.from_file` (mp2sht equivalent; mention SHTdatabase).
  5. `## Calibrate detector-sample geometry` — as in hough_indexing (`EBSDDetector(sample_tilt=70)`, `extract_grid`, `hough_indexing_optimize_pc` PSO or simply the known PC), `det.plot(pattern=...)`; note PC conventions (`det.pc_emsoft()`, `pc_tsl()`, `pc_oxford()`, `pc_bruker()` ↔ EMSphInx `vendor`).
  6. `## Perform indexing` — `indexer = det.get_spherical_indexer(h, bandwidth=68)`; `indexer.project_to_sphere(pattern)` visual; `xmap = s.spherical_indexing(indexer)`; save via `orix.io.save` (commented).
  7. `## Validate` — scores map/hist, IPF-X/Y/Z maps (`IPFColorKeyTSL`), geometrical simulation overlays (`KikuchiPatternSimulator.on_detector`), correlation cube slice `indexer.get_correlation(pattern)`.
  8. `## Compare with Hough and dictionary indexing` — `angle_with` vs bundled `s.xmap`, histogram, timing table.
  9. `## Refinement` — `s.refine_orientation(...)` (pattern-space) vs `indexer.refine_orientations` (spherical Newton).
  10. `## Pseudo-symmetry` — `find_pseudo_symmetry(h, bandwidth=88)` on Ni (none) and note for hcp/tetragonal cases.
  11. `## Other EMSphInx utilities in kikuchipy` — PatternRepack ≡ `kp.load(...)` + `EBSD.downsample()` + `EBSD.save("*.h5")`; EBSPDims ≡ `kp.load("scan.ebsp", lazy=True)` (`original_metadata`, navigation shape from beam positions); ShtWisdom not needed (`scipy.fft`).
  12. `## What's next?` — links.
- **Validation:** execute notebook end-to-end locally (`jupyter nbconvert --execute`), `cd doc && make html`, `make linkcheck`; `pre-commit` (black-jupyter).
- **CHANGELOG (Added):** "New tutorial on spherical indexing."

Final step: upstream PR(s) to `pyxem/kikuchipy:develop`. Suggested split for upstream: PR-A (F1–F4 core math, private), PR-B (F5+F6 API), PR-C (F7+F8 data/tutorial) — or a single PR if maintainers prefer; `specs/` kept only in the fork.

---

## C. Algorithm-level design

### C.1 Data structures
- **Sphere grid:** square Legendre grid, odd side `dim = bw + (2 if bw odd else 3)`; two `(dim, dim)` float64 arrays (north, south), row-major `[j, i]` (Y row, X col) — identical layout to kikuchipy's Lambert master arrays (verified equivalence `L = 2·(X,Y) − 1`). Ring `y` has `max(1, 8y)` points; ring index tables precomputed per `dim` (`_ring_indices`), CCW from φ = 0.
- **SH coefficients:** `np.ndarray` complex128, shape `(bw, bw)`, `alm[m, l] = a_l^m` for `0 ≤ m ≤ l < bw`, zeros for `l < m` (EMSphInx m-major `alm[m*bw + l]`; `.sht` maps directly). Negative m implied by real-signal symmetry `a_l^{-m} = (−1)^m conj(a_l^m)`. Fully-normalized ALFs without Condon–Shortley phase (Schaeffer 2013 recursion), odd-m sign applied in ring weights exactly as EMSphInx (`square_sht.hpp:439, 554`).
- **Wigner tables:** `d_half[k, m, j] = d^j_{k,m}(π/2)`, `(bw, bw, bw)` float64; `d_beta[k, m, j, 0/1] = d^j_{k,m}(β), d^j_{k,m}(π−β)`; pre-tables `e[k, m]`, `w[k, m, j]`, `b[k, m, j]` for the refinement; entries with `j < max(k, m)` never read.
- **Correlation:** half-complex spectrum `fxc[k, n, m]` shape `(slP, slP, bwP)` complex128 (`slP = next_fast_len(2·bw−1)`, `bwP = slP//2 + 1`), real cube `xc[k, n, m]` shape `(bwP, slP, slP)` (first bwP β-slices of the `slP³` cube). Euler mapping: `α = 2πm/slP − π/2`, `β = 2πk/slP − π`, `γ = 2πn/slP − π/2`; glide `k ↔ slP−k, m/n += slP/2` for β-wrap.
- **Result:** `CrystalMap` (see F5); orientations `~Rotation.from_euler([α−π/2, β, γ+π/2])`.

### C.2 SHT implementation choice — in-house numpy/numba (recommended), not pyshtools
- EMSphInx's whole pipeline is tied to the *square Legendre grid* (double-cover, ring FFT lengths `8y`, exact GL quadrature giving bandwidth `dim−2`) — the back-projection LUT, the master-pattern normalization and the window mask all live on it. pyshtools (BSD-3, wheels for win/linux/mac cp39–cp314) offers DH/GLQ (Gauss–Legendre latitude × equispaced longitude) transforms and Wigner-D rotation, but not: this grid, the m-major complex layout, the `d(π/2)` tables in the required layout, symmetry-reduced SO(3) correlation, or the analytic Newton refinement — i.e. > 80 % of the code would still be in-house, plus a large compiled dependency and a grid conversion at every pattern. It also cannot be a hard dependency (kikuchipy has `kikuchipy-base` without optional deps).
- Numerical cost is small: at bw = 68, `dim = 71`, analyze is 36 ring rFFTs (lengths ≤ 280) + an O(bw²·Nt) recursion (~1.7e5 flops·Nt ≈ 6 MFLOP) → < 5 ms in numba. The synthetic and `sph_harm_y` tests provide the correctness oracle. **Recommendation:** no new dependency; revisit pyshtools only as an optional cross-validation oracle in tests if the maintainers want it (would be a `[tests]` extra with `skipif`).
- Kernels: `_alf_coefficients(bw)` (amn/bmn), `_analyze_rings`, `_synthesize_rings` `@njit(cache=True, nogil=True)`; ring FFTs via `scipy.fft.rfft` on a `(Nt, 8·(Nt−1))`-padded batch? — no: lengths differ per ring; do a Python loop of `rfft` per ring (36–200 calls, μs each) or `scipy.fft.rfft` on the padded 2-D array with per-row `n` … simplest faithful path: loop per ring (accepted overhead ≈ 0.5 ms). Legendre layout only in v1 (drop `computeWeightsSkip`/Mazonka Lambert weights; weights from `leggauss`, pole 0, equator weight halved; unit-tested against the Sneeuw solve in a one-off test).
- Master-pattern path: Lambert (any odd size) → optional DCT upsample to ≥ √2·dim → bilinear regrid to Legendre normals (`_vector2lambert` on `_legendre_normals`, both hemispheres) → weighted normalization → analyze.

### C.3 SO(3) cross-correlation FFT plan (scipy.fft)
- `bw` user-chosen; recommend/document fast sizes {53, 63, 68, 74, 88, 95, 113, 122, 123, 158, 172, 188, 203, 221, 263, 284, 313}; `slP = scipy.fft.next_fast_len(2·bw−1, real=True)`; warn if `slP != 2·bw−1` (padding cost).
- Numba `_fill_spectrum` reproduces `Correlator::compute` (`sht_xcorr.hpp:657-858`): loop `k ∈ [0,bw)`, `fm[m, j] = flm[m, j]·d^j_{k,m}(π/2)`, for `n ∈ [0,bwP)` (`n < bw`), `gn[j] = conj(gln[n, j])·d^j_{n,k}(π/2)`, inner `j` sums with `conjMult`, systemic-zero skipping (`m % z_rot`, mirror parity → `dJ = 2`), writing the 4 mirrored slots `(k,n)`, `(−k,−n)`, `(−k,n)`, `(k,−n)`, zero-padding `k ∈ [bw, slP−bw]`. Since the pattern side (`gln`) has no symmetry, only the master's flags are used (as EMSphInx).
- Inverse: `xc_full = scipy.fft.irfftn(fxc, s=(slP,slP,slP), axes=(0,1,2), norm="forward", workers=1, overwrite_x=True)`; keep `xc = xc_full[:bwP]` (contiguous copy). Optional later optimization: skip structurally-zero `m`-planes (multiples of `z_rot`) using separable `ifft` along k,n on `fxc[:, :, ::z_rot]` then `irfft` along m — EMSphInx's `SepRealFFT3D(dx=flmFold)`; keep as a follow-up, measure first.
- Peak: `np.argmax` (numba) → 3×3×3 periodic neighbourhood with glide → tri-quadratic Newton (`_interpolate_maxima`) → ZYZ; refinement: `_derivatives` (numba, `d_beta` via pre-tables) + Newton with Cholesky, step-shrink guard, 2×2/1×1 degeneracy fallbacks, `absEps = 0.01·2π/slP`, ≤ 15 iterations, fallback to interpolated peak on failure.
- Normalized (default): `rDen` volume from `compute(flm, mlm)` and `compute(flm², mlm)` per phase (Huhle et al. 2009), computed once at indexer construction; correlate → `xc *= rDen` → argmax → refine unnormalized → divide by `denominator(zyz)` (two `derivatives(der=False)` calls), exactly as EMSphInx (`sht_xcorr.hpp:1140-1172`), documenting that the chain rule through the window is not applied.

### C.4 Threading
- kikuchipy convention: dask for memory/parallelism across patterns + numba nogil kernels. `EBSD.spherical_indexing` → `_prepare_patterns`-style dask array `(n_patterns, sig_size)` chunked by `chunk_kwargs` (default `chunk_shape=64` as `_prepare_patterns_for_refinement`), `da.map_blocks(_index_chunk, patterns, indexer=..., dtype=float64, drop_axis=1, new_axis=(1,2))` returning `(n, n_best, 6)` (quaternion 4 + score + phase). Each dask thread lazily gets its own `_SphericalCorrelator` buffers (thread-local dict) so no cross-thread mutation; scipy.fft `workers=1`; numba kernels not `parallel=True` (avoids oversubscription; matches EMSphInx's one-pattern-per-thread model). `dask.config.set(scheduler="threads", num_workers=...)` documented for control. `index_patterns()` on numpy arrays runs the same chunk function serially (used by tests and by `find_pseudo_symmetry`).
- Optional inner parallelism (`numba prange` over `k` in `_fill_spectrum`) only via an explicit `parallel=True` build if profiling shows single-pattern latency matters (e.g. `get_correlation`).

### C.5 Memory estimates per worker (float64/complex128; `d(π/2)` = bw³·8 B, `fxc` = slP²·bwP·16 B, full cube slP³·8 B, refinement tables `d_beta` 2bw³·8 + `w`,`b` bw³·8 each, `rDen` per phase bwP·slP²·8 B shared read-only)

| bw | dim | 2bw−1 → slP | bwP | fxc | irfftn cube | d(π/2) | refine tables | rDen/phase (shared) | ≈ total per thread (1 phase) |
|---|---|---|---|---|---|---|---|---|---|
| 63 | 65 | 125 → 125 | 63 | 15.8 MB | 15.6 MB | 2.0 MB | 8.0 MB | 7.9 MB | ≈ 50 MB |
| 88 | 91 | 175 → 175 | 88 | 43.1 MB | 42.9 MB | 5.5 MB | 21.8 MB | 21.6 MB | ≈ 135 MB |
| 113 | 115 | 225 → 225 | 113 | 91.5 MB | 91.1 MB | 11.5 MB | 46.2 MB | 45.8 MB | ≈ 290 MB |

Sphere buffers (2·dim²·8 B ≈ 68–212 kB) and harmonics (bw²·16 B ≤ 0.2 MB) are negligible; master-pattern harmonics at bw = 384: 2.4 MB. With 8 dask threads at bw = 113: ≈ 2.3 GB — acceptable but documented; the cube copy can be avoided (in-place slice) to shave ~30 %. Expected throughput (EMSphInx CPU: 655 pat/s on 20 threads at bw 55 ≈ 33 pat/s/thread) → target ≥ 10 pat/s/thread at bw 68 in numba (nickel_ebsd_large ≈ 1 min on 8 threads).

### C.6 Euler/orientation conventions → orix
- EMSphInx internal Bunge `eu2qu` ≡ `orix.Rotation.from_euler(eu)` (P = +1, passive sample→crystal) — verified numerically identical.
- Correlator peak `zyz = (α, β, γ)` describes the rotation taking the master (crystal frame) onto the back-projected pattern (sample frame) → `bunge = (α − π/2, β, γ + π/2)`; `R_crystal→sample = Rotation.from_euler(bunge)`; kikuchipy/orix `CrystalMap.rotations` are sample→crystal → `xmap.rotations = ~R` (EMSphInx `indexer.hpp:264-269`, `northPoleQuat` = identity not ported).
- Pseudo-symmetry ops applied on the crystal side: `q_candidate = q0 * op` before refinement (EMSphInx `indexImage`), consistent with orix `Orientation` left-symmetry convention; result reduced with `Orientation(..., symmetry=phase.point_group).reduce()` only for reporting/testing (stored rotations remain unreduced like other kikuchipy indexers).
- Convention lock-in tests: (i) `_rotate_harmonics` by cubic operators leaves Ni harmonics invariant; (ii) `mp.get_patterns(R, det)` → indexing recovers `R` (< 0.5°); (iii) EMSphInx golden `.ang` (Bunge radians, TSL) → orix `Rotation.from_euler` → `angle_with` ours.

### C.7 Preprocessing / detector fidelity decisions (kikuchipy-native)
- Not ported: mosaic AHE (`ahe.hpp`), 2-D Gaussian background (`gaussian.hpp`), DCT image quality, `PatternProcessor` — users use `remove_static_background`, `remove_dynamic_background`, `adaptive_histogram_equalization`, `get_image_quality`. Documented in the tutorial and in `SphericalIndexer` Notes.
- Ported: DCT resampler (`_resample_dct`, needed for detector and master upsampling), circular/arbitrary `signal_mask` (replaces `circmask`; `Window("circular")` in the tutorial), solid-angle-weighted window normalization, `scale_factor(dim)`.
- Detector: `EBSDDetector` is the source of truth (`sample_tilt`, `tilt`, `azimuthal`, `twist` all honoured via direction cosines — a superset of EMSphInx `Geometry`); warn if `harmonics.sample_tilt != detector.sample_tilt`. Vendor PC conversions already exist (`pc_emsoft/pc_tsl/pc_oxford/pc_bruker`).

---

## D. Test strategy with real data

### D.1 kikuchipy datasets
| Level | Data | Used in |
|---|---|---|
| Unit (fast, offline) | `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)` (401², uint8, in package) | F1 grid round trip, F2 harmonics/`.sht`, F3 symmetry invariance, F4 autocorrelation peaks, F5 simulated-pattern indexing (`get_patterns`), F6 |
| Integration (fast, offline) | `kp.data.nickel_ebsd_small()` (9 × 60×60, bundled reference xmap, PC array) | F5 `EBSD.spherical_indexing` vs bundled `xmap` |
| Integration (session download 15 MB) | `kp.data.nickel_ebsd_large()` (4125 patterns, bundled xmap) | F5 statistics (weekly if slow), F8 tutorial |
| Weekly / local only | `kp.data.ebsd_master_pattern("ni")` (305 MB, cached locally in `develop`) for MC energy weighting; local hcp masters (`openECCI_RKD/data/Mg-master-17kV.h5`, `Ti-alpha-master-20kV.h5`) skipped unless present | F2, F6 exploratory |
| Golden (weekly) | EMSphInx reference outputs (§D.2) | F7 |

Synthetic tests are deterministic (`np.random.default_rng(0)`), mirroring `EMSphInx/test/sht/*.cpp`.

### D.2 EMSphInx reference outputs — how to generate (one-off, scripts kept in `specs/2026-08-31-emsphinx-reference-regression/scripts/`)
Binaries: `c:/Users/westraadt.1/Repos/EMSphInx/build/Release/{mp2sht,IndexEBSD,sht2png,MasterXcorr}.exe` (built from `master`, no CUDA keys). Note the existing `benchmarks/GPU_test_*` outputs are **not** usable (inputs missing, CPU/CUDA metric scales differ, built from `feature/GPU`).
1. Master harmonics: `mp2sht.exe "%LOCALAPPDATA%\kikuchipy\kikuchipy\Cache\develop\data\ebsd_master_pattern\ni_mc_mp_20kv.h5" ni_20kv_bw384.sht` (≈ 75 kB; MC-weighted 5–20 keV, both hemispheres, sig = 70°). Also keep `EMSphInx/data/Ni {20kV 75.7deg}.sht` as the header/CRC golden file for the `.sht` parser (F2).
2. Patterns: Python script (h5py) writes `ni_small_bruker.h5` with `/Manufacturer = "Bruker Nano"` and `/Scan 1/EBSD/Data/patterns` = `nickel_ebsd_small` after `remove_static_background(); remove_dynamic_background()` (uint8, `(9, 60, 60)`), and the same for `nickel_ebsd_large` (`(4125, 60, 60)`). (EMSphInx's HDF5 reader requires a known root `Manufacturer`; "Bruker Nano" ⇒ no vertical flip, matching kikuchipy's row-0-at-top storage and Bruker PC convention.)
3. `IndexEBSD.exe ni_small.nml` with
   ```
   &IndexEBSD
   patfile = 'ni_small_bruker.h5'
   patdset = 'Scan 1/EBSD/Data/patterns'
   masterfile = 'ni_20kv_bw384.sht'
   patdims = 60, 60
   circmask = 0
   gausbckg = .FALSE.
   nregions = 0
   delta = 480.0
   vendor = 'Bruker'
   pctr = 0.4251, 0.2134, 0.5007
   thetac = 0.0
   scandims = 3, 3, 1.5, 1.5
   roimask = ''
   bw = 68
   normed = .TRUE.
   refine = .TRUE.
   nthread = 1
   batchsize = 1
   datafile = 'ni_small_bw68.h5'
   vendorfile = 'ni_small_bw68.ang'
   ipfmap = 'ni_small_ipf.png'
   qualmap = 'ni_small_xc.png'
   /
   ```
   (`delta = 480` keeps the detector width 28.8 mm inside EMSphInx's [5, 90] mm sanity range; Bruker PC is scale-invariant. Sample tilt is taken from the `.sht` primary angle = 70°.) Repeat for `nickel_ebsd_large` (`scandims = 75, 55, 1.5, 1.5`, `bw = 53` and `68`), and a variant with `normed = .FALSE.`/`refine = .FALSE.` for testing those switches. Also run `sht2png.exe ni_20kv_bw384.sht ni_sqleg.png ni_stereo.png` (image-level check of `to_master_pattern`), and `MasterXcorr.exe 88 0.7 ni_mc_mp_20kv.h5` (stdout operator list; expected: only true symmetry ops).
4. Convert `.ang` → compact `.npz` (`phi1, Phi, phi2` [rad], `metric`, `phase`, `xstar/ystar/zstar`, EMSphInx commit hash `60f3517`, nml text) with orix/numpy (≈ 66 kB for 4125 points).
5. Hosting: `pyxem/kikuchipy-data` (new folder `emsphinx/`: `ni_20kv_bw384.sht`, `nickel_ebsd_small_bw68_ref.npz`, `nickel_ebsd_large_bw53_ref.npz`, `nickel_ebsd_large_bw68_ref.npz`), registered in `src/kikuchipy/data/_registry.py` with MD5 + commit-pinned raw URLs; fetched with `allow_download=True` inside `@pytest.mark.weekly` tests. Fallback until upstream hosting is agreed: `KIKUCHIPY_EMSPHINX_REFERENCE_DIR` env var + `pytest.skip` when absent (fork-only), or ship the 75 kB `.sht` in-package (`src/kikuchipy/data/emsphinx/`) — decision = open question 3.

### D.3 Tolerances / acceptance
- SHT round trip: max |Δa| ≤ 5e-3, mean ≤ 5e-5 (EMSphInx `test/sht/square_sht.cpp`).
- Wigner: 2·eps (values), 24·eps (derivatives), tables vs scalar ≤ 1e-13.
- Rotate-and-recover (synthetic): < 1e-4° with refinement; < 0.012° with symmetry-reduced comparison for symmetric point groups.
- Simulated patterns (`get_patterns`) → indexed: all < 0.5° (refined), < 3° (unrefined) at bw 68.
- `nickel_ebsd_small` vs bundled xmap: `np.all(angles < 2°)` (or ≥ 8/9 within 2° if flaky), median < 1°.
- `nickel_ebsd_large` vs bundled xmap: median < 1°, fraction < 2° ≥ 0.95, `mean(scores) > 0.3`.
- EMSphInx golden (same preprocessing, bw 68, normalized, refined): misorientation median < 0.2°, 95th percentile < 1°, fraction < 2° ≥ 0.98; Pearson r(scores) > 0.95 and mean relative score difference < 5 % (metric = raw normalized correlation, no extra std division, as EMSphInx); phase ids identical.
- `.sht`: byte-identical header/CRC round trip for our writer; EMSphInx-written file parses to the documented header values; unpacked coefficients from `ni_20kv_bw384.sht` vs `from_master_pattern(ebsd_master_pattern("ni"), 384)`: relative RMS difference < 2 % (after applying EMSphInx's mean-bug convention or comparing with DC removed).
- Benchmarks: record baseline patterns/s in `validation.md`; regressions > 20 % flagged.

---

## E. Risks and open questions (≤ 6)
1. **Patent/licence go-ahead:** EMSphInx's ReadMe states the central algorithm is covered by a provisional patent (CMU, 2019); copyright is fine (GPL-2.0-or-later → GPL-3.0-or-later with retained CMU/Lenthe notices), but do you want me to (a) proceed on your fork regardless and (b) draft the pyxem issue + `pyxem.team@gmail.com` e-mail asking maintainers to check the patent status before any upstream merge?
2. **Where the specs live and PR targets:** feature branches → PRs into your fork's `develop` (with `specs/` committed), and a final upstream PR to `pyxem/kikuchipy:develop` *without* `specs/` (or split into 2–3 upstream PRs as suggested in §B)? Or should each feature PR go straight to upstream?
3. **Golden data hosting:** may I plan on `pyxem/kikuchipy-data` (needs maintainer buy-in), or ship the 75 kB `.sht` + ~70 kB `.npz` in-package under `src/kikuchipy/data/emsphinx/`, or keep the EMSphInx regression fork-local behind an env var?
4. **Preprocessing fidelity:** OK to *not* port EMSphInx's mosaic AHE / Gaussian background / DCT image quality (use kikuchipy's `remove_*_background`, `adaptive_histogram_equalization`, `get_image_quality`) and to drop the square-**Lambert** SHT layout (Legendre only), or do you want bit-faithful EMSphInx preprocessing as an opt-in?
5. **Public naming/surface:** `kikuchipy.indexing.MasterPatternHarmonics` (+ `EBSDMasterPattern.get_spherical_harmonics`), `kikuchipy.indexing.SphericalIndexer` (+ `EBSDDetector.get_spherical_indexer`), `EBSD.spherical_indexing`, `kikuchipy.indexing.find_pseudo_symmetry` — acceptable, or do you prefer fewer public names (e.g. no signal/detector convenience methods) or an `EBSD.spherical_indexing(..., xmap=)` refine-only mode?
6. **Performance/compat targets:** is ≥ 10 patterns/s/thread at bw 68 (nickel_ebsd_large ≈ 1 min on 8 threads) acceptable for v1, and must the code stay compatible with kikuchipy's CI "oldest" pins (numba 0.57, numpy 1.23, Python 3.10) — this rules out some newer numba features (e.g. `np.linalg` on complex in nopython for some paths) and affects kernel design.

### Critical files for implementation
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/util/_master_pattern.py` (direction cosines, Lambert helpers, bilinear scheme to reuse)
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/ebsd.py` (indexing method conventions; where `spherical_indexing` goes)
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/__init__.pyi` and `src/kikuchipy/indexing/_refinement/_refinement.py` (public export + dask/numba execution pattern)
- `c:/Users/westraadt.1/Repos/EMSphInx/include/sht/square_sht.hpp`, `include/sht/wigner.hpp`, `include/sht/sht_xcorr.hpp` (algorithm sources of truth)
- `c:/Users/westraadt.1/Repos/EMSphInx/include/modality/ebsd/detector.hpp` and `include/idx/master.hpp` (back-projection LUT and master-pattern harmonics semantics)