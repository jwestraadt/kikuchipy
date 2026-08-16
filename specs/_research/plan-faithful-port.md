# Implementation plan: port EMSphInx spherical indexing (CPU) into kikuchipy

Perspective adopted: **faithfulness & correctness**. Every numerical component is a line-for-line port of the EMSphInx `master` HEAD `60f3517` (same odd square-Legendre grid, same Schaeffer ALF recursion, same Sneeuw ring weights, same SOFT-style SO(3) correlation, same tri-quadratic + Newton refinement, same Huhle normalisation), so results can be regression-tested against `IndexEBSD.exe`, `mp2sht.exe`, `sht2png.exe`, `MasterXcorr.exe`. Only numpy/scipy/numba/h5py/orix (already dependencies) are used. No GPU, no FFTW/pyfftw/shtns/pyshtools.

Facts verified during planning (beyond the reports): the local kikuchipy clone is on `develop` @ `49b1c11c` (fork `jwestraadt/kikuchipy`, no `upstream` remote); the working tree has a stray `IndexEBSD.nml` and modified `doc/tutorials/{hybrid_indexing,load_save_data}.ipynb` (must not be swept into feature commits); `kikuchipy.indexing.__init__.pyi` exports 9 names; `doc/tutorials/index.rst` Indexing nbgallery is at lines 36-48; `CHANGELOG.rst` has an empty `Unreleased` block; kikuchipy h5ebsd files carry a lowercase root dataset `manufacturer = "kikuchipy"` which EMSphInx's `GetVendor` (looks for `Manufacturer`) does not recognise → EMSphInx reference runs need a repacked plain HDF5 with `Manufacturer="Bruker"`; `EBSDDetector._pc_bruker2emsoft(version=4)` gives exactly EMSphInx's `cX=(x*-0.5)w, cY=(0.5-y*)h, sDst=z*·h·pX`; `sht_xcorr.hpp:549-590`, `detector.hpp:552` (`p.idx = i` south bug) and `master.hpp:565-585` (mean over `totW`, stdev over `2*totW`) match the report.

---

## A. Constitution (`specs/`)

### `specs/mission.md`
> kikuchipy gains a numerically faithful, CPU-only, pure-Python (numpy/scipy/numba) port of the EMSphInx spherical-harmonic EBSD indexing algorithm (Lenthe, Singh & De Graef, Ultramicroscopy 207 (2019) 112841) and its supporting tools (master pattern → spherical harmonics / `.sht` conversion, spherical cross-correlation with Newton refinement, pseudo-symmetry prediction, harmonic visualisation, pattern-file utilities), exposed as `EBSD.spherical_indexing()`, `EBSDMasterPattern.get_spherical_harmonics()`, `kikuchipy.indexing.SphericalMasterPattern` and friends, following kikuchipy's coding, testing, documentation and Git conventions, so that a user can index a scan from a notebook with the same grids, transforms and conventions as `IndexEBSD.exe` and obtain regression-tested equivalent orientations. Results are validated against real datasets (`nickel_ebsd_small/large`, Ni master patterns) and against outputs of the built EMSphInx executables. Everything derived from EMSphInx is GPL-2.0-or-later → GPL-3.0-or-later and stays in GPL files.

### `specs/tech-stack.md`
- **Language/runtime**: Python ≥ 3.10 (CI 3.10, 3.13, 3.14), Windows/Linux/macOS.
- **Allowed runtime deps (already in `pyproject.toml`)**: numpy, scipy (`scipy.fft` for rfft/irfft/dct/idct/`next_fast_len`; `scipy.linalg`), numba (`@njit(cache=True, nogil=True, fastmath=False for SHT/Wigner kernels; fastmath only where verified)`), dask (`map_blocks`), h5py, orix (Rotation/Symmetry/CrystalMap/Phase), matplotlib, hyperspy. **Forbidden**: pyfftw, FFTW, shtns, pyshtools, rocket-fft, CUDA/pyopencl. New optional deps only if justified in a spec.
- **Placement/licensing**: core in `src/kikuchipy/_sht/` (private, GPLv3+; every file carries kikuchipy header + `Derived from EMSphInx, Copyright (c) 2019 De Graef Group, Carnegie Mellon University, author William C. Lenthe, GPL-2.0-or-later, translated to Python by <you>, 2026-08` notice per GPLv2 §2a). Never import into BSD-3 areas (`doc/dev/licensing_considerations.rst` on upstream develop).
- **Style**: PEP 8/Black via ruff + ruff-format pre-commit; numpydoc; type hints in signatures only; comment/docstring lines ≤ 72 chars; 3 import blocks with `force-sort-within-sections`; lazy `__init__.py` + `.pyi` stub; private modules `_`-prefixed; `# pragma: no cover` for optional branches; `print()` for progress; `ValueError` messages with given/expected values.
- **Numba**: `@njit(cache=True, nogil=True)` (+`fastmath=True` only for pure accumulate loops); explicit signatures where stable; every kernel tested via `f(...)` **and** `f.py_func(...)`.
- **Parallelism**: dask threaded scheduler over pattern chunks (`da.map_blocks`), numba `nogil=True`, scipy.fft `workers=1` inside workers; no `multiprocessing`.
- **Testing**: pytest, `tests/test_indexing/…`, `tests/test_io/…`, `tests/test_signals/…`; real data first (`kp.data.nickel_ebsd_small()`, `nickel_ebsd_master_pattern_small()`, session-downloaded `nickel_ebsd_large()`), synthetic seeded tests for math; `@pytest.mark.weekly` for heavy regressions; benchmarks in `benchmarks/indexing/`.
- **Commands**: `pre-commit run --all-files`; `pytest --cov -n 4`; `pytest tests/test_indexing/test_spherical* -x`; `pytest --doctest-modules src/kikuchipy/_sht src/kikuchipy/indexing`; `pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py`; `pytest --weekly -k spherical`; `cd doc && make html`; `./doc/tutorials/run_nbval.sh` (after adding the notebook).
- **Docs**: numpydoc + `:cite:` keys added to `doc/user/bibliography.bib` (`lenthe2019spherical`, `lenthe2019pseudo`, `schaeffer2013efficient`, `gutman2008shape`, `huhle2009normalized`, `fukushima2016numerical`, `sneeuw1994global`, `rosca2010new`); tutorial `doc/tutorials/spherical_indexing.ipynb` (black@77, first hidden cell, thumbnail tag, no stored outputs); API via `__init__.pyi` `__all__`; `CHANGELOG.rst` Unreleased/Added.
- **Git**: branch off `develop` (kebab-case), `git commit -s`, merge (not rebase) develop, PR → `develop` with the repo PR template; one feature per PR; specs in `specs/<date>-<name>/{plan.md,requirements.md,validation.md,tasks.md}`; adversarial review pass before commit.
- **Conventions**: SH coefficients `alm[m, l]` (m-major, `l<m` zero, `m≥0` only, complex128); grid `(2, dim, dim)` = (north, south) row-major, `dim` odd; Euler ZYZ internal, orix `Rotation` at the API boundary; correlation output `xc[k, n, m]` = (β, γ, α); PC via `EBSDDetector` (Bruker internal).

### `specs/roadmap.md` (ordered phases = features/branches; each is a PR)
- [ ] **P0 constitution & legal scaffolding** (`spherical-indexing-constitution`) — specs, license notices, bibliography, related_projects, upstream issue draft.
- [ ] **P1 SHT core** (`sht-square-grid-transform`) — square Lambert/Legendre grids, ring tables, Sneeuw weights, `DiscreteSHT.analyze/synthesize`, Wigner d tables + `rotate_harmonics`.
- [ ] **P2 master spectra & `.sht` I/O** (`sht-master-spectra-file`) — `SphericalMasterPattern`, `EBSDMasterPattern.get_spherical_harmonics`, `.sht` reader/writer (mp2sht), Legendre regrid, symmetry LUTs.
- [ ] **P3 spherical cross-correlation** (`spherical-cross-correlation`) — SO(3) correlator, peak interpolation, Newton refinement, normalized correlator.
- [ ] **P4 back-projection & pattern preprocessing** (`spherical-back-projection`) — detector→sphere LUT, DCT rescaler + IQ, Gaussian background, mosaic AHE, unproject.
- [ ] **P5 `EBSD.spherical_indexing`** (`spherical-indexing-ebsd`) — indexer glue, dask/numba parallelism, CrystalMap output, refine-only path, pseudo-symmetry ops, multi-phase.
- [ ] **P6 pseudo-symmetry prediction** (`spherical-pseudo-symmetry`) — MasterXcorr equivalent + volume/plot.
- [ ] **P7 visualisation & utilities** (`sht-visualisation-utilities`) — sht2png equivalent (`to_master_pattern`, `plot`), PatternRepack/EBSPDims equivalents, EMSphInx namelist reader/writer, `.ang` export note.
- [ ] **P8 EMSphInx regression suite** (`spherical-indexing-emsphinx-regression`) — reference data generation scripts, in-package/cached reference files, weekly tests.
- [ ] **P9 tutorial, docs, benchmark, changelog** (`spherical-indexing-tutorial`) — notebook, index.rst, run_nbval, benchmark, changelog, credits.

Dependency chain: P1 → P2 → P3 → P4 → P5 → (P6, P7, P8) → P9. P6/P7/P8 can be developed in parallel after P5.

---

## B. Features (one branch + spec folder each)

Common per-feature workflow: `git switch -c <branch> develop` → write `specs/<date>-<name>/{plan.md,requirements.md,validation.md,tasks.md}` → approval → tests first (real data where possible) → implementation → `/code-review` adversarial pass (numerics, conventions, coverage, licensing) → fix → `git commit -s` → PR into fork `develop` (later a consolidated PR to `pyxem/kikuchipy:develop`).

### P0 — `spherical-indexing-constitution` — `specs/2026-08-16-constitution/`
Files (new): `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `specs/2026-08-16-constitution/{plan,requirements,validation,tasks}.md`; `doc/user/bibliography.bib` (+8 entries); `doc/user/related_projects.rst` (+EMSphInx entry with DOI + licence note); `.license.tmpl` unchanged; new `doc/dev/porting_emsphinx.rst`? — no: instead a `NOTICE`-style paragraph in `src/kikuchipy/_sht/__init__.py` docstring. Also: `git remote add upstream https://github.com/pyxem/kikuchipy.git && git fetch upstream && git merge upstream/develop` (picks up `doc/dev/licensing_considerations.rst`); delete stray `IndexEBSD.nml`; do not commit the modified tutorials.
Acceptance: files exist, `make html` builds bibliography without warnings, patent/legal note recorded in `specs/…/requirements.md` (question E1). Validation: `cd doc && make html`.

### P1 — `sht-square-grid-transform` — `specs/2026-08-16-sht-core/`
**New files**
- `src/kikuchipy/_sht/__init__.py` (docstring: private, EMSphInx provenance, GPL note; no lazy stub) and `_sht/__init__.pyi` (optional; mirror `_utils`).
- `src/kikuchipy/_sht/_square_grid.py`
- `src/kikuchipy/_sht/_discrete_sht.py`
- `src/kikuchipy/_sht/_wigner.py`
- `tests/test_utils/test_sht_square_grid.py`, `tests/test_utils/test_sht_discrete_sht.py`, `tests/test_utils/test_sht_wigner.py` (or a new `tests/test_sht/` package with `__init__.py`; choose `tests/test_sht/` — one test package per source package, like `tests/test_utils`).

**Signatures (Python; all `Real = float64`)**
```python
# _square_grid.py
class Layout(str, Enum): LAMBERT = "lambert"; LEGENDRE = "legendre"
def square_to_sphere(x: np.ndarray, y: np.ndarray) -> np.ndarray            # (n,3), X,Y in [0,1]  (:614-642)
def sphere_to_square(v: np.ndarray) -> np.ndarray                           # (n,2), uses |z|       (:591-606)
def lambert_cos_latitudes(dim: int) -> np.ndarray                           # (Nt,)  1-(2y/(dim-1))^2
def legendre_cos_latitudes(dim: int) -> np.ndarray                          # [1] + positive roots of P_{dim-2} desc.
def legendre_roots_positive(n: int) -> np.ndarray                           # numpy leggauss, sorted desc, x[m1]=0 exact
def grid_normals(dim: int, layout: Layout) -> np.ndarray                    # (dim*dim,3) north hemisphere (:665, :823-869)
def ring_number(dim: int) -> np.ndarray                                     # (dim*dim,) int  (:1144)
def ring_indices(dim: int) -> tuple[np.ndarray, np.ndarray]                 # flat idx (concat, CCW from phi=0) + offsets (Nt+1,)  (readRing :942-1014)
def ring_solid_angles(dim: int, layout: Layout) -> np.ndarray               # (Nt,) (:1105-1138)
def lambert_pixel_solid_angles(dim: int) -> np.ndarray                      # (dim*dim,) Mazonka (:681-736)
def quadrature_weights(dim: int, layout: Layout, max_l: int) -> np.ndarray  # (Nw, Nt) computeWeightsSkip incl. 4π/Nφ scaling (:1022-1063)
def legendre_bounding_indices(dim, cos_lats, v) -> np.ndarray               # (n,4)  (:876-933)
def bandwidth_to_dim(bw: int, layout=Layout.LEGENDRE) -> int                # bw + (3 if bw%2==0 else 2) ; Lambert: 2*bw+1
def check_dim(dim: int) -> None                                             # odd, >=3

# _discrete_sht.py
class DiscreteSHT:
    def __init__(self, dim: int, bandwidth: int, layout: Layout = Layout.LEGENDRE) -> None
    @classmethod legendre(cls, dim) / lambert(cls, dim)
    dim, bandwidth, layout, n_rings, cos_lats, weights, amn, bmn, ring_offsets, ring_index, dft_re, dft_im (per-ring DFT matrices, optional)
    def analyze(self, sphere: np.ndarray, bandwidth: int | None = None) -> np.ndarray   # (2,dim,dim)->(bw,bw) complex128, alm[m,l]
    def synthesize(self, alm: np.ndarray) -> np.ndarray                                  # (bw,bw)->(2,dim,dim)
def _alf_coefficients(bw: int) -> tuple[np.ndarray, np.ndarray]     # amn,bmn (bw,bw) Schaeffer eqs 16-18 (:347-373)
@njit _analyze_kernel(north, south, ring_index, ring_offsets, weights, cos_lats, amn, bmn, bw, out_re, out_im)   # (:414-486)
@njit _synthesize_kernel(alm_re, alm_im, ..., north, south)                                                        # (:495-572)

# _wigner.py
def wigner_d(j, k, m, t, negative_beta=False) -> float                       # (:298-371) symmetry reductions + Fukushima recursion
def wigner_d_half_pi(j, k, m) -> float; def wigner_d_sign(j,k,m) -> int
def wigner_D(j, k, m, zyz) -> complex
def wigner_d_table_half_pi(bw: int, transpose: bool) -> np.ndarray          # (bw,bw,bw) (:699-761)
def wigner_d_table(bw: int, t: float, negative_beta: bool) -> np.ndarray    # (bw,bw,bw,2)  (:452-559)
def wigner_d_table_pre_build(bw) -> tuple[E (bw,bw), W (bw,bw,bw), B (bw,bw,bw)]   # (:575-691)
@njit wigner_d_table_pre(bw, t, nB, E, W, B, out)                            # per-call table used by refinement
def wigner_d_prime(j,k,m,t,nB) / wigner_d_prime2(...)                        # (:814-852)
def rotate_harmonics(alm: np.ndarray, zyz: np.ndarray) -> np.ndarray         # (:769-799) blm[m,j] = Σ_n alm[n,j] D^j_{m,n}
```
Implementation notes (faithfulness): rings CCW from φ=0 with the same `pole/start/quad1..4` index arithmetic; weights `w_y = 4π·ŵ_y/max(1,8y)` with the `skip` sets `Nw=(dim-2)//4+1` (Legendre: skip 0 copied); Condon–Shortley `(-1)^m` applied at ring weight; NumPy `irfft` output multiplied by `N_φ` (FFTW c2r unnormalised) — or, preferred, use precomputed per-ring DFT matrices inside the numba kernel (`Σ_y 8y(4y+1)` complex ≈ 0.46 M for dim=71, 78 M for dim=387; the latter only for the one-off master-pattern transform, where scipy `rfft` in a Python ring loop is used instead when `n_rings > 80`). Ints for `w_jkm` are Python/`int64`. Table slots with `j < max(k,m)` are never read (assert in tests via NaN fill in debug mode).

**Reused**: `kikuchipy.signals.util._master_pattern._lambert2vector/_vector2lambert` (equality test only — affine map `X_EMSphInx = (L/√(π/2)+1)/2`), `scipy.fft`, `numpy.polynomial.legendre.leggauss`.

**Tests** (synthetic, deterministic `np.random.default_rng(0)`; each numba kernel via `.py_func` too):
- `square_to_sphere∘sphere_to_square` identity (1e-12); equals kikuchipy Lambert affinely (1e-10); Legendre normals: `z` equals `leggauss(dim-2)` roots, unit norm; `ring_indices` cover each pixel once, length `8y`, first slot at φ=0, CCW; `ring_solid_angles`/`lambert_pixel_solid_angles` sum to `2·dim²−4(dim−1)` (avg 1) within 1e-9; `quadrature_weights`: `Σ ŵ = 1` before scaling, `w0 = 4π`.
- `analyze(ones) → a00 = √(4π)`, others < 1e-12 (Legendre & Lambert, dim ∈ {5,7,15,65,71,91,115}); single `Y_l^m` synthesized from `scipy.special.sph_harm_y` (accounting for the no-CS convention) → coefficient 1±1e-8 for Legendre bw ≤ dim-2.
- **EMSphInx round-trip test** (`test/sht/square_sht.cpp`): random spectrum, `synthesize→analyze`, max |Δ| ≤ 5e-3, mean ≤ 5e-5, Legendre bw ∈ {4,…,128 step, 384 (weekly)}, Lambert bw ∈ {4,…,64}.
- Wigner: `d(j,k,m,t)` vs `sympy.physics.wigner.wigner_d_small` (j ≤ 8, β ∈ {π/3, π/2, 2π/3, −π/3}) to 1e-12; tables vs scalar (bw=15, β=0.9708055194) to 2·eps·j; derivatives vs finite differences (1e-6) and vs closed forms for j ≤ 4; `rotate_harmonics(rotate_harmonics(a, r), r⁻¹) ≈ a`; `rotate_harmonics` vs brute force `Σ Y_lm(R v)`.
- Real data: `analyze` of `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)` regridded (via P2 helper — so this test moves to P2) — P1 real-data test: SHT of the Ni MP Lambert grid directly on the Lambert layout at dim=401 (bw ≤ 200): synthesize→compare with the master pattern (rel. RMS < 5% at bw=200; visual sanity in tutorial).
Acceptance: all above; `pytest tests/test_sht -n 4` < 60 s (excluding weekly bw=384).
CHANGELOG (Added): "Private spherical harmonic transform on the square Lambert/Legendre grids and Wigner-d utilities ported from EMSphInx (`kikuchipy._sht`)."

### P2 — `sht-master-spectra-file` — `specs/2026-08-17-sht-master-spectra-file/`
**New/changed files**
- `src/kikuchipy/_sht/_symmetry.py` — 32-row table keyed on orix `Symmetry.name`: `z_rotation_order` (fNf), `has_z_mirror` (fMr), `mm_type`, `tsl_number`, `hkl_number`, `iucr_number`; 230-entry `SPACE_GROUP_ROTATION`, `SPACE_GROUP_COMPRESSION` LUTs (`sht_file.in.hpp:1838-1869`); `def symmetry_flags(point_group: Symmetry) -> tuple[int, bool]`.
- `src/kikuchipy/_sht/_master_grid.py` — `to_legendre(nh, sh, dim_new) -> (2,dim,dim)` (`master.hpp:381-416`: √2 DCT oversample then bilinear), `to_lambert(...)` (nearest of 4 bounding), `make_n_fold(sphere, n, m)`, `make_z_mirror`, `make_inversion`, `match_equator`, `dct_rescale(im, w_out, h_out, zero_mean=False, high_pass=0, return_iq=False)` (`image::Rescaler`, scipy `dct` type 2/3, unnormalised = FFTW REDFT10/01), `image_quality_dct(dct, w, h)`; `weighted_normalize(nh, sh, dim, emsphinx_compatible=True)` (reproduces the `totW` mean and quartered corners when `emsphinx_compatible`).
- `src/kikuchipy/indexing/_spherical_master_pattern.py` — public class.
- `src/kikuchipy/io/plugins/emsphinx_sht/{__init__.py, _api.py, specification.yaml}` — `file_reader(filename, npx=None, projection="lambert", hemisphere="both", lazy=False) -> list[dict]` (returns an `EBSDMasterPattern` dict with harmonics in `original_metadata["sht"]`), `file_writer` (from an `EBSDMasterPattern` — computes harmonics), plus low-level `read_sht(path) -> SHTFile`, `write_sht(path, ...)`, dataclasses `SHTHeader, SHTCrystal, SHTAtom, SHTSimulation, SHTHarmonics`, `crc32c_emsphinx(bytes) -> int` (LUT copied verbatim, `sht_file.in.hpp:947-1005`), `pack_harmonics/unpack_harmonics(alm, bw, z_rot, flags)` (`:1706-1831`).
- `src/kikuchipy/signals/ebsd_master_pattern.py` — new method `get_spherical_harmonics`.
- `src/kikuchipy/io/plugins/_emsoft_master_pattern.py` — expose `EMData/MCOpenCL/accum_e`-derived energy weights in `original_metadata["energy_weights"]` (backwards compatible; small change) — needed for a faithful `mp2sht`.
- `src/kikuchipy/indexing/__init__.pyi` (+`SphericalMasterPattern`), `src/kikuchipy/io/plugins/__init__.py(i)` docstring/autosummary (+`emsphinx_sht`), `doc/user/installation.rst` untouched.
- `src/kikuchipy/data/emsphinx_sht/ni_20kv_bw384.sht` (≈75 kB, generated by `mp2sht.exe` from cached `ni_mc_mp_20kv.h5`, see D) + `_registry.py` md5 + `data/_data.py: nickel_sht_master_pattern()` + `data/__init__.pyi`.
- Tests: `tests/test_io/test_emsphinx_sht.py`, `tests/test_indexing/test_spherical_master_pattern.py`, `tests/test_sht/test_master_grid.py`, `tests/test_data/test_data.py` (+1).

**Public API**
```python
class SphericalMasterPattern:
    """Spherical harmonic representation of a master pattern (EMSphInx MasterSpectra)."""
    def __init__(self, coefficients: np.ndarray, phase: Phase, sample_tilt: float, energy: float,
                 layout_source: str = "legendre", metadata: dict | None = None) -> None
    coefficients: np.ndarray            # (bw, bw) complex128, alm[m, l]
    bandwidth: int; phase: Phase; point_group: Symmetry; sample_tilt: float; energy: float
    n_fold: int; has_equatorial_mirror: bool; has_inversion: bool
    @classmethod from_master_pattern(cls, master_pattern: EBSDMasterPattern, bandwidth: int = 384,
                                     energy: int | float | None = None, energy_weights: np.ndarray | None = None,
                                     normalize: bool = True) -> SphericalMasterPattern
    @classmethod from_file(cls, filename) -> SphericalMasterPattern            # .sht (v1.1)
    def save(self, filename, notes: str = "created with kikuchipy", doi: str = ..., overwrite=None) -> None
    def resize(self, bandwidth: int) -> SphericalMasterPattern              # zero-pad/crop (master.hpp:601-614)
    def remove_dc(self) -> SphericalMasterPattern
    def rotate(self, rotation: Rotation) -> SphericalMasterPattern          # via rotate_harmonics(zyz)
    def to_master_pattern(self, npx: int | None = None, projection: str = "lambert",
                          hemisphere: str = "both", dtype: str = "float32") -> EBSDMasterPattern   # sht2png-equivalent core (P7 adds plot)
    def __repr__

EBSDMasterPattern.get_spherical_harmonics(self, bandwidth: int = 384, energy=None, energy_weights=None,
                                          normalize: bool = True) -> SphericalMasterPattern
kp.load("x.sht", npx=..., projection=..., hemisphere=...) -> EBSDMasterPattern   # via plugin
```
Behaviour: `from_master_pattern` requires `projection="lambert"`, `hemisphere="both"` (raise `ValueError` otherwise, listing the given values); sums atoms (already done by reader), energy-weighted mean with `energy_weights` (default: `original_metadata["energy_weights"]` if present, else uniform → warning), `dim_lg = bw + 2 + (bw%2==0)`, `to_legendre(dim_lg)`, `weighted_normalize`, `DiscreteSHT.legendre(dim).analyze(...)`. `.sht` writer follows `initFileEMsoft/addDataEMsoft` byte-for-byte (24ths encoding incl. 1/6,1/3,2/3,5/6 special cases, pad-8 strings, CRC-32C variant, LE only).

**Reused**: `kp.load` EMsoft MP reader (`io/plugins/emsoft_ebsd_master_pattern`), `orix.crystal_map.Phase`, `orix.quaternion.symmetry.get_point_group`, `EBSDMasterPattern` constructor + `_utils/vector.py` hemisphere parsing.

**Tests**
- `.sht` reader on `c:/Users/westraadt.1/Repos/EMSphInx/data/Ni {20kV 75.7deg}.sht` (copied to `tests/…` fixtures? — no, ship the freshly generated `ni_20kv_bw384.sht` in-package and additionally read the EMSphInx one if present via env var `EMSPHINX_ROOT`): every field of the §8 dump equal (bw 384, zRot 4, cmpFlg 7, doubCnt 9312, CRC 0xf2af93ef, sg 225, a=0.35236, atom Ni DW 0.0035, sigStart 75.7, latGridType 1); read→write→read byte-identical; CRC verify on corrupted byte raises.
- mp2sht parity (real data): `SphericalMasterPattern.from_master_pattern(kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both"))` vs `ni_20kv_bw384.sht` from `mp2sht.exe`: relative L2 error of `alm` ≤ 1e-4 and max |Δ|/max|alm| ≤ 1e-3 (weekly, 305 MB cached file); fast test with `nickel_ebsd_master_pattern_small` (in-package, uint8, 1 energy) checks shape/flags/`a00 ≈ 0`-ness (with `emsphinx_compatible=False`), `n_fold == 4`, `has_equatorial_mirror`, systematic zeros (`alm[m % 4 != 0] == 0` within 1e-10 after enforcing? — no: verify they are ≤ 1e-6·max due to symmetric input).
- `to_master_pattern` of the .sht → compare (2,dim,dim) synthesis with `sht2png.exe` PNGs (P7 test) and with the Lambert Ni MP small (correlation > 0.99).
- symmetry tables: for all 32 orix groups, `n_fold` = highest z-axis order (`Symmetry.get_axis_orders`) and `has_z_mirror` = mirror normal ∥ z from improper ops — cross-check table against orix computationally.
Acceptance: above tolerances; `kp.load(".sht")` returns `EBSDMasterPattern` with `phase.point_group.name == "m-3m"`.
CHANGELOG (Added): "`SphericalMasterPattern` and `EBSDMasterPattern.get_spherical_harmonics()`; read/write of EMSphInx/SHTdatabase `.sht` files (`kikuchipy.io.plugins.emsphinx_sht`), including a `kp.data.nickel_sht_master_pattern()` dataset."

### P3 — `spherical-cross-correlation` — `specs/2026-08-18-spherical-cross-correlation/`
**Files**: `src/kikuchipy/_sht/_xcorr.py`, `src/kikuchipy/_sht/_euler.py` (zyz↔quaternion/Bunge/orix), tests `tests/test_sht/test_xcorr.py`, `tests/test_sht/test_euler.py`.
```python
# _euler.py
def zyz_to_bunge(zyz) -> np.ndarray   # (α−π/2, β, γ+π/2)
def bunge_to_zyz(eu) -> np.ndarray
def zyz_to_quaternion(zyz) -> np.ndarray   # rotations.hpp:973-989 (w≥0, orientAxis)
def quaternion_to_zyz(qu) -> np.ndarray    # wrapped [0,2π)
def rotation_from_zyz(zyz, conjugate=True) -> Rotation   # ~Rotation.from_euler(zyz_to_bunge(zyz))
def zyz_from_rotation(r: Rotation) -> np.ndarray

# _xcorr.py
def fast_size(n: int) -> int   # max(1,n) if n<=16 else smallest {2,3,5,7,11,13}-smooth ≥ n (fft.hpp:438-491)  (== scipy next_fast_len for these primes? verify; implement own to be exact)
class SphericalCorrelator:
    def __init__(self, bandwidth: int) -> None       # sl=2bw−1, slp=fast_size(sl), bwp=slp//2+1; tables wig_d_half_pi (transposed), E,W,B; buffers
    def compute(self, flm, gln, mirror: bool, n_fold: int) -> np.ndarray     # xc (bwp, slp, slp) real, layout [k,n,m]
    def find_peak(self, xc) -> int
    def index_to_euler(self, idx) -> np.ndarray / euler_to_index(zyz) -> int
    def extract_neighborhood(self, xc, idx, half=1) -> np.ndarray   # (3,3,3) periodic + glide
    def interpolate_peak(self, xc, idx) -> tuple[float, np.ndarray]  # tri-quadratic Newton (:1261-1366), incl. faithful x[2] bug behind flag
    def derivatives(self, flm, gln, zyz, mirror, n_fold, need_derivatives=True) -> tuple[float, jac(3), hes(3,3)]
    def refine_peak(self, flm, gln, mirror, n_fold, zyz, eps=0.01) -> tuple[float, np.ndarray]   # Newton (:442-499)
    def correlate(self, flm, gln, mirror, n_fold, refine=True, eps=0.01) -> tuple[float, np.ndarray]
    def to_bunge_cube(self, xc) -> np.ndarray   # extractBunge (:595-649), for pseudo-symmetry/plots
class UnnormalizedPhaseCorrelator(flm, mirror, n_fold, correlator)      # stores flm; .correlate(gln, refine) ; .refine(gln, zyz)
class NormalizedPhaseCorrelator(flm, flm2, mirror, n_fold, mlm, correlator)   # rDen (Huhle eq 8/9), .correlate divides, .refine → unnormalized refine / denominator(zyz)
@njit _xcorr_spectrum_kernel(flm_re, flm_im, gln_re, gln_im, wig_d, bw, slp, bwp, mirror, n_fold, fxc_re, fxc_im)   # (:657-858) k,n,m,j loops incl. 4 mirror slots and systemic-zero masks
@njit _derivatives_kernel(...)   # (:889-1119)
@njit _interpolate_maxima(p27, x) -> float
```
FFT plan (scipy.fft, exact EMSphInx normalisation): `tmp = scipy.fft.ifft2(fxc[:, :, ::n_fold] planes only… ` — implemented as `tmp = ifft2(fxc, axes=(0,1), workers=1) * slp*slp` on the non-zero m-planes (fill others 0), then `xc = irfft(tmp[:bwp], n=slp, axis=2) * slp` → `(bwp, slp, slp)`. This reproduces `SepRealFFT3D::inverse` (only `bwp` β-slices materialised, no `slp³` cube). Peak search: `np.argmax` on `xc*rDen` in a single pass (normalized) or `xc`.
Refinement: `scipy.linalg.cho_factor/cho_solve` for the 3×3 Cholesky (raise → fallback 2×2/1×1 branches exactly as `:463-489`), `maxIter=15`, `absEps = eps·2π/slp`, monotone step guard, on failure return interpolated orientation with re-evaluated correlation.
**Tests** (port of `test/sht/sht_xcorr.cpp`): `random_sphere(dim, mirror, n_fold)` (Legendre layout, `make_n_fold`, `match_equator`), `random_pair(bw, mirror, n_fold)` with `rotate_harmonics`; `correlate(..., refine=True)` recovers rotation to ≤ 0.01° for bw ∈ {53,68,88,113,123,158, 54..64 (padded)} (EMSphInx eps ≈ 4.9e-3°; use 1e-2° to allow numba/scipy rounding), disorientation-aware for point groups {112, 11m, 112/m, 3, 4, 4/m, 6, 6/m} bw 53..63 (≤ 0.012° or disorientation via `orix Misorientation.reduce`); normalized variant with the wedge mask (≤ 10× eps); naive triple-loop `Σ_j f g* d d` reference for bw=8 vs `compute` (1e-10) — verifies the kernel incl. mirror slots; `index_to_euler∘euler_to_index` identity; `interpolate_peak` on an analytic quadratic; `derivatives` jac/hes vs finite differences of `derivatives(..., need_derivatives=False)` (1e-5 rel).
**Convention pin (real data)**: `mp = nickel_ebsd_master_pattern_small(lambert, both, 20)`, `smp = mp.get_spherical_harmonics(bandwidth=68)`; take `r = Rotation.from_euler([[30,45,60]], degrees=True)`; `pat = mp.get_patterns(r, det, energy=20)` (kikuchipy forward projection); back-project via P4 → correlate → `rotation_from_zyz(zyz)`; assert `angle_with(r) < 0.5°`. This test lands in P4/P5 but the mapping functions are here.
Acceptance: all; `SphericalCorrelator(68).correlate` ≤ 40 ms/pattern single thread (target; measured, not gating).
CHANGELOG: "Spherical (SO(3)) cross-correlation with Newton refinement (private `kikuchipy._sht`)."

### P4 — `spherical-back-projection` — `specs/2026-08-19-spherical-back-projection/`
**Files**: `src/kikuchipy/_sht/_back_projection.py`, `src/kikuchipy/_sht/_preprocess.py`, `src/kikuchipy/detectors/_ebsd_detector.py` (+`solid_angle_fraction()`, `_emsphinx_geometry()` helpers, small), tests `tests/test_sht/test_back_projection.py`, `tests/test_sht/test_preprocess.py`, `tests/test_detectors/test_ebsd_detector.py` (+).
```python
class SphericalBackProjector:
    def __init__(self, detector: EBSDDetector, dim: int, oversample: float = np.sqrt(2),
                 signal_mask: np.ndarray | None = None, circular_mask: bool = False,
                 flip: bool = False, north_pole_rotation: Rotation | None = None) -> None
    shape_rescaled: tuple[int,int]; indices: np.ndarray (n,4) int64; weights (n,4) float64; sphere_index (n,) int64
    omega (n,) ; omega_window: float; omega_sphere: float
    def unproject(self, pattern: np.ndarray, sphere: np.ndarray, return_iq: bool = False) -> float | tuple[float, float]  # detector.hpp:589-623
    def mask(self, sphere) -> None
    def geometry_dict(self) -> dict     # EMSphInx Geometry fields (for nml export/tests)
@njit _unproject_kernel(pattern_rescaled, indices, weights, sphere_index, omega, omega_window, sphere_out) -> float
def scale_factor(detector, dim) -> float   # detector.hpp:465-469 via solid_angle(501)
def interpolate_pixel(...)                  # direction→(X,Y)+bilinear (detector.hpp:334-373), for LUT build (vectorised numpy)
# _preprocess.py
def gaussian_background_fit(pattern, mask) -> np.ndarray  (BckgSub2D, gaussian.hpp; incl. off-by-one behind emsphinx_compatible)
def gaussian_background_subtract(pattern, mask) -> np.ndarray
def mosaic_ahe(image_u8, n_regions, mask=None) -> np.ndarray   # ahe.hpp (hWdth=0.5), numba
def emsphinx_pattern_processor(pattern, circ_radius=-1, gaussian_background=False, n_regions=10) -> np.ndarray  # imprc.hpp order
def dct_rescale(...) (from P2 _master_grid, re-exported), image_quality_dct(...)
```
Geometry: reuse `_get_direction_cosines_from_detector` semantics? — No: EMSphInx's LUT direction is *sphere→detector* (`interpolatePixel`), so build the LUT from `grid_normals(dim, LEGENDRE)` (+south with `z→−z`, equator ring skipped, **correct** `sphere_index = dim*dim + i` for south, unlike EMSphInx's latent bug — documented, harmless for standard geometry) using EMSphInx's `alpha = 90 − sample_tilt + tilt` formula expressed with kikuchipy's detector: `sample_tilt`, `tilt`, `pc` (Bruker) → `cX=(pcx−0.5)·ncols, cY=(0.5−pcy)·nrows, sDst=pcz·nrows·px_size_binned`, `w=ncols,h=nrows,pX=pY=px_size_binned`. Raise `ValueError` if `azimuthal != 0` or `twist != 0` ("omega tilt not yet supported"). Cross-check in tests that the LUT directions equal `_get_direction_cosines_from_detector(det)` at pixel centres (report §4.2 verified equality) — this ties both codebases together. `flip=False` for kikuchipy-loaded patterns (image convention handled by using rows top-down consistently with Bruker `cY` sign; verified by the forward-projection convention test).
Rescale: `dct_rescale(pattern, w_out, h_out, zero_mean=True, return_iq=True)` with `sclr = Rescaler(w, h, scale_factor(dim)*√2)`; `geometry.rescale(wOut,hOut)`.
**Tests**: real data `nickel_ebsd_small` pattern (0,0) + its detector: LUT covers points only in the north hemisphere for `sample_tilt=70, tilt=0` (α=20°), footprint solid angle fraction ≈ `solid_angle(501)`; `unproject` output has weighted mean 0/std 1 on the window (1e-9); returns `sqrt(ΩW/ΩS·4π)`; zero-std pattern writes ones (mask); convention test (P3) — pattern from `mp.get_patterns(r)` back-projected then SHT-correlated with `smp` recovers `r` (< 0.5° at bw=68, < 0.25° at bw=88); `mosaic_ahe` equals a numpy reference on a 16×16 toy and equals `skimage.exposure.equalize_adapthist(clip_limit=0)`-shape only loosely (documented difference); Gaussian fit recovers synthetic `c·exp(−(x−a)²/b)` background (a within 1 px, b 1%); `image_quality_dct` in [0,1] and monotone with added noise; `dct_rescale` of a constant is constant, of a 2× upsample→downsample round trip ≈ identity (1e-6).
CHANGELOG: "Back-projection of EBSD patterns onto the sphere and EMSphInx-style preprocessing (private)."

### P5 — `spherical-indexing-ebsd` — `specs/2026-08-20-spherical-indexing-ebsd/`
**Files**: `src/kikuchipy/indexing/_spherical_indexing.py` (indexer, chunk functions, info message, `_SphericalIndexingSetup`), `src/kikuchipy/signals/ebsd.py` (+`spherical_indexing`, `refine_orientation_spherical`), `src/kikuchipy/indexing/__init__.pyi` (+`SphericalIndexingResult`? — no; only if needed), `tests/test_indexing/test_spherical_indexing.py`, `tests/test_signals/test_ebsd_spherical_indexing.py`, `benchmarks/indexing/test_spherical_indexing.py`.
```python
def EBSD.spherical_indexing(
    self,
    master_patterns: SphericalMasterPattern | Sequence[SphericalMasterPattern],
    detector: EBSDDetector,
    bandwidth: int = 68,
    normalize: bool = True,
    refine: bool = True,
    keep_n: int = 1,
    pseudo_symmetry_ops: Rotation | None = None,      # single phase only (as EMSphInx)
    navigation_mask: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,             # detector pixels to ignore (circular mask etc.)
    preprocess: bool | dict = False,                   # False: patterns used as-is (kikuchipy style); dict → emsphinx_pattern_processor kwargs
    chunksize: int | None = None,                      # patterns per dask block (default ~ BatchEstimate)
    n_workers: int | None = None,                      # dask threads (default: dask config)
    return_image_quality: bool = True,
    dtype: str = "float64",
) -> CrystalMap
def EBSD.refine_orientation_spherical(self, xmap: CrystalMap, detector, master_patterns, bandwidth=68,
                                      normalize=True, navigation_mask=None, signal_mask=None, preprocess=False,
                                      chunksize=None) -> CrystalMap
# private
def _spherical_indexing(patterns: da.Array, nav_shape, step_sizes, master_patterns, detector, bandwidth, ...) -> CrystalMap
class _SphericalIndexingSetup: dim, sht, back_projector, correlators (per phase), sphere buffer, chunk_func, map_blocks_kwargs, get_info_message()
def _index_chunk(patterns: np.ndarray (n, sig), setup...) -> np.ndarray (n, keep_n, 7)  # [phase, corr, iq, qw, qx, qy, qz]
def _spherical_indexing_info_message(...) -> str
def _batch_estimate(bandwidth, n_threads, n_patterns) -> int    # indexer.hpp:189-205
```
Semantics (faithful to `Indexer::indexImage`, `idx.hpp`): per pattern → optional preprocess → `unproject` (mean/std normalised on window, IQ) → `analyze(bw)` → for each phase `correlate` (normalized or not) → pseudo-symmetry re-refinement (`q0 * q_ps` crystal-frame first) → keep top-`keep_n` by `corr` → `rotation_from_zyz` (conjugate) → `CrystalMap(rotations, phase_id, phase_list=PhaseList([smp.phase …]), prop={"scores": corr (n, keep_n), "iq", "pseudo_symmetry_index"? , "simulation_indices"?})`, `xmap.scan_unit = _get_navigation_axes_unit(am)`; navigation mask → `is_in_data`. Detector: `_detector_is_compatible_with_signal`; sample tilt mismatch between `detector.sample_tilt` and `smp.sample_tilt` → `warnings.warn`. Progress: `print(info)`, `tqdm` per chunk, `print(f"  Indexing speed: {x:.5f} patterns/s")` (matches `tutorials_sanitize.cfg` regex).
Parallelism: `da.map_blocks(_index_chunk, patterns_chunked, dtype=float64, drop_axis=…, new_axis=…)` with `chunksize` patterns per block; inside a block, a Python loop over patterns calling numba (`nogil`) + scipy.fft (`workers=1`); each block allocates its own `xc/fxc/dBeta/sphere` buffers (setup constants shared read-only). Threaded scheduler → near-linear scaling like EMSphInx's ThreadPool. `compute=True` always (returns CrystalMap); lazy input handled by `get_dask_array`.
**Reused**: `ebsd.py:_get_navigation_axes_unit`, `_prepare_patterns_for_refinement`-like reshaping (`get_dask_array`, `get_chunking`), `_detector_is_compatible_with_signal`, `_xmap_is_compatible_with_signal`, `create_coordinate_arrays`, `orix.crystal_map.CrystalMap/PhaseList`, `_dictionary_indexing.py` progress/timing pattern, `_RefinementSetup` structure.
**Tests (real data)**:
1. `nickel_ebsd_small` (+ `remove_static_background(); remove_dynamic_background()`), `det = s.detector` with `pc = pc_average`, `smp = nickel_ebsd_master_pattern_small(...).get_spherical_harmonics(bandwidth=88)` → `xmap = s.spherical_indexing(smp, det, bandwidth=68)`; assert `xmap.size == 9`, phase `ni`, `angles = xmap.orientations.angle_with(s.xmap.orientations, degrees=True)`; `np.all(angles < 2.0)` with `refine=True`, `np.mean(angles < 4) >= 8/9` with `refine=False`; `scores` in (0, 1.2]; `iq` in [0,1]; `keep_n=3` sorted descending; navigation/signal masks; lazy input equal to eager (1e-10); `normalize=False` still < 3°.
2. `nickel_ebsd_large` (session-downloaded): bw=68, `refine=True`, `chunksize=64`; median angle vs bundled `s.xmap` < 0.75°, ≥ 95% < 2°, runtime < 60 s on CI (mark `weekly` if slower; keep an `inav[:10,:10]` subset in the default suite).
3. Convention/forward test: 27 rotations sampled from `orix.sampling.get_sample_fundamental(resolution=15, point_group=Oh)` → `mp.get_patterns(R, det)` (60×60, PC of `nickel_ebsd_small`) → `spherical_indexing` → all `angle_with(R) < 0.5°` (bw=68) — proves the ZYZ→orix map (`~Rotation.from_euler(zyz±π/2)`).
4. Pseudo-symmetry: same as 3 with `pseudo_symmetry_ops` = 90° about [1,0,0] hmm—for cubic that's a symmetry; use a synthetic low-symmetry phase (`make_n_fold` sphere with `pg=4`) — or reuse orix `Rotation.from_axes_angles([0,0,1], 90)` on `nickel_ebsd_master_pattern_small` treated as if triclinic; assert `pseudo_symmetry_index` prop exists.
5. Multi-phase: `[smp_ni, smp_ni.rotate(…)]`? — better `[smp_ni, smp_si]` (si from `ebsd_master_pattern("si")` — Zenodo, weekly) with `phase_id` all 0 for Ni patterns.
6. Errors: wrong bandwidth (<16 or >512), even/odd handling, detector shape mismatch, `azimuthal != 0`, unknown `preprocess` keys.
7. `refine_orientation_spherical(s.xmap)` improves or keeps `scores` for all 9 (≥ −1e-6).
Acceptance: above; benchmark `test_spherical_indexing(benchmark)` on `nickel_ebsd_small` bw=68 records patterns/s (target ≥ 15 pat/s/thread; informational).
CHANGELOG (Added): "Spherical indexing of EBSD patterns via `EBSD.spherical_indexing()` and `EBSD.refine_orientation_spherical()`, a CPU port of EMSphInx (`#NNN`)."

### P6 — `spherical-pseudo-symmetry` — `specs/2026-08-21-spherical-pseudo-symmetry/`
**Files**: `src/kikuchipy/indexing/_spherical_pseudo_symmetry.py`, `__init__.pyi` (+`spherical_master_pattern_cross_correlation`), tests `tests/test_indexing/test_spherical_pseudo_symmetry.py`.
```python
def spherical_master_pattern_cross_correlation(
    master_pattern: SphericalMasterPattern, other: SphericalMasterPattern | None = None,
    bandwidth: int = 88, cutoff: float = 0.5, refine: bool = True, merge_angle: float = 2.0,
    return_volume: bool = False,
) -> tuple[Rotation, np.ndarray] | tuple[Rotation, np.ndarray, np.ndarray]
# rotations sorted by relative intensity (desc), intensities relative to the identity peak (auto) / global max; volume (bw, 2bw-1, 2bw-1) as extractBunge cube (ZXZ) or raw (documented)
def _local_maxima(xc, threshold) -> np.ndarray   # 3x3x3 neighbourhood incl. glide (master_xcorr.cpp:107-140)
def pseudo_symmetry_operators(rotations, intensities, point_group: Symmetry, cutoff, angle_tol=2.5) -> Rotation  # drop symmetry ops (|q·op| > 0.999)
```
Reproduces `MasterXcorr` (bw clamp [53,313], `factor=0.95`, `removeDC()`, autocorrelation identity index `idxIdent`), and adds `Rotation` output directly usable as `pseudo_symmetry_ops` in `EBSD.spherical_indexing`/`refine_orientation`. Optional plotting via orix stereographic plot (`Rotation.axis` markers coloured by intensity) instead of SVG diagram.
**Tests**: real: Ni `.sht` autocorrelation at bw=53 → identity peak intensity 1.0, the 24 proper cubic ops recovered as maxima with intensity ≈ 1 (each within 0.05° of an `Oh.proper_subgroup` element after refinement, tolerance 0.05°); no non-symmetry maxima above 0.5. Synthetic pseudo-symmetric sphere (`make_n_fold(4)` on a random function then break by 10 % → 90° maxima at ~0.9); compare volume with `MasterXcorr.exe pseudo_sym.h5` (weekly, if exe available: Pearson > 0.999 after alignment) — see D.
CHANGELOG: "Pseudo-symmetry prediction by spherical master pattern cross-correlation (`kikuchipy.indexing.spherical_master_pattern_cross_correlation`)."

### P7 — `sht-visualisation-utilities` — `specs/2026-08-22-sht-visualisation-utilities/`
**Files**: `src/kikuchipy/indexing/_spherical_master_pattern.py` (+`plot(projection="stereographic"|"lambert", hemisphere, ...)`, `+summary()` printing the sht2png header dump), `src/kikuchipy/io/plugins/emsphinx_sht/_api.py` (+`describe_sht(path) -> str`), `src/kikuchipy/_sht/_namelist.py` (`read_namelist(path) -> dict`, `write_namelist(params, path)`, `SphericalIndexingParameters` dataclass ↔ `EBSD.spherical_indexing` kwargs + `EBSDDetector` ↔ EMSphInx `pctr/vendor/thetac/delta`), `src/kikuchipy/io/plugins/oxford_binary/_api.py` (expose `beam_x/beam_y` scan grid: `get_scan_positions()` public helper via `original_metadata` — EBSPDims), doc note that PatternRepack ≡ `kp.load(...)`, `EBSD.downsample(n)`, `EBSD.save("*.h5")` (+ helper `kp.io.plugins.emsphinx_sht`? no — add `write_emsphinx_pattern_file(signal, path, manufacturer="Bruker")` in `src/kikuchipy/io/plugins/kikuchipy_h5ebsd/_api.py`? Better: `src/kikuchipy/_sht/_emsphinx_io.py: write_patterns_for_emsphinx(signal, path)` used by the regression harness and exposed as `kp.io.plugins.emsphinx_patterns`? Keep private + documented in tutorial).
Tests: `plot` returns figure (agg); `summary()` string equals the `sht2png.exe` printout format for the shipped `.sht` (golden text fixture generated once); namelist round trip and defaults equal `IndexEBSD -t` template values (compare with the report §1.9 table); `EBSDDetector`→`pctr` conversions for all four vendors vs formulas (`x* = (cX+w/2)/w` etc.) using benchmark values (`-1.92, 0.48, 17902.1` ↔ `0.48, 0.505, 0.560001`); `.ebsp` scan dims from `oxford_binary_file` fixture.
CHANGELOG: "Plotting/summary of `.sht` master patterns; EMSphInx namelist import/export."

### P8 — `spherical-indexing-emsphinx-regression` — `specs/2026-08-23-emsphinx-regression/`
**Files**: `src/kikuchipy/data/emsphinx_sht/{ni_20kv_bw384.sht, ni_small_bw68_reference.npz, ni_large_bw68_reference.npz}` (+`_registry.py`, `_data.py: emsphinx_reference_indexing(name)`), `tests/test_indexing/test_spherical_indexing_emsphinx_regression.py`, `benchmarks/indexing/test_spherical_indexing.py` (extend), `doc/dev/…` no; scripts (not shipped, kept in `specs/2026-08-23-emsphinx-regression/scripts/{make_reference.py, ni_small.nml, ni_large.nml}`).
Tests (weekly + a small default one): see section D for tolerances/commands.

### P9 — `spherical-indexing-tutorial` — `specs/2026-08-24-tutorial-docs/`
**Files**: `doc/tutorials/spherical_indexing.ipynb`, `doc/tutorials/index.rst` (+`spherical_indexing` after `hough_indexing`), `doc/tutorials/run_nbval.sh` (only if outputs are stored — default: no), `doc/tutorials/tutorials_sanitize.cfg` (+regex `Indexing speed: .* patterns/s` already covered by `N patterns/s`), `doc/user/related_projects.rst`, `doc/user/installation.rst` (no new deps → nothing), `CHANGELOG.rst` (consolidated entries), `src/kikuchipy/__init__.py` `credits` + `.zenodo.json` (add yourself), `doc/reference/index.rst` untouched (auto via `.pyi`).
Notebook outline (mirrors `hough_indexing.ipynb`; black@77; hidden first cell; thumbnail tag on the IPF map cell):
1. hidden MD boilerplate → `# Spherical indexing` (what SHT indexing is; Lenthe 2019 DOI; licence/patent note sentence; CPU only).
2. imports; `s = kp.data.nickel_ebsd_large(allow_download=True)`.
3. `## Pre-indexing maps` (VBSE RGB, static/dynamic background, IQ map).
4. `## Master pattern as spherical harmonics`: `mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)`; `smp = mp.get_spherical_harmonics(bandwidth=128)`; `smp` repr; `smp.to_master_pattern(npx=200).plot()` vs `mp.plot()`; bandwidth truncation series (bw 32/68/128) `plot`; fast bandwidth list `{53,63,68,74,88,95,113,123,158,…}`; `smp.save("ni.sht")` and `kp.load("ni.sht")` round trip; note on the EMSphInx SHT database `.sht` files.
5. `## Calibrate detector-sample geometry` (as in hough tutorial: `EBSDDetector(shape, sample_tilt=70)`, `extract_grid`, `hough_indexing_optimize_pc` PSO or simply `pc=[0.4198,0.2136,0.5015]`; `det.plot(pattern=…)`); MD on PC conventions vs EMSphInx `pctr/vendor` (table from `emsphinxebsd.rst`).
6. `## Perform indexing`: `signal_mask = ~kp.filters.Window("circular", det.shape).astype(bool)`; `xmap = s.spherical_indexing(smp, det, bandwidth=68, normalize=True, refine=True, signal_mask=signal_mask)`; `xmap`; save (commented).
7. `## Validate indexing results`: scores + IQ maps & histograms; `IPFColorKeyTSL(m-3m, X)` maps X/Y/Z with inset key; geometrical simulations overlay (`ReciprocalLatticeVector`→`KikuchiPatternSimulator.on_detector`); `get_rgb_navigator`.
8. `## Compare with Hough and dictionary indexing`: `angle_with(s.xmap)` histogram; bandwidth sweep 53/68/88 timings and misorientation vs reference; `normalize`/`refine` effect.
9. `## Refinement`: `refine_orientation_spherical` and kikuchipy `refine_orientation` (NLopt) chaining.
10. `## Pseudo-symmetry`: `spherical_master_pattern_cross_correlation(smp, bandwidth=53)` — Ni shows only symmetry ops; MD on how to pass `pseudo_symmetry_ops`.
11. `## Interoperability with EMSphInx` (write `.sht`, write namelist, PatternRepack/EBSPDims equivalents in 3 cells).
12. `## What's next?`
Validation: `cd doc && make html`; `pytest --nbval doc/tutorials/spherical_indexing.ipynb --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg` (run once locally even if outputs are not stored); notebook runtime < 5 min on a laptop (bw=68, 4125 patterns ≈ 30-60 s with 8 threads).
CHANGELOG (Added, consolidated): the entries above + "Tutorial on spherical indexing."; PR checklist items ticked.

---

## C. Algorithm-level design

**Data structures**
- SH coefficients: `np.ndarray` complex128 shape `(bw, bw)`, `alm[m, l]`, `l < m` zero, only `m ≥ 0` (`a^l_{−m} = (−1)^m conj(a^l_m)`), no Condon–Shortley phase (Schaeffer normalisation) — identical to EMSphInx `alm[m*bw + l]`; numba kernels take split `re/im` C-contiguous arrays.
- Sphere: `(2, dim, dim)` float64 (north, south), row = Y index, col = X index, `dim` odd; ring tables `(ring_index[offsets[y]:offsets[y+1]])` int64 precomputed once per `dim` (LRU-cached module-level dict keyed on `(dim, layout, bw)`).
- SHT constants: `weights (Nw, Nt)`, `cos_lats (Nt,)`, `amn/bmn (bw,bw)`, per-ring DFT matrices (only for `Nt ≤ 80`) else scipy `rfft` loop.
- Correlator constants: `wig_d_half_pi (bw,bw,bw)` transposed layout `[m][k][j]`, `E (bw,bw)`, `W,B (bw,bw,bw)`; buffers `fxc (slp, slp, bwp)` complex128, `xc (bwp, slp, slp)` float64, `d_beta (bw,bw,bw,2)`; normalised: `r_den (bwp, slp, slp)` and `s2m`.
- Correlation grid ↔ Euler: `α = 2πm/slp − π/2`, `β = 2πk/slp − π`, `γ = 2πn/slp − π/2` (ZYZ), glide `R(α,β,γ) = R(α+π, −β, γ+π)` for `k ≥ bwp`.
- Results per pattern: `(keep_n, 7)` float64 `[phase, corr, iq, qw, qx, qy, qz]` → CrystalMap.

**SHT implementation choice**: own numpy/numba port (not pyshtools/shtns): (i) EMSphInx uses a *square Legendre* grid with `N_φ = 8y` rings and Sneeuw weights that no library provides; using pyshtools' DH/GLQ grids would change the back-projection sampling and the coefficient normalisation, breaking bit-level regression against `IndexEBSD`; (ii) no new dependency/licence (pyshtools is BSD-3 but ~40 MB compiled, no ARM wheels; shtns has no wheels); (iii) cost is small: analyze is `O(dim²·bw)` ≈ 0.3 Mflop·… for bw=68 (< 1 ms in numba). scipy.special (`sph_harm_y`) and sympy Wigner used only in tests.

**xcorr FFT plan**: `slp = fast_size(2bw−1)` (own {2,3,5,7,11,13}-smooth search identical to `fft.hpp`; note `scipy.fft.next_fast_len` also uses 2,3,5,7,11 — differences possible for 13-smooth, so own function), `bwp = slp//2+1`. Numba kernel fills `fxc[k,n,m]` for `k∈[0,bw)`, `n∈[0,bwp)`, `m∈[0,bw)` with the four mirror slots and symmetry masks (`m % n_fold != 0` rows zero; `mirror` → `j` step 2 with parity start), zero-pads `k∈[bw, slp−bw]`. Then `tmp = scipy.fft.ifft2(fxc, axes=(0,1)) * slp²` restricted to non-zero `m`-planes (`m % n_fold == 0`), `xc = scipy.fft.irfft(tmp[:bwp], n=slp, axis=2) * slp`. Symmetry reduction cuts the kernel cost by `n_fold` (rows) × 2 (mirror) — same as EMSphInx. Peak: argmax over `xc*r_den` (normalized), then 3×3×3 tri-quadratic, then Newton on `derivatives()` (analytic Wigner-d derivatives, `dTablePre` per iteration, `O(bw³)`).

**Threading**: dask threaded scheduler + `da.map_blocks` over pattern chunks (`chunksize` default from `_batch_estimate`, clamp ≥ 8), numba kernels `nogil=True`, scipy.fft `workers=1` inside blocks; per-block private buffers, shared read-only constants captured in the closure. `dask.config` `num_workers` honoured via `n_workers`. This mirrors kikuchipy refinement (`_refinement.py`) and EMSphInx's ThreadPool batches. `numba parallel=True` not used (FFT not callable in nopython).

**Memory (per worker; constants shared)**
| bw | dim | slp | bwp | fxc (complex) | xc (real) | r_den (shared) | wig_d ½π + W + B (shared) | d_beta per worker | ≈ per worker |
|---|---|---|---|---|---|---|---|---|---|
| 63 | 65 | 125 | 63 | 15.8 MB | 7.9 MB | 7.9 MB | 2.0+4.0 MB | 4.0 MB | ~30 MB |
| 68 (default) | 71 | 135 | 68 | 19.8 MB | 9.9 MB | 9.9 MB | 2.5+5.0 MB | 5.0 MB | ~37 MB |
| 88 | 91 | 175 | 88 | 43.1 MB | 21.6 MB | 21.6 MB | 5.5+10.9 MB | 10.9 MB | ~80 MB |
| 113 | 115 | 225 | 113 | 91.5 MB | 45.8 MB | 45.8 MB | 11.5+23.1 MB | 23.1 MB | ~165 MB |
(+ `tmp` complex `(slp,slp,bwp)` transient of the same size as `fxc`; with 8 threads at bw=113 ≈ 1.5 GB total — acceptable, documented; per-phase `r_den`/`flm` add the shared column per phase.) Master pattern SHT at bw=384: `alm` 2.4 MB, grid (2,387,387) 2.4 MB, DCT oversample (547²) 2.4 MB, weights `Nw×Nt` ≈ 97×194 — trivial.

**Euler conventions → orix**: internal ZYZ `(α,β,γ)` from the correlator (crystal→sample as EMSphInx describes) → `zyz_to_quaternion` (`rotations.hpp:973-989`) → identity `north_pole_rotation` (kept as a hook, default identity like `northPoleQuat()`) → conjugate → sample→crystal passive quaternion `[w,x,y,z]` = orix `Rotation(data)`. Equivalent, verified formula: `Rotation = ~Rotation.from_euler([α−π/2, β, γ+π/2])` (Bunge, `direction="lab2crystal"`); inverse for `refine_orientation_spherical`: `zyz = bunge_to_zyz((~r).to_euler())`. Pseudo-symmetry: `q_ps` applied on the crystal side first (`q0 * q_ps` in EMSphInx quaternion order = `Rotation(q0) * Rotation(q_ps)` in orix). Pinned by tests: `rotate_harmonics` round trip, EMSphInx synthetic recovery, and the forward-projection test with `mp.get_patterns`.

**Faithful reproduction of EMSphInx quirks** (each behind an `emsphinx_compatible: bool = True` module constant/flag documented in `_sht/__init__.py`, default True for regression parity; docs list them): mean over `totW` and quartered corner weights in master normalisation (`master.hpp:565-573`), `interpPeak` bounds check on `x[0]` twice (`sht_xcorr.hpp:421`), Gaussian-fit abscissa off-by-one (`gaussian.hpp`), `sum2` unused (metric not divided by pattern norm), equator ring skipped in south LUT, `north_pole_rotation` identity. Fixed silently (do not affect standard results): south `sphere_index` offset (`detector.hpp:552`), `rescale(scale)` width-twice.

---

## D. Test strategy with real data

**Datasets**: `kp.data.nickel_ebsd_small()` (in-package, 9 patterns, bundled reference `xmap`, per-point PC), `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)` (in-package), `kp.data.nickel_ebsd_large()` (downloaded at pytest session start), `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both")` (Zenodo 305 MB, already in local `develop` cache; weekly), `kp.data.ebsd_master_pattern("si")` (weekly multi-phase), new `kp.data.nickel_sht_master_pattern()` (`ni_20kv_bw384.sht`, in-package), new `kp.data.emsphinx_reference_indexing("nickel_ebsd_small"|"nickel_ebsd_large")` (`.npz` in-package: `phi1, Phi, phi2, metric, iq, phase`, float32; ≈ 100 kB for 4125 points). Local-only extra checks (not committed): `c:/Users/westraadt.1/Repos/EMSphInx/data/Ni {20kV 75.7deg}.sht`, `openECCI_RKD/data/Mg-master-17kV.h5` (hcp: exercises `n_fold=6`, mirror), `openECCI_RKD/data/*.up1` + `.ang` (EDAX up1 flip semantics), gated by `EMSPHINX_ROOT`/`KP_LOCAL_DATA` env vars with `pytest.skip` when absent.

**Reference generation (once, scripts in `specs/2026-08-23-emsphinx-regression/scripts/`, results shipped)**
1. `.sht`: `c:/Users/westraadt.1/Repos/EMSphInx/build/Release/mp2sht.exe "C:\Users\westraadt.1\AppData\Local\kikuchipy\kikuchipy\Cache\develop\data\ebsd_master_pattern\ni_mc_mp_20kv.h5" ni_20kv_bw384.sht` (bw=384, sig=70 — matches the Ni data's 70° tilt; do NOT use `Ni {20kV 75.7deg}.sht` for indexing references). Also `sht2png.exe ni_20kv_bw384.sht ni_sqleg.png ni_stereo.png` (golden images for P7, compare after synthesis: normalised image correlation > 0.999).
2. Repack patterns for `IndexEBSD` (h5py script `make_reference.py`): dataset `/patterns` uint8 `(n, 60, 60)` written row-major exactly as `s.data.reshape(-1,60,60)`, root dataset `Manufacturer = "Bruker"` (→ `flp=false`, matching kikuchipy image convention with Bruker PC), no compression.
3. Namelist `ni_small.nml` (template from `IndexEBSD.exe -t`, values):
```
 ipath = ''
 patfile = 'ni_small_patterns.h5'   patdset = 'patterns'
 masterfile = 'ni_20kv_bw384.sht'   psymfile = ''
 patdims = 60, 60      circmask = 0     gausbckg = .FALSE.   nregions = 4
 delta = 100.0         vendor = 'Bruker' pctr = 0.4251, 0.2134, 0.5007   thetac = 0.0
 scandims = 3, 3, 1.5, 1.5   scanname = ''   roimask = ''
 bw = 68   normed = .TRUE.   refine = .TRUE.   nthread = 1   batchsize = 1
 opath = ''  datafile = 'ni_small_ref.h5'  vendorfile = 'ni_small_ref.ang'  ipfmap = 'ni_small_ipf.png'  qualmap = 'ni_small_xc.png'
```
(`delta=100` only to pass the 5–90 mm sanity check; geometry is ratio-invariant with Bruker fractional PC. `pctr` = `s.detector.pc_average` for the small set; for `nickel_ebsd_large` use `0.42326, 0.21363, 0.50207`, `scandims = 75, 55, 1.5, 1.5`, `patfile='ni_large_patterns.h5'`.) Note the mismatch to reproduce in kikuchipy: pass `preprocess=dict(circ_radius=0, n_regions=4)` and `signal_mask` circular (EMSphInx `circmask=0` → both detector circular mask and AHE mask), and use `pc = pc_average` (EMSphInx has one PC per scan). Also generate a `nregions=0` (no AHE) variant to isolate preprocessing differences.
4. Run: `IndexEBSD.exe ni_small.nml` and `IndexEBSD.exe ni_large.nml`; convert `Scan 1/EBSD/Data/{Phi1,Phi,Phi2,Metric,IQ,Phase}` from the output h5 into `.npz` (+ store `EMSphInx commit 60f3517`, nml text, and `IndexEBSD` log header as string arrays in the npz).
5. `MasterXcorr.exe 53 0.5 "<ni_mc_mp_20kv.h5>" "<ni_mc_mp_20kv.h5>"` → `pseudo_sym.h5:/Cross Correlation` (53×105×105 float64 ≈ 4.7 MB; keep locally / in `kikuchipy-data`, weekly test only) + stdout list of maxima (small `.txt`, ship).

**Tolerances / assertions**
- SHT/Wigner/xcorr unit tests: as in P1/P3 (5e-3 round trip; 1e-12 vs sympy; ≤ 0.01° synthetic rotation recovery).
- mp2sht parity: rel. L2(alm) ≤ 1e-4, per-coefficient ≤ 1e-3·max (weekly, full MP).
- `IndexEBSD` regression (`nickel_ebsd_small`, default suite; `nickel_ebsd_large`, weekly): with `preprocess=dict(circ_radius=0, n_regions=4)`, `bandwidth=68`, `normalize=True`, `refine=True`, `pc=pc_average`: misorientation to reference (`orix Orientation.angle_with`, symmetry-reduced) median < 0.1°, ≥ 99% < 0.5°, max < 2°; `metric` Pearson r > 0.99 and mean |Δmetric| < 0.01; `iq` mean |Δ| < 0.02. (First run establishes actual values; tighten in `validation.md`.)
- vs kikuchipy bundled `xmap` (Hough+refined ground truth): small — all 9 < 2° (refine on); large — median < 0.75°, ≥ 95% < 2°.
- Forward-projection convention: 27 rotations, all < 0.5° (bw=68).
- MasterXcorr: identity peak = 1.0; symmetry-op peaks within 0.05° and intensity > 0.98; volume Pearson > 0.999 vs `pseudo_sym.h5` (weekly).
- CPU/CUDA benchmark files in `EMSphInx/benchmarks/` are **not** used (inputs missing, metric scales differ).
Where files live: `src/kikuchipy/data/emsphinx_sht/` (in-package, md5 in `_registry.py`, no URL) for `.sht` (75 kB), `ni_small_bw68_reference.npz` (~1 kB), `ni_large_bw68_reference.npz` (~100 kB), `masterxcorr_ni_bw53_maxima.txt`; the `pseudo_sym.h5` volume and any >1 MB artefacts go to `kikuchipy-data` (needs upstream) or stay local under `KP_LOCAL_DATA` with skip.

Commands: `pytest tests/test_sht tests/test_indexing/test_spherical* tests/test_io/test_emsphinx_sht.py -n 4 --cov=kikuchipy`; `pytest --weekly -k "spherical or sht"`; `pytest --doctest-modules src/kikuchipy/_sht src/kikuchipy/indexing/_spherical*`; `pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py`; `pre-commit run --all-files`.

---

## E. Risks and open questions for the user (≤ 6)
1. **Legal go/no-go & target**: EMSphInx is GPL-2.0-or-later (fine → GPLv3+) but the ReadMe states the central algorithm is under a *provisional patent* (CMU). Is the deliverable (a) fork-only research code, or (b) an upstream PR to `pyxem/kikuchipy`? For (b) I recommend opening a kikuchipy issue first (referencing EMSphInx issue #7 by hakonanes) and emailing `pyxem.team@gmail.com` about the patent question before P5. Please confirm.
2. **Faithful bugs vs fixes**: default to reproducing EMSphInx's result-affecting quirks (`emsphinx_compatible=True`: 2× master mean, quartered corners, `interpPeak` bound check, Gaussian off-by-one, metric not normalised by pattern norm) so regression is tight, with a flag to fix them — OK? Or fix by default and accept looser regression tolerances?
3. **Preprocessing scope**: keep kikuchipy style (user preprocesses; `spherical_indexing` only back-projects/normalises) with EMSphInx's Gaussian-background + mosaic AHE available via `preprocess=dict(...)`, or also expose them as public `EBSD` methods (`remove_gaussian_background`, `mosaic_histogram_equalization`)? Default plan: private + `preprocess` kwarg only.
4. **Reference data hosting**: ship the `.sht` (75 kB) and `.npz` references (~100 kB) in-package under `src/kikuchipy/data/emsphinx_sht/` (no upstream dependency), and keep the MasterXcorr volume local/kikuchipy-data — acceptable? (Adding to `kikuchipy-data` requires upstream maintainers.)
5. **Scope of first release**: include multi-phase indexing and pseudo-symmetry ops in `EBSD.spherical_indexing` (P5) and the MasterXcorr port (P6) in the first PR series, or ship single-phase indexing first (P1–P5, P8, P9) and P6/P7 later?
6. **`specs/` placement/commit policy**: commit `specs/` (constitution + dated feature specs + reference-generation scripts) to the fork's `develop` and exclude it from any upstream PR (single squashed feature PR), or keep specs outside the repo? Also confirm bandwidth default 68 and the public class name `SphericalMasterPattern` (alternative: `MasterPatternHarmonics`).

---

### Critical Files for Implementation
- `c:/Users/westraadt.1/Repos/EMSphInx/include/sht/square_sht.hpp` (grids, weights, analyze/synthesize — P1 source of truth)
- `c:/Users/westraadt.1/Repos/EMSphInx/include/sht/sht_xcorr.hpp` (correlator, peak interpolation, Newton refinement, normalisation — P3)
- `c:/Users/westraadt.1/Repos/EMSphInx/include/modality/ebsd/detector.hpp` (+ `include/idx/indexer.hpp`, `include/idx/master.hpp`) (back-projection LUT, indexer glue, master spectra — P2/P4/P5)
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/ebsd.py` (new `spherical_indexing`/`refine_orientation_spherical` next to `dictionary_indexing` L1827–1984; helper reuse L2880–3105)
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/__init__.pyi` and `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/_refinement/_refinement.py` (public export + dask `map_blocks`/setup-object pattern to mirror; also `_dictionary_indexing.py` for CrystalMap construction and progress printing)