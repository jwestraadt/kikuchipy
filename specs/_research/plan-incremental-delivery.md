# Implementation plan: port EMSphInx spherical indexing (CPU path) into kikuchipy

Perspective: incremental delivery and risk. Eight small, independently mergeable feature branches (plus a "constitution" branch), each valuable alone, ordered so the riskiest numerical kernels are validated against EMSphInx unit tests before anything touches `EBSD`. All paths absolute unless obviously repo-relative. Base spec date: 2026-08-16.

Repo facts verified while planning (kikuchipy `develop` @ `49b1c11c`, fork `jwestraadt/kikuchipy`, no `upstream` remote; EMSphInx `master` @ `60f3517`):
- Untracked `c:/Users/westraadt.1/Repos/kikuchipy/IndexEBSD.nml` (written by `IndexEBSD.exe -t`) must be deleted before the first commit; `doc/tutorials/hybrid_indexing.ipynb` and `load_save_data.ipynb` are locally modified and must not be swept into feature commits.
- kikuchipy h5ebsd files carry a lowercase `/manufacturer = "kikuchipy"` dataset; EMSphInx's `PatternFile::Read` looks for `Manufacturer` and only accepts EDAX/EMsoft/Oxford/Bruker/Bruker Nano/DREAM.3D (`include/modality/ebsd/pattern.hpp:463-471, 608-637`). So kikuchipy files can NOT be fed to `IndexEBSD.exe` directly; the reference-generation recipe (section D) repacks patterns with h5py.
- Sample tilt in `IndexEBSD` comes from the `.sht` `primaryAngle` (`include/modality/ebsd/idx.hpp`, `sampleTilt(phases[0].getSig())`), not from the namelist. `EMSphInx/data/Ni {20kV 75.7deg}.sht` has sig = 75.7 deg, so a fresh `.sht` from `ni_mc_mp_20kv.h5` (MC sig = 70 deg) is required for any regression against kikuchipy's Ni data.
- Legendre grid roots come from `numpy.polynomial.legendre.leggauss(dim-2)`; the ring quadrature weights must still be solved from the Sneeuw system exactly as `computeWeightsSkip` does (`square_sht.hpp:1022-1063`), because the hemisphere rule with the equator node included is not the plain Gauss-Legendre rule.
- The `_master_pattern.py` numba helpers `_vector2lambert`, `_lambert2vector`, `_get_lambert_interpolation_parameters`, `_get_pixel_from_master_pattern`, and `_get_direction_cosines_from_detector` (`src/kikuchipy/signals/util/_master_pattern.py:83-124, 544-773`) exist and are affine-equivalent / numerically identical to EMSphInx's Lambert and `sampleDir`, respectively (see reports).

---

## A. Constitution (`specs/`)

### `specs/mission.md` (one paragraph)
> kikuchipy gains a pure-Python, CPU-only implementation of EMSphInx spherical indexing (Lenthe, Singh and De Graef, Ultramicroscopy 207 (2019) 112841): master patterns are transformed to spherical-harmonic coefficients on a square-Legendre grid, experimental EBSD patterns are back-projected onto the sphere via `EBSDDetector` geometry, and orientations are found as the maximum of the SO(3) cross-correlation computed by real 3-D FFTs with symmetry-reduced spectra, optionally refined by Newton's method on analytic Wigner-d derivatives. The port must (1) reproduce EMSphInx's C++ unit tests (`test/sht/{square_sht,wigner,sht_xcorr}.cpp`) to their stated tolerances, (2) agree with `IndexEBSD.exe` on kikuchipy's Ni datasets to sub-degree misorientation, (3) follow kikuchipy's numpy/numba/dask conventions with zero new required dependencies, (4) preserve EMSphInx's GPL-2.0-or-later notices under kikuchipy's GPL-3.0-or-later, and (5) be delivered as small independently mergeable PRs, each accompanied by a dated spec folder, real-data tests, and a tutorial notebook at the end. The GUI (`EMSphInxEBSD`) and `ShtWisdom` are not ported; `mp2sht`, `IndexEBSD`, `MasterXcorr`, `sht2png`, `PatternRepack` and `EBSPDims` become code-driven functionality usable from notebooks.

### `specs/tech-stack.md` (bullets)
- Runtime deps: only what `pyproject.toml` already requires (numpy, scipy >= 1.7 (`scipy.fft`, `scipy.linalg`, `scipy.special` for validation), numba >= 0.57, dask[array], orix >= 0.12.1, h5py, hyperspy, matplotlib, tqdm, pooch). No FFTW/pyfftw/shtns/pyshtools/rocket-fft. Optional deps untouched.
- Numerics: float64/complex128 throughout (EMSphInx `Real = double`); float32 fast path is a later optimisation. FFTs via `scipy.fft` with `norm="forward"` on inverse transforms to reproduce FFTW's unnormalised `c2r`/`REDFT01`. DCT via `scipy.fft.dctn(type=2)` / `idctn(type=3)` (identical to FFTW REDFT10/REDFT01, unnormalised).
- Kernels: `@njit(cache=True, nogil=True)` (add `fastmath=True` only after the accuracy tests pass at the tolerance); explicit eager signatures where dtypes are fixed (kikuchipy style, e.g. `_master_pattern.py:127`). No `parallel=True` in kernels; parallelism comes from `dask.array.map_blocks` over pattern chunks with the threaded scheduler, exactly like `_refinement.py:391-424`.
- Layout: private package `src/kikuchipy/indexing/_spherical/` (leading underscore, warning docstring like `src/kikuchipy/indexing/_refinement/__init__.py`), public names exported through `src/kikuchipy/indexing/__init__.pyi` (`lazy_loader.attach_stub`), signal methods on `EBSD` in `src/kikuchipy/signals/ebsd.py` next to `dictionary_indexing` (L1827) / `refine_orientation` (L1986). `.sht` I/O as an io plugin `src/kikuchipy/io/plugins/emsphinx_sht/` (`specification.yaml` + `_api.py`).
- Style: ruff + ruff-format via `pre-commit run --all-files`; numpydoc, types in signatures only, comment/docstring lines <= 72 chars, three import blocks, new-style GPLv3+ header from `.license.tmpl` PLUS an EMSphInx attribution block in every derived file: `Derived from EMSphInx (c) 2019 De Graef Group, Carnegie Mellon University, author William C. Lenthe, GPL-2.0-or-later, relicensed under GPL-3.0-or-later per its "or any later version" clause; translated to Python and modified by <name>, <date>` (GPLv2 s.2a modification notice). Files must live in GPL areas only (kikuchipy `doc/dev/licensing_considerations.rst` on upstream develop).
- Rotations: `orix.quaternion.Rotation` everywhere in public API; ZYZ <-> Bunge via `(alpha - pi/2, beta, gamma + pi/2)`; results are sample->crystal like every other kikuchipy `CrystalMap`.
- Tests: pytest, `tests/test_indexing/test_spherical_*.py`, `tests/test_io/test_emsphinx_sht.py`, `tests/test_signals/test_ebsd_spherical_indexing.py`; every numba kernel tested both as `f(...)` and `f.py_func(...)`; real data via `kp.data.nickel_ebsd_small()`, `kp.data.nickel_ebsd_master_pattern_small(...)`, `kp.data.nickel_ebsd_large(allow_download=True)`, `kp.data.ebsd_master_pattern("ni")` (weekly); heavy tests `@pytest.mark.weekly` (declared marker). Benchmarks in `benchmarks/indexing/test_spherical_indexing.py` (pytest-benchmark).
- Commands (venv is uv-managed, no pip): `uv run pre-commit run --all-files`; `uv run pytest tests/test_indexing/test_spherical_sht.py -n 4`; `uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical`; `uv run pytest --cov=kikuchipy tests/test_indexing tests/test_signals/test_ebsd_spherical_indexing.py`; `uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py`; `uv run pytest --weekly -k spherical`; docs `cd doc && make html`; notebooks `./doc/tutorials/run_nbval.sh`.
- Git: `git remote add upstream https://github.com/pyxem/kikuchipy.git && git fetch upstream && git merge upstream/develop` first (local clone is ~3 months behind and lacks `licensing_considerations.rst`); every feature branch off `develop`; `git commit -s`; merge (not rebase) develop into feature branches; PR into `develop` using `.github/pull_request_template.md`; CHANGELOG `Unreleased -> Added` entry with PR link; add contributor to `src/kikuchipy/__init__.py` `credits` and `.zenodo.json` in the first PR.
- Spec workflow per feature: `specs/YYYY-MM-DD-<name>/{plan.md, requirements.md, validation.md, tasks.md}`; order: plan -> user approval -> spec recorded -> tests written (failing) -> implementation -> adversarial review (`/code-review high` on the diff, plus a checklist of the EMSphInx gotchas in the report section 8) -> fix -> commit -> PR.

### `specs/roadmap.md` (ordered phases = features; checkbox tasks summarised, full lists in section B)
0. constitution and housekeeping
1. sht-core: square Legendre/Lambert grids + discrete SHT (analyze/synthesize) + EMSphInx round-trip tests
2. master-pattern-harmonics: `MasterPatternHarmonics` (mp2sht equivalent), `.sht` read/write, `kp.load("*.sht")`
3. spherical-back-projection: detector -> sphere LUT, DCT rescaler, EMSphInx preprocessing (Gaussian bg, mosaic AHE, DCT IQ), window mask
4. spherical-xcorr-indexing: Wigner d tables, SO(3) cross-correlation, peak interpolation, normalised correlator, `SphericalIndexer`, `EBSD.spherical_indexing` (coarse)
5. spherical-refinement: analytic derivatives, Newton refinement, `refine=True` default, `EBSD.spherical_refine_orientation`
6. spherical-pseudo-symmetry: `find_pseudo_symmetry_operators` (MasterXcorr equivalent) + `pseudo_symmetry_ops` in indexing
7. spherical-visualisation: sht2png equivalents (`to_master_pattern`, plots, `.sht` summary), PatternRepack/EBSPDims mappings, xcorr-volume plots
8. spherical-indexing-tutorial: notebook, data registration (reference `.sht`, EMSphInx reference results), docs index, bibliography, related projects, benchmark, changelog consolidation

Each phase's checkbox list: `[ ] plan.md approved`, `[ ] requirements.md/validation.md recorded`, `[ ] failing tests committed`, `[ ] implementation`, `[ ] adversarial review + fixes`, `[ ] pre-commit clean`, `[ ] CHANGELOG entry`, `[ ] PR opened`.

---

## B. Features / branches

Common to every feature: branch off `develop`; spec folder `specs/2026-08-16-<name>/`; commit with `-s`; PR into fork `develop` (upstream later, see open question 1). New-style license header + EMSphInx attribution block in every file derived from EMSphInx. All numba kernels tested via `.py_func` too. CHANGELOG entries go under `Unreleased -> Added` with the form used at `CHANGELOG.rst:41-45`.

### Feature 0 - `specs-constitution` (branch `spherical-indexing-constitution`)
- Spec folder: `specs/2026-08-16-constitution/` (holds only `mission.md`, `tech-stack.md`, `roadmap.md` at `specs/` root).
- Files: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`; delete `IndexEBSD.nml`; add `lenthe2019spherical` (doi 10.1016/j.ultramic.2019.112841) and `lenthe2019pseudo` (doi 10.1107/S1600576719011233) to `doc/user/bibliography.bib`; add EMSphInx to `doc/user/related_projects.rst`; add contributor to `src/kikuchipy/__init__.py` (`credits`) and `.zenodo.json`.
- Validation: `git status` clean apart from the intended files; `uv run pre-commit run --all-files`; `cd doc && make html` builds (bibtex key resolves).
- CHANGELOG: none (docs only) or "Added EMSphInx to related projects".

### Feature 1 - SHT core (branch `spherical-sht-core`, spec `specs/2026-08-16-sht-core/`)
New files
- `src/kikuchipy/indexing/_spherical/__init__.py` (warning docstring, private package)
- `src/kikuchipy/indexing/_spherical/_grid.py` — square grids: `square_to_sphere(X, Y)`, `sphere_to_square(v)` (thin wrappers around `_lambert2vector`/`_vector2lambert` with the affine map `X = (L/sqrt(pi/2) + 1)/2`), `lambert_cos_latitudes(dim)`, `legendre_cos_latitudes(dim)` (pole + positive `leggauss(dim-2)` roots descending), `legendre_normals(dim) -> (dim*dim, 3)` (port of `legendre::normals`, `square_sht.hpp:823-869`), `ring_number(dim) -> (dim, dim) int` (Chebyshev distance), `ring_indices(dim) -> (starts, flat_index)` (port of `readRing`, `square_sht.hpp:942-1014`, buffer slot 0 at phi = 0, counter-clockwise), `ring_solid_angles(dim, layout)` (per-ring, `square_sht.hpp:1105-1138`), `lambert_solid_angles(dim)` (per-pixel Mazonka, `:681-736`; only needed for Lambert layout, can be deferred), `quadrature_weights(dim, layout, skip)` (port of `computeWeightsSkip`; `np.linalg.solve` on the `cos(2 j theta_i)` matrix, then `w_y = 4*pi*what_y/max(1, 8y)`).
- `src/kikuchipy/indexing/_spherical/_sht.py` — `class SphericalHarmonicTransform` and numba kernels `_alm_recursion_tables(bw) -> (amn, bmn)` (Schaeffer eqs 16-18), `_analyze(north, south, ..., alm)`, `_synthesize(alm, ..., north, south)`; ring DFTs done inside numba with precomputed per-ring cos/sin matrices `(Nt, 8y_max, m_max)` OR via `scipy.fft.rfft` per ring in Python (choose numba DFT-matrix; see section C).
- `tests/test_indexing/test_spherical_grid.py`, `tests/test_indexing/test_spherical_sht.py`.
Changed files: `src/kikuchipy/indexing/__init__.pyi` (export `SphericalHarmonicTransform`), `CHANGELOG.rst`.
Public API
```python
class SphericalHarmonicTransform:
    def __init__(self, bandwidth: int, layout: str = "legendre", dim: int | None = None): ...
    # dim default: bw + (3 if bw % 2 == 0 else 2) for legendre, 2*bw + 1 for lambert
    bandwidth: int; dim: int; layout: str; n_rings: int
    cos_latitudes: np.ndarray            # (n_rings,)
    normals: np.ndarray                  # (dim*dim, 3), north hemisphere
    ring_number: np.ndarray              # (dim, dim)
    quadrature_weights: np.ndarray       # (n_weight_sets, n_rings)
    def analyze(self, north: np.ndarray, south: np.ndarray, bandwidth: int | None = None) -> np.ndarray:  # (bw, bw) complex128, alm[m, l]
    def synthesize(self, alm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # (dim, dim) each
    def to_lambert(self, north, south, dim_out=None) / from_lambert(...)   # regridding helpers (Feature 2 may move these)
```
Reused: `_lambert2vector`, `_vector2lambert` (`src/kikuchipy/signals/util/_master_pattern.py:544, 730`), `numpy.polynomial.legendre.leggauss`, `scipy.special.sph_harm_y` (tests only).
Tests
- unit/synthetic: `test_legendre_normals_unit_length`, `test_ring_indices_cover_grid_once`, `test_ring_indices_phi_zero_first_and_ccw`, `test_quadrature_weights_sum_to_4pi` (`sum(w_y*max(1,8y)) == 4*pi`, `sum(what) == 1` to 1e-10 for dim up to 387), `test_analyze_constant_gives_sqrt_4pi` (all other coefficients < 1e-10), `test_analyze_matches_scipy_sph_harm` (evaluate `Y_l^m` for l < 12 on the Legendre grid via `sph_harm_y`, analyze, expect delta at (m,l) with |1 - value| < 1e-8, sign convention: no Condon-Shortley), `test_round_trip_random_spectrum` parametrised exactly like `test/sht/square_sht.cpp:90-201`: seed 0 uniform(-1,1) spectra, `synthesize -> analyze`, per-coefficient max abs error < 5e-3 and mean < 5e-5, Legendre bw in {4, 8, 16, 32, 53, 63, 68, 88, 113, 158, 384} (384 marked slow/weekly), Lambert bw in {4, 16, 32, 64, 128}; `.py_func` variants for both kernels.
- real data: `test_analyze_nickel_master_pattern_power_spectrum`: `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")` -> Legendre regrid (simple bilinear, moved to Feature 2 if not ready; otherwise use Lambert layout bw = 200) -> analyze -> assert m3m systematic zeros: rows `m % 4 != 0` have relative power < 1e-6 and `(l+m)` odd entries < 1e-6 (equatorial mirror), power decays with l.
Acceptance: all above; `analyze` of one 387x387 pair < 50 ms after JIT (bw 384 < 1 s); numba compile cached.
Validation commands: `uv run pytest tests/test_indexing/test_spherical_grid.py tests/test_indexing/test_spherical_sht.py -n 4`; `uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical/_sht.py`.
CHANGELOG: `- Spherical harmonic transform on square Legendre/Lambert grids, ``kikuchipy.indexing.SphericalHarmonicTransform``, ported from EMSphInx. (#NNN)`
Docs: appears automatically in `doc/reference` via `__all__` (custom class template).

### Feature 2 - Master pattern harmonics + `.sht` I/O (branch `spherical-master-pattern-harmonics`, spec `specs/2026-08-16-master-pattern-harmonics/`)
New files
- `src/kikuchipy/indexing/_spherical/_master_pattern_harmonics.py` — `class MasterPatternHarmonics` (port of `MasterSpectra`, `include/idx/master.hpp:153-206, 550-640`) + `_lambert_to_legendre(north, south, dim_out)` (port of `toLegendre`, `:381-416`: rescale Lambert by DCT to `round(sqrt(2)*dim_out)`, bilinear sample at Legendre normals via `sphere_to_square`), `_legendre_to_lambert` (Feature 7 may improve), `_energy_weights_from_accum_e(path)` (h5py read of `EMData/MCOpenCL/accum_e`, summed over the map, `master.hpp:265-278`), symmetry flag tables `_Z_ROTATION_ORDER`, `_HAS_EQUATORIAL_MIRROR`, `_MIRROR_TYPE` (32 point groups keyed on `orix Symmetry.name`, values from `PointGroup::zRot/zMirror/mmType`).
- `src/kikuchipy/indexing/_spherical/_sht_file.py` — struct-level `.sht` v1.1 reader/writer (`build/_deps/shtfile-src/sht_file.in.hpp`): dataclasses `ShtFileHeader`, `ShtCrystalData`, `ShtAtomData`, `ShtSimulationData`, `ShtHarmonicsData`; `read_sht(path) -> ShtFile`, `write_sht(path, ShtFile)`; `pack_harmonics/unpack_harmonics(alm, bw, z_rot, cmp_flags)` (`:1672-1831`); `SPACE_GROUP_ROT`, `SPACE_GROUP_CMP` 230-entry tables (`:1838-1869`) copied verbatim; `crc32c_emsphinx(bytes)` with the exact LUT (`:947-1005`).
- `src/kikuchipy/io/plugins/emsphinx_sht/{__init__.py, __init__.pyi, _api.py, specification.yaml}` — `file_reader(filename, lazy=False, projection="lambert", hemisphere="both", dim=None) -> list[dict]` producing an `EBSDMasterPattern` by synthesising the harmonics on a Legendre grid and regridding to Lambert (or "legendre" projection kept as metadata); `writes: False` (writing goes through `MasterPatternHarmonics.save`). specification: `file_extensions: ['sht']`, `manufacturer: emsphinx`, no footprints (binary magic `*sht` checked in `_api.py`).
- `tests/test_indexing/test_spherical_master_pattern_harmonics.py`, `tests/test_io/test_emsphinx_sht.py`.
- Data: `src/kikuchipy/data/emsphinx/ni_20kv_bw384.sht` (produced by `mp2sht.exe` from `ni_mc_mp_20kv.h5`, ~75 kB, sig 70) registered in `src/kikuchipy/data/_registry.py` (`_registry_hashes`, in-package, no URL) and exposed as `kp.data.nickel_ebsd_master_pattern_harmonics()`? Prefer a loader `kp.data.emsphinx_ni_sht()` returning the path/`MasterPatternHarmonics` (add to `data/__init__.pyi`, `tests/test_data/test_data.py`). Optionally also copy `EMSphInx/data/Ni {20kV 75.7deg}.sht` under a safe name `ni_20kv_75deg_bw384_emsphinx.sht` as the byte-level parse target (redistribution: file created with GPL EMSphInx from EMsoft data; check provenance - open question 5).
Changed: `src/kikuchipy/indexing/__init__.pyi` (export `MasterPatternHarmonics`), `src/kikuchipy/data/{_data.py,_registry.py,__init__.pyi}`, `CHANGELOG.rst`, `doc/user/installation.rst`? no.
Public API
```python
class MasterPatternHarmonics:
    def __init__(self, alm: np.ndarray, phase: orix.crystal_map.Phase, energy: float, sample_tilt: float, bandwidth: int | None = None, metadata: dict | None = None): ...
    @classmethod
    def from_master_pattern(cls, master_pattern: EBSDMasterPattern, bandwidth: int = 384, normalize: bool = True, energy: float | str | None = "weighted", energy_weights: np.ndarray | None = None) -> "MasterPatternHarmonics"
        # requires projection == "lambert", hemisphere == "both"; "weighted" reads accum_e via h5py from master_pattern.metadata/original_metadata file path when available, else uniform mean; float -> single energy slice
    @classmethod
    def from_file(cls, filename: str | Path) -> "MasterPatternHarmonics"     # .sht
    def save(self, filename: str | Path, notes: str = "created with kikuchipy", doi: str = "https://doi.org/10.1016/j.ultramic.2019.112841") -> None
    def resize(self, bandwidth: int) -> "MasterPatternHarmonics"           # crop/zero-pad m-major
    def remove_dc(self) -> None
    def rotate(self, rotation: Rotation) -> "MasterPatternHarmonics"        # uses Feature 4 rotate_harmonics; stub raises until F4
    def to_master_pattern(self, dim: int | None = None, projection: str = "lambert") -> EBSDMasterPattern   # Feature 7 adds stereographic
    alm: np.ndarray (bw, bw) complex128; bandwidth: int; phase: Phase; energy: float; sample_tilt: float
    z_rotation_order: int  # fNf; has_equatorial_mirror: bool  # fMr; has_inversion: bool
    def power_spectrum(self) -> np.ndarray  # (bw,)
    __repr__ -> "MasterPatternHarmonics: bw=384, phase ni (m-3m), 20.0 keV, tilt 70.0 deg, z-fold 4, mirror True"
```
`kp.load("x.sht", projection="lambert", hemisphere="both", dim=None) -> EBSDMasterPattern` with `phase`, `hemisphere`, `projection` set (Legendre synthesis + `_legendre_to_lambert`).
Reused: `EBSDMasterPattern` (`src/kikuchipy/signals/ebsd_master_pattern.py`), EMsoft reader (`src/kikuchipy/io/plugins/_emsoft_master_pattern.py`, sums atoms already), `orix.quaternion.symmetry.get_point_group`, `orix.crystal_map.Phase`, `scipy.fft.dctn/idctn` for the resize, `SphericalHarmonicTransform`.
Tests
- `.sht` parsing of `EMSphInx/data/Ni {20kV 75.7deg}.sht` (path from env var `EMSPHINX_DATA_DIR` or the in-package copy): every header value listed in report section 8 (magic, version (1,1), software `ve49ad6b`, modality 1, 20.0 keV, 75.7 deg, doi, note, sgEff 225, lattice 0.35236 nm, atom Z=28 DW 0.0035, EMsoftED fields, bw 384, zRot 4, cmpFlg 7, doubCnt 9312, CRC 0xf2af93ef); round trip `read -> write -> bytes identical`; `unpack -> pack` identity; CRC on synthetic bytes.
- `mp2sht` parity (real data): `MasterPatternHarmonics.from_master_pattern(kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both"), bandwidth=384)` vs `mp2sht.exe`-generated `ni_small_20kv_bw384.sht` (in-package reference): compare complex coefficients after `remove_dc()` (EMSphInx mean is 2x too large, gotcha 8.7): relative L2 error < 1e-3, per-coefficient max abs error < 1e-3 * max|alm|; sign/conjugation conventions therefore locked. Weekly variant with `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both")` (305 MB, cached) vs `ni_20kv_bw384.sht`.
- symmetry flags: `z_rotation_order`/`has_equatorial_mirror` for all 32 groups equal the EMSphInx tables (`SPACE_GROUP_ROT/CMP` sanity: sg 225 -> 4, 0x7).
- `kp.load(".sht")` returns `EBSDMasterPattern` with `phase.name`, `projection == "lambert"`, shape `(2, dim, dim)`, and its Lambert image correlates > 0.99 with the EMsoft master pattern of the same phase resampled to `dim` (`nickel_ebsd_master_pattern_small`).
- energy weighting: weights from `accum_e` sum to 1 and equal `mp2sht`'s (indirectly through the parity test).
Acceptance: parity thresholds above; `.sht` byte round trip exact; loader plugin discovered by `kp.load` without extension conflicts.
Validation: `uv run pytest tests/test_indexing/test_spherical_master_pattern_harmonics.py tests/test_io/test_emsphinx_sht.py tests/test_data/test_data.py -n 4`; `uv run pytest --weekly -k harmonics`.
CHANGELOG: `- ``kikuchipy.indexing.MasterPatternHarmonics`` to compute, resize, save and load spherical harmonic coefficients of master patterns; ``kikuchipy.load()`` reads EMSphInx/SHTdatabase ``.sht`` files. (#NNN)`
Docs: reference auto; `doc/user/installation.rst` untouched; note `.sht` in `doc/tutorials/load_save_data.ipynb` (small cell) in Feature 8.

### Feature 3 - Back-projection and preprocessing (branch `spherical-back-projection`, spec `specs/2026-08-16-spherical-back-projection/`)
New files
- `src/kikuchipy/indexing/_spherical/_back_projection.py` — `class SphericalBackProjector` (port of `ebsd::BackProjector`, `include/modality/ebsd/detector.hpp:502-630`) and kernels `_build_backprojection_lut(directions, dim, cos_lats, ...)`, `_unproject(pattern_resampled, lut_idx, lut_w, sphere_idx, ring_weights, out_sphere)`; `_detector_scale_factor` (`detector.hpp:401-469`, fraction of a 502x502 Lambert grid landing on the detector); DCT `_rescale_dct(pattern, shape_out, zero_mean=True, highpass=0)` and `_image_quality_dct(dct)` (`util/image.hpp:489-507, 564-619`).
- `src/kikuchipy/indexing/_spherical/_preprocessing.py` — `_gaussian_background(pattern, mask)` (`util/gaussian.hpp`, incl. the documented off-by-one only if parity demands; default: correct version, flag `emsphinx_compatible=True` for the reference tests), `_mosaic_ahe(pattern_u8, n_regions, mask)` (`util/ahe.hpp`, mosaic tiles, 256 bins, mask-aware), `_preprocess_pattern(pattern, circ_radius, gaussian_bg, n_regions)` (`imprc.hpp:166-191` order).
- `tests/test_indexing/test_spherical_back_projection.py`, `tests/test_indexing/test_spherical_preprocessing.py`.
Changed: `src/kikuchipy/indexing/__init__.pyi` (export `SphericalBackProjector`), `CHANGELOG.rst`.
Public API
```python
class SphericalBackProjector:
    def __init__(self, detector: EBSDDetector, bandwidth: int, signal_mask: np.ndarray | None = None, circular_mask: bool = True, oversampling: float = np.sqrt(2), flip: bool = False): ...
    dim: int; sht: SphericalHarmonicTransform; resampled_shape: tuple[int, int]
    def unproject(self, pattern: np.ndarray, return_image_quality: bool = False) -> tuple[np.ndarray, np.ndarray] | ...  # (north, south) zero-mean/unit-std over window, zeros outside
    def window_mask(self) -> tuple[np.ndarray, np.ndarray]                       # binary window on the sphere
    def window_harmonics(self) -> np.ndarray                                     # mlm = SHT(mask)
    def solid_angle_fraction(self) -> float
    def plot(self, pattern=None, ...)  # Feature 7
```
Reused: `_get_direction_cosines_from_detector` (`_master_pattern.py:83-124`; single PC via `detector.pc_average`), `EBSDDetector` (`src/kikuchipy/detectors/_ebsd_detector.py`), `kp.filters.Window("circular")`, `_get_lambert_interpolation_parameters` pattern for bilinear weights, `kp.pattern.rescale_intensity`, `kp.pattern.remove_dynamic_background` (documented alternative to Gaussian bg).
Tests
- synthetic: `test_unproject_constant_pattern_gives_window_mask` (stdev == 0 branch writes 1s), `test_unproject_zero_mean_unit_std_over_window` (weighted mean 0, std 1 to 1e-10), `test_lut_directions_match_detector` (LUT directions vs `_get_direction_cosines_from_detector` within 1e-12; northern hemisphere only for `sample_tilt=70, tilt=0/10`), `test_backproject_spherical_harmonic_recovers_coefficient` (evaluate a low-order `Y_l^m` on the detector directions (`sph_harm_y`), unproject, analyze; the coefficient vector restricted to l <= 12 correlates > 0.95 with the windowed truth = analyze(mask*Y_l^m) computed on the sphere directly), `test_rescale_dct_matches_fftw_convention` (identity size returns `4*w*h*` scaled input as EMSphInx does, i.e. we assert against explicit `dctn/idctn` product), `test_mosaic_ahe_uniform_image_identity`, `test_gaussian_background_recovers_synthetic_gaussian`, `.py_func` for all kernels.
- real data: `kp.data.nickel_ebsd_small()`: unproject all 9 patterns with `detector = s.detector` (`pc = pc_average`, `sample_tilt=70`, `tilt=0`) at bw 68; window solid angle fraction within 20% of `EBSDDetector`-derived analytic estimate; sphere north hemisphere has ~`fraction*dim^2` non-zero pixels; south all zeros; image quality values finite in (0,1).
- Consistency with `mp.get_patterns`: forward-project `nickel_ebsd_master_pattern_small` at identity rotation on `detector`, unproject, and compare unprojected values with the master pattern sampled at the same directions (Pearson r > 0.98 after zero-mean/unit-std) — this locks the detector frame convention before Feature 4.
Acceptance: above; LUT build for 60x60 -> bw 68 < 0.5 s; `unproject` < 2 ms per pattern.
Validation: `uv run pytest tests/test_indexing/test_spherical_back_projection.py tests/test_indexing/test_spherical_preprocessing.py -n 4`.
CHANGELOG: `- ``kikuchipy.indexing.SphericalBackProjector`` to back-project EBSD patterns onto the sphere via ``EBSDDetector`` geometry, with EMSphInx-compatible preprocessing (Gaussian background, mosaic AHE, DCT image quality). (#NNN)`

### Feature 4 - SO(3) cross-correlation and coarse indexing (branch `spherical-xcorr-indexing`, spec `specs/2026-08-16-spherical-xcorr-indexing/`)
New files
- `src/kikuchipy/indexing/_spherical/_wigner.py` — `_wigner_d_pi2_table(bw, transpose=True) -> (bw,bw,bw)` (`wigner.hpp:699-761`), `_wigner_d_table(bw, t, neg_beta) -> (bw,bw,bw,2)` (`:452-559`), `_wigner_d_pre_build(bw)` + `_wigner_d_table_pre(...)` (`:575-691`), scalar `wigner_d(j, k, m, beta)` (`:298-371`, int64 arithmetic), `rotate_harmonics(alm, zyz) -> blm` (`:769-799`), derivative helpers (Feature 5).
- `src/kikuchipy/indexing/_spherical/_xcorr.py` — `class SphericalCrossCorrelator` (port of `Correlator`, `sht_xcorr.hpp`), kernels `_xcorr_spectrum(flm, gln, wig_d_pi2, f_mirror, f_fold, slp, out_fxc)` (the k, n, m, j loop `:657-858`, writes the 4 mirrored slots, zero pads), `_find_peak(xc)`, `_extract_neighborhood_27(xc, idx, slp, bwp)` (with the glide, `:505-544`; use exact `+slP//2 mod slP` when slP even - decide with test), `_interpolate_maxima(p27) -> (peak, dx)` (`:1261-1366`), `_index_to_euler(idx, slp)`, `_euler_to_index(eu, slp)`, `_extract_bunge`; `class NormalizedSphericalCrossCorrelator` (Huhle denominator `:1182-1204`); helper `fast_bandwidths(bw_min=16, bw_max=512)` (`2*bw-1 == scipy.fft.next_fast_len(2*bw-1, real=True)`, cross-checked with EMSphInx's `{2,3,5,7,11,13}`-smooth rule).
- `src/kikuchipy/indexing/_spherical/_indexer.py` — `class SphericalIndexer` (port of `Indexer`, `include/idx/indexer.hpp`) orchestrating preprocess -> unproject -> analyze -> correlate for a batch, returning arrays; `_zyz_to_rotation(zyz) -> Rotation` (`~Rotation.from_euler([a - pi/2, b, g + pi/2])`, sign of `~` verified by test), `_results_to_crystal_map(...)`, `_spherical_indexing(patterns_dask, indexer, nav_shape, step_sizes, chunksize, ...) -> CrystalMap` (dask `map_blocks`, tqdm/ProgressBar, timing print `"  Indexing speed: {x:.5f} patterns/s"`), info message `"Spherical indexing information:\n  Phase name(s): ...\n  Bandwidth: 68 (grid 135^3)\n  Normalized: True, refine: False\n  Indexing N pattern(s) in M chunk(s)"`.
- `tests/test_indexing/test_spherical_wigner.py`, `tests/test_indexing/test_spherical_xcorr.py`, `tests/test_signals/test_ebsd_spherical_indexing.py`.
Changed: `src/kikuchipy/signals/ebsd.py` (new method `spherical_indexing` after `dictionary_indexing`, imports in the kikuchipy import block L50-61), `src/kikuchipy/indexing/__init__.pyi` (export `SphericalCrossCorrelator`, `SphericalIndexer`, `fast_bandwidths`), `CHANGELOG.rst`, `doc/tutorials/tutorials_sanitize.cfg` (regex for `Indexing N pattern(s) in M chunk(s)` if not covered).
Public API
```python
class SphericalCrossCorrelator:
    def __init__(self, bandwidth: int): ...
    bandwidth: int; side_length: int  # slP; half_side: int  # bwP
    def correlate(self, flm: np.ndarray, gln: np.ndarray, f_mirror: bool, f_fold: int, refine: bool = False, eps: float = 0.01) -> tuple[np.ndarray, float]  # zyz (3,), peak
    def compute(self, flm, gln, f_mirror, f_fold) -> np.ndarray  # xc (bwP, slP, slP) real
    def index_to_euler(self, idx) -> np.ndarray; def euler_to_index(self, zyz) -> int
class NormalizedSphericalCrossCorrelator(SphericalCrossCorrelator):
    def __init__(self, bandwidth, flm, flm2, f_mirror, f_fold, mlm): ...
    def correlate(self, gln, refine=False, eps=0.01) -> tuple[np.ndarray, float]
class SphericalIndexer:
    def __init__(self, harmonics: MasterPatternHarmonics | list[MasterPatternHarmonics], detector: EBSDDetector, bandwidth: int = 68, normalize: bool = True, refine: bool = True, signal_mask=None, n_regions: int = 10, gaussian_background: bool = False, circular_mask: bool = True, pseudo_symmetry_ops: Rotation | None = None): ...
    def index_patterns(self, patterns: np.ndarray, n_best: int = 1) -> dict[str, np.ndarray]   # keys: "zyz" (n, n_best, 3), "scores", "iq", "phase_id"
    def refine_orientations(self, patterns, rotations: Rotation) -> ...   # Feature 5
    def get_info_message(self, n_patterns, chunksize) -> str
def fast_bandwidths(bandwidth_min: int = 16, bandwidth_max: int = 512) -> np.ndarray
def rotate_harmonics(alm: np.ndarray, rotation: Rotation | np.ndarray) -> np.ndarray   # exported for tests/pseudo-symmetry

# EBSD method (src/kikuchipy/signals/ebsd.py)
def spherical_indexing(self, harmonics: MasterPatternHarmonics | list[MasterPatternHarmonics], detector: EBSDDetector, bandwidth: int = 68, normalize: bool = True, refine: bool = True, n_best: int = 1, navigation_mask: np.ndarray | None = None, signal_mask: np.ndarray | None = None, n_regions: int = 10, gaussian_background: bool = False, pseudo_symmetry_ops: Rotation | None = None, chunksize: int | None = None, verbose: int = 1) -> CrystalMap
# CrystalMap props: "scores" (n, n_best), "iq" (n,), phase_id from harmonics list; xmap.scan_unit via _get_navigation_axes_unit
```
Reused: `_get_navigation_axes_unit` (`ebsd.py:3380`), `_detector_is_compatible_with_signal`, `get_dask_array`/`get_chunking` (`src/kikuchipy/signals/util/_dask.py`), `create_coordinate_arrays` (orix), `Rotation.from_euler`, `orix.quaternion.Orientation.angle_with` (tests), `scipy.fft.ifft/irfft/next_fast_len`, `scipy.linalg.cho_solve` (F5), tqdm/`dask.diagnostics.ProgressBar` (as in `_dictionary_indexing.py:92-105`).
Tests
- Wigner (`test/sht/wigner.cpp` port): tables at `beta = pi/2, pi/3, 2pi/3` for j < 5, k,m in (-5,5) generated at test time with `sympy.physics.wigner.wigner_d_small` (or hard-coded values transcribed from the C++ file, `Num=5`) with tol `2*eps`; `_wigner_d_table` vs scalar `wigner_d` at `beta = 0.9708055194` for bw 15; `dTablePre` vs `dTable`; `pi/2` table transposed/untransposed; `rotate_harmonics` composition test (rotate by R1 then R2 == rotate by R2*R1 to 1e-10) and identity.
- xcorr (`test/sht/sht_xcorr.cpp` port): `randomSphere/randomPair` reimplemented with `np.random.default_rng(0)` on the Legendre grid via `SphericalHarmonicTransform`, exact zeroing of `m % nFold != 0` rows, `rotate_harmonics` by a random rotation, `SphericalCrossCorrelator.correlate(..., refine=False)`: misorientation to the true rotation < `360/(2*slP)` deg (half a grid cell) for bw in {53, 54, 55, 56, 58, 60, 62, 64, 68, 88, 113, 123}; normalized correlator with the wedge mask (`:212-287`) same tolerance; point groups {112, 11m, 112/m, 3, 4, 4/m, 6, 6/m} via `f_mirror/f_fold` from Feature 2 tables, compared with symmetry-reduced misorientation (`orix Orientation.angle_with`).
- Convention lock (synthetic real-data hybrid): `mp = nickel_ebsd_master_pattern_small(lambert, both, energy=20)`, `R_true = Rotation.random(20, seed)`, `sim = mp.get_patterns(R_true, detector, energy=20, compute=True)`, `SphericalIndexer(harmonics_from_mp, detector, bw=68, refine=False).index_patterns(sim.data)` -> `angle_with(R_true) < 2 deg` for all 20 (grid ~2.7 deg, interpolated). This decides the `~`/`quNp` question empirically.
- real data: `kp.data.nickel_ebsd_small()` (backgrounds removed as in `test_ebsd_hough_indexing.py:39-45`), `harmonics = MasterPatternHarmonics.from_master_pattern(nickel_ebsd_master_pattern_small(lambert, both), bandwidth=68)`, `s.spherical_indexing(harmonics, s.detector, bandwidth=68, refine=False)`: `angles = xmap.orientations.angle_with(s.xmap.orientations, degrees=True)`; assert `np.median(angles) < 1.5` and `np.sum(angles < 3) >= 8`; scores in (0, 1.2) and monotonic-ish with stored NCC (Spearman > 0.5 is optional). Test also `n_best=3`, `navigation_mask`, lazy signal, `verbose=0`, wrong detector shape -> `ValueError`, print message text.
- Regression vs EMSphInx (`IndexEBSD.exe`, `refine=.FALSE.`, `normed=.TRUE.`, `nregions=0`) reference for `nickel_ebsd_small` (9 points, in-package `.npz`): misorientation < 0.5 deg for all 9 (both interpolate the same 3x3x3 neighbourhood; differences arise only from tiny numerical noise in preprocessing/back-projection resampling), metric abs diff < 0.02.
Acceptance: all above; throughput >= 5 patterns/s per core at bw 68 (60x60 patterns), measured by the benchmark; memory per worker < 150 MB at bw 88.
Validation: `uv run pytest tests/test_indexing/test_spherical_wigner.py tests/test_indexing/test_spherical_xcorr.py tests/test_signals/test_ebsd_spherical_indexing.py -n 4`; `uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py` (added here).
CHANGELOG: `- Spherical indexing of EBSD patterns, ``EBSD.spherical_indexing()`` and ``kikuchipy.indexing.SphericalIndexer``, a CPU port of EMSphInx's spherical harmonic cross-correlation indexing (Lenthe et al. 2019). (#NNN)`
Docs: docstring `See Also` linking `dictionary_indexing`, `hough_indexing`, `refine_orientation`; `Notes` with `:cite:`lenthe2019spherical`` and the bandwidth guidance (fast values 53, 63, 68, 74, 88, 95, 113, 122, 123, 158, ...; range [16, 512]).

### Feature 5 - Newton refinement (branch `spherical-refinement`, spec `specs/2026-08-16-spherical-refinement/`)
New/changed files
- `_wigner.py`: `_d_prime`, `_d_prime2` scalar checks (`wigner.hpp:814-852`) for tests.
- `_xcorr.py`: `_derivatives(flm, gln, zyz, bw, f_mirror, f_fold, wig_e, wig_w, wig_b, work_dtable, compute_derivatives) -> (xc, jac(3), hes(3,3))` (`sht_xcorr.hpp:889-1119`), `_refine_peak(...)` (`:442-499`: Cholesky via a tiny 3x3 numba solver, monotone step, degeneracy fallbacks, `max_iter=15`, `abs_eps = eps*2*pi/slP`), normalized `refine_peak` dividing by `denominator(eu)` (`:1169-1172`); `SphericalCrossCorrelator.correlate(refine=True)`.
- `_indexer.py`: `refine=True` path, `SphericalIndexer.refine_orientations(patterns, rotations, phase_id=None) -> dict` (port of `refineImage`).
- `ebsd.py`: `spherical_indexing(refine=True)` default now honoured; new method `spherical_refine_orientation(self, xmap: CrystalMap, harmonics, detector, navigation_mask=None, signal_mask=None, eps: float = 0.01, chunksize=None) -> CrystalMap` (refine-only from an existing map, EMSphInx `msk & 0x02` path).
- Tests: `tests/test_indexing/test_spherical_xcorr.py::TestRefine` (`test/sht/sht_xcorr.cpp` full port: recovered rotation < `cbrt(FLT_EPSILON) ~ 4.9e-3 deg` for bw sizes {53, 68, 88, 113, 123, 158, 54..64}; normalized within `10*eps`; symmetric groups within `sqrt(eps)*5 ~ 0.012 deg`), derivative tables at `beta = pi/3, 2pi/3` vs `sympy` finite differences (`24*eps`), saddle rejection test (start on a saddle of a synthetic function -> returns unrefined orientation), beta = 0 degeneracy branch; `tests/test_signals/test_ebsd_spherical_indexing.py::TestRefine`: `nickel_ebsd_small` refined: `np.all(angles < 1.0)` vs stored xmap and mean score increases vs `refine=False`; `spherical_refine_orientation(s.xmap, ...)` on the stored xmap moves orientations by < 1 deg and does not decrease scores. Weekly: `nickel_ebsd_large(allow_download=True)`, bw 68, `pc = pc_average`: `>= 95%` of 4125 points within 1.5 deg of the stored xmap; vs the EMSphInx `IndexEBSD.exe` reference (`refine=.TRUE.`): median < 0.2 deg, 98% < 1 deg, metric Pearson r > 0.98. Default suite runs `s.inav[::5, ::5]` (~165 patterns) with the same thresholds relaxed to 95%/1 deg vs EMSphInx.
- Benchmark: `benchmarks/indexing/test_spherical_indexing.py::test_spherical_indexing_refine`.
Acceptance: above; refinement < 3x the coarse time per pattern (EMSphInx ratio is ~1.7x).
Validation: `uv run pytest -k "spherical and refine" -n 4`; `uv run pytest --weekly -k spherical`.
CHANGELOG: `- Newton refinement of spherical indexing results (``refine=True``) and ``EBSD.spherical_refine_orientation()``. (#NNN)`

### Feature 6 - Pseudo-symmetry (branch `spherical-pseudo-symmetry`, spec `specs/2026-08-16-spherical-pseudo-symmetry/`)
New files: `src/kikuchipy/indexing/_spherical/_pseudo_symmetry.py` (`MasterXcorr` port `programs/master_xcorr.cpp`): `find_pseudo_symmetry_operators(harmonics1: MasterPatternHarmonics, harmonics2: MasterPatternHarmonics | None = None, bandwidth: int = 88, cutoff: float = 0.5, merge_angle: float = 2.0, refine: bool = True, exclude_symmetry: bool = True, symmetry_cos_cutoff: float = 0.999) -> tuple[Rotation, np.ndarray, np.ndarray]` (operators as `Rotation`, relative intensities, full xc volume `(bw, 2bw-1, 2bw-1)` in Bunge or ZYZ order documented), `save_pseudo_symmetry_volume(volume, filename)` (HDF5 `Cross Correlation` + XDMF, optional), plus a `pseudo_symmetry_ops` path in `SphericalIndexer.index_patterns` (apply `q0 * q` in the crystal frame, refine, keep top-n, `indexer.hpp:216-270`) and CrystalMap prop `"pseudo_symmetry_index"` (kikuchipy naming from `_refinement.py:122-128`).
Changed: `src/kikuchipy/indexing/__init__.pyi`, `ebsd.py` (`pseudo_symmetry_ops` in `spherical_indexing`), `CHANGELOG.rst`.
Tests: `tests/test_indexing/test_spherical_pseudo_symmetry.py`: Ni m-3m autocorrelation (`nickel_ebsd_master_pattern_small`, bw 53): all maxima above cutoff 0.9 are within 2 deg of `Oh.proper_subgroup` operators (24), and with `exclude_symmetry=True` returns an empty `Rotation`; synthetic 3-fold function with an imposed approximate 6-fold (ring symmetrisation helper) yields a 60 deg z-rotation operator with intensity in (0.5, 1); two-phase mode returns finite intensities; `spherical_indexing(..., pseudo_symmetry_ops=ops)` on `nickel_ebsd_small` with a deliberately wrong pseudo op returns `pseudo_symmetry_index == 0` everywhere. Optional local-only test with `c:/Users/westraadt.1/Repos/openECCI_RKD/data/Mg-master-17kV.h5` (hcp) skipped when the file is missing.
Acceptance/validation: `uv run pytest tests/test_indexing/test_spherical_pseudo_symmetry.py`.
CHANGELOG: `- ``kikuchipy.indexing.find_pseudo_symmetry_operators()`` predicting pseudo-symmetry from master pattern autocorrelation (EMSphInx MasterXcorr), and ``pseudo_symmetry_ops`` support in spherical indexing. (#NNN)`

### Feature 7 - Visualisation and utility equivalents (branch `spherical-visualisation`, spec `specs/2026-08-16-spherical-visualisation/`)
New/changed files: `_master_pattern_harmonics.py` (`to_master_pattern(dim, projection in {"lambert","legendre","stereographic"})` using proper Legendre->Lambert regrid and `EBSDMasterPattern` so `mp.plot()`, `plot_spherical()` work; `plot_power_spectrum(ax=None)`; `describe() -> str` mirroring the `sht2png` header dump), `_back_projection.py` (`SphericalBackProjector.plot(pattern, hemisphere="upper", projection="stereographic")` returning a figure of the back-projected pattern on the sphere), `_xcorr.py` (`SphericalCrossCorrelator.plot_volume(xc, slice="Phi"|..., ...)`/`extract_bunge`), docs section mapping `PatternRepack` -> `kp.load(...).save("*.h5")`/`EBSD.downsample`, `EBSPDims` -> oxford_binary reader (`src/kikuchipy/io/plugins/oxford_binary/_api.py`) which already exposes scan shape; `ShtWisdom` -> `fast_bandwidths()`. Tests: `tests/test_indexing/test_spherical_visualisation.py` (figure creation, `to_master_pattern("stereographic")` correlates > 0.98 with `nickel_ebsd_master_pattern_small(projection="stereographic")` after resampling; `describe()` for the Ni `.sht` contains the exact strings from report section 8). Example script `examples/visualization/plot_master_pattern_harmonics.py` (Sphinx-Gallery). CHANGELOG: `- Plotting of master pattern harmonics, back-projected patterns and cross-correlation volumes. (#NNN)`.

### Feature 8 - Tutorial, data, docs (branch `spherical-indexing-tutorial`, spec `specs/2026-08-16-spherical-indexing-tutorial/`)
Files: `doc/tutorials/spherical_indexing.ipynb` (outline below; hidden first cell, thumbnail tag, black 77, outputs stripped), `doc/tutorials/index.rst` (add `spherical_indexing` to the Indexing nbgallery after `hybrid_indexing`), `doc/tutorials/run_nbval.sh` (only if outputs are stored), `doc/tutorials/tutorials_sanitize.cfg`, `doc/tutorials/load_save_data.ipynb` (one `.sht` cell - beware local modification), `doc/user/related_projects.rst`, `doc/user/bibliography.bib`, `src/kikuchipy/data/{_data.py,_registry.py,__init__.pyi}` (register `emsphinx/ni_20kv_bw384.sht`, `emsphinx/ni_small_20kv_bw384.sht`, `emsphinx/reference/ni_small_bw68.npz`, `emsphinx/reference/ni_large_bw68.npz` (~82 kB) — in-package if total < 300 kB, else `kikuchipy-data` repo with permalink URLs), `src/kikuchipy/data/emsphinx/create_emsphinx_reference.py` (script + `--ignore` in pyproject `addopts`), `benchmarks/indexing/test_spherical_indexing.py` (final), `CHANGELOG.rst` (consolidate), `README.rst` feature bullet if one exists.
Notebook outline (mirrors `hough_indexing.ipynb`, cell numbers approximate):
1. hidden MD boilerplate; `# Spherical indexing` intro (SHT, cross-correlation, cite Lenthe 2019 links, GPL/EMSphInx acknowledgement, no optional deps)
2. imports (`kp`, `orix plot/Phase/PhaseList/Vector3d`, matplotlib rcParams)
3. `s = kp.data.nickel_ebsd_large(allow_download=True)`; `## Pre-indexing maps` (VBSE RGB, static/dynamic background, IQ map)
4. `## The master pattern on the sphere`: `mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")`; `mp.plot()`; `harm = kp.indexing.MasterPatternHarmonics.from_master_pattern(mp, bandwidth=384)`; `harm.plot_power_spectrum()`; `harm.to_master_pattern(projection="stereographic").plot()`; bandwidth truncation round-trip at 53/88/158; `harm.save(".../ni.sht")`, `kp.load(".sht")`
5. `## Calibrate detector-sample geometry` (as in hough tutorial: `EBSDDetector(sig_shape, sample_tilt=70)`, PC from `s.detector.pc_average` or `hough_indexing_optimize_pc`; explain that back-projection needs the PC like DI)
6. `## Back-projection`: `bp = kp.indexing.SphericalBackProjector(det, bandwidth=68)`; `bp.plot(s.inav[0,0].data)`; explain circular mask / AHE options
7. `## Perform indexing`: `xmap = s.spherical_indexing(harm, det, bandwidth=68, normalize=True, refine=True)`; `xmap`; timing table for bw 53/68/88 (`fast_bandwidths()`)
8. `## Validate indexing results`: score + IQ maps and histograms; IPF-X/Y/Z maps with `IPFColorKeyTSL`; geometrical simulations overlay via `KikuchiPatternSimulator.on_detector`; navigator
9. `## Compare to Hough and dictionary indexing`: `angle_with(s.xmap)` histogram, speed table
10. `## Pseudo-symmetry`: `find_pseudo_symmetry_operators(harm, bandwidth=88)` on Ni (empty) — mention low-symmetry use
11. `## What's next?`
Validation: `./doc/tutorials/run_nbval.sh` (if outputs stored) or `jupyter nbconvert --execute doc/tutorials/spherical_indexing.ipynb`; `cd doc && make html && make linkcheck`.
CHANGELOG: `- Tutorial on spherical indexing. (#NNN)`.

---

## C. Algorithm-level design

### Data structures
- SH coefficients: `alm` complex128 array of shape `(bw, bw)` indexed `alm[m, l]` (m-major, l-minor, exactly EMSphInx `alm[m*bw + l]`); only `m >= 0`, entries `l < m` are zero; negative orders via `a^l_{-m} = (-1)^m conj(a^l_m)`; fully normalised (`a_00 = sqrt(4 pi)` for f = 1), no Condon-Shortley in the ALF recursion, hence the `(-1)^m` applied to odd `m` in analyze/synthesize (`square_sht.hpp:439, 554`). Documented in `SphericalHarmonicTransform` class docstring with the `.sht` packing rules.
- Sphere samples: two `(dim, dim)` float64 arrays (north, south), row = Y index, col = X index, `X = i/(dim-1)`; equator ring shared by both. `dim = bw + (3 if bw even else 2)` for Legendre. Precomputed per `dim`: `cos_lats (Nt,)`, `ring_index_starts (Nt+1,)`, `ring_flat_indices (sum 8y + 1,)`, `ring_dft_cos/sin` (list of `(8y, mLim)` matrices or one padded `(Nt, 8*(Nt-1), bw)` array — padded array is simplest for numba), `amn, bmn (bw, bw)`, `wy (Nw, Nt)`.
- Cross-correlation: `fxc` complex128 `(slP, slP, bwP)` with axes `(k=beta, n=gamma, m=alpha)` and the half-complex `m` axis last (matches `fxc[k][n][m]`), `xc` float64 `(bwP, slP, slP)`.
- Wigner tables: `wig_d_pi2 (bw, bw, bw)` transposed layout `[m, k, j]`; `dtable (bw, bw, bw, 2)` for arbitrary beta; `wig_e (bw, bw)`, `wig_w`, `wig_b (bw, bw, bw)`; entries with `j < max(k, m)` unused (never read).
- Results: per pattern `zyz (n_best, 3)`, `score (n_best,)`, `iq`, `phase_id (n_best,)`; converted to `Rotation` and `CrystalMap(prop={"scores", "iq", "pseudo_symmetry_index"?})`.

### SHT implementation choice: own numpy/numba (not pyshtools)
- pyshtools (BSD-3, wheels) uses Driscoll-Healy/GLQ equiangular grids, not EMSphInx's square Legendre grid; the indexer's back-projection LUT, ring weights, `toLegendre`, `.sht` normalisation and unit-test tolerances all assume the square grid. Adding pyshtools would introduce a compiled dependency without covering the grid, and would still leave Wigner-d/SO(3) correlation to be written. Therefore port `DiscreteSHT` directly: ~400 lines of numba, validated by the round-trip test and `sph_harm_y`.
- Ring DFTs: inside the numba kernel via precomputed cosine/sine matrices per ring (`8y x mLim`, `mLim = min(bw, 4y+1)`), so `analyze/synthesize` are single `@njit(nogil=True)` calls with no Python-level per-ring loop; cost is negligible against the xcorr (bw 113: ~1e7 flops per hemisphere; bw 384 for master patterns: ~2e8, sub-second). A `scipy.fft.rfft`-based reference implementation lives in the tests to cross-check the DFT-matrix kernel to 1e-12. Rationale: keeps the per-pattern pipeline (preprocess -> unproject -> analyze) GIL-free for dask threads and avoids 2*Nt tiny FFT calls per pattern.
- Quadrature weights: port `computeWeightsSkip` with `np.linalg.solve` (Legendre layout only needs skip = 0; Lambert layout computes `Nw = (dim-2)//4 + 1` sets); assert `sum(what) - 1 < cbrt(eps)/64` like EMSphInx.

### Cross-correlation FFT plan (scipy.fft)
- `slP = scipy.fft.next_fast_len(2*bw - 1, real=True)` (pocketfft's fast sizes are {2,3,5,7,11}-smooth; EMSphInx also allows 13; verify identical for the recommended list, else implement the EMSphInx `fastSize`) and `bwP = slP//2 + 1`.
- Symmetry reduction exactly as `Correlator::compute`: loop `k in [0, bw)`, precompute `fm[m, j] = flm[m, j] * d^j_{k,m}(pi/2)`, `gn[j] = conj(gln[n, j]) * d^j_{n,k}(pi/2)`, skip rows `n >= bw`, columns `m % f_fold != 0`, and step `j` by 2 when `f_mirror`; write the four mirrored entries `(k,n)`, `(-k,-n)`, `(-k,n)`, `(k,-n)` with the parity signs; zero-pad `k in [bw, slP-bw]`. This is one numba kernel (`_xcorr_spectrum`), O(bw^4/(f_fold*2)) complex mults; ~5-15 ms at bw 68 for m3m.
- Inverse transform, unnormalised (FFTW equivalence): `t = scipy.fft.ifft(fxc, axis=0, norm="forward")[:bwP]` (only the stored half of beta needed, halving the remaining work like `SepRealFFT3D::inverse`), `t = scipy.fft.ifft(t, axis=1, norm="forward")`, `xc = scipy.fft.irfft(t, n=slP, axis=2, norm="forward")`; optionally restrict axis-0/1 transforms to the `m % f_fold == 0` planes and scatter. `workers=1` inside dask threads. All FFT calls release the GIL.
- Peak: `np.argmax` -> `_extract_neighborhood_27` (periodic wrap + glide `R(a,b,g) = R(a+pi,-b,g+pi)`) -> tri-quadratic `_interpolate_maxima` (fix the `x[2]` bounds-check bug) -> `zyz`. Refinement (F5): `_derivatives` at `zyz` with `dTablePre` (bw^3 x 2 doubles per call, ~15 calls max), Newton with 3x3 Cholesky, saddle rejection, degeneracy fallbacks.
- Normalised correlator: precompute `rDen` once per (harmonics, detector, bw) via two `compute()` calls with the window harmonics `mlm` and `flm2 = SHT(f^2)` (synthesise, square, analyse), then per pattern `xc *= rDen` fused with the argmax.

### Threading / execution
- `EBSD.spherical_indexing` reshapes patterns to `(n_patterns, sig)` dask array with `chunksize` patterns per chunk (default from `get_chunking(nav_dim=1, sig_dim=1, dtype=float32)`, like `_prepare_patterns_for_refinement`), builds one `SphericalIndexer` (holds all precomputed tables; read-only, shared across threads), and calls `da.map_blocks(_index_chunk, patterns, indexer=..., dtype=float64, drop_axis=1, new_axis=1)`; each chunk loops over its patterns in Python calling `nogil` numba kernels and `scipy.fft`, so the dask threaded scheduler saturates cores (kikuchipy idiom, `_refinement.py:391-424`). Progress with `dask.diagnostics.ProgressBar` and a final `"  Indexing speed: {x:.5f} patterns/s"` print (matches `tutorials_sanitize.cfg` regex2). Per-thread scratch buffers (`fxc`, `xc`, `dtable`, sphere arrays) allocated per chunk (or cached per thread via `threading.local`) to avoid races. No numba `parallel=True`; no processes (avoids Windows spawn/pickling of tables).
- Batch estimate: expose `chunksize`; default such that `n_chunks >= 4 * n_workers`.

### Memory (per worker, float64/complex128, m3m unaffected)
- bw = 63: `slP = 125` (5^3), `bwP = 63`: `fxc` 125*125*63*16 B = 15.8 MB, `xc` 7.9 MB, `wig_d_pi2` 63^3*8 = 2.0 MB, `dtable` 4.0 MB, `wig_w/wig_b` 4.0 MB, `rDen` 7.9 MB (shared), tables for SHT < 1 MB -> ~35 MB scratch + 8 MB shared.
- bw = 88: `slP = 175` (5^2*7), `bwP = 88`: `fxc` 43.1 MB, `xc` 21.6 MB, `wig_d_pi2` 5.5 MB, `dtable` 10.9 MB, `wig_w/wig_b` 10.9 MB, `rDen` 21.6 MB -> ~85 MB scratch + 27 MB shared.
- bw = 113: `slP = 225` (3^2*5^2), `bwP = 113`: `fxc` 91.5 MB, `xc` 45.8 MB, `wig_d_pi2` 11.5 MB, `dtable` 23.1 MB, `wig_w/wig_b` 23.1 MB, `rDen` 45.8 MB -> ~180 MB scratch + 57 MB shared. With 8 dask threads: bw 113 ~1.5 GB, bw 88 ~0.7 GB, bw 63 ~0.3 GB (patterns themselves are negligible). Document in the `spherical_indexing` docstring; expose `bandwidth` guidance and `chunksize`.
- Time model (from EMSphInx CPU log: 655 pat/s on 20 threads at bw 55 => ~30 ms/pattern/thread with SHT 20%, coarse 30%, refine 50%): Python/numba target 50-100 ms/pattern/thread at bw 68 (>= 10 pat/s/core), i.e. `nickel_ebsd_large` (4125) in ~1 min on 8 cores.

### Euler conventions -> orix
- Correlation grid to ZYZ: `alpha = 2*pi*m/slP - pi/2`, `beta = 2*pi*k/slP - pi`, `gamma = 2*pi*n/slP - pi/2` (`sht_xcorr.hpp:580-590`).
- ZYZ -> Bunge ZXZ: `(phi1, Phi, phi2) = (alpha - pi/2, beta, gamma + pi/2)` (`rotations.hpp:1025-1039`), wrapped to [0, 2pi).
- EMSphInx then does `zyz2qu` (== `Rotation.from_euler(bunge)` in orix, verified byte-identical `eu2qu`), multiplies by `quNp` (identity today) and conjugates: `R_sample_to_crystal = ~Rotation.from_euler(bunge_from_zyz)`. Because kikuchipy back-projects with `_get_direction_cosines_from_detector` (sample-frame directions, includes `tilt`, `azimuthal`, `twist`), no `quNp` is needed. The final sign (`~` or not) is pinned by the forward-projection convention test in Feature 4 (`mp.get_patterns` at known `R_true`) and by the real-data test against `nickel_ebsd_small.xmap`; the code path is a single helper `_zyz_to_rotation` so a convention error is a one-line fix.
- Pseudo-symmetry ops are applied in the crystal frame: `q0 * q_ps` on the crystal->sample quaternion before conjugation, mirroring `indexer.hpp:249-256`; in orix terms `~( (~R0) * ops )` — again isolated in one helper and unit-tested with `refine_orientation`'s `pseudo_symmetry_ops` semantics (`_refinement.py`) so both features agree.

---

## D. Real-data test strategy

Datasets (kikuchipy)
- `kp.data.nickel_ebsd_small()` (in package, 9 x 60x60, stored ground-truth `xmap` (HI + refinement), per-point PC, `sample_tilt=70`, `tilt=0`, `binning=8`) - default suite for every real-data test.
- `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")` (in package, 401x401 uint8, single 20 keV, both hemispheres, `accum_e` present) - default suite master pattern.
- `kp.data.nickel_ebsd_large(allow_download=True)` (15 MB, downloaded at pytest session start by `conftest.py:105`) - subset `inav[::5, ::5]` in default suite, full 4125 in `@pytest.mark.weekly`.
- `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both")` (305 MB, already in the `develop` cache) - weekly only (mp2sht parity at bw 384 with MC energy weighting).
- Local-only (skipped if absent): `c:/Users/westraadt.1/Repos/openECCI_RKD/data/{Mg-master-17kV.h5, Ti-alpha-master-20kV.h5}` for hcp symmetry paths.

EMSphInx reference outputs (generated once by `src/kikuchipy/data/emsphinx/create_emsphinx_reference.py`, run manually on this machine; commit hash of the EMSphInx build recorded in the `.npz` metadata):
1. Master patterns:
   - `"c:/Users/westraadt.1/Repos/EMSphInx/build/Release/mp2sht.exe" "C:/Users/westraadt.1/AppData/Local/kikuchipy/kikuchipy/Cache/develop/data/ebsd_master_pattern/ni_mc_mp_20kv.h5" ni_20kv_bw384.sht`
   - `"c:/Users/westraadt.1/Repos/EMSphInx/build/Release/mp2sht.exe" "c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5" ni_small_20kv_bw384.sht`
   - `"c:/Users/westraadt.1/Repos/EMSphInx/build/Release/sht2png.exe" ni_small_20kv_bw384.sht ni_small_sqleg.png ni_small_stereo.png` (image regression for Feature 7, and its stdout dump saved as `ni_small_sht_dump.txt` for `describe()` tests).
2. Patterns repacked for EMSphInx (h5py): for `nickel_ebsd_small` and `nickel_ebsd_large`, apply kikuchipy `remove_static_background()` + `remove_dynamic_background()` (uint8), then write `/Manufacturer = "Bruker"` (vlen str) and `/patterns` `(n, 60, 60)` uint8 contiguous, uncompressed. Vendor `Bruker` means no vertical flip and the Bruker PC convention, which is kikuchipy's internal convention (`EBSDDetector.pc`), so `pctr = pc_average` verbatim. Preprocessing on the EMSphInx side disabled (`nregions = 0`, `gausbckg = .FALSE.`) so both implementations see identical inputs; `circmask = 0` (largest inscribed circle, matched by `signal_mask`/`circular_mask=True`).
3. Namelist `ni_large_bw68.nml` (and `ni_small_bw68.nml` with `scandims = 3, 3, 1.5, 1.5`, `pctr = 0.4251, 0.2134, 0.5007`); each in two variants `refine = .FALSE.`/`.TRUE.`:
```
 &EMSphInx
 patfile    = 'ni_large_emsphinx.h5',
 patdset    = 'patterns',
 masterfile = 'ni_20kv_bw384.sht',
 patdims    = 60, 60,
 circmask   = 0,
 gausbckg   = .FALSE.,
 nregions   = 0,
 delta      = 500,          ! any value giving 5-90 mm width; Bruker PC is fractional so geometry is unchanged
 pctr       = 0.42326, 0.21363, 0.50207,
 vendor     = 'Bruker',
 thetac     = 0,            ! sample tilt 70 comes from the .sht primaryAngle
 scandims   = 75, 55, 1.5, 1.5,
 roimask    = '',
 bw         = 68,
 normed     = .TRUE.,
 refine     = .TRUE.,
 nthread    = 1,
 batchsize  = 1,
 datafile   = 'ni_large_bw68_ref.h5',
 vendorfile = 'ni_large_bw68_ref.ang',
 ipfmap     = 'ni_large_bw68_ipf.png',
 qualmap    = 'ni_large_bw68_xc.png',
 /
```
   Run: `"c:/Users/westraadt.1/Repos/EMSphInx/build/Release/IndexEBSD.exe" ni_large_bw68.nml` (~10 s). Convert `Scan 1/EBSD/Data/{Phi1,Phi,Phi2,Metric,IQ}` from the output h5 into `ni_large_bw68.npz` (float32, ~82 kB) with attrs `{emsphinx_commit: "60f3517", bw: 68, normed: True, refine: True, pc: [...], nregions: 0}`; same for `nickel_ebsd_small` and for `refine=False`.
   Note: EMSphInx `Metric` is not divided by the pattern norm (report 3.10) and is comparable across patterns for a fixed geometry; kikuchipy will report the same quantity as `scores` so both are directly comparable.
4. `MasterXcorr.exe 88 0.5 ni_mc_mp_20kv.h5` (argv order: bw cutoff master) -> stdout list of `(intensity, quaternion)` saved as `ni_pseudo_sym_bw88.txt` for Feature 6 (`pseudo_sym.h5` volume too large to ship; compare peaks only).
Where reference files live: `.sht` files and `.npz` references in-package under `src/kikuchipy/data/emsphinx/` (each < 100 kB, total < 400 kB) registered in `_registry.py` `_registry_hashes` (in-package, no URL); if the maintainers object to size, move to `pyxem/kikuchipy-data` (`emsphinx/`) with commit-pinned raw URLs and `allow_download=True`. Regenerated `IndexEBSD.nml`-style files and the generation script live in `src/kikuchipy/data/emsphinx/` and are `--ignore`d by pytest.
Tolerances (using `orix Orientation.angle_with` with the phase point group):
- SHT round trip: per-coefficient < 5e-3, mean < 5e-5 (EMSphInx double).
- Synthetic recovery: coarse < half grid cell (`180/slP` deg); refined < 4.9e-3 deg (`cbrt(FLT_EPSILON)`); symmetric groups < 0.012 deg.
- vs EMSphInx reference (same inputs): coarse all 9 < 0.5 deg; refined median < 0.2 deg, 98% < 1 deg (large), all < 1 deg (small); scores: Pearson r > 0.98, mean |diff| < 0.02.
- vs kikuchipy stored xmap (different algorithm, per-point PC): small: median < 1.5 deg coarse / all < 1 deg refined; large: >= 95% < 1.5 deg refined.
- mp2sht parity: relative L2 < 1e-3 on `alm` after `remove_dc()`; `.sht` header/bytes exact.
- Fraction indexed: `xmap.is_indexed.all()` and no NaN scores; `phase_id == 0` everywhere for single phase.

---

## E. Risks and open questions

Risks (with mitigation)
1. Per-pattern speed of the 3-D correlation in Python (bw^4 kernel + FFT + refinement): mitigate by (a) numba `nogil` kernels + dask threads (no GIL contention), (b) `norm="forward"` and slicing to `bwP` beta slices before the axis-1/2 transforms, (c) skipping `m % f_fold != 0` planes, (d) benchmark gate at Feature 4 (>= 5 pat/s/core at bw 68) before investing in refinement, (e) fallbacks: float32/complex64 path, coarse-only default for large maps, `n_best=1` fast path, chunk-batched FFTs (`scipy.fft` over a stacked pattern axis with `workers`).
2. Numba + FFT: no FFT inside `njit`; ring DFTs use precomputed matrices, the 3-D inverse uses `scipy.fft` outside numba; JIT compile time of ~10 kernels (~20-40 s first run) mitigated by `cache=True` and small kernels; Windows numba cache path issues -> CI runs on windows-latest already.
3. Numerical conventions (Condon-Shortley sign, `(-1)^m` in analyze, ZYZ glide, `~` on the final rotation, Bruker vs EMsoft PC): mitigated by porting EMSphInx unit tests first and by the forward-projection convention test; each convention lives in one helper.
4. Bandwidth choices and grid parity: enforce `dim` odd, `bw <= dim-2`, `bw in [16, 512]`, `fast_bandwidths()` documented; the `next_fast_len` (2,3,5,7,11) vs EMSphInx (adds 13) difference could change `slP` for a few bw values -> implement EMSphInx's `fastSize` to keep grid parity for regression tests.
5. Windows/CI: 15-minute CI timeout -> keep default suite < 2 min for spherical tests (small dataset, bw <= 68, subsets), heavy tests weekly; avoid multiprocessing; large tables allocated per chunk to avoid thread races.
6. Legal: copyright is clean (GPL-2.0-or-later -> GPL-3.0-or-later) but the CMU provisional patent status is unknown; keep attribution notices, flag in the first upstream PR/issue (EMSphInx issue #7 by hakonanes shows maintainer interest), do not place code in BSD areas.
7. Reference data provenance: `IndexEBSD.exe` on this machine is built from `master@60f3517` (benchmark artefacts are from `feature/GPU`, unusable); reference `.npz` must record the commit and inputs; the CPU vs CUDA metric discrepancy shows tolerances (not equality) are required.

Open questions for the user (max 6)
1. Should `specs/` folders be committed inside the kikuchipy fork (and later stripped for upstream PRs), or kept in a separate `kikuchipy-specs` repo? And are PRs targeted at your fork's `develop` first, with an upstream issue opened at Feature 4?
2. Public naming: `MasterPatternHarmonics`, `SphericalHarmonicTransform`, `SphericalBackProjector`, `SphericalIndexer`, `EBSD.spherical_indexing`, `EBSD.spherical_refine_orientation`, `find_pseudo_symmetry_operators` - acceptable, or do you prefer EMSphInx names (`MasterSpectra`, `DiscreteSHT`, `Correlator`)?
3. Preprocessing default: replicate EMSphInx (`n_regions=10` mosaic AHE, circular mask, optional Gaussian background) inside `spherical_indexing`, or default to "no internal preprocessing" and rely on kikuchipy's existing `remove_*_background`/`adaptive_histogram_equalization` (cleaner API, but breaks 1:1 parity with `IndexEBSD` defaults)?
4. Reference data location: in-package (`src/kikuchipy/data/emsphinx/`, ~400 kB total) versus the `pyxem/kikuchipy-data` repo (requires a PR there and network in tests). Which do you want for the first PRs?
5. May the EMSphInx-shipped `data/Ni {20kV 75.7deg}.sht` be redistributed inside kikuchipy as a parse fixture (GPL-compatible, but provenance/credit needs a note), or should tests only use `.sht` files we generate from kikuchipy's own EMsoft master patterns?
6. Performance target and float precision: is ~10 patterns/s/core at bw 68 (nickel_ebsd_large in ~1 min on 8 cores) acceptable for v1, and may Feature 4 add a `dtype="float32"` option later rather than now?

### Critical Files for Implementation
- c:/Users/westraadt.1/Repos/EMSphInx/include/sht/square_sht.hpp (grid, weights, analyze/synthesize to port)
- c:/Users/westraadt.1/Repos/EMSphInx/include/sht/sht_xcorr.hpp (correlation, peak interpolation, derivatives, Newton refinement)
- c:/Users/westraadt.1/Repos/EMSphInx/include/modality/ebsd/detector.hpp (back-projection LUT and normalisation) together with c:/Users/westraadt.1/Repos/EMSphInx/include/idx/master.hpp (MasterSpectra, toLegendre)
- c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/util/_master_pattern.py (reused Lambert and direction-cosine kernels)
- c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/ebsd.py (new `spherical_indexing`/`spherical_refine_orientation` methods next to `dictionary_indexing` L1827) plus c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/indexing/__init__.pyi (public exports)