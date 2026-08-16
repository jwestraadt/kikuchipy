# Inventory: real EBSD data & master patterns for tests + a `spherical_indexing` tutorial

> **ACTION REQUIRED (side effects of this survey — I could not delete them under the read-only mandate):**
> 1. `c:/Users/westraadt.1/Repos/kikuchipy/IndexEBSD.nml` (4582 B) was created by running `IndexEBSD.exe -t` (the "print usage" request; `-t` writes the template to CWD instead of stdout). **Delete it before committing.**
> 2. pooch downloaded `nickel_ebsd_large/patterns.h5` (15.4 MB) into two new cache dirs: `…/Cache/0.10.0/data/…` (system Python, kikuchipy 0.10.0) and `…/Cache/develop/data/…` (repo `.venv`, kikuchipy 0.13.dev0). This is exactly what `conftest.py::pytest_sessionstart` does anyway, so it is benign.

---

## 1. `kikuchipy.data` module inventory

### 1.1 Source files

| File | Purpose |
|---|---|
| `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/_data.py` | Dataset functions + `Dataset` class + pooch `marshall` (L32-41) |
| `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/_registry.py` | `_registry_hashes` (L23-72), `KP_DATA_REPO_URL` (L75), `_registry_urls` (L76-101) |
| `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/__init__.py` | lazy_loader stub attach (L36) |
| `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/__init__.pyi` | public API list (L18-38): 8 names |
| `c:/Users/westraadt.1/Repos/kikuchipy/doc/dev/adding_to_data_module.rst` | how to add data (27 lines total) |

pooch config (`_data.py` L32-41):
```python
marshall = pooch.create(
    path=pooch.os_cache("kikuchipy"),
    base_url="", version=__version__.replace(".dev", "+"), version_dev="develop",
    env="KIKUCHIPY_DATA_DIR", registry=registry_hashes, urls=registry_urls,
    retry_if_failed=5)
```
Because `0.13.dev0` → `0.13+0` (contains `+`), pooch uses `version_dev` → cache subdir is **`develop`** for the dev checkout.

### 1.2 Public dataset functions

| Function (`_data.py` line) | Returns | Nav × sig shape | dtype | Phase / kV | Detector / PC | Shipped? | Source |
|---|---|---|---|---|---|---|---|
| `nickel_ebsd_small` (L47) | `EBSD` | (3, 3 \| 60, 60) | uint8 | Ni (Fm-3m, a=3.5236 Å), 20 kV, NORDIF UF-1100, binning 8 | PC per point, shape (3,3,3); `pc_average≈(0.4251, 0.2134, 0.5007)`; `sample_tilt=70`, `tilt=0`, `px_size=1 µm`; step 1.5 µm | **YES, in package** `data/kikuchipy_h5ebsd/patterns.h5` (938 kB-ish; 2 scans "Scan 1"/"Scan 2") | md5 `f5e24fc5…` |
| `nickel_ebsd_large` (L79) | `EBSD` | (55, 75 \| 60, 60) = 4125 | uint8 | Ni, 20 kV, UF-1100 | PC array (55,75,3), `pc_average=[0.42326, 0.21363, 0.50207]`, sample_tilt 70, step 1.5 µm | NO → cache | GitHub `kikuchipy-data` @ `bcab8f7a…/nickel_ebsd_large/patterns_v2.h5` (15.4 MB) |
| `ni_gain(number=1..10)` (L125) | `EBSD` | (149, 200 \| 60, 60) | uint8 (NORDIF `Pattern.dat`, 107.3 MB) | Ni recrystallized, 20 kV, Hitachi SU-6600, WD 24.9 mm, tilt 70°, UF1100, step 1.5 µm, gains 0/3/6/9/12/15/17/20/22/24 dB | No stored PC; hybrid tutorial calibrates from `pc0=[0.42,0.22,0.50]` | NO → cache | Zenodo 10.5281/zenodo.7497682, files `zenodo.org/record/7498632/files/scanN_gainXdb.zip` (~100 MB zipped, unzip then zip deleted) |
| `ni_gain_calibration(number)` (L191) | `EBSD` | (9 \| 480, 480) | — | same, calibration patterns (gain 8, 480×480, exposure 99.5 ms) | `original_metadata.calibration_patterns.indices_scaled`, `roi`, `area` used by `plot_pattern_positions_in_map` | NO → cache (same zip) | same zip; file `ni_gain/N/Setting.txt` |
| `si_ebsd_moving_screen(distance=0\|5\|10)` (L259) | `EBSD` | (\| 480, 480) single pattern | uint8, has `static_background` | Si single crystal, 20 kV, NORDIF UF-420 | PC obtained by moving-screen technique in tutorial; `px_size=90e-3 mm/px`, `sample_tilt=70` | NO → cache | GitHub `kikuchipy-data` @ `bcab8f7a…/silicon_ebsd_moving_screen/si_{in,out5mm,out10mm}.h5` |
| `si_wafer` (L326) | `EBSD` | (50, 50 \| 480, 480) | NORDIF `Pattern.dat` 576 MB | Si wafer, 20 kV, ZEISS SUPRA55 VP, WD 16.1 mm, tilt 70°, UF420, step 40 µm | none stored | NO → cache | Zenodo 10.5281/zenodo.7491388, `ebsd_si_wafer.zip` (311 MB → 581 MB) |
| `nickel_ebsd_master_pattern_small` (L385) | `EBSDMasterPattern` | (401, 401) per hemisphere; `hemisphere="both"` → (2, 401, 401) | **uint8** | Ni, **single energy 20 keV** (`EkeVs=[20.]`, `numEbins=1`), EMsoft, `npx=200` | — | **YES, in package** `data/emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5` (~1 MB) | md5 `807c8306…` |
| `ebsd_master_pattern(phase)` (L447) | `EBSDMasterPattern` | (1001, 1001); both hemi + all energies → (2, nE, 1001, 1001) | float32 | see table below | — | NO → cache | Zenodo (see below) |

`ebsd_master_pattern` phases (`_data.py` L490-500, L513-521; URLs `_registry.py` L94-100):

| phase | file | energies [keV] | size | Zenodo |
|---|---|---|---|---|
| `ni` | `ebsd_master_pattern/ni_mc_mp_20kv.h5` | 5–20 (16 bins) | 0.3 GB (305,510,476 B measured) | `zenodo.org/record/7498645` (docstring says 10.5281/zenodo.7628443 — **docstring/URL mismatch**, and `steel_sigma` also lists 7628443) |
| `al` | `al_mc_mp_20kv.h5` | 10–20 | 0.2 GB | 7628365 |
| `si` | `si_mc_mp_20kv.h5` | 5–20 (16 bins) | 0.3 GB (305,523,212 B) | 7498729 |
| `austenite` | `austenite_mc_mp_20kv.h5` | 10–20 | 0.3 GB | 7628387 |
| `ferrite` | `ferrite_mc_mp_20kv.h5` | 5–20 | 0.3 GB | 7628394 |
| `steel_chi` | `steel_chi_mc_mp_20kv.h5` | 10–20 | 0.6 GB | 7628417 |
| `steel_sigma` | `steel_sigma_mc_mp_20kv.h5` | 5–20 | 1.5 GB | 7628443 |

### 1.3 Also shipped in the package (not exposed via `kp.data.*`, used by tests via `conftest.py` fixtures)
`c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/data/`:
- `edax_binary/edax_binary.up1` (32 kB), `edax_binary.up2` (72 kB) + generator script
- `edax_h5ebsd/patterns.h5` (939 kB)
- `emsoft_ebsd/EBSD_TEST_Ni.h5` (351 kB), `simulated_ebsd.h5` (36 kB)
- `emsoft_ebsd_master_pattern/` (Ni MP + `BetheParameters.nml` + `master_patterns.h5` referenced by `conftest.py` L595-596)
- `kikuchipy_h5ebsd/patterns.h5`, `nordif/`, `oxford_binary/`
- `_dummy_files/{bruker_h5ebsd.py, oxford_h5ebsd.py}` — generate synthetic vendor files at test time

### 1.4 kikuchipy-data repo & how to add a dataset
- Repo: **https://github.com/pyxem/kikuchipy-data** (cited in `_data.py` L108, L305 and `adding_to_data_module.rst` L10-11).
- Procedure (from `adding_to_data_module.rst` + `_data.py`):
  1. Put the file in the package (`src/kikuchipy/data/<subdir>/<file>`) if small, otherwise upload to `kikuchipy-data` (GitHub) or Zenodo.
  2. Add `"<subdir>/<file>": "md5:<hash>"` to `_registry._registry_hashes` (L23-72). Hash with `md5sum`.
  3. If not in package, add a **permalink** (commit-pinned raw.githubusercontent URL — see comment L73-74) or a Zenodo `record/<id>/files/<name>` URL to `_registry._registry_urls` (L76-101).
  4. Write a `def <name>(allow_download=False, show_progressbar=None, **kwargs)` in `_data.py` that builds `Dataset("<subdir>/<file>")` (or `Dataset(..., collection_name="<zip>")` for zipped collections, cf. `si_wafer` L377, `ni_gain` L185) and calls `.fetch_file_path(...)`.
  5. Export it in `__init__.pyi` (L18-38, both the import and `__all__`).
  6. Add a test to `tests/test_data/test_data.py` (mirror `test_ebsd_master_pattern_dataset`, L210-239) plus the weekly availability ping list (L241-273).
- **For an `.sht` file or an EMSphInx reference `.ang`/`.h5`**: an `.sht` (~75 kB for Ni at bw=384) is small enough to ship in-package (like the 1 MB Ni MP); a reference indexing result (`.ang`/`.h5`/`.npy` of Euler angles + metric) is also tiny. Either could be added as `data/emsphinx/<name>.sht` with an md5 in `_registry_hashes` and **no URL** (in-package datasets have `url is None`, cf. `test_data.py` L44). Note `Dataset.file_relpath` is `"data/" + relpath` and `registry_hashes` keys are prefixed with `data/` (`_registry.py` L104-110).

---

## 2. Local pooch cache (`%LOCALAPPDATA%\kikuchipy\kikuchipy\Cache`)

`pooch.os_cache("kikuchipy")` = `C:\Users\westraadt.1\AppData\Local\kikuchipy\kikuchipy\Cache`; `KIKUCHIPY_DATA_DIR` is **unset**.

| version dir | contents (non-`.bmp`) | size |
|---|---|---|
| `0.9.0/data/` | `ebsd_master_pattern/ni_mc_mp_20kv.h5` | 305,510,476 B |
| | `ebsd_master_pattern/si_mc_mp_20kv.h5` | 305,523,212 B |
| | `nickel_ebsd_large/patterns.h5` | 15,368,325 B |
| | `ni_gain/10/Pattern.dat` (+ `Setting.txt`, ~26 `.bmp`) | 107,280,000 B |
| | `si_wafer/Pattern.dat` (+ `Setting.txt`, ~14 `.bmp`) | 576,000,000 B |
| `develop/data/` (used by repo `.venv`, kp 0.13.dev0) | `ebsd_master_pattern/ni_mc_mp_20kv.h5` | 305,510,476 B |
| | `ni_gain/10/{Pattern.dat, Setting.txt, *.bmp}` | 107 MB |
| | `nickel_ebsd_large/patterns.h5` **(downloaded during this survey)** | 15.4 MB |
| `0.10.0/data/` (system Python) | `nickel_ebsd_large/patterns.h5` **(downloaded during this survey)** | 15.4 MB |

**Not cached** (would need download): `silicon_ebsd_moving_screen/*.h5`, `ni_gain/1..9`, all `ebsd_master_pattern/*` except `ni` (develop) and `ni`+`si` (0.9.0), `si_wafer` (0.9.0 only).

**Key point for SHT work:** the *full* Ni master pattern (`ni_mc_mp_20kv.h5`, 5–20 keV, 16 bins, 1001×1001, **both hemispheres**, with `EMData/MCOpenCL/accum_e` (501,501,16)) is **already in the `develop` cache** — no download needed for the dev checkout.

---

## 3. How tests and notebooks use real data

### 3.1 `conftest.py` (28,416 B)
- `DATA_PATH = Path(kp.data.__file__).parent.resolve()` (L58)
- `pytest_sessionstart` (L105-107): **downloads `nickel_ebsd_large` at session start** (`allow_download=True`) → any test may use it offline afterwards.
- `pytest_addoption`/`pytest_runtest_setup` (L64-83): `--weekly` marker gate (used by `test_dataset_availability`).
- Real-data fixtures: `ni_small_axes_manager` (L356), `ebsd_directory` (L378-401, writes 3×3 TIFs from `nickel_ebsd_small`), `kikuchipy_h5ebsd_path` (L404), `nickel_ebsd_large_h5ebsd_renamed` (L409-417, temporarily renames the cached file to force the "must be downloaded" error path), `emsoft_ebsd_master_pattern_file` (L595 → `DATA_PATH/emsoft_ebsd_master_pattern/master_patterns.h5`), `emsoft_ebsd_path` (L600), `nordif_path` (L824), `edax_*`/`oxford_*`/`bruker_*` paths & dummy-file factories.
- Synthetic fixtures: `dummy_signal` (L162-215, `<3,3|3,3>` hard-coded uint8 array — "If this is changed, all tests using this signal will fail"), `dummy_background`, `ebsd_with_axes_and_random_data`, `nickel_structure`/`nickel_phase` (L266-277), `pc1`/`detector` (L280-300), `rotations`, `get_single_phase_xmap`.

### 3.2 Test usage counts (`grep kp.data.` over `tests/`, 115 hits)
`nickel_ebsd_small` 59 · `nickel_ebsd_master_pattern_small` 34 · `nickel_ebsd_large` 9 · `si_ebsd_moving_screen` 5 · `si_wafer` 2 · `ni_gain_calibration` 2 · `ni_gain` 2 · `ebsd_master_pattern` 2.

### 3.3 `tests/test_indexing/test_dictionary_indexing.py` (181 lines)
**Uses only `dummy_signal` (3×3×3×3), no real data.** Assertions are all `np.allclose(xmap.scores[:,0], 1)` (self-matching dictionary), plus `pytest.raises` for invalid metric/shape/masks. Runtime ≈ instant. There is **no accuracy/fraction-correctly-indexed assertion anywhere in DI tests**.

### 3.4 Hough indexing tests: `tests/test_signals/test_ebsd_hough_indexing.py` (390 lines)
(no `tests/test_indexing/test_hough_indexing.py` exists)
- Whole module `skipif dependency_version["pyebsdindex"] is None`.
- `TestHoughIndexing.setup_method` (L39-45): `kp.data.nickel_ebsd_small()` → `remove_static_background()` + `remove_dynamic_background()`; `indexer = s.detector.get_indexer(s.xmap.phases)`.
- **The accuracy assertion** (`test_hough_indexing`, L75-87):
  ```python
  xmap = self.signal.hough_indexing(phase_list, indexer)
  angles = xmap.orientations.angle_with(xmap_ref.orientations, degrees=True)
  assert np.all(angles < 1)          # < 1 degree vs. the stored reference xmap
  ```
  i.e. the *stored* `EBSD.xmap` of `nickel_ebsd_small` (HI + refinement reference) is the ground truth.
- `test_hough_indexing_print_information` (L47-73) asserts the literal string `"  Projection center (Bruker, mean): (0.4251, 0.2134, 0.5007)"`.
- `TestPCOptimization` (L297-367): `assert np.allclose(det.pc_average, det0.pc_average, atol=0.05)`; PSO variant `@pytest.mark.flaky(reruns=3)` with `tol = 0.04`, uses `worker_id` (xdist).
- `TestHoughIndexingNoPyEBSDIndex` (L370-389): ImportError paths.

### 3.5 Refinement tests: `tests/test_indexing/test_ebsd_refinement.py` (~1250 lines)
- `EBSDRefineTestSetup` (L34-50): class attribute `nickel_ebsd_small = kp.data.nickel_ebsd_small()` (backgrounds removed) + a **random 5×5×5 float32 master pattern** with a 2-hemisphere/5-energy axes manager used for almost all speed-critical tests.
- Real-MP tests:
  - `test_refine_orientation_nickel_ebsd_small` (L560-578): `assert np.allclose(xmap_ref.scores, s.xmap.scores, atol=1e-3)`
  - `test_refine_orientation_nickel_ebsd_small_nlopt` (L581-604): `assert xmap_ref.scores.mean() > s.xmap.scores.mean()`
  - `test_refine_orientation_pseudo_symmetry_{nlopt,scipy}` (L608-680): `assert np.allclose(xmap_ref.pseudo_symmetry_index, [2,0,0,0,0,0,0,0,0])` etc.
  - L1175, L1205: more `nickel_ebsd_master_pattern_small(projection="lambert", energy=20)` uses.
- All real-data refinement uses `signal_mask = ~kp.filters.Window("circular", s._signal_shape_rc).astype(bool)`.

### 3.6 Reference values baked into the data (useful as regression targets)
- `nickel_ebsd_small.xmap`: 9 pts, phase `ni` (Fm-3m/m-3m), props `scores`, `z`; `scores` min/mean/max = **0.42198 / 0.47869 / 0.54670**; Euler (deg):
  `[257.91 57.14 91.25] [291.81 61.73 181.07] [291.73 61.84 180.85] [348.57 88.85 147.25] [291.74 62.04 180.86] [291.61 62.13 180.67] [257.95 56.81 1.27] [291.36 62.21 0.75] [202.47 91.01 28.20]`
- `nickel_ebsd_large.xmap`: 4125 pts, `scores` min/mean/max = **0.05060 / 0.48705 / 0.59603**; PC array (55,75,3), `pc_average=[0.42326, 0.21363, 0.50207]`.
- HDF5 layout (both): `Scan 1/EBSD/Data/patterns` (n, 60, 60) uint8; `Scan 1/EBSD/Header/{pcx,pcy,pcz,static_background,sample_tilt=70,binning=8,step_x=step_y=1.5,…}`; large file also has `Scan 1/EBSD/CrystalMap/crystal_map/data/{phi1,Phi,phi2,x,y,scores,…}`.

### 3.7 Notebook narrative structures (to mirror in `spherical_indexing.ipynb`)

**`doc/tutorials/hough_indexing.ipynb` — 63 cells. Dataset: `kp.data.nickel_ebsd_large(allow_download=True)`.**
```
[0]  MD  boilerplate "This notebook is part of the kikuchipy documentation …"
[1]  #   Hough indexing            (intro + alert box about optional dependency PyEBSDIndex)
[2]  code imports (%matplotlib inline, matplotlib, diffpy.structure Atom/Lattice/Structure,
                   diffsims ReciprocalLatticeVector, kikuchipy, orix plot/Phase/PhaseList/Vector3d,
                   plt.rcParams.update({"font.size": 15, "lines.markersize": 15}))
[3]  MD  "Load the dataset of (75, 55) nickel EBSD patterns of (60, 60) pixels, step 1.5 um"
[4]  code s = kp.data.nickel_ebsd_large(allow_download=True); s
[5]  ##  Pre-indexing maps        (VBSE + IQ rationale)
[6]  code kp.imaging.VirtualBSEImager(s); print(vbse_imager.grid_shape)
[8]  code vbse_imager.get_rgb_image(r=(2,1), g=(2,2), b=(2,3))
[10] code maps_vbse_rgb.plot()
[12] code s.remove_static_background(); s.remove_dynamic_background()
[14] code maps_iq = s.get_image_quality()
[16] code s.xmap.plot(maps_iq.ravel(), cmap="gray", colorbar=True, remove_padding=True)
[18] ##  Calibrate detector-sample geometry
[19] code det = kp.detectors.EBSDDetector(sig_shape, sample_tilt=70)
[21] code s_grid, idx = s.extract_grid((5,4), return_indices=True)
[23] code kp.draw.plot_pattern_positions_in_map(rc=..., roi_shape=nav_shape, roi_image=maps_iq)
[25] code phase_list = PhaseList(Phase("ni", space_group=225, structure=Structure(Lattice(3.5236,…), [Atom("Ni",[0,0,0])])))
[26] code indexer = det.get_indexer(phase_list, [[1,1,1],[2,0,0],[2,2,0],[3,1,1]], nBands=10, tSigma=2, rSigma=2)
[29] code det = s_grid.hough_indexing_optimize_pc(pc0=[0.42,0.22,0.50], indexer=indexer, batch=True,
                                                 method="PSO", search_limit=0.05); print mean/std
[31] code det.plot_pc("scatter", s=50, annotate=True)
[33] code det.pc = det.pc_average
[35] code det.plot(pattern=s_grid.inav[0,0].data)
[36] ##  Perform indexing
[37] code indexer = det.get_indexer(...); indexer.PC
[39] code xmap = s.hough_indexing(phase_list=phase_list, indexer=indexer, verbose=2)
[40] code xmap
[42] code (commented) orix.io.save("xmap_ni.h5"/"xmap_ni.ang", …)
[44] ##  Validate indexing results
[45] code 2x2 imshow of ["pq","cm","fit","nmatch"] maps
[46] code 2x2 histograms of the same
[49] code ckey = plot.IPFColorKeyTSL(sym, Vector3d.xvector())
[51] code rgb_x = ckey.orientation2color(xmap.rotations); xmap.plot(rgb_x, overlay="cm", …) + inset color key
[53] code 3 IPF maps (X, Y, Z)
[55] code ReciprocalLatticeVector(hkl=…).symmetrise() → KikuchiPatternSimulator → simulator.on_detector(det, xmap.rotations)
[57] code sim.as_markers(); s.add_marker(markers, permanent=True)
[59] code maps_nav_rgb = kp.draw.get_rgb_navigator(rgb_x…)
[60] code s.plot(maps_nav_rgb)
[61] ##  What's next?             (links to refinement section of pattern_matching; troubleshooting checklist)
```

**`doc/tutorials/pattern_matching.ipynb` — 96 cells. Datasets: `nickel_ebsd_large` + `nickel_ebsd_master_pattern_small`.**
```
[1]  # Pattern matching
[2]  code imports (tempfile, hyperspy.api as hs, orix sampling/plot/io, Vector3d)
[4]  code s = kp.data.nickel_ebsd_large(allow_download=True)
[6]  code s.remove_static_background(); s.remove_dynamic_background()
[9]  ## Dictionary indexing  /  ### Load a master pattern
[10] code energy = 20; mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", energy=energy)
[11] code mp.plot();  [13] ni = mp.phase
[14] ### Sample orientation space
[15] code R = sampling.get_sample_fundamental(method="cubochoric", resolution=3, point_group=ni.point_group)
[17] ### Define the detector-sample geometry
[19] code det = kp.detectors.EBSDDetector(shape=…, pc=[0.4198, 0.2136, 0.5015], sample_tilt=70)
[21] code det.plot(coordinates="gnomonic", pattern=s.inav[0,0].data)
[22] ### Generate dictionary
[23] code sim = mp.get_patterns(rotations=R, detector=det, energy=energy, dtype_out=np.float32, compute=True)
[25] code plot 3 simulated patterns w/ Euler titles
[26] ### Perform indexing
[27] code signal_mask = ~kp.filters.Window("circular", det.shape).astype(bool); show masked/unmasked
[29] code xmap = s.dictionary_indexing(sim, metric="ncc", keep_n=20,
                                       n_per_iteration=sim.axes_manager.navigation_size//10,
                                       signal_mask=signal_mask)
[32] code print(xmap.scores[:,0].mean())
[35] code io.save(temp_dir+"ni.h5"/".ang", xmap)
[36] ### Validate indexing results
[37] code ckey_m3m = plot.IPFColorKeyTSL(pg, Vector3d.xvector())
[39] code xmap.plot(rgb_x, overlay=xmap.scores[:,0], remove_padding=True)
[41] code 3 IPF maps
[43] code ncc_map = xmap.scores[:,0].reshape(*xmap.shape); os_map = kp.indexing.orientation_similarity_map(xmap)
[45] code side-by-side NCC + OS maps
[47] code best_patterns = sim.data[xmap.simulation_indices[:,0]]…; s_best = kp.signals.EBSD(...)
[49] code hs.plot.plot_signals([s, s_best], navigator=rgb_navigator)
[51] code 2x2 experimental-vs-best comparison for 2 grains
[52] ## Refinement / [53] ### Refine orientations
[54] code s.refine_orientation(xmap, det, mp, energy, signal_mask, method="minimize",
                               method_kwargs=dict(method="Nelder-Mead", tol=1e-4))
[56] code print(xmap_ref.scores.mean(), xmap_ref.num_evals.mean())
[58-61] code NCC before/after maps + histogram
[63] code misorientation-angle map DI vs refined
[65] code IPF map w/ inset key;  [67] num_evals map
[68] ### Refine projection centers
[69] code s.refine_projection_center(…, method="minimize", method_kwargs=dict(method="Powell", tol=1e-3),
                                     trust_region=[0.02]*3, compute=False)
[70] code kp.indexing.compute_refine_projection_center_results(...)
[72-80] code stats, maps, histograms, det_ref.plot_pc(), plot_pc("scatter", c=rgb_x, alpha=0.2)
[82] ### Refine orientations and projection centers
[83] code s.refine_orientation_projection_center(…, method="LN_NELDERMEAD",
                                                 trust_region=[2,2,2,0.05,0.05,0.05], rtol=1e-3)
[85-93] code stats, maps, histograms, PC plots
[95] code cleanup: os.remove(ni.h5/ni.ang), os.rmdir
```

**`doc/tutorials/hybrid_indexing.ipynb` — 122 cells (modified in working tree; `git status` shows ` M doc/tutorials/hybrid_indexing.ipynb`). Datasets: `ni_gain(10)` (~100 MB), `ni_gain_calibration(10)`, `ebsd_master_pattern("ni", …, energy=20)` (305 MB).**
```
[1]   # Hybrid indexing
[3]   code imports (+ figure.dpi 75)
[4]   ## Load, process and inspect data
[5]   code s = kp.data.ni_gain(10, allow_download=True)      # ~100 MB into memory
[7]   code remove_static_background / remove_dynamic_background
[9]   code s.average_neighbour_patterns(kp.filters.Window("gaussian", std=1))
[11]  code maps_iq = s.get_image_quality()
[13]  code s.axes_manager.indices = (156, 80); s.plot(hs.signals.Signal2D(maps_iq))
[15]  ## Calibrate geometry
[17]  code s_cal = kp.data.ni_gain_calibration(10)
[19]  code s_cal.remove_static_background("divide") / dynamic "divide"
[21-23] code omd = s_cal.original_metadata; kp.draw.plot_pattern_positions_in_map(...)
[26]  code mp = kp.data.ebsd_master_pattern("ni", projection="lambert", energy=20, allow_download=True)
[27]  code phase = mp.phase
[29]  code g = ReciprocalLatticeVector.from_min_dspacing(phase, 0.07); sanitise_phase();
             calculate_structure_factor(); g = g[F > 0.12*F.max()]; g.print_table()
[31]  code det_cal = s_cal.detector; indexer = det_cal.get_indexer(PhaseList(phase), g.unique(True))
[33]  code s_cal.hough_indexing_optimize_pc(pc0=[0.42,0.22,0.50], indexer, batch=True)
[35]  code det_cal.plot_pc("scatter", annotate=True)
[37]  code xmap_cal = s_cal.hough_indexing(phase_list, indexer, verbose=2)
[39-40] code KikuchiPatternSimulator.on_detector + 3x3 overlay figure
[42]  code s_cal.refine_orientation_projection_center(..., "LN_NELDERMEAD",
             trust_region=[5,5,5,0.05,0.05,0.05], rtol=1e-5, maxeval=300, chunk_kwargs=dict(chunk_shape=7))
[44-51] code stats, angle/PC deviations, overlays, plot_pc
[53]  code det_cal_ref.fit_pc(pc_indices, map_indices=np.indices(nav_shape), transformation="affine")
[55-63] code plot_pc, re-refine orientations w/ fitted PC, overlays
[64]  ## Hough indexing of all patterns
[65-71] code det = det_cal_fit.deepcopy(); indexer=…(rSigma=2,tSigma=2); xmap_hi = s.hough_indexing(...)
[74]  code 2x2 pq/cm/fit/nmatch maps (fit clipped vmin=0,vmax=2)
[76-77] code IPF map + inset key
[79]  ## Identify points for re-indexing
[80]  code 4 histograms
[82]  code mask_reindex = np.logical_or.reduce((~xmap_hi.is_indexed, fit>0.95, nmatch<4, cm<0.25));
             print(f"Fraction to re-index: {100*frac_reindex:.2f}%"); keep/re-index RGB figure
[84]  code nav_mask
[85]  ## Re-indexing with dictionary indexing
[86]  code R = sampling.get_sample_fundamental(resolution=2, point_group=pg)
[87-88] code det_pc1.pc = pc_average; sim = mp.get_patterns(R, det_pc1, energy=20, chunk_shape=R.size//20)
[90]  code signal_mask
[92]  code xmap_di = s.dictionary_indexing(sim, keep_n=1, navigation_mask=nav_mask, signal_mask=signal_mask)
[95-96] code scores mean; IPF map
[98]  ## Refine Hough indexed and dictionary indexed points
[99]  code ref_kw = {detector, master_pattern, energy 20, signal_mask, "LN_NELDERMEAD", trust_region [5,5,5]}
[101-105] code s.refine_orientation(xmap_hi, navigation_mask=~nav_mask, **ref_kw); same for xmap_di
[107] ## Merge results
[108] code kp.indexing.merge_crystal_maps([xmap_hi_ref, xmap_di_ref], navigation_masks=[~nav_mask, nav_mask])
[111] code (commented) io.save
[112] ## Validate indexing results
[113] code HI vs HI+DI+ref side-by-side IPF + inset key
[115-119] code s.xmap = xmap_ref; s.detector = det; extract_grid((4,4)); plot_pattern_positions_in_map;
             simulator.on_detector; 4x4 overlay grid
[121] ## What's next?
```

**Registration of a new tutorial:** add the stem to the `Indexing` `nbgallery` in `doc/tutorials/index.rst` (currently `hough_indexing, pattern_matching, hybrid_indexing, pc_orientation_dependence, pc_fit_plane, pc_extrapolate_plane, pc_calibration_moving_screen_technique`) and to `NOTEBOOKS` in `doc/tutorials/run_nbval.sh` (currently 6 notebooks). Output sanitizing regexes live in `doc/tutorials/tutorials_sanitize.cfg` (dask "Completed | TIME", `N patterns/s`, `N comparisons/s`, tqdm, figure size, `Refining N`, `Matching M/N`).

---

## 4. EMSphInx repo: data, benchmarks, tests, and other local inputs

### 4.1 `c:/Users/westraadt.1/Repos/EMSphInx/data/`
- **`Ni {20kV 75.7deg}.sht`** — 74,828 B. Decoded header:
  - magic `*sht`, file version `(1, 1)`, software version `ve49ad6b`, modality `1` (= EBSD)
  - `beamEnergy = 20.0` keV, `primaryAngle = 75.7°`, `secondaryAngle = 0.0`
  - DOI `https://doi.org/10.1016/j.ultramic.2019.112841`, note `created with mp2sht`, EMsoft version `5_0_0_0`
  - crystal block: a=b=c = 0.35236 nm, α=β=γ = 90° (Ni)
  - **HarmonicsData at byte offset 320: `bw = 384` (int16), `zRot = 4`, `cmpFlg = 7`** — consistent with `mp2sht.cpp` hard-coding `const size_t bw = 384;`
  - This is the *only* `.sht` file anywhere under `c:/Users/westraadt.1/Repos` (`find -iname '*.sht'`).

### 4.2 `c:/Users/westraadt.1/Repos/EMSphInx/benchmarks/`
| file | size |
|---|---|
| `GPU_test_cpu.nml` / `GPU_test_cuda.nml` | 5017 / 5034 B (differ only in `backend`, output paths) |
| `GPU_test_cpu.ang` / `GPU_test_cuda.ang` | 11,234,238 B each (149,810 lines: 34 header + 149,776 data) |
| `GPU_test_cpu.h5` / `GPU_test_cuda.h5` | 3.95 MB each |
| `GPU_test_cpu.log` (3,848 B), `GPU_test_cuda*.log` (5 logs, ~470 kB each) | UTF-16LE |
| `*_IPF.tiff`, `*_CI.tiff` | small |

Benchmark inputs (from `GPU_test_cpu.nml` and echoed in the output h5 `NMLparameters/IndexEBSD/*`):
- `patfile = 'C:\Users\westraadt.1\Desktop\EMSphInx\map20240214000909769.up1'` — **DOES NOT EXIST**; `C:\Users\westraadt.1\Desktop\EMSphInx\` does not exist at all. **The benchmarks are not reproducible locally.**
- `masterfile = 'C:\Users\westraadt.1\Desktop\EMSphInx\Fe_bcc-master-12kV.sht'` — **also missing**.
- `patdims = 96, 96`; `circmask = 0`; `gausbckg = .FALSE.`; `nregions = 4`; `delta = 333` µm; `pctr = -1.92, 0.48, 17902.1` with `vendor = 'EMsoft'`; `thetac = 8`; `scandims = 407, 368, 1, 1` (149,776 pts); `roimask = '63, 21, 83, 69'` (→ 83×69 = **5,727 indexed points**); `bw = 55`; `normed = .TRUE.`; `refine = .TRUE.`; `nthread = 0`; `batchsize = 30`.
- Log header confirms: sample tilt 70°, scintillator distance 17902.1 µm, camera tilt 8°, 96×96 @ 333 µm px, circular mask true, vertical flip true; master point group `m3m`, z-rotational symmetry 4, equatorial mirror yes; bandwidth 55, side length 110, CPU, 20 threads, 8.7 s to index (655.1 pat/s). Program was built from **branch `feature/GPU`, commit `98c251a`**.

Output HDF5 layout (`GPU_test_cpu.h5`): `Scan 1/EBSD/Data/{Phi1,Phi,Phi2,Metric,IQ,Phase}` each (149776,) float32/uint8; `Scan 1/EBSD/Header/{Pattern Center Calibration/{x-star,y-star,z-star}, Phase/1/{Lattice Constant a..gamma, MaterialName, Symmetry}, Sample Tilt, Step X/Y, nColumns, nRows, Grid Type}`; `Scan 1/{IPF Map (368,407,3), IQ Map, XC Map}`; full `NMLparameters/IndexEBSD/*` echo; `EMheader/{ProgramName, Version, ComputeBackend, PatPerS, …}`.

Measured statistics:
- CPU: 5,727 non-zero `Metric`, min/mean/max = 0.3231 / 0.6160 / 0.7383 (normalized XC).
- CUDA: same 5,727 points, Metric min/mean/max = **4.4266 / 4.7014 / 4.7749** — a *different normalization*, and Euler angles differ (`max |Δeuler| = 6.18 rad`, `mean 0.79`, 91% of points differ by >1e-3 — largely symmetry/wrap representation, but the metric scale difference is real). **⇒ these files are unsuitable as a numeric regression baseline; generate a fresh CPU reference instead.**

### 4.3 `c:/Users/westraadt.1/Repos/EMSphInx/test/`
`CMakeLists.txt` (5008 B) builds 15 standalone executables (no CTest registration, no data files):
- high level: `TestDict` (`dict.cpp`), `TestDiag` (`diagram.cpp`)
- `include/sht/*`: `TestWigner` (`sht/wigner.cpp`, 137 kB of hard-coded reference Wigner d values), `TestSquare` (`sht/square_sht.cpp`), `TestXCorr` (`sht/sht_xcorr.cpp`)
- `include/util/*`: `TestBase64`, `TestLinAlg`, `TestThread`, `TestTimer`, `TestColor`, `TestNML`
- `include/xtal/*`: `TestRot`, `TestQuat`, `TestSym`, `TestPos`, `TestHM`

**Test data generation is fully synthetic and deterministic**:
- `test/sht/sht_xcorr.cpp` L101-168: `randomSphere(std::mt19937_64&, dim, mir, nFld)` builds a random master pattern with prescribed symmetry, `randomRotation()`, `randomPair(bw, mir, nFld, flm, gln)`; `const unsigned int seed = 0; // constant for deterministic behavior`.
- `test/sht/square_sht.cpp` L92-97: same seed-0 mt19937_64 pattern.
- `test/dict.cpp` is the exception: it defines a `Dictionary` struct read from an HDF5 `dictfile` and a `masterfile` supplied via an nml (`writeDictNml`: `masterfile='master.h5'`, `dictfile='dict.h5'`, `bw=68`, `sig=70.0`, `numpat=0`) — i.e. it needs external data that is **not** in the repo.
- `test/xtal/emsoft_gen.hpp` (33 kB): pure lookup tables + encode/decode of EMsoft 40-char space-group generator strings (`encode`, `decode`, `gen_from_enc`, `gen_from_num`, `mono_gen_from_num`, `ortho_gen_from_num`, `SGNames` table of 230 short Hermann-Mauguin names). No data files.

### 4.4 Other local repos — usable inputs

**EMsoft master patterns (`.h5`, EMsoft `EMEBSDmaster` format, all readable by kikuchipy and by `mp2sht`):**

| path | size | phase / SG | energies (keV) | mLPNH/mLPSH shape | accum_e |
|---|---|---|---|---|---|
| `c:/Users/westraadt.1/Repos/openECCI-data/ebsd_master_pattern/Fe_fcc-master-20kV.h5` | 114.6 MB | γ-Fe, SG 225, a=0.36 nm | 15–20 (6) | (1, 6, 1001, 1001) f32 | (501,501,6) |
| `…/openECCI-data/ebsd_master_pattern/Si-master-20kV.h5` | 162.7 MB | Si | — | — | — |
| `…/openECCI-data/ebsd_master_pattern/Si1-master-20kV.h5` | 114.6 MB | Si | — | — | — |
| `…/openECCI-data/ebsd_master_pattern/Si-master-ECP-20kV.h5` | 65.0 MB | Si, **ECP** master | — | — | — |
| `c:/Users/westraadt.1/Repos/openECCI_RKD/data/Fe-master-20kV.h5` | 210.1 MB | Fe (xtal `Fe_fcc.xtal`), SG 225, a=0.36599 | 10–20 (11) | (1, 11, 1001, 1001) | (501,501,11) |
| `…/openECCI_RKD/data/Si-master-20kv.h5` | 210.1 MB | Si, SG 227, a=0.543 | 10–20 (11) | (1, 11, 1001, 1001) | (501,501,11) |
| `…/openECCI_RKD/data/Fe_fcc-master-20kV.h5` | 114.6 MB | γ-Fe | 15–20 | — | — |
| `…/openECCI_RKD/data/Fe_fcc-ECPmaster-20kV.h5` | 49.0 MB | ECP | — | — | — |
| `…/openECCI_RKD/data/Mg-master-17kV.h5` | 152.8 MB | Mg (hcp — good low-symmetry SHT test) | — | — | — |
| `…/openECCI_RKD/data/Ti-alpha-master-20kV.h5` | 114.6 MB | α-Ti (hcp) | — | — | — |
| `c:/Users/westraadt.1/Repos/AstroEBSD/phases/dynamic_templates/Fe_fcc-master-20kV.h5` | 114.6 MB | γ-Fe (dup) | — | — | — |
| `…/AstroEBSD/phases/dynamic_templates/Nb6Co7_trigonal.h5` | 51.2 MB | trigonal (low symmetry) | — | — | — |
| `…/AstroEBSD/phases/dynamic_templates/Si_EMsoft.h5` | 7.5 MB | Si (small!) | — | — | — |
| `…/AstroEBSD/phases/dynamic_templates/*.bin`, `Si_(diamond)_20kV_1001.sdf5` | 6–100 MB | AstroEBSD/BinaryWave formats, **not** EMsoft h5 | | | |
| `c:/Users/westraadt.1/Repos/EMsoftH5/Cu_HH4_110_g002*.h5` | 0.1–2.2 MB | **not** master patterns (HH4 TEM contrast images) | | | |

`find -maxdepth 4 -iname '*.h5' -size +1M` under `EMsoft`, `EMsoft_old`, `EMsoftBuild`, `EMsoftH5`, `EMsoftOO`, `EMsoftSuperbuild` → **only** the 3 `EMsoftH5/Cu_HH4_*` files. **No master pattern .h5 in the EMsoft* repos.**

**Experimental EBSD pattern files under `Repos` (depth 4):**
| path | size | notes |
|---|---|---|
| `c:/Users/westraadt.1/Repos/openECCI_RKD/data/T18_46_24_Fe.up1` | **6,554,211,648 B (6.5 GB)** | + `T18_46_24_Fe.ang` (2.48 MB), `T18_46_24_Fe_kp.ang`/`.h5` (kikuchipy-processed), `T18_46_24_Fe_NPAR_SHI127.ang` |
| `…/openECCI_RKD/data/Si_0tilt_0rot.up1`, `Si_3tilt_10rot.up1`, `T18_11_35.up1`, `T18_22_58.up1` | 263,398,310 B each | each with a matching `.ang` (~100 kB) — Si single crystal + others |
| `c:/Users/westraadt.1/Repos/DataSample_Steel/Steel Ferrite-Martensite 40000X w 1x1 Pats.osc` | 425,905 B | EDAX `.osc` (kikuchipy has no `.osc` reader; EMSphInx has no `.osc` reader) |
| `c:/Users/westraadt.1/Repos/DataSample_Steel/Steel Ferrite-Martensite 40000X w 1x1 Pats/*.jpg` | **900 JPEGs** (`…_x{c}y{r}.jpg`) | + `.ang` (84 kB), `.txt` (81 kB), `test.txt`, `Corr_Output.ang` (65 kB), `AnalysisParams_Output.mat` (1.27 MB). **Readable by `kp.load(<dir>)` via the `ebsd_directory` plugin** (cf. `tests/test_io/test_ebsd_directory.py`). |
| `openECCI-data/ebsd_map/*.ctf` | 2 files (Si map, fcc-Fe map) | orientation maps only, no patterns |
| `openECCI-data/fcc_fe/*.tif` | many | ECP/SEM images, not EBSD patterns |
| No `.ebsp`, no `.up2` anywhere under `Repos` (depth 4). |

### 4.5 EMSphInx I/O capabilities relevant to kikuchipy interop
- `include/modality/ebsd/pattern.hpp` L389, L433-535: pattern file extensions supported = **`up1`, `up2`, `ebsp`, and `h5`/`hdf`/`hdf5`** (3-D dataset, `uint8`/`uint16`/`float` only; the dataset path is given by the nml key **`patdset`**). `.data`/NORDIF raw is *not* handled by this code path (despite `PatternRepack` usage text mentioning `*.data`).
- The `IndexEBSD -t` template default is `patdset = 'Scan 1/EBSD/Data/Pattern'` — one character away from kikuchipy's h5ebsd layout `Scan 1/EBSD/Data/patterns` (shape `(4125, 60, 60)` uint8 for `nickel_ebsd_large`). **⇒ `nickel_ebsd_large/patterns.h5` can be fed to `IndexEBSD.exe` directly with `patdset='Scan 1/EBSD/Data/patterns'`, `patdims=60,60`, `scandims=75,55,1.5,1.5`.**
- `include/modality/ebsd/nml.hpp` L241-307: `patfile`, `patdset` (only if `H5::H5File::isHdf5`), `masterfile`, `scandims` (either a `.ang`/`.ctf`/`.h5` filename or 3–4 numbers), `patdims` (2 ints), `pctr` + `vendor ∈ {EMsoft, EDAX, TSL, Oxford, Bruker}`, `thetac`, `roimask`, `bw`, `normed`, `refine`, `nthread`, `batchsize`, `datafile`, `vendorfile`, `ipfmap`, `qualmap`. (The *current* build's template has **no** `backend`/`gpudevice` keys → the built exes are from **master**, unlike the benchmark logs which are from `feature/GPU`.)
- `programs/mp2sht.cpp` requires from the EMsoft `.h5`: `CrystalData/{SpaceGroupNumber, SpaceGroupSetting, LatticeParameters, Natomtypes, AtomData, Atomtypes}`, `NMLparameters/MCCLNameList/{sig, omega, EkeV, Ehistmin, Ebinsize, depthmax, depthstep, totnum_el, multiplier, numsx}`, `NMLparameters/BetheList/{c1,c2,c3,sgdbdiff}`, `NMLparameters/EBSDMasterNameList/{dmin, npx}`. **Verified present in all three of:** kikuchipy's packaged `ni_mc_mp_20kv_uint8_gzip_opts9.h5`, the cached `ni_mc_mp_20kv.h5`, and `openECCI-data/…/Fe_fcc-master-20kV.h5`.
- `include/idx/master.hpp` L242-345 (`MasterPattern::read`): reads `EMData/MCOpenCL/accum_e` (3-D), sums it over the map to get **per-energy weights**, then reads **both** `mLPNH` and `mLPSH` as `NATIVE_FLOAT` (HDF5 auto-converts uint8 → float), sums over the atom axis, and forms the **energy-weighted average** over all bins. Then `MasterSpectra<double>(mp, bw=384, nrm=true)`.
  ⇒ **A faithful kikuchipy re-implementation of `mp2sht` must (a) use both hemispheres in square-Lambert, (b) weight energies by `accum_e` summed over the MC map — which kikuchipy's reader does NOT expose (see §6.4).**

---

## 5. Built EMSphInx binaries — usage text (verbatim, exit code 0 for all)

`c:/Users/westraadt.1/Repos/EMSphInx/build/Release/` also contains `EMSphInxEBSD.exe` (10.7 MB, wxWidgets GUI — **not run**), `ShtWisdom.exe` (1.8 MB), `emsphinx_cuda.lib`.

**IndexEBSD.exe** (4,190,720 B)
```
useage: 	index using a nml file : C:\Users\westraadt.1\Repos\EMSphInx\build\Release\IndexEBSD.exe input.nml
	generate a template nml: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\IndexEBSD.exe -t
```
(`-t` writes `IndexEBSD.nml` into the CWD — see the warning at the top.)

**mp2sht.exe** (3,773,440 B)
```
usage: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\mp2sht.exe inputFile outputFile
	inputFile  - master pattern to read (*.h5)
	outputFile - spherical hamrnoics file to write (*.spx)
```

**MasterXcorr.exe** (3,849,728 B)
```
usage: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\MasterXcorr.exescanFile bandWidth cutoff masterFile1 [masterFile2]
	bandWidth  : bandwidth for cross correlation (2*bw-1 should be product of small primes)
	             88, 95, 113, 123, 158, 172, 188, 203, 221, 263, and 284 are reasonable values
	cutoff     : cutoff for peak consideration [0,1] (relative to maximum cross correlation)
	masterFile1: name of first master pattern file (e.g. Ni.h5)
	masterFile2: name of second master pattern file (e.g. Ni.h5)

note: only symmetry of first pattern will be used for to improve calculation speed
```

**sht2png.exe** (1,931,776 B)
```
usage: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\sht2png.exe inputFile sqLegOut [sterOut]
	inputFile - spherical harmonics file to read (*.sht)
	sqLegOut  - location to write square legendre image (*.png)
	sterOut   - optional location to write stereographic image (*.png)
```

**PatternRepack.exe** (1,941,504 B)
```
usage: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\PatternRepack.exe inputFile outputFile [binning]
	inputFile  - pattern file to read (*.up1, *.up2, *.data, or *.ebsp)
	outputFile - output file (*.hdf)
	binning    - [optional] binning size (must evenly divide into pattern size)
```

**EBSPDims.exe** (1,802,240 B)
```
usage: C:\Users\westraadt.1\Repos\EMSphInx\build\Release\EBSPDims.exe inputFile
	inputFile  - pattern file to read (*.ebsp)
```

---

## 6. Recommendations

### 6.1 (a) Unit tests — small, fast, shipped, offline
**Use `kp.data.nickel_ebsd_small()` + `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)`.**

Justification:
- Both are **shipped in the wheel** (no network, `Dataset.is_in_package == True`, pooch bypassed entirely — `_data.py` L639-649), so tests run in CI and offline.
- 9 patterns × 60×60 uint8 = 32 kB; master pattern 2×401×401 uint8 = 322 kB → sub-second.
- A **ground-truth `xmap` is bundled** (9 orientations from HI + refinement, scores 0.422–0.547) — reuse the exact `test_ebsd_hough_indexing.py::test_hough_indexing` idiom:
  ```python
  angles = xmap.orientations.angle_with(s.xmap.orientations, degrees=True)
  assert np.all(angles < 1)   # or < 2 deg for bw≈68 spherical indexing without Newton refinement
  ```
  Note that spherical indexing at bandwidth 53–88 on 60×60 patterns has an intrinsic angular resolution of roughly `180/bw ≈ 2–3.4°` before Newton refinement — pick the tolerance accordingly (suggest `< 2°` with refinement, `< 4°` without, and add a "fraction of points within 2° ≥ 8/9" style assertion rather than `np.all` if it proves flaky).
- Per-point PC array is stored ((3,3,3)); use `detector.pc_average = (0.4251, 0.2134, 0.5007)`, `sample_tilt=70`, `tilt=0`, `binning=8`, `px_size=1`.
- Caveat: the packaged MP is **uint8** and **single energy (20 keV, `numEbins=1`)**, so an energy-weighted average is trivially the 20 keV slice; quantization to 256 levels will slightly perturb the harmonics — fine for a "does it index" test, not for a bit-exact comparison with `mp2sht` output on the full float32 MP.
- Also add pure-synthetic unit tests in the EMSphInx style (deterministic `np.random.default_rng(0)` sphere + known rotation → recovered rotation), mirroring `test/sht/sht_xcorr.cpp::randomPair` — these are the fastest and most diagnostic.

### 6.2 (b) Integration / regression test against EMSphInx reference
**Do NOT use `benchmarks/GPU_test_*` as the baseline**: the experimental `.up1` and the `Fe_bcc-master-12kV.sht` are missing from the machine, the CPU and CUDA `Metric` scales disagree by ~7×, and the run was made from a different branch (`feature/GPU@98c251a`) than the current build (master).

**Recommended baseline generation (one-off, checked into the repo or into `kikuchipy-data`):**
1. `mp2sht.exe <cache>/develop/data/ebsd_master_pattern/ni_mc_mp_20kv.h5 ni_20kv_bw384.sht`
   → ~75 kB `.sht`, bw = 384, from the *full* float32 Ni MP (5–20 keV, both hemispheres, MC-weighted). Ship this file in `src/kikuchipy/data/emsphinx/` with an md5 in `_registry.py` (it is smaller than the already-shipped 1 MB Ni MP).
   *Alternative reference already on disk*: `EMSphInx/data/Ni {20kV 75.7deg}.sht` (bw=384, zRot=4, cmpFlg=7, 20 keV, Ni a=0.35236 nm) — same phase as `nickel_ebsd_small`, so it can serve as a **byte-level target for a Python `mp2sht` port**, and as the master for a `IndexEBSD` reference run. Its `primaryAngle=75.7°` (vs. the Ni dataset's 70° sample tilt) means it was generated from a different MC simulation — check `sig` before using it as the indexing master.
2. Run `IndexEBSD.exe ni_small.nml` with
   `patfile = '<pkg>/data/kikuchipy_h5ebsd/patterns.h5'`, `patdset = 'Scan 1/EBSD/Data/patterns'`, `patdims = 60, 60`, `scandims = 3, 3, 1.5, 1.5`, `delta = 70` (UF-1100 px size × binning 8 — verify), `vendor = 'Bruker'` with `pctr = 0.4251, 0.2134, 0.5007` (kikuchipy stores Bruker convention), `thetac = 0`, `circmask = 0`, `nregions = 4`, `bw = 68`, `normed = .TRUE.`, `refine = .TRUE.`, `nthread = 1`, `batchsize = 1` (serial ⇒ deterministic).
   Repeat for `nickel_ebsd_large` (4125 pts, ~6 s at 655 pat/s) for a stronger statistical baseline.
3. Store the reference as a tiny `.npz`/`.ang` of `(phi1, Phi, phi2, metric)` (4125×4 float32 ≈ 66 kB) in `kikuchipy-data` or in-package, registered in `_registry_hashes`.
4. Assert, kikuchipy-style: `orientation.angle_with(reference, degrees=True)` → `np.mean(angles < 1.0) > 0.98` and `np.median(angles) < 0.5`, plus `np.corrcoef(metric_kp, metric_ref)[0,1] > 0.95`. Mark it `@pytest.mark.weekly` (the marker infrastructure already exists, `conftest.py` L64-83) or gate it on the presence of the reference file, so the default suite stays fast.
5. Cross-check the SHT pipeline itself with `sht2png.exe` (square-Legendre + stereographic PNGs) against kikuchipy's own transform of the same master pattern — an image-level regression that isolates the transform from the indexing.

**Serial determinism caveat:** `batchsize`/`nthread` affect work partitioning; use `nthread=1, batchsize=1` for the reference run and record the EMSphInx commit hash in the reference file's metadata.

### 6.3 (c) Tutorial notebook `spherical_indexing.ipynb`
**Primary dataset: `kp.data.nickel_ebsd_large(allow_download=True)` (15.4 MB, 4125 patterns, 60×60) + `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)`.**

Justification:
- Exactly what `hough_indexing.ipynb` and `pattern_matching.ipynb` use ⇒ the reader can compare HI vs. DI vs. SI results directly, and the notebook is directly comparable in runtime and figures.
- 15.4 MB download works on MyBinder (the tutorials are advertised as live on MyBinder in `doc/tutorials/index.rst`); `ni_gain(10)` (100 MB) and `ebsd_master_pattern("ni")` (305 MB) — used by `hybrid_indexing.ipynb` — are much heavier and would make CI/nbval slow.
- It is already downloaded at pytest session start (`conftest.py` L105-106), and `nbval` runs the tutorials (`run_nbval.sh`).
- Known-good PC `[0.4198, 0.2136, 0.5015]` / `pc_average=[0.42326, 0.21363, 0.50207]`, `sample_tilt=70`, and a bundled reference `xmap` (mean NCC 0.487) for the "Validate indexing results" section.
- **Secondary/advanced section** (optional, mirroring `hybrid_indexing`): full-resolution `ebsd_master_pattern("ni", energy=20, hemisphere="both")` (already cached locally) to show the effect of master-pattern resolution / energy averaging on the SHT.
- **Low-symmetry demo** (optional): `openECCI_RKD/data/Mg-master-17kV.h5` or `Ti-alpha-master-20kV.h5` — hcp masters exercise `zRot`/mirror compression paths that cubic Ni never touches. These are local-only and cannot go in the notebook, but are excellent for local dev testing.

**Proposed narrative (mirrors `hough_indexing.ipynb` cell-for-cell so the three indexing tutorials read as a set):**
```
[MD] doc boilerplate
#  Spherical indexing                         (intro: what SHT/spherical XC indexing is,
                                               cite Lenthe et al. Ultramicroscopy 2019 doi:10.1016/j.ultramic.2019.112841,
                                               alert box for optional dependency if any)
   code imports
   code s = kp.data.nickel_ebsd_large(allow_download=True); s
## Pre-indexing maps                          (VBSE RGB + IQ map; remove_static/dynamic_background)
## The master pattern on the sphere           (NEW section, replaces "Calibrate detector-sample geometry" ordering)
   code mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both", energy=20)
   code mp.plot() / plot_spherical()          (EBSDMasterPattern.plot_spherical exists, CHANGELOG L693)
   code compute spherical harmonic transform, show bandwidth choice
        (reasonable bw list from the nml comments: 53, 63, 68, 74, 88, 95, 113, 122, 123, 158, 172, 188, 203, 221, 263, 284, 313)
   code round-trip: harmonics -> reconstructed sphere, show truncation effect vs bw
## Calibrate detector-sample geometry         (same as hough_indexing: EBSDDetector(sample_tilt=70),
                                               extract_grid((5,4)), plot_pattern_positions_in_map,
                                               hough_indexing_optimize_pc(pc0=[0.42,0.22,0.50], PSO, search_limit=0.05),
                                               det.pc = det.pc_average, det.plot(pattern=...))
   [MD] note: spherical indexing needs the *pattern back-projected onto the sphere*, so the PC matters
        exactly as in DI; note the vendor PC conventions (EMsoft/EDAX/TSL/Oxford/Bruker) per the EMSphInx nml.
## Perform indexing
   code circular mask / AHE (nregions) preprocessing, matching EMSphInx's imprc defaults
   code xmap = s.spherical_indexing(master_pattern=mp, detector=det, bandwidth=68, normalize=True, refine=True)
   code xmap
   code (commented) orix.io.save("xmap_ni_si.h5"/".ang", xmap)
## Validate indexing results
   code metric map + histogram (analogue of pq/cm/fit/nmatch)
   code IPFColorKeyTSL(m-3m, Vector3d.xvector()); IPF-X map w/ metric overlay + inset color key
   code 3 IPF maps (X, Y, Z)
   code geometrical simulations overlaid (ReciprocalLatticeVector -> KikuchiPatternSimulator.on_detector -> as_markers)
   code s.plot(kp.draw.get_rgb_navigator(rgb_x))
## Compare to Hough indexing and dictionary indexing
   code angle_with() between the SI xmap and the bundled s.xmap reference; histogram of misorientation angles;
        timing comparison table (SI vs DI dictionary size vs HI)
## Refinement
   code s.refine_orientation(xmap, det, mp, energy=20, signal_mask=..., method="LN_NELDERMEAD",
                             trust_region=[2,2,2])   -> before/after NCC histogram
## What's next?
```
Also: append `spherical_indexing` to the `Indexing` nbgallery in `doc/tutorials/index.rst` and to `NOTEBOOKS` in `doc/tutorials/run_nbval.sh`; add sanitize regexes if new progress output appears.

### 6.4 Master pattern availability & the kikuchipy loader — the SHT-critical facts

**Nickel master patterns in `kikuchipy.data`:**

| | `nickel_ebsd_master_pattern_small()` | `ebsd_master_pattern("ni")` |
|---|---|---|
| shipped in package | **yes** | no (Zenodo, 305 MB; **already in the `develop` and `0.9.0` caches**) |
| resolution | 401×401 (`npx=200`) | 1001×1001 (`npx=500`) |
| dtype | uint8 | float32 |
| energies | **1 bin: 20 keV only** (`EkeVs=[20.]`, `numEbins=1`, `Ehistmin=20`, `EkeV=20.1`) | **16 bins: 5,6,…,20 keV** (`Ehistmin=5`, `EkeV=20`, `Ebinsize=1`) |
| hemispheres | **both** (`mLPNH` and `mLPSH`, shape (1,1,401,401) each; `masterSPNH/SH` too) | **both** (`mLPNH`/`mLPSH` (1,16,1001,1001)) |
| `accum_e` | (201,201,1) int32 | (501,501,16) int32 |
| phase | Ni, SG 225, a=0.35236 nm, Z=28, MC `sig=70°`, `omega=0` | identical |

**⇒ Yes, a full-sphere (both-hemisphere) Ni master pattern exists at 20 keV in both flavours; the full-energy-range (5–20 keV) one needed for a faithful `mp2sht`-equivalent MC-weighted average exists only in the 305 MB Zenodo file — which is already in the local cache.**

**kikuchipy's EMsoft master-pattern reader** (`src/kikuchipy/io/plugins/emsoft_ebsd_master_pattern/_api.py` → `src/kikuchipy/io/plugins/_emsoft_master_pattern.py`):
- `file_reader(filename, energy=None, projection="stereographic", hemisphere="upper", lazy=False)` (`_api.py` L52-90). Defaults are `upper`/`stereographic`; pass `hemisphere="both"`, `energy=None` for everything.
- **Verified by loading**: `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both", lazy=True)` → shape **(2, 16, 1001, 1001) float32**, axes `hemisphere(2), energy(16, offset 5.0 keV, scale 1.0), height(1001, offset -501), width(1001)`. Same for `projection="stereographic"`. With `energy=20` → (2, 1001, 1001).
- `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")` → **(2, 401, 401) uint8**.
- ⇒ **The loader can read both hemispheres and all energies. Two important behaviours to account for when porting `mp2sht`:**
  1. `_emsoft_master_pattern.py` L~136-172: for `projection="lambert"` (non-ECP) it **sums over the atom/`numset` axis** — matching `master.hpp`'s atom accumulation. Good.
  2. For `projection="stereographic"` it **flips the data up-down** (`data = data[..., ::-1, :]`). Do the SHT from the **Lambert** arrays, not the stereographic ones.
  3. **`original_metadata` contains only `NMLparameters` (`BetheList`, `EBSDMasterNameList`, `MCCLNameList`) plus `CrystalData` — verified. `EMData/MCOpenCL/accum_e` is NOT read**, so the MC energy-weighting that `master.hpp` L265-277 performs is unavailable through `kp.load`. A port of `mp2sht` must either (i) open the HDF5 directly with `h5py` for `accum_e`, or (ii) extend the reader to expose it (small, backwards-compatible change to `_emsoft_master_pattern.py`).
  4. `check_file_format` requires `EMheader/EBSDmaster/ProgramName ∈ {EMEBSDmaster.f90, EMEBSDmasterOpenCL.f90}` — all local files pass.
