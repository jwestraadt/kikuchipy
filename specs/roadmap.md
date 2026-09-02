# Roadmap: spherical indexing (EMSphInx CPU port)

Dependency chain: 0 -> 1 -> {2, 3} -> 4 -> 5 -> 6 -> 7 -> {8, 9} -> 10 -> 11 (Phase 10 needs the Phase 2 `.sht` writer and the Phase 9 pattern repacker).
Each phase is one branch and one dated spec folder `specs/YYYY-MM-DD-<name>/`
(`requirements.md`, `plan.md`, `validation.md`). A box is ticked only when the work is
committed on the phase's branch (verifiable with `git log`); a phase is complete
when every box is ticked, and the next phase's `plan.md` is drafted only then. The
gate list for every code phase is: plan approved -> spec recorded -> failing tests
committed -> implementation -> adversarial review + fixes -> pre-commit clean ->
CHANGELOG entry -> PR opened -> PR merged. Documentation-only phases (Phase 0)
skip the failing-tests and CHANGELOG gates; phases with no user-facing change
(e.g. Phase 1, private code only) skip the CHANGELOG gate. A phase's definition
of done ends at "PR opened"; "PR merged" is tracked here.

## Phase 0 -- `spherical-indexing-constitution` (spec `2026-08-16-constitution`)
- [x] Sync fork `develop` with `upstream/develop`, stash/restore local notebook edits, delete stray `IndexEBSD.nml`
- [x] `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `specs/_research/*` (planning artefacts)
- [x] Patent search recorded in `mission.md`
- [x] Bibliography entries (Lenthe 2019 x2, Reinecke 2011, Schaeffer 2013, Sneeuw 1994, Huhle 2009, Kostelec & Rockmore 2008, Rosca 2010, Gutman 2008) -- they render on the public Bibliography page immediately (`:all:`)
- [x] EMSphInx entry in `doc/user/related_projects.rst` extended (source repo, SHT database)
- [x] Contributor added to `src/kikuchipy/__init__.py` credits and `.zenodo.json`
- [x] `.pre-commit-config.yaml`: `specs/` added to the top-level `exclude` (fork-only; dropped from any upstream PR)
- [x] Upstream issue text and maintainer email drafted (`specs/2026-08-16-constitution/upstream-issue.md`) -- sent only after the user approves
- [x] Adversarial review of the constitution (3 critics) and fixes applied
- [x] `doc/user/bibliography.bib` parses under pybtex and the 9 entries render in `doc/_build/html/user/bibliography.html` (Sphinx build exit 0; no `:cite:` yet)
- [x] Signed commit pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#1)

## Phase 1 -- `sht-square-grid-transform`
- [x] `_grid.py`: square<->sphere maps, `legendre_normals`, ring tables (`readRing` port), `ring_solid_angles`, `lambert_solid_angles`, `quadrature_weights` (`computeWeightsSkip` port)
- [x] `_fft.py`: `fast_size` (verbatim `fastSize` port), `fast_bandwidths` (private until Phase 6)
- [x] `_sht.py`: `SphericalHarmonicTransform` with dual-path `analyze`/`synthesize`
- [x] Tests: single-harmonic analyze/synthesize oracles vs `sph_harm_y`; signed Condon-Shortley confirmation; EMSphInx `square_sht.cpp` round trip (Lambert at 1e-11 scale-free); weights (`sum(w_hat)=1`, Legendre == Gauss-Legendre with halved equator, Lambert guard at dim 401); grid/ring/solid-angle invariants; Ni master m-3m structural zeros (real data); `fast_size` invariants + 13-smooth minimality
- [x] Adversarial review (fidelity vs compiled C++, conventions, 21-mutation bug injection) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#2)

## Phase 2 -- `sht-master-spectra-and-file`
- [x] `_master_pattern_harmonics.py`: public `kp.indexing.MasterPatternHarmonics` (`from_master_pattern`, `from_file`, `save` = `.sht` writer, `to_master_pattern` (direct Lambert synthesis, needed by `kp.load`), `resize`, `remove_dc`, `power_spectrum`, `describe`) -- `MasterSpectra` port; `toLegendre` DCT regrid; weighted normalisation with compat quirk (default `emsphinx_compatible=True`, parity-first); `accum_e` energy weights; symmetry LUTs (the 38 `_groups` names + the `'2'`/`'m'` aliases returned by `get_point_group`, validated against `orix.quaternion.symmetry._groups`, names confirmed on orix 0.12.1); bandwidth-vs-resolution warning; Phase 1 amendment: lazy `quadrature_weights` in `_sht.py` (Lambert synthesis at any odd `dim`, the Sneeuw guard moves to `analyze`)
- [x] `_sht_file.py` (BSD-3 SHTfile codec; generic modality/simMetaSize; NotImplementedError paths)
- [x] io plugin `emsphinx_master_pattern`; `EBSDMasterPattern.get_spherical_harmonics`
- [x] Data: `ni_20kv_bw384.sht`, `ni_small_20kv_bw384.sht` (mp2sht.exe, sig 70; the latter from an uncompressed repack because `mp2sht.exe` lacks HDF5 deflate); synthetic per-(zRot, cmpFlg) fixtures generated at test time (`_dummy_files/emsphinx_sht.py`, md5s pinned after `sht2png.exe` acceptance)
- [x] Tests: header parse, read->write field/CRC equality, pack/unpack all branches, EMSphInx binaries accept our `.sht` (local-gated), mp2sht parity, `kp.load(".sht")`, bandwidth warning
- [x] Adversarial review (fidelity vs C++/compiled cross-check, conventions, 60-mutation bug injection) and fixes
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#5)

## Phase 3 -- `sht-wigner-d`
- [x] `_wigner.py` (d(pi/2) table, dTable(beta), dTablePre, scalar `wigner_d`, `wigner_D`, `rotate_harmonics`, derivative helpers `wigner_d_prime`/`wigner_d_prime2`); reference-table module `src/kikuchipy/data/emsphinx/wigner_reference_tables.py` (the Mathematica tables of `test/sht/wigner.cpp`)
- [x] `_euler.py` (`zyz_to_quaternion` etc.; explicit port of `test/xtal/rotations.cpp:288-318`; Bunge equivalence test to 1e-14)
- [x] Tests: Mathematica tables from `test/sht/wigner.cpp`, table vs scalar, rotate composition/identity; table-based derivative formulas of `sht_xcorr.hpp:1009-1041` pinned in `test_spherical_wigner.py` against `wigner_d_prime`/`wigner_d_prime2` (Phase 7 copies them)
- [x] Adversarial review (fidelity bitwise vs compiled C++, conventions, 56-mutation bug injection) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#4)

## Phase 4 -- `spherical-cross-correlation`
- [x] `_xcorr.py`: `SphericalCrossCorrelator` (spectrum kernel, separable `scipy.fft` inverse with `m % n_fold` plane skipping, `findPeak`, 27-neighbourhood + glide, tri-quadratic interpolation, `index_to_euler`/`euler_to_index`, `clone()`), `NormalizedSphericalCrossCorrelator` (Huhle `rDen` computed once, fused `xc *= rDen`/argmax); `refine=True` raises until Phase 7; `extractBunge` (`sht_xcorr.hpp:594-649`) **not ported** (no consumer before Phase 9; reversed `zyz2eu` offsets -- use `_euler.bunge_to_zyz` there)
- [x] Tests: `sht_xcorr.cpp` ports (random pairs, symmetric groups, wedge mask), Ni master autocorrelation -> identity + 24 cubic ops, timing baseline at bw 53/68/88; normalised correlator with the Ni master in both `emsphinx_compatible` settings against a known rotation: argmax misorientation within the grid/refinement tolerance, score difference recorded (the D7 gate of Phase 2); two analytic oracles (Phase 3 `wigner_D` triple sum, `rotate_harmonics` inner product), glide identity on the full `irfftn` cube, C++ `extractNeighborhood` defects pinned in both `emsphinx_compatible` settings, memory (`tracemalloc`) at bw 63/68/88/113
- [x] Adversarial review (fidelity vs compiled C++ driver ~1e-16, conventions, 109-mutation bug injection) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#6)

## Phase 5 -- `spherical-back-projection`
- [x] `_back_projection.py` (`SphericalBackProjector`: gather LUT on the north Legendre grid through kikuchipy's detector geometry (exact inverse of `_get_direction_cosines_for_fixed_pc`, pixel-centre convention, physical guard `z_s >= 0` -- the south hemisphere is never gathered), `solidAngle(501)`/`scaleFactor` port with `oversampling = sqrt(2)`, DCT rescaler with mean removal, DCT IQ, window mask built directly (pocketfft's constant DCT is inexact), `mlm`, `squared_harmonics` (`flm2`; the correlator owns `rDen`), single-PC (`navigation_size != 1`) and `azimuthal`/`twist` guards, two empty-window guards (`rescaled_shape < 1 px`, `n_points == 0`), `signal_mask` in kikuchipy polarity with mean fill, `circular_mask=False` default (physical circle), `window_harmonics` eager -- one immutable projector shared across threads)
- [x] `_preprocessing.py` (Gaussian background with the off-by-one behind `emsphinx_compatible`, `cholesky` with the C++ NaN comparison direction, mosaic AHE == `skimage` CLAHE for dividing tiles, `_preprocess_pattern` in EMSphInx order with `IndexEBSD` defaults `n_regions=10`, `gaussian_background=False`, no mask; no `scipy.fft` here -- the DCT IQ lives in `_back_projection.py`)
- [x] Tests: `dctn` convention first, LUT vs `_get_direction_cosines_from_detector`, `Y_l^m` recovery, `signal_mask` changes `rDen`, `nickel_ebsd_small` window/IQ, forward-projection convention lock (27 rotations, flip/asymmetry check), binning != 1 PC test, `azimuthal`/`twist` guard, per-point-PC guard; `dctn` `4 h w` round trip + `idctn` negative control; structural pin of the resample map (the direction oracle cannot see a stretch there); rim structure; forward-projection lock measured: `~R` 0.34/0.72 deg median/max at bw 68 vs 35 deg for `R` -- `rotation_from_zyz` frozen; asymmetric-blob row/column check; scores and `rDen` measured-then-pinned; mosaic AHE vs kikuchipy AHE; Gaussian fit quirks; `nickel_ebsd_large` 20-point subset in the default suite
- [x] Measured mean-PC error floor: `nickel_ebsd_small` refined with `pc_average` vs per-point PC median 0.33 / max 0.54 deg (vs stored xmap 0.30 / 0.56); `nickel_ebsd_large` 165-point subset median 0.29 / p95 0.74 / max 0.96 deg (corr 0.97 with `|pc - pc_average|`; 20-point default-suite subset 0.28 / 0.76 / 0.82); Phase 6 coarse tolerances (small: median < 1.5, >= 8/9 < 3 deg) and Phase 7 refined (small: all < 1.0, median < 0.5; large weekly: median < 0.6, p95 < 1.2, max < 2.0 deg) derive from it
- [x] Spec approved (autonomous mode) and committed (728a7a47); failing tests + stubs committed (9d6d4def); implementation committed pre-review (1e3d0765), 231 tests green
- [x] Adversarial review (fidelity vs compiled C++ headers -- Gaussian fit/AHE/Cholesky bitwise, interpolatePixel accept-set identical on 252k points; conventions; 167+63-mutation bug injection) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop`

## Phase 6 -- `spherical-indexing-ebsd`
- [ ] `_indexer.py` (`SphericalIndexer`, per-pattern failure handling), `EBSD.spherical_indexing` (dask `map_blocks`, info message, masks, multi-phase, `n_best`), benchmark
- [ ] Public `kp.indexing.fast_bandwidths()` exported in `indexing/__init__.pyi` (ShtWisdom stand-in)
- [ ] Public `kp.indexing.SphericalBackProjector` exported with the indexer (one CHANGELOG entry)
- [ ] Tests: `nickel_ebsd_small` coarse vs stored xmap (median < 1.5 deg, >= 8/9 < 3 deg -- from the Phase 5 measured mean-PC floor, median 0.33 / max 0.54 deg, plus the half cell 1.33 deg at `bw` 68), lazy/verbose/mask/error paths, hard floor >= 2 pat/s/core, memory measured; `IndexEBSD.exe` parity runs use the default `emsphinx_compatible=True`

## Phase 7 -- `spherical-refinement`
- [ ] Analytic derivatives + Newton (`_derivatives`, `_refine_peak`, Cholesky 3x3, saddle rejection, degeneracy fallbacks), normalised refine, `refine=True` default, `EBSD.refine_orientation_spherical`
- [ ] Tests: `sht_xcorr.cpp` refine ports, saddle/beta=0, `nickel_ebsd_small` refined all < 1.0 deg, median < 0.5 deg + score increase, weekly `nickel_ebsd_large` (median < 0.6, p95 < 1.2, max < 2.0 deg) -- from the Phase 5 measured mean-PC floor (small 0.33 / 0.54, large 0.29 / 0.74 / 0.96 deg); `IndexEBSD.exe` parity runs use the default `emsphinx_compatible=True`

## Phase 8 -- `spherical-pseudo-symmetry`
- [ ] `_pseudo_symmetry.py` (`find_pseudo_symmetry_operators`, MasterXcorr port incl. two-phase mode, optional volume + stereogram), `pseudo_symmetry_ops` in indexer, psymfile read/write
- [ ] Tests: Ni autocorrelation peaks subset of Oh, `exclude_symmetry` empty, synthetic 3-fold/6-fold, wrong op -> index 0, local hcp masters skip-if-absent

## Phase 9 -- `sht-visualisation-and-interop`
- [ ] sht2png equivalents (stereographic option, `plot_power_spectrum`, `.plot()` conveniences -- `to_master_pattern` and `describe` ship in Phase 2), `SphericalBackProjector.plot`, xcorr volume plot (note: `extractBunge` (`sht_xcorr.hpp:594-649`) uses the reversed ZYZ->Bunge offsets; if ported, use `_euler.bunge_to_zyz` and record the deviation)
- [ ] `write_emsphinx_patterns` (PatternRepack contract in tech-stack.md), EBSPDims probe in `oxford_binary`, `EMSphInxNamelist` read/write (port of `test/util/nml.cpp` round trip), Sphinx-Gallery example
- [ ] Tests: stereographic r > 0.98, IndexEBSD.exe reads a repacked file (local-gated), namelist round trip; out-of-scope list confirmed in mission.md

## Phase 10 -- `spherical-indexing-emsphinx-regression`
- [ ] `create_emsphinx_reference.py` (pinned to 60f3517), `.npz` refs in-package, bidirectional tests (their binaries on our files; ours vs their `.ang/.h5`), `nregions in {0, 4, 10}`, two `delta` values

## Phase 11 -- `spherical-indexing-tutorial`
- [ ] `doc/tutorials/spherical_indexing.ipynb` (uses the full Ni master `ebsd_master_pattern("ni")` or caps bw <= 190 for the 401-px master; wall-clock budget derived from the Phase 6 benchmark, target <= 3 min on 8 threads with the pre-agreed fallback bw 53 / `s.inav[:40, :30]` / `refine=False` for sweep cells; outputs stored), `doc/tutorials/index.rst` entry, `NOTEBOOKS` entry in `doc/tutorials/run_nbval.sh`, `tutorials_sanitize.cfg` for timings, `.sht` cell in `load_save_data.ipynb`, CHANGELOG consolidation, `sphinx-build -b html` and `-b linkcheck`
