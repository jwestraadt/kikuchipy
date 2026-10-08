# Roadmap: spherical indexing (EMSphInx port: CPU reference + optional GPU backend)

Dependency chain: 0 -> 1 -> {2, 3} -> 4 -> 5 -> 6 -> 7 -> 9 (interop) -> 10 -> 11 (Phase 10 needs the Phase 2 `.sht` writer and the Phase 9 pattern repacker) -> 12 (GPU backend; needs 6 + 7 + 8).

**Re-scope (2026-09-02, Johan):** the target is indexing + binary regression +
tutorial. Phase 8 (pseudo-symmetry) and the visualisation half of Phase 9 are
**deferred** -- not dropped -- until after Phase 11. Phase 9 proceeds slimmed to
its interop pieces (`write_emsphinx_patterns`, `EBSPDims` probe,
`EMSphInxNamelist`), which are what Phase 10 needs; the Phase 11 tutorial omits
its pseudo-symmetry section until Phase 8 lands. Phase 8 bolts on cleanly
later: its `refine_zyz` hook shipped in Phase 7, and Phase 10's regression
scenarios do not exercise pseudo-symmetry. Phase numbering is kept stable
(specs and docstring guards reference phases by number).
Each phase is one branch and one dated spec folder `specs/YYYY-MM-DD-<name>/`
(`requirements.md`, `plan.md`, `validation.md`). A box is ticked only when the work is
committed on the phase's branch (verifiable with `git log`); a phase is complete
when every box is ticked, and the next phase's `plan.md` is drafted only then. The
gate list for every code phase is: plan approved -> spec recorded -> failing tests
committed -> implementation -> adversarial review + fixes -> pre-commit clean ->
CHANGELOG entry -> PR opened -> PR merged. Documentation-only phases skip the
failing-tests gate; they also skip the CHANGELOG gate unless they ship a
user-visible documentation deliverable (Phase 0 did not; Phase 11 does); phases
with no user-facing change (e.g. Phase 1, private code only) skip the
CHANGELOG gate. A phase's definition
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
- [x] Measured mean-PC error floor: `nickel_ebsd_small` refined with `pc_average` vs per-point PC median 0.33 / max 0.54 deg (vs stored xmap 0.30 / 0.56); `nickel_ebsd_large` 165-point subset median 0.29 / p95 0.74 / max 0.96 deg (corr 0.97 with `|pc - pc_average|`; 20-point default-suite subset 0.28 / 0.76 / 0.82); Phase 6 coarse tolerances (small: median < 1.5, >= 8/9 < 3 deg) and Phase 7 refined (measured at `bw` 68: small median 0.505 / max 0.695, large 20-pt 0.478 / 1.115, 165-pt 0.456 / p95 0.913 / max 1.140; small: all < 1.0, median < 0.75 pinned on the measured 0.505 with the Phase 6 margin convention (the a-priori < 0.5 assumed refinement reaches the mean-PC floor; the residual adds ~0.38 deg of bw-68 band-limitation -- at `bw` 88 the median is 0.450); large weekly: median < 0.6, p95 < 1.2, max < 2.0 deg; refined normalised scores rise (small 9/9, min +0.0108; the normalised refined score can dip where the window chain rule is omitted, 4/165 measured, worst -4.8e-4 -- `sht_xcorr.hpp:263-264` ported as-is)) derive from it
- [x] Spec approved (autonomous mode) and committed (728a7a47); failing tests + stubs committed (9d6d4def); implementation committed pre-review (1e3d0765), 231 tests green
- [x] Adversarial review (fidelity vs compiled C++ headers -- Gaussian fit/AHE/Cholesky bitwise, interpolatePixel accept-set identical on 252k points; conventions; 167+63-mutation bug injection) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#7)

## Phase 6 -- `spherical-indexing-ebsd`
- [x] `_indexer.py` (`SphericalIndexer`: per-phase `NormalizedSphericalCrossCorrelator` (plain when `normalize=False`) on a shared projector/Wigner table, `IndexEBSD` namelist defaults incl. the `[16, 512]` bandwidth rule, the harmonics-vs-detector `sample_tilt` binding guard (a 70-vs-65 mismatch indexes 4.7 deg wrong at *higher* scores -- measured), per-pattern failure handling extended to `ptp == 0` degeneracies (the AHE of a constant is `255 + O(1e-13)`, so EMSphInx would correlate rounding noise -- measured score 0.23, the one genuine deviation; a constant float correlates the window mask at -2.64, which EMSphInx's positive-score insertion rule then drops itself -- parity, failed earlier), the C++ zero-seeded insertion rule (`score <= 0` never recorded), `BatchEstimate` chunk sizing with a recorded `max(1, .)` clamp (34/15/6 at `bw` 53/68/88)), `EBSD.spherical_indexing` (dask threaded `map_blocks` with truthful `chunks=`, per-chunk `clone()` (0.2 ms at `bw` 68), bitwise deterministic across chunking/threads (CPU backend; the GPU backend of Phase 12 is deterministic run-to-run at fixed device/driver/batch size and tolerance-parity vs the CPU oracle, `specs/2026-09-07-spherical-gpu/` D5), info message printing the memory model (49 MB at `bw` 68), masks incl. a new boolean-dtype navigation-mask check, multi-phase `PhaseList` (orix drops losing phases from `xmap.phases` -- pinned via `nbest_phase_id`), `n_best` with the one-candidate-per-phase fill semantics), benchmark
- [x] Public `kp.indexing.fast_bandwidths()` exported in `indexing/__init__.pyi` (ShtWisdom stand-in)
- [x] Public `kp.indexing.SphericalBackProjector` exported with the indexer (one CHANGELOG entry)
- [x] Tests: `nickel_ebsd_small` coarse vs stored xmap (median < 1.5 deg, >= 8/9 < 3 deg -- from the Phase 5 measured mean-PC floor, median 0.33 / max 0.54 deg, plus the half cell 1.33 deg at `bw` 68), lazy/verbose/mask/error paths, hard floor >= 2 pat/s/core, memory measured; `IndexEBSD.exe` parity runs use the default `emsphinx_compatible=True`; measured coarse vs stored xmap at `bw` 68: small median 0.599 / max 0.838 deg (assert median < 1.5, >= 8/9 < 3, all < 2.0), large 20-pt 0.499 / 1.350 (median < 1.5, max < 3.0), weekly 165-pt 0.530 / p95 1.082 / max 1.495 (median < 1.5, p95 < 2.5, max < 3.5); scores/IQ measured-then-pinned (0.4963-0.6239 / 0.1727-0.2036 at `bw` 68; also pinned: signal_mask 0.4461-0.5762, circular_mask 0.4915-0.6390 with `n_points` 1117, gaussian_background 0.4942-0.6101); multi-phase discrimination via a sign-scrambled copy (`default_rng(42).choice([-1, 1])` -- same power spectrum, real function; 9/9, gaps 0.2970-0.4151; a rotated copy is degenerate -- the peak is rotation invariant, measured gaps +-0.015; kept as a control with the composed-orientation identity `rotation_from_zyz(zyz_b) * O_B == O_A`, 0.68/1.07 deg); hard floor >= 2 pat/s/core passed at 77.6 (bw 68); per-worker memory measured at `bw` 53/63/68/88/113 (peak 21/36/45/98/207 MB, model 23/39/49/108/228)
- [x] Adversarial review (pipeline fidelity vs indexer.hpp/idx.hpp -- no defect; conventions; 105-mutation bug injection, 10 tests added) and fixes; coverage 100 %
- [x] Signed commits pushed; PR opened into fork `develop` (jwestraadt/kikuchipy#8)

## Phase 7 -- `spherical-refinement`
- [x] `_derivatives` kernel with `error_model="numpy"` (the IEEE-degeneracy detector), `_refine_peak` (maxIter 15, `absEps = eps 2pi/slP`, monotone step, Cholesky 3x3 reused from `_preprocessing`, saddle rejection, 1x1/2x2 degeneracy fallbacks, failure returns the coarse triple with the analytic value), `refine_zyz` on both correlators (Phase 8 reuses it), normalised refine dividing by `denominator(eu)` (window chain-rule caveat ported as-is), `refine=True` **default** in `SphericalIndexer` and `EBSD.spherical_indexing` (per-candidate refine before insertion, `indexer.hpp:230`), `EBSD.refine_orientation_spherical` + `SphericalIndexer.refine_patterns` (the `msk & 0x02` work item with the *intended* semantics -- the shipped `refineImage` discards its refinement and stores a zero or stale score, `indexer.hpp:296`, `idx.hpp:406-407`, a newly recorded EMSphInx defect)
- [x] Tests: synthetic refined worst 2.96e-6 deg (30 symmetry-free cases, sizes 53-123 incl. padded) / 4.52e-6 (72 point-group cases) vs the C++ criteria 4.92e-3 / 0.351 deg; normalised wedge worst 1.85e-2 (`(1, F)` cases, gate 4.92e-2) resp. 2.13e-2 (`4/m` cases, gate 0.351 -- the C++'s own split, `sht_xcorr.cpp:316`, `:345`); `nickel_ebsd_small` refined **median 0.505** / max 0.695 deg (assert all < 1.0, median < 0.75 -- **the a-priori `median < 0.5` is amended: the measured value is 0.505** and the pin carries the Phase 6-style margin: the refined residual is the 0.33-deg mean-PC floor plus ~0.38 deg of bw-68 band-limitation and window caveat; at `bw` 88 the median is 0.450, under the old bound), scores up 9/9 (min +0.0108); large 20-pt refined 0.478/1.115 (median < 0.6, max < 2.0, deltas 20/20 > 0); weekly 165-pt refined 0.456 / p95 0.913 / max 1.140 (roadmap bounds median < 0.6 / p95 < 1.2 / max < 2.0 all pass; 161/165 scores up, the 4 dips are the un-applied window chain rule); refine-only-vs-refine=True equivalence 0.0 deg / 2.9e-14 score (assert < 1e-4 deg, < 1e-10); per-pattern refine+denominator 1.39 ms warm at `bw` 68 on 13.2 ms coarse (ratio 1.11x; C++ ~1.7x) (PR jwestraadt/kikuchipy#9)

## Phase 8 -- `spherical-pseudo-symmetry` (spec `2026-09-06-pseudo-symmetry`; un-deferred 2026-09-07)
- [x] `_pseudo_symmetry.py` (`find_pseudo_symmetry_operators`, MasterXcorr port incl. two-phase mode, optional correlation-volume ndarray -- the stereogram is deferred with the Phase 9 visualisation half, provisional per plan open question 9.3; psymfile read/write codec with the D2 conjugation contract; refines pseudo-symmetric candidates through `refine_zyz`, Phase 7) -- 100 % line coverage; faithfully reproduces the exe's identity-seed v_max stall (0.719309 at bw 88); codec golden-bytes pin
- [x] `pseudo_symmetry_ops` on `SphericalIndexer`/`EBSD.spherical_indexing` with the split row-width amendment (`_ROW_WIDTH_INDEX = 7`, `_ROW_WIDTH_REFINE = 6`) and the `pseudo_symmetry_index` prop, plus `MasterPatternHarmonics.rotate` (recorded scope growth on a Phase 2 class); refine-path ops NOT included (plan open question 9.1 unresolved -- refine rows stay width 6, pinned)
- [x] Tests incl. the D9 ladder (Ni positive-count pin then subset-of-Oh and `exclude_symmetry` empty, synthetic 3-fold/6-fold, ~3-deg dedup pair, wrong op -> index 0, two-phase composed-orientation oracle, local masters skip-if-absent) and the D8 gated binary pins (MasterXcorr parity, two-file branch, IndexEBSD psymfile inertness + error paths) -- measured: Ni pin 22 rows (identity refuted, see spec amendments), MasterXcorr parity 22 rows bijective at 2e-6 / intensities rel < 3e-5, rescue scenario 0.287/0.309 deg (18/18 across the seed/noise scan), ops scaling 56.8 -> 46.0 pat/s at 0 -> 4 ops (bw 68)
- [x] Adversarial review + fixes -- fidelity clean vs the C++; conventions applied (CHANGELOG added); bug-injection: all mutants killed after adding the brute-force local-maxima reference test
- [x] Signed commits pushed; PR #13 opened into fork `develop`

## Phase 9 -- `sht-visualisation-and-interop` (slimmed 2026-09-02: interop only, branch `sht-interop`)
- [x] `write_emsphinx_patterns` (PatternRepack port **plus the root `Manufacturer` dataset the C++ program omits** -- `IndexEBSD.exe` refuses a Manufacturer-less file, measured; vlen-**ASCII** Manufacturer -- h5py's default vlen UTF-8 is fatal with a misleading H5Dread error, measured; contiguous `/patterns` with alloc-time-early and **zero filters** (gzip is fatal, measured; alloc/layout are PatternRepack byte-parity only), layout/alloc/offset pinned via h5py -- PatternRepack's own repack of the in-package `patterns.ebsp`: offset 2144, 34544 B, rows flipped; native byte order enforced (`>u2` rejected by the binary, measured); `binAvg` == block mean rounded half away from zero, bitwise vs `PatternRepack.exe` bin 2, 1003/8100 pixels differ under banker's rounding; `binFloat` = float32 block *sum*, unreachable in the shipped binary (`binToFloat=false` const) so NumPy-pinned; per-manufacturer auto flip -- EDAX/EMsoft written unflipped and read-flipped, Oxford/Bruker pre-flipped, both routes measured correct, wrong pairing ~39.6 deg median; flip/bin commute exactly for divisible binning (measured b 2-6); uint16/float32 warned -- the buffered `NATIVE_UINT8` HDF5 read corrupts them, measured 38.9 deg median garbage, and the mmap path is dead code (`0 != getNfilters()` inversion, `pattern.hpp:494`); kikuchipy overwrite/suffix/directory conventions), `get_scan_info` in `oxford_binary` (EBSPDims contract: distinct exact-value beam-x/y sets + `len(x)*len(y) == n_present` regularity + coordinate lists; measured on `patterns.ebsp`: 3 x 3 regular), `EMSphInxNamelist` (nml.hpp parser semantics incl. the `test/util/nml.cpp` suite and the error cases probed through `IndexEBSD.exe` -- among them the column-0-only `!` comment rule, the unskipped whitespace-only line and the two-leading-spaces message; EBSD fields with `defaults()`/`sanity_check()` (13 checks, the negativity bounds live -- `nregions=-5`/`nthread=-1` exit 1, measured)/`to_string` at `.6g` -- line-parity with the captured `IndexEBSD -t` template, 119 lines; the four vendor PC conversions ported from `detector.hpp:85, 249-279`, pinned by bitwise-identical `.ang` output across EMsoft/EDAX/Oxford/Bruker namelists and delta-250-vs-500 Bruker runs (delta cancels for fractional vendors, measured), EMsoft == kikuchipy `pc_emsoft(version=4)` under `binning=1, px_size=delta`, TSL/Oxford deviate from kikuchipy's `pc_tsl`/`pc_oxford` on rectangular detectors -- frozen table, both orientations; the `circmask > 0` semantics (processor-side CircMask kept at radius r, `Geometry::circ` false -- `imprc.hpp:108-122`, `idx.hpp:230, 254`), the `tsl`-lowercase whitelist, the double-`ipath` quirk (reproduced on the derived `pat_path`, storage is raw), the qualmap-conditional `" /"` terminator)
- [x] Tests: IndexEBSD.exe indexes a kikuchipy-written repack + nml + in-package ni .sht (default route: Manufacturer EMsoft, unflipped; nthread=1 batchsize=1, bw 68): refined vs stored xmap median 0.7245 / max 0.9479 deg (assert median < 1.2, max < 1.6), scores mean 0.6283 (approx 0.628 rel 0.05), 112-120 pat/s recorded; Bruker-flip route recorded separately (0.713/0.947, mean 0.6304, equivalent not bitwise); kikuchipy-vs-IndexEBSD context 0.341/0.363 deg, r 0.9607 on the Bruker route (Phase 10's gate, recorded only); namelist round trip incl. template line-parity; out-of-scope list confirmed in mission.md (PR jwestraadt/kikuchipy#10)
- DEFERRED with Phase 8 (visualisation half): sht2png equivalents (stereographic option, `plot_power_spectrum`, `.plot()` conveniences -- `to_master_pattern` and `describe` ship in Phase 2), `SphericalBackProjector.plot`, xcorr volume plot, Sphinx-Gallery example, stereographic r > 0.98 test (note: `extractBunge` (`sht_xcorr.hpp:594-649`) uses the reversed ZYZ->Bunge offsets; if ported, use `_euler.bunge_to_zyz` and record the deviation)

## Phase 10 -- `spherical-indexing-emsphinx-regression`
- [x] `create_emsphinx_reference.py` in `src/kikuchipy/data/emsphinx/` (import-safe; pinned to 60f3517 via a `git rev-parse` hard assert; canonical route: `Manufacturer` EMsoft unflipped, nml vendor Bruker + `pc_average`, `nthread=1 batchsize=1`, `bw` 68, uint8 repacks byte-identical to `signal.data.reshape(-1, h, w)` -- guard-asserted; parses the datafile `Scan 1/EBSD/Data` (float32 Phi1/Phi/Phi2/Metric/IQ, uint8 Phase -- the `.ang` is a 5-decimal-Euler/3-decimal-ci/1-decimal-iq cross-check only, measured); generation-time acceptance guards (acid median < 1.2 deg, coarse/refined IQ bitwise); takes the machine-wide binary lock; measured ~11 s end-to-end, bitwise deterministic run-to-run **given a fixed fftw.wisdom** and md5-identical across full sweeps)
- [x] eight `regression_*.npz` refs in `src/kikuchipy/data/emsphinx/` (scenario matrix: coarse/refined x nregions **{0, 7, 10}** -- 7 replaces the pre-measurement 4 so the mosaic-AHE remainder path is binary-compared (measured 0.335/0.367 deg, IQ 6.6e-9) -- + gausbckg T + the EMsoft vendor route at d500; **the delta axis is retired as measured-inert**: EMsoft-route d250 = d500 bitwise through the binary AND bit-identical round-tripped pc for delta {125, 250, 500}, so the delta invariance is pinned as a namelist round-trip unit test instead; + large `[::15, ::15]` 20-pt and `[::5, ::5]` 165-pt subsets; float32 result arrays + provenance incl. the exact namelist text, the `.6g`-round-tripped `pc` the binary used, and patterns/master md5s; ~60 kB total, each < 100 kB, md5s in `_registry.py`, no URL; NO refine-only scenarios -- `refineImage` discards its refinement, research item 40)
- [x] bidirectional tests: ours-vs-theirs CI bands measured then pinned (refined median 0.31-0.34 -> < 0.7, max 0.34-0.49 -> < 0.75/0.8/1.0, coarse 0.510/0.622 -> < 1.0 median, >= 8/9 < 1.25, all < 4.0 (one grid cell = 2.667 deg); scores r 0.935-0.973 -> > 0.85/0.88/0.90 with mean|diff| < 0.03, max < 0.07; IQ max|diff| <= 1.4e-8 -> < 1e-3 (one uint8 gray level moves one pattern's IQ by up to 5.2e-5 -- the fastmath platform-drift ladder) -- the preprocessing discriminator; route pins: `from_kwargs` == the stored namelist text exactly, pc == its `.6g` round trip; the stretch-emulated diagnostic pins the item-31 decomposition (0.34 -> 0.094 deg); reference integrity guards labelled regeneration-only); theirs-on-ours = Phase 9 acceptance + the local-gated regenerate-and-diff (md5-identical, wisdom-qualified, large weekly); the a-priori refined < 0.2 / coarse < 0.5 / r > 0.98 are re-anchored on the measured item-31 decomposition (mission criterion 2 amended: the < 0.2 gate is met under stretch emulation)
- [x] adversarial review (fidelity to the recorded constraints, conventions, bug-injection list) + fixes
- [x] signed commits pushed; PR #11 opened (PR jwestraadt/kikuchipy#11)

## Phase 11 -- `spherical-indexing-tutorial`
- [x] `doc/tutorials/spherical_indexing.ipynb` (~50 cells, stored outputs, ~1 MB, 7 figures + IPF color key; in-package 401-px master at harmonics bw 188, a fast bandwidth under both the (401-1)/2 = 200 information limit and the cap 190 -- the full `ebsd_master_pattern("ni")` is a Note only, 305 MB download; indexes `nickel_ebsd_large` (4125 patterns) at bw 68 refined in ~20 s on 8 pinned dask workers (204-230 pat/s) on the 20-core drafting machine, coarse bw {53, 68, 88} sweep on a 1200-pattern subset ~20 s reporting median misorientation to the refined map (0.37/0.37/0.25 deg) + speed, Hough ~5 s, `refine_orientation_spherical` on the Hough map ~3.4 s (median to the spherical solution 0.229 -> 0.042 deg); validation vs in-notebook Hough (median 0.23 deg) and the shipped Hough+refined xmap (0.43 deg -- NOT a DI reference, per its 0.8.0 provenance); interop cells write `.sht`/repack/namelist to a temp dir; parity claim scoped to the Phase 10 regression suite per its D10 (0.34 deg at bw 68 vs the `ni_small_20kv_bw384.sht` master); measured total ~80 s end-to-end on the drafting machine -- the <= 3 min / 8-thread budget holds 2.2x there; fallbacks (bw 53, `inav` subset, `refine=False`) stay live for slower runners; the pseudo-symmetry section is omitted until Phase 8 lands (re-scope 2026-09-02))
- [x] `doc/tutorials/index.rst` entry (Indexing gallery, after `pattern_matching`)
- [x] nbval wiring: `NOTEBOOKS` entry in `run_nbval.sh` (alphabetical slot), `tutorials_sanitize.cfg` + regex8 (PyOpenCL bool) + regex9 (chunk counts); stored outputs pass nbval on the drafting machine 24/24 after the `_ = dask.config.set(num_workers=8)` fix (the bare repr's memory address was the one failure)
- [x] `.sht` section in `load_save_data.ipynb` (verified absent from Phase 2: zero grep matches) + format-table row (Read Yes / Write No -- plugin `writes: False`); CHANGELOG consolidation 9 -> 3 entries + the tutorial entry = 4 bullets (PR #12 link)
- [x] adversarial review (validation matrix + failure-mode list -- clean-kernel execute, nbval, html render inspection, linkcheck of the phase's links via output.json, name/spell pass) + fixes; `sphinx-build -b html` exit 0
- [x] signed commits pushed; PR #12 opened

## Phase 12 -- `spherical-indexing-gpu` (spec `2026-09-07-spherical-gpu`)
- [x] `_gpu.py` (three-stage CuPy gate + Windows DLL shim, xp-agnostic device pipeline for stages 4-6, VRAM batch model + OOM halving, per-call `_GpuSession`) -- 100 % coverage (296 stmts); VRAM model calibrated vs live pool high-waters (g(68) model 49.9 MB vs 52.4 measured)
- [x] `backend="cpu"|"gpu"` on `SphericalIndexer`/`EBSD.spherical_indexing` + the `_index_chunk` batched restructure + `gpu_memory_per_batch_bytes` (CPU default bitwise-unchanged, pinned; CPU suite 3188 passed / 0 failed after the epilogue extraction)
- [x] tests -- default-suite numpy-xp oracle + validation logic; local `cupy_gpu` gated parity/determinism/throughput suite (KIKUCHIPY_NO_GPU_TESTS kill switch; measured pins machine-specific: RTX 2000 Ada 8 GB) -- gated 103 passed / 0 skipped under KIKUCHIPY_EXPECT_GPU=1; parity exact everywhere measured (IQ bitwise, 0 winner flips, refined miso 0.0 deg, batch invariance bitwise)
- [x] performance go/no-go: PASS -- 940.7 pat/s best-of-3 refined at bw 68 vs the pinned 236.0 pat/s idle-CPU floor (3.99x, inside the review-corrected 850-1500 band); bw 88 measured 465 pat/s, below its recorded expectation band, explanation recorded
- [x] adversarial review + fixes (incl. the device mutation list) -- 20 findings dispositioned; 4 surviving mutants killed by strengthened tests, kills verified by re-injection
- [x] signed commits; PR #15 opened into fork `develop`

---

# Feature path: HREBSD-DIC (branch `hrebsd-dic`; spec `2026-09-07-hrebsd-dic`)

Separate from the completed spherical phases above; nothing here touches the
spherical mission or its criteria. Homography-based HR-EBSD by
inverse-compositional Gauss-Newton DIC (Ernould et al. 2020/2022) plus the
analysis chain (strain/stress/rotation, HR-KAM, PC shift, scalar GND), written
new from the literature -- no upstream source implements the chain
(EMsoftOO's EMHREBSDDIC is a WIP equation-level reference only, never a
regression target). **Branch policy (user decision 2026-09-07): every commit
of this feature stays on `hrebsd-dic`, pushed to origin but NEVER merged into
`develop`; no PR into `develop` is opened.** The gate list matches the
spherical phases except the two PR gates are replaced by "signed commits
pushed to origin/hrebsd-dic + boxes ticked here"; fork CI triggers on push as
well as PRs (corrected 2026-09-07 at spec review), so pushes of `hrebsd-dic`
run the full matrix as an extra signal (failing-tests commits are pushed
together with their implementation commit, never alone), while the local
gates recorded in
`specs/2026-09-07-hrebsd-dic/validation.md` carry the recorded verification
burden (incl. a recorded local oldest-matrix run per stage). Stages
A -> B -> C, one spec folder.

## Stage A -- IC-GN engine (`EBSD.hrebsd_dic`)
- [x] `src/kikuchipy/indexing/_hrebsd/` engine: numba bicubic kernel (D3), band-pass/border/dead-band preprocessing (D4), phase-XC initial guess (D5), IC-GN with accumulated-W re-warp + corner-norm convergence (D2), homography<->Fe with per-point PC/DD and the beam-scan correction applied BEFORE conversion (D6), `EBSD.hrebsd_dic()` with the frozen signature returning a CrystalMap with homography/Fe/residual/iteration/convergence/grain/reference props (D15); explicit reference modes (`reference="auto"` stubs NotImplementedError until Stage B)
- [x] failing tests first: V0 kernel equality, V1 round trips + direction pin, V2 warp-refit (MTP pins), V3 deformed-master homography recovery, V4 pure-rotation frame/sign pins, V6 PC-shift phantom sign pins, determinism/NaN/mask/get_map_data pins
- [x] measured + recorded: D3 bicubic-vs-quintic decision, D17 f32/f64 verdict, D6.3 signs, D5 capture range, performance baselines
- [x] adversarial review (fidelity/theory + conventions/integration + mutation list) + fixes; coverage 100 % of Stage A `_hrebsd/` modules; full suite green; oldest-matrix run recorded
- [x] signed commits pushed to origin/hrebsd-dic (no PR)

## Stage B -- strain/stress/rotation + references + PC + HR-KAM
- [x] polar decomposition + Biot strain (D8), traction-free sigma33=0 closure with user 6x6 Voigt stiffness / deviatoric fallback (D9), stress + von Mises/hydrostatic/principal maps (D10), `segment_grains` + per-grain best-IQ auto-reference wired into `reference="auto"` (D11), `hrebsd_kam` in mrad (D12), `hrebsd_pc_shift` (D13), `hrebsd_strain_stress` + `voigt_stiffness` public (D15)
- [x] failing tests first: closure/Bond-rotation/derived-map pins, segmentation suite, V3 strain half, V7 KAM identity, V6 function tests
- [x] Si-wafer noise-floor benchmark recorded (strain/rotation/KAM floors; preprocessing/border/KAM sweeps resolve plan open questions 2-4, 10) -- dataset fetched once 2026-09-08, floors 1.22e-02 strain / 1.20e-02 rad / 5.24 mrad KAM RECORDED AS THIS DATASET'S and not the method's (validation entries 43 to 45 measure why), all four open questions closed with every frozen default confirmed
- [x] adversarial review + fixes; coverage; full suite green; oldest-matrix run recorded
- [x] signed commits pushed to origin/hrebsd-dic (no PR)

## Stage C -- GND + tutorial
- [x] `hrebsd_gnd`: exact alpha_i3 + the d/dx3-neglect extra components (Pantleon-tier assumption, D14.2), detector-frame antisymmetry fix, OpenXY 3/5/9-component estimators with literal prefactor pins, m^-2 log-scale maps (D14)
- [x] failing tests first: V7 constant-curvature oracle (validated against the math, never another code), end-to-end curvature tolerance (MTP), NaN safety, Si GND floor recorded
- [x] `doc/tutorials/hrebsd_dic.ipynb` (synthetic walk-through + Si noise floor + map gallery + documented limitations), index.rst + nbval wiring, CHANGELOG (fork-only wording), bibliography keys
- [x] adversarial review + fixes; coverage; full suite green; oldest-matrix run recorded
- [x] signed commits pushed to origin/hrebsd-dic (no PR)

## Si-indent application (plan open question 13; in execution 2026-09-08)

Full-resolution replication of Winkelmann et al. 2025 (Ultramicroscopy 276,
114180) on Zenodo 14059950 with the completed chain; scope per the dated
OQ13 note (full-res only, deviatoric only; super-resolution and stress
deferred). Execution plan: `specs/2026-09-07-hrebsd-dic/si-indent-application-plan.md`.

- [x] housekeeping: `.gitignore` guard for `*.h5oina`, spec bookkeeping (OQ13 note, this section, ledger heading)
- [x] rectangular end-to-end regression test (first non-square `EBSD.hrebsd_dic` run) green
- [x] oxford_h5ebsd binning-read fix ('Camera Mode' vs 'Camera Binning Mode' at format 7.0) + dedicated test
- [x] pre-flight measurements recorded (CCC reference resolution, filter_cutoffs A/B on row 10, measured patterns/s)
- [x] `doc/tutorials/hrebsd_si_indent.ipynb` authored, adversarially reviewed BEFORE the one-shot execute, then executed with stored outputs
- [x] post-execute review (markdown claims vs printed numbers) + nbval/ruff/full-suite gates recorded; signed commits pushed (no PR)

Deferred follow-up paths (performance, super-resolution, stress) are
recorded with baselines and resolving measurements in
`specs/2026-09-07-hrebsd-dic/plan.md` section 9; none is commissioned.

## Stage D -- neighbour-seeded propagation (commissioned 2026-09-09; D20, V8)

VERDICT (measured, ledger 78-88): shipped default-off as a CORRECTNESS
lever, NOT the speed lever it was commissioned as. The plan 9.4 3-8x
projection is REFUTED on real Si (0.56x, i.e. 1.8x slower: the real
steep-gradient field's ~87 iterations/point cancel the cascade saving);
the proven benefit is the synthetic V8(a) basin rescue only (large
about-normal rotation past the ~2 deg phase-XC capture range, which the
default fit misses at any budget). On real Si the regime does not arise,
so zero correctness gain there and a small un-gated error-propagation
risk (3/618 points to a worse optimum). Ships default-off so non-users
pay nothing. Two follow-ups recorded below, neither commissioned.
- [x] failing tests first: default-off bitwise pin, V8 rescue/equivalence/determinism/isolation oracles, seed_round encoding, tie-order pin
- [x] implementation: per-point h0, cascade rounds + rescue pass, seed_round prop; PASS1_CAP kept at 50 (far-field p95 = 10 iters, 5x margin; rim-panic was a patch artifact) + SEED_EQUIVALENCE_TOL/SEED_RESCUE_TOL/PC-transport bound measured and pinned
- [x] adversarial review + bug injection (13 mutants, all killed) + fixes; coverage 100 % of _hrebsd; default path bitwise unchanged; oldest-matrix + full suite green
- [x] D20.7 performance record (seeded vs default, honest 0.56x); signed commits pushed to origin/hrebsd-dic (no PR)

### Stage D follow-ups (recorded 2026-09-09, NOT commissioned)
- [ ] residual-acceptance gate on the cascade: reject a cascade fit whose
  residual exceeds the point's independent-fit alternative (or a
  threshold) and fall back; plan 9.4 called for it, and its absence is
  what let 3 real-Si rim points seed into worse optima. Makes seeding
  never-worse-than-default when used. (D20.7 finding, ledger 84.)
- [ ] revisit PASS1_CAP downward (near 15-20) WITH the cascade
  round-batching overhead, only if real-data speed is ever wanted; would
  trim pass-1 waste but moves real-data seed_round assignments.

## Stage E -- GPU backend for the DIC engine (commissioned 2026-10-06; D21, V9)

Plan section 9 item 5, commissioned 2026-10-06; plan section 11 carries the
tasks, open questions and the recorded defaults awaiting Johan's approval.
Optional `backend="gpu"` (CuPy) on `EBSD.hrebsd_dic`: the CPU path stays the
default, the reference and the parity oracle; no silent fallback; the seed
stage is a frozen, pluggable seam (D21.5); `seed_from_neighbors=True` with
`backend="gpu"` raises (recorded default, approved 2026-10-06, D21.12). Spec-gate
prototype (throwaway, ledger 89-99), at the "mixed" device precision and with
a dedicated prototype reader thread: the whole Si-indent map in 166.5 s = 347
pat/s with the complex128 seed (51x the 8-worker CPU's 2.34 h) and 92.0 s =
628 pat/s with complex64 (92x), with the CPU run's convergence counts. Under
"float64" (the parity and debug build; the D17 amendment that makes "mixed"
the device default was approved 2026-10-06),
the map projects at about 440 s = 131 pat/s (about 19x; an inference, ledger
98). Spec review 2026-10-06: 34 critic findings, all applied (plan 11.5).
- [x] plan approved by Johan: the section 11.4 recorded defaults, the D17
  amendment (D21.4) and the mission/tech-stack amendments (HREBSD sections,
  dated 2026-10-06) -- approved as written 2026-10-06 under his overnight
  waiver (plan 11.4 approval record)
- [x] failing tests first: `test_hrebsd_gpu.py` (default numpy-xp suite + gated
  `cupy_gpu` suite, V9(a)-(p)) incl. the seed-seam h0 oracle and the
  `seed_from_neighbors` raise pin; freeze, defaults and import-audit pins
  updated
- [x] implementation: `_hrebsd/_gpu.py` gate (imports the Phase 12 shim, floor
  and lock; no `_spherical` edit) + session + VRAM model + device runner;
  xp-agnostic batched core with the seed seam and lockstep IC-GN; `backend`
  plumbing; every V9 MTP pin measured and pinned
- [x] adversarial review + bug injection (plan 11 item 4 mutants, kills
  re-verified on the GPU machine) + fixes; coverage 100 % of the touched
  `_hrebsd` modules (default + gated combined); default path bitwise
  unchanged; gated suite 0 skipped under KIKUCHIPY_EXPECT_GPU=1;
  oldest-matrix + full suite green
- [x] D21.16 performance record on the Si-indent data (far256, patch C, whole
  map, both seed and both device precisions, the E8 device-wait fraction) +
  go/no-go floor; docstring, CHANGELOG, tutorial
  markdown cell; signed commits pushed to origin/hrebsd-dic (no PR)

### Stage E follow-ups (recorded 2026-10-07, NOT commissioned; plan 11.3)
Measured at the review and performance gates (ledgers 105-114). Stage E
result: the whole Si-indent map in 358.8-375.7 s on the GPU (22.4-23.5x the
8-worker CPU's 8424 s) at the default precisions, 32-36x with the complex64
seed, 5.4-5.7x under "float64", convergence counts identical to the CPU.
- [ ] skip retired slots in `gather` and `pixel_sums` (41 per cent of slot
  iterations do work on the map; about 2x)
- [ ] ordered batch reading (E8: 19-24 per cent start-up wait, host peak
  15-16 GB)
- [ ] multi-grain batches or B from grain size (E5: 1.6-1.9x on small grains)
- [ ] amend the D21.8 per-point bands for convergence knife edges (E12),
  instead of the 2.34 h CPU re-run
- [ ] decide a public `seed_precision` (E13: 1.44-1.52x)
- [ ] adopt the Toolkit-free overlay in the gate commands (D21.15 amendment)

## Stage F -- Fourier-Mellin rotation initial guess (commissioned 2026-10-06; D22, V10)

Plan open question 5 and the D5 deferral, commissioned 2026-10-06 under
Johan's extended overnight waiver and specified 2026-10-07; plan section 12
carries the tasks, open questions FQ1-FQ14 and the recorded defaults,
approved by Johan on 2026-10-08 as written ("Go ahead with stage F";
plan 12.4 approval record). An opt-in
`fourier_mellin` keyword (`"off"` default, bitwise unchanged; `"auto"`;
`"always"`) on `EBSD.hrebsd_dic`: a rotation-only Fourier-Mellin angle from
the REUSED target spectra (periodic-Hann frequency stencil, log-magnitude,
polar radial-mean ZNCC, +-30 deg window), the target de-rotated through the
existing warp kernel, the Stage E translation step on the de-rotated crop,
the partial row `R(theta) T(t)`, kept only if its criterion at the seed is
lower; one xp-agnostic code path behind the D21.5 seam for both backends
(a new `SeedBatch.outputs` field, a 14-wide packed row on FM runs only), the
CPU route at P = 1 as the parity oracle. Under `"auto"` a host-side gate on
the input CrystalMap's twist about the detector normal (1.5 deg, failing
open); under `"auto"` and `"always"` one post-fit retry of non-converged
points from a forced FM row. Spec-gate prototypes (CPU only, throwaway,
ledger 115-131): the translation seed fails from 2.5 deg on the 480 px
oracle at the default budget; on rigidly rotated real Si it fails from 3 deg
with the default band-pass `(0.05, None)` but reaches 6 deg under the
production `(None, None)` -- above the Si map's largest input twist of 2.6
deg, so there the case for FM is iteration cost and the D5 zero-shift
anchor, not capture. The FROZEN recipe (the V8 angle, the partial row, the
acceptance) converged every oracle case measured, pure twists from -20 to
30 deg in 2 to 3 iterations to the exact seed's error (30 deg is the
window's edge), and cut the iterations on 16 real rim points by 17 per cent; the
second prototype's whitened recipe (not the frozen one) converged a 113-case
sweep to 30 deg and real rotations from -6 to 10 deg. About 0.6 per cent of
the Si-indent map has an input twist above 1.5 deg. Spec review 2026-10-07:
45 critic findings, 44 applied, 1 rejected (plan 12.5). Starts only after
the Stage E gates pass AND Johan has reviewed plan 12.4.
- [x] plan approved by Johan: the section 12.4 recorded defaults (approved
  as written 2026-10-08, "Go ahead with stage F"; recorded in 12.4)
- [ ] failing tests first: `test_hrebsd_fourier_mellin.py` (default numpy-xp
  suite + gated `cupy_gpu` suite, V10(a)-(l)) incl. the angle, look-up-table,
  edge, row, capture, ramp-rescue, acceptance, gate, projection-link, retry,
  CPU-route and numpy-session oracles, the public-method forwarding and
  props arms and the new docstring test; the freeze, defaults and D21.12
  literal pins edited with dated comments
- [ ] implementation: `_hrebsd/_fourier_mellin.py` (look-up table, angle,
  peak, de-rotation, translation, partial row, acceptance, CrystalMap gate
  and routes) + the seam extension (`SeedState.fourier_mellin`, the `extras`
  route key, `SeedBatch.outputs`) + the CPU route at P = 1 + the retry pass
  under both FM modes + 14-wide rows on FM runs + `_run_chunks_gpu(
  seed_extras=)`, the lazy FM state, the inactive forced slot and the VRAM
  keyword + `ebsd.py` forwarding and props; every V10 MTP pin measured and
  pinned; the REQUIRED real-data frame oracle of the gate under its
  pre-registered rule (FQ13)
- [ ] adversarial review + bug injection (plan 12 item 4 mutants FM1-FM52,
  kills re-verified, device ones on the GPU machine) + fixes; coverage 100 %
  of the touched `_hrebsd` modules (default + gated combined); CPU default
  path bitwise unchanged (`fourier_mellin="off"` == no keyword); Stage E
  gated suite green and unchanged; Stage F gated suite 0 skipped under
  KIKUCHIPY_EXPECT_GPU=1; oldest-matrix + full suite green
- [ ] D22.15 performance record on the Si-indent data (whole map `"off"`
  against `"auto"` at both filter settings on both backends, far256 and
  patch C under `"always"`, the D5 anchor census of FQ8); docstring,
  CHANGELOG, the `hrebsd_dic.ipynb` bullet and the `hrebsd_si_indent.ipynb`
  markdown cell; signed commits pushed to origin/hrebsd-dic (no PR)

---

# Feature path: NLPAR (branch `feat-NLPAR`; spec `2026-10-04-nlpar`)

Not a Phase 13: NLPAR is not in the EMSphInx dependency chain above.
Base is fork `develop` (18d59c07); one fork PR `feat-NLPAR -> develop`
(expected jwestraadt/kikuchipy#17, confirmed with `gh pr list` at PR
time), merged only on the user's go with ubuntu/windows CI green
(the go was given in advance on 2026-10-04, so the merge proceeds
when CI is green; macOS known red: the ebsdsim step and the pre-existing pseudo-symmetry
`23 == 22` test). A box ticks only when the work is committed on
`feat-NLPAR` (`git log`). Gate list per code stage: spec recorded ->
failing tests committed -> implementation -> adversarial review + bug
injection + fixes -> pre-commit clean -> CHANGELOG entry -> pushed.
Stage C is documentation: it skips the failing-tests gate and keeps the
CHANGELOG gate (it ships a tutorial).

## Stage A -- engine
- [x] `src/kikuchipy/pattern/_nlpar.py`: five `@njit(cache=True, nogil=True)` kernels (`_window_bounds`, `_nlpar_sigma_kernel`, `_nlpar_distances_kernel`, `_nlpar_weights_kernel`, `_nlpar_weighted_sum_kernel`; no `parallel`, no `fastmath`), chunk wrappers reading `block_info`, the NLPAR depth helper, the eager two-pass driver; NRL change notice in the module header
- [x] `EBSD.average_non_local_neighbour_patterns()` (eager signals; `lam=None`, lazy input and `lazy_output=True` raise `NotImplementedError` until Stage B) and `EBSD.get_nlpar_sigma()` directly after `average_neighbour_patterns` in `signals/ebsd.py`, then `EBSD.get_nlpar_lambda()` as a stub raising `NotImplementedError` until Stage B (the fifth guard)
- [x] Tests: `tests/test_signals/test_util/test_nlpar.py` (kernel discipline + `.py_func`, PyEBSDIndex kernel oracles skipif incl. the compiled parity arms on `nickel_ebsd_large` [download], depth/`calclim`, the two pyebsdindex-free multi-chunk driver tests, policy oracles), `tests/test_signals/test_ebsd_nlpar.py` (float64 NumPy reference, identities, iid-noise, two-grain, method contracts incl. the multi-chunk in-place arm on `nickel_ebsd_large`) and the four synthetic generators as fixtures in the root `conftest.py`; every tolerance measured then pinned in `validation.md`
- [x] Adversarial review (fidelity vs the paper, both PyEBSDIndex kernels and EMsoftOO `mod_NLPAR.f90`; conventions/integration) + bug injection (M1-M22 and S1-S8, plan section 6, Stage A rows) + fixes; coverage 100 % of `_nlpar.py` recorded
- [x] Gates: `-n 0` then `-n 4` (red tests re-run alone), full suite, `SKIP=licenseheaders` pre-commit on explicit files, oldest-matrix recipe, clean-replay grep; CHANGELOG "Added" bullet with the fork PR link; signed commits pushed (the failing-tests commit never alone)

## Stage B -- optimisation and scale
- [x] `EBSD.get_nlpar_lambda()` and `lam=None` (phantom-free Nelder-Mead on the pass-1 distances, `target_weight` 0.34, bounds [1e-3, 10], result logged, bound hit warned); lazy input and `lazy_output` through the two `overlap` + `map_blocks` passes (core-only chunk wrappers) with the depth helper and the minimum-chunksize rechunk
- [x] Pins: eager == lazy bitwise through the method over single, regular, irregular tiny-edge and thinner-than-depth ROW chunkings (a lazy input's column chunking is collapsed by `get_dask_array`'s `_reduce_chunks`; column and both-axes chunkings are pinned at driver level in Stage A) and the explicit (26, 26, 3) last-chunk rows on `nickel_ebsd_large`; issue-230 chunking; 1-D scan == (1, n); scheduler/thread invariance; `nickel_ebsd_large` in the default suite (`allow_download=True`, cached: full-map ADP/IQ gain, the phantom-ratio arm, Hough indexing on the `inav[::5, ::5]` 165-pattern subset; lambda seeds 2026-09-11: 1.1164 / 2.5246 with phantoms, 1.1387 / 2.5787 phantom-free) with the full-map Hough and `si_wafer` weekly; performance baselines recorded, never gated
- [x] Adversarial review + bug injection (lambda and lazy mutants) + fixes; coverage 100 % of `_nlpar.py` re-recorded
- [x] Gates as Stage A; CHANGELOG bullet extended for `lam=None`; signed commits pushed

## Stage C -- tutorial
- [x] `doc/tutorials/nlpar.ipynb` (formulas + acknowledgement; synthetic two-grain demo with sigma map and boundary preservation vs Gaussian `average_neighbour_patterns`; `nickel_ebsd_large` sigma map, lambda-vs-target curve, before/after patterns, IQ/ADP maps, Hough indexing before/after; `si_wafer` numbers quoted from the ledger; parameter guidance; differences from PyEBSDIndex and from upstream #824); `hybrid_indexing.ipynb` untouched, linked
- [x] Registration: `doc/tutorials/index.rst` after `pattern_processing`, `NOTEBOOKS` entry in `run_nbval.sh`, `tutorials_sanitize.cfg` regexes as needed, stored outputs if > ~2 min on the RTD builder; gallery example `examples/pattern_processing/nlpar.py`
- [x] Validation matrix + failure-mode review (clean-kernel execute, nbval, html render inspection, linkcheck, name/spell pass) + fixes; `sphinx-build -b html` exit 0
- [x] CHANGELOG tutorial bullet; signed commit pushed; the three spec documents re-submitted to review (definition of done)

## Fan-out (plan section 1; after the merge)
- [x] Fork PR `feat-NLPAR -> develop` opened with the PR template (number confirmed; CHANGELOG links rewritten if not #17); roadmap tick commit "Tick NLPAR boxes in roadmap (jwestraadt/kikuchipy#17)" -- opened as jwestraadt/kikuchipy#17 (2026-10-06); number confirmed, CHANGELOG links unchanged
- [x] PR merged on the user's go (merge commit; ubuntu/windows CI green); merge sha M recorded here -- merged 2026-10-06 as ef29024a (CI: ubuntu py3.13, windows py3.13/3.14, oldest and wheel green on run 37426647339 after the default suite was trimmed and the fork-only test timeout raised to 20 min; macOS only the pre-existing test_ni_proper_oh_count)
- [x] `hrebsd-dic`: `git merge --no-ff develop`, append-type conflicts resolved HREBSD first then NLPAR; `-k "nlpar or hrebsd"` then the full suite green; nbval on `nlpar.ipynb`; pushed; still never merged into `develop` -- merged as 1532813d plus the comment fix 091b0cbf, pushed; full suite 5212 passed / 0 failed
- [x] `feat-spherical-indexing-nlpar`: clean replay of M with `pick.ps1`/`gate.ps1` as two commits ("Add non-local pattern averaging (NLPAR)", "Add NLPAR tutorial"; `Staged-from:` trailers), equivalence gate and `specs/` grep clean, worktree suite == baseline + NLPAR tests; pushed, no PR; `feat-spherical-indexing` stays at 6723aaf0 -- 03c3ca47 + 8571c081 (Staged-from #17, ef29024a, with the comment fix 4ea4626f folded in; the fork-only tests.yml timeout excluded), pushed; equivalence 9577/9577 lines; worktree full suite 4515 passed / 1251 skipped (= baseline 4118 / 827 + NLPAR 397 / 424); nbval 18/18

---

# Feature path: HROSM (branch `feat-HROSM`; spec `2026-10-06-hrosm`)

Not a Phase 13: HROSM is not in the EMSphInx dependency chain above.
Base is fork `develop` (de27741a); one fork PR `feat-HROSM -> develop`
(expected jwestraadt/kikuchipy#20, confirmed with `gh pr list` at PR
time), merged only on Johan's go (no advance go for HROSM) with
ubuntu/windows CI green; macOS known red on the pre-existing
`test_ni_proper_oh_count` (`23 == 22`). Johan approves
`specs/2026-10-06-hrosm/plan.md` before any test or code is written. A
box ticks only when the work is committed on `feat-HROSM` (`git log`).
Gate list per code stage: spec recorded -> failing tests committed ->
implementation (Stage A: plus the EMsoft reference run) -> adversarial
review + bug injection + fixes -> pre-commit clean -> CHANGELOG entry
-> pushed. Stage C is documentation: it skips the failing-tests gate
and keeps the CHANGELOG gate (it ships a tutorial).

## Stage A -- orientation engine (no dictionary indexing)
- [x] `src/kikuchipy/indexing/_hrosm/` (`_emsoft_quaternions`, `_kam`, `_segmentation`, `_grains`, `_averaging`, `_directional_statistics`, `_sampling`, `_osm` with both OSM forms as array functions, `_emsoft_file`); public `GrainTable`, `average_grain_orientations`, `grain_bounding_boxes`, `grain_reference_orientation_deviation_map`, `kernel_average_misorientation_map`, `misorientation_ball`, `misorientation_ball_spacing`, `segment_grains_kam` in `kikuchipy.indexing`; the EMsoftOO BSD-3 block in every EMsoft-derived module; no numba kernels
- [x] EMsoft references: `src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py` (import safe) run once on this machine (EMDI, EMFitOrientation, EMgetOSM, EMHROSM `center`/`center_dilate`/`wat`, EMsampleRFZ N 6 and 20; pre-flight and acid bands passed); the shipped `.npz` files with provenance (`program_md5` of every program and DLL), md5s in `_registry.py`
- [x] Tests: `tests/test_indexing/test_hrosm_{kam,segmentation,averaging,sampling,osm,emsoft_regression}.py` (compat bitwise against literal transcriptions and the shipped references, correct mode against analytic fields and orix, recovery bands, reader on synthetic EMsoft-layout files, reference-file quad equality); the bin-gated arms (EMsampleRFZ N 20, regenerate-and-diff) and the local-gated arms (Ni6, GRX810, Al) run once and recorded; every tolerance measured then pinned in `validation.md`
- [x] Adversarial review (fidelity vs EMsoftOO `mod_DIsupport`, `mod_cluster`, `mod_so3`, `mod_dirstats`, `mod_Lambert`, `mod_quaternions`; conventions/integration) + bug injection (plan section 6, Stage A rows) + fixes; coverage 100 % of the Stage A `_hrosm/` modules recorded
- [x] Gates: `-n 0` then `-n 4` (red tests re-run alone), full suite, doctests, `SKIP=licenseheaders` pre-commit on explicit files, oldest-matrix recipe, clean-replay grep, default-suite budget measured; CHANGELOG "Added" bullet with the fork PR link; signed commits pushed (the failing-tests commit never alone)

## Stage B -- per-grain re-indexing and `EBSD.hrosm()`
- [x] `src/kikuchipy/indexing/_hrosm/_driver.py` and `EBSD.hrosm()` directly after `dictionary_indexing` in `signals/ebsd.py` (validation order, absent points, grains per phase, the ball composed per grain, eager experimental block, chunked lazy dictionary, `pc="grain"|"single"`, `verbose` 0/1/2, the GROD coverage warning before any simulation, warnings, one output `CrystalMap` with the documented props)
- [x] Backwards-compatible upstream touches: `orientation_similarity_map(..., *, grain_id=None, emsoft_compatible=False)` (legacy path bitwise unchanged at the defaults) and `_dictionary_indexing(..., verbose=True)`
- [x] Tests: `tests/test_signals/test_ebsd_hrosm.py` (contracts, masks, PC policy, skips, `n_per_iteration` and lazy/eager invariance, determinism, warnings, physics sanity on a synthetic sub-grain map, end-to-end tolerance against the EMHROSM references: one grain weekly, full map local + weekly) and additions to `test_orientation_similarity_map.py` and `test_dictionary_indexing.py`; performance baselines recorded as local ledger runs, never gated
- [x] Adversarial review + bug injection (Stage B rows) + fixes; coverage 100 % of `_hrosm/` re-recorded
- [x] Gates as Stage A; CHANGELOG bullet extended with `EBSD.hrosm()` and the two keywords; signed commits pushed

## Stage C -- tutorial
- [x] `doc/tutorials/hrosm.ipynb` (`nickel_ebsd_large`: dictionary indexing and refinement as in `pattern_matching.ipynb`, then `EBSD.hrosm()`; global OSM next to the HROSM OSM; KAM, grain map, GROD and its warning; synthetic sub-grain demonstration; parameter guidance and cost; differences from EMsoftOO's EMHROSM in words); `pattern_matching.ipynb` and `spherical_indexing.ipynb` linked, never edited
- [x] Registration: `doc/tutorials/index.rst` after `pattern_matching`, `NOTEBOOKS` entry in `run_nbval.sh`, `tutorials_sanitize.cfg` sections (if any) numbered from `[regex20]`, stored outputs if > ~2 min on the RTD builder; gallery example `examples/indexing/hrosm.py` with the new section file `examples/indexing/README.rst`
- [x] Validation matrix + failure-mode review (clean-kernel execute, nbval, html render inspection, linkcheck, name/spell pass) + fixes; `sphinx-build -b html` exit 0
- [x] CHANGELOG tutorial bullet; the three spec documents re-submitted to review and the amendments folded in (definition of done); then the signed commit pushed

## Fan-out (plan section 1; after the merge)
- [x] Fork PR `feat-HROSM -> develop` opened with the PR template (number confirmed; CHANGELOG links rewritten if not #20); roadmap tick commit "Tick HROSM boxes in roadmap (jwestraadt/kikuchipy#20)" -- opened as jwestraadt/kikuchipy#20 (2026-10-07); number confirmed, CHANGELOG links unchanged
- [x] PR merged on Johan's go (merge commit; ubuntu/windows CI green); merge sha M recorded here -- merged 2026-10-07 by Johan as 90453bc3 (CI on cdcfced9: ubuntu, windows, oldest (one xdist worker crash in the nlopt refinement test, rerun green) and wheel green; macOS only the pre-existing test_ni_proper_oh_count)
- [ ] `hrebsd-dic`: `git merge --no-ff develop`; the stub imports and `__all__` resolved in sorted order, the other append-type conflicts HREBSD first then HROSM; `segment_grains`/`segment_grains_kam` See Also cross-references added on `hrebsd-dic` only; the HROSM tests, `-k hrebsd`, then the full suite green; nbval on `hrosm.ipynb`; pushed; still never merged into `develop` -- DEFERRED 2026-10-07: the hrebsd-dic worktree (Repos\kikuchipy-hrebsd) holds uncommitted Stage E work of Johan's other session; merge there once it is committed and pushed
- [x] `feat-spherical-indexing-hrosm`: new branch off `feat-spherical-indexing-nlpar` (e49b3d85, untouched), clean replay of M with `pick.ps1`/`gate.ps1` as two commits ("Add high angular resolution orientation similarity maps (HROSM)", "Add HROSM tutorial"; `Staged-from:` trailers), equivalence gate and the clean-replay grep clean, worktree suite == baseline + HROSM tests; pushed, no PR; `feat-spherical-indexing` stays at 6723aaf0 -- f6461dba + f1b42148 (Staged-from #20, 90453bc3), pushed; equivalence 17164/17164 lines; worktree full suite 4941 passed / 1289 skipped (= baseline 4515 / 1251 + HROSM); HROSM selection 440 passed; nbval 23/23; feat-spherical-indexing 6723aaf0 and feat-spherical-indexing-nlpar e49b3d85 unchanged
