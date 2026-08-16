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
- [ ] `_grid.py`: square<->sphere maps, `legendre_normals`, ring tables (`readRing` port), `ring_solid_angles`, `lambert_solid_angles`, `quadrature_weights` (`computeWeightsSkip` port)
- [ ] `_fft.py`: `fast_size` (verbatim `fastSize` port), `fast_bandwidths` (private until Phase 6)
- [ ] `_sht.py`: `SphericalHarmonicTransform` with dual-path `analyze`/`synthesize`
- [ ] Tests: single-harmonic analyze/synthesize oracles vs `sph_harm_y`; signed Condon-Shortley confirmation; EMSphInx `square_sht.cpp` round trip (Lambert at 1e-11 scale-free); weights (`sum(w_hat)=1`, Legendre == Gauss-Legendre with halved equator, Lambert guard at dim 401); grid/ring/solid-angle invariants; Ni master m-3m structural zeros (real data); `fast_size` invariants + 13-smooth minimality

## Phase 2 -- `sht-master-spectra-and-file`
- [ ] `_master_pattern_harmonics.py`: public `kp.indexing.MasterPatternHarmonics` (`from_master_pattern`, `from_file`, `save` = `.sht` writer, `resize`, `remove_dc`, `power_spectrum`, `describe`) -- `MasterSpectra` port; `toLegendre` DCT regrid; weighted normalisation with compat quirk; `accum_e` energy weights; symmetry LUTs (38 groups, validated against `orix.quaternion.symmetry._groups`, names confirmed on orix 0.12.1); bandwidth-vs-resolution warning
- [ ] `_sht_file.py` (BSD-3 SHTfile codec; generic modality/simMetaSize; NotImplementedError paths)
- [ ] io plugin `emsphinx_master_pattern`; `EBSDMasterPattern.get_spherical_harmonics`
- [ ] Data: `ni_20kv_bw384.sht` (mp2sht.exe, sig 70), `ni_small_20kv_bw384.sht`, synthetic per-(zRot, cmpFlg) fixtures
- [ ] Tests: header parse, read->write field/CRC equality, pack/unpack all branches, EMSphInx binaries accept our `.sht` (local-gated), mp2sht parity, `kp.load(".sht")`, bandwidth warning

## Phase 3 -- `sht-wigner-d`
- [ ] `_wigner.py` (d(pi/2) table, dTable(beta), dTablePre, scalar `wigner_d`, `rotate_harmonics`, derivative helpers)
- [ ] `_euler.py` (`zyz_to_quaternion` etc.; explicit port of `test/xtal/rotations.cpp:288-318`; Bunge equivalence test to 1e-14)
- [ ] Tests: Mathematica tables from `test/sht/wigner.cpp`, table vs scalar, rotate composition/identity

## Phase 4 -- `spherical-cross-correlation`
- [ ] `_xcorr.py`: `SphericalCrossCorrelator`, `NormalizedSphericalCrossCorrelator` (Huhle `rDen`), spectrum kernel, 27-neighbourhood + glide, tri-quadratic interpolation, index<->euler
- [ ] Tests: `sht_xcorr.cpp` ports (random pairs, symmetric groups, wedge mask), Ni master autocorrelation -> identity + 24 cubic ops, timing baseline at bw 53/68/88

## Phase 5 -- `spherical-back-projection`
- [ ] `_back_projection.py` (`SphericalBackProjector` gather LUT, DCT rescaler, DCT IQ, window mask, `mlm`, `flm2`/`rDen`, single-PC guard)
- [ ] `_preprocessing.py` (Gaussian background, mosaic AHE, EMSphInx order, all behind keywords)
- [ ] Tests: `dctn` convention first, LUT vs `_get_direction_cosines_from_detector`, `Y_l^m` recovery, `signal_mask` changes `rDen`, `nickel_ebsd_small` window/IQ, forward-projection convention lock (27 rotations, flip/asymmetry check), binning != 1 PC test, `azimuthal`/`twist` guard, per-point-PC guard
- [ ] Measure the mean-PC error floor (`refine_orientation` with `pc_average` vs stored per-point-PC `xmap` on `nickel_ebsd_small/large`), record in `validation.md`; Phase 6/7 tolerances derive from it

## Phase 6 -- `spherical-indexing-ebsd`
- [ ] `_indexer.py` (`SphericalIndexer`, per-pattern failure handling), `EBSD.spherical_indexing` (dask `map_blocks`, info message, masks, multi-phase, `n_best`), benchmark
- [ ] Public `kp.indexing.fast_bandwidths()` exported in `indexing/__init__.pyi` (ShtWisdom stand-in)
- [ ] Tests: `nickel_ebsd_small` coarse vs stored xmap (median < 1.5 deg, >= 8/9 < 3 deg), lazy/verbose/mask/error paths, hard floor >= 2 pat/s/core, memory measured

## Phase 7 -- `spherical-refinement`
- [ ] Analytic derivatives + Newton (`_derivatives`, `_refine_peak`, Cholesky 3x3, saddle rejection, degeneracy fallbacks), normalised refine, `refine=True` default, `EBSD.refine_orientation_spherical`
- [ ] Tests: `sht_xcorr.cpp` refine ports, saddle/beta=0, `nickel_ebsd_small` refined all < 1 deg + score increase, weekly `nickel_ebsd_large`

## Phase 8 -- `spherical-pseudo-symmetry`
- [ ] `_pseudo_symmetry.py` (`find_pseudo_symmetry_operators`, MasterXcorr port incl. two-phase mode, optional volume + stereogram), `pseudo_symmetry_ops` in indexer, psymfile read/write
- [ ] Tests: Ni autocorrelation peaks subset of Oh, `exclude_symmetry` empty, synthetic 3-fold/6-fold, wrong op -> index 0, local hcp masters skip-if-absent

## Phase 9 -- `sht-visualisation-and-interop`
- [ ] sht2png equivalents (`to_master_pattern`, `plot_power_spectrum`, `describe`), `SphericalBackProjector.plot`, xcorr volume plot
- [ ] `write_emsphinx_patterns` (PatternRepack contract in tech-stack.md), EBSPDims probe in `oxford_binary`, `EMSphInxNamelist` read/write (port of `test/util/nml.cpp` round trip), Sphinx-Gallery example
- [ ] Tests: describe values, stereographic r > 0.98, IndexEBSD.exe reads a repacked file (local-gated), namelist round trip; out-of-scope list confirmed in mission.md

## Phase 10 -- `spherical-indexing-emsphinx-regression`
- [ ] `create_emsphinx_reference.py` (pinned to 60f3517), `.npz` refs in-package, bidirectional tests (their binaries on our files; ours vs their `.ang/.h5`), `nregions in {0, 4, 10}`, two `delta` values

## Phase 11 -- `spherical-indexing-tutorial`
- [ ] `doc/tutorials/spherical_indexing.ipynb` (uses the full Ni master `ebsd_master_pattern("ni")` or caps bw <= 190 for the 401-px master; wall-clock budget derived from the Phase 6 benchmark, target <= 3 min on 8 threads with the pre-agreed fallback bw 53 / `s.inav[:40, :30]` / `refine=False` for sweep cells; outputs stored), `doc/tutorials/index.rst` entry, `NOTEBOOKS` entry in `doc/tutorials/run_nbval.sh`, `tutorials_sanitize.cfg` for timings, `.sht` cell in `load_save_data.ipynb`, CHANGELOG consolidation, `sphinx-build -b html` and `-b linkcheck`
