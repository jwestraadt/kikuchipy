# Phase 6 -- `spherical-indexing-ebsd`: plan

Branch `spherical-indexing-ebsd` off `develop` (after the Phase 5 merge,
jwestraadt/kikuchipy#7). Models: this plan and the spec on Fable 5 (xhigh,
ultracode); tests, implementation, adversarial review and fixes by Opus 5
agents (xhigh, ultracode). Autonomous mode (approval gate waived for
spherical-indexing phases; decisions flagged in section 6). Every task group
is independently reviewable; the order below is the implementation order.
Tests are written (failing) before the code they exercise. No new Numba
kernel: no `-n 0` cache-warm subtlety beyond the inherited one (run the new
test module once with `-n 0` before `-n 4` anyway, since it JIT-warms the
merged kernels). The drafting probe -- an end-to-end pipeline on the merged
Phase 1-5 modules plus a dask `map_blocks` prototype
(`p6_probe.py`/`p6_probe2.py`/`p6_probe3.py`/`p6_probe4.py`, scratchpad, not
committed; recipe in `validation.md` "Recorded results") -- produced every
number in `requirements.md`/`validation.md`; the 2026-09-02 adversarial
spec review re-ran the pipeline independently (every headline number
reproduced digit for digit) and its accepted findings were folded in with
fresh measurements (`p6_rev_probe.py`/`p6_rev_probe2.py`, second dated
block in `validation.md`). No compiled C++ driver:
`indexer.hpp` is glue over components validated in Phases 1-5, and the
stored kikuchipy `xmap`s are the accuracy oracle (`IndexEBSD.exe` is
Phase 10).

## 0. Constitution amendments (applied 2026-09-02 in the spec commit; `mission.md` and the deliverables table are untouched by this phase)

1. `specs/roadmap.md`, Phase 6 boxes: rewrite the first box to
   "`_indexer.py` (`SphericalIndexer`: per-phase
   `NormalizedSphericalCrossCorrelator` (plain when `normalize=False`) on a
   shared projector/Wigner table, `IndexEBSD` namelist defaults incl. the
   `[16, 512]` bandwidth rule, the harmonics-vs-detector `sample_tilt`
   binding guard (a 70-vs-65 mismatch indexes 4.7 deg wrong at *higher*
   scores -- measured), per-pattern
   failure handling extended to `ptp == 0` degeneracies (the AHE of a
   constant is `255 + O(1e-13)`, so EMSphInx would correlate rounding noise
   -- measured score 0.23, the one genuine deviation; a constant float
   correlates the window mask at -2.64, which EMSphInx's positive-score
   insertion rule then drops itself -- parity, failed earlier), the C++
   zero-seeded insertion rule (`score <= 0` never recorded),
   `BatchEstimate` chunk sizing with a recorded `max(1, .)` clamp (34/15/6
   at `bw` 53/68/88)), `EBSD.spherical_indexing` (dask threaded
   `map_blocks` with truthful `chunks=`,
   per-chunk `clone()` (0.2 ms at `bw` 68), bitwise deterministic across
   chunking/threads, info message printing the memory model (49 MB at `bw`
   68), masks incl. a new boolean-dtype navigation-mask check, multi-phase
   `PhaseList` (orix drops losing phases from `xmap.phases` -- pinned via
   `nbest_phase_id`), `n_best` with the
   one-candidate-per-phase fill semantics), benchmark"; the test box gains
   "measured coarse vs stored xmap at `bw` 68: small median 0.599 / max
   0.838 deg (assert median < 1.5, >= 8/9 < 3, all < 2.0), large 20-pt
   0.499 / 1.350 (median < 1.5, max < 3.0), weekly 165-pt 0.530 / p95 1.082
   / max 1.495 (median < 1.5, p95 < 2.5, max < 3.5); scores/IQ
   measured-then-pinned (0.4963-0.6239 / 0.1727-0.2036 at `bw` 68; also
   pinned: signal_mask 0.4461-0.5762, circular_mask 0.4915-0.6390 with
   `n_points` 1117, gaussian_background 0.4942-0.6101);
   multi-phase discrimination via a sign-scrambled copy
   (`default_rng(42).choice([-1, 1])` -- same power spectrum, real
   function; 9/9, gaps 0.2970-0.4151; a rotated copy is
   degenerate -- the peak is rotation invariant, measured gaps +-0.015; kept
   as a control with the composed-orientation identity
   `rotation_from_zyz(zyz_b) * O_B == O_A`, 0.68/1.07 deg); hard floor
   >= 2 pat/s/core passed at 77.6 (bw 68); per-worker memory measured at
   `bw` 53/63/68/88/113 (peak 21/36/45/98/207 MB, model 23/39/49/108/228)".
2. `specs/tech-stack.md`, Numerics, result-contract bullet: append "a
   zero-variance pattern (raw `ptp == 0`, or a pipeline that degenerates to
   a constant, `ptp(processed) == 0`) is marked failed rather than
   correlated -- for constant uint8 input a recorded Phase 6 deviation:
   EMSphInx correlates the
   AHE rounding ripple (measured iq ~1.0, score +0.23,
   garbage orientation) because the mosaic AHE of a uniform image returns
   `255 + O(1e-13)`, never exactly constant; a constant *float* via
   `n_regions=0` correlates the window mask at a measured -2.64 which
   EMSphInx's own zero-seeded insertion rule then drops (`Result` rows
   start at `corr 0`/`phase -1` and `upper_bound` never inserts
   `corr <= 0`, `indexer.hpp:217-239`) -- so that case is parity, failed
   earlier; the insertion rule itself is ported (a candidate with a
   non-positive score is never recorded); a non-finite winning score and
   any per-pattern exception fail the same way (`ebsdWorkItem` catch
   semantics)".
3. `specs/tech-stack.md`, Numerics, quirks bullet: append "the AHE
   four-term interpolation of a uniform image leaves an `O(1e-13)` ripple
   (Phase 5 finding), so a constant uint8 pattern never reaches
   `unproject`'s `ptp == 0` mask branch through the default pipeline
   (Phase 6 consequence, intercepted by the indexer's raw/processed
   `ptp` guards)".
4. `specs/tech-stack.md`, Numerics, misorientation-tolerance bullet: append
   "Phase 6 measured end-to-end coarse at `bw` 68 (`pc_average`,
   backgrounds removed, `n_regions=10`): `nickel_ebsd_small` median 0.599 /
   max 0.838 deg, `nickel_ebsd_large` 20-pt 0.499 / 1.350, 165-pt weekly
   0.530 / p95 1.082 / max 1.495 -- the quadrature sum of the mean-PC floor
   and the interpolated grid error predicts 0.47/0.9, confirmed; a rotated
   copy of the same master is not a usable second phase in tests (the
   cross-correlation peak is rotation invariant; measured gaps +-0.015),
   and a *phase*-scrambled copy is not a real spherical function (the
   `m = 0` row must stay real) -- use a **sign**-scrambled copy
   (`alm * default_rng(seed).choice([-1.0, 1.0], ...)`: every `|a_lm|`
   preserved, gaps 0.30-0.42); a harmonics/detector `sample_tilt`
   mismatch indexes ~5 deg wrong at *higher* scores (70 vs 65 measured:
   4.68 deg median, scores 0.51-0.65 vs 0.50-0.62), so the indexer must
   bind them (EMSphInx builds the geometry from the master,
   `idx.hpp:218`)".
5. `specs/tech-stack.md`, Performance bullet: append "Phase 6 baseline
   (60x60, single core, warm): preprocess 0.2 ms + unproject 0.2 ms +
   analyze 0.2/0.5/1.0 ms + correlate 5.7/11.9/29.8 ms at `bw` 53/68/88 =
   160/78/32 pat/s/core -- the >= 2 floor passes 39x at `bw` 68; threaded
   `map_blocks` scaling 2.2x at 4 workers / 2.4x at 8 (recorded, partial
   GIL residency); per-worker chunk kit (correlator clone + out pair)
   resident 9.4/15.9/20.0/43.4/91.9 MB, peak 21.4/35.7/44.9/97.6/207.2 MB
   at `bw` 53/63/68/88/113 (the constitution's {63, 68, 88, 113} set
   measured, 53 extra), model
   `(n_phases if normalize else 1) slP^2 bwP 24 + slP^3 8` bytes exposed
   as `SphericalIndexer.memory_per_worker_bytes` (the constitution's
   warning helper; `index_patterns` warns above 2 GiB x workers -- at
   `bw` 113 x 8 the model gives 1.83 GB = 1.70 GiB, near, not over);
   correlator `clone()` 0.18/0.21/0.33 ms at `bw` 53/68/88 (per-chunk
   cloning < 2 % even at chunksize 1); `BatchEstimate`
   ported for the default chunksize with a recorded `max(1, .)` clamp
   (34/15/6 at `bw` 53/68/88;
   the `nt^2` rule gives chunksize 1 for 9 patterns at 4 workers)".
6. `specs/_research/explore-emsphinx-core-algorithm.md`: addenda -- 6.2/6.3:
   `quNp` is identity as shipped, so the final conversion collapses to
   `rotation_from_zyz` (Phase 6); the per-phase candidate count is one
   (`n_best > n_phases` rows keep the invalid fill; pseudo-symmetry adds
   candidates in Phase 8); the zero-seeded `upper_bound` insertion never
   records a candidate with `corr <= 0` (`indexer.hpp:217-239` -- "only
   keep something with a positive phase"); the geometry `sample_tilt`
   comes *from* the master (`idx.hpp:218`), so a port taking it from the
   detector must add a binding guard (mismatch measured: 4.68 deg wrong
   at higher scores); the exception catch writes phase -1 / identity /
   metric 0 / iq 0 and Phase 6 extends it to `ptp == 0` degeneracies with
   the measured garbage-score rationale (deviation for the AHE-ripple
   case only); 3.9/3.10: the peak value is
   invariant under a rotation of the reference -- a rotated master copy is
   degenerate as a second phase (measured), and a phase-scrambled copy is
   not a real spherical function (sign-scramble instead); section 8: new
   items -- the AHE-ripple consequence, the rotated-copy degeneracy, the
   `resize`-vs-direct normalisation difference (2.9 % max-abs / 10.3 %
   rel-L2 / up to ~100 % per significant coefficient -- parity runs must
   resize from the stored bandwidth as `IndexEBSD` does).
7. `specs/roadmap.md` Phase 7 box: no bound changes (the refined tolerances
   stand); note that `refine=False` stays the Phase 6 default and Phase 7
   flips it to `True` (`IndexEBSD`'s default) when refinement exists.
8. Revision note (2026-09-02, adversarial spec review): **no further
   amendments are needed** for two findings that alleged constitution
   drift -- the tracemalloc bandwidth set of tech-stack.md ("{63, 68, 88,
   113}") is now fully measured (D8 table; 53 is an extra row), and the
   test layout of tech-stack.md (`tests/test_indexing/test_spherical_*.py`
   plus the signal file) is now followed by splitting the module tests
   into `tests/test_indexing/test_spherical_indexer.py` (section 3)
   instead of amending the layout.

## 1. `_indexer.py` -- tests in `tests/test_indexing/test_spherical_indexer.py`

1. Module header: kikuchipy GPL header + the delimited EMSphInx notice
   (`idx/indexer.hpp`, `modality/ebsd/idx.hpp`, `idx/base.hpp` with the
   line ranges and not-ported lists of `requirements.md` D10; "changed by
   Johan Westraadt, 2026-09"). Module docstring: the pipeline order, the
   result contract and failure semantics with the two measured garbage
   scores, the one-candidate-per-phase rule, the thread strategy
   (per-chunk clone, shared projector), the determinism statement, the
   `BatchEstimate` model, the memory model with the measured table, the
   `resize`-vs-direct note.
2. `_batch_estimate(bandwidth, n_workers, n_patterns) -> int`: verbatim
   `BatchEstimate` port (`:189-205`) incl. the golden-ratio constant and
   the `nt^2` load-balancing rule; module-level so the pins (34/15/6;
   9-pattern -> 1) hit it directly.
3. `_index_chunk(patterns_block, indexer, n_best) -> (nc, n_best, 6)
   float64`: clone correlators once per invocation (each phase's normalized
   correlator, or the prototype when `normalize=False`), allocate the
   north/south pair **with `np.zeros`** (D2: zeroed before first use;
   per-pattern re-zeroing is defensive only -- every window point is
   assigned each call), loop patterns: fill-initialise the rows, the raw
   `ptp == 0` guard, `try:` preprocess -> processed `ptp == 0` guard
   (reachable only via a degenerate pipeline, D2) ->
   `unproject(out=..., return_image_quality=True)` ->
   `analyze` -> per-phase correlate + descending `upper_bound` insertion
   that **drops candidates with `score <= 0`** (the C++ zero-seeded fill
   comparison, D3) -> non-finite check `except Exception:` restore fill
   values (never re-raise). Pack `alpha, beta, gamma, score, phase_id,
   iq`. Module-level `_map_chunks(patterns_da, indexer, n_best) ->
   dask.array` wraps the `map_blocks` call with explicit
   `chunks=(patterns_da.chunks[0], n_best, 6)` (D4) so the
   metadata-truthfulness test can hit it before any `compute()`.
4. `class SphericalIndexer` (D1): constructor validation in the frozen
   order (refine first with the phase-number-free message, the
   `[16, 512]` bandwidth rule, harmonics normalisation, shared-geometry
   check, the harmonics-vs-detector `sample_tilt` binding guard,
   `n_regions` rule, then the projector with its Phase 5 guards
   propagating), per-phase `resize(bandwidth)` with the upsize
   `UserWarning`, the shared Wigner table, the correlators, the stored
   configuration and `good_pixels` derivation (`~signal_mask &
   _circular_mask` reduced to present terms); `index_patterns` (D2/D4:
   input checks incl. `n_best < 1` and `chunksize < 1`, chunking --
   `da.from_array`/`rechunk` to
   `(chunksize, -1, -1)`, `_map_chunks` (task 1.3),
   `ProgressBar` when asked, one eager `compute()`, unpack + `phase_id ->
   int32`, the 2 GiB memory `UserWarning`); `get_info_message(n_patterns,
   chunksize)` (D6 template, printing the model -- "49 MB" at `bw` 68);
   `memory_per_worker_bytes` property (D8
   model with the `(n_phases if normalize else 1)` factor); `__repr__`
   (D1, `"?"` for a phase-less harmonics). Docstring `Notes`: the
   un-normalised-scores
   statement, the failure semantics table (with the corrected deviation
   bookkeeping and the positive-score insertion rule), EMSphInx defaults
   equivalence, thread-safety contract (immutable; `index_patterns`
   reentrant), the `Examples` doctest; no roadmap phase numbers anywhere
   in the docstrings (decision 6.14).
5. `src/kikuchipy/indexing/_spherical/__init__.py`: add ``_indexer`` to the
   `Submodules` block ("the spherical indexer: per-pattern pipeline,
   multi-phase top-n bookkeeping, dask chunking"), importing nothing.

## 2. `EBSD.spherical_indexing`, exports, CHANGELOG, benchmark

1. `signals/ebsd.py`: import `SphericalIndexer` (and
   `MasterPatternHarmonics` for the type hint) in the existing
   `kikuchipy.indexing` import block; add `spherical_indexing` directly
   after `dictionary_indexing` with the D5 signature, checks (signal shape
   vs detector, the three `navigation_mask` checks with DI's messages
   **plus the new frozen boolean-dtype check** -- DI has none, D5,
   decision 6.16 -- `.phase` set + unique names), mask compression
   `keep = ~mask.ravel()`,
   the info/speed prints under `verbose >= 1`, the D5 `CrystalMap`
   construction (`create_coordinate_arrays`, `PhaseList` -- documenting
   that orix drops losing phases from `xmap.phases`, D5 -- `(n, n_best)`
   rotations with the `n_best == 1` squeeze, `scores`/`iq`/conditional
   `nbest_phase_id` props, `is_in_data` expansion with deterministic
   fills, `scan_unit`), and the D5 docstring (`:cite:lenthe2019spherical`,
   un-normalised scores note, memory table, See Also, single-PC note,
   the phase-number-free refinement note, the both-keywords
   `emsphinx_compatible` parity note).
2. `indexing/__init__.pyi`: the three imports + sorted `__all__` additions
   (`SphericalBackProjector`, `SphericalIndexer`, `fast_bandwidths`).
3. Docstring publication pass (D9, revision -- the old task 2.3 "drop the
   'private until Phase 6' caveat" was a no-op, the caveat lives in
   `specs/roadmap.md:30` not the docstring): `_fft.fast_bandwidths` gains
   an `Examples` section and `See Also` ->
   `kikuchipy.indexing.SphericalIndexer`, and its `:func:`fast_size``
   cross-reference becomes prose (`fast_size` stays private);
   `_back_projection.py` public docstrings scrubbed of private
   cross-reference targets (`:1001/:1015/:1065/:1341`) and roadmap phase
   numbers (`:1086/:1409`); verify with the automated
   `sphinx-build`/`linkcheck` commands (`validation.md`).
4. `CHANGELOG.rst` `Unreleased -> Added`: the two D9 entries verbatim
   with the pinned `#8` fork-PR link.
5. `benchmarks/indexing/test_spherical_indexing.py`: the D8 benchmark
   (kikuchipy header; harmonics/detector built outside the benchmarked
   callable, `verbose=0`, the scores-mean and floor assertions; the
   docstring states the floor is map-level end-to-end incl. per-call
   construction, ~1/3 of the 9-pattern wall time -- D8).

## 3. Tests -- exact assertions in `validation.md`

1. Two files, the constitution's layout (revision fix -- the first draft
   put everything in the signal file, breaking `tests/test_indexing/
   test_spherical_*.py` and the one-file-per-module pattern of Phases
   1-5). Shared fixtures (class-scoped setup like
   `test_ebsd_hough_indexing.py`, duplicated or conftest-hoisted as the
   implementation prefers): the backgrounds-removed `nickel_ebsd_small`
   signal, `det = s.detector.deepcopy(); det.pc = det.pc_average`, `mph =
   MasterPatternHarmonics.from_master_pattern(nickel_ebsd_master_pattern_
   small(projection="lambert", hemisphere="both"), bandwidth=68)` cached at
   module scope, `ori_ref = Orientation(s.xmap.rotations.data, Oh)`.
   **`tests/test_indexing/test_spherical_indexer.py`** (the `_indexer.py`
   module): `TestSphericalIndexerConstruction` (defaults via
   `inspect.signature`; refine -> `NotImplementedError` `"refine=True"` /
   `"not implemented"` with the monkeypatched-projector order sentinel;
   the `[16, 512]` bandwidth guard; the `sample_tilt` binding guard
   (70 vs 65 -> `ValueError`, `None` skips) with the 4.68-deg
   `record_property` rationale; guards
   propagated -- multi-PC `pc_average` message through the indexer,
   `azimuthal`/`twist`, empty harmonics, wrong type, `n_regions` bounds,
   shared-geometry `ValueError`, upsize `UserWarning`, truncation with no
   `UserWarning` (catch_warnings filtered, D1); attributes, repr incl.
   the phase-less `"?"`, the shared Wigner table `is`-identity across
   correlators, immutability
   -- mutating the caller's detector afterwards changes nothing),
   `TestBatchEstimate` (34/15/6 pins; the `nt^2` rule; the
   `(68, 4, 0) == 1` clamp-deviation pin),
   `TestIndexPatterns` (direct `index_patterns`: `n_best=0`/
   `chunksize=0` -> `ValueError`; the negative-score monkeypatch test --
   rows keep the fill; the guard-(b) monkeypatch test; the non-finite
   monkeypatch test; `_map_chunks` metadata truthfulness --
   `res.shape == (n, n_best, 6)` before `compute()`),
   `TestMemoryModel` (`memory_per_worker_bytes` arithmetic at `bw` 68 =
   49,426,200; two-phase `normalize=False` == single-phase;
   `normalize=True` two-phase larger by exactly `slP^2 bwP 24`; the
   2 GiB `UserWarning` under `dask.config.set(num_workers=64)`),
   `TestExports` (`kp.indexing.SphericalIndexer` etc. resolve via
   the lazy loader; `fast_bandwidths(16, 128)` pin; `__all__` sorted;
   the `fast_bandwidths` docstring assertions and the no-phase-numbers
   docstring grep, D9), and the kernel-flag regression (no
   `CPUDispatcher`, no `scipy.fft` import in `_indexer.py`).
   **`tests/test_signals/test_ebsd_spherical_indexing.py`** (the signal
   method):
   `TestNickelSmall` (D7 assertions at `bw` 68 for `normalize` True/False;
   scores/IQ pins; `bw` 53 recorded row; `CrystalMap` structure: shape,
   coordinates, `scan_unit == "um"`, prop names/shapes/dtypes,
   `phases.names == ["ni"]`, `is_indexed.all()`),
   `TestPreprocessingPaths` (revision addition: `circular_mask=True` --
   `n_points == 1117`, D7 bounds, the pinned scores/IQ;
   `gaussian_background=True` at both `emsphinx_compatible` settings --
   D7 bounds, pinned scores, the `> 1e-4` non-identity, the kwarg spy;
   `n_regions=0` -- the recorded row and IQ band),
   `TestNBest` (single phase `n_best=3`: row 0 real, rows 1-2 fill with
   `nbest_phase_id[:, 1:] == -1`, scores descending, `rotations.shape ==
   (9, 3)`; `n_best=1` squeeze; `n_best=0` -> `ValueError`),
   `TestMultiPhase` (the sign-scrambled discrimination -- 9/9 phase 0,
   min gap > 0.1, `xmap.phases.names == ["ni"]` with the full list pinned
   via `indexer.phases`/`nbest_phase_id`; the rotated-copy degeneracy
   control -- max |gap| < 0.05
   recorded, the composed-orientation identity < 2.5 deg with the three
   wrong compositions > 10 deg; two-phase `n_best=2` ordering; phase-less
   harmonics -> `ValueError` naming
   `.phase`), `TestFailureInjection` (the constant-37 pattern at one map
   point: `is_indexed` False there, phase -1, identity, score 0, iq 0, the
   other eight bitwise equal to the clean run; the all-zero pattern
   likewise; the corrected deviation `record_property` notes),
   `TestMasks` (navigation_mask polarity -- k `False` entries ->
   k indexed points, `is_in_data`, deterministic fills,
   `phases.names[0] == "not_indexed"`, all-`True` ->
   `ValueError`, wrong shape / not-ndarray / the new boolean-dtype
   message; signal_mask -- D7 bounds plus the pinned scores/IQ (the
   flipped-`good_pixels` mutant dies on IQ, the not-forwarded one on
   scores -- both measured), the projector receives it
   (`indexer.projector.signal_mask is not None`)),
   `TestLazyAndDeterminism` (LazyEBSD bitwise == EBSD; chunksize 1/4/9
   bitwise; `num_workers` 1 vs 4 bitwise; explicit chunksize honoured via
   the info message chunk count), `TestDtypes` (uint8 vs float32 vs
   float64 of identical values -> identical maps), `TestVerbose`
   (capsys: `verbose=0` silent, `verbose=1` contains the D6 substrings
   incl. `"Estimated memory per worker: 49 MB"` and
   the speed line), `TestPerformance` (the >= 2 pat/s/core floor as the
   single loose assertion; per-stage/pat timings, per-worker tracemalloc
   numbers at `bw` 63/68 (88/113 weekly) and the
   `memory_per_worker_bytes` model agreement (< 2x)
   `record_property`; the < 200 MB loose peak bound),
   `TestNickelLargeSubset` (default suite 20-pt: D7 bounds; weekly 165-pt:
   D7 bounds; both `record_property`, downloads skip without the `tests`
   extra).
2. Marks: `@pytest.mark.weekly` on the 165-point subset, the `bw` 88
   recorded row and the `bw` 88/113 memory rows; everything else default.
   Estimated default-suite cost:
   harmonics builds ~1 s, the small map indexed a handful of times at 13
   ms/pattern, the large 20-pt run ~3 s -- well under 30 s single process.

## 4. Adversarial review and fixes

1. Ultracode workflow (Opus 5): a fidelity reviewer refutes `_indexer.py`
   against `indexer.hpp`/`idx.hpp` line by line (fill init `:217-222`
   incl. the never-insert-`corr <= 0` consequence,
   insertion `upper_bound` + shift `:235-239`, the geometry-from-master
   tilt binding `idx.hpp:218` vs our guard, the conversion order
   `zyz2qu -> quNp mul -> conjugate` `:264-269` == `rotation_from_zyz`
   given identity `quNp`, `computeHarmonics` order `:312-318`, IQ from the
   processed pattern, `BatchEstimate` arithmetic incl. `int()` truncation
   vs `ceil`, the catch semantics `:427-437`, the correlator wiring
   `:262-291` incl. `flm2` from the *squared synthesis* and `mlm` shared);
   a conventions reviewer (headers, numpydoc, masks polarity stated
   everywhere, no new kernels/FFT sites, `.pyi` mechanics, CHANGELOG
   format); a test-quality reviewer runs the **bug-injection list** --
   every mutation must be killed by a named test:
   **zyz->rotation conjugation dropped** (`Rotation(zyz_to_quaternion(.))`
   instead of `rotation_from_zyz`; dies by TestNickelSmall, ~35 deg, and
   by the composed-orientation identity);
   **an extra conjugation added** (same deaths);
   **phase loop keeping the worse score** (`lower_bound`/ascending
   insertion; dies by TestMultiPhase ordering and the scrambled 9/9);
   **phase_id off by one** (dies by scrambled `phase_id == 0` and
   `nbest_phase_id`);
   **failure semantics wrong** (score kept, phase 0, or exception
   propagating; dies by TestFailureInjection);
   **`ptp` guards dropped** (the constant-37 point indexes with score
   ~0.23; dies by TestFailureInjection's `is_indexed False`);
   **navigation_mask polarity flipped** (dies by the k-points count and
   `is_in_data` test);
   **signal_mask not forwarded to the projector or to `good_pixels`**
   (dies by the projector-received check and the pinned masked-run
   scores, 8-11 % off);
   **`good_pixels` polarity flipped** (`good = signal_mask` -- measured:
   lands *inside* the masked score pins but 60 % off the IQ pins; dies by
   the pinned masked-run IQ -- a revision addition, since the old
   differs-from-unmasked assertion was satisfied by this mutant);
   **tilt binding dropped** (harmonics-vs-detector `sample_tilt` guard
   removed; dies by the construction `ValueError` test -- rationale: the
   mismatch indexes 4.7 deg wrong at higher scores, silently);
   **bandwidth range guard dropped** (`bandwidth=8` constructs; dies by
   the `[16, 512]` `ValueError` test);
   **non-positive scores inserted** (`np.argsort` top-n instead of the
   zero-seeded `upper_bound` drop; dies by the monkeypatched-correlator
   negative-score test: row must keep phase -1/identity/0);
   **guard (b) removed** (processed `ptp == 0` reaches the correlator;
   dies by the monkeypatched-`_preprocess_pattern` constant test);
   **navigation-mask dtype check dropped** (an int 0/1 mask indexes
   everything via bitwise NOT; dies by the boolean-dtype `ValueError`
   test);
   **memory model `normalize` factor dropped** (dies by the two-phase
   `normalize=False` equality test);
   **2 GiB warning threshold or emission removed** (dies by the
   `num_workers=64` `pytest.warns` test);
   **info message prints the measured peak instead of the model** (the
   frozen-template divergence this revision fixed; dies by the
   `"Estimated memory per worker: 49 MB"` substring);
   **`map_blocks` `chunks=` dropped** (declared shape lies; dies by the
   `_map_chunks` metadata test);
   **n_best ordering ascending / fill rows first** (dies by TestNBest
   descending + fill positions);
   **`out=` buffers allocated with `np.empty` / not zeroed before the
   first pattern** (off-window garbage reaches `sht.analyze`; dies by
   TestNickelSmall accuracy and the determinism tests -- a revision
   replacement: the old "not re-zeroed between patterns" mutant is
   behaviourally a no-op, since `unproject` assigns every window point on
   every call and never touches `south` -- measured chunk-level vs
   pattern-level zeroing bitwise equal -- so no test can or need kill
   it);
   **`out=` buffers shared across threads** (one pair on the indexer
   instead of per chunk; dies by the 4-worker bitwise determinism test);
   **clone() skipped** (shared correlator scratch across threads; dies by
   the same);
   **preprocessing order swapped or `n_regions` not forwarded** (dies by
   the IQ pins -- 0.17-0.20 vs 0.29-0.33 -- and a spy on
   `_preprocess_pattern` kwargs);
   **IQ taken from the raw pattern** (same IQ pins);
   **normalize ignored** (always normalized; dies by the un-normalized
   score pins 0.2799-0.3533);
   **`emsphinx_compatible` not forwarded to correlate or to
   `_preprocess_pattern`** (dies by a spy asserting both kwargs, and --
   revision addition, the preprocessing side now has a numeric witness --
   by the `gaussian_background=True` compat-True-vs-False non-identity
   assertion, measured max |score diff| 1.83e-3 > the 1e-4 bar; the
   correlate-side numeric effect stays sub-tolerance, Phase 4);
   **chunk boundary duplication/offset** (map_blocks mis-packing; dies by
   chunk 4-vs-9 bitwise);
   **`BatchEstimate` misport** (`ceil` for `int`, `nt` for `nt^2`; dies by
   the 34/15/6 and 9-pattern pins);
   **`resize` skipped** (a `bw` 120 harmonics fed raw; dies by the
   `squared_harmonics` shape `ValueError` surfaced as a construction test);
   **rows filled with NaN instead of the contract values** (dies by
   TestFailureInjection's exact zeros/identity);
   **iq column mis-packed per candidate** (dies by the `(n,)` iq shape and
   values equal across an `n_best=3` run).
2. Fix, re-run (`-n 0` then `-n 4`), `pre-commit run --files ...`,
   `--doctest-modules src/kikuchipy/indexing/_spherical`, coverage of
   `_indexer.py` and the new `ebsd.py` method `>= 95 %` (target 100 %;
   the coverage command now includes `--cov=kikuchipy.signals.ebsd`,
   revision fix), `sphinx-build -b html` + `-b linkcheck` exit 0
   (revision addition -- the three new public docstrings render under
   numpydoc validation).

## 5. Commit and PR

1. Signed commits (spec + amendments; failing tests; implementation;
   review fixes); tick the Phase 6 boxes in `specs/roadmap.md` with the
   measured numbers; push; PR into fork `develop` with the template,
   stating GPL-only licensing (EMSphInx-derived: BSD opt-out impossible)
   and the CHANGELOG entries.

## 6. Open questions -- decided 2026-09-02 (autonomous mode), flagged for review

1. **A rotated master copy is not the multi-phase discriminator** (D3):
   measured degenerate (6/9, gaps +-0.015) because the correlation peak is
   rotation invariant -- the task brief's "a second phase that is a rotated
   copy loses to the true one" is refuted by measurement. Decided (as
   revised 2026-09-02): a **sign-scrambled** copy
   (`alm * default_rng(42).choice([-1.0, 1.0], ...)`) is the
   discrimination phase (9/9, gaps 0.30-0.42) -- the spec's first choice,
   a *phase* scramble, was itself refuted in review (not a real spherical
   function: `m = 0` row imaginary, 0.99 round-trip error), while the
   sign scramble preserves every `|a_lm|` exactly and discriminates
   better; the rotated copy stays as a
   degeneracy control pinning the composed-orientation identity.
2. **Zero-variance patterns fail instead of correlating garbage** (D2),
   with the counterfactual corrected in revision: the *deliberate,
   recorded deviation* from EMSphInx is the constant-uint8 case only --
   the C++ would index the AHE ripple at score +0.23 with a garbage
   orientation (measured). The constant-float/`n_regions=0` case
   correlates the window mask at -2.64 (measured), but the C++'s own
   zero-seeded insertion rule drops any `corr <= 0`, so EMSphInx also
   reports that point not-indexed -- our guard fails it earlier on the
   same outcome (parity, not deviation; the first draft mis-stated
   "EMSphInx would return score -2.64").
   The constitution's result contract wants `is_indexed False` semantics
   and a garbage orientation with a plausible score is worse than a failed
   point. Alternative (faithful garbage) rejected; the C++ behaviour is
   recorded in the docstring and research addendum.
3. **`n_best` counts phases, not peaks** (D3): one candidate per phase,
   EMSphInx parity; extra rows keep the invalid fill. Alternative
   (top-n peaks of one correlation cube) rejected -- no C++ counterpart,
   new peak machinery, no consumer.
4. **The multi-PC convenience is refused, not silently averaged** (D1/D5):
   the projector's `pc_average`-naming `ValueError` propagates. Alternative
   (auto-average with a warning, the PyEBSDIndex style) rejected: the
   Phase 5 floor shows the mean-PC error is the dominant residual, so the
   user should opt in explicitly. One-line recipe in the error and docs.
5. **`chunksize=None` uses the `BatchEstimate` port, not
   `get_chunking`** (D4): the `nt^2` rule parallelises small maps (9
   patterns -> 9 chunks at 4 workers) where byte-based chunking would give
   one chunk; and it is the C++'s own model, one more ported function.
   `get_chunking`/`get_dask_array` remain for the lazy signal's own
   storage chunks, which are simply re-chunked.
6. **Per-chunk `clone()` rather than per-worker kits** (D4): stateless
   worker function, no thread-local registry; measured cost < 2 % of chunk
   runtime. Alternative (a pool of per-worker kits keyed on thread id)
   rejected as stateful complexity without measured need.
7. **`verbose: int = 1`** mirrors `hough_indexing` (not DI's always-print):
   `0` silences message, speed and progress bar -- needed by the benchmark
   and by scripted use.
8. **`refine=False` default in Phase 6** (D1): `IndexEBSD`'s default is
   `refine=true`, but ours must not raise by default; Phase 7 flips the
   default to `True` when refinement exists (roadmap note, amendment 0.7).
9. **The memory warning helper** (D8) is `memory_per_worker_bytes` + an
   `index_patterns` `UserWarning` above 2 GiB x workers: satisfies the
   constitution's "warning helper lives on `SphericalIndexer`" without
   psutil (total-RAM checks are impossible with the pinned dependency
   set).
10. **`nbest_phase_id` prop only when `n_best > 1`** (D5): a
    `CrystalMap` holds one `phase_id` per point; the extra prop
    self-describes multi-candidate rows (including the `-1` fill), and the
    full per-candidate table is always available from
    `SphericalIndexer.index_patterns`. Alternative (merge_crystal_maps of
    per-phase maps, the DI pattern) rejected: EMSphInx indexes multi-phase
    natively and the merged-map route discards the per-candidate scores'
    provenance.
11. **Harmonics upsize warns, downsize is silent** (D1): both are
    EMSphInx's `resize` semantics; the warning is our addition for the
    resolution-free padding case. The `resize`-vs-direct
    normalisation difference (2.9 % max-abs / 10.3 % rel-L2 / up to
    ~100 % per significant coefficient -- metrics stated in revision) is
    recorded so Phase 10 resizes from the
    stored `.sht` bandwidth exactly as `IndexEBSD` does.

Decisions 12-19 added 2026-09-02 in the spec-revision pass (autonomous
mode; each answers a review finding):

12. **`n_best` keeps its name, diverging from DI's `keep_n`** (D2/D5):
    the semantics genuinely differ -- DI keeps the n best of N dictionary
    matches per point, while here a row is one candidate *per phase*
    (saturating at `n_phases` until pseudo-symmetry lands), the EMSphInx
    `Result[n]` concept; `n_best` also already exists in kikuchipy
    (`orientation_similarity_map(n_best=...)`). Pinned by the
    `inspect.signature` test.
13. **The image-quality prop is `iq`, diverging from HI's `pq`** (D5):
    EMSphInx's own output name (`imQual`/IQ maps), and the metric is the
    Krieger Lassen DCT image quality -- the family of kikuchipy's
    `EBSD.get_image_quality` (correlation 0.62, Phase 5) -- whereas HI's
    `pq` is PyEBSDIndex's Hough pattern quality, a different quantity;
    naming it `pq` would suggest a comparability across methods that
    does not exist. Pinned by the `CrystalMap`-structure test.
14. **No roadmap phase numbers in public text** (D1/D5/D9): error
    messages and docstrings that render to kikuchipy users say
    "refinement ... is not implemented yet" etc. without "Phase 7" /
    "spherical-refinement" (internal spec/plan text keeps them); the
    newly public `SphericalBackProjector`/`fast_bandwidths` docstrings
    are scrubbed likewise (plan 2.3). Rationale: roadmap phases are
    project-internal bookkeeping, meaningless upstream.
15. **The harmonics-side `emsphinx_compatible` flag is not recorded on
    `MasterPatternHarmonics` in this phase** (D1): doing it properly
    (store, `save()`/`from_file` round trip, mismatch warning) touches
    the Phase 2 module and `.sht` metadata beyond this phase's scope;
    instead D1/D5 document that the master-normalisation quirk is chosen
    at `from_master_pattern` time and parity needs `True` in both
    places. Revisit in Phase 10's parity harness if mismatches bite.
16. **`navigation_mask` gains a boolean-dtype check that DI lacks**
    (D5): the flow computes `~navigation_mask`, and bitwise NOT of an
    int 0/1 mask is truthy everywhere (`~1 == -2`) -- silent
    index-everything; the new message is frozen and labelled as this
    phase's addition, not a DI mirror (DI's three checks are shape /
    all-True / is-ndarray only, re-read in revision).
17. **The `[16, 512]` bandwidth rule is a hard `ValueError`, not a
    warning** (D1): it is the same C++ `sanityCheck` the spec already
    ports `n_regions` from, `fast_bandwidths` already defaults to the
    same bounds, and no test of this phase needs `bw < 16`; a warning
    would leave the meaningless 12-deg-half-cell grid reachable.
18. **CHANGELOG PR number pinned as #8** (D9): the next fork PR after
    jwestraadt/kikuchipy#7; the entries are written verbatim in D9 so
    the PR gate is deterministic (updated in the PR commit if GitHub
    disagrees).
19. **The benchmark stays end-to-end** (`s.spherical_indexing`, D8):
    mirrors the DI benchmark and the user-visible call; the ~1/3
    construction share on the 9-pattern map is stated rather than
    excluded, and the pure `index_patterns` floor lives in the default
    suite. Alternative (benchmark `index_patterns` on a prebuilt
    indexer) rejected as measuring a call no signal user makes.
