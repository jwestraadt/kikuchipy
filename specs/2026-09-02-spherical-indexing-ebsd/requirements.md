# Phase 6 -- `spherical-indexing-ebsd`: requirements

## Scope

In scope:
- **`src/kikuchipy/indexing/_spherical/_indexer.py`** (private module, public class), a port of
  `EMSphInx/include/idx/indexer.hpp` @ 60f3517 (`Result` `:54-64`, `Indexer`
  construction `:163-181`, `BatchEstimate` `:189-205`, `indexImage` `:216-270`,
  `computeHarmonics` `:312-318`, `correlate` `:326-331`) and of the wiring and
  failure semantics of `include/modality/ebsd/idx.hpp` (`IndexingData::initialize`
  correlator/projector setup `:252-296`, `ebsdWorkItem` `:382-456`), the
  abstract roles of `include/idx/base.hpp` collapsed into concrete Phase 1-5
  classes: **`class SphericalIndexer`** (D1) holding one shared
  `SphericalBackProjector`, per-phase `NormalizedSphericalCrossCorrelator`s (or
  the spectra for the plain `SphericalCrossCorrelator` when `normalize=False`)
  and the preprocessing configuration, with `index_patterns(patterns, *,
  n_best=1, chunksize=None, progressbar=True) -> dict` (D2-D4),
  `get_info_message(...)` (D6), `memory_per_worker_bytes` (D8), `__repr__`;
  the module-level `_batch_estimate` (verbatim `BatchEstimate` port) and the
  packed per-chunk worker `_index_chunk` (D4). **No new Numba kernel and no
  new `scipy.fft` call site in this phase** (D10).
- **`EBSD.spherical_indexing(...)` in `src/kikuchipy/signals/ebsd.py`** (D5),
  the public signal method next to `dictionary_indexing`, returning a
  `CrystalMap` (masks, multi-phase `PhaseList`, `n_best`, `scan_unit`,
  info message, `:cite:lenthe2019spherical`).
- **Public exports** (D9): `kp.indexing.SphericalIndexer`,
  `kp.indexing.SphericalBackProjector` (Phase 5 D12 / roadmap amendment) and
  `kp.indexing.fast_bandwidths` (the `ShtWisdom` stand-in, mission table)
  through `indexing/__init__.pyi`; CHANGELOG entries (this phase **is**
  user-facing); numpydoc rendering of the three names.
- **Benchmark** `benchmarks/indexing/test_spherical_indexing.py`
  (pytest-benchmark, mirroring `test_dictionary_indexing.py`) (D8).
- **Tests** in two files, the constitution's layout (tech-stack.md "Tests"):
  `tests/test_indexing/test_spherical_indexer.py` (the `_indexer.py` module
  tests -- construction/guards, `_batch_estimate`, repr, memory model,
  exports, kernel-flag regression) and
  `tests/test_signals/test_ebsd_spherical_indexing.py` (the signal method --
  exact assertions in `validation.md`): `nickel_ebsd_small` coarse vs the
  stored `xmap` at `bw` 68 (and recorded at 53), `n_best`, masks (both
  polarities exercised), lazy signal, multi-phase (sign-scrambled
  discrimination and the rotated-copy degeneracy control), per-pattern
  failure injection, guards, verbose/info text, `CrystalMap` structure,
  dtype paths, chunking, determinism, throughput/memory `record_property`
  with the hard floor `>= 2` patterns/s/core, the weekly
  `nickel_ebsd_large` subset.

Out of scope: Newton refinement (`refine=True` raises `NotImplementedError`;
the message is user-facing and names **no roadmap phase numbers** --
internally this is Phase 7, `spherical-refinement`, which flips the default),
pseudo-symmetry candidates (`pSym` stays empty; the `indexImage` inner loop
`:243-261` is Phase 8 -- the insertion machinery it shares with the phase loop
*is* ported now), refine-only work items (`msk[i] & 0x02` alone, `idx.hpp:438-450`
-- Phase 7's `EBSD.refine_orientation_spherical`), `IndexEBSD.exe` regression
runs (Phase 10; **its parity runs use the default `emsphinx_compatible=True`**,
constitution), the EMSphInx ROI grammar (`roimask`; kikuchipy's
`navigation_mask` replaces it, mission out-of-scope list), namelist read/write
(Phase 9), IPF/XC/IQ map images and `.ang`/`.h5` writers (`idx.hpp:313-370`;
kikuchipy users get a `CrystalMap` and orix/kikuchipy plotting), the
`ThreadPool`/`ThreadedIqCalc` classes (dask's threaded scheduler replaces
them; the IQ is computed in the indexing pass, not in a second pool), the
tutorial notebook (Phase 11), a float32 fast path, per-point projection
centres (one PC per call, mission), non-zero `azimuthal`/`twist` (the Phase 5
projector raises), `EBSDDetector.get_indexer`-style construction from a
detector (the indexer takes the detector directly).

## Decisions

Every "measured" number below comes from the drafting probe -- an end-to-end
pipeline composed of the **merged** Phase 1-5 modules
(`MasterPatternHarmonics.from_master_pattern`, `SphericalBackProjector`,
`_preprocess_pattern`, `SphericalHarmonicTransform.analyze`,
`NormalizedSphericalCrossCorrelator`/`SphericalCrossCorrelator`,
`_euler.rotation_from_zyz`) plus a dask `map_blocks` prototype -- run on
`nickel_ebsd_small`/`nickel_ebsd_large` on this machine, recorded in
`validation.md` "Recorded results" and re-measured with the real
implementation. No compiled C++ driver: `indexer.hpp` is glue over components
already validated bitwise/near-bitwise in Phases 1-5, and the stored kikuchipy
`xmap`s are the accuracy oracle (the `IndexEBSD.exe` cross-check is Phase 10).

### D1 -- `SphericalIndexer`: signature, validation, attributes (frozen)

- ```
  SphericalIndexer(
      harmonics: MasterPatternHarmonics | Sequence[MasterPatternHarmonics],
      detector: EBSDDetector,
      *,
      bandwidth: int = 68,
      normalize: bool = True,
      refine: bool = False,
      signal_mask: np.ndarray | None = None,
      n_regions: int = 10,
      gaussian_background: bool = False,
      circular_mask: bool = False,
      emsphinx_compatible: bool = True,
  )
  ```
  Defaults are `IndexEBSD`'s namelist defaults (`nml.hpp:186-218`): `bw` 68,
  `normed` true, `nregions` 10, `gausbckg` false, `circmask` -1
  (`circular_mask=False`, Phase 5 D1) -- so a keywordless call is the
  `IndexEBSD` default configuration, which Phase 10's parity runs need.
- Validation, in order: **`refine=True` -> `NotImplementedError`** with the
  frozen, user-facing message `"Newton refinement (refine=True) is not
  implemented yet; only coarse indexing on the Euler grid is available.
  Leave refine=False."` (no roadmap phase numbers in public text -- a
  revision-review decision, plan 6.14; checked first, before any expensive
  construction, pinned by a monkeypatched-projector sentinel in
  `validation.md`; the merged correlators' `refine` kwarg raises its own
  private-message equivalent -- this guard fails fast at construction
  instead of at the first chunk); **`bandwidth` range**: `16 <= bandwidth
  <= 512` or `ValueError` with the C++ text "unreasonable bandwidth (should
  be [16, 512])" -- the same `nml.hpp` `sanityCheck` (`:635`) the
  `n_regions` rule below comes from (`:630`); the projector's own guard is
  only `bandwidth >= 1`, so without this rule `bandwidth=8` silently builds
  a 12-deg-half-cell grid (revision finding; `fast_bandwidths`' default
  minimum 16 encodes the same bound); `harmonics` normalised to a tuple (a
  single instance is wrapped;
  an empty sequence -> `ValueError`; a non-`MasterPatternHarmonics` entry ->
  `TypeError` naming the index and type); **shared-geometry check** (the C++
  "all master patterns must have the same tilt and kV", `idx.hpp:185`): any
  two phases whose `sample_tilt` (or `beam_energy`) are both set and differ by
  more than `rel 1e-6` -> `ValueError` quoting both values (a `None` on either
  side skips that comparison -- `.sht` files always carry both, hand-built
  harmonics may not; measured for the Ni master: `beam_energy 20.1`,
  `sample_tilt 70.0`); **harmonics-vs-detector tilt binding** (revision
  finding, major): in EMSphInx a tilt mismatch is structurally impossible
  because the geometry is built *from* the master
  (`idx.hpp:218` `geom.sampleTilt(phases.front().getSig())`) -- the port
  takes `sample_tilt` from the `EBSDDetector`, so the binding must be
  re-established as a guard: with `h_tilt` the first set `sample_tilt`
  among the phases (all set ones agree by the previous check; all-`None`
  skips, as above), `abs(h_tilt - detector.sample_tilt) > 1e-6 * max(1,
  abs(detector.sample_tilt))` -> `ValueError` quoting both values and
  naming `EBSDDetector.sample_tilt`. Measured
  rationale: harmonics at 70 deg indexed against `det.sample_tilt = 65`
  give median **4.680** / max **5.099** deg from the stored xmap (correct
  run 0.599 / 0.838) with *higher* normalized scores (0.5053-0.6474 vs
  0.4963-0.6239), so no score sanity check would ever catch it;
  **`n_regions`** validated by the namelist rule
  `0 <= n_regions <= min(detector.shape)` (`nml.hpp` `sanityCheck` `:630`;
  re-validated by `_preprocess_pattern`, but failing at construction beats
  failing in a dask worker); then the **projector**
  `SphericalBackProjector(detector, bandwidth, signal_mask=signal_mask,
  circular_mask=circular_mask)` is built, which owns every geometry guard of
  Phase 5 D1 (detector type, `bandwidth >= 1`, **single PC via
  `navigation_size != 1`** with the `pc_average`/`deepcopy` message,
  `azimuthal`/`twist`, `signal_mask` shape/dtype, the two empty-window
  guards) -- the indexer adds **no duplicate geometry checks** and lets the
  projector's errors propagate unchanged (one message to maintain; the tests
  assert the `pc_average` text through the *indexer* constructor).
- **Bandwidth alignment** (`idx.hpp:182`, `MasterSpectra::resize(nml.bw)`):
  every phase is stored as `harmonics[p].resize(bandwidth)` -- truncation when
  the source bandwidth is larger, zero padding when smaller, exactly EMSphInx.
  A phase with `bandwidth_source < bandwidth` additionally gets a
  `UserWarning` naming both bandwidths (zero padding buys a finer Euler grid
  but no new signal; EMSphInx pads silently -- the warning is our addition,
  consistent with the Phase 2 bandwidth-vs-resolution warning). Measured,
  with the metric stated (revision finding -- the first draft quoted only
  the mildest of the three): `resize(120 -> 68)` differs from
  `from_master_pattern(bw=68)` by `max|delta| / max|alm|` = **2.9 %**,
  relative L2 = **10.3 %**, and up to **~100 %** on individual significant
  coefficients (per-coefficient relative difference over the 187
  coefficients with `|a| > 1 %` of max; `a[0, 0]` itself moves 2.6 %,
  -3.15804 -> -3.07553 -- the weighted normalisation depends on the
  analysis bandwidth). The conclusion is thereby strengthened: the parity
  path of Phase 10 is *resize from the stored `.sht` bandwidth*, exactly
  `IndexEBSD`'s, and every score pin in `validation.md` states which
  construction produced it (direct at `bw`, this spec).
- **Preprocessing configuration** (`imprc.hpp` via Phase 5 D9;
  the `circmask` mapping of the research-doc addendum): the per-pattern call
  is `_preprocess_pattern(pattern, good_pixels=good, gaussian_background=
  gaussian_background, n_regions=n_regions, emsphinx_compatible=
  emsphinx_compatible)` with **`good = ~signal_mask & _circular_mask(shape)`
  reduced to the terms present** -- `None` when neither `signal_mask` nor
  `circular_mask` is given (`circmask = -1`: no histogram mask, `idx.hpp:254`
  via `setSize(w, h, -1, ...)`), the circular mask alone for
  `circular_mask=True` (`circmask = 0` couples the projector circle *and* the
  histogram mask, `idx.hpp:230` + `:254`), the inverted `signal_mask` alone or
  intersected when given (kikuchipy polarity `True = ignore`, inverted once at
  this boundary, Phase 5 D9). `emsphinx_compatible` is stored once and passed
  to both `_preprocess_pattern` (Gaussian-fit off-by-one -- read only when
  `gaussian_background=True`, so inert in the default configuration) and
  `correlate(..., emsphinx_compatible=...)` (the `x[2]` bounds and glide
  defects, Phase 4) -- one keyword, **two** C++ quirks in the default
  configuration (revision finding: the first draft claimed three). The
  third quirk -- the master's 2x-mean-over-`totW` normalisation -- is
  frozen into `alm` at harmonics-construction time by
  `from_master_pattern(..., emsphinx_compatible=True)`'s *own* keyword,
  which the indexer can neither see nor check (not recorded on the
  instance), and the *normalized* correlator is sensitive to that choice
  (constitution). Documented in the D5 docstring and the Phase 10 parity
  note: **both `from_master_pattern` and the indexer must use
  `emsphinx_compatible=True`** for parity. (Recording the flag on
  `MasterPatternHarmonics` would touch the Phase 2 module and is deferred
  to Phase 10's parity harness -- plan 6.15.)
- **Correlators**: one Wigner table `wigner_d_half_pi_table(bandwidth, True)`
  is built once and shared by every correlator via the `wigner_d_half_pi=`
  kwarg (2.5 MB at `bw` 68 paid once, Phase 4 D11). `normalize=True` (the
  default): per phase `NormalizedSphericalCrossCorrelator(bandwidth, alm_p,
  projector.squared_harmonics(alm_p), harmonics_p.n_fold,
  harmonics_p.has_equatorial_mirror, projector.window_harmonics,
  wigner_d_half_pi=table)` -- the `idx.hpp:263-283` wiring on Phase 5's
  `mlm`/`flm2` (the correlator owns `rDen`, Phase 5 D7). `normalize=False`:
  one prototype `SphericalCrossCorrelator(bandwidth, wigner_d_half_pi=table)`
  plus the per-phase `(alm_p, n_fold_p, mirror_p)` triples (the C++
  `UnNormalizedCorrelator` per phase holds only the spectrum `:284-291`; one
  shared scratch correlator per worker serves every phase, since `correlate`
  takes `flm` per call).
- Attributes (read-only after construction): `phases` (tuple of the resized
  `MasterPatternHarmonics`), `n_phases`, `bandwidth`, `normalize`,
  `projector` (the shared `SphericalBackProjector`; `detector` is
  `projector.detector`, already an isolated deepcopy), `signal_mask` /
  `circular_mask` / `n_regions` / `gaussian_background` /
  `emsphinx_compatible` (the stored configuration), `correlators` (tuple, one
  per phase, `normalize=True`) or `correlator` (the prototype) plus the
  spectra triples, `side_length` / `half_cell_degrees` (from the correlator:
  `slP` 135 and `180/135 = 1.333` deg at `bw` 68). `__repr__`:
  `"SphericalIndexer: 1 phase (ni), bw = 68, sphere window 1317 points
  (14.6 %), normalized"` -- phase names from `phase.name` with `"?"` for a
  phase-less harmonics.
- The indexer instance is **immutable after construction and holds no
  per-pattern scratch**: `index_patterns` clones the correlators per chunk
  (D4), so one instance may be reused across calls and signals of the same
  detector geometry (documented; the projector's immutability is Phase 5 D1).

### D2 -- The per-pattern pipeline and the result contract (frozen)

- Per pattern, in EMSphInx's order (`computeHarmonics` `:312-318` then the
  phase loop `:228-262`): (1) `_preprocess_pattern(raw, ...)` (D1
  configuration); (2) `projector.unproject(processed, out=(north, south),
  return_image_quality=True)` on the worker's `(dim, dim)` north/south pair
  -- **the caller-owned-buffer contract of Phase 5 D6**: `unproject` writes
  only the LUT points of `north` and never touches `south`, so both buffers
  MUST be **zeroed before first use** (`np.zeros`, never `np.empty` --
  off-window garbage would reach `sht.analyze`; bug injection 4.1).
  Per-pattern re-zeroing is **defensive only** (revision finding: every
  window point is *assigned* on every call -- `_back_projection.py:398-404`
  and the `ptp == 0` branch both write all of `sphere_index`, and `south`
  is never written -- measured: zeroing once per chunk vs once per pattern
  gives `np.array_equal` True on the stacked `gln` over the nine patterns,
  so a per-pattern-re-zero mutant is behaviourally a no-op and is NOT in
  the kill list); **`iq` is the DCT image quality of the *processed*
  pattern**
  (`:307-309`; measured on the nine backgrounds-removed Ni patterns:
  `0.1727-0.2036` with the default `n_regions=10`, vs `0.2890-0.3269` with
  `n_regions=0` and `0.766-0.779` for the raw patterns of Phase 5 -- the AHE
  flattens the histogram and shifts DCT power upward; these separate a
  raw-IQ mutant); (3) `gln = projector.sht.analyze(north, south)` (the
  `(bw, bw)` spectrum; `sht.analyze` reads only `self`, Phase 5 review, so the
  shared projector is thread-safe); (4) per phase `zyz, score =
  correlator_p.correlate(gln, emsphinx_compatible=...)` (or
  `correlator.correlate(alm_p, gln, n_fold_p, mirror_p, ...)` when
  `normalize=False`) and insertion into the descending top-`n_best` list --
  D3.
- **ZYZ -> rotation, the final conversion of `indexImage` `:264-269`**: the
  C++ does `zyz2qu` -> left-multiply by `quNp` -> conjugate
  (crystal->sample to sample->crystal). `quNp` is
  `BackProjector::northPoleQuat()` = **identity** in EMSphInx as shipped
  (`base.hpp:133`; `Geometry::northPoleQuat` is bypassed -- Phase 5 built the
  LUT in the sample frame precisely so that no detector-frame quaternion
  exists), so the whole chain collapses to
  **`_euler.rotation_from_zyz(zyz) = ~Rotation(zyz_to_quaternion(zyz))`** --
  the sign FROZEN by Phase 5 D8 (0.34/0.72 deg median/max for `~R` vs 35 deg
  for `R` on 27 forward-projected rotations). The stored kikuchipy `xmap`s
  use the same convention (they reproduce patterns through `get_patterns`),
  which the small-map test re-measures end to end: median **0.599** deg at
  `bw` 68 (D7) -- a dropped conjugation gives ~35 deg and dies loudly.
  Applied vectorised once per chunk result, not per pattern.
- **Result contract and failure semantics** (constitution bullet; `Result`
  init `:217-222` + the `ebsdWorkItem` catch `:427-437`): every output row
  starts as the fill value **`zyz = (0, 0, 0)` (-> identity rotation),
  `score = 0`, `phase_id = -1`, `iq = 0`**; a pattern is marked **failed**
  (all its rows keep the fill) when
  (a) **`np.ptp(raw) == 0`** (zero-variance input),
  (b) **`np.ptp(processed) == 0`** (the pipeline degenerated to a constant --
  `unproject` would take its mask branch and correlate the *window mask*),
  (c) the winning score or any winning `zyz` component is non-finite,
  (d) any exception escapes the per-pattern body (caught per pattern exactly
  as `ebsdWorkItem` does -- one bad pattern never kills the run), or
  (e) **no phase scored above 0** -- not a guard but a direct consequence
  of the ported insertion rule (D3, revision finding): the C++ seeds every
  row with `corr = 0`, `phase = -1` ("only keep something with a positive
  phase", `indexer.hpp:219`) and `std::upper_bound` under descending
  `operator<` (`:63`) never lets a candidate with `corr <= 0` displace a
  fill row, so a row keeps the fill unless a candidate beat it strictly.
  Reachability of (b) (revision finding): **no raw input reaches guard (b)**
  -- every constant input has raw `ptp == 0` and is caught by (a) first
  (measured: constant uint8 37 through the default pipeline gives
  `ptp(processed) = 8.5e-14`, so (b) does not even fire there; a constant
  float with `n_regions=0` has raw `ptp` exactly 0). Guard (b) is kept as
  the port of `unproject`'s mask-branch interception for a *pipeline* that
  degenerates a non-constant input (e.g. a pathological `good_pixels`
  set), is reached in tests only via a monkeypatched `_preprocess_pattern`
  returning a constant (named test, `validation.md`), and has its own
  bug-injection entry ("guard (b) removed") so the 100 % coverage target
  stays honest.
  Deviation bookkeeping, corrected in this revision: only the
  **constant-uint8 / AHE-ripple case is a genuine deviation** from
  EMSphInx -- the C++ would normalise the `O(1e-13)` ripple (the Phase 5
  four-term-sum finding) to unit variance and index it at `iq
  0.9999999999999996`, **score +0.2301**, a garbage orientation that
  silently enters the map; the port intercepts it via (a). The constant
  *float* input with `n_regions=0` reaches the mask branch and correlates
  the window mask at **score -2.6402**, but EMSphInx itself would then
  *drop* that candidate under rule (e) and report the point not-indexed --
  so for that case the port fails the pattern *earlier* on the same
  outcome, which is parity, not deviation. Both measurements are pinned as
  the rationale. NaN *pixels* are not guarded (measured: a single-NaN pattern
  survives `_to_uint8` and indexes near-normally, score 0.605 vs 0.624 clean;
  documented, not detected -- (c) catches a NaN that reaches the scores).
- `index_patterns(patterns, *, n_best=1, chunksize=None, progressbar=True)
  -> dict[str, np.ndarray]` with keys **`"zyz"` `(n, n_best, 3)` float64**
  (ZYZ Euler angles -- the raw grid quantity, so Phase 7 can refine from it
  and power users get the un-converted result), **`"scores"` `(n, n_best)`
  float64**, **`"phase_id"` `(n, n_best)` int32**, **`"iq"` `(n,)` float64**.
  `patterns` is a `(n, h, w)` NumPy or dask array (`ValueError` when
  `shape[1:] != detector.shape`, naming both); any real dtype (uint8 is
  EMSphInx's native path; floats go through `_to_uint8` inside
  `_preprocess_pattern` when `n_regions > 0` -- measured: float32 input with
  the same values gives a bitwise-identical processed pattern). `n_best < 1`
  -> `ValueError`; an explicit `chunksize < 1` -> `ValueError` (revision
  finding: nothing rejected it, and `da.from_array(..., chunks=(0, ...))`
  fails obscurely). Always computes eagerly (returns NumPy; the DI
  precedent -- a `compute=False` graph API can come later if a use case
  appears).

### D3 -- Multi-phase, `n_best`, and what a "candidate" is (frozen; one finding)

- The phase loop ports `indexImage` `:228-240`: each phase contributes
  **exactly one candidate** (its global correlation peak); candidates are
  inserted into the descending top-`n_best` list with `upper_bound`
  semantics (stable: an equal score ranks after the earlier phase, C++
  `std::upper_bound` `:235`). **A candidate is inserted only where it
  strictly beats an existing row** (revision finding, major): the rows are
  seeded with `score 0` / `phase -1`, and `upper_bound` under the
  descending `operator<` (`corr > rhs.corr`, `:63`) places a candidate
  with `score <= 0` *after* every zero-valued fill row, so `idx < n` is
  false and it is **dropped** -- the row keeps `phase_id -1`, identity and
  score 0 (`"only keep something with a positive phase"`, `:219`). A
  naive `np.argsort`-and-truncate port would write a mismatched phase with
  a negative score where EMSphInx reports not-indexed (negative scores are
  reachable: the window-mask correlation measured **-2.6402**, D2); killed
  by a monkeypatched-correlator test (returns -1.0 -> row keeps the fill)
  and the bug-injection entry "non-positive scores inserted". With `P`
  phases there are at most `P` candidates, so
  **rows `>= P` of an `n_best > P` request keep the fill values**
  (`phase_id -1`, identity, score 0) -- the documented C++ behaviour
  ("extra points will be filled with an invalid phase", `indexer.hpp:103`);
  per-phase *secondary* peaks are **not** extracted (EMSphInx has no such
  path; a deviation would need new peak-finding machinery in `_xcorr` --
  deferred until a use case, documented in the docstring). Phase 8's
  pseudo-symmetry candidates will enter the same insertion loop.
- **Finding (measured, refutes the drafted test design): a rotated copy of
  the master is NOT a usable discrimination phase.** The peak value of the
  spherical cross-correlation is invariant under a rotation of the reference
  (the rotated master matches the same pattern at a compensated rotation), so
  the "true" phase wins only by grid-sampling noise: measured at `bw` 68 with
  phase B = the Ni master rotated by `zyz_b = (0.9, 0.7, -0.4)` (flags
  `(1, False)` -- the rotation breaks the z-alignment of the symmetry), the
  true phase wins **6/9** with score gaps **-0.0151 to +0.0090**. The
  multi-phase discrimination test therefore uses a **sign-scrambled copy**
  (`mph.alm * default_rng(42).choice([-1.0, 1.0], mph.alm.shape)` --
  revision finding, major: the first draft's *phase* scramble
  `flm * exp(i U(0, 2 pi))` is not the spectrum of a real spherical
  function -- it puts up to 3.12 of imaginary part into the `m = 0` row,
  breaks the realness convention `a_{l,-m} = (-1)^m conj(a_{l,m})` that
  the storage, `synthesize` and the correlator's real inner product
  assume, and `analyze(synthesize(.))` round-trips at 0.99 *relative*
  error, so `squared_harmonics` was built from a different function than
  `flm`. The sign scramble preserves **every** `|a_lm|` exactly (the
  same-power-spectrum claim is now true), keeps `m = 0` real (measured
  `max |Im| = 0.0`), round-trips at 2.8e-10 like the original, and
  discriminates strictly better): the true phase wins **9/9** with gaps
  **0.2970-0.4151** (~50 % of the score; the decoy's own normalized
  scores 0.1993-0.2194, far below the true 0.4963-0.6239).
  The rotated copy stays in the suite as a **degeneracy
  control**: `max |gap| < 0.05` asserted, and the *composed orientation
  identity* `rotation_from_zyz(zyz_b) * O_B == O_A (mod Oh)` -- measured
  median **0.676** / max **1.074** deg (two independent half-cell
  interpolation errors), while the three wrong compositions
  (`~` and/or reversed order) sit at median **24.3-28.7** deg -- pinning
  both the multi-phase bookkeeping and, independently of D2's stored-xmap
  oracle, the rotation convention.
- Multi-phase construction cost: `flm2` and the correlator are per phase
  (D1); `mlm`, the Wigner table, the projector and the SHT are shared.
  Per-phase per-worker memory adds one `fxc`+`xc` pair (D8).

### D4 -- Thread strategy and chunking (frozen)

- **`dask.array.map_blocks` over pattern chunks with the threaded
  scheduler** (constitution: no `numba parallel=True`; every merged
  `scipy.fft` call already passes `workers=1`). The input is chunked
  `(chunksize, -1, -1)` (`da.from_array` for NumPy input, `rechunk` for dask
  input -- the `hough_indexing` idiom); the chunk worker `_index_chunk`
  returns one packed `(nc, n_best, 6)` float64 block
  (`alpha, beta, gamma, score, phase_id, iq`; `drop_axis=(1, 2)`,
  `new_axis=(1, 2)` -- the `_refinement.py` idiom -- **plus explicit
  `chunks=(patterns.chunks[0], n_best, 6)`**, a revision finding: without
  `chunks=` the graph's declared metadata lies -- measured `r.shape ==
  (9, 1, 1)` with `chunks ((1,)*9, (1,), (1,))` while `r.compute()`
  correctly returns `(9, 2, 6)` -- harmless for this phase's eager
  `compute()` but a trap for any pre-compute slicing or a future
  `compute=False` API; the chunking test asserts `res.shape ==
  (n, n_best, 6)` on the dask array *before* `compute()`), unpacked and
  cast (`phase_id -> int32`) after `compute()`. The default scheduler for dask
  arrays (threads) is used as is -- not forced -- so an outer
  `dask.config.set(scheduler=...)` (tests, debugging) is honoured.
- **Per-chunk `clone()` of the correlators; one shared projector.** The
  correlators mutate `fxc`/`xc`/scratch per `compute()` and are NOT
  thread-safe; `clone()` (Phase 4) shares the Wigner table, spectra and
  `r_den` read-only and allocates only the scratch -- the chunk worker clones
  each phase's correlator (or the prototype, `normalize=False`) **once per
  chunk invocation** and allocates one zeroed (`np.zeros`) `(dim, dim)`
  north/south pair (D2: zero before first use; per-pattern re-zeroing is
  defensive only). The projector, its SHT and the preprocessing
  functions are read-only/pure and shared without copies (Phase 5 D1
  established immutability; `window_harmonics` and the SHT weights are eager
  precisely for this). Cloning per *chunk* rather than per *worker* costs
  **`clone()` = 0.18 / 0.21 / 0.33 ms at `bw` 53 / 68 / 88** (revision
  re-measurement -- the first draft wrongly quoted *construction*,
  0.046-0.096 s warm, which is paid once per indexer, not per chunk; the
  `fxc`/`xc` allocations are calloc-lazy). The worst case is the default
  `chunksize=1` on the small map -- one clone per *pattern* -- and even
  there 0.21 ms is **1.6 %** of one 12.7 ms pattern at `bw` 68, so the
  < 2 % claim holds at every chunking; the clone keeps the worker
  function stateless -- no thread-local registry to leak or race.
- **Determinism (measured)**: the per-pattern computation is identical in
  every chunking, so results are **bitwise identical** across
  `chunksize` 1 / 4 / 9 and 1 / 4 threads (measured `np.array_equal` True on
  the packed blocks), and lazy-vs-eager input is likewise bitwise (same
  chain). The suite asserts both (bug injection: chunk-boundary offsets,
  stale buffers).
- **`chunksize=None` -> the `BatchEstimate` port with one recorded
  deviation** `_batch_estimate(bandwidth, n_workers, n_patterns)`
  (`:189-205`: `scl = bw^3 ln(bw^3)`, `k = 1e-8`, `batch = max(1,
  int(pps/phi))`, then the load-balancing rule `ceil(np/batch) < nt^2 ->
  batch = ceil(np/nt^2)`), with `n_workers` from the active dask config
  (`num_workers`) when set, else `os.cpu_count()`. **Deviation (recorded,
  revision finding)**: the final result is clamped `max(1, ...)` -- the
  verbatim `nt^2` branch returns 0 for `n_patterns = 0`
  (`ceil(0/nt^2) = 0`), which the C++ never meets because its mask
  fallback guarantees patterns; our public path guarantees `n_patterns >=
  1` too (the all-`True` navigation-mask guard), but `_batch_estimate` is
  module-level and the clamp keeps its own contract total. Measured/
  pinned: batch **34 / 15 / 6** at `bw` 53 / 68 / 88 for large `np`, and
  the `nt^2` rule gives **chunksize 1 (9 chunks)** for the 9-pattern map
  at 4 workers -- so small maps parallelise, and `n_chunks >= n_workers`
  holds **whenever `n_patterns >= n_workers`** (revision correction: for
  `n_patterns < n_workers` it degrades to one chunk per pattern, e.g. 3
  unmasked points on 4 workers -> 3 chunks, which is still the best
  possible). The C++ estimate
  (`pps` 25.1 at `bw` 68) happens to sit within 3x of this machine's real
   77.6 pat/s -- recorded, not relied on (the estimate only sizes chunks).
- The threaded scheduler's scaling was measured on the prototype (72
  patterns, chunk 3, `bw` 68): **67.6 / 93.4 / 149.3 / 159.2 pat/s at 1 / 2 /
  4 / 8 workers** (2.2x at 4 workers and 2.4x at 8 over the same-run
  single-worker rate -- partial GIL residency in the pocketfft driver loop
  and dask overhead; recorded baseline, not a gate). The hard floor is
  single-core (D8) and unaffected.

### D5 -- `EBSD.spherical_indexing` and the `CrystalMap` contract (frozen)

- ```
  EBSD.spherical_indexing(
      harmonics, detector,
      bandwidth: int = 68, n_best: int = 1,
      navigation_mask=None, signal_mask=None,
      normalize: bool = True, refine: bool = False,
      n_regions: int = 10, gaussian_background: bool = False,
      circular_mask: bool = False, emsphinx_compatible: bool = True,
      chunksize: int | None = None, verbose: int = 1,
  ) -> CrystalMap
  ```
  placed in `signals/ebsd.py` directly after `dictionary_indexing`, working
  for `EBSD` and `LazyEBSD` (the data array goes to `index_patterns` as is;
  lazy input stays lazy until the one eager compute). It builds a
  `SphericalIndexer` per call. Cost of that, measured honestly (revision
  correction -- the first draft called it "noise next to indexing", which
  is false at benchmark scale): warm construction of the whole kit at `bw`
  68 is **0.046-0.096 s** (Wigner table 0.001-0.013 + projector
  0.015-0.025 + `squared_harmonics` 0.001-0.017 + normalized correlator
  0.030-0.041 s) vs **0.12-0.18 s** to index the *nine-pattern* map --
  roughly a 1:2 split on the smallest real map, amortising below 1 % from
  a few hundred patterns on. Power users hold their own indexer and call
  `index_patterns`; the naming decisions `n_best` (not DI's `keep_n`) and
  `iq` (not HI's `pq`) are recorded in plan 6.12/6.13 and pinned by the
  signature/`CrystalMap` tests.
- Own checks (before the indexer's): signal shape vs `detector.shape` ->
  `ValueError` naming both (the `hough_indexing` message style; the projector
  would only catch it at the first `unproject` otherwise);
  `navigation_mask` -- **kikuchipy polarity, only `False` entries are
  indexed** (constitution) -- four checks: shape equal to the navigation
  shape, is a NumPy ndarray, not all `True` (the three
  `dictionary_indexing` checks `ebsd.py:1932-1944`, messages mirrored;
  revision correction: DI has **no dtype check**, so the fourth is NOT
  "mirrored") plus a **new frozen check of this phase**: `dtype != bool`
  -> `ValueError` `"The navigation mask must be a boolean array"` --
  **in the order is-ndarray, dtype, shape, all-`True`** (corrected
  2026-09-02: the order is not free, and the listing order above is
  wrong. `np.ones(nav_shape, int).all()` is `True`, so an integer mask
  which reaches the all-`True` branch first raises DI's "at least one
  value equal to `False`" instead of the dtype message; and the
  is-ndarray check must come first because a list has no `.dtype`) --
  needed because the flow computes `~navigation_mask`, and bitwise NOT of
  an int 0/1 mask (`~1 == -2`, truthy) would silently index everything
  (plan 6.16); every `harmonics` entry must have
  `.phase` set (`ValueError` naming `MasterPatternHarmonics.phase` -- the
  `PhaseList` needs it; `index_patterns` itself does not) and the phase names
  must be unique.
- Flow: patterns `reshape((-1,) + sig_shape)`; `keep = ~navigation_mask
  .ravel()` selects the indexed subset (boolean axis-0 indexing works for
  NumPy and dask alike); info message printed when `verbose >= 1` (D6);
  `res = indexer.index_patterns(patterns_kept, n_best=n_best,
  chunksize=chunksize, progressbar=verbose >= 1)`; indexing speed printed
  when `verbose >= 1` (`"  Indexing speed: {:.5f} patterns/s"`, the DI/HI
  line).
- **`CrystalMap` construction** (the `_dictionary_indexing` +
  `xmap_from_hough_indexing_data` conventions): `create_coordinate_arrays
  (nav_shape, step_sizes)` with the step sizes from the navigation axes;
  `phase_list = PhaseList([p.phase.deepcopy() for p in indexer.phases])`
  (ids 0..P-1 in `harmonics` order = the `phase_id` values; orix adds
  `not_indexed` for `-1` automatically). **What `xmap.phases` then holds
  (revision finding, blocker)**: orix `CrystalMap.__init__` *deletes*
  every phase-list entry whose id never appears in the `phase_id` array
  (probed on orix 0.14.2), so a losing phase is **absent** from
  `xmap.phases` -- the discrimination test must assert `phases.names ==
  ["ni"]`, never `["ni", "scrambled"]` -- and any `-1` in `phase_id`
  (masked-out fills under `is_in_data`, or a failed point) prepends
  `"not_indexed"`. The full configured phase list survives on
  `indexer.phases` and in the `nbest_phase_id` prop, which is where the
  tests pin it (documented in the docstring); `rotations =
  _euler.rotation_from_zyz(res["zyz"])` of shape `(n, n_best)`, flattened to
  `(n,)` when `n_best == 1` (the DI `keep_n == 1` squeeze);
  `phase_id = res["phase_id"][:, 0]` (the best row); `prop["scores"] =
  res["scores"]` (`(n, n_best)`, squeezed when `n_best == 1`),
  `prop["iq"] = res["iq"]` (`(n,)`), and **when `n_best > 1` additionally
  `prop["nbest_phase_id"] = res["phase_id"]`** (`(n, n_best)` int32 -- the
  per-row phase of every candidate; a `CrystalMap` can hold only one
  `phase_id` per point, and for a single phase the extra rows read
  `[0, -1, ...]`, self-describing the fill). With a `navigation_mask` the
  arrays are expanded to the full map with `xmap_kw["is_in_data"] =
  keep` and the fill values of D2 on the masked points (identity, score 0,
  phase -1, iq 0 -- deterministic where DI leaves `np.empty`); failed
  *indexed* points keep `is_in_data True` with `phase_id -1`, so orix reports
  them as `not_indexed` and **`xmap.is_indexed` is `False` exactly there**
  (constitution result contract). `xmap.scan_unit =
  _get_navigation_axes_unit(am)` (`"um"` for the Ni maps, `"px"` fallback).
- Docstring (numpydoc): opens "Index patterns by spherical cross-correlation
  with one or more master patterns :cite:`lenthe2019spherical`"; the
  **un-normalised-scores note verbatim from the constitution** -- the
  `"scores"` prop is not a bounded NCC, comparable only within a fixed
  geometry/bandwidth, and unusable with `orientation_similarity_map`
  (measured range 0.50-0.62 for the Ni data at `bw` 68; `normalize=False`
  0.28-0.35); the memory-guidance table from D8 (per-worker MB at `bw`
  53/68/88/113 and the "choose `bw` from `fast_bandwidths()`" advice);
  the coarse-only note ("orientations land within about half a grid cell,
  `180/slP` deg; refinement of the coarse orientation is not implemented
  yet" -- **no roadmap phase numbers anywhere in public docstrings**,
  revision decision, plan 6.14); the single-PC note naming
  `detector.pc_average`; `See Also`: `dictionary_indexing`,
  `hough_indexing`, `refine_orientation`,
  `kikuchipy.indexing.SphericalIndexer`,
  `kikuchipy.indexing.MasterPatternHarmonics`,
  `kikuchipy.indexing.fast_bandwidths`; `Notes` closing with the
  `IndexEBSD` provenance and that EMSphInx-parity use requires the default
  `emsphinx_compatible=True` **in both `from_master_pattern` and this
  method** (the master-normalisation quirk is chosen at
  harmonics-construction time, D1).

### D6 -- Info message and verbosity (frozen)

- `SphericalIndexer.get_info_message(n_patterns, chunksize) -> str`, printed
  by the signal method when `verbose >= 1` (`verbose: int = 1` mirrors
  `hough_indexing`; `0` silences the message, the speed line *and* the
  progress bar). Template (asserted by substring, not full equality, so
  cosmetic edits do not break tests):
  ```
  Spherical indexing information:
    Phase(s): ni (m-3m; 4-fold, mirror)
    Bandwidth: 68 (Euler side length 135, half cell 1.33 deg)
    Correlation: normalized
    Preprocessing: n_regions = 10, gaussian_background = False
    Projection center (Bruker): (0.4251, 0.2134, 0.5007)
    Indexing 9 pattern(s) in 9 chunk(s) of up to 1 pattern(s)
    Estimated memory per worker: 49 MB
  ```
  (multi-phase: one `Phase(s)` line joining names; the PC line mirrors
  `_hough_indexing._get_info_message`'s rounding; the memory line prints
  **the D8 model `memory_per_worker_bytes`**, which at `bw` 68 single
  phase is 135^2 x 68 x 24 + 135^3 x 8 = 49,426,200 B -> "49 MB" --
  revision fix: the first draft's template said "45 MB", the *measured
  tracemalloc peak* (44.8 MB, within 10 % of the model), contradicting
  its own frozen model; the message prints the model, the measurement
  stays in D8's table). The progress bar is dask's
  `ProgressBar` context around the one `compute()` (the DI idiom).

### D7 -- Accuracy tolerances: derivation and measured values (frozen)

- **Derivation (the stated arithmetic)**: the Phase 5 measured mean-PC error
  floor is the geometry error of indexing a per-point-PC map with
  `pc_average` -- `nickel_ebsd_small` median **0.33** / max **0.54** deg;
  `nickel_ebsd_large` 165-point subset median **0.29** / p95 **0.74** / max
  **0.96** deg (D12 there). The coarse grid contributes at most half a cell,
  **`180/slP = 1.33` deg at `bw` 68**, of which the tri-quadratic
  interpolation recovers most (Phase 5 D8 on noiseless simulations at the
  same `bw`: 0.34 median / 0.72 max). Adding the two independent error
  sources in quadrature predicts a median around
  `sqrt(0.33^2 + 0.34^2) ~ 0.47` and a max around
  `sqrt(0.54^2 + 0.72^2) ~ 0.9` deg for the small map -- and the end-to-end
  measurement lands there: **median 0.599 / max 0.838 deg at `bw` 68**
  (per-pattern: 0.35, 0.75, 0.60, 0.45, 0.48, 0.59, 0.71, 0.68, 0.84).
  The roadmap's coarse bounds (floor + full half cell + interpolation
  defects: median < 1.5, `>= 8/9 < 3` deg) hold with 2.5-3.6x margin.
- **Asserted (small map, `bw` 68, default suite)**: the roadmap bounds
  **median < 1.5 deg** and **at least 8 of 9 < 3 deg** (binding), plus the
  measured-then-pinned tighteners **all nine < 2.0 deg** (2.4x margin on the
  measured max) and **median < 1.2 deg** (2.0x); the nine values and the
  median are `record_property`. Recorded at the other bandwidths (same
  fixtures, cheap): `bw` 53 median 0.747 / max 0.991 (asserted all < 3 only
  -- 1.7 half cells of margin); `bw` 88 median 0.524 / max 0.571 (weekly,
  recorded). `normalize=False` at `bw` 68: median 0.601 / max 0.836 --
  asserted under the same bounds (the metric changes, the argmax barely
  does); `n_regions=0`: 0.605 / 0.852 (recorded -- preprocessing is nearly
  neutral on background-corrected patterns). Non-default configurations,
  measured for this revision so their tests pin *values*, not just
  differences (review finding: the masked/optional paths were previously
  covered by tautology-prone "differs from unmasked" assertions):
  **`signal_mask[20:32, 25:40]`**: median 0.496 / max 0.683 deg, scores
  **0.4461-0.5762** (mean 0.5307), IQ 0.1740-0.2028, max relative score
  change vs unmasked **10.1 %** (the Phase 5 `rDen` 10.8 % in action);
  **`circular_mask=True`**: `projector.n_points` **1117** (vs 1317),
  median 0.604 / max 0.856 deg, scores **0.4915-0.6390**, IQ
  0.1920-0.2224 (the circle enters the histogram too, D1);
  **`gaussian_background=True`**: median 0.594 / max 0.816 deg, scores
  **0.4942-0.6101**, IQ 0.1873-0.2159 with `emsphinx_compatible=True`,
  and with `False` scores 0.4960-0.6109 -- max per-point score difference
  between the two settings **1.83e-3** (max IQ difference 6.7e-4): real
  but sub-tolerance, so the compat mutant is killed by a
  scores-not-identical assertion (`max |diff| > 1e-4`) plus the kwarg
  spy, never by the `rel=0.05` pins.
- **Asserted (`nickel_ebsd_large`, `pc_average`, backgrounds removed)**:
  default suite, the Phase 5 20-point subset (`[::15, ::15]`): measured
  median **0.499** / p90 1.075 / max **1.350** deg -> asserted **median <
  1.5, max < 3.0** (2.2x margin); weekly, the 165-point subset
  (`[::5, ::5]`): measured median **0.530** / p90 0.931 / p95 **1.082** /
  max **1.495** deg, zero points above 2 deg -> asserted **median < 1.5,
  p95 < 2.5, max < 3.5**, all values `record_property`. (The download-backed
  tests skip cleanly without the `tests` extra, Phase 5 idiom.)
- **Scores and IQ measured-then-pinned** (constitution): at `bw` 68,
  default configuration, harmonics built directly at `bw` 68 (D1 note),
  normalized scores over the nine patterns **min 0.4963 / max 0.6239 / mean
  0.5701**, un-normalized **0.2799 / 0.3533**, each
  `pytest.approx(., rel=0.05)`; IQ **min 0.1727 / max 0.2036 / mean 0.1878**
  (`rel=0.05`); large-map scores 0.4602-0.6506 (20-pt) recorded. The
  benchmark asserts `xmap.scores.mean() == approx(0.570, abs=0.03)`.

### D8 -- Performance, memory, the hard floor, the benchmark (frozen gates + recorded baselines)

- **Measured single-core, this machine** (Windows 11, warm JIT, best of 5
  sweeps over the nine 60x60 patterns; per-stage means):

  | `bw` | preprocess | unproject | `analyze` | correlate | total | pat/s/core |
  |---|---|---|---|---|---|---|
  | 53 | 0.20 ms | 0.15 | 0.22 | 5.67 | 6.24 ms | **160.3** |
  | 68 | 0.21-0.23 | 0.20 | 0.46 | 11.8-12.0 | 12.6-12.9 ms | **77.6-79.1** |
  | 88 | 0.22 | 0.20 | 0.95 | 29.8 | 31.2 ms | **32.1** |

  The correlator is > 90 % of the budget (Phase 4's baseline stands);
  preprocessing + back-projection + SHT stay < 8 %.
- **The hard floor `>= 2` patterns/s/core at `bw` 68 (60x60)** (constitution
  gate) passes with **~39x margin**; the fallback list (float32,
  coarse-only default) is untouched. The floor is asserted in the default
  suite as the single loose timing bound (single-thread, warm, the nine
  patterns; every other number is `record_property`) and re-asserted by the
  benchmark job.
- **Memory per worker** (tracemalloc; one chunk kit = per-phase correlator
  clone(s) + the north/south pair; the `bw` 63 and 113 rows were measured
  in the revision pass, completing the constitution's {63, 68, 88, 113}
  set -- 53 is an extra):

  | `bw` | resident after clone | after first correlate | transient peak | model |
  |---|---|---|---|---|
  | 53 | 9.4 MB | 14.2 | 21.4 | 23.3 |
  | 63 | 15.9 | 23.8 | 35.7 | 39.2 |
  | 68 | 20.0 | 30.0 | 44.9 | 49.4 |
  | 88 | 43.4 | 65.1 | 97.6 | 107.6 |
  | 113 (weekly) | 91.9 | 137.7 | 207.2 | 228.4 |

  `+ ~24 bytes x slP^2 x bwP` per *additional* normalized phase.
  `SphericalIndexer.memory_per_worker_bytes` exposes the model
  **`(n_phases if normalize else 1) x slP^2 x bwP x 24 + slP^3 x 8`**
  bytes (model >= peak, within ~10-25 %; the constitution's "warning
  helper on `SphericalIndexer`"). The `normalize` factor is a revision
  fix: with `normalize=False` one shared scratch correlator serves every
  phase (D1 -- the C++ `UnNormalizedCorrelator` holds only a spectrum,
  ~0.07 MB/phase at `bw` 68), so the unconditional `n_phases` factor
  over-estimated un-normalised multi-phase runs 2.8x at 4 phases and
  could fire the warning spuriously; pinned by a
  `memory_per_worker_bytes`-equality test (2 phases `normalize=False` ==
  1 phase). The info message prints the model, and `index_patterns` emits
  a `UserWarning` when `n_workers x memory_per_worker_bytes > 2 GiB`
  (documented threshold, named test at `bw` 68 with
  `dask.config.set(num_workers=64)`: 64 x 49.4 MB = 3.16 GB fires it; at
  `bw` 113 x 8 workers the model gives **1.83 GB = 1.70 GiB** -- the
  first draft's "~1.7 GB" was GiB mislabeled as GB -- and the measured
  peak 207.2 MB x 8 = 1.66 GB: near, not over). Loose
  assertion: peak per-worker `< 200 MB` at `bw` 68 (4.5x margin), the rest
  recorded.
- **Benchmark** `benchmarks/indexing/test_spherical_indexing.py`: one
  `test_spherical_indexing` mirroring `test_dictionary_indexing.py` --
  `nickel_ebsd_small` with backgrounds removed, `pc_average` detector,
  harmonics at `bw` 68 built *outside* the benchmarked callable,
  `benchmark(s.spherical_indexing, harmonics=mph, detector=det, verbose=0)`,
  asserting `np.isclose(xmap.scores.mean(), 0.570, atol=0.03)` and
  `xmap.rotations.size == 9` -- the relaxed way-off check of the DI
  benchmark, plus the floor `9 / benchmark.stats["mean"] >= 2` patterns/s.
  Stated honestly (revision finding): on this nine-pattern map the
  benchmarked callable is roughly **1/3 per-call indexer construction,
  2/3 indexing** (0.046-0.096 s vs 0.12-0.18 s, D5), so this is an
  explicit **map-level end-to-end floor including setup** -- kept
  end-to-end deliberately, because it mirrors the DI benchmark and the
  user-visible call, and even so passes with >= 15x margin (~32-53
  map-level pat/s measured); the *pure* per-pattern floor lives in the
  default suite's `TestPerformance`, which times `index_patterns` on a
  pre-built indexer (77.6 pat/s, 39x). Single-machine benchmark job;
  `benchmark.stats["mean"]` is valid pytest-benchmark API
  (`Metadata.__getitem__`, checked).

### D9 -- Public API, exports, CHANGELOG, docs (frozen)

- `src/kikuchipy/indexing/__init__.pyi` gains
  `from ._spherical._back_projection import SphericalBackProjector`,
  `from ._spherical._fft import fast_bandwidths`,
  `from ._spherical._indexer import SphericalIndexer`, and the three names in
  the sorted `__all__` (`lazy_loader.attach_stub` mechanism; mission table:
  `IndexEBSD -> EBSD.spherical_indexing() + kp.indexing.SphericalIndexer`,
  `ShtWisdom -> kp.indexing.fast_bandwidths()`; Phase 5 D12:
  `SphericalBackProjector` exported here with the indexer). All three render
  in the numpydoc reference (generated from `__all__`), which the
  automated `sphinx-build`/`linkcheck` commands in `validation.md` now
  verify (revision finding: doc/conf.py runs numpydoc validation with
  `"all"` minus a small exclude set, and nothing exercised it).
  **Docstring publication pass** (revision finding -- three docstrings
  become public API documentation): `fast_bandwidths` gains an `Examples`
  section and a `See Also` -> `kikuchipy.indexing.SphericalIndexer`, and
  its `:func:`fast_size`` cross-reference becomes prose (`fast_size` stays
  private and the link would dangle; the first draft's "drop the 'private
  until Phase 6' caveat" task was a no-op -- that caveat lives in
  `specs/roadmap.md:30`, not in the docstring, verified by grep);
  `SphericalBackProjector`'s docstrings are scrubbed of private
  cross-references (`~..._grid.default_dim`,
  `._sht.SphericalHarmonicTransform`,
  `._xcorr.NormalizedSphericalCrossCorrelator` at `_back_projection.py`
  `:1001/:1015/:1065/:1341` -> prose or public targets) and of roadmap
  phase numbers ("Phase 6 shares one projector..." `:1086`, "...reuses
  per thread" `:1409` -> phase-free wording); `SphericalIndexer`'s and
  `EBSD.spherical_indexing`'s public text names no phases (D1/D5).
  `fast_bandwidths` keeps the EMSphInx
  suggested-values note (measured: `fast_bandwidths(16, 128)` returns
  int64 `[17, 18, 20, 23, 25, 28, 32, 33, 38, 39, 41, 46, 50, 53, 59, 61,
  63, 68, 72, 74, 83, 85, 88, 95, 98, 113, 116, 122, 123]`, a superset of
  `nml.hpp:415`'s suggested `53, 63, 68, 74, 88, 95, 113, 122, 123` in this
  range -- pinned).
- CHANGELOG (`Unreleased -> Added`, fork-PR links, constitution format),
  **two entries**, PR number pinned as **#8** (the next fork PR after
  jwestraadt/kikuchipy#7; revision finding -- the constitution's exact
  link form needs the number up front), verbatim:
  (1) "Spherical indexing of EBSD patterns against one or more master
  patterns with ``EBSD.spherical_indexing()``,
  ``kikuchipy.indexing.SphericalIndexer`` and
  ``kikuchipy.indexing.SphericalBackProjector``, a CPU port of EMSphInx's
  ``IndexEBSD`` (`#8 <https://github.com/jwestraadt/kikuchipy/pull/8>`_)";
  (2) "``kikuchipy.indexing.fast_bandwidths()``, the spherical indexing
  bandwidths with fast transforms
  (`#8 <https://github.com/jwestraadt/kikuchipy/pull/8>`_)". (If GitHub
  assigns a different number at PR time, both links are updated in the
  PR commit itself -- the entry text is otherwise frozen.)
- `SphericalIndexer` carries an `Examples` doctest (the `nickel_ebsd_small`
  detector with `pc_average`, `bw` 68, printing the `repr`) for the
  `--doctest-modules` gate; `SphericalBackProjector`'s Phase 5 example
  already passes it.

### D10 -- Kernels, style, licences (frozen)

- **No new `@njit` kernel** (the chunk loop is plain Python over merged
  nogil kernels and `scipy.fft` calls that release the GIL) and **no new
  `scipy.fft` call site** (the Phase 5 recording test on
  `_back_projection.dctn` still covers every FFT call); the kernel-flag
  regression of Phases 4/5 (exactly two `error_model="numpy"` kernels in the
  project) is unchanged and re-asserted.
- `_indexer.py`: kikuchipy GPL header + the delimited EMSphInx notice
  (CMU/Lenthe, GPL-2.0-or-later conveyed under GPL-3.0-or-later, "changed by
  Johan Westraadt, 2026-09") enumerating the ported functions with line
  ranges: `idx/indexer.hpp` (`Result` `:54-64`, `Indexer` `:68-181`,
  `BatchEstimate` `:189-205`, `indexImage` `:216-270` (pseudo-symmetry loop
  `:243-261` deferred to Phase 8), `computeHarmonics` `:312-318`,
  `correlate` `:326-331`; `refineImage`/`refine` not ported until Phase 7),
  `modality/ebsd/idx.hpp` (correlator/projector wiring `:252-296`,
  `ebsdWorkItem` failure semantics `:382-456`; HDF5/PNG output, ROI mask,
  `ThreadedIqCalc` not ported), `idx/base.hpp` (interfaces collapsed into
  the concrete Phase 1-5 classes -- stated). `signals/ebsd.py`,
  `indexing/__init__.pyi`, the test and benchmark modules: kikuchipy header
  only (kikuchipy-convention glue; the EMSphInx-derived logic lives in
  `_indexer.py`).
- Style: numpydoc, type hints in signatures only, comment/docstring lines
  <= 72 chars, three import blocks, `pre-commit run --files` on the changed
  files only. `_spherical/__init__.py` adds ``_indexer`` to the `Submodules`
  block, importing nothing.

## Context

- Algorithm reference: `specs/_research/explore-emsphinx-core-algorithm.md`
  sections 6.1-6.4 (indexer, EBSD driver, namelist defaults incl. the
  `circmask` addendum), 3.8-3.10 (correlate, normalized correlator, the
  metric's meaning), 5.5 and 8 (gotcha list; new items from this spec: the
  AHE constant-ripple consequence for `unproject`'s mask branch, the
  rotated-copy degeneracy, the zero-seeded insertion rule, the
  geometry-from-master tilt binding, the sign-vs-phase scramble for decoy
  phases -- plan 0.6 adds addenda).
- EMSphInx sources read for this spec: `include/idx/indexer.hpp` (all),
  `include/modality/ebsd/idx.hpp:140-456`, `include/idx/base.hpp:40-150`,
  `include/modality/ebsd/nml.hpp` (defaults table via the research doc).
- kikuchipy patterns mirrored: `indexing/_dictionary_indexing.py` (info
  message, `CrystalMap` + `is_in_data` mask handling, keep-1 squeeze, speed
  line), `indexing/_hough_indexing.py` + `EBSD.hough_indexing` (chunksize,
  verbose, shape checks, `xmap_from_hough_indexing_data`'s `phase_id`/
  not-indexed handling, `scan_unit`), `_refinement/_refinement.py`
  (`map_blocks` with `drop_axis`/`new_axis`, packed result rows,
  `get_info_message`), `signals/util/_dask.py` (`get_chunking`/
  `get_dask_array` -- superseded here by the `BatchEstimate` chunking, which
  the spec prefers for its `nt^2` small-map rule), `_get_navigation_axes_unit`,
  `indexing/__init__.pyi`, `CHANGELOG.rst`, `benchmarks/indexing/
  test_dictionary_indexing.py`, `tests/test_signals/test_ebsd_hough_indexing.py`
  (the backgrounds-removed fixture recipe and `angle_with` comparisons).
- Phase 1-5 deliverables composed (their `validation.md` Recorded results
  govern the component-level numbers): `MasterPatternHarmonics`
  (`from_master_pattern`, `resize`, `alm`, `n_fold`,
  `has_equatorial_mirror`, `phase`, `beam_energy`, `sample_tilt`),
  `SphericalBackProjector` (`unproject(out=...)` south-untouched contract,
  `window_harmonics`, `squared_harmonics`, guards),
  `_preprocessing._preprocess_pattern`, `SphericalHarmonicTransform.analyze`,
  `SphericalCrossCorrelator`/`NormalizedSphericalCrossCorrelator`
  (`clone()`, `refine` raises), `_euler.rotation_from_zyz` (sign frozen,
  Phase 5 D8), `_wigner.wigner_d_half_pi_table`/`rotate_harmonics` (test
  fixtures), `_fft.fast_bandwidths`.
- Real data: `nickel_ebsd_small()` (9 patterns, stored `xmap`, per-point PC
  -> `pc_average`), `nickel_ebsd_master_pattern_small(projection="lambert",
  hemisphere="both")`, `nickel_ebsd_large(allow_download=True)` (20-point
  subset default suite, 165-point weekly; skips without the `tests` extra).
- CI lessons applied (constitution): no bitwise assertions across libraries
  or compiled-vs-interpreted; the determinism tests compare the *same* code
  path under different chunking/scheduling only; no `ss == 0`-style knife
  edges in inputs (the failure-injection constant is `ptp == 0` on *raw
  integer input*, an exact integer property, not a float knife edge); the
  composed-orientation test uses only `Rotation.__mul__`/`angle_with`
  (no `outer`/`reduce`); no tight timing bounds (one loose floor);
  deterministic seeds; `-n 4` clean after a `-n 0` warm-up; `scipy.fft`
  `workers=1` throughout (inherited); the per-xdist-worker
  `NUMBA_CACHE_DIR` conftest arrangement is untouched.
