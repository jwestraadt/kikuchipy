# Phase 6 -- `spherical-indexing-ebsd`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py -n 0   # first run: warm the numba caches in one process
uv run pytest tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pytest --cov=kikuchipy.indexing._spherical --cov=kikuchipy.signals.ebsd --cov-report=term-missing tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py
uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py
uv run pre-commit run --files src/kikuchipy/indexing/_spherical/_indexer.py src/kikuchipy/indexing/_spherical/_fft.py src/kikuchipy/indexing/_spherical/_back_projection.py src/kikuchipy/indexing/_spherical/__init__.py src/kikuchipy/indexing/__init__.pyi src/kikuchipy/signals/ebsd.py tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py benchmarks/indexing/test_spherical_indexing.py CHANGELOG.rst
uv run sphinx-build -b html doc doc/_build/html
uv run sphinx-build -b linkcheck doc doc/_build/linkcheck
```

(Revision fixes: the module tests live in
`tests/test_indexing/test_spherical_indexer.py` per the constitution's
test layout; the coverage command now actually covers `signals/ebsd.py`,
which the >= 95 % gate below names; the sphinx commands verify the three
newly public docstrings under numpydoc validation and dead-link checking
-- nothing automated exercised the doc build before;
`_back_projection.py` joins the pre-commit list because its docstrings
are scrubbed for publication, D9.)

Definitions used below: "the signal" is `kp.data.nickel_ebsd_small()` after
`remove_static_background()` and `remove_dynamic_background()` (uint8, the
`test_ebsd_hough_indexing.py` fixture recipe); "the detector" is
`s.detector.deepcopy()` with `pc = pc_average`
(`(0.42513885, 0.21336699, 0.50070692)`, 60x60, `sample_tilt` 70, `tilt` 0);
"the harmonics" is `MasterPatternHarmonics.from_master_pattern(
nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both"),
bandwidth=68)` built **directly at `bw` 68** (flags `(4, True)`, default
`emsphinx_compatible=True`; the D1 note: `resize(120 -> 68)` differs by
2.9 % max-abs / 10.3 % relative-L2 / up to ~100 % on individual
significant coefficients, so the pins below assume direct construction);
"misorientation" is `Orientation(a.data, Oh).angle_with(Orientation(b.data,
Oh), degrees=True)`; "the default call" is `s.spherical_indexing(mph, det,
verbose=0)` (all other parameters at their defaults: `bw` 68, `n_best` 1,
normalized, `n_regions` 10, no masks). Every "measured" value comes from the
drafting probe on the merged Phase 1-5 modules, recorded at the end of this
file; score/IQ assertions are measured-then-pinned
(`pytest.approx(., rel=0.05)` unless stated). Random data uses
`np.random.default_rng(seed)`; nothing depends on test order. Bitwise
assertions appear only between runs of the *same* code path under different
chunking/scheduling/laziness (legitimate: same arithmetic on the same
bytes); nothing bitwise crosses a library boundary.

Required assertions (each is a named test):

Construction and guards (D1)
- `inspect.signature(SphericalIndexer.__init__)` defaults: `bandwidth=68`,
  `normalize=True`, `refine=False`, `signal_mask=None`, `n_regions=10`,
  `gaussian_background=False`, `circular_mask=False`,
  `emsphinx_compatible=True`; `EBSD.spherical_indexing` adds `n_best=1`,
  `navigation_mask=None`, `chunksize=None`, `verbose=1` (the `IndexEBSD`
  namelist defaults, pinned so a drive-by "improvement" is caught).
- `SphericalIndexer(mph, det, refine=True)` -> `NotImplementedError` whose
  message contains `"refine=True"` and `"not implemented"` (the frozen D1
  text; revision decision 6.14: public messages carry no roadmap phase
  numbers), raised before any projector/correlator construction --
  asserted exactly (revision fix of an either/or that included a
  forbidden timing bound): with
  `monkeypatch.setattr(SphericalBackProjector, "__init__",
  raise_assertion)` the call still raises `NotImplementedError`, not
  `AssertionError`, pinning the guard order with no wall clock.
- `SphericalIndexer(mph, det, bandwidth=8)` and `(..., bandwidth=600)` ->
  `ValueError` containing `"unreasonable bandwidth"` and `"[16, 512]"`
  (the `nml.hpp:635` rule, D1 revision addition); `bandwidth=16`
  constructs and indexes one pattern, and `bandwidth=512` is asserted to
  reach the projector through the range guard with the
  `SphericalBackProjector.__init__` sentinel (corrected 2026-09-02:
  `512` is *not* constructed -- `slP` 1024 / `bwP` 513 give 12.91 GB of
  correlator cubes plus an 8.59 GB interpolation cube, 21.5 GB of model,
  plus a 1.07 GB Wigner table; arithmetic recorded below and as a
  `record_property` in the test).
- **Tilt binding** (D1 revision addition, major finding): harmonics with
  `sample_tilt` 70.0 and a detector with `det.sample_tilt = 65.0` ->
  `ValueError` quoting `70`, `65` and naming `EBSDDetector.sample_tilt`;
  a harmonics with `sample_tilt=None` and any detector tilt -> no raise
  (the skip rule). Rationale recorded as `record_property` in the test:
  without the guard the mismatch indexes at median 4.680 / max 5.099 deg
  with *higher* scores (0.5053-0.6474) than the correct run -- silently
  wrong, undetectable from scores.
- Guards propagated through the indexer constructor: the raw
  `nickel_ebsd_small().detector` (9 PCs) -> `ValueError` containing
  `"pc_average"` and `"deepcopy"` (the Phase 5 message, unaltered);
  `det.azimuthal = 5` -> `ValueError` containing `"azimuthal"`;
  `SphericalIndexer([], det)` -> `ValueError`; `SphericalIndexer([mph, 3],
  det)` -> `TypeError` naming index 1; `n_regions = 61` (> min(shape)) or
  `-1` -> `ValueError`; two harmonics with `sample_tilt` 70 vs 60 (or
  `beam_energy` 20.1 vs 15) -> `ValueError` quoting both values; a
  harmonics with `sample_tilt=None` paired with the Ni harmonics -> no
  raise (the skip rule).
- **Bandwidth alignment**: `SphericalIndexer(mph_120, det, bandwidth=68)`
  (a `bw` 120 harmonics) constructs emitting **no `UserWarning`** --
  asserted with `warnings.catch_warnings(record=True)` filtered to
  `UserWarning`, never `pytest.warns(None)`/`simplefilter("error")`
  (revision fix: harmonics construction deep-copies the phase, which
  emits an unrelated diffpy `DeprecationWarning`
  ("GetSpaceGroup ... deprecated"), so an error-on-any-warning
  implementation of this assertion fails) -- and
  `indexer.phases[0].bandwidth == 68` (truncation);
  `SphericalIndexer(mph_53, det, bandwidth=68)` emits a `UserWarning`
  naming 53 and 68 (zero padding); the correlate path still runs (a smoke
  index of one pattern).
- Attributes: `n_phases == 1`, `bandwidth == 68`, `normalize is True`,
  `projector` is a `SphericalBackProjector` with `bandwidth 68` and
  `n_points == 1317` (`circular_mask=False`), `side_length == 135`,
  `half_cell_degrees == pytest.approx(1.3333, abs=1e-3)`; the Wigner table
  is shared: with two phases, `indexer.correlators[0].correlator
  .wigner_d_half_pi is indexer.correlators[1].correlator.wigner_d_half_pi`;
  `repr` contains `"1 phase"`, `"ni"`, `"bw = 68"`, `"normalized"`; the
  `repr` of an indexer built on a phase-less harmonics
  (`MasterPatternHarmonics(mph.alm)`, via `index_patterns`-only
  construction) contains `"?"` (the D1 fallback -- named test, revision
  addition: previously specified but never asserted);
  mutating the caller's `det.pc` after construction changes neither
  `indexer.projector.detector.pc` nor any subsequent result (bitwise rerun).
- `index_patterns(pats, chunksize=0)` (and `-1`) -> `ValueError` (D2
  revision addition).
- `EBSD.spherical_indexing` own checks: a `(60, 59)` detector on the 60x60
  signal -> `ValueError` naming both shapes; `navigation_mask` of wrong
  shape / not-an-ndarray / all-`True` -> the three `ValueError`s with the
  DI messages, and a non-boolean ndarray (`np.ones(nav_shape, int)`) ->
  `ValueError` `"The navigation mask must be a boolean array"` (the new
  frozen message of this phase, D5 -- revision fix: DI has no dtype check
  and no such message, so this one is *not* labelled "mirrored"); a
  phase-less harmonics (constructed from `mph.alm` without
  `phase`) -> `ValueError` containing `"phase"` (index_patterns itself
  accepts it -- asserted by a direct `index_patterns` smoke call).

`_batch_estimate` (D4)
- Pinned: `_batch_estimate(53, 8, 100000) == 34`,
  `_batch_estimate(68, 8, 100000) == 15`, `_batch_estimate(88, 8, 100000)
  == 6` (the `k = 1e-8` model; C++ `int()` truncation, not `round`);
  `_batch_estimate(68, 4, 9) == 1` and `_batch_estimate(68, 1, 9) == 15`
  (the `nt^2` load-balancing rule); `_batch_estimate(68, 4, 0) == 1` --
  the pin of the **recorded `max(1, ...)` clamp deviation** (D4, revision
  fix: the verbatim `nt^2` branch returns `ceil(0/16) = 0` here, so
  "always >= 1" is our one-line addition, not C++ behaviour).

Real data: `nickel_ebsd_small` coarse vs the stored `xmap` (D7)
- **The default call at `bw` 68**: misorientation to `s.xmap` per point --
  **median < 1.5 deg and at least 8 of 9 < 3 deg** (the roadmap bounds,
  binding; measured median 0.599, all nine 0.35-0.84), **all nine < 2.0
  deg** and **median < 1.2 deg** (measured-then-pinned tighteners, 2.0-2.4x
  margin); the nine values, the median and the max are `record_property`.
  Arithmetic stated in the test docstring: Phase 5 floor (median 0.33 /
  max 0.54) (+) interpolated grid error at `slP` 135 (0.34 / 0.72 measured
  Phase 5 D8) in quadrature predicts ~0.47 / ~0.9; measured 0.599 / 0.838.
- **Scores pinned**: over the nine points `scores.min() == approx(0.4963)`,
  `.max() == approx(0.6239)`, `.mean() == approx(0.5701)` (all
  `rel=0.05`); **IQ pinned**: `iq.min() == approx(0.1727)`, `.max() ==
  approx(0.2036)` (`rel=0.05`) -- the processed-pattern IQ (`n_regions=10`);
  the raw-IQ mutant (0.29-0.33 for these backgrounds-removed patterns,
  0.77-0.78 for raw ones) dies here.
- **`normalize=False`**: same misorientation bounds (measured 0.601 /
  0.836); `scores.min() == approx(0.2799)`, `.max() == approx(0.3533)`
  (`rel=0.05`) -- distinct from the normalized range, so a
  `normalize`-ignored mutant dies.
- **`n_regions=0`**: misorientations recorded (measured 0.605 / 0.852);
  IQ `in [0.25, 0.40]` (measured 0.2890-0.3269) -- distinct from the
  `n_regions=10` band, so a preprocessing-not-forwarded mutant dies.
- **`circular_mask=True`** (revision addition -- previously untested):
  `indexer.projector.n_points == 1117` (< 1317, the circle cuts the
  window), same D7 misorientation bounds (measured 0.604 / 0.856);
  `scores.min() == approx(0.4915)`, `.max() == approx(0.6390)`
  (`rel=0.05`); IQ `.min() == approx(0.1920)`, `.max() ==
  approx(0.2224)` (`rel=0.05`) -- the circle reaches the histogram mask
  too (`circmask = 0` coupling, D1), which is what shifts IQ up from the
  unmasked 0.1727-0.2036 band.
- **`gaussian_background=True`** (revision addition -- previously
  untested, and the only path on which the indexer's
  `emsphinx_compatible` has any numeric effect): same D7 bounds (measured
  0.594 / 0.816); `scores.min() == approx(0.4942)`, `.max() ==
  approx(0.6101)` (`rel=0.05`); rerun with `emsphinx_compatible=False`:
  same bounds, and `np.abs(scores_true - scores_false).max() > 1e-4`
  (measured 1.83e-3 -- the Gaussian-fit off-by-one is real but
  sub-tolerance, so this non-identity assertion plus the plan 4.1 kwarg
  spy kill the not-forwarded mutant; the `rel=0.05` pins cannot).
- **`bw` 53** (same fixtures): all nine < 3 deg (measured median 0.747 /
  max 0.991, recorded); **`bw` 88 weekly**: recorded (measured 0.524 /
  0.571).
- **CrystalMap structure** (the default call): `isinstance(xmap,
  CrystalMap)`, `xmap.shape == (3, 3)`, `xmap.scan_unit == "um"`,
  `xmap.phases.names == ["ni"]` (space group Fm-3m carried),
  `xmap.rotations.shape == (9,)` (the `n_best == 1` squeeze),
  `xmap.scores.shape == (9,)` float64, `xmap.iq.shape == (9,)`,
  `xmap.phase_id` all 0, `xmap.is_indexed.all()`,
  `"nbest_phase_id" not in xmap.prop` (only added when `n_best > 1`);
  coordinate arrays match `create_coordinate_arrays((3, 3), (1.5, 1.5))`
  (the map's step sizes).

`n_best` (D3)
- The default call with `n_best=3` (single phase): `rotations.shape ==
  (9, 3)`, `scores.shape == (9, 3)`, row 0 equals the `n_best=1` run
  bitwise; **rows 1-2 carry the fill**: scores exactly `0.0`,
  `nbest_phase_id[:, 0] == 0` and `nbest_phase_id[:, 1:] == -1`, rotations
  1-2 are the identity (`angle_with(identity) == 0`); scores are
  non-increasing along axis 1. `n_best=0` -> `ValueError`.
- **Non-positive scores are never inserted** (D3 revision addition, major
  finding): with the correlator's `correlate` monkeypatched to return
  `((0.5, 0.5, 0.5), -1.0)`, `index_patterns` rows keep the full fill --
  `phase_id == -1`, `"zyz"` exactly zero, `scores == 0.0` (the C++
  zero-seeded `upper_bound` semantics; an `np.argsort`-top-n mutant
  writes phase 0 with score -1.0 and dies here).

Multi-phase (D3)
- **Sign-scrambled discrimination** (fixture revised, major finding: the
  earlier *phase* scramble `alm * exp(i U(0, 2 pi))` was not a real
  spherical function -- `m = 0` row imaginary part up to 3.12,
  `analyze(synthesize(.))` relative error 0.99 -- so correlator and
  `squared_harmonics` saw inconsistent inputs): phase B built as
  `MasterPatternHarmonics(mph.alm * default_rng(42).choice([-1.0, 1.0],
  mph.alm.shape), phase=Phase("scrambled", point_group="1"))` (every
  `|a_lm|` preserved exactly -- genuinely the same power spectrum, a
  different function; lower-triangle zeros and the real `m = 0` row
  preserved by construction; round-trips at 2.8e-10; flags `(1, False)`):
  `s.spherical_indexing([mph, scr], det, n_best=2, verbose=0)` gives
  `phase_id == 0` at **all nine points** (measured 9/9),
  `nbest_phase_id[:, 1] == 1` everywhere, `scores[:, 0] - scores[:, 1] >
  0.1` everywhere (measured **0.2970-0.4151**; the decoy's own scores
  0.1993-0.2194, recorded), and the phase-0 rows equal the
  single-phase run's results bitwise. **`xmap.phases.names == ["ni"]`**
  (revision fix, blocker: orix *deletes* phases whose id never appears in
  `phase_id`, so asserting `["ni", "scrambled"]` cannot pass; the full
  configured list is pinned instead via `[p.phase.name for p in
  indexer.phases] == ["ni", "scrambled"]` and the `nbest_phase_id`
  column).
- **Rotated-copy degeneracy control** (the D3 finding): phase B =
  `rotate_harmonics(mph.alm, zyz_b)` with `zyz_b = (0.9, 0.7, -0.4)`,
  flags `(1, False)`: `max |score_A - score_B| < 0.05` over the nine
  points (measured 0.0151 -- the copy is *not* asserted to lose), and the
  **composed-orientation identity**: with `O_A`/`O_B` the two phases'
  per-point best rotations (from a two-single-phase-indexer run or the
  `n_best=2` rows), `Orientation((rotation_from_zyz(zyz_b) * O_B).data,
  Oh).angle_with(Orientation(O_A.data, Oh))` is `< 2.5` deg for all nine
  (measured median 0.676 / max 1.074) while the three wrong compositions
  (`~rb * ob` sign flipped / order reversed) each have median `> 10` deg
  (measured 24.3-28.7) -- an independent pin of the rotation convention
  and the per-phase bookkeeping.

Per-pattern failure injection (D2)
- A copy of the signal with pattern `[1, 1]` replaced by `np.full((60, 60),
  37, np.uint8)`: the default call gives, at that point, **`is_indexed ==
  False`, `phase_id == -1`, the identity rotation, `scores == 0.0`,
  `iq == 0.0`**, and at the other eight points results **bitwise equal** to
  the clean run; the same with `np.zeros`. The rationale measurements are
  `record_property` notes in the test: through EMSphInx's own path the
  constant-37 pattern would index with score ~0.2301 off AHE ripple
  (`ptp(processed) = 8.5e-14`) -- the genuine deviation -- while a
  constant float with `n_regions=0` correlates the window mask at score
  ~-2.64, which EMSphInx itself then drops under the positive-score
  insertion rule (revision correction: that case is parity, failed
  earlier, not a deviation -- D2).
- **Guard (b) is reachable only synthetically** (revision finding: every
  constant raw input is caught by guard (a) first, so (b) was dead code
  under the coverage gate): with `_preprocess_pattern` monkeypatched to
  return `np.full((60, 60), 1.0)`, the default call marks the pattern
  failed with the exact fill (the same assertions as above) -- reaching
  guard (b) itself; the plan 4.1 list gains "guard (b) removed".
- Direct `index_patterns` on a stack containing the flat pattern: the
  returned dict rows carry the same fill; `"zyz"` rows are exactly zero.
- A NaN pixel is documented as *not* guarded: `index_patterns` on a
  float64 pattern with one NaN completes without raising (measured score
  0.605); no assertion on its result beyond `no exception` (platform
  dependent; the non-finite *score* path (c) is exercised instead by a
  monkeypatched correlator returning `inf` -> the row fails).

Masks (D5)
- **`navigation_mask` polarity**: a mask with exactly three `False`
  entries -> `xmap.is_in_data.sum() == 3`, the three indexed points equal
  the corresponding rows of the unmasked run bitwise, the masked points
  have `is_in_data == False` with identity/0/-1/0 fills, and
  **`xmap.phases.names[0] == "not_indexed"`** (revision fix, blocker
  follow-on: the `-1` fills of masked-out points put `not_indexed` into
  `xmap.phases` even though those points are out of data -- probed on
  orix 0.14.2; asserting `== ["ni"]` here cannot pass); the info message
  says `"3 pattern(s)"`. All-`True` -> `ValueError` ("at least one value
  equal to `False`", the DI message).
- **`signal_mask`**: with `signal_mask[20:32, 25:40] = True` (the Phase 5
  `rDen` block): the default-call orientations still meet the D7 bounds
  (measured 0.496 / 0.683), and the masked run's values are **pinned**
  (revision fix, major finding: the previous "at least one score differs
  by > 1 %" plus "projector.signal_mask is set" pair was satisfied by a
  polarity-flipped `good_pixels` mutant, making the inversion
  `good = ~signal_mask & _circular_mask(shape)` untested):
  `scores.min() == approx(0.4461)`, `.max() == approx(0.5762)`,
  `.mean() == approx(0.5307)`, IQ `.min() == approx(0.1740)`, `.max() ==
  approx(0.2028)` (all `rel=0.05`). Which pin kills which mutant,
  measured: the **not-forwarded** mutant (unmasked preprocessing) has
  scores 0.4963-0.6239 mean 0.5701 -- 8-11 % off every score pin -- and
  dies there; the **polarity-flipped** mutant (`good = signal_mask`)
  happens to land *inside* the score pins (measured 0.4634-0.5546, mean
  0.5220) but its IQ is 0.2866-0.3199, ~60 % off the IQ pins, and dies
  there -- so both die on values, and the IQ pin is load-bearing, not
  decorative;
  `indexer.projector.signal_mask` is set (the mask reached the projector,
  not only the histograms). Wrong shape/dtype -> the projector's
  `ValueError` through the constructor.

Lazy signal, chunking, determinism (D4)
- `s.as_lazy().spherical_indexing(...)` equals the eager call **bitwise**
  (rotations data, scores, iq, phase_id).
- `index_patterns(pats, chunksize=1)`, `(..., chunksize=4)` and
  `(..., chunksize=9)` are pairwise bitwise identical (measured True); the
  same under `dask.config.set(num_workers=1)` vs `(num_workers=4)`
  (measured True) -- kills stale-buffer, shared-scratch and chunk-offset
  mutants.
- `chunksize` honoured: `get_info_message(9, 2)` contains `"5 chunk(s)"`;
  the default `chunksize=None` on the 9-pattern map with `num_workers=4`
  resolves to 1 (9 chunks, the `nt^2` rule) -- asserted through
  `_batch_estimate` and the message text with an explicit worker count
  (never through the machine's real CPU count).
- **Graph metadata is truthful** (D4 revision addition): the module-level
  `_map_chunks(patterns_da, indexer, n_best)` (the factored-out
  `map_blocks` call, plan 1.4) returns a dask array with `res.shape ==
  (n, n_best, 6)` and `res.chunks[0] == patterns_da.chunks[0]` *before*
  `compute()` -- pins the explicit `chunks=` argument (measured without
  it: declared shape `(9, 1, 1)`, a lie the eager path never noticed).

Dtype paths (D5)
- The signal cast to `float32` (same values) and to `float64` give maps
  bitwise equal to the uint8 run (measured: identical processed patterns);
  dtype of the outputs is independent of the input dtype (float64 scores,
  int32 `nbest_phase_id`).

Verbose and info message (D6)
- `verbose=0`: capsys records **no output** (message, speed line and
  progress bar all silenced). `verbose=1`: stdout contains `"Spherical
  indexing information:"`, `"ni"`, `"68"`, `"pattern(s)"`, `"chunk"` and a
  final `"Indexing speed:"` line; `get_info_message` itself contains
  `"Estimated memory per worker: 49 MB"` (revision fix: the substring now
  includes the number, pinning that the line prints the D8 *model* --
  the first draft's template said 45 MB, the measured peak, and no test
  could catch the divergence).

Performance and memory (D8; the only timing assertions)
- **The hard floor**: single-thread (`num_workers=1`), warm, the nine
  patterns through `index_patterns` -- `patterns_per_second >= 2 * 1` at
  `bw` 68 (measured 77.6 pat/s/core, 39x margin). Everything else
  `record_property`: per-stage ms at `bw` 53/68 (88 weekly), pat/s, the
  4-worker throughput, `tracemalloc` resident/peak of one chunk kit at
  `bw` 63 and 68 (measured 23.8 / 35.7 and 30.0 / 44.9 MB; `bw` 88 and
  113 weekly -- with 63 and 113 this completes the constitution's
  {63, 68, 88, 113} measurement set), and the model agreement
  `0.5 < memory_per_worker_bytes / measured_peak < 2`. (Corrected
  2026-09-02: the *peak* is taken around one `_index_chunk` call, but
  `tracemalloc`'s `current` around that call is **not** the resident
  cost and reads 0.0 MB -- the kit is local to the call and only the
  packed `(nc, n_best, 6)` result survives it -- so the resident row is
  measured with a kit of the documented composition, one correlator
  clone per phase plus the zeroed north/south pair, held alive; the
  measurement is wrapped in `try`/`finally` so a raise cannot leave
  `tracemalloc` running for the rest of the session.)
- Loose bound: the per-chunk peak `< 200 MB` at `bw` 68 (measured 44.9).
- **Model, `normalize` factor** (D8 revision fix): a two-phase indexer
  with `normalize=False` has `memory_per_worker_bytes` equal to the
  single-phase value (one shared scratch correlator), while with
  `normalize=True` it is larger by `slP^2 x bwP x 24` exactly.
- **The 2 GiB warning has a named test** (revision finding: specified,
  never asserted): under `dask.config.set(num_workers=64)` a `bw` 68
  `index_patterns` call warns -- `pytest.warns(UserWarning,
  match="2 GiB")` (64 x 49.4 MB = 3.16 GB > 2 GiB; the threads still
  index the nine patterns fine, so the test is cheap and needs no
  monkeypatch).
- Benchmark job: `test_spherical_indexing` (D8) asserts
  `np.isclose(xmap.scores.mean(), 0.570, atol=0.03)` and the map-level
  floor `9 / mean_time >= 2`.

`nickel_ebsd_large` (D7; skips without the `tests` extra)
- **Default suite, 20-point subset** (`[::15, ::15]` kept via
  `navigation_mask`, `pc_average`, backgrounds removed, `bw` 68):
  misorientation to the stored `xmap` **median < 1.5 deg, max < 3.0 deg**
  (measured 0.499 / 1.350); scores in `[0.40, 0.70]` recorded (measured
  0.4602-0.6506); all values `record_property`.
- **Weekly, 165-point subset** (`[::5, ::5]`): **median < 1.5, p95 < 2.5,
  max < 3.5 deg** (measured 0.530 / 1.082 / 1.495; zero points above
  2 deg recorded).

Exports and docs (D9)
- `kp.indexing.SphericalIndexer`, `kp.indexing.SphericalBackProjector`,
  `kp.indexing.fast_bandwidths` resolve through the lazy loader and are in
  the sorted `__all__`; `list(fast_bandwidths(16, 128)) == [17, 18, 20,
  23, 25, 28, 32, 33, 38, 39, 41, 46, 50, 53, 59, 61, 63, 68, 72, 74, 83,
  85, 88, 95, 98, 113, 116, 122, 123]` (pinned; contains `nml.hpp`'s
  suggested 53, 63, 68, 74, 88, 95, 113, 122, 123);
  `fast_bandwidths.__doc__` contains an `Examples` section and a
  `See Also` naming `SphericalIndexer`, and does **not** contain
  `":func:`fast_size`"` (revision fix: the old assertion "no longer says
  private" was a tautology -- the docstring never said private, verified
  by grep; the caveat lives in `specs/roadmap.md:30` -- and the
  `fast_size` cross-reference would dangle in the public reference).
- No public docstring in `_indexer.py`, `_back_projection.py`, `_fft.py`
  or the `spherical_indexing` method contains `"Phase 5"`, `"Phase 6"` or
  `"Phase 7"` (grep-style test over the rendered `__doc__`s; revision
  decision 6.14 -- roadmap phase numbers are project-internal).
- Kernel-flag regression (inherited): exactly two `error_model="numpy"`
  kernels in `kikuchipy.indexing._spherical` (`_interpolate_maxima`,
  `_fit_gaussian_1d_kernel`); `_indexer.py` defines **no**
  `CPUDispatcher` members and never imports `scipy.fft` (`"scipy.fft"
  not in inspect.getsource(_indexer)`).

## Weekly
- `uv run pytest --weekly tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py -n 4`:
  the 165-point `nickel_ebsd_large` subset, the `bw` 88 small-map recorded
  row, the `bw` 88 timing/memory rows, and the `bw` 113 memory row
  (measured for this spec: kit resident 91.9 / after-correlate 137.7 /
  peak 207.2 MB, model 228.4 MB -- completes the constitution's
  {63, 68, 88, 113} tracemalloc set together with the default suite's
  63/68 rows).

## Manual
- Headers: `_indexer.py` carries the kikuchipy GPL header + the delimited
  EMSphInx notice (CMU/Lenthe, GPL-2.0-or-later conveyed under
  GPL-3.0-or-later, "changed by Johan Westraadt, 2026-09") listing
  `idx/indexer.hpp` (`Result` `:54-64`, construction `:163-181`,
  `BatchEstimate` `:189-205`, `indexImage` `:216-270` with the
  pseudo-symmetry loop `:243-261` marked Phase 8, `computeHarmonics`
  `:312-318`, `correlate` `:326-331`; `refineImage` marked Phase 7),
  `modality/ebsd/idx.hpp` (`initialize` wiring `:252-296`, `ebsdWorkItem`
  `:382-456`; HDF5/PNG/vendor output, ROI mask, `ThreadedIqCalc` not
  ported) and `idx/base.hpp` (interfaces collapsed); `signals/ebsd.py`,
  `__init__.pyi`, the test and benchmark modules carry the kikuchipy
  header only.
- CHANGELOG: the two `Added` entries verbatim from D9 with the pinned
  `#8` fork-PR link, format per the constitution.
- Docstrings freeze: the un-normalised-scores note (constitution wording)
  on `EBSD.spherical_indexing` and `SphericalIndexer`; the failure
  semantics with the two measured garbage scores (and the corrected
  deviation bookkeeping of D2: only the AHE-ripple case deviates); the
  one-candidate-per-phase `n_best` rule and the positive-score insertion
  rule; the single-PC recipe; the memory table (9.4/15.9/20/43/92 MB
  resident, 21/36/45/98/207 MB peak at `bw` 53/63/68/88/113, the
  `memory_per_worker_bytes` model with the `normalize` factor); the
  coarse-only note in phase-free wording ("refinement of the coarse
  orientation is not implemented yet" -- no roadmap phase numbers in any
  public docstring, revision decision 6.14); the
  `IndexEBSD` defaults equivalence and the `emsphinx_compatible=True`
  requirement **in both `from_master_pattern` and the indexer** for
  EMSphInx parity; `:cite:lenthe2019spherical` renders (now enforced by
  the automated sphinx-build command); the D6 message template with
  "49 MB"; the `resize`-vs-direct note with all three metrics (2.9 %
  max-abs / 10.3 % rel-L2 / ~100 % per-coefficient); the rotated-copy
  degeneracy note in the multi-phase docs; `xmap.phases` holding only
  phases that won at least one point (losing phases live in
  `nbest_phase_id` and `SphericalIndexer.phases`).
- `specs/roadmap.md` Phase 6 boxes tick only with the measured numbers
  filled in; the constitution amendments of plan 0.1-0.7 applied.
- Coverage of `_indexer.py` and the `spherical_indexing` method `>= 95 %`
  (target 100 %; the exception-catch arm is reachable via the
  monkeypatched-correlator test, guard (b) via the
  monkeypatched-`_preprocess_pattern` test, and the 2 GiB warning via the
  `num_workers=64` test -- no specified branch is untestable, revision
  fix).
- Adversarial review findings addressed or explicitly deferred with
  reason; the bug-injection list of plan 4.1 fully killed.
- Known limitations stated: coarse only (refinement Phase 7); one PC per
  call; `azimuthal`/`twist` raise; one candidate per phase (`n_best`
  saturates at `n_phases` until Phase 8); scores un-normalised and
  geometry-bound; a rotated copy of the same structure is indistinguishable
  by design; threaded scaling ~2x at 4 workers on this machine (recorded);
  zero-variance patterns are failed, deviating from EMSphInx's
  garbage-score behaviour (measured, documented).

## Definition of done
All Phase 6 boxes in `specs/roadmap.md` ticked, default suite green on
Windows (this machine) with `-n 4` after a `-n 0` warm-up, weekly run once
locally and green, benchmark run once locally, `sphinx-build -b html` and
`-b linkcheck` exit 0 with the three new names rendered, PR opened into
fork `develop`; the determination results below re-measured with the real
implementation and appended as a further dated block. "PR merged" is
tracked in the roadmap.

## Recorded results

### 2026-09-02 -- pre-implementation reference measurements (spec drafting)

Environment: this machine (Windows 11), Python 3.13, numpy 2.4.6, scipy
1.17.1, numba 0.65.1, orix 0.14.2, kikuchipy 0.14.dev0 (`develop` after the
Phase 5 merge, 9aa3c5a1). Probes (scratchpad, not committed):
`p6_probe.py` (small-map accuracy/scores/IQ for `bw` 53/68/88 x
`n_regions` 10/0, per-stage timings, per-thread-kit tracemalloc,
multi-phase rotated + scrambled, flat/NaN patterns, `_batch_estimate`
values), `p6_probe2.py` (composed-orientation direction, dask
`map_blocks` prototype -- determinism and scaling, `nickel_ebsd_large`
20-point subset), `p6_probe3.py` (165-point weekly subset,
`fast_bandwidths`, phase metadata), `p6_probe4.py` (score/IQ/misorientation
means, `resize`-vs-direct, wrong-bandwidth guard). The pipeline per
pattern: `_preprocess_pattern` -> zeroed `out=` pair -> `unproject(...,
return_image_quality=True)` -> `sht.analyze` -> `correlate` ->
`rotation_from_zyz`; misorientations vs the stored `xmap`s via
`Orientation.angle_with` (m-3m).

- **Small map coarse vs stored `xmap`** (`pc_average`, backgrounds
  removed, defaults): `bw` 53: normalized median 0.747 / max 0.991 deg
  (per point 0.51 0.99 0.78 0.45 0.67 0.75 0.73 0.77 0.95), un-normalized
  0.738 / 0.993; `bw` 68: normalized **0.599 / 0.838** (0.35 0.75 0.60
  0.45 0.48 0.59 0.71 0.68 0.84), mean 0.6067, un-normalized 0.601 /
  0.836; `n_regions=0`: 0.605 / 0.852 (un-normalized 0.628 / 0.850);
  `bw` 88: 0.524 / 0.571 (`n_regions=0`: 0.504 / 0.554).
- **Scores** (normalized / un-normalized min-max): `bw` 53
  0.4518-0.5374 / 0.2185-0.2587; `bw` 68 **0.4963-0.6239** (mean
  **0.5701**) / **0.2799-0.3533**; `bw` 68 `n_regions=0` 0.5082-0.6412 /
  0.2867-0.3632; `bw` 88 0.5721-0.7134 / 0.3597-0.4325.
- **IQ of the processed patterns** (bandwidth independent): `n_regions=10`
  **0.1727-0.2036** (mean 0.1878; per point 0.2036 0.1727 0.1910 0.1942
  0.1792 0.1864 0.2008 0.1817 0.1806); `n_regions=0` (backgrounds-removed
  input, float64 cast) 0.2890-0.3269. Phase 5's raw-pattern IQ was
  0.766-0.779 -- three separable bands, used to kill IQ-source mutants.
- **Timings** (single core, warm, best of 5 sweeps, ms/pattern
  prep/unproject/analyze/correlate = total -> pat/s): `bw` 53:
  0.20/0.15/0.22/5.67 = 6.24 -> **160.3**; `bw` 68: 0.21-0.23/0.20-0.21/
  0.46/11.8-12.0 = 12.6-12.9 -> **77.6-79.1**; `bw` 88:
  0.22/0.20/0.95/29.8 = 31.2 -> **32.1**. Setup: `from_master_pattern`
  0.02-0.04 s, projector 0.02-0.03 s, normalized correlator 0.03-0.49 s
  (first-call JIT).
- **Per-worker chunk kit** (tracemalloc; clone + `(dim, dim)` pair):
  resident after clone 9.4 / 20.0 / 43.4 MB at `bw` 53 / 68 / 88; after
  one correlate (the `xc` cube) 14.2 / 30.0 / 65.1 MB; transient peak
  21.4 / 44.9 / 97.6 MB. Model `slP^2 bwP 24 + slP^3 8` bytes = 29.7 +
  19.7 = 49 MB at `bw` 68 (peak within 10 %).
- **`_batch_estimate`** (verbatim port): 34 / 15 / 6 at `bw` 53 / 68 / 88
  (model pps 56.4 / 25.1 / 10.9); `(68, 4, 9) -> 1`, `(68, 1, 9) -> 15`.
- **Dask `map_blocks` prototype** (`bw` 68): results bitwise identical
  (`np.array_equal` True) for chunksize 9/1-thread vs 1/4-threads vs
  4/4-threads; 72 patterns, chunk 3: 67.6 / 93.4 / 149.3 / 159.2 pat/s at
  1 / 2 / 4 / 8 workers (speedup 1.0 / 1.4 / 2.2 / 2.4 vs the same-run
  1-worker rate; 2.03x / 2.17x vs the 9-pattern single-thread reference).
- **Multi-phase**: rotated copy (`zyz_b = (0.9, 0.7, -0.4)`, flags
  `(1, False)`): true phase wins only **6/9**, score gaps **-0.0151 to
  +0.0090** -- degenerate, as the correlation peak is rotation invariant;
  phase-scrambled copy (seed 42, uniform phases on `alm`): wins **9/9**,
  gaps **0.2642-0.4080**. Composed orientation: `~Rotation(
  zyz_to_quaternion(zyz_b)) * O_B` (= `rotation_from_zyz(zyz_b) * O_B`)
  vs `O_A`: median **0.676** / max **1.074** deg; the wrong compositions
  `rb * ob` / `ob * ~rb` / `ob * rb`: medians 24.3 / 26.6 / 28.7 deg
  (maxima 24.4 / 59.7 / 44.0).
- **Failure semantics rationale**: constant uint8 37 through the default
  pipeline: AHE output `255 + O(1e-13)` (ptp `8.5e-14`), *not* the
  `unproject` mask branch; correlates the normalised ripple with `iq
  0.9999999999999996`, **score 0.2301**, garbage `zyz`. Constant float
  37.0 with `n_regions=0`: processed ptp exactly 0, the mask branch, `iq
  1.0`, window-mask correlation **score -2.6402**. Raw `np.ptp` of the
  constant uint8 and of zeros: exactly 0 (the guard quantity). A single
  NaN pixel in a float pattern: no exception, near-normal result (score
  0.605 vs 0.624 clean) -- `_to_uint8` swallows it; documented, not
  guarded.
- **Dtype**: float32 input with the same values -> processed pattern
  bitwise equal to the uint8 input's (max |diff| 0.0).
- **`nickel_ebsd_large`** (`pc_average`, backgrounds removed, `bw` 68;
  detector navigation shape (55, 75); window `n_points` 1314, rescaled
  (53, 53), `window_fraction` 0.145534): 20-point subset (`[::15, ::15]`):
  median **0.499** / p90 1.075 / max **1.350** deg (per point: 0.59 0.33
  1.04 0.88 0.31 0.03 1.06 0.72 0.37 0.25 0.62 0.29 1.22 1.35 0.31 0.41
  0.49 0.76 0.50 0.50), scores 0.4602-0.6506; 165-point subset
  (`[::5, ::5]`): median **0.530** / p90 0.931 / p95 **1.082** / max
  **1.495** deg, zero points above 2 deg, scores 0.4464-0.6506.
- **`fast_bandwidths(16, 128)`** = [17, 18, 20, 23, 25, 28, 32, 33, 38,
  39, 41, 46, 50, 53, 59, 61, 63, 68, 72, 74, 83, 85, 88, 95, 98, 113,
  116, 122, 123] (int64), a superset of `nml.hpp`'s suggested values in
  range.
- **Phase metadata** (the Ni master harmonics): `phase.name == "ni"`,
  space group Fm-3m, point group m-3m; `beam_energy 20.1`, `sample_tilt
  70.0` (the shared-geometry check quantities).
- **`resize` vs direct**: `from_master_pattern(bw=120).resize(68)` differs
  from `from_master_pattern(bw=68)` by up to **2.9 % relative** in `alm`
  (bandwidth-dependent weighted normalisation); flags unchanged (4, True).
  `squared_harmonics` with a (120, 120) `alm` raises `ValueError` naming
  the projector's shape and `resize`.

### 2026-09-02 -- revision measurements (adversarial spec review; same environment, probe `p6_rev_probe.py`/`p6_rev_probe2.py`, scratchpad, not committed)

The baseline reproduced digit for digit before anything below was trusted
(median 0.5987 / max 0.8379 deg, scores 0.4963-0.6239 mean 0.5701, IQ
0.1727-0.2036). New and corrected numbers:

- **Tilt binding (new D1 guard)**: harmonics at `sample_tilt` 70.0 vs
  `det.sample_tilt = 65.0`: median **4.680** / max **5.099** deg from the
  stored xmap, normalized scores **0.5053-0.6474** (mean 0.5788) --
  *higher* than the correct run's, so no score-based check can catch a
  dropped binding. IQ unchanged (preprocessing sees no geometry).
- **Sign-scrambled decoy (replaces the phase scramble)**:
  `mph.alm * default_rng(42).choice([-1.0, 1.0], mph.alm.shape)`
  preserves every `|a_lm|` (checked) and the real `m = 0` row (max
  `|Im| = 0.0`); true phase wins **9/9**, gaps **0.2970-0.4151**, decoy
  scores 0.1993-0.2194. The old phase scramble put up to 3.12 of
  imaginary part into the `m = 0` row and round-tripped
  `analyze(synthesize(.))` at 0.99 relative error (reviewer-measured,
  accepted) -- not a real spherical function; its 9/9 with gaps
  0.2642-0.4080 stands as history above but is no longer the fixture.
- **`signal_mask[20:32, 25:40]` run (now pinned)**: misorientation median
  0.4958 / max 0.6828 deg; scores **0.4461-0.5762** mean **0.5307**; IQ
  **0.1740-0.2028**; max relative score change vs unmasked 10.1 %.
  Mutant split: preprocessing-unmasked mutant -> scores 0.4963-0.6239
  (8-11 % off the pins, dies on scores); `good = signal_mask`
  polarity-flip mutant -> scores 0.4634-0.5546 mean 0.5220 (*inside* the
  score pins) but IQ **0.2866-0.3199** (~60 % off -- dies on IQ).
- **`circular_mask=True` run (now pinned)**: `n_points` **1117**;
  misorientation 0.6036 / 0.8562 deg; scores **0.4915-0.6390**; IQ
  **0.1920-0.2224**.
- **`gaussian_background=True` runs (now pinned)**: compat=True --
  misorientation 0.5943 / 0.8158, scores **0.4942-0.6101**, IQ
  0.1873-0.2159; compat=False -- 0.5911 / 0.8169, scores 0.4960-0.6109;
  max per-point |score difference| **1.83e-3**, max |IQ difference|
  6.7e-4 (real, sub-tolerance -> non-identity assertion + kwarg spy).
- **Memory chunk kit** (tracemalloc, clone + zeroed pair, warmed):
  `bw` 63: resident 15.9 / after-correlate 23.8 / peak **35.7** MB
  (model 39.2); `bw` 68 re-measured 20.0 / 29.9 / 44.8 (calibration,
  matches the drafting block); `bw` 113: 91.9 / 137.7 / peak **207.2**
  MB (model **228.4**). Model arithmetic at `bw` 68: 135^2 x 68 x 24 +
  135^3 x 8 = 29,743,200 + 19,683,000 = **49,426,200 B -> "49 MB"** (the
  info-message line); at `bw` 113 x 8 workers: 1.83 GB = 1.70 GiB
  (model) / 1.66 GB (measured peak) -- under 2 GiB (2.15 GB).
- **`clone()` cost** (reviewer-measured, accepted): 0.18 / 0.21 / 0.33 ms
  at `bw` 53 / 68 / 88 vs warm full construction 0.046-0.096 s at `bw`
  68 -- the D4 per-chunk-clone justification now quotes the right number.
- **`resize` vs direct, metrics stated**: max|d|/max|a| **2.88 %**,
  relative L2 **10.28 %**, per-coefficient max over the 187 coefficients
  with `|a| > 1 %` of max: **102 %**; `a[0,0]` -3.15804 -> -3.07553.
- **Re-zero equivalence** (reviewer-measured, accepted): zeroing the
  north/south pair once per chunk vs once per pattern gives
  `np.array_equal` True on the stacked `gln` over the nine patterns --
  per-pattern re-zeroing is defensive, its mutant a no-op.
- **`map_blocks` metadata** (reviewer-measured, accepted): without
  `chunks=` the result declares shape `(9, 1, 1)`; `compute()` is right
  regardless; `chunks=(patterns.chunks[0], n_best, 6)` fixes the
  declaration.
- **`_batch_estimate(68, 4, 0)`**: verbatim arithmetic gives 0
  (`ceil(0/16)`); the `max(1, ...)` clamp is a recorded deviation.
- **orix phase-list behaviour** (reviewer-probed, accepted; orix 0.14.2):
  `CrystalMap.__init__` deletes phase-list entries whose id never occurs
  in `phase_id`; `-1` fills (including masked-out `is_in_data` points)
  prepend `not_indexed`.
- **`ebsd.py:1932-1944` re-read**: DI's navigation-mask checks are shape
  / all-`True` / is-ndarray -- **no dtype check exists**, so the boolean
  check is a new frozen message of this phase, not a mirror.
- **Constant-float window-mask score** re-confirmed at **-2.6402**; under
  the C++ insertion rule (`corr <= 0` never inserted) EMSphInx reports
  that point not-indexed -- the D2 deviation bookkeeping was corrected
  accordingly (only the AHE-ripple +0.2301 case deviates).

### 2026-09-02 -- test-quality review measurements (same environment, probe `p6_fix_probe.py`, scratchpad, not committed)

Measured while applying the test-quality critic's findings to the two
test files; each line is the evidence for one of them.

- **Navigation-mask check order**: `np.ones((3, 3), int).all()` is
  `True` and `~np.ones((3, 3), int)` is `-2` everywhere (truthy). An
  integer mask of ones therefore hits the all-`True` branch first under
  the listing order of `requirements.md`, raising DI's "at least one
  value equal to `False`" instead of the frozen boolean-dtype message;
  the order **is-ndarray, dtype, shape, all-`True`** satisfies all four
  mask tests (a list has no `.dtype`, hence is-ndarray first). The
  requirements sentence is corrected in place and the constraint is
  pinned by a comment on `test_a_non_boolean_navigation_mask_is_refused`.
- **`CrystalMap.phase_id` dtype**: orix 0.14.2's `CrystalMap.__init__`
  does `phase_id = phase_id.astype(int)`, so an `int32` input comes back
  as the platform's default integer (measured `int64` for both `int32`
  and `int64` inputs). The `int32` pin of the D5 contract therefore
  lives on the `nbest_phase_id` prop and on the `index_patterns` result,
  never on `xmap.phase_id`, which the dtype test now pins only to an
  integer kind (it previously compared the implementation to itself).
- **Phase-less harmonics warning**: `MasterPatternHarmonics(mph.alm)`
  emits **nothing** under `simplefilter("always")` (empty record list),
  so the `catch_warnings`/`simplefilter("ignore")` wrapper around it in
  `test_repr_of_a_phase_less_harmonics` was dead scaffolding hiding a
  future regression, and is dropped (the sibling test of the signal file
  never had one).
- **A `bw` 512 kit**: `fast_size(2 x 512 - 1) = 1024`, `bwP = 513` ->
  correlator cubes `1024^2 x 513 x 24` = **12.910 GB**, interpolation
  cube `1024^3 x 8` = **8.590 GB** (model **21.500 GB**), Wigner table
  `512^3 x 8` = **1.074 GB**. For comparison at `bw` 68: `slP` 135,
  `bwP` 68, 0.030 + 0.020 = 0.049 GB, Wigner 0.003 GB. This is why the
  boundary is asserted with the projector sentinel rather than built.
- **`tracemalloc` around a call which frees its kit**: a synthetic
  function allocating 120 MB and returning a six-element array measures
  `current` **0.000 MB** / `peak` 120.0 MB -- the mechanism behind the
  corrected D8 memory row, where `current` after `_index_chunk` returns
  would have recorded "0.0 MB" instead of the resident cost.
- **Model over measured peak**: 1.0888 / 1.0980 / 1.1002 / 1.1025 /
  1.1023 at `bw` 53 / 63 / 68 / 88 / 113, i.e. **+8.9 to +10.2 %**. The
  module and `memory_per_worker_bytes` docstrings said "10 to 25 %",
  wrong at both ends, and now say "about 10 %".
- **Tests added by the review**: the exception arm (failure case (d)),
  injected by a `_preprocess_pattern` which raises for one pattern keyed
  on that pattern's own bytes (order and chunking independent; the nine
  patterns are pairwise distinct, asserted in the test) -- without it
  the "exception propagating" mutant of plan 4.1 survived; the `(9,)`
  `iq` shape and its bitwise equality with the `n_best=1` run, which
  kills the mis-packed iq column mutant; the per-stage ms/pattern and
  pat/s sweep at `bw` 53/68 (88 weekly) and the 4-worker throughput
  record, which the D8 determination list required and only the single
  `bw` 68 per-core row covered.

### 2026-09-02 -- implementation run (the real `_indexer.py` and `EBSD.spherical_indexing`)

Same environment as the two blocks above (Windows 11, Python 3.13,
numpy 2.4.6, scipy 1.17.1, numba 0.65.1, orix 0.14.2, kikuchipy
0.14.dev0). Every number below is a `record_property` of the two test
modules, read out of a `--junitxml` run of the default suite (`-n 0`)
and of `--weekly -n 4`; the pipeline is now the shipped
`SphericalIndexer.index_patterns` / `EBSD.spherical_indexing` rather
than the drafting probe. **Every pinned band of the two blocks above
reproduces digit for digit**, which is the headline determination:
the merged Phase 1-5 probe and the implementation are the same
arithmetic.

Gates (all green):

```
uv run pytest tests/test_indexing/test_spherical_indexer.py tests/test_signals/test_ebsd_spherical_indexing.py -n 0   ->  118 passed, 5 skipped (16.5 s)
uv run pytest ... -n 4                                                ->  118 passed, 5 skipped (15.3 s)
uv run pytest --weekly ... -n 4                                       ->  123 passed (18.6 s)
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical     ->  14 passed (0.7 s)
uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py -> 1 passed
uv run pytest --cov=kikuchipy.indexing._spherical._indexer ...        ->  _indexer.py 229 statements, 100.00 %
uv run pytest tests/test_indexing tests/test_io tests/test_signals -k "spherical or sht or emsphinx or SphericalHarmonics" -n 4 -> 2403 passed, 708 skipped
uv run pytest tests/test_signals/test_ebsd.py tests/test_indexing/test_dictionary_indexing.py -n 4 -> 187 passed
uv run sphinx-build -b html -D nbsphinx_execute=never doc doc/_build/html_check -> exit 0, no numpydoc validation warning for the three new names
uv run pre-commit run --files <the ten changed files>                 ->  passed
```

- **Accuracy, small map** (`bw` 68, default call): per point 0.354
  0.750 0.599 0.446 0.484 0.594 0.713 0.681 0.838 deg, median
  **0.5987**, max **0.8379** -- the drafting block's 0.599 / 0.838.
  `normalize=False`: median **0.6012** / max **0.8363**;
  `n_regions=0`: **0.6047** / **0.8524** with IQ **0.2890-0.3269**;
  `circular_mask=True`: **0.6036** / **0.8562**;
  `gaussian_background=True`: **0.5943** / **0.8158** with a maximum
  per point score difference between the two `emsphinx_compatible`
  settings of **1.826e-3**; `signal_mask[20:32, 25:40]`: **0.4958** /
  **0.6828** with scores **0.4461-0.5762** mean **0.5307**;
  `bw` 53: **0.7475** / **0.9915**; `bw` 88 (weekly): **0.5238** /
  **0.5708**.
- **Scores and IQ** (`bw` 68, default call): scores
  **0.4963-0.6239** mean **0.5701**, IQ **0.1727-0.2036** -- the D7
  pins, unchanged.
- **Multi-phase**: sign-scrambled decoy gaps **0.2970-0.4151** with
  decoy scores **0.1993-0.2194** (9/9 to the true phase); rotated
  copy gaps **-0.0151 to +0.0090**; composed orientation median
  **0.6765** / max **1.0736** deg against the three wrong
  compositions at **24.262 / 26.647 / 28.690** deg (medians).
- **`nickel_ebsd_large`**: 20-point subset median **0.4988** / max
  **1.3497** deg, scores **0.4602-0.6506**; weekly 165-point subset
  median **0.5302** / max **1.4947** deg, **zero** points above 2
  deg, scores **0.4464-0.6506**.
- **Throughput** (the hard floor's own measurement): **63.8**
  patterns/s/core at `bw` 68 single threaded in the `-n 0` run and
  **44.3** in the `-n 4` run (three other xdist workers competing),
  i.e. **22x to 32x** the `>= 2` floor; four workers **76.6**
  patterns/s. Per stage, milliseconds per pattern, best of three
  sweeps, `-n 0`: `bw` 53 preprocess 0.21 / unproject 0.16 / analyze
  0.23 / correlate **5.72** = 6.32 -> **158.2** pat/s; `bw` 68 0.21 /
  0.20 / 0.46 / **12.16** = 13.03 -> **76.7** pat/s; `bw` 88 (weekly,
  under `-n 4`) 0.28 / 0.27 / 1.29 / 38.66 = 40.50 -> 24.7 pat/s. The
  correlation is 90.5 / 93.3 / 95.5 % of the budget, as Phase 4
  predicted.
- **Memory of one chunk kit** (`tracemalloc`, resident after the
  clone / after one correlation / transient peak of one
  `_index_chunk` call / model, MB): `bw` 63 **15.9 / 23.8 / 35.8 /
  39.2**; `bw` 68 **20.0 / 29.9 / 45.0 / 49.4**; `bw` 88 (weekly)
  **43.4 / 64.9 / 97.6 / 107.6**; `bw` 113 (weekly) **91.9 / 137.7 /
  207.4 / 228.4**. The model over the peak is **1.10** at every
  bandwidth, i.e. the "about 10 %" of D8. The `bw` 68 peak is 4.4x
  under the loose 200 MB bound.
- **The 2 GiB warning fires where the model says it does**, and one
  consequence was observed and is recorded rather than changed: on
  this 20 core machine a `bw` 88 call warns, since 20 x 107.6 MB =
  2.15 GB = **2.004 GiB**, a hair over the threshold; at `bw` 68 it
  is 20 x 49.4 MB = 0.92 GiB and silent, which is what
  `test_verbose_zero_is_silent` needs.
- **Benchmark** (`pytest-benchmark`, 5 rounds): mean **129.1 ms**,
  min 126.5, max 131.9 for the whole `EBSD.spherical_indexing` call
  on the nine pattern map, i.e. **69.7** map level patterns/s
  including the per call indexer construction -- **35x** the map
  level floor of 2. `xmap.scores.mean()` 0.5701 against the
  benchmark's 0.570 +- 0.03.
- **Chunking**: `_batch_estimate(68, 20, 9)` (this machine's default
  worker count) gives 1, so the default `chunksize=None` runs the
  nine pattern map as nine chunks, as the `nt^2` rule intends.

Implementation notes worth recording:

- **`iq` is unpacked from the best row's column**, `packed[:, 0, 5]`,
  not from an average or a separate array: every inserted candidate
  carries the pattern's image quality (the C++ `r.iq = iq` before
  each insertion) and a row which no candidate reached keeps the fill
  `0`, so a pattern which fails any of the five ways reports `iq 0`
  without a second code path.
- **Guard (c) reads row 0 only** (`np.isfinite(rows[0, :4]).all()`),
  the winning candidate's three angles and score, as D2 states.
- **One test was added** to `tests/test_indexing/
  test_spherical_indexer.py`,
  `TestSphericalIndexerConstruction::
  test_a_signal_mask_and_the_circle_intersect`: no test of the
  written suite passed **both** a `signal_mask` and
  `circular_mask=True`, so the intersection branch of the
  `good_pixels` derivation (D1, "the inverted `signal_mask` alone or
  intersected when given") was the single uncovered line of
  `_indexer.py` and a mutant keeping only one of the two terms
  survived. With it, `_indexer.py` coverage is **100 %** and
  `EBSD.spherical_indexing`'s body is fully covered as well. No
  existing test was modified.

### 2026-09-02 -- test-strength review by bug injection (the committed suite against 105 mutants)

Method: `_indexer.py` and the `EBSD.spherical_indexing` block were
backed up to the OS temp directory and md5 verified; one mutation was
applied at a time as an exact-string edit asserted to match exactly
once, the two Phase 6 files were run with
`-n 0 -q -x --tb=no -p no:cacheprovider`, and the files were restored
and md5 verified before the next. 105 mutants: the 80 of the plan's
bug-injection list plus 25 of the reviewer's own, aimed at the
attribute table, the information message, the crystal map assembly and
the correlator wiring. Baseline before and after: **118 passed, 5
skipped** (15 s), md5 unchanged.

**Result: 92 of 105 killed by the suite as written, 48 distinct tests
firing.** The busiest killers are `TestMasks::
test_the_navigation_mask_polarity` (8), `TestIndexPatterns::
test_a_flat_pattern_in_a_stack_carries_the_fill` (6),
`TestNickelSmall::test_the_default_call_meets_the_coarse_bounds` (4),
`TestPreprocessingPaths::
test_emsphinx_compatible_changes_the_gaussian_background` (4),
`TestSphericalIndexerConstruction::
test_a_signal_mask_and_the_circle_intersect` (4), `TestBatchEstimate::
test_large_map_pins[53]` (4) and `TestMemoryModel::
test_many_workers_warn` (4). Every mutant the plan names is killed
except the three below.

**13 survivors, of which 12 were genuine gaps.** One was killed by a
gate outside the two files (the `repr` window fraction unscaled: the
class docstring's `1317 points (14.6 %)` is a real doctest under
`NORMALIZE_WHITESPACE`, so `--doctest-modules
src/kikuchipy/indexing/_spherical` catches it), one is provably
equivalent, and ten new tests close the rest:

| survivor | why the suite could not see it | new test |
| --- | --- | --- |
| insertion is `lower_bound`, not `upper_bound` (`>=` for `>`) | no two scores in the suite tie, and the negative-score test's `-1` is dropped by both | `test_a_score_which_ties_the_fill_is_never_recorded`, `test_an_equal_score_ranks_after_the_earlier_phase` |
| the top-n shift loop runs the wrong way | every multi-phase test has the winner already in slot 0, where the shift runs over identical fill rows | `test_a_later_phase_displaces_an_earlier_one` |
| guard (b) deleted (a constant processed pattern reaches the sphere) | the window mask it then correlates scores the measured -2.64, which the insertion rule drops, leaving exactly the same fill the guard would have left | `test_a_constant_processed_pattern_never_reaches_the_projector` |
| the insertion's drop bound off by one (`index > n_best`) | the out-of-range write is swallowed by the per-pattern `except` and the point is failed -- identical to the contract whenever the dropped candidate is the only one | `test_a_dropped_candidate_does_not_fail_the_pattern` |
| guard (c) widened to the whole result block | strictly more conservative; no suite input has a non-finite value outside row 0 columns 0-3 | `test_only_the_winning_row_is_checked_for_finiteness` |
| `index_patterns` ignores an explicit `chunksize` (three variants: the estimate re-run, a lazy input not rechunked, an eager input in one chunk) | results are bitwise identical across chunk sizes by design, and the information message keeps reporting the size which was asked for | `test_an_explicit_chunksize_reaches_the_graph[False/True]` |
| the signal method's detector-shape guard deleted | `index_patterns` catches the same mismatch and its message names both shapes too, after the construction has been paid for | `test_the_shape_is_refused_before_the_indexer_is_built` |
| the information message's phase description reduced to the bare name | only `Phase(s): ni` was pinned, not the point group and the two symmetry flags | `test_the_info_message_describes_the_phase` |
| the `signal_mask` and `circular_mask` attributes frozen at their defaults | `test_attributes` pins the *defaults*, where a frozen attribute is invisible | `test_the_attributes_follow_the_arguments` |

**The one equivalent mutant, with proof.** Writing the image quality
only on an insertion at index 0, instead of on every inserted
candidate as `r.iq = iq` does in the C++, is unobservable: column 5 of
a row at index greater than zero is never read (the chunk body reads
`rows[0, :4]`, and `index_patterns` reads `packed[:, 0, 5]`), and the
shift only ever moves a row *downwards*, so the row standing at index
0 is always the last one written at index 0, with its image quality.
No test can distinguish it, and no test was added.

Every new test was validated twice: it passes on the pristine
implementation, and on its own (`-k` filtered) it fails against its
target mutant. After the ten additions all 13 survivors are killed
except the equivalent one. Suite after: **129 passed, 5 skipped**
(`-n 0` 27.5 s, `-n 4` 20.2 s); `pre-commit run --files` on the two
test files passed. `_indexer.py` and `ebsd.py` are byte-identical to
the pre-review state (md5 verified); only the two test files changed,
by addition.

### 2026-09-02 -- fix pass over the three adversarial reviews

Eleven findings applied, four skipped with evidence. The changed
files are `_indexer.py`, `signals/ebsd.py`, `_back_projection.py` and
the two test files.

**Applied.** (1) The two split Sphinx role targets of
`EBSD.spherical_indexing`'s `Notes` were rejoined onto one line each
-- `PyXRefRole.process_link` overrides the base role's whitespace
collapse, so a newline inside a target never resolves and
`nitpicky = False` hides it; verified in the rebuilt HTML, where
`memory_per_worker_bytes` and `from_master_pattern` now carry an
enclosing `<a>`. Line length yields to the target, as `_indexer.py`
already does. (2) The five remaining private cross-references of the
newly public docstrings became prose, completing the D9 scrub:
`_batch_estimate` x2 (`get_info_message`, `index_patterns`) and, in
`SphericalBackProjector`, `_solid_angle_fraction`,
`_unproject_kernel` x2, `_dct_rescale` and `_dct_image_quality` --
the last one the reviewer missed. `_back_projection.py` is already in
the D9 pre-commit list for exactly this reason. (3) A systematic
failure is no longer silent: `index_patterns` counts
`packed[:, 0, 4] < 0` after the compute and warns
`"N of M pattern(s) could not be indexed ..."`. Never re-raising is
frozen (D2(d)), and a count is not a re-raise; measured before the
fix, a whole map failing returned with no diagnostic at all.
(4) A navigation shape which is not one- or two-dimensional is
refused up front. Measured: `s.inav[0, 0]` (nav `()`) fell through to
`create_coordinate_arrays((), ())`, which returns orix' default
`(5, 10)` = **50** coordinates for a one-point map, so the returned
`CrystalMap` raised `IndexError: ... size of axis is 50 but size of
corresponding boolean axis is 1` on `.shape`/`.x`/`.y`; a 3-D
navigation shape raised inside orix only after the whole map had been
indexed. **`EBSD.dictionary_indexing` has the identical 0-D defect**
(measured on the same signal) -- pre-existing and repo-wide, not
fixed here. (5) `harmonics_list`, not the argument, is handed to the
indexer: a one-shot iterator was exhausted by the `.phase`/unique-name
checks and the indexer then refused it as "empty". (6) Guard (a),
`ptp == 0` on the raw pattern, moved inside the per-pattern `try`, as
`ebsdWorkItem` (`idx.hpp:411-437`) wraps the whole of `indexImage()`:
a guard which raises must fail its own pattern, never the chunk.
(7) The coarse-orientation docstring now says `180 / side_length`,
the frozen D5/D1 quantity, instead of `180 / (2 * bandwidth - 1)`;
the two agree at every fast bandwidth (68 -> 135 -> 1.3333 deg) and
diverge elsewhere (bw 70: 1.2950 vs 1.2857). (8) `prop["scores"]`
before `prop["iq"]`, the D5 listing order and the DI/HI repr order.
(9) The provenance header now cites `Indexer<Real>` `:68-181` (D10's
enumeration; `struct Indexer` is at 68, the constructor 163-181),
`quat::mul` at **267** not 266 (`zyz2qu` is at 266), and records the
fill row's `iq = 0` as a deliberate improvement -- the C++ fill loop
(`:218-222`) never sets `iq` and `ebsdWorkItem` reuses one result
vector across a batch (`:406`), so a not-indexed point there inherits
the previous pattern's image quality. (10) A `Warns` section on
`EBSD.spherical_indexing` and (11) the extended one on
`index_patterns`.

**Skipped, with evidence.** (a) *The 2 GiB warning counting cores
rather than busy workers* (`min(n_workers, n_chunks)`): the formula
`n_workers x memory_per_worker_bytes` is frozen in D8, and the change
would break D8's own named test -- `test_many_workers_warn` runs nine
patterns on `num_workers=64`, where `_batch_estimate` gives chunksize
1, so `min(64, 9) = 9` and `9 x 49.4 MB = 444 MB` never reaches the
threshold. Changing it needs a spec amendment, not a fix pass. (b)
*Guard (c) inspecting row 0 only*: frozen D2 wording, and now pinned
by the bug-injection reviewer's
`test_only_the_winning_row_is_checked_for_finiteness`; widening it is
strictly more conservative but is a contract change. (c) *A
`signal_mask` which is not a NumPy array*: no polarity hazard exists,
because `SphericalBackProjector.__init__` (`:1186-1197`) already runs
`np.asarray` and then refuses a non-boolean dtype and a wrong shape,
so a list of bools converts with its polarity intact and a list of
ints raises "Signal mask of data type int64 must be boolean". D5's
frozen own-check list carries no `signal_mask` entry. (d) *The info
message printing before `n_best`/`chunksize` are validated*: purely
presentational -- the `ValueError` comes from the very next statement
-- and the fix duplicates two checks and their exact messages across
two files, a drift hazard for no behavioural gain.

**Eight tests added** (additive only, no existing test touched):
`test_a_failed_pattern_is_warned_about`,
`test_a_run_without_failures_is_silent`,
`test_the_guards_are_caught_per_pattern`,
`test_no_new_public_docstring_links_a_private_name` (the D9 scrub
regression, excluding `MasterPatternHarmonics`, whose four private
links predate this phase and are recorded below) in
`test_spherical_indexer.py`; and
`test_a_map_which_is_not_one_or_two_dimensional_is_refused`,
`test_a_one_dimensional_map_is_allowed`,
`test_the_navigation_shape_is_refused_before_the_indexer_is_built`,
`test_a_generator_of_harmonics_reaches_the_indexer` in
`test_ebsd_spherical_indexing.py`.

**Recorded, not fixed.** `MasterPatternHarmonics` (public since an
earlier phase, untouched here) carries four unresolved private
cross-references in its public docstrings:
`~..._sht_file.ShtFile.metadata_dict` and
`~..._symmetry.validate_flags` in the class docstring, and
`_check_atom_sum` and `_energy_weights` in `from_master_pattern`.
D9's scrub is scoped to `fast_bandwidths` and
`SphericalBackProjector`, so these are out of this phase.

**Gates after the fix pass.**

```
pytest <the two Phase 6 files> -n 0     ->  137 passed, 5 skipped (30.3 s)
pytest <the two Phase 6 files> -n 4     ->  137 passed, 5 skipped (21.2 s)
pytest --weekly <the two files> -n 4    ->  142 passed (16.3 s)
pytest --doctest-modules src/kikuchipy/indexing/_spherical -> 14 passed
pytest ... -k "spherical or sht or emsphinx or SphericalHarmonics" -n 4 -> 2423 passed, 708 skipped (100.6 s)
pytest tests/test_signals/test_ebsd.py tests/test_indexing/test_dictionary_indexing.py -n 4 -> 187 passed (143.7 s)
pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py -> 1 passed, mean 208.7 ms
pre-commit run --files <the ten files>  ->  passed
sphinx-build -b html -D nbsphinx_execute=never -> exit 0
```

Coverage: `_indexer.py` **232 statements, 0 missed, 100.00 %**;
`EBSD.spherical_indexing`'s body fully covered (`ebsd.py`'s missing
ranges skip from 1992 to 2486, spanning the whole method). The
benchmark's 208.7 ms mean is above the 129.1 ms recorded earlier
because a documentation build shared the machine; the map-level rate
is still 43.1 patterns/s against the floor of 2.

---

### 2026-09-02 -- `ubuntu-latest-py3.10-oldest` CI failure: root cause and fix

**Symptom.** PR #8 run `33629625164` failed only on
`ubuntu-latest-py3.10-oldest` (`dask==2021.8.1 diffsims==0.5.2
hyperspy==2.2 matplotlib==3.6 numba==0.57 numpy==1.23.0 orix==0.12.1
pooch==1.3.0 pyebsdindex==0.3.9.2 scikit-image==0.21.0`, resolved
`scipy 1.13.1`, Python 3.10.21), with **122 failed, 2912 passed, 244
rerun**. The failures were not confined to this phase: they spanned
`test_spherical_xcorr.py` (47), `test_spherical_sht.py` (26),
`test_ebsd_spherical_indexing.py` (16),
`test_spherical_master_pattern_harmonics.py` (11),
`test_spherical_grid.py` (9), `test_spherical_back_projection.py` (6),
`test_spherical_indexer.py` (5), `test_ebsd_master_pattern.py` (1) and
`test_spherical_wigner.py` (1), on all four xdist workers. Run
`33629630800`, the **same commit** with a **byte identical `pip list`**,
passed the same job.

**Root cause: one, not four.** All four reported failure classes are
downstream of a single fault -- `numpy.linalg.solve` returned a wrong
answer inside `_grid._ring_weights_skip`, so every quadrature weight
set in the process was wrong:

- The garbage solutions satisfied the **first** equation of the
  Chebyshev-Vandermonde system. That row is the all ones row, so
  `sum(w_hat) == 1` held to 4e-16 and the existing precision guard
  (`|sum(w_hat) - 1| > cbrt(eps) / 64`, `square_sht.hpp:1057`) never
  fired. Nothing else in the port could tell wrong weights from right
  ones.
- Diagnostic proof from the log: `analyze` of a constant function still
  gave `alm[0, 0] == sqrt(4 pi)` to 1e-10 (which needs only
  `sum(w_hat) == 1`) while every other coefficient was garbage
  (`372.8`, tolerance 1e-9). The dim 35 Legendre case printed the
  solver's output `[-0.748, 0.026, 0.004, 0.217, ...]` against the
  correct `leggauss` weights `[0.0066, 0.0153, 0.0239, ...]` -- so
  `leggauss` (hence `eigvalsh`, hence the Legendre nodes and the
  matrix `a`) was **correct** on that runner and only the solve was
  wrong.
- These systems are **well conditioned** (measured two-norm condition
  numbers 2.9, 3.1, 4.2, 12.6 and 4.0 for dim 19/35/33/65/401): no
  correct solver can miss them. Order <= 9 was unaffected (`[19]`
  passed, `[35]`, `[101]`, `[201]`, `[401]` failed), which points at a
  threading or kernel dispatch fault in that runner's BLAS rather than
  at conditioning.

Mapping the reported classes onto it: (a) the ~35 deg misorientation
medians are wrong weights, **not** orix -- `Orientation.dot` is already
symmetry reduced in orix 0.12.1 (verified: `angle_with` gives
`[50.0, 44.4, 36.2, 37.5]` where the unreduced angles are
`[62.8, 148.6, 36.2, 95.3]`), so the `misorientation` oracle in
`test_ebsd_spherical_indexing.py` is correct on both stacks and was
**left unchanged**; (b) the `np.isfinite` failures are `r_den` built
from wrong window harmonics; (c) the `DID NOT RAISE ValueError` cases
are the precision guard not tripping at lambert dim 401 / 361 / 259-301
because the wrong solutions still summed to one; (d) the `cell_deg`
comparisons are the cross-correlation peaks moving once the master and
window spectra are wrong.

**Not reproducible locally, so reproduced by construction.** The pinned
stack was rebuilt with `uv run --isolated --python 3.10 --with
"numpy==1.23.0" ... --with-editable . pytest ...`; every spherical test
passes there (1419 passed, 758 skipped with `-n 4 --reruns 2`), as does
the Phase 6 gate (137 passed, 5 skipped), on Windows and on the same
NumPy. The fault was therefore emulated with a pytest plugin replacing
`numpy.linalg.solve`, for orders >= 17 with an all ones first row, by a
vector satisfying only that first equation. On the **unfixed** tree
that reproduces the CI signature; measured separation of the check
introduced below, dim 35 legendre: healthy backward error **4.8e-16**,
emulated fault **7.6e-1**, `sum(w_hat) - 1` of the emulated fault
**-4.4e-16** (tolerance 9.5e-08, i.e. invisible to the old guard).

**Fix (production, `src/kikuchipy/indexing/_spherical/_grid.py`).**

- `_backward_error(a, b, x)`: the Oettli-Prager componentwise backward
  error `max_j |a x - b|_j / (|a| |x| + |b|)_j`, formed with element
  wise products and a sum reduction, never with `@` -- the check must
  not be dispatched to the library it is checking. It is of order
  `n * eps` for any backward stable solver however ill-conditioned `a`
  is, which is what lets it pass the legitimately ill-conditioned
  lambert grids whose **forward** error the existing sum guard rejects.
- `_solve_partial_pivot(a, b)`: Gaussian elimination with partial
  pivoting in pure NumPy element wise arithmetic, `numpy.argmax` and
  slicing -- no BLAS or LAPACK. `O(n ** 3)` in `n` vectorized row
  updates, immaterial at `n <= 384`.
- `_ring_weights_skip` now runs the **conditioning guard first**
  (unchanged text, unchanged behaviour, so
  `test_a_negative_weight_residual_also_raises`, the dim 401 raise and
  the dim 259-301 bracket are all preserved), and only then the
  **solver guard**: if the LAPACK solution's backward error exceeds
  `_SOLVE_BACKWARD_ERROR_TOLERANCE = 1e-8`, the set is recomputed
  without BLAS and re-guarded.
- The tolerance: a scan of every reachable `(layout, dim, skip)` up to
  dim 769 measures a worst healthy backward error of **1.43e-15** on
  NumPy 1.23.0 and **1.32e-15** on NumPy 2.4.6, so 1e-8 sits seven
  orders above any healthy value and seven below the observed fault.

**Fix (tests, `tests/test_indexing/test_spherical_grid.py`).** Two
named tests, in the Phase 1 module which owns the code:
`test_a_lapack_which_only_satisfies_the_first_equation_is_recovered`
(3 params) monkeypatches the fault and asserts the recovered sets match
the healthy ones -- measured relative deviations **2.3e-14**
(35, legendre), **3.8e-15** (65, lambert), **4.1e-14** (101, legendre),
three orders inside the asserted `1e-11` bound -- and that they are
still normalized weight sets;
`test_the_solver_guard_leaves_a_healthy_solve_untouched` asserts the
recovery is unreachable on a working LAPACK. **No test was weakened and
no test was skipped.** The `misorientation` oracle was deliberately not
touched.

**Production behaviour unchanged.** Weight sets from the pre-change and
post-change modules were compared for every `(layout, dim)` over
`dim` 3-129 odd plus 201, 259, 275, 301, 361, 401, 501 and 769:
**139 bitwise identical arrays and 5 identical `ValueError` messages**,
zero differences. The recovery is dead code on a healthy stack.

**Gates.**

```
oldest stack (numpy 1.23.0 / py3.10, pins as in tests.yml:48)
  pytest <the two Phase 6 files> -n 0                      -> 137 passed, 5 skipped (17.9 s)
  pytest tests/test_indexing/test_spherical_grid.py -n 0   -> 182 passed (32.0 s)
  pytest <the other seven spherical files> -n 4 --reruns 2 -> 1286 passed, 753 skipped (221.8 s)
  pytest -p broken_lapack <the two Phase 6 files> -n 0     -> 137 passed, 5 skipped (30.0 s)
modern stack
  pytest <the two Phase 6 files> -n 0                      -> 137 passed, 5 skipped (18.4 s)
  pytest <the two Phase 6 files> -n 4                      -> 137 passed, 5 skipped (18.7 s)
  pytest -p broken_lapack <all eight spherical files> -n 4 -> 1425 passed, 701 skipped (29.5 s)
  pytest --doctest-modules src/kikuchipy/indexing/_spherical -> 14 passed
  pre-commit run --files _grid.py test_spherical_grid.py   -> passed
```

`-p broken_lapack` is the emulation plugin described above; it lives in
a scratch directory and is not part of the repository.

**Residual risk, recorded not fixed.** The same runner fault would also
corrupt `numpy.polynomial.legendre.leggauss` (which calls
`numpy.linalg.eigvalsh`) and the two other BLAS calls in the port,
`normals @ matrix.T` in `_back_projection.py` and `np.tensordot` in
`MasterPatternHarmonics.resize`. The CI log shows all three were
correct in the failing job -- `leggauss` demonstrably so -- and none of
them is guarded here, because guarding an unobserved failure would be
speculation. The weight solve was singled out because it is the one
place where a wrong BLAS answer is **silently** cached into every
downstream spectrum.
