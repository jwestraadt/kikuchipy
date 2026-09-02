# Phase 7 -- `spherical-refinement`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_xcorr.py -n 0   # warm new kernel cache
uv run pytest tests/test_indexing tests/test_signals -k spherical -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pytest --benchmark-only benchmarks/indexing/test_spherical_indexing.py
uv run pre-commit run --files <changed files>
uv run sphinx-build -b html doc doc/_build/html
uv run sphinx-build -b linkcheck doc doc/_build/linkcheck
```

The first green run must be the full `-k spherical` selection above,
never just the three modules this phase edits directly: the break
surface includes `tests/test_indexing/test_spherical_back_projection.py`
(its `test_phase_four_keeps_its_single_error_model` loops the
`_xcorr` kernels and goes red when `_derivatives` lands -- plan 2.4
lists every such test).

(Corrected 2026-09-02, test-quality review: the four kernel-surface
tests of plan 2.4 were rewritten to be green **both** before and
after `_derivatives` lands, rather than left red across the
tests-first commit -- the back-projection loop and the `_xcorr`
parametrised flag test read a sanctioned-name set, the `_xcorr`
completeness check and the package-wide error-model test became
inclusions with the refinement's kernel listed separately, and the
equalities they gave up are asserted in
`test_spherical_refinement.py::TestRefinementKernels`
(`test_the_package_has_exactly_three_numpy_error_model_kernels` and
the new `test_the_xcorr_module_has_exactly_two_error_model_kernels`,
which is the home the back-projection assertion previously had
nowhere). The `-k spherical` first-green-run rule stands.)

`tests/test_indexing/test_spherical_xcorr.py` (new classes):

- `TestDerivatives` -- both oracle tests draw spectra with the
  frozen `random_alm` recipe (the C++ `randomPair` coefficients,
  uniform `[-1, 1]`; D2) and `record_property` the value and Hessian
  scales, since the absolute bounds are frozen for that recipe only
  (re-measured scales: `bw` 16 max `|value|` 19.4, `bw` 12 max
  `|hes|` 5.3e2).
  - `test_value_equals_the_inner_product_oracle`: `bw` 16,
    `(n_fold, mirror) in {(1,F), (2,T), (3,F), (4,T)}` x 6 seeded
    rotations, `der` both settings: `abs(value - <rotate_harmonics(
    flm, zyz), gln>) <= 1e-11` (measured worst 6.05e-14, rel
    1.81e-13).
  - `test_jacobian_and_hessian_equal_the_analytic_oracle`: `bw` 12,
    the `wigner_d`/`wigner_d_prime`/`wigner_d_prime2` triple sum
    (negative orders `(-1)^m conj`): value `abs <= 1e-11`, jac
    `abs <= 1e-10`, hes `abs <= 1e-9` (measured 5.51e-14 / 2.98e-13
    / 1.71e-12; corrected 2026-09-02: the fixture also evaluates
    every drawn triple at `zyz * (1, -1, 1)`, since
    `quaternion_to_zyz` returns `beta` in `[0, pi]` and a dropped
    negative-`beta` `csc` sign survived otherwise -- re-measured
    5.68e-14 / 3.06e-13 / 1.48e-12, bounds unchanged); values
    `record_property`.
  - `test_jacobian_and_hessian_match_finite_differences`: `bw` 16,
    central differences `h = 1e-5`: jac `abs <= 1e-5`, hes
    `abs <= 1e-2` (measured 2.97e-7 / 6.66e-4, truncation-limited;
    corrected 2026-09-02: with the negated-`beta` twins above,
    2.72e-7 / 6.65e-4).
  - `test_phase7_formulas_still_pin_the_kernel_inputs`: the copied
    `_phase7_derivatives` helper (per its Phase 3 docstring) agrees
    with `wigner_d_prime`/`wigner_d_prime2` through
    `wigner_d_table_pre` output at `bw` 15 to `1e-12` (the Phase 3
    assertion re-run against the shipped table builder this kernel
    calls).
  - `test_pole_evaluation_produces_the_nan_contract`: at
    `beta in {0.0, -0.0}` (`bw` 24, random spectra): `hes[1, 1]`
    NaN, `jac[1]` NaN, `jac[0]`, `jac[2]`, `hes[0, 0]`,
    `hes[2, 2]` finite; **no exception** (the `error_model` gate).
    At `beta = +-pi` the exact-NaN claim rides on the host libm
    returning `cos(+-pi) == -1.0` (D2; true here, and a
    one-ulp-below-`pi` *input* still yields -1.0 -- measured), so
    the assertion is `isnan(hes[1, 1]) or abs(hes[1, 1]) > 1e12`
    with the observed value `record_property`ed, and in every case
    `_refine_peak` from that start returns without raising
    (measured: 1 iteration through the 1x1 path at `0/-0.0`, 3 at
    `+-pi`).
  - `test_beta_is_wrapped`: value at `(a, b + 2 pi k, g)` equals the
    value at `(a, b, g)` to `rel 1e-12` for `k in {-1, 1, 2}`.
  - `test_py_func_matches_the_compiled_kernel`: value `rel 1e-12`,
    jac/hes `abs 1e-9` (not bitwise; FMA), under `np.errstate` with
    NumPy-scalar inputs, **including one pole evaluation** -- viable
    because D1 keeps the `csc` chain in `np.cos`/`np.sqrt` (a
    `math.*` twin raises `ZeroDivisionError` at the pole despite
    `np.errstate`; measured).
  - Kernel-flag regression: exactly **three** `error_model="numpy"`
    kernels project-wide (`_interpolate_maxima`,
    `_fit_gaussian_1d_kernel`, `_derivatives`); `cache`/`nogil` on,
    no `parallel`/`fastmath`. The Phase 4/5/6 tests that pinned
    "one in `_xcorr`" / "exactly two" are renamed and extended per
    plan 2.4 (incl.
    `test_spherical_back_projection.py::test_phase_four_keeps_its_single_error_model`,
    renamed 2026-09-02 to
    `test_the_correlator_module_keeps_its_sanctioned_error_models`;
    the `_xcorr` one to
    `test_only_the_sanctioned_kernels_use_the_numpy_error_model`
    and the indexer one to
    `test_the_package_error_models_are_the_sanctioned_ones`).
- `TestRefinePeak`
  - `test_an_on_grid_rotation_refines_in_one_iteration`: `bw` 16,
    on-grid pair: refined angle 0 (`<= 1e-9` deg), 1 iteration,
    `peak/power == 1` to `rel 1e-12`.
  - `test_symmetry_free_pairs_are_recovered`: `bw in {53, 54, 57,
    60, 63, 64, 68}` (88, 113, 123 weekly) x 3 seeded rotations,
    `randomPair` recipe: refined symmetry-free misorientation
    `< 4.92e-3` deg (the C++ `cbrt(float eps)` gate; measured worst
    2.958e-6), coarse and refined `record_property`, iterations
    `<= 4` (measured 2), zero failures.
  - `test_the_eight_point_groups_are_recovered`: groups `112, 11m,
    2/m, 3, 4, 4/m, 6, 6/m` x `bw in {53, 60, 63}` x 3: reduced
    misorientation `< 0.351` deg (the C++ gate; measured worst
    4.518e-6; corrected 2026-09-02: that is the drafting probe's
    `11 + bw` seed recipe -- under the shipped fixture's own
    `100 bw + index` seeds the worst is **9.04e-6**, still 38,800x
    inside the gate; the per-case table is in Recorded results).
  - `test_normalized_wedge_masked_pairs_are_recovered`: `testNCorr`
    recipe, `bw in {53, 68}` x `{1, 4/m}` x 3, gated **as the C++
    gates them** (D10): the `(n_fold=1, mirror=False)` cases
    `< 4.92e-2` deg (`epsN`, `sht_xcorr.cpp:316`; measured worst
    1.849e-2 -- 2.7x) and the `4/m` cases `< 0.351` deg
    (`sqrt(eps) * 5`, `:345`, the gate of the C++ normalised
    point-group loop `:373`; measured worst 2.133e-2 -- 16.5x, and
    the overall worst is a `4/m` case); deterministic seeds, every
    per-case value `record_property`ed; the refined normalised
    score above the coarse one in every case (measured 12/12).
  - `test_near_degenerate_targets_refine_under_a_tenth_degree`:
    targets `beta in {0, 1e-3, -1e-3, pi, pi - 1e-3}` at `bw` 24:
    refined `< 0.1` deg (measured 0.0 / 5.73e-2), coarse recorded
    (~0.9-1.2 deg -- the Phase 4 D5 defect zone).
  - `test_a_start_exactly_on_the_pole_uses_the_one_by_one_step`:
    start `(a, 0, g)` on-peak: converges (measured 1 iteration);
    the step's beta/gamma slots stay exactly 0.
  - `test_far_starts_fail_back_to_the_start`: 10 seeded unrelated
    starts at `bw` 24: every failed case returns the start bitwise
    and the analytic value there (measured 9/10 fail, values
    -29.4..+9.3; over the extended 40-case sweep 36/40 fail).
  - `test_a_far_start_can_converge_to_a_lower_value` (new, D5/D10):
    the pinned seeded case that *converges* -- case 32 of the
    `default_rng(101)` sweep (recipe in Recorded results): moved
    **3.642 deg**
    from the start, un-normalised value at the start **-19.936**
    (`der=False`), converged value **-27.293** (`record_property`
    both) -- freezing the fact that a converged foreign start is
    not score-monotone (the 2x2 fallback freezes `step[2]` and
    checks only `det >= euEps`, so the fixed point need not be a
    maximum).
  - `test_a_constructed_saddle_is_rejected`: monkeypatched
    `_derivatives` writing an indefinite Hessian with a finite
    gradient: `_refine_peak` returns the start,
    `converged is False` (kills saddle-acceptance and
    `np.linalg.solve` mutants).  (Corrected 2026-09-02: those three
    assertions do **not** kill the `np.linalg.solve` mutant -- its
    constant step is accepted, the loop times out at the cap and it
    returns the same start, the same -7.25 and the same
    `converged False`.  The stand-in therefore counts its calls and
    the test asserts **1** `der=True` evaluation; measured 1 for
    the port and 15 for the mutant, see Recorded results.)
  - `test_steps_must_shrink`: monkeypatched Hessian sequence whose
    cholesky step grows on iteration 2: the fallback path runs
    (pinned via the recorded step) and `prev_mag2` is not updated by
    fallback steps (the C++ non-update).
  - `test_refine_zyz_validates_before_the_kernel`: wrong-shape
    `flm`/`gln` (the measured 1e225 garbage rationale), non-finite
    or wrong-shape `zyz0` -> `ValueError`; a `bw`-mismatched
    `wigner_d_factors=` kwarg -> `ValueError`.
  - `test_buffers_are_owned_and_reused`: `d_beta` is the same object
    across two `refine_zyz` calls; a `clone()`'s differs; the factor
    triple `is`-shared into a clone once built; two identical calls
    are bitwise equal; the allocation goes through
    `wigner_d_table_pre(..., out=)` so a monkeypatched `np.empty`
    allocation dies on the wrapper's NaN-tripwire raise (D1/D3).
- Phase 4 classes: the coarse-range assertion and coarse pins gain
  `refine=False` explicitly; a new
  `test_correlate_refine_true_returns_the_refined_peak` on both
  correlators (rotated-pair recovery `< 4.92e-3` deg resp.
  normalised `< 4.92e-2`); `inspect.signature` pins on both
  `correlate` methods freeze `refine=False` as the private default
  (D6 -- the deliberate deviation from the C++ `ref = true`);
  `test_a_clone_has_the_attribute_set_of_a_constructed_instance`
  now polices the new factor-triple/`d_beta` attributes (plan 2.4).

`tests/test_indexing/test_spherical_indexer.py`:

- `refine=True` default in `inspect.signature`; construction with
  `refine=True` no longer raises; the factor triple is built once
  and `is`-shared across the per-phase correlators, into the
  `normalize=False` prototype, **and across the clones of the
  `refine_patterns` path** (D3).
- `memory_per_worker_bytes`: `refine=True` at `bw` 68 single phase
  == 49,426,200 + 5,030,912 = **54,457,112**; **two phases
  normalised == 79,169,400 + 2 x 5,030,912 = 89,231,224** (kills a
  dropped `n_correlators` factor, D3); `refine=False` keeps the
  Phase 6 value (the model term is conditional), while
  `_memory_model(True)` on a `refine=False` indexer returns the
  refined number -- what `refine_patterns`' info message prints
  (D8/D9).
- Info message: `Refinement: Newton (on)` /
  `Refinement: off` substrings; `"54 MB"` with the default;
  `get_info_message(..., refining=True)` prints
  `Refining ... orientation(s)` and never `Indexing` (D9).
- `refine_patterns`: not-indexed rows pass through bitwise; a
  monkeypatched always-fail `_refine_peak` leaves every input row
  unchanged; `phase_id >= len(harmonics)` -> `ValueError`; a
  **chunksize and worker-count invariance test** -- the refined
  results are bitwise equal across `chunksize in {1, 4, n}` and 1
  vs 4 dask workers (kills a `zyz`/`phase_id` block mis-alignment,
  D9); a `normalize=False` indexer refines through the prototype
  (the un-normalised branch runs, scores are the analytic values).

`tests/test_signals/test_ebsd_spherical_indexing.py`:

- `TestRefinedNickelSmall` (`bw` 68, default configuration =
  refined): vs the stored xmap -- **all nine < 1.0 deg** (measured
  max 0.695), **median < 0.75** (measured 0.505; the Phase 6-style
  margin, D10); per-pattern values
  + median `record_property`; the coarse twin (`refine=False`)
  reproduces the Phase 6 pins (median < 1.2, all < 2.0, scores
  0.4963-0.6239 `rel=0.05`); **per-point refined score > coarse
  score for all nine** (measured deltas +0.0108..+0.0286) and mean
  delta `> 0.005`; refined scores `pytest.approx` mean 0.5886
  `rel=0.05`; iterations recorded (8x2 + 1x3).
  `test_unnormalized_refinement_raises_the_unnormalized_scores`
  (new, D10): `normalize=False, refine=True` -- refined
  misorientations equal the normalised run's per point to
  `< 1e-4 deg` (measured identical, median 0.505 / max 0.695),
  un-normalised scores up **9/9** (measured coarse
  [0.2799, 0.3533] -> refined [0.2903, 0.3592], deltas
  +0.0059..+0.0164), refined mean `pytest.approx(0.332, rel=0.05)`.
- `TestRefineOrientationSpherical`:
  `refine_orientation_spherical(coarse_xmap, ...)` vs
  `spherical_indexing(refine=True)`: misorientation `< 1e-4` deg and
  `|score diff| < 1e-10` per point (measured 0.0 / 2.92e-14);
  **a point masked at refine time** (the method's own
  `navigation_mask`) and a not-indexed point (failure-injected
  constant pattern) keep their input rows (D9 -- a
  navigation-masked *input map* is not the vehicle here: orix gives
  such a map a bounding-box shape and in-data-only rows, so it is a
  refusal case below); `is_indexed` unchanged;
  iq equals the indexing run's; incompatible xmap
  (`_xmap_is_compatible_with_signal` message) **including a
  sparse-mask map from the Phase 6 `[::5, ::5]` recipe**,
  out-of-range `phase_id`, and a phase whose identity differs from
  `harmonics[id].phase` (`_equal_phase`, message naming both and
  the differing attribute) -> `ValueError`;
  `test_a_foreign_start_is_not_score_monotone` (the D10
  disjunction: one small-map rotation replaced by a seeded random
  one -- the row passes through bitwise *or* records a score at or
  below its input, `record_property` which branch ran);
  `verbose=0` silent, `verbose=1` prints
  `Refining 9 orientation(s)` and `Refinement speed:` (never
  `Indexing`).
- `TestRefinedNickelLarge` (20-pt, default suite): refined
  **median < 0.6** (measured 0.478), **max < 2.0** (measured 1.115),
  all 20 score deltas `> 0` (measured min +0.0020); coarse twin
  keeps the Phase 6 assertions.
- Phase 6 classes re-homed per plan 2.4 (explicit `refine=False`
  everywhere a coarse value is pinned; determinism/lazy/dtype tests
  run both settings, bitwise across chunking and workers; the five
  refusal tests deleted and the error-model/kernel-name/docstring
  guards across all four test modules extended per the plan-2.4
  file-by-file list).
- `TestPerformance`: the `>= 2` patterns/s/core floor asserted on
  the **default (refined)** path (measured ~65-70 at `bw` 68, ~33x);
  per-stage refine cost and the coarse/refined ratio
  `record_property` (no bound; C++ ~1.7x stated as context).
- Benchmark: default call (refined); scores-mean pin moves to
  `np.isclose(mean, 0.589, atol=0.03)`; floor kept.

## Weekly

- Symmetry-free refined suite at `bw in {88, 113, 123}`; the
  point-group suite over `bw` 53-63; the `bw` 24 value oracle.
- `nickel_ebsd_large` 165-pt: refined **median < 0.6** (0.456),
  **p95 < 1.2** (0.913), **max < 2.0** (1.140); score-increase
  fraction `>= 0.9` (measured 161/165) with mean delta `> 0.005`
  (+0.0131); the four caveat dips recorded (worst -4.8e-4).
- The small-map `bw` 88 refined row recorded (median 0.450 / max
  0.549).

## Manual

- Read the final `_derivatives` against `sht_xcorr.hpp:889-1119`
  side by side (coefficients, quadrants, hes mapping); read
  `_refine_peak` against `:442-499` (fallback entry set, prevMag2
  non-update, failure value).
- Re-verify the refineImage defect against `indexer.hpp:277-305`
  **and the score half against `idx.hpp:406-407, 441-446`** (the
  stored `corr` is 0 or the previous pattern's score -- `res` is
  value-initialised and hoisted outside the loop, D5)
  before shipping the documented deviation.
- Confirm no public docstring names roadmap phases (the extended
  guard tests of plan 2.4 machine-check
  `EBSD.refine_orientation_spherical` too); the CHANGELOG
  `Added` entry renders (no `Changed` entry -- D11); licence blocks
  updated in both files, incl. the two removed/rewritten
  `_indexer.py` bullets.

## Definition of done

Spec + amendments committed; failing tests committed first;
implementation; adversarial review (fidelity, conventions,
test-quality with the plan-3 bug-injection list) and fixes; suites
green `-n 0` then `-n 4`; doctests, pre-commit, sphinx html +
linkcheck clean; coverage of touched modules >= 95 % (target 100 %);
CHANGELOG `Added` entry; roadmap boxes ticked with measured numbers; PR #9
opened into fork `develop`.

## Recorded results

### 2026-09-02 -- pre-implementation reference measurements (spec drafting)

Environment: Windows 11, Python 3.13, the repo venv (`uv run`),
warm JIT, single thread. Probes `p7_probe.py` (kernel + refine +
parts `check/synth/synthsym/nsynth/real/large/timing/degen`),
`p7_probe2.py` (eps/bandwidth sensitivity), `p7_probe3.py`
(shape-mismatch diagnosis), `p7_probe4.py` (refine-only
equivalence); scratchpad, not committed. The probe transcribes
`derivatives()`/`refinePeak()`/`denominator()` verbatim (quadrant
weights instead of the four sequential adds -- association only)
on the merged Phase 1-6 modules, with
`_wigner_d_table_pre_kernel` called per evaluation on a reused
NaN-filled `(bw, bw, bw, 2)` buffer and
`_preprocessing._cholesky_solve_3x3` as the 3x3 solver.

**Kernel correctness.**
- Value vs inner-product oracle (`bw` 16, 4 flag combos x 6
  rotations): worst abs 6.051e-14, rel 1.813e-13; `der=True` value
  identical.
- Value/jac/hes vs the analytic `wigner_d_prime(2)` triple-sum
  oracle (`bw` 12, 2 flag combos x 4 rotations): 5.507e-14 /
  2.984e-13 / 1.705e-12 abs (hes scale 139.5).
- Jac/hes vs central finite differences (`bw` 16, `h = 1e-5`):
  2.974e-7 / 6.660e-4 (truncation-limited; value scale 18.4).
- Chebyshev multiple-angle recursion vs direct `sin/cos` at
  `alpha = 1.54983` (adjacent to `pi/2`, worst case), `m <= 119`:
  worst 2.943e-13.
- Pole contract (`bw` 24, on-peak start at `beta = 0`):
  `hes[1,1] = nan`, `jac = [-0, nan, -0]`; no exception under
  `error_model="numpy"` + `np.errstate`.
- **Shape-mismatch lesson**: `(68, 68)` spectra through a `bw` 88
  kernel (a probe bug: `SphericalIndexer(mph, det)` defaulted to
  `bw` 68 while the kit ran at 88) -> silent garbage 4.48e225 /
  -5.97e245 / NaN (BOUNDSCHECK off); with matching shapes the same
  call returns 0.3570 vs oracle 0.3562. Rationale for D1's
  validate-before-kernel rule.

**Synthetic refined accuracy** (recipes of `sht_xcorr.cpp`; seeded;
coarse via the merged `correlate`, refine from the interpolated
triple, `eps = 0.01`).
- Symmetry-free, `bw in {53, 54, 57, 60, 63, 64, 68, 88, 113, 123}`
  x 3: refined worst **2.958e-06 deg** (median 0.0; coarse
  0.055-0.549); every case 2 iterations, 0 failures. C++ gate
  4.92e-3 deg -> 1663x margin.
- Eight point groups x `bw {53, 60, 63}` x 3 (72 cases): worst
  **4.518e-06 deg** (gate 0.351 -> 77,600x); iterations 2-3.
- Normalised wedge (`testNCorr`), `bw {53, 68}` x `{1, 4/m}` x 3
  (12 cases): refined worst **2.133e-02 deg** (gate 4.92e-2 ->
  2.3x); coarse 0.065-0.835; scores rise in 12/12, e.g. 0.56756 ->
  0.69284 (`bw` 68, seed rot 2); iterations 2-3.
- Near-degenerate targets (`bw` 24): `beta = 0`: coarse 1.1973 ->
  refined 0.000e0; `+-1e-3` rad: 1.20 -> 5.731e-2 deg (= the beta
  offset -- the fallbacks freeze the false DoF); `pi`: 0.8984 -> 0;
  `pi - 1e-3`: 0.9231 -> 5.73e-2.
- Pole starts: exactly `beta = 0` on-peak -> 1x1 path, 1 iteration,
  exact; `beta = +-pi` start with the peak at `beta = 0` ->
  converges along the ridge to the antipodal local max (Newton is
  local; recorded).
- Far starts (10 unrelated random starts, `bw` 24): 9/10 fail and
  return the start unchanged ("moved 0.000 deg") with the analytic
  value there (-29.35, +9.26, -5.92, +0.09, -1.79, +2.76, -1.01,
  -0.42, +4.95, -17.58 across cases); 1/10 walks 4.115 deg to a
  local maximum in 6 iterations.
- On-grid identity (`bw` 16): 1 iteration, angle 0, peak/power
  1.000000000000.

**Real data** (`bw` 68 unless stated; default configuration;
`pc_average`; coarse rows reproduce Phase 6 digit for digit).
- `nickel_ebsd_small`: coarse median 0.599 / p90 0.767 / max 0.838;
  refined **median 0.505 / p90 0.601 / p95 0.648 / max 0.695** deg.
  Per-pattern coarse [0.354, 0.750, 0.599, 0.446, 0.484, 0.594,
  0.713, 0.681, 0.838] -> refined [0.465, 0.577, 0.503, 0.493,
  0.417, 0.505, 0.536, 0.532, 0.695]. Normalised scores: mean
  0.5701 -> **0.5886** (range 0.5143-0.6347); per-point deltas
  [+0.0205, +0.0194, +0.0211, +0.0108, +0.0180, +0.0222, +0.0132,
  +0.0286, +0.0126] -- 9/9 up. Iterations 8x2 + 1x3, 0 failures.
  `eps = 0.001`: identical digit for digit (residual systematic).
  `bw` 88: refined median **0.450** / max **0.549**; scores 0.6383
  -> 0.6409, 9/9 up.
- `nickel_ebsd_large` 20-pt (`[::15, ::15]` via the mask recipe,
  own detector `pc_average`): coarse 0.499 / p90 1.075 / max 1.350;
  refined **0.478 / p90 0.815 / p95 0.981 / max 1.115**; scores
  0.5678 -> 0.5813, deltas min +0.00198, 20/20 up; iterations
  17x2 + 3x3.
- 165-pt (`[::5, ::5]`): coarse 0.530 / p90 0.931 / p95 1.082 / max
  1.495; refined **0.456 / p90 0.812 / p95 0.913 / max 1.140**;
  scores 0.5684 -> 0.5815, deltas min **-4.8e-4** .. +0.0333,
  **161/165 up** (the 4 dips are the omitted window chain rule);
  iterations 152x2 + 13x3, 0 failures. Roadmap weekly bounds
  0.6 / 1.2 / 2.0: pass.
- Refine-only equivalence (small map): `rotation_to_zyz(
  rotation_from_zyz(zyz_c))` hands back the glide triple (beta sign
  flipped, start misorientation 0.0); refined A-vs-B misorientation
  **0.000e0 deg** on all nine, score difference worst **2.920e-14**;
  iteration counts identical (2/2, 3/3).

**Timing** (warm, best-effort means over 12 reps).

| `bw` | flags | refine (2 it) | coarse | ratio | dTablePre | der True | der False | `np.full` d_beta |
|---|---|---|---|---|---|---|---|---|
| 53 | `n_fold` 1 | 1.45 ms | 11.3 | 1.13x | 0.85* | 0.71 | 0.18 | 0.50 |
| 53 | `m-3m` | 0.33 | 5.9 | 1.06x | 0.09 | 0.16 | 0.09 | 0.49 |
| 68 | `n_fold` 1 | 3.00 | 24.3 | 1.12x | 0.17 | 1.50 | 0.38 | 1.03 |
| 68 | `m-3m` | 0.65 | 12.1 | 1.05x | 0.17 | 0.32 | 0.17 | 1.03 |
| 88 | `n_fold` 1 | 11.06 | 100.3 | 1.11x | 0.56 | 5.50 | 1.63 | 2.27 |
| 88 | `m-3m` | 2.54 | 44.4 | 1.06x | 0.57 | 1.25 | 0.70 | 2.29 |

(*first-measured jitter.) Real-data per pattern, warm (165-pt run):
coarse 13.17 ms + refine-and-denominator **1.39 ms** = ratio 1.11x
(small-map first run incl. cache load: 23.56 + 6.42, 1.27x). The
`np.full` column is the rejected per-call allocation (would rival
the whole `m-3m` refine). C++ context (Phase 4 D11): refine 2.5 /
5.2 / 12.3 (`n_fold` 1) resp. 0.6 / 2.1 / 2.7 ms (`m-3m`) at `bw`
53 / 68 / 88 on coarse 8.9 / 16.3 / 48.1 resp. 3.3 / 6.9 / 18.9 --
ratio ~1.3-1.7x.

**C++ facts verified for this spec**: `refinePeak` `maxIter = 15`
(`:448`); `interpolateMaxima` `maxIter = 25`, `eps = sqrt(machine
eps)` (`:1312-1313`); `eps` default 0.01 (`:189`); `prevMag2 = 2 pi
3 / slP` (`:450`, the unit mix); the `deg` flag `:909` has no other
occurrence (dead); `Result::corr` uninitialised (`:54-64`);
`refineImage` `:296` drops the `refine()` return and `eu` is const
-> the refine-only work item (`idx.hpp:438-450`) stores the
unrefined orientation and a stale `corr` (defect, D5);
`solve::cholesky` throws at `linalg.hpp:416` (sign mismatch -> our
status 1) and `:422` (small pivot -> status 2), negates
negative-definite matrices (backsolve `:493`) -- the Phase 5
`_cholesky_solve_3x3` is the same routine; the C++ test gates:
sizes `{53, 68, 88, 113, 123, 158, 54, 55, 56, 57, 58, 59, 60, 62,
64}` (`sht_xcorr.cpp:295-298`), `eps = cbrt(float eps)` (`:294`),
`epsN = 10 eps` (`:316`), groups `sqrt(eps) * 5` (`:345`), the
eight groups (`:332-342`), point-group loop `bw` 53..63 (`:350`,
`:373`).

### 2026-09-02 -- revision re-measurements (adversarial spec review)

Environment as above; probe `p7_probe5.py` (parts `nsplit farconv
copyless pyfunc poleulp scales unnorm memory`) extending the
drafting probes, scratchpad, not committed. Three corrections to the
section above, superseding it where they conflict: (a)
`Result::corr` is **not** uninitialised -- `std::vector<Result>
res(om.size())` at `idx.hpp:406` value-initialises (zero) and `res`
is hoisted outside the per-pattern loop, so the refine-only branch
stores 0 or, in a mixed batch, the previous pattern's score; (b)
the far-start "walks 4.115 deg to a local maximum" case is a
**decrease** (see farconv below); (c) the C++ applies `epsN` only to
its symmetry-free normalised loop (`sht_xcorr.cpp:316-329`) -- its
normalised point-group loop runs under the loosened `eps` 0.351
(`:345`, `:371-391`).

- **nsplit** (the wedge suite of `part_nsynth`, same seeds, per
  subset): `(1, F)` cases worst **1.8487e-02 deg** (gate 4.92e-2 ->
  2.7x); `4/m` cases worst **2.1326e-02 deg** (gate 0.351 -> 16.5x);
  the overall worst is a `4/m` case; scores up 12/12.
- **farconv** (`bw` 24, `flm = random_alm(24, default_rng(101))`,
  then 40 sequential `(zyz_true, start)` pairs from the same
  generator -- the drafting `part_degen` recipe extended; the first
  10 reproduce the drafting run): **36/40 fail** and
  return the start bitwise with the analytic value; **4/40
  converge** -- case 0: moved 4.115 deg, value -27.786 -> -29.353
  (the mis-recorded "local maximum"); case 13: moved 1.894, +25.949
  -> +27.428; case 14: moved 1.897, -42.495 -> -46.136; case 32
  (the pinned test case): moved **3.642 deg**, **-19.936 ->
  -27.293**. 3 of 4 converged cases decrease.
- **copyless**: `_refine_peak` with `hes.copy()` vs the C++-exact
  uncopied `hes` over 15 seeded cases (12 far starts + 3
  near-degenerate targets, `bw` 24): `(zyz, value, converged,
  iterations)` **bitwise identical 15/15** -- the copy is dropped
  (D4).
- **pyfunc**: `_euler._wrap_beta(np.float64(0.0))` returns a Python
  `float` (Numba boxes the return); interpreted
  `1.0 / math.sqrt(1.0 - t*t)` at the pole raises
  `ZeroDivisionError` **inside** `np.errstate`; the `np.cos`/
  `np.sqrt` chain yields `np.float64(inf)`; compiled `math.*` vs
  `np.*` over 1000 samples: worst abs diff **0** (bitwise) -- D1's
  NumPy-scalar rule.
- **poleulp**: `math.cos(pi) == -1.0` and `math.cos(-pi) == -1.0`
  true on this host; `beta0 = nextafter(pi, 0)` still gives
  `cos == -1.0` (the ulp gap is far below eps at 1), so
  `hes[1, 1]` is NaN at `0.0, -0.0, +-pi` and the one-ulp input
  here; `_refine_peak` returns without raising in every case (1
  iteration at the `0/-0.0` starts, 3 at `+-pi`). The exact-NaN
  claim at `+-pi` remains libm-conditional (D2) -- the assertion is
  weakened to NaN-or-`> 1e12`.
- **scales** (the frozen `random_alm` fixture): `bw` 16 value
  fixture max `|value|` 19.44, self powers 13.1-166.2; `bw` 12
  jac/hes fixture max `|value|` 12.65, max `|hes|` 529.3
  (seed-dependent; the drafting run's "hes scale 139.5" was its own
  rotation set -- scales are `record_property`ed, not asserted).
- **unnorm** (`normalize=False, refine=True`, small map, `bw` 68):
  refined misorientations **identical per pattern** to the
  normalised run -- median 0.505 / max 0.695 (same coarse cells;
  Newton maximises the un-normalised value in both paths);
  un-normalised scores mean 0.3220 -> **0.3324**, coarse
  [0.2799, 0.3533] -> refined [0.2903, 0.3592], deltas
  [+0.00585, +0.01639], **9/9 up**; iterations 8x2 + 1x3, 0
  failures.
- **memory** (arithmetic): single-phase base 49,426,200, + 1
  `d_beta` = 54,457,112; two-phase base 79,169,400, + 2 `d_beta` =
  **89,231,224**; `16 bw^3` = 5,030,912 / **10,903,552** /
  23,086,352 B at `bw` 68 / 88 / 113 (the draft's "11.5 MB" at 88
  corrected to 10.9).

### 2026-09-02 -- test-quality review re-measurements (fixture seeds)

Environment as above; the review's reference implementation
(`p7_ref.py`, the drafting probe's verbatim `derivatives()` /
`refinePeak()` / `denominator()` with the `np.*` `csc` chain of D1)
monkeypatched into `_xcorr`, plus `p7_fix_check.py`; scratchpad, not
committed. These are the numbers the **shipped fixtures** give,
which differ from the drafting probe's where the seed recipes
differ; they supersede the earlier sections for those fixtures.

- **Point groups under the test file's own seeds**
  (`100 * bw + POINT_GROUPS.index(name)`, not the drafting
  `11 + bw`): worst **9.0355e-06 deg** over the 24 cases, gate
  0.351 -> **38,800x**. Non-zero cases only: `2/m` 5.915e-6 and
  `4/m` **9.036e-6** at `bw` 53, `2/m` 1.708e-6 at 60, `6`
  3.818e-6 at 63; the other 20 are exactly 0.
- **Negated-`beta` twins in the two derivative oracles.**
  `quaternion_to_zyz` returns `beta` in `[0, pi]`, so the drafting
  fixtures never evaluated below the equator and the
  "`csc` sign for negative beta dropped" mutant of plan 3
  **survived both named killers**. Each drawn triple is now
  evaluated at `zyz * (1, -1, 1)` as well:
  - analytic oracle (`bw` 12, betas -2.068..+2.068): value
    5.68e-14, jac 3.06e-13, hes 1.48e-12, worst imaginary part
    3.43e-13; scales max `|value|` 16.80, max `|hes|` 922.3;
  - finite differences (`bw` 16, betas -2.911..+2.911): jac
    2.72e-7, hes 6.65e-4.
  Bounds unchanged (1e-11 / 1e-10 / 1e-9 / 1e-5 / 1e-2). With the
  sign dropped the mutant measures jac **5.79e+01** (analytic) and
  **1.15e+02** (finite differences), i.e. it now dies by both.
- **Constructed saddle, iteration count.** The port takes exactly
  **1** `der=True` evaluation: the Cholesky solve returns the
  indefinite status, the 2 x 2 determinant is -1 and the first
  iteration is a total failure. With `numpy.linalg.solve`
  substituted the constant step (`mag2` 0.03) is accepted on every
  iteration -- equal to `prev_mag2` from the second on, which the
  C++ rule accepts -- and the loop runs the full **15** iterations,
  yet **still** returns the start with value -7.25 and
  `converged False`. The return values do not separate the two; the
  iteration count does, so the test asserts it.
- **Exact-pole starts.** `_refine_peak` from `(0.45, +-0.0, -0.75)`
  with the peak at `(0.4, 0, -0.7)` **converges** through the 1 x 1
  path in one iteration, value 396.238717 -- so the pole test
  asserts `converged is True` instead of the vacuous "finite or
  NaN". At `+-pi` it also converges, with `hes[1, 1]` NaN.
- **Small map derivative calls** (default configuration, `bw` 68):
  **19** with `der=True` and **18** with `der=False`, i.e. exactly
  the drafted `>= 18` bound. The value-only assertion is loosened
  to `>= 9` (one denominator evaluation per pattern) and the exact
  figures `record_property`ed; the counters are taken under a lock,
  since a chunked run increments them from several worker threads.
- **The factor triple's identity.**
  `_validated_wigner_d_factors` ends `return e_km, w_jkm, b_jkm`,
  a **new tuple** of the same three arrays, so
  `correlator.wigner_d_factors is factors` is False while element
  identity is `[True, True, True]` (measured). The sharing
  assertions of "the very same three arrays" are therefore made per
  array, at all six sites.
- **Foreign start on the small map** (`bw` 68, pattern 4, the
  seeded `default_rng(7)` rotation of
  `test_a_foreign_start_is_not_score_monotone`): the analytic
  normalised value at that start is **-0.066475**, and the
  refinement **fails** from it and returns it bitwise with that
  same value (moved 0.0000 deg). The injected input row therefore
  carries -0.066475, not the Ni coarse ~0.57 which no refinement
  from a random orientation could reach and against which the
  disjunction was vacuous.

### 2026-09-02 -- implementation measurements (shipped code)

Environment as above; the numbers below come from the **shipped**
`_derivatives` / `_refine_peak` / `refine_zyz` / `_denominator` /
`refine_patterns` / `EBSD.refine_orientation_spherical`, read out of
the `record_property` entries of a
`uv run pytest tests/test_indexing/test_spherical_refinement.py -n 0`
run (`--junitxml`; 156 passed, 6 weekly skips). Every frozen band of
the sections above is reproduced by the port.

**Kernel oracles** (all inside their frozen bounds):

| quantity | measured | bound |
|---|---|---|
| value vs inner product, `bw` 16, `der` True and False | 6.051e-14 (scale 19.44) | 1e-11 |
| value vs analytic triple sum, `bw` 12 | 5.684e-14 | 1e-11 |
| jacobian vs analytic, `bw` 12 (betas -2.0676..+2.0676) | 3.055e-13 | 1e-10 |
| hessian vs analytic, `bw` 12 (scale 922.3) | 1.478e-12 | 1e-9 |
| jacobian vs central differences, `bw` 16 | 2.724e-07 | 1e-5 |
| hessian vs central differences, `bw` 16 | 6.653e-04 | 1e-2 |
| `_phase7_derivatives` vs `wigner_d_prime(2)`, `bw` 15 | 5.684e-14 | 1e-12 |

- `.py_func` parity is **exactly bitwise here** (worst relative value
  difference 0.000e+00, worst derivative difference 0.000e+00), at
  both `(n_fold, mirror)` settings and both `der` settings, i.e. well
  inside the 1e-12 / 1e-9 bounds the CI lesson keeps.
- Pole contract at `beta` `0.0` and `-0.0` (`bw` 24, on-peak start):
  `hes[1, 1]` NaN, `jac = [-4.883e-13, nan, -4.883e-13]`, value
  396.238717, no exception; `_refine_peak` converges through the
  1 x 1 path in **1** iteration with the step's beta and gamma slots
  exactly 0 (step `[1.186e-17, 0, 0]`). At `beta = +-pi` this host's
  libm gives `cos == -1.0` and `hes[1, 1]` is NaN, as recorded.

**Synthetic recovery** (the C++ gates, split as the C++ splits them):

- symmetry free, `bw in {53, 54, 57, 60, 63, 64, 68}` x 3: worst
  **2.958e-06 deg** (gate 4.92e-3), all 21 cases in **2** iterations,
  zero failures, coarse 0.066-0.549;
- eight point groups x `bw in {53, 60, 63}`: worst **9.035e-06 deg**
  (gate 0.351), the `4/m` case at `bw` 53 -- exactly the figure the
  test-quality review re-measured under the shipped fixture seeds;
  20 of 24 cases are exactly 0;
- normalised wedge: `(1, F)` worst **1.8487e-02 deg** (gate 4.92e-2),
  `4/m` worst **2.1326e-02 deg** (gate 0.351), refined score above
  the coarse one in **12/12** (e.g. `bw` 68 rot 2: 0.56756 ->
  0.69284);
- near-degenerate targets, `bw` 24: `beta = 0` and `beta = pi` refine
  to **0.0 deg** from coarse 1.1973 and 0.8984, and the `+-1e-3`
  offsets to **5.7307e-02** / **5.7308e-02**;
- far starts, `bw` 24: **9/10** fail and return the start bitwise
  with the analytic value there (-29.353 .. +9.263); the pinned case
  32 converges, moving **3.642 deg** from value **-19.936** to
  **-27.293**;
- constructed saddle: **1** `der=True` evaluation, the start
  returned, value -7.25, `converged False`; the monotone-step
  sequence: 4 iterations, gamma frozen by the 2 x 2 fallback, alpha
  moved by 0.1 + 0.5 + 0.15 + 1e-9;
- `eps` 0.01 against 0.0001 at `bw` 53: **0.000e+00 deg** apart.

**Real data** (`bw` 68, default configuration, `pc_average`):

| data | refined median / p95 / max (deg) | scores |
|---|---|---|
| small (9) | **0.5052** / 0.6482 / **0.6953** (coarse 0.5987 / 0.8026 / 0.8379) | 0.5143-0.6347, mean **0.5886**, deltas +0.0108..+0.0286, 9/9 up |
| small, `normalize=False` | 0.5052 / 0.6482 / 0.6953, **0.000e+00 deg** from the normalised run per point | deltas +0.00585..+0.01639, 9/9 up |
| large 20-pt | **0.4780** / 0.9814 / **1.1148** (coarse 0.4988 / 1.2307 / 1.3497) | deltas min +0.00198, max +0.03066, mean +0.01348, **20/20 up** |

- Small-map derivative calls: **19** with `der=True` and **18** with
  `der=False`, i.e. the drafted 8 x 2 + 1 x 3 iterations plus one
  denominator pair per pattern.
- `refine_orientation_spherical` against
  `spherical_indexing(refine=True)`: misorientation **0.000e+00 deg**
  and score difference **2.920e-14** over the nine patterns, image
  quality equal.
- The sparse-mask map (`mask[0] = False`) gets `xmap.shape`
  **(1, 3)** from orix and is refused by
  `_xmap_is_compatible_with_signal`, as D9 predicted.
- Foreign start (pattern 4, `default_rng(7)`): the refinement fails
  and hands back the start with its analytic normalised value
  **-0.066475**; the stored row is the glide-equivalent quaternion of
  that same orientation rather than the input's bits, so the D10
  disjunction is satisfied by its score half.

**Timing on this machine** (single dask worker, `bw` 68, warm):
refined throughput **40.4 patterns/s/core** (the `>= 2` floor keeps a
20x margin); refined over coarse **1.11x** (22.53 -> 25.03 ms per
pattern end to end); per stage, coarse `correlate` 17.87 ms and
refine-with-denominator **1.81 ms** (ratio 1.10x). The absolute
milliseconds sit above the drafting probe's 13.17 + 1.39 because this
run shared the machine with the rest of the suite; the ratio
reproduces.

**Memory model** re-measured through the shipped
`memory_per_worker_bytes`: single phase refined **54,457,112 B**
("54 MB" in the information message), two phases normalised
**89,231,224 B**, `refine=False` unchanged at 49,426,200 B, and
`_memory_model(True)` on a `refine=False` indexer returns the refined
number -- what `refine_patterns` prints.

**One mechanism deviation, measured** (the chunk alignment of D9).
`dask.array.map_blocks` **cannot** carry the `(n, 3)` starting
triples and the `(n,)` phase indices as block arguments of a
`(n, r, c)` pattern array. Measured on the installed dask:
`map_blocks` builds its `argpairs` as `tuple(range(a.ndim))[::-1]`,
i.e. it aligns arrays on their **trailing** axes, so a 9-pattern run
chunked `(4, 4, 1)` hands **every** block the whole `(9, 3)` and
`(9,)` arrays (`shapes (4, 2, 2) (9, 3) (9,)` printed from inside the
chunk function) -- exactly the silent mis-alignment D9 exists to
prevent, and it would still return the right number of rows. The port
therefore maps with `dask.array.blockwise` and an explicit index
expression -- `"ij"` out of `patterns "ikl"`, `zyz "im"`,
`phase_id "i"` -- which is the general form `map_blocks` itself calls
and makes the pattern axis one named index of all three arrays. The
`da.from_array(..., chunks=(patterns_da.chunks[0], ...))` wrapping of
the two row arrays is kept exactly as D9 froze it, and the bitwise
chunksize, worker-count and permutation invariance tests pass.

**Coverage** of the touched modules under

```
uv run pytest tests/test_indexing/test_spherical_refinement.py \
  tests/test_indexing/test_spherical_xcorr.py \
  tests/test_indexing/test_spherical_indexer.py \
  tests/test_signals/test_ebsd_spherical_indexing.py \
  --cov=kikuchipy.indexing._spherical --cov=kikuchipy.signals.ebsd \
  --cov-report=term-missing -n 4
```

`_xcorr.py` **100.00 %** (690 statements) and `_indexer.py`
**100.00 %** (337), with every line of `EBSD.spherical_indexing` and
`EBSD.refine_orientation_spherical` covered too (the misses reported
for `signals/ebsd.py` are all in the unrelated methods this selection
does not exercise). Reaching 100 % needed four groups of additions to
the test module, none of which relaxes an existing assertion: a
`.py_func` parity case at `(n_fold=1, mirror=False)` and both `der`
settings (the mirrored pair of the frozen test never runs the
`(j + m)` parity negation nor the value-only loop), the two
`refine_patterns` guards, and its shape, `chunksize`, lazy-input and
2 GiB paths, and the signal method's detector-shape,
navigation-dimension, four navigation-mask, master-pattern-phase,
keep-n and point-out-of-the-data paths. The last of those records a
case D9 did not name: a `navigation_mask` which removes an
**interior** point leaves the in-data bounding box alone, so such a
map keeps `xmap.shape == (3, 3)`, passes the compatibility check and
is refined with its out-of-data row carried through untouched
(measured) -- only a mask which shrinks the bounding box is the
refusal case.

**Whole-suite `-n 4` worker crashes are pre-existing, measured.**
The four gate selections above are green, but a
`uv run pytest tests/test_signals tests/test_indexing -n 4` over the
*whole* suite loses one to four xdist workers per run on this machine
("worker 'gwN' crashed while running ...", a native crash rather than
an assertion failure), and the tests it takes down differ from run to
run (`test_ebsd_hough_indexing`, `test_ebsd_refinement`,
`test_spherical_wigner`, `test_spherical_xcorr`). Checked against the
pre-implementation tree: the same command on the stashed
tests-and-stubs commit crashes `gw3` and `gw4` too, at the *same*
site (`test_spherical_xcorr.py::TestNormalized::
test_the_compatibility_keyword_reaches_the_interpolation[24]`), so
this is the environmental instability the `NUMBA_CACHE_DIR` note in
`conftest.py` already describes and not a refinement regression. The
narrower `-k "spherical or sht" -n 4` selection (2558 tests) ran
green three times in a row.

### Review follow-up, 2026-09-02 (fidelity / conventions / bug-injection)

**The normalized score is IEEE, not raising (fidelity F1).** The C++
differential probe (`cxx_probe2b.exe` against `py_probe2.py`, `bw` 12,
identical `flm`/`flm2`/`mlm`/`gln`) had two of four starts return
`-nan(ind)` from `NormalizedCorrelator::refinePeak` where the port
raised `ValueError: math domain error` out of `math.sqrt` -- the
whole-cube twin at `_xcorr.py:2900` already produced the quiet NaN.
`_denominator` now takes `np.sqrt` under
`np.errstate(invalid="ignore")` and `refine_zyz` divides in
`numpy.float64`; re-measured, the two finite cases stay bitwise
(`-0.05583698747953895`, `0.24783968938036238`, C++
`-0.055836987479538643`, `0.24783968938036166`) and the two negative
radicands now return `nan` with the refined `eu` matching the C++ to
16 digits (`1.1005905059928804`, `-0.9073430219362185` /
`2.8471347698084934`). A denominator forced to exactly `0.0` gives
`inf`, not `ZeroDivisionError`. Blast radius removed: the phase loop
of `_index_chunk` sits inside the per-pattern `except`, so one phase
with a negative radicand used to discard every sibling phase's
candidates.

**The sentinel of `refine_orientation_spherical` (fidelity F2) is a
re-derivation and stays one.** The proposed exact fix, a seventh
written-flag column surfaced as a fifth dictionary key, is refused by
the acceptance contract: `test_the_result_contract` asserts
`set(results) == {"zyz", "scores", "iq", "phase_id"}`. The reviewer's
alternative, keying on `iq != 0.0` alone, is strictly narrower than
the implemented `scores != 0 | iq != 0` and would misclassify a row
with a non-zero score and a zero image quality, so it is worse. The
condition needs the back-projection to answer a non-constant pattern
with an all-zero spectrum (only reachable when a `signal_mask` makes
`_mean_fill`'s output constant and zero, `_back_projection.py:1491`)
*and* the correlation to land on a float64 zero at the same point.
Recorded as a documented residual instead, in the comment at the
scatter site.

**`refine_patterns` no longer fails in silence (conventions F1).**
With `RuntimeError` injected into `_xcorr._refine_peak`, the previous
code returned the input map with `warnings: []`. It now warns
`"9 of 9 indexed pattern(s) could not be refined ..."`, mirroring
`index_patterns`. A negative `phase_id` is the intended pass-through
and is excluded from the count (`packed[:, 4] >= 0`), so the
not-indexed row of `test_a_not_indexed_row_passes_through_bitwise`
does not warn.

**Timing claim corrected (conventions F4).** The `refine` docstring
said coarse "is faster by 5-30 %"; the recorded refined/coarse ratios
are 1.05-1.27x, i.e. 4.8-21.3 %. Reworded to "costs 5-21 % less time
(measured refined to coarse ratios 1.05-1.27x)".

**Comment width in the acceptance module (conventions F6).** Eleven
73-char `# === Title === #` banners and the two inline comments at the
`test_steps_must_shrink` gradient list were the module's only comment
lines over 72; all are now within it (measured by a `tokenize` scan:
`test_spherical_refinement.py` 0). *(corrected 2026-09-02: the
conventions review's "0 violations in all twelve other
`test_spherical_*.py` files" is wrong -- the same scan finds five
pre-existing ones, `test_spherical_indexer.py:246` (73, a Phase 6
banner), `test_spherical_master_pattern_harmonics.py:940,960` (76),
`test_spherical_wigner.py:1899` (77) and
`test_spherical_xcorr.py:2008,2886` (75, 78). None are Phase 7's and
none were touched.)*

**Survivor killers, each measured against its mutant.** Four genuine
gaps closed and one equivalence pinned at the call:

| mutant | new or changed test | pristine | mutant |
|---|---|---|---|
| `< abs_eps` -> `<=` | `test_the_stopping_threshold_is_the_ported_one`, scale 1.0 added | 15 iterations, `step[0]` bitwise `abs_eps` `0.001963495408493621` | 1 iteration |
| `_REFINE_FIRST_STEP_SCALE` 3.0 -> 1.0 | `test_the_first_step_bound_is_the_ported_seed` | `mag2` 0.29 accepted whole, `gamma` `0.3 -> 0.1`, 2 iterations | first step rejected, 2 x 2 fallback freezes `gamma` at 0.3 |
| `correlators[phase_id]` -> `[0]` and `spectra[phase_id]` -> `[0]` | `test_the_row_s_own_phase_decides_the_correlator[True/False]` | normalized scores phase 0 0.514-0.635 vs phase 1 -0.128--0.014; un-normalised 0.290-0.359 vs -0.071--0.008 | both phases bitwise identical |
| `phase_id >= n_phases` -> `>` | `test_the_phase_index_boundary_is_refused` | `ValueError` at `phase_id == n_phases` | slips through to an `IndexError` swallowed as a zero-score row |
| `_denominator` flags -> the pattern's | `test_the_denominator_uses_the_reference_flags` | `[(True, 4), (True, 4)]` | `[(False, 1), (False, 1)]` |

**The two symmetry-flag mutants really are equivalent (bug-injection
E1/X14), re-measured.** `_derivatives` with `(n_fold, mirror)`
loosened to `(1, False)`: synthetic `4/m` at `bw` 16 is **bitwise
equal** in value, jac and hes over six rotations; the real Ni master
at `bw` 68 (`m-3m` -> `(4, True)`) moves the value by `0.0`, the jac
by 2.2e-15 and the hes by 7.9e-14 on a value of 9.9, since the 2040
skipped coefficients have `max |flm| = 7.1e-16` against an overall
3.16. The `d_j = 2 if mirror` mutant adds 56 degrees whose `|flm|` is
**exactly** `0.000e+00`. No value assertion at any tolerance in the
suite (tightest oracle bound 1e-10) can see either, so the
denominator flags are pinned at the call instead and `d_j` is
recorded as an accepted equivalent mutant.

**Row-shuffle permutation made non-cyclic (bug-injection bonus).**
`test_the_rows_are_not_shuffled_by_the_blocks` used
`[4,5,6,7,8,0,1,2,3]`, which commutes with a row roll, so mutant 28b
(starts rolled by one) passed it and was only caught one stage later.
With `[2,0,5,8,1,7,3,6,4]` the same mutant fails that test directly
(measured).

**Gate set after the fixes.** `test_spherical_refinement.py` **165
passed, 6 weekly skips** at `-n 0` (32.2 s) and at `-n 4` (30.1 s),
up from 156 by the nine new tests. `tests/test_indexing
tests/test_signals -k "spherical or sht" -n 4`: **2567 passed, 714
skipped** (126 s). New-line coverage of the working-tree diff, over
that selection: `_xcorr.py` 249/249 new statements, `_indexer.py`
94/94, `ebsd.py` 89/89 -- **100 %** on all three, with `_xcorr.py`
and `_indexer.py` at 100 % whole-file. Twelve mutants (the five
above plus the two F1 reversions, the warning removal, its
not-indexed miscount, the roll and an inverted `was_refined` key) all
**KILLED**, sources md5-restored. Docs build exits 0 with two
warnings, both intersphinx network failures; the `refine_patterns`
page now renders `refine_orientation_spherical` as a resolved link
with zero multiline xref spans and zero `_spherical._euler` mentions.
Two `--doctest-modules` failures in `signals/ebsd.py` are
pre-existing and outside every changed hunk: the `EBSD` class
docstring's degree sign (console encoding) and
`get_image_quality`'s float32 last digits
(`0.16031407` vs `0.16031112`).
