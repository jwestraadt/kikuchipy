# HREBSD-DIC -- `hrebsd-dic`: validation

**Status note (requirements D19): drafted 2026-09-07 with ZERO test
execution and ZERO measurement** -- the spec stage is read-only
analysis plus `specs/` writes. Every tolerance marked `MTP`
(measured-then-pinned) is filled at the owning stage's
failing-tests/implementation gates (`pytest.approx(measured,
rel=0.05)` or the ~2x margin convention on misorientation/strain
bands); this file ships with placeholder language and an
intentionally near-empty "Recorded results" ledger. **No PR-gated
CI protects this code** (branch policy, plan section 1) --
pushes of the branch DO trigger the fork's on-push full-matrix CI
(`.github/workflows/tests.yml:22-29`; corrected 2026-09-07 at
spec review, an earlier draft wrongly claimed no CI run ever
exercises this code), recorded as an extra signal only, per the
plan section 1 push policy: every gate below
is a LOCAL gate, recorded here with numbers, recipes and the
machine ID (the Recorded results discipline of
`specs/2026-09-07-spherical-gpu/validation.md`). Tests marked
[download] need pooch + network once (cached after); everything
else is pure Python on shipped/synthetic data.

Oracle-first principle (requirements, Reference-implementation
status): there is no binary ground truth -- EMHREBSDDIC was never
built here and its pipeline is WIP -- so validation rests on
synthetic oracles whose expected answers are known EXACTLY by
construction, plus one real-data noise-floor benchmark. GND is
validated against the mathematics, never against another code's
sign conventions (D14.1).

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing tests/test_signals -k "hrebsd" -n 4
```

(one `-n 0` run first per stage that adds a numba kernel.)

### V0 -- Interpolation kernel (`tests/test_indexing/test_hrebsd_interpolation.py`) -- Stage A

- `test_kernel_matches_map_coordinates`: numba bicubic evaluation
  of spline-filtered coefficients equals
  `scipy.ndimage.map_coordinates(order=3, prefilter=False,
  mode="mirror")` to <= 1e-12 relative on random interior points,
  random images, BOTH sides on f64 coefficients (scoped
  2026-09-07, spec review: `map_coordinates` on f32 input
  computes and returns f32, so 1e-12 is unreachable there). The
  f32 coefficient-storage arm (D17) is compared against the same
  f64 reference under its own bound `KERNEL_F32_TOL` (MTP,
  measured alongside the D17 dtype A/B; f32-ULP class expected).
  [D3/D17]
- `test_kernel_py_func`: the same through `.py_func`
  (tech-stack.md:49). [D3/D18]
- `test_gradient_kernels_match_spline_derivative`: analytic
  gradient planes vs `map_coordinates` derivative-of-spline
  reference. [D3]
- `test_mirror_boundary`: near-edge evaluations match the mirror
  reference; no extrapolation blow-up -- the blow-up arm asserts
  the mirrored values stay inside the pattern's own range plus
  `MIRROR_OVERSHOOT_TOL` (MTP; added 2026-09-07 at the review, the
  drafted `< 4 * span` band being unfalsifiable under mirror-mode
  interpolation). [D3]
- `test_order_ab_harness` (measurement harness, Stage A gate;
  added 2026-09-07 at the review): the D3 bicubic-vs-quintic A/B
  of plan open question 1, run where the accuracy difference
  originates -- both interpolators against an ANALYTIC
  band-limited ground truth on random interior points, with a
  timing arm. Bicubic stays the default unless quintic buys more
  than 2x accuracy at under 1.5x cost, which the test asserts
  literally and which a re-pin must record here with its date.
  [D3]

### V1 -- Algebraic round trips (`test_hrebsd_homography.py`) -- Stage A

- `test_shape_function_compose_invert`: `W(h) W(h)^-1 = I`;
  composition associativity; `W33` renormalization idempotent.
  [D2]
- `test_h_fe_round_trip`: `fe_to_homography(homography_to_fe(h,
  PC, DD), PC, DD) == h` to 1e-12 for random h, random PC/DD
  (both corner-origin and PC-centered forms); and the reverse
  composition for random reduced Fe. Machine-precision class, no
  MTP. [D6]
- `test_pure_cases_closed_form`: pure translation, pure in-plane
  rotation, pure isotropic dilation map to their closed-form
  homographies and back. [D2/D6]
- `test_direction_pinned_once`: the D2 warp direction tested in
  BOTH compositions on one synthetic case; the passing direction
  asserted, the failing direction asserted to fail (the
  sign-maze killer; run once, then frozen). [D2]

### V2 -- Synthetic warp-refit oracle (`test_hrebsd_engine.py`) -- Stage A

The core oracle (EMsoftOO's own dev recipe, `play2.f90`,
reproduced in Python): warp a real reference pattern by a known
homography with an INDEPENDENT warper
(`skimage.transform.ProjectiveTransform` + `warp`, test-only
import, D18), refit with the engine, compare `h_fit` to `h_true`.

- `test_warp_refit_small_h_480`: 480x480 synthetic-projection
  reference (V3 helper) or upsampled real pattern; a batch of
  random small homographies (translations to +-5 px, rotations to
  1 deg, strains to 2e-3); recovery metric (re-defined
  2026-09-07, spec review): the D2.5 corner displacement of the
  ERROR warp `W(h_true)^-1 . W(h_fit)` over the SR corners --
  one number in px, unit-consistent across all 8 dofs (raw
  `max|h_fit - h_true|` mixes dimensionless, px and 1/px
  components and is not used) -- <=
  `WARP_REFIT_TOL_480` (MTP; drafting seed <= 0.1 px noise-free:
  the cross-interpolator systematic ~0.01-0.05 px, theory report
  section 6 oracle 2, bounds it below, with the ~2x margin
  convention). [D2/D3/D5]
- `test_intensity_scale_invariance` (added 2026-09-07, spec
  review): one V2 case re-run with `ref -> c*ref`,
  `tar -> c'*tar` for `c, c'` powers of two (2**-10, 2**10):
  fitted h BITWISE identical (IEEE scaling is exact for
  power-of-two factors through the linear preprocessing, ZMN and
  the matched D2.1/D2.3 H/g pairing); one generic-factor case
  agrees to a machine-precision-class band. Kills any mismatched
  H/g normalization (which scales the step by an
  intensity-dependent factor and spuriously converges at the
  seed). [D2]
- `test_warp_refit_60px`: same on `nickel_ebsd_small` 60x60
  patterns, looser MTP band (qualitative regime, recorded). [D2]
- `test_convergence_metadata`: iterations < `max_iterations`,
  `converged` True, `norm_dp < min_step` on the batch; a
  deliberately unreachable `min_step=1e-12` case exits at the cap
  with `converged=False` and NaN downstream props. [D2.5-6]
- `test_seed_required_for_large_translation`: a +-15 px
  translation case fails from identity seeding and succeeds with
  the phase-XC seed (kills a dropped-seed mutant, pins D5's
  reason for existing). [D5]
- `test_dtype_ab_harness` (measurement harness, Stage A gate):
  f32 vs f64 coefficient storage on the V2 batch; verdict pinned
  per D17 and recorded below. [D17]
- `test_two_runs_bitwise`: identical inputs + chunking give
  bitwise-identical props (D16). [D16]
- `test_multiple_grains_are_restored_to_map_order` (added
  2026-09-07 at the review): a 2-grain map
  (`grain_labels = [[0, 0, 1], [0, 1, 1]]`,
  `reference = [0, 2]`) with a DIFFERENT known warp per point;
  each point's fit must be closer to ITS OWN imposed homography
  than to any other point's (a tolerance-free ordering pin), the
  grain identifiers and reference indices must match the labels,
  and `chunksize=1` must agree bitwise with `chunksize=6`. The
  drafted suite never called the engine with more than one grain
  (every call passed a `(row, col)` reference, which D11.3 defines
  as ONE implicit grain), so the D16 grain-by-grain reordering
  contract and the plan-2.5 "grain order not restored" mutant were
  unexercised. [D11/D16]
- `test_reference_state_precompute` (added 2026-09-07 at the
  review): `ReferenceState.steepest_descent`,
  `.hessian`, `.reference` and `.reference_norm` against a plain
  numpy assembly of D2.1's literal `GJ` and
  `H = (2/ref_norm^2) sum GJ GJ^T`, plus explicit
  not-close arms for the swapped-gradient and flipped-perspective
  variants. Kills the plan-2.5 mutants "swap gx/gy in GJ",
  "flip a GJ perspective-term sign" and "Hessian from target
  gradients" DETERMINISTICALLY: all three leave the zero-gradient
  fixed point (or the optimum) unchanged, so `converged is True`
  is not a reliable killer for any of them. [D2.1]
- `test_iterations_match_the_hand_built_update` (added 2026-09-07
  at the review): one and two iterations of the engine from an
  explicit seed against an independent accumulated-W IC-GN
  assembled in the test from `_interpolation` and
  `_preprocessing` only, to `UPDATE_RULE_TOL` (MTP). The
  two-iteration arm is THE killer of "re-warp the warped target"
  (D2.3's accumulated-W deviation, whose effect D2.3 puts at
  1e-4..1e-6, three orders below `WARP_REFIT_TOL_480`); the
  one-iteration arm pins the update side `W <- W . W(dp)^-1`, the
  `H dp = -g` sign and the `W33` renormalization at ENGINE level.
  [D2.3]
- `test_border_and_dead_band`: SR excludes the border and the
  cross; a planted feature there leaks into the fit by no more
  than `BORDER_LEAK_TOL` / `DEAD_BAND_LEAK_TOL` (MTP), and by
  strictly less than the same feature does when the border is NOT
  excluded (the contrast arm). **Corrected 2026-09-07 at the
  review**: the drafted arms demanded BITWISE identical fits,
  which no D3/D4-conformant implementation can deliver -- the
  D4.1 band-pass and the D3 spline prefilter are both
  whole-pattern operations, so a planted defect reaches every
  kept pixel (measured: max 0.055 intensity units of 242 through
  the band-pass alone, median 3.2e-5). [D4]
- `test_get_map_data_2d_prop_pin`: the D15.7 verification on the
  installed orix, result recorded. [D15]

### V3 -- Deformed-master end-to-end oracle (`test_hrebsd_engine.py`, `TestDeformedMaster`) -- Stage A engine half, Stage B tensor half

(File-layout deviation, recorded 2026-09-07 at the Stage A
failing-tests gate: V3, V4 and V6's pattern halves were drafted for
`test_hrebsd_deformed_master.py`, `test_hrebsd_rotation.py` and
`test_hrebsd_pc_shift.py` and are instead folded into
`test_hrebsd_engine.py`, because all three need the same
projection helper, the same cached 480 px reference and the same
engine fixtures; splitting them would either duplicate the helper
or add a shared conftest for three test classes. The analytic half
of V6 stays in `test_hrebsd_geometry.py`. Stage B's tensor halves
may still take their own files.)

The primary end-to-end oracle. Test-local projection helper
reimplements `EBSDMasterPattern.get_patterns`' geometry with an
imposed deformation: obtain detector direction cosines (kikuchipy
internals `_get_direction_cosines_*`), rotate to the crystal
frame, left-multiply `F^-1`, renormalize, interpolate the Lambert
master -- EMsoft's own `applyDeformation` mechanism
(`mod_EBSD.f90:3553, 3570`). Because the engine's geometric model
and this simulation are the same first-order model, the expected
homography is EXACT: `h_expected = fe_to_homography(R_chain(F),
PC, DD)` with the D7 frame chain minded (theory report section
6.3).

- `test_deformed_master_homography_recovery`: impose a set of F
  (pure strains ~1e-3, pure rotations to 2 deg, mixed, incl. an F
  built with sigma33 = 0 from a chosen e and C); recovered h
  within `DEFORMED_MASTER_H_TOL` (MTP) of exact; at 480x480
  detector from the shipped Ni Lambert master
  (`nickel_ebsd_master_pattern_small`, both hemispheres). [D2-D7]
- `test_deformed_master_fe_through_the_engine` (added 2026-09-07
  at the review): the SAME imposed deformations carried through
  `run_hrebsd_dic` on a 2-point map of identical projection
  centres, asserting `properties["Fe"][i].reshape(3, 3)` within
  `DEFORMED_MASTER_FE_TOL` (MTP) of the imposed reduced tensor.
  The drafted suite pinned no `Fe` VALUE anywhere at engine level
  (every `Fe` assertion was NaN, the identity, or one route
  against another), so the D15.6 wiring -- DD in binned pixels,
  the `pcy`/`pcz` roles, row-major flattening, which pattern is
  reference and which target -- survived untested end to end even
  though `homography_to_fe` itself is pinned literally in V1. The
  cases are the ASYMMETRIC ones (pure rotation, mixed), since a
  symmetric tensor is transposition-blind. [D6/D15.6]
- `test_deformed_master_strain_recovery` (Stage B): through
  `hrebsd_strain_stress` with the matching stiffness, at a
  GENERIC (low-symmetry-position) crystal orientation (added
  2026-09-07, spec review: a symmetric orientation would leave
  the crystal->sample stiffness-rotation direction unexercised;
  companion to the D9.5 C16' transpose pin): per-
  component strain error <= `DEFORMED_MASTER_STRAIN_TOL` (MTP;
  drafting seed <= 2e-4 per component at 480^2), recovered `e33`
  matches the built-in sigma33 = 0 value (the closure killer),
  `sigma33` ~ 0 self-check. [D8/D9/D10]
- `test_traction_free_vs_deviatoric_differ`: the two closures give
  measurably different e33 on a non-deviatoric F (kills a
  fallback-always mutant). [D9]

### V4 -- Pure-rotation analytic cases (`test_hrebsd_engine.py`, `TestPureRotations`; see the V3 file-layout note) -- Stage A frames, Stage B split

- `test_in_plane_rotation`: pure rotation about the detector
  normal at 0.1-5 deg (deformed-master patterns): recovered
  homography matches the closed in-plane form; polar R matches
  axis/angle to `ROTATION_TOL` (MTP; seed <= 1e-5 rad); strain
  leakage <= `ROTATION_STRAIN_LEAK` (MTP; seed <= 1e-4 up to
  5 deg). [D1/D2/D7/D8]
- `test_out_of_plane_tilt`: small rotation about detector x:
  leading terms land in h23/h32 as the exact `fe_to_homography`
  predicts (translation ~ -omega*DD, perspective ~ omega/DD);
  THE detector-frame axis/sign pin of D1.5/D7 -- the recovered
  axis must be the imposed sample-frame axis through
  `sample_to_detector`. [D1/D7]
- `test_rotation_sweep_capture_range` (recorded, not gated): sweep
  0.5-12 deg; record the largest angle the translation-only seed
  converges from (D5's recorded capture range; Ruggles 2018's
  benchmark design). [D5]
- `test_small_strain_fast_path_equality` (Stage B, measurement
  harness): polar vs small-strain paths over the sweep; the D8 MTP
  equality threshold and enable-angle recorded. [D8]

### V5 -- Si-wafer noise-floor benchmark -- Stage B [download], parts weekly

`kp.data.si_wafer()` (50x50 map, 480x480 px, single-crystal
nominally strain-free Si; `_data.py:321-440`); pooch-gated, skips
cleanly without the `tests` extra (tech-stack.md:16).

- `test_si_noise_floor_default_route`: default knobs, auto
  reference (Stage B): per-component strain std and rotation std
  recorded; the pinned regression band `SI_STRAIN_FLOOR` (MTP;
  literature context 1e-4..2e-4 class at 480 px -- recorded, not
  presumed) guards regressions thereafter. [D2-D9]
- `test_si_kam_floor`: HR-KAM noise floor in mrad recorded +
  pinned (theory expectation sigma_omega * sqrt(2/N) ~ 0.05-0.1
  mrad class). [D12]
- `test_si_gnd_floor` (Stage C): scalar GND floor recorded vs the
  4-8e12 m^-2 literature scale. [D14]
- Preprocessing/border/KAM-order sweeps (plan open questions 2-4,
  10): measurement harnesses, weekly-marked, results recorded
  below and defaults re-pinned only with dated amendments.
- `test_si_pc_shift_plane`: `hrebsd_pc_shift` residuals near-zero
  mean when the per-point PC from `extrapolate_pc` is used;
  plane-fit comparison recorded. [D13]

### V6 -- PC-shift phantom oracle (`test_hrebsd_engine.py`, `TestPcShiftPhantom`, plus the analytic half in `test_hrebsd_geometry.py`; see the V3 file-layout note) -- Stage A geometry, Stage B function

- `test_phantom_uncorrected`: identity-F deformed-master patterns
  generated on a per-point-PC grid (each pattern projected with
  its own `extrapolate_pc` PC): with the D6.2 correction DISABLED
  (private switch), the fitted homographies equal the closed-form
  phantom in the spec's OWN reference-PC-centered frame
  (D6.2: translation `gamma = delta = PC_t - PC_ref` px, scaling
  `alpha_s` = the DD ratio) to
  `PC_PHANTOM_TOL` (MTP). THE sign pin of D6.3 -- the passing
  sign set is recorded in requirements D6 with the date. **Map
  geometry pinned 2026-09-07 (Stage A failing-tests gate, review
  fix): the phantom map is `navigation_shape=(2, 3)` with
  `step_sizes=(400.0, 400.0)` on the 480 px oracle detector, NOT a
  single-row map with 1.5 unit steps. `extrapolate_pc` builds
  `d_pcy`/`d_pcz` from the ROW offset only, so a one-row map has
  `alpha_s = 1` and `gamma_y = 0` identically and `gamma_x` of
  0.02 to 0.04 px, i.e. at or below the cross-interpolator
  systematic; neither the DD-ratio orientation nor the `gamma_y`
  sign would be exercised by any assertion. MEASURED per-point
  offsets of the pinned map: `gamma_x` to -11.43 px, `gamma_y` to
  -5.37 px, `alpha_s` to 1.00806, every one of them far above the
  0.01 to 0.05 px DIC floor and inside the 24 px border budget.**
  [D6]
- `test_phantom_corrected`: correction ENABLED: `Fe = I`
  everywhere to the interpolation floor; strain phantom killed.
  [D6]
- `test_phantom_corrected_through_stage_b` (Stage B; added
  2026-09-07, spec review): the corrected per-point-PC phantom
  fed THROUGH `hrebsd_strain_stress`: strain ~ 0 everywhere at
  the interpolation floor -- kills a Stage B re-application of
  the D6.2 correction (the stored `Fe` is already corrected,
  plan 3.2), which V3's single-PC geometry cannot see. [D6/D8]
- `test_single_pc_equals_per_point_pc`: a single-PC detector +
  internal `extrapolate_pc` gives the same result as the
  explicit per-point detector built from the same model
  (D6.1 equivalence). [D6]
- `test_pc_shift_function` (Stage B): `hrebsd_pc_shift` returns
  the frozen dict keys; residuals ~ 0 on the phantom set;
  miscalibrated-PC case shows the documented nonzero-mean
  signature. [D13]

### V7 -- Constant-curvature GND + KAM oracle (`test_hrebsd_gnd.py`) -- Stage B KAM, Stage C GND

- `test_alpha_from_analytic_beta`: build a synthetic beta field
  with constant lattice curvature (e.g. omega_3(x1) = kappa*x1):
  the implemented `alpha` components equal the analytic constant
  alpha to < 1 % (pure math, no patterns; validated against the
  DERIVATION, frozen never-against-another-code rule). Single-
  component cases for EACH computed alpha entry, per assumption
  tier (reworded 2026-09-07, spec review): the three exact
  `alpha_i3` (D14.1) and the six d/dx3-neglect entries
  `alpha_i1`/`alpha_i2` (D14.2 beta route) that feed "a5"/"a9".
  The structural assertion is that NO d/dx3 derivative term is
  ever fabricated (the k = 3 derivative slots are identically
  absent/zero in the construction) and that each estimator
  consumes exactly its D14.4 documented set -- NOT that entries
  are "absent from every estimator" (a9 legitimately consumes
  all nine of its documented construction). [D14]
- `test_estimator_prefactors`: 30/10, 30/14, 30/20 literal pins;
  scalar rho = estimator formula on hand-built alpha. [D14]
- `test_antisymmetry_fix_detector_frame`: on a synthetic beta with
  distinct beta13/23 vs beta31/32, the fix replaces the latter
  with the negated former IN THE DETECTOR FRAME (order-of-
  operations pin vs rotation to sample frame); stress path
  unaffected. [D14.3]
- `test_kam_constant_curvature` (Stage B): 1st-order HR-KAM on a
  linear-along-a-grid-axis rotation field equals the
  KERNEL-DERIVED expectation `(6/8) * kappa * step` for the
  frozen all-8-neighbor mean (D12's corrected identity,
  2026-09-07 spec review: the test computes the expectation from
  the kernel offsets -- asserting `kappa * step` would fail a
  correct implementation by the fixed 25 % kernel-geometry
  factor); grain-mask and psi_max guards; mrad units pinned
  literally (a radians-vs-mrad mutant dies here). [D12]
- `test_gnd_end_to_end_curvature` (Stage C): the curvature field
  imposed through deformed-master patterns; recovered rho within
  `GND_E2E_TOL` (MTP) of the analytic density. [D14]
- `test_nan_safety`: grain boundaries, map edges, non-converged
  points produce NaN, never fabricated gradients. [D14.5]

### Stage B unit suites (`test_hrebsd_tensors.py`, `test_hrebsd_segmentation.py`, `test_hrebsd_kam.py`)

- Polar/strain/closure/Bond-rotation/voigt_stiffness/derived-map
  pins per plan 3.1 (analytic, no MTP except where marked).
- `segment_grains`: synthetic two-grain map, one-point grain,
  unindexed -1, connectivity 1 vs 2, threshold boundary case
  (>= vs > mutant), label determinism (row-major first-seen).
  [D11]
- Auto-reference: argmax IQ, tie -> lowest flat index; explicit
  `(row, col)` and per-grain index modes; `reference="auto"`
  NotImplementedError pin in Stage A, replaced in Stage B. [D11]

### Signal-method suite (`tests/test_signals/test_ebsd_hrebsd_dic.py`)

- Frozen-signature/defaults dict pin (the Phase 8 convention);
  masks polarity; lazy input -> eager output; props
  presence/dtype/shape per D15.6; xmap orientations unchanged;
  single-phase stress guard (D9.6); info message includes the
  memory note (D16). [D15/D16]

## Local-gated and weekly

- [download] V5 Si-wafer suite (above): default-suite smoke subset
  on a `[::5, ::5]` sub-grid; full 50x50 map + the sweep
  harnesses `@pytest.mark.weekly`.
- Weekly: V2 warp-refit at 960x960 (upsampled projection,
  Ruggles-2018-scale check, recorded); the D3 bicubic-vs-quintic
  A/B re-run; full-map performance rows (Performance table).
- Local oldest-matrix run once per stage (plan section 1 recipe,
  incl. the scikit-image 0.21.0 pin and the test_signals paths),
  result recorded below -- the recorded oldest-floor gate (the
  on-push CI oldest job is an extra signal only, plan section 1).

## Requirement-to-test mapping

| Decision | Requirement (short) | Killer / evidence | Stage |
|---|---|---|---|
| D1 | pixel unit system, frames, PC-centered coords | V4 out-of-plane axis/sign pin; V1 pure cases; mutation list (unit mutants) | A |
| D2 | IC-GN loop, accumulated-W, corner norm, NaN contract | V2 recovery + metadata; V1 direction pin; convergence sweep | A |
| D3 | bicubic numba kernel + order decision | V0 equality + `.py_func`; D3 A/B recorded | A |
| D4 | band-pass/window/border/dead-band | V2 border test; V5 sweeps | A/B |
| D5 | phase-XC seed | V2 seed-required test; V4 capture range | A |
| D6 | h<->Fe, per-point PC/DD, correction-before-conversion, sign pins | V1 round trip; V6 phantom pair; V3 exactness | A |
| D7 | frame chain | V4 axis pin; V3 tensor recovery | A/B |
| D8 | polar split, Biot, fast path MTP | V4 leakage; V3 strain; fast-path harness | B |
| D9 | closure + stiffness + Bond rotation | V3 e33/sigma33; unit pins; closure-differ test | B |
| D10 | stress + derived maps | derived-map identities; sigma33 self-check | B |
| D11 | segmentation + references | segmentation suite; auto-reference pins | B |
| D12 | HR-KAM freeze | V7 KAM identity; V5 KAM floor | B |
| D13 | PC-shift analysis | V6 function tests; V5 plane test | B |
| D14 | GND conventions, antisymmetry, estimators | V7 suite; prefactor pins; V5 GND floor | C |
| D15 | API/naming/props/get_map_data | signature pins; props pins; V2 get_map_data pin | A-C |
| D16 | dask pairing, determinism, memory note | bitwise test; lazy test; info-message pin | A |
| D17 | f32/f64 discipline | V2 dtype A/B, verdict recorded | A |
| D18 | deps/licensing | conventions review; import audit -- `test_hrebsd_engine.py::TestImportAudit` walks every `_hrebsd/` module source for `skimage.transform` and for `sympy` (added 2026-09-07 at the review: the drafted suite left this to the reviewer's eye alone); skimage >= 0.21.0 tested floor via the oldest-matrix recipe | A-C |

## Performance (recorded baselines, never gates -- D16)

| measurement | recipe | recorded value |
|---|---|---|
| Stage A engine, pat/s, Si-wafer route 480x480, default knobs, 8 workers | fixed-seed timed run, warm numba caches, idle machine | measure at Stage A/B gate |
| Stage A engine, pat/s, `nickel_ebsd_large` 60x60 | same recipe | measure at Stage A gate |
| per-grain precompute cost + resident MB at 480x480 | instrumented run | measure at Stage A gate |
| bicubic vs quintic accuracy/speed (D3) | V2 A/B | record at Stage A gate |
| f32 vs f64 storage accuracy/speed (D17) | V2 dtype A/B | record at Stage A gate |
| Stage B tensor chain + KAM cost on the 50x50 Si map | timed run | measure at Stage B gate |
| Stage C GND cost | timed run | measure at Stage C gate |

## Manual

- User review of the plan-0 constitution amendments at the spec
  commit (the spec-stage verify step; approval gate per the
  autonomous-mode memory is waived for spherical phases only, so
  HREBSD stage boundaries surface to the user).
- Tutorial renders + nbval passes locally (Stage C); map gallery
  eyeballed against the OpenXY "usual maps" checklist, carried
  INLINE here (2026-09-07, spec review: the theory-report
  scratchpad does not survive the session, so the checklist a
  gate consumes must live in the spec). Each item covered or
  dispositioned:
  - strain components e11..e33 -- D8/D15.6 `strain` prop,
    tutorial gallery.
  - stress: von Mises, principal s1/s2/s3 (+ hydrostatic) --
    D10.
  - rotation maps (vector components/magnitude) -- D8;
    misorientation-kernel map = HR-KAM -- D12.
  - dislocation density, total (3/5/9-component) -- D14.
  - split (per-type) dislocation density -- OUT of v1, deferred
    with the L1 split (Scope; plan open question 7).
  - tetragonality -- OUT of v1 with its revisit path recorded in
    Scope (derivable from the stored `Fe` prop).
  - Burgers-vector maps -- OUT of v1, part of the per-type split
    deferral (Scope).
  - misorientation maps, IPF, IQ -- existing kikuchipy/orix
    functionality, demonstrated in the tutorial, nothing new
    built.
  - ROI shifts -- the DIC analogue is D13's
    translation/model/residual maps.
  - SSE/fit-metric maps -- the D10 quality maps
    (`residual`/`num_iterations`/`norm_dp`/`converged`).
- One end-user smoke run from a notebook on the Si wafer:
  `hrebsd_dic` + `hrebsd_strain_stress` + maps, eyeballing that a
  nominally strain-free wafer shows noise, not structure.

## Definition of done

- Per stage: failing tests committed first; implementation;
  adversarial review (both reviewers + the stage's mutation list)
  + fixes; `pre-commit run --files` clean; coverage 100 % of the
  stage's `_hrebsd/` modules with the command output recorded;
  full existing suite green (`uv run pytest tests -n 4`, recorded);
  local oldest-matrix run recorded; signed commits pushed to
  `origin/hrebsd-dic`; roadmap stage boxes ticked. NO PR into
  `develop` (branch policy) -- "PR opened/merged" gates are
  REPLACED by "pushed to origin/hrebsd-dic + roadmap ticked".
- **The three spec documents submitted to adversarial review**
  (fidelity/theory + conventions/integration) with findings folded
  by the fixer before the spec commit is finalized -- the
  spec-stage workflow of the approved plan; the disposition table
  is appended to plan.md (Phase 8 precedent).
- Every `MTP` placeholder replaced by a dated measured value in
  Recorded results (recipe + machine ID); the D2/D6 sign pins, D3
  order decision and D17 dtype verdict recorded in requirements
  with dates.
- Plan section 6 open questions resolved or their conservative
  defaults confirmed by the named measurements, each with a dated
  record.
- All six public names + the `EBSD.hrebsd_dic` method exported,
  documented, in the API reference; tutorial registered and
  nbval-wired (Stage C); CHANGELOG entries present (fork-only
  wording per plan 4.4).
- Si-wafer noise floors (strain, rotation, KAM, GND) recorded --
  the feature's headline honesty numbers, quoted in the tutorial
  at 480x480 only, with 60x60 results labeled qualitative.

## Recorded results

Append-only ledger. Every entry is dated, names the machine (CPU
class + OS at minimum; the drafting machine is the 20-core Windows
11 laptop of the spherical phases), the exact recipe or script
name with its command, and which `MTP` placeholder or decision it
fills. Refutations of frozen decisions are recorded here AND
amended in requirements.md with the same date (Phase 8/10
precedent).

### 2026-09-07 (drafting)

No measurements: the spec stage was read-only analysis plus
`specs/` writes (requirements D19). All EMsoftOO/OpenXY/kikuchipy
behaviour cited in the spec was verified by line reading only
(emsoftoo_report.md, kikuchipy_surface_report.md,
theory_maps_report.md; scratchpad reports do not survive the
session -- every load-bearing citation is carried inline in
requirements.md). This section is filled at each stage's
failing-tests gate (placeholder inventory), implementation gate
(measurements + pins, with recipes), and review gate
(re-measurements), each in its own dated subsection.

### 2026-09-07 (Stage A failing-tests gate, adversarial review fixes)

Machine: the 20-core Windows 11 laptop of the spherical phases,
Python 3.12 in `.venv`. Every number below comes from a library
measurement that needs NO `_hrebsd` implementation, so it stands
before the engine lands.

1. **D1 half-pixel correction (requirements D1.1/D1.3, amended
   with this date).** Recipe:
   `tests/test_indexing/test_hrebsd_engine.py`,
   `TestPcCentredFrame::test_pc_centred_frame_matches_kikuchipy_geometry`,
   both parametrizations, currently PASSING. Mapping
   `EBSDDetector.sample_to_detector` into the spec's y-down
   detector frame and scaling every direction cosine to
   `z = DD_px` reproduces `col + 0.5 - PCx_px` and
   `row + 0.5 - PCy_px` to under 1e-9 px (measured worst 1.8e-14 px
   over a 40 by 60 detector). The literal D1.3 form
   `xi = col - PCx_px` is refuted; requirements D1.1 and D1.3 carry
   the dated correction and the constant `DETECTOR_Y_FLIP` records
   the y-axis relation.
2. **V6 phantom map geometry (see V6 above).** Recipe:
   `EBSDDetector.extrapolate_pc(pc_indices=[0, 0],
   navigation_shape=..., step_sizes=...)` on the oracle detector
   (480 px, binning 1, px_size 70, `pc = (0.4210, 0.5794,
   0.5049)`, sample tilt 70 deg). Measured per-point offsets in
   binned px, `(gamma_x, gamma_y, alpha_s - 1)`: the drafted
   `(1, 3)` map with 1.5 unit steps gives at best
   `(-0.043, 0.0, 0.0)`; the pinned `(2, 3)` map with 400.0 unit
   steps gives up to `(-11.43, -5.37, 8.06e-3)`. The drafted
   closed form (`alpha_s = DD_target / DD_reference`,
   `gamma = PC_target - PC_reference`) was additionally checked
   directly against the projected patterns: warping the reference
   by it and differencing against the pattern projected at that
   point's own PC minimizes the residual exactly at the drafted
   parameters (median residual 0.171 at the closed form versus
   0.246 with `gamma_x` off by 0.05 px, on a 242 intensity scale),
   and the FLIPPED DD ratio raises the worst residual from 12.0 to
   71.0. The remaining residual is the skimage warper's own
   interpolation error at the band edges. This does NOT yet pin
   D6.3: the sign set is pinned by the fitted homographies at the
   implementation gate.
3. **IC-GN update-rule separations** (the band
   `UPDATE_RULE_TOL` must clear). Recipe: the hand-built
   accumulated-W IC-GN of `test_hrebsd_engine.py`
   (`hand_built_icgn`) run with scipy stand-ins for the
   not-yet-written kernel (`spline_filter` plus
   `map_coordinates(order=3, prefilter=False, mode="mirror")`,
   finite-difference gradient planes), on the 480 px oracle
   reference warped by the seed-17 half-budget homography and seeded
   0.3 px off. The loop recovers `h_true` to 1.8e-3 px, which is the
   cross-interpolator systematic against skimage's warper and
   confirms the D2 signs as drafted. Mutant separations from the
   correct loop, in the corner-displacement metric: composing the
   update as `W(dp)^-1 . W` 2.5e-3 px at one iteration and 8.2e-5 px
   at two; a flipped perspective-term sign 4.2e-3 px; warping the
   WARPED target 1.7e-3 px at two iterations and 0 at one (the two
   schemes coincide on the first); `H dp = +g` 0.77 px; swapped
   `gx`/`gy` 0.76 px. So the two smallest are three to four orders
   above the expected engine-versus-hand-built agreement, and the
   implementation gate must pin `UPDATE_RULE_TOL` at or below
   1e-6 px for them to die.
4. **Test-file layout deviation**: recorded in the V3 heading
   above.
5. **MTP placeholder inventory after the review fixes**
   (`test_hrebsd_engine.py`): `WARP_REFIT_TOL_480`,
   `WARP_REFIT_TOL_60`, `ROTATION_TOL_RAD`,
   `INTENSITY_SCALE_GENERIC_TOL`, `DEFORMED_MASTER_H_TOL`,
   `DEFORMED_MASTER_FE_TOL`, `PC_PHANTOM_TOL`,
   `PC_PHANTOM_FE_TOL`, `PC_PHANTOM_TRANSLATION_TOL`,
   `PC_ROUTE_EQUIVALENCE_TOL`, `UPDATE_RULE_TOL`,
   `BORDER_LEAK_TOL`, `DEAD_BAND_LEAK_TOL`;
   (`test_hrebsd_interpolation.py`): `KERNEL_F32_TOL`,
   `GRADIENT_FINITE_DIFFERENCE_TOL`, `MIRROR_OVERSHOOT_TOL`. All
   are `None` and raise the "unfilled MEASURED-THEN-PINNED
   placeholder" assertion until the implementation gate fills them
   with dated values and recipes. `WARP_REFIT_TOL_480` and
   `ROTATION_TOL_RAD` were live drafting seeds (0.1 px, 1e-5 rad)
   and were emptied at this review: a seed 2 to 10 times above the
   expected achievable error is an acceptance gate that passes a
   mediocre implementation silently.
