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

**Stage A MTP pins, ALL FILLED 2026-09-07** (implementation gate;
the measurements, recipes, margins and machine are in Recorded
results entries 1 to 3 below, and the drafting seeds quoted in the
oracle descriptions that follow are superseded by these):
`KERNEL_F32_TOL` 5e-8, `GRADIENT_FINITE_DIFFERENCE_TOL` 3e-8,
`MIRROR_OVERSHOOT_TOL` 5e-3 (fraction of span),
`WARP_REFIT_TOL_480` 0.025 px, `WARP_REFIT_TOL_60` 0.13 px,
`INTENSITY_SCALE_GENERIC_TOL` 6e-9 px, `DEFORMED_MASTER_H_TOL`
0.023 px, `DEFORMED_MASTER_FE_TOL` 3.1e-5, `ROTATION_TOL_RAD`
1.7e-5 rad, `PC_PHANTOM_TOL` 0.014 px, `PC_PHANTOM_FE_TOL` 3.4e-5,
`PC_PHANTOM_TRANSLATION_TOL` 0.0022 px,
`PC_ROUTE_EQUIVALENCE_TOL` 1e-12, `UPDATE_RULE_TOL` 1e-13 px,
`BORDER_LEAK_TOL` 0.18 px, `DEAD_BAND_LEAK_TOL` 0.18 px. One more
was ADDED and filled at the adversarial review, Recorded results
entry 19: `WINDOW_REFIT_TOL_480` 0.092 px, the D4.3 window knob's
own warp-refit band, which no drafted test measured. Two
drafting seeds were refuted by measurement and are amended in
requirements Context with the same date: the 0.1 px warp-recovery
seed (8x looser than achievable) and the "up to 5 deg"
pure-rotation seed (the measured capture range is 2.0 deg).

**Stage B MTP pins, ALL UNFILLED at the failing-tests gate**
(2026-09-07; filled at the Stage B implementation gate, with the
recipes and the machine recorded below). The inventory, with the
module each lives in: `REDUCED_CLOSURE_TOL`, `SIGMA33_TOL`,
`SMALL_STRAIN_EQUALITY_TOL`, `SMALL_STRAIN_ENABLE_ANGLE_DEG`
(`test_hrebsd_tensors.py`); `DEFORMED_MASTER_STRAIN_TOL`,
`DEFORMED_MASTER_SIGMA33_TOL` (`test_hrebsd_deformed_master.py`);
`SI_STRAIN_FLOOR`, `SI_ROTATION_FLOOR`, `SI_KAM_FLOOR`,
`SI_PC_RESIDUAL_MEAN_TOL` (`test_hrebsd_si.py`, [download] gated).
No Stage B tolerance is frozen: `ALGEBRA_TOL` (1e-12) and
`SOLVER_TOL` (1e-10) are the algebraic identity bands of analytic
constructions, and `FAST_PATH_CRITERION` (1e-6) is the
requirements D8 criterion itself, not a measurement.
For calibration when the implementation gate fills them, the four
tensor-module placeholders were measured on a reference chain
built at the Stage B failing-tests review (Recorded results, Stage
B failing-tests gate): `REDUCED_CLOSURE_TOL` 1.1655e-06 (strain),
9.4132e-07 (e33), 9.5697e-07 (beta); `SIGMA33_TOL` 3.0639e-04 GPa;
`SMALL_STRAIN_EQUALITY_TOL` 5.889e-04 over the drafted sweep;
`SMALL_STRAIN_ENABLE_ANGLE_DEG` 0.0779; and the pattern-level pair
`DEFORMED_MASTER_STRAIN_TOL` 9.4045e-06,
`DEFORMED_MASTER_SIGMA33_TOL` 1.8338e-04 GPa. Those numbers are
NOT pins: they are the scales a correct implementation should
reproduce, and the gate measures its own.

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

**Stage B file-layout deviations, recorded 2026-09-07 at the Stage
B failing-tests gate (adversarial review fixes).** Stage B's tensor
halves did take their own files, and six names moved with them:

| named here | delivered as |
|---|---|
| V3 `test_deformed_master_strain_recovery` (this block) | `test_hrebsd_deformed_master.py::TestDeformedMaster::test_deformed_master_strain_recovery` -- a NEW module, not `test_hrebsd_engine.py`, so that the Stage A regression file carries no Stage B failure; the projection helper, the frame matrices and the detector are duplicated there because pytest imports test modules by path |
| V3 `test_traction_free_vs_deviatoric_differ` | `test_hrebsd_tensors.py::TestClosure::test_the_two_closures_differ_on_a_non_deviatoric_tensor` (analytic) and `test_hrebsd_deformed_master.py::TestDeformedMaster::test_the_two_closures_differ_through_the_patterns` (patterns) |
| V4 `test_small_strain_fast_path_equality` | `test_hrebsd_tensors.py::TestSmallStrainFastPath::test_equality_with_the_polar_path_is_measured`, an ANALYTIC sweep -- see the requirements D8 measuring-test correction of the same date |
| V5 the whole suite | `test_hrebsd_si.py` ([download] gated, skips with a message naming `kp.data.si_wafer(allow_download=True)`) |
| V6 `test_phantom_corrected_through_stage_b` | `test_hrebsd_pc_shift.py::TestPhantomThroughTheChain::test_the_corrected_phantom_carries_no_strain` |
| V7 the KAM half (named for `test_hrebsd_gnd.py`) | `test_hrebsd_kam.py` -- the GND half keeps `test_hrebsd_gnd.py` for Stage C |


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
  leakage <= `ROTATION_STRAIN_LEAK` (MTP; ~~seed <= 1e-4 up to
  5 deg~~ **SEED REFUTED 2026-09-07 at the Stage B failing-tests
  gate; requirements D8 amended with the same date**). A pure
  rotation leaks nothing through the polar SPLIT (measured
  `|U - I|max` = 6.7e-16 at 5 deg) but does NOT come back at zero
  through the CHAIN: the reduction by `Fe33` is an isotropic
  dilatation and the D9 closure supplies an isotropic part of its
  own. MEASURED reported `|strain|max` for a pure rotation about
  the detector normal on the 480 px oracle geometry: 1.0154e-04
  deviatoric / 7.2762e-05 traction free at 1 deg, 4.0614e-04 /
  2.9104e-04 at 2 deg, 2.5380e-03 / 1.8187e-03 at 5 deg -- the
  deviatoric value is EXACTLY `2/3` of `sym(R - I)` by algebra and
  the traction-free one 0.478 of it. The seed was 25x too tight at
  5 deg. The band stays MTP; its recipe is this sweep. The same
  measurement refuted the drafted Stage B arm which asserted the
  reported strain of a two degree rotation to be under a tenth of
  that leak (6.7x over) -- see the D8 amendment. [D1/D2/D7/D8]
- `test_out_of_plane_tilt`: small rotation about detector x:
  leading terms land in h23/h32 as the exact `fe_to_homography`
  predicts (translation ~ -omega*DD, perspective ~ omega/DD);
  THE detector-frame axis/sign pin of D1.5/D7 -- the recovered
  axis must be the imposed sample-frame axis through
  `sample_to_detector`. [D1/D7]
- `test_rotation_sweep_capture_range` (recorded, not gated): sweep
  0.5-12 deg; record the largest angle the translation-only seed
  converges from (D5's recorded capture range; Ruggles 2018's
  benchmark design). **RECORDED 2026-09-07 (implementation gate):
  2.0 deg** at 480x480 with the frozen band-pass; from the exact
  seed the basin itself reaches 3.0 deg and fails at 4.0. The
  in-plane cases of `test_in_plane_rotation` were re-pinned from
  0.1-5 deg to 0.1/1.0/2.0 deg for the same reason, and the "up to
  5 deg" acceptance seed is amended in requirements Context. Full
  table with seeds and pair ZNCC in Recorded results entry 6.
  **RECORDED AS A FUNCTION OF `filter_cutoffs` 2026-09-07 (Stage A
  adversarial review, Recorded results entry 21): 2.0 deg at the
  frozen `(0.05, None)` default and 4.0 deg at `(None, None)`, with
  the 2.0 deg error ten times smaller unfiltered.** The capture
  range is a property of the band-pass default, not of the frozen
  D2/D4/D5 design, and the requirements Context clause claiming
  otherwise is struck with that date. The default is not re-pinned
  here: plan open question 10 owns it and the V5 Si-wafer benchmark
  resolves it, since a high-pass can only remove signal on
  noise-free oracles. Entry 6's seed COLUMN does not reproduce and
  is corrected by entry 22; every other column of it does. [D5]
- `test_small_strain_fast_path_equality` (Stage B, measurement
  harness): polar vs small-strain paths over the sweep; the D8 MTP
  equality threshold and enable-angle recorded. [D8]

### V5 -- Si-wafer noise-floor benchmark (`tests/test_indexing/test_hrebsd_si.py`) -- Stage B [download], parts weekly

`kp.data.si_wafer()` (50x50 map, 480x480 px, single-crystal
nominally strain-free Si; `_data.py:321-440`); pooch-gated, skips
cleanly without the `tests` extra (tech-stack.md:16).

**Delivered 2026-09-07 at the Stage B failing-tests gate** (added
there: the drafted Stage B commit shipped none of V5) as
`tests/test_indexing/test_hrebsd_si.py`. It downloads NOTHING: a
missing cache is a skip naming
`kp.data.si_wafer(allow_download=True, lazy=True)`, so the 311 MB
transfer is always a deliberate act. It is therefore UNEXECUTED at
the failing-tests gate (verified `9 skipped`, Recorded results
entry 32) and its four placeholders are filled at the
implementation gate, which fetches the dataset once. Its default
route takes the DEVIATORIC closure and a nominal single
orientation, so the recorded strain/rotation/KAM floors do not
depend on indexing the wafer first; the traction-free stress arm,
which does depend on the orientation, is weekly and recorded, not
gated. MTP placeholders: `SI_STRAIN_FLOOR`, `SI_ROTATION_FLOOR`,
`SI_KAM_FLOOR`, `SI_PC_RESIDUAL_MEAN_TOL`.

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
- `test_corrected_fe_on_a_deformed_per_point_pc_map` (ADDED
  2026-09-07 at the Stage A adversarial review, Recorded results
  entry 14): the SAME per-point-PC map with a real deformation --
  a 1 degree out-of-plane tilt -- imposed on every target, asserting
  `Fe` to `DEFORMED_MASTER_FE_TOL`. This is the oracle neither
  `test_phantom_corrected` (which imposes `Fe = I`, so `h_corr = 0`
  and every conversion frame gives the identity) nor V3's
  `test_deformed_master_fe_through_the_engine` (whose every point
  "shares one projection centre", so `PC_rel = 0` and
  `DD_t = DD_r`) can supply, and it is what refuted the drafted
  D6.2 conversion frame: 3.9967e-04 with it, 2.2255e-05 without.
  [D6.2]
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

### V7 -- Constant-curvature GND + KAM oracle (`test_hrebsd_gnd.py` for the Stage C GND half; `test_hrebsd_kam.py` for the Stage B KAM half, file-layout deviation recorded 2026-09-07 at the Stage B failing-tests gate) -- Stage B KAM, Stage C GND

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
  literally (a radians-vs-mrad mutant dies here).
  **CONDITIONING PIN ADDED 2026-09-07 (Stage B failing-tests
  gate, Recorded results entry 28; requirements D12 amended with
  the same date):** the oracle's pair angle must be
  `arctan2(||skew||/2, (tr - 1)/2)` and NOT `arccos((tr - 1)/2)`,
  which at this 1e-4 rad scale sits 2.6221e-09 relative from the
  closed form and made the module's own two tolerance families
  mutually unsatisfiable. A library test guards it. [D12]
- `test_gnd_end_to_end_curvature` (Stage C): the curvature field
  imposed through deformed-master patterns; recovered rho within
  `GND_E2E_TOL` (MTP) of the analytic density. [D14]
- `test_nan_safety`: grain boundaries, map edges, non-converged
  points produce NaN, never fabricated gradients. [D14.5]

### Stage B unit suites (`test_hrebsd_tensors.py`, `test_hrebsd_stiffness.py`, `test_hrebsd_segmentation.py`, `test_hrebsd_kam.py`, `test_hrebsd_pc_shift.py`, `test_hrebsd_deformed_master.py`, `test_hrebsd_si.py`)

- Polar/strain/closure/Bond-rotation/voigt_stiffness/derived-map
  pins per plan 3.1 (analytic, no MTP except where marked).
- `segment_grains`: synthetic two-grain map, one-point grain,
  unindexed -1, connectivity 1 vs 2 INCLUDING a three-column
  8-neighbour wraparound case, a single-row and a single-COLUMN
  shape pin, a phase boundary, threshold boundary case
  (>= vs > mutant), label determinism (row-major first-seen).
  [D11]
- Auto-reference: argmax IQ, tie -> lowest flat index; explicit
  `(row, col)` and per-grain index modes; `reference="auto"`
  NotImplementedError pin in Stage A, replaced in Stage B. **The
  two Stage A pins in `test_hrebsd_engine.py` pass VACUOUSLY from
  the Stage B failing-tests commit onward (Recorded results entry
  33) and are DELETED at the Stage B implementation gate; the
  pin in `test_ebsd_hrebsd_dic.py` was replaced in that commit.**
  [D11]

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
  It is a PER-STAGE gate, not a Stage B or weekly item: Recorded
  results entry 12 classified it as the latter, which plan.md
  section 1 and the Definition of done both contradict, and Stage
  A's run is discharged and recorded at entry 24 (2026-09-07,
  adversarial review).

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
| Stage A engine, pat/s, 480x480 oracle route, default knobs, 8 workers | `measure_perf.py`, 40 one-grain patterns, warm numba caches, best of 3 (2026-09-07, machine A) | **23.96 pat/s** (1.670 s, all 40 converged, mean 4.65 iterations) |
| Stage A engine, pat/s, 60x60 (`nickel_ebsd_small` reference, warped batch) | same recipe, 100 patterns (2026-09-07, machine A) | **600.4 pat/s** (0.167 s, all 100 converged, mean 3.67 iterations) |
| per-grain precompute cost + resident MB at 480x480 | `ReferenceState` construction, median of 3 (2026-09-07, machine A) | **0.0402 s, 16.54 MB** (186624 subregion px, f32 coefficients) |
| bicubic vs quintic accuracy/speed (D3) | V2 A/B, entry 8 below | bicubic 0.0105 px vs quintic 0.0126 px at 3.31x cost -- **bicubic kept** |
| f32 vs f64 storage accuracy/speed (D17) | V2 dtype A/B, entry 9 below | degradation 4.5e-08 of the f64 error, 0.92 MB saved -- **f32 confirmed** |
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

### 2026-09-07 (Stage A implementation gate, measurement agent)

**Machine A** (the machine ID every number below carries): the
20-core Windows 11 laptop of the spherical phases, Intel64 Family 6
Model 186 (Raptor Lake) with 20 logical cores, Windows 11 build
26200, `.venv` Python **3.13.12** (the failing-tests-gate entry
above says 3.12; the interpreter actually in `.venv` reports
3.13.12, recorded here rather than corrected there, this ledger
being append only), numpy 2.4.6, scipy 1.17.1, numba 0.65.1,
scikit-image 0.26.0, orix 0.14.2, dask 2026.3.0. Idle machine, warm
numba caches.

Everything below discharges the plan section 2.3 measurement debt
and the plan section 6 open questions 1, 9 and 11 (and part of 2).
All tolerance numbers come from ONE reproducible run of the suite's
own measuring tests, and the run was repeated to confirm the values
are BITWISE identical between runs (they are, including under
randomized test order).

1. **MTP pin recipe, common to entries 2 and 3.** Command:

   ```
   .venv/Scripts/python.exe -m pytest \
     tests/test_indexing/test_hrebsd_interpolation.py \
     tests/test_indexing/test_hrebsd_engine.py -q
   ```

   run with a throwaway scratchpad pytest plugin
   (`hrebsd_measure/measure_plugin.py`, never in the repo) which
   sets every unfilled `None` placeholder to `+inf` and records what
   `assert_within` was handed, so ONE run reports the worst measured
   value of every placeholder including the ones hidden behind an
   earlier failure in the same test. Without it each failing test
   reports only its FIRST placeholder: `test_in_plane_rotation`, for
   instance, hides its `DEFORMED_MASTER_H_TOL` arm behind
   `ROTATION_TOL_RAD`. Margin convention, applied to every pin
   below: 2x the measured worst case unless stated otherwise.

2. **Interpolation placeholders (V0,
   `test_hrebsd_interpolation.py`).**

   | constant | measured worst | pinned | margin |
   |---|---|---|---|
   | `KERNEL_F32_TOL` | 2.4377569405821475e-08 | 5e-8 | 2.05x |
   | `GRADIENT_FINITE_DIFFERENCE_TOL` | 1.3660179594901036e-08 | 3e-8 | 2.20x |
   | `MIRROR_OVERSHOOT_TOL` | 0.0 (see below) | 5e-3 | see below |

   `KERNEL_F32_TOL` is the f32-ULP class the D17 A/B predicted, and
   is measured on the same run as that A/B (entry 9).
   `GRADIENT_FINITE_DIFFERENCE_TOL` is the central difference
   truncation of the 1e-4 px step, not the kernel's own error.

   `MIRROR_OVERSHOOT_TOL` needed different treatment and is the one
   pin NOT at 2x its own measurement: the test's frozen
   configuration (seed 71, 48x48, the horizontal line at mid height)
   overshoots the pattern's own range by EXACTLY 0.0 spans, and zero
   carries no multiplicative margin. Recipe for the pin: the same
   class swept wider in the scratchpad (`measure_extras.py mirror`),
   20 seeds by both `SHAPES` by both axes, 80 cases, worst overshoot
   **2.1164e-03 spans**; pinned at 5e-3, i.e. 2.4x that class worst.
   It still kills what it exists to kill by five orders: a cubic
   polynomial fitted to the four edge samples and extrapolated 12 px
   past the same edges (EMsoftOO's `extrap`, `mod_DIC.f90:50-51`)
   overshoots by up to **3.175e+02 spans** over the same 40 cases. A
   deliberate step image, the worst case a cubic B-spline has,
   overshoots 1.08e-01 spans, so the band also sits 20x under the
   interpolant's own overshoot class and cannot be passed by
   accident.

3. **Engine placeholders (V2, V3, V4, V6,
   `test_hrebsd_engine.py`).** Every value in binned pixels except
   where noted; the metric is the D2.5 corner displacement of the
   error warp.

   | constant | measured worst | pinned | measuring test(s), all arms |
   |---|---|---|---|
   | `WARP_REFIT_TOL_480` | 0.012439859159007909 | 0.025 | `test_warp_refit_small_h_480` (0.00820 seed 0, 0.01244 seed 1), `test_direction_pinned_once` (0.00502), `test_seed_required_for_large_translation` (0.01211) |
   | `WARP_REFIT_TOL_60` | 0.06506790630704162 | 0.13 | `test_warp_refit_60px` |
   | `INTENSITY_SCALE_GENERIC_TOL` | 3.0575906516707247e-09 | 6e-9 | `test_intensity_scale_invariance` |
   | `DEFORMED_MASTER_H_TOL` | 0.011302529677402013 | 0.023 | `test_deformed_master_homography_recovery` (0.00404, 0.00348, 0.00424, 0.00308), `test_in_plane_rotation` (0.00624, 0.00397, 0.01130), `test_out_of_plane_tilt` (0.00399, 0.00903), `test_sample_frame_axis_is_pinned` (0.00595) |
   | `DEFORMED_MASTER_FE_TOL` | 1.5474267014765897e-05 | 3.1e-5 | `test_deformed_master_fe_through_the_engine` |
   | `ROTATION_TOL_RAD` | 8.306247905485228e-06 rad | 1.7e-5 | `test_in_plane_rotation` (8.31e-06 at 0.1 deg, 3.36e-06 at 1.0, 2.36e-06 at 2.0) |
   | `PC_PHANTOM_TOL` | 0.007014950695559439 | 0.014 | `test_phantom_uncorrected` |
   | `PC_PHANTOM_FE_TOL` | 1.6955787696912304e-05 | 3.4e-5 | `test_phantom_corrected` |
   | `PC_PHANTOM_TRANSLATION_TOL` | 0.0010834565488911374 | 0.0022 | `test_raw_homography_is_stored_uncorrected` |
   | `PC_ROUTE_EQUIVALENCE_TOL` | 0.0 (bitwise) | 1e-12 (`ALGEBRA_TOL`) | `test_single_pc_equals_per_point_pc` |
   | `UPDATE_RULE_TOL` | 4.0194366942304644e-14 | 1e-13 | `test_iterations_match_the_hand_built_update` (0.0 at one iteration, 4.02e-14 at two) |
   | `BORDER_LEAK_TOL` | 0.08794948499121773 | 0.18 | `test_border_keeps_a_planted_defect_out` |
   | `DEAD_BAND_LEAK_TOL` | 0.08849768737444721 | 0.18 | `test_dead_band_keeps_a_planted_cross_out` |

   Two of these are not a plain 2x of their own measurement, and the
   test file says why at each constant:

   - `PC_ROUTE_EQUIVALENCE_TOL` measures EXACTLY 0.0, a bitwise
     agreement, because the internal route calls the same
     `extrapolate_pc` with the same anchor and step sizes and hands
     the engine an identical projection centre array. Literal zero
     is not pinned: a change in the ORDER of that same arithmetic
     would fail a correct implementation on a last-bit difference.
     The band is the module's frozen machine-precision one, 1e-12,
     and a real route divergence still dies by orders (a per-point
     projection centre wrong by 1e-6 px moves `Fe` by about 1e-9).
   - `UPDATE_RULE_TOL` is pinned at 2.5x its measurement, what
     matters here being the distance to the MUTANTS rather than the
     margin over the measurement: 1e-13 sits eight orders under the
     tightest separation measured at the failing-tests gate
     (8.2e-05 px, the wrong-side composition at two iterations) and
     ten under the smallest one-iteration separation (2.5e-03 px).
     The engine agreeing with the hand-built loop to 4e-14 px is
     itself a finding: the D2.3 accumulated-W deviation is genuinely
     implemented, the warp-of-warp scheme it replaces differing by
     1.7e-03 px at two iterations.

   Both emptied drafting seeds are now refuted with numbers and the
   refutation is amended into requirements Context with this date:
   `WARP_REFIT_TOL_480`'s 0.1 px seed was 8x looser than achievable,
   and `ROTATION_TOL_RAD`'s 1e-5 rad seed sat only 1.2x above the
   achievable error, i.e. no margin at all.

4. **Gate outcome.** With the pins above:

   ```
   .venv/Scripts/python.exe -m pytest \
     tests/test_indexing/test_hrebsd_interpolation.py \
     tests/test_indexing/test_hrebsd_homography.py \
     tests/test_indexing/test_hrebsd_geometry.py \
     tests/test_indexing/test_hrebsd_engine.py \
     tests/test_signals/test_ebsd_hrebsd_dic.py -q
   -> 183 passed, 1 skipped (the weekly capture-range sweep), 14.6 s
   ```

   `ruff format --check` and `ruff check` clean on both edited test
   files. The weekly sweep itself was run separately
   (`-q --weekly -k rotation_sweep_capture_range`) and PASSES, its
   numbers in entry 6.

   Full existing suite, same machine and interpreter:

   ```
   .venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_data
   -> 4257 passed, 797 skipped, 3 rerun, 206 s
   ```

   Zero failures, so HREBSD perturbs no spherical or upstream test.
   One observation recorded because a `-x` run tripped over it
   first: `tests/test_simulations/test_kikuchi_pattern_simulator.py
   ::TestCalculateMasterPattern::test_shape` is FLAKY on this
   machine, failing about one run in three at its
   `np.allclose(mp.data[0], mp.data[1], atol=1e-4)` hemisphere
   comparison and exhausting its own `@pytest.mark.flaky(reruns=5)`
   sometimes. It is upstream code untouched by this branch
   (`git diff HEAD -- src/kikuchipy/simulations tests/test_simulations`
   is empty) and upstream already marks it flaky, so it is NOT an
   HREBSD regression; it is logged here so a future red run on this
   machine is not misread as one.

5. **D6.3 sign pin (requirements D6.3, amended with this date).**
   The DRAFTED sign set passes unchanged:
   `alpha_s = DD_target / DD_reference`,
   `gamma = PC_target - PC_reference` in binned pixels,
   `W_phantom = [[alpha_s, 0, gamma_x], [0, alpha_s, gamma_y],
   [0, 0, 1]]`, correction composed as `W_corr = W_phantom^-1 . W`.
   Evidence: on the `(2, 3)` 400.0-step phantom map the fitted
   uncorrected homographies match that closed form to 0.0070 px
   worst, 6e-4 of the phantom's own 11.43 px size; with the
   correction on, `Fe = I` to 1.6956e-05 in the worst entry against
   8.06e-03 in the same entries uncorrected, a factor of about 475.

6. **D5 capture range (recorded, never gated; requirements D5
   amended with this date).** Recipe: the V4 sweep
   (`test_rotation_sweep_capture_range`, run with `--weekly`, PASSES
   at `largest >= 0.5`) plus a scratchpad detail run that adds the
   intermediate angles, the seed values and the preprocessed ZNCC of
   each pair. Phase cross-correlation seeded, counted as recovered
   when converged AND within 1.0 px of the exact homography:

   | angle | converged | error (px) | iterations | seed (dx, dy) px | ZNCC |
   |---|---|---|---|---|---|
   | 0.5 deg | yes | 0.002145 | 5 | (-0.125, 0.250) | +0.876 |
   | 1.0 deg | yes | 0.003972 | 5 | (0.000, -0.062) | +0.622 |
   | 2.0 deg | yes | 0.011303 | 8 | (0.062, -6.375) | +0.176 |
   | 2.5 deg | no | 148.26 | 50 (cap) | (0.062, -9.812) | +0.049 |
   | 3.0 deg | no | 172.88 | 50 (cap) | (-0.062, -9.312) | -0.024 |
   | 4.0 deg | no | 221.43 | 50 (cap) | (0.000, -15.375) | -0.069 |
   | 5.0 deg | no | 97.93 | 50 (cap) | (0.062, -15.062) | -0.056 |

   **Recorded capture range: 2.0 degrees** of pure in-plane rotation
   at 480x480 with the frozen `(0.05, None)` band-pass. Two limits,
   separated by re-running from the exact (identity) seed: the basin
   itself reaches 3.0 deg (11 iterations at 2.5, 16 at 3.0) and
   fails at 4.0 deg, so 2.5 and 3.0 deg are SEED failures (spurious
   translations of 9.8 and 9.3 px where the truth is zero) while
   4.0 deg and beyond are basin failures. Beyond 2.5 deg the pair is
   simply decorrelated (ZNCC at or below 0.05), which is why no
   conformant implementation of the frozen D2/D4/D5 design can do
   better without the deferred Fourier-Mellin pre-rotation (plan
   open question 5). This number goes into the `hrebsd_dic`
   docstring Notes.

   Discrepancy recorded, not corrected: the `TestPureRotations`
   comment in `test_hrebsd_engine.py`, written at the implementation
   gate, reports the seed returning 128 px and 160 px at 2.5 and
   3.0 deg and the exact-seed basin ending between 4.0 deg (189
   iterations) and 4.5 deg. Neither reproduces here: the seed
   returns 9.8 and 9.3 px, and 4.0 deg does not converge at the
   frozen 50-iteration cap (189 iterations would need a raised cap,
   a different experiment). The comment's CONCLUSION, re-pinning the
   V4 in-plane angles from 5.0 deg to 2.0, is confirmed by both
   measurements; only the intermediate figures differ, and the
   numbers in THIS ledger are the ones with a stated recipe.

7. **D1.4 Hessian conditioning (requirements D1.4, amended with
   this date).** Recipe: `numpy.linalg.cond` and `eigvalsh` on
   `ReferenceState.hessian` built with the default knobs. At
   480x480 (186624 subregion pixels): condition number
   **5.7529e+08**, eigenvalues 1.5120e-01 to 8.6985e+07, so **6.2
   orders of headroom** under the ~1e15 an f64 Cholesky holds,
   exactly the "safe by ~6 orders" the spec claimed. The drafted
   `(half-width)^4 = 3.32e+09` estimate of the spread is 5.8x
   conservative. Symmetrically scaled by its own diagonal the same
   matrix conditions at **4.51**, so the spread is entirely the
   pixel unit system and not a near-degeneracy. At 60x60 (2916
   pixels): 3.0312e+05, scaled 9.55. `scipy.linalg.cho_factor`
   succeeds on both. The recorded normalize-by-pattern-width
   fallback is NOT taken.

8. **D3 bicubic versus quintic (plan open question 1; requirements
   D3 amended with this date). VERDICT: BICUBIC STAYS.** Two
   independent arms, neither meeting the re-pin criterion (more than
   2x accuracy at under 1.5x cost).

   - On the V2 warp-refit oracle, which is where plan 6.1 asks for
     it: a controlled A/B in the scratchpad (`measure_extras.py d3`)
     reruns the whole IC-GN loop with the spline order as the ONLY
     difference (same preprocessing, same phase-XC seed, same
     matched H and g scales, same update rule, same exit criterion,
     gradients by central difference of the SAME interpolant on both
     arms so neither order is favoured by having an analytic
     derivative the other lacks), six random small homographies at
     480 px. Worst recovery error **0.010496 px bicubic versus
     0.012649 px quintic**: quintic is 0.83x as accurate, that is
     WORSE. Mean iterations 5.67 versus 6.17.
   - Cost, on the 186624 subregion points the engine really
     evaluates: the numba bicubic kernel **4.621 ms** versus
     `map_coordinates(order=5, prefilter=False)` **15.310 ms**, a
     ratio of **3.31x**.
   - Against the ANALYTIC band-limited truth
     (`test_order_ab_harness`, where an order difference originates
     at all): bicubic 1.0549e-04, quintic 5.7480e-05 scale relative,
     a gain of **1.84x**, below the 2x threshold and anyway swamped
     at oracle level by the cross-interpolator systematic.

   This reproduces Ruggles 2018 (no significant biquintic gain at
   960 px). Ernould's and EMsoftOO's quintic (`mod_DIC.f90:50-51`)
   stays a recorded deviation.

9. **D17 f32 versus f64 coefficient storage (plan open question 11;
   requirements D17 amended with this date). VERDICT: f32
   CONFIRMED, no longer provisional.** Recipe:
   `test_dtype_ab_harness` on the V2 480 px batch (four random small
   homographies, seed 15), plus timing and byte counts from the
   scratchpad rerun (`measure_extras.py d17`). Worst recovery error
   **0.00628116603 px (f64) versus 0.00628116632 px (f32)**, a
   degradation of **4.5e-08** of the f64 error against the D17
   criterion of 0.10, six orders of margin. Per-case errors agree to
   the ninth digit (f64 0.0039168450, 0.0062811660, 0.0041327015,
   0.0050798531; f32 0.0039168448, 0.0062811663, 0.0041327016,
   0.0050798518). Saving: the coefficient array is **1.84 MB f64
   versus 0.92 MB f32**, resident per-grain precompute **18.27 MB
   versus 17.34 MB** at 480x480. Fit time unchanged within noise
   (0.0713 s f64, 0.0742 s f32 per 480 px fit; the f32 arm is not
   faster because the kernel accumulates in f64 either way). f64
   accumulators stay non-negotiable.

10. **`step_scale` 1.0 / 1.25 / 1.5 (plan open question 9;
    requirements D2.4 amended with this date). VERDICT: 1.0 STAYS;
    the EMsoftOO accelerator is refuted as an accelerator.** Recipe:
    `measure_extras.py step_scale`, the twelve-case V2 batch at
    480 px (seeds 0 and 1) plus a basin arm (an in-plane rotation
    ladder and the 15 px translation case).

    | `step_scale` | worst error (px) | mean iterations | total iterations | non-converged | largest rotation recovered | 15 px translation |
    |---|---|---|---|---|---|---|
    | 1.0 | 0.012440 | 5.17 | 62 | 0 | 2.0 deg | converged, 2 iterations |
    | 1.25 | 0.012461 | 7.33 | 88 | 0 | 2.0 deg | converged, 3 iterations |
    | 1.5 | 0.012482 | 12.75 | 153 | 0 | 2.0 deg | converged, 6 iterations |

    So 1.5 costs 2.5x the iterations for a 0.3 per cent accuracy
    LOSS and no basin gain whatsoever. `step_scale = 1.0` stays
    frozen. (The iteration counts here also feed plan open question
    2: at the frozen `min_step = 1e-3` px the 480 px batch exits in
    5.2 iterations on average and 7 at worst, far under the cap of
    50, and the residual error sits at the interpolation floor, so
    the convergence defaults are not the limiting factor. The full
    error-versus-threshold curve is still a Stage B item.)

11. **Performance baseline (recorded, never a gate, D16).** Recipe:
    `measure_perf.py`, `dask.config.set(num_workers=8,
    scheduler="threads")`, warm numba caches, best of three timed
    runs after a two-pattern warm-up, all patterns one grain against
    one reference so the numbers are engine throughput rather than
    convergence failures.

    | configuration | patterns | time | rate | converged | mean iterations |
    |---|---|---|---|---|---|
    | 480x480 oracle, default knobs, 8 workers | 40 | 1.670 s | **23.96 pat/s** | 40/40 | 4.65 |
    | 60x60 (`nickel_ebsd_small` reference, scaled warp batch), 8 workers | 100 | 0.167 s | **600.4 pat/s** | 100/100 | 3.67 |

    Per-grain precompute at 480x480: **0.0402 s** (median of three)
    and **16.54 MB** resident with f32 coefficients, over 186624
    subregion pixels. Nothing here is a floor: D16 makes performance
    a recorded baseline only, and the spherical 2 pat/s/core floor
    is EMSphInx-scoped. The Si-wafer route row of the Performance
    table is deliberately replaced by the 480x480 ORACLE route,
    which needs no download; the Si-wafer timing lands with the V5
    benchmark at the Stage B gate.

12. **Not measured here, still open.** The V5 Si-wafer noise floors,
    the full convergence sweep of plan open question 2 (only the
    iteration counts above), the border and preprocessing sweeps
    (open questions 3 and 10), the KAM defaults (4), the weekly
    960x960 warp-refit, and the local oldest-matrix run. All are
    Stage B or weekly items and keep their own gates.

13. **Per-grain resident memory, a drafted number REFUTED
    (requirements D16 amended with this date).** Recipe:
    `ReferenceState.memory_bytes()` on the 480x480 oracle reference
    with the default `border=0.05`. The spec's drafted "~5 f32/f64
    planes of the SR, ~4.6 MB at 480x480" is **3.6x too small**: the
    eight steepest-descent columns of D2.1 are themselves eight
    subregion sized f64 planes, so eleven planes plus the f32
    coefficient plane are resident, MEASURED **17344512 B =
    16.54 MB** over 186624 subregion pixels (11.94 MB
    steepest-descent, 4.48 MB reference plus the two coordinate
    planes, 0.92 MB f32 coefficients; 18.27 MB with f64
    coefficients). The information message was already correct, its
    whole-pattern upper bound printing 20.2 MB at this size; only
    the prose was wrong, and the two docstrings repeating it
    (`ReferenceState` and `EBSD.hrebsd_dic`) are corrected with this
    date. The `EBSD.hrebsd_dic` Notes also had the basin "ending
    near 4 degrees" from an exact seed, corrected to "between 3 and
    4" by entry 6's measurement (3.0 deg converges in 16 iterations,
    4.0 deg does not converge at the frozen 50-iteration cap).

### 2026-09-07 (Stage A adversarial review, fixer)

**Machine A** as in the implementation-gate section above: 20-core
Windows 11 laptop (Raptor Lake, build 26200), `.venv` Python 3.13.12,
numpy 2.4.6, scipy 1.17.1, numba 0.65.1, scikit-image 0.26.0, orix
0.14.2, dask 2026.3.0. Warm numba caches, idle machine. Two
adversarial reviewers (fidelity/theory, conventions/integration)
returned 25 findings plus 2 surviving mutants; the disposition table
is in plan.md section 9. Every number below was measured by this
fixer, and each one that moves a frozen decision is amended in
requirements.md with the same date.

14. **D6.2 conversion frame, a frozen decision REFUTED (critical;
    requirements D6.2 amended with this date).** The drafted rule
    converted the ALREADY-CORRECTED homography with
    `pc_rel = PC_t - PC_ref` and `dd = DD_target`, blessed as "first
    order equivalent". Re-derivation: a raw fit in the D1.3 frame is
    `W = T(delta) . diag(1, 1, 1/DD_t) . Fe . diag(1, 1, DD_r)`, and
    its `Fe = I` case reproduces the D6.2 closed-form phantom to
    **0.0** (so the ray model and the V6-pinned phantom are the same
    object); removing that phantom therefore leaves
    `W_corr = diag(1, 1, DD_r)^-1 . Fe . diag(1, 1, DD_r)`, a pure
    reference-frame homography, whose exact conversion is
    `pc_rel = (0, 0)`, `dd = DD_reference`. Recipe and numbers,
    scratchpad `v1_d6_algebra.py` then the committed oracle:

    | quantity | drafted route | corrected route |
    |---|---|---|
    | analytic, V6 geometry, 1 deg out-of-plane tilt | 3.8668e-04 | 6.9e-18 |
    | analytic, V6 geometry, generic 2e-3 `Fe` | 4.0492e-05 | 1.3e-18 |
    | ENGINE, V6 map + 1 deg tilt, worst \|Fe - Fe_true\| | **3.9967e-04** | **2.2255e-05** |

    3.9967e-04 is 13x the pinned `DEFORMED_MASTER_FE_TOL = 3.1e-5`
    and 5x the 8e-5 strain precision the `hrebsd_dic` docstring
    quotes. The blindness was structural: V6's
    `test_phantom_corrected` imposes `Fe = I`, where `h_corr = 0`
    makes every `pc_rel`/`dd` give the identity, and V3's
    `test_deformed_master_fe_through_the_engine` states in its own
    comment that "every point shares one projection centre". NEW
    ORACLE, committed:
    `TestPcShiftPhantom::test_corrected_fe_on_a_deformed_per_point_pc_map`
    -- the V6 `(2, 3)` per-point-PC map with a 1 degree out-of-plane
    tilt imposed on every target, which needs BOTH a moving
    projection centre and a real deformation. It measures 2.2255e-05
    against the 3.1e-5 band (1.39x margin, recorded because it is
    the tightest margin in the suite) and FAILS at 3.9967e-04 with
    the drafted route re-injected. Three geometry tests were
    rewritten with a GENERIC `Fe` (nonzero Fe13/Fe23/Fe31/Fe32) and
    a ray-model construction; `test_phantom_corrected` itself is
    unchanged at 1.6956e-05, exactly as predicted, since at `Fe = I`
    the two routes coincide identically.

15. **Surviving mutant "correction applied after Fe conversion"
    (plan 2.5), resolved by the D6.2 fix + a strengthened test.**
    Under the corrected conversion the map is a conjugation by
    `diag(1, 1, DD_ref)`, a group HOMOMORPHISM, so removing the
    phantom in homography space and removing it in Fe space agree
    **to 2.2e-16**: that form is an EQUIVALENT mutant and is recorded
    as such by `test_correcting_after_the_conversion_is_now_identical`,
    which re-checks the equivalence whenever the conversion moves.
    The NON-equivalent form -- convert with the target PC first,
    then divide out the phantom in Fe space -- dies at
    `test_correction_precedes_conversion` and
    `test_correcting_after_the_conversion_is_now_identical`
    (verified by injection and restore).

16. **Surviving mutant "mirror-boundary derivative sign dropped"
    (`_bicubic_evaluate_gradient`), KILLED.** Recipe: injected
    `out_gx[i] = gradient_x` / `out_gy[i] = gradient_y`, ran
    `test_hrebsd_interpolation.py` -> `1 failed, 29 passed`, restored
    -> `30 passed`. The killer is the new
    `TestMirrorBoundary::test_gradient_mirror_boundary_keeps_the_fold_sign`:
    analytic gradients at out-of-frame coordinates against a central
    difference of the VALUE kernel (which carries no sign logic),
    plus the tolerance-free statement that a fold NEGATES the
    derivative with respect to the original coordinate. The sibling
    `test_mirror_boundary_through_py_func` executes every fold branch
    of the VALUE kernel through `.py_func`, which is the only way
    coverage sees a numba kernel.

17. **`residual` was one iterate behind `homography` (requirements
    D2.7 amended).** MEASURED on the 480 px oracle before the fix:
    a converged fit reported 0.0008888143693887548 against
    0.0008888283289345156 recomputed at the returned homography
    (1.6e-05 relative), and a fit capped at two iterations reported
    **1.5433811939061348 against 1.2239561138811788 -- 26 per cent
    high**, which is exactly the point a D10 quality map is read on.
    The criterion is now evaluated once more after the loop: the
    same two cases measure a relative difference of **0.0** (bitwise)
    against an independent assembly, pinned by
    `TestResidualIsTheFinalCriterion` at `max_iterations` 2 and 50.
    Cost, re-recorded below.

18. **`ReferenceState.memory_bytes()` understated by ten per cent
    (requirements D16 re-amended).** It omitted `reference_subregion`
    (1492992 B at 480x480), a persistent attribute. MEASURED
    **18837504 B = 17.96 MB** over 186624 subregion pixels, against
    the 17344512 B = 16.54 MB the ledger recorded at entry 13, so the
    drafted "~4.6 MB" is 3.9x too small, not 3.6x. The shared arrays
    (mask, transfer function, window) are excluded BY DESIGN and the
    docstring now says so: one of each is built per RUN and shared by
    every reference. The information message model gains the same
    plane and prints **22.0 MB** at 480x480 (was 20.2 MB), which
    still bounds the measurement; `TestPrecomputeMemory` now asserts
    the bound, the closed form and `n_pixels`, none of which any test
    executed before.

19. **The D4.3 `window` knob: uncovered AND implemented against its
    own decision (requirements D4.3 amended).** It was built over the
    WHOLE pattern and multiplied into each pattern in that pattern's
    OWN frame inside `preprocess`, before the warp, so the taper
    travelled with the target and broke the affine intensity model
    ZNSSD assumes. Reviewer measurement, 120 px synthetic, D2.5
    corner-displacement error: identity 0.00000 / 0.00000; translate
    2 px 0.01601 -> 0.30312 (18.9x); strain 2e-3 0.00175 -> 0.00232;
    generic h 0.00660 -> 0.11304 (17.1x) -- the error scaling with
    the translation and vanishing at the identity is the signature.
    Now the window is built over the SUBREGION bounding box and
    applied as a per-pixel WEIGHT on the ZNSSD residual in the
    reference frame, which weights the steepest-descent images by the
    same `w` and leaves the D2.1/D2.3 formulas untouched; the
    zero-mean unit-norm vectors stay unwindowed, so ZNSSD's affine
    invariance is exact. MEASURED on the 480 px seed-35 pair:
    **0.04565 px windowed against 0.00656 px plain**, and the
    pre-correction scheme measures **0.14949 px** on the same two
    cases. New pin `WINDOW_REFIT_TOL_480 = 0.092` (2x the
    measurement; it fails the old scheme by 1.6x), plus a structural
    arm asserting that `preprocess` has no `window` parameter at all
    and one pinning the weighted steepest-descent block.

20. **Scan-step units were guessed (requirements D6.1 amended, the
    D14.5 precedent).** `EBSD.hrebsd_dic` fed the navigation axes'
    `scale` straight into the beam-scan model without reading their
    `units`. Reviewer measurement on the strain-free V6 phantom
    through the single-PC route: `max|Fe - I|` = **1.70e-05** with
    step sizes in um, **4.71e+01** with the same scan described in
    nm, and **4.71e-02** in mm, which equals the completely
    UNCORRECTED phantom, i.e. the correction silently became a no-op;
    no warning in any of the three runs. The units are now read and
    converted to the micrometres `px_size` uses, with a ValueError
    naming the axis for a missing or unrecognized unit (including
    HyperSpy's `"px"`), and only when the detector carries a single
    PC, since the steps are unused otherwise. `px_size` itself
    carries no unit and its 1.0 default is a placeholder --
    kikuchipy's own `nickel_ebsd_small` ships it beside a 1.5 um
    scan step, a 70x mismatch -- so it is DOCUMENTED rather than
    guarded, and recorded here as the caller's responsibility. Tests:
    `TestScanStepUnits` (signal level) and `TestStepSizeUnits`
    (geometry level).

21. **Capture range is a function of `filter_cutoffs`; a Context
    overclaim WITHDRAWN.** Recipe: the V4 deformed-master in-plane
    sweep, same engine, same phase-XC seed, same 50-iteration cap,
    `filter_cutoffs` the only change.

    | angle | `(0.05, None)` | `(None, None)` |
    |---|---|---|
    | 2.0 deg | 8 it, 0.01130 px, ZNCC +0.176 | 11 it, 0.00110 px, ZNCC +0.589 |
    | 2.5 deg | cap, 148.26 px, +0.049 | 15 it, 0.00165 px, +0.472 |
    | 3.0 deg | cap, 172.88 px, -0.024 | 18 it, 0.00167 px, +0.366 |
    | 4.0 deg | cap, 221.43 px, -0.069 | 41 it, 0.00168 px, +0.196 |

    So the default HALVES the capture range and costs a factor of
    ten in the 2.0 deg error on noise-free patterns, and the
    "decorrelated pair" premise is the filter's too. The requirements
    Context clause "no conformant implementation of the frozen
    D2/D4/D5 design converges there" is struck; the `hrebsd_dic`
    docstring now quotes both numbers conditionally. The DEFAULT IS
    NOT RE-PINNED: on noise-free oracles a high-pass can only remove
    signal, and its purpose -- background gradients on real data --
    is the V5 Si-wafer measurement of plan open question 10, at the
    Stage B gate. Also recorded (requirements D4.1): the knob maps to
    `highpass_fft_filter(cutoff=high_pass * width)`, a radius in FFT
    bins, while EMsoftOO's `hipassw` is a normalized-frequency
    Gaussian parameter, so the two share a numeral and not a unit
    system.

22. **Entry 6's seed column does NOT reproduce; the test-file figures
    do.** Re-run of the engine's own recipe
    (`initial_guess(state.reference_subregion,
    preprocess(target, transfer_function)[state.bounds])`,
    `upsample_factor=16`): the seed `(dx, dy)` is
    **(-0.062, +127.812) px at 2.5 deg** and **(-0.062, +159.812) px
    at 3.0 deg**, matching the `TestPureRotations` comment's 128 px
    and 160 px, NOT entry 6's (0.062, -9.812) and (-0.062, -9.312).
    Every other column of that table reproduces exactly here: ZNCC
    +0.176/+0.049/-0.024/-0.069, errors 148.26/172.88/221.43 px,
    8 iterations at 2.0 deg. Entry 6's tie-break sentence ("the
    numbers in THIS ledger are the ones with a stated recipe")
    therefore picked the wrong side on that one column, and this
    entry is the correction; entry 6 stands unedited, the ledger
    being append only. The CONCLUSION both agree on -- a seed failure
    rather than a basin failure at 2.5 and 3.0 deg -- is unaffected,
    since a 128 px spurious translation is a seed failure a fortiori.

23. **Coverage, the Definition-of-done item that had no recorded
    run.** Command and output, this machine, this date:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_interpolation.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py \
      --cov=src/kikuchipy/indexing/_hrebsd --cov-report=term-missing -q
    ->
      __init__.py          0 stmts, 0 miss, 100.00%
      _engine.py         270 stmts, 0 miss, 100.00%
      _geometry.py        55 stmts, 0 miss, 100.00%
      _homography.py      62 stmts, 0 miss, 100.00%
      _interpolation.py  186 stmts, 0 miss, 100.00%
      _preprocessing.py   70 stmts, 0 miss, 100.00%
      _reference.py       59 stmts, 0 miss, 100.00%
      TOTAL              702 stmts, 0 miss, 100.00%
      242 passed, 1 skipped in 21.65 s
    ```

    (`pytest-cov` is not in `.venv`; it was installed into a
    scratchpad directory and put on `PYTHONPATH` for the run, which
    changes nothing about the measurement.) The review found
    91.98 %. The new tests are named in entries 14 to 22 above plus:
    `TestStepScale` (the `step_scale != 1.0` branch, which committed
    the ordering entry 10 recorded -- 1.5 costs more iterations for
    no accuracy gain -- and a tolerance-free `step_scale=0.0` arm),
    `TestArgumentGuards` (every previously unexecuted `raise`, with
    its message asserted), `TestReferenceResolution`'s three new
    `grain_labels` arms including the new same-grain guard,
    `TestNavigationDimensions` (the documented 1-D navigation and the
    dimension guard, neither exercised through the public method),
    `TestGrainLabels` (`grain_labels` never reached the engine
    through `EBSD.hrebsd_dic` at all), the `(None, 0.4)` low-pass-only
    arm of `filter_cutoffs`, and the non-contiguous / non-f64 `out=`
    path of `evaluate`. `EBSD.hrebsd_dic` itself is also fully
    covered (no missing line in 2450-2820). ONE piece of genuinely
    dead code was pruned rather than tested: the
    `if not np.isfinite(residual)` guard inside the IC-GN loop is
    unreachable, because `zero_mean_normalize` already refuses a
    warped subregion whose centred 2-norm is zero or not finite and
    the criterion of two unit-norm vectors is bounded by four times
    the pixel count; D2.6's non-finite-CIC outcome is unchanged and
    is now pinned through that raise instead.

24. **Local oldest-matrix run, Stage A (the plan section 1 recipe;
    validation entry 12 misclassified it as a Stage B or weekly
    item, which plan.md section 1 and the Definition of done both
    contradict -- it is a PER-STAGE gate).** Command:

    ```
    uv run --isolated --python 3.10 \
      --with "numpy==1.23.0" --with "numba==0.57" \
      --with "orix==0.12.1" --with "scikit-image==0.21.0" \
      --with pytest-benchmark --with pytest-rerunfailures \
      --with pytest-xdist --with pytest-randomly \
      pytest tests/test_indexing tests/test_signals -k hrebsd -q
    ->  242 passed, 1 skipped, 4335 deselected in 23.42 s
    ```

    Environment as resolved: Python 3.10.19, numpy 1.23.0, scipy
    1.13.1, numba 0.57.0, orix 0.12.1, scikit-image 0.21.0, dask
    2024.8.1. (The four pytest plugins are needed only because the
    `pyproject.toml` `addopts` reference them; they change no
    behaviour under test.) The D15.7 pin
    `GET_MAP_DATA_2D_PROP_OUTCOME = "TypeError"` holds on orix
    0.12.1 as well as on the venv's 0.14.2.

25. **Performance, re-recorded after the D2.7 final-criterion
    change.** Recipe: 40 warped 480 px targets against one reference,
    `dask.config.set(num_workers=8, scheduler="threads")`,
    `chunksize=6`, warm caches, best of three after a warm-up:
    **22.72 patterns/s** (1.761 s, 41/41 converged, mean 4.75
    iterations), against 23.96 patterns/s at the implementation gate
    on its own 40-pattern recipe -- about 5 per cent, which is the
    one extra interpolation per fit the final criterion costs
    amortized over 4.75 iterations. Per-grain precompute 0.0362 s.
    D16 makes performance a recorded baseline and never a gate.

26. **Findings NOT applied, with the evidence.** (a) The conventions
    reviewer asked for a ValueError when `detector.px_size` is still
    at its 1.0 default: REJECTED, and the reason recorded in
    requirements D6.1 -- a placeholder 1.0 is indistinguishable from
    a genuine 1 um pixel, and kikuchipy's own shipped
    `nickel_ebsd_small` detector carries the placeholder, so the
    guard would refuse the demonstration data. The units half of the
    same finding, which IS decidable, is applied (entry 20). (b) The
    fidelity reviewer asked for the `correct=False` diagnostic path
    to use "the exact general corner-origin form" instead of the
    `beta0` shortcut: REJECTED. That path's behaviour is pinned by
    the frozen `test_conversion_uses_the_relative_target_pc`, it
    feeds only validation V6's uncorrected arm and D13's diagnostics,
    and `homography_to_fe`/`fe_to_homography` are exact mutual
    inverses as they stand (V1); changing it would move a frozen
    algebraic pin for no production effect. What the finding is
    really about -- the CORRECTED path -- is applied in full
    (entry 14).

27. **Full existing suite, after every fix above.**

    ```
    .venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_data
    ->  1 failed, 4315 passed, 797 skipped, 5 rerun in 201.68 s
    ```

    The one failure is
    `tests/test_simulations/test_kikuchi_pattern_simulator.py
    ::TestCalculateMasterPattern::test_shape`, the SAME upstream
    flaky test entry 4 of the implementation gate logged on this
    machine so that a future red run would not be misread as an
    HREBSD regression. It is upstream's own
    `@pytest.mark.flaky(reruns=5)` hemisphere comparison
    (`np.allclose(mp.data[0], mp.data[1], atol=1e-4)` on two
    201x201 master patterns),
    `git diff HEAD -- src/kikuchipy/simulations tests/test_simulations`
    is empty on this branch, and nothing in the HREBSD path is
    reachable from it. Deselecting it, the suite is green.

### 2026-09-07 (Stage B failing-tests gate, adversarial review fixes)

Machine: the 20-core Windows 11 laptop of the spherical phases,
Python 3.12 in `.venv`. The Stage B tests were written failing
before the implementation, then adversarially reviewed; twelve
findings and six mutant-coverage items were dispositioned by the
fixer. Every number below comes either from a LIBRARY measurement
that needs no `_hrebsd` implementation, or from a REFERENCE CHAIN
written in the scratchpad to check that the tests as delivered are
satisfiable and discriminating. None of them is a pin: the pins
are filled at the implementation gate, from the implementation.

28. **HR-KAM pair-angle conditioning (requirements D12, amended
    with this date).** The drafted oracle of
    `test_hrebsd_kam.py` computed the pair angle as
    `arccos((tr - 1)/2)`. MEASURED on its own constant-curvature
    field at `kappa = 1e-4` rad/step: the interior point (2, 3)
    comes out at 0.07499999980334485 mrad against the closed form
    0.075 (relative 2.6221e-09) and the corner (0, 0) at
    0.06666666649186208 against 0.06666666666666667. The module
    asserts the implementation against that oracle at
    `atol = 1e-12` AND against the closed forms at `rel = 1e-9`,
    so the two families were mutually unsatisfiable: no
    implementation could pass both. The well-conditioned form
    `arctan2(||skew(M)||/2, (tr(M) - 1)/2)` gives
    0.07499999999999998 and 0.06666666666666667, relative 3.7e-16
    and 0.0, and both families then hold. APPLIED to the oracle,
    to the `_kam.py` docstring and to requirements D12; a new
    library test, `TestConstantCurvature::test_the_oracle_itself
    _reproduces_the_closed_forms`, guards the conditioning and
    passes today.

29. **Pure-rotation strain floor of the chain (requirements D8
    and validation V4 amended with this date).** Reference chain,
    deviatoric and traction-free closures, on the Stage B tensor
    module's own detector geometry, for a pure sample-frame
    rotation about the detector normal:

    | angle | `sym(R - I)` max | deviatoric | traction free | `F_det[2,2]` dilatation |
    |---|---|---|---|---|
    | 1 deg | 1.5230e-04 | 1.0154e-04 | 7.2762e-05 | 1.7816e-05 |
    | 2 deg | 6.0917e-04 | 4.0614e-04 | 2.9104e-04 | 7.1260e-05 |
    | 5 deg | 3.8053e-03 | 2.5380e-03 | 1.8187e-03 | 4.4514e-04 |

    The deviatoric ratio is 0.6667 at every angle and the
    traction-free one 0.4778, which is algebra and not a
    coincidence: the closure turns `diag(c-1, c-1, 0)` into
    `diag(-(1-c)/3, -(1-c)/3, 2(1-c)/3)`. Consequences, both
    applied: the drafted `TestSmallStrainFastPath::test_the_public
    _path_is_the_polar_one` demanded the reported strain of a two
    degree rotation be under `0.1 * 6.0917e-04 = 6.0917e-05`,
    which the reduction alone (7.13e-05) already exceeds and the
    chain misses by 6.7x, so it was unsatisfiable by any
    conformant implementation; and the V4 seed
    `ROTATION_STRAIN_LEAK <= 1e-4 up to 5 deg` is refuted by 25x.
    The test now decides WHICH SPLIT was taken on the chain's own
    reported `beta`: MEASURED there, the polar and small-strain
    answers differ by 6.0901e-04 at 2 deg while the reported
    strain reproduces the polar one exactly (0.0 difference), so
    the arm is tolerance-free on one side and separated by four
    orders on the other.

30. **Reference-chain scales for the four tensor-module MTP
    placeholders**, so that the implementation gate has something
    to compare its own measurement against (they are NOT pins).
    `REDUCED_CLOSURE_TOL` 1.1655e-06 (strain), 9.4132e-07 (e33),
    9.5697e-07 (beta); `SIGMA33_TOL` 3.0639e-04 GPa;
    `SMALL_STRAIN_EQUALITY_TOL` 5.889e-04 (the worst of the
    drafted sweep); `SMALL_STRAIN_ENABLE_ANGLE_DEG` 0.0779441. On
    the sweep grid `(0.01, 0.05, 0.1, 0.5, 1.0, 2.0)` the errors
    are 9.140e-08, 5.656e-07, 1.604e-06, 3.733e-05, 1.479e-04 and
    5.889e-04, so reading the enabling angle off the grid would
    have recorded exactly 0.05 -- an artefact of the grid. The
    test now BISECTS between the last passing and first failing
    grid point to 1e-6 deg and records the crossing.

31. **V3's tensor half delivered, and the "Bond rotation
    transposed" mutant killed through PATTERNS.** validation V3
    names `test_deformed_master_strain_recovery` as a Stage B
    deliverable and plan 3.4 names it as a co-killer of that
    mutant; the drafted Stage B commit shipped no pattern-level
    tensor oracle at all. Delivered as
    `tests/test_indexing/test_hrebsd_deformed_master.py` (file
    layout recorded in the V3 block above). Reference-chain
    measurement of the delivered oracle -- two imposed
    sigma33 = 0 tensors of the 1e-3 scale at the generic
    orientation, projected onto the shipped Ni Lambert master at
    480 px, carried through `run_hrebsd_dic` and then through the
    chain: engine `|Fe - imposed|max` 1.1997e-05 and 1.1029e-05;
    worst per-component strain error **9.4045e-06**; worst e33
    error 8.3928e-06; `sigma33` worst **1.8338e-04 GPa** against a
    stress scale of 0.2420 GPa; rotation-vector error 4.2064e-06
    against an imposed 8.2551e-04; the reference point's own
    strain 5.35e-12; the two closures separated by 7.1904e-05 in
    e33. With the Bond rotation TRANSPOSED the worst strain error
    is **3.0068e-04**, 32x the correct value, so the mutant dies
    here at pattern level as plan 3.4 intends. NOTE, recorded for
    the mutation list: the `sigma33` self-check does NOT see that
    mutant (3.0914e-04 GPa transposed against 3.0639e-04 correct
    on the analytic route) and must never be quoted as its killer.

32. **V5 delivered as a [download] gated module.** plan 3.3 makes
    the Si-wafer benchmark a Stage B deliverable and it owns plan
    open question 10; the drafted commit shipped none of it.
    Delivered as `tests/test_indexing/test_hrebsd_si.py` with the
    four tests validation V5 names plus the sweep harnesses of
    plan open questions 2, 3, 4 and 10. It downloads nothing:
    without the cached dataset every test SKIPS with a message
    naming `kp.data.si_wafer(allow_download=True, lazy=True)`.
    VERIFIED at this gate: `9 skipped in 0.06 s` (3 cache skips,
    6 `--weekly` skips) on a machine with pooch installed and the
    dataset absent. The module is therefore UNEXECUTED at the
    failing-tests gate and its four placeholders
    (`SI_STRAIN_FLOOR`, `SI_ROTATION_FLOOR`, `SI_KAM_FLOOR`,
    `SI_PC_RESIDUAL_MEAN_TOL`) are filled at the implementation
    gate, which must fetch the dataset once. Its default route
    takes the DEVIATORIC closure and a nominal single orientation,
    so the recorded floors do not depend on indexing the wafer;
    the traction-free arm, which does, is weekly and recorded, not
    gated.

33. **Two Stage A `reference="auto"` pins now pass VACUOUSLY.**
    INSTRUMENTED: `resolve_reference(AUTO_REFERENCE, None, (3, 3))`
    raises `NotImplementedError: segment_grains arrives with
    Stage B of specs/2026-09-07-hrebsd-dic/`, raised at
    `_segmentation.py` line 142 -- not by the Stage A guard, which
    the Stage B wiring of `_reference.py` deleted. The message
    happens to contain the "Stage B" the two tests match on, so
    both still pass and a green run is NOT evidence that the
    Stage A contract holds. The two are
    `test_hrebsd_engine.py::TestReferenceResolution::test_auto
    _raises_naming_stage_b` and
    `::TestOrchestration::test_auto_reference_raises_through_the
    _engine`. DISPOSITION: both carry an explicit superseded-by
    note naming their positive Stage B replacements
    (`test_hrebsd_segmentation.py::TestAutoReference` and
    `test_ebsd_hrebsd_dic.py::TestAutoReference`), and **the Stage
    B IMPLEMENTATION gate DELETES them** -- once `segment_grains`
    lands nothing raises and they fail loudly. They are kept until
    then only so that this failing-tests commit leaves the Stage A
    regression count untouched (verified: 242 passed, 1 skipped
    before and after).

34. **`segment_grains` multi-phase guard struck (requirements
    D11.1 amended with this date).** A drafted ValueError on
    multi-phase maps was pinned by
    `test_hrebsd_segmentation.py::TestSegmentGrains::test_guards`.
    No frozen requirement asks for it, and because
    `reference="auto"` is the FROZEN DEFAULT of `EBSD.hrebsd_dic`
    it would have narrowed the engine to single-phase maps on the
    default path, contradicting D9.6 ("the engine itself is
    phase-agnostic per grain"); `grep -n phase _engine.py` confirms
    Stage A has no phase guard. Struck in favour of per-phase
    segmentation: an edge across a phase boundary simply never
    exists. Two positive tests replace the guard, and the second
    is a LIBRARY measurement of why one point group cannot serve
    the whole map -- 90 deg about z is a symmetry operation of
    m-3m (measured symmetry-reduced angle 0.0 deg) and not of
    6/mmm (measured 30.0 deg), so the same pair is one grain in
    the cubic phase and two in the hexagonal one.

35. **Shape contract of the two `(ny, nx)` diagnostics
    (requirements D11.1 amended with this date).** MEASURED on the
    installed orix 0.14.2: `create_coordinate_arrays((1, 2), ...)`
    gives `xmap.shape == (2,)`, `ndim == 1`, `row = [0, 0]`,
    `col = [0, 1]`; `create_coordinate_arrays((3, 1), ...)` gives
    `xmap.shape == (3,)`, `row = [0, 1, 2]`, `col = [0, 0, 0]`.
    So `xmap.shape` is NOT the navigation shape for a map one
    point wide or tall, and "a one dimensional map is a single
    row" would return `(1, 3)` for a column map, which
    `_reference._flatten_labels` rejects against a `(3, 1)`
    navigation shape. The rule is now the row and column grids,
    stated in both docstrings and pinned by
    `test_hrebsd_segmentation.py::TestSegmentGrains::test_a_single
    _column_map_is_a_single_column` and `test_hrebsd_kam.py::
    TestShapeContract` (three tests, one of them the library
    measurement above, which passes today).

36. **Two blind spots closed in the delivered tests.**
    (a) `TestClosure::test_b7_and_b8_are_not_interchangeable`
    claimed the plan 3.4 "b7/b8 misassigned" mutant. MEASURED by
    injection into the reference closure: correct and swapped
    implementations BOTH give `|first[2,2] - second[2,2]| =
    1.4711e-05`, since the two answers are merely exchanged, so
    the assertion passed either way. The comment is corrected and
    a discriminating arm added -- the closed `e11` and `e22`
    individually reproduce what they were built from (measured
    error 0.0 and 5.4e-20 correct, 1.4711e-05 under the swap).
    The mutant's real killers are
    `test_traction_free_recovers_the_built_in_tensor` (1.4711e-05
    against `SOLVER_TOL` = 1e-10) and that new arm.
    (b) Row wraparound was covered for 4-connectivity but not for
    8: the only `connectivity=2` case sat on a 2x2 map, where
    every point neighbours every other and a `numpy.roll` style
    walk survives. `test_eight_neighbours_do_not_wrap_around_a_row`
    adds a (2, 3) map whose only same-orientation pair, (0, 0) and
    (0, 2), is two columns apart in one row: a flat-index walk
    with the `+nx - 1` offset merges them.

37. **One ulp of orientation reached a bitwise assertion.**
    `TestChain::test_the_chain_and_the_public_function_agree`
    compares `tensor_chain` with `hrebsd_strain_stress` at
    `rtol = 0, atol = 0` while handing the two routes orientation
    matrices from different sources. MEASURED:
    `Rotation.from_axes_angles((1,2,3), deg2rad(37)).to_matrix()`
    and the module's own Rodrigues helper differ by
    2.7755575615628914e-17, and through the reference chain that
    reaches the output -- `stress` by up to 5.55e-17 and `beta` by
    1.08e-19, both of which `atol = 0` rejects. The test now feeds
    the chain `xmap.rotations.to_matrix()`, the same matrices the
    public function reads, and asserts separately that those ARE
    this module's generic orientation. Independence is unaffected:
    that a crystal map's rotations are the `v_crystal = g @
    v_sample` matrix is pinned against `rotate_vector`, not
    against itself, in `test_hrebsd_stiffness.py::
    TestRotationDirection::test_the_matrix_is_the_kikuchipy
    _orientation`.

38. **Gate runs at the close of the fixes.**

    ```
    .venv/Scripts/python.exe -m pytest --collect-only -q
    ->  5388 tests collected, 1 skipped module (psygnal absent),
        no collection error

    .venv/Scripts/python.exe -m pytest \
        tests/test_indexing/test_hrebsd_kam.py \
        tests/test_indexing/test_hrebsd_segmentation.py \
        tests/test_indexing/test_hrebsd_stiffness.py \
        tests/test_indexing/test_hrebsd_tensors.py \
        tests/test_indexing/test_hrebsd_pc_shift.py \
        tests/test_indexing/test_hrebsd_deformed_master.py \
        tests/test_indexing/test_hrebsd_si.py -q
    ->  159 failed, 31 passed, 9 skipped

    .venv/Scripts/python.exe -m pytest \
        tests/test_indexing/test_hrebsd_engine.py \
        tests/test_indexing/test_hrebsd_geometry.py \
        tests/test_indexing/test_hrebsd_homography.py \
        tests/test_indexing/test_hrebsd_interpolation.py \
        tests/test_signals/test_ebsd_hrebsd_dic.py -q
    ->  7 failed, 242 passed, 1 skipped
    ```

    Every one of the 159 Stage B failures is a
    `NotImplementedError` from a Stage B skeleton (verified by
    tallying the exception type of every failure). The 31 passes
    are the freeze pins and the library measurements, which need
    no implementation. The 7 failures of the second run are the
    Stage B `TestAutoReference` class in the signal-method file,
    and 242 passed / 1 skipped is the unchanged Stage A count.
