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

**Stage B MTP pins, ALL FILLED 2026-09-08** (implementation gate;
measurements, recipes, margins and machine in Recorded results
entries 39 to 46 below): `REDUCED_CLOSURE_TOL` 2.4e-06,
`SIGMA33_TOL` 6.2e-04 GPa, `SMALL_STRAIN_EQUALITY_TOL` 1.2e-03,
`SMALL_STRAIN_ENABLE_ANGLE_DEG` 0.039 deg (a LOWER bound),
`DEFORMED_MASTER_STRAIN_TOL` 1.9e-05,
`DEFORMED_MASTER_SIGMA33_TOL` 3.7e-04 GPa, `SI_STRAIN_FLOOR`
2.5e-02, `SI_ROTATION_FLOOR` 2.5e-02 rad, `SI_KAM_FLOOR` 10.5 mrad,
`SI_PC_RESIDUAL_MEAN_TOL` 20.0 px. The six analytic ones reproduce
the reference-chain scales above to every digit. **The four Si ones
do not mean what the drafting expected and must not be quoted as
the method's precision**: the wafer dataset cannot support the
benchmark, for reasons measured in entries 43 and 44 and carried in
the module's own docstring.

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
  **SIGN ARM BUILT 2026-09-08 (Stage C adversarial review;
  requirements D14.2 amended with the same date).** The
  `omega_3(x1) = kappa*x1` recipe this bullet names was not in the
  drafted suite, and without it the "global sign pinned by V7"
  claim was undischarged: both test oracles transcribe the same
  `eps_jkl` contraction and every estimator sums moduli, so a
  coordinated sign convention change was invisible.
  `test_a_constant_lattice_curvature_gives_the_frozen_signed_alpha`
  now builds it and pins the SIGNED `alpha_13 = -kappa`, and
  `test_the_frozen_convention_is_minus_the_classical_nye_tensor`
  meets the contraction with the INDEPENDENT classical relation
  `alpha_ij = kappa_ji - delta_ij*kappa_kk`. MEASURED: the frozen
  convention is exactly minus the classical one. [D14.1/D14.2]
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
  **SELF-RULE ARM ADDED 2026-09-08 (Stage C adversarial review;
  requirements D14.5 clarified with the same date):** the
  non-converged case is pinned at the `nye_tensor` level as well as
  through `hrebsd_gnd`, because a central-difference pair never
  reads its own centre and a pair-only implementation would report
  the full neighbourhood density at a point that never converged
  (D2.6). [D2.6/D14.5]
- Si-wafer GND floor (`test_hrebsd_si.py::TestGndFloor`): record
  the SURVIVING FINITE FRACTION beside `SI_GND_FLOOR`, since the
  D14.5 NaN rule removes far more points than failed to converge,
  and the fix-toggle arm records the measured DIRECTION rather than
  asserting one (added 2026-09-08, Stage C adversarial review: on
  the smoke sub-grid the D14.3 fix RAISES the floor, 1.1237e11
  against 7.7823e10 m^-2, because this wafer's floor is dominated
  by the band-pass artefact of the Stage B ledger and not by the
  beta31/32 noise the 9.6x argument describes). [D14.3/D14.5/V5]

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
| Stage B tensor chain + KAM + segmentation on the 50x50 Si map (2500 points) | `v5_perf_ahe.py`, median of 5 after the map's own DIC run (2026-09-08, machine A), entry 46 | **`hrebsd_strain_stress` 6.13 ms, `segment_grains` 5.59 ms, `hrebsd_kam` 2.36 ms** -- 14.1 ms together, against 372.5 s for the DIC that feeds them |
| Stage B route end to end, 50x50 Si map, 480x480, default knobs | same run | **6.71 pat/s** (372.5 s, 2187/2500 converged, mean 34.1 iterations) |
| Stage C GND cost, `hrebsd_gnd` on the 50x50 Si map (2500 points) | `si_gnd_detail.py`, median of 5 after the map's own DIC run (2026-09-08, machine A), entry 68 | **10.78 ms** (1.594 ms on the 100-point smoke grid), against 509.5 s for the DIC that feeds it -- the most expensive of the four derived maps and 2.1e-05 of the route |

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
    built. **DISPOSITIONED 2026-09-08 (Stage C fix gate; the
    review found the line claimed a demonstration the notebook did
    not contain).** IQ is now demonstrated, and it is the one of
    the three that is load-bearing here: `get_image_quality()` is
    the score `reference="auto"` ranks each grain's candidates by,
    so the tutorial plots it for the Si map and shows that its
    maximum is the point `"reference_index"` names. IPF and the
    Hough-level misorientation map are NOT demonstrated and are
    dispositioned rather than added: both data sets in this
    tutorial are single-orientation by construction -- a synthetic
    map at one imposed orientation and a single-crystal wafer at a
    nominal identity -- so an IPF map is one flat colour and a
    Hough misorientation map is identically zero. Neither would
    show a reader anything, and the high-resolution misorientation
    map that IS meaningful on this data, HR-KAM, has its own
    checklist line above and is plotted twice. `pattern_matching`
    and `hough_indexing` are the tutorials that show IPF on a map
    with orientation contrast.
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

### 2026-09-08 (Stage B implementation gate, measurement agent)

**Machine A**, the same machine every Stage A number carries: the
20-core Windows 11 laptop (Intel64 Family 6 Model 186, Raptor Lake,
20 logical cores, Windows 11 build 26200), `.venv` Python 3.13.12,
numpy 2.4.6, scipy 1.17.1, numba 0.65.1, scikit-image 0.26.0, orix
0.14.2, dask 2026.3.0, pooch 1.9.0. Idle machine, warm numba caches.

This section discharges the Stage B measurement debt: the ten
unfilled `MTP` placeholders, the V5 Si-wafer benchmark (first
execution, the dataset fetched once here), plan open questions 2, 3,
4 and 10, and the Stage B performance row.

39. **The six analytic MTP pins.** Recipe: the Stage A convention of
    entry 1, a throwaway scratchpad pytest plugin
    (`hrebsd_measure_b/measure_plugin.py`, never in the repo) which
    sets every unfilled `None` placeholder to `+inf` and records what
    `assert_within` and `assert_at_least` were handed, so ONE run
    reports the worst measured value of every placeholder including
    the ones hidden behind an earlier failure in the same test.
    Command:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_stiffness.py -q -p measure_plugin
    -> 190 passed
    ```

    Every value below was reproduced BITWISE by a second run,
    including under randomized test order. Margin convention: 2x the
    measured worst case.

    | constant | measured worst | pinned | margin | every arm that consumes it |
    |---|---|---|---|---|
    | `REDUCED_CLOSURE_TOL` | 1.1654500102918543e-06 | 2.4e-06 | 2.06x | strain 1.1655e-06; e33 9.4132e-07 and 6.7578e-07; beta 9.5697e-07 and 6.6399e-08; rotation vector 8.4533e-08 |
    | `SIGMA33_TOL` | 3.0638726905057867e-04 GPa | 6.2e-04 | 2.02x | `test_sigma33_is_the_closure_self_check`, against a stress scale of 0.2420 GPa |
    | `SMALL_STRAIN_EQUALITY_TOL` | 5.889002203349758e-04 | 1.2e-03 | 2.04x | `test_equality_with_the_polar_path_is_measured`, worst of the sweep (2.0 deg) |
    | `SMALL_STRAIN_ENABLE_ANGLE_DEG` | 0.07794410136352782 deg | 0.039 | 2.00x, a LOWER bound | the same test, BISECTED to 1e-6 deg |
    | `DEFORMED_MASTER_STRAIN_TOL` | 9.404545683540204e-06 | 1.9e-05 | 2.02x | strain 9.4045e-06; e33 8.3928e-06 and 1.7138e-06; rotation vector 4.2064e-06 and 1.6741e-06 |
    | `DEFORMED_MASTER_SIGMA33_TOL` | 1.8337937617562972e-04 GPa | 3.7e-04 | 2.02x | `test_sigma33_is_zero_through_the_patterns` |

    Two things are worth recording beyond the table. First, **every
    one of these reproduces the failing-tests-gate reference chain of
    entries 30 and 31 to every digit printed there**, which is itself
    a finding: the implementation and the independent reference chain
    written before it agree, so the pins are not a measurement of the
    implementation against itself. Second, the small-strain sweep is
    recorded in full because requirements D8 asks for it -- errors
    9.1400e-08 (0.01 deg), 5.6564e-07 (0.05), 1.6043e-06 (0.1),
    3.7330e-05 (0.5), 1.4790e-04 (1.0), 5.8890e-04 (2.0), crossing
    the 1e-6 criterion at 0.0779 deg. **The fast path stays
    DISABLED**: D8 asks for the band and the angle to be recorded
    first, and `test_the_public_path_is_the_polar_one` pins that
    every public result still takes the polar split.

    The pins keep their mutants dead by orders. `REDUCED_CLOSURE_TOL`
    2.4e-06 against the Bond rotation transposed (2.9630e-04, 123x)
    and the frame rotation untransposed (8.3352e-04, 347x);
    `DEFORMED_MASTER_STRAIN_TOL` 1.9e-05 against the same Bond mutant
    through PATTERNS (3.0068e-04, 16x). `SIGMA33_TOL` kills nothing
    and its comment now says so: the Bond mutant measures 3.0914e-04
    GPa there against 3.0639e-04 correct, 0.9 per cent away (plan 3.4
    (a), entry 31).

40. **The V5 download, done once.** Command, run deliberately at this
    gate and never by a test:
    `kp.data.si_wafer(allow_download=True, lazy=True)`.
    **311 MB transferred in 63.0 s**, unpacking to **554.41 MB** in
    the kikuchipy cache (`Pattern.dat` 549.32 MB; the zip is deleted
    after unpacking, as the docstring says). What arrives: `(50, 50)`
    navigation by `(480, 480)` signal, `uint8`, navigation scale
    40.0 um on both axes with `units` "um", and a detector of shape
    `(480, 480)`, `pc = (0.5, 0.5, 0.5)`, `px_size = 1.0`,
    `binning = 1`, `sample_tilt = 70`, `tilt = 0`. The projection
    centre and the pixel size are both kikuchipy PLACEHOLDERS, which
    entries 41 and 43 turn out to matter enormously and hardly at all
    respectively.

41. **`px_size` was the placeholder 1.0, and V5 was the first caller
    to trip on requirements D6.1 (amended with this date).** The
    module built its per-point detector straight from the shipped
    one. MEASURED with `px_size = 1.0` and the wafer's 40 um step:
    **1800 px of modelled PCx drift, 1691 px of PCy and 616 px of
    detector distance on a 480 px detector**, and a reported strain
    floor of **1.033**, which fails the module's own frozen order
    check `strain_floor < 100 * LITERATURE_STRAIN_FLOOR` (2e-2) by
    52x. So the delivered module could not pass its own assertions,
    and the fix is at the call site: `per_point_detector` now sets
    `px_size = UF420_PX_SIZE = 90.0`, the NORDIF UF-420 unbinned
    pixel size kikuchipy's own `doc/tutorials/pc_fit_plane.ipynb`
    states for THIS dataset ("about 90 um", from which it derives the
    expected 2000 / 90 = 22 px shift across the scan). MEASURED with
    it: 20.0 px of modelled PCx drift and a strain floor of
    1.2168e-02. Recorded as a dated note in the test file and in
    requirements D6.1; the decision NOT to guard a placeholder
    `px_size` (entry 26 (a)) is unchanged and is now better evidenced.

    For the record, the projection-centre placeholder barely matters:
    substituting the Hough-refined mean of the same tutorial,
    `pc = (0.5195, 0.1554, 0.4870)`, moves the strain floor from
    1.2168e-02 to 1.2460e-02, 2.4 per cent, so the shipped
    `(0.5, 0.5, 0.5)` is kept and the tutorial's fitted number is not
    taken as a dependency.

42. **The V5 miscalibration arm was a provable no-op (requirements
    D13 amended with this date).** `test_si_pc_shift_plane`
    miscalibrated by `wrong.pc[..., 0] += 0.02`, a UNIFORM offset,
    and `beam_scan_model` is built from `PC_target - PC_reference`
    (the D6.2 closed form), from which a constant cancels
    identically. MEASURED: the mean of `residual_x` moves from
    **9.97390698866289 px to 9.973906988662913 px**, a relative
    2.3e-15, so the arm's `> 10 * worst` could never hold and no
    conformant implementation could satisfy it. CORRECTED to
    miscalibrate the scan-step to pixel-size RATIO, which is what a
    wrong beam-scan geometry really gets wrong and what does change
    the differences. MEASURED across `px_size` factors:

    | `px_size` | modelled PCx drift | residual mean | ratio to the consistent geometry |
    |---|---|---|---|
    | 90 um (consistent) | 20.0 px | 9.9739 px | 1.00 |
    | 45 um | 40.0 px | 19.974 px | 2.00 |
    | 9 um | 200.0 px | 99.974 px | 10.02 |
    | 4.5 um | 400.0 px | 199.97 px | 20.05 |
    | 180 um | 10.0 px | 4.9739 px | 0.50 |
    | 900 um | 2.0 px | 0.9739 px | 0.098 |

    The test now uses `px_size / 20`, which clears the frozen
    `> 10 *` by 2x rather than by the 0.2 per cent a factor of ten
    would leave. The ANALYTIC pin of the same signature, which
    offsets the MEASURED translations rather than the geometry, is
    `test_hrebsd_pc_shift.py::test_a_miscalibrated_projection_centre
    _shows_as_a_nonzero_mean` and is untouched and passing.

43. **The V5 floors, MEASURED AND PINNED -- and they are the
    DATASET's floors, not the method's.** Recipe: the module's own
    tests under the same measuring plugin,

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_si.py -q -p measure_plugin
    -> 3 passed, 6 skipped (--weekly), 49.7 s
    ... --weekly -> 10 min 16 s
    ```

    Route: `reference="auto"`, per-point projection centres from
    `extrapolate_pc` at the corrected `px_size` of entry 41, the
    deviatoric closure, RAW patterns (no upstream background removal,
    which requirements D4 leaves to the user and never implies), on
    the `[::5, ::5]` smoke sub-grid of 100 patterns.

    | constant | measured | pinned | margin |
    |---|---|---|---|
    | `SI_STRAIN_FLOOR` | **1.2168e-02** (smoke), **1.2208e-02** (full 50x50) | 2.5e-02 | 2.05x on the larger |
    | `SI_ROTATION_FLOOR` | **1.2006e-02** rad | 2.5e-02 | 2.08x |
    | `SI_KAM_FLOOR` | **5.2402** mrad | 10.5 | 2.00x |
    | `SI_PC_RESIDUAL_MEAN_TOL` | **9.9739** px (`residual_x`; 9.3568 px in `residual_y`) | 20.0 | 2.01x |

    Supporting numbers of the same run: 87 of 100 points converged,
    median residual 0.9496 (a pair ZNCC of 0.525), mean 33.3
    iterations of a cap of 50, median measured translation 0.026 px.
    The smoke sub-grid and the full map agree to 0.3 per cent, so the
    hundred-point estimate is not the noisy one the weekly full-map
    arm was added to guard against, and that arm is now a cheap
    regression check rather than the definitive number.

    **The strain floor is 1.2e-02 against the 1e-4 to 2e-4 literature
    class validation V5 recorded as context: two orders out.** That
    context is not refuted -- it is literature, and the V5 line
    "recorded, not presumed" was right to hedge -- but the benchmark
    it was meant to calibrate does not deliver. Entry 44 is the
    measurement that says why, and it is not the implementation. The
    KAM is internally consistent with the rotation floor it comes
    from: the D12 theory identity `sigma_omega * sqrt(2/N)` at
    1.2006e-02 rad and the frozen 8-neighbour kernel predicts
    6.0e-03 rad = 6.0 mrad, and 5.24 mrad is 13 per cent under that,
    so nothing is wrong with the KAM beyond the rotations it averages.

    Two smaller results recorded from the same runs. The weekly
    traction-free stress arm with the silicon stiffness returns a
    finite positive stress scale, as its (deliberately loose)
    assertions ask, and is not quoted as an absolute stress. And the
    strain floor is essentially independent of the closure and of
    upstream background removal -- 1.2168e-02 raw, 1.0834e-02 with
    the map average subtracted, 1.0442e-02 with a dynamic pass on top
    -- which is itself evidence that the number is set by the DIC's
    failure to track the geometry rather than by anything downstream
    of it.

44. **The control that separates the engine from the data, and the
    diagnosis.** Before any of entry 43 is read as a statement about
    the implementation, the engine was put on the SAME oracle Stage A
    uses, with a real wafer pattern in place of the synthetic one:
    warp a pattern by a known homography with the independent skimage
    warper (`test_hrebsd_engine.py`'s own `warp_with_skimage` and
    `random_small_homographies`, loaded by path) and refit. Recipe:
    `hrebsd_measure_b/control_warp.py`, four seed-0 homographies of
    the V2 design budget, `run_hrebsd_dic` with default knobs,
    D2.5 corner-displacement error in binned pixels.

    | reference pattern | worst error | all four | iterations | residual |
    |---|---|---|---|---|
    | Stage A synthetic 480 px oracle | **0.00725 px** | 0.00725 / 0.00158 / 0.00176 / 0.00457 | 4-7 | 0.00064-0.00071 |
    | si_wafer pattern (24, 25), raw | **0.03257 px** | 0.03257 / 0.00504 / 0.00431 / 0.00205 | 7-17 | 0.0443-0.0487 |
    | the same, map average subtracted | **0.05194 px** | 0.05194 / 0.01112 / 0.00958 / 0.00656 | 7-20 | 0.0823-0.0900 |

    All twelve converge, and the worst wafer number is 2.1x the
    pinned `WARP_REFIT_TOL_480` of 0.025 px, i.e. the same class. **So
    the engine measures a real wafer pattern's deformation to a few
    hundredths of a pixel; nothing in entry 43 is an engine defect.**

    What breaks is the PAIR. Measured with no engine involved, ZNCC
    of two DIFFERENT wafer patterns over the whole 480 px frame
    (`hrebsd_measure_b/probe_zncc.py`, an 11 by 11 block at the map
    centre):

    | pair | unfiltered | band-passed `(0.05, None)` | band-passed, map average subtracted |
    |---|---|---|---|
    | adjacent, 40 um apart | 0.9991 | **0.8424** | **0.4862** |
    | five steps, 200 um | 0.9958 | 0.7729 | 0.2531 |
    | opposite corners of the block | 0.9966 | 0.7492 | 0.1908 |

    against 0.978 for the synthetic-warp control on the same pattern
    (residual 0.0443). Two further numbers name the culprit. (a) The
    band-passed pattern correlates only **0.7514** with its own
    5-point smoothed self once the map average is subtracted (0.9264
    before), so most of what survives the band-pass there is
    pixel-scale noise. (b) `phase_cross_correlation` on band-passed
    pairs returns **exactly 0.0 px** for every pair tested --
    (0, 1), (1, 0), (0, 10), (10, 0), (0, 49), (49, 49), (25, 25) --
    and -0.0625 px for (49, 0). A component that survives a high-pass
    and does NOT move with the beam is pinning the correlation at
    zero shift, and the fits sit in that minimum: the median measured
    translation over the smoke map is **0.026 px** (largest 1.88 px)
    where the beam-scan model puts 20.0 px.

    The drift the fits miss is real and independently measured:
    kikuchipy's own Hough projection-centre fit of this map
    (`doc/tutorials/pc_fit_plane.ipynb`) reports a PCx standard
    deviation of 0.0241 over a grid spanning it, **11.6 detector
    pixels**, with a deviation from its own fitted plane of only
    0.0021. So the geometry moves and the correlation does not see
    it.

    **Subtracting the map average is not the remedy on a
    single-crystal scan**, and this is the one place the measurement
    contradicts ordinary EBSD practice: there the map average IS the
    Kikuchi pattern, so `remove_static_background` removes the signal.
    MEASURED on the smoke map: convergence falls from 87/100 to
    **11/100** (12/100 with a dynamic pass on top), the median
    residual rises from 0.9496 to 1.9477, and the mean iteration
    count from 33.3 to 48.7 of a cap of 50. The recorded floors are
    therefore taken on the RAW route, which is also what
    requirements D4 means by default knobs (upstream background
    removal is the user's choice and never implied).

    Recorded consequence for Stage C's tutorial and for anyone
    quoting these numbers: **the si_wafer dataset was acquired for
    projection-centre calibration**, which reads band POSITIONS and
    is untroubled by a fixed-pattern component; high angular
    resolution DIC reads band SHAPES and is not. Plan open question
    13's Si-indent dataset, deferred to after Stage C, is where a
    method-level floor can be measured, and the V5 module docstring
    now carries this whole paragraph so the number is never read
    alone.

45. **The four sweeps, and plan open questions 2, 3, 4 and 10
    RESOLVED. Every frozen default is CONFIRMED; none is re-pinned.**
    Recipe: `hrebsd_measure_b/v5_sweeps.py`, which reproduces the
    module's own weekly sweeps and prints the table the tests only
    assert on. Same route as entry 43, the smoke sub-grid of 100
    patterns, one knob changed per arm.

    **Open question 10 -- preprocessing defaults (requirements D4.1,
    D4.2, D4.3; the default `(0.05, None)`, no AHE, no window,
    CONFIRMED with this date).**

    | `filter_cutoffs` | converged | median residual | mean iterations | strain floor |
    |---|---|---|---|---|
    | **`(0.05, None)`, frozen** | **87/100** | 0.9496 | 33.3 | 1.2168e-02 |
    | `(None, None)` | **1/100** | 0.0534 | 49.5 | not measurable |
    | `(0.05, 0.4)` | 89/100 | 0.8806 | 32.1 | 1.2246e-02 |
    | `(None, 0.4)` | **1/100** | 0.0482 | 49.5 | not measurable |
    | `window=False`, frozen | 87/100 | 0.9496 | 33.3 | 1.2168e-02 |
    | `window=True` (Hann) | 14/100 | 0.2691 | 48.9 | 1.2474e-02 |

    **This is the measurement plan open question 10 was created for,
    and it comes out the OPPOSITE way to the noise-free oracle.**
    Stage A entry 21 measured that the high-pass HALVES the capture
    range and costs a factor of ten in accuracy on synthetic
    patterns, and withdrew a requirements claim over it while
    explicitly refusing to re-pin the default because "on noise-free
    oracles a high-pass can only remove signal, and its purpose --
    background gradients on real data -- is the V5 Si-wafer
    measurement". On real data the high-pass is not a luxury:
    **without it 1 of 100 patterns converges instead of 87.** The
    default stands, and the two records now bracket it honestly --
    it costs capture range on clean patterns and buys the whole
    measurement on dirty ones.

    The low-pass is the one arm that could have argued for a re-pin
    and does not: `(0.05, 0.4)` converges 89 against 87 and drops the
    median residual from 0.9496 to 0.8806, but leaves the strain
    floor 0.6 per cent WORSE (1.2246e-02 against 1.2168e-02). Two
    points of convergence is not a dated re-pin of a frozen default,
    and `(0.05, None)` stays. The D4.3 Hann window is refuted as a
    default here as clearly as anywhere: 14 of 100 converge with it.
    (Its low median residual, 0.2691, is not a win -- it is the
    taper's own weighting of the criterion, and the 86 points which
    never converged carry it.)

    **Open question 3 -- border and dead-band (requirements D4.4;
    `border=0.05`, `dead_band=None` CONFIRMED with this date; the
    plan's "pin the knee" answered NO KNEE EXISTS here).**

    | `border` | converged | median residual | mean iterations | strain floor |
    |---|---|---|---|---|
    | 0.0 | 93/100 | 0.6211 | 32.2 | 1.2003e-02 |
    | **0.05, frozen** | 87/100 | 0.9496 | 33.3 | **1.2168e-02** |
    | 0.1 | 83/100 | 1.4393 | 37.0 | 1.2146e-02 |
    | 0.2 | 52/100 | 1.7744 | 45.4 | 1.3364e-02 |

    The floor moves 11 per cent over a fourfold change in the knob
    while convergence falls from 93 to 52, so **the floor has no knee
    in the border at all** on this dataset and the sweep cannot pin
    one. It would be wrong to read the flat trend as "use
    `border=0.0`": D4.4 chose 0.05 to cover the few-pixel beam-scan
    translations expected at 480 px, and on THIS dataset the fits do
    not track translations (entry 44), so it is exactly the quantity
    the border exists for that the data cannot exercise. The default
    stands unchanged, on the D4.4 reasoning rather than on this
    table. `dead_band` is untouched: the wafer has no dead cross to
    exclude, and the knob keeps its frozen `None`.

    **Open question 2 -- convergence defaults (requirements D2.5 and
    D2.6; `min_step = 1e-3` px and `max_iterations = 50` CONFIRMED
    with this date).**

    | knob | converged | mean iterations | strain floor | time |
    |---|---|---|---|---|
    | `min_step = 1e-2` | 97/100 | 11.3 | 1.2198e-02 | 7.9 s |
    | **`min_step = 1e-3`, frozen** | 87/100 | 33.3 | **1.2168e-02** | 15.7 s |
    | `min_step = 1e-4` | 26/100 | 48.2 | 1.2416e-02 | 21.0 s |
    | `max_iterations = 10` | 1/100 | 9.9 | not measurable | 8.2 s |
    | `max_iterations = 20` | 8/100 | 19.7 | 1.3133e-02 | 11.5 s |
    | **`max_iterations = 50`, frozen** | 87/100 | 33.3 | 1.2168e-02 | 15.7 s |
    | `max_iterations = 100` | 96/100 | 36.5 | 1.2175e-02 | 18.1 s |

    Two findings. (a) **The floor is FLAT to 2 per cent across a
    hundredfold change in `min_step`** (1.2168e-02 to 1.2416e-02),
    which completes the error-versus-threshold curve plan open
    question 2 asks for and answers it: on real data the convergence
    threshold is NOT what limits the result, so the D2.5 default
    leaves no systematic error above the floor and needs no re-pin.
    A looser 1e-2 converges 97 of 100 in a third of the iterations
    for the same floor, and it is tempting -- but the argument is
    valid only where the fits are meaningless anyway, and Stage A
    entry 10 measured that on the oracle the frozen 1e-3 already
    exits in 5.2 iterations at the interpolation floor, so 1e-3 costs
    nothing where the measurement works. Re-pinning a frozen default
    on a dataset the method cannot measure would be exactly the
    mistake D19 exists to prevent. **NOT re-pinned; recorded.**
    (b) **`max_iterations = 50` is close to a real knee and is doing
    work**: 10 gives 1 of 100, 20 gives 8, 50 gives 87, 100 gives 96.
    The EMsoftOO namelist default turns out to be well chosen for
    noisy data, and a smaller cap would be badly wrong. Raising it to
    100 buys 9 points of convergence for 15 per cent more time and no
    change in the floor (1.2175e-02), which is not a re-pin either.

    **Open question 4 -- KAM kernel defaults (requirements D12;
    `order = 1`, `psi_max = None`, same-grain always, CONFIRMED with
    this date).** Same run, the KAM recomputed on the stored
    rotations, so the arms differ only in the kernel.

    | kernel | median KAM | mean | std | points | cost, 100 pts |
    |---|---|---|---|---|---|
    | **`order = 1`, frozen (8 neighbours)** | **5.2402 mrad** | 5.2105 | 0.1694 | 86 | 0.34 ms |
    | `order = 2` (24 neighbours) | 8.2715 | 8.2588 | 0.5364 | 87 | 0.77 ms |
    | `order = 3` (48 neighbours) | 11.1350 | 11.1790 | 0.5253 | 87 | 1.42 ms |
    | `psi_max = None`, frozen | 5.2402 | -- | -- | 86 | -- |
    | `psi_max = 5.0` mrad | 4.3685 | -- | -- | 86 | -- |
    | `psi_max = 1.0` mrad | NaN | -- | -- | **0** | -- |

    Plan open question 4 asks for "the noise/resolution trade" and
    expects one; **there is none, because order 1 wins on BOTH axes**.
    The rotations here are uncorrelated point to point, so widening
    the kernel does not average the noise down, it reaches further
    into it: the median KAM grows 5.24 -> 8.27 -> 11.14 mrad and its
    spread triples, while the spatial resolution gets worse and the
    cost quadruples. `order = 1` stays frozen, now with a measurement
    behind it rather than a convention. `psi_max` behaves exactly as
    D12 documents: at 1.0 mrad it drops every pair on a map whose
    rotation floor is 12 mrad, leaving nothing -- correct behaviour
    for a guard aimed at sub-grain boundaries, and the reason its
    default is None.

    **What none of these sweeps can do**, recorded so the next reader
    does not over-read them: on a dataset where the DIC has no signal
    (entry 44) every arm's floor is set by the same thing, which is
    why three of the four tables are nearly flat in their floor
    column. What the sweeps DO discriminate is CONVERGENCE, and there
    the differences are large, unambiguous and consistent with every
    frozen decision. When plan open question 13's Si-indent dataset
    lands after Stage C, all four are worth re-running for their
    floor columns.

    **Test-module consequence, applied with this date.** The two
    sweeps whose arms stop converging -- `(None, None)` at 1 of 100
    and `min_step = 1e-4` at 26 of 100 -- ABORTED inside
    `worst_component_std`, which refuses a map where over half the
    points failed. That refusal is right for a floor which gets
    pinned and wrong for a sweep arm, and it contradicted this
    class's own "RECORDED, NOT GATED" docstring, validation V5's
    "measurement harnesses ... results recorded" line and the V4
    `test_rotation_sweep_capture_range` precedent, which records a
    sweep most of whose arms fail. New helper `sweep_floor` returns
    `(nan, 0)` instead of aborting, and each sweep now gates on its
    FROZEN DEFAULT arm (`assert_default_arm_recorded`: finite,
    positive, over half the map) plus a new
    `assert_the_sweep_varied_something`, which is strictly more
    killing power than the drafted `all(isfinite)` gave -- a sweep
    that silently ignored its knob used to pass and now does not.

46. **The Stage B performance row, at the real Si map scale
    (recorded, never a gate -- D16).** Recipe:
    `hrebsd_measure_b/v5_perf_ahe.py`, the FULL 50 by 50 map (2500
    points, 480x480), default knobs, per-point projection centres,
    each Stage B function timed as the median of five calls on the
    map the DIC run produced, warm caches, idle machine.

    | measurement | 2500 points | 100 points (smoke) |
    |---|---|---|
    | `hrebsd_strain_stress` | **6.13 ms** (min 6.00) | 1.10 ms |
    | `segment_grains` | **5.59 ms** (min 5.27) | 0.62 ms |
    | `hrebsd_kam`, `order=1` | **2.36 ms** (min 2.34) | 0.49 ms |
    | all three together | **14.1 ms** | 2.2 ms |
    | the `EBSD.hrebsd_dic` run that feeds them | **372.5 s** | 15.7 s |

    So **the whole Stage B chain costs 3.8e-05 of the run it
    post-processes**, and the D16 position that these are recorded
    baselines and never gates needs no defending: on this map the
    engine is 26000 times the cost of everything Stage B adds. All
    three scale close to linearly from the smoke grid (5.6x, 9.0x and
    4.8x for 25x the points; the sub-linear ones are per-call
    overhead the 100-point numbers are dominated by).

    Engine throughput on this route for the record: **6.71 pat/s**
    (2187 of 2500 converged, mean 34.1 iterations), against the Stage
    A oracle baseline of 22.72 pat/s at the same 480x480 size (entry
    25). The factor of 3.4 is entirely iteration count -- 34.1 here
    against 4.75 there -- which is the same real-data fact entries 44
    and 45 measure from every other angle, not a performance
    regression. The full-map strain floor from this independent run,
    1.2208258467329113e-02, reproduces the weekly test's number to
    every digit.

47. **Requirements D4.2's adaptive-histogram-equalization arm, the
    last piece of plan open question 10 (CONFIRMED with this date).**
    D4.2 refuses AHE in the DIC chain on a theoretical argument -- it
    is a nonlinear, locally varying intensity map, and ZNSSD's
    invariance is only affine -- and asks that "the Si benchmark (V5)
    measures the with/without comparison once and records it before
    any reconsideration". The delivered sweep does not cover it,
    because AHE is an UPSTREAM kikuchipy call
    (`EBSD.adaptive_histogram_equalization`) and not a knob of
    `hrebsd_dic`, so it is measured here instead. Recipe:
    `hrebsd_measure_b/v5_perf_ahe.py`, the same smoke grid and route
    as entry 43, kikuchipy's default AHE kernel.

    | preprocessing | converged | median residual | mean iterations | strain floor | KAM |
    |---|---|---|---|---|---|
    | **none (frozen)** | **87/100** | 0.9496 | 33.3 | 1.2168e-02 | 5.240 mrad |
    | AHE, default kernel | **5/100** | 1.6346 | 49.3 | 1.2560e-02 | 6.358 mrad |

    **The theoretical argument is confirmed by measurement**:
    convergence collapses from 87 per cent to 5 per cent, the median
    residual rises 72 per cent and the mean iteration count goes to
    49.3 of a cap of 50, i.e. essentially every fit runs out of
    iterations. EMsoftOO's shared DI preprocessing applies AHE with
    `nregions=10` (`mod_HREBSDDIC.f90:689-733`) and this fork's
    deviation from it is now evidenced rather than argued.
    Reconsideration is closed.

48. **Gate outcome and everything still open.** With the ten pins of
    entries 39 and 43, the three test corrections of entries 41, 42
    and 45, and no change to any `_hrebsd/` module:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_stiffness.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_interpolation.py -q
    ```

    ```
    -> 437 passed, 1 skipped (the weekly capture-range sweep), 21.9 s
    ```

    and the V5 module, which is not in that list and which the cached
    dataset now makes EXECUTABLE rather than skipped:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_si.py -q
    ->  3 passed, 6 skipped (--weekly), 49.7 s

    ... the same with --weekly
    ->  9 passed, 681.1 s (11 min 21 s)
    ```

    So every V5 test passes, weekly arms included, where before this
    gate the module skipped entirely. `ruff check` and
    `ruff format --check` clean on every edited file, and no
    `_hrebsd/` module changed: the only source edits at this gate are
    the ten pins and the three test corrections.

    NOT measured at this gate and still owned by their own gates: the
    Stage C GND floor on the same wafer (`test_si_gnd_floor`), the
    weekly 960x960 warp-refit, the D3 bicubic-versus-quintic re-run,
    the Stage B coverage run, and the Stage B local oldest-matrix
    run. Plan open questions 5, 6, 7, 8 and 12 are unchanged
    deferrals; 1, 9 and 11 closed at Stage A; **2, 3, 4 and 10 close
    here**; 13 is the post-Stage-C Si-indent application, which
    entries 43 to 45 now make considerably more interesting -- it is
    the dataset on which a method-level noise floor can be measured
    at all.

49. **Full existing suite, and the one failure it stops on.**

    ```
    .venv/Scripts/python.exe -m pytest tests -q -x --ignore=tests/test_data
    ->  1 failed, 4491 passed, 801 skipped, 5 rerun in 262.29 s
    ```

    The one failure is
    `tests/test_simulations/test_kikuchi_pattern_simulator.py
    ::TestCalculateMasterPattern::test_shape`, the SAME upstream flaky
    test that entry 4 of the Stage A implementation gate and entry 27
    of the Stage A review both logged on this machine, precisely so
    that a future red run would not be misread as an HREBSD
    regression. It is NOT a blocker and the evidence is direct rather
    than historical:

    - `git diff HEAD -- src/kikuchipy/simulations tests/test_simulations`
      is EMPTY on this branch, so neither the code nor the test moved.
    - It carries upstream's own `@pytest.mark.flaky(reruns=5)` and its
      assertion is `np.allclose(mp.data[0], mp.data[1], atol=1e-4)` on
      two 201x201 master-pattern hemispheres.
    - RUN IN ISOLATION five times on identical inputs at this gate --
      `pytest "...::test_shape" -q -p no:randomly`, nothing else
      collected, no HREBSD module reachable -- it **passed 2 and
      failed 3**, each failure exhausting all five of its own reruns.
      So it does not need the HREBSD suite to have run first in order
      to fail, which is the strongest form of the disposition entry 27
      reached from history alone.

    Deselecting it, the suite is GREEN:

    ```
    .venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_data \
      --deselect "tests/test_simulations/test_kikuchi_pattern_simulator.py
                  ::TestCalculateMasterPattern::test_shape"
    ->  4513 passed, 803 skipped, 1 deselected, 257.03 s
    ```

    So no non-HREBSD test breaks, and the 22 tests the count gains
    over the `-x` run are the ones that run after the flaky one plus
    the three V5 tests the cached dataset now makes executable.

50. **One transient interpreter crash, observed ONCE and not
    reproduced; recorded rather than suppressed.** The first
    deselected full-suite run terminated with a Windows fatal-exception
    stack dump whose frames were pytest's own shutdown path
    (`pluggy/_hooks.py:512`, `_pytest/config/__init__.py:199` and
    `:223`, `pytest/__main__.py:9`, `runpy`), with no test failure
    reported before it. The IDENTICAL command re-run immediately
    afterwards completed cleanly with the 4513 passed above, so the
    crash is not reproducible on demand. Two things are worth
    recording for whoever meets it next. It happened on the first run
    in which `test_hrebsd_si.py` was BOTH executable (the dataset
    having just been cached) and collected alongside the whole suite,
    so the process peak memory is higher than any earlier Stage A or
    Stage B run: the V5 tests hold 100 eager 480x480 patterns plus
    their dask graphs. And `pytest-randomly` reseeds the collection
    order per run, so the two runs did not execute in the same order.
    It is NOT attributed to any HREBSD code here, because nothing
    supports that attribution beyond coincidence of timing; it is
    logged so that a second occurrence is recognised as a second
    occurrence and investigated with `-p no:randomly` and a memory
    watch rather than rediscovered.

### Stage B adversarial review, fix gate (2026-09-08)

51. **The PC-shift diagnostic was reading non-converged fits, and it
    had reached a pin.** `hrebsd_pc_shift` promised in its docstring
    and in an inline comment that "points which did not converge,
    failed or were masked out are NaN", and built the exclusion out of
    `numpy.isfinite(homography)` alone. Requirements D2.6 keeps a
    non-converged point's finite LAST ITERATE, so that test cannot see
    one, and `converged` was not in `REQUIRED_PROP_NAMES`. MEASURED,
    by re-running entry 43's recipe (the 100-point Si smoke grid,
    `reference="auto"`, per-point PCs at `px_size = 90 um`):

    | | all 100 points | the 87 converged |
    |---|---|---|
    | `mean(residual_x)` | 9.973907 px | **10.995582 px** |
    | `mean(residual_y)` | 9.356793 px | **9.840273 px** |
    | `median(abs(translation_x))` | 0.026 px | 0.045 px |

    The 13 abandoned fits pulled the reported mean DOWN by 10.2 per
    cent, and `SI_PC_RESIDUAL_MEAN_TOL` was pinned at 2x that mixture.
    FIXED in `_pc_shift.py`: `"converged"` joins the required
    properties and `~converged` joins `dropped`. RE-PINNED:
    `SI_PC_RESIDUAL_MEAN_TOL = 2.2e01` (2x 10.9956), with the recorded
    numbers corrected in the module and in requirements D13.

    The gap was invisible because the guarding class,
    `test_hrebsd_pc_shift.py::TestFailedPoints`, only ever injected a
    NaN homography -- a FAILED pattern, never a non-converged one.
    Three tests now cover it: a synthetic non-converged point with a
    finite homography, a discrimination arm showing the residual mean
    move by 7/6 px when such a point is counted, and
    `TestThroughTheEngine`, which runs `EBSD.hrebsd_dic` at
    `max_iterations=1` on `nickel_ebsd_small` so that the engine
    itself produces the finite-homography-with-`converged=False`
    state, and asserts the NaN pattern of all seven maps equals
    `~converged` exactly. The frozen assertions of that module are
    untouched; its `phantom_map` FIXTURE gained `converged`, which is
    a property the engine always stores.

52. **`reference="auto"` ignored `navigation_mask` entirely.**
    `resolve_reference` ran BEFORE the mask was applied and
    `select_references` had no mask argument, so the highest-quality
    pattern of a grain was chosen as its reference even when the
    caller had masked it out. DEMONSTRATED on `nickel_ebsd_small`:
    with the argmax-quality point masked, `reference_index` came back
    naming it at every point of its grain, while that point's own
    `homography` was NaN and `converged` False -- every strain in the
    grain measured against a pattern the caller had excluded. FIXED:
    the mask is threaded into `resolve_reference` and into a new
    `selectable` argument of `select_references`; `grain_id` is
    unchanged by the mask, and a grain with nothing selectable keeps
    the unrestricted choice (no pattern of it is correlated, so the
    index is never read). Requirements D11.2 records both clauses.
    Pinned by `test_hrebsd_segmentation.py::TestAutoReference::
    test_a_masked_out_point_is_never_a_reference` and
    `test_ebsd_hrebsd_dic.py::TestAutoReference::
    test_a_masked_out_pattern_is_never_the_reference`, which also
    asserts the reference is now a fitted point.

53. **The frozen default materialised the whole pattern stack.**
    `select_references` did `numpy.asarray(patterns)` on the same
    (possibly Dask) array the engine otherwise keeps lazy: 576 MB on
    the shipped Si wafer before the first fit, about 57 GB on a
    500 by 500 map of 480 by 480 patterns. MEASURED with a Dask array
    of chunk size 1 whose every block load is counted on
    `nickel_ebsd_small`: 11 block loads through `reference="auto"`
    with 8 of 9 points masked out, against 2 for an explicit
    reference. FIXED: `image_quality` takes optional `indices` and
    reads them in blocks of about 64 MB
    (`IMAGE_QUALITY_BLOCK_BYTES`), and `select_references` asks only
    for the candidates. The kernel is per-pattern either way, so no
    returned number changes; the block-counting test now measures 2
    of 9 patterns read when 2 are asked for. The public Memory note of
    `EBSD.hrebsd_dic` and requirements D11.2 and D16 say so.

54. **A column-oriented one dimensional scan could not use the frozen
    default.** `EBSD.hrebsd_dic` set `engine_nav_shape = (1, n)` for
    any one dimensional navigation while `segment_grains` takes its
    shape from the map's own grids (D11.1(b)). REPRODUCED: a
    three-point signal with a crystal map built from `y = arange(3.0)`
    has `xmap.shape == (3,)`, so the public guard passes, and
    `reference="auto"` then raised `ValueError: segment_grains
    returned labels of shape (3, 1), which must be the navigation
    shape (1, 3)`. FIXED by reading `xmap.row`/`xmap.col`, with the
    single scan step given to the ROW axis of a column scan. No extra
    guard was added for a map which is neither a row nor a column:
    orix reports a diagonal three-point map as shape `(3, 3)` and the
    existing equality check rejects it (MEASURED, and pinned).

55. **A phase with no point group silently changed the segmentation.**
    orix leaves `Orientation(..., symmetry=None)` at C1, so
    `angle_with` returns the raw angle. MEASURED: a 2 by 2 map of
    0/90/0/90 degrees about z segments to one grain with
    `point_group="m-3m"` and to two without it. There is no symmetry
    to invent and refusing such a phase would make the frozen default
    raise on input the rest of kikuchipy accepts, so `segment_grains`
    now emits a `UserWarning` naming the phase and segments on the raw
    angles (requirements D11.1(d)). No test in the suite triggers it:
    every phase in the hrebsd tests carries a space group or a point
    group.

56. **Two survivors of the review's own mutation probe, killed.**

    - `_kam.py`, `angles <= psi_max` -> `angles < psi_max`. It
      contradicts requirements D12 ("drops pairs ABOVE the
      threshold") and is not a rounding-scale difference: MEASURED on
      a one-pair map, the correct form returns the pair angle and the
      mutant returns NaN everywhere. `TestPsiMaxBoundary` feeds
      `psi_max` the angle the module itself reported on a map with
      exactly one pair, so the boundary is exact rather than a
      tolerance, and cross-checks that angle against the module's own
      independent oracle first. RE-INJECTED: 1 failed, 37 passed;
      RESTORED: 38 passed.
    - `_tensors.py`, `symmetric = 0.5 * (stacked + swapaxes(...))` ->
      `symmetric = stacked`. It has no effect on any public number
      today (the only caller feeds `strain_from_stretch`, whose worst
      asymmetry is 3.4e-17), but the function's docstring promises the
      symmetric part is taken and D15.6 makes the result a public
      property with TENSOR shears, so the Stage C antisymmetry fix --
      which works on a deliberately non-symmetric gradient -- would
      read every shear twice over. Pinned with an asymmetric input,
      `[1, 4, 6, 2.5, 1.5, 1.0]` against the mutant's
      `[1, 4, 6, 5, 3, 2]`. RE-INJECTED: 1 failed, 69 passed;
      RESTORED: 70 passed.

57. **Coverage, 100 per cent of the Stage B modules, and the four
    smaller findings the gaps carried.** The Definition-of-done
    command, run with `pytest-cov` installed into a scratchpad
    directory and put on `PYTHONPATH` (the entry-23 route), measured
    **99.11 per cent, 11 lines missed** before this gate:
    `_kam.py:107`, `_reference.py:227,232`, `_segmentation.py:210,
    307-308`, `_stiffness.py:393`, `_tensors.py:615,819,852,907`.
    Each was closed on its merits rather than by touching the line:

    - `_segmentation.py:307-308` is DEAD, not untested: every edge
      offset is skipped only on a one-by-one grid, which
      `_map_grid` cannot reach. Pruned, and the one input that would
      have reached it -- a map orix gives the shape `()` -- now raises
      a `ValueError` naming the map from a shared `map_grids` helper
      the three grid-shaped functions read their grids through,
      instead of orix's `not enough values to unpack` (requirements
      D11.1(f)).
    - `_stiffness.py:393` was worse than untested. The broadcast of
      ONE strain over a stack of stiffnesses was built and then all
      but its first row silently discarded, because `single` was read
      off the strain alone. It now returns `(n, 6)`, which is what
      asking that question means, and is pinned against the per-point
      loop.
    - `_tensors.py:819,907` are the private chain's shape guards,
      which the public function no longer breaks (entry 58): tested
      directly. `:852` is the all-points-NaN early return of D2.6,
      now pinned to return the full seven-key property set at full
      length. `:615` is the stacked return of `small_strain_split`.
    - `_kam.py:107` is the non-integer `order` guard, and
      `_reference.py:227,232` the two `_flatten_labels` guards, the
      first reachable through `resolve_reference` with a navigation
      shape that disagrees with the map's grids.

    ```
    PYTHONPATH=<scratchpad>/covpkgs .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_stiffness.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_interpolation.py \
      --cov=src/kikuchipy/indexing/_hrebsd --cov-report=term-missing -q
    ->
      __init__.py          0 stmts, 0 miss, 100.00%
      _engine.py         270 stmts, 0 miss, 100.00%
      _geometry.py        55 stmts, 0 miss, 100.00%
      _homography.py      62 stmts, 0 miss, 100.00%
      _interpolation.py  186 stmts, 0 miss, 100.00%
      _kam.py             77 stmts, 0 miss, 100.00%
      _pc_shift.py        57 stmts, 0 miss, 100.00%
      _preprocessing.py   70 stmts, 0 miss, 100.00%
      _reference.py       74 stmts, 0 miss, 100.00%
      _segmentation.py   154 stmts, 0 miss, 100.00%
      _stiffness.py       82 stmts, 0 miss, 100.00%
      _tensors.py        184 stmts, 0 miss, 100.00%
      TOTAL             1271 stmts, 0 miss, 100.00%
      474 passed, 1 skipped in 23.5 s
    ```

    `EBSD.hrebsd_dic` itself is fully covered too: the missing ranges
    of `signals/ebsd.py` under the same run (2324-2480, 3030-3212)
    lie outside the method.

58. **The two modules disagreed about a multi-rotation map.**
    `segment_grains` read the best of `(n, k)` rotations;
    `hrebsd_strain_stress` raised `orientation_matrices must have
    shape (n, 3, 3) ... but has shape (12, 3, 3, 3)`, naming a
    parameter of the private chain that the public caller never
    passed and that its documented `Raises` does not mention. Because
    `EBSD.hrebsd_dic` deep-copies the input map, such a map survived
    the whole Stage A run and failed only at the documented next step.
    Both now take the best rotation (requirements D11.1(e)); the
    private `tensor_chain` keeps its strict contract, tested directly.

59. **numpydoc RT02, the one convention Stage B broke.** The repo
    enables `"all"` minus a listed set that does not exempt RT02
    (`doc/conf.py:317-333`), and the four new public functions gave a
    NAMED single return with a type where the whole package gives the
    type alone: `hrebsd_strain_stress`, `segment_grains`,
    `hrebsd_kam`, `hrebsd_pc_shift`. Stage A's six modules and the
    `kikuchipy.indexing` baseline have none. FIXED by dropping the
    ` : <type>` and keeping the name, as `EBSD.hrebsd_dic` and
    `voigt_stiffness` already do. Re-run of the repo's own exempt set
    over Stage B, Stage A and the baseline: **0 non-exempt numpydoc
    errors in all three**.

60. **The Stage B local oldest-matrix run, the last unrecorded
    Definition-of-done item.** Entry 24's recipe, after the fixes
    above:

    ```
    uv run --isolated --python 3.10 \
      --with "numpy==1.23.0" --with "numba==0.57" \
      --with "orix==0.12.1" --with "scikit-image==0.21.0" \
      --with pytest-benchmark --with pytest-rerunfailures \
      --with pytest-xdist --with pytest-randomly \
      pytest tests/test_indexing tests/test_signals -k hrebsd -q
    ->  477 passed, 7 skipped, 4335 deselected in 65.18 s
    ```

    Environment as resolved: Python 3.10.19, numpy 1.23.0, scipy
    1.13.1, numba 0.57.0, orix 0.12.1, scikit-image 0.21.0, dask
    2024.8.1. The 477 include the three executable V5 tests, so the
    orix floor is discharged on the real-data route as well as the
    synthetic ones, and it now covers the orix calls this gate added:
    `CrystalMap.shape` on a grid-less map and `rotations.to_matrix()`
    on a multi-rotation map.

61. **The stress path is 27x the recorded Stage B tensor baseline,
    and the baseline said nothing about it.** Entry 46 records
    `hrebsd_strain_stress` on entry 43's recipe, which passes
    `stiffness=None`, so `rotate_stiffness` is never reached. MEASURED
    on machine A, best of five:

    | | 100 points | 2500 points |
    |---|---|---|
    | `rotate_stiffness` | 7.38 ms | 182.44 ms |
    | `hrebsd_strain_stress`, deviatoric | -- | 7.21 ms |
    | `hrebsd_strain_stress`, traction free | -- | 192.93 ms |
    | vectorised `einsum`, `optimize=True` | -- | 4.42 ms |

    The vectorised form agrees with the per-point loop to 1.99e-13,
    far inside `REDUCED_CLOSURE_TOL = 2.4e-06`, but NOT bitwise, and
    the loop exists precisely so that a stacked call is the loop to
    the last bit. 190 ms on a map whose fits take hours does not buy
    that back, so the loop stands and the number is RECORDED
    (requirements D16), which is what D16 says performance is for.

62. **A latent NaN cast in `hrebsd_kam`.** `grain_id` was cast to
    int64 without a dtype check, so a FLOAT `grain_id` carrying NaN
    emitted `RuntimeWarning: invalid value encountered in cast` and
    relied on the platform's undefined NaN-to-int result being
    negative to be treated as unlabelled. The documented property is
    int32 and takes the same path unchanged, so this is latent rather
    than live; it is fixed anyway (`numpy.where(isfinite, raw, -1)`)
    because the module's NaN discipline is explicit everywhere else,
    and pinned with a `RuntimeWarning`-as-error test.

63. **Narration corrected: nothing raises `NotImplementedError` any
    more.** Every hrebsd test module still told its reader that "every
    test which calls the module fails with `NotImplementedError` until
    X lands", and `test_ebsd_hrebsd_dic.py`'s `run()` helper still
    called the frozen default a Stage B thing. `grep -rn
    "raises(NotImplementedError" tests/ src/` returns no hrebsd hit.
    All twelve docstrings are rewritten in the past tense, each saying
    which failing-tests gate it was written at. No assertion moved.

64. **Fix-gate outcome.**

    ```
    .venv/Scripts/python.exe -m pytest <the eleven modules above> -q
    ->  474 passed, 1 skipped (the weekly capture-range sweep), 21.4 s

    .venv/Scripts/python.exe -m pytest tests/test_indexing/test_hrebsd_si.py -q
    ->  3 passed, 6 skipped (--weekly), 49.2 s
    ```

    `ruff check src tests` -> All checks passed.
    `ruff format --check src tests` -> the one file it would reformat,
    `tests/test_draw/test_ebsd_detector_plots_widgets.py`, is
    untouched by this branch (`git diff --name-only` does not list
    it); every file this gate edited is formatted.

    Source changed at this gate, which entry 48 could still say was
    empty: `_pc_shift.py` (the convergence contract, the anchor
    documentation, the shared grid read), `_segmentation.py` (the
    selectable set, block-wise scoring, the point-group warning, the
    `map_grids` guard, the pruned dead branch), `_reference.py` (the
    mask threading), `_engine.py` (passing the mask on),
    `_kam.py` (the NaN-safe cast, the shared grid read, the `psi_max`
    docstring), `_tensors.py` (the best rotation), `_stiffness.py`
    (the broadcast return) and `signals/ebsd.py` (the one dimensional
    grids, the Memory note, the mask documentation).

65. **Full existing suite at the fix gate, and the same one failure.**

    ```
    .venv/Scripts/python.exe -m pytest tests -q -x --ignore=tests/test_data
    ->  1 failed, 4528 passed, 801 skipped, 5 rerun in 261.98 s
    ```

    The failure is
    `tests/test_simulations/test_kikuchi_pattern_simulator.py
    ::TestCalculateMasterPattern::test_shape` again -- the upstream
    flaky test entries 4, 27 and 49 logged on this machine.
    `git diff HEAD -- src/kikuchipy/simulations tests/test_simulations`
    is still EMPTY on this branch, and RUN IN ISOLATION three times at
    this gate it passed each time, two of the three only after its own
    `@pytest.mark.flaky(reruns=5)` reruns (4 reruns, 0, 1). Deselecting
    it, the suite is GREEN:

    ```
    .venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_data \
      --deselect "tests/test_simulations/test_kikuchi_pattern_simulator.py
                  ::TestCalculateMasterPattern::test_shape"
    ->  4550 passed, 803 skipped, 1 deselected in 252.84 s
    ```

    The 37 tests gained over entry 49's 4513 are exactly the 37 this
    gate added (474 against 437 in the eleven-module list). `pytest`
    was run WITH `pytest-randomly` in its default random order for
    both, and the eleven-module list is green in random order as well
    as under `-p no:randomly`.

    `pre-commit` itself is not installed in this environment; its two
    code hooks are `ruff` and `ruff-format`, both run above, and the
    other two are `black-jupyter` (`.ipynb` only, none touched) and
    `licenseheaders` (every edited file keeps its GPL header
    unchanged).

### 2026-09-08 (Stage C implementation gate, measurement agent)

**Machine A** again, the same 20-core Windows 11 laptop every number
above carries: Intel64 Family 6 Model 186 (Raptor Lake), 20 logical
cores, Windows 11 build 26200, `.venv` Python 3.13.12, numpy 2.4.6,
scipy 1.17.1, numba 0.65.1, scikit-image 0.26.0, orix 0.14.2, dask
2026.3.0. Warm numba caches. The Si wafer was already in the pooch
cache from entry 40, so nothing was downloaded at this gate.

66. **The two Stage C MTP pins.** Recipe: the Stage A and Stage B
    convention, the modules' own measuring tests under the same
    throwaway plugin (`hrebsd_measure_c/measure_plugin.py`, which
    turns every `None` placeholder into `+inf` and makes
    `assert_within` record instead of assert, so one run reports every
    placeholder including those hidden behind an earlier failure):

    ```
    PYTHONPATH=<scratchpad>/hrebsd_measure_c .venv/Scripts/python.exe \
      -m pytest tests/test_indexing/test_hrebsd_gnd.py \
                tests/test_indexing/test_hrebsd_si.py -q -p measure_plugin
    -> 135 passed, 8 skipped (--weekly), 89.2 s   (-p no:randomly)
    -> 135 passed, 8 skipped (--weekly), 90.8 s   (default random order)
    ```

    | constant | measured | pinned | margin |
    |---|---|---|---|
    | `GND_E2E_TOL` | **8.3502e-04** (worst relative error, 9 points x 3 estimators, every point finite) | 1.7e-03 | 2.04x |
    | `SI_GND_FLOOR` | **1.1237e11** m^-2 (smoke sub-grid median, 66 of 100 points finite) | 2.3e11 | 2.05x |

    Both values are BITWISE identical between the two runs and between
    the two test orders. The same two runs re-measure the four Stage B
    pins this pair of modules also carries, and every one reproduces
    its recorded value exactly: `SI_STRAIN_FLOOR` 1.216823308728681e-02
    and `SI_ROTATION_FLOOR` 1.200629820334714e-02 and `SI_KAM_FLOOR`
    5.240242922561412 against entry 43, `SI_PC_RESIDUAL_MEAN_TOL`
    10.99558247515951 against entry 51. Nothing Stage C added to
    `_gnd.py` or moved in `_tensors.py` has disturbed the Stage B
    route.

    **`GND_E2E_TOL` in detail** (`hrebsd_measure_c/gnd_e2e_detail.py`,
    the same helpers the test uses, one estimator at a time). The
    analytic curvature field is imposed through deformed-master
    PATTERNS on a 3 by 3 map at a 1 um step, 5e-3 of distortion change
    per step, and recovered through projection, IC-GN, the D6
    conversion, the D7 frame, the D9 closure and the D14 gradients:

    | estimator | analytic rho | recovered range | worst relative | median relative |
    |---|---|---|---|---|
    | `a3` | 4.02625e13 m^-2 | 4.02289e13 to 4.02877e13 | **8.3502e-04** | 2.4584e-04 |
    | `a5` | 6.17677e13 m^-2 | 6.17674e13 to 6.18007e13 | 5.3347e-04 | 8.5849e-05 |
    | `a9` | 6.66447e13 m^-2 | 6.66264e13 to 6.66724e13 | 4.1620e-04 | 2.2818e-04 |

    All nine points of all three maps are finite, and the whole
    nine-pattern DIC run takes 3.68 s. The pin is `a3`, the estimator
    consuming the fewest alpha entries and therefore the one with the
    least averaging over the gradient noise.

    **0.084 per cent is where the placeholder's own note predicted.**
    That note bounds the number from below by the Stage A homography
    accuracy: a gradient reads a DIFFERENCE of two neighbouring `Fe`
    tensors, so the pinned `DEFORMED_MASTER_FE_TOL` of 3.1e-5 over the
    5e-3 per-step change is 6.2e-03, and the measurement comes in a
    factor of 7.4 UNDER that. The whole tensor and gradient chain
    therefore adds less error than the DIC it reads, which is the
    Stage C half of validation V7 discharged. It is a synthetic
    oracle and says nothing about real data; entry 67 is the real-data
    half and says something quite different.

67. **The Si-wafer GND floor, MEASURED AND PINNED -- and it is
    1.1237e11 m^-2, which is a factor of 36 to 71 BELOW the 4e12 to
    8e12 m^-2 literature class, and that is NOT a better floor.**
    Recipe: `TestGndFloor::test_si_gnd_floor` under the plugin of
    entry 66, with the supporting numbers from
    `hrebsd_measure_c/si_gnd_detail.py` on the same route -- entry
    43's route exactly: `reference="auto"`, per-point projection
    centres from `extrapolate_pc` at the corrected `px_size` of entry
    41, the deviatoric closure, RAW patterns, the `[::5, ::5]` smoke
    sub-grid of 100 patterns, `b = 3.84e-10` m and the default `"a5"`.

    | | smoke sub-grid, 200 um step | full 50x50 map, 40 um step |
    |---|---|---|
    | points | 100 | 2500 |
    | converged | 87 | 2187 |
    | **finite GND points** | **66 (66 %)** | **1969 (78.8 %)** |
    | median rho | **1.1237e11 m^-2** | **1.3717e11 m^-2** |
    | range of the finite points | 8.2333e10 to 1.5852e11 | 6.0706e10 to 3.8729e11 |

    The SURVIVING FINITE FRACTION is recorded first because the V7
    line and the placeholder's own note both demand it: the D14.5 rule
    refuses a stencil touching a non-converged point, so each failure
    takes its in-plane neighbours with it. Measured cost per failure,
    which is the thing that could not be predicted: 34 points lost to
    13 failures on the smoke grid (2.6 each) and 531 lost to 313 on
    the full map (1.70 each), against the 5 a lone failure would cost.
    The failures are heavily clustered, more so on the denser map.
    Map EDGES cost nothing: `_axis_derivative` takes a one-sided
    difference there rather than a NaN, which is why 66 survive on a
    grid with 36 edge points.

    **The literature relation, stated plainly.** Requirements D14.6
    quotes `rho_noise ~ sigma_beta / (b * step)` and about 4e12 to
    8e12 m^-2 at `sigma_beta = 1e-4`, `b = 0.25` nm and a 100 nm step.
    The measured 1.1237e11 m^-2 is 36 to 71 times SMALLER than that
    class, and reading that as this route reaching a lower floor would
    be exactly backwards. The literature number is quoted at a 100 nm
    step; this sub-grid's step is 200 um, TWO THOUSAND times longer,
    and a curvature is a distortion divided by a distance. Put this
    dataset's own numbers into the same identity -- its MEASURED
    rotation floor of 1.2006e-02 rad (entry 43) and its own step --
    and it predicts 1.2006e-02 / (3.84e-10 * 2e-4) = 1.5633e11 m^-2.
    The measurement is 0.72x of that. **So the floor is consistent
    with being noise and nothing else, and the honest reading is that
    it is small only because the divisor is large.** Nothing here
    contradicts the literature class and nothing here reaches it; the
    two are measurements of different quantities.

    **And a measurement of this gate's own says the noise is not even
    white.** The same identity predicts a FIVE-fold rise between the
    two columns above, the step falling from 200 um to 40 um. It
    rises 1.22x. The per-point noise scale is the same on both maps --
    entry 43 measured the smoke and full strain floors at 1.2168e-02
    and 1.2208e-02, agreeing to 0.3 per cent -- so the divisor is the
    only thing that changed, and a white field would have obeyed. It
    does not, so neighbouring points of the recovered distortion field
    are strongly correlated. That is precisely what entry 44's
    diagnosis predicts: the fits sit in a zero-shift minimum pinned by
    a band-pass-surviving component that does not move with the beam,
    so the recovered field varies far more slowly across the map than
    an independent-noise model assumes.

    **Consequence, and it is the same one entries 43 to 45 reached.**
    This is THIS DATASET's GND floor, not the method's, and it must
    never be quoted as "kikuchipy's HR-EBSD GND noise floor". The
    si_wafer scan was acquired for projection-centre calibration; the
    module docstring carries the full paragraph, requirements D14.6
    carries the identity, and plan open question 13's Si-indent
    dataset, deferred past Stage C, is where a method-level GND floor
    can be measured. **CORRECTED 2026-09-08 (Stage C fix gate): at
    the time this entry was written the sentence above was FALSE.
    The module carrying the paragraph was
    `tests/test_indexing/test_hrebsd_si.py`, which no user reads, and
    neither the `_gnd.py` module docstring nor the public
    `hrebsd_gnd` Notes carried the measured floor at all -- they
    quoted only the 4e12-8e12 literature class, which is the one
    number this entry says a reader must never be handed alone. Both
    now carry it, and requirements D14.6 records the discharge (entry
    75).** The pin at 2.3e11 is a regression guard on this
    dataset's behaviour and is nothing else. It also clears the
    full-map median by 1.68x, so it would survive a full-map arm
    being added beside the strain one.

    Two supporting results from the same run. The three D14.4
    estimators on the smoke map give `a3` 1.3840e11, `a5` 1.1237e11
    and `a9` 4.6404e11 m^-2 -- all three different, which is all the
    weekly `test_gnd_estimator_sweep` asserts, and `a9` sits 4.1x
    above `a5` because it consumes the six d/dx3-neglect entries the
    smaller sets drop. And the D14.3 antisymmetry fix RAISES this
    floor, 1.1237e11 with it against 7.7823e10 without: the direction
    the V7 line already recorded at the failing-tests gate, confirmed
    here at the implementation gate on the delivered code, and the
    opposite of the direction requirements D14.3's 9.6x argument
    describes. On a dataset whose floor is set by the band-pass
    artefact rather than by beta31/32 noise, that argument does not
    apply; the fix stays the default on the frozen theory, not on this
    measurement, and this is recorded rather than used to re-decide.

68. **The Stage C performance row (recorded, never a gate, D16).**
    Recipe: `hrebsd_measure_c/si_gnd_detail.py`, the full 50 by 50 Si
    map, `hrebsd_gnd` timed as a median of 5 on the already-computed
    map, with the DIC that feeds it timed alongside.

    | leg | 100-point smoke grid | full 2500-point map |
    |---|---|---|
    | `hrebsd_gnd`, median of 5 | **1.594 ms** (min 1.588) | **10.78 ms** (min 10.69) |
    | the DIC that feeds it | -- | 509.5 s, 4.91 pat/s |

    `hrebsd_gnd` is the most expensive of the four derived maps at
    this scale -- entry 46 measured `hrebsd_strain_stress` at 6.13 ms,
    `segment_grains` at 5.59 ms and `hrebsd_kam` at 2.36 ms on the
    same 2500 points -- and it is still 2.1e-05 of the DIC run it
    reads. Twenty-five times the points cost 6.8x the time, so fixed
    overheads still dominate at 100 points.

    **The timing environment differed from entry 46's and it is
    recorded rather than smoothed.** The DIC leg here took 509.5 s
    against entry 46's 372.5 s, 37 per cent slower, for a
    bitwise-identical outcome on the identical route (2187 of 2500
    converged both times). The machine was less idle at this gate, so
    the `hrebsd_gnd` figures above are if anything an over-estimate;
    they are quoted as a median of five for that reason and neither
    they nor entry 46's row gates anything.

69. **Gate outcome.** With the two pins of entries 66 and 67 in place,
    the eleven-module HREBSD list plus the signal-method suite is
    GREEN:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_gnd.py \
      tests/test_indexing/test_hrebsd_si.py \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_stiffness.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_interpolation.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py -q -n 4
    -> 609 passed, 9 skipped, 107.3 s
    ```

    The 9 skips are all `--weekly` gated; the `[download]` gate of
    validation V5 does NOT skip here, the wafer having been cached at
    entry 40, so the four Si arms including `test_si_gnd_floor` really
    ran. 618 collected against entry 65's 474: the Stage C
    failing-tests commit (`c11ebe6b`) added 88 net new test functions
    across the only three test files it touched --
    `test_hrebsd_gnd.py` (new, 131 collected), `test_hrebsd_si.py` and
    `test_hrebsd_tensors.py` -- which parametrize out to the 144
    collected items gained.

    NO unfilled `MTP` placeholder remains in the HREBSD suite: a
    module-level scan of all thirteen files for an UPPERCASE name
    bound to `None` returns nothing, so the Definition-of-done item
    "every `MTP` placeholder replaced by a dated measured value" is
    discharged for Stages A, B and C together.

    Ruff, the only two `pre-commit` hooks that touch code here, on the
    two edited test files and on every `_hrebsd/` module:
    `ruff check` -> "All checks passed!", `ruff format --check` ->
    "15 files already formatted". (`pre-commit` itself is still not
    installed in this environment; its other two hooks are
    `black-jupyter`, no notebook touched, and `licenseheaders`, every
    edited file keeping its GPL header unchanged.)

70. **Full existing suite, and the upstream flaky test passed this
    time.**

    ```
    .venv/Scripts/python.exe -m pytest tests -q -x --ignore=tests/test_data
    -> 4683 passed, 805 skipped, 5 rerun in 487.71 s
    ```

    NO failures, so `-x` did not stop the run and this tally is a
    complete one -- unlike entries 49 and 65, whose totals are
    truncated at the failure that stopped them. The 5 reruns are
    `tests/test_simulations/test_kikuchi_pattern_simulator.py
    ::TestCalculateMasterPattern::test_shape` taking its own
    `@pytest.mark.flaky(reruns=5)` reruns and then PASSING: the same
    upstream flake entries 4, 27, 49 and 65 logged on this machine,
    passing here rather than exhausting its reruns. Nothing on this
    branch touches it (`git diff HEAD -- src/kikuchipy/simulations
    tests/test_simulations` is still empty).

    Nothing outside HREBSD is broken by this gate, and nothing outside
    HREBSD was touched: the only source changed at this gate is
    `_gnd.py` and `_tensors.py` (the Stage C implementation), and the
    only tests changed are the two `MTP` constants of entries 66 and
    67 with their dated comments.

### Stage C adversarial review, fix gate (2026-09-08)

**Machine A**, the same 20-core Windows 11 laptop and the same `.venv`
as entry 66. Seventeen findings from the two reviewers plus one
surviving mutant. Every finding was verified before it was acted on
and every one of the seventeen is APPLIED; two of their SUGGESTED
FIXES were declined in favour of another remedy, with the reason
recorded in entry 75. The mutant is killed and the kill verified by
re-injection (entry 73). No frozen assertion was touched and no
measured pin moved.

71. **Coverage, 100 per cent of the Stage C `_hrebsd/` modules, with
    the command output recorded.** The Definition-of-done item the
    Stage C gate had left undischarged (the review found it: entries
    66 to 70 record the two MTP pins, the GND floor, the performance
    row, ruff and the full suite, and no coverage command). Run the
    entry-23/57 way, with `pytest-cov` installed into a scratchpad
    directory put on `PYTHONPATH` rather than into the project
    environment:

    ```
    PYTHONPATH=<scratchpad>/covpkgs .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_gnd.py \
      tests/test_indexing/test_hrebsd_si.py \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_stiffness.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_interpolation.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py \
      --cov=src/kikuchipy/indexing/_hrebsd --cov-report=term-missing -q
    ->
      __init__.py            0 stmts, 0 miss, 100.00%
      _engine.py           270 stmts, 0 miss, 100.00%
      _geometry.py          55 stmts, 0 miss, 100.00%
      _gnd.py              119 stmts, 0 miss, 100.00%
      _homography.py        62 stmts, 0 miss, 100.00%
      _interpolation.py    186 stmts, 0 miss, 100.00%
      _kam.py               77 stmts, 0 miss, 100.00%
      _pc_shift.py          57 stmts, 0 miss, 100.00%
      _preprocessing.py     70 stmts, 0 miss, 100.00%
      _reference.py         74 stmts, 0 miss, 100.00%
      _segmentation.py     154 stmts, 0 miss, 100.00%
      _stiffness.py         82 stmts, 0 miss, 100.00%
      _tensors.py          189 stmts, 0 miss, 100.00%
      TOTAL               1395 stmts, 0 miss, 100.00%
    -> 610 passed, 9 skipped, 101.73 s
    ```

    All thirteen modules at 100.00 per cent with no line missed, so
    unlike the Stage B measurement of entry 57 this one closed no
    gaps: there were none. 610 rather than entry 69's 609 because of
    the one test added at this gate (entry 73).

72. **The local oldest-matrix run, recorded -- and two corrections to
    the recipe.** The other undischarged Definition-of-done item.
    Plan section 1's recipe, run against a wheel built FRESH into the
    scratchpad rather than letting `uv` resolve the project:

    ```
    uv build --wheel --out-dir <scratchpad>
    uv run --isolated --python 3.10 \
      --with <scratchpad>/kikuchipy-0.14.dev0-py3-none-any.whl \
      --with "numpy==1.23.0" --with "numba==0.57" \
      --with "orix==0.12.1" --with "scikit-image==0.21.0" \
      --with pytest --with pytest-benchmark --with pytest-rerunfailures \
      --with pytest-xdist \
      pytest tests/test_indexing tests/test_signals -k hrebsd -q
    -> 610 passed, 9 skipped, 4335 deselected, 85.26 s
    ```

    Resolved: Python **3.10.19**, numpy **1.23.0**, numba **0.57.0**,
    orix **0.12.1**, scikit-image **0.21.0**, scipy 1.13.1, dask
    2024.8.1, kikuchipy 0.14.dev0 (the wheel above). The 9 skips are
    the `--weekly` gates; the Si arms skip on `[download]` here
    because the isolated environment has no pooch cache, which is why
    this run is the API-floor gate and entry 71 is the coverage one.

    **Correction 1, to the recipe as plan section 1 writes it: it
    does not run.** `pyproject.toml:165-171` puts `--benchmark-skip`
    in `addopts`, so pytest in the isolated environment aborts at
    collection with "unrecognized arguments: --benchmark-skip" unless
    `--with pytest-benchmark` is added. Recorded here rather than
    rewritten into the plan, which is frozen.

    **Correction 2, the one the review measured: `uv` can serve a
    STALE cached wheel** to the literal recipe, so a run can silently
    test yesterday's code. Building the wheel first and passing it by
    path, as above, is the remedy; `--no-cache` is the other. Every
    future stage gate should use one of the two.

    **The orix 0.12.1 floor holds for the Stage C API too.**
    `CrystalMap.dx`/`dy`, which `_gnd.py:217-218` uses and nothing
    else in the package does, exist in orix 0.12.1
    (`crystal_map.py:351-358`), so no version gate is needed and
    tech-stack.md:17's rule is satisfied without one.

73. **The surviving mutant, killed and the kill verified.** The review
    left one mutant alive: in `in_plane_gradients` the trailing
    reduction `finite = finite.all(axis=-1)` becomes
    `finite.any(axis=-1)`, so a map point counts as usable when ANY
    entry of its 3 by 3 block is finite rather than every one. That
    attacks the docstring rule "Both rules are read PER MAP POINT and
    not per trailing component", which is requirements D2.6 read the
    conservative way.

    Reproduced first. With the mutant injected,
    `tests/test_indexing/test_hrebsd_gnd.py` gave **131 passed**, and
    the module's own NaN arm could not see it: `TestNaNSafety
    ::test_a_non_finite_point_is_nan_in_alpha_itself` holes the WHOLE
    3 by 3 block (`field[1, 1] = np.nan`), which `any` refuses too.
    Measured on the same field with ONE entry holed,
    `field[1, 1, 0, 2] = np.nan`: `nye_tensor` returns **9 of 9 NaN**
    at that point on the delivered code and **0 of 9** under the
    mutant, the finite block being the full constant-curvature answer
    of the neighbourhood. So a point whose measurement is partly
    missing would be handed its neighbourhood's dislocation density.

    It is invisible through `hrebsd_gnd`, which is why the arm goes at
    the `nye_tensor` level: `tensor_chain`'s congruence `M^T A M`
    spreads one NaN over the whole beta block before a gradient sees
    it, so the public density is NaN either way (verified both ways).

    Fixed by ADDING one test, `TestNaNSafety
    ::test_one_non_finite_entry_makes_the_whole_point_nan`, and by
    naming the mutant in the module's mutation map. No frozen
    assertion was touched and no source line changed.
    RE-INJECTED: **1 failed, 131 passed**, the failure being exactly
    the new arm. RESTORED: **132 passed**.

74. **What the D14.3 antisymmetry fix actually does, measured.** The
    review found the fix described as a noise reduction and nothing
    else, in the tutorial and in the public docstring, while it also
    forces two measured quantities to zero. Recipe:
    `<scratchpad>/antisym_check.py`, the tutorial's own analytically
    imposed field fed through the PUBLIC `hrebsd_gnd` with no DIC at
    all, so every number here is the field's own.

    | quantity | without the fix | with the fix |
    |---|---|---|
    | detector-frame max abs `eps13` | 3.504843e-04 | **0.0 exactly** |
    | detector-frame max abs `eps23` | 5.511282e-04 | **0.0 exactly** |
    | rms `eps13` over 2000 random tensors | 7.024588e-04 | **0.0 exactly** |
    | `"a3"` median rho | 2.0766e12 | 2.2064e12 (+6.3 %) |
    | `"a5"` median rho | 2.2203e12 | 2.3972e12 (**+8.0 %**) |
    | `"a9"` median rho | 2.5109e12 | 2.6448e12 (+5.3 %) |

    The no-fix column reproduces the imposed field's OWN Nye content
    to 0.05 per cent (`nye_tensor` on the imposed sample-frame beta
    gives 2.2192e12 for `"a5"`), which is the check that the analytic
    route is right. `max abs(beta_sample after the fix - imposed
    beta)` is **9.7345e-04** against the field's own 1.4000e-03
    amplitude, so the fix perturbs 70 per cent of the field.

    The algebra behind the zeros is one line: `beta31 <- -beta13`
    makes `eps13 = (beta13 + beta31)/2` identically zero, and
    likewise `eps23`. The fix is therefore a TRADE of a measured
    out-of-plane elastic shear for a quieter noise operator, not a
    filter.

    The tutorial's DIC-recovered `"a5"` median is 2.41e12, which is
    the FIXED value and not the imposed one: the chain recovers the
    field it was given to well under one per cent, and the 8 per cent
    is the fix. That comparison is now printed by the tutorial
    itself, so the notebook's opening promise that every recovered
    number can be checked against an answer known by construction
    holds for the GND too.

    Recorded in requirements D14.3 as a dated amendment, in the
    `_gnd.py` module docstring, in `enforce_beta_antisymmetry`'s
    Notes, in the `enforce_antisymmetry` parameter description
    (together with entry 67's opposite-direction Si result) and in
    the tutorial. **The default does not change**: it rests on the
    frozen geometric-noise argument, which neither measurement tests.

75. **The documentation findings, applied.** The twelve remaining
    findings of the two reviewers, each verified against the source or
    the ledger before it was applied:

    - **`sigma33` is stored.** `tensor_chain` writes
      `properties["stress"]` at `STRESS_PROP_SIZE = VOIGT_SIZE = 6`
      in Voigt order, so index 2 IS `sigma33`, and the tutorial cell
      that says it is not stored reads it out of the stored property
      two lines above. Reworded: it is zero by construction of the
      closure rather than by measurement, so it is stored and carries
      no information.
    - **"four orders" was 2.88.** The cell's own printed outputs give
      1.97e-01 / 2.62e-04 = **751.91**, log10 = 2.8762. Now "nearly
      three orders ... a factor of 752", with the factor printed
      beside it so the sentence cannot drift from the number again.
    - **The orientations enter in TWO places.** The tutorial claimed
      they enter "in one place only, the rotation of the elastic
      stiffness". `segment_grains` labels grains from the neighbour
      misorientation of `xmap.rotations`
      (`_segmentation.py:228-245`), and `reference="auto"` picks one
      reference per label, so on the DEFAULT path they control every
      number in the map. The synthetic section only escapes it by
      passing `reference=(0, 0)`. Both that cell and the Si section's
      cell now give the correct reason, the Si one being that a
      single crystal is one grain under ANY constant orientation
      field.
    - **The log10 guard.** The `hrebsd_gnd` docstring said the guarded
      form "is what this feature's tutorial does" and the tutorial
      called a bare `np.log10(gnd)`. The tutorial now uses
      `np.log10(np.where(gnd > 0, gnd, np.nan))`, so the sentence is
      true. `log10_density` stays: it is the D14.6 recipe as
      executable, TESTED code and the killer of the plan 4.3 "log10
      of signed values unguarded" mutant, and a private module cannot
      be imported by a tutorial. Its own docstring claim, that it
      exists so the tutorial need not repeat the recipe, was the
      false half and is corrected.
    - **The Si GND floor reaches the API reference.** Requirements
      D14.6 asks the docstring to quote the measured floor once
      recorded; entry 67 recorded it and asserted the documentation
      existed, and it did not. The `_gnd.py` module docstring and the
      `hrebsd_gnd` Notes now carry 1.1237e11 m^-2, the 200 um step
      that makes it 36 to 71 times below the literature class rather
      than better than it, the 1.5633e11 m^-2 the same identity
      predicts from this data set's own rotation floor, and the
      statement that it is that data set's floor and never the
      method's. Entry 67's own sentence is corrected in place.
    - **"every pair tested" was seven of eight.** Entry 44 records
      `phase_cross_correlation` returning exactly 0.0 px for seven
      listed pairs and -0.0625 px for `(49, 0)`. The tutorial dropped
      the eighth; it now says "0.0 pixel for seven of the eight pairs
      tested and -0.06 pixel for the eighth".
    - **The Si section carries entry 67's antisymmetry result.** The
      fix RAISES that data set's floor by 44 per cent, 1.1237e11
      against 7.7823e10, and the tutorial now says so beside the
      other honesty bullets, with the reason (that map's floor is set
      by the band-pass-surviving fixed-pattern component, not by
      `beta31`/`beta32` noise) and the consequence (the default rests
      on the geometry, not on this measurement).
    - **Bibliography.** Two of D18's nine keys were missing:
      `ernould2022advances` (AIEP 223 (2022) Ch. 2, "Development of a
      homography-based global DIC approach for high-angular
      resolution in the SEM" -- the source SEVEN `_hrebsd/` module
      docstrings name as the primary convention reference) and
      `hardin2015analysis` (J. Microsc. 260 (2015) 73-85, the
      traction-free assumption). Both added, in sorted position.
      `:cite:` roles were absent from every HREBSD object, so none
      linked to the bibliography from the rendered API reference;
      they are now in `EBSD.hrebsd_dic`'s Notes, in
      `hrebsd_strain_stress`'s `closure` parameter and in
      `hrebsd_gnd`'s `estimator` and `enforce_antisymmetry`
      parameters and Notes. The `_hrebsd/` module docstrings keep
      prose citations: those modules are private and their docstrings
      are never rendered, so a role there would link nothing
      (recorded in D18).
    - **`EBSD.hrebsd_dic`'s See Also** named two of the five
      consumers of its result. `hrebsd_gnd`, `hrebsd_kam` and
      `hrebsd_pc_shift` added, so the reference walks both ways.
    - **The ATEX entry in `related_projects.rst`** said
      `hrebsd_dic` is "derived from" ATEX's published equations,
      which reads as a derivation from a third-party product. The
      spec is emphatic that the derivation is from the PAPERS
      (Scope, D18, tech-stack.md:131-137), and the OpenXY bullet two
      lines above already had the right wording. Reworded to match
      it: the reference implementation of the same published
      approach, used as an equation level cross reference, no code
      ported.
    - **The two missing pieces of tutorial content.** D4's preamble
      commits to demonstrating the upstream
      `remove_static_background`/`remove_dynamic_background` and the
      notebook called neither (grep count 0); a cell now shows both
      on the wafer, states that the engine's own band-pass is a
      separate internal step, and says the run below deliberately
      uses the RAW patterns, which the honesty section then explains.
      The Manual checklist's "misorientation maps, IPF, IQ" line is
      dispositioned above: IQ is demonstrated and is the one that is
      load-bearing here (it is the score `reference="auto"` ranks
      candidates by, and the tutorial shows its maximum IS the point
      `"reference_index"` names -- both printed 0), while IPF and a
      Hough misorientation map are not added, because both data sets
      are single-orientation by construction and neither map would
      carry information.

    Two suggestions were REJECTED in part, with reasons. (a) The
    review suggested narrowing `nbval-ignore-output` on the Si results
    cell so that "converged: 87 of 100" and "over 66 finite points"
    stay checked. They are kept ignored: those two numbers are the
    output of a 100-pattern DIC on real data through
    `phase_cross_correlation`, and the weekly job installs the latest
    scikit-image and scipy on every run, so they are the LEAST
    drift-safe numbers in the notebook rather than the most. What
    nbval does and does not check after this gate is recorded in
    entry 76 instead. (b) The review suggested exporting
    `log10_density` or deleting it; neither was done, for the reason
    given above.

76. **The sanitize rule, narrowed -- and what nbval now actually
    regression-checks.** The review found `regex10` anchored on
    `": "` alone, so it blanked EVERY scientific-notation number
    printed after a colon in every notebook nbval runs. Measured
    before the change: it matched **15 outputs, all in
    `hrebsd_dic.ipynb`** and none elsewhere in the nine notebooks of
    `run_nbval.sh`, so there was no collateral -- but the 15 included
    every headline number, the two ANALYTIC ones among them, so the
    notebook's quantitative content was essentially unchecked.

    `regex10` is now an explicit alternation of the labels of the
    fit-derived quantities (`median residual`, `worst strain
    component error`, `worst rotation component error`, `largest
    |sigma_33|`, `stress amplitude`, the two closure differences,
    `recovered from the patterns`, `a3`/`a5`/`a9`), and three narrow
    rules were added for the fixed-point and discrete quantities the
    review measured as fragile: `regex11` for `largest iteration
    count` (three of the 49 fits exit within 3 to 13 per cent of the
    1e-3 px `min_step`, so a scikit-image seed change can take one to
    a fourth iteration), `regex12` for `median HR-KAM` and `regex13`
    for the `N.NNN to N.NNN px` ranges of the projection-centre shift
    cell, whose three decimals ARE the `min_step` scale.

    Verified by two controls rather than by inspection, each a full
    nbval run of the notebook:

    | control | expectation | result |
    |---|---|---|
    | perturb a SANITIZED value, `a5: 2.41e+12` -> `9.99e+12` | pass | **25 passed** |
    | perturb a CHECKED value, `largest imposed distortion: 1.40e-03` -> `1.50e-03` | fail | **1 failed, 24 passed**, on that cell |

    So the two analytic quantities are now genuinely regression
    checked, which the old rule blanked. Collateral re-verified after
    the change: the four rules together match **zero** outputs in the
    other eight notebooks.

    nbval on the tutorial itself: **25 passed in 29.30 s**
    (`uv run --with nbval --with ipykernel pytest -q --nbval
    doc/tutorials/hrebsd_dic.ipynb --nbval-sanitize-with
    doc/tutorials/tutorials_sanitize.cfg`), 25 rather than the gate's
    23 because of the two cells added at entry 75.

    **What nbval checks after this gate, stated so it is not assumed:**
    the two analytic distortion and strain amplitudes, the two
    analytic `imposed field` GND medians added at entry 74, the
    imposed-vs-recovered `Fe - I` block at one decimal of 1e-3, the
    analytic projection-centre drifts of both maps, `converged: 49 of
    49`, both property lists, the stiffness matrix, the reference
    index versus the image-quality maximum, and every `print` of a
    signal, detector or crystal map. It does NOT check the
    fit-derived scalars listed above, nor anything in the two
    `nbval-ignore-output` cells of the Si section.

    **A pre-existing, unrelated failure was found by the full sweep
    and is recorded rather than absorbed.** Running all nine
    notebooks of `run_nbval.sh` on this machine, two cells of
    `doc/tutorials/hybrid_indexing.ipynb` fail: its 15th code cell,
    which prints the mean and standard deviation of a PyEBSDIndex
    Hough projection-centre optimisation at eight decimals
    (`[0.41798854 0.22099907 0.50496854]`), and its 20th, an NLopt
    refinement progress line. Neither can be an effect of this gate:
    nbval sanitizes the STORED and the PRODUCED output with the same
    patterns, so removing a pattern can only turn a pass into a
    failure where that pattern was masking a real difference, and the
    removed `regex10` matched none of that notebook's stored outputs
    (measured) -- a bare `[0.41798854 ...]` has no `": "` before it
    at all. Nothing on this branch touches that notebook. It is
    library drift on this machine, logged for whoever next runs the
    weekly job.

77. **Fix-gate outcome.** All thirteen HREBSD test modules plus the
    signal-method suite, on the delivered code with the one test of
    entry 73 added:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_indexing/test_hrebsd_gnd.py \
      tests/test_indexing/test_hrebsd_si.py \
      tests/test_indexing/test_hrebsd_tensors.py \
      tests/test_indexing/test_hrebsd_stiffness.py \
      tests/test_indexing/test_hrebsd_segmentation.py \
      tests/test_indexing/test_hrebsd_kam.py \
      tests/test_indexing/test_hrebsd_pc_shift.py \
      tests/test_indexing/test_hrebsd_deformed_master.py \
      tests/test_indexing/test_hrebsd_engine.py \
      tests/test_indexing/test_hrebsd_geometry.py \
      tests/test_indexing/test_hrebsd_homography.py \
      tests/test_indexing/test_hrebsd_interpolation.py \
      tests/test_signals/test_ebsd_hrebsd_dic.py -q -n 4
    -> 610 passed, 9 skipped, 90.95 s
    ```

    The 9 skips are the `--weekly` gates; the `[download]` gate does
    not skip, so the four Si arms including `test_si_gnd_floor` really
    ran and `SI_GND_FLOOR` still holds after every change at this
    gate. No pin moved: nothing here touched a measured constant.

    Ruff, the only two `pre-commit` hooks that touch code here, on
    every `_hrebsd/` module, on `signals/ebsd.py` and on the two
    edited test files: `ruff check` -> "All checks passed!",
    `ruff format --check` -> "16 files already formatted".
    `black-jupyter --line-length=77` on the re-executed notebook ->
    "1 file would be left unchanged". (`pre-commit` itself is still
    not installed in this environment; `licenseheaders` is the fourth
    hook and every edited file keeps its GPL header unchanged.)

    Full existing suite, twice, and the SAME upstream flake as
    entries 4, 27, 49 and 65:

    ```
    .venv/Scripts/python.exe -m pytest tests -q -x --ignore=tests/test_data
    -> 1 failed, 4661 passed, 803 skipped, 5 rerun in 291.22 s
    ```

    The one failure is
    `tests/test_simulations/test_kikuchi_pattern_simulator.py
    ::TestCalculateMasterPattern::test_shape` exhausting its own
    `@pytest.mark.flaky(reruns=5)` reruns, which is the flake this
    machine has logged at four earlier gates and which entry 70 saw
    PASS on its reruns. Run alone immediately afterwards it passes,
    on its third rerun: `1 passed, 3 rerun in 0.73 s`. Nothing on this
    branch touches it -- `git diff HEAD -- src/kikuchipy/simulations
    tests/test_simulations` is empty -- and because `-x` stopped the
    run at 97 per cent, that tally is truncated. The complete run,
    without `-x`:

    ```
    .venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_data
    -> 1 failed, 4683 passed, 805 skipped, 5 rerun in 281.16 s
    ```

    The same one failure and nothing else. **4683 passed and 805
    skipped are entry 70's numbers exactly**, and the single extra
    item is the test added at entry 73: entry 70 counted the flake
    among its 4683 because it passed on its reruns that time, so
    4683 + 1 failed here is 4684 non-skipped against entry 70's
    4683, which is the one new test and nothing more. Nothing
    outside HREBSD is broken by this gate.
    

### Si-indent application (2026-09-08 ->; plan open question 13, in execution)

Entries from 78. Machine A unless stated (20-core Raptor Lake laptop,
Windows 11 build 26200, RTX 2000 Ada; the venv of tech-stack section 1).
Scope per the dated OQ13 note in plan.md: full-resolution replication of
Winkelmann et al. 2025 (Ultramicroscopy 276, 114180) on Zenodo 14059950,
deviatoric closure only. Data file local and gitignored, never committed:
AGH__Si_indent_1_512x672.h5oina, 18,882,865,658 bytes, patterns
(58500, 512, 622) uint8. Entries below are append-only, dated, with
recipes, per the ledger discipline above.

78. **The rectangular end-to-end regression test (2026-09-08, plan
    OQ13 step 2(a)).** The Si-indent patterns are 512 rows by 622
    columns, and until this entry NOTHING non-square had been through
    `EBSD.hrebsd_dic` or the four analysis functions after it: every
    end-to-end call of the signal suite runs on the 60 by 60 shipped
    nickel patterns, where every `nrows` versus `ncols` mutant of
    requirements D1.2, D4.1 and D4.4 is the IDENTITY. The rectangular
    pins that did exist were unit level only, the `(37, 61)` arms of
    `test_hrebsd_engine.py`, `test_hrebsd_interpolation.py` and
    `test_hrebsd_geometry.py`. `TestRectangularEndToEnd` in
    `tests/test_signals/test_ebsd_hrebsd_dic.py` closes that gap
    BEFORE the 4 to 6 hour full-map run, which is the point of doing
    it first.

    **The fixture** (all of it deterministic, no random draw): a
    2 by 3 navigation map of 61 by 101 patterns, each one a copy of
    an off-centre crop of the shipped 401 by 401 stereographic nickel
    master (rows 150, columns 120) warped by its own known homography
    with `skimage.transform.ProjectiveTransform` plus `warp`, the
    test-oracle-only warper of requirements D18. 61 by 101 rather
    than the `(37, 61)` of the unit arms because the frozen
    `border=0.05` then leaves 3 rows and 5 columns of margin, enough
    that no subregion sample of the imposed warps reaches the pattern
    edge and the fit measures the engine and not the boundary policy.
    The crop is off centre because the master is four-fold symmetric
    about its centre, and a centred crop would carry a texture nearly
    invariant under the very row/column swap this class must be able
    to see. The detector carries ONE PROJECTION CENTRE PER MAP POINT,
    `pc.shape == (2, 3, 3)`, offsets of the 1e-3-fraction class the
    Si-indent file's own affine PC calibration has, so the D6.2
    beam-scan phantom is non-trivial; the reference is the explicit
    `(0, 1)`, deliberately not `(0, 0)`, since the beam-scan anchor of
    `_geometry.per_point_pc_pixels` IS the first grain reference and
    an anchor at the map origin would be indistinguishable from an
    ignored one.

    **What it pins**, the eight tests:

    - the fixture really is non-square, and a detector whose shape is
      the pattern's TRANSPOSE is refused (`ValueError`, "shape");
    - the full D15.6 Stage A prop set with the right shapes and data
      types on a rectangular map, `homography` `(6, 8)`, `Fe`
      `(6, 9)`, the D15.7 reshape route, and the input orientations
      untouched;
    - every point converges, every property is finite, the reference
      correlates with itself (corner norm 4.2e-09 px), each point's
      fit is closer to ITS OWN imposed warp than to any other point's
      (worst own 0.0440 px against best other 0.9593 px, a factor of
      22, which is what makes a permuted result visible with no
      tolerance);
    - the ASYMMETRIC-warp pin, below;
    - `Fe` against a D6/D6.2 conversion assembled in the test file;
    - lazy input equals eager BITWISE on all eight Stage A
      properties, the path the 18.9 GB run takes;
    - the whole chain runs: `hrebsd_strain_stress` (deviatoric, no
      stiffness, so `stress` is all NaN as designed), `hrebsd_kam`,
      `hrebsd_gnd` (b = 3.84e-10 m) and `hrebsd_pc_shift`, each
      returning the `(2, 3)` navigation shape with finite values, and
      the pc-shift measured translations equal to the raw homography's
      `h13`/`h23` read back through a non-square map grid.

    **The asymmetric-warp pin and HOW a transpose is numerically
    visible**, stated because a pin this tight is only worth what its
    mechanism is. The fit lives in the reference-PC-centred frame of
    D1.3, whose origin is `(pcx * ncols, pcy * nrows)`. On a square
    detector those are the same number; here `ncols - nrows` is 40, so
    the swapped origin sits `d = (+16.92, -23.12)` binned pixels from
    the true one. A homography fitted about a different origin is the
    conjugate `T(d) . W . T(-d)`, which leaves the linear block `A`
    alone and moves the translation by `-(A - I) d`. A PURE
    TRANSLATION is therefore origin invariant and could not see the
    swap at all, which is why the warp used carries an ANISOTROPIC
    linear block (`h11 = +1.5e-2` against `h22 = -1.0e-2`, off-diagonal
    pair antisymmetric) so that `(A - I) d` is large along both axes.
    MEASURED 2026-09-08: the swap moves `h13` by 0.5312 px and `h23`
    by 0.4342 px, while the fit recovers them to 0.0025 px and
    0.00055 px, a separation of 211x and 792x. The same comparison in
    the D2.5 corner norm is 0.0203 px against 0.6900 px.

    The `Fe` half is a SECOND, independent transpose site: `Fe`
    divides the corrected translations by `DD_px = pcz * nrows` and
    multiplies the perspective pair by it, and the phantom removed
    first carries `gamma = PC_target - PC_reference` in the same mixed
    units. A swap rescales `Fe13` and `Fe23` by `ncols / nrows`, 1.66
    here and exactly one on a square detector. MEASURED: the recovered
    `Fe` sits 2.1214e-04 from the right-axes expectation and
    1.0272e-02 from the swapped one, a factor of 48.

    **The two measured-then-pinned bands** (requirements D19; both
    PINNED at about 2x the measurement, the convention of
    `test_hrebsd_engine.py`):

    - `RECT_WARP_TOL = 0.09`, MEASURED worst 0.043956 px over the six
      points. That is the same regime as the engine module's 60 px
      arm (0.06507 px measured, `WARP_REFIT_TOL_60 = 0.13`) and about
      four times its 480 px arm, which is what a pattern this small
      should give; no precision CLAIM is attached to it.
    - `RECT_FE_TOL = 1.5e-3`, MEASURED worst 7.1949e-04.

    **Mutation check, run and recorded rather than assumed** (source
    restored afterwards; `git status` clean under `src/`):

    - `_geometry.pc_to_pixels` with `PCx`/`PCy` swapped
      (`pc[:, 0] * nrows`, `pc[:, 1] * ncols`) -> **3 of the 8 tests
      fail**: the per-point recovery, the asymmetric pin and the `Fe`
      pin.
    - the same function with `DD_px = pcz * ncols` -> the `Fe` pin
      fails (0.005844 against the 0.0015 band); the other seven pass,
      which is right, since the raw homography does not read `DD`.
    - `_preprocessing.band_pass_transfer_function` with
      `width = shape[0]` instead of `shape[1]` -> **SURVIVES all
      eight**, recorded as a limit of this class rather than hidden.
      It is the expected outcome: the same transfer function is
      applied to the reference and to every target, so a wrong cut-off
      changes the filtered content both sides identically and barely
      moves the fit. That mutant is killed at unit level by the
      `(37, 61)` arms of `TestPreprocessing` in
      `test_hrebsd_engine.py`, which compare against
      `kp.filters.highpass_fft_filter(shape, cutoff=0.05 * shape[1])`
      literally.

    **Outcome: the test PASSES against the delivered implementation.**
    It is a regression test and it found no engine bug; nothing in
    `src/` was changed for it. Tally:

    ```
    .venv/Scripts/python.exe -m pytest \
      tests/test_signals/test_ebsd_hrebsd_dic.py -q -k Rectangular
    -> 8 passed, 48 deselected in 0.76 s
    ```

79. **The Oxford H5OINA binning-read fix (2026-09-08, plan OQ13 step
    2(b)).** `oxford_h5ebsd/_api.py::get_binning` chose exactly ONE
    camera-mode dataset name from the H5OINA format version, "Camera
    Mode" at 7.0 and above and "Camera Binning Mode" below, and
    returned `None` when the file carried the other one, at which
    point `get_detector` leaves `binning` at its default 1. The
    version boundary is not a reliable guide: **AZtec 3.2.0.0 writes
    format 7.0 files whose header holds "Camera Binning Mode"**, and
    the Si-indent file is one. MEASURED on it (read-only `h5py`,
    2026-09-08): `Format Version` is `b"7.0"`, the header's camera
    datasets are `Camera Binning Mode`, `Camera Exposure Time` and
    `Camera Gain`, with no `Camera Mode` at all, and the value is
    `b"Speed 1 (622x512 px)"`, which the unchanged regular expression
    and `1024 / 512` turn into a binning of **2**.

    THE FIX is minimal and keeps the existing parse untouched: build
    the list of both names with the version-appropriate one FIRST,
    take the first name present, and log the debug message naming
    both only when neither is. A file carrying both therefore still
    reads the version-appropriate one, so the fallback can never
    silently override a correct reading, which two of the new
    parametrized cases pin.

    **The real-file check, after the fix**:

    ```
    get_binning({"Camera Binning Mode": "Speed 1 (622x512 px)"}, "7.0")
    -> 2
    kp.load("AGH__Si_indent_1_512x672.h5oina", lazy=True).detector
    -> binning 2 (was 1), shape (512, 622), pc (234, 250, 3)
    ```

    **HONESTY NOTE: this is metadata correctness, not numerics, on
    the route the Si-indent tutorial takes.** `binning` is never
    consumed anywhere in the HREBSD-DIC chain on a per-point-PC
    detector. `_geometry.per_point_pc_pixels` (`_geometry.py:236-249`)
    returns `pc_to_pixels(detector.pc_flattened, detector.shape)` the
    moment `detector.navigation_size` equals the map size, and
    `px_size`, `binning` and the scan steps are never read on that
    branch; they enter only through
    `EBSDDetector.extrapolate_pc`'s beam-scan model, which is the
    SINGLE-PC branch. A grep of `src/kikuchipy/indexing/_hrebsd/` for
    `binning` returns docstring prose only. The Si-indent file carries
    one projection centre per map point, so **no number in the
    tutorial's chain changes because of this fix**: what changes is
    that the detector now describes itself truthfully (printed,
    recorded in the notebook, and correct for any later use that does
    read `binning`, such as a conversion to unbinned pixels or a
    single-PC re-run). The notebook still carries a set-then-assert
    binning cell, so the tutorial is correct with or without the fix
    reaching a user's kikuchipy.

    The fix is contained to the never-merged `hrebsd-dic` branch. It
    is a genuine upstream bug and a candidate for a separate upstream
    report; that is recorded here and not acted on at this gate.

    Tally, the new `TestCameraModeDatasetName` plus the untouched
    existing class:

    ```
    .venv/Scripts/python.exe -m pytest tests/test_io/test_oxford_h5ebsd.py -q
    -> 15 passed in 0.49 s
    ```

    Against the code BEFORE the fix, 4 of the 8 new tests fail (the
    three crossed-name cases and the end-to-end one, which reports
    `assert 1 == 17.0`), which is the discrimination check for this
    entry.

**Gate tally for entries 78 and 79 together (2026-09-08).**

```
.venv/Scripts/python.exe -m pytest \
  tests/test_signals/test_ebsd_hrebsd_dic.py \
  tests/test_io/test_oxford_h5ebsd.py -q
-> 71 passed in 3.91 s

.venv/Scripts/python.exe -m pytest \
  tests/test_indexing tests/test_signals -k hrebsd -q -n 4
-> 618 passed, 9 skipped in 101.21 s
```

618 is entry 77's 610 plus the eight tests of entry 78 exactly, and
the 9 skips are the same `--weekly` gates. `ruff check` and
`ruff format --check` on the three touched files (`_api.py`,
`test_oxford_h5ebsd.py`, `test_ebsd_hrebsd_dic.py`): "All checks
passed!" and "3 files already formatted".

80. **The pre-flight on the real file, and the notebook it fixed
    (2026-09-08, plan OQ13 steps 3 and 4).** Twelve read-only scripts
    were run against `AGH__Si_indent_1_512x672.h5oina` before a single
    cell of the tutorial was written, on the venv of tech-stack
    section 1, dask threaded scheduler, 8 workers. They live in the
    session scratchpad
    (`.../8ee4140a-.../scratchpad/si_indent_preflight/`, scripts
    `00_probe.py` to `10_assemble_results.py` plus `results.json`) and
    that location is **ephemeral**: the durable record of every number
    that survived is this entry and the executed
    `doc/tutorials/hrebsd_si_indent.ipynb`, whose cells re-measure the
    decisions rather than quote them.

    **The plan's premise about the reference was wrong, and the
    replacement is better.** The plan expected to find the authors'
    reference point by looking for a self-correlation of 1 in their
    `Cross Correlation Coefficient`. No such point exists: the field
    runs 0.0395 to 0.6931 with a median of 0.6866, the count at 0.999
    or above is zero, and its unique maximum (row 116, column 164, 1e-4
    above the second largest) sits in the deformed region, so it is a
    pattern quality maximum. `Cross Correlation Coefficient`, `Delta R
    Phase` and `Delta R Pseudosymmetry` are bit identical, and their
    `Strain` is nowhere all zero. MapSweeper refined every pattern
    against the dynamically simulated 3001 x 3001 master the same file
    carries, so **their strain is absolute and simulation referenced**
    while ours is relative to a measured pattern. Both fields are
    therefore reference-differenced at the same point before any
    comparison, which is the paper's own section 3.5 argument applied
    to both engines. The reference is the plan's fallback, (row 10,
    column 10), which is also the `X10Y10` reading of the record's
    strain-map PNG: the file's `Pattern Center Calibration` entries
    carry `Position X` 0 to 246 against 250 columns and `Position Y` 0
    to 228 against 234 rows, so X is the column, Y is the row, both
    zero based. Zero versus one based cannot be settled from the data
    and is one 0.2 um step inside a strain-free far field.

    **`filter_cutoffs=(None, None)`, measured twice over.** Row 10,
    250 points, the paper's own strain-free row, with its eq 17/18
    neighbour-pair estimator (std of successive along-row differences
    over sqrt(2)):

    | filter_cutoffs | converged | median sigma_eps | median sigma_theta |
    |---|---|---|---|
    | (0.05, None) | 250/250 | 0.0287 mm/m | 0.0277 mrad |
    | (None, None) | 250/250 | 0.0267 mm/m | 0.0258 mrad |

    No penalty at all: the unfiltered floor is 0.93 of the default's
    on both quantities. The capture-range test then makes the choice
    mandatory rather than merely preferable. On a 5 x 5 patch at the
    indent's south rim (rows 125:130, columns 115:120), at 500
    iterations: `(None, None)` converges 25/25 at a median rotation of
    42.08 mrad, residual 0.312, 128 iterations; `(0.05, None)`
    converges 7/25 at a median 29.28 mrad and residual 1.412, four and
    a half times worse. At the 200-iteration budget of the first
    pre-flight pass the same patch gave 2/25 at a spurious 2.10 mrad,
    residual 1.77. The default high-pass loses the initial phase
    cross-correlation lock at exactly the rotations this map carries.

    **`max_iterations=500` is not a tuning choice, it is the
    difference between a result and an empty map.** The first sub-map
    attempt at the 50 default converged **0 of 400** points. The same
    fits at 300 iterations converge 25/25 with the SAME residual
    (0.312 against 0.324) and the SAME translations (11.876 against
    11.933 px): a `min_step` threshold artefact, not a capture-range
    failure. At 200 iterations the 20 x 20 sub-map converges 345/400;
    retrying the 55 failures at 600 recovers 28 more, which needed 205
    to 497 iterations, and leaves 27 unfittable crater points at a
    residual of 1.19.

    **The crater is unmeasurable and is masked.** A 10 x 10 patch at
    its centre (rows 103:113, columns 122:132) converges **1 of 100**
    at a residual of 1.83. The notebook pre-masks it with
    `navigation_mask` from their CCC below 0.35, 728 points; masked
    points come back NaN exactly as non-converged ones do, so the hole
    in the figures is honest either way and the mask only stops those
    points burning the full iteration budget.

    **Reader traps, sharper than the plan recorded.** The reader does
    NOT return an empty crystal map: it returns a full-size
    **placeholder** of 58500 identity rotations with no phase, an
    empty structure and `scan_unit` "px", which is silently usable and
    silently wrong, so the notebook builds the map from
    `EBSD/Data/Euler`. `binning` reads 2 after the entry 79 fix and
    the notebook keeps the set-then-assert cell anyway. `px_size`
    stays a 1.0 placeholder and is provably unused: the file carries
    one PC per map point, and `_geometry.per_point_pc_pixels` takes
    that branch. The detector reads `sample_tilt` 74.9979 and `tilt`
    6.9933 degrees, not the 75 and 6.979 the plan recorded, and the PC
    spans 1.31, 1.14 and 0.47 binned px over the map.

    **The 20 x 20 sub-map (rows 125:145, columns 115:135, the south
    rim), at 200 iterations**: 345/400 converged, 1.81 patterns/s,
    median residual 0.131, strain inside +-12.4 mm/m so the paper's
    +-15 mm/m Fig 5 scale is the right one, lattice rotation to 58.8
    mrad (3.37 degrees), HR-KAM median 1.58 mrad, GND median 1.3e14
    m^-2 with b = 3.84e-10 m and estimator "a5". The whole analysis
    chain costs 0.15 s.

    **The convention verdict: one rigid frame rotation, not a fitted
    permutation.** Both fields reference-differenced at (10, 10), then
    all 36 component pairings correlated over the 345 converged
    points. The winner is `eps_theirs = R eps_ours R^T` with
    `R = [[0,1,0],[-1,0,0],[0,0,1]]`, a +90 degree rotation of the
    sample frame about z (their x is our y, their y is minus our x),
    in the plain Voigt order (11, 22, 33, 23, 13, 12) with tensor
    shears. All three sign flips fall out of that single rotation.
    Four candidate frames tie on |r|, which is sign blind; the
    regression slopes separate them, and +90 about z is the only one
    with all six slopes positive (mean +0.925 against 0.24 to 0.33 for
    the rest). Per component: e11 r=0.915 slope 0.857, e22 r=0.974
    slope 0.984, e33 r=0.939 slope 0.961, e12 r=0.991 slope 0.953,
    e23 r=0.778 slope 1.104, e13 r=0.536 slope 0.692. The four well
    resolved components (own std above 2 mm/m) all pass |r| > 0.9, so
    the pre-registered invariant-only fallback is NOT engaged. The two
    out-of-plane shears carry the smallest amplitudes of the six and
    are the pair HR-EBSD trades against the rigid translation, which
    is why they are weaker and why they are reported rather than
    hidden. Whole-field agreement 1.08 mm/m rms, 0.49 mm/m median
    absolute difference. Their first three columns are traceless to
    8.3e-5 rms, confirming a deviatoric closure like ours without
    having to assume it.

    **Floors and rates.** Far-field 20 x 20 patch (rows 20:40, columns
    20:40): 400/400 converged, median sigma_eps 0.0271 mm/m, median
    sigma_theta 0.0253 mrad, HR-KAM floor 0.064 mrad, GND floor
    2.39e12 m^-2 against a 4e12 to 8e12 m^-2 literature class, strain
    absmax 0.19 mm/m. **Ours are BELOW the paper's Table 1 (0.079 mm/m
    and 0.043 mrad) at the same pattern resolution, measured with
    their own estimator, but theirs is a 3 x 3 supersampled measure
    and ours 1 x 1**: the notebook labels both columns and makes no
    apples-to-apples victory claim. `hrebsd_pc_shift` in that same far
    field predicts the measured translations to 0.013 px rms in x and
    0.007 px rms in y on translations of about 0.09 px, which
    validates the paper's affine PC calibration and our per-point-PC
    geometry together; in the deformed zone its residual is 15.9 px,
    all of it the 43 mrad lattice rotation acting over a 379.6 binned
    px detector distance, so the notebook fits the plane on the
    strain-free region only and prints that arithmetic beside it. Zone
    rates: 9.97 pat/s on contiguous row 10, 6.75 pat/s on a scattered
    far-field patch, 3.38 in the intermediate ring, 1.81 in the
    deformed zone, 1.00 in the crater; zone-blended projection 3.76 h
    at 200 iterations and about 4.6 h at 500, 3.9 h with the crater
    masked.

    **The notebook and its smoke gate.**
    `doc/tutorials/hrebsd_si_indent.ipynb`, 32 cells (16 code, 16
    markdown), gate = `KIKUCHIPY_LOCAL_DATA_DIR` plus a
    working-directory, parent and grandparent probe (entry 81 makes
    the environment variable exclusive when it is set), every
    data-dependent cell under `if si_path:` and tagged
    `nbval-ignore-output`, so `tutorials_sanitize.cfg` needed no new
    entry. Wired into `doc/tutorials/index.rst`, `run_nbval.sh` and
    `CHANGELOG.rst`. The repository copy is left UNEXECUTED for the
    one-shot full-map run. Before returning it, a copy was patched to
    a rows 5:20, columns 5:25 slice, which contains the (10, 10)
    reference, and executed end to end against the real file:

    ```
    # from .../scratchpad/si_smoke, KIKUCHIPY_LOCAL_DATA_DIR set
    .venv/Scripts/python.exe -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=kikuchipy-spherical \
      --output smoke_out.ipynb hrebsd_si_indent.ipynb
    -> 16 of 16 code cells executed, 0 errors, 5 figures, 264 s
    ```

    The same notebook executed WITHOUT the data file (no environment
    variable, file not found) in 7.2 s: 0 errors, and the only output
    produced by any cell is the gate's "not available" line, which is
    the nbval-without-data guarantee by construction rather than by
    tagging.

    The smoke run earned its keep by falsifying a sentence. The
    pre-flight's "the high-pass locks onto a spurious 2.1 mrad near
    identity solution" was measured at a 200-iteration budget; at the
    500 iterations the notebook gives every trial, the same patch
    converges 7/25 at 29.28 mrad, so the markdown was rewritten to the
    claim the printed table actually supports: a minority converged, a
    residual several times worse, rotations below the unfiltered
    answer.

81. **The pre-execute adversarial review of the Si-indent notebook
    (2026-09-08, plan OQ13 step 5).** Twenty findings from a theory
    and a conventions reviewer, checked one by one against the
    pre-flight record, the file and the code before the one-shot
    execute, because a code fix afterwards costs the whole run. All
    twenty were upheld (four pairs were duplicates across the two
    reviewers, so seventeen distinct defects), and twenty six edits
    were applied. One reviewer's arithmetic was corrected in passing,
    recorded below.

    **The critical one: K3 was a saturation artefact.** The notebook's
    `strain_invariants` helper formed the Lode cosine
    `K3 = (3 sqrt 3 / 2) det(eps) / J2^1.5` from the reported strain,
    which is NOT traceless. Our deviatoric closure sets the trace of
    the DISTORTION to zero (`_tensors.py:350-352`), and the polar
    decomposition that follows leaves a second-order trace in the Biot
    strain, `tr(eps) = |omega|^2` to second order. MEASURED on the
    pre-flight sub-map (`submap_final.npz`, 345 converged points):
    `tr(eps)` against `|omega|^2` gives r = 0.99999, slope 0.9973,
    largest difference 0.0123 mm/m, and `tr(eps)` reaches 3.444 mm/m.
    The consequence, computed directly: `|K3| > 1` at **256 of 345
    points (74 per cent)**, range -1.320 to 0.815, silently hidden by
    the helper's `np.clip`. With the trace removed first: 0 of 345
    outside, range -1.000 to 0.963, and the clipped values differ from
    the true deviatoric ones by more than 0.2 at 18 per cent of points
    (median 0.092). K2 is barely touched (11.80 against 11.72 mm/m).
    The helper now removes the trace before either invariant, keeps
    the clip as a numerical guard, and returns the count that was
    outside; the cell prints that count both ways, and the markdown
    states the closure argument instead of asserting a range the clip
    was enforcing.

    **The trace is now disclosed rather than implied.** Their field is
    traceless to 0.083 mm/m rms (largest 2.261 mm/m); ours carries the
    rotation-driven trace above, up to 3.44 mm/m, 23 per cent of the
    +-15 mm/m display scale, spread isotropically over e11, e22 and
    e33, and invisible to the eqs 17-18 estimator, which measures
    scatter and not smooth variation. On the far-field patch our trace
    is 0.000038 mm/m, so it is purely a large-rotation effect and
    appears exactly where the paper's Fig. 5 has signal. Its effect on
    the comparison, recomputed on the sub-map with the notebook's own
    FRAME: whole-field rms difference 1.0800 mm/m as printed against
    0.9931 mm/m with our trace removed, per component e11 1.328 ->
    0.986 and e22 1.421 -> 0.959 (e33 1.149 -> 1.459). The strain cell
    now prints our trace, its rms and its correlation with
    `|omega|^2`, the comparison cell prints the trace-removed
    agreement beside the reported one, and a limitations bullet names
    the magnitude.

    **The rechunk was 250 times larger than its own comment.**
    `kp.load` returns a 4-D lazy array, so
    `rechunk({0: 32, 1: -1, 2: -1})` set 32 map ROWS per chunk and
    left axis 3 alone. VERIFIED against the real file: as loaded
    `(1, 1, 512, 622)` = 0.32 MB; after the old rechunk
    `(32, 250, 512, 622)` = **2.5477 GB per chunk, 8 blocks**, which
    `EBSD.hrebsd_dic`'s `reshape((-1,) + sig_shape)`
    (`ebsd.py:2835`) turns into flat chunks of 8000 patterns, against
    a comment and a markdown paragraph that both said "about 32
    patterns per chunk" and "never held in memory". The old smoke run
    printed `chunks: (32, 250, 512, 622)` directly under that prose.
    Now `rechunk({0: 1, 1: 32, 2: -1, 3: -1})`, measured
    `(1, 32, 512, 622)` = 10.19 MB, 1872 blocks, flat chunks of 32,
    no dask warnings. It changed no number and it is faster: the
    re-run smoke measured 12.09 pat/s against 10.49 on the same 300
    points, and 8.12 and 9.21 pat/s on row 10 against 7.83 and 8.48.
    Every pre-flight rate behind the run-time projection was measured
    with the old chunking, so the projection is conservative.

    **The noise-floor comparison had the like-for-like number in
    memory and never computed it.** The table compared our measured
    0.0267 mm/m against the paper's PUBLISHED 0.079 mm/m and explained
    the gap as "ours 1x1, theirs 3x3 supersampled", an explanation
    that runs the wrong way, since smoothing lowers a point-to-point
    scatter estimator. Their own delivered field is in the file, its
    header reads `Refinement Binning = None` (read here, read-only),
    so it is their full-resolution refinement of the same patterns.
    The same estimator on the same row 10 of THEIR field gives per
    component [0.0278, 0.0584, 0.0593, 0.0253, 0.0474, 0.0403] mm/m,
    median **0.0438**, which the table now prints as a third column.
    The published 0.079 keeps its own column and is stated as neither
    reproduced by their own field nor explained here, rather than
    attributed to a cause the notebook has not measured.

    **The acceptance gate would have dropped a component it passes.**
    The pre-registered rule "own std over the map > 2 mm/m, then
    |r| > 0.9" was calibrated on a 20x20 deformed sub-map; the full
    map is four fifths far field. Predicting each component's full-map
    std from their field over the 57772 unmasked points times the
    measured slopes: e11 1.978 x 0.857 = **1.70**, e22 2.223 x 0.984 =
    2.19, e33 2.268 x 0.961 = 2.18, e23 0.294 x 1.104 = 0.32, e13
    0.291 x 0.692 = 0.20, e12 3.047 x 0.953 = 2.90. So e11, whose
    correlation is r = +0.915, falls below the old threshold and would
    have been silently dropped from a verdict reading "3 of 3 pass".
    (The reviewer paired their e22 std with the e11 slope and reported
    1.91 and a 5 per cent miss; the pairing is fixed by the rotated
    frame, and the real margin is 15 per cent. The finding stands, the
    arithmetic did not.) The threshold is now 1 mm/m, about forty
    times the 0.0267 mm/m floor, still admitting exactly the four
    components the pre-flight called well resolved and still excluding
    the two out-of-plane shears; the markdown states it and its
    dilution argument before the numbers, and the empty case is
    guarded so no vacuous "0 of 0" can print. The test is also now
    `r > 0.9` rather than `abs(r) > 0.9`, since the frame is pinned to
    one specific rotation and a negative correlation would mean it is
    wrong, and the count of positive slopes, which is the
    discriminator the frame was actually chosen by, is printed with
    the verdict.

    **The nbval gate did not hold on this machine.** The cell-4
    fallback probed the working directory, its parent and its
    grandparent unconditionally; `run_nbval.sh` runs from the
    repository root; the 18.9 GB file sits at the repository root
    (untracked, ignored by `.gitignore`). So nbval, which executes
    every cell and only suppresses output comparison with
    `nbval-ignore-output`, would have started the multi-hour map cell.
    The gate now makes `KIKUCHIPY_LOCAL_DATA_DIR` exclusive: when it
    is set, only that directory is searched, so naming an empty one
    switches the tutorial off; only when it is unset are the working
    directory and its two parents tried, which keeps the one-shot
    execute working with no environment variable set. `run_nbval.sh`
    exports it to a fresh `mktemp -d` when the caller has not, with
    the reason in a comment. MEASURED, both branches, from the
    repository root: unset -> the file at the repository root is
    found; set to an empty directory -> `si_path` is None, and a full
    `nbconvert --execute` of the repository notebook finishes in
    **7.3 s with exactly one output**, the gate's "not available"
    line, on the `nbval-ignore-output` cell. `nbval` itself is not
    installed in this venv, so the plan's step-7 gate is still to be
    run; what is measured here is that it cannot start the
    correlation.

    **The remaining corrections, all against measurements.** Their
    correlation maximum at (row 116, column 164) sits at a band
    contrast of 218 against a map median of 220 and a maximum of 242,
    the 17.5th percentile, so "a pattern quality maximum" was replaced
    by what the pre-flight supports: unique, but only 1e-4 above the
    runner up, and in the deformed region. "The file's own metadata
    says it is the same point the authors used" was replaced by what
    X10Y10 can mean, since their refinement had no experimental
    reference at all. The tracelessness check now prints the next best
    zero-sum triple beside the winner (36.43 mm/m against 2.26 mm/m,
    16x) so it reads as a discrimination. The patterns are lzf
    compressed with the shuffle filter, one per HDF5 chunk, at a ratio
    of 1.0003 (18.630 GB of pixels, 18.625 GB stored), not
    "uncompressed". The `num_workers` pin's comment claimed to pin
    "the chunking reported by the correlation", but every call passes
    an explicit `chunksize` and `verbose=0`, so `estimate_chunksize`
    and `get_info_message` are never reached; it now says what it
    does. The binning note says "released kikuchipy versions", since
    the entry 79 fix is fork-only. `FAR` is labelled as the 20 by 50
    corner it is, distinct from the 250 points of row 10 the sigma
    columns come from. The predicted pattern shift is printed as "at
    most", since it uses the full rotation magnitude including the
    component about the surface normal, which shifts nothing. The
    K2/K3 caveat covers both invariants. Three limitations bullets
    were added: the closure trace above, "relative to one pattern"
    (their absolute field puts our reference at
    [1.18, -2.33, 1.15, -0.61, -0.16, 0.31] mm/m, so our zero is
    demonstrably not zero strain), and "one map, one indent, no
    repeat". The run-time sentence no longer promises "four to five
    hours" ahead of a measurement the faster chunking changes.

    **The re-run smoke.** Nine code cells changed, so the author's
    smoke procedure was repeated on the same rows 5:20, columns 5:25
    slice:

    ```
    # from .../scratchpad/si_smoke, KIKUCHIPY_LOCAL_DATA_DIR set
    .venv/Scripts/python.exe -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=kikuchipy-spherical \
      --output smoke_out2.ipynb hrebsd_si_indent.ipynb
    -> 16 of 16 code cells executed, 0 errors, 5 figures
    ```

    Every number that both smoke runs print is identical (row 10
    floors 0.0287 and 0.0267 mm/m, the rim table 7/25, 0/25 and 25/25
    at 29.28, nan and 42.08 mrad, the residual medians, the PC-shift
    planes), so the chunking fix moved nothing but the rate. The new
    lines print correctly on far-field-only data:
    `chunks: (1, 32, 512, 622)` under the markdown that claims 32
    patterns, K3 outside the range at 0 points both ways, the
    empty-verdict guard reading "no component reaches the threshold,
    so the rule is not tested here", six of six slopes positive, and
    the third noise-floor column at 0.0438 mm/m. The repository
    notebook is still UNEXECUTED, with 0 outputs and every
    `execution_count` None.

82. **The one-shot full-map execute and its post-execute
    verification (2026-09-08, plan OQ13 step 6).** The notebook was
    executed once, in place, on the whole 250 by 234 map with the
    real file present, and every markdown claim in it was then
    checked against the printed outputs. Machine A, the repository
    `.venv` (CPython 3.13.12), kernelspec name `python3`, which is
    what every tutorial but `spherical_indexing.ipynb` carries, and
    `dask.config num_workers=8` pinned in the notebook's first cell.
    Wall clock from the notebook's own `execution` metadata: first
    cell in at 2026-09-08T23:52:12Z, last cell out at
    2026-09-09T02:16:53Z, 2 h 25 min end to end, of which the map
    cell alone is 2 h 20 min.

    ```
    # plan step 6 recipe; no shell log was kept, so the notebook's
    # own per-cell execution metadata is the record of the run
    .venv/Scripts/python.exe -m nbconvert --to notebook --execute \
      --inplace --ExecutePreprocessor.timeout=-1 \
      doc/tutorials/hrebsd_si_indent.ipynb
    -> 16 of 16 code cells executed, execution counts 1..16, no
       error output, 5 figures, 1.39 MB stored
    ```

    **The run, as the notebook prints it.** Crater mask from their
    CCC below 0.35: **728 points masked** (the pre-flight's number
    exactly), 57772 correlated. **2.34 h at 6.85 patterns/s on 8
    workers**, against entry 80's zone-blended projection of about
    4.6 h at 500 iterations and 3.9 h with the crater masked: the
    projection was conservative, as entry 81 predicted it would be,
    since every zone rate behind it was measured with the old
    chunking. Convergence **57685 of 57772 converged, 87 did not,
    728 masked**, so 815 of 58500 points carry NaN. Iterations
    median 14, largest 500, which is the cap itself; residual median
    0.046. The four-function analysis chain after it: 0.43 s on
    58500 points, as the markdown promises.

    **Strain, rotation, invariants.** Deviatoric closure, 1st to
    99th percentile per component (mm/m): e11 -8.62 to +3.15, e22
    -8.23 to +3.15, e33 -0.14 to +9.20, e23 -0.45 to +0.57, e13
    -0.65 to +1.04, e12 -9.75 to +11.11, with 99.94 per cent of all
    components inside the paper's +-15 mm/m Fig. 5 scale, so the
    pre-flight's choice of that scale holds on the full map. Lattice
    rotation median 0.42 mrad, **largest 87.1 mrad = 4.99 degrees**,
    well past the pre-flight sub-map's 58.8 mrad (3.37 degrees) and
    past the 4.0 degree unfiltered figure of V4 (see the markdown
    reconciliation below). Our closure trace: largest 7.587 mm/m,
    rms 0.462 mm/m, correlated with `|omega|^2` at r = 1.00000, and
    0.0871^2 = 7.586 mm/m, so entry 81's `tr(eps) = |omega|^2`
    identity is confirmed to four figures on the real map. **K3
    outside [-1, 1] at 5025 of 57685 points without the trace
    removal and 0 with it**: entry 81's critical fix is load bearing
    here too, at 8.7 per cent rather than the deformed sub-map's 74
    per cent, because four fifths of this map is far field where the
    trace is nothing. HR-KAM median 0.081 mrad, largest 39.3 mrad.
    GND median 3.76e12 m^-2, largest 2.26e15 m^-2, over 57535 finite
    points.

    **The comparison with their own field, the headline result.**
    Both fields reference differenced at (10, 10), ours rotated into
    their frame by the pre-flight's `R = [[0,1,0],[-1,0,0],[0,0,1]]`,
    over the 57685 usable points:

    | component | our std | their std | r | slope | rms difference |
    |-----------|---------|-----------|-------|-------|------|
    | e11 | 1.838 | 1.971 | 0.971 | 0.905 | 0.492 |
    | e22 | 1.967 | 2.189 | 0.982 | 0.882 | 0.483 |
    | e33 | 2.142 | 2.245 | 0.979 | 0.934 | 0.467 |
    | e23 | 0.433 | 0.285 | 0.260 | 0.396 | 0.452 |
    | e13 | 0.320 | 0.277 | 0.224 | 0.259 | 0.396 |
    | e12 | 2.949 | 2.995 | 0.991 | 0.976 | 0.413 |

    (mm/m throughout except r and the slope.) Well resolved by the
    pre-registered rule, own std above 1 mm/m: e11, e22, e33, e12,
    and **4 of 4 pass r > 0.9 with a positive slope**. **6 of 6
    slopes positive**, the discriminator the frame was actually
    chosen by, so the +90 degree frame verdict of entry 80 survives
    the full map. **Whole field agreement: rms difference 0.452 mm/m,
    median absolute difference 0.133 mm/m**, and 0.441 mm/m with our
    rotation driven trace removed first, so the closure trace
    accounts for 0.011 mm/m of the 0.452 and the rest is real
    disagreement between the two implementations. Entry 81's
    threshold change from 2 to 1 mm/m was necessary exactly as it
    predicted: e11's own std is 1.838 and e22's is 1.967, both under
    the old gate, so the old rule would have printed "2 of 2 pass"
    and silently dropped two components that agree at r = 0.971 and
    0.982.

    **The noise floors, all three columns.** Row 10, all 250 points,
    the paper's own strain free row, the eqs 17-18 neighbour pair
    estimator, `(None, None)` band-pass:

    | quantity | ours, measured | theirs, measured | theirs, published |
    |----------|----------------|------------------|-------------------|
    | strain, median | 0.0267 mm/m | 0.0438 mm/m | 0.079 mm/m |
    | rotation, median | 0.0258 mrad | not in the file | 0.043 mrad |

    Per component, ours (mm/m): e11 0.0475, e22 0.0236, e33 0.0465,
    e23 0.0246, e13 0.0160, e12 0.0287; theirs: 0.0278, 0.0584,
    0.0593, 0.0253, 0.0474, 0.0403. The `(0.05, None)` arm of the
    same cell gives 0.0287 mm/m and 0.0277 mrad, so no band-pass is
    the better floor as well as the required capture range, and the
    penalty clause of the pick rule does not fire (ratio 0.929).
    **The caveats that travel with every number in this table**:
    (i) ours is relative to a measured reference and theirs is
    absolute and simulation referenced, so the columns share an
    estimator and not a quantity; (ii) **the two per component rows
    are in different frames**, ours in our sample frame and theirs
    in theirs, and those differ by the +90 degree rotation above, so
    reading them entry against entry is wrong. Paired through the
    frame (ours e11 against their e22, ours e22 against their e11,
    ours e33 against their e33, ours e23 against their e13, ours e13
    against their e23, ours e12 against their e12), OUR FLOOR IS
    LOWER ON ALL SIX: 0.0475/0.0584, 0.0236/0.0278, 0.0465/0.0593,
    0.0246/0.0474, 0.0160/0.0253, 0.0287/0.0403. That pairing is
    arithmetic done in this entry on the printed numbers, not
    something the notebook computes; what the notebook now says is
    that the two rows compare as sets and not entry by entry, which
    is the honest claim. (iii) The estimator measures point to point
    scatter only and is blind to smooth error. (iv) The published
    0.079 mm/m is reproduced by neither measured column and is left
    unexplained rather than attributed to a cause not measured here.

    **The far field, and the GND floor.** 1000 points of the 20 by
    50 strain free corner: largest |strain| 0.221 mm/m, largest
    |rotation| 0.216 mrad, HR-KAM median 0.0610 mrad, **GND median
    2.49e12 m^-2** against the 4e12 to 8e12 m^-2 class quoted for
    HR-EBSD at fine steps, so our floor sits below that class rather
    than inside it, which the markdown now says plainly ("sits
    under" rather than "just under"). The deformed zone reaches
    **2.26e15 m^-2**, 906 times the far field floor. Throughput in
    context: 6.85 pat/s on 8 CPU workers at 622 by 512 px against
    the paper's own about 20 pat/s on two RTX 4090 GPUs.

    **The PC-shift diagnostic.** Strain free corner, 1000 points, in
    binned pixels: translations median -0.0759 (x) and -0.0150 (y)
    at rms 0.1086 and 0.0262; residuals **rms 0.0082 px in x and
    0.0203 px in y**, plane fits -0.0009, -1.34e-4 per column,
    +3.78e-4 per row (x) and -0.0180, -4.19e-4 per column, +1.25e-3
    per row (y). So the paper's affine PC calibration and our
    per-point-PC geometry agree to about one hundredth of a pixel in
    x and two hundredths in y, on measured translations of about a
    tenth of a pixel. Deformed zone, 4489 points above 20 mrad: DD
    379.6 binned px, median |rotation| 32.4 mrad, residual 12.13 px
    against at most 12.28 px predicted from the rotation alone, so
    it is lattice signal and not miscalibration, the same verdict
    the pre-flight reached at its own 43 mrad and 15.9 px.

    **The post-execute markdown verification, and what it rests on.**
    Every quantitative and qualitative claim in the sixteen markdown
    cells was checked against the printed outputs; where an output
    cannot settle a claim, against the file itself (read-only
    `h5py`) or against this ledger. VERIFIED FROM THE FILE in this
    entry: their CCC maximum is unique by **9.239e-5** ("only 1e-4
    above the runner up") at row 116, column 164, at a band contrast
    of 218 against a map median of 220, the 17.5th percentile, and
    38.5 map points from the centroid of the sub-0.35 crater, so
    "unique, in the deformed region, a little below the median band
    contrast" all hold; `Refinement Binning` reads `b"None"`; the
    master is 3001 by 3001; `Pattern Center X`, `Pattern Center Y`
    and `Detector Distance` carry 58500 values each, so
    `detector.pc` is (234, 250, 3); `Camera Binning Mode` is
    `b"Speed 1 (622x512 px)"`. VERIFIED FROM THIS LEDGER: the crater
    patch converging 1 of 100 at residual 1.83 (entry 80), the
    iteration-50 versus later-iterate residuals and translations
    (entry 80), and the 2.0 against 4.0 degree capture range, which
    is `EBSD.hrebsd_dic`'s own docstring number from V4.

    **The figures, and the honest limit of that check.** Five PNGs,
    110 to 244 KB, 902 by 249 to 906 by 481 px, 2928 to 14420
    distinct colours, none blank or collapsed. Panel counts from the
    `text/plain` companion: 12, 12, 6, 12 and 6 axes, which is 6, 6,
    3, 6 and 3 map panels plus one colour bar each, as the gallery
    calls expect. The NaN grey the helper sets (`cmap.set_bad`,
    0.75 grey) covers 0.62 to 0.79 per cent of the four figures
    built on our own fields and 0.07 per cent of the their-field
    figure, which has no NaN at all: consistent with 815 missing
    points of 58500 spread over map panels that occupy about half of
    each figure. THE LIMIT: the pixels were not looked at, only
    their statistics and the axis counts, so what is established is
    "not blank, right number of panels, the mask hole is present and
    of the right size", not "the figure looks right".

    **Hygiene, all checked on the final file.** 32 cells (16 code,
    16 markdown); execution counts 1..16, monotonic, none missing or
    `None`; zero error outputs; every output-producing code cell
    tagged `nbval-ignore-output`, the two untagged ones (imports,
    helper definitions) producing no output at all; exactly one
    `nbsphinx-thumbnail`, the strain gallery of cell 21, carrying
    both the tag and the tooltip metadata; kernelspec `python3`; no
    em-dash or en-dash anywhere in any cell source or output, and no
    non-ASCII character in any markdown source. The three
    `UserWarning`s in the stored stderr (18 of 25 and 25 of 25 on
    the rim configurations, 87 of 57772 on the map) are the
    demonstration itself, and the markdown around them says so
    before they appear, so they are kept.

    **Thirteen markdown edits, no code cell and no output touched,
    so no re-execution.** Applied with `nbformat` under a
    one-occurrence assertion each.

    1. Cell 1, "the same 58500 points" -> "the 57685 points of 58500
       where our own fit returns a number": the crater mask and the
       87 failures are not in the comparison.
    2. Cell 5, the binning bullet now adds that the run stored below
       prints 2, from the entry 79 fix that is not in a release,
       since the claim around it is about released versions and the
       output says 2.
    3. **Cell 11, the capture range reconciliation**, a new
       paragraph after the pick rule: the 50 mrad was an estimate,
       the map reaches 87.1 mrad = 4.99 degrees, past the 4.0 degree
       unfiltered figure, and that is not a contradiction, because
       the 4.0 degrees is measured on a pure IN-PLANE rotation,
       which nothing in the initial guess can undo, whereas a
       lattice rotation about an in-plane axis mostly TRANSLATES the
       pattern, by omega times DD pixels, and every fit is seeded
       per point by the phase cross-correlation of `initial_guess`,
       which measures exactly that translation. The PC-shift cell's
       12.13 px against 12.28 px is that arithmetic on this map.
    4. Cell 14, "the same as at iteration 300" -> "iteration 500",
       which is the comparison the table below it actually prints
       (the 300 is true of the pre-flight run of entry 80 and
       appears nowhere in the notebook).
    5. Cell 16, "the rate varies over the map by a factor of five"
       -> "close to an order of magnitude", and "about ten
       iterations" -> "ten to fifteen", against the printed median
       14 over the map and 128 on the rim.
    6. Cell 18, the closure trace "reaches a few mm/m" -> "reaches
       7.6 mm/m", against the printed largest 7.587 mm/m.
    7. Cell 20, a new sentence: the rotation panels saturate,
       because the map's 87.1 mrad is above the paper's +-50 mrad
       scale the panels are drawn at.
    8. Cell 26, "sits just under the 4e12 to 8e12" -> "sits under",
       for a measured 2.49e12.
    9. Cell 26, the new frame caveat on the two per component
       noise-floor rows, described above.
    10. Cell 28, "a rotation of 43 mrad is a shift of 16 pixels" ->
        "32 mrad ... 12 pixels", the map's own printed numbers
        rather than the pre-flight sub-map's.
    11. Cell 30, "to about a millimetre per metre" -> "to about half
        a millimetre per metre", against the printed 0.452 mm/m rms
        and 0.133 mm/m median absolute.
    12. Cell 30, their tracelessness "to a fraction of a hundredth
        of a mm/m" -> "to 0.08 mm/m rms, 2.3 mm/m at worst", against
        the printed 0.08316 and 2.2610 mm/m.
    13. Cell 30, our trace "a few mm/m at the largest lattice
        rotations" -> "7.6 mm/m at the largest lattice rotation".

    One repair was needed on the way in: a heredoc ate a backslash
    and put a TAB into cell 11's `\times`, caught by a tab scan over
    every cell and fixed; the final file carries no TAB anywhere.

    **No BLOCKING findings: nothing in a code cell needs changing.**
    Every printed number the markdown depends on is computed by the
    cell that prints it, the two invariants are formed on a
    trace-removed tensor as entry 81 requires, the comparison
    rotates our tensors rather than permuting columns, the verdict
    counts are computed from the same arrays they describe, and the
    empty-case guards were not needed on the full map but are still
    correct. The one structural remark, recorded and NOT acted on
    because it would be a code-cell change: the imports cell and the
    helper cell carry no `nbval-ignore-output`, which is right today
    because they print nothing, but a future deprecation warning
    from an import would then be an nbval diff rather than an
    ignored output.

    Plan step 7 (nbval on the full list, ruff, untouched-suite
    check, commit) is still to run; nothing in this entry is a gate
    result.

### V8 -- neighbour-seeded propagation oracles (Stage D, 2026-09-09)

(a) Rescue oracle: a deformed-master synthetic map carrying a smooth
rotation ramp whose far points exceed the phase-XC-seeded basin but
whose neighbour chain is everywhere within it; the default path must
FAIL those points (establishing the premise) and the seeded path must
recover the imposed field within the deformed-master band. (b) D20.6
equivalence: on points both paths converge WITHIN THE RUN'S OWN
BUDGET, corner-displacement agreement within SEED_EQUIVALENCE_TOL
(MTP). (c) Determinism: seeded runs bitwise across repeats,
chunksizes, lazy/eager, on a map whose cascade rounds each carry TWO
points with different seeds. (d) Isolation: grain-boundary and mask
non-crossing, constructed so a violation corrupts visibly.
(e) PASS1_CAP semantics and the rescue pass. (f) D20.7 Si rim timing,
recorded. Entries from 83 below.

#### V8 recorded results, failing-tests gate (2026-09-09)

Measured on this machine with the venv Python, engine at commit
cec39de4 (the last pre-Stage-D commit) for every default-path number.
`tests/test_indexing/test_hrebsd_seeding.py` carries each recipe in
the comment next to the constant it feeds.

**V8(b), the D20.6 counterexample (the reason D20.6 carries a dated
correction).** Default path on the V8(a) ramp row, `fit_pattern`
against the row's own reference, seeded by phase cross-correlation:

| ramp point | budget 200 | budget 2000 | seeded (chained) |
| --- | --- | --- | --- |
| 0.8 deg | conv, 5 it, 0.0019 px | conv, 5 it, 0.0019 px | conv, 5 it, 0.0019 px |
| 1.6 deg | conv, 10 it, 0.0056 px | conv, 10 it, 0.0056 px | conv, 5 it, 0.0056 px |
| 2.4 deg | NOT conv, 200 it, 47.66 px | conv, 291 it, 50.84 px | conv, 5 it, 0.0096 px |
| 3.2 deg | NOT conv, 200 it, 48.70 px | conv, 281 it, 50.81 px | conv, 5 it, 0.0143 px |
| 4.0 deg | NOT conv, 200 it, 42.85 px | conv, 320 it, 49.90 px | conv, 5 it, 0.0119 px |
| 4.8 deg | NOT conv, 200 it, 105.24 px | conv, 728 it, 108.04 px | conv, 5 it, 0.0138 px |

Errors are the V2 corner-displacement metric against the EXACT
imposed homography. So the unqualified D20.6 is refuted: above a
budget of about 291 the default path converges those points to a
different optimum ~50 px away. D20.6 is narrowed to the budget-bounded
claim; the both-converged population at `MAX_ITERATIONS = 200` is
`{0, 1, 2, 17}` and is now pinned literally in the test.

**V8(d), what the grain gate buys (the crossed-seed simulation).**
`fit_pattern` of the isolation map's hard point against grain 1's own
reference, budget 200, seeded with the exact homography of the
neighbour named:

- from column 3, the same-grain neighbour: CONVERGED in 19 iterations,
  0.100195 px of the 36.4788 px imposed (0.27 %)
- from column 1, the cross-grain neighbour: NOT converged, budget
  exhausted, 30.8866 px away (84.67 %)

Residual ordering premise, independent fits: cross 8.039623e-04 <
same 1.024855e-02, a 12.75-fold ordering, so the lowest-residual
neighbour of the hard point IS the one across the boundary.

**Mutant M10, the two-ramp fixture (per-point h0 off by one in flat
order).** Cascade round membership measured on the pre-existing
fixtures is ONE point per round everywhere (ramp, isolation, and the
four-point pin map), so the mutant had no designed killer. On the new
two-ramp map (rows 0 and 2 carrying opposite-signed ramps, row 1
masked) each round carries two points with seeds 3.2 degrees apart:

| round | correct seed | swapped seed (the mutant) |
| --- | --- | --- |
| 1, row 0 col 3 | conv, 5 it, 0.0096 px | NOT conv, 200 it, 47.60 px (365 % of imposed) |
| 1, row 2 col 3 | conv, 5 it, 0.0078 px | NOT conv, 200 it, 49.51 px (379 %) |
| 2, row 0 col 4 | conv, 5 it, 0.0143 px | NOT conv, 200 it, 47.37 px (272 %) |
| 2, row 2 col 4 | conv, 5 it, 0.0098 px | NOT conv, 200 it, 52.78 px (303 %) |

**D20.1, the pre-Stage-D pin.** The default-off pin used to compare
two live calls on the same code path (`f(x) == f(x)`) and was
demonstrated unable to fail. It now compares against frozen literals
measured on commit cec39de4 and verified bitwise against today's
`False` path. Sizing on the four-point pin map, corner-displacement
against the frozen homography: two live runs 5.6843e-14 px (the
metric's own inversion floor), `upsample_factor=4` 1.9614e-06 px,
`upsample_factor=8` 3.1815e-06 px, `min_step=1e-2` 5.6042e-06 px.
Pinned at 1e-9 px. The exact `num_iterations` pin `[1, 5, 10, 200]`
independently kills every regression that moves an iteration count
(`max_iterations=6` gives `[1, 5, 6, 6]`, `min_step=1e-2` gives
`[1, 4, 10, 200]`, `border=0.06` gives `[1, 5, 7, 9]`). Same
correction and same provenance in the signal-level twin, on the
shipped Ni map.

**D20.2, the PC-transport bound, re-measured on the REAL projection
centre.** Read from `AGH__Si_indent_1_512x672.h5oina` 2026-09-09
(`/1/EBSD/Data`): Pattern Center X mean 0.526845 span 0.002110,
Pattern Center Y mean 0.678911 span 0.001827, Detector Distance mean
0.610303 span 0.000759. Oxford normalises all three by the pattern
WIDTH, so those spans times ncols = 622 reproduce ledger entry 80's
1.31, 1.14 and 0.47 binned px. In the kikuchipy (Bruker) frame the PC
is (327.70, 89.72, 379.61) px. Worst per-step phantom and its
fraction of the 2.0 degree capture range:

| reading | worst (px) | fraction |
| --- | --- | --- |
| kikuchipy Bruker frame (the real one) | 0.009727 | 5.637e-04 |
| raw Oxford, PCy from the top | 0.008578 | 4.972e-04 |
| raw Oxford, PCy from the bottom | 0.009727 | 5.637e-04 |
| fractions of the pattern height | 0.009171 | 6.014e-04 |
| the superseded central stand-in | 0.009478 | 7.508e-04 |

All five stay under `PC_TRANSPORT_BOUND = 0.019` and
`PC_TRANSPORT_BASIN_FRACTION = 1.6e-3`, so the D20.2 decision is
unchanged; only its provenance is corrected. The test now sweeps all
five rather than asserting the one.

**PASS1_CAP admissible window.** The frozen `seed_round` EXPECTED
array holds for `PASS1_CAP` in [10, 112]. It follows from two
measurements: the 1.6 degree ramp point converges from its own phase
cross-correlation seed in 10 iterations (a cap of 9 shifts the whole
ramp by one round), and the isolated rescue point in 113 (a cap of
113 or more converges it in pass 1, so `seed_round[17]` becomes 0
rather than -2 and the rescue pass stops being exercised). The
drafting candidate 50 sits comfortably inside.

**Tightest margin in the module.** The rescue point recovers to
0.388833 px of the 91.2533 px it imposes, 0.4261 % against the 1 %
`SAME_OPTIMUM_FRACTION` bound, i.e. 2.35x -- next to 23x for the
easiest ramp point and 3.6x for the isolation hard point.

**Imposed corner norms of the fixtures**, which correct two
denominators the `SAME_OPTIMUM_FRACTION` audit comment had wrong:
ramp 0, 4.3500, 8.6999, 13.0493, 17.3980, 21.7460, 26.0928 px; rescue
point 91.2533 px; unfittable point 43.4654 px; isolation hard 36.4788,
cross 4.3500, same 26.8657 px. So a recovered field is 0.044 to
0.43 % of its imposed displacement and a wrong optimum 85 to 403 %,
two populations about 200x apart.


#### V8 recorded results, Stage D measurement-close gate (2026-09-09)

Machine A (the 20-core Windows 11 laptop of every entry above:
Intel64 Family 6 Model 186 Raptor Lake, 20 logical cores, Windows 11
build 26200, `.venv` Python 3.13.12, numpy 2.4.6, scipy 1.17.1, numba
0.65.1, scikit-image 0.26.0, orix 0.14.2, dask 2026.3.0, 8 dask
workers). Scripts in the session scratchpad `hrebsd_measure_d/`
(`common.py` load recipe = the executed `hrebsd_si_indent.ipynb`:
`AGH__Si_indent_1_512x672.h5oina` lazy, binning 2, per-point Euler
crystal map, reference (10, 10), `filter_cutoffs=(None, None)`; the
18.9 GB file READ ONLY). These close the D20.2 PASS1_CAP debt and the
D20.7 performance record, and they CORRECT the flawed
rim-in-isolation reading of the failing-tests gate (`rim_timing.json`).
No engine constant, requirement or frozen test was changed by this
gate; the whole hrebsd suite stays green (`132 passed` on the two
Stage D files, `694 passed, 9 weekly-skipped` on the hrebsd `-n 4`
selection).

83. **PASS1_CAP re-pin, done right: the far-field convergence
distribution, and why the rim-in-isolation panic was misleading
(requirements D20.2; decision KEPT at 50).** [script
`05_anchor_bearing.py`, `anchor_bearing.json`]

PASS1_CAP must sit just above the iteration count the EASIEST points
need, so anchors form wherever easy points exist. MEASURED on a clean
far-field patch (rows 20:40 cols 20:40, 400 points, default path,
budget 500): every point converges, in min 6, median 9, p90 9, **p95
10**, p99 10, max 10 iterations (378 of 400 in 5 to 10, 22 in 10 to
15). The shipped `PASS1_CAP = 50` sits **5.0x above that p95**, so it
catches every easy point in pass 1 with wide margin. KEPT at 50; the
engine constant is untouched.

The rim-in-isolation cap sweep of the failing-tests gate
(`cap_sweep.log`, on rows 125:145 cols 115:135: cap 50 gives 0 pass-1
anchors, a dead cascade and 373/400 all-rescue; cap 80 gives 8 anchors
plus 189 cascade) is a PATCH ARTEFACT, not a reason to raise the
global default. That patch lies entirely inside the deformed rim and
excludes the far field, so it has NO easy points at all: at cap 50
nothing converges in pass 1 because nothing in it converges under 50
phase-XC iterations, so no anchor can form. On the full map the far
field surrounds the rim and seeds inward, which the anchor-bearing
patch of entry 84 demonstrates directly (46 pass-1 anchors seeding 571
cascade conversions at the shipped cap 50). The measurement does NOT
say 50 is wrong for anchor-bearing use, so per the Stage D
commissioning rule the constant is KEPT and the rim panic is recorded
as misleading. (50 carries generous margin over p95 = 10; a lower cap
near 15 to 20 would trim the pass-1 waste of entry 84 but is a speed
micro-optimisation for a feature whose real-data speed benefit is
negative (entry 84), is uncommissioned, and would move real-data
`seed_round` assignments, so it is refused.)

84. **The CORRECTED D20.7 performance record, on an ANCHOR-BEARING
patch (requirements D20.7).** [same script and json]

The failing-tests D20.7 (`rim_timing.json`) measured the
rim-in-isolation patch and found 0.94x (seeding slightly SLOWER) with
+0 extra conversions, because at cap 50 that all-deformed patch formed
zero anchors and the cascade was dead (all 373 via the rescue pass).
The honest measurement needs a patch that straddles the strain-field
edge so anchors exist and can propagate. CHOSEN by inspecting the
Oxford CCC map: rows 115:140 cols 100:130 with the entry-80 crater
mask (CCC < 0.35), 618 fitted points, a strong far-field margin (cols
100 to 108 at CCC 0.7) wrapping the crater rim. Both ways at
`max_iterations=500` and the shipped `PASS1_CAP=50`:

| path | wall | pat/s | converged | iterations | phase groups |
|---|---|---|---|---|---|
| default | 338.3 s | 1.83 | 618/618 | 54032 | 1 |
| seeded | 603.2 s | 1.02 | 617/618 | 61879 | 23 |

Speedup **0.56x (seeding 1.8x SLOWER)**, iteration ratio 0.87x
(seeding used 15 % MORE iterations), extra conversions **-1**. The
cascade ENGAGED strongly this time (46 pass-1 anchors, then 20 cascade
rounds converting 571 points, 0 rescued, 1 never), so the number is a
real cascade, not a dead one, and it is slower for two measured
reasons. (a) The pass-1 cap of 50 spends 30693 of the 61879 seeded
iterations, and only 46 of 618 points converge inside it, because
patch C's near-indent "far field" is not fast: those points carry a
real lattice rotation against the (10,10) reference and need a mean of
87 phase-XC iterations (54032/618), not the 9 of the pristine far
field of entry 83. So 572 points burn ~50 wasted pass-1 iterations
before the cascade re-fits them. (b) The cascade fits are cheap per
point (~37 iterations from a neighbour seed) but 50 wasted + 37 is
about 87, the default's own per-point count, so the saving is
cancelled, and the 23 phase-group launches add orchestration overhead
the single default batch never pays. The plan 9.4 projection of 3 to
8x assumed neighbour seeds collapse deformed-point iterations to a
handful, which they do on the SMOOTH synthetic ramp (5 iterations) but
not on the real steep-gradient indent field (37), where phase-XC is
already only ~87.

85. **The correctness disentanglement: on real Si the cascade rescues
NOTHING and slightly HARMS the marginal rim (requirements D20.6,
honest record).** [script `06_disagreement.py`,
`disagreement.json`/`.npz`]

For the "extra conversion" question D20.7 asks, points the seeded path
converges that the default does not, the answer on patch C at budget
500 is NONE (0 extra; seeding LOST one, r128 c122, a CCC-0.382 rim
point default solved in 456 iterations and seeding left unconverged at
500). So the correctness benefit is not exercised on real Si at this
budget: phase-XC already reaches every convergeable optimum. To find
whether seeding nonetheless changes WHICH optimum, the per-point
seeded-vs-default corner-displacement disagreement was measured over
the 617 both-converged points. 612 of 617 agree to the metric's float
floor (median 6.3e-04 px). FIVE disagree by more than 0.1 px, every
one of them a low-CCC crater-rim point converted in a late cascade
round:

| r | c | CCC | seed_round | disagree | def resid | seeded resid | winner |
|---|---|---|---|---|---|---|---|
| 125 | 126 | 0.431 | 16 | 0.107 px | 0.8901 | 0.8901 | tie |
| 129 | 122 | 0.371 | 12 | 52.6 px | 0.8974 | 1.0628 | default |
| 131 | 120 | 0.489 | 12 | 0.154 px | 0.5986 | 0.5987 | tie |
| 133 | 120 | 0.417 | 14 | 22.0 px | 0.6835 | 0.6988 | default |
| 134 | 120 | 0.382 | 15 | 26.4 px | 0.6641 | 0.6738 | default |

On all three large-disagreement points the SEEDED fit has the HIGHER
ZNSSD residual: the neighbour seed dragged the point into a WORSE local
optimum tens of pixels away. Seeded strictly better 0; default strictly
better 3; tie 2 (and over all 617 both-converged the median residual is
identical, 0.1141 either way, with seeded strictly lower on none). This
is ERROR PROPAGATION across the steep rim gradient, the risk plan 9.4
named, and it is un-gated: the cascade accepts any converged fit and
never compares its residual to the independent one, so plan 9.4's own
"residual-gated acceptance" is NOT in the implementation. The damage is
confined to already-marginal points (residual ~0.7 to 1.1, an order
above the good-field ~0.11 floor, right at the CCC-0.35 mask edge), so
on the trustworthy bulk of the field seeding and independent fitting
are float-noise identical, which is exactly what the frozen
`SEED_EQUIVALENCE_TOL` populations (the synthetic ramp and the rim/far
Si patches, worst 2.3e-13 px) measure. The frozen test is untouched and
still passes; this entry records the empirical caveat that D20.6's
budget-bounded equivalence, though it holds on well-conditioned points,
is NOT universal on real data: at steep-gradient rim points both paths
converge yet reach different optima, and the seeded one can be the
worse. A future gate that wants Stage D trustworthy on such points
should add the residual-acceptance gate before widening D20.6.

86. **Ship recommendation: a correctness lever for a narrow
basin-failure regime, not a speed lever, shipped default-off.**

Weighing all of the above:
- CORRECTNESS, proven and real, but narrow. The synthetic V8(a) rescue
  oracle (frozen, passing) proves the guarantee Stage D was
  commissioned for: where a large about-detector-normal rotation
  carries a point outside the ~2 deg phase-XC capture range of D5, the
  neighbour cascade reaches the correct optimum in ~5 iterations while
  the independent fit lands in a spurious basin ~50 px away even at 10x
  budget (ramp points 2.4 to 4.8 deg, ledger V8(b): default converges
  to 49.9 to 108 px of the imposed field at budget 2000, seeded to
  0.0096 to 0.0143 px in 5 iterations). No higher `max_iterations`
  recovers those; only a better seed does. This value is
  data-independent and genuine.
- On real Si-indent data that regime does not arise, phase-XC reaches
  every convergeable optimum at budget 500, so Stage D delivers zero
  measurable correctness benefit there (entry 85) and a small
  error-propagation RISK at the marginal rim (3 of 618 points to a
  worse optimum, 1 lost), because the residual-acceptance gate plan 9.4
  envisioned is not implemented.
- SPEED is negative on real Si: 0.56x on the anchor-bearing patch
  (entry 84), because the PASS1_CAP waste cancels the per-fit cascade
  saving on a field whose per-point cost is already only ~87
  iterations. The plan 9.4 3 to 8x projection does not hold on this
  data.

VERDICT: Stage D earns its place ONLY as a correctness lever for the
specific basin-failure regime (rotations beyond the phase-XC capture
range of D5), shipped correctly as `seed_from_neighbors=False` by
default so non-users pay nothing (the default path is bitwise
unchanged, pinned by `PRE_STAGE_D_*`). It is NOT a speed feature and
must not be sold as one; on smooth or marginal real fields it can only
match or slightly harm the independent fit at ~1.8x the wall time, so
it should be reached for only where large rotations actually defeat the
phase-XC seed. Two follow-ups are recorded, neither commissioned here:
add plan 9.4's residual-acceptance gate to the cascade (it would have
caught the three harmed points of entry 85), and, if Stage D speed on
real data is ever wanted, revisit PASS1_CAP downward (entry 83)
together with the cascade round-batching overhead (entry 84).


#### V8 recorded results, Stage D adversarial review fix gate (2026-09-09)

Fixer pass over the theory and conventions review. The applied
findings are documentation and coverage only: the stale PASS1_CAP
engine comment reconciled to entry 83's pinned state; the seeded-run
`num_iterations` last-phase semantics documented at the signal and
engine level; the engine-level `seed_round` Returns given its int32
dtype and 0 / r>=1 / -2 / -1 encoding; the signal-level "Seeding from a
neighbour" example reframed from the indent field (which entry 85 shows
does NOT benefit) to the measured basin-failure regime with the entry
85 error-propagation caveat; a fork-style CHANGELOG entry with honest
framing (a capture-range correctness lever, NOT a speedup, per entry
86); and the coverage gap of entry 88 closed. No engine constant, no
requirement, and no frozen test ASSERTION was changed; the new test of
entry 88 only ADDS coverage. Two measurements are re-recorded here.

87. **The Stage D local oldest-matrix run, the per-stage gate plan
    section 10 item 4 lists and the measurement-close section had left
    unrecorded (requirements D18; plan section 1 recipe).** Command
    (the constitution recipe, the four pytest plugins added because the
    `pyproject.toml` `addopts` reference them, exactly as Stage A entry
    24):

    ```
    uv run --isolated --python 3.10 \
      --with "numpy==1.23.0" --with "numba==0.57" \
      --with "orix==0.12.1" --with "scikit-image==0.21.0" \
      --with pytest-benchmark --with pytest-rerunfailures \
      --with pytest-xdist --with pytest-randomly \
      pytest tests/test_indexing tests/test_signals -k hrebsd -q
    ->  695 passed, 9 skipped, 4335 deselected in 126.75 s
    ```

    Environment as resolved: Python 3.10.19, numpy 1.23.0, scipy
    1.13.1, numba 0.57.0, orix 0.12.1, scikit-image 0.21.0, dask
    2024.8.1, kikuchipy 0.14.dev0 (BUILT from the branch source, so the
    edited engine and the new `TestVerboseProgress` test both ran).
    The 695 is the Stage-C figure 694 plus that one new test; the 9
    skips are the weekly `[download]` markers. The only orix API the
    Stage D tests add is `Rotation.from_axes_angles`
    (`test_hrebsd_seeding.py`), already exercised by the Stage A/B
    hrebsd files this same recipe covers, so the 0.12.1 floor was never
    at risk; the gap the review found was the missing RECORD, now
    closed. (A first attempt WITHOUT the four plugins failed at
    conftest import because the `--benchmark-skip` addopt is unknown
    without `pytest-benchmark`; the recorded recipe carries them for
    exactly this reason.)

88. **Coverage of the touched `_hrebsd` modules, re-closed to 100 %
    after the two Stage D verbose progress-print bodies were found
    uncovered.** The implementation gate left `_engine.py` at 99.44 %
    (2 of 355 statements: the `Cascade round` line and the `Rescue
    pass` line, both behind `if verbose >= 1`, which every seeded
    fixture ran at `verbose=0` through `run_map`). `TestVerboseProgress`
    now runs one seeded ramp map at `verbose=1`; the ramp walks the
    cascade one point per round and the isolated slow point goes to the
    rescue pass, so both lines execute, and the test asserts only that
    the two messages appear (the counts they carry are pinned by other
    fixtures). Re-measured on the venv Python over `tests/test_indexing
    tests/test_signals -k hrebsd`: `_engine.py` 355 statements, 0
    missed, 100.00 %; every other `_hrebsd` module 100.00 %; the
    package total 1480 / 1480. `694 passed` becomes `695 passed, 9
    weekly-skipped`.


### V9 -- GPU backend oracles (Stage E, 2026-10-06)

Requirements D21 govern. One new file,
`tests/test_indexing/test_hrebsd_gpu.py` (D21.14), plus the three
edits of existing pins that D15.4 and D21.13 name
(`FROZEN_SIGNATURE`, `test_run_defaults_are_frozen`,
`TestImportAudit`). The CPU path is the oracle (D21.8). Two suites:
DEFAULT (the numpy namespace and a faked cupy; runs everywhere, CI
included) and GATED (the `cupy_gpu` fixture; local, `-n 0`, through
the pinned overlay of D21.15). CI green is zero evidence for the
GPU path. Every band below is MTP: the failing-tests commit carries
placeholders marked `FIXME-pin` (the Phase 12 convention); the
CPU-side literals are measured at the failing-tests gate, and the
device bands and device-side literals at the implementation gate
(no device implementation exists before it; corrected 2026-10-06,
spec review), each pinned at about 2x margin with the recipe and
machine ID in the comment beside the constant. The spec-gate
numbers quoted (ledger 93 to 95) are the scales a correct
implementation should reproduce, not pins. Bands measured under the
numpy namespace are never reused for device asserts, and neither
are the CPU's pinned bands. The numpy twin reproduces the
`"float64"` build only (D21.14.3), so every numpy-twin arm runs at
float64 and the mixed build is asserted in the gated suite alone.

Fixtures, all synthetic or shipped and generated in the test with
fixed seeds: **F1** the V2 480 px twelve-case batch (seeds 0 and 1;
the fixture `WARP_REFIT_TOL_480` was measured on); **F2** a
64-pattern 480 px batch (the prototype's "480 seed 0" and "seed 1"
sets; gated only); **F3** a 64-pattern 512x622 rectangular synthetic
(the real detector's shape, whose 622 = 2 * 311 is an awkward FFT
length; gated only) and **F3s** its first 8 patterns (the default
suite's 512x622 slice, for CI cost); **F4** the Stage A 60x60 route
(`nickel_ebsd_small` reference, warped batch); **F5** a two-grain
map carrying a masked point, an INTEGER-constant target, a target
with a non-finite pixel and a cap-hitting target, run at
`filter_cutoffs=(None, None)` (the production Si route), with a
band-passed-constant arm at the default cutoffs; **F6** the
four-point pre-Stage-D pin map (`test_hrebsd_seeding.py:342-412`);
**F7** a synthetic many-grain map (tens of grains of a few points
each, 60x60 patterns) for the residency bound; and the shipped Ni
map (`nickel_ebsd_small`). The default suite uses F1, F3s, F4, F5,
F6, F7 and the Ni map only; F2 and F3 are gated. The default-suite
wall time is recorded at the failing-tests gate (D21.14.3). The
Si-indent file is never a test fixture (no test reads it; it is
local and 18.9 GB): its parity and timing are RECORDED
measurements, (q) below.

(a) **Backend switch, check order and the docstring** [default]
(D21.1, D21.17). `backend="cpu"` equals the call without the keyword
bitwise on F6 and on the shipped Ni map; the pre-Stage-D literal
pins of both files re-run unmodified; `ValueError` with the D21.1
message for `"GPU"`, `"cuda"` and `""`; a spy on the gate records
zero calls under `"cpu"`; under `"gpu"`, with the gate faked to
pass and a numpy session, call-order spies show the existing
argument checks first, then the backend checks in the frozen D21.1
order, the gate before `resolve_reference` and before the first
`ReferenceState`; `coefficient_dtype=np.float64` under `"gpu"`
raises `ValueError` before the gate. The docstring test pins the
`backend` parameter entry, the optional-dependency sentence, the
three Raises entries (`MemoryError`, `NotImplementedError`,
`ValueError`) and the Notes topics of D21.17, with no stage letter.
Expectation: the frozen literals. Band: bitwise.

(b) **The Stage D raise** [default] (D21.12). `backend="gpu",
seed_from_neighbors=True` raises `NotImplementedError` whose message
equals the D21.12 literal (asserted in full, so `backend='cpu'` and
`Fourier-Mellin` are pinned as written); with the gate patched to
raise `AssertionError` and spies on `resolve_reference`,
`ReferenceState` and the pattern gather, every spy records zero
calls, so the raise precedes any device or reference work; on a
machine without cupy no `ImportError` surfaces first. The string
check still wins: an invalid `backend` with
`seed_from_neighbors=True` raises `ValueError`.
Amended 2026-10-07 (Stage F, requirements D22.11 and the D21.12
amendment): the literal asserted in full is the amended one, which
names `seed_from_neighbors=False` with `fourier_mellin='auto'` as the
GPU remedy; `backend='cpu'` stays pinned as written. With
`fourier_mellin` other than `"off"`, the D22.11 `ValueError` fires
before this raise (V10(a)).

(c) **Availability gate** [default with the fake cupy; gated with
the real one] (D21.2). The three stages and the shim run in the
frozen order; each failure raises its HREBSD-worded message with
its remedy (the prefix pinned literally); a cached failure
re-raises a fresh copy chained to the cached one; stage (c) issues
one complex128 FFT, one complex64 FFT, one matmul and one RawKernel
compile and launch (the fake records the calls, so an FFT-only
probe fails the test). Identity pins:
`_hrebsd._gpu._add_nvidia_dll_directories`, `_CUPY_MINIMUM_VERSION`
and `_DEVICE_LOCK` ARE the `_spherical._gpu` objects, and an HREBSD
gate run leaves the spherical gate's cache untouched (and the
reverse). Gated: the real gate passes on the overlay and caches.

(d) **Seed seam contract** [default, numpy namespace; gated for the
device contract and parity] (D21.5). `seed_spectra` returns `(P, sr,
sc)` at the seed precision; `seed_homographies` returns a `(P, 8)`
float64 C-contiguous array of `ctx.xp` whose entries 0, 1, 3, 4, 6
and 7 are exactly zero; under numpy the rows equal the CPU
`initial_guess` on F1, F3s, F4, F6 and the shipped Ni map
(`NUMPY_SEED_EQUAL_COUNT`, measured then pinned, expected value
"all equal"; the Stage F gate for routing the CPU through the seam,
plan 11.3), at `upsample_factor` 16, 2 (a 3x3 upsampled region) and
1 (the coarse branch, no rounding), and at 0 every row is
non-finite (the CPU fails every pattern there, D21.5). A dimmed arm
scales the preprocessed target and reference by 1e-10, below which
an un-normalised cross-power spectrum meets the `100*eps` floor:
the rows still equal `initial_guess` (the designed killer of a
crop that is not ZMN'd, M15). A spy shows the runner calling
`_batched.seed_homographies` (and `seed_spectra`) through the module
global ONCE PER PREPROCESSING SUB-BATCH, ceil(B/P) times per batch
with P rows each, the tail sub-batch of a batch whose B is not a
multiple of P padded to P; the `SeedBatch` it receives carries
`pattern_index` equal to the flat map indices of the slots in fit
order and -1 on every padded slot, `coefficients` of the same
targets, and empty `extras`; the `SeedContext` carries the
session's namespace and `KernelNamespace`.
`SeedState.reference_spectrum` is complex128 by default and
complex64 under `seed_precision="complex64"` (dtype asserts, the
Phase 12 M10 lesson); a planted non-finite row gives the D2.6
failure contract to that pattern only. Gated, the device contract
(Johan decision 1's shape and dtype test on the device itself):
`h0` is a `cupy.ndarray`, float64, C-contiguous, shape `(P, 8)`;
`seed_spectra` returns a `cupy.ndarray` `(P, sr, sc)` at the
requested complex dtype. Gated parity: complex128 device seeds
equal `initial_guess` row for row on F1 to F4
(`GPU_SEED_EQUAL_COUNT`, pinned at the measured count; spec gate
256 of 256, ledger 94) and on the `upsample_factor` 2 and 1 arms;
complex64 differences counted (`GPU_SEED_C64_DIFF_COUNT`, MTP;
spec gate 0 or 1 per 64-pattern set, each off by 1/16 px).

(e) **The seed-seam h0 oracle** [default numpy at float64 on F1
and F3s; gated on F1 to F3 at both device precisions] (D21.5(iii),
Johan decision 1). The seed stage is monkeypatched to return
ARBITRARY per-pattern rows, each chosen INSIDE the fixture's
measured capture range (corrected 2026-10-06, spec review: the
draft planted the identity, a pure 1.0 deg rotation and an
unbounded h31/h32 row and then asserted an h band and EQUAL counts
on every fit, which a correct implementation can fail on a start
that does not converge; V2's `test_seed_required_for_large_
translation` shows identity seeding fails at large translation):
the exact imposed homography; the imposed homography composed with
a 0.5 px translation and a 0.5 deg in-plane rotation; the imposed
homography composed with a 1.5 deg in-plane rotation (inside the
~2 deg capture range of D5); the imposed homography with `h31`
and `h32` perturbed by a small measured amount; and a NaN row.
Expectation, per pattern: the CPU `fit_pattern(state, target,
h0=row)`. Bands, per planted-row type: the (f) `h` band on points
BOTH backends converge; `num_iterations` and `converged`
differences as COUNT budgets, measured then pinned
(`GPU_SEAM_ITERATION_DIFF_COUNT`, `GPU_SEAM_CONVERGED_FLIP_COUNT`;
the (g) discipline); a pinned MINIMUM both-converged count per row
type (`GPU_SEAM_BOTH_CONVERGED_MIN`), so the oracle cannot go
vacuous; and the NaN row gives the failure contract on both
backends, exactly. Two arms make the oracle discriminating, because
a converged `h` cannot tell a honoured row from an ignored one when
both reach the same optimum: (1) a ONE-ITERATION arm
(`max_iterations=1`) for every finite row type, the returned `h`
against the CPU's within the (f) first-step band -- the iterate
after one step is `h0` composed with one update, so a device that
recomputes or ignores the seam's rows fails at once; (2) per row
type, the number of patterns whose CPU iteration count from the
planted row differs from the CPU count from the translation seed
is pinned at a minimum (`SEAM_DISCRIMINATING_MIN`), and the device
reproduces the planted-row counts within the budget. This is the
oracle that makes the seam real.

(f) **Per-point parity bands** [gated; numpy-namespace twins at
float64 with their own bands on F1, F3s and F4] (D21.8(c), (d)). F1
to F4, each device precision, each seed precision where the seed
matters. `GPU_PARITY_H_TOL_MIXED` (corner displacement, px; spec
gate at most 6.6e-7 on the synthetic sets and 1.3e-6 on the Si rim
set) and `GPU_PARITY_H_TOL_F64` (spec gate at most 2.1e-12);
`GPU_FIRST_STEP_TOL_MIXED` on the iteration-1 increment (spec gate
1.0e-7 px at 480; pure-f32 reductions measured 5.3e-7, so that
mutant dies here; the float64 twin measured 1.4e-13);
`GPU_PARITY_RESIDUAL_RTOL` (spec gate at most 2.7e-7 relative) with
`GPU_PARITY_RESIDUAL_ATOL` for near-zero criteria;
`GPU_PARITY_FE_TOL` on Fe through the shared host conversion. The
per-kernel A/B arm [gated] (D21.14.3): each `KernelNamespace` entry
point (`gather`, `pixel_sums`, `reduce_solve_update`,
`final_criterion`) run under the numpy twin and under cupy at
float64 on identical inputs, within `GPU_KERNEL_AB_TOL_F64` (MTP).

(g) **Iteration and convergence budgets** [gated] (D21.8(e)).
`GPU_ITERATION_DIFF_COUNT` and `GPU_CONVERGED_FLIP_COUNT` per
fixture, pinned at the measured COUNT (spec gate: 0 differences in
320 compared patterns); no fractional assert at these N. The
real-data rates are ledger records, (q).

(h) **Batched semantics and every knob** [default numpy at float64;
gated at both device precisions] (D21.6). On F5 plus an easy
pattern (at most 10 iterations): every pattern's result equals the
same pattern fitted ALONE (a batch of one, padded to the same B)
bitwise, so neither the active set nor the batch's other patterns
touch a survivor's arithmetic; the same holds for a slot in the
LAST sub-batch of a batch whose B is not a multiple of P (B = 40,
P = 32, F1 repeated to fill it; D21.7.3); `num_iterations` is each
pattern's own count; `residual` is the criterion at the returned `h`
(the `TestResidualIsTheFinalCriterion` analogue, on the cap-hitting
pattern, where the two differ by 26 per cent on the CPU). Knob arms,
each against the CPU `fit_pattern` within (f)'s bands on F1 or F4
(corrected 2026-10-06, spec review: the draft had no arm for four
of the D21.6 knobs): `step_scale` 0.5 and 1.5; `window=True` (the
D21.6.5 algebra); a `dead_band`; `max_iterations` 0 and 1;
`filter_cutoffs` `(None, None)` -- the production Si route, where
the device SKIPS the band-pass, with a target carrying a large DC
offset so a moment reconstructed against the wrong shift shows --
and `(0.05, 0.4)` (a low-pass cutoff); `upsample_factor` 8;
`min_step` 1e-2; `border` 0.1. Padded slots never reach the output
(sentinel-filled padding).

(i) **Failure contract** [default numpy; gated] (D21.6.1-3). On F5
at `filter_cutoffs=(None, None)`: the masked point gives NaN props
with truthful `grain_id` and `reference_index`; the integer-constant
target gives the D2.6 failure contract on both backends through the
SEED's crop ZMN (the CPU norm is exactly 0 there, measured
2026-10-06, `initial_guess` raising); a target with a non-finite
pixel gives it too; the cap-hitting target keeps its last iterate
with `converged=False` and NaN Fe; the `UserWarning` count equals
the CPU's. The band-passed-constant arm (default cutoffs; the CPU
fails it only in the criterion after the f32 cast, D21.6.2)
asserts `converged=False` on both backends and nothing more.
Kernel level, on the numpy twin and on the device kernel: a planted
non-finite coordinate is flagged before the mirror fold; planted
finite coordinates of +-3e9, +-1e30 and FLT_MAX fold to the CPU's
folded value with no device error (D21.6.3); a planted NaN corner
displacement gives a NaN `norm_dp` and never `converged` (D21.6.1).

(j) **Intensity scale** [gated] (D21.8(f)). Power-of-two rescales
bitwise on the device at both precisions; the generic factor at
`GPU_INTENSITY_SCALE_GENERIC_TOL_MIXED` and `_F64` (spec gate
3.2e-8 and 3.6e-9 px).

(k) **Update-rule analogue** [gated] (D21.8(g)). The device against
the hand-built numpy loop of the `UPDATE_RULE_TOL` test in
`test_hrebsd_engine.py`: `GPU_UPDATE_RULE_TOL_F64` (spec gate
1.4e-13 to 1.6e-11 px) and `GPU_UPDATE_RULE_TOL_MIXED` (spec gate
9.2e-7 px, so about 2e-6, 40x under the 8.2e-5 px tightest mutant
separation). This is the ONLY killer of the wrong-side composition
`W(dp)^-1 . W` on the device, as on the CPU
(`test_hrebsd_engine.py:419-424`: both arms share the fixed point,
so no accuracy band separates them).

(l) **Drift tripwire** [gated; the CPU half default] (D21.8(h)). On
F1, per case, the CPU's and the device's recovery errors against
the EXACT imposed homographies are pinned to dated literals within
`GPU_DRIFT_TRIPWIRE_PX` (MTP; the CPU literals at the failing-tests
gate, the device literals at the implementation gate; spec gate: on
the prototype's 64-pattern 480 set the mixed recovery maximum moved
by 3.4e-7 px from the CPU's 0.02251 px). The CPU half runs in the
default suite, so a shared-code edit is caught even where no GPU
exists.

(m) **Determinism and laziness** [gated; the layout pin, the source
pin and the laziness oracle default] (D21.7, D21.11). Two runs
bitwise at a fixed B, both device precisions, both seed precisions;
B invariance at B in {8, 32, 40, default} measured, then pinned
bitwise or at the measured tolerance (`GPU_BATCH_INVARIANCE_TOL`,
MTP, possibly 0; expected bitwise among the B >= 32 by
construction, D21.7.3); lazy equals eager bitwise; a 4-dask-worker
lock stress run equals the 1-worker run bitwise. The layout pin
[default]: `_launch_layout` returns the same block size and blocks
per pattern for every B at a given subregion pixel count (a
pure-function test); and [gated] a spy on the launch calls shows
the same per-pattern grid and block dimensions at B = 8 and B = 32
(the call site, which the pure function cannot see). The source
pin [default]: no `atomicAdd` anywhere in the CUDA source. The
laziness oracle [default, numpy session]: with a dask callback
recording every computed block of the pattern array, no computed
block holds more than B patterns, and the runner never receives an
array of the whole map.

(n) **Robustness** [gated; numpy-session twins default] (D21.9,
D21.10). An out-of-memory at session build halves B (spy); a
mid-compute one rebuilds at B/2 and re-runs (`built == [8, 4]`);
the B = 1 floor raises `MemoryError` with the D21.10 text in both
windows; a run under a real pool limit (`set_limit`) leaves at
most a pinned residue after recovery (leak pin), and the recovery
run equals an unlimited run at the same final B bitwise; on F7, a
spy on resident uploads and evictions shows at most `R_MAX`
references resident at any time and every grain uploaded once, and
[gated] the F7 run under `set_limit` completes and leaves no
residue beyond the leak pin; a planted device-stage exception fails
the run and yields no NaN rows (D21.9.4); a spy on the compute's
`scheduler` keyword reads `"threads"`; no device attribute survives
on any returned object; the session's dask token is the D21.9.1
tuple.

(o) **VRAM model, batch size and information message** [default;
calibration gated] (D21.10). The pure-math model's three terms;
`_default_batch_size` returns the largest B in {64, ..., 1} whose
whole model fits half of free VRAM (free VRAM faked: a value where
the per-slot term alone would allow 64 but the P and R terms do
not); an explicit `chunksize` wins; `ValueError` for a `chunksize`
below 1 under `"gpu"` (the CPU path's clamp unchanged); B is never
clamped to the number of fitted points (a three-point map runs one
padded batch at the default B); the information message carries
the device block with B and P, and the warning line when the model
exceeds free VRAM. Gated calibration of g, p and r SEPARATELY
against pool high-water marks (spec gate: r = 14.2 MB mixed and
19.4 MB float64 at 512x622, 10.3 and 14.0 MB at 480; g dominated by
the 1.27 MB coefficient plane; p about 29 MB per pattern at
512x622; ledger 92), pinned with the measured margin.

(p) **Gating, canary, import hygiene** [default] (D21.13, D21.14).
The skip-order pins with the probe forbidden from running; the
`KIKUCHIPY_EXPECT_GPU` canary; no module-scope cupy (the AST arm
of `TestImportAudit`, a source regex over `_hrebsd/*.py`, and the
subprocess `import kikuchipy` check); the audit's allowed tuple
gains exactly `cupy` and `gc` (no `typing`, no `__future__`); an
AST check that `_gpu.py` and `_batched.py` import `_engine` at no
module scope (D21.9.5).

(q) **Real-data parity and the performance record** [local,
RECORDED in the ledger, never a test] (D21.8(e), D21.11, D21.16).
On the Si-indent file (read only, never copied), with the executed
tutorial's load recipe: per-point GPU against CPU on the far-field
patch (256 points) and on patch C (618 points; the CPU run is
338.3 s) at both seed precisions -- the count and the largest size
of iteration differences, convergence flips, the h band on
both-converged points, the residual band -- then the D21.16 timing
rows at both device precisions, the E8 device-wait fraction of the
v1 read route, and the go/no-go floor. The whole map is timed on the
GPU only and compared by aggregate counts against ledger 82: no
per-point CPU map exists on disk and a re-run costs 2.34 h (plan
open question E12).

V9 requirement-to-oracle map:

| D21 item | oracle(s) | suite |
|---|---|---|
| D21.1 API, default path, check order | (a) | default |
| D21.2 gate | (c) | default + gated |
| D21.3 device scope | (a), (f) | default + gated |
| D21.4 precisions | (d) dtypes, (f), (j), (k) | gated + numpy twins (float64) |
| D21.5 seed seam | (d), (e) | default + gated |
| D21.6 batched semantics, knobs, flags | (h), (i) | default + gated |
| D21.7 determinism | (m) | gated (layout and source pins default) |
| D21.8 parity, tripwire | (f), (g), (j), (k), (l), (q) | gated + ledger |
| D21.9 topology, names, residency | (n), (p) | default + gated |
| D21.10 VRAM, batch size, out of memory | (n), (o) | default + gated |
| D21.11 lazy streaming | (m) lazy equals eager, laziness oracle; (q) E8 | default + gated + ledger |
| D21.12 Stage D raise | (b) | default |
| D21.13 optional cupy | (p) | default |
| D21.14 gating, coverage | (p), the recorded coverage command | default + gated |
| D21.16 performance | (q) | ledger |
| D21.17 docs | (a) docstring test | default |

Gate commands, recorded verbatim with their output at each Stage E
gate (Git Bash, from the worktree); the overlay is PINNED to the
D21.15 versions (an unpinned overlay can move device-side bits on a
later day):

```
# default suite (cupy absent: the gated classes skip at stage (a))
uv run pytest tests/test_indexing tests/test_signals -k hrebsd -n 0
uv run pytest tests/test_indexing tests/test_signals -k hrebsd -n 4
# gated suite (pinned overlay; -n 0 by construction; canary armed)
OVERLAY="--with cupy-cuda12x==14.2.0 \
  --with nvidia-cufft-cu12==11.4.1.4 \
  --with nvidia-cublas-cu12==12.9.2.10 \
  --with nvidia-cusolver-cu12==11.7.5.82 \
  --with nvidia-cusparse-cu12==12.5.10.65 \
  --with nvidia-nvjitlink-cu12==12.9.86"
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY \
  pytest tests/test_indexing/test_hrebsd_gpu.py -n 0 -q --weekly
# coverage: the default run, then the gated run appended
uv run pytest tests/test_indexing tests/test_signals -k hrebsd \
  -n 0 --cov=kikuchipy.indexing._hrebsd --cov-report=
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY \
  pytest tests/test_indexing/test_hrebsd_gpu.py -n 0 --weekly \
  --cov=kikuchipy.indexing._hrebsd --cov-append \
  --cov-report=term-missing
```

#### V9 recorded results, spec gate (2026-10-06)

Measured on machine A by the Stage E spec-phase exploration: a
read-only engine-anatomy probe, then a THROWAWAY GPU prototype.
Neither edited a repository file, and nothing of the prototype is
committed. Scripts in the session scratchpad: `gpu_map/
profile_stages.py`, and `hrebsd_gpu_proto/` (`common.py`,
`cpu_baseline.py`, `cpu_api.py`, `cpu_api_blas1.py`, `kernels.cu`,
`gpu_icgn.py`, `kernels_tiled.cu`, `gpu_tiled.py`, `smoke.py`,
`gpu_timing.py`, `gpu_kernel_opt.py`, `gpu_tiled_timing.py`,
`gpu_accuracy.py`, `gpu_det_seeds.py`, `gpu_e2e.py`,
`gpu_fullmap.py`, `compare_maps.py`, `io_ceiling.py`, each with a
`.log` and a `.json`; result arrays `fullmap_c16.npz`,
`fullmap_c8.npz`, `e2e_*.npz`). The scratchpad does not survive
the session, so every number is carried here. These are PROTOTYPE
numbers: the Stage E gates re-measure every one that feeds a pin
or a decision on the real implementation.

89. **Conditions of every entry below (requirements D21.15).**
    Machine A: Intel i7-13700H (Raptor Lake, 14 cores, 20 logical),
    32 GB, Windows 11 build 26200; GPU NVIDIA RTX 2000 Ada
    Generation Laptop GPU, 8 GB (8188 MiB), 24 SMs, 32 MB L2,
    128-bit memory bus. CPU runs: the worktree `.venv` (CPython
    3.13.12, numpy 2.4.6, scipy 1.17.1, numba 0.65.1, scikit-image
    0.26.0, dask 2026.3.0, OpenBLAS 0.3.31 at 20 threads unless
    stated). GPU runs: CuPy 14.2.0 from the uv overlay, never
    installed into the `.venv`:

    ```
    uv run --with cupy-cuda12x --with nvidia-cufft-cu12 \
      --with nvidia-cublas-cu12 --with nvidia-cusolver-cu12 \
      --with nvidia-cusparse-cu12 --with nvidia-nvjitlink-cu12 \
      python <script.py>
    # after kikuchipy.indexing._spherical._gpu.
    #   _add_nvidia_dll_directories() and before "import cupy"
    ```

    Verified on the day: about 6.6 GiB of VRAM free, an NVRTC
    `RawKernel` compile, cuFFT `rfft2` plus `irfft2` of 32 patterns
    at 512x622 in 3.9 ms, a batched 8x8 solve. WHICH DLL SUPPLIED
    NVRTC (a pip wheel, or a CUDA Toolkit on PATH) was not recorded
    (plan open question E6). Added at the spec review (2026-10-06),
    the overlay's resolved versions read through
    `importlib.metadata` inside the overlay (no cupy import, no GPU
    work): `cupy-cuda12x` 14.2.0, `nvidia-cufft-cu12` 11.4.1.4,
    `nvidia-cublas-cu12` 12.9.2.10, `nvidia-cusolver-cu12`
    11.7.5.82, `nvidia-cusparse-cu12` 12.5.10.65,
    `nvidia-nvjitlink-cu12` 12.9.86, and `nvidia-cuda-nvrtc-cu12`
    12.9.86 resolved TRANSITIVELY (so an NVRTC wheel was present in
    the overlay, whichever DLL the process loaded); NVIDIA driver
    595.71 (`nvidia-smi`). These are the versions D21.15 pins; the
    prototype ran on the same overlay the same day, but its own
    resolution was not recorded at the time. `nvidia-smi` was read
    before every GPU timing: 0 per cent use and no compute apps at
    every check;
    during runs only the measuring process was listed, so no
    foreign GPU process was ever seen and no timing needed
    repeating. Laptop boost clocks moved during runs: SM 1.4-2.5
    GHz, memory 6.0-7.8 GHz, P1-P4. CPU load from other sessions
    was 1-33 per cent in total (other python processes at 40-100
    per cent of one core, VS Code, Chrome). The commit limit was
    tight (8.9 GB of free virtual memory): an 8-process I/O test
    hit `MemoryError` and its numbers are discarded. Data:
    `C:\Users\westraadt.1\Repos\kikuchipy\AGH__Si_indent_1_512x672.h5oina`,
    READ ONLY, never copied; load recipe of the executed
    `hrebsd_si_indent.ipynb` (reference (10, 10),
    `filter_cutoffs=(None, None)`, crater mask CCC < 0.35, budget
    500).

90. **The CPU side, two probes and the public API (requirements
    D21.3, D21.16).** (i) `profile_stages.py`: OMP, OpenBLAS and
    MKL threads at 1 (what one dask worker sees), a synthetic
    smoothed-noise pattern with bands and a sub-pixel shifted
    target, default knobs, f32 coefficients, warm numba cache,
    median of 15 runs (5 for whole fits), CPU load 3 per cent at
    the start:

    | stage, single thread | 512x622 (257600 px) | 480x480 (186624 px) |
    |---|---|---|
    | `ReferenceState`, once per grain | 107 ms | 53 ms |
    | `preprocess` (band-pass) | 35.5 ms | 10.9 ms |
    | `initial_guess` (phase XC, upsample 16) | 34.5 ms | 23.0 ms |
    | `spline_coefficients` | 9.9 ms | 4.1 ms |
    | per iteration: warp coordinates | 8.3 ms | 6.5 ms |
    | per iteration: bicubic evaluate | 12.2 ms | 8.7 ms |
    | per iteration: ZMN | 1.5 ms | 1.1 ms |
    | per iteration: residual + CIC | 1.1 ms | 0.8 ms |
    | per iteration: `SD^T r` | 0.9 ms | 0.6 ms |
    | per iteration: solve, corner norm, update | 0.1 ms | 0.1 ms |
    | whole fit at 1 / 14 / 50 iterations | 127 / 449 / 1344 ms | 73 / 312 / 954 ms |

    At the map median of 14 iterations (449 ms at 512x622): fixed
    per-pattern cost 18 per cent (band-pass 8, seed 8, spline 2),
    bicubic 41, coordinate warp 28, ZMN plus residual plus CIC 9,
    gradient 3, solve and update under 1; marginal cost 24.8 ms
    per iteration. The 622 = 2 * 311 FFT length is why 512x622
    costs 3.3x the 480 px band-pass for 1.38x the pixels. Inference,
    not a measurement: Stage A's 22.72 patterns/s at 480 px on 8
    workers is about 45 per cent of 8x the single-thread rate, so
    GPU speedups are quoted against 8-worker baselines, never
    projected from single-thread numbers.
    (ii) `cpu_baseline.py`: one worker, OpenBLAS at 1 thread;
    per-iteration cost is the slope between 1 and 21 forced
    iterations on 8 patterns (median of 3), components medians of
    5:

    | size | ms/iteration | fixed ms (band-pass + spline + final criterion) | ms per converged pattern |
    |---|---|---|---|
    | 60x60 (2916 px) | 0.348 | 0.56 | 2.60 (3.69 it, 16/16) |
    | 480x480 (186624 px) | 18.83 | 39.1 | 149.1 (4.81 it, 64/64) |
    | 512x622 synthetic (257600 px) | 27.10 | 68.8 | 239.2 (5.08 it, 64/64) |
    | Si far, rows 40:48 cols 30:38 | -- | -- | 306 (9-10 it, 16/16) = 3.27 pat/s |
    | Si rim, rows 125:133 cols 115:123 | 27.1 | 93 | 5509 (83-500 it, 7/8) |

    Components at 512x622: band-pass 35.2, spline 10.0, seed 35.2
    ms (37.0 on Si); per iteration numpy warp coordinates 8.5,
    numba bicubic 12.2, ZMN 1.5, `SD^T r` 0.9-1.9, solve and update
    0.13 ms. The two probes' per-iteration figures (24.8 against
    27.10 ms) come from different recipes -- a component sum on one
    synthetic pair against a slope over 8 patterns -- and both are
    recorded. (iii) `cpu_api.py`, the public method on 8 dask
    workers, 64 patterns: 480 px synthetic best 15.3, median 13.8
    patterns/s (4.77 iterations; Stage A's record on this machine
    is 23.96); an 8x8 Si far-field patch 7.5 patterns/s (9.2
    iterations); an 8x8 Si rim patch 0.77 patterns/s (mean 176,
    median 128, max 500 iterations, 63 of 64 converged).
    (iv) `cpu_api_blas1.py`, OpenBLAS at 1 thread: 480 px best 22.3
    patterns/s (22.7 at 16 workers; the default-thread re-run best
    16.7); Si far 9.6 and 8.3 patterns/s. The default 20-thread
    OpenBLAS appears to oversubscribe the 8 dask workers; this is
    indicative only, the runs were noisy (a CPU-side note, out of
    Stage E scope, D21.18). (v) The calibrated 8-worker map model:
    ledger 82's 8424 s (2.34 h) against the GPU-measured map total
    of 1110802 iterations over 57772 points fits about 17 ms fixed
    plus 6.7 ms per iteration per pattern, an effective 4.05x over
    one worker.

91. **Device per-iteration cost by precision, measured both ways
    (requirements D21.4, D21.10; the D17 amendment's evidence).**
    `gpu_timing.py`, the baseline prototype build, 20 forced
    iterations, median of 3; each cell is ms per iteration / us
    per pattern-iteration at 512x622:

    | precision | B=1 | 8 | 32 | 64 | 128 |
    |---|---|---|---|---|---|
    | float64 | 0.462 / 462 | 3.36 / 420 | 14.1 / 440 | 27.2 / 424 | 51.9 / 405 |
    | mixed (f32 + f64 reductions) | 0.0715 / 71.5 | 0.186 / 23.3 | 1.06 / 33.3 | 2.59 / 40.4 | 5.38 / 42.0 |
    | mixed, f64 per-pixel accumulation | 0.191 / 191 | 0.889 / 111 | 2.39 / 74.8 | 4.84 / 75.6 | 10.2 / 79.8 |
    | pure f32 | 0.0744 / 74.4 | 0.228 / 28.5 | 1.13 / 35.4 | 2.53 / 39.5 | 4.85 / 37.9 |

    At 480x480, us per pattern-iteration at B = 1 / 8 / 32 / 64 /
    128: mixed 60.7 / 18.4 / 15.3 / 23.6 / 28.5, f32 66.0 / 20.6 /
    16.3 / 28.8 / 28.0, float64 about 300-340. The faster build
    (`gpu_kernel_opt.py`, "recip6": the eight IEEE f32 divisions by
    6 in the B-spline weights replaced by a multiplication) at
    512x622: mixed **22.7 us** at B=64, 23.0 at B=8, 25.5 at B=128;
    f32 22.4 at B=64; float64 **270 us** (11.9x mixed); at 480 px
    mixed 15.6 and f32 14.6 us. `--use_fast_math` gives the same
    speed-up; accuracy unchanged either way. Against one CPU worker
    that is 27.10 ms / 22.7 us = about 1190x per iteration, and
    about 295x against the map-effective 8-worker 6.7 ms. Further:
    a blocks-per-pattern sweep at B=64 is flat (2.03-2.30 ms per
    iteration); deterministic and atomic reductions cost the same
    within 3 per cent; with 128 patterns resident, an iteration
    costs 4.11 ms with 128 active, 2.08 with 64, 0.53 with 16 and
    0.197 with 1 (so retired slots cost little); a multi-pattern
    tiled kernel (`kernels_tiled.cu`, 1-8 patterns per block, 480
    px only) gained nothing, the 7.2 MB of shared reference arrays
    already sitting in L2 (its 512x622 run ran out of memory in
    that setup).

92. **Fixed device costs and VRAM (requirements D21.3, D21.10).**
    512x622, B=64, ms per pattern (480 px in brackets):

    | step | ms per pattern |
    |---|---|
    | upload uint8, pageable | 0.063 (0.038) |
    | upload uint8, pinned | 0.029 |
    | upload f64 | 0.44 |
    | band-pass, complex128 | 1.79 (0.66) |
    | band-pass, rfft2 f64 | (0.39; matches fft2 to 8.5e-14) |
    | band-pass, rfft2 f32 | (0.072) |
    | no filter (the Si route) | 0.013 |
    | spline prefilter (f64, cast to f32) | 0.24 |
    | phase-XC seed, complex128 | 2.16 (1.26) |
    | phase-XC seed, complex64 | 0.52 |
    | K shift, boolean mask mean | 0.097 |
    | K shift, box mean | (0.009) |
    | final two-pass criterion | 0.10 |

    VRAM: per reference 14.2 MB mixed and 19.4 MB float64 (10.3
    and 14.0 MB at 480), including the 4.1 MB complex128 seed
    spectrum; per resident pattern one f32 coefficient plane, 1.27
    MB (0.92 MB at 480); preprocessing transient about 29 MB per
    pattern (a 64-pattern one-shot preparation peaked at a 1.87 GB
    pool, a 256-pattern one ran out of memory, so preparation runs
    in sub-batches of 32 or fewer); whole-map runs by `nvidia-smi`
    1501-2475 MiB with the complex128 seed and 995 MiB with
    complex64.

93. **Accuracy against the CPU on identical inputs, by precision
    (requirements D21.4, D21.8; the D17 amendment's evidence).**
    `gpu_accuracy.py`, re-measured fields in `gpu_det_seeds.py`
    (entry 99). The GPU starts from the CPU's seeds; an
    instrumented CPU copy reproduces `fit_pattern` bitwise; the
    metric is the V2 corner-displacement error warp, px:

    | set (n = 64) | precision | dp error, all its. / it. 1 | final h max (median) | worst rel. component | its. agree | conv. agree | V2 recovery max, GPU / CPU |
    |---|---|---|---|---|---|---|---|
    | 480 seed 0 | float64 | 2.8e-13 / 1.4e-13 | 2.0e-13 | 2.2e-13 | 64/64 | 64/64 | 0.02251 / 0.02251 |
    | 480 seed 0 | mixed | 7.9e-7 / 1.0e-7 | 6.6e-7 (1.5e-7) | 7.9e-8 | 64/64 | 64/64 | 0.02251 (moves 3.4e-7) |
    | 480 seed 0 | f32 | 7.0e-7 / 5.3e-7 | 4.2e-7 | 8.0e-8 | 64/64 | 64/64 | 0.02251 |
    | 480 seed 1 | mixed / f32 | 6.8e-7 / 6.9e-7 | 3.5e-7 / 3.7e-7 | 9.6e-8 | 64/64 | 64/64 | 0.01817 / 0.01817 |
    | 512x622 seed 0 | float64 | 1.6e-11 | 2.1e-12 | 1.2e-12 | 64/64 | 64/64 | 0.02112 |
    | 512x622 seed 0 | mixed / f32 | 8.8e-7 / 8.9e-7 | 5.5e-7 / 5.6e-7 | 1.0e-7 | 64/64 | 64/64 | 0.02112 |
    | Si far (9.2 it) | mixed / f32 | 6.9e-8 / 1.5e-7 | 6.3e-8 / 9.0e-8 | 2.3e-6 (h32 near 0) | 64/64 | 64/64 | n/a |
    | Si rim (176 it mean) | mixed / f32 | 9.2e-7 / 7.8e-7 | 1.3e-6 / 1.4e-6 | 5.9e-8 | 64/64 | 64/64 (63 converged on both) | n/a |

    Worst absolute error per component (mixed, all sets): h13
    2.5e-7 px, h23 4.1e-7 px, the linear terms at most 1.5e-9,
    h31/h32 at most 4.1e-12 per px; CIC relative difference at most
    2.7e-7. Iteration counts and convergence flags agree on 320 of
    320 compared patterns. The tiled kernel gives the same parity
    (its Si rim final-h maximum 1.6e-6 px). Against the CPU pins:
    `WARP_REFIT_TOL_480` (0.025 px) holds for the f32-arithmetic
    device, recovery moving by at most 4.4e-7 px -- but the CPU
    ITSELF reaches 0.02251 px on a 64-pattern batch, 90 per cent of
    the pin, which was measured on 12 patterns (0.01244 px), so the
    margin is thin at large N (D21.8); `UPDATE_RULE_TOL` (1e-13
    px) is a CPU-implementation pin that no device precision meets
    as such -- float64 comes closest (1.4e-13 to 1.6e-11 px on the
    increments against the CPU), so it gets its own band too, and
    mixed needs about 2e-6 px (twice its 9.2e-7), still 40x below
    the tightest mutant separation of 8.2e-5 px; power-of-two
    intensity invariance is bitwise on the device for every
    precision; `INTENSITY_SCALE_GENERIC_TOL` (6e-9 px) is met by
    float64 (3.6e-9) and missed by mixed (3.2e-8) and f32
    (1.1e-7), so the device
    needs its own band.

94. **Seeds (requirements D21.5).** `gpu_det_seeds.py`: complex128
    device seeds are bitwise equal to the CPU's skimage seeds on
    all 256 compared patterns (480 seed 0, 512x622 seed 0, Si far,
    Si rim; 64 each). Complex64 seeds: 64 of 64 equal on three sets
    and 63 of 64 on Si far, the one differing by 0.0625 px (one
    1/16 px step). Whole map, complex64 against complex128
    (`compare_maps.py` on `fullmap_c8.npz` and `fullmap_c16.npz`):
    7 points differ by one iteration, 12 differ by more than 1e-4
    px (max 6.0e-4 px, below `min_step`), the rest are identical;
    both runs converge 57685 points.

95. **Determinism (requirements D21.7).** `gpu_det_seeds.py`, 5 runs
    per arm compared with the first: the two-stage reduction (a
    fixed shared-memory tree per block, then a fixed-order
    cross-block pass) is bitwise identical in 4 of 4 repeats on all
    four sets for float64, mixed and f32, and the whole device
    pipeline, cuFFT preprocessing and seeds included, is bitwise
    identical over 3 runs per set (0 patterns differing). Atomics
    are not: f32 atomic reductions differ on 64 of 64 patterns on
    every run (up to 4.7e-7); the mixed variant's f64 atomic
    accumulation differed on one pattern by 4.4e-16 at 480 px
    (identical elsewhere), so it is not guaranteed either. Bits
    depend on the LAUNCH LAYOUT: the tiled kernel at 1 and at 8
    patterns per block is each bitwise repeatable but differs from
    the other (480 seed 0, mixed: final-h maxima 4.41e-7 and 5.81e-7
    px against the CPU), and the prototype derived its block count
    from B -- hence D21.7.3's rule tying the layout to the pattern
    geometry only.

96. **The file-read ceiling (requirements D21.11).**
    `io_ceiling.py` (first run in `io_ceiling_run1.log`). The
    file's LZF chunks do not compress: 318432 of 318464 bytes per
    pattern.

    | route | patterns/s, first read of a range | re-read |
    |---|---|---|
    | h5py, one thread, 500 consecutive patterns | 3342 / 4318 / 3421 | 4653 |
    | h5py slice read | 3516 | |
    | thread pools of 2, 4, 8 | 2465, 1587, 1955 (the h5py lock) | 4265 (8 threads) |
    | 4 processes | 5350 | |
    | `kp.load(lazy=True)` + dask, 8 threads | 1129 | 1103 |

    The 8-process rows are invalid (`MemoryError`, entry 89).
    During the whole-map runs of entry 97 a single DEDICATED h5py
    reader thread (the prototype's, not a v1 component) was busy
    40.8 s (complex128) and 50.6 s (complex64) and the device
    waited 0.33 s in total for data -- a property of that reader,
    not of the v1 dask route (D21.11). LIMIT, recorded at the spec
    review: the `kp.load` plus dask row times
    `data[start:start + N].compute()` on a CONTIGUOUS slice
    rechunked to 32 (`io_ceiling.py:92-96`), not the engine's
    fancy-index gather.
    ADDENDUM (2026-10-06, spec review; `io_gather_probe.py`, CPU
    load 1-4 per cent, no GPU work): the v1 route itself,
    `data[fit_indices].rechunk((64, -1, -1))` on
    `kp.load(lazy=True)` reshaped to `(58500, 512, 622)` (stored
    chunks of one pattern), 8 dask threads, each block reduced to
    per-pattern sums so memory stays small:

    | route, B = 64 | points | patterns/s, first read | re-read |
    |---|---|---|---|
    | fancy-index gather, crater band flat 26900:33100, 362 masked removed | 5838 | 1640 | 2142 |
    | contiguous slice, flat 8000:13838 | 5838 | 1504 | |
    | fancy-index gather, flat 41000:45200 (none masked) | 4200 | 1527 | 1881 |
    | contiguous slice, flat 52000:56200 | 4200 | 1651 | |

    "First read" is first in that process; the whole-map runs had
    read the file earlier that day, so the OS cache state is
    unknown. So the gather costs no more than a contiguous read and
    sits about 2.6x above the fastest whole-map device rate (628
    patterns/s); whether concurrent chunks keep the device fed
    under `_DEVICE_LOCK` is E8, an implementation-gate measurement.

97. **End to end and the whole map (requirements D21.16).**
    `gpu_e2e.py`, `gpu_fullmap.py`; the recip6 build, mixed, static
    batches of 128, device preparation in sub-batches of 32,
    `nvidia-smi` clean before each. Relabelled at the spec review
    (2026-10-06): the far256 and patch C rows are DEVICE-ONLY --
    `gpu_e2e.py` measures the host read separately (h5py, one
    thread: 2546 patterns/s on far256, 4514 on patch C) and does
    NOT include it -- so the read-inclusive figures (device time
    plus that read, serial) stand beside them:

    | run | complex128 seed, device only / with read | complex64 seed, device only / with read |
    |---|---|---|
    | far256 (rows 20:36, cols 20:36; mean 8.7 it, 256/256 conv.) | 395 pat/s (0.647 s) / about 342 pat/s (0.748 s) | 947 pat/s (0.270 s) / about 691 pat/s (0.371 s) |
    | patch C (618 points; mean 87.4, median 76, max 497 it; 618/618 conv.) | 212 pat/s (2.92 s) / about 202 pat/s (3.06 s) | 300 pat/s (2.06 s) / about 281 pat/s (2.20 s) |

    Slot refill ("stream") against static batches: far256 411 and
    919 patterns/s, patch C 193 and 304 (2.03 s), so no gain. The
    CPU took 338.3 s on patch C (ledger 84), I/O included: 116x and
    164x against the device-only GPU time, about 111x and 154x
    against the read-inclusive one (the like-for-like ratio). The
    GPU's patch C iteration total is 54032, exactly the CPU's
    (ledger 84), with 618 of 618 converged on both. WHOLE MAP,
    57772 fitted (728 masked), the h5oina read included through the
    prototype's DEDICATED reader thread (a two-deep queue of
    contiguous map-order blocks of 512 patterns, each read pattern
    by pattern with h5py, masked points dropped; static B = 128;
    `gpu_fullmap.py:43-69`), a setup v1 does not freeze (D21.11):
    complex128 **166.5 s = 347 patterns/s**, complex64 **92.0 s =
    628 patterns/s**, against 2.34 h = 6.85 patterns/s on 8 CPU
    workers (ledger 82), i.e. **51x and 92x**, both at the MIXED
    device precision. Both runs: 57685
    converged and 87 not (all 87 at the 500 cap), iterations median
    14, mean 19.2, p90 30, p99 106, max 500 (totals 1110802 and
    1110805), residual median of converged points 0.04626 -- ledger
    82's counts. GPU utilisation 51-96 per cent during the runs.
    THE LIMIT of this comparison: it is aggregate. No per-point CPU
    arrays of the map exist on disk (the executed notebook stores
    figures and printed statistics), so map-level per-point parity
    is not established here (plan open question E12).

98. **The model, the projection and the new bottleneck
    (requirements D21.3, D21.5, D21.16).** Fitted to entries 97 and
    90: GPU time per pattern = F + 0.024 ms x iterations, F = 2.42
    ms with the complex128 seed and 1.13 ms with complex64; CPU on
    8 workers = 17 ms + 6.7 ms x iterations. Every GPU figure here
    is the MIXED device precision. LIMIT, recorded at the spec
    review: the fit mixes the device-only far256 and patch C rows
    of entry 97 with the read-inclusive whole-map rows (the
    prototype's dedicated reader thread), so F carries some read
    time on the map and none on the patches.

    | regime | CPU, 8 workers | GPU, complex128 seed | GPU, complex64 seed | bottleneck |
    |---|---|---|---|---|
    | far field, 14 it | 9.0 pat/s (ledger zones 10-12) | 362 pat/s (40x) | 680 pat/s (75x) | CPU: iterations 85 %; GPU: fixed costs 88 % / 77 % |
    | rim, 87 it | 1.67 pat/s (ledger 1.8) | 222 pat/s (133x) | 311 pat/s (186x) | CPU: iterations 97 %; GPU: fixed 54 % and iterations 46 % / iterations 65 % |
    | whole map, measured (mixed) | 6.85 pat/s, 2.34 h | 347 pat/s, 166.5 s (51x) | 628 pat/s, 92.0 s (92x) | GPU with complex128 seed: seed about 62 %, iterations 16 %, the rest 21 % |
    | whole map, FLOAT64 device precision, projected (added 2026-10-06, spec review) | 6.85 pat/s, 2.34 h | about 131 pat/s, about 440 s (about 19x) | about 158 pat/s, about 365 s (about 23x) | iterations about 68 % / 82 % |

    The float64 row is an INFERENCE, not a measurement: the same F
    (preprocessing and seed are f64 under both device precisions)
    plus the measured float64 per-iteration cost of 0.270 ms per
    pattern-iteration (entry 91) times the map's 1110802
    iterations: 2.42 x 57772 + 0.270 x 1110802 ms = 139.8 + 299.9
    s, and 65.3 + 299.9 s with complex64. It is the figure that
    applies while the D17 amendment is unapproved (the default in
    force, D21.4); far field and rim project to about 161 and 39
    patterns/s with the complex128 seed.
    So the plan 9.5 estimate of 5-20x was conservative under mixed
    and about right under float64, and under mixed the next device
    lever is the SEED, not IC-GN: the complex64 seed (about 1.8x end
    to end), then an rfft2 band-pass, a box-mean K0, pinned
    double-buffered upload and CUDA graphs (all D21.18 follow-ups).
    The `kp.load` plus dask read route (about 1100 patterns/s on
    contiguous slices, entry 96; the v1 gather 1500-2100 patterns/s
    at B = 64, entry 96 addendum) could cap a tuned device path.
    For scale only: the paper's MapSweeper runs about 20 patterns/s
    on two RTX 4090 GPUs (ledger 82; a different method with
    heavier per-candidate work).

99. **What the prototype does NOT establish, and a harness bug
    (requirements D21.6, D21.8).** Not prototyped: `window=True`;
    multiple grains per batch (would need a per-pattern state
    index); the NaN contract for failed patterns (non-finite norms
    must be flagged in the update kernel); `dead_band` (the code
    path exists, untested); the Stage D cascade. Those are
    specified by parity alone (D21.6) and their oracles are V9(h)
    and (i). A bug in the prototype's own harness was found and
    fixed: its `run()` updated the caller's `h0` and K arrays in
    place, which corrupted the determinism and end-to-end seed
    fields of `gpu_accuracy.json` (they read "seeds equal 0 of 64");
    those fields were re-measured in `gpu_det_seeds.json`, the
    source of entries 94 and 95. CPU-side findings recorded for a
    separate note, outside Stage E (D21.18): limiting OpenBLAS to
    one thread gives up to about 1.3-1.5x on 8 workers, and numpy
    warp coordinates are 31 per cent of each CPU iteration (fusing
    them into the numba kernel not measured). Precedent for reading
    every number above: Phase 12's implementation landed below its
    projection (0.809 ms of device time per pattern against 0.65-0.75
    ms; 465 patterns/s at bw 88 against 650-850 expected), so the
    Stage E performance record re-measures rather than inherits.
    Prototype choices the spec does NOT freeze (recorded 2026-10-06
    at the spec review, from `kernels.cu`, `gpu_icgn.py` and
    `gpu_fullmap.py`; every number above was measured WITH them):
    the adaptive K was reconstructed as an f64 `K + mp` against an
    f32-rounded subtraction (`kernels.cu:331`; D21.4 now reconstructs
    against the shift actually subtracted); the corner-norm maximum
    started at 0 and swallowed NaN (`kernels.cu:281-288`; D21.6.1
    now propagates it); the cell index was converted to an integer
    before the mirror fold (`kernels.cu:31-33`; D21.6.3 folds in
    floating point first); `h` was carried and the matrix rebuilt
    as `1 + h` (D21.6.4 carries the matrix); the algebra assumed
    `window=None` and raised otherwise (D21.6.5 writes out the
    windowed form); the reference spectrum was computed on the
    device (`gpu_icgn.py:118`; D21.5 now says so); the reference
    vector, `xi_x`, `xi_y` and the weighted gradient columns were
    STORED in f32 under mixed (`gpu_icgn.py:89-93`; named in the D17
    amendment); and the whole-map runs used a dedicated reader
    thread and static B = 128 (D21.11 and D21.10.3 freeze neither).
    Each frozen alternative differs from the prototype only at
    rounding level on the measured inputs, or on inputs the
    prototype never met; the implementation gate re-measures every
    band on the frozen design.

#### V9 recorded results, failing-tests gate (2026-10-06)

Measured on machine A (ledger 89) by the Stage E failing-tests gate,
CPU only (no GPU work), on the worktree `.venv` (CPython 3.13.12,
numpy 2.4.6, scipy 1.17.1, numba 0.65.1, scikit-image 0.26.0), on
commit 49d8bbad plus the Stage E skeleton (`_hrebsd/_gpu.py` and
`_hrebsd/_batched.py` stubs; `backend`, `device_precision` and
`seed_precision` added to `run_hrebsd_dic` and `backend` to
`EBSD.hrebsd_dic`, where `backend="cpu"` passes straight through to
the unchanged CPU path). Script: session scratchpad
`measure_e100.py`, which loads the scaffold
`tests/test_indexing/test_hrebsd_gpu.py` by path and calls its own
fixture builders and helpers, so the numbers are those of the
scaffold's own recipe; run twice (by the interrupted skeleton agent
and again by the finishing agent), with identical numbers.

100. **The drift tripwire's CPU half and the default-suite fixtures
    (requirements D21.8(h), D21.14.3; V9(l), fixtures F1 to F7).**
    (i) The CPU half of (l). RECIPE: `drift_recovery_cpu()` of the
    scaffold -- F1 (`f1_batch()`: the V2 480 px reference at
    `PC_480`, warped by `random_small_homographies(n=6, seed=s)` for
    s = 0, 1 with the independent skimage warper) as a `(1, 13)` map
    with the same projection centre at every point, through
    `run_hrebsd_dic(backend="cpu", reference=(0, 0), verbose=0)` at
    every other default, then per case the V2 recovery metric
    against the EXACT imposed homography. In F1 order, px:

    | seed 0 | 0.007831707107991194 | 0.004932800467962715 | 0.001150689518357858 | 0.0029205924994070817 | 0.0028893302430281925 | 0.008199674743814368 |
    |---|---|---|---|---|---|---|
    | seed 1 | 0.012439859159007909 | 0.003212421044706529 | 0.0022442932279219748 | 0.0024240622676751796 | 0.002017307940207109 | 0.009035331439900308 |

    Iterations 4, 5, 6, 6, 5, 6 and 4, 7, 5, 6, 4, 4; 12 of 12
    converged. The worst, 0.012439859 px, is the 0.01244 px that
    `WARP_REFIT_TOL_480` was pinned on at the Stage A gate. Two runs
    are BITWISE equal, and `fit_pattern` on the same reference state
    gives the same twelve numbers to the bit. Shared-code sensitivity
    on the same recipe, the largest per-case move: 2.13e-9 px with
    `coefficient_dtype=np.float64`, 2.25e-7 px with
    `upsample_factor=8`, 5.74e-5 px with `min_step=1e-2`. PINNED in
    the scaffold as `CPU_DRIFT_RECOVERY_PX` (the twelve literals),
    `CPU_DRIFT_NUM_ITERATIONS` (exact) and `CPU_DRIFT_TRIPWIRE_PX =
    1e-9` px. RECORDED DEVIATION from the letter of D21.8(h), which
    names one MTP band `GPU_DRIFT_TRIPWIRE_PX` for both halves: the
    CPU half gets its own FROZEN float-noise band now (the
    `PRE_STAGE_D_PIN_TOL = 1e-9` precedent of
    `test_hrebsd_seeding.py`), because the CPU path is deterministic
    and its half must run green in the default suite from this gate
    on; `GPU_DRIFT_TRIPWIRE_PX` and the device literals stay
    `FIXME-pin` until the implementation gate.
    (ii) Fixture premises, measured with the same builders. F5
    (`f5_map()`, two grains of 60 px patterns, references at flat 0
    and 4, default `max_iterations=50`), at `(None, None)` and at
    the default `(0.05, None)` alike: references converge in 1
    iteration, the easy points in 4 (grain A) and 3 (grain B); the
    masked point, the integer-constant target (7.0) and the target
    with one NaN pixel give the D2.6 failure contract with truthful
    `grain_id` and `reference_index`; the capped point (a 25 degree
    in-plane rotation of B, V3 deformed master) exhausts the 50
    iterations with a finite last iterate 22.2 px from its imposed
    homography, residual 1.777 at `(None, None)`. F6 (`f6_map()`)
    reproduces the pre-Stage-D counts 1, 5, 10, 200 and flags True,
    True, True, False. F7 (`f7_map()`, 18 grains of 2 by 2 points at
    60 px): 72 of 72 converge, at most 4 iterations.
    F6 also reproduces `PRE_STAGE_D_HOMOGRAPHY` of
    `test_hrebsd_seeding.py` to 5.7e-14 px (the metric's floor), its
    residuals and iteration counts exactly, and `backend="cpu"` equals
    the call without the keyword bitwise on F6 (script
    `check_f6.py`). Build times over the two runs: F1 1.4 to 1.5 s,
    F3s 0.26 to 0.34 s, F4 0.10 to 0.12 s, F6 0.19 to 0.32 s, F5 and
    F7 under 0.05 s each (the Ni master cached); the F6 run 3.2 to
    4.6 s, the F7 run 0.17 to 0.24 s, each F5 run 0.03 s.
    (iii) The default-suite wall time of D21.14.3 is recorded when the
    test classes are in place, later at this gate.

101. **The failing-tests gate tally and the default-suite wall time
    (requirements D21.14.3; V9(a) to (p); plan 11 items 1 and 4).**
    The default-suite fragment (15 classes) and the gated fragment
    (12 classes) were spliced into
    `tests/test_indexing/test_hrebsd_gpu.py` (6700 lines with the
    plan 11 item 4 mutation map at its foot), imports merged at the
    module top; `ruff check` and `ruff format --check` are clean on
    every touched `.py` file.
    (i) Default suite, CPU `.venv` (pytest 9.0.3), command
    `uv run pytest tests/test_indexing/test_hrebsd_gpu.py -n 0 -q -p
    no:cacheprovider`: **154 failed, 29 passed, 271 skipped** (no
    errors, nothing collected wrongly). Every failure is for the
    right reason: 129 raise `NotImplementedError: Stage E: not
    implemented yet` from the skeleton, 17 assert behaviour the
    skeleton lacks (gate
    messages, the cached-failure copy, the frozen Stage D raise
    literal, the D21.17 docstring entry, the CUDA source file, the
    precision values and planted exceptions reaching the gate or the
    seam, and the numpy session in a subprocess, all pre-empted by
    the skeleton's `NotImplementedError`), and 8 are the
    unfilled `SEAM_DISCRIMINATING_MIN` `FIXME-pin` placeholders of
    V9(e) (CPU-side counts measured as F1 12, 8, 11, 12 and F3s 8, 5,
    6, 8 for the exact, shift_rotate, rotate_1_5 and perspective
    rows; left unpinned because the gated suite reads the same table
    under ("F1", "exact") and ("F1", "perspective") with a different
    perspective row, a key collision the review gate resolves). The
    29 that pass now do so by design: the `backend="cpu"` bitwise
    arms on F6 and the Ni map and the never-touches-the-gate spy,
    the signature and forwarding pins, the existing argument checks
    coming first, the precision knobs read under `"gpu"` only, the
    Stage D CPU cascade, the three spherical names by identity, the
    frozen constants and structural values, the launch-layout
    signature, the two CUDA source pins (no atomics, no fast math;
    `test_the_cuda_source_exists` keeps them from passing
    vacuously), the drift tripwire's CPU half (3), five fixture-gating
    tests, and six of the seven import-hygiene tests. Skipped: 204
    gated tests with the gate's `NotImplementedError` text as the
    reason, 66 F2 and F3 seam arms needing `--weekly`, and the canary
    (`KIKUCHIPY_EXPECT_GPU` unset). **Default-suite wall time
    (D21.14.3): 46.9 s as pytest reports it, 54.6 s wall including
    `uv` start-up**, at `-n 0` on machine A; two earlier runs of the
    same file at the same gate took 76.0 s and 85.2 s while the
    machine was shared with an EMHROSM CPU job, so the figure is
    load dependent.
    (ii) Gated run through the PINNED overlay of D21.15 with
    `KIKUCHIPY_EXPECT_GPU=1` at `-n 0` (`nvidia-smi`: 0 MiB, 0 %
    before the run): **155 failed, 29 passed, 270 skipped** in 45.7 s
    (66 s wall). Collection is clean; the 270 gated tests are 204
    skips on the gate's `NotImplementedError` reason plus the same 66
    weekly skips; the extra failure is the canary, failing for the
    right reason ("would skip here: Stage E: not implemented yet").
    (iii) ONE WRONG-REASON CASE found and fixed: in the overlay,
    `TestNoModuleScopeCupy::test_importing_kikuchipy_imports_no_cupy`
    failed because `dask.array.chunk_types` try-imports cupy at module
    scope, so any process importing `dask.array` holds cupy whenever
    it is installed; the bare "`'cupy' not in sys.modules`" check
    was therefore wrong outside the CPU `.venv`. Both subprocess tests
    now run `CUPY_IMPORTER_PROBE` first, which records every cupy
    import whose importing code is a `kikuchipy` module (through
    `builtins.__import__`, cached or not, and
    `importlib.import_module`). Checked on both environments: no
    false positive from dask, and a planted kikuchipy-scope `import
    cupy` and `importlib.import_module("cupy.fft")` are both caught
    with cupy already cached (scratch `probe_check.py`).
    (iv) The existing suites: `uv run pytest tests -k hrebsd -n 4 -q`
    gives 154 failed, 725 passed, 280 skipped in 112.9 s, every
    failure in `test_hrebsd_gpu.py`; the CPU path is unchanged.
    (v) Mutation map: every M1 to M53 names at least one designed
    killer (D: default suite, G: gated); M3, M4 and M36 have gated
    killers only; M1's gated determinism test is noted as not a
    reliable killer; M15 is flagged possibly equivalent. None of the
    kills is verified yet (the plan 11 bug-injection pass).

102. **Test-critic disposition at the failing-tests gate
    (2026-10-06/07; V9(a) to (p); plan 11 items 1 and 4).** The test
    critic raised 13 findings (1 blocker, 4 major, 8 minor). Each was
    checked against the file before acting: **12 accepted** (one of
    them, F4, in part), **1 rejected in part** (F4's plan.md mirror),
    and F11 accepted as a record only. Edits are confined to
    `tests/test_indexing/test_hrebsd_gpu.py` and the docstrings of
    `_batched.py` and `_gpu.py`; no CPU code moved.
    (i) **F1 (blocker), accepted.** The V9(e) pin tables were shared
    by the two fragments with incompatible keys and meanings, and the
    two row builders differed. There is now ONE builder,
    `seam_table(name, row_type)`, with the row types `exact`,
    `translate_rotate`, `rotate_1p5`, `perspective` and `nan`. It plants
    every map point, the reference included, and the perspective step
    `d` moves the subregion corners by `SEAM_PERSPECTIVE_PX = 0.5` px
    (2.7505e-6 per px at 480, 1.9628e-6 at 512x622). There is also ONE
    packed CPU oracle, `seam_cpu(name, row_type, max_iterations)`,
    where `None` is the translation seed. Both suites compare over the
    same map points. The pins are split: `NUMPY_SEAM_BOTH_CONVERGED_MIN`,
    `NUMPY_SEAM_ITERATION_DIFF_COUNT` and `NUMPY_SEAM_CONVERGED_FLIP_COUNT`
    are keyed `(fixture, row_type)`, while the `GPU_SEAM_*` tables are
    keyed `(fixture, device_precision, row_type)`.
    `SEAM_DISCRIMINATING_MIN` is now a CPU-side literal keyed
    `(fixture, row_type)` and **MEASURED and PINNED at this gate**,
    from the scratch script `measure_seam.py` on machine A's CPU,
    worktree `.venv`. Each value counts the map points whose CPU
    iteration count from the planted row differs from the
    translation seed's, at `max_iterations=50`, listed as exact,
    translate_rotate, rotate_1p5, perspective:

    | fixture (points) | exact | translate_rotate | rotate_1p5 | perspective |
    | --- | --- | --- | --- | --- |
    | F1 (13) | 12 | 9 | 12 | 13 |
    | F3s (9) | 8 | 6 | 7 | 9 |
    | F2-0 (65) | 64 | 39 | 63 | 62 |
    | F2-1 (65) | 64 | 39 | 65 | 64 |
    | F3 (65) | 64 | 47 | 61 | 62 |

    A second fit of every planted row was bitwise equal. Every planted
    finite row converges on the CPU at every point. The CPU iteration
    counts from the planted rows are 1 or 2 for exact, 4 or 5 for
    translate_rotate, 6 or 7 for rotate_1p5 and 3 for perspective. The
    NaN row gives the D2.6 failure contract at every point, and
    `max_iterations=1` gives exactly 1 iteration everywhere. The
    measurement took 21 s (F1), 20 s (F3s), 91 s and 87 s (F2-0, F2-1)
    and 141 s (F3). The 8 default tests that ledger 101 recorded red on
    this placeholder now pass.
    (ii) **F2 (major), accepted.** On F5 and on the single-grain
    fixtures the fit order equals the map order, so M17 and M52's
    map-order half had no default killer. Two tests were added to
    `TestRunBatchesNumpySession`:
    - `test_the_map_order_on_f7` runs the twin against the CPU at
      chunksize 4. It asserts a coarse 1e-3 px per-point bound
      independent of every pin, then `assert_twin_parity` under the
      key "F7". This kills M17.
    - `test_the_seam_carries_the_fit_order_on_f7` is a seam spy
      through the new helper `assert_seam_fit_order`. Every
      `SeedBatch.pattern_index` must be a contiguous run of
      `grain_fit_order(fixture)`, with the real slots first and the
      -1 padding after. The runs must tile the fit list once, in any
      batch order the threaded scheduler picks. Each slot's targets
      and coefficients must equal its own pattern's. This kills M52.
    Premises checked: F7's grain order is not sorted (it starts 0, 1,
    12, 13, 2, 3, ...), `grain_fit_order(F5)` equals
    `F5_FIT_INDICES`, and the CPU F7 run converges 72 of 72. A planted
    map-order record set fails the helper, while the fit order and a
    reversed batch order pass (scratch `check_order.py`). The same
    fit-order assert was added to the gated
    `TestGatedBatchedSemantics::test_many_grain_map_order`.
    (iii) **F3 (major), accepted.** The gated M15 arm now uses
    `DIM_SCALE` (1e-13) and the default recipe: dim the raw reference
    and targets, then preprocess. It re-asserts the `100 * eps` floor
    premise and that the dimmed CPU seeds equal the undimmed ones.
    (iv) **F4 (major), accepted except one part.** The module docstring
    of the test file gains "The call-time seams this commit freezes",
    which lists every patched or spied name:
    - `_engine` reaches `_verify_gpu_or_raise`, `_gpu._run_chunks_gpu`,
      `ReferenceState` and `resolve_reference`.
    - `_gpu` reaches `_make_session`, `_free_device_bytes` and
      `_default_batch_size`.
    - `_batched` exposes `build_resident`, `build_seed_state`,
      `seed_spectra` and `seed_homographies` through the module
      object.
    - `session.kernels.gather`, `.pixel_sums`, `.reduce_solve_update`
      and `.final_criterion` are reached by attribute in every
      lockstep iteration. `KernelNamespace` is mutable. Fusion is
      allowed only inside one entry point.
    The same section restates the gate-probe API surface that the fake
    cupy provides. The `_batched` and `_gpu` module docstrings and the
    `make_kernel_namespace` docstring now say the same. **Rejected:**
    mirroring this into plan 11 item 2. Plan.md is outside this phase's
    allowed edits, so the reconciliation is recorded here instead: the
    plan's "fused pixel kernel" means the fused kernel BEHIND
    `pixel_sums` (or behind `gather`), never a fusion across the two
    entry points.
    (v) **F5 (major), accepted (the freeze option).** Every kernel,
    the gate probe's included, is constructed through the attribute
    `cupy.RawKernel` inside the call that uses it, once per call. A
    `cupy.RawModule` is not allowed, and neither is a Python-level
    cache of kernel objects across calls; CuPy's own compile cache is
    fine. This is stated in the module docstring, in
    `make_kernel_namespace` and in `_spy_raw_kernels`.
    (vi) **F6 (minor), accepted.** The comments of both M43 tests and
    the M43 map line now say that they plant NaN sums or values, not a
    NaN corner displacement from a finite step. They kill M43 only in
    an implementation without a separate non-finite-step flag;
    otherwise M43 is reviewed-equivalent at the review gate, because it
    needs an exact 0/0 projective corner. No helper was added.
    (vii) **F7 (minor), accepted.** The F7 residency spy now asserts at
    most `R_MAX - 1` live residents and seed states on entry to every
    upload (D21.9.3: evict BEFORE upload).
    (viii) **F8 (minor), accepted.** The information-message test runs
    at chunksize 40, so B = 40 and P = 32, and asserts both `\b40\b`
    and `\b32\b`.
    (ix) **F9 (minor), accepted.** A new
    `TestCudaSourcePins::test_no_atomic_scatter_through_cupy` forbids
    `scatter_add` and `.add.at(` in every `_hrebsd` `.py` file. It
    passes on the skeleton by design, like the two existing source
    pins.
    (x) **F10 (minor), accepted.** In the mutation map, M37 to M42 and
    M44 now carry "D*": the mutation lives in the CUDA source, and a
    default-suite kill counts only for the numpy-twin variant of the
    injection. The bug-injection pass injects both, and M42 is
    mixed-only.
    (xi) **F11 (minor), accepted as a record.** Ledger 101(i)'s 46.9 s
    is the SKELETON-stage time: 129 to 131 tests stop at the stub, so
    it says nothing of the twin's 480 px fits after implementation. The
    implementation gate must re-record the D21.14.3 default-suite wall
    time as a dated entry. If it is large, the heaviest arms move to
    `weekly` with a recorded deviation, with the F1tail last-sub-batch
    arm first in line. Ledger 101 is not edited; it is append-only.
    (xii) **F12 (minor), accepted (the comment option).** The bitwise
    B >= 32 assert stays, with a comment: a measured non-bitwise result
    needs a dated deviation and a move to `GPU_BATCH_INVARIANCE_TOL`.
    (xiii) **F13 (minor), accepted.** The splice markers are gone. The
    helper pairs are merged into one each:
    - `ROW_SLOTS` and `DEFAULT_FIT_OPTIONS`.
    - `assert_at_least`, `assert_failure_contract` and `_pin` (which
      replaces `_pinned`).
    - `cpu_run`, now one cache, with `FIXTURE_BUILDERS` and
      `fixture_of` covering F2-0, F2-1 and F3.
    - `spy_seam`, now xp-agnostic through `to_host`.
    - `plant_seam(monkeypatch, rows_of, *, call_original=True)` with
      `plant_table(monkeypatch, table)`.
    - `in_plane_matrix`.
    - `run_direct(..., device_precision=, seed_precision=, **options)`,
      which replaces `_run_rows`.
    - `packed`, `h_band` and `perspective_step`.
    - `F5_FIT_INDICES` and `F5_STATE_OF_POINT`.
    - `assert_device_residual_band`, which wraps
      `assert_residual_band`.
    The DC offset is now `F4_DC_OFFSET = 1000` in both suites; the
    gated DC arm uses `f4_dc_batch` and `cpu_run("F4dc", ...)`. The
    numpy-twin and device pin names stay separate.
    (xiv) **Re-runs.**
    - Default suite (`uv run pytest tests/test_indexing/test_hrebsd_gpu.py
      -n 0 -q -p no:cacheprovider`): **148 failed, 38 passed, 271
      skipped** (457 collected). All failures are for the right
      reason: 131 `NotImplementedError: Stage E: not implemented yet`,
      and the same 17 skeleton-behaviour asserts as ledger 101. No
      FIXME-pin failure remains in the default suite. Wall time
      (D21.14.3, skeleton stage) was **82.1 s and 44.3 s** as pytest
      reports it, on two runs minutes apart, while the machine was
      shared with the HROSM session. The figure depends on load.
    - Gated (pinned overlay, `KIKUCHIPY_EXPECT_GPU=1`, `-n 0`;
      `nvidia-smi` 83 MiB, 3 % before the run): **149 failed, 38
      passed, 270 skipped** (457 collected) in 43.3 s. The extra
      failure is the canary, for the right reason. The skips are 204
      on the gate's `NotImplementedError` reason and 66 weekly F2 and
      F3 seam arms.
    - `uv run pytest tests -k hrebsd -n 4 -q`: **148 failed, 734 passed,
      280 skipped** in 202.2 s. Every failure is in
      `test_hrebsd_gpu.py`; the other suites are unchanged.
    Two earlier `-n 4` attempts lost their xdist workers to
    `MemoryError` during import. Commit memory sat at 38 to 39 GB of a
    39.4 GB commit limit, under the concurrent HROSM job. This was
    environmental, not code: `-n 2` on a small file passed at the same
    moment, and the third attempt was clean.
    `ruff check` and `ruff format` are clean on every touched file.
    (xv) **Mutation map.** Every M1 to M53 still names at least one
    designed killer. M17 now points at the F7 tests, and M52 at the F7
    seam spy and the gated F7 fit-order assert. M1 adds the scatter
    pin, M51 the `R_MAX - 1` bound, M15's gated killer uses the
    `DIM_SCALE` recipe, M43 carries its caveat, and M37 to M42 and M44
    are marked D*. None of the kills has been verified yet; that is the
    bug-injection pass.

#### V9 recorded results, implementation gate (2026-10-07)

Measured on machine A (ledger 89) by the Stage E implementation gate,
on the worktree `.venv` (CPython 3.13.12, numpy 2.4.6, scipy 1.17.1,
numba 0.65.1, scikit-image 0.26.0) for the default suite, and on the
pinned overlay of D21.15 (`-n 0`) for the gated runs quoted for
information. Base commit f297867e plus the uncommitted Stage E
implementation (`_hrebsd/_batched.py`, the new device-only
`_hrebsd/_cuda.py`, and the `_gpu.py`/`_engine.py`/`ebsd.py`
plumbing of the parallel implementer).

103. **The xp-agnostic batched core and the numpy-twin pins
    (implementer A; requirements D21.4 to D21.7, D21.9.5, D21.14.3;
    V9(d), (e), (f), (g), (h), (i), (m)).**
    (i) **What exists.** `_batched.py` holds the seed seam
    (`SeedContext`, `SeedState`, `SeedBatch`, `build_seed_state`,
    `seed_spectra`, `seed_homographies`: the batched transcription of
    skimage's `phase_cross_correlation` on the zero-mean unit-norm
    crops, complex128 by default and complex64 on request, NaN rows
    for a zero or non-finite crop norm or a non-finite peak and for
    every slot at `upsample_factor < 1`), `build_resident` (host f64
    derivation, then upload at the pixel precision; the constants
    `SD_w^T (w ref)` and `SD_w^T w` from the WEIGHTED block),
    `initial_shifts` (K0, the subregion mean rounded to the pixel
    precision), `run_lockstep` (the carried f64 matrix, padded slots
    and non-finite seeds never active, the non-finite-coordinate flag
    before the sums, retirement, the final criterion at the returned
    homography, the 12-wide packing), `solve_update` (the shared f64
    step algebra: NaN-propagating corner maximum, closed-form 3x3
    inverse, flags for a zero or non-finite norm, step, determinant or
    W33), `_launch_layout` (256 threads, about 16 pixels per thread,
    a function of n only) and the numpy `KernelNamespace` twin at
    float64 (the numba gather, the ten shifted sums of the module
    docstring as per-row `add.reduce` along the pixel axis, never a
    BLAS product over the batch, and a two-pass final criterion).
    `_cuda.py` holds the CUDA build of the same four entry points for
    both precisions (no atomics, fixed-order two-stage reductions, no
    fast math, every kernel built through `cupy.RawKernel` per
    `make_kernel_namespace` call; the gate is the parallel
    implementer's).
    (ii) **Seeds, V9(d).** RECIPE:
    `TestSeedSeamContract::test_rows_equal_initial_guess`. The numpy
    seed equals `initial_guess` bitwise on every row of F1 (12), F3s
    (8), F4 (3), F6 (4) and the Ni map (9) at upsample 16, 2 and 1;
    the dimmed arm is bitwise equal too. Pinned AT the counts
    (`NUMPY_SEED_EQUAL_COUNT`).
    (iii) **The numpy twin's bands and counts, V9(e), (f), (g).**
    RECIPE: the default-suite test bodies run unchanged under recording
    assert helpers (scratch `measure_numpy_pins.py`): the F1, F3s and
    F4 parity and first-step tests, the eleven knob arms, F5 and F7 map
    order, and the seam arms on F1 and F3s at every finite row type.
    Every iteration-count difference and every `converged` flip is 0,
    on every key; every planted seam row converged on both (13 of 13
    on F1, 9 of 9 on F3s). Worst values against the CPU, pinned at
    about 2x: h band 2.344e-13 px (seam F1 "perspective") ->
    `NUMPY_PARITY_H_TOL_F64 = 5e-13`; first step 1.798e-13 px (seam F3s
    "translate_rotate") -> `NUMPY_FIRST_STEP_TOL_F64 = 4e-13`;
    residuals near 1e-16 differ by at most 2.498e-16 absolute (F7) and
    the one large criterion (F5's capped point, 1.777) by 2.5e-16
    relative -> `NUMPY_PARITY_RESIDUAL_ATOL = 5e-16`,
    `NUMPY_PARITY_RESIDUAL_RTOL = 5e-16`; Fe 5.551e-16 (low-pass arm)
    -> `NUMPY_PARITY_FE_TOL = 1.2e-15`; the counts
    `NUMPY_ITERATION_DIFF_COUNT`, `NUMPY_CONVERGED_FLIP_COUNT`,
    `NUMPY_SEAM_ITERATION_DIFF_COUNT` and
    `NUMPY_SEAM_CONVERGED_FLIP_COUNT` = 0 (scalars, every key), and
    `NUMPY_SEAM_BOTH_CONVERGED_MIN` = 13 (F1) and 9 (F3s) per row
    type. Bitwise by construction and passing: batched equals alone on
    F5 at B = 8, the B = 40 tail-sub-batch slot equals alone, and the
    1e30 sentinel on padded slots leaves the output unchanged. The
    gather's bitwise agreement with the CPU's `evaluate` on planted
    NaN, +-inf and huge (3e9, 1e30, FLT_MAX) coordinates passes.
    (iv) **Default-suite state.** `uv run pytest
    tests/test_indexing/test_hrebsd_gpu.py -n 0 -q -k "Seed or
    BatchedCoreNumpy or LaunchLayout or CudaSourcePins"`: 103 passed,
    114 skipped. The whole default file at `-n 2`: 185 passed, 271
    skipped, 1 failed (`TestBatchModel::
    test_the_terms_sit_in_the_calibrated_bounds`, a VRAM calibration
    pin of the parallel implementer). `_batched.py` default-suite line
    coverage 97.8 %; the uncovered lines (the cupy branch of
    `make_kernel_namespace`, the mixed resident, the mixed K rounding)
    run in the gated suite. No test was changed except to fill the pins
    named above.
    (v) **Gated, for information only (no device pin filled here).**
    `nvidia-smi` showed the HROSM session's `EMDI.exe` on the GPU (about
    33 % utilisation) during these runs, so no timing was taken. Every
    gated failure in `TestGatedSeedParity`, `TestGatedParity`,
    `TestGatedKernelAB` and `TestGatedBatchedSemantics` is an unfilled
    FIXME-pin placeholder. Measured on the way: per-kernel A/B at
    float64 (scale-free relative difference) gather 1.47e-14, the far
    coordinates 0 to 1.5e-16, `pixel_sums` 8.7e-16,
    `reduce_solve_update` matrices 1.0e-16; complex128 device seeds
    equal `initial_guess` on every row of F1, F2-0, F2-1, F3 and F4 at
    upsample 16, 2 and 1 (and the dimmed and runner arms), complex64
    differing on 0 rows; converged h bands mixed 5.7e-8 to 5.2e-7 px
    and float64 1.6e-13 to 3.2e-12 px; first-step bands mixed 3.9e-8 to
    1.2e-7 px and float64 1.6e-14 to 1.6e-11 px; the Ni map through the
    public method 7.0e-8 px (mixed). The scales match ledger 93. The
    launch-dimension spy passes once the solve kernel runs one
    single-thread block per pattern (a grid of `ceil(B / 64)` blocks
    failed it). The resident set holds one copy per plane at the pixel
    precision, plus int32 pixel indices under `"mixed"` only, and the
    seed state's zero-mean normalisation works in place. With that,
    the resident high-water marks at 512x622 are 23.2 MB (mixed,
    complex128) and 26.2 MB (float64, complex128). Before the trim they
    were 27.3 and 30.4 MB, and mixed complex64 was 19.0 MB. Still open,
    and not in the batched core: the VRAM `r` model (20.4 MB mixed
    complex128) is still below the mixed complex128 mark, and
    `test_batch_size_invariance[float64]` reports that its pool limit
    did not force a halving. Both belong to the calibration step.

104. **The device-side MTP pins (requirements D21.4 to D21.8,
    D21.10; V9(d) to (o); integrator).** Every device pin in
    `test_hrebsd_gpu.py` that cites "ledger 104" was measured by the
    first integrator on 2026-10-07 (before the 717f0c15 checkpoint)
    with the gated suite run once under recording assert helpers
    (scratch `recpins.py`, the test bodies unchanged; `--weekly`,
    `-n 0`, the pinned overlay of D21.15, `KIKUCHIPY_EXPECT_GPU=1`,
    `nvidia-smi` idle). That integrator was stopped before writing
    this entry, so it is written here, and every value was RE-READ
    from the 2026-10-07 04:24 `--weekly` gated run of entry 106
    (junit `record_property` values; `nvidia-smi` 0 %, 0 MiB, no
    compute process before and after). The values agree with the pins'
    comments.
    (i) Seeds, V9(d): complex128 device seeds equal to `initial_guess`
    bitwise on every row: F1 12, F2-0, F2-1 and F3 64 each, F4 3, at
    upsample 16, 2 and 1, plus the dimmed F1 arm (12) and the runner
    arm (13) -> `GPU_SEED_EQUAL_COUNT` pinned AT the counts; complex64
    seeds differ on 0 rows of every fixture (max shift difference 0.0
    px) -> `GPU_SEED_C64_DIFF_COUNT = 0`.
    (ii) Seam, V9(e): planted rows on F1, F2-0, F2-1 and F3 at both
    device precisions and all four finite row types: 0
    iteration-count differences, 0 `converged` flips, every point
    converged on both (13 on F1, 65 on each 64-pattern set, the
    reference included) -> `GPU_SEAM_*` pinned AT the counts.
    Discriminating counts (CPU side, `SEAM_DISCRIMINATING_MIN`): F1
    12, 9, 12, 13; F2-0 64, 39, 63, 62; F2-1 64, 39, 65, 64; F3 64,
    47, 61, 62 (exact, translate_rotate, rotate_1p5, perspective).
    (iii) Bands, V9(f), each the worst over every gated use, pinned at
    about 2x: h band mixed 1.256e-6 px (the DC-offset arm, no
    band-pass; the parity fixtures at most 5.22e-7 px on F3) ->
    `GPU_PARITY_H_TOL_MIXED = 2.5e-6`; float64 3.19e-12 px (F2-1) ->
    `7e-12`. First step mixed 5.05e-7 px (the planted F3
    "perspective" seam row; 1.20e-7 px on F1 parity) -> `1e-6`;
    float64 1.58e-11 px (F3) -> `3.2e-11`. Residual relative 2.24e-6
    (DC offset, mixed) -> rtol `4.5e-6`, absolute 2.6e-14 on
    near-zero criteria -> atol `6e-14`. Fe 1.20e-9 (F6, mixed) ->
    `2.4e-9`. Kernel A/B at float64, scale-free relative: worst
    1.469e-14 (gather) -> `GPU_KERNEL_AB_TOL_F64 = 3e-14`. FLAGGED
    FOR THE REVIEW GATE: the mixed first-step pin (1e-6 px) sits above
    the spec gate's 5.3e-7 px measurement of M3 (f32 reductions),
    because the seam's one-iteration arm shares it; M3's kill must be
    verified by re-injection.
    (iv) Counts, V9(g): 0 iteration differences and 0 flips on every
    key (F1 to F4 and F6 at both device and seed precisions, the Ni
    map, the F4 knob arms, padded F4, F5, F7, DC offset) ->
    `GPU_ITERATION_DIFF_COUNT = GPU_CONVERGED_FLIP_COUNT = 0`.
    (v) Intensity scale, V9(j): 2.147e-8 px mixed, 3.058e-9 px
    float64 -> `4.3e-8`, `6.1e-9`. Update rule, V9(k): 9.06e-8 px mixed
    (one iteration), 1.025e-13 px float64 (0 after one iteration) ->
    `1.8e-7`, `2e-13`, about 450x under the 8.2e-5 px mutant
    separation of ledger 93.
    (vi) Drift tripwire, V9(l): the twelve F1 recovery literals per
    device precision, `GPU_DRIFT_RECOVERY_PX_MIXED` and `_F64`; the
    worst case moves from the CPU's 0.012439859 px by 5.7e-8 px
    (mixed) and 2.8e-14 px (float64); band frozen at the CPU half's
    1e-9 px. These literals also hold under NVRTC 12.9 (entry 105
    (iii)), so they are not specific to the compiler that measured
    them.
    (vii) Determinism, V9(m): B = 32, 40 and the default (64) equal
    B = 8 bitwise on F1 at both device precisions ->
    `GPU_BATCH_INVARIANCE_TOL = 0.0` (the F2 to F4 arms are entry 105
    (ii)). The launch-dimension spy passes at B = 8 and 32: every
    kernel's grid is (256,) or (1,) or (10,) blocks per pattern, a
    function of the pattern geometry only.
    (viii) Robustness, V9(n): under a real `set_limit` the runner
    builds B = 64, 32, 16, 8, 4 and recovers with a 0-byte pool
    residue on F1 (limit 92,422,144 B) and F7 (2,582,016 B) ->
    `GPU_LEAK_RESIDUE_BYTES = 0`.
    (ix) VRAM, V9(o): the calibration is entry 105 (i).

105. **The implementation gate's measurement debt (plan 11 item 3;
    E3, E4, E5, E6, E13 and the V9(o) calibration; integrator).**
    Machine A (i7-13700H, RTX 2000 Ada Laptop GPU 8 GB, driver
    595.71, Windows 11, on AC power, High performance scheme), the
    pinned overlay of D21.15 at `-n 0` unless stated. Scratch
    scripts `vram_calib.py`, `e_debt.py` (items E3, E4, E5, E13),
    `e5_rect.py`, `e6.py`, `e6c.py`, driven by `run_debt.sh`, which
    logs `nvidia-smi` (utilisation, memory, compute processes) before
    and after each item. Every `nvidia-smi` read between 04:39 and
    04:59 showed 0 %, 0 MiB and no compute process, so no GPU-side
    contamination. The host was NOT idle: a VS Code process took
    about one core and total CPU sat at 24 to 36 % with no job of
    ours running, and end-to-end times moved by up to 1.6x between
    sessions (the first integrator's runs at 02:13 to 02:40 against
    these). So every timing below is quoted as a range over the
    sessions, and only ratios taken within one session are used for a
    decision.
    (i) **VRAM model terms, calibrated SEPARATELY (V9(o)).**
    RECIPE: `vram_calib.py`; the pool high-water marks at 512x622
    (F3s; 318,464 pattern pixels, subregion and crop 257,600) through
    `_PoolHighWater`, (r) `build_resident` alone, `build_seed_state`
    alone and both; (g, p) whole F3s runs at B = 8, 16, 32 and 64,
    with g = (peak(64) - peak(32)) / 32 and p = (peak(16) - peak(8))
    / 8 - g, the test's recipe. 04:39, deterministic: identical to
    the byte with the 02:10 run and with the gated run's
    `record_property` values.

    | term | measured | model (`_vram_model_terms`) | pinned bounds |
    |---|---|---|---|
    | r, mixed, complex128 | 23,159,296 B peak (resident 10,794,496 + seed 12,364,800 peak; 19,037,696 held) | 30,638,080 B | (11.5e6, 46e6) |
    | r, float64, complex128 | 26,248,704 B (resident 13,883,904) | 38,281,216 B | (13e6, 52e6) |
    | r, mixed, complex64 | 16,978,432 B | 22,994,944 B | -- |
    | r, float64, complex64 | 20,067,840 B | 30,638,080 B | -- |
    | g, both precisions | 1,273,936 B | 2,613,248 (mixed), 3,887,104 (float64) | (0.6e6, 8e6) |
    | p, both precisions | 21,094,832 B | 30,572,544 B (complex128) | (10.5e6, 42e6) |

    Every model term is at least its measured mark, so ledger 103's
    open point (the old `r` model of 20.4 MB below the 23.2 MB mark)
    is closed, and every measured term sits inside its pinned (o)
    bounds: the bounds are CONFIRMED. The whole model bounds every
    whole run: peaks 197,988,864, 376,939,008, 734,839,296 and
    775,605,248 B (mixed, B = 8 to 64) against 326,762,496,
    592,248,832, 1,123,221,504 and 1,206,845,440 B. The `float64`
    arm of `test_batch_size_invariance` (103's other open point)
    passes in entry 106. RECORDED, not a defect: the E3 sweep below
    took pool peaks inside its timing loops (cuFFT plan caches alive),
    and there the peak came within 3 % of the model once (548.1 MB
    against 564.8 MB, mixed, B = 16, 260 points). The default B
    budgets the model against HALF the free VRAM, so that run still
    had about 2x headroom.
    (ii) **B invariance across B, E4 (D21.7.3).** RECIPE: `e_debt.py
    E4`: `run_fixture` at `chunksize` 8, 32, 40 and None (the default,
    64 here, read from a `_make_session` spy) on F1, F2-0, F2-1, F3
    and F4 at both device precisions (complex128 seeds), compared by
    `assert_properties_bitwise` against B = 8 and among B >= 32.
    04:40: BITWISE on all 10 fixture-precision pairs (max |h| difference
    0.0 at every B), and the same at 02:2x. E4 RESOLVED: bitwise, and
    `GPU_BATCH_INVARIANCE_TOL = 0.0` stands with no deviation. The
    alternative, a constant P = 32 at every B (11.4 item 23), is not
    needed.
    (iii) **NVRTC provenance and a wheel-only compile, E6 (D21.2).**
    RECIPE: `e6.py` (the HREBSD gate, then `make_kernel_namespace`,
    an FFT and a matmul, then the process module list through
    `EnumProcessModules`), and `e6c.py` (`RawKernel.compile()` of all
    five kernels at both precisions plus the spline prefilter: 11
    kernels).
    - On the pinned overlay as the gated suite runs it, NVRTC is
      NOT a wheel. The process loads `nvrtc64_130_0.dll` and
      `nvrtc-builtins64_131.dll` (NVRTC 13.1) from the CUDA Toolkit
      v13.1 on PATH (`C:\Program Files\NVIDIA GPU Computing
      Toolkit\CUDA\v13.1\bin\x64`), and the toolkit's `cufft64_12.dll`,
      `cublas64_13.dll` and `cublasLt64_13.dll` load beside the
      overlay wheels' `cufft64_11.dll`, `cublas64_12.dll` and
      `cublasLt64_12.dll`, which cupy's own `.pyd` modules use. The
      user-level `CUDA_PATH` points at the `nvidia\cuda_runtime`
      folder of another venv (`venvs\kikuchipy-gpu`, read only, never
      modified); cupy finds the CUDA headers there. Under `uv run
      --with`, `cuda.pathfinder` does not search the overlay's
      `site-packages`, so the overlay's own `nvidia-cuda-nvrtc-cu12`
      12.9.86 (installed transitively through `nvidia-cublas-cu12`)
      is never loaded. Every device pin in entry 104 was therefore
      measured with NVRTC 13.1.
    - Wheel-only, in a scratch venv (`uv venv`, kikuchipy installed
      from this worktree, every package pinned to the worktree
      `.venv`'s versions, cupy-cuda12x 14.2.0 and the five wheels of
      D21.15), with every PATH entry naming CUDA removed and
      `CUDA_PATH` unset: the gate FAILS at stage (c) with "Failed to
      find CUDA headers. Please install CUDA toolkit headers (e.g.,
      pip install cupy-cuda12x[ctk]) or specify CUDA_PATH". NVRTC
      itself came from the transitive `nvidia-cuda-nvrtc-cu12`; the
      missing piece is the headers. cupy 14.2.0 cannot derive a CUDA
      root from the CUDA 12 split wheel layout (`_get_cuda_path`
      returns None).
    - Adding `nvidia-cuda-runtime-cu12==12.9.79` (which ships
      `include/`) fixes it, with nothing else set: the gate passes,
      all 11 kernels compile, and the process loads only wheel DLLs
      (`nvrtc64_120_0.dll`, NVRTC 12.9, and `nvrtc-builtins64_129.dll`
      from `nvidia\cuda_nvrtc\bin`, `cufft64_11.dll`, `cublas64_12.dll`
      and `cublasLt64_12.dll`).
    - Compiler independence: the whole gated suite with `--weekly` in
      that wheel-only venv (NVRTC 12.9, `KIKUCHIPY_EXPECT_GPU=1`,
      `-n 0`, `nvidia-smi` idle) gives **457 passed, 0 failed, 0
      skipped** in 828 s. Every device pin of entry 104, the bitwise B
      invariance and the 1e-9 px drift literals included, holds under
      NVRTC 12.9 as well as 13.1.
    - FIX (D21.2's conditional: "the message gains
      `nvidia-cuda-nvrtc-cu12` if a wheel-only machine needs it
      named"): the stage-(c) message `_gpu._GATE_LIBRARY_MESSAGE`
      and the `backend` entry of the `hrebsd_dic` docstring now also
      name `nvidia-cuda-nvrtc-cu12` and `nvidia-cuda-runtime-cu12`
      ("NVRTC and the CUDA headers the kernels compile against"), and
      the test pin `GATE_WHEELS` gains both (a dated comment). DATED
      DEVIATION, 2026-10-07, for the review gate: D21.2 authorised
      only the NVRTC wheel; the runtime-headers wheel is added because
      the measurement shows it is the one a wheel-only machine
      actually lacks. The D21.15 overlay itself is unchanged.
      Re-pinning it with `nvidia-cuda-runtime-cu12` and running the
      gated suite with the toolkit off PATH would make the pins
      toolkit-independent by construction. That is a review-gate
      proposal, not done here (the overlay is frozen).
    (iv) **Default B sweep, E3 (D21.10.3).** RECIPE: `e_debt.py E3`:
    F3 tiled 4x (260 points, 512x622, synthetic, one grain), best of
    2 after a warm-up at B in {8, 16, 32, 64, 128}, complex128 seeds,
    with the pool peak of one more run. Free VRAM 7,426,015,232 B, so
    the chooser returns 64 at both precisions (model at 64: 1151 MB
    mixed, 1243 MB float64). Patterns/s over three sessions (02:2x,
    04:42, 04:53):

    | B | mixed | float64 | pool peak MB (mixed) | model MB (mixed) |
    |---|---|---|---|---|
    | 8 | 89, 56, 67 | 73, 56, 53 | 189 to 220 | 312 |
    | 16 | 92, 70, 69 | 74, 58, 58 | 485 to 548 | 565 |
    | 32 | 99, 83, 96 | 78, 69, 70 | 701 to 952 | 1071 |
    | 64 | **102, 89, 99** | **78, 70, 73** | 740 to 991 | 1151 |
    | 128 | 94, 84, 84 | 72, 66, 65 | 817 to 943 | 1310 |

    B = 64 is the fastest at both precisions in all three sessions,
    and B = 128 (above the frozen cap) is slower. E3 RESOLVED on
    synthetic data: the in-force rule and the cap of 64 stand. The
    sweep on far256 and patch C that E3 names is NOT run here. The
    Si-indent file sits only in the main checkout, which this
    integrator must not touch, so it moves to the D21.16 performance
    record (plan 11 item 7), where that data is read anyway.
    (v) **Grain-pure batching and the `R_MAX` bound, E5 (D21.9.3,
    D21.10.2).** RECIPE: `e_debt.py E5` (spies on
    `_GpuSession.residents` and `_batched.build_resident`) and
    `e5_rect.py`. F7 (18 grains of 4 points, 60 px) at B = 4, 8 and the
    default: at most **2** residents alive at any time, and **18**
    resident uploads, one per grain, with no re-upload, in both
    sessions. The `R_MAX = 2` bound holds. Many-grain against one-grain
    maps of the same size, best of 3, mixed, complex128:

    | map | default B (64) | B = 32 | B = 16 |
    |---|---|---|---|
    | 60 px, 20 grains x 20 points | 1.01, 1.72, 1.89 s | -- | 1.27, 1.71, 2.23 s |
    | 60 px, 1 grain x 400 points | 0.52, 0.82, 0.92 s | -- | 1.25, 2.03, 2.13 s |
    | 512x622, 8 grains x 20 points | 4.84 s | 3.93 s | 4.29 s |
    | 512x622, 1 grain x 160 points | 1.83 s | 1.84 s | 2.21 s |

    (Iteration totals agree: 1270 against 1287 and 748 against 754;
    all points converge.) Same-session ratio, many-grain over
    one-grain, at the default B: 1.96x, 2.10x and 2.05x at 60 px, and
    **2.64x at 512x622**. Even at the best B for the many-grain map
    (32) it is 2.1x. **E5 TRIGGER FIRED**: grain-pure batching with
    per-grain padding costs far more than the 20 % threshold on a map
    of 20-point grains. Two sources, both by design (D21.7.3): each
    grain runs at least one batch padded to B, and the padded slots
    go through the upload, preprocessing, spline prefilter and seed
    FFTs before being discarded; and each grain builds its own
    residents and seed state. Not changed here: multi-grain batches,
    skipping all-padding sub-batches, or a B per grain would each
    change frozen D21.7.3 and D21.9.3 behaviour and the V9(h)
    sub-batch-count pins. The decision goes to the review gate and
    Johan. Context: the Si-indent map (ledger 82) has few grains of
    thousands of points each, where this overhead is a few batches
    per grain against hundreds.
    (vi) **The complex64 seed's end-to-end gain, E13 (D21.1, D21.5).**
    RECIPE: `e_debt.py E13`: F3 tiled 4x (260 points, 512x622) and F2-0
    tiled 4x (260 points, 480 px) at the default B (64), best of 3
    after a warm-up; plus the seed stage alone (`seed_spectra` then
    `seed_homographies`) on one 32-slot sub-batch at 512x622, best of
    5. complex128 over complex64 time, three sessions:

    | case | mixed | float64 |
    |---|---|---|
    | F3 x4, 512x622 | 1.19x, 1.20x, 1.19x | 1.14x, 1.13x, 1.17x |
    | F2-0 x4, 480 px | 1.24x, 1.22x, 1.16x | 1.13x, 1.05x, 1.08x |
    | seed stage only, 32 slots 512x622 | 59.5, 58.6, 57.9 ms against 11.8, 11.7, 12.6 ms (about 5.0x) | -- |

    The seed stage alone is 5x faster at complex64, but end to end the
    gain on the synthetic 512x622 batch at the default B is **1.19x
    to 1.20x**, below the 1.5x of E13, so the public `seed_precision`
    rule does NOT fire and no public keyword is proposed on this
    record. The IC-GN loop dominates: about 2.1 s of 2.5 to 2.9 s on
    F3 x4. Mixed over float64 (complex128): 1.21x to 1.30x on F3 x4,
    1.30x to 1.53x on F2-0 x4. The synthetic batches run 90 to 100
    patterns/s mixed against the prototype's 395 on far256 (ledger
    97, device only), so the real-data figures and E13's whole-map
    record belong to the D21.16 performance record (plan 11 item 7),
    which may change this verdict.

106. **The implementation gate's runs and the CPU default path
    (D21.14, D21.15; plan 11 items 3 and 6; integrator).** Machine A,
    2026-10-07, worktree at 3146b82e plus the uncommitted
    integrator edits (entry 105 (iii)).
    (i) Default suite: `uv run pytest
    tests/test_indexing/test_hrebsd_gpu.py -n 0 -q -p no:cacheprovider`
    gives **186 passed, 0 failed, 271 skipped** in 291.3 s as pytest
    reports it, 5 min 4.9 s wall with `uv` start-up (04:03). A repeat
    with `--durations=12` (04:59, while the wheel-only gated run used
    the GPU and some CPU) gives 186 passed, 271 skipped in 315.1 s.
    **Default-suite wall time (D21.14.3, re-recorded as ledger 101
    asked): 291 s**, against 46.9 s for the failing-tests skeleton.
    The cost is the numpy twin at float64 on the 512x622 F3s: the
    slowest tests are the F3s and F1 seam arms (`TestSeedSeamHonours
    ArbitraryH0Numpy::test_planted_rows_reproduce_the_cpu_fit`, 9.8 to
    25.0 s each), `test_a_last_sub_batch_slot_equals_alone` (19.6
    s) and the F3s and F1 parity tests (18.5 and 13.1 s). Recorded for
    CI cost, with no budget in force. The 271 skips are the gated
    classes, skipped at stage (a) because cupy is absent from the
    CPU `.venv`, plus the canary.
    (ii) Gated suite: `KIKUCHIPY_EXPECT_GPU=1 uv run --with
    <the D21.15 overlay> pytest tests/test_indexing/test_hrebsd_gpu.py
    -n 0 -p no:cacheprovider -q -rs` gives **391 passed, 0 failed, 66
    skipped** (the 66 are "Needs --weekly") in 649.5 s, 11 min 33 s
    wall (04:12 to 04:24; `nvidia-smi` 0 %, 0 MiB, no compute
    process before and after).
    (iii) Gated with `--weekly` (same command plus `--weekly`, junit
    with `junit_family=legacy` for the `record_property` values):
    **457 passed, 0 failed, 0 skipped** in 816.4 s, 14 min 20 s wall
    (04:24 to 04:39; `nvidia-smi` idle before and after). The
    recorded throughput, best of 3 (D21.16, information only;
    patterns/s mixed c128 / mixed c64 / float64 c128 / float64 c64):
    F2-0 (65 points, 480 px) 91.1 / 99.1 / 74.2 / 83.8, F3 (65 points,
    512x622) 54.5 / 68.4 / 48.2 / 44.9. The same `--weekly` suite under
    the wheel-only NVRTC 12.9 venv: 457 passed (entry 105 (iii)).
    (iv) `uv run pytest tests -k hrebsd -n 2 -q -p no:cacheprovider`:
    **882 passed, 0 failed, 280 skipped** in 393.0 s (6 min 42 s wall).
    `-n 2`, not the `-n 4` of plan 11 item 6: the machine is shared
    and memory constrained, and xdist workers have died with
    `MemoryError` at `-n 4`. The skips are the gated classes under
    xdist (the structural `-n 0` rule) and the weekly arms.
    (v) **CPU default path bitwise unchanged.** The pre-Stage-D
    literal pins (`test_hrebsd_seeding.py`, `PRE_STAGE_D_PIN_TOL`)
    and the drift tripwire's CPU half pass in (iv), and
    `TestBackendSwitch::test_cpu_equals_no_keyword_bitwise_on_f6` and
    `..._on_the_ni_map` (`backend="cpu"` against no keyword, bitwise)
    pass in (i). By inspection of `git diff 49d8bbad -- _engine.py`,
    the only CPU-path change is that the unchanged chunksize lines
    (`estimate_chunksize`, the clamp, the information message) now
    sit in the `else` branch of `if use_gpu`, and the `_run_chunks`
    dispatch in the `else` branch of `elif use_gpu`.
    (vi) Fixes at this step: one, the E6 message and docstring wheel
    set (entry 105 (iii)); `TestAvailabilityGate` and the docstring
    test re-run green after it (28 passed, `-k "Gate or docstring or
    Message"`), the default suite re-runs green after it (186
    passed, 271 skipped in 198.0 s), and so do the gated classes it
    touches on the overlay (`-k "Gate or Canary or docstring"`: 227
    passed, 66 weekly skips, `nvidia-smi` idle at 05:21). No other implementation bug and no test bug was found:
    every red seen at this step was zero. `uv run ruff check` and
    `uv run ruff format --check` are clean on `_gpu.py`, `_batched.py`,
    `_cuda.py`, `_engine.py`, `signals/ebsd.py` and
    `test_hrebsd_gpu.py`.
    Still open for the review gate (plan 11 items 4 to 7), not
    measured here: the real-data parity on far256 and patch C (V9(q),
    E12), the D21.16 performance record with the go/no-go floor (E7),
    the E8 device-wait fraction, E11 (`--use_fast_math`), the
    combined coverage figure, and the `--fmad=false` option of the
    spline prefilter (`_cuda._SPLINE_OPTIONS`), a departure from a
    recorded default that the review must settle.

#### V9 recorded results, review gate (2026-10-07)

107. **Bug injection, M1 to M27 (plan 11 item 4; injector, run
    ALONE).** Machine A, 2026-10-07 05:50 to 06:55, worktree at
    3146b82e plus the uncommitted integrator edits of entry 106. Every
    touched file was backed up to the scratchpad with its md5 before
    the run; each mutant is the smallest edit realising its plan
    definition, applied by a harness that asserts the edit matches
    exactly, runs ONLY the designed killers, then restores the file
    from the backup and verifies its md5 before the next mutant (no
    git operation). Default killers: `uv run pytest <nodes> -n 0 -x`;
    gated killers: the D21.15 overlay with `KIKUCHIPY_EXPECT_GPU=1 -n
    0 -x --weekly`, `nvidia-smi` checked before every gated run (no
    compute process at any point). Baselines before injection: the
    40 default killer nodes 102 passed; the 30 gated killer nodes 155
    passed with `--weekly`. 42 variants, then 13 re-injections or
    re-checks.
    (i) **Killed (23 of 27 ids):** M1 (atomicAdd in the block write:
    `test_no_atomic_anywhere`; a cupy `.add.at` cross-block pass:
    `test_no_atomic_scatter_through_cupy`); M2 (definition:
    `test_the_signature_takes_the_pixel_count_only`; call site,
    blocks x max(1, B // 8): `test_launch_dimensions_do_not_depend_
    on_batch_size[mixed]`; a first call-site variant scaling by 8 // B
    was inert at the spy's B = 8 and 32, an injection flaw, not a
    survivor); M3 (f32 block tree and cross-block pass:
    `test_first_step_band[F1-mixed]`, 1.22e-6 > 1e-6, a THIN margin);
    M4 (`test_reduce_solve_update`, 5.5e-9 > 3e-14); M5 (twin:
    `test_own_iteration_counts_on_f5`, 4 != 0; device, converged slots
    left active: see (iv)); M6 (twin `test_parity_with_the_cpu[F1]`
    9.3e-4; device `test_converged_parity[F1-mixed-complex128]`
    9.3e-4); M7 (twin and device residual-is-final-criterion, 3.26e-5);
    M8 (twin `[nan]` flag; device, `bad = 1` removed:
    `test_planted_non_finite_coordinates_are_flagged[mixed]`); M10
    (seam called, rows discarded for an import-time copy: twin and
    device one-iteration arms, 6.24 px); M11 (the spy count 0 == 2 in
    both); M12, M13 (seed equal count 0 != 12 in both); M14 (target
    spectra c64: `test_spectra_and_rows_layout[complex128]`, device
    `test_device_output_contract[complex128]`; reference spectrum c64:
    `test_the_seed_state[complex128]`); M16 (the LAST n rows of the
    padded batch returned: device `test_padded_slots_never_reach_the_
    output[mixed]`; see (iv) for the twin); M17 (twin F7 map order
    0.187 px; device F7 h band 1.64); M18 (LRU entry popped whatever
    the grain: the twin identity spy, the device F5 contract); M19
    (DID NOT RAISE, both); M20 halving and floor variants (B - 1 at
    build: [8, 7, 6, 5, 4] and device [8, 7]; B - 1 mid-compute: [8,
    7] both; no B = 1 floor: [8, 4, 2, 1, 0] both); M21 (shim after
    (c), and shim before (a): the stage-order pin); M22 (FFT-only
    probe: twin and device stage-(c) records lack matmul); M23 (lock
    dropped: the twin's held-lock flags; default scheduler: the twin
    spy and the device `{None} == {'threads'}`); M24 (guarded
    module-scope import: the AST pin); M25 (raise after the gate and
    after reference resolution: "gate ran before the D21.12 raise");
    M26 (fallback: `test_gate_failure_raises_with_no_cpu_fallback`).
    (ii) **Reviewed-equivalent (1): M9**, the zero-norm guard dropped
    in the twin `solve_update` (M9a), the twin final criterion (M9b)
    and both device sites (M9dev); designed killers and the next ones
    (the F5 failure contract, the band-passed constant, the NaN-step
    test, the warning count, F1/F3s parity, `TestGatedBatchedSemantics`
    whole, the kernel A/B) all pass. Argument: a norm of exactly 0
    means every centred value squares to 0, so `centred / norm` is
    0/0 = NaN (or +-inf), the gradient is non-finite in every
    component, the triangular solves keep it non-finite (no finite
    value is recovered from a NaN or inf operand without a 0 * inf,
    itself NaN), and `bad_step` fails the slot through the same
    branch with the same flags and NaN `norm_dp`; in the criterion
    0/0 gives a NaN residual, which `run_lockstep` (and the device
    path) already fails through `~isfinite(residual)`. The explicit
    `norm > 0` guards are redundant belt and braces, kept.
    (iii) **Survived (3):**
    M15 (target crops not ZMN'd in `seed_spectra`): the dimmed arm
    passes in both namespaces and so do the seed-row tests. With a
    ZMN'd reference the cross-power DC bin is 0 whatever the target,
    and a positive scale cancels in `X / max(|X|, 100 eps)` up to
    ulps and the guard, so the ROWS are equal on these fixtures; but
    the SEAM OUTPUT is not (DC bin and Parseval norm differ), and the
    frozen contract says the spectra are `fft2` of the ZMN crop.
    Killer for the fixer: assert `seed_spectra` against `fft2` of a
    host ZMN crop (DC bin 0, unit Parseval norm) on an offset, scaled
    target, in both namespaces.
    M20 (pools not freed): `_free_pools()` removed from window (b)
    (M20c) and `free_all_blocks()` removed from `_GpuSession.close`
    (M20e) both pass `test_real_pool_limit_recovers_without_leak` and
    all of `TestGatedRobustness` (13 passed). Cause: the leak pin
    reads `pool.used_bytes()`, which `free_all_blocks` never changes
    (it releases CACHED free blocks, counted in `total_bytes()`), and
    in window (b) the `finally: session.close()` already freed the
    pool before `_free_pools`, so M20c alone is near equivalent.
    Killer for the fixer: after a run (plan caches cleared, no
    references held), assert `pool.total_bytes()` within the residue
    pin, which M20e fails.
    M27 (a shared helper perturbed): `x / 6.0` -> `x * (1.0 / 6.0)`
    in the x weights of `_bicubic_evaluate` and its gradient twin
    (M27a, the device's own form) and `centred / norm` -> `centred *
    (1.0 / norm)` in `zero_mean_normalize` (M27b; 2931 of 10 000
    values change by an ulp) survive the designed killers
    (`TestDriftTripwireCpuHalf`, `TestBackendSwitch::test_cpu_equals_
    no_keyword_bitwise_*`, `TestDefaultOffIsBitwiseUnchanged`) AND the
    whole of `test_hrebsd_interpolation.py`, `test_hrebsd_engine.py`
    and `test_hrebsd_seeding.py` (208 and 178 passed). Yet the CPU
    path moves: F1's twelve CPU fits differ from the unmutated ones
    in 20 (M27a) and 19 (M27b) of 96 homography entries, max 1.8e-15
    and 6.7e-16, while the recovery-error literals stay bitwise
    equal. Every CPU pin is at 1e-9 (`CPU_DRIFT_TRIPWIRE_PX`,
    `PRE_STAGE_D_PIN_TOL`) or compares the mutated path with itself,
    so "the CPU default path is BITWISE unchanged" is enforced only
    to 1e-9. Killer for the fixer: a bitwise literal pin of the F1 (or
    F6) CPU homographies (`np.array_equal` on float64 literals,
    measured at this commit).
    (iv) **Mutation-map corrections** (the killer named in the map is
    not the one that sees the mutant): M5 on the device: converged
    slots left active pass `test_alone_equals_batched_on_f5` (alone
    and batched are both mutated) and die at
    `TestGatedParity::test_converged_parity[F1-mixed-complex128]`
    (1.04e-5 > 2.5e-6). M16 in the twin: the designed
    `test_padded_slots_never_reach_the_output` passes because a
    padded zero pattern always fails the final criterion, so the
    leaked rows are NaN in both the clean and the planted (both
    mutated) runs; `test_a_last_sub_batch_slot_equals_alone` kills it.
    Gated arms that cannot fire, the default-suite killer standing:
    M21a gated (`test_gate_passes_and_caches` SKIPS, the gate failing
    on Windows when the probe precedes the DLL shim; only the canary
    would turn it red), M23a (`test_four_worker_lock_stress` passes
    without the lock), M26 (the gate passes on a real device).
    (v) Integrity: after the run all 20 recorded files (the 16
    `_hrebsd` modules, `signals/ebsd.py`, `test_hrebsd_gpu.py`,
    `validation.md`, `plan.md`) match their pre-run md5, and `git
    status` lists only the four files entry 106 left modified. The
    default suite (`uv run pytest tests/test_indexing/
    test_hrebsd_gpu.py -n 0 -q -p no:cacheprovider`, 06:50): **186
    passed, 0 failed, 271 skipped** in 191.6 s.

108. **Review-gate fixes, survivor killers and the closing runs (plan
    11 item 5; fixer).** Machine A, 2026-10-07 07:03 to 08:25,
    worktree at 3146b82e plus the uncommitted edits of entries 105,
    106 and this one. `nvidia-smi` showed 0 %, 0 MiB and no compute
    process before and after every GPU run below. Scratch under
    `scratchpad\fix\`. Disposition of every finding and survivor:
    plan 11.6.
    (i) **`--fmad=false` reverted (RF-E1, RC-R3).** The spline
    prefilter now compiles with `_KERNEL_OPTIONS` (`--std=c++14`
    only), so `--fmad` is at the NVRTC default for every kernel as
    D21.4 records; `_SPLINE_OPTIONS` is deleted and the comments say
    the kernel is the algebraic mirror-mode recursion, equal to
    scipy's coefficients up to f64 rounding. The reviewer's device
    measurement (`rev\fmad.py`, 32 random patterns each at 512x622,
    60x60 and 480x480 against host `spline_filter`): with the flag
    10.56 to 10.59 % of f64 values bitwise equal, RMS 6.86 to 6.94e-14,
    2 / 0 / 2 f32-cast mismatches in about 1e7 values; NVRTC default
    10.49 to 10.57 %, RMS 6.84 to 6.92e-14, 0 / 0 / 1. The flag bought
    no parity. Pin: `TestCudaSourcePins::test_fmad_stays_at_the_nvrtc_
    default` (no `--fmad` option literal in any `_hrebsd` source,
    `_KERNEL_OPTIONS == ("--std=c++14",)`, no `_SPLINE_OPTIONS`).
    (ii) **The float64 build divides by 6 (RF-E2).** `basis<T>`
    multiplies by the reciprocal `1/6` under `HREBSD_MIXED` only and
    divides by 6 in the float64 build, as `_bicubic_evaluate` does.
    (iii) **Sub-batches of padding only are skipped (RC-R1, E5;
    DATED DEVIATION of D21.7.3, plan 11.6).** `_gpu._fit_batch` runs
    the preprocessing, the seed seam and the lockstep over the first
    m slots only, m the smallest multiple of P (at most B) covering
    the last real slot. RECIPE `fix\e5_skip.py` (the entry 105 (v)
    maps; the pre-fix `_fit_batch` loaded from a copy of `_gpu.py` and
    swapped in; default B = 64; best of 3 after a warm-up; mixed and
    float64, complex128 seeds), times in s, pre-fix / post-fix:

    | map | mixed | float64 |
    |---|---|---|
    | 512x622, 8 grains x 20 points | 4.17 / 2.85 | 4.95 / 3.39 |
    | 512x622, 1 grain x 160 points | 1.63 / 1.50 | 2.38 / 2.18 |
    | 60 px, 8 grains x 20 points | 0.55 / 0.57 | 0.58 / 0.57 |
    | 60 px, 1 grain x 160 points | 0.27 / 0.29 | 0.28 / 0.30 |

    Every output (homography, residual, iterations, norm_dp,
    converged, Fe) is BITWISE equal pre-fix against post-fix on all
    eight runs. Many-grain over one-grain at 512x622: 2.56x to 1.91x
    (mixed), 2.08x to 1.56x (float64); at 60 px about 2x either way
    (per-grain resident and seed-state builds and per-batch launch
    overhead dominate). The E5 rule (20 %) is still exceeded; the
    remaining options (multi-grain batches, a B from the grain-size
    distribution) change D21.9.3 or D21.10.3 and go to Johan. Tests
    re-pinned for it: `TestBatchModel::test_a_small_map_runs_one_
    padded_batch` (three points at B = 64 reach the seam in ONE
    sub-batch of P = 32); the gated seam spy on F1 tiled to 40 points
    (`f1_tail_batch`), `GPU_SEED_EQUAL_COUNT['F1 runner', 16]` 13 ->
    **40** (40 of 40 measured); the V9(o) g and p calibration on
    one-batch prefixes of F3 (`f3_prefix_batch(B)`, B points at B)
    instead of F3s, whose padded slots are no longer allocated. A
    first try on the whole of F3 (65 points, several batches) gave
    run-to-run varying peaks (p 12.9 to 33.5 MB; one run failed
    `p_model >= p` at float64), the multi-thread cuFFT plan caches of
    entry 103; the one-batch prefixes give, twice, peaks IDENTICAL to
    the byte to entry 105 (i): mixed 197,988,864 / 376,939,008 /
    734,839,296 / 775,605,248 B, float64 201,078,272 / 380,028,416 /
    737,928,704 / 778,694,656 B, g 1,273,936 B and p 21,094,832 B at
    both precisions. The pinned (o) bounds stand.
    (iv) **The information message (RC-R4).** `get_info_message`
    gains `n_chunks` (CPU output unchanged); under `"gpu"` it is the
    grain-pure batch count `len(_gpu._batch_chunks(state_of_point,
    B))`. Test: F5 at B = 8 prints 2 chunks, not ceil(7 / 8) = 1.
    (v) **The public Notes (RC-R6)** now read "a few 1e-6 binned
    pixels of corner displacement on converged points of synthetic
    test patterns" (pinned band 2.5e-6 px, worst measured 1.256e-6);
    revisit after the V9(q) real-data parity.
    (vi) **Coverage tests (RC-R2).** Default: `TestRunnerEdgesNumpy`
    (13 tests, listed in plan 11.6) and the information-message test;
    gated: `TestGatedKernelAB::test_reduce_solve_update_takes_a_host_
    lockstep` (host and Fortran-order lockstep arrays converted in
    place, bitwise equal to the device-array update) and
    `::test_the_spline_prefilter_leaves_a_length_one_axis`. The three
    non-import-guard pragmas of `_gpu.py` are removed.
    (vii) **Survivor killers, re-injected (plan 11 item 4).** Harness
    `fix\inject_fix.py` (fresh backup with md5, the ledger 107
    mutant edits from `mutants.py`, ONLY the new killers, restore and
    md5 check; no git operation), 07:27 to 07:30; every restored file
    matched its md5.
    - M15 (crops not ZMN'd): KILLED by `TestSeedSeamContract::test_
      the_spectra_are_those_of_the_zmn_crops[complex128, complex64]`
      and its gated twin in `TestGatedSeedParity` (spectra against
      host `fft2` of the CPU-ZMN crop of F4 scaled by 3.5 plus 250;
      tolerance `SEED_SPECTRA_TOL * sqrt(n)`, 5.4e-13 and 5.4e-5):
      max difference 2.5e4 in all four.
    - M20c (window (b) pools not freed): KILLED by `TestGatedRobustness
      ::test_real_pool_limit_recovers_without_leak`, which now reads
      `pool.total_bytes()` as each halved session is built
      (`GPU_REBUILD_POOL_BYTES`, MEASURED 0 at all four rebuilds on F1
      and F7, pinned 0): 10,802,688 B (F1) and 1,093,632 B (F7) at the
      first rebuild. M20c is therefore not equivalent, as ledger 107
      thought: without `_free_pools` (its `gc.collect` and
      `free_all_blocks` after the traceback is dropped) the failed
      attempt's memory is still in the pool when B / 2 is built.
    - M20e (`close` never frees the default pool): KILLED by the same
      test's `pool.total_bytes()` after the run within
      `GPU_LEAK_RESIDUE_BYTES` = 0 (measured 0): 89,195,520 B (F1) and
      1,630,208 B (F7).
    - M27a, M27b: KILLED by `TestCpuHelpersBitwise` (default suite):
      `_bicubic_evaluate` and `evaluate` against an independent
      scalar transcription of the pre-Stage-E formula on 404 points
      (inside, on and far outside every edge), and
      `zero_mean_normalize` against `centred / norm`, both
      `np.array_equal`. Platform independent by construction (no
      literals).
    (viii) **Decision-critical numbers re-measured** (RECIPE: the gated
    `--weekly` suite with the `recpins` plugin of entry 104, 07:31 to
    07:44, 480 passed, `fix\rec_fix.jsonl`, compared pin by pin with
    `rec_gated.jsonl`): every `assert_within` measurement is inside
    its pin and every count equals its pin. The mixed bands are
    unchanged (the `--fmad` revert moved no mixed figure); the
    float64 bands narrowed with RF-E2, e.g. F2-1 h band 3.19e-12 ->
    0.86e-12 px, F2-1 first step 3.31e-12 -> 1.34e-12, the seam
    F2-1 float64 bands about 3.1e-12 -> 0.8e-12 (pins 7e-12 and
    3.2e-11 stand); the f64 gather A/B 1.47e-14 against 3e-14.
    `test_batch_size_invariance` passes (bitwise across B, both
    precisions), so E4 stands with the skip of (iii); the precision
    verdict (D17 amendment, mixed default) is unchanged.
    (ix) **Closing runs.**
    - Default: `uv run pytest tests/test_indexing/test_hrebsd_gpu.py
      -n 0 -q -p no:cacheprovider --cov=kikuchipy.indexing._hrebsd`:
      **205 passed, 0 failed, 275 skipped** in 203.8 s (186 + 19 new).
    - Gated: the D21.15 overlay, `KIKUCHIPY_EXPECT_GPU=1 -n 0
      --weekly`, `--cov-append`: **480 passed, 0 failed, 0 skipped**
      in 630.7 s (07:45 to 07:56).
    - `uv run pytest tests -k hrebsd -n 2 -q` (with `--cov-append`):
      **901 passed, 0 failed, 284 skipped** in 256.8 s; and at `-n 0`:
      901 passed, 284 skipped in 373.0 s.
    - Coverage, default + gated + `-k hrebsd` combined: every
      `_hrebsd` module **100.00 %** (`_gpu` 407, `_batched` 406,
      `_cuda` 88, `_engine` 397 statements; package 2423 / 2423), no
      `# pragma: no cover` left in `_gpu.py`, `_batched.py` or
      `_cuda.py` beyond import guards.
    - Oldest matrix (the entry 87 recipe, Python 3.10, numpy 1.23.0,
      numba 0.57, orix 0.12.1, scikit-image 0.21.0):
      **901 passed, 0 failed, 284 skipped** in 399.4 s; the gated
      classes skip at stage (a) there.
    - Full suite `uv run pytest tests -n 2 -q -p no:cacheprovider`:
      **5422 passed, 1 failed, 1531 skipped** in 741.8 s. The one
      failure is `tests/test_simulations/test_kikuchi_pattern_
      simulator.py::TestCalculateMasterPattern::test_shape` (an
      `np.allclose` of two master patterns, after 5 reruns), in code
      this work does not touch; re-run alone it passes (after 5
      reruns again), so it is a pre-existing flake, recorded, not a
      Stage E defect.
    - `uv run ruff check` and `ruff format --check` clean on the
      `_hrebsd` package, `signals/ebsd.py` and `test_hrebsd_gpu.py`.
    DEVIATION (dated, plan 11.6): `-n 2` in place of the `-n 4` of
    plan 11 item 6 for `-k hrebsd` and the full suite (shared,
    memory-constrained machine; `MemoryError` in xdist workers at
    `-n 4`). Not run here and still open: the hermetic overlay of
    RC-R5 (a D21.15 amendment for Johan), the second injection half
    (M28 to M53), V9(q)/E12 real-data parity, the D21.16 performance
    record with E7 and E8, E11, and the CHANGELOG and tutorial note.

109. **Bug injection, M28 to M53 (plan 11 item 4, second half;
    injector, run ALONE).** Machine A, 2026-10-07 08:30 to 09:30,
    worktree at 5d503491 (the WIP checkpoint of the Stage E review
    fixes of entry 108, clean tree). The first half's backups were
    stale (the fixer had changed `_gpu.py`, `_cuda.py`, `_engine.py`,
    `ebsd.py` and the test file), so every candidate file was backed
    up afresh with its md5 before the run; all 48 recipe anchors were
    re-checked against the current code (each found exactly once, none
    moved), and every killer node was checked against a `--weekly`
    collection. Same harness discipline as entry 107: smallest edit
    per variant, exact-match assertion, ONLY the designed killers
    (default `uv run pytest <nodes> -n 0 -x`; gated via the D21.15
    overlay, `KIKUCHIPY_EXPECT_GPU=1 -n 0 -x --weekly`), restore from
    the backup and md5 verified before the next variant (no git
    operation). `nvidia-smi` before every gated run: none, then from
    08:50 an idle Jupyter kernel (pid 19480, a CUDA context, 0 MiB,
    0 % utilisation); nothing here is a timing, so no run was
    repeated. Baselines on the unmutated tree: the 24 default killer
    nodes 48 passed; the 31 gated killer nodes 103 passed. 48 variants
    (device, numpy-twin and shared variants of the 26 ids, plus M52c
    added below), then 14 re-checks with the next-most-relevant tests
    (`-x` off). Tally: **37 killed (34 by the designed killers, 3 only
    by re-check), 7 reviewed-equivalent, 4 SURVIVED**.
    (i) **Killed by the designed killers:** M28 (device step_scale
    ignored and the norm taken from the unscaled step:
    `test_knob_arm[mixed-step_scale_0.5]` 9.7e-4 and 4.8e-4 > 2.5e-6;
    twin: `[step_scale_0.5]` 9.7e-4 > 5e-13); M29 (shared and device:
    the window residual, worst 0.0816); M30 device
    (`test_converged_parity[F1-mixed-complex128]` 0.0134); M31 (device
    1.09e-5, twin 1.08e-5); M32 (device 1.58e-5, twin 1.57e-5); M33
    loop (`[max_iterations_1]` 0.177; gated `test_max_iterations_one
    [mixed]`) and twin (`test_own_iteration_counts_on_f5` 1 != 0);
    M34 (default and gated `test_an_explicit_chunksize_wins`, a
    TypeError on the None chunksize); M35 (`test_the_p_and_r_terms_
    decide`; gated `test_per_slot_and_transient_terms[mixed]`); M36
    (`test_update_matches_the_hand_built_loop[mixed-1]` 2.5e-3 >
    1.8e-7, its sole killer as designed); M37, M38, M39 (device
    `test_first_step_band[F1-mixed]` 6.15, 4.41, 8.69 > 1e-6; twins
    `test_the_first_step[F1]` 6.10, 4.29, 8.18); M40 (window arm 0.229
    and 0.231); M41 (default: the window arm went vacuous; gated
    window arm); M44 (device `test_planted_far_coordinates_fold
    [float64-3e9]` 0.904 > 3e-14; twin `[3e9]`); M46b (gated
    `test_dc_offset_without_band_pass[mixed]`); M47 (seed equal count
    0 != 12 in both); M48 (shared and device: `min_step_1e-2` arm
    2.9e-4); M50 (`test_the_runner_calls_the_seam_once_per_sub_batch
    [40]`; gated `test_the_runner_reaches_the_seam_per_padded_sub_
    batch`; the designed `test_a_last_sub_batch_slot_equals_alone`
    passes, the slots are independent); M51 (`test_the_residency_
    bound_on_f7`; gated `test_real_pool_limit_recovers_without_leak
    [F7-3600-4]`, the MemoryError at B = 1); M52a gated (the seam
    spy); M52c (added: `_compute_map` hands the batch its pattern
    index in MAP order while the patterns stay in fit order, the
    reachable form of the M52 definition: `test_the_seam_carries_the_
    fit_order_on_f7`, gated `test_many_grain_map_order[*]`); M53
    (spy count 1 != 2 in both).
    (ii) **Killed only by re-check (designed killers blind):** M33
    device (retire at `it >= max_iterations - 1`): the 0 and 1 arms
    cannot see it (no kernel call at 0; at 1 both conditions hold
    after the first step); killed by `test_update_matches_the_hand_
    built_loop[*-2]`, `test_converged_parity[F6-*]` (iteration count
    1 != 0) and `test_f5_failure_contract_and_map_order[*]`. M42b
    device (K never updated): killed by `TestGatedKernelAB::test_
    reduce_solve_update` (shifts 9.3e-14 > 3e-14, the float64 build).
    M49a (corner-norm support from the whole pattern): the border arm
    passes, but the default border kills it (`test_parity_with_the_
    cpu[F1]` 1.04e-5 > 5e-13, gated `test_converged_parity[F1-mixed-
    *]` 1.03e-5). M52a on the default suite (the designed numpy
    killers pass; `TestSeedSeamContract::test_the_runner_calls_the_
    seam_once_per_sub_batch[40, 12]` kills it).
    (iii) **Reviewed-equivalent (7 variants):** M30 twin (W33
    renormalisation skipped in the float64 twin): the twin's gather
    divides by the full third row and `parameters_from_matrices`
    divides by W33, so a scaled carried matrix is the same projective
    warp; only f64 rounding differs (17 re-check tests pass at 5e-13).
    The device variant is killed because the mixed gather's
    displacement form assumes W33 = 1. M42 twin (K never updated, f64):
    K cancels algebraically from every centred moment and the norm;
    17 re-check tests pass. M43 device and twin (corner max swallowing
    NaN): a NaN in `dp` sets the separate non-finite-step flag first,
    so a NaN corner needs a FINITE step with an exact 0/0 projective
    corner or an inf/inf overflow (|dp| about 1e306), unreachable from
    sums bounded by f32 data (critic F6, the map's own caveat); 8
    gated and 9 default re-checks pass. Hazard noted: in the device
    mutant four NaN corners would leave `nd = 0` and report the slot
    converged. M45 device and twin (diagonal rebuilt as `1 + (m - 1)`):
    by Sterbenz `m - 1` is exact for m in [0.5, 2] and so is the sum,
    so the variant is bitwise identical below 50 % strain; any other
    "rebuilt from h" in f64 is the renormalisation M30 already covers,
    and the f32 form is M32. M52b (the sub-batch's index sorted): every
    batch is grain-pure and the fit list is a stable argsort of
    `np.flatnonzero(keep)` (`_engine.py` 1178, 1233), so it is already
    ascending inside a batch and the sort is the identity; M52c
    (above) is the reachable form and is killed.
    (iv) **SURVIVED (4 variants; for the fixer):**
    M42a (mixed solve kernel: next K kept in f64): every consumer casts
    K to f32 and the next update re-anchors on that cast, so K is off
    by at most half an f32 ulp and does not accumulate; the 3 designed
    and 12 re-check gated tests pass. The only mixed-build observable
    is the stored shift. Proposed killer: a mixed-build arm of
    `TestGatedKernelAB::test_reduce_solve_update` against the twin's
    `round_to_pixel_precision(..., "mixed")` (the A/B is float64
    only).
    M42c (K0 = 0 in the mixed build) and M49b (K0 the whole-pattern
    mean, border ignored): both only move the initial shift. K cancels
    algebraically, and after the first update K is the warped mean
    again, so only the first step's f32 rounding changes and IC-GN
    reaches the same fixed point (38 + 12 and 2 + 28 gated tests
    pass, the F4 DC-offset arm among them). Proposed killer: a
    default-suite pin of `_batched.initial_shifts` (K0 equals the mean
    over `mask_index`, f32-rounded under a mixed namespace), and/or a
    `max_iterations=1` arm on the F4 DC-offset fixture in the mixed
    build.
    M46a (band-pass always applied, identity transfer under
    `(None, None)`): a cuFFT round trip changes the targets by a few
    ulp, and not at all for the F5 integer constant (DC bin only, the
    M15 argument), so all 4 designed and 32 gated plus 18 default
    re-check tests pass; it is a performance defect (two FFTs per
    sub-batch). The designed default killers can never see M46: the
    numpy session's `_prepare_sub_batch` takes the host `preprocess`
    branch. Proposed killer: under `(None, None)` the seam spy's
    targets must equal the raw patterns as float64 BITWISE on the
    cupy session, or a spy on `session.fft.fft2` must record no call.
    Mutation-map corrections for the fixer: M33 (add `[*-2]` hand
    loop and F6 counts as killers), M46 (D killers are blind by
    construction), M49 (the default-border parity, not the border
    arm, kills the corner variant), M50 (only the seam-count spies
    kill), M52 (`TestSeedSeamContract::test_the_runner_calls_the_seam_
    once_per_sub_batch` kills the padded-index variant on the default
    suite).
    (v) **Closing.** After the last variant every backed-up file
    (the 16 `_hrebsd` modules, `signals/ebsd.py`, the test file,
    `validation.md`, `plan.md`, `CHANGELOG.rst`) matched its pre-run
    md5 (`md5sum -c`, all OK). Default suite once, `uv run pytest
    tests/test_indexing/test_hrebsd_gpu.py -n 0 -q -p no:cacheprovider`:
    **205 passed, 0 failed, 275 skipped** in 125.6 s. Logs and JSON
    per variant in the scratchpad `inject_results3/`.
110. **M28 to M53 fix round: survivor killers, the hermetic overlay
    check (RC-R5) and the closing runs (plan 11 items 4 to 6;
    fixer).** Machine A, 2026-10-07 09:20 to 10:05, worktree at
    5d503491 plus ledger 109 (uncommitted).
    (i) **Survivors killed (4 of 4).** Killers added to
    `tests/test_indexing/test_hrebsd_gpu.py` (7 test instances, no
    source change):
    - `TestBatchedCoreNumpy::test_initial_shifts_are_the_masked_mean
      [border 0.05, 0.1][float64, mixed]` (default suite, 4): F4
      targets, `_batched.initial_shifts` against the boolean-mask mean
      of each preprocessed target, rtol 1e-13 in float64; under a
      namespace carrying `precision="mixed"` the result must be
      f32-representable bitwise and within half an f32 ulp of that
      mean. Premise asserted: the masked means (0.72 to 1.25) differ
      from the whole-pattern means (0.035) by more than 0.5.
    - `TestGatedKernelAB::test_the_mixed_update_keeps_k_at_pixel_
      precision` (gated, 1): one mixed-build `reduce_solve_update` on
      the A/B inputs from a deliberately poor f32-exact K0; no slot
      fails, every K moves, and every updated K is bitwise
      f32-representable (D21.4).
    - `TestGatedKernelAB::test_no_band_pass_uploads_the_raw_patterns
      [mixed, float64]` (gated, 2): a real cupy session's
      `_gpu._prepare_sub_batch` on two F4-DC patterns returns the raw
      patterns as f64 BITWISE under `filter_cutoffs=(None, None)`
      (the host `preprocess` too); with `(0.05, None)` the targets
      change and match the host within 1e-9 relative.
    Re-injection with fresh backups (`inject_backup3\`, md5 list
    `pre_run4_md5.txt`, harness `inject4.py` + `mutants4.py`, restore
    from backup and md5 verified, no git; `nvidia-smi`: only the idle
    Jupyter kernel pid 19480, 0 %): M42a (mixed kernel keeps K + mean
    in f64) 1 failed, the representability assert; M42c (K0 = 0 in
    mixed) 2 failed (`[*-mixed]`, the half-ulp assert); M49b (K0 the
    whole-pattern mean) 4 failed (all arms); M46a (band-pass always
    applied) 2 failed (the bitwise raw-pattern assert, both
    precisions). Afterwards `md5sum -c` all OK. The new tests pass on
    the unmutated tree. Mutation-map corrections of ledger 109 (iv)
    and one row per mutant M28 to M53 are in plan 11.6.
    (ii) **Hermetic overlay check (RC-R5, Johan's waiver).** The
    D21.15 overlay plus `--with nvidia-cuda-runtime-cu12==12.9.79`
    (the version of entry 105 (iii)) and `--with
    nvidia-cuda-nvrtc-cu12==12.9.86`, every PATH entry naming CUDA
    removed and `CUDA_PATH`, `CUDA_PATH_V10_0`, `CUDA_PATH_V11_7` and
    `CUDA_PATH_V13_1` unset for that process only
    (`scratchpad\hermetic.sh`).
    - AS SPECIFIED (variant A) the gate FAILS at stage (c): "Failed
      to find CUDA headers". Cause: under `uv run --with` the overlay
      wheels sit in a layered `archive-v0\...\site-packages` on
      `sys.path`, but `cuda.pathfinder` 1.8.3 (used by cupy 14.2.0
      for NVRTC, cuFFT and the CUDA root) searches only
      `site.getsitepackages()`, i.e. the ephemeral
      `builds-v0\.tmp...` environment; `find_nvidia_header_directory
      ("cudart")` returns None and `load_nvidia_dynamic_lib("nvrtc")`
      raises. With `CUDA_PATH` set in-process to the overlay's own
      `nvidia\cuda_runtime` (variant B) the headers resolve and NVRTC
      is not found; adding the overlay's `nvidia\cuda_nvrtc\bin` to
      PATH (variant C) then fails on cuFFT. The kikuchipy DLL shim
      (`os.add_dll_directory`) does not help, since pathfinder
      searches by file existence and plain `LoadLibrary`. So the
      seven-wheel overlay is not hermetic through `uv run --with`
      alone; the wheel-only VENV of entry 105 (iii) is (pathfinder
      sees a real site-packages there).
    - WORKING FORM (variant D): the same command, toolkit off PATH and
      every `CUDA_PATH*` unset, run through
      `scratchpad\overlay_cuda_path.py`, which prepends every
      `<overlay site-packages>\nvidia\*\bin` (found on `sys.path`) to
      PATH inside the Python process before cupy is imported, then
      calls `pytest.main`. DLLs loaded (`EnumProcessModules` after the
      gate): `nvrtc64_120_0.dll` and `nvrtc-builtins64_129.dll` from
      the overlay's `nvidia\cuda_nvrtc\bin` (NVRTC 12.9,
      `nvrtc.getVersion()` (12, 9)), `cufft64_11.dll`,
      `cublas64_12.dll` and `cublasLt64_12.dll` from the overlay
      wheels, runtime 12090, driver 13020; no Toolkit DLL. For
      comparison the ordinary D21.15 overlay in this shell loads
      `nvrtc64_130_0.dll` (NVRTC 13.1) and also `cufft64_12.dll`,
      `cublas64_13.dll` and `cublasLt64_13.dll` from the CUDA 13.1
      Toolkit beside the wheels' `cufft64_11.dll` and
      `cublas64_12.dll`, with `CUDA_PATH` pointing at the
      `kikuchipy-gpu` venv's `nvidia\cuda_runtime`.
    - RESULT, variant D, `KIKUCHIPY_EXPECT_GPU=1 -n 0 --weekly`,
      split in two calls by `-k`: `TestGatedSeedSeam or
      TestSeedSeamHonoursArbitraryH0Numpy` **114 passed** in 334.5 s
      (09:44 to 09:50), the rest **373 passed** in 305.9 s (09:50 to
      09:56): **487 passed, 0 failed, 0 skipped** (480 + the 7 new
      instances). Every device pin, bitwise invariance and drift
      literal holds under wheel NVRTC 12.9 with no Toolkit and no
      `CUDA_PATH`. `nvidia-smi` before each part: 0 %, 332 MiB, only
      the idle pid 19480; no timing was taken.
    - For the D21.15 amendment (main session): the seven `--with`
      pins alone are not enough under `uv run --with`; the command
      also needs the overlay's `nvidia\*\bin` on PATH (or the gated
      run in a wheel-only venv as in entry 105 (iii)).
    (iii) **Closing runs.**
    - Default: `uv run pytest tests/test_indexing/test_hrebsd_gpu.py
      -n 0 -q -p no:cacheprovider`: **209 passed, 0 failed, 278
      skipped** in 156.0 s (205 + 4 new; 275 + 3 new gated skips).
    - `uv run pytest tests -k hrebsd -n 2 -q -p no:cacheprovider`:
      **905 passed, 0 failed, 287 skipped** in 207.1 s.
    - `ruff format` (1 file reformatted, the test file) and `ruff
      check` on `_hrebsd`, `signals/ebsd.py` and the test file: all
      checks passed, 18 files formatted.
    Logs: scratchpad `inject_results4\`, `hermetic_p1.log`,
    `hermetic_p2.log`, `fix4_default.log`, `fix4_hrebsd.log`.

#### V9 recorded results, performance record (2026-10-07)

111. **Conditions, the same-session CPU baseline and the far256 and
    patch C matrix; the go/no-go floor (requirements D21.16; plan 11
    item 7; E7).** Machine A as in entry 89 (i7-13700H, RTX 2000 Ada
    Laptop GPU 8 GB, driver 595.71, Windows 11), 2026-10-07 10:05 to
    12:11, worktree at 5d503491 plus the uncommitted Stage E edits.
    GPU runs on the D21.15 overlay (the toolkit NVRTC 13.1 of entry
    105 (iii)), CPU runs on the worktree `.venv`. RECIPE (scratchpad
    `perf\perf_run.py`, results `perf\results.jsonl`, per-point arrays
    `perf\arrays\*.npz`): the executed `hrebsd_si_indent.ipynb` load,
    cells 4 to 10 and 17 reproduced (lazy `kp.load`, rechunk `{0: 1,
    1: 32, 2: -1, 3: -1}`, `detector.binning = 2`, the per-point PC,
    the `CrystalMap` from the file's Euler angles and the Si phase,
    reference (10, 10), `filter_cutoffs=(None, None)`,
    `max_iterations=500`, crater mask CCC < 0.35,
    `dask.config.set(num_workers=8)`), through the PUBLIC
    `EBSD.hrebsd_dic`; the CPU at the tutorial's `chunksize=16`, the
    GPU at `chunksize=None` (the default B, 64 at every run here). The
    engine-only precisions reach the engine by replacing
    `kikuchipy.signals.ebsd.run_hrebsd_dic` in-process with
    `functools.partial(run_hrebsd_dic, device_precision=...,
    seed_precision=...)`, so every other argument is the method's own.
    far256 = rows 20:36, cols 20:36 (256 points); patch C = rows
    115:140, cols 100:130 minus the crater (618 points; ledger 84).
    `nvidia-smi` before every timing: 0 %, and the only compute
    process an idle Jupyter kernel of another session (pid 19480,
    `ipykernel_launcher`, 332 MiB, 0 %) until 11:23, none after; no
    run was contaminated and none was repeated for that reason. Host
    load (psutil, 1 s sample before each timed run): 0.2 to 20.4 %
    (the 20.4 % before the first mixed complex64 map run, entry 112).
    GPU rows: one warm-up, then best of 3, each combination in its
    own process. CPU far256: one warm-up then 3 runs in one process.
    CPU patch C: three runs, each its own process (about 5 min each),
    the first doubling as the warm-up (the numba cache was already
    warm on disk from the far256 process); recorded as such.

    | patch | backend, device / seed precision | runs (s) | best | patterns/s | vs same-session CPU |
    |---|---|---|---|---|---|
    | far256 | CPU, 8 workers | 15.94, 16.28, 15.92 (warm-up 16.98) | 15.92 s | 16.08 | 1.00x |
    | far256 | GPU mixed / complex128 | 1.660, 1.947, 1.982 | 1.660 s | 154.2 | 9.59x |
    | far256 | GPU mixed / complex64 | 1.392, 1.427, 1.540 | 1.392 s | 183.9 | 11.44x |
    | far256 | GPU float64 / complex128 | 2.973, 3.084, 3.258 | 2.973 s | 86.1 | 5.35x |
    | far256 | GPU float64 / complex64 | 2.884, 2.936, 2.760 | 2.760 s | 92.7 | 5.76x |
    | patch C | CPU, 8 workers | 285.12, 290.79, 307.81 | 285.12 s | 2.17 | 1.00x |
    | patch C | GPU mixed / complex128 | 10.859, 10.877, 10.972 | 10.859 s | 56.9 | 26.3x |
    | patch C | GPU mixed / complex64 | 9.739, 9.787, 9.533 | 9.533 s | 64.8 | 29.9x |
    | patch C | GPU float64 / complex128 | 79.932, 79.668, 79.529 | 79.529 s | 7.8 | 3.59x |
    | patch C | GPU float64 / complex64 | 79.461, 79.573, 79.160 | 79.160 s | 7.8 | 3.60x |

    Every run: far256 256/256 converged, 2231 iterations (median 9, max
    10), residual median 0.04269; patch C 618/618 converged, median 76,
    max 497 iterations, residual median 0.11426, iteration total 54032
    (CPU and GPU float64) and 54033 (GPU mixed, entry 113). The CPU
    patch C (285 to 308 s) ran faster than ledger 84's 338.3 s; the
    three CPU patch C runs are bitwise identical to each other.
    **GO/NO-GO FLOOR (E7): PASS.** Best GPU on far256 at the default
    precisions (mixed, complex128) 154.2 patterns/s against the
    same-session 8-worker CPU's 16.08: 9.6x; every one of the four
    combinations clears it (lowest 86.1 patterns/s, float64
    complex128, 5.35x).
    Against the prototype expectation (ledger 97, DEVICE ONLY): far256
    395 and 947 patterns/s, patch C 212 and 300 (mixed). Shortfall
    explained by two measured terms. (a) Per-call fixed costs: on
    far256 the device sections (`_gpu._device_batch`, 4 batches) are
    busy 0.79 s (mixed complex128, about 324 patterns/s device only)
    and 0.41 s (complex64, about 624), the rest of the 1.4 to 2.0 s is
    the 0.37 to 0.45 s before the first device section (graph, first
    read) plus about 0.55 s outside `_compute_map` (session build,
    reference state). (b) LOCKSTEP WITHOUT RETIRED-SLOT SAVINGS: each
    batch iterates until its slowest slot finishes, and by inspection
    `_batched.run_lockstep` calls `kernels.gather` and
    `kernels.pixel_sums` without the active mask, so a retired slot
    costs what an active one does (the prototype's retired slots were
    nearly free, entry 91). Patch C's 10 batches run 2703 lockstep
    iterations, 172992 slot-iterations at B = 64 for 54033 real ones
    (lockstep efficiency 0.31; far256 0.89). Per lockstep iteration at
    B = 64: mixed about 3.5 ms (9.34 s busy / 2703), float64 about 29
    ms (78.3 s / 2703), i.e. 0.45 ms per slot-iteration in float64
    against the prototype's 0.270 ms per ACTIVE pattern-iteration.
    That is why float64 patch C runs at 7.8 patterns/s against the
    ledger 98 projection of about 39. Recorded as a D21.18 follow-up
    candidate (mask retired slots in `gather` and `pixel_sums`, or
    compact or refill slots), not changed here. The E3 B sweep on
    far256 and patch C (entry 105 (iv)) was NOT run in this record.

112. **The whole map on the GPU and the E8 device-wait fraction
    (requirements D21.11, D21.16; plan 11 item 7; E8).** Same recipe
    and conditions as entry 111, all 57772 fitted points (728
    masked), one run per combination in the chain `perf\maps.sh`
    (10:44 to 11:49), then one repeat of each mixed combination with
    a host-memory sampler (psutil RSS every 0.5 s; 11:59 to 12:11).
    "Compute" is `_gpu._compute_map` (graph build plus the dask
    compute); "busy" is the summed wall time of the 903
    `_gpu._device_batch` sections; wait fraction = 1 - busy /
    compute. CPU reference: ledger 82, 8424 s (2.34 h, 6.86
    patterns/s) on 8 workers, 2026-09-08, NOT same-session.

    | device / seed | wall | patterns/s | vs CPU (ledger 82) | conv. / not | its. median / max / total | residual median | first section after | wait fraction (after first section) | host load |
    |---|---|---|---|---|---|---|---|---|---|
    | mixed / complex128 | 375.7 s | 153.8 | 22.4x | 57685 / 87 | 14 / 500 / 1110801 | 0.046258 | 76.1 s | 21.7 % (0.44 %) | 0.8 % |
    | mixed / complex128, repeat | 358.8 s | 161.0 | 23.5x | 57685 / 87 | 14 / 500 / 1110801 | 0.046258 | 64.6 s | 19.3 % (0.4 %) | 6.8 % |
    | mixed / complex64 | 261.7 s | 220.8 | 32.2x | 57685 / 87 | 14 / 500 / 1110804 | 0.046258 | 60.4 s | 24.4 % (0.59 %) | 20.4 % |
    | mixed / complex64, repeat | 235.4 s | 245.4 | 35.8x | 57685 / 87 | 14 / 500 / 1110804 | 0.046258 | 43.4 s | 19.8 % (0.6 %) | 8.7 % |
    | float64 / complex128 | 1553.6 s | 37.2 | 5.42x | 57685 / 87 | 14 / 500 / 1110803 | 0.046258 | 57.7 s | 3.9 % (0.08 %) | 4.8 % |
    | float64 / complex64 | 1489.7 s | 38.8 | 5.65x | 57685 / 87 | 14 / 500 / 1110806 | 0.046258 | 65.2 s | 4.7 % (0.16 %) | 3.4 % |

    AGGREGATE PARITY WITH LEDGER 82 (E12's map-level clause): every
    run converges 57685 points and leaves 87 (all at the 500 cap),
    iterations median 14 and max 500, residual median of converged
    points 0.04626, exactly ledger 82's counts at all four
    combinations; iteration totals 1110801 to 1110806 (prototype
    1110802 and 1110805, ledger 97). So the map-level clause of the
    E12 trigger does NOT fire. Between GPU runs (`perf\mapcmp.py`):
    complex64 against complex128 differ at 7 points by one iteration,
    39 homographies are not bitwise equal, 12 differ by more than 1e-4
    px (max 5.98e-4 px, below `min_step`), 0 convergence flips, at
    both device precisions, the ledger 94 numbers exactly; mixed
    against float64 (complex128): 6 points differ by one iteration, 0
    flips, 6 points over 1e-4 px (max 8.5e-4 px), median corner
    difference 4.2e-8 px.
    Against the expectation per precision (D21.16): mixed 347 and 628
    patterns/s (prototype, dedicated reader, B = 128) against 154 to
    161 and 221 to 245 measured, 2.2x to 2.6x short; float64 about
    131 and 158 (projection) against 37.2 and 38.8, 3.5x to 4.1x
    short. Both shortfalls are the two terms of entry 111: lockstep
    efficiency 0.41 on the map (903 batches, 42292 lockstep
    iterations, 2706688 slot-iterations for 1110801 real; 20 batches
    hold a point at the 500 cap and run all 64 slots for 500
    iterations), about 35 ms per lockstep iteration in float64
    (1490.5 s busy / 42292) and about 6.9 ms in mixed, and the
    start-up wait below. A shortfall is a pass with the gap explained
    (D21.16).
    **E8: the device-wait fraction of the v1 read route on the default
    combination (mixed, complex128) is 21.7 % and 19.3 % (complex64
    24.4 % and 19.8 %), above the about-10 % threshold, so the E8
    follow-up (dedicated reader, pinned or double-buffered upload) is
    TRIGGERED.** The wait is NOT streaming starvation: after the
    first device section the device is busy 99.4 to 99.9 % of the
    time; all of the wait is the 43 to 76 s before the first device
    section starts. Cause, probed (`perf\e8_probe.py`,
    `perf\e8_order.py`, CPU only, no GPU): the gather graph
    (`s.data.reshape(-1, 512, 622)[fit_indices].rechunk(64)`, 128972
    tasks) builds in 0.13 s and the threaded scheduler is running in
    5 s, but it completes the 903 batch blocks in an order unrelated
    to batch number (median position error 565); batch 0 is the
    1772nd of 1806 matched block completions (the probe's callback
    matched two per batch), at 43.3 s of a 43.8 s read.
    `_OrderedDevice` runs batches strictly in order, so the device
    waits for nearly the whole map to be read, and the blocks that
    arrive early are held in `pending`: the process RSS peaked at
    **15.9 GB** (complex64 repeat) and **15.2 GB** (complex128 repeat)
    on this 32 GB machine, against 2.4 to 3.5 GB when the first
    section started. That host-memory peak is a SECOND finding for the
    review gate (a whole-map run holds most of the 18.4 GB map in RAM;
    on a smaller host or a larger map it would fail), and a fix to
    the read order (a reader that walks batches in order, or bounding
    `pending`) addresses both. Nothing changed here.

113. **Real-data per-point parity, V9(q) and E12 (requirements D21.8;
    plan 11 item 7).** RECIPE: `perf\parity.py` on the per-point arrays
    of the first timed run of each entry 111 row, GPU against the CPU
    (far256 against `far256_cpu`, patch C against the first CPU run,
    which is bitwise equal to the other two); the homography metric is
    the V2 corner displacement of `W(h_cpu)^-1 W(h_gpu)` over the
    PC-centred subregion corners of each point's own PC (border 0.05);
    the bands are the pinned V9(f) constants `GPU_PARITY_H_TOL_MIXED =
    2.5e-6` px, `GPU_PARITY_H_TOL_F64 = 7e-12` px,
    `GPU_PARITY_RESIDUAL_RTOL = 4.5e-6` with `ATOL = 6e-14`.

    | patch | device / seed | both conv. | flips | its. differing (max) | corner max / median, px | max abs h component diff / median | over the h band | residual rel. max | over residual band |
    |---|---|---|---|---|---|---|---|---|---|
    | far256 | mixed / c128 | 256/256 | 0 | 0 | 4.90e-8 / 2.11e-8 | 1.26e-8 / 4.3e-9 | 0 | 1.04e-8 | 0 |
    | far256 | mixed / c64 | 256/256 | 0 | 0 | 4.90e-8 / 2.11e-8 | 1.26e-8 / 4.3e-9 | 0 | 1.04e-8 | 0 |
    | far256 | float64 / c128 | 256/256 | 0 | 0 | 3.60e-13 / 1.27e-13 | 9.1e-15 / 1.5e-15 | 0 | 5.7e-15 | 0 |
    | far256 | float64 / c64 | 256/256 | 0 | 0 | 3.60e-13 / 1.27e-13 | 9.1e-15 / 1.5e-15 | 0 | 5.7e-15 | 0 |
    | patch C | mixed / c128 | 618/618 | 0 | 1 (+1) | **7.33e-4** / 6.02e-7 | 2.30e-5 / 1.47e-7 | **1** | 1.53e-6 | 0 |
    | patch C | mixed / c64 | 618/618 | 0 | 1 (+1) | **7.33e-4** / 6.02e-7 | 2.30e-5 / 1.47e-7 | **1** | 1.53e-6 | 0 |
    | patch C | float64 / c128 | 618/618 | 0 | 0 | **6.61e-11** / 1.80e-13 | 2.6e-12 / 3.6e-15 | **1** | 3.3e-12 | 0 |
    | patch C | float64 / c64 | 618/618 | 0 | 0 | **6.61e-11** / 1.80e-13 | 2.6e-12 / 3.6e-15 | **1** | 3.3e-12 | 0 |

    On both patches the complex64 GPU outputs are BITWISE equal to the
    complex128 ones at each device precision (homographies and
    iteration counts), so the seed precision changes nothing on these
    874 points (the map-level complex64 differences of entry 112 lie
    elsewhere). The two exceedances, one point each:
    - mixed, point (125, 100): the CPU stops at 51 iterations with
      `norm_dp` 9.9988e-4, just under `min_step = 1e-3`; the GPU's
      step there is a last-bit above it, so it takes one more
      iteration (52, final `norm_dp` 7.34e-4) and lands 7.33e-4 px
      away, the D21.8(e) mechanism (an exit-test flip into +-1
      iteration), not a kernel error. Every other point is within
      1.47e-6 px (point (124, 108)), inside the 2.5e-6 band.
    - float64, point (130, 109): 84 iterations on both, `norm_dp`
      agreeing to 1.7e-13, corner difference 6.61e-11 px, 9.4x the
      7e-12 band, which was pinned on synthetic fixtures of at most
      tens of iterations; the next worst point is 1.88e-12 px (378
      iterations, within the band). Accumulated rounding over a long
      real-data fit, 6.6e-11 px in absolute terms.
    No residual exceeds its band, no convergence flag flips, and the
    far256 patch is clean at every combination.
    **E12 VERDICT: the trigger FIRES by its letter** ("a disagreement
    beyond the V9(f) bands there"): one of 618 patch C points exceeds
    the h band at each device precision. Per the instruction for this
    record the 2.34 h CPU map re-run was NOT started; the decision goes
    to the main session and Johan, with the two diagnoses above (one
    exit-test iteration flip under mixed; one long fit 9.4x over a
    synthetic-pinned float64 band) and the map-level aggregates of
    entry 112, which equal ledger 82 exactly.

114. **E13 at whole-map level, and the record's verdicts (requirements
    D21.1, D21.16; plan 11 item 7; E13).** From entry 112, same
    machine and session, end-to-end wall time:
    - complex64 over complex128, device precision mixed: **1.44x** on
      the first pair (375.7 / 261.7 s, the complex64 run under 20.4 %
      host load) and **1.52x** on the repeat pair (358.8 / 235.4 s);
      best over best 1.52x; device-busy ratio 1.48x and 1.54x.
      Float64: **1.04x** (1553.6 / 1489.7 s, one pair). So under mixed
      the gain sits AT the about-1.5x line (1.44x to 1.52x, borderline,
      not clearly above it); under float64 it is far below. On the
      synthetic F3 x4 batch it was 1.19x to 1.20x (entry 105 (vi)).
    - mixed over float64 (complex128): **4.13x** (1553.6 / 375.7 s) and
      4.33x against the repeat; complex64 6.33x. This one is well above
      1.5x.
    E13's measured rule ("a gain above about 1.5x on the record making
    a public keyword ... the proposal"): it is MET for
    `device_precision` (mixed over float64, 4.1x to 4.3x) and
    BORDERLINE for `seed_precision` (1.44x to 1.52x under mixed, 1.04x
    under float64). Recorded only; the main session decides any public
    keyword.
    Verdicts of this record: go/no-go floor PASS (9.6x on far256);
    E12 trigger fires on patch C (one point per device precision,
    diagnosed above; no CPU re-run started); map aggregates equal
    ledger 82 at all four combinations; E8 triggered (19 to 24 %
    start-up wait from out-of-order dask reads, with a 15 to 16 GB
    host RSS peak); E13 met for the device precision, borderline for
    the seed precision; two performance follow-up candidates
    (retired-slot masking in the lockstep, in-order reading) recorded
    for D21.18. No repository source or test file was changed by this
    record.

### V10 -- Fourier-Mellin oracles (Stage F, 2026-10-07)

Requirements D22 govern. One new file,
`tests/test_indexing/test_hrebsd_fourier_mellin.py`, laid out like
`test_hrebsd_gpu.py` (dated pin constants; the default-suite classes;
the gated ones). Test modules cannot import each other, so the
helpers are DUPLICATED (the D21.14.1 rule), each copy carrying a
comment that names its source and line range at f297867e and kept
identical to it (a drift is a test edit with a dated comment): from
`test_hrebsd_gpu.py`, the gate and fake-cupy helpers
(`_cupy_gpu_skip_reason`, `cupy_gpu`, `make_fake_cupy`,
`install_fake_cupy`, `hide_cupy`, `fresh_gate`; lines 585-753), the
assertion helpers (754-796), the oracle helpers (`matrix_of` to
`deformed_pattern`; 797-991), the V9 F6 map and the Ni inputs
(`f6_map`, `ni_inputs`, `run_ni`, `run_engine`; 1224-1347) and the
numpy-session helpers (`make_state` to `run_gpu_numpy`; 1376-1538)
and, where an arm needs a V9(d) seed fixture, `seed_case` and
`numpy_seed_rows` (2165-2194); from `test_hrebsd_seeding.py`, the V8 ramp map (the `RAMP_`
constants, lines 170-203; `in_plane_fe`, `out_of_plane_fe`,
`orientation_a`, `deformed_pattern`, `oracle_reference`, `ramp_map`;
713-830); about 1000 lines in all (plan 12.4 item 16 records the
alternative). Expensive runs are cached with `functools.lru_cache`
(the V8 precedent). Three edits of existing pins, each with a dated
comment: `FROZEN_SIGNATURE` in `test_ebsd_hrebsd_dic.py` (D15.4
amendment), `test_run_defaults_are_frozen` in
`test_hrebsd_engine.py`, and `SEED_FROM_NEIGHBORS_GPU_MESSAGE` in
`test_hrebsd_gpu.py` (D21.12 amendment). No existing docstring test
changes; the docstring is pinned by the NEW
`TestFourierMellinSwitch::test_the_docstring_documents_fourier_mellin`
(D22.16). The CPU route of D22.10 is the oracle for every device
output (D22.14). Every band below is MTP: the failing-tests commit
carries `FIXME-pin` placeholders (the V9 convention); CPU-side
literals are measured at the failing-tests gate, device-side ones at
the implementation gate, each pinned at about 2x margin with the
recipe and machine ID beside the constant, and where a pin must also
kill a mutant the kill separation is stated beside it. The
spec-gate numbers quoted (ledger 115 to 131) are the scales a
correct implementation should reproduce, not pins; they were
measured on throwaway prototypes, and the frozen recipe itself only
in ledgers 128, 129 and 131. The Si-indent file is never a test
fixture: its numbers are RECORDED measurements, (m) below.

Fixtures, synthetic and generated in the test with fixed seeds:
**G1** the V3/V4 deformed-master 480 px oracle at `PC_480` (0.4210,
0.5794, 0.5049), `border=0.05` (in-frame to 5.62 deg) and
`border=0.15` (in-frame to 23.8 deg; ledger 130); **G2** the V8
projection centre (0.49, 0.51, 0.5049) and the V8(a) ramp map
(`ramp_map`; in-frame to 6.62 deg); **G3** a 512x622 synthetic at
`PC_480`, crop 460x560 at `border=0.05` (the V9 F3s shape), for the
physical-frequency pin (ledger 131); **G4** G1 under a detector-fixed
background (an off-centre Gaussian plus a linear ramp, the same image
multiplying reference and target, ledger 119), noise-free and at a
Poisson full-scale count of 50, both at `filter_cutoffs=(None,
None)` and `(0.05, None)`; **G5** the anchor pair: G1 reference and
target carrying the SAME fixed-pattern image (detector-fixed WHITE
per-pixel gain and offset noise, not banded noise), the target with
a 10 to 20 px projection centre shift and a 1.5 to 3 deg twist, so
the D5 seed is pulled to (0, 0) as on the real rim (ledger 122) --
NOT built at the spec gate: the fixed-pattern amplitude is set at
the failing-tests gate until BOTH run-time premises hold, (i) the D5
seed is exactly (0, 0) and (ii) `theta_hat` is within
`FM_ANGLE_TOL_DEG` of the imposed twist (fixed-pattern energy is
identical and unrotated in both FM spectra and can pull the angle to
0, too), and recorded; if no amplitude meets both, the G5 arms of
(g) are dropped with that record, and the FQ8 anchor census of (m)
covers the real-data case; **G6** an unrelated pair at the V8 PC (a
second orientation 20 deg about (0, 1, 1), a grain boundary): at a
budget of 200 the translation-seeded fit does NOT converge and keeps
a FINITE `h` (residual 1.89), the acceptance refuses the FM row
(criterion 2.080 against 2.040) and a forced FM-seeded fit does not
converge either (1.66; ledger 131, correcting ledger 119(iv)); **G7**
gate maps: orientations built as `g_t = g_r @ (M^T R_det M).T` for
imposed detector-frame rotations `R_det` (ledger 127), with a
symmetry-equivalent copy, about 0.5 deg orientation noise, an
unindexed point, a NaN rotation, a second phase, a two-grain map, an
all-identity map and a constant NON-identity map (every point at the
reference's own orientation), on the default detector and on a
tilted one (tilt 10, azimuthal 4, twist 1.5 deg); **G8** a retry map
on the V8 geometry at `MAX_ITERATIONS = 200`: one point whose pattern
carries a 4.0 deg twist while its `xmap` orientation reads the
reference's (so the pre-fit gate does not route it; premise asserted
at run time: the default seed does not converge at 200 and keeps a
finite `h`, measured 42.9 px away, while the FM row converges in 3
iterations; ledger 131 -- 3.0 deg is NOT usable, the default seed
converges there in 14), one G6 unrelated-pattern point whose `xmap`
also reads the reference's orientation (premises asserted at run
time: first pass not converged with a finite `h`; a forced FM-seeded
fit not converged), a constant pattern (the D2.6 contract), a masked
point and two easy points (0 and 0.8 deg).

CI cost, per class, fits at 480 px (60 px on the Ni map): (a) about
six Ni-map runs and two V9 F6 runs; (b), (c) and (h) no fit; (d) one
fit per planted-failure arm at the 50 budget; (e) the five starred
arms twice each (the `"off"` premise and the FM run), at most 50
iterations; (f) the ten fitted ramp points three times (the `"off"`
premise, `"always"`, `"auto"`), about five of them capped at 200 on
the premise; (g) about six; (i) the six fitted G8 points under two
modes plus two direct fits; (j) about six runs of a six-point map;
(k) about four numpy-session runs. About 100 fits in all, of which
about 15 run to a 50 or 200 cap. The default-suite wall time is
recorded at the failing-tests gate (the D21.14.3 precedent) and the
weekly marker takes any class that exceeds the Stage E per-class
envelope.

(a) **API, default path, check order, forwarding, docstring**
[default] (D22.1, D22.9, D22.11, D22.16). `fourier_mellin="off"`
equals the call without the keyword bitwise on the V9 F6 map (the
four-point pre-Stage-D pin map) and on the shipped Ni map, on the
CPU and through the numpy session of V9; the pre-Stage-D literal pins
and the V9 `extras == {}` asserts re-run unmodified; with `"off"` the
prop set is exactly the pre-Stage-F set, no `FourierMellinState` is
built and `SeedBatch.outputs` stays empty (spies). `ValueError` with
the D22.1 message for `"AUTO"`, `"on"`, `""`, `True` and `None`.
`seed_from_neighbors=True` with `"auto"` and with `"always"` raises
the D22.11 `ValueError` (literal asserted in full) on both backends,
with spies on the D21.2 gate, `resolve_reference`, `ReferenceState`
and the pattern gather at zero calls; an invalid `backend` still
raises its own `ValueError` first. `"auto"` with `xmap=None` at the
engine raises the D22.1 message; `"always"` with `xmap=None` runs.
Forwarding: with `run_hrebsd_dic` replaced in `kikuchipy.signals.ebsd`
by a raising spy (the `test_public_method_forwards_backend`
precedent, `test_hrebsd_gpu.py:2537` at f297867e),
`EBSD.hrebsd_dic(fourier_mellin=v)` passes `v` for each of the three
values. Public result: `EBSD.hrebsd_dic` on the Ni map under
`"always"` returns a CrystalMap carrying `fourier_mellin_seed` and
`fourier_mellin_angle` with the D22.9 dtypes and shapes, and under
`"off"` carries neither. The new docstring test pins the
`fourier_mellin` entry, the Raises entries, the warning, the Notes
topics and the rewritten Limitations sentence of D22.16, with no
stage letter; `FROZEN_SIGNATURE` and `test_run_defaults_are_frozen`
carry the new keyword in its slot. Expectation: the frozen literals.
Band: bitwise.

(b) **The FM angle, seed level** [default] (D22.3). Through
`fourier_mellin_angles` alone, no fit. On G1 at `border=0.05` and
the default band-pass: twists 0, +-0.37, 1.83, 2.71, -4.42, 5.29,
9.61, 19.13 deg (off the 0.5 deg grid on purpose; ledger 128) and
the two combined cases (6.13 deg about z then 1 deg about x; 3.27
deg about z then 2 deg about y); expectation `atan2(h21 - h12, 2 +
h11 + h22)` of the exact imposed homography; band
`FM_ANGLE_TOL_DEG` (MTP; spec gate 0.123 deg noise-free, 0.113 deg
with Gaussian noise of half the pattern's standard deviation on
both images, ledger 128). Sign: `theta_hat` has the sign of the
imposed `h21` at every nonzero twist. The look-up table, unit level
and the PRIMARY bin-unit (FM3) killer: from `fourier_mellin_lut(sr,
sc)` at 432x432 and 460x560, the weight-averaged (column, row) of
each entry's four bins, unwrapped, reconstructs `(rho_j cos theta_k
sc, rho_j sin theta_k sr)` within 1e-12 for every entry; `n_theta =
360` angles over [0, pi) (not 2 pi), `n_rho` by the D22.3.3 formula
(165 and 175), shape and entry count (237600 and 252000) pinned. G3,
seed level, at 2, 5, 8, 15 and 20 deg and -8 deg:
`FM_ANGLE_TOL_NONSQUARE_DEG` (spec gate at most 0.027 deg; the
bin-unit mutant measured 0.074, 0.196, 0.344, 1.03, 2.29 and 0.386
deg; ledger 131), so a 2x pin of about 0.06 deg kills the mutant at
every angle from 5 deg with at least 3x separation -- the second
FM3 killer. Dead band: G1 with a constant 8 px dead-band cross in
both images (the `dead_band` subregion) at 3 and 8 deg, within
`FM_ANGLE_TOL_DEADBAND_DEG` (spec gate 0.12 deg against 0.02 without
the cross, ledger 131). Peak search, unit level on planted
correlations through `fourier_mellin_peak`: a global maximum at a 40
deg lag and a local one at 5 deg returns 5 deg (the search window);
an exact tie between lags +k and -k returns +k (the lowest output
index) under numpy and, gated, under cupy; a peak at the window's
edge bin takes its parabolic offset from the raw neighbour outside
the window; a non-negative-curvature peak gets offset 0.

(c) **Edge treatment and reuse** [default] (D22.3.1, D22.6). The
windowed spectrum the FM stage builds from `target_spectra` equals
`fft2` of the periodic-Hann-windowed zero-mean unit-norm crop within
`FM_STENCIL_RTOL` (relative; spec gate 1.6e-16, ledger 119). After
`seed_homographies`, `target_spectra` is bitwise the array
`seed_spectra` returned (a copy taken before the call), and the
translation rows of unrouted slots are bitwise the Stage E rows. G4
background lock: at `(None, None)` the recipe stays within
`FM_ANGLE_TOL_BACKGROUND_DEG` (spec gate 0.151 deg noise-free, 0.252
and 0.281 deg at Poisson 50 under the two filter settings; ledger
128), while the recipe WITHOUT the edge treatment misses by more
than 1 deg on most cases at `(None, None)` -- asserted on a local
re-implementation inside the test, so the premise is checked
whatever the production code does (`FM_LOCK_PREMISE_COUNT`, MTP;
spec gate on the untreated amplitude variant 8 of 9 above 1 deg,
worst 80 deg, ledger 128; the untreated log variant locked at zero
too, ledger 119).

(d) **Seed rows and failure semantics** [default] (D22.4, D22.5,
D22.6). On G1 at `border=0.15`, twists 8, -12, 5 and 15 deg with
projection centre shifts (12, -9), (10, 7), (-8, 5) and (6, -4) px:
the FM row's corner error to the exact imposed homography is below
`FM_SEED_TOL_PX` on every case, and a row composed in the wrong order
(`T(t) R`) exceeds it on every case (spec gate 0.083 to 0.220 px
against 0.898 to 2.655 px, the `2 sin(theta/2) |t|` separation;
ledger 125; the pin sits between the two populations); a de-rotation
about the detector centre instead of the PC (the FM43 mutant, error
`2 sin(theta/2) |c|` with `|c|` about 54 px at `PC_480`) exceeds it
too; the row equals `(c - 1, -s, c tx - s ty, s, c - 1, s tx + c ty,
0, 0)` of its own `theta_hat` and `t` exactly. The de-rotated crop of
`fourier_mellin_derotate` equals the CPU bicubic evaluation of the
target coefficients at `R xi` over every pixel of the D5 bounding box
within 1e-12 (numpy twin), with a `dead_band` arm showing the
dead-band pixels present in the crop. Never NaN, each on a planted
failure through a frozen patch point: `fourier_mellin_angles`
returning NaN for one slot, `fourier_mellin_derotate` returning a
NaN pixel or `ok` False for one slot, `fourier_mellin_translate`
returning a non-finite row for one slot -- that slot's returned row
is `h_T` bitwise, `"fourier_mellin_applied"` False, and the other
slots untouched; a NaN `h_T` stays NaN although a finite FM row
exists; a constant target gives the D2.6 contract exactly as with
`"off"`. The outputs rule: on route-0 and padded slots the angle is
NaN and `applied` False even though the masked branch computed them.

(e) **Capture, end to end** [default for the starred arms; the rest
weekly] (D22.3, D22.4). Expectation: the exact imposed homography,
by the V2 corner metric. Premise asserted at run time on each arm:
the `"off"` path FAILS at `max_iterations=50` (the D5 measurement:
2.5 deg and beyond on G1). G1 at `border=0.05`, twists 2.5*, 3*,
4, 5* and -3*, -5 deg, and the combined rotations (2, -1.5, 3)*,
(-2, 2, 4) and (1, 1, -3.5) deg (rotation vectors, detector frame):
the FM-seeded fit converges and recovers within `FM_CAPTURE_TOL_PX`
(spec gate 0.0078 to 0.0912 px in 2 to 7 iterations with the frozen
partial row, ledger 129). G1 at `border=0.15` [weekly]: 8, 10, 15,
20, 25 and -20 deg converge in at most `FM_CAPTURE_ITERATIONS` (spec
gate 2 to 3) to the error the EXACT-seeded fit reaches, within
`FM_FIXED_POINT_TOL_PX` (spec gate at most 8e-6 px apart, the errors
themselves 0.0010 to 0.0058 px; ledger 131); a window-edge arm at 30
deg is RECORDED, not asserted (spec gate `theta_hat` 29.931 deg,
converged in 3 iterations to 0.033 px, the exact seed's 0.033 px).
Every arm through `run_hrebsd_dic` with `"always"` on a one-row map,
so the runner, the route, the packed row and the props are
exercised, not only the seam.

(f) **Rescue without propagation** [default] (D22.7, D22.8, D22.10;
the V8(a) premise). The V8 ramp map through `run_hrebsd_dic(...,
seed_from_neighbors=False, max_iterations=200)` with `"always"` (no
`xmap`) and with `"auto"` and a matching G7-built `xmap`. Premises
asserted at run time on the `"off"` run: ramp columns 3 to 6 (2.4 to
4.8 deg) and the 8 deg point (`RAMP_UNFITTABLE_INDEX`) do not
converge; the 9 deg tilt (`RAMP_RESCUE_INDEX`) converges (113
iterations). Expected per point, both FM modes unless stated:
columns 3 to 6 converge to the imposed field within the V8
`SAME_OPTIMUM_FRACTION` band, code 1, finite angle; the 8 deg point
CONVERGES (code 1; it is out of frame beyond 6.62 deg, so its error
is the mirror-boundary scale, `FM_RAMP_FAR_TOL_PX`, spec gate 0.053
px from the FM row in 3 iterations at a budget of 200, ledger 131; a
D2.6 outcome there fails the test unless measured so); the 9 deg
tilt converges within `SAME_OPTIMUM_FRACTION` under both modes --
under `"auto"` it is not routed (twist 0), so its result is bitwise
the `"off"` result, code 0, angle NaN; under `"always"` the
acceptance keeps the FM row (spec gate criterion 1.279 against 1.293)
and the fit takes 116 iterations, code 1 (both pinned as measured,
ledger 131); columns 1 and 2 (0.8, 1.6 deg) converge, under `"auto"`
column 1 is not routed (code 0, angle NaN, bitwise `"off"`) and
column 2 is (code 1), under `"always"` both carry the code measured
(expected 1); the reference point carries code 0 under `"auto"`
(angle NaN) and under `"always"` the code measured (expected 0: both
criteria at rounding level) with a finite angle; the constant pattern
keeps the D2.6 contract (NaN `h`, 0 iterations, not converged; code
0, angle NaN) although the retry visits it (D22.8: its forced FM
estimate fails, so it is never refitted); masked points carry code
-1 and angle NaN. No neighbour seeding anywhere (no `seed_round`
prop).

(g) **Acceptance** [default] (D22.5). G5 (if its premises hold): the
D5 seed returns exactly (0, 0), `theta_hat` is within
`FM_ANGLE_TOL_DEG` (premises asserted), the FM row's criterion at the
seed is strictly lower and is kept, and the FM-seeded fit converges
where the `"off"` fit does not, or in fewer iterations (MTP). A
planted wrong angle (`theta_hat + 10` deg through a patched
`fourier_mellin_angles`): `h_T` is kept, bitwise. A planted tie
(`fourier_mellin_criteria` patched to return the same values for
both rows): `h_T` is kept and `applied` is False. A planted NaN
criterion for a finite `h_T` beside a finite, lower FM criterion
(the same patch point): `h_T` is kept and `applied` is False (the
D22.5 rule; the FM51 killer). G6 (unrelated
pair): no exception, no non-finite row introduced, the acceptance
refuses the FM row (`applied` False), and the fit's `converged` flag
equals the `"off"` path's. Criterion calls, counted on
`fourier_mellin_criteria` (never on `kernels.gather`, which the
de-rotation also calls): on the CPU route at P = 1, exactly two
calls per route-1 seam call (`h_T` and `h_FM`) and none for route 0
or route 2; in the numpy session, calls only in sub-batches with a
nonzero route-1 slot, each at the full P.

(h) **The pre-fit gate** [default] (D22.7). `twist_about_detector_
normal` on G7: imposed twists +-0.5, +-1.5, +-2.5 and +-4.8 deg
recovered within `FM_GATE_TWIST_TOL_DEG` (spec gate 1e-6 deg,
ledger 127) on both detectors, with the sign of `in_plane_fe` (the
V4 sense); a symmetry-equivalent target orientation (raw angle about
90 deg) gives the same twist; a pure 3 deg rotation about the
detector x or y axis gives zero twist to rounding, and twist plus
swing gives the twist; each point is measured against ITS grain
reference on the two-grain map. Routing through
`fourier_mellin_routes` on PLANTED twists: `FM_GATE_DEG` is routed
and `numpy.nextafter(FM_GATE_DEG, 0)` is not, a negative twist is
routed by its magnitude, NaN is routed; through the engine, the
unindexed point, the NaN rotation and the other-phase point are
routed (fail open) and masked points never. The all-identity map and
the constant non-identity map each issue the D22.7 `UserWarning`
(literal asserted) and route nothing pre-fit. Projection link: a
pattern projected through the master-pattern projection from a G7
`g_t` (on both detectors, twists 2.5 and -4.8 deg) equals the
detector-frame deformation construction of the same `R_det` within
1e-10 relative (spec gate 4e-14 to 1.4e-12, ledger 127), and the FM
`theta_hat` of that pair matches the gate's twist within
`FM_ANGLE_TOL_DEG`, so a sign or frame error between the gate and
the projected patterns cannot hide behind `|twist|`. With 0.5 deg
orientation noise the routed set of a 2.5 deg twist population is
recorded, not asserted. The map-level real-data frame check is (m).

(i) **Retry** [default] (D22.8). On G8 under `"auto"`: the
mislabelled 4.0 deg point is not routed pre-fit, fails, is retried
from its FM row (forced, no acceptance) and converges, and its
stored result equals a direct `fit_pattern(state, target,
h0=fm_row)` bitwise (`fm_row` from the numpy seam at P = 1 with
route 2), `num_iterations` the retry fit's own count,
`fourier_mellin_seed` 2; the unrelated-pattern point is retried, its
forced fit does not converge, and it keeps its first-pass result
bitwise (`fourier_mellin_seed` 0, finite angle); the constant pattern
is in the retry subset, its forced FM estimate fails, it gets no
second fit (spy) and keeps the D2.6 contract bitwise; the masked
point and the converged points are never retried; a spy on the
runner shows exactly ONE retry pass whose subset is those three
points in fit order. With `fourier_mellin_angles` patched to fail on
the mislabelled point in the retry, that point gets no second fit
and keeps its first result bitwise (code 0). Under `"always"`: the
mislabelled point converges in the first pass from its FM row (code
1, not retried), and the retry subset is exactly the unrelated point
and the constant pattern.

Amended 2026-10-08 (critic F7, plan 12.6; D22.8 amendment): the G8
constant pattern under the default band-pass IS refitted once by the
retry (its FM row is finite), and that fit returns the D2.6 contract
again, so `FM_G8_CONSTANT_SECOND_FITS` is 1 there and the first-pass
result stands bitwise; the "never refitted" spy is asserted on the
`(None, None)` arm, where the translation row is NaN.

(j) **CPU route** [default] (D22.9, D22.10, D22.13). On a map mixing
routed, unrouted and retried points under `"auto"`: every point that
is neither routed nor retried, and every routed point whose
translation row won, equals the `"off"` run bitwise; chunksize 1, 3
and the default give bitwise equal results; lazy equals eager
bitwise; two runs are bitwise equal; the seam is reached through the
`_batched` module globals at P = 1 (spy), and `_batched.seed_spectra`
is called exactly once per route-1 or route-2 point and never for an
unrouted one (the FM44 over-routing killer: the numpy seam equals
skimage on these fixtures, so only the count sees it); `_fit_chunk`
returns 14-wide rows on FM runs and 12-wide rows on `"off"` runs
(spy); the props carry the D22.9 dtypes and shapes.

(k) **Numpy session** [default] (D22.6, D22.11, D22.14, D22.18). The
device runner under the numpy session of V9 (patched
`_gpu._make_session`) with `seed_extras`: the route flags arrive in
`SeedBatch.extras` in fit order with 0 on padded slots (spy); after
every seam call `extras` is the same host dictionary holding
exactly the route key, unchanged, and the angle and applied arrays
live in `SeedBatch.outputs` only (the FM45 killer); the
outputs exist exactly when the route key has a nonzero entry (an
all-zero route key: no outputs and the Stage E rows bitwise); a
planted `seed_homographies` that writes no outputs is read as angle
NaN and `applied` False, without error; a sub-batch with no routed
slot never calls `fourier_mellin_rows` (spy) and returns the Stage E
rows bitwise; a routed slot's row is bitwise equal whether 1 or P
slots of its sub-batch are routed (the masked full-P rule); the FM
state is built lazily, once per grain with a routed slot and never
for a grain without one, and the `resident` it holds IS the lockstep's
resident (identity spy); a forced slot whose FM estimate fails is
inactive in the lockstep (kernel spy) and its packed row carries
`converged` 0 and `fm_applied` 0; `row_slots` carries the FM keys and
width 14 on FM runs and the Stage E dictionary on `"off"` runs; the
retry pass's `_run_chunks_gpu` call receives the first pass's final
B as `chunksize` (spy). VRAM functions (the V9(o) style): with
`fourier_mellin=False`, `_vram_model_terms`, `_vram_model_bytes` and
`_default_batch_size` return the Stage E values bitwise; with True
the terms are at least the False ones, and `_default_batch_size`
returns the False B whenever the FM terms fit the remaining headroom
and a halved one only when they do not (fake free VRAM). Against the
CPU route per point, `theta_hat` and the rows within the numpy-twin
bands (`FM_NUMPY_ANGLE_TOL_DEG`, the V9(f) twin bands for `h`; pinned
bitwise if measured bitwise). Under `seed_precision="complex64"` a
dtype spy shows every FM step after the spectra read in complex128
or float64 (D22.12). Import hygiene: no module-scope cupy, `_engine`
or `_batched` import in `_fourier_mellin.py` (the V9(p) AST arms
extended).

(l) **Device suite** [gated, `-n 0`, the pinned overlay of D21.15]
(D22.12 to D22.14, D22.18). The output keys' device contract
(`cupy.ndarray`, float64 and bool, `(P,)`); `theta_hat` against the
CPU route on G1, G3 and G4 within `FM_ANGLE_PARITY_DEG` at both seed
precisions; acceptance decisions within `FM_ACCEPT_FLIP_COUNT`; `h`
on both-converged FM-seeded points within the D21.8(c) bands per
device precision; iteration and convergence budgets as D21.8(e);
retry routing within `FM_RETRY_FLIP_COUNT`; two runs bitwise at a
fixed B; ROUTING INVARIANCE: a routed slot's row bitwise the same
with 1 of P and P of P slots routed (the FM29 device killer); B in
{8, 32, 40, default} as E4; the default B equal under `"off"` and
`"auto"` on the test card (D22.18); the FM terms of the VRAM model
calibrated against pool high-water marks; the D21.8(h) drift
tripwire unchanged under `"off"`; a throughput record
(`record_property`, asserting completion only).

(m) **Real-data records** [local, RECORDED in the ledger, never a
test] (D22.7, D22.15; plan open questions FQ1, FQ2, FQ3, FQ7, FQ8,
FQ11, FQ13). On the Si-indent file (read only, never copied), with
the executed tutorial's load recipe: (1) the REQUIRED frame oracle,
by the pre-registered rule of D22.7 -- an `"always"` run; the fitted
twist from the polar decomposition of each converged Fe, with the
homography read-out as a cross-check; points with |fitted twist| >=
1 deg, converged, outside the crater; the correlation for the
as-read frame and the four alternative frames; the slope of the
xmap twist on the fitted twist and the Deming slope with the
far-field Hough variance; PASS if the as-read frame correlates best
and its slope lies in [0.8, 1.25], else the stage stops for Johan;
(2) the whole map under `"auto"` at `(None, None)` and `(0.05,
None)` on both backends -- routed, accepted, retried, converted and
worsened counts against `"off"`, wall time, the device seed share,
the number of sub-batches and slots the FM branch ran on, the
`theta_eff` of D22.7 beside the twist for every point (the FQ1, FQ2
candidate), and the distribution of |`theta_hat`| and |twist| over
the routed points (FQ11: reopen if any routed |twist| exceeds about
25 deg or any `theta_hat` sits at the window edge); (3) far256 and
patch C under `"always"`; (4) the D5 anchor census of FQ8 (the
fraction of points whose D5 seed is exactly zero while a
band-limited phase correlation is not).
**AMENDED 2026-10-08 (frame oracle; ledgers 143 and 144; plan
12.8).** The load recipe of (m) applies the Oxford conversion: the
CrystalMap is built from `Rotation.from_euler(euler_ox) *
Rotation.from_axes_angles([0, 0, 1], -90, degrees=True)` (Oxford
CS1 to kikuchipy's sample frame, D22.7 amendment), as the
tutorial's conversion cell does; the reader-derived detector is
unchanged. (1) was first measured on the raw angles and FAILED
(correlation 0.604, slope 0.522, best frame "sample z -90"); the
cause is the vendor's frame, and on the converted map the same 445
points PASS (0.993, slope 0.985, Deming 0.987), signed off by Johan
on 2026-10-08. Every `"auto"` record of (2) uses the converted map;
the `"off"` runs, the `"always"` fits of (3) and the census of (4)
do not read the orientations and stay valid with recomputed twist
columns.

V10 requirement-to-oracle map:

| D22 item | oracle(s) | suite |
|---|---|---|
| D22.1 API, default path, check order, forwarding | (a) | default |
| D22.2, D22.3 the FM angle, look-up table, peak | (b), (c) | default + gated parity |
| D22.4 de-rotation and the row | (d), (e) | default |
| D22.5 acceptance, failure semantics | (d), (g) | default |
| D22.6 seam extension, outputs, patch points | (c), (d), (k) | default + gated |
| D22.7 gate, routing, warning | (h), (f); (m) frame | default + ledger |
| D22.8 retry | (f), (i), (k) | default |
| D22.9 props and packed row | (a), (f), (i), (j), (k) | default |
| D22.10 CPU route | (j) | default |
| D22.11 interplay, device wiring | (a), (k) | default |
| D22.12 to D22.14 dtypes, determinism, parity | (j), (k), (l) | default + gated |
| D22.15 performance | (l) record, (m) | gated + ledger |
| D22.16 docs | (a) docstring test | default |
| D22.17 dependencies | (k) import hygiene | default |
| D22.18 VRAM model, batch choice | (k), (l) | default + gated |

Gate commands, recorded verbatim with their output at each Stage F
gate (Git Bash, from the worktree; at most `-n 2` while the machine
is shared, the 2026-10-06 memory lesson):

```
# default suite (cupy absent: the gated classes skip at stage (a))
uv run pytest tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0
uv run pytest tests/test_indexing tests/test_signals -k hrebsd -n 2
# gated suites: Stage F, then Stage E unchanged (pinned overlay)
OVERLAY="--with cupy-cuda12x==14.2.0 \
  --with nvidia-cufft-cu12==11.4.1.4 \
  --with nvidia-cublas-cu12==12.9.2.10 \
  --with nvidia-cusolver-cu12==11.7.5.82 \
  --with nvidia-cusparse-cu12==12.5.10.65 \
  --with nvidia-nvjitlink-cu12==12.9.86"
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0 -q --weekly
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_gpu.py -n 0 -q --weekly
# coverage: default run, then both gated runs appended
uv run pytest tests/test_indexing tests/test_signals -k hrebsd \
  -n 0 --cov=kikuchipy.indexing._hrebsd --cov-report=
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_fourier_mellin.py \
  tests/test_indexing/test_hrebsd_gpu.py -n 0 --weekly \
  --cov=kikuchipy.indexing._hrebsd --cov-append \
  --cov-report=term-missing
```

#### V10 recorded results, spec gate (2026-10-07)

Measured on machine A (ledger 89) by the Stage F spec-phase
exploration: a literature and method study with a throwaway CPU
prototype, a read-only seam and frame probe, a second throwaway CPU
prototype of the full seed, and, while drafting this spec, a
head-to-head of the two prototypes' recipes on identical inputs and
an end-to-end check of the recipe D22 recommends, and, after the
spec review, a re-measurement of the frozen recipe wherever the
critics found a number taken from another recipe or row (ledger
131). Numbers from a recipe other than the frozen one (D22.3 V8, the
partial row, the acceptance) are tagged with their recipe. CPU ONLY
(no GPU work: Stage E was measuring GPU timing pins at the time), one
process each. Nothing of the prototypes is committed and no
repository file was edited; the scratchpad does not survive the
session, so every number is carried here. These are PROTOTYPE
numbers: the Stage F gates re-measure every one that feeds a pin or
a decision.

115. **Conditions of every entry below.** Machine A (i7-13700H, 20
    logical cores, 32 GB, Windows 11 build 26200); the worktree
    `.venv` (CPython 3.13.12, numpy 2.4.6, scipy 1.17.1, numba
    0.65.1, scikit-image 0.26.0, orix 0.14.2); OMP, MKL, OpenBLAS and
    numba threads at 1 in the second prototype's fixture module and
    every drafting run, single-threaded numpy timings in the method
    study. Engine
    code: the CPU path of the working tree on top of f297867e, which
    Stage E leaves unchanged (its default-path pins); the prototypes
    called only `fit_pattern`, `initial_guess`, `ReferenceState`,
    `preprocess`, `spline_coefficients`, `evaluate`,
    `zero_mean_normalize`, `fe_to_homography`,
    `homography_parameters`, `shape_function` and
    `per_point_pc_pixels`; Stage E's in-progress `_batched.py` was
    read for the seam's names only. Synthetic oracle: the V3/V4
    deformed master at 480x480 through helper copies of
    `test_hrebsd_engine.py` (`PC_480` unless stated; crop 432x432).
    Real data: `C:\Users\westraadt.1\Repos\kikuchipy\AGH__Si_indent_
    1_512x672.h5oina`, READ ONLY, never copied (h5py reads checked
    bitwise equal to `kp.load`), binning 2 (512x622, crop 460x560),
    reference (10, 10) at PC (328.3, 90.2, 379.4) px, per-point PCs as
    `run_hrebsd_dic` builds them, `border=0.05`. The machine was
    shared (the HROSM session, the Stage E workflow's test runs);
    identical code timed 33 to 58 ms in different runs, so timings
    carry about +-30 per cent; free virtual memory was 2.3 GB when
    the drafting runs started. Scripts (session scratchpad
    `stageF_spec/`): the method study `fm_lib.py`, `exp1_angle.py` to
    `exp10_anchor.py` (outputs `exp7.out`, `exp7b.out`); the probes
    `probe_frames.py`, `probe_si_gate.py`, `probe_peak_gate.py`,
    `budget.py`; the seed prototype `proto/` (`fixtures.py`, `fm.py`,
    `sweep.py`, `sweep2.py`, `gridstudy.py`, `gridstudy2.py`,
    `realdata.py`, `realdata2.py`, `realrot.py`, `cost.py`,
    `sparse_polar.py`, `rimseed.py`, `zerolag.py`, `inframe.py`,
    `analyze_*.py`, outputs in `proto/out/`); the drafting runs
    `proto/h2h.py`, `proto/e2e.py`, `proto/cost_v8.py` and
    `h2h_gatecheck.py` (logs `proto/out/h2h.log`, `e2e.log`,
    `cost_v8.log`; `h2h.json`, `e2e.jsonl`); the spec-review runs
    `proto/e2e_partial.py` and `proto/deadband.py` (logs
    `proto/out/e2e_partial.log`, `e2e_partial.jsonl`,
    `deadband.log`) and the critics' `stageF_critic/subbatch_amp.py`
    and `stageF_critic/dilution.py`. Below, "periodic" is
    the reused spectrum minus the Moisan smooth component, "hann" a
    windowed spectrum, "raw" the reused spectrum untreated.

116. **FM angle accuracy on the oracle, and the physics checks
    (requirements D22.2, D22.3).** (i) Pure twists 0 to 15 deg plus
    -3 and -12 (13 angles), 360 angles, rho
    0.05-0.35, radial-mean ZNCC, +-30 deg window, parabolic peak;
    max |error| in deg:

    | spectrum | amplitude | log |
    |---|---|---|
    | raw (reused) | 0.076 | 0.055 |
    | periodic (reused + Moisan) | 0.079 | 0.031 |
    | hann (a second FFT) | 0.015 | 0.012 |

    Periodic-log sensitivity: 180 / 360 / 720 angles 0.036 / 0.031 /
    0.032; rho_max 0.25 / 0.35 / 0.45: 0.046 / 0.031 / 0.027;
    rho_min 0.02 / 0.05 / 0.10: 0.024 / 0.031 / 0.032; a 1-bin and a
    0.5-bin radial step identical. Angular peak 0.90-0.99, 1.000 at
    zero. (ii) In-plane-axis bias, predicted `(w1 xbar + w2 ybar) /
    (2 DD)` with the subregion centroid at (+37.9, -38.1) px and DD
    242.4 px: +-0.235 deg for 3 deg about x and about y; the
    windowed FM measured +0.231, -0.178, -0.230 and +0.190 deg. On
    the Si-indent geometry (centroid (-17.3, +165.8) px from the PC)
    the formula gives `theta_FM ~= w3 - 0.023 w1 + 0.218 w2`, so 5
    deg about y reads as 1.1 deg of twist. (iii) Pseudo-scale
    `(3/2)(w2 xbar - w1 ybar) / DD`: 1.2 per cent for 3 deg on the
    oracle, 5.7 per cent for a 5 deg w1 on the Si geometry -- the
    reason D22.2 excludes scale.

117. **Capture end to end on the oracle (requirements D22.3,
    D22.4; the D5 premise).** (i) `fit_pattern(max_iterations=50)`,
    periodic-log FM (the METHOD-STUDY recipe; the frozen recipe's
    rows are ledger 129), the engine's spline de-rotation,
    `W0 = R(theta) T(t)`; V2 corner error to the exact homography:

    | rotation vector (deg) | translation seed: conv, it, err px | FM seed: conv, it, err px |
    |---|---|---|
    | (0, 0, 1) | yes, 5, 0.004 | yes, 2, 0.004 |
    | (0, 0, 2) | yes, 8, 0.011 | yes, 2, 0.011 |
    | (0, 0, 2.5) | NO, 50, 148 | yes, 2, 0.012 |
    | (0, 0, 3) | NO, 50, 173 | yes, 2, 0.013 |
    | (0, 0, 4) | NO, 50, 221 | yes, 2, 0.008 |
    | (0, 0, 5) | NO, 50, 98 | yes, 2, 0.010 |
    | (0, 0, -6) | NO, 50, 171 | yes, 2, 0.033 |
    | (0, 0, 8) | NO, 50, 53 | yes, 3, 0.160 |
    | (3, 0, 0) | yes, 9, 0.117 | yes, 9, 0.117 |
    | (2, -1.5, 3) | NO, 50, 53 | yes, 7, 0.017 |
    | (-2, 2, 4) | NO, 50, 68 | yes, 7, 0.091 |
    | (1, 1, -3.5) | NO, 50, 34 | yes, 6, 0.077 |

    At 5 deg and beyond the corners leave the pattern at
    `border=0.05` (ledger 130), so the 8 deg 0.160 px is the mirror
    boundary, not the seed. (ii) The second prototype (whitened
    per-radius recipe, nearest rule, `W(R) W(T)`; NOT the frozen
    recipe), 113 cases: twists
    0 to 30 deg and -3, -10, with projection centre shifts and
    tilts, on four set-ups (A: `PC_480`, border 0.05; B: the V8 PC,
    border 0.05; C: `PC_480`, border 0.15; A and C also unfiltered).
    The translation seed converges up to 2.0 deg on A and B and
    fails from 2.5 deg on both (pure, with a (10, 7) px shift, with a
    y tilt), with ERRATIC success beyond (B converges at 3.0 deg; C
    up to 3.0; A unfiltered up to 4.0; C unfiltered at 4 and 6 but
    not 3). The FM seed: 113 of 113 converge to the exact-seed fixed
    point within 7.5e-5 px in 1 to 9 iterations (median 2 on pure
    twists; 9 only with 3 deg tilts); 112 of 113 within 1 px of the
    exact homography (A unfiltered at 15 deg, 1.074 px, as the exact
    seed). Angle error on pure twists: median 0.006, p90 0.041, max
    0.33 deg (at 0.5 deg true, pulled towards 0; those fits still
    converge in 3 to 4 iterations). (iii) The translation seed's
    phase-correlation peak: 1.000 at 0 deg, 0.037 at 1, 0.031 at 2,
    0.018 at 2.5 -- it collapses before the seed fails, so it is no
    gate (ledger 127).

118. **Real Si patterns rotated rigidly about the PC (requirements
    D22.3; the D5 premise on real data).** Recipes: (i) the method
    study's variants, (ii) the second prototype's whitened
    per-radius recipe; neither is the frozen one. (i) Target (10, 13), a
    different real pattern, rotated about the reference PC by the
    engine's spline; `max_iterations=200`; FM error over 0 to 8 deg
    (both signs) -0.027 to -0.002 deg (raw-amplitude), -0.016 to
    +0.008 (raw-log), -0.016 to +0.004 (periodic-log), -0.048 to
    -0.009 (hann-log). Default band-pass: the translation seed
    converges at 0, 1 and 2 deg in 7, 19 and 124 iterations and
    fails at 3, -4, 5 and 8 deg (residual about 1.93); the FM seed
    converges at every angle in 6 to 8 iterations (residual 0.1771 to
    0.1773 where both converge; 0.189 at 8 deg, the mirror boundary).
    `(None, None)`: translation 14, 26, 32, 47 and 70 iterations at
    1, 2, 3, -4 and 5 deg, not converged at 8; FM 5 to 8 at every
    angle. FM-seeded fits recover the imposed angle to about 1e-3
    deg up to 5 deg. Real angular peaks 0.96-0.98. (ii) Four
    far-field targets ((12, 40), (20, 20), (30, 60), (9, 13)) rotated
    by an INDEPENDENT warper (skimage, bicubic, reflect), budget
    500: the FM seed converged 16 of 16 at every angle from -6 to 10
    deg under both filter settings, in 5 to 11 iterations, within
    2.5e-3 px of the exact-seeded fit, angle error at most 0.021 deg;
    the translation seed reached 6 deg unfiltered (107 to 168
    iterations) and failed at 10, and reached 3 deg with the default
    band-pass (196 to 206 iterations) and failed from 4.

119. **Edge treatment and the background lock (requirements
    D22.3.1).** (i) G4's detector-fixed background, noise-free, max
    |error| over 0 to 8 deg: with `(0.05, None)` raw at most 0.044,
    periodic 0.012, hann 0.014 deg; with `(None, None)` the RAW
    spectrum LOCKS AT ZERO (errors -0.97, -1.99, -3.00, -5.02 and
    -7.99 deg at 1, 2, 3, 5 and 8 deg), periodic 0.012 and hann
    0.014. (ii) The exact frequency stencil of the periodic Hann
    window equals a second FFT of the windowed crop to 1.6e-16
    relative; the stencil evaluated only at the samples' neighbours
    equals the dense one to 9e-16. (iii) On 32 real rim points
    (`(None, None)`), the whitened recipe without any window had a
    46.8 deg outlier where the windowed one had none (ledger 128).
    (iv) G6, the unrelated pair: FM returned 12.36 deg with angular
    peak 0.56; both rows are junk. CORRECTED at the spec review
    (ledger 131 (iii); the original "the fit fails by the D2.6
    contract" was inferred, not measured): at the V8 PC and a budget
    of 200 the translation-seeded fit runs to the cap with a FINITE
    `h` (residual 1.89), the acceptance refuses the FM row, and a
    forced FM-seeded fit does not converge either.

120. **Shot noise (requirements D22.3; plan open question FQ4).**
    Poisson noise at full-scale counts of 50, 20 and 10 on the G4
    oracle, periodic correction, max |error| over 0 to 8 deg:

    | magnitude, rho_max | (0.05, None) 50 / 20 / 10 | (None, None) 50 / 20 / 10 |
    |---|---|---|
    | amplitude, 0.20 | 0.121 / 0.203 / 0.843 | 0.290 / 0.402 / 1.518 |
    | amplitude, 0.35 | 0.133 / 0.543 / 0.379 | 0.235 / 0.412 / 3.420 |
    | log, 0.20 | 0.176 / 0.178 / 1.041 | 0.581 / 0.837 / 3.289 |
    | log, 0.35 | 0.365 / 0.939 / 0.814 | 0.517 / 0.869 / 3.418 |

    At low counts amplitude over a band ending at 0.20 cycles/px was
    the most robust; the angular peak tracked reliability (0.85-0.88
    at 50, 0.44-0.62 at 10) but is 0.25-0.82 on real deformed points
    where the seed still helps (ledger 121), so it is no gate. Fits
    from either seed did not converge at these synthetic noise
    levels in 100 iterations, so this entry says nothing end to end.

121. **Real deformed patterns, the indent's south rim (requirements
    D22.5, D22.8).** (i) Method study, 30 points (rows 125-129 x
    columns 115-119 plus (120, 110), (130, 125), (118, 122), (135,
    118), (126, 108)), `max_iterations=500`, periodic-amplitude FM
    over rho <= 0.20; T the D5 seed, P the FM partial row, C the FM
    complete row:

    | filter | T: converged, iterations total (median) | P | C |
    |---|---|---|---|
    | `(0.05, None)` | 9/30, 13234 (500) | 23/30, 8456 (261) | 24/30, 7494 (214) |
    | `(None, None)` | 30/30, 3503 (122) | 30/30, 2840 (92) | 30/30, 2721 (91) |

    Two of T's 9 converged default-filter points sat at residuals
    1.95 and 1.86 where FM reached 1.14 and 0.53 (the false
    convergence of D22.8). Where all three converge they reach the
    same residual (median 0.3101 unfiltered). Polar decomposition of
    the converged unfiltered Fe: w1 0.75-2.4 deg, w2 -1.2 to 2.0,
    twist 0.06-0.95 deg -- the rim's gain is about translation and
    perspective, not twist. FM over-read the twist by -0.41 to +0.86
    deg; the first-order bias formula removed half to two thirds of
    the far-rim over-read and changed no fit by more than 3
    iterations. The D2.7 criterion at the seed: T 1.06-2.14 on every
    point; P and C 0.26-0.69 wherever `theta_FM >= 0.43` deg (17
    points), and between 0.19 lower and 0.07 higher than T elsewhere.
    (ii) Second prototype, 32 far-field points (rows 8-11, columns
    20-27) and 32 rim points (rows 125-130 x 115-119 plus (124, 117),
    (124, 118)), `(None, None)`, budget 500, every point fitted from
    both seeds: both converge 32/32 in both groups; `|delta h|`
    median / max 1.6e-4 / 4.1e-4 px far and 2.8e-4 / 4.1e-3 px rim;
    residual relative difference at most 3.9e-6 and 3.1e-7;
    iterations median 8 / 8 far and 122 / 111 rim; FM angle against
    the fitted in-plane rotation at most 0.011 deg far and median
    0.049, max 0.19 deg rim -- no harm.

122. **The D5 seed is anchored at zero shift on real Si
    (requirements D22.5, D5 amendment; plan open question FQ8).** (i)
    At 10 of 10 deformed points under both filter settings, the
    full-band phase correlation of `initial_guess` returns exactly
    (0, 0) (peak 0.094-0.104), while a band-limited phase correlation
    (cross-power masked to rho in [0.02, 0.20] cycles/px) or the
    plain correlation returns 6 to 18 px, consistent with the
    converged translations (at (125, 115): band (+12, -11) default
    and (+13, -11) periodic, plain (+10, -8), converged (+10.2, -7.0)
    px at the PC). At undeformed points D5 also returns (0, 0) where
    the others give 1 px, the order of the map's PC span. (ii) The
    second prototype on rim points: (0.00, 0.00) against converged
    translations of 10 to 21 px (median 16.2); zero-lag peak
    0.093-0.101 against 0.012-0.016 near the true shift; seeds 22 to
    66 px off (corner norm); unfiltered the fits recover in 77 to 152
    iterations, with the default band-pass 3 of 3 checked did not
    converge in 500. (iii) Drafting run, ledger 129: all 24 rim seeds
    were exactly (0.0, 0.0). Cause: whitening gives every frequency
    equal weight, and detector-fixed content (camera fixed pattern,
    hot pixels, scintillator texture), identical in both images, owns
    the high frequencies. The FM branch escapes it because the
    de-rotation also rotates the target's fixed content; the escape
    was seen from about 0.4 deg of de-rotation upward.

123. **Non-square crops and the composition order (requirements
    D22.3.3, D22.4).** Real Si, crop 460x560, target (10, 13) rotated
    about the PC, amplitude, rho <= 0.20: the physical-frequency
    look-up table measured +0.002, +1.997, +4.988 and +7.990 deg at
    0, 2, 5 and 8 deg (angular peak 0.99-1.00); a table in bin units
    on both axes measured +0.003, +2.018, +5.054 and +8.185 deg (peak
    down to 0.88 and 0.73), a bias of about 2.3 per cent of the
    angle that stays inside the IC-GN basin (this method-study
    recipe; with the frozen recipe on the G3 synthetic the bias is
    0.34 deg at 8 deg and 2.29 deg at 20, ledger 131 (iv)) -- only a
    seed-level angle test at 5 deg or more on a non-square crop, or
    the look-up-table coordinate pin of V10(b), kills it. The composition mutant `T(t) R`
    was indistinguishable from `R T(t)` on that pair (t = 0); ledger
    125 measures the fixture that separates them.

124. **The 180 deg ambiguity (requirements D22.3.7).** Second
    prototype on 32 real rim points, `(None, None)`: choosing
    between theta and theta + 180 by the criterion at the seed picked
    the 180 deg candidate on 3 points (scores 0.688 against 0.597,
    0.513 against 0.240, 0.025 against 0.000), because the smooth
    background is close to point symmetric about the PC; (125, 119)
    then converged 1108.6 px and (127, 117) 1083.4 px from the
    translation-seeded result, and (130, 118) did not converge in
    500. On the synthetic sweep the rejected candidate never scored
    above 0.096. The nearest rule (theta in [-90, 90)) had 0 failures
    and saves one de-rotation and one cross-correlation (seam-form
    seed 80 / 101 ms against 173 / 232 ms at 432x432 / 460x560).

125. **Partial against complete initialisation (requirements
    D22.4.3; plan open question FQ5).** (i) Method study on the
    oracle, iterations and seed corner error: (3, 0, 0) partial 9,
    19.06 px against complete 4, 1.43; (0, 3, 0) 9, 19.03 against 4,
    0.89; (2, -1.5, 3) 7, 16.54 against 4, 1.38; (-2, 2, 4) 7, 19.19
    against 4, 1.31; (1, 1, -3.5) 6, 12.59 against 4, 0.84; (3, 3, 5)
    17, 36.95 against 6, 1.98; final homographies equal to 1e-4 px.
    On the real rim (ledger 121) 23 against 24 of 30 converged with
    the default band-pass and 2840 against 2721 iterations
    unfiltered. (ii) Drafting run (`e2e.py`, the D22 recipe): on the
    oracle's pure twists both forms take 2 to 3 iterations; on the
    five combined rotations partial 6 to 9 and complete 4, both to
    the same final error. (iii) Drafting run, `PC_480`,
    `border=0.15`, twists with PROJECTION CENTRE SHIFTS (a pure
    translation, `make_case(pc_shift_px=...)`), seed corner error to
    the exact homography, px:

    | twist, shift | D5 seed | partial | partial `T(t) R` (mutant) | complete, d = R t | complete, d = t (Ernould E.2-E.3) |
    |---|---|---|---|---|---|
    | +8, (12, -9) | 132.03 | 0.177 | 2.226 | 24.471 | 24.521 |
    | -12, (10, 7) | 51.28 | 0.105 | 2.655 | 13.657 | 12.245 |
    | +5, (-8, 5) | 29.29 | 0.083 | 0.898 | 12.245 | 12.498 |
    | +15, (6, -4) | 77.52 | 0.220 | 1.946 | 11.077 | 10.854 |

    The complete row reads a beam-scan translation as two lattice
    rotations and seeds 11 to 24 px off (its fits still converged in
    5 to 8 iterations to 0.001-0.003 px); the partial row seeds
    within 0.22 px, and its wrong-order mutant sits 0.90 to 2.66 px
    off, the `2 sin(theta/2) |t|` separation the D22.4 composition
    pin uses. The two complete orderings are both valid
    parameterisations and differ at second order. Hence D22.4 adopts
    the partial row; fitted from it, these four cases converge in 2
    to 3 iterations to 0.0011-0.0027 px (ledger 131 (ii)). Sign check of
    `fe_to_homography` used by the complete row (DD 242.35 px): 1 deg
    about y gives h13 = +4.230 px (= DD tan 1 deg), 1 deg about x
    gives h23 = -4.230, 1 deg about z gives h21 = +0.01745.

126. **Cost on one CPU thread (requirements D22.15).** (i) Method
    study, 432x432 per pattern: `fft2` complex128 10.4 ms; log1p of
    the magnitude with a per-slot median 5.3; Moisan smooth spectrum
    rebuilt per call 9.9; polar profile through a 187200-entry table
    2.0 (0.5-bin step, 374400 entries, 4.2); 1D correlation and
    peak 0.05; spline de-rotation of 186624 px 20.0; skimage phase
    XC 26.3. Batched at P = 16 with precomputed tables, the whole
    angle path cost 5.89 ms (432x432) and 8.01 ms (460x560), about
    0.8 to 0.95 of one batched `fft2`. (ii) Second prototype (its
    whitened 180x32 recipe), medians of 30, 432x432 / 460x560: CPU
    `initial_guess` 44-61 / 55-68 ms; dense angle path 14-18 / 18-23
    ms, sparse 1.3 / 1.4 ms; dense numpy stencil 10-18 ms against a
    second `fft2` of 3.3-6.5 ms; de-rotation gather over the box
    18-19 / 25-27 ms; XC on the de-rotated crop (seam form) 40-49 /
    48-57 ms; one criterion evaluation 20-24 / 28-30 ms; the FM seed
    80 / 101 ms dense, about 66 / 83 ms with the sparse angle path;
    one IC-GN iteration 21-23 / 31 ms; the per-reference precompute
    19-22 / 26-30 ms. (iii) Drafting run (`cost_v8.py`, the D22
    recipe, dense, scipy FFT, medians of 15, CPU load 4 per cent):
    look-up table 237600 entries (165 radii) built in 4.0 ms and
    252000 (175 radii) in 4.7 ms; `fft2` 1.43 / 2.57 ms; stencil
    6.54 / 8.80; log1p of the magnitude 1.29 / 1.79; gather and
    radial mean 1.16 / 1.24; correlation 0.014; the angle path 9.00
    / 11.85 ms per slot. The dense numpy stencil dominates and costs
    more than a second `fft2`; a fused kernel is the obvious
    follow-up, and on the device it is one elementwise pass. GPU
    cost: NOT measured (no GPU work at this gate); about 2.2 to 2.5
    ms per SLOT of a sub-batch holding a routed point is an inference
    from the 2.16 ms device seed (ledger 92); the slot count is
    ledger 131 (vii).

127. **The CrystalMap gate (requirements D22.7; plan open questions
    FQ1, FQ2, FQ13).** (i) Frames, synthetic (`probe_frames.py`, 240 px,
    Ni master, sample tilt 70, detector (tilt, azimuthal, twist) (0,
    0, 0) and (10, 4, 1.5)): `to_matrix` and `rotate_vector` agree to
    1.1e-16; a target built from an orientation, `g_t = g_r @
    R_s.T` with `R_s = M^T Fe_det M`, projects the same pattern as
    the V4/V8 deformation construction to 4e-14-1.4e-12 relative; the
    twist is recovered to 1e-6 deg at -3.0, +2.5 and +4.8 deg, with
    `|R_det - Rz| <= 9e-16`; a cubic symmetry operator applied to
    `g_t` gives a raw angle of 88 to 94 deg and the reduced twist
    equals the truth (orix `angle_with` agrees). (ii) Threshold, the
    113-case sweep of ledger 117 with exact orientations, "needed" =
    the translation seed fails:

    | threshold (deg) | needed but not routed | routed, not needed |
    |---|---|---|
    | 1.0 | 0 | 30 |
    | 1.5 | 0 | 20 |
    | 2.0 | 0 | 15 |
    | 2.25 | 0 | 15 |
    | 2.5 | 6 | 10 |

    Routed-but-not-needed points converged to the same answer in
    fewer iterations. (iii) The Si-indent map from its Euler angles
    (no pattern read): detector from the file header (sample tilt
    74.998, tilt 6.993), `n_s = M^T e_z = (0.3745, 0, 0.9272)`; of
    58055 points within 5 deg of the reference, |twist| median 0.014,
    p99 1.305, p99.9 2.138, max 2.588 deg, and 1090 / 385 / 90 / 5
    points above 1.0 / 1.5 / 2.0 / 2.5 deg, in rows 79-142, columns
    119-141; the far field (rows 20:36, columns 20:36) at most 0.031
    deg; patch C 122 and 24 of 650 above 1.5 and 2.0 deg. Over all
    58500 points with the 728-point crater (CCC < 0.35) separated:
    1515 / 793 / 483 / 389 above 1.0 / 1.5 / 2.0 / 2.5 deg, of which
    1012 / 336 / 73 / 4 outside the crater, in a ring 3 to 7 um from
    its centre; far-field rows 0-39 median 0.0055, max 0.041 deg.
    (iv) Frame sensitivity: assuming a sample-frame offset about
    sample z of +90, 180 or -90 deg moves the count above 1.5 deg
    from 385 to 79, 116 or 280 (correlation with the as-read values
    0.38, -0.29, 0.54); the tutorial passes the raw h5oina Euler
    angles, and cubic stiffness (D9) is blind to a 90 deg z offset,
    so no earlier stage pins this frame. (v) Drafting check
    (`h2h_gatecheck.py`, on the second prototype's records): on 32
    rim points the as-read twist against the converged in-plane
    rotation `atan2(h21 - h12, 2 + h11 + h22)` has correlation
    0.836, least-squares slope 0.45, intercept 0.13 deg, |difference|
    median 0.075 and max 0.35 deg, signs agreeing on 24 of 32; the
    range is only 0.05 to 0.45 deg, so the frame is NOT pinned (plan
    open question FQ13); on 32 far-field points both quantities are
    below 0.03 deg (correlation -0.04, Hough-limited). Spec review
    (ledger 131 (viii)): with the fitted rotation as the regressor
    the slope is 1.545 (geometric mean 0.54); the noisy input twist
    as regressor attenuates the 0.45, so these 32 points bracket 1
    and decide nothing, and the frame decision moves to the
    pre-registered rule of D22.7 on the whole map. (vi)
    Frame-free gates, rejected: the translation seed's
    phase-correlation peak at the V8 PC (480 px, default band-pass)
    reads 0.097, 0.037, 0.033, 0.022, 0.020, 0.021 and 0.016 at 0.5,
    1, 2, 2.5, 3, 4 and 4.8 deg, with a peak-to-second ratio of 1.05
    to 1.44 at every angle; the coarse shift is spurious at 2.5 deg
    (10 px), 4.0 (14 px) and 4.8 (80 px) but 0 at 3.0, so neither the
    height nor the shift separates the regimes.

128. **Head-to-head of the angle recipes on identical inputs
    (requirements D22.3; drafting run `h2h.py`, 46.6 s).** Eleven
    variants of the angle, each evaluated on the same 11 sets (no
    fits; nearest rule; errors against `atan2(h21 - h12, 2 + h11 +
    h22)` of the exact homography, or of the converged
    translation-seeded fit on real rim points): synthetic G1 sets of
    9 cases (the off-grid twists of V10(b) plus the two combined
    rotations) clean, with Gaussian noise, and under the G4
    background (noise-free and Poisson 50, at both filter settings);
    real far-field targets rotated by skimage (16), real rim points
    (32) and rim points rotated by 2.71 or -4.42 deg (16), at both
    filter settings. Max |error| in deg:

    | set | V1 whitened per-radius (180 angles, rho .02-.40, 32 radii, hann, log) | V3 radial-mean ZNCC (360, .05-.20, Moisan, amplitude) | V8 radial-mean ZNCC (360, .02-.40, hann, log) -- D22 | V10 as V8 with Moisan | V11 as V3 untreated (raw) |
    |---|---|---|---|---|---|
    | synthetic clean | 0.225 | 0.082 | 0.123 | 0.066 | 0.084 |
    | synthetic noise 0.5 sd | 0.194 | 0.094 | 0.113 | 0.117 | 0.098 |
    | background, `(None, None)` | 0.254 | 0.228 | 0.151 | 0.149 | 80.346 (8 of 9 > 1) |
    | background + Poisson 50, `(None, None)` | 0.366 | 0.306 | 0.252 | 0.182 | 80.378 (8 of 9 > 1) |
    | background + Poisson 50, `(0.05, None)` | 0.399 | 0.303 | 0.281 | 0.223 | 0.290 |
    | real far rotated, `(None, None)` | 0.223 | 0.032 | 0.034 | 0.023 | 0.035 |
    | real rim, `(None, None)` | 0.186 | 0.772 | 0.595 | 0.606 | 0.840 |
    | real rim rotated, `(None, None)` | 0.795 | 0.769 | 0.604 | 0.623 | 0.776 |
    | real far rotated, `(0.05, None)` | 0.212 | 0.030 | 0.037 | 0.019 | 0.032 |
    | real rim, `(0.05, None)` | 0.491 | 0.759 | 0.604 | 0.592 | 0.759 |
    | real rim rotated, `(0.05, None)` | 0.799 | 0.761 | 0.614 | 0.591 | 0.762 |

    Worst over the 11 sets: V1 0.799, V3 0.772, V8 0.614, V10 0.623;
    the other crossings: V1 with the per-radius plain correlation
    0.565, V3 with the window 0.718, V3 per-radius 0.786, V1 over
    rho .05-.20 with amplitude 0.762, V3 with the window at 180
    angles 0.730, and V1
    with the Moisan correction 46.678 (one outlier, rim rotated,
    default band-pass). No variant except V11 estimated beyond 30 deg
    on any case. On the rim the whitened V1 tracked the fitted
    in-plane rotation best (median 0.049 against 0.339 for V8), but
    the second prototype's own sweep (`gridstudy2.py`, 19 variants)
    found it fragile in its neighbourhood: 720 angles gave a 61.1 deg
    outlier on the rim, rho up to 0.5 a 41.6 deg one, no window a
    46.8 deg one, while the plain-correlation variants had none. V8
    and V10 are equivalent within 0.07 deg everywhere; D22.3 takes V8
    (the window is Ernould's own edge treatment and its stencil has
    an exact one-line oracle) and keeps V10 and the amplitude and
    narrow-band variants open under FQ4.

129. **End-to-end check of the D22 recipe (requirements D22.3 to
    D22.5; drafting run `e2e.py`, 127 s).** The V8 angle, the
    engine's bicubic `evaluate` over the bounding box for the
    de-rotation, `initial_guess` on the de-rotated crop, the partial
    (P) and complete (C) rows, and the D2.7 criterion at each seed
    (`fit_pattern(max_iterations=0)`). (i) G1, `border=0.05`,
    default band-pass, budget 50, V2 corner error to the exact
    homography:

    | rotation vector (deg) | theta_hat (true) | T: conv, it, err px, criterion at seed | P: conv, it, criterion | C: conv, it, criterion |
    |---|---|---|---|---|
    | (0, 0, 1) | +1.002 (+1.000) | yes, 5, 0.0040, 0.756 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 2) | +1.994 (+2.000) | yes, 8, 0.0113, 1.647 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 2.5) | +2.497 (+2.500) | NO, 50, 148.26, 2.048 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 3) | +3.013 (+3.000) | NO, 50, 172.88, 1.990 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 4) | +4.026 (+4.000) | NO, 50, 221.43, 2.023 | yes, 3, 0.002 | yes, 3, 0.002 |
    | (0, 0, 5) | +5.007 (+5.000) | NO, 50, 97.93, 1.950 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, -3) | -2.995 (-3.000) | NO, 50, 23.11, 2.019 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, -5) | -5.000 (-5.000) | NO, 50, 164.72, 1.990 | yes, 2, 0.002 | yes, 2, 0.002 |
    | (3, 0, 0) | +0.224 (0) | yes, 9, 0.1171, 0.702 | yes, 9, 0.706 | yes, 4, 0.090 |
    | (0, 3, 0) | -0.157 (0) | yes, 9, 0.0373, 0.673 | yes, 9, 0.665 | yes, 4, 0.063 |
    | (2, -1.5, 3) | +3.267 (+3.000) | NO, 50, 53.21, 1.983 | yes, 7, 0.490 | yes, 4, 0.123 |
    | (-2, 2, 4) | +3.759 (+4.001) | NO, 50, 68.46, 1.950 | yes, 7, 0.565 | yes, 4, 0.132 |
    | (1, 1, -3.5) | -3.485 (-3.500) | NO, 50, 33.53, 1.987 | yes, 6, 0.259 | yes, 4, 0.025 |

    Every FM fit reached the error the converged T fit reaches where
    T converges (0.0040 to 0.1171 px) and 0.0078 to 0.0912 px where
    it does not; the angle's in-plane-axis bias shows on the combined
    rotations (up to 0.267 deg), inside the basin. (ii) The
    projection-centre-shift fixture: ledger 125 (iii). (iii) Real
    rim, 16 points (rows 125-128 x 115-118), `(None, None)`, budget
    500, T against the COMPLETE FM row with the acceptance (the
    frozen PARTIAL row on the same points and budgets: ledger 131
    (v)): every
    T seed exactly (0.0, 0.0); `theta_hat` 0.237 to 0.977 deg (peak
    0.58-0.77) against fitted in-plane rotations of 0.06 to 0.41 deg;
    both seeds converge on all 16 to the same residual (to 4
    digits); criterion at the seed T 1.058-1.856, C 0.369-1.767; the
    acceptance kept C on 13 points and T on 3 ((125, 118), (127,
    118), (128, 118); `theta_hat` 0.237-0.275; T and C iterations
    91/91, 122/123, 134/135); iterations total 2002 with T against
    1616 with the accepted rows (-19 per cent), e.g. 136 -> 90 and
    123 -> 65 on the far rim. (iv) Real rim, 8 points (rows 127-128 x
    115-118), `(0.05, None)`, budget 200: T converged 0 of 8; C
    converged 5 of 8 (35 to 67 iterations, residuals 0.67-0.99
    against T's capped 1.61-1.85), and the acceptance kept C on
    exactly those 5 (criterion 1.150-1.235 against 1.899-1.941) and
    T on the 3 that neither row converged (`theta_hat` 0.197-0.308).
    On these points the gain comes from breaking the D5 anchor
    (ledger 122), not from large twists: under `"auto"` at 1.5 deg
    none of them would be routed pre-fit, and the default-filter
    failures would reach the FM row through the retry (D22.8).

130. **In-frame budgets and the two capture numbers (requirements
    D22.16).** The rotation about the PC at which a subregion corner
    leaves the pattern, computed exactly (`inframe.py`, `budget.py`):
    `PC_480` 5.62 deg at `border=0.05` and 23.8 deg at 0.15; the V8
    PC 6.62 deg and 38.7 deg (corner radius 311.6 px at 0.05). The
    3.84 deg of the `TestPureRotations` comment is a conservative
    bound. Beyond the in-frame angle the mirror boundary, not the
    seed, limits accuracy, and the exact seed fares the same: at
    `PC_480`, `border=0.05`, at most 0.024 px to 6 deg, then 0.16,
    0.09, 0.025, 0.27 and 0.49 px at 8, 10, 15, 20 and 30 deg (0.30
    to 1.07 px unfiltered from 8 deg); at `border=0.15` at most
    0.0044 px to 20 deg and 0.033-0.048 px at 30. So the second prototype's recipe captured every case to
    30 deg (ledger 117(ii)) and the frozen recipe every case
    measured, to 30 deg at `border=0.15`, the edge of its search
    window (ledger 131 (i)); nothing beyond 30 deg can be captured
    with that window. Accurate results extend to the in-frame angle
    of the chosen border.

131. **Spec-review re-measurement of the frozen recipe (requirements
    D22 lead, D22.3 to D22.5, D22.7, D22.15; validation V10(b), (e),
    (f), (i); drafting runs `proto/e2e_partial.py`, 91.3 s, and
    `proto/deadband.py`, plus the critics' `subbatch_amp.py` and
    `dilution.py`).** Conditions as 115, free virtual memory 3.0 GB
    at the start. Recipe: the V8 angle of D22.3, the engine's bicubic
    `evaluate` over the bounding box for the de-rotation,
    `initial_guess` on the de-rotated crop, the PARTIAL row (P), and
    the acceptance as D22.5 (`fit_pattern(max_iterations=0)`
    criterion, strictly lower keeps P); T is the D5 seed. (i) G1,
    `PC_480`, `border=0.15`, default band-pass, budget 50, V2 corner
    error to the exact homography:

    | twist (deg) | theta_hat | P seed err px | criterion T / P | kept | P: it, err px | exact seed: it, err px |
    |---|---|---|---|---|---|---|
    | +8 | +7.978 | 0.109 | 1.984 / 0.0012 | P | 2, 0.00242 | 2, 0.00242 |
    | +10 | +9.970 | 0.154 | 1.987 / 0.0015 | P | 3, 0.00408 | 2, 0.00408 |
    | +15 | +14.973 | 0.137 | 2.028 / 0.0014 | P | 2, 0.00103 | 2, 0.00102 |
    | +20 | +19.967 | 0.169 | 2.027 / 0.0017 | P | 2, 0.00226 | 2, 0.00225 |
    | +25 | +24.954 | 0.234 | 2.021 / 0.0026 | P | 3, 0.00274 | 2, 0.00274 |
    | +29 | +28.933 | 0.302 | 2.005 / 0.0061 | P | 3, 0.01740 | 2, 0.01740 |
    | +30 | +29.931 | 0.309 | 1.977 / 0.0078 | P | 3, 0.03257 | 2, 0.03257 |
    | -20 | -20.004 | 0.018 | 2.001 / 0.0011 | P | 2, 0.00581 | 2, 0.00581 |

    The translation seed converged on none; angular peak 0.95-0.99;
    the P and exact-seed errors agree within 9e-6 px. At 30 deg, the
    window's edge bin, the estimate is 29.931 deg and the fit
    converges. (ii) The projection-centre-shift fixture of 125 (iii)
    FITTED from P, budget 50: (+8, (12, -9)) 3 iterations to 0.0027
    px; (-12, (10, 7)) 2 to 0.0016; (+5, (-8, 5)) 2 to 0.0013; (+15,
    (6, -4)) 3 to 0.0011; the acceptance kept P on all four
    (criterion 0.0011-0.0026 against 1.954-2.015). (iii) The V8 PC,
    `border=0.05`, default band-pass, budget 200, seam-level proxies
    of the ramp-map points built with the prototype fixtures (the
    orientation, PC, border and Fe of `ramp_map`; not the ramp map
    run through the engine): pure twist 3.0 deg -- T CONVERGES in 14
    iterations (0.0107 px), so 3.0 deg cannot be G8's twist; 3.5 deg
    -- T not converged at 200, finite `h`, residual 1.515, 53.0 px
    off; P kept, 3 iterations, 0.0140 px; 4.0 deg -- T not converged,
    finite `h`, residual 1.368, 42.9 px off; P kept (1.983 against
    0.0022), 3 iterations, 0.0119 px; 8.0 deg (`RAMP_UNFITTABLE`) --
    T not converged, finite `h`, residual 1.938, 144.5 px off; P kept
    (1.980 against 0.0068), 3 iterations, 0.053 px (out of frame
    beyond 6.62 deg); the 9 deg reduced tilt about x
    (`RAMP_RESCUE`) -- `theta_hat` -0.101 deg, criterion T 1.293
    against P 1.279, so P is kept; T converges in 113 iterations and
    P in 116, both 0.3888 px; G6, the unrelated pair -- `theta_hat`
    +12.009 deg (peak 0.56), criterion T 2.040 against P 2.080, so T
    is kept; T not converged at 200 with a FINITE `h` (residual
    1.888); a forced P not converged either (1.656). (iv) G3, a
    512x622 synthetic at `PC_480`, crop 460x560, seed level, angle
    error in deg, physical-frequency table against the bin-unit
    mutant: 2 deg -0.003 / -0.074; 5 deg +0.017 / -0.196; 8 deg
    +0.004 / -0.344; 15 deg +0.018 / -1.030; 20 deg +0.027 / -2.293;
    -8 deg -0.015 / +0.386. The frozen radial grid (`rho_j = 0.02 +
    j / m`) against the evidence grid (`linspace`, 176 radii at
    460x560): within 0.0005 deg on G3 and on G1 at the eight
    off-grid twists of V10(b) (G1 max |error| 0.0134 deg either
    way). (v) The real rim with P and the acceptance, the points and
    budgets of 129 (iii) and (iv), T's fits read from `e2e.jsonl`
    (same code): `(None, None)`, budget 500 -- P kept on 13 of 16
    (criterion 0.455-1.403 against T's 1.058-1.856) and T on the same
    3 as with the complete row ((125, 118), (127, 118), (128, 118):
    P 1.138, 1.556, 1.751 against 1.115, 1.491, 1.690); all 16
    converge to the same residual from either row; iterations with
    the accepted rows 1660 against 2002 for T (-17 per cent; the
    complete row 1616, -19 per cent), e.g. 145 -> 99, 136 -> 94, 123
    -> 79. `(0.05, None)`, budget 200 -- T converged 0 of 8; P kept
    on 5 (criterion 1.336-1.398 against 1.899-1.924), which converge
    in 87 to 161 iterations (the complete row 35 to 67) to the
    residuals the complete row reached (0.67-0.99); T kept on 3 (P
    1.932-1.975 against 1.923-1.941), where neither row converges.
    (vi) Dead-band cross (`deadband.py`): G1 with columns and rows
    236:244 set to the reference mean in both images and the same
    cross as the D4 `dead_band`; angle error with / without the
    cross at 1, 3, 5, 8 and -3 deg: `(0.05, None)` -0.086 / +0.002,
    -0.120 / +0.013, -0.073 / +0.007, +0.084 / +0.019, +0.067 /
    +0.005; `(None, None)` -0.066 / -0.002, -0.071 / +0.022, -0.040 /
    +0.005, +0.084 / +0.019, +0.059 / +0.007; peak 0.87-0.99. The
    cross pulls the angle towards 0 below 8 deg, at most 0.12 deg.
    (vii) Device slots of the masked full-P branch
    (`subbatch_amp.py`): the gate twist map of 127 (iii), the 57772
    fitted points outside the crater in map order (the fit order of
    a single-grain map; the real grain ordering is MTP), P = 32, 1806
    sub-batches: thresholds 1.0 / 1.5 / 2.0 deg route 1012 / 336 /
    73 points in 111 / 55 / 17 sub-batches, 3552 / 1760 / 544 slots
    (3.5x / 5.2x / 7.5x the routed count). (viii) Regression dilution
    (`dilution.py`) on the 32 rim points of 127 (v): standard
    deviation of the input twist 0.221 deg, of the fitted rotation
    0.120 deg; slope with the input twist as regressor 0.452, with
    the fitted rotation as regressor 1.545 (its inverse 0.647),
    geometric mean 0.541, median ratio input / fitted (|fitted| > 0.1
    deg) 1.19.

#### V10 recorded results, failing-tests gate (2026-10-08)

Measured on machine A (ledger 89; `nvidia-smi` idle, 12 GB physical
memory free at the start; no GPU work at this gate) by the Stage F
failing-tests scaffold agent, CPU only, on the worktree `.venv`
(CPython 3.13.12, numpy 2.4.6, scipy 1.17.1, numba 0.65.1,
scikit-image 0.26.0, orix 0.14.2), OMP and MKL threads at 2, on
a499129d plus the Stage F skeleton: `_hrebsd/_fourier_mellin.py`
(every frozen name of D22.3 to D22.7 as a stub raising
`NotImplementedError("Stage F: not implemented yet")`, the frozen
constants, literals and the `FourierMellinState` container); the
inert seam fields `SeedState.fourier_mellin=None` and
`SeedBatch.outputs={}` in `_batched.py`; `fourier_mellin="off"` in
its frozen slot on `run_hrebsd_dic` (any other value raises the
skeleton `NotImplementedError` after the `backend` string check) and
on `EBSD.hrebsd_dic` (forwarded by name); `FOURIER_MELLIN_PROP_NAMES`
re-exported by `_engine`. Scripts: session scratchpad
`measure_premises` (`m1.py` to `m13.py`), each loading the scaffold
`tests/test_indexing/test_hrebsd_fourier_mellin.py` by path and
calling its own builders and its LOCAL re-implementation of the D22.3
angle (`local_angles` and friends, written from the requirements
text, never importing `_fourier_mellin`), so the numbers are the
scaffold's own recipe.

132. **CPU-side premises of V10 (plan 12 item 1; requirements D22.3
    to D22.5, D22.7, D22.8; fixtures G1 to G8).** Every number below
    is pinned or quoted in the scaffold's MEASURED block with this
    recipe; the device-side and new-code-path literals stay
    `FIXME-pin` (`None`).
    (i) **G1 angle, local recipe** (border 0.05, default band-pass):
    max |error| 0.0134 deg over the nine twists of V10(b) (worst at
    9.61 deg; 0.37 deg -0.0002, -0.37 deg -0.0019) and 0.1155 deg on
    the two combined cases (6.13 deg about z then 1 deg about x:
    +0.0716; 3.27 then 2 about y: -0.1155, the in-plane-axis bias);
    ledger 131 (iv) had 0.0134. The look-up table: 165 and 175 radii,
    237600 and 252000 entries, weights summing to 1 within 2e-16.
    The PC-shift fixture: the local partial row seeds +8 deg with
    (12, -9) px at 0.177 px (ledger 125: 0.177) and the exact
    homography `T(s) H(fe)` of `g1_pair` fits from itself in 2
    iterations to 0.0027 px.
    (ii) **G3, local recipe**, signed error in deg at 2, 5, 8, 15, 20
    and -8 deg (crop 460x560, bounds (26, 486, 31, 591)): physical
    -0.0031, +0.0170, +0.0038, +0.0183, +0.0277, -0.0147; the
    bin-unit table (`(fx m, fy m)`, `m = min(sr, sc)`) -0.0737,
    -0.1959, -0.3444, -1.0309, -2.2919, +0.3860 -- ledger 131 (iv) to
    1e-3 deg. Premise band 0.06; the mutant is at least 3.3x beyond
    it from 5 deg on.
    (iii) **G4, the background lock** (nine cases: the seven nonzero
    off-grid twists and the two combined rotations). FIXTURE CHANGE
    against the V10 wording, recorded: a smooth background (Gaussian
    plus ramp) multiplying the RAW projected pattern (mean 44.9,
    standard deviation 28.4) never locks the untreated recipe, 0 of 9
    above 1 deg for background amplitudes from 1x up to a 200x
    Gaussian with a 100x ramp, at both filter settings, log and
    amplitude alike, because the multiplied Kikuchi contrast grows
    with the background. An ADDITIVE background of 5x the pattern mean
    locks 8 of 9. G4 therefore multiplies the background (1 + 4
    Gauss(centre (0.68, 0.30) of the frame, sigma 0.30 nrows) + 2
    column / ncols) into the raw-camera model `1 + 0.1 (p - mean) /
    std` (a 10 per cent Kikuchi contrast, an affine intensity map, so
    the exact homography is unchanged); contrasts 0.05, 0.2 and 0.3
    were measured too (0.05: treated 0.82 and 1.31 deg at Poisson 50;
    0.3: only 4 of 9 locked with amplitude). With 0.1, the D22.3
    recipe WITH the edge treatment: max |error| 0.117 / 0.177 deg
    (noise-free / Poisson 50, seed 1904 and 1905 + case) at `(None,
    None)` and 0.186 / 0.249 deg at `(0.05, None)` (spec gate 0.151,
    0.252, 0.281); WITHOUT it, the untreated amplitude variant over
    rho [0.05, 0.20] (ledger 128's V11): 8 of 9 above 1 deg at
    `(None, None)` both noise-free and at Poisson 50 (worst 19.1 deg:
    the 30 deg window caps the lock at zero, where ledger 128's
    nearest rule reached 80 deg), 0 of 9 at `(0.05, None)`; the
    untreated LOG variant also 8 of 9. `FM_LOCK_PREMISE_COUNT =
    {"none": 8, "poisson50": 8}`, pinned at the measured count.
    (iv) **Dead band**, local recipe, 8 px cross at columns and rows
    236:244 in both images and `dead_band=(236, 244, 236, 244)`:
    -0.0701 / +0.0838 deg at 3 / 8 deg under `(None, None)`, -0.1193
    / +0.0841 under `(0.05, None)` (ledger 131 (vi) to 1e-3 deg).
    (v) **G5 BUILT** (both run-time premises hold; its (g) arms are
    KEPT): the G1 pair of a 2.5 deg twist and a (14, -10) px PC shift,
    both images times the same white gain `1 + 0.02 N(0, 1)` plus the
    same white offset `0.02 std(reference) N(0, 1)` (seed 2205),
    default band-pass. (i) `initial_guess` returns exactly (-0.0,
    -0.0) (the clean pair: (-31.875, -0.125)); (ii) local angle 2.468
    deg (error -0.032). Scan: gains and offsets in {0, 0.02, 0.05,
    0.1, 0.2, 0.4} all gave an exactly zero seed and angle errors of
    at most 0.083 deg under both filter settings. At 0.02: criterion
    at the seed 2.1104 (translation row) against 0.0145 (local FM
    row); `"off"` fit not converged at 50 (28.58 px off); local
    FM-seeded fit converged in 4 iterations to 0.0068 px. At 0.1 the
    `"off"` fit converged in 8 iterations to a wrong optimum 30.68 px
    away (the false convergence of D22.8), which is why 0.02 is used.
    (vi) **G6** (V8 PC, default band-pass, budget 200), with the
    target orientation `Rotation((0, 1, 1), 20 deg) * orientation_a()`
    (orix product, extra rotation on the LEFT): local angle +12.008
    deg; criterion at the seed 2.040 (translation) against 2.080
    (local FM row), so the acceptance refuses; translation-seeded fit
    not converged at 200 with a finite `h`, residual 1.888; forced
    local-FM fit not converged, residual 1.656 -- ledger 131 (iii) to
    the printed digits. The other compositions do NOT reproduce it:
    `orientation_a() * R` gave 12.877 deg and criteria 1.985 / 1.986
    (forced 1.953); `orientation_a() * ~R` -12.858 deg, 2.050 / 1.506
    (the FM row would be KEPT); `~R * orientation_a()` -12.164 deg,
    2.006 / 1.555; the absolute orientation `R` alone converges from
    the FM row (18 iterations), so it is not unrelated.
    (vii) **G8** (V8 geometry, default band-pass, budget 200;
    `fit_pattern` from the D5 seed per column): reference and easy 0
    deg converge in 1; easy 0.8 deg in 5 (0.0019 px); the mislabelled
    4.0 deg twist NOT converged, finite `h`, 42.85 px off, residual
    1.368, while the local FM row (angle 4.030 deg) converges in 3
    iterations to 0.0119 px (ledger 131 (iii): 42.9 px, 3); the
    unrelated column as G6. PREMISE NOT HOLDING AS WORDED, the
    constant pattern: under the default band-pass the preprocessed
    constant crop is rounding noise (standard deviation 4.0e-16, not
    0), so `initial_guess` and the numpy seam BOTH return a FINITE
    translation row (0.75, 167.0) px and the local FM estimate does
    not fail (angle 5.05 deg, peak 0.29); only `fit_pattern` refuses
    the pattern (the D2.6 contract: 0 iterations, NaN `h`, from any
    `h0`, zeros included). Under `(None, None)` the crop is exactly
    constant and the translation row is NaN. So on the V8 ramp map
    and on G8 (both default band-pass) the forced FM row of the
    constant pattern is not provably a failed estimate: unless the
    implementation adds a guard, it is FITTED in the retry and returns
    the D2.6 contract again, keeping the first result bitwise. The
    V10(f) and (i) arms "its forced FM estimate fails, so it is never
    refitted" and "it gets no second fit (spy)" need a decision at
    insertion: assert the outcome (D2.6 contract bitwise, code 0,
    angle NaN or finite as measured) and pin the second-fit count as
    measured, or run the constant pattern under `(None, None)`.
    Recorded as `G8_CONSTANT_TRANSLATION_ROW_FINITE = True`.
    (viii) **The D5 failure premise of (e)**, `run_hrebsd_dic` on the
    one-row maps of `g1_row_map`, `max_iterations=50`, `"off"`: every
    arm fails. Border 0.05: twists 2.5*, 3*, 4, 5*, -3*, -5 at 148.26,
    172.88, 221.43, 97.93, 23.11, 164.72 px; rotation vectors (2,
    -1.5, 3)*, (-2, 2, 4), (1, 1, -3.5) at 53.21, 68.46, 33.53 px
    (ledger 129 to the printed digits). Border 0.15: 8, 10, 15, 20,
    25, -20 and 30 deg at 39.52, 81.23, 79.13, 154.59, 138.15, 120.27
    and 175.08 px.
    (ix) **The V8 ramp premise of (f)** (`seed_from_neighbors=False`,
    budget 200, the V8 mask): columns 0 to 2 converge in 1, 5, 10
    iterations (0, 0.0019, 0.0056 px); columns 3 to 6 do not (47.66,
    48.70, 42.85, 105.24 px); the 8 deg point does not (144.50 px);
    the 9 deg tilt converges in 113 iterations to 0.3888 px; the
    constant pattern gives the D2.6 contract (0 iterations); the prop
    set is exactly the pre-Stage-D one. Bitwise the V8 record.
    (x) **G7 projection link** (the (h) arm's premise): the pattern
    projected from `g_t = g_r (M^T R_det M)^T` equals the
    detector-frame deformation construction of `R_det` to 5.5e-13 and
    4.2e-13 relative (twists 2.5 and -4.8 deg, default detector) and
    3.8e-13 and 1.2e-12 (tilted detector 10, 4, 1.5 deg), inside the
    frozen 1e-10 (ledger 127: 4e-14 to 1.4e-12). The special map's
    phase list carries the not-indexed phase -1, `ni` and `al`, and
    exactly one NaN rotation.
    (xi) **Default path and gates.** `fourier_mellin="off"` equals the
    call without the keyword bitwise on F6 (`run_hrebsd_dic`) and on
    the Ni map (`EBSD.hrebsd_dic`, every prop); `"auto"` and
    `"always"` raise the skeleton `NotImplementedError`. `uv run
    pytest tests -k hrebsd -n 2 -q`: 4 failed, 901 passed, 287
    skipped in 246 s (re-run after the audit fix below), the four failures exactly the intended
    pin move of the amended D21.12 literal
    (`SEED_FROM_NEIGHBORS_GPU_MESSAGE`; `TestBackendSwitch::
    test_backend_checks_run_in_the_frozen_order` and three
    `TestSeedFromNeighborsOnGpuRaises` arms), which the implementation
    turns green by amending the engine literal. The `FROZEN_SIGNATURE`
    and `test_run_defaults_are_frozen` edits pass with the skeleton's
    keyword. A first run also failed `TestImportAudit::
    test_no_new_required_dependency` on a docstring line of
    `_fourier_mellin.py` beginning with "import of" (the audit reads
    lines); reworded, green. The scaffold collects 0 tests (the
    classes are inserted at its four markers), so the default-suite
    wall time and the per-class fit counts of V10's intro are recorded
    when the classes are inserted. `ruff check` and `ruff format`
    clean on every touched file.

133. **The failing-tests gate tally and the default-suite wall time
    (plan 12 item 1; V10 (a) to (l)).** Same machine, environment
    and skeleton as ledger 132; the three class blocks (written
    separately, (a) to (e), (f) to (k) and (l)) spliced into
    `tests/test_indexing/test_hrebsd_fourier_mellin.py` at their
    markers, their imports merged into the module header (`contextlib`,
    `re`, `subprocess`, `threading`, `time` and `dask.array` added; the
    scaffold's `noqa: F401` markers dropped, every name now used), and
    the mutation map filled: FM1 to FM52 each name their designed
    killer(s) in this module; FM39 alone has NO KILLER YET, recorded
    reviewed-equivalent (the G7 maps carry one projection centre and
    the twist depends only on the detector tilt chain, the
    `TestFourierMellinGate` docstring). `ruff check` and `ruff format`
    clean; no duplicate module-level name across the blocks (the (f)
    to (k) helpers are `fk_`-prefixed, the (l) copies of
    `test_hrebsd_gpu.py` names `GATED_FM_` / `_gated_fm_`-prefixed).
    (i) **Default suite**, `uv run pytest
    tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0 -q -p
    no:cacheprovider`: 130 failed, 10 passed, 117 skipped (12 weekly:
    the F6 numpy-session `"off"` pin and 11 capture arms; 105 gated),
    37.5 s in pytest, 46 s wall. Per class (failed / passed, seconds
    at `-n 0`): Switch 16 / 9, 8.7; Angle 13 / 0, 2.6;
    EdgeTreatment 10 / 0, 2.5; SeedRows 18 / 0, 3.0; Capture 5 / 0,
    4.1; RampRescue 7 / 0, 5.7; Acceptance 10 / 0, 6.1; Gate 22 / 0,
    0.4; Retry 8 / 0, 0.2; CpuRoute 6 / 0, 0.0; NumpySession 15 / 1,
    3.1. Every failure is for the right reason: 126 the skeleton's
    `NotImplementedError("Stage F: not implemented yet")`, 63 from
    the engine's skeleton raise and 63 from `_fourier_mellin` stubs
    (36 `build_fourier_mellin_state`, 15 `twist_about_detector_normal`,
    the rest the look-up-table, stencil, peak, partial-row and route
    stubs), each
    after every run-time premise that precedes the FM call held; 2 `TypeError` on the
    missing `fourier_mellin` keyword of `_gpu._vram_model_terms` and
    `_gpu._vram_model_bytes` (D22.18, an implementation edit);
    `test_the_fourier_mellin_path_imports_no_cupy`, whose child
    process raises the stub's `NotImplementedError`; and
    `test_the_docstring_documents_fourier_mellin` (the
    `EBSD.hrebsd_dic` docstring has no `fourier_mellin` entry yet).
    No premise assertion, import, fixture or name error. The 10
    passes are the ones that must pass on the skeleton: the `"off"`
    bitwise pins (F6 and the Ni map on the CPU, the Ni map through
    the numpy session), the two FM16 no-work spies under `"off"`, the
    forwarding spy (FM41), the prop-name re-export, the two
    frozen-slot signature arms and the static import-hygiene check.
    The per-class FIT counts of V10's intro need the FM path and are
    recorded at the implementation gate; the writers' estimate once
    implemented is about 35 to 45 s for (k) and about 15 s for each
    cached G8 run shared by (g) and (i).
    (ii) **Gated suite** through the PINNED overlay of D21.15,
    `KIKUCHIPY_EXPECT_GPU=1`, `-n 0 --weekly -k TestGated`
    (`nvidia-smi` idle, 0 MiB used, before the run): 105 collected, 99
    failed, 6 passed, 21.2 s in pytest, 41 s wall including the
    overlay resolution. All 99 failures are the stub's
    `NotImplementedError` (74 from `build_fourier_mellin_state`, 25
    from the engine); the 6 passes are the `"off"` device default
    path and the D21.8(h) drift tripwire under `"off"` (both device
    precisions) and the record-only `"off"` throughput arms. Per
    class (failed / passed): Contract 10 / 4, Parity 54 / 0,
    Determinism 18 / 0, Vram 11 / 0, Throughput 6 / 2.
    (iii) **Stage E and earlier**, `uv run pytest tests -k hrebsd -n
    2 -q`: 134 failed, 911 passed, 404 skipped in 213 s (221 s wall):
    the 130 FM failures of (i) and exactly the four intended failures
    of the amended D21.12 pin recorded in ledger 132 (xi); 911 = 901
    of ledger 132 plus the 10 FM passes. Nothing else changed.

134. **Critic disposition of the failing-tests gate (plan 12 item 1;
    V10 (b), (d), (g), (h), (i), (k)).** A read-only critic reviewed
    the gate (no blockers, 2 major, 7 minor); same machine,
    environment and skeleton as ledger 132. Every finding was
    re-checked before it was applied. Disposition:
    (F1, major, APPLIED) the gated arm of V10(b) was missing:
    `TestGatedFourierMellinContract::test_the_peak_rule_on_the_device`
    feeds the planted correlations (the 40 / 5 deg window pair, the +6
    / -6 lag tie, the edge peak at lag 60, the flat row and the
    plateau at lag -60) to `fourier_mellin_peak(cupy, ...)`. It asserts
    `cupy.ndarray` float64 `(6,)` outputs, the V10(b) values, and host
    values bitwise equal to the numpy call. It is the FM8 gated twin.
    (F2, major, APPLIED) the FM12 separations are now asserted as local
    premises through the new `local_phase_peak(s)` helpers. These take
    the maximum of the real inverse FFT of the NORMALISED cross-power
    spectrum of the zero-mean unit-norm crops, translation crop against
    the crop de-rotated by the local angle. G5: 0.2878 (`h_T`) against
    0.2528 (FM row), while the criteria are 2.110 / 0.0145, so
    acceptance by the peak refuses the row the G5 arm keeps. G6: 0.0174
    against 0.0206, opposite to the criteria 2.040 / 2.080, so
    `test_the_unrelated_pair_refuses_the_fm_row` kills FM12 too (the
    critic read 0.0179 for G6's `h_T`, from a slightly different peak
    recipe; the ordering is the same). The G5 margin is about 12 %,
    and the UNNORMALISED cross-correlation orders G5 the other way
    (0.084 against 0.933), so the peak definition is part of the
    premise. This is stated beside the helper. Mutation map FM12 lists
    both arms.
    (F3, minor, APPLIED) the `"always"` retry subset is pinned
    literally, `[G8_UNRELATED, G8_CONSTANT]` (4, 5), both route 2. The
    expression derived from the outputs stays only as a consistency
    check. Map: FM25 and FM47.
    (F4, minor, APPLIED) the lazy-build arm now counts one FM-state
    build per `_run_chunks_gpu` pass that fits a routed-grain point.
    The arm stays robust to a D22.8 retry in a new session, and a
    rebuild per sub-batch still fails it. Every build must be the
    routed grain's reference and hold that pass's resident.
    (F5, minor, APPLIED) the `_run_chunks`, `_run_chunks_gpu` and
    `run_lockstep` spies take `*args, **kwargs`. They read their
    arguments through `inspect.signature(...).bind` and forward the
    call unchanged, so they do not care whether an argument arrives by
    position or by keyword.
    (F6, minor, APPLIED) the constant non-identity arm of
    `test_a_map_without_twists_warns_and_routes_nothing` first asserts
    that the production twists are not all exactly 0.0 and all lie
    below `FROZEN_ZERO_TWIST_DEG` (D22.7: of order 1e-15). The
    magnitude is recorded at the implementation gate. On the skeleton
    this arm now raises from the `twist_about_detector_normal` stub
    instead of the engine (the 63 / 63 split of ledger 133 (i) becomes
    64 / 62).
    (F7, minor, PARTLY APPLIED: recorded, NOT decided) the G8 constant
    pattern under the default band-pass (ledger 132 (vii)) leaves a
    behavioural choice the spec has not made. D22.8 and V10(f)/(i)
    still say "never refitted". This gate may not edit requirements or
    plan text, so the decision is recorded here as OPEN and must get a
    dated disposition in plan 12 / V10 BEFORE the implementation.
    Option A: no guard, pin `FM_G8_CONSTANT_SECOND_FITS = 1` and amend
    the D22.8 sentence. Option B: a specified degenerate-crop guard,
    pin 0. The FIXME-pin comment now says the literal takes the DECIDED
    value, not "as measured".
    (F8, minor, APPLIED here) deviation from plan 12 item 1, recorded:
    the default-suite wall time of ledger 133 (i), and the one below,
    were measured against stubs. They are a LOWER BOUND, not the CI
    cost, and the per-class FIT counts move to the implementation
    gate. The FK-MIXED premises of the module comment, copied into the
    ledger: `"off"` at budget 20 converged (T, T, T, F, F) on points 0
    to 4 in (1, 5, 10, 20, 20) iterations, every `h` finite. The 2.4
    deg mislabelled point is 23.4 px off, and its local FM row (angle
    2.3988 deg) converges in 2 iterations to 0.0096 px. The 1.6 deg
    point's seed criteria are 1.6251 (translation) against 0.00087
    (FM). The unrelated point gives 2.0402 / 2.0801, and its forced
    local-FM fit does not converge at 20. A direct `fit_pattern` is
    bitwise the engine's stored result. FK-TWO-GRAIN: fit order (0, 2,
    4, 1, 3), routes {2, 4} on the label-1 grain, references (3, 0).
    CI-cost scale (the critic's measurement): FK-MIXED `"off"` on the
    CPU 0.66 s, through the numpy session 2.6 s, G8 `"off"` 3.2 s.
    (F9, minor, APPLIED) `local_seed_row_errors` gains the FM38 mutant
    `reused`: `R(theta) T(t_T)`, where `t_T` is the `initial_guess` of
    the UN-de-rotated preprocessed crop. Re-measured on the four
    projection-centre-shift cases (8 deg (12, -9) px, -12 deg (10, 7),
    5 deg (-8, 5), 15 deg (6, -4)):
    - partial row: 0.177, 0.102, 0.129, 0.221 px;
    - `T(t) R` (FM2): 2.225, 2.653, 0.950, 1.948 px;
    - detector-centre de-rotation (FM43): 7.559, 11.384, 4.771, 14.133
      px;
    - reused translation (FM38): 112.7, 13.0, 8.9, 6.6 px.
    The premise `partial < min(reused)` and the kill separation
    `FM_SEED_TOL_PX < min(reused)` are asserted beside FM2 and FM43.
    Rejected: none.
    **Re-run, final counts.** (i) Default suite, `uv run pytest
    tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0 -q -p
    no:cacheprovider`: 130 failed, 10 passed, 118 skipped (12 weekly,
    106 gated), 47.4 s in pytest, 56 s wall (a stub-time lower bound,
    F8). Every failure is for the right reason:
    - 126 skeleton `NotImplementedError` (64 from `_fourier_mellin`
      stubs, 62 from the engine);
    - 2 `TypeError` on the `_gpu` VRAM keyword;
    - the import-hygiene child process (the stub's
      `NotImplementedError`);
    - the docstring test.
    The new premises (G5 and G6 peaks, FM38) HOLD: their tests reach
    the stub. Per class (failed / passed / skipped): Switch 16 / 9 / 1,
    Angle 13 / 0 / 0, EdgeTreatment 10 / 0 / 0, SeedRows 18 / 0 / 0,
    Capture 5 / 0 / 11, RampRescue 7 / 0 / 0, Acceptance 10 / 0 / 0,
    Gate 22 / 0 / 0, Retry 8 / 0 / 0, CpuRoute 6 / 0 / 0, NumpySession
    15 / 1 / 0. (ii) Gated suite through the PINNED overlay,
    `KIKUCHIPY_EXPECT_GPU=1`, `-n 0 --weekly -k TestGated`
    (`nvidia-smi` idle, 0 MiB used): 106 collected, 100 failed, 6
    passed, 0 skipped, 31.6 s in pytest, 69 s wall. All 100 failures
    are the stub's `NotImplementedError` (75 from `_fourier_mellin`,
    including the new peak arm at `fourier_mellin_peak`, 25 from the
    engine); the 6 passes are those of ledger 133 (ii). Contract 11 / 4,
    Parity 54 / 0, Determinism 18 / 0, Vram 11 / 0, Throughput 6 / 2.
    (iii) `uv run pytest tests -k hrebsd -n 2 -q`: 134 failed, 911
    passed, 405 skipped in 219 s (227 s wall). The failures are the
    130 of (i) plus exactly the four intended D21.12 pin moves of
    ledger 132 (xi), with nothing else; 405 = 404 + the new gated arm.
    Mutation map: FM1 to FM52 each name a designed killer, except FM39,
    which is still reviewed-equivalent (ledger 133). `ruff check` and
    `ruff format --check` are clean.

#### V10 recorded results, implementation gate (2026-10-08)

135. **The `_fourier_mellin` module under numpy and its seed-level pins
    (plan 12 item 2, implementer A; requirements D22.3 to D22.7).**
    Machine A CPU (i7-13700H, Windows 11 build 26200; worktree .venv,
    CPython 3.13.12, numpy 2.4.6, orix 0.14.2), scratch
    `fmA_measure1.py` and `fmA_measure2.py`, every number through the
    PRODUCTION frozen names on the numpy seam (`numpy_fm_seam`):
    - angle (V10(b)): G1 twists max 0.0134 deg (9.61 deg), G1_COMBINED
      0.0716 and 0.1155, G5 -0.0323, the projection link 0.0032 and
      0.0003 (untilted, 2.5 and -4.8 deg) and 0.0048 and 0.0037
      (tilted); each equal to the local recipe of ledger 132 to the
      printed digit. `FM_ANGLE_TOL_DEG = 0.25`; the FM9 flipped-offset
      error on the sweep is 0.420 (1.7x the pin);
    - G3 460x560: 0.0277 deg (20 deg). `FM_ANGLE_TOL_NONSQUARE_DEG =
      0.06`, 3 x 0.06 = 0.18 < 0.1959 (FM3 from 5 deg);
    - dead band: 0.0838 deg at `(None, None)`, 0.1193 at `(0.05,
      None)`. `FM_ANGLE_TOL_DEADBAND_DEG = 0.25`;
    - stencil against the second FFT (V10(c)): 4.58e-16 relative.
      `FM_STENCIL_RTOL = 1e-15`;
    - G4 background: 0.117 and 0.177 deg at `(None, None)`, 0.186 and
      0.249 at `(0.05, None)` (noise-free, Poisson 50).
      `FM_ANGLE_TOL_BACKGROUND_DEG = 0.5`;
    - seed rows, projection-centre-shift fixture at border 0.15, route
      2 (V10(d)): 0.177, 0.102, 0.129, 0.221 px (the local partial row
      to the digit). `FM_SEED_TOL_PX = 0.45`; kill separations FM2
      0.950 (2.1x), FM43 4.77, FM38 6.64 px;
    - G5 end to end on the CPU route under `"always"`: the FM-seeded fit
      converges in 4 iterations (`FM_G5_ITERATIONS = 4`, a count, at
      the measured value; the local premise's 4);
    - gate (V10(h)): worst twist recovery 8.1e-14 deg over the G7
      twist, two-grain and special maps on both detectors; the pure
      out-of-plane arms give 4.9e-14 and 2.1e-14 to 3.0e-14 deg.
      `FM_GATE_TWIST_TOL_DEG = 1e-12` (about 12x, a rounding-level
      number; 2x would sit at the ulp noise of the quaternion round
      trip). Noise population (0.5 deg about random axes on a 2.5 deg
      twist, untilted): 32 of 32 routed;
    - FIXER NOTE (2), the constant NON-identity map (`g7_uniform_map(
      "constant")`, every point at grain A's orientation): the twist
      is -1.766e-31 deg on every point, NOT exactly 0.0, so the FM36
      float-equality kill HOLDS (`any(twist != 0.0)` and every `|twist|
      < 1e-9`); the all-identity map gives exactly 0.0. The magnitude
      is far below D22.7's quoted "order 1e-15", so the kill rests on
      a rounding residue that another BLAS or platform could make
      exactly zero: recorded as a fragility for the review gate.
    Decisions recorded: the reference profile is built in complex128 and
    float64 from the reference's zero-mean unit-norm D5 crop whatever
    the seed precision (D22.12); the look-up table is cached once per
    crop shape on the host and uploaded once per `SeedContext` (a weak
    key cache, D22.3.3); `fourier_mellin_translate` reaches the Stage E
    body `_batched._translation_rows` (the function
    `seed_homographies` runs for `h_T`), never the seam's patch point,
    so a spy on `seed_homographies` sees one call per sub-batch; the
    D22.7 warning is issued by the engine, not the module; the
    criterion is called twice at the full P only when a slot carries
    route 1, and a NaN-free `peak` row is required for a finite angle
    (a non-finite correlation row gives a NaN angle, the estimate's
    failure). Default classes Angle, EdgeTreatment, SeedRows,
    Acceptance and Gate: 73 passed at `-n 2` (53 s). Module coverage
    over the default file 98 per cent; uncovered: the `"mixed"` box
    columns (gated suite), the `None` return of
    `build_fourier_mellin_state` (non-finite reference profile) and the
    no-point-group warning of the gate, which the review gate's tests
    must reach.

136. **Integration close: the deliverables audit (plan 12 Deliverables
    and item 2; requirements D22.1 to D22.18).** The implementation
    workflow of 2026-10-08 07:12 to 08:05 lost its three implementer
    agents (the `_fourier_mellin` module and gate; the engine, seam,
    CPU route and retry; the GPU wiring) to a network outage at 07:33
    to 07:39 before they reported, and its integrator was stopped at
    08:05; their work was checkpointed uncommitted-then-WIP at
    46a9b6d8, and only implementer A's ledger entry (135) was written.
    This close audited the tree against plan 12 item by item, reading
    the code, and found every item PRESENT; no functional gap needed
    implementing.
    ONE conformance defect was found by the final `-k hrebsd` run and
    FIXED here: D22.17 (no import outside the D21.13 audit's allowed
    tuple) was violated by three standard-library imports the
    interrupted run added -- `functools` (the host look-up-table
    `lru_cache`) and `weakref` (the per-context upload cache) in
    `_fourier_mellin.py`, and `threading` (a thread-local for the
    first pass's final B) in `_gpu.py`; `TestImportAudit::
    test_no_new_required_dependency` failed on `import functools`.
    Replaced, behaviour unchanged: a module dictionary `_HOST_LUTS`
    keyed by crop shape; the per-context uploads cached on the
    `SeedContext` itself (attribute `_fourier_mellin_luts`, so they
    live and die with the context, as the weak-key cache did); and a
    module dictionary `_RUN_STATE` for the final B (one run's passes
    are sequential calls from one thread). Every suite was re-run
    after the fix (entry 139). Items audited:
    - D22.1 / D22.11: `fourier_mellin="off"` keyword on both entries;
      `_check_fourier_mellin` after the `backend` string check and
      before every `"gpu"`-only check, in the frozen order (value with
      the D21.1 message shape, a bool refused; the D22.11 combination
      raise; `"auto"` without `xmap`); the D21.12 literal amended
      (`SEED_FROM_NEIGHBORS_GPU_MESSAGE`, the "is planned" clause gone);
      the FM settings never enter `fit_options`.
    - D22.3: `fourier_mellin_lut` in physical frequency, built once per
      crop shape on the host (read-only, `lru_cache`) and uploaded once
      per context (weak-key cache); the exact frequency-domain Hann
      stencil on a NEW complex128 array; `log1p(|X_w|)`; the radial
      mean, zero mean and unit norm (NaN on a zero or non-finite norm);
      the circular ZNCC by length-360 FFTs; `fourier_mellin_peak` with
      the window, first-maximum (lowest output index) ties and the
      raw-neighbour parabolic offset.
    - D22.4: the box resident (every pixel of the D5 box, dead band
      included); `fourier_mellin_derotate` through `kernels.gather`
      about the grain reference PC; `fourier_mellin_translate` reusing
      the Stage E body `_batched._translation_rows` unchanged on the
      de-rotated crops; `fourier_mellin_partial_row`, `W0 = R T(t)`.
    - D22.5: `fourier_mellin_criteria` through `kernels.gather` on
      `fm_state.resident` with `_batched.initial_shifts` K and
      `kernels.final_criterion`, called twice at the full P only when a
      slot carries route 1; strict `<`, ties and NaN keep `h_T`; every
      failure returns `h_T` with `applied` False; a non-finite `h_T` is
      never rescued.
    - D22.6: `SeedState.fourier_mellin`, `SeedBatch.outputs`, the route
      check on the host before any device work, `h_T` first, the masked
      full-P branch reached through the module object, the outputs
      rule (angle NaN on route-0 and padded slots); every listed stage
      reached through module globals; no module-scope `_engine`,
      `_batched` or cupy import in `_fourier_mellin`.
    - D22.7: `twist_about_detector_normal`, `fourier_mellin_routes`
      (fails open on NaN), the frozen warning issued by the engine on
      the input condition `|twist| < 1e-9` (never float equality).
    - D22.8 / D22.10: the CPU route at P = 1 inside `_fit_chunk`, its
      route block paired by the Stage D blockwise index
      (`_fit_chunk_routed`), the numpy `SeedState` with its FM state
      built once per reference with a routed point, before the graph;
      a route-1 point whose FM row lost runs `h0=None` (bitwise
      `"off"`), a forced slot not applied is not fitted; 14-wide rows
      on FM runs only; the retry under both modes over the not
      converged, translation-seeded points of a grain with an FM state
      (NaN-`h` points included; the D22.8 option-A amendment: no
      degenerate-crop guard), replacing wholly only when converged,
      the device retry at the first pass's FINAL B (`_last_batch_size`,
      thread-local, reset before the first pass).
    - D22.9: the two props and the seed codes from `fm_applied` of the
      pass whose result is stored; `FOURIER_MELLIN_PROP_NAMES`.
    - D22.11 / D22.18 device: `_run_chunks_gpu(seed_extras=)` sliced per
      sub-batch slot (0 on padded slots); the lazy FM state in the
      session, built once per residency the first time a batch of the
      grain carries a nonzero route, evicted with the resident; forced
      slots not applied made inactive before the lockstep; the 14-wide
      `row_slots`; `fourier_mellin=` keyword-only on
      `_vram_model_terms`, `_vram_model_bytes` and
      `_default_batch_size`, the latter choosing B from the `"off"`
      model first and halving only where the FM terms do not fit; the
      engine passes `fourier_mellin=True` iff a route flag is nonzero.
    - `ebsd.py`: the keyword forwarded, `FOURIER_MELLIN_PROP_NAMES`
      appended to the `_engine` import block, the prop loop extended,
      and the D22.16 docstring (the entry, the Notes paragraph with
      both capture numbers, the props and the cost, the two `Raises`
      entries, the warning, the rewritten Limitations paragraph; no
      stage letters); the D16 information message's FM host-bytes line.
    NOT done here, by this close's scope (no `doc/` or CHANGELOG edit
    allowed): the CHANGELOG entry, the `hrebsd_dic.ipynb` bullet and
    the `hrebsd_si_indent.ipynb` cell of D22.16, which plan 12 item 2
    places after the performance record anyway. The real-data
    measurements of plan 12 item 3 (V10(m)(1) frame oracle, V10(m)(2)
    whole-map record) are NOT part of this close either.
    Test-module edits by the interrupted run, recorded here as the
    ledger they cite: (i) TEST BUG, provable: `fk_spies` installed over
    a first `fk_spies` captured the first's `(*args, **kwargs)` spies,
    so `inspect.signature(...).bind` could not bind `fit_indices`; each
    spy now carries `__wrapped__`, which `inspect.signature` follows
    (no assertion changed); (ii) every `FIXME-pin` placeholder replaced
    by a measured literal with its recipe and machine-A comment; (iii)
    `FM_G8_CONSTANT_SECOND_FITS = 1`, the DECIDED option A of plan
    12.6. This close's own test edits: the `FM_ACCEPT_FLIP_COUNT`
    comment corrected (entry 137) and the stale "FIXME-pin:" label of
    the G8 literal's comment reworded; no assertion and no literal
    changed.

137. **Integration close: every pin re-measured against the FINAL code
    (46a9b6d8 plus this close's edits of entry 136; machine A CPU and GPU,
    the recipes of ledger 135 and of the pins' comments, re-run
    unmodified: scratch `fmA_measure1.py`, `fmA_measure2.py`,
    `impl_b/m_capture.py`, `impl_b/m_ramp.py`, `impl_b/m_numpy.py`,
    `integ/parity.py`, `integ/parity2.py`, `integ/e2e.py`,
    `integ/vram_r.py`, plus `close/flip.py`, `close/fm12.py`,
    `close/g3db.py` and the gated suite's `record_property` values).**
    Every number reproduced the interrupted run's to the printed digit;
    every pin is KEPT (value unchanged), its ~2x margin and its stated
    kill separation holding:
    - `FM_ANGLE_TOL_DEG = 0.25`: worst use 0.1155 deg (G1_COMBINED),
      G1 twists 0.0134, G5 0.0323, projection link 0.0048 (2.2x);
      FM9's flipped offset 0.4197 (1.68x above the pin).
    - `FM_ANGLE_TOL_NONSQUARE_DEG = 0.06`: 0.0277 deg (worst of 2 to 20
      and -8 deg; 2.2x); FM3: 3 x 0.06 =
      0.18 <= 0.1959.
    - `FM_ANGLE_TOL_DEADBAND_DEG = 0.25`: 0.0838 deg at `(None, None)`, 0.1193
      at `(0.05, None)` (2.1x).
    - `FM_STENCIL_RTOL = 1e-15`: 4.58e-16 (2.2x).
    - `FM_ANGLE_TOL_BACKGROUND_DEG = 0.5`: 0.117 / 0.177 at `(None,
      None)`, 0.186 / 0.249 at `(0.05, None)` (2.0x).
    - `FM_SEED_TOL_PX = 0.45`: 0.102 to 0.221 px (2.03x); separations
      FM2 0.950 (2.1x above), FM43 4.77, FM38 6.64 px.
    - `FM_CAPTURE_TOL_PX = 0.2`: worst 0.0912 px (rotation vector (-2,
      2, 4)), starred arms 0.0170 (2.2x); `FM_CAPTURE_ITERATIONS = 6`:
      border 0.15 arms 2 to 3 iterations (2x); `FM_FIXED_POINT_TOL_PX =
      2.5e-5`: worst gap 1.13e-5 px at 20 deg (2.2x); the 30 deg edge
      29.930 deg, 3 iterations, 0.0326 px.
    - `FM_RAMP_FAR_TOL_PX = 0.11`: 0.0531 px in 3 iterations under both
      modes (2.1x); `FM_RAMP_RESCUE_ALWAYS_ITERATIONS = 116` and
      `FM_RAMP_ALWAYS_CODES` exact (the `"auto"` rescue 113
      iterations); the constant pattern (index 19) carries code 0 and
      the retry's finite angle 5.0544 deg under both modes (option A:
      one zero-iteration refit, first result kept).
    - `FM_G5_ITERATIONS = 4`, exact. FM12's separations, the
      NORMALISED cross-power peak (`local_phase_peaks`): G5 `h_T`
      0.2878 against `h_FM` 0.2528 (the peak would refuse the FM row
      the criterion keeps; applied True), G6 0.01741 against 0.02057
      (the peak would keep the FM row the criterion refuses; applied
      False) -- both orderings hold.
    - `FM_GATE_TWIST_TOL_DEG = 1e-12`: worst 8.08e-14 deg; the pure
      out-of-plane arms 4.93e-14 / 4.90e-14 (untilted) and 2.07e-14 /
      2.98e-14 (tilted). FIXER NOTE (FM36), re-logged: the constant
      NON-identity map gives a twist of -1.76556e-31 deg on every
      point, not exactly 0.0 (the all-identity map gives 0.0), so the
      float-equality kill HOLDS on this machine; the fragility of
      ledger 135 (a residue 1e16 below D22.7's "order 1e-15") stands
      for the review gate. Noise population: 32 of 32 routed.
    - `FM_NUMPY_ANGLE_TOL_DEG = 2e-14`, `FM_NUMPY_ROW_TOL_PX = 1e-12`:
      NOT bitwise; seed codes equal ([0 0 1 2 0 -1] both), angles
      within 1.78e-15 deg (11x), converged homographies within 8.53e-14
      px (12x), iterations equal.
    - `FM_ANGLE_PARITY_DEG = 3e-7`: worst 5.77e-15 deg at complex128,
      1.577e-7 at complex64 (G4-poisson50-default; 1.9x), end to end
      2.30e-8. FM37 separation (all FM arithmetic in complex64/float32,
      `parity2.py`): G1 4.29e-7, G3 2.27e-7, G4-none-open 1.52e-6,
      G4-none-default 4.74e-7, G4-poisson50-open 2.83e-6,
      G4-poisson50-default 1.19e-6 -- above the pin on five of six sets
      (G1 only 1.43x; G3 below); the stencil-only variant 1.73e-7, below
      the pin, is killed by the default dtype spy only.
    - `FM_ACCEPT_FLIP_COUNT = 2`: CORRECTED RECORD. The pin comment said
      0 flips at the seam on every key; the re-run (and the interrupted
      run's own `integ/parity.log`) shows ONE seam flip, at ("mixed",
      "complex128") route 1 on G1, slot 0 (the zero twist): device
      criteria 3.111e-15 (`h_T`) against 3.017e-15 (`h_FM`) through
      the f32 gather, an exact tie at 1.765e-16 on the CPU route, so the
      CPU keeps `h_T` and the device applies `h_FM`; 0 on every other
      key and route. End to end the same single flip, on the reference
      point under `"always"` at ("mixed", "complex128"). Worst count 1,
      so the 2x pin is KEPT; the comment is re-dated with these numbers.
    - `FM_RETRY_FLIP_COUNT = 0`: retry subsets [3, 4, 5] on both
      backends at both device precisions, 0 flips.
    - `FM_VRAM_P_BOUNDS = (5_150_000, 30_900_000)`: per-slot FM term at
      512x622, 10,304,064 B at complex128 and 15,456,064 B at
      complex64, equal at both device precisions.
      `FM_VRAM_R_BOUNDS = (16_400_000, 65_800_000)`: 32,883,712 B high
      water at every precision (13.31 MB held).
    - `GATED_FM_ITERATION_DIFF_COUNT = 0`, `GATED_FM_CONVERGED_FLIP_COUNT
      = 0`: 0 and 0 at all four precision keys; the FM-seeded `h` band
      1.063e-6 px (mixed) and 8.5e-14 / 4.8e-12 px (float64).
    - `GATED_FM_BATCH_INVARIANCE_TOL = 0.0`: 0.0 at B = 32, 40 and the
      default (64) against B = 8, both device precisions.
    - Device peak (fixer note): `fourier_mellin_peak` under cupy is
      BITWISE the numpy call on the planted rows (asserted in the gated
      contract class, green).

138. **Integration close: the synthetic measurement debt of plan 12
    item 3 (not the real-data V10(m)).** Machine A (i7-13700H,
    Windows 11 build 26200; worktree .venv, CPython 3.13.12, numpy
    2.4.6; GPU NVIDIA RTX 2000 Ada Generation Laptop 8 GB, driver
    595.71, CuPy 14.2.0 on the pinned overlay of D21.15,
    `KIKUCHIPY_EXPECT_GPU=1`; `nvidia-smi` 332 MiB used, 0 % before
    every GPU run), scratch `close/`:
    (i) **The numpy session against the CPU route (D22.14)**, FK-MIXED
    under `"auto"`: NOT bitwise; seed codes identical, `theta_hat`
    within 1.78e-15 deg, converged homographies within 8.53e-14 px,
    iterations equal (pinned at the measured bands, entry 137).
    (ii) **`theta_hat` parity and acceptance flips on the device (FQ14)**,
    six seam sets x two device precisions x two routes: complex128
    worst 5.77e-15 deg, complex64 worst 1.577e-7 deg (the promoted
    complex64 spectra), routes 1 and 2 identical; applied-row
    difference at most 4.4e-16 (complex128) and 1.33e-8 (complex64);
    acceptance flips 1 in 48 keyed runs (entry 137), retry-subset flips
    0. End to end on the capture map: angle-prop difference 2.2e-15 /
    2.3e-8 deg, seed-code flips 1 / 0 / 0 / 0, iteration and converged
    flips 0.
    (iii) **The FM VRAM terms, calibrated separately (D22.18)**, at
    512x622 (460x560 crop): per routed slot measured 10.30 MB
    (complex128) and 15.46 MB (complex64; the promotion copies),
    against the model's 38.22 / 33.12 MB under `"mixed"` and 40.76 /
    35.67 MB under `"float64"` (2.1x to 4.0x over, an upper bound);
    per reference the build high water 32.88 MB (the 4.03 MB look-up
    table plus 112 B per crop pixel; 28.85 MB with the table cached)
    against the model's 47.01 MB (`"mixed"`) and 49.56 MB (`"float64"`;
    1.43x / 1.51x). Whole model against an `"always"` run's high water:
    293.7 / 555.1 MB (`"mixed"`, B = 8 / 16) and 296.8 / 558.2 MB
    (`"float64"`) against 726.5 / 1297.7 and 777.5 / 1379.3 MB. Default
    B on this card (7.7 GB free): 64 under `"off"` and `"auto"` at
    480x480 and 512x622, all four precision keys (and 64 / [64, 64]
    sessions in the gated chooser arm); the FM check first changes B
    only below about 0.5 GB free (4 -> 1 at 512x622).
    (iv) **B and routing invariance with FM on**: max homography
    difference 0.0 between B = 8, 32, 40 and 64 under `"auto"` on the
    ramp map, both device precisions; the 32-slot sub-batch with one
    slot routed (positions 0, 13, 31) against all 32 routed: row, angle
    and decision BITWISE at both routes, device and seed precisions
    (`test_routing_invariance`, green); unrouted ramp points under
    `"auto"` bitwise `"off"` (3 points).
    (v) **CPU host bytes per routed reference (D22.10)**: held numpy
    state per reference 16.00 MB at 480x480 (spectrum 2.85, box 3.56,
    subregion resident 9.59 MB) and 22.09 MB at 512x622 (3.93, 4.91,
    13.24 MB); look-up table 3.63 / 3.85 MB once. The information
    message's note says "up to 18.5 MB ... plus 4.0 MB" and "up to 25.5
    MB ... plus 4.3 MB" (the pattern-size bound, 1.15x / 1.16x over),
    an honest upper bound.
    (vi) **Costs (D22.15), synthetic** -- CPU, one point, best of 5,
    default threads, at 480x480 (G1, 3 deg) / 512x622 (G3, 8 deg): the
    whole P = 1 seam call of a routed point 92.7 / 138.6 ms (forced
    66.7 / 102.9, no acceptance), of which the translation row 28.6 /
    50.2, the angle alone 10.4 / 14.3 and one criterion 12.3 / 17.1 ms;
    `fit_pattern` at 0 iterations 35.8 / 62.2 ms; one IC-GN iteration
    11.5 / 15.9 ms, so a routed point adds about 8.1 / 8.7 iterations
    (the spec's 7 to 11; absolute times below the spec-gate estimate on
    this machine). GPU, P = 32, best of 3 after a warm-up: the seam
    46.5 / 75.6 ms unrouted against 151.3 / 224.0 ms all routed, i.e.
    3.28 / 4.64 ms per routed slot (the spec's inference was 2.2 to
    2.5 ms at 480x480). Ramp map (21 points, 480x480) best of 3:
    `"off"` 0.449 s, `"auto"` 0.964 s, `"always"` 0.965 s under
    `"mixed"`; 1.607, 2.422, 2.410 s under `"float64"`.
    (vii) **Per-class fits and the default-suite wall time**:
    `uv run pytest tests/test_indexing/test_hrebsd_fourier_mellin.py
    -n 0 -q -p no:cacheprovider`, 140 passed, 118 skipped, 124.7 s in
    pytest, 130 s wall (the interrupted run's 210.9 s was taken under
    load). Per class (tests run, seconds; CPU `fit_pattern` calls,
    numpy-session lockstep slots; counted by a scratch plugin wrapping
    `_engine.fit_pattern` and `_batched.run_lockstep`, cached runs
    counted once, in the class that first builds them): Switch 25, 6.7
    s, 59 fits, 22 slots; Angle 13, 3.3 s, 0; EdgeTreatment 10, 4.5 s,
    0; SeedRows 18, 9.8 s, 5 fits; Capture 5, 5.8 s, 20 fits;
    RampRescue 7, 15.9 s, 32 fits; Acceptance 10, 24.7 s, 36 fits, 6
    slots; Gate 22, 1.4 s, 0; Retry 8, 4.7 s, 10 fits; CpuRoute 6, 8.0
    s, 47 fits; NumpySession 16, 38.9 s, 0 fits, 74 slots. In all 209
    CPU fits and 102 lockstep slots, against V10's estimate of about
    100 fits (the excess is mostly the switch and CPU-route classes'
    small-map runs); the slowest class is NumpySession at 38.9 s; no
    class was moved to the weekly marker at this close.

139. **Integration close: the final runs.** Same machine and
    environment as entry 138.
    - Default FM suite, `-n 0`: 140 passed, 118 skipped (106 gated, 12
      weekly), 0 failed (entry 138 (vii)).
    - Gated FM suite, the pinned overlay, `KIKUCHIPY_EXPECT_GPU=1`, `-n
      0`: 246 passed, 12 skipped (the weekly arms), 235.8 s; then
      `--weekly`: 258 passed, 0 skipped, 383.9 s.
    - Stage E: `tests/test_indexing/test_hrebsd_gpu.py` default at `-n
      2`: 209 passed, 278 skipped, 113.7 s; gated through the overlay at
      `-n 0 --weekly`: 487 passed, 0 skipped, 577.4 s.
    - `uv run pytest tests -k hrebsd -n 2 -q`: before the import fix 1
      failed (`TestImportAudit::test_no_new_required_dependency`, entry
      136), 1044 passed, 405 skipped; after it 1045 passed, 405
      skipped, 0 failed, 351.6 s.
    - After the import fix every suite above was re-run: default FM
      suite 140 passed, 118 skipped (183.4 s in pytest, the machine
      busier; the 124.7 s of entry 138 (vii) is the quiet figure);
      gated FM 246 passed, 12 skipped; `--weekly` 258 passed, 0
      skipped; Stage E gated `--weekly` 487 passed, 0 skipped (722.4 s,
      a stray CPU load during part of it); the Stage E default suite
      inside the `-k hrebsd` run above. Every recorded
      measurement of entries 137 and 138 reproduced bitwise (the
      timings within noise: ramp `"off"` / `"auto"` / `"always"` 0.567
      / 1.248 / 1.236 s mixed, 1.691 / 2.602 / 2.590 s float64; 3.05 /
      4.45 ms per routed slot).
    - `ruff check` and `ruff format --check` on `_engine.py`,
      `_batched.py`, `_gpu.py`, `_fourier_mellin.py`, `ebsd.py` and the
      test module: clean.
    - `"off"` bitwise unchanged on both backends: the `fourier_mellin=
      "off"` == no-keyword pins (CPU, numpy session, device), the V9
      `extras == {}` asserts, the pre-Stage-D literal pins and the
      whole Stage E gated suite are green unmodified.
    No red test remains. Coverage, the adversarial review, bug
    injection on FM1 to FM52, the real-data V10(m) record and the
    D22.16 docs and tutorial edits are the next gates (plan 12 items
    3 to 8).

#### V10 recorded results, review gate (2026-10-08)

140. **Bug injection, FM1 to FM26 (plan 12 item 4; injector, run
    ALONE).** Machine A, 2026-10-08, worktree at 21351744 (clean),
    with the host loaded by the concurrent performance-record workflow
    (GPU at 99 % utilisation, 2.1 of 8 GB in use, before the gated
    runs). Every file that could be touched (`_fourier_mellin.py`,
    `_engine.py`, `_batched.py`, `_gpu.py`, the test module) was
    backed up to the scratchpad with its md5 before the run. Each
    mutant is the smallest edit realising its plan definition,
    applied by a harness that asserts each anchor matches exactly
    once, runs ONLY the designed killers of the module's mutation map,
    then restores every file from the backup and verifies its md5
    before the next mutant (no git operation). Default killers: `uv
    run pytest <nodes> -n 0`; gated twins: the D21.15 overlay with
    `KIKUCHIPY_EXPECT_GPU=1 -n 0 --weekly`. 36 variants (26 ids; FM15,
    FM16, FM17, FM22, FM23, FM25 and FM26 in more than one
    realisation), 7 gated twin runs, 3 re-checks.
    (i) **Killed by a designed default killer (35 of 36 variants, 25
    of 26 ids):** FM1 (`R(-theta)`: the bicubic crop 138.3 > 1e-12,
    the PC-shift rows 169.9 px > `FM_SEED_TOL_PX` 0.45, all four
    capture arms); FM2 (`T(t) R`: the PC-shift rows 2.65 px > 0.45,
    its only killer, as designed); FM3 (bin-unit table: the table pin
    at 460x560, 39.8 bins, and G3 2.29 deg > 0.06); FM4 (`[0, 2 pi)`:
    the table pin at both shapes and G1); FM5 (NaN on a failed
    routed estimate: all 8 planted-failure arms); FM6 (in-place
    stencil: the new-array arm and the three spectra-bitwise arms;
    `test_unrouted_rows_are_bitwise_the_stage_e_rows` passes because
    it does not read the spectra); FM7 (no edge treatment: the G4
    lock, 19.13 deg > 0.5, both no-filter arms); FM8 (no window: `[40,
    -40] != [5, -5]`); FM9 (offset sign: G1 0.420 deg >
    `FM_ANGLE_TOL_DEG` 0.25, reproducing the 1.7x separation in the
    pin comment); FM10 (inverted: G5, the planted wrong angle, the
    capture codes 2 != 1); FM11 (wins forced True: wrong-angle and
    tie arms); FM12 (acceptance by the normalised phase-correlation
    peak of the reference against the target crop and against the
    de-rotated crop, with the criteria still called: the G5 arm
    refuses the FM row and the G6 arm keeps it, both as designed);
    FM13 (`h_T` finiteness dropped from `valid`: the route-2 arm only,
    because route 1's comparison with NaN already refuses); FM14
    (`<=`: the tie arm); FM15a (seam level: route-0 slots of a routed
    sub-batch evaluated as route 1, with the `_batched` mask dropped;
    killed by `EdgeTreatment::test_unrouted_rows_are_bitwise_the_
    stage_e_rows` slot 0 only, while the designed NumpySession and
    CpuRoute arms PASS, see (iii)); FM15b (CPU engine, every fitted
    point through the seam as route 1: CpuRoute `test_unrouted_and_
    translation_won_points_equal_off`, the unrouted angle 0.0 is not
    NaN); FM15c (diagnostic, route-0 slots treated as forced:
    NumpySession and EdgeTreatment); FM16a (`"off"` through the FM
    path with all-zero routes: all three arms,
    `build_fourier_mellin_state ran on a fourier_mellin='off' run`,
    `extras` not empty); FM16b (the device runner writes an all-zero
    route into `extras` on `"off"`): the numpy-session arm only,
    `{'fourier_mellin_route': [0, ...]} != {}`; FM16c (the device
    session builds the FM state on `"off"`): the numpy-session arm;
    FM17a (`M^T R_s M`: 0.561 deg > 1e-12) and FM17b (no y flip:
    every sign inverted). BOTH are killed on the untilted detector
    too, so the plan's "on an untilted one `M M = I` makes it
    equivalent" does not hold for the G7 default detector (its sample
    tilt makes `M` non-symmetric); the tilted arm remains the
    designed killer. FM18 (no symmetry: 75.7 deg > 1e-12, both
    detectors); FM19 (total angle: 3.0 deg on the out-of-plane arms,
    both detectors); FM20 (signed: `-1.5` and `-2.0` not routed); FM21
    (`>`: the exact boundary not routed); FM22a (routes fail closed
    on NaN: the engine arm on both detectors; the function arm
    passes, as it must) and FM22b (unusable twist 0, not NaN: all four
    arms); FM23a (function level, every point against the first
    reference: the two-grain arm, signs `[1, 1] != [1, -1]`); FM24
    (non-converged retry stored: `('homography', 4)` not bitwise
    `"off"`); FM25a (converged points retried: subset `[0..5] != [3,
    4, 5]`), FM25b (masked point 6 retried) and FM25c (the retry pass
    run twice: 3 runner calls != 2), each killed by two or three of
    the three arms; FM26a (seam level, route-2 slots through the
    acceptance: criteria recorded for route 2, all three arms) and
    FM26b (engine level, the retry routed with code 1: `2 in {1}`
    fails, all three arms).
    (ii) **Gated twins.** Killed: FM6 (`test_seam_output_contract`,
    the spectra not bitwise, all four precision pairs), FM8
    (`test_the_peak_rule_on_the_device`, `[40, -40, ...]`), FM15a and
    FM15c (`test_unrouted_points_keep_the_off_bits` at both
    precisions, where `untouched.sum() >= 2` fails at 0 or 1, and
    `test_seam_output_contract` all four), FM16a and FM16c
    (`test_off_is_the_stage_e_device_path`, the refusing state
    builder raises, both precisions). FM16b's gated twin SURVIVES (see
    (iv)).
    (iii) **A designed killer blind to its mutant (the mutant is still
    killed):** FM15a passes `NumpySession::test_unrouted_slots_and_
    sub_batches_return_the_stage_e_rows` and `CpuRoute::test_
    unrouted_and_translation_won_points_equal_off`. Their unrouted
    points (0.8 deg and the references) LOSE the acceptance, so
    evaluating them as route 1 leaves `h_T` bitwise. FM15c (the same
    points forced, no acceptance) is killed by the NumpySession arm,
    which confirms that the acceptance, not a missing route-0 slot,
    is what hides FM15a. The EdgeTreatment arm (0.37 deg) and the
    gated determinism arm kill FM15a, so the mutation map should name
    them as FM15's acceptance-level killers.
    (iv) **Survivors.** **FM23b (engine level: every fitted point
    gated against the first fitted point's reference, the function
    unchanged) SURVIVED** its designed killer
    (`Gate::test_each_point_is_measured_against_its_grain_reference`
    calls the function with explicit references, so it cannot see the
    engine's `point_reference`) and the re-check (NumpySession
    `test_route_flags_arrive_in_fit_order_with_zero_padding`,
    `test_the_fm_state_is_built_lazily_once_per_routed_grain` and
    `test_unrouted_slots_and_sub_batches_return_the_stage_e_rows`;
    Gate `test_masked_points_are_never_routed` and
    `test_the_engine_fails_open_on_unusable_points`: 6 passed).
    Reason: the only engine-level two-grain map, FK-TWO-GRAIN, puts
    grain B's reference 0.8 deg from grain A's, below the 1.5 deg
    gate, so gating grain B against grain A's reference reproduces
    the same routes `{0: 0, 2: 1, 4: 1, 1: 0, 3: 0}`; the G7 maps go
    through the engine with one reference only. The mutant is not
    equivalent: a scratch probe (the `fk_gate_probe` recipe on
    `g7_two_grain_map` with its `grain_labels` and `references`,
    stopping at the first `_run_chunks`) gives routes `[0, 1, 0, 1]`
    unmutated and `[0, 1, 1, 1]` under FM23b on both detectors (grain
    B's reference is routed against grain A's). Designed killer to
    add: that probe as a Gate arm, with `fk_gate_probe` taking
    `grain_labels` and `reference`.
    **FM16b, gated twin only** (killed in the default suite):
    `test_off_is_the_stage_e_device_path`, re-checked with
    `test_drift_tripwire_unchanged_under_off`,
    `test_unrouted_points_keep_the_off_bits` and
    `test_runner_contract` (6 passed). Reviewed-equivalent on the
    device's OBSERVABLES: an all-zero route makes the seam skip on
    the host before any device work (rows bitwise Stage E, no
    outputs, no forced slot, 12-wide rows), so the only breach is the
    D22.6 contract that `extras == {}` on `"off"`. That contract is
    shared code (`_gpu._fit_batch` runs unchanged under the numpy
    session) and is pinned by
    `Switch::test_off_builds_no_state_and_writes_nothing_numpy_session`.
    (v) **Close.** After the last mutant every touched file's md5
    equals its pre-run md5 (`md5sum -c`, 5 of 5 OK, `git status`
    clean). The default FM suite at `-n 2`: 140 passed, 118 skipped,
    0 failed (163.4 s). The harness, mutant definitions, per-mutant
    logs and `results.jsonl` are in the session scratchpad under
    `injF\`.

141. **Bug injection, FM27 to FM52 (plan 12 item 4; injector, run
    ALONE, second half).** Machine A, 2026-10-08, worktree at
    21351744; the only uncommitted change in it was ledger entry 140
    in this file. The host was loaded by the concurrent
    performance-record workflow, then in its CPU phase (the GPU was
    idle, 0 %, 332 MiB of 8 GB, when the gated runs started). The FM1
    to FM26 harness was reused (`injF\harness2.py`, `mutants2.py`)
    with FRESH backups and md5s of the current tree, with `ebsd.py`
    added: `_fourier_mellin.py`, `_engine.py`, `_batched.py`,
    `_gpu.py`, `ebsd.py` and the test module (the md5 of
    `validation.md` recorded too). Before any run, every anchor was
    checked to match exactly once (40 of 40). After each mutant every
    file was restored from the backup and its md5 verified (no git
    operation). Default killers: `uv run pytest <nodes> -n 0`; gated
    twins: the D21.15 overlay with `KIKUCHIPY_EXPECT_GPU=1 -n 0
    --weekly`. 40 variants of 25 ids (FM39 cannot be injected, see
    (iv)), 16 gated runs, 9 re-checks.
    (i) **Killed by a designed default killer (32 of 40 variants):**
    - FM27 (`num_iterations` summed): `203 == 3`, the direct-fit
      oracle.
    - FM28a (routes computed in map order): `(1, 2)`, `0 == 1`.
      FM28b (a device batch's padded slots routed): `(1, -1)`, `1 ==
      0`.
    - FM29 (the branch compacted to the routed slots): the 1-of-P row
      is not bitwise the P-of-P row UNDER NUMPY. The default arm is
      therefore a real killer, not reviewed-equivalent.
    - FM30 (the CPU seam at P = chunk length): killed by
      `test_chunksize_invariance_and_repeatability` (homography not
      bitwise). The P = 1 spy passes, because on this host FK-MIXED's
      default CPU chunks hold one pattern each (`estimate_chunksize` =
      ceil(5 / (4 workers)) = 1).
    - FM31 (the subregion resident scattered into the box): the
      dead-band crop is off by 17.4 > 1e-12.
    - FM32 (theta in degrees read as radians): the frozen-formula and
      own-partial-row arms at atol 1e-12, and all four capture arms.
    - FM33a (props written on `"off"`): the `"off"` prop-set arm and
      the public `"always"` arm. FM33b (codes 1 and 2 swapped): `2 ==
      1` in RampRescue and in both Retry arms.
    - FM34 (the D22.11 raise missing): all four backend arms and the
      signal arm; the xmap message is raised instead.
    - FM35 (`"auto"` without `xmap` routes every point): "resolve ran
      before the Fourier-Mellin checks".
    - FM36a (warning missing): both no-twist arms. FM36b (float
      equality): the constant arm only, as designed.
    - FM37 (complex64 kept): the stencil dtype spy, `complex64 ==
      complex128`.
    - FM38 (the translation taken from the reused target spectrum):
      PC-shift rows 112.7 px > `FM_SEED_TOL_PX` 0.45, and all four
      capture arms.
    - FM40a (the CPU engine fits an unapplied forced point): two fits,
      `2 == 1`, in both Retry arms. FM40b (the device runner keeps
      that slot active): the NumpySession inactive arm.
    - FM41 (forwarding dropped): `None == 'off'`.
    - FM42 (the prop loop not extended): `fourier_mellin_seed` is
      missing from the public result.
    - FM43 (de-rotation about the detector centre): PC-shift rows
      14.1 px > 0.45, bicubic crop off by 82.5 > 1e-12.
    - FM44 (every CPU point sent through the seam): seam counts `{0:
      1, 1: 1, 2: 1, 3: 2, ...}` against `{2: 1, 3: 1, 4: 2}`. This is
      its only killer, as designed.
    - FM45a (outputs written into `extras`): extra keys in `extras`.
      FM45b (the seam pops the route key): `set() == {route}`.
    - FM46d (an extra realisation, B always halved once on FM runs):
      `32 == 64` on the headroom arm.
    - FM47 (no retry under `"always"`): one pass, `1 == 2`.
    - FM48a (the retry B re-chosen): `None == 4`. FM48b (the retry at
      the initial B, the halving ignored): `8 == 4`.
    - FM49 (NaN-`h` points excluded from the retry): `5 in [3, 4]`.
    - FM50a (the device FM state built for every grain) and FM50b
      (rebuilt per batch): two builds against one, both.
    - FM51 (a NaN `h_T` criterion counted as a loss): the FM row is
      kept, not bitwise `h_T`.
    - FM52c (both angle masks dropped): route-0 angle 3.01 deg is not
      NaN, in both default arms.
    (ii) **Gated twins.** Killed:
    - FM45a and FM45b: `test_seam_output_contract`, all four
      precision pairs.
    - FM52c: the same test, route-0 angles 2.71 deg.
    - FM46d: `test_default_batch_size_is_the_off_one`, `32 == 64`.
    - FM48a: `test_retry_routing_parity`, both device precisions,
      `None == 64`.
    Survived on the device, each one killed in the default suite:
    - **FM29**: `test_routing_invariance`, 8 passed. On this card with
      cuFFT 11.4.1.4 the compacted branch is bitwise the full-P one,
      so the docstring's premise that "cuFFT's batched plans [are]
      batch-count sensitive" does not hold here. The numpy arm is
      FM29's effective killer.
    - **FM37**: `test_theta_hat_and_acceptance_parity` and
      `test_frozen_functions_and_state_on_the_device`, 52 passed. The
      complex64 parity band absorbs a float32 profile, and the device
      contract test does not check the stencil's dtype. The numpy
      dtype spy is FM37's effective killer.
    - **FM48b**: no real out-of-memory halving happens on the card, so
      the initial B and the final B agree. The default arm plants the
      halving.
    (iii) **Survivors of the default suite (8 variants).** Each was
    re-checked with its next-most-relevant classes.
    - **FM28c** (the tail padding routed when B is not a multiple of
      P). Re-checks: the designed killer plus the whole NumpySession
      and CpuRoute classes, 22 passed; gated
      `test_batch_size_invariance` at B = 40 and
      `test_unrouted_points_keep_the_off_bits`, 4 passed. SURVIVED;
      the results are unchanged. A routed padded slot only adds
      masked work: its angle and applied flag are cut at `keep`, and
      a route-0 real slot's row is still `h_T` bitwise. The only
      breach is the D22.11 contract "0 on padded slots" in the seam's
      `extras`. The designed killer cannot reach it: every
      default-suite B is a power of two of at most 64, so B is a
      multiple of P = min(32, B) and the tail branch never runs.
      Killer to add: the route-flags arm at `chunksize=40` (P = 32,
      an 8-slot tail).
    - **FM45c** (the seam writes the int8 copy of the route back into
      `extras`). Re-checks: NumpySession, Gate and Switch, 63 passed;
      gated contract, 4 passed. Reviewed-equivalent for every
      internal caller. Both runners hand the seam a C-contiguous int8
      route, so `np.ascontiguousarray(route, dtype=np.int8)` returns
      the SAME object and the assignment writes it back unchanged. It
      differs only for a direct caller passing another dtype or
      layout.
    - **FM46a** (the literal FM46: B picked from the FM model in the
      first loop). Re-checks: the same classes, 63 passed; gated, 1
      passed. Reviewed-equivalent, with proof. `_vram_model_bytes`
      is nondecreasing in B, and the FM model is at least the `"off"`
      model at every B, so no B above the `"off"` choice B0 fits with
      the FM terms. The ladder `(64, 32, ..., 1)` is consecutive
      halvings, so "halve from B0 while the FM model does not fit"
      and "take the largest ladder B whose FM model fits" pick the
      same B, and both fall back to 1. No test can kill it. FM46d (an
      always-halve variant) is the realisation the designed killers
      do see.
    - **FM46b** (the engine asks the FM model on every FM run, routed
      or not) and **FM46c** (the runner asks it whenever a route key
      is present). Re-checks: the same classes, 63 passed each; gated,
      1 passed each. SURVIVED; this is a test gap. The designed arms
      run `"off"` and `"auto"` with routed points, never an FM run
      whose routes are all zero. FM46b is reachable through
      `run_hrebsd_dic`: `"auto"` with every twist below the gate and
      `chunksize=None`. On a card without headroom B is then halved,
      which, per the Determinism docstring, can change the unrouted
      points' bits below B = 32. FM46c is reachable only by a direct
      `_run_chunks_gpu(..., chunksize=None)` call with an all-zero
      route, because the engine always passes an explicit B on FM
      runs. Killer to add: an `"auto"` arm with no routed point (the
      G7 no-twist map, or FK-MIXED with every twist below the gate)
      in `test_the_runner_asks_the_fm_model_only_on_routed_runs`,
      asserting that every chooser call has `fourier_mellin=False`.
    - **FM50c** (an extra CPU realisation: the CPU FM state built
      eagerly for every reference on an FM pass). Re-checks: the
      designed killer and the CpuRoute, Switch and Retry classes, 39
      passed. SURVIVED; the results are unchanged. A grain without a
      routed point is never passed to the seam, and its retry
      eligibility is decided by the same `has_state`. Only the cost
      (one state build per unrouted reference) and the D22.10 "once
      per reference in *positions*" wording differ. No CPU
      build-count spy exists; the designed FM50 killer is the
      device's lazy-build spy.
    - **FM52a** (the angle mask dropped in `fourier_mellin_rows` only)
      and **FM52b** (dropped in the `_batched` seam only). Re-checks:
      the designed killers and the SeedRows, NumpySession and
      Acceptance classes, 44 passed each; gated contract, 4 passed
      each. The two masks are redundant, so either one alone keeps
      the seam outputs right. FM52b is reviewed-equivalent:
      `fourier_mellin_rows` already returns NaN on route-0 slots.
      FM52a SURVIVED as a gap at the function level. The docstring of
      the frozen `fourier_mellin_rows` promises NaN on route-0 slots,
      but its two direct callers in the tests
      (`test_a_nan_translation_row_is_never_rescued` and the device
      contract) route every slot. Killer to add: a direct
      `fourier_mellin_rows` call with a mixed route, asserting NaN
      angles on its route-0 slots.
    (iv) **FM39** (the gate run from the target's own PC frame)
    cannot be injected; it is reviewed-equivalent, as the mutation
    map states. No PC enters `twist_about_detector_normal`: it uses
    `sample_to_detector_matrix(detector)` = `DETECTOR_Y_FLIP @
    detector.sample_to_detector`, which depends on the tilts only. A
    scratch probe confirms it: with the same tilts, the matrix of a
    single-PC detector and that of a (2, 3) per-point-PC detector are
    bitwise equal (`True`). FM43's second realisation (rotation about
    the target's own PC) cannot be expressed either, because no
    per-target PC reaches the seam: `SeedBatch` carries only the
    targets, coefficients, `pattern_index`, `extras` and `outputs`.
    (v) **Counts.** Default suite: 32 of 40 variants killed. Every
    one of the 25 injectable ids has at least one realisation killed,
    except the literal FM46 (FM46a, proven equivalent). The 8
    survivors:
    - reviewed-equivalent: FM45c, FM46a and FM52b;
    - results-equivalent, cost only: FM50c;
    - test gaps, each with a killer named in (iii): FM28c (its results
      are unchanged), FM46b, FM46c and FM52a.
    Gated twins: 5 killed and 9 survived (FM29, FM37, FM45c, FM46a
    to FM46c, FM48b, FM52a and FM52b). Each survivor is killed in the
    default suite or dispositioned in (iii).
    (vi) **Close.** After the last mutant every touched file's md5
    equals its pre-run md5 (`md5sum -c`, 6 of 6 OK), and
    `validation.md` was unchanged until this entry. The default FM
    suite at `-n 2`: 140 passed, 118 skipped, 0 failed (285.4 s, on
    the loaded host). The harness, mutants, logs (`logs2\`) and
    `results2.jsonl` are in the session scratchpad under `injF\`.

142. **Implementation-review fixes (plan 12 item 5; fixer, run
    2026-10-08 on 21351744 plus the working tree; loaded host, the
    concurrent performance-record workflow running).**
    (i) **Findings.** All nine review findings held and were applied
    (plan 12.7 has the table): RF-F1 the retry B passes per call
    (`_run_chunks_gpu(..., run_state=None)`; `_RUN_STATE`,
    `_last_batch_size` and `_reset_last_batch_size` deleted); RF-F2
    the FM36 arm plants a 1e-11 deg twist (`g7_subthreshold_map`,
    measured 9.95e-12 deg; the constant map's twist is -1.77e-31 deg
    on this host); RF-F3 the gate matches flat indices through
    `_segmentation._map_grid` (`navigation_shape` keyword, the engine
    passes it; a sliced map's missing point gives NaN and route 1, a
    one-point map is point 0); RC-F-CONV-1 the refusal contract on the
    CPU, numpy-session and device routes (bitwise "off", seed 0 on
    fitted points, angle NaN, no retry on the CPU); line 635 reached
    through a planted NaN profile, because `ReferenceState` refuses a
    zero-contrast reference first (`ValueError`, measured);
    RC-F-CONV-2 `_info_lines(..., fourier_mellin=)` and the engine
    passes `fm_routed`; RC-F-CONV-3 the retry line now reads
    `"  Fourier-Mellin retry: {applied} of {subset} pattern(s)
    re-fitted from the rotation seed"`, printed after the pass (FK-MIXED
    2 of 2; with a planted forced-angle failure 1 of 2), and the D22.18
    line pinned (480 x 480: 18.5 MB and 4.0 MB; 512 x 622: 25.5 MB and
    4.3 MB); RC-F-CONV-4 the no-point-group arm (2.0 deg recovered,
    the symmetry copy reads the raw 77.68 deg, one warning);
    RC-F-CONV-5 the route-shape error, the `chunksize=None` chooser
    keyword (`[False, True, False]`) and the tail padding (B = 3 with
    `SUB_BATCH_SIZE` patched to 2 on FK-TWO-GRAIN, props bitwise the
    B = 2 run); RC-F-CONV-6 the pragma'd `try/except` removed.
    (ii) **Survivors re-injected** (fresh backups of the fixed tree,
    md5 restore, never git; `fixF\harness3.py`, `mutants3.py`,
    `results3.jsonl`, `logs3\`): FM23b, FM28c, FM46b, FM46c, FM50c and
    FM52a each KILLED by its new killer; FM36b KILLED on the constant
    and the planted sub-threshold arms. FM45c, FM46a and FM52b stay
    reviewed-equivalent; the gated-only survivors FM16b, FM29, FM37 and
    FM48b stay killed in the default suite. Regression mutants of the
    fixes, all KILLED: RF1 (retry B from module state: `[8, 16] !=
    [8, 4]`), RF3 (flat indices as xmap points), CONV1 (CPU `has_state`
    filter dropped), CONV2 and CONV2b (printed model not FM), CONV3
    (subset counted; first SURVIVED, killed after the planted-failure
    arm was added), CONV3b (LUT bytes), CONV4 (no-point-group phase
    skipped), CONV5a (route-shape check weakened). After every mutant
    all five touched files matched their md5 (`md5sum -c`, 5 of 5 OK).
    (iii) **Pins.** No pin moved: the gate change leaves full maps'
    indices unchanged (every Gate pin passed unchanged), the retry B
    and the `"off"` info block are bitwise; no literal re-measured
    except the new ones in (i). `"off"` stays bitwise on both backends
    (the Stage E gated suite and the off arms pass unchanged).
    (iv) **Final runs.** Default FM suite (`-n 2`): 159 passed, 120
    skipped, 0 failed (227.7 s). Gated FM suite `--weekly` (overlay,
    `KIKUCHIPY_EXPECT_GPU=1`, `-n 0`, five calls by class because of
    the time limit): 71 + 37 + 68 + 60 + 43 = 279 passed, 0 failed.
    Stage E gated `--weekly`: 487 passed, 0 failed (1061.7 s).
    `uv run pytest tests -k hrebsd -n 2 -q`: 1064 passed, 407 skipped,
    0 failed (575.6 s). Coverage, default + gated FM + Stage E gated +
    the `-k hrebsd` CPU suite combined: `_fourier_mellin` 100 %,
    `_engine` 100 %, `_gpu` 100 %, `_batched` 100 %, `_cuda` 100 %
    (every `_hrebsd` module 100 %). Ruff check and format clean.
    (v) **Counts.** 19 default tests and 2 gated tests added (default
    140 to 159, gated FM 258 to 279).

#### V10 recorded results, performance record (2026-10-08)

The frame oracle of V10(m)(1) and the GPU records of V10(m)(2) to (4)
with the converted map (plan 12.8); the CPU whole-map `"auto"`
records follow.

143. **V10(m)(1) frame oracle (requirements D22.7; plan open questions
    FQ1, FQ2): the as-read FAIL, its root cause, and the
    converted-frame PASS signed off by Johan (2026-10-08).** Snapshot
    `snapF_21351744` (commit 21351744), CPU-only analysis, no HREBSD
    re-run. Fits: the GPU whole map under `"always"` at `(None, None)`
    (57772 fitted points, 57746 converged; ledger record of the earlier
    run, `perfF/arrays/map_gpu_always_none.npz`). Selection exactly as
    pre-registered: converged, |fitted twist| >= 1 deg (polar
    decomposition of Fe), outside the crater (their CCC >= 0.35): 445
    points, fitted twist -2.25 to +1.73 deg. Homography read-out
    against the polar twist on those points: correlation 0.999996, max
    |diff| 0.040 deg, median 0.0018 deg.
    (i) **As read: FAIL by the D22.7 rule.** The tutorial's load recipe
    passes the raw h5oina `EBSD/Data/Euler` to `Rotation.from_euler`.
    On that map, xmap twist against fitted twist, 445 points (the
    as-read frame and the four alternative frames of the rule; the
    Deming slopes use the far-field Hough variance of rows 0 to 40 and
    of far256 as the error variance of the xmap twist):

    | frame | corr | OLS slope | intercept (deg) | Deming rows0_40 / far256 | median abs diff (deg) | sign agree | n abs twist >= 1.5 (all 57772) |
    |---|---|---|---|---|---|---|---|
    | as read | 0.604 | 0.522 | +0.061 | 0.576 / 0.529 | 0.582 | 320 | 336 |
    | sample z +90 | -0.557 | -0.267 | -0.099 | -0.286 / -0.270 | 1.660 | 105 | 256 |
    | sample z 180 | 0.411 | 0.196 | -0.165 | 0.214 / 0.198 | 1.121 | 260 | 103 |
    | sample z -90 | **0.993** | **0.985** | -0.004 | 0.987 / 0.985 | 0.051 | 445 | 57 |
    | no y flip | -0.604 | -0.522 | -0.061 | -0.575 / -0.529 | 1.755 | 125 | 336 |

    Best frame "sample z -90", not "as read", and the as-read slope
    0.522 lies outside [0.8, 1.25]: FAIL on both criteria. The stage
    stopped for Johan, as the rule requires. ("sample z a" means
    R_s' = Rz(a)^T R_s Rz(a) applied to the sample-frame
    misorientation, `perfF/frame.py`; the gate itself,
    `twist_about_detector_normal`, matched the as-read column to 0.0
    deg.)
    (ii) **Root cause: the vendor's sample-frame convention, not code.**
    The h5oina Euler angles are given in Oxford's CS1 sample frame,
    which is kikuchipy's (EDAX/TSL) sample frame turned 90 deg about
    the surface normal: Oxford X1 = +Y_kp (along the tilt axis,
    parallel to detector X_g), Y1 = -X_kp, Z1 = Z_kp; vectors
    v_ox = q v_kp with q = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]].
    Orientations: `R_kp = Rotation.from_euler(euler_ox) *
    Rotation.from_axes_angles([0, 0, 1], -90, degrees=True)`, i.e.
    phi1_kp = phi1_ox + 90 deg with Phi and phi2 unchanged (EMsoft's
    'hkl' to TSL rule; checked against the matrix form to 2.7e-14
    deg). Naming hazard: the tutorial's "+90 deg rotation of the axes"
    (ledger 80) and the oracle's "sample z -90" are the SAME matrix q;
    records quote the matrix or the axis statement, never a bare signed
    angle. It entered through the tutorial's CrystalMap recipe (cell 8)
    and V10(m) inherited it; neither kikuchipy nor orix converts these
    angles (the Oxford reader returns an identity placeholder map). A
    sample-side rotation leaves every crystal-frame misorientation angle
    unchanged, so segmentation, KAM and misorientation angles cannot see
    it; the gate's twist about the detector normal (n_s = (0.375, 0,
    0.927)) is the first consumer sensitive to it. Not at fault: the
    gate code, the detector/sample geometry (`DETECTOR_Y_FLIP`
    confirmed: no-y-flip correlation -0.604), the PC convention.
    Ledger 127 (iv) had flagged exactly this risk. Five evidence lines
    (`frameinv/verdict.md`):
    - **Literature and other software** (strong on axis and magnitude,
      moderate-strong on sign): Zhu et al. 2019 (Oxford and Bruker share
      a frame, EDAX's is turned 90 deg about Zs); Britton et al. 2016
      (Bruker X_s parallel to the tilt axis and X_d); EMsoft 'hkl' adds
      90 deg to phi1 to get TSL; PyEBSDIndex composes Rz(-90) for EDAX,
      EMsoft and kikuchipy but not for OXFORD or BRUKER; kikuchipy's own
      reference-frames tutorial uses `R_sample["OXFORD"] = Rz(-90)`,
      validated by simulation to under 0.5 deg (maintainer reply on
      #746, issue #748). No first-party Oxford statement; the H5OINA
      spec is silent on the CS1 axes.
    - **Header metadata** (weak to moderate): `Specimen Orientation
      Euler` = 0 (CS0 = CS1, nothing further to apply); `Scanning
      Rotation Angle` = pi; a derivation from the file alone (beam frame,
      SRA as MTEX reads it, PC gradients) gives the same sign; without
      the SRA step it gives +90, which the oracle refutes (corr -0.557).
    - **Pattern simulation, absolute, no HREBSD or FM code**
      (strongest): dynamical (EMsoft Si master pattern), 18 points: -90
      wins 18 of 18 (12 far-field, 6 deformed); start NCC 0.68 to 0.69
      far-field and 0.50 to 0.64 deformed against <= 0.18 for every
      other frame (margin 0.50 to 0.62); start-to-best distance 0.17 to
      0.50 deg against >= 2.2 deg (about 42 deg for y-flip); also 18 of
      18 with the reader's detector angles. Kinematical (separate code),
      8 points: -90 nearest the optimum, median 0.50 deg against 2.2 to
      3.2 deg. autoECCI Si: 12 of 12 nearest -90 (supportive). Caveat:
      on the far field the separation is only 2.4 deg (Si [001] is 1.7
      deg from Z); the deformed points add 3.4 to 12 deg.
    - **Frame oracle** (confirming, not primary: a choice among frames
      made after the fact on the test that failed): -90 corr 0.993,
      slope 0.985, Deming 0.987, median diff 0.05 deg; +90 -0.557, 180
      0.411, no y flip -0.604.
    - **The tutorial's MapSweeper strain comparison** (ledgers 80 to 82;
      independent observable, Fe via M, no xmap): one rigid rotation
      fits, its matrix identical to q; all 6 slopes positive, all 4
      resolved components r > 0.9, whole-map rms difference 0.45 mm/m
      (itself chosen by a search: four candidates tie on |r|).
    Combined confidence 95% or better for this file. Open: files with
    SRA other than pi (both local files have SRA = pi), so a fixed
    Rz(-90) is not yet told apart from "+90 plus the scan rotation";
    the physical reading favours the fixed Rz(-90). Separate,
    second-order defect found on the way: the Oxford reader drops the
    header's detector azimuthal (+0.784 deg) and twist (+0.593 deg),
    keeping only the tilt (about 0.02 w on the gate twist; parked as a
    develop reader fix; the records below keep the reader-derived
    detector so they stay comparable).
    (iii) **Converted-frame re-evaluation: PASS.** The SAME 445 points
    and the SAME `"always"` fits (under `"always"` the gate is bypassed
    and Fe does not depend on the map's orientations, so no re-run),
    with only the CrystalMap orientations converted as in (ii)
    (`perfF2/frame_conv.py`). The library gate on the converted map
    reproduces the as-read "sample z -90" column to 2.0e-14 deg and the
    oracle's own converted as-read column to 0.0 deg.

    | frame (relative to the converted map) | corr | OLS slope | intercept (deg) | Deming rows0_40 / far256 / delta 1 | median abs diff (deg) | sign agree |
    |---|---|---|---|---|---|---|
    | as read (converted) | **0.993** | **0.985** | -0.004 | 0.987 / 0.985 / 0.992 | 0.051 | 445 of 445 |
    | sample z +90 (= the raw h5oina map) | 0.604 | 0.522 | +0.061 | 0.576 / 0.529 / 0.786 | 0.582 | 320 |
    | sample z 180 | -0.557 | -0.267 | -0.099 | -0.286 / -0.270 / -0.313 | 1.660 | 105 |
    | sample z -90 | 0.411 | 0.196 | -0.165 | 0.214 / 0.198 / 0.239 | 1.121 | 260 |
    | no y flip | -0.993 | -0.985 | +0.004 | -0.987 / -0.985 / -0.992 | 2.349 | 0 |

    Best frame: as read (converted); OLS slope 0.985 in [0.8, 1.25]
    (reverse regression, fitted on xmap, 1.000); far-field Hough SD of
    xmap minus fitted twist 0.012 deg (rows 0 to 40) and 0.010 deg
    (far256). PASS on both criteria of D22.7 once the input is in
    kikuchipy's sample frame. Over all 57746 converged points the
    correlation is 0.980 (as read 0.261).
    (iv) **Sign-off.** Johan accepted the diagnosis and the
    converted-frame PASS on 2026-10-08 (about 11:15), knowing the input
    was corrected after the result was seen; the defence is that -90
    was predicted beforehand by kikuchipy's own reference-frames
    documentation and #746, and confirmed by the independent dynamical
    simulation with no HREBSD or FM code (-90 winning 18 of 18 points).
    He asked for the `"auto"` records to be redone with the converted
    map, keeping the tutorial's reader-derived detector.
    (v) **Invalidation.** Every `"auto"` record of the earlier run
    (`perfF`: GPU `map_gpu_auto_none` and `map_gpu_auto_005`, their
    compare tables, routed / accepted / retried / converted / worsened
    counts, wall times and the theta_eff tables) used the as-read
    routing and is SUPERSEDED. Routed points at FM_GATE_DEG = 1.5 deg,
    by the engine's own `_fourier_mellin_routes("auto", ...)`: as read
    336 (bitwise the earlier GPU run's routes), converted **57**; 33 in
    both, 303 only as read, 24 only converted. Still valid with
    recomputed twist columns: the `"off"` runs, the `"always"` fits
    (whole map both cutoffs, far256, patch C) and the D5 anchor census.

144. **V10(m)(2) and FQ11 columns recomputed with the converted map
    from the still-valid earlier arrays (GPU `"off"` and `"always"`,
    `perfF/arrays`), no re-run.** `perfF2/arrays/converted_columns.npz`
    holds, per fitted point (57772, map order): the converted xmap
    twist and theta_eff (both signs), the converted and as-read routes,
    the fitted twist and fitted theta_eff of `map_gpu_off_none`,
    `map_gpu_always_none` and `map_gpu_always_005`, and theta_hat of
    the two `"always"` runs. theta_eff uses the D5 bounding-box centroid
    from the reference PC, (xbar, ybar) = (-17.3, 165.8) px (bounds
    (26, 486, 31, 591), reference PC (328.3, 90.2, 379.4) px).
    - **Routing the converted gate would make** (1.5 deg): 57 routed,
      0 NaN twists, rows 88 to 128 and columns 122 to 146 (the indent's
      deformed lobes; median their CCC 0.51). For comparison 489
      points have |converted twist| >= 1.0 deg.
    - **theta_eff against theta_hat** (the FQ1, FQ2 candidate), on the
      445 selected points: corr(theta_hat, converted theta_eff) 0.997
      at `(None, None)` and 0.998 at `(0.05, None)`, against 0.987 /
      0.989 with the converted twist; median |theta_hat - theta_eff|
      0.146 / 0.134 deg against |theta_hat - twist| 0.551 / 0.526 deg;
      the other sign of the centroid term gives 0.937 / 0.944. So with
      the frame corrected, theta_eff (not the twist) is what FM
      measures, as D22.7 / ledger 116 predicted; under the as-read map
      the same correlation was 0.493 (alt sign 0.633). Counts at 1.5
      deg: |theta_eff| 331, |theta_eff alt| 7, |twist| 57; on the 57
      routed points max |theta_eff - twist| 2.76 deg.
    - **Fitted twist against converted xmap twist** on each run's own
      selection: `"off"` (None) 0.990 (447 points), `"always"` (None)
      0.993 (445), `"always"` (0.05) 0.987 (446). Converged points
      with |fitted twist| >= 1.5 deg: 50 / 47 / 44.
    - **FQ11 over the 57 converted-routed points:** |twist| median
      1.67, p90 2.04, p99 2.66, max 2.68 deg; none above 25 deg (the
      largest |twist| over all 57772 fitted points is 2.68 deg).
      |theta_eff| median 2.14, p90 4.34, max 5.44 deg. |theta_hat|
      from the `"always"` runs over the same 57: median 2.40 / 2.37,
      p90 4.12 / 11.7, max 24.18 / 24.41 deg; no theta_hat at the
      window edge (|theta_hat| >= 29.5) anywhere on the map. Note: the
      maxima are a small cluster of spurious angles near -24 deg at
      column 130 to 131, rows 102 to 106 (3 routed points at (None,
      None), 6 at (0.05, None)), where the converted twist is 1.5 to 2.7
      deg, theta_eff 4.3 to 5.4 deg and the converged fitted twist is
      within 0.1 deg of 0 (or not converged): wrong FM peaks the
      acceptance test must reject. FQ11's reopen trigger is not met.
    - **As-read routed set, for contrast only:** 336 points, |twist|
      median 1.68, max 2.59 deg.

145. **V10(m) GPU records with the CONVERTED map (ledger 143 (ii) to
    (v)): set-up.** Snapshot `snapF_b9551c30` (commit b9551c30; every
    run asserted `kikuchipy.__file__` under it), CuPy overlay
    (cupy-cuda12x 14.2.0, cuFFT 11.4.1.4, cuBLAS 12.9.2.10, cuSOLVER
    11.7.5.82, cuSPARSE 12.5.10.65, nvJitLink 12.9.86), NVIDIA RTX 2000
    Ada Laptop (8 GB). Load recipe of `doc/tutorials/hrebsd_si_indent.ipynb`
    (per-point PC, binning 2, Si phase, reference (10, 10),
    max_iterations 500, crater their CCC < 0.35 masked, 57772 fitted
    points) with ONLY the CrystalMap orientations converted:
    `Rotation.from_euler(euler_ox) * Rotation.from_axes_angles([0, 0, 1],
    -90, degrees=True)`. Runner `perfF3/fm_run.py` (the earlier
    `perfF/fm_run.py` with the conversion and the b9551c30 signatures),
    log `perfF3/maps.log`, records `perfF3/results.jsonl`, per-point
    arrays `perfF3/arrays/`. Before every timing nvidia-smi showed 0 %
    utilisation and no other compute process (whole maps: 0 MiB, no
    apps; patches: only the run's own process after its warm-up). The
    host was loaded by the CPU-records workflow (psutil host CPU % in
    the second before each timing: patches 42 to 62; whole maps 85.9
    (auto, (None, None)), 58.4 (off, (0.05, None)), 37.7 (auto, (0.05,
    None)); the reused off (None, None) record had 20.5), so whole-map
    wall ratios carry a host-load confound of a few percent. The b9551c30
    source changes against 21351744 touch only the gate's grid mapping
    (identity on this full map), the retry batch-size plumbing and the
    information message; the reused `perfF/arrays/map_gpu_off_none.npz`
    ("off" needs no xmap orientations) stays valid. All runs B = 64,
    P = 32 (no out-of-memory halving).

146. **V10(m)(2) GPU whole map `"off"` at `filter_cutoffs=(0.05,
    None)`** (`map_gpu_off_005`; the earlier attempt in `perfF` died
    silently at 3.5 min and left no record). Wall 746.2 s (77.4
    points/s), busy 641.6 s, 903 batches, D5 seed share 0.193; 55498
    converged, **2274 not converged**; iterations median 16, total
    2,610,454; median residual of converged 0.2021; host RSS peak 14.1
    GB.

147. **V10(m)(2) GPU whole map `"auto"`, converted map, `(None,
    None)`** (`map_gpu_auto_none_conv`, against `map_gpu_off_none`;
    `perfF3/compare_map_gpu_auto_none_conv.json`,
    `perfF3/arrays/pointwise_map_gpu_auto_none_conv.npz`).
    - **Routed 57** (as ledger 144 predicted; 0 NaN twists): accepted
      45 (seed 1), kept h_T 12 (seed 0), retried 87 (every first-pass
      non-converged point), retry converted 68 (seed 2, all unrouted).
    - Against "off": converged 57685 -> 57750; **converted 68** (0 by
      routing, 68 by the retry); **worsened 6, all routed**: 3 lost
      convergence (rows/cols (92, 136), (93, 136), (94, 138): accepted
      FM seeds, theta_hat -2.90 / -2.87 / -2.22 deg against theta_eff
      -2.57 / -2.56 / -2.21, then 500 iterations; "off" had converged at
      489 / 386 / 171) and 3 accepted points converged with a residual
      above "off" by 0.9e-5 to 1.7e-5 relative (just outside the parity
      band 4.5e-6, physically the same fit). 6 residuals lower beyond
      the band. Unrouted, unretried points: 57647 of 57647 bitwise the
      "off" homography.
    - Iterations: routed points 11971 -> 10109; accepted points median
      106 -> 63; whole map 1,110,801 -> 1,079,027.
    - Wall 393.6 s against 390.5 s (ratio 1.008, host load 85.9 against
      20.5); busy 300.9 s against 335.2 s. Device seed share 0.399
      (off 0.418); FM time 2.49 s, FM share 0.008. Passes: 57772
      points in 905 batches (20 FM sub-batches), retry 87 points (3
      FM sub-batches); **FM sub-batches 23, slots 736, routed slots
      144**. RSS peak 15.0 GB.

148. **V10(m)(2) GPU whole map `"auto"`, converted map, `(0.05,
    None)`** (`map_gpu_auto_005_conv` against `map_gpu_off_005`;
    `perfF3/compare_map_gpu_auto_005_conv.json`).
    - Routed 57: accepted 40, kept h_T 15, routed then retried 2;
      **retried 2249**, retry converted 2025 (seed 2).
    - Against "off": converged 55498 -> 57547; **converted 2049** (26
      routed, 2025 by the retry, 2 of them routed); **worsened 0** (no
      lost convergence, no residual above the band); 15 residuals lower
      beyond the band. Routed points converged 26 -> 52. Unrouted,
      unretried: 55692 of 55692 bitwise "off".
    - Iterations: routed 18123 -> 9125; accepted median 500 -> 106;
      whole map 2,610,454 -> 1,709,717 (-35 %).
    - Wall 821.5 s against 746.2 s (ratio 1.101; host load 37.7 against
      58.4); first pass 731.8 s, retry pass 83.2 s for 2249 points;
      busy 684.7 s against 641.6 s; device seed share 0.194 (off
      0.193), FM time 9.32 s, FM share 0.014; batches 939; **FM
      sub-batches 91 (20 first pass, 71 retry), slots 2912, routed
      slots 2306**. RSS peak 15.6 GB.

149. **FQ11 and theta_eff on the converted routed set (both cutoffs;
    `perfF3/routed_map_gpu_auto_*_conv.txt` / `.json` hold the 57-row
    table: row, col, seed, conv off/auto, iterations, twist, theta_eff,
    theta_hat).** theta_eff = twist + centroid term, centroid (-17.3,
    165.8) px from the reference PC (D22.7, ledger 116).
    - |twist| over the 57 routed: median 1.67, p90 2.04, p99 2.66, max
      2.68 deg; none above 25 deg. |theta_eff|: median 2.14, p90 4.34,
      max 5.44 deg; max |theta_eff - twist| 2.76 deg. Map counts at 1.5
      deg: |theta_eff| 331, |theta_eff alt sign| 7, |twist| 57.
    - |theta_hat| over the routed: (None, None) median 2.40, p90 4.12,
      p99 24.17, max 24.18 deg; (0.05, None) median 2.37, p90 11.7, max
      24.41. No theta_hat at the window edge (>= 29.5 deg) anywhere.
      All finite theta_hat on the map: 136 / 2300 points, median 1.43 /
      0.93 deg.
    - The maxima are the spurious cluster at rows 102 to 106, columns
      130 to 131 (3 routed points at (None, None), 6 at (0.05, None),
      theta_hat -23.6 to -24.4 deg against twist +1.5 to +2.7 and
      theta_eff +4.3 to +5.4): every one REJECTED (seed 0, h_T kept);
      the whole 8-point cluster (all routed, all seed 0) stays
      unconverged under both "off" and "auto" at (None, None) and
      converges, bitwise as "off", under both at (0.05, None). Including them, corr(theta_hat, theta_eff) on the routed
      set is -0.22 / -0.74; **excluding |theta_hat| > 10 deg (54 / 51
      points): corr(theta_hat, theta_eff) 0.993 / 0.980 against
      corr(theta_hat, twist) 0.983 / 0.957; median |theta_hat -
      theta_eff| 0.166 / 0.169 deg against |theta_hat - twist| 0.609 /
      0.517 deg.** Accepted points: median |theta_hat - theta_eff|
      0.152 / 0.139 deg; rejected 0.72 / 0.82 deg. Every rejected
      routed point (12 / 15) ends bitwise on the "off" homography. So FM measures
      theta_eff on the correct frame, and the acceptance test rejects
      the wrong peaks. FQ11's reopen trigger is not met.

150. **far256 and patch C under `"always"` on the GPU (best of 3 after
    one warm-up; `"off"` beside it for reference), converted map** (the
    `"always"` fits do not depend on the map's orientations).

    | patch, cutoffs | n fit | always best (s) | off best (s) | conv always / off | not conv always / off | iterations always / off | seed 0/1/2 | FM sub-batches, slots, routed slots |
    |---|---|---|---|---|---|---|---|---|
    | far256 (None, None) | 256 | 4.44 | 2.91 | 256 / 256 | 0 / 0 | 2202 / 2231 | 156/100/0 | 8, 256, 256 |
    | far256 (0.05, None) | 256 | 5.17 | 3.65 | 256 / 256 | 0 / 0 | 2216 / 2297 | 123/133/0 | 8, 256, 256 |
    | patch C (None, None) | 618 | 16.69 | 13.28 | 618 / 618 | 0 / 0 | 39273 / 54033 | 37/581/0 | 20, 640, 618 |
    | patch C (0.05, None) | 618 | 27.31 | 21.87 | 587 / 298 | 31 / 320 | 82336 / 242530 | 91/526/1 | 21, 672, 637 |

    (patch C = rows 115:140, cols 100:130, 750 points minus 132 crater
    points; patch C (0.05, None) "always" retried 19 points.) Reps
    spread: far256 always 4.44 to 5.06 s, patch C always 16.69 to 17.21
    / 27.31 to 27.71 s. Device seed share always / off: far256 0.84 /
    0.66 and 0.67 / 0.38; patch C 0.35 / 0.14 and 0.21 / 0.19; FM share
    always: far256 0.40 / 0.32, patch C 0.16 / 0.10. Host load 42 to
    62 %. "always" costs +53 % (far256, nothing to gain) and +26 % (patch
    C (None, None), iterations -27 %); at (0.05, None) on patch C it
    converts 289 more points (298 -> 587) and cuts iterations 66 %.

151. **FQ8 D5 anchor census (GPU, complex128 seed, bitwise skimage's):
    the fraction of fitted points whose D5 translation seed is exactly
    zero while the band-limited phase correlation (rho in [0.02, 0.20]
    cycles/px, integer peak) is not.** Whole map from the still-valid
    `"always"` captures (`perfF/arrays/map_gpu_always_*.npz`; a retry
    recapture of 12 / 169 points gave the identical D5 seed), patches
    from the new runs. `perfF3/census_*.json`.

    | run | D5 exactly zero | band nonzero | **anchored** | anchored not conv under off | its median anchored / other (off) | conv translation error, zero seed / band seed (median px, anchored, conv off) |
    |---|---|---|---|---|---|---|
    | map (None, None) | 10083 (17.5 %) | 43486 | **7318 (12.7 %)** | 79 of 87 non-conv | 39 / 14 | 9.03 / 0.91 |
    | map (0.05, None) | 9511 (16.5 %) | 44477 | **7267 (12.6 %)** | 2179 of 2274 non-conv | 148 / 15 | 6.14 / 0.99 |
    | far256 (None, None) | 247 (96.5 %) | 0 | 0 | 0 | - / 9 | - |
    | far256 (0.05, None) | 236 (92.2 %) | 0 | 0 | 0 | - / 9 | - |
    | patch C (None, None) | 609 (98.5 %) | 600 | **592 (95.8 %)** | 0 | 76 / 97 | 15.9 / 1.61 |
    | patch C (0.05, None) | 607 (98.2 %) | 608 | **598 (96.8 %)** | 316 of 320 non-conv | 500 / 230 | 3.57 / 16.5 (converged off points near zero shift) |

    Whole map (None, None): anchored band-shift norm median 9.1, p90
    18.6 px (7236 >= 2 px, 6012 >= 5 px); anchored points lie in rows 54
    to 175 only (none in the far field rows 0 to 40, where D5 zero is
    correct: 1841 of 10000 with no band shift); on the 7239 anchored
    points converged under "off", the band seed is closer than zero for
    7220. By converged translation: 0 to 1 px 0 anchored; 1 to 3 px 175
    of 25006; 3 to 6 px 1872 of 2128; 6 to 12 px 2889 of 2955; >= 12 px
    2303 of 2474. So about 12.7 % of the map (and 93 % of the points
    whose converged shift exceeds 3 px) start from a zero translation
    the band-limited correlation would place within about 1 px; at
    (0.05, None) these anchored points are 96 % of the "off"
    non-converged points (2179 of 2274).

152. **Summary.** On the Si indent map with the CORRECT (converted)
    frame, "auto" routes 57 points (the deformed lobes), accepts 45 / 40
    FM seeds whose angle tracks theta_eff to 0.15 deg and rejects every
    spurious -24 deg peak; at (None, None) it converts 68 points (all
    via the retry) and worsens 6 routed points (3 lose convergence), at
    (0.05, None) it converts 2049 (2025 via the retry) and worsens none,
    cutting iterations 35 %. Wall time: +1 % / +10 % against "off"
    (host-load confounded; the 0.05 retry pass alone is 83 s), FM share
    of device time 0.8 % / 1.4 %. The larger lever on this map is the D5
    zero-translation anchor (12.7 % of points, 96 % of the 0.05
    non-converged), which "auto" fixes only through the retry.
