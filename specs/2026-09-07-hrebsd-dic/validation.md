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
