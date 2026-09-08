# HREBSD-DIC -- `hrebsd-dic`: plan

Branch `hrebsd-dic` off `develop`. Models per the recorded working
practice: plan/spec on Fable 5 (xhigh, ultracode); tests,
implementation, adversarial review and fixes by Opus 5 agents
(xhigh, ultracode). Tests are written failing before the code they
exercise. Three build stages (user decision 1, 2026-09-07) under
this one spec folder; requirements D-numbers govern; validation.md
holds the oracle suite and the append-only Recorded results ledger.
The spec-stage workflow (per the approved plan): this author draft
-> two adversarial reviewers in parallel (fidelity/theory vs
conventions/integration) -> fixer folds findings -> user verifies +
materializes; re-submission of the three documents to review is a
definition-of-done gate (validation.md).

## 0. Constitution amendments (applied in the spec commit)

1. `specs/roadmap.md`: a new, clearly separated feature-path
   section after the spherical Phase 12 block -- heading
   "Feature path: HREBSD-DIC", branch `hrebsd-dic`, spec folder
   named, the branch policy stated (never merged to `develop`),
   and three stage sub-headings (A/B/C) with gate-shaped
   checkboxes mirroring the spherical phase style. Boxes tick only
   as gates complete on the branch.
2. `specs/mission.md`: a short "Fork-only feature path:
   HREBSD-DIC" section appended -- one scope paragraph; the
   spherical mission text and criteria above it are untouched.
3. `specs/tech-stack.md`: a new "HREBSD-DIC feature path" section
   appended -- everything above it applies on `hrebsd-dic` too,
   plus: the stiffness-input convention (6x6 Voigt GPa crystal
   frame, engineering shears in the Hooke product, D9.4); the
   hand-written numba bicubic kernel note (map_coordinates is not
   jittable; kernel under the standard njit flags with the
   map_coordinates equality oracle, D3);
   `skimage.registration.phase_cross_correlation` as the
   initial-guess dependency-already-present (D5) and
   skimage.transform as test-oracle-only; no new required deps;
   the float-discipline scoping note (f32 storage / f64
   accumulators, D17, the float64 rule being EMSphInx-scoped);
   the branch/CI note (section 1 below).
4. No other constitution file changes. The protected files
   `specs/2026-08-16-constitution/upstream-issue.md` and
   `specs/_research/plan-upstream-merge-0.13.1.md` are never
   touched by any commit of this feature.

## 1. Branch policy and CI implications (binding, user decision 2026-09-07)

- ALL work -- spec, tests, implementation, fixes, tutorial -- lives
  on `hrebsd-dic`. The branch is pushed to `origin/hrebsd-dic` for
  safekeeping. It is **NEVER merged into `develop`** (local or
  remote) and **NO PR into `develop` is opened**. `develop` stays
  HREBSD-free until the user explicitly decides otherwise.
- Stages commit directly onto `hrebsd-dic` in gate order (a stage
  MAY use a short-lived local sub-branch merged back into
  `hrebsd-dic` only; no sub-branch is pushed).
- **CI implication, stated explicitly (corrected 2026-09-07 at
  spec review)**: the fork's tests workflow triggers on PUSH to
  every branch as well as on pull requests
  (`.github/workflows/tests.yml:22-29`; `hrebsd-dic` matches
  neither ignore pattern), so every push of `hrebsd-dic` to
  origin runs the full CI matrix, including the oldest job
  (numpy 1.23.0, numba 0.57, orix 0.12.1, scikit-image 0.21.0).
  An earlier draft wrongly claimed no CI run would ever exercise
  this code. **Push policy, recorded**: these push runs are an
  EXTRA signal, never a recorded gate; to avoid meaningless red
  runs, a failing-tests commit is pushed together with its
  stage's implementation commit (one push per
  implementation/review gate), never alone. The local gates
  recorded in validation.md carry the RECORDED verification
  burden:
  full local suite runs (`-n 0` once for numba caches, then
  `-n 4`), local coverage commands with recorded output, the
  oldest-matrix risk handled by version-gating every orix API
  newer than 0.12.1 (tech-stack.md:17) plus one recorded local
  oldest-matrix run per stage using the constitution's uv recipe
  extended for this feature (scikit-image pinned to the CI oldest
  job's 0.21.0 because the D5 function does not exist at the
  declared 0.16.2 floor, requirements D18, and the signal-method
  suite included so the CrystalMap-assembly path runs on the orix
  floor):
  `uv run --isolated --python 3.10 --with "numpy==1.23.0" --with
  "numba==0.57" --with "orix==0.12.1" --with
  "scikit-image==0.21.0" pytest tests/test_indexing
  tests/test_signals -k hrebsd`. Merging `develop` INTO
  `hrebsd-dic` to stay current
  is allowed (constitution merge rule, tech-stack.md:10).
- Commits signed off (`git commit -s`); the user's uncommitted
  notebook edits and `upstream-issue.md` are never swept
  (tech-stack.md:11).

## 2. Stage A -- IC-GN engine + oracles

Deliverables (D1-D7, D15-D18): `_hrebsd/` package with
`_interpolation.py`, `_preprocessing.py`, `_homography.py`,
`_geometry.py`, `_engine.py`, `_reference.py` (explicit modes
only); `EBSD.hrebsd_dic()` with the frozen signature
(`reference="auto"` raising NotImplementedError until Stage B);
props per D15.6 Stage A list; `.pyi` untouched in Stage A if no
free function ships early (the six public names may land with
their stages). Tests in `tests/test_indexing/test_hrebsd_*.py` +
`tests/test_signals/test_ebsd_hrebsd_dic.py`.

1. Failing tests first, keyed to validation.md V0-V4 + V6:
   interpolation kernel equality + `.py_func` (V0); shape-function
   compose/invert closure and h<->Fe round trip at 1e-12 (V1);
   synthetic warp-refit recovery with MTP placeholder bands (V2);
   pure-rotation analytic cases incl. the frame/sign pins (V4);
   deformed-master end-to-end oracle (V3, engine half); PC-shift
   phantom oracle pinning the D6.3 signs (V6); signature/defaults
   freeze pins; determinism (two runs bitwise); NaN/non-converged
   contract; navigation-mask polarity; lazy input; the D15.7
   `get_map_data` 2-D-prop pin.
2. Implementation in D-order: kernel -> preprocessing -> homography
   algebra -> geometry (per-point PC via `pc_flattened` /
   `extrapolate_pc`) -> engine loop -> dask orchestration
   (`da.blockwise` pairing per the `_map_refine_chunks` precedent)
   -> signal method.
3. Measurement debt discharged at the implementation gate
   (validation Recorded results, dated, with recipes + machine
   ID): V2 tolerance pins; D3 bicubic-vs-quintic order decision;
   D17 f32/f64 storage verdict; D6.3 sign pins; D5 capture-range
   record; D1.4 Hessian-conditioning check; performance baseline
   rows.
4. Adversarial review (two reviewers): fidelity/theory refutes
   against Ernould's equations and the D1/D2/D6 derivations (signs,
   frames, unit system, the corner-norm criterion, the h<->Fe
   algebra re-derived independently) and against the EMsoftOO
   deviation ledger (every deviation either justified or
   reproduced -- none silently in between); conventions/
   integration refutes layout, `.pyi` sorting, numba flags,
   docstring rules, mask polarity, orix floor, license headers,
   coverage 100 % of `_hrebsd/` Stage A modules.
5. Mutation list (every mutant dies by a named test or is recorded
   reviewed-only with its killer stated): transpose the shape
   function; drop the projective divide; compose the IC update as
   `W(dp)^-1 . W`; skip the `W33` renormalization; swap
   `gx`/`gy` in GJ; flip a GJ perspective-term sign; Hessian from
   target gradients; solve `H dp = +g`; warp the reference instead
   of the target; re-warp the warped target (reintroduce
   warp-of-warp); drop the ZMN mean or normalize by stdev vs
   vector norm; mismatch the D2.1/D2.3 H-vs-g normalization
   pairing (dies by V2's intensity-scale-invariance test);
   seed `(dx, dy)` transposed; border applied to one
   side only; PC-centered coordinates built with `+PC`; DD in
   unbinned pixels; `pcy`/`pcz` swapped; correction applied after
   Fe conversion; correction sign flipped (dies by V6);
   `Fe13` not divided by DD; `Fe31` not multiplied by DD;
   non-converged zeroed instead of NaN; grain order not restored.
6. Gate order: failing tests committed -> implementation + measured
   pins -> review + fixes -> `pre-commit run --files` -> signed
   commits on `hrebsd-dic` -> roadmap Stage A boxes ticked ->
   branch pushed. No PR (section 1).

## 3. Stage B -- strain/stress/rotation, references, PC, HR-KAM

Deliverables (D8-D13, D15): `_segmentation.py`, `_tensors.py`,
`_stiffness.py`, `_kam.py`, `_pc_shift.py`; public names
`hrebsd_strain_stress`, `hrebsd_kam`, `hrebsd_pc_shift`,
`segment_grains`, `voigt_stiffness` into `indexing/__init__.pyi`
(sorted); `reference="auto"` implemented (the Stage A
NotImplementedError pin replaced); Stage B props per D15.6.

1. Failing tests: polar-decomposition and strain-measure units
   (pure stretch, pure rotation, composed); closure 3x3 solve vs
   an independent construction (impose F with sigma33 = 0 built in
   from chosen e and C, recover e33 exactly -- the V3 tensor
   half); Bond-rotation invariance + 45-deg cubic textbook pins +
   the 22.5-deg C16' sign pin (transpose-sensitive, D9.5);
   deviatoric fallback; `voigt_stiffness` values vs
   literature Ni/Ti constants; sigma33 ~ 0 self-check; von
   Mises/hydrostatic/principal identities on hand-built tensors;
   `segment_grains` on synthetic label maps (two grains, a
   one-point grain, unindexed -1, 4- vs 8-connectivity,
   threshold sweep); auto-reference = argmax IQ with the tie rule;
   HR-KAM on a constant-curvature synthetic field
   (`= (6/8) * kappa * step` for the frozen 8-neighbor mean --
   the kernel-derived expectation, D12), grain-mask exclusion,
   psi_max guard, NaN
   for isolated points, mrad units (V7 KAM half); PC-shift
   phantom fed through `hrebsd_pc_shift` (V6 reuse) and its
   corrected form fed through `hrebsd_strain_stress` (strain ~ 0;
   the V6 double-correction killer); small-strain
   fast-path equality measurement harness (D8, MTP).
2. Implementation; the private tensor chain (stored
   detector-frame `Fe`, ALREADY D6.2-corrected by the Stage A
   engine -> sample frame -> closure -> split; corrected
   2026-09-07, spec review: an earlier draft interposed a
   "corrected" step here, which would re-apply the D6.2
   correction the homography path already performed -- the exact
   convert-then-correct confusion the spec refuses in D6.2) is
   ONE function shared by `hrebsd_strain_stress` and Stage C's
   `hrebsd_gnd` (D14 preamble).
3. Si-wafer noise-floor benchmark (V5) first executed here
   (download-gated): strain/rotation floors per component recorded
   in validation.md; the D4 preprocessing defaults
   (band-pass/AHE/window) measured against it and re-pinned only
   with a dated record. **Delivered 2026-09-07 at the Stage B
   failing-tests gate as `tests/test_indexing/test_hrebsd_si.py`
   (it was missing from the drafted commit) and executed at the
   IMPLEMENTATION gate: it downloads nothing and skips with a
   message naming `kp.data.si_wafer(allow_download=True)`, so the
   implementation gate fetches the dataset once and fills the four
   placeholders (validation Recorded results entry 32).**
4. Adversarial review: theory reviewer refutes closure algebra,
   Voigt/engineering-shear bookkeeping, frame chain, Biot vs
   Green-Lagrange bookkeeping, KAM definition vs the D12 freeze;
   conventions reviewer as in Stage A plus `.pyi` and docs.
   Mutation list: engineering-shear factor dropped/doubled in the
   Hooke product; Bond rotation transposed (dies by the D9.5
   22.5-deg C16' sign pin + V3's generic-orientation strain
   case); the D6.2 correction re-applied in the Stage B chain
   (dies by the V6 through-`hrebsd_strain_stress` phantom test);
   closure rows swapped;
   b7/b8 misassigned; `R` vs `R^T` in the frame rotation (dies by
   V4); Biot computed as `V - I` (left stretch); KAM mean over
   all-pairs including cross-grain; KAM in radians reported as
   mrad; segmentation threshold compared with >=; principal
   stresses ascending; per-grain reference off-by-one in flat
   index.
   **Killer corrections, dated 2026-09-07 (Stage B failing-tests
   adversarial review; every claim re-measured, validation
   Recorded results entries 28 to 37).** (a) "Bond rotation
   transposed" dies by the D9.5 22.5-deg C16' pin, by the
   ANALYTIC `TestChain::test_strain_recovery_at_a_generic
   _orientation` (2.9630e-04 against 1.1655e-06) and by V3's
   pattern case, now delivered as `test_hrebsd_deformed_master.py`
   (3.0068e-04 against 9.4045e-06). It does NOT die by the
   `sigma33` self-check (3.0914e-04 GPa transposed against
   3.0639e-04 correct), which must never be quoted as its killer.
   (b) "`R` vs `R^T` in the frame rotation" does not die by V4:
   V4's Stage B split is delivered as a pure-algebra harness, so
   the killers are `TestFrameChain::test_the_rotation_uses_the
   _transpose` and the chain's own strain recovery (8.3352e-04
   against 1.1655e-06). (c) "b7/b8 misassigned" does NOT die by
   `test_b7_and_b8_are_not_interchangeable`'s first arm, which is
   blind to it (1.4711e-05 with and without the mutant); it dies
   by `test_traction_free_recovers_the_built_in_tensor` and by
   that test's new per-component arm. (d) Three mutants are ADDED
   to this list, each with its killer: "DETECTOR_Y_FLIP dropped
   from the frame matrix", the Stage B analogue of Stage A's D1.1
   correction (dies by `TestFrameChain::test_the_detector_frame_is
   _the_y_down_one`, an independent library measurement, and
   `::test_sample_to_detector_matrix_carries_the_flip`); "shear
   terms of the traction-free right-hand side dropped" (dies by
   the closure recovery test, 2.7692e-05, and by
   `test_the_shear_terms_of_the_right_hand_side_are_used`); and
   "the small-strain fast path taken for a public result" (dies by
   `TestSmallStrainFastPath::test_the_public_path_is_the_polar
   _one`, which separates the two candidates by 6.09e-04 on the
   chain's own reported `beta`).
5. Gate order as Stage A; roadmap Stage B boxes ticked; push.

## 4. Stage C -- GND + tutorial

Deliverables (D14, D15, D18): `_gnd.py` + public `hrebsd_gnd`;
`doc/tutorials/hrebsd_dic.ipynb`; CHANGELOG entries; bibliography
keys; docs registration (`doc/tutorials/index.rst`, nbval
`NOTEBOOKS` array, sanitize regexes as needed -- tech-stack.md:52).

1. Failing tests: Nye-convention constant-curvature oracle (V7:
   analytic beta field with known constant alpha, < 1 %
   agreement, validated against the math, never another code);
   per-component consumption per assumption tier (no d/dx3
   derivative term is ever fabricated -- estimators
   "a3"/"a5"/"a9" consume exactly their D14.4 documented sets:
   a3 the exact alpha_i3, a5/a9 additionally the d/dx3-neglect
   beta-route entries, D14.2); prefactor pins (30/10, 30/14, 30/20
   asserted literally); antisymmetry fix applied in the detector
   frame only and only on the GND path; NaN-safety at grain
   boundaries and map edges; end-to-end curvature-through-patterns
   tolerance (MTP); Si-wafer GND noise floor recorded vs the
   4-8e12 m^-2 literature scale.
2. Tutorial: synthetic deformation walk-through (deformed-master
   patterns at 480x480, impose F, recover strain/rotation maps),
   Si-wafer noise-floor section, full map gallery (strain
   components, rotation, von Mises, principal stresses, HR-KAM,
   PC-shift residuals, log10 GND), the documented limitations
   (optical distortion, relative-only, single-phase stress), the
   stiffness-input recipe. Notebook rules: hidden first cell,
   thumbnail, black at 77, stored outputs iff > ~2 min on the RTD
   builder, `dask.config.set(num_workers=8)` pinned, drift-safe
   print precision (tech-stack.md:52). The download cells
   (si_wafer) follow the `nickel_ebsd_large(allow_download=True)`
   convention.
3. Mutation list: curl index convention transposed (dies by V7);
   prefactor swapped between estimators; antisymmetry replacing
   beta13/23 by -beta31/32 (backwards); gradients in
   pixels-not-meters; b in nm not m; log10 of signed values
   unguarded.
4. Adversarial review + fixes; nbval local pass recorded; gate
   order as before; roadmap Stage C boxes ticked; push. CHANGELOG
   entries are fork-only (no PR link exists by policy; the entry
   cites the spec folder instead -- recorded deviation from the
   PR-link convention, tech-stack.md:53).

## 5. Stage-independent recipe (applies to every stage)

- Failing tests committed before implementation; every numba
  kernel `.py_func`-tested; run once `-n 0` then `-n 4`; coverage
  100 % of the stage's `_hrebsd/` modules with the command output
  recorded in validation.md.
- Every tolerance MTP with the measuring test named; refutations
  amend requirements.md with a dated correction.
- Review = two independent reviewers (fidelity/theory,
  conventions/integration) + the stage's mutation list; fixes
  folded; decision-critical numbers re-measured at review.
- The full existing suite (`uv run pytest tests -n 4`) green
  before every stage-closing commit -- HREBSD must not perturb any
  spherical or upstream test (recorded per stage).

## 6. Open questions (numbered; each with the conservative default in force and the measurement that resolves it)

1. **Interpolation order** (D3). Default in force: bicubic numba
   kernel. Resolves: Stage A warp-refit oracle (V2) accuracy/speed
   A/B vs `map_coordinates(order=5)`; re-pin only if quintic buys
   > 2x h-accuracy at < 1.5x cost.
2. **`min_step`/`max_iterations` defaults** (D2.5-6). In force:
   1e-3 px / 50 (Ernould criterion / EMsoftOO namelist). Resolves:
   Stage A convergence sweep on V2 + V5 (iterations-to-converge
   histogram, error vs threshold curve); re-pin with dated record
   if the defaults leave systematic error above the interpolation
   floor.
3. **Border/dead-band defaults** (D4.4). In force: `border=0.05`,
   `dead_band=None`. Resolves: V2 with the design shift budget +
   V5 border sweep (noise floor vs border fraction); pin the knee.
4. **KAM kernel defaults** (D12). In force: `order=1`,
   `psi_max=None`, same-grain always. Resolves: V7 constant-
   curvature identity + V5 Si KAM noise floor at orders 1-3;
   record the noise/resolution trade, keep order=1 unless refuted.
5. **Fourier-Mellin initial guess** (D5). DEFERRED from v1. In
   force: translation-only phase-XC seed. Resolves (for the
   record, not for v1): the V4 rotation sweep measures the capture
   range; the recorded angle bounds the regime where v1 is valid
   and sizes the future FM stage.
6. **Optical-distortion correction** (Scope). OUT of v1,
   documented limitation (Ernould 2021). No measurement resolves
   it in v1; the tutorial and docstring state the strain band
   (1e-4..2e-3 on lens-coupled detectors) where it matters.
7. **Per-system L1 GND split** (D14). DEFERRED (user decision 3).
   Scalar estimators ship; the L1 split needs a slip-system
   catalog design (diffsims/orix structures) -- a future spec.
8. **Simulated-reference absolute strain** (D11.4). DEFERRED. In
   force: relative-to-reference only, reference index stored.
   Future spec would build on `EBSDMasterPattern.get_patterns`
   plus the V3 projection helper.
9. **`step_scale` default** (D2.4). In force: 1.0. Resolves: Stage
   A V2 iteration-count/robustness A/B at 1.0/1.25/1.5; re-pin
   only on a measured win with no basin loss.
10. **Preprocessing defaults: band-pass cutoffs, AHE, window**
    (D4). In force: (0.05, None), no AHE, no window. Resolves: V5
    Si noise floor with/without each; dated re-pin on refutation.
11. **f32 storage** (D17). In force provisionally: f32
    patterns/coefficients, f64 accumulators. Resolves: V2 dtype
    A/B at the Stage A gate; verdict recorded in requirements
    D17.
12. **`get_map_data` on `(n, k)` props** (D15.7). In force:
    documented reshape route. Resolves: the Stage A pin on the
    venv orix + a note against the 0.12.1 floor from the local
    oldest-matrix run (section 1).
13. **Si-indent real-data application** (user request 2026-09-07,
    during the Stage A build): AFTER Stages A-C complete, apply the
    full chain to the Zenodo Si-indentation dataset of Cios and
    Winkelmann, record 14059950
    (`AGH__Si_indent_1_512x672.h5oina`, 18.9 GB, patterns stored at
    622x512 px, Oxford h5oina, CC-BY-4.0, DOI
    10.5281/zenodo.14059950; download URL
    https://zenodo.org/records/14059950/files/AGH__Si_indent_1_512x672.h5oina?download=1),
    which reproduces the Wilkinson and Britton (2012) Si-indent
    conditions (doi:10.1016/S1369-7021(12)70163-3). Deliverable:
    strain/rotation/HR-KAM/PC-shift/GND maps around the indent
    compared against the record's own strain-map image
    (`AGH__Si_indent_1-Strain Maps_672x512_X10Y10_SS3x3.png`,
    reference point X10Y10, 3x3 subsampling) and the
    Wilkinson-Britton figures. NOT part of Stages A-C; no
    auto-download (18.9 GB); starts on the user's go once Stage C
    is closed.

## 7. Commits

Signed commits in gate order per stage: (spec commit: this folder
+ section 0 amendments) -> per stage (failing tests) ->
(implementation + measured pins + CHANGELOG when user-facing) ->
(review fixes + re-measurements). Never touch
`specs/2026-08-16-constitution/upstream-issue.md`,
`specs/_research/plan-upstream-merge-0.13.1.md`, or the user's
uncommitted notebook edits. Push `hrebsd-dic` to origin after each
implementation- or review-gate commit; a failing-tests commit
rides along with its stage's implementation commit and is never
pushed alone (the section 1 push-CI policy -- pushes trigger the
fork's on-push CI, an extra signal). NO PR into `develop`, ever,
under this plan
(section 1); the roadmap feature-path boxes are the progress
record.

## 8. Spec-review disposition table (2026-09-07, fixer)

19 findings from the two adversarial spec reviewers
(fidelity/theory, conventions/integration); every finding was
re-verified against the cited sources (EMsoftOO `mod_DIC.f90`,
OpenXY `DislocationDensityCalculate.m`, `.github/workflows/
tests.yml`, `pyproject.toml`, kikuchipy `src/`) and APPLIED; none
rejected. Overlapping pairs are folded into one row.

| # | Finding (short) | Disposition |
|---|---|---|
| 1 | D2.1/D2.3 Gauss-Newton H/g scaling mismatched; "scale cancels in H^-1 g" false (critical) | applied: matched `2/ref_norm^2` / `2/ref_norm` pairing (EMsoftOO `mod_DIC.f90:817, 861-862` verified via the research report), dated correction in D2.3; V2 `test_intensity_scale_invariance` added; plan 2.5 mutant added |
| 2 | D14.4 "a9" cannot come from a "Pantleon-completed set" (6 knowns, not 9); contradicts V7 (both reviewers) | applied: D14.2/D14.4 rewritten to the OpenXY d/dx3-neglect beta route (verified `DislocationDensityCalculate.m:239, 324-336`); a5's route pinned to the same beta route; V7 assertion restated per assumption tier; Scope/roadmap/plan 4.1 aligned |
| 3 | D12/V7 constant-curvature KAM identity off by the fixed 6/8 kernel factor | applied: identity corrected to `(6/8)*kappa*step` for the frozen 8-neighbor mean; V7 computes the expectation from kernel offsets; plan 3.1 updated |
| 4 | D1.3 per-pattern-own-PC centering contradicts D6.2/V6; gamma formulas only in EMsoftOO convention | applied: one common grain-reference-PC frame frozen; D6.2 gains the spec-frame closed form (`gamma = delta`, `alpha_s` = DD ratio, V6-pinned); D6 block parenthetical fixed |
| 5 | Crystal->sample Bond rotation transpose unpinned (both D9.5 pins transpose-blind) | applied: D9.5 gains the 22.5-deg C16' sign pin (odd in theta, transpose-sensitive); V3 strain case requires a generic orientation; plan 3.4 names the killer |
| 6 | V0 1e-12 relative unachievable on the f32 arm | applied: scoped to f64 both sides; `KERNEL_F32_TOL` MTP beside the D17 A/B |
| 7 | V2 scalar `max\|h_fit - h_true\|` mixes px / 1/px units; 1e-4 seed contradicts the 0.01-0.05 px systematic | applied: corner-displacement error-warp metric (px) with 0.1 px drafting seed; requirements Context note |
| 8 | `skimage.registration.phase_cross_correlation` absent at the declared 0.16.2 floor (both reviewers) | applied: D5/D18/tech-stack record the 0.18 introduction and the 0.21.0 tested floor; oldest-matrix recipe pins `scikit-image==0.21.0` |
| 9 | D15.6 `homography` raw-vs-corrected unpinned (D13 needs raw); D8 promises an unstored prop (both reviewers) | applied: RAW h pinned in D15.6 (Fe stays corrected); D13 wording; D8 small-rotation phrase trimmed |
| 10 | CI-on-push premise wrong in plan/roadmap/tech-stack/validation | applied: corrected in all four (tests.yml triggers on push); push policy recorded: failing-tests commits never pushed alone, push CI an extra signal, local gates stay the recorded burden |
| 11 | plan 3.2 tensor chain re-applies the D6.2 correction | applied: chain reworded (stored Fe already corrected); V6 through-`hrebsd_strain_stress` phantom strain test added; plan 3.4 mutant added |
| 12 | "usual maps" gate cites a non-surviving scratchpad; tetragonality silently dropped | applied: checklist inlined in validation Manual with per-item dispositions; tetragonality added to Scope with its revisit path |
| 13 | tech-stack "never imported in `src/`" refuted by `_fit_projection_center.py:32` | applied: scoped to `_hrebsd/` modules; D18 carries the same scoping |
| 14 | oldest-matrix recipe omits `tests/test_signals` | applied: recipe runs `tests/test_indexing tests/test_signals -k hrebsd` |
| 15 | D14.5 silently assumes micrometer steps | applied: `CrystalMap.scan_unit`-driven conversion; ValueError on unknown/"px" units |

## 9. Stage A code-review disposition table (2026-09-07, fixer)

25 findings from the two adversarial reviewers (fidelity/theory,
conventions/integration) plus 2 surviving mutants. Every finding was
re-verified by measurement on Machine A before disposition; the
numbers, recipes and the coverage/oldest-matrix command output are in
validation.md "Recorded results" entries 14 to 26. Overlapping pairs
are folded into one row. 23 applied, 2 rejected with evidence.

| # | Finding (short) | Disposition |
|---|---|---|
| 1 | D6.2 converts the CORRECTED homography with the target PC/DD; re-derivation gives `(0, 0)`/`DD_ref` (critical, fidelity) | applied: confirmed by measurement (3.9967e-04 vs 2.2255e-05 at engine level, 13x the pinned band); `fe_from_homography` fixed, requirements D6.2 amended, new oracle `test_corrected_fe_on_a_deformed_per_point_pc_map` + three rewritten geometry tests (entry 14) |
| M1 | surviving mutant "correction applied after Fe conversion" | resolved by the D6.2 fix: the non-equivalent form dies at `test_correction_precedes_conversion` (verified by injection); the reference-frame form is now provably EQUIVALENT (2.2e-16) and is recorded as such by its own test (entry 15) |
| M2 | surviving mutant "mirror-boundary derivative sign dropped" | killed: `test_gradient_mirror_boundary_keeps_the_fold_sign` (injected -> 1 failed, restored -> 30 passed), plus `test_mirror_boundary_through_py_func` for the value kernel's fold branches (entry 16) |
| 2, 7 | scan-step units guessed; `px_size` silently consumed (major fidelity + critical conventions) | applied in part: units READ and converted with a ValueError naming the axis (D14.5 precedent), requirements D6.1 amended, `TestScanStepUnits`/`TestStepSizeUnits` added. The `px_size == 1.0` guard is REJECTED with evidence -- a placeholder is indistinguishable from a genuine 1 um pixel and the guard would refuse kikuchipy's own shipped data -- and is documented instead (entries 20, 26a) |
| 3 | capture-range limit attributed to the seed, not to the band-pass default (major, fidelity) | applied: measured both ways (2.0 deg vs 4.0 deg), requirements Context clause "no conformant implementation converges there" struck, D4.1 gains the provenance and effect record, docstring restated conditionally, V4 records it. Default NOT re-pinned: V5/plan open question 10 owns it (entry 21) |
| 4 | validation entry 6's seed column does not reproduce (minor, fidelity) | applied: re-measured (127.8 / 159.8 px, matching the test-file comment), correction appended as entry 22; entry 6 left unedited, the ledger being append only |
| 5, 23 | `residual` belongs to the iterate before the last composition (minor x2) | applied: the criterion is re-evaluated at the returned homography (26 per cent error on capped points before), requirements D2.7 amended, `TestResidualIsTheFinalCriterion` added, performance re-recorded (entry 17) |
| 6, 24 | `memory_bytes()` understates by ten per cent (minor x2) | applied: `reference_subregion` counted, exclusions documented, 17344512 -> 18837504 B re-recorded in requirements D16, validation and both docstrings; info-message model and its 22.0 MB updated; `TestPrecomputeMemory` added (entry 18) |
| 8 | `window` uncovered and applied whole-pattern pre-warp (major, conventions) | applied: built over the subregion and applied as a residual weight in the reference frame; requirements D4.3 amended; `WINDOW_REFIT_TOL_480` measured and pinned; `TestWindow` added (entry 19) |
| 9 | coverage 91.98 %, no recorded command output | applied: 100.00 % of every `_hrebsd/` Stage A module, command and output recorded (entry 23) |
| 10 | mirror-fold branches of both kernels uncovered | applied with M2 (entry 16) |
| 11 | `step_scale != 1.0` never executed | applied: `TestStepScale` commits the D2.4 ordering plus a tolerance-free `step_scale=0.0` arm |
| 12 | `grain_labels` never reaches the engine through the public method; no same-grain guard | applied: guard added to `_resolve_index_array` with the positional pairing documented; `TestGrainLabels` and three `TestReferenceResolution` arms added; requirements D11.3 amended |
| 13 | 1-D navigation and the dimension guard untested | applied: `TestNavigationDimensions`; the unreachable 0-D step-size fallback pruned |
| 14 | twelve argument-validation `raise`s unexecuted | applied: `TestArgumentGuards` with every message asserted; the non-finite-CIC guard was UNREACHABLE dead code and is pruned, its D2.6 outcome pinned through `zero_mean_normalize` instead |
| 15 | `n_pixels`/`memory_bytes`/`get_info_message(chunksize=None)` uncovered | applied with finding 6 |
| 16 | low-pass-only `filter_cutoffs` arm uncovered | applied: parametrized over `(0.05, None)`, `(None, 0.4)`, `(0.05, 0.4)` |
| 17 | D17 verdict overclaims its own measurement | applied: narrowed to "f32 spline coefficients confirmed, measured", with the f64 steepest-descent/reference/coordinate planes recorded as by design (they are accumulands, not bulk storage) |
| 18 | oldest-matrix run misclassified as Stage B; Stage A's undischarged | applied: run and recorded with versions and output; validation's Local-gated section corrected (entry 24) |
| 19 | `grain_labels`/`misorientation_threshold` promise Stage B behaviour | applied: both descriptions reworded; `misorientation_threshold` stated to have no effect in this release |
| 20 | numpydoc: `_as_parameters`/`_prepare` missing sections; three missing Raises | applied: `numpydoc.validate` now reports no PR01/RT01 on any `_hrebsd` object |
| 21 | `_RESIDENT_PLANES` comment and the reused `HOMOGRAPHY_PROP_SIZE` | applied: `_N_STEEPEST_DESCENT_COLUMNS` and `_RESIDENT_F64_PLANES` replace it, comments corrected |
| 22 | one docstring line over 72 characters | applied: an AST/tokenize scan of all six modules now reports none |
| 25 | import audit allows top-level `skimage` | applied: `test_scikit_image_is_never_imported_at_module_scope`, which also asserts the deferred import IS inside `initial_guess` |
| -- | `correct=False` diagnostic path should use the exact conjugation (part of finding 1) | REJECTED with evidence: pinned by the frozen `test_conversion_uses_the_relative_target_pc`, feeds only V6's uncorrected arm and D13, and the two conversions are exact mutual inverses as they stand (entry 26b) |

## 10. Stage B failing-tests review disposition table (2026-09-07, fixer)

12 findings from the adversarial reviewer of the Stage B
failing-tests commit, plus 6 mutant-coverage items. Every finding
was re-verified by measurement on Machine A before disposition --
the numbers, the recipes and the gate runs are in validation.md
"Recorded results" entries 28 to 38. 12 applied (two of them with
a better remedy than the one suggested, recorded in the row), 0
rejected. Every mutant-coverage item is closed by a named test or
by a recorded correction to the killer this plan's section 3.4
names.

| # | Finding (short) | Disposition |
|---|---|---|
| 1 | KAM module's two tolerance families mutually unsatisfiable; the oracle's `arccos` pair angle loses 8 digits at 1e-4 rad (critical) | applied: measured (oracle 2.6221e-09 relative from its own closed form; `arctan2` form 3.7e-16), `pair_angle` rewritten, requirements D12 amended with the date, `_kam.py` docstring warned, and a new library test guards the conditioning (entry 28) |
| 2 | `test_the_public_path_is_the_polar_one` unsatisfiable: the chain's pure-rotation strain floor is the reduction plus the closure, not the split (critical) | applied with a STRONGER remedy than the suggested MTP band: the test now decides which split was taken on the chain's own reported `beta`, where the polar answer is reproduced exactly and the small-strain one is 6.09e-04 away, so no tolerance is guessed at all. Requirements D8 and validation V4 amended with the dated floor table; the V4 `ROTATION_STRAIN_LEAK` seed is refuted by 25x (entry 29) |
| 3 | V3's tensor half and the whole V5 Si benchmark absent from the commit (major) | applied: `test_hrebsd_deformed_master.py` delivers V3's pattern-level strain oracle (2 MTP placeholders; measured worst strain error 9.4045e-06, and it kills the transposed-Bond mutant by 32x), and `test_hrebsd_si.py` delivers V5's four tests plus the four sweep harnesses, download-gated and skipping cleanly. Both file-layout deviations recorded in validation V3/V5 (entries 31, 32) |
| 4 | `test_the_chain_and_the_public_function_agree` compares at `atol=0` across two constructions of one orientation, which differ by an ulp (major) | applied: the chain is fed `xmap.rotations.to_matrix()`, `rtol=0, atol=0` kept, and the substitution's independence is argued from the separate `rotate_vector` pin (entry 37) |
| 5 | Two Stage A `reference="auto"` pins pass vacuously; the caught error now comes from the `_segmentation.py` skeleton (major) | applied via the finding's own documented alternative rather than deletion, because the failing-tests commit must not move the Stage A regression count: both carry an explicit superseded-by note naming their positive replacements, and the Stage B IMPLEMENTATION gate deletes them (recorded in validation's Stage B unit-suite block and entry 33) |
| 6 | `_tensors.py` uses `M = diag(1,-1,1) @ R` while frozen D7 writes `R^T beta R`; the amendment was only promised (major) | applied: requirements D7 carries the dated correction naming the composed matrix and its two measuring tests, D1.5 records that the flip is a reflection (det -1) and cannot be absorbed into the detector rotation, and validation records both |
| 7 | `test_b7_and_b8_are_not_interchangeable` is blind to the mutant it claims (minor) | applied: measured (1.4711e-05 with AND without the mutant), comment corrected to say what the test does, and a discriminating per-component arm added; the real killer is named in the module's mutation map and in section 3.4 above (entry 36a) |
| 8 | `segment_grains`' multi-phase ValueError narrows the frozen default `reference="auto"` against D9.6 (minor) | applied via the finding's FIRST option (segment per phase) rather than its second (amend D11.1 to record the narrowing): narrowing a frozen decision with no measurement refuting it is the worse of the two, and the implementation does not exist yet so the cost is a docstring and a test. Requirements D11.1 records the decision with the date; two positive tests replace the guard, the second a library measurement of m-3m vs 6/mmm (entry 34) |
| 9 | The `(ny, nx)` shape contract is implicit for a one-row map and undecided for a one-column map (minor) | applied: measured on orix 0.14.2 that BOTH flatten to `(n,)`, the rule pinned to the row/col grids in both docstrings, requirements D11.1 amended, and three tests added across the two modules (entry 35) |
| 10 | Stage B file-layout deviations and the new MTP placeholders absent from validation.md (minor) | applied: a Stage B file-layout table in the V3 block, per-block notes in V5 and V7, a Stage B MTP inventory beside the Stage A one, and entries 28 to 38 in the ledger |
| 11 | `SMALL_STRAIN_ENABLE_ANGLE_DEG` is quantized to the sweep grid; D8 names a measuring test that is not what is delivered (minor) | applied: the test now brackets and BISECTS the 1e-6 crossing to 1e-6 deg (grid would record 0.05, crossing is 0.0779), and requirements D8 carries a dated measuring-test correction saying the equality is measured analytically and why (entry 30) |
| 12 | 8-connectivity row wraparound uncovered: the only `connectivity=2` case is a 2x2 map (minor) | applied: `test_eight_neighbours_do_not_wrap_around_a_row` on a (2, 3) map whose only same-orientation pair is two columns apart in one row (entry 36b) |

Mutant-coverage items, all closed: the transposed-Bond co-killer is
delivered (V3 patterns) and the `sigma33` self-check is recorded as
NOT a killer of it; the `R` vs `R^T` row of section 3.4 is corrected
to name its real killers; the b7/b8 row likewise; three mutants are
added to section 3.4 with their killers (`DETECTOR_Y_FLIP` dropped,
traction-free shear terms dropped, the small-strain path taken
publicly); and the two items the reviewer verified as already dying
hard -- the D6.2 re-application and the plan's remaining Stage B
mutants -- are left as they are, with their measured margins
recorded above.
