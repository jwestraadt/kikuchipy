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
   floor. **RESOLVED 2026-09-08 (Stage B implementation gate,
   validation entry 45; requirements D2.5/D2.6 amended): BOTH
   CONFIRMED, neither re-pinned.** The error-versus-threshold curve
   on the Si wafer is flat to 2 per cent over a hundredfold change
   in `min_step` while convergence goes 97/87/26 of 100, so the
   threshold is not the limiting factor; and the cap is doing real
   work (1/100 converge at 10, 8 at 20, 87 at 50, 96 at 100).
3. **Border/dead-band defaults** (D4.4). In force: `border=0.05`,
   `dead_band=None`. Resolves: V2 with the design shift budget +
   V5 border sweep (noise floor vs border fraction); pin the knee.
   **RESOLVED 2026-09-08 (validation entry 45; requirements D4.4
   amended): NO KNEE EXISTS on this dataset and the default STANDS
   on its original reasoning.** The floor moves 11 per cent over a
   fourfold change in the knob while convergence falls 93/87/83/52
   of 100; the border exists for beam-scan translations, which is
   precisely what these fits do not track, so the sweep cannot pin
   it. Worth re-running on open question 13's dataset.
4. **KAM kernel defaults** (D12). In force: `order=1`,
   `psi_max=None`, same-grain always. Resolves: V7 constant-
   curvature identity + V5 Si KAM noise floor at orders 1-3;
   record the noise/resolution trade, keep order=1 unless refuted.
   **RESOLVED 2026-09-08 (validation entry 45; requirements D12
   amended): `order=1` CONFIRMED, and there is NO trade to record.**
   Median KAM 5.2402 / 8.2715 / 11.1350 mrad at orders 1 / 2 / 3
   with the spread tripling and the cost quadrupling, so order 1
   wins on noise, on resolution and on cost at once.
5. **Fourier-Mellin initial guess** (D5). DEFERRED from v1. In
   force: translation-only phase-XC seed. Resolves (for the
   record, not for v1): the V4 rotation sweep measures the capture
   range; the recorded angle bounds the regime where v1 is valid
   and sizes the future FM stage.
   **PLANNED 2026-10-06 (Johan decision 1): Fourier-Mellin is
   planned as Stage F, AFTER Stage E, specified for both backends
   at once as one xp-agnostic code path with the CPU as parity
   oracle, behind the seed seam Stage E freezes (requirements
   D21.5); design notes in section 11 "Follow-ups". Not yet
   commissioned; the default above stays in force until it is.**
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
    **RESOLVED 2026-09-08 (validation entry 45; requirements D4.1,
    D4.2 and D4.3 amended): ALL THREE CONFIRMED, none re-pinned.**
    The high-pass is the whole measurement on real data -- 1 of 100
    patterns converges without it against 87 with it, the exact
    opposite of the noise-free oracle result Stage A recorded -- the
    low-pass buys two converged points for a 0.6 per cent worse
    floor, and the Hann window takes convergence to 14 of 100.
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
    IN EXECUTION (2026-09-08, user go): scope narrowed by user
    decision to the FULL-RESOLUTION replication of Winkelmann et
    al. 2025 (Ultramicroscopy 276, 114180) only, deviatoric
    closure only; their simulation-supersampling ("parameter
    super-resolution") method and a stress section are DEFERRED
    follow-ups. Shape correction: the file stores patterns as 512
    rows x 622 columns, (58500, 512, 622) uint8 (the record
    filename's 672 is not the stored pattern width). Execution
    plan: `si-indent-application-plan.md` in this folder; gates in
    roadmap.md section "Si-indent application"; recorded results
    from entry 78.

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

## 9. Performance and super-resolution follow-up paths (recorded 2026-09-09)

Measured baselines these paths are judged against (ledger entries 80-82):
full Si-indent map at 512x622 px ran 2.34 h = 6.85 patterns/s on 8 workers
(median 14 iterations; far-field zones 10-12 patterns/s, indent rim ~1.8);
row-10 noise floor 0.0267 mm/m / 0.0258 mrad; the paper's MapSweeper runs
~20 patterns/s on two RTX 4090 GPUs. Every estimate below is a PROJECTION
until its named measurement runs; none is commissioned by this section.

1. **More Dask workers** (config only, free). The IC-GN loop is
   compute-bound and pattern-parallel (unlike the spherical FFT path,
   which saturates memory bandwidth near 8 workers), so 16-20 workers on
   a 20-logical-core machine should approach 2x. The 8-worker pin stays
   in tutorials for machine-independent stored output; non-tutorial runs
   may raise it freely. Resolves: one timed far-field patch at 8 vs 16
   vs 20 workers, recorded.
2. **Plain pattern binning** (one-evening experiment). Cost per
   iteration scales with pixel count: 2x2 binning to 256x311 is ~4x
   speed (~27 patterns/s map-wide projected; ~55 with item 1). The
   floor cost is the unknown: with an experimental reference BOTH images
   lose information, est. ~2x floor (to ~0.05-0.08 mm/m, still under
   the paper's published full-res 0.079). Camera-side binning
   additionally buys acquisition rate and per-pixel SNR. Resolves:
   rebin the Si-indent data 2x2 and 4x4, rerun the row-10 estimator +
   a timed patch, record the speed-vs-floor curve.
3. **min_step relaxation** (one-evening experiment). Entry 80: deformed
   points reach their answer well before the corner-norm criterion
   declares convergence (~128 iterations declared where the fit was
   already stationary; the 50-cap failure was a threshold artefact, not
   capture). A looser min_step (2e-3 to 5e-3 px) could cut iteration
   counts ~2x in deformed zones. Resolves: V2 warp-refit accuracy +
   Si row-10 floor at each candidate threshold, dated re-pin only on a
   measured win.
4. **Neighbour-seeded propagation** (reliability-guided DIC; a feature,
   ~half to one build stage). Today every point seeds from
   translation-only phase cross-correlation; deformed points then spend
   80-500 iterations. Seeding from an already-converged neighbour's
   homography (processing order by a reliability queue, residual-gated
   acceptance, phase-XC fallback) collapses iterations in smooth fields
   - the classic DIC strategy. Iterations, not pixels, dominate
   deformed-zone cost, so est. 3-8x there. Risks to spec: error
   propagation across grain/twin boundaries (gate by residual and
   grain_id), determinism of the processing order. Oracle: identical
   converged answers to independent seeding within a pinned band.
5. **GPU backend for the DIC engine** (a feature, ~one build stage,
   Phase-12 recipe). Batch many patterns' IC-GN iterations in lockstep
   on the device (CuPy): batched bicubic evaluation, gradient
   reductions, 8x8 solves; backend="gpu" with the CPU path as parity
   oracle, no silent fallback. Est. 5-20x; the paper's own two-4090
   figure (~20 patterns/s at full res, heavier per-candidate work) is
   the calibration point. Risks: divergent per-pattern iteration counts
   (mask-and-retire within the batch), f32 discipline vs the pinned
   bands, device bicubic parity.
   **COMMISSIONED 2026-10-06 as Stage E (section 11; requirements
   D21, validation V9).**
6. **Binned-gradient hybrid** (cheap partial win, rides with 7b).
   Precompute the steepest-descent images and Hessian from a FULL-RES
   reference, then bin them to the target grid: better gradients than
   differentiating a binned image, one-time per-grain cost, zero
   per-iteration cost. May claw back part of item 2's floor penalty.
   Resolves: item 2's experiment re-run with hybrid gradients.
7. **Super-resolution** (the deferred OQ13 scope; a feature path).
   Explicitly NOT an analysis-speed lever: Winkelmann et al. 2025 state
   the operation count stays comparable across binning x supersampling
   scenarios, and the op-count argument reaches the same conclusion for
   any variant here. What it buys is detector pixels, storage (~two
   orders of magnitude) and acquisition rate at near-full-res precision
   (their Table 1: 311x256 @ 7x7 supersampling matches full resolution).
   Two variants to spec:
   (a) Simulation-reference, as published: dynamical master + projection
   model + supersampled binning of the simulation; requires the bias
   treatment (relative differencing) and a new reference mode; largest
   scope.
   (b) Asymmetric experimental: one slow full-resolution reference per
   grain, fast binned targets, reference evaluated with pixel-footprint
   integration (s^2 samples per binned pixel, so compute returns to
   full-res scale per comparison); no simulation needed; moderate scope;
   pairs naturally with item 6.
8. **Combined outlook** (projection): items 1+2(+3) put CPU-only maps in
   the tens of patterns/s; item 4 or 5 alone reaches the same class at
   full resolution; 1+4+5 together plausibly exceed 100 patterns/s.
   Order of attack when commissioned: 1-3 (measurements), then 4, then
   5, with 6/7 as the acquisition-economy track.

## 10. Stage D -- neighbour-seeded propagation (commissioned 2026-09-09)

Plan section 9 item 4, commissioned by the user 2026-09-09; requirements
D20 (frozen 2026-09-09) govern; oracles in validation V8; branch policy
section 1 unchanged (hrebsd-dic only, no PR, never merged).

Deliverables: `seed_from_neighbors` on `EBSD.hrebsd_dic` (D20.1), the
three-phase cascade inside `run_hrebsd_dic`/_engine.py with per-point h0
support (D20.2), the `seed_round` prop (D20.5), PASS1_CAP and
SEED_EQUIVALENCE_TOL measured then pinned, and the D20.7 performance
record on the Si rim patch.

1. Failing tests first (tests/test_signals/test_ebsd_hrebsd_dic.py +
   a new tests/test_indexing/test_hrebsd_seeding.py): default-off
   bitwise pin; signature freeze; V8 rescue oracle (a deformed-master
   map with a rotation ramp where phase-XC-only fails at far points but
   the chain rescues them, expectation from the imposed field); D20.6
   equivalence band (MTP placeholder); determinism pins (two runs,
   chunksize, lazy) on the seeded path; grain-boundary isolation (a
   two-grain map with a discontinuity where a crossed seed would
   visibly corrupt the second grain); mask isolation; seed_round
   encoding incl. rescue -2 and never -1; tie-order pin (a constructed
   neighbourhood where the frozen offset order decides); PASS1_CAP
   semantics (cap never truncates a final answer, rescue pass exists);
   the PC-transport bound measurement recorded.
2. Implementation: per-point h0 plumbing through the chunked fit;
   frontier/round orchestration on the host (round membership and seed
   choice computed from completed rounds only); the rescue pass;
   prop assembly. Measurement debt: PASS1_CAP (fraction of pass-1
   conversions lost at candidate caps on Si data), SEED_EQUIVALENCE_TOL
   (~2x measured), the PC-transport bound, D20.7 rim timing.
3. Adversarial review (theory: cascade correctness, determinism proof
   read, equivalence honesty; conventions: API/docs/coverage 100 % of
   touched _hrebsd modules) then bug injection ALONE. Mutation list:
   seed from the HIGHEST-residual neighbour; tie order shuffled; seed
   across grain_id; use same-round results (race); ignore PASS1_CAP;
   skip the rescue pass; mislabel seed_round; seed from an unconverged
   neighbour; drop the mask check; per-point h0 array off by one in
   flat order. Then fixer; every surviving mutant killed and
   re-verified.
   **Mapped at the failing-tests gate (2026-09-09)**, in the mutation
   map at the foot of `tests/test_indexing/test_hrebsd_seeding.py`,
   with the coverage the review found missing closed there: "per-point
   h0 array off by one" had NO designed killer (every fixture fitted
   at most one point per cascade round) and now has one, the two-ramp
   map of `TestSimultaneousSeeding`, verified by simulation; and
   "highest-residual seed", "tie order shuffled" and "seed from an
   unconverged neighbour" were unit-only kills resting on an
   unasserted seam, which `TestSeedChoiceIsTheSeamTheCascadeUses` now
   asserts by spying on `_engine.choose_seed_indices` during a real
   seeded run. Two limitations are RECORDED rather than papered over:
   the frozen offset order equals ascending flat index for every map
   width >= 2, so a tie case cannot separate it from a flat-index
   argmin (the `NEIGHBOR_OFFSETS` constant pin carries that mutant),
   and the mask gate on the SEED side is unobservable at map level
   because a masked point is never converged.
4. Gates as every stage (hrebsd suite green, default-path bitwise
   unchanged, coverage, ruff, oldest-matrix, full suite), signed
   commits pushed together (failing tests ride with implementation),
   roadmap Stage D boxes ticked, ledger entries from 83.

## 11. Stage E -- GPU backend for the DIC engine (commissioned 2026-10-06)

Plan section 9 item 5, commissioned by the user 2026-10-06;
requirements D21 (drafted 2026-10-06) govern; oracles in validation
V9; spec-gate measurements in ledger entries 89 to 99; branch policy
section 1 unchanged (hrebsd-dic only, no PR, never merged). Johan's
decisions of 2026-10-06: (1) the seed stage is a frozen, pluggable
seam (D21.5), because Fourier-Mellin follows as Stage F for both
backends; (2) `backend="gpu"` with `seed_from_neighbors=True`
raises (D21.12, a recorded default, approved at this plan gate
on 2026-10-06, 11.4 approval record); (3) the complex128 device seed default with a
complex64 opt-in is a main-session lean, listed for approval
below, not a decision. THE PLAN GATE IS THE APPROVAL GATE: no task
below starts before Johan has approved or changed every item of
11.4, the D17 amendment (D21.4) and the two constitution
amendments (mission.md and tech-stack.md, HREBSD sections, dated
2026-10-06). PASSED 2026-10-06 under Johan's overnight waiver:
every item approved as written (11.4 approval record).

Deliverables: `backend` on `EBSD.hrebsd_dic` and `run_hrebsd_dic`
(D21.1) with the freeze and defaults pins updated (the D15.4
amendment); `_hrebsd/_gpu.py` -- the HREBSD gate importing the
Phase 12 shim, version floor and lock (D21.2), the per-call
session, the VRAM model and batch chooser with the out-of-memory
handling (D21.10), and the device runner at the `_run_chunks`
contract (D21.9); `_hrebsd/_batched.py` -- the xp-agnostic batched
core: `SeedContext`, `SeedState`, `SeedBatch`, `seed_spectra` and
`seed_homographies` (the seam, D21.5), `_launch_layout` and the
lockstep IC-GN with status flags and retirement (D21.6), over a
numpy (float64 only) and a cupy `KernelNamespace` (the CUDA source
lives in a device-only module; module layout is implementation
freedom, the D21.5 and D21.9.5 names are not); the engine-only
`device_precision` and `seed_precision` (D21.4, D21.5); the
`seed_from_neighbors` raise (D21.12); the import-audit amendment
(D21.13); the docstring, the CHANGELOG entry and the tutorial
markdown cell (D21.17); every V9 MTP pin measured and pinned; the
D21.16 performance record.

Files touched outside `_hrebsd/`, kept to the minimum for merge
hygiene (develop is merged INTO this branch periodically):
`signals/ebsd.py` (inside the fork-only `hrebsd_dic` method and its
docstring only), `CHANGELOG.rst` (one append in the HREBSD block),
`doc/tutorials/hrebsd_si_indent.ipynb` (one markdown cell; a
fork-only file), the fork-only `tests/test_signals/
test_ebsd_hrebsd_dic.py` and `tests/test_indexing/
test_hrebsd_engine.py`, and the new `tests/test_indexing/
test_hrebsd_gpu.py`. NEVER `_spherical/*`, the root `conftest.py`,
`pyproject.toml` or `doc/user/installation.rst`. Spec files
(added 2026-10-06, spec review): this spec commit edits
`specs/mission.md` (one dated paragraph inserted immediately before
the "Fork feature path: NLPAR" heading), `specs/tech-stack.md` (one
bullet appended to the HREBSD list, immediately before the "NLPAR
feature path" heading) and `specs/roadmap.md` (the "Stage E"
section inserted immediately before the "Feature path: NLPAR"
heading; its boxes are ticked later). HROSM is developed
develop-first, so develop may edit near those anchors: a conflict
there is resolved by keeping both sides, and the full suite runs
after every merge of develop (item 6).

1. Failing tests first, keyed to V9, in the new
   `tests/test_indexing/test_hrebsd_gpu.py` laid out as D21.14.1
   says, targeting the frozen names of D21.5 and D21.9.5 (the
   argument lists the failing tests fix are recorded in the test
   module's docstring). Default-suite classes: `TestBackendSwitch`
   (V9(a), the check order and the docstring test),
   `TestSeedFromNeighborsOnGpuRaises` (V9(b)),
   `TestAvailabilityGate` (V9(c), fake cupy), `TestSeedSeamContract`
   (V9(d), numpy), `TestSeedSeamHonoursArbitraryH0Numpy` (the V9(e)
   numpy twin), `TestBatchedCoreNumpy` (V9(f) numpy twins at
   float64, (h), (i)), `TestRunBatchesNumpySession` (through
   `run_hrebsd_dic` with a numpy session built by a patched
   `_gpu._make_session`: wiring probes, map order on the two-grain
   F5, the failure contract, the residency spy on F7, the (n)
   twins), `TestLazinessNumpySession` (the V9(m) laziness oracle),
   `TestLaunchLayout` (the V9(m) layout pin), `TestCudaSourcePins`
   (V9(m): no `atomicAdd`), `TestBatchModel` (V9(o)),
   `TestDriftTripwireCpuHalf` (V9(l)), `TestFixtureGating`,
   `TestExpectGpuCanary` and `TestNoModuleScopeCupy` (V9(p), with
   the `_engine` import-direction check). Gated classes:
   `TestGatedGate`, `TestGatedSeedParity` (with the device output
   contract of V9(d)), `TestGatedSeedSeam` (V9(e), Johan decision
   1), `TestGatedParity` (V9(f), (g)), `TestGatedKernelAB` (the
   V9(f) per-kernel A/B at float64), `TestGatedBatchedSemantics`
   (V9(h), (i)), `TestGatedIntensityAndUpdateRule` (V9(j), (k)),
   `TestGatedDriftTripwire` (V9(l)), `TestGatedDeterminism` (V9(m),
   with the launch-dimension spy), `TestGatedRobustness` (V9(n)),
   `TestGatedVramCalibration` (V9(o)), and `TestGatedThroughput`
   (recorded with `record_property`, asserting nothing but
   completion: the floor is a ledger record, D21.16). The default
   suite's wall time is recorded at this gate (D21.14.3). Edits of
   existing pins, each with a dated comment: `FROZEN_SIGNATURE`
   gains `backend` (D15.4 amendment);
   `test_run_defaults_are_frozen` gains `backend`,
   `device_precision` and `seed_precision`; `TestImportAudit` gains
   `cupy` and `gc` in the allowed tuple and the cupy module-scope
   arm (D21.13). Every device band and device-side literal carries
   a `FIXME-pin` placeholder, and so does every literal that needs
   the new numpy path (it does not exist yet); only the CPU-side
   literals, the (l) CPU half, are measured at this gate (D21.8).
   The failing-tests commit is not pushed alone (section 1).
2. Implementation, in dependency order: the gate (D21.2) -> the
   `backend` plumbing and the D21.1 check order with the D21.12
   raise -> the xp-agnostic batched core under numpy (the seed seam
   with `SeedContext`, `SeedState` and `SeedBatch`; P sub-batches
   padded to P; the lockstep loop with the carried matrix, the
   adaptive K, the windowed algebra, the NaN-propagating corner
   norm and the flags; retirement, final criterion, packing; the
   numpy twin at float64 only, reductions without BLAS) until the
   numpy-twin tests pass -> the CUDA kernels (the fused pixel kernel
   and the reduce, solve, update and retire kernel; two-stage
   fixed-order reductions; the layout from the pattern geometry
   only; the fold in floating point before any integer conversion,
   with the non-finite flag first; explicit reciprocal constants,
   no `--use_fast_math`) -> the session, the `R_MAX` residency and
   grain-pure batching with padding -> the VRAM model, the default
   B chooser and the out-of-memory loop -> the information message
   -> the docstring. The CPU path is not edited beyond the
   `backend` branch point (which sits before the shared chunksize
   lines, D21.10.1).
3. Measurement debt at the implementation gate (validation, ledger
   from entry 100, recipe and machine ID each): every V9 MTP pin;
   the VRAM model's terms calibrated separately; B invariance at B
   in {8, 32, default} (E4); NVRTC provenance and a wheel-only
   compile check (E6); the default-B sweep (E3); the many-grain
   overhead of grain-pure batching and the `R_MAX` bound on F7 (E5);
   the real-data per-point parity on far256 and patch C at both
   seed precisions (V9(q), E12); the D21.16 performance record at
   both device precisions and the go/no-go floor (E7); and,
   REQUIRED, the device-wait fraction of the v1 read route in the
   whole-map run (E8, D21.11). Every gated measurement runs on the
   pinned overlay of D21.15; a version bump re-measures the
   device-side pins first.
4. Adversarial review (two reviewers -- fidelity/theory: device
   semantics against the CPU loop line by line, the status flags
   against the CPU exception contract, the determinism argument,
   the precision surface against the D17 amendment, the seam
   contract against Johan decision 1; conventions/integration:
   gate wording, import audit, coverage 100 % of the touched
   `_hrebsd` modules, docstrings, merge hygiene, no `_spherical`
   edit, no file outside the list above), THEN bug injection ALONE
   on the GPU machine, each mutant re-injected to verify its kill.
   Mutation list (each dies by a named V9 oracle or is recorded
   reviewed-equivalent with its argument): M1 an atomic reduction
   in place of the two-stage one ((m) the `atomicAdd` source pin;
   (m) two runs alone is NOT a reliable killer: the mixed build's
   f64-atomic variant was identical 4 of 4 at 512x622, ledger 95);
   M2 blocks per pattern derived from B ((m) the launch-dimension
   spy at B = 8 against 32, and B invariance; the pure-function
   layout pin cannot see the call site); M3 f32
   block and cross-block reductions ((f) first-step band); M4 the
   8x8 solve or the update in f32 ((f), (k)); M5 the retire mask
   ignored, retired patterns keep updating ((h) alone against
   batched, counts); M6 the exit test before the update is applied
   ((f) counts, h band); M7 no final criterion at retirement ((h)
   residual-is-final-criterion on the capped pattern); M8 a
   non-finite coordinate not flagged before the fold ((i) planted
   coordinate); M9 a zero ZMN norm not flagged ((i) the
   integer-constant target at `filter_cutoffs=(None, None)`); M10
   the seam's rows ignored and the seed recomputed ((e) the
   one-iteration arm and the discriminating count); M11 the seed
   stage reached through a captured reference
   instead of the module global ((d) spy); M12 the seed's
   `(dy, dx)` transposed ((d) equality); M13 the cross-power
   spectrum conjugated on the wrong side ((d)); M14 complex64 run
   when complex128 is requested ((d) dtype asserts, seed count);
   M15 the crops not ZMN'd before the FFT ((d) the dimmed arm, the
   designed killer; with a ZMN'd reference the DC bin of the
   cross-power spectrum is 0 whatever the target, so on unscaled
   fixtures M15 is very likely equivalent, and if the dimmed arm
   cannot separate it either it is recorded reviewed-equivalent
   with that argument); M16 padded slots
   written into the output ((h) sentinel padding); M17 grain order
   not restored after grain-pure batching ((a) wiring, F5 map
   order); M18 a batch run against the wrong grain's residents (F5
   parity; an identity spy on the residents); M19 a device
   exception swallowed into NaN rows ((n) planted exception); M20
   an out-of-memory loop that never halves or never terminates, or
   pools not freed ((n) spies, leak pin); M21 gate stages
   reordered, or the shim after stage (c) ((c) order pin); M22 an
   FFT-only stage-(c) probe ((c) fake records); M23 the lock
   dropped or the threaded scheduler not forced ((n) scheduler
   spy, 4-worker stress); M24 cupy imported at module scope ((p));
   M25 the `seed_from_neighbors` raise after the gate or after
   reference resolution ((b) spies); M26 a silent CPU fallback on
   gate failure ((a), (c)); M27 a shared helper perturbed, moving
   the CPU path (pre-Stage-D literal pins, (l) CPU half); M28
   `step_scale` ignored on the device, or `norm_dp` taken before
   scaling ((h) `step_scale` arms); M29 the window weights not
   applied on the device ((h) window arm); M30 the W33
   renormalisation skipped ((f), (k)); M31 the corner norm over
   the wrong corners ((f) counts); M32 `h` held in f32 on the device
   ((f) float64 band, (k)); M33 `max_iterations` off by one in the
   device loop ((h) the 0 and 1 arms, capped counts); M34 an
   explicit `chunksize` ignored under `"gpu"` ((o) chooser, batch
   spy); M35 the VRAM model without the preprocessing transient
   ((o) calibration, and the faked free VRAM where only the P and
   R terms decide). Added 2026-10-06 at the spec review, the
   device-only algebra, which the CPU's precompute tests never see:
   M36 the update composed on the wrong side, `W(dp)^-1 . W` ((k),
   the ONLY killer, as on the CPU: both arms share the fixed point,
   `test_hrebsd_engine.py:419-424`); M37 the steepest-descent
   rebuild with `gx` and `gy` swapped, M38 with `xi_x` and `xi_y`
   swapped in one column, M39 with a perspective term's sign
   flipped ((f) first-step band, (k)); M40 the per-reference
   constants built from the UNWEIGHTED block under `window` ((h)
   window arm, (f)); M41 the window weight applied once instead of
   squared ((h) window arm); M42 the shift reconstructed as an f64
   `K + mp` against an f32-rounded subtraction, or K never
   updated, or K = 0 in the mixed build ((f) first-step and
   residual bands; (h) the `(None, None)` arm with its DC-offset
   target); M43 the corner-norm maximum swallowing NaN ((i) the
   planted NaN corner); M44 the cell index converted to an integer
   before the fold ((i) the planted +-3e9, +-1e30 and FLT_MAX
   coordinates); M45 the matrix rebuilt as `1 + h` each iteration
   instead of carried ((k) float64, (f) float64 band); M46 the
   band-pass skip branch inverted, or the band-pass always applied
   ((h) `filter_cutoffs` arms); M47 `upsample_factor` hard-coded to
   16 ((d) the 2 and 1 arms, (h) the 8 arm); M48 `min_step`
   hard-coded ((h) `min_step` arm); M49 `border` ignored ((h)
   `border` arm); M50 the tail sub-batch run unpadded, at its own
   FFT batch count ((h) the last-sub-batch slot against alone, (d)
   sub-batch spy); M51 residents never evicted ((n) the F7
   residency spy and `set_limit` arm); M52 `pattern_index` wrong on
   padded slots, or in map order instead of fit order ((d)
   `SeedBatch` spy); M53 the seam called once per batch instead of
   per sub-batch ((d) spy count).
5. Fixer: every finding dispositioned in a dated table appended to
   this plan (the section 9/10 precedent); every surviving mutant
   killed by a strengthened test and re-verified by re-injection;
   decision-critical numbers re-measured (the V9(f) bands, the
   precision verdict, B invariance).
6. Gates, each recorded with its output in validation.md: the
   default suite under numpy (`-k hrebsd`, `-n 0` then `-n 4`)
   green; the gated suite through the overlay with
   `KIKUCHIPY_EXPECT_GPU=1` at `-n 0 --weekly`, 0 skipped (V9
   gate commands); the default path bitwise unchanged (the
   pre-Stage-D literal pins and `backend="cpu"` == no keyword);
   coverage 100 % of every touched `_hrebsd` module, default and
   gated runs combined; `uv run pre-commit run --files <explicit
   list>` (ruff, ruff-format; never `--all-files`, never `specs/`);
   the oldest-matrix recipe of ledger entry 87 (cupy absent there,
   so the gated classes skip at stage (a) and the default classes
   run on the oldest floor); the full suite `uv run pytest tests -n
   4`, and, as a RECORDED GATE from Stage E on (added 2026-10-06,
   spec review), the full suite again after every merge of develop
   into hrebsd-dic, so a develop-side rename of the three imported
   `_spherical._gpu` names fails at the merge (D21.2); every gated
   command on the pinned overlay of D21.15; signed commits (`git
   commit -s`) pushed to
   `origin/hrebsd-dic` together (failing tests ride with the
   implementation), NO PR; roadmap Stage E boxes ticked.
7. Performance record (D21.16) on the Si-indent data, machine idle
   and `nvidia-smi` clean before each timing, best of 3 after a
   warm-up: far256, patch C and the whole map at both seed
   precisions and both device precisions (the D21.16 matrix), each
   against the expectation for its own precision, with the E8
   device-wait fraction; the same-session 8-worker CPU on far256
   and patch C, and the go/no-go floor; recorded honestly whatever
   it measures.
8. Tutorial and docs, minimal (D21.17): the `hrebsd_dic` docstring;
   one CHANGELOG entry citing the spec folder and ending with the
   precedent's fork-only closing sentence (D21.17); one markdown
   cell in `hrebsd_si_indent.ipynb` with the measured whole-map GPU
   and CPU times, each number naming machine A and its device and
   seed precisions, and the call as a fenced code block -- no
   executed GPU
   cell, no stored output touched, nbval unaffected (checked by
   running nbval on that notebook once); `installation.rst`
   untouched.
9. Model assignment (the 2026-10-05 global rule, superseding the
   header of this file for Stage E): this spec and plan on Opus
   5.5 at effort xhigh with ultracode; failing tests,
   implementation, adversarial review, bug injection and fixes on
   Opus 5.5 at effort medium with ultracode, run as Workflow agents
   with `{model: 'opus', effort: 'medium'}` (the Agent tool cannot
   set effort). Fable is an ESCALATION only: a step moves to Fable
   only on repeated errors AND an inconsistency -- the same gate
   failing again after a fix round, reviewers or critics
   contradicting each other, or the spec and the code still
   disagreeing after a fix; only that step escalates, the reason is
   recorded in the ledger or the commit message, and the work drops
   back to Opus 5.5 afterwards.

### 11.1 Risks

1. **The precision surface needs an approval it may not get**
   (D21.4). RESOLVED 2026-10-06: approved (11.4 approval
   record); "mixed" is in force. Original mitigation: both precisions are built and banded; the
   float64 default stays in force until the D17 amendment is
   approved, and float64 still projects far above the CPU (270 us
   per pattern-iteration against 6.7 ms on 8 workers; the whole
   map about 440 s = 131 patterns/s, about 19x, with the complex128
   seed, and about 365 s = 158 patterns/s, about 23x, with
   complex64, an inference from the ledger 98 model, against the
   51x and 92x measured under mixed).
2. **Pinned CPU bands do not transfer**: mixed misses
   `INTENSITY_SCALE_GENERIC_TOL` and no device precision meets
   `UPDATE_RULE_TOL` as such (ledger 93); `WARP_REFIT_TOL_480`
   has a thin margin at large N. Mitigation: device-specific bands
   (V9(f), (j), (k)) and the twelve-case F1 for V2 asserts.
3. **Determinism is fragile** to launch layout and cuFFT plan
   choice (ledger 95). Mitigation: layout from geometry only,
   padding to a fixed B, B invariance measured before it is
   pinned (E4).
4. **Silent failure modes**: a device exception inside
   `fit_pattern`'s catch-all would become NaN rows; a NaN
   coordinate in the fold is undefined behaviour in CUDA C.
   Mitigation: D21.9.4 and the D21.6 flags, with V9(i) and (n)
   and mutants M8, M9 and M19.
5. **Unprototyped features** (`window`, `dead_band`, multi-grain
   maps, the failure flags; ledger 99) carry no spec-gate
   evidence. Mitigation: parity oracles V9(h) and (i) on each.
6. **The VRAM model under-counts** or WDDM misreports free memory
   (the Phase 12 5 MB temporaries lesson). Mitigation: separate
   calibration of each term, half-free headroom, the two-window
   halving.
7. **Contaminated timings**: the machine is shared with the HROSM
   session, and laptop clocks decay thermally (Phase 12's first
   idle baseline was 22 per cent soft). Mitigation: `nvidia-smi`
   before every timing, best of 3 rested, repeats recorded.
8. **NVRTC provenance unknown** (ledger 89): a wheel-only machine
   might fail at the first kernel compile. Mitigation: E6 before
   the stage-(c) message is pinned.
9. **Merge hygiene**: the gate imports develop-owned private names
   from `_spherical/_gpu.py`, and `ebsd.py` is shared. Mitigation:
   a rename fails at the module-scope import of `_hrebsd/_gpu.py`
   and at the identity pins, and the full suite now runs after
   every merge of develop (item 6, a recorded gate added at the
   spec review); edits confined to the fork-only method; the three
   spec-file insertion anchors listed above.
10. **Prototype optimism**: Phase 12's implementation landed below
    its projection (ledger 99). Mitigation: the performance record
    re-measures, and the go/no-go floor is the modest one of
    D21.16.
11. **Seed parity is measured, not constructed** (D21.5):
    complex128 equality could fail on some pattern. Mitigation:
    counts pinned on the fixtures, not an equality promise; the
    h-level bands of V9(f) are what parity rests on.
12. **Static batches hold slow points**: a 500-iteration rim point
    keeps its batch alive. Measured acceptable: an iteration with
    1 active pattern of 128 costs 0.197 ms against 4.11 with all
    active (ledger 91), and patch C ran in 2.92 s (ledger 97).
13. **Library drift moves device bits** (added 2026-10-06, spec
    review): an unpinned overlay can resolve newer cuFFT, cuBLAS or
    NVRTC wheels and move device-side pins. Mitigation: every gate
    command pins the D21.15 versions; a bump re-measures the
    device-side pins as a dated entry.
14. **The frozen design is not the prototype** (added 2026-10-06,
    spec review): the adaptive-K reconstruction, the NaN-propagating
    corner norm, the fold before the integer conversion, the carried
    matrix, the windowed algebra, P-padded sub-batches, the `R_MAX`
    residency and the v1 read route all differ from what the
    prototype measured (ledger 99). Mitigation: each differs only at
    rounding level on the measured inputs or on inputs the prototype
    never met; every band and every timing is re-measured on the
    implementation, and E8 is required.

### 11.2 Open questions (each with the conservative default in force and the measurement that resolves it)

E1. **Device precision default** (D21.4). RESOLVED 2026-10-06
    (11.4 approval record): `"mixed"` is in force. Before the
    approval: in force `"float64"`
    (the whole map projected at about 440 s = 131 patterns/s, about
    19x, with the complex128 seed; ledger 98, an inference).
    Proposed: `"mixed"` (measured 166.5 s = 347 patterns/s, 51x,
    with the prototype's dedicated reader; ledger 97). Resolves:
    Johan's approval of the D17 amendment at this gate (11.4 item
    3), then the implementation-gate re-measurement of the V9(f)
    bands under both, and the whole-map record at both device
    precisions (D21.16).
E2. **Seed precision default** (D21.5). In force: `"complex128"`
    (bitwise with the CPU seed on 256 of 256; ledger 94). Resolves:
    the implementation's whole-map complex64-against-complex128
    record at the default device precision (prototype, mixed: 7
    points +-1 iteration, 12 above 1e-4 px, max 6.0e-4 px, about
    1.8x end to end); the default changes only if Johan approves it
    on that record (11.4 items 2 and 4, which also settle whether
    the opt-in is public, E13).
E3. **Default batch size and cap** (D21.10.3). In force: the largest
    power of two in [1, 64] whose whole VRAM model fits half of free
    VRAM. Resolves: an end-to-end sweep at B in {8, 16, 32, 64, 128}
    on far256 and patch C with pool high-water marks, and the three
    calibrated model terms.
E4. **Batch-size invariance across B** (D21.7.3). In force: bitwise
    run to run at a fixed B; P = min(32, B) with every sub-batch
    padded to P, so every B >= 32 shares the FFT shapes. Resolves: B
    in {8, 32, 40, default} on F1 to F4; pinned bitwise if bitwise,
    else at the measured tolerance with the deviation recorded. The
    alternative of a constant P = 32 at every B is 11.4 item 23.
E5. **Multi-grain batches and the residency bound** (D21.9.3,
    D21.10.2). In force: grain-pure batches, at most `R_MAX = 2`
    residents, least recently used evicted. Resolves: F7 and a
    synthetic many-grain map (tens of grains of tens of points)
    timed against a one-grain map of the same size; multi-grain
    batches only if per-grain padding costs more than about 20 per
    cent of the run.
E6. **NVRTC provenance and the stage-(c) remedy** (D21.2). In force:
    the message names the verified overlay's wheel set; the spec
    review found `nvidia-cuda-nvrtc-cu12` 12.9.86 resolved
    transitively in the overlay (ledger 89). Resolves: record the DLL
    that supplied NVRTC in the overlay (the process module list),
    and a compile check with no CUDA Toolkit on PATH;
    `nvidia-cuda-nvrtc-cu12` joins the message if needed.
E7. **The go/no-go floor** (D21.16). In force: ADOPTED as a
    recorded default (11.4 item 14): GPU at least the same-session
    idle 8-worker CPU on far256, a local record, never a test or CI
    gate. Resolves: the far256 measurement at the implementation
    gate; below the floor, Stage E records a negative result and the
    review gate decides whether the backend ships.
E8. **A dedicated reader or pinned double-buffered upload**
    (D21.11). In force: none; the dask gather (a RECORDED DEFAULT
    since the spec review, no longer frozen). Resolves: a REQUIRED
    implementation-gate measurement -- the device-wait fraction of
    the v1 route, whole map, lazy h5oina, default B; a follow-up
    opens only if the device waits on data for more than about 10
    per cent of wall time. Spec-review probe: the v1 gather reads
    1500-2100 patterns/s at B = 64 (ledger 96 addendum).
E9. **Tutorial scope** (D21.17). Moved to 11.4 item 18 at the spec
    review (a scope choice that no measurement resolves). Trigger to
    reopen: a request for an executed GPU section, which would need a
    separate, output-stored notebook (the docs builder and the nbval
    runners have no GPU).
E10. **`installation.rst`** (D21.17). Moved to 11.4 item 17 at the
     spec review. Trigger to reopen: a develop-side edit of the cupy
     bullet of `doc/user/installation.rst` arriving in a merge, which
     makes a one-line HREBSD addition free of a fresh conflict.
E11. **`--use_fast_math` against explicit reciprocals** (D21.4). In
     force: no `--use_fast_math`. Resolves: the kernel timing at the
     implementation gate (prototype: the same speed either way).
E12. **Real-data per-point parity scope** (D21.8, V9(q)). In force:
     far256 and patch C per point; the whole map by aggregate
     counts only (no per-point CPU map exists; a re-run is 2.34 h).
     Resolves: the far256 and patch C per-point records; a
     disagreement beyond the V9(f) bands there, or a whole-map
     aggregate count differing from ledger 82, triggers the 2.34 h
     CPU re-run for a map-level per-point record (11.4 item 15).
E13. **Public precision knobs** (D21.1). In force: engine-only
     `device_precision` and `seed_precision`, not public in v1, so
     the complex64 opt-in is reachable through the private engine
     entry only. Resolves: the implementation's measured whole-map
     end-to-end gains -- complex64 over complex128, and mixed over
     float64 -- with a gain above about 1.5x on the record making a
     public keyword (or a `gpu_options` mapping) and a second D15.4
     amendment the proposal; Johan decides at this gate whether to
     take that now (11.4 items 2 and 4).

### 11.3 Follow-ups

- **Stage F -- Fourier-Mellin initial guess, both backends
  (PLANNED after Stage E, Johan decision 1 of 2026-10-06; not
  commissioned).** Ernould 2020's cascaded Fourier-Mellin plus
  cross-correlation seed (plan open question 5; the D5 deferral),
  specified for BOTH backends at once as ONE xp-agnostic
  (numpy/cupy) code path, with the CPU as parity oracle and CI
  coverage under numpy, so nothing needs porting later. It plugs in
  behind the D21.5 seam without touching the batched IC-GN. Design
  notes recorded now: (i) it REUSES the target FFT the translation
  seed already computes (`seed_spectra`; the magnitude is
  translation invariant); (ii) it adds a per-reference log-polar
  magnitude spectrum, precomputed once per reference as an
  EXTENSION of `SeedState`; (iii) it de-rotates through the existing
  warp kernel -- `SeedContext.kernels.gather` on
  `SeedBatch.coefficients`, both carried by the seam since the spec
  review -- then the translation phase XC runs on the de-rotated
  target, and the seam returns the full 8-parameter `h0` (rotation
  plus translation); (iv) it is GATED, because on the GPU the seed
  is already about 62 per cent of device time under mixed (ledger
  98): the recommended gate is pre-fit, on the input `CrystalMap`'s
  misorientation about the detector normal against the grain
  reference, at a threshold of about 1.5-2 deg (the phase-XC capture
  range is 2.0 deg at the default band-pass, D5); the runner puts
  the per-slot gate flag into `SeedBatch.extras` from host data
  keyed by `SeedBatch.pattern_index`, so the gate needs no signature
  change; a post-fit retry of non-converged points is cheap on the
  GPU and may serve as a safety net -- it is a SECOND RUNNER PASS
  over the non-converged subset, a runner change, not a seam or
  engine change; (v) its parity is on CONVERGED results within a
  band, not bitwise seeds (log-polar interpolation is not bitwise
  portable); (vi) rotation only, no scale, is the conservative
  default; (vii) failure semantics (added at the spec review): an
  FM estimate that fails returns the translation row the wrapper
  started from, never a NaN row, so the seam keeps its "non-finite
  means seed failure" meaning (D21.5); (viii) how the CPU joins
  (added at the spec review): `backend="cpu"` never calls the seam
  today (`_fit_chunk` -> `fit_pattern` -> `initial_guess`,
  `_engine.py:697-702`); in Stage F only the FM-GATED points route
  through the numpy seam on the CPU (batched, then fitted with
  `h0=row` through the Stage D plumbing), so the ungated default
  path stays bitwise unchanged and the pre-Stage-D literal pins
  stand as its gate; the numpy seam's measured equality with
  `initial_guess` (V9(d), `NUMPY_SEED_EQUAL_COUNT` on F1, F3s, F4,
  F6 and the Ni map) is what any wider routing would have to keep.
  It covers the regime D21.12 leaves to the CPU cascade (rotation
  about the detector normal past ~2 deg), per point and without
  propagation risk. Rotations about in-plane axes mainly translate
  the pattern, which the translation XC already catches; the
  remaining projective distortion stays IC-GN's, as now.
- The Stage D residual-acceptance gate and the PASS1_CAP revisit
  (roadmap "Stage D follow-ups") are unaffected and stay
  uncommissioned.
- Device speed levers once the seed stops dominating (D21.18): an
  rfft2 band-pass (not bitwise with the CPU), a box-mean K0, pinned
  double-buffered upload, CUDA graphs; a dedicated reader (E8);
  multi-grain batches (E5).
- CPU-side notes from the spec gate, each its own decision because
  each touches the CPU default path (ledger 90, 99): OpenBLAS
  oversubscription (up to about 1.3-1.5x on 8 workers) and fusing
  the numpy warp coordinates (31 per cent of a CPU iteration) into
  the numba kernel.

### 11.4 Recorded defaults for Johan's approval

Every default below is one a reasonable person might choose
differently; each stands as written until approved or changed at
this gate. Items 21 to 25 were added at the spec review
(2026-10-06, 11.5).

**APPROVAL RECORD (2026-10-06, about 20:00).** Johan waived this
gate for the night ("going to bed, just continue automatically and
accept recommendations") and then extended the waiver to Stage F
("continue with the merlin guess implementation after the gpu
implementation is done"). Under that waiver every item below is
APPROVED AS WRITTEN, the recommendation taken wherever an item
offers an alternative:
- item 1 (Johan decision 2): approved; D21.12 is final.
- items 2 and 4: complex128 is the device seed default and both
  precision knobs stay engine-only in v1; E13's measured rule
  (public when the end-to-end gain is at least about 1.5x) is
  applied at the implementation gate and, if it fires, adds a
  public `seed_precision` through its own dated D15.4 amendment.
- item 3: the D17 amendment is APPROVED; `"mixed"` is the device
  default in force from this date (D21.4; E1 resolved), and
  `"float64"` stays built as the parity and debug mode.
- item 19: the mission.md and tech-stack.md amendments are approved.
- item 20: carried to Stage F's own spec gate (same waiver).
- item 21: the seam extension (`SeedContext`, `SeedBatch`) is
  confirmed.
- every other item, 22 to 25 included: approved as written.
The "pending" markers in D17, D21, mission.md, tech-stack.md and
roadmap.md are resolved by this record and point to it.

1. `backend="gpu"` with `seed_from_neighbors=True` raises
   `NotImplementedError`, pointing to `backend="cpu"` and the
   planned Fourier-Mellin seed, before the gate and any work
   (D21.12; Johan decision 2, pending his final approval).
2. Device seed precision complex128 by default, complex64 an
   explicit opt-in (D21.5; the decision 3 lean). INTERACTION with
   item 4, stated at the spec review: with item 4 as written the
   complex64 opt-in -- about 1.8x end to end, the largest measured
   lever -- is reachable ONLY through the private
   `kikuchipy.indexing._hrebsd._engine.run_hrebsd_dic`, not through
   `EBSD.hrebsd_dic`. The paired alternative, offered as ONE
   decision with item 4: a public keyword-only `seed_precision`
   (default `"complex128"`) on `EBSD.hrebsd_dic`, with its own dated
   D15.4 amendment and `FROZEN_SIGNATURE` edit.
3. The D17 amendment: `"mixed"` device precision as the default
   once approved, `"float64"` in force until then; both built
   (D21.4). It sanctions f32 per-pixel arithmetic, f32 per-thread
   partial sums and f32 STORAGE of the reference vector, the
   coordinate planes and the gradient columns, reversing for
   `backend="gpu"` only the f64-by-design clause of D17's
   narrowed-scope paragraph. Under float64 the whole map projects
   at about 19x (complex128 seed) against 51x measured under mixed.
4. The precision knobs are engine-only, not public, in v1 (D21.1;
   E13) -- see item 2 for what that costs.
5. The `backend` slot: after `navigation_mask`, immediately before
   `chunksize`, the `spherical_indexing` slot (D21.1).
6. The full per-pattern pipeline on the device (band-pass, spline,
   seed, IC-GN, final criterion) rather than a hybrid with host
   preprocessing and seed (D21.3).
7. The gate imports the Phase 12 shim, version floor and lock
   (sharing one device lock across both backends) and re-implements
   the three stages with HREBSD wording and its own cache, rather
   than wrapping the spherical gate or copying the shim (D21.2).
8. Stage (c) probes cuFFT (both precisions), cuBLAS and an NVRTC
   compile; the message prefix "EBSD.hrebsd_dic with backend='gpu'
   requires" (D21.2).
9. No `--use_fast_math`; explicit reciprocal constants; `--fmad`
   at the compiler default (D21.4; E11).
10. Static batches with mask-and-retire; no slot refill, no
    compaction (D21.6).
11. Grain-pure device batches; at most `R_MAX = 2` references
    resident, the least recently used evicted before the next
    grain's upload (D21.9.3; E5).
12. `chunksize` is the device batch size; the default B is the
    largest power of two in [1, 64] whose whole VRAM model (per-slot,
    preprocessing and resident terms) fits half of free VRAM
    (D21.10; E3); under `"gpu"` a `chunksize` below 1 raises and B
    is never clamped to the number of fitted points (D21.10.1).
13. The Phase 12 topology (dask `blockwise`, threaded scheduler,
    one device section per chunk under the lock); no feeder or
    reader thread, no pinned double-buffered upload in v1 -- a
    RECORDED DEFAULT since the spec review (it was frozen in the
    draft on evidence measured with a dedicated reader), with E8 a
    required implementation-gate measurement (D21.9, D21.11).
14. A local go/no-go floor (GPU at least the same-session idle
    8-worker CPU on far256), a ledger record and never a test or
    CI gate (D21.16; E7).
15. Real-data parity per point on far256 and patch C only; the
    whole map by aggregate counts, no 2.34 h CPU re-run unless the
    E12 trigger fires (D21.8; E12).
16. A new drift tripwire anchored on dated literals of both
    backends on F1 (D21.8(h)).
17. `doc/user/installation.rst` left untouched (D21.17; E10).
18. The tutorial: one markdown cell in `hrebsd_si_indent.ipynb`, no
    executed GPU cell (D21.17; E9).
19. The constitution amendments (mission.md, tech-stack.md, HREBSD
    sections, dated 2026-10-06), including the per-backend
    determinism wording and the float-discipline scoping.
20. For Stage F, recorded now and decided there: the pre-fit
    misorientation gate at about 1.5-2 deg, the post-fit retry as a
    safety net, rotation-only, and banded parity on converged
    results (11.3).
21. The seed-seam EXTENSION (D21.5): `SeedContext` (xp, fft, the
    kernel namespace) and a per-sub-batch `SeedBatch` (targets,
    coefficients, `pattern_index`, `extras`) carry Johan's three
    inputs plus what his own Stage F notes need -- which map point
    each slot is (for the pre-fit gate) and the coefficients and
    warp kernel (for de-rotation) -- so Stage F slots in without
    breaking a frozen signature. Alternative: the draft's positional
    signatures, exactly the three listed inputs, accepting that
    Stage F changes them.
22. The seam's NaN-row semantics: a non-finite row is a per-pattern
    seed FAILURE (the CPU's `initial_guess`-raises outcome), the
    opposite of `_fit_chunk`'s "fall back to the phase-XC seed";
    Stage F's wrapper falls back to the translation row it wraps
    (D21.5, 11.3 (vii)).
23. Preprocessing sub-batches P = min(32, B), every sub-batch padded
    to P (D21.7.3; E4). Alternative: P = 32 at every B, which makes
    B invariance hold by construction at every B, at the cost of
    preprocessing 32 slots per sub-batch even at B = 1 and a P term
    (about 0.93 GB at 512x622) that the out-of-memory halving cannot
    shrink.
24. The uv overlay pinned to the D21.15 versions in every gate
    command, and a library or driver bump re-measuring the
    device-side pins as a dated entry (D21.7.2, D21.15).
25. Under `"gpu"`, a `coefficient_dtype` other than float32 raises
    `ValueError` (D21.1); the numpy kernel twin covers the float64
    build only, and the mixed build is covered by the gated suite
    alone (D21.14.3).

### 11.5 Spec-review disposition table (2026-10-06)

34 findings from the two adversarial critics of the Stage E draft
(N-E: 15, C-E: 19; 15 major, 19 minor, no blocker). Every finding
was re-verified against the code, the surviving prototype
scratchpad (`kernels.cu`, `gpu_icgn.py`, `gpu_fullmap.py`,
`gpu_e2e.py`, `io_ceiling.py`, `gpu_accuracy.json`,
`gpu_det_seeds.json`) or a fresh measurement before disposition:
the CPU's constant-target failure sites and the
`upsample_factor` edges (run on the CPU, 2026-10-06), the v1
fancy-index gather rate on the Si-indent file (ledger 96 addendum)
and the overlay's resolved wheel versions (ledger 89). All 34 hold
and are APPLIED; none rejected. Where the applied fix differs from
the critic's proposal the row says so. None contradicts Johan's
decisions of 2026-10-06: N-E-SEAM-1 and C-E-MAJ-1 EXTEND the seam
around his three inputs and are listed for his confirmation (11.4
item 21). Overlapping findings share a resolution and keep one row
each.

| # | Finding (short) | Disposition |
|---|---|---|
| N-E-SEAM-1 (major) | frozen seam cannot carry the Stage F gate's per-pattern identity or de-rotation's coefficients and kernels; h0 frame unstated | accepted: D21.5 rewritten with `SeedContext` and per-sub-batch `SeedBatch` (`pattern_index` with -1 padding, `coefficients`, `extras`), clause (iv) extends all three classes, the h0 frame stated (D1.3, binned px, reference->target); V9(d) `SeedBatch` spy; 11.3 notes (iii), (iv) and the post-fit retry as a runner pass; 11.4 item 21 |
| N-E-DET-1 (major) | P <= 32 sub-batches contradict "one FFT shape"; seam "once per batch" spy inconsistent | accepted: D21.7.3 P = min(32, B) fixed per run, every sub-batch padded to P; the seam called once per sub-batch (D21.5(ii)), V9(d) spy count; V9(h) last-sub-batch arm at B = 40; V9(m) B = 40; E4; 11.4 item 23 (P = 32 everywhere offered as the alternative); M50, M53 |
| N-E-VRAM-1 (major) | chooser omits P and R terms; no residency bound; OOM loop cannot free residents | accepted: D21.9.3 `R_MAX = 2` with LRU eviction; D21.10.2 model `B*g + P*p + R_MAX*r` with g the batch-lifetime set; D21.10.3 chooser = largest power of two whose whole model fits half free; D21.10.4 rewritten; V9(n) F7 residency and `set_limit` arms, V9(o) faked-VRAM arm; 11.4 items 11, 12; M51 |
| N-E-PERF-1 (major) | I/O and throughput evidence measured with a dedicated reader, contiguous reads, B = 128, device-only rows | accepted: ledger 96 limit and addendum (v1 gather probed: 1527-1640 first read, 1881-2142 re-read, contiguous 1504-1651); ledger 97 rows relabelled device-only with read-inclusive figures (about 342 / 691 pat/s, 111x / 154x); the whole-map setup recorded; ledger 98 limit; D21.11 downgraded to a recorded default; D21.16 expectation qualified |
| N-E-PAR-1 (major) | "the SAME outcome" overclaims on degenerate inputs; NaN-swallowing corner norm | accepted, CPU measured: D21.6.2 narrows the contract to exact constants and non-finite values; F5 uses an integer constant at `(None, None)` (seed fails at exact 0) plus a band-passed arm asserting `converged=False` only; D21.6.1 freezes a NaN-propagating corner norm; V9(i) arms; M43 |
| N-E-ORC-1 (major) | V9(e) asserts h bands and equal counts on fits that may not converge | accepted: V9(e) rewritten -- rows inside the capture range, h band on both-converged points, measured count budgets, a pinned both-converged minimum, the NaN row exact; added a one-iteration arm and a discriminating count so an ignored seam still fails |
| N-E-DET-2 (major) | determinism pledge and overlay omit library versions | accepted: D16 note, D21.7.2, D21.15 and tech-stack (f) scoped to CuPy and CUDA-library versions; versions resolved and recorded (ledger 89: cupy 14.2.0, cufft 11.4.1.4, cublas 12.9.2.10, cusolver 11.7.5.82, cusparse 12.5.10.65, nvjitlink 12.9.86, nvrtc 12.9.86 transitive, driver 595.71) and pinned in every V9 gate command; bump rule; risk 13; 11.4 item 24 |
| N-E-D17-1 (major) | D17 amendment hides f32 storage of the planes D17 reserved as f64; mixed SD/host-constant inconsistency; pure-f32 comparison partial | accepted: the amendment, D21.4 and tech-stack (b) name the f32-stored planes and the reversal for `"gpu"` only; the f32 SD against f64 host Hessian and constants stated; D21.4 records pure f32 in full (final h within 1.43x of mixed, below it on 480 seed 0) |
| N-E-MUT-1 (major) | mutation list misses the device-only update algebra | accepted: M36 (wrong-side composition, killed by (k) only), M37-M39 (SD rebuild), M40-M41 (window), M42 (K shift), M45 (carried matrix); (k) names its sole-killer role |
| N-E-K-1 (minor) | constant K frozen where adaptive K was measured; reference spectrum said host-built | accepted: D21.4 freezes the measured adaptive K, rounded to pixel precision, every moment reconstructed against the shift actually subtracted; D21.5 says the spectrum is built in the batch's namespace (cuFFT, as measured) |
| N-E-WIN-1 (minor) | window applied twice on the CPU; prototype algebra valid for `window=None` only | accepted: D21.6.5 writes out the windowed algebra (w^2 in gradient and criterion, unweighted ZMN, weighted constants); weight plane in r (D21.10.2); M40, M41 |
| N-E-SEED-1 (minor) | `upsample_factor == 1` and `< 1` branches unspecified | accepted, CPU measured: D21.5 freezes the coarse branch at 1 and the every-pattern failure below 1 (no new error path); V9(d) arms at 16, 2, 1 and 0; M47 |
| N-E-MTP-1 (minor) | device literals "measured at the failing-tests gate" | accepted: D21.8 intro and (h), V9 intro and (l), plan 11 item 1 split CPU-side (failing-tests gate) from device-side (implementation gate) |
| N-E-COORD-1 (minor) | int conversion before the fold; "matrix rebuilt exactly as the CPU" inaccurate | accepted: D21.6.3 folds in floating point before a range-guarded conversion, V9(i) +-3e9, +-1e30, FLT_MAX arm, M44; (b) went further than the proposed rewording: D21.6.4 freezes the carried matrix as on the CPU, M45 |
| N-E-STF-1 (minor) | how `backend="cpu"` reaches the seam in Stage F; numpy seed equality only on F1-F3 | accepted: 11.3 (viii); V9(d) numpy arm on F1, F3s, F4, F6 and the Ni map as `NUMPY_SEED_EQUAL_COUNT`, expected "all equal" |
| C-E-MAJ-1 (major) | frozen seam cannot carry the recorded Stage F design | accepted, same resolution as N-E-SEAM-1; raised to Johan as an additive extension (11.4 item 21) |
| C-E-MAJ-2 (major) | no stable internal names for failing tests; xp-agnosticism overstated | accepted: D21.9.5 name table (`_GpuSession`, `_make_session`, `_vram_model_bytes`, `_default_batch_size`, `_run_chunks_gpu`, `_launch_layout`, `KernelNamespace` with `out_of_memory_error` and four entry points) and the import direction; D21.14.3 restated (orchestration and seed single-source, numpy twin float64 only, no BLAS reductions); V9(f) per-kernel A/B at float64 |
| C-E-MAJ-3 (major) | D21.11 rests on a dedicated-reader setup it rejects; dask rate on contiguous slices | accepted, same resolution as N-E-PERF-1; E8 made a required measurement and probed in advance (ledger 96 addendum); D21.16 expectation labelled |
| C-E-MAJ-4 (major) | headline speed-ups are mixed; float64 (the default in force) figure absent | accepted: every headline tagged "mixed"; float64 projection (about 440 s = 131 pat/s, 19x; 365 s = 158 pat/s, 23x) added to the D21 intro, D21.16, ledger 98, roadmap, risk 1, E1 and 11.4 item 3; D21.16 records the map at both device precisions |
| C-E-MAJ-5 (major) | no parity oracle for `filter_cutoffs`, `upsample_factor`, `min_step`, `border` | accepted: V9(h) arms for `(None, None)` (with a DC-offset target), `(0.05, 0.4)`, `upsample_factor` 8, `min_step` 1e-2, `border` 0.1; D21.6.6; M46-M49 |
| C-E-MAJ-6 (major) | complex64 opt-in unreachable through the public API; interaction hidden | accepted: stated in D21.1, D21.5, 11.4 items 2 and 4 and E13, with the public `seed_precision` keyword offered as the paired alternative; listed for Johan |
| C-E-MIN-1 (minor) | audit tuple has no `typing`/`__future__`; likely `_engine` import cycle | accepted: D18 amendment and D21.13 record the no-typing/`__future__` convention (verified: no `_hrebsd` module uses either); D21.9.5 freezes the import direction; V9(p) AST check |
| C-E-MIN-2 (minor) | V9(e) bands stricter than D21.8 allows | accepted, same resolution as N-E-ORC-1 |
| C-E-MIN-3 (minor) | weak killers for M1, M2, M15; no K-shift mutant | accepted, verified (the f64-atomic mixed variant identical 4 of 4 at 512x622): M1 source pin, M2 launch-dimension spy, M15 dimmed-target arm or reviewed-equivalent with the DC-bin argument, M42 (K = 0 included) |
| C-E-MIN-4 (minor) | seam NaN-row meaning opposite to `_fit_chunk`; E4 alternative not offered | accepted: D21.5 NaN-ROW SEMANTICS with Stage F's fallback rule (11.3 (vii)); 11.4 items 22 and 23 |
| C-E-MIN-5 (minor) | seam output contract checked only under numpy | accepted: gated device-contract arm in V9(d) (`cupy.ndarray`, float64, C-contiguous, `(P, 8)`; spectra dtype) in `TestGatedSeedParity` |
| C-E-MIN-6 (minor) | docstring Raises incomplete, no docstring test, message literal unfrozen, CHANGELOG wording | accepted: D21.17 Raises lists all three, docstring test in `TestBackendSwitch` (V9(a)), the D21.12 message frozen literally and asserted in full (V9(b)), the CHANGELOG closing sentence required, numbers qualified by machine and precision |
| C-E-MIN-7 (minor) | touched-files list omits spec files; "post-merge gate" undefined | accepted: the three spec files and their anchors listed; item 6 adds the full suite after every develop merge as a recorded gate; D21.2 and risk 9 reworded (identity pins kept as a cheap second signal) |
| C-E-MIN-8 (minor) | tech-stack (b) says "every reduction" stays f64 | accepted: (b) reworded to D21.4's "every block and cross-block reduction", with the f32-stored planes named |
| C-E-MIN-9 (minor) | `chunksize < 1`, the n_fit clamp, `coefficient_dtype` and check order unspecified under `"gpu"` | accepted: D21.10.1 (`ValueError` below 1, no n_fit clamp, padded batch), D21.1 (`coefficient_dtype` check, the backend checks after the existing ones and before `resolve_reference`); V9(a), (o); 11.4 items 12 and 25 |
| C-E-MIN-10 (minor) | no oracle that lazy input stays lazy | accepted: V9(m) laziness oracle (dask callback: no computed block above B patterns, no whole-map array at the runner), `TestLazinessNumpySession` |
| C-E-MIN-11 (minor) | E7, E9, E10, E13 resolve by "Johan" alone | accepted: E7 resolves by the far256 measurement, E13 by the measured end-to-end gains, E9 and E10 moved to 11.4 with reopen triggers; E12 given a trigger too |
| C-E-MIN-12 (minor) | Stage F notes silent on how the CPU joins the seam | accepted, same resolution as N-E-STF-1 (11.3 (viii)) |
| C-E-MIN-13 (minor) | default-suite numpy twins on full F2/F3 too slow for CI | accepted: default suite on F1, F3s (8 patterns), F4, F5, F6, F7 and the Ni map; F2 and F3 gated only; wall time recorded at the failing-tests gate (D21.14.3) |

### 11.6 Implementation-review disposition table (2026-10-07)

Fixer, plan 11 item 5, on the two reviews (fidelity and theory;
conventions and integration) and the first half of bug injection
(M1 to M27, validation.md V9 ledger 107). Evidence in ledger 108. The
second injection half (M28 to M53) had not reported when this table
was written.

| id | finding | disposition |
|---|---|---|
| RF-E1 (major) | `--fmad=false` on the spline prefilter departs from D21.4 on a false premise (the kernel is not scipy's order of operations; the flag buys no parity) | accepted: `_cuda._SPLINE_OPTIONS` deleted, the prefilter compiles with `_KERNEL_OPTIONS` like every kernel; the comment above `_SPLINE_POLE`, the `_KERNEL_OPTIONS` comment, the module docstring and the `_gpu._prepare_sub_batch` docstring rewritten (algebraic mirror-mode recursion, equal to scipy up to f64 rounding); source pin `TestCudaSourcePins::test_fmad_stays_at_the_nvrtc_default`; gated `--weekly` re-run green (ledger 108) |
| RF-E2 (minor) | the float64 build multiplies the B-spline weights by `1/6`, where the CPU divides | accepted: `basis<T>` multiplies by the reciprocal under `HREBSD_MIXED` only and divides by 6 in the float64 build; float64 device pins re-measured (ledger 108) |
| RC-R1 (major) | the E5 trigger fired (2.0x to 2.64x on maps of 20-point grains) and nothing was done | accepted in part, DATED DEVIATION of D21.7.3 (2026-10-07): `_gpu._fit_batch` skips every sub-batch of padded slots only (no preprocessing, no seed seam call, no lockstep slots: the lockstep runs over the smallest multiple of P covering the last real slot). Results bitwise unchanged (old and new runners compared bitwise on two maps at both precisions; the B-invariance pin holds). E5 re-measured: 512x622, 8 grains x 20 points against 1 grain x 160, default B: 2.56x to 1.91x (mixed), 2.08x to 1.56x (float64); 60 px unchanged (about 2x, per-grain overhead). The residual cost is above the 20 % rule: multi-grain batches or a grain-size-driven B (option (b)) change D21.9.3 or D21.10.3 and are left to Johan. Tests re-pinned: `test_a_small_map_runs_one_padded_batch` (one seam call), the gated seam spy on F1 tiled to 40 points (`GPU_SEED_EQUAL_COUNT['F1 runner']` 40), the V9(o) g/p calibration on one-batch F3 prefixes (`f3_prefix_batch`, marks identical to ledger 105) |
| RC-R2 (major) | combined coverage below 100 %; three pragmas not on import guards | accepted: `TestRunnerEdgesNumpy` (13 default tests: closed-device submit, `packed` refusal, failing plan-cache clear, straddling batch, uncopyable cached gate failure, session B < 1 and bad namespace, `_device_description('numpy')`, unreadable free VRAM in the floor message, empty fit list, runner `chunksize < 1`, runner-chosen B, a non-dtype `coefficient_dtype`); gated `test_reduce_solve_update_takes_a_host_lockstep` and `test_the_spline_prefilter_leaves_a_length_one_axis`; the three pragmas removed (lines now covered). Combined figure in ledger 108 |
| RC-R3 (minor) | `--fmad=false` undocumented and unpinned | accepted, same resolution as RF-E1 (option removed, pin added, comments corrected) |
| RC-R4 (minor) | the information message prints `ceil(n / B)` chunks under `"gpu"` | accepted: `get_info_message` gains `n_chunks` (CPU output unchanged); the GPU branch passes `len(_gpu._batch_chunks(state_of_point, B))`; test on F5 (two grains, B = 8: 2 chunks, not 1) |
| RC-R5 (minor) | the D21.15 overlay is not hermetic (NVRTC and headers from the Toolkit and a foreign CUDA_PATH) | rejected here, FOR JOHAN: adding `nvidia-cuda-runtime-cu12` and `nvidia-cuda-nvrtc-cu12` to the gate commands edits frozen spec text (D21.15, V9) and the fixer's GPU runs are bound to the recorded overlay; the wheel-only venv run of ledger 105 (iii) already shows all 457 gated tests pass under NVRTC 12.9 without the Toolkit. Recommended: adopt the seven-wheel overlay as a dated D21.15 amendment and use it for every later device pin |
| RC-R6 (minor) | the public Notes quote "about 1e-6" px, below the pinned band, on synthetic data only | accepted: "a few 1e-6 binned pixels of corner displacement on converged points of synthetic test patterns"; revisit after the V9(q) real-data parity |
| RC-R7 (minor) | oldest-matrix recipe, `-k hrebsd -n 0`, full suite and combined coverage not run | accepted: all run and recorded in ledger 108. DATED DEVIATION (2026-10-07): `-k hrebsd` at `-n 2` in place of plan 11 item 6's `-n 4`, because the shared machine is memory constrained and xdist workers died with `MemoryError` at `-n 4`; the full suite also at `-n 2` |
| M15 (survivor) | target crops not zero-mean unit-norm in `seed_spectra` | KILLED: `TestSeedSeamContract::test_the_spectra_are_those_of_the_zmn_crops` and `TestGatedSeedParity::test_the_spectra_are_those_of_the_zmn_crops` (spectra against host `fft2` of the CPU-ZMN crop, offset and scaled F4, both seed precisions); re-injected, max difference 2.5e4 against 5.4e-13 and 5.4e-5 |
| M20c (survivor) | window (b) does not free the pools before the halved retry | KILLED: `test_real_pool_limit_recovers_without_leak` now reads `pool.total_bytes()` as each halved session is built (`GPU_REBUILD_POOL_BYTES` = 0); re-injected, 10,802,688 B (F1) and 1,093,632 B (F7) |
| M20e (survivor) | `_GpuSession.close` does not free the default pool | KILLED: the same test asserts `pool.total_bytes()` after the run within `GPU_LEAK_RESIDUE_BYTES` = 0; re-injected, 89,195,520 B (F1) and 1,630,208 B (F7) |
| M27a (survivor) | `/ 6.0` to `* (1.0 / 6.0)` in the CPU B-spline weights | KILLED: `TestCpuHelpersBitwise::test_the_bicubic_evaluation_is_the_frozen_formula` (`_bicubic_evaluate` and `evaluate` against an independent scalar transcription, bitwise) |
| M27b (survivor) | `/ norm` to `* (1.0 / norm)` in `zero_mean_normalize` | KILLED: `TestCpuHelpersBitwise::test_the_zero_mean_normalisation_is_the_frozen_formula` (bitwise against `centred / norm`) |
| M9 (equivalent on review) | zero-norm guard dropped | accepted as equivalent on the injector's argument (ledger 107 (ii): 0/0 is non-finite and fails the slot through the same branch, flags and NaN `norm_dp`); guards kept |
