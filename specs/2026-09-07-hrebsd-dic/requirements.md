# HREBSD-DIC -- `hrebsd-dic`: requirements

Branch `hrebsd-dic` off `develop` (long-lived feature branch; see the
branch policy in plan.md section 1: the branch is pushed to origin
but NEVER merged into `develop`, and no PR into `develop` is opened).
Spec folder `specs/2026-09-07-hrebsd-dic/`, one folder for all three
build stages (user decision 1, 2026-09-07). The feature: homography-
based high-angular-resolution EBSD by inverse-compositional
Gauss-Newton digital image correlation (Ernould et al., Acta Mater.
191 (2020) 131-148; AIEP 223 (2022) Ch. 2), plus the full analysis
chain -- strain/rotation/stress maps, HR-KAM, PC-shift analysis,
scalar GND density -- which exists in NO upstream source and is
written new here from the literature.

**Drafting caveat (D19):** nothing was executed while drafting this
spec (spec-stage hard constraint: read-only analysis plus file
writes under `specs/`). Every tolerance and several defaults below
are marked `MEASURED-THEN-PINNED` (MTP) with the measuring test
named; they are filled at the failing-tests/implementation gates of
the stage that owns them, per the tech-stack assertion convention.
Numbers that ARE quoted come from line-verified source reading of
EMsoftOO/OpenXY/kikuchipy (cited file:line) or from the published
literature (cited). EMsoftOO references are at `develop` HEAD
c127868; OpenXY at the local clone `C:/Users/westraadt.1/Repos/
OpenXY`; kikuchipy line references are on this fork's `develop`
(0.14.dev0 + spherical port).

**Reference-implementation status, recorded:** EMsoftOO's
`EMHREBSDDIC` (`mod_DIC.f90` + `mod_HREBSDDIC.f90`) is the only
executable-adjacent reference, and it is WIP-quality: nonstandard
inverted-diagonal shape function with a global post-fit sign flip
(`mod_DIC.f90:888-934`, `mod_HREBSDDIC.f90:902`), hardcoded 70 deg
sample tilt in the geometry term (`mod_HREBSDDIC.f90:925`), mixed
pixel/normalized units in the correction chain (commented-out
`/binx` divisions, `:916-924`), corrected homographies computed but
Fehat built from the UNCORRECTED ones (`:978-982`), all
stress/closure code commented out (`:987-1021`), non-converged
points silently zeroed (`:868-870`), and no built binary exists
anywhere on this machine (emsoftoo_report.md section 6).
**Consequence, frozen: every convention in this spec is re-derived
from Ernould's equations (as carried in the theory report) and
pinned by kikuchipy-internal oracles; EMsoftOO quirks are recorded
as deviations and never reproduced.** EMsoftOO's own dev recipe --
warp a reference by a known homography, refit, compare
(`Source/TestPrograms/play2.f90`) -- is fully reproducible in
Python and is the core oracle (validation.md).

## Scope

In scope (three stages on one branch; plan.md sections 2-4):

- **Stage A -- IC-GN engine.** New private package
  `src/kikuchipy/indexing/_hrebsd/` (D15): numba bicubic
  interpolation kernel (D3), in-engine preprocessing (band-pass,
  optional window, border/dead-band SR, D4), initial guess by
  upsampled phase cross-correlation (D5), reference
  gradient/Hessian precompute + IC-GN loop with accumulated-W
  re-warping of the original target (D2), homography<->Fe with
  per-point PC/DD and the beam-scan geometry correction (D6),
  `EBSD.hrebsd_dic()` (name frozen, D15) returning a `CrystalMap`
  with homography/Fe/residual/iteration/convergence props.
  Explicit-index and user-grain-map reference inputs work in Stage
  A; `reference="auto"` raises `NotImplementedError` until Stage B
  (D11).
- **Stage B -- tensors, maps, references.** Polar decomposition and
  the rotation/strain split (D8), traction-free/deviatoric 9th-dof
  closure (D9), stiffness handling incl. `voigt_stiffness` and the
  crystal->sample Bond rotation (D9), stress and derived maps
  (D10), grain auto-segmentation `segment_grains` + per-grain
  best-quality reference auto-selection wired into
  `reference="auto"` (D11), HR-KAM `hrebsd_kam` (D12), PC-shift
  analysis `hrebsd_pc_shift` (D13).
- **Stage C -- GND + tutorial.** Scalar GND densities `hrebsd_gnd`
  (Nye alpha_i3 exact, d/dx3-neglect extra components
  (Pantleon-tier assumption, D14.2), enforced-antisymmetry
  noise fix, OpenXY 3/5/9-component estimators, D14), the tutorial
  notebook `doc/tutorials/hrebsd_dic.ipynb`, CHANGELOG/api/docs
  registration finalization (D15, plan 4).
- **Data**: synthetic oracles + the downloadable strain-free
  `si_wafer` benchmark (50x50 map of 480x480 px patterns,
  `src/kikuchipy/data/_data.py:321-440`); no user-supplied data
  (user decision 2). The shipped Ni 60x60 datasets are
  synthetic-tests/API-demo only: at 60x60 the strain floor is
  ~1e-3, an order or two above literature claims (theory report
  sections 6.8, 7); every precision claim is tied to >= 480x480.
- **Constitution amendments** listed in plan section 0.

Out of scope (confirmed; each recorded with its revisit path):

- **Optical/radial distortion correction** (Ernould,
  Ultramicroscopy 221 (2021) 113158 folds it into the IC-GN loop):
  OUT of v1, a DOCUMENTED LIMITATION in the `hrebsd_dic` docstring
  Notes and the tutorial -- Ernould's thesis data says real
  lens-coupled detectors need it for strains in the 1e-4..2e-3
  band; fiber/direct detectors are less affected (plan open
  question 6).
- **Per-slip-system L1 GND split** (weighted-L1 over a
  slip-system catalog, OpenXY `DDS/resolvedisloc.m:292-363`):
  deferred; v1 ships the scalar estimators (user decision 3; plan
  open question 7). The per-type "split dislocation density" and
  Burgers-vector maps of OpenXY's gallery are deferred with it.
- **Tetragonality map** (OpenXY `CalcTet.m`/
  `TetragonalityOutput.m`, c/a deviation from the recovered F;
  added to the ledger 2026-09-07, spec review -- it is on the
  OpenXY "usual maps" checklist now carried inline in
  validation.md Manual): OUT of v1; revisit path: a one-line
  derived map from the stored `Fe` prop when a use case appears.
- **Simulated-reference absolute strain** (OpenXY
  'Simulated'/'Dynamic Simulated' modes,
  `GetDefGradientTensor.m:74-165`; EMsoftOO's planned-but-absent
  extension, `mod_HREBSDDIC.f90:41-44`): deferred; v1 is strictly
  relative to a measured per-grain reference and the assumption is
  documented + the reference index stored per point (plan open
  question 8).
- **Fourier-Mellin rotation initial guess** (Ernould 2020's
  cascaded FM + XC): deferred; the translation-only seed's capture
  range is measured and recorded instead (D5; plan open question
  5).
- **Cross-correlation (Wilkinson-style multi-ROI) HREBSD**: not
  this feature; nothing from EMsoftOO's classic `mod_HREBSD.f90`
  debug-state module is ported (its formulas are used only as
  cross-references for stiffness rotation and frames, D9).
- **EMsoftOO-compatible HDF5 output layout**: not provided; results
  are a `CrystalMap` (theory report open question 7 resolved: no).
- **Remap/multi-subset grids, robust losses, pyramid multiscale**:
  none of these exist in the reference (emsoftoo_report section
  1.6) and none ship in v1.
- **New required dependencies**: none (D18). GUI, plotting classes
  beyond documented `CrystalMap.get_map_data`/reshape recipes and
  the tutorial's matplotlib cells: none.

## Decisions

### D1 -- Unit system and coordinate frames (frozen)

The single most damaging EMsoftOO defect class is mixed units
(normalized patcent vs pixel stepx/Dref in one formula,
`mod_HREBSDDIC.f90:916-924`; porting risk table rows 1-2). Frozen
resolution: **one unit system, binned detector pixels,
everywhere inside the engine.**

1. Pattern coordinates: `x` = column index (right-positive), `y` =
   row index (down-positive), origin at the upper-left pixel
   center, in binned pixels -- the numpy array frame of the
   kikuchipy `EBSD` signal.
2. PC in pixels from the stored Bruker fractions
   (`_ebsd_detector.py:212-268`): `PCx_px = pcx * Nx`,
   `PCy_px = pcy * Ny`, `DD_px = pcz * Ny` (Bruker `pcz` is DD in
   fractions of `Ny`). No sign flips: Bruker y is from the top,
   same as the array frame. (EMsoftOO's `patcenty = 0.5 - ypc`
   flip, `mod_HREBSDDIC.f90:917`, is an EMsoft-convention artifact
   and never appears here.)
3. DIC coordinates are PC-centered on the GRAIN REFERENCE's PC:
   `xi = (x - PCx_px_ref, y - PCy_px_ref)`, ONE common frame for
   the reference and every target of that grain during DIC
   (correction 2026-09-07, spec review: an earlier draft centered
   each pattern on its OWN PC, which would absorb the pure-PC-
   shift translation into the coordinates, make gamma ~ 0, turn
   the D6.2 gamma removal into a double correction and falsify
   V6's uncorrected-phantom expectation; the Ernould/EMsoftOO
   correction architecture requires correlating both patterns in
   a single frame). Per-point PCs enter ONLY the D6.2 correction
   and the D6 conversion (which uses the target's own PC/DD). The
   homography is parameterized about the PC, not the pattern
   center (deviation
   from EMsoftOO, which warps about (0.5, 0.5) and defers the PC
   to the conversion, `mod_DIC.f90:513-514`,
   `mod_HREBSDDIC.f90:661-667`; recorded -- with PC-centered
   coordinates the h<->Fe map loses its PC cross-terms and the
   pure-translation degeneracy is explicit).
4. Deviation from EMsoftOO's normalized [0,1] coordinates
   (`mod_DIC.f90:188-201`) recorded. Pixel coordinates make
   Ernould's 0.001 px convergence criterion native and kill the
   unit trap. Consequence measured: the 8x8 Hessian's conditioning
   spans ~(half-width)^4 between translation and perspective dofs;
   f64 Cholesky holds ~1e15 so this is safe by ~6 orders at
   480 px, and the warp-refit oracle (validation V2) measures it.
   If (and only if) the oracle refutes f64 solvability, the
   recorded fallback is internal normalization by the pattern
   width -- an implementation detail invisible in the API, to be
   recorded here with the date.
5. Frames: `detector` frame = the pixel frame above with z along
   the beam-to-screen normal (right-handed, x right, y down, z
   from sample toward screen so that `r = (xi_x, xi_y, DD_px)` is
   the ray direction used in D6). `sample` frame = kikuchipy's
   sample frame; detector<->sample via the orix rotation
   `EBSDDetector.sample_to_detector` (built from `sample_tilt`,
   `tilt`, `azimuthal`, `twist`, `_ebsd_detector.py:835-845`) --
   never a hand-built tilt matrix, never a hardcoded angle
   (EMsoftOO's hardcoded 70, `mod_HREBSDDIC.f90:925`, recorded
   deviation). `crystal` frame via the per-point orientation of
   the input `CrystalMap`. The exact detector-frame axis
   conventions (y down vs up relative to `sample_to_detector`) are
   PINNED by the pure-rotation oracle (validation V4), not
   asserted from reading: a pure sample-frame rotation about a
   known axis must come back with the right axis and sign through
   the whole chain.

### D2 -- IC-GN engine (frozen)

8-dof homography `h = (h11, h12, h13, h21, h22, h23, h31, h32)`,
textbook shape function (theory report section 2.1; the form
EMsoftOO comments out, `mod_DIC.f90:918-926`):

```
W(h) = [ 1+h11   h12    h13 ]
       [ h21    1+h22   h23 ]
       [ h31    h32      1  ]
```

acting projectively on PC-centered pixel coordinates:
`(x', y', s) = W . (xi_x, xi_y, 1)`, `xi' = (x'/s, y'/s)`. **The
fitted h maps reference coordinates to target coordinates**: the
converged solution satisfies `target(xi') ~= reference(xi)` after
zero-mean normalization. This is the direction Ernould's
homography2Fe derivation assumes (a lattice direction through
`(xi, DD)` deformed by Fe re-intersects the screen at the warped
position; theory report section 2.3), so NO post-fit sign flip
exists (EMsoftOO's inverted-diagonal `W` + `h <- -h`,
`mod_DIC.f90:888-934` + `mod_HREBSDDIC.f90:902, 961`, are recorded
deviations, never reproduced; risk table row 1). The direction is
pinned by the round-trip + warp-refit oracles (V1, V2) in BOTH
compositions once, then frozen.

Loop, per target pattern (Ernould/Pan IC-GN; cross-checked against
`mod_DIC.f90` with deviations noted):

1. Per-reference precompute (once per grain): interpolation
   coefficients (D3), subregion (SR) gradients `(gx, gy)` from
   analytic spline derivatives on the grid, zero-mean unit-norm
   reference vector `ref_zmn = (ref - mean)/||ref - mean||_2`
   (vector 2-norm, matching `applyZMN_`, `mod_DIC.f90:583-624`;
   the normalization constant is recorded as `ref_norm`),
   steepest-descent images
   `GJ = (gx*x, gx*y, gx, gy*x, gy*y, gy,
   -(gx*x^2 + gy*x*y), -(gx*x*y + gy*y^2))` with `(x, y) = xi`
   (the STANDARD Pan/Ernould signs; EMsoftOO's sign-structure
   variant `getHessian_` rows 7-8, `mod_DIC.f90:785-826`, follows
   its inverted W and is not reproduced), Hessian
   `H = (2/ref_norm^2) * sum_px GJ GJ^T` (SPD up to degeneracy;
   the scale is NOT cosmetic, it must match D2.3's gradient scale
   -- see the dated correction there), Cholesky factor
   computed once (`scipy.linalg.cho_factor`; 8x8, matching DPOSV's
   role, `mod_DIC.f90:870-874`).
2. Initialize `W` from the initial guess (D5).
3. Iterate: warp the ORIGINAL target by the ACCUMULATED `W`
   (evaluate the target's interpolant at `W . xi`), zero-mean
   unit-norm the warped SR, `residuals = ref_zmn - tar_zmn`,
   `CIC = sum(residuals^2)` (ZNSSD), gradient
   `g = (2/ref_norm) * GJ^T residuals`, solve `H dp = -g` via the
   cached Cholesky factor, inverse-compositional update
   `W <- W . W(dp)^{-1}`, renormalize `W <- W / W[2, 2]`.
   **Correction 2026-09-07 (spec review): H and g MUST carry the
   matched normalization above.** The ZNSSD residual's Jacobian is
   `GJ/ref_norm`, so the Gauss-Newton step is
   `dp = -ref_norm * (sum GJ GJ^T)^-1 GJ^T residuals` -- exactly
   EMsoftOO's `2/refstdev^2` (Hessian) vs `2/refstdev` (gradient)
   pairing (`mod_DIC.f90:817, 861-862`). The earlier draft paired
   an UNSCALED `H = sum GJ GJ^T` with the `2/ref_norm` gradient
   and claimed the scale "cancels in `H^-1 g`" -- false: a scale
   on g alone does not cancel; that pairing underscales the step
   by `2/ref_norm^2` (~3e-4 for a 480 px SR on [0, 1]
   intensities), makes the step magnitude depend on the raw
   intensity scale (`dp ~ 1/s^2` under `ref -> s*ref`), and can
   spuriously satisfy the D2.5 exit at the initial guess. Matched
   normalization makes the step invariant under an affine
   intensity rescale, pinned by V2's intensity-scale-invariance
   test.
   **Accumulated-W re-warp of the original target is a frozen,
   recorded deviation from EMsoftOO's progressive warp-of-warp
   with per-iteration re-splining and re-min/max-normalization
   (`mod_DIC.f90:684-747`, esp. 714-722)**: the EMsoftOO scheme
   accumulates interpolation smoothing each iteration; the
   accumulated-W scheme is the textbook IC-GN and interpolates
   exactly once per iteration from the original data. Results
   differ from EMsoftOO at the 1e-4..1e-6 level by construction
   (emsoftoo_report section 5.7); no bit-parity with EMsoftOO is
   claimed anywhere in this spec.
4. Step scaling: `W(dp)` uses `dp * step_scale`;
   `step_scale = 1.0` FROZEN default (plain Gauss-Newton;
   EMsoftOO's 1.5-code/1.25-template accelerator,
   `mod_HREBSDDIC.f90:846-854`, is an unproven heuristic --
   measured on the warp-refit oracle and re-pinned only if it
   demonstrably helps, plan open question 9).
5. Convergence norm, frozen: `norm_dp` = the maximum displacement
   in pixels that the increment warp `W(dp)` induces over the four
   SR corners, `max_corners |proj(W(dp), xi_c) - xi_c|_2`. This is
   the quantity both published criteria approximate (Ernould's
   squared-half-width weighted norm; EMsoftOO's first-power
   `xi1max/xi2max` variant, `mod_DIC.f90:877-881` -- deviation
   recorded, emsoftoo_report section 5.8). Exit when
   `norm_dp < min_step`; `min_step = 1e-3` px FROZEN (Ernould's
   0.001 px criterion; the EMsoftOO template's own conversion note
   equates its normalized 2.5e-5 to this, template lines 89-92).
   Adequacy measured by the convergence sweep (plan open question
   2; validation V2).
6. Cap `max_iterations = 50` FROZEN (EMsoftOO namelist default,
   `mod_HREBSDDIC.f90:242-288`). **Non-converged points keep their
   last iterate with `converged=False` and get NaN in every
   derived prop downstream; they are NEVER zeroed** (deviation
   from `mod_HREBSDDIC.f90:868-870`, risk row 11, recorded).
   Failed patterns (`ptp == 0`, non-finite CIC, any per-pattern
   exception) are marked not-indexed with NaN props, following the
   spherical result-contract precedent (tech-stack.md:33).
7. Outputs per point: `h` (8), final `CIC`, iteration count,
   final `norm_dp`, `converged`, plus the D6-derived `Fe`.

### D3 -- Interpolation (frozen default, order MTP)

- **Default: bicubic B-spline, hand-written numba kernel.**
  `scipy.ndimage.map_coordinates` is not numba-jittable
  (kikuchipy_surface_report section 7), so the inner loop uses a
  hand-written kernel under the constitution's
  `@njit(cache=True, nogil=True)` (no `parallel=True`, no
  `fastmath` until tolerances pass, `.py_func` tested;
  tech-stack.md:44).
- Coefficients: global cubic B-spline prefilter once per pattern
  (`scipy.ndimage.spline_filter(order=3, mode="mirror")`), the
  numba kernel then evaluates the B-spline basis at warped
  positions. **Equality oracle, frozen**: the kernel must match
  `scipy.ndimage.map_coordinates(coeffs, order=3,
  prefilter=False, mode="mirror")` to <= 1e-12 relative on random
  interior points, BOTH sides on f64 coefficients (validation V0;
  scoped 2026-09-07, spec review: `map_coordinates` on f32 input
  computes and returns f32, so 1e-12 is unreachable there -- the
  f32 storage arm carries its own MTP bound measured alongside
  the D17 dtype A/B). Reference gradients are the
  analytic B-spline derivative kernels from the same coefficients.
- Out-of-frame samples during warp evaluate through the `mirror`
  boundary; the SR border (D4) keeps warped samples interior for
  the design shift budget, and the residual boundary bias is
  measured at the warp-refit oracle (V2). (EMsoftOO enables
  b-spline extrapolation instead, `mod_DIC.f90:50-51`; deviation
  recorded.)
- **Order is decided by measurement (plan open question 1)**:
  Ruggles 2018 found biquintic gave no significant gain over
  bicubic at 960x960 (theory report section 1.2); Ernould/EMsoftOO
  use quintic (`bspline-fortran`, kx=ky=5, `mod_DIC.f90:50-51`).
  The Stage A gate runs the warp-refit oracle (V2) with the numba
  bicubic vs a `map_coordinates(order=5)`-backed reference
  implementation and records accuracy (h-recovery error) and
  speed; bicubic stays the default unless quintic buys > 2x
  h-accuracy at < 1.5x cost, in which case the default is
  re-pinned here with the dated measurement. The
  `interpolation="bicubic"` keyword stays either way.

### D4 -- Preprocessing, subregion, border, dead-band (frozen)

Applied identically to the reference and every target, inside the
engine (upstream `remove_static_background`/
`remove_dynamic_background` remain the user's choice and are
demonstrated in the tutorial, never implied):

1. **Band-pass**: FFT-domain Gaussian annulus filter built from
   the existing `kp.filters.Window` lowpass/highpass transfer
   functions (`filters/window.py:150-155, 455, 522`; a band-pass
   is their elementwise product -- kikuchipy_surface_report
   section 2). Knob `filter_cutoffs=(hi, lo)` in fractions of the
   pattern width; default `(0.05, None)` -- high-pass at 0.05,
   no low-pass -- FROZEN as the starting point (EMsoftOO's
   `hipassw = 0.05` default, `mod_HREBSDDIC.f90:242-288`); the
   Si-wafer noise-floor benchmark (V5) measures the default's
   effect and any re-pin is recorded (plan open question 10).
   `scipy.fft` calls pass `workers=1` (tech-stack.md:44).
2. **No adaptive histogram equalization** in the DIC chain --
   recorded deviation from EMsoftOO's shared DI preprocessing
   (`PreProcessPatterns` with AHE nregions=10,
   `mod_HREBSDDIC.f90:689-733`): AHE is a nonlinear, locally
   varying intensity map that violates the affine intensity model
   ZNSSD assumes; the Si benchmark (V5) measures the with/without
   comparison once and records it before any reconsideration.
3. **Window**: optional Hann window over the SR
   (`window=False` FROZEN default -- neither Ernould's chain nor
   EMsoftOO's DIC path windows, emsoftoo_report section 1.7; the
   knob exists because the classic-HREBSD literature windows
   ROIs). Measured on the Si benchmark, recorded.
4. **Subregion**: full pattern minus a border of
   `border` (fraction of the pattern side per edge) --
   `border=0.05` MTP (measuring tests: V2 warp-refit with the
   design shift budget, V5 Si noise floor vs border sweep; the
   conservative 0.05 covers the expected few-px beam-scan
   translations at 480 px). `dead_band=None |
   (x0, x1, y0, y1)` FROZEN semantics: excludes a vertical +
   horizontal cross of dead camera pixels from the SR, matching
   EMsoftOO `cross(4)` (`mod_DIC.f90:524-559`). One global SR per
   run (Ernould-style global DIC; no multi-subset grid, scope).
5. Patterns are NOT re-min/max-normalized after warping (EMsoftOO
   re-normalizes every iteration, `mod_DIC.f90:714-715`; with
   ZNSSD the affine intensity normalization is inherent and the
   re-scaling is a no-op up to float noise -- deviation recorded).

### D5 -- Initial guess (frozen)

`skimage.registration.phase_cross_correlation(ref_pre, tar_pre,
upsample_factor=upsample_factor)` on the preprocessed SR
(scikit-image is a required dependency already,
`pyproject.toml:62` -- but the declared floor 0.16.2 predates the
function, which moved into `skimage.registration` in scikit-image
0.18; the effective/tested floor for this feature path is
recorded in D18); the measured `(dy, dx)` shift seeds
`W0 = W((0, 0, dx, 0, 0, dy, 0, 0))`. `upsample_factor = 16`
FROZEN (1/16 px seed precision, far inside the IC-GN basin;
kikuchipy choice, no upstream equivalent -- EMsoftOO always starts
from identity, `mod_HREBSDDIC.f90:836-858`, which fails for large
translations; deviation recorded in our favor).

- The seed's rotation capture range (largest pure in-plane and
  out-of-plane rotation from which IC-GN converges with
  translation-only seeding) is MEASURED on the pure-rotation sweep
  (V4) and RECORDED in the docstring Notes; Fourier-Mellin
  pre-rotation (Ernould 2020) is deferred (plan open question 5).
- A future orientation-delta seed from the input `CrystalMap`
  (compose the reference->target misorientation into `W0` via D6)
  is recorded as a deferred extension beside Fourier-Mellin, not
  built in v1: within-grain misorientations are small by
  construction and Hough noise (~0.5 deg) is at the scale of what
  it would seed.

### D6 -- Homography <-> Fe, per-point PC/DD, geometry correction (frozen)

Exact conversion in PC-centered pixel coordinates (theory report
section 2.3; cross-checked against `homography2Fe_`/
`Fe2homography_`, `mod_DIC.f90:970-1040`, which are the same
equations in corner-origin coordinates):

```
beta0 = 1 - h31*PCx_rel - h32*PCy_rel     (rel = 0 when the
                                           conversion PC is the
                                           frame origin -- the
                                           grain reference's PC,
                                           D1.3, and every
                                           single-PC run; a
                                           per-point target
                                           converts with
                                           PC_rel = PC_t - PC_ref
                                           through the general
                                           corner-origin form
                                           kept in the private
                                           helper, D6.2)
Fe = (1/beta0) *
     [ 1+h11        h12          h13/DD      ]
     [ h21          1+h22        h23/DD      ]
     [ DD*h31       DD*h32       beta0       ]
```

with `DD = DD_px` (D1). The stored `Fe` prop is the REDUCED tensor
`Fe_hat = Fe / Fe33` (Fe33 = 1 by construction here): the 9th dof
is unobservable and its closure is D9. The inverse map
`fe_to_homography(Fe, PC, DD)` is implemented alongside (needed by
every synthetic oracle) and the round trip is exact to 1e-12 (V1).

Per-point PC/DD and the beam-scan correction, frozen:

1. **Per-point PC is first-class.** When
   `detector.navigation_size > 1` the engine consumes
   `detector.pc_flattened` (`(n, 3)` Bruker,
   `_ebsd_detector.py:481-491`) directly -- the fork's
   `fit_pc`/`extrapolate_pc`/`refine_orientation_projection_center`
   workflows produce exactly this (`_ebsd_detector.py:1315-1424,
   1450`; `ebsd.py:3224`). When the detector carries a single PC,
   the engine derives per-point PCs internally via
   `EBSDDetector.extrapolate_pc` (Singh & De Graef appendix A
   equations, docstring `_ebsd_detector.py:1360-1379`) anchored at
   the REFERENCE pattern's scan position, with step sizes from the
   signal axes -- this is mathematically the same beam-scan model
   as Ernould Ch. 2 section 3.3.2 (theory report section 3.7
   establishes the equivalence).
2. **Correction before conversion.** For each target, the measured
   homography contains a rigid translation `gamma = (g1, g2)` and
   an isotropic scaling `alpha_s` induced purely by the
   reference->target PC/DD change. **Closed form IN THE SPEC'S OWN
   FRAME (added 2026-09-07, spec review)**: in the D1.3
   reference-PC-centered pixel coordinates a pure PC/DD change
   maps `xi' = alpha_s * xi + gamma` with `gamma = delta =
   PC_target - PC_reference` (px) EXACTLY and `alpha_s` the DD
   ratio (drafted as `DD_target/DD_reference`; orientation of the
   ratio pinned by V6, D6.3). The phantom factor is the affinity
   `W_phantom = [[alpha_s, 0, g1], [0, alpha_s, g2], [0, 0, 1]]`
   and the correction removes it by composition (drafted
   `W_corr = W_phantom^-1 . W`; composition side pinned by V6).
   EMsoftOO's formula block (`gamma_i = delta_i +
   (delta_i - patcent_i)(alpha_s - 1)`, `mod_HREBSDDIC.f90:
   955-970`; carried in theory-report section 3.7) is written for
   its absolute-PC/pattern-center coordinates and is NOT
   transplanted -- the `(delta - patcent)(alpha_s - 1)` term
   vanishes only when the PC is the coordinate origin, which is
   exactly the spec's frame (Ernould Ch. 2 section 3.3.2 is the
   common source). The correction is applied analytically BEFORE
   the Fe conversion, and the conversion then uses the TARGET's
   own `(PC, DD)` expressed in the same frame
   (`PC_rel = PC_t - PC_ref` via the D6 general form; the
   residual bookkeeping choices here are first-order equivalent
   and pinned by V6's corrected-phantom `Fe = I` plus V3's
   exactness). This is a frozen deviation
   from EMsoftOO, which computes the corrected homographies but
   converts the uncorrected ones (`mod_HREBSDDIC.f90:978-982`,
   risk row 10, recorded).
3. **Signs are pinned by oracle, not by reading.** The exact signs
   of `deltaDD` and `alpha_s` (EMsoftOO's `alpha =
   (Dref - deltaDD)/Dref` with `deltaD = -stepy*sin(...)`,
   `mod_HREBSDDIC.f90:925-949`, sit in the unit-trap zone) are
   determined by the PC-shift phantom oracle (V6): identity-F
   patterns synthesized on a per-point-PC grid must produce (a)
   the closed-form phantom homography when the correction is OFF
   and (b) `Fe = I` everywhere to the noise floor when it is ON.
   The pinned signs are then recorded here with the date.
4. `sample_tilt`/`tilt`/`azimuthal`/`twist` all come from the
   `EBSDDetector` (D1.5); nothing is hardcoded.

### D7 -- Frame chain (frozen)

`Fe_hat` is measured in the DETECTOR frame. The chain to reported
quantities: `beta_det = Fe_hat - I` -> rotate to the sample frame
`beta_s = R^T beta_det R` with
`R = detector.sample_to_detector.to_matrix()` (direction and
transpose PINNED by the pure-rotation oracle V4, exactly as the
Phase 5 forward-projection lock pinned `rotation_from_zyz`) ->
closure in the sample frame (D9) -> strain/rotation/stress in the
sample frame (D8, D10); the crystal frame enters only through the
stiffness rotation (D9) using the per-point orientation of the
INPUT `CrystalMap` (orientations are never modified by this
feature; the HR rotation lives in props, D15.6).

### D8 -- Rotation/strain split (frozen)

- From the closed full `F_s = I + beta_s` (D9): polar
  decomposition `F_s = R_hr U` (closed-form 3x3 via SVD,
  `numpy.linalg.svd` per point, vectorized). Rotation reported as
  the rotation vector `omega = axis * angle` (radians, sample
  frame) -- the D15.6 `rotation_vector` prop; the small-rotation
  components are its entries at small angle and are NOT stored
  as a separate prop (trimmed 2026-09-07, spec review); strain default
  **Biot, `e = U - I`** FROZEN (OpenXY-compatible,
  `GetDefGradientTensor.m:149-257` reports `U - I`; theory report
  section 3.3), `strain_measure="green-lagrange"`
  (`(U^T U - I)/2`) as the documented alternative.
- **Small-strain fast path** (`e = sym(beta_s)`,
  `w = skew(beta_s)`) exists as a private path and is enabled as
  an automatic optimization ONLY after the Stage B gate measures
  it identical to the polar path within 1e-6 strain over the V4
  rotation sweep up to the recorded threshold angle
  (MTP; measuring test: V4 + V2 tensor comparison). Until then
  every public result goes through the polar path. Error scale for
  the record: small-strain error is O(omega^2) ~ 3e-4 at 1 deg
  (theory report section 3.3).

### D9 -- 9th-dof closure and stiffness (frozen)

The projection loses the isotropic "radial" dof; measurement
determines all off-diagonals plus the differences
`b7 = beta11 - beta33`, `b8 = beta22 - beta33` (theory report
sections 1.1, 3.1). Closure, frozen:

1. **`closure="auto"` default**: traction-free when `stiffness` is
   given, deviatoric otherwise. Explicit `"traction_free"`
   (ValueError without stiffness) and `"deviatoric"` accepted.
2. **Traction-free sigma33 = 0, sample frame**: the OpenXY linear
   post-hoc solve, frozen (CalcF.m:406-542; theory report section
   3.2): solve the 3x3 system for `(e11, e22, e33)`

   ```
   [C3311 C3322 C3333][e11]   [-C3312*(b12+b21) - C3313*(b13+b31)
   [  1     0    -1  ][e22] =    - C3323*(b23+b32)]
   [  0     1    -1  ][e33]   [ b7 ]
                              [ b8 ]
   ```

   with `C` the stiffness rotated crystal->sample at the point's
   orientation. EMsoftOO's in-fit NLopt constraint variant
   (`main_minf`/`myconstraint`, `mod_HREBSD.f90:3118-3332`) is NOT
   used: the post-hoc linear solve applied to the DIC-recovered
   beta is exact for the linearized problem and testable in
   isolation.
3. **Deviatoric fallback**: `tr(beta) = 0` (Ruggles 2018's own
   choice for the ICGN benchmark, theory report section 1.2).
4. **Stiffness input, frozen convention**: a 6x6 Voigt matrix in
   GPa, CRYSTAL frame, Voigt order `(11, 22, 33, 23, 13, 12)`,
   engineering-shear convention (`sigma = C @ [e11, e22, e33,
   2*e23, 2*e13, 2*e12]`) -- stated in every docstring that
   touches it. No elastic-constants database exists anywhere in
   the dependency set (kikuchipy_surface_report section 5) and
   none is added: the stiffness is user-supplied per phase.
   Convenience builder `voigt_stiffness("cubic", c11=, c12=,
   c44=)` / `voigt_stiffness("hexagonal", c11=, c12=, c13=, c33=,
   c44=)` -> `(6, 6)` float64 ndarray (frozen name, D15).
5. **Crystal->sample rotation of C**: the Bond 6x6 construction
   `C_s = M(g) C_c N(g)^-1` equivalently the 4th-order rotation
   `C_ijkl g-products` (OpenXY CalcF.m:324-405; EMsoftOO
   `StiffnessRotation`, `mod_HREBSD.f90:3336-3396` as
   cross-references) -- implemented as the explicit 4th-order
   rotation on the (3,3,3,3) tensor (numpy einsum, unambiguous),
   converted back to Voigt. Pinned by unit tests: rotating cubic C
   by a cubic symmetry op is invariant; a 45 deg z rotation of
   cubic C reproduces the textbook C11' = (C11+C12+2C44)/2 etc.;
   AND (added 2026-09-07, spec review -- the two pins above are
   both invariant under transposing the rotation, so neither
   catches a g-vs-g^T convention error) a +22.5 deg z rotation of
   cubic C reproduces the hand-derived
   `C16'(theta) = (1/4) * sin(4*theta) * (C12 + 2*C44 - C11)`
   INCLUDING its sign (odd in theta, hence transpose-sensitive;
   the test re-derives the closed form independently). This pin,
   with V3's generic-orientation strain case, kills the plan-3.4
   "Bond rotation transposed" mutant and fixes which way the
   input `CrystalMap` orientation enters the rotation of C.
6. Multi-phase maps: v1 requires a single-phase `CrystalMap` for
   the stress path (ValueError naming the limitation); the engine
   itself (Stage A) is phase-agnostic per grain. Recorded
   extension point: per-phase stiffness dict.

### D10 -- Stress and derived maps (frozen)

`sigma_s = C_s : e` (sample frame, GPa; input GPa propagates,
frozen). Derived per-point props (D15.6): von Mises
`sqrt(3/2 s:s)` with `s = sigma - tr(sigma)/3 I`; hydrostatic
`tr(sigma)/3` (documented as closure-derived, theory report
section 3.5); principal stresses (descending eigvalsh, `(n, 3)`).
`sigma33 ~= 0` is NOT stored but IS asserted in tests as the
closure self-check (V3). Quality/masking maps come free from Stage
A props (`residual`, `num_iterations`, `norm_dp`, `converged`).

### D11 -- Grain segmentation and reference selection (frozen)

No grain segmentation exists in orix 0.14.2 or kikuchipy
(kikuchipy_surface_report section 4: zero grep hits); built here,
in scope by user decision 4.

1. **`segment_grains(xmap, *, misorientation_threshold=5.0,
   connectivity=1)`** (public, frozen name): connected components
   on the map grid where an edge links 4-neighbors (`connectivity=1`
   FROZEN default; `2` = 8-neighbor accepted) whose
   symmetry-reduced misorientation angle <
   `misorientation_threshold` (5.0 deg FROZEN default -- the
   standard HREBSD/KAM grain-boundary threshold, OpenXY/MTEX
   practice, theory report section 3.8). Misorientation angles via
   `Orientation.angle_with` on neighbor pairs (orix-0.12.1-safe;
   the constitution bans `reduce()` in tests, tech-stack.md:17).
   Returns `(ny, nx)` int32 labels, 0-based, row-major first-seen
   order, unindexed points -1. Uses `CrystalMap.row/col` grids
   (`crystal_map.py:381, 407`).
2. **Reference auto-selection** (per grain): the point maximizing
   pattern image quality computed internally with the existing
   `get_image_quality` kernel (`pattern/_pattern.py:698`) on the
   raw patterns -- deterministic, independent of which props the
   input xmap happens to carry; ties broken by lowest flat index
   (FROZEN). A `min_boundary_distance` refinement is a recorded
   possible v2 nicety, not built.
3. **`reference` parameter semantics, frozen**:
   - `"auto"` (default from Stage B; NotImplementedError in Stage
     A): segment via (1) unless `grain_labels` is given, then
     select per (2).
   - `(row, col)` tuple: one global reference, one implicit grain
     (all points), matching the EMsoftOO single-reference model
     (`patx/paty`, `mod_HREBSDDIC.f90:744-752`).
   - integer array of flat indices, one per grain label, paired
     with a user-supplied `grain_labels` map: explicit control.
   `grain_labels : (ny, nx) int array | None` accepts a
   user-supplied grain map with any of the above.
4. Everything relative: each point's homography is measured
   against ITS grain's reference; `grain_id` and
   `reference_index` are stored per point (D15.6); cross-grain
   absolute comparison is out of scope (documented; deferred with
   simulated references, plan open question 8).

### D12 -- HR-KAM (frozen)

There is no separately standardized "HR-KAM" in the literature; it
is frozen HERE as the standard KAM evaluated on the HR rotation
field (theory report section 3.8):

- `hrebsd_kam(xmap, *, order=1, psi_max=None)` -> `(ny, nx)`
  float64, **milliradians** (FROZEN unit; the HR noise floor is
  5e-5..1e-4 rad, two orders below conventional KAM's degrees).
- Kernel: all points within Chebyshev distance <= `order` of the
  center (square grid; `order=1` FROZEN default = 8 neighbors;
  the "all within order" convention -- MTEX-style -- is stated in
  the docstring against the OIM perimeter-only alternative).
- Disorientation per pair: `angle(R_p R_q^T)` from the HR
  rotations (Stage B `rotation_vector` prop composed per point);
  within a grain relative to a common reference, so symmetry
  operators are unnecessary and never applied (documented).
- Masking: pairs must share `grain_id` (FROZEN, always on -- the
  HR field is only defined within a grain); `psi_max` (mrad,
  optional, `None` FROZEN default) additionally drops pairs above
  the threshold (sub-grain-boundary guard). Points with no valid
  neighbor -> NaN.
- Mean over surviving pairs (FROZEN; not median).
- Neighborhood machinery reuses the `_map_helper` window pattern
  (`signals/util/_map_helper.py:35-93`) or plain shifted-array
  arithmetic; either is an implementation detail.
- Cross-check identity for the oracle (V7), kernel factor
  included (corrected 2026-09-07, spec review): on a field
  varying linearly along a grid axis (`omega_3(x1) = kappa*x1`,
  kappa in rad/step), the FROZEN order-1 all-8-neighbor mean sees
  6 neighbors differing by `kappa*step` and the 2 along x2 by 0,
  so 1st-order KAM = `(6/8) * kappa * step` EXACTLY -- the 25 %
  factor is kernel geometry, not discretization error (a diagonal
  gradient gives a different factor again); the V7 test computes
  the expected value from the kernel offsets, never asserts
  `kappa * step`.

### D13 -- PC-shift analysis (frozen)

`hrebsd_pc_shift(xmap, detector)` -> dict of named `(ny, nx)`
float64 arrays (FROZEN return type; a diagnostic, not a
CrystalMap):

- `"translation_x"`, `"translation_y"`: the measured RAW
  homography translation components (h13, h23 of the stored
  uncorrected `homography` prop, D15.6; px) per point.
- `"translation_x_model"`, `"translation_y_model"`,
  `"scaling_model"`: the geometric beam-scan model's predicted
  translation and isotropic scaling from the detector's per-point
  PC (D6.1's `extrapolate_pc` geometry).
- `"residual_x"`, `"residual_y"`: measured minus model -- the
  PC-miscalibration probe (Ruggles 2020: nonzero-mean beta13/23
  residuals indicate delta-P miscalibration; theory report section
  3.7).
- Docstring recipe: fitting a plane to the residuals and feeding
  it back through `EBSDDetector.fit_pc`
  (`_ebsd_detector.py:1450`) refines the PC plane -- documented
  workflow, wired in the tutorial; no automatic feedback loop in
  v1 (recorded).

### D14 -- Scalar GND (frozen)

`hrebsd_gnd(xmap, detector, burgers_vector_length, *,
stiffness=None, estimator="a5", enforce_antisymmetry=True)` ->
`(ny, nx)` float64, m^-2. Recomputes the tensor chain internally
from the stored detector-frame `Fe` prop through the SAME private
chain as `hrebsd_strain_stress` (single source of truth, frozen),
because the antisymmetry fix is defined in the detector frame:

1. **Nye convention, frozen**: `alpha = curl(beta_e)` with
   `alpha_ij = eps_jkl d(beta_il)/d(x_k)`; from a 2-D surface map
   exactly the three components
   `alpha_i3 = d(beta_i2)/dx1 - d(beta_i1)/dx2` are exact (theory
   report section 3.6; Ruggles 2020). Sign/index conventions
   differ across papers and codes; THIS convention is validated
   against the constant-curvature synthetic field (V7), never
   against another code's sign (frozen validation principle).
2. **Assumption tiers beyond the exact three (rewritten
   2026-09-07, spec review).** Pantleon (Scripta Mater. 58 (2008)
   994): from the six curvatures `kappa_ij = d(theta_i)/d(x_j)`
   (j = 1, 2), neglecting elastic-strain gradients, additionally
   `alpha_12`, `alpha_21` and the DIFFERENCE
   `alpha_11 - alpha_22` -- six knowns total, NOT nine
   (`alpha_11`/`alpha_22` individually and `alpha_31`,
   `alpha_32` stay inaccessible on that route). The IMPLEMENTED
   extension feeding D14.4's "a5"/"a9" extras is instead
   OpenXY's d/dx3-neglect beta route: in-plane derivatives of the
   measured (closed, sample-frame) `beta_i3` column with every
   `d/dx3` term set to zero give, in the D14.1 convention,
   `alpha_i1 = +d(beta_i3)/dx2` and `alpha_i2 = -d(beta_i3)/dx1`
   (global sign pinned by V7) -- six more entries, all nine in
   total (`DislocationDensityCalculate.m:239, 324-336`: only the
   in-plane derivative slots of `Beta` are ever filled; the
   d/dx3 slots read as zeros -- verified by line reading, NOT a
   Pantleon curvature completion). The two routes coincide when
   elastic-strain gradients vanish and differ by elastic-strain-
   gradient terms; the beta route is FROZEN here because the
   D14.4 prefactors are calibrated for exactly that construction.
   Pantleon's curvature route is recorded context, not built.
3. **Enforced antisymmetry** (`enforce_antisymmetry=True` FROZEN
   default; Ruggles 2020): in the DETECTOR frame, replace
   `beta31, beta32 <- -beta13, -beta23` before rotating to the
   sample frame -- the beta31/32 terms are ~9.6x noisier than
   beta13/23 for typical geometry and dominate GND noise; the fix
   reduces the noise-operator 2-norm ~2.8x. Applied on the GND
   path ONLY, never to stress (frozen). A full pure-rotation
   projection (polar-R only) is explicitly NOT offered (Ruggles
   2020: discarding strain derivatives corrupts GND
   identification; recorded).
4. **Estimators, frozen with prefactors and consumption sets**
   (OpenXY `DislocationDensityCalculate.m:355-375`; rewritten
   2026-09-07, spec review):
   `"a3"`: `(30/10) * sum|alpha_i3| / b` -- the three EXACT
   entries (D14.1) only;
   `"a5"`: `(30/14) * (sum|alpha_i3| + |alpha_12| + |alpha_21|)
   / b` with `alpha_12`/`alpha_21` from the D14.2 d/dx3-neglect
   beta route (OpenXY's own route, `alpha_filt(1,2)/(2,1)` from
   the beta curl -- NOT Pantleon's curvature route; the two
   differ by elastic-strain-gradient terms, recorded);
   `"a9"`: `(30/20) * sum over all nine |alpha_ij| / b`, the six
   extra entries from the same d/dx3-neglect route (an earlier
   draft wrongly attributed them to a "Pantleon-completed set",
   which supplies only six knowns and cannot form nine `|.|`
   terms). Default `"a5"` FROZEN (adds exactly the two entries
   whose neglect Pantleon's analysis independently supports;
   recorded choice). The prefactors are the OpenXY
   L1-extrapolation constants, documented as such (an estimate,
   not a measurement), meaningful only with the matching
   consumption sets above.
5. Gradients: central differences on the map grid, step sizes from
   the xmap coordinate arrays converted to meters using
   `CrystalMap.scan_unit` (set by kikuchipy from the signal axes,
   `ebsd.py:4230-4235`): "um"/"µm" -> 1e-6, "nm" -> 1e-9,
   "m" -> 1; any other or missing unit -- including the "px"
   fallback -- raises ValueError naming `scan_unit` (amended
   2026-09-07, spec review: an earlier draft silently assumed
   micrometers, the exact unit-guessing this decision's own
   `burgers_vector_length` rationale forbids); one-sided
   at map edges; pairs crossing a `grain_id` boundary or touching
   a non-converged/NaN point contribute NaN (frozen NaN-safety
   rule). `burgers_vector_length` in METERS, required, no default
   (frozen: deriving b from the phase structure is a recorded v2
   nicety; silently guessing units is how prefactor bugs hide).
6. Maps plotted log10 in the tutorial; the docstring quotes the
   expected noise floor scale `rho_noise ~ sigma_beta/(b * step)`
   ~ 4-8e12 m^-2 at sigma_beta 1e-4, b 0.25 nm, step 100 nm
   (Jiang/Britton/Wilkinson 2013; Ernould thesis; theory report
   section 3.6) and the Si-wafer measured floor once recorded
   (V5/V7).

### D15 -- Public API surface, naming, props (frozen)

1. **Package**: `src/kikuchipy/indexing/_hrebsd/` (private,
   leading underscore, warning docstring), sibling of
   `_spherical/` -- the binding layout rule (tech-stack.md:43;
   kikuchipy_surface_report section 1). Planned private modules
   (implementation freedom retained): `_interpolation.py`,
   `_engine.py`, `_homography.py`, `_geometry.py`,
   `_preprocessing.py`, `_reference.py` (Stage B:
   `_segmentation.py`, `_tensors.py`, `_stiffness.py`, `_kam.py`,
   `_pc_shift.py`; Stage C: `_gnd.py`).
2. **Method name, frozen: `EBSD.hrebsd_dic()`** in
   `signals/ebsd.py` beside `spherical_indexing`
   (`ebsd.py:1997`). Rationale, recorded: the family methods name
   technique + algorithm (`hough_indexing`, `dictionary_indexing`,
   `spherical_indexing`); `hrebsd` names the technique (it is not
   indexing -- it refines relative to an indexed map, so no
   `_indexing` suffix), `_dic` names the algorithm and leaves the
   name `hrebsd_xcorr` free for a possible future classic
   cross-correlation variant. Alternatives rejected:
   `refine_strain` (wrong family verb; the refinement family
   returns orientations), `hrebsd` bare (blocks the variant
   split).
3. **Public names** (sorted into `indexing/__init__.pyi`
   `__all__`; the API reference regenerates from `__all__`,
   tech-stack.md:52), frozen: `hrebsd_gnd`, `hrebsd_kam`,
   `hrebsd_pc_shift`, `hrebsd_strain_stress`, `segment_grains`,
   `voigt_stiffness` (free functions, following the
   `merge_crystal_maps`/`orientation_similarity_map` flat-function
   precedent). No public engine class in v1 (the per-grain state
   is per-call; recorded -- a public class becomes worthwhile only
   with simulated-reference reuse, deferred).
4. **`EBSD.hrebsd_dic` signature, frozen** (defaults as decided in
   D2-D6, D11):

   ```python
   def hrebsd_dic(
       self,
       xmap: CrystalMap,
       detector: EBSDDetector,
       *,
       reference: str | tuple[int, int] | np.ndarray = "auto",
       grain_labels: np.ndarray | None = None,
       misorientation_threshold: float = 5.0,
       filter_cutoffs: tuple = (0.05, None),
       window: bool = False,
       border: float = 0.05,
       dead_band: tuple | None = None,
       interpolation: str = "bicubic",
       upsample_factor: int = 16,
       max_iterations: int = 50,
       min_step: float = 1e-3,
       step_scale: float = 1.0,
       navigation_mask: np.ndarray | None = None,
       chunksize: int | None = None,
       verbose: int = 1,
   ) -> CrystalMap
   ```

   `xmap` is an argument like the refinement methods take it
   (`ebsd.py:3226` precedent), never `self.xmap` implicitly.
   Masks follow kikuchipy polarity (True = masked out). Harmonics
   are NOT inputs anywhere (nothing spherical is touched).
5. **`hrebsd_strain_stress(xmap, detector, *, stiffness=None,
   closure="auto", strain_measure="biot") -> CrystalMap`**
   (frozen): returns a shallow-copied CrystalMap with the Stage B
   props added; input map unmodified.
6. **Props, frozen** (names, shapes, dtypes; all first-axis n =
   map size, stored FLATTENED -- the fork's 2-D-prop precedent is
   `scores`/`nbest_phase_id`, `ebsd.py:2450-2457`;
   `CrystalMapProperties` accepts `(n, ...)`,
   `crystal_map_properties.py:23-95`):
   - Stage A: `homography` (n, 8) f64 -- stores the RAW fitted h
     (pinned 2026-09-07, spec review: the D6.2 correction is
     applied transiently on the Fe path only, so D13's
     translation analysis reads the real beam-scan translations;
     with a corrected h stored, D13's residuals would be
     identically ~0. EMsoftOO stores both arrays; here raw
     `homography` + corrected-derived `Fe` suffice); `Fe` (n, 9)
     f64 (reduced, detector frame, D6.2-CORRECTED, row-major);
     `residual` (n,) f64 (final CIC);
     `num_iterations` (n,) int32; `norm_dp` (n,) f64 (px);
     `converged` (n,) bool; `grain_id` (n,) int32;
     `reference_index` (n,) int32 (flat index).
   - Stage B: `strain` (n, 6) f64, sample frame, Voigt order
     (11, 22, 33, 23, 13, 12), TENSOR shears (not engineering --
     stated in the docstring; the engineering factor lives only
     inside the Hooke product); `rotation_vector` (n, 3) f64
     (radians, sample frame); `beta` (n, 9) f64 (closed, sample
     frame); `stress` (n, 6) f64 GPa (same order);
     `stress_von_mises` (n,) f64; `stress_hydrostatic` (n,) f64;
     `stress_principal` (n, 3) f64 (descending).
   - `hrebsd_kam`/`hrebsd_gnd`/`hrebsd_pc_shift` return arrays,
     never store props (frozen).
   - Rotations of the returned CrystalMap are the INPUT
     orientations, unchanged (D7).
7. **`get_map_data` verification task (Stage A, frozen)**: a test
   pins `CrystalMap.get_map_data` behavior for a 2-D `(n, k)` prop
   (`crystal_map.py:803` reshapes 1-D/2-D; >2-D unverified --
   kikuchipy_surface_report section 2) on the installed orix and
   the result is recorded in validation.md; regardless of outcome,
   the DOCUMENTED retrieval route for tensor props is
   `xmap.prop["Fe"].reshape(ny, nx, 9)` (conservative default;
   docstrings and tutorial use it), with `get_map_data` mentioned
   only for scalar props. The orix floor 0.12.1 must hold for
   every orix API touched; anything newer is version-gated
   (tech-stack.md:17).

### D16 -- Orchestration, performance, memory (frozen)

- Dask: per-pattern arrays (patterns, seed shifts, per-point PC,
  reference id) are paired with `da.blockwise` exactly as the
  spherical refinement path does (`map_blocks` cannot index
  multiple arguments along one axis --
  `_spherical/_indexer.py:1252-1291` and its comment block;
  kikuchipy_surface_report section 2). Threaded scheduler,
  truthful `chunks=` metadata, worker count via
  `dask.config.get("num_workers")` -- the `_indexer.py` precedents
  (`:1055-1110, 455-466`).
- The engine may internally order patterns grain-by-grain (each
  chunk correlates against one reference's precomputed state) and
  MUST restore map order in the result; the run is deterministic
  for fixed inputs and chunking (frozen; pinned by a
  two-runs-bitwise test).
- Per-grain resident state (coefficients + gradients + Hessian
  factor): ~5 f32/f64 planes of the SR, ~4.6 MB at 480x480 --
  the memory note in the info message follows the
  `SphericalIndexer.get_info_message` precedent
  (`_indexer.py:1964`).
- Performance numbers are recorded baselines in validation.md,
  never merge gates (tech-stack.md:39); no hard floor is set for
  v1 (recorded; the spherical >= 2 pat/s/core floor is
  EMSphInx-scoped). Baseline recipe: patterns/s on the Si-wafer
  route at 480x480 and on `nickel_ebsd_large` at 60x60 (V5,
  validation Performance).

### D17 -- Float discipline (frozen policy, MTP pin)

The constitution's float64-throughout rule is EMSphInx-scoped
(tech-stack.md:21). HREBSD-DIC, frozen policy: **f64 for every
solver accumulator** (Hessian, gradient, residual sums, CIC,
h/Fe/tensor math) -- non-negotiable; **f32 for bulk per-pattern
storage** (patterns, spline coefficients, gradient planes) --
PROVISIONAL, MTP: the Stage A gate runs the warp-refit oracle (V2)
with f32 vs f64 coefficient storage and pins f32 only if the
h-recovery degradation is < 10 % of the f64 error; the verdict and
numbers are recorded in validation.md. Whichever way it lands, the
choice is recorded HERE with the date.

### D18 -- Dependencies, licensing, attribution (frozen)

- **No new required dependency** (mission criterion 3 pattern):
  numpy/scipy/numba/dask + scikit-image (already required,
  `pyproject.toml:62`) + orix >= 0.12.1 (`pyproject.toml:58`)
  cover everything. **scikit-image floor caveat, recorded
  (2026-09-07, spec review)**: `skimage.registration.
  phase_cross_correlation` (D5) does not exist at the declared
  pyproject floor 0.16.2 (the function landed in
  `skimage.registration` in 0.18); the effective, TESTED floor
  for the hrebsd path is 0.21.0 -- the CI oldest job's pin
  (`.github/workflows/tests.yml:48`) -- which the plan section 1
  local oldest-matrix recipe pins explicitly. No import-time
  version gate is added: below 0.18 the D5 import itself fails
  with a plain ImportError at first use, and no supported
  environment (CI or the recipe) sits below 0.21.0.
  `skimage.transform.ProjectiveTransform/warp`
  are TEST-ORACLE-ONLY imports for this feature (an independent
  warper for V2's cross-interpolator leg), never imported by any
  `_hrebsd/` module (the pre-existing upstream import in
  `detectors/_fit_projection_center.py:32` backing `fit_pc` is
  untouched and out of scope); sympy stays banned; pooch stays
  optional (Si-wafer tests skip cleanly without it,
  tech-stack.md:16).
- **License/attribution**: all `_hrebsd/` code is written from
  scratch against the published equations; no EMsoftOO or OpenXY
  code is ported verbatim. Files get the plain kikuchipy GPL
  header (the default `licenseheaders` hook stamps it,
  `.pre-commit-config.yaml:43-46`). Module docstrings cite
  Ernould 2020/2021/2022, Ruggles 2018/2020, Pantleon 2008,
  Wilkinson 2006, Jiang 2013, Hardin 2015 (added to
  `doc/user/bibliography.bib` as `:cite:` keys, Stage C), and name
  EMsoftOO `mod_DIC.f90` (BSD-3) and OpenXY (GPL) as
  cross-checked references. IF any code is later ported verbatim
  from either, the `_master_pattern.py:20-57` third-party-block
  convention applies with the source's own license (BSD-3 for
  EMsoftOO -- compatible; OpenXY is GPL-2.0 -- compatible with
  attribution); the reviewer checks this at every stage gate.
- `pre-commit run --files <changed>` only, never `--all-files`
  (tech-stack.md:44).

### D19 -- Measurement deferral (recorded)

No probes ran at drafting (spec-stage constraint). Gates: (1) each
stage's failing-tests commit carries placeholder bands marked
`MEASURED-THEN-PINNED` (`pytest.approx(measured, rel=0.05)` or the
~2x margin convention); (2) the implementation gate replaces every
placeholder with a dated measured value in validation.md "Recorded
results" (recipe + machine ID); (3) the adversarial review
re-measures the decision-critical ones (the D2/D6 sign pins, the
D3 order comparison, the D17 dtype verdict, the V5 noise floor).
Any decision refuted by measurement is amended in THIS file with a
dated correction (the Phase 8/10 precedent).

## Context

- **Constitution**: `specs/tech-stack.md` (layout rule :43, numba
  rules :44, assertion convention :51, notebook rules :52,
  process :70-84 -- all binding here; float64 rule :21 is
  EMSphInx-scoped, see D17); `specs/roadmap.md` (gains the
  HREBSD-DIC feature path, plan 0.1); `specs/mission.md` (gains
  the fork-only scope note, plan 0.2). The approved plan of record
  is `C:/Users/westraadt.1/.claude/plans/
  i-want-a-spherical-shimmying-catmull.md` (top section), whose
  user decisions 1-4 and branch policy are restated in Scope and
  plan.md section 1.
- **Research** (2026-09-07 session scratchpad, does not survive;
  every load-bearing fact is carried inline above):
  `hrebsd_research/emsoftoo_report.md` (EMsoftOO line map, quirk
  and risk tables -- the D1/D2/D6 deviation lists), `hrebsd_
  research/kikuchipy_surface_report.md` (fork surface, reuse
  points, KAM/stiffness absences -- D11/D12/D15/D16),
  `hrebsd_research/theory_maps_report.md` (equations, literature
  precision table, OpenXY inventory, oracle designs -- D6-D14 and
  validation.md).
- **Key literature** (full citations in the theory report):
  Ernould et al. Acta Mater. 191 (2020) 131; Ernould et al.
  Ultramicroscopy 221 (2021) 113158; Ernould et al. AIEP 223
  (2022) Chs. 1-3; Ruggles et al. Ultramicroscopy 195 (2018)
  85; Ruggles et al. Ultramicroscopy 210 (2020) 112927; Pantleon
  Scripta Mater. 58 (2008) 994; Wilkinson/Meaden/Dingley
  Ultramicroscopy 106 (2006) 307; Hardin et al. J. Microscopy 260
  (2015) 73; Jiang/Britton/Wilkinson Ultramicroscopy 125 (2013)
  1; Britton et al. Ultramicroscopy 110 (2010) 1443.
- **Precision expectations carried for pinning context** (theory
  report section 7): ICGN homography at 960x960: strain ~7.8e-5,
  rotation 0.0053 deg (Ruggles 2018); CC-HREBSD ideal ~1e-4-2e-4;
  GND floor 4-8e12 m^-2; kikuchipy Ni 60x60 expectation ~1e-3
  class (tests only). Suggested spec-level acceptance seeds
  (validation.md, all MTP): synthetic warp recovery <= 1e-4
  h-units at 480^2 noise-free (the theory report's phrasing;
  V2 re-expresses it as a unit-consistent corner-displacement
  metric in px, since raw h components mix px and 1/px units --
  2026-09-07 spec review); deformed-master strain error <=
  2e-4 per component at 480^2; pure-rotation strain leakage <=
  1e-4 up to 5 deg; constant-curvature GND oracle < 1 % on
  analytic beta.
- **EMsoftOO quirk ledger consumed by the deviations above**
  (single index): inverted W + sign flip (`mod_DIC.f90:888-967`,
  `mod_HREBSDDIC.f90:902`); warp-of-warp re-splining
  (`mod_DIC.f90:684-747`); per-iteration re-normalization
  (`:714-715`); first-power convergence norm (`:877-881`);
  hardcoded 70 deg tilt (`mod_HREBSDDIC.f90:925`); mixed units
  (`:916-924`); uncorrected Fehat (`:978-982`); zeroed
  non-converged points (`:868-870`); stress code commented out
  (`:987-1021`); recl mismatch latent bug (`:643-659` vs `:742`);
  identity-only initial guess (`:836-858`). None reproduced.
