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
   kikuchipy `EBSD` signal. **Dated correction 2026-09-07 (Stage A
   failing-tests gate, measurement): the PC offset carries a
   half-pixel term, `x_from_PC = col + 0.5 - PCx_px` and
   `y_from_PC = row + 0.5 - PCy_px`** -- see D1.3. The Bruker
   fractions of D1.2 are measured from the detector EDGE, while a
   column INDEX names a pixel CENTRE, so the two differ by half a
   binned pixel. MEASURED, not argued: mapping
   `EBSDDetector.sample_to_detector` into this frame and scaling
   every direction cosine to `z = DD_px` reproduces
   `col + 0.5 - PCx_px` to 2e-14 px over a whole 40 by 60 detector
   (`tests/test_indexing/test_hrebsd_engine.py`,
   `TestPcCentredFrame::test_pc_centred_frame_matches_kikuchipy_geometry`;
   recorded in validation.md). Dropping the term would give every
   PC-derived quantity a systematic half-pixel offset against
   `fit_pc`, `extrapolate_pc` and every stored projection centre.
2. PC in pixels from the stored Bruker fractions
   (`_ebsd_detector.py:212-268`): `PCx_px = pcx * Nx`,
   `PCy_px = pcy * Ny`, `DD_px = pcz * Ny` (Bruker `pcz` is DD in
   fractions of `Ny`). No sign flips: Bruker y is from the top,
   same as the array frame. (EMsoftOO's `patcenty = 0.5 - ypc`
   flip, `mod_HREBSDDIC.f90:917`, is an EMsoft-convention artifact
   and never appears here.)
3. DIC coordinates are PC-centered on the GRAIN REFERENCE's PC:
   `xi = (col + 0.5 - PCx_px_ref, row + 0.5 - PCy_px_ref)` (the
   pixel-centre form of D1.1's dated correction, 2026-09-07; an
   earlier draft wrote `xi = (x - PCx_px_ref, y - PCy_px_ref)`
   without the half-pixel term, which the measurement above
   refutes), ONE common frame for
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
   **MEASURED 2026-09-07 (Stage A implementation gate),
   CONCLUSION CONFIRMED, the estimate itself slightly
   conservative**: the assembled 8x8 Hessian of the 480 px oracle
   reference (186624 subregion pixels) has 2-norm condition number
   5.75e+08 (eigenvalues 1.51e-01 to 8.70e+07), i.e. 6.2 orders of
   headroom under 1e15 exactly as claimed, though the spread is
   5.8x SMALLER than the drafted `(half-width)^4 = 3.32e+09`
   estimate. Symmetrically scaled by its own diagonal the same
   matrix conditions at 4.5, so the spread is entirely the unit
   system and not a near-degeneracy. At 60 px the condition number
   is 3.03e+05 (scaled 9.5). `scipy.linalg.cho_factor` succeeds on
   both, and the V2 recovery bands sit at the interpolation floor,
   so the recorded normalize-by-width fallback is NOT taken.
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
   **MEASURED AND PINNED 2026-09-07 (Stage B failing-tests gate;
   see the D7 correction below).** The y-down detector frame of
   this clause is `sample_to_detector` composed with
   `diag(1, -1, 1)`, kikuchipy's own gnomonic detector frame
   having y UP. That flip is a REFLECTION (determinant -1), so it
   cannot be absorbed into the detector's own rotation and cannot
   be reached by any choice of `sample_tilt`/`tilt`/`azimuthal`/
   `twist`: the composed matrix, not the orix rotation, is what
   the frame chain of D7 uses. Nothing is hardcoded by this --
   the rotation still comes from the `EBSDDetector` -- and the
   determinant is asserted in
   `test_hrebsd_tensors.py::TestFrameChain::test_the_detector
   _frame_is_the_y_down_one`.

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
   **MEASURED 2026-09-07 (Stage A implementation gate):
   `step_scale = 1.0` CONFIRMED; the EMsoftOO accelerator is
   refuted as an accelerator.** On the twelve-case V2 batch at
   480 px (seeds 0 and 1), all three settings converge every case
   to the same accuracy -- worst recovery error 0.012440 px at
   1.0, 0.012461 at 1.25, 0.012482 at 1.5 -- while the iteration
   count goes the WRONG way: mean 5.17 (62 total) at 1.0, 7.33
   (88) at 1.25, 12.75 (153) at 1.5. The basin is unchanged too:
   the largest in-plane rotation recovered is 2.0 degrees for all
   three, and the 15 px translation case converges at each (2, 3
   and 6 iterations). So 1.5 costs 2.5x the iterations for a
   0.3 % accuracy loss and no robustness gain. No re-pin.
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
   **BOTH DEFAULTS CONFIRMED AND NOT RE-PINNED 2026-09-08; plan open
   question 2 is CLOSED** (Stage B implementation gate, validation
   entry 45). The error-versus-threshold curve the plan asked for,
   MEASURED on the Si wafer, is FLAT: the strain floor is
   1.2198e-02 at `min_step = 1e-2`, 1.2168e-02 at the frozen 1e-3
   and 1.2416e-02 at 1e-4, a 2 per cent spread over a hundredfold
   change, while convergence goes 97 / 87 / 26 of 100 and the mean
   iteration count 11.3 / 33.3 / 48.2. So the threshold is not what
   limits the result and the default leaves no systematic error
   above the floor, which is the condition the plan set for keeping
   it. A looser 1e-2 is tempting (97 of 100 in a third of the
   iterations for the same floor) and is REFUSED: that argument
   holds only on a dataset where the fits carry no signal anyway
   (the D13 and D4.1 records of the same date), and on the Stage A
   oracle the frozen 1e-3 already exits in 5.2 iterations at the
   interpolation floor (Recorded results entry 10), so it costs
   nothing where the measurement works.
6. Cap `max_iterations = 50` FROZEN (EMsoftOO namelist default,
   `mod_HREBSDDIC.f90:242-288`).
   **CONFIRMED 2026-09-08 with the same sweep, and it is doing real
   work on real data**: convergence is 1 of 100 at a cap of 10,
   8 at 20, 87 at the frozen 50 and 96 at 100, with the strain
   floor unchanged (1.3133e-02 at 20, 1.2168e-02 at 50, 1.2175e-02
   at 100). A smaller cap would be badly wrong; raising it to 100
   buys nine points of convergence for 15 per cent more time and no
   change in the floor, which is not a re-pin. **Non-converged points keep their
   last iterate with `converged=False` and get NaN in every
   derived prop downstream; they are NEVER zeroed** (deviation
   from `mod_HREBSDDIC.f90:868-870`, risk row 11, recorded).
   Failed patterns (`ptp == 0`, non-finite CIC, any per-pattern
   exception) are marked not-indexed with NaN props, following the
   spherical result-contract precedent (tech-stack.md:33).
7. Outputs per point: `h` (8), final `CIC`, iteration count,
   final `norm_dp`, `converged`, plus the D6-derived `Fe`.
   **"Final CIC" made literal 2026-09-07 (Stage A adversarial
   review).** The criterion is evaluated ONCE MORE after the loop
   exits, at the returned `h`, so that the stored `residual` and the
   stored `homography` describe the SAME iterate. The earlier code
   reported the criterion of the iterate BEFORE the last
   composition: MEASURED 1.6e-05 relative high on a converged 480 px
   fit, which is negligible, and **26 per cent high** on a fit
   capped by `max_iterations` (1.5434 reported against 1.2240 at the
   returned homography) -- exactly the points D10's quality map is
   read on. Cost, recorded: one extra interpolation per fit, so the
   480 px oracle baseline moves from 23.96 to 22.72 patterns/s,
   about 5 per cent, on a quantity D16 makes a recorded baseline and
   never a gate.

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
- **ORDER DECIDED 2026-09-07 (Stage A implementation gate):
  BICUBIC STAYS THE DEFAULT.** Measured both ways and neither
  meets the re-pin criterion. On the V2 warp-refit oracle
  (controlled A/B, identical loop/preprocessing/seed/Hessian/exit,
  only the spline order changed, six random small homographies at
  480 px): worst corner-displacement recovery error 0.0105 px
  bicubic versus 0.0126 px QUINTIC, i.e. quintic is 0.83x as
  accurate -- WORSE, not 2x better -- at 3.31x the evaluation cost
  (4.62 ms versus 15.31 ms for the 186624 subregion points). On
  the analytic band-limited truth (`test_order_ab_harness`)
  quintic is 1.84x more accurate (5.75e-05 versus 1.05e-04 scale
  relative), which is below the 2x threshold and is anyway swamped
  at oracle level by the cross-interpolator systematic. This
  reproduces Ruggles 2018's finding (no significant biquintic gain)
  and refutes nothing in the drafted design; Ernould/EMsoftOO's
  quintic choice stays a recorded deviation. Recorded in
  validation.md with the recipe and machine.

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
   **PROVENANCE AND EFFECT, recorded 2026-09-07 (Stage A
   adversarial review); the DEFAULT IS UNCHANGED.** Two things
   about it are now on the record rather than assumed. (a) The
   number is not a transplant of EMsoftOO's `hipassw`: kikuchipy's
   knob maps to `highpass_fft_filter(cutoff=high_pass * width)`, a
   cut-off RADIUS in FFT bins (24 bins at 480 px, DC transmission
   `exp(-8) = 3.4e-4`), while EMsoftOO's `hipassw` is a
   normalized-frequency Gaussian parameter. The two share a numeral,
   not a unit system; the kikuchipy definition is self-consistent
   and is what is frozen, and the EMsoftOO citation is demoted to
   "the same numeral, a different filter". (b) It COSTS capture
   range and noise-free accuracy, MEASURED on the V4
   deformed-master in-plane sweep with `filter_cutoffs` the ONLY
   change: `(0.05, None)` recovers 0.5/1.0/2.0 deg (5, 5, 8
   iterations; 0.00215/0.00397/0.01130 px) and fails from 2.5 deg,
   while `(None, None)` recovers 2.0 (11 it, 0.00110 px), 2.5 (15,
   0.00165), 3.0 (18, 0.00167) and 4.0 (41, 0.00168) and fails at
   5.0. The pair ZNCC at 4.0 deg is -0.069 with the default and
   +0.196 without, so the "decorrelated pair" of the D5 record is a
   property of the filter too. The default STAYS: on noise-free
   oracles a high-pass can only remove signal, and what it exists
   for -- background gradients on real noisy patterns -- is the V5
   Si-wafer measurement of plan open question 10, at the Stage B
   gate. Recorded so that the V5 re-pin has a baseline, and quoted
   CONDITIONALLY in the `hrebsd_dic` docstring.
   **THAT MEASUREMENT IS NOW IN, AND THE DEFAULT IS CONFIRMED, NOT
   RE-PINNED: plan open question 10 is CLOSED 2026-09-08** (Stage B
   implementation gate, validation Recorded results entry 45). On the
   Si wafer, one knob changed per arm on 100 patterns, the high-pass
   is not a luxury but the whole measurement:

   | `filter_cutoffs` | converged | median residual | strain floor |
   |---|---|---|---|
   | `(0.05, None)`, frozen | **87/100** | 0.9496 | 1.2168e-02 |
   | `(None, None)` | **1/100** | 0.0534 | not measurable |
   | `(0.05, 0.4)` | 89/100 | 0.8806 | 1.2246e-02 |
   | `(None, 0.4)` | **1/100** | 0.0482 | not measurable |

   So the two records bracket the default honestly and neither is
   withdrawn: on noise-free synthetic patterns the high-pass halves
   the capture range and costs a factor of ten in accuracy (the
   record above), and on real ones it takes convergence from 1 per
   cent to 87 per cent. The low-pass arm is the only one that could
   have argued for a re-pin and does not -- two extra converged
   points and a 0.6 per cent WORSE strain floor -- so `(0.05, None)`
   stands unchanged.
2. **No adaptive histogram equalization** in the DIC chain --
   recorded deviation from EMsoftOO's shared DI preprocessing
   (`PreProcessPatterns` with AHE nregions=10,
   `mod_HREBSDDIC.f90:689-733`): AHE is a nonlinear, locally
   varying intensity map that violates the affine intensity model
   ZNSSD assumes; the Si benchmark (V5) measures the with/without
   comparison once and records it before any reconsideration.
   **THAT COMPARISON IS NOW MADE AND THE REFUSAL IS CONFIRMED
   2026-09-08** (Stage B implementation gate, validation Recorded
   results entry 47): on the Si wafer, kikuchipy's own
   `EBSD.adaptive_histogram_equalization` at its default kernel takes
   convergence from **87 of 100 patterns to 5**, raises the median
   ZNSSD residual from 0.9496 to 1.6346 and drives the mean iteration
   count to 49.3 of a cap of 50, so essentially every fit runs out of
   iterations. The theoretical argument is measured, not merely
   argued, and reconsideration is closed. (It is measured OUTSIDE the
   sweep because AHE is an upstream kikuchipy call and not a knob of
   `hrebsd_dic`; the recipe is in the ledger entry.)
3. **Window**: optional Hann window over the SR
   (`window=False` FROZEN default -- neither Ernould's chain nor
   EMsoftOO's DIC path windows, emsoftoo_report section 1.7; the
   knob exists because the classic-HREBSD literature windows
   ROIs). Measured on the Si benchmark, recorded.
   **WHERE IT IS APPLIED, made explicit 2026-09-07 (Stage A
   adversarial review, which found the shipped knob implemented
   against this decision and uncovered by any test).** "Over the SR"
   is now literal on both counts: the Hann is built over the
   SUBREGION bounding box (`hann_window(shape, bounds=...)`), and
   the engine applies it as a per-pixel WEIGHT on the ZNSSD residual
   IN THE REFERENCE FRAME -- after the target is warped -- which
   weights the steepest-descent images by the same `w` and leaves
   the D2.1 Hessian and D2.3 gradient formulas untouched. The
   zero-mean unit-norm vectors stay UNwindowed, so the affine
   intensity invariance of ZNSSD is exact. The earlier code built
   the window over the WHOLE pattern and multiplied it into each
   pattern in that pattern's OWN frame inside `preprocess`, before
   the warp, so the taper travelled with the target and broke the
   affine intensity model: MEASURED on a 120 px synthetic, a 2 px
   translation recovered 0.30312 px against 0.01601 px unwindowed
   (18.9x) and a generic homography 0.11304 against 0.00660 (17.1x),
   with the error vanishing at the identity -- the signature of a
   window applied in the un-warped frame. With the correction, the
   480 px V2 pair measures 0.04565 px windowed against 0.00656 px
   plain, and the band `WINDOW_REFIT_TOL_480 = 0.092` (2x) is now
   exercised end to end; the pre-correction scheme measures
   0.14949 px on the same two cases and fails it.
   **`window=False` CONFIRMED AS THE DEFAULT ON REAL DATA
   2026-09-08** (Stage B implementation gate, validation entry 45,
   the plan open question 10 sweep): with the Hann window on, 14 of
   100 Si-wafer patterns converge against 87 with it off, and the
   strain floor is 1.2474e-02 against 1.2168e-02. The window's low
   median residual there (0.2691 against 0.9496) is not a win and
   must not be read as one -- it is the taper's own weighting of the
   criterion, carried by the 86 points that never converged. The
   knob stays, correctly implemented per the note above, and stays
   off by default.
4. **Subregion**: full pattern minus a border of
   `border` (fraction of the pattern side per edge) --
   `border=0.05` MTP (measuring tests: V2 warp-refit with the
   design shift budget, V5 Si noise floor vs border sweep; the
   conservative 0.05 covers the expected few-px beam-scan
   translations at 480 px).
   **CONFIRMED AND NOT RE-PINNED 2026-09-08; plan open question 3 is
   CLOSED** (Stage B implementation gate, validation entry 45). The
   plan asked the V5 sweep to "pin the knee" of the noise floor
   against the border fraction. MEASURED on the Si wafer, there is no
   knee to pin: the floor moves 11 per cent over a fourfold change in
   the knob (1.2003e-02 at 0.0, 1.2168e-02 at 0.05, 1.2146e-02 at
   0.1, 1.3364e-02 at 0.2) while convergence falls 93 / 87 / 83 / 52
   of 100 and the median residual rises 0.62 / 0.95 / 1.44 / 1.77.
   The flat floor must NOT be read as "use `border=0.0`": the border
   exists for the few-pixel beam-scan translations of the sentence
   above, and on that dataset the fits do not track translations at
   all (the D13 record of the same date), so the one quantity the
   knob is for is the one the data cannot exercise. The default
   stands on its original reasoning; the sweep is worth re-running on
   plan open question 13's Si-indent dataset. `dead_band=None |
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
  **MEASURED 2026-09-07 (Stage A implementation gate): the capture
  range is 2.0 degrees of pure in-plane rotation at 480x480 with
  the frozen `(0.05, None)` band-pass, NOT the 5 degrees the
  Context section listed as a drafting seed (that seed is amended
  there with this date, requirements D19).** Sweep on the
  deformed-master oracle, phase-XC seeded, converged AND within
  1.0 px of the exact homography: 0.5 deg (5 iterations,
  0.0021 px), 1.0 deg (5, 0.0040 px), 2.0 deg (8, 0.0113 px);
  2.5 deg onward fail at the 50-iteration cap. Two separate limits,
  measured apart: (a) the SEED decorrelates -- the preprocessed
  ZNCC of the pair falls 0.876, 0.622, 0.176, 0.049, -0.024 at
  0.5/1.0/2.0/2.5/3.0 deg, and the phase cross-correlation returns
  9.8 px and 9.3 px of spurious translation at 2.5 and 3.0 deg
  where the true translation is zero; (b) from the exact
  (identity) seed the IC-GN basin itself reaches 3.0 deg
  (11 and 16 iterations at 2.5 and 3.0) and fails at 4.0 deg.
  This is what sizes the deferred Fourier-Mellin stage, and it is
  the number the `hrebsd_dic` docstring Notes carry.
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
   **SCAN-STEP UNITS, added 2026-09-07 (Stage A adversarial review;
   the D14.5 precedent applied to this path).** That model divides
   every step by `px_size * binning`
   (`_ebsd_detector.py:1360-1379`), so the navigation axes' `scale`
   and `EBSDDetector.px_size` must share a unit. `EBSD.hrebsd_dic`
   therefore READS `axes_manager.navigation_axes[i].units` and
   converts to the micrometres `px_size` is measured in, raising a
   ValueError naming the axis for a missing or unrecognized unit
   (HyperSpy's unscaled `"px"` included) whenever the detector
   carries a single PC -- the same rule D14.5 pins for `scan_unit`,
   for the same reason. The earlier code fed `scale` in unchecked,
   which the review MEASURED: a strain-free phantom map whose steps
   were described in nm instead of um gave `max|Fe - I| = 4.71e+01`
   instead of 1.70e-05, and in mm the correction silently became a
   no-op (4.71e-02, the uncorrected value), with no warning in any
   of the three runs. `px_size` itself carries NO unit and cannot be
   checked -- its default 1.0 is a placeholder and kikuchipy's own
   `nickel_ebsd_small` ships it beside a 1.5 um scan step, a 70x
   mismatch -- so the requirement that it be set is DOCUMENTED in
   the `hrebsd_dic` Notes, in `run_hrebsd_dic` and in
   `per_point_pc_pixels`, and stays the caller's (recorded: a
   placeholder is indistinguishable from a genuine 1 um pixel, so a
   guard there would fire on the shipped demonstration data).
   **THE FIRST CALLER TO TRIP ON IT WAS OUR OWN BENCHMARK, recorded
   2026-09-08 (Stage B implementation gate, validation Recorded
   results entry 41); the decision NOT to guard STANDS.**
   `tests/test_indexing/test_hrebsd_si.py` built its per-point
   detector straight from `kp.data.si_wafer()`, which ships the
   placeholder `px_size = 1.0` beside a 40 um scan step, an 90x
   mismatch. MEASURED consequence on a 480 px detector: 1800 px of
   modelled PCx drift, 1691 px of PCy and 616 px of detector
   distance, and a reported strain floor of **1.033**, which fails
   that module's own order check by 52x. With the NORDIF UF-420
   pixel size kikuchipy's own `doc/tutorials/pc_fit_plane.ipynb`
   states for this very dataset (about 90 um, from which it derives
   the expected 2000/90 = 22 px shift), the same run measures 20.0 px
   of modelled drift and a floor of 1.2e-02. So the documented
   responsibility is real and is discharged AT THE CALL SITE, with
   the dated note the test file carries; a guard is still refused
   for the reason above, and the lesson recorded here is that every
   kikuchipy-shipped dataset needs its pixel size supplied before
   any beam-scan model is trusted.
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
   the Fe conversion. This is a frozen deviation
   from EMsoftOO, which computes the corrected homographies but
   converts the uncorrected ones (`mod_HREBSDDIC.f90:978-982`,
   risk row 10, recorded).
   **CONVERSION FRAME, CORRECTED 2026-09-07 (Stage A adversarial
   review, requirements D19). The drafted rule -- "the conversion
   then uses the TARGET's own `(PC, DD)` expressed in the same
   frame, `PC_rel = PC_t - PC_ref`; the residual bookkeeping choices
   here are first-order equivalent" -- is REFUTED by re-derivation
   and by measurement. The corrected homography converts with
   `PC_rel = (0, 0)` and `DD = DD_REFERENCE`.** Re-derivation: a raw
   fit in the D1.3 frame is
   `W = T(delta) . diag(1, 1, 1/DD_t) . Fe . diag(1, 1, DD_r)`,
   whose `Fe = I` case is EXACTLY the closed-form phantom above (so
   the ray model and the V6-pinned phantom agree, to 0.0). Removing
   that phantom on the D6.2 side therefore leaves
   `W_corr = diag(1, 1, DD_r)^-1 . Fe . diag(1, 1, DD_r)`, a PURE
   reference-frame homography in which both the PC offset and the DD
   ratio have already cancelled by construction; and at
   `PC_rel = (0, 0)` the D6 conversion IS that conjugation, so
   `Fe = homography_to_fe(W_corr, (0, 0), DD_ref)` is exact.
   Measurement: the drafted route injects the spurious isotropic
   strain `-(Fe31*dx + Fe32*dy)/DD_ref` into Fe11/Fe22 and rescales
   Fe13/Fe23/Fe31/Fe32 by the DD ratio. On the V6 phantom geometry
   with a 1 degree out-of-plane tilt imposed, the stored `Fe` was
   wrong by **3.9967e-04** at the three map points carrying a row
   offset -- 13x the pinned `DEFORMED_MASTER_FE_TOL = 3.1e-5` and 5x
   the 8e-5 strain precision the `hrebsd_dic` docstring quotes --
   against **2.2255e-05** through the corrected route (inside the
   band). The blindness was structural, which is why the drafted
   oracles all passed: V6's `test_phantom_corrected` imposes `Fe = I`,
   where `h_corr = 0` makes EVERY `pc_rel`/`dd` give the identity,
   and V3's `test_deformed_master_fe_through_the_engine` says in its
   own comment that "every point shares one projection centre", so
   `PC_rel = 0` and `DD_t = DD_r` there. A new oracle,
   `TestPcShiftPhantom::test_corrected_fe_on_a_deformed_per_point_pc_map`,
   supplies BOTH a moving projection centre and a real deformation
   and is the killer. Consequence, recorded: with the conversion a
   group homomorphism (a conjugation), removing the phantom in
   homography space and removing it in Fe space now agree EXACTLY
   (measured 2.2e-16), so the plan-2.5 "correction applied after Fe
   conversion" mutant is an EQUIVALENT mutant of the corrected
   design; the non-equivalent form, which keeps the target PC in the
   conversion, dies at
   `test_hrebsd_geometry.py::TestFeFromHomography::test_correction_precedes_conversion`.
   The `correct=False` DIAGNOSTIC path keeps `PC_rel = PC_t - PC_ref`
   and `DD_t`, which is what V6's uncorrected arm reads.
3. **Signs are pinned by oracle, not by reading.** The exact signs
   of `deltaDD` and `alpha_s` (EMsoftOO's `alpha =
   (Dref - deltaDD)/Dref` with `deltaD = -stepy*sin(...)`,
   `mod_HREBSDDIC.f90:925-949`, sit in the unit-trap zone) are
   determined by the PC-shift phantom oracle (V6): identity-F
   patterns synthesized on a per-point-PC grid must produce (a)
   the closed-form phantom homography when the correction is OFF
   and (b) `Fe = I` everywhere to the noise floor when it is ON.
   The pinned signs are then recorded here with the date.
   **SIGNS PINNED 2026-09-07 (Stage A implementation gate): the
   DRAFTED set passes, unchanged.** On the `(2, 3)` phantom map
   with 400.0 unit steps (per-point offsets to `gamma_x = -11.43`
   px, `gamma_y = -5.37` px, `alpha_s - 1 = 8.06e-3`),
   `test_phantom_uncorrected` recovers the closed form
   `alpha_s = DD_target / DD_reference` (the ratio in THAT
   orientation), `gamma = PC_target - PC_reference` in binned
   pixels, `W_phantom = [[alpha_s, 0, gamma_x],
   [0, alpha_s, gamma_y], [0, 0, 1]]`, to 0.0070 px worst -- 6e-4
   of the phantom's own size. `test_phantom_corrected` then gives
   `Fe = I` to 1.70e-05 in the worst entry with the correction
   composed as `W_corr = W_phantom^-1 . W` (also the drafted side),
   against 8.06e-03 in the same entries uncorrected: the correction
   removes the strain phantom by a factor of about 475. A flipped
   DD ratio or a flipped `gamma` sign doubles the phantom instead
   of removing it and dies by three orders. Recorded in
   validation.md.
4. `sample_tilt`/`tilt`/`azimuthal`/`twist` all come from the
   `EBSDDetector` (D1.5); nothing is hardcoded.

### D7 -- Frame chain (frozen)

`Fe_hat` is measured in the DETECTOR frame. The chain to reported
quantities: `beta_det = Fe_hat - I` -> rotate to the sample frame
`beta_s = M^T beta_det M` with

**`M = diag(1, -1, 1) @ detector.sample_to_detector.to_matrix()`
-- DATED CORRECTION, 2026-09-07 (Stage B failing-tests gate).**
The drafted form of this clause wrote `beta_s = R^T beta_det R`
with the unflipped `R = detector.sample_to_detector.to_matrix()`,
which is wrong and is struck. `R` maps the sample frame into
kikuchipy's GNOMONIC detector frame, whose y points UP, while the
`Fe` the engine measures lives in the numpy array frame of D1.1,
whose y points DOWN; the two differ by the same reflection the
Stage A half-pixel measurement already recorded for the
coordinates (validation Recorded results entry 1, D1.1/D1.3, and
`test_hrebsd_engine.py::TestPcCentredFrame`, 1.8e-14 px over a
whole detector). Dropping the flip mirrors `beta13`, `beta23`,
`beta31` and `beta32` and flips two components of every reported
rotation vector, so it is not cosmetic. MEASURING TESTS:
`test_hrebsd_tensors.py::TestFrameChain::test_the_detector_frame
_is_the_y_down_one` (an independent library measurement from
`_get_direction_cosines_from_detector`, passing before any Stage
B code exists) and `::test_sample_to_detector_matrix_carries_the
_flip`; Stage A already used the composed matrix in
`test_hrebsd_engine.py`'s `impose_detector_frame_fe`, so every V3
and V4 deformation was imposed through it. Recorded in
validation.md with this date. The flip is a reflection and cannot
be absorbed into the detector's rotation (D1.5). Direction and
transpose PINNED by the pure-rotation oracle V4 and by
`TestFrameChain::test_the_rotation_uses_the_transpose`, exactly as
the Phase 5 forward-projection lock pinned `rotation_from_zyz` ->
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
  **MEASURING-TEST CORRECTION, 2026-09-07 (Stage B failing-tests
  gate).** The equality is measured ANALYTICALLY, by
  `test_hrebsd_tensors.py::TestSmallStrainFastPath::test_equality
  _with_the_polar_path_is_measured`, and not "over the V4 rotation
  sweep" through patterns: the difference between the two splits is
  a property of the split alone, given the same `F`, and is
  independent of how `F` was obtained, so patterns would add DIC
  noise to a pure-algebra quantity without adding coverage. The
  recorded enabling angle is the BISECTED crossing of the 1e-6
  criterion, not the largest angle of the sweep grid inside it: on
  the drafted grid the errors are 9.140e-08, 5.656e-07, 1.604e-06,
  3.733e-05, 1.479e-04 and 5.889e-04, so a grid readout would have
  recorded exactly 0.05 deg for a crossing near 0.0779 deg.
- **Pure-rotation strain floor of the CHAIN, MEASURED AND RECORDED
  2026-09-07 (Stage B failing-tests gate).** A pure lattice
  rotation leaks NO strain through the polar SPLIT (measured
  `|U - I|max` = 6.7e-16 at 5 deg), but it does not come back at
  zero through the whole chain, and the reason is structural, not
  numerical: the stored `Fe` is REDUCED by `Fe33`, which for a
  rotation through theta is an isotropic dilatation of
  `1 - cos(theta)` order (measured 7.13e-05 at 2 deg on the
  480 px oracle geometry), and the ninth-degree-of-freedom closure
  (D9) then supplies an isotropic part of its own. MEASURED
  reported `|strain|max` for a pure sample-frame rotation about
  the detector normal: 1.0154e-04 (deviatoric) / 7.2762e-05
  (traction free) at 1 deg, 4.0614e-04 / 2.9104e-04 at 2 deg, and
  2.5380e-03 / 1.8187e-03 at 5 deg. Under the deviatoric closure
  the ratio is EXACTLY `2/3` of `sym(R - I)` (algebra:
  `diag(c-1, c-1, 0)` closes to `diag(-(1-c)/3, -(1-c)/3,
  2(1-c)/3)`) and under traction free it is 0.478 of it. The
  validation V4 drafting seed `ROTATION_STRAIN_LEAK <= 1e-4 up to
  5 deg` is REFUTED by these numbers and is struck; the band is
  MTP and its recipe is the same sweep. This floor is a property
  of the reduction plus the closure, so it can never discriminate
  between the polar and the small-strain path: which split a
  public result took is pinned instead against the chain's own
  reported `beta` (`test_the_public_path_is_the_polar_one`), where
  the two candidates separate by 6.09e-04 at 2 deg.

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
   **Multi-phase maps and the returned shape, RECORDED 2026-09-07
   (Stage B failing-tests gate; two clauses the drafted text left
   implicit).** (a) A map of several indexed phases is segmented
   PER PHASE: an edge whose two points carry different phase
   identifiers never links, and each phase's angles use its own
   point group. A phase boundary is a grain boundary, which is the
   natural extension of the connected-components rule and needs no
   new decision. A Stage B draft instead raised a ValueError on a
   multi-phase map, which no requirement asks for and which would
   have narrowed the FROZEN DEFAULT `reference="auto"` (D11.3) to
   single-phase maps only, contradicting D9.6 ("the engine itself
   is phase-agnostic per grain"): a multi-phase map that Stage A
   indexes happily with an explicit reference would then have
   raised on the default path. That guard is struck. The
   single-phase restriction of D9.6 stays where it belongs, on the
   STRESS path. (b) The returned shape is ALWAYS two dimensional
   and comes from the row and column grids, never from
   `CrystalMap.shape`: MEASURED on the installed orix 0.14.2, BOTH
   a `(1, n)` and an `(n, 1)` navigation shape give `xmap.shape ==
   (n,)` with `ndim == 1`, so "a one dimensional map is a single
   row" would return `(1, ny)` for a COLUMN map and the exact
   shape comparison of the `"auto"` wiring would reject it. The
   grids settle which it is: a `(3, 1)` map has `row = [0, 1, 2]`
   and `col = [0, 0, 0]` and segments to `(3, 1)`. `hrebsd_kam`
   (D12) returns the same shape by the same rule.
   **(c) AND SO DOES THE ENGINE WIRING, corrected 2026-09-08 (Stage B
   adversarial review).** `EBSD.hrebsd_dic` was deriving the engine's
   two dimensional shape from the SIGNAL alone, `(1, n)` for any one
   dimensional scan, while the segmentation took its shape from the
   map's grids: a COLUMN line scan therefore passed the public shape
   guard and then raised `segment_grains returned labels of shape
   (3, 1), which must be the navigation shape (1, 3)` out of an
   internal function on the FROZEN DEFAULT `reference="auto"`
   (REPRODUCED on a three-point signal with `y = arange(3.0)`). The
   method now reads `xmap.row`/`xmap.col` for the one dimensional
   case, exactly as (b) freezes for `segment_grains`, and gives the
   single scan step to the ROW axis of a column scan. No separate
   guard is needed for a map which is neither: orix reports a
   diagonal three-point map as shape `(3, 3)` and the existing shape
   equality check rejects it (MEASURED).
   **(d) A PHASE WITH NO POINT GROUP IS WARNED ABOUT, added
   2026-09-08 (same review).** `Orientation(rotations, symmetry=None)`
   leaves the symmetry at C1, so `angle_with` returns the RAW angle
   and the symmetry reduction this decision requires silently does
   not happen. MEASURED: a 2 by 2 map of 0/90/0/90 degrees about z is
   ONE grain with `point_group="m-3m"` and TWO without it, and the
   difference reaches each point's reference pattern and every strain
   measured against it. There is no symmetry to invent, so the choice
   is between refusing such a phase and saying so: `segment_grains`
   emits a `UserWarning` naming the phase and segments on the raw
   angles. Refusing was rejected because a phase list without point
   groups is legal input to every other part of kikuchipy and the
   frozen default would then raise on it.
   **(e) SEVERAL ROTATIONS PER POINT, made consistent 2026-09-08
   (same review).** A map from dictionary indexing with `n_best > 1`
   carries `(n, k)` rotations. `segment_grains` already read the BEST
   of them; `hrebsd_strain_stress` raised instead, with a message
   naming `orientation_matrices`, a parameter of the private chain
   which the public caller never passed and which is not in its
   documented `Raises`. Both now take the best rotation, which is the
   map's own; the private `tensor_chain` keeps its strict `(n, 3, 3)`
   contract.
   **(f) A MAP WITH NO GRID IS NAMED, added 2026-09-08 (same
   review).** orix reports the shape `()` both for a one-point map
   and for a map whose points all sit at one scan position, and
   reading `row` on either raises `not enough values to unpack
   (expected 2, got 0)` from inside orix, which names nothing the
   caller passed. The three grid-shaped functions -- `segment_grains`,
   `hrebsd_kam` and `hrebsd_pc_shift` -- now read the grids through
   one shared helper which raises a `ValueError` naming the map
   instead. With that guard the one-by-one grid is unreachable, so
   the empty-edge branch of the private `_candidate_edges` is DEAD
   code and was pruned rather than tested, on the precedent of
   validation entry 23.
2. **Reference auto-selection** (per grain): the point maximizing
   pattern image quality computed internally with the existing
   `get_image_quality` kernel (`pattern/_pattern.py:698`) on the
   raw patterns -- deterministic, independent of which props the
   input xmap happens to carry; ties broken by lowest flat index
   (FROZEN). A `min_boundary_distance` refinement is a recorded
   possible v2 nicety, not built.
   **THE CANDIDATE SET IS THE MASKED-IN, LABELLED POINTS, corrected
   2026-09-08 (Stage B adversarial review).** The selection saw the
   whole map: `navigation_mask` reached the fit list and nothing
   else, so a pattern the caller had explicitly masked out could be
   chosen as its grain's reference and become the origin of every
   measurement in that grain -- while its own `homography` came back
   NaN with `converged=False`. DEMONSTRATED on `nickel_ebsd_small`:
   masking exactly the highest-quality point left
   `reference_index = [0 8 8 0 8 8 0 8 8]` naming it anyway. The mask
   is now threaded into the selection, which maximizes over the
   masked-in points of each grain. Two clauses go with it, both
   frozen here: `grain_id` is UNCHANGED by the mask (it reports the
   true label of every point, masked or not, which is what lets a
   caller see what was skipped), and a grain in which NOTHING is
   selectable keeps the unrestricted choice, since no pattern of such
   a grain is correlated and its index is never read -- the
   one-index-per-label pairing of D11.3 still wants an entry.
   **AND THE SCORING IS BLOCK-WISE, corrected the same date.** The
   selection materialised the ENTIRE pattern stack
   (`numpy.asarray(patterns)`) before the first fit, on the frozen
   default path: 576 MB on the shipped Si wafer and about 57 GB on a
   500 by 500 map of 480 by 480 patterns, which defeats the lazy
   design of D16 and is not what the public Memory note described.
   The kernel is per-pattern, so the fix costs nothing: candidates
   are read in blocks of about 64 MB, and the patterns outside every
   grain or masked out are not read at all. MEASURED with a
   block-counting Dask array: two of nine patterns loaded when two
   are asked for. The public Memory note gains the sentence.
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
   **GUARD AND PAIRING RULE, added 2026-09-07 (Stage A adversarial
   review).** The index array pairs POSITIONALLY with the SORTED
   UNIQUE labels, so labels may have gaps (`(0, 3, 7)` pairs
   `reference[1]` with label 3), and each index MUST lie inside the
   grain it serves -- a ValueError names the offending index, the
   label it carries and the label it was paired with. Without that
   guard a transposed or off-by-one array silently correlates one
   grain's patterns against another grain's reference while
   `grain_id` truthfully reports different grains, which no accuracy
   band can see. The `(row, col)` + `grain_labels` combination is
   unchanged and deliberate: one global reference serves every
   label, and only the reported `grain_id` comes from the labels.
   `misorientation_threshold` is threaded through but READ BY
   NOTHING in Stage A (it belongs to the `"auto"` segmentation of
   Stage B); the docstring says so rather than implying an internal
   segmentation that does not exist yet.
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
  **`order=1` AND `psi_max=None` CONFIRMED 2026-09-08; plan open
  question 4 is CLOSED** (Stage B implementation gate, validation
  Recorded results entry 45). The plan asked the V5 sweep to record
  "the noise/resolution trade" at orders 1 to 3 and to keep order 1
  unless refuted. MEASURED on the Si wafer, THERE IS NO TRADE --
  order 1 wins on both axes:

  | kernel | median KAM | std | cost, 100 points |
  |---|---|---|---|
  | `order=1` (8 neighbours) | **5.2402 mrad** | 0.1694 | 0.34 ms |
  | `order=2` (24) | 8.2715 | 0.5364 | 0.77 ms |
  | `order=3` (48) | 11.1350 | 0.5253 | 1.42 ms |

  The rotations are uncorrelated point to point on a strain-free
  crystal, so a wider kernel does not average the noise down, it
  reaches further into it: the reported KAM and its spread both grow
  while the spatial resolution gets worse and the cost quadruples.
  `psi_max` behaves exactly as documented: at 5.0 mrad it trims the
  median to 4.3685 mrad and at 1.0 mrad it drops EVERY pair on a map
  whose rotation floor is 12 mrad, leaving no finite point -- correct
  for a guard aimed at sub-grain boundaries, and the reason its
  default is `None`.
- Disorientation per pair: `angle(R_p R_q^T)` from the HR
  rotations (Stage B `rotation_vector` prop composed per point);
  within a grain relative to a common reference, so symmetry
  operators are unnecessary and never applied (documented).
  **CONDITIONING, MEASURED AND RECORDED 2026-09-07 (Stage B
  failing-tests gate).** That angle must NOT be evaluated as
  `arccos((tr - 1)/2)`: `arccos` has a square-root singularity at
  the identity and loses about eight significant digits at the
  1e-4 rad scale this field lives at. MEASURED on the V7
  constant-curvature oracle, the arccos form returns
  0.07499999980334485 mrad where the closed form is 0.075, a
  relative error of 2.6221e-09 -- three orders above the band the
  identity is asserted in, and enough to make the oracle and its
  own closed form mutually unsatisfiable. Use
  `arctan2(||skew(M)||/2, (tr(M) - 1)/2)`, which reproduces the
  closed form to 3.7e-16 relative. Both the module and the V7
  oracle carry the note, and the oracle pins its own conditioning
  before the implementation exists.
- Masking: pairs must share `grain_id` (FROZEN, always on -- the
  HR field is only defined within a grain); `psi_max` (mrad,
  optional, `None` FROZEN default) additionally drops pairs above
  the threshold (sub-grain-boundary guard). Points with no valid
  neighbor -> NaN.
  **THE THRESHOLD SIDE, PINNED 2026-09-08 (Stage B adversarial
  review).** "Above" is meant literally: the comparison is `<=`, so a
  pair sitting EXACTLY at `psi_max` is KEPT. This is the KAM analogue
  of the segmentation threshold side of D11.1, which is pinned, and
  it was not: the `<=` -> `<` mutant survived the whole suite. It is
  not a rounding-scale difference -- MEASURED on a one-pair map, the
  correct form returns the pair angle and the mutant returns NaN
  everywhere -- and the pin now feeds `psi_max` an angle the module
  itself reported, which is the only way the boundary is exact rather
  than a tolerance.
  Non-converged points reach this function already NaN, through the
  `rotation_vector` prop the tensor chain derives (D2.6), so no
  convergence flag is read here.
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
- **WHICH MISCALIBRATIONS THE RESIDUAL CAN SEE, made explicit
  2026-09-08 (Stage B implementation gate; validation Recorded
  results entry 42).** The model above is a DIFFERENCE,
  `PC_target - PC_reference` (the D6.2 closed form), so a GLOBAL
  offset of the projection centre cancels identically and is
  invisible here. MEASURED on the V5 wafer map: adding 0.02 to
  every point's `pcx` moves the mean of `residual_x` from
  9.97390698866289 px to 9.973906988662913 px, a relative
  2.3e-15, i.e. floating point noise. What the residual DOES see
  is anything that changes the differences: a wrong scan-step to
  `px_size` ratio (MEASURED: `px_size / 20` takes the modelled
  PCx drift from 20.0 px to 400.0 px and the residual mean to
  199.97 px, a factor of 20.05), a wrong sample tilt, and a real
  per-point departure from the beam-scan plane. This is a
  property of the frozen model and not a defect; it is recorded
  because the drafted V5 arm miscalibrated by a global offset and
  was therefore unsatisfiable, and is corrected with this date.
  The analytic pin of the signature, which offsets the MEASURED
  translations rather than the geometry, is
  `test_hrebsd_pc_shift.py::test_a_miscalibrated_projection
  _centre_shows_as_a_nonzero_mean` and is unaffected.
- **WHICH POINTS ARE MEASUREMENTS, corrected 2026-09-08 (Stage B
  adversarial review).** The maps are NaN at a point which failed,
  was masked out **or did not converge**, and the last of those three
  needs the `converged` prop, which is why `hrebsd_pc_shift` requires
  it beside `homography` and `reference_index`. Reading the
  finiteness of the homography alone is not enough and is exactly the
  trap D2.6 sets: a non-converged point KEEPS its finite last
  iterate. MEASURED on the V5 wafer route, 13 of 100 points did not
  converge, their translations were being averaged into the residual
  means with the 87 real fits, and they pulled `residual_x` from
  10.9956 px down to 9.9739 px -- a 10.2 per cent contamination of
  the one number this decision is read on, and of the recorded
  `SI_PC_RESIDUAL_MEAN_TOL`. Both are re-recorded with this date
  (validation entry 51).
- **REPRODUCING THE GEOMETRY THE RUN USED, documented 2026-09-08
  (same review).** A single-PC detector is still refused here, since
  this function cannot read a step-size unit (D6.1), but the
  docstring's remedy now names what the engine actually does rather
  than only naming `extrapolate_pc`: the anchor is the scan position
  of the FIRST grain reference, `divmod(reference_index.min(), nx)`,
  and not `[0, 0]`, and the steps must be in the micrometres
  `px_size` is measured in. The translation half of the model is a
  difference and does not feel the anchor; `scaling_model` does.
- **THE DIAGNOSTIC FIRING ON REAL DATA, recorded 2026-09-08.** On
  `kp.data.si_wafer()` with the UF-420 pixel size the residual
  means are 9.97 px and 9.36 px against a modelled drift of
  20.0 px, because the fits measure a median translation of
  0.026 px: the diagnostic is CORRECT and is reporting exactly
  what D13 built it to report, a geometry inconsistent with the
  measurement. The cause there is on the measurement side (the
  band-passed patterns retain a component which does not move
  with the beam; see the D4.1 record of the same date), so the
  recorded `SI_PC_RESIDUAL_MEAN_TOL` is a regression band on that
  dataset's behaviour and NOT evidence about how close to zero a
  well-conditioned map gets. (RE-MEASURED over the converged points
  alone later the same day, per the correction two bullets above:
  10.9956 px and 9.8403 px against the same 20.0 px modelled drift,
  the 87 real fits measuring a median translation of 0.045 px. The
  conclusion is unchanged and the band moves to 2x the larger,
  `SI_PC_RESIDUAL_MEAN_TOL = 2.2e01`.)

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
   (global sign pinned by V7 -- **DISCHARGED 2026-09-08 at the Stage
   C adversarial review, which found the claim undischarged: two
   transcriptions of one formula cannot see a global sign and every
   estimator sums moduli, so V7's own named rotation-field recipe
   `omega_3(x1) = kappa*x1` is now built and its SIGNED answer
   pinned. MEASURED AND RECORDED with it: this frozen convention is
   exactly MINUS the classical Nye/Pantleon tensor
   `alpha_ij = kappa_ji - delta_ij*kappa_kk`; on that field it gives
   `alpha_13 = -kappa` where the classical relation gives `+kappa`.
   Nothing shipped sees the sign, but a reader comparing the signed
   `nye_tensor` output with a paper must know it**) -- six more
   entries, all nine in
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
   **AMENDED 2026-09-08 (Stage C fix gate), no behaviour change:
   the default STANDS, and two measurements are recorded beside its
   argument because the argument alone misdescribes the operation.**
   (a) The replacement is not a filter. It makes the lower left of
   the detector-frame tensor exactly minus its upper right, so the
   detector-frame out-of-plane elastic shear strains
   `eps13 = (beta13 + beta31)/2` and `eps23` become IDENTICALLY
   ZERO: a measured elastic shear is traded for a quieter noise
   operator. Measured on the tutorial's analytically imposed field,
   both fall from 3.5e-04 and 5.5e-04 to exactly 0.0, the
   sample-frame field moves by up to 9.7e-04 against its own
   1.4e-03 amplitude, and the `"a5"` density rises to 2.3972e12
   m^-2, 8.0 % above the 2.2192e12 m^-2 of that field's own Nye
   content (validation entry 74). (b) On the
   ONE real data set measured, `kp.data.si_wafer()`, the fix RAISES
   the floor -- 1.1237e11 with it against 7.7823e10 without, x1.44,
   the OPPOSITE of the direction the 9.6x argument describes
   (validation entry 67). That data set's floor is set by a
   band-pass-surviving fixed-pattern component rather than by
   beta31/32 noise, so the argument does not apply to it and the
   measurement does not refute the argument; neither does it
   support it. The default therefore rests on the frozen geometry
   alone, which the `hrebsd_gnd` docstring and the tutorial now
   both say.
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
   rule). **CLARIFIED 2026-09-08 (Stage C adversarial review; no
   behaviour changes, and D2.6 already governs it): the pair rule
   above is not the whole NaN contract, because a central-difference
   pair never reads its own centre. A point whose OWN `beta` is
   non-finite is NaN in both derivatives, in `alpha` and in the
   density, whatever its neighbours hold -- that is D2.6's "NaN in
   every derived prop downstream; they are NEVER zeroed" applied to
   this path, and without it a pair-only implementation would report
   the full neighbourhood density at a point that never converged.
   Pinned at the `nye_tensor` level, not only at `hrebsd_gnd`
   (`test_hrebsd_gnd.py::TestNaNSafety`).**
   `burgers_vector_length` in METERS, required, no default
   (frozen: deriving b from the phase structure is a recorded v2
   nicety; silently guessing units is how prefactor bugs hide).
6. Maps plotted log10 in the tutorial; the docstring quotes the
   expected noise floor scale `rho_noise ~ sigma_beta/(b * step)`
   ~ 4-8e12 m^-2 at sigma_beta 1e-4, b 0.25 nm, step 100 nm
   (Jiang/Britton/Wilkinson 2013; Ernould thesis; theory report
   section 3.6) and the Si-wafer measured floor once recorded
   (V5/V7). **DISCHARGED 2026-09-08 (Stage C fix gate; the Stage C
   adversarial review found the second half undischarged -- the
   floor had been recorded at validation entry 67 and the public
   docstring still quoted only the literature class, which is the
   one number a reader must not be handed alone).** Both the
   `_gnd.py` module docstring and the `hrebsd_gnd` Notes now carry
   the measured 1.1237e11 m^-2, why it is BELOW the literature
   class rather than better than it (a step 2000x longer), the
   1.5633e11 m^-2 the same identity predicts from that data set's
   own rotation floor, and the statement that it is that data set's
   floor and never the method's. The log10 guard the docstring
   asserted the tutorial used, the tutorial now uses, at
   `np.log10(np.where(gnd > 0, gnd, np.nan))` (validation entry
   75).

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
  **CORRECTED 2026-09-07 (Stage A implementation gate,
  requirements D19): the drafted "~5 planes, ~4.6 MB" is refuted by
  measurement and is 3.6x too small.** The eight steepest-descent
  columns of D2.1 are themselves eight subregion sized f64 planes,
  so the real count is eleven planes plus the f32 coefficient
  plane: MEASURED `ReferenceState.memory_bytes()` = 17344512 B =
  **16.54 MB** at 480x480 with `border=0.05` (186624 subregion
  pixels), of which 11.94 MB is the steepest-descent block, 0.92 MB
  the f32 coefficients and 4.48 MB the reference and the two
  coordinate planes. The information message was already right, its
  whole-pattern upper bound printing 20.2 MB at this size; only the
  prose was wrong, and the two docstrings repeating it are
  corrected with this date.
  **RE-CORRECTED 2026-09-07 (Stage A adversarial review): 16.54 MB
  itself understated the figure by ten per cent.**
  `ReferenceState.memory_bytes()` omitted `reference_subregion`, the
  preprocessed subregion bounding box the D5 phase-correlation seed
  is measured on, which is a persistent attribute for the
  reference's whole life: 1492992 B at 480x480. MEASURED
  **18837504 B = 17.96 MB** over 186624 subregion pixels (11.94 MB
  steepest-descent, 4.48 MB reference plus the two coordinate
  planes, 1.42 MB reference subregion, 0.92 MB f32 coefficients), so
  the drafted "~4.6 MB" is 3.9x too small. `memory_bytes()` now
  counts every array the instance ALLOCATES and documents what it
  excludes and why: the subregion mask, the band-pass transfer
  function and the window are built once per RUN and shared by every
  reference, so counting them per instance would multiply one
  allocation by the grain count. The information message's model
  gains the same plane and prints **22.0 MB** at 480x480, which
  still bounds the measurement from above; a test now asserts that
  bound rather than leaving it to prose.
- **The `reference="auto"` selection is block-wise too, corrected
  2026-09-08 (Stage B adversarial review; D11.2 carries the
  measurement).** It reads each candidate pattern once before the
  first fit, in blocks of about 64 MB, so the peak it adds is one
  block and not the data set. The public Memory note says so, since
  it is the default path.
- Performance numbers are recorded baselines in validation.md,
  never merge gates (tech-stack.md:39); no hard floor is set for
  v1 (recorded; the spherical >= 2 pat/s/core floor is
  EMSphInx-scoped). Baseline recipe: patterns/s on the Si-wafer
  route at 480x480 and on `nickel_ebsd_large` at 60x60 (V5,
  validation Performance).
  **THE STAGE B TENSOR BASELINE IS THE DEVIATORIC ROUTE ONLY,
  recorded 2026-09-08 (same review).** Validation entry 46's
  `hrebsd_strain_stress` number is measured on entry 43's recipe,
  which passes `stiffness=None`, so `rotate_stiffness` is never
  reached. The stress path costs about 27x that: MEASURED on machine
  A at 2500 points, 7.21 ms deviatoric against 192.93 ms traction
  free, of which 182.44 ms is `rotate_stiffness`. The per-point
  `einsum` loop is KEPT deliberately -- it makes a stacked call the
  loop over the single-matrix one to the last bit, which a vectorised
  `optimize=True` call does not (MEASURED: 4.42 ms and agreeing to
  1.99e-13, comfortably inside `REDUCED_CLOSURE_TOL = 2.4e-06`, but
  not bitwise) -- and 190 ms on a map whose fits take hours is not a
  cost worth that. Recorded rather than optimised, per this
  decision's own rule that performance is never a gate.

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

**VERDICT 2026-09-07 (Stage A implementation gate): f32 bulk
storage CONFIRMED, no longer provisional.** Measured by
`test_dtype_ab_harness` on the V2 480 px batch (four random small
homographies, seed 15): worst corner-displacement recovery error
0.00628116603 px with f64 coefficients versus 0.00628116632 px with
f32, a degradation of 4.5e-08 of the f64 error against a criterion
of 0.10 -- six orders of margin, because the f32 coefficient error
(measured 2.4e-08 scale relative in V0's `KERNEL_F32_TOL` arm) sits
far below the cross-interpolator systematic the fit is limited by.
The saving is the point: 0.92 MB per reference on the coefficient
array (1.84 MB f64 versus 0.92 MB f32) and 17.34 MB versus 18.27 MB
of resident per-grain precompute at 480x480. Fit time is unchanged
within noise (0.0713 s f64, 0.0742 s f32 per 480 px fit). f64
accumulators stay non-negotiable and untouched.

**SCOPE OF THAT VERDICT, narrowed 2026-09-07 (Stage A adversarial
review).** The policy sentence above names three bulk-storage
categories -- "patterns, spline coefficients, gradient planes" --
but the A/B harness varies only `coefficient_dtype`, i.e. the single
0.92 MB coefficient array, which is 5 per cent of the 17.96 MB
resident state. The 11.94 MB steepest-descent block (66 per cent),
the reference vector and the two coordinate planes are hard-coded
f64 and were never A/B'd. The verdict is therefore recorded as
**"f32 SPLINE COEFFICIENTS confirmed, measured"**, with the
steepest-descent, reference and coordinate planes staying f64 BY
DESIGN: they are solver accumulands, not bulk storage, and D17's own
first clause makes f64 non-negotiable for those. The gradient planes
named in the policy are not resident at all -- `gradient_planes()`
output is consumed into the steepest-descent block and discarded --
so nothing is left provisional. Extending the harness to a separate
storage dtype for those planes would be a re-opening of D17, not a
discharge of it, and is not proposed.

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
  cross-checked references. **COMPLETED 2026-09-08 (Stage C fix
  gate; the review found two of the nine keys missing and no
  `:cite:` role anywhere in `src/`).** `ernould2022advances` (AIEP
  223 (2022) Ch. 2 -- the source seven module docstrings name as
  the primary convention reference) and `hardin2015analysis`
  (J. Microsc. 260 (2015) 73-85, the traction-free assumption)
  were added, so all nine keys are present. The keys are consumed
  by `:cite:` roles in the docstrings the API reference RENDERS --
  `EBSD.hrebsd_dic` Notes (Ernould 2020/2022, Ruggles 2018,
  Wilkinson 2006), `hrebsd_strain_stress`'s `closure` parameter
  (Hardin 2015) and `hrebsd_gnd`'s `estimator` and
  `enforce_antisymmetry` parameters and Notes (Pantleon 2008,
  Ruggles 2020, Jiang 2013) -- and by the tutorial's
  `<cite data-cite=...>` markers. The `_hrebsd/` module docstrings
  keep their prose citations: those modules are PRIVATE and their
  docstrings are never rendered, so a `:cite:` role in them would
  link nothing (recorded decision, not an omission).
  IF any code is later ported verbatim
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

### D20 -- Neighbour-seeded propagation (Stage D, commissioned 2026-09-09)

User go 2026-09-09 for plan section 9 item 4 (reliability-guided DIC).
Motivation, measured: iterations dominate deformed-zone cost (Si-indent
rim ~1.8 patterns/s at 80-500 iterations vs far field 10-12 patterns/s
at median 5-14; ledger entries 80, 82); a converged neighbour's
homography is a far better seed than the translation-only phase
cross-correlation there.

- **D20.1 API (FROZEN)**: one new keyword-only parameter on
  `EBSD.hrebsd_dic`: `seed_from_neighbors: bool = False`. The default
  False path is BITWISE-UNCHANGED current behaviour (pinned by the
  existing determinism tests re-run unmodified plus an explicit
  default-off bitwise pin against a pre-Stage-D result). No other
  public knob in v1; internal constants are MTP.
- **D20.2 Algorithm (FROZEN)**: with `seed_from_neighbors=True`, three
  phases inside `run_hrebsd_dic`:
  (1) PASS 1: the existing per-point phase-XC-seeded fit, with the
  iteration budget capped at `min(max_iterations, PASS1_CAP)` where
  `PASS1_CAP` is an internal constant, MEASURED-THEN-PINNED on the
  Si-indent data (drafting candidate 50: far-field median is 5-14
  iterations, so the cap must catch essentially every easy point;
  the measurement is the fraction of pass-1 conversions lost at the
  cap vs the full budget).
  (2) CASCADE ROUNDS: round r fits, in parallel, every not-yet-
  converged, unmasked point that has at least one converged
  SAME-GRAIN neighbour in its 8-neighbourhood from rounds < r
  (pass 1 counts as round 0). The seed h0 is the converged
  neighbour's homography, chosen as the one with the LOWEST residual;
  ties broken by the FROZEN neighbour offset order
  ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)).
  The raw homography is copied without PC-frame transport: the
  neighbour PC differs by ~5e-3 px per step on real data (ledger
  entry 80 spans), a bound the build must MEASURE and record as the
  justification. Cascade fits run at the FULL `max_iterations`
  budget. Rounds repeat until a round converts nothing new.
  (3) RESCUE PASS: every point still unconverged gets ONE fit at the
  full budget from its best available seed (lowest-residual converged
  neighbour if any exists by then, else its own pass-1 last iterate,
  else the phase-XC seed); the result keeps the D2.6 non-converged
  semantics if it still fails. No point is ever left with a
  cap-truncated pass-1 iterate as its final answer.
- **D20.3 Determinism (FROZEN)**: the seed of every fit is a pure
  function of results from STRICTLY EARLIER phases/rounds and frozen
  tie rules; nothing depends on intra-round scheduling or chunking.
  Pins: two seeded runs bitwise-identical; chunksize invariance
  bitwise; lazy == eager bitwise (the Stage A pin extended to the
  seeded path).
- **D20.4 Isolation (FROZEN)**: seeds never cross `grain_id`
  boundaries; masked points neither seed nor get seeded nor get
  fitted; the reference point itself is round 0 by construction.
- **D20.5 Props (FROZEN)**: one new int32 prop `seed_round`: 0 =
  converged in pass 1 (or the reference point), r >= 1 = converged in
  cascade round r, -2 = converged only in the rescue pass, -1 = never
  converged or masked. D15.6's Stage A list is amended by this entry
  for seeded runs only; with `seed_from_neighbors=False` the prop is
  ABSENT (default path emits exactly the pre-Stage-D prop set).
- **D20.6 Equivalence oracle (MTP)**: wherever the independent
  (default-path) fit converges, the seeded run's answer agrees within
  `SEED_EQUIVALENCE_TOL` (corner-displacement metric, measured then
  pinned at ~2x on the deformed-master oracle map and the Si sub-map)
  -- seeding may only change HOW the optimum is reached, never WHICH
  optimum, on points both paths solve.
- **D20.7 Performance (recorded, never a gate)**: the Si-indent rim
  patch (ledger entry 80's rows 125:145, cols 115:135) re-measured
  seeded vs default; the plan 9.4 projection is 3-8x on deformed
  zones. Recorded honestly whatever it measures, with a full-map
  wall-time row if run.

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
- **AMENDED 2026-09-07 (Stage A implementation gate,
  requirements D19): two of those drafting seeds are refuted by
  measurement and are superseded by the pinned values.** (a) "up
  to 5 deg" in the pure-rotation seed is not reached at the frozen
  DEFAULTS: the measured capture range is 2.0 deg with the phase-XC
  seed and 3.0 deg from the exact one (D5, recorded), and at 5 deg
  the preprocessed pair's ZNCC is -0.056; the V4 in-plane cases are
  pinned at 0.1, 1.0 and 2.0 deg instead.
  **WITHDRAWN 2026-09-07 (Stage A adversarial review): the clause
  "so no conformant implementation of the frozen D2/D4/D5 design
  converges there" is FALSE and is struck.** The limit belongs to
  the frozen `filter_cutoffs=(0.05, None)` default, which is a
  PUBLIC keyword of the same frozen signature, not to D2/D4/D5.
  MEASURED with `filter_cutoffs` the only change: `(None, None)`
  recovers 2.5, 3.0 and 4.0 deg (15, 18 and 41 iterations) and
  improves the 2.0 deg error tenfold, and the 4.0 deg pair's ZNCC is
  +0.196 rather than -0.069, so the decorrelation is the filter's
  too (D4.1, recorded there with the full table). What survives is
  the conditional statement, which is what the `hrebsd_dic`
  docstring now carries: 2.0 deg at the default band-pass, 4.0 deg
  with none. The V4 pins stay at the defaults they gate. (b) The V2 warp-recovery seed was 2 to 10 times looser
  than achievable: the MEASURED worst corner-displacement error is
  0.0124 px at 480x480 against the 0.1 px the seed carried, so the
  band is pinned at 0.025 px. The 480x480 rotation error is
  8.3e-06 rad against a 1e-5 rad seed which left no margin at all,
  pinned at 1.7e-5. The Stage B seeds (strain 2e-4 per component,
  GND 1 %) are untouched and stay MTP for their own stages.
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
