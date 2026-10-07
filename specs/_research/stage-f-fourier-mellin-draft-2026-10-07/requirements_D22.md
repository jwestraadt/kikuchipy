<!--
INSERTABLE BLOCKS for specs/2026-09-07-hrebsd-dic/requirements.md,
drafted 2026-10-07 by the Stage F spec workflow and REVISED the same
day after the two-critic spec review (plan 12.5 carries the
disposition table; no repo file was edited). Block 1 is the new
decision entry: insert it immediately after the last D21 bullet
(D21.18, "super-resolution (plan 9.7).") and before the "## Context"
heading. Blocks 2 to 11 are dated amendments of existing entries;
each names its anchor. Ledger numbers 200 to 216 are provisional
(the main session renumbers them on splice, and every "ledger 2xx"
reference below moves with them). Plan open questions are FQ1 to
FQ14 (renamed from F1 to F14 at the review, so that they never
collide with the V9 fixtures F1 to F7).
-->

<!-- ===================== BLOCK 1: new entry D22 ===================== -->

### D22 -- Fourier-Mellin rotation initial guess (Stage F, commissioned 2026-10-06)

User go 2026-10-06: Johan extended his overnight waiver to Stage F
("continue with the merlin guess implementation after the gpu
implementation is done"; plan 11.4 approval record, item 20 carried
to this spec gate). Narrowed 2026-10-07 by Johan's own instruction:
the Stage F SPEC is written and committed, then work STOPS before
any Stage F failing test or code; the Stage F plan gate is PENDING
his review, and the 2026-10-06 waiver no longer covers it, so no
item below is approved yet. Motivation, measured: the
translation-only phase-XC seed of D5 captures 2.0 deg of pure
in-plane rotation on
the 480 px oracle and fails from 2.5 deg at the default budget of 50
(ledger 202); on rigidly rotated REAL Si-indent patterns WITH THE
DEFAULT BAND-PASS `(0.05, None)` it needs 19 and 124 iterations at
1 and 2 deg and fails from 3 deg at a budget of 200, while with
`(None, None)` -- the production Si-indent route -- it reached 6 deg
(107 to 168 iterations at a budget of 500) and failed at 10 (ledger
203). So on the Si map, whose largest input twist is 2.6 deg (ledger
212), the real-data case for FM under `(None, None)` is iteration
cost and the D5 zero-shift anchor (FQ8), not a capture failure; the
capture failure is the default band-pass's and the synthetic
oracle's. With the FROZEN recipe of D22.3 to D22.5 (the V8 angle,
the partial row, the acceptance) every oracle case measured end to
end converged: pure twists of -5 to 5 deg at `border=0.05` and of 8
to 30 deg and -20 deg at `border=0.15`, in 2 to 3 iterations to the
error the exact seed reaches; the four projection-centre-shift cases
in 2 to 3; the combined rotations in 6 to 9 (ledgers 214, 216). The
second prototype's whitened per-radius recipe (NOT this one; ledgers
202(ii), 203(ii)) converged a 113-case synthetic sweep to 30 deg in
1 to 9 iterations and every rigidly rotated real case from -6 to 10
deg; those numbers motivate, the V8 ones pin. On the Si-indent map
the input orientations put the twist about the detector normal above
1.5 deg on 336 fitted points outside the crater (about 0.6 per cent),
up to 2.6 deg (ledger 212). D21.12 leaves this regime to the CPU
cascade; D22 covers it per point, on both backends, without
propagation risk.

Method sources (carried here because the scratchpad does not
survive, D19): Ernould et al., Acta Mater. 191 (2020) 131-148,
doi:10.1016/j.actamat.2020.03.026, section 2.3 and appendix C (FMT-CC
rotation, de-rotation, FT-CC translation, partial and complete
initialisation); Ernould et al., Ultramicroscopy 221 (2021) 113158,
doi:10.1016/j.ultramic.2020.113158, appendix E (the TARGET is
de-rotated; eq. E.1, W0 = R(theta) T(t); eqs. E.2-E.3, the complete
form's lattice rotations read from the translation corrected for the
rotation centre); C. Ernould, PhD thesis, Universite de Lorraine
(2020), 2020LORR0225, chapter III.2 (polar sampling of the upper
half-plane, the 1D radial-mean signature, the sub-sample peak) and
IV.2.4 (seed accuracy 0.25 deg below 2 deg disorientation, 0.5 +-
0.25 deg to 14 deg); B.S. Reddy and B.N. Chatterji, IEEE Trans.
Image Process. 5 (1996) 1266-1271 (the 180 deg ambiguity of |F|); L.
Moisan, J. Math. Imaging Vis. 39 (2011) 161-179 (the
periodic-plus-smooth alternative); H.S. Stone et al., J. Vis.
Commun. Image Represent. 14 (2003) 114-135 (rotation-dependent
aliasing near Nyquist). Physics, D1.3 frame: a lattice rotation about
the DETECTOR NORMAL is an exact rigid image rotation about the PC at
any angle, h = (c-1, -s, 0, s, c-1, 0, 0, 0); rotations about the
in-plane axes translate the PC point by DD tan(w) px, add a
perspective term and rotate the pattern LOCALLY by
(w1 x + w2 y)/(2 DD), so an FM angle measured over a subregion whose
centroid is off the PC carries that bias (measured within 0.004-0.06
deg of the formula on the oracle; ledger 201) and the subregion sees
a pseudo-scale (3/2)(w2 xbar - w1 ybar)/DD of 1.2 to 5.7 per cent
(ledger 201), against elastic strains of order 1e-3.

Johan's recorded notes (plan 11.3 (i) to (viii), 11.4 item 20) are
binding: reuse the target FFT; a per-reference precomputed spectrum
signature as a `SeedState` extension (his note says log-polar; the
radial axis stays linear and is averaged out, D22.2 and D22.3);
de-rotation through the existing warp kernel; a gated seed with a
pre-fit `CrystalMap` misorientation gate at about 1.5-2 deg and a
post-fit retry as the safety net; rotation only; parity on converged
results within a band; a failed FM estimate returns the translation
row; on the CPU only the routed points pass through the numpy seam.
Two recorded deviations, each with its reason: the radial axis
(D22.2) and the CPU route's per-point P = 1 seeding inside the chunk
function instead of a batched host pre-pass (D22.10). Every number
below is from the THROWAWAY spec-gate prototypes (ledger 200 to
216); nothing of them is committed, and each prototype number is
tagged with the recipe that produced it where that recipe is not
the frozen one. Items marked FROZEN are frozen at this spec gate,
subject like everything here to Johan's review at the plan gate;
items marked RECORDED DEFAULT stand as written until Johan approves
or changes them at the Stage F plan gate (plan 12.4; PENDING
Johan's review, not pre-accepted); MTP numbers are filled at the
Stage F gates.

- **D22.1 API (FROZEN)**: one new keyword-only parameter on
  `EBSD.hrebsd_dic` and on `run_hrebsd_dic`, `fourier_mellin: str =
  "off"`, immediately after `seed_from_neighbors`. The frozen order
  becomes `..., step_scale, seed_from_neighbors, fourier_mellin,
  navigation_mask, backend, chunksize, verbose`, every one keyword
  only (dated D15.4 amendment). Slot, recorded: "immediately after
  `backend`" is NOT available, because D21.1 freezes `chunksize`
  immediately after `backend` and V9(a) pins it on both entries
  (`test_hrebsd_gpu.py:2513, 2532` at f297867e); after `chunksize`
  would put a result-affecting algorithm knob among the orchestration
  knobs; after `seed_from_neighbors` keeps the two seeding knobs
  together (the D20.1 precedent). Values, case-sensitive: `"off"` --
  the translation-only seed of D5, exactly as before; `"auto"` -- the
  FM seed on the points the pre-fit gate routes (D22.7), plus one
  post-fit retry pass (D22.8); `"always"` -- the FM seed evaluated,
  with the acceptance of D22.5, on every fitted point, plus the same
  retry pass (it then retries only the points whose acceptance kept
  the translation row and whose fit did not converge). Anything else,
  a bool included, raises `ValueError` in the D21.1 message shape:
  "fourier_mellin {fourier_mellin!r} not in the list of supported
  values ['off', 'auto', 'always']". DEFAULT `"off"`,
  BITWISE-UNCHANGED current behaviour on both backends: no FM state
  is built, the gate is not computed, no retry runs,
  `SeedBatch.extras` and `SeedBatch.outputs` stay empty, the packed
  rows keep their width of 12 and no new prop is written; pinned by
  the pre-Stage-D literal pins re-run unmodified, the Stage E
  `backend="cpu"` == no-keyword pin, the V9 `extras == {}` asserts
  (`test_hrebsd_gpu.py:2975, 5102` at f297867e) and a new
  `fourier_mellin="off"` == no-keyword pin on both backends
  (V10(a)). The default is a RECORDED DEFAULT (plan open question
  FQ3: `"auto"` once the whole-map record shows no point made worse).
  Check order, frozen (dated D21.1 amendment): inside
  `run_hrebsd_dic`, after the `backend` string check of D21.1 and
  before every `"gpu"`-only check: (1) the `fourier_mellin` value
  check; (2) the D22.11 combination raise; (3) under `"auto"` only,
  an `xmap` of `None` raises `ValueError` ("fourier_mellin='auto'
  requires the input CrystalMap; pass fourier_mellin='always' to
  seed every point without it") -- reachable through the engine
  entry only, since `EBSD.hrebsd_dic` always passes its `xmap`. All
  three fire before the gate of D21.2, reference resolution and any
  pattern read. The FM settings never enter `fit_options`, which
  `run_hrebsd_dic` forwards to `fit_pattern` as keywords
  (`_engine.py:1212-1217` at f297867e). `EBSD.hrebsd_dic` forwards
  the keyword by name to `run_hrebsd_dic` (it builds the engine call
  keyword by keyword, `ebsd.py:3460` at f297867e) and copies the two
  D22.9 props into the returned map (V10(a) pins both).
- **D22.2 Scope of the estimate (FROZEN; Johan note (vi))**:
  rotation about the detector normal only; no scale, no log-polar
  axis. Reasons, measured or derived: the physical scale changes
  between a grain's patterns (DD change along the beam scan, elastic
  strain) are of order 1e-3, below a log-polar bin; the off-PC
  pseudo-scale of in-plane-axis rotations is perspective, not
  isotropic scale, so a scale seed would give the wrong shape (ledger
  201); the 1D radial-mean signature that makes the step cheap exists
  only with scale fixed. In-plane-axis rotations mainly TRANSLATE the
  pattern, which the translation step catches; their projective
  remainder stays IC-GN's, as now.
- **D22.3 The FM angle (FROZEN recipe; constants RECORDED DEFAULTS,
  each MTP)**: per slot, from the REUSED `target_spectra` of
  `seed_spectra` (Johan note (i); never modified in place), and per
  reference from `SeedState.reference_spectrum`, which is the same
  transform of the reference crop:
  1. Edge treatment: the periodic Hann window applied EXACTLY in the
     frequency domain, separably per axis, `X_w[k] = X[k]/2 -
     (X[k-1] + X[k+1])/4` with circular indices -- the 3-tap DFT of
     the periodic Hann window -- so `X_w` equals `fft2` of the
     windowed zero-mean unit-norm crop to rounding (measured 1.6e-16
     relative; ledger 204) with no second 2D FFT. Dense or evaluated
     only where step 3 reads it: implementation freedom. Why
     required: the crop boundary is fixed to the detector, not to
     the content; the reused UNWINDOWED spectrum locks to zero
     rotation under a detector-fixed background without the
     high-pass (errors of -0.97 to -7.99 deg at 1 to 8 deg, and 80
     deg worst with 8 of 9 cases above 1 deg on the head-to-head;
     ledgers 204, 213), and the production Si route runs
     `filter_cutoffs=(None, None)`.
  2. Magnitude: `m = log1p(|X_w|)`, unit scale (the crop has unit
     norm, so `|X|` is of order 1; no per-slot median).
  3. Polar sampling of the upper half-plane (|X| of a real image is
     point symmetric), exact formulas: `FM_N_THETA = 360` angles
     `theta_k = k pi / 360`, k = 0 .. 359, over [0, pi); with `m =
     min(sr, sc)`, `n_rho = floor((FM_RHO_MAX - FM_RHO_MIN) m +
     1e-9) + 1` radii `rho_j = FM_RHO_MIN + j / m` cycles/px, j = 0
     .. n_rho - 1, `FM_RHO_MIN = 0.02`, `FM_RHO_MAX = 0.40` (165 radii
     at 432x432, 175 at 460x560; the 1e-9 guards the floor against
     rounding); each sample in PHYSICAL frequency, `(fx, fy) = rho_j
     (cos theta_k, sin theta_k)`, read at the fractional bin
     coordinates `(fx sc, fy sr)` (column, row) of the UNSHIFTED
     spectrum with bilinear weights over the four neighbouring bins,
     indices modulo `(sr, sc)` -- never bin units on both axes, which
     shears a non-square crop (by 2.3 per cent of the angle on real
     Si with the method-study recipe, ledger 208; by 0.34 deg at 8
     deg, 1.03 at 15 and 2.29 at 20 on the G3 synthetic with this
     recipe, against at most 0.027 deg in physical frequency; ledger
     216). The evidence grid of ledgers 213 and 214 was `linspace(
     0.02, 0.40, round(0.38 m) + 1)` (176 radii at 460x560); the two
     grids give angles within 0.0005 deg on G1 and G3 (ledger 216).
     The look-up table, frozen form: `fourier_mellin_lut(sr, sc) ->
     (indices, weights)`, `(n_theta, n_rho, 4)` int64 flat indices
     into the `(sr, sc)` spectrum and `(n_theta, n_rho, 4)` float64
     weights (237600 entries at 432x432, 252000 at 460x560), built
     ONCE PER RUN on the host and cached by crop shape (every
     reference of a run shares the D5 bounding-box shape), uploaded
     once per device session.
  4. Profile: the mean over radius of each angle's samples (Ernould's
     1D reduction), then zero mean and unit norm per slot; a gather
     and a fixed-shape sum, no scatter and no atomics.
  5. Correlation: the circular ZNCC `c[k] = sum_j p_ref[j] p_tgt[j +
     k]` by length-`FM_N_THETA` FFTs (exact for a pi-periodic
     profile, so no window); output index k carries lag `k` for `k <
     n/2` and `k - n` otherwise, i.e. lags in [-n/2, n/2).
  6. Peak, frozen name `fourier_mellin_peak(xp, correlation,
     search_deg) -> (theta_deg, peak)` on a `(P, n_theta)`
     correlation: the argmax over the output indices whose lag has
     `|lag| * 180 / FM_N_THETA <= FM_SEARCH_DEG = 30`, ties to the
     LOWEST OUTPUT INDEX (the FFT order lag 0, 1, ..., n/2 - 1, then
     -n/2, ..., -1, so `+k` wins a tie with `-k`) on both
     namespaces; a parabolic sub-bin offset `delta = (c[k-1] -
     c[k+1]) / (2 (c[k-1] - 2 c[k] + c[k+1]))` from the RAW circular
     neighbours (values outside the window included) when the
     denominator is negative, else 0; `theta_hat = (lag + delta) *
     180 / FM_N_THETA` deg. Sign, frozen: `theta_hat` is the rotation
     about +z of D1.5, i.e. the row's `h21 = +sin(theta_hat)`
     (measured on the oracle, ledger 214). The peak value `c[k]` is a
     diagnostic only and is never thresholded (0.24 to 0.82 on real
     deformed rim points across the recipes tried, where the seed
     still helps; ledgers 205, 213).
  7. The 180 deg ambiguity is resolved by the search window (a
     subset of the "nearest" rule, theta in [-90, 90)); there is NO
     two-candidate arbitration. Measured: arbitration by the
     criterion chose the 180 deg candidate on 3 of 32 real rim
     points under `(None, None)` (the smooth background is nearly
     point symmetric about the PC), two of which then converged about
     1100 px away (ledger 209); no estimate of any variant on any
     set exceeded 30 deg (ledger 213). The window also excludes the
     60 and 90 deg pseudo-symmetry aliases of 3-, 4- and 6-fold zone
     axes. A rotation beyond 30 deg cannot be captured, by
     construction; at 30 deg exactly the V8 recipe measured 29.931
     deg and converged (ledger 216).
  Recipe choice, measured on identical inputs (ledger 213): the
  radial-mean ZNCC family was the only one with no error above 1 deg
  on any of 11 synthetic and real sets; the whitened per-radius
  angular phase correlation had a better median on the deformed rim
  but outliers of 41 to 73 deg at neighbouring settings (ledger
  213). Accuracy of this recipe (ledgers 213, 216): at most 0.12 deg
  on the noise-free G1 oracle and 0.027 deg on G3, 0.15 deg under a
  detector-fixed background at `(None, None)`, 0.037 deg on rigidly
  rotated real Si, and within 0.61 deg of the fitted in-plane
  rotation on deformed real rim points (the in-plane-axis bias and
  strain; a seed, not a measurement of w3). Recorded limitation: the
  angle is read from the spectrum of the WHOLE bounding box,
  dead-band pixels included, so a D4 dead-band cross of constant
  content in both images puts identical energy on the frequency axes
  and pulls `theta_hat` towards 0; measured with an 8 px cross on
  G1, errors of up to 0.12 deg against 0.02 without, at both filter
  settings (ledger 216), far inside the IC-GN basin; no axis masking
  is applied. The alternatives (amplitude instead of log, a [0.05,
  0.20] band, the Moisan correction instead of the window, masking
  the look-up samples within one bin of the axes) stay open under
  plan open question FQ4.
- **D22.4 De-rotation, translation and the starting row (FROZEN;
  the partial form a RECORDED DEFAULT)**:
  1. De-rotation through the existing warp kernel (Johan note
     (iii)), frozen name `fourier_mellin_derotate(ctx, batch,
     fm_state, theta_deg) -> (crops, ok)`: carried matrices
     `R(theta_hat)`, `(P, 3, 3)` float64, the rotation about the
     grain reference PC (the D1.3 origin), passed to
     `ctx.kernels.gather(box, batch.coefficients, matrices)`, where
     `box` is a per-reference resident whose subregion is EVERY pixel
     of the D5 bounding box (dead-band pixels included, because the
     D5 seed correlates the whole crop), so the result reshapes to
     the `(P, sr, sc)` float64 crop `D(xi) = target(R xi)`; `ok` is
     the per-slot conjunction of the D21.6.1 coordinate flags (a
     pure rotation makes no non-finite coordinate; the flags are
     honoured anyway).
  2. Translation, frozen name `fourier_mellin_translate(ctx, crops,
     seed_state) -> rows_t`: the Stage E phase-XC code UNCHANGED on
     the de-rotated crops (zero-mean unit-norm, `fft2` at
     `seed_precision`, the guarded cross-power with
     `reference_spectrum`, `ifft2`, the argmax, the upsampled DFT at
     `upsample_factor`), returning the `(P, 8)` float64 translation
     rows the Stage E seed would return for those crops; `t = (rows_t
     [:, 2], rows_t[:, 5])`, with `D(xi + t) = reference(xi)`. The
     module reaches the Stage E code through a function-scope import
     of `_batched` (the import direction of D22.6).
  3. The row, RECORDED DEFAULT the PARTIAL initialisation of
     Ernould 2021 eq. E.1: `W0 = R(theta_hat) T(t)`, i.e. `h_FM = (c
     - 1, -s, c tx - s ty, s, c - 1, s tx + c ty, 0, 0)` --
     calibration free, no DD and no frame algebra beyond D1.3. The
     composition order is frozen: the PC point moves by `d = R t`,
     never by `t` (the `T(t) R` mutant is off by `2 sin(theta/2)
     |t|`, 0.90 to 2.66 px on the projection-centre-shift fixture;
     ledger 210). Ernould's rotation-centre corrections (eqs.
     E.2-E.3) vanish, because the de-rotation is about the PC. The
     COMPLETE initialisation (the translation read as two lattice
     rotations through D6) is NOT adopted. Its two orderings are both
     valid parameterisations -- Ernould's eqs. E.2-E.3 with the
     rotation centre at the PC read the lattice rotations from `d =
     t`, the prototype from `d = R t`; they differ at second order
     (24.47 against 24.52 px in ledger 210(iii)) -- and the word
     "mutant" is reserved for the partial row's `T(t) R`. The
     complete form saved iterations where the translation is a
     lattice rotation (4 against 6-9 on the oracle's combined
     rotations; 1616 against 1660 iterations on 16 unfiltered real
     rim points; under `(0.05, None)` the same 5 of 8 rim points
     converted in 35-67 against 87-161 iterations; ledgers 214, 216)
     but seeded 11.1 to 24.5 px off where the translation is a
     projection-centre shift, against 0.08 to 0.22 px for the partial
     row, which then converged in 2 to 3 iterations to 0.0011-0.0027
     px (ledgers 210, 216). Plan open question FQ5 keeps it, with a
     PC-shift-aware variant, for measurement.
- **D22.5 Acceptance and failure semantics (FROZEN)**: on every
  route-1 slot (D22.6) the translation row `h_T` (the Stage E row,
  computed first for every slot of the sub-batch) and `h_FM` are
  both evaluated by the D2.7 criterion AT THE SEED -- the IC-GN's own
  objective, frozen name `fourier_mellin_criteria(ctx, batch,
  fm_state, rows) -> (P,)` float64, through `kernels.gather` on the
  grain's SUBREGION resident (`fm_state.resident`, D22.6), the K
  shifts computed exactly as the lockstep computes them at its start
  (today the mean of each slot's preprocessed subregion,
  `_batched.initial_shifts`), and `kernels.final_criterion` under the
  D21.4 rules. On the CPU route it agrees with `fit_pattern(state,
  target, h0=row, max_iterations=0)["residual"]` within
  `NUMPY_PARITY_RESIDUAL_RTOL` (the numpy twin's two-pass reduction
  against the BLAS dot; ledger 103), never bitwise; tests read the
  decision from the `"fourier_mellin_applied"` output, never
  re-derive it through `fit_pattern`. `h_FM` is kept if and only if
  it is finite throughout and its criterion is finite and STRICTLY
  lower than `h_T`'s, both as computed in the slot's own namespace;
  ties keep `h_T`, a non-finite FM criterion keeps `h_T`, and so
  does a non-finite `h_T` criterion with a finite `h_T` (the
  comparison is False against NaN; kept deliberately, since a
  translation row whose own criterion cannot be evaluated is an
  unusable crop, not a lost race). Measured with the frozen partial
  row (ledgers 214, 216): on the oracle `h_FM` won every routed
  case (criterion 0.001-0.71 against 0.67-2.05 for `h_T`) except a
  pure 3 deg rotation about the detector x axis, where the rows were
  within 0.004 and both fits took 9 iterations; on 16 unfiltered
  real rim points the partial row won 13 (0.455-1.403 against
  1.058-1.856) and lost 3 where the rows were nearly equal (fits
  91/91, 122/123, 134/134 iterations), and the accepted rows cut the
  iterations from 2002 to 1660 (-17 per cent; all 16 converge to the
  same residual either way); under the default band-pass it won the
  5 of 8 points it went on to converge (87 to 161 iterations; the
  translation seed converged none in 200) and lost the 3 that
  neither row converged. The complete row gave the same accept and
  convert counts (ledger 214). The phase-correlation peak is NEVER
  an acceptance signal: on real rim points the detector-fixed
  zero-shift peak (0.09-0.10) beats the true Kikuchi peak
  (0.012-0.025; ledger 207). Failure semantics (Johan note (vii),
  D21.5): a non-finite `h_T` is returned unchanged and is never
  rescued by a finite `h_FM` (it means the crop itself is unusable);
  an FM estimate that fails for a slot (a non-finite or zero-norm
  profile, a non-finite `theta_hat`, a de-rotated crop with a false
  `ok` or a non-finite pixel, a non-finite `t` or `h_FM`) returns
  `h_T` for that slot with `applied` False; the FM stage never
  introduces a non-finite row and never raises for a per-slot
  failure. Forced slots (route 2, the retry, D22.8) skip the
  acceptance, because the translation row has already failed there;
  a forced slot whose FM estimate fails, or whose `h_T` is
  non-finite, is reported as not applied and is NEVER FITTED (D22.6,
  D22.8).
- **D22.6 The seam extension (FROZEN; D21.5 clause (iv))**: the two
  seam signatures and every existing field are unchanged; every
  extension is a NEW field or a NEW `extras` input key. New names,
  frozen so that the failing tests have stable targets and patch
  points (the D21.9.5 precedent), in a new private module
  `_hrebsd/_fourier_mellin.py` (plain kikuchipy GPL header):
  `FourierMellinState`; `build_fourier_mellin_state(ctx, state,
  resident) -> FourierMellinState | None` (`None` when the reference
  profile is non-finite or of zero norm: that grain then runs as
  under `"off"`, is never routed and never retried);
  `fourier_mellin_lut` (D22.3.3); `fourier_mellin_angles(ctx,
  target_spectra, fm_state) -> (theta_deg, peak)` (each `(P,)`
  float64 in `ctx.xp`, testable alone); `fourier_mellin_peak`
  (D22.3.6); `fourier_mellin_derotate` and `fourier_mellin_translate`
  (D22.4); `fourier_mellin_criteria` (D22.5);
  `fourier_mellin_rows(ctx, batch, target_spectra, seed_state, h_t,
  route) -> (rows, angle, applied)` (the whole FM branch on one
  sub-batch, `(P, 8)` float64, `(P,)` float64, `(P,)` bool, in
  `ctx.xp`); the gate functions of D22.7
  (`twist_about_detector_normal`, `fourier_mellin_routes`); and the
  module constants `FM_N_THETA`, `FM_RHO_MIN`, `FM_RHO_MAX`,
  `FM_SEARCH_DEG` and `FM_GATE_DEG`. Call-time seams, frozen (the
  D21.9.5 rule): `seed_homographies` reaches `fourier_mellin_rows`
  through the `_fourier_mellin` module object; `fourier_mellin_rows`
  reaches `fourier_mellin_angles`, `fourier_mellin_peak`,
  `fourier_mellin_derotate`, `fourier_mellin_translate` and
  `fourier_mellin_criteria` through its own module globals at call
  time; `run_hrebsd_dic` reaches `twist_about_detector_normal` and
  `fourier_mellin_routes` through the `_fourier_mellin` module
  object; the CPU route reaches `_batched.seed_spectra` and
  `_batched.seed_homographies` through the `_batched` module object;
  never a `from ... import name` binding, a default argument or a
  local alias taken before a loop. Import direction: the module
  imports neither `_engine` nor `_batched` at module scope (a
  function-scope import of `_batched` is allowed), and no cupy
  anywhere at module scope. Extensions:
  - `FourierMellinState` (per reference): `lut` (the run's shared
    look-up table, D22.3.3), `reference_profile` (`(n_theta,)`
    float64 in `ctx.xp`), `box` (the bounding-box resident of
    D22.4.1), `resident` (a REFERENCE to the grain's existing
    subregion `ReferenceResident` -- on the device the very object
    the lockstep uses, no second upload, evicted with its
    `SeedState` under `R_MAX`; on the CPU route the numpy resident
    built once per routed reference), `n_theta`, `rho_min`,
    `rho_max`, `search_deg`.
  - `SeedState.fourier_mellin`: `None` (the default, and always under
    `"off"`) or the reference's `FourierMellinState`, a new keyword
    argument of the class with default `None`.
  - `SeedBatch.extras` INPUT key `"fourier_mellin_route"`: `(P,)`
    int8 HOST array filled by the runner from host data keyed by
    `pattern_index`: 0 not routed, 1 routed (with the acceptance), 2
    forced (the retry); 0 on padded slots. `extras` stays input only
    and on the host.
  - `SeedBatch.outputs`: a NEW field, a dictionary (a new keyword
    argument with default `None`, giving a new empty dictionary),
    written only by `seed_homographies`. Rule, one for every caller:
    the outputs exist if and only if the route key is present AND has
    a nonzero entry; then `"fourier_mellin_angle"` (`(P,)` float64 in
    `ctx.xp`: `theta_hat` in deg on route-1 and route-2 slots, NaN on
    route-0 and padded slots even though the masked branch computed
    an angle there, and NaN where the angle failed) and
    `"fourier_mellin_applied"` (`(P,)` bool in `ctx.xp`, True exactly
    where the returned row is `h_FM`). A runner that finds no outputs
    -- FM off, an all-zero route, or a planted `seed_homographies`
    (the V9(e) arms) -- treats every slot as angle NaN and applied
    False.
  - `seed_homographies`: when `seed_state.fourier_mellin is None`, or
    the route key is absent or all zero -- decided on the HOST before
    any device work -- it returns exactly the Stage E rows from the
    same code, bitwise, and writes nothing. Otherwise it computes
    `h_T` for every slot first, then calls `fourier_mellin_rows` at
    the FULL sub-batch shape P with unrouted and padded slots
    computed and discarded (masked), so that no routed slot's bits
    depend on which other slots are routed (D21.7.3); its rows are
    `h_T` except where D22.5 keeps `h_FM` (route 1) or the FM row is
    forced and valid (route 2). Compaction to the routed slots is a
    measured follow-up (plan open question FQ9).
  - `seed_spectra`: unchanged; the FM stage writes only NEW arrays,
    so `target_spectra` is identical before and after the call.
  `SeedContext` gains nothing.
- **D22.7 The pre-fit gate under "auto" (FROZEN recipe; threshold a
  RECORDED DEFAULT, MTP)**: computed on the HOST in `run_hrebsd_dic`
  after `resolve_reference` and the grain fit ordering and before
  any fit, so it costs nothing on the device for unrouted slots and
  routes identically on both backends. The frozen function
  `twist_about_detector_normal(xmap, detector, point_index,
  reference_index) -> (n,) float64` returns, per fitted point, the
  twist in deg about the detector normal of the symmetry-reduced
  misorientation from its GRAIN reference, NaN where it cannot be
  computed. Recipe, every convention existing code:
  1. `g = best_orientation_matrices(xmap)` (`_tensors.py`; sample to
     crystal, `v_c = g v_s`), map points matched to flat map indices
     exactly as `segment_grains` matches them
     (`_segmentation.py:224-231` at f297867e).
  2. `S` over the proper point group of the reference's phase,
     chosen to maximise `trace(g_t^T S^T g_r)`; `R_s = g_t^T S^T
     g_r`. A phase without a point group uses no symmetry and warns
     (the `_segmentation.py:243-258` precedent).
  3. `M = sample_to_detector_matrix(detector)` (`DETECTOR_Y_FLIP @
     sample_to_detector`, determinant -1; the D7 chain, so the gate
     and the Stage B rotations share one frame); `R_det = M R_s
     M^T`, a proper rotation.
  4. `twist = atan2(R_det[1, 0] - R_det[0, 1], R_det[0, 0] +
     R_det[1, 1])`, which equals `2 atan2(q_z, q_w)`: the exact
     twist of the swing-twist split, independent of the out-of-plane
     swing. Positive turns +x towards +y, the sense of V4's
     `atan2(h21, 1 + h11)` read-out. Matrices, never a rotation
     vector (with det M = -1 the small-angle form flips sign).
  Routing, frozen name `fourier_mellin_routes(twist_deg, gate_deg)
  -> (n,) int8` (so that the boundary is testable on a planted
  twist): 1 if and only if `|twist| >= gate_deg` or the twist is
  NaN, else 0; `run_hrebsd_dic` calls it with `FM_GATE_DEG`
  (RECORDED DEFAULT 1.5 deg, MTP). NaN covers a point not indexed, a
  non-finite rotation, a phase other than its reference's and an
  unusable reference: the gate FAILS OPEN, because an FM row can
  only replace a translation row by winning D22.5. Masked points are
  never routed. Warning, frozen literal (no stage letters):
  "fourier_mellin='auto' found no rotation about the detector normal
  in the crystal map (every point matches its reference
  orientation); no point is routed before the fit and only the
  post-fit retry applies", a `UserWarning` issued if and only if no
  twist of a fitted non-reference point is NaN and every one has
  `|twist| < 1e-9` deg (an input condition, not a float equality: a
  constant non-identity placeholder gives twists of order 1e-15, not
  exactly 0). Why the twist alone, recorded: it is the physical
  quantity the D5 seed fails on, computed exactly from orientation
  data on both backends; the effective rotation of the crop,
  `theta_eff ~= twist + (w1 xbar + w2 ybar) / (2 DD)` (ledger 201),
  differs by up to about 0.5 deg on the rim and is a candidate gate
  quantity recorded beside the twist in the whole-map census (FQ1,
  FQ2). Evidence: with exact orientations, every threshold below 2.5
  deg routed every case of the 113-case sweep whose translation seed
  failed, and the translation seed's success between 2.5 and 6 deg
  is erratic (ledger 212); 1.5 deg leaves 1.0 deg for Hough noise
  (about 0.5 deg on good patterns, D5; 1 to 1.2 deg per component in
  deformed zones, Ernould 2021 section 5). The recipe is measured
  exact on synthetic orientations (recovery to 1e-6 deg at -3.0,
  2.5 and 4.8 deg on two detector geometries; ledger 212). ON REAL
  DATA THE FRAME IS NOT YET PINNED, and the 32 rim points measured
  so far CANNOT pin it: over a range of only 0.05 to 0.45 deg the
  input twist correlates with the fitted in-plane rotation at 0.836,
  but the least-squares slope is 0.452 with the noisy input twist as
  the regressor and 1.545 the other way round (geometric mean 0.54,
  median ratio 1.19; standard deviations 0.221 and 0.120 deg; ledger
  216) -- regression dilution brackets 1; and a 90 or 180 deg
  sample-frame offset moves the routed count from 385 to 79-280
  (ledger 212). The frame decision is a REQUIRED implementation-gate
  measurement with a pre-registered rule (V10(m)(1), FQ13): the
  converged homographies of an `"always"` run (independent of the
  frame under test); per point the FITTED twist, step 4 applied to
  the rotation of the polar decomposition of the converged
  detector-frame Fe (D7), with the homography read-out `atan2(h21 -
  h12, 2 + h11 + h22)` as a cross-check; points selected on |fitted
  twist| >= 1 deg (never on the xmap twist), converged, outside the
  crater; for the as-read frame and four alternatives (sample-frame
  offsets of +90, 180 and -90 deg about sample z, and `M` without
  the y flip) the correlation of the xmap twist with the fitted
  twist; for the as-read frame the slope of the xmap twist ON the
  fitted twist (the low-noise regressor) and a Deming slope with the
  Hough variance measured on the far field. PASS: the as-read frame
  has the highest correlation of the five and its slope lies in
  [0.8, 1.25]; otherwise the stage stops and goes to Johan. The data
  gate (`theta_hat` on every slot) is the recorded alternative (plan
  open question FQ1).
- **D22.8 The post-fit retry (RECORDED DEFAULT; plan 11.3 (iv))**:
  under `"auto"` and `"always"` alike, after the first pass, ONE
  retry pass over every fitted, unmasked point of a grain with an FM
  state whose first-pass result is not converged and whose first
  fit was seeded by `h_T` (seed code 0 of D22.9) -- including a
  point whose first-pass `h` is NaN, because the D2.6 contract also
  arises from a fit that diverged from a spurious seed; an unusable
  crop makes the forced FM estimate fail, so such a point is not
  refitted. Under `"always"` the subset is exactly the points whose
  acceptance kept `h_T` and whose fit did not converge. The retry
  seeds each such point with its FM row, FORCED (route 2, no
  acceptance; D22.5), at the run's own `max_iterations`. A forced
  slot whose FM estimate fails, or whose `h_T` is non-finite, is NOT
  FITTED: on the CPU route the chunk skips the fit after the P = 1
  seam call; on the device the runner marks the slot inactive, like
  a padded slot, between the seam call and the lockstep (it reads
  `batch.outputs` there), and its packed row carries `converged` 0
  and `fm_applied` 0. Replacement, frozen: the retry result replaces
  the first-pass result wholly (`h`, `residual`, `num_iterations`
  counting the retry fit only, `norm_dp`, `converged`; the D20
  precedent) if and only if the retry converged; otherwise the first
  result stands bitwise (so an unfitted slot can never replace
  anything). The retry is a runner change, not a seam or engine
  change: on the CPU a second chunked pass over the subset (D22.10);
  on the GPU a second `_run_chunks_gpu` call over the subset (a new
  session, D21.9.1; the subset padded per grain as any run) at the
  FIRST pass's final batch size, passed as an explicit `chunksize`,
  so that one run's results come from one B unless the retry itself
  runs out of memory, which follows D21.10.4 for the retry subset
  only and is recorded. Trigger limits, recorded: non-convergence
  alone misses a fit that CONVERGES to a wrong optimum, which the
  default path does at large budgets (V8(b): 50 to 108 px away at
  2000 iterations; on real data 2 of 9 converged rim points sat at
  residuals 1.95 and 1.86 where the FM seed reached 1.14 and 0.53;
  ledger 206); a residual trigger is plan open question FQ7.
- **D22.9 Props and the packed row (FROZEN; the D20.5 absence
  rule)**: on runs with `fourier_mellin` other than `"off"`, two new
  props: `fourier_mellin_seed` `(n,)` int32 -- 0 the stored result's
  fit was seeded by the translation row (not routed, or the
  acceptance kept `h_T`, or the D5 path), 1 by an FM row in the
  first pass, 2 by an FM row in the retry pass, -1 masked or not
  fitted; and `fourier_mellin_angle` `(n,)` float64 -- the most
  recent `theta_hat` in deg estimated for the point (the retry's if
  it was retried, else the first pass's if it was routed), NaN
  elsewhere, so a point whose translation row won D22.5 carries code
  0 and a finite angle. With `fourier_mellin="off"` both are ABSENT
  and the prop set is exactly the pre-Stage-F one. D15.6's Stage A
  list is amended by this entry for FM runs only (the D20.5
  precedent: no D15.6 text edit). Output channel, frozen: on FM runs
  ONLY, the packed per-point row widens from 12 to 14, slot 12
  `fm_angle` (`theta_hat` or NaN) and slot 13 `fm_applied` (1.0
  where the slot's seed row was `h_FM`, else 0.0): the CPU
  `_fit_chunk` builds 14-wide rows and `_run_chunks` declares
  `new_axes={"j": 14}`; the engine passes the device runner
  `row_slots` with `"width": 14, "fm_angle": 12, "fm_applied": 13`
  added to the Stage E keys; `run_hrebsd_dic` derives the seed code
  from `fm_applied` of the pass whose result it stores. With FM off
  the width stays 12 and the Stage E `row_slots` dictionary is
  passed unchanged (the V9 pins). The `ebsd.py` prop loop
  (`(*STAGE_A_PROP_NAMES, SEED_ROUND_PROP_NAME)`, `ebsd.py:3492` at
  f297867e) gains the two names through one new engine constant,
  `FOURIER_MELLIN_PROP_NAMES`, added to the existing `_engine`
  import block (`ebsd.py:59-63`; plan 12 merge hygiene).
- **D22.10 How the CPU joins (FROZEN; plan 11.3 (viii), with a
  recorded deviation)**: `backend="cpu"` keeps `_fit_chunk` ->
  `fit_pattern` -> `initial_guess` for every point that is neither
  routed nor retried, so those points are BITWISE the `"off"` path.
  A routed point (route 1, or route 2 in the retry) is seeded inside
  the chunk function, with its route flag paired to its pattern by
  the same blockwise index as the Stage D `h0` block, through the
  numpy seam at P = 1: `SeedContext(np, np.fft,
  make_kernel_namespace("numpy", "float64"))`, the reference's numpy
  `SeedState` with its `FourierMellinState` (holding the reference's
  numpy subregion resident and box resident, built once per
  reference that has a routed point, on the host, before the graph;
  the look-up table once per run), a one-slot `SeedBatch` with the
  route key, `_batched.seed_spectra` and `_batched.seed_homographies`
  through the module globals. If `"fourier_mellin_applied"` is True
  the fit runs from the FM row (`h0=row`); otherwise a route-1 point
  runs with `h0=None`, so a point whose translation row won D22.5 is
  bitwise the `"off"` result even where the numpy seam's translation
  row differed from skimage's, and a route-2 point is not fitted
  (D22.8). P = 1 makes the CPU result independent of chunking by
  construction (D16, D20.3). This is the ONE xp-agnostic FM code
  path the device runs (D21.14.3); the CPU route's rows are the
  parity oracle. Under `"always"` every fitted point takes this
  route, the translation row of the numpy seam included. The routed
  point is preprocessed and splined once for the seam and again
  inside `fit_pattern`: the price of keeping `fit_pattern`'s
  signature frozen (costed in D22.15). Deviation from note (viii),
  recorded: the note routes gated CPU points through the numpy seam
  "batched, then fitted with `h0=row` through the Stage D plumbing";
  D22.10 seeds per point inside the chunk instead, because P = 1
  makes each result independent of chunking and batch composition by
  construction and needs no host pre-pass that reads the routed
  lazy patterns twice; the batched host pre-pass through
  `_run_chunks(h0=...)` is the recorded alternative (plan 12.4 item
  11). Host memory, recorded: each routed reference holds a numpy
  `SeedState` (the reference spectrum, 3.0 MB at 432x432 and 4.1 MB
  at 460x560), the box resident (two float64 coordinate planes of
  the box, the same sizes) and the numpy subregion resident (about
  12 to 15 MB at 460x560, MTP), about 20 to 25 MB per reference
  beside the 17.96 MB of its `ReferenceState`, for the whole run
  (every grain under `"always"`); the look-up table adds 3.8 to 4.0
  MB once. The D16 information message's memory note counts them
  when FM is on (D22.18).
- **D22.11 Interplay (RECORDED DEFAULT)**: (1) `seed_from_neighbors=
  True` with `fourier_mellin` other than `"off"` raises `ValueError`
  on both backends: "seed_from_neighbors=True cannot be combined with
  fourier_mellin={fourier_mellin!r}; use one seeding strategy" (no
  stage letters, the Phase 12 D13 rule). It fires second in the
  D22.1 order, so under `backend="gpu"` it precedes the D21.12
  `NotImplementedError` for this new combination only; no existing
  pin moves. Composition on the CPU (FM rows feeding PASS 1) is plan
  open question FQ10. (2) `backend="gpu"`: FM is the device's answer
  for the regime D21.12 leaves to the CPU. Wiring, frozen: the route
  flags reach the device runner as a host `(n_fit,)` int8 array in
  fit order through one new keyword-only argument,
  `_run_chunks_gpu(..., seed_extras=None)`, a mapping of extras key
  to host array that the runner slices per sub-batch slot into
  `SeedBatch.extras` (0 on padded slots); `None` is the Stage E
  behaviour. FM is ON in the device runner if and only if
  `seed_extras` carries `"fourier_mellin_route"` with a nonzero
  entry. The session builds a grain's FM state LAZILY in
  `residents()`, the first time a sub-batch of that grain carries a
  nonzero route, by `_fourier_mellin.build_fourier_mellin_state(ctx,
  state, resident)` with the grain's just-built or cached resident,
  and attaches it to that grain's `SeedState`; it is evicted with
  the `SeedState` under `R_MAX`, and a grain with no routed slot
  never builds one. The runner reads `batch.outputs` after the seam
  call: it makes forced slots that were not applied inactive (D22.8)
  and writes `fm_angle` and `fm_applied` into the packed row (D22.9).
  No per-point `h0` enters the device runner (D21.12 holds). (3) The
  D21.12 message literal loses its "is planned" clause (dated D21.12
  amendment, Block 10).
- **D22.12 Float discipline (FROZEN)**: the gate in float64 on the
  host; the look-up table built on the host (float64 weights, int64
  indices, uploaded); every FM step after reading `target_spectra`
  (the stencil, the magnitude, the gather, the profile, the
  correlation, the peak) in complex128 and float64, the spectra
  promoted at the read (a no-op under the default complex128 seed);
  `R(theta_hat)` and the composed row in float64; the de-rotation
  through `kernels.gather` at the device precision's per-pixel rules
  (the f32 gather on the f32 coefficients under `"mixed"`, float64
  under `"float64"` and in the numpy twin, D21.4); the de-rotated
  crop's translation at `seed_precision` (the Stage E code); the
  acceptance criterion through the D21.6 kernels. Under
  `seed_precision="complex64"` the angle is computed from promoted
  complex64 spectra; its effect on `theta_hat` is MTP (plan open
  question FQ14).
- **D22.13 Determinism (FROZEN)**: the D16 pledge and D21.7 extend
  to the FM path per backend (dated D16 note, Block 6). Bitwise run
  to run: no atomics; every FM reduction is a gather and a
  fixed-shape sum; the peak's ties break to the lowest output index
  (asserted on a constructed tie under both namespaces); de-rotation
  and criterion reuse the D21.6 kernels and the D21.7.2 reduction
  rules. CPU: chunksize invariance and lazy equals eager, bitwise (P
  = 1; the route is a pure per-point function of the `xmap`, the
  grain reference and the point's pattern). GPU: the FM branch runs
  masked at the full P (D22.6), so the D21.7.3 rules hold unchanged;
  a routed slot's row is the same whether 1 or all P slots of its
  sub-batch are routed (asserted, V10(k), (l)); B invariance is
  measured as in E4; the default B is the `"off"` model's unless the
  FM terms do not fit (D22.18), and the retry runs at the first
  pass's B (D22.8). The retry subset is a pure function of the first
  pass's per-point results.
- **D22.14 Parity contract (FROZEN discipline; bands MTP)**: the CPU
  route is the oracle (Johan note (v)); seeds are never pinned
  bitwise across backends. (a) Pre-fit routes: identical on both
  backends, exactly (host computed from the same inputs). (b)
  `theta_hat`, device against CPU: within `FM_ANGLE_PARITY_DEG`
  (MTP; the angle depends only on `target_spectra`, so the expected
  scale is FFT rounding, far below the 0.5 deg bin). (c) Acceptance
  decisions: a COUNT budget `FM_ACCEPT_FLIP_COUNT` (MTP, small N; a
  near-tie of the two criteria may flip). (d) `h` on points both
  backends converge from FM rows: the D21.8(c) bands per device
  precision. (e) Iteration counts and `converged`: the D21.8(e) count
  budgets. (f) Retry routing: a count budget (a first-pass
  `converged` flag can flip at the cap, D21.8(e)). The numpy session
  against the CPU route (both numpy, float64): `theta_hat` and the
  rows measured, pinned bitwise if bitwise, else at the measured
  band.
- **D22.15 Performance record (RECORDED, never a CI or test gate)**:
  re-measured on the implementation and recorded with recipe and
  machine: the CPU cost of the FM branch per routed point and of the
  angle alone; the GPU cost per slot of a routed sub-batch, the
  number of such slots and the seed's share of device time; the
  whole Si-indent map under `"auto"` at `(None, None)` and `(0.05,
  None)` on both backends (routed, accepted, retried, converted and
  worsened counts; wall time against `"off"`); far256 and patch C
  under `"always"`. Spec-gate expectations (ledgers 211, 216): CPU,
  one thread, the angle path 9.0 and 11.9 ms per slot (the dense
  numpy stencil 6.5 and 8.8 ms of it, at 432x432 and 460x560); a
  routed point adds the numpy seam's translation row (44-61 / 55-68
  ms), the FM seed (about 66-80 / 83-101 ms: angle, de-rotation
  gather, XC on the de-rotated crop), two criterion evaluations
  (20-24 / 28-30 ms each) and the duplicate preprocessing and spline
  of D22.10 (15 / 45 ms; ledger 90), about 165 to 205 ms at 480x480
  and 240 to 275 ms at 512x622, i.e. about 7 to 11 IC-GN iterations
  (21-23 / 24-31 ms each), plus one more translation seed (23-35 ms)
  inside `fit_pattern` when `h_T` wins. GPU: about 2.2 to 2.5 ms per
  SLOT of every sub-batch that holds a routed point (an INFERENCE
  from the 2.16 ms device seed, ledger 92, never measured), because
  the branch runs masked at the full P = 32: on the Si map in fit
  order with the crater masked, the 336 points routed at 1.5 deg
  touch 55 of 1806 sub-batches, 1760 slots (5.2x); 73 at 2.0 deg give
  544 slots (7.5x); 1012 at 1.0 deg give 3552 slots (3.5x) (ledger
  216) -- about 4 s, roughly 2.5 per cent of the 166.5 s mixed map
  run at 1.5 deg; under `"always"` about +130 to +145 s, roughly +80
  per cent, consistent with the seed's 62 per cent share of device
  time (ledger 98).
- **D22.16 Docs, CHANGELOG, tutorials (FROZEN; tutorial cell a
  RECORDED DEFAULT)**: the `hrebsd_dic` docstring gains the
  `fourier_mellin` entry (values, default, the gate and its
  CrystalMap dependence, the retry under both FM modes, the
  acceptance, rotation only) and Notes giving both capture numbers
  -- the seed captures rotations about the detector normal up to its
  30 degree search window (measured to 30 degrees on synthetic 480
  pixel patterns with this recipe; ledger 216), and the fit's
  accuracy holds to the in-frame angle of the chosen border (5.6 and
  6.6 deg at `border=0.05` for the two oracle PCs, 23.8 and 38.7 deg
  at 0.15; ledger 215) -- plus the two props and the cost; Raises
  gains the two `ValueError` entries of D22.1 and D22.11 and the
  D22.7 warning; the Limitations paragraph ("The initial guess is a
  translation only phase cross-correlation ...", `ebsd.py:3343` at
  f297867e) is rewritten to say that the default seed is
  translation only with the measured capture range and that
  `fourier_mellin` lifts it; no stage letters in public text. A NEW
  docstring test, `TestFourierMellinSwitch::
  test_the_docstring_documents_fourier_mellin`, pins all of them (the
  D21.17 precedent); no existing docstring test changes. One
  fork-style CHANGELOG entry in the HREBSD block with the
  precedent's closing sentence (`CHANGELOG.rst:73-75`), D22 cited.
  Tutorials: in `doc/tutorials/hrebsd_dic.ipynb`, the markdown
  bullet "A finite rotation capture range" ("... A Fourier-Mellin
  pre-rotation stage would lift it and is not built.", lines
  1877-1881 of the file at f297867e) is reworded to name the opt-in
  `fourier_mellin` keyword -- one markdown edit, no stored output
  touched, nbval re-run once; in `hrebsd_si_indent.ipynb` one
  MARKDOWN cell with the whole-map `"auto"` record and the call as a
  fenced code block, no executed cell and no stored output touched
  (the D21.17 precedent).
- **D22.17 Dependencies (FROZEN)**: no new dependency and no new
  import outside the audit's allowed tuple (numpy, scipy and orix
  are already required); the `TestImportAudit` arms of D21.13 cover
  the new module (no module-scope cupy, no module-scope `_engine` or
  `_batched` import).
- **D22.18 VRAM model and batch choice (FROZEN form; calibration
  MTP)**: `_vram_model_terms`, `_vram_model_bytes` and
  `_default_batch_size` each gain a keyword-only `fourier_mellin:
  bool = False`, so the frozen positional lists of D21.9.5 do not
  change and, with `False`, every one returns the Stage E value
  bitwise (V10(k)). FM adds to p (per slot of a routed sub-batch:
  the promoted spectrum or its stencil, the polar gather, the
  de-rotated crop, its spectrum and cross-power, two criterion
  passes) and to r (per reference: the box resident's coordinates
  and the profile; the subregion resident is the existing one, not a
  copy; the run's look-up table, about 3.8 MB at 432x432, counted
  once inside r, which `R_MAX * r` over-counts safely). The runner
  passes `fourier_mellin=True` if and only if a route flag of the
  run is nonzero. Batch choice, frozen: `_default_batch_size` picks
  B from the `"off"` model, then, with `fourier_mellin=True`, checks
  the FM terms against the remaining headroom and halves B only if
  they do not fit -- so on a card with headroom FM never changes B,
  and unrouted points under `"auto"` keep the bits of `"off"`; where
  it does change B, the run records it, and the FQ3 comparison is
  made at a fixed explicit `chunksize`. The sizes quoted are
  array-size estimates, not measurements; each term is calibrated
  against pool high-water marks at the implementation gate (the
  D21.10.2 rule). Host side: the D16 information message's memory
  note adds the D22.10 per-reference FM bytes when FM is on.
- **D22.19 Out of scope (recorded, each with its revisit path)**:
  scale and log-polar sampling (D22.2); a two-candidate 180 deg test
  (FQ11); a bias correction of `theta_hat` for in-plane-axis
  rotations (FQ12); the data gate and a `theta_eff` gate (FQ1); a
  residual-triggered retry (FQ7); FM with `seed_from_neighbors`
  (FQ10); the complete initialisation (FQ5); FM sub-batch compaction
  on the device (FQ9); public FM constants; dead-band axis masking
  of the look-up table (FQ4); the D5 seed's zero-shift anchor on
  real data (FQ8, its own decision: Stage F does not change D5); the
  orientation-delta seed of D5 (the gate computes `R_det`, its
  input, but the seed stays deferred); a transmitted-beam mask for
  on-axis TKD (Ernould 2021 appendix C).

<!-- ======= BLOCK 2: Scope amendment (the Fourier-Mellin bullet) ======= -->
<!--
ANCHOR: requirements.md, "## Scope", the bullet "**Fourier-Mellin
rotation initial guess** (Ernould 2020's cascaded FM + XC): ...
backends behind the D21.5 seed seam (not commissioned)." (lines
115-119 at f297867e). Append inside that bullet.
-->

  SPECIFIED 2026-10-07 as Stage F (requirements D22; commissioned
  2026-10-06): opt-in through the `fourier_mellin` keyword, default
  `"off"`, so v1's translation-only default stands.

<!-- ============ BLOCK 3: D5 amendment (Stage F specified) ============ -->
<!--
ANCHOR: requirements.md, D5, the bold paragraph that ends "...stays
the behaviour of both backends until it is.**" (the PLANNED
2026-10-06 note). Insert immediately after it, inside the same
bullet, as a new bold paragraph.
-->

  **SPECIFIED 2026-10-07 (Stage F, requirements D22; commissioned
  2026-10-06 under Johan's extended waiver).** The FM seed is the
  opt-in `fourier_mellin` keyword (D22.1, default `"off"`), one
  xp-agnostic code path behind the D21.5 seam for both backends. The
  translation-only seed above stays the `"off"` behaviour of both
  backends, bitwise, and is unchanged inside the FM branch too (it
  measures the residual translation of the de-rotated target,
  D22.4). A measured property of this seed on REAL data, found at the
  Stage F spec gate and NOT changed by Stage F: on the deformed
  Si-indent rim the full-band phase correlation returns exactly
  (0, 0) at every point tested, anchored by detector-fixed content
  (zero-shift peak 0.09-0.10 against 0.012-0.025 at the true 6 to 21
  px; ledger 207); plan open question FQ8 carries it.

<!-- ======== BLOCK 4: D5 amendment (orientation-delta bullet) ======== -->
<!--
ANCHOR: requirements.md, D5, the last bullet ("A future
orientation-delta seed ... at the scale of what it would seed.").
Append to that bullet.
-->

  Note 2026-10-07 (Stage F): the D22.7 gate computes the
  detector-frame misorientation `R_det` this seed would compose
  through D6 (`fe_to_homography(R_det, (0, 0), DD_ref)`, verified on
  synthetic orientations; ledger 212); the seed itself stays
  deferred.

<!-- =============== BLOCK 5: D15.4 amendment (signature) ============== -->
<!--
ANCHOR: requirements.md, D15.4, the paragraph "**AMENDED 2026-10-06
(Stage E, requirements D21.1).** ... are never public in v1." Append
after its last sentence, inside item 4.
-->

   **AMENDED 2026-10-07 (Stage F, requirements D22.1).** A third
   dated extension: `fourier_mellin: str = "off"` immediately after
   `seed_from_neighbors`. The frozen order after Stage F ends `...,
   step_scale, seed_from_neighbors, fourier_mellin, navigation_mask,
   backend, chunksize, verbose`, every one keyword only; `backend`
   is still immediately followed by `chunksize` (D21.1). The freeze
   test `TestFrozenSignature::test_signature_is_frozen` gains
   `("fourier_mellin", KEYWORD_ONLY, "off")` after
   `seed_from_neighbors`, with a dated comment like the D20.1 and
   D21.1 ones, and `TestOrchestration::test_run_defaults_are_frozen`
   gains `fourier_mellin` (`"off"`) on `run_hrebsd_dic`. (D15.6 is
   amended by reference in D22.9, the D20.5 precedent; its text is
   not edited.)

<!-- ============ BLOCK 6: D16 note (determinism extended) ============ -->
<!--
ANCHOR: requirements.md, D16, second bullet, after the paragraph
"**Re-scoped per backend 2026-10-06 (Stage E, requirements D21.7):**
... tolerance parity against the CPU (D21.8)." Append inside that
bullet.
-->

  **Extended 2026-10-07 (Stage F, requirements D22.13):** the
  pledge covers the Fourier-Mellin path per backend: on the CPU the
  FM seed runs per point at P = 1, so results stay bitwise
  invariant to chunking and lazy equals eager; on the GPU the FM
  branch runs masked at the full sub-batch P, so the D21.7 rules
  hold unchanged. The FM per-reference host bytes join the
  information message's memory note when FM is on (D22.10, D22.18).

<!-- ============ BLOCK 7: D21.1 amendment (order of checks) ============ -->
<!--
ANCHOR: requirements.md, D21.1, after its last sentence
("`LazyEBSD` inherits the method unchanged."). Append inside the
bullet.
-->

  **AMENDED 2026-10-07 (Stage F, requirements D22.1).** The frozen
  order of checks gains three, between the `backend` string check
  and the `"gpu"`-only checks: the `fourier_mellin` value check, the
  D22.11 `seed_from_neighbors` combination `ValueError`, and, under
  `fourier_mellin="auto"` only, the `xmap is None` `ValueError`. The
  rest of the order, and every existing raise and its literal except
  the D21.12 one (Block 10), is unchanged.

<!-- ============ BLOCK 8: D21.5 amendment (seam extension) ============ -->
<!--
ANCHOR: requirements.md, D21.5, the sentence "Fourier-Mellin itself
is OUT of Stage E (D21.18; Stage F)." at the end of the bullet.
Append immediately after it.
-->

  **EXTENDED 2026-10-07 (Stage F, requirements D22.6; clause (iv)
  applied, purely additive).** `SeedState` gains the field
  `fourier_mellin` (`None`, or the reference's `FourierMellinState`,
  which holds a reference to the grain's existing subregion
  resident); `SeedBatch` gains the INPUT `extras` key
  `"fourier_mellin_route"` (host int8, 0 / 1 / 2, 0 on padded slots)
  and the NEW field `outputs` (a dictionary, empty by default,
  written only by `seed_homographies`: `"fourier_mellin_angle"` and
  `"fourier_mellin_applied"`, in `ctx.xp`, present if and only if
  the route key has a nonzero entry). `extras` stays input only and
  on the host. The two signatures, `SeedContext` and every existing
  field are unchanged; with FM off, `extras` and `outputs` stay empty
  and `seed_homographies` returns the Stage E rows bitwise. The
  NaN-row semantics above hold: the FM branch returns the
  translation row it wraps whenever its estimate fails (D22.5).

<!-- ========= BLOCK 9: D21.9.5 amendment (runner names) ========= -->
<!--
ANCHOR: requirements.md, D21.9, item 5 ("Names, frozen ..."), after
its last sentence ("...so no import cycle exists."). Append as a
dated note inside item 5.
-->

     Amended 2026-10-07 (Stage F, D22.6, D22.9, D22.11, D22.18):
     keyword-only additions, the positional argument lists above
     unchanged: `_run_chunks_gpu(..., seed_extras=None)` (a mapping
     of `SeedBatch.extras` key to a host `(n_fit,)` array in fit
     order, sliced per slot by the runner; `None` is the Stage E
     behaviour), and `fourier_mellin=False` on `_vram_model_terms`,
     `_vram_model_bytes` and `_default_batch_size`. On FM runs only,
     `row_slots` carries `"width": 14, "fm_angle": 12, "fm_applied":
     13` beside the Stage E keys; FM-off runs pass the Stage E
     dictionary unchanged. The session builds a grain's FM state
     lazily in `residents()` and evicts it with the grain's
     `SeedState`. The new module `_hrebsd/_fourier_mellin.py` follows
     the same import direction (no module-scope import of `_engine`,
     `_batched` or cupy), and its call-time seams are frozen in
     D22.6.

<!-- ============ BLOCK 10: D21.12 amendment (message literal) ============ -->
<!--
ANCHOR: requirements.md, D21.12, after the sentence "...the seed seam
(D21.5) is the one place `h0` enters the device path." at the end of
the bullet. Append immediately after it.
-->

  **AMENDED 2026-10-07 (Stage F, D22.11).** The frozen literal is
  replaced, because the Fourier-Mellin seed is no longer "planned":
  "seed_from_neighbors=True is not supported with backend='gpu'; use
  backend='cpu' for neighbour-seeded propagation, or
  seed_from_neighbors=False with fourier_mellin='auto' for large
  rotations about the detector normal" (the remedy names
  `seed_from_neighbors=False` because the combination with
  `fourier_mellin` itself raises, D22.11). V9(b) asserts the new
  literal in full, with a dated comment on
  `SEED_FROM_NEIGHBORS_GPU_MESSAGE` (`test_hrebsd_gpu.py:335` at
  f297867e). The raise itself, its position in the D21.1 order and
  its reasons are unchanged; with `fourier_mellin` other than
  `"off"`, the D22.11 `ValueError` fires first.

<!-- ============== BLOCK 11: D21.18 amendment (out of scope) ============== -->
<!--
ANCHOR: requirements.md, D21.18, after "super-resolution (plan
9.7)." Append.
-->

  Amended 2026-10-07: the Fourier-Mellin seed is specified as Stage
  F in D22 and is no longer out of scope for the project; it stays
  outside the Stage E commit set.
