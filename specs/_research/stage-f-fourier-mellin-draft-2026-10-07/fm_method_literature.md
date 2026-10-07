# Stage F -- Fourier-Mellin initial guess: method and literature

Prepared 2026-10-07 for the Stage F spec (HREBSD-DIC, branch
`hrebsd-dic`, requirements D5 deferral, plan open question 5, plan
11.3 Follow-ups and 11.4 item 20, the frozen D21.5 seed seam).
Repo files were READ only. Every number marked MEASURED below comes
from a throwaway CPU prototype in this folder (`fm_lib.py`,
`exp1_angle.py` to `exp10_anchor.py`, outputs `exp7.out` and
`exp7b.out`; numpy, one process, no GPU), and is carried inline here
because the session scratchpad does not survive (the D19 rule).
The prototype's spectra, crops, preprocessing and fits are the
engine's own (`preprocess`, `ReferenceState`, `fit_pattern`,
`initial_guess`, the V3/V4 deformed-master oracle of
`test_hrebsd_engine.py`), so the numbers are on the frozen D2/D4/D5
chain, not on a reimplementation. Real patterns are read-only reads
of `AGH__Si_indent_1_512x672.h5oina` in the main folder (never
copied). Stage E's mid-implementation `_batched.py` was read for the
seam's names only; nothing here depends on its internals.

## 0. Summary

- **What Ernould's initial guess is** (sections 2, 3): one global
  square ROI, Tukey-Hann apodised and zero-meaned under the window;
  FFT; log-amplitude; POLAR (not log-polar) resampling of the upper
  half-plane only (angles in [0, pi)), angular step pi/2^n for a
  2^n ROI, radial step 0.5 px, 3-point bilinear interpolation from
  a precomputed look-up table; each angle row replaced by its mean
  over radius, giving ONE 1D angular signature; 1D cross-correlation;
  Gaussian peak fit; one rotation estimate, no iteration; then the
  target ROI is rotated by minus that angle and an ordinary FFT
  cross-correlation measures the residual translation; the
  homography is initialised as W = R(theta) T(t) ("partial", h31 =
  h32 = 0) or, reading the translation as two lattice rotations
  atan(t/DD), as the full homography of that rotation ("complete").
  Reported: angular resolution 0.35 deg at 512 px; initial-guess
  scatter 0.2-0.3 deg per rotation component on deformed IF steel
  against 1-1.2 deg for Hough indexing; 0.25 deg typical below
  2 deg disorientation and 0.5 +- 0.25 deg up to 14 deg on
  simulated patterns; rotations "up to approximately ten degrees".
- **The EBSD physics** (section 4): a lattice rotation about the
  DETECTOR NORMAL is an EXACT image rotation about the projection
  centre, at any angle, with no gnomonic distortion (h = (c-1, -s,
  0, s, c-1, 0, 0, 0) in the D1.3 frame). Rotations about the two
  in-plane axes translate the PC point by DD tan(w) px and add a
  perspective term; they also rotate the pattern LOCALLY by
  (w1 x + w2 y)/(2 DD), so the FM angle of a subregion whose centre
  is off the PC is biased by (w1 xbar + w2 ybar)/(2 DD) -- MEASURED
  with a windowed spectrum to within 0.004-0.06 deg of the formula
  on the oracle -- and sees a pseudo-scale (3/2)(w2 xbar - w1
  ybar)/DD, which is why rotation-only is the safe default (5.7 per
  cent pseudo-scale for a 5 deg w1 on the Si-indent geometry,
  against elastic strains of 1e-3).
- **Measured on the frozen chain** (section 5): the FM angle is
  accurate to 0.01-0.08 deg on the noise-free oracle (0-15 deg) and
  0.02-0.05 deg on REAL Si-indent patterns rotated rigidly about the
  PC; FM-seeded IC-GN converges in 2-8 iterations at every angle
  tested up to 8 deg, where the translation-only seed fails from
  2.5 deg (oracle) or 3 deg (real Si, default band-pass) and already
  needs 19 / 124 iterations at 1 / 2 deg on real Si. On REAL
  deformed patterns (the indent rim) the FM angle is a 0.1-0.7 deg
  seed (it over-reads the twist through the in-plane-axis bias and
  strain), and still: with the default band-pass 24 of 30 rim points
  converge with the FM complete seed against 9 of 30 with the D5
  seed; unfiltered all converge and the FM seed saves 22 per cent of
  the iterations (3503 -> 2721).
- **Two findings that change the design** (M5, M7, M8): (1) the
  REUSED, unwindowed target spectrum is fine with the default
  high-pass but LOCKS TO ZERO ROTATION without it when a
  detector-fixed background is present; the Moisan periodic-plus-
  smooth correction, computed from the crop's boundary rows and
  columns and subtracted from the reused spectrum, removes that
  failure and keeps the reuse. (2) The frozen D5 full-band phase
  correlation is ANCHORED AT ZERO SHIFT by detector-fixed content on
  real Si: it returns exactly (0, 0) at every deformed point tested,
  where a band-limited phase correlation or the plain correlation
  finds the true 6-18 px. The FM branch escapes the anchor because
  the de-rotation also rotates the target's fixed content; this is
  much of what FM buys on the rim. Consequence: the phase-correlation
  peak is the WRONG acceptance signal (the anchor's 0.1 beats the
  true peak's 0.02); the ZNSSD criterion at the seed (the IC-GN's
  own objective) is the right one (T 1.06-2.14 against FM 0.26-0.69
  on the rim).
- **Recommended recipe** (section 6): reuse `target_spectra`;
  subtract the periodic-plus-smooth smooth component; amplitude |P|
  (not log) over rho in [0.05, 0.20] cycles/px; polar LUT in physical
  frequency, 360 angles over [0, pi), 1 bin radial step; fixed-order
  gather-sum to a 1D profile; circular ZNCC by a length-360 FFT;
  search window |theta| <= 30 deg (this also disposes of the 180 deg
  ambiguity); parabolic sub-bin peak; gate; de-rotate the TARGET
  about the PC through `kernels.gather` with the carried matrix
  R(theta); the Stage E phase-XC on the de-rotated crop; complete
  initialisation from (t, theta); keep whichever of the FM and the
  translation rows has the lower ZNSSD at the seed; never NaN.
- **Gate** (section 6.5): the data gate (|theta_hat| from the reused
  spectra, measured at 0.8-0.95 of one batched 2D FFT per slot on
  the CPU) is cheaper and sharper than feared and does not depend on
  indexing noise, which in deformed zones (1-1.2 deg per component,
  S1) is of the order of the 1.5-2 deg CrystalMap threshold itself;
  the CrystalMap twist gate stays available through
  `SeedBatch.extras`, and the post-fit retry needs a residual
  criterion as well as `converged` (V8(b)'s false convergence). The
  default (opt-in until measured) and the threshold (0.5 deg
  proposed, 1.0 deg cheaper) are open question F3, and the D5 anchor
  itself is F8, in section 7.

## 1. Sources

| # | Source | Access | Used for |
|---|---|---|---|
| S1 | C. Ernould, B. Beausir, J.-J. Fundenberger, V. Taupin, E. Bouzy, Global DIC approach guided by a cross-correlation based initial guess for HR-EBSD and on-axis HR-TKD, Acta Mater. 191 (2020) 131-148, doi:10.1016/j.actamat.2020.03.026 | full text, HAL hal-03006680 | sections 2.3, 5, App. C, App. E |
| S2 | C. Ernould et al., Integrated correction of optical distortions for global HR-EBSD techniques, Ultramicroscopy 221 (2021) 113158, doi:10.1016/j.ultramic.2020.113158 | full text, HAL hal-03138727 | App. E (improved IG: target rotated; eqs. E.1-E.6) |
| S3 | C. Ernould et al., Characterization at high spatial and angular resolutions of deformed nanostructures by on-axis HR-TKD, Scripta Mater. 185 (2020) 30-35, doi:10.1016/j.scriptamat.2020.04.005 | full text, HAL hal-03006691 | same IG on TKD; no new method |
| S4 | C. Ernould, PhD thesis, Universite de Lorraine (2020), 2020LORR0225, HAL tel-03254732 | full text | ch. III.2 (the IG in detail, eqs. III.17-III.31), ch. IV.2.4 (IG accuracy, iterations) |
| S5 | B. Pan, Y. Wang, L. Tian, Automated initial guess in digital image correlation aided by Fourier-Mellin transform, Opt. Eng. 56 (2017) 014103, doi:10.1117/1.OE.56.1.014103 | abstract | the method Ernould transferred (FMT-CC seed, rotations up to +-180 deg) |
| S6 | B.S. Reddy, B.N. Chatterji, An FFT-based technique for translation, rotation, and scale-invariant image registration, IEEE Trans. Image Process. 5 (1996) 1266-1271, doi:10.1109/83.506761 | secondary (citations, implementations) | log-polar FM; high-pass emphasis; 180 deg ambiguity |
| S7 | E. De Castro, C. Morandi, Registration of translated and rotated images using finite Fourier transforms, IEEE TPAMI 9 (1987) 700-703, doi:10.1109/TPAMI.1987.4767966 | from memory | phase correlation with a rotation search |
| S8 | Q.-S. Chen, M. Defrise, F. Deconinck, Symmetric phase-only matched filtering of Fourier-Mellin transforms for image registration and recognition, IEEE TPAMI 16 (1994) 1156-1168, doi:10.1109/34.387491 | cited by S1 | FM phase-only matching |
| S9 | L. Moisan, Periodic plus smooth image decomposition, J. Math. Imaging Vis. 39 (2011) 161-179, doi:10.1007/s10851-010-0227-1; https://helios2.mi.parisdescartes.fr/~moisan/p+s | page + formula re-derived, MEASURED | removing the DFT boundary cross without a window |
| S10 | H.S. Stone, B. Tao, M. McGuire, Analysis of image registration noise due to rotationally dependent aliasing, J. Vis. Commun. Image Represent. 14 (2003) 114-135 | abstract | DFT and rotation do not commute for sampled images |
| S11 | Y. Keller, Y. Shkolnisky, A. Averbuch, The angular difference function and its application to image registration, IEEE TPAMI 27 (2005) 969-976 | abstract | the radial integral of the spectrum as a rotation signature (Ernould's row mean); pseudo-polar FFT avoids interpolation |
| S12 | S. Derrode, F. Ghorbel, Robust and efficient Fourier-Mellin transform approximations..., Comput. Vis. Image Underst. 83 (2001) 57-78, doi:10.1006/cviu.2001.0922 | cited by S1 | analytic FMT alternatives |
| S13 | A. Foden, D.M. Collins, A.J. Wilkinson, T.B. Britton, Indexing EBSPs with a refined template matching approach, Ultramicroscopy 207 (2019) 112845, doi:10.1016/j.ultramic.2019.112845 | cited by S1 | image-domain polar resampling about the PC (rotation only, no translation invariance) |
| S14 | T.B. Britton, A.J. Wilkinson, HR-EBSD measurements of elastic strain variations in the presence of larger lattice rotations, Ultramicroscopy 114 (2012) 82-95, doi:10.1016/j.ultramic.2012.01.004; C. Maurice, J.H. Driver, R. Fortunier, Ultramicroscopy 113 (2012) 171-181, doi:10.1016/j.ultramic.2011.10.013 | cited by S1, S4 | remapping, the orientation-delta seed (the D5 deferred extension) |
| S15 | T.J. Ruggles et al., Ultramicroscopy 195 (2018) 85-92, doi:10.1016/j.ultramic.2018.08.020; T. Vermeij, J.P.M. Hoefnagels, Ultramicroscopy 191 (2018) 44-50, doi:10.1016/j.ultramic.2018.05.001 | cited by S1, S4 | IC-GN convergence vs seed misorientation |

Nothing later than S2/S4 was found that changes the initial guess:
the 2021 paper's appendix E is the last published revision of the
method (search 2026-10-07; arXiv 2502.11628 and 2604.25869, both
recent HR-EBSD/DIC preprints, do not use an FM initial guess).

## 2. What Ernould's initial guess computes

### 2.1 The four steps (S1 section 2.3; S4 III.2.2)

1. Measure the in-plane rotation theta0 about X3 (the screen normal)
   by Fourier-Mellin-transform cross-correlation (FMT-CC).
2. Rotate an ROI about its CENTRE X0 by theta0. In S1 the REFERENCE
   ROI is rotated; S2 appendix E and S4 III.2.2.2.2 record the
   improved implementation that rotates the TARGET ROI by -theta0
   instead.
3. Measure the remaining translation (t1, t2) by ordinary FFT
   cross-correlation (FT-CC) of ZNCC-normalised ROIs.
4. Initialise the eight homography parameters from (theta0, t1, t2).

### 2.2 ROI, windowing, normalisation

(S1 App. C, eq. C.2; S4 III.2.2.1, eqs. III.19-III.22.)

- One large square ROI, side a power of two (resized if needed):
  512x512 cut from 600x600 on-axis TKD patterns in S1; 1024x1024
  for the IG in the S4 numerical validation and in S2 (the IC-GN
  itself uses 901x901 there).
- Apodisation: a Tukey-Hann (tapered cosine) window applied
  separably in both directions; S4 fig. III.6b shows the taper over
  the outer 25 per cent at each end and flat over the central half
  (alpha = 0.5). Purpose stated: spectral leakage.
- Mean removal UNDER the window: s_hat = (s - mean(s w)/mean(w)) w,
  then division by the root of the variance, so ZNCC applies.

### 2.3 Magnitude and filtering

- S1 App. C: "a resampling of the log-amplitude of the FT in a polar
  or log-polar frame". S4 says "module" (modulus) in the text and
  figure; the published paper says log-amplitude.
- No other spectral emphasis filter is described for EBSD. For
  on-axis TKD the PATTERNS get a high-pass log filter (S1 App. E,
  local log minus local mean log over a (2k+1)^2 kernel), median and
  Gaussian 1 px filters, and a noise MASK over the transmitted beam
  and diffraction spots on the reference, kept through the IC-GN;
  without the mask the transmitted beam "anchors" the translation
  XC at zero (S1 fig. C2c).

### 2.4 Polar sampling (S4 III.2.2.2.1, fig. III.8)

- POLAR, not log-polar: "a polar frame is sufficient since
  negligible scale changes are expected" (S1); the scale factor is
  fixed to 1.
- Only the two upper quadrants are resampled, because the modulus
  of a real image's spectrum is centro-symmetric: angles span
  [0, pi).
- Steps: pi/2^n in angle (2^n samples over pi for a 2^n ROI), 0.5 px
  in radius. Angular resolution "about 0.35 deg for a 512x512 ROI"
  (180/512).
- Coordinates precomputed once in a look-up table; 3-point bilinear
  interpolation (P.R. Smith, Ultramicroscopy 6 (1981) 201-204,
  doi:10.1016/S0304-3991(81)80199-4) to avoid the 4-point scheme's
  checkerboard artefact.

### 2.5 Reduction to 1D and the correlation (S4 III.2.2.2.1; S1 App. C)

- "Fig. C1d,d' can be replaced by a 1D-array containing the average
  value of each row": each angle's samples are averaged over radius,
  so the FM is reduced to a 1D angular signature (the radial
  integral of the spectrum, which is Keller's angular difference
  function idea, S11).
- That 1D signal is then windowed and normalised and transformed;
  the XCF peak gives the angular shift. (Windowing a pi-periodic
  signal is not needed; a circular correlation is exact for it --
  section 6.)
- Rotation angle theta0 = pi delta / 2^n with delta the peak offset
  in samples (eq. III.25).

### 2.6 Sub-pixel peak (S4 p. 72-73; S1 App. C end)

Gaussian fit to the XCF peak, as in local (Wilkinson) HR-EBSD, over
an ADAPTIVE neighbourhood: start from the maximum, grow the
neighbourhood while the mean of its edge values keeps decreasing
and stays above one third of the maximum; in 1D the "edge mean" is
the mean of the two end values.

### 2.7 Iteration

None. One FMT-CC, one de-rotation, one FT-CC. The magnitude spectrum
is translation invariant, so a second rotation estimate after the
translation would see the same spectrum up to edge content.

### 2.8 Capture and accuracy reported

- FT-CC alone "is however unsuitable for images rotated by more than
  6-8 deg" (S1, after S5); "~7 deg" (S4).
- Examples pre-aligned: on-axis TKD pair about 18-20 deg in plane
  (theta0 measured 19.6 deg in S4 fig. III.7) with a residual
  translation of about 47 and -111 px (S4; S1 quotes about 10 and
  -118 px for its reference-rotated variant); EBSD pair disoriented
  about 23 deg (S1 fig. 5); IF steel map up to about 12 deg.
- S4 abstract: the pre-alignment "fairly accounts for rotations up to
  approximately ten degrees with an accuracy typically between 0.1
  and 0.5 deg".
- S4 IV.2.4.1 (simulated, distortion-free, fixed PC, 1024 px IG):
  IG typically 0.25 deg off the solution when the disorientation is
  below 2 deg and 0.5 +- 0.25 deg at large disorientations (to
  14 deg), at equivalent elastic strain <= 5e-3; it degrades
  strongly at elastic strains >= 1e-2 but stays under 1 deg "in the
  vast majority of cases". The error is LARGEST for a pure rotation
  about X2 because the PC sat 200 px above the pattern centre (33
  per cent of the half width): the upper and lower halves are
  distorted differently by the gnomonic projection. "The global
  cross-correlation approach tends to measure displacements between
  the images in order to pre-align them, rather than to measure the
  crystal rotations precisely."
- S1 section 5 (experimental, IF steel 15 per cent strain): rotation
  components of the CC-based IG scatter 0.2-0.3 deg against
  1-1.2 deg for Hough indexing (HTI); about X3 0.31 deg, "related to
  the angular resolution of the FMT-CC, namely 0.35 deg here
  (180 deg/512 pixels)"; 99.7 per cent of points converge in fewer
  than 200 iterations with the CC IG against 90.9 per cent with
  HTI, which diverges (> 1000 iterations) on 4.2 per cent; HTI
  convergence trouble starts above 3-3.5 deg disorientation, where
  its angular deviation exceeds 1 deg; every diverging HTI point
  had at least one component more than 2 deg off.
- S4 IV.2.4.2: at elastic strain <= 2e-3, IG misorientations of
  0.1 deg and 0.5 deg cost 6-8 and 20-50 IC-GN iterations.

### 2.9 Homography initialisation and rotation centre

(S1 eqs. 13-14; S2 eqs. E.1-E.6; S4 eqs. III.26-III.31.)

- Partial: the reference-to-target map is the translation THEN the
  rotation, W = R(theta0) T(t):

  ```
  [1+h11 h12   h13]   [cos  -sin  t1 cos - t2 sin]
  [h21   1+h22 h23] = [sin   cos  t1 sin + t2 cos]
  [h31   h32   1  ]   [0     0    1              ]
  ```

  (S2 E.1, S4 III.27; S1 eq. 13 printed the un-rotated (t1, t2),
  which is the reference-rotated variant.)
- Complete: estimate the three lattice rotations in the detector
  frame, w1 = atan(-Delta2/DD), w2 = atan(Delta1/DD), w3 = theta0,
  build R, and set h from R/R33 through the homography-to-Fe
  relations (S1 eq. 14). Delta corrects the measured translation for
  the rotation having been applied about the ROI centre X0 rather
  than the PC: Delta1 = t1 + x01 (cos - 1) + x02 sin, Delta2 = t2 -
  x01 sin + x02 (cos - 1), with (x01, x02) the PC relative to X0
  (S2 E.2-E.3, S4 III.29). "The projection geometry does not need to
  be very accurate at this stage."
- Effect: complete init gives about 6 per cent lower initial
  residuals and about 10 per cent fewer iterations on IF-steel EBSD,
  and no difference on on-axis TKD (86 +- 26 iterations both),
  "because the on-axis configuration leads to a lower pattern
  distortion due to the gnomonic projection" (S1 section 5). Partial
  init keeps the DIC fully calibration-free.
- In the D1.3 PC-centred frame of this feature the rotation centre
  IS the PC, so every Delta correction vanishes when the de-rotation
  is done about the frame origin (section 6).

### 2.10 Cost (S4 III.5.2)

FT-CC 3 FFTs, FMT-CC 5 FFTs (with the 2D illustration; the 1D
reduction makes the FMT part cheaper): for 1024^2,
(3 + 5) N^2 log2 N, "about one iteration of local HR-EBSD with 40
ROIs of 256^2". Remapping (S14) was rejected as limited to small
rotations about X3; demons remapping (Zhu et al. 2020) as too
costly (1-4 s per point).

### 2.11 What Ernould proposes next (S1 section 6)

"The proposed CC-based initial guess can be only used when an
initialisation of the homography from the neighbouring points fails
to ensure a satisfying convergence of the IC-GN algorithm" -- i.e.
couple the path-independent CC seed with a reliability-guided,
path-dependent one. That is exactly the Stage D (neighbour seeding)
plus Stage F (FM seed) pairing of this feature; D21.12 already
leaves `seed_from_neighbors` to the CPU.

## 3. Classic Fourier-Mellin registration, the facts that matter here

1. **Rotation-centre independence.** If t(x) = r(R^-1 (x - c) + c)
   then T(f) = exp(-2 pi i f.(c - R c)) R(R^-1 f), so |T(f)| =
   |R(R^-1 f)| for EVERY centre c: the magnitude spectrum rotates
   about the frequency origin by the image rotation, whatever point
   the image rotated about, and any translation lives in the phase
   only (S4 eqs. III.23-III.24; S6). Consequence here: the FM angle
   needs no PC; only the de-rotation and the frame algebra do.
2. **The 180 deg ambiguity.** For a real image |F(f)| = |F(-f)|, so
   the angular signature is pi-periodic and theta and theta + 180
   cannot be told apart from the magnitude (S4 samples only the
   upper half-plane for this reason). Reddy and Chatterji (S6)
   resolve it by de-rotating by both candidates and keeping the one
   with the higher phase-correlation peak, which doubles the
   translation step. For HR-EBSD inside a grain the physical range
   is a few degrees, so the cheap resolution is a SEARCH WINDOW
   |theta| <= theta_max << 90 deg on the correlation (section 6).
3. **Edge effects.** The DFT periodises the crop; the jump between
   opposite edges puts a bright "+" along the two frequency axes
   that is fixed to the crop, not to the content, so it correlates
   at zero rotation and pulls the estimate towards 0. Remedies:
   apodisation (Hann/Tukey; S1, S4, S5), at the price of a separate
   FFT of the windowed crop, or the periodic-plus-smooth
   decomposition (S9): u = p + s with s the smooth solution of a
   Poisson problem driven by the boundary jumps, whose DFT is
   closed-form,

   ```
   v = boundary image: v[0,:] = u[M-1,:] - u[0,:],
       v[M-1,:] = -(that); v[:,0] += u[:,N-1] - u[:,0],
       v[:,N-1] -= (that)
   V(q, r) = D(r) (1 - exp(2 pi i q/M)) + E(q) (1 - exp(2 pi i r/N))
       D = fft1d(u[M-1,:] - u[0,:]),  E = fft1d(u[:,N-1] - u[:,0])
   S(q, r) = V(q, r) / (2 cos(2 pi q/M) + 2 cos(2 pi r/N) - 4),
       S(0, 0) = 0
   P = U - S        (U the EXISTING spectrum of u)
   ```

   so the periodic component's spectrum costs two 1D FFTs of the
   boundary differences plus one elementwise pass, and keeps the
   reuse of `target_spectra`. MEASURED here (section 5, M5): it
   fixes the zero-lock failure completely.
4. **Rotationally dependent aliasing** (S10). Sampling and the DFT
   do not commute with rotation: content near Nyquist and in the
   corners of the rectangular spectrum (outside the inscribed
   circle rho = 0.5 cycles/px) does not rotate with the image, and
   interpolation error grows towards Nyquist. Remedy: an annulus
   rho_min <= rho <= rho_max well inside 0.5 cycles/px.
5. **Angular resolution vs grid size.** A sample at radius r bins
   moves r dtheta bins for a rotation dtheta, so angular steps finer
   than about 1/r_max rad oversample the spectrum (0.22 deg at
   r_max = 256 bins; Ernould samples pi/2^n, 0.35 deg at 512 px,
   and fits the peak). Here 0.5 deg bins (360 over pi) plus a
   parabolic sub-bin fit measure 0.01-0.08 deg errors (M1, M4); 180
   bins lose little, 720 gain nothing. The error budget that matters
   is the IC-GN basin (3 deg about the normal from an exact
   translation, D5), so the resolution is two orders of magnitude
   finer than needed; what limits the seed in practice is the
   in-plane-axis bias of section 4.2 and noise (M6, M7).
6. **The scale axis.** Log-polar turns scale into a shift along log
   r; the S6 method then recovers scale and rotation together.
   Rotation only is safer here for three reasons: (a) the physical
   scale changes between a grain's patterns are the DD change of the
   beam scan and elastic strain, both of order 1e-3, far below a
   log-polar bin (ln(rho_max/rho_min)/N_r is of order 1e-2), so a
   scale estimate is noise; (b) in-plane-axis rotations off the PC
   produce a pseudo-scale of order 1e-2 (section 4.3) that is
   perspective, not isotropic scale, so seeding h11 = h22 from it
   is the wrong shape; (c) the 1D radial average (S4) is only
   available because scale is fixed, and it is what makes the step
   cheap.
7. **High-pass emphasis.** Reddy and Chatterji (S6) multiply the
   magnitude by H = (1 - X)(2 - X), X = cos(pi xi) cos(pi eta),
   -0.5 <= xi, eta <= 0.5, before log-polar resampling, to stop the
   low frequencies (oversampled by log-polar) from dominating
   (recalled from S6 and its reference implementations; check
   against the paper if it is ever adopted). Here the D4 band-pass
   is already applied and rho_min excludes the low band, and with
   POLAR (not log-polar) sampling there is no low-frequency
   oversampling to correct: no extra emphasis filter is recommended.
8. **Alternatives considered and not recommended.** De Castro and
   Morandi (S7): a direct search over trial angles with phase
   correlation at each, which costs one de-rotation and one 2D XC
   per trial. Pseudo-polar FFT (S11): interpolation-free but needs a
   non-standard transform on both backends and gives up the reuse.
   Image-domain polar resampling about the PC (S13): exact for
   rotation about the normal, but not translation invariant, so any
   in-plane-axis rotation breaks it; the FM's translation invariance
   is the point.

## 4. EBSD physics: which crystal rotations become which image motions

Frame: D1.3/D1.5, PC-centred binned pixels (x right, y down, z from
the sample towards the screen), DD in binned px; h maps reference
coordinates to target coordinates (D2); the conversion is D6 with
PC_rel = 0, i.e. H = diag(1, 1, 1/DD) Fe diag(1, 1, DD), normalised
by H33.

### 4.1 Exact cases

| lattice rotation | homography (normalised) | image motion |
|---|---|---|
| w3 = theta about z (detector normal) | h11 = h22 = cos - 1, h12 = -sin, h21 = sin, rest 0 | EXACT rigid rotation about the PC, any angle, no gnomonic distortion |
| w1 = a about x | h11 = 1/cos a - 1, h23 = -DD tan a, h32 = tan a/DD | PC point moves DD tan a px along -y; perspective |
| w2 = b about y | h22 = 1/cos b - 1, h13 = DD tan b, h31 = -tan b/DD | PC point moves DD tan b px along +x; perspective |

The 1 deg tilt pin of `TestPureRotations::test_out_of_plane_tilt`
(|h23| = DD tan w, 4.2 px at 1 deg on the 480 oracle) is the same
algebra. On the Si-indent reference geometry (DD = 379.4 binned px,
section 5 M3) one degree about an in-plane axis is 6.6 px, and the
map's largest rotation, 87.1 mrad (validation, Si-indent notebook
entry), is 33 px of translation.

### 4.2 First order: translation, local rotation, the FM bias

For small w = (w1, w2, w3), H = I + [[0, -w3, DD w2], [w3, 0,
-DD w1], [-w2/DD, w1/DD, 0]], and the displacement field is

```
u_x = -w3 y + DD w2 + (w2 x^2 - w1 x y)/DD
u_y =  w3 x - DD w1 + (w2 x y - w1 y^2)/DD
```

Its local rotation and local mean dilatation are

```
omega(x, y)  = w3 + (w1 x + w2 y) / (2 DD)
dilat(x, y)  = (3/2) (w2 x - w1 y) / DD      (per axis, mean)
```

So the in-plane-axis rotations rotate the pattern LOCALLY, by an
amount that grows linearly away from the PC and averages to zero
only over a subregion centred on the PC. The FM measures a
content-weighted mean of omega over the SR, hence

```
theta_FM ~= w3 + (w1 xbar + w2 ybar) / (2 DD)
```

with (xbar, ybar) the SR centroid relative to the PC. MEASURED
(M2): on the 480 oracle (xbar, ybar) = (+37.9, -38.1) px, DD =
242.4 px, the formula gives +-0.235 deg for 3 deg about x and y; the
Hann-windowed FM measures +0.231, -0.178, -0.230, +0.190 deg (the
reused-spectrum variants measure smaller biases, their content
weighting being less uniform). On the Si-indent geometry the SR
centre sits (-17.3, +165.8) px from the PC (PCy = 90 px of 512), so
theta_FM ~= w3 - 0.023 w1 + 0.218 w2: a 5 deg rotation about y
appears as a 1.1 deg in-plane rotation. That is inside the IC-GN
basin from a seed whose translation is right, but it matters for a
GATE (section 6.5). The complete initialisation knows w1 and w2
from the translation, so it can subtract the first-order term,
w3_hat = theta_FM - (w1 xbar + w2 ybar)/(2 DD); on the real rim
that removes half to two thirds of the over-read and changes no
fit's iteration count by more than 3 (M7, open question F5).

### 4.3 Pseudo-scale

The same field carries an isotropic dilatation (3/2)(w2 xbar - w1
ybar)/DD over an off-centre SR: 1.2 per cent for 3 deg on the 480
oracle and 5.7 per cent for a 5 deg w1 on the Si-indent geometry.
It is perspective, not lattice dilatation. A log-polar scale axis
would read it and seed the wrong shape (section 3, item 6).

### 4.4 Gnomonic distortion away from the PC

Band centre lines are gnomonic projections of great circles, hence
straight; band edges are hyperbolae whose width grows away from the
PC. The magnitude spectrum of a pattern is dominated by radial
streaks perpendicular to the bands. Under w3 the whole streak star
rotates rigidly (exact, 4.1). Under w1 and w2 the bands' screen
directions change non-uniformly (4.2) and the far part of the screen
is the most distorted: Ernould's worst case is exactly the rotation
whose translation axis crosses the asymmetric PC offset (S4
IV.2.4.1). Practical consequences: (a) the FM is a seed for w3, not
a measurement of it; (b) the partial (rigid) seed of a large w1/w2
leaves corner errors of 13-37 px on the oracle (M4) that the
complete seed reduces to 1-2 px.

### 4.5 Detector-fixed content

Everything fixed to the detector does not rotate with the lattice:
the smooth background (the D4 high-pass removes most of it, and
rho_min excludes its band), the crop boundary (section 3 item 3),
a dead-band cross (D4.4), scintillator dust and a transmitted beam
in TKD (S1 App. C). With the reused spectrum, the crop boundary is
the one that matters (M5).

## 5. Prototype measurements (2026-10-07, CPU, numpy, throwaway)

Common setup unless stated: the V3/V4 deformed-master oracle,
480x480, PC (0.4210, 0.5794, 0.5049), DD 242.35 px, the frozen
`(0.05, None)` band-pass and `border=0.05`, crop 432x432 = the D5
bounding box; FM over 360 angles in [0, pi), polar bilinear LUT in
PHYSICAL frequency (cycles/px), circular ZNCC of the 1D profiles,
search window +-30 deg, parabolic sub-bin; "raw" = the spectrum
`seed_spectra` already computes (fft2 of the ZMN'd crop),
"periodic" = raw minus the Moisan smooth spectrum, "hann" = a
SEPARATE fft2 of the Hann-windowed crop.

**M1 -- FM angle accuracy, pure rotation about the normal, 13 angles
in 0..15 deg plus -3 and -12** (`exp1_angle.py`), max |error| deg,
rho in [0.05, 0.35]:

| spectrum | abs | log |
|---|---|---|
| raw (reused) | 0.076 | 0.055 |
| periodic (reused + Moisan) | 0.079 | 0.031 |
| hann (extra FFT) | 0.015 | 0.012 |

Grid and band, periodic-log: n_theta 180 / 360 / 720 -> 0.036 /
0.031 / 0.032; rho_max 0.25 / 0.35 / 0.45 -> 0.046 / 0.031 /
0.027; rho_min 0.02 / 0.05 / 0.10 -> 0.024 / 0.031 / 0.032.
Radial step 1.0 bin and 0.5 bin identical (M4). Angular peak 0.90-
0.99 throughout, 1.000 at zero.

**M2 -- end to end, `fit_pattern(max_iterations=50)`**
(`exp2_endtoend.py`); FM = periodic-log; de-rotation of the target
by the engine's own bicubic spline at R(theta_hat) xi; then
`initial_guess` on the de-rotated crop; h0 = R(theta_hat) T(t):

| rotation vector (deg) | translation seed: conv, it, err px | FM seed: conv, it, err px | FM seed err px |
|---|---|---|---|
| (0, 0, 1) | yes, 5, 0.004 | yes, 2, 0.004 | 0.06 |
| (0, 0, 2) | yes, 8, 0.011 | yes, 2, 0.011 | 0.09 |
| (0, 0, 2.5) | NO, 50, 148 | yes, 2, 0.012 | 0.08 |
| (0, 0, 3) | NO, 50, 173 | yes, 2, 0.013 | 0.10 |
| (0, 0, 4) | NO, 50, 221 | yes, 2, 0.008 | 0.13 |
| (0, 0, 5) | NO, 50, 98 | yes, 2, 0.010 | 0.03 |
| (0, 0, -6) | NO, 50, 171 | yes, 2, 0.033 | 0.02 |
| (0, 0, 8) | NO, 50, 53 | yes, 3, 0.160 | 0.20 |
| (3, 0, 0) | yes, 9, 0.117 | yes, 9, 0.117 | 19.06 |
| (0, 3, 0) | yes, 9, 0.037 | yes, 9, 0.037 | 19.03 |
| (2, -1.5, 3) | NO, 50, 53 | yes, 7, 0.017 | 16.54 |
| (-2, 2, 4) | NO, 50, 68 | yes, 7, 0.091 | 19.19 |
| (1, 1, -3.5) | NO, 50, 34 | yes, 6, 0.077 | 12.59 |

(err = the V2 corner-displacement metric against the exact
homography; at 5 deg and beyond the SR corners leave the pattern,
`TestPureRotations` (a), so the 8 deg 0.160 px is the mirror
boundary, not the seed.) The translation-only phase-correlation peak
(the normalised cross-power IFFT maximum) is 1.000 at 0 deg but
already 0.037 at 1 deg, 0.031 at 2 deg and 0.018 at 2.5 deg: it
collapses BEFORE the seed fails, so it is useless as a gate. After
FM de-rotation it is 0.77-0.81 on pure rotations and 0.12-0.20 on the
combined ones, against 0.013-0.023 translation-only -- a clean
signal on these synthetics, which carry no detector-fixed content;
on real patterns it is the WRONG acceptance signal (M7, M8).

**M3 -- REAL Si-indent patterns** (`exp3_si.py`, read-only on
`AGH__Si_indent_1_512x672.h5oina`, binning 2, 512x622, reference
(10, 10), PC (328.3, 90.2, 379.4) px, crop 460x560; target (10, 13)
-- a different real pattern with independent noise -- rotated
synthetically about the reference PC by theta through the engine's
bicubic spline; fits with `max_iterations=200`):

| theta deg | FM error deg, raw-abs / raw-log / per-log / hann-log | translation seed: conv, it | FM seed: conv, it |
|---|---|---|---|
| default `(0.05, None)` | | | |
| 0 | -0.002 / -0.008 / -0.005 / -0.018 | yes, 7 | yes, 7 |
| 1 | -0.023 / +0.008 / +0.004 / -0.025 | yes, 19 | yes, 6 |
| 2 | -0.005 / -0.007 / -0.010 / -0.030 | yes, 124 | yes, 8 |
| 3 | -0.027 / -0.009 / -0.014 / -0.033 | NO, 200 | yes, 8 |
| -4 | -0.010 / -0.007 / -0.013 / -0.009 | NO, 200 (residual 1.93) | yes, 8 |
| 5 | -0.019 / -0.015 / -0.014 / -0.030 | NO, 200 (1.93) | yes, 8 |
| 8 | -0.020 / -0.016 / -0.016 / -0.048 | NO, 200 (1.95) | yes, 8 (0.189) |
| `(None, None)` | | | |
| 1 / 2 / 3 / -4 / 5 / 8 | per-log -0.015..+0.004; raw-log up to +0.049 | yes 14 / 26 / 32 / 47 / 70, NO at 8 | yes 5-8 at all |

Where both seeds converge they reach the same residual (0.1771-
0.1773 default, 0.0404 unfiltered, for the spline-resampled
targets; 0.1846 / 0.0422 for the un-resampled pair at 0 deg), and
the FM-seeded fits recover the imposed angle to about 1e-3 deg up to
5 deg and to 2.5e-3 / 5.8e-3 deg at 8 deg (residual 0.189 / 0.043:
the mirror boundary). Real-pattern FM angular peaks: 0.96-0.98.
On real data the translation seed's practical capture with
the default band-pass is about 2 deg with a 200-iteration budget and
below 2 deg with the default 50.

**M4 -- cost split and the complete initialisation**
(`exp4_cost_complete.py`, numpy single thread, per pattern, 432x432):

| step | ms |
|---|---|
| fft2, complex128 (what `seed_spectra` already does) | 10.4 |
| log1p(abs) with a per-slot median scale (the median is the cost; plain abs is an elementwise pass) | 5.3 |
| Moisan smooth spectrum, as prototyped (rebuilds the per-shape exp tables each call) | 9.9 |
| polar profile, LUT 360 x 130 x 4 = 187200 entries (1 bin radial step) | 2.0 |
| same at 0.5 bin (374400 entries) | 4.2 |
| 1D circular correlation + peak | 0.05 |
| de-rotation, bicubic evaluation of 186624 px | 20.0 |
| translation phase-XC (skimage, incl. its FFTs and the 24x24 upsampled DFT) | 26.3 |

The FM ANGLE (the data gate) costs 5.9 ms per slot once the
per-shape tables are precomputed and the batch is processed at
once (M10), against 37 ms for this unbatched translation seed
(fft2 + skimage XC); the de-rotation BRANCH (gather + second fft2 +
XC) costs about 1.3 translation seeds. Device ratios are MTP (the
gather is a CUDA kernel, the FFTs are cuFFT).

Partial (rigid) against complete (rotation vector (atan(-d_y/DD),
atan(d_x/DD), theta_hat) -> `fe_to_homography`, with d = R t the
PC point's displacement), oracle:

| rotation vector (deg) | partial: it, seed err px | complete: it, seed err px |
|---|---|---|
| (3, 0, 0) | 9, 19.06 | 4, 1.43 |
| (0, 3, 0) | 9, 19.03 | 4, 0.89 |
| (2, -1.5, 3) | 7, 16.54 | 4, 1.38 |
| (-2, 2, 4) | 7, 19.19 | 4, 1.31 |
| (1, 1, -3.5) | 6, 12.59 | 4, 0.84 |
| (3, 3, 5) | 17, 36.95 | 6, 1.98 |

Same final homographies to 1e-4 px. This is a much larger gain than
S1's 10 per cent, consistent with the short DD (0.505 Ny) of the
oracle: the gnomonic distortion that complete init captures grows
as 1/DD.

**M5 -- robustness: detector-fixed background and no high-pass**
(`exp5_robust.py`): the oracle multiplied by a detector-fixed
background (off-centre Gaussian plus a linear ramp, the same for
reference and target), noise-free:

| filter | raw-abs / raw-log | per-log | hann-log |
|---|---|---|---|
| `(0.05, None)` | <= 0.044 | <= 0.012 | <= 0.014 |
| `(None, None)` | errors -0.97, -1.99, -3.00, -5.02, -7.99 at 1, 2, 3, 5, 8 deg: LOCKED AT ZERO | <= 0.012 | <= 0.014 |

The reused raw spectrum is safe ONLY with a high-pass; the
Si-indent tutorial itself runs `filter_cutoffs=(None, None)`. The
periodic correction removes the failure. Unrelated pair (a second
orientation 20 deg about (0, 1, 1), i.e. a grain boundary): FM
returns 12.36 deg with angular peak 0.56, and the two phase peaks
are 0.0166 (translation) and 0.0218 (FM): both rows are junk, the
acceptance test picks one, the fit fails by the D2.6 contract as it
does today.

**M6 -- shot noise** (`exp6_noise.py`, Poisson at a full-scale count
of 50 / 20 / 10 on the background-modulated oracle; periodic
correction throughout), max |error| deg over 0..8 deg:

| magnitude, rho_max | (0.05, None) 50 / 20 / 10 | (None, None) 50 / 20 / 10 |
|---|---|---|
| abs, 0.20 | 0.121 / 0.203 / 0.843 | 0.290 / 0.402 / 1.518 |
| abs, 0.35 | 0.133 / 0.543 / 0.379 | 0.235 / 0.412 / 3.420 |
| sqrt, 0.20 | 0.159 / 0.192 / 1.018 | 0.528 / 0.778 / 2.171 |
| log, 0.20 | 0.176 / 0.178 / 1.041 | 0.581 / 0.837 / 3.289 |
| log, 0.35 | 0.365 / 0.939 / 0.814 | 0.517 / 0.869 / 3.418 |

Amplitude (abs) over a band ending at 0.20 cycles/px is the most
noise-robust; log (Ernould's choice) amplifies the white-noise floor
that dominates high radii. The angular peak tracks reliability
(0.85-0.88 at 50, 0.66-0.74 at 20, 0.44-0.62 at 10; 0.96 on the
rigidly rotated real pair; but 0.25-0.82 on the deformed rim, M7,
where FM still helps, so it is a diagnostic, not a gate). Fits on
these synthetic noise levels did not converge for EITHER seed in
100 iterations, so M6 says nothing end to end.

**M7 -- REAL deformed patterns, the indent's south rim**
(`exp7_si_rim.py`, `exp7b_rim_diag.py`; reference (10, 10); the 25
tutorial rim points rows 125-129 x columns 115-119 plus (120, 110),
(130, 125), (118, 122), (135, 118), (126, 108); `max_iterations=
500`; FM = periodic-amplitude, rho <= 0.20; three seeds per point:
T = the D5 translation seed, P = FM partial, C = FM complete):

| filter | T: converged, iterations total (median) | P | C |
|---|---|---|---|
| `(0.05, None)` | 9/30, 13234 (500); 2 of the 9 at a wrong optimum (residual 1.95 / 1.86 where FM reaches 1.14 / 0.53) | 23/30, 8456 (261) | 24/30, 7494 (214) |
| `(None, None)` | 30/30, 3503 (122) | 30/30, 2840 (92) | 30/30, 2721 (91) |

Same residual for all three wherever all three converge (median
0.3101 unfiltered). Per point, C against T unfiltered: 104 -> 39,
79 -> 28, 123 -> 66 iterations at the far rim; no change where the
FM angle is small. Under the default band-pass the far-rim points
go from the 500 cap to 30-73 iterations with C.

What the rim's rotations ARE (polar decomposition of the converged
unfiltered Fe, detector frame): w1 = 0.75-2.4 deg, w2 = -1.2 to
2.0 deg, twist w3 = 0.06-0.95 deg. So on this map the gain is NOT
about large twists: it is about the translation and the perspective
(next paragraph and M8). The FM angle mostly over-reads the twist:
theta_FM - w3 spans -0.41 to +0.86 deg over the 30 points, +0.63 to
+0.86 deg on the far rim (e.g. 0.922 against 0.255; 1.177 against
0.444), and the first-order formula of section 4.2 removes about
half to two thirds of it there (bias-corrected 0.486 against 0.255;
0.696 against 0.444); the one under-read is (130, 125), 0.018
against 0.426, w = (2.39, -1.17, 0.43). FM accuracy on deformed
real patterns is therefore about 0.1-0.7 deg, in line with S4's
0.25-0.5 deg on strained patterns, not the 0.02-0.05 deg of
the rigidly rotated pair (M3). The bias-corrected complete seed
changed no point's iteration count by more than 3.

The ZNSSD criterion at the seed (the D2.7 quantity, evaluated with
`fit_pattern(max_iterations=0)`), unfiltered: T 1.06-2.14 on every
rim point; P and C 0.26-0.69 wherever the de-rotation broke the
zero anchor of M8 (theta_FM >= 0.43 deg, 17 points), and between
0.19 lower and 0.07 higher than T where it did not (theta_FM <=
0.38 deg). The phase-correlation peak is the WRONG acceptance
signal on real data: T's peak is 0.09-0.10 (the detector-fixed
zero-shift peak of M8) and FM's 0.014-0.025
(the true, decorrelated Kikuchi peak), so "FM only if its peak is
higher" would have rejected FM at every rim point.

**M8 -- the D5 seed is anchored at zero shift on real Si**
(`exp10_anchor.py`, no fits): the full-band phase correlation of
`initial_guess` returns EXACTLY (0, 0) at every deformed point
tested (10 of 10, peak 0.094-0.104, both filters), where a
band-limited phase correlation (cross-power masked to rho in [0.02,
0.20] cycles/px) or the plain unwhitened cross-correlation of the
same ZMN'd crops returns 6-18 px, consistent with the converged
translations (e.g. (125, 115): band (+12, -11) default / (+13, -11)
periodic, plain (+10, -8), converged (+10.2, -7.0) at the PC). On
undeformed points it returns (0, 0) where the band-limited and plain
correlations give 1 px ((200, 30): (0, -1); (30, 220): (-1, 0)),
which is the order of the map's PC span (1.31 / 1.14 px, ledger 80).
Cause: whitening gives every frequency equal weight, and the
detector-fixed content (camera fixed-pattern noise, hot pixels,
scintillator texture), identical in reference and target, owns the
high frequencies where the Kikuchi signal is weak; the zero-shift
peak it makes (about 0.1) beats the decorrelated Kikuchi peak
(about 0.01). This is S1's on-axis TKD "anchor" (the transmitted
beam) in another form. It is a property of the FROZEN D5 seed on
real data, not of Stage F; the FM branch escapes it because the
de-rotation also rotates the target's fixed content (measured
escape from theta_FM of about 0.4 deg up). Unfiltered, IC-GN still
converges from (0, 0) on the rim (the 10-20 px is inside its
translation basin, at 80-150 iterations); with the default band-pass
it mostly does not (M7, 9/30). See open question F8.

**M9 -- non-square crop** (`exp8_nonsquare.py`, real Si, crop
460x560, target (10, 13) rotated about the PC, amplitude, rho <=
0.20): the physical-frequency LUT measures +0.002, +1.997, +4.988,
+7.990 deg at 0, 2, 5, 8 deg (angular peak 0.99-1.00); a LUT in bin
units on both axes (the plausible slip) measures +0.003, +2.018,
+5.054, +8.185 deg with the peak falling to 0.88 and 0.73: a bias of
about 2.3 per cent of the angle, the aspect ratio's shear. Inside the
IC-GN basin, so only a seed-level angle test at >= 5 deg on a
non-square crop with a band under 0.05 deg kills it. The composition
mutant T(t) R against R T(t) was indistinguishable here (t = 0 on
this pair): the two differ by |R t - t| = 2 sin(theta/2) |t|, 2.4 px
at theta = 8 deg and |t| = 17 px, so it also needs a seed-level
oracle with a large translation.

**M10 -- batched cost of the FM ANGLE path** (`exp9_cost2.py`, numpy
one thread, P = 16, per pattern; precomputed per-shape Moisan tables;
amplitude; rho <= 0.20 LUT):

| crop | fft2 (reference point) | Moisan S | abs(U - S) | profile | 1D corr | FM angle total |
|---|---|---|---|---|---|---|
| 432x432 | 6.23 ms | 3.66 | 1.46 | 0.74 (93600 LUT entries) | 0.03 | 5.89 ms |
| 460x560 | 9.80 ms | 5.20 | 2.00 | 0.78 (100800) | 0.03 | 8.01 ms |

So the FM angle on every slot costs about 0.8-0.95 of one batched
2D FFT, i.e. a fraction of the translation seed (whose fft2 is
reused and which adds the cross-power, an ifft2 and the upsampled
DFT). The Moisan elementwise outer combination is the largest part;
it can be fused with the abs on the device.

## 6. Recommended Stage F method

### 6.1 Per reference (EXTENDS `SeedState`, D21.5 clause (iv))

1. `fm_lut_index`, `fm_lut_weight`: the polar LUT for the crop shape
   (sr, sc): n_theta = 360 angles theta_k = k pi/360 in [0, pi);
   radii from rho_min to rho_max in steps of 1/min(sr, sc)
   cycles/px; sample point (fx, fy) = rho (cos theta_k, sin theta_k)
   in PHYSICAL frequency, bin coordinates (kx, ky) = (fx sc, fy sr)
   (NOT bin units on both axes: the Si crop is 460x560, M9),
   bilinear 4-neighbour indices taken modulo (sr, sc) into the
   UNSHIFTED spectrum, float64 weights. Shape (n_theta, n_rho, 4):
   93600 entries at 432 px and 100800 at 460x560 with rho_max =
   0.20 (under 1 MB per array). All references of a run share the
   crop shape, so one LUT per run would do; per reference fits the
   seam without a new object.
2. `fm_profile`: the reference's angular profile from
   `reference_spectrum` through 6.2 steps 2-4, zero-mean unit-norm.
3. A full-crop resident (or coordinate set) for the gather: `gather`
   evaluates a resident's SR pixels, while the Stage E translation
   seed correlates the WHOLE (sr, sc) crop, dead-band pixels
   included; a resident built on the crop's bounding box (mask all
   true) returns exactly that crop through the unchanged `gather`.
4. `dd_px` and the SR centroid (xbar, ybar) relative to the PC, for
   the complete initialisation (6.2 step 9).
5. Options: rho_min, rho_max, n_theta, theta_max, tau_theta (6.3).

### 6.2 Per sub-batch

A wrapper installed as the module global
`_batched.seed_homographies` (D21.5 clause (ii)).

1. Translation seed exactly as Stage E: h_T (P, 8). Ungated slots
   return it bitwise.
2. Periodic correction: from the ZMN'd target crops' first and last
   rows and columns, two batched 1D FFTs and three precomputed
   per-shape tables -- (1 - exp(2 pi i q/M)), (1 - exp(2 pi i r/N)),
   1/(2 cos(2 pi q/M) + 2 cos(2 pi r/N) - 4) with the (0, 0) entry
   zeroed -- give S (section 3 item 3); P = U - S into a NEW array
   (U is `target_spectra`, shared with step 1 and step 8; never in
   place).
3. A = |P| in the seed precision's real type.
4. Profile: g = A.reshape(P, -1)[:, idx] (P, n_theta, n_rho, 4);
   profile = (g * w).sum(-1).mean(-1): a fixed-order reduction, no
   scatter-add, no atomics, so bitwise run to run on each backend
   (D21.7). Zero-mean unit-norm per slot.
5. Correlation: c = ifft(conj(fft(p_ref)) * fft(p_tgt)).real,
   length n_theta (circular, because the profile is pi-periodic: no
   window); lags wrapped to (-n_theta/2, n_theta/2]; lags with |lag|
   pi/n_theta > theta_max masked to -inf; argmax (first index on
   ties, both namespaces); parabolic sub-bin offset delta =
   (c[k-1] - c[k+1]) / (2 (c[k-1] - 2 c[k] + c[k+1])) when the
   denominator is negative, else 0; theta_hat = (lag + delta)
   pi/n_theta; angular peak a = c[k] kept as a diagnostic. Sign,
   MEASURED: with c[k] = sum_j p_ref[j] p_tgt[j + k] and theta_k =
   atan2(fy, fx) in the y-down array frame, theta_hat is the
   rotation about +z of the D1.5 frame, h21 = +sin(theta_hat).
6. Gate (6.5): gated = |theta_hat| >= tau_theta [data gate], or
   `extras["fm_gate"]` [CrystalMap gate, host-filled], or
   `extras["fm_force"]` [retry pass]. Only gated slots go on; the
   batch can be compacted to the gated slots (a host-visible count)
   or run masked at full P (simpler, deterministic shapes).
7. De-rotation: carried matrices R(theta_hat) (P, 3, 3) float64 in
   the D1.3 frame (rotation about the PC, the frame origin);
   values = kernels.gather(full-crop resident, batch.coefficients,
   matrices), reshaped to (P, sr, sc). A pure rotation never makes a
   non-finite coordinate, but `coordinate_ok` is honoured anyway.
8. Translation on the de-rotated crops: the Stage E phase-XC code,
   unchanged (ZMN, fft2, the guarded cross-power with
   `reference_spectrum`, ifft2, argmax, the upsampled DFT at
   `upsample_factor`): t with derotated(xi + t) = reference(xi).
   (Open question F8 records the band-limited variant.)
9. Composition, D1.3 frame. Partial: W0 = R(theta_hat) T(t), h0 =
   (c - 1, -s, c tx - s ty, s, c - 1, s tx + c ty, 0, 0). Complete
   (recommended): d = R t, w1 = atan(-d_y/DD), w2 = atan(d_x/DD),
   w3 = theta_hat, R = exp([w]x), h0 = fe_to_homography(R, 0, DD)
   (D6 with PC_rel = 0; a batched xp transcription or host math on
   (P, 3)). The Ernould rotation-centre corrections (S2 E.2-E.3)
   vanish because the de-rotation is about the PC. A centroid-aware
   variant (F4) reads t as the displacement at (xbar, ybar) rather
   than at the PC, which removes the factor (1 + ybar^2/DD^2) = 1.19
   by which the PC-based reading over-estimates w1 on the Si
   geometry (the FM translations of M7 exceed the converged PC
   translations by about that much).
10. Acceptance: evaluate the D2.7 criterion at h_T and at h_FM with
    the engine's own kernels (`gather` on the SR resident, then
    `final_criterion`), and keep the row with the LOWER criterion;
    a non-finite criterion loses; a NaN h_T with a finite h_FM is
    NOT rescued (the D21.5 failure meaning is kept: a NaN h_T means
    the crop itself is unusable). Measured on the rim (M7): T
    1.06-2.14, FM 0.26-0.69 where the de-rotation broke the anchor,
    within -0.19/+0.07 of T where it did not. The phase-correlation
    peak must NOT be used (M7, M8).

### 6.3 Constants (each MTP at the Stage F gates)

| constant | proposed | evidence |
|---|---|---|
| n_theta | 360 over [0, pi) (0.5 deg bins) | M1, M4: 180 loses <= 0.005 deg, 720 gains nothing |
| radial step | 1 bin (1/min(sr, sc) cycles/px) | M4: identical to 0.5 bin at half the cost |
| rho_min | 0.05 cycles/px (the D4 high-pass radius) | M1: 0.02-0.10 all <= 0.032 deg |
| rho_max | 0.20 cycles/px | M6: most noise-robust; M1: 0.25-0.45 equally fine noise-free; M8: the detector-fixed content lives above it |
| magnitude | amplitude of the periodic-corrected spectrum | M5 (correction needed without high-pass), M6 (amplitude beats log and sqrt under noise) |
| theta_max | 30 deg | physical range of an in-grain twist; disposes of the 180 deg ambiguity |
| tau_theta | 0.5 deg proposed, 1.0 deg the cheaper alternative | M3: D5 seed at 19 / 124 iterations at 1 / 2 deg on real Si; M7: the anchor breaks from about 0.4 deg, and 0.5 / 1.0 deg route 13 / 4 of the 15 rim conversions (6.5 G2) |
| acceptance | lower ZNSSD at the seed | M7: 0.26-0.69 against 1.06-2.14 |
| initialisation | complete | M4: 6-17 -> 4-6 iterations; M7: 2840 -> 2721 unfiltered, 23 -> 24 of 30 converged default |
| angular peak a | diagnostic only, no threshold | M6 vs M7: synthetic noise says a >= 0.45 is safe, but real deformed rim points sit at a = 0.25-0.82 with FM helping; the ZNSSD acceptance is the guard |

### 6.4 Failure semantics

As 6.2 step 10 and plan 11.3 (vii): the wrapper never introduces a
NaN row; it can only REPLACE a finite translation row by a finite FM
row that won the acceptance test. Ungated slots return h_T
bitwise, so the Stage E parity pins hold on every ungated slot.

### 6.5 Gate options

(G1) **CrystalMap twist** (Johan's recorded recommendation, plan
11.3 (iv)): for each fitted point, the detector-frame misorientation
dR = C_t C_ref^-1 with C the D7 crystal-to-detector chain (it
contains the diag(1, -1, 1) reflection; the conjugated product is
still a proper rotation); its twist about z by the swing-twist split,
twist = 2 atan2(q_z, q_w) of dR's quaternion, wrapped to (-180, 180]
deg; gate |twist| >= tau_ori. Host-side, keyed by `pattern_index`
into `extras["fm_gate"]`, zero device cost on ungated slots. Limits:
Hough noise is about 0.5 deg on good patterns (the D5 note) and
1-1.2 deg per rotation component on deformed IF steel (S1 section
5), and the reference's own error adds (about sqrt 2 on the
difference), so at tau_ori = 1.5 deg a true 2.5 deg twist is missed
with probability Phi((1.5 - 2.5)/(0.5 sqrt 2)) = 8 per cent at
0.5 deg noise and Phi(-0.64) = 26 per cent at 1.1 deg; the deformed
zones where FM matters are where indexing is worst; it cannot see
the zero anchor of M8 at all (on the Si rim the twist never reaches
1 deg, M7, yet FM applied to every rim point turns 15 of the 30 from
non-converged to converged under the default band-pass); and a
CrystalMap may carry identity rotations (the Si-indent reader does;
the tutorial rebuilds it from the file's Euler angles).

(G2) **Data gate**: |theta_hat| from 6.2 steps 2-5 on EVERY slot,
gate at tau_theta. MEASURED CPU cost 5.9 / 8.0 ms per slot at 432x432
/ 460x560, about 0.8-0.95 of one batched 2D FFT (M10); accuracy
0.02-0.05 deg on rigidly rotated real patterns (M3) and 0.1-0.7 deg
on deformed ones (M7, an over-read that makes the gate fire on large
in-plane-axis rotations too, which is harmless under the ZNSSD
acceptance and useful against the anchor). No dependence on indexing
or on the CrystalMap at all. The threshold matters on the rim: of
the 15 points FM converts there (default band-pass), tau_theta =
0.5 deg routes 13 (theta_FM 0.51-1.69 deg; the two it misses sit at
0.43 and 0.49 deg) and tau_theta = 1.0 deg routes 4, plus (129, 115)
in both cases, whose D5 fit "converges" at residual 1.86 against
FM's 0.53.

(G3) **Post-fit retry**: a second runner pass over the failed points
with `extras["fm_force"]`, a runner change (plan 11.3 (iv)). Caveat
from V8(b): with a large budget the default path can CONVERGE to a
wrong optimum about 50 px away (291 to 728 iterations on the ramp),
and M7 shows the same on real data (2 of the D5 seed's 9 converged
rim points sit at residual 1.95 / 1.86 where FM reaches 1.14 /
0.53), so the retry must key on non-convergence OR a residual above
a per-grain threshold, not on `converged` alone.

Recommendation for the spec: G2 as the gate, the ZNSSD acceptance as
the guard, G3 as the safety net, G1 carried in `extras` as an
optional host pre-filter; FM itself opt-in until open question F3 is
measured on the whole map.

### 6.6 How the CPU joins (plan 11.3 (viii))

`backend="cpu"` calls `initial_guess` per point inside
`fit_pattern`, whose skimage FFT is not reusable. Under G1 the CPU
routes only the gated points through the numpy seam (batched) and
fits them with `h0=row` through the Stage D plumbing, as plan 11.3
records. Under G2 the CPU must compute theta_hat for every point:
one extra fft2 of the crop plus 6.2 steps 2-5, about 15-20 ms per
pattern (M10 plus the unbatched fft2 of M4; an estimate) against
roughly 1.2 s of fitting per pattern-worker on the Si-indent map
(2.34 h, 8 workers, 57772 points: ledger 82), i.e. about 1-2 per
cent. Ungated points stay bitwise on the pre-Stage-F path either way.

### 6.7 Determinism and parity

- Per backend: every FM reduction is a gather plus a fixed-shape sum
  (no bincount, no atomics), argmax ties break to the first index on
  both namespaces, and the acceptance criterion reuses the D21.6
  kernels, so D21.7's run-to-run bitwise claim survives.
- GPU against CPU: theta_hat differs by FFT rounding (cuFFT against
  pocketfft) and by the f32 gather under "mixed"; the GATE can flip
  for a slot within rounding of tau_theta, and the acceptance can
  flip where the two criteria are nearly equal, routing a slot
  differently on the two backends. Parity is therefore on CONVERGED
  results within the D21.8 band (plan 11.3 (v)), plus a recorded
  routing-agreement count, never on seeds.
- `seed_precision="complex64"`: the FM runs on the same spectra; the
  effect on theta_hat is MTP (expected far below the 0.5 deg bin).

### 6.8 Cost model on the GPU (inference, to be measured)

D21: the seed is about 62 per cent of device time under "mixed" with
the complex128 seed (2.16 ms per pattern), median 14 iterations. G2
adds about one 2D-FFT equivalent of elementwise work, two 1D FFTs
per slot and a 100k-entry gather on every slot; each GATED slot adds
about 1.3 translation seeds (gather + fft2 + phase XC, M4 ratio) plus
two criterion evaluations (about two IC-GN iterations). Gated slots
are the ones that otherwise run to the 50 / 200 / 500 cap and hold
a lockstep batch open (D21.6), so the branch should pay for itself
through the batch tail rather than the median; the whole-map run of
F3 decides.

## 7. Open questions

Each with the conservative default in force and the measurement
that resolves it.

F1. **Spectrum reuse vs a windowed FFT.** In force: reuse
    `target_spectra` with the periodic-plus-smooth correction (M1,
    M5). Resolves: the V-F (a) angle sweep on the oracle and on real
    Si under both filter settings; a Hann FFT is adopted only if the
    corrected reuse misses the angle band.
F2. **Magnitude and band.** In force: amplitude, rho in [0.05, 0.20]
    cycles/px. Resolves: the same sweep, the rim (M7) and the Si
    wafer of V5.
F3. **Default on or off, and which gate.** In force: OPT-IN (FM off
    unless asked), so no existing pin moves. Candidates G1, G2, G3
    (6.5). Resolves: the whole Si-indent map on both backends, FM off
    / G1 at 1.5 deg / G2 at 1.0 and 0.5 deg / G2 + G3, under both
    `(0.05, None)` and `(None, None)`: converged count, residual
    median, iterations (median and tail), wall time, routing
    agreement CPU against GPU.
F4. **Partial vs complete, PC-based vs centroid-aware.** In force:
    complete, PC-based (Ernould's), for FM-routed rows only.
    Resolves: M4 and M7 repeated with the centroid-aware solve of
    6.2 step 9; revert to partial if complete ever loses iterations.
F5. **Bias correction of theta** (section 4.2). In force: none (it
    changed no rim point's iterations by more than 3, M7). Resolves:
    a V-F in-plane-axis sweep; adopt only if the gate fires
    spuriously enough to cost time.
F6. **Search window and the 180 deg ambiguity.** In force: |theta|
    <= 30 deg, no two-candidate test. Resolves: nothing in a grain
    should exceed it; the S6 two-candidate test only for a use case
    beyond 30 deg.
F7. **Retry criterion.** In force: no retry until F3; then
    non-converged OR residual above a per-grain threshold (V8(b),
    M7).
F8. **The D5 zero anchor** (M8). In force: D5 unchanged everywhere
    (the frozen default path), and the FM branch reuses the Stage E
    phase-XC unchanged, because the de-rotation escapes the anchor
    from about 0.4 deg. Candidate: a band-limited cross-power (rho in
    [0.02, 0.20] cycles/px) inside the FM branch only (a recorded
    divergence from D5 confined to gated slots), and, separately and
    for Johan, the same change to D5 itself, which would move the
    default path and every pin that depends on it. Resolves: M8 on
    the whole map (fraction of points whose D5 seed is exactly zero
    while the band-limited one is not; their iteration and
    convergence counts with each seed), and the Si wafer of V5
    (whose "fits do not track translations", D4.4, may be the same
    anchor; a hypothesis, not measured).

## 8. Recorded defaults (for Johan's approval)

1. Reuse the translation seed's target spectra; Moisan periodic-
   plus-smooth correction instead of a window (no extra 2D FFT).
2. Polar (not log-polar) sampling of the upper half-plane, 360
   angles, 1 bin radial step, physical-frequency coordinates,
   bilinear LUT; the 1D radial-mean signature (Ernould's reduction).
3. Amplitude, rho in [0.05, 0.20] cycles/px; no extra emphasis
   filter (the D4 band-pass and rho_min do that job).
4. Circular ZNCC of the profiles, +-30 deg search window, parabolic
   sub-bin peak; no 180 deg two-candidate test.
5. Rotation only, no scale.
6. De-rotate the TARGET about the PC (the D1.3 origin) through
   `kernels.gather` on a full-crop resident; the Stage E phase-XC
   for the residual translation; W0 = R(theta) T(t).
7. Complete initialisation (PC-based) for FM-routed rows.
8. Acceptance by the lower ZNSSD criterion at the seed (the IC-GN's
   own objective, through the D21.6 kernels); never by the
   phase-correlation peak; a failed FM estimate returns the
   translation row, never NaN.
9. FM opt-in until F3 is measured; G2 at tau_theta = 0.5 deg as the
   recommended gate (1.0 deg the cheaper alternative), G1 carried in
   `extras`, G3 as a runner retry pass keyed on non-convergence or a
   high residual.
10. Parity on converged results within the D21.8 band, with a
    recorded routing-agreement count; seeds never pinned bitwise
    across backends.

## 9. Oracles and mutants worth designing

- V-F (a) angle sweep: pure w3 in 0..10 deg, both signs, on the 480
  oracle and on real Si rotated about its PC (non-square crop);
  theta_hat within an MTP band (measured 0.01-0.08 deg).
- V-F (b) capture: the M2 table as a test (the FM seed converges
  where the D5 seed fails), plus the V8(a) about-normal ramp (FM
  rescues it per point, without propagation).
- V-F (c) combined rotations: the M2 / M4 combinations; complete
  init iteration counts.
- V-F (d) background lock: M5's detector-fixed background under
  `(None, None)`; kills a mutant that drops the periodic correction.
- V-F (e) anchor: a synthetic pair carrying the SAME fixed-pattern
  noise image added to reference and target (detector-fixed) and a
  10-20 px true translation; reproduces M8's (0, 0) for the D5 seed,
  and tests the ZNSSD acceptance (and the band-limited variant if
  F8 adopts it).
- Mutants: de-rotation by +theta instead of -theta (the seed doubles
  the rotation); composition T(t) R instead of R T(t) (they differ
  by 2 sin(theta/2) |t|, 2.4 px at 8 deg and 17 px, so only a
  SEED-level oracle with a large translation kills it); bin-unit LUT
  on a non-square crop (M9: 2.3 per cent of the angle); log-polar
  instead of polar (a scale leak); angles over 2 pi instead of pi
  (half the resolution, the doubled-angle artefact); NaN instead of
  the translation row on a failed FM; an in-place periodic
  correction that corrupts `target_spectra` for the translation
  step; argmax over the unmasked correlation (a spurious > 30 deg
  lag); acceptance by the phase-correlation peak (M7: FM rejected at
  every rim point).

## 10. References with links

- Ernould et al. 2020, Acta Mater. 191:131-148,
  https://doi.org/10.1016/j.actamat.2020.03.026,
  https://hal.univ-lorraine.fr/hal-03006680
- Ernould et al. 2021, Ultramicroscopy 221:113158,
  https://doi.org/10.1016/j.ultramic.2020.113158,
  https://hal.univ-lorraine.fr/hal-03138727
- Ernould et al. 2020, Scripta Mater. 185:30-35,
  https://doi.org/10.1016/j.scriptamat.2020.04.005,
  https://hal.univ-lorraine.fr/hal-03006691
- Ernould, PhD thesis, Universite de Lorraine 2020 (2020LORR0225),
  https://hal.univ-lorraine.fr/tel-03254732
- Pan, Wang, Tian 2017, Opt. Eng. 56:014103,
  https://doi.org/10.1117/1.OE.56.1.014103
- Reddy, Chatterji 1996, IEEE TIP 5:1266-1271,
  https://doi.org/10.1109/83.506761
- De Castro, Morandi 1987, IEEE TPAMI 9:700-703,
  https://doi.org/10.1109/TPAMI.1987.4767966
- Chen, Defrise, Deconinck 1994, IEEE TPAMI 16:1156-1168,
  https://doi.org/10.1109/34.387491
- Moisan 2011, J. Math. Imaging Vis. 39:161-179,
  https://doi.org/10.1007/s10851-010-0227-1,
  https://helios2.mi.parisdescartes.fr/~moisan/p+s
- Stone, Tao, McGuire 2003, J. Vis. Commun. Image Represent.
  14:114-135
- Keller, Shkolnisky, Averbuch 2005, IEEE TPAMI 27:969-976
- Derrode, Ghorbel 2001, Comput. Vis. Image Underst. 83:57-78,
  https://doi.org/10.1006/cviu.2001.0922
- Foden et al. 2019, Ultramicroscopy 207:112845,
  https://doi.org/10.1016/j.ultramic.2019.112845
- Britton, Wilkinson 2012, Ultramicroscopy 114:82-95,
  https://doi.org/10.1016/j.ultramic.2012.01.004
- Maurice, Driver, Fortunier 2012, Ultramicroscopy 113:171-181,
  https://doi.org/10.1016/j.ultramic.2011.10.013
- Ruggles et al. 2018, Ultramicroscopy 195:85-92,
  https://doi.org/10.1016/j.ultramic.2018.08.020
- Vermeij, Hoefnagels 2018, Ultramicroscopy 191:44-50,
  https://doi.org/10.1016/j.ultramic.2018.05.001
- Smith 1981, Ultramicroscopy 6:201-204,
  https://doi.org/10.1016/S0304-3991(81)80199-4
