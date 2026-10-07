<!--
INSERTABLE BLOCK for specs/roadmap.md, drafted 2026-10-07 by the
Stage F spec workflow (no repo file was edited).
ANCHOR: it REPLACES the one-paragraph note at the foot of the Stage E
section ("Stage F (Fourier-Mellin initial guess, both backends, behind
the D21.5 seed seam) is planned after Stage E (plan section 11.3);
not commissioned."), immediately before the `---` that precedes
"# Feature path: NLPAR". Ledger numbers 200-215 move with the main
session's renumbering.
-->

## Stage F -- Fourier-Mellin rotation initial guess (commissioned 2026-10-06; D22, V10)

Plan open question 5 and the D5 deferral, commissioned 2026-10-06 under
Johan's extended overnight waiver; plan section 12 carries the tasks, open
questions F1-F14 and the recorded defaults (pre-accepted 2026-10-06). An
opt-in `fourier_mellin` keyword (`"off"` default, bitwise unchanged;
`"auto"`; `"always"`) on `EBSD.hrebsd_dic`: a rotation-only Fourier-Mellin
angle from the REUSED target spectra (periodic-Hann frequency stencil,
log-magnitude, polar radial-mean ZNCC), the target de-rotated through the
existing warp kernel, the Stage E translation step on the de-rotated crop,
the row `R(theta) T(t)`, kept only if its criterion at the seed is lower;
one xp-agnostic code path behind the D21.5 seam for both backends, the CPU
route at P = 1 as the parity oracle. Under `"auto"`: a host-side gate on the
input CrystalMap's twist about the detector normal (1.5 deg, failing open)
plus one post-fit retry of non-converged points. Spec-gate prototypes (CPU
only, throwaway, ledger 200-215): the translation seed fails from 2.5 deg on
the oracle and from about 2-3 deg on rigidly rotated real Si; the FM seed
converged every synthetic case to 30 deg and every real case from -6 to 10
deg in 1-11 iterations; about 0.6 per cent of the Si-indent map has an input
twist above 1.5 deg. Starts only after the Stage E gates pass.
- [ ] plan approved: the section 12.4 recorded defaults (pre-accepted by
  Johan 2026-10-06; approval recorded at the splice)
- [ ] failing tests first: `test_hrebsd_fourier_mellin.py` (default numpy-xp
  suite + gated `cupy_gpu` suite, V10(a)-(l)) incl. the angle, edge, row,
  capture, ramp-rescue, acceptance, gate and retry oracles; freeze, defaults,
  docstring and D21.12-literal pins updated
- [ ] implementation: `_hrebsd/_fourier_mellin.py` (angle, de-rotation,
  partial row, acceptance, CrystalMap gate) + the seam extension
  (`SeedState.fourier_mellin`, `extras` route and output keys) + the CPU
  route at P = 1 + the retry pass + `_run_chunks_gpu(seed_extras=)` + props;
  every V10 MTP pin measured and pinned; the REQUIRED real-data frame oracle
  of the gate (F13)
- [ ] adversarial review + bug injection (plan 12 item 4 mutants FM1-FM40,
  kills re-verified, device ones on the GPU machine) + fixes; coverage 100 %
  of the touched `_hrebsd` modules (default + gated combined); CPU default
  path bitwise unchanged (`fourier_mellin="off"` == no keyword); Stage E
  gated suite green and unchanged; Stage F gated suite 0 skipped under
  KIKUCHIPY_EXPECT_GPU=1; oldest-matrix + full suite green
- [ ] D22.15 performance record on the Si-indent data (whole map `"off"`
  against `"auto"` at both filter settings on both backends, far256 and
  patch C under `"always"`, the D5 anchor census of F8); docstring,
  CHANGELOG, tutorial markdown cell; signed commits pushed to
  origin/hrebsd-dic (no PR)
