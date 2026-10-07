<!--
INSERTABLE BLOCK for specs/roadmap.md, drafted 2026-10-07 by the
Stage F spec workflow and REVISED the same day against the
two-critic spec review (plan 12.5; no repo file was edited).
ANCHOR: it REPLACES the one-paragraph note at the foot of the
Stage E section ("Stage F (Fourier-Mellin initial guess, both
backends, behind the D21.5 seed seam) is planned after Stage E (plan
section 11.3); not commissioned."), immediately before the `---`
that precedes "# Feature path: NLPAR". Ledger numbers 200-216 move
with the main session's renumbering.
-->

## Stage F -- Fourier-Mellin rotation initial guess (commissioned 2026-10-06; D22, V10)

Plan open question 5 and the D5 deferral, commissioned 2026-10-06 under
Johan's extended overnight waiver and specified 2026-10-07; plan section 12
carries the tasks, open questions FQ1-FQ14 and the recorded defaults,
PENDING Johan's review (his instruction of 2026-10-07: spec only, then
stop; the 2026-10-06 waiver no longer covers Stage F). An opt-in
`fourier_mellin` keyword (`"off"` default, bitwise unchanged; `"auto"`;
`"always"`) on `EBSD.hrebsd_dic`: a rotation-only Fourier-Mellin angle from
the REUSED target spectra (periodic-Hann frequency stencil, log-magnitude,
polar radial-mean ZNCC, +-30 deg window), the target de-rotated through the
existing warp kernel, the Stage E translation step on the de-rotated crop,
the partial row `R(theta) T(t)`, kept only if its criterion at the seed is
lower; one xp-agnostic code path behind the D21.5 seam for both backends
(a new `SeedBatch.outputs` field, a 14-wide packed row on FM runs only), the
CPU route at P = 1 as the parity oracle. Under `"auto"` a host-side gate on
the input CrystalMap's twist about the detector normal (1.5 deg, failing
open); under `"auto"` and `"always"` one post-fit retry of non-converged
points from a forced FM row. Spec-gate prototypes (CPU only, throwaway,
ledger 200-216): the translation seed fails from 2.5 deg on the 480 px
oracle at the default budget; on rigidly rotated real Si it fails from 3 deg
with the default band-pass `(0.05, None)` but reaches 6 deg under the
production `(None, None)` -- above the Si map's largest input twist of 2.6
deg, so there the case for FM is iteration cost and the D5 zero-shift
anchor, not capture. The FROZEN recipe (the V8 angle, the partial row, the
acceptance) converged every oracle case measured, pure twists from -20 to
30 deg in 2 to 3 iterations to the exact seed's error (30 deg is the
window's edge), and cut the iterations on 16 real rim points by 17 per cent; the
second prototype's whitened recipe (not the frozen one) converged a 113-case
sweep to 30 deg and real rotations from -6 to 10 deg. About 0.6 per cent of
the Si-indent map has an input twist above 1.5 deg. Spec review 2026-10-07:
45 critic findings, 44 applied, 1 rejected (plan 12.5). Starts only after
the Stage E gates pass AND Johan has reviewed plan 12.4.
- [ ] plan approved by Johan: the section 12.4 recorded defaults (PENDING
  his review; nothing approved yet; his decision recorded in 12.4 with its
  date)
- [ ] failing tests first: `test_hrebsd_fourier_mellin.py` (default numpy-xp
  suite + gated `cupy_gpu` suite, V10(a)-(l)) incl. the angle, look-up-table,
  edge, row, capture, ramp-rescue, acceptance, gate, projection-link, retry,
  CPU-route and numpy-session oracles, the public-method forwarding and
  props arms and the new docstring test; the freeze, defaults and D21.12
  literal pins edited with dated comments
- [ ] implementation: `_hrebsd/_fourier_mellin.py` (look-up table, angle,
  peak, de-rotation, translation, partial row, acceptance, CrystalMap gate
  and routes) + the seam extension (`SeedState.fourier_mellin`, the `extras`
  route key, `SeedBatch.outputs`) + the CPU route at P = 1 + the retry pass
  under both FM modes + 14-wide rows on FM runs + `_run_chunks_gpu(
  seed_extras=)`, the lazy FM state, the inactive forced slot and the VRAM
  keyword + `ebsd.py` forwarding and props; every V10 MTP pin measured and
  pinned; the REQUIRED real-data frame oracle of the gate under its
  pre-registered rule (FQ13)
- [ ] adversarial review + bug injection (plan 12 item 4 mutants FM1-FM52,
  kills re-verified, device ones on the GPU machine) + fixes; coverage 100 %
  of the touched `_hrebsd` modules (default + gated combined); CPU default
  path bitwise unchanged (`fourier_mellin="off"` == no keyword); Stage E
  gated suite green and unchanged; Stage F gated suite 0 skipped under
  KIKUCHIPY_EXPECT_GPU=1; oldest-matrix + full suite green
- [ ] D22.15 performance record on the Si-indent data (whole map `"off"`
  against `"auto"` at both filter settings on both backends, far256 and
  patch C under `"always"`, the D5 anchor census of FQ8); docstring,
  CHANGELOG, the `hrebsd_dic.ipynb` bullet and the `hrebsd_si_indent.ipynb`
  markdown cell; signed commits pushed to origin/hrebsd-dic (no PR)
