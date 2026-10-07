<!--
INSERTABLE BLOCK for specs/2026-09-07-hrebsd-dic/plan.md, drafted
2026-10-07 by the Stage F spec workflow (no repo file was edited).
ANCHOR: append after the last Stage E subsection at splice time
(today "### 11.5 Spec-review disposition table (2026-10-06)" and its
table; any later 11.x disposition tables Stage E adds come first).
Ledger numbers 200 to 215 are provisional and move with the
main session's renumbering. A one-line edit of 11.3 is suggested at
the foot of this file.
-->

## 12. Stage F -- Fourier-Mellin rotation initial guess (commissioned 2026-10-06)

Plan open question 5 and the D5 deferral, planned in 11.3 as Stage F
(Johan decision 1 of 2026-10-06) and commissioned the same night when
Johan extended his overnight waiver ("continue with the merlin guess
implementation after the gpu implementation is done"; 11.4 approval
record, item 20 carried to this gate). Requirements D22 (drafted
2026-10-07) govern; oracles in validation V10; spec-gate
measurements in ledger entries 200 to 215; branch policy section 1
unchanged (hrebsd-dic only, no PR, never merged). Johan's Stage F
notes (11.3 (i) to (viii)) are binding and are carried into D22.
Stage F STARTS ONLY AFTER Stage E's gates have passed and its
commits are pushed (roadmap Stage E boxes ticked): D22 plugs into
the D21.5 seam and the D21.9.5 names, and the failing-tests agent
re-reads Stage E's final module before writing a test. THE PLAN GATE
IS THE APPROVAL GATE: no task below starts before every item of 12.4
is approved or changed. Johan pre-accepted the recommendations on
2026-10-06 (the waiver: "just continue automatically and accept
recommendations", extended to Stage F); the main session records the
approval at the splice.

Deliverables: `fourier_mellin` on `EBSD.hrebsd_dic` and
`run_hrebsd_dic` (D22.1) with the freeze, defaults and docstring pins
updated (the D15.4 amendment); the new private module
`_hrebsd/_fourier_mellin.py` -- the look-up table and
`FourierMellinState`, `build_fourier_mellin_state`, the
xp-agnostic angle `fourier_mellin_angles` (D22.3), the de-rotation,
translation and partial row (D22.4), the acceptance (D22.5) and the
gate `twist_about_detector_normal` (D22.7); the seam extension in
`_batched.py` (`SeedState.fourier_mellin`, the `extras` route and
output keys, the host-side skip and the masked full-P branch; D22.6);
the CPU route at P = 1 inside the chunk function and the retry pass
in `run_hrebsd_dic` (D22.8, D22.10); `_run_chunks_gpu(seed_extras=)`
and the VRAM model keyword (D22.11, D22.18); the D22.11 combination
raise and the amended D21.12 literal; the two props (D22.9); the
docstring, the CHANGELOG entry and the tutorial markdown cell
(D22.16); every V10 MTP pin measured and pinned; the D22.15
performance record.

Files touched outside `_hrebsd/`, kept to the Stage E minimum for
merge hygiene: `signals/ebsd.py` (inside the fork-only `hrebsd_dic`
method, its docstring and its prop loop only), `CHANGELOG.rst` (one
append in the HREBSD block), `doc/tutorials/hrebsd_si_indent.ipynb`
(one markdown cell), the fork-only `tests/test_signals/
test_ebsd_hrebsd_dic.py`, `tests/test_indexing/test_hrebsd_engine.py`
and `tests/test_indexing/test_hrebsd_gpu.py` (one literal, with a
dated comment), and the new `tests/test_indexing/
test_hrebsd_fourier_mellin.py`. NEVER `_spherical/*`, the root
`conftest.py`, `pyproject.toml` or `doc/user/installation.rst`.
Spec files: this spec commit edits `specs/roadmap.md` (the "Stage F"
section replaces the one-paragraph Stage F note at the foot of the
Stage E section, before the `---` that precedes "Feature path:
NLPAR") and, inside this folder, requirements.md (D22 and its dated
amendments), validation.md (V10) and this plan; no constitution
amendment is needed (no new dependency, no new device, the D21
determinism and float scoping cover the FM path).

1. Failing tests first, keyed to V10, in the new
   `tests/test_indexing/test_hrebsd_fourier_mellin.py` (the
   `test_hrebsd_gpu.py` layout; the `cupy_gpu` fixture and the
   fake-cupy helpers DUPLICATED, D21.14.1), targeting the frozen
   names of D22.6, D22.7 and D21.9.5 (argument lists the failing
   tests fix are recorded in the module docstring, the D21.9.5
   rule). Default-suite classes: `TestFourierMellinSwitch` (V10(a),
   with the check order, the D22.11 raise and the docstring test),
   `TestFourierMellinAngle` (V10(b), including the planted-correlation
   peak-search units and the non-square arm),
   `TestFourierMellinEdgeTreatment` (V10(c)),
   `TestFourierMellinSeedRows` (V10(d)), `TestFourierMellinCapture`
   (V10(e); the starred arms default, the rest weekly),
   `TestFourierMellinRampRescue` (V10(f)),
   `TestFourierMellinAcceptance` (V10(g)), `TestFourierMellinGate`
   (V10(h)), `TestFourierMellinRetry` (V10(i)),
   `TestFourierMellinCpuRoute` (V10(j)) and
   `TestFourierMellinNumpySession` (V10(k), with the import-hygiene
   arms). Gated classes (V10(l)): `TestGatedFourierMellinContract`,
   `TestGatedFourierMellinParity`,
   `TestGatedFourierMellinDeterminism`, `TestGatedFourierMellinVram`
   and `TestGatedFourierMellinThroughput` (record only). Edits of
   existing pins, each with a dated comment: `FROZEN_SIGNATURE` gains
   `("fourier_mellin", KEYWORD_ONLY, "off")` after
   `seed_from_neighbors`; the `hrebsd_dic` docstring test gains the
   keyword; `test_run_defaults_are_frozen` gains `fourier_mellin`;
   `SEED_FROM_NEIGHBORS_GPU_MESSAGE` becomes the D21.12-amended
   literal. The V9 `extras == {}` asserts are NOT edited (FM off keeps
   them true). Every device band and device-side literal carries a
   `FIXME-pin` placeholder, and so does every literal that needs the
   new code path; only CPU-side premises are measured at this gate:
   the G1/G3/G4 angle premises through a local re-implementation in
   the test module, the D5 failure premises of (e), (f) and (g), and
   the V8 ramp premise. The default suite's wall time is recorded at
   this gate. The failing-tests commit is not pushed alone (section
   1).
2. Implementation, in dependency order: the D22.1 keyword plumbing
   and check order with the D22.11 raise and the D21.12 literal ->
   `_fourier_mellin.py` under numpy (the look-up table in physical
   frequency, the window stencil on the reused spectrum, the
   magnitude, the radial-mean profile, the ZNCC, the windowed peak
   with lowest-index ties; the box resident; the de-rotation through
   `kernels.gather`; the Stage E translation step reused on the
   de-rotated crops; the partial row; the acceptance through
   `kernels.final_criterion`; the never-NaN fallbacks) until V10(b)
   to (d) and (g) pass -> the seam extension in `_batched.py` (the
   host-side route check, `h_T` first, the masked full-P branch, the
   output keys) -> the gate `twist_about_detector_normal` and its
   routing rules -> the CPU route at P = 1 in the chunk function,
   paired by the Stage D blockwise index, with the per-reference FM
   state built once per reference -> the retry pass and the props ->
   `_run_chunks_gpu(seed_extras=)`, the GPU retry call and the VRAM
   model terms -> the docstring and `ebsd.py` prop loop -> CHANGELOG
   and the tutorial cell (after the performance record). The CPU
   default path is not edited beyond the branch points that the
   `"off"` value skips.
3. Measurement debt at the implementation gate (validation, ledger
   after the last Stage E entry, recipe and machine ID each): every
   V10 MTP pin; the numpy session against the CPU route (bitwise or
   banded, D22.14); `theta_hat` parity and the acceptance flip count
   on the device at both seed precisions (F14); the FM terms of the
   VRAM model, calibrated separately; B invariance with FM on; the
   CPU and GPU costs of D22.15; and, REQUIRED, the gate's real-data
   frame oracle (V10(m)(1), F13) and the whole-map `"auto"` record
   at both filter settings on both backends (V10(m)(2); F1, F2, F3,
   F7), with the D5 anchor census of F8 taken from the same run.
   Every gated measurement runs on the pinned overlay of D21.15;
   every test run on the shared machine stays at `-n 2` or below.
4. Adversarial review (two reviewers -- fidelity/theory: the angle
   recipe against Ernould and the measured physics of ledger 201,
   the frame algebra of the gate against D7 and D1.3, the
   composition and acceptance against D22.4 and D22.5, the never-NaN
   paths, the determinism argument with the masked full-P branch;
   conventions/integration: the seam extension against D21.5 clause
   (iv), the Stage E pins left green, the check order, the import
   direction, coverage 100 % of the touched `_hrebsd` modules,
   docstrings, no stage letters, merge hygiene), THEN bug injection
   ALONE, each mutant re-injected after the fixes to verify its
   kill. Mutation list (FM-prefixed to keep Stage E's M1 to M53
   distinct; each dies by a named V10 oracle or is recorded
   reviewed-equivalent with its argument; D default suite, G gated):
   FM1 de-rotation by `R(-theta)` (D: (e) capture, (d) seed error);
   FM2 the row composed as `T(t) R` ((d) the projection-centre-shift
   fixture, the ONLY killer: on pure twists t is near 0); FM3 a
   bin-unit look-up table ((b) the G3 non-square arm at 5 and 8 deg,
   the ONLY killer: the bias stays inside the basin, ledger 208);
   FM4 angles over [0, 2 pi) ((b) the table pin and the sweep); FM5
   a failed FM estimate returning NaN instead of `h_T` ((d) planted
   failures); FM6 the edge treatment written in place into
   `target_spectra` ((c) the bitwise spectra and unrouted-row
   asserts); FM7 the edge treatment dropped ((c) the G4 lock arm at
   `(None, None)`); FM8 the search window dropped ((b) the planted
   40/5 deg correlation); FM9 the parabolic offset's sign flipped
   ((b) the off-grid twists, with `FM_ANGLE_TOL_DEG` pinned below
   the flipped error); FM10 the acceptance inverted ((g) G5 and the
   planted wrong angle, (e) capture); FM11 the acceptance dropped
   ((g) the planted wrong angle and the tie arm); FM12 acceptance by
   the phase-correlation peak ((g) G5: the anchor's peak wins, the
   FM row is refused and the fit fails or slows, against the frozen
   rule); FM13 a NaN `h_T` rescued by a finite FM row ((d)); FM14
   ties keeping `h_FM` ((g) the tie arm, `applied` False); FM15 the
   route ignored, FM applied to every slot ((k) unrouted rows bitwise
   Stage E, (j) unrouted points bitwise `"off"`); FM16 FM state built
   or `extras` written on `"off"` runs ((a) spies, the V9
   `extras == {}` asserts); FM17 the gate conjugated the wrong way
   (`M^T R_s M`) or without the y flip ((h) the sign pins on the
   tilted detector); FM18 the symmetry reduction dropped ((h) the
   symmetry-equivalent arm); FM19 the total misorientation angle
   gated instead of the twist ((h) the pure out-of-plane arm); FM20
   the signed twist compared ((h) negative twists); FM21 `>` instead
   of `>=` ((h) the boundary arm); FM22 the gate failing closed on
   unusable orientations ((h) the unindexed, NaN and other-phase
   arms); FM23 every point gated against the map's first reference
   ((h) the two-grain arm); FM24 a non-converged retry replacing the
   first result ((i) the unrelated-pattern point); FM25 the retry on
   converged, masked or D2.6-failed points, or run twice ((i) the
   subset spy and counts); FM26 the retry run WITH the acceptance
   ((g) the criterion spy: forced slots evaluate none; a converged
   outcome alone cannot see it, because the FM row usually wins
   anyway); FM27 a retried point's `num_iterations` summed over both
   fits ((i)); FM28 route flags in map order, or padded slots routed
   ((k) the `SeedBatch` spy); FM29 the FM branch compacted to the
   routed slots ((k) the 1-of-P against P-of-P bitwise arm under
   numpy if numpy's batched FFT is batch-count sensitive, else
   reviewed-equivalent there and killed by (l) B and routing
   invariance on the device); FM30 the CPU route at a
   chunk-dependent P ((j) the P = 1 spy and chunksize invariance);
   FM31 de-rotation through the subregion resident instead of the
   box ((d) the `dead_band` arm); FM32 degrees and radians confused
   in the row ((d) the row identity, (e)); FM33 props written on
   `"off"` runs, or seed codes 1 and 2 swapped ((a), (f), (i)); FM34
   the D22.11 raise missing ((a)); FM35 `"auto"` without `xmap`
   silently seeding every point ((a)); FM36 the all-zero-twist
   warning missing ((h)); FM37 the FM arithmetic left in complex64
   under `seed_precision="complex64"` ((k) the dtype spy; G (l) the
   parity band); FM38 the translation of the de-rotated target read
   from the REUSED target spectrum instead of the de-rotated crop's
   own ((d) seed error at 8 deg and more, (e)); FM39 the gate run
   from the target's own PC frame instead of the grain reference's
   ((h) on the tilted detector, only if the fixtures carry per-point
   PCs; otherwise reviewed-equivalent, because the twist does not
   depend on the PC); FM40 the forced retry slot refitted although
   its FM estimate failed ((i) a planted angle failure on the
   retried point: no second fit, first result bitwise). Then fixer;
   every surviving mutant killed by a strengthened test and
   re-verified by re-injection.
5. Fixer: every finding dispositioned in a dated table appended to
   this plan (the 11.5 precedent); decision-critical numbers
   re-measured (the angle bands, the acceptance flip count, the
   frame oracle).
6. Gates, each recorded with its output in validation.md: the
   default suite under numpy (`test_hrebsd_fourier_mellin.py` at
   `-n 0`, then `-k hrebsd` over `tests/test_indexing` and
   `tests/test_signals` at `-n 2`) green; the Stage F gated suite
   through the pinned overlay with `KIKUCHIPY_EXPECT_GPU=1` at `-n 0
   --weekly`, 0 skipped; the STAGE E GATED SUITE re-run unchanged and
   green, 0 skipped (the V10 gate commands); the CPU default path
   bitwise unchanged (the pre-Stage-D literal pins, `backend="cpu"`
   == no keyword, `fourier_mellin="off"` == no keyword) and the Stage
   E device default path unchanged (`"off"` through the gated
   suite); coverage 100 % of every touched `_hrebsd` module, default
   and gated runs combined; `uv run pre-commit run --files <explicit
   list>` (ruff, ruff-format; never `--all-files`, never `specs/`);
   the oldest-matrix recipe of ledger 87 (cupy absent there, so the
   gated classes skip at stage (a)); the full suite (`uv run pytest
   tests -n 4`, or `-n 2` while the machine is shared), and again
   after every merge of develop into hrebsd-dic (the Stage E
   recorded gate); signed commits (`git commit -s`) pushed to
   `origin/hrebsd-dic` together (failing tests ride with the
   implementation), NO PR; roadmap Stage F boxes ticked.
7. Performance record (D22.15) on the Si-indent data, machine idle
   and `nvidia-smi` clean before each GPU timing, best of 3 after a
   warm-up: the whole map under `"off"` and `"auto"` at both filter
   settings on both backends (the CPU `"auto"` run reuses the
   ledger 82 recipe on 8 workers; if its 2.34 h is not affordable on
   the shared machine, patch C and the routed band rows 79-142,
   columns 100-150 are the recorded fallback, stated as such),
   far256 and patch C under `"always"`; recorded honestly whatever
   it measures.
8. Tutorial and docs (D22.16): the `hrebsd_dic` docstring; one
   CHANGELOG entry citing D22 and ending with the precedent's
   fork-only closing sentence; one markdown cell in
   `hrebsd_si_indent.ipynb` with the whole-map `"auto"` record and
   the call as a fenced code block -- no executed cell, no stored
   output touched, nbval re-run once on that notebook;
   `installation.rst` untouched.
9. Model assignment (the 2026-10-05 global rule, as for Stage E):
   this spec and plan on Opus 5.5 at effort xhigh with ultracode;
   failing tests, implementation, adversarial review, bug injection
   and fixes on Opus 5.5 at effort medium with ultracode, run as
   Workflow agents with `{model: 'opus', effort: 'medium'}` (the
   Agent tool cannot set effort). Fable is an ESCALATION only: a step
   moves to Fable only on repeated errors AND an inconsistency -- the
   same gate failing again after a fix round, reviewers or critics
   contradicting each other, or the spec and the code still
   disagreeing after a fix; only that step escalates, the reason is
   recorded in the ledger or the commit message, and the work drops
   back to Opus 5.5 afterwards.

### 12.1 Risks

1. **The gate's real-data frame is not pinned** (ledger 212 (iv),
   (v)): on the rim the input twist correlates with the fitted
   in-plane rotation (0.836) but with slope 0.45 over a 0.4 deg
   range, and a 90 deg sample-frame offset would move the routed
   count by up to 5x. Mitigation: the required frame oracle on the
   map's large-twist points (F13) before any default changes; the
   gate fails open on unusable orientations; the retry catches
   non-converged misses; the data gate (F1) stays the alternative.
2. **Indexing noise is the size of the threshold in deformed
   zones** (about 1 to 1.2 deg per component, S1 section 5, against
   1.5 deg). Mitigation: the retry; F2 measures the routed and
   rescued counts on the whole map; F1.
3. **The recipe was chosen on small real sets** (32 rim points;
   ledger 213 shows how fragile a recipe tuned on them can be).
   Mitigation: the radial-mean ZNCC was chosen for its worst case on
   every set, not its rim median; every band is re-measured; F4
   keeps the near-equivalent alternatives.
4. **FM's real-data gain is confounded with the D5 anchor** (ledger
   214 (iii), (iv)): on the rim the FM rows win mostly by moving the
   target off detector-fixed content, not by a large twist.
   Mitigation: D22 claims only the large-twist regime; F8 takes the
   anchor to its own decision; the whole-map record separates
   routed-by-twist from retried points.
5. **Device cost is an inference** (no GPU work at this gate;
   ledger 211): the masked full-P branch costs a whole sub-batch for
   one routed slot. Mitigation: the host-side skip of unrouted
   sub-batches; routed points cluster spatially and in grain order;
   F9 measures compaction.
6. **Stage E is still being built**: the in-progress names D22.10
   relies on (`build_resident`, `build_seed_state`,
   `make_kernel_namespace`, the `run_batch` sub-batch loop,
   `ReferenceResident.mask_index`) are not frozen by D21.
   Mitigation: Stage F starts after Stage E's gates; D22 freezes only
   the D21.5 and D21.9.5 names plus its own; the failing-tests agent
   re-reads Stage E's final module first.
7. **Stage E pins Stage F must not break or must edit**: the slot
   tests (`backend` then `chunksize`), the V9 `extras == {}` asserts,
   the D21.12 literal, the drift tripwire. Mitigation: the slot after
   `seed_from_neighbors`; FM off keeps `extras` empty; the one
   literal edited with a dated comment; the Stage E gated suite is a
   Stage F gate.
8. **The acceptance can be fooled by detector-fixed content**, as
   the 180 deg arbitration was (ledger 209). Mitigation: it only
   chooses between `h_T` and `h_FM`, so a wrong choice of `h_T` is
   exactly today's outcome; F6 counts the refusals on the whole map.
9. **The retry misses false convergence** (V8(b), ledger 206).
   Mitigation: at large budgets the pre-fit gate carries the load;
   F7 measures a residual trigger.
10. **Shared, memory-constrained machine** (commit memory near its
    39.4 GB limit during HROSM jobs; workers lost to `MemoryError`
    at `-n 4`; timings +-30 per cent). Mitigation: `-n 2` at most
    while shared, a red test re-run alone before assuming a bug,
    timings best of 3 with repeats recorded.
11. **The complete initialisation is tempting and wrong on beam-scan
    translations** (ledger 210 (iii)). Mitigation: the partial row is
    frozen; F5 measures a projection-centre-aware complete row before
    any change.

### 12.2 Open questions (each with the conservative default in force and the measurement that resolves it)

F1. **Which gate** (D22.7). In force: the `CrystalMap` twist (zero
    device cost for unrouted slots, identical routing on both
    backends). Alternatives: the data gate (`theta_hat` on every
    slot, about 9 to 12 ms per slot on one CPU thread with the dense
    numpy stencil, ledger 211, and one sub-batch FM angle on the
    device), or both. Resolves: the whole-map `"auto"` record on both
    backends with a data-gate arm (routed, converted and worsened
    counts, wall time); adopt the data gate only if it converts
    points the twist gate and the retry both miss, at a recorded
    cost.
F2. **The threshold** `FM_GATE_DEG` (D22.7). In force: 1.5 deg (0
    misses on the synthetic sweep below 2.5 deg; 336 of the map's
    points outside the crater; ledger 212). Candidates 1.0 and 2.0.
    Resolves: the whole map -- the routed count and the number of
    points the retry still had to rescue at each threshold -- plus
    the synthetic sweep with 0.5 and 1.0 deg orientation noise.
F3. **The default value** (D22.1). In force: `"off"` (every existing
    pin bitwise). Candidate: `"auto"`. Resolves: the whole-map
    `"auto"` record at both filter settings on both backends showing
    no point with a worse outcome than `"off"` (converged to
    not-converged, or a higher residual among both-converged beyond
    the D21.8 bands) and a recorded cost; the flip is a dated D22.1
    amendment and moves the V10(a) default pins.
F4. **Angle recipe constants** (D22.3). In force: the window
    stencil, log1p, 360 angles, rho 0.02-0.40, 1-bin step (V8,
    ledger 213). Alternatives: the Moisan correction (V10,
    equivalent within 0.07 deg), amplitude, a [0.05, 0.20] band (the
    low-count winner of ledger 205). Resolves: the V10(b), (c) bands
    on the implementation, a low-count arm (Poisson 10 and 20) and
    the V5 Si-wafer data at its own noise; change only if V8 misses
    a band an alternative meets.
F5. **Partial against complete initialisation** (D22.4.3). In force:
    partial. Candidate: complete with the projection-centre shift
    removed, `d = R t - gamma` with `gamma` the per-point PC offset
    from the grain reference (`gamma` would enter as an `extras` key)
    and `h = T(gamma) H(R_lat)`. Resolves: ledger 210's two fixtures
    (combined rotations; PC shifts) plus the rim and a large-scan
    synthetic map; adopt only if it never seeds worse than partial
    and saves iterations on the whole map.
F6. **The acceptance** (D22.5). In force: on. Resolves: the whole-map
    count of routed slots where `h_T` was kept and the outcome of
    their fits against a forced-FM refit; drop only if it never
    changes an outcome and its cost (two criterion passes per routed
    slot) shows in the record.
F7. **The retry trigger** (D22.8). In force: not converged only.
    Candidate: also a residual above a per-grain threshold (V8(b);
    ledger 206's 1.95 and 1.86). Resolves: the whole-map census of
    converged points whose residual exceeds the grain's
    median-plus-k-MAD, refitted from the FM row, with the number that
    reach a lower residual.
F8. **The D5 zero anchor on real data** (ledger 207). In force: D5
    unchanged everywhere (the frozen default path) and unchanged
    inside the FM branch. Candidates, for Johan: a band-limited
    cross-power (rho in [0.02, 0.20] cycles/px) inside the FM branch
    only (a recorded divergence confined to routed slots), or in D5
    itself (moves the default path and every pin on it). Resolves:
    the anchor census on the whole map (points whose D5 seed is
    exactly zero while the band-limited one is not; their
    convergence and iterations with each seed) and the V5 Si wafer,
    whose "fits do not track translations" (D4.4) may be the same
    anchor -- a hypothesis, not measured.
F9. **FM sub-batch compaction on the device** (D22.6). In force:
    the FM branch masked at the full P (deterministic by
    construction). Resolves: the GPU `"auto"` whole-map record; a
    separate FM queue padded to P opens only if FM sub-batches cost
    more than about 10 per cent of the run.
F10. **FM with `seed_from_neighbors`** (D22.11). In force: the
     combination raises. Resolves: a request plus the V8 maps run
     with FM rows feeding PASS 1; the composition would carry its
     own D20.3 determinism proof.
F11. **Search window and the 180 deg ambiguity** (D22.3.7). In force:
     30 deg, no two-candidate test. Resolves: nothing inside a grain
     should exceed it (the default grain threshold is 5 deg, D11);
     reopen only for a use case beyond 30 deg, with the S6
     two-candidate test and ledger 209's failure mode as its oracle.
F12. **A bias correction of `theta_hat`** (in-plane-axis rotations,
     ledger 201). In force: none; it changed no rim fit by more than 3
     iterations (ledger 206). Resolves: the data gate's spurious-fire
     count if F1 ever adopts it (the correction matters for a gate on
     `theta_hat`, not for the seed).
F13. **The gate's real-data frame** (D22.7). In force: the as-read
     orientations through `sample_to_detector_matrix`. Resolves: the
     REQUIRED V10(m)(1) oracle; a slope far from +1 stops the stage
     and goes to Johan (it would also question the sample frame of
     the Stage B rotation maps).
F14. **FM under the complex64 seed** (D22.12). In force: complex64
     spectra promoted before any FM arithmetic. Resolves: the gated
     `theta_hat` parity at both seed precisions; nothing changes
     unless complex64 moves `theta_hat` by more than a tenth of a
     bin.

### 12.3 Follow-ups

- The D5 anchor decision (F8), raised to Johan with the census.
- A fused window-and-magnitude kernel for the CPU route (the dense
  numpy stencil is 6.5 to 8.8 ms of the 9 to 12 ms angle path,
  ledger 211), only if the `"always"` record shows it.
- The orientation-delta seed of D5 (the gate already computes
  `R_det`), still deferred.
- FM rows for the CPU cascade (F10) and the residual-triggered retry
  (F7), each its own decision on the whole-map record.

### 12.4 Recorded defaults for Johan's approval

Every default below is one a reasonable person might choose
differently; each stands as written until approved or changed at
this gate. PRE-ACCEPTANCE NOTE: Johan accepted the recommendations in
advance on 2026-10-06 (his overnight waiver, "just continue
automatically and accept recommendations", extended to Stage F); the
main session records the approval at the splice, with the date, in
the 11.4 style.

1. The keyword `fourier_mellin: str = "off"` with values `"off"`,
   `"auto"` and `"always"`, immediately after `seed_from_neighbors`
   on both entries (D22.1; F3). Alternatives: a bool (no `"always"`
   mode for parity and xmap-less runs), or the slot after
   `chunksize`.
2. The angle recipe: the periodic Hann window applied as the exact
   frequency stencil on the reused spectrum, `log1p(|X|)`, 360 polar
   angles over [0, pi) in physical frequency, rho 0.02-0.40
   cycles/px at a 1-bin step, the radial-mean ZNCC, a parabolic
   sub-bin peak (D22.3; F4). Alternatives: the Moisan correction,
   amplitude over [0.05, 0.20], or the whitened per-radius phase
   correlation (best rim median, fragile).
3. Rotation only, no scale, no log-polar axis (D22.2; Johan note
   (vi)).
4. A +-30 deg search window resolves the 180 deg ambiguity; no
   two-candidate test (D22.3.7; F11).
5. The target de-rotated about the grain reference PC through
   `kernels.gather` on a bounding-box resident; the Stage E
   translation step unchanged on the de-rotated crop (D22.4; Johan
   note (iii)).
6. The PARTIAL row `W0 = R(theta) T(t)`; the complete row not
   adopted (D22.4.3; F5).
7. Acceptance by the strictly lower D2.7 criterion at the seed,
   ties to the translation row, never the phase-correlation peak
   (D22.5; F6). Alternative: no acceptance (the second prototype's
   choice, cheaper by two criterion passes per routed point).
8. The pre-fit gate on the `CrystalMap` twist about the detector
   normal at `FM_GATE_DEG = 1.5`, failing open on unusable
   orientations, a `UserWarning` when every twist is exactly 0, and
   a `ValueError` for `"auto"` without an `xmap` at the engine
   (D22.7; F1, F2, F13).
9. One post-fit retry under `"auto"`: non-converged first fits
   seeded by the translation row with a finite `h`, FM row forced,
   the run's own budget, replacement only if the retry converges; no
   retry under `"always"` (D22.8; F7).
10. Two props on FM runs only, `fourier_mellin_seed` and
    `fourier_mellin_angle` (D22.9). Alternative: the seed code only.
11. The CPU route: routed points through the numpy seam at P = 1
    inside the chunk function; unrouted points, and routed points
    whose translation row won, through `initial_guess` bitwise
    (D22.10).
12. `seed_from_neighbors=True` with FM raises `ValueError` (D22.11;
    F10). Alternative: compose on the CPU.
13. The D21.12 literal amended to point at `fourier_mellin='auto'`
    (D22.11).
14. On the device, the host-side skip of sub-batches with no routed
    slot and the FM branch masked at the full P (D22.6; F9).
15. The new module `_hrebsd/_fourier_mellin.py` and the frozen names
    of D22.6 and D22.7; `_run_chunks_gpu(..., seed_extras=None)`;
    the keyword-only `fourier_mellin` on the two VRAM functions
    (D22.6, D22.11, D22.18).
16. The new test file with the `cupy_gpu` fixture duplicated a third
    time (D21.14.1), rather than adding Stage F classes to the
    6700-line `test_hrebsd_gpu.py`.
17. The tutorial: one markdown cell with the whole-map `"auto"`
    record, no executed cell (D22.16).
18. Stage F does not change the D5 seed; its real-data anchor goes
    to Johan as its own decision (F8).

<!--
SUGGESTED ONE-LINE EDIT of plan 11.3 (first bullet), for the main
session: after "(PLANNED after Stage E, Johan decision 1 of
2026-10-06; not commissioned)." add "COMMISSIONED 2026-10-06 and
specified 2026-10-07: section 12, requirements D22, validation V10."
-->
