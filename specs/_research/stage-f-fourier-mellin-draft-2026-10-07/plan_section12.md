<!--
INSERTABLE BLOCKS for specs/2026-09-07-hrebsd-dic/plan.md, drafted
2026-10-07 by the Stage F spec workflow and REVISED the same day
against the two-critic spec review (12.5 is the disposition table;
no repo file was edited). Block 1, section 12 with 12.1 to 12.5, is
appended after the last Stage E subsection at splice time (today
"### 11.5 Spec-review disposition table (2026-10-06)" and its
table; any later 11.x tables Stage E adds come first). Blocks 2 and
3, at the foot of this file, are dated amendments of plan open
question 5 and of the 11.3 Stage F bullet. Ledger numbers 200 to
216 are provisional and move with the main session's renumbering.
Plan open questions are FQ1 to FQ14 (renamed from F1 to F14 at the
review, so that they never collide with the V9 fixtures F1 to F7).
-->

<!-- ===================== BLOCK 1: section 12 ===================== -->

## 12. Stage F -- Fourier-Mellin rotation initial guess (commissioned 2026-10-06; specified 2026-10-07)

Plan open question 5 and the D5 deferral, planned in 11.3 as Stage F
(Johan decision 1 of 2026-10-06) and commissioned the same night when
Johan extended his overnight waiver ("continue with the merlin guess
implementation after the gpu implementation is done"; 11.4 approval
record, item 20 carried to this gate). Requirements D22 (drafted and
revised 2026-10-07) govern; oracles in validation V10; spec-gate
measurements in ledger entries 200 to 216; branch policy section 1
unchanged (hrebsd-dic only, no PR, never merged). Johan's Stage F
notes (11.3 (i) to (viii)) are binding and are carried into D22 with
two recorded deviations, each with its reason: the linear radial
axis (D22.2) and the CPU route's per-point seeding inside the chunk
function (D22.10). Stage F STARTS ONLY AFTER Stage E's gates have
passed and its commits are pushed (roadmap Stage E boxes ticked): D22
plugs into the D21.5 seam and the D21.9.5 names, and the
failing-tests agent re-reads Stage E's final modules before writing a
test.

THE PLAN GATE IS THE APPROVAL GATE, AND IT IS PENDING JOHAN'S
REVIEW. Johan's instruction of 2026-10-07: the Stage F spec is
written and committed, then work STOPS before any Stage F failing
test or code; his 2026-10-06 waiver no longer covers Stage F. No
task below starts until Johan has reviewed 12.4 and approved or
changed every item; his decision is then recorded under 12.4 with
its date, in the 11.4 style.

Deliverables: `fourier_mellin` on `EBSD.hrebsd_dic` and
`run_hrebsd_dic` (D22.1), forwarded by the public method, with the
freeze and defaults pins updated (the D15.4 amendment) and a new
docstring test; the new private module `_hrebsd/_fourier_mellin.py`
with the frozen names of D22.6 and D22.7 -- the per-run look-up
table, `FourierMellinState`, `build_fourier_mellin_state`, the
xp-agnostic angle and peak (D22.3), the de-rotation, translation and
partial row (D22.4), the acceptance (D22.5), the whole branch
`fourier_mellin_rows`, and the gate `twist_about_detector_normal`
with `fourier_mellin_routes` (D22.7); the seam extension in
`_batched.py` (`SeedState.fourier_mellin`, the `extras` route key,
the new `SeedBatch.outputs` field, the host-side skip and the masked
full-P branch; D22.6); the packed row widened to 14 on FM runs only
(D22.9); the CPU route at P = 1 inside the chunk function and the
retry pass in `run_hrebsd_dic` under both FM modes (D22.8, D22.10);
`_run_chunks_gpu(seed_extras=)`, the lazy FM state in the session's
`residents()`, the inactive forced slot and the VRAM keyword with the
`"off"`-first batch choice (D22.11, D22.18); the D22.11 combination
raise and the amended D21.12 literal; the two props (D22.9); the
docstring with the rewritten Limitations paragraph, the CHANGELOG
entry, the `hrebsd_dic.ipynb` bullet and the `hrebsd_si_indent.ipynb`
markdown cell (D22.16); every V10 MTP pin measured and pinned; the
D22.15 performance record.

Files touched outside `_hrebsd/`, kept to the Stage E minimum for
merge hygiene: `signals/ebsd.py` -- inside the fork-only
`hrebsd_dic` method (the keyword forwarding, its docstring and its
prop loop) plus ONE name, `FOURIER_MELLIN_PROP_NAMES`, appended to
the existing `from kikuchipy.indexing._hrebsd._engine import (...)`
block (`ebsd.py:59-63` at f297867e), which tech-stack.md's NLPAR
fan-out lists among the expected append conflicts with develop
(resolved by keeping both sides, HREBSD first); `CHANGELOG.rst` (one
append in the HREBSD block); `doc/tutorials/hrebsd_dic.ipynb` (one
markdown bullet, no stored output, nbval re-run once);
`doc/tutorials/hrebsd_si_indent.ipynb` (one markdown cell); the
fork-only `tests/test_signals/test_ebsd_hrebsd_dic.py`
(`FROZEN_SIGNATURE`), `tests/test_indexing/test_hrebsd_engine.py`
(`test_run_defaults_are_frozen`) and
`tests/test_indexing/test_hrebsd_gpu.py`
(`SEED_FROM_NEIGHBORS_GPU_MESSAGE`), each one edit with a dated
comment; and the new `tests/test_indexing/
test_hrebsd_fourier_mellin.py`. NEVER `_spherical/*`, the root
`conftest.py`, `pyproject.toml` or `doc/user/installation.rst`. Spec
files: this spec commit edits `specs/roadmap.md` (the "Stage F"
section replaces the one-paragraph Stage F note at the foot of the
Stage E section, before the `---` that precedes "Feature path:
NLPAR") and, inside this folder, requirements.md (D22 and its ten
dated amendment blocks: Scope, D5 twice, D15.4, D16, D21.1, D21.5,
D21.9.5, D21.12, D21.18), validation.md (V10, its spec-gate ledger
and the dated V9(b) note) and this plan (section 12, the dated
amendment of open question 5 and the 11.3 note); no constitution
amendment is needed (no new dependency, no new device, the D21
determinism and float scoping cover the FM path).

1. Failing tests first, keyed to V10, in the new
   `tests/test_indexing/test_hrebsd_fourier_mellin.py` (the
   `test_hrebsd_gpu.py` layout), with every helper V10's intro lists
   DUPLICATED from its named source and line range (D21.14.1; about
   1000 lines; each copy kept identical to its source, a drift being
   a test edit with a dated comment), targeting the frozen names and
   call-time seams of D22.6, D22.7 and D21.9.5 (argument lists the
   failing tests fix beyond those are recorded in the module
   docstring, the D21.9.5 rule). Default-suite classes:
   `TestFourierMellinSwitch` (V10(a): the check order, the D22.11
   raise, the forwarding spy, the public-result arm and the NEW
   `test_the_docstring_documents_fourier_mellin`),
   `TestFourierMellinAngle` (V10(b): the look-up-table coordinate
   pin, the G3 non-square arm, the dead-band arm and the
   planted-correlation peak units), `TestFourierMellinEdgeTreatment`
   (V10(c)), `TestFourierMellinSeedRows` (V10(d)),
   `TestFourierMellinCapture` (V10(e); the starred arms default, the
   rest weekly), `TestFourierMellinRampRescue` (V10(f)),
   `TestFourierMellinAcceptance` (V10(g)), `TestFourierMellinGate`
   (V10(h), with the projection-link arm), `TestFourierMellinRetry`
   (V10(i)), `TestFourierMellinCpuRoute` (V10(j)) and
   `TestFourierMellinNumpySession` (V10(k), with the VRAM-keyword and
   import-hygiene arms). Gated classes (V10(l)):
   `TestGatedFourierMellinContract`, `TestGatedFourierMellinParity`,
   `TestGatedFourierMellinDeterminism` (with the routing-invariance
   arm), `TestGatedFourierMellinVram` and
   `TestGatedFourierMellinThroughput` (record only). Exactly three
   edits of existing pins, each with a dated comment:
   `FROZEN_SIGNATURE` gains `("fourier_mellin", KEYWORD_ONLY,
   "off")` after `seed_from_neighbors`; `test_run_defaults_are_frozen`
   gains `fourier_mellin`; `SEED_FROM_NEIGHBORS_GPU_MESSAGE` becomes
   the D21.12-amended literal. No existing docstring test changes,
   and the V9 `extras == {}` asserts are NOT edited (FM off keeps
   them true). Every device band and device-side literal carries a
   `FIXME-pin` placeholder, and so does every literal that needs the
   new code path; only CPU-side premises are measured at this gate:
   the G1, G3 and G4 angle premises through a local re-implementation
   in the test module, the G5 premises (or the record that drops its
   arms), the G6 and G8 premises, the D5 failure premises of (e), (f)
   and (g), and the V8 ramp premise. Expensive runs are cached with
   `functools.lru_cache`; the per-class fit counts of V10's intro and
   the default suite's wall time are recorded at this gate. The
   failing-tests commit is not pushed alone (section 1).
2. Implementation, in dependency order: the D22.1 keyword plumbing
   and check order with the D22.11 raise and the D21.12 literal ->
   `_fourier_mellin.py` under numpy (the per-run look-up table in
   physical frequency, the window stencil on the reused spectrum,
   the magnitude, the radial-mean profile, the ZNCC, the windowed
   peak with lowest-output-index ties; the box resident; the
   de-rotation through `kernels.gather`; the Stage E translation step
   reused on the de-rotated crops; the partial row; the acceptance
   through `kernels.final_criterion` with `initial_shifts` K on the
   grain's own subregion resident; the never-NaN fallbacks;
   everything reached through the module globals) until V10(b) to
   (d) and (g) pass -> the seam extension in `_batched.py` (the
   host-side route check, `h_T` first, the masked full-P branch,
   `SeedBatch.outputs`) -> the gate `twist_about_detector_normal`,
   `fourier_mellin_routes` and the warning -> the CPU route at P = 1
   in the chunk function, paired by the Stage D blockwise index, the
   FM states built once per reference with a routed point, 14-wide
   rows on FM runs -> the retry pass under both FM modes, the props
   and the seed-code derivation from `fm_applied` ->
   `_run_chunks_gpu(seed_extras=)`, the lazy FM state in
   `residents()`, the inactive forced slot, the 14-wide `row_slots`,
   the GPU retry at the first pass's B and the VRAM keyword with the
   `"off"`-first batch choice -> the `ebsd.py` forwarding, import
   name and prop loop, the docstring and the D16 information
   message's memory note -> CHANGELOG, the `hrebsd_dic.ipynb` bullet
   and the `hrebsd_si_indent.ipynb` cell (after the performance
   record). The CPU default path is not edited beyond the branch
   points that the `"off"` value skips.
3. Measurement debt at the implementation gate (validation, ledger
   after the last Stage E entry, recipe and machine ID each): every
   V10 MTP pin, each with its kill separation where it kills a
   mutant (FM3, FM9, FM43); the numpy session against the CPU route
   (bitwise or banded, D22.14); `theta_hat` parity and the
   acceptance flip count on the device at both seed precisions
   (FQ14); the FM terms of the VRAM model, calibrated separately
   against pool high-water marks, and the default B under `"off"`
   and `"auto"`; B invariance and routing invariance with FM on; the
   CPU host bytes per routed reference (D22.10) and the information
   message's note; the CPU and GPU costs of D22.15, the GPU slot
   count in the REAL grain fit order (ledger 216 (vii) used map
   order); and, REQUIRED, the gate's real-data frame oracle under
   the pre-registered rule (V10(m)(1), FQ13) and the whole-map
   `"auto"` record at both filter settings on both backends
   (V10(m)(2); FQ1, FQ2, FQ3, FQ6, FQ7, FQ9, FQ11, with `theta_eff`
   beside the twist), with the D5 anchor census of FQ8 taken from
   the same run. Every gated measurement runs on the pinned overlay
   of D21.15; every test run on the shared machine stays at `-n 2`
   or below.
4. Adversarial review (two reviewers -- fidelity/theory: the angle
   recipe against Ernould and the measured physics of ledger 201,
   the frame algebra of the gate against D7 and D1.3, the
   composition and acceptance against D22.4 and D22.5, the never-NaN
   paths, the determinism argument with the masked full-P branch;
   conventions/integration: the seam extension against D21.5 clause
   (iv) (purely additive: `extras` input only, `outputs` new), the
   packed-row width, the Stage E pins left green, the check order,
   the call-time seams and import direction, coverage 100 % of the
   touched `_hrebsd` modules, docstrings, no stage letters, merge
   hygiene), THEN bug injection ALONE, each mutant re-injected after
   the fixes to verify its kill. Mutation list (FM-prefixed to keep
   Stage E's M1 to M53 distinct; each dies by a named V10 oracle or
   is recorded reviewed-equivalent with its argument; [D] killed in
   the default suite, [G] in the gated suite, [D/G] both):
   FM1 [D] de-rotation by `R(-theta)` ((e) capture, (d) seed error);
   FM2 [D] the row composed as `T(t) R` ((d) the
   projection-centre-shift fixture, the ONLY killer: on pure twists
   t is near 0); FM3 [D] a bin-unit look-up table ((b) the
   look-up-table coordinate pin, PRIMARY; the G3 seed-level arm at 5
   to 20 deg second, the mutant's 0.196 to 2.29 deg against a pin
   of about 0.06 deg, at least 3x apart, ledger 216 (iv)); FM4 [D]
   angles over [0, 2 pi) ((b) the table pin and the sweep); FM5 [D]
   a failed FM estimate returning NaN instead of `h_T` ((d) planted
   failures); FM6 [D] the edge treatment written in place into
   `target_spectra` ((c) the bitwise spectra and unrouted-row
   asserts); FM7 [D] the edge treatment dropped ((c) the G4 lock arm
   at `(None, None)`); FM8 [D] the search window dropped ((b) the
   planted 40/5 deg correlation); FM9 [D] the parabolic offset's
   sign flipped ((b) the off-grid twists, `FM_ANGLE_TOL_DEG` pinned
   below the flipped error with the separation stated); FM10 [D] the
   acceptance inverted ((e) capture, (g) the planted wrong angle and
   G5); FM11 [D] the acceptance dropped ((g) the planted wrong angle
   and the tie arm); FM12 [D] acceptance by the phase-correlation
   peak ((g) G5, where the anchor's peak wins and refuses the FM
   row; the G6 arm too if its measured peaks order opposite to its
   criteria, recorded at the failing-tests gate; if G5 is dropped
   and G6 does not separate, reviewed-equivalent on the default suite
   with the ledger 207 argument, watched by the FQ6 whole-map count);
   FM13 [D] a NaN `h_T` rescued by a finite FM row ((d)); FM14 [D]
   ties keeping `h_FM` ((g) the tie arm); FM15 [D] the route
   ignored, FM applied to every slot ((k) unrouted rows bitwise
   Stage E, (j) unrouted points bitwise `"off"`); FM16 [D] an FM
   state built, `extras` or `outputs` written, or rows widened on
   `"off"` runs ((a) spies, the V9 `extras == {}` asserts, (j) the
   12-wide rows); FM17 [D] the gate conjugated the wrong way (`M^T
   R_s M`) or without the y flip ((h) the sign pins on the tilted
   detector; on an untilted one `M M = I` makes it equivalent);
   FM18 [D] the symmetry reduction dropped ((h) the
   symmetry-equivalent arm); FM19 [D] the total misorientation angle
   gated instead of the twist ((h) the pure out-of-plane arm); FM20
   [D] the signed twist compared ((h) negative twists); FM21 [D] `>`
   instead of `>=` ((h) the planted boundary arm); FM22 [D] the gate
   failing closed on unusable orientations ((h) the unindexed, NaN
   and other-phase arms); FM23 [D] every point gated against the
   map's first reference ((h) the two-grain arm); FM24 [D] a
   non-converged retry replacing the first result ((i) the
   unrelated-pattern point, whose forced fit provably does not
   converge at 200, ledger 216 (iii)); FM25 [D] the retry on
   converged or masked points, or run twice ((i) the subset spy and
   counts); FM26 [D] the retry run WITH the acceptance ((g) the
   criterion count: none for route 2; a converged outcome alone
   cannot see it, because the FM row usually wins anyway); FM27 [D]
   a retried point's `num_iterations` summed over both fits ((i));
   FM28 [D] route flags in map order, or padded slots routed ((k)
   the `SeedBatch` spy); FM29 [D/G] the FM branch compacted to the
   routed slots ((k) the 1-of-P against P-of-P bitwise arm if
   numpy's batched FFT is batch-count sensitive, else
   reviewed-equivalent under numpy; (l) the routing-invariance arm
   on the device); FM30 [D] the CPU route at a chunk-dependent P ((j)
   the P = 1 spy and chunksize invariance); FM31 [D] de-rotation
   through the subregion resident instead of the box ((d) the
   `dead_band` arm); FM32 [D] degrees and radians confused in the row
   ((d) the row identity, (e)); FM33 [D] props written on `"off"`
   runs, or seed codes 1 and 2 swapped ((a), (f), (i)); FM34 [D] the
   D22.11 raise missing ((a)); FM35 [D] `"auto"` without `xmap`
   silently seeding every point ((a)); FM36 [D] the all-zero-twist
   warning missing, or triggered by float equality ((h) the
   all-identity and constant non-identity arms); FM37 [D/G] the FM
   arithmetic left in complex64 under `seed_precision="complex64"`
   ((k) the dtype spy; (l) the parity band); FM38 [D] the
   translation of the de-rotated target read from the REUSED target
   spectrum instead of the de-rotated crop's own ((d) seed error at 8
   deg and more, (e)); FM39 [D] the gate run from the target's own
   PC frame instead of the grain reference's ((h) on the tilted
   detector if the fixtures carry per-point PCs; otherwise
   reviewed-equivalent, because the twist does not depend on the
   PC); FM40 [D] a forced slot fitted although not applied (its FM
   estimate failed or its `h_T` is non-finite) ((i) a planted angle
   failure on the retried point: no second fit, first result
   bitwise; (k) the device-runner kernel spy: the slot inactive in
   the lockstep, its packed row `converged` 0 and `fm_applied` 0);
   FM41 [D] `EBSD.hrebsd_dic` not forwarding `fourier_mellin` ((a)
   the forwarding spy); FM42 [D] the `ebsd.py` prop loop not
   extended ((a) the public-result arm); FM43 [D] de-rotation about
   the detector centre or the target's own PC instead of the grain
   reference PC ((d) the seed-row band, error `2 sin(theta/2) |c|`);
   FM44 [D] every CPU point routed through the numpy seam ((j) the
   `_batched.seed_spectra` call count: the numpy seam equals skimage
   on the fixtures, so only the count sees it); FM45 [D] the outputs
   written into `SeedBatch.extras`, or `extras` mutated ((k) the
   `extras`-unchanged spy); FM46 [D/G] `_default_batch_size` picking
   B from the FM model where the FM terms fit ((k) the fake-VRAM
   arm; (l) the default B equal under `"off"` and `"auto"`); FM47
   [D] the retry skipped under `"always"` ((i) the `"always"` arm);
   FM48 [D] the retry at a newly chosen B ((k) the `chunksize` spy);
   FM49 [D] NaN-`h` first-pass points excluded from the retry ((i)
   the subset spy: the constant pattern is in the subset); FM50 [D]
   the FM state built eagerly for every grain, or rebuilt per
   sub-batch ((k) the lazy-build spy: once per grain with a routed
   slot, never without); FM51 [D] a NaN `h_T` criterion treated as a
   loss, so the FM row is kept ((g) the planted NaN-criterion arm);
   FM52 [D] the masked branch's angles reported on route-0 or padded
   slots ((d) the outputs rule). Then fixer; every surviving mutant
   killed by a strengthened test and re-verified by re-injection.
5. Fixer: every finding dispositioned in a dated table appended to
   this plan (the 11.5 and 12.5 precedent); decision-critical
   numbers re-measured (the angle bands, the acceptance flip count,
   the frame oracle).
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
   and gated runs combined; nbval once on `hrebsd_dic.ipynb` and on
   `hrebsd_si_indent.ipynb`; `uv run pre-commit run --files
   <explicit list>` (ruff, ruff-format; never `--all-files`, never
   `specs/`); the oldest-matrix recipe of ledger 87 (cupy absent
   there, so the gated classes skip at stage (a)); the full suite
   (`uv run pytest tests -n 4`, or `-n 2` while the machine is
   shared), and again after every merge of develop into hrebsd-dic
   (the Stage E recorded gate); signed commits (`git commit -s`)
   pushed to `origin/hrebsd-dic` together (failing tests ride with
   the implementation), NO PR; roadmap Stage F boxes ticked.
7. Performance record (D22.15) on the Si-indent data, machine idle
   and `nvidia-smi` clean before each GPU timing, best of 3 after a
   warm-up: the whole map under `"off"` and `"auto"` at both filter
   settings on both backends (the CPU `"auto"` run reuses the
   ledger 82 recipe on 8 workers; if its 2.34 h is not affordable on
   the shared machine, patch C and the routed band rows 79-142,
   columns 100-150 are the recorded fallback, stated as such),
   far256 and patch C under `"always"`; recorded honestly whatever
   it measures.
8. Tutorial and docs (D22.16): the `hrebsd_dic` docstring (the
   `fourier_mellin` entry, Notes, Raises, the warning and the
   rewritten Limitations paragraph); one CHANGELOG entry citing D22
   and ending with the precedent's fork-only closing sentence; the
   "A finite rotation capture range" bullet of `hrebsd_dic.ipynb`
   reworded to name the opt-in keyword (one markdown edit); one
   markdown cell in `hrebsd_si_indent.ipynb` with the whole-map
   `"auto"` record and the call as a fenced code block -- no
   executed cell and no stored output touched in either notebook,
   nbval re-run once on each; `installation.rst` untouched.
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
   (v), 216 (viii)): on 32 rim points the input twist correlates
   with the fitted in-plane rotation at 0.836 over only 0.05 to 0.45
   deg, and the least-squares slopes, 0.452 with the noisy input
   twist as regressor and 1.545 the other way round, bracket 1 by
   regression dilution and decide nothing; a 90 deg sample-frame
   offset would move the routed count by up to 5x. Mitigation: the
   REQUIRED frame oracle under the pre-registered rule of D22.7
   (FQ13) before any default changes; the gate fails open on
   unusable orientations; the retry catches non-converged misses;
   the data gate (FQ1) stays the alternative.
2. **Indexing noise is the size of the threshold in deformed
   zones** (about 1 to 1.2 deg per component, S1 section 5, against
   1.5 deg). Mitigation: the retry; FQ2 measures the routed and
   rescued counts on the whole map; FQ1.
3. **The recipe was chosen on small real sets** (32 rim points;
   ledger 213 shows how fragile a recipe tuned on them can be), and
   the frozen partial row with the acceptance has been fitted on 24
   real rim points only (ledger 216 (v)). Mitigation: the
   radial-mean ZNCC was chosen for its worst case on every set, not
   its rim median; every band is re-measured; FQ4 keeps the
   near-equivalent alternatives; the whole-map record (V10(m)(2)) is
   the real-data test of the frozen configuration.
4. **FM's real-data gain is confounded with the D5 anchor** (ledger
   214 (iii), (iv), 216 (v)): on the rim the FM rows win mostly by
   moving the target off detector-fixed content, not by a large
   twist. Mitigation: D22 claims only the large-twist regime; FQ8
   takes the anchor to its own decision; the whole-map record
   separates routed-by-twist from retried points.
5. **Device cost is an inference** (no GPU work at this gate;
   ledgers 211, 216 (vii)): the masked full-P branch runs on every
   slot of a sub-batch holding a routed point -- 1760 slots for the
   336 points routed at 1.5 deg on the Si map in map order (5.2x),
   about 4 s or 2.5 per cent of the 166.5 s mixed run, and about +80
   per cent under `"always"`. Mitigation: the host-side skip of
   unrouted sub-batches; grain ordering clusters routed points; FQ9
   measures compaction.
6. **Stage E is still being built**: the in-progress names D22
   relies on (`build_resident`, `build_seed_state`,
   `make_kernel_namespace`, `initial_shifts`, the session's
   `residents()`, the `run_batch` sub-batch loop,
   `ReferenceResident.mask_index`) are not frozen by D21.
   Mitigation: Stage F starts after Stage E's gates; D22 freezes only
   the D21.5 and D21.9.5 names plus its own; the failing-tests agent
   re-reads Stage E's final modules first.
7. **Stage E pins Stage F must not break or must edit**: the slot
   tests (`backend` then `chunksize`), the V9 `extras == {}` asserts,
   the 12-wide `row_slots`, the D21.12 literal, the drift tripwire.
   Mitigation: the slot after `seed_from_neighbors`; FM off keeps
   `extras` and `outputs` empty and the rows 12 wide; the one literal
   edited with a dated comment; the Stage E gated suite is a Stage F
   gate.
8. **The acceptance can be fooled by detector-fixed content**, as
   the 180 deg arbitration was (ledger 209). Mitigation: it only
   chooses between `h_T` and `h_FM`, so a wrong choice of `h_T` is
   exactly today's outcome; FQ6 counts the refusals on the whole map.
9. **The retry misses false convergence** (V8(b), ledger 206).
   Mitigation: at large budgets the pre-fit gate carries the load;
   FQ7 measures a residual trigger.
10. **Shared, memory-constrained machine** (commit memory near its
    39.4 GB limit during HROSM jobs; workers lost to `MemoryError`
    at `-n 4`; timings +-30 per cent). Mitigation: `-n 2` at most
    while shared, a red test re-run alone before assuming a bug,
    timings best of 3 with repeats recorded.
11. **The complete initialisation is tempting and wrong on beam-scan
    translations** (ledger 210 (iii)). Mitigation: the partial row is
    the recorded default; FQ5 measures a projection-centre-aware
    complete row before any change.
12. **The CPU route's host memory** (D22.10): about 20 to 25 MB per
    routed reference, held for the whole run beside the 17.96 MB of
    its `ReferenceState`, every grain under `"always"`, on a machine
    whose commit memory is the binding constraint. Mitigation: built
    only for references with a routed point; counted in the D16
    information message's memory note (D22.18); measured in the
    `"always"` record.

### 12.2 Open questions (each with the conservative default in force and the measurement that resolves it)

FQ1. **Which gate** (D22.7). In force: the `CrystalMap` twist (zero
     device cost for unrouted slots, identical routing on both
     backends). Alternatives: the data gate (`theta_hat` on every
     slot, about 9 to 12 ms per slot on one CPU thread with the
     dense numpy stencil, ledger 211, and one sub-batch FM angle on
     the device); the effective rotation of the crop, `theta_eff ~=
     twist + (w1 xbar + w2 ybar) / (2 DD)` with the bounding-box
     centroid (free, since `R_det` carries w1 and w2; up to about 0.5
     deg off the twist on the rim, ledger 201); or both. Resolves: the
     whole-map `"auto"` record on both backends with `theta_eff`
     recorded beside the twist for every point and a data-gate arm
     (routed, converted and worsened counts, wall time); adopt an
     alternative only if it converts points the twist gate and the
     retry both miss, at a recorded cost.
FQ2. **The threshold** `FM_GATE_DEG` (D22.7). In force: 1.5 deg on
     the twist (0 misses on the synthetic sweep below 2.5 deg; 336
     of the map's points outside the crater; ledger 212).
     Candidates 1.0 and 2.0, on the twist or on `theta_eff`.
     Resolves: the whole map -- the routed count, the sub-batches and
     slots the device branch ran on (ledger 216 (vii): 3552, 1760
     and 544 slots at 1.0, 1.5 and 2.0 deg in map order) and the
     number of points the retry still had to rescue at each
     threshold -- plus the synthetic sweep with 0.5 and 1.0 deg
     orientation noise.
FQ3. **The default value** (D22.1). In force: `"off"` (every existing
     pin bitwise). Candidate: `"auto"`. Resolves: the whole-map
     `"auto"` record at both filter settings on both backends showing
     no point with a worse outcome than `"off"` (converged to
     not-converged, or a higher residual among both-converged beyond
     the D21.8 bands) and a recorded cost; where FM changes the
     default device B (D22.18), the comparison runs at a fixed
     explicit `chunksize`. The flip is a dated D22.1 amendment and
     moves the V10(a) default pins.
FQ4. **Angle recipe constants** (D22.3). In force: the window
     stencil, log1p, 360 angles over [0, pi), `rho_j = 0.02 + j /
     min(sr, sc)` up to 0.40, the radial-mean ZNCC, the parabolic
     peak (V8, ledgers 213, 216), and no masking of the frequency
     axes. Alternatives: the Moisan correction (V10, equivalent
     within 0.07 deg), amplitude, a [0.05, 0.20] band (the low-count
     winner of ledger 205), and masking the look-up samples within
     one bin of the axes (a constant dead-band cross pulls
     `theta_hat` up to 0.12 deg towards 0, ledger 216 (vi)).
     Resolves: the V10(b), (c) bands on the implementation, the
     dead-band arm, a low-count arm (Poisson 10 and 20) and the V5
     Si-wafer data at its own noise; change only if V8 misses a band
     an alternative meets.
FQ5. **Partial against complete initialisation** (D22.4.3). In
     force: partial. Candidate: complete with the projection-centre
     shift removed, `d = R t - gamma` with `gamma` the per-point PC
     offset from the grain reference (`gamma` would enter as an
     `extras` key) and `h = T(gamma) H(R_lat)`. Both orderings of the
     complete form are valid parameterisations (Ernould 2021 eqs.
     E.2-E.3 read the lattice rotations from `d = t` with the
     rotation centre at the PC, the prototype from `d = R t`; they
     differ at second order, ledger 210 (iii)). Resolves: ledger
     210's two fixtures (combined rotations; PC shifts) plus the rim
     and a large-scan synthetic map; adopt only if it never seeds
     worse than partial and saves iterations on the whole map.
FQ6. **The acceptance** (D22.5). In force: on. Resolves: the
     whole-map count of routed slots where `h_T` was kept and the
     outcome of their fits against a forced-FM refit; drop only if it
     never changes an outcome and its cost (two criterion passes per
     routed slot, 20 to 30 ms each on one CPU thread) shows in the
     record.
FQ7. **The retry trigger** (D22.8). In force: not converged only,
     on first fits seeded by the translation row, NaN-`h` first
     passes included (a recorded choice: the D2.6 contract also
     arises from a fit that diverged from a spurious seed, and an
     unusable crop makes the forced FM estimate fail, so it is never
     refitted). Candidate: also a residual above a per-grain
     threshold (V8(b); ledger 206's 1.95 and 1.86). Resolves: the
     whole-map census of converged points whose residual exceeds the
     grain's median-plus-k-MAD, refitted from the FM row, with the
     number that reach a lower residual, and the number of NaN-`h`
     points the retry converted.
FQ8. **The D5 zero anchor on real data** (ledger 207). In force: D5
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
FQ9. **FM sub-batch compaction on the device** (D22.6). In force:
     the FM branch masked at the full P (deterministic by
     construction), costing every slot of a sub-batch that holds a
     routed point (5.2x the routed count at 1.5 deg on the Si map in
     map order, ledger 216 (vii)). Resolves: the GPU `"auto"`
     whole-map record in the real grain fit order; a separate FM
     queue padded to P opens only if FM sub-batches cost more than
     about 10 per cent of the run.
FQ10. **FM with `seed_from_neighbors`** (D22.11). In force: the
      combination raises. Resolves: a request plus the V8 maps run
      with FM rows feeding PASS 1; the composition would carry its
      own D20.3 determinism proof.
FQ11. **Search window and the 180 deg ambiguity** (D22.3.7). In
      force: 30 deg, no two-candidate test. Resolves: the whole-map
      census of V10(m)(2) -- |`theta_hat`| and |twist| over the
      routed points; reopen if any routed |twist| exceeds about 25
      deg or any `theta_hat` sits at the window's edge bin. The D11
      default grain threshold of 5 deg does not bound it alone:
      user-supplied `grain_labels` and tuple references can pair
      orientations further apart. A reopened window carries the S6
      two-candidate test with ledger 209's failure mode as its
      oracle.
FQ12. **A bias correction of `theta_hat`** (in-plane-axis rotations,
      ledger 201). In force: none; it changed no rim fit by more than
      3 iterations (ledger 206). Resolves: the data gate's
      spurious-fire count if FQ1 ever adopts it (the correction
      matters for a gate on `theta_hat`, not for the seed).
FQ13. **The gate's real-data frame** (D22.7). In force: the as-read
      orientations through `sample_to_detector_matrix`. Resolves: the
      REQUIRED V10(m)(1) oracle under the rule pre-registered in
      D22.7 -- the converged homographies of an `"always"` run
      (independent of the frame under test); the fitted twist from
      the polar decomposition of each converged Fe; points selected
      on |fitted twist| >= 1 deg (never on the xmap twist),
      converged, outside the crater; the as-read frame and four
      alternatives (+90, 180 and -90 deg about sample z, and `M`
      without the y flip) compared by correlation; the slope of the
      xmap twist on the fitted twist and a Deming slope with the
      far-field Hough variance. PASS: the as-read frame correlates
      best and its slope lies in [0.8, 1.25]; otherwise the stage
      stops and goes to Johan (it would also question the sample
      frame of the Stage B rotation maps). The 32 rim points
      measured so far bracket 1 and decide nothing (ledger 216
      (viii)).
FQ14. **FM under the complex64 seed** (D22.12). In force: complex64
      spectra promoted before any FM arithmetic. Resolves: the gated
      `theta_hat` parity at both seed precisions; nothing changes
      unless complex64 moves `theta_hat` by more than a tenth of a
      bin.

### 12.3 Follow-ups

- The D5 anchor decision (FQ8), raised to Johan with the census.
- A fused window-and-magnitude kernel for the CPU route (the dense
  numpy stencil is 6.5 to 8.8 ms of the 9 to 12 ms angle path,
  ledger 211), only if the `"always"` record shows it.
- Freeing a reference's CPU-route FM states after its grain's
  chunks, only if the `"always"` record shows the host bytes of
  risk 12 binding.
- The orientation-delta seed of D5 (the gate already computes
  `R_det`), still deferred.
- FM rows for the CPU cascade (FQ10) and the residual-triggered
  retry (FQ7), each its own decision on the whole-map record.

### 12.4 Recorded defaults for Johan's approval (PENDING his review)

Every default below is one a reasonable person might choose
differently; each stands as written until Johan approves or changes
it at this gate. STATUS 2026-10-07: PENDING JOHAN'S REVIEW. His
instruction of 2026-10-07 stops the work after this spec commit and
before any Stage F failing test or code; the 2026-10-06 waiver
("just continue automatically and accept recommendations") no longer
covers Stage F, so NONE of these items is approved. His decision
will be recorded here with its date, in the 11.4 style, before task
1 starts. Items 19 to 21 were added at the spec review (12.5).

1. The keyword `fourier_mellin: str = "off"` with values `"off"`,
   `"auto"` and `"always"`, immediately after `seed_from_neighbors`
   on both entries, forwarded by name by the public method (D22.1;
   FQ3). Alternatives: a bool (no `"always"` mode for parity and
   xmap-less runs), or the slot after `chunksize`.
2. The angle recipe: the periodic Hann window applied as the exact
   frequency stencil on the reused spectrum, `log1p(|X|)`, 360 polar
   angles over [0, pi) in physical frequency, `rho_j = 0.02 + j /
   min(sr, sc)` up to 0.40 cycles/px, the radial-mean ZNCC, the
   argmax over lags in [-n/2, n/2) with ties to the lowest FFT output
   index, a parabolic sub-bin peak from the raw neighbours, one
   look-up table per run (D22.3; FQ4). Alternatives: the Moisan
   correction, amplitude over [0.05, 0.20], or the whitened
   per-radius phase correlation (best rim median, fragile).
3. Rotation only, no scale, no log-polar axis (D22.2; Johan note
   (vi), with the radial axis as the recorded deviation).
4. A +-30 deg search window resolves the 180 deg ambiguity; no
   two-candidate test (D22.3.7; FQ11).
5. The target de-rotated about the grain reference PC through
   `kernels.gather` on a bounding-box resident; the Stage E
   translation step unchanged on the de-rotated crop (D22.4; Johan
   note (iii)).
6. The PARTIAL row `W0 = R(theta) T(t)`; the complete row not
   adopted (D22.4.3; FQ5).
7. Acceptance by the strictly lower D2.7 criterion at the seed,
   evaluated on the grain's own subregion resident; ties, a
   non-finite FM criterion and a non-finite `h_T` criterion all
   keep the translation row; never the phase-correlation peak (D22.5;
   FQ6). Alternative: no acceptance (the second prototype's choice,
   cheaper by two criterion passes per routed point).
8. The pre-fit gate on the `CrystalMap` twist about the detector
   normal at `FM_GATE_DEG = 1.5`, failing open on unusable
   orientations, the frozen `UserWarning` when no twist is NaN and
   every one is below 1e-9 deg, and a `ValueError` for `"auto"`
   without an `xmap` at the engine (D22.7; FQ1, FQ2, FQ13).
9. One post-fit retry under `"auto"` AND `"always"`: every fitted,
   unmasked, non-converged point whose first fit was seeded by the
   translation row, NaN-`h` first passes included; the FM row forced
   (no acceptance); the run's own budget; on the device at the first
   pass's final B, passed as an explicit `chunksize`, an
   out-of-memory in the retry following D21.10.4 for the retry
   subset only; a forced slot whose FM estimate fails is never
   fitted; replacement only if the retry converges (D22.8; FQ7).
   Alternative: no retry under `"always"` (it then converts fewer
   points than `"auto"`).
10. Two props on FM runs only, `fourier_mellin_seed` and
    `fourier_mellin_angle`, carried out of both runners by a packed
    row widened from 12 to 14 on FM runs only (D22.9). Alternative:
    the seed code only.
11. The CPU route: routed points through the numpy seam at P = 1
    inside the chunk function; unrouted points, and routed points
    whose translation row won, through `initial_guess` bitwise
    (D22.10). This deviates from note (viii)'s batched seeding,
    because P = 1 makes each result independent of chunking by
    construction and needs no host pre-pass that reads the routed
    lazy patterns twice. Alternative: the batched host pre-pass
    through `_run_chunks(h0=...)`, as note (viii) describes.
12. `seed_from_neighbors=True` with FM raises `ValueError` (D22.11;
    FQ10). Alternative: compose on the CPU.
13. The D21.12 literal amended to name `seed_from_neighbors=False`
    with `fourier_mellin='auto'` as the device remedy (D22.11).
14. On the device, the host-side skip of sub-batches with no routed
    slot, the FM branch masked at the full P, and the FM state built
    lazily per grain in the session's `residents()` (D22.6, D22.11;
    FQ9).
15. The new module `_hrebsd/_fourier_mellin.py` and the frozen names
    and call-time seams of D22.6 and D22.7; the new
    `SeedBatch.outputs` field (outputs never in `extras`);
    `_run_chunks_gpu(..., seed_extras=None)`; the keyword-only
    `fourier_mellin` on `_vram_model_terms`, `_vram_model_bytes`
    and `_default_batch_size` (D22.6, D22.11, D22.18).
16. The new test file with the `cupy_gpu` fixture and the V10 helper
    set duplicated (about 1000 lines, D21.14.1), rather than adding
    Stage F classes to the 6700-line `test_hrebsd_gpu.py`.
17. The tutorials: the `hrebsd_dic.ipynb` capture-range bullet
    reworded and one markdown cell with the whole-map `"auto"`
    record in `hrebsd_si_indent.ipynb`, no executed cell (D22.16).
18. Stage F does not change the D5 seed; its real-data anchor goes
    to Johan as its own decision (FQ8).
19. The device batch choice: B from the `"off"` model, halved only
    if the FM terms do not fit the remaining headroom, so FM never
    changes B on a card with headroom (D22.18; FQ3).
20. The frame decision rule pre-registered in D22.7: the five
    candidate frames, selection on the fitted twist, the PASS band
    [0.8, 1.25] on the slope, and a stop for Johan otherwise
    (FQ13).
21. No masking of the frequency axes against a dead-band cross; the
    measured bias (at most 0.12 deg towards 0, ledger 216 (vi)) is a
    documented limitation (D22.3; FQ4).

### 12.5 Spec-review disposition table (2026-10-07)

45 findings from the two adversarial critics of the Stage F draft
(critic A, method and numerics: F01 to F21; critic B, conventions,
integration and testability: C-F-MAJ-1 to 8 and C-F-MIN-1 to 16; 14
major, 31 minor, no blocker). The first reviser was stopped mid-way
when the work was parked (2026-10-07, about 02:50); this table was
completed on the second pass the same day. Every finding was
re-checked against the current draft text, then verified against
the code (the f297867e tests and `ebsd.py` through `git show`, the
working-tree `_batched.py` and `_gpu.py`), the frozen D21.5 and
D21.9.5 text, `prototype_fm_lib.py.txt`, `fm_method_literature.md`
and the prototype logs (`e2e_partial.log`, `deadband.log`, whose
numbers ledger 216 carries and which were re-read; the real-rim
iteration totals were recomputed from the log: 2002 for the
translation seed, 1660 with the accepted partial rows). Status:
"already-applied" -- the fix was in the D22 and V10 drafts before
this pass, verified, at most propagated here into plan 12 or the
roadmap; "accepted" -- applied or completed in this pass (a named
location still carried the defect); "rejected" -- with its reason.
Counts: 20 accepted, 24 already-applied, 1 rejected. None
contradicts Johan's decisions of 2026-10-06 or his instruction of
2026-10-07; C-F-MAJ-1 and F06 EXTEND the seam additively under
D21.5 clause (iv) (a new field, a new input key), and the packed row
widens only on FM runs.

| finding | issue | disposition |
|---|---|---|
| F01 (major) | real-data acceptance and gain measured with the complete row only | already-applied: `e2e_partial.py` fitted the frozen partial row with the acceptance on the real rim at both filter settings and on the PC-shift arm (ledger 216 (ii), (v)); D22.5 restated with those numbers (13 of 16 kept, 2002 -> 1660 iterations; 5 of 8 converted in 87-161); ledger 214 (iii), (iv) tagged complete-row |
| F02 (major) | capture numbers from rejected recipes; "at least 30 deg" impossible | accepted: the D22 lead and ledgers 202 (ii), 203 (ii), 215 were already tagged and V8 re-measured to the window edge (216 (i): 8 to 30 and -20 deg at `border=0.15`, 2 to 3 iterations, 29.931 deg at 30); D22.16 says "up to its 30 degree search window"; V10(e) weekly to 25 deg plus a recorded 30 deg arm; the roadmap paragraph rewritten in this pass |
| F03 (major) | frame-check rule ill-posed (regression dilution) | accepted: D22.7 and V10(m)(1) already carried the pre-registered rule (`"always"` run, selection on the fitted twist, five candidate frames, reverse and Deming slopes, PASS in [0.8, 1.25]) and ledger 216 (viii) the 1.545 slope; FQ13, risk 1 and 12.4 item 20 rewritten in this pass (the old "slope far from +1 stops the stage") |
| F04 (major) | V10(f) asserts the 8 deg ramp point keeps its `"off"` outcome | already-applied: measured on a seam-level proxy (216 (iii): the FM row converges in 3 iterations to 0.053 px); V10(f) now expects convergence, code 1, `FM_RAMP_FAR_TOL_PX`, and specifies the 9 deg rescue point under both modes |
| F05 (major) | the FM3 pin contradicts the 2x rule | accepted: V10(b) already carried the measured G3 arm (216 (iv): 0.027 deg against the mutant's 0.196-2.29 deg from 5 deg) and the look-up-table coordinate pin; FM3 in item 4 rewritten in this pass (the pin PRIMARY, G3 second, separation stated) |
| F06 (major) | output keys in `SeedBatch.extras` change a frozen field | already-applied: the new `SeedBatch.outputs` field, `extras` input only and on the host, Block 8 purely additive; strengthened in this pass by the V10(k) `extras`-unchanged arm and FM45 |
| F07 (minor) | GPU cost per routed pattern, not per slot | accepted: D22.15 was restated per slot with the `subbatch_amp.py` counts and the `"always"` expectation; risk 5, FQ2 and FQ9 restated in this pass |
| F08 (minor) | CPU route cost omits required work | accepted: D22.15 lists every component; the totals corrected in this pass to 165-205 ms at 480x480 and 240-275 ms at 512x622, about 7 to 11 iterations (ledgers 90, 211 summed) |
| F09 (minor) | no retry under `"always"`, stated reason false | accepted (the first option): D22.1 and D22.8 already retried under both modes; 12.4 item 9 and the roadmap, which still said "no retry under always", corrected in this pass; FM47 |
| F10 (minor) | a forced slot whose FM estimate failed is refitted on the device | already-applied: the runner marks it inactive between the seam call and the lockstep (D22.8, D22.11 (2)); V10(k) kernel spy; FM40's device killer added in this pass |
| F11 (minor) | dead-band cross biases the angle | already-applied: measured (216 (vi): at most 0.12 deg towards 0); a recorded limitation in D22.3 with the V10(b) dead-band arm; FQ4 and 12.4 item 21 carry masking in this pass |
| F12 (minor) | G5 fixed-pattern noise may pull the angle to 0 | already-applied: G5 has the second run-time premise, white per-pixel noise and the drop rule |
| F13 (minor) | G8's unrelated point may give FM24 no killer | already-applied: measured (216 (iii): first pass not converged with a finite `h`, residual 1.89; forced FM fit not converged, 1.66); ledger 204 (iv) corrected; premises asserted at run time |
| F14 (minor) | the D21.12 literal's remedy itself raises | already-applied: Block 10 carries the proposed literal |
| F15 (minor) | tie order, lag range, edge neighbours, NaN comparison | already-applied: D22.3.5-6 (lags in [-n/2, n/2), as `peak_angle` in the prototype; ties to the lowest FFT output index; raw neighbours) and D22.5 (a non-finite `h_T` criterion keeps `h_T`, with the reason; "within `NUMPY_PARITY_RESIDUAL_RTOL`"); the V10(g) planted NaN-criterion arm and FM51 added in this pass |
| F16 (minor) | gating on the twist ignores the in-plane-axis term | accepted: D22.7 already gave the reason for the twist and V10(m)(2) records `theta_eff`; FQ1 and FQ2 name it as a candidate in this pass |
| F17 (minor) | no test links the gate's convention to projected patterns | already-applied: the V10(h) projection-link arm (1e-10 relative; `theta_hat` against the twist) |
| F18 (minor) | the VRAM keyword can change B and the bits of unrouted points | already-applied: D22.18 picks B from the `"off"` model and halves only on misfit; V10(k), (l); FQ3, FM46 and 12.4 item 19 propagated in this pass |
| F19 (minor) | the look-up table per reference; FM host memory unaccounted | already-applied: one table per run cached by crop shape (D22.3.3); per-reference bytes in D22.10 and the information message (D22.18); risk 12 added in this pass |
| F20 (minor) | Ernould's complete form uses `d = t`, not a mutant | already-applied: D22.4.3 and ledger 210 (iii) state both orderings as valid (verified against the literature report, section 2.9) and reserve "mutant" for `T(t) R`; FQ5 propagated in this pass |
| F21 (minor) | the message constant cited at line 335, said to be 337 | rejected: `SEED_FROM_NEIGHBORS_GPU_MESSAGE = (` is line 335 of `test_hrebsd_gpu.py` at f297867e, 717f0c15 and the working tree (checked with `git show`, 2026-10-07); critic B confirms 335 |
| C-F-MAJ-1 (major) | no output channel out of the runners | accepted: D22.9 widens the packed row to 14 on FM runs only (`fm_angle` 12, `fm_applied` 13) on both runners, with `row_slots` extended and the seed code derived from `fm_applied`; FM-off stays 12 wide; V10(j), (k) width and sentinel arms; plan item 2, 12.4 item 10 and the deliverables completed in this pass; FM40 |
| C-F-MAJ-2 (major) | the acceptance needs the subregion resident the seam never sees | already-applied: `FourierMellinState.resident` references the grain's existing `ReferenceResident` (no second upload; evicted with its `SeedState`); K from `_batched.initial_shifts` (present in the working tree); V10(k) identity spy |
| C-F-MAJ-3 (major) | GPU wiring unspecified | already-applied: D22.11 (2) (FM on iff the route key has a nonzero entry; the lazy FM state in `residents()`), D22.18 and Block 9 (`fourier_mellin` on `_vram_model_terms` too; True iff any route is nonzero); the V10(k) VRAM arm; FM50 added in this pass |
| C-F-MAJ-4 (major) | V10(f) false for the 8 deg point; G8 twist range breaks its premise | already-applied: as F04; G8 pinned at 4.0 deg and a budget of 200 (216 (iii): 3.0 deg converges in 14 from the translation seed, 4.0 deg does not, 42.9 px); the reference point's codes given |
| C-F-MAJ-5 (major) | headline evidence from other recipes | accepted: as F02; "1 to 9" in ledger 202 (ii); the D22 lead already carried the band-pass qualifier and the `(None, None)` 6 deg figure with its meaning (iteration cost and the D5 anchor on the Si map, not capture); the roadmap paragraph rewritten in this pass |
| C-F-MAJ-6 (major) | no frozen patch points | already-applied: D22.6 freezes `fourier_mellin_peak`, `_derotate`, `_translate`, `_criteria`, `_rows`, `_routes` and `twist_about_detector_normal` with their argument lists, and the call-time module-global rule |
| C-F-MAJ-7 (major) | no test of the public method | accepted: V10(a) already had the forwarding spy and the public-result arm; FM41 and FM42 added to item 4 in this pass |
| C-F-MAJ-8 (major) | missing dated amendments | accepted: requirements Blocks 2 (Scope), 6 (D16), 7 (D21.1), validation Block 3 (V9(b)) and D22.16 (the `hrebsd_dic.ipynb` bullet, lines 1877-1881, and the Limitations paragraph near `ebsd.py:3343`, both verified at f297867e) were in the drafts; Blocks 2 and 3 of this file (open question 5, 11.3) and the files-touched list added in this pass |
| C-F-MIN-1 (minor) | when the output keys exist | already-applied: one rule in D22.6 (present iff the route key has a nonzero entry; absent read as NaN and False); V10(d), (k) arms |
| C-F-MIN-2 (minor) | criterion spy contradicts the masked full-P branch | already-applied: V10(g) counts `fourier_mellin_criteria` calls, exact at P = 1 on the CPU route, sub-batch level at full P in the numpy session |
| C-F-MIN-3 (minor) | "equals fit_pattern residual" is not bitwise | already-applied: D22.5 "within `NUMPY_PARITY_RESIDUAL_RTOL`"; tests read the decision from `applied` |
| C-F-MIN-4 (minor) | the retry may run at another B | accepted: D22.8 and D22.13 already fixed the first pass's final B; 12.4 item 9 (named by the critic) completed in this pass; V10(k) spy; FM48 |
| C-F-MIN-5 (minor) | the NaN-`h` exclusion overclaimed | accepted (the first option): D22.8 retries NaN-`h` points (an unusable crop's forced estimate fails, so it is never refitted); listed under FQ7 and FM25 reworded in this pass; FM49 the opposite mutant |
| C-F-MIN-6 (minor) | duplicated helpers and CI cost understated | accepted: the V10 intro lists every helper with its f297867e line range (verified) and the copies-identical rule, `lru_cache`, the per-class fit count; `seed_case` and `numpy_seed_rows` (2165-2194) and plan item 1 completed in this pass; 12.4 item 16 keeps the alternative |
| C-F-MIN-7 (minor) | which docstring test changes | accepted: one NEW `TestFourierMellinSwitch::test_the_docstring_documents_fourier_mellin`, three existing-pin edits (D22.16, V10 intro); plan item 1 corrected in this pass |
| C-F-MIN-8 (minor) | F1-F14 collide with the V9 fixtures | accepted: renamed FQ1 to FQ14 (D22 and V10 already; plan 12 and the roadmap in this pass) |
| C-F-MIN-9 (minor) | no warning literal; float-equality trigger | already-applied: frozen literal and the `< 1e-9` deg trigger in D22.7, the G7 constant non-identity arm in V10(h); 12.4 item 8 and FM36 propagated in this pass |
| C-F-MIN-10 (minor) | exact radial grid, tie and lag rules | already-applied: D22.3.3 to D22.3.6 (165 and 175 radii, 237600 and 252000 entries, re-derived), the evidence grid recorded and measured within 0.0005 deg (216 (iv)) |
| C-F-MIN-11 (minor) | weak or missing killers; no D/G tags | accepted: V10 already had the look-up-table pin, the (l) routing-invariance arm, the (d) rotation-centre band and the (j) `seed_spectra` count; item 4 rewritten in this pass with FM43, FM44 and a D/G tag on every mutant |
| C-F-MIN-12 (minor) | the note (viii) deviation unrecorded | accepted: D22.10 already recorded it with its reasons; 12.4 item 11 carries the batched host pre-pass as the alternative in this pass |
| C-F-MIN-13 (minor) | the `ebsd.py` import block is a conflict zone | accepted: D22.9 names `FOURIER_MELLIN_PROP_NAMES` in the existing `_engine` import block (`ebsd.py:59-63`, verified); the files-touched list states it in this pass (tech-stack.md lists that block among the NLPAR fan-out's append conflicts, verified) |
| C-F-MIN-14 (minor) | CPU route host memory unaccounted | already-applied (the bound recorded): D22.10 per-reference bytes, D22.18 the information message; risk 12 and the 12.3 freeing follow-up added in this pass |
| C-F-MIN-15 (minor) | the D15.6 edit departs from D20.5 | already-applied (the first option): no D15.6 text edit, D22.9 amends it by reference, Block 5 says so |
| C-F-MIN-16 (minor) | F11 resolves by an argument | accepted: V10(m)(2) already carried the census; FQ11 rewritten in this pass (the census, reopen above about 25 deg or at the edge bin, `grain_labels` and tuple references noted) |

<!-- ======== BLOCK 2: plan open question 5 amendment ======== -->
<!--
ANCHOR: plan.md, section 6, open question 5 ("**Fourier-Mellin
initial guess** (D5). DEFERRED from v1. ..."), after the bold
paragraph ending "Not yet commissioned; the default above stays in
force until it is.**" (lines 348-358 at f297867e). Append inside
the item.
-->

   **SPECIFIED 2026-10-07 (Stage F, section 12; requirements D22,
   validation V10).** Commissioned 2026-10-06; the spec is committed
   with its plan gate PENDING Johan's review. The default above stays
   in force: the FM seed is the opt-in `fourier_mellin` keyword,
   `"off"` by default and bitwise the translation-only seed; a
   default of `"auto"` is FQ3.

<!-- ============ BLOCK 3: plan 11.3 one-line amendment ============ -->
<!--
ANCHOR: plan.md, 11.3, first bullet, immediately after its bold
lead "**Stage F -- Fourier-Mellin initial guess, both backends
(PLANNED after Stage E, Johan decision 1 of 2026-10-06; not
commissioned).**" Insert as one sentence, not bold, before "Ernould
2020's cascaded Fourier-Mellin ...".
-->

COMMISSIONED 2026-10-06 and specified 2026-10-07: section 12,
requirements D22, validation V10 (plan gate PENDING Johan's review).
