# Phase 8 -- `spherical-pseudo-symmetry`: plan

Branch `spherical-pseudo-symmetry` off `develop` (after the Phase 11
merge, jwestraadt/kikuchipy#12). Models: plan/spec on Fable 5 (xhigh,
ultracode); tests, implementation, adversarial review and fixes by
Opus 5 agents (xhigh, ultracode). Autonomous mode (approval gate
waived for spherical-indexing phases; decisions flagged in section 8;
genuinely-user calls in section 9). Tests are written failing before
the code they exercise. **No measurement or execution happened at
drafting (2026-09-06/07): a ~1h compute job owned the machine -- see
requirements D11.** The measurement debt is discharged at the
tests/implementation gates; the adversarial review re-measures the
decision-critical values. The drafting adversarial review's 26
findings were folded in at drafting time (disposition appendix at the
end of this file); re-submitting the three documents to review is a
definition-of-done gate (validation.md).

## 0. Constitution amendments (applied in the spec commit)

1. `specs/roadmap.md`: the Phase 8 heading becomes
   "`spherical-pseudo-symmetry` (spec `2026-09-06-pseudo-symmetry`;
   un-deferred 2026-09-07)" -- recording the folder-vs-branch name
   mismatch explicitly (renaming instead is open question 9.6). The
   two roadmap boxes are rewritten into five gate-shaped boxes (the
   Phase 10 precedent): (1) `_pseudo_symmetry.py`
   (`find_pseudo_symmetry_operators` single/two-phase, optional
   correlation-volume ndarray, psymfile codec with the D2
   conjugation contract) -- the box's "optional volume + stereogram"
   wording amended per the resolution of open question 9.3
   (provisional: volume ndarray now, stereogram deferred with the
   Phase 9 visualisation half); (2) `pseudo_symmetry_ops` on
   `SphericalIndexer`/`EBSD.spherical_indexing` with the split
   row-width amendment (`_ROW_WIDTH_INDEX = 7`,
   `_ROW_WIDTH_REFINE = 6`) and the `pseudo_symmetry_index` prop,
   **plus `MasterPatternHarmonics.rotate`** (recorded scope growth
   on a Phase 2 class); (3) tests incl. the D9 ladder and the D8
   gated binary pins; (4) adversarial review + fixes; (5) signed
   commits + PR #13. Measured numbers enter the boxes as gates tick
   (D11). Refine-path ops enter box 2 only if open question 9.1
   resolves yes.
2. `specs/tech-stack.md:11` never-sweep list: extend to
   `doc/tutorials/spherical_indexing.ipynb` (it carries uncommitted
   user edits RIGHT NOW per git status; constitutional protection
   currently names only hybrid_indexing/load_save_data).
3. `specs/tech-stack.md` EMSphInx-quirks table, new rows: the
   `factor = 0.95` candidate gate; the 2-deg quaternion-dot dedup
   (pre-refinement, keep-brighter); the 0.999 cosine filter;
   `exclude_symmetry=True` deviating from the C++'s unfiltered
   stdout list. Preserved-quirk rows: exact identity skip on
   psymfile read; variants always Newton-refined even at
   `refine=False`; refine-failure returns the seed's analytic
   value; normalised variants refined un-normalised then
   denominator-divided. Not-reproduced rows: inert `pSym` wiring;
   `sl`-vs-`slP` MasterXcorr scan; `fabs(nFld - nFld)` dead branch.
4. `specs/tech-stack.md` "Numerics" and "Tests, docs, data":
   - the result-contract sentence: packed INDEXING result rows are
     now `(alpha, beta, gamma, score, phase_id, iq, psym_index)`
     (`_ROW_WIDTH_INDEX = 7`; fill `(0,0,0,0,-1,0,-1)`);
     refine-only rows stay 6 (`_ROW_WIDTH_REFINE`); the
     `"pseudo_symmetry_index"` prop (int32; 0 base, 1-based op
     index, -1 fill) exists only when `pseudo_symmetry_ops` was
     given, with the 2-D `"nbest_pseudo_symmetry_index"` beside it
     at `n_best > 1`; ops are single-phase-only.
   - the Euler bullet gains the frozen conversion sentence:
     "pseudo-symmetry operators are public in the NCC convention
     (`op * rot`); a psymfile row is the CONJUGATE (`~op`, D2 of
     the Phase 8 spec): EMSphInx's `qp = q0 * q` on
     MasterXcorr-printed quaternions equals `op * rot` with
     `op = rotation_from_zyz(zyz_peak) = ~q_printed`".
   - the env-var sentence gains **`KIKUCHIPY_LOCAL_MASTERS_DIR`**
     (a directory of user-supplied EMsoft master `.h5` files for
     skip-if-absent pseudo-symmetry tests; the second local-data
     convention beside `KIKUCHIPY_EMSPHINX_DIR`); the
     skip-if-absent tests' skip reasons name the exact expected
     filenames (open question 9.5 supplies them).
5. `specs/2026-08-16-sht-master-spectra-and-file/validation.md:53`:
   the pinned `rotate` `NotImplementedError` test is superseded by
   the D7 contract tests -- the recorded pin is updated where it
   lives.
6. Research addenda, `specs/_research/explore-emsphinx-core-
   algorithm.md` section 8 (new items, from the 2026-09-06
   reports): (52) the psym machinery is inert in every shipped CLI
   at 60f3517/87b2387 (`indexer.hpp:175-180`, no wiring, `clone()`
   drops; the fix is the user's feature/GPU two-liner); (53) the D2
   equivalence derivation (EMSphInx `q0 * q` o MasterXcorr output
   == NCC `op * rot` with `op = rotation_from_zyz(peak)`); (54)
   MasterXcorr defects: `sl` vs `slP` flat scan, line-267
   self-comparison, stale usage string, stdout-only output incl.
   true-symmetry rows; (55) `nml.hpp:247` never `ipath`-prefixes
   `pSymFile` (CWD resolution) -- the already-ported quirk's psym
   face. Cross-file pointer added to
   `explore-emsphinx-programs-and-formats.md`'s MasterXcorr
   section.
7. `specs/mission.md:23`: names `find_pseudo_symmetry_operators` +
   psymfile read/write + stereogram plot. If open question 9.3
   resolves to defer the stereogram (the provisional position), the
   stereogram clause moves to the Phase 9 visualisation half;
   otherwise verify wording, amend only if delivered names differ.
8. Tick roadmap Phase 8 boxes only as gates complete.

## 1. `_pseudo_symmetry.py`: codec + prediction (tests: `tests/test_indexing/test_spherical_pseudo_symmetry.py`)

1. Module skeleton: GPL header + CMU/Lenthe block + Changed-by
   notice per D10; module docstring mapping functions to
   `master_xcorr.cpp`/`master.hpp` line ranges; private helpers
   (`_local_maxima`, `_dedup_keep_brighter`, `_exclude_true_ops`)
   kept module-private.
2. Codec first (`read_emsphinx_psym_file` /
   `write_emsphinx_psym_file`, D6): grammar, qu-only ValueError,
   exact-identity skip, count validation, comma/whitespace, the
   D2.4 conjugation at both ends, the frozen full-precision writer
   format, round-trip identity. Failing tests: `TestPsymFileCodec`
   (grammar cases from `emsoft.hpp` semantics; the conjugation
   contract asserted two ways -- written rows == `(~ops).data` to
   1e-15, AND the D2.6a **golden literal file-content test**: one
   known non-involutory op produces the specific `w x y z` digits
   of `~op`, pinned bytes-exact).
3. `find_pseudo_symmetry_operators` (D3): steps 1-9 in order;
   `PseudoSymmetryOperators` dataclass + `save` (no `plot` this
   phase -- open question 9.3). The local-maxima scan is attempted
   vectorised (numpy over the true `(bwP, slP, slP)` cube with
   rolled views honouring the `emsphinx_compatible` glide via
   `_extract_neighborhood` parity tests); a numba kernel only if
   the vectorised form is measured too slow at bw 88-113 (then
   `@njit(cache=True, nogil=True)` + `.py_func` test; no
   `error_model="numpy"` without a recorded reason).
4. Failing tests: `TestFindPseudoSymmetryOperators` (the D9.1 Ni
   **positive-count pin** -- identity + (close to) the 24 proper Oh
   rotations at `exclude_symmetry=False` and a stated cutoff --
   then subset-of-Oh and `True`-empty, both sequenced after the
   count pin so neither is vacuous; sorted/normalised intensities;
   volume shape/optionality; cutoff filtering; guards D6.1-5;
   synthetic 3-fold/6-fold recovery about +z with the explicit
   cutoff 0.25 incl. one off-grid op for the 0.95 factor; the
   targeted ~3-deg dedup pair (D9.4); two-phase rotated-copy
   oracle via the Phase 6 composed-orientation identity;
   bandwidth-resize path), and
   `TestPseudoSymmetryOperatorsObject` (save round trip == codec;
   empty-set save ValueError).

## 2. `MasterPatternHarmonics.rotate` (tests: `tests/test_indexing/test_spherical_master_pattern_harmonics.py`)

1. Implement per D7 on top of `_wigner.rotate_harmonics`; the
   uniform flag-neutralization rule (phase replaced so
   `n_fold == 1`, `has_equatorial_mirror == False`); docstring
   states the active-rotation contract `g(n) = f((~rotation) n)`
   and the neutralization.
2. Replace the pinned `NotImplementedError` test with: the
   composition identity, the synthesis-oracle direction test, the
   identity-rotation no-op, the flag-neutralization assertion, a
   round trip (`rotate(R).rotate(~R)` recovers the coefficients),
   and the Phase 3 `rotate_harmonics` equivalence
   (`h.rotate(R).coefficients == rotate_harmonics(h.coefficients,
   zyz)` for the derived zyz -- pins the plumbing).

## 3. The indexing loop (tests: additions to `tests/test_indexing/test_spherical_indexer.py`)

1. `_indexer.py`: the `_ROW_WIDTH` split (`_ROW_WIDTH_INDEX = 7`,
   `_ROW_WIDTH_REFINE = 6`) + fill row `(0,0,0,0,-1,0,-1)` +
   unpacking + prop creation (only when ops given; winner 1-D +
   per-rank 2-D at `n_best > 1`) per D4; the psym loop in
   `_index_chunk` after each phase's insertion (base honours
   `refine`; variants seeded `op * rotation_from_zyz(zyz_best)`,
   ALWAYS `refine_zyz`d, same insertion; tie rule keeps the base);
   ctor + validation (`Rotation` flattened; size-0 ops
   None-equivalent; multi-phase ValueError, D5); provenance
   comment rewrite (D10); Notes amendments (the frozen
   one-candidate replacement wording, the `P*(1+N)` fill sentence,
   metric-inconsistency note, duplicate-rows note, memory/timing
   sentence at build).
2. Failing tests: seed-chain unit pin (== C++ `qu2zyz(q0 * q)` to
   1e-14, via `_euler`); multi-phase ValueError; wrong-op -> index
   0 on `nickel_ebsd_small`; true-op tie (measured-then-pinned;
   winner in {0, i}, base on exact ties -- the tie-rule pin);
   `refine=False` still refines variants (assert variant scores
   equal a direct `refine_zyz` call, not the interpolated value);
   `n_best = 1 + n_ops` ranked variants (orientations op-related,
   scores descending, psym column correct); the duplicate-rows pin
   (two variants converging into the same peak both kept, D4);
   fill-row shape/value pins; **refine rows stay width 6**
   (`refine_patterns` output shape unchanged -- the split-constant
   pin); the D9.5 rescue scenario (build-measured; recorded
   fallback per D9.5).

## 4. Signal methods (tests: additions to `tests/test_signals/test_ebsd_spherical_indexing.py`)

1. `EBSD.spherical_indexing`: parameter threading (appended after
   `emsphinx_compatible`, before `chunksize`), prop plumbing
   (winner-only 1-D `"pseudo_symmetry_index"`, plus 2-D
   `"nbest_pseudo_symmetry_index"` beside `nbest_phase_id` at
   `n_best > 1`), docstring Notes amendments; the NCC-docstring
   semantics sentence adapted verbatim-in-meaning from
   `ebsd.py:2801-2809`. `refine_orientation_spherical` /
   `refine_patterns` are NOT touched (open question 9.1; a yes
   re-opens this task group with the refine-contract amendment of
   D4).
2. Failing tests: prop existence iff ops passed; dtype int32;
   prop shapes at `n_best` 1 and > 1; the perturbation oracle
   mirroring `test_ebsd_refinement.py:617-670` -- perturb with
   `rot_ps[0]` -> index 2, `rot_ps[1]` -> index 1, wrong op -> 0 --
   run through `EBSD.spherical_indexing`; and the D2.6b
   **conjugation killer**: ONE non-involutory op through a psymfile
   write/read round trip feeding BOTH `refine_orientation` (NCC
   engine) and `EBSD.spherical_indexing`, exact-index and
   winner-misorientation agreement.

## 5. Binary interop (local-gated; tests in `test_spherical_pseudo_symmetry.py`, gated classes)

1. `TestMasterXcorrParity` (gated + weekly + pooch; D8.1): temp
   cwd, machine lock, bw 88 (asserted to satisfy
   `fast_size(2*bw-1) == 2*bw-1` -- the D8 bandwidth discipline,
   stated in the test), double-space-tolerant stdout parser (D2.7);
   kikuchipy ops == CONJUGATE of the printed rows at ~1e-6;
   intensity parity measured-then-pinned; count parity under
   `exclude_symmetry=False` + same cutoff; record the fftw.wisdom
   caveat in the failure message (Phase 10 convention).
2. `test_two_file_branch_same_master` (gated; D8.2i): the same ni
   h5 passed TWICE -- exercises the two-file branch (argmax
   seeding instead of the identity cell) with no new data; the
   mandatory local-gated two-phase discriminator. The true
   two-master run vs `ebsd_master_pattern("al")` is weekly
   (D8.2ii, open question 9.7).
3. `TestIndexEBSDPsymFile` (gated; D8.3): the inertness pin exactly
   per D1 (one-op psymfile: `Scan 1` dataset-level equal to the
   no-psym run excluding the `EMheader` time fields; `Scan
   2/EBSD/Data` Phase = 255 / identity Euler / Metric = 0 / IQ = 0;
   image maps never asserted; `ipath` empty) and the error-path
   pins (eu-type file; count mismatch; two-phase + psymfile).
   Reuses the Phase 10 canonical-route input builders.
4. These tests ship skipping (env var absent on CI) and are first
   executed locally at the implementation gate -- after the compute
   job has finished (D11).

## 6. Tutorial, CHANGELOG, docs

1. CHANGELOG `Added` entry (PR #13 link) with the code commit,
   naming `find_pseudo_symmetry_operators`, the psymfile codec,
   `pseudo_symmetry_ops` on `SphericalIndexer` /
   `EBSD.spherical_indexing`, and `MasterPatternHarmonics.rotate`.
2. Tutorial `##` section before "What's next?": predict on the
   tutorial's Ni master (`exclude_symmetry=False` showing the Oh
   subset, then `True` showing empty -- the honest "Ni has no
   pseudo-symmetry" teaching moment), a synthetic-blend
   demonstration, `pseudo_symmetry_ops` on
   `EBSD.spherical_indexing` with the index prop mapped, an
   XDMF/ParaView note for `keep_volume=True`, and the psymfile
   export cell. Stored outputs per notebook rules; nbval wiring
   only if new nondeterminism appears. **Sequenced behind the
   user's uncommitted notebook edits (open question 9.8)** -- the
   code commits never touch the notebook until cleared (the
   never-sweep amendment in 0.2 makes this constitutional).
3. Docs plumbing: the quirks-table rows (0.3), the `.pyi` `__all__`
   additions (4 names, D10; the numpydoc API reference regenerates
   from `__all__`), the `_indexer.py:74-81` provenance rewrite
   ("deliberately not ported ... pseudo-symmetry" is now false),
   the new module's GPL + CMU/Lenthe block with
   `master_xcorr.cpp`/`indexer.hpp` line references and a dated
   modification notice. No new docs pages.

## 7. Adversarial review and fixes

1. Fidelity reviewer refutes against: `master_xcorr.cpp` (steps/
   constants of D3, incl. the true-cube deviation, the seed
   translation, and the near-identity seed equivalence for fast
   bandwidths), `indexer.hpp:228-261` under D1 (base-vs-variant
   refine asymmetry; insertion; tie rule; seed chain),
   `master.hpp:220-233` + `emsoft.hpp` (codec grammar,
   exact-identity skip), the D2 derivation line by line against
   `specs/tech-stack.md`'s Euler contract, and the NCC mirror
   against `_refinement.py`/`_solvers.py` (variant construction,
   argmax, prop semantics, 1-based index). Conventions reviewer:
   GPL/CMU blocks, `__init__.pyi` sorting, no phase numbers in
   public docstrings, message style, keyword placement deviations
   recorded, coverage 100 %, the EMSphInx-gotcha checklist of
   `specs/_research/explore-emsphinx-core-algorithm.md` section 8.
2. Bug-injection (mutation) list -- every mutant dies by a named
   test or is recorded reviewed-only with its killer stated:
   - codec read drops the conjugation (returns `Rotation(rows)`):
     dies by the D2.6b psymfile-round-trip perturbation oracle
     (non-involutory op: index flips 1 -> 0) and by MasterXcorr
     parity only in the two-phase case (autocorr sets are
     inversion-closed -- stated honestly).
   - codec write drops the conjugation but read keeps it: dies by
     the round-trip identity test AND the golden-bytes/written-rows
     unit pins.
   - identity row not skipped on read: dies by the codec unit test
     (count) and by a wasted-variant tie in the indexer tests.
   - `find_` returns `~rotation_from_zyz` (double conjugation):
     dies by the two-phase composed-orientation oracle; the
     single-phase Ni test is conjugation-blind (Oh closed) --
     stated.
   - `v_max` taken from the coarse argmax in single-phase mode (not
     the near-identity refine): dies by the Ni intensity pin
     (identity autocorrelation == 1.0 within refinement tolerance).
   - search factor 0.95 dropped (threshold = cutoff): dies by the
     off-grid synthetic-op test.
   - dedup keeps the dimmer: dies by the ~3-deg close-pair test
     (D9.4).
   - **half-angle/misorientation threshold swap** (2 deg read as
     misorientation, or 0.999 halved/doubled): dies by the ~3-deg
     dedup pair, which sits between the two readings.
   - **tie-rule inversion** (variant beats base on exact ties):
     dies by the true-op tie pin.
   - **`slP`/`sl` index confusion reintroduced**: dies by a
     `_local_maxima` unit test at a bandwidth where
     `fast_size(2*bw - 1) != 2*bw - 1`.
   - exclusion uses ALL 48 Oh ops (improper included): improper
     ops are not rotations of the volume, no peak matches them --
     no observable; reviewed-only, killer = code inspection.
   - exclusion threshold 0.999 -> 0.9: dies by the synthetic blend
     test (verify direction at build; else reviewed-only).
   - variants refined only when `refine=True`: dies by the
     `refine=False` variant-score test (3.2).
   - seed uses the running global best instead of the phase best:
     unreachable while ops are single-phase-only (one phase == both
     definitions coincide) -- recorded unreachable, guarded by the
     single-phase ValueError test.
   - seed composition `rot * op` instead of `op * rot`: dies by the
     seed-chain 1e-14 unit pin and the perturbation oracles.
   - psym column 0-based (`ops[i]` reported as i): dies by the
     NCC-mirror oracles (index 2 expected where the inverse op
     wins).
   - **fill psym value 0 instead of -1**: dies by the fill-row pin.
   - row-width constants swapped or refine rows widened to 7: dies
     by the refine-rows-stay-6 pin plus every existing
     `refine_patterns` test.
   - `_ROW_WIDTH_INDEX` unpack off-by-one: dies by every existing
     indexer test (scores/iq garbage) -- belt and braces.
   - prop created when ops is None: dies by the
     prop-existence-iff-ops test.
   - insertion accepts non-positive variant scores: dies by the
     existing zero-seeded insertion tests + a failed-refine variant
     case.
   - `rotate` direction flipped: dies by the synthesis-direction
     oracle (2.2); the blend recovery is inversion-blind -- stated.
   - **flag-neutralization removed from `rotate`**: dies by the
     neutralization test; consequence otherwise = wrong
     plane-skipping in the correlator, also caught by the blend
     recovery (the correlator folds by the first master's flags).
   - MasterXcorr parity parses ours conjugated: dies by the parity
     test (2x angular error on every non-involutory op).
3. Fix, re-run (`-n 0` once before `-n 4` if a numba kernel was
   added), `pre-commit run --files <changed>` (never `specs/`),
   coverage 100 % of touched spherical modules; re-measure the
   review-critical numbers (D11.3).

## 8. Open questions -- decided at drafting (autonomous mode), flagged for review

1. **Reference baseline = 60f3517 + the user's feature/GPU wiring**
   (D1): the shipped CLI is provably inert, so "strict 60f3517"
   would mean porting a no-op; the two-line wiring is the user's
   own work and the loop logic is identical to master. Recorded
   with provenance; the inertness itself becomes a gated binary
   pin.
2. **Keep the literal `q0 * q` composition; put the conjugation in
   the codec** (D2): the derivation shows the upstream chain is
   self-consistent with MasterXcorr's output, so the "author doubt"
   resolves without divergence; the public convention is the NCC
   one, and only the file boundary converts. Killers named (D2.6);
   the write direction's analytically-derived-only status is
   recorded (D2.5).
3. **Variants always Newton-refined even at `refine=False`** (D4):
   ported as-is (the user's CUDA port preserved it); the metric
   inconsistency is documented, not "fixed".
4. **Winner reporting via the split row widths +
   `pseudo_symmetry_index` prop** (D4): the NCC mirror beats
   EMSphInx's rank-only scans (which cannot attribute a variant to
   an operator); the frozen packed-row contract is amended on the
   record -- indexing rows only, refine rows untouched -- rather
   than smuggling a second result path.
5. **Normalised-correlation subtlety replicated** (refine
   un-normalised, divide by the analytic denominator at the refined
   orientation): it is what Phase 7's `refine_zyz` already does;
   consistency chosen over "fixing" EMSphInx (`sht_xcorr.hpp:
   1168-1172` caveat ported as-is).
6. **Exact-identity skip on psymfile read kept literal** (D6): a
   tolerance would be a behavioural divergence with no upstream
   precedent; near-identity ops are harmless (documented).
7. **`bandwidth` default 88** (D3): the C++ requires it explicitly;
   the default is a kikuchipy affordance -- the smallest
   recommended-and-fast value. (The `cutoff` default is open
   question 9.4 for the user; no test depends on it.)
8. **`exclude_symmetry=True` default** (D3.7): the returned object
   is meant to feed `pseudo_symmetry_ops`, where true-symmetry ops
   are pure waste; the C++ printed list's unfiltered behaviour is
   available via `False` (and is the parity setting).
9. **No shipped psym `.npz`; parity is gated-only** (D8): no
   shippable pseudo-symmetric input exists and the shipped binary
   cannot exercise the loop.
10. **`MasterPatternHarmonics.rotate` implemented now** (D7): Phase
    8 is its natural consumer; keeping the stub would force tests
    onto the private `rotate_harmonics`. Uniform flag
    neutralization chosen over a z-rotation special case
    (simpler, safe; tests reassign the phase when they need
    folding).
11. **`KIKUCHIPY_LOCAL_MASTERS_DIR`** as the local-masters
    convention (D9.6): `KIKUCHIPY_DATA_DIR` redirects the pooch
    cache and is not a foreign-file dropbox.
12. **Spec folder name `2026-09-06-pseudo-symmetry` kept,
    provisionally**, with the deviation recorded in the roadmap
    heading (0.1); the rename alternative is open question 9.6 for
    the user.
13. **No new benchmark**: the per-op cost is one `refine_zyz` per
    pattern per op, already baselined in Phase 7; a Notes sentence
    carries the measured ratio at build (D11).

## 9. Open questions FOR THE USER (not decided autonomously)

1. **Refine-path scope**: should `EBSD.refine_orientation_spherical`
   and `SphericalIndexer.refine_patterns` gain `pseudo_symmetry_ops`
   in Phase 8? EMSphInx's refine-only work item never consults
   `pSym`, so this would be a kikuchipy-native extension mirroring
   NCC's `refine_orientation`. Provisional spec position until
   answered: **no** -- refine rows stay width 6 (split constant),
   the D2 cross-engine killer uses the existing NCC
   `refine_orientation` plus `spherical_indexing`. A yes grows the
   refine-row contract to 7 and defines the unrefined-point variant
   sentinel.
2. **feature/GPU rebuild**: build a patched CPU IndexEBSD.exe with
   your own two wiring lines (`idx.hpp:386`, `indexer.hpp:244` on
   feature/GPU) for a one-shot manual end-to-end validation of the
   psymfile write direction? Provisional: no -- the direction is
   recorded as analytically derived and empirically unverifiable,
   with parity scoped to inertness + error paths.
3. **Stereogram/volume scope**: the roadmap box promises "optional
   volume + stereogram"; D3 drops the C++ writers. Provisional:
   volume returned as an ndarray option now, stereogram deferred
   with the Phase 9 visualisation half, roadmap box amended
   accordingly. Confirm or pull the stereogram into Phase 8.
4. **Default `cutoff`** for `find_pseudo_symmetry_operators`
   (EMSphInx has no default -- it is a positional CLI argument).
   Provisional: 0.5; all tests pass explicit cutoffs so the default
   is not load-bearing.
5. **Local hcp/TiAl masters**: which EMsoft master h5 files exist on
   this machine (exact filenames + phases) for the
   `KIKUCHIPY_LOCAL_MASTERS_DIR` skip-if-absent tests?
6. **Folder name**: keep `specs/2026-09-06-pseudo-symmetry` with the
   deviation recorded in the roadmap line (provisional), or rename
   to match branch `spherical-pseudo-symmetry` before anything
   lands?
7. *(minor)* Two-master weekly parity uses the
   `ebsd_master_pattern("al")` download (~0.3 GB, cached). Object if
   unwanted; the recorded-gap alternative is drafted in
   validation.md.
8. **The tutorial notebook's uncommitted local edits**
   (`doc/tutorials/spherical_indexing.ipynb`, modified in the
   working tree): commit/stash them yourself, or approve the exact
   sequencing for the Phase 8 section append? Until cleared, Phase
   8 commits exclude the notebook.

## 10. Commit and PR

Signed commits in gate order: (1) spec + section-0 amendments
(constitution edits + research addenda + roadmap heading); (2)
failing tests (placeholder bands marked, gated tests skipping); (3)
implementation + measured pins + CHANGELOG (+ tutorial only per
9.8); (4) review fixes + re-measurements. Never touch the user's
uncommitted notebook edits (`spherical_indexing.ipynb`,
`hybrid_indexing.ipynb`, `load_save_data.ipynb`) or
`specs/2026-08-16-constitution/upstream-issue.md` (also locally
modified). Push; PR **#13** into fork `develop` with the template,
the GPL statement (new code GPL-only; the CMU/Lenthe block carried),
and the D11 measurement-deferral note in the description.

## Appendix: adversarial review of the spec -- findings applied at drafting, 2026-09-07

The drafting review returned 26 findings (F1-F26; F1 and F12 the two
blockers) against the drafting decisions and the two research
reports, before any spec file existed. Every finding was folded into
these three documents at drafting time; none was rejected. The
re-review diffs against this table.

- **F1 (blocker -- no spec files on the branch)**: applied. The
  three documents were drafted fresh with all findings folded in;
  re-submission to adversarial review is a definition-of-done gate
  (validation.md). Branching/committing happens outside the
  drafting session (compute job).
- **F2 (extra-scan ground truth wrong)**: applied -- requirements
  D1: the pin asserts Phase = 255 / identity Bunge Euler /
  Metric = 0 / IQ = 0 on `Scan i/EBSD/Data`, not raw
  `corr=0/phase=-1/qu=0`; the research-report error is corrected in
  Context.
- **F3 (fast-size cube / oracle bandwidth discipline)**: applied --
  D3.4 (scan the true `(bwP, slP, slP)` cube, recorded deviation)
  and D8 (every oracle run pins `fast_size(2*bw-1) == 2*bw-1`,
  safe/unsafe lists stated).
- **F4 (two-phase oracle coverage)**: applied -- D8.2: the
  mandatory same-file-twice discriminator plus the weekly
  two-master parity run (or its recorded-gap alternative, open
  question 9.7).
- **F5 (row-width amendment must not touch refine rows)**: applied
  -- D4: `_ROW_WIDTH` split into `_ROW_WIDTH_INDEX = 7` /
  `_ROW_WIDTH_REFINE = 6`; refine contracts untouched (with F15).
- **F6 (constants without stage/metric; returned-list semantics;
  blend flags)**: applied -- D3 steps 4-7 pin each constant's stage
  and metric and freeze the per-mode returned-list semantics; D7's
  uniform flag-neutralization rule.
- **F7 (v_max seeds unspecified)**: applied -- D3.3: both seeds are
  requirements; intensity parity depends on them.
- **F8 (stdout precision facts wrong)**: applied -- D2.7:
  `Quat::to_string(6)`, leading alignment spaces, double-space
  tolerant parser, quaternion comparison at ~1e-6.
- **F9 (public NotImplementedError embeds a phase name)**: applied
  -- D7 removes the stub; plan 0.5 updates the recorded pin where
  it lives.
- **F10**: applied -- folded into the redrafted decisions with the
  rest (the review-fixer's blueprint records it applied without a
  separate named anchor).
- **F11 (spec folder name vs branch name)**: applied -- the
  deviation is recorded in the roadmap heading (plan 0.1); the
  rename alternative is open question 9.6 (with F26).
- **F12 (blocker -- duplicate of F1)**: applied with F1.
- **F13 (psymfile write direction)**: parts a+b applied -- the
  golden literal file-content test (D2.6a, format frozen in D6) and
  the MasterXcorr conjugate-of-printed-rows parity assertion
  (D2.6d). Part c: the write direction's analytically-derived,
  empirically-unverifiable status is recorded (D2.5); the
  feature/GPU CPU rebuild is open question 9.2.
- **F14 (Ni tests vacuous on an empty return)**: applied -- D9.1
  positive-count pin (identity + close-to-24 proper Oh); the
  subset/empty tests sequenced after it.
- **F15 (refine-path ops scope)**: open for the user -- question
  9.1; provisional position no; D4 keeps refine rows at width 6.
- **F16 (D4 under-specification)**: applied -- D4: fill value -1,
  the `n_best > 1` prop layout (winner 1-D +
  `nbest_pseudo_symmetry_index` 2-D), the `P*(1+N)` fill-row
  sentence, the duplicate-rows documentation + pin, the tie rule;
  D3.1: `cutoff`/bandwidth validation (F16.6).
- **F17 (guard semantics)**: applied -- D6 guards 1-5, incl. the
  recorded orix-unitization deviation (F17.4).
- **F18 (performance unaddressed)**: applied -- D11 (a)-(d):
  ops-scaling baseline, hard-floor scoping + 2-op re-measurement,
  BatchEstimate deviation record, Notes-caveat.
- **F19 (angle-space ambiguity)**: applied -- D3's frozen
  angle-space statement (both thresholds are quaternion-dot
  half-angles, misorientation equivalents stated), the ~3-deg dedup
  case (D9.4), and the threshold-swap mutation entry (7.2).
- **F20 (D1 build traps: double-ipath, image-map bug, EMheader)**:
  applied -- D1: `ipath` empty, image maps never asserted,
  dataset-level Scan 1 equality excluding the EMheader time fields.
- **F21 (identity-seed translation; two-file branch untested)**:
  applied -- D3.3 (seed translated to the slP grid; two-file mode
  seeds from the interpolated argmax) and D8.2i (same-file-twice
  discriminator).
- **F22 (blend fixtures fragile; cutoff coupling)**: applied --
  D9.2: blend ops about +z only, flags neutralized anyway,
  test-local coefficient-addition helper, explicit cutoff 0.25
  decoupled from the default; the default `cutoff` itself is open
  question 9.4.
- **F23 (plan obligations)**: applied -- plan 0.2 (never-sweep
  extension for the tutorial notebook), 0.3 (quirks rows), 0.4
  (env-var constitution note), and task groups 5-6.
- **F24 (roadmap drift)**: applied -- plan 0.1: the folder name in
  the heading, `MasterPatternHarmonics.rotate` added to the box,
  the "optional volume + stereogram" wording amended per open
  question 9.3's resolution.
- **F25 (D2 derivation must survive the session)**: applied -- the
  full derivation is carried in requirements D2 itself; every
  load-bearing research citation is inlined in D1-D9 and Context.
- **F26 (folder deviation + env var + local filenames)**: applied --
  deviation recorded (0.1), `KIKUCHIPY_LOCAL_MASTERS_DIR` named in
  the constitution note (0.4) with skip reasons naming exact
  filenames; the filenames are open question 9.5; the folder rename
  is open question 9.6.

Rejected: none. Every minor was verified against EMSphInx source by
the reviewer and none conflicts with cited EMSphInx behaviour.
