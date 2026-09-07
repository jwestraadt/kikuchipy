# Phase 8 -- `spherical-pseudo-symmetry`: validation

**Status note (requirements D11): drafted with ZERO test execution
and ZERO measurement** -- a ~1h compute job owned this machine's
working tree throughout drafting (editable install;
timing-sensitive), so nothing here has run and nothing may run until
the job releases the tree. Every tolerance marked `MTP` is
measured-then-pinned at the tests/implementation gates
(`pytest.approx(measured, rel=0.05)` or the Phase 6 ~1.7-2.1x margin
convention on misorientation bands); validation.md therefore ships
with placeholder tables labelled "measure at implementation gate",
and "Recorded results" below is intentionally empty except for the
deferral record. Tests marked [binary] need `KIKUCHIPY_EMSPHINX_DIR`
(+ the machine-wide lock); everything else is pure Python on
shipped/synthetic data.

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing tests/test_signals -k "spherical" -n 4
```

(one `-n 0` run first if a numba kernel is added -- plan 1.3).

### Codec (`TestPsymFileCodec`, `tests/test_indexing/test_spherical_pseudo_symmetry.py`) -- pure Python

- `test_read_qu_file`: header `qu`, count, comma/whitespace-mixed
  rows parse to the conjugated `Rotation` (D6/D2.4). [D6]
- `test_read_rejects_non_qu`: `eu`/`om`/`ax`/`ro`/`ho`/`cu` tokens
  raise ValueError (mirrors `master.hpp:225`). [D6]
- `test_read_skips_exact_identity` / `test_read_keeps_near_identity`
  (`1.0000001 0 0 0` kept -- exact-equality parity,
  `master.hpp:227-231`). [D6]
- `test_read_count_mismatch_raises` (too few AND too many). [D6]
- `test_write_format`: `qu` header, count, the frozen full-precision
  `w x y z` rows; rows equal `(~ops).data` to 1e-15 (the conjugation
  unit pin). `test_write_empty_raises`. [D6/D2]
- `test_write_golden_bytes` (the D2.6a killer): writing ONE known
  non-involutory op produces the specific `w x y z` digits of `~op`,
  derived from the D2 derivation and pinned **bytes-exact** against
  a literal expected file body (well-defined because D6 freezes the
  writer's number format). [D2/D6]
- `test_round_trip_identity`: `read(write(ops)) == ops` exactly. [D6]

### Prediction (`TestFindPseudoSymmetryOperators`) -- pure Python

- `test_ni_identity_plus_oh_count` (the D9.1 positive-count pin,
  `exclude_symmetry=False`, stated cutoff, in-package Ni `.sht` at
  bw 68 or 88): the return holds the identity plus (close to) the
  24 proper Oh rotations (Phase 4 measured exactly this cube,
  roadmap.md:66); each op within an MTP angular tolerance of a
  proper Oh rotation; intensities MTP (AMENDED 2026-09-07: the
  drafted "~1.0 with identity == 1.0" expectation is withdrawn --
  the identity-seeded refine's stalled `v_max` makes values above 1
  the faithful expectation, see Recorded results); intensities
  sorted descending. [D9.1, roadmap box 2]
- `test_ni_ops_subset_of_oh` / `test_ni_exclude_symmetry_empty`
  (`True` -> zero operators): both sequenced after the count pin so
  neither can pass on an empty return. [D3.7/D9.1]
- `test_cutoff_filters` / `test_cutoff_out_of_range_raises` /
  `test_volume_shape_and_optionality` (`(bwP, slP, slP)` float64;
  `None` unless `keep_volume`). [D3]
- `test_guards`: size-0 ops None-equivalent; all-identity psymfile
  -> empty list -> behaves as `ops=None` (D6 guards 1 and 4; guards
  2-3 are documentation, checked by the conventions review). [D6]
- `test_synthetic_three_fold_recovered` /
  `test_synthetic_six_fold_recovered`: blend
  `f + lam * f.rotate(S)` with S about +z, blend flags neutralized,
  **explicit cutoff 0.25** (D9.2); the specific expected op
  recovered within MTP of S or ~S; intensity MTP. [D3/D7/D9.2]
- `test_off_grid_operator_found`: S at a z-angle off the euler
  grid; kills a dropped 0.95 factor (MTP feasibility at build;
  recorded fallback: a threshold-sweep unit test on
  `_local_maxima`). [D3.4]
- `test_dedup_three_deg_pair` (D9.4): two synthetic peaks ~3 deg
  apart in quaternion-dot space -- one kept, the brighter wins;
  sits between the half-angle and misorientation readings of the
  2-deg threshold, so the plan-7.2 convention-swap mutants die
  here. [D3.5/D9.4]
- `test_two_phase_rotated_copy`: `find_(h, h.rotate(S))` recovers
  op == S (composed-orientation identity, Phase 6 measured
  0.68/1.07 deg -- the pure-Python conjugation-sensitive oracle).
  [D2.6c/D3.9]
- `test_bandwidth_resize_path`: harmonics at bw 384 `.sht` resized;
  documented non-equivalence note asserted only structurally (no
  crash, ops returned). [D3.1]

### Result object (`TestPseudoSymmetryOperatorsObject`) -- pure Python

- `test_save_round_trip` (== codec); `test_save_empty_raises`. (No
  `plot` this phase -- open question 9.3.) [D3]

### `rotate` (`tests/test_indexing/test_spherical_master_pattern_harmonics.py`) -- pure Python

- `test_rotate_composition` (`rotate(r1).rotate(r2) ==
  rotate(r2*r1)` on coefficients, 1e-12); `test_rotate_direction`
  (synthesis oracle: a feature at n0 moves to rotation*n0);
  `test_rotate_identity_noop` (no-op on the COEFFICIENTS only; the returned
  object's symmetry flags are still neutralized per the uniform D7 rule, so
  the assertion must not compare the full object); `test_rotate_round_trip`
  (`rotate(R).rotate(~R)` recovers the coefficients);
  `test_rotate_neutralizes_symmetry_flags` (`n_fold == 1`,
  `has_equatorial_mirror == False` on the returned object);
  `test_rotate_matches_rotate_harmonics` (plumbing pin). Replaces
  the pinned `NotImplementedError` stub test (plan 0.5). [D7]

### Indexing loop (`tests/test_indexing/test_spherical_indexer.py`) -- pure Python

- `test_seed_chain_matches_cpp` (unit: seed zyz == `qu2zyz(q0*q)`
  chain through `_euler`, 1e-14). [D4/D2]
- `test_ops_multi_phase_raises` (ValueError, frozen message). [D5]
- `test_wrong_op_index_zero` (`nickel_ebsd_small`, 30 deg z op:
  prop all zeros; winner orientation unchanged within MTP).
  [D4/D9.1, roadmap box 2]
- `test_true_op_ties_base` (an Oh op: variant score MTP-equal to
  base; winner index in {0, i} recorded, base expected on exact
  ties -- the D4 tie-rule/`upper_bound` pin). [D4]
- `test_variants_refined_at_refine_false` (variant rows carry
  analytic `refine_zyz` scores, not interpolated ones). [D4]
- `test_n_best_ranked_variants` (`n_best = 1 + n_ops`: kept rows
  op-related to the base within MTP; scores descending; psym column
  values correct; fill rows `(0,0,0,0,-1,0,-1)`). [D4]
- `test_duplicate_variant_rows_survive` (the D4 duplicate-rows pin:
  two variants Newton-converging into the same peak both kept,
  distinguished only by variant index -- no post-refinement dedup
  in the indexing loop, matching EMSphInx). [D4/D3.6]
- `test_row_width_seven_and_fill` (indexing rows width 7 with and
  without ops; fill psym value -1). [D4]
- `test_refine_rows_stay_width_six` (`refine_patterns` output shape
  unchanged -- the `_ROW_WIDTH_INDEX`/`_ROW_WIDTH_REFINE` split
  pin; provisional on open question 9.1). [D4]
- `test_rescue_scenario` (D9.5, build-measured: fixed-seed noisy
  synthetic-blend patterns; without ops the coarse winner is the
  pseudo basin -- pinned; with ops the winner has
  `pseudo_symmetry_index != 0`, higher score, true orientation
  within MTP). Recorded fallback if no deterministic construction
  survives measurement: unit-level forced-candidate insertion test
  + the cross-engine oracle below, and the limitation recorded
  here. [D4/D9.5]

### Signal methods (`tests/test_signals/test_ebsd_spherical_indexing.py`) -- pure Python

- `test_pseudo_symmetry_index_only_with_ops` (prop absent
  otherwise); `test_prop_dtype_and_shape` (int32; 1-D winner-only
  at `n_best=1`; at `n_best > 1` the winner-only 1-D prop PLUS the
  2-D `(n, n_best)` `"nbest_pseudo_symmetry_index"` beside
  `nbest_phase_id`). [D4]
- `test_spherical_indexing_perturbation_oracle`: the NCC scheme
  (`test_ebsd_refinement.py:617-670`) run through
  `EBSD.spherical_indexing`: perturb with `rot_ps[0]` -> index 2,
  `rot_ps[1]` -> index 1, wrong op -> 0 (exact indices,
  non-involutory ops). [D4/D9.3]
- `test_conjugation_killer_via_psymfile`: ONE non-involutory op
  (~25 deg, low-symmetry axis) written + read back; feeds BOTH
  `refine_orientation` (NCC engine -- an independent
  detector-space implementation) and `EBSD.spherical_indexing`;
  both report the same, correct 1-based index on a map perturbed
  by that op, winners' misorientation compared. THE decisive
  cross-engine conjugation test (D2.6b). [D2/D6/D9.3]
- (Refine-path tests enter only if open question 9.1 resolves yes;
  the refine contract is then amended in D4 first.)

Kernel hygiene: any new numba kernel gets a `.py_func` test and the
kernel-flag test stays unchanged -- no new `error_model="numpy"`
without a recorded reason per the tech-stack rule
(tech-stack.md:44). [D10]

## Local-gated (KIKUCHIPY_EMSPHINX_DIR; skipped on CI) -- [binary]

All classes hold `<tmp>/kikuchipy-emsphinx-program.lock`; every exe
run gets an isolated temp CWD (the exe hard-writes four files into
the CWD, D8); failure messages name the fftw.wisdom state as suspect
#1 (Phase 10 convention). Oracle bandwidths are asserted to satisfy
`fast_size(2*bw - 1) == 2*bw - 1` and to lie inside the binary's
`[53, 313]` clamp (D8). First executed at the implementation gate,
AFTER the compute job has finished -- never during drafting.

- `TestIndexEBSDPsymFile` (cheap, gated only):
  - `test_psymfile_is_inert_at_60f3517`: one-op psymfile on the
    Phase 10 canonical small route, `ipath` empty; `Scan 1`
    dataset-level equal to the no-psym run's `Scan 1` excluding the
    `EMheader` StartTime/StopTime/PatPerS fields; `Scan 2/EBSD/
    Data` asserts Phase = 255 (uint8), Phi1/Phi/Phi2 = the Bunge
    Euler of the identity, Metric = 0, IQ = 0; the per-scan
    `IPF/XC/IQ Map` images are NEVER asserted (shipped save() bug).
    The D1 baseline's executable evidence. [D1/D8.3]
  - `test_eu_type_psymfile_rejected` /
    `test_psymfile_count_mismatch_rejected` /
    `test_two_phase_with_psymfile_rejected` (exit != 0). [D6/D5]
- `TestMasterXcorrParity` (gated + weekly + pooch;
  `ebsd_master_pattern("ni")` h5 cached locally):
  - `test_masterxcorr_stdout_parity`: bw 88, shared cutoff,
    `exclude_symmetry=False`, at-bandwidth `from_master_pattern`
    construction; the double-space-tolerant parser (D2.7);
    kikuchipy's returned ops equal the **conjugate** of the printed
    rows at ~1e-6; intensities within an MTP band; both engines
    find the identity + proper-Oh set; count parity.
    [D8.1/D2.6d/D3]
  - `test_two_file_branch_same_master` (D8.2i): the same ni h5
    passed TWICE -- exercises the two-file branch (sub-pixel
    interpolated argmax seeding instead of the identity cell) with
    no new data; the mandatory two-phase-mode discriminator.
    [D8.2/D2.6c]

## Local-gated (KIKUCHIPY_LOCAL_MASTERS_DIR)

- hcp/TiAl mechanism tests on user-supplied EMsoft master h5 files
  (D9.6): `find_pseudo_symmetry_operators` on a genuinely
  pseudo-symmetric phase returns a non-empty operator set at
  `exclude_symmetry=True` with MTP intensity/angle pins recorded on
  first execution. Tests glob for the documented candidate names
  and `pytest.skip` with a reason naming the exact expected
  filenames (supplied by open question 9.5; the tests ship dormant
  until then). [D9.6, roadmap box 2]

## Weekly

- The MasterXcorr parity test above (`@pytest.mark.weekly` on top of
  the gate -- h5 SHT at bw 88 on the 1001 px master is the cost).
- `test_two_master_parity_al` (D8.2ii): true two-master run,
  `MasterXcorr <bw> <cutoff> <ni.h5> <al.h5>` vs
  `find_(ni_h, al_h)` -- download-gated on
  `ebsd_master_pattern("al")` (~0.3 GB, cached per
  tech-stack.md:50; skips cleanly without pooch). **Recorded-gap
  alternative if open question 9.7 rejects the download**: the
  two-file branch is then covered only by the same-file-twice
  discriminator plus the pure-Python rotated-copy oracle, and this
  gap is recorded here as a limitation.
- A full-map psym run (nickel_ebsd_large route, 2 ops): timing
  recorded (see Performance).
- The synthetic rescue scenario at a second bandwidth (robustness,
  MTP), if D9.5 lands with margin.

## Performance (D11; measured at the implementation gate)

| measurement | recipe | recorded value |
|---|---|---|
| ops-scaling baseline: pat/s at 0/1/2/4 ops, bw 68, `nickel_ebsd_small` | fixed-seed timing run, warm numba caches | measure at implementation gate |
| per-op increment vs the Phase 7 refine baseline (~3.0 ms `n_fold` 1 / ~0.65 ms m-3m at bw 68; each op ~25 % of a coarse correlate) | derived from the row above | measure at implementation gate |
| hard floor `>= 2 pat/s/core`: scoped to `ops=None`; psym run at 2 ops re-measured and recorded (not floored) | existing hard-floor test + a recorded 2-op run | measure at implementation gate |
| `BatchEstimate` deviation: chunk sizing ignores per-op cost -- recorded deviation + chunksize guidance in the Notes, no model change | docs check + a recorded observation | record at implementation gate |
| `EBSD.spherical_indexing` Notes refined-ratio claims (1.05-1.27x): caveat added that ops are not covered + a new measurement entry | docs check + the 2-op run | record at implementation gate |

## Manual

- Optional (user, open question 9.2 / D8.5): rebuild `IndexEBSD`
  from `feature/GPU` (the two-line wiring branch), run a one-op
  psymfile scenario, and compare its extra scans against
  `spherical_indexing(..., n_best=1+n_ops)` rank-for-rank -- the
  only possible live-loop binary oracle for the psymfile WRITE
  direction (D2.5). Recorded here if run.
- Only if open question 9.3 pulls the stereogram into Phase 8:
  eyeball the operator stereogram for the synthetic 3-fold blend vs
  the `pseudo.svg` MasterXcorr writes in the parity run's temp cwd
  (expect: same axes; C++ draws every op as a rotor -- its line-267
  dead branch).
- Tutorial section renders + nbval passes locally (per plan 6.2 and
  the 9.8 sequencing).

## Definition of done

- All gates of the roadmap sentence: spec recorded; failing tests
  committed; implementation; adversarial review (incl. the
  `specs/_research/explore-emsphinx-core-algorithm.md` section 8
  checklist + the plan 7.2 mutation list) + fixes; pre-commit
  clean; CHANGELOG entry; PR #13 opened.
- **The three spec documents re-submitted to adversarial review**
  (the F1 blocker's demand -- the drafting review ran before the
  files existed; the re-review diffs against the plan.md findings
  appendix).
- The open questions of plan section 9 answered or their
  provisional resolutions recorded.
- Every `MTP` placeholder replaced by a dated measured value in
  "Recorded results"; the D2 conjugation oracle, MasterXcorr
  parity, and the D9.5 rescue (or its recorded fallback) all green
  locally with binaries present.
- Coverage 100 % of `_pseudo_symmetry.py` and the touched spherical
  modules; `.py_func` coverage for any new kernel.
- The four public names exported, documented, and in the API
  reference; no roadmap phase number in any public docstring.
- Tutorial section landed per the 9.8 sequencing (or the follow-up
  recorded in the PR).

## Recorded results

### 2026-09-06/07 (drafting)

No measurements: a ~1h compute job owned the working tree and the
hard constraint for this stage was read-only analysis (requirements
D11). The D2 derivation and every EMSphInx behaviour cited were
verified by line reading of 60f3517 (and `feature/GPU` for the D1
wiring) only. The adversarial drafting review's 26 findings were
folded in at drafting (plan.md appendix). This section is filled at
the failing-tests gate (placeholder inventory), the implementation
gate (measurements + pins, with recipes), and the review gate
(re-measurements), each in its own dated subsection, per the Phase
10 pattern.

### 2026-09-07 (failing-tests gate, Stage A)

Measurements taken before any Phase 8 implementation exists, from
pure geometry (merged `_euler`/`_wigner`/`_xcorr` code) and from the
shipped `MasterXcorr.exe` (60f3517, `build/Release`, machine-wide
program lock held, isolated temp CWDs).  Scripts:
`phase8_geometry_measurements.py` and `phase8_masterxcorr_runs.py`
(session scratchpad; commands `uv run python <script>`).  Pinned
into the failing tests where noted; everything needing
`find_pseudo_symmetry_operators` or the indexing loop stays a
`MEASURED-THEN-PINNED` FIXME-pin placeholder.

1. **D2 seed-chain identity, machine-verified** (was line-verified
   only, D11): over 2000 random `(zyz, op)` pairs,
   `rotation_to_zyz(op * rotation_from_zyz(zyz))` equals the literal
   C++ chain `qu2zyz(zyz2qu(zyz) * q_file)` with `q_file = (~op).data`
   to a worst deviation of **8.882e-16 rad** (quaternion components
   2.22e-16).  The spec's 1e-14 pin stands with ~11x margin; pinned
   in `test_seed_chain_matches_cpp`.
2. **D2.6a golden literal**: `op = from_axes_angles([1, 2, 3],
   25 deg)` gives `(~op).data =` `0.9762960071199334
   -0.057845920020143056 -0.11569184004028611 -0.17353776006042917`
   (`op * op` = 50 deg, non-involutory confirmed).  Frozen file body
   `"qu\n1\n<row>\n"` pinned bytes-exact in
   `test_write_golden_bytes`.
3. **Two-phase direction, machine-verified** (D2.6c/D3.9): at bw 24
   on a random real spectrum, the peak of
   `correlate(f, rotate_harmonics(f, zyz_r))` with
   `zyz_r = (0.9, 0.7, -0.4)` lands on
   `rotation_from_zyz(zyz_r)` to 0.000000 deg while its inverse is
   97.884642 deg away.  Hence for `h2 = h.rotate(S)` the returned
   operator is exactly `~S` (equivalently: the phase-1 equivalent of
   a phase-2 orientation is `op * O_2`, consistent with the Phase 6
   composed-orientation identity).  Exact-direction assertions
   pinned in `test_two_phase_rotated_copy`.
4. **MasterXcorr.exe runs, bw 88, EMsoft `ni_mc_mp_20kv.h5`
   (cached), each ~1 s** (`MasterXcorr 88 <cutoff> <ni.h5>`):
   - Single-file (auto) mode, cutoff 0.9: `maximum intensity:
     0.719309`; **22 printed rows** -- 7 at intensity 1.3521 (the
     three 180-degree axis rotations, four 90-degree z/x-adjacent
     ops as printed) and 15 at 1.2082 (the eight 120-degree <111>
     ops, remaining 90-degree and 180-degree <110> ops).  Cutoff
     0.5 adds one row: `0.8653  0.000000 0.012636 -0.703813
     0.710273` (a displaced near-C2' at the glide edge); 23 local
     maxima extracted in both runs.
   - **Two findings that REFUTE drafted D9.1 expectations** (D11
     correction path; requirements.md amendment left to the
     orchestrator since this stage only appends here): (a) the
     **identity peak is NOT in the printed list** -- its cell sits
     at the stored beta edge (beta = -pi/175) and the identity-cell
     seeded reference refinement stalls at 0.719309 = 74 % of the
     true peak value (0.972597); (b) consequently the printed
     **intensities sit ABOVE one** (1.3521/1.2082), not "~1.0", and
     the "identity peak intensity == 1.0" pin is unsatisfiable as
     drafted.  The kikuchipy port reproduces the same seeds (D3.3),
     so the parity pins in the failing tests use the measured
     22-row/1.3521/1.2082/0.719309 values instead; the v_max mutant
     of plan 7.2 still dies (an argmax-seeded v_max rescales every
     intensity to 1.0000/0.8936, far outside the 5 % band).
   - **D8.2i correction, measured**: `masterFile1 == masterFile2`
     is a FILENAME STRING comparison (`master_xcorr.cpp:87`), so
     "the same ni h5 passed TWICE" (same spelling) does NOT
     exercise the two-file branch -- output byte-identical to the
     single-file run.  Passing the same file under two path
     SPELLINGS (backslash vs forward slash) does: `maximum
     intensity: 0.972597` (coarse-argmax seeding), intensities
     1.0000 (7 rows), 0.8936 (15 rows), 0.6400 (edge row, cutoff
     0.5).  `test_two_file_branch_same_master` keeps its planned
     name and uses the two-spellings route; both v_max values are
     pinned at rel 0.05.
   - Stdout format facts of D2.7 confirmed: intensity at fixed
     precision 4, quaternion via `Quat::to_string(6)` with a
     leading alignment space per non-negative component (double
     spaces present); the four hard-coded CWD outputs
     (`pseudo_sym.h5` 21.6 MB, `.xdmf`, two SVGs) confirmed
     written, so every exe run keeps its isolated CWD.
5. **Placeholder inventory (FIXME-pin markers in the committed
   failing tests)**: `NI_OPS_COUNT`/`NI_OH_ANGLE_TOL_DEG`/
   `NI_TOP_INTENSITY` (pure-Python .sht/bw-68 route),
   `BLEND_ANGLE_TOL_DEG`/`BLEND_INTENSITY_BOUNDS`, the off-grid
   0.95-factor discriminating cutoff, `MASTERXCORR_INTENSITY_RTOL`,
   `TRUE_OP_TIE_RTOL`, `RANKED_VARIANT_TOL_DEG`,
   `DUPLICATE_ROW_TOL_DEG`, `KILLER_WINNER_TOL_DEG`, and every
   `RESCUE_*` lever/tolerance of the D9.5 scenario (deterministic
   construction still to be established, fallback recorded in the
   test docstring).
6. **Stage A test-design deviations, recorded for the review**:
   (a) `test_dedup_three_deg_pair` is implemented at unit level on
   `_dedup_keep_brighter` with BOTH discriminating geometries (3
   and 6 degrees of misorientation, i.e. 1.5 and 3 degrees of
   quaternion-dot half-angle) -- validation.md's "3 deg apart in
   quaternion-dot space ... one kept" is internally inconsistent
   with D3.5's frozen metric (at 3 deg of half-angle nothing
   merges); the end-to-end blend construction is deferred to the
   implementation-gate measurement.  (b)
   `test_spherical_indexing_perturbation_oracle` asserts index 0
   everywhere with unmoved winners: the NCC 2/1/0 scheme perturbs a
   STARTING map, which from-scratch global indexing does not have;
   the exact NCC indices live in
   `test_conjugation_killer_via_psymfile` and the nonzero spherical
   winner-index path in `test_rescue_scenario` (D9.5).  (c) The
   D9.6 local-master candidate filenames are provisional
   placeholders pending open question 9.5.
7. **Gated binary tests first-executed (Stage A bonus; they need no
   Phase 8 kikuchipy code)**, `KIKUCHIPY_EMSPHINX_DIR` set to the
   local checkout, machine-wide lock via the `emsphinx_program`
   fixture, command `KIKUCHIPY_EMSPHINX_DIR=... uv run pytest
   tests/test_indexing/test_spherical_pseudo_symmetry.py::TestIndexEBSDPsymFile
   -n 0 -q`:
   - **All four `TestIndexEBSDPsymFile` tests PASS (1.2 s)** -- the
     D1 inertness baseline is now executable-verified, not only
     line-verified: one-op psymfile run exits 0 and writes `Scan 1`
     + `Scan 2` (unpadded names); `Scan 1` is data-set-level equal
     to the no-psymfile run excluding the `EMheader`
     StartTime/StopTime/PatPerS fields (image maps never compared);
     `Scan 2/EBSD/Data` carries Phase = 255 (uint8), Metric = 0,
     IQ = 0 and a Phi1/Phi/Phi2 triple whose rotation is the
     identity to < 1e-4 deg -- exactly the corrected D1 ground
     truth.  The eu-type, count-mismatch and two-master+psymfile
     error paths all exit non-zero.  These gated tests pass at the
     failing-tests stage BY DESIGN: they pin the shipped binary,
     not Phase 8 code.
   - `TestMasterXcorrParity::test_two_file_branch_same_master` run
     gated: the exe leg (two path spellings, argmax-seeded
     `maximum intensity` 0.972597, top row 1.0000) passes its
     assertions and the test then fails on
     `find_pseudo_symmetry_operators`'s `NotImplementedError` -- the
     right reason -- validating the stdout parser and both vMax
     pins against the real binary.
8. **Whole-suite health at the failing-tests stage**: `uv run
   pytest tests/test_indexing tests/test_signals -k "spherical"
   -n 4 -q` gives **52 failed, 3063 passed, 738 skipped in 139 s**;
   the 52 failures are exactly the new Phase 8 tests (51
   `NotImplementedError`, 1 the `(9, 2, 6) != (9, 2, 7)` row-width
   assertion), and full-repo collection is clean (4822 tests).  The
   8 new tests that PASS at this stage are deliberate pins of
   state the skeleton already establishes: the four export names +
   sorted `__all__` + docstring hygiene, the refine-rows-stay-6
   split-constant pin, and the two amended signature pins
   (`pseudo_symmetry_ops: None` in both frozen-defaults dicts, plus
   the after-`emsphinx_compatible` placement pin).

### 2026-09-07 (failing-tests gate, Stage A -- test-critic fixes)

The Stage A test-critic review returned 3 majors, 5 minors and 3
nits against the committed failing tests.  Dispositions (applied
unless stated), with the re-run evidence at the end:

1. **Major, slP-vs-sl scan mutant unkillable (APPLIED)**: every
   drafted bandwidth satisfied `fast_size(2*bw-1) == 2*bw-1`, so
   the D3.4 recorded deviation (scan the true `(bwP, slP, slP)`
   cube) had no killer despite plan 7.2 naming one.  Added
   `NON_COINCIDENT_BANDWIDTH = 64` (127 prime, `fast_size` 128) and
   two tests: `test_local_maxima_finds_a_planted_off_fast_grid_peak`
   (unit, synthetic `(65, 128, 128)` cube, planted interior peak
   pinned by exact flat index, a super-threshold non-maximum
   shoulder rejected, `emsphinx_compatible` threaded both ways) and
   `test_volume_shape_at_a_non_coincident_bandwidth` (the volume
   shape where it discriminates).
2. **Major, D3 step-1 "caller's objects never modified" uncovered
   (APPLIED)**: added `test_callers_harmonics_are_never_modified`
   (auto and two-phase modes; `alm` snapshot compared
   `np.array_equal` after the call) -- kills the removeDC-in-place
   mutant, which idempotence hides from every value test, and
   protects the suite's lru-cached shared harmonics.
3. **Major, hollow "NOT inversion closed" claim on the ni/al weekly
   run (APPLIED in the test file; wording here is the
   orchestrator's)**: Ni and Al are both fcc m-3m in the same
   EMsoft setting, so the cross-master peak set is (near-)identity
   composed with proper Oh -- inversion closed and
   conjugation-blind like the Ni autocorrelation parity.
   `test_two_master_parity_al`'s docstring now states this honestly:
   NO binary run anywhere compares kikuchipy's quaternion DIRECTION;
   that burden rests wholly on the pure-Python D2.6c rotated-copy
   oracle (machine-verified, inverse 97.9 deg away) plus the D2
   derivation.  **Recorded oracle gap**: a systematic error in the
   D2 derivation itself would leave kikuchipy self-consistent yet
   divergent from EMSphInx undetected.  The critic's candidate
   closer -- MasterXcorr on ni.h5 vs a rotated-ni EMsoft h5 written
   by the test -- is REJECTED for this phase: it needs harmonic
   synthesis onto the EMsoft Lambert grid plus an EMsoft-h5 writer
   the binary accepts, machinery outside Phase 8 scope; revisit if
   the D8.5 manual `feature/GPU` rebuild oracle is run, which also
   closes the direction question.  This item must reach the
   orchestrator's amendment pass (the automated/weekly sections
   above still carry the refuted wording).
4. **Minor, data-dependent tie pin (APPLIED)**: added
   `test_a_forced_exact_tie_keeps_the_base_first` (indexer suite):
   monkeypatched `correlate` AND `refine_zyz` return one fixed
   score, so base and variant tie exactly on every pattern and the
   `upper_bound` strictly-beats rule must keep the base row first
   with index 0 -- the plan-7.2 tie-rule-inversion killer made
   deterministic, in the file's established monkeypatch style.
   `test_true_op_ties_base` keeps its measured-tie clause but its
   docstring now names it data-dependent.
5. **Minor, rescue fixture intensity scale (APPLIED)**:
   `test_rescue_scenario` now calls `get_patterns(...,
   dtype_out=np.uint8)`, rescaling synthesis to [0, 255] so the
   sigma-40 noise and the `clip(0, 255).astype(uint8)` perturb
   rather than destroy the signal -- a structural fixture bug that
   risked the implementation-gate measurement concluding "no
   deterministic construction survives" for a fixable reason.
6. **Minor, error paths pinned returncode only (APPLIED)**: the
   three gated rejection tests now also pin the thrown message in
   stdout+stderr (`index_ebsd.cpp:193` prints `e.what()` to
   stdout): "only quaternion angle files are supported", "not
   enough orientions in angle file" (the binary's own typo), and
   "psuedo-symmetry files currently only supported for single phase
   indexing" -- fail-for-the-right-reason by construction,
   re-verified against the shipped exe (all four gated tests pass,
   0.8 s).
7. **Minor, dedup keep-the-last-seen mutant (APPLIED)**:
   `test_dedup_three_deg_pair` gained the mirrored brighter-FIRST
   geometry (intensities [0.9, 0.7] -> keep [True, False]).
8. **Minor, no frozen-defaults pin on `find_` (APPLIED)**: added
   `test_signature_defaults_are_the_d3_frozen_ones` (defaults dict
   per the suite's freeze convention, `cutoff` deliberately
   excluded per open question 9.4, keyword-only tail pinned).  It
   passes at this stage (a signature pin on the skeleton), raising
   the deliberate Stage A passes from 8 to 9.  The critic's
   observation that no test toggles `emsphinx_compatible=False` on
   `find_` is PARTIALLY addressed: the new `_local_maxima` unit
   test threads the flag both ways (interior peak, semantics agree);
   an edge-glide discriminating case remains an implementation-gate
   review item.
9. **Nit, misleading test name (APPLIED)**:
   `test_ni_identity_plus_oh_count` renamed
   `test_ni_proper_oh_count` to match the measured 22-row,
   identity-absent reality.  The automated section above and the
   D9.1 wording still carry the drafted name/phrasing -- orchestrator
   amendment pending.
10. **Nit, stale collection count (APPLIED as a correction here)**:
    the Stage A record's "4822 tests" was stale; the critic
    measured 4826 on the same tree and the post-fix tree collects
    **4831 tests, cleanly** (4826 + the 5 new tests).
11. **Nit, blanket `except Exception: skip` in `_full_ni_h5_path`
    (APPLIED)**: the `Dataset` is now constructed OUTSIDE the try
    (a registry typo raises `KeyError` loudly) and the except is
    narrowed to `(ImportError, ValueError)`; the Al fetch in the
    weekly test got the same treatment.  Of the two kin: the
    MasterXcorr parity nearest-matching is now asserted to be a
    BIJECTION in both parity tests (a duplicated match can no
    longer mask a missing peak under count parity); the absence of
    a direct find_-side no-post-refinement-dedup test (D3.6) is
    ACCEPTED as a recorded review item -- the indexer-side
    `test_duplicate_variant_rows_survive` pin and the D8.1 count
    parity (which fails if a second dedup collapses the two
    surviving refined duplicates the binary prints) carry it until
    the implementation-gate review.

**Re-run evidence (2026-09-07, post-fix)**: the four Phase 8 files
(`test_spherical_indexer.py`, `test_spherical_pseudo_symmetry.py`,
`test_ebsd_spherical_indexing.py`,
`test_spherical_master_pattern_harmonics.py`) give **56 failed, 323
passed, 17 skipped in 17 s** at `-n 4` (`-n 0` runs of the two
touched files first): the 56 = the prior 52 + the 4 new
right-reason failures, all 55 `NotImplementedError` plus the one
expected `(9, 2, 6) != (9, 2, 7)` row-width assertion; the gated
`TestIndexEBSDPsymFile` passes 4/4 against the shipped exe with the
new message pins; full-repo collection is clean at **4831 tests**.
