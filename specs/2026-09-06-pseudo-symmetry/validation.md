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

- `test_ni_proper_oh_count` (the D9.1 positive-count pin,
  `exclude_symmetry=False`, stated cutoff, in-package Ni `.sht` at
  bw 68 or 88; AMENDED 2026-09-07, renamed from
  `test_ni_identity_plus_oh_count`): the return holds the MEASURED
  22-row proper-Oh subset -- the identity is NOT in the list (its
  cell sits at the stored beta edge and the identity-seeded refine
  stalls, see Recorded results) and one folded C2' is absorbed by
  dedup; each op within an MTP angular tolerance of a
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
  tech-stack.md:50; skips cleanly without pooch). **AMENDED
  2026-09-07 (fix-stage disposition 3)**: ni and al are both fcc
  m-3m in the same EMsoft setting, so this cross-master peak set is
  inversion closed and conjugation-blind -- even when run it does
  NOT compare kikuchipy's quaternion DIRECTION. That burden rests
  wholly on the pure-Python D2.6c rotated-copy oracle plus the D2
  derivation, and the recorded oracle gap (a systematic error in
  the D2 derivation would leave kikuchipy self-consistent yet
  divergent, undetected by any binary run) stands regardless of
  open question 9.7. **Recorded-gap alternative if 9.7 rejects the
  download**: the two-file branch is then covered only by the
  two-spellings discriminator (the drafted same-file-twice run was
  refuted as vacuous, see Recorded results) plus the rotated-copy
  oracle, recorded here as a limitation.
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

### 2026-09-07 (implementation gate, module 3: `MasterPatternHarmonics.rotate`)

D7 implemented in `_master_pattern_harmonics.py` on top of the
tested `_wigner.rotate_harmonics`: the derived angles are
`zyz = quaternion_to_zyz(rotation.data)`, i.e. the zyz whose
`Rotation(zyz_to_quaternion(zyz))` is the given rotation, so the
Wigner identity `g(n) = f((~R) n)` realises the active contract
directly (this is exactly the chain the plumbing pin
`test_rotate_matches_rotate_harmonics` freezes).  Flag
neutralization is by phase replacement: a phase with point group
`"1"` carrying over the source phase's name and structure (the
constructor derives `n_fold == 1` / `has_equatorial_mirror ==
False` from it, never warns on a no-symmetry claim, and `save()`
stays usable via the effective space group 1, i.e. lossless
packing); a source without a phase stays phaseless (already the
safe default).  Both phase branches are exercised by the committed
D7 tests (`_random_harmonics` phaseless; the Ni-from-file objects
carry space group 225).

- **MTP inventory for this module: none.**  All six D7 tests were
  committed at Stage A with fixed analytic tolerances (1e-12 on
  coefficients, 1e-8 against the scipy synthesis oracle) and no
  `MEASURED-THEN-PINNED` placeholder; nothing to fill.
- All six pass first-run: `uv run pytest
  tests/test_indexing/test_spherical_master_pattern_harmonics.py::TestRotate
  -n 0 -q` -> **6 passed in 0.15 s** (composition, synthesis
  direction incl. the flipped-direction bite check, identity no-op
  on the coefficients, round trip, flag neutralization for
  identity/z/non-z rotations, `rotate_harmonics` plumbing pin).
- Full harmonics file: `uv run pytest
  tests/test_indexing/test_spherical_master_pattern_harmonics.py
  -n 0 -q` -> **184 passed, 4 skipped** (2 weekly, 2 binary-gated).
- Shared-suite regression check: `uv run pytest tests/test_indexing
  -k "spherical and not pseudo" -n 4 -q` -> **2994 passed, 724
  skipped, 0 failed** -- no regression from the module edit (the
  only file touched is `_master_pattern_harmonics.py`).

### 2026-09-07 (implementation gate, module 2: `pseudo_symmetry_ops` indexing path)

Implemented per D4/D5: the `indexer.hpp:241-261` psym loop in
`_index_chunk` (base honours `refine`; every variant seeded by
`_variant_seed_zyz` == the literal C++ `qu2zyz(q0 * q_file)` chain
and ALWAYS Newton refined; same positive-score `upper_bound`
insertion; variants all seeded from the phase's base, never
chained), the `_ROW_WIDTH` split rewired (`_ROW_WIDTH_INDEX = 7`
on the whole indexing path incl. fill `(0,0,0,0,-1,0,-1)`,
`_ROW_WIDTH_REFINE = 6` untouched on `refine_patterns`), ctor
validation (flatten, size-0 None-equivalence, the frozen
single-phase ValueError), the results key `pseudo_symmetry_index`
(int32 `(n, n_best)`) present iff ops passed, `EBSD.
spherical_indexing` prop plumbing (1-D winner
`"pseudo_symmetry_index"`; 2-D `"nbest_pseudo_symmetry_index"`
beside `nbest_phase_id` at `n_best > 1`; masked points fill -1),
and the D4/D10/D11 doc amendments (provenance comment rewrite, the
frozen one-candidate replacement wording, `P*(1+N)` fill sentence,
metric-inconsistency + duplicate-rows + hazard notes, refined-ratio
caveat, chunksize guidance).  One implementation choice recorded: a
`pseudo_symmetry_ops` construction always builds and shares the
Wigner d factor triple (as a `refine=True` construction does),
since variants always refine -- without it every chunk clone would
lazily rebuild its own 5 MB triple.

**MTP fills, measured** (script `phase8_indexer_measurements.py`,
session scratchpad; command `uv run python
phase8_indexer_measurements.py`; warm numba caches, best of 3 for
timings; measurements taken with modules 1 and 3 already present in
the tree):

1. `TRUE_OP_TIE_RTOL`: max relative gap of the true-op (90 deg z)
   variant score to the base score over the nine `nickel_ebsd_small`
   patterns = **4.65e-5** (per-pattern 3.4e-6 .. 4.7e-5, one at
   3.4e-11); pinned **1e-4** (~2.2x).  Recorded: the variant
   epsilon-BEATS the base on all nine patterns (winner index 1
   everywhere, allowed by the `{0, i}` clause); no exact bitwise tie
   occurred, so the data-dependent exact-tie clause is vacuous as
   its docstring predicts and the deterministic forced-tie test
   carries the tie rule.
2. `RANKED_VARIANT_TOL_DEG`: measured **0.0 deg exactly** (ops 90 /
   180 deg about z on the 4-fold Ni master shift only gamma, so the
   variant peak is the base peak exactly; the acos resolution floor
   hides anything below ~1e-8 deg); pinned **0.05 deg**, ~2x the
   bw-68 Newton stop scale `eps*pi/bw` = 0.027 deg.  Ranked run
   indices came out `[2, 1, 0, -1]` -- both true-op variants
   epsilon-beat the base, scores 0.62724045 / 0.62724045 /
   0.62722812 / 0 descending, fill row exact.
3. `DUPLICATE_ROW_TOL_DEG`: measured **0.0 deg** (91 deg z variant
   Newton-walks 1 deg back into the 90 deg peak); pinned **0.05
   deg** (same scale rationale).  The drafted inline score-equality
   tolerance `rel=1e-6` was REFUTED by measurement: the two
   duplicate rows' scores differ by rel **1.383e-6** (the Newton
   stop's second-order lag); replaced by the named constant
   `DUPLICATE_ROW_SCORE_RTOL = 3e-6` (~2.2x) -- an MTP fill of a
   tolerance the drafted test carried unmarked (D11's blanket
   "no tolerance below is measured yet" covers it).
4. `KILLER_WINNER_TOL_DEG`: cross-engine winner misorientation at
   the perturbed point = **0.465 deg** under m-3m (all nine points
   0.42-0.70 -- the engines' shared systematic residual scale);
   pinned **1.0 deg** (~2.2x).  NCC recovered index `[1, 0, ..., 0]`
   exactly; the spherical from-scratch run reports all zeros with
   the codec-read operator, and `test_conjugation_killer_via_
   psymfile` passes end-to-end through the module-1 codec.

**D9.5 rescue scenario -- drafted construction refuted, deterministic
construction established** (scan script `phase8_rescue_scan.py`,
session scratchpad; command `uv run python phase8_rescue_scan.py`):

- The drafted single-copy +z blend `f + lam f.rotate(S)` CANNOT
  land the coarse search in the pseudo basin, for two measured and
  derivable reasons: (a) its base peak weight `1 + lam^2` exceeds
  the single variant peak `lam` by ~2x at any `lam <= 1`, far
  beyond discretization noise; (b) the Ni function's EXACT 4-fold z
  symmetry makes every z-blend degenerate under `Rz(90 k)` and
  populates Oh ghost families at weight ~1, which is where the
  drafted run's winner actually went (measured `off_pseudo` 59.9
  deg at the drafted levers -- neither true nor pseudo).
- The established construction is the 3-fold two-copy blend about a
  general axis, `g = f + lam (f.rotate(S) + f.rotate(~S))` with
  `S` = 120 deg about `[1, 2, 3]`: true peak `1 + 2 lam^2`, pseudo
  peaks `2 lam + lam^2`, gap `(1 - lam)^2` = 0.25 % at lam 0.95 --
  small enough for the bw-53 coarse grid to decide the basin, large
  enough for the always-refined variants' analytic comparison to
  recover the true peak; Oh ghosts stay at weight ~lam, well
  separated.  All other drafted levers survive unchanged:
  `RESCUE_SEED = 8`, `RESCUE_BANDWIDTH = 53`, `RESCUE_WEIGHT =
  0.95`, `RESCUE_NOISE_SIGMA = 40`, true rotation 40 deg about
  `[1, 2, 1]`, uint8 synthesis + seeded noise + clip.
- Measured at the pinned levers: `off_true` **119.80 deg** (pin
  > 30 stands), `off_pseudo` **0.287 deg** (pinned
  `RESCUE_PSEUDO_TOL_DEG = 0.6`, ~2.1x), winner index **1**, score
  gain **+0.0016**, rescue error **0.309 deg** (pinned
  `RESCUE_TRUE_TOL_DEG = 0.65`, ~2.1x).  Robustness: the scan's
  full neighbourhood at true angles 37 and 40 deg x sigma
  {10, 25, 40} x seeds {0, 1, 8} passes 18/18 (at 43 deg the grid
  luck flips and the true basin wins coarse -- printed in the scan
  output), so the pinned point is not razor-edge.  The recorded
  fallback of D9.5 is NOT invoked.

**Stage A test correction, recorded**: the pre-Phase-8
`test_the_graph_metadata_is_truthful` (indexer suite) pinned the
lazy result shape `(9, 2, 6)`; the D4 packed-row amendment makes
width 7 the contract on the indexing path, so the pin is updated to
`(9, 2, 7)` with a comment naming this record -- the one edit to a
pre-existing test, provably outdated by the amended spec.

**Performance (D11 a-d), recorded** (bw 68, `nickel_ebsd_small`,
one chunk of 9, warm, best of 3; single worker):

- ops-scaling baseline: **56.84 / 54.02 / 51.44 / 46.01 pat/s** at
  0 / 1 / 2 / 4 ops (17.59 / 18.51 / 19.44 / 21.73 ms per pattern).
- per-op per-pattern increment: **0.92 / 0.92 / 1.04 ms** derived
  from the row above -- the Phase 7 warm `refine_zyz` scale
  (~0.65 ms m-3m) plus seed conversion and insertion overhead.
- hard floor `>= 2 pat/s/core` stays scoped to `ops=None` (the
  existing floor test is untouched); the 2-op run is re-measured at
  **51.44 pat/s**, far above the floor, recorded not floored.
- `BatchEstimate` ignores per-op cost -- recorded deviation;
  chunksize guidance added to the module doc and the
  `EBSD.spherical_indexing` Notes (D11 c).
- the refined-ratio claims (1.05-1.27x) now carry the
  "without pseudo-symmetry operators" caveat in the `refine`
  parameter doc (D11 d); the full-map 2-op timing entry remains a
  weekly item.

**Module-2 test evidence (2026-09-07)**: `uv run pytest
tests/test_indexing/test_spherical_indexer.py -n 0 -q` -> **88
passed** (all 12 Phase 8 indexing-loop tests incl. the rescue
green); `uv run pytest
tests/test_signals/test_ebsd_spherical_indexing.py -n 0 -q` -> **64
passed, 5 skipped** (weekly) incl. all 5 `TestPseudoSymmetryOps`
tests with the cross-engine conjugation killer.

**Module-2 coverage disposition (2026-09-07)**: the Stage A test
plan exercised the psym variant loop only through the normalized
correlators, leaving the `normalize=False` twin (the shared
un-normalised prototype's `refine_zyz` variant branch of
`_index_chunk`) uncovered against the 100 %-coverage gate.  One
focused test was ADDED at the implementation gate (no existing
assertion touched): `test_variants_on_the_un_normalised_path`
(indexer suite) -- true 90-deg-z op, `normalize=False`, `n_best=2`
on one Ni pattern; measured max relative variant-to-base score gap
over the nine patterns **4.79e-5** (probe
`phase8_unnormalized_probe.py`, session scratchpad), inside the
same `TRUE_OP_TIE_RTOL = 1e-4` pin, winner indices `[1, 0]` as on
the normalized path.  With it, `_indexer.py` measures **100.00 %**
(372 statements, 0 missed) under the four spherical suites
(`uv run --with pytest-cov pytest tests/test_indexing/
test_spherical_indexer.py tests/test_signals/
test_ebsd_spherical_indexing.py tests/test_indexing/
test_spherical_pseudo_symmetry.py tests/test_indexing/
test_spherical_refinement.py -n 4 -q
--cov=kikuchipy.indexing._spherical._indexer
--cov-report=term-missing` -> 361 passed, 19 skipped).

### 2026-09-07 (implementation gate, module 1: `_pseudo_symmetry.py`)

The `find_pseudo_symmetry_operators`/codec/carrier implementation
landed against the Stage A tests.  Probe scripts `probe_blend.py`,
`probe_blend64_ni68.py`, `probe_pins.py` (session scratchpad; each
`uv run python <script>`); test commands quoted per item.

1. **Ni mechanism pins measured on the .sht/bw-68 route** (`uv run
   pytest tests/test_indexing/test_spherical_pseudo_symmetry.py -n 0
   -q`): count **22** at cutoff 0.9 AND at cutoff 0.5 (agreeing with
   the binary's bw-88 h5 count; `NI_OPS_COUNT` unchanged), top
   intensity **1.495790** (`NI_TOP_INTENSITY` pinned 1.4958, rel
   0.05), smallest kept 0.948959, max angle to the nearest proper Oh
   rotation **1.9948 deg** -- one displaced near-C2' glide-edge row
   at intensity 0.949, the bulk far closer -- so the drafted
   `NI_OH_ANGLE_TOL_DEG = 1.0` placeholder was refuted and is pinned
   **2.5** (1.25x).  `exclude_symmetry=True` returns **0** operators
   at both cutoffs (the D3.7 empty pin).
2. **Blend fixture correction (test provably wrong against the
   D3.3-frozen seeding; `BLEND_BANDWIDTH` 53 -> 60)**: at bw 53
   (odd `slP` 105) the identity-cell-seeded reference refinement
   Newton-steps off the gimbal-degenerate stored-beta-edge cell
   (52, 26, 26) into a NEGATIVE stationary value -- measured
   `v_max = -0.256673` against a true cube max of 2.428933 on the
   3-fold blend -- poisoning every normalized intensity, so the
   D9.2 recovery assertions cannot hold at any cutoff there.  (The
   C++ would do the same: same seed, same Newton, same monotone
   step rule; its bw-88 Ni "stall at 74 %" is the benign face of
   the same fragility.)  At an EVEN `slP` the translated
   nearest-to-identity cell (D3.3) is exactly the identity, the
   seed sits on the autocorrelation's critical point, and the
   reference refine returns the true peak (measured
   `v_max == cube max` at bw 60 and 64).  Pinned fixture:
   `BLEND_BANDWIDTH = 60` (`fast_size(119) = 120`, even), which
   also makes the blends a second slP-vs-sl non-coincident regime
   (comment on `NON_COINCIDENT_BANDWIDTH` updated).
3. **Blend pins measured at bw 60, weight 0.7, cutoff 0.25**:
   3-fold and 6-fold operators recovered at **0.1748 deg** from S
   and from ~S, matched intensity **0.5039** (the D9.2 "expected
   ~0.5" narrative: `lam / (1 + lam^2) = 0.4698`); off-grid
   100.7-deg operator at **0.4221 deg**.  Pins:
   `BLEND_ANGLE_TOL_DEG = 0.85` (~2x the worst),
   `BLEND_INTENSITY_BOUNDS = (0.48, 0.53)` (rel 0.05).
4. **Off-grid 0.95-factor discriminator measured** (D3.4): the
   off-grid operator's brightest grid voxel reads **0.4800** of
   `v_max`, its refined intensity **0.5510**, so
   `OFF_GRID_CUTOFF = 0.49` is pinned inside the discriminating
   window (with the factor the candidate gate is 0.4655 <= 0.4800;
   without it 0.49 > 0.4800 and the operator is lost);
   `test_off_grid_operator_found` now passes that cutoff instead of
   `BLEND_CUTOFF`.
5. **Two-phase rotated-copy oracle at bw 60**: operator equals ~S
   to **0.0000 deg**, the wrong direction 24.4808 deg away
   (`test_two_phase_rotated_copy` passing; direction burden of the
   Stage A honesty note discharged in-suite).
6. **Gated binary evidence** (`KIKUCHIPY_EMSPHINX_DIR=
   C:/Users/westraadt.1/Repos/EMSphInx uv run pytest
   tests/test_indexing/test_spherical_pseudo_symmetry.py --weekly
   -n 0 -q` -> **49 passed, 2 skipped**): `TestIndexEBSDPsymFile`
   4/4 again; `test_two_file_branch_same_master` now passes
   END-TO-END (exe leg 0.972597 argmax-seeded v_max as pinned;
   kikuchipy two-file leg top intensity 1.0 and count parity);
   `test_masterxcorr_stdout_parity` (weekly) **PASSES**: 22-row
   count parity at bw 88/cutoff 0.9, kikuchipy operators equal the
   CONJUGATE of the printed rows at 2e-6 under a bijective match,
   intensities inside `MASTERXCORR_INTENSITY_RTOL = 0.05` (pinned
   as measured-sufficient), exe v_max 0.719309 -- the faithful port
   reproduces the identity-seed stall on the same route.  The Al
   two-master weekly run still skips cleanly (download pending open
   question 9.7).
7. **Coverage**: `uv run --with coverage python -m coverage run -m
   pytest tests/test_indexing/test_spherical_pseudo_symmetry.py
   -n 0 -q` then `coverage report --include=*_pseudo_symmetry* -m`
   -> **100.00 % (166 statements, 0 missed)** from the default
   suite alone (43 passed, 8 skipped without the gate).  Whole
   spherical selection after the module landed: `uv run pytest
   tests/test_indexing tests/test_signals -k "spherical" -n 4 -q`
   -> **3121 passed, 738 skipped** (zero failures; ruff check +
   format clean at the pinned 0.15.15).
8. **Implementation dispositions, module 1** (each documented in
   the module): (a) zero surviving local maxima yield an EMPTY
   result where the C++ CLI exits with an error -- recorded
   library-vs-CLI deviation in the `find_` Notes; (b) the
   `PseudoSymmetryOperators.intensities` and `cutoff` docstrings
   were corrected from the drafted "identity peak == 1.0" wording
   per the Stage A D9.1 amendment (values above one are the
   faithful expectation); (c) `_exclude_true_ops` realises the two
   0.999 filters as ONE growing exclusion set seeded with the
   proper rotations -- outcome-identical to the C++ pair under the
   D3.7 kept-versus-kept reading, and it keeps the line-coverage
   gate honest; (d) the reader lets Python's own `ValueError`
   surface for a missing/non-integer count or a non-numeric row
   entry (messages not pinned); a negative count is caught by the
   count-mismatch check; (e) the auto-mode identity cell is
   computed as `euler_to_index((0, 0, 0), slP)`, verified equal to
   the C++ `idxIdent` cell at every coincident bandwidth used
   (52, 26, 26 at bw 53; 67, 34, 34 at bw 68) and exactly the
   identity at even `slP` (64, 32, 32 at bw 64; beta = 0).

### 2026-09-07 (adversarial review gate, conventions-and-quality)

Independent re-measurements and gate verifications on the assembled
tree (all three implementation modules present), per D11.3:

- **Suite re-runs (review)**: the four Phase 8 files together at
  `-n 4` -> 380 passed, 17 skipped; the module file serially
  (`-n 0`) -> 43 passed, 8 skipped; the full spherical selection
  (`uv run pytest tests/test_indexing tests/test_signals -k
  "spherical" -n 4 -q`) -> **3121 passed, 738 skipped, 0 failed**
  (50 s) -- reproducing the implementation-gate counts exactly.
- **Gated binary re-runs (review)**: with `KIKUCHIPY_EMSPHINX_DIR`
  set, `TestIndexEBSDPsymFile` (4/4) plus
  `test_two_file_branch_same_master` -> 5 passed (3.4 s), and the
  weekly `test_masterxcorr_stdout_parity` -> 1 passed (2.6 s): the
  D2 conjugate-of-printed-rows parity at 2e-6, the 22-row count
  parity and both pinned v_max values (0.719309 auto / 0.972597
  two-file) all hold on re-execution.  The D9.5 rescue and the
  D2.6b cross-engine killer re-run green serially (17 psym
  indexing/signal tests, 3.5 s).
- **Coverage (review)**: `_pseudo_symmetry.py` **100.00 %** (166
  statements) under the default module suite;
  `_indexer.py` **100.00 %** (372 statements) under the recorded
  four-suite combination (indexer + ebsd_spherical_indexing +
  pseudo_symmetry + spherical_refinement, `-n 4`).  Every Phase 8
  line of `_master_pattern_harmonics.py` (`rotate`, lines
  2123-2160) and of `ebsd.py` (psym plumbing, prop assembly) is
  covered; the residual misses in those files under the four-suite
  selection are pre-existing branches owned by other suites.
- **Conventions verified clean**: GPL + CMU/Lenthe header matches
  the `_spherical` family (ranges, Changed-by, GPL-2.0-or-later
  conveyed under GPL-3.0-or-later); `__init__.pyi` carries the four
  names sorted; `doc/reference/generated/` is untracked build
  output, so the API reference genuinely regenerates from
  `__all__`; the D5 frozen message is byte-exact; the bandwidth
  message reuses `_indexer`'s wording; ruff check + format clean on
  all seven changed files; prop dtypes int32 and the D4
  `n_best`/dual-prop layout as frozen; ctor keyword-only placement
  after `refine` and the signal-method placement after
  `emsphinx_compatible` both pinned by passing tests; no new numba
  kernel, so the kernel-flag conventions are untouched.
- **Review findings passed to the fix stage** (dispositions):
  (a) the required CHANGELOG `Added` entry (D10) is absent from
  the tree -- it must land with the implementation commit even
  while the tutorial waits on open question 9.8; (b) the
  `MASTERXCORR_INTENSITY_RTOL` comments in
  `test_spherical_pseudo_symmetry.py` (near the constant and at
  its parity use) still carry the Stage A "FIXME-pin ... awaiting
  the kikuchipy side" wording although the pin was validated at
  the implementation gate and re-validated here -- stale markers
  to be replaced with the dated record; (c) the
  `write_emsphinx_psym_file` Notes' round-trip parenthetical is
  garbled and claims the reader's identity skip "cannot trigger",
  which is false (the writer refuses only an empty set; an
  exact-identity operator is written as given and the reader then
  skips it) -- to be reworded to D6's "exactly, modulo the
  identity-skip rule"; (d) the previously recorded orchestrator
  amendments (requirements D9.1 "identity plus ... 24" count
  phrasing; this file's automated section naming
  `test_ni_identity_plus_oh_count`) remain outstanding.

### 2026-09-07 (adversarial review gate, fidelity)

Line-by-line refutation pass against `master_xcorr.cpp`,
`indexer.hpp:207-345`, `master.hpp:220-233`, `emsoft.hpp:48-145`,
`sht_xcorr.hpp` (correlator ctor, `correlate`, `interpPeak`,
`refinePeak`, `extractNeighborhood`, `eulerIndex`, `indexEuler`,
`findPeak`) and `rotations.hpp`/`quaternion.hpp` (`zyz2qu`,
`qu2zyz`, `quat::mul`, `Rotation operator>>`), re-read at 60f3517
for this review.  Scripts `review_d2_verification.py`,
`review_exclude_divergence.py`, `review_blend_vmax.py` (session
scratchpad, each `uv run python <script>`).

1. **D2 conjugation independently re-verified** (D11.3): the C++
   chain was re-transcribed from the sources into an independent
   script (its own `zyz2qu`/`qu2zyz`/Hamilton `mul` with
   ``pijk = +1``, no kikuchipy `_euler` reuse).  Over 500 random
   `(zyz, op)` pairs: `zyz_to_quaternion` equals `zyz2qu` bitwise
   (worst 0.0); the literal seed chain
   `qu2zyz(zyz2qu(zyz) * (~op).data)` equals `_variant_seed_zyz`
   to a worst wrapped deviation of **8.882e-16 rad**, and the
   realised variant map rotation equals `op * rotation_from_zyz
   (zyz)` to 0.0 deg.  The concrete non-involutory walk-through
   (op = 25 deg about [1, 2, 3], `op * op` = 50 deg) reproduces
   the frozen golden psymfile row digit-for-digit, and the closure
   `save()`-row == MasterXcorr-printed `zyz2qu(zyz_peak)` holds to
   2.2e-16 over 200 random peaks (i.e. a user pasting exe stdout
   rows into a psymfile gets exactly kikuchipy's operators back).
2. **BLEND_BANDWIDTH 53 -> 60 correction re-measured** (D11.3):
   independently reproduced digit-for-digit -- bw 53 (slP 105,
   identity cell (52, 26, 26) = the C++ `idxIdent` cell,
   coincident) refines to **v_max = -0.256673** against a cube max
   of 2.428933 on the 3-fold z blend; bw 60 (slP 120, identity
   cell (60, 30, 30), seed exactly (0, 0, 0)) returns **v_max =
   2.827391 = the cube max** exactly.  The provably-wrong-test
   invocation stands: at bw 53 the C++ would seed the same cell
   into the same Newton failure.
3. **Identity-cell translation verified against the C++**:
   `euler_to_index((0,0,0), slP)` reproduces `idxIdent =
   (bw-1, bw/2, bw/2)` at every coincident bandwidth checked (52,
   26, 26 at 53; 67, 34, 34 at 68; 87, 44, 44 at 88, beta
   -pi/175), the D3.3 translation.
4. **Gated binaries re-run by this review**:
   `KIKUCHIPY_EMSPHINX_DIR=... uv run pytest
   tests/test_indexing/test_spherical_pseudo_symmetry.py --weekly
   -n 0 -q` -> **49 passed, 2 skipped** (local-masters dormant, Al
   download pending 9.7), including `TestIndexEBSDPsymFile` 4/4,
   `test_masterxcorr_stdout_parity` and
   `test_two_file_branch_same_master`.  Full spherical selection
   re-run: 3121 passed, 738 skipped, 0 failed; ruff clean.
5. **Stage-A-vs-tree test diff audited**: every changed value is a
   marked MTP fill with its measurement recorded above
   (tightened: TRUE_OP_TIE_RTOL, RANKED/DUPLICATE tolerances,
   KILLER/RESCUE tolerances; loosened with recorded measurements
   refuting the drafted placeholders: NI_OH_ANGLE_TOL_DEG 1.0 ->
   2.5 at measured 1.9948, the unmarked duplicate-score rel 1e-6
   -> named 3e-6 at measured 1.383e-6); the one pre-existing-test
   edit is the recorded `(9, 2, 6) -> (9, 2, 7)` correction; no
   assertion was weakened without a recorded refuting measurement.
6. **FIDELITY DIVERGENCE FOUND AND RECORDED (exclusion filter,
   corner geometry)**: `_exclude_true_ops` deduplicates a
   candidate against the proper rotations plus the already-KEPT
   operators (the D3.7 "kept-versus-kept" reading the spec
   froze), but the literal C++ SVG-stage duplicate check
   (`master_xcorr.cpp:255-261`, `for j < i`) runs against **all
   earlier above-cutoff maxima including those already skipped as
   true-symmetry matches**.  Demonstrated concretely
   (`review_exclude_divergence.py`, point group 4): A = Rz(94 deg)
   (dot cos(2 deg) = 0.99939 > 0.999 to Rz(90), dropped by both),
   B = Rz(98 deg) (dot 0.99756 to Rz(90), dot 0.99939 to A) --
   the literal C++ drops B against the excluded A, kikuchipy keeps
   B.  Disposition: **no code change** -- the spec's frozen D3.7
   reading is the binding contract, `exclude_symmetry=True` is
   already a recorded kikuchipy-native deviation (the C++ never
   filters stdout, only `pseudo.svg`), and the divergent geometry
   needs a candidate within the 0.999 cosine (~2.56 deg
   half-angle) of an excluded-but-not-kept earlier candidate while
   itself clearing every true operator -- but the reading is now
   recorded as a deviation from the SVG-stage C++ pair, not an
   equivalence.  The fix stage may add one sentence to the
   `_exclude_true_ops` Notes naming it.
7. **Codec header-tokenisation corner case, recorded**: the C++
   type token is read as exactly two characters BEFORE the
   comma-as-whitespace locale is imbued (`rotations.hpp:1213-1215`,
   `emsoft.hpp:130-136`), so the binary rejects `"qu,2"` (the
   count read fails on the comma) yet accepts `"q u"` (whitespace
   between the two characters), while the reader here tokenises
   commas-as-whitespace globally and does the opposite in both
   corners.  The D6 grammar ("first token `qu`") is what the port
   implements; unpinned, record-only.
8. Verdict on the mandated checks: cube construction/fast-size
   discipline, both v_max seeding paths (the exe's measured
   identity-seed stall reproduced, not papered over -- the
   0.719309 pin re-verified against the binary), the
   `v_max * cutoff * 0.95` chain, the true-cube neighbour scan in
   C++ scan order, the 2-deg half-angle keep-brighter dedup with
   in-place replacement and first-minimum nearest search, the
   unconditional survivor refinement from un-interpolated grid
   Eulers, the D2 boundary placement, the seed chain, the
   strictly-beats tie rule, the always-refine-variants asymmetry,
   and `rotate()`'s active direction are all **faithful**;
   preserved quirks (exact identity skip incl. -0.0 semantics,
   variants refined at `refine=False`, refine-failure seed
   fallback, normalised variants refined un-normalised then
   denominator-divided) verified in code and tests; excluded
   quirks (inert `pSym` wiring, `sl`-vs-`slP` scan, `fabs(nFld -
   nFld)` dead branch, the CLI clamp) verified NOT reproduced.

### 2026-09-07 (fix stage, post-review dispositions)

Fixes applied for the review findings (conventions, fidelity and
mutation reviews), with the re-run evidence at the end.  Probe
script `fixstage_parity_intensity_probe.py` (session scratchpad,
`uv run python <script>`; machine-wide program lock held for the
exe leg).

1. **CHANGELOG (major, D10) APPLIED**: the required `Added` entry
   now heads the Unreleased section of `CHANGELOG.rst`, naming
   `find_pseudo_symmetry_operators` (+ the
   `PseudoSymmetryOperators` carrier), the psymfile codec
   (`read_emsphinx_psym_file`/`write_emsphinx_psym_file`),
   `pseudo_symmetry_ops` on `SphericalIndexer` /
   `EBSD.spherical_indexing` (with the `pseudo_symmetry_index`
   prop), and `MasterPatternHarmonics.rotate`, with the PR #13
   link -- landing with the code as D10 requires, tutorial still
   sequenced behind open question 9.8.
2. **P11 survived mutant (major) KILLED**: added
   `test_local_maxima_agree_with_a_brute_force_reference`
   (module suite): 26 probe pairs planted one per neighbour offset
   on a sub-threshold random cube -- each probe (0.9) beaten ONLY
   by its single planted neighbour (1.0) -- asserted equal to an
   in-test brute-force 26-neighbour reference, all candidates
   interior, `emsphinx_compatible` threaded both ways.  Kill
   verified by re-injecting the P11 mutant
   (`neighborhood[:2]`, the k+1 plane never compared): the new
   test FAILS at the set-equality assertion under the mutant and
   passes on the restored tree.  By construction it kills every
   single-offset- and plane-dropping mutant of the scan
   comparison, so the review's alternative killer (pinning the raw
   bw-60 blend list at 100 rows) was not also added.
3. **Stale `MASTERXCORR_INTENSITY_RTOL` FIXME-pin markers (minor)
   APPLIED, with the missing measurement taken**: probe on the
   parity route (bw 88, cutoff 0.9, Ni h5, exe v_max 0.719309, 22
   bijectively matched rows) measured **max relative intensity
   deviation 2.852e-5** (mean 8.9e-6) -- the 0.05 band stands as
   measured-sufficient with ~1750x margin.  Both comments (at the
   constant and at the parity use) now carry the dated record; the
   band itself is deliberately NOT tightened, per the finding
   ("comment/record fix -- no assertion changes") and because the
   same constant guards the never-yet-executed Al two-master
   weekly route, whose deviation scale is unmeasured.
4. **`write_emsphinx_psym_file` garbled round-trip Notes (minor)
   APPLIED**: reworded to the D6 formulation -- read of the
   written file reproduces the operators exactly *modulo the
   reader's identity-skip rule*; only an EMPTY set is refused, an
   exact-identity operator is written as given and the reader then
   silently skips it.  The module docstring's "Round trip ...
   exactly" line gained the same qualifier.
5. **`_exclude_true_ops` SVG-stage divergence (minor,
   record-not-fix) APPLIED as documentation**: the Notes now name
   the second recorded deviation -- the dedup measures against
   KEPT operators only (the frozen D3.7 reading), where the
   literal C++ `for j < i` check also measures against earlier
   above-cutoff candidates already dropped as true-symmetry
   matches (fidelity item 6 above).  No behavior change; the
   "outcome-identical" phrasing of the module-1 record holds only
   under the D3.7 reading, as item 6 records.
6. **Codec header-tokenisation corner (nit) APPLIED as
   documentation**: `read_emsphinx_psym_file` Notes now record the
   second deviation (the C++ reads the type token as exactly two
   characters before imbuing the comma-as-whitespace locale, so
   the binary rejects `qu,2` yet accepts `q u`; this reader does
   the opposite in both corners; real psymfiles carry neither).
   Unpinned, as fidelity item 7 disposed.
7. **`_ANGLE_FILE_TYPES` dead constant (nit) APPLIED**: now used
   in the reader's rejection message (which still contains the
   pinned "quaternion" substring), so the constant earns its
   definition.
8. **Dataclass `__eq__` (nit) APPLIED**:
   `@dataclass(frozen=True, eq=False)` with a comment -- the
   generated `__eq__` would compare the ndarray/Rotation fields
   elementwise and raise on `bool()`.  Spec-discretionary; no test
   compared instances.
9. **Dead parameter (nit 12a) APPLIED**: `_exclude_true_ops` no
   longer takes `intensities`; the descending-order precondition
   is stated in its docstring and at the call site.  Coverage of
   `_pseudo_symmetry.py` re-measured **100.00 %** (165 statements
   after the `del` removal, 0 missed) under the default module
   suite.
10. **Nits 12b/12c REJECTED, recorded**: `_flat_to_knm` stays in
    `_pseudo_symmetry` (relocating the two-line divmod decode into
    `_xcorr` is a cross-module refactor with no behavioral gain);
    no size-one guard added to `MasterPatternHarmonics.rotate`
    (sibling methods do not guard either -- house style, as the
    finding itself notes).
11. **Spec wording amendments (minor) NOT APPLIED HERE -- re-flagged
    for the orchestrator**: this stage may only append to this
    file, so requirements D9.1's "identity plus ... 24" phrasing,
    the Automated section's stale `test_ni_identity_plus_oh_count`
    name/wording, and the ni/al weekly honesty wording remain for
    the orchestrator's amendment pass before the commit (as Stage
    A items 3/9 and the conventions review item (d) already
    requested).
12. **P12 sharper rationale, appended per the mutation review**:
    the pre-registered survival of the `point_group.data`-for-
    `proper_subgroup.data` mutant is confirmed reviewed-only, but
    the plan 7.2 rationale ("improper ops are not rotations of the
    volume") is not the mechanism: for centrosymmetric m-3m the
    48-element quaternion set collapses in absolute value to
    exactly the 24 proper quaternions (orix stores an improper
    op's quaternion part, duplicating its proper partner), and the
    blends use point group 1 -- zero observable difference today.
    For a NON-centrosymmetric group with mirrors/rotoinversions
    (e.g. mm2, -4), `point_group.data` WOULD contribute quaternion
    parts equal to two-fold rotations absent from the proper
    subgroup and wrongly exclude genuine pseudo-symmetry
    operators: the shipped `proper_subgroup` line is load-bearing
    even though no current test phase can see it.

**Fix-stage re-run evidence (2026-09-07)**: the four Phase 8 files
-> **381 passed, 17 skipped** at `-n 0` (27 s) and again at `-n 4`
(19 s) -- the review gate's 380 + the new P11 killer; gated + weekly
module run (`KIKUCHIPY_EMSPHINX_DIR` set, `--weekly`, `-n 0`) ->
**50 passed, 2 skipped** (local-masters dormant, Al download
pending 9.7) -- all binary parity green after the code edits; full
spherical selection (`-k "spherical" -n 4`) -> **3122 passed, 738
skipped, 0 failed** (51 s); full-repo collection clean at **4833
tests** (4831 Stage A + the implementation-gate un-normalised-path
test + the P11 killer); ruff check + format clean on the changed
files; `_pseudo_symmetry.py` coverage 100.00 %.
