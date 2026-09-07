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
  proper Oh rotation; intensities ~1.0 within MTP; identity peak
  intensity == 1.0 within MTP (kills a wrong `v_max`); intensities
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
