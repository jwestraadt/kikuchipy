# Phase 12 (provisional) -- `spherical-indexing-gpu`: validation

**Status note:** drafted 2026-09-07 WITH live GPU measurements --
the device micro-benchmarks and CuPy probes ran this session on
this machine in an ephemeral environment (never the project venv)
and are recorded below. **Every GPU number is machine-specific**:
NVIDIA RTX 2000 Ada Generation Laptop GPU (compute capability 8.9,
8188 MiB VRAM), driver 595.71 (CUDA 13.2), CuPy 14.2.0 on CUDA
runtime 12.9, Windows 11; pins derived from them are recorded as
such and re-measured, never assumed portable. End-to-end
parity/performance tolerances marked `MTP` are measured-then-pinned
at the tests/implementation gates (`pytest.approx(measured,
rel=0.05)` or the ~2x margin convention). **GitHub CI green is zero
evidence for the GPU path** (no CI runner has CUDA): every gated
check below is a local definition-of-done gate recorded with
numbers and the machine ID.

## Automated (default suite; runs on CI, no GPU)

```
uv run pytest tests/test_indexing tests/test_signals -k "spherical" -n 4
```

New file `tests/test_indexing/test_spherical_gpu.py` (default-suite
classes) plus additions to `test_spherical_indexer.py` and
`tests/test_signals/test_ebsd_spherical_indexing.py`.

### Backend switch and CPU-default protection

- `test_backend_validation_raises`: unknown string ->
  `ValueError` naming `{"cpu", "gpu"}` (frozen message shape).
  [D1]
- `test_backend_cpu_is_bitwise_default`: `backend="cpu"` result
  bitwise-equal to a call without the keyword on
  `nickel_ebsd_small` -- the guard that shared-code edits never
  perturb the reference path. [D1/D5.1]
- Signature pins: `backend` placement after `pseudo_symmetry_ops`
  before `chunksize`; frozen-defaults dicts gain
  `backend="cpu"`. [D1]

### Availability gate (cupy faked/monkeypatched)

- `test_gate_stages_actionable`: three simulated failures (import,
  device count, cuFFT probe) -> three distinct frozen messages,
  each naming its remedy (`pip install cupy-cuda12x` family /
  driver / CUDA Toolkit or nvidia wheels + DLL note). [D6.2]
- `test_gate_version_floor`: `cupy < 13` simulated -> actionable
  ImportError. [D6.2]
- `test_dll_shim_is_silent`: non-Windows and absent
  `site-packages/nvidia` are silent no-ops (no raise, no log
  spam). [D6.5]
- `test_no_module_scope_cupy_import`: importing
  `kikuchipy.indexing._spherical._gpu` (and `kikuchipy`) succeeds
  with cupy absent from `sys.modules` afterwards. [D6.1]

### xp-agnostic pipeline core (the CI-side correctness anchor)

- `test_pipeline_numpy_xp_matches_cpu`: stages 4-6 run with
  xp=numpy at bw 16-24 synthetic spectra + one real Ni bw-32 case
  vs `SphericalCrossCorrelator`/`NormalizedSphericalCross
  Correlator` (BOTH normalize settings -- the un-normalized
  correlator path is in scope, D2 stage-6): identical argmax cell;
  cube values within an `MTP` f32 band; n_fold in {1, 4}; mirror
  flag both ways. Kills the sign/conj/slice/norm mutant family
  (plan 7.2), specifically incl. the `(-1)^(j+m)`-vs-`(-1)^j`
  `A2` seed (odd-m columns flip -- ONLY visible at n_fold=1, the
  review's blocking find), the missing table NaN-zeroing
  (all-NaN, self-revealing) and the even-slP `n >= bw` guard (the
  bw-16 case: slP 32, bwP 17). [D2/D3/D11.1]
- `test_argmax_first_occurrence_tie`: two planted equal maxima --
  first index wins (`_find_peak` strict-`>` parity). Runs under
  xp=numpy ONLY (review-noted): it proves nothing about
  `cp.argmax`; the cupy twin `test_argmax_tie_gpu` lives in the
  gated suite. [D4/D2]
- `test_peak_epilogue_parity` (review-added): the extracted
  neighborhood-fed epilogue vs `interp_peak` directly on random
  cubes, BOTH compat settings, incl. planted step-rejection cases
  so the `|x[0]|`-twice bounds branch is exercised (end-to-end
  runs mask it -- step rejection is rare, Newton reconverges).
  [D2 stage-7]
- `test_neighborhood_offsets_parity`: `_neighborhood_offsets` vs
  `_extract_neighborhood` index arithmetic on random cubes, odd
  AND even slP, BOTH `emsphinx_compatible` settings, edge/wrap
  cells enumerated (incl. the even-slP one-past clamp). [D2]
- `test_padding_and_stripping`: planted peak inside the padded
  tail never reaches results; `n % B != 0` maps return correct
  shapes/rows. [D7.3]
- `test_failed_patterns_excluded`: `ptp == 0` and masked patterns
  produce the identical fill rows `(0,0,0,0,-1,0,-1)`; the device
  batch shrinks accordingly. [D4.6/D7.4]

### Batch model and helpers

- `test_vram_model_and_default_batch`: `g(bw)` pure-math pins
  (calibration constants named, `MTP`); default-B clamp `[1, 64]`;
  half-free-VRAM headroom rule; explicit `chunksize` override.
  [D8.1-2]
- `test_gpu_memory_helper`: `gpu_memory_per_batch_bytes` =
  batch term + per-phase resident term; monotone in B and bw.
  [D8.3]
- `test_kill_switch`: `KIKUCHIPY_NO_GPU_TESTS=1` makes the
  `cupy_gpu` fixture skip regardless of availability. [D10.3]
- `test_xdist_worker_skip` (review-added): the fixture skips when
  `PYTEST_XDIST_WORKER` is set (faked) -- the structural `-n 0`
  rule that keeps the default `-n 4` command above GPU-silent
  even on cupy-capable machines. [D10.4]

## Local-gated (`cupy_gpu` fixture; skipped on CI) -- run `-n 0`

Fixture skip reasons are instruction-bearing per stage; the
fixture also skips under xdist (`PYTEST_XDIST_WORKER` set), so
this suite runs ONLY at `-n 0` by construction and the default
`-n 4` command above never reaches it (D10.4, review-added). All
pins in this section carry a comment naming the machine (RTX 2000
Ada 8 GB, driver 595.71, CuPy 14.2.0). Command:

```
uv run pytest tests/test_indexing/test_spherical_gpu.py -n 0 -q
```

- `test_gate_passes_here`: three stages green; device/driver/cupy
  versions recorded. [D6]
- `test_iq_bitwise_vs_cpu`: IQ exactly equal (`==`) on
  `nickel_ebsd_small` -- the wiring probe. [D4.1]
- `test_refined_parity_small` / `test_refined_parity_large_20pt`
  (+ `test_refined_parity_large_165pt` weekly): both backends
  `refine=True` -- winner (phase, variant) agreement: the
  small/20-pt tests pin exact flip COUNTS (expect 0; pinned at
  the measured count -- a > 99.9 % fraction is unmeasurable at
  N = 9/20, review-corrected), the flip-RATE pin (expect
  > 99.9 %, `MTP`) lives on the weekly 165-pt and full-map runs;
  refined-to-refined misorientation
  median/max (expect median << 0.05 deg; sanity: far inside the
  0.31-0.34 deg CPU-vs-EMSphInx band, `mission.md:40`; `MTP`);
  scores Pearson r (expect ~1 - eps) and relative |diff| (expect
  1e-5-1e-4; `MTP`). [D4.2/D4.4/D4.5]
- `test_coarse_cell_agreement`: argmax-cell agreement fraction and
  tie/flip census measured and recorded (not the gate -- D4.4 is).
  [D4.3]
- `test_refine_false_coarse_scores`: coarse interpolated scores,
  relative |diff| `MTP` (expect ~1e-6-1e-5, f32 cube). [D3.4/D4.5]
- `test_unnormalised_gpu_parity` (review-added): `normalize=False`
  dual-backend on `nickel_ebsd_small` -- no `r_den` resident, the
  multiply skipped, plain `_find_peak` parity; bands shared with
  D4. [D2 stage-6/D9]
- `test_argmax_tie_gpu` (review-added): planted f32 tie and
  all-equal cube ON DEVICE -- `cp.argmax` first occurrence
  (probe-confirmed at drafting review; the pin guards regression
  and portability, since real-data parity runs essentially never
  hit exact f32 ties). [D4/D2]
- `test_emsphinx_compatible_toggle_parity`: GPU-vs-CPU refined
  parity holds under BOTH `emsphinx_compatible` settings
  (device stages are compat-neutral). [D2]
- `test_psym_dual_backend`: true 90-deg-z op, `n_best=2`:
  `pseudo_symmetry_index` values, orientations and scores parity
  vs CPU (`MTP` bands shared with D4). [D9]
- `test_multiphase_dual_backend`: Ni + sign-scrambled copy: winner
  phase parity per pattern; kills cross-phase table/r_den/n_fold
  mix-ups. [D2/plan 7.2]
- `test_multiphase_mixed_symmetry_dual_backend` (review-added):
  second phase with DIFFERENT declared symmetry (e.g. the
  scrambled master re-declared n_fold=1/no mirror) -- kills stale
  `fxc` columns across phases (the same-master scramble shares
  zero structure and cannot). [D2 freshness rule/plan 7.2]
- `test_run_to_run_bitwise`: two identical GPU runs bitwise-equal
  (fixed B). [D5.2]
- `test_batch_size_invariance`: B in {8, 32, default}: bitwise if
  the drafting-gate measurement says so, else the measured
  tolerance pinned + the deviation recorded here. [D5.3]
- `test_oom_halving`: cupy pool limit forced small -> (a) OOM at
  session build: B halves, the run completes; (b) OOM
  mid-compute: the compute aborts, `index_patterns` rebuilds
  session + graph at B/2, the run completes with results entirely
  from the final B; limit below the B=1 need -> `MemoryError`
  with the frozen remedy text (D8.4 two-window rule,
  review-restructured). [D8.4]
- `test_device_error_propagates` (review-added): a monkeypatched
  non-OOM device-stage exception FAILS the run -- never swallowed
  into per-pattern fill rows (no silent chunk loss). [D7.7/D14.8]
- `test_lazy_input_eager_output` (dask-lazy in -> NumPy out,
  scheduler forced to threads); `test_verbose_gpu_info_line`
  (device name, VRAM, B, model bytes printed; warning above
  free-VRAM). [D7.5/D8.3]
- `test_throughput_floor`: the D12.1 go/no-go -- skips with an
  instruction-bearing reason until the idle-machine CPU baseline
  is recorded below (open question 9.5). [D12]
- Coverage (recorded here when run):

```
uv run --with pytest-cov pytest tests/test_indexing/test_spherical_gpu.py \
  tests/test_indexing/test_spherical_indexer.py \
  tests/test_signals/test_ebsd_spherical_indexing.py -n 0 -q \
  --cov=kikuchipy.indexing._spherical._gpu \
  --cov=kikuchipy.indexing._spherical._indexer --cov-report=term-missing
```

  Gate: `_gpu.py` 100 % from the default + gated union;
  device-independent lines already 100 % from the default suite
  alone; `# pragma: no cover` only on import-guard branches
  (enumerated in the record). Review-added: the CI Codecov patch
  report WILL show the cupy-gated lines uncovered -- accepted,
  recorded, stated in the PR description; this local command is
  the gate. [D11]

## Weekly

- `test_refined_parity_large_165pt` (above).
- Full-map GPU run on `nickel_ebsd_large` (4125 patterns, bw 68,
  refined): throughput + parity spot check recorded.
- bw 88 parity + throughput run; bw 113 VRAM feasibility probe
  (D8.5) recorded, not gated.

## Requirement-to-test mapping

| Decision | Requirement (short) | Killer / evidence | Suite |
|---|---|---|---|
| D1 | explicit backend keyword, cpu default untouched | `test_backend_validation_raises`, `test_backend_cpu_is_bitwise_default`, signature pins | default |
| D2 | narrow device split, compat neutrality, table sanitization, `(-1)^(j+m)`, fxc freshness, epilogue extraction, normalize both ways | `test_pipeline_numpy_xp_matches_cpu` (n_fold=1 + bw-16 even-slP cases), `test_peak_epilogue_parity`, `test_neighborhood_offsets_parity`, `test_emsphinx_compatible_toggle_parity`, `test_multiphase_mixed_symmetry_dual_backend`, `test_unnormalised_gpu_parity`; review grep (no compat branch in device code) | default + gated |
| D3 | c64/f32 device, f64 host, spectrum precision measured | numpy-xp band, `test_refine_false_coarse_scores`, the both-ways D3.2 measurement (Recorded results) | default + gated |
| D4 | parity oracle: IQ bitwise, refined-to-refined MTP, scores same scale, argmax tie semantics | `test_iq_bitwise_vs_cpu`, `test_refined_parity_*` (counts small-N, rates weekly), `test_coarse_cell_agreement`, `test_refine_false_coarse_scores`, `test_argmax_tie_gpu` | gated |
| D5 | determinism per backend | `test_backend_cpu_is_bitwise_default`, `test_run_to_run_bitwise`, `test_batch_size_invariance` | default + gated |
| D6 | optional-dep gate, no extra, DLL shim | `test_gate_stages_actionable`, `test_gate_version_floor`, `test_dll_shim_is_silent`, `test_no_module_scope_cupy_import`; installation.rst bullet (review) | default |
| D7 | lock-guarded chunk topology, padding, no device state on indexer, device errors fail the run | `test_padding_and_stripping`, `test_failed_patterns_excluded`, `test_lazy_input_eager_output`, `test_device_error_propagates`, no-device-attrs assert; 4-worker stress of `test_run_to_run_bitwise` | default + gated |
| D8 | VRAM model, default B, two-window OOM, helper | `test_vram_model_and_default_batch`, `test_gpu_memory_helper`, `test_oom_halving` (both windows), `test_verbose_gpu_info_line` | default + gated |
| D9 | psym + refine unchanged | `test_psym_dual_backend`; refine parity inside D4 tests | gated |
| D10 | local gating, kill switch, structural xdist skip, no marker reuse | `test_kill_switch`, `test_xdist_worker_skip`, fixture skip reasons; review: `gpu` marker untouched | default |
| D11 | coverage split, xp-agnostic core | numpy-xp tests on CI; recorded coverage command | default + gated |
| D12 | performance floor | `test_throughput_floor` + the Performance table | gated/manual |
| D13 | docs scope, licensing | conventions review; CHANGELOG present | review |
| D14 | recorded deviations | this file's Recorded results + docstring Notes (review) | review |

## Performance (D12; measured at the implementation gate, this machine, idle)

| measurement | recipe | recorded value |
|---|---|---|
| 8-worker CPU baseline, `nickel_ebsd_large` route, bw 68 refined, idle machine | fixed-seed run, warm caches, both concurrent workflows quiet (open question 9.5) | measure at implementation gate |
| GPU end-to-end, same route/settings | same run pair, `backend="gpu"`, default B | measure at implementation gate |
| **Floor: GPU >= CPU-8-worker** (else negative result recorded) | ratio of the two rows above | decide at implementation gate |
| projection cross-check (expected ~850-1500 pat/s refined at bw 68, review-corrected: + per-batch zeroing, strided-write mirror-fill rate; gap explained if outside) | derived | record at implementation gate |
| coarse-only (`refine=False`) throughput | same route | record |
| bw 88 refined throughput (expected 650-850 pat/s) | same route at bw 88, B=8-16 | record |
| 2-op psym GPU run + host-refine crossover op count | 0/1/2/4 ops timing | record |
| VRAM pool high-water at (bw 68, B=32) and (bw 88, B=8/16); `g(bw)` calibration + default-B chooser cross-check vs the comfortable band (review-added: the formula yields B~33 at bw 88 with 7.45 GB free, outside the benched B=8-16) | mempool hooks | record (drafting: bw 68 ~B x 50 MB anchored on the measured <= 1.4 GB pool high-water; bw 88 ~B x 110 MB is a SCALING estimate -- no separable-pipeline measurement exists at bw 88) |
| bw 113 feasibility on 8 GB (model ~230 MB/pattern -> B=4-8) | short probe | record, not gated |
| D7.1 lock-wait share at 8 workers (feeder-thread trigger > ~20 %) | timing hooks | record |

## Manual

- Review plan section 0 amendments with the user BEFORE the spec
  commit (open questions 9.1 phase number, 9.2 scope sign-off).
- One end-user smoke run from a notebook: `backend="gpu"` on
  `nickel_ebsd_large`, eyeball the IPF map vs the CPU map.
- Verify the three gate failure messages by hand once each on this
  machine (uninstalled cupy in an ephemeral env; nvidia wheels
  removed for the cuFFT stage) -- the probe-measured failure modes
  reproduced against the shipped messages.

## Definition of done

- All gates of the roadmap sentence: spec recorded (with plan 0
  applied and `<N>` resolved); failing tests committed;
  implementation; adversarial review (incl. the plan 7.2 device
  mutation list and the section-8 gotcha checklist for host touch
  points) + fixes; pre-commit clean; CHANGELOG entry; PR opened.
- **The three spec documents re-submitted to adversarial review**
  (the Phase 8 precedent gate).
- Plan section 9 open questions answered or their provisional
  resolutions recorded; 9.1/9.2 answered BEFORE the spec commit.
- Every `MTP` placeholder replaced by a dated measured value in
  "Recorded results" (with recipe + machine ID); the D3.2
  spectrum-precision both-ways measurement, the D5.3 batch
  invariance verdict, and the D8 VRAM calibration all recorded.
- The D12 floor decided: pass recorded, or the negative result +
  ship/no-ship review decision recorded.
- Parity gates green locally: IQ bitwise, refined-to-refined
  bands, psym/multiphase dual-backend, both compat settings.
- Coverage per D11 (100 % `_gpu.py` union; default-suite-only for
  device-independent lines) with the command output recorded.
- `backend="cpu"` bitwise-default guard green and the FULL
  existing spherical suite green (no CPU-path regression).
- Docs landed: docstrings, installation.rst bullet, module-doc
  determinism re-scope; no public docstring names a phase number.

## Recorded results

### 2026-09-07 (drafting; measured live on this machine)

Environment: NVIDIA RTX 2000 Ada Generation Laptop GPU, cc 8.9,
8188 MiB (7.45 GB free at bench start, WDDM/display attached),
driver 595.71 (CUDA 13.2), CuPy 14.2.0 on CUDA runtime 12.9,
ephemeral `uv run --no-project --with cupy-cuda12x --with
nvidia-cufft-cu12 --with nvidia-cublas-cu12` (the fork venv was
never touched -- a concurrent tutorial build owned it). Scripts
`gpu_bench.py`, `cupy_probe.py`, `profile_hotpath.py` lived in the
session scratchpad (does not survive); the load-bearing numbers
are carried here and in requirements D2-D8/Context.

1. **CPU hot-path profile** (fork venv python 3.13.12, single
   thread, warm numba, `nickel_ebsd_small`, `pc_average`; upper
   bounds -- concurrent CPU load): bw 68 total 14.29 ms/pattern
   (70 pat/s), of which inverse FFT 8.83 ms (61.8 %), spectrum
   1.76 (12.3 %), scale+argmax 1.13, interp 0.34, refine 1.38;
   bw 88 total 32.23 ms (31 pat/s), inverse FFT 21.17 (65.7 %).
   Stages 4-6 = 82 % -- the device target. Scores 0.5143-0.6347,
   inside the pinned band (`_indexer.py:1176-1181`).
2. **Shapes verified live**: bw 68 -> slP 135, bwP 68, `fxc`
   (135,135,68) c128 = 19.83 MB, `xc` (68,135,135) f64 = 9.91 MB,
   sphere dim 71 (1317 window points); bw 88 -> slP 175, `fxc`
   43.12 MB, `xc` 21.56 MB, dim 91.
3. **Device micro-benchmarks** (c64, CUDA events, best-of-3):
   - separable bw-68 pipeline transform ((B,135,135,68) c64 ->
     (B,68,135,135) f32, `norm="forward"`): 0.483 / 0.482 / 0.520
     ms/pattern at B=8/32/64; with `m % 4` alpha pruning (B=32)
     **0.210 ms/pattern** -- 18x-42x the measured CPU
     `_inverse_fft` (8.83 ms);
   - full 3D C2C inverse FFT: 136^3 0.804 ms/cube flat across
     B=8/32/64; 176^3 1.75-1.89 (B=64 left 0.46 GB VRAM free --
     the practical ceiling);
   - Wigner-scale contraction `(k, B*n, j) @ (k, j, m)` c64
     GEMM: 0.045 ms/pattern at bw 68 B=32 (~3.8 TFLOP/s
     effective; einsum 1.56x slower); 0.127 at bw 88 B=8;
   - scale + argmax over (B, 68*135*135) f32: 0.320 / 0.125 /
     0.095 ms/pattern at B=8/32/64 (argmax ~104 GB/s of ~224);
   - transfers: H2D 8.7 GB/s pageable / 12.0 pinned; D2H result
     rows 0.016-0.055 ms; the D2 split ships ~37-74 KB/pattern in,
     ~120 B/pattern out (~2-10 us/pattern -- noise);
   - all bw-68 configurations left >= 4.2 GB VRAM free; pool
     <= 1.4 GB.
4. **CuPy probe facts** (each measured): `importlib.metadata.
   version("cupy")` raises `PackageNotFoundError` with
   `cupy-cuda12x` 14.2.0 installed (`version("cupy-cuda12x")` =
   14.2.0); `import cupy` + `getDeviceCount() == 1` succeed while
   the FIRST FFT raises `ImportError: DLL load failed while
   importing cufft` unless every `site-packages/nvidia/*/bin` is
   registered via `os.add_dll_directory` before use (the
   `cupy-cuda12x` wheel bundles no cuFFT; CuPy 14.2.0 does not
   auto-discover the `nvidia-cufft-cu12` wheel on Windows) -- the
   D6 three-stage gate and the D6.5 shim are built on exactly
   these failure modes; `cupy.fft` `norm="forward"`
   ifft/ifft/irfft matches `scipy.fft` to 2.8e-14; c64 -> f32
   works; `workers=1` is scipy-only (dropped on the device
   branch); `cupy.fft` is not a uarray backend (no
   `__ua_domain__`) -- `set_backend` dispatch rejected;
   un-tuned single 138^3 `irfftn` 40.9 (c128) / 47.9 (c64) ms --
   the "why batching/plan-reuse/pruning are mandatory" prior.
5. **Projection arithmetic** (recorded expectation, not a gate)
   [SUPERSEDED by the review-fix entry below -- omitted per-batch
   zeroing and a strided-write-optimistic mirror-fill rate;
   corrected to ~0.65-0.75/~0.95 ms/pattern GPU-side, end-to-end
   ~850-1500 pat/s, D12.2]: GPU-side ~0.57 ms/pattern (m-3m
   pruned) to ~0.84 (no z-fold) -> 1190-1750 pat/s; host residue
   2.56 ms/pattern-core -> ~2350 (6 cores) to ~3120 (8) pat/s;
   end-to-end ~900-1600 pat/s refined at bw 68 (~6-8x the 205-216
   pat/s 8-worker baseline, ~19-25x the 63.8 pat/s single core);
   bw 88 ~650-850 pat/s.
6. **Priors carried as cautions**: the user's EMSphInx
   `feature/GPU` all-double CUDA build measured 6.6x SLOWER than
   20-thread CPU (`specs/_research/explore-emsphinx-programs-and-
   formats.md:701-708`); EMSphInx's own CPU/CUDA score scales
   diverged ~7x (`explore-real-data-and-tests.md:334`) -- the D4.5
   same-scale requirement exists to not repeat it.
7. **Nothing else ran**: no project tests, no venv changes, no git
   state changes; every end-to-end tolerance in this file is an
   `MTP` placeholder awaiting the tests/implementation gates.

### 2026-09-07 (spec adversarial review -- fixes applied)

Two independent adversarial reviews of the three spec documents
ran with live probes (ephemeral env, this machine; bw-6 GEMM
mapping check vs `_xcorr_spectrum.py_func`, `cp.argmax`
tie/NaN probes). Every blocker/major fixed; every minor applied;
nits applied. Ledger:

**Applied (spec text corrected/decided):**

1. **`A2 = (-1)^(j+m) A`** replaces the drafted `(-1)^j` form (D2
   stage-4) -- probe-verified bitwise vs the CPU kernel at bw 6,
   all four (mirror, n_fold {1,4}) combos; the drafted form flips
   every odd-m column and is invisible at even n_fold (both
   benchmarked m-3m cases). The scratchpad research report's
   section-2.1 hardening carried the same error; the frozen spec
   text is now the corrected formulation and the research report
   (dead scratchpad) is superseded. Plan 7.2 mutant updated.
2. **Table NaN-zeroing** made an explicit frozen step (D2): the
   pi/2 table is majority-NaN by contract; dense GEMMs on the raw
   table are ~99.5 % NaN; A/A2/table_T are built from a NaN-zeroed
   copy (probe-verified to reproduce the CPU j-range exactly).
   Even-slP guards (G rows n < bw only; m = bw column never
   written) derived in the text; bw-16 oracle case exercises them.
3. **`fxc` contract correction + freshness rule** (D2): the CPU
   WRITES its systematic zeros every call (the drafted "never
   written" sentence was inverted and is flagged never to reach a
   docstring); the device buffer is re-zeroed (or fully written)
   per phase; memset cost added to the D8 model + D12 projection;
   `test_multiphase_mixed_symmetry_dual_backend` added (the
   sign-scramble twin shares zero structure and cannot catch
   stale columns).
4. **`normalize=False` supported under `backend="gpu"`** (D2
   stage-6/D9, decided): multiply skipped, no `r_den` residents;
   `test_unnormalised_gpu_parity` added; numpy-xp oracle covers
   both correlators explicitly.
5. **Two-window OOM rule** (D8.4, restructured): session-build OOM
   halves B in place; mid-compute OOM aborts the `.compute()` and
   `index_patterns` rebuilds session + graph at B/2 (no device
   sub-batching -- chunk == batch and the D5.3 uniform-shape
   argument survive); results of a halved run come entirely from
   the final B (D5.2 applies there); `test_oom_halving` covers
   both windows.
6. **Device errors are never per-pattern failures** (new D7.7 +
   D14.8): non-OOM device exceptions (and OOM past the halving
   floor) propagate and fail the run -- never converted to fill
   rows; `test_device_error_propagates` added.
7. **Stage-7 epilogue extraction specified** (D2 stage-7 block +
   plan 2.3): `interp_peak` needs the full host cube, so "host,
   unchanged" was unachievable; a shared neighborhood-fed entry
   point reproduces the epilogue byte-for-byte (incl. the
   `|x[0]|`-twice bounds bug and refine seeding from the
   interpolated triple); `test_peak_epilogue_parity` added with
   planted step-rejection cases.
8. **Keyword-only claim fixed** (D1): keyword-only on the ctor
   only; `EBSD.spherical_indexing` has no bare `*`, so `backend`
   is a plain positional-or-keyword append (the actual Phase 8
   precedent), placement deviation recorded.
9. **Structural xdist skip** (D10.4, decided): the `cupy_gpu`
   fixture skips when `PYTEST_XDIST_WORKER` is set, so the
   default `-n 4` command can never auto-run the gated suite on a
   GPU machine; `test_xdist_worker_skip` added; plan 0.12
   amendment text updated.
10. **VRAM provenance downgraded** (D8.2): bw-68 ~50 MB anchored
    on the measured pool high-water (real working set ~40-44
    MB/pattern); bw-88 ~110 MB relabelled a scaling ESTIMATE
    (only 176^3 C2C probes ran at bw 88); default-B chooser
    cross-check vs the comfortable band added to the MTP
    calibration (formula yields B~33 at bw 88 -- unbenched).
    The reviews also re-derived the drafting component numbers
    (fxc c64 at B=32/bw 68 is 317 MB, xc f32 159 MB -- the
    scratchpad report's 634/317 figures were c128/f64 values
    mislabelled c64/f32; those figures never entered the spec
    text, only the dead scratchpad; the end model survives
    because it is anchored on the measured high-water, direction
    conservative).
11. **Projection corrected** (D12.2): + per-batch zeroing
    (~0.05-0.08 ms/pattern) and a strided-write-realistic mirror
    fill (~0.1-0.2 ms/pattern) -> GPU-side ~0.65-0.75 (m-3m) to
    ~0.95 (no z-fold) ms/pattern, end-to-end ~850-1500 pat/s at
    bw 68; floor unaffected; drafting item 5 above marked
    superseded.
12. **D4.7 NaN deviation corrected**: NaN in slot 0 AGREES
    (probed both sides return 0); the divergence is NaN at any
    OTHER index; the `rDen = +inf` case matches; the `r_den`
    f64 -> f32 inf-overflow family recorded beside it (D14.3
    reworded).
13. **Winner-agreement falsifiability** (D4.2): flip COUNTS
    pinned on the small/20-pt tests, flip RATES only on the
    weekly 165-pt/full-map runs -- no vacuous fraction asserts
    at N = 9/20.
14. **`gpu_memory_per_batch_bytes` clarified** (D8.3): a METHOD
    taking `batch_size`, pure model math, no device query (the
    VRAM query lives in `index_patterns`).
15. **Codecov consequence recorded** (D11.3 + plan 0.13 + the
    coverage gate here): the patch report will show cupy-gated
    lines uncovered; accepted, stated in the PR description.

**Rejected/not-applicable (with reasons):**

- The "634/317 MB" figure corrections had no in-spec target: those
  numbers lived only in the scratchpad research report (does not
  survive); the spec's D8.2 wording fix (item 10) covers the part
  that leaked in ("measured" at bw 88).
- One reviewer's "even a 5x shortfall clears the floor against a
  ~150-190 pat/s baseline" clause was NOT carried into D12: the
  historical baseline band is 205-216 pat/s and the idle-machine
  re-measurement (open question 9.5) is the only number the floor
  may use; the corrected projection stands on its own.
- No finding required a new user decision: plan section 9 is
  unchanged (9.1-9.7 stand as the open questions for the user).

(Sections for the failing-tests gate, the implementation-gate
measurements + pins, and the review-gate re-measurements are
appended here, dated, per the Phase 8/10 pattern.)
