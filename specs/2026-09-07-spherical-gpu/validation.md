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

### 2026-09-07 (failing-tests gate; skeletons + failing suite + the D12.1 CPU baseline)

**Skeletons landed** (full signatures + docstrings, every body
`NotImplementedError` until the implementation gate):
`src/kikuchipy/indexing/_spherical/_gpu.py` (GPL + CMU/Lenthe
headers with the dated modification notice; module doc mapping the
D2 stage split to `_xcorr.py` and the `feature/GPU`
`include/gpu/pipeline.hpp` design; the three-stage gate
`_verify_gpu_or_raise` with seams `_import_cupy` / `_device_count` /
`_probe_cufft`, the frozen D6.2 message constants
(`_GATE_IMPORT_MESSAGE` / `_GATE_VERSION_MESSAGE` /
`_GATE_DEVICE_MESSAGE` / `_GATE_CUFFT_MESSAGE`), the `_gate_result`
cache and the Windows DLL shim `_add_nvidia_dll_directories`; the
xp-agnostic core `_sanitized_table`, `_build_a_tables`,
`_build_g_batch`, `_spectrum_batch`, `_inverse_fft_batch`,
`_scale_argmax_batch`, `_neighborhood_offsets`,
`_gather_neighborhoods`, `_pad_batch`, `_strip_padding`; the VRAM
model trio `_gpu_memory_per_pattern_bytes` /
`_gpu_resident_bytes_per_phase` / `_default_batch_size`; and
`_GpuSession`).  The D1 plumbing is IMPLEMENTED (pure signature
work, deliberately passing at this gate): `backend: str = "cpu"`
keyword-only after `pseudo_symmetry_ops` on the `SphericalIndexer`
ctor and positional-or-keyword after `pseudo_symmetry_ops` / before
`chunksize` on `EBSD.spherical_indexing` (the recorded positional
shift of `chunksize`/`verbose`; no positional caller exists in repo
or tests), the ValueError guard in the nlopt message shape, the
ctor-fired gate for `"gpu"` (bound in `_indexer`'s namespace for
tests to patch), `SphericalIndexer.backend` (the string only), the
`gpu_memory_per_batch_bytes(batch_size)` METHOD stub (D8.3), and
the stage-7 epilogue entry point
`SphericalCrossCorrelator._interp_peak_from_neighborhood` stubbed
in `_xcorr.py` (D2 stage-7 / plan 2.3).  No `.pyi` change needed:
`_gpu.py` adds no public export (checked
`src/kikuchipy/indexing/__init__.pyi`).

**Failing suite landed**:
`tests/test_indexing/test_spherical_gpu.py`, 72 collected tests
(56 test functions; default + gated in the one file, D10.4), plus
spec-driven updates to the sibling suites: the frozen-defaults
dicts of `test_spherical_indexer.py` and
`test_ebsd_spherical_indexing.py` gain `backend: "cpu"`, the
placement pin now reads `pseudo_symmetry_ops` -> `backend` ->
`chunksize`, and `Phase 11`/`Phase 12` joined the
no-phase-numbers-in-public-docstrings pin.  Every MTP placeholder
carries a `FIXME-pin` marker: `XP_CUBE_ATOL_SCALE`,
`REFINED_FLIP_COUNT_SMALL/20PT`, `REFINED_AGREEMENT_RATE_165PT`,
`REFINED_MISO_MEDIAN/MAX_DEG`, `SCORE_PEARSON_MIN`,
`REFINED/COARSE_SCORE_REL_DIFF`, `COARSE_CELL_AGREEMENT_MIN`,
`PSYM_INDEX_FLIP_COUNT`, `VRAM_G68/G88_BOUNDS`,
`VRAM_RESIDENT68_BOUNDS`.  One documented test-design guard: the
numpy-xp oracle's argmax-cell equality carries a symmetric-copy
escape (at even slP an flm z-fold/mirror symmetry duplicates cube
values EXACTLY on the grid -- e.g. slP 32 with n_fold 4 -- so f64
and f32 rounding may break the exact tie at different copies; the
escape requires the two cells value-tied at the maximum, and the
cube-band assert stays the strict mutant killer).

**Run record** (this machine, fork venv, 2026-09-07):

- `uv run pytest tests/test_indexing/test_spherical_gpu.py -n 0 -q`:
  **30 failed, 16 passed, 26 skipped** -- every failure a
  `NotImplementedError` from a skeleton body (two gate-message tests
  surface it as an AssertionError on the placeholder text because
  `NotImplementedError` subclasses `RuntimeError` and is caught by
  their `pytest.raises(RuntimeError)` -- still the unimplemented
  gate, right-reason).  The 16 passes are the implemented D1
  plumbing pins (backend validation both surfaces, bitwise-default
  guard, signature placements, attribute, ctor-gate wiring) and the
  D10 fixture-decision tests (kill switch, structural xdist skip,
  instruction-bearing reasons, wgpu marker unused) -- gating
  infrastructure that must work for the gating to gate.  The 26
  skips are the 25 `cupy_gpu`-gated tests (reason: gate not
  implemented yet -- the fixture's NotImplementedError branch) + 1
  weekly.
- Structural xdist live check
  (`uv run pytest tests/test_indexing/test_spherical_gpu.py -n 4
  -q`): same 30/16/26, with every gated skip reason now "GPU tests
  run only at -n 0 (PYTEST_XDIST_WORKER is set...)" -- the D10.4
  rule observed live before any probe.
- Existing spherical suite (`uv run pytest tests/test_indexing
  tests/test_signals -k "spherical"
  --ignore=tests/test_indexing/test_spherical_gpu.py -n 4 -q`):
  **3122 passed, 741 skipped, 0 failed** in 89 s -- the CPU path is
  unperturbed by the skeleton edits (the D5.1 protection, also
  pinned by the new bitwise-default test which passes).
- Whole-suite collection sanity (`uv run pytest --co -q`): **4908
  tests collected, no errors**.
- Overlay collection (`uv run --with cupy-cuda12x pytest
  tests/test_indexing/test_spherical_gpu.py --co -q`): **72
  collected, clean** -- the gated file collects under a
  cupy-bearing environment.

**D12.1 idle-machine 8-worker CPU baseline** (MEASURED -- the
floor's LEFT side; the open-question-9.5 dedicated window used).
Recipe: the `nickel_ebsd_large` route -- full 4125-pattern map
(55 x 75, uint8 60 x 60), backgrounds removed on the full map
(`remove_static_background` + `remove_dynamic_background`),
detector `pc_average`, harmonics built DIRECTLY at bw 68 from the
shipped Ni master (`from_master_pattern`, lambert, both
hemispheres), `SphericalIndexer` all defaults (normalize=True,
refine=True, emsphinx_compatible=True, n_regions=10),
`chunksize=None` (the ported `_batch_estimate` gives 15/chunk),
`dask.config.set(num_workers=8)`, threaded scheduler,
`progressbar=False`; warm-up `index_patterns` on a 256-pattern
slice, then three timed full-map runs, best of 3.  Command:
`uv run python d12_cpu_baseline.py` (scratchpad script; recipe
carried verbatim here since the scratchpad does not survive).
Machine: the RTX 2000 Ada machine's 20-core laptop CPU, Windows 11
(10.0.26200), fork venv python 3.13.12, idle/dedicated.  Measured:

| run | wall s | pat/s |
|---|---|---|
| warm-up (256) | 2.39 | -- |
| 1 | 22.45 | **183.8** |
| 2 | 26.75 | 154.2 |
| 3 | 26.53 | 155.5 |

**Best of 3 = 183.8 pat/s**, pinned into
`test_spherical_gpu.py::CPU_BASELINE_8_WORKERS_PAT_S` (the
`test_throughput_floor` left side).  Sanity: scores 0.2289-0.6903,
zero failed rows of 4125.  Two notes recorded: (a) runs 2-3 dip
~16 % below run 1 -- laptop thermal behaviour after a sustained
full-map run; best-of-3 is the D12 convention and run 1 followed
the warm-up as intended; (b) the value sits below the historical
non-idle 205-216 pat/s Anes-reproduction band, which was a
different workload/session -- D12.1 requires exactly this
same-machine idle re-measurement, which this is.  The GPU side of
the floor and the ratio row remain implementation-gate
measurements.

**Deferred to the implementation gate** (need the implementation):
every `FIXME-pin` band above; the D3.2 both-ways
spectrum-precision measurement; the D5.3 batch-invariance verdict;
the D8 VRAM calibration (pool high-water at bw 68/88, chooser
cross-check) + the bw-113 probe; the D12 GPU throughput +
projection cross-check + coarse-only/bw-88/psym rows + lock-wait
share; the D9 host-refine crossover op count (needs the measured
GPU rate); the coverage command run; the three gate failure
messages verified by hand (Manual section).

### 2026-09-07 (failing-tests gate: test-critic review -- fixes applied)

An independent test-critic review of the Stage A deliverables (the
failing suite, the skeletons and this file) returned 2 major,
6 minor and 3 nit findings.  Every finding was applied or recorded;
**no assertion was weakened**.  Ledger:

**Applied (tests/pins strengthened):**

1. **(major) D12.1 CPU baseline re-pinned 183.8 -> 236.0 pat/s.**
   The recorded 183.8 / 154.2 / 155.5 pat/s window (warm-up 2.39 s)
   FAILED idle-machine consistency re-runs of the identical recipe
   on the now-dedicated machine: **236.0 / 231.6 / 228.3** pat/s
   (warm-up 1.35 s; critic session) and **221.9 / 223.8 / 209.1**
   (warm-up 1.48 s; this fix-application session, script
   `d12_cpu_baseline_rerun.py` re-created in the session scratchpad
   from the recipe carried verbatim above) -- each re-run with the
   identical score band 0.2289-0.6903 and 0 failed rows of 4125, so
   the route matches and the original window was not thermally/load
   clean (~22 % soft: a 200 pat/s GPU run slower than the honest
   idle CPU would have "passed" the floor).
   `CPU_BASELINE_8_WORKERS_PAT_S = 236.0` now pins the HIGHEST
   honest idle measurement (the hardest floor), re-confirmed in the
   same session/thermal state as the GPU measurement at the
   implementation gate (the D12 convention).  This note amends the
   D12.1 entry above per the append-only rule.
2. **(major) `_gather_neighborhoods` now has a direct unit test**:
   `test_gather_neighborhoods_parity` (odd/even slP x both compat
   settings) gathers DISTINCT random cubes at per-cube edge/wrap +
   random centers through the function under test and asserts
   per-cube `_extract_neighborhood` equality for every b -- the
   wrong-base batched-gather mutant (per-cube flat offsets need
   `+ b * cube_size`) now dies on CI, closing the
   offsets -> gather -> epilogue chain the D11.1 coverage claim
   needed (the offsets test gathers with raw fancy indexing and
   never called the gather).
3. **(minor) D4.2 flip count de-vacuoused**: `assert_refined_parity`
   now counts winner-CELL flips (refined zyz rounded back via
   `euler_to_index`, the `test_coarse_cell_agreement` vocabulary)
   beside the phase_id count, which is constant-0-vs-constant-0 in
   the four single-phase callers; both counts share the FIXME-pin,
   split at the implementation gate if they measure apart.
4. **(minor) D8.3 warning branch + info-line pins**: new gated
   `test_verbose_warns_above_free_vram` (chunksize=100_000 models
   ~5 PB at bw 68 -- the message must carry the warning);
   `test_verbose_gpu_info_line` now also pins the device name
   (queried live from the runtime), the batch size B and a
   model-bytes figure with its unit.
5. **(minor) D8.1/D8.2 wiring pinned**: new gated
   `test_default_batch_size_wiring` spies `_default_batch_size`
   (and `_GpuSession`) through a `chunksize=None` GPU run -- it
   must be consulted, fed a real free-VRAM figure, and its return
   must BE the session batch size; the
   left-on-CPU-`_batch_estimate` mutant (15/chunk at bw 68, silent
   throughput loss) now dies.
6. **(minor) the D7 4-worker lock stress implemented**: new gated
   `test_run_to_run_bitwise_4_workers` (`dask` `num_workers=4`,
   `chunksize=2` -> 5 chunks over 9 patterns, bitwise run-to-run)
   realises the "4-worker stress of `test_run_to_run_bitwise`"
   named by the D7 mapping row and plan 7.2 -- the dask-worker
   axis, not pytest `-n`, is the relevant concurrency axis under
   the structural xdist skip.  The evidence row now has an
   implementation under that node name.
7. **(minor) D2 freshness mutant CI-killable**: `run_xp_pipeline`
   now hands `_spectrum_batch` a NaN-seeded (dirty) `fxc` instead
   of a pre-zeroed one -- the frozen contract re-zeroes (or fully
   writes) per phase, so the rely-on-caller-zeroing mutant dies at
   every default-suite parametrisation instead of only in the gated
   mixed-symmetry run.
8. **(minor) session-build OOM window strengthened**:
   `test_oom_halving_at_session_build` now also asserts no fill
   rows and a CPU-reference misorientation band on the halved run's
   results (matching its mid-compute twin) -- a
   completed-but-corrupted retry fails.
9. **(nit) D8.4 MemoryError pins tightened**: standalone-68 regex
   (no longer matchable inside a byte count), a
   bytes-figure-with-unit regex, "VRAM", the smaller-bandwidth
   remedy and a `backend="cpu"` regex.
10. **(nit) gate failure-caching + shim registration failure
    covered**: new `test_gate_failure_is_cached` (the cached
    exception re-raises with NO re-probe -- a success-only cache
    now fails) and `test_dll_shim_registration_failure_is_silent`
    (Windows + nvidia package present + `os.add_dll_directory`
    raising -> silent `None`, no output; D6.5's third silent
    contract).

**Recorded as append-only notes (no spec-text edit above):**

11. **(nit) traceability name drift**: the mapping table's
    `test_gate_stages_actionable` is implemented as the split
    `test_import_stage_message` / `test_device_stage_message` /
    `test_cufft_stage_message` (+ `test_gate_version_floor`,
    `test_gate_stage_order_and_caching`,
    `test_gate_failure_is_cached`); `test_oom_halving` as
    `test_oom_halving_at_session_build` /
    `test_oom_halving_mid_compute` /
    `test_oom_below_batch_one_raises_memory_error`; the D7 row's
    4-worker stress is `test_run_to_run_bitwise_4_workers` (item
    6).  Supersets of the named evidence; the table text stands
    unedited per the append-only discipline and this note is the
    DoD-walk key.
12. **(nit) bitwise-default guard scope clarified**:
    `test_backend_cpu_is_bitwise_default` pins that the keyword is
    a no-op and that the default is "cpu" (with the two signature
    pins); it CANNOT catch a shared-code edit that perturbs both
    arms equally.  The actual perturbation guard is the pinned
    existing spherical suite (3122 passed, 0 failed at this gate),
    listed separately in the DoD -- never trust the one test alone.

**Rejected: none.**

**Re-run record after the fixes** (this machine, fork venv,
2026-09-07):

- `uv run pytest tests/test_indexing/test_spherical_gpu.py -n 0
  -q`: **36 failed, 16 passed, 29 skipped** (81 collected, was 72:
  +6 default-suite failures -- the four gather-parity
  parametrisations, the cached-failure gate test and the shim
  registration-failure test -- and +3 gated skips -- the warning
  branch, the default-B wiring and the 4-worker stress).  Every
  failure right-reason: 34 direct skeleton `NotImplementedError`s
  plus the two known gate-message tests asserting on the
  placeholder text (`NotImplementedError` subclasses
  `RuntimeError`).
- Whole-suite collection (`uv run pytest --co -q`): **4917
  collected, no errors** (was 4908; +9 = the new tests).
- Overlay collection (`uv run --with cupy-cuda12x pytest
  tests/test_indexing/test_spherical_gpu.py --co -q`): **81
  collected, clean**.

### 2026-09-07 (implementation gate, module A: `_gpu.py` device core + `_xcorr.py` epilogue)

Scope of this entry: the module A deliverables of plan 5.1 -- the
three-stage gate + Windows DLL shim, the xp-agnostic pipeline core,
the VRAM model trio, `_GpuSession`, and the stage-7 epilogue
extraction `SphericalCrossCorrelator._interp_peak_from_neighborhood`
(with `interp_peak` delegating to it -- single source of truth,
bitwise-guarded below).  The indexer/signal integration (module B)
is recorded separately.  Machine: NVIDIA RTX 2000 Ada Generation
Laptop GPU (cc 8.9, 8188 MiB), driver 595.71 (CUDA 13.2), CuPy
14.2.0 on CUDA runtime 12.9, Windows 11; GPU runs via the ephemeral
overlay `uv run --with cupy-cuda12x --with nvidia-cufft-cu12 --with
nvidia-cublas-cu12` (the fork venv never touched).

1. **`XP_CUBE_ATOL_SCALE` pinned 1e-5 (placeholder) -> 3e-7.**
   Measured max relative cube deviation |xc_f32 - cube_f64| /
   max|cube| over ALL 50 oracle cases (bw 16/20 x n_fold {1,4} x
   mirror x both normalize branches x 3 cubes, + the real Ni bw-32
   case both branches, the tests' identical seeds): **1.350e-07**
   under the shipped D3.2 choice (i), 1.479e-07 under the rejected
   (ii) formulation; every argmax cell agreed exactly in every case
   (no tie-escape ever taken).  Pinned at ~2x.  Command:
   `uv run python measure_xp_band.py` (scratchpad script importing
   the test module so seeds/cells are bitwise the tests' own).
2. **D3.2 both-ways spectrum-precision measurement (decided: (i),
   all-c64).**  At bw 68, B=32, n_fold 4, mirror, CUDA events,
   best of 3: (ii) c128-multiply-then-cast G build 3.606 ms/batch
   (112.7 us/pattern), device stages 4-5 0.819 ms/pattern, cube
   max rel err vs the f64 CPU cube 1.251e-07; (i) all-c64 G build
   0.773 ms/batch (24.2 us/pattern), stages 4-5 0.702 ms/pattern,
   err 1.397e-07.  Parity indistinguishable (same order, both
   orders of magnitude inside every D4 band), (i) cheaper ->
   **kept (i)**: `_build_g_batch` casts gln -> complex64 and the
   table -> float32 internally, the session holds the table
   resident float32, and the `A`/`A2` tables are still built from
   the float64 table at complex128 and cast once (D3.2's
   "cast once" preserved on the resident side).  Both ways and the
   choice recorded here per D3.2.
3. **cupy correctness of the shipped core** (the D11.1 A/B lever
   exercised for real): the bw-32 synthetic grid (n_fold {1,4} x
   mirror x both normalize branches, 3 cubes each) under xp=cupy
   vs the f64 CPU oracle -- worst cube rel err **3.438e-07**, all
   24 argmax cells equal.  (`measure_gpu_final.py`, scratchpad.)
4. **VRAM model calibrated (D8.2/D8.3), pins filled.**  The
   drafting component sum omitted the two mixed-quadrant sign
   products of the fill (`value[:, 1:] * sign`,
   `negated[:, :, 1:] * sign` -- ~5.0 MB/pattern at bw 68); the
   measured bw-68 unpruned high-water exposed it and the component
   was added.  Model now: g(68) = 49,866,256 B (~49.9 MB), g(88) =
   108,293,856 (~108.3), g(113) = 229,640,656 (~229.6) -- landing
   on the drafting anchors (~50/~110/~230).  Measured mempool
   working sets (residents excluded), full device stages 4-6 incl.
   gather: bw 68 B=32 **52.4 MB/pattern unpruned** (n_fold 1) /
   **37.6 pruned** (n_fold 4); bw 88 B=8 **82.3 pruned**; bw 113
   B=4 **200.1 pruned** (pool high-water 852 MB -- D8.5: bw 113
   comfortably feasible on 8 GB at small B; chooser gives B=16 at
   7.45 GB free).  The ~5 % the unpruned bw-68 run sits above the
   sum is pool block granularity, absorbed by the half-free
   headroom.  Residents measured vs model: bw 68 11.2 vs 11.25 MB,
   bw 88 24.4 vs 24.4, bw 113 51.7 vs 51.7 (exact).  Pins:
   `VRAM_G68_BOUNDS = (45e6, 55e6)`, `VRAM_G88_BOUNDS =
   (95e6, 120e6)`, `VRAM_RESIDENT68_BOUNDS = (10e6, 12.5e6)` --
   bands bracket the model values, measurement recorded here.
5. **D8.2 chooser cross-check (review-added item validated).**
   At 7.45 GB free the chooser now yields B=64 (bw 68), **B=34**
   (bw 88; was ~38 pre-calibration) and B=16 (bw 113).  The bw-88
   concern ("outside the benched B=8-16 band") was validated by
   RUNNING the chooser's own B: bw 88 at B=38, full stages 4-6,
   pool high-water 3121 MB (81.5 MB/pattern) on the idle card,
   completing cleanly with 7.43 GB free afterwards -- the larger B
   is validated on this card and no bw-dependent cap is added.
6. **Stages 4-6 device timing record** (not a gate; D12.2 context):
   bw 68 B=32 incl. gln H2D upload, G build, GEMM fill, separable
   inverse FFT, scale+argmax: **0.809 ms/pattern pruned** (n_fold
   4) / 0.962 unpruned, best of 3.  Slightly above the ~0.65-0.75
   corrected projection band (the projection excluded the upload
   and per-call G build); the D12 floor verdict belongs to the
   module B end-to-end measurement.
7. **Test fix recorded (provably-wrong-vs-spec rule invoked,
   strengthen-only):** the even-slP one-past clamp branch of
   `_neighborhood_offsets` (the `offset > last` clamp,
   `_xcorr.py:739-746` semantics) was **unexercised** by
   `test_neighborhood_offsets_parity`/`test_gather_neighborhoods_
   parity` -- their enumerated centers never reach the documented
   clamp family `(bwP-1, bwP-2, m0)` / `(bwP-1, n0, bwP-2)` (the
   `_extract_neighborhood` docstring's reachable set), so the
   "incl. the even-slP one-past clamp" claim of this file and the
   in-test comments was vacuous (coverage proved the miss).  Fixed
   by ADDING clamp-family centers `(bwP-1, bwP-2, 0)`,
   `(bwP-1, bwP-2, slP//2)`, `(bwP-1, 0, bwP-2)` (offsets test)
   and `(bwP-1, bwP-2, slP-1)` (gather test); no assertion
   weakened; the clamp lines are now covered and parity-pinned.
8. **Coverage (D11)**: `_gpu.py` **100.00 %** (285 statements,
   0 missed) from the default + gated union -- command:
   `uv run --with pytest-cov --with cupy-cuda12x --with
   nvidia-cufft-cu12 --with nvidia-cublas-cu12 pytest
   tests/test_indexing/test_spherical_gpu.py -n 0 -q -k "not
   test_throughput_floor" --cov=kikuchipy.indexing._spherical._gpu
   --cov-report=term-missing` (the deselected floor test adds no
   `_gpu.py` lines; it runs in the module B D12 window).  Default
   suite alone: 85.96 % -- the 40 uncovered lines are exactly the
   cupy-touching remainder (real gate probes, session allocation,
   D2H/H2D), the recorded D11 split.  No `# pragma: no cover`
   anywhere in `_gpu.py` outside the pre-existing TYPE_CHECKING
   import guard.
9. **Run record** (this machine, 2026-09-07):
   - default `uv run pytest tests/test_indexing/test_spherical_gpu.py
     -n 0 -q`: **52 passed, 29 skipped, 0 failed**; `-n 4`: same
     with every gated test skipping structurally (D10.4 observed).
   - gated overlay `-n 0 -k "not test_throughput_floor"`:
     **79 passed, 1 skipped (weekly), 1 deselected** -- the whole
     gated suite green on this machine through the module B
     integration, incl. both OOM windows, device-error
     propagation, run-to-run bitwise (plain and 4-worker), batch
     size invariance, psym/multiphase/mixed-symmetry and both
     compat settings.
   - existing spherical suite (`uv run pytest tests/test_indexing
     tests/test_signals -k "spherical"
     --ignore=tests/test_indexing/test_spherical_gpu.py -n 4 -q`):
     **3122 passed, 741 skipped, 0 failed** AFTER the
     `interp_peak` -> `_interp_peak_from_neighborhood` delegation
     -- the CPU path (incl. every bitwise pin) is unperturbed by
     the stage-7 extraction, the D2 bitwise guard.
   - `pre-commit run --files` on the three changed files: clean
     (ruff, ruff-format, licenseheaders).

### 2026-09-07 (implementation gate, module B: integration, parity pins, the D12 floor)

Scope of this entry: the module B deliverables of plan 5.2/5.4 --
the `_index_chunk` GPU restructure (`_index_chunk_gpu`: host stages
0-3 batched with failed-pattern exclusion, ONE lock-guarded device
section per chunk resolving every pipeline seam through the `_gpu`
module at call time, host stages 7-9 per pattern in original order
via the shared epilogue), the `_GpuSession` lifecycle in
`index_patterns` (D8.4 two-window OOM halving flooring at the
frozen `MemoryError`; `scheduler="threads"` forced; session per
call, disposed in a `finally`), chunk == batch semantics with the
`_default_batch_size` default and explicit-`chunksize` override
(`_batch_estimate` bypassed), `gpu_memory_per_batch_bytes`, the
`get_info_message` device block (device name, free/total VRAM,
batch size, model bytes, the above-free-VRAM warning), the
module-doc determinism re-scope, the `EBSD.spherical_indexing`
GPU Notes paragraph, the D13 CHANGELOG entry (PR #15 assumed --
next free number at open time) and the `installation.rst` cupy
bullet + `[all]`-excludes extension.  The session is threaded into
the dask graph through a `_SessionHandle` wrapper whose
`__dask_tokenize__` is the session identity, so nothing
device-resident is tokenized or pickled at graph build; the CPU
graph still calls `_index_chunk(block, indexer, n_best)` with the
pre-GPU arity (no session argument appended on the CPU path).
Machine: the same RTX 2000 Ada / driver 595.71 / CuPy 14.2.0 /
Windows 11 environment as the module A entry; all GPU runs via the
same ephemeral overlay.

1. **D4/D9 parity measured and pinned** (script
   `gpu_parity_measure.py`, scratchpad; command `uv run --with
   cupy-cuda12x --with nvidia-cufft-cu12 --with nvidia-cublas-cu12
   python gpu_parity_measure.py`; run twice -- before and after the
   module A D3.2 choice landed -- final numbers are the shipped
   build's):

   | set / config | phase flips | cell flips | miso median / max (deg) | Pearson r | score rel diff max |
   |---|---|---|---|---|---|
   | small (9) refined | 0 | 0 | 0.0 / 0.0 | 1.0 (10 digits) | 8.401e-11 |
   | small coarse (refine=False) | 0 | 0 | 0.0 / 0.0 | 1.0 | 1.273e-07 |
   | small refined compat=False | 0 | 0 | 0.0 / 0.0 | 1.0 | 8.401e-11 |
   | small refined normalize=False | 0 | 0 | 0.0 / 0.0 | 1.0 | 6.439e-11 |
   | psym 90-deg-z, n_best=2, unnorm. | 0 index flips | -- | 0.0 / 0.0 | -- | 6.439e-11 |
   | multiphase m-3m scramble | phase_id equal | -- | max 0.0 | -- | 8.401e-11 |
   | multiphase mixed-symmetry ("1") | phase_id equal | -- | max 0.0 | -- | 8.401e-11 |
   | large 20-pt refined | 0 | 0 | 0.0 / 0.0 | 1.0 | 1.282e-11 |
   | large 165-pt refined (weekly) | 0 | 0 | 0.0 / 0.0 | 1.0 | 1.560e-11 |

   IQ bitwise-equal everywhere (D4.1).  The refined-to-refined
   misorientation is EXACTLY 0.0 deg on every set: the f32 coarse
   argmax cell never moved (165-pt agreement 1.000000, coarse-cell
   agreement 9/9) and the float64 Newton refinement reconverges
   below the quaternion `angle_with` resolution.  Pins filled
   (test constants, ~2x margin): `REFINED_FLIP_COUNT_SMALL/20PT`
   = 0 (measured 0), `REFINED_AGREEMENT_RATE_165PT` = 0.99
   (measured 1.0; allows one flip at N=165),
   `REFINED_MISO_MEDIAN/MAX_DEG` = 0.001 / 0.01 (measured 0.0;
   generous absolute margin over numerical zero, 300x inside the
   0.31-0.34 deg CPU-vs-EMSphInx anchor), `SCORE_PEARSON_MIN` =
   0.999999, `REFINED_SCORE_REL_DIFF` = 2e-10 (measured 8.4e-11),
   `COARSE_SCORE_REL_DIFF` = 3e-7 (measured 1.27e-7, the f32-cube
   surface), `COARSE_CELL_AGREEMENT_MIN` = 0.99 (measured 1.0),
   `PSYM_INDEX_FLIP_COUNT` = 0 (measured 0 over all 18 rows
   despite the true-operator near-ties).  The D4.5 same-scale
   requirement holds by construction and by measurement -- the
   EMSphInx ~7x CPU/CUDA scale divergence is not repeated.
2. **D5.2/D5.3 determinism verdicts.**  Run-to-run bitwise at
   fixed B: green (`test_run_to_run_bitwise`), including the
   4-worker dask lock stress (`test_run_to_run_bitwise_4_workers`,
   5 chunks over 9 patterns, `num_workers=4`).  **Batch-size
   invariance MEASURED BITWISE**: B in {8, 32, default(=64)}
   produce bitwise-equal results on the small set
   (`test_batch_size_invariance` green with the bitwise assert) --
   the D7.3 uniform-batch-shape padding delivered; NO tolerance
   relaxation, no recorded deviation needed for D5.3.
3. **D8 VRAM/batch observations, integration side** (script
   `gpu_vram_measure.py`, scratchpad; real `nickel_ebsd_small`
   route, m-3m = pruned; run BEFORE the module A recalibration
   landed, so the "model" column of that log is superseded by the
   module A entry -- the measured pool high-waters stand):
   bw 68 B=32: pool high-water **1213 MB** (37.9 MB/pattern,
   matching module A's 37.6 pruned); bw 68 B=64 (the default-B
   route): **2414 MB**; bw 88 B=8: 763 MB; B=16: 1328 MB;
   bw 88 B=38 (the pre-recalibration chooser value -- the
   review-flagged "B~33 outside the benched band" case): **3120 MB
   high-water, run completed, 0 failed rows** -- the chooser's
   larger-than-benched B at bw 88 is VALIDATED on this card, no
   bw-dependent cap added (the shipped recalibrated chooser gives
   B=34 at 7.45 GB free, slightly more conservative); bw 113 B=4:
   928 MB high-water, completed, 0 failed rows (D8.5: feasible on
   8 GB; shipped chooser gives B=16).  Every free-VRAM figure and
   the verbose info line/warning were exercised live
   (`test_verbose_gpu_info_line`, `test_verbose_warns_above_free_
   vram`, `test_default_batch_size_wiring` all green).
4. **D12 throughput protocol** (script `d12_throughput.py` +
   rested re-rows `d12_rested_rows.py`, scratchpad; the recipe
   verbatim from the D12.1 baseline entry: full 4125-pattern
   `nickel_ebsd_large` map, bw 68, all defaults, warm-up 256, three
   timed full-map runs, best of 3; machine dedicated and idle, GPU
   0 % / 49-55 C at run starts; overlay):

   | row | runs (pat/s) | best |
   |---|---|---|
   | CPU 8-worker refined (same-session re-confirmation) | 195.9 / 190.4 / 188.0 | 195.9 |
   | **GPU refined bw 68, default B=64, default workers** | 911.1 / 886.0 / 917.6 | **917.6** |
   | GPU refined bw 68, default B=64, 8 workers | 961.2 / 987.4 / 928.5 | 987.4 |
   | GPU refined bw 68, floor-test node itself (`test_throughput_floor`, rested) | -- | 901.4 |
   | GPU coarse-only (refine=False) bw 68 | 899.6 / 912.4 / 921.1 | 921.1 |
   | GPU refined bw 88, default B=33-34 (rested, 45-56 C) | 459.3 / 454.9 / 465.1 | 465.1 |

   **FLOOR VERDICT: PASS.**  GPU best-of-3 917.6 pat/s (script) /
   901.4 (the green `test_throughput_floor` node) against the
   pinned hardest idle CPU baseline **236.0 pat/s** -> **3.8-3.9x
   the floor** (4.2x at 8 workers).  Honesty note on the left
   side: the same-session CPU re-confirmation measured 195.9
   best -- 17 % below the 236.0 pin (pinned from the cleaner
   critic-session window; today's session had run GPU suites
   first).  The floor is judged against the HARDEST pin per the
   D12.1 convention and passes with 3.9x margin; against the
   same-session pair the ratio is 4.7x.  Projection cross-check
   (D12.2): 917.6 (default workers) / 987.4 (8 workers) sits
   inside the corrected ~850-1500 pat/s band -- no gap to
   explain at bw 68.  bw 88 measured 465.1 vs the ~650-850
   recorded expectation: below the band; the shortfall is
   consistent with the strided mirror-fill/zeroing terms scaling
   with slP^3 (175^3/135^3 = 2.2x) beyond the drafting
   arithmetic -- the review's "strided-write-sensitive" flag
   materialised at bw 88; recorded, not floored (the D12.1 floor
   binds bw 68 only).  First-pass psym/bw-88 rows measured during
   a thermally-decaying window (bw 88 209-460, psym-0 455) were
   re-measured rested with temperature gates; the rested rows are
   the recorded ones.
5. **D9 psym op scaling + host-refine crossover** (rested,
   temperature-gated, 8 workers, full map): 0 ops **974.2**,
   1 op **657.6**, 2 ops **497.2**, 4 ops **292.6** pat/s ->
   ~0.49-0.60 ms/pattern of added WALL time per operator (the
   Phase 8 per-op host cost overlaps only partially at 8
   workers).  **Crossover op count ~= 2**: at 2 operators the
   added host-refinement wall time (~1.0 ms/pattern) equals the
   device-bound base (1.03 ms/pattern at 974 pat/s), so beyond ~2
   ops host refinement, not the GPU, bounds throughput --
   recorded, not acted on (device-batched variant refinement
   stays the deferred non-goal).
6. **D7.1 lock-wait share** (timing-proxy run, 8 workers, full
   map, hooks add overhead): wall 4.26 s, lock HELD 4.16 s =
   **97.5 % of wall** -- the device section IS the run;
   cumulative worker wait 12.0 s over 69 acquisitions (35 % of
   8-worker core time).  The plan 6 feeder-thread trigger
   ("device-section wait > ~20 % of wall at 8 workers") is
   formally exceeded, BUT the hold share shows the GPU itself is
   saturated: total host core time (~2.56 ms/pattern x 4125 /
   8 workers ~= 1.3 s) is far below the wall, so waiting workers
   are idle capacity, not lost throughput, and a feeder thread
   cannot make the saturated device faster.  Recorded: the
   fallback is NOT triggered in substance; revisit only if the
   device sections ever stop dominating.
7. **Suite record after all edits + `ruff format`** (this
   machine, fork venv):
   - `uv run pytest tests/test_indexing/test_spherical_gpu.py -n 0
     -q`: **52 passed, 29 skipped** (skips = the gated suite, the
     stage-(a) message as reason -- cupy absent from the venv).
   - same at `-n 4`: 52 passed, 29 skipped, every gated skip
     reason the structural-xdist one (D10.4 observed live).
   - existing spherical suite (`-k "spherical"` minus the GPU
     file, `-n 4`): **3122 passed, 741 skipped, 0 failed** -- the
     CPU reference path is unperturbed (D5.1; the one adjustment
     during integration: `_map_chunks` appends the session
     argument ONLY on the GPU path, so the pre-existing
     `test_an_explicit_chunksize_reaches_the_graph` spy arity and
     the CPU graph shape survive verbatim).
   - whole-suite collection: **4917 collected, no errors**.
   - full gated run, overlay, `-n 0 --weekly` (floor included):
     **81 passed, 0 failed** (twice: pre- and post-format).
   - coverage union run (the D11 command + the cupy overlay):
     `_gpu.py` **100.00 %**; `_indexer.py` 92.46 % on the
     three-file subset -- the uncovered lines are (a)
     pre-existing refine-path lines covered by other suite files
     and (b) an enumerated residual of NEW integration lines
     with no test in the frozen suite: the `_to_host` numpy
     branch, pass-1 guard-(b)/exception arms of
     `_index_chunk_gpu`, the normalized-psym variant seeding on
     the GPU path (the pinned D9 test is the un-normalized twin,
     per spec), pass-3 guard-(c)/exception arms, the
     `gpu_memory_per_batch_bytes` batch_size<1 ValueError, the
     `get_info_message` chunksize=None device branch, the
     GPU-path ProgressBar branch, the window-(b) B=1 MemoryError
     twin and the defensive `memGetInfo` catch of
     `_gpu_out_of_memory_error`.  Flagged for the adversarial
     review/fix stage (add tests there if wanted -- the
     tests-frozen rule bound this gate).
   - `pre-commit run --files` on the five module B files
     (`_indexer.py`, `ebsd.py`, `CHANGELOG.rst`,
     `installation.rst`, `test_spherical_gpu.py`): clean after
     one `ruff format` reformat of `_indexer.py`; all suites
     re-run green post-format.
8. **Docs landed (D13, module B share)**: CHANGELOG `Added`
   entry naming `backend="gpu"` (PR #15 expected);
   `installation.rst` cupy bullet (wheel-per-CUDA + the Windows
   nvidia-wheels note) and the `[all]`-excludes note extended to
   name cupy beside pyopencl; `EBSD.spherical_indexing` Notes
   gained the GPU paragraph (float32 coarse stage, determinism
   scoping, VRAM model + halving + MemoryError, score-sensitivity
   escape hatch, no silent fallback); the `_indexer` module doc
   determinism sentence re-scoped per backend (plan 2.6).  No
   public docstring names a roadmap phase number (pin green).

### 2026-09-07 (implementation adversarial review -- fixes applied)

Two independent adversarial reviews of the Phase 12 implementation
(one code/parity review with live GPU probes, one mutation-testing
review) returned 20 findings: 5 major, 8 minor, 7 nit.  Every
finding applied or recorded; **no assertion weakened**.  Machine:
the same RTX 2000 Ada / driver 595.71 / CuPy 14.2.0 / Windows 11
environment; all GPU runs via the ephemeral overlay.  Ledger:

**Applied (code fixed):**

1. **(minor) D8.4 OOM traceback pinning fixed** (`_indexer.py`,
   both windows of `_index_patterns_gpu`): the caught
   `OutOfMemoryError`'s traceback frames kept the dead batch's
   device arrays alive until a cyclic gc pass (measured with a real
   30 MB pool limit: 22.6 MB still allocated after the caught
   MemoryError, and the frozen message's free-VRAM figure queried
   while pinned).  Fix: `error.__traceback__ = None` +
   `gc.collect()` before `free_all_blocks()` in both except arms,
   so halved retries start clean and the bottomed-out MemoryError
   reports an honest free-VRAM figure.  Behaviourally pinned by the
   new gated `test_oom_real_pool_limit_floor` (below).
2. **(minor x2, two reviewers) DLL shim `os.listdir` moved inside
   the try** (`_gpu.py` `_add_nvidia_dll_directories`): a listing
   failure (ACL-restricted / concurrently-removed nvidia subtree)
   escaped the D6.5 silent contract and would poison the cached
   gate verdict with a raw OSError.  Per-base try, `continue` on
   failure; new default test
   `test_dll_shim_listdir_failure_is_silent`.
3. **(minor) cuBLAS probed beside cuFFT in gate stage (c)**
   (`_gpu.py` `_probe_cufft`): a tiny complex64 matmul joins the
   probe FFT -- the recorded message-frozen-compatible extension
   (the frozen stage-(c) message already names both wheels).
   Confirmed live at review: cufft wheel without cublas wheel
   passed the FFT-only gate and died mid-run at the first spectrum
   GEMM with a cryptic DLL error; now it fails the ctor gate
   actionably.  `test_cufft_stage_message` extended with the
   cublas-failure shape (fake `matmul_error` seam).
4. **(minor) device residency bounded to the D8 model**
   (`_indexer.py` `_index_chunk_gpu`): `del g, g2, gln_batch, xc,
   indices` at the end of the lock-guarded section -- the batch
   factors and last phase's arrays no longer stay referenced as
   frame locals through host pass 3, so finished chunks' leftovers
   cannot stack transient VRAM beyond the single-batch g(bw)
   budget under N dask workers.
5. **(nit, partial) gate cache re-raises a FRESH copy**
   (`_gpu.py` `_verify_gpu_or_raise`): `type(cached)(*cached.args)
   from cached`, falling back to the cached instance for
   unreconstructible signatures; the one cached object's traceback
   no longer grows across raises and two threads never raise the
   same instance.  Covered by the strengthened
   `test_gate_failure_is_cached` (fresh-copy + cause asserts) and
   the new `test_gate_failure_cache_unreconstructible`.  The
   cold-start double-probe race is NOT locked: recorded as
   accepted (idempotent duplicate work, converging verdict -- the
   finding's own assessment).
6. **(nit) docstring completeness**: `gpu_memory_per_batch_bytes`
   gained its `Raises` section; `EBSD.spherical_indexing` `Raises`
   gained the D8.4 `MemoryError` entry (`ebsd.py`).
7. **(nit) `_inverse_fft_batch` docstring softened + test-constant
   comment**: the numpy path runs at COMPARABLE, not identical,
   precision to the device (numpy c128-and-re-round vs cupy true
   c64); `XP_CUBE_ATOL_SCALE` now carries a NUMPY-XP ONLY warning
   naming the measured cupy cube errors (3.44e-7 gate / 3.59e-7
   review) that already exceed it.

**Applied (tests strengthened -- the four SURVIVING mutants each
verified killed by re-applying the mutant against the strengthened
suite on this machine):**

8. **(major, M21 wrong-phase r_den)**:
   `test_multiphase_dual_backend` now spies
   `_gpu._scale_argmax_batch` through a real 2-phase GPU run and
   asserts the received `r_den` **is** `session.r_dens[p]` by
   identity, in phase order per chunk (the scrambled phase's r_den
   differs only ~10 % smoothly -- measured max rel diff 1.03e-1 at
   bw 68 -- and moves no sharp argmax, so no value-level assert
   can see the wiring).  Mutant re-applied: 1 failed.
9. **(major, M12b window-(b)-only never-halving)**:
   `test_oom_halving_mid_compute` gained the `_GpuSession`
   batch-size spy of its session-build twin and asserts
   `built == [8, 4]` -- the rebuild at HALF the original B.
   Mutant re-applied (`batch_size //= 1`, window (b) only):
   1 failed (no hang -- the spy kills before any loop).
10. **(major, M13 green-by-skip)**: new `TestExpectGpuCanary` --
    when `KIKUCHIPY_EXPECT_GPU` is set (and neither the kill
    switch nor xdist applies), a non-fixture test asserts
    `_cupy_gpu_skip_reason() is None`, so a gate/shim regression
    on a known-GPU machine FAILS the run instead of skipping the
    whole gated suite with exit 0.  **Recipe amendment: the gated
    -n 0 command on the dedicated machine now sets
    `KIKUCHIPY_EXPECT_GPU=1`.**  Mutant re-applied (shim
    `bin -> binn`): gated tests skip as designed, canary FAILS --
    exit non-zero, kill confirmed.
11. **(major, M10 forced complex128 at the D3 seam)**:
    `run_xp_pipeline` now asserts `a/a2/g/g2` are complex64 (every
    numpy-xp oracle parametrisation is a killer -- the D4/D3 bands
    are upper bounds a more-accurate c128 GEMM trivially
    satisfies), and the new gated `test_session_dtypes` pins the
    REAL session (table float32, A/A2 complex64, r_den float32,
    fxc complex64).  Mutant re-applied: 16/16 oracle rows failed.
12. **(major, union coverage residual) every enumerated NEW
    integration line now covered**, closing the module-B item-7
    flag: (a) new default-suite
    `TestIndexChunkGpuNumpySession` drives `_index_chunk_gpu`
    through a numpy-backed session honouring the `_GpuSession`
    attribute contract -- pass-1 guard-(b) and per-pattern except
    arms (planted constant-after-preprocessing and raising
    preprocessing), pass-3 guard-(c) and except arms (planted
    non-finite and raising epilogue), the normalized-path psym
    variant seeding (true 90-deg-z op, n_best 2, vs the CPU
    `_index_chunk` under identical plants) and the `_to_host`
    numpy branch -- the amended-D11 device-independent lines now
    default-covered; (b) new default tests
    `test_gpu_memory_helper_rejects_batch_below_one` and
    `test_to_host_both_branches`; (c) new gated
    `test_oom_mid_compute_below_batch_one_raises_memory_error`
    (persistent window-(b) OOM at chunksize 1 -> the frozen
    MemoryError with all content pins -- the window-(b) B=1
    re-raise, previously executed by NO test),
    `test_progressbar_gpu_smoke` (the GPU ProgressBar branch),
    `test_verbose_gpu_info_line_default_chunksize` (the
    `get_info_message` chunksize=None device branch) and
    `test_oom_message_defensive_free_vram_query` (the defensive
    memGetInfo catch: 0 MB reported, remedies kept).  Coverage
    re-run (both recorded commands): `_gpu.py` **100.00 %** union;
    the `_indexer.py` union-missing intersection now contains ONLY
    pre-existing refine-path/`refine_patterns` lines covered by
    other suite files -- zero NEW integration lines uncovered.
13. **(minor, recommendation of the parity review)** new gated
    `test_psym_normalized_dual_backend` (normalize=True, true
    90-deg-z op, n_best=2): the normalized-path GPU variant
    seeding now has a gated dual-backend killer (review probe had
    measured 0 flips, miso 0.0, score rel diff 8.4e-11); and new
    gated `test_oom_real_pool_limit_floor`: a REAL 30 MB pool
    limit drives window (a) halving 64 -> 1, the B=1 window-(b)
    OOM, the frozen MemoryError, the leak pin
    (`pool.used_bytes() <= 8 MB` right after the caught error;
    pre-fix 22.6 MB) and the clean same-process recovery run.
14. **(nit) numpy-xp oracle extended to n_fold {2, 3, 6}**:
    `test_pipeline_numpy_xp_matches_cpu` parametrisation grew from
    {1, 4} x mirror to 8 rows incl. (2, False), (3, False),
    (3, True), (6, True) -- the pruning/sign family where a wrong
    `(-1)^(j+m)`/pruning interaction hides at n_fold 4 (review's
    adversarial grid measured worst rel err 1.33e-7, inside the
    band); 16 oracle tests total, all green.
15. **(nit) stale `NotImplementedError` branch dropped** from
    `_cupy_gpu_skip_reason` (dead failing-tests-stage code with a
    misleading reason).

**Recorded only (no code/test change, with reasons):**

16. **(minor, M02)** disabling the `m % n_fold` alpha-plane
    pruning in `_inverse_fft_batch` is a reviewed-EQUIVALENT
    mutant: the pruned planes are exact zeros, the irfft input is
    bitwise identical (verified by the review on device) -- a
    performance-shape mutant only, nominally guarded by the D12
    floor; no correctness test can kill it and none is added.
17. **(minor, M11)** rebuilding the sanitized Wigner table per
    chunk instead of `session.table` is reviewed-EQUIVALENT
    (bitwise, ~10 us-scale per ~20 ms device section); no
    performance-semantics pin warranted at this cost scale.  If
    the D7.2 residency discipline is ever to be pinned, a
    call-count spy on `_sanitized_table` (exactly one call per
    `index_patterns`) is the cheap option.
18. **(nit, M08b observation)** the odd-slP rows of the
    neighborhood parity tests are LOAD-BEARING for the non-compat
    glide shift sign (at even slP, +slp/2 = -slp/2 mod slp --
    structurally blind); the odd-slP parametrisation must never be
    dropped.  M09's killer ([True-32-17] clamp rows) confirmed
    load-bearing.  No action -- the rows exist.
19. **(nit, commit-stage note)** the CHANGELOG `Added` entry links
    PR #15 on the recorded next-free-number assumption -- CONFIRM
    (and edit if wrong) when the PR is opened.  The two never-touch
    files (`doc/tutorials/spherical_indexing.ipynb`,
    `specs/2026-08-16-constitution/upstream-issue.md`) carry
    working-tree modifications that PREDATE this phase (in the
    session-start git status) and must be EXCLUDED from the
    phase's commits.  Both were untouched by this fix stage.

**Rejected: none** (the un-adopted halves of two nits -- a gate
lock, an M11 call-count pin -- are declined with reasons in items
5 and 17).

**Run record after all fixes** (this machine, 2026-09-07; the
gated recipe now carries `KIKUCHIPY_EXPECT_GPU=1`):

- full gated overlay `-n 0 --weekly` (floor + weekly included):
  **103 passed, 0 failed, 0 skipped** (was 81; +22 = 14 default +
  7 gated + the canary).
- default `-n 0` (fork venv, cupy absent): **66 passed, 37
  skipped** (36 gated + the unarmed canary), skip reasons the
  stage-(a) message.
- default-suite spherical selection
  (`tests/test_indexing tests/test_signals -k "spherical" -n 4`):
  **3188 passed, 778 skipped, 0 failed** in 94 s -- the CPU
  reference path unperturbed by the fix-stage edits (D5.1).
- whole-suite collection: **4939 collected, no errors**.
- coverage union re-run: `_gpu.py` **100.00 %** (296 statements);
  `_indexer.py` union residual = pre-existing lines covered by
  other suite files only (see item 12).
- `uvx pre-commit run --files` on the four changed files: clean
  (ruff, ruff-format, licenseheaders).
- mutant-kill verification: M10 (16 failed), M12b (1 failed), M21
  (1 failed), M13 (canary failed, exit non-zero) -- each mutant
  applied to a scratchpad-backed working tree, run against the
  strengthened suite, then reverted; final tree re-verified green.

**D12 floor RE-CONFIRMED after the device-path edits** (the
lock-section `del` and the OOM-path changes touch device-path
code; quiet dedicated machine, serial run, junitxml-recorded):
`test_throughput_floor` rested re-run measured **940.7 pat/s**
best-of-3 against the pinned hardest idle CPU baseline 236.0
pat/s -> **3.99x the floor, PASS** (marginally above the
implementation-gate 901.4/917.6 -- the bounded residency did not
cost throughput).  The full-suite floor run in the same session
also passed.
