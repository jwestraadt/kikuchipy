# Phase 12 (provisional) -- `spherical-indexing-gpu`: plan

Branch `spherical-indexing-gpu` off `develop` (after the Phase 8
merge, jwestraadt/kikuchipy#13; the concurrent
`pseudo-symmetry-tutorial` branch is independent -- coordinate the
phase number, open question 9.1, before the spec commit). Models:
plan/spec on Fable 5 (xhigh, ultracode); tests, implementation,
adversarial review and fixes by Opus 5 agents (xhigh, ultracode).
Autonomous mode -- **with one carve-out**: this phase amends the
mission's scope table, so plan section 0 is put to the user for
explicit sign-off before the spec commit even if the
spherical-phases approval waiver otherwise applies (open question
9.2). Tests are written failing first. Drafting happened while a
tutorial-build workflow owned the CPU and the fork venv: the GPU
micro-benchmarks and CuPy probes RAN (ephemeral env, this machine's
RTX 2000 Ada) and are recorded in validation.md; nothing was
installed into the project venv, no project tests ran, and every
end-to-end tolerance stays measured-then-pinned (MTP) at the
tests/implementation gates.

## 0. Constitution amendments (DRAFTED here; applied in the spec commit at build -- never by the drafting session)

Each item names its exact target; the build stage applies them
verbatim-in-meaning and ticks roadmap boxes only as gates complete.

APPLIED 2026-09-07 (build stage): items 1-14 applied to
`specs/mission.md`, `specs/roadmap.md`, `specs/tech-stack.md` and
`specs/_research/plan-faithful-port.md` with every `<N>` filled as
**12** (open question 9.1; phases 0-11 exist, nothing else claimed
12), after the user's scope sign-off (open question 9.2); item 15
is the recorded no-op it declares.

1. `specs/mission.md:3`: "a pure-Python, **CPU-only**
   implementation" becomes "a pure-Python implementation ...
   (CPU-first; the CPU path is the reference implementation, with
   an optional CuPy GPU backend for the coarse correlation stage
   since Phase 12)".
2. `specs/mission.md:29`: the deliverables row "| CUDA/GPU indexer
   | **out of scope** -- CPU multi-threaded path only |" becomes
   "| CUDA/GPU indexer | optional `backend=\"gpu\"` (CuPy) device
   coarse-correlate inside `EBSD.spherical_indexing` /
   `SphericalIndexer` -- CPU remains the default and the parity
   oracle; full-pipeline GPU port and device refinement remain out
   of scope |".
3. `specs/mission.md:40` (success criterion 2): append "; the GPU
   backend is held to a recorded CPU-parity band (IQ bitwise,
   refined-to-refined misorientation and score bands measured then
   pinned, `specs/2026-09-07-spherical-gpu/`), so criterion 2
   continues to be defined against the CPU path".
4. `specs/mission.md:41` (success criterion 3): append "; CuPy is
   optional, never a required dependency, never imported at module
   scope, never installed on CI".
5. `specs/roadmap.md:1`: title becomes "# Roadmap: spherical
   indexing (EMSphInx port: CPU reference + optional GPU
   backend)".
6. `specs/roadmap.md:3`: chain gains "-> 12 (GPU backend; needs 6
   + 7 + 8)". New phase section "## Phase 12 --
   `spherical-indexing-gpu` (spec `2026-09-07-spherical-gpu`)"
   with gate-shaped boxes:
   (1) `_gpu.py` (three-stage CuPy gate + Windows DLL shim,
   xp-agnostic device pipeline for stages 4-6, VRAM batch model +
   OOM halving, per-call `_GpuSession`);
   (2) `backend="cpu"|"gpu"` on
   `SphericalIndexer`/`EBSD.spherical_indexing` + the
   `_index_chunk` batched restructure + `gpu_memory_per_batch_
   bytes` (CPU default bitwise-unchanged, pinned);
   (3) tests -- default-suite numpy-xp oracle + validation logic;
   local `cupy_gpu` gated parity/determinism/throughput suite
   (KIKUCHIPY_NO_GPU_TESTS kill switch; measured pins
   machine-specific: RTX 2000 Ada 8 GB);
   (4) performance go/no-go: GPU >= same-machine idle 8-worker CPU
   throughput on the `nickel_ebsd_large` route at bw 68, else a
   recorded negative result (projection band ~850-1500 pat/s
   refined, review-corrected, recorded as expectation, not gate);
   (5) adversarial review + fixes (incl. the device mutation
   list); (6) signed commits; PR opened (next free number; #15
   expected). Measured numbers enter the boxes as gates tick.
7. `specs/roadmap.md:80` (Phase 6 box): the sentence "bitwise
   deterministic across chunking/threads" gains "(CPU backend; the
   GPU backend of Phase 12 is deterministic run-to-run at fixed
   device/driver/batch size and tolerance-parity vs the CPU
   oracle, `specs/2026-09-07-spherical-gpu/` D5)".
8. `specs/tech-stack.md:16` (Runtime dependencies): append "cupy
   is likewise **optional and unregistered**: never imported at
   module scope in `src/`, never added to
   `_constants.deps_for_version_check` (the cupy-cuda12x dist name
   defeats `importlib.metadata.version("cupy")` -- measured), never
   installed on any CI job, gated at use by the three-stage probe
   in `_spherical/_gpu.py`; no `[gpu]` extra (the pyopencl
   precedent, `_constants.py:50-51`)".
9. `specs/tech-stack.md:21` (Numerics): the float64 sentence
   becomes "float64/complex128 throughout on the CPU path
   (EMSphInx `Real = double`); **the GPU backend (Phase 12) is
   the sanctioned float32/complex64 surface, scoped to the device
   coarse-correlate stages 4-6 only**, with CPU-parity tolerances
   measured then pinned in `specs/2026-09-07-spherical-gpu/`; a
   CPU float32 fast path remains a later, measured optimisation
   (unspent)."
10. `specs/tech-stack.md:39` (Performance): append "The GPU
    backend carries its own floor: >= the same-machine idle
    8-worker CPU throughput on the `nickel_ebsd_large` bw-68
    route, else the phase records a negative result (Phase 12
    D12); GPU numbers are machine-specific recorded baselines,
    never portable claims."
11. `specs/tech-stack.md:44` (parallelism): append "The dask rule
    governs the CPU stages; the GPU backend keeps `map_blocks` and
    the threaded scheduler for the host stages but serializes its
    device sections under one process-wide lock (one GPU consumer
    per process), forces `scheduler=\"threads\"` on its compute,
    and calls `cupy.fft` without the scipy-only `workers=1`
    kwarg."
12. `specs/tech-stack.md:50` (env vars / gating): append "GPU
    tests are availability-probed (the `cupy_gpu` fixture skips
    unless cupy imports, a CUDA device is present AND cuFFT
    loads, each with an instruction-bearing reason), skip
    structurally under xdist (`PYTEST_XDIST_WORKER` set -- the
    gated suite runs only at `-n 0` by construction, so default
    `-n N` runs stay GPU-silent even on cupy-capable machines),
    and honour the `KIKUCHIPY_NO_GPU_TESTS=1` kill switch -- the
    third local-gating convention; the wgpu `gpu` marker
    (`pyproject.toml:203-206`) is never reused for CUDA."
13. `specs/tech-stack.md` (Tests, docs, data -- coverage
    convention): append "Coverage for GPU-touching modules:
    device-independent logic (xp-agnostic pipeline core,
    validation, batch model, gate decisions) is covered by the
    default suite; cupy-touching lines are covered to 100 % by
    the local gated suite with the coverage command recorded in
    the phase's validation.md; `# pragma: no cover` only on
    import-guard branches (Phase 12 D11). The Codecov patch
    report on such a PR will show the cupy-gated lines uncovered
    -- the accepted, recorded outcome; the local recorded command
    is the gate."
14. `specs/_research/plan-faithful-port.md:16`: one-line dated
    addendum "(CUDA ban superseded for the optional CuPy backend
    by `specs/2026-09-07-spherical-gpu/` -- the 'justified in a
    spec' instrument of this same sentence; FFTW/pyfftw/shtns/
    pyshtools/rocket-fft remain forbidden)".
15. NOT amended, noted: `specs/2026-08-16-constitution/
    upstream-issue.md` ("no CUDA") is frozen and never edited; the
    messaging drift is recorded here and resolved when/if the
    upstream series opens (open question 9.4).
    `doc/tutorials/spherical_indexing.ipynb` is on the never-sweep
    list and gains no GPU section (requirements D13).

## 1. `_gpu.py`: gate, shim, session (tests: `tests/test_indexing/test_spherical_gpu.py`)

1. Module skeleton: GPL header + CMU/Lenthe third-party block +
   dated modification notice (requirements D13); module docstring
   mapping the device stages to `_xcorr.py` line ranges and the
   `feature/GPU` `include/gpu/pipeline.hpp` split; NO cupy import
   at module scope.
2. Three-stage gate `_verify_gpu_or_raise()` (D6.2): importable
   (+ `cupy >= 13` floor) -> device present -> cuFFT loadable;
   result cached; each failure message frozen at implementation
   (actionable remedies verbatim from D6.2); the Windows-only
   best-effort `os.add_dll_directory` shim over
   `site-packages/nvidia/*/bin` runs before stage (c), silent on
   failure/absence, no-op off Windows (D6.5).
3. `_GpuSession`: built per `index_patterns` call (D7.2); holds
   the device tables per phase (`A`, `A2` c64 with the frozen
   `(-1)^(j+m)` factor, built from the NaN-ZEROED pi/2 table
   copy; transposed sanitized table; `r_den` f32 only when
   `normalize=True` -- D2 stage-6), the batch buffers (zeroed at
   build, re-zeroed per phase per the D2 freshness rule), the
   cuFFT plan-bearing arrays, the process-wide lock handle, and
   `close()` releasing the pool. OOM halving at session build;
   mid-compute OOM aborts the `.compute()` and `index_patterns`
   rebuilds session + graph at B/2 (D8.4 two-window rule);
   non-OOM device errors propagate (D7.7).
4. xp-agnostic pipeline core (D11.1): pure functions taking
   `(xp, fft_ns, ...)` -- `_sanitized_table` (NaN -> 0 copy of the
   pi/2 table; D2), `_build_g_batch` (n < bw rows only -- the
   even-slP guard), `_spectrum_batch` (two GEMMs + mirror fill +
   zero pads + the per-phase buffer re-zero), `_inverse_fft_batch`
   (separable, `norm="forward"`, `m % n_fold` pruning),
   `_scale_argmax_batch` (multiply skipped when
   `normalize=False`), `_gather_neighborhoods` (device `take`
   at host offsets), `_pad_batch`/`_strip_padding`. Under numpy
   these run the identical code path the device runs under cupy.
5. Host offsets helper `_neighborhood_offsets(flat_idx, bwP, slP,
   emsphinx_compatible)` porting the `_extract_neighborhood` index
   arithmetic incl. the per-slot glide and even-slP clamp defect
   (`_xcorr.py:196-211, 730-746`); parity unit test is the killer
   for its mutants (plan 7).
6. VRAM model `g(bw)` + default-B chooser (D8.2) as pure
   functions; MTP calibration constants named and marked.

## 2. Indexer + signal integration (tests: additions to `test_spherical_gpu.py`, `test_spherical_indexer.py`, `test_ebsd_spherical_indexing.py`)

1. `SphericalIndexer`: `backend` keyword (validation ValueError;
   gate fired at ctor for `"gpu"`; D1), `gpu_memory_per_batch_
   bytes` (D8.3), chunksize semantics for the GPU path (chunk ==
   batch, `_batch_estimate` bypassed, default from the VRAM
   model; D8.1-2).
2. `_index_chunk` restructure under a session argument (D7.1):
   host stages 0-3 batched collection (gln c128 + iq + failure
   mask), per-phase device batch under the lock, host stages 7-9
   per pattern in original order; failed patterns excluded from
   the batch and filled identically (D7.4); padded tail computed
   and discarded (D7.3); device-section exceptions propagate --
   never swallowed into per-pattern fill rows (D7.7).
3. **Epilogue extraction in `_xcorr.py`** (review-added
   deliverable, D2 stage-7): the `correlate()` peak epilogue --
   flat-index -> (k, n, m) decomposition, `_interpolate_maxima`,
   the `|x[0]|`-twice compat bounds bug, step rejection, the zyz
   grid formula (`_xcorr.py:2467-2492`) -- extracted into a
   shared neighborhood-fed entry point; `correlate()` calls it
   (CPU semantics bitwise-guarded); the GPU path feeds it the
   device-gathered 27 values + index; both paths seed
   `refine_zyz` from the INTERPOLATED triple, as `correlate()`
   does today (`_xcorr.py:2563-2568`).
4. `index_patterns`: `_GpuSession` lifecycle around the dask
   compute; compute-level OOM catch -> session + graph rebuild at
   B/2 (D8.4 window b); `scheduler="threads"` forced on the GPU
   path (D7.5); verbose info message extended with the
   device/VRAM/batch line + the VRAM warning (D8.3).
5. `EBSD.spherical_indexing`: `backend` threading (placement per
   D1 -- positional-or-keyword append, review-corrected),
   docstring Notes (float32 coarse stage, determinism
   scoping, VRAM model, no silent fallback, score-sensitivity
   note, optional-dependency sentence), chunksize doc updated for
   the GPU meaning.
6. Module doc `_indexer.py:231-241`: determinism sentence
   re-scoped per backend (mirrors plan 0.7).
7. `refine_patterns`/`refine_orientation_spherical`: untouched
   (recorded in requirements Scope).

## 3. Failing tests -- default suite (no GPU; run on CI)

Written failing first (Opus 5), with MTP placeholders marked:

1. `test_backend_validation_raises` (frozen message shape);
   `test_backend_cpu_is_bitwise_default` (backend="cpu" ==
   no-kwarg call, bitwise, on `nickel_ebsd_small` -- the D1/D5.1
   guard).
2. `test_gate_stages_actionable` (monkeypatched cupy import /
   device count / fft probe failures -> the three frozen messages,
   each naming its remedy); `test_dll_shim_is_silent`
   (non-Windows / missing nvidia dir -> no raise);
   `test_gate_version_floor`.
3. `test_pipeline_numpy_xp_matches_cpu`: bw 16-24 synthetic
   spectra + a real Ni bw-32 case, xp=numpy end-to-end stages 4-6
   vs `SphericalCrossCorrelator`/`Normalized...` -- identical
   argmax cell, cube values to f32 tolerance (MTP), both n_fold 1
   and 4, both mirror flags -- the CI-side correctness anchor
   (D11.1).
4. `test_neighborhood_offsets_parity`: vs `_extract_neighborhood`
   on random cubes, odd and even slP, BOTH `emsphinx_compatible`
   settings, edge/wrap cells enumerated (D2).
   `test_peak_epilogue_parity` (review-added): the extracted
   neighborhood-fed epilogue vs `interp_peak` directly on random
   cubes, both compat settings, incl. planted step-rejection
   cases (the `|x[0]|`-twice bounds branch must be hit, not
   masked by Newton reconvergence -- D2 stage-7).
   `test_argmax_first_occurrence_tie` (xp=numpy planted tie; the
   cupy twin lives in the gated suite -- numpy proves nothing
   about `cp.argmax`, review-corrected).
5. `test_padding_and_stripping` (planted peaks; padded rows never
   reach results; n % B != 0 map shapes); `test_failed_patterns_
   excluded` (ptp==0 + masked -> fill rows, batch shrinks).
6. `test_vram_model_and_default_batch` (pure-math pins; clamp
   bounds; override by chunksize); `test_gpu_memory_helper`.
7. `test_kill_switch` (fixture logic honours
   KIKUCHIPY_NO_GPU_TESTS); `test_xdist_worker_skip` (fixture
   skips with `PYTEST_XDIST_WORKER` faked set -- the structural
   `-n 0` rule, D10.4 review-added).
8. Docstring/signature pins: placement after
   `pseudo_symmetry_ops`; frozen-defaults dict gains
   `backend="cpu"`.

## 4. Failing tests -- local `cupy_gpu` gated suite (this machine)

All in `test_spherical_gpu.py`, `-n 0`, machine-specific pins
commented as such (D10):

1. `test_gate_passes_here` (three stages green; records
   device/driver/cupy versions into the assertion message).
2. `test_iq_bitwise_vs_cpu` (D4.1).
3. `test_refined_parity_small` / `test_refined_parity_large_20pt`
   (+ 165-pt weekly): winner agreement, refined-to-refined
   misorientation median/max, score Pearson + rel |diff| -- all
   MTP (D4.2-5); `test_coarse_cell_agreement` (D4.3, recorded).
4. `test_refine_false_coarse_scores` (rel |diff| MTP, D3.4/D4.5);
   `test_unnormalised_gpu_parity` (review-added:
   `normalize=False` dual-backend -- no `r_den` resident, multiply
   skipped, plain `_find_peak` parity; D2 stage-6/D9).
5. `test_emsphinx_compatible_toggle_parity` (both settings, D2).
6. `test_psym_dual_backend` (90-deg-z op, n_best=2: indices,
   orientations, scores parity; D9).
7. `test_multiphase_dual_backend` (sign-scrambled second phase:
   winner phase parity; also kills cross-phase table mix-ups);
   `test_multiphase_mixed_symmetry_dual_backend` (review-added:
   second phase with DIFFERENT symmetry, e.g. the scrambled
   master re-declared n_fold=1/no mirror -- kills stale `fxc`
   columns across phases, which the same-zero-structure
   sign-scramble case cannot; D2 freshness rule).
8. `test_run_to_run_bitwise` (D5.2);
   `test_batch_size_invariance` (B in {8, 32, default}: bitwise
   if measured so, else pinned tolerance + recorded deviation,
   D5.3); `test_argmax_tie_gpu` (review-added: planted f32 tie
   and all-equal cube on device -- `cp.argmax` first-occurrence
   pinned against regression/portability; probe-confirmed today).
9. `test_oom_halving` (cupy pool limit forced small: (a) at
   session build -> B halves, completes; (b) mid-compute -> the
   compute aborts and `index_patterns` rebuilds at B/2, completes,
   results entirely from the final B; limit below the B=1 need ->
   MemoryError message pin; D8.4 two-window rule);
   `test_device_error_propagates` (review-added: monkeypatched
   non-OOM device-stage exception -> the run FAILS, no fill rows
   emitted; D7.7).
10. `test_lazy_input_eager_output`, `test_verbose_gpu_info_line`.
11. `test_throughput_floor` (D12.1; runs only with the idle-machine
    baseline recorded -- skip with reason until then).
12. Coverage run for `_gpu.py` (default + gated union = 100 %),
    command recorded in validation.md.

## 5. Implementation order

1. Module 1 (`_gpu.py` core + gate) against test groups 3.2-3.7.
2. Module 2 (indexer/signal integration) against 3.1, 3.8 and the
   gated wiring tests.
3. First gated parity measurement pass -> fill every MTP
   placeholder (D3.2 both-ways spectrum-precision measurement, D4
   bands, D5.3 batch invariance, D8 VRAM calibration + bw-113
   probe) -> pin.
4. Idle-machine baseline pair + throughput floor (D12; scheduled
   with the user, open question 9.5).
5. Docs (docstrings, installation.rst, CHANGELOG), module-doc
   re-scope.

## 6. Deliberately not built (recorded)

- Dedicated GPU feeder thread/stream pipeline (fallback if the
  D7.1 lock is measured to convoy -- trigger: device-section wait
  time > ~20 % of wall time at 8 workers; recorded, not built).
- Fused ElementwiseKernel mirror fill and cub-based fused
  multiply-argmax (each ~0.1 ms/pattern potential; only if the D12
  floor needs them -- measured decision).
- Pinned-memory H2D staging (transfers are ~2-10 us/pattern at the
  D2 split; not worth complexity).
- On-device refinement, full-pipeline port, multi-GPU, auto
  backend, `[gpu]` extra, tutorial section (requirements
  non-goals).

## 7. Adversarial review and fixes

1. Fidelity reviewer refutes against: `_xcorr.py` stages 4-6
   (spectrum symmetry/sign structure `:389-560`, separable FFT
   `:1736-1809`, `_find_peak` `:580-599`, `_extract_neighborhood`
   `:642-763`) line by line vs the xp-agnostic core -- NOTE: the
   spec-stage adversarial review already ran the stage-4
   re-derivation with live probes (bw 6, ephemeral env) and it
   CAUGHT and fixed two frozen-text errors (the `(-1)^j`-only
   `A2` factor, corrected to `(-1)^(j+m)`; the missing table
   NaN-zeroing) -- the build-stage reviewer re-runs it against
   the code, not the spec; the stage-7 extracted epilogue vs
   `interp_peak`/`correlate()` (`:2457-2492, 2563-2568`); the D2
   compat-neutrality claim (grep: no `emsphinx_compatible` branch
   inside `_gpu.py` device code paths); the D7 restructure vs
   `_index_chunk` insertion/failure semantics; the D6 gate vs the
   probe-measured failure modes; the plan 0 amendments vs every
   collision listed in requirements Context.
2. Bug-injection (mutation) list -- every mutant dies by a named
   test or is recorded reviewed-only with its killer stated:
   - FFT `norm="backward"` (or missing) on any of the three
     transforms: dies by `test_pipeline_numpy_xp_matches_cpu`
     (values off by slP factors) and every gated parity test.
   - slice-to-bwP off-by-one after the k-axis ifft: dies by the
     numpy-xp oracle (cube values) + refined parity.
   - `irfft` output length != slP: shape asserts + numpy-xp test.
   - conj dropped in the `G` build / `G2` conj added: numpy-xp
     oracle.
   - `A2` seed miswritten as `(-1)^j` instead of the frozen
     `(-1)^(j+m)` (the pre-review drafted error -- INVISIBLE at
     even n_fold): numpy-xp oracle at n_fold=1; final `(-1)^k`
     dropped; `(m+n)`-parity mirror sign flipped; mirror written
     to unflipped views: numpy-xp oracle (each perturbs the cube
     asymmetrically).
   - table NaN slots not zeroed before the GEMM build: all-NaN
     cube, self-revealing at the first numpy-xp oracle run (D2
     sanitization step).
   - `G` rows built for `n >= bw` at even slP / `m = bw` column
     written: numpy-xp oracle at bw 16 (slP 32, bwP 17).
   - `fxc` batch buffer not re-zeroed per phase (stale columns
     across differing n_fold/mirror):
     `test_multiphase_mixed_symmetry_dual_backend` (the
     sign-scramble twin CANNOT catch it -- same zero structure).
   - `m % n_fold` pruning applied at n_fold=1 (zeroes real
     planes): numpy-xp oracle at n_fold=1.
   - pruning uses the WRONG phase's n_fold in a multi-phase run:
     `test_multiphase_dual_backend`.
   - `r_den` multiply dropped / wrong phase's `r_den`: gated
     normalized parity scores + multiphase test.
   - `r_den` multiplied on the un-normalized path (or `r_den`
     resident built with `normalize=False`):
     `test_unnormalised_gpu_parity`.
   - epilogue seeds `refine_zyz` from the coarse cell instead of
     the interpolated triple; wrong step-bound branch:
     `test_peak_epilogue_parity` (planted step-rejection cases) +
     refined parity.
   - device-section exception swallowed into per-pattern fill
     rows: `test_device_error_propagates`.
   - argmax taken over the padded rows / unstripped tail:
     `test_padding_and_stripping` (planted peak in the pad).
   - `cp.argmax` replaced by a last-occurrence variant: numpy-xp
     tie fixture (two equal maxima, first must win --
     `_find_peak` strict-`>` parity).
   - offsets helper: glide sign flipped, even-slP clamp dropped,
     compat flag ignored: `test_neighborhood_offsets_parity`
     (both settings, even+odd slP).
   - f32 cast moved ahead of the D3.2 decision point: killed by
     the recorded both-ways measurement if measurable; else
     reviewed-only with the measurement cited.
   - failed pattern included in the device batch:
     `test_failed_patterns_excluded` (fill-row bitwise pin).
   - OOM halving loop non-terminating / pool not freed:
     `test_oom_halving`.
   - gate stage order swapped (fft probe before import):
     `test_gate_stages_actionable` (message per stage).
   - DLL shim raising on Linux/absent dir: `test_dll_shim_is_
     silent`.
   - device lock dropped: killed by a 4-worker gated stress run of
     `test_run_to_run_bitwise` if it bites; else recorded
     reviewed-only (single-consumer design makes the race
     window small).
   - `scheduler="threads"` not forced: reviewed-only + a
     monkeypatch assert on the compute kwargs.
   - backend="cpu" path perturbed by shared-code edits:
     `test_backend_cpu_is_bitwise_default` + the entire existing
     spherical suite.
   - `_GpuSession` stored on the indexer (pickling/graph hazard):
     reviewed-only, killer = code inspection + a
     no-device-attrs assert test.
3. Conventions reviewer: GPL/CMU blocks, no module-scope cupy
   import (grep), no new public export beyond the keyword +
   helper, docstring optional-dep wording, no phase number in
   public docstrings, `# pragma: no cover` only on import guards,
   message style, CHANGELOG, the EMSphInx-gotcha checklist
   (`specs/_research/explore-emsphinx-core-algorithm.md` section
   8) for the host-side touch points.
4. Fix, re-run (`-n 0` gated suite; default suite `-n 4`),
   `pre-commit run --files <changed>` (never `specs/`), coverage
   per D11, re-measure the review-critical numbers (D4 bands,
   D5.3, D12 floor).
5. **Re-submit all three spec documents to adversarial review**
   (definition-of-done gate, validation.md).

## 8. Open questions -- decided at drafting (autonomous mode), flagged for review

1. **Lock-guarded device section inside `_index_chunk`** over a
   dedicated feeder thread (D7.1): keeps every dask/eager
   contract, minimal diff; host stages overlap, device sections
   serialize at ~20 ms granularity. Fallback trigger recorded
   (plan 6).
2. **No `[gpu]` extra** (D6.4): the pyopencl precedent +
   measured wheel fragmentation; deviates from the drafting
   brief's phrasing -- ALSO raised to the user (9.3).
3. **float32/complex64 device stages, no public dtype knob**
   (D3): the 6.6x-slower all-double prior decides it; the
   refined-to-refined oracle absorbs the discrete risk;
   `backend="cpu"` is the escape hatch.
4. **Pad the last batch to B** (D7.3) over a second plan set:
   one plan set, uniform batch shape (feeds the D5.3
   measurement), bounded waste.
5. **`KIKUCHIPY_NO_GPU_TESTS` kill switch added** (D10.3): cheap,
   and this very machine demonstrates the need (concurrent GPU
   job).
6. **In-code Windows DLL shim** (D6.5) over docs-only: the
   failure it prevents is cryptic (`DLL load failed while
   importing cufft` after a SUCCESSFUL device probe, measured)
   and the shim is 10 guarded lines.
7. **Chunk == GPU batch; `_batch_estimate` bypassed** (D8.1):
   one number to reason about; the CPU model's 15/chunk at bw 68
   is below the efficient device batch.
8. **No silent CPU fallback** on unavailability or OOM (D1/D8.4):
   backend choice is result-affecting; errors are actionable
   instead.
9. **xp-agnostic pipeline core** (D11.1): buys CI coverage of the
   device MATH (not just mocks) and a numpy-vs-cupy A/B debugging
   lever; slight abstraction cost accepted.
10. **cupy >= 13 floor** (D6.2): NumPy-2-compatible line; only
    14.2.0 is tested here, recorded as machine-specific.
11. **Gate fires at ctor** (D1.3), session built per
    `index_patterns` call (D7.2): fail fast + no device state on
    the shared indexer.
12. **Branch/folder mismatch recorded** (`spherical-indexing-gpu`
    vs `2026-09-07-spherical-gpu`), the Phase 8 precedent; PR
    number recorded at open time.

Decided at the spec adversarial-review fix stage (2026-09-07,
same autonomy; the review's own probes are the evidence):

13. **`normalize=False` supported under `backend="gpu"`** (skip
    the multiply, no `r_den` residents) over raising: it is
    public API, D9 mirrors the un-normalized psym precedent, the
    numpy-xp oracle already names both correlators, and the
    implementation is one conditional (D2 stage-6).
14. **Mid-compute OOM -> abort + whole-run rebuild at B/2** over
    device sub-batching inside a chunk: preserves chunk == batch,
    the one-plan-set rationale and the D5.3 uniform-batch-shape
    argument; recompute waste is bounded and exceptional (D8.4).
15. **Structural xdist skip** (`PYTEST_XDIST_WORKER`) over a
    marker/flag opt-in or a documented kill-switch workflow: makes
    the `-n 0` rule unforgettable and keeps the default `-n 4`
    command GPU-silent on GPU dev machines (D10.4).
16. **Stage-7 epilogue extraction into a shared neighborhood-fed
    entry point** over duplicating the epilogue in `_gpu.py`:
    single source of truth for the compat/step/zyz defects,
    bitwise-guarded on the CPU path (D2 stage-7).
17. **Device-section exceptions fail the run** over per-chunk fill
    rows: fill rows would silently discard a whole chunk --
    the silent-failure mode the no-fallback rule exists to forbid
    (D7.7, D14.8).

## 9. Open questions FOR THE USER (not decided autonomously)

1. **Phase number**: 12 or 13? The concurrent
   `pseudo-symmetry-tutorial` workflow has no roadmap section yet
   and may claim 12. Numbers are load-bearing
   (`roadmap.md:12-13`); every `<N>` in plan 0 is filled at the
   spec commit after coordination.
   **ANSWERED 2026-09-07: 12.** Phases 0-11 exist and nothing
   else has claimed 12 (the concurrent branch took no roadmap
   section); every `<N>` in plan 0 was filled with 12 when the
   amendments were applied.
2. **Scope-amendment sign-off**: does the autonomous-mode waiver
   ("approval gate waived for spherical-indexing phases") cover
   amending the mission's scope table (CPU-only -> optional GPU
   backend)? Provisional position: yes, but plan section 0 is
   explicitly put to you before the spec commit. Confirm or
   review.
   **ANSWERED 2026-09-07: APPROVED by the user.** The scope
   amendment (CPU-only -> CPU-default + optional CuPy backend) is
   signed off; the plan-0 application of this date is the result.
3. **Extras group**: confirm the no-`[gpu]`-extra decision (D6.4)
   or mandate an extra (if mandated: name and contents, knowing a
   `cupy-cuda12x` pin mis-serves CUDA 11/13/ROCm users).
   **ANSWERED 2026-09-07: NO `[gpu]` extra, confirmed by the
   user.** D6.4 stands.
4. **Upstream messaging**: when the upstream PR series opens, does
   the GPU backend ship in it (pitch rewrite drafted fresh at that
   time -- `upstream-issue.md` itself is never edited) or stay
   fork-only until the maintainers react? Provisional: fork-only.
   *2026-09-07: not decided by the user; the provisional default
   (fork-only) stands.*
5. **Idle-machine baseline scheduling**: the D12 floor needs a
   clean same-machine pair (8-worker CPU baseline + GPU run) with
   both concurrent workflows quiet and the GPU free. Say when the
   machine can be dedicated (~15 min).
   **ANSWERED 2026-09-07: build authorized by the user**,
   including the ~15 min dedicated-machine window for the D12
   idle baseline pair.
6. **f64 diagnostic mode**: keep the internal-only float64 device
   path used by the parity measurement scripts private
   (provisional), or expose a public knob for score-sensitive
   `refine=False` workflows?
   *2026-09-07: not decided by the user; the provisional default
   (internal-only, no public knob) stands.*
7. *(minor)* Should the installation bullet and the stage-(c)
   error text name `nvidia-cufft-cu12`/`nvidia-cublas-cu12`
   explicitly as the Windows pip remedy (provisional: yes -- it is
   the measured fix on this machine)?
   *2026-09-07: not decided by the user; the provisional default
   (yes, name them) stands.*

## 10. Commit and PR

Signed commits in gate order on `spherical-indexing-gpu`:
(1) spec + the section-0 amendments (AFTER open questions 9.1/9.2
resolve); (2) failing tests (MTP placeholders marked, gated tests
skipping); (3) implementation + measured pins + CHANGELOG;
(4) review fixes + re-measurements. Never touch
`doc/tutorials/spherical_indexing.ipynb`,
`specs/2026-08-16-constitution/upstream-issue.md`, or the
concurrent tutorial branch's files. Push; PR (next free number,
#15 expected) into fork `develop` with the template, the GPL
statement up front (CMU/Lenthe block carried; BSD opt-out
impossible), the machine-specific-pins note, and the D12
floor/negative-result status in the description.
