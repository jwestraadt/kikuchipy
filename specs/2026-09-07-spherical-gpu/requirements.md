# Phase 12 (provisional) -- `spherical-indexing-gpu`: requirements

Branch `spherical-indexing-gpu` off `develop` (after the Phase 8
merge, jwestraadt/kikuchipy#13). Spec folder
`specs/2026-09-07-spherical-gpu/` (folder-vs-branch name mismatch
recorded, the Phase 8 precedent). **The phase number is provisional
(open question U1, plan 9.1)**: the next free roadmap number is 12,
but a concurrent workflow (branch `pseudo-symmetry-tutorial`,
`doc/tutorials/pseudo_symmetry.ipynb`) has no roadmap section yet and
may claim it; phase numbers are load-bearing (`roadmap.md:12-13`), so
the number is fixed at spec-commit time, before anything cites it.

The phase adds an **optional CuPy GPU backend** for the coarse
correlation stage of spherical indexing: `backend="gpu"` on
`SphericalIndexer` and `EBSD.spherical_indexing` runs the
per-pattern hot path's stages 4-6 (cross-correlation spectrum
product, separable inverse FFT, normalise + argmax + neighborhood
extraction) as float32/complex64 device batches, leaving every other
stage -- preprocessing, back-projection, SHT analysis, tri-quadratic
peak interpolation with its `emsphinx_compatible` defects,
positive-score insertion, Newton refinement, the pseudo-symmetry
variant loop -- on the CPU, byte-for-byte unchanged. **The CPU path
remains the default, the reference implementation, and the parity
oracle for every GPU output.**

**Drafting basis (2026-09-07):** unlike Phase 8, this spec IS
backed by fresh measurements -- two research sessions ran live GPU
micro-benchmarks and CuPy probes on this machine the same day
(scratchpad `gpu_bench.py`, `cupy_probe.py`, `profile_hotpath.py`;
the scratchpad does not survive, so every load-bearing number is
carried inline below, the Phase 8 F25 convention). Two caveats:
(a) all GPU numbers are **machine-specific** -- NVIDIA RTX 2000 Ada
Generation Laptop GPU, compute capability 8.9, 8188 MiB VRAM
(7.45 GB free at bench start), driver 595.71 (CUDA 13.2), CuPy
14.2.0 on CUDA runtime 12.9, Windows 11 -- and every GPU pin in
validation.md is recorded as such, never asserted as portable;
(b) the CPU stage profile was taken while a tutorial-build workflow
owned part of the machine, so CPU absolute times are upper bounds
(they land inside the recorded band 63.8-77.6 pat/s of
`specs/2026-09-02-spherical-indexing-ebsd/validation.md:748-756`).
End-to-end parity/performance tolerances remain
measured-then-pinned (MTP) at the tests/implementation gates.

**Constitution-amendment caveat:** this phase amends the mission's
scope table (CPU-only -> optional GPU backend), the float64 rule and
the determinism pledge. The amendments are DRAFTED in plan section 0
and applied only at the build stage; whether the autonomous-mode
waiver stretches to a scope-amending phase is open question U2 --
the provisional position is that it does, with the plan flagged for
explicit user review before the spec commit.

## Scope

In scope:

- **`src/kikuchipy/indexing/_spherical/_gpu.py`** (new, private):
  the three-stage CuPy availability gate with actionable errors
  (D6), the Windows DLL-directory shim (D6.5), the xp-agnostic
  device coarse-correlate pipeline -- per-phase resident tables
  `A`/`A2`/`r_den`/transposed pi/2 Wigner table, two batched
  complex64 GEMMs + elementwise mirror fill for the spectrum
  product, separable inverse FFT with the `m % n_fold` alpha-plane
  pruning, fused scale + first-occurrence argmax, 27-neighborhood
  gather (D2, D3) -- the VRAM batch model + OOM halving (D8), and a
  per-`index_patterns`-call device session object (D7).
- **`backend: str = "cpu"`** parameter -- keyword-only on the
  `SphericalIndexer` ctor (which has a keyword-only group);
  positional-or-keyword append on `EBSD.spherical_indexing`
  (after `pseudo_symmetry_ops`, before `chunksize`; the method has
  no bare `*` -- D1, review-corrected), accepting `"cpu"` and
  `"gpu"` (D1).
- **`_index_chunk` restructure under `backend="gpu"`**: host stages
  0-3 for the whole chunk, one lock-guarded device batch per phase
  for stages 4-6, host stages 7-9 per pattern in the original order
  (D2, D7). Chunk size doubles as the device batch size (D8).
- **Epilogue extraction in `_xcorr.py`** (review-added): the
  `correlate()` peak epilogue (index decomposition +
  `_interpolate_maxima` + compat bounds/step handling + zyz grid
  formula) extracted into a shared neighborhood-fed entry point
  used byte-identically by both backends (D2 stage-7 block);
  CPU-path semantics bitwise-guarded.
- **`SphericalIndexer.gpu_memory_per_batch_bytes(batch_size)`**
  model helper -- a METHOD, not a property; the
  `memory_per_worker_bytes` counterpart in spirit only (D8.3,
  review-clarified).
- **Tests**: default-suite tests that need no GPU (validation, the
  numpy-xp pipeline oracle, batch/padding/offset logic, gate
  decision logic with a faked cupy handle) plus a local-gated
  `cupy_gpu` suite (parity oracle, determinism, throughput floor)
  with the `KIKUCHIPY_NO_GPU_TESTS` kill switch (D10, D11).
- **Docs**: method docstrings with the optional-dependency wording,
  an `installation.rst` optional-dependency bullet, CHANGELOG entry
  (user-facing -- required). No tutorial change (D13).
- **Constitution amendments** drafted in plan section 0, applied at
  build.

Out of scope (honest non-goals, each with the recorded reason):

- **On-device Newton refinement** (the `feature/GPU`
  `refinePeakBatchDevice` analogue): refinement is float64,
  serial, convergence-dependent, rebuilds a `(bw,bw,bw,2)` buffer
  per iteration, and `_derivatives` is a 930-line scalar-recurrence
  kernel; Ada fp64 runs at 1/64 rate. Measured host cost 1.38
  ms/pattern at bw 68 (~10 % of the CPU budget) parallelises across
  dask workers already. Deferred; the pseudo-symmetry op count at
  which host refinement becomes the bottleneck is measured and
  recorded (D9), not acted on.
- **Full-pipeline GPU port** (preprocess/back-project/SHT/refine on
  device, the EMSphInx `feature/GPU` end state): v1 risk/benefit
  favours the narrow stage -- the coarse correlate is ~85-90 % of
  the 13-14 ms/pattern CPU budget at bw 68, and every frozen
  EMSphInx quirk stays in already-reviewed CPU code. Revisit later.
- **`backend="auto"`** / dispatch-on-availability: a GPU backend is
  result-affecting (float32 cube, different FFT), and
  result-affecting choices are explicit per-call keywords
  (`tech-stack.md:36`); the Hough path's silent
  pyopencl-if-installed dispatch (`ebsd.py:1680-1682`) is the
  cautionary precedent, not the model. Revisit only after the
  parity tolerances of D4 are established and stable.
- **Multi-GPU**: single-device scope; the batching model scales
  linearly across devices in principle, recorded as one Notes
  sentence with no claim tested.
- **Cross-device/driver reproducibility assertions**: recorded, not
  asserted (D5).
- **cupy-backed dask arrays** and **concurrent multi-worker device
  execution**: one GPU consumer per process, device sections
  serialized under a lock (D7).
- **GPU for anything else**: Hough indexing (pyebsdindex/pyopencl
  own that), dictionary indexing, NCC refinement, and
  `find_pseudo_symmetry_operators` (a one-shot master-vs-master
  scan, not per-pattern) all stay CPU.
- **A pyproject `[gpu]` extra**: deliberately NOT added (D6.4;
  flagged to the user, open question U3, since the drafting brief's
  phrasing assumed one).
- **A public float64 device mode**: no dtype knob in v1;
  `backend="cpu"` is the full-precision escape hatch (D3.4;
  open question U6).
- **Tutorial/notebook GPU section**: the Read the Docs builder has
  no GPU so the cells could never execute there, and
  `doc/tutorials/spherical_indexing.ipynb` carries uncommitted user
  edits and is on the never-sweep list. Docs scope is docstrings +
  installation bullet only (D13).
- **Upstream messaging changes**: the frozen
  `specs/2026-08-16-constitution/upstream-issue.md` pitch ("no
  CUDA, no new required dependencies") predates this phase, remains
  accurate for the CPU PR series it describes, and is NEVER edited;
  the GPU backend stays fork-only until the maintainers react
  (open question U4).
- Any change to `refine_patterns` /
  `EBSD.refine_orientation_spherical` (refinement is host-side in
  both backends; no `backend` keyword there in v1, recorded).

## Decisions

### D1 -- The backend switch: explicit per-call keyword (frozen)

`backend: str = "cpu"` -- **keyword-only on the `SphericalIndexer`
ctor** (after `pseudo_symmetry_ops` in the existing keyword-only
group, `_indexer.py:1272-1286`); on `EBSD.spherical_indexing` a
**plain positional-or-keyword append** after
`pseudo_symmetry_ops`, before `chunksize` (review-corrected: the
method has NO bare `*` -- all 15 parameters are
positional-or-keyword, `ebsd.py:1997-2014` -- so keyword-only
there would require inserting `*`, breaking positional
`chunksize`/`verbose` callers; the insertion's shift of the
`chunksize`/`verbose` positional slots is recorded, with no such
positional call in the repo or tests -- exactly the Phase 8 D4
precedent: ctor keyword-only, signal-method plain append,
placement deviation recorded, `specs/2026-09-06-pseudo-symmetry/
requirements.md:406-412`).
Accepted values `"cpu"` and `"gpu"`; anything else raises
`ValueError` naming the accepted set (the
`SUPPORTED_OPTIMIZATION_METHODS` message shape,
`_refinement/_refinement.py:1084-1088`).

Rationale, frozen:

1. Result-affecting choices are explicit per-call keywords, never a
   module global or env var (`tech-stack.md:36`; numba freezes
   globals) and never silent dispatch. Default `"cpu"` keeps every
   bitwise pin, the Phase 10 regression suite, and the Phase 6
   determinism pledge untouched -- a `backend="cpu"` call is
   bitwise identical to a call without the keyword (pinned).
2. The nlopt method-string gate (`_refinement.py:1081-1097`) is the
   in-repo precedent for an optional-dependency-backed mode behind
   an explicit per-call choice.
3. The three-stage availability gate (D6) fires in
   `SphericalIndexer.__init__` when `backend="gpu"` -- fail fast,
   the "Getting an indexer" precedent (`_hough_indexing.py:168`).
   Constructing with `backend="cpu"` never touches cupy.
4. No separate class/method (`SphericalIndexerGPU`): doubles the
   API surface, no repo precedent.

### D2 -- The device stage split and compat neutrality (frozen)

Device work is **stages 4-6 only** of the frozen per-pattern
pipeline (`_indexer.py:144-157`); everything else runs the existing
host code unchanged (with one review-added exception: stage 7's
epilogue is EXTRACTED into shared host code with byte-identical
semantics -- the stage-7 block below):

| Stage | Where | Why |
|---|---|---|
| 0 guards, 1 preprocess, 2 back-project, 3 SHT analyze | host, unchanged | 0.85 ms/pattern total; carries the Gaussian-fit `emsphinx_compatible` switch and the bitwise IQ |
| 4 xcorr spectrum | device | measured 1.76 ms CPU (12.3 %); maps to 2 batched c64 GEMMs + elementwise fills |
| 5 inverse FFT | device | measured 8.83 ms CPU (61.8 % at bw 68; 65.7 % at 88) -- THE target; 18-42x measured on device |
| 6 normalise + argmax (+ 27-neighborhood gather) | device | measured 1.13 ms CPU; 0.095-0.125 ms device |
| 7 peak interpolation, 8 Newton refine, 9 insertion | host; shared epilogue, semantics byte-identical (extraction below) | carry ALL remaining `emsphinx_compatible` defects + the unconditional `vPeak` defect; refine must stay float64 |

Stage-4 device formulation (no custom CUDA kernel is mandatory;
measured shapes at bw 68 in parentheses):

- **Table sanitization (review-added, load-bearing)**: the shared
  f64 pi/2 Wigner table is NaN in every undefined slot
  `j < max(k, m)` by validated contract (`_xcorr.py:2213-2215`;
  `_xcorr_spectrum` doc `:415-416`; probe: 125/216 slots at bw 6).
  The CPU loops never read those slots (their j-loops start at
  `max(m, k)` / `max(k, n)`, `_xcorr.py:471, 489`); the dense
  GEMMs below read ALL of them, and `0 * NaN = NaN`, so a GEMM on
  the raw table is ~99.5 % NaN. Every device-resident table
  product (`A`, `A2`, `table_T`) is therefore built from a
  **NaN-zeroed copy** of the table. Probe-verified (review
  session, bw 6): the zeroed slots plus the upper-triangular
  `flm`/`gln` zeros reproduce the CPU j-range restriction
  `start = max(k, n, m)` exactly -- the GEMM then equals the CPU
  loop bitwise.
- Per phase, resident: `A[k,j,m] = flm[m,j] * table[m,k,j]` and
  **`A2[k,j,m] = (-1)^(j+m) A[k,j,m]`**, complex64 (2.51 MB each;
  5.45 at bw 88), built once per `index_patterns` call from the
  host c128 `flm` and the NaN-zeroed f64 table, cast once (D3.2).
  **The `(-1)^(j+m)` factor is frozen and review-corrected**: the
  CPU kernel seeds its negated-sum toggle from `(start + m) % 2`
  (`_xcorr.py:504-541`), so the per-term sign is `(-1)^(j+m)`, not
  the drafted `(-1)^j`; the negated fill slot
  `fxc[slP-k, slP-n, m]` carries no fill sign that could restore a
  missing `(-1)^m` (`_xcorr.py:543-555`). Probe-verified against
  `_xcorr_spectrum.py_func` at bw 6: `(-1)^(j+m)` matches the CPU
  loop to 0.0 (bitwise) across all four (mirror, n_fold in
  {1, 4}) combinations, while `(-1)^j` alone flips the sign of
  every odd-m column (max abs err ~1.5) -- an error exactly
  invisible at even `n_fold` (contributing m all even, both
  benchmarked Ni m-3m cases) and fatal at odd `n_fold`, including
  1. The numpy-xp oracle's n_fold=1 case is the named killer.
- Per batch: `G[b,k,n,j] = conj(gln[b,n,j]) * table_T[k,n,j]`, `G2`
  without the conj, both built **for `n < bw` only** -- the
  even-slP guard: at even slP, `bwP = slP//2 + 1 > bw` (the bw-16
  oracle case: slP 32, bwP 17), and the quadrant rows
  `n in [bw, bwP)` stay zero, as does the never-written `m = bw`
  column; `value = matmul(G, A)`, `negated = matmul(G2, A2)` as
  batched GEMMs over the k axis (`(k, B*n, j) @ (k, j, m)`)
  -- measured 0.045 ms/pattern each at B=32 (einsum is 1.56x
  slower; matmul frozen). Then the 4-quadrant mirror fill: the
  `(m+n)`-parity signs on the two mixed quadrants
  (`fxc[slP-k, n, m]`, `fxc[k, slP-n, m]`), no extra sign on the
  doubly-negated quadrant, and the final `(-1)^k`, as strided
  assignments/elementwise ops on views of the `fxc` batch buffer.
- **`fxc` buffer freshness (review-corrected)**: the CPU kernel
  WRITES the `m % n_fold != 0` systematic-zero columns, the zero
  rows and the pad slices on every call (`_xcorr.py:492-501` --
  "written, not assumed") precisely so its buffer is reusable
  across n_fold/mirror changes; the only never-written slots are
  the `m = bw` column at even slP (`_xcorr.py:423-432`). The
  drafted "systematic-zero columns are never written" sentence
  inverted this and must not survive into any docstring. Device
  rule, frozen: the batch `fxc` buffer is **zeroed at session
  build and re-zeroed (or fully written across all m columns) per
  phase** before the fill, so a buffer cycling between phases with
  different n_fold/mirror never carries stale columns; the
  per-batch memset cost (~0.05-0.08 ms/pattern at B=32, bw 68)
  enters the D8 model and the D12 projection. Killer: the
  mixed-symmetry multiphase gated test (validation) -- the
  sign-scrambled same-master convention shares zero structure and
  cannot catch stale columns.
- **Recorded systematic deviation (mirror=True)**: under mirror
  the CPU sums only `(j+m)`-even terms (stride-2 loop,
  `_xcorr.py:505-522`); the dense GEMM also sums the `(j+m)`-odd
  coefficients, which the `SYMMETRY_POWER_TOLERANCE = 1e-8`
  relative-power validation guarantees only near-zero -- a small
  systematic deviation distinct from f32 rounding, absorbed by the
  D4 MTP bands and named here (and in D14).

Stage-5: mirror the CPU separable path verbatim
(`_xcorr.py:1777-1809`): batched `ifft` along k (length slP), slice
to bwP, batched `ifft` along n, batched `irfft` along m to slP, all
`norm="forward"`; with `n_fold > 1` only alpha planes
`m % n_fold == 0` are transformed, scattered into a zeroed
half-complex buffer (the exact CPU skip, `_xcorr.py:1787-1801`).
Measured: 0.48 ms/pattern unpruned, 0.21 pruned (n_fold 4, B=32).

Stage-6: `xc *= r_den` (f32 resident per phase, 4.96 MB at bw 68)
then `argmax` over the flattened cube -- `cp.argmax` returns the
first occurrence, matching `_find_peak`'s strict-`>` first-max
(`_xcorr.py:580-599`; re-pinned by a planted-tie case in BOTH the
numpy-xp default suite and the gated cupy suite -- the numpy case
alone proves nothing about cupy's reduction). **`normalize=False`
is supported unchanged (review-decided)**: the public
`normalize: bool = True` (`ebsd.py:2005`, `_indexer.py:1278`)
selects the un-normalized branch (`_indexer.py:710-728`), which
has NO `r_den` anywhere -- under `backend="gpu"` the `_GpuSession`
then holds no `r_den` residents and stage 6 is the argmax on the
raw f32 cube (the multiply skipped), matching plain `_find_peak`.
Never a raise, never untested: pinned by a gated un-normalized
dual-backend parity test (validation). The 27-neighborhood is
gathered ON DEVICE at
host-computed glide offsets and shipped back (B, 27) with the
argmax indices and peak values (~120 B/pattern D2H): the offsets
are computed by a host helper that ports the
`_extract_neighborhood` index arithmetic exactly, INCLUDING the
`emsphinx_compatible` per-slot glide and the even-slP
one-past-the-axis clamp defect (`_xcorr.py:196-211, 730-746`),
pinned by a parity unit test against `_extract_neighborhood` on
random cubes in BOTH compat settings (validation).

Stage-7 epilogue (host, EXTRACTED -- review-added, closes the
"host, unchanged" impossibility): `interp_peak` requires the full
host cube (`self.xc`) and calls `_extract_neighborhood` itself
(`_xcorr.py:2457-2473`), and the CPU `correlate()` drives
compute + `_find_peak` + `interp_peak` + `refine_zyz` as one
method (`_xcorr.py:2563-2568`); under this split the cube never
reaches the host, so stage 7 cannot literally run "unchanged".
Frozen instead: the `correlate()` epilogue is extracted into a
shared, private, neighborhood-fed entry point (an
interp-from-neighborhood function on the correlator) reproducing
byte-for-byte the flat-index -> (k, n, m) decomposition,
`_interpolate_maxima`, the `emsphinx_compatible` bounds check
that reads `|x[0]|` twice and never `|x[2]|` (the line-421 bug),
the step-rejection reset, and the zyz grid formula
(`_xcorr.py:2467-2492`). The CPU path calls it from `correlate()`
(bitwise-guarded by `test_backend_cpu_is_bitwise_default` plus
the full existing spherical suite); the GPU path calls it with
the device-gathered 27 values + argmax index; and BOTH paths seed
`refine_zyz` from the INTERPOLATED zyz triple, exactly as
`correlate()` does today -- never from the coarse cell. A direct
default-suite epilogue-parity unit test pins the extraction
against `interp_peak` on random cubes in both compat settings
(validation) -- end-to-end tests alone would mask a wrong
step-bound branch because step rejection is rare and Newton
reconverges.

**Compat neutrality, frozen**: all three `emsphinx_compatible`
switch points (Gaussian-fit offset -- stage 1; neighborhood glide
defect -- the offsets helper; the `|x[0]|`-twice bounds bug -- the
extracted stage-7 epilogue) plus the unconditional `vPeak`
wrong-monomial defect (stage 7) live in host code, the host
offsets helper or the shared host epilogue; the
device stages are compat-neutral. `emsphinx_compatible` therefore
has IDENTICAL semantics across backends, pinned by a dual-backend
toggle test.

### D3 -- Dtype strategy (frozen; two measurement points MTP)

1. **Device stages 4-6 run complex64/float32.** Rationale: Ada
   fp64 is 1/64 rate; c128 doubles every buffer and halves the
   feasible batch; the user's own EMSphInx `feature/GPU` CUDA build
   kept all-double device buffers and measured **6.6x slower** than
   the 20-thread CPU path
   (`specs/_research/explore-emsphinx-programs-and-formats.md:701-708`)
   -- the strongest prior against f64-on-device. The float32 cube
   affects results ONLY through (a) which cell wins the argmax and
   (b) the 27 neighborhood values handed to the host interpolator
   (float64 thereafter); (b) perturbs the sub-cell offset by
   O(1e-7) relative, and (a) is the discrete risk the
   refined-to-refined oracle absorbs (D4).
2. **Spectrum-product precision, MTP**: `A`/`A2` are formed from
   the host c128 `flm` and the NaN-zeroed f64 table (D2), cast to
   c64 once. The
   per-batch `G` build is measured BOTH ways at the implementation
   gate -- (i) all-c64 on device vs (ii) multiply at c128 (gln
   uploaded c128, NaN-zeroed f64 table resident) then cast -- and
   the cheaper
   is kept iff the D4 parity metrics are indistinguishable; the
   choice and both measurements are recorded.
3. **Stages 7-8 stay float64 on host** -- the refined
   score/orientation is the deliverable and the D4 oracle depends
   on it.
4. **No public dtype knob in v1**: a score-sensitive `refine=False`
   workflow (coarse scores come off the f32 cube) uses
   `backend="cpu"`; documented in the Notes. An internal
   f64-device diagnostic path may exist for the parity measurement
   scripts but is not public API (open question U6).

### D4 -- Parity oracle: CPU is the reference; tolerances MTP (frozen discipline)

The CPU backend is the oracle for every GPU output; comparisons use
the Phase 10 vocabulary (orientations via
`Orientation.angle_with(..., degrees=True)` bands, scores via
Pearson + |diff| bands -- never equality; `tech-stack.md:50-51`).
All tolerance NUMBERS are measured on this machine at the
tests/implementation gates and then pinned with the ~2x margin
convention; the expectations below are recorded a-priori reasoning,
not pins.

1. **IQ: bitwise equal** (`==`, not near). Preprocessing and the
   DCT IQ never leave the host, so any IQ difference means the GPU
   path corrupted routing -- a free wiring probe, deliberately
   stronger than Phase 10's 1e-3 cross-engine band.
2. **Winner agreement**: identical (phase, pseudo-symmetry-variant)
   winner on > 99.9 % of patterns expected; the flip rate is
   measured on `nickel_ebsd_small` (9), the `nickel_ebsd_large`
   20-pt subset (default suite) and the 165-pt subset (weekly),
   then pinned. Flips concentrate at pseudo-symmetric/degenerate
   near-ties by construction. **Falsifiability (review-added)**: a
   fraction like > 99.9 % is unmeasurable at 9 or 20 patterns (one
   flip = 11 % / 5 %); the flip-RATE pin lives on the weekly
   165-pt and 4125-pattern runs, while the small/20-pt gated tests
   pin exact flip COUNTS (expected 0; pinned at the measured
   count) -- never a fractional assert below the resolvable N.
3. **Coarse argmax cell**: agreement fraction measured (expected
   ~all equal; adjacent-cell moves possible on plateau peaks at
   f32), recorded; the honest gate is 4, not this.
4. **Refined-to-refined misorientation** (both backends
   `refine=True`): median expected well under 0.05 deg -- the f64
   Newton refinement from the same (or an adjacent) coarse cell
   reconverges to the same stationary point; a-priori ceiling where
   the cell moved is half a grid cell, 1.33 deg at bw 68. Sanity
   anchor: the CPU port itself sits 0.31-0.34 deg from EMSphInx
   (`mission.md:40`); the GPU-vs-CPU band must be far inside it.
   Measured then pinned (median + max) on the three data sets.
5. **Scores: same scale by construction** (the device stage feeds
   the host interpolator/refiner) -- the explicit
   anti-EMSphInx-defect requirement: EMSphInx's own CPU vs CUDA
   metric scales diverged ~7x
   (`specs/_research/explore-real-data-and-tests.md:334`); this
   port must NOT repeat that. Pins: Pearson r ~ 1.0 minus a
   measured epsilon; refined-score relative |diff| expected
   1e-5-1e-4; `refine=False` coarse-score relative |diff| expected
   ~1e-6-1e-5 (f32 cube); all MTP.
6. **Failure semantics identical**: `ptp == 0` degenerate patterns,
   `navigation_mask`, non-finite guards produce the identical fill
   rows (`(0,0,0,0,-1,0,-1)`, `tech-stack.md:33`); failed patterns
   are excluded from the device batch (D7.4).
7. **Recorded deviation, not emulated (review-corrected)**:
   `_find_peak` skips NaN ANYWHERE except the flat-index-0 seed
   (`_xcorr.py:582-590`), while `cp.argmax` returns the position
   of any NaN. Probe-verified (review session): NaN in slot 0
   AGREES (both return 0 -- the drafted framing named the one case
   that matches); the divergent case is NaN at any OTHER index
   (CPU skips it and returns the true max; cupy returns the NaN's
   index -- probed, NaN at 3 -> 3). Such cubes are reachable only
   via a degenerate `r_den` (`_xcorr.py:2819-2837`; the
   `rDen = +inf` first-occurrence semantics MATCH). Same
   pathological-input family, recorded: the per-phase `r_den`
   f64 -> f32 residency cast can overflow a large-but-finite f64
   denominator reciprocal to `+inf`, letting the device argmax
   diverge from CPU on near-degenerate windows. Documented; no
   test emulates any of these.

### D5 -- Determinism contract, per backend (frozen; one MTP point)

The Phase 6 pledge -- results bitwise deterministic across
chunking/threads (`roadmap.md:80`, module doc
`_indexer.py:231-241`) -- **cannot extend to GPU-vs-CPU** (float32
stage) and is not asserted GPU-vs-GPU across batch sizes a priori
(cuFFT plan choice and GEMM reduction order may change with B).
Frozen contract:

1. `backend="cpu"`: the existing pledge, untouched, re-pinned by
   the bitwise default-path guard test.
2. `backend="gpu"`, fixed (device, driver, batch size): bitwise
   deterministic run-to-run -- asserted in the gated suite. The
   device code must not introduce nondeterminism: no atomics-based
   argmax/reductions (library `cp.argmax`/`matmul`/cuFFT only).
3. `backend="gpu"` across batch sizes: MEASURED (B in {8, 32,
   default} on the Ni sets). If bitwise, pinned bitwise; else
   pinned at the measured tolerance and recorded as the documented
   deviation. The last-batch padding rule (D7.3) makes every
   pattern's transform run at the same batch shape, which is what
   gives this a chance of holding.
4. GPU-vs-CPU: tolerance parity per D4 only -- a recorded
   constitutional deviation (plan 0 amends the pledge's wording to
   scope it per backend).
5. Cross-device/driver variation: recorded if ever observed, never
   asserted (the cholesky-status precedent, `tech-stack.md:36`).

### D6 -- CuPy as an optional dependency: gate, guard, packaging (frozen)

1. **Never at module scope; never in `_constants.py` at import
   time.** Probe-measured hard fact:
   `importlib.metadata.version("cupy")` raises
   `PackageNotFoundError` when `cupy-cuda12x` 14.2.0 is installed
   (dist name != module name), so the existing
   `deps_for_version_check` registry (`_constants.py:14-37`)
   cannot gate CuPy as-is; also `import cupy` costs ~1 s and must
   not tax `import kikuchipy` (the pooch module-scope ban is the
   template, `tech-stack.md:16`). The gate lives in `_gpu.py`,
   lazily, cached after first evaluation.
2. **Three-stage runtime gate**, each stage with an actionable
   message (probe-measured: each stage fails independently in the
   wild):
   (a) *importable* -- `try: import cupy`; on failure raise
   `ImportError` with the `verify_dependency_or_raise` wording
   shape (`_constants.py:40-45`): "Spherical indexing with
   backend='gpu' requires that 'cupy' is installed (e.g. 'pip
   install cupy-cuda12x' matching your CUDA version; cupy-cuda11x /
   cupy-cuda13x / ROCm wheels exist)";
   (b) *device present* -- `cupy.cuda.runtime.getDeviceCount() >
   0`, catching `CUDARuntimeError`, message naming drivers;
   (c) *cuFFT loadable* -- one tiny `cupy.fft` probe transform;
   probe-measured: on Windows `import cupy` and the device count
   SUCCEED while the first FFT raises `ImportError: DLL load failed
   while importing cufft` (the `cupy-cuda12x` wheel bundles no
   cuFFT and CuPy 14.2.0 does not auto-discover the
   `nvidia-cufft-cu12` wheel on Windows); the failure message names
   both remedies (full CUDA Toolkit on PATH, or `pip install
   nvidia-cufft-cu12 nvidia-cublas-cu12`).
   Version floor: `cupy >= 13`, checked via `cupy.__version__` in
   stage (a); only 14.2.0 is actually tested (machine-specific,
   recorded).
3. **Guard-then-function-scope-import** at every cupy call site
   (the 13-site pyebsdindex/nlopt pattern); docstring wording:
   "Requires that :mod:`cupy` is installed, which is an optional
   dependency of kikuchipy. See :ref:`dependencies` for details."
   (`_hough_indexing.py:163-166`).
4. **No pyproject `[gpu]` extra** (recorded deviation from the
   drafting brief's phrasing; open question U3): a `cupy-cuda12x`
   pin is wrong for CUDA 11/13/ROCm users and bare `cupy` builds
   from source; the exact precedent is the pyopencl stance --
   "should not be an optional kikuchipy dependency"
   (`_constants.py:50-51`) and `installation.rst:145-147` (the
   `[all]` extra deliberately excludes pyopencl). Instead:
   an `installation.rst` optional-dependency bullet
   (`:126-147` list) documenting `pip install cupy-cuda12x` (plus
   the Windows nvidia-wheel note), and the same command in the
   stage-(a) error text. `[all]` unchanged; CuPy is never installed
   on any CI job.
5. **Windows DLL shim, in code**: before the stage-(c) probe, on
   `os.name == "nt"` only, best-effort register every existing
   `site-packages/nvidia/*/bin` directory via
   `os.add_dll_directory` (located relative to the `nvidia`
   namespace package), wrapped so absence or failure is silent --
   the stage-(c) message then carries the manual remedy. Without
   the shim every FFT dies on a Windows pip install even with the
   nvidia wheels present (probe-measured); with a full CUDA Toolkit
   the shim is a no-op. Linux is untouched (CuPy's own preload
   works there; recorded, untested here).

### D7 -- Integration topology (frozen)

1. **Device section inside `_index_chunk`, lock-guarded.** The dask
   `map_blocks` structure, the threaded scheduler, truthful
   `chunks=`, the eager-compute promise (`_indexer.py:1722-1731,
   1765-1778`) and per-chunk correlator clones all survive. Under
   `backend="gpu"` a chunk runs: host stages 0-3 for all its
   patterns (collecting the c128 `gln` batch + iq), then per phase
   ONE device batch (stages 4-6) inside a process-wide device lock,
   then host stages 7-9 per pattern in the original per-pattern
   order (insertion semantics untouched). Rationale: host stages
   (~2.56 ms/pattern-core measured) keep overlapping across dask
   workers while the ~0.6-0.85 ms/pattern device sections
   serialize; a chunk-sized device section (~20 ms at B=32) is
   coarse enough that lock convoy is not expected to dominate --
   measured at the implementation gate; the dedicated
   feeder-thread/stream pipeline is the recorded fallback
   optimisation if it does (plan 8.1).
2. **One GPU consumer per process; no device state on the
   indexer.** Device tables and buffers live in a private
   `_GpuSession` created inside `index_patterns` when
   `backend="gpu"` and disposed at the end of the call (pool
   released); the `SphericalIndexer` object itself holds only the
   backend string, so nothing device-resident can be captured in a
   dask graph, pickled, or shared across calls. The session is
   shared read-only by all chunk invocations of that one call;
   correlator clones do NOT duplicate device tables.
3. **Last partial batch: pad to B.** The final chunk of a map is
   zero-padded to the fixed batch size; padded rows are computed
   and discarded before stage 7. Rationale: one cuFFT/GEMM plan
   set for the whole run, and every real pattern is transformed at
   the same batch shape (feeds D5.3). Cost: at most one batch of
   waste per call.
4. **Failed patterns** (`ptp == 0` guards, masked points) are
   excluded from the device batch (their slots padded) and take
   the identical CPU fill-row path.
5. **Scheduler pinning**: the GPU-path `.compute()` passes
   `scheduler="threads"` explicitly, so a globally configured
   process/distributed scheduler can never move device handles
   across process boundaries.
6. **FFT shim**: the device branch calls `cupy.fft`/`cupyx`
   equivalents WITHOUT the scipy-only `workers=1` kwarg
   (probe-measured: `norm="forward"` matches `scipy.fft` to
   2.8e-14); the host branch is untouched. A
   `scipy.fft.set_backend` mechanism is rejected (cupy.fft is not
   a uarray backend -- probed `no __ua_domain__` -- and it could
   not cover the numba stages anyway).
7. **Device errors are never per-pattern failures (review-added)**:
   the frozen CPU contract catches exceptions PER PATTERN so one
   bad pattern never kills a run (`_indexer.py:182-183, 664-737`
   -- the whole per-pattern body sits in the try); that contract
   continues to govern the host stages 0-3 and 7-9 under BOTH
   backends. The device section runs once per chunk, OUTSIDE any
   per-pattern scope, and its exceptions -- cuFFT plan failure,
   driver reset, invalid-value errors, and OOM after the D8.4
   halving is exhausted -- MUST propagate and fail the run; they
   are never converted to per-pattern fill rows (a naive
   restructure that let the per-pattern try/except swallow a
   device-section exception would emit a whole chunk of silent
   fill rows -- exactly the silent-failure mode D8.4 forbids).
   Recorded as a deviation in D14; pinned by a monkeypatched gated
   test (a planted device-stage exception fails the run and
   produces no fill rows).

### D8 -- Batch sizing, VRAM model, OOM handling (frozen; calibration MTP)

1. **Chunk size == device batch size** under `backend="gpu"`. The
   CPU `_batch_estimate` (34/15/6 patterns at bw 53/68/88,
   `_indexer.py:377-432`) is bypassed -- 15/chunk at bw 68 is
   smaller than the efficient device batch (measured flat
   0.48 ms/pattern across B=8-64, but per-batch fixed costs favour
   B >= 16-32).
2. **Default batch from a VRAM model**: per-pattern device bytes
   `g(bw)` (fxc c64 + separable FFT intermediates + xc f32, with
   buffer reuse, plus the per-batch zero fills). Provenance,
   review-corrected: the **bw-68 ~50 MB figure is anchored on the
   measured pool high-water** (<= 1.4 GB at B=32 -> real c64
   working set ~40-44 MB/pattern; ~50 is deliberately
   conservative), while the **bw-88 ~110 MB figure is a scaling
   ESTIMATE, not a measurement** -- only 176^3 C2C probes ran at
   bw 88, never the separable pipeline (naive no-reuse component
   sum ~95-97 MB/pattern, plausible). Both are calibrated MTP at
   the implementation gate from measured pool high-water marks.
   Default `B = clamp(floor(0.5 * free_vram / g(bw)), 1, 64)`
   (half the free VRAM as headroom; WDDM display-attached cards
   lie about "free"). Measured guidance recorded: B=32 comfortable
   at bw 68 (<= 1.4 GB pool, >= 4.2 GB free), B=8-16 at bw 88,
   B=64 at bw 88 out of reach on 8 GB (0.46 GB free at the 176^3
   probe). **Chooser cross-check (review-added)**: at bw 88 with
   7.45 GB free the formula yields B~33 -- outside the benched
   comfortable band; the MTP calibration must check the chooser
   against the recorded guidance band per bandwidth and either
   validate the larger B or add a bw-dependent cap, recorded
   either way. An explicit `chunksize` overrides the default.
3. **`gpu_memory_per_batch_bytes(batch_size)`** on
   `SphericalIndexer`: the model exposed in the spirit of
   `memory_per_worker_bytes` (`tech-stack.md:39`) but -- review
   clarifier -- as a **METHOD taking `batch_size`, not a
   property** (`memory_per_worker_bytes` is argument-less,
   `_indexer.py:1531`; this helper is parameterized, pure model
   math, and performs NO device query -- the free-VRAM query
   happens in `index_patterns`), plus the per-phase resident term
   (~11 MB at bw 68, ~24 at 88: A/A2, table_T f32, r_den f32 when
   `normalize=True`). `index_patterns` prints it in the
   verbose info message (device name, total/free VRAM, B, model
   bytes) and warns when the model exceeds the measured-free VRAM
   -- the 2 GiB host-warning counterpart.
4. **Graceful OOM (review-restructured)**: two windows, because
   the dask graph's chunking is fixed at graph build (D8.1
   freezes chunk == batch) and cannot be halved mid-compute:
   (a) at **session build**, before the graph exists, an
   `cupy.cuda.memory.OutOfMemoryError` halves B, frees the pool,
   and retries; (b) **inside the compute** -- the first batch (the
   realistic window: cuFFT work areas materialize at the first
   transform) or ANY later batch (realistic on a WDDM
   display-attached 8 GB card under external VRAM pressure) -- the
   OOM propagates out of the worker and aborts the `.compute()`;
   `index_patterns` catches it, disposes the session, frees the
   pool, and REBUILDS session + graph at B/2, re-running the
   whole compute (already-computed chunks are recomputed --
   bounded waste on an exceptional path; there is NO device
   sub-batching inside a chunk, which would break the
   chunk == batch identity and the D5.3 uniform-batch-shape
   argument). Both loops floor at B=1: there the error re-raises
   as `MemoryError` naming bw, the model bytes, the free VRAM,
   and the remedies (smaller bandwidth, `backend="cpu"`). A
   completed halved run's results come ENTIRELY from the final B
   (whole-run rebuild, never mixed batch sizes), so the D5.2
   run-to-run bitwise pin applies at that final B. Non-OOM device
   errors never enter the halving loop -- they propagate per
   D7.7. **Never a silent fallback to the CPU backend** --
   backend choice is explicit and result-affecting (D1).
5. **bw 113+ feasibility on 8 GB**: model extrapolates to ~230
   MB/pattern -> B=4-8 plausible; measured and recorded (not
   gated) at the implementation gate.

### D9 -- Pseudo-symmetry and refinement semantics (frozen)

`pseudo_symmetry_ops` is supported UNCHANGED under `backend="gpu"`
-- for free under the D2 split: the Phase 8 variant loop is host
code operating on the host-resident `flm` and the host correlator's
`refine_zyz`, and variants are ALWAYS Newton-refined regardless of
`refine` (D4 of `specs/2026-09-06-pseudo-symmetry/requirements.md`,
the `indexer.hpp:252` asymmetry) -- the device stage only replaces
where the BASE coarse maximum comes from. Same for `refine=True`:
host `refine_zyz` from the GPU-coarse seed, packed rows width 7
with identical fill (`tech-stack.md:33`). Same for `normalize` --
BOTH settings are supported under `backend="gpu"` (review-decided,
D2 stage-6: `normalize=False` skips the `r_den` multiply and
holds no `r_den` residents), each with its own gated parity test.
Pinned by a dual-backend psym test (true 90-deg-z op, `n_best=2`,
mirroring `test_variants_on_the_un_normalised_path`). Recorded, not acted on:
each op adds ~0.92-1.04 ms/pattern of host refinement (Phase 8
measured), so at high op counts host refinement, not the GPU,
bounds throughput -- the crossover op count is computed from the
measured rates and recorded in validation; device-batched variant
refinement stays deferred (non-goal).

### D10 -- Test gating: local GPU suite, no CI evidence (frozen)

1. **GitHub CI has no CUDA anywhere**; the existing `gpu` marker is
   wgpu/Metal-specific and runs on macOS CI
   (`tests.yml:137-143`, `pyproject.toml:203-206`,
   `conftest.py:81-113`) -- it is NOT reused (its documented
   meaning and `--gpu` flag semantics would silently broaden).
2. A **`cupy_gpu` fixture** skips unless the three-stage gate (D6)
   passes, with an instruction-bearing skip reason per stage (the
   `emsphinx_dir` convention, `conftest.py:658-673`) -- an
   availability probe, not an env-var opt-in, since there is
   nothing to point at.
3. **`KIKUCHIPY_NO_GPU_TESTS=1` kill switch**: the fixture skips
   unconditionally when set, so the default suite stays GPU-silent
   on machines where cupy is installed but the operator wants
   CPU-only runs (e.g. a concurrent job owning the GPU -- the
   situation on this very machine at drafting).
4. GPU tests live in ONE file
   (`tests/test_indexing/test_spherical_gpu.py`) and are run
   `-n 0` locally (xdist workers each initialising CUDA on an 8 GB
   card is the hazard; no machine-wide lock needed -- CUDA
   contexts share a device). **Structural `-n 0` (review-added)**:
   the `cupy_gpu` fixture ALSO skips whenever the
   `PYTEST_XDIST_WORKER` env var is set (i.e. under any `-n N`
   run), because the availability probe would otherwise auto-run
   the full gated parity/throughput suite inside every default
   `-n 4` suite command on a cupy-capable dev machine (this one)
   -- 4 workers each initialising CUDA, GPU contention with
   concurrent jobs, and a surprise multi-minute "default" run.
   With the skip, `-n 0` is enforced by construction and the
   default-suite command stays GPU-silent everywhere;
   `KIKUCHIPY_NO_GPU_TESTS` remains the manual kill switch. The
   xdist-skip logic itself is default-suite tested (env var
   faked).
5. **CI green is zero evidence for the GPU path** -- written into
   validation.md; every GPU gate is a local definition-of-done
   check recorded with measured numbers and the machine ID.

### D11 -- Coverage: the xp-agnostic core and the amended gate (frozen)

The Phase 8 binding rule -- 100 % coverage from the default suite
alone (`specs/2026-09-06-pseudo-symmetry/validation.md:289-290,
835-839`) -- cannot hold verbatim for a module whose lines need a
GPU. Amended deliberately (plan 0):

1. **The device pipeline core is xp-agnostic**: the stage-4-6 math
   takes an array-module handle (`xp`, plus an fft namespace) and
   runs under numpy in the default suite at small bandwidths,
   asserted against the CPU correlator -- so the pipeline LOGIC
   (spectrum build, mirror fill, pruned FFT ordering, argmax,
   gather, padding, batch assembly) is CI-covered, not just
   unit-mocked.
2. Validation/batch-model/gate-decision logic is default-suite
   covered (the cupy import faked/monkeypatched where needed).
3. The cupy-touching remainder (the gate's real probes, the session
   allocation, D2H/H2D) is covered to 100 % by the local gated
   suite, coverage measured locally with the command recorded in
   validation.md (the Phase 8 coverage-command pattern).
   **CI-side consequence, recorded (review-added)**: `tests.yml`
   uploads `--cov=kikuchipy` to Codecov (`tests.yml:39, 149-151`),
   so the Codecov patch report on the PR WILL show the cupy-gated
   `_gpu.py` lines uncovered -- the first time in this project.
   That is the accepted, recorded outcome, stated in the PR
   description; the locally recorded coverage command is the
   gate, and a degraded codecov patch status on those lines is
   not a review finding.
4. `# pragma: no cover` ONLY on import-guard branches
   (`_constants.py:35,52,62` precedent); never whole functions; no
   pyproject coverage `omit` (rejected: hides real lines and
   touches an upstream-shared file section).
5. Mutation/bug-injection runs locally -- this machine has the GPU
   -- and the plan 7 list includes device-stage mutants.

### D12 -- Performance go/no-go (frozen floor; numbers MTP)

In the spirit of the >= 2 pat/s/core hard floor
(`tech-stack.md:39`), the phase carries its own measured gate:

1. **Floor (go/no-go)**: end-to-end `backend="gpu"` refined
   throughput on the `nickel_ebsd_large` route at bw 68 (m-3m,
   `refine=True`, default settings) **>= the 8-worker
   `backend="cpu"` throughput measured on the SAME machine, idle**
   -- else the phase records a negative result and the review gate
   decides whether the backend ships marked experimental or not at
   all. The CPU baseline is RE-MEASURED on this machine when idle
   (the historical 205-216 pat/s band is from the Anes-reproduction
   runs; a clean same-machine pair is required -- scheduling is
   open question U5).
2. **Recorded expectation, not a gate (review-corrected)**: the
   drafted component sum (0.57/0.84 ms/pattern GPU-side) omitted
   the per-batch zeroing of the `fxc` and pruned-planes buffers
   (~0.05-0.08 ms/pattern at B=32, bw 68 -- now required by the
   D2 freshness rule) and rested the 0.10 ms mirror-fill estimate
   on contiguous-bandwidth rates that negative-stride
   (flipped-view) elementwise writes typically do not reach
   (realistic ~0.1-0.2 ms/pattern; partially cancelled by a
   2x-inflated byte count in the drafting arithmetic). Corrected
   projection: GPU-side ~0.65-0.75 (m-3m, pruned) to ~0.95 (no
   z-fold) ms/pattern -> ~1050-1550 pat/s device-bound; host
   residue 2.56 ms/pattern-core; end-to-end ~850-1500 pat/s
   refined at bw 68 (~5-7x the 8-worker CPU baseline, ~15-25x the
   63.8 pat/s single core), ~650-850 pat/s at bw 88 (~25x single
   core). The mirror-fill rate is marked strided-write-sensitive
   for the implementation-gate measurement. A shortfall against
   the projection that still clears the floor is a pass with the
   gap recorded and explained.
3. **Sobering priors recorded** (why the floor is not
   hypothetical): the all-double `feature/GPU` CUDA build was 6.6x
   SLOWER than 20-thread CPU; the un-tuned single 138^3 `irfftn`
   probe took 41-48 ms against the CPU's 11.9 ms whole coarse
   correlate. Batching, plan reuse and the `n_fold` plane skip are
   the difference between speedup and regression -- all three are
   frozen into D2/D7.
4. Also measured and recorded (not floored): coarse-only
   (`refine=False`) throughput; bw 88 run; 2-op psym run; the D8
   VRAM high-water marks.

### D13 -- Docs, packaging surface, licensing (frozen)

- Docstrings: `backend` parameter docs on both methods with the
  optional-dependency sentence (D6.3), the Notes covering the
  float32 coarse stage, the determinism scoping, the VRAM
  model/warning, the no-silent-fallback rule, and the
  score-sensitivity note (D3.4). No roadmap phase numbers in
  public docstrings.
- `doc/user/installation.rst`: one bullet in the optional
  dependencies list (cupy; what it enables; `pip install
  cupy-cuda12x`; the Windows nvidia-wheels note); the
  `[all]`-excludes note extended to name cupy beside pyopencl.
- CHANGELOG: one `Added` entry naming `backend="gpu"` on
  `EBSD.spherical_indexing`/`SphericalIndexer` with the fork PR
  link (next free number at open time; #15 expected -- the
  concurrent tutorial phase may claim it first).
- Licensing: `_gpu.py` carries the kikuchipy GPL header + the
  verbatim CMU/Lenthe third-party block (the stage split mirrors
  the user's `feature/GPU` `include/gpu/pipeline.hpp` design and
  the pipeline derives from the CPU port's `sht_xcorr.hpp`
  lineage) + the dated modification notice; GPL-2.0-or-later
  conveyed under GPL-3.0-or-later; BSD opt-out impossible, stated
  in the PR.
- Module doc: `_indexer.py:231-241` determinism sentence re-scoped
  per backend (implementation task, mirrors the plan 0 roadmap
  amendment).

### D14 -- Recorded deviations, in one place (recorded)

1. GPU-vs-CPU results are tolerance-parity, not bitwise (D4/D5) --
   constitutional amendment drafted in plan 0.
2. Bitwise-across-batch-size is measured, not promised (D5.3).
3. The `_find_peak` NaN pathology -- NaN anywhere BUT slot 0
   diverges (slot 0 agrees), plus the f32 `r_den` inf-overflow
   family -- is not emulated on device (D4.7, review-corrected).
4. No `[gpu]` extra despite the drafting brief's phrasing (D6.4,
   open question U3).
5. `_batch_estimate` is bypassed for the GPU backend; chunksize
   means device batch size there (D8.1) -- docstring documents the
   changed meaning.
6. The GPU backend bypasses none of the dask machinery but
   serializes its device sections -- the tech-stack parallelism
   sentence is amended to scope the dask rule to the CPU stages
   (plan 0).
7. All GPU pins are machine-specific (RTX 2000 Ada, 8 GB, cc 8.9,
   driver 595.71, CuPy 14.2.0, Windows 11) and say so in the test
   comments and validation records.
8. **Device-section exceptions fail the whole run** (D7.7/D8.4) --
   a recorded deviation from the per-pattern isolation contract,
   which continues to govern the host stages in both backends
   (review-added).
9. With `mirror=True` the dense GEMM sums the `(j+m)`-odd
   coefficients the CPU's stride-2 loop skips -- guaranteed only
   <= 1e-8 relative power by the symmetry validation; a recorded
   systematic deviation distinct from f32 rounding (D2,
   review-added).

## Context

- Constitution collisions this phase amends (drafted plan 0,
  applied at build): `mission.md:3` ("pure-Python, CPU-only"),
  `mission.md:29` (deliverables row "CUDA/GPU indexer | out of
  scope -- CPU multi-threaded path only"), `mission.md:40-41`
  (success criteria 2/3 gain GPU clauses), `roadmap.md:1` (title
  "EMSphInx CPU port"), `roadmap.md:3` (chain), `roadmap.md:80`
  (bitwise-determinism pledge), `tech-stack.md:16-17` (optional
  deps + CI), `tech-stack.md:21` (float64 rule -- this phase is
  the sanctioned float32 surface), `tech-stack.md:39` (perf
  fallback list cross-reference), `tech-stack.md:44` (parallelism
  constitution), `tech-stack.md:50` (gating conventions), the
  coverage-gate convention (D11). The research-artefact line
  `specs/_research/plan-faithful-port.md:16` ("Forbidden: ...
  CUDA/pyopencl. New optional deps only if justified in a spec")
  contains its own escape hatch -- this spec is the justifying
  instrument; the line is cited as superseded, the file gains a
  one-line addendum at build.
- Research: the two 2026-09-07 scratchpad reports
  (`gpu_hotpath_research.md` -- hot-path profile, CuPy stage
  mapping, micro-benchmarks; `gpu_fork_research.md` -- constitution
  collisions, optional-dep conventions, API options, CI/coverage
  strategy, psym interaction). The scratchpad does not survive the
  session; every load-bearing number and citation is carried
  inline in D1-D14 and in validation.md "Recorded results".
- Measured hot-path profile (this session, single thread, bw 68 /
  bw 88 best): preprocess 0.19 / 0.19, unproject 0.19 / 0.18,
  analyze 0.46 / 0.92, spectrum 1.76 / 4.09, **inverse FFT 8.83 /
  21.17 (61.8 % / 65.7 %)**, scale+argmax 1.13 / 2.45, interp 0.34
  / 0.66, refine 1.38 / 2.57 ms; totals 14.29 / 32.23 ms -> 70 /
  31 pat/s. Stages 4-6 = 82 % of the pattern. Shapes verified
  live: bw 68 -> slP 135, bwP 68, `fxc` (135,135,68) c128 19.83
  MB, `xc` (68,135,135) f64 9.91 MB, `gln` (68,68) c128; bw 88 ->
  slP 175, `fxc` 43.12 MB, `xc` 21.56 MB. (The task brief's
  "136^3/176^3 cubes" are even roundings; the stored cubes are
  half-cubes `(bwP, slP, slP)`.)
- Measured device micro-benchmarks (this machine, c64, CUDA
  events, best-of-3): separable bw-68 pipeline transform 0.483 /
  0.482 / 0.520 ms/pattern at B=8/32/64, 0.210 pruned (n_fold 4,
  B=32); full 3D C2C 136^3 0.804 ms/cube flat across B, 176^3
  1.75-1.89; Wigner-scale contraction GEMM 0.045 ms/pattern (bw
  68, B=32, ~3.8 TFLOP/s effective; einsum 1.56x slower), 0.127
  (bw 88, B=8); scale+argmax 0.320/0.125/0.095 ms/pattern at
  B=8/32/64 (argmax streams ~104 GB/s); H2D 8.7 GB/s pageable /
  12.0 pinned, D2H result rows 0.02-0.06 ms -- transfers are
  noise (~2-10 us/pattern at the D2 split).
- Wigner tables: pi/2 table `(bw,bw,bw)` f64 (2.52 MB at 68, 5.45
  at 88), built once, shared read-only by all correlator clones
  (`_xcorr.py:2301-2304`); the refinement `d_beta` `(bw,bw,bw,2)`
  buffer (5.03 MB at 68) is per-clone, rebuilt per `_derivatives`
  call -- stays host-side (D2).
- EMSphInx `feature/GPU` reference design (read via `git show`,
  never checked out): all-double device buffers; device does
  spectra product + FFT + `peakIndex` + 27-value `neighborhood`;
  HOST does `interpPeakNeighborhood` -- the same split D2 freezes
  (minus their optional `refinePeakBatchDevice`). Its measured
  outcome (6.6x slower than 20-thread CPU) is the float64 caution
  D3 answers.
- Phase deliverables composed: Phase 4 correlator internals
  (spectrum kernel, separable FFT, `_extract_neighborhood`,
  `index_to_euler`), Phase 6 indexer/insertion/dask machinery,
  Phase 7 `refine_zyz`, Phase 8 psym loop + width-7 rows, Phase 10
  regression suite (untouched -- CPU default), Phase 9/10
  local-gating conventions.
- Baselines cited: single-core 63.8 pat/s at bw 68
  (`specs/2026-09-02-spherical-indexing-ebsd/validation.md:748`);
  8-worker production ~205-216 pat/s (Anes reproduction, median
  216.5, auto-memory) -- to be re-measured on this machine idle
  (D12.1, open question U5).
