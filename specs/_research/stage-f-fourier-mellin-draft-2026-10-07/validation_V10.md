<!--
INSERTABLE BLOCKS for specs/2026-09-07-hrebsd-dic/validation.md,
drafted 2026-10-07 by the Stage F spec workflow and REVISED the same
day after the two-critic spec review (plan 12.5; no repo file was
edited). Block 1 (the V10 oracle section) goes immediately after the
last V9 subsection the Stage E gates have appended by splice time
(today "#### V9 recorded results, failing-tests gate (2026-10-06)",
plus whatever Stage E's implementation gate adds after it). Block 2
(the spec-gate ledger) follows Block 1 directly. Block 3 is a dated
note inside V9(b). Ledger entries are
numbered from 200 on purpose: the main session renumbers them to
follow the last Stage E entry at splice time, and the references in
requirements D22 and plan section 12 move with them.
-->

<!-- ========================= BLOCK 1: V10 ========================= -->

### V10 -- Fourier-Mellin oracles (Stage F, 2026-10-06)

Requirements D22 govern. One new file,
`tests/test_indexing/test_hrebsd_fourier_mellin.py`, laid out like
`test_hrebsd_gpu.py` (dated pin constants; the default-suite classes;
the gated ones). Test modules cannot import each other, so the
helpers are DUPLICATED (the D21.14.1 rule), each copy carrying a
comment that names its source and line range at f297867e and kept
identical to it (a drift is a test edit with a dated comment): from
`test_hrebsd_gpu.py`, the gate and fake-cupy helpers
(`_cupy_gpu_skip_reason`, `cupy_gpu`, `make_fake_cupy`,
`install_fake_cupy`, `hide_cupy`, `fresh_gate`; lines 585-753), the
assertion helpers (754-796), the oracle helpers (`matrix_of` to
`deformed_pattern`; 797-991), the V9 F6 map and the Ni inputs
(`f6_map`, `ni_inputs`, `run_ni`, `run_engine`; 1224-1347) and the
numpy-session helpers (`make_state` to `run_gpu_numpy`; 1376-1538);
from `test_hrebsd_seeding.py`, the V8 ramp map (the `RAMP_`
constants, lines 170-203; `in_plane_fe`, `out_of_plane_fe`,
`orientation_a`, `deformed_pattern`, `oracle_reference`, `ramp_map`;
713-830); about 1000 lines in all (plan 12.4 item 16 records the
alternative). Expensive runs are cached with `functools.lru_cache`
(the V8 precedent). Three edits of existing pins, each with a dated
comment: `FROZEN_SIGNATURE` in `test_ebsd_hrebsd_dic.py` (D15.4
amendment), `test_run_defaults_are_frozen` in
`test_hrebsd_engine.py`, and `SEED_FROM_NEIGHBORS_GPU_MESSAGE` in
`test_hrebsd_gpu.py` (D21.12 amendment). No existing docstring test
changes; the docstring is pinned by the NEW
`TestFourierMellinSwitch::test_the_docstring_documents_fourier_mellin`
(D22.16). The CPU route of D22.10 is the oracle for every device
output (D22.14). Every band below is MTP: the failing-tests commit
carries `FIXME-pin` placeholders (the V9 convention); CPU-side
literals are measured at the failing-tests gate, device-side ones at
the implementation gate, each pinned at about 2x margin with the
recipe and machine ID beside the constant, and where a pin must also
kill a mutant the kill separation is stated beside it. The
spec-gate numbers quoted (ledger 200 to 216) are the scales a
correct implementation should reproduce, not pins; they were
measured on throwaway prototypes, and the frozen recipe itself only
in ledgers 213, 214 and 216. The Si-indent file is never a test
fixture: its numbers are RECORDED measurements, (m) below.

Fixtures, synthetic and generated in the test with fixed seeds:
**G1** the V3/V4 deformed-master 480 px oracle at `PC_480` (0.4210,
0.5794, 0.5049), `border=0.05` (in-frame to 5.62 deg) and
`border=0.15` (in-frame to 23.8 deg; ledger 215); **G2** the V8
projection centre (0.49, 0.51, 0.5049) and the V8(a) ramp map
(`ramp_map`; in-frame to 6.62 deg); **G3** a 512x622 synthetic at
`PC_480`, crop 460x560 at `border=0.05` (the V9 F3s shape), for the
physical-frequency pin (ledger 216); **G4** G1 under a detector-fixed
background (an off-centre Gaussian plus a linear ramp, the same image
multiplying reference and target, ledger 204), noise-free and at a
Poisson full-scale count of 50, both at `filter_cutoffs=(None,
None)` and `(0.05, None)`; **G5** the anchor pair: G1 reference and
target carrying the SAME fixed-pattern image (detector-fixed WHITE
per-pixel gain and offset noise, not banded noise), the target with
a 10 to 20 px projection centre shift and a 1.5 to 3 deg twist, so
the D5 seed is pulled to (0, 0) as on the real rim (ledger 207) --
NOT built at the spec gate: the fixed-pattern amplitude is set at
the failing-tests gate until BOTH run-time premises hold, (i) the D5
seed is exactly (0, 0) and (ii) `theta_hat` is within
`FM_ANGLE_TOL_DEG` of the imposed twist (fixed-pattern energy is
identical and unrotated in both FM spectra and can pull the angle to
0, too), and recorded; if no amplitude meets both, the G5 arms of
(g) are dropped with that record, and the FQ8 anchor census of (m)
covers the real-data case; **G6** an unrelated pair at the V8 PC (a
second orientation 20 deg about (0, 1, 1), a grain boundary): at a
budget of 200 the translation-seeded fit does NOT converge and keeps
a FINITE `h` (residual 1.89), the acceptance refuses the FM row
(criterion 2.080 against 2.040) and a forced FM-seeded fit does not
converge either (1.66; ledger 216, correcting ledger 204(iv)); **G7**
gate maps: orientations built as `g_t = g_r @ (M^T R_det M).T` for
imposed detector-frame rotations `R_det` (ledger 212), with a
symmetry-equivalent copy, about 0.5 deg orientation noise, an
unindexed point, a NaN rotation, a second phase, a two-grain map, an
all-identity map and a constant NON-identity map (every point at the
reference's own orientation), on the default detector and on a
tilted one (tilt 10, azimuthal 4, twist 1.5 deg); **G8** a retry map
on the V8 geometry at `MAX_ITERATIONS = 200`: one point whose pattern
carries a 4.0 deg twist while its `xmap` orientation reads the
reference's (so the pre-fit gate does not route it; premise asserted
at run time: the default seed does not converge at 200 and keeps a
finite `h`, measured 42.9 px away, while the FM row converges in 3
iterations; ledger 216 -- 3.0 deg is NOT usable, the default seed
converges there in 14), one G6 unrelated-pattern point whose `xmap`
also reads the reference's orientation (premises asserted at run
time: first pass not converged with a finite `h`; a forced FM-seeded
fit not converged), a constant pattern (the D2.6 contract), a masked
point and two easy points (0 and 0.8 deg).

CI cost, per class, fits at 480 px (60 px on the Ni map): (a) about
six Ni-map runs and two V9 F6 runs; (b), (c) and (h) no fit; (d) one
fit per planted-failure arm at the 50 budget; (e) the five starred
arms twice each (the `"off"` premise and the FM run), at most 50
iterations; (f) the ten fitted ramp points three times (the `"off"`
premise, `"always"`, `"auto"`), about five of them capped at 200 on
the premise; (g) about six; (i) the six fitted G8 points under two
modes plus two direct fits; (j) about six runs of a six-point map;
(k) about four numpy-session runs. About 100 fits in all, of which
about 15 run to a 50 or 200 cap. The default-suite wall time is
recorded at the failing-tests gate (the D21.14.3 precedent) and the
weekly marker takes any class that exceeds the Stage E per-class
envelope.

(a) **API, default path, check order, forwarding, docstring**
[default] (D22.1, D22.9, D22.11, D22.16). `fourier_mellin="off"`
equals the call without the keyword bitwise on the V9 F6 map (the
four-point pre-Stage-D pin map) and on the shipped Ni map, on the
CPU and through the numpy session of V9; the pre-Stage-D literal pins
and the V9 `extras == {}` asserts re-run unmodified; with `"off"` the
prop set is exactly the pre-Stage-F set, no `FourierMellinState` is
built and `SeedBatch.outputs` stays empty (spies). `ValueError` with
the D22.1 message for `"AUTO"`, `"on"`, `""`, `True` and `None`.
`seed_from_neighbors=True` with `"auto"` and with `"always"` raises
the D22.11 `ValueError` (literal asserted in full) on both backends,
with spies on the D21.2 gate, `resolve_reference`, `ReferenceState`
and the pattern gather at zero calls; an invalid `backend` still
raises its own `ValueError` first. `"auto"` with `xmap=None` at the
engine raises the D22.1 message; `"always"` with `xmap=None` runs.
Forwarding: with `run_hrebsd_dic` replaced in `kikuchipy.signals.ebsd`
by a raising spy (the `test_public_method_forwards_backend`
precedent, `test_hrebsd_gpu.py:2537` at f297867e),
`EBSD.hrebsd_dic(fourier_mellin=v)` passes `v` for each of the three
values. Public result: `EBSD.hrebsd_dic` on the Ni map under
`"always"` returns a CrystalMap carrying `fourier_mellin_seed` and
`fourier_mellin_angle` with the D22.9 dtypes and shapes, and under
`"off"` carries neither. The new docstring test pins the
`fourier_mellin` entry, the Raises entries, the warning, the Notes
topics and the rewritten Limitations sentence of D22.16, with no
stage letter; `FROZEN_SIGNATURE` and `test_run_defaults_are_frozen`
carry the new keyword in its slot. Expectation: the frozen literals.
Band: bitwise.

(b) **The FM angle, seed level** [default] (D22.3). Through
`fourier_mellin_angles` alone, no fit. On G1 at `border=0.05` and
the default band-pass: twists 0, +-0.37, 1.83, 2.71, -4.42, 5.29,
9.61, 19.13 deg (off the 0.5 deg grid on purpose; ledger 213) and
the two combined cases (6.13 deg about z then 1 deg about x; 3.27
deg about z then 2 deg about y); expectation `atan2(h21 - h12, 2 +
h11 + h22)` of the exact imposed homography; band
`FM_ANGLE_TOL_DEG` (MTP; spec gate 0.123 deg noise-free, 0.113 deg
with Gaussian noise of half the pattern's standard deviation on
both images, ledger 213). Sign: `theta_hat` has the sign of the
imposed `h21` at every nonzero twist. The look-up table, unit level
and the PRIMARY bin-unit (FM3) killer: from `fourier_mellin_lut(sr,
sc)` at 432x432 and 460x560, the weight-averaged (column, row) of
each entry's four bins, unwrapped, reconstructs `(rho_j cos theta_k
sc, rho_j sin theta_k sr)` within 1e-12 for every entry; `n_theta =
360` angles over [0, pi) (not 2 pi), `n_rho` by the D22.3.3 formula
(165 and 175), shape and entry count (237600 and 252000) pinned. G3,
seed level, at 2, 5, 8, 15 and 20 deg and -8 deg:
`FM_ANGLE_TOL_NONSQUARE_DEG` (spec gate at most 0.027 deg; the
bin-unit mutant measured 0.074, 0.196, 0.344, 1.03, 2.29 and 0.386
deg; ledger 216), so a 2x pin of about 0.06 deg kills the mutant at
every angle from 5 deg with at least 3x separation -- the second
FM3 killer. Dead band: G1 with a constant 8 px dead-band cross in
both images (the `dead_band` subregion) at 3 and 8 deg, within
`FM_ANGLE_TOL_DEADBAND_DEG` (spec gate 0.12 deg against 0.02 without
the cross, ledger 216). Peak search, unit level on planted
correlations through `fourier_mellin_peak`: a global maximum at a 40
deg lag and a local one at 5 deg returns 5 deg (the search window);
an exact tie between lags +k and -k returns +k (the lowest output
index) under numpy and, gated, under cupy; a peak at the window's
edge bin takes its parabolic offset from the raw neighbour outside
the window; a non-negative-curvature peak gets offset 0.

(c) **Edge treatment and reuse** [default] (D22.3.1, D22.6). The
windowed spectrum the FM stage builds from `target_spectra` equals
`fft2` of the periodic-Hann-windowed zero-mean unit-norm crop within
`FM_STENCIL_RTOL` (relative; spec gate 1.6e-16, ledger 204). After
`seed_homographies`, `target_spectra` is bitwise the array
`seed_spectra` returned (a copy taken before the call), and the
translation rows of unrouted slots are bitwise the Stage E rows. G4
background lock: at `(None, None)` the recipe stays within
`FM_ANGLE_TOL_BACKGROUND_DEG` (spec gate 0.151 deg noise-free, 0.252
and 0.281 deg at Poisson 50 under the two filter settings; ledger
213), while the recipe WITHOUT the edge treatment misses by more
than 1 deg on most cases at `(None, None)` -- asserted on a local
re-implementation inside the test, so the premise is checked
whatever the production code does (`FM_LOCK_PREMISE_COUNT`, MTP;
spec gate on the untreated amplitude variant 8 of 9 above 1 deg,
worst 80 deg, ledger 213; the untreated log variant locked at zero
too, ledger 204).

(d) **Seed rows and failure semantics** [default] (D22.4, D22.5,
D22.6). On G1 at `border=0.15`, twists 8, -12, 5 and 15 deg with
projection centre shifts (12, -9), (10, 7), (-8, 5) and (6, -4) px:
the FM row's corner error to the exact imposed homography is below
`FM_SEED_TOL_PX` on every case, and a row composed in the wrong order
(`T(t) R`) exceeds it on every case (spec gate 0.083 to 0.220 px
against 0.898 to 2.655 px, the `2 sin(theta/2) |t|` separation;
ledger 210; the pin sits between the two populations); a de-rotation
about the detector centre instead of the PC (the FM43 mutant, error
`2 sin(theta/2) |c|` with `|c|` about 54 px at `PC_480`) exceeds it
too; the row equals `(c - 1, -s, c tx - s ty, s, c - 1, s tx + c ty,
0, 0)` of its own `theta_hat` and `t` exactly. The de-rotated crop of
`fourier_mellin_derotate` equals the CPU bicubic evaluation of the
target coefficients at `R xi` over every pixel of the D5 bounding box
within 1e-12 (numpy twin), with a `dead_band` arm showing the
dead-band pixels present in the crop. Never NaN, each on a planted
failure through a frozen patch point: `fourier_mellin_angles`
returning NaN for one slot, `fourier_mellin_derotate` returning a
NaN pixel or `ok` False for one slot, `fourier_mellin_translate`
returning a non-finite row for one slot -- that slot's returned row
is `h_T` bitwise, `"fourier_mellin_applied"` False, and the other
slots untouched; a NaN `h_T` stays NaN although a finite FM row
exists; a constant target gives the D2.6 contract exactly as with
`"off"`. The outputs rule: on route-0 and padded slots the angle is
NaN and `applied` False even though the masked branch computed them.

(e) **Capture, end to end** [default for the starred arms; the rest
weekly] (D22.3, D22.4). Expectation: the exact imposed homography,
by the V2 corner metric. Premise asserted at run time on each arm:
the `"off"` path FAILS at `max_iterations=50` (the D5 measurement:
2.5 deg and beyond on G1). G1 at `border=0.05`, twists 2.5*, 3*,
4, 5* and -3*, -5 deg, and the combined rotations (2, -1.5, 3)*,
(-2, 2, 4) and (1, 1, -3.5) deg (rotation vectors, detector frame):
the FM-seeded fit converges and recovers within `FM_CAPTURE_TOL_PX`
(spec gate 0.0078 to 0.0912 px in 2 to 7 iterations with the frozen
partial row, ledger 214). G1 at `border=0.15` [weekly]: 8, 10, 15,
20, 25 and -20 deg converge in at most `FM_CAPTURE_ITERATIONS` (spec
gate 2 to 3) to the error the EXACT-seeded fit reaches, within
`FM_FIXED_POINT_TOL_PX` (spec gate at most 8e-6 px apart, the errors
themselves 0.0010 to 0.0058 px; ledger 216); a window-edge arm at 30
deg is RECORDED, not asserted (spec gate `theta_hat` 29.931 deg,
converged in 3 iterations to 0.033 px, the exact seed's 0.033 px).
Every arm through `run_hrebsd_dic` with `"always"` on a one-row map,
so the runner, the route, the packed row and the props are
exercised, not only the seam.

(f) **Rescue without propagation** [default] (D22.7, D22.8, D22.10;
the V8(a) premise). The V8 ramp map through `run_hrebsd_dic(...,
seed_from_neighbors=False, max_iterations=200)` with `"always"` (no
`xmap`) and with `"auto"` and a matching G7-built `xmap`. Premises
asserted at run time on the `"off"` run: ramp columns 3 to 6 (2.4 to
4.8 deg) and the 8 deg point (`RAMP_UNFITTABLE_INDEX`) do not
converge; the 9 deg tilt (`RAMP_RESCUE_INDEX`) converges (113
iterations). Expected per point, both FM modes unless stated:
columns 3 to 6 converge to the imposed field within the V8
`SAME_OPTIMUM_FRACTION` band, code 1, finite angle; the 8 deg point
CONVERGES (code 1; it is out of frame beyond 6.62 deg, so its error
is the mirror-boundary scale, `FM_RAMP_FAR_TOL_PX`, spec gate 0.053
px from the FM row in 3 iterations at a budget of 200, ledger 216; a
D2.6 outcome there fails the test unless measured so); the 9 deg
tilt converges within `SAME_OPTIMUM_FRACTION` under both modes --
under `"auto"` it is not routed (twist 0), so its result is bitwise
the `"off"` result, code 0, angle NaN; under `"always"` the
acceptance keeps the FM row (spec gate criterion 1.279 against 1.293)
and the fit takes 116 iterations, code 1 (both pinned as measured,
ledger 216); columns 1 and 2 (0.8, 1.6 deg) converge, under `"auto"`
column 1 is not routed (code 0, angle NaN, bitwise `"off"`) and
column 2 is (code 1), under `"always"` both carry the code measured
(expected 1); the reference point carries code 0 under `"auto"`
(angle NaN) and under `"always"` the code measured (expected 0: both
criteria at rounding level) with a finite angle; the constant pattern
keeps the D2.6 contract (NaN `h`, 0 iterations, not converged; code
0, angle NaN) although the retry visits it (D22.8: its forced FM
estimate fails, so it is never refitted); masked points carry code
-1 and angle NaN. No neighbour seeding anywhere (no `seed_round`
prop).

(g) **Acceptance** [default] (D22.5). G5 (if its premises hold): the
D5 seed returns exactly (0, 0), `theta_hat` is within
`FM_ANGLE_TOL_DEG` (premises asserted), the FM row's criterion at the
seed is strictly lower and is kept, and the FM-seeded fit converges
where the `"off"` fit does not, or in fewer iterations (MTP). A
planted wrong angle (`theta_hat + 10` deg through a patched
`fourier_mellin_angles`): `h_T` is kept, bitwise. A planted tie
(`fourier_mellin_criteria` patched to return the same values for
both rows): `h_T` is kept and `applied` is False. G6 (unrelated
pair): no exception, no non-finite row introduced, the acceptance
refuses the FM row (`applied` False), and the fit's `converged` flag
equals the `"off"` path's. Criterion calls, counted on
`fourier_mellin_criteria` (never on `kernels.gather`, which the
de-rotation also calls): on the CPU route at P = 1, exactly two
calls per route-1 seam call (`h_T` and `h_FM`) and none for route 0
or route 2; in the numpy session, calls only in sub-batches with a
nonzero route-1 slot, each at the full P.

(h) **The pre-fit gate** [default] (D22.7). `twist_about_detector_
normal` on G7: imposed twists +-0.5, +-1.5, +-2.5 and +-4.8 deg
recovered within `FM_GATE_TWIST_TOL_DEG` (spec gate 1e-6 deg,
ledger 212) on both detectors, with the sign of `in_plane_fe` (the
V4 sense); a symmetry-equivalent target orientation (raw angle about
90 deg) gives the same twist; a pure 3 deg rotation about the
detector x or y axis gives zero twist to rounding, and twist plus
swing gives the twist; each point is measured against ITS grain
reference on the two-grain map. Routing through
`fourier_mellin_routes` on PLANTED twists: `FM_GATE_DEG` is routed
and `numpy.nextafter(FM_GATE_DEG, 0)` is not, a negative twist is
routed by its magnitude, NaN is routed; through the engine, the
unindexed point, the NaN rotation and the other-phase point are
routed (fail open) and masked points never. The all-identity map and
the constant non-identity map each issue the D22.7 `UserWarning`
(literal asserted) and route nothing pre-fit. Projection link: a
pattern projected through the master-pattern projection from a G7
`g_t` (on both detectors, twists 2.5 and -4.8 deg) equals the
detector-frame deformation construction of the same `R_det` within
1e-10 relative (spec gate 4e-14 to 1.4e-12, ledger 212), and the FM
`theta_hat` of that pair matches the gate's twist within
`FM_ANGLE_TOL_DEG`, so a sign or frame error between the gate and
the projected patterns cannot hide behind `|twist|`. With 0.5 deg
orientation noise the routed set of a 2.5 deg twist population is
recorded, not asserted. The map-level real-data frame check is (m).

(i) **Retry** [default] (D22.8). On G8 under `"auto"`: the
mislabelled 4.0 deg point is not routed pre-fit, fails, is retried
from its FM row (forced, no acceptance) and converges, and its
stored result equals a direct `fit_pattern(state, target,
h0=fm_row)` bitwise (`fm_row` from the numpy seam at P = 1 with
route 2), `num_iterations` the retry fit's own count,
`fourier_mellin_seed` 2; the unrelated-pattern point is retried, its
forced fit does not converge, and it keeps its first-pass result
bitwise (`fourier_mellin_seed` 0, finite angle); the constant pattern
is in the retry subset, its forced FM estimate fails, it gets no
second fit (spy) and keeps the D2.6 contract bitwise; the masked
point and the converged points are never retried; a spy on the
runner shows exactly ONE retry pass whose subset is those three
points in fit order. With `fourier_mellin_angles` patched to fail on
the mislabelled point in the retry, that point gets no second fit
and keeps its first result bitwise (code 0). Under `"always"`: the
mislabelled point converges in the first pass from its FM row (code
1, not retried), and the retry subset is exactly the unrelated point
and the constant pattern.

(j) **CPU route** [default] (D22.9, D22.10, D22.13). On a map mixing
routed, unrouted and retried points under `"auto"`: every point that
is neither routed nor retried, and every routed point whose
translation row won, equals the `"off"` run bitwise; chunksize 1, 3
and the default give bitwise equal results; lazy equals eager
bitwise; two runs are bitwise equal; the seam is reached through the
`_batched` module globals at P = 1 (spy), and `_batched.seed_spectra`
is called exactly once per route-1 or route-2 point and never for an
unrouted one (the FM44 over-routing killer: the numpy seam equals
skimage on these fixtures, so only the count sees it); `_fit_chunk`
returns 14-wide rows on FM runs and 12-wide rows on `"off"` runs
(spy); the props carry the D22.9 dtypes and shapes.

(k) **Numpy session** [default] (D22.6, D22.11, D22.14, D22.18). The
device runner under the numpy session of V9 (patched
`_gpu._make_session`) with `seed_extras`: the route flags arrive in
`SeedBatch.extras` in fit order with 0 on padded slots (spy); the
outputs exist exactly when the route key has a nonzero entry (an
all-zero route key: no outputs and the Stage E rows bitwise); a
planted `seed_homographies` that writes no outputs is read as angle
NaN and `applied` False, without error; a sub-batch with no routed
slot never calls `fourier_mellin_rows` (spy) and returns the Stage E
rows bitwise; a routed slot's row is bitwise equal whether 1 or P
slots of its sub-batch are routed (the masked full-P rule); the FM
state is built lazily, once per grain with a routed slot and never
for a grain without one, and the `resident` it holds IS the lockstep's
resident (identity spy); a forced slot whose FM estimate fails is
inactive in the lockstep (kernel spy) and its packed row carries
`converged` 0 and `fm_applied` 0; `row_slots` carries the FM keys and
width 14 on FM runs and the Stage E dictionary on `"off"` runs; the
retry pass's `_run_chunks_gpu` call receives the first pass's final
B as `chunksize` (spy). VRAM functions (the V9(o) style): with
`fourier_mellin=False`, `_vram_model_terms`, `_vram_model_bytes` and
`_default_batch_size` return the Stage E values bitwise; with True
the terms are at least the False ones, and `_default_batch_size`
returns the False B whenever the FM terms fit the remaining headroom
and a halved one only when they do not (fake free VRAM). Against the
CPU route per point, `theta_hat` and the rows within the numpy-twin
bands (`FM_NUMPY_ANGLE_TOL_DEG`, the V9(f) twin bands for `h`; pinned
bitwise if measured bitwise). Under `seed_precision="complex64"` a
dtype spy shows every FM step after the spectra read in complex128
or float64 (D22.12). Import hygiene: no module-scope cupy, `_engine`
or `_batched` import in `_fourier_mellin.py` (the V9(p) AST arms
extended).

(l) **Device suite** [gated, `-n 0`, the pinned overlay of D21.15]
(D22.12 to D22.14, D22.18). The output keys' device contract
(`cupy.ndarray`, float64 and bool, `(P,)`); `theta_hat` against the
CPU route on G1, G3 and G4 within `FM_ANGLE_PARITY_DEG` at both seed
precisions; acceptance decisions within `FM_ACCEPT_FLIP_COUNT`; `h`
on both-converged FM-seeded points within the D21.8(c) bands per
device precision; iteration and convergence budgets as D21.8(e);
retry routing within `FM_RETRY_FLIP_COUNT`; two runs bitwise at a
fixed B; ROUTING INVARIANCE: a routed slot's row bitwise the same
with 1 of P and P of P slots routed (the FM29 device killer); B in
{8, 32, 40, default} as E4; the default B equal under `"off"` and
`"auto"` on the test card (D22.18); the FM terms of the VRAM model
calibrated against pool high-water marks; the D21.8(h) drift
tripwire unchanged under `"off"`; a throughput record
(`record_property`, asserting completion only).

(m) **Real-data records** [local, RECORDED in the ledger, never a
test] (D22.7, D22.15; plan open questions FQ1, FQ2, FQ3, FQ7, FQ8,
FQ11, FQ13). On the Si-indent file (read only, never copied), with
the executed tutorial's load recipe: (1) the REQUIRED frame oracle,
by the pre-registered rule of D22.7 -- an `"always"` run; the fitted
twist from the polar decomposition of each converged Fe, with the
homography read-out as a cross-check; points with |fitted twist| >=
1 deg, converged, outside the crater; the correlation for the
as-read frame and the four alternative frames; the slope of the
xmap twist on the fitted twist and the Deming slope with the
far-field Hough variance; PASS if the as-read frame correlates best
and its slope lies in [0.8, 1.25], else the stage stops for Johan;
(2) the whole map under `"auto"` at `(None, None)` and `(0.05,
None)` on both backends -- routed, accepted, retried, converted and
worsened counts against `"off"`, wall time, the device seed share,
the number of sub-batches and slots the FM branch ran on, the
`theta_eff` of D22.7 beside the twist for every point (the FQ1, FQ2
candidate), and the distribution of |`theta_hat`| and |twist| over
the routed points (FQ11: reopen if any routed |twist| exceeds about
25 deg or any `theta_hat` sits at the window edge); (3) far256 and
patch C under `"always"`; (4) the D5 anchor census of FQ8 (the
fraction of points whose D5 seed is exactly zero while a
band-limited phase correlation is not).

V10 requirement-to-oracle map:

| D22 item | oracle(s) | suite |
|---|---|---|
| D22.1 API, default path, check order, forwarding | (a) | default |
| D22.2, D22.3 the FM angle, look-up table, peak | (b), (c) | default + gated parity |
| D22.4 de-rotation and the row | (d), (e) | default |
| D22.5 acceptance, failure semantics | (d), (g) | default |
| D22.6 seam extension, outputs, patch points | (c), (d), (k) | default + gated |
| D22.7 gate, routing, warning | (h), (f); (m) frame | default + ledger |
| D22.8 retry | (f), (i), (k) | default |
| D22.9 props and packed row | (a), (f), (i), (j), (k) | default |
| D22.10 CPU route | (j) | default |
| D22.11 interplay, device wiring | (a), (k) | default |
| D22.12 to D22.14 dtypes, determinism, parity | (j), (k), (l) | default + gated |
| D22.15 performance | (l) record, (m) | gated + ledger |
| D22.16 docs | (a) docstring test | default |
| D22.17 dependencies | (k) import hygiene | default |
| D22.18 VRAM model, batch choice | (k), (l) | default + gated |

Gate commands, recorded verbatim with their output at each Stage F
gate (Git Bash, from the worktree; at most `-n 2` while the machine
is shared, the 2026-10-06 memory lesson):

```
# default suite (cupy absent: the gated classes skip at stage (a))
uv run pytest tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0
uv run pytest tests/test_indexing tests/test_signals -k hrebsd -n 2
# gated suites: Stage F, then Stage E unchanged (pinned overlay)
OVERLAY="--with cupy-cuda12x==14.2.0 \
  --with nvidia-cufft-cu12==11.4.1.4 \
  --with nvidia-cublas-cu12==12.9.2.10 \
  --with nvidia-cusolver-cu12==11.7.5.82 \
  --with nvidia-cusparse-cu12==12.5.10.65 \
  --with nvidia-nvjitlink-cu12==12.9.86"
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_fourier_mellin.py -n 0 -q --weekly
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_gpu.py -n 0 -q --weekly
# coverage: default run, then both gated runs appended
uv run pytest tests/test_indexing tests/test_signals -k hrebsd \
  -n 0 --cov=kikuchipy.indexing._hrebsd --cov-report=
KIKUCHIPY_EXPECT_GPU=1 uv run $OVERLAY pytest \
  tests/test_indexing/test_hrebsd_fourier_mellin.py \
  tests/test_indexing/test_hrebsd_gpu.py -n 0 --weekly \
  --cov=kikuchipy.indexing._hrebsd --cov-append \
  --cov-report=term-missing
```

<!-- ================ BLOCK 2: spec-gate ledger (200 on) ================ -->

#### V10 recorded results, spec gate (2026-10-07)

Measured on machine A (ledger 89) by the Stage F spec-phase
exploration: a literature and method study with a throwaway CPU
prototype, a read-only seam and frame probe, a second throwaway CPU
prototype of the full seed, and, while drafting this spec, a
head-to-head of the two prototypes' recipes on identical inputs and
an end-to-end check of the recipe D22 recommends, and, after the
spec review, a re-measurement of the frozen recipe wherever the
critics found a number taken from another recipe or row (ledger
216). Numbers from a recipe other than the frozen one (D22.3 V8, the
partial row, the acceptance) are tagged with their recipe. CPU ONLY
(no GPU work: Stage E was measuring GPU timing pins at the time), one
process each. Nothing of the prototypes is committed and no
repository file was edited; the scratchpad does not survive the
session, so every number is carried here. These are PROTOTYPE
numbers: the Stage F gates re-measure every one that feeds a pin or
a decision.

200. **Conditions of every entry below.** Machine A (i7-13700H, 20
    logical cores, 32 GB, Windows 11 build 26200); the worktree
    `.venv` (CPython 3.13.12, numpy 2.4.6, scipy 1.17.1, numba
    0.65.1, scikit-image 0.26.0, orix 0.14.2); OMP, MKL, OpenBLAS and
    numba threads at 1 in the second prototype's fixture module and
    every drafting run, single-threaded numpy timings in the method
    study. Engine
    code: the CPU path of the working tree on top of f297867e, which
    Stage E leaves unchanged (its default-path pins); the prototypes
    called only `fit_pattern`, `initial_guess`, `ReferenceState`,
    `preprocess`, `spline_coefficients`, `evaluate`,
    `zero_mean_normalize`, `fe_to_homography`,
    `homography_parameters`, `shape_function` and
    `per_point_pc_pixels`; Stage E's in-progress `_batched.py` was
    read for the seam's names only. Synthetic oracle: the V3/V4
    deformed master at 480x480 through helper copies of
    `test_hrebsd_engine.py` (`PC_480` unless stated; crop 432x432).
    Real data: `C:\Users\westraadt.1\Repos\kikuchipy\AGH__Si_indent_
    1_512x672.h5oina`, READ ONLY, never copied (h5py reads checked
    bitwise equal to `kp.load`), binning 2 (512x622, crop 460x560),
    reference (10, 10) at PC (328.3, 90.2, 379.4) px, per-point PCs as
    `run_hrebsd_dic` builds them, `border=0.05`. The machine was
    shared (the HROSM session, the Stage E workflow's test runs);
    identical code timed 33 to 58 ms in different runs, so timings
    carry about +-30 per cent; free virtual memory was 2.3 GB when
    the drafting runs started. Scripts (session scratchpad
    `stageF_spec/`): the method study `fm_lib.py`, `exp1_angle.py` to
    `exp10_anchor.py` (outputs `exp7.out`, `exp7b.out`); the probes
    `probe_frames.py`, `probe_si_gate.py`, `probe_peak_gate.py`,
    `budget.py`; the seed prototype `proto/` (`fixtures.py`, `fm.py`,
    `sweep.py`, `sweep2.py`, `gridstudy.py`, `gridstudy2.py`,
    `realdata.py`, `realdata2.py`, `realrot.py`, `cost.py`,
    `sparse_polar.py`, `rimseed.py`, `zerolag.py`, `inframe.py`,
    `analyze_*.py`, outputs in `proto/out/`); the drafting runs
    `proto/h2h.py`, `proto/e2e.py`, `proto/cost_v8.py` and
    `h2h_gatecheck.py` (logs `proto/out/h2h.log`, `e2e.log`,
    `cost_v8.log`; `h2h.json`, `e2e.jsonl`); the spec-review runs
    `proto/e2e_partial.py` and `proto/deadband.py` (logs
    `proto/out/e2e_partial.log`, `e2e_partial.jsonl`,
    `deadband.log`) and the critics' `stageF_critic/subbatch_amp.py`
    and `stageF_critic/dilution.py`. Below, "periodic" is
    the reused spectrum minus the Moisan smooth component, "hann" a
    windowed spectrum, "raw" the reused spectrum untreated.

201. **FM angle accuracy on the oracle, and the physics checks
    (requirements D22.2, D22.3).** (i) Pure twists 0 to 15 deg plus
    -3 and -12 (13 angles), 360 angles, rho
    0.05-0.35, radial-mean ZNCC, +-30 deg window, parabolic peak;
    max |error| in deg:

    | spectrum | amplitude | log |
    |---|---|---|
    | raw (reused) | 0.076 | 0.055 |
    | periodic (reused + Moisan) | 0.079 | 0.031 |
    | hann (a second FFT) | 0.015 | 0.012 |

    Periodic-log sensitivity: 180 / 360 / 720 angles 0.036 / 0.031 /
    0.032; rho_max 0.25 / 0.35 / 0.45: 0.046 / 0.031 / 0.027;
    rho_min 0.02 / 0.05 / 0.10: 0.024 / 0.031 / 0.032; a 1-bin and a
    0.5-bin radial step identical. Angular peak 0.90-0.99, 1.000 at
    zero. (ii) In-plane-axis bias, predicted `(w1 xbar + w2 ybar) /
    (2 DD)` with the subregion centroid at (+37.9, -38.1) px and DD
    242.4 px: +-0.235 deg for 3 deg about x and about y; the
    windowed FM measured +0.231, -0.178, -0.230 and +0.190 deg. On
    the Si-indent geometry (centroid (-17.3, +165.8) px from the PC)
    the formula gives `theta_FM ~= w3 - 0.023 w1 + 0.218 w2`, so 5
    deg about y reads as 1.1 deg of twist. (iii) Pseudo-scale
    `(3/2)(w2 xbar - w1 ybar) / DD`: 1.2 per cent for 3 deg on the
    oracle, 5.7 per cent for a 5 deg w1 on the Si geometry -- the
    reason D22.2 excludes scale.

202. **Capture end to end on the oracle (requirements D22.3,
    D22.4; the D5 premise).** (i) `fit_pattern(max_iterations=50)`,
    periodic-log FM (the METHOD-STUDY recipe; the frozen recipe's
    rows are ledger 214), the engine's spline de-rotation,
    `W0 = R(theta) T(t)`; V2 corner error to the exact homography:

    | rotation vector (deg) | translation seed: conv, it, err px | FM seed: conv, it, err px |
    |---|---|---|
    | (0, 0, 1) | yes, 5, 0.004 | yes, 2, 0.004 |
    | (0, 0, 2) | yes, 8, 0.011 | yes, 2, 0.011 |
    | (0, 0, 2.5) | NO, 50, 148 | yes, 2, 0.012 |
    | (0, 0, 3) | NO, 50, 173 | yes, 2, 0.013 |
    | (0, 0, 4) | NO, 50, 221 | yes, 2, 0.008 |
    | (0, 0, 5) | NO, 50, 98 | yes, 2, 0.010 |
    | (0, 0, -6) | NO, 50, 171 | yes, 2, 0.033 |
    | (0, 0, 8) | NO, 50, 53 | yes, 3, 0.160 |
    | (3, 0, 0) | yes, 9, 0.117 | yes, 9, 0.117 |
    | (2, -1.5, 3) | NO, 50, 53 | yes, 7, 0.017 |
    | (-2, 2, 4) | NO, 50, 68 | yes, 7, 0.091 |
    | (1, 1, -3.5) | NO, 50, 34 | yes, 6, 0.077 |

    At 5 deg and beyond the corners leave the pattern at
    `border=0.05` (ledger 215), so the 8 deg 0.160 px is the mirror
    boundary, not the seed. (ii) The second prototype (whitened
    per-radius recipe, nearest rule, `W(R) W(T)`; NOT the frozen
    recipe), 113 cases: twists
    0 to 30 deg and -3, -10, with projection centre shifts and
    tilts, on four set-ups (A: `PC_480`, border 0.05; B: the V8 PC,
    border 0.05; C: `PC_480`, border 0.15; A and C also unfiltered).
    The translation seed converges up to 2.0 deg on A and B and
    fails from 2.5 deg on both (pure, with a (10, 7) px shift, with a
    y tilt), with ERRATIC success beyond (B converges at 3.0 deg; C
    up to 3.0; A unfiltered up to 4.0; C unfiltered at 4 and 6 but
    not 3). The FM seed: 113 of 113 converge to the exact-seed fixed
    point within 7.5e-5 px in 1 to 9 iterations (median 2 on pure
    twists; 9 only with 3 deg tilts); 112 of 113 within 1 px of the
    exact homography (A unfiltered at 15 deg, 1.074 px, as the exact
    seed). Angle error on pure twists: median 0.006, p90 0.041, max
    0.33 deg (at 0.5 deg true, pulled towards 0; those fits still
    converge in 3 to 4 iterations). (iii) The translation seed's
    phase-correlation peak: 1.000 at 0 deg, 0.037 at 1, 0.031 at 2,
    0.018 at 2.5 -- it collapses before the seed fails, so it is no
    gate (ledger 212).

203. **Real Si patterns rotated rigidly about the PC (requirements
    D22.3; the D5 premise on real data).** Recipes: (i) the method
    study's variants, (ii) the second prototype's whitened
    per-radius recipe; neither is the frozen one. (i) Target (10, 13), a
    different real pattern, rotated about the reference PC by the
    engine's spline; `max_iterations=200`; FM error over 0 to 8 deg
    (both signs) -0.027 to -0.002 deg (raw-amplitude), -0.016 to
    +0.008 (raw-log), -0.016 to +0.004 (periodic-log), -0.048 to
    -0.009 (hann-log). Default band-pass: the translation seed
    converges at 0, 1 and 2 deg in 7, 19 and 124 iterations and
    fails at 3, -4, 5 and 8 deg (residual about 1.93); the FM seed
    converges at every angle in 6 to 8 iterations (residual 0.1771 to
    0.1773 where both converge; 0.189 at 8 deg, the mirror boundary).
    `(None, None)`: translation 14, 26, 32, 47 and 70 iterations at
    1, 2, 3, -4 and 5 deg, not converged at 8; FM 5 to 8 at every
    angle. FM-seeded fits recover the imposed angle to about 1e-3
    deg up to 5 deg. Real angular peaks 0.96-0.98. (ii) Four
    far-field targets ((12, 40), (20, 20), (30, 60), (9, 13)) rotated
    by an INDEPENDENT warper (skimage, bicubic, reflect), budget
    500: the FM seed converged 16 of 16 at every angle from -6 to 10
    deg under both filter settings, in 5 to 11 iterations, within
    2.5e-3 px of the exact-seeded fit, angle error at most 0.021 deg;
    the translation seed reached 6 deg unfiltered (107 to 168
    iterations) and failed at 10, and reached 3 deg with the default
    band-pass (196 to 206 iterations) and failed from 4.

204. **Edge treatment and the background lock (requirements
    D22.3.1).** (i) G4's detector-fixed background, noise-free, max
    |error| over 0 to 8 deg: with `(0.05, None)` raw at most 0.044,
    periodic 0.012, hann 0.014 deg; with `(None, None)` the RAW
    spectrum LOCKS AT ZERO (errors -0.97, -1.99, -3.00, -5.02 and
    -7.99 deg at 1, 2, 3, 5 and 8 deg), periodic 0.012 and hann
    0.014. (ii) The exact frequency stencil of the periodic Hann
    window equals a second FFT of the windowed crop to 1.6e-16
    relative; the stencil evaluated only at the samples' neighbours
    equals the dense one to 9e-16. (iii) On 32 real rim points
    (`(None, None)`), the whitened recipe without any window had a
    46.8 deg outlier where the windowed one had none (ledger 213).
    (iv) G6, the unrelated pair: FM returned 12.36 deg with angular
    peak 0.56; both rows are junk. CORRECTED at the spec review
    (ledger 216 (iii); the original "the fit fails by the D2.6
    contract" was inferred, not measured): at the V8 PC and a budget
    of 200 the translation-seeded fit runs to the cap with a FINITE
    `h` (residual 1.89), the acceptance refuses the FM row, and a
    forced FM-seeded fit does not converge either.

205. **Shot noise (requirements D22.3; plan open question FQ4).**
    Poisson noise at full-scale counts of 50, 20 and 10 on the G4
    oracle, periodic correction, max |error| over 0 to 8 deg:

    | magnitude, rho_max | (0.05, None) 50 / 20 / 10 | (None, None) 50 / 20 / 10 |
    |---|---|---|
    | amplitude, 0.20 | 0.121 / 0.203 / 0.843 | 0.290 / 0.402 / 1.518 |
    | amplitude, 0.35 | 0.133 / 0.543 / 0.379 | 0.235 / 0.412 / 3.420 |
    | log, 0.20 | 0.176 / 0.178 / 1.041 | 0.581 / 0.837 / 3.289 |
    | log, 0.35 | 0.365 / 0.939 / 0.814 | 0.517 / 0.869 / 3.418 |

    At low counts amplitude over a band ending at 0.20 cycles/px was
    the most robust; the angular peak tracked reliability (0.85-0.88
    at 50, 0.44-0.62 at 10) but is 0.25-0.82 on real deformed points
    where the seed still helps (ledger 206), so it is no gate. Fits
    from either seed did not converge at these synthetic noise
    levels in 100 iterations, so this entry says nothing end to end.

206. **Real deformed patterns, the indent's south rim (requirements
    D22.5, D22.8).** (i) Method study, 30 points (rows 125-129 x
    columns 115-119 plus (120, 110), (130, 125), (118, 122), (135,
    118), (126, 108)), `max_iterations=500`, periodic-amplitude FM
    over rho <= 0.20; T the D5 seed, P the FM partial row, C the FM
    complete row:

    | filter | T: converged, iterations total (median) | P | C |
    |---|---|---|---|
    | `(0.05, None)` | 9/30, 13234 (500) | 23/30, 8456 (261) | 24/30, 7494 (214) |
    | `(None, None)` | 30/30, 3503 (122) | 30/30, 2840 (92) | 30/30, 2721 (91) |

    Two of T's 9 converged default-filter points sat at residuals
    1.95 and 1.86 where FM reached 1.14 and 0.53 (the false
    convergence of D22.8). Where all three converge they reach the
    same residual (median 0.3101 unfiltered). Polar decomposition of
    the converged unfiltered Fe: w1 0.75-2.4 deg, w2 -1.2 to 2.0,
    twist 0.06-0.95 deg -- the rim's gain is about translation and
    perspective, not twist. FM over-read the twist by -0.41 to +0.86
    deg; the first-order bias formula removed half to two thirds of
    the far-rim over-read and changed no fit by more than 3
    iterations. The D2.7 criterion at the seed: T 1.06-2.14 on every
    point; P and C 0.26-0.69 wherever `theta_FM >= 0.43` deg (17
    points), and between 0.19 lower and 0.07 higher than T elsewhere.
    (ii) Second prototype, 32 far-field points (rows 8-11, columns
    20-27) and 32 rim points (rows 125-130 x 115-119 plus (124, 117),
    (124, 118)), `(None, None)`, budget 500, every point fitted from
    both seeds: both converge 32/32 in both groups; `|delta h|`
    median / max 1.6e-4 / 4.1e-4 px far and 2.8e-4 / 4.1e-3 px rim;
    residual relative difference at most 3.9e-6 and 3.1e-7;
    iterations median 8 / 8 far and 122 / 111 rim; FM angle against
    the fitted in-plane rotation at most 0.011 deg far and median
    0.049, max 0.19 deg rim -- no harm.

207. **The D5 seed is anchored at zero shift on real Si
    (requirements D22.5, D5 amendment; plan open question FQ8).** (i)
    At 10 of 10 deformed points under both filter settings, the
    full-band phase correlation of `initial_guess` returns exactly
    (0, 0) (peak 0.094-0.104), while a band-limited phase correlation
    (cross-power masked to rho in [0.02, 0.20] cycles/px) or the
    plain correlation returns 6 to 18 px, consistent with the
    converged translations (at (125, 115): band (+12, -11) default
    and (+13, -11) periodic, plain (+10, -8), converged (+10.2, -7.0)
    px at the PC). At undeformed points D5 also returns (0, 0) where
    the others give 1 px, the order of the map's PC span. (ii) The
    second prototype on rim points: (0.00, 0.00) against converged
    translations of 10 to 21 px (median 16.2); zero-lag peak
    0.093-0.101 against 0.012-0.016 near the true shift; seeds 22 to
    66 px off (corner norm); unfiltered the fits recover in 77 to 152
    iterations, with the default band-pass 3 of 3 checked did not
    converge in 500. (iii) Drafting run, ledger 214: all 24 rim seeds
    were exactly (0.0, 0.0). Cause: whitening gives every frequency
    equal weight, and detector-fixed content (camera fixed pattern,
    hot pixels, scintillator texture), identical in both images, owns
    the high frequencies. The FM branch escapes it because the
    de-rotation also rotates the target's fixed content; the escape
    was seen from about 0.4 deg of de-rotation upward.

208. **Non-square crops and the composition order (requirements
    D22.3.3, D22.4).** Real Si, crop 460x560, target (10, 13) rotated
    about the PC, amplitude, rho <= 0.20: the physical-frequency
    look-up table measured +0.002, +1.997, +4.988 and +7.990 deg at
    0, 2, 5 and 8 deg (angular peak 0.99-1.00); a table in bin units
    on both axes measured +0.003, +2.018, +5.054 and +8.185 deg (peak
    down to 0.88 and 0.73), a bias of about 2.3 per cent of the
    angle that stays inside the IC-GN basin (this method-study
    recipe; with the frozen recipe on the G3 synthetic the bias is
    0.34 deg at 8 deg and 2.29 deg at 20, ledger 216 (iv)) -- only a
    seed-level angle test at 5 deg or more on a non-square crop, or
    the look-up-table coordinate pin of V10(b), kills it. The composition mutant `T(t) R`
    was indistinguishable from `R T(t)` on that pair (t = 0); ledger
    210 measures the fixture that separates them.

209. **The 180 deg ambiguity (requirements D22.3.7).** Second
    prototype on 32 real rim points, `(None, None)`: choosing
    between theta and theta + 180 by the criterion at the seed picked
    the 180 deg candidate on 3 points (scores 0.688 against 0.597,
    0.513 against 0.240, 0.025 against 0.000), because the smooth
    background is close to point symmetric about the PC; (125, 119)
    then converged 1108.6 px and (127, 117) 1083.4 px from the
    translation-seeded result, and (130, 118) did not converge in
    500. On the synthetic sweep the rejected candidate never scored
    above 0.096. The nearest rule (theta in [-90, 90)) had 0 failures
    and saves one de-rotation and one cross-correlation (seam-form
    seed 80 / 101 ms against 173 / 232 ms at 432x432 / 460x560).

210. **Partial against complete initialisation (requirements
    D22.4.3; plan open question FQ5).** (i) Method study on the
    oracle, iterations and seed corner error: (3, 0, 0) partial 9,
    19.06 px against complete 4, 1.43; (0, 3, 0) 9, 19.03 against 4,
    0.89; (2, -1.5, 3) 7, 16.54 against 4, 1.38; (-2, 2, 4) 7, 19.19
    against 4, 1.31; (1, 1, -3.5) 6, 12.59 against 4, 0.84; (3, 3, 5)
    17, 36.95 against 6, 1.98; final homographies equal to 1e-4 px.
    On the real rim (ledger 206) 23 against 24 of 30 converged with
    the default band-pass and 2840 against 2721 iterations
    unfiltered. (ii) Drafting run (`e2e.py`, the D22 recipe): on the
    oracle's pure twists both forms take 2 to 3 iterations; on the
    five combined rotations partial 6 to 9 and complete 4, both to
    the same final error. (iii) Drafting run, `PC_480`,
    `border=0.15`, twists with PROJECTION CENTRE SHIFTS (a pure
    translation, `make_case(pc_shift_px=...)`), seed corner error to
    the exact homography, px:

    | twist, shift | D5 seed | partial | partial `T(t) R` (mutant) | complete, d = R t | complete, d = t (Ernould E.2-E.3) |
    |---|---|---|---|---|---|
    | +8, (12, -9) | 132.03 | 0.177 | 2.226 | 24.471 | 24.521 |
    | -12, (10, 7) | 51.28 | 0.105 | 2.655 | 13.657 | 12.245 |
    | +5, (-8, 5) | 29.29 | 0.083 | 0.898 | 12.245 | 12.498 |
    | +15, (6, -4) | 77.52 | 0.220 | 1.946 | 11.077 | 10.854 |

    The complete row reads a beam-scan translation as two lattice
    rotations and seeds 11 to 24 px off (its fits still converged in
    5 to 8 iterations to 0.001-0.003 px); the partial row seeds
    within 0.22 px, and its wrong-order mutant sits 0.90 to 2.66 px
    off, the `2 sin(theta/2) |t|` separation the D22.4 composition
    pin uses. The two complete orderings are both valid
    parameterisations and differ at second order. Hence D22.4 adopts
    the partial row; fitted from it, these four cases converge in 2
    to 3 iterations to 0.0011-0.0027 px (ledger 216 (ii)). Sign check of
    `fe_to_homography` used by the complete row (DD 242.35 px): 1 deg
    about y gives h13 = +4.230 px (= DD tan 1 deg), 1 deg about x
    gives h23 = -4.230, 1 deg about z gives h21 = +0.01745.

211. **Cost on one CPU thread (requirements D22.15).** (i) Method
    study, 432x432 per pattern: `fft2` complex128 10.4 ms; log1p of
    the magnitude with a per-slot median 5.3; Moisan smooth spectrum
    rebuilt per call 9.9; polar profile through a 187200-entry table
    2.0 (0.5-bin step, 374400 entries, 4.2); 1D correlation and
    peak 0.05; spline de-rotation of 186624 px 20.0; skimage phase
    XC 26.3. Batched at P = 16 with precomputed tables, the whole
    angle path cost 5.89 ms (432x432) and 8.01 ms (460x560), about
    0.8 to 0.95 of one batched `fft2`. (ii) Second prototype (its
    whitened 180x32 recipe), medians of 30, 432x432 / 460x560: CPU
    `initial_guess` 44-61 / 55-68 ms; dense angle path 14-18 / 18-23
    ms, sparse 1.3 / 1.4 ms; dense numpy stencil 10-18 ms against a
    second `fft2` of 3.3-6.5 ms; de-rotation gather over the box
    18-19 / 25-27 ms; XC on the de-rotated crop (seam form) 40-49 /
    48-57 ms; one criterion evaluation 20-24 / 28-30 ms; the FM seed
    80 / 101 ms dense, about 66 / 83 ms with the sparse angle path;
    one IC-GN iteration 21-23 / 31 ms; the per-reference precompute
    19-22 / 26-30 ms. (iii) Drafting run (`cost_v8.py`, the D22
    recipe, dense, scipy FFT, medians of 15, CPU load 4 per cent):
    look-up table 237600 entries (165 radii) built in 4.0 ms and
    252000 (175 radii) in 4.7 ms; `fft2` 1.43 / 2.57 ms; stencil
    6.54 / 8.80; log1p of the magnitude 1.29 / 1.79; gather and
    radial mean 1.16 / 1.24; correlation 0.014; the angle path 9.00
    / 11.85 ms per slot. The dense numpy stencil dominates and costs
    more than a second `fft2`; a fused kernel is the obvious
    follow-up, and on the device it is one elementwise pass. GPU
    cost: NOT measured (no GPU work at this gate); about 2.2 to 2.5
    ms per SLOT of a sub-batch holding a routed point is an inference
    from the 2.16 ms device seed (ledger 92); the slot count is
    ledger 216 (vii).

212. **The CrystalMap gate (requirements D22.7; plan open questions
    FQ1, FQ2, FQ13).** (i) Frames, synthetic (`probe_frames.py`, 240 px,
    Ni master, sample tilt 70, detector (tilt, azimuthal, twist) (0,
    0, 0) and (10, 4, 1.5)): `to_matrix` and `rotate_vector` agree to
    1.1e-16; a target built from an orientation, `g_t = g_r @
    R_s.T` with `R_s = M^T Fe_det M`, projects the same pattern as
    the V4/V8 deformation construction to 4e-14-1.4e-12 relative; the
    twist is recovered to 1e-6 deg at -3.0, +2.5 and +4.8 deg, with
    `|R_det - Rz| <= 9e-16`; a cubic symmetry operator applied to
    `g_t` gives a raw angle of 88 to 94 deg and the reduced twist
    equals the truth (orix `angle_with` agrees). (ii) Threshold, the
    113-case sweep of ledger 202 with exact orientations, "needed" =
    the translation seed fails:

    | threshold (deg) | needed but not routed | routed, not needed |
    |---|---|---|
    | 1.0 | 0 | 30 |
    | 1.5 | 0 | 20 |
    | 2.0 | 0 | 15 |
    | 2.25 | 0 | 15 |
    | 2.5 | 6 | 10 |

    Routed-but-not-needed points converged to the same answer in
    fewer iterations. (iii) The Si-indent map from its Euler angles
    (no pattern read): detector from the file header (sample tilt
    74.998, tilt 6.993), `n_s = M^T e_z = (0.3745, 0, 0.9272)`; of
    58055 points within 5 deg of the reference, |twist| median 0.014,
    p99 1.305, p99.9 2.138, max 2.588 deg, and 1090 / 385 / 90 / 5
    points above 1.0 / 1.5 / 2.0 / 2.5 deg, in rows 79-142, columns
    119-141; the far field (rows 20:36, columns 20:36) at most 0.031
    deg; patch C 122 and 24 of 650 above 1.5 and 2.0 deg. Over all
    58500 points with the 728-point crater (CCC < 0.35) separated:
    1515 / 793 / 483 / 389 above 1.0 / 1.5 / 2.0 / 2.5 deg, of which
    1012 / 336 / 73 / 4 outside the crater, in a ring 3 to 7 um from
    its centre; far-field rows 0-39 median 0.0055, max 0.041 deg.
    (iv) Frame sensitivity: assuming a sample-frame offset about
    sample z of +90, 180 or -90 deg moves the count above 1.5 deg
    from 385 to 79, 116 or 280 (correlation with the as-read values
    0.38, -0.29, 0.54); the tutorial passes the raw h5oina Euler
    angles, and cubic stiffness (D9) is blind to a 90 deg z offset,
    so no earlier stage pins this frame. (v) Drafting check
    (`h2h_gatecheck.py`, on the second prototype's records): on 32
    rim points the as-read twist against the converged in-plane
    rotation `atan2(h21 - h12, 2 + h11 + h22)` has correlation
    0.836, least-squares slope 0.45, intercept 0.13 deg, |difference|
    median 0.075 and max 0.35 deg, signs agreeing on 24 of 32; the
    range is only 0.05 to 0.45 deg, so the frame is NOT pinned (plan
    open question FQ13); on 32 far-field points both quantities are
    below 0.03 deg (correlation -0.04, Hough-limited). Spec review
    (ledger 216 (viii)): with the fitted rotation as the regressor
    the slope is 1.545 (geometric mean 0.54); the noisy input twist
    as regressor attenuates the 0.45, so these 32 points bracket 1
    and decide nothing, and the frame decision moves to the
    pre-registered rule of D22.7 on the whole map. (vi)
    Frame-free gates, rejected: the translation seed's
    phase-correlation peak at the V8 PC (480 px, default band-pass)
    reads 0.097, 0.037, 0.033, 0.022, 0.020, 0.021 and 0.016 at 0.5,
    1, 2, 2.5, 3, 4 and 4.8 deg, with a peak-to-second ratio of 1.05
    to 1.44 at every angle; the coarse shift is spurious at 2.5 deg
    (10 px), 4.0 (14 px) and 4.8 (80 px) but 0 at 3.0, so neither the
    height nor the shift separates the regimes.

213. **Head-to-head of the angle recipes on identical inputs
    (requirements D22.3; drafting run `h2h.py`, 46.6 s).** Eleven
    variants of the angle, each evaluated on the same 11 sets (no
    fits; nearest rule; errors against `atan2(h21 - h12, 2 + h11 +
    h22)` of the exact homography, or of the converged
    translation-seeded fit on real rim points): synthetic G1 sets of
    9 cases (the off-grid twists of V10(b) plus the two combined
    rotations) clean, with Gaussian noise, and under the G4
    background (noise-free and Poisson 50, at both filter settings);
    real far-field targets rotated by skimage (16), real rim points
    (32) and rim points rotated by 2.71 or -4.42 deg (16), at both
    filter settings. Max |error| in deg:

    | set | V1 whitened per-radius (180 angles, rho .02-.40, 32 radii, hann, log) | V3 radial-mean ZNCC (360, .05-.20, Moisan, amplitude) | V8 radial-mean ZNCC (360, .02-.40, hann, log) -- D22 | V10 as V8 with Moisan | V11 as V3 untreated (raw) |
    |---|---|---|---|---|---|
    | synthetic clean | 0.225 | 0.082 | 0.123 | 0.066 | 0.084 |
    | synthetic noise 0.5 sd | 0.194 | 0.094 | 0.113 | 0.117 | 0.098 |
    | background, `(None, None)` | 0.254 | 0.228 | 0.151 | 0.149 | 80.346 (8 of 9 > 1) |
    | background + Poisson 50, `(None, None)` | 0.366 | 0.306 | 0.252 | 0.182 | 80.378 (8 of 9 > 1) |
    | background + Poisson 50, `(0.05, None)` | 0.399 | 0.303 | 0.281 | 0.223 | 0.290 |
    | real far rotated, `(None, None)` | 0.223 | 0.032 | 0.034 | 0.023 | 0.035 |
    | real rim, `(None, None)` | 0.186 | 0.772 | 0.595 | 0.606 | 0.840 |
    | real rim rotated, `(None, None)` | 0.795 | 0.769 | 0.604 | 0.623 | 0.776 |
    | real far rotated, `(0.05, None)` | 0.212 | 0.030 | 0.037 | 0.019 | 0.032 |
    | real rim, `(0.05, None)` | 0.491 | 0.759 | 0.604 | 0.592 | 0.759 |
    | real rim rotated, `(0.05, None)` | 0.799 | 0.761 | 0.614 | 0.591 | 0.762 |

    Worst over the 11 sets: V1 0.799, V3 0.772, V8 0.614, V10 0.623;
    the other crossings: V1 with the per-radius plain correlation
    0.565, V3 with the window 0.718, V3 per-radius 0.786, V1 over
    rho .05-.20 with amplitude 0.762, V3 with the window at 180
    angles 0.730, and V1
    with the Moisan correction 46.678 (one outlier, rim rotated,
    default band-pass). No variant except V11 estimated beyond 30 deg
    on any case. On the rim the whitened V1 tracked the fitted
    in-plane rotation best (median 0.049 against 0.339 for V8), but
    the second prototype's own sweep (`gridstudy2.py`, 19 variants)
    found it fragile in its neighbourhood: 720 angles gave a 61.1 deg
    outlier on the rim, rho up to 0.5 a 41.6 deg one, no window a
    46.8 deg one, while the plain-correlation variants had none. V8
    and V10 are equivalent within 0.07 deg everywhere; D22.3 takes V8
    (the window is Ernould's own edge treatment and its stencil has
    an exact one-line oracle) and keeps V10 and the amplitude and
    narrow-band variants open under FQ4.

214. **End-to-end check of the D22 recipe (requirements D22.3 to
    D22.5; drafting run `e2e.py`, 127 s).** The V8 angle, the
    engine's bicubic `evaluate` over the bounding box for the
    de-rotation, `initial_guess` on the de-rotated crop, the partial
    (P) and complete (C) rows, and the D2.7 criterion at each seed
    (`fit_pattern(max_iterations=0)`). (i) G1, `border=0.05`,
    default band-pass, budget 50, V2 corner error to the exact
    homography:

    | rotation vector (deg) | theta_hat (true) | T: conv, it, err px, criterion at seed | P: conv, it, criterion | C: conv, it, criterion |
    |---|---|---|---|---|
    | (0, 0, 1) | +1.002 (+1.000) | yes, 5, 0.0040, 0.756 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 2) | +1.994 (+2.000) | yes, 8, 0.0113, 1.647 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 2.5) | +2.497 (+2.500) | NO, 50, 148.26, 2.048 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 3) | +3.013 (+3.000) | NO, 50, 172.88, 1.990 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, 4) | +4.026 (+4.000) | NO, 50, 221.43, 2.023 | yes, 3, 0.002 | yes, 3, 0.002 |
    | (0, 0, 5) | +5.007 (+5.000) | NO, 50, 97.93, 1.950 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, -3) | -2.995 (-3.000) | NO, 50, 23.11, 2.019 | yes, 2, 0.001 | yes, 2, 0.001 |
    | (0, 0, -5) | -5.000 (-5.000) | NO, 50, 164.72, 1.990 | yes, 2, 0.002 | yes, 2, 0.002 |
    | (3, 0, 0) | +0.224 (0) | yes, 9, 0.1171, 0.702 | yes, 9, 0.706 | yes, 4, 0.090 |
    | (0, 3, 0) | -0.157 (0) | yes, 9, 0.0373, 0.673 | yes, 9, 0.665 | yes, 4, 0.063 |
    | (2, -1.5, 3) | +3.267 (+3.000) | NO, 50, 53.21, 1.983 | yes, 7, 0.490 | yes, 4, 0.123 |
    | (-2, 2, 4) | +3.759 (+4.001) | NO, 50, 68.46, 1.950 | yes, 7, 0.565 | yes, 4, 0.132 |
    | (1, 1, -3.5) | -3.485 (-3.500) | NO, 50, 33.53, 1.987 | yes, 6, 0.259 | yes, 4, 0.025 |

    Every FM fit reached the error the converged T fit reaches where
    T converges (0.0040 to 0.1171 px) and 0.0078 to 0.0912 px where
    it does not; the angle's in-plane-axis bias shows on the combined
    rotations (up to 0.267 deg), inside the basin. (ii) The
    projection-centre-shift fixture: ledger 210 (iii). (iii) Real
    rim, 16 points (rows 125-128 x 115-118), `(None, None)`, budget
    500, T against the COMPLETE FM row with the acceptance (the
    frozen PARTIAL row on the same points and budgets: ledger 216
    (v)): every
    T seed exactly (0.0, 0.0); `theta_hat` 0.237 to 0.977 deg (peak
    0.58-0.77) against fitted in-plane rotations of 0.06 to 0.41 deg;
    both seeds converge on all 16 to the same residual (to 4
    digits); criterion at the seed T 1.058-1.856, C 0.369-1.767; the
    acceptance kept C on 13 points and T on 3 ((125, 118), (127,
    118), (128, 118); `theta_hat` 0.237-0.275; T and C iterations
    91/91, 122/123, 134/135); iterations total 2002 with T against
    1616 with the accepted rows (-19 per cent), e.g. 136 -> 90 and
    123 -> 65 on the far rim. (iv) Real rim, 8 points (rows 127-128 x
    115-118), `(0.05, None)`, budget 200: T converged 0 of 8; C
    converged 5 of 8 (35 to 67 iterations, residuals 0.67-0.99
    against T's capped 1.61-1.85), and the acceptance kept C on
    exactly those 5 (criterion 1.150-1.235 against 1.899-1.941) and
    T on the 3 that neither row converged (`theta_hat` 0.197-0.308).
    On these points the gain comes from breaking the D5 anchor
    (ledger 207), not from large twists: under `"auto"` at 1.5 deg
    none of them would be routed pre-fit, and the default-filter
    failures would reach the FM row through the retry (D22.8).

215. **In-frame budgets and the two capture numbers (requirements
    D22.16).** The rotation about the PC at which a subregion corner
    leaves the pattern, computed exactly (`inframe.py`, `budget.py`):
    `PC_480` 5.62 deg at `border=0.05` and 23.8 deg at 0.15; the V8
    PC 6.62 deg and 38.7 deg (corner radius 311.6 px at 0.05). The
    3.84 deg of the `TestPureRotations` comment is a conservative
    bound. Beyond the in-frame angle the mirror boundary, not the
    seed, limits accuracy, and the exact seed fares the same: at
    `PC_480`, `border=0.05`, at most 0.024 px to 6 deg, then 0.16,
    0.09, 0.025, 0.27 and 0.49 px at 8, 10, 15, 20 and 30 deg (0.30
    to 1.07 px unfiltered from 8 deg); at `border=0.15` at most
    0.0044 px to 20 deg and 0.033-0.048 px at 30. So the second prototype's recipe captured every case to
    30 deg (ledger 202(ii)) and the frozen recipe every case
    measured, to 30 deg at `border=0.15`, the edge of its search
    window (ledger 216 (i)); nothing beyond 30 deg can be captured
    with that window. Accurate results extend to the in-frame angle
    of the chosen border.

216. **Spec-review re-measurement of the frozen recipe (requirements
    D22 lead, D22.3 to D22.5, D22.7, D22.15; validation V10(b), (e),
    (f), (i); drafting runs `proto/e2e_partial.py`, 91.3 s, and
    `proto/deadband.py`, plus the critics' `subbatch_amp.py` and
    `dilution.py`).** Conditions as 200, free virtual memory 3.0 GB
    at the start. Recipe: the V8 angle of D22.3, the engine's bicubic
    `evaluate` over the bounding box for the de-rotation,
    `initial_guess` on the de-rotated crop, the PARTIAL row (P), and
    the acceptance as D22.5 (`fit_pattern(max_iterations=0)`
    criterion, strictly lower keeps P); T is the D5 seed. (i) G1,
    `PC_480`, `border=0.15`, default band-pass, budget 50, V2 corner
    error to the exact homography:

    | twist (deg) | theta_hat | P seed err px | criterion T / P | kept | P: it, err px | exact seed: it, err px |
    |---|---|---|---|---|---|---|
    | +8 | +7.978 | 0.109 | 1.984 / 0.0012 | P | 2, 0.00242 | 2, 0.00242 |
    | +10 | +9.970 | 0.154 | 1.987 / 0.0015 | P | 3, 0.00408 | 2, 0.00408 |
    | +15 | +14.973 | 0.137 | 2.028 / 0.0014 | P | 2, 0.00103 | 2, 0.00102 |
    | +20 | +19.967 | 0.169 | 2.027 / 0.0017 | P | 2, 0.00226 | 2, 0.00225 |
    | +25 | +24.954 | 0.234 | 2.021 / 0.0026 | P | 3, 0.00274 | 2, 0.00274 |
    | +29 | +28.933 | 0.302 | 2.005 / 0.0061 | P | 3, 0.01740 | 2, 0.01740 |
    | +30 | +29.931 | 0.309 | 1.977 / 0.0078 | P | 3, 0.03257 | 2, 0.03257 |
    | -20 | -20.004 | 0.018 | 2.001 / 0.0011 | P | 2, 0.00581 | 2, 0.00581 |

    The translation seed converged on none; angular peak 0.95-0.99;
    the P and exact-seed errors agree within 9e-6 px. At 30 deg, the
    window's edge bin, the estimate is 29.931 deg and the fit
    converges. (ii) The projection-centre-shift fixture of 210 (iii)
    FITTED from P, budget 50: (+8, (12, -9)) 3 iterations to 0.0027
    px; (-12, (10, 7)) 2 to 0.0016; (+5, (-8, 5)) 2 to 0.0013; (+15,
    (6, -4)) 3 to 0.0011; the acceptance kept P on all four
    (criterion 0.0011-0.0026 against 1.954-2.015). (iii) The V8 PC,
    `border=0.05`, default band-pass, budget 200, seam-level proxies
    of the ramp-map points built with the prototype fixtures (the
    orientation, PC, border and Fe of `ramp_map`; not the ramp map
    run through the engine): pure twist 3.0 deg -- T CONVERGES in 14
    iterations (0.0107 px), so 3.0 deg cannot be G8's twist; 3.5 deg
    -- T not converged at 200, finite `h`, residual 1.515, 53.0 px
    off; P kept, 3 iterations, 0.0140 px; 4.0 deg -- T not converged,
    finite `h`, residual 1.368, 42.9 px off; P kept (1.983 against
    0.0022), 3 iterations, 0.0119 px; 8.0 deg (`RAMP_UNFITTABLE`) --
    T not converged, finite `h`, residual 1.938, 144.5 px off; P kept
    (1.980 against 0.0068), 3 iterations, 0.053 px (out of frame
    beyond 6.62 deg); the 9 deg reduced tilt about x
    (`RAMP_RESCUE`) -- `theta_hat` -0.101 deg, criterion T 1.293
    against P 1.279, so P is kept; T converges in 113 iterations and
    P in 116, both 0.3888 px; G6, the unrelated pair -- `theta_hat`
    +12.009 deg (peak 0.56), criterion T 2.040 against P 2.080, so T
    is kept; T not converged at 200 with a FINITE `h` (residual
    1.888); a forced P not converged either (1.656). (iv) G3, a
    512x622 synthetic at `PC_480`, crop 460x560, seed level, angle
    error in deg, physical-frequency table against the bin-unit
    mutant: 2 deg -0.003 / -0.074; 5 deg +0.017 / -0.196; 8 deg
    +0.004 / -0.344; 15 deg +0.018 / -1.030; 20 deg +0.027 / -2.293;
    -8 deg -0.015 / +0.386. The frozen radial grid (`rho_j = 0.02 +
    j / m`) against the evidence grid (`linspace`, 176 radii at
    460x560): within 0.0005 deg on G3 and on G1 at the eight
    off-grid twists of V10(b) (G1 max |error| 0.0134 deg either
    way). (v) The real rim with P and the acceptance, the points and
    budgets of 214 (iii) and (iv), T's fits read from `e2e.jsonl`
    (same code): `(None, None)`, budget 500 -- P kept on 13 of 16
    (criterion 0.455-1.403 against T's 1.058-1.856) and T on the same
    3 as with the complete row ((125, 118), (127, 118), (128, 118):
    P 1.138, 1.556, 1.751 against 1.115, 1.491, 1.690); all 16
    converge to the same residual from either row; iterations with
    the accepted rows 1660 against 2002 for T (-17 per cent; the
    complete row 1616, -19 per cent), e.g. 145 -> 99, 136 -> 94, 123
    -> 79. `(0.05, None)`, budget 200 -- T converged 0 of 8; P kept
    on 5 (criterion 1.336-1.398 against 1.899-1.924), which converge
    in 87 to 161 iterations (the complete row 35 to 67) to the
    residuals the complete row reached (0.67-0.99); T kept on 3 (P
    1.932-1.975 against 1.923-1.941), where neither row converges.
    (vi) Dead-band cross (`deadband.py`): G1 with columns and rows
    236:244 set to the reference mean in both images and the same
    cross as the D4 `dead_band`; angle error with / without the
    cross at 1, 3, 5, 8 and -3 deg: `(0.05, None)` -0.086 / +0.002,
    -0.120 / +0.013, -0.073 / +0.007, +0.084 / +0.019, +0.067 /
    +0.005; `(None, None)` -0.066 / -0.002, -0.071 / +0.022, -0.040 /
    +0.005, +0.084 / +0.019, +0.059 / +0.007; peak 0.87-0.99. The
    cross pulls the angle towards 0 below 8 deg, at most 0.12 deg.
    (vii) Device slots of the masked full-P branch
    (`subbatch_amp.py`): the gate twist map of 212 (iii), the 57772
    fitted points outside the crater in map order (the fit order of
    a single-grain map; the real grain ordering is MTP), P = 32, 1806
    sub-batches: thresholds 1.0 / 1.5 / 2.0 deg route 1012 / 336 /
    73 points in 111 / 55 / 17 sub-batches, 3552 / 1760 / 544 slots
    (3.5x / 5.2x / 7.5x the routed count). (viii) Regression dilution
    (`dilution.py`) on the 32 rim points of 212 (v): standard
    deviation of the input twist 0.221 deg, of the fitted rotation
    0.120 deg; slope with the input twist as regressor 0.452, with
    the fitted rotation as regressor 1.545 (its inverse 0.647),
    geometric mean 0.541, median ratio input / fitted (|fitted| > 0.1
    deg) 1.19.

<!-- ============ BLOCK 3: V9(b) dated note (the amended literal) ============ -->
<!--
ANCHOR: validation.md, V9, item (b) ("**The Stage D raise**"),
after its last sentence ("...with `seed_from_neighbors=True` raises
`ValueError`."). Append inside the item.
-->

Amended 2026-10-07 (Stage F, requirements D22.11 and the D21.12
amendment): the literal asserted in full is the amended one, which
names `seed_from_neighbors=False` with `fourier_mellin='auto'` as the
GPU remedy; `backend='cpu'` stays pinned as written. With
`fourier_mellin` other than `"off"`, the D22.11 `ValueError` fires
before this raise (V10(a)).
