#
# Copyright 2019-2026 the kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
#

"""Tests of the optional CuPy GPU backend of the HREBSD IC-GN engine
(``kikuchipy.indexing._hrebsd._gpu`` and ``._batched``, and the
``backend`` plumbing of ``run_hrebsd_dic`` and ``EBSD.hrebsd_dic``).

Covers ``specs/2026-09-07-hrebsd-dic/validation.md`` V9, oracles (a)
to (p); (q) is a recorded ledger measurement, never a test.
Requirements D21 govern; the CPU path is the oracle for every GPU
output (D21.8).

- **Default suite** (no GPU, runs on CI): the backend switch, its
  check order and the docstring (a); the Stage D raise (b); the
  availability gate on a FAKE cupy (c); the seed seam under numpy (d),
  (e); the batched core's numpy twin at ``"float64"`` (f), (h), (i);
  the runner through ``run_hrebsd_dic`` with a numpy session (a), (h),
  (i), (n); the laziness oracle, the launch-layout pin and the CUDA
  source pin (m); the VRAM model and batch chooser (o); the drift
  tripwire's CPU half (l); gating, canary and import hygiene (p).
- **Locally gated suite** (``cupy_gpu`` fixture, ``-n 0`` only,
  through the PINNED overlay of D21.15): everything above on the real
  device, at both device precisions where the numpy twin covers only
  ``"float64"``.  CI green is ZERO evidence for the GPU path.

Layout (D21.14.1): the dated pin constants, then the ``cupy_gpu``
fixture and the fake-cupy helpers DUPLICATED from
``test_spherical_gpu.py`` (test modules cannot import each other under
``--import-mode=importlib`` and the root ``conftest.py`` is
develop-owned), then the V9 fixtures F1 to F7 and the Ni map, then the
default-suite classes, then the gated ones, then the mutation map of
plan 11 item 4.

"Bitwise" throughout means :func:`numpy.array_equal` with
``equal_nan=True`` on floats (value identity, the Stage A convention).

**The argument lists the failing tests fix** (plan 11 item 1; D21.5
and D21.9.5 freeze the NAMES, this commit freezes the argument lists):

``kikuchipy.indexing._hrebsd._engine``:

- ``run_hrebsd_dic(..., navigation_mask=None, backend="cpu",
  chunksize=None, verbose=1, step_sizes=(1.0, 1.0),
  correct_pc_shift=True, coefficient_dtype=np.float32,
  device_precision="mixed", seed_precision="complex128")``, all
  keyword only (D21.1, D21.4, D21.5).
- ``_verify_gpu_or_raise`` imported into ``_engine``'s namespace (the
  D21.2 test seam), and the module ``_gpu`` as ``_engine._gpu``.

``kikuchipy.signals.EBSD.hrebsd_dic(..., navigation_mask=None,
backend="cpu", chunksize=None, verbose=1)`` (the dated D15.4
amendment); the precision knobs are engine only.

``kikuchipy.indexing._hrebsd._gpu``:

- ``_add_nvidia_dll_directories``, ``_CUPY_MINIMUM_VERSION``,
  ``_DEVICE_LOCK``: the ``_spherical._gpu`` objects by identity.
- ``_gate_result``: the HREBSD gate cache, ``None`` until first use.
- ``_verify_gpu_or_raise() -> None``.
- ``_GpuSession(namespace, batch_size, *, device_precision,
  seed_precision)`` with ``close()`` (idempotent) and
  ``__dask_tokenize__() -> ("kikuchipy-hrebsd-gpu-session",
  id(session))``; attributes ``xp``, ``fft``, ``kernels``,
  ``batch_size``, ``sub_batch_size``, ``device_precision``,
  ``seed_precision``, ``lock``.
- ``_make_session(namespace, batch_size, *, device_precision,
  seed_precision) -> _GpuSession``; *namespace* is the STRING
  ``"cupy"`` (what the runner passes) or ``"numpy"`` (the twin); the
  module is imported inside.  The default-suite helper
  :func:`install_numpy_session` patches it to force ``"numpy"``.
- ``_free_device_bytes(namespace) -> int``: the one device query of
  the batch choice; tests patch it to fake free VRAM.
- ``_vram_model_terms(n_pixels, device_precision, seed_precision) ->
  (g, p, r)`` bytes, and ``_vram_model_bytes(batch_size, n_pixels,
  device_precision, seed_precision) -> int`` ``= B*g + min(32, B)*p +
  R_MAX*r``; *n_pixels* is the PATTERN pixel count ``nrows * ncols``.
- ``_default_batch_size(free_bytes, n_pixels, device_precision,
  seed_precision) -> int`` (frozen by D21.9.5).
- ``R_MAX = 2`` and ``_BATCH_SIZES = (64, 32, 16, 8, 4, 2, 1)``.
- ``_run_chunks_gpu(patterns, fit_indices, state_of_point, states,
  chunksize, *, progressbar, options, device_precision,
  seed_precision, row_slots) -> (n, 12) float64``; *chunksize* ``None``
  means the default B; *options* is the engine's ``fit_options``
  dictionary; *row_slots* is ``{"width": 12, "residual": 8,
  "iterations": 9, "norm_dp": 10, "converged": 11}``.

``kikuchipy.indexing._hrebsd._batched``:

- ``SUB_BATCH_SIZE = 32``, ``SEED_PRECISIONS = ("complex128",
  "complex64")``, ``DEVICE_PRECISIONS = ("mixed", "float64")``.
- ``NumpyOutOfMemoryError(MemoryError)``, the numpy namespace's
  out-of-memory type.
- ``KernelNamespace(xp, fft, precision, out_of_memory_error, gather,
  pixel_sums, reduce_solve_update, final_criterion)``, a container;
  the entry points:
  ``gather(resident, coefficients, matrices) -> (values,
  coordinate_ok)``; ``pixel_sums(resident, values, shifts) -> sums``
  ``(B, n_sums)`` float64; ``reduce_solve_update(resident, sums,
  lockstep, options) -> None`` (in place on the
  :class:`LockstepState`; *options* carries ``min_step``,
  ``step_scale``, ``max_iterations``); ``final_criterion(resident,
  values, shifts) -> (B,)`` float64.
- ``make_kernel_namespace(namespace, device_precision) ->
  KernelNamespace``; ``("numpy", "mixed")`` raises ``ValueError``.
- ``SeedContext(xp, fft, kernels)``; ``SeedState(bounds,
  reference_spectrum, precision, upsample_factor)``;
  ``SeedBatch(targets, coefficients, pattern_index, extras=None)``
  (``extras`` defaults to a new empty dictionary).
- ``build_seed_state(ctx, state, *, precision="complex128",
  upsample_factor=16) -> SeedState``.
- ``seed_spectra(ctx, batch, seed_state)`` and
  ``seed_homographies(ctx, batch, target_spectra, seed_state)`` (D21.5,
  frozen).
- ``ReferenceResident`` (opaque) and ``build_resident(ctx, state) ->
  ReferenceResident``.
- ``LockstepState(matrices, shifts, iterations, norm_dp, active,
  converged, failed)``, a container of ``(B, ...)`` arrays.
- ``_launch_layout(n_pixels) -> (threads, blocks_per_pattern,
  pixels_per_thread)``; *n_pixels* is the SUBREGION pixel count.

**The call-time seams this commit freezes** (2026-10-06, failing-tests
gate critic finding F4/F5, validation.md ledger entry 102).  The tests
patch or spy the names below, so the implementation must look each one
up AT CALL TIME through its module global (``_batched.name(...)`` or a
bare module-level name resolved at the call, never a ``from ... import
name`` binding, a default argument or a local alias taken before the
loop), and must reach each entry point by attribute:

- ``_engine`` calls ``_verify_gpu_or_raise`` from its own namespace and
  the device runner as ``_gpu._run_chunks_gpu``; it builds states as
  ``ReferenceState`` and resolves through ``resolve_reference`` from its
  own namespace (both already so on the CPU path).
- ``_gpu`` calls ``_make_session``, ``_free_device_bytes`` and
  ``_default_batch_size`` through its own module globals, and
  ``_batched.build_resident``, ``_batched.build_seed_state``,
  ``_batched.seed_spectra`` and ``_batched.seed_homographies`` through
  the ``_batched`` module object.
- Every lockstep iteration calls ``session.kernels.gather``,
  ``session.kernels.pixel_sums``, ``session.kernels.reduce_solve_update``
  and, at retirement, ``session.kernels.final_criterion`` as ATTRIBUTES
  of the session's :class:`KernelNamespace` (a plain mutable container,
  attributes assignable), never bound to locals before the loop.  Any
  fusion lives INSIDE one entry point (plan 11 item 2's "fused pixel
  kernel" is the fused kernel behind ``pixel_sums``, or behind
  ``gather``), never across two of them: ``pixel_sums`` is called on
  ``gather``'s values every iteration.  An out-of-memory raised by any
  entry point mid-compute enters the D21.10.4 halving loop.
- Under ``"cupy"`` every CUDA kernel, the gate probe's included, is
  CONSTRUCTED through the attribute ``cupy.RawKernel`` (looked up on the
  module at construction) inside the call that uses it
  (``make_kernel_namespace`` or ``_verify_gpu_or_raise``), once per
  call: no ``cupy.RawModule``, and no Python-level cache of kernel
  objects across calls (CuPy's own compile cache below ``RawKernel`` is
  fine).  The gated launch spy subclasses ``cupy.RawKernel`` and records
  every ``__call__``.
- The gate probe uses exactly the surface :func:`make_fake_cupy`
  provides: ``cupy.__version__``, ``cupy.cuda.runtime.getDeviceCount``
  and ``.CUDARuntimeError`` (and optionally ``.deviceSynchronize``),
  ``cupy.fft.<name>(array)``, ``cupy.matmul``, ``cupy.RawKernel(code,
  name, ...)`` with a call, the array constructors ``asarray``,
  ``array``, ``ascontiguousarray``, ``ones``, ``zeros``, ``arange``,
  ``empty``, ``full`` and the dtypes ``float32``, ``float64``,
  ``complex64``, ``complex128``, ``int32``.

**Placeholders**: every device band and device-side literal, and
every literal that needs the new numpy path, is ``None`` with a
``FIXME-pin`` comment (the Phase 12 convention): :func:`assert_within`
and :func:`assert_count` fail LOUDLY on ``None``.  Only CPU-side
literals are measured at this gate: those of the (l) drift tripwire
(D21.8) and the V9(e) :data:`SEAM_DISCRIMINATING_MIN` (ledger 102).

Written failing before the implementation, at the Stage E
failing-tests gate (2026-10-06): every test that reaches the GPU
path fails with ``NotImplementedError`` from the skeleton, the gated
classes skip with that message as their reason, and the tally is
recorded in validation.md V9 "recorded results, failing-tests gate".
"""

import ast
import fractions
import functools
import gc
import importlib.metadata
import inspect
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import types
import warnings
import weakref

import dask
import dask.array as da
from dask.callbacks import Callback
import numpy as np
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd import _batched, _engine, _gpu
from kikuchipy.indexing._hrebsd._engine import (
    ReferenceState,
    fit_pattern,
    initial_guess,
    run_hrebsd_dic,
)
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    fe_to_homography,
)
from kikuchipy.indexing._hrebsd._interpolation import evaluate, spline_coefficients
from kikuchipy.indexing._hrebsd._preprocessing import (
    band_pass_transfer_function,
    hann_window,
    preprocess,
    subregion_bounds,
    subregion_mask,
)
from kikuchipy.indexing._spherical import _gpu as _spherical_gpu
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# Pattern shapes of the V9 fixtures: the V2 oracle size, the shipped
# Ni size and the real Si-indent detector shape (622 = 2 * 311, an
# awkward FFT length, which is why F3 exists)
SHAPE_480 = (480, 480)
SHAPE_60 = (60, 60)
SHAPE_RECT = (512, 622)

# The V2 projection centre of ``test_hrebsd_engine.py``, off the
# pattern centre so that a PC-centred frame stays distinguishable
PC_480 = (0.4210, 0.5794, 0.5049)

# The less off-centre projection centre of ``test_hrebsd_seeding.py``
# (F6's), whose rotated subregion stays inside the pattern
PC_SEEDING = (0.49, 0.51, 0.5049)

# The generic crystal orientation of ``test_hrebsd_engine.py``
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0

# The second grain's orientation, a large misorientation away from the
# first (``test_hrebsd_engine.py::TestOrchestration``)
ORIENTATION_B_AXIS = (3.0, -1.0, 2.0)
ORIENTATION_B_ANGLE_DEG = 52.0

# The 60 px share of the 480 px V2 design budget
SCALE_60 = SHAPE_60[0] / SHAPE_480[0]

# F1, the V2 480 px twelve-case batch ``WARP_REFIT_TOL_480`` was
# measured on: six random small homographies per seed, seeds 0 and 1,
# in that order
F1_SEEDS = (0, 1)
F1_CASES_PER_SEED = 6
F1_NAVIGATION_SHAPE = (1, 1 + len(F1_SEEDS) * F1_CASES_PER_SEED)

# F2 (gated only), the prototype's "480 seed 0" and "seed 1" sets
F2_SEEDS = (0, 1)
F2_SIZE = 64

# F3 (gated only) and its first-8 slice F3s (default suite), the
# prototype's "512x622 seed 0" set
F3_SEED = 0
F3_SIZE = 64
F3S_SIZE = 8

# F4, the Stage A 60 px route of
# ``test_hrebsd_engine.py::test_warp_refit_60px``
F4_SEED = 5
F4_SIZE = 3

# F5, the two-grain failure-contract map (see :func:`f5_map`)
F5_NAVIGATION_SHAPE = (2, 4)
F5_REFERENCES = np.array([0, 4])
F5_EASY_A = 1
F5_MASKED = 2
F5_CONSTANT = 3
F5_EASY_B = 5
F5_NON_FINITE = 6
F5_CAPPED = 7
F5_CONSTANT_VALUE = 7.0
F5_NON_FINITE_PIXEL = (30, 30)
# The production Si route (D21.6.2): no band-pass at all
F5_FILTER_CUTOFFS = (None, None)

# F6, the four-point pre-Stage-D pin map of
# ``test_hrebsd_seeding.py:342-412``: the first four ramp points,
# reference (0, 0), ``max_iterations=200``
F6_NAVIGATION_SHAPE = (1, 4)
F6_RAMP_ANGLES = (0.0, 0.8, 1.6, 2.4)
F6_MAX_ITERATIONS = 200

# F7, the synthetic many-grain map of the residency bound (D21.9.3):
# 18 grains of 2 by 2 points each, 60 px patterns, the reference of
# every grain at its block's upper left point
F7_NAVIGATION_SHAPE = (6, 12)
F7_BLOCK = (2, 2)
F7_SEED = 41

# The Ni map: the shipped ``nickel_ebsd_small``, static background
# removed, reference (1, 1), as ``test_ebsd_hrebsd_dic.py`` runs it
NI_NAVIGATION_SHAPE = (3, 3)
NI_REFERENCE = (1, 1)

# The frozen public values of D21.1, D21.12 and D21.2
SUPPORTED_BACKENDS = ("cpu", "gpu")
BACKEND_ERROR_MESSAGE = (
    "Backend {backend!r} not in the list of supported backends ['cpu', 'gpu']"
)
SEED_FROM_NEIGHBORS_GPU_MESSAGE = (
    "seed_from_neighbors=True is not supported with backend='gpu'; use "
    "backend='cpu' for neighbour-seeded propagation (a Fourier-Mellin "
    "rotation seed covering the same regime is planned)"
)
GATE_MESSAGE_PREFIX = "EBSD.hrebsd_dic with backend='gpu' requires"
# The wheel set the stage-(c) message names (D21.2)
GATE_WHEELS = (
    "nvidia-cufft-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12",
)

# The frozen structural values of D21.7.3, D21.9.3 and D21.10.3
FROZEN_SUB_BATCH_SIZE = 32
FROZEN_R_MAX = 2
FROZEN_BATCH_SIZES = (64, 32, 16, 8, 4, 2, 1)
FROZEN_FREE_VRAM_FRACTION = 0.5
FROZEN_DEVICE_PRECISIONS = ("mixed", "float64")
FROZEN_SEED_PRECISIONS = ("complex128", "complex64")

# The dask token of a session (D21.9.1)
SESSION_TOKEN_NAME = "kikuchipy-hrebsd-gpu-session"

# Machine A of D21.15, named in every device-side pin comment
MACHINE_A = (
    "machine A: i7-13700H, NVIDIA RTX 2000 Ada Generation Laptop GPU 8 GB, "
    "driver 595.71, CuPy 14.2.0 on the pinned overlay of D21.15, Windows 11"
)

# --------- CPU-side literals, MEASURED at this gate (D21.8(h)) ------- #

# MEASURED 2026-10-06 (Stage E failing-tests gate) [D21.8(h), V9(l)],
# the CPU HALF of the drift tripwire, on machine A's CPU (worktree
# .venv, CPython 3.13.12, numpy 2.4.6, scipy 1.17.1, numba 0.65.1).
# RECIPE: :func:`f1_map` through ``run_hrebsd_dic(backend="cpu",
# reference=(0, 0), verbose=0)`` at every other default, then per
# case the V2 recovery metric :func:`recovery_error` of the stored
# homography against the EXACT imposed one, on the corners of
# :func:`subregion_corners`; the twelve cases in F1 order (seed 0's
# six, then seed 1's).  Recorded in validation.md V9 ledger entry 100.
# Two runs are BITWISE equal, and the ``fit_pattern`` route on the same
# reference state gives the same twelve numbers to the bit.  The
# worst, 0.012439859 px (seed 1, case 0), is the 0.01244 px measurement
# ``WARP_REFIT_TOL_480`` was pinned on at the Stage A gate.
# REFRESH RULE: these change only on a DELIBERATE change to the shared
# CPU code, recorded in validation.md with its date and reason; a
# failure here is a shared-code drift until proven otherwise
CPU_DRIFT_RECOVERY_PX = np.array(
    [
        0.007831707107991194,
        0.004932800467962715,
        0.001150689518357858,
        0.0029205924994070817,
        0.0028893302430281925,
        0.008199674743814368,
        0.012439859159007909,
        0.003212421044706529,
        0.0022442932279219748,
        0.0024240622676751796,
        0.002017307940207109,
        0.009035331439900308,
    ]
)

# The band the CPU half is held to, per case, absolute, px.  FROZEN at
# this gate rather than MTP: the CPU path is deterministic (two runs
# bitwise, above), so this is a float-noise band for cross-BLAS
# last-bit variation, the ``PRE_STAGE_D_PIN_TOL`` precedent of
# ``test_hrebsd_seeding.py``.  MEASURED 2026-10-06 on the same recipe,
# the smallest demonstrated shared-code regressions move the worst
# case by: ``coefficient_dtype=np.float64`` 2.13e-9 px,
# ``upsample_factor=8`` 2.25e-7 px, ``min_step=1e-2`` 5.74e-5 px.
# PINNED at 1e-9 px, below all three
CPU_DRIFT_TRIPWIRE_PX = 1e-9

# The per-case iteration counts of the same run, MEASURED 2026-10-06
# and pinned EXACTLY beside the recovery literals (12 of 12 converged)
CPU_DRIFT_NUM_ITERATIONS = np.array([4, 5, 6, 6, 5, 6, 4, 7, 5, 6, 4, 4])

# MEASURED 2026-10-06 (Stage E failing-tests gate, critic finding F1)
# [V9(e)], the CPU-side discriminating count read by BOTH suites, keyed
# (fixture, row_type): over EVERY map point of the batch fixture (the
# reference included), the points whose CPU ``fit_pattern`` iteration
# count from the :func:`seam_table` row differs from the count from the
# ordinary translation seed (:func:`seam_cpu` with ``None``), at the
# default ``max_iterations=50``.  Same machine and environment as the
# drift literals above; a second fit of every planted row was bitwise
# equal (scratch ``measure_seam.py``).  Every planted row converged on
# the CPU at every point (13 of 13 on F1, 9 of 9 on F3s, 65 of 65 on
# F2-0, F2-1 and F3), the NaN row gives the D2.6 failure contract on
# every point, and ``max_iterations=1`` gives 1 iteration everywhere.
# Pinned AT the measured value (a deterministic CPU count, the (g)
# discipline).  Recorded in validation.md V9 ledger entry 102
SEAM_DISCRIMINATING_MIN = {
    ("F1", "exact"): 12,
    ("F1", "translate_rotate"): 9,
    ("F1", "rotate_1p5"): 12,
    ("F1", "perspective"): 13,
    ("F3s", "exact"): 8,
    ("F3s", "translate_rotate"): 6,
    ("F3s", "rotate_1p5"): 7,
    ("F3s", "perspective"): 9,
    ("F2-0", "exact"): 64,
    ("F2-0", "translate_rotate"): 39,
    ("F2-0", "rotate_1p5"): 63,
    ("F2-0", "perspective"): 62,
    ("F2-1", "exact"): 64,
    ("F2-1", "translate_rotate"): 39,
    ("F2-1", "rotate_1p5"): 65,
    ("F2-1", "perspective"): 64,
    ("F3", "exact"): 64,
    ("F3", "translate_rotate"): 47,
    ("F3", "rotate_1p5"): 61,
    ("F3", "perspective"): 62,
}

# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Every value below was a ``FIXME-pin`` placeholder (``None``) until
# the Stage E implementation gate (2026-10-07), which measured it on the frozen
# design and pins it at about 2x margin with the recipe and the
# machine ID beside it, recorded in validation.md.  The spec-gate
# numbers quoted are the SCALES a correct implementation should
# reproduce (ledger 93 to 95, a throwaway prototype), never pins.
# Bands measured under the numpy namespace are never reused for device
# asserts, and the CPU's pinned bands never stand in for either.

# --- (d) seeds ---
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(d)]: the
# numpy-namespace seeds equal to ``initial_guess`` row for row
# (bitwise, NaN equal to NaN), keyed (fixture, upsample_factor).
# RECIPE: :meth:`TestSeedSeamContract.test_rows_equal_initial_guess`,
# i.e. :func:`numpy_seed_rows` (one sub-batch of every target, the
# complex128 seed) against :func:`cpu_seed_rows` on the same host
# preprocessed targets, on machine A's CPU (worktree .venv, CPython
# 3.13.12, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0).  ALL
# EQUAL on every fixture and arm (12 of 12 on F1, 8 of 8 on F3s, 3 of
# 3 on F4, 4 of 4 on F6, 9 of 9 on the Ni map), the "all equal"
# expected; pinned AT the measured count (validation.md V9 ledger 103)
NUMPY_SEED_EQUAL_COUNT = {
    (name, upsample_factor): count
    for name, count in (("F1", 12), ("F3s", 8), ("F4", 3), ("F6", 4), ("Ni", 9))
    for upsample_factor in (16, 2, 1)
}
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(d)]: complex128
# device seeds equal to ``initial_guess`` row for row (bitwise), keyed
# (fixture, upsample_factor): ALL EQUAL on F1 (12), F2-0, F2-1 and F3
# (64 each) and F4 (3) at upsample 16, 2 and 1, the dimmed F1 arm (12)
# and the runner arm (13 rows, the reference included); complex64
# seeds differ from ``initial_guess`` on 0 rows of every fixture (the
# spec gate allowed 0 or 1 per 64).  Pinned AT the measured counts.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_SEED_EQUAL_COUNT = {
    **{
        (name, upsample_factor): count
        for name, count in (
            ("F1", 12),
            ("F2-0", 64),
            ("F2-1", 64),
            ("F3", 64),
            ("F4", 3),
        )
        for upsample_factor in (16, 2, 1)
    },
    ("F1 dimmed", 16): 12,
    ("F1 runner", 16): 13,
}
GPU_SEED_C64_DIFF_COUNT = 0

# --- (e) the seam h0 oracle ---
# Key conventions (2026-10-06, critic finding F1): every DEVICE table
# is keyed ``(fixture, device_precision, row_type)``, every numpy-twin
# table ``(fixture, row_type)``, and the CPU-side
# :data:`SEAM_DISCRIMINATING_MIN` ``(fixture, row_type)``, read by both
# suites over the SAME planted rows (:func:`seam_table`) and the SAME
# map points (the reference included)
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(e)]: on F1,
# F2-0, F2-1 and F3 at both device precisions and every finite planted
# row type, 0 iteration-count differences and 0 ``converged`` flips
# against the CPU ``fit_pattern(h0=row)`` (scalars, every key), and
# every map point converged on both (13 on F1, 65 on each 64-pattern
# set, the reference included); pinned AT the measured counts.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_SEAM_ITERATION_DIFF_COUNT = 0
GPU_SEAM_CONVERGED_FLIP_COUNT = 0
GPU_SEAM_BOTH_CONVERGED_MIN = {
    (name, device_precision, row_type): count
    for name, count in (("F1", 13), ("F2-0", 65), ("F2-1", 65), ("F3", 65))
    for device_precision in ("mixed", "float64")
    for row_type in ("exact", "translate_rotate", "rotate_1p5", "perspective")
}
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(e)]: the
# numpy twin's own budgets and minimum at float64 (never mixed with
# the device pins above), keyed (fixture, row_type) or one scalar for
# every key.  RECIPE: the default-suite test bodies run unchanged under recording
# assert helpers (scratch ``measure_numpy_pins.py``), i.e. the numpy
# twin through ``run_hrebsd_dic(backend="gpu",
# device_precision="float64")`` on a numpy session against the CPU
# oracle, on machine A's CPU (worktree .venv, CPython 3.13.12, numpy
# 2.4.6, scipy 1.17.1, numba 0.65.1).  The seam planted with
# :func:`seam_table` on F1 and F3s at every finite row type: 0
# iteration-count differences and 0 ``converged`` flips on every key,
# and every map point converged on both (13 of 13 on F1, 9 of 9 on
# F3s); pinned AT the measured counts (validation.md V9 ledger 103)
NUMPY_SEAM_ITERATION_DIFF_COUNT = 0
NUMPY_SEAM_CONVERGED_FLIP_COUNT = 0
NUMPY_SEAM_BOTH_CONVERGED_MIN = {
    (name, row_type): count
    for name, count in (("F1", 13), ("F3s", 9))
    for row_type in ("exact", "translate_rotate", "rotate_1p5", "perspective")
}

# --- (f) per-point parity bands, corner displacement in px ---
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(f)]: the
# DEVICE bands against the CPU, each the worst over EVERY gated use
# (parity on F1 to F4 and F6 at both seed precisions, the Ni map, the
# planted seam arms on F1 to F3, the V9(h) knob arms, padded F4, F5,
# F7, the DC-offset arm and the max_iterations arms), pinned at about
# 2x.  h band, mixed: worst 1.256e-6 px (the DC-offset arm without a
# band-pass: f32 values at a large offset), the rest at most 5.22e-7
# px (F3) -> 2.5e-6; float64: worst 3.19e-12 px (F2-1) -> 7e-12.  First
# step, mixed: worst 5.05e-7 px (the planted F3 "perspective" seam
# row; 1.20e-7 px on F1 parity, above the spec gate's 1.0e-7) -> 1e-6;
# float64: worst 1.58e-11 px (F3) -> 3.2e-11.  NOTE (review gate): the
# mixed first-step pin sits above the spec gate's 5.3e-7 px M3 (f32
# reductions) measurement because the seam's one-iteration arm shares
# it; M3's kill must be verified at the bug-injection pass.  Residual
# band ``atol + rtol * |cpu|``: worst relative 2.24e-6 (DC offset,
# mixed; at most 6.1e-7 elsewhere) -> rtol 4.5e-6; worst absolute on
# near-zero criteria 2.6e-14 -> atol 6e-14.  Fe: worst 1.20e-9 (F6,
# mixed; 5.4e-15 float64) -> 2.4e-9.  Kernel A/B at float64
# (scale-free relative): worst 1.469e-14 (gather; pixel_sums 8.7e-16,
# final criterion 4.9e-16, the update 1.1e-16, far coordinates at most
# 1.5e-16) -> 3e-14.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_PARITY_H_TOL_MIXED = 2.5e-6
GPU_PARITY_H_TOL_F64 = 7e-12
GPU_FIRST_STEP_TOL_MIXED = 1e-6
GPU_FIRST_STEP_TOL_F64 = 3.2e-11
GPU_PARITY_RESIDUAL_RTOL = 4.5e-6
GPU_PARITY_RESIDUAL_ATOL = 6e-14
GPU_PARITY_FE_TOL = 2.4e-9
GPU_KERNEL_AB_TOL_F64 = 3e-14
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(f)]: the
# numpy twin's OWN bands at float64 against the CPU (never reused for
# a device assert), each the worst over EVERY default-suite use: the
# F1, F3s and F4 parity and first-step tests, every V9(h) knob arm, the
# F5 and F7 map-order runs and the V9(e) seam arms.  RECIPE: the default-suite test bodies run unchanged under recording
# assert helpers (scratch ``measure_numpy_pins.py``), i.e. the numpy
# twin through ``run_hrebsd_dic(backend="gpu",
# device_precision="float64")`` on a numpy session against the CPU
# oracle, on machine A's CPU (worktree .venv, CPython 3.13.12, numpy
# 2.4.6, scipy 1.17.1, numba 0.65.1).
# Worst h band 2.344e-13 px (seam F1 "perspective"), pinned 5e-13 px;
# worst first step 1.798e-13 px (seam F3s "translate_rotate"), pinned
# 4e-13 px; residuals: every compared criterion but one sits near
# 1e-16 (synthetic exact recoveries) with absolute differences of at
# most 2.498e-16 (F7), so the absolute floor is pinned 5e-16, and the
# one large criterion (F5's capped point, 1.777) differs by 2.5e-16
# relative, so the relative band is pinned 5e-16; worst Fe difference
# 5.551e-16 (the low-pass arm), pinned 1.2e-15 (validation.md V9
# ledger 103)
NUMPY_PARITY_H_TOL_F64 = 5e-13
NUMPY_FIRST_STEP_TOL_F64 = 4e-13
NUMPY_PARITY_RESIDUAL_RTOL = 5e-16
NUMPY_PARITY_RESIDUAL_ATOL = 5e-16
NUMPY_PARITY_FE_TOL = 1.2e-15

# --- (g) iteration and convergence COUNT budgets, per fixture ---
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(g)]: 0
# iteration-count differences and 0 ``converged`` flips on every key
# (F1 to F4 and F6 at both device and seed precisions, the Ni map, the
# F4 knob arms, padded F4, F5, F7, DC offset); pinned AT the measured
# counts, one scalar for every key.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_ITERATION_DIFF_COUNT = 0
GPU_CONVERGED_FLIP_COUNT = 0
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(f)/(g)]: the
# numpy twin's budgets at float64, one scalar for every key (F1, F3s,
# F4, every F4 knob arm, F5 and F7): 0 iteration-count differences and
# 0 ``converged`` flips on every key, by the recipe of the bands above;
# pinned AT the measured counts (validation.md V9 ledger 103)
NUMPY_ITERATION_DIFF_COUNT = 0
NUMPY_CONVERGED_FLIP_COUNT = 0

# --- (j) intensity scale ---
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(j)]: worst
# 2.147e-8 px mixed and 3.058e-9 px float64 (spec gate 3.2e-8 and
# 3.6e-9), pinned at about 2x.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_INTENSITY_SCALE_GENERIC_TOL_MIXED = 4.3e-8
GPU_INTENSITY_SCALE_GENERIC_TOL_F64 = 6.1e-9

# --- (k) update-rule analogue ---
# [V9(k)] the spec-gate scale: 1.4e-13 to 1.6e-11 px float64 and
# 9.2e-7 px mixed (so about 2e-6), 40x under the 8.2e-5 px tightest
# mutant separation (ledger 93)
# MEASURED 2026-10-07 (Stage E implementation gate): worst 1.025e-13
# px float64 (0 after one iteration) and 9.06e-8 px mixed, pinned at
# about 2x, still about 450x under the 8.2e-5 px mutant separation.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_UPDATE_RULE_TOL_F64 = 2e-13
GPU_UPDATE_RULE_TOL_MIXED = 1.8e-7

# --- (l) the drift tripwire's device half ---
# [V9(l)] the device recovery literals on F1 per device
# precision, the device half of :data:`CPU_DRIFT_RECOVERY_PX`; spec
# gate: the mixed recovery maximum moved by 3.4e-7 px from the CPU's
# on the prototype's 64-pattern 480 set
# MEASURED 2026-10-07 (Stage E implementation gate): the recovery
# metric of the CPU half per F1 case, from the device run at each
# device precision (default B, complex128 seeds); the worst case moves
# from the CPU's 0.012439859 px by 5.7e-8 px (mixed) and 2.8e-14 px
# (float64).  The device is bitwise deterministic run to run at a
# fixed B and B invariant (V9(m)), so the band is a float-noise band
# FROZEN at the CPU half's 1e-9 px, below the smallest demonstrated
# shared-code regression (2.13e-9 px).
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_DRIFT_RECOVERY_PX_MIXED = np.array(
    [
        0.007831716823155942,
        0.004932840998911446,
        0.0011508283167937412,
        0.002920500320123324,
        0.002889366834162639,
        0.00819946258668157,
        0.012439915755084598,
        0.0032123659913164465,
        0.0022441218966642845,
        0.002424294268855096,
        0.002017255717508676,
        0.009035342830673431,
    ]
)
GPU_DRIFT_RECOVERY_PX_F64 = np.array(
    [
        0.007831707108041115,
        0.004932800467882851,
        0.001150689518357858,
        0.002920592499459284,
        0.0028893302430839257,
        0.008199674743774109,
        0.012439859159036198,
        0.0032124210447276235,
        0.002244293227945333,
        0.0024240622677309683,
        0.002017307940207109,
        0.009035331439850204,
    ]
)
GPU_DRIFT_TRIPWIRE_PX = 1e-9

# --- (m) determinism ---
# [V9(m)] B invariance at B in {8, 32, 40, default},
# possibly 0 (expected bitwise among the B >= 32 by construction)
# MEASURED 2026-10-07 (Stage E implementation gate): B = 32, 40 and
# the default B (64 here) equal B = 8 BITWISE on F1 at both device
# precisions (and on F2-0, F2-1, F3 and F4, ledger 104 E4); pinned 0.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_BATCH_INVARIANCE_TOL = 0.0

# --- (n) robustness ---
# [V9(n)] the pool residue in bytes a recovered run may
# leave under a real ``set_limit``
# MEASURED 2026-10-07 (Stage E implementation gate): 0 bytes on F1
# and F7 (built at B = 64, 32, 16, 8, 4 under the limit), once the
# runner empties the worker threads' cuFFT plan caches; pinned 0.
# RECIPE: the gated suite (``--weekly``, ``-n 0``, the pinned overlay,
# ``KIKUCHIPY_EXPECT_GPU=1``) run once with recording assert helpers
# (scratch ``recpins.py``, the test bodies unchanged), on machine A
# (see MACHINE_A; ``nvidia-smi`` idle before the run).  Recorded in
# validation.md V9 ledger 104
GPU_LEAK_RESIDUE_BYTES = 0

# --- (o) VRAM calibration, bytes, at 512x622 (pattern pixels) ---
# MEASURED 2026-10-07 (Stage E implementation gate) [V9(o)], g, p and r
# each calibrated SEPARATELY against cupy pool high-water marks on F3s
# (512x622), on machine A (see MACHINE_A).  RECIPE:
# :meth:`TestGatedVramCalibration.test_resident_term` (r: the high-water
# mark of ``build_resident`` plus ``build_seed_state``) and
# ``::test_per_slot_and_transient_terms`` (g = (peak(64) - peak(32)) /
# 32, p = (peak(16) - peak(8)) / 8 - g), deterministic run to run once
# the runner empties the worker threads' cuFFT plan caches (scratch
# ``vram_calib.py``).  Measured: g 1,273,936 B (the f32 coefficient
# plane, both device precisions); p 21,094,832 B; r 23,159,296 B
# (mixed) and 26,248,704 B (float64) at complex128.  The model's own
# terms (the default suite reads the SAME bounds): g 2,613,248 B mixed
# and 3,887,104 B float64, p 30,572,544 B, r 30,638,080 B mixed and
# 38,281,216 B float64.  Pinned at about 2x around the measured marks,
# low = measured / 2 and high = 2 x measured, except g's high, which is
# 2 x the float64 model term the default suite checks (the model counts
# the gather's value plane per slot, which the seed-stage peak does not
# reach).  Recorded in validation.md V9 ledger 104
VRAM_G_BOUNDS_RECT = (600_000, 8_000_000)
VRAM_P_BOUNDS_RECT = (10_500_000, 42_000_000)
VRAM_R_BOUNDS_RECT_MIXED = (11_500_000, 46_000_000)
VRAM_R_BOUNDS_RECT_F64 = (13_000_000, 52_000_000)

# --- the default-suite wall time (D21.14.3), recorded not asserted ---
# Recorded in validation.md V9 ledger 101 (failing-tests gate) and
# re-recorded at the implementation gate (ledger 104)


# --------------------- The gated-suite fixture ---------------------- #


def _cupy_gpu_skip_reason() -> "str | None":
    """Return why the gated GPU suite must skip, or ``None`` to run.

    The decision order is frozen (D21.14.2) and the first two checks
    come BEFORE the availability probe, so neither the kill switch
    nor an xdist worker ever initialises CUDA:

    1. ``KIKUCHIPY_NO_GPU_TESTS`` set (to any non-empty value): the
       manual kill switch, e.g. while the HROSM session owns the GPU.
    2. ``PYTEST_XDIST_WORKER`` set: the structural ``-n 0`` rule.
    3. The HREBSD availability gate ``_gpu._verify_gpu_or_raise``: its
       actionable message becomes the skip reason.
    """
    if os.environ.get("KIKUCHIPY_NO_GPU_TESTS"):
        return (
            "KIKUCHIPY_NO_GPU_TESTS is set, the manual GPU-test kill "
            "switch; unset it to run the locally gated GPU tests"
        )
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        return (
            "GPU tests run only at -n 0 (PYTEST_XDIST_WORKER is set, "
            "the structural xdist skip); run 'uv run pytest "
            "tests/test_indexing/test_hrebsd_gpu.py -n 0' through the "
            "pinned overlay to exercise them"
        )
    try:
        _gpu._verify_gpu_or_raise()
    except Exception as error:
        # the gate's own message is the instruction-bearing skip reason
        return str(error)
    return None


@pytest.fixture
def cupy_gpu():
    """Yield the imported :mod:`cupy` when the HREBSD GPU backend is
    usable here, else skip with an instruction-bearing reason
    (D21.14.2).  An availability probe, not an env-var opt-in."""
    reason = _cupy_gpu_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    import cupy

    return cupy


# --------------------------- Fake cupy ------------------------------ #


def make_fake_cupy(
    version="14.2.0",
    device_count=1,
    device_error=None,
    fft_error=None,
    matmul_error=None,
    kernel_error=None,
    calls=None,
):
    """Return ``(fake, calls)``: a fake :mod:`cupy` stand-in for the
    HREBSD gate's three stages, with plantable failures, recording
    every probe call in order in the list *calls*.

    The recorded entries: ``("getDeviceCount",)``; ``("fft", name,
    dtype)`` for every ``fake.fft.<name>(array)`` with the input's
    dtype name (``"complex128"`` or ``"complex64"``, D21.2 stage (c));
    ``("matmul",)``; ``("RawKernel", name)`` at construction (the
    NVRTC compile) and ``("launch", name)`` at a call.  So an FFT-only
    or single-precision probe is visible (mutant M22).  Failures:
    *device_error* raises ``fake.cuda.runtime.CUDARuntimeError`` from
    ``getDeviceCount``; *fft_error*, *matmul_error* and *kernel_error*
    are exception INSTANCES raised by the matching call.
    """
    calls = [] if calls is None else calls

    class FakeCudaRuntimeError(Exception):
        pass

    def get_device_count():
        calls.append(("getDeviceCount",))
        if device_error is not None:
            raise FakeCudaRuntimeError(device_error)
        return device_count

    class _FftNamespace:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                array = np.asarray(args[0]) if args else np.zeros(1)
                calls.append(("fft", name, array.dtype.name))
                if fft_error is not None:
                    raise fft_error
                return array

            return call

    def fake_matmul(*args, **kwargs):
        calls.append(("matmul",))
        if matmul_error is not None:
            raise matmul_error
        return np.matmul(*args, **kwargs)

    class FakeRawKernel:
        def __init__(self, code, name, *args, **kwargs):
            self.code = code
            self.name = name
            calls.append(("RawKernel", name))
            if kernel_error is not None:
                raise kernel_error

        def __call__(self, *args, **kwargs):
            calls.append(("launch", self.name))
            if kernel_error is not None:
                raise kernel_error

        def compile(self, *args, **kwargs):
            calls.append(("compile", self.name))

    fake = types.SimpleNamespace()
    fake.__version__ = version
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            getDeviceCount=get_device_count,
            CUDARuntimeError=FakeCudaRuntimeError,
            # a no-op the stage-(c) probe may call after the launch;
            # deliberately NOT recorded, so the call-order pins of
            # (c) do not depend on whether the gate synchronises
            deviceSynchronize=lambda: None,
        )
    )
    fake.fft = _FftNamespace()
    fake.matmul = fake_matmul
    fake.RawKernel = FakeRawKernel
    fake.ndarray = np.ndarray
    for name in ("asarray", "array", "ascontiguousarray"):
        setattr(fake, name, np.asarray)
    for name in ("ones", "zeros", "arange", "empty", "full"):
        setattr(fake, name, getattr(np, name))
    for name in ("float32", "float64", "complex64", "complex128", "int32"):
        setattr(fake, name, getattr(np, name))
    return fake, calls


def install_fake_cupy(monkeypatch, fake, *, shim_calls=None):
    """Install *fake* as ``sys.modules["cupy"]`` for one test and,
    when *shim_calls* is a list, replace ``_gpu._add_nvidia_dll_
    directories`` by a recorder appending ``("shim",)`` to it, so the
    stage order (a), (b), shim, (c) reads from one list (mutant M21).
    Pass the SAME list as ``make_fake_cupy(calls=...)`` for that."""
    monkeypatch.setitem(sys.modules, "cupy", fake)
    if shim_calls is not None:
        monkeypatch.setattr(
            _gpu, "_add_nvidia_dll_directories", lambda: shim_calls.append(("shim",))
        )


def hide_cupy(monkeypatch):
    """Make ``import cupy`` raise ``ImportError`` for one test, the
    stage-(a) failure on a machine that HAS cupy."""
    monkeypatch.setitem(sys.modules, "cupy", None)


@pytest.fixture
def fresh_gate(monkeypatch):
    """Reset the HREBSD gate's cached verdict for one test."""
    monkeypatch.setattr(_gpu, "_gate_result", None)


# ------------------------ Assertion helpers ------------------------- #


def assert_within(measured, bound, name: str) -> None:
    """Assert ``measured <= bound``, failing loudly and informatively
    while *bound* is an unfilled ``FIXME-pin`` placeholder."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V9, "
            f"measured-then-pinned); measured {measured!r}. Fill it with a dated "
            "value, the recipe and the machine ID, and record it in validation.md"
        )
    assert measured <= bound, f"{name}: {measured} > {bound}"


def assert_count(measured: int, pinned, name: str) -> None:
    """Assert a COUNT equals its pin (the (g) discipline: counts are
    pinned at the measured value, never a fraction), failing loudly
    while *pinned* is an unfilled ``FIXME-pin`` placeholder."""
    if pinned is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V9); "
            f"measured count {measured!r}. Pin it with a dated comment"
        )
    assert measured == pinned, f"{name}: {measured} != {pinned}"


def assert_properties_bitwise(left: dict, right: dict) -> None:
    """Assert two engine property dictionaries are bitwise equal: the
    same keys, and every array equal by value (NaN equal to NaN on
    floats), with equal dtypes and shapes."""
    assert set(left) == set(right)
    for name in left:
        a = np.asarray(left[name])
        b = np.asarray(right[name])
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        if np.issubdtype(a.dtype, np.floating):
            assert np.array_equal(a, b, equal_nan=True), name
        else:
            assert np.array_equal(a, b), name


# ----------------- Oracle algebra (duplicated, V2) ------------------ #


def matrix_of(h):
    """Return the ``(3, 3)`` shape function of the eight parameters,
    built here rather than taken from the code under test."""
    h = np.asarray(h, dtype=np.float64)
    return np.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )


def parameters_of(matrix):
    """Return the eight parameters of a ``(3, 3)`` shape function,
    renormalised by ``W[2, 2]``."""
    matrix = np.asarray(matrix, dtype=np.float64) / matrix[2, 2]
    return np.array(
        [
            matrix[0, 0] - 1.0,
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 0],
            matrix[1, 1] - 1.0,
            matrix[1, 2],
            matrix[2, 0],
            matrix[2, 1],
        ]
    )


def project_with(matrix, x, y):
    """Return the projective image of ``(x, y)``, divide included."""
    s = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / s,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / s,
    )


def matrix_corner_norm(matrix, corners) -> float:
    """Return the D2.5 norm of a ``(3, 3)`` warp over *corners*, px."""
    x, y = corners[:, 0], corners[:, 1]
    warped_x, warped_y = project_with(matrix, x, y)
    return float(np.hypot(warped_x - x, warped_y - y).max())


def warp_norm(h, corners) -> float:
    """Return the D2.5 corner norm of ``W(h)``, px."""
    return matrix_corner_norm(matrix_of(h), corners)


def recovery_error(h_fit, h_true, corners) -> float:
    """Return the V2 recovery metric, the corner norm of the error warp
    ``W(h_true)**-1 . W(h_fit)``, px.  Also the GPU-against-CPU
    homography metric of D21.8(c) with *h_true* the CPU's ``h``."""
    matrix = np.linalg.inv(matrix_of(h_true)) @ matrix_of(h_fit)
    return matrix_corner_norm(matrix, corners)


def pc_pixels_of(detector, index=0):
    """Return ``(PCx_px, PCy_px, DD_px)`` of one projection centre."""
    nrows, ncols = detector.shape
    pcx, pcy, pcz = np.atleast_2d(detector.pc_flattened)[index]
    return np.array([pcx * ncols, pcy * nrows, pcz * nrows])


def subregion_corners(shape, pc_px, border=0.05):
    """Return the ``(4, 2)`` PC-centred corners of the subregion."""
    nrows, ncols = shape
    margin_row = int(round(border * nrows))
    margin_col = int(round(border * ncols))
    x0 = margin_col + 0.5 - pc_px[0]
    x1 = ncols - 1 - margin_col + 0.5 - pc_px[0]
    y0 = margin_row + 0.5 - pc_px[1]
    y1 = nrows - 1 - margin_row + 0.5 - pc_px[1]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


# --------------------- Pattern construction ------------------------- #


def make_detector(shape=SHAPE_480, pc=PC_480, binning=1, navigation_shape=None):
    """Return an oracle detector with one Bruker projection centre, or
    with the SAME one tiled over *navigation_shape* (so the beam-scan
    phantom is exactly the identity and nothing here measures it)."""
    pc = np.asarray(pc, dtype=np.float64)
    if navigation_shape is None:
        pc = pc[None, :]
    else:
        pc = np.tile(pc, (*navigation_shape, 1))
    return kp.detectors.EBSDDetector(
        shape=shape,
        binning=binning,
        px_size=70.0,
        pc=pc,
        sample_tilt=70.0,
        tilt=0.0,
    )


def generic_rotation():
    """Return the generic crystal orientation of grain A."""
    return Rotation.from_axes_angles(
        ORIENTATION_AXIS, np.deg2rad(ORIENTATION_ANGLE_DEG)
    )


def rotation_b():
    """Return the crystal orientation of grain B."""
    return Rotation.from_axes_angles(
        ORIENTATION_B_AXIS, np.deg2rad(ORIENTATION_B_ANGLE_DEG)
    )


@functools.lru_cache(maxsize=1)
def ni_master_arrays():
    """Return the shipped Ni Lambert master as
    ``(upper, lower, npx, npy, scale)`` in 64-bit floats."""
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    upper, lower = master._get_master_pattern_arrays_from_energy()
    npx, npy = master.axes_manager.signal_shape
    return (
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        int(npx),
        int(npy),
        float((npx - 1) / 2),
    )


def _bilinear(master, nii, nij, niip, nijp, di, dj, dim, djm):
    """Vectorized twin of ``_get_pixel_from_master_pattern``."""
    return (
        master[nii, nij] * dim * djm
        + master[niip, nij] * di * djm
        + master[nii, nijp] * dim * dj
        + master[niip, nijp] * di * dj
    )


def project_pattern(detector, rotation, deformation=None):
    """Return one pattern projected from the Ni Lambert master, with an
    optional imposed CRYSTAL-frame deformation gradient (the V3
    mechanism); the first projection centre of *detector* is used."""
    upper, lower, npx, npy, scale = ni_master_arrays()
    cosines = _get_direction_cosines_from_detector(detector)
    if cosines.ndim == 3:
        cosines = cosines[0]
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    vectors = rotate_vector(quaternion, np.ascontiguousarray(cosines))
    if deformation is not None:
        vectors = vectors @ np.linalg.inv(deformation).T
        vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, None]
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    parameters = _get_lambert_interpolation_parameters(vectors, npx, npy, scale)
    north = _bilinear(upper, *parameters)
    south = _bilinear(lower, *parameters)
    pattern = np.where(vectors[:, 2] >= 0, north, south)
    return pattern.reshape(detector.shape)


def sample_to_crystal_matrix(rotation):
    """Return ``R`` with ``v_crystal = R @ v_sample``."""
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    return rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T


def crystal_to_detector_matrix(detector, rotation):
    """Return ``R`` with ``v_detector = R @ v_crystal`` in the spec's
    y-DOWN detector frame (the D7 chain)."""
    matrix = detector.sample_to_detector.to_matrix().squeeze()
    return (np.diag([1.0, -1.0, 1.0]) @ matrix) @ sample_to_crystal_matrix(rotation).T


def in_plane_fe(angle_deg):
    """Return the detector-frame tensor of an in-plane rotation."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def deformed_pattern(detector, rotation, fe_detector):
    """Return ``(pattern, h_exact)`` of one imposed reduced
    detector-frame deformation gradient (the V3 mechanism, EXACT)."""
    reduced = np.asarray(fe_detector, dtype=np.float64)
    reduced = reduced / reduced[2, 2]
    chain = crystal_to_detector_matrix(detector, rotation)
    pattern = project_pattern(detector, rotation, chain.T @ reduced @ chain)
    pc_px = pc_pixels_of(detector)
    return pattern, fe_to_homography(reduced, np.zeros(2), pc_px[2])


def warp_with_skimage(image, h, pc_px):
    """Return *image* warped by ``W(h)`` with the INDEPENDENT
    ``skimage.transform`` warper (a test-oracle-only import, D18),
    the V2 recipe of ``test_hrebsd_engine.py``."""
    from skimage.transform import ProjectiveTransform, warp

    image = np.array(image, dtype=np.float64)
    translation = np.eye(3)
    translation[0, 2] = pc_px[0] - 0.5
    translation[1, 2] = pc_px[1] - 0.5
    array_matrix = translation @ matrix_of(h) @ np.linalg.inv(translation)
    transform = ProjectiveTransform(matrix=np.linalg.inv(array_matrix))
    return warp(image, transform, order=3, mode="reflect", preserve_range=True)


def random_small_homographies(n=8, dd=240.0, seed=0, scale=1.0):
    """Return random small homographies of the V2 design budget
    (translations to 5 px, rotations to 1 degree, strains to 2e-3),
    the exact recipe of ``test_hrebsd_engine.py``."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.deg2rad(1.0), np.deg2rad(1.0), size=n) * scale
    h = np.zeros((n, N_HOMOGRAPHY_PARAMETERS))
    h[:, 0] = rng.uniform(-2e-3, 2e-3, size=n)
    h[:, 4] = rng.uniform(-2e-3, 2e-3, size=n)
    h[:, 1] = -np.sin(theta) + rng.uniform(-1e-3, 1e-3, size=n) * scale
    h[:, 3] = np.sin(theta) + rng.uniform(-1e-3, 1e-3, size=n) * scale
    h[:, 2] = rng.uniform(-5.0, 5.0, size=n) * scale
    h[:, 5] = rng.uniform(-5.0, 5.0, size=n) * scale
    h[:, 6] = rng.uniform(-np.deg2rad(0.5), np.deg2rad(0.5), size=n) * scale / dd
    h[:, 7] = rng.uniform(-np.deg2rad(0.5), np.deg2rad(0.5), size=n) * scale / dd
    return h


def _read_only(array):
    """Return *array* as a contiguous read-only array."""
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


def _warped_batch(shape, pc, binning, rotation, seeds, n_per_seed, scale=1.0):
    """Return ``(reference, targets, exact, detector)``: a projected
    reference and its skimage-warped targets, ``n_per_seed`` random
    small homographies per seed, in seed order."""
    detector = make_detector(shape=shape, pc=pc, binning=binning)
    pc_px = pc_pixels_of(detector)
    reference = project_pattern(detector, rotation)
    exact = np.concatenate(
        [
            random_small_homographies(n=n_per_seed, dd=pc_px[2], seed=s, scale=scale)
            for s in seeds
        ]
    )
    targets = np.stack([warp_with_skimage(reference, h, pc_px) for h in exact])
    return reference, targets, exact, detector


def _batch_fixture(reference, targets, exact, detector, shape):
    """Return the uniform dictionary of a single-grain batch fixture:
    ``reference`` ``(nrows, ncols)``, ``targets`` ``(N, nrows, ncols)``,
    ``exact`` ``(N, 8)`` imposed homographies, ``detector`` (one PC),
    ``pc_px`` and ``corners``; plus the map form ``patterns``
    ``(1 + N, nrows, ncols)`` (the reference first), its
    ``navigation_shape`` ``(1, 1 + N)``, a per-point ``map_detector``
    with the same PC tiled, and ``reference_index`` ``(0, 0)``.
    Every array is read-only."""
    pc_px = pc_pixels_of(detector)
    n = int(targets.shape[0])
    navigation_shape = (1, 1 + n)
    pc = np.atleast_2d(detector.pc_flattened)[0]
    return {
        "reference": _read_only(reference),
        "targets": _read_only(targets),
        "exact": _read_only(exact),
        "detector": detector,
        "pc_px": _read_only(pc_px),
        "corners": _read_only(subregion_corners(shape, pc_px)),
        "patterns": _read_only(np.concatenate([reference[None], targets])),
        "navigation_shape": navigation_shape,
        "map_detector": make_detector(
            shape=shape,
            pc=pc,
            binning=int(detector.binning),
            navigation_shape=navigation_shape,
        ),
        "reference_index": (0, 0),
    }


# ----------------------- The V9 fixtures ---------------------------- #


@functools.lru_cache(maxsize=1)
def f1_batch() -> dict:
    """Return F1, the V2 480 px twelve-case batch (seeds 0 and 1, six
    cases each, the batch ``WARP_REFIT_TOL_480`` was measured on), as
    the dictionary of :func:`_batch_fixture` at ``SHAPE_480`` and
    ``PC_480``."""
    reference, targets, exact, detector = _warped_batch(
        SHAPE_480, PC_480, 1, generic_rotation(), F1_SEEDS, F1_CASES_PER_SEED
    )
    return _batch_fixture(reference, targets, exact, detector, SHAPE_480)


@functools.lru_cache(maxsize=2)
def f2_batch(seed: int) -> dict:
    """Return F2 for one seed (GATED ONLY): 64 random small
    homographies at 480 px, the prototype's "480 seed 0/1" sets."""
    reference, targets, exact, detector = _warped_batch(
        SHAPE_480, PC_480, 1, generic_rotation(), (seed,), F2_SIZE
    )
    return _batch_fixture(reference, targets, exact, detector, SHAPE_480)


@functools.lru_cache(maxsize=1)
def f3_batch() -> dict:
    """Return F3 (GATED ONLY): 64 random small homographies at the
    real detector's 512 by 622 shape, seed 0."""
    reference, targets, exact, detector = _warped_batch(
        SHAPE_RECT, PC_480, 1, generic_rotation(), (F3_SEED,), F3_SIZE
    )
    return _batch_fixture(reference, targets, exact, detector, SHAPE_RECT)


@functools.lru_cache(maxsize=1)
def f3s_batch() -> dict:
    """Return F3s, the first eight patterns of F3 (bitwise the same
    targets, built without F3's other 56), the default suite's
    512 by 622 slice."""
    detector = make_detector(shape=SHAPE_RECT, pc=PC_480)
    pc_px = pc_pixels_of(detector)
    reference = project_pattern(detector, generic_rotation())
    exact = random_small_homographies(n=F3_SIZE, dd=pc_px[2], seed=F3_SEED)
    exact = exact[:F3S_SIZE]
    targets = np.stack([warp_with_skimage(reference, h, pc_px) for h in exact])
    return _batch_fixture(reference, targets, exact, detector, SHAPE_RECT)


@functools.lru_cache(maxsize=1)
def f4_batch() -> dict:
    """Return F4, the Stage A 60 px route: the shipped
    ``nickel_ebsd_small`` pattern (0, 0), static background removed,
    as the reference, warped by three random small homographies SCALED
    to the pattern (seed 5), the recipe of
    ``test_hrebsd_engine.py::test_warp_refit_60px``; binning 8."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    reference = np.asarray(signal.data[0, 0], dtype=np.float64)
    detector = make_detector(shape=SHAPE_60, binning=8)
    pc_px = pc_pixels_of(detector)
    exact = random_small_homographies(
        n=F4_SIZE, dd=pc_px[2], seed=F4_SEED, scale=SCALE_60
    )
    targets = np.stack([warp_with_skimage(reference, h, pc_px) for h in exact])
    return _batch_fixture(reference, targets, exact, detector, SHAPE_60)


@functools.lru_cache(maxsize=1)
def f5_map() -> dict:
    """Return F5, the two-grain failure-contract map of V9(h) and (i),
    60 px patterns on a ``(2, 4)`` map, run at ``F5_FILTER_CUTOFFS``.

    Row 0 is grain 0 (reference A at flat 0), row 1 grain 1
    (reference B at flat 4); flat indices:

    - ``F5_EASY_A`` (1), ``F5_EASY_B`` (5): small scaled warps of
      their grain's reference, the "easy pattern" of V9(h);
    - ``F5_MASKED`` (2): masked out by ``navigation_mask``;
    - ``F5_CONSTANT`` (3): the INTEGER constant
      ``F5_CONSTANT_VALUE``, whose centred norm is exactly 0 under
      ``(None, None)``, so the CPU fails it in the SEED;
    - ``F5_NON_FINITE`` (6): an easy warp of B with one NaN pixel at
      ``F5_NON_FINITE_PIXEL``, inside the subregion;
    - ``F5_CAPPED`` (7): an in-plane rotation of B far outside the
      capture range, which exhausts the budget with a FINITE last
      iterate.

    Keys: ``patterns`` ``(8, 60, 60)`` float64, ``navigation_shape``,
    ``detector`` (per point, the same PC), ``grain_labels`` ``(2, 4)``,
    ``references`` ``F5_REFERENCES``, ``navigation_mask`` ``(2, 4)``,
    ``exact`` ``(8, 8)`` (NaN where no exact homography exists),
    ``corners``.  Every array read-only.

    MEASURED 2026-10-06 on the CPU (ledger entry 100), at the default
    ``max_iterations=50`` and at both ``(None, None)`` and the default
    ``(0.05, None)``: the two references converge in 1 iteration, the
    easy points in 4 (A) and 3 (B); the masked, constant and
    non-finite points give the D2.6 failure contract (NaN ``h`` and
    ``residual``, 0 iterations, ``converged=False``) with truthful
    ``grain_id`` and ``reference_index``; the capped point exhausts
    the 50 iterations with a finite last iterate 22.2 px from its
    imposed homography (residual 1.777).  The classes that use the map
    re-assert these as premises.
    """
    detector = make_detector(shape=SHAPE_60, binning=8)
    pc_px = pc_pixels_of(detector)
    reference_a = project_pattern(detector, generic_rotation())
    reference_b = project_pattern(detector, rotation_b())
    warps = random_small_homographies(n=3, dd=pc_px[2], seed=51, scale=SCALE_60)
    easy_a = warp_with_skimage(reference_a, warps[0], pc_px)
    masked = warp_with_skimage(reference_a, warps[1], pc_px)
    easy_b = warp_with_skimage(reference_b, warps[2], pc_px)
    non_finite = easy_b.copy()
    non_finite[F5_NON_FINITE_PIXEL] = np.nan
    capped, h_capped = deformed_pattern(detector, rotation_b(), in_plane_fe(25.0))
    constant = np.full(SHAPE_60, F5_CONSTANT_VALUE)
    patterns = np.stack(
        [reference_a, easy_a, masked, constant, reference_b, easy_b, non_finite, capped]
    )
    nan_row = np.full(N_HOMOGRAPHY_PARAMETERS, np.nan)
    zero = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    exact = np.stack(
        [zero, warps[0], warps[1], nan_row, zero, warps[2], warps[2], h_capped]
    )
    navigation_mask = np.zeros(F5_NAVIGATION_SHAPE, dtype=bool)
    navigation_mask.flat[F5_MASKED] = True
    return {
        "patterns": _read_only(patterns),
        "navigation_shape": F5_NAVIGATION_SHAPE,
        "detector": make_detector(
            shape=SHAPE_60,
            binning=8,
            navigation_shape=F5_NAVIGATION_SHAPE,
        ),
        "grain_labels": _read_only(np.array([[0, 0, 0, 0], [1, 1, 1, 1]])),
        "references": _read_only(F5_REFERENCES.copy()),
        "navigation_mask": _read_only(navigation_mask),
        "exact": _read_only(exact),
        "corners": _read_only(subregion_corners(SHAPE_60, pc_px)),
    }


@functools.lru_cache(maxsize=1)
def f6_map() -> dict:
    """Return F6, the four-point pre-Stage-D pin map of
    ``test_hrebsd_seeding.py:342-412`` (``pin_map``): the undeformed
    480 px reference at ``PC_SEEDING`` and the 0.8, 1.6 and 2.4 degree
    in-plane rotation ramp points (V3 deformed master, EXACT), run at
    reference (0, 0) and ``F6_MAX_ITERATIONS``.  Keys: ``patterns``
    ``(4, 480, 480)``, ``navigation_shape``, ``detector`` (per point),
    ``exact`` ``(4, 8)``, ``corners``, ``max_iterations``."""
    detector = make_detector(pc=PC_SEEDING)
    patterns = [project_pattern(detector, generic_rotation())]
    exact = [np.zeros(N_HOMOGRAPHY_PARAMETERS)]
    for angle in F6_RAMP_ANGLES[1:]:
        pattern, h = deformed_pattern(detector, generic_rotation(), in_plane_fe(angle))
        patterns.append(pattern)
        exact.append(h)
    return {
        "patterns": _read_only(np.stack(patterns)),
        "navigation_shape": F6_NAVIGATION_SHAPE,
        "detector": make_detector(pc=PC_SEEDING, navigation_shape=F6_NAVIGATION_SHAPE),
        "exact": _read_only(np.stack(exact)),
        "corners": _read_only(subregion_corners(SHAPE_480, pc_pixels_of(detector))),
        "max_iterations": F6_MAX_ITERATIONS,
    }


@functools.lru_cache(maxsize=1)
def f7_map() -> dict:
    """Return F7, the synthetic many-grain map of the residency bound
    (D21.9.3): ``F7_NAVIGATION_SHAPE`` 60 px points in
    ``F7_BLOCK``-sized grains (18 of 4 points), every grain a
    projection of its own random orientation (seed ``F7_SEED``) with
    its reference at the block's upper left point and three scaled
    random small warps of it.  Keys: ``patterns`` ``(72, 60, 60)``,
    ``navigation_shape``, ``detector`` (per point), ``grain_labels``
    ``(6, 12)``, ``references`` (flat index per label, label order),
    ``n_grains``."""
    rows, cols = F7_NAVIGATION_SHAPE
    block_rows, block_cols = F7_BLOCK
    detector = make_detector(shape=SHAPE_60, binning=8)
    pc_px = pc_pixels_of(detector)
    rng = np.random.default_rng(F7_SEED)
    labels = np.zeros(F7_NAVIGATION_SHAPE, dtype=np.int64)
    patterns = np.zeros((rows * cols, *SHAPE_60))
    references = []
    label = 0
    for r0 in range(0, rows, block_rows):
        for c0 in range(0, cols, block_cols):
            axis = rng.normal(size=3)
            angle = rng.uniform(10.0, 80.0)
            reference = project_pattern(
                detector, Rotation.from_axes_angles(axis, np.deg2rad(angle))
            )
            warps = random_small_homographies(
                n=block_rows * block_cols - 1,
                dd=pc_px[2],
                seed=F7_SEED + 1 + label,
                scale=SCALE_60,
            )
            members = [
                (r, c)
                for r in range(r0, r0 + block_rows)
                for c in range(c0, c0 + block_cols)
            ]
            for position, (r, c) in enumerate(members):
                labels[r, c] = label
                flat = r * cols + c
                if position == 0:
                    patterns[flat] = reference
                    references.append(flat)
                else:
                    patterns[flat] = warp_with_skimage(
                        reference, warps[position - 1], pc_px
                    )
            label += 1
    return {
        "patterns": _read_only(patterns),
        "navigation_shape": F7_NAVIGATION_SHAPE,
        "detector": make_detector(
            shape=SHAPE_60, binning=8, navigation_shape=F7_NAVIGATION_SHAPE
        ),
        "grain_labels": _read_only(labels),
        "references": _read_only(np.array(references)),
        "n_grains": label,
    }


def ni_inputs():
    """Return a FRESH ``(signal, xmap, detector)`` of the shipped Ni
    map: ``nickel_ebsd_small`` with its static background removed and
    a single-projection-centre detector, the recipe of
    ``test_ebsd_hrebsd_dic.py``.  Not cached: a signal is mutable."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average
    return signal, signal.xmap.deepcopy(), detector


def run_ni(**kwargs):
    """Return ``EBSD.hrebsd_dic`` on the Ni map at reference
    ``NI_REFERENCE`` and ``verbose=0`` (each overridable), the
    non-convergence ``UserWarning`` silenced."""
    signal, xmap, detector = ni_inputs()
    kwargs.setdefault("reference", NI_REFERENCE)
    kwargs.setdefault("verbose", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return signal.hrebsd_dic(xmap, detector, **kwargs)


# ------------------------- Engine helpers --------------------------- #


def run_engine(patterns, navigation_shape, detector, **kwargs) -> dict:
    """Return ``run_hrebsd_dic`` properties at ``verbose=0`` (keyword
    overridable), the non-convergence ``UserWarning`` silenced.  Any
    keyword passes through, ``backend`` and the precisions included."""
    kwargs.setdefault("verbose", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run_hrebsd_dic(patterns, navigation_shape, detector, **kwargs)


def run_fixture(fixture: dict, **kwargs) -> dict:
    """Return :func:`run_engine` on a fixture DICTIONARY in map form:
    a batch fixture (``patterns``, ``navigation_shape``,
    ``map_detector``, ``reference_index``) or a map fixture
    (``patterns``, ``navigation_shape``, ``detector``, and
    ``references`` with ``grain_labels`` when it has them, else
    reference (0, 0); F5's ``navigation_mask``, ``F5_FILTER_CUTOFFS``
    and F6's ``max_iterations`` are applied as defaults)."""
    if "map_detector" in fixture:
        detector = fixture["map_detector"]
        kwargs.setdefault("reference", fixture["reference_index"])
    else:
        detector = fixture["detector"]
        if "references" in fixture:
            kwargs.setdefault("reference", np.asarray(fixture["references"]))
            kwargs.setdefault("grain_labels", np.asarray(fixture["grain_labels"]))
        else:
            kwargs.setdefault("reference", (0, 0))
    if "navigation_mask" in fixture:
        kwargs.setdefault("navigation_mask", np.array(fixture["navigation_mask"]))
        kwargs.setdefault("filter_cutoffs", F5_FILTER_CUTOFFS)
    if "max_iterations" in fixture:
        kwargs.setdefault("max_iterations", fixture["max_iterations"])
    return run_engine(
        np.asarray(fixture["patterns"]), fixture["navigation_shape"], detector, **kwargs
    )


def make_state(reference, pc_px, **kwargs) -> ReferenceState:
    """Return a :class:`ReferenceState` built exactly as
    ``run_hrebsd_dic`` builds one (keywords ``border``,
    ``filter_cutoffs``, ``window``, ``dead_band`` and
    ``coefficient_dtype``, each at the engine default)."""
    border = kwargs.pop("border", 0.05)
    cutoffs = kwargs.pop("filter_cutoffs", (0.05, None))
    use_window = kwargs.pop("window", False)
    dead_band = kwargs.pop("dead_band", None)
    shape = tuple(reference.shape)
    mask = subregion_mask(shape, border=border, dead_band=dead_band)
    window = None
    if use_window:
        window = hann_window(shape, bounds=subregion_bounds(shape, border=border))
    return ReferenceState(
        reference,
        mask,
        pc_px,
        transfer_function=band_pass_transfer_function(shape, cutoffs),
        window=window,
        **kwargs,
    )


def fixture_state(fixture: dict, **kwargs) -> ReferenceState:
    """Return the :class:`ReferenceState` of a batch fixture's
    reference, keywords as :func:`make_state`."""
    return make_state(np.asarray(fixture["reference"]), fixture["pc_px"], **kwargs)


def cpu_fits(state, targets, h0=None, **options) -> list:
    """Return the CPU oracle: ``fit_pattern(state, target, h0=row,
    **options)`` per target, a list of result dictionaries.  *h0* is
    ``None`` (the phase cross-correlation seed) or ``(N, 8)`` rows,
    passed AS GIVEN (a NaN row reaches ``fit_pattern`` as a NaN
    seed, whose outcome is the D2.6 failure contract)."""
    results = []
    for i, target in enumerate(targets):
        row = None if h0 is None else np.asarray(h0[i], dtype=np.float64)
        results.append(fit_pattern(state, target, h0=row, **options))
    return results


def cpu_packed(state, targets, h0=None, **options) -> np.ndarray:
    """Return :func:`cpu_fits` packed as the ``(N, 12)`` float64 rows
    ``_fit_chunk`` packs (homography, residual, iterations, norm_dp,
    converged), the row the device runner returns."""
    rows = np.empty((len(targets), 12), dtype=np.float64)
    for i, result in enumerate(cpu_fits(state, targets, h0=h0, **options)):
        rows[i, :8] = result["h"]
        rows[i, 8] = result["residual"]
        rows[i, 9] = result["num_iterations"]
        rows[i, 10] = result["norm_dp"]
        rows[i, 11] = 1.0 if result["converged"] else 0.0
    return rows


def preprocessed_targets(state, targets):
    """Return ``(preprocessed, coefficients)``: the D4-preprocessed
    float64 targets ``(N, nrows, ncols)`` and their float32 spline
    coefficients, built with the host functions exactly as
    ``_fit_pattern`` builds them, the inputs of a numpy
    :class:`~kikuchipy.indexing._hrebsd._batched.SeedBatch`."""
    preprocessed = np.stack(
        [
            preprocess(target, transfer_function=state.transfer_function)
            for target in targets
        ]
    )
    coefficients = np.stack(
        [
            np.ascontiguousarray(
                spline_coefficients(
                    p, interpolation=state.interpolation, dtype=state.coefficient_dtype
                )
            )
            for p in preprocessed
        ]
    )
    return preprocessed, coefficients


def cpu_seed_rows(state, targets, upsample_factor=16) -> np.ndarray:
    """Return the CPU seed oracle, ``initial_guess`` on the
    preprocessed crops per target, ``(N, 8)`` float64, a NaN row where
    the CPU raises (the D2.6 outcome, D21.5)."""
    r0, r1, c0, c1 = state.bounds
    preprocessed, _ = preprocessed_targets(state, targets)
    rows = np.full((len(targets), N_HOMOGRAPHY_PARAMETERS), np.nan)
    for i, target in enumerate(preprocessed):
        try:
            rows[i] = initial_guess(
                state.reference_subregion,
                target[r0:r1, c0:c1],
                upsample_factor=upsample_factor,
            )
        except Exception:
            pass
    return rows


def numpy_seed_context(device_precision="float64") -> "_batched.SeedContext":
    """Return the numpy :class:`~kikuchipy.indexing._hrebsd._batched.
    SeedContext` at *device_precision* (the twin exists at
    ``"float64"`` only, D21.14.3)."""
    kernels = _batched.make_kernel_namespace("numpy", device_precision)
    return _batched.SeedContext(np, np.fft, kernels)


def numpy_seed_batch(state, targets, pattern_index=None) -> "_batched.SeedBatch":
    """Return a numpy :class:`~kikuchipy.indexing._hrebsd._batched.
    SeedBatch` of *targets* (no padding): preprocessed targets,
    coefficients, and *pattern_index* (default ``arange(N)``)."""
    preprocessed, coefficients = preprocessed_targets(state, targets)
    if pattern_index is None:
        pattern_index = np.arange(len(targets), dtype=np.int64)
    return _batched.SeedBatch(
        preprocessed, coefficients, np.asarray(pattern_index, dtype=np.int64)
    )


def install_numpy_session(monkeypatch, *, free_bytes=8 * 2**30):
    """Route ``backend="gpu"`` to the NUMPY twin for one test and
    return a recorder.

    Patches (D21.9.5 seams): ``_engine._verify_gpu_or_raise`` to pass;
    ``_gpu._free_device_bytes`` to return *free_bytes* (bytes; 8 GiB
    default); ``_gpu._make_session`` to call the REAL factory with
    namespace ``"numpy"`` whatever the runner asks for.  The recorder
    is a ``types.SimpleNamespace`` with ``built`` (the batch size of
    every session build attempt, in order), ``namespaces`` (what the
    runner asked for) and ``sessions`` (the built sessions).  Callers
    pass ``device_precision="float64"`` (the twin's only precision);
    see :func:`run_gpu_numpy`.
    """
    recorder = types.SimpleNamespace(built=[], namespaces=[], sessions=[])
    real_make_session = _gpu._make_session

    def make_numpy_session(namespace, batch_size, **kwargs):
        recorder.namespaces.append(namespace)
        recorder.built.append(int(batch_size))
        session = real_make_session("numpy", batch_size, **kwargs)
        recorder.sessions.append(session)
        return session

    monkeypatch.setattr(_engine, "_verify_gpu_or_raise", lambda: None)
    monkeypatch.setattr(_gpu, "_free_device_bytes", lambda namespace: int(free_bytes))
    monkeypatch.setattr(_gpu, "_make_session", make_numpy_session)
    return recorder


def run_gpu_numpy(monkeypatch, patterns, navigation_shape, detector, **kwargs):
    """Return ``(properties, recorder)`` of ``run_hrebsd_dic`` with
    ``backend="gpu"`` on the numpy twin (:func:`install_numpy_session`)
    at ``device_precision="float64"`` and ``verbose=0`` (each
    overridable)."""
    recorder = install_numpy_session(monkeypatch)
    kwargs.setdefault("backend", "gpu")
    kwargs.setdefault("device_precision", "float64")
    properties = run_engine(patterns, navigation_shape, detector, **kwargs)
    return properties, recorder


def drift_recovery_cpu() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(errors, iterations)`` of the (l) drift tripwire's CPU
    half: :func:`run_fixture` on F1 with ``backend="cpu"``, then per
    case :func:`recovery_error` against the exact imposed homography,
    in F1 order; the measuring recipe of
    :data:`CPU_DRIFT_RECOVERY_PX`."""
    fixture = f1_batch()
    properties = run_fixture(fixture, backend="cpu")
    homography = np.asarray(properties["homography"])[1:]
    errors = np.array(
        [
            recovery_error(h, h_true, fixture["corners"])
            for h, h_true in zip(homography, fixture["exact"])
        ]
    )
    iterations = np.asarray(properties["num_iterations"])[1:].astype(np.int64)
    return errors, iterations


def cuda_source_texts() -> dict:
    """Return ``{file name: text}`` of every ``_hrebsd`` source file
    that can carry CUDA C (``*.py``, ``*.cu``, ``*.cuh``), the scope
    of the V9(m) ``atomicAdd`` source pin (the CUDA source's module is
    implementation freedom, plan 11 item 4)."""
    import pathlib

    directory = pathlib.Path(_batched.__file__).parent
    texts = {}
    for pattern in ("*.py", "*.cu", "*.cuh"):
        for path in sorted(directory.glob(pattern)):
            texts[path.name] = path.read_text(encoding="utf-8")
    return texts


# ---------------- Suite helpers (default and gated) ----------------- #

# The packed-row layout ``_engine`` hands the device runner (the
# argument list fixed by this commit, module docstring)
ROW_SLOTS = {
    "width": 12,
    "residual": 8,
    "iterations": 9,
    "norm_dp": 10,
    "converged": 11,
}

# The engine's fit-option defaults, the ``options`` dictionary of
# ``_run_chunks_gpu`` when every knob is at its default
DEFAULT_FIT_OPTIONS = {
    "upsample_factor": 16,
    "max_iterations": 50,
    "min_step": 1e-3,
    "step_scale": 1.0,
}

# The DC offset of the ``filter_cutoffs=(None, None)`` arm of V9(h),
# both suites: added to every TARGET of F4, so a moment reconstructed
# against a shift the pixels did not see shows (mutant M42)
F4_DC_OFFSET = 1000.0

# The F5 fit list in grain order (the masked point 2 left out) and the
# state of every entry, the ``_run_chunks`` contract inputs
F5_FIT_INDICES = np.array([0, 1, 3, 4, 5, 6, 7])
F5_STATE_OF_POINT = np.array([0, 0, 0, 1, 1, 1, 1])

# The B = 40, P = 32 arm of V9(h) (D21.7.3): F1 repeated to 40 map
# points, and a slot in the LAST sub-batch
TAIL_BATCH_SIZE = 40
TAIL_SLOT = 35

# The size of the 60 px spy map: F4's reference and its three targets
# tiled to 40 points, so B = 40 needs two sub-batches of P = 32
SPY_MAP_SIZE = 40

# The dimmed arm of V9(d) (mutant M15).  DEVIATION, MEASURED
# 2026-10-06 at this gate: the spec's 1e-10 does NOT push an
# un-normalised cross-power spectrum below the ``100 * eps`` floor on
# F1 (max |X| 5.3e-10 at 1e-10, 5.3e-14 at 1e-12, both above 2.2e-14);
# 1e-13 does (5.3e-16), and the CPU seeds at that scale still equal
# the undimmed ones.  The premise is re-asserted in the test
DIM_SCALE = 1e-13

# The planted-row types of V9(e), ONE set for both suites (critic
# finding F1): the exact imposed homography; it composed with a 0.5
# degree rotation and a 0.5 px translation; with a 1.5 degree
# rotation; with h31 and h32 moved by +d and -d where d moves the
# subregion corners by ``SEAM_PERSPECTIVE_PX``; and the NaN row.  Each
# finite row is inside the capture range of D5
SEAM_FINITE_ROW_TYPES = ("exact", "translate_rotate", "rotate_1p5", "perspective")
SEAM_ROW_TYPES = SEAM_FINITE_ROW_TYPES + ("nan",)
# The corner displacement, px, of the "perspective" row's perturbation
SEAM_PERSPECTIVE_PX = 0.5

# The coarse bound of the F5 map-order probe (mutant M17): the two
# grains are 52 degrees apart, so a mis-ordered row is off by tens of
# pixels, while the float64 twin sits orders of magnitude below this
MAP_ORDER_COARSE_PX = 1e-3


def _pin(pin, key):
    """Return ``pin[key]`` for a per-key dictionary pin, else *pin*
    (a scalar pin, or ``None`` while it is a ``FIXME-pin``)."""
    if isinstance(pin, dict):
        return pin.get(key)
    return pin


def assert_at_least(measured, pinned, name: str) -> None:
    """Assert ``measured >= pinned`` (a pinned MINIMUM count), failing
    loudly while *pinned* is an unfilled ``FIXME-pin`` placeholder."""
    if pinned is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V9); "
            f"measured count {measured!r}. Pin the minimum with a dated comment"
        )
    assert measured >= pinned, f"{name}: {measured} < {pinned}"


def assert_residual_band(measured, expected, rtol, atol, name: str) -> None:
    """Assert ``|measured - expected| <= atol + rtol * |expected|``
    elementwise (the D21.8(d) relative band with its absolute floor),
    failing loudly while either is an unfilled ``FIXME-pin``."""
    measured = np.asarray(measured, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    worst = float(np.max(np.abs(measured - expected), initial=0.0))
    if rtol is None or atol is None:
        raise AssertionError(
            f"{name} (rtol, atol) is an unfilled FIXME-pin placeholder "
            f"(validation V9); measured worst absolute difference {worst!r}"
        )
    bound = atol + rtol * np.abs(expected)
    assert np.all(np.abs(measured - expected) <= bound), f"{name}: worst {worst}"


def in_plane_matrix(angle_deg, tx=0.0, ty=0.0):
    """Return the PC-centred ``(3, 3)`` shape function of an in-plane
    rotation by *angle_deg* about the PC plus a translation, px."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, tx], [sin, cos, ty], [0.0, 0.0, 1.0]])


@functools.lru_cache(maxsize=1)
def f4_dc_batch() -> dict:
    """Return F4 with ``F4_DC_OFFSET`` added to every TARGET (not the
    reference), the DC-offset target of the V9(h) ``(None, None)``
    arm."""
    f4 = f4_batch()
    targets = np.asarray(f4["targets"]) + F4_DC_OFFSET
    fixture = _batch_fixture(
        np.asarray(f4["reference"]),
        targets,
        np.asarray(f4["exact"]),
        f4["detector"],
        SHAPE_60,
    )
    return fixture


@functools.lru_cache(maxsize=1)
def f1_tail_batch() -> dict:
    """Return F1 repeated: its reference and its twelve targets tiled
    to ``TAIL_BATCH_SIZE`` map points (V9(h), D21.7.3)."""
    f1 = f1_batch()
    n = TAIL_BATCH_SIZE - 1
    tiles = np.arange(n) % len(f1["targets"])
    return _batch_fixture(
        np.asarray(f1["reference"]),
        np.asarray(f1["targets"])[tiles],
        np.asarray(f1["exact"])[tiles],
        f1["detector"],
        SHAPE_480,
    )


@functools.lru_cache(maxsize=1)
def f4_spy_batch() -> dict:
    """Return F4's reference and its three targets tiled to
    ``SPY_MAP_SIZE`` map points, the cheap 60 px map of the seam
    spies."""
    f4 = f4_batch()
    n = SPY_MAP_SIZE - 1
    tiles = np.arange(n) % len(f4["targets"])
    return _batch_fixture(
        np.asarray(f4["reference"]),
        np.asarray(f4["targets"])[tiles],
        np.asarray(f4["exact"])[tiles],
        f4["detector"],
        SHAPE_60,
    )


FIXTURE_BUILDERS = {
    "F1": f1_batch,
    "F2-0": functools.partial(f2_batch, 0),
    "F2-1": functools.partial(f2_batch, 1),
    "F3": f3_batch,
    "F3s": f3s_batch,
    "F4": f4_batch,
    "F4dc": f4_dc_batch,
    "F4spy": f4_spy_batch,
    "F1tail": f1_tail_batch,
    "F5": f5_map,
    "F6": f6_map,
    "F7": f7_map,
}


def fixture_of(name: str) -> dict:
    """Return the V9 fixture dictionary named *name*."""
    return FIXTURE_BUILDERS[name]()


@functools.lru_cache(maxsize=None)
def _cached_run(name: str, backend: str, items: tuple) -> dict:
    """Return one cached map run of a named fixture: the CPU oracle
    (``backend="cpu"``, shared by both suites; the CPU is deterministic,
    treat it read-only) or the numpy twin (``backend="gpu"`` on a numpy
    session at ``"float64"``).  An exception is never cached."""
    kwargs = dict(items)
    fixture = fixture_of(name)
    if backend == "cpu":
        return run_fixture(fixture, backend="cpu", **kwargs)
    with pytest.MonkeyPatch.context() as mp:
        install_numpy_session(mp)
        kwargs.setdefault("device_precision", "float64")
        return run_fixture(fixture, backend="gpu", **kwargs)


def cpu_run(name: str, **kwargs) -> dict:
    """Return the cached CPU map run of fixture *name*."""
    return _cached_run(name, "cpu", tuple(sorted(kwargs.items())))


def twin_run(name: str, **kwargs) -> dict:
    """Return the cached numpy-twin map run of fixture *name*."""
    return _cached_run(name, "gpu", tuple(sorted(kwargs.items())))


def assert_twin_parity(twin: dict, cpu: dict, corners, key, *, all_finite=False):
    """Assert the numpy twin's map run against the CPU's with the
    twin's OWN float64 bands and count pins (V9(f), (g) discipline):
    the corner-displacement ``h`` band and the residual band on points
    both converge (on every point both return a finite ``h`` with
    *all_finite*), the Fe band, the iteration and convergence COUNTS,
    and the host-only outputs bitwise."""
    assert set(twin) == set(cpu)
    for name in cpu:
        assert np.asarray(twin[name]).dtype == np.asarray(cpu[name]).dtype, name
        assert np.asarray(twin[name]).shape == np.asarray(cpu[name]).shape, name
    for name in ("grain_id", "reference_index"):
        assert np.array_equal(twin[name], cpu[name]), name
    h_t = np.asarray(twin["homography"])
    h_c = np.asarray(cpu["homography"])
    converged_t = np.asarray(twin["converged"])
    converged_c = np.asarray(cpu["converged"])
    if all_finite:
        compared = np.isfinite(h_t).all(axis=1) & np.isfinite(h_c).all(axis=1)
    else:
        compared = converged_t & converged_c
    errors = [recovery_error(h_t[i], h_c[i], corners) for i in np.flatnonzero(compared)]
    assert errors, f"[{key}] no point to compare: the arm went vacuous"
    assert_within(
        max(errors), NUMPY_PARITY_H_TOL_F64, f"NUMPY_PARITY_H_TOL_F64 [{key}]"
    )
    assert_residual_band(
        np.asarray(twin["residual"])[compared],
        np.asarray(cpu["residual"])[compared],
        NUMPY_PARITY_RESIDUAL_RTOL,
        NUMPY_PARITY_RESIDUAL_ATOL,
        f"NUMPY_PARITY_RESIDUAL [{key}]",
    )
    both = converged_t & converged_c
    if both.any():
        fe_diff = np.abs(np.asarray(twin["Fe"])[both] - np.asarray(cpu["Fe"])[both])
        assert_within(
            float(fe_diff.max()), NUMPY_PARITY_FE_TOL, f"NUMPY_PARITY_FE_TOL [{key}]"
        )
    iteration_diff = int(
        np.sum(np.asarray(twin["num_iterations"]) != np.asarray(cpu["num_iterations"]))
    )
    assert_count(
        iteration_diff,
        _pin(NUMPY_ITERATION_DIFF_COUNT, key),
        f"NUMPY_ITERATION_DIFF_COUNT [{key}]",
    )
    flips = int(np.sum(converged_t != converged_c))
    assert_count(
        flips,
        _pin(NUMPY_CONVERGED_FLIP_COUNT, key),
        f"NUMPY_CONVERGED_FLIP_COUNT [{key}]",
    )


def assert_failure_contract(properties: dict, index: int) -> None:
    """Assert the D2.6 failure contract at flat *index*: NaN ``h``,
    ``residual``, ``norm_dp`` and Fe, 0 iterations, not converged."""
    assert np.isnan(np.asarray(properties["homography"])[index]).all()
    assert np.isnan(np.asarray(properties["Fe"])[index]).all()
    assert np.isnan(np.asarray(properties["residual"])[index])
    assert np.isnan(np.asarray(properties["norm_dp"])[index])
    assert int(np.asarray(properties["num_iterations"])[index]) == 0
    assert not bool(np.asarray(properties["converged"])[index])


def run_direct(
    patterns,
    fit_indices,
    state_of_point,
    states,
    chunksize,
    *,
    device_precision="float64",
    seed_precision="complex128",
    **options,
):
    """Return ``_gpu._run_chunks_gpu`` rows as a host array (the D21.9
    contract), *options* over :data:`DEFAULT_FIT_OPTIONS`, at
    *device_precision* (``"float64"``, the numpy twin's only one, by
    default).  A default-suite caller installs the numpy session
    first."""
    merged = dict(DEFAULT_FIT_OPTIONS)
    merged.update(options)
    return np.asarray(
        _gpu._run_chunks_gpu(
            np.asarray(patterns),
            np.asarray(fit_indices, dtype=np.int64),
            np.ascontiguousarray(np.asarray(state_of_point, dtype=np.int64)),
            list(states),
            chunksize,
            progressbar=False,
            options=merged,
            device_precision=device_precision,
            seed_precision=seed_precision,
            row_slots=dict(ROW_SLOTS),
        )
    )


@functools.lru_cache(maxsize=2)
def f5_states(cutoffs=F5_FILTER_CUTOFFS) -> tuple:
    """Return the two F5 reference states built as ``run_hrebsd_dic``
    builds them (grain 0 then grain 1, the ``unique_references``
    order), at *cutoffs*."""
    f5 = f5_map()
    pc_px = pc_pixels_of(f5["detector"])
    patterns = np.asarray(f5["patterns"])
    return tuple(
        make_state(patterns[int(r)], pc_px, filter_cutoffs=cutoffs)
        for r in F5_REFERENCES
    )


def to_host(array):
    """Return *array* as a host numpy array: ``.get()`` on a cupy
    array, :func:`numpy.asarray` on anything else."""
    if type(array).__module__.split(".")[0] == "cupy":
        return array.get()
    return np.asarray(array)


def spy_seam(monkeypatch, *, keep_states=True):
    """Wrap ``_batched.seed_spectra`` and ``_batched.seed_homographies``
    by recorders through the module globals, in either namespace, and
    return the record list: one dictionary per call, in call order,
    with HOST copies of what the runner passes and receives (V9(d)).

    A ``"seed_spectra"`` record carries ``pattern_index``,
    ``spectra_id``, ``type``, ``shape``, ``dtype`` and ``precision``; a
    ``"seed_homographies"`` record carries ``pattern_index`` (host
    int64) and its ``pattern_index_dtype``, ``spectra_id``,
    ``ctx_xp``, ``kernels`` and ``ctx_kernels_id``, ``targets_type``,
    ``targets_shape``, ``targets_dtype``, ``coefficients_shape``,
    ``coefficients_dtype``, ``extras``, ``spectra_dtype``,
    ``precision``, ``reference_spectrum_dtype``, ``upsample_factor``,
    the output's ``type``, ``dtype``, ``c_contiguous`` and host ``h0``,
    and with *keep_states* host ``targets`` and ``coefficients``."""
    records = []
    original_spectra = _batched.seed_spectra
    original_rows = _batched.seed_homographies

    @functools.wraps(original_spectra)
    def spectra_spy(ctx, batch, seed_state):
        spectra = original_spectra(ctx, batch, seed_state)
        records.append(
            {
                "call": "seed_spectra",
                "pattern_index": to_host(batch.pattern_index),
                "spectra_id": id(spectra),
                "type": type(spectra),
                "shape": tuple(spectra.shape),
                "dtype": np.dtype(spectra.dtype),
                "precision": seed_state.precision,
            }
        )
        return spectra

    @functools.wraps(original_rows)
    def rows_spy(ctx, batch, target_spectra, seed_state):
        rows = original_rows(ctx, batch, target_spectra, seed_state)
        record = {
            "call": "seed_homographies",
            "pattern_index": to_host(batch.pattern_index),
            "pattern_index_dtype": np.dtype(batch.pattern_index.dtype),
            "spectra_id": id(target_spectra),
            "ctx_xp": ctx.xp,
            "kernels": ctx.kernels,
            "ctx_kernels_id": id(ctx.kernels),
            "targets_type": type(batch.targets),
            "targets_shape": tuple(batch.targets.shape),
            "targets_dtype": np.dtype(batch.targets.dtype),
            "coefficients_shape": tuple(batch.coefficients.shape),
            "coefficients_dtype": np.dtype(batch.coefficients.dtype),
            "extras": dict(batch.extras),
            "spectra_dtype": np.dtype(target_spectra.dtype),
            "precision": seed_state.precision,
            "reference_spectrum_dtype": np.dtype(seed_state.reference_spectrum.dtype),
            "upsample_factor": seed_state.upsample_factor,
            "type": type(rows),
            "dtype": np.dtype(rows.dtype),
            "c_contiguous": bool(rows.flags.c_contiguous),
            "h0": to_host(rows).copy(),
        }
        if keep_states:
            record["targets"] = to_host(batch.targets).copy()
            record["coefficients"] = to_host(batch.coefficients).copy()
        records.append(record)
        return rows

    monkeypatch.setattr(_batched, "seed_spectra", spectra_spy)
    monkeypatch.setattr(_batched, "seed_homographies", rows_spy)
    return records


def plant_seam(monkeypatch, rows_of, *, call_original=True) -> list:
    """Replace ``_batched.seed_homographies`` (the module global, either
    namespace) by a planter returning ``rows_of(pattern_index,
    original_rows)`` per call as a float64 C-contiguous ``(P, 8)`` array
    of ``ctx.xp``; *pattern_index* is the host int64 index and
    *original_rows* the real seam's host output (computed first), or
    ``None`` without *call_original*.  Returns the list of every call's
    host ``pattern_index``."""
    original = _batched.seed_homographies
    calls = []

    def planted(ctx, batch, target_spectra, seed_state):
        index = to_host(batch.pattern_index).astype(np.int64)
        calls.append(index)
        rows = None
        if call_original:
            rows = to_host(original(ctx, batch, target_spectra, seed_state))
            rows = np.array(rows, dtype=np.float64)
        out = np.asarray(rows_of(index, rows), dtype=np.float64)
        return ctx.xp.ascontiguousarray(ctx.xp.asarray(out, dtype=ctx.xp.float64))

    monkeypatch.setattr(_batched, "seed_homographies", planted)
    return calls


def plant_table(monkeypatch, table) -> list:
    """Plant the seam (:func:`plant_seam`, the real seam never called)
    with ``table[pattern_index]`` per slot, *table* ``(map_size, 8)``,
    zeros on padded slots; returns the call list."""
    table = np.asarray(table, dtype=np.float64)

    def rows_of(index, _):
        rows = np.zeros((index.size, N_HOMOGRAPHY_PARAMETERS))
        real = index >= 0
        rows[real] = table[index[real]]
        return rows

    return plant_seam(monkeypatch, rows_of, call_original=False)


def packed(properties: dict) -> np.ndarray:
    """Return engine properties packed as ``(map_size, 12)`` rows in
    the :data:`ROW_SLOTS` layout."""
    homography = np.asarray(properties["homography"])
    rows = np.empty((homography.shape[0], ROW_SLOTS["width"]), dtype=np.float64)
    rows[:, :8] = homography
    rows[:, ROW_SLOTS["residual"]] = properties["residual"]
    rows[:, ROW_SLOTS["iterations"]] = properties["num_iterations"]
    rows[:, ROW_SLOTS["norm_dp"]] = properties["norm_dp"]
    rows[:, ROW_SLOTS["converged"]] = np.asarray(
        properties["converged"], dtype=np.float64
    )
    return rows


def h_band(h_a, h_b, corners, where) -> float:
    """Return the largest V2 corner metric between two homography
    arrays over the flat indices where *where* holds, px (0 if none)."""
    h_a = np.asarray(h_a)
    h_b = np.asarray(h_b)
    return max(
        (recovery_error(h_a[i], h_b[i], corners) for i in np.flatnonzero(where)),
        default=0.0,
    )


def perspective_step(corners) -> float:
    """Return ``d`` such that ``h31 = d, h32 = -d`` alone moves the
    subregion *corners* by :data:`SEAM_PERSPECTIVE_PX`, the "small
    measured amount" of V9(e), derived from the corners themselves."""
    probe = 1e-9
    h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    h[6], h[7] = probe, -probe
    return probe * SEAM_PERSPECTIVE_PX / warp_norm(h, corners)


@functools.lru_cache(maxsize=None)
def seam_table(name: str, row_type: str) -> np.ndarray:
    """Return the ``(map_size, 8)`` planted rows of V9(e), ONE builder
    for both suites, of one row type (:data:`SEAM_ROW_TYPES`) on a
    batch fixture in map form: every map point, the reference
    included, gets its exact homography (the identity for the
    reference, the imposed one for every target) modified by the row
    type."""
    fixture = fixture_of(name)
    exact = np.concatenate(
        [np.zeros((1, N_HOMOGRAPHY_PARAMETERS)), np.asarray(fixture["exact"])]
    )
    if row_type == "exact":
        rows = exact.copy()
    elif row_type == "translate_rotate":
        extra = in_plane_matrix(0.5, tx=0.5, ty=0.0)
        rows = np.stack([parameters_of(matrix_of(h) @ extra) for h in exact])
    elif row_type == "rotate_1p5":
        extra = in_plane_matrix(1.5)
        rows = np.stack([parameters_of(matrix_of(h) @ extra) for h in exact])
    elif row_type == "perspective":
        d = perspective_step(fixture["corners"])
        rows = exact.copy()
        rows[:, 6] += d
        rows[:, 7] -= d
    elif row_type == "nan":
        rows = np.full_like(exact, np.nan)
    else:  # pragma: no cover
        raise ValueError(row_type)
    return _read_only(rows)


@functools.lru_cache(maxsize=None)
def seam_cpu(
    name: str,
    row_type: "str | None",
    max_iterations: int = DEFAULT_FIT_OPTIONS["max_iterations"],
) -> np.ndarray:
    """Return the CPU oracle of V9(e), shared by both suites:
    ``fit_pattern(state, pattern, h0=row)`` per MAP POINT, packed
    ``(map_size, 12)``; *row_type* ``None`` is the ordinary translation
    seed."""
    fixture = fixture_of(name)
    h0 = None if row_type is None else seam_table(name, row_type)
    return _read_only(
        cpu_packed(
            fixture_state(fixture),
            np.asarray(fixture["patterns"]),
            h0=h0,
            max_iterations=max_iterations,
        )
    )


@functools.lru_cache(maxsize=None)
def seam_twin(
    name: str,
    row_type: str,
    max_iterations: int = DEFAULT_FIT_OPTIONS["max_iterations"],
) -> dict:
    """Return the numpy twin's map run of batch fixture *name* with the
    seam planted with :func:`seam_table` (treat it read-only)."""
    with pytest.MonkeyPatch.context() as mp:
        install_numpy_session(mp)
        plant_table(mp, seam_table(name, row_type))
        return run_fixture(
            fixture_of(name),
            backend="gpu",
            device_precision="float64",
            max_iterations=max_iterations,
        )


def grain_fit_order(fixture: dict) -> np.ndarray:
    """Return the flat indices a multi-grain map fixture's run fits, in
    the engine's GRAIN order: the unmasked points stably sorted by the
    position of their reference in the sorted unique references (the
    ``unique_references`` order of ``run_hrebsd_dic``)."""
    labels = np.asarray(fixture["grain_labels"]).ravel()
    reference_of_point = np.asarray(fixture["references"])[labels]
    keep = np.ones(labels.size, dtype=bool)
    if "navigation_mask" in fixture:
        keep &= ~np.asarray(fixture["navigation_mask"]).ravel()
    fit = np.flatnonzero(keep)
    state = np.searchsorted(np.unique(reference_of_point[fit]), reference_of_point[fit])
    return fit[np.argsort(state, kind="stable")]


def assert_seam_fit_order(records: list, fixture: dict) -> None:
    """Assert, from :func:`spy_seam` records of one run of a
    multi-grain map fixture, that every ``SeedBatch.pattern_index``
    lists its real slots FIRST and as a CONTIGUOUS run of the
    grain-ordered fit indices (:func:`grain_fit_order`), the padded
    slots ``-1`` after them, and that the runs tile the whole fit list
    exactly once (mutant M52: ``pattern_index`` in map order).  Robust
    to the order in which the threaded scheduler runs the batches."""
    expected = grain_fit_order(fixture)
    # premise: the fixture discriminates grain order from map order
    assert not np.array_equal(expected, np.sort(expected))
    position = {int(index): k for k, index in enumerate(expected)}
    runs = []
    for record in records:
        if record["call"] != "seed_homographies":
            continue
        index = np.asarray(record["pattern_index"])
        n_real = int((index >= 0).sum())
        assert np.all(index[:n_real] >= 0) and np.all(index[n_real:] == -1), index
        if n_real == 0:
            continue
        positions = np.array([position[int(i)] for i in index[:n_real]])
        assert np.array_equal(positions, positions[0] + np.arange(n_real)), index
        runs.append((int(positions[0]), n_real))
    covered = np.concatenate([start + np.arange(n) for start, n in sorted(runs)])
    assert np.array_equal(covered, np.arange(expected.size))


@functools.lru_cache(maxsize=None)
def seed_case(name: str) -> tuple:
    """Return ``(state, targets)`` of one V9(d) seed fixture: F1, F3s
    and F4 (reference and targets), F6 (point 0 the reference, all four
    points targets) and the shipped Ni map (reference ``NI_REFERENCE``,
    all nine points targets)."""
    if name in ("F1", "F3s", "F4"):
        fixture = FIXTURE_BUILDERS[name]()
        return fixture_state(fixture), np.asarray(fixture["targets"])
    if name == "F6":
        f6 = f6_map()
        patterns = np.asarray(f6["patterns"])
        return make_state(patterns[0], pc_pixels_of(f6["detector"])), patterns
    signal, _, detector = ni_inputs()
    data = np.asarray(signal.data, dtype=np.float64).reshape(-1, *SHAPE_60)
    reference = data[NI_REFERENCE[0] * NI_NAVIGATION_SHAPE[1] + NI_REFERENCE[1]]
    return make_state(reference, pc_pixels_of(detector)), data


def numpy_seed_rows(state, targets, *, upsample_factor=16, precision="complex128"):
    """Return ``(rows, spectra, seed_state)`` of the numpy seed seam on
    *targets* against *state*, one sub-batch holding every target."""
    ctx = numpy_seed_context()
    seed_state = _batched.build_seed_state(
        ctx, state, precision=precision, upsample_factor=upsample_factor
    )
    batch = numpy_seed_batch(state, targets)
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    rows = _batched.seed_homographies(ctx, batch, spectra, seed_state)
    return rows, spectra, seed_state


def wrap_sessions(monkeypatch, wrap):
    """Wrap the CURRENT ``_gpu._make_session`` (normally the numpy
    session of :func:`install_numpy_session`) so that ``wrap(session)``
    runs on every session it returns."""
    inner = _gpu._make_session

    def make(namespace, batch_size, **kwargs):
        session = inner(namespace, batch_size, **kwargs)
        wrap(session)
        return session

    monkeypatch.setattr(_gpu, "_make_session", make)


def failing_sessions(monkeypatch, fails, attempts):
    """Wrap the CURRENT ``_gpu._make_session`` so that a build whose
    batch size satisfies ``fails(batch_size)`` raises the numpy
    namespace's out-of-memory error; *attempts* records every
    requested batch size in order."""
    inner = _gpu._make_session

    def make(namespace, batch_size, **kwargs):
        attempts.append(int(batch_size))
        if fails(int(batch_size)):
            raise _batched.NumpyOutOfMemoryError("planted out of memory at build")
        return inner(namespace, batch_size, **kwargs)

    monkeypatch.setattr(_gpu, "_make_session", make)


def run_f4_twin(monkeypatch, **kwargs):
    """Return ``(properties, recorder)`` of the numpy twin on F4 in map
    form (four points, reference (0, 0))."""
    f4 = f4_batch()
    kwargs.setdefault("reference", f4["reference_index"])
    return run_gpu_numpy(
        monkeypatch,
        np.asarray(f4["patterns"]),
        f4["navigation_shape"],
        f4["map_detector"],
        **kwargs,
    )


def run_f4_engine(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on F4 in map form under
    ``backend="gpu"`` at ``"float64"``; the caller installs the numpy
    session (and any wrapper) first."""
    f4 = f4_batch()
    kwargs.setdefault("reference", f4["reference_index"])
    kwargs.setdefault("backend", "gpu")
    kwargs.setdefault("device_precision", "float64")
    return run_engine(
        np.asarray(f4["patterns"]), f4["navigation_shape"], f4["map_detector"], **kwargs
    )


def subregion_pixels(shape, border=0.05) -> int:
    """Return the subregion pixel count of *shape* at *border*."""
    return int(subregion_mask(shape, border=border).sum())


def hrebsd_module_paths() -> list:
    """Return the paths of every ``_hrebsd`` Python module."""
    directory = pathlib.Path(_batched.__file__).parent
    return sorted(directory.glob("*.py"))


def module_scope_imports(tree) -> list:
    """Return ``(node, module name)`` of every import executed at
    module scope, including inside module-level ``if``, ``try`` and
    ``with`` blocks, but never inside a function or class body."""
    found = []

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                found.extend((node, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                found.append((node, module))
                for alias in node.names:
                    found.append((node, f"{module}.{alias.name}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            elif isinstance(node, ast.ClassDef):
                continue
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    children = getattr(node, field, None)
                    if isinstance(children, list):
                        visit(children)

    visit(tree.body)
    return found


# ================== DEFAULT SUITE: V9 (a) to (p) ==================== #


class TestBackendSwitch:
    """V9(a), D21.1 and D21.17: the ``backend`` keyword, the
    bitwise-unchanged default path, the frozen check order and the
    docstring.

    Mutants: M26 (a silent CPU fallback: the gate failure and the
    numpy-session run both forbid the CPU runner), M27 (``backend=
    "cpu"`` equals the call without the keyword, bitwise, on F6 and on
    the Ni map), M17 (the host-only wiring, through
    :class:`TestRunBatchesNumpySession`)."""

    @staticmethod
    def f4_call(**kwargs):
        f4 = f4_batch()
        kwargs.setdefault("reference", f4["reference_index"])
        return run_engine(
            np.asarray(f4["patterns"]),
            f4["navigation_shape"],
            f4["map_detector"],
            **kwargs,
        )

    @staticmethod
    def forbid_gate(monkeypatch):
        def forbidden():
            raise AssertionError("the gate must not run here")

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", forbidden)

    def test_cpu_equals_no_keyword_bitwise_on_f6(self):
        f6 = f6_map()
        explicit = run_fixture(f6, backend="cpu")
        implicit = run_fixture(f6)
        assert_properties_bitwise(explicit, implicit)

    def test_cpu_equals_no_keyword_bitwise_on_the_ni_map(self):
        explicit = run_ni(backend="cpu")
        implicit = run_ni()
        assert set(explicit.prop) == set(implicit.prop)
        assert_properties_bitwise(dict(explicit.prop), dict(implicit.prop))

    def test_cpu_never_touches_the_gate_the_session_or_cupy(self, monkeypatch):
        self.forbid_gate(monkeypatch)

        def forbidden(*args, **kwargs):
            raise AssertionError("no session may be built under backend='cpu'")

        monkeypatch.setattr(_gpu, "_make_session", forbidden)
        monkeypatch.setattr(_gpu, "_run_chunks_gpu", forbidden)
        hide_cupy(monkeypatch)
        properties = self.f4_call(backend="cpu")
        assert np.asarray(properties["converged"]).all()

    def test_precision_knobs_are_read_under_gpu_only(self, monkeypatch):
        # D21.1: the precision checks and the coefficient check run
        # under "gpu" only; under "cpu" these values are inert and
        # ``coefficient_dtype=np.float64`` is a valid CPU run
        self.forbid_gate(monkeypatch)
        reference = self.f4_call(coefficient_dtype=np.float64)
        properties = self.f4_call(
            backend="cpu",
            device_precision="bogus",
            seed_precision="bogus",
            coefficient_dtype=np.float64,
        )
        assert_properties_bitwise(properties, reference)

    @pytest.mark.parametrize("bad", ["GPU", "cuda", "", "auto"])
    def test_invalid_backend_raises_the_frozen_message(self, monkeypatch, bad):
        self.forbid_gate(monkeypatch)
        with pytest.raises(ValueError) as info:
            self.f4_call(backend=bad)
        assert BACKEND_ERROR_MESSAGE.format(backend=bad) in str(info.value)

    def test_invalid_backend_raises_through_the_signal_method(self, monkeypatch):
        self.forbid_gate(monkeypatch)
        with pytest.raises(ValueError) as info:
            run_ni(backend="cuda")
        assert BACKEND_ERROR_MESSAGE.format(backend="cuda") in str(info.value)

    def test_existing_argument_checks_come_first(self, monkeypatch):
        # the interpolation, shape and mask ValueErrors precede every
        # backend check, unchanged
        self.forbid_gate(monkeypatch)
        with pytest.raises(ValueError, match="interpolation"):
            self.f4_call(backend="bogus", interpolation="bilinear")
        with pytest.raises(ValueError, match="navigation mask"):
            self.f4_call(
                backend="gpu",
                navigation_mask=np.zeros(f4_batch()["navigation_shape"], dtype=int),
            )
        with pytest.raises(ValueError, match="navigation_shape"):
            f4 = f4_batch()
            run_engine(
                np.asarray(f4["patterns"]),
                (1, 2, 2),
                f4["map_detector"],
                reference=(0, 0),
                backend="gpu",
            )

    def test_backend_checks_run_in_the_frozen_order(self, monkeypatch):
        # D21.1: backend string, then (under "gpu") device_precision,
        # seed_precision, coefficient_dtype, then the D21.12 raise,
        # then the gate -- each one shown to win over every later one
        self.forbid_gate(monkeypatch)
        with pytest.raises(ValueError) as info:
            self.f4_call(
                backend="GPU", device_precision="bogus", seed_from_neighbors=True
            )
        assert BACKEND_ERROR_MESSAGE.format(backend="GPU") in str(info.value)
        with pytest.raises(ValueError, match="device_precision"):
            self.f4_call(
                backend="gpu",
                device_precision="bogus",
                seed_precision="bogus",
                coefficient_dtype=np.float64,
                seed_from_neighbors=True,
            )
        with pytest.raises(ValueError, match="seed_precision"):
            self.f4_call(
                backend="gpu",
                seed_precision="complex256",
                coefficient_dtype=np.float64,
                seed_from_neighbors=True,
            )
        with pytest.raises(ValueError, match="coefficient_dtype"):
            self.f4_call(
                backend="gpu", coefficient_dtype=np.float64, seed_from_neighbors=True
            )
        with pytest.raises(NotImplementedError) as info:
            self.f4_call(backend="gpu", seed_from_neighbors=True)
        assert str(info.value) == SEED_FROM_NEIGHBORS_GPU_MESSAGE

    @pytest.mark.parametrize("precision", FROZEN_DEVICE_PRECISIONS)
    def test_every_frozen_precision_passes_the_checks(self, monkeypatch, precision):
        # the accepted values reach the gate (here a sentinel raise)
        sentinel = RuntimeError("gate sentinel")

        def raising_gate():
            raise sentinel

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", raising_gate)
        for seed_precision in FROZEN_SEED_PRECISIONS:
            with pytest.raises(RuntimeError, match="gate sentinel"):
                self.f4_call(
                    backend="gpu",
                    device_precision=precision,
                    seed_precision=seed_precision,
                )

    def test_gate_failure_raises_with_no_cpu_fallback(self, monkeypatch):
        # M26: a failed gate raises; nothing is resolved and the CPU
        # runner never runs
        def raising_gate():
            raise RuntimeError("gate sentinel")

        def forbidden(*args, **kwargs):
            raise AssertionError("no silent CPU fallback (D21.1)")

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", raising_gate)
        monkeypatch.setattr(_engine, "resolve_reference", forbidden)
        monkeypatch.setattr(_engine, "_run_chunks", forbidden)
        monkeypatch.setattr(_engine, "_fit_chunk", forbidden)
        with pytest.raises(RuntimeError, match="gate sentinel"):
            self.f4_call(backend="gpu", device_precision="float64")

    def test_gate_precedes_reference_resolution_and_states(self, monkeypatch):
        # call-order spies under "gpu" with the gate faked to pass and
        # a numpy session: the gate, then resolve_reference, then the
        # first ReferenceState, then the device runner -- and never
        # the CPU runner (M26)
        install_numpy_session(monkeypatch)
        order = []
        original_resolve = _engine.resolve_reference
        original_state = _engine.ReferenceState
        original_runner = _gpu._run_chunks_gpu

        def gate():
            order.append("gate")

        def resolve(*args, **kwargs):
            order.append("resolve")
            return original_resolve(*args, **kwargs)

        def state(*args, **kwargs):
            order.append("state")
            return original_state(*args, **kwargs)

        def runner(*args, **kwargs):
            order.append("runner")
            return original_runner(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("the CPU runner must not run under backend='gpu'")

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", gate)
        monkeypatch.setattr(_engine, "resolve_reference", resolve)
        monkeypatch.setattr(_engine, "ReferenceState", state)
        monkeypatch.setattr(_gpu, "_run_chunks_gpu", runner)
        monkeypatch.setattr(_engine, "_run_chunks", forbidden)
        monkeypatch.setattr(_engine, "_fit_chunk", forbidden)
        properties = self.f4_call(backend="gpu", device_precision="float64")
        assert order[:3] == ["gate", "resolve", "state"], order
        assert order.count("gate") == 1
        assert "runner" in order
        assert order.index("runner") > max(
            i for i, name in enumerate(order) if name == "state"
        )
        assert np.asarray(properties["converged"]).all()

    def test_engine_signature_slots(self):
        parameters = inspect.signature(run_hrebsd_dic).parameters
        names = list(parameters)
        position = names.index("backend")
        assert names[position - 1] == "navigation_mask"
        assert names[position + 1] == "chunksize"
        coefficient = names.index("coefficient_dtype")
        assert names[coefficient + 1 : coefficient + 3] == [
            "device_precision",
            "seed_precision",
        ]
        for name, default in (
            ("backend", "cpu"),
            ("device_precision", "mixed"),
            ("seed_precision", "complex128"),
        ):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default == default

    def test_public_method_has_backend_but_no_precision_knob(self):
        parameters = inspect.signature(kp.signals.EBSD.hrebsd_dic).parameters
        names = list(parameters)
        position = names.index("backend")
        assert names[position - 1] == "navigation_mask"
        assert names[position + 1] == "chunksize"
        assert parameters["backend"].default == "cpu"
        assert "device_precision" not in parameters
        assert "seed_precision" not in parameters

    def test_public_method_forwards_backend(self, monkeypatch):
        seen = {}

        class Forwarded(Exception):
            pass

        def spy(*args, **kwargs):
            seen.update(kwargs)
            raise Forwarded

        # the signal method holds its own module-scope binding
        monkeypatch.setattr(
            sys.modules["kikuchipy.signals.ebsd"], "run_hrebsd_dic", spy
        )
        for backend in SUPPORTED_BACKENDS:
            seen.clear()
            with pytest.raises(Forwarded):
                run_ni(backend=backend)
            assert seen.get("backend") == backend
            assert "device_precision" not in seen
            assert "seed_precision" not in seen

    def test_the_docstring_documents_the_backend(self):
        # D21.17: the parameter entry, the optional-dependency
        # sentence, the three Raises entries and the Notes topics, and
        # no stage letter in public text
        docstring = kp.signals.EBSD.hrebsd_dic.__doc__
        flat = " ".join(docstring.split())
        assert "backend" in docstring
        assert (
            "Requires that :mod:`cupy` is installed, which is an optional "
            "dependency of kikuchipy"
        ) in flat
        assert "nvidia-cufft-cu12" in flat
        raises = docstring.split("Raises")[1].split("Warns")[0]
        for name in ("MemoryError", "NotImplementedError", "ValueError"):
            assert name in raises, name
        assert "backend" in raises
        assert "seed_from_neighbors" in raises
        notes = " ".join(docstring.split("Notes")[1].split())
        for topic in (
            "device precision",
            "determinis",
            "chunksize",
            "VRAM",
            "fallback",
            "seed_from_neighbors",
        ):
            assert topic in notes, topic
        assert re.search(r"\bStage [A-Z]\b", docstring) is None


class TestSeedFromNeighborsOnGpuRaises:
    """V9(b), D21.12: ``backend="gpu"`` with ``seed_from_neighbors=
    True`` raises ``NotImplementedError`` with the frozen literal,
    before the gate, reference resolution and any pattern read.

    Mutants: M25 (the raise after the gate or after reference
    resolution: the forbidden spies record the first call)."""

    @staticmethod
    def forbid_everything(monkeypatch):
        calls = []

        def forbidden(name):
            def call(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f"{name} ran before the D21.12 raise")

            return call

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", forbidden("gate"))
        monkeypatch.setattr(_engine, "resolve_reference", forbidden("resolve"))
        monkeypatch.setattr(_engine, "ReferenceState", forbidden("state"))
        monkeypatch.setattr(_engine, "_gather", forbidden("gather"))
        monkeypatch.setattr(_gpu, "_make_session", forbidden("session"))
        return calls

    def test_the_raise_carries_the_frozen_literal(self, monkeypatch):
        calls = self.forbid_everything(monkeypatch)
        with pytest.raises(NotImplementedError) as info:
            TestBackendSwitch.f4_call(backend="gpu", seed_from_neighbors=True)
        assert str(info.value) == SEED_FROM_NEIGHBORS_GPU_MESSAGE
        assert calls == []

    def test_the_raise_comes_through_the_signal_method(self, monkeypatch):
        calls = self.forbid_everything(monkeypatch)
        with pytest.raises(NotImplementedError) as info:
            run_ni(backend="gpu", seed_from_neighbors=True)
        assert str(info.value) == SEED_FROM_NEIGHBORS_GPU_MESSAGE
        assert calls == []

    def test_no_import_error_surfaces_first_without_cupy(self, monkeypatch, fresh_gate):
        # the REAL gate stays in place: on a machine without cupy the
        # D21.12 raise still comes first
        hide_cupy(monkeypatch)
        with pytest.raises(NotImplementedError) as info:
            TestBackendSwitch.f4_call(backend="gpu", seed_from_neighbors=True)
        assert str(info.value) == SEED_FROM_NEIGHBORS_GPU_MESSAGE

    def test_the_string_check_still_wins(self, monkeypatch):
        self.forbid_everything(monkeypatch)
        with pytest.raises(ValueError) as info:
            TestBackendSwitch.f4_call(backend="cuda", seed_from_neighbors=True)
        assert BACKEND_ERROR_MESSAGE.format(backend="cuda") in str(info.value)

    def test_the_cpu_cascade_is_untouched(self, monkeypatch):
        # the same keyword under "cpu" never meets the raise or the gate
        def forbidden():
            raise AssertionError("the gate must not run under backend='cpu'")

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", forbidden)
        properties = TestBackendSwitch.f4_call(backend="cpu", seed_from_neighbors=True)
        assert "seed_round" in properties


class TestAvailabilityGate:
    """V9(c), D21.2: HREBSD's own three-stage gate on a FAKE cupy: the
    frozen order with the shim between (b) and (c), the HREBSD-worded
    messages and remedies, the cache with its fresh-copy re-raise,
    stage (c) probing cuFFT at both complex precisions, cuBLAS and
    NVRTC, the identity of the three imported spherical names and the
    separation of the two caches.

    Mutants: M21 (stages reordered, or the shim after (c): the order
    pin), M22 (an FFT-only stage (c): the fake's records), M26 (a gate
    failure through the engine raises; no CPU fallback)."""

    @staticmethod
    def stage_c_calls(calls):
        return [
            c
            for c in calls
            if c[0] in ("fft", "matmul", "RawKernel", "launch", "compile")
        ]

    def test_stage_a_import_failure(self, monkeypatch, fresh_gate):
        hide_cupy(monkeypatch)
        with pytest.raises(ImportError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert message.startswith(GATE_MESSAGE_PREFIX)
        assert "cupy" in message
        assert "install" in message.lower()

    def test_stage_a_version_floor(self, monkeypatch, fresh_gate):
        fake, calls = make_fake_cupy(version="12.6.0")
        install_fake_cupy(monkeypatch, fake)
        with pytest.raises(ImportError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert message.startswith(GATE_MESSAGE_PREFIX)
        assert "12.6.0" in message
        assert str(_gpu._CUPY_MINIMUM_VERSION) in message
        assert calls == []

    @pytest.mark.parametrize(
        "kwargs", [{"device_count": 0}, {"device_error": "no CUDA-capable device"}]
    )
    def test_stage_b_device_failure(self, monkeypatch, fresh_gate, kwargs):
        shim = []
        fake, calls = make_fake_cupy(calls=shim, **kwargs)
        install_fake_cupy(monkeypatch, fake, shim_calls=shim)
        with pytest.raises(RuntimeError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert message.startswith(GATE_MESSAGE_PREFIX)
        assert "device" in message.lower()
        assert "driver" in message.lower()
        # nothing after stage (b) ran
        assert ("shim",) not in calls
        assert self.stage_c_calls(calls) == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fft_error": ImportError("DLL load failed while importing cufft")},
            {"matmul_error": ImportError("DLL load failed while importing cublas")},
            {"kernel_error": RuntimeError("nvrtc: compilation failed")},
        ],
        ids=["cufft", "cublas", "nvrtc"],
    )
    def test_stage_c_library_failure(self, monkeypatch, fresh_gate, kwargs):
        fake, _ = make_fake_cupy(**kwargs)
        install_fake_cupy(monkeypatch, fake, shim_calls=[])
        with pytest.raises(RuntimeError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert message.startswith(GATE_MESSAGE_PREFIX)
        for wheel in GATE_WHEELS:
            assert wheel in message, wheel
        assert "CUDA Toolkit" in message
        assert "Windows" in message

    def test_stage_order_with_the_shim_between_b_and_c(self, monkeypatch, fresh_gate):
        calls = []
        fake, _ = make_fake_cupy(calls=calls)
        install_fake_cupy(monkeypatch, fake, shim_calls=calls)
        _gpu._verify_gpu_or_raise()
        assert calls[0] == ("getDeviceCount",)
        assert calls.count(("shim",)) == 1
        shim = calls.index(("shim",))
        stage_c = [i for i, c in enumerate(calls) if c in self.stage_c_calls(calls)]
        assert stage_c, "stage (c) probed nothing"
        assert shim < min(stage_c)

    def test_stage_c_probes_every_library_the_device_path_calls(
        self, monkeypatch, fresh_gate
    ):
        fake, calls = make_fake_cupy()
        install_fake_cupy(monkeypatch, fake, shim_calls=[])
        _gpu._verify_gpu_or_raise()
        fft_dtypes = {c[2] for c in calls if c[0] == "fft"}
        assert "complex128" in fft_dtypes
        assert "complex64" in fft_dtypes
        assert ("matmul",) in calls
        assert any(c[0] == "RawKernel" for c in calls)
        assert any(c[0] == "launch" for c in calls)

    def test_a_passing_verdict_is_cached(self, monkeypatch, fresh_gate):
        fake, calls = make_fake_cupy()
        install_fake_cupy(monkeypatch, fake, shim_calls=calls)
        _gpu._verify_gpu_or_raise()
        first = list(calls)
        assert first
        _gpu._verify_gpu_or_raise()
        assert calls == first
        assert _gpu._gate_result is not None

    def test_a_cached_failure_re_raises_a_fresh_copy(self, monkeypatch, fresh_gate):
        fake, calls = make_fake_cupy(
            matmul_error=ImportError("DLL load failed while importing cublas")
        )
        install_fake_cupy(monkeypatch, fake, shim_calls=calls)
        with pytest.raises(RuntimeError) as first:
            _gpu._verify_gpu_or_raise()
        probed = list(calls)
        with pytest.raises(RuntimeError) as second:
            _gpu._verify_gpu_or_raise()
        assert calls == probed
        assert second.value is not first.value
        assert type(second.value) is type(first.value)
        assert second.value.args == first.value.args
        assert second.value.__cause__ is first.value

    def test_the_three_spherical_names_by_identity(self):
        assert _gpu._add_nvidia_dll_directories is (
            _spherical_gpu._add_nvidia_dll_directories
        )
        assert _gpu._CUPY_MINIMUM_VERSION is _spherical_gpu._CUPY_MINIMUM_VERSION
        assert _gpu._DEVICE_LOCK is _spherical_gpu._DEVICE_LOCK

    def test_the_two_gate_caches_are_separate(self, monkeypatch, fresh_gate):
        monkeypatch.setattr(_spherical_gpu, "_gate_result", None)
        monkeypatch.setattr(_spherical_gpu, "_add_nvidia_dll_directories", lambda: None)
        fake, _ = make_fake_cupy()
        install_fake_cupy(monkeypatch, fake, shim_calls=[])
        _gpu._verify_gpu_or_raise()
        assert _spherical_gpu._gate_result is None
        monkeypatch.setattr(_gpu, "_gate_result", None)
        _spherical_gpu._verify_gpu_or_raise()
        assert _gpu._gate_result is None

    def test_a_failed_gate_fails_the_engine_call(self, monkeypatch, fresh_gate):
        # M26 through the REAL gate: no cupy, so stage (a) raises out
        # of ``run_hrebsd_dic`` and the CPU runner never runs
        hide_cupy(monkeypatch)

        def forbidden(*args, **kwargs):
            raise AssertionError("no silent CPU fallback (D21.1)")

        monkeypatch.setattr(_engine, "_run_chunks", forbidden)
        monkeypatch.setattr(_engine, "resolve_reference", forbidden)
        with pytest.raises(ImportError) as info:
            TestBackendSwitch.f4_call(backend="gpu", device_precision="float64")
        assert str(info.value).startswith(GATE_MESSAGE_PREFIX)


class TestSeedSeamContract:
    """V9(d), D21.5, numpy namespace: the containers, the dtypes, the
    rows against ``initial_guess`` at upsample 16, 2 and 1 (and every
    row non-finite at 0), the dimmed arm, and the runner's use of the
    seam through the module global, once per preprocessing sub-batch.

    Mutants: M11 (a captured reference: the module-global spy), M12
    and M13 (transposed shift, wrong-side conjugate: row equality),
    M14 (complex64 for complex128: the dtype asserts), M15 (crops not
    ZMN'd: the dimmed arm, possibly equivalent), M47 (upsample hard
    coded: the 2 and 1 arms), M50 (an unpadded tail sub-batch: every
    call has P slots), M52 (``pattern_index`` wrong or in map order:
    the ``SeedBatch`` spy), M53 (the seam once per batch: the call
    count)."""

    def test_the_numpy_kernel_namespace(self):
        kernels = _batched.make_kernel_namespace("numpy", "float64")
        assert isinstance(kernels, _batched.KernelNamespace)
        assert kernels.xp is np
        assert kernels.fft is np.fft
        assert kernels.precision == "float64"
        assert kernels.out_of_memory_error is _batched.NumpyOutOfMemoryError
        assert issubclass(kernels.out_of_memory_error, MemoryError)
        for name in ("gather", "pixel_sums", "reduce_solve_update", "final_criterion"):
            assert callable(getattr(kernels, name)), name

    def test_the_numpy_twin_has_no_mixed_build(self):
        with pytest.raises(ValueError):
            _batched.make_kernel_namespace("numpy", "mixed")

    def test_the_frozen_constants(self):
        assert _batched.SUB_BATCH_SIZE == FROZEN_SUB_BATCH_SIZE
        assert _batched.SEED_PRECISIONS == FROZEN_SEED_PRECISIONS
        assert _batched.DEVICE_PRECISIONS == FROZEN_DEVICE_PRECISIONS
        for name in ("seed_spectra", "seed_homographies"):
            parameters = list(inspect.signature(getattr(_batched, name)).parameters)
            expected = ["ctx", "batch", "seed_state"]
            if name == "seed_homographies":
                expected = ["ctx", "batch", "target_spectra", "seed_state"]
            assert parameters == expected, name

    @pytest.mark.parametrize("precision", FROZEN_SEED_PRECISIONS)
    def test_the_seed_state(self, precision):
        state, _ = seed_case("F4")
        ctx = numpy_seed_context()
        seed_state = _batched.build_seed_state(
            ctx, state, precision=precision, upsample_factor=2
        )
        r0, r1, c0, c1 = state.bounds
        assert seed_state.bounds == tuple(state.bounds)
        assert seed_state.precision == precision
        assert seed_state.upsample_factor == 2
        spectrum = seed_state.reference_spectrum
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.dtype == np.dtype(precision)
        assert spectrum.shape == (r1 - r0, c1 - c0)

    def test_the_default_seed_state_is_complex128(self):
        state, _ = seed_case("F4")
        seed_state = _batched.build_seed_state(numpy_seed_context(), state)
        assert seed_state.precision == "complex128"
        assert seed_state.upsample_factor == 16
        assert seed_state.reference_spectrum.dtype == np.complex128

    @pytest.mark.parametrize("precision", FROZEN_SEED_PRECISIONS)
    def test_spectra_and_rows_layout(self, precision):
        state, targets = seed_case("F4")
        rows, spectra, _ = numpy_seed_rows(state, targets, precision=precision)
        r0, r1, c0, c1 = state.bounds
        assert isinstance(spectra, np.ndarray)
        assert spectra.shape == (len(targets), r1 - r0, c1 - c0)
        assert spectra.dtype == np.dtype(precision)
        assert isinstance(rows, np.ndarray)
        assert rows.shape == (len(targets), N_HOMOGRAPHY_PARAMETERS)
        assert rows.dtype == np.float64
        assert rows.flags.c_contiguous
        # a translation seed: every entry but h13 and h23 exactly zero
        assert np.all(rows[:, [0, 1, 3, 4, 6, 7]] == 0.0)

    @pytest.mark.parametrize("upsample_factor", [16, 2, 1])
    @pytest.mark.parametrize("name", ["F1", "F3s", "F4", "F6", "Ni"])
    def test_rows_equal_initial_guess(self, name, upsample_factor):
        state, targets = seed_case(name)
        rows, _, _ = numpy_seed_rows(state, targets, upsample_factor=upsample_factor)
        expected = cpu_seed_rows(state, targets, upsample_factor=upsample_factor)
        equal = int(
            sum(np.array_equal(a, b, equal_nan=True) for a, b in zip(rows, expected))
        )
        assert_count(
            equal,
            _pin(NUMPY_SEED_EQUAL_COUNT, (name, upsample_factor)),
            f"NUMPY_SEED_EQUAL_COUNT [{name}, {upsample_factor}]",
        )

    def test_upsample_zero_fails_every_row(self):
        state, targets = seed_case("F4")
        # premise: the CPU fails every pattern there (D21.5)
        assert not np.isfinite(cpu_seed_rows(state, targets, upsample_factor=0)).any()
        rows, _, _ = numpy_seed_rows(state, targets, upsample_factor=0)
        assert rows.shape == (len(targets), N_HOMOGRAPHY_PARAMETERS)
        assert not np.isfinite(rows).all(axis=1).any()

    def test_the_dimmed_arm(self):
        # M15: at DIM_SCALE an un-normalised cross-power spectrum sits
        # wholly below the ``100 * eps`` floor (premise asserted), so
        # only crops ZMN'd before the FFT keep the CPU's rows
        f1 = f1_batch()
        state = make_state(np.asarray(f1["reference"]) * DIM_SCALE, f1["pc_px"])
        targets = np.asarray(f1["targets"]) * DIM_SCALE
        preprocessed, _ = preprocessed_targets(state, targets[:1])
        r0, r1, c0, c1 = state.bounds
        cross = np.fft.fft2(state.reference_subregion) * np.conj(
            np.fft.fft2(preprocessed[0, r0:r1, c0:c1])
        )
        assert np.abs(cross).max() < 100 * np.finfo(np.float64).eps
        expected = cpu_seed_rows(state, targets)
        assert np.array_equal(expected, cpu_seed_rows(fixture_state(f1), f1["targets"]))
        rows, _, _ = numpy_seed_rows(state, targets)
        assert np.array_equal(rows, expected)

    @pytest.mark.parametrize("chunksize", [40, 8, 12])
    def test_the_runner_calls_the_seam_once_per_sub_batch(self, monkeypatch, chunksize):
        fixture = f4_spy_batch()
        recorder = install_numpy_session(monkeypatch)
        records = spy_seam(monkeypatch)
        run_fixture(
            fixture, backend="gpu", device_precision="float64", chunksize=chunksize
        )
        n_fit = SPY_MAP_SIZE
        sub_batch = min(FROZEN_SUB_BATCH_SIZE, chunksize)
        n_batches = math.ceil(n_fit / chunksize)
        per_batch = math.ceil(chunksize / sub_batch)
        rows_calls = [r for r in records if r["call"] == "seed_homographies"]
        spectra_calls = [r for r in records if r["call"] == "seed_spectra"]
        assert len(rows_calls) == n_batches * per_batch
        assert len(spectra_calls) == len(rows_calls)
        # spectra then rows, pairwise, the rows call receiving exactly
        # the spectra of its own sub-batch
        for spectra, rows in zip(records[0::2], records[1::2]):
            assert spectra["call"] == "seed_spectra"
            assert rows["call"] == "seed_homographies"
            assert rows["spectra_id"] == spectra["spectra_id"]
            assert np.array_equal(rows["pattern_index"], spectra["pattern_index"])
        # fit order (one grain: map order), every batch padded to B and
        # every sub-batch, the tail included, to P with -1
        expected = []
        for start in range(0, n_fit, chunksize):
            batch = list(range(start, min(start + chunksize, n_fit)))
            batch += [-1] * (chunksize - len(batch))
            for s in range(0, chunksize, sub_batch):
                part = batch[s : s + sub_batch]
                expected.append(part + [-1] * (sub_batch - len(part)))
        session = recorder.sessions[-1]
        for record, index in zip(rows_calls, expected):
            assert record["pattern_index"].tolist() == index
            assert record["pattern_index_dtype"] == np.int64
            assert record["targets_shape"] == (sub_batch, *SHAPE_60)
            assert record["targets_dtype"] == np.float64
            assert record["coefficients_shape"] == (sub_batch, *SHAPE_60)
            assert record["coefficients_dtype"] == np.float32
            assert record["extras"] == {}
            assert record["ctx_xp"] is np
            assert record["ctx_kernels_id"] == id(session.kernels)

    def test_the_sub_batch_carries_the_targets_and_their_coefficients(
        self, monkeypatch
    ):
        fixture = f4_spy_batch()
        install_numpy_session(monkeypatch)
        records = spy_seam(monkeypatch)
        run_fixture(fixture, backend="gpu", device_precision="float64", chunksize=8)
        state = fixture_state(fixture)
        patterns = np.asarray(fixture["patterns"])
        record = [r for r in records if r["call"] == "seed_homographies"][0]
        for slot, index in enumerate(record["pattern_index"]):
            if index < 0:
                continue
            preprocessed, coefficients = preprocessed_targets(state, patterns[[index]])
            scale = np.abs(preprocessed).max()
            np.testing.assert_allclose(
                record["targets"][slot], preprocessed[0], rtol=0, atol=1e-12 * scale
            )
            np.testing.assert_allclose(
                record["coefficients"][slot],
                coefficients[0],
                rtol=1e-6,
                atol=1e-6 * np.abs(coefficients).max(),
            )

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    def test_the_seed_precision_reaches_the_seam(self, monkeypatch, seed_precision):
        install_numpy_session(monkeypatch)
        records = spy_seam(monkeypatch)
        run_fixture(
            f4_batch(),
            backend="gpu",
            device_precision="float64",
            seed_precision=seed_precision,
            upsample_factor=2,
        )
        rows_calls = [r for r in records if r["call"] == "seed_homographies"]
        assert rows_calls
        for record in rows_calls:
            assert record["precision"] == seed_precision
            assert record["reference_spectrum_dtype"] == np.dtype(seed_precision)
            assert record["spectra_dtype"] == np.dtype(seed_precision)
            assert record["upsample_factor"] == 2

    def test_a_planted_non_finite_row_fails_that_pattern_only(self, monkeypatch):
        clean = twin_run("F4")

        def rows_of(pattern_index, rows):
            rows = rows.copy()
            rows[pattern_index == 2] = np.nan
            return rows

        install_numpy_session(monkeypatch)
        plant_seam(monkeypatch, rows_of)
        planted = run_fixture(f4_batch(), backend="gpu", device_precision="float64")
        assert_failure_contract(planted, 2)
        keep = np.array([0, 1, 3])
        for name in clean:
            assert np.array_equal(
                np.asarray(planted[name])[keep],
                np.asarray(clean[name])[keep],
                equal_nan=True,
            ), name


class TestSeedSeamHonoursArbitraryH0Numpy:
    """V9(e) numpy twin at float64 on F1 and F3s (D21.5(iii), Johan
    decision 1): the seam planted with arbitrary rows inside the
    capture range, against the CPU ``fit_pattern(state, pattern,
    h0=row)``.  The planted rows, the compared map points (the
    reference included) and the CPU oracle are the gated suite's
    (:func:`seam_table`, :func:`seam_cpu`; critic finding F1); the
    twin's pins are ``NUMPY_SEAM_*`` keyed ``(fixture, row_type)``.

    Mutant: M10 (the seam's rows ignored and the seed recomputed: the
    one-iteration arm and the discriminating count)."""

    @pytest.mark.parametrize("name", ["F1", "F3s"])
    def test_a_nan_row_gives_the_failure_contract_on_both(self, name):
        twin = seam_twin(name, "nan")
        cpu = seam_cpu(name, "nan")
        # the CPU premise: a NaN h0 is the D2.6 failure contract there
        assert np.all(np.isnan(cpu[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]))
        assert np.all(cpu[:, 9] == 0) and np.all(cpu[:, 11] == 0)
        for index in range(cpu.shape[0]):
            assert_failure_contract(twin, index)

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("name", ["F1", "F3s"])
    def test_planted_rows_reproduce_the_cpu_fit(self, name, row_type):
        fixture = fixture_of(name)
        twin = packed(seam_twin(name, row_type))
        cpu = seam_cpu(name, row_type)
        key = (name, row_type)
        both = (cpu[:, 11] > 0.5) & (twin[:, 11] > 0.5)
        assert_at_least(
            int(both.sum()),
            _pin(NUMPY_SEAM_BOTH_CONVERGED_MIN, key),
            f"NUMPY_SEAM_BOTH_CONVERGED_MIN {key}",
        )
        assert_within(
            h_band(cpu[:, :8], twin[:, :8], fixture["corners"], both),
            NUMPY_PARITY_H_TOL_F64,
            f"NUMPY_PARITY_H_TOL_F64 [seam {key}]",
        )
        assert_count(
            int(np.sum(cpu[:, 9] != twin[:, 9])),
            _pin(NUMPY_SEAM_ITERATION_DIFF_COUNT, key),
            f"NUMPY_SEAM_ITERATION_DIFF_COUNT {key}",
        )
        assert_count(
            int(np.sum((cpu[:, 11] > 0.5) != (twin[:, 11] > 0.5))),
            _pin(NUMPY_SEAM_CONVERGED_FLIP_COUNT, key),
            f"NUMPY_SEAM_CONVERGED_FLIP_COUNT {key}",
        )

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("name", ["F1", "F3s"])
    def test_the_one_iteration_arm(self, name, row_type):
        # the iterate after ONE step is the planted row composed with
        # one update, so a device that recomputes or ignores the rows
        # misses the CPU's at once (M10)
        fixture = fixture_of(name)
        twin = packed(seam_twin(name, row_type, max_iterations=1))
        cpu = seam_cpu(name, row_type, 1)
        assert np.array_equal(twin[:, 9], cpu[:, 9])
        finite = np.isfinite(cpu[:, :8]).all(axis=1)
        assert finite.all()
        assert_within(
            h_band(cpu[:, :8], twin[:, :8], fixture["corners"], finite),
            NUMPY_FIRST_STEP_TOL_F64,
            f"NUMPY_FIRST_STEP_TOL_F64 [seam {name}, {row_type}]",
        )

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("name", ["F1", "F3s"])
    def test_the_planted_rows_are_discriminating(self, name, row_type):
        # CPU side: the planted rows change the CPU's own iteration
        # counts against the translation seed on at least the pinned
        # number of map points, so equal counts on the twin mean the
        # rows were honoured
        planted = seam_cpu(name, row_type)
        translation = seam_cpu(name, None)
        differing = int(np.sum(planted[:, 9] != translation[:, 9]))
        assert_at_least(
            differing,
            _pin(SEAM_DISCRIMINATING_MIN, (name, row_type)),
            f"SEAM_DISCRIMINATING_MIN [{name}, {row_type}]",
        )


class TestBatchedCoreNumpy:
    """V9(f) numpy twins at float64 (F1, F3s, F4), V9(h) batched
    semantics and every knob, V9(i) at kernel level.

    Mutants: M5 (the retire mask ignored: alone against batched and
    the own counts), M6 (exit test before the update: counts, h band),
    M7 (no final criterion: the capped pattern), M8 (a non-finite
    coordinate unflagged), M16 (padded slots in the output: the
    sentinel padding), M28 (``step_scale`` arms), M29, M40, M41 (the
    window arm), M30, M31, M32, M45 (float64 bands and counts), M33
    (``max_iterations`` 0 and 1), M37 to M39 (the first-step band),
    M42 (the ``(None, None)`` DC arm), M43 (a NaN reaching the corner
    norm), M44 (huge finite coordinates), M46 (the ``filter_cutoffs``
    arms), M47 (``upsample_factor`` 8), M48 (``min_step``), M49
    (``border``), M50 (the last-sub-batch slot)."""

    @pytest.mark.parametrize("name", ["F1", "F3s", "F4"])
    def test_parity_with_the_cpu(self, name):
        fixture = FIXTURE_BUILDERS[name]()
        assert_twin_parity(twin_run(name), cpu_run(name), fixture["corners"], name)

    @pytest.mark.parametrize("name", ["F1", "F3s", "F4"])
    def test_the_first_step(self, name):
        fixture = FIXTURE_BUILDERS[name]()
        twin = twin_run(name, max_iterations=1)
        cpu = cpu_run(name, max_iterations=1)
        h_t = np.asarray(twin["homography"])
        h_c = np.asarray(cpu["homography"])
        errors = [recovery_error(a, b, fixture["corners"]) for a, b in zip(h_t, h_c)]
        assert np.isfinite(errors).all()
        assert_within(
            max(errors), NUMPY_FIRST_STEP_TOL_F64, f"NUMPY_FIRST_STEP_TOL_F64 [{name}]"
        )

    def test_batched_equals_alone_on_f5(self, monkeypatch):
        # every pattern of the two grains, fitted in grain-pure batches
        # of B = 8 (the capped one runs 50 iterations beside easy ones
        # that retire after 3 or 4), equals the same pattern fitted
        # ALONE in a batch of one padded to the same B, bitwise
        install_numpy_session(monkeypatch)
        f5 = f5_map()
        states = f5_states()
        patterns = np.asarray(f5["patterns"])
        fit = F5_FIT_INDICES
        grain = F5_STATE_OF_POINT
        batched = run_direct(patterns, fit, grain, states, 8)
        assert batched.shape == (fit.size, ROW_SLOTS["width"])
        for row, (index, g) in enumerate(zip(fit, grain)):
            alone = run_direct(patterns, [index], [g], states, 8)
            assert np.array_equal(batched[row], alone[0], equal_nan=True), index

    def test_own_iteration_counts_on_f5(self):
        twin = twin_run("F5")
        cpu = cpu_run("F5")
        diff = int(
            np.sum(
                np.asarray(twin["num_iterations"]) != np.asarray(cpu["num_iterations"])
            )
        )
        assert_count(
            diff,
            _pin(NUMPY_ITERATION_DIFF_COUNT, "F5"),
            "NUMPY_ITERATION_DIFF_COUNT [F5]",
        )
        # premises of the fixture (measured, f5_map): 4 and 3 iterations
        # for the easy points, the full budget for the capped one
        assert int(np.asarray(cpu["num_iterations"])[F5_CAPPED]) == 50
        assert int(np.asarray(twin["num_iterations"])[F5_CAPPED]) == 50

    def test_the_residual_is_the_final_criterion(self):
        # M7: on the capped pattern the CPU's criterion AT the twin's
        # returned homography (``max_iterations=0`` evaluates it there)
        # is what the twin reports
        twin = twin_run("F5")
        h = np.asarray(twin["homography"])[F5_CAPPED]
        assert np.isfinite(h).all()
        state = f5_states()[1]
        target = np.asarray(f5_map()["patterns"])[F5_CAPPED]
        at_h = fit_pattern(state, target, h0=h, max_iterations=0)["residual"]
        assert_residual_band(
            [np.asarray(twin["residual"])[F5_CAPPED]],
            [at_h],
            NUMPY_PARITY_RESIDUAL_RTOL,
            NUMPY_PARITY_RESIDUAL_ATOL,
            "NUMPY_PARITY_RESIDUAL [F5 capped final criterion]",
        )

    def test_a_last_sub_batch_slot_equals_alone(self, monkeypatch):
        # M50: at B = 40 (P = 32) slot 35 sits in the padded tail
        # sub-batch; it must equal the same pattern alone at B = 40
        install_numpy_session(monkeypatch)
        fixture = f1_tail_batch()
        state = fixture_state(fixture)
        patterns = np.asarray(fixture["patterns"])
        fit = np.arange(TAIL_BATCH_SIZE)
        batched = run_direct(
            patterns, fit, np.zeros_like(fit), [state], TAIL_BATCH_SIZE
        )
        alone = run_direct(patterns, [TAIL_SLOT], [0], [state], TAIL_BATCH_SIZE)
        assert np.array_equal(batched[TAIL_SLOT], alone[0], equal_nan=True)

    def test_padded_slots_never_reach_the_output(self, monkeypatch):
        # M16: the seam hands every PADDED slot a huge finite sentinel
        # row; the output keeps one row per fitted point, unchanged
        install_numpy_session(monkeypatch)
        f4 = f4_batch()
        state = fixture_state(f4)
        patterns = np.asarray(f4["patterns"])
        fit = np.arange(4)
        clean = run_direct(patterns, fit, np.zeros(4), [state], 8)

        def rows_of(pattern_index, rows):
            rows = rows.copy()
            rows[pattern_index < 0] = 1e30
            return rows

        plant_seam(monkeypatch, rows_of)
        planted = run_direct(patterns, fit, np.zeros(4), [state], 8)
        assert planted.shape == (4, ROW_SLOTS["width"])
        assert np.array_equal(planted, clean, equal_nan=True)

    @pytest.mark.parametrize(
        "name, kwargs, all_finite",
        [
            ("F4", (("step_scale", 0.5),), False),
            ("F4", (("step_scale", 1.5),), False),
            ("F4", (("window", True),), False),
            ("F4", (("dead_band", (28, 31, 28, 31)),), False),
            ("F4", (("max_iterations", 0),), True),
            ("F4", (("max_iterations", 1),), True),
            ("F4dc", (("filter_cutoffs", (None, None)),), False),
            ("F4", (("filter_cutoffs", (0.05, 0.4)),), False),
            ("F4", (("upsample_factor", 8),), False),
            ("F4", (("min_step", 1e-2),), False),
            ("F4", (("border", 0.1),), False),
        ],
        ids=[
            "step_scale_0.5",
            "step_scale_1.5",
            "window",
            "dead_band",
            "max_iterations_0",
            "max_iterations_1",
            "no_band_pass_dc_offset",
            "low_pass",
            "upsample_8",
            "min_step",
            "border",
        ],
    )
    def test_every_knob_against_the_cpu(self, name, kwargs, all_finite):
        kwargs = dict(kwargs)
        fixture = FIXTURE_BUILDERS[name]()
        corners = fixture["corners"]
        if "border" in kwargs:
            corners = subregion_corners(
                SHAPE_60, fixture["pc_px"], border=kwargs["border"]
            )
        twin = twin_run(name, **kwargs)
        cpu = cpu_run(name, **kwargs)
        key = f"{name} {kwargs}"
        assert_twin_parity(twin, cpu, corners, key, all_finite=all_finite)
        if kwargs.get("max_iterations") is not None:
            assert np.array_equal(twin["num_iterations"], cpu["num_iterations"])
            assert np.array_equal(twin["converged"], cpu["converged"])
            assert np.array_equal(
                np.isnan(twin["norm_dp"]), np.isnan(np.asarray(cpu["norm_dp"]))
            )

    @staticmethod
    def kernel_inputs(n_slots=3):
        f4 = f4_batch()
        state = fixture_state(f4)
        ctx = numpy_seed_context()
        resident = _batched.build_resident(ctx, state)
        targets = np.asarray(f4["targets"])[:n_slots]
        _, coefficients = preprocessed_targets(state, targets)
        matrices = np.tile(np.eye(3), (n_slots, 1, 1))
        return state, ctx, resident, coefficients, matrices

    @staticmethod
    def cpu_values(state, coefficients, matrix):
        xi_x, xi_y = state.xi_x, state.xi_y
        scale = matrix[2, 0] * xi_x + matrix[2, 1] * xi_y + matrix[2, 2]
        x = (matrix[0, 0] * xi_x + matrix[0, 1] * xi_y + matrix[0, 2]) / scale
        y = (matrix[1, 0] * xi_x + matrix[1, 1] * xi_y + matrix[1, 2]) / scale
        return evaluate(
            coefficients,
            np.ascontiguousarray(x + state.pc_pixels[0] - 0.5),
            np.ascontiguousarray(y + state.pc_pixels[1] - 0.5),
        )

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_gather_flags_a_non_finite_coordinate(self, bad):
        # M8: flagged before the fold, the other slots untouched
        state, ctx, resident, coefficients, matrices = self.kernel_inputs()
        matrices[1, 0, 2] = bad
        values, coordinate_ok = ctx.kernels.gather(resident, coefficients, matrices)
        assert np.asarray(coordinate_ok).tolist() == [True, False, True]
        for slot in (0, 2):
            expected = self.cpu_values(state, coefficients[slot], matrices[slot])
            assert np.array_equal(np.asarray(values)[slot], expected)

    @pytest.mark.parametrize(
        "offset", [3e9, -3e9, 1e30, -1e30, float(np.finfo(np.float32).max)]
    )
    def test_gather_folds_huge_coordinates_like_the_cpu(self, offset):
        # M44: finite coordinates of any size fold IN FLOATING POINT to
        # the CPU's folded value, with no error
        state, ctx, resident, coefficients, matrices = self.kernel_inputs()
        matrices[0, 0, 2] = offset
        matrices[1, 1, 2] = offset
        matrices[2, 0, 2] = -offset
        values, coordinate_ok = ctx.kernels.gather(resident, coefficients, matrices)
        assert np.asarray(coordinate_ok).all()
        for slot in range(3):
            expected = self.cpu_values(state, coefficients[slot], matrices[slot])
            assert np.isfinite(expected).all()
            assert np.array_equal(np.asarray(values)[slot], expected)

    def test_a_nan_reaching_the_step_never_converges(self):
        # NaN sums in slot 1 never give ``converged``; slots 0 and 2 are
        # untouched.  This plants NaN SUMS, not a NaN corner
        # displacement with a finite step (critic finding F6): a
        # CPU-faithful kernel already fails the slot through the
        # separate D21.6.1 non-finite-norm or non-finite-step flags, so
        # this kills M43 (a NaN-swallowing corner maximum) only in an
        # implementation WITHOUT such a flag.  A NaN corner from a
        # finite step needs an exact 0/0 projective corner (a huge
        # finite step gives a finite norm, measured 6175 px at 1e308
        # scaling), so M43 is otherwise reviewed-equivalent at the
        # review gate
        state, ctx, resident, coefficients, matrices = self.kernel_inputs()
        kernels = ctx.kernels
        values, _ = kernels.gather(resident, coefficients, matrices)
        shifts = np.zeros(3)
        sums = np.array(kernels.pixel_sums(resident, values, shifts), dtype=np.float64)
        options = {"min_step": 1e-3, "step_scale": 1.0, "max_iterations": 50}

        def step(planted_sums):
            lockstep = _batched.LockstepState(
                matrices.copy(),
                shifts.copy(),
                np.zeros(3, dtype=np.int64),
                np.full(3, np.nan),
                np.ones(3, dtype=bool),
                np.zeros(3, dtype=bool),
                np.zeros(3, dtype=bool),
            )
            kernels.reduce_solve_update(resident, planted_sums, lockstep, options)
            return lockstep

        clean = step(sums.copy())
        planted_sums = sums.copy()
        planted_sums[1] = np.nan
        planted = step(planted_sums)
        assert not bool(planted.converged[1])
        assert np.isnan(planted.norm_dp[1])
        assert bool(planted.failed[1])
        assert not bool(planted.active[1])
        for slot in (0, 2):
            assert np.array_equal(planted.matrices[slot], clean.matrices[slot])
            assert planted.converged[slot] == clean.converged[slot]
            assert np.array_equal(
                planted.norm_dp[slot], clean.norm_dp[slot], equal_nan=True
            )


class TestRunBatchesNumpySession:
    """The runner through ``run_hrebsd_dic`` with a numpy session built
    by a patched ``_gpu._make_session``: the session contract, the
    V9(a) wiring probes and map order on F5, the V9(i) failure
    contract, the residency spies on F5 and F7, and the V9(n) twins.

    Mutants: M17 (grain order not restored: map order on F7, whose fit
    order differs from its map order; F5's does not), M52
    (``pattern_index`` in map order: the F7 seam spy), M18 (the
    wrong grain's residents: the identity spy), M19 (a device
    exception swallowed: the planted exception), M20 (an out-of-memory
    loop that never halves or never ends: the build and compute spies
    and the B = 1 floor), M23 (the lock dropped or the threaded
    scheduler not forced: the scheduler spy and the lock probe), M9 (a
    zero ZMN norm unflagged: the integer constant), M51 (residents
    never evicted: the F7 spy)."""

    def test_the_session_contract(self):
        session = _gpu._make_session(
            "numpy", 8, device_precision="float64", seed_precision="complex64"
        )
        try:
            assert isinstance(session, _gpu._GpuSession)
            assert session.xp is np
            assert session.fft is np.fft
            assert isinstance(session.kernels, _batched.KernelNamespace)
            assert session.kernels.precision == "float64"
            assert session.batch_size == 8
            assert session.sub_batch_size == 8
            assert session.device_precision == "float64"
            assert session.seed_precision == "complex64"
            assert session.lock is _gpu._DEVICE_LOCK
            assert session.__dask_tokenize__() == (SESSION_TOKEN_NAME, id(session))
            large = _gpu._make_session(
                "numpy", 40, device_precision="float64", seed_precision="complex128"
            )
            assert large.sub_batch_size == FROZEN_SUB_BATCH_SIZE
            large.close()
        finally:
            session.close()
        session.close()  # idempotent

    def test_the_free_device_bytes_query(self):
        free = _gpu._free_device_bytes("numpy")
        assert isinstance(free, int)
        assert free > 0

    def test_the_wiring_probes_on_f5(self):
        twin = twin_run("F5")
        cpu = cpu_run("F5")
        assert set(twin) == set(cpu) == set(_engine.STAGE_A_PROP_NAMES)
        for name in cpu:
            assert np.asarray(twin[name]).dtype == np.asarray(cpu[name]).dtype, name
            assert np.asarray(twin[name]).shape == np.asarray(cpu[name]).shape, name
        assert np.array_equal(twin["grain_id"], cpu["grain_id"])
        assert np.array_equal(twin["reference_index"], cpu["reference_index"])
        for name in ("homography", "Fe", "residual", "norm_dp"):
            assert np.array_equal(
                np.isnan(np.asarray(twin[name])), np.isnan(np.asarray(cpu[name]))
            ), name

    def test_the_map_order_on_f5(self):
        f5 = f5_map()
        twin = twin_run("F5")
        cpu = cpu_run("F5")
        converged = np.asarray(cpu["converged"])
        assert converged[[0, F5_EASY_A, 4, F5_EASY_B]].all()
        for index in np.flatnonzero(converged):
            error = recovery_error(
                np.asarray(twin["homography"])[index],
                np.asarray(cpu["homography"])[index],
                f5["corners"],
            )
            assert error < MAP_ORDER_COARSE_PX, (index, error)
        assert_twin_parity(twin, cpu, f5["corners"], "F5")

    def test_the_map_order_on_f7(self):
        # M17 (critic finding F2): on F5 the grain order IS the map
        # order, so a runner returning rows in fit order passes there;
        # F7's 18 grains of 2 by 2 points are fitted in an order that
        # differs from the map order (premise asserted), so a row left
        # in fit order lands on another point, far beyond the coarse
        # bound (another grain, or another warp of the same reference)
        f7 = f7_map()
        order = grain_fit_order(f7)
        assert not np.array_equal(order, np.sort(order))
        twin = twin_run("F7", chunksize=4)
        cpu = cpu_run("F7", chunksize=4)
        corners = subregion_corners(SHAPE_60, pc_pixels_of(f7["detector"]))
        converged = np.asarray(cpu["converged"])
        assert converged.all()
        for index in range(converged.size):
            error = recovery_error(
                np.asarray(twin["homography"])[index],
                np.asarray(cpu["homography"])[index],
                corners,
            )
            assert error < MAP_ORDER_COARSE_PX, (index, error)
        assert_twin_parity(twin, cpu, corners, "F7")

    def test_the_seam_carries_the_fit_order_on_f7(self, monkeypatch):
        # M52 (critic finding F2): every SeedBatch's pattern_index is a
        # contiguous run of the GRAIN-ordered fit list (never the map
        # order), and each slot's targets and coefficients are those of
        # the pattern at that index
        f7 = f7_map()
        install_numpy_session(monkeypatch)
        records = spy_seam(monkeypatch)
        run_fixture(f7, backend="gpu", device_precision="float64", chunksize=4)
        assert_seam_fit_order(records, f7)
        patterns = np.asarray(f7["patterns"])
        # every grain's state shares the transfer function, the
        # interpolation and the coefficient dtype the targets use
        state = make_state(
            patterns[int(f7["references"][0])], pc_pixels_of(f7["detector"])
        )
        rows_calls = [r for r in records if r["call"] == "seed_homographies"]
        for record in rows_calls:
            for slot, index in enumerate(record["pattern_index"]):
                if index < 0:
                    continue
                preprocessed, coefficients = preprocessed_targets(
                    state, patterns[[index]]
                )
                scale = np.abs(preprocessed).max()
                np.testing.assert_allclose(
                    record["targets"][slot], preprocessed[0], rtol=0, atol=1e-12 * scale
                )
                np.testing.assert_allclose(
                    record["coefficients"][slot],
                    coefficients[0],
                    rtol=1e-6,
                    atol=1e-6 * np.abs(coefficients).max(),
                )

    def test_the_failure_contract_on_f5(self):
        twin = twin_run("F5")
        cpu = cpu_run("F5")
        for index in (F5_MASKED, F5_CONSTANT, F5_NON_FINITE):
            assert_failure_contract(cpu, index)
            assert_failure_contract(twin, index)
        for name in ("grain_id", "reference_index"):
            assert np.asarray(twin[name])[F5_MASKED] == np.asarray(cpu[name])[F5_MASKED]
        h = np.asarray(twin["homography"])[F5_CAPPED]
        assert np.isfinite(h).all()
        assert not bool(np.asarray(twin["converged"])[F5_CAPPED])
        assert np.isnan(np.asarray(twin["Fe"])[F5_CAPPED]).all()
        assert int(np.asarray(twin["num_iterations"])[F5_CAPPED]) == 50

    def test_the_band_passed_constant_never_converges(self):
        twin = twin_run("F5", filter_cutoffs=(0.05, None))
        cpu = cpu_run("F5", filter_cutoffs=(0.05, None))
        assert not bool(np.asarray(cpu["converged"])[F5_CONSTANT])
        assert not bool(np.asarray(twin["converged"])[F5_CONSTANT])

    def test_the_warning_count_equals_the_cpu(self, monkeypatch):
        f5 = f5_map()
        install_numpy_session(monkeypatch)

        def warnings_of(**kwargs):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_hrebsd_dic(
                    np.asarray(f5["patterns"]),
                    f5["navigation_shape"],
                    f5["detector"],
                    reference=np.asarray(f5["references"]),
                    grain_labels=np.asarray(f5["grain_labels"]),
                    navigation_mask=np.array(f5["navigation_mask"]),
                    filter_cutoffs=F5_FILTER_CUTOFFS,
                    verbose=0,
                    **kwargs,
                )
            return [str(w.message) for w in caught if w.category is UserWarning]

        cpu = warnings_of(backend="cpu")
        twin = warnings_of(backend="gpu", device_precision="float64")
        assert len(cpu) >= 1
        assert twin == cpu

    def test_every_batch_runs_against_its_own_grain(self, monkeypatch):
        # M18: per batch, the seed state and the residents the kernels
        # receive are those of the batch's own grain; batches are
        # grain pure
        f5 = f5_map()
        labels = np.asarray(f5["grain_labels"]).ravel()
        install_numpy_session(monkeypatch)
        events = []
        created = []
        resident_of_state = {}
        original_state = _engine.ReferenceState
        original_build_resident = _batched.build_resident
        original_rows = _batched.seed_homographies

        def state_spy(*args, **kwargs):
            state = original_state(*args, **kwargs)
            created.append(state)
            return state

        def build_resident_spy(ctx, state):
            resident = original_build_resident(ctx, state)
            resident_of_state[id(state)] = resident
            return resident

        def rows_spy(ctx, batch, target_spectra, seed_state):
            index = np.asarray(batch.pattern_index)
            grains = set(labels[index[index >= 0]].tolist())
            assert len(grains) <= 1, "a batch straddles grains"
            if grains:
                events.append(("seam", grains.pop()))
            return original_rows(ctx, batch, target_spectra, seed_state)

        def wrap(session):
            gather = session.kernels.gather

            def gather_spy(resident, coefficients, matrices):
                events.append(("gather", id(resident)))
                return gather(resident, coefficients, matrices)

            session.kernels.gather = gather_spy

        monkeypatch.setattr(_engine, "ReferenceState", state_spy)
        monkeypatch.setattr(_batched, "build_resident", build_resident_spy)
        monkeypatch.setattr(_batched, "seed_homographies", rows_spy)
        wrap_sessions(monkeypatch, wrap)
        run_fixture(f5, backend="gpu", device_precision="float64", chunksize=4)
        assert len(created) == 2
        grain_resident = [id(resident_of_state[id(state)]) for state in created]
        current = None
        n_gathers = 0
        for kind, value in events:
            if kind == "seam":
                current = value
            else:
                assert current is not None
                assert value == grain_resident[current], "wrong grain's residents"
                n_gathers += 1
        assert n_gathers > 0
        assert {value for kind, value in events if kind == "seam"} == {0, 1}

    def test_the_residency_bound_on_f7(self, monkeypatch):
        # M51: the least recently used reference is evicted BEFORE the
        # next grain's upload (D21.9.3), so at most R_MAX - 1 residents
        # are alive when an upload is entered (critic finding F7: an
        # evict-after-upload variant holds R_MAX + 1 for a moment and
        # fails here); every grain uploaded once, nothing resident
        # after the run
        f7 = f7_map()
        install_numpy_session(monkeypatch)
        alive = {"resident": [], "seed": []}
        live_counts = {"resident": [], "seed": []}
        uploads = []
        original_resident = _batched.build_resident
        original_seed_state = _batched.build_seed_state

        def count(kind):
            gc.collect()
            live_counts[kind].append(sum(r() is not None for r in alive[kind]))

        def resident_spy(ctx, state):
            count("resident")
            uploads.append(id(state))
            resident = original_resident(ctx, state)
            alive["resident"].append(weakref.ref(resident))
            return resident

        def seed_state_spy(*args, **kwargs):
            count("seed")
            seed_state = original_seed_state(*args, **kwargs)
            alive["seed"].append(weakref.ref(seed_state))
            return seed_state

        monkeypatch.setattr(_batched, "build_resident", resident_spy)
        monkeypatch.setattr(_batched, "build_seed_state", seed_state_spy)
        properties = run_fixture(
            f7, backend="gpu", device_precision="float64", chunksize=4
        )
        assert np.asarray(properties["converged"]).all()
        assert len(uploads) == f7["n_grains"]
        assert len(set(uploads)) == f7["n_grains"]
        for kind in ("resident", "seed"):
            assert len(alive[kind]) == f7["n_grains"], kind
            assert max(live_counts[kind]) <= FROZEN_R_MAX - 1, kind
        gc.collect()
        for kind in ("resident", "seed"):
            assert all(r() is None for r in alive[kind]), f"{kind} survives the run"

    def test_an_out_of_memory_at_build_halves_b(self, monkeypatch):
        reference = twin_run("F4", chunksize=4)
        install_numpy_session(monkeypatch)
        attempts = []
        failing_sessions(monkeypatch, lambda b: b > 4, attempts)
        properties = run_f4_engine(chunksize=8)
        assert attempts == [8, 4]
        assert_properties_bitwise(properties, reference)

    def test_an_out_of_memory_mid_compute_rebuilds_and_re_runs(self, monkeypatch):
        reference = twin_run("F4", chunksize=4)
        recorder = install_numpy_session(monkeypatch)
        closed = []
        wrap_sessions(
            monkeypatch,
            lambda s: setattr(s, "close", _recording_close(s, closed)),
        )
        original = _batched.seed_homographies

        def rows(ctx, batch, target_spectra, seed_state):
            if len(batch.pattern_index) == 8:
                raise _batched.NumpyOutOfMemoryError("planted out of memory")
            return original(ctx, batch, target_spectra, seed_state)

        monkeypatch.setattr(_batched, "seed_homographies", rows)
        properties = run_f4_engine(chunksize=8)
        assert recorder.built == [8, 4]
        assert len(closed) >= 2
        assert_properties_bitwise(properties, reference)

    @pytest.mark.parametrize("window", ["build", "compute"])
    def test_the_batch_one_floor_raises_memory_error(self, monkeypatch, window):
        install_numpy_session(monkeypatch)
        attempts = []
        if window == "build":
            failing_sessions(monkeypatch, lambda b: True, attempts)
        else:
            inner = _gpu._make_session

            def make(namespace, batch_size, **kwargs):
                attempts.append(int(batch_size))
                return inner(namespace, batch_size, **kwargs)

            monkeypatch.setattr(_gpu, "_make_session", make)

            def rows(ctx, batch, target_spectra, seed_state):
                raise _batched.NumpyOutOfMemoryError("planted out of memory")

            monkeypatch.setattr(_batched, "seed_homographies", rows)
        with pytest.raises(MemoryError) as info:
            run_f4_engine(chunksize=8)
        message = str(info.value)
        assert attempts == [8, 4, 2, 1]
        # the D21.10.4 error, not the planted one passing through
        assert type(info.value) is MemoryError
        assert "60" in message
        assert "MB" in message
        assert "backend='cpu'" in message or 'backend="cpu"' in message
        assert "chunksize" in message

    def test_a_device_exception_fails_the_run(self, monkeypatch):
        # M19: no NaN rows, no retry, the session closed
        recorder = install_numpy_session(monkeypatch)
        closed = []
        wrap_sessions(
            monkeypatch,
            lambda s: setattr(s, "close", _recording_close(s, closed)),
        )

        def rows(ctx, batch, target_spectra, seed_state):
            raise RuntimeError("planted device failure")

        monkeypatch.setattr(_batched, "seed_homographies", rows)
        with pytest.raises(RuntimeError, match="planted device failure"):
            run_f4_engine(chunksize=8)
        assert recorder.built == [8]
        assert closed

    def test_the_threaded_scheduler_and_the_device_lock(self, monkeypatch):
        # M23: every compute of the device graph names the threaded
        # scheduler, and the seam runs inside the shared device lock
        install_numpy_session(monkeypatch)
        schedulers = []
        held = []
        original_compute = dask.base.compute

        def compute_spy(*args, **kwargs):
            schedulers.append(kwargs.get("scheduler"))
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(dask.base, "compute", compute_spy)
        monkeypatch.setattr(dask, "compute", compute_spy)
        original_rows = _batched.seed_homographies

        def rows(ctx, batch, target_spectra, seed_state):
            held.append(_gpu._DEVICE_LOCK.locked())
            return original_rows(ctx, batch, target_spectra, seed_state)

        monkeypatch.setattr(_batched, "seed_homographies", rows)
        run_fixture(f4_batch(), backend="gpu", device_precision="float64", chunksize=2)
        assert schedulers
        assert all(s == "threads" for s in schedulers), schedulers
        assert held and all(held)
        assert not _gpu._DEVICE_LOCK.locked()

    def test_no_device_object_survives_and_the_session_closes(self, monkeypatch):
        recorder = install_numpy_session(monkeypatch)
        closed = []
        wrap_sessions(
            monkeypatch,
            lambda s: setattr(s, "close", _recording_close(s, closed)),
        )
        properties = run_fixture(
            f4_batch(), backend="gpu", device_precision="float64", chunksize=2
        )
        for name, value in properties.items():
            assert type(value) is np.ndarray, name
        assert len(recorder.sessions) == 1
        assert len(closed) >= 1


def _recording_close(session, closed):
    """Return a ``close`` that records the call and then closes
    *session* for real (the D21.9.1 ``finally``)."""
    original = session.close

    def close():
        closed.append(id(session))
        return original()

    return close


class TestLazinessNumpySession:
    """V9(m) laziness (D21.11, D21.7.4), numpy session: lazy input stays
    lazy, no computed block holds more than B patterns, the runner
    never receives the whole map, and lazy equals eager bitwise."""

    def test_no_block_exceeds_b_and_the_runner_stays_lazy(self, monkeypatch):
        f4 = f4_batch()
        lazy = da.from_array(np.asarray(f4["patterns"]), chunks=(1, -1, -1))
        install_numpy_session(monkeypatch)
        received = []
        original = _gpu._run_chunks_gpu

        def runner(patterns, *args, **kwargs):
            received.append(patterns)
            return original(patterns, *args, **kwargs)

        monkeypatch.setattr(_gpu, "_run_chunks_gpu", runner)
        sizes = []

        def posttask(key, result, dsk, state, worker_id):
            if isinstance(result, np.ndarray) and result.ndim == 3:
                sizes.append(int(result.shape[0]))

        with Callback(posttask=posttask):
            run_engine(
                lazy,
                f4["navigation_shape"],
                f4["map_detector"],
                reference=(0, 0),
                backend="gpu",
                device_precision="float64",
                chunksize=2,
            )
        assert received
        assert all(isinstance(p, da.Array) for p in received)
        assert sizes
        assert max(sizes) <= 2

    def test_lazy_equals_eager_bitwise(self, monkeypatch):
        f4 = f4_batch()
        eager = twin_run("F4", chunksize=2)
        lazy_patterns = da.from_array(np.asarray(f4["patterns"]), chunks=(1, -1, -1))
        lazy, _ = run_gpu_numpy(
            monkeypatch,
            lazy_patterns,
            f4["navigation_shape"],
            f4["map_detector"],
            reference=(0, 0),
            chunksize=2,
        )
        assert_properties_bitwise(lazy, eager)


class TestLaunchLayout:
    """V9(m) layout pin (D21.7.3): ``_launch_layout`` is a pure
    function of the SUBREGION pixel count; B is not an input at all.

    Mutant: M2 at the definition (the call-site half is the gated
    launch-dimension spy of ``TestGatedDeterminism``)."""

    def test_the_signature_takes_the_pixel_count_only(self):
        assert list(inspect.signature(_batched._launch_layout).parameters) == [
            "n_pixels"
        ]

    @pytest.mark.parametrize("shape", [SHAPE_60, SHAPE_480, SHAPE_RECT])
    def test_the_layout_covers_the_subregion(self, shape):
        n_pixels = subregion_pixels(shape)
        layout = _batched._launch_layout(n_pixels)
        assert layout == _batched._launch_layout(n_pixels)
        threads, blocks, per_thread = layout
        for value in layout:
            assert isinstance(value, (int, np.integer))
            assert value >= 1
        assert threads <= 1024
        assert threads * blocks * per_thread >= n_pixels

    def test_the_layout_is_independent_of_everything_but_the_pixels(self):
        n_pixels = subregion_pixels(SHAPE_RECT)
        first = _batched._launch_layout(n_pixels)
        # building sessions of other batch sizes changes nothing
        for batch_size in (1, 8, 32, 40, 64):
            session = _gpu._make_session(
                "numpy",
                batch_size,
                device_precision="float64",
                seed_precision="complex128",
            )
            session.close()
            assert _batched._launch_layout(n_pixels) == first


class TestCudaSourcePins:
    """V9(m) source pins (D21.7.2, D21.4): no atomic reduction anywhere
    in the CUDA source and no ``--use_fast_math``; the source exists in
    ``_hrebsd`` so the pins cannot pass vacuously.

    Mutant: M1 (an atomic reduction in place of the two-stage one, in
    CUDA C or through cupy's ``scatter_add`` / ``.add.at``)."""

    def test_the_cuda_source_exists(self):
        texts = cuda_source_texts()
        assert any("__global__" in text for text in texts.values()), sorted(texts)

    def test_no_atomic_anywhere(self):
        for name, text in cuda_source_texts().items():
            assert re.search(r"\batomic\w*\s*\(", text) is None, name

    def test_no_atomic_scatter_through_cupy(self):
        # an M1 variant (critic finding F9): a device reduction through
        # cupy's atomic scatter APIs is as non-deterministic (ledger 95)
        for path in hrebsd_module_paths():
            text = path.read_text(encoding="utf-8")
            assert "scatter_add" not in text, path.name
            assert re.search(r"\.add\.at\s*\(", text) is None, path.name

    def test_no_fast_math(self):
        # the compile OPTION, as a string literal; prose naming the
        # flag (as the skeleton's docstrings do) is allowed
        for name, text in cuda_source_texts().items():
            pattern = r"""["']-{1,2}use_fast_math["']"""
            assert re.search(pattern, text) is None, name


class TestBatchModel:
    """V9(o), D21.10: the pure-math VRAM model, the default B chooser,
    an explicit ``chunksize``, the ``chunksize`` floor, no clamp to the
    map size, and the information message.

    Mutants: M34 (an explicit ``chunksize`` ignored: the batch spy),
    M35 (the model without the P term: the formula pin and the faked
    free VRAM where only the P and R terms decide)."""

    N_PIXELS_RECT = SHAPE_RECT[0] * SHAPE_RECT[1]

    @staticmethod
    def precisions():
        return [
            (d, s) for d in FROZEN_DEVICE_PRECISIONS for s in FROZEN_SEED_PRECISIONS
        ]

    def test_the_frozen_structural_values(self):
        assert _gpu.R_MAX == FROZEN_R_MAX
        assert _gpu._BATCH_SIZES == FROZEN_BATCH_SIZES
        assert _gpu._FREE_VRAM_FRACTION == FROZEN_FREE_VRAM_FRACTION

    def test_the_model_is_the_frozen_formula(self):
        for device_precision, seed_precision in self.precisions():
            g, p, r = _gpu._vram_model_terms(
                self.N_PIXELS_RECT, device_precision, seed_precision
            )
            for term in (g, p, r):
                assert isinstance(term, int)
                assert term > 0
            for batch_size in (*FROZEN_BATCH_SIZES, 40, 50, 100):
                expected = (
                    batch_size * g
                    + min(FROZEN_SUB_BATCH_SIZE, batch_size) * p
                    + FROZEN_R_MAX * r
                )
                assert (
                    _gpu._vram_model_bytes(
                        batch_size, self.N_PIXELS_RECT, device_precision, seed_precision
                    )
                    == expected
                )

    def test_the_terms_order_as_measured(self):
        terms = {
            key: _gpu._vram_model_terms(self.N_PIXELS_RECT, *key)
            for key in self.precisions()
        }
        # float64 residents exceed mixed ones, complex128 seed spectra
        # complex64 ones (ledger 92)
        assert terms[("float64", "complex128")][2] > terms[("mixed", "complex128")][2]
        assert terms[("mixed", "complex128")][2] > terms[("mixed", "complex64")][2]
        assert terms[("mixed", "complex128")][1] >= terms[("mixed", "complex64")][1]
        small = _gpu._vram_model_terms(
            SHAPE_480[0] * SHAPE_480[1], "mixed", "complex128"
        )
        for a, b in zip(small, terms[("mixed", "complex128")]):
            assert a < b

    def test_the_terms_sit_in_the_calibrated_bounds(self):
        for device_precision, bounds_r in (
            ("mixed", VRAM_R_BOUNDS_RECT_MIXED),
            ("float64", VRAM_R_BOUNDS_RECT_F64),
        ):
            g, p, r = _gpu._vram_model_terms(
                self.N_PIXELS_RECT, device_precision, "complex128"
            )
            for value, bounds, name in (
                (g, VRAM_G_BOUNDS_RECT, "VRAM_G_BOUNDS_RECT"),
                (p, VRAM_P_BOUNDS_RECT, "VRAM_P_BOUNDS_RECT"),
                (r, bounds_r, f"VRAM_R_BOUNDS_RECT [{device_precision}]"),
            ):
                if bounds is None:
                    assert_within(value, None, name)
                low, high = bounds
                assert low <= value <= high, f"{name}: {value} not in {bounds}"

    def test_the_default_batch_size_rule(self):
        n = self.N_PIXELS_RECT
        for device_precision, seed_precision in self.precisions():

            def model(b):
                return _gpu._vram_model_bytes(b, n, device_precision, seed_precision)

            def choose(free):
                return _gpu._default_batch_size(
                    free, n, device_precision, seed_precision
                )

            assert choose(0) == 1
            assert choose(10**15) == 64
            for b in FROZEN_BATCH_SIZES:
                free = 2 * model(b)
                assert choose(free) == b
                if b > 1:
                    assert choose(free - 2) < b

    def test_the_p_and_r_terms_decide(self):
        # M35: free VRAM where the per-slot term alone, and the model
        # without its P term, would both allow 64
        n = self.N_PIXELS_RECT
        g, p, r = _gpu._vram_model_terms(n, "mixed", "complex128")
        free = 2 * (64 * g + FROZEN_R_MAX * r)
        chosen = _gpu._default_batch_size(free, n, "mixed", "complex128")
        assert chosen < 64
        assert _gpu._vram_model_bytes(chosen, n, "mixed", "complex128") <= 0.5 * free

    def test_the_model_makes_no_device_query(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("the VRAM model queried the device")

        monkeypatch.setattr(_gpu, "_free_device_bytes", forbidden)
        hide_cupy(monkeypatch)
        _gpu._vram_model_terms(3600, "mixed", "complex128")
        _gpu._vram_model_bytes(8, 3600, "mixed", "complex128")
        _gpu._default_batch_size(10**9, 3600, "mixed", "complex128")

    def test_an_explicit_chunksize_wins(self, monkeypatch):
        _, recorder = run_f4_twin(monkeypatch, chunksize=4)
        assert recorder.built == [4]

    def test_the_default_chunksize_is_the_chooser(self, monkeypatch):
        n = SHAPE_60[0] * SHAPE_60[1]
        free = 2 * _gpu._vram_model_bytes(8, n, "float64", "complex128")
        recorder = install_numpy_session(monkeypatch, free_bytes=free)
        f4 = f4_batch()
        run_engine(
            np.asarray(f4["patterns"]),
            f4["navigation_shape"],
            f4["map_detector"],
            reference=(0, 0),
            backend="gpu",
            device_precision="float64",
        )
        assert recorder.built == [8]

    @pytest.mark.parametrize("chunksize", [0, -1])
    def test_a_chunksize_below_one_raises_under_gpu_only(self, monkeypatch, chunksize):
        install_numpy_session(monkeypatch)
        with pytest.raises(ValueError, match="chunksize"):
            TestBackendSwitch.f4_call(
                backend="gpu", device_precision="float64", chunksize=chunksize
            )
        # the CPU path keeps its silent clamp
        properties = TestBackendSwitch.f4_call(backend="cpu", chunksize=chunksize)
        assert np.asarray(properties["converged"]).all()

    def test_a_small_map_runs_one_padded_batch(self, monkeypatch):
        # B is never clamped to the number of fitted points: three
        # points run one batch at the default B, padded
        f4 = f4_batch()
        n = SHAPE_60[0] * SHAPE_60[1]
        free = 8 * 2**30
        expected_b = _gpu._default_batch_size(free, n, "float64", "complex128")
        recorder = install_numpy_session(monkeypatch, free_bytes=free)
        records = spy_seam(monkeypatch, keep_states=False)
        detector = make_detector(shape=SHAPE_60, binning=8, navigation_shape=(1, 3))
        run_engine(
            np.asarray(f4["patterns"])[:3],
            (1, 3),
            detector,
            reference=(0, 0),
            backend="gpu",
            device_precision="float64",
        )
        assert recorder.built == [expected_b]
        sub_batch = min(FROZEN_SUB_BATCH_SIZE, expected_b)
        rows_calls = [r for r in records if r["call"] == "seed_homographies"]
        assert len(rows_calls) == math.ceil(expected_b / sub_batch)
        indices = np.concatenate([r["pattern_index"] for r in rows_calls])
        assert indices.size == expected_b
        assert indices[:3].tolist() == [0, 1, 2]
        assert np.all(indices[3:] == -1)

    def test_the_information_message_carries_the_device_block(
        self, monkeypatch, capsys
    ):
        # V9(o), D21.10.5: B and P both named; at chunksize 40 they
        # differ (B = 40, P = 32), so a message missing P or printing
        # B for P fails (critic finding F8)
        f4 = f4_batch()
        recorder = install_numpy_session(monkeypatch)
        run_engine(
            np.asarray(f4["patterns"]),
            f4["navigation_shape"],
            f4["map_detector"],
            reference=(0, 0),
            backend="gpu",
            device_precision="float64",
            chunksize=40,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert recorder.built == [40]
        assert "VRAM" in out
        assert "MB" in out
        assert re.search(r"\b40\b", out)
        assert re.search(r"\b32\b", out)
        assert "exceed" not in out.lower()

    def test_the_information_message_warns_above_free_vram(self, monkeypatch, capsys):
        f4 = f4_batch()
        install_numpy_session(monkeypatch, free_bytes=1024)
        run_engine(
            np.asarray(f4["patterns"]),
            f4["navigation_shape"],
            f4["map_detector"],
            reference=(0, 0),
            backend="gpu",
            device_precision="float64",
            chunksize=4,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert "exceed" in out.lower()


class TestDriftTripwireCpuHalf:
    """V9(l), D21.8(h), the CPU HALF of the drift tripwire, runnable
    wherever no GPU exists: F1 through ``run_hrebsd_dic(backend=
    "cpu")``, per case the recovery error against the EXACT imposed
    homography, pinned to the literals measured at this gate.

    Mutant: M27 (a shared helper perturbed, moving the CPU path)."""

    def test_the_cpu_recovery_matches_the_literals(self):
        errors, _ = drift_recovery_cpu()
        assert errors.shape == CPU_DRIFT_RECOVERY_PX.shape
        drift = np.abs(errors - CPU_DRIFT_RECOVERY_PX)
        assert np.all(drift <= CPU_DRIFT_TRIPWIRE_PX), drift.max()

    def test_the_cpu_iterations_match_the_literals(self):
        _, iterations = drift_recovery_cpu()
        assert np.array_equal(iterations, CPU_DRIFT_NUM_ITERATIONS)
        converged = np.asarray(run_fixture(f1_batch(), backend="cpu")["converged"])
        assert converged.all()

    def test_the_fit_pattern_route_agrees(self):
        fixture = f1_batch()
        state = fixture_state(fixture)
        errors = np.array(
            [
                recovery_error(result["h"], h_true, fixture["corners"])
                for result, h_true in zip(
                    cpu_fits(state, fixture["targets"]), fixture["exact"]
                )
            ]
        )
        assert np.all(np.abs(errors - CPU_DRIFT_RECOVERY_PX) <= CPU_DRIFT_TRIPWIRE_PX)


class TestFixtureGating:
    """V9(p), D21.14.2: the ``cupy_gpu`` fixture's frozen decision
    order, the kill switch, then the structural xdist skip, then (and
    only then) the availability probe, whose message is the reason."""

    def test_the_kill_switch(self, monkeypatch):
        monkeypatch.setenv("KIKUCHIPY_NO_GPU_TESTS", "1")
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")

        def forbidden():
            raise AssertionError("the probe must not run under the kill switch")

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", forbidden)
        reason = _cupy_gpu_skip_reason()
        assert reason is not None
        assert "KIKUCHIPY_NO_GPU_TESTS" in reason

    def test_the_xdist_worker_skip(self, monkeypatch):
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")

        def forbidden():
            raise AssertionError("the probe must not run under xdist")

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", forbidden)
        reason = _cupy_gpu_skip_reason()
        assert reason is not None
        assert "-n 0" in reason

    def test_a_gate_failure_is_the_reason(self, monkeypatch):
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        frozen = f"{GATE_MESSAGE_PREFIX} a remedy goes here"

        def failing():
            raise ImportError(frozen)

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", failing)
        assert _cupy_gpu_skip_reason() == frozen

    def test_a_passing_gate_runs_the_suite(self, monkeypatch):
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", lambda: None)
        assert _cupy_gpu_skip_reason() is None

    def test_the_reason_here_is_the_hrebsd_gate_message(self, monkeypatch, fresh_gate):
        # on any machine the probe's reason is the HREBSD gate's own
        # message (or None where the GPU is usable), never a bare
        # error of a gate that does not exist yet
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        reason = _cupy_gpu_skip_reason()
        if reason is not None:
            assert reason.startswith(GATE_MESSAGE_PREFIX), reason

    def test_the_wgpu_marker_is_not_reused(self):
        source = inspect.getsource(sys.modules[__name__])
        assert re.search(r"@pytest\.mark\.gpu\b", source) is None


class TestExpectGpuCanary:
    """V9(p), D21.14.2: the green-by-skip guard.  With
    ``KIKUCHIPY_EXPECT_GPU`` set (the gated-run recipe sets it), a
    machine that expects the gated suite to run FAILS here on any gate
    or DLL-shim regression instead of skipping every gated test."""

    def test_the_gate_passes_when_a_gpu_is_expected(self):
        if not os.environ.get("KIKUCHIPY_EXPECT_GPU"):
            pytest.skip(
                "KIKUCHIPY_EXPECT_GPU is not set; set it in the gated-run "
                "command on machines where the gated HREBSD GPU suite is "
                "expected to run, so a gate or DLL-shim regression fails this "
                "canary instead of silently skipping the gated suite"
            )
        if os.environ.get("KIKUCHIPY_NO_GPU_TESTS"):
            pytest.skip("the KIKUCHIPY_NO_GPU_TESTS kill switch wins")
        if os.environ.get("PYTEST_XDIST_WORKER") is not None:
            pytest.skip("structural xdist skip; the canary guards -n 0 runs only")
        reason = _cupy_gpu_skip_reason()
        assert reason is None, (
            "KIKUCHIPY_EXPECT_GPU is set but the gated HREBSD GPU suite would "
            f"skip here: {reason}"
        )


# The import probe the two V9(p) subprocesses run first: it records
# every ``cupy`` import whose IMPORTING code is a ``kikuchipy`` module,
# through ``__import__`` (every import statement, cached or not) and
# ``importlib.import_module``.  A bare "``'cupy' not in sys.modules``"
# check is wrong wherever cupy is installed: ``dask.array.chunk_types``
# try-imports cupy itself at module scope, so any process that imports
# ``dask.array`` holds cupy (found 2026-10-06 in the pinned overlay)
CUPY_IMPORTER_PROBE = (
    "import builtins, importlib, sys\n"
    "_CUPY_IMPORTERS = []\n"
    "def _record(name, frame):\n"
    "    caller = frame.f_globals.get('__name__', '') if frame else ''\n"
    "    if name.split('.')[0] == 'cupy' and caller.split('.')[0] == 'kikuchipy':\n"
    "        _CUPY_IMPORTERS.append((caller, name))\n"
    "_builtin_import = builtins.__import__\n"
    "def _probe_import(name, *args, **kwargs):\n"
    "    _record(name, sys._getframe(1))\n"
    "    return _builtin_import(name, *args, **kwargs)\n"
    "builtins.__import__ = _probe_import\n"
    "_import_module = importlib.import_module\n"
    "def _probe_import_module(name, *args, **kwargs):\n"
    "    _record(name, sys._getframe(1))\n"
    "    return _import_module(name, *args, **kwargs)\n"
    "importlib.import_module = _probe_import_module\n"
)


class TestNoModuleScopeCupy:
    """V9(p), D21.13 and D21.9.5: no module-scope cupy anywhere in
    ``_hrebsd`` (AST, nested module-level blocks included, and a source
    regex), ``import kikuchipy`` and a numpy session import no cupy (a
    subprocess, so a gated test cannot poison it), the new modules use
    neither ``typing`` nor ``__future__``, and the import direction:
    ``_engine`` imports ``_gpu`` at module scope, ``_gpu`` and
    ``_batched`` never import ``_engine`` there.

    Mutant: M24 (cupy imported at module scope)."""

    def test_no_module_scope_cupy_by_ast(self):
        for path in hrebsd_module_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node, module in module_scope_imports(tree):
                assert module.split(".")[0] != "cupy", f"{path.name}:{node.lineno}"

    def test_no_module_scope_cupy_by_regex(self):
        for path in hrebsd_module_paths():
            source = path.read_text(encoding="utf-8")
            assert not re.search(r"^(import cupy|from cupy)", source, re.MULTILINE), (
                path.name
            )

    def test_importing_kikuchipy_imports_no_cupy(self):
        script = CUPY_IMPORTER_PROBE + (
            "import kikuchipy\n"
            "import kikuchipy.indexing._hrebsd._engine\n"
            "import kikuchipy.indexing._hrebsd._gpu\n"
            "import kikuchipy.indexing._hrebsd._batched\n"
            "assert not _CUPY_IMPORTERS, ('cupy imported at module scope', "
            "_CUPY_IMPORTERS)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr

    def test_a_numpy_session_imports_no_cupy(self):
        script = CUPY_IMPORTER_PROBE + (
            "from kikuchipy.indexing._hrebsd import _batched, _gpu\n"
            "_batched.make_kernel_namespace('numpy', 'float64')\n"
            "session = _gpu._make_session('numpy', 2, device_precision='float64', "
            "seed_precision='complex128')\n"
            "session.close()\n"
            "_gpu._vram_model_bytes(8, 3600, 'mixed', 'complex128')\n"
            "assert not _CUPY_IMPORTERS, ('the numpy path imported cupy', "
            "_CUPY_IMPORTERS)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr

    def test_the_new_modules_use_neither_typing_nor_future(self):
        for module in (_gpu, _batched):
            tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
            for node, name in module_scope_imports(tree):
                root = name.lstrip(".").split(".")[0]
                assert root not in ("typing", "__future__"), (module.__name__, name)

    def test_the_new_modules_never_import_the_engine_at_module_scope(self):
        for module in (_gpu, _batched):
            tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
            for node, name in module_scope_imports(tree):
                assert "_engine" not in name.split("."), (module.__name__, name)

    def test_the_engine_imports_the_gate_seam_at_module_scope(self):
        assert _engine._gpu is _gpu
        assert _engine._verify_gpu_or_raise is _gpu._verify_gpu_or_raise
        tree = ast.parse(pathlib.Path(_engine.__file__).read_text(encoding="utf-8"))
        names = [name for _, name in module_scope_imports(tree)]
        assert "kikuchipy.indexing._hrebsd._gpu" in names


# ===================== Locally gated GPU suite ====================== #
#
# Everything below needs the ``cupy_gpu`` fixture: it runs ONLY at
# ``-n 0`` through the PINNED overlay of D21.15, on a machine where the
# HREBSD gate passes, and every device-side number is machine specific
# (:data:`MACHINE_A`).  Against the Stage E skeleton every class here
# SKIPS, the gate's ``NotImplementedError`` text being the skip reason
# (the failing-tests gate's expected outcome, recorded in
# validation.md); a correct implementation turns them green once the
# ``FIXME-pin`` placeholders are measured and pinned.  F2 and F3 arms
# of the planted-seam oracle carry the ``weekly`` marker for their CPU
# oracle cost; the gate command runs ``--weekly``.

# The pinned overlay of D21.15: every device-side pin is valid for
# exactly these distributions (a bump re-measures the pins first)
GATED_PINNED_OVERLAY = {
    "cupy-cuda12x": "14.2.0",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cusolver-cu12": "11.7.5.82",
    "nvidia-cusparse-cu12": "12.5.10.65",
    "nvidia-nvjitlink-cu12": "12.9.86",
}

# "At most 10 iterations" of the V9(h) alone-against-batched arm
GATED_ALONE_MAX_ITERATIONS = 10

# The planted finite coordinates of V9(i) (D21.6.3)
GATED_FLT_MAX = float(np.finfo(np.float32).max)
GATED_PLANTED_OFFSETS = (3e9, -3e9, 1e30, -1e30, GATED_FLT_MAX, -GATED_FLT_MAX)

# The knob arms of V9(h), each against the CPU on F4
GATED_KNOB_ARMS = {
    "step_scale_0.5": {"step_scale": 0.5},
    "step_scale_1.5": {"step_scale": 1.5},
    "window": {"window": True},
    "dead_band": {"dead_band": (28, 31, 28, 31)},
    "filter_low_pass": {"filter_cutoffs": (0.05, 0.4)},
    "upsample_8": {"upsample_factor": 8},
    "min_step_1e-2": {"min_step": 1e-2},
    "border_0.1": {"border": 0.1},
}

# The seed-parity fixtures (V9(d) gated: F1 to F4) and the per-point
# parity fixtures (V9(f), (g): F1 to F4, plus the F6 pin map)
GATED_SEED_FIXTURES = ("F1", "F2-0", "F2-1", "F3", "F4")
GATED_PARITY_FIXTURES = ("F1", "F2-0", "F2-1", "F3", "F4", "F6")


def _weekly_if_large(names):
    """Return pytest params of *names*, the 64-pattern F2 and F3 sets
    marked ``weekly`` (their CPU oracle dominates the gated wall time)."""
    return [
        pytest.param(n, marks=pytest.mark.weekly) if n[:2] in ("F2", "F3") else n
        for n in names
    ]


def _assert_in_bounds(measured, bounds, name: str) -> None:
    """Assert ``low <= measured <= high`` for a pinned ``(low, high)``,
    failing loudly while *bounds* is an unfilled placeholder."""
    if bounds is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V9(o)); "
            f"measured {measured!r} bytes. Pin (low, high) with the recipe and "
            "the machine ID"
        )
    low, high = bounds
    assert low <= measured <= high, f"{name}: {measured} not in [{low}, {high}]"


def _h_tol(device_precision: str):
    """Return the V9(f) homography band of *device_precision*."""
    if device_precision == "mixed":
        return GPU_PARITY_H_TOL_MIXED
    return GPU_PARITY_H_TOL_F64


def _first_step_tol(device_precision: str):
    """Return the V9(f) first-step band of *device_precision*."""
    if device_precision == "mixed":
        return GPU_FIRST_STEP_TOL_MIXED
    return GPU_FIRST_STEP_TOL_F64


def _gpu_kwargs(device_precision="mixed", seed_precision="complex128") -> dict:
    """Return the engine keywords of a device run."""
    return {
        "backend": "gpu",
        "device_precision": device_precision,
        "seed_precision": seed_precision,
    }


def _fitted(properties: dict) -> np.ndarray:
    """Return the flat mask of the points the run fitted (a reference
    resolved and a nonzero iteration count or a failure row)."""
    return np.asarray(properties["reference_index"]) >= 0


def _both_converged(cpu: dict, gpu: dict) -> np.ndarray:
    """Return the mask of points BOTH backends converged."""
    return np.asarray(cpu["converged"]) & np.asarray(gpu["converged"])


def assert_device_residual_band(cpu_residual, gpu_residual, name: str) -> None:
    """Assert the V9(f) DEVICE residual band, ``|gpu - cpu| <= atol +
    rtol * |cpu|`` elementwise, through :func:`assert_residual_band`
    with the ``GPU_PARITY_RESIDUAL_*`` pins."""
    assert_residual_band(
        gpu_residual,
        cpu_residual,
        GPU_PARITY_RESIDUAL_RTOL,
        GPU_PARITY_RESIDUAL_ATOL,
        f"GPU_PARITY_RESIDUAL {name}",
    )


def _count_differences(cpu: dict, gpu: dict, where) -> tuple[int, int]:
    """Return ``(iteration differences, converged flips)`` over the
    flat indices where *where* holds, the (g) COUNTS."""
    where = np.asarray(where, dtype=bool)
    iterations = np.asarray(cpu["num_iterations"]) != np.asarray(gpu["num_iterations"])
    flips = np.asarray(cpu["converged"]) != np.asarray(gpu["converged"])
    return int((iterations & where).sum()), int((flips & where).sum())


def _assert_host_outputs_bitwise(cpu: dict, gpu: dict) -> None:
    """Assert the D21.8(a) wiring probes: the property set with its
    names, dtypes and shapes, and every host-only output, bitwise."""
    assert set(cpu) == set(gpu)
    for name in cpu:
        a = np.asarray(cpu[name])
        b = np.asarray(gpu[name])
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert isinstance(gpu[name], np.ndarray), name
    for name in ("grain_id", "reference_index"):
        assert np.array_equal(np.asarray(cpu[name]), np.asarray(gpu[name])), name


def _relative_difference(a, b) -> float:
    """Return ``max|a - b| / max|a|`` (NaN positions must agree), the
    one scale-free metric of the per-kernel A/B oracle."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN positions differ"
    finite = ~np.isnan(a)
    if not finite.any():
        return 0.0
    scale = max(float(np.abs(a[finite]).max()), np.finfo(np.float64).tiny)
    return float(np.abs(a[finite] - b[finite]).max()) / scale


def _device_context(cupy, device_precision="float64") -> "_batched.SeedContext":
    """Return the cupy :class:`~kikuchipy.indexing._hrebsd._batched.
    SeedContext` at *device_precision*."""
    kernels = _batched.make_kernel_namespace("cupy", device_precision)
    return _batched.SeedContext(cupy, cupy.fft, kernels)


def _device_seed(
    cupy,
    state,
    preprocessed,
    coefficients,
    *,
    precision="complex128",
    upsample_factor=16,
):
    """Return ``(seed_state, batch, target_spectra, h0)`` of the device
    seam on host *preprocessed* targets and their *coefficients*: the
    two frozen seam functions called directly, one sub-batch of all
    targets, ``pattern_index`` a host ``arange``."""
    ctx = _device_context(cupy)
    seed_state = _batched.build_seed_state(
        ctx, state, precision=precision, upsample_factor=upsample_factor
    )
    batch = _batched.SeedBatch(
        cupy.asarray(preprocessed),
        cupy.asarray(coefficients),
        np.arange(len(preprocessed), dtype=np.int64),
    )
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    h0 = _batched.seed_homographies(ctx, batch, spectra, seed_state)
    return seed_state, batch, spectra, h0


def _equal_rows(a, b) -> int:
    """Return the number of rows of two ``(N, 8)`` arrays that are
    bitwise equal (NaN equal to NaN)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return int(
        sum(np.array_equal(a[i], b[i], equal_nan=True) for i in range(a.shape[0]))
    )


def _spy_sessions(monkeypatch, on_build=None, on_session=None) -> list:
    """Wrap the module-global session factory: record every build's
    batch size (in order), call ``on_build(attempt, batch_size)``
    before the real build (it may raise) and ``on_session(attempt,
    session)`` after it.  Returns the batch-size list."""
    built = []
    real = _gpu._make_session

    def make_session(namespace, batch_size, **kwargs):
        built.append(int(batch_size))
        if on_build is not None:
            on_build(len(built), int(batch_size))
        session = real(namespace, batch_size, **kwargs)
        if on_session is not None:
            on_session(len(built), session)
        return session

    monkeypatch.setattr(_gpu, "_make_session", make_session)
    return built


def _out_of_memory(cupy):
    """Return a fresh cupy out-of-memory error instance."""
    return cupy.cuda.memory.OutOfMemoryError(0, 0)


def _f5_runner_inputs(max_iterations=GATED_ALONE_MAX_ITERATIONS):
    """Return ``(patterns, states, options)`` of F5 for direct
    ``_run_chunks_gpu`` calls: the two grain references' states built
    as the engine builds them at ``F5_FILTER_CUTOFFS``."""
    patterns = np.asarray(f5_map()["patterns"])
    options = dict(DEFAULT_FIT_OPTIONS)
    options["max_iterations"] = int(max_iterations)
    return patterns, list(f5_states()), options


def _lockstep(xp, matrices) -> "_batched.LockstepState":
    """Return a fresh :class:`~kikuchipy.indexing._hrebsd._batched.
    LockstepState` of *matrices* in *xp*: zero shifts and iterations,
    NaN ``norm_dp``, every slot active, none converged or failed."""
    matrices = xp.asarray(np.array(matrices, dtype=np.float64))
    n = int(matrices.shape[0])
    return _batched.LockstepState(
        matrices,
        xp.zeros(n, dtype=xp.float64),
        xp.zeros(n, dtype=xp.int64),
        xp.full(n, np.nan, dtype=xp.float64),
        xp.ones(n, dtype=bool),
        xp.zeros(n, dtype=bool),
        xp.zeros(n, dtype=bool),
    )


def _lockstep_to_host(cupy, lockstep) -> dict:
    """Return the fields of a :class:`LockstepState` as host arrays."""
    return {
        name: cupy.asnumpy(getattr(lockstep, name))
        for name in (
            "matrices",
            "shifts",
            "iterations",
            "norm_dp",
            "active",
            "converged",
            "failed",
        )
    }


def _ab_inputs(n_targets=4):
    """Return ``(state, coefficients, matrices)`` of the per-kernel A/B
    oracle: F1's reference state, the f32 spline coefficients of its
    first *n_targets* targets and their EXACT imposed homographies as
    carried matrices (inside the basin, off the identity)."""
    fixture = f1_batch()
    state = fixture_state(fixture)
    targets = np.asarray(fixture["targets"])[:n_targets]
    _, coefficients = preprocessed_targets(state, targets)
    matrices = np.stack([matrix_of(h) for h in fixture["exact"][:n_targets]])
    return state, coefficients, matrices


def _hand_built_update(state, target, h0, n_iterations) -> np.ndarray:
    """Return the homography after *n_iterations* of the INDEPENDENT
    hand-built IC-GN loop of ``test_hrebsd_engine.py::
    TestReferenceStateAndUpdate::test_iterations_match_the_hand_built_
    update``, on the host precompute the device uploads (D21.3) and
    with the target's coefficients stored f32 as on the device
    (D21.4): accumulated ``W <- W . W(dp)**-1``, ``H dp = -g`` by a
    plain solve, W33 renormalisation, the ORIGINAL target re-warped."""
    preprocessed = preprocess(target, transfer_function=state.transfer_function)
    coefficients = np.ascontiguousarray(
        spline_coefficients(preprocessed, dtype=np.float32)
    )
    xi_x, xi_y = state.xi_x, state.xi_y
    offset_x = state.pc_pixels[0] - 0.5
    offset_y = state.pc_pixels[1] - 0.5
    matrix = matrix_of(h0)
    for _ in range(int(n_iterations)):
        warped_x, warped_y = project_with(matrix, xi_x, xi_y)
        values = evaluate(
            coefficients,
            np.ascontiguousarray(warped_x + offset_x),
            np.ascontiguousarray(warped_y + offset_y),
        )
        centred = values - values.mean()
        residuals = state.reference - centred / np.linalg.norm(centred)
        gradient = (2.0 / state.reference_norm) * (state.steepest_descent.T @ residuals)
        step = np.linalg.solve(state.hessian, -gradient)
        matrix = matrix @ np.linalg.inv(matrix_of(step))
        matrix = matrix / matrix[2, 2]
    return parameters_of(matrix)


@functools.lru_cache(maxsize=1)
def _update_rule_case():
    """Return ``(reference, target, h0, state, corners)`` of the (k)
    oracle, the recipe of the CPU ``UPDATE_RULE_TOL`` test: 480 px,
    seed 17 at half scale, ``h0`` off the answer by (0.3, -0.25) px."""
    detector = make_detector()
    pc_px = pc_pixels_of(detector)
    reference = project_pattern(detector, generic_rotation())
    h_true = random_small_homographies(n=1, dd=pc_px[2], seed=17, scale=0.5)[0]
    target = warp_with_skimage(reference, h_true, pc_px)
    h0 = h_true.copy()
    h0[2] += 0.3
    h0[5] -= 0.25
    state = make_state(reference, pc_px)
    corners = subregion_corners(SHAPE_480, pc_px)
    return _read_only(reference), _read_only(target), _read_only(h0), state, corners


@functools.lru_cache(maxsize=1)
def _intensity_case():
    """Return ``(reference, target, corners)`` of the (j) oracle, the
    recipe of the CPU ``test_intensity_scale_invariance``: 480 px, one
    random small homography of seed 7."""
    detector = make_detector()
    pc_px = pc_pixels_of(detector)
    reference = project_pattern(detector, generic_rotation())
    h_true = random_small_homographies(n=1, dd=pc_px[2], seed=7)[0]
    target = warp_with_skimage(reference, h_true, pc_px)
    corners = subregion_corners(SHAPE_480, pc_px)
    return _read_only(reference), _read_only(target), corners


def _two_point_run(reference, target, **kwargs) -> dict:
    """Return the engine on the two-point map ``[reference, target]``
    at reference ``(0, 0)``, one projection centre tiled."""
    detector = make_detector(navigation_shape=(1, 2))
    return run_engine(
        np.stack([reference, target]), (1, 2), detector, reference=(0, 0), **kwargs
    )


class _PoolHighWater:
    """Track the high-water mark of cupy's default memory pool through
    a recording allocator, relative to the pool's use on entry (the
    pool and the FFT plan cache are emptied first)."""

    def __init__(self, cupy):
        self.cupy = cupy
        self.pool = cupy.get_default_memory_pool()
        self.base = 0
        self.peak = 0

    def __enter__(self):
        self.pool.free_all_blocks()
        self.cupy.fft.config.get_plan_cache().clear()
        self.base = int(self.pool.used_bytes())
        self.peak = self.base

        def malloc(size):
            memory = self.pool.malloc(size)
            self.peak = max(self.peak, int(self.pool.used_bytes()))
            return memory

        self.cupy.cuda.set_allocator(malloc)
        return self

    def __exit__(self, *exc_info):
        self.cupy.cuda.set_allocator(self.pool.malloc)
        return False

    @property
    def high_water(self) -> int:
        return int(self.peak - self.base)


def _spy_raw_kernels(monkeypatch, cupy) -> list:
    """Replace ``cupy.RawKernel`` by a recording subclass; every launch
    appends ``(name, grid, block)`` as integer tuples.  Requires every
    kernel to be CONSTRUCTED through the attribute ``cupy.RawKernel``
    inside the call that uses it (``make_kernel_namespace``, the gate
    probe), with no ``cupy.RawModule`` and no Python-level kernel cache
    across calls: the call-time seam the module docstring freezes
    (critic finding F5)."""
    launches = []
    base = cupy.RawKernel

    def as_tuple(value):
        if isinstance(value, tuple | list):
            return tuple(int(v) for v in value)
        return (int(value),)

    class RecordingRawKernel(base):
        def __call__(self, grid, block, args, *rest, **kwargs):
            launches.append((self.name, as_tuple(grid), as_tuple(block)))
            return super().__call__(grid, block, args, *rest, **kwargs)

    monkeypatch.setattr(cupy, "RawKernel", RecordingRawKernel)
    return launches


def _per_pattern_layout(launches, batch_size) -> set:
    """Return ``{(name, block, grid blocks per pattern)}`` of recorded
    launches at *batch_size*, the per-pattern geometry of V9(m)."""
    return {
        (name, block, fractions.Fraction(int(np.prod(grid)), int(batch_size)))
        for name, grid, block in launches
    }


def _session_host_check(value, name) -> None:
    """Assert *value* is not a cupy object (D21.9.1)."""
    assert not type(value).__module__.startswith("cupy"), name


class TestGatedGate:
    """V9(c) gated: the REAL HREBSD gate on the pinned overlay of
    D21.15, its cache, its independence from the spherical gate's
    cache, the libraries stage (c) really probes on the device, and
    the engine reaching it once per call.

    Mutants: M21 and M22 (gated twins of the fake-cupy order pin and
    probe records; the designed killers are the default suite's), M26
    (the device runner really runs: no silent CPU fallback)."""

    def test_the_overlay_is_the_pinned_one(self, cupy_gpu, record_property):
        # every device-side pin is valid for EXACTLY these wheels
        # (D21.15): an unpinned overlay can move device-side bits
        for distribution, version in GATED_PINNED_OVERLAY.items():
            assert importlib.metadata.version(distribution) == version, distribution
        assert cupy_gpu.__version__ == GATED_PINNED_OVERLAY["cupy-cuda12x"]
        properties = cupy_gpu.cuda.runtime.getDeviceProperties(0)
        record_property("device", properties["name"].decode())
        record_property("driver", cupy_gpu.cuda.runtime.driverGetVersion())
        record_property("cuda_runtime", cupy_gpu.cuda.runtime.runtimeGetVersion())
        record_property("machine", MACHINE_A)

    def test_gate_passes_and_caches(self, cupy_gpu, fresh_gate, monkeypatch):
        _gpu._verify_gpu_or_raise()
        assert _gpu._gate_result is True
        # a cached verdict probes nothing: a second call never reaches
        # the device count again
        calls = []
        real_count = cupy_gpu.cuda.runtime.getDeviceCount

        def counting():
            calls.append(True)
            return real_count()

        monkeypatch.setattr(cupy_gpu.cuda.runtime, "getDeviceCount", counting)
        _gpu._verify_gpu_or_raise()
        assert calls == []

    def test_the_spherical_cache_is_untouched(self, cupy_gpu, fresh_gate, monkeypatch):
        # HREBSD's own process-global cache, never the spherical one
        # (D21.2), in both directions; the spherical verdict is
        # restored after the test
        monkeypatch.setattr(_spherical_gpu, "_gate_result", _spherical_gpu._gate_result)
        before = _spherical_gpu._gate_result
        _gpu._verify_gpu_or_raise()
        assert _spherical_gpu._gate_result is before
        hrebsd = _gpu._gate_result
        _spherical_gpu._verify_gpu_or_raise()
        assert _gpu._gate_result is hrebsd

    def test_stage_c_probes_every_library_on_the_device(
        self, cupy_gpu, fresh_gate, monkeypatch
    ):
        # the gated twin of the fake-cupy record (M22): on the REAL
        # cupy, one complex128 and one complex64 FFT, one matmul and
        # one RawKernel compiled and launched
        records = []
        for name in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"):
            real = getattr(cupy_gpu.fft, name)

            def recording(array, *args, _real=real, _name=name, **kwargs):
                records.append(("fft", _name, np.dtype(array.dtype).name))
                return _real(array, *args, **kwargs)

            monkeypatch.setattr(cupy_gpu.fft, name, recording)
        real_matmul = cupy_gpu.matmul

        def recording_matmul(*args, **kwargs):
            records.append(("matmul",))
            return real_matmul(*args, **kwargs)

        monkeypatch.setattr(cupy_gpu, "matmul", recording_matmul)
        launches = _spy_raw_kernels(monkeypatch, cupy_gpu)
        _gpu._verify_gpu_or_raise()
        fft_dtypes = {r[2] for r in records if r[0] == "fft"}
        assert {"complex128", "complex64"} <= fft_dtypes
        assert ("matmul",) in records
        assert len(launches) >= 1

    def test_the_engine_reaches_the_gate_once_per_call(self, cupy_gpu, monkeypatch):
        calls = []
        real = _engine._verify_gpu_or_raise

        def counting():
            calls.append(True)
            return real()

        monkeypatch.setattr(_engine, "_verify_gpu_or_raise", counting)
        run_fixture(f4_batch(), **_gpu_kwargs())
        assert len(calls) == 1

    def test_the_device_runner_is_what_runs(self, cupy_gpu, monkeypatch):
        # M26: under "gpu" the CPU chunk runner is never called and a
        # cupy session is built
        cpu_calls = []

        def forbidden(*args, **kwargs):
            cpu_calls.append(True)
            raise AssertionError("backend='gpu' reached the CPU _run_chunks")

        monkeypatch.setattr(_engine, "_run_chunks", forbidden)
        namespaces = []
        real = _gpu._make_session

        def recording(namespace, batch_size, **kwargs):
            namespaces.append(namespace)
            return real(namespace, batch_size, **kwargs)

        monkeypatch.setattr(_gpu, "_make_session", recording)
        properties = run_fixture(f4_batch(), **_gpu_kwargs())
        assert cpu_calls == []
        assert namespaces and set(namespaces) == {"cupy"}
        assert np.asarray(properties["converged"]).any()


class TestGatedSeedParity:
    """V9(d) gated (D21.5, D21.8(b)): the device output contract of
    the seam (Johan decision 1's shape and dtype test on the device
    itself), complex128 device seeds equal ``initial_guess`` row for
    row on F1 to F4 at ``upsample_factor`` 16, 2 and 1 and non-finite
    throughout at 0, the dimmed arm, complex64 differences counted,
    and the runner reaching the seam through the module global once
    per padded sub-batch with a truthful ``SeedBatch``.

    Mutants: M11 (module-global spy), M12 and M13 (equality), M14
    (dtype asserts, seed count), M15 (dimmed arm), M47 (the 2 and 1
    arms), M50 (tail sub-batch padded to P, spy), M52 (``SeedBatch``
    ``pattern_index`` spy), M53 (spy count)."""

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    def test_device_output_contract(self, cupy_gpu, seed_precision):
        cp = cupy_gpu
        fixture = f1_batch()
        state = fixture_state(fixture)
        preprocessed, coefficients = preprocessed_targets(state, fixture["targets"])
        seed_state, batch, spectra, h0 = _device_seed(
            cp, state, preprocessed, coefficients, precision=seed_precision
        )
        r0, r1, c0, c1 = state.bounds
        n = len(preprocessed)
        assert seed_state.precision == seed_precision
        assert isinstance(seed_state.reference_spectrum, cp.ndarray)
        assert seed_state.reference_spectrum.dtype == np.dtype(seed_precision)
        assert tuple(seed_state.reference_spectrum.shape) == (r1 - r0, c1 - c0)
        assert isinstance(spectra, cp.ndarray)
        assert spectra.dtype == np.dtype(seed_precision)
        assert tuple(spectra.shape) == (n, r1 - r0, c1 - c0)
        assert isinstance(h0, cp.ndarray)
        assert h0.dtype == np.float64
        assert h0.flags.c_contiguous
        assert tuple(h0.shape) == (n, N_HOMOGRAPHY_PARAMETERS)
        host = cp.asnumpy(h0)
        assert np.all(host[:, [0, 1, 3, 4, 6, 7]] == 0.0)

    @pytest.mark.parametrize("upsample_factor", [16, 2, 1])
    @pytest.mark.parametrize("name", GATED_SEED_FIXTURES)
    def test_complex128_seeds_equal_initial_guess(
        self, cupy_gpu, name, upsample_factor, record_property
    ):
        fixture = fixture_of(name)
        state = fixture_state(fixture)
        targets = np.asarray(fixture["targets"])
        preprocessed, coefficients = preprocessed_targets(state, targets)
        *_, h0 = _device_seed(
            cupy_gpu,
            state,
            preprocessed,
            coefficients,
            upsample_factor=upsample_factor,
        )
        expected = cpu_seed_rows(state, targets, upsample_factor=upsample_factor)
        equal = _equal_rows(cupy_gpu.asnumpy(h0), expected)
        record_property("equal", equal)
        record_property("compared", len(targets))
        assert_count(
            equal,
            _pin(GPU_SEED_EQUAL_COUNT, (name, upsample_factor)),
            f"GPU_SEED_EQUAL_COUNT[{name!r}, {upsample_factor}]",
        )

    @pytest.mark.parametrize("upsample_factor", [0, -3])
    def test_upsample_below_one_fails_every_pattern(self, cupy_gpu, upsample_factor):
        # the CPU fails EVERY pattern there (D21.5); the device
        # reproduces that outcome with no new error path
        fixture = f4_batch()
        state = fixture_state(fixture)
        targets = np.asarray(fixture["targets"])
        preprocessed, coefficients = preprocessed_targets(state, targets)
        assert (
            not np.isfinite(
                cpu_seed_rows(state, targets, upsample_factor=upsample_factor)
            )
            .all(axis=1)
            .any()
        )
        *_, h0 = _device_seed(
            cupy_gpu,
            state,
            preprocessed,
            coefficients,
            upsample_factor=upsample_factor,
        )
        rows = cupy_gpu.asnumpy(h0)
        assert not np.isfinite(rows).all(axis=1).any()

    def test_dimmed_arm_still_equals_initial_guess(self, cupy_gpu):
        # M15, the default arm's recipe (critic finding F3): the RAW
        # reference and targets dimmed by DIM_SCALE (1e-10 was measured
        # NOT to reach the floor), then preprocessed; at that scale an
        # un-normalised cross-power spectrum sits wholly below the
        # ``100 * eps`` floor (premise asserted), and the CPU seeds equal
        # the undimmed ones (premise asserted), so only crops ZMN'd
        # before the FFT keep the CPU's rows on the device
        fixture = f1_batch()
        state = make_state(
            np.asarray(fixture["reference"]) * DIM_SCALE, fixture["pc_px"]
        )
        targets = np.asarray(fixture["targets"]) * DIM_SCALE
        preprocessed, coefficients = preprocessed_targets(state, targets)
        r0, r1, c0, c1 = state.bounds
        cross = np.fft.fft2(state.reference_subregion) * np.conj(
            np.fft.fft2(preprocessed[0, r0:r1, c0:c1])
        )
        assert np.abs(cross).max() < 100 * np.finfo(np.float64).eps
        expected = cpu_seed_rows(state, targets)
        undimmed = cpu_seed_rows(fixture_state(fixture), fixture["targets"])
        assert np.array_equal(expected, undimmed)
        *_, h0 = _device_seed(cupy_gpu, state, preprocessed, coefficients)
        assert_count(
            _equal_rows(cupy_gpu.asnumpy(h0), expected),
            _pin(GPU_SEED_EQUAL_COUNT, ("F1 dimmed", 16)),
            "GPU_SEED_EQUAL_COUNT['F1 dimmed', 16]",
        )

    @pytest.mark.parametrize("name", GATED_SEED_FIXTURES)
    def test_complex64_differences_are_counted(self, cupy_gpu, name, record_property):
        fixture = fixture_of(name)
        state = fixture_state(fixture)
        targets = np.asarray(fixture["targets"])
        preprocessed, coefficients = preprocessed_targets(state, targets)
        *_, h0 = _device_seed(
            cupy_gpu, state, preprocessed, coefficients, precision="complex64"
        )
        rows = cupy_gpu.asnumpy(h0)
        expected = cpu_seed_rows(state, targets)
        different = len(targets) - _equal_rows(rows, expected)
        record_property("different", different)
        record_property(
            "max_shift_difference_px",
            float(np.nanmax(np.abs(rows - expected), initial=0.0)),
        )
        assert_count(
            different,
            _pin(GPU_SEED_C64_DIFF_COUNT, name),
            f"GPU_SEED_C64_DIFF_COUNT[{name!r}]",
        )

    def test_the_runner_reaches_the_seam_per_padded_sub_batch(
        self, cupy_gpu, monkeypatch
    ):
        # M11, M50, M52, M53: B = 40 on F1's 13 points is one batch of
        # 40 slots, so P = 32 and ceil(40 / 32) = 2 sub-batches of 32
        # rows each, the tail padded; slots in fit order, -1 padded
        cp = cupy_gpu
        records = spy_seam(monkeypatch, keep_states=False)
        fixture = f1_batch()
        run_fixture(fixture, chunksize=40, **_gpu_kwargs())
        p = FROZEN_SUB_BATCH_SIZE
        seeds_calls = [r for r in records if r["call"] == "seed_homographies"]
        spectra_calls = [r for r in records if r["call"] == "seed_spectra"]
        assert len(seeds_calls) == 2
        assert len(spectra_calls) == 2
        n_points = int(np.prod(fixture["navigation_shape"]))
        index = np.concatenate([call["pattern_index"] for call in seeds_calls])
        expected_index = np.concatenate(
            [np.arange(n_points), np.full(2 * p - n_points, -1)]
        )
        assert np.array_equal(index, expected_index)
        state = fixture_state(fixture)
        nrows, ncols = fixture["reference"].shape
        r0, r1, c0, c1 = state.bounds
        for spectra in spectra_calls:
            assert issubclass(spectra["type"], cp.ndarray)
            assert spectra["shape"] == (p, r1 - r0, c1 - c0)
            assert spectra["dtype"] == np.complex128
        for call in seeds_calls:
            assert call["ctx_xp"] is cp
            assert isinstance(call["kernels"], _batched.KernelNamespace)
            assert call["kernels"].precision == "mixed"
            assert call["pattern_index_dtype"] == np.int64
            assert issubclass(call["targets_type"], cp.ndarray)
            assert call["targets_shape"] == (p, nrows, ncols)
            assert call["targets_dtype"] == np.float64
            assert call["coefficients_shape"] == (p, nrows, ncols)
            assert call["coefficients_dtype"] == np.float32
            assert call["extras"] == {}
            assert issubclass(call["type"], cp.ndarray)
            assert call["dtype"] == np.float64
            assert call["c_contiguous"]
            assert call["h0"].shape == (p, N_HOMOGRAPHY_PARAMETERS)
        seeds = np.concatenate([call["h0"] for call in seeds_calls])
        real = index >= 0
        expected = cpu_seed_rows(state, np.asarray(fixture["patterns"]))
        assert_count(
            _equal_rows(seeds[real], expected[index[real]]),
            _pin(GPU_SEED_EQUAL_COUNT, ("F1 runner", 16)),
            "GPU_SEED_EQUAL_COUNT['F1 runner', 16]",
        )


class TestGatedSeedSeam:
    """V9(e) gated, Johan decision 1 (D21.5(iii)): the seed stage is
    monkeypatched through its module global to return ARBITRARY
    per-pattern rows inside the measured capture range, and the device
    honours every one, against the CPU ``fit_pattern(state, target,
    h0=row)`` on F1 to F3 at both device precisions.  Bands: the (f)
    ``h`` band on points BOTH backends converge; iteration and
    convergence differences as COUNT budgets; a pinned minimum
    both-converged count per row type; the NaN row's failure contract
    exactly; the ONE-ITERATION arm within the (f) first-step band; and
    the CPU-side discriminating count.

    The planted rows and the CPU oracle are the default suite's
    (:func:`seam_table`, :func:`seam_cpu`: every map point, the
    reference included).  Pin keys: ``(fixture, device_precision,
    row_type)`` for the ``GPU_SEAM_*`` tables, ``(fixture, row_type)``
    for the CPU-side :data:`SEAM_DISCRIMINATING_MIN` both suites read.

    Mutants: M10 (the seam's rows ignored and the seed recomputed: the
    one-iteration arm and the discriminating count, its designed
    killers), M6 (counts)."""

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", _weekly_if_large(["F1", "F2-0", "F2-1", "F3"]))
    def test_planted_rows_are_honoured(
        self, cupy_gpu, monkeypatch, name, device_precision, row_type, record_property
    ):
        fixture = fixture_of(name)
        calls = plant_table(monkeypatch, seam_table(name, row_type))
        gpu = packed(run_fixture(fixture, **_gpu_kwargs(device_precision)))
        assert calls, "the device runner never called the seed seam"
        cpu = seam_cpu(name, row_type, DEFAULT_FIT_OPTIONS["max_iterations"])
        both = (cpu[:, 11] > 0.5) & (gpu[:, 11] > 0.5)
        key = (name, device_precision, row_type)
        record_property("both_converged", int(both.sum()))
        assert_at_least(
            int(both.sum()),
            _pin(GPU_SEAM_BOTH_CONVERGED_MIN, key),
            f"GPU_SEAM_BOTH_CONVERGED_MIN{key}",
        )
        assert_within(
            h_band(cpu[:, :8], gpu[:, :8], fixture["corners"], both),
            _h_tol(device_precision),
            f"seam h band {key}",
        )
        assert_count(
            int((cpu[:, 9] != gpu[:, 9]).sum()),
            _pin(GPU_SEAM_ITERATION_DIFF_COUNT, key),
            f"GPU_SEAM_ITERATION_DIFF_COUNT{key}",
        )
        assert_count(
            int(((cpu[:, 11] > 0.5) != (gpu[:, 11] > 0.5)).sum()),
            _pin(GPU_SEAM_CONVERGED_FLIP_COUNT, key),
            f"GPU_SEAM_CONVERGED_FLIP_COUNT{key}",
        )

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", _weekly_if_large(["F1", "F2-0", "F2-1", "F3"]))
    def test_one_iteration_arm(
        self, cupy_gpu, monkeypatch, name, device_precision, row_type
    ):
        # the iterate after one step is h0 composed with one update,
        # so a device that recomputes or ignores the rows fails here
        # at once (M10's designed killer)
        fixture = fixture_of(name)
        plant_table(monkeypatch, seam_table(name, row_type))
        gpu = packed(
            run_fixture(fixture, max_iterations=1, **_gpu_kwargs(device_precision))
        )
        cpu = seam_cpu(name, row_type, 1)
        assert np.array_equal(gpu[:, 9], cpu[:, 9])
        finite = np.isfinite(cpu[:, :8]).all(axis=1)
        assert finite.all()
        assert_within(
            h_band(cpu[:, :8], gpu[:, :8], fixture["corners"], finite),
            _first_step_tol(device_precision),
            f"seam first step {(name, device_precision, row_type)}",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", _weekly_if_large(["F1", "F2-0", "F2-1", "F3"]))
    def test_nan_row_gives_the_failure_contract(
        self, cupy_gpu, monkeypatch, name, device_precision
    ):
        fixture = fixture_of(name)
        plant_table(monkeypatch, seam_table(name, "nan"))
        gpu = run_fixture(fixture, **_gpu_kwargs(device_precision))
        cpu = seam_cpu(name, "nan", DEFAULT_FIT_OPTIONS["max_iterations"])
        # the CPU premise: a NaN h0 is the D2.6 failure contract there
        assert np.all(np.isnan(cpu[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]))
        assert np.all(cpu[:, 9] == 0) and np.all(cpu[:, 11] == 0)
        for index in range(cpu.shape[0]):
            assert_failure_contract(gpu, index)

    @pytest.mark.parametrize("row_type", SEAM_FINITE_ROW_TYPES)
    @pytest.mark.parametrize("name", _weekly_if_large(["F1", "F2-0", "F2-1", "F3"]))
    def test_planted_rows_discriminate(self, cupy_gpu, name, row_type, record_property):
        # the CPU-side half of the discriminating arm: the planted row
        # changes the CPU's own iteration count on enough patterns
        # that a device ignoring it could not reproduce the counts the
        # honoured-rows test pins (M10)
        planted = seam_cpu(name, row_type, DEFAULT_FIT_OPTIONS["max_iterations"])
        translation = seam_cpu(name, None, DEFAULT_FIT_OPTIONS["max_iterations"])
        discriminating = int((planted[:, 9] != translation[:, 9]).sum())
        record_property("discriminating", discriminating)
        assert_at_least(
            discriminating,
            _pin(SEAM_DISCRIMINATING_MIN, (name, row_type)),
            f"SEAM_DISCRIMINATING_MIN{(name, row_type)}",
        )


class TestGatedParity:
    """V9(f) and (g) gated (D21.8(a) to (e)): per-point parity of the
    device against the CPU on F1 to F4 and the F6 pin map, at each
    device precision and each seed precision; the wiring probes
    bitwise; the ``h`` band on points both converge; the first-step
    band on the iteration-1 increment; the residual band; Fe through
    the shared host conversion; iteration and convergence COUNTS
    pinned per ``(fixture, device_precision, seed_precision)``; and
    the shipped Ni map through the public method.

    Mutants: M3 (first-step band, its designed killer), M4, M30, M32,
    M45 (float64 band), M6 and M31 (counts, ``h`` band), M37, M38, M39
    (first-step band), M42 (first-step and residual bands)."""

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", GATED_PARITY_FIXTURES)
    def test_converged_parity(
        self, cupy_gpu, name, device_precision, seed_precision, record_property
    ):
        fixture = fixture_of(name)
        cpu = cpu_run(name)
        gpu = run_fixture(fixture, **_gpu_kwargs(device_precision, seed_precision))
        _assert_host_outputs_bitwise(cpu, gpu)
        key = (name, device_precision, seed_precision)
        both = _both_converged(cpu, gpu)
        assert both.any()
        band = h_band(cpu["homography"], gpu["homography"], fixture["corners"], both)
        record_property("h_band_px", band)
        assert_within(band, _h_tol(device_precision), f"h band {key}")
        assert_device_residual_band(
            np.asarray(cpu["residual"])[both],
            np.asarray(gpu["residual"])[both],
            f"residual {key}",
        )
        fe_difference = float(
            np.abs(np.asarray(gpu["Fe"])[both] - np.asarray(cpu["Fe"])[both]).max()
        )
        assert_within(fe_difference, GPU_PARITY_FE_TOL, f"GPU_PARITY_FE_TOL {key}")
        iterations, flips = _count_differences(cpu, gpu, _fitted(cpu))
        record_property("iteration_differences", iterations)
        record_property("converged_flips", flips)
        assert_count(
            iterations,
            _pin(GPU_ITERATION_DIFF_COUNT, key),
            f"GPU_ITERATION_DIFF_COUNT{key}",
        )
        assert_count(
            flips,
            _pin(GPU_CONVERGED_FLIP_COUNT, key),
            f"GPU_CONVERGED_FLIP_COUNT{key}",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", GATED_PARITY_FIXTURES)
    def test_first_step_band(self, cupy_gpu, name, device_precision, record_property):
        fixture = fixture_of(name)
        cpu = cpu_run(name, max_iterations=1)
        gpu = run_fixture(fixture, max_iterations=1, **_gpu_kwargs(device_precision))
        assert np.array_equal(cpu["num_iterations"], gpu["num_iterations"])
        finite = np.isfinite(np.asarray(cpu["homography"])).all(axis=1)
        band = h_band(cpu["homography"], gpu["homography"], fixture["corners"], finite)
        record_property("first_step_px", band)
        assert_within(
            band,
            _first_step_tol(device_precision),
            f"first-step band {(name, device_precision)}",
        )

    def test_ni_map_through_the_public_method(self, cupy_gpu, record_property):
        # EBSD.hrebsd_dic(backend="gpu") end to end on the shipped Ni
        # map, at the public default device precision ("mixed")
        cpu = run_ni()
        gpu = run_ni(backend="gpu")
        _, _, detector = ni_inputs()
        corners = subregion_corners(SHAPE_60, pc_pixels_of(detector))
        assert set(cpu.prop.keys()) == set(gpu.prop.keys())
        for name in cpu.prop.keys():
            a = np.asarray(cpu.prop[name])
            b = np.asarray(gpu.prop[name])
            assert a.dtype == b.dtype and a.shape == b.shape, name
        for name in ("grain_id", "reference_index"):
            assert np.array_equal(cpu.prop[name], gpu.prop[name]), name
        both = np.asarray(cpu.prop["converged"]) & np.asarray(gpu.prop["converged"])
        assert both.any()
        assert_within(
            h_band(cpu.prop["homography"], gpu.prop["homography"], corners, both),
            GPU_PARITY_H_TOL_MIXED,
            "GPU_PARITY_H_TOL_MIXED (Ni map)",
        )
        cpu_props = {n: np.asarray(cpu.prop[n]) for n in cpu.prop.keys()}
        gpu_props = {n: np.asarray(gpu.prop[n]) for n in gpu.prop.keys()}
        iterations, flips = _count_differences(cpu_props, gpu_props, _fitted(cpu_props))
        key = ("Ni", "mixed", "complex128")
        assert_count(
            iterations,
            _pin(GPU_ITERATION_DIFF_COUNT, key),
            f"GPU_ITERATION_DIFF_COUNT{key}",
        )
        assert_count(
            flips,
            _pin(GPU_CONVERGED_FLIP_COUNT, key),
            f"GPU_CONVERGED_FLIP_COUNT{key}",
        )


class TestGatedKernelAB:
    """V9(f) per-kernel A/B (D21.14.3): each ``KernelNamespace`` entry
    point run under the numpy twin and under cupy at ``"float64"`` on
    IDENTICAL inputs, within ``GPU_KERNEL_AB_TOL_F64`` as the scale-free
    :func:`_relative_difference`; the twin is the f64 build only, so
    the mixed build is pinned by its output dtypes here and by the
    parity classes.

    Mutants: M3, M4, M30, M32, M37, M38, M39, M45 (per-kernel
    attribution of the parity failures), M44 (gather on far
    coordinates, also in :class:`TestGatedBatchedSemantics`)."""

    def _residents(self, cupy, state):
        np_ctx = numpy_seed_context("float64")
        cp_ctx = _device_context(cupy, "float64")
        return (
            np_ctx,
            _batched.build_resident(np_ctx, state),
            cp_ctx,
            _batched.build_resident(cp_ctx, state),
        )

    def test_gather(self, cupy_gpu):
        cp = cupy_gpu
        state, coefficients, matrices = _ab_inputs()
        np_ctx, np_resident, cp_ctx, cp_resident = self._residents(cp, state)
        values_np, ok_np = np_ctx.kernels.gather(np_resident, coefficients, matrices)
        values_cp, ok_cp = cp_ctx.kernels.gather(
            cp_resident, cp.asarray(coefficients), cp.asarray(matrices)
        )
        assert isinstance(values_cp, cp.ndarray)
        assert values_cp.dtype == np.float64
        assert tuple(values_cp.shape) == (len(matrices), state.n_pixels)
        assert np.array_equal(cp.asnumpy(ok_cp), np.asarray(ok_np))
        assert_within(
            _relative_difference(values_np, cp.asnumpy(values_cp)),
            GPU_KERNEL_AB_TOL_F64,
            "GPU_KERNEL_AB_TOL_F64 (gather)",
        )

    def test_pixel_sums_and_final_criterion(self, cupy_gpu):
        cp = cupy_gpu
        state, coefficients, matrices = _ab_inputs()
        np_ctx, np_resident, cp_ctx, cp_resident = self._residents(cp, state)
        values, _ = np_ctx.kernels.gather(np_resident, coefficients, matrices)
        values = np.asarray(values, dtype=np.float64)
        shifts = values.mean(axis=1)
        sums_np = np_ctx.kernels.pixel_sums(np_resident, values, shifts)
        sums_cp = cp_ctx.kernels.pixel_sums(
            cp_resident, cp.asarray(values), cp.asarray(shifts)
        )
        assert sums_cp.dtype == np.float64
        assert tuple(sums_cp.shape) == tuple(np.asarray(sums_np).shape)
        assert_within(
            _relative_difference(sums_np, cp.asnumpy(sums_cp)),
            GPU_KERNEL_AB_TOL_F64,
            "GPU_KERNEL_AB_TOL_F64 (pixel_sums)",
        )
        criterion_np = np_ctx.kernels.final_criterion(np_resident, values, shifts)
        criterion_cp = cp_ctx.kernels.final_criterion(
            cp_resident, cp.asarray(values), cp.asarray(shifts)
        )
        assert criterion_cp.dtype == np.float64
        assert tuple(criterion_cp.shape) == (len(matrices),)
        assert_within(
            _relative_difference(criterion_np, cp.asnumpy(criterion_cp)),
            GPU_KERNEL_AB_TOL_F64,
            "GPU_KERNEL_AB_TOL_F64 (final_criterion)",
        )

    def test_reduce_solve_update(self, cupy_gpu):
        cp = cupy_gpu
        state, coefficients, matrices = _ab_inputs()
        np_ctx, np_resident, cp_ctx, cp_resident = self._residents(cp, state)
        values, _ = np_ctx.kernels.gather(np_resident, coefficients, matrices)
        values = np.asarray(values, dtype=np.float64)
        shifts = values.mean(axis=1)
        sums = np.asarray(np_ctx.kernels.pixel_sums(np_resident, values, shifts))
        options = {
            "min_step": DEFAULT_FIT_OPTIONS["min_step"],
            "step_scale": DEFAULT_FIT_OPTIONS["step_scale"],
            "max_iterations": DEFAULT_FIT_OPTIONS["max_iterations"],
        }
        lockstep_np = _lockstep(np, matrices)
        lockstep_np.shifts = shifts.copy()
        lockstep_cp = _lockstep(cp, matrices)
        lockstep_cp.shifts = cp.asarray(shifts)
        np_ctx.kernels.reduce_solve_update(np_resident, sums, lockstep_np, options)
        cp_ctx.kernels.reduce_solve_update(
            cp_resident, cp.asarray(sums), lockstep_cp, options
        )
        device = _lockstep_to_host(cp, lockstep_cp)
        for name in ("iterations", "active", "converged", "failed"):
            assert np.array_equal(device[name], np.asarray(getattr(lockstep_np, name)))
        # the update really moved every matrix
        assert not np.allclose(device["matrices"], matrices, rtol=0.0, atol=1e-12)
        for name in ("matrices", "shifts", "norm_dp"):
            assert_within(
                _relative_difference(getattr(lockstep_np, name), device[name]),
                GPU_KERNEL_AB_TOL_F64,
                f"GPU_KERNEL_AB_TOL_F64 (reduce_solve_update, {name})",
            )

    def test_the_mixed_build_dtypes(self, cupy_gpu):
        # D21.4: f32 per-pixel values, f64 sums, f64 criterion
        cp = cupy_gpu
        state, coefficients, matrices = _ab_inputs()
        ctx = _device_context(cp, "mixed")
        assert ctx.kernels.precision == "mixed"
        assert ctx.kernels.out_of_memory_error is cp.cuda.memory.OutOfMemoryError
        resident = _batched.build_resident(ctx, state)
        values, ok = ctx.kernels.gather(
            resident, cp.asarray(coefficients), cp.asarray(matrices)
        )
        assert values.dtype == np.float32
        assert ok.dtype == np.bool_
        shifts = cp.asarray(np.round(cp.asnumpy(values).mean(axis=1)))
        sums = ctx.kernels.pixel_sums(resident, values, shifts)
        assert sums.dtype == np.float64
        criterion = ctx.kernels.final_criterion(resident, values, shifts)
        assert criterion.dtype == np.float64


class TestGatedBatchedSemantics:
    """V9(h) and (i) gated, at both device precisions (D21.6): every
    pattern's result equals the same pattern fitted ALONE in a padded
    batch of the same B, bitwise (F5 plus an easy pattern, at most 10
    iterations; and a slot of the LAST sub-batch at B = 40, P = 32);
    the residual is the criterion at the returned ``h``; every knob
    arm against the CPU on F4; the ``max_iterations`` 0 and 1 edges;
    padded slots never reach the output; the failure contract and the
    map order on the two-grain F5 and the 18-grain F7; and the
    kernel-level coordinate and NaN arms on the device kernel.

    Knob-arm count pins are keyed ``("F4 <arm>", device_precision,
    "complex128")`` in ``GPU_ITERATION_DIFF_COUNT`` and
    ``GPU_CONVERGED_FLIP_COUNT``.

    Mutants: M5 (alone against batched), M7 (residual is the final
    criterion), M8 (non-finite coordinate, F5 non-finite pixel), M9
    (the integer-constant target), M16 (padding), M17 and M18 (F5 and
    F7 map order and parity, the resident identity spy), M52 (the F7
    seam's ``pattern_index`` in grain-ordered fit order), M28
    (``step_scale`` arms), M29, M40, M41 (window arm), M33 (the 0 and
    1 arms, capped count), M42 and M46 (the ``(None, None)`` DC arm
    and the low-pass arm), M43 (the NaN step's corner norm), M44 (the
    planted far coordinates), M47 (upsample 8 arm), M48 (``min_step``
    arm), M49 (``border`` arm), M50 (the last-sub-batch slot)."""

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_alone_equals_batched_on_f5(self, cupy_gpu, device_precision):
        patterns, states, options = _f5_runner_inputs()
        batch_size = 8
        batched = run_direct(
            patterns,
            F5_FIT_INDICES,
            F5_STATE_OF_POINT,
            states,
            batch_size,
            device_precision=device_precision,
            **options,
        )
        assert batched.shape == (F5_FIT_INDICES.size, 12)
        for k in range(F5_FIT_INDICES.size):
            alone = run_direct(
                patterns,
                F5_FIT_INDICES[k : k + 1],
                F5_STATE_OF_POINT[k : k + 1],
                states,
                batch_size,
                device_precision=device_precision,
                **options,
            )
            assert alone.shape == (1, 12)
            assert np.array_equal(alone[0], batched[k], equal_nan=True), k

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_last_sub_batch_slot_equals_alone(self, cupy_gpu, device_precision):
        # B = 40, P = 32: slot 35 sits in the tail sub-batch, padded
        # to P (D21.7.3, M50); slot 3 in the first one
        fixture = f1_batch()
        targets = np.asarray(fixture["targets"])
        tiled = targets[np.arange(40) % len(targets)]
        patterns = np.concatenate([np.asarray(fixture["reference"])[None], tiled])
        states = [fixture_state(fixture)]
        fit_indices = np.arange(1, 41)
        state_of_point = np.zeros(40, dtype=np.int64)
        batched = run_direct(
            patterns,
            fit_indices,
            state_of_point,
            states,
            40,
            device_precision=device_precision,
            **DEFAULT_FIT_OPTIONS,
        )
        for slot in (3, 35):
            alone = run_direct(
                patterns,
                fit_indices[slot : slot + 1],
                state_of_point[slot : slot + 1],
                states,
                40,
                device_precision=device_precision,
                **DEFAULT_FIT_OPTIONS,
            )
            assert np.array_equal(alone[0], batched[slot], equal_nan=True), slot

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_padded_slots_never_reach_the_output(self, cupy_gpu, device_precision):
        # M16: F4's four points in one batch of 8 (four padded slots)
        fixture = f4_batch()
        state = fixture_state(fixture)
        patterns = np.asarray(fixture["patterns"])
        rows = run_direct(
            patterns,
            np.arange(4),
            np.zeros(4, dtype=np.int64),
            [state],
            8,
            device_precision=device_precision,
            **DEFAULT_FIT_OPTIONS,
        )
        assert rows.shape == (4, 12)
        cpu = cpu_packed(state, patterns)
        both = (cpu[:, 11] > 0.5) & (rows[:, 11] > 0.5)
        assert both.all()
        assert_within(
            h_band(cpu[:, :8], rows[:, :8], fixture["corners"], both),
            _h_tol(device_precision),
            f"padded F4 h band ({device_precision})",
        )
        # and the masked point of the map form keeps the NaN row
        mask = np.zeros(fixture["navigation_shape"], dtype=bool)
        mask.flat[2] = True
        properties = run_fixture(
            fixture, navigation_mask=mask, chunksize=8, **_gpu_kwargs(device_precision)
        )
        assert np.all(np.isnan(np.asarray(properties["homography"])[2]))
        assert int(properties["num_iterations"][2]) == 0
        assert not properties["converged"][2]

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_f5_failure_contract_and_map_order(
        self, cupy_gpu, monkeypatch, device_precision
    ):
        # M8, M9, M17, M18 and M33's capped count, on the two-grain map
        # at the production (None, None) route
        residents = []

        def wrap_gather(attempt, session):
            real_gather = session.kernels.gather

            def spying_gather(resident, coefficients, matrices):
                residents.append(resident)
                return real_gather(resident, coefficients, matrices)

            session.kernels.gather = spying_gather

        _spy_sessions(monkeypatch, on_session=wrap_gather)
        fixture = f5_map()
        cpu = cpu_run("F5")
        gpu = run_fixture(fixture, **_gpu_kwargs(device_precision))
        _assert_host_outputs_bitwise(cpu, gpu)
        # the premises the fixture docstring measured, on the CPU
        for index in (0, F5_EASY_A, 4, F5_EASY_B):
            assert cpu["converged"][index], index
        for index in (F5_CONSTANT, F5_NON_FINITE):
            assert_failure_contract(cpu, index)
        # the masked point: NaN props, truthful grain and reference
        assert np.all(np.isnan(np.asarray(gpu["homography"])[F5_MASKED]))
        assert np.isnan(gpu["residual"][F5_MASKED])
        assert int(gpu["num_iterations"][F5_MASKED]) == 0
        assert not gpu["converged"][F5_MASKED]
        # the integer constant (the SEED's crop ZMN) and the
        # non-finite pixel: the D2.6 failure contract
        for index in (F5_CONSTANT, F5_NON_FINITE):
            assert_failure_contract(gpu, index)
        # the capped point keeps a finite last iterate
        assert np.all(np.isfinite(np.asarray(gpu["homography"])[F5_CAPPED]))
        assert not gpu["converged"][F5_CAPPED]
        assert np.all(np.isnan(np.asarray(gpu["Fe"])[F5_CAPPED]))
        assert int(gpu["num_iterations"][F5_CAPPED]) == int(
            cpu["num_iterations"][F5_CAPPED]
        )
        assert (
            int(gpu["num_iterations"][F5_CAPPED])
            == DEFAULT_FIT_OPTIONS["max_iterations"]
        )
        # map order and the right grain's residents: each converged
        # point within the h band of the CPU, grain B's against B
        for index in (0, F5_EASY_A, 4, F5_EASY_B):
            assert gpu["converged"][index], index
        both = _both_converged(cpu, gpu)
        assert_within(
            h_band(cpu["homography"], gpu["homography"], fixture["corners"], both),
            _h_tol(device_precision),
            f"F5 h band ({device_precision})",
        )
        assert len({id(r) for r in residents}) == len(F5_REFERENCES)

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_user_warning_count_equals_the_cpu(self, cupy_gpu, device_precision):
        fixture = f5_map()
        counts = {}
        for backend, extra in (("cpu", {}), ("gpu", _gpu_kwargs(device_precision))):
            extra = dict(extra)
            extra.setdefault("backend", backend)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_hrebsd_dic(
                    np.asarray(fixture["patterns"]),
                    fixture["navigation_shape"],
                    fixture["detector"],
                    reference=np.asarray(fixture["references"]),
                    grain_labels=np.asarray(fixture["grain_labels"]),
                    navigation_mask=np.array(fixture["navigation_mask"]),
                    filter_cutoffs=F5_FILTER_CUTOFFS,
                    verbose=0,
                    **extra,
                )
            counts[backend] = sum(issubclass(w.category, UserWarning) for w in caught)
        assert counts["gpu"] == counts["cpu"]
        assert counts["cpu"] >= 1

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_band_passed_constant_is_not_converged(self, cupy_gpu, device_precision):
        # the default cutoffs: the CPU fails it only in the criterion
        # after the f32 cast (D21.6.2), so converged=False and nothing
        # more on both backends
        fixture = f5_map()
        cpu = cpu_run("F5", filter_cutoffs=(0.05, None))
        gpu = run_fixture(
            fixture, filter_cutoffs=(0.05, None), **_gpu_kwargs(device_precision)
        )
        assert not cpu["converged"][F5_CONSTANT]
        assert not gpu["converged"][F5_CONSTANT]

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_residual_is_the_final_criterion(self, cupy_gpu, device_precision):
        # M7: on the capped point the criterion at the returned h and
        # the one before the last composition differ (MEASURED
        # 2026-10-06 on the CPU: 1.7770157 against 1.7769830, 1.8e-5
        # relative, about 70x the mixed residual band's spec-gate
        # scale); the device must report the former, so it must sit
        # closer to the final criterion than to the pre-update one
        fixture = f5_map()
        patterns = np.asarray(fixture["patterns"])
        pc_px = pc_pixels_of(fixture["detector"])
        state_b = make_state(
            patterns[int(F5_REFERENCES[1])], pc_px, filter_cutoffs=F5_FILTER_CUTOFFS
        )
        cap = DEFAULT_FIT_OPTIONS["max_iterations"]
        final = fit_pattern(state_b, patterns[F5_CAPPED], max_iterations=cap)
        before = fit_pattern(state_b, patterns[F5_CAPPED], max_iterations=cap - 1)
        assert final["num_iterations"] == cap and not final["converged"]
        separation = abs(final["residual"] - before["residual"])
        assert separation > 1e-6 * final["residual"]
        gpu = run_fixture(fixture, **_gpu_kwargs(device_precision))
        residual = float(gpu["residual"][F5_CAPPED])
        assert abs(residual - final["residual"]) < abs(residual - before["residual"])
        assert_device_residual_band(
            np.array([final["residual"]]),
            np.array([residual]),
            f"capped residual ({device_precision})",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_many_grain_map_order(self, cupy_gpu, monkeypatch, device_precision):
        # M17, M18 on F7: 18 grains, so grain-pure batching, padding,
        # residency and the map-order restore all act
        # (critic finding F2): and the seam's ``pattern_index`` carries
        # the grain-ordered fit indices, not the map order (M52)
        fixture = f7_map()
        cpu = cpu_run("F7")
        records = spy_seam(monkeypatch, keep_states=False)
        gpu = run_fixture(fixture, chunksize=4, **_gpu_kwargs(device_precision))
        assert_seam_fit_order(records, fixture)
        _assert_host_outputs_bitwise(cpu, gpu)
        assert np.array_equal(cpu["converged"], gpu["converged"])
        corners = subregion_corners(SHAPE_60, pc_pixels_of(fixture["detector"]))
        assert_within(
            h_band(
                cpu["homography"],
                gpu["homography"],
                corners,
                _both_converged(cpu, gpu),
            ),
            _h_tol(device_precision),
            f"F7 h band ({device_precision})",
        )

    @pytest.mark.parametrize("arm", sorted(GATED_KNOB_ARMS))
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_knob_arm(self, cupy_gpu, device_precision, arm, record_property):
        fixture = f4_batch()
        knobs = GATED_KNOB_ARMS[arm]
        cpu = cpu_run("F4", **knobs)
        gpu = run_fixture(fixture, **knobs, **_gpu_kwargs(device_precision))
        both = _both_converged(cpu, gpu)
        assert both.any()
        band = h_band(cpu["homography"], gpu["homography"], fixture["corners"], both)
        record_property("h_band_px", band)
        assert_within(
            band, _h_tol(device_precision), f"knob {arm} ({device_precision})"
        )
        assert_device_residual_band(
            np.asarray(cpu["residual"])[both],
            np.asarray(gpu["residual"])[both],
            f"knob {arm} residual ({device_precision})",
        )
        key = (f"F4 {arm}", device_precision, "complex128")
        iterations, flips = _count_differences(cpu, gpu, _fitted(cpu))
        assert_count(
            iterations,
            _pin(GPU_ITERATION_DIFF_COUNT, key),
            f"GPU_ITERATION_DIFF_COUNT{key}",
        )
        assert_count(
            flips,
            _pin(GPU_CONVERGED_FLIP_COUNT, key),
            f"GPU_CONVERGED_FLIP_COUNT{key}",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_dc_offset_without_band_pass(self, cupy_gpu, device_precision):
        # M42, M46: (None, None) skips the band-pass, so the targets'
        # large DC offset reaches the device moments, where a moment
        # reconstructed against the wrong shift (or K = 0 in the mixed
        # build) shows; the DC-offset fixture is the default suite's
        # (``f4_dc_batch``, ``F4_DC_OFFSET`` on the targets only)
        fixture = f4_dc_batch()
        patterns = np.asarray(fixture["patterns"])
        cpu = cpu_run("F4dc", filter_cutoffs=(None, None))
        gpu = run_fixture(
            fixture, filter_cutoffs=(None, None), **_gpu_kwargs(device_precision)
        )
        both = _both_converged(cpu, gpu)
        assert both.sum() == len(patterns)
        assert_within(
            h_band(cpu["homography"], gpu["homography"], fixture["corners"], both),
            _h_tol(device_precision),
            f"DC-offset h band ({device_precision})",
        )
        assert_device_residual_band(
            np.asarray(cpu["residual"]),
            np.asarray(gpu["residual"]),
            f"DC-offset residual ({device_precision})",
        )
        key = ("F4 dc_offset", device_precision, "complex128")
        iterations, _ = _count_differences(cpu, gpu, _fitted(cpu))
        assert_count(
            iterations,
            _pin(GPU_ITERATION_DIFF_COUNT, key),
            f"GPU_ITERATION_DIFF_COUNT{key}",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_max_iterations_zero(self, cupy_gpu, device_precision):
        # M33: no update at all, whatever the CPU returns there: h is
        # the seed, the residual its criterion, norm_dp NaN, 0 counts
        fixture = f4_batch()
        cpu = cpu_run("F4", max_iterations=0)
        gpu = run_fixture(fixture, max_iterations=0, **_gpu_kwargs(device_precision))
        assert np.array_equal(cpu["num_iterations"], gpu["num_iterations"])
        assert np.array_equal(cpu["converged"], gpu["converged"])
        assert np.array_equal(np.isnan(cpu["norm_dp"]), np.isnan(gpu["norm_dp"]))
        finite = np.isfinite(np.asarray(cpu["homography"])).all(axis=1)
        assert finite.all()
        assert_within(
            h_band(cpu["homography"], gpu["homography"], fixture["corners"], finite),
            _first_step_tol(device_precision),
            f"max_iterations=0 h ({device_precision})",
        )
        assert_device_residual_band(
            np.asarray(cpu["residual"]),
            np.asarray(gpu["residual"]),
            f"max_iterations=0 residual ({device_precision})",
        )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_max_iterations_one(self, cupy_gpu, device_precision):
        fixture = f4_batch()
        cpu = cpu_run("F4", max_iterations=1)
        gpu = run_fixture(fixture, max_iterations=1, **_gpu_kwargs(device_precision))
        assert np.array_equal(cpu["num_iterations"], gpu["num_iterations"])
        assert np.all(np.asarray(gpu["num_iterations"]) == 1)
        assert np.array_equal(cpu["converged"], gpu["converged"])
        finite = np.isfinite(np.asarray(cpu["homography"])).all(axis=1)
        assert_within(
            h_band(cpu["homography"], gpu["homography"], fixture["corners"], finite),
            _first_step_tol(device_precision),
            f"max_iterations=1 h ({device_precision})",
        )

    @pytest.mark.parametrize("offset", GATED_PLANTED_OFFSETS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_planted_far_coordinates_fold(self, cupy_gpu, device_precision, offset):
        # M44 (D21.6.3): every finite coordinate, however large, is
        # folded in floating point before any integer conversion, with
        # no device error; at float64 the folded values equal the
        # numpy twin's (the CPU's numba kernel)
        cp = cupy_gpu
        fixture = f4_batch()
        state = fixture_state(fixture)
        _, coefficients = preprocessed_targets(state, fixture["targets"])
        matrices = np.stack([np.eye(3)] * len(coefficients))
        matrices[:, 0, 2] = offset
        matrices[:, 1, 2] = -offset
        ctx = _device_context(cp, device_precision)
        resident = _batched.build_resident(ctx, state)
        values, ok = ctx.kernels.gather(
            resident, cp.asarray(coefficients), cp.asarray(matrices)
        )
        cp.cuda.runtime.deviceSynchronize()
        values = cp.asnumpy(values)
        assert cp.asnumpy(ok).all()
        assert np.isfinite(values).all()
        if device_precision == "float64":
            np_ctx = numpy_seed_context("float64")
            np_values, np_ok = np_ctx.kernels.gather(
                _batched.build_resident(np_ctx, state), coefficients, matrices
            )
            assert np.asarray(np_ok).all()
            assert_within(
                _relative_difference(np_values, values),
                GPU_KERNEL_AB_TOL_F64,
                f"GPU_KERNEL_AB_TOL_F64 (gather at {offset!r})",
            )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_planted_non_finite_coordinates_are_flagged(
        self, cupy_gpu, device_precision
    ):
        # M8: flagged BEFORE the fold, never wrapped (D21.6.1)
        cp = cupy_gpu
        fixture = f4_batch()
        state = fixture_state(fixture)
        _, coefficients = preprocessed_targets(state, fixture["targets"])
        matrices = np.stack([np.eye(3)] * len(coefficients))
        matrices[0, 0, 2] = np.nan
        matrices[1, 1, 2] = np.inf
        ctx = _device_context(cp, device_precision)
        resident = _batched.build_resident(ctx, state)
        _, ok = ctx.kernels.gather(
            resident, cp.asarray(coefficients), cp.asarray(matrices)
        )
        cp.cuda.runtime.deviceSynchronize()
        assert np.array_equal(cp.asnumpy(ok), np.array([False, False, True]))

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_nan_values_give_nan_norm_dp_never_converged(
        self, cupy_gpu, device_precision
    ):
        # NaN VALUES in slot 0 (not a NaN corner displacement with a
        # finite step, critic finding F6) give a NaN norm_dp and never
        # converged; the other slots are untouched.  A CPU-faithful
        # kernel fails the slot through the D21.6.1 non-finite flags
        # first, so this kills M43 (a NaN-swallowing corner maximum)
        # only in an implementation without such a flag; otherwise M43
        # is reviewed-equivalent (a NaN corner from a finite step needs
        # an exact 0/0 projective corner)
        cp = cupy_gpu
        state, coefficients, matrices = _ab_inputs()
        ctx = _device_context(cp, device_precision)
        resident = _batched.build_resident(ctx, state)
        values, _ = ctx.kernels.gather(
            resident, cp.asarray(coefficients), cp.asarray(matrices)
        )
        values = cp.array(values)
        values[0, 5] = np.nan
        host = cp.asnumpy(values).astype(np.float64)
        shifts = cp.asarray(np.round(np.nanmean(host, axis=1)))
        sums = ctx.kernels.pixel_sums(resident, values, shifts)
        lockstep = _lockstep(cp, matrices)
        lockstep.shifts = shifts
        options = {
            "min_step": 1e300,
            "step_scale": 1.0,
            "max_iterations": DEFAULT_FIT_OPTIONS["max_iterations"],
        }
        ctx.kernels.reduce_solve_update(resident, sums, lockstep, options)
        device = _lockstep_to_host(cp, lockstep)
        # min_step 1e300: every FINITE norm converges at once, so only
        # a NaN-swallowing maximum could mark slot 0 converged
        assert not device["converged"][0]
        assert np.isnan(device["norm_dp"][0])
        assert device["converged"][1:].all()
        assert np.isfinite(device["norm_dp"][1:]).all()


class TestGatedIntensityAndUpdateRule:
    """V9(j) and (k) gated (D21.8(f), (g)): power-of-two intensity
    rescales bitwise on the device and the generic factor within its
    device band; and the device against the INDEPENDENT hand-built
    numpy IC-GN loop of the CPU ``UPDATE_RULE_TOL`` test, with the
    seed planted through the seam, at one and two iterations.

    Mutants: M36 (the update composed on the wrong side,
    ``W(dp)^-1 . W``: (k) is its ONLY killer, as on the CPU, because
    both arms share the fixed point), M4, M30, M32, M45 (float64 arm),
    M37, M38, M39 (the steepest-descent rebuild, (k))."""

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_power_of_two_rescales_are_bitwise(self, cupy_gpu, device_precision):
        reference, target, _ = _intensity_case()
        kwargs = _gpu_kwargs(device_precision)
        base = _two_point_run(reference, target, **kwargs)["homography"][1]
        assert np.all(np.isfinite(base))
        for scale_reference, scale_target in [
            (2.0**-10, 1.0),
            (1.0, 2.0**10),
            (2.0**10, 2.0**-10),
        ]:
            scaled = _two_point_run(
                scale_reference * reference, scale_target * target, **kwargs
            )["homography"][1]
            assert np.array_equal(scaled, base), (scale_reference, scale_target)

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_generic_intensity_factor(
        self, cupy_gpu, device_precision, record_property
    ):
        reference, target, corners = _intensity_case()
        kwargs = _gpu_kwargs(device_precision)
        base = _two_point_run(reference, target, **kwargs)["homography"][1]
        generic = _two_point_run(3.7 * reference, 0.31 * target, **kwargs)[
            "homography"
        ][1]
        measured = recovery_error(generic, base, corners)
        record_property("generic_px", measured)
        bound = (
            GPU_INTENSITY_SCALE_GENERIC_TOL_MIXED
            if device_precision == "mixed"
            else GPU_INTENSITY_SCALE_GENERIC_TOL_F64
        )
        assert_within(
            measured, bound, f"GPU_INTENSITY_SCALE_GENERIC_TOL ({device_precision})"
        )

    @pytest.mark.parametrize("n_iterations", [1, 2])
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_update_matches_the_hand_built_loop(
        self, cupy_gpu, monkeypatch, device_precision, n_iterations, record_property
    ):
        reference, target, h0, state, corners = _update_rule_case()
        table = np.stack([np.zeros(N_HOMOGRAPHY_PARAMETERS), h0])
        calls = plant_table(monkeypatch, table)
        properties = _two_point_run(
            reference,
            target,
            max_iterations=n_iterations,
            min_step=1e-12,
            **_gpu_kwargs(device_precision),
        )
        assert calls, "the device runner never called the seed seam"
        h = np.asarray(properties["homography"])[1]
        assert int(properties["num_iterations"][1]) == n_iterations
        expected = _hand_built_update(state, target, h0, n_iterations)
        measured = recovery_error(h, expected, corners)
        record_property("update_rule_px", measured)
        bound = (
            GPU_UPDATE_RULE_TOL_MIXED
            if device_precision == "mixed"
            else GPU_UPDATE_RULE_TOL_F64
        )
        assert_within(
            measured,
            bound,
            f"GPU_UPDATE_RULE_TOL ({device_precision}, {n_iterations} iterations)",
        )
        # the update really moved, so the comparison is not satisfied
        # by a device that does nothing
        assert recovery_error(h, h0, corners) > 1e-3


class TestGatedDriftTripwire:
    """V9(l) gated (D21.8(h)): on F1, per case, the device's recovery
    errors against the EXACT imposed homographies are pinned to dated
    literals per device precision within ``GPU_DRIFT_TRIPWIRE_PX``, and
    the CPU half is re-asserted in the same session, so an edit to
    shared code that moves both backends together fails loudly.

    Mutants: M27 (a shared helper perturbed; the CPU half is the
    default-suite killer, the device half its gated twin)."""

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_device_recovery_literals(
        self, cupy_gpu, device_precision, record_property
    ):
        fixture = f1_batch()
        properties = run_fixture(fixture, **_gpu_kwargs(device_precision))
        homography = np.asarray(properties["homography"])[1:]
        errors = np.array(
            [
                recovery_error(h, h_true, fixture["corners"])
                for h, h_true in zip(homography, fixture["exact"])
            ]
        )
        record_property("recovery_px", [float(e) for e in errors])
        literals = (
            GPU_DRIFT_RECOVERY_PX_MIXED
            if device_precision == "mixed"
            else GPU_DRIFT_RECOVERY_PX_F64
        )
        if literals is None:
            assert_within(
                float(errors.max()),
                None,
                f"GPU_DRIFT_RECOVERY_PX_{device_precision.upper()}",
            )
        literals = np.asarray(literals, dtype=np.float64)
        assert literals.shape == errors.shape
        assert_within(
            float(np.abs(errors - literals).max()),
            GPU_DRIFT_TRIPWIRE_PX,
            f"GPU_DRIFT_TRIPWIRE_PX ({device_precision})",
        )

    def test_cpu_half_in_the_gated_session(self, cupy_gpu):
        errors, iterations = drift_recovery_cpu()
        assert np.array_equal(iterations, CPU_DRIFT_NUM_ITERATIONS)
        assert_within(
            float(np.abs(errors - CPU_DRIFT_RECOVERY_PX).max()),
            CPU_DRIFT_TRIPWIRE_PX,
            "CPU_DRIFT_TRIPWIRE_PX",
        )


class TestGatedDeterminism:
    """V9(m) gated (D21.7, D21.11): two runs bitwise at a fixed B at
    both device and both seed precisions; B invariance at B in {8, 32,
    40, default}, bitwise among B >= 32 by construction and within
    ``GPU_BATCH_INVARIANCE_TOL`` (max absolute homography parameter
    difference, 0 when bitwise) against B = 8; lazy equals eager
    bitwise on the uint8 Ni map; a 4-dask-worker lock stress run equal
    to the 1-worker run; and the launch-dimension spy at B = 8 against
    B = 32 (the call site the pure layout function cannot see).

    Mutants: M2 (the launch-dimension spy, its designed killer, and B
    invariance), M23 (the 4-worker stress), M1 (run to run; NOT a
    reliable killer, ledger 95: the ``atomicAdd`` source pin is)."""

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_two_runs_bitwise(self, cupy_gpu, device_precision, seed_precision):
        kwargs = _gpu_kwargs(device_precision, seed_precision)
        first, second = (
            run_fixture(f1_batch(), chunksize=8, **kwargs) for _ in range(2)
        )
        assert_properties_bitwise(first, second)

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_batch_size_invariance(
        self, cupy_gpu, monkeypatch, device_precision, record_property
    ):
        built = _spy_sessions(monkeypatch)
        results = {}
        for chunksize in (8, 32, 40, None):
            start = len(built)
            results[chunksize] = run_fixture(
                f1_batch(), chunksize=chunksize, **_gpu_kwargs(device_precision)
            )
            if chunksize is None:
                default_batch = built[start]
        record_property("default_batch_size", default_batch)
        large = [32, 40] + ([None] if default_batch >= 32 else [])
        # bitwise among B >= 32 is EXPECTED by construction (P = 32
        # fixed, D21.7.3), not measured yet; V9(m) makes cross-B
        # invariance MTP, so a measured non-bitwise result here needs a
        # dated deviation in validation.md and this assert moved to
        # GPU_BATCH_INVARIANCE_TOL (critic finding F12)
        for chunksize in large[1:]:
            assert_properties_bitwise(results[large[0]], results[chunksize])
        worst = 0.0
        for chunksize in (32, 40, None):
            difference = np.abs(
                np.asarray(results[chunksize]["homography"])
                - np.asarray(results[8]["homography"])
            )
            measured = float(np.nanmax(difference, initial=0.0))
            record_property(f"max_h_difference_{chunksize}", measured)
            worst = max(worst, measured)
        assert_within(worst, GPU_BATCH_INVARIANCE_TOL, "GPU_BATCH_INVARIANCE_TOL")

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_lazy_equals_eager(self, cupy_gpu, device_precision):
        # the shipped Ni map is uint8: the upload converts the stored
        # dtype exactly, so lazy and eager agree bitwise (D21.7.4)
        signal, _, detector = ni_inputs()
        eager = np.asarray(signal.data)
        eager = eager.reshape((-1,) + eager.shape[-2:])
        lazy = da.from_array(eager, chunks=(3, -1, -1))
        results = [
            run_engine(
                patterns,
                NI_NAVIGATION_SHAPE,
                detector,
                reference=NI_REFERENCE,
                chunksize=4,
                **_gpu_kwargs(device_precision),
            )
            for patterns in (eager, lazy)
        ]
        assert_properties_bitwise(results[0], results[1])

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_four_worker_lock_stress(self, cupy_gpu, device_precision):
        # four concurrent dask workers contend over seven chunks for
        # the lock-guarded device section (M23)
        kwargs = _gpu_kwargs(device_precision)
        runs = []
        for workers in (1, 4):
            with dask.config.set(scheduler="threads", num_workers=workers):
                runs.append(run_fixture(f1_batch(), chunksize=2, **kwargs))
        assert_properties_bitwise(runs[0], runs[1])

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_launch_dimensions_do_not_depend_on_batch_size(
        self, cupy_gpu, monkeypatch, device_precision, record_property
    ):
        # M2: the per-pattern grid and the block are a function of the
        # pattern geometry only (D21.7.3), at the CALL SITE
        launches = _spy_raw_kernels(monkeypatch, cupy_gpu)
        layouts = {}
        for batch_size in (8, 32):
            start = len(launches)
            run_fixture(
                f4_batch(), chunksize=batch_size, **_gpu_kwargs(device_precision)
            )
            recorded = launches[start:]
            assert recorded, (
                "no RawKernel launch recorded: the kernels must be built "
                "through cupy.RawKernel inside make_kernel_namespace"
            )
            layouts[batch_size] = _per_pattern_layout(recorded, batch_size)
        record_property("layout_8", sorted(map(str, layouts[8])))
        assert layouts[8] == layouts[32]
        threads, blocks_per_pattern, _ = _batched._launch_layout(
            fixture_state(f4_batch()).n_pixels
        )
        assert any(
            int(np.prod(block)) == threads and per_pattern == blocks_per_pattern
            for _, block, per_pattern in layouts[8]
        )


class TestGatedRobustness:
    """V9(n) gated (D21.9, D21.10): an out-of-memory at session build
    halves B; a mid-compute one rebuilds at B/2 and re-runs the whole
    map from its final B; the B = 1 floor raises ``MemoryError`` with
    the D21.10 text in both windows; a run under a REAL pool limit
    recovers, leaves at most ``GPU_LEAK_RESIDUE_BYTES`` and equals an
    unlimited run at the same final B bitwise, on F1 and on the
    many-grain F7; a planted device-stage exception fails the run; the
    compute's ``scheduler`` is ``"threads"``; no device attribute
    survives on any returned object; and the session's contract and
    dask token.

    Mutants: M19 (planted exception), M20 (halving spies, the leak
    pin), M23 (scheduler spy), M51 (F7 under ``set_limit``)."""

    def _clean(self, fixture, batch_size, device_precision="mixed"):
        return run_fixture(
            fixture, chunksize=batch_size, **_gpu_kwargs(device_precision)
        )

    def test_oom_at_session_build_halves(self, cupy_gpu, monkeypatch):
        def flaky(attempt, batch_size):
            if attempt == 1:
                raise _out_of_memory(cupy_gpu)

        built = _spy_sessions(monkeypatch, on_build=flaky)
        properties = run_fixture(f4_batch(), chunksize=8, **_gpu_kwargs())
        assert built == [8, 4]
        monkeypatch.undo()
        assert_properties_bitwise(properties, self._clean(f4_batch(), 4))

    def test_oom_mid_compute_rebuilds_at_half(self, cupy_gpu, monkeypatch):
        def flaky(attempt, session):
            if attempt != 1:
                return

            def raising(*args, **kwargs):
                raise _out_of_memory(cupy_gpu)

            session.kernels.pixel_sums = raising

        built = _spy_sessions(monkeypatch, on_session=flaky)
        properties = run_fixture(f4_batch(), chunksize=8, **_gpu_kwargs())
        assert built == [8, 4]
        monkeypatch.undo()
        assert_properties_bitwise(properties, self._clean(f4_batch(), 4))

    @staticmethod
    def _assert_memory_error_text(message: str) -> None:
        # the pattern shape, the model MB per term, the free VRAM and
        # both remedies (D21.10.4)
        assert re.search(r"60\s*[x×,]\s*60", message)
        assert re.search(r"\d[\d,_.]*\s*(GiB|GB|MiB|MB)\b", message)
        assert "VRAM" in message
        assert re.search(r"backend\s*=\s*['\"]cpu['\"]", message)
        assert "chunksize" in message

    def test_floor_at_session_build_raises_memory_error(self, cupy_gpu, monkeypatch):
        def always(attempt, batch_size):
            raise _out_of_memory(cupy_gpu)

        built = _spy_sessions(monkeypatch, on_build=always)
        with pytest.raises(MemoryError) as info:
            run_fixture(f4_batch(), chunksize=8, **_gpu_kwargs())
        assert built == [8, 4, 2, 1]
        self._assert_memory_error_text(str(info.value))

    def test_floor_mid_compute_raises_memory_error(self, cupy_gpu, monkeypatch):
        def always(attempt, session):
            def raising(*args, **kwargs):
                raise _out_of_memory(cupy_gpu)

            session.kernels.pixel_sums = raising

        built = _spy_sessions(monkeypatch, on_session=always)
        with pytest.raises(MemoryError) as info:
            run_fixture(f4_batch(), chunksize=1, **_gpu_kwargs())
        assert built == [1]
        self._assert_memory_error_text(str(info.value))

    def test_planted_device_exception_fails_the_run(self, cupy_gpu, monkeypatch):
        # D21.9.4: a non-OOM device error never enters the halving
        # loop and is never turned into NaN rows
        def planted(attempt, session):
            def raising(*args, **kwargs):
                raise RuntimeError("planted device failure")

            session.kernels.final_criterion = raising

        built = _spy_sessions(monkeypatch, on_session=planted)
        with pytest.raises(RuntimeError, match="planted device failure"):
            run_fixture(f4_batch(), chunksize=8, **_gpu_kwargs())
        assert built == [8]

    @pytest.mark.parametrize(
        "name, n_pixels, limit_batch",
        [("F1", SHAPE_480[0] * SHAPE_480[1], 2), ("F7", SHAPE_60[0] * SHAPE_60[1], 4)],
    )
    def test_real_pool_limit_recovers_without_leak(
        self, cupy_gpu, monkeypatch, name, n_pixels, limit_batch, record_property
    ):
        # a REAL pool limit at the model's own size of a small B
        # drives the halving without any monkeypatched error; the
        # recovered run leaves at most the leak pin and equals an
        # unlimited run at its final B bitwise
        cp = cupy_gpu
        fixture = fixture_of(name)
        pool = cp.get_default_memory_pool()
        limit = _gpu._vram_model_bytes(limit_batch, n_pixels, "mixed", "complex128")
        record_property("limit_bytes", int(limit))
        built = _spy_sessions(monkeypatch)
        pool.free_all_blocks()
        cp.fft.config.get_plan_cache().clear()
        pool.set_limit(size=int(limit))
        try:
            limited = run_fixture(fixture, chunksize=None, **_gpu_kwargs())
            residue = int(pool.used_bytes())
        finally:
            pool.set_limit(size=0)
            pool.free_all_blocks()
        record_property("built", list(built))
        record_property("residue_bytes", residue)
        assert len(built) >= 2, "the limit did not force a halving"
        assert_within(residue, GPU_LEAK_RESIDUE_BYTES, "GPU_LEAK_RESIDUE_BYTES")
        monkeypatch.undo()
        assert_properties_bitwise(limited, self._clean(fixture, built[-1]))

    def test_the_compute_is_threaded(self, cupy_gpu, monkeypatch):
        captured = []
        real_compute = da.Array.compute
        real_dask_compute = dask.compute

        def spying_compute(self, **kwargs):
            captured.append(kwargs.get("scheduler"))
            return real_compute(self, **kwargs)

        def spying_dask_compute(*args, **kwargs):
            captured.append(kwargs.get("scheduler"))
            return real_dask_compute(*args, **kwargs)

        monkeypatch.setattr(da.Array, "compute", spying_compute)
        monkeypatch.setattr(dask, "compute", spying_dask_compute)
        run_fixture(f4_batch(), chunksize=2, **_gpu_kwargs())
        assert captured, "the device runner must compute through dask"
        assert set(captured) == {"threads"}

    def test_no_device_object_survives(self, cupy_gpu):
        properties = run_fixture(f4_batch(), **_gpu_kwargs())
        for name, value in properties.items():
            assert isinstance(value, np.ndarray), name
            _session_host_check(value, name)
        xmap = run_ni(backend="gpu")
        for name in xmap.prop.keys():
            _session_host_check(xmap.prop[name], name)
        for name, value in vars(xmap).items():
            _session_host_check(value, name)

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_session_contract(self, cupy_gpu, device_precision, seed_precision):
        cp = cupy_gpu
        session = _gpu._make_session(
            "cupy",
            2,
            device_precision=device_precision,
            seed_precision=seed_precision,
        )
        try:
            assert isinstance(session, _gpu._GpuSession)
            assert session.xp is cp
            assert session.fft is cp.fft
            assert isinstance(session.kernels, _batched.KernelNamespace)
            assert session.kernels.precision == device_precision
            assert session.kernels.out_of_memory_error is (
                cp.cuda.memory.OutOfMemoryError
            )
            assert session.batch_size == 2
            assert session.sub_batch_size == min(FROZEN_SUB_BATCH_SIZE, 2)
            assert session.device_precision == device_precision
            assert session.seed_precision == seed_precision
            assert session.lock is _gpu._DEVICE_LOCK
            token = session.__dask_tokenize__()
            assert token == (SESSION_TOKEN_NAME, id(session))
            assert dask.base.tokenize(session) == dask.base.tokenize(session)
        finally:
            session.close()
        session.close()


class TestGatedVramCalibration:
    """V9(o) gated (D21.10): the three terms of the VRAM model
    calibrated SEPARATELY against pool high-water marks at 512x622
    (F3s), each within its pinned ``(low, high)`` and never above the
    model's own term; the whole model at least the high-water mark of
    every run; the default B fed by a real free-VRAM query becoming
    the session's B and never clamped to the number of fitted points;
    an explicit ``chunksize`` winning; and the information message's
    device block and its warning line.

    Mutants: M34 (explicit ``chunksize`` ignored: the batch spy), M35
    (the model without the preprocessing transient: the calibration,
    its designed killer here; the faked-free-VRAM chooser is the
    default suite's)."""

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_resident_term(
        self, cupy_gpu, device_precision, seed_precision, record_property
    ):
        cp = cupy_gpu
        state = fixture_state(f3s_batch())
        n_pixels = SHAPE_RECT[0] * SHAPE_RECT[1]
        with _PoolHighWater(cp) as tracker:
            ctx = _device_context(cp, device_precision)
            resident = _batched.build_resident(ctx, state)
            seed_state = _batched.build_seed_state(ctx, state, precision=seed_precision)
            held = int(tracker.pool.used_bytes()) - tracker.base
        measured = tracker.high_water
        record_property("r_high_water_bytes", measured)
        record_property("r_held_bytes", held)
        assert resident is not None and seed_state is not None
        _, _, r_model = _gpu._vram_model_terms(
            n_pixels, device_precision, seed_precision
        )
        assert r_model >= measured
        if seed_precision == "complex128":
            bounds = (
                VRAM_R_BOUNDS_RECT_MIXED
                if device_precision == "mixed"
                else VRAM_R_BOUNDS_RECT_F64
            )
            _assert_in_bounds(
                measured, bounds, f"VRAM_R_BOUNDS_RECT ({device_precision})"
            )

    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    def test_per_slot_and_transient_terms(
        self, cupy_gpu, device_precision, record_property
    ):
        cp = cupy_gpu
        n_pixels = SHAPE_RECT[0] * SHAPE_RECT[1]
        peaks = {}
        for batch_size in (8, 16, 32, 64):
            with _PoolHighWater(cp) as tracker:
                run_fixture(
                    f3s_batch(),
                    chunksize=batch_size,
                    **_gpu_kwargs(device_precision),
                )
            peaks[batch_size] = tracker.high_water
            record_property(f"high_water_{batch_size}", peaks[batch_size])
            # the whole model (R_MAX residents) bounds one-grain runs
            assert peaks[batch_size] <= _gpu._vram_model_bytes(
                batch_size, n_pixels, device_precision, "complex128"
            ), batch_size
        # B >= 32: P = 32 fixed, so the difference is 32 slots of g;
        # below 32 P follows B, so 8 slots of g + p
        g = (peaks[64] - peaks[32]) / 32
        p = (peaks[16] - peaks[8]) / 8 - g
        record_property("g_bytes", g)
        record_property("p_bytes", p)
        g_model, p_model, _ = _gpu._vram_model_terms(
            n_pixels, device_precision, "complex128"
        )
        assert g_model >= g
        assert p_model >= p
        _assert_in_bounds(g, VRAM_G_BOUNDS_RECT, "VRAM_G_BOUNDS_RECT")
        _assert_in_bounds(p, VRAM_P_BOUNDS_RECT, "VRAM_P_BOUNDS_RECT")

    def test_default_batch_size_wiring(self, cupy_gpu, monkeypatch):
        chosen = []
        real_default = _gpu._default_batch_size

        def spying_default(free_bytes, n_pixels, device_precision, seed_precision):
            value = real_default(free_bytes, n_pixels, device_precision, seed_precision)
            chosen.append((free_bytes, n_pixels, value))
            return value

        monkeypatch.setattr(_gpu, "_default_batch_size", spying_default)
        built = _spy_sessions(monkeypatch)
        run_fixture(f4_batch(), chunksize=None, **_gpu_kwargs())
        assert chosen, "chunksize=None must consult _default_batch_size"
        free_bytes, n_pixels, value = chosen[-1]
        assert free_bytes > 0
        assert n_pixels == SHAPE_60[0] * SHAPE_60[1]
        assert value in FROZEN_BATCH_SIZES
        # the model's B IS the session's B, never clamped to F4's four
        # fitted points (D21.10.1)
        assert built[0] == value

    @pytest.mark.parametrize("chunksize", [3, 8, 40])
    def test_an_explicit_chunksize_wins(self, cupy_gpu, monkeypatch, chunksize):
        built = _spy_sessions(monkeypatch)
        run_fixture(f4_batch(), chunksize=chunksize, **_gpu_kwargs())
        assert built == [chunksize]

    def test_information_message(self, cupy_gpu, monkeypatch, capsys):
        device_name = cupy_gpu.cuda.runtime.getDeviceProperties(0)["name"].decode()
        run_fixture(f4_batch(), chunksize=2, verbose=1, **_gpu_kwargs())
        out = capsys.readouterr().out
        assert "HREBSD-DIC information:" in out
        assert device_name in out
        assert "VRAM" in out
        assert re.search(r"\d[\d,_.]*\s*(GiB|GB|MiB|MB)\b", out)
        assert "exceed" not in out.lower()
        # the warning line when the model exceeds the free VRAM
        monkeypatch.setattr(_gpu, "_free_device_bytes", lambda namespace: 1024)
        run_fixture(f4_batch(), chunksize=2, verbose=1, **_gpu_kwargs())
        out = capsys.readouterr().out
        assert "exceed" in out.lower()


class TestGatedThroughput:
    """D21.16, RECORDED only (``record_property``), asserting nothing
    but completion: best of three timed runs after a warm-up on the
    480 px F2 set and the 512x622 F3 set, at both device and both seed
    precisions.  The go/no-go floor is a ledger record on the
    Si-indent data (V9(q)), never a test.  No mutant: no killer."""

    @pytest.mark.parametrize("seed_precision", FROZEN_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", FROZEN_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", ["F2-0", "F3"])
    def test_throughput_recorded(
        self, cupy_gpu, name, device_precision, seed_precision, record_property
    ):
        fixture = fixture_of(name)
        kwargs = _gpu_kwargs(device_precision, seed_precision)
        n_points = int(np.prod(fixture["navigation_shape"]))
        run_fixture(fixture, **kwargs)  # warm-up: kernels, plans, pools
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            properties = run_fixture(fixture, **kwargs)
            best = min(best, time.perf_counter() - start)
        record_property("machine", MACHINE_A)
        record_property("best_of_3_s", best)
        record_property("patterns_per_s", n_points / best)
        assert np.asarray(properties["homography"]).shape == (n_points, 8)


# --------------------------- Mutation map --------------------------- #
#
# Plan 11 item 4, M1-M53, each mapped 2026-10-06 (failing-tests gate)
# to the test(s) DESIGNED to kill it.  "D:" names a default-suite
# killer (numpy session or twin, runs on CI), "G:" a gated one (real
# device, -n 0, pinned overlay).  None of these kills is VERIFIED
# yet: the plan 11 bug-injection pass simulates each mutant and
# corrects any line whose killer is wrong; a mutant no test kills is
# recorded reviewed-equivalent with its argument at the review gate.
# Every mutant below has at least one designed killer; M3, M4 and M36
# have GATED killers only (no default-suite twin can see a float32
# reduction or a device-side update order).
# "D*:" (M37 to M42 and M44, critic finding F10) marks a mutation that
# lives in the CUDA source, which the numpy twin never runs: the G
# killers are the device's; a D* test kills only the numpy-twin
# VARIANT of the same injection, and the bug-injection pass must
# inject both and never count a twin-only kill as a device kill (M42
# is defined for the mixed build, which the float64 twin lacks).
#
#  M1 atomic reduction ......... D: TestCudaSourcePins::test_no_atomic_anywhere,
#                                ::test_no_atomic_scatter_through_cupy (the
#                                cupy scatter_add / .add.at variant, critic F9)
#                                (+ ::test_the_cuda_source_exists, so the pin
#                                cannot pass vacuously).  G: TestGatedDeterminism
#                                ::test_two_runs_bitwise is NOT a reliable killer
#  M2 blocks per pattern from B  D: TestLaunchLayout::test_the_signature_takes_the
#                                _pixel_count_only, ::test_the_layout_is_independent
#                                _of_everything_but_the_pixels (definition half).
#                                G: TestGatedDeterminism::test_launch_dimensions_do
#                                _not_depend_on_batch_size (call-site half),
#                                ::test_batch_size_invariance
#  M3 f32 reductions ........... G only: TestGatedParity::test_first_step_band,
#                                TestGatedKernelAB::test_pixel_sums_and_final
#                                _criterion, ::test_reduce_solve_update
#  M4 8x8 solve/update in f32 .. G only: TestGatedKernelAB::test_reduce_solve
#                                _update, TestGatedIntensityAndUpdateRule::test
#                                _update_matches_the_hand_built_loop,
#                                TestGatedParity::test_converged_parity
#  M5 retire mask ignored ...... D: TestBatchedCoreNumpy::test_batched_equals_alone
#                                _on_f5, ::test_own_iteration_counts_on_f5.
#                                G: TestGatedBatchedSemantics::test_alone_equals
#                                _batched_on_f5
#  M6 exit test before update .. D: TestBatchedCoreNumpy::test_parity_with_the_cpu
#                                (counts, h band).  G: TestGatedParity::test
#                                _converged_parity, TestGatedSeedSeam::test
#                                _planted_rows_are_honoured
#  M7 no final criterion ....... D: TestBatchedCoreNumpy::test_the_residual_is_the
#                                _final_criterion.  G: TestGatedBatchedSemantics
#                                ::test_residual_is_the_final_criterion
#  M8 non-finite coord unflagged D: TestBatchedCoreNumpy::test_gather_flags_a_non
#                                _finite_coordinate.  G: TestGatedBatchedSemantics
#                                ::test_planted_non_finite_coordinates_are_flagged
#  M9 zero ZMN norm unflagged .. D: TestRunBatchesNumpySession::test_the_band
#                                _passed_constant_never_converges, ::test_the
#                                _failure_contract_on_f5.  G: TestGatedBatched
#                                Semantics::test_band_passed_constant_is_not
#                                _converged, ::test_f5_failure_contract_and_map
#                                _order
#  M10 seam rows ignored ....... D: TestSeedSeamHonoursArbitraryH0Numpy::test_the
#                                _one_iteration_arm, ::test_planted_rows_reproduce
#                                _the_cpu_fit, ::test_the_planted_rows_are
#                                _discriminating.  G: TestGatedSeedSeam::test_one
#                                _iteration_arm, ::test_planted_rows_are_honoured,
#                                ::test_planted_rows_discriminate
#  M11 seam via captured ref ... D: TestSeedSeamContract::test_the_runner_calls_the
#                                _seam_once_per_sub_batch.  G: TestGatedSeedParity
#                                ::test_the_runner_reaches_the_seam_per_padded_sub
#                                _batch
#  M12 seed (dy, dx) transposed  D: TestSeedSeamContract::test_rows_equal_initial
#                                _guess.  G: TestGatedSeedParity::test_complex128
#                                _seeds_equal_initial_guess
#  M13 cross-power conjugate ... D: TestSeedSeamContract::test_rows_equal_initial
#                                _guess.  G: TestGatedSeedParity::test_complex128
#                                _seeds_equal_initial_guess
#  M14 complex64 for complex128  D: TestSeedSeamContract::test_the_seed_state,
#                                ::test_spectra_and_rows_layout, ::test_the_seed
#                                _precision_reaches_the_seam.  G: TestGatedSeed
#                                Parity::test_device_output_contract,
#                                ::test_complex128_seeds_equal_initial_guess
#  M15 crops not ZMN-ed ........ D: TestSeedSeamContract::test_the_dimmed_arm
#                                (DIM_SCALE 1e-13, premise asserted).
#                                G: TestGatedSeedParity::test_dimmed_arm_still
#                                _equals_initial_guess (the same DIM_SCALE recipe
#                                and premises since critic F3).  Possibly
#                                equivalent
#  M16 padded slots in output .. D: TestBatchedCoreNumpy::test_padded_slots_never
#                                _reach_the_output.  G: TestGatedBatchedSemantics
#                                ::test_padded_slots_never_reach_the_output
#  M17 grain order not restored  D: TestRunBatchesNumpySession::test_the_map_order
#                                _on_f7 (F7: fit order differs from map order;
#                                F5's does not, so ::test_the_map_order_on_f5 and
#                                ::test_the_wiring_probes_on_f5 cannot see it,
#                                critic F2).  G: TestGatedBatchedSemantics::test
#                                _many_grain_map_order
#  M18 wrong grain's residents . D: TestRunBatchesNumpySession::test_every_batch
#                                _runs_against_its_own_grain.  G: TestGated
#                                BatchedSemantics::test_f5_failure_contract_and
#                                _map_order
#  M19 device exc swallowed .... D: TestRunBatchesNumpySession::test_a_device
#                                _exception_fails_the_run.  G: TestGatedRobustness
#                                ::test_planted_device_exception_fails_the_run
#  M20 OOM loop never halves ... D: TestRunBatchesNumpySession::test_an_out_of
#                                _memory_at_build_halves_b, ::test_an_out_of_memory
#                                _mid_compute_rebuilds_and_re_runs, ::test_the
#                                _batch_one_floor_raises_memory_error.
#                                G: TestGatedRobustness::test_oom_at_session_build
#                                _halves, ::test_oom_mid_compute_rebuilds_at_half,
#                                ::test_real_pool_limit_recovers_without_leak
#  M21 gate stages reordered ... D: TestAvailabilityGate::test_stage_order_with_the
#                                _shim_between_b_and_c.  G: TestGatedGate::test
#                                _gate_passes_and_caches
#  M22 FFT-only stage (c) ...... D: TestAvailabilityGate::test_stage_c_probes_every
#                                _library_the_device_path_calls, ::test_stage_c
#                                _library_failure.  G: TestGatedGate::test_stage_c
#                                _probes_every_library_on_the_device
#  M23 lock dropped/no threads . D: TestRunBatchesNumpySession::test_the_threaded
#                                _scheduler_and_the_device_lock.  G: TestGated
#                                Determinism::test_four_worker_lock_stress,
#                                TestGatedRobustness::test_the_compute_is_threaded
#  M24 cupy at module scope .... D: TestNoModuleScopeCupy::test_no_module_scope
#                                _cupy_by_ast, ::test_no_module_scope_cupy_by
#                                _regex, ::test_importing_kikuchipy_imports_no
#                                _cupy, ::test_a_numpy_session_imports_no_cupy
#  M25 Stage D raise too late .. D: TestSeedFromNeighborsOnGpuRaises::test_the
#                                _raise_carries_the_frozen_literal, ::test_no
#                                _import_error_surfaces_first_without_cupy,
#                                ::test_the_raise_comes_through_the_signal_method
#  M26 silent CPU fallback ..... D: TestBackendSwitch::test_gate_failure_raises
#                                _with_no_cpu_fallback, TestAvailabilityGate::test
#                                _a_failed_gate_fails_the_engine_call.
#                                G: TestGatedGate::test_the_device_runner_is_what
#                                _runs
#  M27 shared helper perturbed . D: TestDriftTripwireCpuHalf (all three), the
#                                pre-Stage-D pins of test_hrebsd_engine.py and
#                                TestBackendSwitch::test_cpu_equals_no_keyword
#                                _bitwise_on_f6 / _on_the_ni_map.  G: TestGated
#                                DriftTripwire::test_cpu_half_in_the_gated_session
#  M28 step_scale ignored ...... D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[step_scale*].  G: TestGatedBatched
#                                Semantics::test_knob_arm
#  M29 window not applied ...... D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[window].  G: TestGatedBatchedSemantics
#                                ::test_knob_arm
#  M30 W33 renorm skipped ...... D: TestBatchedCoreNumpy::test_parity_with_the_cpu.
#                                G: TestGatedParity::test_converged_parity,
#                                TestGatedIntensityAndUpdateRule::test_update
#                                _matches_the_hand_built_loop
#  M31 corner norm wrong corners D: TestBatchedCoreNumpy::test_parity_with_the_cpu
#                                (counts).  G: TestGatedParity::test_converged
#                                _parity
#  M32 h held in f32 ........... D: TestBatchedCoreNumpy::test_parity_with_the_cpu
#                                (float64 band).  G: TestGatedParity::test
#                                _converged_parity, TestGatedIntensityAndUpdate
#                                Rule::test_update_matches_the_hand_built_loop
#  M33 max_iterations off by one D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[max_iterations*], ::test_own_iteration
#                                _counts_on_f5.  G: TestGatedBatchedSemantics::test
#                                _max_iterations_zero, ::test_max_iterations_one
#  M34 explicit chunksize ignored D: TestBatchModel::test_an_explicit_chunksize
#                                _wins.  G: TestGatedVramCalibration::test_an
#                                _explicit_chunksize_wins
#  M35 VRAM model without P .... D: TestBatchModel::test_the_p_and_r_terms_decide,
#                                ::test_the_model_is_the_frozen_formula.
#                                G: TestGatedVramCalibration::test_per_slot_and
#                                _transient_terms
#  M36 update on the wrong side  G only: TestGatedIntensityAndUpdateRule::test
#                                _update_matches_the_hand_built_loop
#  M37 gx/gy swapped in SD ..... D*: TestBatchedCoreNumpy::test_the_first_step.
#                                G: TestGatedParity::test_first_step_band,
#                                TestGatedIntensityAndUpdateRule::test_update
#                                _matches_the_hand_built_loop
#  M38 xi_x/xi_y swapped ....... D*: TestBatchedCoreNumpy::test_the_first_step.
#                                G: TestGatedParity::test_first_step_band,
#                                TestGatedIntensityAndUpdateRule::test_update
#                                _matches_the_hand_built_loop
#  M39 perspective sign flipped  D*: TestBatchedCoreNumpy::test_the_first_step.
#                                G: TestGatedParity::test_first_step_band,
#                                TestGatedIntensityAndUpdateRule::test_update
#                                _matches_the_hand_built_loop
#  M40 unweighted constants .... D*: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[window], ::test_parity_with_the_cpu.
#                                G: TestGatedBatchedSemantics::test_knob_arm
#  M41 window applied once ..... D*: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[window].  G: TestGatedBatchedSemantics
#                                ::test_knob_arm
#  M42 K reconstruction ........ D*: TestBatchedCoreNumpy::test_parity_with_the_cpu,
#                                ::test_every_knob_against_the_cpu[no_band_pass].
#                                G: TestGatedBatchedSemantics::test_dc_offset
#                                _without_band_pass, TestGatedParity::test
#                                _converged_parity
#  M43 corner max swallows NaN . D: TestBatchedCoreNumpy::test_a_nan_reaching_the
#                                _step_never_converges.  G: TestGatedBatched
#                                Semantics::test_nan_values_give_nan_norm_dp_never
#                                _converged.  Both plant NaN sums/values, not a
#                                NaN corner from a finite step, so they kill M43
#                                only in an implementation WITHOUT a separate
#                                non-finite-step flag (D21.6.1); otherwise M43 is
#                                reviewed-equivalent at the review gate (a NaN
#                                corner from a finite step needs an exact 0/0
#                                projective corner, critic F6)
#  M44 int before the fold ..... D*: TestBatchedCoreNumpy::test_gather_folds_huge
#                                _coordinates_like_the_cpu.  G: TestGatedBatched
#                                Semantics::test_planted_far_coordinates_fold,
#                                TestGatedKernelAB::test_gather
#  M45 matrix rebuilt as 1 + h . D: TestBatchedCoreNumpy::test_parity_with_the_cpu
#                                (float64 band).  G: TestGatedIntensityAndUpdate
#                                Rule::test_update_matches_the_hand_built_loop,
#                                TestGatedKernelAB::test_reduce_solve_update
#  M46 band-pass skip inverted . D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[no_band_pass, low_pass].  G: TestGated
#                                BatchedSemantics::test_dc_offset_without_band
#                                _pass, ::test_knob_arm
#  M47 upsample hard-coded ..... D: TestSeedSeamContract::test_rows_equal_initial
#                                _guess[*-2, *-1], TestBatchedCoreNumpy::test
#                                _every_knob_against_the_cpu[upsample_8].
#                                G: TestGatedSeedParity::test_complex128_seeds
#                                _equal_initial_guess, TestGatedBatchedSemantics
#                                ::test_knob_arm
#  M48 min_step hard-coded ..... D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[min_step].  G: TestGatedBatchedSemantics
#                                ::test_knob_arm
#  M49 border ignored .......... D: TestBatchedCoreNumpy::test_every_knob_against
#                                _the_cpu[border].  G: TestGatedBatchedSemantics
#                                ::test_knob_arm
#  M50 tail sub-batch unpadded . D: TestBatchedCoreNumpy::test_a_last_sub_batch
#                                _slot_equals_alone, TestSeedSeamContract::test_the
#                                _runner_calls_the_seam_once_per_sub_batch.
#                                G: TestGatedBatchedSemantics::test_last_sub_batch
#                                _slot_equals_alone, TestGatedSeedParity::test_the
#                                _runner_reaches_the_seam_per_padded_sub_batch
#  M51 residents never evicted . D: TestRunBatchesNumpySession::test_the_residency
#                                _bound_on_f7 (at most R_MAX - 1 alive at upload
#                                entry, critic F7, so evict-after-upload fails
#                                too).  G: TestGatedRobustness::test_real_pool
#                                _limit_recovers_without_leak
#  M52 pattern_index wrong ..... D: TestRunBatchesNumpySession::test_the_seam
#                                _carries_the_fit_order_on_f7 (map order against
#                                grain order, critic F2), TestSeedSeamContract
#                                ::test_the_sub_batch_carries_the_targets_and
#                                _their_coefficients (wrong index).
#                                G: TestGatedBatchedSemantics::test_many_grain
#                                _map_order (the same fit-order assert),
#                                TestGatedSeedParity::test_the_runner_reaches_the
#                                _seam_per_padded_sub_batch
#  M53 seam once per batch ..... D: TestSeedSeamContract::test_the_runner_calls
#                                _the_seam_once_per_sub_batch.  G: TestGatedSeed
#                                Parity::test_the_runner_reaches_the_seam_per
#                                _padded_sub_batch
