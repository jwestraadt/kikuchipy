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

"""Tests of the opt-in Fourier-Mellin rotation initial guess of the
HREBSD IC-GN engine (``kikuchipy.indexing._hrebsd._fourier_mellin``,
its seam extension in ``._batched``, the CPU route and retry in
``._engine``, the device wiring in ``._gpu`` and the ``fourier_mellin``
keyword of ``run_hrebsd_dic`` and ``EBSD.hrebsd_dic``).

Covers ``specs/2026-09-07-hrebsd-dic/validation.md`` V10, oracles (a)
to (l); (m) is a recorded ledger measurement, never a test.
Requirements D22 govern; the CPU route of D22.10 is the oracle for
every device output (D22.14).

- **Default suite** (no GPU, runs on CI): the switch, its check order,
  forwarding and docstring (a); the angle and the look-up table (b);
  the edge treatment and the reuse of the target spectra (c); the seed
  rows and the failure semantics (d); capture end to end (e, starred
  arms; the rest weekly); rescue without propagation on the V8 ramp
  (f); the acceptance (g); the pre-fit gate (h); the retry (i); the
  CPU route (j); the device runner under the numpy session (k).
- **Locally gated suite** (``cupy_gpu`` fixture, ``-n 0`` only,
  through the PINNED overlay of D21.15): (l).

Layout (D21.14.1, plan 12.4 item 16): the dated pin constants; the
helpers DUPLICATED from ``test_hrebsd_gpu.py`` and
``test_hrebsd_seeding.py`` (test modules cannot import each other
under ``--import-mode=importlib`` and the root ``conftest.py`` is
develop-owned), each copied BY NAME from the current source on
2026-10-08 (V10 cites line ranges at f297867e; the sources have moved
since) and kept identical to it, a drift being a test edit with a
dated comment; the local re-implementation of the D22.3 angle that
checks the premises whatever the production code does; the fixture
builders G1 to G8 of V10; then the default-suite classes, the gated
ones and the mutation map of plan 12 item 4, each after its marker.

"Bitwise" throughout means :func:`numpy.array_equal` with
``equal_nan=True`` on floats (value identity, the Stage A convention).

**The argument lists the failing tests fix** (plan 12 item 1; D22.6,
D22.7 and D21.9.5 freeze the NAMES, this commit freezes the argument
lists beyond them):

``kikuchipy.indexing._hrebsd._fourier_mellin`` (no module-scope import
of ``_engine``, ``_batched`` or cupy):

- constants ``FM_N_THETA = 360``, ``FM_RHO_MIN = 0.02``,
  ``FM_RHO_MAX = 0.40``, ``FM_SEARCH_DEG = 30.0``, ``FM_GATE_DEG =
  1.5``, ``FM_ZERO_TWIST_DEG = 1e-9``; ``FOURIER_MELLIN_VALUES =
  ("off", "auto", "always")``; the keys ``FOURIER_MELLIN_ROUTE_KEY =
  "fourier_mellin_route"``, ``FOURIER_MELLIN_ANGLE_KEY =
  "fourier_mellin_angle"``, ``FOURIER_MELLIN_APPLIED_KEY =
  "fourier_mellin_applied"``; the route codes ``ROUTE_NONE = 0``,
  ``ROUTE_ACCEPT = 1``, ``ROUTE_FORCED = 2``; the seed codes
  ``FOURIER_MELLIN_SEED_TRANSLATION = 0``, ``..._FIRST_PASS = 1``,
  ``..._RETRY = 2``, ``..._NONE = -1``;
  ``FOURIER_MELLIN_PROP_NAMES = ("fourier_mellin_seed",
  "fourier_mellin_angle")``; the message literals
  ``FOURIER_MELLIN_VALUE_MESSAGE``, ``FOURIER_MELLIN_NEIGHBORS_MESSAGE``
  (both ``str.format`` templates with the field ``fourier_mellin``),
  ``FOURIER_MELLIN_XMAP_MESSAGE`` and the warning
  ``FOURIER_MELLIN_NO_TWIST_WARNING``.  The tests assert the frozen
  literals of this module, never the source's constants.
- ``FourierMellinState(lut, reference_profile, box, resident,
  n_theta=FM_N_THETA, rho_min=FM_RHO_MIN, rho_max=FM_RHO_MAX,
  search_deg=FM_SEARCH_DEG)``, a plain container of those fields.
- ``fourier_mellin_lut(sr, sc) -> (indices, weights)``, HOST numpy
  arrays ``(n_theta, n_rho, 4)`` int64 and float64; the four corners
  in the order ``(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 +
  1)`` (row-major flat indices modulo ``(sr, sc)``) is NOT frozen:
  the coordinate pin of V10(b) reconstructs the sample position from
  the weight-averaged unwrapped bin coordinates, whatever the order.
- ``fourier_mellin_hann_stencil(xp, spectra) -> windowed`` (a NEW
  complex128 array, D22.3.1), ``fourier_mellin_profiles(xp, spectra,
  lut) -> (P, n_theta)`` float64 (D22.3.2, D22.3.4) and
  ``fourier_mellin_peak(xp, correlation, search_deg) -> (theta_deg,
  peak)``.
- ``fourier_mellin_angles(ctx, target_spectra, fm_state) ->
  (theta_deg, peak)``; ``fourier_mellin_derotate(ctx, batch,
  fm_state, theta_deg) -> (crops, ok)``; ``fourier_mellin_translate(
  ctx, crops, seed_state) -> rows_t``;
  ``fourier_mellin_partial_row(xp, theta_deg, rows_t) -> rows``;
  ``fourier_mellin_criteria(ctx, batch, fm_state, rows) -> (P,)``;
  ``fourier_mellin_rows(ctx, batch, target_spectra, seed_state, h_t,
  route) -> (rows, angle, applied)``, *route* the ``(P,)`` int8 HOST
  array; ``build_fourier_mellin_state(ctx, state, resident) ->
  FourierMellinState | None``.
- ``twist_about_detector_normal(xmap, detector, point_index,
  reference_index) -> (n,)`` float64 degrees, both index arguments
  ``(n,)`` integer FLAT map indices (one grain reference per point);
  ``fourier_mellin_routes(twist_deg, gate_deg) -> (n,)`` int8.

``kikuchipy.indexing._hrebsd._batched`` (D22.6, purely additive):
``SeedState(bounds, reference_spectrum, precision, upsample_factor,
fourier_mellin=None)`` and ``SeedBatch(targets, coefficients,
pattern_index, extras=None, outputs=None)`` (``outputs`` defaults to a
new empty dictionary).

``kikuchipy.indexing._hrebsd._engine``:

- ``run_hrebsd_dic(..., seed_from_neighbors=False,
  fourier_mellin="off", navigation_mask=None, backend="cpu", ...)``.
- ``FOURIER_MELLIN_PROP_NAMES`` re-exported (the name ``ebsd.py``
  imports, D22.9), and the module object as ``_engine._fourier_mellin``
  (the call-time seam of D22.6: ``run_hrebsd_dic`` calls
  ``_fourier_mellin.twist_about_detector_normal`` and
  ``_fourier_mellin.fourier_mellin_routes``).
- CPU route (D22.10): ``_run_chunks(patterns, fit_indices,
  state_of_point, states, chunksize, *, progressbar, options, h0=None,
  fm_route=None, fm_states=None)``, *fm_route* a ``(n_fit,)`` int8
  host array in fit order (``None`` on ``"off"`` runs), *fm_states* a
  tuple aligned with *states* holding each reference's numpy
  ``SeedState`` with its ``FourierMellinState``, or ``None``;
  ``_fit_chunk(patterns_block, state_index_block, h0_block=None,
  states=(), options=None, route_block=None, fm_states=())`` returns
  ``(n, 12)`` rows when *route_block* is ``None`` and ``(n, 14)``
  otherwise.  ``run_hrebsd_dic`` reaches both through ``_engine``'s
  module globals at call time; the first pass is ONE ``_run_chunks``
  call and the retry pass (D22.8) a SECOND one over the retry subset
  in fit order, every route 2, made only when the subset is not
  empty.
- The packed-row slots on FM runs: ``"width": 14, "fm_angle": 12,
  "fm_applied": 13`` beside the Stage E keys (:data:`ROW_SLOTS_FM`).

``kikuchipy.indexing._hrebsd._gpu`` (D21.9.5 as amended 2026-10-07):
``_run_chunks_gpu(..., row_slots, seed_extras=None)``;
``_vram_model_terms(n_pixels, device_precision, seed_precision, *,
fourier_mellin=False)``, ``_vram_model_bytes(batch_size, n_pixels,
device_precision, seed_precision, *, fourier_mellin=False)``,
``_default_batch_size(free_bytes, n_pixels, device_precision,
seed_precision, *, fourier_mellin=False)``; the session builds a
grain's FM state through ``_fourier_mellin.build_fourier_mellin_state``
(module object, call time) in ``residents()``; the retry's
``_run_chunks_gpu`` call receives the first pass's final B as
``chunksize``.

``kikuchipy.signals.EBSD.hrebsd_dic(..., seed_from_neighbors=False,
fourier_mellin="off", navigation_mask=None, backend="cpu",
chunksize=None, verbose=1)``, forwarding ``fourier_mellin`` by name.

**Placeholders**: every device band and device-side literal, and
every literal that needs the new code path, is ``None`` with a
``FIXME-pin`` comment (the V9 convention): :func:`assert_within` and
:func:`assert_count` fail LOUDLY on ``None``.  Only CPU-side premises
are measured at this gate, through the local re-implementation
(:func:`local_angles` and friends) and the existing CPU engine
(validation.md V10, "recorded results, failing-tests gate").

Written failing before the implementation, at the Stage F
failing-tests gate (2026-10-08): every test that reaches the FM path
fails with ``NotImplementedError("Stage F: not implemented yet")``
from the skeleton.
"""

import ast
import contextlib
import functools
import inspect
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import types
import warnings

import dask.array as da
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd import (
    _batched,
    _engine,
    _fourier_mellin,
    _gpu,
)
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
    zero_mean_normalize,
)
from kikuchipy.indexing._hrebsd._tensors import sample_to_detector_matrix
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# Pattern shapes (copied from ``test_hrebsd_gpu.py``, 2026-10-08): the
# V2 oracle size, the shipped Ni size and the real Si-indent detector
# shape (G3's 512 by 622)
SHAPE_480 = (480, 480)
SHAPE_60 = (60, 60)
SHAPE_RECT = (512, 622)

# The V2 projection centre (G1, G3, G4, G5, G7), and the V8 one (G2,
# G6, G8); copied from ``test_hrebsd_gpu.py``, 2026-10-08
PC_480 = (0.4210, 0.5794, 0.5049)
PC_SEEDING = (0.49, 0.51, 0.5049)

# The generic crystal orientation of grain A and grain B (copied from
# ``test_hrebsd_gpu.py``, 2026-10-08)
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0
ORIENTATION_B_AXIS = (3.0, -1.0, 2.0)
ORIENTATION_B_ANGLE_DEG = 52.0

# F6 and the Ni map of V9 (copied from ``test_hrebsd_gpu.py``,
# 2026-10-08), the V10(a) default-path pin maps
F6_NAVIGATION_SHAPE = (1, 4)
F6_RAMP_ANGLES = (0.0, 0.8, 1.6, 2.4)
F6_MAX_ITERATIONS = 200
NI_NAVIGATION_SHAPE = (3, 3)
NI_REFERENCE = (1, 1)

# The V8(a) ramp map (copied from ``test_hrebsd_seeding.py``
# lines 165-203 and 315 of 2026-10-08, comments shortened): the
# imposed in-plane rotation ramp in degrees along row 0, the
# iteration budget, the map shape, the three isolated points of row 2
# (an 8 degree twist beyond every capture range, a 9 degree tilt about
# the detector x axis that converges in 113 iterations from the phase
# cross-correlation seed, a constant pattern), and the
# same-optimum discriminator (a fraction of the imposed corner
# displacement, never a precision claim)
RAMP_ANGLES = (0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8)
RAMP_STEP_DEG = 0.8
MAX_ITERATIONS = 200
RAMP_NAVIGATION_SHAPE = (3, 7)
RAMP_SIZE = RAMP_NAVIGATION_SHAPE[0] * RAMP_NAVIGATION_SHAPE[1]
RAMP_UNFITTABLE_INDEX = 15
RAMP_RESCUE_INDEX = 17
RAMP_FAILED_INDEX = 19
RAMP_UNFITTABLE_ANGLE = 8.0
RAMP_RESCUE_TILT = 9.0
SAME_OPTIMUM_FRACTION = 0.01

# The frozen public values of D22.1, D22.7 and D22.11 (literals,
# asserted in full; no stage letters)
FOURIER_MELLIN_VALUES = ("off", "auto", "always")
FOURIER_MELLIN_INVALID_VALUES = ("AUTO", "on", "", True, None)
FOURIER_MELLIN_VALUE_MESSAGE = (
    "fourier_mellin {fourier_mellin!r} not in the list of supported values "
    "['off', 'auto', 'always']"
)
FOURIER_MELLIN_NEIGHBORS_MESSAGE = (
    "seed_from_neighbors=True cannot be combined with "
    "fourier_mellin={fourier_mellin!r}; use one seeding strategy"
)
FOURIER_MELLIN_XMAP_MESSAGE = (
    "fourier_mellin='auto' requires the input CrystalMap; pass "
    "fourier_mellin='always' to seed every point without it"
)
FOURIER_MELLIN_NO_TWIST_WARNING = (
    "fourier_mellin='auto' found no rotation about the detector normal in "
    "the crystal map (every point matches its reference orientation); no "
    "point is routed before the fit and only the post-fit retry applies"
)
# The D21.12 literal as amended 2026-10-07 by D22.11 (the same literal
# ``test_hrebsd_gpu.py`` pins, duplicated)
SEED_FROM_NEIGHBORS_GPU_MESSAGE = (
    "seed_from_neighbors=True is not supported with backend='gpu'; use "
    "backend='cpu' for neighbour-seeded propagation, or "
    "seed_from_neighbors=False with fourier_mellin='auto' for large "
    "rotations about the detector normal"
)
BACKEND_ERROR_MESSAGE = (
    "Backend {backend!r} not in the list of supported backends ['cpu', 'gpu']"
)

# The props, keys and codes of D22.6 and D22.9
FOURIER_MELLIN_PROP_NAMES = ("fourier_mellin_seed", "fourier_mellin_angle")
FOURIER_MELLIN_PROP_DTYPES = {
    "fourier_mellin_seed": np.dtype(np.int32),
    "fourier_mellin_angle": np.dtype(np.float64),
}
ROUTE_KEY = "fourier_mellin_route"
ANGLE_KEY = "fourier_mellin_angle"
APPLIED_KEY = "fourier_mellin_applied"
ROUTE_NONE, ROUTE_ACCEPT, ROUTE_FORCED = 0, 1, 2
SEED_TRANSLATION, SEED_FIRST_PASS, SEED_RETRY, SEED_NONE = 0, 1, 2, -1

# The packed-row slots: Stage E's (FM off) and the FM runs' (D22.9)
ROW_SLOTS = {
    "width": 12,
    "residual": 8,
    "iterations": 9,
    "norm_dp": 10,
    "converged": 11,
}
ROW_SLOTS_FM = {**ROW_SLOTS, "width": 14, "fm_angle": 12, "fm_applied": 13}

# The frozen recipe constants of D22.3 and D22.7
FROZEN_N_THETA = 360
FROZEN_RHO_MIN = 0.02
FROZEN_RHO_MAX = 0.40
FROZEN_SEARCH_DEG = 30.0
FROZEN_GATE_DEG = 1.5
FROZEN_ZERO_TWIST_DEG = 1e-9

# The look-up table pins of V10(b): radii and entry counts at the two
# crop shapes (165 radii at 432x432, 175 at 460x560), and the
# coordinate reconstruction band (frozen by V10(b), not MTP)
LUT_SHAPES = {(432, 432): (360, 165, 4), (460, 560): (360, 175, 4)}
LUT_ENTRY_COUNTS = {(432, 432): 237600, (460, 560): 252000}
LUT_COORDINATE_TOL = 1e-12

# The untreated amplitude variant's radial band of the V10(c) lock
# premise (V3/V11 of ledger 128), cycles/px
UNTREATED_RHO_BAND = (0.05, 0.20)

# The projection-link band of V10(h), relative (frozen by V10; spec
# gate 4e-14 to 1.4e-12, ledger 127)
PROJECTION_LINK_RTOL = 1e-10

# ------------------------- The V10 fixtures ------------------------- #

# G1: the V3/V4 deformed-master 480 px oracle at ``PC_480``.  A case is
# a SEQUENCE of detector-frame rotations, ``(axis, degrees)`` steps
# applied in order (``"x"``, ``"y"``, ``"z"``; the matrix is the product
# with the LAST step leftmost), or one ``("rotvec", (w1, w2, w3))``
# step, a rotation vector in degrees (:func:`rotation_fe`).  The
# off-grid twists of V10(b), 0.5 degree grid avoided on purpose
G1_BORDER = 0.05
G1_WIDE_BORDER = 0.15
G1_TWISTS_DEG = (0.0, 0.37, -0.37, 1.83, 2.71, -4.42, 5.29, 9.61, 19.13)
# "6.13 deg about z then 1 deg about x"; "3.27 deg about z then 2 deg
# about y" (V10(b)); the expectation is read from the EXACT imposed
# homography, so the composition convention only defines the fixture
G1_COMBINED = ((("z", 6.13), ("x", 1.0)), (("z", 3.27), ("y", 2.0)))
# V10(d): (twist in degrees, projection centre shift in px), border 0.15
G1_PC_SHIFT_CASES = (
    (8.0, (12.0, -9.0)),
    (-12.0, (10.0, 7.0)),
    (5.0, (-8.0, 5.0)),
    (15.0, (6.0, -4.0)),
)
# V10(e) at border 0.05: twists and rotation vectors (detector frame,
# degrees); the STARRED arms run in the default suite, the rest weekly
G1_CAPTURE_TWISTS = (2.5, 3.0, 4.0, 5.0, -3.0, -5.0)
G1_CAPTURE_TWISTS_STARRED = (2.5, 3.0, 5.0, -3.0)
G1_CAPTURE_ROTVECS = ((2.0, -1.5, 3.0), (-2.0, 2.0, 4.0), (1.0, 1.0, -3.5))
G1_CAPTURE_ROTVECS_STARRED = ((2.0, -1.5, 3.0),)
G1_CAPTURE_MAX_ITERATIONS = 50
# V10(e) at border 0.15 (weekly) and the recorded window-edge arm
G1_WIDE_CAPTURE_TWISTS = (8.0, 10.0, 15.0, 20.0, 25.0, -20.0)
G1_WINDOW_EDGE_TWIST = 30.0
# V10(b) dead band: a constant 8 px cross, columns and rows 236:244, in
# both images (the reference mean), and the same cross as the D4
# ``dead_band`` ``(x0, x1, y0, y1)``
G1_DEAD_BAND = (236, 244, 236, 244)
G1_DEAD_BAND_TWISTS = (3.0, 8.0)

# G3: 512x622 at ``PC_480``, crop 460x560 at border 0.05 (the V9 F3s
# shape), seed level
G3_TWISTS = (2.0, 5.0, 8.0, 15.0, 20.0, -8.0)

# G4: G1 times one detector-fixed background image (an off-centre
# Gaussian plus a linear ramp), noise-free and at a Poisson full-scale
# count of 50, at both filter settings.  The nine cases of ledger 128:
# the seven nonzero off-grid twists and the two combined rotations
G4_CASES = tuple(
    ((("z", t),) for t in (0.37, 1.83, 2.71, -4.42, 5.29, 9.61, 19.13))
) + (G1_COMBINED)
G4_FILTERS = ((None, None), (0.05, None))
G4_NOISES = ("none", "poisson50")
G4_FULL_SCALE_COUNT = 50.0
G4_SEED = 1904
# The background: ``1 + G4_GAUSS_AMPLITUDE * exp(-r**2 / (2 sigma**2))
# + G4_RAMP_AMPLITUDE * column / ncols``, centred at the fractional
# position ``G4_GAUSS_CENTRE`` (column, row) with sigma
# ``G4_GAUSS_SIGMA * nrows``
G4_GAUSS_AMPLITUDE = 4.0
G4_GAUSS_CENTRE = (0.68, 0.30)
G4_GAUSS_SIGMA = 0.30
G4_RAMP_AMPLITUDE = 2.0
# The Kikuchi contrast the background multiplies: a pattern ``p`` enters
# as ``1 + G4_CONTRAST * (p - mean) / std`` (the reference's mean and
# standard deviation), the raw-camera model "smooth background times
# (1 + small contrast)"; an affine intensity map, so the exact
# homography is unchanged.  See the MEASURED block for why the
# multiplied pattern carries this pedestal
G4_CONTRAST = 0.1

# G5: the anchor pair (V10, validation G5), measured at this gate, see
# the MEASURED block below
G5_TWIST_DEG = 2.5
G5_PC_SHIFT_PX = (14.0, -10.0)
G5_SEED = 2205
G5_FILTER_CUTOFFS = (0.05, None)

# G6: the unrelated pair at the V8 PC: the target carries a second
# orientation, 20 degrees about (0, 1, 1) composed onto grain A's on
# the left (a grain boundary; :func:`g6_rotation`)
G6_AXIS = (0.0, 1.0, 1.0)
G6_ANGLE_DEG = 20.0
G6_MAX_ITERATIONS = 200

# G7: gate maps.  Imposed detector-frame twists (V10(h)), the tilted
# detector (tilt, azimuthal, twist), the noise population and the
# rotation of the pure out-of-plane arms
G7_TWISTS_DEG = (0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 4.8, -4.8)
G7_TILTED = (10.0, 4.0, 1.5)
G7_OUT_OF_PLANE_DEG = 3.0
G7_SYMMETRY_TWIST_DEG = 2.5
G7_NOISE_DEG = 0.5
G7_NOISE_TWIST_DEG = 2.5
G7_NOISE_SIZE = 32
G7_NOISE_SEED = 727
G7_PROJECTION_TWISTS = (2.5, -4.8)
# The cubic symmetry operator applied to make the symmetry-equivalent
# copy: 90 degrees about the crystal [001]
G7_SYMMETRY_OPERATOR = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# G8: the retry map on the V8 geometry, ``MAX_ITERATIONS`` 200, one row
# in this column order (V10, validation G8)
G8_NAVIGATION_SHAPE = (1, 7)
G8_REFERENCE = 0
G8_EASY_ZERO = 1
G8_EASY_SMALL = 2
G8_MISLABELLED = 3
G8_UNRELATED = 4
G8_CONSTANT = 5
G8_MASKED = 6
G8_EASY_SMALL_TWIST = 0.8
G8_MISLABELLED_TWIST = 4.0
G8_CONSTANT_VALUE = 7.0

# --------- CPU-side premises, MEASURED at this gate (V10) ----------- #
#
# Every value below was measured on 2026-10-08 (Stage F failing-tests
# gate) on machine A's CPU (i7-13700H, Windows 11 build 26200; worktree
# .venv, CPython 3.13.12, numpy 2.4.6, scipy 1.17.1, numba 0.65.1,
# scikit-image 0.26.0, orix 0.14.2) through the LOCAL re-implementation
# of this module and the existing CPU engine, scratch
# ``measure_premises.py``; recorded in validation.md V10 "recorded
# results, failing-tests gate".  These are premises the tests assert at
# run time, never production pins.

# (b) G1 at border 0.05, default band-pass, local recipe: max |error|
# 0.0134 deg over the nine twists of :data:`G1_TWISTS_DEG` (at 9.61
# deg) and 0.1155 deg over :data:`G1_COMBINED` (the in-plane-axis bias
# of ledger 116); ledger 131 (iv) had 0.0134.  Premise bands at about
# 2x
LOCAL_G1_TWIST_ANGLE_TOL_DEG = 0.03
LOCAL_G1_COMBINED_ANGLE_TOL_DEG = 0.25

# (b) G3, local recipe, signed error in deg per twist: the physical-
# frequency table and the bin-unit (FM3) table.  Both reproduce ledger
# 131 (iv) to 1e-3 deg.  Physical worst 0.0277 (20 deg), premise band
# 0.06; the mutant is at least 0.1959 deg from 5 deg on, 3.3x the band
LOCAL_G3_ANGLE_TOL_DEG = 0.06
LOCAL_G3_ERROR_DEG = {
    2.0: -0.0031,
    5.0: 0.0170,
    8.0: 0.0038,
    15.0: 0.0183,
    20.0: 0.0277,
    -8.0: -0.0147,
}
LOCAL_G3_BIN_UNIT_ERROR_DEG = {
    2.0: -0.0737,
    5.0: -0.1959,
    8.0: -0.3444,
    15.0: -1.0309,
    20.0: -2.2919,
    -8.0: 0.3860,
}

# (c) G4, local recipe WITH the edge treatment, max |error| in deg over
# the nine cases: 0.117 and 0.177 (noise-free, Poisson 50) at (None,
# None), 0.186 and 0.249 at (0.05, None) (spec gate 0.151, 0.252,
# 0.281); premise band 0.5.  WITHOUT it (the untreated amplitude
# variant over :data:`UNTREATED_RHO_BAND`, ledger 128's V11): at (None,
# None) 8 of 9 cases above 1 deg, noise-free and at Poisson 50 (the
# 0.37 deg case is the ninth; worst 19.1 deg, the window-limited lock
# at zero), and 0 of 9 at (0.05, None); the untreated LOG variant gives
# the same 8 of 9.  Pinned AT the measured count (a deterministic CPU
# count).  G4 needed :data:`G4_CONTRAST`: a background multiplying the
# RAW projected pattern (mean 44.9, standard deviation 28.4) scales
# the Kikuchi contrast with it, so no amplitude locked (0 of 9 up to a
# 200x Gaussian); with the pedestal of a 10 per cent contrast the lock
# appears and the treated recipe reproduces the spec-gate scale
LOCAL_G4_ANGLE_TOL_DEG = 0.5
FM_LOCK_PREMISE_COUNT = {"none": 8, "poisson50": 8}

# (b) dead band, local recipe, signed error in deg at 3 and 8 deg:
# -0.0701 and +0.0838 at (None, None), -0.1193 and +0.0841 at (0.05,
# None) (ledger 131 (vi): -0.071, +0.084, -0.120, +0.084)
LOCAL_DEADBAND_ANGLE_TOL_DEG = 0.25

# G5, BUILT at this gate (V10 G5): white fixed-pattern gain and offset
# standard deviations of 0.02 (offset relative to the reference's
# standard deviation), seed ``G5_SEED``, default band-pass, border
# 0.05.  Both run-time premises hold: (i) the D5 seed is EXACTLY (0, 0)
# (``initial_guess`` returns -0.0, -0.0; the clean pair gives (-31.875,
# -0.125)), (ii) the local angle is 2.468 deg for the imposed 2.5
# (error -0.032, within the 0.123 deg spec-gate scale).  Every gain and
# offset in {0, 0.02, 0.05, 0.1, 0.2, 0.4} gave a zero seed and an
# angle error of at most 0.083 deg at both filter settings; 0.02 is
# taken because the outcome premise of (g) is cleanest there: the
# criterion at the seed is 2.1104 for the translation row against
# 0.0145 for the local FM row, the ``"off"`` fit does NOT converge at
# 50 (28.58 px off) and the local FM-seeded fit converges in 4
# iterations to 0.0068 px (at 0.1 the ``"off"`` fit converged, in 8, to
# a wrong optimum 30.7 px away)
G5_GAIN_SD = 0.02
G5_OFFSET_SD = 0.02
G5_SEED_PREMISE = (0.0, 0.0)
LOCAL_G5_ANGLE_TOL_DEG = 0.25

# G6 (V8 PC, default band-pass, budget 200), MEASURED with
# :func:`g6_rotation` (the orix product with the extra rotation on the
# LEFT; the right-hand and inverse compositions gave -12.86 and -12.16
# deg and an FM row the acceptance would KEEP, so they are not the
# ledger's pair): local angle +12.008 deg, criterion at the seed 2.040
# (translation row) against 2.080 (local FM row), so the acceptance
# refuses the FM row; the translation-seeded fit does not converge
# (200 iterations) and keeps a FINITE ``h`` (residual 1.888); a forced
# local-FM-seeded fit does not converge either (residual 1.656) --
# ledger 131 (iii) to the printed digits
G6_PREMISE_CRITERIA = (2.040, 2.080)
G6_PREMISE_RESIDUALS = (1.888, 1.656)

# G8 (budget 200, default band-pass), per column, the ``"off"`` fit
# (``fit_pattern`` from the D5 seed): reference and easy 0 deg converge
# in 1; easy 0.8 deg in 5 (0.0019 px); the mislabelled 4.0 deg twist
# does NOT converge, finite ``h``, 42.85 px off (residual 1.368),
# while the local FM row converges in 3 iterations to 0.0119 px; the
# unrelated pattern as G6.  CONSTANT PATTERN, a premise that does NOT
# hold as V10(f)/(i) word it: under the default band-pass the
# preprocessed constant crop is rounding noise (standard deviation
# 4.0e-16, not 0), so ``initial_guess`` AND the numpy seam return a
# FINITE translation row (0.75, 167.0) px and the local FM angle is
# finite (5.05 deg); only ``fit_pattern`` refuses it (the D2.6
# contract, 0 iterations, NaN ``h``, from any ``h0``).  Under
# ``(None, None)`` the crop is exactly constant and the translation row
# is NaN.  Recorded for the inserted (f)/(i) arms
G8_CONSTANT_TRANSLATION_ROW_FINITE = True

# (e) the ``"off"`` premise through ``run_hrebsd_dic`` on the one-row
# maps of :func:`g1_row_map` at ``max_iterations=50``: EVERY arm fails
# (not converged at 50): border 0.05, twists 2.5, 3, 4, 5, -3, -5 at
# 148.26, 172.88, 221.43, 97.93, 23.11, 164.72 px and the rotation
# vectors at 53.21, 68.46, 33.53 px (ledger 129 to the printed digits);
# border 0.15, twists 8, 10, 15, 20, 25, -20 and 30 at 39.5 to 175.1 px
CAPTURE_OFF_CONVERGED = False

# (f) the V8 ramp premise, ``seed_from_neighbors=False`` at 200: ramp
# columns 0 to 2 converge in 1, 5 and 10 iterations; columns 3 to 6
# (2.4 to 4.8 deg) do not (47.66, 48.70, 42.85, 105.24 px); the 8 deg
# point does not (144.50 px); the 9 deg tilt converges in 113
# iterations to 0.3888 px; the constant pattern gives the D2.6
# contract (0 iterations); bitwise the V8 recorded numbers
RAMP_RESCUE_OFF_ITERATIONS = 113

# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# FIXME-pin placeholders (``None``): every band below needs the new
# code path (seed level or end to end) or the device, and is measured
# at the Stage F implementation gate, pinned at about 2x margin with the
# recipe and the machine ID beside it and, where it kills a mutant, the
# kill separation stated.  The spec-gate numbers quoted are SCALES.

# (b) angle, seed level, deg.  Spec gate 0.123 noise-free, 0.113 with
# noise (ledger 128); must sit below the FM9 flipped-offset error
FM_ANGLE_TOL_DEG = None  # FIXME-pin
# (b) G3, deg.  Spec gate at most 0.027; the bin-unit mutant 0.196 to
# 2.29 from 5 deg (ledger 131 (iv)); a 2x pin of about 0.06 kills FM3
FM_ANGLE_TOL_NONSQUARE_DEG = None  # FIXME-pin
# (b) dead band, deg.  Spec gate 0.12 (ledger 131 (vi))
FM_ANGLE_TOL_DEADBAND_DEG = None  # FIXME-pin
# (c) the stencil against a second FFT, relative.  Spec gate 1.6e-16
FM_STENCIL_RTOL = None  # FIXME-pin
# (c) G4, deg.  Spec gate 0.151 noise-free, 0.252 and 0.281 at Poisson 50
FM_ANGLE_TOL_BACKGROUND_DEG = None  # FIXME-pin
# (d) seed-row corner error, px; between 0.083-0.220 (partial) and
# 0.898-2.655 (``T(t) R``), ledger 125
FM_SEED_TOL_PX = None  # FIXME-pin
# (e) capture, px; iterations at border 0.15; fixed-point gap, px
FM_CAPTURE_TOL_PX = None  # FIXME-pin
FM_CAPTURE_ITERATIONS = None  # FIXME-pin
FM_FIXED_POINT_TOL_PX = None  # FIXME-pin
# (f) the 8 degree ramp point, px (spec gate 0.053, ledger 131 (iii))
FM_RAMP_FAR_TOL_PX = None  # FIXME-pin
# (f) the measured codes and iterations the ramp arms pin
FM_RAMP_RESCUE_ALWAYS_ITERATIONS = None  # FIXME-pin (spec gate 116)
FM_RAMP_ALWAYS_CODES = None  # FIXME-pin (expected 1 on columns 1, 2; 0 at 0)
# (g) G5's FM-seeded outcome against ``"off"`` (MTP)
FM_G5_ITERATIONS = None  # FIXME-pin
# (h) the gate's twist recovery, deg (spec gate 1e-6, ledger 127)
FM_GATE_TWIST_TOL_DEG = None  # FIXME-pin
# (k) the numpy session against the CPU route
FM_NUMPY_ANGLE_TOL_DEG = None  # FIXME-pin
FM_NUMPY_ROW_TOL_PX = None  # FIXME-pin
# (l) device parity and budgets
FM_ANGLE_PARITY_DEG = None  # FIXME-pin
FM_ACCEPT_FLIP_COUNT = None  # FIXME-pin
FM_RETRY_FLIP_COUNT = None  # FIXME-pin
FM_VRAM_P_BOUNDS = None  # FIXME-pin
FM_VRAM_R_BOUNDS = None  # FIXME-pin

# --- the default-suite wall time and the per-class fit counts, recorded
# not asserted, in validation.md V10 ledger (failing-tests gate and
# implementation gate)


# --------------------- The gated-suite fixture ---------------------- #


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (``_cupy_gpu_skip_reason``; V10 cites lines 585-753 at f297867e)
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
def hide_cupy(monkeypatch):
    """Make ``import cupy`` raise ``ImportError`` for one test, the
    stage-(a) failure on a machine that HAS cupy."""
    monkeypatch.setitem(sys.modules, "cupy", None)


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
@pytest.fixture
def fresh_gate(monkeypatch):
    """Reset the HREBSD gate's cached verdict for one test."""
    monkeypatch.setattr(_gpu, "_gate_result", None)


# ------------------------ Assertion helpers ------------------------- #


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py (V10
# cites lines 754-796 at f297867e).  DRIFT 2026-10-08: the placeholder
# message names validation V10 instead of V9
def assert_within(measured, bound, name: str) -> None:
    """Assert ``measured <= bound``, failing loudly and informatively
    while *bound* is an unfilled ``FIXME-pin`` placeholder."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V10, "
            f"measured-then-pinned); measured {measured!r}. Fill it with a dated "
            "value, the recipe and the machine ID, and record it in validation.md"
        )
    assert measured <= bound, f"{name}: {measured} > {bound}"


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (DRIFT 2026-10-08: V10 instead of V9 in the placeholder message)
def assert_count(measured: int, pinned, name: str) -> None:
    """Assert a COUNT equals its pin (the (g) discipline: counts are
    pinned at the measured value, never a fraction), failing loudly
    while *pinned* is an unfilled ``FIXME-pin`` placeholder."""
    if pinned is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V10); "
            f"measured count {measured!r}. Pin it with a dated comment"
        )
    assert measured == pinned, f"{name}: {measured} != {pinned}"


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
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
# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py,
# ``matrix_of`` to ``deformed_pattern`` (V10 cites lines 797-991 at
# f297867e); ``test_hrebsd_seeding.py``'s ``in_plane_fe`` and
# ``deformed_pattern`` have identical bodies and are not copied twice


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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_seeding.py
def out_of_plane_fe(angle_deg):
    """Return the REDUCED detector-frame tensor of an out-of-plane
    tilt about the detector x axis."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]) / cos


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_seeding.py
def orientation_a():
    """Return the generic crystal orientation of the first grain,
    deliberately away from any symmetry position."""
    return Rotation.from_axes_angles((1.0, 2.0, 3.0), np.deg2rad(37.0))


# ------------------ V9 fixtures (duplicated, F1 to F6) --------------- #
# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py: the
# V9 F6 map and the Ni inputs (V10 cites lines 1224-1347 at f297867e),
# and the closure ``seed_case`` needs (``warp_with_skimage``,
# ``random_small_homographies``, ``_read_only``, ``_warped_batch``,
# ``_batch_fixture``, ``f1_batch``, ``f3s_batch``, ``f4_batch``);
# DRIFT 2026-10-08: ``FIXTURE_BUILDERS`` holds only the three names
# ``seed_case`` reads, with the constants they need
F1_SEEDS = (0, 1)
F1_CASES_PER_SEED = 6
F3_SEED = 0
F3_SIZE = 64
F3S_SIZE = 8
F4_SEED = 5
F4_SIZE = 3
SCALE_60 = SHAPE_60[0] / SHAPE_480[0]


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


FIXTURE_BUILDERS = {"F1": f1_batch, "F3s": f3s_batch, "F4": f4_batch}


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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
def run_engine(patterns, navigation_shape, detector, **kwargs) -> dict:
    """Return ``run_hrebsd_dic`` properties at ``verbose=0`` (keyword
    overridable), the non-convergence ``UserWarning`` silenced.  Any
    keyword passes through, ``backend`` and the precisions included."""
    kwargs.setdefault("verbose", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run_hrebsd_dic(patterns, navigation_shape, detector, **kwargs)


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py, the
# numpy-session helpers ``make_state`` to ``run_gpu_numpy`` (V10 cites
# lines 1376-1538 at f297867e)
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py (V10
# cites lines 2165-2194 at f297867e)
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


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
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


# ------------------- The V8 ramp map (duplicated) -------------------- #


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_seeding.py.
# DRIFT 2026-10-08: the seeding module's own ``make_detector()`` (no
# argument, its ``PC``) is written ``make_detector(pc=PC_SEEDING)``
# here, which builds the identical detector
@functools.lru_cache(maxsize=1)
def oracle_reference():
    """Return the undeformed 480 px reference pattern of grain 0."""
    pattern = project_pattern(make_detector(pc=PC_SEEDING), orientation_a())
    pattern.flags.writeable = False
    return pattern


# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_seeding.py.
# DRIFT 2026-10-08: ``make_detector()`` -> ``make_detector(pc=PC_SEEDING)``
# and ``SHAPE`` -> ``SHAPE_480`` (identical values)
@functools.lru_cache(maxsize=1)
def ramp_map():
    """Return the V8(a)/V8(e) map: ``(patterns, mask, exact, corners)``.

    Row 0 is the rotation ramp against the reference at ``(0, 0)``.
    Row 1 is masked out entirely, so that every ramp point has masked
    neighbours and the cascade has to walk ALONG the row.  Row 2 holds
    the three isolated points, each with its whole neighbourhood
    masked: the unfittable one, the rescue one and the failed one.
    """
    detector = make_detector(pc=PC_SEEDING)
    reference = oracle_reference()
    row0, exact = [], []
    for angle in RAMP_ANGLES:
        if angle == 0.0:
            row0.append(reference)
            exact.append(np.zeros(N_HOMOGRAPHY_PARAMETERS))
        else:
            pattern, h = deformed_pattern(detector, orientation_a(), in_plane_fe(angle))
            row0.append(pattern)
            exact.append(h)
    unfittable, h_unfittable = deformed_pattern(
        detector, orientation_a(), in_plane_fe(RAMP_UNFITTABLE_ANGLE)
    )
    rescue, h_rescue = deformed_pattern(
        detector, orientation_a(), out_of_plane_fe(RAMP_RESCUE_TILT)
    )
    # a constant pattern has no variance at all, the D2.6 failure
    # contract, and is what pins ``seed_round`` on a point which is
    # neither masked nor fittable
    failed = np.full(SHAPE_480, 7.0)

    row2 = [reference, unfittable, reference, rescue, reference, failed, reference]
    patterns = np.stack(row0 + [reference] * 7 + row2)
    patterns.flags.writeable = False

    mask = np.zeros(RAMP_NAVIGATION_SHAPE, dtype=bool)
    mask[1] = True
    mask[2] = True
    for column in (1, 3, 5):
        mask[2, column] = False
    mask.flags.writeable = False

    exact = (
        exact
        + [np.zeros(N_HOMOGRAPHY_PARAMETERS)] * 7
        + [
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            h_unfittable,
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            h_rescue,
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
        ]
    )
    corners = subregion_corners(SHAPE_480, pc_pixels_of(detector))
    return patterns, mask, np.stack(exact), corners


# ========== Local re-implementation of the D22.3 angle ============== #
#
# Written here from the requirements text alone, so that every premise
# is checked WHATEVER the production code does (V10(c)); never imported
# from ``_fourier_mellin``.  ``edge_treatment=False`` is the recipe
# without the D22.3.1 window; ``magnitude="amplitude"`` with
# ``rho_band=UNTREATED_RHO_BAND`` and no edge treatment is the
# UNTREATED AMPLITUDE variant of ledger 128 (V11) the background-lock
# premise is stated on (V10(c)); ``bin_units=True`` is the FM3 mutant's
# look-up table (bin units on both axes, ``(fx m, fy m)`` with ``m =
# min(sr, sc)``; it reproduces ledger 131 (iv) to 1e-3 deg)


def rotation_angle_of(h) -> float:
    """Return the in-plane rotation read-out of V10(b) in degrees,
    ``atan2(h21 - h12, 2 + h11 + h22)`` (the V4 sense)."""
    h = np.asarray(h, dtype=np.float64)
    return float(np.rad2deg(np.arctan2(h[3] - h[1], 2.0 + h[0] + h[4])))


@functools.lru_cache(maxsize=None)
def local_lut(
    sr: int, sc: int, bin_units: bool = False, rho_band=(FROZEN_RHO_MIN, FROZEN_RHO_MAX)
) -> tuple:
    """Return ``(indices, weights)``, the D22.3.3 look-up table
    ``(360, n_rho, 4)`` int64 and float64, read-only, over the radii
    *rho_band* (cycles/px; the frozen band by default)."""
    m = min(sr, sc)
    rho_min, rho_max = rho_band
    n_rho = int(np.floor((rho_max - rho_min) * m + 1e-9)) + 1
    theta = np.arange(FROZEN_N_THETA) * np.pi / FROZEN_N_THETA
    rho = rho_min + np.arange(n_rho) / m
    fx = rho[None, :] * np.cos(theta)[:, None]
    fy = rho[None, :] * np.sin(theta)[:, None]
    if bin_units:
        u, v = fx * m, fy * m
    else:
        u, v = fx * sc, fy * sr
    c0 = np.floor(u)
    r0 = np.floor(v)
    du, dv = u - c0, v - r0
    c0 = c0.astype(np.int64)
    r0 = r0.astype(np.int64)
    indices = np.empty((*u.shape, 4), dtype=np.int64)
    weights = np.empty((*u.shape, 4), dtype=np.float64)
    for k, (a, b, w) in enumerate(
        (
            (0, 0, (1 - dv) * (1 - du)),
            (0, 1, (1 - dv) * du),
            (1, 0, dv * (1 - du)),
            (1, 1, dv * du),
        )
    ):
        indices[..., k] = ((r0 + a) % sr) * sc + (c0 + b) % sc
        weights[..., k] = w
    return _read_only(indices), _read_only(weights)


def local_hann_stencil(spectra) -> np.ndarray:
    """Return the D22.3.1 frequency stencil of the periodic Hann window,
    separably on the last two axes, as a NEW complex128 array."""
    x = np.asarray(spectra).astype(np.complex128)
    for axis in (-1, -2):
        x = x / 2 - (np.roll(x, 1, axis=axis) + np.roll(x, -1, axis=axis)) / 4
    return x


def local_profiles(
    spectra,
    *,
    edge_treatment=True,
    bin_units=False,
    magnitude="log",
    rho_band=(FROZEN_RHO_MIN, FROZEN_RHO_MAX),
) -> np.ndarray:
    """Return the ``(P, 360)`` zero-mean unit-norm radial-mean profiles
    of ``(P, sr, sc)`` *spectra* (D22.3.2, D22.3.4); *magnitude* is
    ``"log"`` (the frozen ``log1p``) or ``"amplitude"``."""
    spectra = np.asarray(spectra).astype(np.complex128)
    if spectra.ndim == 2:
        spectra = spectra[None]
    n, sr, sc = spectra.shape
    indices, weights = local_lut(int(sr), int(sc), bool(bin_units), tuple(rho_band))
    if edge_treatment:
        spectra = local_hann_stencil(spectra)
    values = np.abs(spectra).reshape(n, -1)
    if magnitude == "log":
        values = np.log1p(values)
    samples = (values[:, indices] * weights[None]).sum(axis=-1)
    profiles = samples.mean(axis=-1)
    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    return profiles / np.linalg.norm(profiles, axis=1, keepdims=True)


def local_peak(correlation, search_deg=FROZEN_SEARCH_DEG) -> tuple:
    """Return ``(theta_deg, peak)`` of ``(P, n)`` correlations by the
    D22.3.6 rule: windowed argmax, ties to the lowest output index, the
    parabolic offset from the raw circular neighbours."""
    correlation = np.atleast_2d(np.asarray(correlation, dtype=np.float64))
    n_slots, n = correlation.shape
    index = np.arange(n)
    lags = np.where(index < n // 2, index, index - n)
    inside = np.abs(lags) * 180.0 / n <= search_deg
    masked = np.where(inside[None], correlation, -np.inf)
    k = np.argmax(masked, axis=1)
    rows = np.arange(n_slots)
    centre = correlation[rows, k]
    left = correlation[rows, (k - 1) % n]
    right = correlation[rows, (k + 1) % n]
    denominator = left - 2 * centre + right
    with np.errstate(all="ignore"):
        delta = np.where(denominator < 0, (left - right) / (2 * denominator), 0.0)
    return (lags[k] + delta) * 180.0 / n, centre


def local_angles(reference_spectrum, target_spectra, **kwargs) -> tuple:
    """Return ``(theta_deg, peak)`` of every target spectrum against the
    reference spectrum by the D22.3 recipe (keywords of
    :func:`local_profiles`)."""
    reference = local_profiles(reference_spectrum, **kwargs)[0]
    targets = local_profiles(target_spectra, **kwargs)
    correlation = np.real(
        np.fft.ifft(np.conj(np.fft.fft(reference))[None] * np.fft.fft(targets, axis=1))
    )
    return local_peak(correlation)


def _zmn_spectrum(crop) -> np.ndarray:
    """Return ``fft2`` of the zero-mean unit-norm *crop*, NaN where the
    crop carries no contrast (the seam's failure, D21.5)."""
    crop = np.asarray(crop, dtype=np.float64)
    try:
        zmn, _ = zero_mean_normalize(crop.ravel())
    except ValueError:
        return np.full(crop.shape, np.nan, dtype=np.complex128)
    return np.fft.fft2(zmn.reshape(crop.shape))


def local_crop_spectra(state, targets) -> np.ndarray:
    """Return ``(N, sr, sc)`` complex128 ``fft2`` of every target's D5
    crop, preprocessed by *state*'s chain and made zero-mean unit-norm
    (the frozen meaning of the seam's ``target_spectra``), NaN for a
    crop with no contrast."""
    r0, r1, c0, c1 = state.bounds
    preprocessed, _ = preprocessed_targets(state, targets)
    return np.stack([_zmn_spectrum(target[r0:r1, c0:c1]) for target in preprocessed])


def local_reference_spectrum(state) -> np.ndarray:
    """Return the ``(sr, sc)`` complex128 spectrum of the reference's
    zero-mean unit-norm D5 crop."""
    return _zmn_spectrum(state.reference_subregion)


def local_state_angles(state, targets, **kwargs) -> tuple:
    """Return ``(theta_deg, peak)`` of *targets* against *state*."""
    return local_angles(
        local_reference_spectrum(state), local_crop_spectra(state, targets), **kwargs
    )


def local_derotated_crop(state, target, theta_deg) -> np.ndarray:
    """Return the ``(sr, sc)`` de-rotated crop ``D(xi) = target(R xi)``
    over every pixel of the D5 bounding box, ``R`` the rotation by
    *theta_deg* about the grain reference PC (D22.4.1), through the
    engine's bicubic :func:`evaluate` on the float32 coefficients of
    the preprocessed target."""
    r0, r1, c0, c1 = state.bounds
    _, coefficients = preprocessed_targets(state, [target])
    pcx, pcy = float(state.pc_pixels[0]), float(state.pc_pixels[1])
    rows, columns = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    x = columns + 0.5 - pcx
    y = rows + 0.5 - pcy
    theta = np.deg2rad(theta_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    xr = cos * x - sin * y
    yr = sin * x + cos * y
    return evaluate(coefficients[0], xr + pcx - 0.5, yr + pcy - 0.5)


def local_partial_row(theta_deg, row_t) -> np.ndarray:
    """Return the D22.4.3 PARTIAL row ``W0 = R(theta) T(t)``."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    tx, ty = float(row_t[2]), float(row_t[5])
    return np.array([c - 1, -s, c * tx - s * ty, s, c - 1, s * tx + c * ty, 0.0, 0.0])


def local_fm_row(state, target, theta_deg=None, upsample_factor=16) -> tuple:
    """Return ``(row, theta_deg, row_t)``: the FM partial row of one
    target by the local recipe (the angle unless given, the de-rotated
    crop, ``initial_guess`` on it, the partial row)."""
    if theta_deg is None:
        theta_deg = float(local_state_angles(state, [target])[0][0])
    crop = local_derotated_crop(state, target, theta_deg)
    row_t = initial_guess(
        state.reference_subregion, crop, upsample_factor=upsample_factor
    )
    return local_partial_row(theta_deg, row_t), theta_deg, row_t


def local_criterion(state, target, row) -> float:
    """Return the D2.7 criterion at the seed *row* (``fit_pattern`` at
    ``max_iterations=0``), the D22.5 CPU yardstick."""
    return float(fit_pattern(state, target, h0=row, max_iterations=0)["residual"])


def local_phase_peak(state, crop) -> float:
    """Return the phase-correlation peak of the reference's D5 crop
    against *crop* (both zero-mean unit-norm): the maximum of the real
    inverse FFT of the normalised cross-power spectrum, the FM12
    mutant's acceptance yardstick (critic F2, 2026-10-08).  The
    UNnormalised cross-correlation orders G5 the other way, so the
    definition is part of the premise."""
    cross = local_reference_spectrum(state) * np.conj(_zmn_spectrum(crop))
    magnitude = np.abs(cross)
    nonzero = magnitude > 0
    cross = np.where(nonzero, cross / np.where(nonzero, magnitude, 1.0), 0.0)
    return float(np.real(np.fft.ifft2(cross)).max())


def local_phase_peaks(state, target) -> tuple:
    """Return ``(peak_t, peak_fm)``: :func:`local_phase_peak` of the
    preprocessed target crop (the translation row's yardstick) and of
    the crop de-rotated by the local angle (the FM row's)."""
    r0, r1, c0, c1 = state.bounds
    preprocessed, _ = preprocessed_targets(state, [target])
    theta = float(local_state_angles(state, [target])[0][0])
    return (
        local_phase_peak(state, preprocessed[0][r0:r1, c0:c1]),
        local_phase_peak(state, local_derotated_crop(state, target, theta)),
    )


# ======================== Fixture builders ========================== #


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    """Return the detector-frame rotation by *degrees* about *axis*."""
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return in_plane_fe(degrees)


def rotation_fe(spec) -> np.ndarray:
    """Return the detector-frame rotation of a G1 case *spec*: a tuple
    of ``(axis, degrees)`` steps applied in order (the last leftmost),
    or of one ``("rotvec", (w1, w2, w3))`` step in degrees."""
    matrix = np.eye(3)
    for axis, value in spec:
        if axis == "rotvec":
            step = ScipyRotation.from_rotvec(np.deg2rad(np.asarray(value))).as_matrix()
        else:
            step = _axis_rotation(axis, float(value))
        matrix = step @ matrix
    return matrix


def twist_spec(degrees: float) -> tuple:
    """Return the G1 case spec of a pure twist about the detector normal."""
    return (("z", float(degrees)),)


def rotvec_spec(rotation_vector) -> tuple:
    """Return the G1 case spec of a detector-frame rotation vector."""
    return (("rotvec", tuple(float(w) for w in rotation_vector)),)


@functools.lru_cache(maxsize=4)
def g1_reference(shape=SHAPE_480, pc=PC_480) -> np.ndarray:
    """Return the undeformed grain-A pattern at *shape* and *pc*."""
    return _read_only(
        project_pattern(make_detector(shape=shape, pc=pc), orientation_a())
    )


@functools.lru_cache(maxsize=None)
def g1_pair(spec, pc_shift_px=(0.0, 0.0), shape=SHAPE_480, pc=PC_480) -> dict:
    """Return one G1-family pair (G1 at the defaults; G3 at
    ``SHAPE_RECT``; the V8 geometry at ``PC_SEEDING``): the reference
    of :func:`g1_reference` and a target carrying the detector-frame
    rotation of *spec* (:func:`rotation_fe`) and, when given, a
    projection centre shift *pc_shift_px* ``(dx, dy)`` in px (the
    target projected with its PC moved by that much, a pure translation
    of the content).  Keys: ``reference``, ``target``, ``exact`` (the
    EXACT imposed homography ``T(s) H(fe)`` in the reference's
    PC-centred frame), ``detector`` (the reference's), ``pc_px``,
    ``fe``.  Read-only."""
    nrows, ncols = shape
    detector = make_detector(shape=shape, pc=pc)
    fe = rotation_fe(spec)
    shifted_pc = (
        pc[0] + pc_shift_px[0] / ncols,
        pc[1] + pc_shift_px[1] / nrows,
        pc[2],
    )
    target, h = deformed_pattern(
        make_detector(shape=shape, pc=shifted_pc), orientation_a(), fe
    )
    translation = np.eye(3)
    translation[0, 2] = pc_shift_px[0]
    translation[1, 2] = pc_shift_px[1]
    exact = parameters_of(translation @ matrix_of(h))
    return {
        "reference": g1_reference(shape, pc),
        "target": _read_only(target),
        "exact": _read_only(exact),
        "detector": detector,
        "pc_px": _read_only(pc_pixels_of(detector)),
        "fe": _read_only(fe),
    }


def g1_state(pair: dict, **kwargs) -> ReferenceState:
    """Return the :class:`ReferenceState` of a pair's reference,
    keywords as :func:`make_state`."""
    return make_state(np.asarray(pair["reference"]), pair["pc_px"], **kwargs)


def g1_row_map(spec, pc_shift_px=(0.0, 0.0), shape=SHAPE_480, pc=PC_480) -> dict:
    """Return one pair in map form for ``run_hrebsd_dic`` (V10(e): a
    one-row map, the reference at (0, 0) and the target at (0, 1),
    the reference's PC tiled).  Keys: ``patterns`` ``(2, nrows,
    ncols)``, ``navigation_shape`` ``(1, 2)``, ``detector``, ``exact``
    ``(8,)`` (the target's), ``pc_px``."""
    pair = g1_pair(spec, pc_shift_px, shape, pc)
    navigation_shape = (1, 2)
    return {
        "patterns": _read_only(np.stack([pair["reference"], pair["target"]])),
        "navigation_shape": navigation_shape,
        "detector": make_detector(
            shape=shape, pc=pc, navigation_shape=navigation_shape
        ),
        "exact": pair["exact"],
        "pc_px": pair["pc_px"],
    }


@functools.lru_cache(maxsize=None)
def g1_dead_band_pair(twist_deg: float) -> dict:
    """Return the V10(b) dead-band pair: the G1 twist pair with the
    columns and rows of :data:`G1_DEAD_BAND` set to the reference mean
    in BOTH images; run it with ``dead_band=G1_DEAD_BAND``."""
    pair = g1_pair(twist_spec(twist_deg))
    x0, x1, y0, y1 = G1_DEAD_BAND
    value = float(np.mean(pair["reference"]))
    out = dict(pair)
    for key in ("reference", "target"):
        image = np.array(pair[key], dtype=np.float64)
        image[:, x0:x1] = value
        image[y0:y1, :] = value
        out[key] = _read_only(image)
    return out


def g3_pair(twist_deg: float) -> dict:
    """Return the G3 pair: a 512x622 pattern at ``PC_480`` and its
    *twist_deg* twist (crop 460x560 at border 0.05)."""
    return g1_pair(twist_spec(twist_deg), shape=SHAPE_RECT)


@functools.lru_cache(maxsize=2)
def g4_background(shape=SHAPE_480) -> np.ndarray:
    """Return the G4 detector-fixed background image (see the
    ``G4_`` constants), the SAME image for reference and target."""
    nrows, ncols = shape
    rows, columns = np.mgrid[0:nrows, 0:ncols].astype(np.float64)
    cx, cy = G4_GAUSS_CENTRE[0] * ncols, G4_GAUSS_CENTRE[1] * nrows
    sigma = G4_GAUSS_SIGMA * nrows
    gauss = np.exp(-((columns - cx) ** 2 + (rows - cy) ** 2) / (2 * sigma**2))
    return _read_only(
        1.0 + G4_GAUSS_AMPLITUDE * gauss + G4_RAMP_AMPLITUDE * columns / ncols
    )


@functools.lru_cache(maxsize=None)
def g4_pair(case: int, noise: str = "none") -> dict:
    """Return G4 case *case* (an index into :data:`G4_CASES`): the G1
    pair at contrast :data:`G4_CONTRAST` times :func:`g4_background`,
    noise-free (``"none"``) or with
    Poisson noise at :data:`G4_FULL_SCALE_COUNT` (``"poisson50"``; the
    images scaled by one factor so the reference's maximum is the
    count, the reference drawn with seed ``G4_SEED`` and the target
    with ``G4_SEED + 1 + case``).  Keys as :func:`g1_pair`."""
    pair = g1_pair(G4_CASES[case])
    background = g4_background()
    mean = float(np.mean(pair["reference"]))
    sd = float(np.std(pair["reference"]))

    def contrast(pattern):
        return 1.0 + G4_CONTRAST * (np.asarray(pattern) - mean) / sd

    reference = background * contrast(pair["reference"])
    target = background * contrast(pair["target"])
    if noise == "poisson50":
        factor = G4_FULL_SCALE_COUNT / reference.max()
        reference = np.random.default_rng(G4_SEED).poisson(reference * factor)
        target = np.random.default_rng(G4_SEED + 1 + case).poisson(target * factor)
        reference = reference.astype(np.float64)
        target = target.astype(np.float64)
    out = dict(pair)
    out["reference"] = _read_only(reference)
    out["target"] = _read_only(target)
    return out


@functools.lru_cache(maxsize=None)
def g5_pair(gain_sd=None, offset_sd=None) -> dict:
    """Return G5, the anchor pair: the G1 pair of a
    :data:`G5_TWIST_DEG` twist and a :data:`G5_PC_SHIFT_PX` projection
    centre shift, both images carrying the SAME detector-fixed WHITE
    fixed-pattern image (per-pixel gain ``1 + gain_sd * N(0, 1)`` and
    offset ``offset_sd * std(reference) * N(0, 1)``, seed
    :data:`G5_SEED`); the defaults are the MEASURED amplitudes
    :data:`G5_GAIN_SD` and :data:`G5_OFFSET_SD`.  Keys as
    :func:`g1_pair` plus ``clean_reference`` and ``clean_target``."""
    gain_sd = G5_GAIN_SD if gain_sd is None else gain_sd
    offset_sd = G5_OFFSET_SD if offset_sd is None else offset_sd
    pair = g1_pair(twist_spec(G5_TWIST_DEG), G5_PC_SHIFT_PX)
    rng = np.random.default_rng(G5_SEED)
    gain = 1.0 + gain_sd * rng.standard_normal(SHAPE_480)
    offset = (
        offset_sd * float(np.std(pair["reference"])) * rng.standard_normal(SHAPE_480)
    )
    out = dict(pair)
    out["clean_reference"] = pair["reference"]
    out["clean_target"] = pair["target"]
    out["reference"] = _read_only(np.asarray(pair["reference"]) * gain + offset)
    out["target"] = _read_only(np.asarray(pair["target"]) * gain + offset)
    return out


def g6_rotation() -> Rotation:
    """Return G6's target orientation, ``Rotation(G6_AXIS, G6_ANGLE_DEG)
    * orientation_a()`` (the orix product, the extra rotation on the
    LEFT), the composition that reproduces ledger 131 (iii) to the
    printed digits (see the MEASURED block)."""
    extra = Rotation.from_axes_angles(G6_AXIS, np.deg2rad(G6_ANGLE_DEG))
    return extra * orientation_a()


@functools.lru_cache(maxsize=1)
def g6_pair() -> dict:
    """Return G6, the unrelated pair at the V8 PC: grain A's reference
    and a pattern of :func:`g6_rotation`.  Keys ``reference``,
    ``target``, ``detector``, ``pc_px``."""
    detector = make_detector(pc=PC_SEEDING)
    return {
        "reference": oracle_reference(),
        "target": _read_only(project_pattern(detector, g6_rotation())),
        "detector": detector,
        "pc_px": _read_only(pc_pixels_of(detector)),
    }


def g7_detector(tilted: bool = False, navigation_shape=None):
    """Return G7's detector at ``PC_480``: the default (tilt 0) or the
    tilted one (tilt, azimuthal, twist :data:`G7_TILTED`)."""
    tilt, azimuthal, twist = G7_TILTED if tilted else (0.0, 0.0, 0.0)
    pc = np.asarray(PC_480, dtype=np.float64)
    pc = pc[None] if navigation_shape is None else np.tile(pc, (*navigation_shape, 1))
    return kp.detectors.EBSDDetector(
        shape=SHAPE_480,
        binning=1,
        px_size=70.0,
        pc=pc,
        sample_tilt=70.0,
        tilt=tilt,
        azimuthal=azimuthal,
        twist=twist,
    )


def gate_orientation(g_reference, r_det, detector) -> np.ndarray:
    """Return the sample-to-crystal matrix ``g_t = g_r (M^T R_det M)^T``
    of a target whose misorientation from *g_reference* is the
    detector-frame rotation *r_det* (ledger 127), ``M`` the
    :func:`sample_to_detector_matrix` of *detector*."""
    m = sample_to_detector_matrix(detector)
    return np.asarray(g_reference) @ (m.T @ np.asarray(r_det) @ m).T


def gate_xmap(matrices, navigation_shape, phase_id=None, phases=None) -> CrystalMap:
    """Return a crystal map of ``(n, 3, 3)`` sample-to-crystal
    *matrices* (a NaN matrix gives a NaN rotation) over
    *navigation_shape* with unit steps in um; *phase_id* defaults to
    0 everywhere (-1 is not indexed) and *phases* to one Ni phase
    (space group 225)."""
    arrays, size = create_coordinate_arrays(navigation_shape, (1.0, 1.0))
    matrices = np.asarray(matrices, dtype=np.float64).reshape(size, 3, 3)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    data = np.full((size, 4), np.nan)
    safe = np.where(finite[:, None, None], matrices, np.eye(3))
    data[:] = Rotation.from_matrix(safe).data
    data[~finite] = np.nan
    arrays["rotations"] = Rotation(data)
    arrays["phase_id"] = (
        np.zeros(size, dtype=int)
        if phase_id is None
        else np.asarray(phase_id, dtype=int)
    )
    if phases is None:
        phases = PhaseList(Phase(name="ni", space_group=225))
    arrays["phase_list"] = phases
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    return xmap


def g_reference_matrix() -> np.ndarray:
    """Return grain A's sample-to-crystal matrix."""
    return sample_to_crystal_matrix(orientation_a())


@functools.lru_cache(maxsize=2)
def g7_twist_map(tilted: bool = False) -> dict:
    """Return the G7 twist map: one row, the reference at (0, 0)
    followed by one point per :data:`G7_TWISTS_DEG` twist.  Keys
    ``xmap``, ``detector`` (one PC), ``navigation_shape``,
    ``point_index``, ``reference_index`` (flat, the non-reference
    points against point 0) and ``expected`` (deg)."""
    detector = g7_detector(tilted)
    g_r = g_reference_matrix()
    matrices = [g_r] + [
        gate_orientation(g_r, in_plane_fe(t), detector) for t in G7_TWISTS_DEG
    ]
    navigation_shape = (1, len(matrices))
    return {
        "xmap": gate_xmap(np.stack(matrices), navigation_shape),
        "detector": detector,
        "navigation_shape": navigation_shape,
        "point_index": np.arange(1, len(matrices)),
        "reference_index": np.zeros(len(matrices) - 1, dtype=np.int64),
        "expected": np.array(G7_TWISTS_DEG),
    }


# The columns of :func:`g7_special_map`
G7_SPECIAL_SYMMETRY = 1
G7_SPECIAL_ABOUT_X = 2
G7_SPECIAL_ABOUT_Y = 3
G7_SPECIAL_TWIST_SWING = 4
G7_SPECIAL_UNINDEXED = 5
G7_SPECIAL_NAN = 6
G7_SPECIAL_OTHER_PHASE = 7


@functools.lru_cache(maxsize=2)
def g7_special_map(tilted: bool = False) -> dict:
    """Return the G7 special-case map, one row of eight points: the
    reference; a symmetry-equivalent copy of a
    :data:`G7_SYMMETRY_TWIST_DEG` twist (operator
    :data:`G7_SYMMETRY_OPERATOR` applied on the crystal side, raw angle
    about 90 degrees); pure :data:`G7_OUT_OF_PLANE_DEG` rotations about
    the detector x and y axes (twist 0); that twist then the x rotation
    (twist plus swing, twist :data:`G7_SYMMETRY_TWIST_DEG`); an
    unindexed point (phase -1); a NaN rotation; a point of a second
    phase.  Keys as :func:`g7_twist_map`, ``expected`` NaN where the
    gate cannot compute (fail open)."""
    detector = g7_detector(tilted)
    g_r = g_reference_matrix()
    twist = in_plane_fe(G7_SYMMETRY_TWIST_DEG)
    swing = _axis_rotation("x", G7_OUT_OF_PLANE_DEG)
    matrices = [
        g_r,
        G7_SYMMETRY_OPERATOR @ gate_orientation(g_r, twist, detector),
        gate_orientation(g_r, _axis_rotation("x", G7_OUT_OF_PLANE_DEG), detector),
        gate_orientation(g_r, _axis_rotation("y", G7_OUT_OF_PLANE_DEG), detector),
        gate_orientation(g_r, swing @ twist, detector),
        gate_orientation(g_r, twist, detector),
        np.full((3, 3), np.nan),
        gate_orientation(g_r, twist, detector),
    ]
    phase_id = [0, 0, 0, 0, 0, -1, 0, 1]
    phases = PhaseList(names=["ni", "al"], space_groups=[225, 225], point_groups=None)
    navigation_shape = (1, len(matrices))
    return {
        "xmap": gate_xmap(np.stack(matrices), navigation_shape, phase_id, phases),
        "detector": detector,
        "navigation_shape": navigation_shape,
        "point_index": np.arange(1, len(matrices)),
        "reference_index": np.zeros(len(matrices) - 1, dtype=np.int64),
        "expected": np.array(
            [
                G7_SYMMETRY_TWIST_DEG,
                0.0,
                0.0,
                G7_SYMMETRY_TWIST_DEG,
                np.nan,
                np.nan,
                np.nan,
            ]
        ),
    }


@functools.lru_cache(maxsize=2)
def g7_two_grain_map(tilted: bool = False) -> dict:
    """Return the G7 two-grain map, one row of four points: grain A's
    reference, a 2.5 degree twist from it, grain B's reference
    (:func:`rotation_b`) and a -1.5 degree twist from grain B.  Keys as
    :func:`g7_twist_map` plus ``grain_labels`` ``(1, 4)`` and
    ``references`` (flat, per label); ``reference_index`` is each
    point's OWN grain reference."""
    detector = g7_detector(tilted)
    g_a = g_reference_matrix()
    g_b = sample_to_crystal_matrix(rotation_b())
    matrices = [
        g_a,
        gate_orientation(g_a, in_plane_fe(2.5), detector),
        g_b,
        gate_orientation(g_b, in_plane_fe(-1.5), detector),
    ]
    navigation_shape = (1, 4)
    return {
        "xmap": gate_xmap(np.stack(matrices), navigation_shape),
        "detector": detector,
        "navigation_shape": navigation_shape,
        "grain_labels": np.array([[0, 0, 1, 1]]),
        "references": np.array([0, 2]),
        "point_index": np.array([1, 3]),
        "reference_index": np.array([0, 2]),
        "expected": np.array([2.5, -1.5]),
    }


@functools.lru_cache(maxsize=2)
def g7_uniform_map(kind: str = "identity", size: int = 4) -> dict:
    """Return the G7 all-identity map (``kind="identity"``, every
    rotation the identity) or the constant NON-identity map
    (``kind="constant"``, every point at grain A's orientation), one
    row of *size* points on the default detector."""
    matrix = np.eye(3) if kind == "identity" else g_reference_matrix()
    navigation_shape = (1, size)
    return {
        "xmap": gate_xmap(np.stack([matrix] * size), navigation_shape),
        "detector": g7_detector(False),
        "navigation_shape": navigation_shape,
        "point_index": np.arange(1, size),
        "reference_index": np.zeros(size - 1, dtype=np.int64),
    }


@functools.lru_cache(maxsize=2)
def g7_noise_map(tilted: bool = False) -> dict:
    """Return the G7 noise population: the reference then
    :data:`G7_NOISE_SIZE` points of a :data:`G7_NOISE_TWIST_DEG` twist,
    each perturbed by a random rotation of :data:`G7_NOISE_DEG` about a
    random detector-frame axis (seed :data:`G7_NOISE_SEED`); recorded,
    not asserted (V10(h))."""
    detector = g7_detector(tilted)
    g_r = g_reference_matrix()
    rng = np.random.default_rng(G7_NOISE_SEED)
    matrices = [g_r]
    for _ in range(G7_NOISE_SIZE):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        noise = ScipyRotation.from_rotvec(np.deg2rad(G7_NOISE_DEG) * axis).as_matrix()
        matrices.append(
            gate_orientation(g_r, noise @ in_plane_fe(G7_NOISE_TWIST_DEG), detector)
        )
    navigation_shape = (1, len(matrices))
    return {
        "xmap": gate_xmap(np.stack(matrices), navigation_shape),
        "detector": detector,
        "navigation_shape": navigation_shape,
        "point_index": np.arange(1, len(matrices)),
        "reference_index": np.zeros(len(matrices) - 1, dtype=np.int64),
    }


def g7_projected_pair(twist_deg: float, tilted: bool = False) -> tuple:
    """Return ``(projected, constructed)`` of the V10(h) projection
    link: the pattern projected from the G7 orientation ``g_t`` of a
    *twist_deg* twist, and the detector-frame deformation construction
    of the same ``R_det`` (:func:`deformed_pattern`), on the G7
    detector."""
    detector = g7_detector(tilted)
    r_det = in_plane_fe(twist_deg)
    g_t = gate_orientation(g_reference_matrix(), r_det, detector)
    projected = project_pattern(detector, Rotation.from_matrix(g_t))
    constructed, _ = deformed_pattern(detector, orientation_a(), r_det)
    return projected, constructed


@functools.lru_cache(maxsize=1)
def ramp_xmap() -> CrystalMap:
    """Return the G7-built crystal map MATCHING the V8 ramp map (V10(f)
    under ``"auto"``): ramp column ``c`` at the twist ``RAMP_ANGLES[c]``,
    the unfittable point at :data:`RAMP_UNFITTABLE_ANGLE`, the rescue
    point at the :data:`RAMP_RESCUE_TILT` rotation about the detector
    x axis (twist 0), every other point at the reference's own
    orientation, on the V8 detector."""
    detector = make_detector(pc=PC_SEEDING)
    g_r = g_reference_matrix()
    matrices = np.tile(g_r, (RAMP_SIZE, 1, 1))
    for column, angle in enumerate(RAMP_ANGLES):
        matrices[column] = gate_orientation(g_r, in_plane_fe(angle), detector)
    matrices[RAMP_UNFITTABLE_INDEX] = gate_orientation(
        g_r, in_plane_fe(RAMP_UNFITTABLE_ANGLE), detector
    )
    matrices[RAMP_RESCUE_INDEX] = gate_orientation(
        g_r, _axis_rotation("x", RAMP_RESCUE_TILT), detector
    )
    return gate_xmap(matrices, RAMP_NAVIGATION_SHAPE)


def ramp_run(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on the V8 ramp map at reference (0, 0),
    ``seed_from_neighbors=False``, ``MAX_ITERATIONS`` and its mask
    (each overridable), the V10(f) runs."""
    patterns, mask, _, _ = ramp_map()
    kwargs.setdefault("reference", (0, 0))
    kwargs.setdefault("seed_from_neighbors", False)
    kwargs.setdefault("max_iterations", MAX_ITERATIONS)
    kwargs.setdefault("navigation_mask", np.array(mask))
    return run_engine(
        np.asarray(patterns),
        RAMP_NAVIGATION_SHAPE,
        make_detector(pc=PC_SEEDING, navigation_shape=RAMP_NAVIGATION_SHAPE),
        **kwargs,
    )


@functools.lru_cache(maxsize=1)
def ramp_off() -> dict:
    """Return the cached ``"off"`` run of the ramp map, the V10(f)
    premise (treat it read-only)."""
    return ramp_run()


@functools.lru_cache(maxsize=1)
def g8_map() -> dict:
    """Return G8, the retry map on the V8 geometry (V10(i)), one row in
    the ``G8_`` column order: the reference; an easy 0 degree point
    (the reference pattern); an easy :data:`G8_EASY_SMALL_TWIST` twist
    whose xmap orientation is its true one (below the gate); the
    MISLABELLED :data:`G8_MISLABELLED_TWIST` twist whose xmap reads the
    reference's orientation (not routed pre-fit); the G6 unrelated
    pattern, xmap at the reference's orientation; a constant pattern
    (the D2.6 contract); a masked point.  Keys ``patterns``,
    ``navigation_shape``, ``detector`` (per point), ``navigation_mask``,
    ``xmap``, ``exact`` ``(7, 8)`` (NaN where none exists), ``corners``,
    ``max_iterations``."""
    detector = make_detector(pc=PC_SEEDING)
    reference = oracle_reference()
    small = g1_pair(twist_spec(G8_EASY_SMALL_TWIST), pc=PC_SEEDING)
    mislabelled = g1_pair(twist_spec(G8_MISLABELLED_TWIST), pc=PC_SEEDING)
    patterns = np.stack(
        [
            reference,
            reference,
            small["target"],
            mislabelled["target"],
            g6_pair()["target"],
            np.full(SHAPE_480, G8_CONSTANT_VALUE),
            reference,
        ]
    )
    exact = np.full((7, N_HOMOGRAPHY_PARAMETERS), np.nan)
    exact[[G8_REFERENCE, G8_EASY_ZERO, G8_MASKED]] = 0.0
    exact[G8_EASY_SMALL] = small["exact"]
    exact[G8_MISLABELLED] = mislabelled["exact"]
    mask = np.zeros(G8_NAVIGATION_SHAPE, dtype=bool)
    mask[0, G8_MASKED] = True
    g_r = g_reference_matrix()
    matrices = np.tile(g_r, (7, 1, 1))
    matrices[G8_EASY_SMALL] = gate_orientation(
        g_r, in_plane_fe(G8_EASY_SMALL_TWIST), detector
    )
    return {
        "patterns": _read_only(patterns),
        "navigation_shape": G8_NAVIGATION_SHAPE,
        "detector": make_detector(pc=PC_SEEDING, navigation_shape=G8_NAVIGATION_SHAPE),
        "navigation_mask": _read_only(mask),
        "xmap": gate_xmap(matrices, G8_NAVIGATION_SHAPE),
        "exact": _read_only(exact),
        "corners": _read_only(subregion_corners(SHAPE_480, pc_pixels_of(detector))),
        "max_iterations": MAX_ITERATIONS,
    }


def g8_run(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on G8 at reference (0, 0) with its
    mask, xmap and budget (each overridable)."""
    g8 = g8_map()
    kwargs.setdefault("reference", (0, 0))
    kwargs.setdefault("xmap", g8["xmap"])
    kwargs.setdefault("max_iterations", g8["max_iterations"])
    kwargs.setdefault("navigation_mask", np.array(g8["navigation_mask"]))
    return run_engine(
        np.asarray(g8["patterns"]), G8_NAVIGATION_SHAPE, g8["detector"], **kwargs
    )


# ===== DEFAULT (a)-(e) (inserted) =====


# ------------- Helpers of the (a) to (e) classes (2026-10-08) -------- #
#
# Shared by TestFourierMellinSwitch to TestFourierMellinCapture; every
# production call goes through the frozen names of D22.6 and the
# call-time seams (``_fourier_mellin.<name>``, ``_batched.<name>``).


def numpy_fm_seam(state, targets, route=None, *, pattern_index=None):
    """Return a ``types.SimpleNamespace`` of the numpy FM seam on
    *targets* against *state* (one sub-batch holding every target):
    ``ctx``, ``seed_state`` (complex128, upsample 16, its
    ``fourier_mellin`` the reference's :class:`FourierMellinState`
    built by ``build_fourier_mellin_state`` on the numpy subregion
    resident), ``resident``, ``fm_state``, ``batch`` (with the route
    key in ``extras`` when *route* is given) and ``spectra`` (what
    ``seed_spectra`` returned)."""
    ctx = numpy_seed_context()
    seed_state = _batched.build_seed_state(ctx, state)
    resident = _batched.build_resident(ctx, state)
    fm_state = _fourier_mellin.build_fourier_mellin_state(ctx, state, resident)
    seed_state.fourier_mellin = fm_state
    batch = numpy_seed_batch(state, targets, pattern_index)
    if route is not None:
        batch.extras[ROUTE_KEY] = np.asarray(route, dtype=np.int8)
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    return types.SimpleNamespace(
        ctx=ctx,
        seed_state=seed_state,
        resident=resident,
        fm_state=fm_state,
        batch=batch,
        spectra=spectra,
    )


def seam_rows(seam) -> np.ndarray:
    """Return ``_batched.seed_homographies`` on a :func:`numpy_fm_seam`
    (the outputs land in ``seam.batch.outputs``)."""
    return np.asarray(
        _batched.seed_homographies(seam.ctx, seam.batch, seam.spectra, seam.seed_state)
    )


def stage_e_rows(state, targets) -> np.ndarray:
    """Return the Stage E translation rows ``h_T`` of the numpy seam (no
    FM state), the bitwise yardstick of D22.6."""
    return np.asarray(numpy_seed_rows(state, targets)[0])


def production_angles(state, targets) -> tuple:
    """Return ``(theta_deg, peak)`` of ``fourier_mellin_angles`` on the
    numpy seam, host arrays."""
    seam = numpy_fm_seam(state, targets)
    theta, peak = _fourier_mellin.fourier_mellin_angles(
        seam.ctx, seam.spectra, seam.fm_state
    )
    return np.asarray(theta), np.asarray(peak)


def pair_targets(specs, **kwargs) -> tuple:
    """Return ``(pairs, targets, expected)`` of G1-family *specs*:
    the pairs, their stacked targets and the expected angles
    ``rotation_angle_of(exact)`` in deg (keywords of :func:`g1_pair`)."""
    pairs = [g1_pair(spec, **kwargs) for spec in specs]
    targets = np.stack([np.asarray(p["target"]) for p in pairs])
    expected = np.array([rotation_angle_of(p["exact"]) for p in pairs])
    return pairs, targets, expected


def periodic_hann(n: int) -> np.ndarray:
    """Return the length-*n* periodic Hann window ``1/2 - cos(2 pi k /
    n) / 2``, whose DFT is the 3-tap stencil of D22.3.1."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def windowed_crop_spectra(state, targets) -> np.ndarray:
    """Return ``fft2`` of every target's periodic-Hann-windowed
    zero-mean unit-norm D5 crop, the oracle of the V10(c) stencil arm
    (a second 2D FFT, which the production stencil avoids)."""
    r0, r1, c0, c1 = state.bounds
    window = periodic_hann(r1 - r0)[:, None] * periodic_hann(c1 - c0)[None, :]
    preprocessed, _ = preprocessed_targets(state, targets)
    out = []
    for target in preprocessed:
        crop = target[r0:r1, c0:c1]
        zmn, _ = zero_mean_normalize(crop.ravel())
        out.append(np.fft.fft2(zmn.reshape(crop.shape) * window))
    return np.stack(out)


def relative_error(a, b) -> float:
    """Return ``||a - b|| / ||b||`` (Frobenius over every axis)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def local_derotated_crop_about(state, target, theta_deg, centre=(0.0, 0.0)):
    """Return the D5 bounding-box crop de-rotated by *theta_deg* about
    *centre* (PC-centred px; ``(0, 0)`` is the grain reference PC and
    gives :func:`local_derotated_crop`), ``D(xi) = target(R (xi - c) +
    c)``: the FM43 mutant's crop for the detector centre."""
    r0, r1, c0, c1 = state.bounds
    _, coefficients = preprocessed_targets(state, [target])
    pcx, pcy = float(state.pc_pixels[0]), float(state.pc_pixels[1])
    rows, columns = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    x = columns + 0.5 - pcx - centre[0]
    y = rows + 0.5 - pcy - centre[1]
    theta = np.deg2rad(theta_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    xr = cos * x - sin * y + centre[0]
    yr = sin * x + cos * y + centre[1]
    return evaluate(coefficients[0], xr + pcx - 0.5, yr + pcy - 0.5)


def local_seed_row_errors(state, pair, corners) -> dict:
    """Return the local recipe's corner errors (px) on one V10(d) pair:
    ``partial`` (the frozen ``R(theta) T(t)``), ``reversed`` (the FM2
    mutant ``T(t) R(theta)`` from the same ``t``), ``centre`` (the
    FM43 mutant: de-rotation about the detector centre, then the
    partial row) and ``reused`` (the FM38 mutant: ``R(theta) T(t_T)``,
    the translation of the un-de-rotated crop)."""
    target = np.asarray(pair["target"])
    exact = np.asarray(pair["exact"])
    row, theta, row_t = local_fm_row(state, target)
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    tx, ty = float(row_t[2]), float(row_t[5])
    reversed_row = np.array([c - 1, -s, tx, s, c - 1, ty, 0.0, 0.0])
    nrows, ncols = target.shape
    centre = (
        ncols / 2 - float(state.pc_pixels[0]),
        nrows / 2 - float(state.pc_pixels[1]),
    )
    crop = local_derotated_crop_about(state, target, theta, centre)
    centre_t = initial_guess(state.reference_subregion, crop, upsample_factor=16)
    centre_row = local_partial_row(theta, centre_t)
    # FM38: the translation read from the REUSED target spectrum (the
    # un-de-rotated crop's ``initial_guess``, i.e. ``h_T``'s), composed
    # with ``R(theta)`` (critic F9, 2026-10-08)
    r0, r1, c0, c1 = state.bounds
    preprocessed, _ = preprocessed_targets(state, [target])
    reused_t = initial_guess(
        state.reference_subregion,
        preprocessed[0][r0:r1, c0:c1],
        upsample_factor=16,
    )
    reused_row = local_partial_row(theta, reused_t)
    return {
        "partial": recovery_error(row, exact, corners),
        "reversed": recovery_error(reversed_row, exact, corners),
        "centre": recovery_error(centre_row, exact, corners),
        "reused": recovery_error(reused_row, exact, corners),
    }


def forbid_before_fm_checks(monkeypatch) -> list:
    """Replace the D21.2 gate, ``resolve_reference``, ``ReferenceState``,
    the pattern gather and the session factory by recorders that raise,
    and return the list of names that ran (D22.1: every FM check fires
    before all of them)."""
    calls = []

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} ran before the Fourier-Mellin checks")

        return call

    monkeypatch.setattr(_engine, "_verify_gpu_or_raise", forbidden("gate"))
    monkeypatch.setattr(_engine, "resolve_reference", forbidden("resolve"))
    monkeypatch.setattr(_engine, "ReferenceState", forbidden("state"))
    monkeypatch.setattr(_engine, "_gather", forbidden("gather"))
    monkeypatch.setattr(_gpu, "_make_session", forbidden("session"))
    return calls


def forbid_fm_work(monkeypatch) -> list:
    """Replace every FM entry point of ``_fourier_mellin`` (the state,
    its builder, the branch and the gate) by recorders that raise, and
    return the list of names that ran (FM16: none on ``"off"``)."""
    calls = []

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} ran on a fourier_mellin='off' run")

        return call

    for name in (
        "FourierMellinState",
        "build_fourier_mellin_state",
        "fourier_mellin_lut",
        "fourier_mellin_angles",
        "fourier_mellin_rows",
        "twist_about_detector_normal",
        "fourier_mellin_routes",
    ):
        monkeypatch.setattr(_fourier_mellin, name, forbidden(name))
    return calls


def f6_run(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on F6 at reference (0, 0) and its
    budget (keywords pass through)."""
    f6 = f6_map()
    kwargs.setdefault("reference", (0, 0))
    kwargs.setdefault("max_iterations", f6["max_iterations"])
    return run_engine(
        np.asarray(f6["patterns"]), f6["navigation_shape"], f6["detector"], **kwargs
    )


def f4_call(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on the cheap 60 px F4 map (keywords pass
    through), the precedent's check-order fixture."""
    f4 = f4_batch()
    kwargs.setdefault("reference", f4["reference_index"])
    return run_engine(
        np.asarray(f4["patterns"]), f4["navigation_shape"], f4["map_detector"], **kwargs
    )


def ni_engine_inputs() -> tuple:
    """Return ``(patterns, navigation_shape, detector)`` of the shipped
    Ni map for a direct engine call (default step sizes)."""
    signal, _, detector = ni_inputs()
    patterns = np.asarray(signal.data, dtype=np.float64).reshape(-1, *SHAPE_60)
    return patterns, NI_NAVIGATION_SHAPE, detector


@functools.lru_cache(maxsize=None)
def capture_off(spec, border) -> dict:
    """Return the cached ``"off"`` run of a V10(e) one-row map at
    *border* and ``G1_CAPTURE_MAX_ITERATIONS``, the run-time premise
    (treat it read-only)."""
    row = g1_row_map(spec)
    return run_engine(
        np.asarray(row["patterns"]),
        row["navigation_shape"],
        row["detector"],
        reference=(0, 0),
        border=border,
        max_iterations=G1_CAPTURE_MAX_ITERATIONS,
    )


def capture_run(spec, border, **kwargs) -> dict:
    """Return ``run_hrebsd_dic`` with ``fourier_mellin="always"`` (no
    ``xmap``) on the V10(e) one-row map of *spec* at *border* and
    ``G1_CAPTURE_MAX_ITERATIONS`` (keywords pass through)."""
    row = g1_row_map(spec)
    kwargs.setdefault("fourier_mellin", "always")
    return run_engine(
        np.asarray(row["patterns"]),
        row["navigation_shape"],
        row["detector"],
        reference=(0, 0),
        border=border,
        max_iterations=G1_CAPTURE_MAX_ITERATIONS,
        **kwargs,
    )


def capture_corners(border) -> np.ndarray:
    """Return the subregion corners of the G1 geometry at *border*."""
    pc_px = pc_pixels_of(make_detector())
    return subregion_corners(SHAPE_480, pc_px, border=border)


# ===================== V10(a) the switch =========================== #


class TestFourierMellinSwitch:
    """V10(a), D22.1, D22.9, D22.11, D22.16: the ``fourier_mellin``
    keyword, its bitwise ``"off"`` default on both backends, the frozen
    check order (backend string, value, the D22.11 combination raise,
    ``"auto"`` without ``xmap``, then every ``"gpu"``-only check and the
    gate), the forwarding of ``EBSD.hrebsd_dic``, the public props and
    the docstring.

    Mutants and their designed killers:
    FM16 (FM state built, ``extras``/``outputs`` written or rows widened
    on ``"off"``): :meth:`test_off_builds_no_state_and_writes_nothing_cpu`
    and :meth:`test_off_builds_no_state_and_writes_nothing_numpy_session`;
    FM33 (props written on ``"off"``): the four ``off_equals`` arms and
    :meth:`test_the_public_result_carries_the_props_under_always`;
    FM34 (the D22.11 raise missing):
    :meth:`test_the_combination_raises_on_both_backends`;
    FM35 (``"auto"`` without ``xmap`` silently seeding every point):
    :meth:`test_auto_without_xmap_raises_at_the_engine`;
    FM41 (keyword not forwarded):
    :meth:`test_public_method_forwards_fourier_mellin`;
    FM42 (``ebsd.py`` prop loop not extended):
    :meth:`test_the_public_result_carries_the_props_under_always`."""

    # ------------------------------ default path ------------------ #

    def test_off_equals_no_keyword_bitwise_on_f6(self):
        explicit = f6_run(fourier_mellin="off")
        implicit = f6_run()
        assert_properties_bitwise(explicit, implicit)
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(explicit)

    def test_off_equals_no_keyword_bitwise_on_the_ni_map(self):
        explicit = run_ni(fourier_mellin="off")
        implicit = run_ni()
        assert set(explicit.prop) == set(implicit.prop)
        assert_properties_bitwise(dict(explicit.prop), dict(implicit.prop))
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(explicit.prop)

    @pytest.mark.weekly
    def test_off_equals_no_keyword_on_f6_through_the_numpy_session(self, monkeypatch):
        # weekly: two F6 runs on the numpy twin cost 207 s at -n 2 on
        # machine A (2026-10-08, failing-tests gate); the default suite
        # pins the numpy session on the Ni map below
        explicit, _ = run_gpu_numpy(
            monkeypatch,
            np.asarray(f6_map()["patterns"]),
            F6_NAVIGATION_SHAPE,
            f6_map()["detector"],
            reference=(0, 0),
            max_iterations=F6_MAX_ITERATIONS,
            fourier_mellin="off",
        )
        implicit, _ = run_gpu_numpy(
            monkeypatch,
            np.asarray(f6_map()["patterns"]),
            F6_NAVIGATION_SHAPE,
            f6_map()["detector"],
            reference=(0, 0),
            max_iterations=F6_MAX_ITERATIONS,
        )
        assert_properties_bitwise(explicit, implicit)
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(explicit)

    def test_off_equals_no_keyword_on_ni_through_the_numpy_session(self, monkeypatch):
        patterns, navigation_shape, detector = ni_engine_inputs()
        ni_explicit, _ = run_gpu_numpy(
            monkeypatch,
            patterns,
            navigation_shape,
            detector,
            reference=NI_REFERENCE,
            fourier_mellin="off",
        )
        ni_implicit, _ = run_gpu_numpy(
            monkeypatch, patterns, navigation_shape, detector, reference=NI_REFERENCE
        )
        assert_properties_bitwise(ni_explicit, ni_implicit)
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(ni_explicit)

    def test_off_builds_no_state_and_writes_nothing_cpu(self, monkeypatch):
        # FM16 on the CPU: with "off" (and an xmap present, so a gate
        # COULD run) no FM name is reached, the CPU runner is called
        # ONCE (no retry pass) with no route and returns 12-wide rows
        calls = forbid_fm_work(monkeypatch)
        runs = []
        real_run_chunks = _engine._run_chunks

        def run_chunks_spy(*args, **kwargs):
            packed = real_run_chunks(*args, **kwargs)
            runs.append(
                (kwargs.get("fm_route"), kwargs.get("fm_states"), packed.shape[1])
            )
            return packed

        monkeypatch.setattr(_engine, "_run_chunks", run_chunks_spy)
        xmap = run_ni(fourier_mellin="off")
        assert calls == []
        assert runs == [(None, None, ROW_SLOTS["width"])]
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(xmap.prop)

    def test_off_builds_no_state_and_writes_nothing_numpy_session(self, monkeypatch):
        # FM16 on the device runner: every seed state carries no FM
        # state, ``extras`` and ``outputs`` stay empty, the session's
        # rows keep the Stage E width
        calls = forbid_fm_work(monkeypatch)
        records = []
        real_rows = _batched.seed_homographies

        def rows_spy(ctx, batch, target_spectra, seed_state):
            rows = real_rows(ctx, batch, target_spectra, seed_state)
            records.append(
                (seed_state.fourier_mellin, dict(batch.extras), dict(batch.outputs))
            )
            return rows

        monkeypatch.setattr(_batched, "seed_homographies", rows_spy)
        properties, _ = run_gpu_numpy(
            monkeypatch,
            np.asarray(f4_batch()["patterns"]),
            f4_batch()["navigation_shape"],
            f4_batch()["map_detector"],
            reference=f4_batch()["reference_index"],
            fourier_mellin="off",
        )
        assert calls == []
        assert records, "the seam never ran"
        for fm_state, extras, outputs in records:
            assert fm_state is None
            assert extras == {}
            assert outputs == {}
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(properties)

    # ------------------------------ values ------------------------ #

    @pytest.mark.parametrize("bad", FOURIER_MELLIN_INVALID_VALUES)
    def test_invalid_values_raise_the_frozen_message(self, monkeypatch, bad):
        calls = forbid_before_fm_checks(monkeypatch)
        with pytest.raises(ValueError) as info:
            f4_call(fourier_mellin=bad)
        assert str(info.value) == FOURIER_MELLIN_VALUE_MESSAGE.format(
            fourier_mellin=bad
        )
        assert calls == []

    def test_an_invalid_value_raises_through_the_signal_method(self, monkeypatch):
        calls = forbid_before_fm_checks(monkeypatch)
        with pytest.raises(ValueError) as info:
            run_ni(fourier_mellin="on")
        assert str(info.value) == FOURIER_MELLIN_VALUE_MESSAGE.format(
            fourier_mellin="on"
        )
        assert calls == []

    # --------------------------- check order ---------------------- #

    @pytest.mark.parametrize("backend", ["cpu", "gpu"])
    @pytest.mark.parametrize("value", ["auto", "always"])
    def test_the_combination_raises_on_both_backends(self, monkeypatch, backend, value):
        # FM34: the D22.11 ValueError, literal in full, before the gate,
        # reference resolution, every state and any pattern read; under
        # "gpu" it precedes the D21.12 NotImplementedError
        calls = forbid_before_fm_checks(monkeypatch)
        with pytest.raises(ValueError) as info:
            f4_call(backend=backend, seed_from_neighbors=True, fourier_mellin=value)
        assert str(info.value) == FOURIER_MELLIN_NEIGHBORS_MESSAGE.format(
            fourier_mellin=value
        )
        assert calls == []

    def test_the_combination_raises_through_the_signal_method(self, monkeypatch):
        calls = forbid_before_fm_checks(monkeypatch)
        with pytest.raises(ValueError) as info:
            run_ni(backend="gpu", seed_from_neighbors=True, fourier_mellin="always")
        assert str(info.value) == FOURIER_MELLIN_NEIGHBORS_MESSAGE.format(
            fourier_mellin="always"
        )
        assert calls == []

    def test_the_checks_run_in_the_frozen_order(self, monkeypatch):
        # backend string > value > combination > "auto" without xmap >
        # the "gpu"-only precision checks > the D21.12 raise > the gate;
        # each shown to win over every later one
        calls = forbid_before_fm_checks(monkeypatch)
        with pytest.raises(ValueError) as info:
            f4_call(backend="cuda", fourier_mellin="AUTO", seed_from_neighbors=True)
        assert str(info.value) == BACKEND_ERROR_MESSAGE.format(backend="cuda")
        with pytest.raises(ValueError) as info:
            f4_call(
                backend="gpu",
                fourier_mellin="AUTO",
                seed_from_neighbors=True,
                device_precision="bogus",
            )
        assert str(info.value) == FOURIER_MELLIN_VALUE_MESSAGE.format(
            fourier_mellin="AUTO"
        )
        with pytest.raises(ValueError) as info:
            f4_call(
                backend="gpu",
                fourier_mellin="auto",
                seed_from_neighbors=True,
                device_precision="bogus",
            )
        assert str(info.value) == FOURIER_MELLIN_NEIGHBORS_MESSAGE.format(
            fourier_mellin="auto"
        )
        with pytest.raises(ValueError) as info:
            f4_call(backend="gpu", fourier_mellin="auto", device_precision="bogus")
        assert str(info.value) == FOURIER_MELLIN_XMAP_MESSAGE
        with pytest.raises(ValueError, match="device_precision"):
            f4_call(backend="gpu", fourier_mellin="always", device_precision="bogus")
        assert calls == []

    def test_auto_without_xmap_raises_at_the_engine(self, monkeypatch):
        # FM35: "auto" never silently seeds every point without the map
        calls = forbid_before_fm_checks(monkeypatch)
        for backend in ("cpu", "gpu"):
            with pytest.raises(ValueError) as info:
                f4_call(backend=backend, fourier_mellin="auto")
            assert str(info.value) == FOURIER_MELLIN_XMAP_MESSAGE
        assert calls == []

    def test_always_without_xmap_runs(self):
        properties = f4_call(fourier_mellin="always")
        n = int(np.prod(f4_batch()["navigation_shape"]))
        for name in FOURIER_MELLIN_PROP_NAMES:
            assert name in properties, name
            values = np.asarray(properties[name])
            assert values.dtype == FOURIER_MELLIN_PROP_DTYPES[name], name
            assert values.shape == (n,), name
        assert np.asarray(properties["converged"]).all()
        assert set(np.asarray(properties["fourier_mellin_seed"]).tolist()) <= {
            SEED_TRANSLATION,
            SEED_FIRST_PASS,
        }

    # ------------------------ public method ----------------------- #

    def test_public_method_forwards_fourier_mellin(self, monkeypatch):
        # FM41: the signal method passes the keyword by name for each of
        # the three values (its own module-scope binding is replaced)
        seen = {}

        class Forwarded(Exception):
            pass

        def spy(*args, **kwargs):
            seen.update(kwargs)
            raise Forwarded

        monkeypatch.setattr(
            sys.modules["kikuchipy.signals.ebsd"], "run_hrebsd_dic", spy
        )
        for value in FOURIER_MELLIN_VALUES:
            seen.clear()
            with pytest.raises(Forwarded):
                run_ni(fourier_mellin=value)
            assert seen.get("fourier_mellin") == value

    def test_the_public_result_carries_the_props_under_always(self):
        # FM42 (the prop loop extended) and FM33 (never on "off")
        xmap = run_ni(fourier_mellin="always")
        n = int(np.prod(NI_NAVIGATION_SHAPE))
        for name in FOURIER_MELLIN_PROP_NAMES:
            assert name in xmap.prop, name
            values = np.asarray(xmap.prop[name])
            assert values.dtype == FOURIER_MELLIN_PROP_DTYPES[name], name
            assert values.shape == (n,), name
        codes = np.asarray(xmap.prop["fourier_mellin_seed"])
        assert set(codes.tolist()) <= {
            SEED_TRANSLATION,
            SEED_FIRST_PASS,
            SEED_RETRY,
            SEED_NONE,
        }
        off = run_ni(fourier_mellin="off")
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(off.prop)

    # --------------------------- signatures ----------------------- #

    def test_the_engine_re_exports_the_prop_names(self):
        assert tuple(_engine.FOURIER_MELLIN_PROP_NAMES) == FOURIER_MELLIN_PROP_NAMES
        assert _engine._fourier_mellin is _fourier_mellin

    @pytest.mark.parametrize(
        "function", [run_hrebsd_dic, kp.signals.EBSD.hrebsd_dic], ids=["engine", "ebsd"]
    )
    def test_the_keyword_sits_in_its_frozen_slot(self, function):
        parameters = inspect.signature(function).parameters
        names = list(parameters)
        position = names.index("fourier_mellin")
        assert names[position - 1] == "seed_from_neighbors"
        assert names[position + 1] == "navigation_mask"
        assert parameters["fourier_mellin"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["fourier_mellin"].default == "off"

    def test_the_docstring_documents_fourier_mellin(self):
        # D22.16: the parameter entry (values, default, the gate and its
        # crystal map, the retry, the acceptance, rotation only), Notes
        # with both capture numbers, the two props, the Raises and
        # Warns entries, the rewritten Limitations paragraph, and no
        # stage letter in public text
        docstring = kp.signals.EBSD.hrebsd_dic.__doc__
        # by stripped lines: CPython 3.13 dedents docstrings
        lines = docstring.split("Parameters")[1].split("Returns")[0].splitlines()
        names = [line.strip() for line in lines]
        assert "fourier_mellin" in names
        start = names.index("fourier_mellin") + 1
        entry = " ".join(
            " ".join(lines[start : names.index("navigation_mask")]).split()
        )
        for token in ('"off"', '"auto"', '"always"', "Default", "detector normal"):
            assert token in entry, token
        for token in ("xmap", "retr", "accept", "rotation"):
            assert token in entry.lower() or token in entry, token
        raises = " ".join(docstring.split("Raises")[1].split("Warns")[0].split())
        assert "fourier_mellin" in raises
        assert "seed_from_neighbors" in raises
        warns = " ".join(docstring.split("Warns")[1].split("See Also")[0].split())
        assert "fourier_mellin" in warns
        notes = " ".join(docstring.split("Notes")[1].split())
        for token in ("30 degree", "23.8", "5.6", *FOURIER_MELLIN_PROP_NAMES):
            assert token in notes, token
        limitations = notes.split("**Limitations")[1].split("**", 2)[1]
        limitations = limitations.split("**")[0]
        assert "translation only" in limitations
        assert "fourier_mellin" in limitations
        assert "not built" not in limitations
        assert re.search(r"\bStage [A-Z]\b", docstring) is None


# ===================== V10(b) the angle ============================ #


class TestFourierMellinAngle:
    """V10(b), D22.3: the FM angle at seed level, no fit, through
    ``fourier_mellin_angles``, ``fourier_mellin_lut`` and
    ``fourier_mellin_peak`` alone; the local re-implementation asserts
    each fixture's premise first.

    Mutants and their designed killers:
    FM3 (a bin-unit look-up table):
    :meth:`test_the_lut_reconstructs_physical_frequency` (PRIMARY, the
    460x560 shape; at 432x432 the two tables coincide) and
    :meth:`test_g3_non_square_seed_level` (second, from 5 deg on with at
    least 3x separation, ledger 132 (ii));
    FM4 (angles over [0, 2 pi)):
    :meth:`test_the_lut_reconstructs_physical_frequency` and the sweep
    :meth:`test_g1_twists_seed_level`;
    FM8 (the search window dropped):
    :meth:`test_the_peak_honours_the_search_window`;
    FM9 (the parabolic offset's sign flipped): the off-grid sweep
    :meth:`test_g1_twists_seed_level`, ``FM_ANGLE_TOL_DEG`` asserted
    below the locally computed flipped-offset error."""

    # ------------------------ look-up table ----------------------- #

    @pytest.mark.parametrize(
        "shape", sorted(LUT_SHAPES), ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_the_lut_shape_dtype_and_entry_count(self, shape):
        sr, sc = shape
        indices, weights = _fourier_mellin.fourier_mellin_lut(sr, sc)
        indices = np.asarray(indices)
        weights = np.asarray(weights)
        assert indices.dtype == np.int64
        assert weights.dtype == np.float64
        assert indices.shape == LUT_SHAPES[shape]
        assert weights.shape == LUT_SHAPES[shape]
        assert indices.size == LUT_ENTRY_COUNTS[shape]
        assert indices.min() >= 0 and indices.max() < sr * sc
        assert np.all(weights >= 0.0)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
        # the premise: the local table has the same frozen shape
        local_indices, _ = local_lut(sr, sc)
        assert local_indices.shape == LUT_SHAPES[shape]

    @pytest.mark.parametrize(
        "shape", sorted(LUT_SHAPES), ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_the_lut_reconstructs_physical_frequency(self, shape):
        # FM3 PRIMARY, FM4: the weight-averaged (column, row) of each
        # entry's four bins, unwrapped, is ``(rho_j cos theta_k sc,
        # rho_j sin theta_k sr)`` with theta over [0, pi) and the
        # D22.3.3 radii, within LUT_COORDINATE_TOL
        sr, sc = shape
        indices, weights = _fourier_mellin.fourier_mellin_lut(sr, sc)
        indices = np.asarray(indices)
        weights = np.asarray(weights)
        n_theta, n_rho, _ = LUT_SHAPES[shape]
        m = min(sr, sc)
        theta = np.arange(n_theta) * np.pi / n_theta
        rho = FROZEN_RHO_MIN + np.arange(n_rho) / m
        u = rho[None, :] * np.cos(theta)[:, None] * sc
        v = rho[None, :] * np.sin(theta)[:, None] * sr
        rows, columns = np.divmod(indices, sc)
        # unwrap every bin to the period nearest the expected coordinate
        columns = columns - sc * np.round((columns - u[..., None]) / sc)
        rows = rows - sr * np.round((rows - v[..., None]) / sr)
        column_hat = (weights * columns).sum(axis=-1)
        row_hat = (weights * rows).sum(axis=-1)
        assert np.abs(column_hat - u).max() <= LUT_COORDINATE_TOL
        assert np.abs(row_hat - v).max() <= LUT_COORDINATE_TOL

    # ------------------------------ G1 ---------------------------- #

    def test_g1_twists_seed_level(self):
        # FM9, FM4: the off-grid sweep at border 0.05, default band-pass
        _, targets, expected = pair_targets([twist_spec(t) for t in G1_TWISTS_DEG])
        state = g1_state(g1_pair(twist_spec(0.0)))
        local, _ = local_state_angles(state, targets)
        assert np.abs(local - expected).max() <= LOCAL_G1_TWIST_ANGLE_TOL_DEG
        theta, peak = production_angles(state, targets)
        assert theta.shape == (len(G1_TWISTS_DEG),) and theta.dtype == np.float64
        assert peak.shape == theta.shape and peak.dtype == np.float64
        assert np.isfinite(peak).all()
        error = np.abs(theta - expected)
        assert_within(float(error.max()), FM_ANGLE_TOL_DEG, "FM_ANGLE_TOL_DEG")
        # the sign: theta_hat has the sign of the imposed h21 (D22.3.6)
        for twist, value in zip(G1_TWISTS_DEG, theta):
            if twist != 0.0:
                assert np.sign(value) == np.sign(twist), (twist, value)
        # the FM9 kill separation: the pin sits below the error of the
        # flipped parabolic offset on this sweep (local recipe)
        bins = np.rint(local / 0.5)
        flipped = (2.0 * bins - local / 0.5) * 0.5
        assert FM_ANGLE_TOL_DEG < float(np.abs(flipped - expected).max())

    def test_g1_combined_seed_level(self):
        _, targets, expected = pair_targets(G1_COMBINED)
        state = g1_state(g1_pair(twist_spec(0.0)))
        local, _ = local_state_angles(state, targets)
        assert np.abs(local - expected).max() <= LOCAL_G1_COMBINED_ANGLE_TOL_DEG
        theta, _ = production_angles(state, targets)
        assert_within(
            float(np.abs(theta - expected).max()), FM_ANGLE_TOL_DEG, "FM_ANGLE_TOL_DEG"
        )

    # ------------------------------ G3 ---------------------------- #

    def test_g3_non_square_seed_level(self):
        # FM3, second killer: 460x560 crop, physical frequency within
        # FM_ANGLE_TOL_NONSQUARE_DEG; the bin-unit table misses by at
        # least 3x that from 5 deg on (premise asserted locally)
        pairs = [g3_pair(t) for t in G3_TWISTS]
        targets = np.stack([np.asarray(p["target"]) for p in pairs])
        expected = np.array([rotation_angle_of(p["exact"]) for p in pairs])
        state = g1_state(pairs[0])
        assert tuple(np.asarray(state.reference_subregion).shape) == (460, 560)
        local, _ = local_state_angles(state, targets)
        local_bin, _ = local_state_angles(state, targets, bin_units=True)
        for i, twist in enumerate(G3_TWISTS):
            assert abs(local[i] - expected[i] - LOCAL_G3_ERROR_DEG[twist]) <= 1e-3
            assert (
                abs(local_bin[i] - expected[i] - LOCAL_G3_BIN_UNIT_ERROR_DEG[twist])
                <= 1e-3
            )
        assert np.abs(local - expected).max() <= LOCAL_G3_ANGLE_TOL_DEG
        theta, _ = production_angles(state, targets)
        error = np.abs(theta - expected)
        assert_within(
            float(error.max()), FM_ANGLE_TOL_NONSQUARE_DEG, "FM_ANGLE_TOL_NONSQUARE_DEG"
        )
        mutant = [
            abs(LOCAL_G3_BIN_UNIT_ERROR_DEG[t]) for t in G3_TWISTS if abs(t) >= 5.0
        ]
        assert 3.0 * FM_ANGLE_TOL_NONSQUARE_DEG <= min(mutant)

    # --------------------------- dead band ------------------------ #

    @pytest.mark.parametrize(
        "filter_cutoffs", [(None, None), (0.05, None)], ids=["no_filter", "default"]
    )
    def test_the_dead_band_seed_level(self, filter_cutoffs):
        # D22.3 recorded limitation: the whole box is read, the cross
        # pulls the angle towards 0, within FM_ANGLE_TOL_DEADBAND_DEG
        pairs = [g1_dead_band_pair(t) for t in G1_DEAD_BAND_TWISTS]
        targets = np.stack([np.asarray(p["target"]) for p in pairs])
        expected = np.array([rotation_angle_of(p["exact"]) for p in pairs])
        state = g1_state(
            pairs[0], dead_band=G1_DEAD_BAND, filter_cutoffs=filter_cutoffs
        )
        local, _ = local_state_angles(state, targets)
        assert np.abs(local - expected).max() <= LOCAL_DEADBAND_ANGLE_TOL_DEG
        theta, _ = production_angles(state, targets)
        assert_within(
            float(np.abs(theta - expected).max()),
            FM_ANGLE_TOL_DEADBAND_DEG,
            "FM_ANGLE_TOL_DEADBAND_DEG",
        )

    # ---------------------- planted correlations ------------------ #

    @staticmethod
    def peak(correlation):
        theta, peak = _fourier_mellin.fourier_mellin_peak(
            np,
            np.atleast_2d(np.asarray(correlation, dtype=np.float64)),
            FROZEN_SEARCH_DEG,
        )
        return np.asarray(theta), np.asarray(peak)

    @staticmethod
    def bump(c, index, height, side=0.5):
        n = c.size
        c[index % n] = height
        c[(index - 1) % n] = height * side
        c[(index + 1) % n] = height * side

    def test_the_peak_honours_the_search_window(self):
        # FM8: a global maximum at a 40 deg lag and a local one at 5
        # deg return 5 deg; mirrored at -40 / -5 deg
        n = FROZEN_N_THETA
        positive = np.zeros(n)
        self.bump(positive, 80, 1.0)
        self.bump(positive, 10, 0.5)
        negative = np.zeros(n)
        self.bump(negative, -80, 1.0)
        self.bump(negative, -10, 0.5)
        theta, peak = self.peak(np.stack([positive, negative]))
        assert theta.shape == (2,) and theta.dtype == np.float64
        assert theta.tolist() == [5.0, -5.0]
        assert peak.tolist() == [0.5, 0.5]
        # the premise: the local rule agrees
        assert local_peak(np.stack([positive, negative]))[0].tolist() == [5.0, -5.0]

    def test_ties_go_to_the_lowest_output_index(self):
        # lags +6 and -6 (3 deg) tie exactly: +k wins (output index 6
        # before 354)
        c = np.zeros(FROZEN_N_THETA)
        self.bump(c, 6, 1.0)
        self.bump(c, -6, 1.0)
        theta, peak = self.peak(c)
        assert theta.tolist() == [3.0]
        assert peak.tolist() == [1.0]

    def test_an_edge_peak_takes_its_offset_from_the_raw_outside_neighbour(self):
        # the peak at lag 60 (30 deg, the window's edge bin); its right
        # neighbour, lag 61, lies OUTSIDE the window and still enters
        # the parabola raw
        c = np.zeros(FROZEN_N_THETA)
        c[59], c[60], c[61] = 0.5, 1.0, 0.9
        theta, peak = self.peak(c)
        delta = (0.5 - 0.9) / (2.0 * (0.5 - 2.0 + 0.9))
        assert abs(theta[0] - (60 + delta) * 180.0 / FROZEN_N_THETA) <= 1e-12
        assert peak.tolist() == [1.0]

    def test_a_non_negative_curvature_gets_offset_zero(self):
        # a constant correlation: argmax at lag 0, zero curvature, offset
        # 0 (never 0 / 0); a plateau entering the window at lag -60:
        # zero curvature there, theta exactly -30 deg
        flat = np.full(FROZEN_N_THETA, 0.25)
        plateau = np.zeros(FROZEN_N_THETA)
        plateau[[299, 300, 301]] = 1.0
        theta, peak = self.peak(np.stack([flat, plateau]))
        assert theta.tolist() == [0.0, -30.0]
        assert peak.tolist() == [0.25, 1.0]


# ================== V10(c) edge treatment and reuse ================ #


class TestFourierMellinEdgeTreatment:
    """V10(c), D22.3.1, D22.6: the frequency-domain Hann stencil equals
    a second FFT of the windowed crop, the reused ``target_spectra``
    are never modified, the translation rows of unrouted slots are
    bitwise the Stage E rows, and the edge treatment locks the angle
    under a detector-fixed background (G4) where the untreated recipe
    does not (premise asserted on the local re-implementation).

    Mutants and their designed killers:
    FM6 (the edge treatment written in place into ``target_spectra``):
    :meth:`test_the_stencil_returns_a_new_array`,
    :meth:`test_seed_homographies_leaves_the_target_spectra_bitwise` and
    :meth:`test_unrouted_rows_are_bitwise_the_stage_e_rows`;
    FM7 (the edge treatment dropped): :meth:`test_the_g4_background_lock`
    at ``(None, None)``."""

    TWISTS = (0.37, 2.71, -4.42)

    def fixture(self):
        _, targets, _ = pair_targets([twist_spec(t) for t in self.TWISTS])
        return g1_state(g1_pair(twist_spec(0.0))), targets

    def test_the_stencil_equals_fft2_of_the_windowed_crop(self):
        state, targets = self.fixture()
        spectra = np.asarray(numpy_seed_rows(state, targets)[1])
        expected = windowed_crop_spectra(state, targets)
        # the premise: the local stencil reproduces the second FFT
        assert relative_error(local_hann_stencil(spectra), expected) <= 1e-14
        windowed = np.asarray(_fourier_mellin.fourier_mellin_hann_stencil(np, spectra))
        assert windowed.dtype == np.complex128
        assert windowed.shape == spectra.shape
        assert_within(
            relative_error(windowed, expected), FM_STENCIL_RTOL, "FM_STENCIL_RTOL"
        )

    def test_the_stencil_returns_a_new_array(self):
        # FM6 at unit level, and the D22.12 promotion of complex64
        state, targets = self.fixture()
        spectra = np.asarray(numpy_seed_rows(state, targets)[1])
        before = spectra.copy()
        windowed = _fourier_mellin.fourier_mellin_hann_stencil(np, spectra)
        assert windowed is not spectra
        assert not np.shares_memory(windowed, spectra)
        assert np.array_equal(spectra, before)
        single = spectra.astype(np.complex64)
        single_before = single.copy()
        promoted = np.asarray(_fourier_mellin.fourier_mellin_hann_stencil(np, single))
        assert promoted.dtype == np.complex128
        assert np.array_equal(single, single_before)

    @pytest.mark.parametrize("route", [(1, 1, 1), (2, 2, 2), (0, 1, 2)])
    def test_seed_homographies_leaves_the_target_spectra_bitwise(self, route):
        # FM6: after the FM branch ran, ``target_spectra`` is bitwise
        # the array ``seed_spectra`` returned
        state, targets = self.fixture()
        seam = numpy_fm_seam(state, targets, route)
        before = np.array(seam.spectra, copy=True)
        rows = seam_rows(seam)
        assert np.array_equal(seam.spectra, before)
        assert ANGLE_KEY in seam.batch.outputs
        assert np.isfinite(rows).all()

    def test_unrouted_rows_are_bitwise_the_stage_e_rows(self):
        # the translation rows of unrouted slots, and of routed slots
        # whose acceptance kept h_T, are bitwise Stage E's (FM6, FM15)
        _, targets, _ = pair_targets([twist_spec(t) for t in (0.37, 2.71, -4.42, 5.29)])
        state = g1_state(g1_pair(twist_spec(0.0)))
        h_t = stage_e_rows(state, targets)
        seam = numpy_fm_seam(state, targets, (0, 1, 0, 2))
        rows = seam_rows(seam)
        applied = np.asarray(seam.batch.outputs[APPLIED_KEY])
        for slot in (0, 2):
            assert np.array_equal(rows[slot], h_t[slot]), slot
            assert not applied[slot]
        for slot in (1, 3):
            if not applied[slot]:
                assert np.array_equal(rows[slot], h_t[slot]), slot

    @pytest.mark.parametrize("noise", G4_NOISES)
    @pytest.mark.parametrize("filter_cutoffs", G4_FILTERS, ids=["no_filter", "default"])
    def test_the_g4_background_lock(self, filter_cutoffs, noise):
        # FM7: under the detector-fixed background the frozen recipe
        # stays within FM_ANGLE_TOL_BACKGROUND_DEG, while the recipe
        # WITHOUT the edge treatment locks at zero on the measured count
        # at (None, None) (premise, local)
        pairs = [g4_pair(case, noise) for case in range(len(G4_CASES))]
        targets = np.stack([np.asarray(p["target"]) for p in pairs])
        expected = np.array([rotation_angle_of(p["exact"]) for p in pairs])
        state = g1_state(pairs[0], filter_cutoffs=filter_cutoffs)
        treated, _ = local_state_angles(state, targets)
        assert np.abs(treated - expected).max() <= LOCAL_G4_ANGLE_TOL_DEG
        if filter_cutoffs == (None, None):
            untreated, _ = local_state_angles(
                state,
                targets,
                edge_treatment=False,
                magnitude="amplitude",
                rho_band=UNTREATED_RHO_BAND,
            )
            locked = int(np.count_nonzero(np.abs(untreated - expected) > 1.0))
            assert locked == FM_LOCK_PREMISE_COUNT[noise]
        theta, _ = production_angles(state, targets)
        assert_within(
            float(np.abs(theta - expected).max()),
            FM_ANGLE_TOL_BACKGROUND_DEG,
            "FM_ANGLE_TOL_BACKGROUND_DEG",
        )


# ================= V10(d) seed rows and failure semantics ========== #


class TestFourierMellinSeedRows:
    """V10(d), D22.4, D22.5, D22.6: the FM row's corner error on the
    projection-centre-shift fixture (border 0.15), the row identity, the
    de-rotated crop against the CPU bicubic evaluation (with the dead
    band present), the never-NaN fallbacks on planted failures through
    the frozen patch points, a NaN ``h_T`` never rescued, the constant
    target, and the outputs rule.

    Mutants and their designed killers:
    FM1 (de-rotation by ``R(-theta)``):
    :meth:`test_the_derotated_crop_is_the_cpu_bicubic_evaluation` and
    :meth:`test_pc_shift_rows_within_the_seed_band`;
    FM2 (``T(t) R``, the ONLY killer, on pure twists ``t`` is near 0):
    :meth:`test_pc_shift_rows_within_the_seed_band` (the pin asserted
    below every local ``T(t) R`` error);
    FM5 (a failed estimate returning NaN):
    :meth:`test_a_planted_failure_returns_h_t_for_that_slot_only`;
    FM13 (a NaN ``h_T`` rescued):
    :meth:`test_a_nan_translation_row_is_never_rescued`;
    FM31 (de-rotation through the subregion resident):
    :meth:`test_the_dead_band_pixels_are_in_the_crop`;
    FM32 (degrees and radians confused):
    :meth:`test_the_partial_row_is_the_frozen_formula` and
    :meth:`test_the_routed_rows_are_their_own_partial_rows`;
    FM38 (the translation read from the reused spectrum):
    :meth:`test_pc_shift_rows_within_the_seed_band` (the pin asserted
    below the local mutant's error, 6.6 to 113 px against at most 0.22
    px for the partial row);
    FM43 (de-rotation about the detector centre):
    :meth:`test_pc_shift_rows_within_the_seed_band` (pin asserted below
    the local mutant's ``2 sin(theta / 2) |c|`` error) and
    :meth:`test_the_derotated_crop_is_the_cpu_bicubic_evaluation`;
    FM52 (outputs on route-0 or padded slots):
    :meth:`test_the_outputs_rule`."""

    TWISTS = (3.0, 5.0, -3.0)

    def fixture(self):
        _, targets, _ = pair_targets([twist_spec(t) for t in self.TWISTS])
        return g1_state(g1_pair(twist_spec(0.0))), targets

    # --------------------------- seed rows ------------------------ #

    def test_pc_shift_rows_within_the_seed_band(self):
        pairs = [g1_pair(twist_spec(t), shift) for t, shift in G1_PC_SHIFT_CASES]
        targets = np.stack([np.asarray(p["target"]) for p in pairs])
        state = g1_state(pairs[0], border=G1_WIDE_BORDER)
        corners = subregion_corners(SHAPE_480, pairs[0]["pc_px"], border=G1_WIDE_BORDER)
        local = [local_seed_row_errors(state, p, corners) for p in pairs]
        partial = max(e["partial"] for e in local)
        reversed_min = min(e["reversed"] for e in local)
        centre_min = min(e["centre"] for e in local)
        reused_min = min(e["reused"] for e in local)
        # the premise: the three mutant populations sit above the partial
        assert partial < reversed_min and partial < centre_min, local
        assert partial < reused_min, local
        seam = numpy_fm_seam(state, targets, (ROUTE_FORCED,) * len(pairs))
        rows = seam_rows(seam)
        applied = np.asarray(seam.batch.outputs[APPLIED_KEY])
        assert applied.all()
        errors = [
            recovery_error(rows[i], p["exact"], corners) for i, p in enumerate(pairs)
        ]
        assert_within(max(errors), FM_SEED_TOL_PX, "FM_SEED_TOL_PX")
        # the kill separations of FM2, FM43 and FM38 (local mutants)
        assert FM_SEED_TOL_PX < reversed_min
        assert FM_SEED_TOL_PX < centre_min
        assert FM_SEED_TOL_PX < reused_min

    def test_the_partial_row_is_the_frozen_formula(self):
        # FM32 at unit level: ``(c - 1, -s, c tx - s ty, s, c - 1, s tx +
        # c ty, 0, 0)`` of theta in DEGREES and t = rows_t[:, (2, 5)]
        rng = np.random.default_rng(1208)
        theta = np.array([0.0, 2.5, -8.0, 19.13, 29.9])
        rows_t = np.zeros((theta.size, 8))
        rows_t[:, 2] = rng.uniform(-30.0, 30.0, theta.size)
        rows_t[:, 5] = rng.uniform(-30.0, 30.0, theta.size)
        rows = np.asarray(_fourier_mellin.fourier_mellin_partial_row(np, theta, rows_t))
        assert rows.shape == (theta.size, 8) and rows.dtype == np.float64
        expected = np.stack([local_partial_row(a, r) for a, r in zip(theta, rows_t)])
        np.testing.assert_allclose(rows, expected, rtol=0, atol=1e-12)

    def test_the_routed_rows_are_their_own_partial_rows(self):
        # FM32 end to end: each applied row is the partial row of its
        # own reported angle and the translation of its own de-rotated
        # crop (the frozen names recomputed on the same batch)
        state, targets = self.fixture()
        seam = numpy_fm_seam(state, targets, (ROUTE_FORCED,) * len(self.TWISTS))
        rows = seam_rows(seam)
        angle = np.asarray(seam.batch.outputs[ANGLE_KEY])
        assert np.asarray(seam.batch.outputs[APPLIED_KEY]).all()
        crops, ok = _fourier_mellin.fourier_mellin_derotate(
            seam.ctx, seam.batch, seam.fm_state, angle
        )
        assert np.asarray(ok).all()
        rows_t = np.asarray(
            _fourier_mellin.fourier_mellin_translate(seam.ctx, crops, seam.seed_state)
        )
        expected = np.stack([local_partial_row(a, r) for a, r in zip(angle, rows_t)])
        np.testing.assert_allclose(rows, expected, rtol=0, atol=1e-12)
        # and the angle is the seed-level one (D22.3)
        theta, _ = _fourier_mellin.fourier_mellin_angles(
            seam.ctx, seam.spectra, seam.fm_state
        )
        assert np.array_equal(angle, np.asarray(theta))

    # --------------------------- de-rotation ---------------------- #

    def test_the_derotated_crop_is_the_cpu_bicubic_evaluation(self):
        # FM1, FM43: ``D(xi) = target(R xi)`` about the grain reference
        # PC over every pixel of the D5 bounding box, within 1e-12 of
        # the CPU bicubic evaluation (numpy twin)
        state, targets = self.fixture()
        seam = numpy_fm_seam(state, targets)
        theta = np.array([3.0, 5.0, -3.0])
        crops, ok = _fourier_mellin.fourier_mellin_derotate(
            seam.ctx, seam.batch, seam.fm_state, theta
        )
        crops = np.asarray(crops)
        r0, r1, c0, c1 = state.bounds
        assert crops.shape == (len(theta), r1 - r0, c1 - c0)
        assert crops.dtype == np.float64
        assert np.asarray(ok).dtype == np.bool_ and np.asarray(ok).all()
        for i, target in enumerate(targets):
            expected = local_derotated_crop(state, target, theta[i])
            assert np.abs(crops[i] - expected).max() <= 1e-12, i

    def test_the_dead_band_pixels_are_in_the_crop(self):
        # FM31: the box resident covers the dead band too (the D5 seed
        # correlates the whole crop); a subregion-resident de-rotation
        # cannot reshape to the box, nor carry these pixels
        pair = g1_dead_band_pair(G1_DEAD_BAND_TWISTS[1])
        state = g1_state(pair, dead_band=G1_DEAD_BAND)
        target = np.asarray(pair["target"])
        seam = numpy_fm_seam(state, target[None])
        theta = np.array([G1_DEAD_BAND_TWISTS[1]])
        crops, _ = _fourier_mellin.fourier_mellin_derotate(
            seam.ctx, seam.batch, seam.fm_state, theta
        )
        crops = np.asarray(crops)
        r0, r1, c0, c1 = state.bounds
        assert crops.shape == (1, r1 - r0, c1 - c0)
        expected = local_derotated_crop(state, target, theta[0])
        x0, x1, y0, y1 = G1_DEAD_BAND
        band = np.s_[y0 - r0 : y1 - r0, x0 - c0 : x1 - c0]
        assert np.isfinite(crops[0][band]).all()
        assert np.abs(crops[0][band] - expected[band]).max() <= 1e-12
        assert np.abs(crops[0] - expected).max() <= 1e-12

    # ------------------------ failure semantics ------------------- #

    @staticmethod
    def plant(monkeypatch, kind, slot=1):
        """Plant one per-slot failure through a frozen patch point."""
        if kind == "angle_nan":
            real = _fourier_mellin.fourier_mellin_angles

            def planted(ctx, target_spectra, fm_state):
                theta, peak = real(ctx, target_spectra, fm_state)
                theta = np.array(theta, copy=True)
                theta[slot] = np.nan
                return theta, peak

            monkeypatch.setattr(_fourier_mellin, "fourier_mellin_angles", planted)
        elif kind in ("derotate_nan_pixel", "derotate_not_ok"):
            real = _fourier_mellin.fourier_mellin_derotate

            def planted(ctx, batch, fm_state, theta_deg):
                crops, ok = real(ctx, batch, fm_state, theta_deg)
                crops = np.array(crops, copy=True)
                ok = np.array(ok, copy=True)
                if kind == "derotate_nan_pixel":
                    crops[slot, 3, 4] = np.nan
                else:
                    ok[slot] = False
                return crops, ok

            monkeypatch.setattr(_fourier_mellin, "fourier_mellin_derotate", planted)
        else:
            real = _fourier_mellin.fourier_mellin_translate

            def planted(ctx, crops, seed_state):
                rows_t = np.array(real(ctx, crops, seed_state), copy=True)
                rows_t[slot, 2] = np.inf
                return rows_t

            monkeypatch.setattr(_fourier_mellin, "fourier_mellin_translate", planted)

    @pytest.mark.parametrize("route", [ROUTE_ACCEPT, ROUTE_FORCED])
    @pytest.mark.parametrize(
        "kind",
        ["angle_nan", "derotate_nan_pixel", "derotate_not_ok", "translate_nonfinite"],
    )
    def test_a_planted_failure_returns_h_t_for_that_slot_only(
        self, monkeypatch, kind, route
    ):
        # FM5: the failed slot's row is h_T bitwise, applied False, no
        # NaN introduced, and the other slots exactly as unplanted
        state, targets = self.fixture()
        h_t = stage_e_rows(state, targets)
        routes = (route,) * len(self.TWISTS)
        clean = numpy_fm_seam(state, targets, routes)
        clean_rows = seam_rows(clean)
        self.plant(monkeypatch, kind)
        seam = numpy_fm_seam(state, targets, routes)
        rows = seam_rows(seam)
        applied = np.asarray(seam.batch.outputs[APPLIED_KEY])
        angle = np.asarray(seam.batch.outputs[ANGLE_KEY])
        assert np.isfinite(rows).all()
        assert np.array_equal(rows[1], h_t[1])
        assert not applied[1]
        if kind == "angle_nan":
            assert np.isnan(angle[1])
        clean_applied = np.asarray(clean.batch.outputs[APPLIED_KEY])
        clean_angle = np.asarray(clean.batch.outputs[ANGLE_KEY])
        for slot in (0, 2):
            assert np.array_equal(rows[slot], clean_rows[slot]), slot
            assert applied[slot] == clean_applied[slot], slot
            assert np.array_equal(angle[slot], clean_angle[slot], equal_nan=True)
        if route == ROUTE_FORCED:
            assert clean_applied.all()

    @pytest.mark.parametrize("route", [ROUTE_ACCEPT, ROUTE_FORCED])
    def test_a_nan_translation_row_is_never_rescued(self, route):
        # FM13: a non-finite h_T is returned unchanged although a finite
        # FM row exists for that slot (the other slots carry it)
        state, targets = self.fixture()
        seam = numpy_fm_seam(state, targets)
        h_t = stage_e_rows(state, targets)
        h_t[1] = np.nan
        routes = np.full(len(self.TWISTS), route, dtype=np.int8)
        rows, angle, applied = _fourier_mellin.fourier_mellin_rows(
            seam.ctx, seam.batch, seam.spectra, seam.seed_state, h_t, routes
        )
        rows = np.asarray(rows)
        applied = np.asarray(applied)
        assert rows.shape == (len(self.TWISTS), 8) and rows.dtype == np.float64
        assert np.asarray(angle).shape == (len(self.TWISTS),)
        assert applied.dtype == np.bool_
        assert np.isnan(rows[1]).all()
        assert not applied[1]
        assert np.isfinite(rows[[0, 2]]).all()
        if route == ROUTE_FORCED:
            assert applied[[0, 2]].all()

    def test_a_constant_target_gives_the_failure_contract(self):
        # D2.6 exactly as with "off" (default band-pass; ledger 132
        # (vii): the translation row and the angle are finite there,
        # only the fit refuses the pattern, so the outcome is asserted)
        reference = np.asarray(g1_reference())
        patterns = np.stack([reference, np.full(SHAPE_480, G8_CONSTANT_VALUE)])
        detector = make_detector(navigation_shape=(1, 2))
        common = {"reference": (0, 0), "max_iterations": G1_CAPTURE_MAX_ITERATIONS}
        off = run_engine(patterns, (1, 2), detector, **common)
        on = run_engine(patterns, (1, 2), detector, fourier_mellin="always", **common)
        assert int(np.asarray(off["num_iterations"])[1]) == 0
        assert np.isnan(np.asarray(off["homography"])[1]).all()
        for name in (
            "homography",
            "residual",
            "num_iterations",
            "norm_dp",
            "converged",
        ):
            assert np.array_equal(
                np.asarray(on[name])[1], np.asarray(off[name])[1], equal_nan=True
            ), name
        assert set(FOURIER_MELLIN_PROP_NAMES) <= set(on)

    # --------------------------- outputs rule --------------------- #

    def test_the_outputs_rule(self):
        # FM52: on route-0 and padded slots the angle is NaN and
        # applied False, although the masked branch computed them; the
        # rows there are h_T bitwise; the routed slots carry an angle
        state, targets = self.fixture()
        padded = np.concatenate([targets, np.asarray(g1_reference())[None]])
        h_t = stage_e_rows(state, padded)
        seam = numpy_fm_seam(
            state,
            padded,
            (ROUTE_NONE, ROUTE_ACCEPT, ROUTE_FORCED, ROUTE_NONE),
            pattern_index=[0, 1, 2, -1],
        )
        rows = seam_rows(seam)
        outputs = seam.batch.outputs
        assert set(outputs) == {ANGLE_KEY, APPLIED_KEY}
        angle = np.asarray(outputs[ANGLE_KEY])
        applied = np.asarray(outputs[APPLIED_KEY])
        assert angle.shape == (4,) and angle.dtype == np.float64
        assert applied.shape == (4,) and applied.dtype == np.bool_
        for slot in (0, 3):
            assert np.isnan(angle[slot]), slot
            assert not applied[slot], slot
            assert np.array_equal(rows[slot], h_t[slot]), slot
        assert np.isfinite(angle[[1, 2]]).all()
        assert applied[2]

    def test_an_all_zero_route_writes_nothing(self):
        # D22.6: decided on the host, the Stage E rows bitwise, no
        # outputs; likewise a route key with no FM state
        state, targets = self.fixture()
        h_t = stage_e_rows(state, targets)
        seam = numpy_fm_seam(state, targets, (ROUTE_NONE,) * len(self.TWISTS))
        rows = seam_rows(seam)
        assert np.array_equal(rows, h_t)
        assert seam.batch.outputs == {}
        ctx = numpy_seed_context()
        seed_state = _batched.build_seed_state(ctx, state)
        batch = numpy_seed_batch(state, targets)
        batch.extras[ROUTE_KEY] = np.full(len(self.TWISTS), ROUTE_FORCED, np.int8)
        spectra = _batched.seed_spectra(ctx, batch, seed_state)
        rows = np.asarray(_batched.seed_homographies(ctx, batch, spectra, seed_state))
        assert np.array_equal(rows, h_t)
        assert batch.outputs == {}


# ====================== V10(e) capture, end to end ================= #

_STARRED = [twist_spec(t) for t in G1_CAPTURE_TWISTS_STARRED] + [
    rotvec_spec(w) for w in G1_CAPTURE_ROTVECS_STARRED
]
_WEEKLY = [
    twist_spec(t) for t in G1_CAPTURE_TWISTS if t not in G1_CAPTURE_TWISTS_STARRED
]
_WEEKLY += [
    rotvec_spec(w) for w in G1_CAPTURE_ROTVECS if w not in G1_CAPTURE_ROTVECS_STARRED
]
CAPTURE_CASES = [pytest.param(spec, id=str(spec[0][1])) for spec in _STARRED] + [
    pytest.param(spec, id=str(spec[0][1]), marks=pytest.mark.weekly) for spec in _WEEKLY
]


class TestFourierMellinCapture:
    """V10(e), D22.3, D22.4: capture end to end through
    ``run_hrebsd_dic`` with ``fourier_mellin="always"`` on one-row maps
    (the runner, the route, the packed row and the props exercised),
    the expectation the exact imposed homography by the V2 corner
    metric; premise on every arm, asserted at run time: the ``"off"``
    path fails at ``max_iterations=50``.  The starred arms of V10(e)
    run in the default suite, the rest under ``--weekly``.

    Mutants and their designed killers:
    FM1 (de-rotation by ``R(-theta)``), FM32 (degrees and radians) and
    FM38 (translation from the reused spectrum):
    :meth:`test_the_fm_seed_captures_what_off_misses` (the FM-seeded fit
    does not converge to the exact homography) and, at 8 to 25 deg,
    :meth:`test_wide_border_converges_to_the_exact_seeded_optimum`;
    FM10 (acceptance inverted):
    :meth:`test_the_fm_seed_captures_what_off_misses` (the FM row is
    refused, seed code 0, the ``"off"`` failure returns)."""

    @pytest.mark.parametrize("spec", CAPTURE_CASES)
    def test_the_fm_seed_captures_what_off_misses(self, spec):
        off = capture_off(spec, G1_BORDER)
        assert bool(np.asarray(off["converged"])[1]) == CAPTURE_OFF_CONVERGED
        properties = capture_run(spec, G1_BORDER)
        exact = g1_row_map(spec)["exact"]
        assert bool(np.asarray(properties["converged"])[1])
        error = recovery_error(
            np.asarray(properties["homography"])[1], exact, capture_corners(G1_BORDER)
        )
        assert_within(error, FM_CAPTURE_TOL_PX, "FM_CAPTURE_TOL_PX")
        codes = np.asarray(properties["fourier_mellin_seed"])
        assert codes.dtype == np.int32
        assert codes[1] == SEED_FIRST_PASS
        angle = np.asarray(properties["fourier_mellin_angle"])
        assert np.isfinite(angle[1])

    @pytest.mark.weekly
    @pytest.mark.parametrize(
        "twist", G1_WIDE_CAPTURE_TWISTS, ids=[str(t) for t in G1_WIDE_CAPTURE_TWISTS]
    )
    def test_wide_border_converges_to_the_exact_seeded_optimum(self, twist):
        # border 0.15: converged in at most FM_CAPTURE_ITERATIONS to the
        # optimum the EXACT-seeded fit reaches, within
        # FM_FIXED_POINT_TOL_PX (FM1, FM32, FM38 at 8 deg and more)
        spec = twist_spec(twist)
        off = capture_off(spec, G1_WIDE_BORDER)
        assert bool(np.asarray(off["converged"])[1]) == CAPTURE_OFF_CONVERGED
        pair = g1_pair(spec)
        state = g1_state(pair, border=G1_WIDE_BORDER)
        exact_fit = fit_pattern(
            state,
            np.asarray(pair["target"]),
            h0=np.asarray(pair["exact"]),
            max_iterations=G1_CAPTURE_MAX_ITERATIONS,
        )
        assert exact_fit["converged"]
        properties = capture_run(spec, G1_WIDE_BORDER)
        assert bool(np.asarray(properties["converged"])[1])
        assert_within(
            int(np.asarray(properties["num_iterations"])[1]),
            FM_CAPTURE_ITERATIONS,
            "FM_CAPTURE_ITERATIONS",
        )
        gap = recovery_error(
            np.asarray(properties["homography"])[1],
            exact_fit["h"],
            capture_corners(G1_WIDE_BORDER),
        )
        assert_within(gap, FM_FIXED_POINT_TOL_PX, "FM_FIXED_POINT_TOL_PX")
        assert np.asarray(properties["fourier_mellin_seed"])[1] == SEED_FIRST_PASS

    @pytest.mark.weekly
    def test_the_window_edge_is_recorded(self, record_property):
        # RECORDED, not asserted (V10(e)): 30 deg at border 0.15; spec
        # gate theta_hat 29.931 deg, converged in 3 to 0.033 px
        spec = twist_spec(G1_WINDOW_EDGE_TWIST)
        properties = capture_run(spec, G1_WIDE_BORDER)
        error = recovery_error(
            np.asarray(properties["homography"])[1],
            g1_row_map(spec)["exact"],
            capture_corners(G1_WIDE_BORDER),
        )
        record_property(
            "fm_window_edge",
            {
                "theta_hat": float(np.asarray(properties["fourier_mellin_angle"])[1]),
                "converged": bool(np.asarray(properties["converged"])[1]),
                "iterations": int(np.asarray(properties["num_iterations"])[1]),
                "error_px": float(error),
            },
        )
        assert set(FOURIER_MELLIN_PROP_NAMES) <= set(properties)


# ===== DEFAULT (f)-(k) (inserted) =====


# ===== DEFAULT (f)-(k): fixtures, premises and spies of this block ===== #
#
# Everything below is prefixed ``fk_`` / ``FK_`` (the (f) to (k)
# block) so that it never collides with the helpers of the other
# blocks.

# FK-MIXED: the (i)/(j)/(k) map mixing routed, unrouted and retried
# points on the V8 geometry (``PC_SEEDING``), one row of six at a
# budget of 20 (cheap: every capped fit stops at 20).  Columns: the
# reference; a 0.8 degree twist whose xmap reads its true orientation
# (below the gate, not routed); a 1.6 degree twist, xmap true (routed,
# the FM row wins); a 2.4 degree twist whose xmap reads the reference
# (MISLABELLED: not routed, fails, retried from its FM row); the G6
# unrelated pattern whose xmap is labelled with a 5 degree twist
# (routed, the acceptance keeps the translation row, fails, retried,
# the forced fit fails too); a masked reference copy
FK_MIXED_NAVIGATION_SHAPE = (1, 6)
FK_MIXED_REFERENCE = 0
FK_MIXED_UNROUTED = 1
FK_MIXED_ROUTED = 2
FK_MIXED_MISLABELLED = 3
FK_MIXED_UNRELATED = 4
FK_MIXED_MASKED = 5
FK_MIXED_TWISTS = (0.0, 0.8, 1.6, 2.4)
FK_MIXED_UNRELATED_LABEL_DEG = 5.0
FK_MIXED_MAX_ITERATIONS = 20
# The first-pass routes under "auto", fit order (one grain: the map
# order of the fitted points 0 to 4)
FK_MIXED_ROUTES = (0, 0, 1, 0, 1)
FK_MIXED_FITTED = (0, 1, 2, 3, 4)
FK_MIXED_RETRIED = (3, 4)

# FK-TWO-GRAIN: the (k) fit-order and lazy-build map, one row of five,
# two grains interleaved in map order: label 1 = points 0 (its
# reference, grain A's pattern), 2 (a 2.5 degree twist, routed) and 4
# (a -2.0 degree twist, routed); label 0 = points 1 (grain A's pattern,
# xmap at grain A's orientation: a -0.8 degree twist from its
# reference, not routed) and 3 (its reference, the 0.8 degree twist
# pattern, so the two grains' reference crops differ).
# ``unique_references`` = (0, 3), so the fit order (0, 2, 4, 1, 3)
# differs from the map order
FK_TWO_GRAIN_NAVIGATION_SHAPE = (1, 5)
FK_TWO_GRAIN_LABELS = ((1, 0, 1, 0, 1),)
FK_TWO_GRAIN_REFERENCES = (3, 0)
FK_TWO_GRAIN_FIT_ORDER = (0, 2, 4, 1, 3)
FK_TWO_GRAIN_ROUTES = {0: 0, 2: 1, 4: 1, 1: 0, 3: 0}
FK_TWO_GRAIN_ROUTED_GRAIN = (0, 2, 4)
FK_TWO_GRAIN_UNROUTED_GRAIN = (1, 3)
FK_TWO_GRAIN_MAX_ITERATIONS = 10

# The (i) constant-pattern arm at ``(None, None)``: one row of two, the
# V8 reference and a constant pattern, where the translation row of
# the constant crop IS non-finite (see the MEASURED block: under the
# default band-pass it is not, ``G8_CONSTANT_TRANSLATION_ROW_FINITE``)
FK_CONSTANT_NAVIGATION_SHAPE = (1, 2)
FK_CONSTANT_FILTER_CUTOFFS = (None, None)

# The fit-carrying property names compared per point
FK_FIT_PROPS = ("homography", "residual", "num_iterations", "norm_dp", "converged")

# The planted wrong angle of V10(g), deg
FK_PLANTED_ANGLE_OFFSET_DEG = 10.0
FK_PLANTED_TWIST_DEG = 3.0

# -------- (f)-(k) CPU-side premises, MEASURED at this gate ---------- #
#
# Measured 2026-10-08 on machine A's CPU (worktree .venv, the recipe of
# the MEASURED block above) through the local re-implementation and the
# existing CPU engine, scratch ``fk/p1.py`` to ``fk/p3.py``:
#
# FK-MIXED ``"off"`` at 20: converged (T, T, T, F, F) on points 0 to 4
# in (1, 5, 10, 20, 20) iterations, every ``h`` finite; the 2.4 degree
# point 23.4 px off; the local FM row of the 2.4 degree point (angle
# 2.3988) converges in 2 iterations to 0.0096 px; the 1.6 degree
# point's criteria at the seed 1.6251 (translation) against 0.00087
# (local FM row); the unrelated point's 2.0402 against 2.0801 (the G6
# premise), and its forced local-FM fit does not converge at 20.  A
# direct ``fit_pattern(make_state(reference, pc), target)`` is BITWISE
# the engine's stored result on this geometry (the (i) oracle's
# premise).  The constant pattern's D5 seed under ``(None, None)`` is
# NaN throughout.  The planted wrong angle (V10(g)) on G1 at 3 deg:
# local criteria 1.9900 (translation), 0.0011 (true FM row), 2.0218
# (the FM row at theta + 10), so the planted row loses by 0.032.
FK_MIXED_OFF_CONVERGED = (True, True, True, False, False)
FK_PREMISE_CRITERION_ATOL = 5e-3

# FIXME-pin: the number of SECOND fits the retry gives G8's constant
# pattern under the default band-pass (V10(i); the premise "its forced
# FM estimate fails" does NOT hold there, the MEASURED block's
# ``G8_CONSTANT_TRANSLATION_ROW_FINITE``): 1 without a guard (the forced
# FM row is finite, fitted, and returns the D2.6 contract again), 0 with
# one.  The no-second-fit spy is asserted on the ``(None, None)`` arm
# instead, where it is provable.  OPEN SPEC DECISION (critic F7,
# 2026-10-08; ledger 134): D22.8 and V10(f)/(i) still word the G8 arm
# as "never refitted", so the guard-or-no-guard choice is a behaviour
# the spec has not made.  It needs a dated disposition in plan 12 / V10
# BEFORE the implementation (option A: no guard, pin 1 and amend the
# D22.8 sentence; option B: a degenerate-crop guard, specified, pin 0);
# this literal then takes the DECIDED value, not "as measured"
# DECIDED 2026-10-08: option A, no guard (plan 12.6; D22.8 amendment)
FM_G8_CONSTANT_SECOND_FITS = 1


class _FkAbort(Exception):
    """Raised by the gate probe's ``_run_chunks`` stand-in: the routes
    are captured and the run stops before any fit."""


_FK_LOCAL = threading.local()


def fk_bitwise(a, b) -> bool:
    """Return whether two arrays are bitwise equal (NaN equal to NaN on
    floats), dtypes and shapes included."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(
        a.dtype, np.complexfloating
    ):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def fk_assert_point_bitwise(left: dict, right: dict, index: int, names=None) -> None:
    """Assert one map point's properties are bitwise equal in two
    property dictionaries (default: every Stage A property)."""
    names = _engine.STAGE_A_PROP_NAMES if names is None else names
    for name in names:
        assert fk_bitwise(
            np.asarray(left[name])[index], np.asarray(right[name])[index]
        ), (
            name,
            index,
        )


def fk_assert_result_equals_fit(properties: dict, index: int, result: dict) -> None:
    """Assert a stored map point equals a direct ``fit_pattern`` result
    bitwise (``h``, ``residual``, ``num_iterations``, ``norm_dp``,
    ``converged``)."""
    assert fk_bitwise(np.asarray(properties["homography"])[index], result["h"])
    assert fk_bitwise(np.asarray(properties["residual"])[index], result["residual"])
    assert int(np.asarray(properties["num_iterations"])[index]) == int(
        result["num_iterations"]
    )
    assert fk_bitwise(np.asarray(properties["norm_dp"])[index], result["norm_dp"])
    assert bool(np.asarray(properties["converged"])[index]) == bool(result["converged"])


def fk_same_optimum(h, exact, corners) -> bool:
    """Return the V8 same-optimum test: the recovery error below
    ``SAME_OPTIMUM_FRACTION`` of the imposed corner displacement (at
    least 1 px)."""
    imposed = matrix_corner_norm(matrix_of(exact), corners)
    error = recovery_error(h, exact, corners)
    return error < SAME_OPTIMUM_FRACTION * max(imposed, 1.0)


def fk_assert_pinned_mapping(measured: dict, pinned, name: str) -> None:
    """Assert a per-point mapping of measured COUNTS or CODES equals its
    pin, failing loudly while the pin is an unfilled ``FIXME-pin``."""
    if pinned is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V10); "
            f"measured {measured!r}. Pin it with a dated comment"
        )
    assert dict(measured) == dict(pinned), f"{name}: {measured} != {pinned}"


def fk_assert_fm_props(properties: dict, map_size: int) -> None:
    """Assert the two D22.9 props exist with their frozen dtypes and
    shapes beside exactly the Stage A set (no ``seed_round``)."""
    assert set(properties) == set(_engine.STAGE_A_PROP_NAMES) | set(
        FOURIER_MELLIN_PROP_NAMES
    )
    for name in FOURIER_MELLIN_PROP_NAMES:
        value = np.asarray(properties[name])
        assert value.dtype == FOURIER_MELLIN_PROP_DTYPES[name], name
        assert value.shape == (map_size,), name


def fk_identify(image, lookup) -> int:
    """Return the first index of *lookup* whose image equals *image* (to
    1e-9 relative), or -1 (also when *lookup* is ``None``)."""
    if lookup is None:
        return -1
    image = np.asarray(image, dtype=np.float64)
    differences = [
        float(np.max(np.abs(image - np.asarray(candidate, dtype=np.float64))))
        for candidate in lookup
    ]
    best = int(np.argmin(differences))
    scale = max(1.0, float(np.max(np.abs(np.asarray(lookup[best])))))
    return best if differences[best] <= 1e-9 * scale else -1


def fk_route_of(batch):
    """Return a host copy of the batch's route key, or ``None``."""
    route = batch.extras.get(ROUTE_KEY)
    return None if route is None else np.array(route)


def fk_slot_identity(batch, lookup) -> list:
    """Return the map index of every slot of *batch*: matched against
    the preprocessed *lookup* when given (the CPU route's one-slot batch
    need not carry the flat index), else the batch's ``pattern_index``."""
    index = np.asarray(batch.pattern_index, dtype=np.int64)
    if lookup is None:
        return [int(i) for i in index]
    return [fk_identify(target, lookup) for target in np.asarray(batch.targets)]


@contextlib.contextmanager
def fk_patched(pairs):
    """Temporarily set ``(module, name, value)`` attributes, restoring
    them afterwards (the cached runs cannot use ``monkeypatch``)."""
    saved = [(module, name, getattr(module, name)) for module, name, _ in pairs]
    try:
        for module, name, value in pairs:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def fk_install(monkeypatch, pairs) -> None:
    """Install ``(module, name, value)`` spies through *monkeypatch*."""
    for module, name, value in pairs:
        monkeypatch.setattr(module, name, value)


def fk_spies(raw_lookup=None, seam_lookup=None, *, stage_e=False, planted_angles=None):
    """Return ``(pairs, recorder)``: recording wrappers of every call-time
    seam the (f) to (k) oracles watch, to install with
    :func:`fk_patched` or :func:`fk_install`.

    Wrapped: ``_engine._run_chunks`` (``fit_indices``, ``fm_route``,
    ``chunksize``), ``_engine._fit_chunk`` (row width),
    ``_engine.fit_pattern`` (the raw target's map index by *raw_lookup*,
    ``h0``, ``max_iterations``), ``_gpu._run_chunks_gpu`` (``chunksize``,
    ``row_slots``, ``seed_extras``, the returned packed rows; it counts
    the passes), ``_gpu._default_batch_size`` (its ``fourier_mellin``
    keyword and result), ``_batched.seed_spectra`` and
    ``_batched.seed_homographies`` (per call: ``P``, the route, the
    slot identities by *seam_lookup* or ``pattern_index``, the
    ``extras`` object before and after, the outputs, the rows, the FM
    state, the criteria and ``fourier_mellin_rows`` calls made INSIDE
    it, and with *stage_e* the Stage E rows of the same batch),
    ``_batched.build_resident``, ``_batched.run_lockstep`` (resident,
    ``real``, ``h0``, the pass), and in ``_fourier_mellin`` the
    criteria, the rows, the angles (with the optional
    ``planted_angles(record, theta) -> theta`` hook, *record* the
    enclosing seam call's) and ``build_fourier_mellin_state``.
    Thread safe: the seam record is thread local, as the chunks run in
    dask's worker threads."""
    rec = types.SimpleNamespace(
        run_chunks=[],
        run_chunks_gpu=[],
        fit_chunks=[],
        fits=[],
        spectra=[],
        seams=[],
        criteria_orphans=[],
        builds=[],
        residents=[],
        lockstep=[],
        default_batch=[],
        pass_number=0,
    )
    lock = threading.Lock()
    run_chunks = _engine._run_chunks
    fit_chunk = _engine._fit_chunk
    fit = _engine.fit_pattern
    run_chunks_gpu = _gpu._run_chunks_gpu
    default_batch_size = _gpu._default_batch_size
    spectra = _batched.seed_spectra
    homographies = _batched.seed_homographies
    build_resident = _batched.build_resident
    lockstep = _batched.run_lockstep
    criteria = _fourier_mellin.fourier_mellin_criteria
    rows_branch = _fourier_mellin.fourier_mellin_rows
    angles = _fourier_mellin.fourier_mellin_angles
    build_fm = _fourier_mellin.build_fourier_mellin_state

    # the spies bind their arguments to the wrapped signature, so they
    # are agnostic to positional against keyword calls and forward any
    # later argument unchanged (critic F5, 2026-10-08)
    def bound(function, args, kwargs) -> dict:
        arguments = inspect.signature(function).bind(*args, **kwargs).arguments
        extra = arguments.pop("kwargs", None) or arguments.pop("kw", None) or {}
        return {**arguments, **extra}

    def run_chunks_spy(*args, **kwargs):
        named = bound(run_chunks, args, kwargs)
        route = named.get("fm_route")
        rec.run_chunks.append(
            {
                "fit_indices": np.array(named["fit_indices"], dtype=np.int64),
                "fm_route": None if route is None else np.array(route),
                "chunksize": named["chunksize"],
            }
        )
        return run_chunks(*args, **kwargs)

    def fit_chunk_spy(*args, **kwargs):
        out = fit_chunk(*args, **kwargs)
        with lock:
            rec.fit_chunks.append({"width": int(np.asarray(out).shape[1])})
        return out

    def fit_spy(state, target, **kwargs):
        result = fit(state, target, **kwargs)
        h0 = kwargs.get("h0")
        with lock:
            rec.fits.append(
                {
                    "index": fk_identify(target, raw_lookup),
                    "h0": None if h0 is None else np.array(h0),
                    "max_iterations": kwargs.get("max_iterations", 50),
                    "converged": bool(result["converged"]),
                }
            )
        return result

    def run_chunks_gpu_spy(*args, **kwargs):
        named = bound(run_chunks_gpu, args, kwargs)
        rec.pass_number += 1
        extras = named.get("seed_extras")
        record = {
            "fit_indices": np.array(named["fit_indices"], dtype=np.int64),
            "chunksize": named["chunksize"],
            "row_slots": dict(named.get("row_slots") or {}),
            "seed_extras": None
            if extras is None
            else {key: np.array(value) for key, value in extras.items()},
            "has_seed_extras": "seed_extras" in named,
        }
        rec.run_chunks_gpu.append(record)
        packed = run_chunks_gpu(*args, **kwargs)
        record["packed"] = np.array(packed)
        return packed

    def default_batch_spy(*args, **kwargs):
        result = default_batch_size(*args, **kwargs)
        rec.default_batch.append(
            {"fourier_mellin": kwargs.get("fourier_mellin", False), "result": result}
        )
        return result

    def spectra_spy(ctx, batch, seed_state):
        with lock:
            rec.spectra.append(
                {
                    "P": int(np.asarray(batch.pattern_index).size),
                    "route": fk_route_of(batch),
                    "identity": fk_slot_identity(batch, seam_lookup),
                }
            )
        return spectra(ctx, batch, seed_state)

    def homographies_spy(ctx, batch, target_spectra, seed_state):
        record = {
            "P": int(np.asarray(batch.pattern_index).size),
            "pattern_index": np.array(batch.pattern_index, dtype=np.int64),
            "route": fk_route_of(batch),
            "identity": fk_slot_identity(batch, seam_lookup),
            "extras_object": batch.extras,
            "fm_state": seed_state.fourier_mellin,
            "criteria": [],
            "rows_calls": 0,
        }
        if stage_e:
            bare_batch = _batched.SeedBatch(
                batch.targets, batch.coefficients, batch.pattern_index
            )
            bare_state = _batched.SeedState(
                seed_state.bounds,
                seed_state.reference_spectrum,
                seed_state.precision,
                seed_state.upsample_factor,
            )
            record["stage_e"] = np.array(
                homographies(ctx, bare_batch, target_spectra, bare_state)
            )
        _FK_LOCAL.record = record
        try:
            rows = homographies(ctx, batch, target_spectra, seed_state)
        finally:
            _FK_LOCAL.record = None
        record["extras_same_object"] = batch.extras is record["extras_object"]
        record["extras_keys"] = set(batch.extras)
        record["route_after"] = fk_route_of(batch)
        record["outputs"] = {
            key: np.array(value) for key, value in batch.outputs.items()
        }
        record["rows"] = np.array(rows)
        with lock:
            rec.seams.append(record)
        return rows

    def build_resident_spy(ctx, state):
        resident = build_resident(ctx, state)
        with lock:
            rec.residents.append(resident)
        return resident

    def lockstep_spy(*args, **kwargs):
        named = bound(lockstep, args, kwargs)
        with lock:
            rec.lockstep.append(
                {
                    "pass": rec.pass_number,
                    "resident": named["resident"],
                    "real": np.array(named["real"], dtype=bool),
                    "h0": np.array(named["h0"]),
                }
            )
        return lockstep(*args, **kwargs)

    def criteria_spy(ctx, batch, fm_state, rows):
        values = criteria(ctx, batch, fm_state, rows)
        entry = {
            "P": int(np.asarray(batch.pattern_index).size),
            "rows": np.array(rows),
            "values": np.array(values),
        }
        record = getattr(_FK_LOCAL, "record", None)
        if record is None:
            with lock:
                rec.criteria_orphans.append(entry)
        else:
            record["criteria"].append(entry)
        return values

    def rows_spy(ctx, batch, target_spectra, seed_state, h_t, route):
        record = getattr(_FK_LOCAL, "record", None)
        if record is not None:
            record["rows_calls"] += 1
        return rows_branch(ctx, batch, target_spectra, seed_state, h_t, route)

    def angles_spy(ctx, target_spectra, fm_state):
        theta, peak = angles(ctx, target_spectra, fm_state)
        record = getattr(_FK_LOCAL, "record", None)
        if planted_angles is not None and record is not None:
            theta = ctx.xp.asarray(planted_angles(record, np.array(theta)))
        return theta, peak

    def build_fm_spy(ctx, state, resident):
        result = build_fm(ctx, state, resident)
        with lock:
            rec.builds.append({"state": state, "resident": resident, "result": result})
        return result

    pairs = [
        (_engine, "_run_chunks", run_chunks_spy),
        (_engine, "_fit_chunk", fit_chunk_spy),
        (_engine, "fit_pattern", fit_spy),
        (_gpu, "_run_chunks_gpu", run_chunks_gpu_spy),
        (_gpu, "_default_batch_size", default_batch_spy),
        (_batched, "seed_spectra", spectra_spy),
        (_batched, "seed_homographies", homographies_spy),
        (_batched, "build_resident", build_resident_spy),
        (_batched, "run_lockstep", lockstep_spy),
        (_fourier_mellin, "fourier_mellin_criteria", criteria_spy),
        (_fourier_mellin, "fourier_mellin_rows", rows_spy),
        (_fourier_mellin, "fourier_mellin_angles", angles_spy),
        (_fourier_mellin, "build_fourier_mellin_state", build_fm_spy),
    ]
    return pairs, rec


def fk_planted_angle_failure(target_index: int, route_code: int = ROUTE_FORCED):
    """Return a ``planted_angles`` hook of :func:`fk_spies` that makes the
    angle NaN on every slot of map point *target_index* carrying
    *route_code* (the retry's forced slot by default)."""

    def planted(record, theta):
        theta = np.array(theta, dtype=np.float64)
        route = record["route"]
        if route is None:
            return theta
        for slot, index in enumerate(record["identity"]):
            if index == target_index and int(route[slot]) == route_code:
                theta[slot] = np.nan
        return theta

    return planted


# ------------------------- Seam-level helpers ------------------------ #


def fk_fm_seed_state(ctx, state, *, precision="complex128", upsample_factor=16):
    """Return ``(seed_state, resident)``: the numpy ``SeedState`` of
    *state* carrying its ``FourierMellinState`` (D22.6) and the grain's
    numpy subregion resident the FM state holds."""
    plain = _batched.build_seed_state(
        ctx, state, precision=precision, upsample_factor=upsample_factor
    )
    resident = _batched.build_resident(ctx, state)
    fm_state = _fourier_mellin.build_fourier_mellin_state(ctx, state, resident)
    assert fm_state is not None
    seed_state = _batched.SeedState(
        plain.bounds,
        plain.reference_spectrum,
        plain.precision,
        plain.upsample_factor,
        fourier_mellin=fm_state,
    )
    return seed_state, resident


def fk_seam(state, targets, routes, *, pattern_index=None, precision="complex128"):
    """Return one numpy seam call on *targets* against *state* with the
    route key *routes*: a namespace with ``rows`` (host), ``outputs``
    (host copies), ``h_t`` (the Stage E rows of the same targets),
    ``batch``, ``spectra``, ``spectra_copy`` (taken before the rows),
    ``seed_state``, ``resident``, ``ctx`` and ``route``."""
    ctx = numpy_seed_context()
    seed_state, resident = fk_fm_seed_state(ctx, state, precision=precision)
    batch = numpy_seed_batch(state, targets, pattern_index)
    route = np.asarray(routes, dtype=np.int8)
    batch.extras[ROUTE_KEY] = route
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    spectra_copy = np.array(spectra)
    rows = _batched.seed_homographies(ctx, batch, spectra, seed_state)
    stage_e, _, _ = numpy_seed_rows(state, targets, precision=precision)
    return types.SimpleNamespace(
        ctx=ctx,
        batch=batch,
        spectra=spectra,
        spectra_copy=spectra_copy,
        seed_state=seed_state,
        resident=resident,
        rows=np.array(rows),
        outputs={key: np.array(value) for key, value in batch.outputs.items()},
        h_t=np.array(stage_e),
        route=route,
    )


def fk_v8_state(reference=None, **kwargs) -> ReferenceState:
    """Return the engine-equal ``ReferenceState`` of the V8 geometry
    (``PC_SEEDING``), keywords as :func:`make_state`."""
    reference = oracle_reference() if reference is None else reference
    return make_state(
        np.asarray(reference), pc_pixels_of(make_detector(pc=PC_SEEDING)), **kwargs
    )


# ---------------------------- Map builders --------------------------- #


@functools.lru_cache(maxsize=1)
def fk_mixed_map() -> dict:
    """Return FK-MIXED (see the ``FK_MIXED_`` constants).  Keys
    ``patterns``, ``navigation_shape``, ``detector`` (per point),
    ``navigation_mask``, ``xmap``, ``exact`` (NaN for the unrelated
    point), ``corners``, ``max_iterations``."""
    detector = make_detector(pc=PC_SEEDING)
    reference = oracle_reference()
    pairs = [g1_pair(twist_spec(t), pc=PC_SEEDING) for t in FK_MIXED_TWISTS[1:]]
    patterns = np.stack(
        [
            reference,
            pairs[0]["target"],
            pairs[1]["target"],
            pairs[2]["target"],
            g6_pair()["target"],
            reference,
        ]
    )
    exact = np.full((6, N_HOMOGRAPHY_PARAMETERS), np.nan)
    exact[[FK_MIXED_REFERENCE, FK_MIXED_MASKED]] = 0.0
    for column, pair in zip((1, 2, 3), pairs):
        exact[column] = pair["exact"]
    mask = np.zeros(FK_MIXED_NAVIGATION_SHAPE, dtype=bool)
    mask[0, FK_MIXED_MASKED] = True
    g_r = g_reference_matrix()
    matrices = np.tile(g_r, (6, 1, 1))
    matrices[FK_MIXED_UNROUTED] = gate_orientation(g_r, in_plane_fe(0.8), detector)
    matrices[FK_MIXED_ROUTED] = gate_orientation(g_r, in_plane_fe(1.6), detector)
    matrices[FK_MIXED_UNRELATED] = gate_orientation(
        g_r, in_plane_fe(FK_MIXED_UNRELATED_LABEL_DEG), detector
    )
    return {
        "patterns": _read_only(patterns),
        "navigation_shape": FK_MIXED_NAVIGATION_SHAPE,
        "detector": make_detector(
            pc=PC_SEEDING, navigation_shape=FK_MIXED_NAVIGATION_SHAPE
        ),
        "navigation_mask": _read_only(mask),
        "xmap": gate_xmap(matrices, FK_MIXED_NAVIGATION_SHAPE),
        "exact": _read_only(exact),
        "corners": _read_only(subregion_corners(SHAPE_480, pc_pixels_of(detector))),
        "max_iterations": FK_MIXED_MAX_ITERATIONS,
    }


def fk_mixed_kwargs(**kwargs) -> dict:
    """Return the ``run_hrebsd_dic`` keywords of FK-MIXED (reference
    (0, 0), its xmap, mask and budget), each overridable."""
    mixed = fk_mixed_map()
    kwargs.setdefault("reference", (0, 0))
    kwargs.setdefault("xmap", mixed["xmap"])
    kwargs.setdefault("max_iterations", mixed["max_iterations"])
    kwargs.setdefault("navigation_mask", np.array(mixed["navigation_mask"]))
    return kwargs


def fk_mixed_run(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on FK-MIXED (CPU unless overridden)."""
    mixed = fk_mixed_map()
    return run_engine(
        np.asarray(mixed["patterns"]),
        FK_MIXED_NAVIGATION_SHAPE,
        mixed["detector"],
        **fk_mixed_kwargs(**kwargs),
    )


def fk_mixed_session(monkeypatch, **kwargs):
    """Return ``(properties, recorder)`` of FK-MIXED through the numpy
    session (``chunksize`` 8 unless overridden)."""
    mixed = fk_mixed_map()
    kwargs.setdefault("chunksize", 8)
    return run_gpu_numpy(
        monkeypatch,
        np.asarray(mixed["patterns"]),
        FK_MIXED_NAVIGATION_SHAPE,
        mixed["detector"],
        **fk_mixed_kwargs(**kwargs),
    )


@functools.lru_cache(maxsize=1)
def fk_mixed_session_off() -> dict:
    """Return the cached ``"off"`` run of FK-MIXED through the numpy
    session (``chunksize`` 8), treat it read-only."""
    patcher = pytest.MonkeyPatch()
    try:
        properties, _ = fk_mixed_session(patcher, fourier_mellin="off")
    finally:
        patcher.undo()
    return properties


@functools.lru_cache(maxsize=1)
def fk_mixed_lookup() -> np.ndarray:
    """Return the D4-preprocessed FK-MIXED patterns (the CPU route's
    one-slot targets are matched against them)."""
    preprocessed, _ = preprocessed_targets(
        fk_v8_state(), np.asarray(fk_mixed_map()["patterns"])
    )
    return _read_only(preprocessed)


@functools.lru_cache(maxsize=None)
def fk_mixed_cpu(mode: str = "off") -> dict:
    """Return the cached CPU run of FK-MIXED under *mode*."""
    return fk_mixed_run(fourier_mellin=mode)


@functools.lru_cache(maxsize=None)
def fk_mixed_recorded(mode: str) -> tuple:
    """Return the cached ``(properties, recorder)`` of FK-MIXED on the
    CPU under *mode*, every :func:`fk_spies` seam recorded."""
    mixed = fk_mixed_map()
    pairs, rec = fk_spies(np.asarray(mixed["patterns"]), fk_mixed_lookup())
    with fk_patched(pairs):
        properties = fk_mixed_run(fourier_mellin=mode)
    return properties, rec


def fk_assert_mixed_off_premise() -> dict:
    """Assert the FK-MIXED ``"off"`` premise at run time and return the
    ``"off"`` properties."""
    off = fk_mixed_cpu("off")
    converged = np.asarray(off["converged"])
    assert tuple(bool(c) for c in converged[list(FK_MIXED_FITTED)]) == (
        FK_MIXED_OFF_CONVERGED
    )
    homography = np.asarray(off["homography"])
    assert np.isfinite(homography[list(FK_MIXED_RETRIED)]).all()
    return off


@functools.lru_cache(maxsize=1)
def fk_two_grain_map() -> dict:
    """Return FK-TWO-GRAIN (see the ``FK_TWO_GRAIN_`` constants).  Keys
    ``patterns``, ``navigation_shape``, ``detector`` (per point),
    ``xmap``, ``grain_labels``, ``references``, ``pc_px``."""
    detector = make_detector(pc=PC_SEEDING)
    reference = oracle_reference()
    g_r = g_reference_matrix()
    patterns = np.stack(
        [
            reference,
            reference,
            g1_pair(twist_spec(2.5), pc=PC_SEEDING)["target"],
            g1_pair(twist_spec(0.8), pc=PC_SEEDING)["target"],
            g1_pair(twist_spec(-2.0), pc=PC_SEEDING)["target"],
        ]
    )
    matrices = np.stack(
        [
            g_r,
            g_r,
            gate_orientation(g_r, in_plane_fe(2.5), detector),
            gate_orientation(g_r, in_plane_fe(0.8), detector),
            gate_orientation(g_r, in_plane_fe(-2.0), detector),
        ]
    )
    return {
        "patterns": _read_only(patterns),
        "navigation_shape": FK_TWO_GRAIN_NAVIGATION_SHAPE,
        "detector": make_detector(
            pc=PC_SEEDING, navigation_shape=FK_TWO_GRAIN_NAVIGATION_SHAPE
        ),
        "xmap": gate_xmap(matrices, FK_TWO_GRAIN_NAVIGATION_SHAPE),
        "grain_labels": np.array(FK_TWO_GRAIN_LABELS),
        "references": np.array(FK_TWO_GRAIN_REFERENCES, dtype=np.int64),
        "pc_px": _read_only(pc_pixels_of(detector)),
    }


def fk_two_grain_session(monkeypatch, **kwargs):
    """Return ``(properties, recorder)`` of FK-TWO-GRAIN under
    ``"auto"`` through the numpy session (``chunksize`` 2: batches
    (0, 2), (4, pad), (1, 3))."""
    fk2 = fk_two_grain_map()
    kwargs.setdefault("chunksize", 2)
    kwargs.setdefault("fourier_mellin", "auto")
    kwargs.setdefault("max_iterations", FK_TWO_GRAIN_MAX_ITERATIONS)
    return run_gpu_numpy(
        monkeypatch,
        np.asarray(fk2["patterns"]),
        FK_TWO_GRAIN_NAVIGATION_SHAPE,
        fk2["detector"],
        xmap=fk2["xmap"],
        grain_labels=fk2["grain_labels"],
        reference=fk2["references"],
        **kwargs,
    )


@functools.lru_cache(maxsize=None)
def fk_ramp(mode: str) -> dict:
    """Return the cached V10(f) ramp run under *mode*: ``"always"`` with
    no xmap, ``"auto"`` with :func:`ramp_xmap`."""
    if mode == "auto":
        return ramp_run(fourier_mellin="auto", xmap=ramp_xmap())
    return ramp_run(fourier_mellin=mode)


def fk_assert_ramp_off_premise() -> dict:
    """Assert the V8(a) premise of V10(f) on the ``"off"`` run at run
    time and return it."""
    off = ramp_off()
    converged = np.asarray(off["converged"])
    homography = np.asarray(off["homography"])
    assert converged[[0, 1, 2]].all()
    for column in range(3, 7):
        assert not converged[column], column
    assert not converged[RAMP_UNFITTABLE_INDEX]
    assert converged[RAMP_RESCUE_INDEX]
    assert int(off["num_iterations"][RAMP_RESCUE_INDEX]) == RAMP_RESCUE_OFF_ITERATIONS
    assert np.isnan(homography[RAMP_FAILED_INDEX]).all()
    assert int(off["num_iterations"][RAMP_FAILED_INDEX]) == 0
    assert not converged[RAMP_FAILED_INDEX]
    assert _engine.SEED_ROUND_PROP_NAME not in off
    return off


@functools.lru_cache(maxsize=None)
def fk_g8(mode: str) -> tuple:
    """Return the cached ``(properties, recorder)`` of G8 under *mode*;
    ``"off"`` carries no recorder."""
    if mode == "off":
        return g8_run(), None
    g8 = g8_map()
    pairs, rec = fk_spies(np.asarray(g8["patterns"]))
    with fk_patched(pairs):
        properties = g8_run(fourier_mellin=mode)
    return properties, rec


def fk_assert_g8_off_premise() -> dict:
    """Assert G8's ``"off"`` premises (V10 G8) at run time and return
    the ``"off"`` properties."""
    off, _ = fk_g8("off")
    converged = np.asarray(off["converged"])
    homography = np.asarray(off["homography"])
    assert converged[[G8_REFERENCE, G8_EASY_ZERO, G8_EASY_SMALL]].all()
    for index in (G8_MISLABELLED, G8_UNRELATED):
        assert not converged[index], index
        assert np.isfinite(homography[index]).all(), index
    assert np.isnan(homography[G8_CONSTANT]).all()
    assert int(off["num_iterations"][G8_CONSTANT]) == 0
    assert not converged[G8_CONSTANT]
    return off


def fk_g8_fitted() -> np.ndarray:
    """Return G8's fitted flat indices in fit order (one grain)."""
    mask = np.asarray(g8_map()["navigation_mask"]).ravel()
    return np.flatnonzero(~mask)


def fk_g8_forced(index: int):
    """Return the numpy seam at P = 1 with route 2 on G8 point *index*
    (the V10(i) oracle's ``fm_row`` source)."""
    target = np.asarray(g8_map()["patterns"])[index]
    return fk_seam(fk_v8_state(), [target], [ROUTE_FORCED], pattern_index=[index])


def fk_gate_probe(gate_map: dict, *, navigation_mask=None) -> tuple:
    """Run ``run_hrebsd_dic(fourier_mellin="auto")`` on a G7 gate map
    (every pattern grain A's reference, which the gate never reads) and
    stop at the first ``_run_chunks`` call; return ``(captured,
    messages)``: the first pass's ``fit_indices`` and ``fm_route``, and
    every warning message issued (none silenced)."""
    navigation_shape = gate_map["navigation_shape"]
    size = navigation_shape[0] * navigation_shape[1]
    patterns = np.repeat(np.asarray(g1_reference())[None], size, axis=0)
    detector = gate_map["detector"]
    captured = {}

    def stop(patterns, fit_indices, state_of_point, states, chunksize, **kwargs):
        captured["fit_indices"] = np.array(fit_indices, dtype=np.int64)
        route = kwargs.get("fm_route")
        captured["fm_route"] = None if route is None else np.array(route)
        raise _FkAbort

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fk_patched([(_engine, "_run_chunks", stop)]):
            with pytest.raises(_FkAbort):
                run_hrebsd_dic(
                    patterns,
                    navigation_shape,
                    detector,
                    xmap=gate_map["xmap"],
                    reference=(0, 0),
                    fourier_mellin="auto",
                    navigation_mask=navigation_mask,
                    verbose=0,
                )
    return captured, [str(w.message) for w in caught]


def fk_twists(gate_map: dict) -> np.ndarray:
    """Return the production twists of a G7 map's points against their
    grain references (D22.7)."""
    twist = _fourier_mellin.twist_about_detector_normal(
        gate_map["xmap"],
        gate_map["detector"],
        gate_map["point_index"],
        gate_map["reference_index"],
    )
    twist = np.asarray(twist)
    assert twist.dtype == np.float64
    assert twist.shape == (len(gate_map["point_index"]),)
    return twist


def fk_rotation_angle_deg(matrix) -> float:
    """Return the rotation angle of a ``(3, 3)`` rotation, deg."""
    cosine = np.clip((np.trace(np.asarray(matrix)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def fk_hrebsd_source(name: str) -> str:
    """Return the source text of one ``_hrebsd`` module."""
    directory = pathlib.Path(_batched.__file__).parent
    return (directory / f"{name}.py").read_text(encoding="utf-8")


def fk_module_scope_imports(tree) -> list:
    """Return ``(node, module name)`` of every import executed at module
    scope, nested module-level blocks included, never inside a function
    or class body (a copy of ``test_hrebsd_gpu.py``'s
    ``module_scope_imports`` as of 2026-10-08)."""
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
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    children = getattr(node, field, None)
                    if isinstance(children, list):
                        visit(children)

    visit(tree.body)
    return found


# ============================ (f) ================================== #


class TestFourierMellinRampRescue:
    """V10(f), D22.7, D22.8, D22.10: the V8 ramp map rescued without
    propagation (``seed_from_neighbors=False``, budget 200) under
    ``"always"`` (no xmap) and ``"auto"`` (the matching G7-built xmap,
    :func:`ramp_xmap`), the V8(a) premise asserted at run time on the
    ``"off"`` run.

    Mutants: FM33 (props written on ``"off"`` runs, or seed codes 1 and
    2 swapped: the routed ramp points carry code 1, the first pass's,
    never 2; the ``"off"`` prop set is pinned in the premise).  The
    constant pattern is written for the MEASURED state of
    ``G8_CONSTANT_TRANSLATION_ROW_FINITE``: its outcome is asserted
    (the D2.6 contract bitwise ``"off"``), its angle is not (finite
    without a guard, NaN with one)."""

    def test_always_rescues_the_far_ramp_and_the_eight_degree_point(self):
        fk_assert_ramp_off_premise()
        _, _, exact, corners = ramp_map()
        properties = fk_ramp("always")
        converged = np.asarray(properties["converged"])
        homography = np.asarray(properties["homography"])
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        for column in range(3, 7):
            assert converged[column], column
            assert fk_same_optimum(homography[column], exact[column], corners), column
            assert seed[column] == SEED_FIRST_PASS, column
            assert np.isfinite(angle[column]), column
        index = RAMP_UNFITTABLE_INDEX
        assert converged[index]
        assert seed[index] == SEED_FIRST_PASS
        assert np.isfinite(angle[index])
        assert_within(
            recovery_error(homography[index], exact[index], corners),
            FM_RAMP_FAR_TOL_PX,
            "FM_RAMP_FAR_TOL_PX",
        )

    def test_always_keeps_the_fm_row_on_the_tilt_and_pins_the_codes(self):
        fk_assert_ramp_off_premise()
        _, _, exact, corners = ramp_map()
        properties = fk_ramp("always")
        converged = np.asarray(properties["converged"])
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        index = RAMP_RESCUE_INDEX
        assert converged[index]
        assert fk_same_optimum(properties["homography"][index], exact[index], corners)
        assert_count(
            int(properties["num_iterations"][index]),
            FM_RAMP_RESCUE_ALWAYS_ITERATIONS,
            "FM_RAMP_RESCUE_ALWAYS_ITERATIONS",
        )
        for column in (0, 1, 2):
            assert converged[column], column
            assert np.isfinite(angle[column]), column
        assert np.isfinite(angle[index])
        # The codes the spec pins "as measured": the reference (expected
        # 0, both criteria at rounding level), ramp columns 1 and 2
        # (expected 1), the 9 degree tilt (expected 1, criterion 1.279
        # against 1.293) and the constant pattern (0 if its acceptance
        # kept the translation row, else 1; see the class docstring)
        measured = {
            i: int(seed[i]) for i in (0, 1, 2, RAMP_RESCUE_INDEX, RAMP_FAILED_INDEX)
        }
        fk_assert_pinned_mapping(measured, FM_RAMP_ALWAYS_CODES, "FM_RAMP_ALWAYS_CODES")

    def test_auto_rescues_exactly_the_routed_points(self):
        off = fk_assert_ramp_off_premise()
        _, _, exact, corners = ramp_map()
        properties = fk_ramp("auto")
        converged = np.asarray(properties["converged"])
        homography = np.asarray(properties["homography"])
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        # routed by the gate (|twist| >= 1.5 deg): ramp columns 2 to 6
        # and the 8 degree point; code 1 (the FIRST pass: FM33's swap
        # would write 2)
        for index in (2, 3, 4, 5, 6, RAMP_UNFITTABLE_INDEX):
            assert converged[index], index
            assert seed[index] == SEED_FIRST_PASS, index
            assert np.isfinite(angle[index]), index
        for column in range(2, 7):
            assert fk_same_optimum(homography[column], exact[column], corners), column
        assert_within(
            recovery_error(
                homography[RAMP_UNFITTABLE_INDEX],
                exact[RAMP_UNFITTABLE_INDEX],
                corners,
            ),
            FM_RAMP_FAR_TOL_PX,
            "FM_RAMP_FAR_TOL_PX",
        )
        # not routed (twist 0 or 0.8 deg) and converged: bitwise "off",
        # code 0, never estimated
        for index in (0, 1, RAMP_RESCUE_INDEX):
            fk_assert_point_bitwise(properties, off, index)
            assert seed[index] == SEED_TRANSLATION, index
            assert np.isnan(angle[index]), index
        assert fk_same_optimum(
            homography[RAMP_RESCUE_INDEX], exact[RAMP_RESCUE_INDEX], corners
        )

    @pytest.mark.parametrize("mode", ["always", "auto"])
    def test_the_constant_pattern_keeps_the_failure_contract(self, mode):
        off = fk_assert_ramp_off_premise()
        # the class docstring: the arm is written for the measured state
        # of the default band-pass, where the constant crop's
        # translation row is finite
        assert G8_CONSTANT_TRANSLATION_ROW_FINITE
        properties = fk_ramp(mode)
        index = RAMP_FAILED_INDEX
        fk_assert_point_bitwise(properties, off, index)
        assert np.isnan(np.asarray(properties["homography"])[index]).all()
        assert int(properties["num_iterations"][index]) == 0
        assert not properties["converged"][index]
        if mode == "auto":
            assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION

    @pytest.mark.parametrize("mode", ["always", "auto"])
    def test_masked_points_and_the_prop_set(self, mode):
        fk_assert_ramp_off_premise()
        _, mask, _, _ = ramp_map()
        properties = fk_ramp(mode)
        fk_assert_fm_props(properties, RAMP_SIZE)
        assert _engine.SEED_ROUND_PROP_NAME not in properties
        masked = np.flatnonzero(np.asarray(mask).ravel())
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        assert (seed[masked] == SEED_NONE).all()
        assert np.isnan(angle[masked]).all()
        fitted = np.flatnonzero(~np.asarray(mask).ravel())
        assert np.isin(
            seed[fitted], (SEED_TRANSLATION, SEED_FIRST_PASS, SEED_RETRY)
        ).all()


# ============================ (g) ================================== #


@functools.lru_cache(maxsize=None)
def fk_g5_run(mode: str) -> dict:
    """Return the cached one-row G5 map run (reference at (0, 0), target
    at (0, 1), ``PC_480`` tiled, budget 50) under *mode*."""
    pair = g5_pair()
    patterns = np.stack([np.asarray(pair["reference"]), np.asarray(pair["target"])])
    return run_engine(
        patterns,
        (1, 2),
        make_detector(navigation_shape=(1, 2)),
        reference=(0, 0),
        max_iterations=G1_CAPTURE_MAX_ITERATIONS,
        filter_cutoffs=G5_FILTER_CUTOFFS,
        fourier_mellin=mode,
    )


def fk_planted_pair():
    """Return ``(state, target)`` of the planted-acceptance arms: G1 at
    :data:`FK_PLANTED_TWIST_DEG`, border 0.05, default band-pass."""
    pair = g1_pair(twist_spec(FK_PLANTED_TWIST_DEG))
    return g1_state(pair), np.asarray(pair["target"])


class TestFourierMellinAcceptance:
    """V10(g), D22.5: the acceptance at the seed by the D2.7 criterion,
    on G5 (the anchor pair; BUILT, both run-time premises hold), planted
    wrong angles, ties and NaN criteria, the G6 unrelated pair, and the
    criterion call counts on the CPU route and in the numpy session.

    Mutants: FM10 (acceptance inverted: the G5 arm and the unplanted
    control of the wrong-angle arm see the FM row refused; the
    wrong-angle arm sees it kept), FM11 (acceptance dropped: the
    wrong-angle and tie arms), FM12 (acceptance by the
    phase-correlation peak: G5, where the zero-shift anchor's peak
    wins and would refuse the FM row, and G6, where the peaks order
    opposite to the criteria and would keep it; both orderings asserted
    as local premises), FM14 (ties keep ``h_FM``: the tie
    arm), FM26 (the retry run WITH the acceptance: the criterion count,
    none for route 2), FM51 (a NaN ``h_T`` criterion treated as a loss:
    the NaN-criterion arm)."""

    def test_g5_premises_and_the_fm_row_is_kept(self, monkeypatch):
        pair = g5_pair()
        state = g1_state(pair, filter_cutoffs=G5_FILTER_CUTOFFS)
        target = np.asarray(pair["target"])
        # premise (i): the D5 seed is pulled to EXACTLY (0, 0)
        seed = cpu_seed_rows(state, [target])[0]
        assert (seed[2], seed[5]) == G5_SEED_PREMISE
        # premise (ii), local recipe
        imposed = rotation_angle_of(pair["exact"])
        local = float(local_state_angles(state, [target])[0][0])
        assert abs(local - imposed) <= LOCAL_G5_ANGLE_TOL_DEG
        # premise (iii), the FM12 separation (critic F2, 2026-10-08):
        # the phase-correlation peak prefers h_T (measured 0.288 against
        # 0.253), the criterion h_FM, so acceptance by the peak refuses
        # the FM row this arm requires
        peak_t, peak_fm = local_phase_peaks(state, target)
        assert peak_t > peak_fm, (peak_t, peak_fm)
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        seam = fk_seam(state, [target], [ROUTE_ACCEPT])
        angle = seam.outputs[ANGLE_KEY]
        assert_within(
            abs(float(angle[0]) - imposed), FM_ANGLE_TOL_DEG, "FM_ANGLE_TOL_DEG"
        )
        assert bool(seam.outputs[APPLIED_KEY][0])
        assert not fk_bitwise(seam.rows[0], seam.h_t[0])
        # the FM seam call (``fk_seam``'s Stage E reference call carries
        # no route key)
        (record,) = [r for r in rec.seams if r["route"] is not None]
        assert len(record["criteria"]) == 2
        by_row = {}
        for entry in record["criteria"]:
            key = "h_t" if fk_bitwise(entry["rows"][0], seam.h_t[0]) else "h_fm"
            by_row[key] = float(entry["values"][0])
        assert set(by_row) == {"h_t", "h_fm"}
        assert by_row["h_fm"] < by_row["h_t"]

    def test_g5_the_fm_seeded_fit_converges_where_off_does_not(self):
        off = fk_g5_run("off")
        assert not off["converged"][1]
        properties = fk_g5_run("always")
        assert properties["converged"][1]
        assert properties["fourier_mellin_seed"][1] == SEED_FIRST_PASS
        assert_count(
            int(properties["num_iterations"][1]), FM_G5_ITERATIONS, "FM_G5_ITERATIONS"
        )

    def test_a_planted_wrong_angle_keeps_the_translation_row(self, monkeypatch):
        state, target = fk_planted_pair()
        # premise, local: the planted FM row loses to the translation row
        h_t_cpu = cpu_seed_rows(state, [target])[0]
        theta = float(local_state_angles(state, [target])[0][0])
        planted_row, _, _ = local_fm_row(
            state, target, theta_deg=theta + FK_PLANTED_ANGLE_OFFSET_DEG
        )
        assert local_criterion(state, target, planted_row) > local_criterion(
            state, target, h_t_cpu
        )
        # the control: unplanted, the FM row wins (FM10)
        control = fk_seam(state, [target], [ROUTE_ACCEPT])
        assert bool(control.outputs[APPLIED_KEY][0])
        original = _fourier_mellin.fourier_mellin_angles

        def planted(ctx, target_spectra, fm_state):
            theta_hat, peak = original(ctx, target_spectra, fm_state)
            return theta_hat + FK_PLANTED_ANGLE_OFFSET_DEG, peak

        monkeypatch.setattr(_fourier_mellin, "fourier_mellin_angles", planted)
        seam = fk_seam(state, [target], [ROUTE_ACCEPT])
        assert fk_bitwise(seam.rows, seam.h_t)
        assert not bool(seam.outputs[APPLIED_KEY][0])
        assert np.isfinite(seam.outputs[ANGLE_KEY][0])

    def test_a_planted_tie_keeps_the_translation_row(self, monkeypatch):
        state, target = fk_planted_pair()
        calls = []

        def tie(ctx, batch, fm_state, rows):
            calls.append(np.array(rows))
            return ctx.xp.zeros(int(rows.shape[0]), dtype=ctx.xp.float64)

        monkeypatch.setattr(_fourier_mellin, "fourier_mellin_criteria", tie)
        seam = fk_seam(state, [target], [ROUTE_ACCEPT])
        assert len(calls) == 2
        assert fk_bitwise(seam.rows, seam.h_t)
        assert not bool(seam.outputs[APPLIED_KEY][0])

    def test_a_nan_translation_criterion_keeps_the_translation_row(self, monkeypatch):
        state, target = fk_planted_pair()
        h_t = numpy_seed_rows(state, [target])[0]
        calls = []

        def nan_for_h_t(ctx, batch, fm_state, rows):
            rows = np.asarray(rows)
            calls.append(np.array(rows))
            values = np.zeros(rows.shape[0], dtype=np.float64)
            for slot in range(rows.shape[0]):
                if fk_bitwise(rows[slot], np.asarray(h_t)[slot]):
                    values[slot] = np.nan
            return ctx.xp.asarray(values)

        monkeypatch.setattr(_fourier_mellin, "fourier_mellin_criteria", nan_for_h_t)
        seam = fk_seam(state, [target], [ROUTE_ACCEPT])
        # the patch saw both rows, the NaN went to h_T and the finite,
        # lower 0.0 to h_FM
        assert len(calls) == 2
        assert sum(fk_bitwise(rows[0], np.asarray(h_t)[0]) for rows in calls) == 1
        assert fk_bitwise(seam.rows, seam.h_t)
        assert not bool(seam.outputs[APPLIED_KEY][0])

    def test_the_unrelated_pair_refuses_the_fm_row(self):
        state = fk_v8_state()
        target = np.asarray(g6_pair()["target"])
        # the G6 premise at the seed, local recipe (ledger 131 (iii))
        h_t_cpu = cpu_seed_rows(state, [target])[0]
        fm_row, _, _ = local_fm_row(state, target)
        criterion_t = local_criterion(state, target, h_t_cpu)
        criterion_fm = local_criterion(state, target, fm_row)
        assert criterion_t == pytest.approx(
            G6_PREMISE_CRITERIA[0], abs=FK_PREMISE_CRITERION_ATOL
        )
        assert criterion_fm == pytest.approx(
            G6_PREMISE_CRITERIA[1], abs=FK_PREMISE_CRITERION_ATOL
        )
        # the FM12 separation (critic F2, 2026-10-08): the peaks order
        # OPPOSITE to the criteria (measured 0.0174 against 0.0206), so
        # acceptance by the peak would keep the FM row this arm refuses
        peak_t, peak_fm = local_phase_peaks(state, target)
        assert peak_fm > peak_t, (peak_t, peak_fm)
        seam = fk_seam(state, [target], [ROUTE_ACCEPT])
        assert np.isfinite(seam.rows).all()
        assert not bool(seam.outputs[APPLIED_KEY][0])
        assert fk_bitwise(seam.rows, seam.h_t)
        assert np.isfinite(seam.outputs[ANGLE_KEY][0])

    def test_the_unrelated_point_converges_as_off(self):
        off = fk_assert_g8_off_premise()
        properties, _ = fk_g8("always")
        index = G8_UNRELATED
        assert bool(properties["converged"][index]) == bool(off["converged"][index])
        assert np.isfinite(np.asarray(properties["homography"])[index]).all()
        # the translation row won, so the first-pass fit IS the "off" fit
        # (D22.10, h0=None), and the forced retry did not converge
        fk_assert_point_bitwise(properties, off, index)
        assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION

    @pytest.mark.parametrize("mode", ["always", "auto"])
    def test_criterion_calls_on_the_cpu_route(self, mode):
        fk_assert_g8_off_premise()
        _, rec = fk_g8(mode)
        assert rec.seams
        assert rec.criteria_orphans == []
        routes_seen = set()
        for record in rec.seams:
            assert record["P"] == 1
            route = int(record["route"][0])
            routes_seen.add(route)
            if route == ROUTE_ACCEPT:
                assert len(record["criteria"]) == 2
                for entry in record["criteria"]:
                    assert entry["rows"].shape == (1, N_HOMOGRAPHY_PARAMETERS)
                    assert entry["values"].shape == (1,)
            else:
                assert record["criteria"] == []
        # non-vacuous: the retry's forced calls exist in both modes, and
        # "always" routes every fitted point with the acceptance
        assert ROUTE_FORCED in routes_seen
        if mode == "always":
            assert ROUTE_ACCEPT in routes_seen

    def test_criterion_calls_in_the_numpy_session(self, monkeypatch):
        fk_assert_mixed_off_premise()
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        fk_mixed_session(monkeypatch, fourier_mellin="always")
        assert rec.criteria_orphans == []
        n_accept, n_forced = 0, 0
        for record in rec.seams:
            route = record["route"]
            assert route is not None
            if (route == ROUTE_ACCEPT).any():
                n_accept += 1
                assert len(record["criteria"]) == 2
                for entry in record["criteria"]:
                    assert entry["P"] == record["P"]
                    assert entry["rows"].shape == (record["P"], N_HOMOGRAPHY_PARAMETERS)
                    assert entry["values"].shape == (record["P"],)
            else:
                n_forced += int((route == ROUTE_FORCED).any())
                assert record["criteria"] == []
        assert n_accept >= 1
        # the unrelated point's translation row wins and it fails: the
        # retry's forced sub-batch calls no criterion (FM26)
        assert n_forced >= 1


# ============================ (h) ================================== #


class TestFourierMellinGate:
    """V10(h), D22.7: ``twist_about_detector_normal`` on G7 (both
    detectors), the routing rule on planted twists, the engine's
    fail-open routing and masked points, the D22.7 warning, and the
    projection link between the gate and the projected patterns.

    Mutants: FM17 (conjugated the wrong way or without the y flip: the
    sign and value pins on the TILTED detector; on the untilted one
    ``M M = I`` makes it equivalent), FM18 (symmetry reduction dropped:
    the symmetry-equivalent arm, raw angle about 90 deg), FM19 (the
    total misorientation angle gated: the pure out-of-plane arms, total
    3 deg, twist 0), FM20 (the signed twist compared: the negative
    planted twists), FM21 (``>`` instead of ``>=``: the planted
    boundary), FM22 (the gate failing closed: the unindexed, NaN and
    other-phase points, function and engine level), FM23 (every point
    against the map's first reference: the two-grain arm), FM36 (the
    warning missing, or triggered by float equality: the all-identity
    and the constant NON-identity maps).  FM39 (the gate from the
    target's own PC frame) is reviewed-equivalent here: the G7 maps
    carry one projection centre and the twist does not depend on it
    (``M`` is the detector's tilt chain only)."""

    @pytest.mark.parametrize("tilted", [False, True])
    def test_imposed_twists_are_recovered_with_their_sign(self, tilted):
        gate_map = g7_twist_map(tilted)
        twist = fk_twists(gate_map)
        expected = gate_map["expected"]
        assert (np.sign(twist) == np.sign(expected)).all()
        assert_within(
            float(np.max(np.abs(twist - expected))),
            FM_GATE_TWIST_TOL_DEG,
            "FM_GATE_TWIST_TOL_DEG",
        )

    @pytest.mark.parametrize("tilted", [False, True])
    def test_a_symmetry_equivalent_orientation_gives_the_same_twist(self, tilted):
        gate_map = g7_special_map(tilted)
        detector = gate_map["detector"]
        g_r = g_reference_matrix()
        g_t = G7_SYMMETRY_OPERATOR @ gate_orientation(
            g_r, in_plane_fe(G7_SYMMETRY_TWIST_DEG), detector
        )
        # premise: the raw (unreduced) misorientation is about 90 deg
        assert fk_rotation_angle_deg(g_t.T @ g_r) == pytest.approx(90.0, abs=5.0)
        twist = fk_twists(gate_map)[G7_SPECIAL_SYMMETRY - 1]
        assert np.sign(twist) == np.sign(G7_SYMMETRY_TWIST_DEG)
        assert_within(
            abs(float(twist) - G7_SYMMETRY_TWIST_DEG),
            FM_GATE_TWIST_TOL_DEG,
            "FM_GATE_TWIST_TOL_DEG",
        )

    @pytest.mark.parametrize("tilted", [False, True])
    def test_out_of_plane_rotations_give_zero_twist_and_swing_is_ignored(self, tilted):
        gate_map = g7_special_map(tilted)
        detector = gate_map["detector"]
        m = sample_to_detector_matrix(detector)
        g_r = g_reference_matrix()
        twist = fk_twists(gate_map)
        for column in (G7_SPECIAL_ABOUT_X, G7_SPECIAL_ABOUT_Y):
            # premise: a total misorientation of 3 deg, above the gate
            axis = "x" if column == G7_SPECIAL_ABOUT_X else "y"
            g_t = gate_orientation(
                g_r, _axis_rotation(axis, G7_OUT_OF_PLANE_DEG), detector
            )
            total = fk_rotation_angle_deg(m @ (g_t.T @ g_r) @ m.T)
            assert total == pytest.approx(G7_OUT_OF_PLANE_DEG, abs=1e-9)
            assert total >= FROZEN_GATE_DEG
            assert_within(
                abs(float(twist[column - 1])),
                FM_GATE_TWIST_TOL_DEG,
                "FM_GATE_TWIST_TOL_DEG",
            )
        swing = float(twist[G7_SPECIAL_TWIST_SWING - 1])
        assert_within(
            abs(swing - G7_SYMMETRY_TWIST_DEG),
            FM_GATE_TWIST_TOL_DEG,
            "FM_GATE_TWIST_TOL_DEG",
        )

    @pytest.mark.parametrize("tilted", [False, True])
    def test_unusable_points_give_nan(self, tilted):
        gate_map = g7_special_map(tilted)
        twist = fk_twists(gate_map)
        for column in (G7_SPECIAL_UNINDEXED, G7_SPECIAL_NAN, G7_SPECIAL_OTHER_PHASE):
            assert np.isnan(twist[column - 1]), column
        usable = [
            G7_SPECIAL_SYMMETRY,
            G7_SPECIAL_ABOUT_X,
            G7_SPECIAL_ABOUT_Y,
            G7_SPECIAL_TWIST_SWING,
        ]
        assert np.isfinite(twist[[c - 1 for c in usable]]).all()

    @pytest.mark.parametrize("tilted", [False, True])
    def test_each_point_is_measured_against_its_grain_reference(self, tilted):
        gate_map = g7_two_grain_map(tilted)
        twist = fk_twists(gate_map)
        assert (np.sign(twist) == np.sign(gate_map["expected"])).all()
        assert_within(
            float(np.max(np.abs(twist - gate_map["expected"]))),
            FM_GATE_TWIST_TOL_DEG,
            "FM_GATE_TWIST_TOL_DEG",
        )

    def test_the_routes_on_planted_twists(self):
        gate = FROZEN_GATE_DEG
        planted = np.array(
            [gate, np.nextafter(gate, 0.0), -gate, -2.0, np.nan, 0.0, 1.0, 4.8, -0.5]
        )
        routes = _fourier_mellin.fourier_mellin_routes(planted, gate)
        routes = np.asarray(routes)
        assert routes.dtype == np.int8
        assert routes.shape == planted.shape
        assert routes.tolist() == [1, 0, 1, 1, 1, 0, 0, 1, 0]
        assert _fourier_mellin.FM_GATE_DEG == FROZEN_GATE_DEG

    @pytest.mark.parametrize("tilted", [False, True])
    def test_the_engine_fails_open_on_unusable_points(self, tilted):
        gate_map = g7_special_map(tilted)
        captured, _ = fk_gate_probe(gate_map)
        size = gate_map["navigation_shape"][1]
        assert captured["fit_indices"].tolist() == list(range(size))
        route = captured["fm_route"]
        assert route is not None
        assert route.dtype == np.int8
        expected = {
            0: 0,
            G7_SPECIAL_SYMMETRY: 1,
            G7_SPECIAL_ABOUT_X: 0,
            G7_SPECIAL_ABOUT_Y: 0,
            G7_SPECIAL_TWIST_SWING: 1,
            G7_SPECIAL_UNINDEXED: 1,
            G7_SPECIAL_NAN: 1,
            G7_SPECIAL_OTHER_PHASE: 1,
        }
        assert {i: int(route[i]) for i in range(size)} == expected

    def test_masked_points_are_never_routed(self):
        gate_map = g7_twist_map(False)
        expected_twists = np.concatenate([[0.0], gate_map["expected"]])
        # mask the two largest twists (|4.8| deg), which would be routed
        masked = np.flatnonzero(np.abs(expected_twists) > 4.0)
        mask = np.zeros(gate_map["navigation_shape"], dtype=bool)
        mask[0, masked] = True
        captured, _ = fk_gate_probe(gate_map, navigation_mask=mask)
        fit_indices = captured["fit_indices"]
        assert not np.isin(masked, fit_indices).any()
        route = captured["fm_route"]
        assert route.shape == fit_indices.shape
        for slot, index in enumerate(fit_indices):
            twist = abs(expected_twists[index])
            if twist == FROZEN_GATE_DEG:
                # the exact boundary rides on the recovery rounding; the
                # planted-boundary arm pins it
                continue
            assert int(route[slot]) == int(twist >= FROZEN_GATE_DEG), index

    @pytest.mark.parametrize("kind", ["identity", "constant"])
    def test_a_map_without_twists_warns_and_routes_nothing(self, kind):
        gate_map = g7_uniform_map(kind)
        if kind == "constant":
            # the FM36 float-equality kill needs twists that are NOT
            # exactly zero but below the trigger (D22.7: order 1e-15;
            # critic F6, 2026-10-08; magnitude recorded at the
            # implementation gate)
            twists = fk_twists(gate_map)
            assert np.any(twists != 0.0), twists
            assert np.all(np.abs(twists) < FROZEN_ZERO_TWIST_DEG), twists
        captured, messages = fk_gate_probe(gate_map)
        assert messages.count(FOURIER_MELLIN_NO_TWIST_WARNING) == 1
        route = captured["fm_route"]
        assert route is not None
        assert not route.any()

    def test_a_map_with_twists_does_not_warn(self):
        captured, messages = fk_gate_probe(g7_twist_map(False))
        assert FOURIER_MELLIN_NO_TWIST_WARNING not in messages
        assert captured["fm_route"].any()

    @pytest.mark.parametrize("tilted", [False, True])
    @pytest.mark.parametrize("twist_deg", G7_PROJECTION_TWISTS)
    def test_the_projection_link(self, twist_deg, tilted):
        projected, constructed = g7_projected_pair(twist_deg, tilted)
        relative = float(
            np.max(np.abs(projected - constructed)) / np.max(np.abs(constructed))
        )
        assert relative <= PROJECTION_LINK_RTOL
        detector = g7_detector(tilted)
        g_r = g_reference_matrix()
        g_t = gate_orientation(g_r, in_plane_fe(twist_deg), detector)
        two_points = {
            "xmap": gate_xmap(np.stack([g_r, g_t]), (1, 2)),
            "detector": detector,
            "point_index": np.array([1]),
            "reference_index": np.array([0]),
        }
        gate_twist = float(fk_twists(two_points)[0])
        reference = project_pattern(detector, orientation_a())
        state = make_state(reference, pc_pixels_of(detector))
        ctx = numpy_seed_context()
        seed_state, _ = fk_fm_seed_state(ctx, state)
        batch = numpy_seed_batch(state, [projected])
        spectra = _batched.seed_spectra(ctx, batch, seed_state)
        theta, _ = _fourier_mellin.fourier_mellin_angles(
            ctx, spectra, seed_state.fourier_mellin
        )
        theta = float(np.asarray(theta)[0])
        assert np.sign(theta) == np.sign(gate_twist) == np.sign(twist_deg)
        assert_within(abs(theta - gate_twist), FM_ANGLE_TOL_DEG, "FM_ANGLE_TOL_DEG")

    def test_the_noise_population_is_recorded(self, record_property):
        gate_map = g7_noise_map(False)
        twist = fk_twists(gate_map)
        routes = np.asarray(
            _fourier_mellin.fourier_mellin_routes(twist, FROZEN_GATE_DEG)
        )
        # recorded, not asserted (V10(h)): the routed share of a 2.5
        # degree population with 0.5 degree orientation noise
        record_property("fm_gate_noise_routed", int(routes.sum()))
        record_property("fm_gate_noise_size", int(routes.size))
        assert routes.shape == (G7_NOISE_SIZE,)


# ============================ (i) ================================== #


@functools.lru_cache(maxsize=1)
def fk_constant_map() -> dict:
    """Return the ``(None, None)`` constant-pattern arm's one-row map:
    the V8 reference and a constant pattern."""
    patterns = np.stack([oracle_reference(), np.full(SHAPE_480, G8_CONSTANT_VALUE)])
    return {
        "patterns": _read_only(patterns),
        "detector": make_detector(
            pc=PC_SEEDING, navigation_shape=FK_CONSTANT_NAVIGATION_SHAPE
        ),
    }


class TestFourierMellinRetry:
    """V10(i), D22.8: the post-fit retry on G8 (budget 200) under
    ``"auto"`` and ``"always"``, the direct-fit oracle of the retried
    point, the first-result rule, the constant pattern (asserted as
    MEASURED on G8, ``G8_CONSTANT_TRANSLATION_ROW_FINITE``, and with the
    no-second-fit spy on the ``(None, None)`` arm, where its forced FM
    estimate provably fails), and a planted angle failure in the retry
    (on FK-MIXED, budget 20, for CI cost).

    Mutants: FM24 (a non-converged retry replacing the first result:
    the unrelated point), FM25 (the retry on converged or masked points,
    or run twice: the subset spy, exactly two ``_run_chunks`` calls),
    FM27 (``num_iterations`` summed: the direct-fit oracle), FM33 (codes
    1 and 2 swapped: code 2 on the retried point, 1 on the first-pass
    FM point under ``"always"``), FM40 (an unapplied forced slot
    fitted: the ``(None, None)`` and the planted-failure arms), FM47
    (the retry skipped under ``"always"``), FM49 (NaN-``h`` first-pass
    points excluded: the constant pattern is in the subset)."""

    def test_auto_retries_exactly_the_failed_points_once(self):
        fk_assert_g8_off_premise()
        _, rec = fk_g8("auto")
        fitted = fk_g8_fitted()
        assert len(rec.run_chunks) == 2
        first, retry = rec.run_chunks
        assert first["fit_indices"].tolist() == fitted.tolist()
        # nothing is routed before the fit (the xmap reads the
        # reference's orientation, or 0.8 deg)
        assert first["fm_route"] is not None
        assert first["fm_route"].dtype == np.int8
        assert not first["fm_route"].any()
        assert retry["fit_indices"].tolist() == [
            G8_MISLABELLED,
            G8_UNRELATED,
            G8_CONSTANT,
        ]
        assert retry["fm_route"].tolist() == [ROUTE_FORCED] * 3

    def test_the_mislabelled_point_converges_from_its_fm_row(self):
        fk_assert_g8_off_premise()
        properties, _ = fk_g8("auto")
        index = G8_MISLABELLED
        target = np.asarray(g8_map()["patterns"])[index]
        seam = fk_g8_forced(index)
        assert bool(seam.outputs[APPLIED_KEY][0])
        direct = fit_pattern(
            fk_v8_state(), target, h0=seam.rows[0], max_iterations=MAX_ITERATIONS
        )
        assert direct["converged"]
        fk_assert_result_equals_fit(properties, index, direct)
        assert properties["fourier_mellin_seed"][index] == SEED_RETRY
        angle = np.asarray(properties["fourier_mellin_angle"])[index]
        assert fk_bitwise(angle, seam.outputs[ANGLE_KEY][0])
        assert np.isfinite(angle)

    def test_the_unrelated_point_keeps_its_first_result(self):
        off = fk_assert_g8_off_premise()
        properties, _ = fk_g8("auto")
        index = G8_UNRELATED
        target = np.asarray(g8_map()["patterns"])[index]
        # premise at run time: the forced FM-seeded fit does not converge
        seam = fk_g8_forced(index)
        assert bool(seam.outputs[APPLIED_KEY][0])
        forced = fit_pattern(
            fk_v8_state(), target, h0=seam.rows[0], max_iterations=MAX_ITERATIONS
        )
        assert not forced["converged"]
        fk_assert_point_bitwise(properties, off, index)
        assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION
        assert np.isfinite(properties["fourier_mellin_angle"][index])

    def test_converged_and_masked_points_are_never_retried(self):
        off = fk_assert_g8_off_premise()
        properties, rec = fk_g8("auto")
        retried = set(rec.run_chunks[-1]["fit_indices"].tolist())
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        for index in (G8_REFERENCE, G8_EASY_ZERO, G8_EASY_SMALL):
            assert index not in retried
            fk_assert_point_bitwise(properties, off, index)
            assert seed[index] == SEED_TRANSLATION
            assert np.isnan(angle[index])
        assert G8_MASKED not in retried
        assert all(G8_MASKED not in call["fit_indices"] for call in rec.run_chunks)
        assert seed[G8_MASKED] == SEED_NONE
        assert np.isnan(angle[G8_MASKED])
        fk_assert_fm_props(properties, G8_NAVIGATION_SHAPE[1])

    def test_the_constant_pattern_on_g8_keeps_the_failure_contract(self):
        off = fk_assert_g8_off_premise()
        assert G8_CONSTANT_TRANSLATION_ROW_FINITE
        properties, rec = fk_g8("auto")
        index = G8_CONSTANT
        # FM49: a NaN-h first-pass point is in the retry subset
        assert index in rec.run_chunks[-1]["fit_indices"].tolist()
        fk_assert_point_bitwise(properties, off, index)
        assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION
        fits = [f for f in rec.fits if f["index"] == index and f["max_iterations"] != 0]
        assert_count(
            len(fits) - 1, FM_G8_CONSTANT_SECOND_FITS, "FM_G8_CONSTANT_SECOND_FITS"
        )

    def test_a_constant_pattern_without_band_pass_is_never_refitted(self):
        constant = fk_constant_map()
        patterns = np.asarray(constant["patterns"])
        # premise: under (None, None) the constant crop's translation row
        # is NaN throughout, so its forced FM estimate fails (D22.5)
        state = fk_v8_state(filter_cutoffs=FK_CONSTANT_FILTER_CUTOFFS)
        assert np.isnan(cpu_seed_rows(state, [patterns[1]])).all()
        pairs, rec = fk_spies(patterns)
        with fk_patched(pairs):
            properties = run_engine(
                patterns,
                FK_CONSTANT_NAVIGATION_SHAPE,
                constant["detector"],
                reference=(0, 0),
                max_iterations=MAX_ITERATIONS,
                filter_cutoffs=FK_CONSTANT_FILTER_CUTOFFS,
                fourier_mellin="always",
            )
        assert len(rec.run_chunks) == 2
        assert rec.run_chunks[1]["fit_indices"].tolist() == [1]
        assert rec.run_chunks[1]["fm_route"].tolist() == [ROUTE_FORCED]
        fits = [f for f in rec.fits if f["index"] == 1 and f["max_iterations"] != 0]
        assert len(fits) == 1
        assert np.isnan(np.asarray(properties["homography"])[1]).all()
        assert int(properties["num_iterations"][1]) == 0
        assert not properties["converged"][1]
        assert properties["fourier_mellin_seed"][1] == SEED_TRANSLATION
        assert np.isnan(properties["fourier_mellin_angle"][1])

    def test_a_planted_angle_failure_in_the_retry_skips_the_second_fit(self):
        off = fk_assert_mixed_off_premise()
        mixed = fk_mixed_map()
        index = FK_MIXED_MISLABELLED
        pairs, rec = fk_spies(
            np.asarray(mixed["patterns"]),
            fk_mixed_lookup(),
            planted_angles=fk_planted_angle_failure(index),
        )
        with fk_patched(pairs):
            properties = fk_mixed_run(fourier_mellin="auto")
        # it was in the retry subset, its forced seam call was made and
        # failed, and it got no second fit
        assert index in rec.run_chunks[-1]["fit_indices"].tolist()
        forced = [
            r
            for r in rec.seams
            if r["identity"] == [index] and int(r["route"][0]) == ROUTE_FORCED
        ]
        assert len(forced) == 1
        assert not bool(forced[0]["outputs"][APPLIED_KEY][0])
        fits = [f for f in rec.fits if f["index"] == index and f["max_iterations"] != 0]
        assert len(fits) == 1
        fk_assert_point_bitwise(properties, off, index)
        assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION
        assert np.isnan(properties["fourier_mellin_angle"][index])

    def test_always_fits_the_mislabelled_point_in_the_first_pass(self):
        fk_assert_g8_off_premise()
        properties, rec = fk_g8("always")
        converged = np.asarray(properties["converged"])
        seed = np.asarray(properties["fourier_mellin_seed"])
        assert converged[G8_MISLABELLED]
        assert seed[G8_MISLABELLED] == SEED_FIRST_PASS
        fitted = fk_g8_fitted()
        assert len(rec.run_chunks) == 2
        first, retry = rec.run_chunks
        assert first["fit_indices"].tolist() == fitted.tolist()
        assert first["fm_route"].tolist() == [ROUTE_ACCEPT] * fitted.size
        # V10(i) pins the subset LITERALLY (critic F3, 2026-10-08):
        # exactly the unrelated point and the constant pattern, in fit
        # order, both forced
        assert retry["fit_indices"].tolist() == [G8_UNRELATED, G8_CONSTANT]
        assert retry["fm_route"].tolist() == [ROUTE_FORCED] * 2
        # consistency only (circular on its own): the D22.8 subset read
        # back from the outputs -- the fitted points whose acceptance
        # kept the translation row and whose fit did not converge
        expected = [
            int(i)
            for i in fitted
            if (not converged[i] and seed[i] == SEED_TRANSLATION)
            or seed[i] == SEED_RETRY
        ]
        assert retry["fit_indices"].tolist() == expected


# ============================ (j) ================================== #


class TestFourierMellinCpuRoute:
    """V10(j), D22.9, D22.10, D22.13: the CPU route on FK-MIXED (routed,
    unrouted and retried points; budget 20) under ``"auto"``.

    Mutants: FM15 (the route ignored, FM applied to every slot: the
    unrouted 0.8 degree point is no longer bitwise ``"off"``), FM16 (rows
    widened or props written on ``"off"`` runs: the ``_fit_chunk``
    width spy and the absent props), FM30 (the CPU route at a
    chunk-dependent P: the P = 1 spy and chunksize invariance), FM44
    (every CPU point routed through the numpy seam: the
    ``_batched.seed_spectra`` count, the only killer, since the numpy
    seam equals skimage on these fixtures)."""

    def test_unrouted_and_translation_won_points_equal_off(self):
        off = fk_assert_mixed_off_premise()
        properties, rec = fk_mixed_recorded("auto")
        first = rec.run_chunks[0]
        assert first["fit_indices"].tolist() == list(FK_MIXED_FITTED)
        assert first["fm_route"].dtype == np.int8
        assert first["fm_route"].tolist() == list(FK_MIXED_ROUTES)
        seed = np.asarray(properties["fourier_mellin_seed"])
        angle = np.asarray(properties["fourier_mellin_angle"])
        # neither routed nor retried: bitwise "off", never estimated
        for index in (FK_MIXED_REFERENCE, FK_MIXED_UNROUTED):
            fk_assert_point_bitwise(properties, off, index)
            assert seed[index] == SEED_TRANSLATION
            assert np.isnan(angle[index])
        # routed, the translation row won, retried without convergence:
        # bitwise "off", code 0, a finite (the retry's) angle
        fk_assert_point_bitwise(properties, off, FK_MIXED_UNRELATED)
        assert seed[FK_MIXED_UNRELATED] == SEED_TRANSLATION
        assert np.isfinite(angle[FK_MIXED_UNRELATED])
        # routed and the FM row won; retried and converged
        assert properties["converged"][FK_MIXED_ROUTED]
        assert seed[FK_MIXED_ROUTED] == SEED_FIRST_PASS
        assert properties["converged"][FK_MIXED_MISLABELLED]
        assert seed[FK_MIXED_MISLABELLED] == SEED_RETRY

    def test_chunksize_invariance_and_repeatability(self):
        fk_assert_mixed_off_premise()
        reference = fk_mixed_cpu("auto")
        for chunksize in (1, 3):
            assert_properties_bitwise(
                fk_mixed_run(fourier_mellin="auto", chunksize=chunksize), reference
            )
        assert_properties_bitwise(fk_mixed_run(fourier_mellin="auto"), reference)

    def test_lazy_equals_eager(self):
        fk_assert_mixed_off_premise()
        mixed = fk_mixed_map()
        lazy = da.from_array(np.asarray(mixed["patterns"]), chunks=(2, -1, -1))
        properties = run_engine(
            lazy,
            FK_MIXED_NAVIGATION_SHAPE,
            mixed["detector"],
            **fk_mixed_kwargs(fourier_mellin="auto"),
        )
        assert_properties_bitwise(properties, fk_mixed_cpu("auto"))

    def test_the_seam_runs_at_one_slot_for_routed_and_retried_points_only(self):
        fk_assert_mixed_off_premise()
        _, rec = fk_mixed_recorded("auto")
        assert rec.spectra
        assert all(call["P"] == 1 for call in rec.spectra)
        assert all(call["P"] == 1 for call in rec.seams)
        counts = {}
        for call in rec.spectra:
            (identity,) = call["identity"]
            counts[identity] = counts.get(identity, 0) + 1
        # once per route-1 point (first pass) and once per route-2 point
        # (retry); never for an unrouted, unretried one (FM44)
        assert counts == {
            FK_MIXED_ROUTED: 1,
            FK_MIXED_MISLABELLED: 1,
            FK_MIXED_UNRELATED: 2,
        }
        routes = sorted(
            (record["identity"][0], int(record["route"][0])) for record in rec.seams
        )
        assert routes == sorted(
            [
                (FK_MIXED_ROUTED, ROUTE_ACCEPT),
                (FK_MIXED_UNRELATED, ROUTE_ACCEPT),
                (FK_MIXED_MISLABELLED, ROUTE_FORCED),
                (FK_MIXED_UNRELATED, ROUTE_FORCED),
            ]
        )

    def test_fit_chunk_rows_are_14_wide_on_fm_runs_and_12_on_off(self):
        fk_assert_mixed_off_premise()
        _, rec = fk_mixed_recorded("auto")
        assert rec.fit_chunks
        assert {chunk["width"] for chunk in rec.fit_chunks} == {ROW_SLOTS_FM["width"]}
        _, rec_off = fk_mixed_recorded("off")
        assert rec_off.fit_chunks
        assert {chunk["width"] for chunk in rec_off.fit_chunks} == {ROW_SLOTS["width"]}
        assert all(call["fm_route"] is None for call in rec_off.run_chunks)
        assert len(rec_off.run_chunks) == 1
        assert rec_off.spectra == []

    def test_the_props_carry_the_frozen_dtypes_shapes_and_codes(self):
        fk_assert_mixed_off_premise()
        properties = fk_mixed_cpu("auto")
        size = FK_MIXED_NAVIGATION_SHAPE[1]
        fk_assert_fm_props(properties, size)
        seed = np.asarray(properties["fourier_mellin_seed"])
        assert seed.tolist() == [
            SEED_TRANSLATION,
            SEED_TRANSLATION,
            SEED_FIRST_PASS,
            SEED_RETRY,
            SEED_TRANSLATION,
            SEED_NONE,
        ]
        angle = np.asarray(properties["fourier_mellin_angle"])
        assert np.isnan(
            angle[[FK_MIXED_REFERENCE, FK_MIXED_UNROUTED, FK_MIXED_MASKED]]
        ).all()
        assert np.isfinite(
            angle[[FK_MIXED_ROUTED, FK_MIXED_MISLABELLED, FK_MIXED_UNRELATED]]
        ).all()
        off = fk_mixed_cpu("off")
        assert set(off) == set(_engine.STAGE_A_PROP_NAMES)


# ============================ (k) ================================== #


class TestFourierMellinNumpySession:
    """V10(k), D22.6, D22.11, D22.14, D22.18: the device runner under the
    numpy session of V9 (patched ``_gpu._make_session``) with
    ``seed_extras``; the VRAM functions' ``fourier_mellin`` keyword and
    the ``"off"``-first batch choice; the numpy session against the CPU
    route; the complex64 dtype spy; import hygiene.

    Mutants: FM15 (the route ignored: route-0 slots and unrouted
    sub-batches return the Stage E rows bitwise), FM28 (route flags in
    map order, or padded slots routed: FK-TWO-GRAIN, whose fit order
    differs from its map order), FM29 (the FM branch compacted to the
    routed slots: the 1-of-P against P-of-P arm, reviewed-equivalent
    under numpy if numpy's batched FFT is not batch-count sensitive),
    FM37 (FM arithmetic left in complex64: the dtype spy), FM40 (an
    unapplied forced slot fitted: the lockstep spy and its packed
    row), FM45 (outputs written into ``extras``, or ``extras``
    mutated), FM46 (B picked from the FM model where the FM terms fit:
    the fake-VRAM arm), FM48 (the retry at a newly chosen B: the
    ``chunksize`` spy behind a planted out-of-memory halving), FM50 (the
    FM state built eagerly or per sub-batch: the lazy-build spy)."""

    def test_route_flags_arrive_in_fit_order_with_zero_padding(self, monkeypatch):
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        fk_two_grain_session(monkeypatch)
        assert rec.seams
        seen = []
        for record in rec.seams:
            route = record["route"]
            index = record["pattern_index"]
            assert route is not None
            assert route.dtype == np.int8
            assert route.shape == index.shape
            for slot, point in enumerate(index):
                expected = 0 if point < 0 else FK_TWO_GRAIN_ROUTES[int(point)]
                assert int(route[slot]) == expected, (slot, int(point))
            seen.extend(int(point) for point in index if point >= 0)
        assert seen[: len(FK_TWO_GRAIN_FIT_ORDER)] == list(FK_TWO_GRAIN_FIT_ORDER)
        assert any((record["pattern_index"] < 0).any() for record in rec.seams)

    def test_extras_stay_input_only_and_outputs_follow_the_route(self, monkeypatch):
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        fk_two_grain_session(monkeypatch)
        n_routed, n_unrouted = 0, 0
        for record in rec.seams:
            route = record["route"]
            assert record["extras_same_object"]
            assert record["extras_keys"] == {ROUTE_KEY}
            assert fk_bitwise(record["route_after"], route)
            outputs = record["outputs"]
            if route.any():
                n_routed += 1
                assert set(outputs) == {ANGLE_KEY, APPLIED_KEY}
                angle, applied = outputs[ANGLE_KEY], outputs[APPLIED_KEY]
                assert angle.dtype == np.float64
                assert applied.dtype == np.bool_
                assert angle.shape == applied.shape == (record["P"],)
                assert np.isnan(angle[route == ROUTE_NONE]).all()
                assert not applied[route == ROUTE_NONE].any()
                assert np.isfinite(angle[route != ROUTE_NONE]).all()
            else:
                n_unrouted += 1
                assert outputs == {}
        assert n_routed >= 1
        assert n_unrouted >= 1

    def test_unrouted_slots_and_sub_batches_return_the_stage_e_rows(self, monkeypatch):
        pairs, rec = fk_spies(stage_e=True)
        fk_install(monkeypatch, pairs)
        fk_two_grain_session(monkeypatch)
        n_unrouted = 0
        for record in rec.seams:
            route = record["route"]
            if not route.any():
                n_unrouted += 1
                assert record["rows_calls"] == 0
                assert fk_bitwise(record["rows"], record["stage_e"])
            else:
                assert record["rows_calls"] == 1
                keep = route == ROUTE_NONE
                assert fk_bitwise(record["rows"][keep], record["stage_e"][keep])
        assert n_unrouted >= 1

    def test_a_planted_seam_without_outputs_reads_as_not_applied(self, monkeypatch):
        fk_assert_mixed_off_premise()
        off = fk_mixed_session_off()
        original = _batched.seed_homographies

        def planted(ctx, batch, target_spectra, seed_state):
            bare = _batched.SeedState(
                seed_state.bounds,
                seed_state.reference_spectrum,
                seed_state.precision,
                seed_state.upsample_factor,
            )
            return original(ctx, batch, target_spectra, bare)

        monkeypatch.setattr(_batched, "seed_homographies", planted)
        properties, _ = fk_mixed_session(monkeypatch, fourier_mellin="auto")
        for name in _engine.STAGE_A_PROP_NAMES:
            assert fk_bitwise(properties[name], off[name]), name
        seed = np.asarray(properties["fourier_mellin_seed"])
        expected = np.full(FK_MIXED_NAVIGATION_SHAPE[1], SEED_TRANSLATION)
        expected[FK_MIXED_MASKED] = SEED_NONE
        assert seed.tolist() == expected.tolist()
        assert np.isnan(np.asarray(properties["fourier_mellin_angle"])).all()

    def test_a_routed_row_does_not_depend_on_how_many_slots_are_routed(self):
        mixed = fk_mixed_map()
        patterns = np.asarray(mixed["patterns"])
        order = [FK_MIXED_ROUTED, FK_MIXED_MISLABELLED, FK_MIXED_UNROUTED, 0]
        targets = patterns[order]
        state = fk_v8_state()
        one = fk_seam(state, targets, [1, 0, 0, 0], pattern_index=order)
        every = fk_seam(state, targets, [1, 1, 1, 1], pattern_index=order)
        assert fk_bitwise(one.rows[0], every.rows[0])
        assert fk_bitwise(one.outputs[ANGLE_KEY][0], every.outputs[ANGLE_KEY][0])
        assert bool(one.outputs[APPLIED_KEY][0]) == bool(every.outputs[APPLIED_KEY][0])
        assert bool(one.outputs[APPLIED_KEY][0])
        # the masked full-P branch computed the other slots' angles, and
        # the outputs rule reports them NaN (FM52's sibling)
        assert np.isnan(one.outputs[ANGLE_KEY][1:]).all()
        assert not one.outputs[APPLIED_KEY][1:].any()
        assert fk_bitwise(one.rows[1:], one.h_t[1:])

    def test_the_fm_state_is_built_lazily_once_per_routed_grain(self, monkeypatch):
        fk2 = fk_two_grain_map()
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        properties, _ = fk_two_grain_session(monkeypatch)
        # premise: the unrouted grain's points converge, so no retry ever
        # needs that grain's FM state
        converged = np.asarray(properties["converged"])
        assert converged[list(FK_TWO_GRAIN_UNROUTED_GRAIN)].all()
        # one build per runner pass that fits a routed-grain point: the
        # first pass, plus a retry pass (a NEW session, D22.8 / D21.9.1)
        # only if a routed-grain point failed (critic F4, 2026-10-08:
        # robust to that retry; a rebuild per sub-batch still dies)
        routed = set(FK_TWO_GRAIN_ROUTED_GRAIN)
        passes = [
            call
            for call in rec.run_chunks_gpu
            if routed & {int(i) for i in call["fit_indices"]}
        ]
        assert passes and len(rec.builds) == len(passes)
        routed_reference = make_state(
            np.asarray(fk2["patterns"])[FK_TWO_GRAIN_REFERENCES[1]], fk2["pc_px"]
        )
        results = []
        for build in rec.builds:
            assert np.array_equal(
                build["state"].reference_subregion,
                routed_reference.reference_subregion,
            )
            assert any(build["resident"] is resident for resident in rec.residents)
            assert build["result"].resident is build["resident"]
            assert any(entry["resident"] is build["resident"] for entry in rec.lockstep)
            results.append(build["result"])
        for record in rec.seams:
            points = {int(p) for p in record["pattern_index"] if p >= 0}
            if points <= routed:
                assert any(record["fm_state"] is result for result in results)
            else:
                assert points <= set(FK_TWO_GRAIN_UNROUTED_GRAIN)
                assert record["fm_state"] is None

    def test_a_forced_slot_whose_estimate_fails_is_inactive(self, monkeypatch):
        fk_assert_mixed_off_premise()
        off = fk_mixed_session_off()
        index = FK_MIXED_MISLABELLED
        pairs, rec = fk_spies(planted_angles=fk_planted_angle_failure(index))
        fk_install(monkeypatch, pairs)
        properties, _ = fk_mixed_session(monkeypatch, fourier_mellin="auto")
        assert len(rec.run_chunks_gpu) == 2
        retry = rec.run_chunks_gpu[1]
        retried = retry["fit_indices"].tolist()
        assert retried == list(FK_MIXED_RETRIED)
        slot = retried.index(index)
        other = retried.index(FK_MIXED_UNRELATED)
        retry_lockstep = [entry for entry in rec.lockstep if entry["pass"] == 2]
        assert retry_lockstep
        for entry in retry_lockstep:
            active = entry["real"] & np.isfinite(entry["h0"]).all(axis=1)
            assert not active[slot]
            assert active[other]
        packed = retry["packed"]
        assert packed.shape[1] == ROW_SLOTS_FM["width"]
        assert packed[slot, ROW_SLOTS_FM["converged"]] == 0.0
        assert packed[slot, ROW_SLOTS_FM["fm_applied"]] == 0.0
        fk_assert_point_bitwise(properties, off, index)
        assert properties["fourier_mellin_seed"][index] == SEED_TRANSLATION
        assert np.isnan(properties["fourier_mellin_angle"][index])

    def test_row_slots_width_and_seed_extras(self, monkeypatch):
        fk_assert_mixed_off_premise()
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        fk_mixed_session(monkeypatch, fourier_mellin="auto")
        first, retry = rec.run_chunks_gpu
        assert first["row_slots"] == ROW_SLOTS_FM
        assert first["packed"].shape == (len(FK_MIXED_FITTED), ROW_SLOTS_FM["width"])
        route = first["seed_extras"][ROUTE_KEY]
        assert route.dtype == np.int8
        assert route.tolist() == list(FK_MIXED_ROUTES)
        assert retry["row_slots"] == ROW_SLOTS_FM
        assert retry["seed_extras"][ROUTE_KEY].tolist() == [ROUTE_FORCED] * len(
            FK_MIXED_RETRIED
        )
        pairs_off, rec_off = fk_spies()
        fk_install(monkeypatch, pairs_off)
        fk_mixed_session(monkeypatch, fourier_mellin="off")
        (only,) = rec_off.run_chunks_gpu
        assert only["row_slots"] == ROW_SLOTS
        assert only["seed_extras"] is None
        assert only["packed"].shape[1] == ROW_SLOTS["width"]
        assert all(record["extras_keys"] == set() for record in rec_off.seams)
        assert all(record["outputs"] == {} for record in rec_off.seams)

    def test_the_retry_runs_at_the_first_pass_batch_size(self, monkeypatch):
        fk_assert_mixed_off_premise()
        pairs, rec = fk_spies()
        fk_install(monkeypatch, pairs)
        recorder = install_numpy_session(monkeypatch)
        inner = _gpu._make_session
        builds = {"n": 0}

        def flaky(namespace, batch_size, **kwargs):
            builds["n"] += 1
            if builds["n"] == 1:
                # the first pass's first build runs out of memory, so its
                # final B is half the requested one
                raise _batched.NumpyOutOfMemoryError("planted")
            return inner(namespace, batch_size, **kwargs)

        monkeypatch.setattr(_gpu, "_make_session", flaky)
        mixed = fk_mixed_map()
        run_engine(
            np.asarray(mixed["patterns"]),
            FK_MIXED_NAVIGATION_SHAPE,
            mixed["detector"],
            **fk_mixed_kwargs(
                backend="gpu",
                device_precision="float64",
                chunksize=8,
                fourier_mellin="auto",
            ),
        )
        assert len(rec.run_chunks_gpu) == 2
        first, retry = rec.run_chunks_gpu
        assert first["chunksize"] == 8
        assert recorder.built[0] == 4
        assert retry["chunksize"] == 4
        assert recorder.built[-1] == 4

    def test_the_vram_functions_take_the_fourier_mellin_keyword(self):
        precisions = (
            ("mixed", "complex128"),
            ("float64", "complex128"),
            ("mixed", "complex64"),
        )
        for n_pixels in (SHAPE_480[0] * SHAPE_480[1], SHAPE_RECT[0] * SHAPE_RECT[1]):
            for device_precision, seed_precision in precisions:
                args = (n_pixels, device_precision, seed_precision)
                base = _gpu._vram_model_terms(*args)
                assert _gpu._vram_model_terms(*args, fourier_mellin=False) == base
                fm = _gpu._vram_model_terms(*args, fourier_mellin=True)
                assert all(a >= b for a, b in zip(fm, base))
                assert fm[1] > base[1]
                assert fm[2] > base[2]
                for batch_size in _gpu._BATCH_SIZES:
                    assert _gpu._vram_model_bytes(
                        batch_size, *args, fourier_mellin=False
                    ) == _gpu._vram_model_bytes(batch_size, *args)
                for free in (8 * 2**30, 2 * 2**30, 64 * 2**20):
                    assert _gpu._default_batch_size(
                        free, *args, fourier_mellin=False
                    ) == _gpu._default_batch_size(free, *args)

    def test_the_batch_choice_halves_only_when_the_fm_terms_do_not_fit(self):
        args = (SHAPE_480[0] * SHAPE_480[1], "mixed", "complex128")
        fraction = _gpu._FREE_VRAM_FRACTION
        # headroom: the FM terms fit beside the "off" model, B unchanged
        free = 8 * 2**30
        b_off = _gpu._default_batch_size(free, *args)
        assert (
            _gpu._vram_model_bytes(b_off, *args, fourier_mellin=True) <= fraction * free
        )
        assert _gpu._default_batch_size(free, *args, fourier_mellin=True) == b_off
        # no headroom: the "off" model at B = 8 fills the budget exactly
        batch_size = 8
        free = int(np.ceil(_gpu._vram_model_bytes(batch_size, *args) / fraction))
        assert _gpu._default_batch_size(free, *args) == batch_size
        budget = fraction * free
        assert _gpu._vram_model_bytes(batch_size, *args, fourier_mellin=True) > budget
        chosen = _gpu._default_batch_size(free, *args, fourier_mellin=True)
        halvings = [batch_size // 2**k for k in range(1, 4) if batch_size // 2**k >= 1]
        fitting = [
            b
            for b in halvings
            if _gpu._vram_model_bytes(b, *args, fourier_mellin=True) <= budget
        ]
        assert chosen == (fitting[0] if fitting else 1)

    def test_the_runner_asks_the_fm_model_only_on_routed_runs(self, monkeypatch):
        fk_assert_mixed_off_premise()
        # a fake card on which the "off" model picks B = 8 (cheap: the
        # 8 GiB default picks 64 and iterates 32 slots per sub-batch)
        n_pixels = SHAPE_480[0] * SHAPE_480[1]
        model = _gpu._vram_model_bytes(8, n_pixels, "float64", "complex128")
        free = int(np.ceil(model / _gpu._FREE_VRAM_FRACTION))
        assert _gpu._default_batch_size(free, n_pixels, "float64", "complex128") == 8
        mixed = fk_mixed_map()
        for mode, expected in (("auto", True), ("off", False)):
            pairs, rec = fk_spies()
            fk_install(monkeypatch, pairs)
            install_numpy_session(monkeypatch, free_bytes=free)
            run_engine(
                np.asarray(mixed["patterns"]),
                FK_MIXED_NAVIGATION_SHAPE,
                mixed["detector"],
                **fk_mixed_kwargs(
                    backend="gpu",
                    device_precision="float64",
                    chunksize=None,
                    fourier_mellin=mode,
                ),
            )
            assert rec.default_batch, mode
            for call in rec.default_batch:
                assert bool(call["fourier_mellin"]) is expected, mode

    def test_the_numpy_session_against_the_cpu_route(self, monkeypatch):
        fk_assert_mixed_off_premise()
        cpu = fk_mixed_cpu("auto")
        twin, _ = fk_mixed_session(monkeypatch, fourier_mellin="auto")
        assert fk_bitwise(twin["fourier_mellin_seed"], cpu["fourier_mellin_seed"])
        twin_angle = np.asarray(twin["fourier_mellin_angle"])
        cpu_angle = np.asarray(cpu["fourier_mellin_angle"])
        assert np.array_equal(np.isnan(twin_angle), np.isnan(cpu_angle))
        finite = np.isfinite(cpu_angle)
        assert finite.any()
        assert_within(
            float(np.max(np.abs(twin_angle[finite] - cpu_angle[finite]))),
            FM_NUMPY_ANGLE_TOL_DEG,
            "FM_NUMPY_ANGLE_TOL_DEG",
        )
        corners = fk_mixed_map()["corners"]
        both = np.asarray(twin["converged"]) & np.asarray(cpu["converged"])
        assert both[[FK_MIXED_ROUTED, FK_MIXED_MISLABELLED]].all()
        worst = max(
            recovery_error(twin["homography"][i], cpu["homography"][i], corners)
            for i in np.flatnonzero(both)
        )
        assert_within(worst, FM_NUMPY_ROW_TOL_PX, "FM_NUMPY_ROW_TOL_PX")

    def test_complex64_keeps_the_fm_arithmetic_in_double(self, monkeypatch):
        records = {}

        def wrap(name, describe):
            original = getattr(_fourier_mellin, name)
            signature = inspect.signature(original)

            def spy(*args, **kwargs):
                out = original(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                values = tuple(bound.arguments.values())
                records.setdefault(name, []).append(describe(values, out))
                return out

            monkeypatch.setattr(_fourier_mellin, name, spy)

        def dtype(value):
            return np.dtype(np.asarray(value).dtype)

        wrap("fourier_mellin_angles", lambda a, o: (dtype(a[1]), dtype(o[0])))
        wrap("fourier_mellin_peak", lambda a, o: (dtype(a[1]), dtype(o[0])))
        wrap("fourier_mellin_hann_stencil", lambda a, o: (dtype(a[1]), dtype(o)))
        wrap("fourier_mellin_profiles", lambda a, o: (dtype(a[1]), dtype(o)))
        wrap("fourier_mellin_derotate", lambda a, o: (dtype(a[3]), dtype(o[0])))
        wrap("fourier_mellin_translate", lambda a, o: (dtype(a[1]), dtype(o)))
        wrap("fourier_mellin_partial_row", lambda a, o: (dtype(a[1]), dtype(o)))
        wrap("fourier_mellin_criteria", lambda a, o: (dtype(a[3]), dtype(o)))
        state, target = fk_planted_pair()
        seam = fk_seam(state, [target], [ROUTE_ACCEPT], precision="complex64")
        assert seam.spectra.dtype == np.complex64
        complex128, float64 = np.dtype(np.complex128), np.dtype(np.float64)
        # the read: complex64 spectra in, float64 angles out
        assert records["fourier_mellin_angles"]
        for spectra, theta in records["fourier_mellin_angles"]:
            assert (spectra, theta) == (np.dtype(np.complex64), float64)
        assert records["fourier_mellin_peak"]
        for correlation, theta in records["fourier_mellin_peak"]:
            assert (correlation, theta) == (float64, float64)
        for _, windowed in records.get("fourier_mellin_hann_stencil", []):
            assert windowed == complex128
        for _, profiles in records.get("fourier_mellin_profiles", []):
            assert profiles == float64
        for theta, crops in records["fourier_mellin_derotate"]:
            assert (theta, crops) == (float64, float64)
        for crops, rows in records["fourier_mellin_translate"]:
            assert (crops, rows) == (float64, float64)
        for theta, rows in records["fourier_mellin_partial_row"]:
            assert (theta, rows) == (float64, float64)
        for rows, values in records["fourier_mellin_criteria"]:
            assert (rows, values) == (float64, float64)
        assert seam.rows.dtype == float64
        assert seam.outputs[ANGLE_KEY].dtype == float64

    def test_import_hygiene_of_the_fourier_mellin_module(self):
        tree = ast.parse(fk_hrebsd_source("_fourier_mellin"))
        for node, name in fk_module_scope_imports(tree):
            parts = name.lstrip(".").split(".")
            assert parts[0] != "cupy", (node.lineno, name)
            assert "_engine" not in parts, (node.lineno, name)
            assert "_batched" not in parts, (node.lineno, name)
            assert parts[0] not in ("typing", "__future__"), (node.lineno, name)
        assert not re.search(
            r"^(import cupy|from cupy)",
            fk_hrebsd_source("_fourier_mellin"),
            re.MULTILINE,
        )
        # the call-time seams of D22.6: never a ``from ... import name``
        # binding of a Fourier-Mellin function, anywhere in the callers
        for module in ("_batched", "_engine", "_gpu"):
            for node in ast.walk(ast.parse(fk_hrebsd_source(module))):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        assert not alias.name.startswith(
                            ("fourier_mellin_", "build_fourier_mellin", "twist_about")
                        ), (module, alias.name)
        assert _engine._fourier_mellin is _fourier_mellin

    def test_the_fourier_mellin_path_imports_no_cupy(self):
        script = (
            "import sys\n"
            "from kikuchipy.indexing._hrebsd import _fourier_mellin\n"
            "indices, weights = _fourier_mellin.fourier_mellin_lut(432, 432)\n"
            "assert indices.shape == (360, 165, 4)\n"
            "assert 'cupy' not in sys.modules, 'the FM path imported cupy'\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr


# ===== GATED (l) (inserted) =====


# ===================== Locally gated GPU suite (l) ================== #
#
# V10(l) (D22.12 to D22.14, D22.18): run ONLY at ``-n 0`` through the
# PINNED overlay of D21.15 with ``KIKUCHIPY_EXPECT_GPU=1`` (the V10 gate
# commands); every test takes the ``cupy_gpu`` fixture, so the whole
# block skips at stage (a) without cupy.  The CPU route of D22.10 is
# the oracle for every device output (D22.14): at the seam it is the
# numpy seam at P = 1 with the reference's numpy ``SeedState`` and its
# ``FourierMellinState`` (:func:`gated_fm_cpu_route`); end to end it is
# ``run_hrebsd_dic(backend="cpu")``.  Written failing at the Stage F
# failing-tests gate (2026-10-08): every arm that reaches the FM path
# fails with ``NotImplementedError("Stage F: not implemented yet")``
# from the skeleton, and every device band is a ``FIXME-pin``
# placeholder measured at the implementation gate on machine A.  Two
# arms pass already and must stay green: the ``"off"`` drift tripwire
# and the ``"off"`` device default path.
#
# The constants and helpers of this block that come from
# ``test_hrebsd_gpu.py`` are copies as of 2026-10-08, RENAMED with a
# ``GATED_FM_`` / ``_gated_fm_`` prefix so that the gated block's names
# never collide with the default suite's; each names its source.

# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (``FROZEN_DEVICE_PRECISIONS``, ``FROZEN_SEED_PRECISIONS``,
# ``FROZEN_BATCH_SIZES``, ``FROZEN_SUB_BATCH_SIZE``), the frozen values
# of D21.4, D21.5, D21.7.3 and D21.10.3
GATED_FM_DEVICE_PRECISIONS = ("mixed", "float64")
GATED_FM_SEED_PRECISIONS = ("complex128", "complex64")
GATED_FM_BATCH_SIZES = (64, 32, 16, 8, 4, 2, 1)
GATED_FM_SUB_BATCH_SIZE = 32

# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (``MACHINE_A``), named in every device-side pin comment
GATED_FM_MACHINE_A = (
    "machine A: i7-13700H, NVIDIA RTX 2000 Ada Generation Laptop GPU 8 GB, "
    "driver 595.71, CuPy 14.2.0 on the pinned overlay of D21.15, Windows 11"
)

# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (``GPU_PARITY_H_TOL_MIXED``, ``GPU_PARITY_H_TOL_F64``; Stage E
# implementation gate, validation.md V9 ledger 104): the D21.8(c)
# ``h`` bands per device precision, px (V2 corner metric), which V10(l)
# applies UNCHANGED to the points both backends converge from FM rows
GATED_FM_H_TOL = {"mixed": 2.5e-6, "float64": 7e-12}

# Copied 2026-10-08 from tests/test_indexing/test_hrebsd_gpu.py
# (``GPU_DRIFT_RECOVERY_PX_MIXED``, ``GPU_DRIFT_RECOVERY_PX_F64``,
# ``GPU_DRIFT_TRIPWIRE_PX``; Stage E implementation gate, ledger 104):
# the device half of the D21.8(h) drift tripwire on F1, which V10(l)
# re-asserts under ``fourier_mellin="off"``
GATED_FM_DRIFT_RECOVERY_PX = {
    "mixed": np.array(
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
    ),
    "float64": np.array(
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
    ),
}
GATED_FM_DRIFT_TRIPWIRE_PX = 1e-9

# --- (l) device-side pins of THIS block, MTP (FIXME-pin placeholders,
# measured at the Stage F implementation gate on machine A through the
# pinned overlay, pinned at about 2x with the recipe beside them).
# The angle parity, acceptance and retry budgets and the VRAM bounds are
# the scaffold's ``FM_ANGLE_PARITY_DEG``, ``FM_ACCEPT_FLIP_COUNT``,
# ``FM_RETRY_FLIP_COUNT``, ``FM_VRAM_P_BOUNDS`` and ``FM_VRAM_R_BOUNDS``.
# (D21.8(e)) iteration-count differences and ``converged`` flips on the
# points both backends seed from FM rows (Stage E measured 0 and 0 on
# translation seeds; expected 0 here too, the FM seed rows differ by FFT
# rounding and, under "mixed", the f32 de-rotation gather)
GATED_FM_ITERATION_DIFF_COUNT = None  # FIXME-pin
GATED_FM_CONVERGED_FLIP_COUNT = None  # FIXME-pin
# (E4) B invariance with FM on, max absolute homography parameter
# difference against B = 8 (Stage E measured 0.0 with FM off; the
# masked full-P branch keeps P = 32 for B >= 32 but P = 8 at B = 8)
GATED_FM_BATCH_INVARIANCE_TOL = None  # FIXME-pin

# The seam fixtures of the angle parity (V10(l): "theta_hat against the
# CPU route on G1, G3 and G4"): G1's nine twists plus its two combined
# rotations; G3's six twists at 512x622; G4's nine cases per noise and
# filter setting
GATED_FM_PARITY_SETS = (
    "G1",
    "G3",
    "G4-none-open",
    "G4-none-default",
    "G4-poisson50-open",
    "G4-poisson50-default",
)
# The end-to-end FM-seeded fit parity map: the starred capture arms of
# V10(e) (twists and rotation vector), each a target of ONE map against
# the G1 reference at the capture budget
GATED_FM_CAPTURE_SPECS = tuple(
    twist_spec(t) for t in G1_CAPTURE_TWISTS_STARRED
) + tuple(rotvec_spec(w) for w in G1_CAPTURE_ROTVECS_STARRED)
# The routing-invariance arm (FM29): P slots, the routed slot alone at
# these positions against all P routed
GATED_FM_INVARIANCE_P = 32
GATED_FM_INVARIANCE_SLOTS = (0, 13, 31)
# The VRAM per-slot calibration at the two sub-batch sizes (P = B < 32)
GATED_FM_VRAM_SLOTS = (8, 16)
# The whole-model bound runs (the G3 map at 512x622)
GATED_FM_VRAM_MAP_SIZE = 16
GATED_FM_VRAM_MAX_ITERATIONS = 10
# The throughput record's modes
GATED_FM_THROUGHPUT_MODES = ("off", "auto", "always")


# ----------------------- Gated-block helpers ------------------------ #


def _gated_fm_kwargs(device_precision="mixed", seed_precision="complex128") -> dict:
    """Return the engine keywords of a device run (copy 2026-10-08 of
    ``_gpu_kwargs`` from tests/test_indexing/test_hrebsd_gpu.py)."""
    return {
        "backend": "gpu",
        "device_precision": device_precision,
        "seed_precision": seed_precision,
    }


def _gated_fm_assert_in_bounds(measured, bounds, name: str) -> None:
    """Assert ``low <= measured <= high`` for a pinned ``(low, high)``,
    failing loudly while *bounds* is an unfilled placeholder (copy
    2026-10-08 of ``_assert_in_bounds`` from
    tests/test_indexing/test_hrebsd_gpu.py; DRIFT 2026-10-08: the
    message names V10(l) instead of V9(o))."""
    if bounds is None:
        raise AssertionError(
            f"{name} is an unfilled FIXME-pin placeholder (validation V10(l)); "
            f"measured {measured!r} bytes. Pin (low, high) with the recipe and "
            "the machine ID"
        )
    low, high = bounds
    assert low <= measured <= high, f"{name}: {measured} not in [{low}, {high}]"


def _gated_fm_h_band(h_a, h_b, corners, where) -> float:
    """Return the largest V2 corner metric between two homography
    arrays over the flat indices where *where* holds, px (0 if none);
    copy 2026-10-08 of ``h_band`` from
    tests/test_indexing/test_hrebsd_gpu.py."""
    h_a = np.asarray(h_a)
    h_b = np.asarray(h_b)
    return max(
        (recovery_error(h_a[i], h_b[i], corners) for i in np.flatnonzero(where)),
        default=0.0,
    )


class _GatedFmPoolHighWater:
    """Track the high-water mark of cupy's default memory pool through
    a recording allocator, relative to the pool's use on entry (the
    pool and the FFT plan cache are emptied first); copy 2026-10-08 of
    ``_PoolHighWater`` from tests/test_indexing/test_hrebsd_gpu.py."""

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


def _gated_fm_spy_sessions(monkeypatch) -> list:
    """Wrap the module-global session factory and return the list of
    every build's batch size, in order (the recording half of
    ``_spy_sessions`` of tests/test_indexing/test_hrebsd_gpu.py, copy
    2026-10-08)."""
    built = []
    real = _gpu._make_session

    def make_session(namespace, batch_size, **kwargs):
        built.append(int(batch_size))
        return real(namespace, batch_size, **kwargs)

    monkeypatch.setattr(_gpu, "_make_session", make_session)
    return built


def _gated_fm_spy_runner(monkeypatch, module, name) -> list:
    """Wrap ``module.name`` (``_engine._run_chunks`` or
    ``_gpu._run_chunks_gpu``, both reached through their module objects
    at call time, D22.8, D22.10) and return the list of its calls, each
    a ``types.SimpleNamespace`` with ``fit_indices`` (a host copy),
    ``chunksize`` (the fifth positional or the keyword), ``kwargs`` and
    ``shape`` (the returned packed rows' shape)."""
    calls = []
    real = getattr(module, name)

    def runner(*args, **kwargs):
        record = types.SimpleNamespace(
            fit_indices=np.array(args[1], dtype=np.int64, copy=True),
            chunksize=args[4] if len(args) > 4 else kwargs.get("chunksize"),
            kwargs=dict(kwargs),
            shape=None,
        )
        calls.append(record)
        packed = real(*args, **kwargs)
        record.shape = tuple(np.shape(packed))
        return packed

    monkeypatch.setattr(module, name, runner)
    return calls


def _gated_fm_retry_subset(calls) -> set:
    """Return the flat indices of the retry pass (the SECOND runner call
    of a run, D22.8), empty when the run made no second call."""
    assert len(calls) <= 2, f"{len(calls)} runner calls: one retry pass at most"
    return set() if len(calls) < 2 else {int(i) for i in calls[1].fit_indices}


def _gated_fm_context(cupy, device_precision="float64") -> "_batched.SeedContext":
    """Return the cupy ``SeedContext`` at *device_precision* (copy
    2026-10-08 of ``_device_context`` from
    tests/test_indexing/test_hrebsd_gpu.py)."""
    kernels = _batched.make_kernel_namespace("cupy", device_precision)
    return _batched.SeedContext(cupy, cupy.fft, kernels)


def _gated_fm_seed_state(ctx, state, seed_precision="complex128") -> tuple:
    """Return ``(seed_state, resident, fm_state)`` of one reference in
    the namespace of *ctx*, built as the session builds them (D22.11
    (2)): the Stage E resident and seed state, then the FM state from
    that very resident, attached to the seed state."""
    resident = _batched.build_resident(ctx, state)
    seed_state = _batched.build_seed_state(ctx, state, precision=seed_precision)
    fm_state = _fourier_mellin.build_fourier_mellin_state(ctx, state, resident)
    assert fm_state is not None, "a G-fixture reference must build an FM state"
    seed_state.fourier_mellin = fm_state
    return seed_state, resident, fm_state


def _gated_fm_route(route, n) -> np.ndarray:
    """Return a fresh ``(n,)`` int8 host route array of *route* (a code
    broadcast to every slot, or a sequence)."""
    return np.ascontiguousarray(np.broadcast_to(np.asarray(route, np.int8), (n,)))


def _gated_fm_device_seed(
    cupy,
    state,
    targets,
    route,
    *,
    device_precision="float64",
    seed_precision="complex128",
    pattern_index=None,
    seed=None,
    prepared=None,
) -> types.SimpleNamespace:
    """Run the DEVICE seam on *targets* in ONE sub-batch of
    ``len(targets)`` slots with the route flags *route* (``None``: no
    route key, the Stage E call) and return a namespace with ``ctx``,
    ``seed_state``, ``resident``, ``fm_state``, ``batch``, ``extras``
    (the dictionary passed in), ``spectra``, ``spectra_before`` (a
    device copy taken before ``seed_homographies``) and ``rows``, all
    still on the device.  *seed* reuses a previous result's ``ctx``,
    ``seed_state``, ``resident`` and ``fm_state``; *prepared* is a
    precomputed ``(preprocessed, coefficients)`` of *targets*."""
    if seed is None:
        ctx = _gated_fm_context(cupy, device_precision)
        seed_state, resident, fm_state = _gated_fm_seed_state(
            ctx, state, seed_precision
        )
    else:
        ctx, seed_state = seed.ctx, seed.seed_state
        resident, fm_state = seed.resident, seed.fm_state
    if prepared is None:
        prepared = preprocessed_targets(state, targets)
    preprocessed, coefficients = prepared
    n = len(preprocessed)
    if pattern_index is None:
        pattern_index = np.arange(n, dtype=np.int64)
    extras = {} if route is None else {ROUTE_KEY: _gated_fm_route(route, n)}
    batch = _batched.SeedBatch(
        cupy.asarray(preprocessed),
        cupy.asarray(coefficients),
        np.asarray(pattern_index, dtype=np.int64),
        extras=extras,
    )
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    spectra_before = spectra.copy()
    rows = _batched.seed_homographies(ctx, batch, spectra, seed_state)
    return types.SimpleNamespace(
        ctx=ctx,
        seed_state=seed_state,
        resident=resident,
        fm_state=fm_state,
        batch=batch,
        extras=extras,
        spectra=spectra,
        spectra_before=spectra_before,
        rows=rows,
    )


def _gated_fm_host_outputs(cupy, result) -> tuple:
    """Return ``(rows, angle, applied)`` of a device seam result as host
    arrays; a run without outputs reads angle NaN and applied False
    (the D22.6 runner rule)."""
    rows = cupy.asnumpy(result.rows)
    n = rows.shape[0]
    outputs = result.batch.outputs
    angle = (
        cupy.asnumpy(outputs[ANGLE_KEY]) if ANGLE_KEY in outputs else np.full(n, np.nan)
    )
    applied = (
        cupy.asnumpy(outputs[APPLIED_KEY])
        if APPLIED_KEY in outputs
        else np.zeros(n, dtype=bool)
    )
    return rows, angle, applied


@functools.lru_cache(maxsize=None)
def gated_fm_parity_set(name: str) -> tuple:
    """Return ``(state, targets)`` of one angle-parity seam set of
    :data:`GATED_FM_PARITY_SETS`: the reference's ``ReferenceState`` as
    the engine builds it (``"open"`` is ``filter_cutoffs=(None,
    None)``, ``"default"`` the engine default ``(0.05, None)``) and the
    ``(N, nrows, ncols)`` targets.  Read-only."""
    if name == "G1":
        pairs = [g1_pair(twist_spec(t)) for t in G1_TWISTS_DEG]
        pairs += [g1_pair(spec) for spec in G1_COMBINED]
        state = g1_state(pairs[0])
    elif name == "G3":
        pairs = [g3_pair(t) for t in G3_TWISTS]
        state = g1_state(pairs[0])
    else:
        _, noise, setting = name.split("-")
        cutoffs = (None, None) if setting == "open" else (0.05, None)
        pairs = [g4_pair(case, noise) for case in range(len(G4_CASES))]
        state = g1_state(pairs[0], filter_cutoffs=cutoffs)
    targets = _read_only(np.stack([np.asarray(pair["target"]) for pair in pairs]))
    return state, targets


@functools.lru_cache(maxsize=None)
def gated_fm_cpu_route(name: str, route: int) -> tuple:
    """Return ``(rows, angle, applied)`` of the CPU ROUTE oracle of
    D22.10 on the parity set *name*: per target, the numpy seam at
    P = 1 (``SeedContext(np, np.fft, make_kernel_namespace("numpy",
    "float64"))``, the reference's numpy complex128 ``SeedState`` with
    its ``FourierMellinState`` built ONCE from the numpy subregion
    resident, a one-slot ``SeedBatch`` carrying the route key),
    ``_batched.seed_spectra`` and ``_batched.seed_homographies``.
    Host arrays, read-only."""
    state, targets = gated_fm_parity_set(name)
    ctx = numpy_seed_context()
    seed_state, _, _ = _gated_fm_seed_state(ctx, state)
    preprocessed, coefficients = preprocessed_targets(state, targets)
    n = len(targets)
    rows = np.full((n, N_HOMOGRAPHY_PARAMETERS), np.nan)
    angle = np.full(n, np.nan)
    applied = np.zeros(n, dtype=bool)
    for i in range(n):
        batch = _batched.SeedBatch(
            preprocessed[i : i + 1],
            coefficients[i : i + 1],
            np.array([i], dtype=np.int64),
            extras={ROUTE_KEY: _gated_fm_route(route, 1)},
        )
        spectra = _batched.seed_spectra(ctx, batch, seed_state)
        rows[i] = np.asarray(
            _batched.seed_homographies(ctx, batch, spectra, seed_state)
        )[0]
        angle[i] = float(np.asarray(batch.outputs[ANGLE_KEY])[0])
        applied[i] = bool(np.asarray(batch.outputs[APPLIED_KEY])[0])
    return _read_only(rows), _read_only(angle), _read_only(applied)


@functools.lru_cache(maxsize=1)
def gated_fm_capture_map() -> dict:
    """Return the FM-seeded fit parity map: the G1 reference at (0, 0)
    then one target per :data:`GATED_FM_CAPTURE_SPECS`, one row, the
    reference's PC tiled.  Keys ``patterns``, ``navigation_shape``,
    ``detector``, ``exact`` ``(n, 8)`` (zeros for the reference),
    ``corners``."""
    pairs = [g1_pair(spec) for spec in GATED_FM_CAPTURE_SPECS]
    navigation_shape = (1, 1 + len(pairs))
    detector = make_detector(navigation_shape=navigation_shape)
    exact = np.zeros((1 + len(pairs), N_HOMOGRAPHY_PARAMETERS))
    exact[1:] = np.stack([pair["exact"] for pair in pairs])
    return {
        "patterns": _read_only(
            np.stack([g1_reference()] + [pair["target"] for pair in pairs])
        ),
        "navigation_shape": navigation_shape,
        "detector": detector,
        "exact": _read_only(exact),
        "corners": _read_only(subregion_corners(SHAPE_480, pairs[0]["pc_px"])),
    }


def gated_fm_capture_run(**kwargs) -> dict:
    """Return ``run_hrebsd_dic`` on :func:`gated_fm_capture_map` at
    reference (0, 0), ``fourier_mellin="always"`` (no crystal map
    needed) and the capture budget (each overridable)."""
    fixture = gated_fm_capture_map()
    kwargs.setdefault("reference", (0, 0))
    kwargs.setdefault("fourier_mellin", "always")
    kwargs.setdefault("max_iterations", G1_CAPTURE_MAX_ITERATIONS)
    return run_engine(
        np.asarray(fixture["patterns"]),
        fixture["navigation_shape"],
        fixture["detector"],
        **kwargs,
    )


@functools.lru_cache(maxsize=1)
def gated_fm_capture_cpu() -> dict:
    """Return the cached CPU run of :func:`gated_fm_capture_run`, the
    end-to-end oracle (treat it read-only)."""
    return gated_fm_capture_run()


@functools.lru_cache(maxsize=1)
def gated_fm_g8_cpu() -> tuple:
    """Return ``(properties, retry_subset)`` of the cached CPU G8 run
    under ``"auto"``, the retry subset read from the second
    ``_engine._run_chunks`` call (a local spy, no monkeypatch fixture:
    the module attribute is restored in a ``finally``)."""
    calls = []
    real = _engine._run_chunks

    def runner(*args, **kwargs):
        calls.append(np.array(args[1], dtype=np.int64, copy=True))
        return real(*args, **kwargs)

    _engine._run_chunks = runner
    try:
        properties = g8_run(fourier_mellin="auto")
    finally:
        _engine._run_chunks = real
    assert len(calls) <= 2
    subset = set() if len(calls) < 2 else {int(i) for i in calls[1]}
    return properties, frozenset(subset)


def gated_fm_ramp_run(**kwargs) -> dict:
    """Return :func:`ramp_run` with the matching crystal map
    :func:`ramp_xmap` (the V10(f) ``"auto"`` map: columns 2 to 6 of the
    ramp twist at 1.6 deg or more and are routed pre-fit)."""
    kwargs.setdefault("xmap", ramp_xmap())
    return ramp_run(**kwargs)


@functools.lru_cache(maxsize=1)
def gated_fm_g3_map() -> dict:
    """Return the 512x622 map of the VRAM whole-model arm: the G3
    reference then :data:`GATED_FM_VRAM_MAP_SIZE` - 1 targets cycling
    through the G3 twists, one row, the PC tiled."""
    pairs = [g3_pair(t) for t in G3_TWISTS]
    n = GATED_FM_VRAM_MAP_SIZE
    targets = [pairs[i % len(pairs)]["target"] for i in range(n - 1)]
    navigation_shape = (1, n)
    return {
        "patterns": _read_only(np.stack([pairs[0]["reference"]] + targets)),
        "navigation_shape": navigation_shape,
        "detector": make_detector(shape=SHAPE_RECT, navigation_shape=navigation_shape),
    }


@functools.lru_cache(maxsize=None)
def gated_fm_slots(name: str, p: int) -> tuple:
    """Return ``(state, targets, prepared)`` of *p* device slots: the
    parity set *name*'s targets cycled to *p* slots and their
    ``(preprocessed, coefficients)``, prepared once (read-only)."""
    state, targets = gated_fm_parity_set(name)
    slots = np.asarray(targets)[np.arange(p) % len(targets)]
    preprocessed, coefficients = preprocessed_targets(state, slots)
    prepared = (_read_only(preprocessed), _read_only(coefficients))
    return state, _read_only(slots), prepared


class TestGatedFourierMellinContract:
    """V10(l) gated (D22.6, D22.9, D22.11 (2), D22.12): the device
    output contract of the FM seam extension -- ``SeedBatch.outputs``
    holding ``"fourier_mellin_angle"`` (``cupy.ndarray``, float64,
    ``(P,)``) and ``"fourier_mellin_applied"`` (``cupy.ndarray``, bool,
    ``(P,)``), NaN angles on route-0 and padded slots, the Stage E rows
    bitwise wherever the FM row is not applied, ``extras`` the same host
    dictionary unchanged and ``target_spectra`` untouched, the frozen
    function outputs in the device namespace (float64 at BOTH seed
    precisions, D22.12) and the FM state holding the very resident it
    was given; through the runner, the lazily built FM state holding
    the lockstep's resident, the 14-wide packed rows with the FM
    ``row_slots`` and the two props' host dtypes; and the ``"off"``
    device default path: the Stage E D21.8(h) drift tripwire unchanged
    and ``"off"`` bitwise the call without the keyword, with no FM work.

    Mutants (gated twins; the designed killers are the default suite's
    numpy arms): FM8 (the search window dropped: the planted
    correlations of V10(b) through ``fourier_mellin_peak`` in the device
    namespace, with the tie and edge rows), FM37 [D/G] (the float64
    angle and profile under
    ``seed_precision="complex64"``, beside the parity band of
    :class:`TestGatedFourierMellinParity`), FM45 (``extras`` unchanged,
    outputs only in ``SeedBatch.outputs``), FM52 (NaN angles on route-0
    and padded slots), FM6 (spectra bitwise before and after), FM15
    (the unrouted slot's row bitwise Stage E), FM16 (no FM work on
    ``"off"`` runs, 12-wide rows, Stage E ``row_slots``)."""

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_seam_output_contract(self, cupy_gpu, device_precision, seed_precision):
        cp = cupy_gpu
        state, targets = gated_fm_parity_set("G1")
        # four routed slots (accept, accept, forced, accept), one
        # route-0 slot and one PADDED slot (pattern index -1, route 0)
        slots = np.asarray(targets)[[0, 3, 5, 7, 4, 4]]
        route = np.array([1, 1, 2, 1, 0, 0], dtype=np.int8)
        pattern_index = np.array([0, 1, 2, 3, 4, -1], dtype=np.int64)
        result = _gated_fm_device_seed(
            cp,
            state,
            slots,
            route,
            device_precision=device_precision,
            seed_precision=seed_precision,
            pattern_index=pattern_index,
        )
        n = len(slots)
        # the rows: the D21.5 contract unchanged
        assert isinstance(result.rows, cp.ndarray)
        assert result.rows.dtype == np.float64
        assert result.rows.flags.c_contiguous
        assert tuple(result.rows.shape) == (n, N_HOMOGRAPHY_PARAMETERS)
        # the outputs: exactly the two keys, in the device namespace
        outputs = result.batch.outputs
        assert set(outputs) == {ANGLE_KEY, APPLIED_KEY}
        angle, applied = outputs[ANGLE_KEY], outputs[APPLIED_KEY]
        assert isinstance(angle, cp.ndarray)
        assert angle.dtype == np.float64
        assert tuple(angle.shape) == (n,)
        assert isinstance(applied, cp.ndarray)
        assert applied.dtype == np.bool_
        assert tuple(applied.shape) == (n,)
        # extras: the SAME host dictionary, exactly the route key, unchanged
        assert result.batch.extras is result.extras
        assert set(result.batch.extras) == {ROUTE_KEY}
        held = result.batch.extras[ROUTE_KEY]
        assert isinstance(held, np.ndarray)
        assert held.dtype == np.int8
        assert np.array_equal(held, route)
        # target_spectra untouched by the FM stage (D22.6)
        assert bool(cp.array_equal(result.spectra, result.spectra_before))
        # the outputs rule: angles on routed slots only
        rows, host_angle, host_applied = _gated_fm_host_outputs(cp, result)
        routed = route != ROUTE_NONE
        assert np.isnan(host_angle[~routed]).all()
        assert np.isfinite(host_angle[routed]).all()
        assert not host_applied[~routed].any()
        # the Stage E rows bitwise wherever the FM row is not applied
        stage_e = _gated_fm_device_seed(
            cp, state, slots, None, pattern_index=pattern_index, seed=result
        )
        assert stage_e.batch.outputs == {}
        expected = cp.asnumpy(stage_e.rows)
        kept = ~host_applied
        assert np.array_equal(rows[kept], expected[kept], equal_nan=True)

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_frozen_functions_and_state_on_the_device(
        self, cupy_gpu, device_precision, seed_precision
    ):
        cp = cupy_gpu
        state, targets = gated_fm_parity_set("G1")
        ctx = _gated_fm_context(cp, device_precision)
        seed_state, resident, fm_state = _gated_fm_seed_state(
            ctx, state, seed_precision
        )
        r0, r1, c0, c1 = state.bounds
        crop = (r1 - r0, c1 - c0)
        # the state: a REFERENCE to the given resident, the profile in
        # float64 on the device whatever the seed precision (D22.12),
        # the host-built table uploaded (int64 indices, float64 weights)
        assert isinstance(fm_state, _fourier_mellin.FourierMellinState)
        assert fm_state.resident is resident
        assert isinstance(fm_state.reference_profile, cp.ndarray)
        assert fm_state.reference_profile.dtype == np.float64
        assert tuple(fm_state.reference_profile.shape) == (FROZEN_N_THETA,)
        indices, weights = fm_state.lut
        assert isinstance(indices, cp.ndarray) and isinstance(weights, cp.ndarray)
        assert indices.dtype == np.int64 and weights.dtype == np.float64
        assert tuple(indices.shape) == LUT_SHAPES[crop]
        assert tuple(weights.shape) == LUT_SHAPES[crop]
        assert fm_state.n_theta == FROZEN_N_THETA
        assert fm_state.search_deg == FROZEN_SEARCH_DEG
        # the frozen functions, called directly on a device sub-batch
        preprocessed, coefficients = preprocessed_targets(state, targets)
        n = len(targets)
        batch = _batched.SeedBatch(
            cp.asarray(preprocessed),
            cp.asarray(coefficients),
            np.arange(n, dtype=np.int64),
            extras={ROUTE_KEY: _gated_fm_route(ROUTE_ACCEPT, n)},
        )
        spectra = _batched.seed_spectra(ctx, batch, seed_state)
        assert spectra.dtype == np.dtype(seed_precision)
        theta, peak = _fourier_mellin.fourier_mellin_angles(ctx, spectra, fm_state)
        for value in (theta, peak):
            assert isinstance(value, cp.ndarray)
            assert value.dtype == np.float64
            assert tuple(value.shape) == (n,)
        h_t = _batched.seed_homographies(
            ctx,
            _batched.SeedBatch(batch.targets, batch.coefficients, batch.pattern_index),
            spectra,
            seed_state,
        )
        rows, angle, applied = _fourier_mellin.fourier_mellin_rows(
            ctx, batch, spectra, seed_state, h_t, _gated_fm_route(ROUTE_ACCEPT, n)
        )
        assert isinstance(rows, cp.ndarray) and rows.dtype == np.float64
        assert tuple(rows.shape) == (n, N_HOMOGRAPHY_PARAMETERS)
        assert isinstance(angle, cp.ndarray) and angle.dtype == np.float64
        assert tuple(angle.shape) == (n,)
        assert isinstance(applied, cp.ndarray) and applied.dtype == np.bool_
        assert tuple(applied.shape) == (n,)
        # the angle the branch reports IS the angle function's
        assert np.array_equal(cp.asnumpy(angle), cp.asnumpy(theta), equal_nan=True)

    def test_the_peak_rule_on_the_device(self, cupy_gpu):
        # V10(b) gated (critic F1, 2026-10-08): the planted correlations
        # of TestFourierMellinAngle in the device namespace -- the
        # 40 / 5 deg window pair (FM8), the exact +6 / -6 lag tie (+k,
        # the lowest output index), the edge peak at lag 60 with its raw
        # outside neighbour, the flat row and the plateau entering the
        # window at lag -60.  A device argmax over a -inf-masked window
        # is where tie and edge behaviour can part from numpy
        cp = cupy_gpu
        n = FROZEN_N_THETA
        bump = TestFourierMellinAngle.bump
        correlation = np.zeros((6, n))
        bump(correlation[0], 80, 1.0)
        bump(correlation[0], 10, 0.5)
        bump(correlation[1], -80, 1.0)
        bump(correlation[1], -10, 0.5)
        bump(correlation[2], 6, 1.0)
        bump(correlation[2], -6, 1.0)
        correlation[3, 59], correlation[3, 60], correlation[3, 61] = 0.5, 1.0, 0.9
        correlation[4] = 0.25
        correlation[5, [299, 300, 301]] = 1.0
        delta = (0.5 - 0.9) / (2.0 * (0.5 - 2.0 + 0.9))
        edge = (60 + delta) * 180.0 / n
        # the premise: the local rule gives the V10(b) values
        local_theta, local_peak_value = local_peak(correlation)
        assert local_theta[[0, 1, 2, 4, 5]].tolist() == [5.0, -5.0, 3.0, 0.0, -30.0]
        assert abs(local_theta[3] - edge) <= 1e-12
        assert local_peak_value.tolist() == [0.5, 0.5, 1.0, 1.0, 0.25, 1.0]
        theta, peak = _fourier_mellin.fourier_mellin_peak(
            cp, cp.asarray(correlation), FROZEN_SEARCH_DEG
        )
        for value in (theta, peak):
            assert isinstance(value, cp.ndarray)
            assert value.dtype == np.float64
            assert tuple(value.shape) == (6,)
        host_theta, host_peak = cp.asnumpy(theta), cp.asnumpy(peak)
        assert host_theta[[0, 1, 2, 4, 5]].tolist() == [5.0, -5.0, 3.0, 0.0, -30.0]
        assert abs(host_theta[3] - edge) <= 1e-12
        assert host_peak.tolist() == [0.5, 0.5, 1.0, 1.0, 0.25, 1.0]
        # bitwise the numpy call on the same rows
        numpy_theta, numpy_peak = _fourier_mellin.fourier_mellin_peak(
            np, correlation, FROZEN_SEARCH_DEG
        )
        assert np.array_equal(host_theta, np.asarray(numpy_theta))
        assert np.array_equal(host_peak, np.asarray(numpy_peak))

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_runner_contract(self, cupy_gpu, monkeypatch, device_precision):
        cp = cupy_gpu
        residents, fm_builds, seams = [], [], []
        real_resident = _batched.build_resident
        real_fm_state = _fourier_mellin.build_fourier_mellin_state
        real_seam = _batched.seed_homographies

        def build_resident(ctx, state):
            resident = real_resident(ctx, state)
            residents.append(resident)
            return resident

        def build_fm_state(ctx, state, resident):
            fm_state = real_fm_state(ctx, state, resident)
            fm_builds.append((resident, fm_state))
            return fm_state

        def seed_homographies(ctx, batch, target_spectra, seed_state):
            extras_before = {k: np.array(v, copy=True) for k, v in batch.extras.items()}
            rows = real_seam(ctx, batch, target_spectra, seed_state)
            seams.append((batch, extras_before, rows))
            return rows

        monkeypatch.setattr(_batched, "build_resident", build_resident)
        monkeypatch.setattr(
            _fourier_mellin, "build_fourier_mellin_state", build_fm_state
        )
        monkeypatch.setattr(_batched, "seed_homographies", seed_homographies)
        runs = _gated_fm_spy_runner(monkeypatch, _gpu, "_run_chunks_gpu")
        properties = gated_fm_capture_run(
            chunksize=8, **_gated_fm_kwargs(device_precision)
        )
        n_points = int(np.prod(gated_fm_capture_map()["navigation_shape"]))
        # the props: host arrays of the D22.9 dtypes and shapes
        for name in FOURIER_MELLIN_PROP_NAMES:
            assert isinstance(properties[name], np.ndarray), name
            assert properties[name].dtype == FOURIER_MELLIN_PROP_DTYPES[name], name
            assert properties[name].shape == (n_points,), name
        # the first pass: 14-wide rows, the FM row slots, a route array
        assert runs and runs[0].kwargs["row_slots"] == ROW_SLOTS_FM
        assert runs[0].shape == (n_points, ROW_SLOTS_FM["width"])
        extras = runs[0].kwargs["seed_extras"]
        assert set(extras) == {ROUTE_KEY}
        assert extras[ROUTE_KEY].dtype == np.int8
        assert extras[ROUTE_KEY].shape == (n_points,)
        # one grain: the FM state built once, lazily, from the very
        # resident the lockstep uses (no second upload)
        assert len(fm_builds) == 1
        resident, fm_state = fm_builds[0]
        assert any(resident is built for built in residents)
        assert fm_state.resident is resident
        # every seam call with a routed slot: the device outputs and the
        # host extras unchanged
        routed_calls = [s for s in seams if s[0].extras.get(ROUTE_KEY) is not None]
        assert routed_calls
        for batch, extras_before, rows in routed_calls:
            p = int(rows.shape[0])
            assert set(batch.extras) == {ROUTE_KEY}
            assert isinstance(batch.extras[ROUTE_KEY], np.ndarray)
            assert np.array_equal(batch.extras[ROUTE_KEY], extras_before[ROUTE_KEY])
            assert batch.extras[ROUTE_KEY].shape == (p,)
            assert not batch.extras[ROUTE_KEY][batch.pattern_index < 0].any()
            if batch.extras[ROUTE_KEY].any():
                assert isinstance(batch.outputs[ANGLE_KEY], cp.ndarray)
                assert batch.outputs[ANGLE_KEY].dtype == np.float64
                assert tuple(batch.outputs[ANGLE_KEY].shape) == (p,)
                assert isinstance(batch.outputs[APPLIED_KEY], cp.ndarray)
                assert batch.outputs[APPLIED_KEY].dtype == np.bool_
                assert tuple(batch.outputs[APPLIED_KEY].shape) == (p,)

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_off_is_the_stage_e_device_path(
        self, cupy_gpu, monkeypatch, device_precision
    ):
        def refuse(*args, **kwargs):
            raise AssertionError("no FM state may be built on an 'off' run")

        f6 = f6_map()
        kwargs = dict(
            reference=(0, 0),
            max_iterations=f6["max_iterations"],
            chunksize=8,
            **_gated_fm_kwargs(device_precision),
        )
        patterns = np.asarray(f6["patterns"])
        plain = run_engine(patterns, f6["navigation_shape"], f6["detector"], **kwargs)
        monkeypatch.setattr(_fourier_mellin, "build_fourier_mellin_state", refuse)
        runs = _gated_fm_spy_runner(monkeypatch, _gpu, "_run_chunks_gpu")
        off = run_engine(
            patterns,
            f6["navigation_shape"],
            f6["detector"],
            fourier_mellin="off",
            **kwargs,
        )
        assert_properties_bitwise(plain, off)
        assert not set(FOURIER_MELLIN_PROP_NAMES) & set(off)
        assert len(runs) == 1
        assert runs[0].kwargs["row_slots"] == ROW_SLOTS
        assert runs[0].kwargs.get("seed_extras") is None
        assert runs[0].shape[1] == ROW_SLOTS["width"]

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_drift_tripwire_unchanged_under_off(
        self, cupy_gpu, device_precision, record_property
    ):
        # the D21.8(h) device half on F1, the Stage E literals, under
        # an EXPLICIT "off" (default B, complex128 seeds as recorded)
        fixture = f1_batch()
        properties = run_engine(
            np.asarray(fixture["patterns"]),
            fixture["navigation_shape"],
            fixture["map_detector"],
            reference=fixture["reference_index"],
            fourier_mellin="off",
            **_gated_fm_kwargs(device_precision),
        )
        homography = np.asarray(properties["homography"])[1:]
        errors = np.array(
            [
                recovery_error(h, h_true, fixture["corners"])
                for h, h_true in zip(homography, fixture["exact"])
            ]
        )
        record_property("recovery_px", [float(e) for e in errors])
        literals = GATED_FM_DRIFT_RECOVERY_PX[device_precision]
        assert literals.shape == errors.shape
        assert_within(
            float(np.abs(errors - literals).max()),
            GATED_FM_DRIFT_TRIPWIRE_PX,
            f"GATED_FM_DRIFT_TRIPWIRE_PX ({device_precision})",
        )


class TestGatedFourierMellinParity:
    """V10(l) gated (D22.14, D22.12; plan open question FQ14): the CPU
    route is the oracle.  (b) ``theta_hat`` of the device seam against
    the CPU route (the numpy seam at P = 1, complex128) on G1, G3 and
    G4 within ``FM_ANGLE_PARITY_DEG`` at BOTH seed precisions and both
    device precisions, route 1 (with the acceptance) and route 2
    (forced); (c) the acceptance decisions (``applied``) within the
    COUNT budget ``FM_ACCEPT_FLIP_COUNT``; end to end under
    ``"always"``, (d) ``h`` on the points both backends converge from
    FM rows within the D21.8(c) bands per device precision, the angle
    prop within the parity band and the seed codes within the flip
    budget, (e) the iteration and convergence COUNT budgets on the
    FM-seeded points; (f) on G8 under ``"auto"`` the retry routing
    (the second runner call's subset) within ``FM_RETRY_FLIP_COUNT``,
    the retry run at the first pass's B (D22.8).

    Mutants: FM37 [D/G] (the FM arithmetic left in complex64 under
    ``seed_precision="complex64"``: the parity band at complex64 against
    the complex128 CPU route is its designed device killer, the band
    pinned below the complex64-arithmetic error with the separation
    stated); FM48 (gated twin: the retry's ``chunksize`` is the first
    pass's B)."""

    @pytest.mark.parametrize("route", [ROUTE_ACCEPT, ROUTE_FORCED])
    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("name", GATED_FM_PARITY_SETS)
    def test_theta_hat_and_acceptance_parity(
        self, cupy_gpu, name, device_precision, seed_precision, route, record_property
    ):
        state, targets = gated_fm_parity_set(name)
        cpu_rows, cpu_angle, cpu_applied = gated_fm_cpu_route(name, int(route))
        result = _gated_fm_device_seed(
            cupy_gpu,
            state,
            targets,
            route,
            device_precision=device_precision,
            seed_precision=seed_precision,
        )
        rows, angle, applied = _gated_fm_host_outputs(cupy_gpu, result)
        # premise: the CPU route estimates every angle of these sets
        assert np.isfinite(cpu_angle).all()
        assert np.array_equal(np.isnan(angle), np.isnan(cpu_angle))
        difference = float(np.abs(angle - cpu_angle).max())
        flips = int((applied != cpu_applied).sum())
        both = applied & cpu_applied
        row_difference = float(np.abs(rows[both] - cpu_rows[both]).max(initial=0.0))
        record_property("theta_hat_max_difference_deg", difference)
        record_property("acceptance_flips", flips)
        record_property("applied_row_max_difference", row_difference)
        key = (name, device_precision, seed_precision, int(route))
        assert_within(difference, FM_ANGLE_PARITY_DEG, f"FM_ANGLE_PARITY_DEG {key}")
        if route == ROUTE_FORCED:
            # a forced row is applied wherever the estimate is valid
            assert np.array_equal(applied, cpu_applied)
        assert_within(flips, FM_ACCEPT_FLIP_COUNT, f"FM_ACCEPT_FLIP_COUNT {key}")

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_fm_seeded_fit_parity(
        self, cupy_gpu, device_precision, seed_precision, record_property
    ):
        fixture = gated_fm_capture_map()
        cpu = gated_fm_capture_cpu()
        gpu = gated_fm_capture_run(**_gated_fm_kwargs(device_precision, seed_precision))
        assert set(cpu) == set(gpu)
        for name in ("grain_id", "reference_index"):
            assert np.array_equal(np.asarray(cpu[name]), np.asarray(gpu[name])), name
        key = (device_precision, seed_precision)
        cpu_seed = np.asarray(cpu["fourier_mellin_seed"])
        gpu_seed = np.asarray(gpu["fourier_mellin_seed"])
        # (c) the acceptance decisions end to end, a COUNT budget
        flips = int((cpu_seed != gpu_seed).sum())
        record_property("seed_code_flips", flips)
        assert_within(flips, FM_ACCEPT_FLIP_COUNT, f"FM_ACCEPT_FLIP_COUNT e2e {key}")
        # (b) the angle prop
        cpu_angle = np.asarray(cpu["fourier_mellin_angle"])
        gpu_angle = np.asarray(gpu["fourier_mellin_angle"])
        assert np.array_equal(np.isnan(cpu_angle), np.isnan(gpu_angle))
        angle_difference = float(np.nanmax(np.abs(gpu_angle - cpu_angle), initial=0.0))
        record_property("angle_prop_max_difference_deg", angle_difference)
        assert_within(
            angle_difference, FM_ANGLE_PARITY_DEG, f"FM_ANGLE_PARITY_DEG e2e {key}"
        )
        # (d) h on the points BOTH backends converged from FM rows
        fm_seeded = (cpu_seed == SEED_FIRST_PASS) & (gpu_seed == SEED_FIRST_PASS)
        both = fm_seeded & np.asarray(cpu["converged"]) & np.asarray(gpu["converged"])
        # premise: the FM rows capture the starred arms on the CPU
        assert both.sum() >= len(GATED_FM_CAPTURE_SPECS) - 1
        band = _gated_fm_h_band(
            cpu["homography"], gpu["homography"], fixture["corners"], both
        )
        record_property("h_band_px", band)
        assert_within(band, GATED_FM_H_TOL[device_precision], f"h band {key}")
        # (e) the D21.8(e) COUNT budgets on the FM-seeded points
        iterations = int(
            (
                (np.asarray(cpu["num_iterations"]) != np.asarray(gpu["num_iterations"]))
                & fm_seeded
            ).sum()
        )
        converged = int(
            (
                (np.asarray(cpu["converged"]) != np.asarray(gpu["converged"]))
                & fm_seeded
            ).sum()
        )
        record_property("iteration_differences", iterations)
        record_property("converged_flips", converged)
        assert_count(
            iterations,
            GATED_FM_ITERATION_DIFF_COUNT,
            f"GATED_FM_ITERATION_DIFF_COUNT {key}",
        )
        assert_count(
            converged,
            GATED_FM_CONVERGED_FLIP_COUNT,
            f"GATED_FM_CONVERGED_FLIP_COUNT {key}",
        )

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_retry_routing_parity(
        self, cupy_gpu, monkeypatch, device_precision, record_property
    ):
        cpu, cpu_subset = gated_fm_g8_cpu()
        runs = _gated_fm_spy_runner(monkeypatch, _gpu, "_run_chunks_gpu")
        gpu = g8_run(fourier_mellin="auto", **_gated_fm_kwargs(device_precision))
        gpu_subset = _gated_fm_retry_subset(runs)
        # premise (V10(i), ledger 131): the CPU retries the mislabelled
        # twist, the unrelated pattern and the NaN-h constant pattern
        assert {G8_MISLABELLED, G8_UNRELATED} <= set(cpu_subset)
        assert G8_MASKED not in cpu_subset
        flips = len(set(cpu_subset) ^ gpu_subset)
        record_property("cpu_retry_subset", sorted(cpu_subset))
        record_property("gpu_retry_subset", sorted(gpu_subset))
        record_property("retry_flips", flips)
        assert_within(
            flips, FM_RETRY_FLIP_COUNT, f"FM_RETRY_FLIP_COUNT ({device_precision})"
        )
        # the retry runs at the FIRST pass's B, passed explicitly (D22.8)
        if len(runs) == 2:
            assert runs[1].chunksize == runs[0].chunksize
        # the pre-fit routes are host computed: no point routed on G8
        # under "auto" (every xmap twist below the gate), on both
        first_extras = runs[0].kwargs.get("seed_extras") or {}
        first_pass_routes = first_extras.get(ROUTE_KEY)
        assert first_pass_routes is None or not np.asarray(first_pass_routes).any()
        seed_flips = int(
            (
                np.asarray(cpu["fourier_mellin_seed"])
                != np.asarray(gpu["fourier_mellin_seed"])
            ).sum()
        )
        record_property("seed_code_flips", seed_flips)


class TestGatedFourierMellinDeterminism:
    """V10(l) gated (D22.13, D21.7.3): two runs bitwise at a fixed B
    with FM on, at both device and both seed precisions; ROUTING
    INVARIANCE, a routed slot's row, angle and decision bitwise the same
    whether 1 or all P slots of its sub-batch are routed (the masked
    full-P branch of D22.6); B invariance at B in {8, 32, 40, default}
    under ``"auto"`` (E4, bitwise among B >= 32, banded against B = 8);
    and the points ``"auto"`` neither routes nor retries keeping the
    ``"off"`` run's bits at the default B (D22.18).

    Mutants: FM29 [D/G] (the FM branch compacted to the routed slots:
    the routing-invariance arm is its designed DEVICE killer, cuFFT's
    batched plans being batch-count sensitive), FM46 [D/G] (gated twin:
    a default B picked from the FM model changes the unrouted points'
    bits through the sub-batch composition), FM15 (gated twin: the
    route ignored changes the unrouted points)."""

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_two_runs_bitwise(self, cupy_gpu, device_precision, seed_precision):
        kwargs = _gated_fm_kwargs(device_precision, seed_precision)
        first, second = (gated_fm_capture_run(chunksize=8, **kwargs) for _ in range(2))
        assert_properties_bitwise(first, second)
        assert set(FOURIER_MELLIN_PROP_NAMES) <= set(first)

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_two_auto_runs_bitwise_with_the_retry(self, cupy_gpu, device_precision):
        kwargs = _gated_fm_kwargs(device_precision)
        first, second = (
            g8_run(fourier_mellin="auto", chunksize=8, **kwargs) for _ in range(2)
        )
        assert_properties_bitwise(first, second)

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("route", [ROUTE_ACCEPT, ROUTE_FORCED])
    def test_routing_invariance(
        self, cupy_gpu, route, device_precision, seed_precision
    ):
        # FM29: one sub-batch of P = 32 slots; every slot routed against
        # ONLY slot k routed; slot k's row, angle and decision bitwise
        cp = cupy_gpu
        p = GATED_FM_INVARIANCE_P
        state, targets, prepared = gated_fm_slots("G1", p)
        everyone = _gated_fm_device_seed(
            cp,
            state,
            targets,
            route,
            device_precision=device_precision,
            seed_precision=seed_precision,
            prepared=prepared,
        )
        all_rows, all_angle, all_applied = _gated_fm_host_outputs(cp, everyone)
        assert np.isfinite(all_angle).all()
        for k in GATED_FM_INVARIANCE_SLOTS:
            alone = np.zeros(p, dtype=np.int8)
            alone[k] = route
            single = _gated_fm_device_seed(
                cp, state, targets, alone, seed=everyone, prepared=prepared
            )
            rows, angle, applied = _gated_fm_host_outputs(cp, single)
            assert np.array_equal(rows[k], all_rows[k], equal_nan=True), k
            assert np.array_equal(angle[k], all_angle[k], equal_nan=True), k
            assert applied[k] == all_applied[k], k
            # the other slots: route 0, the Stage E rows, NaN angles
            others = np.arange(p) != k
            assert np.isnan(angle[others]).all(), k
            assert not applied[others].any(), k

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_batch_size_invariance(
        self, cupy_gpu, monkeypatch, device_precision, record_property
    ):
        built = _gated_fm_spy_sessions(monkeypatch)
        results = {}
        default_batch = None
        for chunksize in (8, 32, 40, None):
            start = len(built)
            results[chunksize] = gated_fm_ramp_run(
                fourier_mellin="auto",
                chunksize=chunksize,
                **_gated_fm_kwargs(device_precision),
            )
            if chunksize is None:
                default_batch = built[start]
        record_property("default_batch_size", default_batch)
        large = [32, 40] + ([None] if default_batch >= 32 else [])
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
        assert_within(
            worst, GATED_FM_BATCH_INVARIANCE_TOL, "GATED_FM_BATCH_INVARIANCE_TOL"
        )

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_unrouted_points_keep_the_off_bits(
        self, cupy_gpu, device_precision, record_property
    ):
        kwargs = _gated_fm_kwargs(device_precision)
        off = gated_fm_ramp_run(fourier_mellin="off", **kwargs)
        auto = gated_fm_ramp_run(fourier_mellin="auto", **kwargs)
        seed = np.asarray(auto["fourier_mellin_seed"])
        angle = np.asarray(auto["fourier_mellin_angle"])
        untouched = (seed == SEED_TRANSLATION) & np.isnan(angle)
        # premises: routed points exist (ramp columns 2 to 6) and so do
        # untouched fitted ones (ramp columns 0, 1, the reference rows)
        assert np.isfinite(angle).any()
        assert untouched.sum() >= 2
        record_property("untouched_points", int(untouched.sum()))
        for name in off:
            a = np.asarray(off[name])[untouched]
            b = np.asarray(auto[name])[untouched]
            if np.issubdtype(a.dtype, np.floating):
                assert np.array_equal(a, b, equal_nan=True), name
            else:
                assert np.array_equal(a, b), name


class TestGatedFourierMellinVram:
    """V10(l) gated (D22.18, D21.10.2): the FM terms of the VRAM model
    calibrated SEPARATELY against pool high-water marks at 512x622
    (G3) -- the per-reference FM term (the FM state: the box resident,
    the profile and the run's look-up table) within ``FM_VRAM_R_BOUNDS``
    and never above the model's ``r(True) - r(False)``, the per-slot FM
    term of a routed sub-batch within ``FM_VRAM_P_BOUNDS`` and never
    above ``p(True) - p(False)``; the whole FM model bounding an
    ``"always"`` run's high-water mark; and the default B equal under
    ``"off"`` and ``"auto"`` on this card, the chooser told
    ``fourier_mellin=True`` exactly when a route is nonzero, the retry
    session built at the first pass's B.

    Mutants: FM46 [D/G] (``_default_batch_size`` picking B from the FM
    model where the FM terms fit: the equal-default-B arm is its gated
    killer, the fake-VRAM arm of (k) the default one), FM48 (gated twin:
    a retry session at a newly chosen B)."""

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_fm_resident_term(
        self, cupy_gpu, device_precision, seed_precision, record_property
    ):
        cp = cupy_gpu
        state, _ = gated_fm_parity_set("G3")
        n_pixels = SHAPE_RECT[0] * SHAPE_RECT[1]
        ctx = _gated_fm_context(cp, device_precision)
        resident = _batched.build_resident(ctx, state)
        seed_state = _batched.build_seed_state(ctx, state, precision=seed_precision)
        with _GatedFmPoolHighWater(cp) as tracker:
            fm_state = _fourier_mellin.build_fourier_mellin_state(ctx, state, resident)
            held = int(tracker.pool.used_bytes()) - tracker.base
        measured = tracker.high_water
        record_property("r_fm_high_water_bytes", measured)
        record_property("r_fm_held_bytes", held)
        assert fm_state is not None and seed_state is not None
        assert fm_state.resident is resident
        _, _, r_off = _gpu._vram_model_terms(n_pixels, device_precision, seed_precision)
        _, _, r_on = _gpu._vram_model_terms(
            n_pixels, device_precision, seed_precision, fourier_mellin=True
        )
        assert r_on - r_off >= measured
        _gated_fm_assert_in_bounds(
            measured, FM_VRAM_R_BOUNDS, f"FM_VRAM_R_BOUNDS ({device_precision})"
        )

    @pytest.mark.parametrize("seed_precision", GATED_FM_SEED_PRECISIONS)
    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_fm_per_slot_term(
        self, cupy_gpu, device_precision, seed_precision, record_property
    ):
        cp = cupy_gpu
        n_pixels = SHAPE_RECT[0] * SHAPE_RECT[1]
        state, _ = gated_fm_parity_set("G3")
        ctx = _gated_fm_context(cp, device_precision)
        seed_state, _, _ = _gated_fm_seed_state(ctx, state, seed_precision)
        extra = {}
        for p in GATED_FM_VRAM_SLOTS:
            _, _, (preprocessed, coefficients) = gated_fm_slots("G3", p)
            peaks = {}
            for routed in (False, True):
                extras = {ROUTE_KEY: _gated_fm_route(ROUTE_ACCEPT, p)} if routed else {}
                batch = _batched.SeedBatch(
                    cp.asarray(preprocessed),
                    cp.asarray(coefficients),
                    np.arange(p, dtype=np.int64),
                    extras=extras,
                )
                spectra = _batched.seed_spectra(ctx, batch, seed_state)
                with _GatedFmPoolHighWater(cp) as tracker:
                    rows = _batched.seed_homographies(ctx, batch, spectra, seed_state)
                    del rows
                peaks[routed] = tracker.high_water
                record_property(f"seam_high_water_{p}_{routed}", peaks[routed])
                del batch, spectra
            extra[p] = peaks[True] - peaks[False]
        low, high = GATED_FM_VRAM_SLOTS
        p_fm = (extra[high] - extra[low]) / (high - low)
        record_property("p_fm_bytes", p_fm)
        _, p_off, _ = _gpu._vram_model_terms(n_pixels, device_precision, seed_precision)
        _, p_on, _ = _gpu._vram_model_terms(
            n_pixels, device_precision, seed_precision, fourier_mellin=True
        )
        assert p_on - p_off >= p_fm
        _gated_fm_assert_in_bounds(
            p_fm, FM_VRAM_P_BOUNDS, f"FM_VRAM_P_BOUNDS ({device_precision})"
        )

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    def test_the_fm_model_bounds_an_always_run(
        self, cupy_gpu, device_precision, record_property
    ):
        cp = cupy_gpu
        fixture = gated_fm_g3_map()
        n_pixels = SHAPE_RECT[0] * SHAPE_RECT[1]
        for batch_size in (8, 16):
            with _GatedFmPoolHighWater(cp) as tracker:
                run_engine(
                    np.asarray(fixture["patterns"]),
                    fixture["navigation_shape"],
                    fixture["detector"],
                    reference=(0, 0),
                    fourier_mellin="always",
                    max_iterations=GATED_FM_VRAM_MAX_ITERATIONS,
                    chunksize=batch_size,
                    **_gated_fm_kwargs(device_precision),
                )
            record_property(f"high_water_{batch_size}", tracker.high_water)
            model = _gpu._vram_model_bytes(
                batch_size,
                n_pixels,
                device_precision,
                "complex128",
                fourier_mellin=True,
            )
            assert tracker.high_water <= model, batch_size

    def test_default_batch_size_is_the_off_one(
        self, cupy_gpu, monkeypatch, record_property
    ):
        calls = []
        real_default = _gpu._default_batch_size

        def spying_default(*args, **kwargs):
            value = real_default(*args, **kwargs)
            calls.append((args, dict(kwargs), int(value)))
            return value

        monkeypatch.setattr(_gpu, "_default_batch_size", spying_default)
        built = _gated_fm_spy_sessions(monkeypatch)
        chosen = {}
        for mode in ("off", "auto"):
            start_calls, start_built = len(calls), len(built)
            gated_fm_ramp_run(fourier_mellin=mode, chunksize=None, **_gated_fm_kwargs())
            mode_calls = calls[start_calls:]
            assert mode_calls, f"{mode}: chunksize=None must consult the chooser"
            flags = {
                bool(kwargs.get("fourier_mellin", False)) for _, kwargs, _ in mode_calls
            }
            # the runner passes fourier_mellin=True iff a route is nonzero
            # (ramp columns 2 to 6 are routed under "auto")
            if mode == "auto":
                assert True in flags, mode
            else:
                assert flags == {False}, mode
            chosen[mode] = built[start_built:]
            record_property(f"sessions_{mode}", chosen[mode])
        assert chosen["off"] and chosen["auto"]
        assert chosen["auto"][0] == chosen["off"][0]
        assert chosen["auto"][0] in GATED_FM_BATCH_SIZES
        # the retry pass (if any) builds its session at the first B
        assert set(chosen["auto"]) == {chosen["auto"][0]}


class TestGatedFourierMellinThroughput:
    """D22.15, RECORDED only (``record_property``), asserting nothing
    but completion: best of three timed runs after a warm-up of the V8
    ramp map with its crystal map under ``"off"``, ``"auto"`` and
    ``"always"`` at both device precisions, and the device cost per
    slot of a routed sub-batch at P = 32 (the seam with every slot
    routed against no route key) at 480x480 (G1) and 512x622 (G3).
    The whole-map Si-indent record is a ledger measurement (V10(m)),
    never a test.  No mutant: no killer."""

    @pytest.mark.parametrize("device_precision", GATED_FM_DEVICE_PRECISIONS)
    @pytest.mark.parametrize("mode", GATED_FM_THROUGHPUT_MODES)
    def test_map_throughput_recorded(
        self, cupy_gpu, mode, device_precision, record_property
    ):
        kwargs = _gated_fm_kwargs(device_precision)
        gated_fm_ramp_run(fourier_mellin=mode, **kwargs)  # warm-up
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            properties = gated_fm_ramp_run(fourier_mellin=mode, **kwargs)
            best = min(best, time.perf_counter() - start)
        record_property("machine", GATED_FM_MACHINE_A)
        record_property("best_of_3_s", best)
        record_property("patterns_per_s", RAMP_SIZE / best)
        assert np.asarray(properties["homography"]).shape == (RAMP_SIZE, 8)

    @pytest.mark.parametrize("name", ["G1", "G3"])
    def test_seam_cost_per_routed_slot_recorded(self, cupy_gpu, name, record_property):
        cp = cupy_gpu
        p = GATED_FM_SUB_BATCH_SIZE
        state, slots, prepared = gated_fm_slots(name, p)
        seed = _gated_fm_device_seed(
            cp, state, slots, ROUTE_ACCEPT, prepared=prepared
        )  # warm-up
        timings = {}
        for label, route in (("off", None), ("routed", ROUTE_ACCEPT)):
            best = float("inf")
            for _ in range(3):
                cp.cuda.Device().synchronize()
                start = time.perf_counter()
                result = _gated_fm_device_seed(
                    cp, state, slots, route, seed=seed, prepared=prepared
                )
                cp.cuda.Device().synchronize()
                best = min(best, time.perf_counter() - start)
            timings[label] = best
        record_property("machine", GATED_FM_MACHINE_A)
        record_property("seam_s_off", timings["off"])
        record_property("seam_s_routed", timings["routed"])
        record_property(
            "fm_ms_per_slot", 1e3 * (timings["routed"] - timings["off"]) / p
        )
        assert tuple(result.rows.shape) == (p, N_HOMOGRAPHY_PARAMETERS)


# ===== MUTATION MAP =====
#
# Plan 12 item 4, FM1 to FM52, each with its DESIGNED killer(s) in
# this module (filled at the Stage F failing-tests gate, 2026-10-08).
# [D] default suite, [G] gated, [D/G] both.  Class names drop the
# ``TestFourierMellin`` prefix; ``Gated...`` stands for
# ``TestGatedFourierMellin...``.  Every killer fails now on the
# skeleton except the FM16 and FM41 arms of the switch class, which
# pass now by design (they guard the unchanged ``"off"`` path and the
# forwarding the skeleton already has) and see the mutant once it
# exists.
#
#  FM1 [D]:
#      SeedRows::test_the_derotated_crop_is_the_cpu_bicubic_evaluation,
#      SeedRows::test_pc_shift_rows_within_the_seed_band,
#      Capture::test_the_fm_seed_captures_what_off_misses (weekly also
#      Capture::test_wide_border_converges_to_the_exact_seeded_optimum)
#  FM2 [D]: SeedRows::test_pc_shift_rows_within_the_seed_band (the ONLY
#      killer)
#  FM3 [D]: Angle::test_the_lut_reconstructs_physical_frequency,
#      Angle::test_g3_non_square_seed_level
#  FM4 [D]: Angle::test_the_lut_reconstructs_physical_frequency,
#      Angle::test_g1_twists_seed_level
#  FM5 [D]:
#      SeedRows::test_a_planted_failure_returns_h_t_for_that_slot_only
#  FM6 [D]: EdgeTreatment::test_the_stencil_returns_a_new_array,
#      EdgeTreatment::test_seed_homographies_leaves_the_target_spectra_bitwise,
#      EdgeTreatment::test_unrouted_rows_are_bitwise_the_stage_e_rows;
#      gated twin GatedContract::test_seam_output_contract
#  FM7 [D]: EdgeTreatment::test_the_g4_background_lock
#  FM8 [D/G]: Angle::test_the_peak_honours_the_search_window; gated
#      twin GatedContract::test_the_peak_rule_on_the_device (the window,
#      tie, edge and plateau rows bitwise numpy; critic F1, 2026-10-08)
#  FM9 [D]: Angle::test_g1_twists_seed_level
#  FM10 [D]: Acceptance::test_g5_premises_and_the_fm_row_is_kept,
#      Acceptance::test_a_planted_wrong_angle_keeps_the_translation_row,
#      Capture::test_the_fm_seed_captures_what_off_misses
#  FM11 [D]:
#      Acceptance::test_a_planted_wrong_angle_keeps_the_translation_row,
#      Acceptance::test_a_planted_tie_keeps_the_translation_row
#  FM12 [D]: Acceptance::test_g5_premises_and_the_fm_row_is_kept (G5
#      arm; peaks 0.288 / 0.253),
#      Acceptance::test_the_unrelated_pair_refuses_the_fm_row (G6 arm;
#      peaks 0.0174 / 0.0206, opposite to the criteria); both
#      separations asserted as local premises (critic F2, 2026-10-08)
#  FM13 [D]: SeedRows::test_a_nan_translation_row_is_never_rescued
#  FM14 [D]: Acceptance::test_a_planted_tie_keeps_the_translation_row
#  FM15 [D]:
#      NumpySession::test_unrouted_slots_and_sub_batches_return_the_stage_e_rows,
#      CpuRoute::test_unrouted_and_translation_won_points_equal_off,
#      EdgeTreatment::test_unrouted_rows_are_bitwise_the_stage_e_rows;
#      gated twins
#      GatedDeterminism::test_unrouted_points_keep_the_off_bits,
#      GatedContract::test_seam_output_contract
#  FM16 [D]: Switch::test_off_builds_no_state_and_writes_nothing_cpu,
#      Switch::test_off_builds_no_state_and_writes_nothing_numpy_session,
#      CpuRoute::test_fit_chunk_rows_are_14_wide_on_fm_runs_and_12_on_off;
#      gated twin GatedContract::test_off_is_the_stage_e_device_path
#  FM17 [D]:
#      Gate::test_imposed_twists_are_recovered_with_their_sign[True]
#      (the tilted detector)
#  FM18 [D]:
#      Gate::test_a_symmetry_equivalent_orientation_gives_the_same_twist
#  FM19 [D]:
#      Gate::test_out_of_plane_rotations_give_zero_twist_and_swing_is_ignored
#  FM20 [D]: Gate::test_the_routes_on_planted_twists (negative twists),
#      Gate::test_imposed_twists_are_recovered_with_their_sign
#  FM21 [D]: Gate::test_the_routes_on_planted_twists (planted boundary)
#  FM22 [D]: Gate::test_unusable_points_give_nan,
#      Gate::test_the_engine_fails_open_on_unusable_points
#  FM23 [D]:
#      Gate::test_each_point_is_measured_against_its_grain_reference
#  FM24 [D]: Retry::test_the_unrelated_point_keeps_its_first_result
#  FM25 [D]: Retry::test_auto_retries_exactly_the_failed_points_once,
#      Retry::test_converged_and_masked_points_are_never_retried,
#      Retry::test_always_fits_the_mislabelled_point_in_the_first_pass
#      (the "always" subset, literal)
#  FM26 [D]: Acceptance::test_criterion_calls_on_the_cpu_route,
#      Acceptance::test_criterion_calls_in_the_numpy_session
#  FM27 [D]: Retry::test_the_mislabelled_point_converges_from_its_fm_row
#      (the direct-fit oracle)
#  FM28 [D]:
#      NumpySession::test_route_flags_arrive_in_fit_order_with_zero_padding
#      (FK-TWO-GRAIN)
#  FM29 [D/G]: GatedDeterminism::test_routing_invariance (designed
#      DEVICE killer);
#      NumpySession::test_a_routed_row_does_not_depend_on_how_many_slots_are_routed
#      (may be reviewed-equivalent under numpy)
#  FM30 [D]:
#      CpuRoute::test_the_seam_runs_at_one_slot_for_routed_and_retried_points_only,
#      CpuRoute::test_chunksize_invariance_and_repeatability
#  FM31 [D]: SeedRows::test_the_dead_band_pixels_are_in_the_crop
#  FM32 [D]: SeedRows::test_the_partial_row_is_the_frozen_formula,
#      SeedRows::test_the_routed_rows_are_their_own_partial_rows,
#      Capture::test_the_fm_seed_captures_what_off_misses
#  FM33 [D]:
#      Switch::test_the_public_result_carries_the_props_under_always,
#      RampRescue::test_auto_rescues_exactly_the_routed_points,
#      RampRescue::test_masked_points_and_the_prop_set,
#      Retry::test_the_mislabelled_point_converges_from_its_fm_row,
#      Retry::test_always_fits_the_mislabelled_point_in_the_first_pass
#  FM34 [D]: Switch::test_the_combination_raises_on_both_backends,
#      Switch::test_the_combination_raises_through_the_signal_method
#  FM35 [D]: Switch::test_auto_without_xmap_raises_at_the_engine
#  FM36 [D]: Gate::test_a_map_without_twists_warns_and_routes_nothing
#      (the constant arm asserts its twists nonzero and below 1e-9 deg,
#      so the float-equality variant dies; critic F6, 2026-10-08),
#      Gate::test_a_map_with_twists_does_not_warn
#  FM37 [D/G]:
#      NumpySession::test_complex64_keeps_the_fm_arithmetic_in_double;
#      GatedParity::test_theta_hat_and_acceptance_parity (complex64
#      arms, designed DEVICE killer),
#      GatedContract::test_frozen_functions_and_state_on_the_device
#  FM38 [D]: SeedRows::test_pc_shift_rows_within_the_seed_band (the
#      pin asserted below the local mutant's corner errors; critic F9,
#      2026-10-08), Capture::test_the_fm_seed_captures_what_off_misses (weekly
#      also
#      Capture::test_wide_border_converges_to_the_exact_seeded_optimum)
#  FM39 [D]: NO KILLER YET: reviewed-equivalent on the G7 maps (one
#      projection centre, the twist depends on the detector tilt chain
#      only; Gate class docstring)
#  FM40 [D]:
#      Retry::test_a_planted_angle_failure_in_the_retry_skips_the_second_fit,
#      Retry::test_a_constant_pattern_without_band_pass_is_never_refitted,
#      NumpySession::test_a_forced_slot_whose_estimate_fails_is_inactive
#  FM41 [D]: Switch::test_public_method_forwards_fourier_mellin
#  FM42 [D]:
#      Switch::test_the_public_result_carries_the_props_under_always
#  FM43 [D]: SeedRows::test_pc_shift_rows_within_the_seed_band,
#      SeedRows::test_the_derotated_crop_is_the_cpu_bicubic_evaluation
#  FM44 [D]:
#      CpuRoute::test_the_seam_runs_at_one_slot_for_routed_and_retried_points_only
#      (the seed_spectra count, the ONLY killer)
#  FM45 [D]:
#      NumpySession::test_extras_stay_input_only_and_outputs_follow_the_route;
#      gated twin GatedContract::test_seam_output_contract
#  FM46 [D/G]:
#      NumpySession::test_the_batch_choice_halves_only_when_the_fm_terms_do_not_fit,
#      NumpySession::test_the_runner_asks_the_fm_model_only_on_routed_runs;
#      GatedVram::test_default_batch_size_is_the_off_one (designed
#      DEVICE killer)
#  FM47 [D]:
#      Retry::test_always_fits_the_mislabelled_point_in_the_first_pass
#      (the subset pinned literally, [unrelated, constant]; critic F3,
#      2026-10-08)
#  FM48 [D]:
#      NumpySession::test_the_retry_runs_at_the_first_pass_batch_size;
#      gated twins GatedParity::test_retry_routing_parity,
#      GatedVram::test_default_batch_size_is_the_off_one
#  FM49 [D]:
#      Retry::test_the_constant_pattern_on_g8_keeps_the_failure_contract
#  FM50 [D]:
#      NumpySession::test_the_fm_state_is_built_lazily_once_per_routed_grain
#      (one build per runner pass that fits a routed-grain point;
#      critic F4, 2026-10-08)
#  FM51 [D]:
#      Acceptance::test_a_nan_translation_criterion_keeps_the_translation_row
#  FM52 [D]: SeedRows::test_the_outputs_rule,
#      NumpySession::test_extras_stay_input_only_and_outputs_follow_the_route;
#      gated twin GatedContract::test_seam_output_contract
