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

"""Tests of the non-local pattern averaging (NLPAR) kernels and drivers
in ``kikuchipy.pattern._nlpar``.

Covers:

- Kernel discipline (``TestKernels``): the five Numba kernels are
  compiled with ``cache=True`` and ``nogil=True`` and neither
  ``parallel`` nor ``fastmath``, agree with their ``py_func``, keep
  float32 through both routes, contain no power operator, and the
  window bounds equal PyEBSDIndex's expressions with a clamp for an
  axis shorter than the window. Neither the module nor the NLPAR
  methods of ``EBSD`` import PyEBSDIndex.
- Parity with PyEBSDIndex's compiled ``NLPAR.sigma_numba``
  (``TestSigmaOracle``: sigma and the normalised 3 x 3 distances) and
  ``NLPAR.nlpar_nb`` (``TestAveragingOracle``: the averaged patterns,
  the border policy, the per-pair pixel count, the kept region of a
  haloed block), on small synthetic maps and on ``nickel_ebsd_large``.
- The halo depth rule, the pass-1 chunk wrapper and the two Dask
  drivers on multi-chunk arrays (``TestDepthAndHalo``), without
  PyEBSDIndex, with call-counting spies on the chunk wrappers and a
  recorder of the chunks reaching ``dask.array.overlap.overlap``, so
  that the chunked route itself is checked and not only its values.
- The lambda objective and its optimiser (``TestLambdaOracle``): the
  closed form on constructed distance fields, the exclusion of
  out-of-map neighbours, the mean over points, the bound warnings, the
  stride of large maps, and a test-local transcription of PyEBSDIndex's
  objective on its own distances.
- The recorded deviations from PyEBSDIndex (``TestPolicyOracles``),
  each pinned against a test-local transcription and, where
  PyEBSDIndex is installed, against its compiled kernels.
- Runtimes on ``nickel_ebsd_large`` (``TestPerformance``), recorded and
  never asserted.

The oracle is always called through the compiled dispatchers. Their
``py_func`` is not bitwise equal to the compiled kernels (NumPy and
Numba promote ``float32 ** int32``, ``2.0 * float32`` and a float64
counter differently), so it is never an oracle. Every navigation axis
passed to ``nlpar_nb`` is at least ``2 sr + 1`` long, since it indexes
out of bounds otherwise, and every call gets a fresh ``calclim``
array, since the kernel writes into its default argument.

The synthetic map generators, the shared ``exp_kernel_ulp`` tolerance,
the inscribed-circle signal mask ``circle_mask`` and the call-counting
``counting_spy`` are fixtures of the root ``conftest.py``.
"""

import ast
from collections.abc import Callable
from importlib.metadata import version
import inspect
import math
import textwrap
import time
import warnings

import dask
import dask.array as da
from dask.array.overlap import ensure_minimum_chunksize
import h5py
import numpy as np
from packaging.version import Version
import pytest
from scipy.optimize import minimize

import kikuchipy as kp
from kikuchipy._constants import dependency_version
from kikuchipy.pattern import _nlpar
from kikuchipy.signals.util._dask import get_dask_array

# ------------------------- Measured-then-pinned ------------------------ #

# Measured, then pinned: the largest float32 ulp difference allowed
# between our sigma and the sigma of the compiled NLPAR.sigma_numba. The
# drafting seed was 0, i.e. bitwise; a non-zero pin must name the
# platform that forces it.
# Pinned 2026-10-05 at the measured 0: the 18 sigma parity tests (four
# generators x mask x protection, nickel_ebsd_large raw and corrected)
# are bitwise on a 20-core Intel Raptor Lake laptop, Windows 11.
SIGMA_PARITY_ULP = 0

# Measured, then pinned: the largest float32 ulp difference allowed per
# pixel between our averaged patterns and the output of the compiled
# NLPAR.nlpar_nb. The drafting seed was 0, i.e. bitwise.
# Pinned 2026-10-05 at the measured 0: the 392 averaging parity tests,
# nickel_ebsd_large included, are bitwise on the same machine.
AVERAGE_PARITY_ULP = 0

# Measured, then pinned: the smallest worst-pixel difference, in grey
# levels, between the shifted-inward search window and a clamped or a
# zero-extended window on the border band of a (7, 8 | 12, 12) map of
# one pattern plus Gaussian noise of sigma 8 at search radius 2. The
# drafting seed was of the order of 1 grey level; the pin is half the
# measured minimum.
# Pinned 2026-10-05: measured 7.2600 grey levels, the same worst border
# pixel for the oracle and for our route against both the clamped and
# the zero-extended window; pinned at half, 3.63.
BORDER_BAND_MIN_DIFF: float = 3.63

# Recorded, never asserted, and not read by any test (documentation of
# the measurement only): the wall time in seconds of the first calls of
# both compiled PyEBSDIndex kernels in the pyebsdindex_kernels fixture.
# The drafting seed was 7.2 s, the cold compile of 4.0 s (sigma_numba)
# plus 3.2 s (nlpar_nb) measured outside pytest.
# Recorded 2026-10-05, three fresh processes each: a cold compile (fresh
# dispatchers without an on-disk cache) takes 7.39-7.40 s (4.06-4.08 s
# plus 3.31-3.34 s); with the kernels' numba cache present the fixture
# measures 0.078-0.088 s.
PYEBSDINDEX_JIT_WARMUP_S = 7.4

# The largest relative difference allowed between the optimised lambda
# and the closed form sqrt(-c / ln((1 / tw - 1) / 8)) on the
# constructed fields of eight neighbours at distance c. The drafting
# seed was 1e-3, the class in lambda that Nelder-Mead's fatol of 1e-4
# on the objective gives.
# Pinned 2026-10-05: measured 6.87e-5 at worst (the border-masked c = 2
# field at target weight 0.34; the nine all-valid fields 5.3e-7 to
# 4.9e-5), identical over three runs; pinned at ~2x the worst.
LAMBDA_CLOSED_FORM_REL = 1.4e-4

# Band (low, high) of the ratio of our phantom-free lambda to the lambda
# of PyEBSDIndex's objective (out-of-map slots counted with weight 1) at
# target weight 0.34 on nickel_ebsd_large, raw and background-corrected.
# Drafting seeds 1.1387 / 1.1164 = 1.020 (raw) and 2.5787 / 2.5246 =
# 1.021 (corrected).
# Pinned 2026-10-05: measured 1.138672 / 1.116406 = 1.019944 (raw) and
# 2.578711 / 2.524609 = 1.021430 (corrected), identical over three runs;
# pinned on the excess over 1, from half the smaller excess to twice the
# larger, so a ratio of 1 (phantoms counted by ours too) stays outside.
LAMBDA_PHANTOM_RATIO: tuple[float, float] = (1.0099, 1.043)

# ------------------------------ Constants ----------------------------- #

# Every Numba kernel of the module, for the flag and py_func tests
KERNEL_NAMES = [
    "_window_bounds",
    "_nlpar_sigma_kernel",
    "_nlpar_distances_kernel",
    "_nlpar_weights_kernel",
    "_nlpar_weighted_sum_kernel",
]

# Oracle maps: every navigation axis >= 7, so search radius 3 is legal
# in nlpar_nb
ORACLE_NAV = (7, 8)
ORACLE_SIG = (12, 12)

# Navigation chunkings of the multi-chunk driver tests on (10, 16 | 16, 16)
DRIVER_CHUNKINGS = [
    (5, 8),
    ((3, 3, 4), (7, 7, 2)),
    ((1, 9), (2, 14)),
    (1, 1),
    (2, 3),
]

# Kernel py_func cases: (map shape, (radius rows, radius cols), kept
# region (row start, row stop, col start, col stop)); the (1, 7) and
# (2, 2) maps have axes shorter than the window
DISTANCE_CASES = [
    ((5, 6, 8, 8), (1, 2), (0, 5, 0, 6)),
    ((5, 6, 8, 8), (3, 3), (0, 5, 0, 6)),
    ((5, 6, 8, 8), (1, 2), (1, 4, 2, 5)),
    ((1, 7, 4, 4), (1, 2), (0, 1, 0, 7)),
    ((1, 7, 4, 4), (3, 3), (0, 1, 0, 7)),
    ((2, 2, 4, 4), (3, 3), (0, 2, 0, 2)),
]

# Closed-form lambda, to four decimals, of a point with eight neighbours
# at distance c for the target weights 0.5, 0.34 and 0.25 (computed
# 2026-10-04 from sqrt(-c / ln((1 / tw - 1) / 8)))
CLOSED_FORM_LAMBDA = {
    (2, 0.5): 0.9807,
    (2, 0.34): 1.1884,
    (2, 0.25): 1.4280,
    (5, 0.5): 1.5506,
    (5, 0.34): 1.8790,
    (5, 0.25): 2.2578,
    (12, 0.5): 2.4022,
    (12, 0.34): 2.9110,
    (12, 0.25): 3.4978,
}

requires_pyebsdindex = pytest.mark.skipif(
    dependency_version["pyebsdindex"] is None, reason="pyebsdindex is not installed"
)

# PyEBSDIndex's NLPAR.nlpar_nb (parallel=True) does not compile under
# numba 0.57.0, whose parfor pass rejects a dtype keyword of np.arange
# ("got an unexpected keyword argument 'dtype'"); it compiles under
# numba 0.58.0 and later. NLPAR.sigma_numba compiles under 0.57.0, so
# only the calls of nlpar_nb are skipped there
NLPAR_NB_COMPILES = Version(version("numba")) >= Version("0.58.0")

# ---------------------------- Generic helpers -------------------------- #


def _njit_kernel_names(module) -> list[str]:
    """Return the names of the module's own Numba kernels.

    Only dispatchers whose Python function is defined in the module
    itself are returned, so that a kernel imported from another
    module of the package is not counted.
    """
    return sorted(
        name
        for name, value in vars(module).items()
        if type(value).__name__ == "CPUDispatcher"
        and getattr(value, "py_func", None) is not None
        and value.py_func.__module__ == module.__name__
    )


def _py_func(kernel):
    """Return the pure Python function of a Numba kernel.

    Falls back to the function itself while it is still an undecorated
    stub. Every caller first asserts that the kernel does carry a
    ``py_func``, so that an implementation without ``@njit`` fails
    loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


def _require_placeholder(value: int | float | None, name: str) -> int | float:
    """Return a measured-then-pinned tolerance, or fail if unfilled."""
    if value is None:
        pytest.fail(f"unfilled MEASURED-THEN-PINNED placeholder {name}")
    return value


def _assert_within_ulp(actual: np.ndarray, desired: np.ndarray, n_ulp: int) -> None:
    """Assert float32 arrays equal, bitwise for 0 ulp, else within
    ``n_ulp`` float32 ulps.
    """
    assert actual.dtype == np.float32, actual.dtype
    assert desired.dtype == np.float32, desired.dtype
    assert actual.shape == desired.shape
    if n_ulp == 0:
        np.testing.assert_array_equal(actual, desired)
    else:
        np.testing.assert_array_max_ulp(actual, desired, maxulp=n_ulp, dtype=np.float32)


def _assert_average_parity(ours: np.ndarray, expected: np.ndarray) -> None:
    """Assert our float32 average equals the oracle's, with the hard
    bound of the integer output policy: after ``rint`` no pixel differs
    by 2 grey levels or more.
    """
    _assert_within_ulp(ours, expected, AVERAGE_PARITY_ULP)
    diff = np.abs(
        np.rint(ours.astype(np.float64)) - np.rint(expected.astype(np.float64))
    )
    assert np.count_nonzero(diff >= 2) == 0


def _kept_indices(
    signal_mask: np.ndarray | None, sig_shape: tuple[int, int]
) -> np.ndarray:
    """Return the flat int64 indices of the kept pixels, PyEBSDIndex's
    ``indices`` for a kikuchipy mask.
    """
    if signal_mask is None:
        return np.arange(sig_shape[0] * sig_shape[1], dtype=np.int64)
    keep = ~np.asarray(signal_mask, dtype=bool).ravel()
    return np.flatnonzero(keep).astype(np.int64)


def _as_pixels(patterns: np.ndarray) -> np.ndarray:
    """Return a (n_rows, n_cols, h, w) map as float32 of shape
    (n_rows, n_cols, h * w).
    """
    nrows, ncols, h, w = patterns.shape
    return patterns.astype(np.float32).reshape(nrows, ncols, h * w)


def _block_info(location: tuple[int, int], num_chunks: tuple[int, int]) -> dict:
    """Return a hand-built Dask ``block_info`` for a 4D block and its
    4D second operand, navigation chunk location and chunk counts as
    given, one chunk along each signal axis.
    """
    info = {
        "chunk-location": tuple(location) + (0, 0),
        "num-chunks": tuple(num_chunks) + (1, 1),
    }
    return {0: dict(info), 1: dict(info), None: dict(info)}


def _dask_map(x: np.ndarray, nav_chunks: tuple) -> da.Array:
    """Return a Dask array of a 4D map with the given navigation chunks
    and each signal axis in one chunk.
    """
    nav = da.core.normalize_chunks(nav_chunks, shape=x.shape[:2])
    return da.from_array(x, chunks=nav + ((x.shape[2],), (x.shape[3],)))


def _hand_built_map(
    nav_shape: tuple[int, int] = (3, 3),
    n_pix: int = 16,
    seed: int = 0,
    noise: float = 8.0,
) -> np.ndarray:
    """Return a float32 map of shape ``nav_shape + (n_pix,)``: one ramp
    from 40 to 180 plus seeded Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(40.0, 180.0, n_pix, dtype=np.float32)
    noisy = base + rng.normal(0.0, noise, nav_shape + (n_pix,))
    return noisy.astype(np.float32)


def _record_overlap_chunks(monkeypatch) -> list:
    """Wrap :func:`dask.array.overlap.overlap` so that it records the
    chunks of every array passed to it, and return the list of recorded
    chunks.

    Dask's ``overlap`` (2026.3.0: ``allow_rechunk=True``) rechunks an
    axis thinner than its depth by itself, so the values of a driver
    that skips its own rechunk are often unchanged; the chunks reaching
    ``overlap`` are not.
    """
    original = da.overlap.overlap
    recorded = []

    def recorder(x, *args, **kwargs):
        recorded.append(tuple(tuple(c) for c in x.chunks))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(da.overlap, "overlap", recorder)
    return recorded


def _ref_depth_chunks(
    chunks: tuple[tuple[int, ...], ...], radius: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Return the halo depth per navigation axis and the chunks of the
    whole array for the averaging pass, by the depth rule written
    independently of the module.

    Per navigation axis with radius ``r``, chunks ``c`` and length
    ``n``: one chunk -> depth 0, chunks unchanged; ``2 r + 1 >= n`` ->
    depth 0, one chunk; else depth ``max(r, 2 r + 1 - min(c[0], c[-1]))``
    and chunks ``ensure_minimum_chunksize(depth, c)``. The signal axes
    keep their chunks.
    """
    depth = []
    nav_out = []
    for c, r in zip(chunks, radius):
        c = tuple(c)
        n = sum(c)
        if len(c) == 1:
            depth.append(0)
            nav_out.append(c)
        elif 2 * r + 1 >= n:
            depth.append(0)
            nav_out.append((n,))
        else:
            d = max(r, 2 * r + 1 - min(c[0], c[-1]))
            depth.append(d)
            nav_out.append(tuple(ensure_minimum_chunksize(d, c)))
    sig = tuple(tuple(c) for c in chunks[len(radius) :])
    return tuple(depth), tuple(nav_out) + sig


def _n_nav_blocks(chunks: tuple[tuple[int, ...], ...]) -> int:
    """Return the number of blocks over the two navigation axes."""
    return len(chunks[0]) * len(chunks[1])


# ------------------ Test-local references (NumPy only) ----------------- #


def _ref_bounds(center: int, radius: int, n: int, shift: bool) -> tuple[int, int]:
    """Return the half-open window bounds along one axis, written
    independently of the module: clipped (``shift=False``) or
    PyEBSDIndex's shifted-inward window, the whole axis when the axis is
    shorter than the window.
    """
    if not shift:
        return max(center - radius, 0), min(center + radius, n - 1) + 1
    if n < 2 * radius + 1:
        return 0, n
    start = max(center - radius, 0) - max(center + radius - (n - 1), 0)
    stop = min(center + radius, n - 1) + max(radius - center, 0) + 1
    return start, stop


def _sequential_sum(values: np.ndarray) -> np.float32:
    """Return the float32 sum of ``values`` accumulated one by one in
    order, as the kernels do.
    """
    if values.size == 0:
        return np.float32(0.0)
    return np.cumsum(values, dtype=np.float32)[-1]


def _sigma_threshold(
    max_value, saturation_protect: bool, factor: float = 0.9961
) -> np.float64:
    """Return the float64 saturation threshold of the sigma pass."""
    max64 = np.float64(np.float32(max_value))
    if saturation_protect:
        return max64 * np.float64(factor)
    return max64 + np.float64(1.0)


def _average_threshold(
    max_value, saturation_protect: bool, factor: np.float32 = np.float32(0.999)
) -> np.float32:
    """Return the float32 saturation threshold of the averaging pass."""
    max32 = np.float32(max_value)
    if saturation_protect:
        return np.float32(max32 * np.float32(factor))
    return np.float32(max32 + np.float32(1.0))


def _sigma_at(
    values: np.ndarray,
    below: np.ndarray,
    point: tuple[int, int],
    neighbours: list[tuple[int, int]],
    duplicate_guard: str = "positive",
) -> np.float32:
    """Return the float32 sigma of one point from a list of neighbours:
    the square root of the minimum of ``d2 / (2 n2)`` over the pairs
    passing the duplicate guard (``d2 > 0``, or PyEBSDIndex's
    ``d2 >= 1e-3`` evaluated in float64), ``1e12`` if none passes.
    """
    j, i = point
    mind = np.float32(1e24)
    for jn, i_n in neighbours:
        both = below[j, i] & below[jn, i_n]
        diff = values[j, i, both] - values[jn, i_n, both]
        d2 = _sequential_sum(diff * diff)
        n2 = np.float32(np.count_nonzero(both))
        if duplicate_guard == "positive":
            passes = d2 > np.float32(0.0)
        else:
            passes = np.float64(d2) >= np.float64(1e-3)
        if passes:
            s = np.float32(d2 / np.float32(np.float32(2.0) * n2))
            if s < mind:
                mind = s
    return np.float32(np.sqrt(mind))


def _sigma_reference(
    data: np.ndarray,
    kept: np.ndarray,
    max_value,
    saturation_protect: bool,
    factor: float = 0.9961,
    window: str = "clipped",
    duplicate_guard: str = "positive",
) -> np.ndarray:
    """Return the float32 sigma map of a (n_rows, n_cols, n_pix) map by
    the sigma formula over the 3 x 3 window, clipped at the map edges
    (or, for ``window="shifted"``, shifted inward like the search
    window), with sequential float32 accumulation and the float64
    saturation threshold.
    """
    nrows, ncols, _ = data.shape
    threshold = _sigma_threshold(max_value, saturation_protect, factor)
    values = data[:, :, kept]
    below = values.astype(np.float64) < threshold
    shift = window == "shifted"
    sigma = np.empty((nrows, ncols), dtype=np.float32)
    for j in range(nrows):
        r0, r1 = _ref_bounds(j, 1, nrows, shift)
        for i in range(ncols):
            c0, c1 = _ref_bounds(i, 1, ncols, shift)
            neighbours = [
                (jn, i_n)
                for jn in range(r0, r1)
                for i_n in range(c0, c1)
                if (jn, i_n) != (j, i)
            ]
            sigma[j, i] = _sigma_at(values, below, (j, i), neighbours, duplicate_guard)
    return sigma


def _pair_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    s0: np.float32,
    s1: np.float32,
    threshold: np.float32 | np.float64,
    n_correction: int | None = None,
) -> tuple[np.float32, np.float32]:
    """Return the float32 normalised distance and pixel count of one
    pair, a transcription of the pair loop of PyEBSDIndex's
    ``nlpar_nb`` (``pyebsdindex/nlpar_cpu.py`` lines 901-916): ``d2`` and
    ``n2`` accumulated in float32 over the pixels where both values are
    below the threshold, ``d2 -= n2 (s0 + s1)``,
    ``dnorm = (s1 + s0) sqrt(2 n2)``, and ``1e6 n2`` when
    ``dnorm <= 1e-8`` (so 0 when ``n2 == 0``).

    ``s0`` and ``s1`` are squared sigmas. The pixels are compared with
    the threshold in float64, which is exact for a float32 threshold
    and keeps a float64 one (a deliberately wrong variant) unrounded:
    NumPy 1.x value-based casting would round a float64 scalar to
    float32 in a comparison with a float32 array. ``n_correction``
    replaces the per-pair ``n2`` in the correction and in ``dnorm`` by a
    global pixel count (another deliberately wrong variant).
    """
    threshold64 = np.float64(threshold)
    below0 = np.asarray(p0, dtype=np.float64) < threshold64
    below1 = np.asarray(p1, dtype=np.float64) < threshold64
    both = below0 & below1
    diff = p0[both] - p1[both]
    d2 = _sequential_sum(diff * diff)
    n2 = np.float32(np.count_nonzero(both))
    n = n2 if n_correction is None else np.float32(n_correction)
    d2 = np.float32(d2 - np.float32(n * np.float32(s0 + s1)))
    dnorm = np.float32(np.float32(s1 + s0) * np.sqrt(np.float32(np.float32(2.0) * n)))
    if dnorm > np.float32(1e-8):
        return np.float32(d2 / dnorm), n2
    return np.float32(np.float32(1e6) * n2), n2


def _weight(d: np.float32, lam: float, dthresh: float) -> np.float32:
    """Return the float32 weight of a distance as PyEBSDIndex's compiled
    ``nlpar_nb`` evaluates it: ``max(d - dthresh, 0)`` in float32, the
    exponent in float64, the result rounded to float32.
    """
    lam2 = 1.0 / (float(lam) * float(lam))
    x = np.maximum(np.float32(d) - np.float32(dthresh), np.float32(0.0))
    return np.float32(math.exp(-1.0 * float(x) * lam2))


def _transcribed_distances(
    data: np.ndarray,
    sigma2: np.ndarray,
    kept: np.ndarray,
    threshold: np.float32 | np.float64,
    radius: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the transcribed normalised distances, pixel counts and
    self-slot mask of a (n_rows, n_cols, n_pix) map in our slot layout,
    ``slot = wr * (2 rc + 1) + wc`` over the shifted-inward window.

    Self slots are ``-inf``; padding slots of an axis shorter than the
    window stay ``+inf`` with a count of 0.
    """
    nrows, ncols, _ = data.shape
    rr, rc = radius
    n_slots = (2 * rr + 1) * (2 * rc + 1)
    d = np.full((nrows, ncols, n_slots), np.inf, dtype=np.float32)
    n2 = np.zeros((nrows, ncols, n_slots), dtype=np.float32)
    is_self = np.zeros((nrows, ncols, n_slots), dtype=bool)
    values = data[:, :, kept]
    for j in range(nrows):
        r0, r1 = _ref_bounds(j, rr, nrows, shift=True)
        for i in range(ncols):
            c0, c1 = _ref_bounds(i, rc, ncols, shift=True)
            for jn in range(r0, r1):
                for i_n in range(c0, c1):
                    slot = (jn - r0) * (2 * rc + 1) + (i_n - c0)
                    if (jn, i_n) == (j, i):
                        d[j, i, slot] = -np.inf
                        is_self[j, i, slot] = True
                        continue
                    d[j, i, slot], n2[j, i, slot] = _pair_distance(
                        values[j, i],
                        values[jn, i_n],
                        sigma2[j, i],
                        sigma2[jn, i_n],
                        threshold,
                    )
    return d, n2, is_self


def _transcribed_average(
    data: np.ndarray,
    sigma: np.ndarray,
    radius: tuple[int, int],
    lam: float,
    dthresh: float,
    kept: np.ndarray,
    threshold: np.float32 | np.float64,
    policy: str = "shift",
    n_correction: int | None = None,
) -> np.ndarray:
    """Return the float32 average of a (n_rows, n_cols, n_pix) map, a
    transcription of PyEBSDIndex's ``nlpar_nb`` (``nlpar_cpu.py`` lines
    860-934) with a pluggable border policy.

    Policies: ``"shift"``, PyEBSDIndex's shifted-inward window;
    ``"clamp"``, the window clipped at the map edges and normalised over
    the surviving slots; ``"zero_extend"``, the centred window over a
    map padded with zero patterns of zero sigma, normalised by the sum
    of the window's weights (the edge rule of
    ``average_neighbour_patterns``). The self slot is ``-1e6`` (weight
    1), weights are summed sequentially in float32 in window order,
    divided by the sum, and the output accumulated in float32 in window
    order over all pixels.
    """
    nrows, ncols, n_pix = data.shape
    rr, rc = radius
    sigma = np.asarray(sigma, dtype=np.float32)
    sigma2 = (sigma * sigma).astype(np.float32)
    zero = np.zeros(n_pix, dtype=np.float32)
    out = np.zeros((nrows, ncols, n_pix), dtype=np.float32)
    for j in range(nrows):
        for i in range(ncols):
            if policy == "shift":
                rows = range(*_ref_bounds(j, rr, nrows, shift=True))
                cols = range(*_ref_bounds(i, rc, ncols, shift=True))
            elif policy == "clamp":
                rows = range(*_ref_bounds(j, rr, nrows, shift=False))
                cols = range(*_ref_bounds(i, rc, ncols, shift=False))
            else:
                rows = range(j - rr, j + rr + 1)
                cols = range(i - rc, i + rc + 1)
            weights = []
            neighbours = []
            for jn in rows:
                for i_n in cols:
                    inside = 0 <= jn < nrows and 0 <= i_n < ncols
                    p1 = data[jn, i_n] if inside else zero
                    if (jn, i_n) == (j, i):
                        d = np.float32(-1.0e6)
                    else:
                        s1 = sigma2[jn, i_n] if inside else np.float32(0.0)
                        d, _ = _pair_distance(
                            data[j, i, kept],
                            p1[kept],
                            sigma2[j, i],
                            s1,
                            threshold,
                            n_correction,
                        )
                    weights.append(_weight(d, lam, dthresh))
                    neighbours.append(p1)
            total = np.float32(0.0)
            for w in weights:
                total = np.float32(total + w)
            acc = np.zeros(n_pix, dtype=np.float32)
            for w, p1 in zip(weights, neighbours):
                acc = acc + p1 * np.float32(w / total)
            out[j, i] = acc
    return out


# --------------------------- Oracle call recipe ------------------------ #


def _oracle_sigma(
    nlpar,
    patterns: np.ndarray,
    signal_mask: np.ndarray | None = None,
    saturation_protect: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``sigma``, ``dout`` and ``nout`` of the compiled
    ``NLPAR.sigma_numba`` on a (n_rows, n_cols, h, w) map.

    ``dout`` and ``nout`` have the in-map neighbours in compact order
    (row outer, column inner) with trailing zeros; the self slot holds
    ``nout = h * w`` and a pair without a kept pixel ``nout = 1e-12``.
    """
    nrows, ncols, h, w = patterns.shape
    data2d = patterns.astype(np.float32).reshape(nrows * ncols, h * w)
    indices = _kept_indices(signal_mask, (h, w))
    return nlpar.sigma_numba(
        data2d,
        1,
        int(nrows),
        int(ncols),
        np.array([0, nrows], dtype=np.int64),
        np.array([0, ncols], dtype=np.int64),
        indices,
        bool(saturation_protect),
    )


def _oracle_average(
    nlpar,
    patterns: np.ndarray,
    sigma: np.ndarray,
    search_radius: int,
    lam: float,
    dthresh: float,
    signal_mask: np.ndarray | None = None,
    saturation_protect: bool = True,
    calclim: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return the output of the compiled ``NLPAR.nlpar_nb`` on a
    (n_rows, n_cols, h, w) map, reshaped to (n_rows, n_cols, h, w).

    ``lam`` is passed as a Python float and ``dthresh`` as float32. The
    output is the full block, zero outside ``calclim = [cstart, rstart,
    ncolcalc, nrowcalc]``; a fresh ``calclim`` array is built per call.
    The calling test is skipped where ``nlpar_nb`` does not compile
    (``NLPAR_NB_COMPILES``).
    """
    if not NLPAR_NB_COMPILES:
        pytest.skip(
            f"PyEBSDIndex's NLPAR.nlpar_nb does not compile under numba "
            f"{version('numba')}"
        )
    nrows, ncols, h, w = patterns.shape
    sr = int(search_radius)
    if min(nrows, ncols) < 2 * sr + 1:
        raise ValueError("every navigation axis must be >= 2 sr + 1 for nlpar_nb")
    data2d = patterns.astype(np.float32).reshape(nrows * ncols, h * w)
    indices = _kept_indices(signal_mask, (h, w))
    if calclim is None:
        calclim = (0, 0, ncols, nrows)
    dataout = nlpar.nlpar_nb(
        data2d,
        float(lam),
        sr,
        np.float32(dthresh),
        np.ascontiguousarray(sigma, dtype=np.float32),
        int(nrows),
        int(ncols),
        np.array(calclim, dtype=np.int64),
        indices,
        bool(saturation_protect),
        np.float32(0.0),
    )
    return dataout.reshape(nrows, ncols, h, w)


def _assert_sigma_pass_parity(
    sigma: np.ndarray,
    d: np.ndarray,
    n2: np.ndarray,
    valid: np.ndarray,
    n_kept: int,
    oracle: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Assert our sigma pass equals the compiled ``sigma_numba``.

    Sigma within ``SIGMA_PARITY_ULP``; our valid slots, in slot order,
    are PyEBSDIndex's compact slots in order; the normalised distances
    and counts of every neighbour slot (self slot excluded) with
    ``nout >= 1`` are bitwise equal to ``dout`` and ``nout``; slots where
    PyEBSDIndex holds ``nout = 1e-12`` have ``n2 == 0`` and ``d == +inf``
    on our side; our self slot holds ``n2 = n_kept`` and ``d = -inf``.
    """
    sigma_o, dout, nout = oracle
    _assert_within_ulp(sigma, sigma_o, SIGMA_PARITY_ULP)
    assert d.dtype == np.float32
    assert n2.dtype == np.float32
    assert valid.dtype == bool
    assert d.shape == n2.shape == valid.shape == sigma.shape + (9,)
    assert np.all(valid[..., 4])
    np.testing.assert_array_equal(np.count_nonzero(nout, axis=-1), valid.sum(axis=-1))
    position = np.where(valid, np.cumsum(valid, axis=-1) - 1, 0)
    dout_slots = np.take_along_axis(dout, position, axis=-1)
    nout_slots = np.take_along_axis(nout, position, axis=-1)
    neighbour = valid.copy()
    neighbour[..., 4] = False
    compared = neighbour & (nout_slots >= 1)
    empty = neighbour & ~(nout_slots >= 1)
    np.testing.assert_array_equal(d[compared], dout_slots[compared])
    np.testing.assert_array_equal(n2[compared], nout_slots[compared])
    assert np.all(n2[empty] == 0)
    assert np.all(np.isposinf(d[empty]))
    assert np.all(n2[..., 4] == np.float32(n_kept))
    assert np.all(np.isneginf(d[..., 4]))


# ------------------------------ Our routes ----------------------------- #


def _ours_sigma_pass(
    patterns: np.ndarray,
    signal_mask: np.ndarray | None = None,
    saturation_protect: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return our sigma, normalised 3 x 3 distances, counts and valid
    slots of a (n_rows, n_cols, h, w) map from the sigma kernel on the
    whole map and the NumPy normalisation.
    """
    _, _, h, w = patterns.shape
    data = _as_pixels(patterns)
    kept = _kept_indices(signal_mask, (h, w))
    max_value = np.float32(data.max())
    sigma, d2, n2, valid = _nlpar._nlpar_sigma_kernel(
        data, kept, max_value, bool(saturation_protect)
    )
    d = _nlpar._nlpar_normalized_distances(d2, n2, valid, sigma)
    return sigma, d, n2, valid


def _ours_method_average(
    patterns: np.ndarray,
    search_radius: int,
    lam: float,
    dthresh: float,
    sigma: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
    saturation_protect: bool = True,
) -> np.ndarray:
    """Return the float32 average of the eager EBSD method on a map."""
    s = kp.signals.EBSD(patterns)
    s_out = s.average_non_local_neighbour_patterns(
        search_radius=search_radius,
        lam=lam,
        dthresh=dthresh,
        sigma=sigma,
        signal_mask=signal_mask,
        saturation_protect=saturation_protect,
        dtype_out="float32",
        show_progressbar=False,
        inplace=False,
    )
    return s_out.data


# ---------------------------- Lambda helpers --------------------------- #


def _closed_form_lambda(c: float, target_weight: float) -> float:
    """Return the lambda for which a point with eight neighbours, all at
    normalised distance ``c``, keeps the target weight:
    ``1 / (1 + 8 exp(-c / lam^2)) = tw``.
    """
    return math.sqrt(-c / math.log((1.0 / target_weight - 1.0) / 8.0))


def _constructed_field(
    nav_shape: tuple[int, int] = (10, 10), c: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a constructed float32 field of 3 x 3 distances, every
    non-self slot at ``np.float32(c)`` and the self slot (4) at
    ``-inf``, with every slot valid.
    """
    d = np.full(nav_shape + (9,), np.float32(c), dtype=np.float32)
    d[..., 4] = -np.inf
    valid = np.ones(nav_shape + (9,), dtype=bool)
    return d, valid


def _set_non_self(d: np.ndarray, where: np.ndarray, c: float) -> None:
    """Set every non-self slot of the points selected by ``where`` to
    ``np.float32(c)``, in place.
    """
    for slot in (0, 1, 2, 3, 5, 6, 7, 8):
        d[where, slot] = np.float32(c)


def _in_map_slots(nav_shape: tuple[int, int]) -> np.ndarray:
    """Return whether every slot of the clipped 3 x 3 neighbourhood lies
    inside a map, bool of shape ``nav_shape + (9,)``; slot
    ``(dj + 1) * 3 + (di + 1)`` is the neighbour at offset (dj, di).
    """
    nrows, ncols = nav_shape
    rows = np.arange(nrows)[:, None, None]
    cols = np.arange(ncols)[None, :, None]
    slots = np.arange(9)[None, None, :]
    jn = rows + slots // 3 - 1
    i_n = cols + slots % 3 - 1
    return (jn >= 0) & (jn < nrows) & (i_n >= 0) & (i_n < ncols)


def _self_weight_terms(
    lam: np.ndarray,
    d: np.ndarray,
    valid: np.ndarray,
    dthresh: float,
    target_weight: float,
) -> np.ndarray:
    """Return the per-point terms ``|tw - 1 / S_i|`` of the lambda
    objective in the frozen form: float64 weights by promotion with the
    float64 ``lam``, invalid slots zeroed, nine-slot sum in slot order.
    """
    w = np.exp(-np.maximum(d - dthresh, np.float32(0.0)) / lam**2)
    w[~valid] = 0.0
    s = w.sum(axis=-1)
    return np.abs(target_weight - 1.0 / s)


def _loptfunc_local(
    lam: np.ndarray,
    d2: np.ndarray,
    mask: np.ndarray,
    tw: float,
    dthresh: float,
) -> float:
    """Return PyEBSDIndex's ``loptfunc`` (``nlpar_cpu.py``) transcribed
    with the two corrections of our objective and without its
    ``+ 1e-12`` on the weight sum: slots outside ``mask`` are excluded
    and the distance enters as ``max(d2 - dthresh, 0)``.
    """
    temp = np.maximum(d2 - dthresh, np.float32(0.0))
    dw = np.exp(-(temp) / lam**2)
    dw[~mask] = 0.0
    w = np.sum(dw, axis=2)
    return float(np.mean(np.abs(tw - 1.0 / w)))


def _loptfunc_pyebsdindex(
    lam: np.ndarray, d2: np.ndarray, tw: float, dthresh: float
) -> float:
    """Return PyEBSDIndex's ``loptfunc`` transcribed verbatim: every slot
    counted (out-of-map slots of ``dout`` hold 0.0, i.e. weight 1),
    ``max(d2, dthresh)`` and ``+ 1e-12`` on the weight sum.
    """
    temp = np.maximum(d2, dthresh)
    dw = np.exp(-(temp) / lam**2)
    w = np.sum(dw, axis=2) + 1e-12
    return float(np.mean(np.abs(tw - 1.0 / w)))


def _loptfunc_dthresh_floor(
    lam: np.ndarray,
    d2: np.ndarray,
    mask: np.ndarray,
    tw: float,
    dthresh: float,
) -> float:
    """Return the masked objective with PyEBSDIndex's ``max(d2,
    dthresh)`` form of the distance threshold in place of ours.
    """
    dw = np.exp(-np.maximum(d2, np.float32(dthresh)) / lam**2)
    dw[~mask] = 0.0
    w = np.sum(dw, axis=2)
    return float(np.mean(np.abs(tw - 1.0 / w)))


def _minimize_like_pyebsdindex(func: Callable[..., float], args: tuple) -> float:
    """Return the minimiser of ``func`` with PyEBSDIndex's optimiser
    settings: bounded Nelder-Mead from 1.0 in [1e-3, 10], ``fatol``
    1e-4.
    """
    result = minimize(
        func,
        x0=np.array([1.0]),
        args=args,
        method="Nelder-Mead",
        bounds=[(1e-3, 10.0)],
        options={"fatol": 1e-4},
    )
    return float(result.x[0])


def _compact_self_slot(nav_shape: tuple[int, int]) -> np.ndarray:
    """Return the position of every point's own slot in PyEBSDIndex's
    compact 3 x 3 slot order (in-map neighbours, row outer, column
    inner), int of shape ``nav_shape``.
    """
    nrows, ncols = nav_shape
    position = np.empty(nav_shape, dtype=np.int64)
    for j in range(nrows):
        for i in range(ncols):
            row_start, col_start = max(j - 1, 0), max(i - 1, 0)
            n_win_cols = min(i + 1, ncols - 1) - col_start + 1
            position[j, i] = (j - row_start) * n_win_cols + (i - col_start)
    return position


def _bound_messages(records: list[warnings.WarningMessage]) -> list[str]:
    """Return the messages of the recorded warnings about a lambda
    bound.
    """
    return [
        str(r.message)
        for r in records
        if "NLPAR lambda optimisation hit the" in str(r.message)
    ]


# ------------------------------- Fixtures ------------------------------ #


@pytest.fixture(scope="module")
def pyebsdindex_kernels(record_testsuite_property):
    """Return PyEBSDIndex's ``NLPAR`` class with both compiled kernels
    warmed on a (3, 3 | 4, 4) float32 map, recording the warm-up time.

    The time is recorded as the test-suite property
    ``pyebsdindex_jit_warmup_s``, since a module-scoped fixture cannot
    use the function-scoped ``record_property``. That property is
    written only into a junit-xml report of a run without xdist
    (``-n 0 --junit-xml=<file>``); on xdist workers, as in CI, it is
    recorded nowhere. Where ``nlpar_nb``
    does not compile (``NLPAR_NB_COMPILES``), only ``sigma_numba`` is
    warmed, so that the sigma oracle tests still run.
    """
    from pyebsdindex import nlpar_cpu

    nlpar = nlpar_cpu.NLPAR
    rng = np.random.default_rng(0)
    patterns = rng.uniform(20.0, 240.0, (3, 3, 4, 4)).astype(np.float32)
    tic = time.perf_counter()
    sigma, _, _ = _oracle_sigma(nlpar, patterns)
    if NLPAR_NB_COMPILES:
        _ = _oracle_average(nlpar, patterns, sigma, 1, 1.0, 0.0)
    record_testsuite_property("pyebsdindex_jit_warmup_s", time.perf_counter() - tic)
    return nlpar


class _NickelLargePatterns(dict):
    """The raw and the background-corrected uint8 patterns of
    ``nickel_ebsd_large``, (55, 75 | 60, 60), the corrected ones made
    on first access, so that a worker running only raw arms skips the
    background correction.
    """

    def __init__(self, s: kp.signals.EBSD):
        super().__init__(raw=s.data.copy())
        self._signal = s

    def __missing__(self, key: str) -> np.ndarray:
        if key != "corrected":
            raise KeyError(key)
        s = self._signal
        s.remove_static_background(show_progressbar=False)
        s.remove_dynamic_background(show_progressbar=False)
        self[key] = s.data.copy()
        return self[key]


@pytest.fixture(scope="module")
def nickel_large_patterns() -> dict[str, np.ndarray]:
    """Return the raw and the background-corrected uint8 patterns of
    ``nickel_ebsd_large``, (55, 75 | 60, 60).
    """
    s = kp.data.nickel_ebsd_large(allow_download=True)
    return _NickelLargePatterns(s)


# ------------------------------ Parameters ----------------------------- #


def _mask_id(masked: bool) -> str:
    return "automask" if masked else "nomask"


def _protect_id(protect: bool) -> str:
    return "protect" if protect else "noprotect"


SIGMA_ARMS = [
    pytest.param(masked, protect, id=f"{_mask_id(masked)}_{_protect_id(protect)}")
    for masked in (False, True)
    for protect in (True, False)
]

# The default suite runs an orthogonal subset of the averaging parity
# arms: every search radius, both lambdas, both thresholds, protection
# and mask on and off and both injections at least once. The first arm
# is the M22a/M22b killer of the Stage A bug injection (validation.md
# ledger entry 11). The full product runs weekly
AVERAGE_DEFAULT_ARMS = {
    (1, 0.7, 0.0, True, False, "injected"),
    (2, 2.5, 0.5, False, True, "end_to_end"),
    (3, 0.7, 0.5, True, True, "end_to_end"),
}

AVERAGE_ARMS = [
    pytest.param(
        sr,
        lam,
        dthresh,
        protect,
        masked,
        injection,
        id=(
            f"sr{sr}_lam{lam}_dthresh{dthresh:g}_{_protect_id(protect)}"
            f"_{_mask_id(masked)}_{injection}"
        ),
        marks=(
            ()
            if (sr, lam, dthresh, protect, masked, injection) in AVERAGE_DEFAULT_ARMS
            else pytest.mark.weekly
        ),
    )
    for sr in (1, 2, 3)
    for lam in (0.7, 2.5)
    for dthresh in (0.0, 0.5)
    for protect in (True, False)
    for masked in (False, True)
    for injection in ("injected", "end_to_end")
]

ORACLE_ARMS = [
    pytest.param("policy", id="policy"),
    pytest.param("oracle", id="oracle", marks=requires_pyebsdindex),
]


# ---------------------- Kernel test input builders --------------------- #


def _kernel_data(shape: tuple[int, int, int, int]) -> np.ndarray:
    """Return a random float32 (n_rows, n_cols, h * w) map from
    ``default_rng(0)`` with two pixels at the map maximum 250.
    """
    nrows, ncols, h, w = shape
    rng = np.random.default_rng(0)
    data = rng.uniform(20.0, 240.0, (nrows, ncols, h * w)).astype(np.float32)
    data[0, 0, 1] = np.float32(250.0)
    data[-1, -1, 2] = np.float32(250.0)
    return data


def _kernel_kept(n_pix: int, masked: bool) -> np.ndarray:
    """Return all pixel indices, or every second one."""
    step = 2 if masked else 1
    return np.arange(0, n_pix, step, dtype=np.int64)


def _kernel_sigma2(nav_shape: tuple[int, int]) -> np.ndarray:
    """Return random float32 squared sigmas of a navigation shape."""
    rng = np.random.default_rng(1)
    sigma = rng.uniform(30.0, 70.0, nav_shape).astype(np.float32)
    return (sigma * sigma).astype(np.float32)


def _kernel_weights(
    nav_shape: tuple[int, int],
    radius: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Return random float32 weights of a kept region in the slot layout
    of the search window, 0 on the padding slots of an axis shorter than
    the window.
    """
    rr, rc = radius
    row_start, row_stop, col_start, col_stop = bounds
    width_r, width_c = 2 * rr + 1, 2 * rc + 1
    rng = np.random.default_rng(2)
    weights = rng.random(
        (row_stop - row_start, col_stop - col_start, width_r * width_c)
    ).astype(np.float32)
    slots = np.arange(width_r * width_c)
    wr, wc = slots // width_c, slots % width_c
    in_map = (wr < min(width_r, nav_shape[0])) & (wc < min(width_c, nav_shape[1]))
    weights[..., ~in_map] = 0.0
    return weights


def _is_njit_decorator(node: ast.expr) -> bool:
    """Return whether a decorator node is ``njit``/``jit``, called or
    not, bare or as an attribute.
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id in ("njit", "jit")
    if isinstance(target, ast.Attribute):
        return target.attr in ("njit", "jit")
    return False


# -------------------------------- Tests -------------------------------- #


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # The flag test and the py_func tests are parametrised over the
        # literal list, so a kernel added later would escape both
        assert _njit_kernel_names(_nlpar) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _nlpar"
        )
        # The normalisation of the sigma pass is a NumPy driver
        assert callable(_nlpar._nlpar_normalized_distances)
        assert "_nlpar_normalized_distances" not in _njit_kernel_names(_nlpar)

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        # Dropping either option, or adding parallel or fastmath, leaves
        # every other test passing, so the Numba attributes are read
        kernel = getattr(_nlpar, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    def test_window_bounds_py_func_equals_the_compiled_kernel(self):
        kernel = _nlpar._window_bounds
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        for n in (1, 2, 3, 5, 8):
            for radius in (0, 1, 2, 3):
                for center in range(n):
                    for shift in (True, False):
                        compiled = kernel(center, radius, n, shift)
                        python = _py_func(kernel)(center, radius, n, shift)
                        assert tuple(int(v) for v in compiled) == tuple(
                            int(v) for v in python
                        ), (n, radius, center, shift)

    @pytest.mark.parametrize("shape", [(5, 6, 8, 8), (1, 7, 4, 4), (2, 2, 4, 4)])
    @pytest.mark.parametrize("protect", [True, False])
    @pytest.mark.parametrize("masked", [False, True])
    def test_nlpar_sigma_kernel_py_func_equals_the_compiled_kernel(
        self, shape, protect, masked
    ):
        kernel = _nlpar._nlpar_sigma_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        data = _kernel_data(shape)
        kept = _kernel_kept(data.shape[-1], masked)
        max_value = np.float32(data.max())
        compiled = kernel(data, kept, max_value, protect)
        python = _py_func(kernel)(data, kept, max_value, protect)
        for name, c, p in zip(("sigma", "d2", "n2", "valid"), compiled, python):
            assert c.dtype == p.dtype, name
            np.testing.assert_array_equal(c, p, err_msg=name)

    @pytest.mark.parametrize("shape, radius, bounds", DISTANCE_CASES)
    @pytest.mark.parametrize("protect", [True, False])
    @pytest.mark.parametrize("masked", [False, True])
    def test_nlpar_distances_kernel_py_func_equals_the_compiled_kernel(
        self, shape, radius, bounds, protect, masked
    ):
        # Both threshold branches and a strided set of kept pixels
        kernel = _nlpar._nlpar_distances_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        data = _kernel_data(shape)
        sigma2 = _kernel_sigma2(shape[:2])
        kept = _kernel_kept(data.shape[-1], masked)
        max_value = np.float32(data.max())
        args = (data, sigma2, kept, max_value, protect) + tuple(radius) + tuple(bounds)
        compiled = kernel(*args)
        python = _py_func(kernel)(*args)
        assert compiled.dtype == python.dtype == np.float32
        np.testing.assert_array_equal(compiled, python)

    @pytest.mark.parametrize("lam", [0.7, 2.5])
    @pytest.mark.parametrize("dthresh", [0.0, 0.5])
    def test_nlpar_weights_kernel_py_func_equals_the_compiled_kernel(
        self, lam, dthresh, exp_kernel_ulp
    ):
        kernel = _nlpar._nlpar_weights_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(0)
        d = rng.normal(0.0, 3.0, (5, 6, 25)).astype(np.float32)
        d[0, 0, 0] = -np.inf
        d[0, 0, 1] = np.inf
        d[1, 1] = 0.0
        compiled = kernel(d, float(lam), np.float32(dthresh))
        python = _py_func(kernel)(d, float(lam), np.float32(dthresh))
        assert compiled.dtype == python.dtype == np.float32
        assert compiled.shape == python.shape == d.shape
        # Read after both routes ran, so that the measurement is reported:
        # the weights are non-negative, so their int32 views are ordered
        compiled_bits = np.ascontiguousarray(compiled).view(np.int32).astype(np.int64)
        python_bits = np.ascontiguousarray(python).view(np.int32).astype(np.int64)
        measured = int(np.max(np.abs(compiled_bits - python_bits)))
        n_ulp = _require_placeholder(
            exp_kernel_ulp,
            f"EXP_KERNEL_ULP (measured {measured} ulp, compiled vs py_func)",
        )
        _assert_within_ulp(compiled, python, n_ulp)
        assert compiled[0, 0, 0] == 1.0
        assert compiled[0, 0, 1] == 0.0

    @pytest.mark.parametrize("shape, radius, bounds", DISTANCE_CASES)
    def test_nlpar_weighted_sum_kernel_py_func_equals_the_compiled_kernel(
        self, shape, radius, bounds
    ):
        kernel = _nlpar._nlpar_weighted_sum_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        data = _kernel_data(shape)
        weights = _kernel_weights(shape[:2], radius, bounds)
        args = (data, weights) + tuple(radius) + tuple(bounds)
        compiled = kernel(*args)
        python = _py_func(kernel)(*args)
        assert compiled.dtype == python.dtype == np.float32
        np.testing.assert_array_equal(compiled, python)

    def test_kernel_literals_are_float32(self):
        # No power operator inside an njit function: Numba types
        # float32 ** int32 as float32 and NumPy as float64, so the
        # py_func route would leave float32; squares are products
        tree = ast.parse(inspect.getsource(_nlpar))
        njit_functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(_is_njit_decorator(dec) for dec in node.decorator_list)
        ]
        assert sorted(fn.name for fn in njit_functions) == sorted(KERNEL_NAMES)
        for fn in njit_functions:
            powers = [
                node
                for node in ast.walk(fn)
                if isinstance(node, (ast.BinOp, ast.AugAssign))
                and isinstance(node.op, ast.Pow)
            ]
            assert not powers, f"{fn.name} uses the power operator"

        # Both routes return float32 on float32 input (a bare Python
        # float literal promotes the py_func result to float64 on old
        # NumPy)
        data = _kernel_data((5, 6, 8, 8))
        sigma2 = _kernel_sigma2((5, 6))
        kept = _kernel_kept(data.shape[-1], masked=False)
        max_value = np.float32(data.max())
        sigma_kernel = _nlpar._nlpar_sigma_kernel
        distances_kernel = _nlpar._nlpar_distances_kernel
        assert hasattr(sigma_kernel, "py_func")
        assert hasattr(distances_kernel, "py_func")
        for route in (lambda k: k, _py_func):
            sigma, d2, n2, valid = route(sigma_kernel)(data, kept, max_value, True)
            assert sigma.dtype == np.float32
            assert d2.dtype == np.float32
            assert n2.dtype == np.float32
            assert valid.dtype == bool
            d = route(distances_kernel)(
                data, sigma2, kept, max_value, True, 1, 2, 0, 5, 0, 6
            )
            assert d.dtype == np.float32

    @pytest.mark.parametrize(
        "n, r, center",
        [
            (n, r, center)
            for n in (1, 2, 3, 4, 5, 7, 9)
            for r in (0, 1, 2, 3)
            for center in range(n)
        ],
    )
    def test_window_bounds_matches_pyebsdindex_expressions_and_clamps(
        self, n, r, center
    ):
        shifted = tuple(int(v) for v in _nlpar._window_bounds(center, r, n, True))
        clipped = tuple(int(v) for v in _nlpar._window_bounds(center, r, n, False))

        # PyEBSDIndex's winstart/winend (nlpar_cpu.py lines 872-873,
        # 877-878), legal only when the axis holds the whole window
        if n >= 2 * r + 1:
            start = max(center - r, 0) - max(center + r - (n - 1), 0)
            stop = min(center + r, n - 1) + max(r - center, 0) + 1
            assert shifted == (start, stop)
        else:
            assert shifted == (0, n)
        assert shifted[1] - shifted[0] == min(2 * r + 1, n)
        assert 0 <= shifted[0] <= shifted[1] <= n

        # PyEBSDIndex's clipped nn_r_start/nn_r_end (lines 770-774)
        nn_start = center - r if center - r >= 0 else 0
        nn_end = (center + r if center + r < n else n - 1) + 1
        assert clipped == (nn_start, nn_end)
        assert (
            clipped[1] - clipped[0] == min(center + r, n - 1) - max(center - r, 0) + 1
        )

    def test_nlpar_code_never_names_pyebsdindex(self):
        # No import of PyEBSDIndex anywhere in the module, functions
        # included
        tree = ast.parse(inspect.getsource(_nlpar))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert not module.startswith("pyebsdindex"), module

        # The code of the three NLPAR methods, docstrings removed, never
        # names PyEBSDIndex (the docstrings may credit it)
        for name in (
            "average_non_local_neighbour_patterns",
            "get_nlpar_sigma",
            "get_nlpar_lambda",
        ):
            method = getattr(kp.signals.EBSD, name)
            source = textwrap.dedent(inspect.getsource(method))
            func = ast.parse(source).body[0]
            assert isinstance(func, ast.FunctionDef), name
            if ast.get_docstring(func) is not None:
                func.body = func.body[1:]
            assert "pyebsdindex" not in ast.unparse(func), name


@pytest.mark.skipif(
    dependency_version["pyebsdindex"] is None, reason="pyebsdindex is not installed"
)
class TestSigmaOracle:
    @staticmethod
    def _check_sigma_parity(nlpar, patterns, signal_mask, protect):
        oracle = _oracle_sigma(nlpar, patterns, signal_mask, protect)
        n_kept = _kept_indices(signal_mask, patterns.shape[2:]).size
        sigma, d, n2, valid = _ours_sigma_pass(patterns, signal_mask, protect)
        _assert_sigma_pass_parity(sigma, d, n2, valid, n_kept, oracle)

    @pytest.mark.parametrize("masked, protect", SIGMA_ARMS)
    def test_sigma_parity_compiled_identical_plus_gaussian(
        self, pyebsdindex_kernels, identical_plus_gaussian, circle_mask, masked, protect
    ):
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG, dtype=np.uint8)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_sigma_parity(pyebsdindex_kernels, patterns, mask, protect)

    @pytest.mark.parametrize("masked, protect", SIGMA_ARMS)
    def test_sigma_parity_compiled_two_grain(
        self, pyebsdindex_kernels, two_grain, circle_mask, masked, protect
    ):
        patterns = two_grain(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_sigma_parity(pyebsdindex_kernels, patterns, mask, protect)

    @pytest.mark.parametrize("masked, protect", SIGMA_ARMS)
    def test_sigma_parity_compiled_random_uniform_saturated(
        self,
        pyebsdindex_kernels,
        random_uniform_saturated,
        circle_mask,
        masked,
        protect,
    ):
        patterns = random_uniform_saturated(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_sigma_parity(pyebsdindex_kernels, patterns, mask, protect)

    @pytest.mark.parametrize("masked, protect", SIGMA_ARMS)
    def test_sigma_parity_compiled_exact_duplicates(
        self, pyebsdindex_kernels, exact_duplicates, circle_mask, masked, protect
    ):
        patterns = exact_duplicates(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_sigma_parity(pyebsdindex_kernels, patterns, mask, protect)

    @pytest.mark.parametrize(
        "variant", ["raw", pytest.param("corrected", marks=pytest.mark.weekly)]
    )
    def test_sigma_parity_compiled_nickel_ebsd_large(
        self, pyebsdindex_kernels, nickel_large_patterns, variant, record_property
    ):
        patterns = nickel_large_patterns[variant]
        _, _, h, w = patterns.shape
        tic = time.perf_counter()
        oracle = _oracle_sigma(pyebsdindex_kernels, patterns)
        record_property("pyebsdindex_sigma_numba_s", time.perf_counter() - tic)

        # The chunked eager route of the method: the map is split along
        # the navigation axes by get_dask_array
        s = kp.signals.EBSD(patterns)
        dask_array = get_dask_array(signal=s, chunk_bytes=8e6, rechunk=True)
        assert max(dask_array.numblocks[:2]) > 1

        kept = np.arange(h * w, dtype=np.int64)
        sigma, d2, n2, valid = _nlpar._nlpar_sigma(
            dask_array,
            mask_indices=kept,
            max_value=np.float32(patterns.max()),
            saturation_protect=True,
        )
        d = _nlpar._nlpar_normalized_distances(d2, n2, valid, sigma)
        _assert_sigma_pass_parity(sigma, d, n2, valid, kept.size, oracle)
        sigma_method = s.get_nlpar_sigma(show_progressbar=False)
        _assert_within_ulp(sigma_method, oracle[0], SIGMA_PARITY_ULP)

    def test_sigma_window_is_clipped_not_shifted(
        self, pyebsdindex_kernels, identical_plus_gaussian
    ):
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG, seed=0)
        data = _as_pixels(patterns)
        kept = _kept_indices(None, ORACLE_SIG)
        max_value = np.float32(data.max())
        border = np.ones(ORACLE_NAV, dtype=bool)
        border[1:-1, 1:-1] = False

        clipped = _sigma_reference(data, kept, max_value, True, window="clipped")
        shifted = _sigma_reference(data, kept, max_value, True, window="shifted")

        # The oracle clips (its sigma equals the clipped formula)
        sigma_o, _, _ = _oracle_sigma(pyebsdindex_kernels, patterns)
        np.testing.assert_array_equal(sigma_o, clipped)

        # A shifted window minimises over a superset of candidates at
        # every border pixel, strictly smaller on at least a quarter of
        # them (expected 5/8 of the corners and 3/8 of the edge pixels)
        assert np.all(shifted[border] <= clipped[border])
        strict = np.count_nonzero(shifted[border] < clipped[border])
        assert strict >= 0.25 * np.count_nonzero(border)

        sigma, _, _, _ = _ours_sigma_pass(patterns)
        np.testing.assert_array_equal(sigma[border], clipped[border])
        np.testing.assert_array_equal(sigma, sigma_o)

    def test_sigma_fallback_and_duplicates(self, pyebsdindex_kernels, exact_duplicates):
        kept = _kept_indices(None, ORACLE_SIG)

        # (1, 2) is an exact copy of (1, 1): the pair contributes
        # nothing, so the sigma of (1, 1) is the minimum over its seven
        # remaining neighbours
        patterns = exact_duplicates(ORACLE_NAV, ORACLE_SIG)
        data = _as_pixels(patterns)
        max_value = np.float32(data.max())
        values = data[:, :, kept]
        below = values.astype(np.float64) < _sigma_threshold(max_value, True)
        remaining = [
            (jn, i_n)
            for jn in range(3)
            for i_n in range(3)
            if (jn, i_n) not in [(1, 1), (1, 2)]
        ]
        expected_11 = _sigma_at(values, below, (1, 1), remaining)
        assert np.isfinite(expected_11)
        assert expected_11 > 0
        oracle = _oracle_sigma(pyebsdindex_kernels, patterns)
        assert oracle[0][1, 1] == expected_11

        # (3, 4) copied into its eight neighbours: no neighbour with a
        # non-zero distance, so the 1e12 fallback, while each neighbour
        # keeps a finite sigma from its outer neighbours
        block = exact_duplicates(ORACLE_NAV, ORACLE_SIG, block=True)
        oracle_block = _oracle_sigma(pyebsdindex_kernels, block)
        assert oracle_block[0][3, 4] == np.float32(1e12)
        around = np.ones((3, 3), dtype=bool)
        around[1, 1] = False
        assert np.all(oracle_block[0][2:5, 3:6][around] < np.float32(1e12))

        sigma, _, _, _ = _ours_sigma_pass(patterns)
        assert sigma[1, 1] == expected_11
        _assert_within_ulp(sigma, oracle[0], SIGMA_PARITY_ULP)

        sigma_block, _, _, _ = _ours_sigma_pass(block)
        assert sigma_block.dtype == np.float32
        assert sigma_block[3, 4] == np.float32(1e12)
        neighbours = sigma_block[2:5, 3:6][around]
        assert np.all(np.isfinite(neighbours))
        assert np.all(neighbours < np.float32(1e12))
        _assert_within_ulp(sigma_block, oracle_block[0], SIGMA_PARITY_ULP)

    @pytest.mark.parametrize("dtype", ["uint8", "uint16"])
    def test_sigma_saturation_threshold_constant(
        self, pyebsdindex_kernels, random_uniform_saturated, dtype
    ):
        patterns = random_uniform_saturated(ORACLE_NAV, ORACLE_SIG)
        if dtype == "uint16":
            # Scaled to the uint16 range (255 -> 65535), then values in
            # [65280, 65469] planted in the last six pixels of every
            # pattern: excluded by 0.9961 x 65535 = 65279.4, kept by
            # 0.999 x 65535 = 65469.5
            patterns = patterns.astype(np.uint16) * np.uint16(257)
            rng = np.random.default_rng(4)
            flat = patterns.reshape(ORACLE_NAV + (-1,))
            flat[..., -6:] = rng.integers(65280, 65470, ORACLE_NAV + (6,))
            assert patterns.max() == 65535
        data = _as_pixels(patterns)
        kept = _kept_indices(None, ORACLE_SIG)
        max_value = np.float32(data.max())

        ref = _sigma_reference(data, kept, max_value, True, factor=0.9961)
        ref_swapped = _sigma_reference(data, kept, max_value, True, factor=0.999)
        ref_off = _sigma_reference(data, kept, max_value, False)
        if dtype == "uint8":
            # Both constants exclude only the saturated 255s here
            np.testing.assert_array_equal(ref, ref_swapped)
        else:
            assert not np.array_equal(ref, ref_swapped)

        # The oracle's effective threshold is the float64 0.9961 x max,
        # and protection off excludes nothing (max + 1)
        np.testing.assert_array_equal(
            _oracle_sigma(pyebsdindex_kernels, patterns)[0], ref
        )
        np.testing.assert_array_equal(
            _oracle_sigma(pyebsdindex_kernels, patterns, saturation_protect=False)[0],
            ref_off,
        )

        sigma, _, _, _ = _ours_sigma_pass(patterns)
        np.testing.assert_array_equal(sigma, ref)
        if dtype == "uint16":
            assert not np.array_equal(sigma, ref_swapped)
        sigma_off, _, _, _ = _ours_sigma_pass(patterns, saturation_protect=False)
        np.testing.assert_array_equal(sigma_off, ref_off)

    def test_sigma_mask_is_forwarded(
        self, pyebsdindex_kernels, identical_plus_gaussian, circle_mask
    ):
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG, dtype=np.uint8)
        mask = circle_mask(ORACLE_SIG)
        sigma_o_full = _oracle_sigma(pyebsdindex_kernels, patterns)[0]
        sigma_o_masked = _oracle_sigma(pyebsdindex_kernels, patterns, mask)[0]
        assert not np.array_equal(sigma_o_masked, sigma_o_full)

        s = kp.signals.EBSD(patterns)
        sigma_full = s.get_nlpar_sigma(show_progressbar=False)
        sigma_masked = s.get_nlpar_sigma(signal_mask=mask, show_progressbar=False)
        sigma_flipped = s.get_nlpar_sigma(signal_mask=~mask, show_progressbar=False)

        # True means excluded: the mask reaches the sigma pass with
        # kikuchipy's polarity
        _assert_within_ulp(sigma_full, sigma_o_full, SIGMA_PARITY_ULP)
        _assert_within_ulp(sigma_masked, sigma_o_masked, SIGMA_PARITY_ULP)
        assert not np.array_equal(sigma_masked, sigma_full)
        assert not np.array_equal(sigma_flipped, sigma_masked)
        assert not np.array_equal(sigma_flipped, sigma_full)


class TestAveragingOracle:
    # Every test calling PyEBSDIndex carries its own skip marker, so that
    # the oracle-free small-map test runs without PyEBSDIndex too

    @staticmethod
    def _check_average_parity(
        nlpar, patterns, sr, lam, dthresh, protect, signal_mask, injection
    ):
        sigma_o, _, _ = _oracle_sigma(nlpar, patterns, signal_mask, protect)
        expected = _oracle_average(
            nlpar, patterns, sigma_o, sr, lam, dthresh, signal_mask, protect
        )
        # Injected: PyEBSDIndex's sigma fed to both, isolating the
        # averaging kernels. End to end: the method computes its own
        # sigma, bitwise PyEBSDIndex's by the sigma parity tests
        sigma_in = sigma_o if injection == "injected" else None
        ours = _ours_method_average(
            patterns, sr, lam, dthresh, sigma_in, signal_mask, protect
        )
        _assert_average_parity(ours, expected)

    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "sr, lam, dthresh, protect, masked, injection", AVERAGE_ARMS
    )
    def test_average_parity_compiled_identical_plus_gaussian(
        self,
        pyebsdindex_kernels,
        identical_plus_gaussian,
        sr,
        lam,
        dthresh,
        protect,
        masked,
        injection,
        circle_mask,
    ):
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG, dtype=np.uint8)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_average_parity(
            pyebsdindex_kernels, patterns, sr, lam, dthresh, protect, mask, injection
        )

    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "sr, lam, dthresh, protect, masked, injection", AVERAGE_ARMS
    )
    def test_average_parity_compiled_two_grain(
        self,
        pyebsdindex_kernels,
        two_grain,
        sr,
        lam,
        dthresh,
        protect,
        masked,
        injection,
        circle_mask,
    ):
        patterns = two_grain(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_average_parity(
            pyebsdindex_kernels, patterns, sr, lam, dthresh, protect, mask, injection
        )

    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "sr, lam, dthresh, protect, masked, injection", AVERAGE_ARMS
    )
    def test_average_parity_compiled_random_uniform_saturated(
        self,
        pyebsdindex_kernels,
        random_uniform_saturated,
        sr,
        lam,
        dthresh,
        protect,
        masked,
        injection,
        circle_mask,
    ):
        patterns = random_uniform_saturated(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_average_parity(
            pyebsdindex_kernels, patterns, sr, lam, dthresh, protect, mask, injection
        )

    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "sr, lam, dthresh, protect, masked, injection", AVERAGE_ARMS
    )
    def test_average_parity_compiled_exact_duplicates(
        self,
        pyebsdindex_kernels,
        exact_duplicates,
        sr,
        lam,
        dthresh,
        protect,
        masked,
        injection,
        circle_mask,
    ):
        patterns = exact_duplicates(ORACLE_NAV, ORACLE_SIG)
        mask = circle_mask(ORACLE_SIG) if masked else None
        self._check_average_parity(
            pyebsdindex_kernels, patterns, sr, lam, dthresh, protect, mask, injection
        )

    # One arm in the default suite, the other seven weekly
    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "variant, injection, lam",
        [
            pytest.param(
                variant,
                injection,
                lam,
                id=f"{variant}-{injection}-{lam}",
                marks=(
                    ()
                    if (variant, injection, lam) == ("raw", "end_to_end", 2.5)
                    else pytest.mark.weekly
                ),
            )
            for variant in ("raw", "corrected")
            for injection in ("injected", "end_to_end")
            for lam in (0.7, 2.5)
        ],
    )
    def test_average_parity_compiled_nickel_ebsd_large(
        self,
        pyebsdindex_kernels,
        nickel_large_patterns,
        variant,
        injection,
        lam,
        record_property,
    ):
        # The eager method splits this map along the navigation axes, so
        # this pins the chunked eager route end to end
        patterns = nickel_large_patterns[variant]
        sigma_o, _, _ = _oracle_sigma(pyebsdindex_kernels, patterns)
        tic = time.perf_counter()
        expected = _oracle_average(pyebsdindex_kernels, patterns, sigma_o, 3, lam, 0.0)
        record_property("pyebsdindex_nlpar_nb_s", time.perf_counter() - tic)
        sigma_in = sigma_o if injection == "injected" else None
        ours = _ours_method_average(patterns, 3, lam, 0.0, sigma=sigma_in)
        _assert_average_parity(ours, expected)

    @requires_pyebsdindex
    def test_border_band_differs_from_clamp_and_zero_extend(
        self, pyebsdindex_kernels, identical_plus_gaussian
    ):
        sr, lam, dthresh = 2, 1.0, 0.0
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG)
        data = _as_pixels(patterns)
        kept = _kept_indices(None, ORACLE_SIG)
        threshold = _average_threshold(data.max(), True)
        nrows, ncols = ORACLE_NAV
        interior = np.zeros(ORACLE_NAV, dtype=bool)
        interior[sr : nrows - sr, sr : ncols - sr] = True
        border = ~interior

        # The same injected sigma and pair weights under three border
        # policies
        sigma_o, _, _ = _oracle_sigma(pyebsdindex_kernels, patterns)
        expected = _oracle_average(
            pyebsdindex_kernels, patterns, sigma_o, sr, lam, dthresh
        )
        policies = {
            policy: _transcribed_average(
                data, sigma_o, (sr, sr), lam, dthresh, kept, threshold, policy
            ).reshape(patterns.shape)
            for policy in ("shift", "clamp", "zero_extend")
        }

        # The test-local transcription reproduces PyEBSDIndex's
        # shifted-inward window bitwise, so the alternatives share its
        # arithmetic and differ from it only in the border policy
        np.testing.assert_array_equal(policies["shift"], expected)
        worst_expected = {}
        for policy in ("clamp", "zero_extend"):
            alternative = policies[policy]
            np.testing.assert_array_equal(alternative[interior], expected[interior])
            worst_expected[policy] = np.abs(
                alternative[border].astype(np.float64) - expected[border]
            ).max()

        ours = _ours_method_average(patterns, sr, lam, dthresh, sigma=sigma_o)
        np.testing.assert_array_equal(ours, expected)
        worst_ours = {}
        for policy in ("clamp", "zero_extend"):
            alternative = policies[policy]
            np.testing.assert_array_equal(ours[interior], alternative[interior])
            worst_ours[policy] = np.abs(
                ours[border].astype(np.float64) - alternative[border]
            ).max()

        # Read after our route ran, so that the measurement is reported
        measured = min(min(worst_expected.values()), min(worst_ours.values()))
        min_diff = _require_placeholder(
            BORDER_BAND_MIN_DIFF,
            f"BORDER_BAND_MIN_DIFF (measured minimum {measured!r})",
        )
        for policy in ("clamp", "zero_extend"):
            assert worst_expected[policy] >= min_diff, (policy, worst_expected[policy])
            assert worst_ours[policy] >= min_diff, (policy, worst_ours[policy])

    @requires_pyebsdindex
    def test_saturation_arm_separates_per_pair_n2_from_global_n(
        self, pyebsdindex_kernels, random_uniform_saturated
    ):
        sr, lam, dthresh = 2, 1.0, 0.0
        patterns = random_uniform_saturated(ORACLE_NAV, ORACLE_SIG)
        data = _as_pixels(patterns)
        kept = _kept_indices(None, ORACLE_SIG)
        n_global = kept.size
        nlpar = pyebsdindex_kernels

        # Protection on: the per-pair count varies, and replacing it by
        # the global unmasked pixel count changes the output
        sigma_on, _, _ = _oracle_sigma(nlpar, patterns)
        expected_on = _oracle_average(nlpar, patterns, sigma_on, sr, lam, dthresh)
        threshold_on = _average_threshold(data.max(), True)
        sigma2_on = (sigma_on * sigma_on).astype(np.float32)
        _, n2_on, is_self = _transcribed_distances(
            data, sigma2_on, kept, threshold_on, (sr, sr)
        )
        assert np.unique(n2_on[~is_self]).size >= 2
        mutant_on = _transcribed_average(
            data,
            sigma_on,
            (sr, sr),
            lam,
            dthresh,
            kept,
            threshold_on,
            n_correction=n_global,
        ).reshape(patterns.shape)
        assert not np.array_equal(mutant_on, expected_on)

        # Protection off: every count is the global one, so the variant
        # is bitwise the oracle (the arm discriminates only under
        # saturation)
        sigma_off, _, _ = _oracle_sigma(nlpar, patterns, saturation_protect=False)
        expected_off = _oracle_average(
            nlpar, patterns, sigma_off, sr, lam, dthresh, saturation_protect=False
        )
        threshold_off = _average_threshold(data.max(), False)
        mutant_off = _transcribed_average(
            data,
            sigma_off,
            (sr, sr),
            lam,
            dthresh,
            kept,
            threshold_off,
            n_correction=n_global,
        ).reshape(patterns.shape)
        np.testing.assert_array_equal(mutant_off, expected_off)

        ours_on = _ours_method_average(patterns, sr, lam, dthresh, sigma=sigma_on)
        np.testing.assert_array_equal(ours_on, expected_on)
        ours_off = _ours_method_average(
            patterns, sr, lam, dthresh, sigma=sigma_off, saturation_protect=False
        )
        np.testing.assert_array_equal(ours_off, expected_off)

    @requires_pyebsdindex
    def test_calclim_from_block_info(
        self, pyebsdindex_kernels, identical_plus_gaussian
    ):
        # Rows chunked (3, 3, 3) at search radius 2: depth
        # max(2, 5 - 3) = 2, no rechunk; columns unchunked. Protection
        # off on both sides, so PyEBSDIndex's block-local maximum does
        # not matter
        sr, lam, dthresh = 2, 1.0, 0.0
        nlpar = pyebsdindex_kernels
        patterns = identical_plus_gaussian((9, 8), ORACLE_SIG)
        kept = _kept_indices(None, ORACLE_SIG)
        max_value = np.float32(patterns.max())
        sigma, _, _ = _oracle_sigma(nlpar, patterns, saturation_protect=False)
        whole = _oracle_average(
            nlpar, patterns, sigma, sr, lam, dthresh, saturation_protect=False
        )

        # Haloed blocks [0, 5), [1, 8), [4, 9) keep map rows [0, 3),
        # [3, 6), [6, 9): calclim = [0, rstart, 8, 3] with rstart 0, 2, 2
        blocks = [(0, 5, 0), (1, 8, 2), (4, 9, 2)]
        expected_cores = []
        for k, (start, stop, rstart) in enumerate(blocks):
            block_out = _oracle_average(
                nlpar,
                patterns[start:stop],
                sigma[start:stop],
                sr,
                lam,
                dthresh,
                saturation_protect=False,
                calclim=(0, rstart, 8, 3),
            )
            assert block_out.shape == (stop - start, 8) + ORACLE_SIG
            core = block_out[rstart : rstart + 3, 0:8]
            np.testing.assert_array_equal(core, whole[3 * k : 3 * k + 3])
            expected_cores.append(core)

        cores = []
        for k, (start, stop, _) in enumerate(blocks):
            core = _nlpar._nlpar_average_chunk(
                patterns[start:stop],
                sigma[start:stop, :, None, None],
                lam=lam,
                dthresh=np.float32(dthresh),
                radius=(sr, sr),
                mask_indices=kept,
                max_value=max_value,
                saturation_protect=False,
                depth=(2, 0),
                dtype_out=np.dtype(np.float32),
                block_info=_block_info((k, 0), (3, 1)),
            )
            # Only the kept region is returned, never the halo
            assert core.shape == (3, 8) + ORACLE_SIG
            assert core.dtype == np.float32
            np.testing.assert_array_equal(core, expected_cores[k])
            cores.append(core)
        np.testing.assert_array_equal(np.concatenate(cores, axis=0), whole)

        with pytest.raises(AssertionError):
            _nlpar._nlpar_average_chunk(
                patterns[0:5],
                sigma[0:5, :, None, None],
                lam=lam,
                dthresh=np.float32(dthresh),
                radius=(sr, sr),
                mask_indices=kept,
                max_value=max_value,
                saturation_protect=False,
                depth=(2, 0),
                dtype_out=np.dtype(np.float32),
                block_info=None,
            )

    @pytest.mark.parametrize("nav_shape", [(2, 2), (3, 5)])
    def test_small_map_padding_slots_are_inf(self, nav_shape):
        # PyEBSDIndex is not an oracle here: its window index would run
        # out of bounds on an axis shorter than 2 sr + 1
        r = 3
        nrows, ncols = nav_shape
        n_pix = 16
        rng = np.random.default_rng(0)
        data = rng.uniform(20.0, 240.0, nav_shape + (n_pix,)).astype(np.float32)
        sigma2 = np.full(nav_shape, 3600.0, dtype=np.float32)
        kept = np.arange(n_pix, dtype=np.int64)
        max_value = np.float32(data.max())
        width = 2 * r + 1
        len_r, len_c = min(width, nrows), min(width, ncols)
        slots = np.arange(width * width)
        wr, wc = slots // width, slots % width
        in_map = (wr < len_r) & (wc < len_c)

        d = _nlpar._nlpar_distances_kernel(
            data, sigma2, kept, max_value, True, r, r, 0, nrows, 0, ncols
        )
        assert d.shape == nav_shape + (width * width,)
        assert d.dtype == np.float32
        assert np.all(np.isposinf(d[..., ~in_map]))
        for j in range(nrows):
            for i in range(ncols):
                # The window is the whole map, so the point's own slot
                # sits at its map indices
                self_slot = j * width + i
                assert np.isneginf(d[j, i, self_slot])
                others = in_map.copy()
                others[self_slot] = False
                assert np.all(np.isfinite(d[j, i, others]))

        weights = _nlpar._nlpar_weights_kernel(d, 1e6, np.float32(0.0))
        assert np.all(weights[..., ~in_map] == 0.0)
        assert np.all(weights[..., in_map] == 1.0)
        out = _nlpar._nlpar_weighted_sum_kernel(data, weights, r, r, 0, nrows, 0, ncols)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        box_mean = data.astype(np.float64).mean(axis=(0, 1))
        n_window = nrows * ncols
        bound = n_window * np.spacing(np.abs(box_mean).astype(np.float32))
        for j in range(nrows):
            for i in range(ncols):
                assert np.all(np.abs(out[j, i].astype(np.float64) - box_mean) <= bound)

    @pytest.mark.weekly
    @requires_pyebsdindex
    def test_file_based_pyebsdindex_end_to_end(
        self, pyebsdindex_kernels, nickel_large_patterns, tmp_path, record_property
    ):
        # kikuchipy writes the corrected map to a kikuchipy h5ebsd file,
        # PyEBSDIndex reads it and runs its own file-based driver
        sr, lam, dthresh = 3, 2.5, 0.0
        patterns = nickel_large_patterns["corrected"]
        nrows, ncols, h, w = patterns.shape
        path_in = tmp_path / "nickel_ebsd_large_corrected.h5"
        kp.signals.EBSD(patterns.copy()).save(path_in)

        # Two mismatches between PyEBSDIndex's kikuchipy reader and the
        # current writer are patched in the temporary file, neither
        # touching the patterns: the reader compares the file version as
        # a string with "0.3.dev0", so a version such as "0.14.dev0"
        # sorts before it and the header (map and pattern shape) is not
        # read, and it reads EBSD/Header/grid_type, which the writer
        # keeps in the crystal map header only
        from pyebsdindex import ebsd_pattern

        with h5py.File(path_in, "r") as f:
            file_version = f["version"][()][0].decode("utf-8")
        if file_version < "0.3.dev0":
            unread = ebsd_pattern.get_pattern_file_obj([str(path_in), None])
            assert unread.nRows is None
            with h5py.File(path_in, "r+") as f:
                del f["version"]
                f.create_dataset("version", data=np.array([b"0.9.0"]))
        with h5py.File(path_in, "r+") as f:
            header = f["Scan 1/EBSD/Header"]
            if "grid_type" not in header:
                header.create_dataset("grid_type", data=np.array([b"square"]))
        pattern_file = ebsd_pattern.get_pattern_file_obj([str(path_in), None])
        assert (pattern_file.nRows, pattern_file.nCols) == (nrows, ncols)
        assert (pattern_file.patternH, pattern_file.patternW) == (h, w)

        nl = pyebsdindex_kernels(
            filename=str(path_in),
            lam=lam,
            searchradius=sr,
            dthresh=dthresh,
            automask=False,
        )
        tic = time.perf_counter()
        path_out = nl.calcnlpar(
            fileout=str(tmp_path / "nickel_ebsd_large_nlpar.h5"),
            chunksize=32e9,
            verbose=0,
        )
        record_property("pyebsdindex_file_driver_s", time.perf_counter() - tic)
        assert path_out

        # The patterns PyEBSDIndex reads back equal ours bitwise (the
        # h5ebsd round trip)
        pattern_file = nl.getinfileobj()
        read, _ = pattern_file.read_data(
            patStartCount=[[0, 0], [ncols, nrows]],
            convertToFloat=True,
            returnArrayOnly=True,
        )
        read = np.asarray(read, dtype=np.float32).reshape(nrows, ncols, h, w)
        np.testing.assert_array_equal(read, patterns.astype(np.float32))

        # Its driver's sigma, a minimum over overlapping tiles, is the
        # whole-map sigma: every point is interior to some tile
        sigma_driver = np.asarray(nl.sigma, dtype=np.float32)
        sigma_o, _, _ = _oracle_sigma(pyebsdindex_kernels, read)
        np.testing.assert_array_equal(sigma_driver, sigma_o)

        # The driver splits the map into tiles of at least two columns,
        # each with its own saturation maximum: the recorded quirk. The
        # file it writes truncates to integers, so the comparison is made
        # on the float32 kernel output over the whole map instead
        chunks = nl._calcchunks(
            [w, h], ncols, nrows, target_bytes=32e9, col_overlap=sr, row_overlap=sr
        )
        record_property("pyebsdindex_file_driver_tiles", int(chunks[0] * chunks[1]))
        expected = _oracle_average(
            pyebsdindex_kernels, read, sigma_driver, sr, lam, dthresh
        )
        ours = _ours_method_average(patterns, sr, lam, dthresh)
        _assert_average_parity(ours, expected)


class TestLambdaOracle:
    # The pyebsdindex-free tests drive the objective and the optimiser
    # on constructed fields of 3 x 3 normalised distances (self slot 4
    # at -inf), the oracle tests on PyEBSDIndex's own dout

    @pytest.mark.parametrize("target_weight", [0.5, 0.34, 0.25])
    @pytest.mark.parametrize("c", [2, 5, 12])
    def test_closed_form_lambda_on_constructed_distances(self, c, target_weight):
        rel = _require_placeholder(LAMBDA_CLOSED_FORM_REL, "LAMBDA_CLOSED_FORM_REL")
        expected = _closed_form_lambda(c, target_weight)
        assert round(expected, 4) == CLOSED_FORM_LAMBDA[(c, target_weight)]

        # Every point has eight valid neighbours at distance c, so the
        # closed form holds at every point, the border included
        d, valid = _constructed_field((10, 10), c)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lam = _nlpar._nlpar_optimize_lambda(
                d, valid, target_weight, np.float32(0.0)
            )
        assert _bound_messages(records) == []
        assert type(lam) is float
        assert abs(lam / expected - 1) <= rel

    def test_phantom_free_objective_excludes_missing_neighbours(self):
        # The border points of a (10, 10) map: 4 corners with 5
        # out-of-map slots, 32 edge points with 3, as in a real map; the
        # out-of-map slots carry d = 0.0, PyEBSDIndex's dout convention
        # for unvisited slots
        nav_shape = (10, 10)
        d, _ = _constructed_field(nav_shape, 2)
        valid = _in_map_slots(nav_shape)
        assert np.count_nonzero(~valid) == 4 * 5 + 32 * 3
        d_zero = d.copy()
        d_zero[~valid] = np.float32(0.0)
        d_inf = d.copy()
        d_inf[~valid] = np.inf
        tw, dthresh = 0.34, np.float32(0.0)

        for lam_value in (1.0, 2.0):
            lam = np.array([lam_value])
            value_zero = _nlpar._nlpar_lambda_objective(lam, d_zero, valid, dthresh, tw)
            value_inf = _nlpar._nlpar_lambda_objective(lam, d_inf, valid, dthresh, tw)
            # Exclusion through valid, not through the stored distance
            assert value_zero == value_inf
            # Counting the out-of-map slots with weight 1 (measured
            # 2026-10-04: 0.049 apart at lam 1.0, 0.042 at lam 2.0)
            phantom = _loptfunc_local(lam, d_zero, np.ones_like(valid), tw, dthresh)
            assert abs(value_zero - phantom) > 1e-3
            phantom_own = _nlpar._nlpar_lambda_objective(
                lam, d_zero, np.ones_like(valid), dthresh, tw
            )
            assert phantom_own == phantom

        # The minimiser still sits at the all-valid closed form
        rel = _require_placeholder(LAMBDA_CLOSED_FORM_REL, "LAMBDA_CLOSED_FORM_REL")
        lam = _nlpar._nlpar_optimize_lambda(d_zero, valid, tw, dthresh)
        assert abs(lam / _closed_form_lambda(2, tw) - 1) <= rel

    def test_mixed_c_field_distinguishes_mean_from_median(self):
        # 70 points with every non-self slot at c = 2 and 30 at c = 12,
        # all valid; no optimiser call
        d, valid = _constructed_field((10, 10), 2)
        far = np.zeros((10, 10), dtype=bool)
        far[7:] = True
        _set_non_self(d, far, 12)
        assert np.count_nonzero(far) == 30
        lam = np.array([1.5])
        tw, dthresh = 0.34, np.float32(0.0)

        terms = _self_weight_terms(lam, d, valid, dthresh, tw)
        mean, median = float(np.mean(terms)), float(np.median(terms))
        # Computed 2026-10-04: mean 0.2616, median 0.1068 (the c = 2
        # term alone)
        assert round(mean, 4) == 0.2616
        assert round(median, 4) == 0.1068

        value = _nlpar._nlpar_lambda_objective(lam, d, valid, dthresh, tw)
        assert type(value) is float
        assert value == mean
        assert abs(value - median) > 1e-3

    def test_lambdas_ascend_with_decreasing_target(self, identical_plus_gaussian):
        targets = (0.5, 0.34, 0.25)
        d, valid = _constructed_field((10, 10), 5)
        on_field = [
            _nlpar._nlpar_optimize_lambda(d, valid, tw, np.float32(0.0))
            for tw in targets
        ]
        assert on_field[0] < on_field[1] < on_field[2]

        # The distances of the sigma pass of a synthetic map
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG)
        _, d_map, _, valid_map = _ours_sigma_pass(patterns)
        on_map = [
            _nlpar._nlpar_optimize_lambda(d_map, valid_map, tw, np.float32(0.0))
            for tw in targets
        ]
        assert on_map[0] < on_map[1] < on_map[2]

    @pytest.mark.parametrize("arm", ["upper", "lower", "inside", "flat"])
    def test_bound_hit_warns(self, arm):
        tw, dthresh = 0.34, np.float32(0.0)
        if arm == "upper":
            # One neighbour at 1.0, seven with weight exactly 0: the self
            # weight 1 / (1 + exp(-1 / lam^2)) stays above 0.5 > 0.34 for
            # every lambda, so the fit runs into the upper bound
            # (measured 2026-10-04 with scipy 1.17.1: 10.0)
            d = np.full((10, 10, 9), np.inf, dtype=np.float32)
            d[..., 4] = -np.inf
            d[..., 0] = np.float32(1.0)
            valid = np.ones(d.shape, dtype=bool)
        elif arm == "lower":
            # Closed form sqrt(1e-7 / 1.41617) = 2.66e-4 < 1e-3 (measured
            # 2026-10-04: 1e-3)
            d, valid = _constructed_field((10, 10), 1e-7)
        elif arm == "inside":
            # Closed form 0.0084, inside the bounds: no warning
            d, valid = _constructed_field((10, 10), 1e-4)
        else:
            # Every non-self weight below 2^-53 relative to 1 near the
            # start: the objective is flat in float64 and the fit stays
            # at its start 1.0 without a warning (measured 2026-10-04)
            d, valid = _constructed_field((10, 10), 200)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lam = _nlpar._nlpar_optimize_lambda(d, valid, tw, dthresh)
        messages = _bound_messages(records)

        if arm in ("upper", "lower"):
            if arm == "upper":
                assert lam >= 9.9
            else:
                assert lam <= 1.01e-3
            assert messages == [
                f"NLPAR lambda optimisation hit the {arm} bound ({lam:.4f}); the "
                f"target weight {tw} is not supported by the data"
            ]
            assert all(
                issubclass(r.category, UserWarning)
                for r in records
                if "NLPAR lambda" in str(r.message)
            )
        elif arm == "inside":
            assert messages == []
            assert abs(lam / _closed_form_lambda(1e-4, tw) - 1) <= 1e-2
        else:
            assert messages == []
            assert abs(lam - 1.0) < 1e-6

    # Weekly: the rule needs a field of 1e6 points, so no cheaper arm
    # pins it
    @pytest.mark.weekly
    def test_stride_above_1e6_points(self):
        # c = 12 on the points of even row and even column, c = 2
        # elsewhere: the strided grid d[::2, ::2] is all c = 12, the full
        # one three quarters c = 2, and the two fits differ (measured
        # 2026-10-05: 2.9109 strided, 1.1884 full from x0 = 1.0)
        def grid(nav_shape):
            d, valid = _constructed_field(nav_shape, 2)
            even = np.zeros(nav_shape, dtype=bool)
            even[::2, ::2] = True
            _set_non_self(d, even, 12)
            return d, valid

        tw, dthresh = 0.34, np.float32(0.0)

        # >= 1e6 points: strided before the objective, so the call equals
        # the one on the strided field (< 1e6 points, not strided again).
        # Exactly 1e6 points, so a strict > threshold fails here
        d, valid = grid((1000, 1000))
        assert d.shape[0] * d.shape[1] == 1e6
        assert d[::2, ::2].shape[0] * d[::2, ::2].shape[1] < 1e6
        full = _nlpar._nlpar_optimize_lambda(d, valid, tw, dthresh)
        strided = _nlpar._nlpar_optimize_lambda(
            d[::2, ::2], valid[::2, ::2], tw, dthresh
        )
        assert full == strided
        del d, valid

        # < 1e6 points: never strided
        d, valid = grid((999, 1000))
        assert d.shape[0] * d.shape[1] < 1e6
        full = _nlpar._nlpar_optimize_lambda(d, valid, tw, dthresh)
        strided = _nlpar._nlpar_optimize_lambda(
            d[::2, ::2], valid[::2, ::2], tw, dthresh
        )
        assert full != strided

    @requires_pyebsdindex
    def test_objective_equals_test_local_loptfunc_on_pyebsdindex_dout(
        self, pyebsdindex_kernels, identical_plus_gaussian
    ):
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG)
        _, dout, nout = _oracle_sigma(pyebsdindex_kernels, patterns)
        tw = 0.34
        dthresh = np.float32(0.0)

        # The slots with a compared pair; the self slot replaced by
        # weight 1 (d = -inf) at its compact position
        mask = nout >= 1
        d = dout.copy()
        self_slot = _compact_self_slot(ORACLE_NAV)
        np.put_along_axis(d, self_slot[..., None], -np.inf, axis=-1)
        assert np.all(np.take_along_axis(mask, self_slot[..., None], axis=-1))
        assert np.count_nonzero(~mask) > 0

        for lam_value in (0.7, 1.0, 2.5):
            lam = np.array([lam_value])
            ours = _nlpar._nlpar_lambda_objective(lam, d, mask, dthresh, tw)
            assert ours == _loptfunc_local(lam, d, mask, tw, dthresh)
            # The recorded differences: PyEBSDIndex's own objective on
            # its raw dout counts the out-of-map slots as weight 1
            assert ours != _loptfunc_pyebsdindex(lam, dout, tw, dthresh)
            # ... and floors the distances at dthresh instead of
            # shifting them, which differs from dthresh > 0 on
            dthresh_half = np.float32(0.5)
            ours_half = _nlpar._nlpar_lambda_objective(lam, d, mask, dthresh_half, tw)
            assert ours_half == _loptfunc_local(lam, d, mask, tw, dthresh_half)
            assert ours_half != _loptfunc_dthresh_floor(lam, d, mask, tw, dthresh_half)

        # Deterministic Nelder-Mead from the same start and options
        expected = _minimize_like_pyebsdindex(_loptfunc_local, (d, mask, tw, dthresh))
        assert _nlpar._nlpar_optimize_lambda(d, mask, tw, dthresh) == expected

    @requires_pyebsdindex
    @pytest.mark.parametrize(
        "variant", ["raw", pytest.param("corrected", marks=pytest.mark.weekly)]
    )
    def test_phantom_deviation_is_measured_and_pinned(
        self, pyebsdindex_kernels, nickel_large_patterns, variant, record_property
    ):
        band = _require_placeholder(LAMBDA_PHANTOM_RATIO, "LAMBDA_PHANTOM_RATIO")
        patterns = nickel_large_patterns[variant]
        nrows, ncols = patterns.shape[:2]
        tw, dthresh = 0.34, np.float32(0.0)

        # PyEBSDIndex: its raw dout with the unmasked loptfunc; the
        # out-of-map slots hold 0.0 (4 x 5 + (2 x 53 + 2 x 73) x 3 = 776
        # of 37125 on the (55, 75) map)
        _, dout, nout = _oracle_sigma(pyebsdindex_kernels, patterns)
        assert (nrows, ncols) == (55, 75)
        assert np.count_nonzero(nout == 0) == 776
        assert np.all(dout[nout == 0] == 0.0)
        theirs = _minimize_like_pyebsdindex(_loptfunc_pyebsdindex, (dout, tw, dthresh))

        # Ours: the phantom-free objective on our own sigma pass
        _, d, _, valid = _ours_sigma_pass(patterns)
        ours = _nlpar._nlpar_optimize_lambda(d, valid, tw, dthresh)

        ratio = ours / theirs
        record_property(f"lambda_ours_{variant}", ours)
        record_property(f"lambda_pyebsdindex_{variant}", theirs)
        record_property(f"lambda_phantom_ratio_{variant}", ratio)
        assert band[0] <= ratio <= band[1]


class TestDepthAndHalo:
    @pytest.mark.parametrize("nav_chunks", DRIVER_CHUNKINGS, ids=str)
    def test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel(
        self, counting_spy, random_uniform_saturated, nav_chunks
    ):
        # Saturated pixels in the first (5, 8) block only, so a
        # per-chunk maximum differs from the global one
        x = random_uniform_saturated((10, 16), (16, 16), one_block_only=True)
        multi = _dask_map(x, nav_chunks)
        assert multi.numblocks[0] > 1
        assert multi.numblocks[1] > 1
        kept = np.arange(16 * 16, dtype=np.int64)
        max_value = np.float32(x.max())
        assert max_value == 255

        # One wrapper call per block of the unchanged chunks: the route
        # is overlap + map_blocks, not one kernel call on the computed
        # array, which would give the same values
        calls = counting_spy(_nlpar, "_nlpar_sigma_chunk")
        with dask.config.set(scheduler="synchronous"):
            result = _nlpar._nlpar_sigma(
                multi, mask_indices=kept, max_value=max_value, saturation_protect=True
            )
        assert len(calls) == _n_nav_blocks(multi.chunks)
        expected = _nlpar._nlpar_sigma_kernel(_as_pixels(x), kept, max_value, True)
        for name, r, e in zip(("sigma", "d2", "n2", "valid"), result, expected):
            assert r.dtype == e.dtype, name
            np.testing.assert_array_equal(r, e, err_msg=name)

    @pytest.mark.parametrize("nav_chunks", DRIVER_CHUNKINGS, ids=str)
    def test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk(
        self, monkeypatch, counting_spy, random_uniform_saturated, nav_chunks
    ):
        # At sr 3 on the (3, 3, 4) rows the depth is max(3, 7 - 3) = 4
        # and the rows are rechunked to (6, 4); on the (1, 1) chunking
        # the depth is 6 on both axes and the columns become (6, 10)
        x = random_uniform_saturated((10, 16), (16, 16), one_block_only=True)
        multi = _dask_map(x, nav_chunks)
        single = da.from_array(x, chunks=x.shape)
        assert multi.numblocks[0] > 1
        assert multi.numblocks[1] > 1
        kept = np.arange(16 * 16, dtype=np.int64)
        max_value = np.float32(x.max())
        sigma = _sigma_reference(_as_pixels(x), kept, max_value, True)
        overlap_chunks = _record_overlap_chunks(monkeypatch)
        calls = counting_spy(_nlpar, "_nlpar_average_chunk")
        for sr in (1, 3):
            kwargs = {
                "lam": 1.0,
                "dthresh": np.float32(0.0),
                "radius": (sr, sr),
                "mask_indices": kept,
                "max_value": max_value,
                "saturation_protect": True,
                "dtype_out": np.dtype(np.float32),
            }
            _, chunks_out = _ref_depth_chunks(multi.chunks, (sr, sr))
            if sr == 1:
                # Test power: every chunking stays multi-block at sr 1
                assert _n_nav_blocks(chunks_out) > 1
            with dask.config.set(scheduler="synchronous"):
                expected = _nlpar._nlpar_average(single, sigma, **kwargs).compute()

                # The explicit rechunk precedes overlap: both operands
                # reach it in the post-rechunk navigation chunks, and
                # the result is chunked as them, one wrapper call per
                # block
                overlap_chunks.clear()
                calls.clear()
                lazy = _nlpar._nlpar_average(multi, sigma, **kwargs)
                if _n_nav_blocks(chunks_out) > 1:
                    assert overlap_chunks, "overlap was not called"
                for chunks in overlap_chunks:
                    assert chunks[:2] == chunks_out[:2], (sr, chunks)
                assert lazy.chunks == chunks_out
                assert lazy.dtype == np.float32
                result = lazy.compute()
            assert len(calls) == _n_nav_blocks(chunks_out)
            assert result.shape == x.shape
            assert result.dtype == np.float32
            np.testing.assert_array_equal(result, expected, err_msg=f"sr={sr}")

    @pytest.mark.parametrize(
        "nav_shape, nav_chunks, sr",
        [
            pytest.param((5, 16), ((2, 3), (8, 8)), 2, id="rows_5_2_3_r2"),
            pytest.param((3, 16), ((1, 2), (8, 8)), 3, id="rows_3_1_2_r3"),
        ],
    )
    def test_pass_two_driver_rechunks_an_axis_no_longer_than_the_window(
        self, monkeypatch, random_uniform_saturated, nav_shape, nav_chunks, sr
    ):
        # A chunked axis of length n <= 2 r + 1 is rechunked to one
        # chunk with depth 0. Dask's overlap rechunks an axis thinner
        # than a non-zero depth by itself but leaves a depth-0 axis as
        # it is, so without the explicit rechunk every 1-, 2- or 3-row
        # block would average over its own rows only
        x = random_uniform_saturated(nav_shape, (16, 16), one_block_only=True)
        multi = _dask_map(x, nav_chunks)
        single = da.from_array(x, chunks=x.shape)
        kept = np.arange(16 * 16, dtype=np.int64)
        max_value = np.float32(x.max())
        sigma = _sigma_reference(_as_pixels(x), kept, max_value, True)
        depth, chunks_out = _ref_depth_chunks(multi.chunks, (sr, sr))
        assert multi.numblocks[0] > 1
        assert depth == (0, sr)
        assert chunks_out[:2] == ((nav_shape[0],), (8, 8))
        kwargs = {
            "lam": 1.0,
            "dthresh": np.float32(0.0),
            "radius": (sr, sr),
            "mask_indices": kept,
            "max_value": max_value,
            "saturation_protect": True,
            "dtype_out": np.dtype(np.float32),
        }
        with dask.config.set(scheduler="synchronous"):
            expected = _nlpar._nlpar_average(single, sigma, **kwargs).compute()
            overlap_chunks = _record_overlap_chunks(monkeypatch)
            lazy = _nlpar._nlpar_average(multi, sigma, **kwargs)
            assert overlap_chunks, "overlap was not called"
            for chunks in overlap_chunks:
                assert chunks[:2] == chunks_out[:2], chunks
            assert lazy.chunks == chunks_out
            result = lazy.compute()
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)

    def test_drivers_take_split_signal_axes_and_a_dask_maximum(
        self, random_uniform_saturated
    ):
        # A lazy signal keeps its signal chunks, so the drivers can get
        # signal axes split into several chunks: both passes rechunk
        # them to one chunk and equal the single-chunk route bitwise.
        # The global maximum of a chunked map is one Dask reduction,
        # the same float32 as the maximum of the in-memory map
        x = random_uniform_saturated((10, 16), (16, 16), one_block_only=True)
        split = da.from_array(x, chunks=(5, 8, 8, 4))
        single = da.from_array(x, chunks=x.shape)
        assert split.numblocks == (2, 2, 2, 4)

        max_value = _nlpar._nlpar_saturation_max(split)
        assert type(max_value) is np.float32
        assert max_value == _nlpar._nlpar_saturation_max(x) == 255

        kept = np.arange(16 * 16, dtype=np.int64)
        kwargs = {
            "mask_indices": kept,
            "max_value": max_value,
            "saturation_protect": True,
        }
        average_kwargs = {
            "lam": 1.0,
            "dthresh": np.float32(0.0),
            "radius": (2, 2),
            "dtype_out": np.dtype(np.float32),
            **kwargs,
        }
        with dask.config.set(scheduler="synchronous"):
            result = _nlpar._nlpar_sigma(split, **kwargs)
            expected = _nlpar._nlpar_sigma(single, **kwargs)
            for name, r, e in zip(("sigma", "d2", "n2", "valid"), result, expected):
                assert r.dtype == e.dtype, name
                np.testing.assert_array_equal(r, e, err_msg=name)

            lazy = _nlpar._nlpar_average(split, result[0], **average_kwargs)
            assert lazy.chunks == ((5, 5), (8, 8), (16,), (16,))
            averaged = lazy.compute()
            expected_average = _nlpar._nlpar_average(
                single, expected[0], **average_kwargs
            ).compute()
        assert averaged.dtype == np.float32
        np.testing.assert_array_equal(averaged, expected_average)

    def test_sigma_chunk_returns_the_packed_core(self, random_uniform_saturated):
        # Rows chunked (3, 3) with depth 1, columns unchunked: the haloed
        # blocks [0, 4) and [2, 6) keep map rows [0, 3) and [3, 6). The
        # clipped 3 x 3 window of a core row lies inside its haloed
        # block, so each core equals the whole-map kernel on its rows
        x = random_uniform_saturated((6, 5), (4, 4))
        kept = np.arange(16, dtype=np.int64)
        max_value = np.float32(x.max())
        sigma, d2, n2, valid = _nlpar._nlpar_sigma_kernel(
            _as_pixels(x), kept, max_value, True
        )

        packed_cores = []
        for k, (start, stop) in enumerate([(0, 4), (2, 6)]):
            packed = _nlpar._nlpar_sigma_chunk(
                x[start:stop],
                mask_indices=kept,
                max_value=max_value,
                saturation_protect=True,
                depth=(1, 0),
                block_info=_block_info((k, 0), (2, 1)),
            )
            # Only the core, packed: d2, n2, valid as 1.0/0.0, and sigma
            # in slot 0 of the last plane
            assert packed.shape == (3, 5, 4, 9)
            assert packed.dtype == np.float32
            rows = slice(3 * k, 3 * k + 3)
            np.testing.assert_array_equal(packed[..., 0, :], d2[rows])
            np.testing.assert_array_equal(packed[..., 1, :], n2[rows])
            np.testing.assert_array_equal(
                packed[..., 2, :], valid[rows].astype(np.float32)
            )
            np.testing.assert_array_equal(packed[..., 3, 0], sigma[rows])
            assert np.all(packed[..., 3, 1:] == 0)
            packed_cores.append(packed)

        unpacked = _nlpar._nlpar_unpack_sigma_pass(np.concatenate(packed_cores))
        for name, u, e in zip(
            ("sigma", "d2", "n2", "valid"), unpacked, (sigma, d2, n2, valid)
        ):
            assert u.dtype == e.dtype, name
            np.testing.assert_array_equal(u, e, err_msg=name)

        with pytest.raises(AssertionError):
            _nlpar._nlpar_sigma_chunk(
                x[0:4],
                mask_indices=kept,
                max_value=max_value,
                saturation_protect=True,
                depth=(1, 0),
                block_info=None,
            )

    @pytest.mark.parametrize(
        "nav_shape, row_chunks, col_chunks, radius, depth, rows_out, cols_out",
        [
            # A shifted window of the last three rows reaches 2 r = 6
            # rows inward, past a 3-row chunk plus a 3-row halo
            pytest.param(
                (55, 75),
                (26, 26, 3),
                (75,),
                (3, 3),
                (4, 0),
                (26, 25, 4),
                (75,),
                id="rows_26_26_3_r3",
            ),
            pytest.param(
                (55, 75),
                (26, 26, 3),
                (75,),
                (4, 4),
                (6, 0),
                (26, 23, 6),
                (75,),
                id="rows_26_26_3_r4",
            ),
            pytest.param(
                (55, 75),
                (47, 8),
                (75,),
                (3, 3),
                (3, 0),
                (47, 8),
                (75,),
                id="rows_47_8_r3",
            ),
            pytest.param(
                (55, 75),
                (47, 8),
                (75,),
                (4, 4),
                (4, 0),
                (47, 8),
                (75,),
                id="rows_47_8_r4",
            ),
            pytest.param(
                (75, 55),
                (75,),
                (26, 26, 3),
                (3, 3),
                (0, 4),
                (75,),
                (26, 25, 4),
                id="cols_26_26_3_r3",
            ),
            pytest.param(
                (55, 55),
                (26, 26, 3),
                (26, 26, 3),
                (3, 4),
                (4, 6),
                (26, 25, 4),
                (26, 23, 6),
                id="per_axis_radius_3_4",
            ),
            pytest.param(
                (10, 16),
                (3, 3, 4),
                (7, 7, 2),
                (3, 3),
                (4, 5),
                (6, 4),
                (7, 9),
                id="rows_3_3_4_cols_7_7_2_r3",
            ),
            pytest.param(
                (5, 16),
                (2, 3),
                (16,),
                (2, 2),
                (0, 0),
                (5,),
                (16,),
                id="rows_5_r2_single_chunk",
            ),
            pytest.param(
                (3, 16),
                (1, 2),
                (16,),
                (3, 3),
                (0, 0),
                (3,),
                (16,),
                id="rows_3_r3_single_chunk",
            ),
            pytest.param(
                (55, 75),
                (55,),
                (75,),
                (3, 3),
                (0, 0),
                (55,),
                (75,),
                id="unchunked",
            ),
        ],
    )
    def test_depth_helper(
        self, nav_shape, row_chunks, col_chunks, radius, depth, rows_out, cols_out
    ):
        # Self-check of the table against the rule: one chunk -> 0;
        # 2 r + 1 >= n -> one chunk and 0; else
        # max(r, 2 r + 1 - min(c[0], c[-1])) and the minimum chunk size
        for n, chunks, r, d, out in zip(
            nav_shape, (row_chunks, col_chunks), radius, depth, (rows_out, cols_out)
        ):
            assert sum(chunks) == n
            if len(chunks) == 1:
                assert d == 0
                assert out == chunks
            elif 2 * r + 1 >= n:
                assert d == 0
                assert out == (n,)
            else:
                assert d == max(r, 2 * r + 1 - min(chunks[0], chunks[-1]))
                assert tuple(ensure_minimum_chunksize(d, chunks)) == out

        sig_chunks = ((8,), (8,))
        chunks = (row_chunks, col_chunks) + sig_chunks
        depth_out, chunks_out = _nlpar._nlpar_depth(chunks, radius)
        assert dict(depth_out) == {0: depth[0], 1: depth[1], 2: 0, 3: 0}
        assert tuple(tuple(c) for c in chunks_out) == (rows_out, cols_out) + sig_chunks


class TestPolicyOracles:
    @pytest.mark.parametrize("arm", ORACLE_ARMS)
    def test_n2_zero_pair_gets_weight_zero(self, arm, request):
        # Pattern (0, 0) is saturated at every pixel (all values equal
        # the map maximum), so with protection on it shares no
        # comparable pixel with any other pattern
        data = _hand_built_map(seed=21)
        data[0, 0] = np.float32(255.0)
        max_value = np.float32(data.max())
        assert max_value == 255
        kept = np.arange(16, dtype=np.int64)
        sigma = np.full((3, 3), 8.0, dtype=np.float32)
        sigma2 = (sigma * sigma).astype(np.float32)
        others = np.ones((3, 3), dtype=bool)
        others[0, 0] = False

        if arm == "oracle":
            # PyEBSDIndex gives the pair d = 1e6 x 0 = 0 and so weight 1,
            # which its transcription reproduces
            nlpar = request.getfixturevalue("pyebsdindex_kernels")
            expected = _oracle_average(
                nlpar, data.reshape(3, 3, 4, 4), sigma, 1, 1.0, 0.0
            )
            transcribed = _transcribed_average(
                data, sigma, (1, 1), 1.0, 0.0, kept, _average_threshold(max_value, True)
            )
            np.testing.assert_array_equal(transcribed.reshape(3, 3, 4, 4), expected)

        # Slot 0 of every window of the 3 x 3 map is pattern (0, 0); the
        # pure Python kernel takes the same branch, bitwise
        kernel = _nlpar._nlpar_distances_kernel
        args = (data, sigma2, kept, max_value, True, 1, 1, 0, 3, 0, 3)
        d = kernel(*args)
        np.testing.assert_array_equal(_py_func(kernel)(*args), d)
        assert np.all(np.isposinf(d[..., 0][others]))
        weights = _nlpar._nlpar_weights_kernel(d, 1.0, np.float32(0.0))
        assert np.all(weights[..., 0][others] == 0.0)

        # The other patterns are unaffected by the saturated one
        out = _nlpar._nlpar_weighted_sum_kernel(data, weights, 1, 1, 0, 3, 0, 3)
        replaced = data.copy()
        replaced[0, 0] = data[2, 2]
        out_replaced = _nlpar._nlpar_weighted_sum_kernel(
            replaced, weights, 1, 1, 0, 3, 0, 3
        )
        np.testing.assert_array_equal(out[others], out_replaced[others])

        if arm == "oracle":
            assert not np.array_equal(out.reshape(3, 3, 4, 4)[1, 1], expected[1, 1])

    def test_tiny_dnorm_branch_gives_1e6_n2(self):
        # sigma 1e-6 everywhere: dnorm = 2e-12 sqrt(2 n2) < 1e-8, so every
        # non-self pair takes d = 1e6 n2
        rng = np.random.default_rng(11)
        data = rng.uniform(20.0, 240.0, (3, 3, 16)).astype(np.float32)
        kept = np.arange(16, dtype=np.int64)
        max_value = np.float32(data.max())
        sigma2 = np.full((3, 3), 1e-12, dtype=np.float32)
        threshold = _average_threshold(max_value, True)

        d_ref, n2_ref, is_self = _transcribed_distances(
            data, sigma2, kept, threshold, (1, 1)
        )
        assert np.all(n2_ref[~is_self] > 0)
        np.testing.assert_array_equal(
            d_ref[~is_self], np.float32(1e6) * n2_ref[~is_self]
        )

        # The compiled kernel and its pure Python function alike
        kernel = _nlpar._nlpar_distances_kernel
        for route in (kernel, _py_func(kernel)):
            d = route(data, sigma2, kept, max_value, True, 1, 1, 0, 3, 0, 3)
            assert d.dtype == np.float32
            np.testing.assert_array_equal(d[~is_self], d_ref[~is_self])
            np.testing.assert_array_equal(
                d[~is_self], np.float32(1e6) * n2_ref[~is_self]
            )
            assert np.all(np.isneginf(d[is_self]))

    @pytest.mark.parametrize("arm", ORACLE_ARMS)
    def test_duplicate_neighbour_is_skipped_in_sigma_but_averaged(self, arm, request):
        rng = np.random.default_rng(31)
        kept = np.arange(16, dtype=np.int64)

        # Integer data with (1, 2) an exact copy of (1, 1)
        ints = rng.integers(20, 240, (3, 3, 16)).astype(np.float32)
        ints[1, 2] = ints[1, 1]
        max_ints = np.float32(ints.max())
        ref_ints = _sigma_reference(ints, kept, max_ints, True)

        # Float data in [0, 1] with (1, 2) a near copy of (1, 1),
        # 0 < d2 < 1e-3: our d2 > 0 guard keeps the pair, PyEBSDIndex's
        # d2 >= 1e-3 skips it
        floats = rng.random((3, 3, 16)).astype(np.float32)
        floats[1, 1, 0] = np.float32(0.5)
        floats[1, 2] = floats[1, 1]
        floats[1, 2, 0] = np.float32(0.51)
        diff = floats[1, 1] - floats[1, 2]
        d2_near = _sequential_sum(diff * diff)
        assert 0 < d2_near < 1e-3
        max_floats = np.float32(floats.max())
        ref_floats = _sigma_reference(floats, kept, max_floats, True)
        ref_floats_pyebsdindex = _sigma_reference(
            floats, kept, max_floats, True, duplicate_guard="pyebsdindex"
        )
        assert ref_floats[1, 1] != ref_floats_pyebsdindex[1, 1]

        if arm == "oracle":
            nlpar = request.getfixturevalue("pyebsdindex_kernels")
            sigma_o_ints = _oracle_sigma(nlpar, ints.reshape(3, 3, 4, 4))[0]
            np.testing.assert_array_equal(sigma_o_ints, ref_ints)
            expected_ints = _oracle_average(
                nlpar, ints.reshape(3, 3, 4, 4), sigma_o_ints, 1, 1.0, 0.0
            )
            sigma_o_floats = _oracle_sigma(nlpar, floats.reshape(3, 3, 4, 4))[0]
            np.testing.assert_array_equal(sigma_o_floats, ref_floats_pyebsdindex)

        # The exact copy does not set the sigma of (1, 1)
        sigma_ints, _, _, _ = _nlpar._nlpar_sigma_kernel(ints, kept, max_ints, True)
        np.testing.assert_array_equal(sigma_ints, ref_ints)
        assert np.isfinite(sigma_ints[1, 1])
        assert sigma_ints[1, 1] > 0

        # ... yet gets weight exactly 1 in the average: d < 0 and
        # max(d - 0, 0) = 0. Every window of the 3 x 3 map is the whole
        # map, so (1, 2) is slot 5 for (1, 1) and (1, 1) slot 4 for (1, 2)
        sigma2 = (sigma_ints * sigma_ints).astype(np.float32)
        d = _nlpar._nlpar_distances_kernel(
            ints, sigma2, kept, max_ints, True, 1, 1, 0, 3, 0, 3
        )
        weights = _nlpar._nlpar_weights_kernel(d, 1.0, np.float32(0.0))
        assert d[1, 1, 5] < 0
        assert d[1, 2, 4] < 0
        assert weights[1, 1, 5] == 1.0
        assert weights[1, 2, 4] == 1.0

        # On float data the near copy sets our sigma
        sigma_floats, _, _, _ = _nlpar._nlpar_sigma_kernel(
            floats, kept, max_floats, True
        )
        np.testing.assert_array_equal(sigma_floats, ref_floats)

        if arm == "oracle":
            out_ints = _nlpar._nlpar_weighted_sum_kernel(
                ints, weights, 1, 1, 0, 3, 0, 3
            )
            np.testing.assert_array_equal(out_ints.reshape(3, 3, 4, 4), expected_ints)
            assert sigma_floats[1, 1] != sigma_o_floats[1, 1]

    @pytest.mark.parametrize("lam", [1.0, 2.5])
    def test_dthresh_is_consistent_between_kernel_and_objective(
        self, lam, identical_plus_gaussian
    ):
        dthresh = np.float32(0.5)
        tw = 0.34
        lam_arr = np.array([lam])

        # Exact arm: a (4, 5) field whose non-self distances are at or
        # below dthresh (weight exactly 1 in the averaging kernel) or
        # +inf (weight exactly 0), with the out-of-map slots of the map
        # invalid; the kernel's float32 weights are then exact and the
        # objective recomputed from them is bitwise the objective
        nav_shape = (4, 5)
        rng = np.random.default_rng(61)
        values = np.array([-0.3, 0.0, 0.25, 0.5, np.inf], dtype=np.float32)
        d = rng.choice(values, size=nav_shape + (9,)).astype(np.float32)
        d[..., 4] = -np.inf
        valid = _in_map_slots(nav_shape)
        weights = _nlpar._nlpar_weights_kernel(d, lam, dthresh)
        assert set(np.unique(weights)) == {0.0, 1.0}
        w = weights.astype(np.float64)
        w[~valid] = 0.0
        from_kernel = float(np.mean(np.abs(tw - 1.0 / w.sum(axis=-1))))
        value = _nlpar._nlpar_lambda_objective(lam_arr, d, valid, dthresh, tw)
        assert value == from_kernel
        # PyEBSDIndex's max(d, dthresh) floors the distances at dthresh
        # (weight exp(-0.5 / lam^2) < 1 on the slots at or below it)
        floor = _loptfunc_dthresh_floor(lam_arr, d, valid, tw, dthresh)
        assert abs(value - floor) > 1e-3
        # The two forms coincide at dthresh = 0
        zero = np.float32(0.0)
        assert _nlpar._nlpar_lambda_objective(
            lam_arr, d, valid, zero, tw
        ) == _loptfunc_dthresh_floor(lam_arr, d, valid, tw, zero)

        # Map arm: the 3 x 3 distances of the sigma pass of a synthetic
        # map, where the kernel's weights are rounded to float32
        # (measured 2026-10-05: 2e-10 from the objective at lam 1.0 and
        # 2.5) and the floored form is 0.057 and 0.011 away
        patterns = identical_plus_gaussian(ORACLE_NAV, ORACLE_SIG)
        _, d_map, _, valid_map = _ours_sigma_pass(patterns)
        w_map = _nlpar._nlpar_weights_kernel(d_map, lam, dthresh).astype(np.float64)
        w_map[~valid_map] = 0.0
        from_kernel_map = float(np.mean(np.abs(tw - 1.0 / w_map.sum(axis=-1))))
        value_map = _nlpar._nlpar_lambda_objective(
            lam_arr, d_map, valid_map, dthresh, tw
        )
        assert abs(value_map - from_kernel_map) <= 1e-6
        floor_map = _loptfunc_dthresh_floor(lam_arr, d_map, valid_map, tw, dthresh)
        assert abs(value_map - floor_map) > 1e-3

    @pytest.mark.parametrize("arm", ["wrapper", "lazy"])
    def test_saturation_max_is_global(self, arm, monkeypatch):
        # A (3, 6 | 4, 4) map whose left half holds the maximum 255 (in
        # column 0) and whose right half has the maximum 200 (pixel 15 of
        # every pattern); columns chunked (3, 3) at search radius 1:
        # depth max(1, 3 - 3) = 1, haloed blocks [0, 4) and [2, 6)
        data = np.clip(_hand_built_map((3, 6), seed=41, noise=4.0), 20.0, 190.0)
        data[:, 3:, 15] = np.float32(200.0)
        data[1, 0, 0] = np.float32(255.0)
        patterns = data.reshape(3, 6, 4, 4).astype(np.float32)
        sigma = np.full((3, 6), 4.0, dtype=np.float32)
        kept = np.arange(16, dtype=np.int64)
        global_max = np.float32(patterns.max())
        blocks = [(0, 4), (2, 6)]
        block_maxima = [np.float32(patterns[:, a:b].max()) for a, b in blocks]
        assert global_max == 255
        assert block_maxima == [255, 200]

        # Self-check with the transcription: on the right block, the
        # threshold from its own maximum changes the kept core
        right = data[:, 2:6]
        own = _transcribed_average(
            right, sigma[:, 2:6], (1, 1), 1.0, 0.0, kept, _average_threshold(200, True)
        )
        glob = _transcribed_average(
            right, sigma[:, 2:6], (1, 1), 1.0, 0.0, kept, _average_threshold(255, True)
        )
        assert not np.array_equal(own[:, 1:], glob[:, 1:])

        if arm == "lazy":
            # Through the method a column chunking of this tiny map is
            # merged into one chunk by the signal's rechunk, which keeps
            # only the row chunks; the transposed map (6, 3 | 4, 4) with
            # rows chunked (3, 3) keeps its two chunks, the 255 in the
            # upper chunk and the maximum 200 in the lower one. The
            # transcription is symmetric in rows and columns at radius
            # (1, 1), so the self-check above holds transposed
            patterns_t = np.ascontiguousarray(patterns.transpose(1, 0, 2, 3))
            sigma_t = np.ascontiguousarray(sigma.T)
            lazy_chunks = ((3, 3), (3,), (4,), (4,))
            row_blocks = [(0, 3), (3, 6)]
            maxima_t = [np.float32(patterns_t[a:b].max()) for a, b in row_blocks]
            assert maxima_t == [255, 200]
            for sigma_in in (sigma_t, None):
                kwargs = dict(
                    search_radius=1,
                    lam=1.0,
                    sigma=sigma_in,
                    dtype_out="float32",
                    show_progressbar=False,
                    inplace=False,
                )
                eager = kp.signals.EBSD(patterns_t.copy())
                expected = eager.average_non_local_neighbour_patterns(**kwargs).data
                recorded = _record_overlap_chunks(monkeypatch)
                lazy = kp.signals.LazyEBSD(
                    da.from_array(patterns_t, chunks=lazy_chunks)
                )
                s_out = lazy.average_non_local_neighbour_patterns(**kwargs)
                assert isinstance(s_out, kp.signals.LazyEBSD)
                s_out.compute(show_progressbar=False)
                monkeypatch.undo()
                # The two row chunks reached the overlap
                assert any(c[0] == (3, 3) for c in recorded)
                assert s_out.data.dtype == np.float32
                np.testing.assert_array_equal(s_out.data, expected)
            return

        def average(block, sigma_block, max_value, depth, location, num_chunks):
            return _nlpar._nlpar_average_chunk(
                block,
                sigma_block[..., None, None],
                lam=1.0,
                dthresh=np.float32(0.0),
                radius=(1, 1),
                mask_indices=kept,
                max_value=max_value,
                saturation_protect=True,
                depth=depth,
                dtype_out=np.dtype(np.float32),
                block_info=_block_info(location, num_chunks),
            )

        single = average(patterns, sigma, global_max, (0, 0), (0, 0), (1, 1))
        cores_global = []
        cores_own = []
        for k, (a, b) in enumerate(blocks):
            cores_global.append(
                average(
                    patterns[:, a:b], sigma[:, a:b], global_max, (0, 1), (0, k), (1, 2)
                )
            )
            cores_own.append(
                average(
                    patterns[:, a:b],
                    sigma[:, a:b],
                    block_maxima[k],
                    (0, 1),
                    (0, k),
                    (1, 2),
                )
            )
        assert cores_global[0].shape == cores_global[1].shape == (3, 3, 4, 4)

        # The block with the lower maximum sees a different threshold
        np.testing.assert_array_equal(cores_own[0], cores_global[0])
        assert not np.array_equal(cores_own[1], cores_global[1])
        np.testing.assert_array_equal(np.concatenate(cores_global, axis=1), single)

    def test_uint16_two_threshold_arm(self):
        # Pixel 0 of every pattern in [65280, 65469], the map maximum
        # 65535 at pixel 5 of pattern (1, 1): excluded from the sigma
        # pass by 0.9961 x 65535 = 65279.4, kept by the averaging pass's
        # 0.999 x 65535 = 65469.5
        rng = np.random.default_rng(5)
        data_u16 = rng.integers(5000, 60000, (3, 3, 16)).astype(np.uint16)
        data_u16[..., 0] = rng.integers(65280, 65470, (3, 3))
        data_u16[1, 1, 5] = 65535
        data = data_u16.astype(np.float32)
        max_value = np.float32(65535)
        kept = np.arange(16, dtype=np.int64)
        sigma_threshold = _sigma_threshold(max_value, True)
        average_threshold = _average_threshold(max_value, True)
        assert np.all(data[..., 0] >= sigma_threshold)
        assert np.all(data[..., 0] < average_threshold)

        # Test-local references with each constant, and the swapped
        # single-constant variants, which differ both ways
        sigma_ref = _sigma_reference(data, kept, max_value, True, factor=0.9961)
        sigma_swapped = _sigma_reference(data, kept, max_value, True, factor=0.999)
        assert not np.array_equal(sigma_ref, sigma_swapped)
        sigma2 = (sigma_ref * sigma_ref).astype(np.float32)
        d_ref, n2_ref, is_self = _transcribed_distances(
            data, sigma2, kept, average_threshold, (1, 1)
        )
        swapped_threshold = _average_threshold(
            max_value, True, factor=np.float32(0.9961)
        )
        d_swapped, _, _ = _transcribed_distances(
            data, sigma2, kept, swapped_threshold, (1, 1)
        )
        assert not np.array_equal(d_ref[~is_self], d_swapped[~is_self])
        assert np.all(n2_ref[~is_self] >= 15)
        sigma_off_ref = _sigma_reference(data, kept, max_value, False)
        d_off_ref, n2_off_ref, _ = _transcribed_distances(
            data, sigma2, kept, _average_threshold(max_value, False), (1, 1)
        )
        assert np.all(n2_off_ref[~is_self] == 16)

        # Ours: excluded from sigma, kept in the distances
        sigma, _, n2, valid = _nlpar._nlpar_sigma_kernel(data, kept, max_value, True)
        np.testing.assert_array_equal(sigma, sigma_ref)
        assert not np.array_equal(sigma, sigma_swapped)
        neighbour = valid.copy()
        neighbour[..., 4] = False
        assert np.all(n2[neighbour] <= 15)
        d = _nlpar._nlpar_distances_kernel(
            data, sigma2, kept, max_value, True, 1, 1, 0, 3, 0, 3
        )
        np.testing.assert_array_equal(d[~is_self], d_ref[~is_self])
        assert not np.array_equal(d[~is_self], d_swapped[~is_self])

        # Protection off keeps every pixel in both passes
        sigma_off, _, n2_off, _ = _nlpar._nlpar_sigma_kernel(
            data, kept, max_value, False
        )
        np.testing.assert_array_equal(sigma_off, sigma_off_ref)
        assert np.all(n2_off[neighbour] == 16)
        d_off = _nlpar._nlpar_distances_kernel(
            data, sigma2, kept, max_value, False, 1, 1, 0, 3, 0, 3
        )
        np.testing.assert_array_equal(d_off[~is_self], d_off_ref[~is_self])

        # Protection off with a float32 maximum of 2**24 or more: max + 1
        # rounds to max in float32, so the distances drop the pixel at
        # the maximum, as PyEBSDIndex's float32 threshold does, while the
        # float64 threshold of the sigma pass keeps it
        big_max = np.float32(2**25)
        big = data.copy()
        big[1, 1, 5] = big_max
        assert _average_threshold(big_max, False) == big_max
        _, _, n2_big, _ = _nlpar._nlpar_sigma_kernel(big, kept, big_max, False)
        assert np.all(n2_big[neighbour] == 16)
        d_drop, n2_drop, _ = _transcribed_distances(big, sigma2, kept, big_max, (1, 1))
        assert set(np.unique(n2_drop[~is_self])) == {15.0, 16.0}
        d_big = _nlpar._nlpar_distances_kernel(
            big, sigma2, kept, big_max, False, 1, 1, 0, 3, 0, 3
        )
        np.testing.assert_array_equal(d_big[~is_self], d_drop[~is_self])

    @pytest.mark.parametrize("arm", ORACLE_ARMS)
    def test_sigma_threshold_is_float64_and_average_threshold_float32(
        self, arm, request
    ):
        # A map maximum of 242 rounds both float32 products below the
        # float64 ones: float32(242 x float32(0.9961)) < 242 x 0.9961 and
        # float32(242 x float32(0.999)) < 242 x 0.999 (also against the
        # float64 products with the float32 factors). Pixel 0 of every
        # pattern sits exactly at the float32 sigma product, pixel 1 at
        # the float32 averaging product, and the maximum is pixel 5 of
        # pattern (1, 1). The float64 sigma threshold keeps pixel 0, a
        # float32 one would drop it; the float32 averaging threshold
        # drops pixel 1, a float64 one would keep it
        max_value = np.float32(242.0)
        sigma_f32 = np.float32(max_value * np.float32(0.9961))
        average_f32 = np.float32(max_value * np.float32(0.999))
        for product, factor in [(sigma_f32, 0.9961), (average_f32, 0.999)]:
            assert np.float64(product) < np.float64(max_value) * factor
            factor32 = np.float64(np.float32(factor))
            assert np.float64(product) < np.float64(max_value) * factor32
        sigma_threshold = _sigma_threshold(max_value, True)
        average_threshold = _average_threshold(max_value, True)
        assert sigma_f32 < sigma_threshold
        assert average_threshold == average_f32

        rng = np.random.default_rng(51)
        data = rng.uniform(20.0, 200.0, (3, 3, 16)).astype(np.float32)
        data[..., 0] = sigma_f32
        data[..., 1] = average_f32
        data[1, 1, 5] = max_value
        patterns = data.reshape(3, 3, 4, 4)
        kept = np.arange(16, dtype=np.int64)

        # Sigma pass: 15 kept pixel pairs per neighbour pair (pixel 1
        # dropped), 14 for the pairs with pattern (1, 1). Dropping pixel
        # 0 as well, as a float32 threshold does, changes the sigmas
        sigma_ref = _sigma_reference(data, kept, max_value, True)
        assert not np.array_equal(
            sigma_ref, _sigma_reference(data, kept[1:], max_value, True)
        )
        expected_n2 = np.zeros((3, 3, 9), dtype=np.float32)
        neighbour = np.zeros((3, 3, 9), dtype=bool)
        for j in range(3):
            for i in range(3):
                for k in range(9):
                    jn, i_n = j + k // 3 - 1, i + k % 3 - 1
                    if k == 4 or not (0 <= jn < 3 and 0 <= i_n < 3):
                        continue
                    neighbour[j, i, k] = True
                    with_center = (1, 1) in ((j, i), (jn, i_n))
                    expected_n2[j, i, k] = 14.0 if with_center else 15.0

        # Averaging pass on the sigmas of the sigma pass: pixel 0 kept,
        # pixel 1 dropped; a float64 threshold keeps pixel 1 and changes
        # every distance
        sigma2 = (sigma_ref * sigma_ref).astype(np.float32)
        d_ref, n2_ref, is_self = _transcribed_distances(
            data, sigma2, kept, average_threshold, (1, 1)
        )
        threshold64 = np.float64(max_value) * 0.999
        d_64, n2_64, _ = _transcribed_distances(data, sigma2, kept, threshold64, (1, 1))
        assert set(np.unique(n2_ref[~is_self])) == {14.0, 15.0}
        np.testing.assert_array_equal(n2_64[~is_self], n2_ref[~is_self] + 1)
        assert np.all(d_64[~is_self] != d_ref[~is_self])
        average_ref = _transcribed_average(
            data, sigma_ref, (1, 1), 1.0, 0.0, kept, average_threshold
        )
        average_64 = _transcribed_average(
            data, sigma_ref, (1, 1), 1.0, 0.0, kept, threshold64
        )
        assert not np.array_equal(average_ref, average_64)

        if arm == "oracle":
            # The compiled kernels count pixel 0 in the sigma pass and
            # drop pixel 1 in the averaging pass
            nlpar = request.getfixturevalue("pyebsdindex_kernels")
            oracle = _oracle_sigma(nlpar, patterns)
            np.testing.assert_array_equal(oracle[0], sigma_ref)
            np.testing.assert_array_equal(oracle[2][0, 0, :4], [16.0, 15.0, 15.0, 14.0])
            _assert_sigma_pass_parity(*_ours_sigma_pass(patterns), 16, oracle)
            expected = _oracle_average(nlpar, patterns, oracle[0], 1, 1.0, 0.0)
            np.testing.assert_array_equal(average_ref.reshape(patterns.shape), expected)
            ours = _ours_method_average(patterns, 1, 1.0, 0.0, sigma=oracle[0])
            _assert_average_parity(ours, expected)

        # Ours, compiled and pure Python: the sigma kernel's threshold is
        # the float64 product, the distances kernel's the float32 one
        kernel = _nlpar._nlpar_sigma_kernel
        for route in (kernel, _py_func(kernel)):
            sigma, _, n2, valid = route(data, kept, max_value, True)
            np.testing.assert_array_equal(sigma, sigma_ref)
            np.testing.assert_array_equal(valid & neighbour, neighbour)
            np.testing.assert_array_equal(n2[neighbour], expected_n2[neighbour])
        kernel = _nlpar._nlpar_distances_kernel
        for route in (kernel, _py_func(kernel)):
            d = route(data, sigma2, kept, max_value, True, 1, 1, 0, 3, 0, 3)
            np.testing.assert_array_equal(d[~is_self], d_ref[~is_self])

    def test_sigma_fallback_value_is_1e12(self):
        # The fallback is sqrt of the float32 seed 1e24 of the minimum
        assert np.sqrt(np.float32(1e24)) == np.float32(1e12)

        # Module constants read by the kernels: a Python float meeting
        # the float64 maximum in the sigma pass, a float32 scalar meeting
        # the float32 maximum in the averaging pass
        assert _nlpar.SIGMA_SATURATION_FACTOR == 0.9961
        assert type(_nlpar.SIGMA_SATURATION_FACTOR) is float
        assert _nlpar.AVERAGE_SATURATION_FACTOR == np.float32(0.999)
        assert type(_nlpar.AVERAGE_SATURATION_FACTOR) is np.float32

        # A constant map has no neighbour with d2 > 0, with protection
        # on (every pixel at the maximum, nothing kept) and off
        constant = np.full((3, 3, 16), 100.0, dtype=np.float32)
        kept = np.arange(16, dtype=np.int64)
        for protect in (True, False):
            sigma, _, _, _ = _nlpar._nlpar_sigma_kernel(
                constant, kept, np.float32(100.0), protect
            )
            assert sigma.dtype == np.float32
            np.testing.assert_array_equal(sigma, np.full((3, 3), np.float32(1e12)))


@pytest.mark.weekly
class TestPerformance:
    # Recorded baselines, never asserted: the numbers are read from a
    # junit-xml report of a run without xdist (weekly: nothing to gate)

    @staticmethod
    def _best_of(func, n_repeats: int = 3) -> float:
        """Return the best wall time in seconds of ``n_repeats`` calls
        after one warm-up call.
        """
        func()
        times = []
        for _ in range(n_repeats):
            tic = time.perf_counter()
            func()
            times.append(time.perf_counter() - tic)
        return min(times)

    def test_runtime_is_recorded(self, nickel_large_patterns, record_property):
        s = kp.signals.EBSD(nickel_large_patterns["raw"].copy())
        sr, lam = 3, 2.5

        def sigma_pass():
            s.get_nlpar_sigma(show_progressbar=False)

        def average_pass():
            s.average_non_local_neighbour_patterns(
                search_radius=sr, lam=lam, inplace=False, show_progressbar=False
            )

        for scheduler in ("threads", "synchronous"):
            with dask.config.set(scheduler=scheduler):
                record_property(
                    f"nlpar_sigma_nickel_ebsd_large_{scheduler}_s",
                    self._best_of(sigma_pass),
                )
                record_property(
                    f"nlpar_average_nickel_ebsd_large_sr{sr}_{scheduler}_s",
                    self._best_of(average_pass),
                )
