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

"""Tests of the NLPAR methods of the EBSD signal,
:meth:`~kikuchipy.signals.EBSD.average_non_local_neighbour_patterns`,
:meth:`~kikuchipy.signals.EBSD.get_nlpar_sigma` and
:meth:`~kikuchipy.signals.EBSD.get_nlpar_lambda`.

Oracles used here, none of which is PyEBSDIndex's NLPAR:

* :func:`nlpar_reference`, a float64 NumPy transcription of the three
  NLPAR equations of Brewick, Wright and Rowenhorst (2019), written
  independently of :mod:`kikuchipy.pattern._nlpar`;
* analytic identities whose answer is exact by construction (constant
  map, radius 0, injected sigma, huge lambda, power-of-two scaling);
* seeded synthetic maps whose statistics are derived (iid noise, two
  grains), from the generators of the root ``conftest.py``;
* the eager route as the oracle of the lazy route (bitwise);
* real datasets (``nickel_ebsd_large``, ``si_wafer``), on which map
  metrics and Hough indexing quality must improve.

Tolerances that were measured on the finished implementation and then
pinned are module constants, each with a dated record of the
measurement. Every test reading one computes its result first and
reports the measured value if it fails; a constant left at None, as a
placeholder of a later stage would be, fails with "unfilled
MEASURED-THEN-PINNED placeholder".
"""

import logging
import re
import time
import warnings

import dask
import dask.array as da
import hyperspy.api as hs
import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation
import pytest
from scipy.ndimage import correlate
from scipy.optimize import minimize

import kikuchipy as kp
from kikuchipy._constants import dependency_version
import kikuchipy.pattern._nlpar as nlpar_module
from kikuchipy.pattern._nlpar import (
    _nlpar_distances_kernel,
    _nlpar_normalized_distances,
    _nlpar_sigma_kernel,
    _nlpar_weights_kernel,
)
import kikuchipy.signals.ebsd as ebsd_module
from kikuchipy.signals.util._dask import get_dask_array

# ------------------- Measured, then pinned tolerances ---------------- #
# Each constant below was measured at the implementation gate, recorded
# with date, machine and recipe, and pinned with the ~2x margin
# convention or as a band around the measured value. The seeds quoted
# are drafting seeds (measured 2026-09-11 or derived 2026-10-04), not
# pins.

# Largest absolute difference, in grey levels on a 0-255 scale, between
# the float32 output of the method and the float64 reference
# nlpar_reference. Seed 1e-3 grey levels (float32 weights with 1e-6 to
# 1e-5 relative error applied to values spread over 20-240), pinned at
# ~2x the measured value.
# Pinned 2026-10-05: measured 6.91e-5 at worst over the eleven arms of
# the reference agreement and the two small-map arms (sr=1, lam=0.7);
# pinned at ~2x, 1.4e-4.
REFERENCE_MAX_ABS_GREY: float = 1.4e-4

# Band (low, high) of the median over the interior of the ratio of the
# estimated to the true noise level on the iid-noise map at N = 1024
# pixels. Seed (0.969, 0.978): the minimum over eight correlated
# estimates of sigma^2, each with relative spread sqrt(2 / N), sits 1.0
# to 1.4 of that spread below the mean.
# Pinned 2026-10-05: measured 0.97302 (seed 0). The four iid-noise bands
# are the measured value plus or minus the full range of the statistic
# over seeds 0-19 (twice its half-range), rounded outward; here range
# 0.96852-0.97762, band (0.963, 0.983). The mean or the median of the
# eight estimates instead of the minimum gives 1.0009 or 1.0002.
NOISE_SIGMA_RATIO_BAND: tuple[float, float] = (0.963, 0.983)

# Band (low, high) of the mean of the normalised 3 x 3 distances on the
# iid-noise map at N = 1024. Seed: about +1.2 (a 5 % low sigma^2 bias
# shifts the mean by about 0.05 sqrt(N / 2)).
# Pinned 2026-10-05: measured 1.2049 (seed 0), seeds 0-19 range
# 1.1367-1.3270, band (1.01, 1.40). A sqrt(n2) denominator gives 1.704,
# the flipped correction sign 46.5.
D_MEAN_BAND: tuple[float, float] = (1.01, 1.40)

# Band (low, high) of the standard deviation of the normalised 3 x 3
# distances on the iid-noise map at N = 1024. Seed (2026-09-11, derived
# again 2026-10-04): about 1.0, the standard deviation with the exact
# sigma; refuted 2026-10-05.
# Pinned 2026-10-05: measured 0.83 (0.8266 at seed 0; 0.754-0.888 over
# seeds 0-19), identical to the standard deviation of the compiled
# PyEBSDIndex sigma_numba distances on the same map. The test uses the
# estimated sigma: each 3 x 3 pair estimate enters the minimum of both
# of its points, so every 3 x 3 distance is >= 0 (down to -1.5e-6 in
# float32) and the distribution is cut at 0, which narrows it. Band
# (0.69, 0.97). A sqrt(n2) denominator gives 1.169, a d2 / n2
# normalisation 4.45.
D_STD_BAND: tuple[float, float] = (0.69, 0.97)

# Band (low, high) of the fraction of neighbour weights exactly 1.0
# (normalised distance <= 0) on the iid-noise map. Seed: about 0.1.
# Pinned 2026-10-05: measured 0.09606 (seed 0), seeds 0-19 range
# 0.06047-0.09954, band (0.057, 0.136).
UNIT_WEIGHT_FRACTION_BAND: tuple[float, float] = (0.057, 0.136)

# Largest relative difference between the measured residual noise
# variance of the output and sigma_true^2 times the mean over patterns
# of the sum of squared normalised weights. Seed: about 0.05
# (finite-sample class at 144 patterns x 1024 pixels).
# Pinned 2026-10-05: measured 0.0388 (variance 2.061 vs 1.984 expected;
# seeds 0-19 range 0.010-0.068); pinned at ~2x, 0.08.
NOISE_REDUCTION_TOL: float = 0.08

# Smallest fraction of the boundary-column contrast of the two-grain
# map retained after NLPAR. Seed 0.995.
# Pinned 2026-10-05: measured 0.99842 (lam 0.7) and 0.99776 (lam 2.5),
# a loss of 0.0022 at worst (seeds 1-20 range 0.991-1.007, no
# systematic loss); pinned at ~2x the loss, 0.995. The Gaussian window
# retains 0.401.
TWO_GRAIN_CONTRAST_MIN: float = 0.995

# Largest ratio of the residual rms on the two boundary columns of the
# two-grain map to the residual rms on the interior columns. Seed: about
# 1.0-1.3 (boundary patterns have fewer same-grain neighbours in their
# window and average less).
# Pinned 2026-10-05: measured 1.3228 (rms 1.932 on the boundary columns
# vs 1.461 inside; seeds 1-20 range 1.246-1.347); pinned at ~2x the
# excess over 1, 1.65.
TWO_GRAIN_BOUNDARY_RESIDUAL_TOL: float = 1.65

# ----------------- MEASURED-THEN-PINNED placeholders ---------------- #
# Each constant below is a placeholder, measured then pinned: None
# until the implementation gate measures it, records date, machine and
# recipe, and pins it as a band around the measured value or with the
# ~2x margin convention. The seeds quoted are drafting seeds (measured
# 2026-09-11), not pins; where none exists the quantity is unmeasured.

# Band (low, high) of get_nlpar_lambda() at target weight 0.34 on the
# raw nickel_ebsd_large. Placeholder, measured then pinned. Seed 1.1387
# (phantom-free; 1.1164 with PyEBSDIndex's phantom-counting objective).
# Which seed pair belongs to the raw and which to the corrected map is
# re-measured at the gate, not assumed.
LAMBDA_NI_RAW: tuple[float, float] | None = None

# Band (low, high) of get_nlpar_lambda() at target weight 0.34 on the
# background-corrected nickel_ebsd_large (static, then dynamic
# background removed with the defaults). Placeholder, measured then
# pinned. Seed 2.5787 (phantom-free; 2.5246 with phantoms).
LAMBDA_NI_CORRECTED: tuple[float, float] | None = None

# Band (low, high) of the mean average neighbour dot product of the
# background-corrected nickel_ebsd_large before NLPAR. Placeholder,
# measured then pinned. Seed 0.600.
ADP_BEFORE: tuple[float, float] | None = None

# Band (low, high) of the same mean after NLPAR with the optimised
# lambda (lam=None, about 2.52) at search radius 3. Placeholder,
# measured then pinned. Seed 0.904.
ADP_AFTER_AUTO: tuple[float, float] | None = None

# Band (low, high) of the same mean after NLPAR with lam=0.7 at search
# radius 3. Placeholder, measured then pinned. Seed 0.766.
ADP_AFTER_07: tuple[float, float] | None = None

# Band (low, high) of the mean image quality of the background-corrected
# nickel_ebsd_large before NLPAR. Placeholder, measured then pinned.
# Seed: unmeasured (0.184 seen while drafting these tests, 2026-10-05,
# not a measurement of record).
IQ_BEFORE: tuple[float, float] | None = None

# Band (low, high) of the same mean after NLPAR with lam=None.
# Placeholder, measured then pinned. Seed: unmeasured.
IQ_AFTER_AUTO: tuple[float, float] | None = None

# Smallest gains of the medians of the Hough indexing quality metrics on
# the 165 patterns of s.inav[::5, ::5] of the corrected map, after NLPAR
# minus before (pattern quality, cross-correlation metric, number of
# matched bands) and before minus after (band fit, in degrees, lower is
# better). Placeholders, measured then pinned: 0.5 x the measured gain
# where it is positive, 0.0 ("not worse") otherwise, recorded as such.
# Seeds: unmeasured.
HOUGH_PQ_GAIN: float | None = None
HOUGH_FIT_GAIN: float | None = None
HOUGH_NMATCH_GAIN: float | None = None
HOUGH_CM_GAIN: float | None = None

# Largest median misorientation, in degrees, between the Hough
# orientations after NLPAR and the orientations stored with the dataset,
# on the same 165 patterns compared by point order. Placeholder,
# measured then pinned (~2x); the value before NLPAR is recorded beside
# it. Seed: unmeasured.
HOUGH_MISO_MEDIAN_AFTER: float | None = None

# Largest coefficient of variation of the sigma map of si_wafer (a
# single crystal, so a flat map). Placeholder, measured then pinned.
# Seed: unmeasured.
SI_SIGMA_CV: float | None = None

# Band (low, high) of the median number of effective neighbours,
# 1 / sum_j w_ij^2 of the normalised search-window weights, of si_wafer
# at the optimised lambda. Placeholder, measured then pinned. Seed:
# unmeasured.
SI_NEFF_MEDIAN: tuple[float, float] | None = None

# Smallest ratio of the mean image quality of si_wafer after NLPAR to
# the one before (lam=None). Placeholder, measured then pinned. Seed:
# unmeasured, > 1.
SI_IQ_GAIN: float | None = None

# --------------------------- Small fixtures ------------------------- #

NAV_SHAPE = (4, 5)
SIG_SHAPE = (6, 6)

# Base pattern of the generators identical_plus_gaussian and two_grain
# at signal shape (32, 32): a float32 ramp from 40 to 200; grain B of
# the two-grain map adds 30 to it.
RAMP_32 = np.linspace(40.0, 200.0, 32 * 32, dtype=np.float32).reshape(32, 32)
TWO_GRAIN_DELTA = np.float32(30.0)


def _require_pin(value, name: str, measured):
    """Return a measured-then-pinned value, or fail loudly, reporting
    the measured quantity, while it is an unfilled placeholder.
    """
    if value is None:
        raise AssertionError(
            f"{name} is an unfilled MEASURED-THEN-PINNED placeholder; measured "
            f"{measured!r}. Pin it with a dated value and its recipe."
        )
    return value


def _assert_at_most(measured: float, bound: float | None, name: str) -> None:
    bound = _require_pin(bound, name, measured)
    assert measured <= bound, f"{name}: {measured!r} > {bound!r}"


def _assert_at_least(measured: float, bound: float | None, name: str) -> None:
    bound = _require_pin(bound, name, measured)
    assert measured >= bound, f"{name}: {measured!r} < {bound!r}"


def _assert_in_band(
    measured: float, band: tuple[float, float] | None, name: str
) -> None:
    low, high = _require_pin(band, name, measured)
    assert low <= measured <= high, f"{name}: {measured!r} not in [{low}, {high}]"


def _float32_ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the distance in float32 ulps between two arrays of
    non-negative float32 values.
    """
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def _random_map(random_uniform_saturated, nav_shape: tuple[int, ...] = NAV_SHAPE):
    """Return the unsaturated uint8 random map of shape
    ``nav_shape + (6, 6)``, values in [20, 239].
    """
    return random_uniform_saturated(nav_shape, SIG_SHAPE, frac=0.0, seed=1)


def _resolve(kwargs: dict, sig_shape: tuple[int, int], circle_mask) -> dict:
    """Return a copy of method keyword arguments with the mask name
    "circle" replaced by the boolean inscribed-circle mask of the
    ``circle_mask`` fixture, and "circle_int" by the same mask as int64
    0/1.
    """
    kwargs = dict(kwargs)
    name = kwargs.get("signal_mask")
    if isinstance(name, str):
        mask = circle_mask(sig_shape)
        kwargs["signal_mask"] = mask.astype(np.int64) if name == "circle_int" else mask
    return kwargs


def _average(s, **kwargs) -> np.ndarray:
    """Return the data of the NLPAR-averaged copy of a signal."""
    s_out = s.average_non_local_neighbour_patterns(inplace=False, **kwargs)
    return s_out.data


def _lazy_signal(data: np.ndarray | da.Array, nav_chunks: tuple) -> kp.signals.LazyEBSD:
    """Return a lazy EBSD signal of a map of patterns with the given
    navigation chunks (a chunk shape or explicit chunks, one entry per
    navigation axis, -1 for one chunk) and each signal axis in one
    chunk.
    """
    chunks = tuple(nav_chunks) + (-1, -1)
    if isinstance(data, da.Array):
        dask_array = data.rechunk(chunks)
    else:
        dask_array = da.from_array(data, chunks=chunks)
    return kp.signals.LazyEBSD(dask_array)


def _average_lazy(s: kp.signals.LazyEBSD, **kwargs) -> np.ndarray:
    """Return the computed data of the NLPAR-averaged copy of a lazy
    signal, asserting that the copy is lazy.
    """
    s_out = s.average_non_local_neighbour_patterns(inplace=False, **kwargs)
    assert isinstance(s_out, kp.signals.LazyEBSD)
    return s_out.data.compute()


def _whole_map_sigma_pass(data: np.ndarray, saturation_protect: bool = True):
    """Return sigma, d2, n2, valid of the sigma kernel on a whole 2D
    map of patterns, every pixel kept.
    """
    n_rows, n_cols = data.shape[:2]
    data3 = np.ascontiguousarray(data, dtype=np.float32).reshape(n_rows, n_cols, -1)
    mask_indices = np.arange(data3.shape[-1], dtype=np.int64)
    max_value = np.float32(data3.max())
    return _nlpar_sigma_kernel(data3, mask_indices, max_value, saturation_protect)


def _whole_map_search_distances(
    data: np.ndarray,
    sigma: np.ndarray,
    search_radius: int,
    saturation_protect: bool = True,
) -> np.ndarray:
    """Return the search-window normalised distances of the distances
    kernel on a whole 2D map of patterns, every pixel kept.
    """
    n_rows, n_cols = data.shape[:2]
    data3 = np.ascontiguousarray(data, dtype=np.float32).reshape(n_rows, n_cols, -1)
    sigma = np.asarray(sigma, dtype=np.float32)
    sigma2 = (sigma * sigma).astype(np.float32)
    mask_indices = np.arange(data3.shape[-1], dtype=np.int64)
    max_value = np.float32(data3.max())
    return _nlpar_distances_kernel(
        data3,
        sigma2,
        mask_indices,
        max_value,
        saturation_protect,
        search_radius,
        search_radius,
        0,
        n_rows,
        0,
        n_cols,
    )


# ------------------ Float64 NumPy reference of NLPAR ---------------- #


def _reference_search_window(center: int, radius: int, n: int) -> range:
    """Return the search window along one axis: ``2 radius + 1``
    points shifted inward at the borders, or the whole axis if it is
    shorter than that.
    """
    size = 2 * radius + 1
    if n < size:
        return range(n)
    start = min(max(center - radius, 0), n - size)
    return range(start, start + size)


def _reference_clipped_window(center: int, radius: int, n: int) -> range:
    """Return the window along one axis clipped at the borders."""
    return range(max(center - radius, 0), min(center + radius, n - 1) + 1)


def _reference_radius(
    search_radius: int | tuple[int, ...], nav_dim: int
) -> tuple[int, int]:
    """Return the (row, column) search radius, a 1D scan being one row."""
    if np.ndim(search_radius) == 0:
        radius = (int(search_radius),) * nav_dim
    else:
        radius = tuple(int(r) for r in search_radius)
    if nav_dim == 1:
        return 0, radius[0]
    return radius[0], radius[1]


def _reference_pair(
    p0: np.ndarray, p1: np.ndarray, kept: np.ndarray, threshold: float
) -> tuple[float, int]:
    """Return the sum of squared differences and the number of pixel
    pairs that are kept and both strictly below the threshold.
    """
    use = kept & (p0 < threshold) & (p1 < threshold)
    diff = p0[use] - p1[use]
    return float(np.sum(diff * diff)), int(np.count_nonzero(use))


def _reference_sigma2(p: np.ndarray, kept: np.ndarray, threshold: float) -> np.ndarray:
    """Return sigma^2 of every pattern: the minimum over the neighbours
    of the clipped 3 x 3 window with a non-zero distance of
    ``d2 / (2 n)``, or 1e24 (sigma 1e12) if there is none.
    """
    n_rows, n_cols = p.shape[:2]
    sigma2 = np.full((n_rows, n_cols), 1e24)
    for j in range(n_rows):
        for i in range(n_cols):
            candidates = []
            for jn in _reference_clipped_window(j, 1, n_rows):
                for i_n in _reference_clipped_window(i, 1, n_cols):
                    if (jn, i_n) == (j, i):
                        continue
                    d2, n = _reference_pair(p[j, i], p[jn, i_n], kept, threshold)
                    if d2 > 0:
                        candidates.append(d2 / (2 * n))
            if candidates:
                sigma2[j, i] = min(candidates)
    return sigma2


def _reference_distance(
    p0: np.ndarray, p1: np.ndarray, kept: np.ndarray, threshold: float, s2: float
) -> float:
    """Return the normalised distance of a pair given
    ``s2 = sigma_i^2 + sigma_j^2``: +inf without a comparable pixel,
    ``1e6 n`` if the normaliser is at most 1e-8.
    """
    d2, n = _reference_pair(p0, p1, kept, threshold)
    if n == 0:
        return np.inf
    dnorm = s2 * np.sqrt(2.0 * n)
    if dnorm <= 1e-8:
        return 1e6 * n
    return (d2 - n * s2) / dnorm


def nlpar_reference(
    data: np.ndarray,
    search_radius: int | tuple[int, ...],
    lam: float,
    dthresh: float,
    sigma: float | np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
    saturation_protect: bool = True,
) -> np.ndarray:
    """Return NLPAR-averaged patterns computed in float64 directly from
    the three equations of Brewick, Wright and Rowenhorst (2019).

    Parameters
    ----------
    data
        Patterns of shape ``nav_shape + (h, w)`` with a 1D or 2D
        navigation shape.
    search_radius
        Search radius, an int for every navigation axis or one per axis
        in (row, column) order.
    lam
        Weight decay lambda.
    dthresh
        Distance threshold.
    sigma
        Noise level: None to estimate it, a scalar for a constant map,
        or an array of the navigation shape.
    signal_mask
        Boolean mask of the signal shape, True for pixels excluded from
        the distances. All pixels are averaged.
    saturation_protect
        Whether a pixel pair counts only if both values are strictly
        below 0.9961 (sigma) or 0.999 (search window) times the global
        maximum of the map.

    Returns
    -------
    averaged
        float64 array of the shape of ``data``.

    Notes
    -----
    With ``n_ij`` the per-pair count of kept pixel pairs:
    ``sigma_i^2 = min_j sum_k (p_ik - p_jk)^2 / (2 n_ij)`` over the
    neighbours of the clipped 3 x 3 window with a non-zero distance
    (1e24 if none); ``d_ij = [sum_k (p_ik - p_jk)^2 - n_ij s] /
    [s sqrt(2 n_ij)]`` with ``s = sigma_i^2 + sigma_j^2``, +inf if
    ``n_ij = 0``; ``w_ij = exp(-max(d_ij - dthresh, 0) / lam^2)``,
    ``w_ii = 1``; and the output is ``sum_j w_ij p_j / sum_j w_ij``
    over the search window, shifted inward at the borders and the whole
    axis when the axis is shorter than ``2 r + 1``.
    """
    data = np.asarray(data)
    nav_shape = data.shape[:-2]
    radius_rows, radius_cols = _reference_radius(search_radius, len(nav_shape))
    if len(nav_shape) == 1:
        n_rows, n_cols = 1, nav_shape[0]
    else:
        n_rows, n_cols = nav_shape
    p = data.astype(np.float64).reshape(n_rows, n_cols, -1)
    if signal_mask is None:
        kept = np.ones(p.shape[-1], dtype=bool)
    else:
        kept = ~np.asarray(signal_mask, dtype=bool).ravel()

    max_value = float(p.max())
    if saturation_protect:
        threshold_sigma = max_value * 0.9961
        threshold_search = max_value * 0.999
    else:
        threshold_sigma = threshold_search = max_value + 1.0

    if sigma is None:
        sigma2 = _reference_sigma2(p, kept, threshold_sigma)
    elif np.ndim(sigma) == 0:
        sigma2 = np.full((n_rows, n_cols), float(sigma) ** 2)
    else:
        sigma2 = np.asarray(sigma, dtype=np.float64).reshape(n_rows, n_cols) ** 2

    lam2 = float(lam) ** 2
    out = np.zeros_like(p)
    for j in range(n_rows):
        for i in range(n_cols):
            weight_sum = 0.0
            for jn in _reference_search_window(j, radius_rows, n_rows):
                for i_n in _reference_search_window(i, radius_cols, n_cols):
                    if (jn, i_n) == (j, i):
                        w = 1.0
                    else:
                        s2 = sigma2[j, i] + sigma2[jn, i_n]
                        d = _reference_distance(
                            p[j, i], p[jn, i_n], kept, threshold_search, s2
                        )
                        w = float(np.exp(-max(d - dthresh, 0.0) / lam2))
                    out[j, i] += w * p[jn, i_n]
                    weight_sum += w
            out[j, i] /= weight_sum
    return out.reshape(data.shape)


def _box_mean(
    data: np.ndarray, search_radius: int | tuple[int, int], window
) -> np.ndarray:
    """Return the float64 unweighted mean of every pattern's window on a
    2D map, the window per axis given by ``window(center, radius, n)``.
    """
    n_rows, n_cols = data.shape[:2]
    radius_rows, radius_cols = _reference_radius(search_radius, 2)
    p = data.astype(np.float64)
    out = np.zeros_like(p)
    for j in range(n_rows):
        rows = list(window(j, radius_rows, n_rows))
        for i in range(n_cols):
            cols = list(window(i, radius_cols, n_cols))
            out[j, i] = p[np.ix_(rows, cols)].mean(axis=(0, 1))
    return out


def shifted_box_mean(
    data: np.ndarray, search_radius: int | tuple[int, int]
) -> np.ndarray:
    """Return the unweighted mean over the shifted-inward search window."""
    return _box_mean(data, search_radius, _reference_search_window)


def clamped_box_mean(
    data: np.ndarray, search_radius: int | tuple[int, int]
) -> np.ndarray:
    """Return the unweighted mean over the window clipped at the borders."""
    return _box_mean(data, search_radius, _reference_clipped_window)


# ----------------------------- Identities --------------------------- #

# Arms of the reference agreement: (fixture, method keyword arguments,
# keyword arguments of runs whose output must differ). The default arm
# is search_radius=1, lam=0.7 with dthresh 0, no mask and protection on.
_DEFAULT_ARM = {"search_radius": 1, "lam": 0.7}
REFERENCE_ARMS = [
    pytest.param("random", {"search_radius": 1, "lam": 0.7}, [], id="sr=1-lam=0.7"),
    pytest.param("random", {"search_radius": 1, "lam": 2.5}, [], id="sr=1-lam=2.5"),
    pytest.param("random", {"search_radius": 2, "lam": 0.7}, [], id="sr=2-lam=0.7"),
    pytest.param("random", {"search_radius": 2, "lam": 2.5}, [], id="sr=2-lam=2.5"),
    pytest.param(
        "random",
        {"search_radius": 1, "lam": 0.7, "dthresh": 0.5},
        [_DEFAULT_ARM],
        id="dthresh=0.5",
    ),
    pytest.param(
        "random",
        {"search_radius": 1, "lam": 0.7, "signal_mask": "circle"},
        [_DEFAULT_ARM],
        id="signal_mask=circle",
    ),
    pytest.param(
        "saturated",
        {"search_radius": 1, "lam": 0.7, "saturation_protect": True},
        [],
        id="saturation_protect=True",
    ),
    pytest.param(
        "saturated",
        {"search_radius": 1, "lam": 0.7, "saturation_protect": False},
        [{"search_radius": 1, "lam": 0.7, "saturation_protect": True}],
        id="saturation_protect=False",
    ),
    pytest.param(
        "random",
        {"search_radius": (1, 2), "lam": 0.7},
        [_DEFAULT_ARM, {"search_radius": (2, 1), "lam": 0.7}],
        id="(1, 2)",
    ),
    pytest.param(
        "random",
        {"search_radius": (2, 1), "lam": 0.7},
        [_DEFAULT_ARM, {"search_radius": (1, 2), "lam": 0.7}],
        id="(2, 1)",
    ),
    pytest.param("scan_1d", {"search_radius": 2, "lam": 0.7}, [], id="1d-sr=2"),
]


class TestIdentities:
    """Formulas pinned through the public method against answers known
    exactly or computed by the float64 reference.
    """

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize("saturation_protect", [False, True])
    @pytest.mark.parametrize("search_radius, n_window", [(1, 9), (3, 20)])
    def test_constant_map_is_an_identity(
        self, search_radius, n_window, saturation_protect, dtype
    ):
        # Every pixel of every pattern is 100. At radius 1 the 3 x 3
        # window shifts at every border point (9 weights), at radius 3
        # it is the whole (4, 5) map (20 weights)
        data = np.full(NAV_SHAPE + SIG_SHAPE, 100, dtype=dtype)
        s = kp.signals.EBSD(data.copy())
        kwargs = {
            "search_radius": search_radius,
            "lam": 1.0,
            "saturation_protect": saturation_protect,
        }
        out_f32 = _average(s, dtype_out="float32", **kwargs)
        out_default = _average(s, **kwargs)
        sigma = s.get_nlpar_sigma(saturation_protect=saturation_protect)

        assert out_f32.dtype == np.float32
        assert out_default.dtype == np.dtype(dtype)
        assert np.array_equal(s.data, data)
        # No neighbour has a non-zero distance: the 1e12 fallback
        assert sigma.dtype == np.float32
        assert np.all(sigma == np.float32(1e12))

        p = np.float32(100)
        if saturation_protect:
            # Every pixel equals the global maximum, so the strict
            # thresholds exclude every pixel, every pair has no
            # comparable pixel and weight 0: the self pattern exactly
            assert np.array_equal(out_f32, data.astype(np.float32))
            assert np.array_equal(out_default, data)
            d = _whole_map_search_distances(data, sigma, search_radius)
            w = _nlpar_weights_kernel(d, 1.0, np.float32(0.0))
            is_self = np.isneginf(d)
            assert np.all(is_self.sum(axis=-1) == 1)
            assert np.all(w[is_self] == 1.0)
            assert np.all(w[~is_self] == 0.0)
        else:
            # Every pair is kept with d2 = 0, so every weight is 1 and
            # the window mean of identical patterns is the pattern, up
            # to the sequential float32 accumulation of n_window products
            bound = n_window * np.spacing(p)
            assert np.all(np.abs(out_f32 - p) <= bound)
            if dtype is np.uint8:
                # Rounding to nearest absorbs the accumulation error
                assert np.array_equal(out_default, data)
            else:
                assert np.all(np.abs(out_default - p) <= bound)

    @pytest.mark.parametrize("search_radius", [0, (0, 0)])
    def test_search_radius_zero_warns_and_is_a_no_op(
        self, caplog, random_uniform_saturated, search_radius
    ):
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data.copy())
        # The radius-0 check precedes the lambda optimisation, so the
        # default lam=None also returns quietly, without optimising
        for kwargs in [{"lam": 1.0}, {"lam": 1.0, "inplace": False}, {}]:
            with (
                warnings.catch_warnings(record=True) as record,
                caplog.at_level(logging.INFO, logger=NLPAR_LOGGER),
            ):
                caplog.clear()
                warnings.simplefilter("always")
                out = s.average_non_local_neighbour_patterns(
                    search_radius=search_radius, **kwargs
                )
            assert _info_records(caplog) == []
            messages = [
                str(r.message) for r in record if issubclass(r.category, UserWarning)
            ]
            assert any("no averaging is therefore performed" in m for m in messages)
            assert out is None
            assert np.array_equal(s.data, data)

        # ... but follows the validation of every other argument: a zero
        # radius with an invalid argument raises, without the warning
        for kwargs, message in [
            ({"lam": -1}, "lam must be > 0"),
            ({"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
            ({"dtype_out": bool}, "dtype_out must be an integer or floating dtype"),
        ]:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                with pytest.raises(ValueError, match=message):
                    s.average_non_local_neighbour_patterns(
                        search_radius=search_radius, **kwargs
                    )
            assert not any("no averaging" in str(r.message) for r in record)
            assert np.array_equal(s.data, data)

        # A per-axis radius (0, 2) averages along the columns
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = _average(s, search_radius=(0, 2), lam=1.0)
        assert not any("no averaging" in str(r.message) for r in record)
        assert not np.array_equal(out, data)
        assert np.array_equal(s.data, data)

    @pytest.mark.parametrize("lam", [0.7, 2.5])
    @pytest.mark.parametrize("search_radius", [1, 3])
    def test_injected_tiny_sigma_is_an_identity(
        self, random_uniform_saturated, search_radius, lam
    ):
        # sigma = 1e-6 gives dnorm = 2e-12 sqrt(2 n2) < 1e-8, so every
        # non-self pair takes d = 1e6 n2, whose weight exp(-1e6 n2 /
        # lam^2) is exactly 0.0 in float32; the self weight is 1
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data)
        kwargs = {"search_radius": search_radius, "lam": lam, "sigma": 1e-6}
        out_f32 = _average(s, dtype_out="float32", **kwargs)
        out_u8 = _average(s, **kwargs)
        s32 = kp.signals.EBSD(data.astype(np.float32))
        out_from_f32 = _average(s32, **kwargs)

        assert out_f32.dtype == np.float32
        assert np.array_equal(out_f32, data.astype(np.float32))
        assert out_u8.dtype == np.uint8
        assert np.array_equal(out_u8, data)
        assert out_from_f32.dtype == np.float32
        assert np.array_equal(out_from_f32, data.astype(np.float32))

    @pytest.mark.parametrize("search_radius, n_window", [(1, 9), ((1, 2), 15), (2, 20)])
    def test_huge_lambda_is_the_shifted_window_box_mean(
        self, random_uniform_saturated, search_radius, n_window
    ):
        # lam = 1e6: exp(-d / 1e12) rounds to 1.0 in float32 for |d| <
        # ~60 (the random map's |d| stays below 10), so the output is
        # the unweighted mean over the shifted-inward window. At radius
        # 1 both axes shift at every border point; at (1, 2) the rows
        # shift and the 5 columns are the whole axis; at 2 both axes
        # are the whole map
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data)
        out = _average(
            s, search_radius=search_radius, lam=1e6, dthresh=0.0, dtype_out="float32"
        )
        ref = shifted_box_mean(data, search_radius)

        if search_radius == 2:
            plain_mean = data.astype(np.float64).mean(axis=(0, 1))
            assert np.allclose(ref, plain_mean[None, None], rtol=0, atol=1e-9)
        else:
            # Test power: on the border band the shifted window differs
            # from the window clipped at the borders by grey levels
            clamped = clamped_box_mean(data, search_radius)
            assert np.max(np.abs(clamped - ref)) > 1.0

        assert out.dtype == np.float32
        # Sequential float32 accumulation of n_window products
        bound = n_window * np.spacing(ref.astype(np.float32))
        assert np.all(np.abs(out.astype(np.float64) - ref) <= bound)

    @pytest.mark.parametrize("exponent", [-16, -8, -4, 4])
    def test_power_of_two_scaling_is_exact(self, identical_plus_gaussian, exponent):
        # IEEE scaling by a power of two is exact through squares, sums,
        # ratios, square roots and the global saturation maximum, as
        # long as every value stays a normal float32. With noise sigma 8
        # and N = 1024 a neighbour pair has d2 ~ 1024 x 128 x 2^(2e):
        # ~2.0 at 2^-8, ~3e-5 at 2^-16. At 2^-16 every d2 of the map is
        # below 1e-3 (largest 3.3e-5), so an absolute duplicate guard d2
        # >= 1e-3 would drop every pair and give the 1e12 fallback, while
        # the scale-free d2 > 0 guard keeps them. The normaliser (s_i^2 +
        # s_j^2) sqrt(2 n2) is ~1.2e-6 at 2^-16 (~0.08 at 2^-8), far
        # above the 1e-8 guard
        data = identical_plus_gaussian((5, 6), (32, 32), sigma=8.0)
        scale = np.float32(2.0**exponent)
        scaled = (data * scale).astype(np.float32)
        assert np.array_equal(scaled / scale, data)

        sigma, d2, n2, valid = _whole_map_sigma_pass(data)
        sigma_s, d2_s, n2_s, valid_s = _whole_map_sigma_pass(scaled)
        if exponent == -16:
            # Test power: every neighbour d2 lies in (0, 1e-3)
            neighbour = valid_s.copy()
            neighbour[..., 4] = False
            assert np.all((d2_s[neighbour] > 0) & (d2_s[neighbour] < 1e-3))
        d3 = _nlpar_normalized_distances(d2, n2, valid, sigma)
        d3_s = _nlpar_normalized_distances(d2_s, n2_s, valid_s, sigma_s)

        d = _whole_map_search_distances(data, sigma, 2)
        d_s = _whole_map_search_distances(scaled, sigma_s, 2)
        w = _nlpar_weights_kernel(d, 1.0, np.float32(0.0))
        w_s = _nlpar_weights_kernel(d_s, 1.0, np.float32(0.0))

        out = _average(
            kp.signals.EBSD(data), search_radius=2, lam=1.0, dtype_out="float32"
        )
        out_s = _average(
            kp.signals.EBSD(scaled), search_radius=2, lam=1.0, dtype_out="float32"
        )

        assert np.array_equal(valid_s, valid)
        assert np.array_equal(n2_s, n2)
        assert np.array_equal(sigma_s, sigma * scale)
        assert np.array_equal(d2_s, d2 * scale * scale)
        # Normalised 3 x 3 distances and search-window distances and
        # weights are scale free, bitwise
        assert np.array_equal(d3_s[valid], d3[valid])
        assert np.array_equal(d_s, d)
        assert np.array_equal(w_s, w)
        # The averaged patterns scale exactly
        assert out_s.dtype == np.float32
        assert np.array_equal(out_s, out * scale)

    @pytest.mark.parametrize("lam", [0.7, 2.5])
    @pytest.mark.parametrize("dthresh", [0.0, 0.5])
    def test_weight_formula_on_injected_distances(self, exp_kernel_ulp, dthresh, lam):
        values = [-np.inf, -3.0, 0.0, 0.25, 0.5, 2.0, 50.0, np.inf]
        d = np.array(values, dtype=np.float32).reshape(1, 1, -1)
        w = _nlpar_weights_kernel(d, float(lam), np.float32(dthresh))

        assert w.dtype == np.float32
        assert w.shape == d.shape
        # A distance at or below dthresh (incl. the self slot -inf)
        # gives exactly 1, +inf exactly 0, and no weight exceeds 1
        assert np.all(w[d <= np.float32(dthresh)] == 1.0)
        assert w[0, 0, -1] == 0.0
        assert np.all((w >= 0.0) & (w <= 1.0))

        x = np.maximum(d.astype(np.float64) - dthresh, 0.0)
        expected = np.exp(-x / lam**2).astype(np.float32)
        ulp = int(_float32_ulp_distance(w, expected).max())
        _assert_at_most(ulp, exp_kernel_ulp, "EXP_KERNEL_ULP")

    @pytest.mark.parametrize("fixture_kind, kwargs, differs_from", REFERENCE_ARMS)
    def test_reference_agrees_with_the_method_on_random_maps(
        self, random_uniform_saturated, circle_mask, fixture_kind, kwargs, differs_from
    ):
        if fixture_kind == "random":
            data = _random_map(random_uniform_saturated)
        elif fixture_kind == "saturated":
            # 5 % of the pixels of every pattern at 255, the map maximum
            data = random_uniform_saturated(NAV_SHAPE, SIG_SHAPE)
        else:
            data = _random_map(random_uniform_saturated, nav_shape=(7,))
        sig_shape = data.shape[-2:]
        s = kp.signals.EBSD(data)
        method_kwargs = _resolve(kwargs, sig_shape, circle_mask)

        out_f32 = _average(s, dtype_out="float32", **method_kwargs)
        out_int = _average(s, **method_kwargs)
        others = [
            _average(s, dtype_out="float32", **_resolve(other, sig_shape, circle_mask))
            for other in differs_from
        ]
        ref = nlpar_reference(
            data,
            search_radius=method_kwargs["search_radius"],
            lam=method_kwargs["lam"],
            dthresh=method_kwargs.get("dthresh", 0.0),
            signal_mask=method_kwargs.get("signal_mask"),
            saturation_protect=method_kwargs.get("saturation_protect", True),
        )

        assert out_f32.dtype == np.float32
        assert out_f32.shape == data.shape
        assert out_int.dtype == np.uint8
        assert out_int.shape == data.shape
        ref_int = np.clip(np.rint(ref), 0, 255)
        # Hard bound: no pixel differs by 2 or more grey levels
        assert np.all(np.abs(out_int.astype(np.float64) - ref_int) < 2)
        # Every non-default keyword is seen to be forwarded, and the
        # per-axis radius is applied in (row, column) order
        for other in others:
            assert not np.array_equal(out_f32, other)

        max_abs = float(np.max(np.abs(out_f32.astype(np.float64) - ref)))
        _assert_at_most(max_abs, REFERENCE_MAX_ABS_GREY, "REFERENCE_MAX_ABS_GREY")
        # Away from a half-integer the integer output is the reference
        # rounded to nearest and clipped (a truncating cast is not)
        far = np.abs(ref - np.floor(ref) - 0.5) > REFERENCE_MAX_ABS_GREY
        assert np.array_equal(out_int[far], ref_int[far].astype(np.uint8))


# ------------------------------ iid noise --------------------------- #


def _noise_map(identical_plus_gaussian, **kwargs) -> np.ndarray:
    """Return the (12, 12 | 32, 32) float32 map of the ramp plus N(0, 8)
    noise, N = 1024 pixels per pattern.
    """
    return identical_plus_gaussian((12, 12), (32, 32), sigma=8.0, **kwargs)


class TestNoiseOracle:
    """Statistics predicted on identical patterns plus iid Gaussian
    noise of sigma_true = 8 (never 1, where sigma and sigma^2 coincide).
    """

    sigma_true = 8.0

    def test_sigma_recovery_median_ratio(self, identical_plus_gaussian):
        data = _noise_map(identical_plus_gaussian)
        sigma = kp.signals.EBSD(data).get_nlpar_sigma()

        assert sigma.shape == (12, 12)
        assert sigma.dtype == np.float32
        ratio = float(np.median(sigma[1:-1, 1:-1] / self.sigma_true))
        _assert_in_band(ratio, NOISE_SIGMA_RATIO_BAND, "NOISE_SIGMA_RATIO_BAND")

    def test_normalised_distance_moments(self, identical_plus_gaussian):
        data = _noise_map(identical_plus_gaussian)
        sigma, d2, n2, valid = _whole_map_sigma_pass(data)
        d = _nlpar_normalized_distances(d2, n2, valid, sigma)

        assert d.dtype == np.float32
        assert d.shape == (12, 12, 9)
        # The self slot (4) is -inf, i.e. weight exactly 1
        assert np.all(np.isneginf(d[..., 4]))
        neighbours = valid.copy()
        neighbours[..., 4] = False
        values = d[neighbours].astype(np.float64)
        assert np.all(np.isfinite(values))
        _assert_in_band(float(values.mean()), D_MEAN_BAND, "D_MEAN_BAND")
        _assert_in_band(float(values.std()), D_STD_BAND, "D_STD_BAND")

    def test_fraction_of_unit_weights(self, identical_plus_gaussian):
        data = _noise_map(identical_plus_gaussian)
        sigma = _whole_map_sigma_pass(data)[0]
        d = _whole_map_search_distances(data, sigma, 3)
        w = _nlpar_weights_kernel(d, 1.0, np.float32(0.0))

        neighbour_w = w[~np.isneginf(d)]
        assert np.all((neighbour_w >= 0.0) & (neighbour_w <= 1.0))
        fraction = float(np.mean(neighbour_w == 1.0))
        assert fraction > 0
        _assert_in_band(
            fraction, UNIT_WEIGHT_FRACTION_BAND, "UNIT_WEIGHT_FRACTION_BAND"
        )

    def test_noise_reduction_matches_the_weights(self, identical_plus_gaussian):
        # Every pattern of the (12, 12) map has a full 7 x 7 window
        # (shifted inward at the borders), so all 144 patterns count
        data = _noise_map(identical_plus_gaussian)
        s = kp.signals.EBSD(data)
        out = _average(s, search_radius=3, lam=1.0, dtype_out="float32")
        sigma = s.get_nlpar_sigma()
        d = _whole_map_search_distances(data, sigma, 3)
        w = _nlpar_weights_kernel(d, 1.0, np.float32(0.0)).astype(np.float64)

        w_normalised = w / w.sum(axis=-1, keepdims=True)
        expected = self.sigma_true**2 * float(np.mean(np.sum(w_normalised**2, axis=-1)))
        residual = out.astype(np.float64) - RAMP_32
        measured = float(np.var(residual))
        assert measured < self.sigma_true**2
        relative = abs(measured / expected - 1)
        _assert_at_most(relative, NOISE_REDUCTION_TOL, "NOISE_REDUCTION_TOL")

    def test_noise_reduction_is_monotone_in_lambda(self, identical_plus_gaussian):
        data = _noise_map(identical_plus_gaussian)
        s = kp.signals.EBSD(data)
        variances = []
        for lam in [0.5, 1.0, 2.0, 4.0]:
            out = _average(s, search_radius=3, lam=lam, dtype_out="float32")
            variances.append(float(np.var(out.astype(np.float64) - RAMP_32)))
        assert all(a > b for a, b in zip(variances, variances[1:])), variances

    def test_mask_polarity_separates(self, identical_plus_gaussian):
        # The right half of the columns of every pattern is three times
        # noisier (sigma 24); the mask excludes it. Protection is off on
        # both sides: the global maximum pixel, always excluded under
        # protection, moves when the right half is cropped away
        data = _noise_map(identical_plus_gaussian, sigma_right=24.0)
        s = kp.signals.EBSD(data)
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, 16:] = True

        sigma_masked = s.get_nlpar_sigma(signal_mask=mask, saturation_protect=False)
        s_left = s.isig[:16, :]
        assert s_left.data.shape == (12, 12, 32, 16)
        sigma_left = s_left.get_nlpar_sigma(saturation_protect=False)
        sigma_flipped = s.get_nlpar_sigma(signal_mask=~mask, saturation_protect=False)

        # Same pixels in the same order: bitwise
        assert np.array_equal(sigma_masked, sigma_left)
        # The flipped polarity sees the noisier half: a factor ~3 (24 /
        # 8), each estimate spreading by ~6 % at 512 pixels
        assert not np.array_equal(sigma_flipped, sigma_masked)
        assert np.all(sigma_flipped > 2 * sigma_masked)


# ------------------------------ Two grains -------------------------- #


def _two_grain_base(n_cols: int = 16) -> np.ndarray:
    """Return the noise-free base of every pattern of the two-grain
    map: the ramp in the left half of the navigation columns (grain A)
    and the ramp + 30 in the right half (grain B).
    """
    base = np.broadcast_to(RAMP_32, (10, n_cols, 32, 32)).copy()
    base[:, n_cols // 2 :] += TWO_GRAIN_DELTA
    return base


def _boundary_contrast(x: np.ndarray) -> float:
    """Return the mean over rows and pixels of the grain-B boundary
    column 8 minus the grain-A boundary column 7.
    """
    return float(np.mean(x[:, 8].astype(np.float64) - x[:, 7].astype(np.float64)))


class TestTwoGrain:
    """NLPAR does not average across a grain boundary that a Gaussian
    window blurs. Grain A is columns 0-7, grain B columns 8-15 of a
    (10, 16 | 32, 32) map, contrast 30 and noise sigma 8. The derived
    cross-boundary distance is about 159 at N = 1024, so -d / lam^2 is
    below the float32 exp underflow (between -103.9 and -104.0) for lam
    <= 1.2.
    """

    @pytest.mark.parametrize("lam", [0.7, 1.0])
    def test_cross_boundary_weights_are_exactly_zero(self, two_grain, lam):
        data = two_grain()
        n_rows, n_cols = data.shape[:2]
        radius = 3
        size = 2 * radius + 1
        sigma = _whole_map_sigma_pass(data)[0]
        d = _whole_map_search_distances(data, sigma, radius)
        w = _nlpar_weights_kernel(d, lam, np.float32(0.0))

        assert w.shape == (n_rows, n_cols, size * size)
        n_cross = 0
        for i in range(n_cols):
            window_cols = np.array(_reference_search_window(i, radius, n_cols))
            cross_cols = (window_cols < n_cols // 2) != (i < n_cols // 2)
            # Slot wr * size + wc: the grain of a neighbour is set by wc
            cross = np.tile(cross_cols, size)
            w_i = w[:, i]
            assert np.all(w_i[:, cross] == 0.0)
            n_cross += int(cross.sum()) * n_rows
            # Test power: same-grain neighbours keep non-zero weights
            same = ~cross & ~np.isneginf(d[:, i])
            assert np.any(w_i[same] > 0.0)
        assert n_cross > 0

    @pytest.mark.parametrize("lam", [0.7, 2.5])
    def test_contrast_is_retained(self, two_grain, lam):
        data = two_grain()
        s = kp.signals.EBSD(data)
        out = _average(s, search_radius=3, lam=lam, dtype_out="float32")

        # Comparison arm: the normalised (5, 5) Gaussian window (std 1)
        # along the navigation axes, no rescale. The boundary columns
        # mix 2 of their 5 window columns across the boundary, retaining
        # ~0.40 of the contrast
        window = np.asarray(kp.filters.Window("gaussian", (5, 5), std=1))
        window = window / window.sum()
        gaussian = correlate(data, weights=window[:, :, None, None])

        before = _boundary_contrast(data)
        nlpar_ratio = _boundary_contrast(out) / before
        gaussian_ratio = _boundary_contrast(gaussian) / before
        assert gaussian_ratio < 0.9
        assert nlpar_ratio > gaussian_ratio
        _assert_at_least(nlpar_ratio, TWO_GRAIN_CONTRAST_MIN, "TWO_GRAIN_CONTRAST_MIN")

    def test_boundary_columns_stay_within_the_within_grain_residual(self, two_grain):
        # Interior columns 3, 4 and 11, 12 have a 7-column window inside
        # their own grain
        data = two_grain()
        s = kp.signals.EBSD(data)
        out = _average(s, search_radius=3, lam=1.0, dtype_out="float32")

        residual = out.astype(np.float64) - _two_grain_base()
        rms_boundary = float(np.sqrt(np.mean(residual[:, [7, 8]] ** 2)))
        rms_interior = float(np.sqrt(np.mean(residual[:, [3, 4, 11, 12]] ** 2)))
        assert rms_interior < 8.0
        ratio = rms_boundary / rms_interior
        _assert_at_most(
            ratio, TWO_GRAIN_BOUNDARY_RESIDUAL_TOL, "TWO_GRAIN_BOUNDARY_RESIDUAL_TOL"
        )


# ------------------------- Lambda optimisation ---------------------- #

# Logger of the one INFO record per lambda optimisation
NLPAR_LOGGER = "kikuchipy.pattern._nlpar"
LAMBDA_RECORD = re.compile(r"NLPAR: optimised lambda ([0-9.]+) for target weight 0\.34")

# Bounds of the lambda search; a result within 1 % of one warns
LAMBDA_SEARCH_LOW = 1.01e-3
LAMBDA_SEARCH_HIGH = 9.9


def _info_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the INFO records of the NLPAR module logger."""
    return [
        record
        for record in caplog.records
        if record.name == NLPAR_LOGGER and record.levelno == logging.INFO
    ]


def _nickel_corrected() -> kp.signals.EBSD:
    """Return ``nickel_ebsd_large`` with the static and then the
    dynamic background removed, both with the default arguments.
    """
    s = kp.data.nickel_ebsd_large(allow_download=True)
    s.remove_static_background(show_progressbar=False)
    s.remove_dynamic_background(show_progressbar=False)
    return s


def _whole_map_lambda_distances(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalised 3 x 3 distances and the in-map slot mask
    of the sigma pass on a whole 2D map, every pixel kept, protection
    on.
    """
    sigma, d2, n2, valid = _whole_map_sigma_pass(data)
    return _nlpar_normalized_distances(d2, n2, valid, sigma), valid


def _phantom_counting_lambda(
    d: np.ndarray, valid: np.ndarray, target_weight: float = 0.34
) -> float:
    """Return the lambda of a test-local transcription of PyEBSDIndex's
    objective ``loptfunc``, in which every out-of-map slot counts as a
    neighbour of weight 1 (its distance is 0.0, as in PyEBSDIndex's
    ``dout``) and the distance enters as ``max(d, dthresh)`` with
    ``dthresh = 0``, minimised with PyEBSDIndex's optimiser settings.

    Slots without comparable pixels keep their ``+inf`` (weight 0), so
    only the out-of-map slots separate this from the phantom-free
    objective.
    """
    d_phantom = np.where(valid, d, np.float32(0.0)).astype(np.float32)

    def loptfunc(lam, d2, tw, dthresh):
        temp = np.maximum(d2, dthresh)
        dw = np.exp(-temp / lam**2)
        w = np.sum(dw, axis=2) + 1e-12
        return np.mean(np.abs(tw - 1.0 / w))

    result = minimize(
        loptfunc,
        1.0,
        args=(d_phantom, target_weight, np.float32(0.0)),
        method="Nelder-Mead",
        bounds=[[0.001, 10.0]],
        options={"fatol": 0.0001},
    )
    return float(result.x[0])


class TestLambdaMethod:
    """Contracts of :meth:`~kikuchipy.signals.EBSD.get_nlpar_lambda`
    and of ``lam=None`` in
    :meth:`~kikuchipy.signals.EBSD.average_non_local_neighbour_patterns`.
    """

    def test_get_nlpar_lambda_on_nickel_ebsd_large(self, caplog, record_property):
        signals = {
            "raw": kp.data.nickel_ebsd_large(allow_download=True),
            "corrected": _nickel_corrected(),
        }
        measured = {}
        for name, s in signals.items():
            data = s.data.copy()
            with caplog.at_level(logging.INFO, logger=NLPAR_LOGGER):
                caplog.clear()
                lam = s.get_nlpar_lambda(show_progressbar=False)
                records = _info_records(caplog)
            assert isinstance(lam, float)
            assert len(records) == 1, name
            assert LAMBDA_SEARCH_LOW < lam < LAMBDA_SEARCH_HIGH
            record_property(f"lambda_ni_{name}", lam)

            # The optimiser on the distances of the sigma pass of the
            # whole map, with the default target weight and threshold
            d, valid = _whole_map_lambda_distances(data)
            assert (
                nlpar_module._nlpar_optimize_lambda(d, valid, 0.34, np.float32(0.0))
                == lam
            )

            # Recorded deviation: PyEBSDIndex counts the 776 out-of-map
            # slots of this 55 x 75 map as neighbours of weight 1, which
            # makes border patterns look more self-similar and lowers
            # lambda (seeds +2.0 % raw and +2.1 % corrected)
            assert np.sum(~valid) == 776
            lam_phantom = _phantom_counting_lambda(d, valid)
            record_property(f"lambda_ni_{name}_phantom_counting", lam_phantom)
            record_property(f"lambda_ni_{name}_phantom_ratio", lam / lam_phantom)
            assert lam > lam_phantom

            # Lambda decreases with the target weight, which is why
            # PyEBSDIndex's median of the fits to 0.5, 0.34 and 0.25 is
            # the fit to 0.34
            lam_05 = s.get_nlpar_lambda(target_weight=0.5, show_progressbar=False)
            lam_025 = s.get_nlpar_lambda(target_weight=0.25, show_progressbar=False)
            assert lam_05 < lam < lam_025
            assert np.array_equal(s.data, data)
            measured[name] = lam

        # lam=None optimises the same value, logs it once and averages
        # with it
        s = signals["corrected"]
        kwargs = {"dtype_out": "float32", "show_progressbar": False}
        with caplog.at_level(logging.INFO, logger=NLPAR_LOGGER):
            caplog.clear()
            out_none = _average(s, lam=None, **kwargs)
            records = _info_records(caplog)
        assert len(records) == 1
        match = LAMBDA_RECORD.search(records[0].getMessage())
        assert match is not None, records[0].getMessage()
        assert match.group(1) == f"{measured['corrected']:.4f}"
        out_given = _average(s, lam=measured["corrected"], **kwargs)
        assert np.array_equal(out_none, out_given)

        _assert_in_band(measured["raw"], LAMBDA_NI_RAW, "LAMBDA_NI_RAW")
        _assert_in_band(
            measured["corrected"], LAMBDA_NI_CORRECTED, "LAMBDA_NI_CORRECTED"
        )

    def test_lam_none_logs_the_optimised_lambda(
        self, caplog, capsys, identical_plus_gaussian
    ):
        s = kp.signals.EBSD(identical_plus_gaussian((12, 12), (32, 32)))
        kwargs = {"dtype_out": "float32", "show_progressbar": False}
        capsys.readouterr()
        with caplog.at_level(logging.INFO, logger=NLPAR_LOGGER):
            caplog.clear()
            out_none = _average(s, lam=None, **kwargs)
            records_average = _info_records(caplog)
            caplog.clear()
            lam = s.get_nlpar_lambda(show_progressbar=False)
            records_lambda = _info_records(caplog)
        # Reported through logging, never printed
        assert capsys.readouterr().out == ""

        # Exactly one record per call, of the value get_nlpar_lambda
        # returns
        assert len(records_average) == 1
        assert len(records_lambda) == 1
        for record in records_average + records_lambda:
            match = LAMBDA_RECORD.search(record.getMessage())
            assert match is not None, record.getMessage()
            assert match.group(1) == f"{lam:.4f}"
        assert LAMBDA_SEARCH_LOW < lam < LAMBDA_SEARCH_HIGH
        assert np.array_equal(out_none, _average(s, lam=lam, **kwargs))

    def test_target_weight_is_forwarded_through_lam_none(self, identical_plus_gaussian):
        s = kp.signals.EBSD(identical_plus_gaussian((12, 12), (32, 32)))
        kwargs = {"dtype_out": "float32"}
        lam_05 = s.get_nlpar_lambda(target_weight=0.5)
        lam_034 = s.get_nlpar_lambda()
        lam_025 = s.get_nlpar_lambda(target_weight=0.25)
        # Lambda decreases with the target weight
        assert lam_05 < lam_034 < lam_025
        assert lam_034 == s.get_nlpar_lambda(target_weight=0.34)

        out_none_05 = _average(s, lam=None, target_weight=0.5, **kwargs)
        assert np.array_equal(out_none_05, _average(s, lam=lam_05, **kwargs))
        out_none_034 = _average(s, lam=None, **kwargs)
        assert np.array_equal(out_none_034, _average(s, lam=lam_034, **kwargs))
        # The keyword reaches the optimiser: one fit to the given
        # target, neither a fixed 0.34 nor a median of three fits
        assert not np.array_equal(out_none_05, out_none_034)

    def test_get_nlpar_lambda_accepts_injected_sigma_and_mask(
        self, identical_plus_gaussian, circle_mask
    ):
        data = identical_plus_gaussian((12, 12), (32, 32), dtype=np.uint8)
        # A saturated corner in every pattern, so saturation protection
        # changes the distances
        data[..., :2, :2] = 255
        s = kp.signals.EBSD(data.copy())
        lam = s.get_nlpar_lambda()
        sigma = s.get_nlpar_sigma()

        # The sigma map of get_nlpar_sigma reproduces sigma=None exactly
        assert s.get_nlpar_lambda(sigma=sigma) == lam
        assert s.get_nlpar_lambda(sigma=sigma.astype(np.float64)) == lam
        # Another sigma map changes the distances and lambda
        assert s.get_nlpar_lambda(sigma=sigma * np.float32(1.5)) != lam
        # A scalar is a constant map
        lam_scalar = s.get_nlpar_lambda(sigma=8.0)
        lam_constant = s.get_nlpar_lambda(sigma=np.full((12, 12), 8.0, np.float32))
        assert lam_scalar == lam_constant
        assert lam_scalar != lam

        # The mask and the protection reach the sigma pass and the
        # distances
        mask = circle_mask((32, 32))
        lam_mask = s.get_nlpar_lambda(signal_mask=mask)
        assert lam_mask != lam
        sigma_mask = s.get_nlpar_sigma(signal_mask=mask)
        assert s.get_nlpar_lambda(signal_mask=mask, sigma=sigma_mask) == lam_mask
        assert s.get_nlpar_lambda(signal_mask=mask.astype(np.int64)) == lam_mask
        lam_unprotected = s.get_nlpar_lambda(saturation_protect=False)
        assert lam_unprotected != lam
        sigma_unprotected = s.get_nlpar_sigma(saturation_protect=False)
        assert (
            s.get_nlpar_lambda(saturation_protect=False, sigma=sigma_unprotected)
            == lam_unprotected
        )

        # lam=None uses the same mask, sigma and protection as the
        # averaging that follows
        kwargs = {"signal_mask": mask, "dtype_out": "float32"}
        out_none = _average(s, lam=None, **kwargs)
        assert np.array_equal(out_none, _average(s, lam=lam_mask, **kwargs))
        assert np.array_equal(s.data, data)

    def test_lambda_forwards_dthresh_sigma_and_protection(
        self, identical_plus_gaussian
    ):
        # The saturated-corner map of the test above. Reference values
        # of the optimiser on its sigma-pass distances (drafting
        # measurement 2026-10-05, a NumPy transcription of the
        # objective): 0.8926 at dthresh 0, 0.6229 at dthresh 0.5, 0.8935
        # without protection and 1.0 (the start, weights flat in
        # lambda) with sigma x 1.5, so every keyword below moves lambda
        data = identical_plus_gaussian((12, 12), (32, 32), dtype=np.uint8)
        data[..., :2, :2] = 255
        s = kp.signals.EBSD(data.copy())
        lam = s.get_nlpar_lambda()

        # dthresh reaches the objective, with the same meaning as in
        # the averaging weights
        lam_dthresh = s.get_nlpar_lambda(dthresh=0.5)
        assert lam_dthresh != lam
        d, valid = _whole_map_lambda_distances(data)
        assert lam_dthresh == nlpar_module._nlpar_optimize_lambda(
            d, valid, 0.34, np.float32(0.5)
        )

        # lam=None optimises with the dthresh, sigma and protection of
        # the averaging that follows, not with their defaults
        sigma = s.get_nlpar_sigma()
        sigma_scaled = sigma * np.float32(1.5)
        lam_sigma = s.get_nlpar_lambda(sigma=sigma_scaled)
        lam_unprotected = s.get_nlpar_lambda(saturation_protect=False)
        for kwargs, lam_forwarded in [
            ({"dthresh": 0.5}, lam_dthresh),
            ({"sigma": sigma_scaled}, lam_sigma),
            ({"saturation_protect": False}, lam_unprotected),
        ]:
            assert lam_forwarded != lam, kwargs
            kwargs = {**kwargs, "dtype_out": "float32"}
            out_none = _average(s, lam=None, **kwargs)
            assert np.array_equal(out_none, _average(s, lam=lam_forwarded, **kwargs))
            assert not np.array_equal(out_none, _average(s, lam=lam, **kwargs))
        assert np.array_equal(s.data, data)


# ------------------------- Contracts (eager) ------------------------ #

_NAN_SIGMA = np.ones(NAV_SHAPE, dtype=np.float32)
_NAN_SIGMA[1, 2] = np.nan
# One element whose float32 square is finite but whose 2 n sigma^2 (n =
# 36 kept pixels) is not
_HUGE_SIGMA = np.ones(NAV_SHAPE, dtype=np.float32)
_HUGE_SIGMA[2, 3] = 1e19

# Fragment of the message of the float32 range check of a given sigma
SIGMA_RANGE_MESSAGE = "sigma must be > 0 with sigma"

# (method, keyword arguments, expected): expected is the fragment of
# the ValueError message, or keyword arguments of an accepted run that
# must give the same output bitwise. The methods "average_0d" and
# "sigma_0d" run on a signal without navigation axes. The mask names
# "circle" and "circle_int" are resolved by _resolve.
VALIDATION_ARMS = [
    ("average", {"search_radius": -1}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": 1.5}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": True}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": (1, 2, 3)}, "one radius per navigation axis"),
    ("average", {"search_radius": np.int64(2)}, {"search_radius": 2}),
    ("average", {"lam": 0.0}, "lam must be > 0"),
    ("average", {"lam": -1}, "lam must be > 0"),
    ("average", {"lam": np.inf}, "lam must be > 0"),
    ("average", {"dthresh": -0.1}, "dthresh must be >= 0"),
    ("average", {"dthresh": np.nan}, "dthresh must be >= 0"),
    ("average", {"dthresh": np.inf}, "dthresh must be >= 0"),
    ("average", {"target_weight": 0.0}, "0 < target_weight < 1"),
    ("average", {"target_weight": 1.0}, "0 < target_weight < 1"),
    ("average", {"target_weight": True}, "0 < target_weight < 1"),
    ("average", {"sigma": 0.0}, "sigma must be > 0"),
    ("average", {"sigma": -1.0}, "sigma must be > 0"),
    ("average", {"sigma": True}, "sigma must be > 0"),
    ("average", {"sigma": 1e39}, "sigma must be > 0"),
    ("average", {"sigma": 1e19}, SIGMA_RANGE_MESSAGE),
    ("average", {"sigma": 1e-30}, SIGMA_RANGE_MESSAGE),
    ("average", {"sigma": _HUGE_SIGMA}, SIGMA_RANGE_MESSAGE),
    ("average", {"sigma": np.array(8.0)}, {"sigma": 8.0}),
    ("average", {"sigma": np.zeros(NAV_SHAPE, dtype=np.float32)}, "sigma must be > 0"),
    ("average", {"sigma": _NAN_SIGMA}, "sigma must be > 0"),
    ("average", {"sigma": np.ones((2, 2))}, "navigation shape"),
    ("average", {"dtype_out": bool}, "dtype_out must be an integer or floating dtype"),
    (
        "average",
        {"dtype_out": complex},
        "dtype_out must be an integer or floating dtype",
    ),
    ("average", {"dtype_out": "foo"}, "dtype_out must be an integer or floating dtype"),
    ("average", {"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
    (
        "average",
        {"signal_mask": np.ones(SIG_SHAPE, dtype=bool)},
        "excludes every pixel",
    ),
    ("average", {"signal_mask": "circle_int"}, {"signal_mask": "circle"}),
    ("average_0d", {}, "nothing to average"),
    ("average_0d", {"search_radius": 3}, "nothing to average"),
    ("sigma_0d", {}, "nothing to average"),
    ("sigma", {"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
    ("sigma", {"signal_mask": np.ones(SIG_SHAPE, dtype=bool)}, "excludes every pixel"),
    ("sigma", {"signal_mask": "circle_int"}, {"signal_mask": "circle"}),
    ("lambda", {"target_weight": 0.0}, "0 < target_weight < 1"),
    ("lambda", {"target_weight": 1.0}, "0 < target_weight < 1"),
    ("lambda", {"target_weight": True}, "0 < target_weight < 1"),
    ("lambda", {"dthresh": -0.1}, "dthresh must be >= 0"),
    ("lambda", {"dthresh": np.nan}, "dthresh must be >= 0"),
    ("lambda", {"sigma": 0.0}, "sigma must be > 0"),
    ("lambda", {"sigma": -1.0}, "sigma must be > 0"),
    ("lambda", {"sigma": True}, "sigma must be > 0"),
    ("lambda", {"sigma": np.zeros(NAV_SHAPE, dtype=np.float32)}, "sigma must be > 0"),
    ("lambda", {"sigma": _NAN_SIGMA}, "sigma must be > 0"),
    ("lambda", {"sigma": 1e19}, SIGMA_RANGE_MESSAGE),
    ("lambda", {"sigma": np.ones((2, 2))}, "navigation shape"),
    ("lambda", {"sigma": np.array(8.0)}, {"sigma": 8.0}),
    ("lambda", {"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
    ("lambda", {"signal_mask": np.ones(SIG_SHAPE, dtype=bool)}, "excludes every pixel"),
    ("lambda", {"signal_mask": "circle_int"}, {"signal_mask": "circle"}),
    ("lambda_0d", {}, "nothing to average"),
]


def _assert_balanced_registration(events: list[str]) -> None:
    """Assert that progress bar events start with a registration, end
    with an unregistration and never unregister more than registered.
    """
    assert events
    assert events[0] == "register"
    assert events[-1] == "unregister"
    depth = 0
    for event in events:
        depth += 1 if event == "register" else -1
        assert depth >= 0
    assert depth == 0


# Input navigation chunkings of the lazy == eager arms on the
# (10, 16 | 16, 16) map, as (id, chunks, processed rows):
# get_dask_array(rechunk=True) keeps the row chunks and makes the 16
# columns one chunk (measured 2026-10-04 and 2026-10-05, dask 2026.3.0),
# so these arms pin the row chunkings; the column and both-axes
# chunkings reach the drivers directly in their own tests
LAZY_CHUNKINGS = [
    ("single", (-1, -1), (10,)),
    ("(5, 8)", (5, 8), (5, 5)),
    ("((3, 3, 4), (7, 7, 2))", ((3, 3, 4), (7, 7, 2)), (3, 3, 4)),
    ("((1, 9), (2, 14))", ((1, 9), (2, 14)), (1, 9)),
    ("(1, 1)", (1, 1), (1,) * 10),
    ("(2, 3)", (2, 3), (2,) * 5),
    ("(4, 4)", (4, 4), (4, 4, 2)),
]


class TestLazyAndContracts:
    """The method contract inherited from ``average_neighbour_patterns``
    on in-memory and lazy signals, lazy == eager over chunkings and
    schedulers, the dtype policy, 1D navigation, small maps and
    argument validation.
    """

    @pytest.mark.parametrize("method, kwargs, expected", VALIDATION_ARMS)
    def test_argument_validation(
        self, random_uniform_saturated, circle_mask, method, kwargs, expected
    ):
        data = _random_map(random_uniform_saturated)
        if method.endswith("_0d"):
            data = data[0, 0]
            method = method.removesuffix("_0d")
        s = kp.signals.EBSD(data.copy())
        kwargs = _resolve(kwargs, SIG_SHAPE, circle_mask)
        if isinstance(expected, dict):
            expected = _resolve(expected, SIG_SHAPE, circle_mask)

        def call(**kw):
            kw = dict(kw)
            if method == "average":
                kw.setdefault("lam", 1.0)
                return _average(s, **kw)
            if method == "lambda":
                return s.get_nlpar_lambda(**kw)
            return s.get_nlpar_sigma(**kw)

        if isinstance(expected, str):
            with pytest.raises(ValueError, match=expected):
                call(**kwargs)
        else:
            assert np.array_equal(call(**kwargs), call(**expected))
        assert np.array_equal(s.data, data)

    @pytest.mark.parametrize("route", ["eager", "lazy"])
    @pytest.mark.parametrize("nav_shape", [(2, 2), (3, 5)])
    def test_map_smaller_than_the_window(
        self, random_uniform_saturated, nav_shape, route
    ):
        # At radius 3 both axes are shorter than the 7-point window, so
        # every pattern averages over the whole map (PyEBSDIndex would
        # index out of bounds here); padding slots must carry weight 0
        data = _random_map(random_uniform_saturated, nav_shape=nav_shape)
        s = kp.signals.EBSD(data)
        kwargs = {"search_radius": 3, "lam": 1.0, "dtype_out": "float32"}
        out = _average(s, **kwargs)
        ref = nlpar_reference(data, search_radius=3, lam=1.0, dthresh=0.0)

        assert out.dtype == np.float32
        assert out.shape == data.shape
        assert np.all(np.isfinite(out))
        max_abs = float(np.max(np.abs(out.astype(np.float64) - ref)))
        _assert_at_most(max_abs, REFERENCE_MAX_ABS_GREY, "REFERENCE_MAX_ABS_GREY")

        if route == "lazy":
            # One row per chunk: every axis is rechunked to one chunk,
            # the window being longer than the axis
            s_lazy = _lazy_signal(data, (1, -1))
            out_lazy = _average_lazy(s_lazy, **kwargs)
            assert out_lazy.dtype == np.float32
            assert np.array_equal(out_lazy, out)

    @pytest.mark.parametrize("route", ["eager", "lazy"])
    def test_one_dimensional_navigation_equals_a_one_row_map(
        self, random_uniform_saturated, route
    ):
        data_1d = _random_map(random_uniform_saturated, nav_shape=(7,))
        s_1d = kp.signals.EBSD(data_1d)
        s_2d = kp.signals.EBSD(data_1d[None])
        assert s_1d.axes_manager.navigation_dimension == 1
        assert s_2d._navigation_shape_rc == (1, 7)

        for dtype_out in [None, "float32"]:
            kwargs = {"search_radius": 2, "lam": 1.0, "dtype_out": dtype_out}
            out_1d = _average(s_1d, **kwargs)
            out_2d = _average(s_2d, **kwargs)
            assert out_1d.shape == data_1d.shape
            assert np.array_equal(out_1d, out_2d[0])
            assert not np.array_equal(out_1d.astype(np.float32), data_1d)

            if route == "lazy":
                # A lazy 1D scan and the lazy one-row map, both given
                # in two chunks, (3, 4): the method merges the short
                # scan into one chunk, so this covers the single-chunk
                # 1D route
                s_1d_lazy = _lazy_signal(data_1d, ((3, 4),))
                processed = get_dask_array(
                    signal=s_1d_lazy, chunk_bytes=8e6, rechunk=True
                ).chunks
                assert processed[0] == (7,)
                processed = get_dask_array(
                    signal=_lazy_signal(data_1d[None], (1, (3, 4))),
                    chunk_bytes=8e6,
                    rechunk=True,
                ).chunks
                assert processed[:2] == ((1,), (7,))
                out_1d_lazy = _average_lazy(s_1d_lazy, **kwargs)
                assert out_1d_lazy.shape == data_1d.shape
                assert np.array_equal(out_1d_lazy, out_1d)
                s_2d_lazy = _lazy_signal(data_1d[None], (1, (3, 4)))
                assert np.array_equal(_average_lazy(s_2d_lazy, **kwargs), out_2d)

        if route == "lazy":
            sigma_1d_lazy = _lazy_signal(data_1d, ((3, 4),)).get_nlpar_sigma()
            assert sigma_1d_lazy.shape == (7,)
            assert sigma_1d_lazy.dtype == np.float32
            assert np.array_equal(sigma_1d_lazy, s_1d.get_nlpar_sigma())

        # A 1-tuple radius is the int radius of the one axis
        out_int = _average(s_1d, search_radius=2, lam=1.0, dtype_out="float32")
        out_tuple = _average(s_1d, search_radius=(2,), lam=1.0, dtype_out="float32")
        assert np.array_equal(out_tuple, out_int)

        sigma_1d = s_1d.get_nlpar_sigma()
        sigma_2d = s_2d.get_nlpar_sigma()
        assert sigma_1d.shape == (7,)
        assert sigma_1d.dtype == np.float32
        assert np.array_equal(sigma_1d, sigma_2d[0])

    @pytest.mark.parametrize(
        "dtype, shifted",
        [
            (np.uint8, False),
            (np.uint16, False),
            (np.float32, False),
            (np.float64, False),
            (np.float32, True),
        ],
        ids=["uint8", "uint16", "float32", "float64", "float32-shifted"],
    )
    def test_dtype_round_trip(self, two_grain, dtype, shifted):
        data = two_grain(nav_shape=(6, 8), sig_shape=(16, 16), dtype=dtype)
        if shifted:
            # As after a background correction: 1.6 x - 80 maps the base
            # range [40, 230] to [-16, 288], beyond the uint8 range at
            # both ends and beyond the int8 range at the top
            data = data * np.float32(1.6) - np.float32(80.0)
            assert data.dtype == np.float32
        s = kp.signals.EBSD(data.copy())
        kwargs = {"search_radius": 3, "lam": 1.0}
        out_default = _average(s, **kwargs)
        out_f32 = _average(s, dtype_out="float32", **kwargs)
        out_u16 = _average(s, dtype_out=np.uint16, **kwargs)
        out_f64 = _average(s, dtype_out="float64", **kwargs)
        out_u8 = _average(s, dtype_out=np.uint8, **kwargs)
        out_i8 = _average(s, dtype_out=np.int8, **kwargs)

        assert np.array_equal(s.data, data)
        # Integer targets other than the input type: rint of the float32
        # average, clipped to the range of the output type (not of the
        # input type, which is (-1, 1) for a float input)
        assert out_u8.dtype == np.uint8
        expected_u8 = np.clip(np.rint(out_f32), 0, 255).astype(np.uint8)
        assert np.array_equal(out_u8, expected_u8)
        assert out_i8.dtype == np.int8
        expected_i8 = np.clip(np.rint(out_f32), -128, 127).astype(np.int8)
        assert np.array_equal(out_i8, expected_i8)
        if shifted:
            # Test power: the clip binds at both ends of the uint8 range,
            # where a cast without it wraps around
            rounded = np.rint(out_f32)
            assert np.any(rounded < 0)
            assert np.any(rounded > 255)
            assert not np.array_equal(out_u8, rounded.astype(np.int64).astype(np.uint8))
        assert out_default.dtype == np.dtype(dtype)
        assert out_f32.dtype == np.float32
        # Integer targets: rint of the float32 average, clipped to the
        # range of the data type; float targets: a cast
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            expected = np.clip(np.rint(out_f32), info.min, info.max).astype(dtype)
            assert np.array_equal(out_default, expected)
            # Test power: rounding to nearest differs from a truncating
            # cast on a large fraction of the pixels (~0.5 expected)
            assert np.mean(np.rint(out_f32) != np.floor(out_f32)) > 0.25
        else:
            assert np.array_equal(out_default, out_f32.astype(dtype))
        expected_u16 = np.clip(np.rint(out_f32), 0, 65535).astype(np.uint16)
        assert out_u16.dtype == np.uint16
        assert np.array_equal(out_u16, expected_u16)
        assert out_f64.dtype == np.float64
        assert np.array_equal(out_f64, out_f32.astype(np.float64))
        # No per-pattern rescale: a convex combination keeps the mean
        mean_in = float(data.astype(np.float64).mean())
        assert abs(float(out_default.astype(np.float64).mean()) - mean_in) <= 0.5
        assert abs(float(out_f32.astype(np.float64).mean()) - mean_in) <= 0.5

    def test_inplace_with_dtype_out_changes_the_data_dtype(
        self, random_uniform_saturated
    ):
        data = _random_map(random_uniform_saturated)
        kwargs = {"search_radius": 3, "lam": 1.0}

        s = kp.signals.EBSD(data.copy())
        expected_f32 = _average(s, dtype_out="float32", **kwargs)
        out = s.average_non_local_neighbour_patterns(dtype_out="float32", **kwargs)
        assert out is None
        assert s.data.dtype == np.float32
        assert np.array_equal(s.data, expected_f32)

        s2 = kp.signals.EBSD(data.copy())
        expected_u8 = _average(s2, **kwargs)
        s2.average_non_local_neighbour_patterns(**kwargs)
        assert s2.data.dtype == np.uint8
        assert np.array_equal(s2.data, expected_u8)
        assert not np.array_equal(s2.data, data)

        # Anchors independent of the integer output route: rint of the
        # float32 average clipped to the uint8 range (a truncating cast
        # differs), and no per-pattern rescale. At radius 3 the window
        # of every pattern of the (4, 5) map is the whole map, and the
        # weights sum to 1, so every output pixel lies between the
        # smallest and the largest input value of that pixel over the
        # map (the float32 error, below 1e-3 grey levels for 20 terms
        # of at most 239, vanishes in rint at integer bounds)
        assert np.array_equal(
            s2.data, np.clip(np.rint(expected_f32), 0, 255).astype(np.uint8)
        )
        assert np.all(s2.data >= data.min(axis=(0, 1)))
        assert np.all(s2.data <= data.max(axis=(0, 1)))

    @pytest.mark.parametrize("route", ["eager", "lazy"])
    def test_inplace_lazy_output_contract(self, random_uniform_saturated, route):
        data = _random_map(random_uniform_saturated)
        if route == "lazy":
            self._lazy_arms_of_the_inplace_lazy_output_contract(data)
            return
        s = kp.signals.EBSD(data.copy())
        message = "'lazy_output=True' requires 'inplace=False'"
        # Checked before anything else
        with pytest.raises(ValueError, match=message):
            s.average_non_local_neighbour_patterns(lam=1.0, lazy_output=True)
        with pytest.raises(ValueError, match=message):
            s.average_non_local_neighbour_patterns(lazy_output=True)
        with pytest.raises(ValueError, match=message):
            s.average_non_local_neighbour_patterns(search_radius=-1, lazy_output=True)
        assert np.array_equal(s.data, data)

        s_out = s.average_non_local_neighbour_patterns(lam=1.0, inplace=False)
        assert isinstance(s_out, kp.signals.EBSD)
        assert not s_out._lazy
        assert np.array_equal(s.data, data)
        assert not np.array_equal(s_out.data, data)

        returned = s.average_non_local_neighbour_patterns(lam=1.0)
        assert returned is None
        assert np.array_equal(s.data, s_out.data)

    @staticmethod
    def _lazy_arms_of_the_inplace_lazy_output_contract(data: np.ndarray) -> None:
        """Assert the lazy arms of the inplace and lazy_output contract
        on a map of shape (4, 5 | 6, 6), against the eager run.
        """
        kwargs = {"search_radius": 1, "lam": 1.0}
        expected = _average(kp.signals.EBSD(data.copy()), **kwargs)
        assert not np.array_equal(expected, data)
        nav_chunks = ((2, 2), (5,))

        # An in-memory signal with lazy_output=True gives a lazy signal
        s = kp.signals.EBSD(data.copy())
        s_out = s.average_non_local_neighbour_patterns(
            inplace=False, lazy_output=True, **kwargs
        )
        assert isinstance(s_out, kp.signals.LazyEBSD)
        assert np.array_equal(s.data, data)
        s_out.compute()
        assert isinstance(s_out, kp.signals.EBSD)
        assert np.array_equal(s_out.data, expected)

        # A lazy input stays lazy unless lazy_output=False
        s_lazy = _lazy_signal(data, nav_chunks)
        s_out = s_lazy.average_non_local_neighbour_patterns(inplace=False, **kwargs)
        assert isinstance(s_out, kp.signals.LazyEBSD)
        assert np.array_equal(s_out.data.compute(), expected)
        s_out = s_lazy.average_non_local_neighbour_patterns(
            inplace=False, lazy_output=False, **kwargs
        )
        assert isinstance(s_out, kp.signals.EBSD)
        assert not s_out._lazy
        assert np.array_equal(s_out.data, expected)
        assert s_lazy._lazy
        assert np.array_equal(s_lazy.data.compute(), data)

        # In place, a lazy input keeps its chunks and stays lazy
        s_lazy = _lazy_signal(data, nav_chunks)
        old_chunks = s_lazy.data.chunks
        returned = s_lazy.average_non_local_neighbour_patterns(**kwargs)
        assert returned is None
        assert s_lazy._lazy
        assert isinstance(s_lazy.data, da.Array)
        assert s_lazy.data.chunks == old_chunks
        assert np.array_equal(s_lazy.data.compute(), expected)

        # In place with lazy_output=False, the data is computed into an
        # in-memory array
        s_lazy = _lazy_signal(data, nav_chunks)
        returned = s_lazy.average_non_local_neighbour_patterns(
            lazy_output=False, **kwargs
        )
        assert returned is None
        assert isinstance(s_lazy.data, np.ndarray)
        assert np.array_equal(s_lazy.data, expected)

        # lazy_output=True still requires inplace=False on a lazy input
        with pytest.raises(ValueError, match="'lazy_output=True' requires"):
            _lazy_signal(data, nav_chunks).average_non_local_neighbour_patterns(
                lazy_output=True, **kwargs
            )

    @pytest.mark.parametrize(
        "nav_chunks, processed_rows",
        [chunking[1:] for chunking in LAZY_CHUNKINGS],
        ids=[chunking[0] for chunking in LAZY_CHUNKINGS],
    )
    def test_lazy_equals_eager_chunking(
        self, counting_spy, random_uniform_saturated, nav_chunks, processed_rows
    ):
        # Saturated pixels in the first (5, 8) navigation block only, so
        # a saturation maximum taken per chunk differs from the global
        # one on every other block
        x = random_uniform_saturated((10, 16), (16, 16), one_block_only=True)
        assert x[:5, :8].max() == 255
        assert x[5:].max() < 255
        assert x[:, 8:].max() < 255
        s = kp.signals.EBSD(x.copy())
        s_lazy = _lazy_signal(x, nav_chunks)

        # What the method processes: the row chunks of the input, the
        # 16 columns in one chunk
        processed = get_dask_array(signal=s_lazy, chunk_bytes=8e6, rechunk=True).chunks
        assert processed[:2] == (processed_rows, (16,))

        sigma_lazy = s_lazy.get_nlpar_sigma()
        assert sigma_lazy.dtype == np.float32
        assert np.array_equal(sigma_lazy, s.get_nlpar_sigma())

        average_calls = counting_spy(nlpar_module, "_nlpar_average_chunk")
        for search_radius in (1, 3):
            kwargs = {"search_radius": search_radius, "lam": 1.0}
            expected = _average(s, dtype_out="float32", **kwargs)
            average_calls.clear()
            out = _average_lazy(s_lazy, dtype_out="float32", **kwargs)
            # Through the chunked averaging pass, one wrapper call per
            # block of the depth-rechunked processed chunks
            chunks_out = nlpar_module._nlpar_depth(
                processed, (search_radius, search_radius)
            )[1]
            assert len(average_calls) == len(chunks_out[0]) * len(chunks_out[1])
            assert out.dtype == np.float32
            assert np.array_equal(out, expected)

            # The integer output, from the same float32 average
            out_u8 = _average_lazy(s_lazy, **kwargs)
            assert out_u8.dtype == np.uint8
            assert np.array_equal(out_u8, _average(s, **kwargs))

        # The lambda path on the lazy route: the same optimised lambda
        # (the distances use the global saturation maximum, not one per
        # chunk), and lam=None averages with it
        lam = s.get_nlpar_lambda()
        assert s_lazy.get_nlpar_lambda() == lam
        kwargs = {"search_radius": 1, "lam": None, "dtype_out": "float32"}
        expected = _average(s, **kwargs)
        assert np.array_equal(_average_lazy(s_lazy, **kwargs), expected)
        assert np.array_equal(
            expected, _average(s, search_radius=1, lam=lam, dtype_out="float32")
        )
        assert np.array_equal(s_lazy.data.compute(), x)

    @pytest.mark.parametrize("search_radius", [3, 4])
    def test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3(
        self, search_radius
    ):
        s = kp.data.nickel_ebsd_large(allow_download=True)
        data = s.data.copy()
        kwargs = {"search_radius": search_radius, "lam": 2.5, "dtype_out": "float32"}
        # The eager route runs on the default chunking
        default_chunks = get_dask_array(signal=s, chunk_bytes=8e6, rechunk=True).chunks
        assert default_chunks[:2] == ((47, 8), (47, 28))
        expected = _average(s, **kwargs)

        # Explicit rows (26, 26, 3), processed as ((26, 26, 3), (40, 35)):
        # the 3-row last chunk is thinner than the depth of the shifted
        # window of the last rows (4 at radius 3, 6 at radius 4)
        s_lazy = _lazy_signal(data, ((26, 26, 3), (75,)))
        processed = get_dask_array(signal=s_lazy, chunk_bytes=8e6, rechunk=True).chunks
        assert processed[:2] == ((26, 26, 3), (40, 35))
        _, chunks_out = nlpar_module._nlpar_depth(
            processed, (search_radius, search_radius)
        )
        expected_rows = {3: (26, 25, 4), 4: (26, 23, 6)}[search_radius]
        assert chunks_out[0] == expected_rows
        out = _average_lazy(s_lazy, **kwargs)
        assert np.array_equal(out, expected)

        # The chunking of the signal made lazy, as the method processes it
        out_as_lazy = _average_lazy(s.as_lazy(), **kwargs)
        assert np.array_equal(out_as_lazy, expected)
        assert np.array_equal(s.data, data)

    @pytest.mark.parametrize("variant", ["zeros", "noisy"])
    def test_issue_230_irregular_chunks(self, variant):
        # The chunking of the lazy regression test of
        # average_neighbour_patterns (pyxem/kikuchipy#230): rows thinner
        # than the depth of the averaging pass
        chunks = ((3, 3, 4, 3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3, 4), (75,), (6,), (6,))
        shape = (55, 75, 6, 6)
        if variant == "zeros":
            data = np.zeros(shape, dtype=np.uint8)
        else:
            rng = np.random.default_rng(230)
            data = rng.integers(20, 240, shape, dtype=np.uint8)
        s_lazy = kp.signals.LazyEBSD(da.from_array(data, chunks=chunks))
        processed = get_dask_array(signal=s_lazy, chunk_bytes=8e6, rechunk=True).chunks
        assert processed == chunks

        s_out = s_lazy.average_non_local_neighbour_patterns(
            lam=1.0, inplace=False, lazy_output=True
        )
        assert isinstance(s_out, kp.signals.LazyEBSD)
        out = s_out.data.compute()
        expected = _average(kp.signals.EBSD(data.copy()), lam=1.0)
        assert out.dtype == np.uint8
        assert np.array_equal(out, expected)
        if variant == "noisy":
            assert not np.array_equal(out, data)

    def test_scheduler_and_thread_invariance(self, random_uniform_saturated):
        x = random_uniform_saturated((10, 16), (16, 16), one_block_only=True)
        s = kp.signals.EBSD(x.copy())
        kwargs = {"search_radius": 3, "lam": 1.0, "dtype_out": "float32"}
        expected = _average(s, **kwargs)
        expected_sigma = s.get_nlpar_sigma()

        configs = [
            {"scheduler": "synchronous"},
            {"scheduler": "threads", "num_workers": 1},
            {"scheduler": "threads", "num_workers": 4},
        ]
        for config in configs:
            s_lazy = _lazy_signal(x, ((3, 3, 4), -1))
            with dask.config.set(**config):
                sigma = s_lazy.get_nlpar_sigma()
                out = _average_lazy(s_lazy, **kwargs)
                out_eager = _average(s, **kwargs)
            assert np.array_equal(sigma, expected_sigma), config
            assert np.array_equal(out, expected), config
            assert np.array_equal(out_eager, expected), config

    @pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
    def test_inplace_equals_inplace_false_on_a_multichunk_eager_signal(
        self, counting_spy, scheduler
    ):
        s = kp.data.nickel_ebsd_large(allow_download=True)
        data = s.data.copy()
        # The in-memory signal is processed in several navigation chunks
        # (((47, 8), (47, 28)) with dask 2026.3.0, ((47, 8), (25, 25, 25))
        # with dask 2021.8.1), whose halos read the signal's own buffer,
        # so an in-place result must not be written into that buffer
        # while the graph runs
        chunks = get_dask_array(signal=s, chunk_bytes=8e6, rechunk=True).chunks
        assert max(len(chunks[0]), len(chunks[1])) > 1

        # Each run goes through both chunked passes: one sigma wrapper
        # call per block of these chunks and one averaging wrapper call
        # per block of the averaging-pass chunks. An eager route around
        # the drivers gives the same values, so the calls are counted
        sigma_calls = counting_spy(nlpar_module, "_nlpar_sigma_chunk")
        average_calls = counting_spy(nlpar_module, "_nlpar_average_chunk")
        kwargs = {"search_radius": 3, "lam": 2.5}
        with dask.config.set(scheduler=scheduler):
            s_out = s.average_non_local_neighbour_patterns(inplace=False, **kwargs)
            first_run = (len(sigma_calls), len(average_calls))
            s.average_non_local_neighbour_patterns(**kwargs)

        assert s.data.dtype == np.uint8
        assert not np.array_equal(s_out.data, data)
        assert np.array_equal(s.data, s_out.data)

        n_sigma = len(chunks[0]) * len(chunks[1])
        chunks_out = nlpar_module._nlpar_depth(chunks, (3, 3))[1]
        n_average = len(chunks_out[0]) * len(chunks_out[1])
        assert n_average > 1
        assert first_run == (n_sigma, n_average)
        assert (len(sigma_calls), len(average_calls)) == (2 * n_sigma, 2 * n_average)

    def test_custom_attributes_are_carried(self, dummy_signal):
        s_out = dummy_signal.average_non_local_neighbour_patterns(
            search_radius=1, lam=1.0, inplace=False
        )
        assert isinstance(s_out, kp.signals.EBSD)
        assert np.allclose(s_out.static_background, dummy_signal.static_background)
        assert np.allclose(s_out.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s_out.xmap.rotations.data, dummy_signal.xmap.rotations.data)
        assert np.array_equal(s_out.xmap.phase_id, dummy_signal.xmap.phase_id)

    @pytest.mark.parametrize(
        "show_progressbar, preference, registered",
        [
            (True, False, True),
            (False, True, False),
            (None, True, True),
            (None, False, False),
        ],
    )
    @pytest.mark.parametrize("method", ["average", "get_nlpar_sigma"])
    def test_show_progressbar_registers_and_unregisters(
        self,
        monkeypatch,
        random_uniform_saturated,
        method,
        show_progressbar,
        preference,
        registered,
    ):
        events = []

        class RecordingProgressBar:
            def register(self):
                events.append("register")

            def unregister(self):
                events.append("unregister")

        monkeypatch.setattr(ebsd_module, "ProgressBar", RecordingProgressBar)
        monkeypatch.setattr(hs.preferences.General, "show_progressbar", preference)

        # The final computation of the inplace=False signal: HyperSpy
        # would otherwise draw its own bar (following the preference, so
        # even with show_progressbar=False)
        compute_bars = []
        original_compute = kp.signals.LazyEBSD.compute

        def recording_compute(self, *args, **kwargs):
            compute_bars.append(kwargs.get("show_progressbar"))
            return original_compute(self, *args, **kwargs)

        monkeypatch.setattr(kp.signals.LazyEBSD, "compute", recording_compute)

        s = kp.signals.EBSD(_random_map(random_uniform_saturated))
        if method == "average":
            _average(s, lam=1.0, show_progressbar=show_progressbar)
            assert compute_bars == [False]
        else:
            s.get_nlpar_sigma(show_progressbar=show_progressbar)
            assert compute_bars == []

        if registered:
            _assert_balanced_registration(events)
        else:
            assert events == []

    def test_sigma_argument_contract(self, monkeypatch, random_uniform_saturated):
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data)
        kwargs = {"search_radius": 1, "lam": 1.0, "dtype_out": "float32"}
        out_none = _average(s, **kwargs)
        sigma = s.get_nlpar_sigma()

        # A given sigma skips the sigma pass
        def no_sigma_pass(*args, **kw):
            raise AssertionError("the sigma pass ran although sigma was given")

        monkeypatch.setattr(nlpar_module, "_nlpar_sigma", no_sigma_pass)
        monkeypatch.setattr(ebsd_module, "_nlpar_sigma", no_sigma_pass, raising=False)

        out_array = _average(s, sigma=sigma, **kwargs)
        out_array_f64 = _average(s, sigma=sigma.astype(np.float64), **kwargs)
        assert np.array_equal(out_array, out_none)
        assert np.array_equal(out_array_f64, out_none)

        out_scalar = _average(s, sigma=8.0, **kwargs)
        out_constant = _average(s, sigma=np.full(NAV_SHAPE, 8.0, np.float32), **kwargs)
        out_int_scalar = _average(s, sigma=8, **kwargs)
        assert np.array_equal(out_scalar, out_constant)
        assert np.array_equal(out_int_scalar, out_scalar)
        assert not np.array_equal(out_scalar, out_none)

        # The navigation shape exactly: a transposed, a smaller or a
        # flattened array of the right size is never reshaped
        for wrong_shape in [(5, 4), (2, 2), (20,)]:
            with pytest.raises(ValueError, match="navigation shape"):
                _average(s, sigma=np.ones(wrong_shape, np.float32), **kwargs)
        # A 1D scan of 7 patterns has the navigation shape (7,), although
        # it is processed as a (1, 7) map
        s_1d = kp.signals.EBSD(_random_map(random_uniform_saturated, nav_shape=(7,)))
        with pytest.raises(ValueError, match="navigation shape"):
            _average(s_1d, sigma=np.ones((1, 7), np.float32), **kwargs)
        out_1d = _average(s_1d, sigma=np.full(7, 8.0, np.float32), **kwargs)
        assert out_1d.shape == s_1d.data.shape

        # A given sigma must keep the float32 distances finite: sigma^2
        # > 0 and 2 n sigma^2 finite, n being the number of kept pixels
        # (36 here, a bound between 1e18 and 1e19), else every pattern
        # comes back NaN. The bound follows the mask
        out_large = _average(s, sigma=1e18, **kwargs)
        assert np.all(np.isfinite(out_large))
        with pytest.raises(ValueError, match=SIGMA_RANGE_MESSAGE):
            _average(s, sigma=1e19, **kwargs)
        one_pixel = np.ones(SIG_SHAPE, dtype=bool)
        one_pixel[2, 3] = False
        out_one_pixel = _average(s, sigma=1e19, signal_mask=one_pixel, **kwargs)
        assert np.all(np.isfinite(out_one_pixel))
        with pytest.raises(ValueError, match=SIGMA_RANGE_MESSAGE):
            _average(s, sigma=1e-30, signal_mask=one_pixel, **kwargs)

    @pytest.mark.parametrize("dtype", [np.uint32, np.int32])
    def test_integer_output_keeps_the_maximum_of_32_bit_types(self, dtype):
        # A map at the maximum of the type, except one pattern at 1000.
        # With protection on, every pixel at the global maximum is
        # excluded, so every non-self weight is 0 and the output equals
        # the input. The float32 average of the maximum is 2**32 (2**31
        # for int32), beyond the type: rounded and clipped in float32 it
        # stays there, and the cast wraps it around (to 0, or to the
        # minimum of int32)
        top = np.iinfo(dtype).max
        data = np.full(NAV_SHAPE + (3, 3), top, dtype=dtype)
        data[0, 0] = 1000
        s = kp.signals.EBSD(data.copy())
        out = _average(s, search_radius=1, lam=0.7)
        assert out.dtype == dtype
        assert np.array_equal(out, data)

    def test_integer_output_clips_64_bit_types_inside_their_range(self):
        # Values beyond both 64-bit ranges, unchanged by the averaging as
        # above: the maxima round up to 2**64 and 2**63 in float64, so
        # the clip bounds are the largest float64 below them
        data = np.full(NAV_SHAPE + (3, 3), 1e20, dtype=np.float32)
        data[0, 0] = 1000
        data[0, 0, 0, 0] = -1e20
        s = kp.signals.EBSD(data.copy())
        limits = {np.uint64: (0, 2**64 - 2048), np.int64: (-(2**63), 2**63 - 1024)}
        for dtype, (low, high) in limits.items():
            out = _average(s, search_radius=1, lam=0.7, dtype_out=dtype)
            assert out.dtype == dtype
            assert out[0, 0, 0, 0] == low
            assert np.all(out[0, 0].ravel()[1:] == 1000)
            assert np.all(out[0, 1:] == high)
            assert np.all(out[1:] == high)


class TestSigmaMethod:
    """Contracts of :meth:`~kikuchipy.signals.EBSD.get_nlpar_sigma`."""

    def test_returns_float32_of_navigation_shape(self, random_uniform_saturated):
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data.copy())
        sigma = s.get_nlpar_sigma()
        assert isinstance(sigma, np.ndarray)
        assert sigma.dtype == np.float32
        assert sigma.shape == NAV_SHAPE
        assert np.all(np.isfinite(sigma) & (sigma > 0))
        assert np.array_equal(s.data, data)

        s_1d = kp.signals.EBSD(_random_map(random_uniform_saturated, nav_shape=(7,)))
        sigma_1d = s_1d.get_nlpar_sigma()
        assert sigma_1d.dtype == np.float32
        assert sigma_1d.shape == (7,)

    @pytest.mark.parametrize(
        "sigma_kwargs",
        [{}, {"signal_mask": "circle", "saturation_protect": False}],
        ids=["default", "circle-unprotected"],
    )
    def test_equals_the_pass_one_of_the_method(
        self, random_uniform_saturated, circle_mask, sigma_kwargs
    ):
        data = random_uniform_saturated(NAV_SHAPE, SIG_SHAPE)
        s = kp.signals.EBSD(data)
        sigma_kwargs = _resolve(sigma_kwargs, SIG_SHAPE, circle_mask)
        kwargs = {"lam": 1.0, "dtype_out": "float32", **sigma_kwargs}
        sigma = s.get_nlpar_sigma(**sigma_kwargs)
        out_given = _average(s, sigma=sigma, **kwargs)
        out_none = _average(s, sigma=None, **kwargs)
        assert np.array_equal(out_given, out_none)

        # The (7,) sigma map of a 1D scan is accepted back as it is
        # returned, although the scan is processed as a (1, 7) map
        s_1d = kp.signals.EBSD(random_uniform_saturated((7,), SIG_SHAPE))
        sigma_1d = s_1d.get_nlpar_sigma(**sigma_kwargs)
        assert sigma_1d.shape == (7,)
        out_given_1d = _average(s_1d, sigma=sigma_1d, **kwargs)
        out_none_1d = _average(s_1d, sigma=None, **kwargs)
        assert np.array_equal(out_given_1d, out_none_1d)

    def test_forwards_mask_and_protection(self, random_uniform_saturated, circle_mask):
        # 5 % of the pixels of every pattern at 255, the map maximum
        data = random_uniform_saturated(NAV_SHAPE, SIG_SHAPE)
        s = kp.signals.EBSD(data)
        sigma = s.get_nlpar_sigma()
        sigma_mask = s.get_nlpar_sigma(signal_mask=circle_mask(SIG_SHAPE))
        sigma_unprotected = s.get_nlpar_sigma(saturation_protect=False)
        assert not np.array_equal(sigma_mask, sigma)
        assert not np.array_equal(sigma_unprotected, sigma)


# ----------------------------- Real data ---------------------------- #

requires_pyebsdindex = pytest.mark.skipif(
    dependency_version["pyebsdindex"] is None, reason="pyebsdindex is not installed"
)

# Hough indexing quality metrics and the sign of their improvement:
# higher is better except for the band fit, in degrees
HOUGH_METRIC_SIGNS = {"pq": 1.0, "fit": -1.0, "nmatch": 1.0, "cm": 1.0}


def _hough_quality(s: kp.signals.EBSD, reference: CrystalMap) -> tuple[dict, float]:
    """Return the medians of the Hough indexing quality metrics of a
    signal and the median misorientation, in degrees, of its
    orientations to those of a reference crystal map, compared by point
    order.
    """
    phase_list = reference.phases
    indexer = s.detector.get_indexer(phase_list)
    xmap = s.hough_indexing(phase_list, indexer, verbose=0)
    assert xmap.size == reference.size
    medians = {key: float(np.median(xmap.prop[key])) for key in HOUGH_METRIC_SIGNS}
    point_group = phase_list[phase_list.ids[0]].point_group
    ori = Orientation(xmap.rotations, symmetry=point_group)
    ori_reference = Orientation(reference.rotations, symmetry=point_group)
    misorientation = ori.angle_with(ori_reference, degrees=True)
    return medians, float(np.median(misorientation))


def _as_numpy(x: np.ndarray | da.Array) -> np.ndarray:
    """Return a computed NumPy array of an array that may be lazy."""
    if isinstance(x, da.Array):
        x = x.compute()
    return np.asarray(x)


class TestRealData:
    """NLPAR improves the background-corrected ``nickel_ebsd_large`` by
    the map metrics and by Hough indexing quality, at search radius 3.
    Hough gains are pinned at 0.5 x the measured gain where the metric
    improves, and as "not worse" otherwise.
    """

    def test_adp_improves(self, record_property):
        s = _nickel_corrected()
        kwargs = {"inplace": False, "show_progressbar": False}
        s_auto = s.average_non_local_neighbour_patterns(lam=None, **kwargs)
        s_07 = s.average_non_local_neighbour_patterns(lam=0.7, **kwargs)
        adp = {}
        for name, signal in [("before", s), ("auto", s_auto), ("07", s_07)]:
            adp_map = signal.get_average_neighbour_dot_product_map(
                show_progressbar=False
            )
            adp[name] = float(np.mean(adp_map))
            record_property(f"adp_{name}", adp[name])

        assert adp["auto"] > adp["before"]
        assert adp["07"] > adp["before"]
        _assert_in_band(adp["before"], ADP_BEFORE, "ADP_BEFORE")
        _assert_in_band(adp["auto"], ADP_AFTER_AUTO, "ADP_AFTER_AUTO")
        _assert_in_band(adp["07"], ADP_AFTER_07, "ADP_AFTER_07")

    def test_iq_improves(self, record_property):
        s = _nickel_corrected()
        s_auto = s.average_non_local_neighbour_patterns(
            lam=None, inplace=False, show_progressbar=False
        )
        iq_before = float(np.mean(s.get_image_quality(show_progressbar=False)))
        iq_after = float(np.mean(s_auto.get_image_quality(show_progressbar=False)))
        record_property("iq_before", iq_before)
        record_property("iq_after_auto", iq_after)

        assert iq_after > iq_before
        _assert_in_band(iq_before, IQ_BEFORE, "IQ_BEFORE")
        _assert_in_band(iq_after, IQ_AFTER_AUTO, "IQ_AFTER_AUTO")

    @requires_pyebsdindex
    def test_hough_quality_before_and_after(self, record_property):
        s = _nickel_corrected()
        s_after = s.average_non_local_neighbour_patterns(
            lam=None, inplace=False, show_progressbar=False
        )
        # Every fifth pattern along both axes: 165 patterns in the
        # row-major order of the stored crystal map points
        s_sub = s.inav[::5, ::5]
        s_sub_after = s_after.inav[::5, ::5]
        assert s_sub.data.shape == (11, 15, 60, 60)
        assert s_sub_after.data.shape == (11, 15, 60, 60)
        reference = s_sub.xmap
        assert reference.size == 165

        before, miso_before = _hough_quality(s_sub, reference)
        after, miso_after = _hough_quality(s_sub_after, reference)
        gains = {
            key: sign * (after[key] - before[key])
            for key, sign in HOUGH_METRIC_SIGNS.items()
        }
        for key in HOUGH_METRIC_SIGNS:
            record_property(f"hough_{key}_median_before", before[key])
            record_property(f"hough_{key}_median_after", after[key])
            record_property(f"hough_{key}_gain", gains[key])
        record_property("hough_miso_median_before", miso_before)
        record_property("hough_miso_median_after", miso_after)

        pins = {
            "pq": HOUGH_PQ_GAIN,
            "fit": HOUGH_FIT_GAIN,
            "nmatch": HOUGH_NMATCH_GAIN,
            "cm": HOUGH_CM_GAIN,
        }
        for key, pin in pins.items():
            _assert_at_least(gains[key], pin, f"HOUGH_{key.upper()}_GAIN")
        _assert_at_most(miso_after, HOUGH_MISO_MEDIAN_AFTER, "HOUGH_MISO_MEDIAN_AFTER")

    @pytest.mark.weekly
    @requires_pyebsdindex
    def test_hough_quality_full_map(self, record_property):
        s = _nickel_corrected()
        s_after = s.average_non_local_neighbour_patterns(
            lam=None, inplace=False, show_progressbar=False
        )
        reference = s.xmap
        assert reference.size == 4125
        before, miso_before = _hough_quality(s, reference)
        after, miso_after = _hough_quality(s_after, reference)
        # Recorded only
        for key in HOUGH_METRIC_SIGNS:
            record_property(f"hough_full_{key}_median_before", before[key])
            record_property(f"hough_full_{key}_median_after", after[key])
        record_property("hough_full_miso_median_before", miso_before)
        record_property("hough_full_miso_median_after", miso_after)
        assert np.all(np.isfinite(list(after.values())))
        assert np.isfinite(miso_after)

    def test_sigma_map_is_plausible(self, record_property):
        s_raw = kp.data.nickel_ebsd_large(allow_download=True)
        s_corrected = _nickel_corrected()
        sigma_raw = s_raw.get_nlpar_sigma(show_progressbar=False)
        sigma_corrected = s_corrected.get_nlpar_sigma(show_progressbar=False)
        for sigma in (sigma_raw, sigma_corrected):
            assert sigma.shape == (55, 75)
            assert sigma.dtype == np.float32
            assert np.all(np.isfinite(sigma) & (sigma > 0))
        # Recorded, not gated: a noise level of grey levels on a 0-255
        # scale, and how the background correction scales it
        median_raw = float(np.median(sigma_raw))
        median_corrected = float(np.median(sigma_corrected))
        record_property("sigma_median_raw", median_raw)
        record_property("sigma_median_corrected", median_corrected)
        record_property(
            "sigma_median_ratio_corrected_raw", median_corrected / median_raw
        )
        assert 0 < median_corrected < 255


def _si_wafer() -> kp.signals.LazyEBSD:
    """Return the lazy ``si_wafer`` dataset, (50, 50 | 480, 480), from
    the local cache, or skip: the suite never downloads it by itself.
    """
    try:
        return kp.data.si_wafer(allow_download=False, lazy=True)
    except (ValueError, FileNotFoundError) as error:  # pragma: no cover
        pytest.skip(
            "si_wafer is not cached; fetch it once with "
            f"kp.data.si_wafer(allow_download=True): {error}"
        )


@pytest.mark.weekly
class TestSiWafer:
    """Low-signal patterns of a single crystal, processed lazily with
    the method's own chunking.
    """

    def test_si_wafer_sigma_cv(self, record_property):
        s = _si_wafer()
        t0 = time.perf_counter()
        sigma = s.get_nlpar_sigma(show_progressbar=False)
        record_property("si_wafer_sigma_runtime_s", time.perf_counter() - t0)
        assert sigma.shape == (50, 50)
        assert np.all(np.isfinite(sigma) & (sigma > 0))
        sigma = sigma.astype(np.float64)
        cv = float(np.std(sigma) / np.mean(sigma))
        record_property("si_wafer_sigma_median", float(np.median(sigma)))
        record_property("si_wafer_sigma_cv", cv)
        _assert_at_most(cv, SI_SIGMA_CV, "SI_SIGMA_CV")

    def test_si_wafer_effective_neighbours(self, record_property):
        s = _si_wafer()
        sigma = s.get_nlpar_sigma(show_progressbar=False)
        lam = s.get_nlpar_lambda(show_progressbar=False)
        record_property("si_wafer_lambda", lam)

        # The search windows of the interior points (3:13, 3:13) of the
        # block (17:33, 17:33) are those of the whole map at radius 3
        radius = 3
        block = _as_numpy(s.data[17:33, 17:33]).astype(np.float32)
        data3 = np.ascontiguousarray(block.reshape(16, 16, -1))
        sigma_block = np.ascontiguousarray(sigma[17:33, 17:33], dtype=np.float32)
        sigma2 = (sigma_block * sigma_block).astype(np.float32)
        mask_indices = np.arange(data3.shape[-1], dtype=np.int64)
        max_value = np.float32(_as_numpy(s.data.max()))
        d = _nlpar_distances_kernel(
            data3, sigma2, mask_indices, max_value, True, radius, radius, 3, 13, 3, 13
        )
        w = _nlpar_weights_kernel(d, lam, np.float32(0.0)).astype(np.float64)
        w /= w.sum(axis=-1, keepdims=True)
        n_eff = 1.0 / np.sum(w**2, axis=-1)
        assert np.all((n_eff >= 1) & (n_eff <= 49 + 1e-9))
        median = float(np.median(n_eff))
        record_property("si_wafer_n_eff_median", median)
        _assert_in_band(median, SI_NEFF_MEDIAN, "SI_NEFF_MEDIAN")

    def test_si_wafer_iq_gain(self, record_property):
        s = _si_wafer()
        t0 = time.perf_counter()
        s_after = s.average_non_local_neighbour_patterns(
            lam=None, inplace=False, lazy_output=False, show_progressbar=False
        )
        record_property("si_wafer_nlpar_runtime_s", time.perf_counter() - t0)
        assert isinstance(s_after, kp.signals.EBSD)
        assert not s_after._lazy
        iq_before = _as_numpy(s.get_image_quality(show_progressbar=False))
        iq_after = _as_numpy(s_after.get_image_quality(show_progressbar=False))
        ratio = float(np.mean(iq_after) / np.mean(iq_before))
        record_property("si_wafer_iq_before", float(np.mean(iq_before)))
        record_property("si_wafer_iq_after", float(np.mean(iq_after)))
        record_property("si_wafer_iq_gain", ratio)
        assert ratio > 1
        _assert_at_least(ratio, SI_IQ_GAIN, "SI_IQ_GAIN")
