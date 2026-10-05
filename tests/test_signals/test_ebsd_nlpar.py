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
:meth:`~kikuchipy.signals.EBSD.average_non_local_neighbour_patterns`
and :meth:`~kikuchipy.signals.EBSD.get_nlpar_sigma`.

Oracles used here, none of which is PyEBSDIndex:

* :func:`nlpar_reference`, a float64 NumPy transcription of the three
  NLPAR equations of Brewick, Wright and Rowenhorst (2019), written
  independently of :mod:`kikuchipy.pattern._nlpar`;
* analytic identities whose answer is exact by construction (constant
  map, radius 0, injected sigma, huge lambda, power-of-two scaling);
* seeded synthetic maps whose statistics are derived (iid noise, two
  grains), from the generators of the root ``conftest.py``.

Tolerances that are measured on the finished implementation and then
pinned are module constants set to None until measured; a test reading
one fails with "unfilled MEASURED-THEN-PINNED placeholder" and reports
the measured value. Every such test computes its result first, so that
while the methods are stubs it fails on the stub.
"""

import functools
import warnings

import dask
import hyperspy.api as hs
import numpy as np
import pytest
from scipy.ndimage import correlate

import kikuchipy as kp
import kikuchipy.pattern._nlpar as nlpar_module
from kikuchipy.pattern._nlpar import (
    _nlpar_distances_kernel,
    _nlpar_normalized_distances,
    _nlpar_sigma_kernel,
    _nlpar_weights_kernel,
)
import kikuchipy.signals.ebsd as ebsd_module
from kikuchipy.signals.util._dask import get_dask_array

# ----------------- MEASURED-THEN-PINNED placeholders ---------------- #
# Each constant below is a placeholder, measured then pinned: None
# until the implementation gate measures it, records date, machine and
# recipe, and pins it with pytest.approx(measured, rel=0.05) or the ~2x
# margin convention. The seeds quoted are drafting seeds (measured
# 2026-09-11 or derived 2026-10-04), not pins.

# Largest absolute difference, in grey levels on a 0-255 scale, between
# the float32 output of the method and the float64 reference
# nlpar_reference. Seed 1e-3 grey levels (float32 weights with 1e-6 to
# 1e-5 relative error applied to values spread over 20-240), pinned at
# ~2x the measured value.
REFERENCE_MAX_ABS_GREY: float | None = None

# Band (low, high) of the median over the interior of the ratio of the
# estimated to the true noise level on the iid-noise map at N = 1024
# pixels. Seed (0.969, 0.978): the minimum over eight correlated
# estimates of sigma^2, each with relative spread sqrt(2 / N), sits 1.0
# to 1.4 of that spread below the mean.
NOISE_SIGMA_RATIO_BAND: tuple[float, float] | None = None

# Band (low, high) of the mean of the normalised 3 x 3 distances on the
# iid-noise map at N = 1024. Seed: about +1.2 (a 5 % low sigma^2 bias
# shifts the mean by about 0.05 sqrt(N / 2)).
D_MEAN_BAND: tuple[float, float] | None = None

# Band (low, high) of the standard deviation of the normalised 3 x 3
# distances on the iid-noise map at N = 1024. Seed: about 1.0.
D_STD_BAND: tuple[float, float] | None = None

# Band (low, high) of the fraction of neighbour weights exactly 1.0
# (normalised distance <= 0) on the iid-noise map. Seed: about 0.1.
UNIT_WEIGHT_FRACTION_BAND: tuple[float, float] | None = None

# Largest relative difference between the measured residual noise
# variance of the output and sigma_true^2 times the mean over patterns
# of the sum of squared normalised weights. Seed: about 0.05
# (finite-sample class at 144 patterns x 1024 pixels).
NOISE_REDUCTION_TOL: float | None = None

# Smallest fraction of the boundary-column contrast of the two-grain
# map retained after NLPAR. Seed 0.995.
TWO_GRAIN_CONTRAST_MIN: float | None = None

# Largest ratio of the residual rms on the two boundary columns of the
# two-grain map to the residual rms on the interior columns. Seed: about
# 1.0-1.3 (boundary patterns have fewer same-grain neighbours in their
# window and average less).
TWO_GRAIN_BOUNDARY_RESIDUAL_TOL: float | None = None

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


def _circle_mask(sig_shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask that is True outside the circle inscribed
    in the signal shape (kikuchipy polarity, True = excluded).
    """
    h, w = sig_shape
    rr, cc = np.ogrid[:h, :w]
    radius = min(h, w) / 2
    return (rr - (h - 1) / 2) ** 2 + (cc - (w - 1) / 2) ** 2 > radius**2


def _resolve(kwargs: dict, sig_shape: tuple[int, int]) -> dict:
    """Return a copy of method keyword arguments with the mask name
    "circle" replaced by the inscribed-circle mask.
    """
    kwargs = dict(kwargs)
    if isinstance(kwargs.get("signal_mask"), str):
        kwargs["signal_mask"] = _circle_mask(sig_shape)
    return kwargs


def _average(s, **kwargs) -> np.ndarray:
    """Return the data of the NLPAR-averaged copy of a signal."""
    s_out = s.average_non_local_neighbour_patterns(inplace=False, **kwargs)
    return s_out.data


def _counting_spy(monkeypatch, module, name: str) -> list:
    """Replace ``module.name`` by a wrapper that records every call and
    delegates to the original, and return the list of recorded calls.

    ``functools.wraps`` keeps the original signature visible to
    :func:`inspect.signature`, so Dask still passes ``block_info`` to a
    wrapped chunk function.
    """
    original = getattr(module, name)
    calls = []

    @functools.wraps(original)
    def spy(*args, **kwargs):
        calls.append(kwargs.get("block_info"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


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
        self, random_uniform_saturated, search_radius
    ):
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data.copy())
        # The radius-0 check precedes the lam=None guard, so the default
        # lam also returns quietly
        for kwargs in [{"lam": 1.0}, {"lam": 1.0, "inplace": False}, {}]:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                out = s.average_non_local_neighbour_patterns(
                    search_radius=search_radius, **kwargs
                )
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
        self, random_uniform_saturated, fixture_kind, kwargs, differs_from
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
        method_kwargs = _resolve(kwargs, sig_shape)

        out_f32 = _average(s, dtype_out="float32", **method_kwargs)
        out_int = _average(s, **method_kwargs)
        others = [
            _average(s, dtype_out="float32", **_resolve(other, sig_shape))
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


# ------------------------- Contracts (eager) ------------------------ #

_INT_MASK = _circle_mask(SIG_SHAPE).astype(np.int64)
_BOOL_MASK = _circle_mask(SIG_SHAPE)
_NAN_SIGMA = np.ones(NAV_SHAPE, dtype=np.float32)
_NAN_SIGMA[1, 2] = np.nan

# (method, keyword arguments, expected): expected is the fragment of
# the ValueError message, or keyword arguments of an accepted run that
# must give the same output bitwise. The methods "average_0d" and
# "sigma_0d" run on a signal without navigation axes.
VALIDATION_ARMS = [
    ("average", {"search_radius": -1}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": 1.5}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": True}, "search_radius must be a non-negative int"),
    ("average", {"search_radius": (1, 2, 3)}, "one radius per navigation axis"),
    ("average", {"search_radius": np.int64(2)}, {"search_radius": 2}),
    ("average", {"lam": 0.0}, "lam must be > 0"),
    ("average", {"lam": -1}, "lam must be > 0"),
    ("average", {"dthresh": -0.1}, "dthresh must be >= 0"),
    ("average", {"sigma": 0.0}, "sigma must be > 0"),
    ("average", {"sigma": -1.0}, "sigma must be > 0"),
    ("average", {"sigma": True}, "sigma must be > 0"),
    ("average", {"sigma": np.zeros(NAV_SHAPE, dtype=np.float32)}, "sigma must be > 0"),
    ("average", {"sigma": _NAN_SIGMA}, "sigma must be > 0"),
    ("average", {"sigma": np.ones((2, 2))}, "navigation shape"),
    ("average", {"dtype_out": bool}, "dtype_out must be an integer or floating dtype"),
    (
        "average",
        {"dtype_out": complex},
        "dtype_out must be an integer or floating dtype",
    ),
    ("average", {"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
    (
        "average",
        {"signal_mask": np.ones(SIG_SHAPE, dtype=bool)},
        "excludes every pixel",
    ),
    ("average", {"signal_mask": _INT_MASK}, {"signal_mask": _BOOL_MASK}),
    ("average_0d", {}, "nothing to average"),
    ("average_0d", {"search_radius": 3}, "nothing to average"),
    ("sigma_0d", {}, "nothing to average"),
    ("sigma", {"signal_mask": np.ones((2, 2), dtype=bool)}, "signal shape"),
    ("sigma", {"signal_mask": np.ones(SIG_SHAPE, dtype=bool)}, "excludes every pixel"),
    ("sigma", {"signal_mask": _INT_MASK}, {"signal_mask": _BOOL_MASK}),
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


class TestLazyAndContracts:
    """The method contract inherited from ``average_neighbour_patterns``
    on in-memory signals, the dtype policy, 1D navigation, small maps,
    argument validation and the guards of the paths that are not
    implemented yet.
    """

    @pytest.mark.parametrize("method, kwargs, expected", VALIDATION_ARMS)
    def test_argument_validation(
        self, random_uniform_saturated, method, kwargs, expected
    ):
        data = _random_map(random_uniform_saturated)
        if method.endswith("_0d"):
            data = data[0, 0]
            method = method.removesuffix("_0d")
        s = kp.signals.EBSD(data.copy())

        def call(**kw):
            kw = dict(kw)
            if method == "average":
                kw.setdefault("lam", 1.0)
                return _average(s, **kw)
            return s.get_nlpar_sigma(**kw)

        if isinstance(expected, str):
            with pytest.raises(ValueError, match=expected):
                call(**kwargs)
        else:
            assert np.array_equal(call(**kwargs), call(**expected))
        assert np.array_equal(s.data, data)

    @pytest.mark.parametrize(
        "guard",
        [
            "lam_none",
            "lazy_input",
            "lazy_output",
            "get_nlpar_sigma",
            "get_nlpar_lambda",
        ],
    )
    def test_stage_a_guards_raise_not_implemented(
        self, random_uniform_saturated, guard
    ):
        # Five guards of the paths that land with the lambda optimisation
        # and the lazy route; this test is deleted when they land. The
        # method's guards fire after the argument validation, the
        # get_nlpar_lambda stub before any validation
        lambda_message = "lambda optimisation lands in Stage B"
        lazy_message = "lazy NLPAR lands in Stage B"
        data = _random_map(random_uniform_saturated)
        s = kp.signals.EBSD(data.copy())

        if guard == "lam_none":
            # The guard is specific to lam=None
            assert isinstance(_average(s, lam=1.0), kp.signals.EBSD)
            with pytest.raises(
                ValueError, match="search_radius must be a non-negative"
            ):
                s.average_non_local_neighbour_patterns(search_radius=-1, lam=None)
            with pytest.raises(NotImplementedError, match=lambda_message):
                s.average_non_local_neighbour_patterns(lam=None, inplace=False)
            with pytest.raises(NotImplementedError, match=lambda_message):
                s.average_non_local_neighbour_patterns()
        elif guard == "lazy_input":
            # The guard is specific to a lazy input
            assert isinstance(_average(s, lam=1.0), kp.signals.EBSD)
            s_lazy = s.as_lazy()
            with pytest.raises(
                ValueError, match="search_radius must be a non-negative"
            ):
                s_lazy.average_non_local_neighbour_patterns(search_radius=-1, lam=1.0)
            for kwargs in [
                {},
                {"inplace": False},
                {"inplace": False, "lazy_output": False},
            ]:
                with pytest.raises(NotImplementedError, match=lazy_message):
                    s_lazy.average_non_local_neighbour_patterns(lam=1.0, **kwargs)
        elif guard == "lazy_output":
            # The guard is specific to lazy_output=True
            out = s.average_non_local_neighbour_patterns(
                lam=1.0, inplace=False, lazy_output=False
            )
            assert isinstance(out, kp.signals.EBSD)
            assert not out._lazy
            with pytest.raises(
                ValueError, match="search_radius must be a non-negative"
            ):
                s.average_non_local_neighbour_patterns(
                    search_radius=-1, lam=1.0, inplace=False, lazy_output=True
                )
            with pytest.raises(NotImplementedError, match=lazy_message):
                s.average_non_local_neighbour_patterns(
                    lam=1.0, inplace=False, lazy_output=True
                )
        elif guard == "get_nlpar_sigma":
            # The guard is specific to a lazy input
            sigma = s.get_nlpar_sigma()
            assert sigma.shape == NAV_SHAPE
            with pytest.raises(NotImplementedError, match=lazy_message):
                s.as_lazy().get_nlpar_sigma()
        else:
            with pytest.raises(NotImplementedError, match=lambda_message):
                s.get_nlpar_lambda()
            # Before any validation
            with pytest.raises(NotImplementedError, match=lambda_message):
                s.get_nlpar_lambda(target_weight=5.0, dthresh=-1.0)
        assert np.array_equal(s.data, data)

    @pytest.mark.parametrize("nav_shape", [(2, 2), (3, 5)])
    def test_map_smaller_than_the_window(self, random_uniform_saturated, nav_shape):
        # At radius 3 both axes are shorter than the 7-point window, so
        # every pattern averages over the whole map (PyEBSDIndex would
        # index out of bounds here); padding slots must carry weight 0
        data = _random_map(random_uniform_saturated, nav_shape=nav_shape)
        s = kp.signals.EBSD(data)
        out = _average(s, search_radius=3, lam=1.0, dtype_out="float32")
        ref = nlpar_reference(data, search_radius=3, lam=1.0, dthresh=0.0)

        assert out.dtype == np.float32
        assert out.shape == data.shape
        assert np.all(np.isfinite(out))
        max_abs = float(np.max(np.abs(out.astype(np.float64) - ref)))
        _assert_at_most(max_abs, REFERENCE_MAX_ABS_GREY, "REFERENCE_MAX_ABS_GREY")

    def test_one_dimensional_navigation_equals_a_one_row_map(
        self, random_uniform_saturated
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

    def test_inplace_lazy_output_contract(self, random_uniform_saturated):
        data = _random_map(random_uniform_saturated)
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

    @pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
    def test_inplace_equals_inplace_false_on_a_multichunk_eager_signal(
        self, monkeypatch, scheduler
    ):
        s = kp.data.nickel_ebsd_large(allow_download=True)
        data = s.data.copy()
        # The in-memory signal is processed in several navigation chunks
        # (((47, 8), (47, 28)) with dask 2026.3.0), whose halos read the
        # buffer an in-place store writes into
        chunks = get_dask_array(signal=s, chunk_bytes=8e6, rechunk=True).chunks
        assert max(len(chunks[0]), len(chunks[1])) > 1

        # Each run goes through both chunked passes: one sigma wrapper
        # call per block of these chunks and one averaging wrapper call
        # per block of the averaging-pass chunks. An eager route around
        # the drivers gives the same values, so the calls are counted
        sigma_calls = _counting_spy(monkeypatch, nlpar_module, "_nlpar_sigma_chunk")
        average_calls = _counting_spy(monkeypatch, nlpar_module, "_nlpar_average_chunk")
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
        s = kp.signals.EBSD(_random_map(random_uniform_saturated))
        if method == "average":
            _average(s, lam=1.0, show_progressbar=show_progressbar)
        else:
            s.get_nlpar_sigma(show_progressbar=show_progressbar)

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
        self, random_uniform_saturated, sigma_kwargs
    ):
        data = random_uniform_saturated(NAV_SHAPE, SIG_SHAPE)
        s = kp.signals.EBSD(data)
        sigma_kwargs = _resolve(sigma_kwargs, SIG_SHAPE)
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

    def test_forwards_mask_and_protection(self, random_uniform_saturated):
        # 5 % of the pixels of every pattern at 255, the map maximum
        data = random_uniform_saturated(NAV_SHAPE, SIG_SHAPE)
        s = kp.signals.EBSD(data)
        sigma = s.get_nlpar_sigma()
        sigma_mask = s.get_nlpar_sigma(signal_mask=_circle_mask(SIG_SHAPE))
        sigma_unprotected = s.get_nlpar_sigma(saturation_protect=False)
        assert not np.array_equal(sigma_mask, sigma)
        assert not np.array_equal(sigma_unprotected, sigma)
