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

"""Tests of ``kikuchipy.indexing._spherical._preprocessing``.

Covers every named assertion of
``specs/2026-08-17-spherical-back-projection/validation.md`` which
belongs to the pattern preprocessing:

- The integer circular mask against ``kikuchipy.filters.Window``.
- The 1-D Gaussian fit: exact recovery, the ripple and modulated
  cases, the ``emsphinx_compatible`` flag which must stay a no-op
  here, the ``ss == 0`` non-convergence of an integer mean, the
  ``0 / 0`` flat input behind ``error_model="numpy"`` and the NaN
  semantics and comparison **direction** of the ported
  ``solve::cholesky``.
- The 2-D separable background: the one pixel offset behind
  ``emsphinx_compatible``, the flat fallback, the faithful
  ``[0, 0]`` initialisation of the row and column maxima and the
  masked pixels set to zero by the subtraction.
- The mosaic adaptive histogram equalisation: the tile and
  interpolation tables, a uniform image mapping to 255, the
  all-masked tile identity, masked out pixels still being equalised
  and the measured equivalence with scikit-image's and kikuchipy's
  CLAHE.
- ``_to_uint8``, which rounds half away from zero and so is **not**
  ``kikuchipy.pattern.rescale_intensity``.
- The frozen order of ``_preprocess_pattern`` and its ``IndexEBSD``
  defaults, including the unsigned 8-bit short-cut.
- ``py_func`` of every kernel, the Numba compilation flags
  (``error_model="numpy"`` on the fit kernel alone) and recorded
  timing baselines.

The discrete cosine image quality is **not** tested here: it lives
in ``_back_projection`` together with the module's only
``scipy.fft`` binding.
"""

import bisect
import functools
import inspect
import math
import time
import warnings

import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.indexing._spherical import _preprocessing

EPS = float(np.finfo(np.float64).eps)

# Every Numba kernel of the module, for the flag and py_func tests
KERNEL_NAMES = [
    "_ahe_cdf_kernel",
    "_ahe_equalize_kernel",
    "_cholesky_solve_3x3",
    "_fit_gaussian_1d_kernel",
    "_row_col_max_kernel",
    "_to_uint8_kernel",
]

# Measured good-pixel counts of the integer circular mask
CIRCLE_COUNTS = {(60, 60): 2819, (61, 61): 2821, (7, 7): 29, (48, 60): 1792}

# Measured tile bounds along one axis of a 60 pixel side
TILE_BOUNDS = {
    10: [
        (0, 6),
        (6, 12),
        (12, 18),
        (18, 24),
        (24, 30),
        (30, 36),
        (36, 42),
        (42, 48),
        (48, 54),
        (54, 60),
    ],
    7: [(0, 9), (9, 17), (17, 26), (26, 34), (34, 43), (43, 51), (51, 60)],
}
TILE_MIDPOINTS = {
    10: [3, 9, 15, 21, 27, 33, 39, 45, 51, 57],
    7: [4, 13, 21, 30, 39, 47, 56],
}

# The synthetic separable background of D9
BACKGROUND_PARAMETERS = {"ax": 32.4, "ay": 27.9, "bx": 800.0, "by": 648.0, "c": 180.0}

# Measured parameters of the first ``nickel_ebsd_small`` pattern
NI_GX = (23.873, 1101.768, 232.838)
NI_GY = (25.810, 1484.992, 233.768)


# ----------------------------- Helpers ------------------------------ #


def _njit_kernel_names(module):
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

    Falls back to the function itself while it is still an
    undecorated stub. Every caller first asserts that the kernel does
    carry a ``py_func``, so that an implementation without ``@njit``
    fails loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


@functools.lru_cache(maxsize=1)
def _ni_signal_data():
    """Return the nine ``nickel_ebsd_small`` patterns, cached.

    The array is made **read-only**: it is passed straight into the
    module, so an in-place write there would otherwise corrupt every
    later test instead of failing where it happens.
    """
    data = kp.data.nickel_ebsd_small().data
    data.flags.writeable = False
    return data


def gaussian_samples(a, b, c, n=58):
    """Return ``c exp(-(i - a)^2 / b)`` on ``0 .. n - 1``."""
    x = np.arange(n, dtype=np.float64)
    return c * np.exp(-((x - a) ** 2) / b)


def separable_background(shape=(60, 60), **kwargs):
    """Return the separable Gaussian background of D9."""
    parameters = dict(BACKGROUND_PARAMETERS)
    parameters.update(kwargs)
    rows = np.arange(shape[0], dtype=np.float64)
    columns = np.arange(shape[1], dtype=np.float64)
    row_factor = np.exp(-((rows - parameters["ay"]) ** 2) / parameters["by"])
    column_factor = np.exp(-((columns - parameters["ax"]) ** 2) / parameters["bx"])
    return parameters["c"] * row_factor[:, None] * column_factor[None, :]


def local_gauss_newton(y, rule="faithful"):
    """Return ``(a, b, c, steps)`` of ``gaussian::Model<Real>::fit``
    with a swappable stopping rule.

    A second implementation of ``include/util/gaussian.hpp`` lines
    140-231, transcribed here so that the **stopping rule** can be
    varied while everything else -- the estimate, the Jacobian and
    the module's own :func:`_cholesky_solve_3x3` -- stays identical.
    ``"faithful"`` is the C++'s ``metric >= metricPrev and
    metric < 1e-4`` (line 226); ``"metric_only"`` drops the
    non-decreasing half and ``"metric_le"`` reverses it.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = y.shape[0]
    i_max = int(np.argmax(y))
    c = y[i_max]
    a = float(i_max)
    xy = 0.0
    y2 = 0.0
    for i in range(n):
        yc = y[i] / c
        if yc > 0:
            dx = a - i
            yi = math.log(yc)
            y2 += yi * yi
            xy -= yi * dx * dx
    b = xy / y2
    ss_prev = 0.0
    metric_prev = float(np.finfo(np.float64).max)
    for step_count in range(1, 51):
        jtj = np.zeros((3, 3))
        jtr = np.zeros(3)
        ss = 0.0
        for i in range(n):
            dx = a - i
            dxb = dx / b
            dfdc = math.exp(-dx * dxb)
            fx = c * dfdc
            fxb = fx * dxb
            jacobian = (-fxb * 2, fxb * dxb, dfdc)
            residual = y[i] - fx
            for u in range(3):
                jtr[u] += residual * jacobian[u]
                for v in range(u, 3):
                    jtj[u, v] += jacobian[u] * jacobian[v]
            ss += residual * residual
        jtj[1, 0] = jtj[0, 1]
        jtj[2, 0] = jtj[0, 2]
        jtj[2, 1] = jtj[1, 2]
        step = np.zeros(3)
        if _preprocessing._cholesky_solve_3x3(jtj, jtr, step) != 0:
            return a, b, c, step_count
        a += step[0]
        b += step[1]
        c += step[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            metric = abs((ss_prev - ss) / ss)
        if rule == "faithful":
            stop = metric >= metric_prev and metric < 1e-4
        elif rule == "metric_only":
            stop = metric < 1e-4
        else:
            stop = metric <= metric_prev and metric < 1e-4
        if stop:
            return a, b, c, step_count
        metric_prev = metric
        ss_prev = ss
    return a, b, c, 50


def column_ramp(shape=(60, 60)):
    """Return an unsigned 8-bit image whose columns ramp ``0, 4,
    ..., 236``.
    """
    return np.tile(np.arange(shape[1], dtype=np.uint8) * 4, (shape[0], 1))


def literal_tile_bounds(n, n_regions):
    """Return ``(mids, starts, ends)`` of ``setSize()`` on one axis.

    ``include/util/ahe.hpp`` lines 128-147 written out for a single
    axis, with the mosaic half width ``0.5`` of line 124 and
    ``std::round`` as ``floor(x + 0.5)`` for a non-negative
    argument.
    """
    tile = n / n_regions
    mids, starts, ends = [], [], []
    for index in range(n_regions):
        centre = tile * index + tile / 2
        mids.append(int(math.floor(centre + 0.5)))
        starts.append(int(math.floor(max(centre - tile / 2, 0.0) + 0.5)))
        ends.append(int(math.floor(min(centre + tile / 2, float(n)) + 0.5)))
    return mids, starts, ends


def literal_interpolation_pairs(mids, n):
    """Return ``(l, u, c, f)`` per index, ``ahe.hpp`` lines 150-166.

    ``std::upper_bound`` is :func:`bisect.bisect_right` rather than
    :func:`numpy.searchsorted`, so that the module's own search is
    not simply repeated.
    """
    pairs = []
    for index in range(n):
        bound = bisect.bisect_right(mids, index)
        if bound == len(mids) or bound == 0:
            tile = 0 if bound == 0 else len(mids) - 1
            pairs.append((tile, tile, 0.5, 0.5))
        else:
            far = (index - mids[bound - 1]) / (mids[bound] - mids[bound - 1])
            pairs.append((bound - 1, bound, 1.0 - far, far))
    return pairs


def literal_mosaic_ahe(image, n_regions, good=None):
    """Return the out of place ``AdaptiveHistogramEqualizer::
    equalize()`` written out.

    An independent transcription of ``include/util/ahe.hpp`` lines
    117-262 which keeps the row and the column roles apart **by
    name**, so that a rectangular image tells a row/column mix-up
    in the tile table, the histograms or the interpolation pairs
    from a faithful implementation.  Only the shape comes from the
    module under test.
    """
    image = np.asarray(image)
    height, width = image.shape
    mid_x, start_x, end_x = literal_tile_bounds(width, n_regions)
    mid_y, start_y, end_y = literal_tile_bounds(height, n_regions)

    cdfs = np.zeros((n_regions, n_regions, 256))
    for ty in range(n_regions):
        for tx in range(n_regions):
            histogram = np.zeros(256, dtype=np.int64)
            for j in range(start_y[ty], end_y[ty]):
                for i in range(start_x[tx], end_x[tx]):
                    if good is None or good[j, i]:
                        histogram[image[j, i]] += 1
            if good is not None and histogram.max() == 0:
                # "there were no good pixels", lines 211-213
                histogram[:] = 1
            cumulative = np.cumsum(histogram).astype(np.float64)
            cdfs[ty, tx] = (255 / cumulative[255]) * cumulative

    row_pairs = literal_interpolation_pairs(mid_y, height)
    column_pairs = literal_interpolation_pairs(mid_x, width)
    equalised = np.zeros((height, width))
    for j, (j_l, j_u, j_c, j_f) in enumerate(row_pairs):
        for i, (i_l, i_u, i_c, i_f) in enumerate(column_pairs):
            value = image[j, i]
            equalised[j, i] = (
                cdfs[j_l, i_l, value] * j_c * i_c
                + cdfs[j_l, i_u, value] * j_c * i_f
                + cdfs[j_u, i_l, value] * j_f * i_c
                + cdfs[j_u, i_u, value] * j_f * i_f
            )
    return equalised


# --------------------------- Circular mask -------------------------- #


class TestCircularMask:
    @pytest.mark.parametrize("shape", [(60, 60), (61, 61), (7, 7)])
    def test_it_equals_the_kikuchipy_window_for_square_shapes(self, shape):
        mask = _preprocessing._circular_mask(shape)
        assert mask.dtype == np.bool_
        assert mask.shape == shape
        assert int(mask.sum()) == CIRCLE_COUNTS[shape]
        window = kp.filters.Window("circular", shape).astype(bool)
        assert np.array_equal(mask, window)

    def test_it_differs_from_the_kikuchipy_window_for_rectangles(self, record_property):
        shape = (48, 60)
        mask = _preprocessing._circular_mask(shape)
        window = kp.filters.Window("circular", shape).astype(bool)
        record_property("rectangular_window_count", str(int(window.sum())))
        assert int(mask.sum()) == CIRCLE_COUNTS[shape]
        assert not np.array_equal(mask, window)

    def test_an_explicit_radius(self):
        mask = _preprocessing._circular_mask((60, 60), radius=10)
        rows, columns = np.indices((60, 60))
        expected = (columns - 30) ** 2 + (rows - 30) ** 2 <= 100
        assert np.array_equal(mask, expected)
        assert int(mask.sum()) == 317

    def test_a_negative_radius_raises(self):
        with pytest.raises(ValueError):
            _preprocessing._circular_mask((60, 60), radius=-1)


# ------------------------ The 1-D Gaussian fit ---------------------- #


class TestGaussianFit:
    @pytest.mark.parametrize("a, b, c", [(25.3, 288.0, 150.0), (40.7, 60.5, 90.0)])
    def test_an_exact_gaussian_is_recovered(self, a, b, c):
        found_a, found_b, found_c, r2, converged = _preprocessing._fit_gaussian_1d(
            gaussian_samples(a, b, c)
        )
        assert converged
        assert found_a == pytest.approx(a, abs=1e-9)
        assert found_b == pytest.approx(b, abs=1e-9)
        assert found_c == pytest.approx(c, abs=1e-9)
        assert r2 == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(
        "a, b, c", [(25.3, 288.0, 150.0), (40.7, 60.5, 90.0), (30.0, 128.0, 200.0)]
    )
    def test_emsphinx_compatible_does_not_change_the_fit(self, a, b, c):
        # the flag is documented as a no-op here -- it acts in
        # ``_gaussian_background``, which shifts the abscissa -- so
        # an implementation which shifted it inside the fit instead
        # would still pass every other test of this class
        x = np.arange(58, dtype=np.float64)
        y = gaussian_samples(a, b, c) + 0.5 * np.sin(x)
        faithful = _preprocessing._fit_gaussian_1d(y)
        corrected = _preprocessing._fit_gaussian_1d(y, emsphinx_compatible=False)
        # same inputs through the same code path, so bitwise is fair
        assert faithful == corrected

    @pytest.mark.parametrize(
        "a, b, c", [(30.0, 128.0, 200.0), (25.3, 288.0, 150.0), (40.7, 60.5, 90.0)]
    )
    def test_a_small_ripple_barely_moves_the_fit(self, a, b, c, record_property):
        # measured |a - a0| 2.8e-5 / 1.0e-3 / 1.5e-4 and |b - b0|/b0
        # 1.1e-5 / 1.8e-4 / 1.6e-4 for the three cases: the ripple
        # is not a rounding error, so the mean bound is 2e-3 and not
        # the 1e-3 of an earlier draft, which (25.3, 288, 150) fails
        x = np.arange(58, dtype=np.float64)
        y = gaussian_samples(a, b, c) + 0.5 * np.sin(x)
        found_a, found_b, _, _, converged = _preprocessing._fit_gaussian_1d(y)
        record_property(
            f"ripple_{a}",
            f"da {abs(found_a - a):.3e} db_rel {abs(found_b - b) / b:.3e}",
        )
        assert converged
        assert abs(found_a - a) < 2e-3
        assert abs(found_b - b) / b < 1e-3

    @pytest.mark.parametrize(
        "a, b, c", [(30.0, 128.0, 200.0), (25.3, 288.0, 150.0), (40.7, 60.5, 90.0)]
    )
    def test_a_modulated_signal_still_finds_the_mean(self, a, b, c, record_property):
        x = np.arange(58, dtype=np.float64)
        y = gaussian_samples(a, b, c) + 3 * np.sin(x / 2) + 5
        found_a, _, _, r2, converged = _preprocessing._fit_gaussian_1d(y)
        record_property(f"modulated_{a}", f"a {found_a:.4f} r2 {r2:.5f}")
        assert converged
        assert abs(found_a - a) < 0.05
        assert r2 > 0.97

    def test_an_integer_mean_never_converges(self):
        # ``ss`` is exactly zero at every iteration, so the stopping
        # metric is NaN and no comparison is ever true; the C++ does
        # the same and falls back to a flat background
        *_, converged = _preprocessing._fit_gaussian_1d(
            gaussian_samples(30.0, 128.0, 200.0)
        )
        # the wrapper is annotated ``-> (..., bool)``, so the type is
        # pinned once here; everywhere else ``bool()`` is applied so
        # that a ``numpy.bool_`` fails only on this one assertion
        assert isinstance(converged, bool)
        assert bool(converged) is False

    @pytest.mark.parametrize("value", [5.0, 0.0])
    def test_a_flat_input_does_not_raise(self, value, record_property):
        y = np.full(58, value)
        *_, converged = _preprocessing._fit_gaussian_1d(y)
        assert bool(converged) is False

        kernel = _preprocessing._fit_gaussian_1d_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        params = np.zeros(4)
        compiled_status = kernel(y, params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            interpreted_status = _py_func(kernel)(y, np.zeros(4))
        # the status depends on the platform's NaN sign propagation,
        # so it is recorded and only its non-zeroness is asserted
        record_property(
            f"flat_fit_status_{value}",
            f"compiled {compiled_status} interpreted {interpreted_status}",
        )
        assert compiled_status != 0
        assert interpreted_status != 0

    def test_a_ramp_is_fitted_outside_the_range(self, record_property):
        y = np.arange(1, 59, dtype=np.float64)
        found_a, _, _, r2, converged = _preprocessing._fit_gaussian_1d(y)
        record_property("ramp_fit", f"a {found_a:.3f} r2 {r2:.4f}")
        assert converged
        assert found_a > 57

    def test_fewer_than_three_samples(self):
        kernel = _preprocessing._fit_gaussian_1d_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        assert kernel(np.array([1.0, 2.0]), np.zeros(4)) == 1
        assert _py_func(kernel)(np.array([1.0, 2.0]), np.zeros(4)) == 1
        with pytest.raises(ValueError):
            _preprocessing._fit_gaussian_1d(np.array([1.0, 2.0]))

    @pytest.mark.parametrize("y", [np.zeros((4, 4)), np.zeros((2, 3, 4))])
    def test_a_multidimensional_input_raises(self, y):
        with pytest.raises(ValueError, match="one-dimensional"):
            _preprocessing._fit_gaussian_1d(y)

    @pytest.mark.parametrize("a, b, c", [(25.3, 288.0, 150.0), (40.7, 60.5, 90.0)])
    def test_both_halves_of_the_stopping_rule_are_needed(
        self, a, b, c, record_property
    ):
        # the C++ returns on ``metric >= metricPrev and
        # metric < 1e-4`` (line 226).  Dropping the non-decreasing
        # half, or writing it ``<=``, stops the Gauss-Newton loop
        # 5 to 8 steps early; every other test here bounds only the
        # *mean*, which moves by 1e-7, so both mutations survive
        # them.  ``b`` moves by 4e-5 relative, which this pins
        # against a local transcription differing in the rule alone
        x = np.arange(58, dtype=np.float64)
        y = gaussian_samples(a, b, c) + 3 * np.sin(x / 2) + 5
        found_a, found_b, found_c, _, converged = _preprocessing._fit_gaussian_1d(y)
        assert converged

        faithful = local_gauss_newton(y, "faithful")
        record_property(
            f"stop_rule_{a}_faithful", f"{faithful[:3]} in {faithful[3]} steps"
        )
        # same arithmetic on the same platform, so this is tight
        assert (found_a, found_b, found_c) == pytest.approx(faithful[:3], rel=1e-9)

        for rule in ("metric_only", "metric_le"):
            loose = local_gauss_newton(y, rule)
            gap = abs(loose[1] - faithful[1])
            record_property(
                f"stop_rule_{a}_{rule}",
                f"{loose[3]} steps, b differs by {gap / abs(faithful[1]):.3e} relative",
            )
            # the two rules really do part company on this input, so
            # the comparison below cannot go vacuous
            assert loose[3] < faithful[3]
            assert gap / abs(faithful[1]) > 1e-8
            # and the kernel sits a hundred times closer to the
            # faithful rule than the two rules are to each other
            assert abs(found_b - faithful[1]) < 0.01 * gap

    @pytest.mark.parametrize("compiled", [True, False])
    def test_the_cholesky_comparison_direction(self, compiled):
        kernel = _preprocessing._cholesky_solve_3x3
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        solve = kernel if compiled else _py_func(kernel)
        b = np.full(3, np.nan)

        # both C++ tests are false for NaN, so this "succeeds"; the
        # defensive ``not (pivot >= eps)`` would give 2
        with np.errstate(invalid="ignore"):
            assert solve(np.full((3, 3), np.nan), b, np.zeros(3)) == 0

        flipped = np.full((3, 3), np.nan)
        flipped[1, 1] = -np.nan
        with np.errstate(invalid="ignore"):
            assert solve(flipped, b, np.zeros(3)) == 1

        # ``< eps`` and not ``< 0``: a tiny positive pivot fails
        assert solve(np.diag([1e-17, 1.0, 1.0]), np.ones(3), np.zeros(3)) == 2
        singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert solve(singular, np.ones(3), np.zeros(3)) == 2

    @pytest.mark.parametrize("compiled", [True, False])
    def test_the_cholesky_negation_of_a_negative_definite_system(self, compiled):
        kernel = _preprocessing._cholesky_solve_3x3
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        solve = kernel if compiled else _py_func(kernel)
        matrix = -np.array([[4.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        rhs = np.array([1.0, 2.0, 3.0])
        solution = np.zeros(3)
        assert solve(matrix.copy(), rhs, solution) == 0
        assert np.abs(matrix @ solution - rhs).max() <= 1e-12


# ---------------------- The 2-D separable background ---------------- #


class TestGaussianBackground:
    def test_the_one_pixel_offset_behind_emsphinx_compatible(self, record_property):
        good = _preprocessing._circular_mask((60, 60))
        truth = separable_background()
        faithful, info = _preprocessing._gaussian_background(truth, good)
        corrected, _ = _preprocessing._gaussian_background(
            truth, good, emsphinx_compatible=False
        )
        assert info["gx"][0] == pytest.approx(BACKGROUND_PARAMETERS["ax"] - 1, abs=1e-3)
        assert info["gy"][0] == pytest.approx(BACKGROUND_PARAMETERS["ay"] - 1, abs=1e-3)
        faithful_error = float(np.abs(faithful - truth)[good].max())
        corrected_error = float(np.abs(corrected - truth)[good].max())
        record_property(
            "background_errors", f"{faithful_error:.4f} / {corrected_error:.4f}"
        )
        assert faithful_error > 5
        assert corrected_error < 0.01

    def test_a_modulated_background_error_is_recorded(self, record_property):
        good = _preprocessing._circular_mask((60, 60))
        rows, columns = np.indices((60, 60))
        truth = separable_background()
        pattern = truth + 10 * np.cos(rows / 3) * np.cos(columns / 4) + 20
        faithful, _ = _preprocessing._gaussian_background(pattern, good)
        corrected, _ = _preprocessing._gaussian_background(
            pattern, good, emsphinx_compatible=False
        )
        record_property(
            "modulated_background_errors",
            f"{np.abs(faithful - truth)[good].max():.2f} / "
            f"{np.abs(corrected - truth)[good].max():.2f}",
        )

    def test_a_flat_pattern_falls_back_to_the_mean(self):
        pattern = np.full((60, 60), 5.0)
        background, info = _preprocessing._gaussian_background(pattern)
        assert bool(info["converged_x"]) is False
        assert bool(info["converged_y"]) is False
        assert math.isinf(info["gx"][1])
        assert math.isinf(info["gy"][1])
        assert np.allclose(background, 5.0)
        subtracted = _preprocessing._remove_gaussian_background(pattern)
        assert np.allclose(subtracted, 0.0)

    def test_an_exact_gaussian_axis_falls_back_too(self):
        # the integer mean quirk propagates: the fit reports
        # non-convergence, so the axis takes the flat fallback
        pattern = separable_background(ax=30.0, ay=30.0, bx=128.0, by=128.0, c=200.0)
        _, info = _preprocessing._gaussian_background(pattern)
        assert bool(info["converged_x"]) is False
        assert math.isinf(info["gx"][1])

    def test_the_failed_axis_fallback_triple(self):
        # the flat pattern above cannot see two thirds of the C++
        # fallback ``(w / 2, inf, mean(cWrk))`` (lines 274-277): on a
        # flat pattern the mean of the maxima *is* their maximum, and
        # ``b = inf`` makes the surface independent of ``a``.  An
        # integer-mean Gaussian fails both axes on a pattern whose
        # maxima are not flat, where ``mean`` is 66.83 and ``max``
        # is 200, a factor 3 in the background
        pattern = separable_background(ax=30.0, ay=30.0, bx=128.0, by=128.0, c=200.0)
        row_max = np.empty(60)
        col_max = np.empty(60)
        _preprocessing._row_col_max_kernel(
            np.ascontiguousarray(pattern),
            np.ones((60, 60), dtype=bool),
            row_max,
            col_max,
        )
        assert col_max.max() == pytest.approx(200.0, rel=1e-12)
        assert col_max.mean() < 0.4 * col_max.max()

        background, info = _preprocessing._gaussian_background(pattern)
        assert bool(info["converged_x"]) is False
        assert bool(info["converged_y"]) is False
        assert info["gx"][0] == pytest.approx(30.0, abs=1e-12)
        assert info["gy"][0] == pytest.approx(30.0, abs=1e-12)
        assert info["gx"][2] == pytest.approx(col_max.mean(), rel=1e-12)
        assert info["gy"][2] == pytest.approx(row_max.mean(), rel=1e-12)
        # the exponential factor of a failed axis is exactly one, so
        # the surface is the shared amplitude everywhere
        assert np.allclose(background, col_max.mean(), rtol=1e-12)

    @pytest.mark.parametrize("shape", [(48, 60), (60, 48)])
    def test_each_axis_is_fitted_on_its_own_length(self, shape, monkeypatch):
        # the column fit runs on ``cWrk[1 : w - 1]`` and the row fit
        # on ``rWrk[1 : h - 1]`` (lines 273 and 280).  Every other
        # pattern in this module is square, where the two interior
        # lengths are equal and a wrong axis length is invisible
        height, width = shape
        real = _preprocessing._fit_gaussian_1d_kernel
        sizes = []

        def spy(y, params):
            sizes.append(int(y.shape[0]))
            return real(y, params)

        monkeypatch.setattr(_preprocessing, "_fit_gaussian_1d_kernel", spy)
        pattern = separable_background(shape, ax=width / 3, ay=height / 3)
        _preprocessing._gaussian_background(pattern)
        # the column fit first, then the row fit
        assert sizes == [width - 2, height - 2]

        # and a failed axis falls back to *its own* half width
        sizes.clear()
        background, info = _preprocessing._gaussian_background(np.full(shape, 5.0))
        assert sizes == [width - 2, height - 2]
        assert info["gx"][0] == width / 2
        assert info["gy"][0] == height / 2
        assert background.shape == shape

    def test_the_maxima_skip_masked_out_pixels(self):
        # the mask reaches the maxima only through the ``good[j, i]``
        # test of lines 262-270.  Ignoring it changes nothing in the
        # background of the fixtures used elsewhere -- on Ni pattern
        # 0 with the circle the fitted triple is bitwise the same --
        # so the kernel needs a masked-out pixel which would take
        # over its row and column
        pattern = np.zeros((60, 60))
        pattern[0, 0] = 3.0
        pattern[30, 30] = 255.0
        good = np.ones((60, 60), dtype=bool)
        good[30, 30] = False
        row_max = np.empty(60)
        col_max = np.empty(60)
        kernel = _preprocessing._row_col_max_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        kernel(pattern, good, row_max, col_max)
        # only the seed of line 259 is left in that row and column
        assert row_max[30] == 3.0
        assert col_max[30] == 3.0
        # without the mask the pixel takes both over, so this is not
        # a statement about the pattern
        kernel(pattern, np.ones((60, 60), dtype=bool), row_max, col_max)
        assert row_max[30] == 255.0
        assert col_max[30] == 255.0

    def test_the_row_and_column_maxima_start_at_pixel_zero_zero(self):
        pattern = np.zeros((60, 60))
        pattern[0, 0] = 255.0
        good = np.ones((60, 60), dtype=bool)
        good[0, 0] = False
        row_max = np.empty(60)
        col_max = np.empty(60)
        kernel = _preprocessing._row_col_max_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        kernel(pattern, good, row_max, col_max)
        # faithful to lines 259-260: the seed ignores the mask
        assert row_max[0] == 255.0
        assert col_max[0] == 255.0

    def test_masked_pixels_are_zero_after_the_subtraction(self):
        good = _preprocessing._circular_mask((60, 60))
        pattern = separable_background() + 3.0
        subtracted = _preprocessing._remove_gaussian_background(pattern, good)
        assert np.all(subtracted[~good] == 0.0)
        assert np.any(subtracted[good] != 0.0)
        assert subtracted.dtype == np.float64

    def test_the_parameters_of_a_real_pattern(self, record_property):
        pattern = _ni_signal_data()[0, 0]
        _, info = _preprocessing._gaussian_background(pattern)
        record_property("ni_gx", str(info["gx"]))
        record_property("ni_gy", str(info["gy"]))
        for found, expected in zip(info["gx"], NI_GX):
            assert found == pytest.approx(expected, rel=0.05)
        for found, expected in zip(info["gy"], NI_GY):
            assert found == pytest.approx(expected, rel=0.05)
        assert info["c"] == pytest.approx(max(NI_GX[2], NI_GY[2]), rel=0.05)

    @pytest.mark.parametrize(
        "good", [np.ones((60, 61), dtype=bool), np.zeros((60, 60), dtype=int)]
    )
    def test_good_pixels_is_validated(self, good):
        with pytest.raises(ValueError):
            _preprocessing._gaussian_background(np.zeros((60, 60)), good)

    @pytest.mark.parametrize("shape", [(60,), (3, 4, 5)])
    def test_a_non_two_dimensional_pattern_raises(self, shape):
        for function in (
            _preprocessing._gaussian_background,
            _preprocessing._remove_gaussian_background,
        ):
            with pytest.raises(ValueError, match="two-dimensional"):
                function(np.zeros(shape))

    @pytest.mark.parametrize("shape", [(0, 60), (60, 0), (0, 0)])
    def test_an_empty_pattern_raises_instead_of_reading_out_of_bounds(self, shape):
        # ``_row_col_max_kernel`` seeds from ``pattern[0, 0]`` and
        # Numba compiles with bounds checking off, so without the
        # guard an empty axis is an out of bounds read rather than
        # an exception
        for function in (
            _preprocessing._gaussian_background,
            _preprocessing._remove_gaussian_background,
        ):
            with pytest.raises(ValueError, match="empty"):
                function(np.zeros(shape))


# ------------------- Mosaic histogram equalisation ------------------ #


class TestMosaicAHE:
    @pytest.mark.parametrize("n_regions", [10, 7])
    def test_the_tile_table(self, n_regions):
        tiles, j_pairs, i_pairs = _preprocessing._ahe_tiles((60, 60), n_regions)
        assert tiles.shape == (n_regions * n_regions, 4)
        bounds = TILE_BOUNDS[n_regions]
        for index, (start, end) in enumerate(bounds):
            assert tuple(tiles[index, :2]) == (start, end)
            assert tuple(tiles[index * n_regions, 2:]) == (start, end)

        midpoints = TILE_MIDPOINTS[n_regions]
        j_l, j_u, j_c, j_f = j_pairs
        i_l, i_u, i_c, i_f = i_pairs
        assert j_l.shape == (60,)
        assert i_l.shape == (60,)
        for index, midpoint in enumerate(midpoints[:-1]):
            # ``upper_bound`` on the midpoints: a pixel sitting on a
            # midpoint takes that tile with weight one
            assert (j_l[midpoint], j_u[midpoint]) == (index, index + 1)
            assert j_c[midpoint] == pytest.approx(1.0)
            assert j_f[midpoint] == pytest.approx(0.0)
            assert (i_l[midpoint], i_u[midpoint]) == (index, index + 1)

    def test_the_tile_table_of_a_rectangle(self):
        # a square shape cannot tell ``(i_start, i_end)`` from
        # ``(j_start, j_end)``, nor the row pairs from the column
        # ones.  ``(48, 60)`` at ``n_regions`` 6 gives 8 pixel tile
        # rows against 10 pixel tile columns
        tiles, j_pairs, i_pairs = _preprocessing._ahe_tiles((48, 60), 6)
        assert tiles.shape == (36, 4)
        # the first two bounds come from the width, the last two
        # from the height
        assert list(tiles[:6, 0]) == [0, 10, 20, 30, 40, 50]
        assert list(tiles[:6, 1]) == [10, 20, 30, 40, 50, 60]
        assert list(tiles[::6, 2]) == [0, 8, 16, 24, 32, 40]
        assert list(tiles[::6, 3]) == [8, 16, 24, 32, 40, 48]
        assert tiles[:, 1].max() == 60
        assert tiles[:, 3].max() == 48
        # and the row pairs are as long as the image is high
        assert [array.shape for array in j_pairs] == [(48,)] * 4
        assert [array.shape for array in i_pairs] == [(60,)] * 4

    @pytest.mark.parametrize("shape", [(48, 60), (60, 48), (37, 41)])
    @pytest.mark.parametrize("n_regions", [4, 7])
    def test_a_rectangle_equals_the_literal_equaliser(self, shape, n_regions):
        # every other case in this class is square, where swapping
        # the row and column interpolation pairs, or reading a tile
        # bound off the wrong axis, is a no-op
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, shape, dtype=np.uint8)
        masks = [None, _preprocessing._circular_mask(shape, min(shape) // 3)]
        for good in masks:
            ours = _preprocessing._mosaic_ahe(image, n_regions, good)
            theirs = literal_mosaic_ahe(image, n_regions, good)
            assert ours.shape == shape
            # same arithmetic in the same order, so this is tight
            assert np.abs(ours - theirs).max() <= 1e-9
        # the two masks really do give different images, so the
        # comparison above cannot go vacuous
        assert not np.allclose(
            _preprocessing._mosaic_ahe(image, n_regions, masks[1]),
            _preprocessing._mosaic_ahe(image, n_regions),
        )

    def test_the_interpolation_pairs_of_three_pixels(self):
        _, _, i_pairs = _preprocessing._ahe_tiles((60, 60), 10)
        i_l, i_u, i_c, i_f = i_pairs
        # ``lower_bound`` would take the end branch here
        assert (i_l[3], i_u[3], i_c[3], i_f[3]) == (0, 1, 1.0, 0.0)
        assert (i_l[0], i_u[0], i_c[0], i_f[0]) == (0, 0, 0.5, 0.5)
        assert (i_l[59], i_u[59], i_c[59], i_f[59]) == (9, 9, 0.5, 0.5)

    def test_a_uniform_image_maps_to_the_top_of_the_range(self):
        # faithful: the cumulative histogram is a step at the image's
        # own value, so nobody should "fix" this to the identity
        # the cumulative histogram entry is exactly 255.0 and the
        # four interpolation weights sum to exactly 1.0, yet the
        # literal four-term sum of ``ahe.hpp:254-257`` still lands on
        # 255.00000000000003 (999 px) and 255.00000000000009 (81 px)
        # for 1080 of the 3600 pixels, so this is a 1e-9 bound and
        # not an equality, which only a refactored sum would pass
        image = np.full((60, 60), 100, dtype=np.uint8)
        equalised = _preprocessing._mosaic_ahe(image, 10)
        assert equalised.dtype == np.float64
        assert np.abs(equalised - 255.0).max() <= 1e-9

    def test_a_column_ramp(self):
        equalised = _preprocessing._mosaic_ahe(column_ramp(), 10)
        assert equalised.min() == pytest.approx(42.5, abs=1e-9)
        assert equalised.max() == pytest.approx(255.0, abs=1e-9)

    def test_an_all_masked_tile_is_the_identity_ramp(self):
        image = column_ramp()
        good = np.zeros((60, 60), dtype=bool)
        good[:6, :6] = True
        equalised = _preprocessing._mosaic_ahe(image, 10, good)
        block = equalised[30:36, 30:36]
        expected = (image[30:36, 30:36].astype(np.float64) + 1) * 255 / 256
        assert np.abs(block - expected).max() <= 1e-10

    def test_masked_out_pixels_are_equalised_too(self, record_property):
        image = _ni_signal_data()[0, 0]
        good = _preprocessing._circular_mask((60, 60))
        masked = _preprocessing._mosaic_ahe(image, 10, good)
        plain = _preprocessing._mosaic_ahe(image, 10)
        difference = float(np.abs(masked - plain).max())
        record_property("ahe_mask_difference", f"{difference:.2f}")
        record_property(
            "ahe_masked_corner_range",
            f"[{masked[~good].min():.2f}, {masked[~good].max():.2f}]",
        )
        assert difference > 50
        assert np.any(masked[~good] != 0)

    @pytest.mark.parametrize(
        "n_regions, kernel_size, tolerance", [(10, (6, 6), 8.0), (4, (15, 15), 2.0)]
    )
    def test_it_equals_scikit_images_clahe(
        self, n_regions, kernel_size, tolerance, record_property
    ):
        exposure = pytest.importorskip("skimage.exposure")
        for index in (0, 4, 8):
            image = _ni_signal_data().reshape(-1, 60, 60)[index]
            ours = _preprocessing._mosaic_ahe(image, n_regions)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theirs = 255 * exposure.equalize_adapthist(
                    image, kernel_size=kernel_size, clip_limit=0, nbins=256
                )
            difference = float(np.abs(ours - theirs).max())
            record_property(
                f"clahe_{n_regions}_pattern{index}", f"max {difference:.4f}"
            )
            assert difference < tolerance
            assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99999

    def test_non_dividing_tiles_divergence_from_skimage_is_recorded(
        self, record_property
    ):
        exposure = pytest.importorskip("skimage.exposure")
        image = _ni_signal_data()[0, 0]
        ours = _preprocessing._mosaic_ahe(image, 7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = 255 * exposure.equalize_adapthist(
                image, kernel_size=(60 // 7, 60 // 7), clip_limit=0, nbins=256
            )
        record_property(
            "clahe_7_divergence",
            f"max {np.abs(ours - theirs).max():.2f} corr "
            f"{np.corrcoef(ours.ravel(), theirs.ravel())[0, 1]:.4f}",
        )

    def test_it_equals_kikuchipys_equalisation(self, record_property):
        signal = kp.data.nickel_ebsd_small()
        raw = signal.data.copy()
        signal.adaptive_histogram_equalization(
            kernel_size=(6, 6), clip_limit=0, nbins=256, show_progressbar=False
        )
        worst = 0.0
        for index in range(9):
            image = raw.reshape(-1, 60, 60)[index]
            ours = _preprocessing._mosaic_ahe(image, 10)
            rescaled = 255 * (ours - ours.min()) / np.ptp(ours)
            theirs = signal.data.reshape(-1, 60, 60)[index].astype(np.float64)
            worst = max(worst, float(np.abs(rescaled - theirs).max()))
        record_property("kikuchipy_ahe_difference", f"{worst:.3f}")
        assert worst < 2.0

    @pytest.mark.parametrize("n_regions", [-1, 61])
    def test_n_regions_is_validated(self, n_regions):
        with pytest.raises(ValueError):
            _preprocessing._mosaic_ahe(np.zeros((60, 60), dtype=np.uint8), n_regions)

    def test_good_pixels_is_validated(self):
        with pytest.raises(ValueError):
            _preprocessing._mosaic_ahe(
                np.zeros((60, 60), dtype=np.uint8), 10, np.zeros((60, 61), dtype=bool)
            )

    def test_the_image_is_validated(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            _preprocessing._mosaic_ahe(np.zeros(60, dtype=np.uint8), 10)
        # the kernels index ``cdfs`` with the pixel value, so
        # anything wider than 8 bits would read out of bounds
        with pytest.raises(ValueError, match="unsigned 8-bit"):
            _preprocessing._mosaic_ahe(np.zeros((60, 60), dtype=np.uint16), 10)


# ---------------------------- To uint8 ------------------------------ #


class TestToUint8:
    @pytest.mark.parametrize(
        "buffer, expected",
        [
            (np.array([0.0, 1.0, 6.0]), [0, 43, 255]),
            (np.array([0.0, 3.0, 6.0]), [0, 128, 255]),
            (np.array([10.0, 10.5, 12.0], dtype=np.float32), [0, 64, 255]),
        ],
    )
    def test_it_rounds_half_away_from_zero(self, buffer, expected):
        result = _preprocessing._to_uint8(buffer)
        assert result.dtype == np.uint8
        assert list(result) == expected

    def test_a_flat_buffer_becomes_zeros_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _preprocessing._to_uint8(np.full(10, 3.0))
        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_it_is_not_kikuchipys_rescale_intensity(self):
        buffer = np.array([0.0, 1.0, 6.0])
        theirs = kp.pattern.rescale_intensity(buffer, dtype_out=np.uint8)
        assert list(theirs) == [0, 42, 255]
        assert not np.array_equal(_preprocessing._to_uint8(buffer), theirs)

    @pytest.mark.parametrize("shape", [(0,), (0, 5), (5, 0)])
    def test_an_empty_buffer_raises_instead_of_reading_out_of_bounds(self, shape):
        # the kernel seeds from ``buffer[0]`` with bounds checking
        # off, so this would be an out of bounds read
        with pytest.raises(ValueError, match="empty"):
            _preprocessing._to_uint8(np.zeros(shape))


# --------------------------- Process order -------------------------- #


class TestProcessOrder:
    @staticmethod
    def _spies(monkeypatch):
        """Record the order in which the two stages are called."""
        order = []
        seen = {}
        real_background = _preprocessing._gaussian_background
        real_ahe = _preprocessing._mosaic_ahe

        def background(*args, **kwargs):
            order.append("background")
            return real_background(*args, **kwargs)

        def ahe(pattern_uint8, *args, **kwargs):
            order.append("ahe")
            seen["ahe_input"] = np.asarray(pattern_uint8).copy()
            return real_ahe(pattern_uint8, *args, **kwargs)

        monkeypatch.setattr(_preprocessing, "_gaussian_background", background)
        monkeypatch.setattr(_preprocessing, "_mosaic_ahe", ahe)
        return order, seen

    @pytest.mark.parametrize(
        "gaussian_background, n_regions, expected",
        [
            (True, 10, ["background", "ahe"]),
            (False, 10, ["ahe"]),
            (True, 0, ["background"]),
            (False, 0, []),
        ],
    )
    def test_the_order_of_the_stages(
        self, monkeypatch, gaussian_background, n_regions, expected
    ):
        order, _ = self._spies(monkeypatch)
        pattern = _ni_signal_data()[0, 0]
        result = _preprocessing._preprocess_pattern(
            pattern, gaussian_background=gaussian_background, n_regions=n_regions
        )
        assert order == expected
        assert result.dtype == np.float64
        assert result.shape == pattern.shape
        if not gaussian_background and n_regions == 0:
            assert np.array_equal(result, pattern.astype(np.float64))

    def test_an_unsigned_eight_bit_input_skips_the_rescale(self, monkeypatch):
        order, seen = self._spies(monkeypatch)
        pattern = _ni_signal_data()[0, 0]
        assert pattern.dtype == np.uint8
        assert pattern.min() > 0 and pattern.max() < 255
        _preprocessing._preprocess_pattern(pattern, n_regions=10)
        assert order == ["ahe"]
        assert np.array_equal(seen["ahe_input"], pattern)

        order.clear()
        _preprocessing._preprocess_pattern(pattern.astype(np.float32), n_regions=10)
        assert order == ["ahe"]
        assert not np.array_equal(seen["ahe_input"], pattern)
        assert seen["ahe_input"].min() == 0
        assert seen["ahe_input"].max() == 255

    def test_the_defaults_are_the_indexebsd_ones(self):
        signature = inspect.signature(_preprocessing._preprocess_pattern)
        defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        assert defaults == {
            "good_pixels": None,
            "gaussian_background": False,
            "n_regions": 10,
            "emsphinx_compatible": True,
        }

    def test_the_input_is_never_modified(self):
        pattern = _ni_signal_data()[0, 0].copy()
        reference = pattern.copy()
        _preprocessing._preprocess_pattern(
            pattern, gaussian_background=True, n_regions=10
        )
        assert np.array_equal(pattern, reference)

    @pytest.mark.parametrize("n_regions", [-1, 61])
    def test_n_regions_is_validated(self, n_regions):
        with pytest.raises(ValueError):
            _preprocessing._preprocess_pattern(
                _ni_signal_data()[0, 0], n_regions=n_regions
            )

    @pytest.mark.parametrize("shape", [(60,), (3, 60, 60)])
    def test_a_non_two_dimensional_pattern_raises(self, shape):
        with pytest.raises(ValueError, match="two-dimensional"):
            _preprocessing._preprocess_pattern(np.zeros(shape, dtype=np.uint8))

    @pytest.mark.parametrize(
        "gaussian_background, n_regions",
        [(True, 10), (False, 10), (True, 0), (False, 0)],
    )
    def test_good_pixels_is_validated_in_every_branch(
        self, gaussian_background, n_regions
    ):
        # three of the four branches hand the mask to
        # ``_mosaic_ahe`` or ``_remove_gaussian_background``, which
        # validate it themselves; the fourth returns the float cast
        # and would accept a wrong-shape mask silently without the
        # entry check
        with pytest.raises(ValueError):
            _preprocessing._preprocess_pattern(
                _ni_signal_data()[0, 0],
                good_pixels=np.zeros((60, 61), dtype=bool),
                gaussian_background=gaussian_background,
                n_regions=n_regions,
            )


# ------------------------- Kernels and flags ------------------------ #


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # the flag and py_func tests are parametrised over the
        # literal list above, so a kernel added during the
        # implementation would silently escape both of them
        assert _njit_kernel_names(_preprocessing) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _preprocessing"
        )

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        kernel = getattr(_preprocessing, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_only_the_gaussian_fit_uses_the_numpy_error_model(self, name):
        # the C++ divides 0 / 0 in the estimate and in the stopping
        # metric and relies on IEEE semantics to reach its fallback;
        # ``_cholesky_solve_3x3`` never divides by an exact zero
        kernel = getattr(_preprocessing, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        expected = "numpy" if name == "_fit_gaussian_1d_kernel" else None
        assert kernel.targetoptions.get("error_model") == expected

    def test_row_col_max_kernel_py_func(self):
        kernel = _preprocessing._row_col_max_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(10)
        pattern = rng.uniform(0, 255, (60, 60))
        good = rng.random((60, 60)) > 0.3
        results = []
        for function in (kernel, _py_func(kernel)):
            row_max = np.empty(60)
            col_max = np.empty(60)
            function(pattern, good, row_max, col_max)
            results.append((row_max, col_max))
        assert np.array_equal(results[0][0], results[1][0])
        assert np.array_equal(results[0][1], results[1][1])

    @pytest.mark.parametrize(
        "a, b",
        [
            (
                np.array([[4.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]]),
                np.array([1.0, 2.0, 3.0]),
            ),
            (np.diag([2.0, 5.0, 7.0]), np.array([-1.0, 0.5, 2.0])),
        ],
    )
    def test_cholesky_py_func(self, a, b):
        kernel = _preprocessing._cholesky_solve_3x3
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        results = []
        for function in (kernel, _py_func(kernel)):
            solution = np.zeros(3)
            status = function(a.copy(), b, solution)
            results.append((status, solution))
        assert results[0][0] == results[1][0]
        assert np.abs(results[0][1] - results[1][1]).max() <= 4 * EPS

    @pytest.mark.parametrize(
        "a, b, c", [(25.3, 288.0, 150.0), (40.7, 60.5, 90.0), (30.0, 128.0, 200.0)]
    )
    def test_fit_kernel_py_func(self, a, b, c):
        kernel = _preprocessing._fit_gaussian_1d_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        y = gaussian_samples(a, b, c)
        results = []
        for function in (kernel, _py_func(kernel)):
            params = np.zeros(4)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                status = function(y, params)
            results.append((status, params))
        assert results[0][0] == results[1][0]
        assert results[0][1][:3] == pytest.approx(results[1][1][:3], rel=1e-9)

    def test_ahe_kernels_py_func(self):
        cdf_kernel = _preprocessing._ahe_cdf_kernel
        equalize_kernel = _preprocessing._ahe_equalize_kernel
        assert hasattr(cdf_kernel, "py_func"), "kernel must be @njit-decorated"
        assert hasattr(equalize_kernel, "py_func"), "kernel must be @njit-decorated"
        image = _ni_signal_data()[0, 0]
        good = _preprocessing._circular_mask((60, 60))
        tiles, j_pairs, i_pairs = _preprocessing._ahe_tiles((60, 60), 10)
        cdfs = []
        for function in (cdf_kernel, _py_func(cdf_kernel)):
            buffer = np.zeros((100, 256))
            function(image, good, True, tiles, buffer)
            cdfs.append(buffer)
        assert np.array_equal(cdfs[0], cdfs[1])

        outputs = []
        for function in (equalize_kernel, _py_func(equalize_kernel)):
            out = np.zeros((60, 60))
            function(image, cdfs[0], 10, *j_pairs, *i_pairs, out)
            outputs.append(out)
        assert np.array_equal(outputs[0], outputs[1])

    def test_to_uint8_kernel_py_func(self):
        kernel = _preprocessing._to_uint8_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(11)
        buffer = rng.uniform(-10, 10, 500)
        results = []
        for function in (kernel, _py_func(kernel)):
            out = np.zeros(500, dtype=np.uint8)
            function(buffer, out)
            results.append(out)
        assert np.array_equal(results[0], results[1])

    def test_the_branches_only_the_compiled_kernel_reaches(self):
        # each of these is exercised only through the *compiled*
        # kernel elsewhere, which no coverage tool can see into
        to_uint8 = _preprocessing._to_uint8_kernel
        assert hasattr(to_uint8, "py_func"), "kernel must be @njit-decorated"
        for function in (to_uint8, _py_func(to_uint8)):
            out = np.full(9, 3, dtype=np.uint8)
            function(np.full(9, 7.0), out)
            # ``255 / 0`` is infinite and the C++ cast undefined
            assert np.array_equal(out, np.zeros(9, dtype=np.uint8))

        cdf_kernel = _preprocessing._ahe_cdf_kernel
        assert hasattr(cdf_kernel, "py_func"), "kernel must be @njit-decorated"
        image = column_ramp()
        good = np.zeros((60, 60), dtype=bool)
        good[:6, :6] = True
        tiles, _, _ = _preprocessing._ahe_tiles((60, 60), 10)
        # the identity ramp of the flat histogram, lines 211-213
        expected = 255 * np.arange(1, 257) / 256
        for function in (cdf_kernel, _py_func(cdf_kernel)):
            cdfs = np.zeros((100, 256))
            function(image, good, True, tiles, cdfs)
            assert np.abs(cdfs[55] - expected).max() <= 1e-12
            assert not np.allclose(cdfs[0], expected)

        # the unmasked histogram branch is the other half of the
        # ``has_mask`` fork, and it is equivalent to the masked one
        # under an all-``True`` mask -- recorded, since a mutant
        # forcing the masked branch is an equivalent mutation
        keep_all = np.ones((60, 60), dtype=bool)
        branches = []
        for function in (cdf_kernel, _py_func(cdf_kernel)):
            for has_mask in (False, True):
                cdfs = np.zeros((100, 256))
                function(image, keep_all, has_mask, tiles, cdfs)
                branches.append(cdfs)
        for other in branches[1:]:
            assert np.array_equal(branches[0], other)


class TestBaselines:
    @pytest.mark.parametrize(
        "gaussian_background, n_regions",
        [(False, 10), (True, 10), (True, 0), (False, 0)],
    )
    def test_preprocess_timing_is_recorded(
        self, gaussian_background, n_regions, record_property
    ):
        pattern = _ni_signal_data()[0, 0]
        _preprocessing._preprocess_pattern(
            pattern, gaussian_background=gaussian_background, n_regions=n_regions
        )  # warm the Numba cache
        best = math.inf
        for _ in range(5):
            start = time.perf_counter()
            result = _preprocessing._preprocess_pattern(
                pattern,
                gaussian_background=gaussian_background,
                n_regions=n_regions,
            )
            best = min(best, time.perf_counter() - start)
        record_property(
            f"preprocess_seconds_bg{gaussian_background}_n{n_regions}", f"{best:.5f}"
        )
        record_property(
            f"preprocess_range_bg{gaussian_background}_n{n_regions}",
            f"[{result.min():.2f}, {result.max():.2f}]",
        )
        assert best < 1.0
