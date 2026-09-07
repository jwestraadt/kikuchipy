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

"""Tests of ``kikuchipy.indexing._hrebsd._interpolation``.

Covers oracle V0 of ``specs/2026-09-07-hrebsd-dic/validation.md``:
the hand written numba bicubic B-spline kernel against
``scipy.ndimage.map_coordinates(order=3, prefilter=False,
mode="mirror")``, the same through ``.py_func``, the analytic
gradient kernels, the mirror boundary and the 32-bit coefficient
storage arm of requirements D17, plus the D3 bicubic-versus-quintic
order harness of plan open question 1 (added 2026-09-07 at the
adversarial review, which found that the Stage A gate's order A/B
decision had no test at all).

Written before the implementation exists: every test here calls the
skeleton and therefore fails with ``NotImplementedError`` until the
kernel lands, then passes unchanged.
"""

import time

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter

from kikuchipy.indexing._hrebsd import _interpolation
from kikuchipy.indexing._hrebsd._interpolation import (
    BOUNDARY_MODE,
    SPLINE_ORDERS,
    SUPPORTED_INTERPOLATION,
    _bicubic_evaluate,
    _bicubic_evaluate_gradient,
    evaluate,
    evaluate_gradient,
    gradient_planes,
    spline_coefficients,
)

# ------------------------- Frozen constants ------------------------- #

# The V0 equality bound, FROZEN, not measured: the kernel and
# ``map_coordinates`` evaluate the same basis on the same 64-bit
# coefficients, so the only difference is summation order.  Applied
# as a scale relative absolute band, ``atol = KERNEL_TOL * max|ref|``,
# because a spline value may pass through zero
KERNEL_TOL = 1e-12

# Pattern shapes exercised, deliberately including a NON-SQUARE one:
# a square-only suite cannot see a row/column transposition anywhere
# in the kernel or in the coefficient prefilter
SHAPES = [(48, 48), (37, 61)]

# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders (requirements D19).  Each is replaced by a
# dated measured value at the Stage A implementation gate and the
# measurement recorded in validation.md "Recorded results".

# MTP [D17/V0]: the band of the 32-bit coefficient storage arm against
# the same 64-bit reference.  ``map_coordinates`` on 32-bit input
# computes AND returns 32-bit, so the V0 1e-12 bound is unreachable
# there and this arm carries its own bound.  Expected class: 32-bit
# ULP of the pattern intensity scale.
# MEASURING RECIPE: ``test_kernel_f32_coefficient_storage`` below,
# run alongside the ``test_dtype_ab_harness`` A/B of
# ``test_hrebsd_engine.py`` at the implementation gate; record the
# worst scale relative error over all SHAPES and both seeds
KERNEL_F32_TOL = None

# MTP [D3/V0]: the band of the analytic gradient kernels against a
# central difference of the reference ``map_coordinates`` evaluation.
# The bound is set by the finite difference truncation, not by the
# kernel, so it cannot be reasoned to a priori.
# MEASURING RECIPE: ``test_gradient_kernels_match_spline_derivative``
# below; record the worst scale relative error at the pinned step
GRADIENT_FINITE_DIFFERENCE_TOL = None

# The central difference step of the gradient oracle, in pixels
GRADIENT_FINITE_DIFFERENCE_STEP = 1e-4

# MTP [D3/V0]: how far outside the pattern's OWN intensity range a
# mirrored evaluation may stray, as a fraction of that range.  A
# cubic B-spline interpolant overshoots at a step, so it cannot be
# zero, and the drafted ``< 4 * span`` band was unfalsifiable: no
# mirror-mode implementation can violate it, so the extrapolation
# blow-up it claims to catch would have gone unnoticed.
# MEASURING RECIPE: ``test_no_extrapolation_blow_up`` below; record
# the worst excursion over the swept line and both edges
MIRROR_OVERSHOOT_TOL = None

# The D3 order decision of plan open question 1, quoted literally:
# bicubic stays the default unless quintic buys more than this much
# accuracy at less than this much cost
ORDER_AB_ACCURACY_FACTOR = 2.0
ORDER_AB_COST_FACTOR = 1.5


# ----------------------------- Helpers ------------------------------ #


def assert_within(measured: float, bound, name: str) -> None:
    """Assert ``measured <= bound``, failing loudly and informatively
    while *bound* is an unfilled measured-then-pinned placeholder."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled MEASURED-THEN-PINNED placeholder "
            f"(requirements D19); measured {measured!r}. Fill it with "
            "a dated value and record the recipe in validation.md"
        )
    assert measured <= bound, f"{name}: {measured} > {bound}"


def random_image(shape, seed=0):
    """Return a smooth random pattern, band limited by a small box
    blur so the cubic spline is a fair approximation of it."""
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, size=shape)
    kernel = np.ones((3, 3)) / 9.0
    padded = np.pad(image, 1, mode="reflect")
    smooth = np.zeros_like(image)
    for i in range(3):
        for j in range(3):
            smooth += kernel[i, j] * padded[i : i + shape[0], j : j + shape[1]]
    return np.ascontiguousarray(smooth, dtype=np.float64)


def analytic_image(x, y, shape, n_waves=6, seed=99):
    """Return a band-limited analytic image at arbitrary ``(x, y)``.

    A short sum of cosines, so the exact value is known everywhere and
    both interpolation orders can be measured against the FUNCTION
    rather than against each other.  The highest frequency is six
    cycles across the pattern, about 11 px per cycle at the shape used
    here, which is the regime a Kikuchi band's own gradients sit in.
    """
    nrows, ncols = shape
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.zeros(np.shape(x), dtype=np.float64)
    for k in range(1, n_waves + 1):
        u, v = rng.integers(1, n_waves + 1, size=2)
        phase = rng.uniform(0.0, 2 * np.pi)
        values += (1.0 / k) * np.cos(
            2 * np.pi * (u * x / ncols + v * y / nrows) + phase
        )
    return values


def random_points(shape, n=500, margin=4.0, seed=1):
    """Return random interior sample coordinates ``(x, y)``, columns
    first, kept *margin* pixels away from every edge."""
    rng = np.random.default_rng(seed)
    nrows, ncols = shape
    x = rng.uniform(margin, ncols - 1 - margin, size=n)
    y = rng.uniform(margin, nrows - 1 - margin, size=n)
    return x, y


def reference_evaluate(coefficients, x, y):
    """Return the frozen V0 reference: ``map_coordinates`` on the same
    coefficients with the prefilter switched off."""
    coordinates = np.vstack([np.ravel(y), np.ravel(x)])
    values = map_coordinates(
        coefficients,
        coordinates,
        order=SPLINE_ORDERS["bicubic"],
        prefilter=False,
        mode=BOUNDARY_MODE,
    )
    return values.reshape(np.shape(x))


def scale_relative_error(got, expected):
    """Return ``max|got - expected| / max|expected|``, the metric of
    every band in this module."""
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.abs(expected).max())
    if scale == 0.0:
        scale = 1.0
    return float(np.abs(got - expected).max() / scale)


# ======================= Coefficients (D3) ========================== #


class TestSplineCoefficients:
    """The prefilter is ``scipy.ndimage.spline_filter`` itself, not a
    reimplementation of it.  [D3]"""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_coefficients_are_scipy_spline_filter(self, shape):
        # requirements D3 names the exact call, so the equality is
        # bitwise: a hand rolled prefilter with a different boundary
        # treatment would break the V0 equality below in a way that
        # is far harder to diagnose from the kernel test alone
        image = random_image(shape, seed=shape[0])
        expected = spline_filter(
            image, order=SPLINE_ORDERS["bicubic"], mode=BOUNDARY_MODE
        )
        got = spline_coefficients(image, dtype=np.float64)
        assert got.shape == image.shape
        assert got.dtype == np.float64
        assert np.array_equal(got, expected)

    def test_coefficients_f32_storage(self):
        # the D17 provisional bulk storage type: the coefficients are
        # the 64-bit ones cast down, never a 32-bit prefilter
        image = random_image(SHAPES[0], seed=5)
        f64 = spline_coefficients(image, dtype=np.float64)
        f32 = spline_coefficients(image, dtype=np.float32)
        assert f32.dtype == np.float32
        assert np.array_equal(f32, f64.astype(np.float32))

    def test_unsupported_interpolation_raises(self):
        # the keyword survives whichever order the D3 measurement
        # pins, so an unknown name must name the accepted set
        image = random_image((16, 16))
        with pytest.raises(ValueError, match="bicubic"):
            spline_coefficients(image, interpolation="bilinear")

    def test_supported_names_and_orders(self):
        # frozen surface: bicubic is the default in force (plan open
        # question 1), and its order is three
        assert SUPPORTED_INTERPOLATION == ("bicubic",)
        assert SPLINE_ORDERS["bicubic"] == 3
        assert BOUNDARY_MODE == "mirror"


# ================ V0 -- kernel versus map_coordinates =============== #


class TestKernelEquality:
    """The numba kernel reproduces ``map_coordinates(order=3,
    prefilter=False, mode="mirror")`` on 64-bit coefficients.
    [D3/V0]"""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", [0, 7])
    def test_kernel_matches_map_coordinates(self, shape, seed):
        image = random_image(shape, seed=seed)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, seed=seed + 1)
        got = evaluate(coefficients, x, y)
        expected = reference_evaluate(coefficients, x, y)
        assert got.shape == x.shape
        assert got.dtype == np.float64
        atol = KERNEL_TOL * float(np.abs(expected).max())
        np.testing.assert_allclose(got, expected, rtol=0, atol=atol)

    def test_kernel_py_func(self):
        # the constitution's numba rule: every kernel is also tested
        # through its pure Python twin
        shape = SHAPES[1]
        image = random_image(shape, seed=3)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, seed=4)
        out = np.zeros(x.size, dtype=np.float64)
        _bicubic_evaluate.py_func(coefficients, x, y, out)
        expected = reference_evaluate(coefficients, x, y)
        atol = KERNEL_TOL * float(np.abs(expected).max())
        np.testing.assert_allclose(out, expected, rtol=0, atol=atol)

    def test_kernel_and_py_func_agree(self):
        # compiled and interpreted arms of the same source
        shape = SHAPES[0]
        image = random_image(shape, seed=11)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=64, seed=12)
        compiled = np.zeros(x.size, dtype=np.float64)
        interpreted = np.zeros(x.size, dtype=np.float64)
        _bicubic_evaluate(coefficients, x, y, compiled)
        _bicubic_evaluate.py_func(coefficients, x, y, interpreted)
        assert np.array_equal(compiled, interpreted)

    def test_column_row_convention_pinned(self):
        # requirements D1.1: ``x`` is the COLUMN, ``y`` the ROW.  A
        # non-square image plus the deliberately swapped reference
        # kills the transposition mutant without any tolerance
        shape = (37, 61)
        image = random_image(shape, seed=21)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=128, seed=22)
        got = evaluate(coefficients, x, y)
        correct = reference_evaluate(coefficients, x, y)
        atol = KERNEL_TOL * float(np.abs(correct).max())
        np.testing.assert_allclose(got, correct, rtol=0, atol=atol)
        # the swapped reference is a genuinely different function
        swapped = map_coordinates(
            coefficients,
            np.vstack([x, y]),
            order=3,
            prefilter=False,
            mode=BOUNDARY_MODE,
        )
        assert not np.allclose(got, swapped)

    def test_shape_is_preserved(self):
        # the wrapper takes coordinates of any shape and returns that
        # shape, which the engine relies on for its subregion planes
        shape = SHAPES[0]
        image = random_image(shape, seed=31)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=60, seed=32)
        x2 = x.reshape(6, 10)
        y2 = y.reshape(6, 10)
        assert evaluate(coefficients, x2, y2).shape == (6, 10)

    def test_out_argument_is_written_in_place(self):
        shape = SHAPES[0]
        image = random_image(shape, seed=41)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=32, seed=42)
        out = np.zeros(x.shape, dtype=np.float64)
        returned = evaluate(coefficients, x, y, out=out)
        assert returned is out
        assert np.any(out != 0.0)

    def test_order_ab_harness(self):
        # the D3 measurement harness of plan open question 1, run
        # where the accuracy difference between the two orders
        # ORIGINATES: both interpolators against an ANALYTIC band
        # limited ground truth, so neither is judged by the other.
        # Ruggles 2018 found biquintic bought nothing over bicubic at
        # 960 px; Ernould and EMsoftOO use quintic
        # (``mod_DIC.f90:50-51``).  Bicubic stays the default in
        # force unless quintic buys more accuracy than
        # ORDER_AB_ACCURACY_FACTOR at less than ORDER_AB_COST_FACTOR
        # of the cost, and a re-pin is a dated record in
        # requirements D3 and validation.md
        shape = (64, 96)
        nrows, ncols = shape
        rows, cols = np.indices(shape, dtype=np.float64)
        image = analytic_image(cols, rows, shape)
        x, y = random_points(shape, n=400, margin=8.0, seed=61)
        truth = analytic_image(x, y, shape)

        coefficients = spline_coefficients(image, dtype=np.float64)
        bicubic = evaluate(coefficients, x, y)
        quintic = map_coordinates(image, np.vstack([y, x]), order=5, mode=BOUNDARY_MODE)
        bicubic_error = scale_relative_error(bicubic, truth)
        quintic_error = scale_relative_error(quintic, truth)

        repeats = 20
        start = time.perf_counter()
        for _ in range(repeats):
            evaluate(coefficients, x, y)
        bicubic_cost = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repeats):
            map_coordinates(image, np.vstack([y, x]), order=5, mode=BOUNDARY_MODE)
        quintic_cost = time.perf_counter() - start

        accuracy_gain = bicubic_error / max(quintic_error, np.finfo(float).tiny)
        cost_ratio = quintic_cost / max(bicubic_cost, np.finfo(float).tiny)
        assert not (
            accuracy_gain > ORDER_AB_ACCURACY_FACTOR
            and cost_ratio < ORDER_AB_COST_FACTOR
        ), (
            f"quintic buys {accuracy_gain:.2f}x accuracy (bicubic "
            f"{bicubic_error:.3e}, quintic {quintic_error:.3e}) at "
            f"{cost_ratio:.2f}x cost, which meets the D3 re-pin "
            "criterion; re-pin the default with a dated measurement "
            "in requirements D3 and validation.md"
        )

    def test_kernel_f32_coefficient_storage(self):
        # the D17 arm: 32-bit coefficients against the 64-bit
        # reference under their OWN measured band, never under the
        # 1e-12 of the 64-bit arm
        shape = SHAPES[0]
        image = random_image(shape, seed=51)
        coefficients64 = spline_coefficients(image, dtype=np.float64)
        coefficients32 = spline_coefficients(image, dtype=np.float32)
        x, y = random_points(shape, seed=52)
        got = evaluate(coefficients32, x, y)
        assert got.dtype == np.float64  # accumulation stays 64-bit
        expected = reference_evaluate(coefficients64, x, y)
        error = scale_relative_error(got, expected)
        assert_within(error, KERNEL_F32_TOL, "KERNEL_F32_TOL")


class TestMirrorBoundary:
    """Out-of-frame samples fold back through the ``"mirror"``
    boundary, never through b-spline extrapolation (a recorded
    deviation from EMsoftOO's ``mod_DIC.f90:50-51``).  [D3/V0]"""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_mirror_boundary(self, shape):
        image = random_image(shape, seed=61)
        coefficients = spline_coefficients(image, dtype=np.float64)
        nrows, ncols = shape
        # edge, one-past-the-edge and well outside on every side
        x = np.array([-3.4, -0.5, 0.0, 0.5, ncols - 1.0, ncols + 2.7, 1.0, 1.0])
        y = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -2.2, nrows + 1.3])
        got = evaluate(coefficients, x, y)
        expected = reference_evaluate(coefficients, x, y)
        assert np.all(np.isfinite(got))
        atol = KERNEL_TOL * float(np.abs(expected).max())
        np.testing.assert_allclose(got, expected, rtol=0, atol=atol)

    def test_no_extrapolation_blow_up(self):
        # a b-spline extrapolating instead of mirroring diverges
        # rapidly outside the frame; the mirrored values stay inside
        # the range of the pattern ITSELF, plus the spline's own
        # overshoot at a step.  The drafted band (``4 * span`` about
        # the mean) could not be violated by any mirror-mode
        # implementation and so tested nothing
        shape = SHAPES[0]
        image = random_image(shape, seed=71)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x = np.linspace(-12.0, shape[1] + 12.0, 200)
        y = np.full_like(x, shape[0] / 2.0)
        got = evaluate(coefficients, x, y)
        span = float(image.max() - image.min())
        assert np.all(np.isfinite(got))
        overshoot = max(
            float(got.max() - image.max()), float(image.min() - got.min()), 0.0
        )
        assert_within(overshoot / span, MIRROR_OVERSHOOT_TOL, "MIRROR_OVERSHOOT_TOL")


# ==================== V0 -- analytic gradients ====================== #


class TestGradients:
    """The reference gradients are the analytic B-spline derivatives
    of the same coefficients, never a finite difference of the
    sampled image.  [D2.1/D3/V0]"""

    def test_gradient_kernels_match_spline_derivative(self):
        shape = SHAPES[1]
        image = random_image(shape, seed=81)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=200, margin=6.0, seed=82)
        values, gx, gy = evaluate_gradient(coefficients, x, y)
        # the values arm is the frozen V0 equality
        expected_values = reference_evaluate(coefficients, x, y)
        atol = KERNEL_TOL * float(np.abs(expected_values).max())
        np.testing.assert_allclose(values, expected_values, rtol=0, atol=atol)
        # the derivative arm against a central difference of the same
        # reference, whose band is measured then pinned
        step = GRADIENT_FINITE_DIFFERENCE_STEP
        gx_ref = (
            reference_evaluate(coefficients, x + step, y)
            - reference_evaluate(coefficients, x - step, y)
        ) / (2 * step)
        gy_ref = (
            reference_evaluate(coefficients, x, y + step)
            - reference_evaluate(coefficients, x, y - step)
        ) / (2 * step)
        error = max(scale_relative_error(gx, gx_ref), scale_relative_error(gy, gy_ref))
        assert_within(
            error,
            GRADIENT_FINITE_DIFFERENCE_TOL,
            "GRADIENT_FINITE_DIFFERENCE_TOL",
        )

    def test_gradient_axes_are_not_swapped(self):
        # an image varying ONLY along the columns has a zero row
        # derivative everywhere in the interior; the swapped-gradient
        # mutant of plan 2.5 dies here with no tolerance to argue
        # about beyond the frozen 1e-12
        ncols = 64
        ramp = np.tile(np.linspace(0.0, 1.0, ncols), (48, 1))
        coefficients = spline_coefficients(ramp, dtype=np.float64)
        x = np.linspace(10.0, ncols - 11.0, 50)
        y = np.full_like(x, 24.0)
        _, gx, gy = evaluate_gradient(coefficients, x, y)
        assert np.all(np.abs(gx) > 1e-3)
        assert float(np.abs(gy).max()) <= KERNEL_TOL * float(np.abs(gx).max())

    def test_gradient_py_func(self):
        shape = SHAPES[0]
        image = random_image(shape, seed=91)
        coefficients = spline_coefficients(image, dtype=np.float64)
        x, y = random_points(shape, n=64, seed=92)
        values = np.zeros(x.size)
        gx = np.zeros(x.size)
        gy = np.zeros(x.size)
        _bicubic_evaluate_gradient.py_func(coefficients, x, y, values, gx, gy)
        compiled = tuple(np.zeros(x.size) for _ in range(3))
        _bicubic_evaluate_gradient(coefficients, x, y, *compiled)
        assert np.array_equal(values, compiled[0])
        assert np.array_equal(gx, compiled[1])
        assert np.array_equal(gy, compiled[2])

    def test_gradient_planes_equal_grid_evaluation(self):
        # the precompute evaluates the same analytic derivative on the
        # integer pixel grid, so the two routes must agree to the
        # frozen algebra band
        shape = (24, 31)
        image = random_image(shape, seed=101)
        coefficients = spline_coefficients(image, dtype=np.float64)
        plane_x, plane_y = gradient_planes(coefficients)
        assert plane_x.shape == shape
        assert plane_y.shape == shape
        rows, cols = np.indices(shape, dtype=np.float64)
        _, gx, gy = evaluate_gradient(coefficients, cols.ravel(), rows.ravel())
        scale = float(np.abs(gx).max())
        np.testing.assert_allclose(plane_x.ravel(), gx, rtol=0, atol=KERNEL_TOL * scale)
        np.testing.assert_allclose(plane_y.ravel(), gy, rtol=0, atol=KERNEL_TOL * scale)


# ======================= Numba flags (D18) ========================== #


class TestKernelFlags:
    """The constitution's numba rules, asserted on the decorated
    objects themselves.  [D3/tech-stack]"""

    @pytest.mark.parametrize(
        "name", ["_bicubic_evaluate", "_bicubic_evaluate_gradient"]
    )
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        kernel = getattr(_interpolation, name)
        assert hasattr(kernel, "targetoptions"), f"{name} needs @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        # no parallelism inside a kernel: the dask threaded scheduler
        # owns every core, and fastmath waits until tolerances pass
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)
        # no IEEE error model: nothing here divides 0/0 on purpose
        assert kernel.targetoptions.get("error_model") is None

    @pytest.mark.parametrize(
        "name", ["_bicubic_evaluate", "_bicubic_evaluate_gradient"]
    )
    def test_kernels_expose_py_func(self, name):
        kernel = getattr(_interpolation, name)
        assert callable(kernel.py_func)
