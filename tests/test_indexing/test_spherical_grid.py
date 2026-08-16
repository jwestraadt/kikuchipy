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

"""Tests of ``kikuchipy.indexing._spherical._grid``.

Covers the "Grids" assertions of
``specs/2026-08-16-sht-square-grid-transform/validation.md``:

- ``validate_dim``: ``dim < 3`` and even ``dim`` raise ``ValueError``.
- ``normals(dim, layout)`` is unit length to 1e-15, the centre pixel is
  ``(0, 0, 1)``, and the asymmetric axis probes
  ``normals(dim, "lambert")[j, i] == square_to_sphere([i / (dim - 1),
  j / (dim - 1)])`` hold for ``(i, j)`` in ``{(1, 0), (0, 1),
  (dim - 1, 1)}``.
- ``ring_number`` equals the Chebyshev distance from the centre pixel.
- ``ring_indices``: ``offsets[0] == 0``,
  ``offsets[y + 1] - offsets[y] == max(1, 8 * y)``,
  ``offsets[-1] == dim * dim``, every pixel appears exactly once,
  ``flat[offsets[y]]`` is ``(dim // 2, dim // 2 + y)``,
  ``flat[offsets[y] + 1]`` is ``(dim // 2 + 1, dim // 2 + y)``, and the
  ring azimuths start at zero and increase by ``2 * pi / (8 * y)``.
- Legendre normals have ``z == cos_latitudes[ring]`` and azimuths
  identical to the Lambert normals of the same ``dim`` (1e-14).
- ``square_to_sphere(sphere_to_square(v)) == v`` (1e-13),
  ``sphere_to_square([0, 0, 1]) == (0.5, 0.5)``, and the two affine
  maps to kikuchipy's ``_lambert2vector``/``_vector2lambert`` (1e-12).
- ``lambert_cos_latitudes(dim) == 1 - (2 * y / (dim - 1)) ** 2``
  (``rtol=1e-15``); ``legendre_cos_latitudes(dim)[0] == 1``, strictly
  decreasing, last entry exactly ``0.0``; ``legendre_roots(n)`` equals
  a transcription of EMSphInx' bisection ``legendre::roots()`` to
  1e-13 for ``n`` in ``{5, 69, 385}``.
- Ring solid angle invariant (1e-12) for both layouts at ``dim`` in
  ``{9, 33, 101, 201}``.
- Lambert pixel solid angles at ``dim`` in ``{11, 21, 51, 101}``:
  positive, 8-fold symmetric, pole pixel ``2 / pi`` within 1 %, every
  pixel with ``ring >= 1`` within 6 % of
  ``n_grid_points / 2 / (dim - 1) ** 2``, and
  ``sum(interior) + sum(edge) / 2 + sum(corner) / 4 ==
  dim * dim - 2 * dim + 2`` with ``rtol=1e-6``.
- Weights: ``sum(w_hat[k]) == 1`` within ``cbrt(eps) / 64``,
  ``w_hat[k, k] == 0``,
  ``w[k, y] == 4 * pi * w_hat[k, y] / max(1, 8 * y)``; the Legendre
  weight sets are all identical and equal the Gauss-Legendre weights
  with a halved equator weight (absolute 1e-13, and 1e-14 for the
  halved equator weight itself) for ``dim`` in ``{19, 35, 101, 201,
  401}``; every Lambert weight set reproduces the Chebyshev moments
  it is solved from to 1e-10 for ``dim`` in ``{33, 65}``;
  ``quadrature_weights(201, "lambert")`` succeeds while
  ``quadrature_weights(401, "lambert")`` raises ``ValueError``, and the
  smallest tripping Lambert ``dim`` is recorded.
- ``.py_func`` variants of both Numba kernels.
"""

import math

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import leggauss
import pytest

from kikuchipy.indexing._spherical import _grid
from kikuchipy.signals.util._master_pattern import (
    _lambert2vector,
    _vector2lambert,
)

EPS = np.finfo(np.float64).eps
WEIGHT_SUM_TOLERANCE = np.cbrt(EPS) / 64


def _py_func(kernel):
    """Return the pure Python function of a Numba kernel.

    Falls back to the function itself while it is still an undecorated
    stub. Every caller first asserts that the kernel does carry a
    ``py_func``, so that an implementation without ``@njit`` fails
    loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


def _legendre_roots_bisection(n: int) -> list[float]:
    """Return the non-negative roots of ``P_n``, descending.

    Transcription of EMSphInx' ``square::legendre::roots()``
    (``include/sht/square_sht.hpp`` lines 746-818): the roots are the
    eigenvalues of the symmetric tridiagonal Jacobi matrix with a zero
    diagonal and the sub-diagonal ``i / sqrt(4 * i ** 2 - 1)``, found
    by bisection of a Sturm sequence (Barth, Martin and Wilkinson).
    """
    b = [0.0] * n
    beta = [0.0] * n
    for i in range(1, n):
        den = float(4 * i * i - 1)
        b[i] = i / math.sqrt(den)
        beta[i] = i * i / den

    m1 = n // 2
    eps1 = EPS
    relfeh = EPS
    beta[0] = b[0] = 0.0
    eps2 = relfeh
    eps2 = eps1 / 2 + 7 * eps2  # noqa: F841  (unused, as in EMSphInx)

    x = [1.0] * (m1 + 1)
    wu = [0.0] * (m1 + 1)
    if n % 2 == 1:
        x[m1] = 0.0
    x0 = 1.0
    z = 0
    limit = n * 32
    for k in range(m1):
        xu = 0.0
        for i in range(k, m1):
            if xu < wu[i]:
                xu = wu[i]
                break
        if x0 > x[k]:
            x0 = x[k]
        while x0 - xu > relfeh * (abs(xu) + abs(x0)) * 2 + eps1:
            x1 = (xu + x0) / 2
            z += 1
            if z > limit:
                raise RuntimeError("too many iterations computing roots")
            a = n
            q = 1.0
            for i in range(n):
                if q != 0:
                    q = -(x1 + beta[i] / q)
                else:
                    q = -(x1 + abs(b[i]) / relfeh)
                if math.copysign(1.0, q) < 0:
                    a -= 1
            if a > k:
                if a >= m1:
                    xu = wu[n - 1 - m1] = x1
                else:
                    xu = wu[a - 1] = x1
                    if x[a] > x1:
                        x[a] = x1
            else:
                x0 = x1
            x[k] = (x0 + xu) / 2
    return x


def _azimuths(v: np.ndarray) -> np.ndarray:
    """Return ``atan2(y, x)`` of vectors of shape ``(..., 3)``."""
    return np.arctan2(v[..., 1], v[..., 0])


def _wrapped_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the difference of two angles wrapped to [-pi, pi)."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def _unscaled_weights(dim: int, layout: str, skip: int) -> np.ndarray:
    """Return one unscaled ring weight set."""
    return _grid._ring_weights_skip(dim, _grid.cos_latitudes(dim, layout), skip)


class TestDimensionValidation:
    @pytest.mark.parametrize("dim", [-1, 0, 1, 2])
    def test_validate_dim_raises_for_side_length_below_three(self, dim):
        with pytest.raises(ValueError):
            _grid.validate_dim(dim)

    @pytest.mark.parametrize("dim", [4, 10, 100, 400])
    def test_validate_dim_raises_for_even_side_length(self, dim):
        with pytest.raises(ValueError):
            _grid.validate_dim(dim)

    @pytest.mark.parametrize("dim", [3, 5, 11, 201, 401])
    def test_validate_dim_accepts_odd_side_length_of_at_least_three(self, dim):
        assert _grid.validate_dim(dim) is None

    @pytest.mark.parametrize(
        "dim, n_rings, n_grid_points",
        [
            (3, 2, 10),
            (9, 5, 130),
            (11, 6, 202),
            (201, 101, 80002),
            (401, 201, 320002),
        ],
    )
    def test_n_rings_and_n_grid_points_match_emsphinx(
        self, dim, n_rings, n_grid_points
    ):
        assert _grid.n_rings(dim) == n_rings
        assert _grid.n_grid_points(dim) == n_grid_points
        assert n_grid_points == 2 * dim * dim - 4 * (dim - 1)

    @pytest.mark.parametrize(
        "bandwidth, layout, dim",
        [
            (4, "legendre", 7),
            (5, "legendre", 7),
            (68, "legendre", 71),
            (63, "legendre", 65),
            (4, "lambert", 9),
            (32, "lambert", 65),
            (100, "lambert", 201),
        ],
    )
    def test_default_dim_matches_the_emsphinx_round_trip_test(
        self, bandwidth, layout, dim
    ):
        assert _grid.default_dim(bandwidth, layout) == dim

    @pytest.mark.parametrize(
        "dim, layout, bandwidth",
        [
            (71, "legendre", 69),
            (11, "legendre", 9),
            (65, "lambert", 32),
            (201, "lambert", 100),
        ],
    )
    def test_max_bandwidth_matches_emsphinx_constants_check(
        self, dim, layout, bandwidth
    ):
        assert _grid.max_bandwidth(dim, layout) == bandwidth

    @pytest.mark.parametrize("bandwidth", [0, -1])
    def test_default_dim_raises_below_bandwidth_one(self, bandwidth):
        with pytest.raises(ValueError, match="must be at least one"):
            _grid.default_dim(bandwidth, "legendre")

    @pytest.mark.parametrize(
        "function",
        ["cos_latitudes", "normals", "ring_solid_angles", "quadrature_weights"],
    )
    def test_unknown_layout_raises_value_error(self, function):
        with pytest.raises(ValueError):
            getattr(_grid, function)(9, "mollweide")


class TestSquareLambertMapping:
    def test_square_to_sphere_of_sphere_to_square_returns_the_vector(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(1000, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[v[:, 2] < 0] *= -1
        v2 = _grid.square_to_sphere(_grid.sphere_to_square(v))
        assert np.allclose(v2, v, atol=1e-13)

    def test_sphere_to_square_maps_the_north_pole_to_the_square_centre(self):
        xy = _grid.sphere_to_square(np.array([0.0, 0.0, 1.0]))
        assert np.allclose(xy, [0.5, 0.5], atol=1e-15)

    def test_square_to_sphere_raises_outside_the_unit_square(self):
        with pytest.raises(ValueError):
            _grid.square_to_sphere(np.array([1.5, 0.5]))

    def test_square_to_sphere_equals_lambert2vector_on_the_shifted_square(
        self,
    ):
        # _lambert2vector() takes coordinates in [-1, 1], the port
        # takes them in [0, 1]
        rng = np.random.default_rng(0)
        xy = rng.uniform(-1, 1, size=(500, 2))
        v_kp = _lambert2vector(
            np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
        )
        v_port = _grid.square_to_sphere((xy + 1) / 2)
        assert np.allclose(v_port, v_kp, atol=1e-12)

    def test_sphere_to_square_equals_the_rescaled_vector2lambert(self):
        # _vector2lambert() returns coordinates scaled by sqrt(pi / 2)
        # in [-1, 1], the port returns them in [0, 1]
        rng = np.random.default_rng(0)
        v = rng.normal(size=(500, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[v[:, 2] < 0] *= -1
        xy_kp = _vector2lambert(np.ascontiguousarray(v))
        xy_kp = (xy_kp / np.sqrt(np.pi / 2) + 1) / 2
        xy_port = _grid.sphere_to_square(v)
        assert np.allclose(xy_port, xy_kp, atol=1e-12)

    def test_square_to_sphere_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_grid._square_to_sphere_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 1, size=(100, 2))
        # The square centre is the pole branch, which random points
        # never reach
        xy = np.vstack([xy, [0.5, 0.5]])
        compiled = _grid._square_to_sphere_kernel(xy)
        interpreted = _py_func(_grid._square_to_sphere_kernel)(xy)
        assert np.allclose(compiled, interpreted, atol=1e-15)
        assert np.array_equal(interpreted[-1], [0.0, 0.0, 1.0])
        outside = np.array([[2.0, 0.5]])
        with pytest.raises(ValueError):
            _py_func(_grid._square_to_sphere_kernel)(outside)

    def test_sphere_to_square_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_grid._sphere_to_square_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        rng = np.random.default_rng(0)
        v = rng.normal(size=(100, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # Both poles are the |z| == 1 branch, which random unit
        # vectors never reach
        v = np.vstack([v, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        compiled = _grid._sphere_to_square_kernel(v)
        interpreted = _py_func(_grid._sphere_to_square_kernel)(v)
        assert np.allclose(compiled, interpreted, atol=1e-15)
        assert np.array_equal(interpreted[-2:], [[0.5, 0.5], [0.5, 0.5]])


class TestKernelCompileOptions:
    @pytest.mark.parametrize(
        "name", ["_square_to_sphere_kernel", "_sphere_to_square_kernel"]
    )
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        # Dropping either option leaves every other test passing, so
        # the private Numba attributes are read directly
        kernel = getattr(_grid, name)
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)


class TestRingLatitudes:
    @pytest.mark.parametrize("dim", [3, 9, 33, 101, 201, 401])
    def test_lambert_cos_latitudes_equal_the_closed_form(self, dim):
        cos_lats = _grid.lambert_cos_latitudes(dim)
        y = np.arange(_grid.n_rings(dim))
        # rtol 1e-14, not 1e-15: the port accumulates the integer
        # recursion of EMSphInx (one rounding) while the closed form
        # rounds three times and cancels catastrophically near the
        # equator, where it is up to 25 ulp (4.3e-15 relative) off
        assert np.allclose(cos_lats, 1 - (2 * y / (dim - 1)) ** 2, rtol=1e-14, atol=0)

    @pytest.mark.parametrize("dim", [9, 33, 101, 201])
    def test_legendre_cos_latitudes_run_from_pole_to_equator(self, dim):
        cos_lats = _grid.legendre_cos_latitudes(dim)
        assert cos_lats.size == _grid.n_rings(dim)
        assert cos_lats[0] == 1.0
        assert np.all(np.diff(cos_lats) < 0)
        assert cos_lats[-1] == 0.0

    @pytest.mark.parametrize("n", [5, 69, 385])
    def test_legendre_roots_equal_the_transcribed_bisection(self, n):
        roots = _grid.legendre_roots(n)
        reference = np.asarray(_legendre_roots_bisection(n))
        assert roots.size == reference.size
        assert np.allclose(roots, reference, atol=1e-13)

    @pytest.mark.parametrize("n", [5, 69, 385])
    def test_legendre_roots_are_descending_with_an_exact_zero_last(self, n):
        roots = _grid.legendre_roots(n)
        assert np.all(np.diff(roots) < 0)
        assert roots[-1] == 0.0

    @pytest.mark.parametrize("dim", [9, 33, 101])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_cos_latitudes_dispatches_on_the_layout(self, dim, layout):
        cos_lats = _grid.cos_latitudes(dim, layout)
        if layout == "lambert":
            expected = _grid.lambert_cos_latitudes(dim)
        else:
            expected = _grid.legendre_cos_latitudes(dim)
        assert np.array_equal(cos_lats, expected)


class TestNormals:
    @pytest.mark.parametrize("dim", [3, 9, 33, 101])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_normals_are_unit_vectors(self, dim, layout):
        v = _grid.normals(dim, layout)
        assert v.shape == (dim, dim, 3)
        assert np.allclose(np.linalg.norm(v, axis=-1), 1, atol=1e-15)

    @pytest.mark.parametrize("dim", [3, 9, 33, 101])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_centre_pixel_is_the_north_pole(self, dim, layout):
        v = _grid.normals(dim, layout)
        assert np.allclose(v[dim // 2, dim // 2], [0, 0, 1], atol=1e-15)

    @pytest.mark.parametrize("dim", [9, 33, 101])
    def test_lambert_normals_axis_probes_are_not_symmetric(self, dim):
        # Locks that axis 1 (columns) moves x and axis 0 (rows) moves
        # y, i.e. EMSphInx' j * dim + i indexing
        v = _grid.normals(dim, "lambert")
        for i, j in [(1, 0), (0, 1), (dim - 1, 1)]:
            expected = _grid.square_to_sphere(np.array([i / (dim - 1), j / (dim - 1)]))
            assert np.allclose(v[j, i], expected, atol=1e-15), f"({i}, {j})"

    @pytest.mark.parametrize("dim", [9, 33, 101])
    def test_legendre_normals_have_the_legendre_ring_latitudes(self, dim):
        v = _grid.normals(dim, "legendre")
        cos_lats = _grid.cos_latitudes(dim, "legendre")
        rings = _grid.ring_number(dim)
        assert np.allclose(v[..., 2], cos_lats[rings], atol=1e-14)

    @pytest.mark.parametrize("dim", [9, 33, 101])
    def test_legendre_normals_share_the_lambert_azimuths(self, dim):
        v_lam = _grid.normals(dim, "lambert")
        v_leg = _grid.normals(dim, "legendre")
        delta = _wrapped_difference(_azimuths(v_leg), _azimuths(v_lam))
        assert np.allclose(delta, 0, atol=1e-14)


class TestRingIndices:
    @pytest.mark.parametrize("dim", [3, 9, 33, 101])
    def test_ring_number_is_the_chebyshev_distance_to_the_centre(self, dim):
        rings = _grid.ring_number(dim)
        j, i = np.indices((dim, dim))
        expected = np.maximum(np.abs(i - dim // 2), np.abs(j - dim // 2)).astype(
            np.int64
        )
        assert rings.dtype == np.int64
        assert np.array_equal(rings, expected)

    @pytest.mark.parametrize("dim", [3, 9, 33, 101])
    def test_ring_offsets_hold_max_1_8y_points_per_ring(self, dim):
        offsets, flat = _grid.ring_indices(dim)
        assert offsets.shape == (_grid.n_rings(dim) + 1,)
        assert offsets[0] == 0
        assert offsets[-1] == dim * dim
        assert flat.shape == (dim * dim,)
        for y in range(_grid.n_rings(dim)):
            assert offsets[y + 1] - offsets[y] == max(1, 8 * y), f"ring {y}"

    @pytest.mark.parametrize("dim", [3, 9, 33, 101])
    def test_ring_indices_cover_every_pixel_exactly_once(self, dim):
        _, flat = _grid.ring_indices(dim)
        assert np.array_equal(np.sort(flat), np.arange(dim * dim))

    @pytest.mark.parametrize("dim", [9, 33, 101])
    def test_ring_slots_zero_and_one_lock_the_walking_direction(self, dim):
        offsets, flat = _grid.ring_indices(dim)
        half = dim // 2
        for y in range(1, _grid.n_rings(dim)):
            first = flat[offsets[y]]
            second = flat[offsets[y] + 1]
            assert first == half * dim + (half + y), f"ring {y} slot 0"
            assert second == (half + 1) * dim + (half + y), f"ring {y} slot 1"

    @pytest.mark.parametrize("dim", [9, 33, 101])
    def test_ring_azimuths_start_at_zero_and_increase_counter_clockwise(self, dim):
        offsets, flat = _grid.ring_indices(dim)
        v = _grid.normals(dim, "lambert").reshape(dim * dim, 3)
        for y in range(1, _grid.n_rings(dim)):
            ring = flat[offsets[y] : offsets[y + 1]]
            azimuths = _azimuths(v[ring])
            expected = 2 * np.pi * np.arange(8 * y) / (8 * y)
            delta = _wrapped_difference(azimuths, expected)
            assert np.allclose(delta, 0, atol=1e-12), f"ring {y}"


class TestSolidAngles:
    @pytest.mark.parametrize("dim", [9, 33, 101, 201])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_ring_solid_angles_cover_the_hemisphere(self, dim, layout):
        omega = _grid.ring_solid_angles(dim, layout)
        n_ring = _grid.n_rings(dim)
        assert omega.shape == (n_ring,)
        y = np.arange(n_ring)
        n_phi = np.maximum(1, 8 * y)
        half = np.ones(n_ring)
        half[-1] = 0.5
        total = np.sum(omega * n_phi * half) * 2 / _grid.n_grid_points(dim)
        assert abs(total - 1) < 1e-12

    @pytest.mark.parametrize("dim", [11, 21, 51, 101])
    def test_lambert_solid_angles_are_positive(self, dim):
        omega = _grid.lambert_solid_angles(dim)
        assert omega.shape == (dim, dim)
        assert np.all(omega > 0)

    @pytest.mark.parametrize("dim", [11, 21, 51, 101])
    def test_lambert_solid_angles_have_the_eight_square_symmetries(self, dim):
        omega = _grid.lambert_solid_angles(dim)
        assert np.allclose(omega, omega.T, atol=1e-15)
        assert np.allclose(omega, omega[::-1], atol=1e-15)
        assert np.allclose(omega, omega[:, ::-1], atol=1e-15)

    @pytest.mark.parametrize("dim", [11, 21, 51, 101])
    def test_lambert_pole_pixel_solid_angle_converges_to_two_over_pi(self, dim):
        omega = _grid.lambert_solid_angles(dim)
        pole = omega[dim // 2, dim // 2]
        # rel 2e-2, not 1e-2: the convergence is O(1 / (dim - 1) ** 2)
        # and the exact geodesic quad at dim 11 is 0.646212, i.e.
        # 1.5 % above 2 / pi (confirmed against an independent
        # spherical-excess area of the same four corners)
        assert pole == pytest.approx(2 / np.pi, rel=2e-2)

    @pytest.mark.parametrize("dim", [11, 21, 51, 101])
    def test_lambert_non_pole_solid_angles_are_within_six_percent_of_one(self, dim):
        omega = _grid.lambert_solid_angles(dim)
        target = _grid.n_grid_points(dim) / 2 / (dim - 1) ** 2
        rings = _grid.ring_number(dim)
        ratio = omega[rings >= 1] / target
        assert np.all(np.abs(ratio - 1) < 0.06)

    @pytest.mark.parametrize("dim", [11, 21, 51, 101])
    def test_lambert_solid_angles_sum_to_half_the_grid_points(self, dim):
        omega = _grid.lambert_solid_angles(dim)
        weight = np.ones((dim, dim))
        weight[0] *= 0.5
        weight[-1] *= 0.5
        weight[:, 0] *= 0.5
        weight[:, -1] *= 0.5
        total = np.sum(omega * weight)
        assert total == pytest.approx(dim * dim - 2 * dim + 2, rel=1e-6)


class TestQuadratureWeights:
    @pytest.mark.parametrize("dim", [9, 33, 101, 201])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_unscaled_weights_sum_to_one_for_every_skipped_ring(self, dim, layout):
        for skip in range((dim - 2) // 4 + 1):
            w_hat = _unscaled_weights(dim, layout, skip)
            assert abs(np.sum(w_hat) - 1) <= WEIGHT_SUM_TOLERANCE, f"skip = {skip}"

    @pytest.mark.parametrize("dim", [9, 33, 101, 201])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_the_skipped_ring_has_zero_weight(self, dim, layout):
        for skip in range((dim - 2) // 4 + 1):
            w_hat = _unscaled_weights(dim, layout, skip)
            assert w_hat.shape == (_grid.n_rings(dim),)
            assert w_hat[skip] == 0.0, f"skip = {skip}"

    @pytest.mark.parametrize("dim", [9, 33, 101, 201])
    @pytest.mark.parametrize("layout", ["lambert", "legendre"])
    def test_weights_are_scaled_by_four_pi_over_the_ring_point_count(self, dim, layout):
        weights = _grid.quadrature_weights(dim, layout)
        n_weights = (dim - 2) // 4 + 1
        assert weights.shape == (n_weights, _grid.n_rings(dim))
        n_phi = np.maximum(1, 8 * np.arange(_grid.n_rings(dim)))
        for skip in range(n_weights):
            # The Legendre layout solves the skip 0 system once and
            # replicates it (square_sht.hpp lines 376-378), so every
            # row is the skip 0 set there
            w_hat = _unscaled_weights(dim, layout, skip if layout == "lambert" else 0)
            assert np.allclose(
                weights[skip], 4 * np.pi * w_hat / n_phi, rtol=1e-14, atol=0
            ), f"skip = {skip}"

    @pytest.mark.parametrize("dim", [33, 65])
    def test_lambert_weight_sets_solve_the_chebyshev_moment_system(self, dim):
        # The unscaled weights are the unique solution of
        # sum_i w_hat[i] T_j(2 x_i ** 2 - 1) = int_0^1 T_j(2 x ** 2 - 1)
        # dx, which is 1 for j == 0 and -1 / (4 j ** 2 - 1) otherwise,
        # over the rings the set keeps. A wrong right hand side or a
        # transposed system would still sum to one, but would not
        # reproduce the moments
        weights = _grid.quadrature_weights(dim, "lambert")
        cos_lats = _grid.cos_latitudes(dim, "lambert")
        n_rings = _grid.n_rings(dim)
        n_phi = np.maximum(1, 8 * np.arange(n_rings))
        for skip in range(weights.shape[0]):
            w_hat = weights[skip] * n_phi / (4 * np.pi)
            keep = np.arange(n_rings) != skip
            x = cos_lats[keep]
            j = np.arange(x.size)
            a = chebvander(2 * x**2 - 1, x.size - 1).T
            b = np.empty(x.size)
            b[0] = 1
            b[1:] = -1 / (4 * j[1:] ** 2 - 1)
            residual = np.abs(a @ w_hat[keep] - b).max()
            assert residual < 1e-10, f"skip = {skip}"

    @pytest.mark.parametrize("dim", [19, 35, 101, 201, 401])
    def test_legendre_weight_sets_are_all_the_skip_zero_set(self, dim):
        weights = _grid.quadrature_weights(dim, "legendre")
        for skip in range(1, weights.shape[0]):
            assert np.array_equal(weights[skip], weights[0]), f"skip = {skip}"

    @pytest.mark.parametrize("dim", [19, 35, 101, 201, 401])
    def test_legendre_weights_are_gauss_legendre_with_a_halved_equator(self, dim):
        w_hat = _unscaled_weights(dim, "legendre", 0)
        _, w_gauss = leggauss(dim - 2)
        # Non-negative half, descending, so that the zero root is last
        w_desc = w_gauss[(dim - 3) // 2 :][::-1]
        # Absolute, since the smallest Gauss-Legendre weights of the
        # dim 401 set are themselves of order 1e-5: the measured worst
        # differences there are 9.1e-15 and 1.35e-15
        assert np.allclose(w_hat[1:-1], w_desc[:-1], rtol=0, atol=1e-13)
        assert w_hat[-1] == pytest.approx(w_desc[-1] / 2, abs=1e-14)

    def test_lambert_weights_are_computable_at_dim_201(self):
        weights = _grid.quadrature_weights(201, "lambert")
        assert weights.shape == ((201 - 2) // 4 + 1, 101)
        assert np.all(np.isfinite(weights))

    def test_lambert_weights_raise_at_dim_401(self):
        with pytest.raises(ValueError):
            _grid.quadrature_weights(401, "lambert")

    def test_a_negative_weight_residual_also_raises(self, monkeypatch):
        # EMSphInx tests the signed residual only (square_sht.hpp line
        # 1057). A large negative residual is equally unusable, so the
        # port tests the absolute residual, which nothing else pins
        def fake_solve(a, b):
            w_hat = np.zeros(b.shape, dtype=np.float64)
            w_hat[0] = 1 - 10 * WEIGHT_SUM_TOLERANCE
            return w_hat

        monkeypatch.setattr(np.linalg, "solve", fake_solve)
        with pytest.raises(ValueError, match="Insufficient precision"):
            _grid.quadrature_weights(51, "lambert")

    def test_a_negative_residual_of_an_ill_conditioned_grid_raises(self):
        # Natural data confirmation of the test above: dim 361
        # skipping ring 16 has a residual of -7.0e-03, i.e. 7.4e+04
        # times the tolerance, but of the sign a one sided guard lets
        # through
        cos_lats = _grid.cos_latitudes(361, "lambert")
        with pytest.raises(ValueError, match="Insufficient precision"):
            _grid._ring_weights_skip(361, cos_lats, 16)

    def test_smallest_lambert_dim_tripping_the_precision_guard(self, record_property):
        # Recorded determination: the reviewers bracketed the first
        # failure between dim 259 and 301
        first_failure = None
        for dim in range(259, 303, 2):
            try:
                _grid.quadrature_weights(dim, "lambert")
            except ValueError:
                first_failure = dim
                break
        record_property("smallest_tripping_lambert_dim", str(first_failure))
        assert first_failure is not None
        assert 259 <= first_failure <= 301
