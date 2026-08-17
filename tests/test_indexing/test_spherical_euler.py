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

"""Tests of ``kikuchipy.indexing._spherical._euler``.

Covers the "Euler" assertions of
``specs/2026-08-16-sht-wigner-d/validation.md``:

- The ZYZ to Bunge relation ``(alpha + pi/2, beta, gamma - pi/2)``
  against :meth:`orix.quaternion.Rotation.from_euler` on 1000 random
  triples, with the reversed relation of EMSphInx' own ``zyz2eu()``
  as a guard.
- A port of the ZYZ block of EMSphInx' rotation round trip test
  (``EMSphInx/test/xtal/rotations.cpp``, lines 288-318) on the
  ``n = 25`` and ``n = 15`` Euler grids of lines 134-138, against the
  test-local transcription ``_emsphinx_eu2qu()`` of ``eu2qu()``
  (``include/xtal/rotations.hpp``, lines 410-429, including
  ``detail::orientAxis()`` of lines 247-260 and the normalization),
  and against orix as rotations.
- ``quaternion_to_zyz()`` round trips, ranges, the two degenerate
  branches with Python modulo, and near degenerate ``beta``.
- ``zyz_to_quaternion()`` sign conventions, the ``orientAxis`` cases
  of rotations by pi, shapes and the ``ValueError`` paths.
- ``rotation_from_zyz()``/``rotation_to_zyz()`` identities and shapes.
- ``wrap_beta()`` cases, including the preserved negative zero.
- ``KERNEL_NAMES`` covering every Numba kernel of the module.
- ``.py_func`` of every kernel and the Numba compilation flags.
- A recorded ``zyz_to_quaternion`` timing baseline.
"""

import math
import time

import numpy as np
from orix.quaternion import Rotation
import pytest

from kikuchipy.indexing._spherical import _euler

EPS = float(np.finfo(np.float64).eps)
R_EPS = math.sqrt(EPS)
THR = 10 * EPS
TWO_PI = 2 * math.pi

# Every Numba kernel of the module, for the flag test
KERNEL_NAMES = [
    "_wrap_beta",
    "_orient_axis",
    "_zyz_to_quaternion_single",
    "_zyz_to_quaternion_2d",
    "_quaternion_to_zyz_single",
    "_quaternion_to_zyz_2d",
]


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

    Falls back to the function itself while it is still an undecorated
    stub. Every caller first asserts that the kernel does carry a
    ``py_func``, so that an implementation without ``@njit`` fails
    loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


def _orient_axis_reference(ax):
    """Transcription of EMSphInx' ``detail::orientAxis()``.

    ``include/xtal/rotations.hpp`` lines 247-260, in place on a
    length three array.
    """
    if abs(ax[2]) < R_EPS:
        ax[2] = 0.0
        if abs(ax[1]) < R_EPS:
            ax[1] = 0.0
            ax[0] = 1.0
        else:
            mag = math.copysign(math.sqrt(ax[0] * ax[0] + ax[1] * ax[1]), ax[1])
            ax[0] /= mag
            ax[1] /= mag
    elif math.copysign(1.0, ax[2]) < 0:
        ax[0] = -ax[0]
        ax[1] = -ax[1]
        ax[2] = -ax[2]


def _emsphinx_eu2qu(eu):
    """Transcription of EMSphInx' ``eu2qu()`` for one Bunge triple.

    ``include/xtal/rotations.hpp`` lines 410-429 with ``pijk = +1``,
    including the normalization and the ``orientAxis`` branch which
    ``zyz2qu()`` also has. This is the independent oracle of the grid
    test; it is bitwise equal to
    :meth:`orix.quaternion.Rotation.from_euler`, which is asserted
    below.

    Note the differences from ``zyz2qu()``, which are faithful and
    deliberate: ``sigma``/``delta`` are built from ``phi1 +- phi2``
    (not ``gamma +- alpha``), the sign flip tests ``w < 0`` (not
    ``signbit(w)``) and the pi branch tests against ``thr`` (not
    ``rEps``).
    """
    c = math.cos(eu[1] / 2)
    s = math.sin(eu[1] / 2)
    sigma = (eu[0] + eu[2]) / 2
    delta = (eu[0] - eu[2]) / 2
    qu = np.empty(4)
    qu[0] = c * math.cos(sigma)
    qu[1] = -(s * math.cos(delta))
    qu[2] = -(s * math.sin(delta))
    qu[3] = -(c * math.sin(sigma))
    if qu[0] < 0:
        qu = -qu
    qu = qu / math.sqrt(float(np.dot(qu, qu)))
    if abs(qu[0]) <= THR:
        _orient_axis_reference(qu[1:])
        qu[0] = 0.0
    return qu


def _random_zyz(n=1000, seed=0):
    """Return ``n`` random ZYZ triples in ``[0, 2 pi) x [0, pi] x
    [0, 2 pi)``.
    """
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(0, TWO_PI, n),
            rng.uniform(0, math.pi, n),
            rng.uniform(0, TWO_PI, n),
        ],
        axis=1,
    )


def _euler_grid(n):
    """Return the Bunge Euler grid of ``rotations.cpp`` lines 134-138.

    ``phi = pi * i / (n - 1)`` for ``i = 0 ... 2 (n - 1)`` and
    ``theta = phi[:n]``, taken as the outer product, i.e.
    ``(2 n - 1) ** 2 * n`` triples: 60025 at ``n = 25`` and 12615 at
    ``n = 15``.
    """
    phi = math.pi * np.arange(2 * (n - 1) + 1) / (n - 1)
    theta = phi[:n]
    a, b, g = np.meshgrid(phi, theta, phi, indexing="ij")
    return np.stack([a.ravel(), b.ravel(), g.ravel()], axis=1)


def _angle_difference(a, b):
    """Return the difference of two angles wrapped into [-pi, pi]."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


class TestWrapBeta:
    @pytest.mark.parametrize(
        "beta, expected",
        [
            (4.0, 4.0 - TWO_PI),
            (-4.0, TWO_PI - 4.0),
            (7.0, 7.0 - TWO_PI),
            (math.pi, math.pi),
            (-math.pi, -math.pi),
            (0.0, 0.0),
            (1.0, 1.0),
        ],
    )
    def test_wrap_beta_cases(self, beta, expected):
        assert _euler.wrap_beta(beta) == pytest.approx(expected, abs=1e-15)

    def test_wrap_beta_keeps_the_sign_of_negative_zero(self):
        # fmod(-0.0, x) is -0.0, which is harmless because the
        # negative beta swap is a no-op at beta == 0, but it is the
        # documented behaviour
        wrapped = _euler.wrap_beta(-0.0)
        assert wrapped == 0.0
        assert math.copysign(1.0, wrapped) == -1.0
        assert math.copysign(1.0, _euler.wrap_beta(0.0)) == 1.0

    def test_wrap_beta_is_idempotent_and_periodic(self):
        rng = np.random.default_rng(0)
        for beta in rng.uniform(-20, 20, 200):
            wrapped = _euler.wrap_beta(float(beta))
            assert -math.pi <= wrapped <= math.pi
            assert _euler.wrap_beta(wrapped) == pytest.approx(wrapped, abs=0)
            assert _angle_difference(wrapped, beta) < 1e-12


class TestZyzToBunge:
    def test_zyz_to_quaternion_matches_orix_from_euler_of_bunge(self):
        zyz = _random_zyz()
        qu = _euler.zyz_to_quaternion(zyz)
        expected = Rotation.from_euler(_euler.zyz_to_bunge(zyz)).data
        assert np.abs(qu - expected).max() < 1e-14
        assert (qu[:, 0] >= 0).all()
        assert (expected[:, 0] >= 0).all()

    def test_the_reversed_offsets_of_emsphinx_zyz2eu_are_wrong(self):
        # EMSphInx' own zyz2eu()/eu2zyz() (rotations.hpp lines
        # 1025-1039) state these offsets reversed; the reversed
        # relation is a 180 degree rotation about z away
        zyz = _random_zyz()
        qu = _euler.zyz_to_quaternion(zyz)
        reversed_bunge = zyz + np.array([-math.pi / 2, 0.0, math.pi / 2])
        wrong = Rotation.from_euler(reversed_bunge).data
        assert np.abs(qu - wrong).max() > 0.5

    def test_zyz_to_bunge_of_the_origin(self):
        assert _euler.zyz_to_bunge([0.0, 0.0, 0.0]) == pytest.approx(
            [math.pi / 2, 0.0, -math.pi / 2], abs=0
        )

    def test_bunge_to_zyz_inverts_zyz_to_bunge(self):
        # (x + pi/2) - pi/2 is one ulp off x for a large fraction of
        # doubles -- 38138 of 100000 random triples differ, worst
        # 4.44e-16 -- so no "pure affine, no wrapping" implementation
        # can invert exactly on alpha and gamma. beta is untouched by
        # both maps and is asserted bitwise
        zyz = _random_zyz(100)
        back = _euler.bunge_to_zyz(_euler.zyz_to_bunge(zyz))
        assert np.abs(back - zyz).max() <= 1e-15
        assert np.array_equal(back[..., 1], zyz[..., 1])

    def test_zyz_to_bunge_does_not_wrap(self):
        # a pure affine shift, so a large angle stays large
        out = _euler.zyz_to_bunge([10.0, -3.0, 20.0])
        assert out[0] == pytest.approx(10.0 + math.pi / 2, abs=0)
        assert out[1] == pytest.approx(-3.0, abs=0)
        assert out[2] == pytest.approx(20.0 - math.pi / 2, abs=0)

    @pytest.mark.parametrize("shape", [(3,), (5, 3), (2, 4, 3)])
    def test_zyz_to_bunge_keeps_the_shape(self, shape):
        rng = np.random.default_rng(0)
        zyz = rng.uniform(-1, 1, shape)
        assert _euler.zyz_to_bunge(zyz).shape == shape
        assert _euler.bunge_to_zyz(zyz).shape == shape

    @pytest.mark.parametrize("bad", [(2,), (4,), (5, 2)])
    def test_zyz_to_bunge_raises_on_a_wrong_last_dimension(self, bad):
        with pytest.raises(ValueError):
            _euler.zyz_to_bunge(np.zeros(bad))
        with pytest.raises(ValueError):
            _euler.bunge_to_zyz(np.zeros(bad))


class TestEmsphinxRotationsGrid:
    """Port of the ZYZ block of ``EMSphInx/test/xtal/rotations.cpp``
    lines 288-318 on the Euler grids of lines 134-138.
    """

    def test_the_transcribed_eu2qu_equals_orix_from_euler_to_a_few_ulp(self):
        # pins specs/_research/explore-emsphinx-xtal-util-vs-orix.md
        # 1.2, and thereby the whole grid test as an EMSphInx oracle.
        # Bitwise on this machine, but numpy's CPU-dispatched sin/cos
        # (AVX512 vs AVX2 runners) and LLVM on arm64 differ in the last
        # ulp, so the cross-library comparison is pinned to 4 eps
        # (measured 0.0 here; CI failed bitwise on macOS/Windows/Ubuntu)
        bunge = _euler.zyz_to_bunge(_random_zyz())
        mine = np.array([_emsphinx_eu2qu(e) for e in bunge])
        assert np.abs(mine - Rotation.from_euler(bunge).data).max() <= 4 * EPS

    @pytest.mark.parametrize("n, n_triples", [(25, 60025), (15, 12615)])
    def test_zyz_to_quaternion_matches_eu2qu_on_the_grid(self, n, n_triples):
        eu = _euler_grid(n)
        assert eu.shape == (n_triples, 3)
        qu = _euler.zyz_to_quaternion(_euler.bunge_to_zyz(eu))
        expected = np.array([_emsphinx_eu2qu(e) for e in eu])
        assert np.abs(qu - expected).max() <= 10 * EPS

    @pytest.mark.parametrize("n", [25, 15])
    def test_quaternion_to_zyz_round_trips_on_the_grid(self, n):
        # the euDelta comparison of rotations.cpp lines 313-323:
        # compare the re-converted quaternions, not the angles
        eu = _euler_grid(n)
        expected = np.array([_emsphinx_eu2qu(e) for e in eu])
        bunge = _euler.zyz_to_bunge(_euler.quaternion_to_zyz(expected))
        again = np.array([_emsphinx_eu2qu(e) for e in bunge])
        assert np.abs(expected - again).max() <= 10 * EPS

    @pytest.mark.parametrize("n", [25, 15])
    def test_grid_agrees_with_orix_as_rotations(self, n, record_property):
        # q and -q are the same rotation; orientAxis flips the sign of
        # the w == 0 cases where orix keeps its own, so the
        # component-wise difference is 2.0 for those
        eu = _euler_grid(n)
        qu = _euler.zyz_to_quaternion(_euler.bunge_to_zyz(eu))
        orix_data = Rotation.from_euler(eu).data
        plus = np.abs(qu - orix_data).max(axis=-1)
        minus = np.abs(qu + orix_data).max(axis=-1)
        flipped = int((minus < plus).sum())
        record_property(f"euler_grid_n{n}_sign_flips", f"{flipped} of {eu.shape[0]}")
        record_property(
            f"euler_grid_n{n}_max_rotation_delta",
            f"{np.minimum(plus, minus).max():.3e}",
        )
        assert np.minimum(plus, minus).max() <= 10 * EPS


class TestZyzToQuaternion:
    def test_w_is_non_negative_and_signbit_clean(self):
        qu = _euler.zyz_to_quaternion(_random_zyz())
        assert (qu[:, 0] >= 0).all()
        assert not np.signbit(qu[:, 0]).any()

    @pytest.mark.parametrize(
        "zyz, expected",
        [
            ((0.0, math.pi, 0.0), (0.0, 0.0, 1.0, 0.0)),
            ((math.pi / 2, math.pi, math.pi / 2), (0.0, 0.0, 1.0, 0.0)),
            ((0.3, math.pi, 0.3), (0.0, 0.0, 1.0, 0.0)),
            ((0.0, math.pi, math.pi), (0.0, 1.0, 0.0, 0.0)),
            ((math.pi, math.pi, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ],
    )
    def test_pi_rotations_take_the_orient_axis_convention(self, zyz, expected):
        # +z hemisphere, +y half of the equator, and (1, 0, 0) when
        # both y and z vanish (rotations.hpp lines 247-260)
        qu = _euler.zyz_to_quaternion(zyz)
        assert np.array_equal(qu, np.asarray(expected))
        assert not np.signbit(qu).any()

    @pytest.mark.parametrize(
        "zyz, expected",
        [
            ((0.0, math.pi - 2e-9, 0.0), (0.0, 0.0, 1.0, 0.0)),
            ((0.0, math.pi - 2e-9, math.pi), (0.0, 1.0, 0.0, 0.0)),
        ],
    )
    def test_near_pi_rotations_use_the_r_eps_threshold(self, zyz, expected):
        # zyz2qu() and orientAxis() both compare against rEps
        # (1.49e-8) and never thr (2.2e-15), and beta = pi exactly
        # cannot tell the two apart: there c = cos(beta / 2) is
        # 6.1e-17, so w and z fall below both. At beta = pi - 2e-9
        # the first case puts |w| = 1.0e-9 and the second
        # |z| = 1.0e-9 into the band between them, where thr would
        # skip the pi branch of zyz2qu() (giving (1e-9, -0, -1, -0))
        # and the equator branch of orientAxis() respectively
        qu = _euler.zyz_to_quaternion(zyz)
        assert np.array_equal(qu, np.asarray(expected))
        assert not np.signbit(qu).any()

    @pytest.mark.parametrize(
        "shape_in, shape_out", [((3,), (4,)), ((7, 3), (7, 4)), ((2, 5, 3), (2, 5, 4))]
    )
    def test_zyz_to_quaternion_shapes(self, shape_in, shape_out):
        rng = np.random.default_rng(0)
        assert _euler.zyz_to_quaternion(rng.uniform(0, 1, shape_in)).shape == shape_out

    @pytest.mark.parametrize("bad", [(2,), (4,), (6, 2)])
    def test_zyz_to_quaternion_raises_on_a_wrong_last_dimension(self, bad):
        with pytest.raises(ValueError):
            _euler.zyz_to_quaternion(np.zeros(bad))

    def test_zyz_to_quaternion_casts_to_float64(self):
        qu = _euler.zyz_to_quaternion(np.zeros(3, dtype=np.float32))
        assert qu.dtype == np.float64

    @pytest.mark.parametrize("bad", [1.0, np.float64(0.0)])
    def test_a_zero_dimensional_input_raises(self, bad):
        # the zero dimensional guard of the shared shape check, which
        # would otherwise be an IndexError from shape[-1] instead of
        # the documented ValueError
        with pytest.raises(ValueError):
            _euler.zyz_to_quaternion(bad)
        with pytest.raises(ValueError):
            _euler.quaternion_to_zyz(bad)
        with pytest.raises(ValueError):
            _euler.zyz_to_bunge(bad)


class TestQuaternionToZyz:
    def test_round_trip_of_random_triples(self):
        zyz = _random_zyz()
        qu = _euler.zyz_to_quaternion(zyz)
        back = _euler.quaternion_to_zyz(qu)
        assert np.abs(_euler.zyz_to_quaternion(back) - qu).max() < 1e-14
        assert _angle_difference(back, zyz).max() < 1e-13

    def test_ranges(self):
        back = _euler.quaternion_to_zyz(_euler.zyz_to_quaternion(_random_zyz()))
        assert (back[:, 0] >= 0).all() and (back[:, 0] < TWO_PI).all()
        assert (back[:, 1] >= 0).all() and (back[:, 1] <= math.pi).all()
        assert (back[:, 2] >= 0).all() and (back[:, 2] < TWO_PI).all()

    def test_beta_zero_branch_returns_alpha_plus_gamma_modulo_two_pi(self):
        # Python modulo into [0, 2 pi), which is what the "+2 pi"
        # wrap of rotations.hpp line 1019 produces; math.fmod would
        # be negative whenever alpha < gamma
        rng = np.random.default_rng(1)
        pairs = rng.uniform(0, TWO_PI, (2000, 2))
        zyz = np.stack([pairs[:, 0], np.zeros(2000), pairs[:, 1]], axis=1)
        back = _euler.quaternion_to_zyz(_euler.zyz_to_quaternion(zyz))
        expected = (pairs[:, 0] + pairs[:, 1]) % TWO_PI
        assert _angle_difference(back[:, 0], expected).max() < 1e-14
        assert np.array_equal(back[:, 1], np.zeros(2000))
        assert np.array_equal(back[:, 2], np.zeros(2000))

    def test_beta_pi_branch_returns_alpha_minus_gamma_modulo_two_pi(self):
        rng = np.random.default_rng(2)
        pairs = rng.uniform(0, TWO_PI, (2000, 2))
        zyz = np.stack([pairs[:, 0], np.full(2000, math.pi), pairs[:, 1]], axis=1)
        back = _euler.quaternion_to_zyz(_euler.zyz_to_quaternion(zyz))
        expected = (pairs[:, 0] - pairs[:, 1]) % TWO_PI
        assert _angle_difference(back[:, 0], expected).max() < 1e-14
        assert np.array_equal(back[:, 1], np.full(2000, math.pi))
        assert np.array_equal(back[:, 2], np.zeros(2000))

    def test_the_degenerate_branch_uses_python_modulo_not_fmod(self):
        # alpha = 1, gamma = 2 gives 5.283185307179586, not -1.0
        back = _euler.quaternion_to_zyz(_euler.zyz_to_quaternion([1.0, math.pi, 2.0]))
        assert back[0] == pytest.approx((1.0 - 2.0) % TWO_PI, abs=1e-14)
        assert back[0] > 0

    @pytest.mark.parametrize(
        "beta", [0.0, 1e-9, 1e-7, math.pi - 1e-7, math.pi - 1e-9, math.pi]
    )
    def test_round_trip_near_the_degenerate_betas(self, beta):
        zyz = np.array([0.7, beta, -1.2])
        qu = _euler.zyz_to_quaternion(zyz)
        back = _euler.quaternion_to_zyz(qu)
        assert np.abs(_euler.zyz_to_quaternion(back) - qu).max() < 1e-14

    def test_a_quaternion_and_its_negative_give_the_same_angles(self):
        qu = _euler.zyz_to_quaternion(_random_zyz(100))
        assert np.array_equal(
            _euler.quaternion_to_zyz(qu), _euler.quaternion_to_zyz(-qu)
        )

    @pytest.mark.parametrize(
        "shape_in, shape_out", [((4,), (3,)), ((7, 4), (7, 3)), ((2, 5, 4), (2, 5, 3))]
    )
    def test_quaternion_to_zyz_shapes(self, shape_in, shape_out):
        qu = np.zeros(shape_in)
        qu[..., 0] = 1.0
        assert _euler.quaternion_to_zyz(qu).shape == shape_out

    @pytest.mark.parametrize("bad", [(5,), (3,), (6, 5)])
    def test_quaternion_to_zyz_raises_on_a_wrong_last_dimension(self, bad):
        with pytest.raises(ValueError):
            _euler.quaternion_to_zyz(np.zeros(bad))


class TestRotationFromZyz:
    def test_is_the_conjugate_of_the_quaternion_exactly(self):
        zyz = _random_zyz(200)
        rotation = _euler.rotation_from_zyz(zyz)
        expected = (~Rotation(_euler.zyz_to_quaternion(zyz))).data
        assert np.array_equal(rotation.data, expected)

    def test_matches_the_conjugate_of_orix_from_euler_of_bunge(self):
        zyz = _random_zyz(200)
        rotation = _euler.rotation_from_zyz(zyz)
        expected = (~Rotation.from_euler(_euler.zyz_to_bunge(zyz))).data
        assert np.abs(rotation.data - expected).max() < 1e-14

    def test_rotation_to_zyz_round_trips(self):
        zyz = _random_zyz(200)
        back = _euler.rotation_to_zyz(_euler.rotation_from_zyz(zyz))
        assert _angle_difference(back, zyz).max() < 1e-13

    @pytest.mark.parametrize(
        "shape_in, shape_out", [((3,), (1,)), ((7, 3), (7,)), ((2, 5, 3), (2, 5))]
    )
    def test_rotation_from_zyz_shapes(self, shape_in, shape_out):
        rng = np.random.default_rng(0)
        rotation = _euler.rotation_from_zyz(rng.uniform(0, 1, shape_in))
        assert isinstance(rotation, Rotation)
        assert rotation.shape == shape_out
        assert _euler.rotation_to_zyz(rotation).shape == shape_out + (3,)

    def test_rotation_from_zyz_raises_on_a_wrong_last_dimension(self):
        with pytest.raises(ValueError):
            _euler.rotation_from_zyz(np.zeros(2))


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # the flag test and the py_func tests are parametrised over
        # the literal list above, so a kernel added during the
        # implementation would silently escape both of them
        assert _njit_kernel_names(_euler) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _euler"
        )

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        # Dropping either option leaves every other test passing, so
        # the private Numba attributes are read directly
        kernel = getattr(_euler, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    def test_wrap_beta_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_euler._wrap_beta, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(0)
        for beta in np.concatenate([rng.uniform(-20, 20, 100), [0.0, -0.0, math.pi]]):
            compiled = _euler._wrap_beta(float(beta))
            interpreted = _py_func(_euler._wrap_beta)(float(beta))
            assert compiled == interpreted
            assert math.copysign(1.0, compiled) == math.copysign(1.0, interpreted)

    def test_orient_axis_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_euler._orient_axis, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(0)
        axes = rng.uniform(-1, 1, (100, 3))
        axes = np.concatenate(
            [axes, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.4, 0.0]])]
        )
        for axis in axes:
            compiled = axis.copy()
            interpreted = axis.copy()
            _euler._orient_axis(compiled)
            _py_func(_euler._orient_axis)(interpreted)
            assert np.array_equal(compiled, interpreted)

    def test_zyz_to_quaternion_kernels_py_func_equal_the_compiled_kernels(self):
        assert hasattr(_euler._zyz_to_quaternion_single, "py_func"), (
            "kernel must be @njit-decorated"
        )
        assert hasattr(_euler._zyz_to_quaternion_2d, "py_func"), (
            "kernel must be @njit-decorated"
        )
        zyz = _random_zyz(100)
        compiled = _euler._zyz_to_quaternion_2d(zyz)
        interpreted = _py_func(_euler._zyz_to_quaternion_2d)(zyz)
        assert np.array_equal(compiled, interpreted)
        single_compiled = np.empty(4)
        single_interpreted = np.empty(4)
        _euler._zyz_to_quaternion_single(zyz[0], single_compiled)
        _py_func(_euler._zyz_to_quaternion_single)(zyz[0], single_interpreted)
        assert np.array_equal(single_compiled, single_interpreted)
        assert np.array_equal(single_compiled, compiled[0])

    def test_quaternion_to_zyz_kernels_py_func_equal_the_compiled_kernels(self):
        assert hasattr(_euler._quaternion_to_zyz_single, "py_func"), (
            "kernel must be @njit-decorated"
        )
        assert hasattr(_euler._quaternion_to_zyz_2d, "py_func"), (
            "kernel must be @njit-decorated"
        )
        qu = _euler.zyz_to_quaternion(_random_zyz(100))
        compiled = _euler._quaternion_to_zyz_2d(qu)
        interpreted = _py_func(_euler._quaternion_to_zyz_2d)(qu)
        assert np.array_equal(compiled, interpreted)
        single_compiled = np.empty(3)
        single_interpreted = np.empty(3)
        _euler._quaternion_to_zyz_single(qu[0], single_compiled)
        _py_func(_euler._quaternion_to_zyz_single)(qu[0], single_interpreted)
        assert np.array_equal(single_compiled, single_interpreted)
        assert np.array_equal(single_compiled, compiled[0])

    @pytest.mark.parametrize(
        "zyz, expected",
        [
            ((0.0, math.pi, 0.0), (0.0, 0.0, 1.0, 0.0)),
            ((0.0, math.pi, math.pi), (0.0, 1.0, 0.0, 0.0)),
        ],
    )
    def test_zyz_to_quaternion_py_func_takes_the_pi_branch(self, zyz, expected):
        # the random triples above never come within rEps of a pi
        # rotation, so the |w| <= rEps branch of zyz2qu() and the
        # orientAxis() call it guards are otherwise never run in the
        # interpreted path
        kernel = _euler._zyz_to_quaternion_single
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        compiled = np.empty(4)
        interpreted = np.empty(4)
        kernel(np.asarray(zyz), compiled)
        _py_func(kernel)(np.asarray(zyz), interpreted)
        assert np.array_equal(compiled, interpreted)
        assert np.array_equal(interpreted, np.asarray(expected))
        assert not np.signbit(interpreted).any()

    @pytest.mark.parametrize(
        "qu, beta",
        [((1.0, 0.0, 0.0, 0.0), 0.0), ((0.0, 1.0, 0.0, 0.0), math.pi)],
    )
    def test_quaternion_to_zyz_py_func_takes_the_degenerate_branches(self, qu, beta):
        # both arms of the chi <= thr block of qu2zyz(), which the
        # random quaternions above never reach
        kernel = _euler._quaternion_to_zyz_single
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        compiled = np.empty(3)
        interpreted = np.empty(3)
        kernel(np.asarray(qu), compiled)
        _py_func(kernel)(np.asarray(qu), interpreted)
        assert np.array_equal(compiled, interpreted)
        assert interpreted[1] == beta
        assert interpreted[2] == 0.0


class TestConstants:
    def test_module_constants_match_emsphinx(self):
        # xtal/constants.hpp lines 95-96 and pi2 at line 82
        assert _euler._R_EPS == math.sqrt(EPS)
        assert _euler._THR == 10 * EPS
        assert _euler._TWO_PI == TWO_PI


class TestTimingBaseline:
    def test_zyz_to_quaternion_timing_baseline_is_recorded(self, record_property):
        zyz = _random_zyz(100_000, seed=3)
        _euler.zyz_to_quaternion(zyz[:10])  # warm the Numba cache
        start = time.perf_counter()
        _euler.zyz_to_quaternion(zyz)
        elapsed = time.perf_counter() - start
        record_property("zyz_to_quaternion_seconds_1e5", f"{elapsed:.4f}")
        assert elapsed < 5.0
