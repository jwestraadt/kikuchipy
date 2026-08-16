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

"""Tests of ``kikuchipy.indexing._spherical._sht``.

Covers the "SHT", real data, timing and "Weekly" assertions of
``specs/2026-08-16-sht-square-grid-transform/validation.md``:

- Analyze oracle: the complex ``Y_l^m`` sampled on the grid normals
  gives ``alm[m, l] == 1`` (1e-10) and every other entry below 1e-10
  (Lambert ``bw`` 32, ``dim`` 65) or below 1e-8 (Legendre ``bw`` 68,
  ``dim`` 71 with ``m % 4 != 0``, where the ``m_lim`` truncation of
  ring 1 leaks into the high degrees of ``m = 5``); the Legendre
  ``m % 4 == 0`` errors are recorded and asserted below 1e-10.
- Synthesize oracle: a single unit ``alm`` entry synthesizes to
  ``2 Re Y_l^m`` (``m > 0``) or ``Y_l^0`` on both hemispheres, to
  1e-11 (Lambert) and 1e-10 (Legendre, ``m % 4 != 0``), on the rings
  which carry the order, i.e. ``4 * y >= m``. The single ring
  ``y == m // 4`` of orders ``m % 4 == 0`` is excluded and locked
  separately, see ``TestSynthesizeOracle``.
- Condon-Shortley confirmation with signed values on ``Re Y_l^m``.
- Round trip at the EMSphInx tolerances for the Legendre default
  bandwidth set and a scale-free, ``dim`` dependent bound for the
  Lambert set (1e-11 up to ``dim`` 129, pinned measured values above
  it), plus the full ``@pytest.mark.weekly`` sweeps of
  ``EMSphInx/test/sht/square_sht.cpp``.
- ``analyze(ones, ones)`` gives ``alm[0, 0] == sqrt(4 * pi)``.
- The Numba ring DFT path and the ``scipy.fft`` path agree to 1e-12,
  and ``.py_func`` of every kernel equals the compiled result.
- ``analyze(..., bandwidth=k)`` is bitwise equal to
  ``analyze(...)[:k, :k]``.
- Real data: the ``m % 4 != 0`` power of the Ni master pattern is
  below 1e-25 of the total, the antisymmetric branch is exactly zero
  for a symmetric sphere and alive for an asymmetric one, and the DC
  and Parseval determinations are recorded.
- A recorded ``analyze`` timing baseline, the ``bw`` 384 case
  weekly.
"""

import time

import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.indexing._spherical import _grid, _sht

# (l, m) pairs of the single harmonic oracles
HARMONICS = [(1, 0), (2, 1), (3, 2), (3, 3), (6, 4), (9, 5), (12, 8), (15, 12)]

# Bandwidths of the default round trip suites
LEGENDRE_BANDWIDTHS = [
    4,
    5,
    8,
    15,
    16,
    31,
    32,
    53,
    63,
    64,
    68,
    88,
    113,
    127,
    128,
    129,
    158,
]
LAMBERT_BANDWIDTHS = [4, 8, 15, 16, 32, 63, 64, 100, 128]

# EMSphInx' round trip tolerances (test/sht/square_sht.cpp lines
# 152-153)
CPP_MAX_ERROR = 0.005
CPP_MEAN_ERROR = 0.00005

SQRT_FOUR_PI = np.sqrt(4 * np.pi)


def _py_func(kernel):
    """Return the pure Python function of a Numba kernel.

    Falls back to the function itself while it is still an undecorated
    stub. Every caller first asserts that the kernel does carry a
    ``py_func``, so that an implementation without ``@njit`` fails
    loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


def _sph_harm_y():
    """Return ``scipy.special.sph_harm_y``, skipping if SciPy is too
    old.

    The gate is on the top level package, since ``scipy.special`` has
    no ``__version__``.
    """
    pytest.importorskip("scipy", minversion="1.15")
    from scipy.special import sph_harm_y

    return sph_harm_y


def _sample(sph_harm_y, v: np.ndarray, degree: int, order: int) -> np.ndarray:
    """Return the complex ``Y_l^m`` sampled at unit vectors ``v``.

    ``sph_harm_y`` is a ``MultiUFunc`` and rejects keyword arguments
    ("_() got an unexpected keyword argument 'n'" on SciPy 1.17), so
    the arguments are positional ``(n, m, theta, phi)``. The argument
    order is pinned by
    ``TestScipyOracle.test_sph_harm_y_carries_the_condon_shortley_phase``.
    """
    theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    phi = np.arctan2(v[..., 1], v[..., 0])
    return sph_harm_y(degree, order, theta, phi)


def _hemispheres(sph_harm_y, dim: int, layout: str, degree: int, order: int):
    """Return the complex ``Y_l^m`` on the northern and southern
    hemisphere of a square grid.
    """
    north = _grid.normals(dim, layout)
    south = north.copy()
    south[..., 2] = -south[..., 2]
    return (
        _sample(sph_harm_y, north, degree, order),
        _sample(sph_harm_y, south, degree, order),
    )


def _analyze_complex(sht, north: np.ndarray, south: np.ndarray) -> np.ndarray:
    """Return the coefficients of a complex function.

    The transform is complex linear in the function, so a complex
    function is analyzed as two real transforms.
    """
    return sht.analyze(
        np.ascontiguousarray(north.real), np.ascontiguousarray(south.real)
    ) + 1j * sht.analyze(
        np.ascontiguousarray(north.imag), np.ascontiguousarray(south.imag)
    )


def _random_alm(bandwidth: int) -> np.ndarray:
    """Return a random band-limited spectrum of a real function.

    Entries with ``l < m`` are zero and the ``m == 0`` entries are
    real, as in ``EMSphInx/test/sht/square_sht.cpp`` lines 98-107. The
    C++ Mersenne twister draw is not portable, so parity of the draw
    is not attempted.
    """
    rng = np.random.default_rng(0)
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for order in range(bandwidth):
        for degree in range(order, bandwidth):
            if order == 0:
                alm[order, degree] = rng.uniform(-1, 1)
            else:
                alm[order, degree] = rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)
    return alm


def _random_hemispheres(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two random hemispheres of a square grid."""
    rng = np.random.default_rng(0)
    return (
        rng.uniform(-1, 1, size=(dim, dim)),
        rng.uniform(-1, 1, size=(dim, dim)),
    )


def _round_trip_errors(bandwidth: int, layout: str):
    """Return the maximum and mean round trip error of a random
    spectrum, and the maximum absolute input coefficient.
    """
    alm_in = _random_alm(bandwidth)
    sht = _sht.SphericalHarmonicTransform(bandwidth, layout)
    north, south = sht.synthesize(alm_in)
    alm_out = sht.analyze(north, south)
    delta = alm_out - alm_in
    error = np.maximum(np.abs(delta.real), np.abs(delta.imag))
    return error.max(), error.mean(), np.abs(alm_in).max()


def _lambert_round_trip_bound(dim: int) -> float:
    """Return the scale-free Lambert round trip bound of a grid.

    The Lambert ring weight sets are solved from an increasingly ill
    conditioned moment system, so the round trip loses digits as
    ``dim`` grows. The reviewers measured
    ``max |delta| / max |alm_in|`` as 3.9e-13 (``bw`` 16), 7.9e-13
    (``bw`` 64), 3.6e-9 (``bw`` 100, ``dim`` 201) and 2.9e-6
    (``bw`` 128, ``dim`` 257); the bounds above ``dim`` 129 are those
    measurements pinned with roughly one order of magnitude of
    margin.
    """
    if dim <= 129:
        return 1e-11
    if dim <= 201:
        return 1e-8
    return 1e-5


class TestScipyOracle:
    def test_sph_harm_y_carries_the_condon_shortley_phase(self):
        sph_harm_y = _sph_harm_y()
        # Positional (n, m, theta, phi): a swapped n/m or theta/phi
        # would change the value
        value = sph_harm_y(1, 1, np.pi / 2, 0.0)
        assert value.real == pytest.approx(-0.5 * np.sqrt(3 / (2 * np.pi)), rel=1e-14)
        assert abs(value.imag) < 1e-15
        # |m| > n vanishes, so a swapped (n, m) is caught here
        assert sph_harm_y(1, 2, 0.7, 0.3) == 0
        assert sph_harm_y(2, 1, 0.7, 0.3) != 0


class TestConstruction:
    @pytest.mark.parametrize(
        "bandwidth, layout, dim",
        [
            (8, "legendre", 11),
            (68, "legendre", 71),
            (63, "legendre", 65),
            (32, "lambert", 65),
            (100, "lambert", 201),
        ],
    )
    def test_default_dim_and_attributes_match_the_grid_helpers(
        self, bandwidth, layout, dim
    ):
        sht = _sht.SphericalHarmonicTransform(bandwidth, layout)
        assert sht.dim == dim
        assert sht.bandwidth == bandwidth
        assert sht.layout == layout
        assert sht.n_rings == _grid.n_rings(dim)
        assert np.array_equal(sht.cos_latitudes, _grid.cos_latitudes(dim, layout))
        assert np.array_equal(
            sht.quadrature_weights, _grid.quadrature_weights(dim, layout)
        )
        offsets, flat = _grid.ring_indices(dim)
        assert np.array_equal(sht.ring_offsets, offsets)
        assert np.array_equal(sht.ring_indices, flat)

    def test_repr_states_layout_bandwidth_and_side_length(self):
        sht = _sht.SphericalHarmonicTransform(68, "legendre")
        text = repr(sht)
        assert "SphericalHarmonicTransform" in text
        assert "legendre" in text
        assert "68" in text
        assert "71" in text

    @pytest.mark.parametrize("dim", [10, 2])
    def test_even_or_too_small_dim_raises_value_error(self, dim):
        with pytest.raises(ValueError):
            _sht.SphericalHarmonicTransform(4, "legendre", dim)

    def test_bandwidth_above_the_grid_limit_raises_value_error(self):
        with pytest.raises(ValueError):
            _sht.SphericalHarmonicTransform(32, "lambert", 9)

    def test_unknown_layout_raises_value_error(self):
        with pytest.raises(ValueError):
            _sht.SphericalHarmonicTransform(8, "mollweide")

    def test_analyze_raises_on_a_wrong_hemisphere_shape(self):
        sht = _sht.SphericalHarmonicTransform(8, "legendre")
        north = np.ones((sht.dim, sht.dim))
        with pytest.raises(ValueError):
            sht.analyze(north, np.ones((sht.dim + 2, sht.dim + 2)))

    def test_analyze_raises_above_the_construction_bandwidth(self):
        sht = _sht.SphericalHarmonicTransform(8, "legendre")
        north = np.ones((sht.dim, sht.dim))
        with pytest.raises(ValueError):
            sht.analyze(north, north, bandwidth=9)


class TestAnalyzeOracle:
    @pytest.mark.parametrize("degree, order", HARMONICS)
    def test_lambert_analyze_returns_one_for_a_single_harmonic(self, degree, order):
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 32, 65
        north, south = _hemispheres(sph_harm_y, dim, "lambert", degree, order)
        sht = _sht.SphericalHarmonicTransform(bandwidth, "lambert", dim)
        alm = _analyze_complex(sht, north, south)
        assert alm[order, degree] == pytest.approx(1, abs=1e-10)
        others = alm.copy()
        others[order, degree] = 0
        assert np.abs(others).max() < 1e-10

    @pytest.mark.parametrize("degree, order", [p for p in HARMONICS if p[1] % 4 != 0])
    def test_legendre_analyze_returns_one_for_a_single_harmonic(self, degree, order):
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 68, 71
        north, south = _hemispheres(sph_harm_y, dim, "legendre", degree, order)
        sht = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
        alm = _analyze_complex(sht, north, south)
        assert alm[order, degree] == pytest.approx(1, abs=1e-10)
        others = alm.copy()
        others[order, degree] = 0
        # Recorded determination: ring 1 carries orders m < 5 only, so
        # the (l, m) = (9, 5) probe leaks 8.6e-10 into (m, l) = (5, 67)
        assert np.abs(others).max() < 1e-8

    @pytest.mark.parametrize("degree, order", [p for p in HARMONICS if p[1] % 4 == 0])
    def test_legendre_analyze_nyquist_orders_stay_below_1e_10(
        self, degree, order, record_property
    ):
        # Recorded determination: the Legendre layout uses the skip 0
        # weight set for every order, so ring y = m / 4 (whose real
        # transform has a structurally real bin m) is not excluded.
        # The defect is quadratic in analyze, so at bw 68, dim 71 the
        # diagonal is still exact to 6e-16 and the worst other entry
        # is 2.8e-12
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 68, 71
        north, south = _hemispheres(sph_harm_y, dim, "legendre", degree, order)
        sht = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
        alm = _analyze_complex(sht, north, south)
        others = alm.copy()
        others[order, degree] = 0
        record_property(
            f"legendre_nyquist_l{degree}_m{order}",
            f"|alm[m, l] - 1| = {abs(alm[order, degree] - 1):.3e}, "
            f"max other = {np.abs(others).max():.3e}",
        )
        assert alm[order, degree] == pytest.approx(1, abs=1e-10)
        assert np.abs(others).max() < 1e-10


class TestSynthesizeOracle:
    @staticmethod
    def _expected(sph_harm_y, dim, layout, degree, order):
        north, south = _hemispheres(sph_harm_y, dim, layout, degree, order)
        if order == 0:
            return north.real, south.real
        return 2 * north.real, 2 * south.real

    @staticmethod
    def _mask_compared_pixels(dim, m):
        """Return a mask of the pixels to compare.

        ``synthesize()`` evaluates order ``m`` on ring ``y`` only when
        ``m < m_lim(y) = min(bandwidth, 4 * y + 1)``, so the inner
        rings with ``4 * y < m`` hold no contribution at all and are
        excluded.

        EMSphInx' ``synthesize()`` further writes the one sided
        coefficient into the structurally real bin ``m`` of ring
        ``y = m / 4``, where the true coefficient is twice that. That
        ring is excluded here and locked by
        ``test_synthesize_halves_the_nyquist_ring``.
        """
        rings = _grid.ring_number(dim)
        mask = 4 * rings >= m
        if m > 0 and m % 4 == 0:
            mask &= rings != m // 4
        return mask

    @pytest.mark.parametrize("degree, order", HARMONICS)
    def test_lambert_synthesize_returns_a_single_harmonic(self, degree, order):
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 32, 65
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        alm[order, degree] = 1
        sht = _sht.SphericalHarmonicTransform(bandwidth, "lambert", dim)
        north, south = sht.synthesize(alm)
        north_ref, south_ref = self._expected(sph_harm_y, dim, "lambert", degree, order)
        mask = self._mask_compared_pixels(dim, order)
        assert np.abs(north - north_ref)[mask].max() < 1e-11
        assert np.abs(south - south_ref)[mask].max() < 1e-11

    @pytest.mark.parametrize("degree, order", [p for p in HARMONICS if p[1] % 4 != 0])
    def test_legendre_synthesize_returns_a_single_harmonic(self, degree, order):
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 68, 71
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        alm[order, degree] = 1
        sht = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
        north, south = sht.synthesize(alm)
        north_ref, south_ref = self._expected(
            sph_harm_y, dim, "legendre", degree, order
        )
        mask = self._mask_compared_pixels(dim, order)
        assert np.abs(north - north_ref)[mask].max() < 1e-10
        assert np.abs(south - south_ref)[mask].max() < 1e-10

    @pytest.mark.parametrize("degree, order", [(6, 4), (12, 8)])
    def test_synthesize_halves_the_nyquist_ring(self, degree, order):
        # Locks the EMSphInx behaviour excluded above: on ring
        # y = m / 4 the synthesized values are exactly half the true
        # ones, because bin m of that ring is its Nyquist bin. The
        # forward transform never sees it, since the weight set of
        # order m skips exactly this ring.
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 32, 65
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        alm[order, degree] = 1
        sht = _sht.SphericalHarmonicTransform(bandwidth, "lambert", dim)
        north, _ = sht.synthesize(alm)
        north_ref, _ = self._expected(sph_harm_y, dim, "lambert", degree, order)
        ring = _grid.ring_number(dim) == order // 4
        assert np.abs(north[ring] - north_ref[ring] / 2).max() < 1e-11


class TestCondonShortley:
    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
    def test_analyze_of_the_real_harmonic_has_a_positive_signed_value(self, degree):
        # A dropped Condon-Shortley phase gives -0.5 for odd m
        sph_harm_y = _sph_harm_y()
        bandwidth, dim = 8, 51
        sht = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
        for order in range(degree + 1):
            north, south = _hemispheres(sph_harm_y, dim, "legendre", degree, order)
            alm = sht.analyze(
                np.ascontiguousarray(north.real), np.ascontiguousarray(south.real)
            )
            expected = 1.0 if order == 0 else 0.5
            assert alm[order, degree].real == pytest.approx(expected, abs=1e-8), (
                f"(degree, order) = ({degree}, {order})"
            )
            assert abs(alm[order, degree].imag) < 1e-10, (
                f"(degree, order) = ({degree}, {order})"
            )


class TestRoundTrip:
    @pytest.mark.parametrize("bandwidth", LEGENDRE_BANDWIDTHS)
    def test_legendre_round_trip_within_the_emsphinx_tolerances(self, bandwidth):
        max_error, mean_error, _ = _round_trip_errors(bandwidth, "legendre")
        assert max_error < CPP_MAX_ERROR
        assert mean_error < CPP_MEAN_ERROR

    @pytest.mark.parametrize("bandwidth", LAMBERT_BANDWIDTHS)
    def test_lambert_round_trip_is_scale_free_accurate(self, bandwidth):
        max_error, _, scale = _round_trip_errors(bandwidth, "lambert")
        bound = _lambert_round_trip_bound(_grid.default_dim(bandwidth, "lambert"))
        assert max_error / scale < bound

    @pytest.mark.weekly
    @pytest.mark.parametrize("bandwidth", list(range(4, 385)))
    def test_legendre_round_trip_sweep(self, bandwidth):
        max_error, mean_error, _ = _round_trip_errors(bandwidth, "legendre")
        assert max_error < CPP_MAX_ERROR
        assert mean_error < CPP_MEAN_ERROR

    @pytest.mark.weekly
    @pytest.mark.parametrize("bandwidth", list(range(4, 129)))
    def test_lambert_round_trip_sweep(self, bandwidth):
        max_error, _, scale = _round_trip_errors(bandwidth, "lambert")
        bound = _lambert_round_trip_bound(_grid.default_dim(bandwidth, "lambert"))
        assert max_error / scale < bound


class TestConstantFunction:
    @pytest.mark.parametrize("bandwidth", [8, 68, 128])
    def test_legendre_analyze_of_one_gives_sqrt_four_pi(self, bandwidth):
        sht = _sht.SphericalHarmonicTransform(bandwidth, "legendre")
        ones = np.ones((sht.dim, sht.dim))
        alm = sht.analyze(ones, ones)
        assert alm[0, 0].real == pytest.approx(SQRT_FOUR_PI, abs=1e-12)
        assert abs(alm[0, 0].imag) < 1e-12
        others = alm.copy()
        others[0, 0] = 0
        assert np.abs(others).max() < 1e-10

    # Recorded determination: the dim 201 weight sets come from a more
    # ill conditioned moment system, giving a worst other entry of
    # 6.0e-11 against below 1e-10 at dim 65
    @pytest.mark.parametrize(
        "bandwidth, dim, others_tolerance", [(32, 65, 1e-10), (100, 201, 1e-9)]
    )
    def test_lambert_analyze_of_one_gives_sqrt_four_pi(
        self, bandwidth, dim, others_tolerance
    ):
        sht = _sht.SphericalHarmonicTransform(bandwidth, "lambert", dim)
        ones = np.ones((dim, dim))
        alm = sht.analyze(ones, ones)
        assert alm[0, 0].real == pytest.approx(SQRT_FOUR_PI, abs=1e-10)
        assert abs(alm[0, 0].imag) < 1e-10
        others = alm.copy()
        others[0, 0] = 0
        assert np.abs(others).max() < others_tolerance


class TestDualPath:
    @staticmethod
    def _both_paths(monkeypatch, bandwidth, layout):
        """Return two transformers of the same grid, one forced onto
        the Numba ring transform path and one onto the
        :mod:`scipy.fft` path.

        The class attribute is patched *before* each construction, so
        that the constructor is the one which decides, and the two
        instances are asserted to report different paths.
        """
        cls = _sht.SphericalHarmonicTransform
        dim = _grid.default_dim(bandwidth, layout)
        monkeypatch.setattr(cls, "numba_ring_dft_max_dim", dim)
        sht_numba = cls(bandwidth, layout)
        monkeypatch.setattr(cls, "numba_ring_dft_max_dim", 0)
        sht_scipy = cls(bandwidth, layout)
        assert sht_numba.uses_numba_ring_dft is True
        assert sht_scipy.uses_numba_ring_dft is False
        return sht_numba, sht_scipy

    @pytest.mark.parametrize(
        "bandwidth, layout",
        [(16, "legendre"), (68, "legendre"), (128, "legendre"), (32, "lambert")],
    )
    def test_numba_and_scipy_analyze_paths_agree(self, bandwidth, layout, monkeypatch):
        sht_numba, sht_scipy = self._both_paths(monkeypatch, bandwidth, layout)
        north, south = _random_hemispheres(sht_numba.dim)
        alm_numba = sht_numba.analyze(north, south)
        alm_scipy = sht_scipy.analyze(north, south)
        scale = np.abs(alm_scipy).max()
        assert np.abs(alm_numba - alm_scipy).max() <= 1e-12 * scale

    @pytest.mark.parametrize(
        "bandwidth, layout",
        [(16, "legendre"), (68, "legendre"), (128, "legendre"), (32, "lambert")],
    )
    def test_numba_and_scipy_synthesize_paths_agree(
        self, bandwidth, layout, monkeypatch
    ):
        sht_numba, sht_scipy = self._both_paths(monkeypatch, bandwidth, layout)
        alm = _random_alm(bandwidth)
        north_numba, south_numba = sht_numba.synthesize(alm)
        north_scipy, south_scipy = sht_scipy.synthesize(alm)
        scale = np.abs(north_scipy).max()
        assert np.abs(north_numba - north_scipy).max() <= 1e-12 * scale
        assert np.abs(south_numba - south_scipy).max() <= 1e-12 * scale

    def test_default_numba_ring_dft_max_dim_is_131(self):
        assert _sht.SphericalHarmonicTransform.numba_ring_dft_max_dim == 131


class TestBandwidthArgument:
    @pytest.mark.parametrize("bandwidth", [4, 16, 33])
    def test_analyze_with_a_smaller_bandwidth_is_a_bitwise_slice(self, bandwidth):
        sht = _sht.SphericalHarmonicTransform(68, "legendre")
        north, south = _random_hemispheres(sht.dim)
        full = sht.analyze(north, south)
        part = sht.analyze(north, south, bandwidth=bandwidth)
        assert part.shape == (bandwidth, bandwidth)
        assert np.array_equal(part, full[:bandwidth, :bandwidth])


class TestKernels:
    @staticmethod
    def _transform_arrays(sht):
        amn, bmn = _sht._alf_recursion_tables(sht.bandwidth)
        offsets, cos_table, sin_table = _sht._ring_dft_tables(sht.dim, sht.bandwidth)
        return amn, bmn, offsets, cos_table, sin_table

    def test_alf_recursion_tables_have_the_bandwidth_shape(self):
        amn, bmn = _sht._alf_recursion_tables(16)
        assert amn.shape == (16, 16)
        assert bmn.shape == (16, 16)
        assert amn[0, 0] == pytest.approx(np.sqrt(1 / (4 * np.pi)), rel=1e-15)
        assert np.all(np.tril(bmn, 1) == 0)

    def test_ring_dft_tables_are_ragged_and_ring_sized(self):
        dim, bandwidth = 11, 8
        offsets, cos_table, sin_table = _sht._ring_dft_tables(dim, bandwidth)
        assert offsets.shape == (_grid.n_rings(dim) + 1,)
        assert offsets[0] == 0
        assert cos_table.shape == sin_table.shape == (offsets[-1],)
        for y in range(_grid.n_rings(dim)):
            n_phi = max(1, 8 * y)
            m_lim = min(bandwidth, 4 * y + 1)
            assert offsets[y + 1] - offsets[y] == m_lim * n_phi, f"ring {y}"

    def test_analyze_ring_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_sht._analyze_ring_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth, m_lim = 8, 5
        amn, bmn = _sht._alf_recursion_tables(bandwidth)
        rng = np.random.default_rng(0)
        g_sym = rng.normal(size=m_lim) + 1j * rng.normal(size=m_lim)
        g_asym = rng.normal(size=m_lim) + 1j * rng.normal(size=m_lim)
        alm_compiled = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        alm_interpreted = np.zeros_like(alm_compiled)
        args = (0.3, amn, bmn, bandwidth, m_lim)
        _sht._analyze_ring_kernel(alm_compiled, g_sym, g_asym, *args)
        _py_func(_sht._analyze_ring_kernel)(alm_interpreted, g_sym, g_asym, *args)
        assert np.allclose(alm_compiled, alm_interpreted, atol=1e-15)

    def test_synthesize_ring_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_sht._synthesize_ring_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth, m_lim = 8, 5
        amn, bmn = _sht._alf_recursion_tables(bandwidth)
        alm = _random_alm(bandwidth)
        args = (alm, 0.3, amn, bmn, bandwidth, m_lim)
        compiled = _sht._synthesize_ring_kernel(*args)
        interpreted = _py_func(_sht._synthesize_ring_kernel)(*args)
        for a, b in zip(compiled, interpreted):
            assert np.allclose(a, b, atol=1e-15)

    def test_analyze_numba_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_sht._analyze_numba, "py_func"), "kernel must be @njit-decorated"
        sht = _sht.SphericalHarmonicTransform(8, "legendre", 11)
        north, south = _random_hemispheres(sht.dim)
        amn, bmn, offsets, cos_table, sin_table = self._transform_arrays(sht)
        args = (
            north,
            south,
            sht.bandwidth,
            sht.cos_latitudes,
            sht.quadrature_weights,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
            offsets,
            cos_table,
            sin_table,
        )
        compiled = _sht._analyze_numba(*args)
        interpreted = _py_func(_sht._analyze_numba)(*args)
        assert np.allclose(compiled, interpreted, atol=1e-14)

    def test_synthesize_numba_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_sht._synthesize_numba, "py_func"), (
            "kernel must be @njit-decorated"
        )
        sht = _sht.SphericalHarmonicTransform(8, "legendre", 11)
        alm = _random_alm(sht.bandwidth)
        amn, bmn, offsets, cos_table, sin_table = self._transform_arrays(sht)
        args = (
            alm,
            sht.dim,
            sht.cos_latitudes,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
            offsets,
            cos_table,
            sin_table,
        )
        compiled = _sht._synthesize_numba(*args)
        interpreted = _py_func(_sht._synthesize_numba)(*args)
        for a, b in zip(compiled, interpreted):
            assert np.allclose(a, b, atol=1e-14)

    def test_rfft_helpers_agree_with_the_numba_kernels(self):
        sht = _sht.SphericalHarmonicTransform(8, "legendre", 11)
        north, south = _random_hemispheres(sht.dim)
        amn, bmn, offsets, cos_table, sin_table = self._transform_arrays(sht)
        alm_numba = _sht._analyze_numba(
            north,
            south,
            sht.bandwidth,
            sht.cos_latitudes,
            sht.quadrature_weights,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
            offsets,
            cos_table,
            sin_table,
        )
        alm_rfft = _sht._analyze_rfft(
            north,
            south,
            sht.bandwidth,
            sht.cos_latitudes,
            sht.quadrature_weights,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
        )
        assert np.allclose(alm_numba, alm_rfft, atol=1e-12)
        north_numba, south_numba = _sht._synthesize_numba(
            alm_rfft,
            sht.dim,
            sht.cos_latitudes,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
            offsets,
            cos_table,
            sin_table,
        )
        north_rfft, south_rfft = _sht._synthesize_rfft(
            alm_rfft,
            sht.dim,
            sht.cos_latitudes,
            sht.ring_offsets,
            sht.ring_indices,
            amn,
            bmn,
        )
        assert np.allclose(north_numba, north_rfft, atol=1e-12)
        assert np.allclose(south_numba, south_rfft, atol=1e-12)


class TestNickelMasterPattern:
    """The shipped Ni master pattern is bit-for-bit identical on both
    hemispheres and invariant under all eight square symmetries, so it
    cannot lock axis conventions or the antisymmetric branch. Those are
    locked by the synthetic probes above.
    """

    bandwidth = 100
    dim = 201

    @pytest.fixture(scope="class")
    def hemispheres(self):
        mp = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        data = mp.data[:, ::2, ::2].astype(np.float64)
        assert data.shape == (2, self.dim, self.dim)
        return data[0], data[1]

    def _transform(self):
        """Build the transformer inside each test, so that a missing
        implementation fails the test instead of erroring its setup.
        """
        return _sht.SphericalHarmonicTransform(self.bandwidth, "lambert", self.dim)

    def test_orders_not_divisible_by_four_are_structurally_zero(
        self, hemispheres, record_property
    ):
        # m-3m has a four fold axis along z, so only m % 4 == 0
        # survives
        north, south = hemispheres
        alm = self._transform().analyze(north, south)
        power = np.sum(np.abs(alm) ** 2, axis=1)
        m = np.arange(self.bandwidth)
        ratio = power[m % 4 != 0].sum() / power.sum()
        record_property("ni_relative_power_m_mod_4", f"{ratio:.3e}")
        assert ratio < 1e-25

    def test_antisymmetric_branch_is_zero_for_equal_hemispheres(self, hemispheres):
        north, south = hemispheres
        alm = self._transform().analyze(north, south)
        odd = self._odd_mask(self.bandwidth)
        assert np.sum(np.abs(alm[odd]) ** 2) == 0.0

    def test_antisymmetric_branch_is_alive_for_unequal_hemispheres(
        self, hemispheres, record_property
    ):
        north, south = hemispheres
        alm = self._transform().analyze(north, 0.5 * south)
        odd = self._odd_mask(self.bandwidth)
        power = np.abs(alm) ** 2
        record_property(
            "ni_relative_power_odd_l_plus_m_halved_south",
            f"{power[odd].sum() / power.sum():.3e}",
        )
        assert power[odd].sum() > 1e-3 * power.sum()

    def test_dc_coefficient_matches_the_solid_angle_weighted_mean(
        self, hemispheres, record_property
    ):
        north, south = hemispheres
        alm = self._transform().analyze(north, south)
        weight = self._hemisphere_weights(self.dim)
        mean = np.sum(weight * (north + south)) / (2 * np.sum(weight))
        determined = alm[0, 0].real / SQRT_FOUR_PI
        record_property("ni_dc", f"{determined:.9f} vs weighted mean {mean:.9f}")
        assert determined == pytest.approx(mean, rel=1e-2)

    def test_parseval_sum_matches_the_weighted_mean_square(
        self, hemispheres, record_property
    ):
        north, south = hemispheres
        alm = self._transform().analyze(north, south)
        power = np.abs(alm) ** 2
        total = power[0].sum() + 2 * power[1:].sum()
        weight = self._hemisphere_weights(self.dim)
        mean_square = np.sum(weight * (north**2 + south**2)) / (2 * np.sum(weight))
        expected = 4 * np.pi * mean_square
        record_property("ni_parseval", f"{total:.6e} vs 4 pi <f^2> {expected:.6e}")
        assert total == pytest.approx(expected, rel=1e-2)

    @staticmethod
    def _odd_mask(bandwidth):
        """Return a mask of the odd ``l + m`` coefficients."""
        order, degree = np.meshgrid(
            np.arange(bandwidth), np.arange(bandwidth), indexing="ij"
        )
        return ((degree + order) % 2 == 1) & (degree >= order)

    @staticmethod
    def _hemisphere_weights(dim):
        """Return per pixel solid angles for one hemisphere, halving
        the equator so that the double cover is not counted twice.
        """
        weight = _grid.lambert_solid_angles(dim).copy()
        weight[0] *= 0.5
        weight[-1] *= 0.5
        weight[:, 0] *= 0.5
        weight[:, -1] *= 0.5
        return weight


class TestTimingBaseline:
    @pytest.mark.parametrize(
        "bandwidth, layout, dim",
        [
            (68, "legendre", 71),
            (128, "legendre", 131),
            pytest.param(384, "legendre", 387, marks=pytest.mark.weekly),
        ],
    )
    def test_analyze_timing_baseline_is_recorded(
        self, bandwidth, layout, dim, record_property
    ):
        sht = _sht.SphericalHarmonicTransform(bandwidth, layout, dim)
        north, south = _random_hemispheres(dim)
        sht.analyze(north, south)  # warm the Numba cache
        start = time.perf_counter()
        sht.analyze(north, south)
        elapsed = time.perf_counter() - start
        record_property(
            f"analyze_seconds_bw{bandwidth}_{layout}_dim{dim}", f"{elapsed:.4f}"
        )
