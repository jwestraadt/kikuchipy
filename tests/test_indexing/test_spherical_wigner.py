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

"""Tests of ``kikuchipy.indexing._spherical._wigner`` and of the
reference tables in ``kikuchipy.data.emsphinx``.

Covers the "Reference tables", "Wigner d", ``rotate_harmonics``,
"Derivatives", "Kernels and conventions" and "Weekly" assertions of
``specs/2026-08-16-sht-wigner-d/validation.md``:

- The seven Mathematica tables transcribed from
  ``EMSphInx/test/sht/wigner.cpp`` are structurally sound, and the
  ports reproduce them at the C++ tolerances: ``2 eps`` for ``d`` and
  the single ``D^3_{2,1}`` value (``testDjkm``, lines 112-394) and
  ``24 eps`` for the two derivatives (``testDerivatives``, lines
  563-868).
- The symmetry identities of ``wigner.hpp`` lines 301-315, the NaN
  rule, the ``t = +-1`` edges, unitarity and a
  :func:`scipy.special.eval_jacobi` closed form up to degree 511,
  including the pinned underflow limitation of the naive recursion.
- The table constructors are bitwise equal to the scalar function
  (port of ``testTables(15)``, lines 405-555), to each other, and
  carry NaN on exactly the undefined slots.
- ``rotate_harmonics`` satisfies the group identities, a
  dependency-free brute force ``sum_n a^l_n D^l_{m,n}`` oracle which
  pins the direction of the transform even where SciPy is too old
  for ``sph_harm_y``, the ``sph_harm_y`` direction oracle pointwise
  and through Phase 1's transform, and the symmetry invariances of
  the shipped Ni master pattern.
- The derivatives against five point finite differences and against
  the table based formulas of ``EMSphInx/include/sht/sht_xcorr.hpp``
  lines 1009-1041, which Phase 7 will inline.
- ``.py_func`` of every kernel (including the NaN guard of the two
  derivatives, which only the ``py_func`` path pins), the Numba
  compilation flags, ``KERNEL_NAMES`` covering every kernel of the
  module, the ``ValueError`` paths and recorded timing and memory
  baselines.
"""

import math
import time

import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d
import pytest

import kikuchipy as kp
from kikuchipy.data.emsphinx import wigner_reference_tables as ref
from kikuchipy.indexing._spherical import _euler, _grid, _sht, _wigner

EPS = float(np.finfo(np.float64).eps)
TWO_PI = 2 * math.pi

# pi / 2 / golden ratio, the "random" angle of wigner.cpp line 412
BETA_TEST = 0.9708055194

# Bandwidths of the default and the weekly table sweeps
TABLE_BANDWIDTHS = [1, 2, 3, 15, 32]
WEEKLY_BANDWIDTHS = [68, 88, 113]
WEEKLY_BETAS = [BETA_TEST, 0.3, 2.5, math.pi / 2, 1e-3, math.pi - 1e-3]

# The Ni master pattern sub-grid used by the real data tests
NI_BANDWIDTH = 50
NI_DIM = 101

# The seven reference tables and the angle each was tabulated at
REFERENCE_TABLES = [
    ("D_PI_2", ref.BETA_PI_2),
    ("D_PI_3", ref.BETA_PI_3),
    ("D_2PI_3", ref.BETA_2PI_3),
    ("D_PRIME_PI_3", ref.BETA_PI_3),
    ("D_PRIME_2PI_3", ref.BETA_2PI_3),
    ("D_PRIME2_PI_3", ref.BETA_PI_3),
    ("D_PRIME2_2PI_3", ref.BETA_2PI_3),
]

# Every Numba kernel of the module, for the flag test
KERNEL_NAMES = [
    "_u_jkm_0",
    "_u_jkm_1",
    "_u_jkm_2",
    "_v_jkm",
    "_w_jkm",
    "_a_jkm_0",
    "_a_jkm_1",
    "_a_jkm_2",
    "_a_jkm_0_pre",
    "_a_jkm_2_pre",
    "_b_jkm",
    "_u_km_0",
    "_u_km_1",
    "_u_km_2",
    "_a_km_0",
    "_a_km_1",
    "_a_km_2",
    "_e_km",
    "_wigner_d_core",
    "wigner_d",
    "wigner_d_half_pi",
    "wigner_d_sign",
    "wigner_D",
    "_wigner_d_table_kernel",
    "_wigner_d_table_factors_kernel",
    "_wigner_d_table_pre_kernel",
    "_wigner_d_half_pi_table_kernel",
    "_rotate_harmonics_kernel",
    "wigner_d_prime",
    "wigner_d_prime2",
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


def _sph_harm_y():
    """Return ``scipy.special.sph_harm_y``, skipping if SciPy is too
    old.

    The gate is on the top level package, since ``scipy.special`` has
    no ``__version__``.
    """
    pytest.importorskip("scipy", minversion="1.15")
    from scipy.special import sph_harm_y

    return sph_harm_y


def _closed_form(j, k, m, beta):
    """Return ``d^j_{k,m}(beta)`` from Fukushima equation 1.

    The prefactor, the two half angle powers and the Jacobi
    polynomial are combined in log space, so that this oracle does
    **not** underflow where the recursion of ``wigner.hpp`` does
    (which is the point of the underflow tests below): the direct
    product form gives ``-0.0`` at ``(511, 400, 300)`` and
    ``beta = 2.5`` while the log form gives the true
    ``-1.209e-184``.

    Valid for ``0 <= m <= k <= j`` and ``0 < beta < pi``.
    """
    from scipy.special import eval_jacobi, gammaln

    if j < max(abs(k), abs(m)):
        return math.nan
    log_value = 0.5 * (
        gammaln(j + k + 1)
        + gammaln(j - k + 1)
        - gammaln(j + m + 1)
        - gammaln(j - m + 1)
    )
    log_value += (k + m) * math.log(math.cos(beta / 2))
    log_value += (k - m) * math.log(math.sin(beta / 2))
    jacobi = eval_jacobi(j - k, k - m, k + m, math.cos(beta))
    if jacobi == 0.0:
        return 0.0
    return math.copysign(math.exp(log_value + math.log(abs(jacobi))), jacobi)


def _random_alm(bandwidth, seed=0):
    """Return a random band-limited spectrum of a real function.

    Entries with ``l < m`` are zero and the ``m == 0`` entries are
    real, as in Phase 1's tests and
    ``EMSphInx/test/sht/square_sht.cpp`` lines 98-107.
    """
    rng = np.random.default_rng(seed)
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for order in range(bandwidth):
        for degree in range(order, bandwidth):
            if order == 0:
                alm[order, degree] = rng.uniform(-1, 1)
            else:
                alm[order, degree] = rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)
    return alm


def _random_zyz(n, seed=1):
    """Return ``n`` random ZYZ triples with ``beta`` in [-pi, pi]."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-math.pi, math.pi, n),
            rng.uniform(-math.pi, math.pi, n),
            rng.uniform(-math.pi, math.pi, n),
        ],
        axis=1,
    )


def _defined_mask(bandwidth):
    """Return the ``(bw, bw, bw)`` mask of slots ``j >= max(k, m)``."""
    index = np.arange(bandwidth)
    k = index[:, np.newaxis, np.newaxis]
    m = index[np.newaxis, :, np.newaxis]
    j = index[np.newaxis, np.newaxis, :]
    return j >= np.maximum(k, m)


def _scalar_beta_table(bandwidth, t, negative_beta):
    """Return the two beta table slots built from
    :func:`kikuchipy.indexing._spherical._wigner.wigner_d` alone.
    """
    slot0 = np.full((bandwidth,) * 3, np.nan)
    slot1 = np.full((bandwidth,) * 3, np.nan)
    for k in range(bandwidth):
        for m in range(bandwidth):
            for j in range(max(k, m), bandwidth):
                slot0[k, m, j] = _wigner.wigner_d(j, k, m, t, negative_beta)
                slot1[k, m, j] = _wigner.wigner_d(j, k, m, -t, negative_beta)
    return slot0, slot1


def _brute_force_rotate(alm, zyz, transpose=False):
    """Return ``b^l_m = sum_n a^l_n D^l_{m,n}(zyz)`` by direct
    summation.

    Uses only NumPy and the module's own
    :func:`kikuchipy.indexing._spherical._wigner.wigner_D`, which is
    itself pinned to Mathematica at ``2 eps``, so this oracle needs
    no SciPy and runs on the oldest supported job. With
    ``transpose=True`` it sums ``D^l_{n,m}`` instead, the guard
    against the transposed table read of the rotation kernel.
    """
    bandwidth = alm.shape[0]
    angles = np.asarray(zyz, dtype=np.float64)
    out = np.zeros_like(alm)
    for m in range(bandwidth):
        for degree in range(m, bandwidth):
            total = 0j
            for n in range(-degree, degree + 1):
                if n >= 0:
                    coefficient = alm[n, degree]
                else:
                    coefficient = (-1) ** (-n) * np.conj(alm[-n, degree])
                if transpose:
                    d = _wigner.wigner_D(degree, n, m, angles)
                else:
                    d = _wigner.wigner_D(degree, m, n, angles)
                total += coefficient * d
            out[m, degree] = total
    return out


def _evaluate(sph_harm_y, alm, directions):
    """Return the real function of ``alm`` at unit vectors.

    ``alm[m, l]`` holds only ``m >= 0``, and
    ``a^l_{-m} = (-1)^m conj(a^l_m)``, so the ``m > 0`` terms are
    doubled real parts (Phase 1 convention, Condon-Shortley included
    in ``sph_harm_y``).
    """
    bandwidth = alm.shape[0]
    flat = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    theta = np.arccos(np.clip(flat[:, 2], -1.0, 1.0))
    phi = np.arctan2(flat[:, 1], flat[:, 0])
    total = np.zeros(flat.shape[0])
    for degree in range(bandwidth):
        for order in range(degree + 1):
            coefficient = alm[order, degree]
            if coefficient == 0:
                continue
            harmonic = sph_harm_y(degree, order, theta, phi)
            weight = 1.0 if order == 0 else 2.0
            total += weight * (coefficient * harmonic).real
    return total.reshape(np.asarray(directions).shape[:-1])


def _phase7_derivatives(table, m, n, j, t, negative_beta):
    """Return ``(d1P, d1N, d2P, d2N)`` of ``sht_xcorr.hpp``.

    Verbatim transcription of ``Correlator::derivatives()``
    (``EMSphInx/include/sht/sht_xcorr.hpp`` lines 1009-1041) which
    Phase 7 will inline into its Newton kernel: the first
    derivatives at lines 1038-1039 and the second at lines
    1040-1041, with ``d0P_1``/``d0N_1`` zero for ``m >= j`` and
    ``d0P_2``/``d0N_2`` zero for ``m + 1 >= j``. It exists here so
    that Phase 7 inherits a pinned formula, not to be shipped.
    """
    csc = (-1.0 if negative_beta else 1.0) / math.sqrt(1.0 - t * t)
    mm = m * m
    nn = n * n
    coef2_0a = t * t * mm + (nn - m)
    coef2_0b = t * n * (1 - 2 * m)
    coef2_1a = t * (1 + 2 * m)
    coef1_0pp = (t * m - n) * csc
    coef1_0pn = (t * m + n) * csc
    coef2_0pp = (coef2_0a + coef2_0b) * csc * csc
    coef2_0pn = (coef2_0a - coef2_0b) * csc * csc
    coef2_1pp = (coef2_1a - 2 * n) * csc
    coef2_1pn = (coef2_1a + 2 * n) * csc

    d0p = table[m, n, j, 0]
    d0n = table[m, n, j, 1]
    d0p_1 = 0.0 if m >= j else table[m + 1, n, j, 0]
    d0n_1 = 0.0 if m >= j else table[m + 1, n, j, 1]
    d0p_2 = 0.0 if m + 1 >= j else table[m + 2, n, j, 0]
    d0n_2 = 0.0 if m + 1 >= j else table[m + 2, n, j, 1]

    jm = j - m
    rjm = math.sqrt((jm) * (j + m + 1))
    coef2_2 = 0.0 if jm == 0 else math.sqrt((jm - 1) * (j + m + 2)) * rjm
    d1p = d0p * coef1_0pp - d0p_1 * rjm
    d1n = d0n * coef1_0pn + d0n_1 * rjm
    d2p = d0p * coef2_0pp - d0p_1 * rjm * coef2_1pp + d0p_2 * coef2_2
    d2n = d0n * coef2_0pn + d0n_1 * rjm * coef2_1pn + d0n_2 * coef2_2
    return d1p, d1n, d2p, d2n


@pytest.fixture(scope="module")
def nickel_alm():
    """Return the harmonics of the shipped Ni master pattern.

    The ``dim`` 101 centred sub-grid (``bw`` 50) of the ``(2, 401,
    401)`` uint8 master, where the aliasing amplification of Phase
    1's Lambert layout is mild.
    """
    master_pattern = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    data = master_pattern.data[:, ::4, ::4].astype(np.float64)
    assert data.shape == (2, NI_DIM, NI_DIM)
    transform = _sht.SphericalHarmonicTransform(NI_BANDWIDTH, "lambert", NI_DIM)
    return transform.analyze(data[0], data[1])


def _d_of_beta(j, k, m, beta):
    """Return ``d^j_{k,m}(beta)`` for a signed ``beta``."""
    return _wigner.wigner_d(j, k, m, math.cos(beta), math.copysign(1.0, beta) < 0)


class TestReferenceTables:
    """Structural tests of
    ``kikuchipy.data.emsphinx.wigner_reference_tables``, the verbatim
    transcription of the Mathematica tables of ``wigner.cpp``.
    """

    @staticmethod
    def _index_set():
        return {
            (j, k, m)
            for j in range(ref.NUM)
            for k in range(-(ref.NUM - 1), ref.NUM)
            for m in range(-(ref.NUM - 1), ref.NUM)
            if abs(k) <= j and abs(m) <= j
        }

    @pytest.mark.parametrize("name, _", REFERENCE_TABLES)
    def test_each_table_has_the_165_defined_entries(self, name, _):
        table = getattr(ref, name)
        assert len(table) == 165
        assert set(table) == self._index_set()

    @pytest.mark.parametrize("name, _", REFERENCE_TABLES)
    def test_every_value_is_finite(self, name, _):
        table = getattr(ref, name)
        assert all(np.isfinite(value) for value in table.values())

    @pytest.mark.parametrize("name, _", REFERENCE_TABLES)
    def test_as_array_puts_nan_on_the_undefined_slots(self, name, _):
        array = ref.as_array(getattr(ref, name))
        assert array.shape == (5, 9, 9)
        assert array.dtype == np.float64
        nan = np.isnan(array)
        assert nan.sum() == 240
        for j in range(ref.NUM):
            for k in range(-4, 5):
                for m in range(-4, 5):
                    undefined = j < max(abs(k), abs(m))
                    assert nan[j, k + 4, m + 4] == undefined, (j, k, m)

    @pytest.mark.parametrize(
        "name, key, value",
        [
            ("D_PI_2", (0, 0, 0), 1.0),
            ("D_PI_2", (1, 1, 0), 0.7071067811865476),
            ("D_PI_2", (1, 0, 1), -0.7071067811865476),
            ("D_PI_2", (4, 4, 4), 0.0625),
            ("D_PI_3", (2, 2, 1), 0.649519052838329),
            ("D_PI_3", (2, 1, 0), 0.5303300858899106),
            ("D_2PI_3", (3, -2, 1), 0.2567449488305466),
            ("D_PRIME_PI_3", (2, 1, 0), -0.6123724356957945),
            ("D_PRIME2_PI_3", (2, 1, 0), -2.1213203435596424),
        ],
    )
    def test_hand_checked_entries(self, name, key, value):
        assert getattr(ref, name)[key] == value

    def test_the_single_wigner_uppercase_d_value(self):
        assert ref.D_3_2_1 == (
            0.19764235376052370824993084652704 - 0.34232659844072882091060611425050j
        )

    def test_the_angles_are_the_tabulated_ones(self):
        assert ref.BETA_PI_2 == math.pi / 2
        assert ref.BETA_PI_3 == math.pi / 3
        assert ref.BETA_2PI_3 == 2 * math.pi / 3
        assert ref.NUM == 5


class TestMathematicaTables:
    """Port of ``testDjkm()``, ``EMSphInx/test/sht/wigner.cpp`` lines
    112-394, at its own ``2 eps`` tolerance.
    """

    tolerance = 2 * EPS

    @staticmethod
    def _cases():
        """Return ``(label, callable, reference array)`` for the six
        comparisons of ``wigner.cpp`` lines 295-311.
        """
        pi_2 = ref.as_array(ref.D_PI_2)
        pi_3 = ref.as_array(ref.D_PI_3)
        two_pi_3 = ref.as_array(ref.D_2PI_3)
        swap = (0, 2, 1)
        return [
            ("pi/2 special", lambda j, k, m: _wigner.wigner_d_half_pi(j, k, m), pi_2),
            (
                "pi/2",
                lambda j, k, m: _wigner.wigner_d(j, k, m, 0.0, False),
                pi_2,
            ),
            ("pi/3", lambda j, k, m: _wigner.wigner_d(j, k, m, 0.5, False), pi_3),
            (
                "-pi/3",
                lambda j, k, m: _wigner.wigner_d(j, k, m, 0.5, True),
                pi_3.transpose(swap),
            ),
            (
                "2pi/3",
                lambda j, k, m: _wigner.wigner_d(j, k, m, -0.5, False),
                two_pi_3,
            ),
            (
                "-2pi/3",
                lambda j, k, m: _wigner.wigner_d(j, k, m, -0.5, True),
                two_pi_3.transpose(swap),
            ),
        ]

    @pytest.mark.parametrize("case", range(6))
    def test_wigner_d_matches_mathematica(self, case, record_property):
        label, function, expected = self._cases()[case]
        worst = 0.0
        for j in range(ref.NUM):
            for k in range(-4, 5):
                for m in range(-4, 5):
                    got = function(j, k, m)
                    want = expected[j, k + 4, m + 4]
                    if np.isnan(got) and np.isnan(want):
                        continue
                    assert not np.isnan(got), (j, k, m, label)
                    assert not np.isnan(want), (j, k, m, label)
                    worst = max(worst, abs(got - want))
        record_property(f"mathematica_worst_{label}", f"{worst:.3e}")
        assert worst <= self.tolerance

    def test_undefined_indices_are_nan_in_both(self):
        expected = ref.as_array(ref.D_PI_2)
        for j in range(ref.NUM):
            for k in range(-4, 5):
                for m in range(-4, 5):
                    if j >= max(abs(k), abs(m)):
                        continue
                    assert np.isnan(expected[j, k + 4, m + 4])
                    assert np.isnan(_wigner.wigner_d_half_pi(j, k, m))
                    assert np.isnan(_wigner.wigner_d(j, k, m, 0.5, False))

    def test_wigner_d_sign_reproduces_the_pi_over_two_table(self, record_property):
        # wigner.cpp lines 342-370; nan == nan is False, so the
        # undefined slots are skipped as the C++ does and their
        # positions are pinned by the NaN tests instead
        expected = ref.as_array(ref.D_PI_2)
        mismatches = 0
        compared = 0
        for j in range(ref.NUM):
            for k in range(-4, 5):
                for m in range(-4, 5):
                    signed = expected[j, k + 4, m + 4]
                    absolute = expected[j, abs(k) + 4, abs(m) + 4]
                    scaled = absolute * _wigner.wigner_d_sign(j, k, m)
                    if np.isnan(signed) and np.isnan(scaled):
                        continue
                    compared += 1
                    mismatches += int(signed != scaled)
        record_property("d_sign_compared", str(compared))
        assert compared == 165
        assert mismatches == 0

    def test_wigner_uppercase_d_matches_mathematica(self, record_property):
        # wigner.cpp lines 372-390: WignerD[{3,2,1}, Pi/6, Pi/2, Pi/3]
        zyz = np.array([math.pi / 3, math.pi / 2, math.pi / 6])
        got = _wigner.wigner_D(3, 2, 1, zyz)
        delta = abs(got - ref.D_3_2_1)
        record_property("wigner_D_321_delta", f"{delta:.3e}")
        assert delta <= self.tolerance

    def test_wigner_uppercase_d_keeps_the_sign_of_a_zero_component(self):
        # wigner.hpp line 438 multiplies a std::complex<Real> by a
        # Real, which libstdc++ scales componentwise, so a slot with
        # sin(total) == 0.0 and d < 0 has imaginary part -0.0. The
        # complex(cos, sin) * d form promotes d to a complex number
        # and uses the four multiplication rule instead, which gives
        # +0.0 on 23 of 9464 sampled slots, all with k == m == 0
        checked = 0
        for j in range(1, 6):
            for beta in (1.1, 2.0, 2.9, -0.7):
                d = _wigner.wigner_d(j, 0, 0, math.cos(beta), beta < 0.0)
                if not d < 0.0:
                    continue
                # alpha and gamma are zero, so total == 0 and
                # sin(total) is +0.0
                got = _wigner.wigner_D(j, 0, 0, np.array([0.0, beta, 0.0]))
                assert got.real == d, (j, beta)
                assert got.imag == 0.0, (j, beta)
                assert math.copysign(1.0, got.imag) < 0.0, (j, beta)
                checked += 1
        assert checked >= 8


class TestSymmetries:
    """The identities of ``wigner.hpp`` lines 301-315.

    Two of the five are exact for every argument and three only when
    both orders are non-zero; see :meth:`_assert_identity`.
    """

    @staticmethod
    def _assert_identity(lhs, rhs, j, k, m, t):
        """Assert one identity of ``wigner.hpp`` lines 301-315.

        With both orders non-zero the symmetry reduction of
        ``wigner_d()`` sends the two sides of an identity to the very
        same core recursion, so they agree bit for bit. When ``k`` or
        ``m`` is zero one side no longer trips the branch it is meant
        to trip -- ``d^j_{k,-0}`` *is* ``d^j_{k,0}`` -- so the two
        sides run the recursion at ``t`` and at ``-t`` and agree only
        to rounding. Measured with a faithful transcription on 20000
        random draws: 408, 219 and 205 of the draws are inexact for
        the three identities, worst 7.216e-16 (3.25 eps), and not one
        inexact draw has both orders non-zero.
        """
        if k != 0 and m != 0:
            assert lhs == rhs, (j, k, m, t)
        else:
            assert abs(lhs - rhs) <= 8 * EPS, (j, k, m, t)

    @staticmethod
    def _random_indices(n=200, seed=0):
        rng = np.random.default_rng(seed)
        cases = []
        while len(cases) < n:
            j = int(rng.integers(0, 41))
            k = int(rng.integers(-j, j + 1)) if j else 0
            m = int(rng.integers(-j, j + 1)) if j else 0
            t = float(rng.uniform(-1, 1))
            cases.append((j, k, m, t))
        return cases

    def test_negative_beta_swaps_the_orders(self):
        for j, k, m, t in self._random_indices():
            assert _wigner.wigner_d(j, k, m, t, True) == _wigner.wigner_d(
                j, m, k, t, False
            ), (j, k, m, t)

    def test_both_orders_negated(self):
        for j, k, m, t in self._random_indices():
            sign = 1 if (k - m) % 2 == 0 else -1
            self._assert_identity(
                _wigner.wigner_d(j, -k, -m, t, False),
                sign * _wigner.wigner_d(j, k, m, t, False),
                j,
                k,
                m,
                t,
            )

    def test_second_order_negated(self):
        for j, k, m, t in self._random_indices():
            sign = 1 if (j + k) % 2 == 0 else -1
            self._assert_identity(
                _wigner.wigner_d(j, k, -m, t, False),
                sign * _wigner.wigner_d(j, k, m, -t, False),
                j,
                k,
                m,
                t,
            )

    def test_first_order_negated(self):
        for j, k, m, t in self._random_indices():
            sign = 1 if (j + m) % 2 == 0 else -1
            self._assert_identity(
                _wigner.wigner_d(j, -k, m, t, False),
                sign * _wigner.wigner_d(j, k, m, -t, False),
                j,
                k,
                m,
                t,
            )

    def test_orders_swapped(self):
        for j, k, m, t in self._random_indices():
            sign = 1 if (k - m) % 2 == 0 else -1
            assert _wigner.wigner_d(j, m, k, t, False) == (
                sign * _wigner.wigner_d(j, k, m, t, False)
            ), (j, k, m, t)

    @pytest.mark.parametrize("beta", [4.0, -4.0, 7.0, -7.0, -0.6, 1.1, math.pi + 0.5])
    def test_wigner_uppercase_d_wraps_beta(self, beta):
        # D takes cos(beta) and signbit(beta) separately, so without
        # the [-pi, pi] wrap of sht_xcorr.hpp lines 895-899 a beta
        # outside the principal interval silently transposes the two
        # orders through equation 5, an O(1) error (measured up to
        # 1.20 at beta = 2 pi - 0.6). rotate_harmonics has its own
        # wrap test; this is the same guard on the scalar D
        alpha, gamma = 0.3, -1.1
        wrapped = _euler.wrap_beta(beta)
        negative = math.copysign(1.0, wrapped) < 0.0
        for j, k, m in [(3, 2, 1), (4, -3, 2), (5, 1, -4), (2, 0, 2), (6, -5, -2)]:
            got = _wigner.wigner_D(j, k, m, np.array([alpha, beta, gamma]))
            total = alpha * m + gamma * k
            expected = _wigner.wigner_d(j, k, m, math.cos(wrapped), negative) * complex(
                math.cos(total), math.sin(total)
            )
            assert got == pytest.approx(expected, abs=8 * EPS), (j, k, m, beta)
            # 2 pi periodicity, exact on the wrapped form and broken
            # by the unwrapped one whenever the shift crosses zero
            shifted = _wigner.wigner_D(j, k, m, np.array([alpha, beta + TWO_PI, gamma]))
            assert shifted == pytest.approx(got, abs=8 * EPS), (j, k, m, beta)

    @pytest.mark.parametrize("interpreted", [False, True], ids=["njit", "py_func"])
    @pytest.mark.parametrize(
        "function",
        ["wigner_d", "wigner_d_half_pi", "wigner_d_prime", "wigner_d_prime2"],
    )
    def test_nan_exactly_when_the_degree_is_too_small(self, function, interpreted):
        # the py_func run is what pins the explicit NaN guard of the
        # two derivatives (the first deviation of requirements.md):
        # drop that guard and the compiled kernel still returns NaN
        # through Numba's sqrt of a negative number, while the pure
        # Python function raises ValueError at (j, k, m) = (0, 1, 0)
        kernel = getattr(_wigner, function)
        assert hasattr(kernel, "py_func"), f"{function} must be @njit-decorated"
        call = _py_func(kernel) if interpreted else kernel
        for j in range(-2, 6):
            for k in range(-6, 7):
                for m in range(-6, 7):
                    undefined = j < max(abs(k), abs(m))
                    if function == "wigner_d_half_pi":
                        got = call(j, k, m)
                    else:
                        got = call(j, k, m, 0.5, False)
                    assert np.isnan(got) == undefined, (
                        function,
                        interpreted,
                        j,
                        k,
                        m,
                    )


class TestEdges:
    @pytest.mark.parametrize("negative_beta", [False, True])
    def test_beta_zero_gives_the_kronecker_delta(self, negative_beta):
        worst = 0.0
        for j in range(9):
            for k in range(-j, j + 1):
                for m in range(-j, j + 1):
                    got = _wigner.wigner_d(j, k, m, 1.0, negative_beta)
                    assert not np.isnan(got), (j, k, m)
                    worst = max(worst, abs(got - float(k == m)))
        assert worst < 1e-14

    def test_beta_pi_gives_the_mirrored_kronecker_delta(self):
        worst = 0.0
        for j in range(9):
            for k in range(-j, j + 1):
                for m in range(-j, j + 1):
                    sign = 1 if (j + k) % 2 == 0 else -1
                    expected = sign * float(m == -k)
                    got = _wigner.wigner_d(j, k, m, -1.0, False)
                    assert not np.isnan(got), (j, k, m)
                    worst = max(worst, abs(got - expected))
        assert worst < 1e-14

    @pytest.mark.parametrize("t", [1.0, -1.0])
    def test_the_table_has_no_nan_in_a_defined_slot_at_the_edges(self, t):
        table = _wigner.wigner_d_table(8, t, False)
        mask = _defined_mask(8)
        assert not np.isnan(table[..., 0][mask]).any()
        assert not np.isnan(table[..., 1][mask]).any()


class TestHighDegree:
    @pytest.mark.parametrize("degree", [15, 63, 127, 511])
    def test_unitarity(self, degree, record_property):
        t = math.cos(BETA_TEST)
        worst = 0.0
        for k in (0, degree // 2, degree):
            total = 0.0
            for m in range(-degree, degree + 1):
                total += _wigner.wigner_d(degree, k, m, t, False) ** 2
            worst = max(worst, abs(total - 1.0))
        record_property(f"unitarity_j{degree}", f"{worst:.3e}")
        assert worst < 1e-12

    @pytest.mark.parametrize("beta", [BETA_TEST, 2.5])
    @pytest.mark.parametrize(
        "j, k, m",
        [
            (1, 1, 0),
            (2, 2, 1),
            (5, 3, 1),
            (15, 10, 4),
            (63, 40, 20),
            (127, 100, 90),
            (300, 250, 200),
            (511, 0, 0),
            (511, 511, 0),
        ],
    )
    def test_matches_the_jacobi_closed_form(self, j, k, m, beta, record_property):
        expected = _closed_form(j, k, m, beta)
        got = _wigner.wigner_d(j, k, m, math.cos(beta), False)
        assert np.isfinite(got)
        relative = abs(got - expected) / abs(expected)
        record_property(f"closed_form_{j}_{k}_{m}_beta{beta}", f"{relative:.3e}")
        assert relative <= 1e-10

    def test_matches_the_closed_form_at_the_deepest_index(self, record_property):
        # the same slot which underflows at beta 2.5 below
        expected = _closed_form(511, 400, 300, BETA_TEST)
        got = _wigner.wigner_d(511, 400, 300, math.cos(BETA_TEST), False)
        relative = abs(got - expected) / abs(expected)
        record_property("closed_form_511_400_300", f"{relative:.3e}")
        assert relative <= 1e-10

    def test_the_recursion_underflows_to_exactly_zero(self, record_property):
        # PINNED LIMITATION, not a bug to fix here: without
        # Fukushima's extended exponent arithmetic the seed
        # c2 ** (k + m) underflows (cos(1.25) ** 700 == 0.0), so the
        # whole (k, m) column is returned as exactly 0.0 while the
        # true value is -1.209e-184 (closed form, confirmed with
        # mpmath at 50 digits). The test exists so a later phase
        # raising the bandwidth meets the limitation by name instead
        # of retuning the closed form tolerance around it
        got = _wigner.wigner_d(511, 400, 300, math.cos(2.5), False)
        true_value = _closed_form(511, 400, 300, 2.5)
        record_property("underflow_true_value", f"{true_value:.6e}")
        assert got == 0.0
        assert abs(true_value) < 1e-180

    @pytest.mark.weekly
    @pytest.mark.parametrize(
        "degree, stride, betas",
        [
            (511, 7, (2.5, 3.0, BETA_TEST)),
            (127, 3, (2.5, 3.0, 3.13, math.pi - 1e-3)),
        ],
    )
    def test_closed_form_scan(self, degree, stride, betas, record_property):
        for beta in betas:
            t = math.cos(beta)
            zeroed = 0
            largest_zeroed = 0.0
            worst_relative = 0.0
            points = 0
            for k in range(0, degree + 1, stride):
                for m in range(0, k + 1, stride):
                    points += 1
                    got = _wigner.wigner_d(degree, k, m, t, False)
                    expected = _closed_form(degree, k, m, beta)
                    if got == 0.0 and expected != 0.0:
                        zeroed += 1
                        largest_zeroed = max(largest_zeroed, abs(expected))
                        continue
                    if abs(expected) >= 1e-100:
                        worst_relative = max(
                            worst_relative, abs(got - expected) / abs(expected)
                        )
            record_property(
                f"scan_j{degree}_beta{beta}",
                f"{points} points, {zeroed} zeroed, largest zeroed "
                f"{largest_zeroed:.3e}, worst rel {worst_relative:.3e}",
            )
            assert largest_zeroed < 1e-130
            assert worst_relative <= 1e-8


class TestBetaTables:
    """Port of ``testTables()``, ``EMSphInx/test/sht/wigner.cpp``
    lines 405-555.
    """

    @pytest.mark.parametrize("bandwidth", TABLE_BANDWIDTHS)
    @pytest.mark.parametrize("negative_beta", [False, True])
    def test_table_is_bitwise_equal_to_the_scalar_function(
        self, bandwidth, negative_beta
    ):
        # zero margin on purpose: one re-associated product in
        # _b_jkm breaks this at 1.7e-16 (requirements.md), so the
        # shared coefficient helpers are load bearing
        t = math.cos(BETA_TEST)
        table = _wigner.wigner_d_table(bandwidth, t, negative_beta)
        assert table.shape == (bandwidth, bandwidth, bandwidth, 2)
        assert table.dtype == np.float64
        assert table.flags.c_contiguous
        slot0, slot1 = _scalar_beta_table(bandwidth, t, negative_beta)
        mask = _defined_mask(bandwidth)
        assert np.array_equal(table[..., 0][mask], slot0[mask])
        assert np.array_equal(table[..., 1][mask], slot1[mask])

    @pytest.mark.parametrize("bandwidth", TABLE_BANDWIDTHS)
    def test_table_has_nan_exactly_on_the_undefined_slots(self, bandwidth):
        table = _wigner.wigner_d_table(bandwidth, math.cos(BETA_TEST), False)
        mask = _defined_mask(bandwidth)
        assert np.array_equal(np.isnan(table[..., 0]), ~mask)
        assert np.array_equal(np.isnan(table[..., 1]), ~mask)

    @pytest.mark.weekly
    @pytest.mark.parametrize("bandwidth", WEEKLY_BANDWIDTHS)
    @pytest.mark.parametrize("beta", WEEKLY_BETAS)
    @pytest.mark.parametrize("negative_beta", [False, True])
    def test_table_is_bitwise_equal_to_the_scalar_function_weekly(
        self, bandwidth, beta, negative_beta, record_property
    ):
        t = math.cos(beta)
        table = _wigner.wigner_d_table(bandwidth, t, negative_beta)
        slot0, slot1 = _scalar_beta_table(bandwidth, t, negative_beta)
        mask = _defined_mask(bandwidth)
        mismatches = int((table[..., 0][mask] != slot0[mask]).sum()) + int(
            (table[..., 1][mask] != slot1[mask]).sum()
        )
        record_property(
            f"table_vs_scalar_bw{bandwidth}_beta{beta}_nB{negative_beta}",
            f"{mismatches} of {2 * int(mask.sum())}",
        )
        assert mismatches == 0

    @pytest.mark.parametrize("bandwidth", TABLE_BANDWIDTHS)
    @pytest.mark.parametrize("negative_beta", [False, True])
    def test_pre_table_is_bitwise_equal_to_the_plain_table(
        self, bandwidth, negative_beta
    ):
        t = math.cos(BETA_TEST)
        plain = _wigner.wigner_d_table(bandwidth, t, negative_beta)
        factors = _wigner.wigner_d_table_factors(bandwidth)
        pre = _wigner.wigner_d_table_pre(bandwidth, t, negative_beta, *factors)
        mask = _defined_mask(bandwidth)
        assert np.array_equal(pre[..., 0][mask], plain[..., 0][mask])
        assert np.array_equal(pre[..., 1][mask], plain[..., 1][mask])
        assert np.array_equal(np.isnan(pre[..., 0]), ~mask)
        assert np.array_equal(np.isnan(pre[..., 1]), ~mask)

    @pytest.mark.parametrize("bandwidth", [2, 15])
    def test_pre_table_reuses_a_given_buffer(self, bandwidth):
        t = math.cos(BETA_TEST)
        factors = _wigner.wigner_d_table_factors(bandwidth)
        buffer = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        returned = _wigner.wigner_d_table_pre(bandwidth, t, False, *factors, out=buffer)
        assert returned is buffer
        expected = _wigner.wigner_d_table(bandwidth, t, False)
        mask = _defined_mask(bandwidth)
        assert np.array_equal(returned[..., 0][mask], expected[..., 0][mask])
        assert np.array_equal(np.isnan(returned[..., 0]), ~mask)
        # a buffer from a previous call at a different t still holds
        # its NaN slots, which is what Phase 7 relies on
        other = math.cos(2.0)
        again = _wigner.wigner_d_table_pre(
            bandwidth, other, False, *factors, out=buffer
        )
        assert again is buffer
        expected_other = _wigner.wigner_d_table(bandwidth, other, False)
        assert np.array_equal(again[..., 0][mask], expected_other[..., 0][mask])
        assert np.array_equal(again[..., 1][mask], expected_other[..., 1][mask])
        assert np.array_equal(np.isnan(again[..., 0]), ~mask)
        assert np.array_equal(np.isnan(again[..., 1]), ~mask)

    @pytest.mark.parametrize("bandwidth", [2, 15])
    def test_pre_table_rejects_a_buffer_which_is_not_nan_filled(self, bandwidth):
        # np.empty() would silently leave about five sixths of the
        # slots as garbage, since the kernel never writes them
        t = math.cos(BETA_TEST)
        factors = _wigner.wigner_d_table_factors(bandwidth)
        shape = (bandwidth, bandwidth, bandwidth, 2)
        empty = np.empty(shape)
        # np.empty may hand back NaN by chance, so the two tripwire
        # slots are made deterministic
        empty[bandwidth - 1, 0, 0, 0] = 0.0
        empty[0, bandwidth - 1, 0, 1] = 0.0
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(bandwidth, t, False, *factors, out=empty)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(
                bandwidth, t, False, *factors, out=np.zeros(shape)
            )

    @pytest.mark.parametrize("bandwidth", [2, 15])
    def test_pre_table_rejects_a_wrong_buffer(self, bandwidth):
        t = math.cos(BETA_TEST)
        factors = _wigner.wigner_d_table_factors(bandwidth)
        shape = (bandwidth, bandwidth, bandwidth, 2)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(
                bandwidth, t, False, *factors, out=np.full((bandwidth, 2), np.nan)
            )
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(
                bandwidth,
                t,
                False,
                *factors,
                out=np.full(shape, np.nan, dtype=np.float32),
            )
        wide = np.full((bandwidth, bandwidth, bandwidth, 4), np.nan)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(
                bandwidth, t, False, *factors, out=wide[..., ::2]
            )

    def test_negative_beta_swaps_the_two_order_axes(self):
        bandwidth = 15
        t = math.cos(BETA_TEST)
        positive = _wigner.wigner_d_table(bandwidth, t, False)
        negative = _wigner.wigner_d_table(bandwidth, t, True)
        assert np.array_equal(negative, positive.transpose(1, 0, 2, 3), equal_nan=True)

    @pytest.mark.parametrize("bandwidth", [0, -1])
    def test_bandwidth_below_one_raises(self, bandwidth):
        with pytest.raises(ValueError):
            _wigner.wigner_d_table(bandwidth, 0.5, False)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_factors(bandwidth)
        with pytest.raises(ValueError):
            _wigner.wigner_d_half_pi_table(bandwidth, False)

    @pytest.mark.parametrize("bandwidth", [0, -1])
    def test_pre_table_bandwidth_below_one_raises(self, bandwidth):
        # the other three constructors are covered by
        # test_bandwidth_below_one_raises, which cannot take
        # wigner_d_table_pre because it needs its factor tables.
        # bandwidth 0 is the case that pins the guard: at -1 the NaN
        # allocation itself refuses the negative dimensions
        factors = _wigner.wigner_d_table_factors(1)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(bandwidth, 0.5, False, *factors)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_pre_table_rejects_factors_of_another_bandwidth(self, index):
        # the kernel reads w_jkm[k, m, i] without bounds checking
        # (numba.config.BOUNDSCHECK is off), so an undersized factor
        # table is silently out of bounds rather than an error:
        # measured |table - wigner_d_table| of 3.4e140 at bandwidth 6
        # with the factors of bandwidth 3, and no exception raised
        bandwidth = 6
        factors = list(_wigner.wigner_d_table_factors(bandwidth))
        factors[index] = _wigner.wigner_d_table_factors(3)[index]
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(bandwidth, 0.3, False, *factors)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_pre_table_rejects_single_precision_factors(self, index):
        # a float32 factor table specialises the kernel silently and
        # costs 5.4e-8 against wigner_d_table, which the bitwise
        # assertions of this file would never see
        bandwidth = 8
        factors = list(_wigner.wigner_d_table_factors(bandwidth))
        factors[index] = factors[index].astype(np.float32)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(bandwidth, 0.3, False, *factors)

    @pytest.mark.parametrize("cos_beta", [1.5, -1.5, 2.0])
    def test_a_cosine_outside_the_unit_interval_raises(self, cos_beta):
        with pytest.raises(ValueError):
            _wigner.wigner_d_table(8, cos_beta, False)
        factors = _wigner.wigner_d_table_factors(8)
        with pytest.raises(ValueError):
            _wigner.wigner_d_table_pre(8, cos_beta, False, *factors)


class TestFactorTables:
    def test_e_km_is_the_scaled_half_pi_seed(self):
        bandwidth = 20
        e_km, _, _ = _wigner.wigner_d_table_factors(bandwidth)
        assert e_km.shape == (bandwidth, bandwidth)
        for k in range(bandwidth):
            for m in range(k + 1):
                # d^k_{k,m}(pi/2) = 2^-k e_km, and the power of two
                # is exact, so this is an equality
                expected = _wigner.wigner_d_half_pi(k, k, m) * 2.0**k
                assert e_km[k, m] == expected, (k, m)

    def test_w_and_b_match_their_closed_forms(self):
        bandwidth = 15
        _, w_jkm, b_jkm = _wigner.wigner_d_table_factors(bandwidth)
        assert w_jkm.shape == (bandwidth,) * 3
        assert b_jkm.shape == (bandwidth,) * 3
        for k in range(bandwidth):
            for m in range(k + 1):
                for i in range(k + 2, bandwidth):
                    w = 1.0 / (
                        math.sqrt((i + k) * (i - k) * (i + m) * (i - m)) * (i - 1)
                    )
                    b = w * (
                        math.sqrt((i + k - 1) * (i - k - 1) * (i + m - 1) * (i - m - 1))
                        * i
                    )
                    assert w_jkm[k, m, i] == w, (k, m, i)
                    assert b_jkm[k, m, i] == b, (k, m, i)

    def test_undefined_factor_slots_are_nan(self):
        bandwidth = 8
        e_km, w_jkm, b_jkm = _wigner.wigner_d_table_factors(bandwidth)
        for k in range(bandwidth):
            for m in range(bandwidth):
                assert np.isnan(e_km[k, m]) == (m > k), (k, m)
                for i in range(bandwidth):
                    undefined = m > k or i < k + 2
                    assert np.isnan(w_jkm[k, m, i]) == undefined, (k, m, i)
                    assert np.isnan(b_jkm[k, m, i]) == undefined, (k, m, i)


class TestHalfPiTable:
    """Port of ``testTables()`` lines 515-552."""

    @pytest.mark.parametrize("bandwidth", TABLE_BANDWIDTHS)
    @pytest.mark.parametrize("transpose", [False, True])
    def test_bitwise_equal_to_the_scalar_function(self, bandwidth, transpose):
        table = _wigner.wigner_d_half_pi_table(bandwidth, transpose)
        assert table.shape == (bandwidth,) * 3
        assert table.dtype == np.float64
        assert table.flags.c_contiguous
        for k in range(bandwidth):
            for m in range(bandwidth):
                for j in range(max(k, m), bandwidth):
                    expected = _wigner.wigner_d_half_pi(j, k, m)
                    got = table[m, k, j] if transpose else table[k, m, j]
                    assert got == expected, (j, k, m, transpose)

    def test_the_transposed_table_is_the_transpose(self):
        bandwidth = 15
        plain = _wigner.wigner_d_half_pi_table(bandwidth, False)
        transposed = _wigner.wigner_d_half_pi_table(bandwidth, True)
        assert np.array_equal(transposed, plain.transpose(1, 0, 2), equal_nan=True)

    def test_the_order_swap_sign(self):
        bandwidth = 15
        table = _wigner.wigner_d_half_pi_table(bandwidth, False)
        for k in range(bandwidth):
            for m in range(bandwidth):
                for j in range(max(k, m), bandwidth):
                    sign = 1 if (k - m) % 2 == 0 else -1
                    assert table[k, m, j] == sign * table[m, k, j], (j, k, m)

    @pytest.mark.parametrize("transpose", [False, True])
    def test_nan_exactly_on_the_undefined_slots(self, transpose):
        bandwidth = 15
        table = _wigner.wigner_d_half_pi_table(bandwidth, transpose)
        mask = _defined_mask(bandwidth)
        if transpose:
            mask = mask.transpose(1, 0, 2)
        assert np.array_equal(np.isnan(table), ~mask)

    def test_agrees_with_the_general_table_at_cosine_zero(self, record_property):
        # not bitwise: the pi/2 overload seeds with 2^-k e_km while
        # the general one seeds with the two half angle powers
        bandwidth = 15
        half_pi = _wigner.wigner_d_half_pi_table(bandwidth, False)
        general = _wigner.wigner_d_table(bandwidth, 0.0, False)
        mask = _defined_mask(bandwidth)
        delta = np.abs(half_pi[mask] - general[..., 0][mask]).max()
        record_property("half_pi_vs_general", f"{delta:.3e}")
        assert delta < 1e-15

    def test_the_two_general_slots_coincide_at_cosine_zero(self):
        # pi - pi/2 == pi/2
        bandwidth = 15
        general = _wigner.wigner_d_table(bandwidth, 0.0, False)
        mask = _defined_mask(bandwidth)
        assert np.array_equal(general[..., 0][mask], general[..., 1][mask])


class TestRotateHarmonics:
    bandwidths = [8, 16, 32]

    @staticmethod
    def _zyz(seed=1):
        return _random_zyz(1, seed)[0]

    @pytest.mark.parametrize("bandwidth", bandwidths)
    def test_identity_rotation_returns_the_input(self, bandwidth):
        alm = _random_alm(bandwidth)
        assert (
            np.abs(_wigner.rotate_harmonics(alm, (0.0, 0.0, 0.0)) - alm).max() < 1e-14
        )

    @pytest.mark.parametrize("beta", [0.0, -0.0])
    def test_pure_z_rotation_is_a_phase(self, beta):
        bandwidth = 16
        alm = _random_alm(bandwidth)
        alpha, gamma = 0.7, -1.2
        got = _wigner.rotate_harmonics(alm, (alpha, beta, gamma))
        order = np.arange(bandwidth)[:, np.newaxis]
        expected = alm * np.exp(1j * order * (alpha + gamma))
        assert np.abs(got - expected).max() < 1e-14

    def test_the_glide_identity(self):
        alm = _random_alm(16)
        alpha, beta, gamma = self._zyz()
        first = _wigner.rotate_harmonics(alm, (alpha, beta, gamma))
        second = _wigner.rotate_harmonics(
            alm, (alpha + math.pi, -beta, gamma + math.pi)
        )
        assert np.abs(first - second).max() < 1e-13

    def test_beta_is_wrapped_into_the_principal_interval(self):
        # the recorded deviation from rotateHarmonics(), which does
        # not wrap and differs by 2.2 between these two calls
        alm = _random_alm(16)
        alpha, gamma = 0.7, -1.2
        base = _wigner.rotate_harmonics(alm, (alpha, 1.1, gamma))
        shifted = _wigner.rotate_harmonics(alm, (alpha, 1.1 + TWO_PI, gamma))
        assert np.abs(base - shifted).max() < 1e-13
        four = _wigner.rotate_harmonics(alm, (alpha, 4.0, gamma))
        wrapped = _wigner.rotate_harmonics(alm, (alpha, 4.0 - TWO_PI, gamma))
        assert np.abs(four - wrapped).max() < 1e-13

    def test_the_inverse_rotation(self):
        alm = _random_alm(16)
        alpha, beta, gamma = self._zyz()
        rotated = _wigner.rotate_harmonics(alm, (alpha, beta, gamma))
        back = _wigner.rotate_harmonics(rotated, (-gamma, -beta, -alpha))
        assert np.abs(back - alm).max() < 1e-13

    def test_the_inverse_through_the_conjugate_quaternion(self):
        alm = _random_alm(16)
        zyz = self._zyz()
        rotated = _wigner.rotate_harmonics(alm, zyz)
        quaternion = Rotation(_euler.zyz_to_quaternion(zyz))
        back = _wigner.rotate_harmonics(
            rotated, _euler.quaternion_to_zyz((~quaternion).data)[0]
        )
        assert np.abs(back - alm).max() < 1e-13

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_composition_puts_the_later_rotation_on_the_left(self, seed):
        alm = _random_alm(16)
        first, second = _random_zyz(2, seed)
        stepwise = _wigner.rotate_harmonics(
            _wigner.rotate_harmonics(alm, first), second
        )
        q1 = Rotation(_euler.zyz_to_quaternion(first))
        q2 = Rotation(_euler.zyz_to_quaternion(second))
        combined = _wigner.rotate_harmonics(
            alm, _euler.quaternion_to_zyz((q2 * q1).data)[0]
        )
        assert np.abs(stepwise - combined).max() < 1e-12
        wrong = _wigner.rotate_harmonics(
            alm, _euler.quaternion_to_zyz((q1 * q2).data)[0]
        )
        assert np.abs(stepwise - wrong).max() > 0.1

    def test_per_degree_power_is_conserved(self):
        alm = _random_alm(16)
        rotated = _wigner.rotate_harmonics(alm, self._zyz())

        def power(array):
            magnitude = np.abs(array) ** 2
            return magnitude[0] + 2 * magnitude[1:].sum(axis=0)

        # atol=0: the per-degree powers here span 0.075 to 25.2, so
        # np.allclose' default atol=1e-8 would swallow a 1e-9
        # perturbation of every one of them
        assert np.allclose(power(rotated), power(alm), rtol=1e-12, atol=0)

    def test_the_zero_order_stays_real(self):
        alm = _random_alm(16)
        rotated = _wigner.rotate_harmonics(alm, self._zyz())
        assert np.abs(rotated[0, :].imag).max() < 1e-14

    def test_entries_below_the_diagonal_are_ignored_and_zeroed(self):
        bandwidth = 12
        alm = _random_alm(bandwidth)
        poisoned = alm.copy()
        for order in range(bandwidth):
            for degree in range(order):
                poisoned[order, degree] = 1e6 + 1e6j
        zyz = self._zyz()
        assert np.array_equal(
            _wigner.rotate_harmonics(alm, zyz), _wigner.rotate_harmonics(poisoned, zyz)
        )
        rotated = _wigner.rotate_harmonics(poisoned, zyz)
        for order in range(bandwidth):
            for degree in range(order):
                assert rotated[order, degree] == 0

    def test_the_input_is_not_modified(self):
        alm = _random_alm(8)
        copy = alm.copy()
        rotated = _wigner.rotate_harmonics(alm, self._zyz())
        assert rotated is not alm
        assert np.array_equal(alm, copy)

    @pytest.mark.parametrize("alm_shape", [(8,), (8, 9), (2, 8, 8)])
    def test_a_wrong_alm_shape_raises(self, alm_shape):
        with pytest.raises(ValueError):
            _wigner.rotate_harmonics(
                np.zeros(alm_shape, dtype=np.complex128), (0, 0, 0)
            )

    @pytest.mark.parametrize("zyz", [(0.0, 0.0), (0.0, 0.0, 0.0, 0.0)])
    def test_a_wrong_zyz_shape_raises(self, zyz):
        with pytest.raises(ValueError):
            _wigner.rotate_harmonics(np.zeros((4, 4), dtype=np.complex128), zyz)


class TestRotateHarmonicsOracles:
    """The two assertions which pin the *direction* of the rotation.

    Every group theoretic identity above survives the transposed
    ``dBeta[n, m, j]`` read of the rotation kernel, because that
    mistake is conjugation by the two fold about z, an inner
    automorphism. Only an oracle which fixes ``D`` itself kills it.
    """

    zyz_cases = [
        (0.7, 1.1, -2.3),
        (-2.0, -0.6, 1.3),
        (0.4, math.pi, -1.0),
        (2.5, 0.0, 0.3),
    ]

    @pytest.mark.parametrize("bandwidth", [6, 8])
    @pytest.mark.parametrize("zyz", zyz_cases)
    def test_brute_force_wigner_d_sum(self, bandwidth, zyz, record_property):
        # dependency free: NumPy plus the module's own wigner_D,
        # which is pinned to Mathematica at 2 eps, so this runs on
        # the CI job whose SciPy is too old for sph_harm_y
        alm = _random_alm(bandwidth)
        rotated = _wigner.rotate_harmonics(alm, zyz)
        expected = _brute_force_rotate(alm, zyz)
        delta = np.abs(rotated - expected).max()
        record_property(f"brute_force_bw{bandwidth}_{zyz}", f"{delta:.3e}")
        assert delta <= 1e-13

    @pytest.mark.parametrize("bandwidth", [6, 8])
    @pytest.mark.parametrize("zyz", zyz_cases[:3])
    def test_the_transposed_brute_force_sum_is_wrong(self, bandwidth, zyz):
        # at beta == 0 the two coincide, which is why the fourth
        # case is not a guard
        alm = _random_alm(bandwidth)
        rotated = _wigner.rotate_harmonics(alm, zyz)
        transposed = _brute_force_rotate(alm, zyz, transpose=True)
        assert np.abs(rotated - transposed).max() > 0.1

    @staticmethod
    def _random_directions(n=200, seed=5):
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(n, 3))
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    @pytest.mark.parametrize("zyz", zyz_cases[:3])
    def test_direction_against_sph_harm_y(self, zyz, record_property):
        sph_harm_y = _sph_harm_y()
        bandwidth = 16
        alm = _random_alm(bandwidth)
        directions = self._random_directions()
        rotation = Rotation(_euler.zyz_to_quaternion(zyz))
        rotated = _wigner.rotate_harmonics(alm, zyz)

        got = _evaluate(sph_harm_y, rotated, directions)
        inverse = ((~rotation) * Vector3d(directions)).data
        expected = _evaluate(sph_harm_y, alm, inverse)
        delta = np.abs(got - expected).max()
        record_property(f"sph_harm_y_direction_{zyz}", f"{delta:.3e}")
        # measured 5.3e-14, 2.9e-13 and 1.6e-13 for the three cases,
        # so the thinnest margin on this bound is 3.4x; most of it is
        # the pointwise sph_harm_y/orix evaluation itself
        assert delta < 1e-12

        # the same thing written with the active ZYZ matrix
        matrix_form = _evaluate(sph_harm_y, alm, directions @ rotation.to_matrix()[0])
        assert np.abs(got - matrix_form).max() < 1e-12

    @pytest.mark.parametrize("zyz", zyz_cases[:2])
    def test_the_opposite_direction_is_wrong(self, zyz):
        # zyz_cases[2] = (0.4, pi, -1.0) is a rotation by pi, whose
        # quaternion has w exactly 0.0, so it is an involution and
        # R * n equals (~R) * n bit for bit (measured 0.0, against
        # 1.90 and 1.57 for the two cases kept here). The guard is
        # therefore vacuous there -- the same structural reason the
        # transposed brute force guard drops beta = 0
        sph_harm_y = _sph_harm_y()
        alm = _random_alm(16)
        directions = self._random_directions()
        rotation = Rotation(_euler.zyz_to_quaternion(zyz))
        got = _evaluate(sph_harm_y, _wigner.rotate_harmonics(alm, zyz), directions)
        wrong = _evaluate(sph_harm_y, alm, (rotation * Vector3d(directions)).data)
        assert np.abs(got - wrong).max() > 1.0

    @pytest.mark.parametrize("zyz", zyz_cases[:3])
    def test_direction_through_the_phase_one_transform(self, zyz, record_property):
        # Phase 1's Lambert synthesize writes order m on ring y only
        # for m < min(bw, 4 y + 1), so rings 1-3 are not pointwise
        # evaluations of the bw-16 series (9.8e-4 / 6.5e-7 / 2.3e-10)
        # and the comparison is masked to ring_number >= 4, the same
        # rule Phase 1's own synthesize oracle adopted
        sph_harm_y = _sph_harm_y()
        content_bandwidth = 16
        transform = _sht.SphericalHarmonicTransform(32, "lambert")
        assert transform.dim == 65
        alm = np.zeros((32, 32), dtype=np.complex128)
        alm[:content_bandwidth, :content_bandwidth] = _random_alm(content_bandwidth)

        north, south = transform.synthesize(_wigner.rotate_harmonics(alm, zyz))
        normals = _grid.normals(65, "lambert")
        southern = normals.copy()
        southern[..., 2] = -southern[..., 2]
        rotation = Rotation(_euler.zyz_to_quaternion(zyz))
        mask = _grid.ring_number(65) >= 4

        worst = 0.0
        for got, directions in ((north, normals), (south, southern)):
            flat = directions.reshape(-1, 3)
            inverse = ((~rotation) * Vector3d(flat)).data.reshape(65, 65, 3)
            expected = _evaluate(sph_harm_y, alm, inverse)
            worst = max(worst, np.abs(got[mask] - expected[mask]).max())
        record_property(f"through_transform_{zyz}", f"{worst:.3e}")
        assert worst < 1e-11

    @pytest.mark.parametrize("zyz", zyz_cases[:2])
    def test_the_opposite_direction_is_wrong_through_the_transform(self, zyz):
        # dropped for zyz_cases[2] for the involution reason given on
        # test_the_opposite_direction_is_wrong above
        sph_harm_y = _sph_harm_y()
        content_bandwidth = 16
        transform = _sht.SphericalHarmonicTransform(32, "lambert")
        alm = np.zeros((32, 32), dtype=np.complex128)
        alm[:content_bandwidth, :content_bandwidth] = _random_alm(content_bandwidth)
        north, _ = transform.synthesize(_wigner.rotate_harmonics(alm, zyz))
        normals = _grid.normals(65, "lambert")
        rotation = Rotation(_euler.zyz_to_quaternion(zyz))
        mask = _grid.ring_number(65) >= 4
        flat = normals.reshape(-1, 3)
        wrong = _evaluate(
            sph_harm_y, alm, (rotation * Vector3d(flat)).data.reshape(65, 65, 3)
        )
        assert np.abs(north[mask] - wrong[mask]).max() > 1.0

    @pytest.mark.weekly
    @pytest.mark.parametrize("zyz", zyz_cases[:3])
    def test_round_trip_through_the_phase_one_transform(self, zyz, record_property):
        # no sph_harm_y and no mask: analyze inverts the same per
        # ring truncation synthesize applies
        content_bandwidth = 16
        transform = _sht.SphericalHarmonicTransform(32, "lambert")
        alm = np.zeros((32, 32), dtype=np.complex128)
        alm[:content_bandwidth, :content_bandwidth] = _random_alm(content_bandwidth)
        rotated = _wigner.rotate_harmonics(alm, zyz)
        back = transform.analyze(*transform.synthesize(rotated))
        delta = np.abs(back - rotated).max()
        record_property(f"round_trip_{zyz}", f"{delta:.3e}")
        assert delta < 1e-11


class TestNickelMasterPattern:
    """The shipped Ni master pattern (m-3m, north == south) is exactly
    invariant under the eight square symmetries of the Lambert grid,
    so its harmonics must be invariant under the corresponding
    rotations. The three fold about [111] is not a grid symmetry and
    is an aliasing limited discrimination, not a precision test.
    """

    @staticmethod
    def _relative_change(alm, zyz):
        rotated = _wigner.rotate_harmonics(alm, zyz)
        return np.linalg.norm(rotated - alm) / np.linalg.norm(alm)

    @staticmethod
    def _zyz_of(axis, angle):
        rotation = Rotation.from_axes_angles(Vector3d(axis), angle)
        return _euler.quaternion_to_zyz(rotation.data)[0]

    @pytest.mark.parametrize("zyz", [(0.0, 0.0, math.pi / 2), (math.pi / 2, 0.0, 0.0)])
    def test_four_fold_about_z_is_a_symmetry(self, nickel_alm, zyz, record_property):
        change = self._relative_change(nickel_alm, zyz)
        record_property(f"ni_four_fold_z_{zyz}", f"{change:.3e}")
        assert change < 1e-12

    def test_two_fold_about_x_is_a_symmetry(self, nickel_alm, record_property):
        # the beta = pi branch, which exercises table slot 1 at t = -1
        zyz = self._zyz_of([1, 0, 0], math.pi)
        assert zyz[1] == pytest.approx(math.pi, abs=1e-15)
        change = self._relative_change(nickel_alm, zyz)
        record_property("ni_two_fold_x", f"{change:.3e}")
        assert change < 1e-12

    def test_two_fold_about_110_is_a_symmetry(self, nickel_alm, record_property):
        zyz = self._zyz_of([1, 1, 0], math.pi)
        change = self._relative_change(nickel_alm, zyz)
        record_property("ni_two_fold_110", f"{change:.3e}")
        assert change < 1e-12

    def test_three_fold_about_111_is_discriminated_from_a_non_symmetry(
        self, nickel_alm, record_property
    ):
        # the three fold is a crystal symmetry but not a grid
        # symmetry, so the aliased spectrum of the uint8 image moves
        # a little (7.9e-2); a 90 degree rotation about the same axis
        # is not a symmetry at all and moves much more (3.9e-1)
        three_fold = self._relative_change(
            nickel_alm, self._zyz_of([1, 1, 1], 2 * math.pi / 3)
        )
        control = self._relative_change(
            nickel_alm, self._zyz_of([1, 1, 1], math.pi / 2)
        )
        record_property("ni_three_fold_111", f"{three_fold:.3e}")
        record_property("ni_ninety_degrees_111", f"{control:.3e}")
        assert three_fold < 0.2
        assert control > 0.3


class TestDerivatives:
    """Port of ``testDerivatives()``,
    ``EMSphInx/test/sht/wigner.cpp`` lines 563-868, at its own
    ``24 eps`` tolerance.
    """

    tolerance = 24 * EPS

    @staticmethod
    def _cases():
        prime_pi_3 = ref.as_array(ref.D_PRIME_PI_3)
        prime_2pi_3 = ref.as_array(ref.D_PRIME_2PI_3)
        prime2_pi_3 = ref.as_array(ref.D_PRIME2_PI_3)
        prime2_2pi_3 = ref.as_array(ref.D_PRIME2_2PI_3)
        return [
            ("d' pi/3", _wigner.wigner_d_prime, 0.5, False, prime_pi_3, 0),
            ("d' 2pi/3", _wigner.wigner_d_prime, -0.5, False, prime_2pi_3, 0),
            ("d'' pi/3", _wigner.wigner_d_prime2, 0.5, False, prime2_pi_3, 0),
            ("d'' 2pi/3", _wigner.wigner_d_prime2, -0.5, False, prime2_2pi_3, 0),
            ("d' -pi/3", _wigner.wigner_d_prime, 0.5, True, prime_pi_3, 1),
            ("d' -2pi/3", _wigner.wigner_d_prime, -0.5, True, prime_2pi_3, 1),
            ("d'' -pi/3", _wigner.wigner_d_prime2, 0.5, True, prime2_pi_3, 2),
            ("d'' -2pi/3", _wigner.wigner_d_prime2, -0.5, True, prime2_2pi_3, 2),
        ]

    @pytest.mark.parametrize("case", range(8))
    def test_derivatives_match_mathematica(self, case, record_property):
        # sign mode 0 keeps the table value, 1 multiplies by
        # (-1)^(|k| + |m| + 1) and 2 by (-1)^(|k| + |m|), i.e. the
        # "neg" and "-neg" of wigner.cpp lines 815-824
        label, function, t, negative_beta, expected, mode = self._cases()[case]
        worst = 0.0
        for j in range(ref.NUM):
            for k in range(-4, 5):
                for m in range(-4, 5):
                    want = expected[j, k + 4, m + 4]
                    if mode == 1:
                        want = want * (1 if (abs(k) + abs(m) + 1) % 2 == 0 else -1)
                    elif mode == 2:
                        want = want * (1 if (abs(k) + abs(m)) % 2 == 0 else -1)
                    got = function(j, k, m, t, negative_beta)
                    if np.isnan(got) and np.isnan(want):
                        continue
                    assert not np.isnan(got), (label, j, k, m)
                    # without this, a finite value on an undefined
                    # slot would give max(worst, nan) == worst
                    assert not np.isnan(want), (label, j, k, m)
                    worst = max(worst, abs(got - want))
        record_property(f"derivative_worst_{label}", f"{worst:.3e}")
        assert worst <= self.tolerance

    @pytest.mark.parametrize(
        "j, m, expected",
        [
            (0, 0, 0.0),
            (1, 0, -0.6123724356957947),
            (2, 1, -1.299038105676658),
            (3, 3, 0.421875),
        ],
    )
    def test_second_derivative_is_finite_where_k_equals_j(self, j, m, expected):
        # the C++ evaluates d2Coef = rjk * sqrt((j-k-1)(j+k+2))
        # unconditionally (wigner.hpp line 845) and its radicand is
        # -2, -4, -6, -8 on exactly these slots; math.sqrt raises
        # there, so the guarded port must evaluate the product only
        # inside the k + 1 < j branch
        assert hasattr(_wigner.wigner_d_prime2, "py_func"), (
            "kernel must be @njit-decorated"
        )
        got = _wigner.wigner_d_prime2(j, j, m, 0.5, False)
        assert np.isfinite(got)
        assert got == pytest.approx(expected, abs=self.tolerance)
        interpreted = _py_func(_wigner.wigner_d_prime2)(j, j, m, 0.5, False)
        assert np.isfinite(interpreted)
        assert interpreted == pytest.approx(expected, abs=self.tolerance)

    def test_first_derivative_py_func_is_finite_at_the_origin(self):
        assert hasattr(_wigner.wigner_d_prime, "py_func"), (
            "kernel must be @njit-decorated"
        )
        assert np.isfinite(_py_func(_wigner.wigner_d_prime)(0, 0, 0, 0.5, False))

    @pytest.mark.parametrize("beta", [BETA_TEST, -BETA_TEST, 2.5, -2.5])
    def test_five_point_finite_differences(self, beta, record_property):
        h = 1e-3
        worst_first = 0.0
        worst_second = 0.0
        for j in range(15):
            for k in range(-j, j + 1):
                for m in range(-j, j + 1):
                    values = [
                        _d_of_beta(j, k, m, beta + i * h) for i in (-2, -1, 0, 1, 2)
                    ]
                    first = (values[0] - 8 * values[1] + 8 * values[3] - values[4]) / (
                        12 * h
                    )
                    second = (
                        -values[0]
                        + 16 * values[1]
                        - 30 * values[2]
                        + 16 * values[3]
                        - values[4]
                    ) / (12 * h * h)
                    t = math.cos(beta)
                    negative_beta = math.copysign(1.0, beta) < 0
                    worst_first = max(
                        worst_first,
                        abs(first - _wigner.wigner_d_prime(j, k, m, t, negative_beta)),
                    )
                    worst_second = max(
                        worst_second,
                        abs(
                            second - _wigner.wigner_d_prime2(j, k, m, t, negative_beta)
                        ),
                    )
        record_property(f"finite_difference_first_beta{beta}", f"{worst_first:.3e}")
        record_property(f"finite_difference_second_beta{beta}", f"{worst_second:.3e}")
        assert worst_first <= 1e-7
        assert worst_second <= 1e-6

    @pytest.mark.parametrize("beta", [BETA_TEST, -BETA_TEST, 2.5, -2.5])
    def test_phase_seven_table_formulas_are_pinned(self, beta, record_property):
        # transcription of sht_xcorr.hpp lines 1009-1041, which
        # Phase 7 will inline; pinned here against the scalar ports
        bandwidth = 15
        t = math.cos(beta)
        negative_beta = math.copysign(1.0, beta) < 0
        table = _wigner.wigner_d_table(bandwidth, t, negative_beta)
        worst = [0.0, 0.0, 0.0, 0.0]
        for m in range(bandwidth):
            for n in range(bandwidth):
                for j in range(max(m, n), bandwidth):
                    d1p, d1n, d2p, d2n = _phase7_derivatives(
                        table, m, n, j, t, negative_beta
                    )
                    sign = 1 if (j + m) % 2 == 0 else -1
                    expected = (
                        _wigner.wigner_d_prime(j, m, n, t, negative_beta),
                        sign * _wigner.wigner_d_prime(j, m, -n, t, negative_beta),
                        _wigner.wigner_d_prime2(j, m, n, t, negative_beta),
                        sign * _wigner.wigner_d_prime2(j, m, -n, t, negative_beta),
                    )
                    for index, (got, want) in enumerate(
                        zip((d1p, d1n, d2p, d2n), expected)
                    ):
                        worst[index] = max(worst[index], abs(got - want))
        record_property(
            f"phase7_beta{beta}",
            "d1P {:.3e} d1N {:.3e} d2P {:.3e} d2N {:.3e}".format(*worst),
        )
        assert max(worst) <= 1e-12


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # the flag test and the py_func tests are parametrised over
        # the literal list above, so a kernel added during the
        # implementation would silently escape both of them
        assert _njit_kernel_names(_wigner) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _wigner"
        )

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        # Dropping either option leaves every other test passing, so
        # the private Numba attributes are read directly
        kernel = getattr(_wigner, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    @pytest.mark.parametrize(
        "name, args",
        [
            ("_u_jkm_0", (7, 3, 2, 0.3)),
            ("_u_jkm_1", (7, 3, 2)),
            ("_u_jkm_2", (7, 3, 2, -0.3)),
            ("_v_jkm", (7, 3, 2)),
            ("_w_jkm", (7, 3, 2)),
            ("_a_jkm_0", (7, 3, 2, 0.3)),
            ("_a_jkm_1", (7, 3, 2)),
            ("_a_jkm_2", (7, 3, 2, -0.3)),
            ("_b_jkm", (7, 3, 2)),
            ("_u_km_0", (3, 2, 0.3)),
            ("_u_km_1", (3, 2)),
            ("_u_km_2", (3, 2, -0.3)),
            ("_a_km_0", (3, 2, 0.3)),
            ("_a_km_1", (3, 2)),
            ("_a_km_2", (3, 2, -0.3)),
            ("_e_km", (9, 4)),
            ("_wigner_d_core", (7, 3, 2, 0.3)),
            ("wigner_d_half_pi", (7, 3, 2)),
            ("wigner_d_sign", (7, -3, 2)),
        ],
    )
    def test_scalar_kernel_py_func_equals_the_compiled_kernel(self, name, args):
        kernel = getattr(_wigner, name)
        assert hasattr(kernel, "py_func"), f"{name} must be @njit-decorated"
        assert kernel(*args) == _py_func(kernel)(*args)

    @pytest.mark.parametrize("name", ["_a_jkm_0_pre", "_a_jkm_2_pre"])
    def test_precomputed_coefficient_py_func_equals_the_compiled_kernel(self, name):
        kernel = getattr(_wigner, name)
        assert hasattr(kernel, "py_func"), f"{name} must be @njit-decorated"
        w = _wigner._w_jkm(7, 3, 2)
        assert kernel(w, 7, 3, 2, 0.3) == _py_func(kernel)(w, 7, 3, 2, 0.3)

    @pytest.mark.parametrize(
        "args",
        [
            (3, 5, 2, 0.3),  # j < k, the NaN guard
            (3, 3, 2, 0.3),  # j == k, the seed alone
            (4, 3, 2, 0.3),  # j == k + 1, the second term
            (7, 3, 2, -0.3),  # t < 0, the type 2 coefficients
            (7, 3, 2, 0.0),  # t == 0, the type 1 coefficients
        ],
    )
    def test_wigner_d_core_py_func_covers_every_branch(self, args):
        # the single row of the parametrize table above enters the
        # t > 0 arm of a full recursion only, so the early returns
        # and the two other coefficient types are never taken in the
        # interpreted path
        kernel = _wigner._wigner_d_core
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        compiled = kernel(*args)
        interpreted = _py_func(kernel)(*args)
        if np.isnan(compiled):
            assert np.isnan(interpreted), args
        else:
            assert compiled == interpreted, args

    @pytest.mark.parametrize("args", [(7, -3, -2), (7, 3, -2), (7, -3, 2), (7, 3, 2)])
    def test_wigner_d_sign_py_func_covers_every_branch(self, args):
        # equations 6, 7, 8 and the identity fall-through
        kernel = _wigner.wigner_d_sign
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        assert kernel(*args) == _py_func(kernel)(*args)

    def test_wigner_d_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner.wigner_d, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(0)
        for _ in range(50):
            j = int(rng.integers(0, 25))
            k = int(rng.integers(-j, j + 1)) if j else 0
            m = int(rng.integers(-j, j + 1)) if j else 0
            t = float(rng.uniform(-1, 1))
            negative_beta = bool(rng.integers(0, 2))
            compiled = _wigner.wigner_d(j, k, m, t, negative_beta)
            interpreted = _py_func(_wigner.wigner_d)(j, k, m, t, negative_beta)
            assert compiled == interpreted, (j, k, m, t, negative_beta)

    def test_wigner_uppercase_d_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner.wigner_D, "py_func"), "kernel must be @njit-decorated"
        zyz = np.array([0.7, 1.1, -2.3])
        assert _wigner.wigner_D(5, 3, 2, zyz) == _py_func(_wigner.wigner_D)(
            5, 3, 2, zyz
        )

    def test_table_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner._wigner_d_table_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth = 15
        t = math.cos(BETA_TEST)
        compiled = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        interpreted = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        _wigner._wigner_d_table_kernel(bandwidth, t, False, compiled)
        _py_func(_wigner._wigner_d_table_kernel)(bandwidth, t, False, interpreted)
        assert np.array_equal(compiled, interpreted, equal_nan=True)

    @pytest.mark.parametrize(
        "name",
        ["_wigner_d_table_kernel", "_wigner_d_table_pre_kernel"],
    )
    def test_table_kernel_py_func_covers_the_other_branches(self, name):
        # the runs above use negative_beta=False and a positive t, so
        # the interpreted path never takes the sign swap of equation
        # 9 nor the "not is_type_0" arms of the two recursions
        kernel = getattr(_wigner, name)
        assert hasattr(kernel, "py_func"), f"{name} must be @njit-decorated"
        bandwidth = 8
        t = math.cos(2.5)  # negative, so is_type_0 is False
        assert t < 0
        shape = (bandwidth, bandwidth, bandwidth, 2)
        compiled = np.full(shape, np.nan)
        interpreted = np.full(shape, np.nan)
        extra = ()
        if name.endswith("pre_kernel"):
            extra = _wigner.wigner_d_table_factors(bandwidth)
        kernel(bandwidth, t, True, compiled, *extra)
        _py_func(kernel)(bandwidth, t, True, interpreted, *extra)
        assert np.array_equal(compiled, interpreted, equal_nan=True)

    def test_factor_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner._wigner_d_table_factors_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth = 15
        arrays = [
            (
                np.full((bandwidth, bandwidth), np.nan),
                np.full((bandwidth,) * 3, np.nan),
                np.full((bandwidth,) * 3, np.nan),
            )
            for _ in range(2)
        ]
        _wigner._wigner_d_table_factors_kernel(bandwidth, *arrays[0])
        _py_func(_wigner._wigner_d_table_factors_kernel)(bandwidth, *arrays[1])
        for compiled, interpreted in zip(*arrays):
            assert np.array_equal(compiled, interpreted, equal_nan=True)

    def test_pre_table_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner._wigner_d_table_pre_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth = 15
        t = math.cos(BETA_TEST)
        factors = _wigner.wigner_d_table_factors(bandwidth)
        compiled = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        interpreted = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        _wigner._wigner_d_table_pre_kernel(bandwidth, t, False, compiled, *factors)
        _py_func(_wigner._wigner_d_table_pre_kernel)(
            bandwidth, t, False, interpreted, *factors
        )
        assert np.array_equal(compiled, interpreted, equal_nan=True)

    @pytest.mark.parametrize("transpose", [False, True])
    def test_half_pi_table_kernel_py_func_equals_the_compiled_kernel(self, transpose):
        assert hasattr(_wigner._wigner_d_half_pi_table_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth = 15
        compiled = np.full((bandwidth,) * 3, np.nan)
        interpreted = np.full((bandwidth,) * 3, np.nan)
        _wigner._wigner_d_half_pi_table_kernel(bandwidth, compiled, transpose)
        _py_func(_wigner._wigner_d_half_pi_table_kernel)(
            bandwidth, interpreted, transpose
        )
        assert np.array_equal(compiled, interpreted, equal_nan=True)

    def test_rotate_harmonics_kernel_py_func_equals_the_compiled_kernel(self):
        assert hasattr(_wigner._rotate_harmonics_kernel, "py_func"), (
            "kernel must be @njit-decorated"
        )
        bandwidth = 8
        alm = _random_alm(bandwidth)
        alpha, beta, gamma = 0.7, 1.1, -2.3
        table = _wigner.wigner_d_table(bandwidth, math.cos(beta), False)
        compiled = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        interpreted = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        _wigner._rotate_harmonics_kernel(alm, alpha, gamma, table, compiled)
        _py_func(_wigner._rotate_harmonics_kernel)(
            alm, alpha, gamma, table, interpreted
        )
        assert np.array_equal(compiled, interpreted)

    def test_derivative_py_funcs_equal_the_compiled_kernels(self):
        rng = np.random.default_rng(0)
        for kernel in (_wigner.wigner_d_prime, _wigner.wigner_d_prime2):
            assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
            for _ in range(50):
                j = int(rng.integers(0, 15))
                k = int(rng.integers(-j, j + 1)) if j else 0
                m = int(rng.integers(-j, j + 1)) if j else 0
                t = float(rng.uniform(-0.99, 0.99))
                negative_beta = bool(rng.integers(0, 2))
                compiled = kernel(j, k, m, t, negative_beta)
                interpreted = _py_func(kernel)(j, k, m, t, negative_beta)
                assert compiled == interpreted or (
                    np.isnan(compiled) and np.isnan(interpreted)
                ), (j, k, m, t, negative_beta)


class TestTimingBaseline:
    @pytest.mark.parametrize(
        "bandwidth",
        [68, 88, 113, pytest.param(158, marks=pytest.mark.weekly)],
    )
    def test_table_timing_and_memory_are_recorded(self, bandwidth, record_property):
        t = math.cos(BETA_TEST)
        _wigner.wigner_d_table(2, t, False)  # warm the Numba cache
        start = time.perf_counter()
        table = _wigner.wigner_d_table(bandwidth, t, False)
        allocating = time.perf_counter() - start
        record_property(f"wigner_d_table_seconds_bw{bandwidth}", f"{allocating:.4f}")
        record_property(f"wigner_d_table_mb_bw{bandwidth}", f"{table.nbytes / 1e6:.1f}")

        factors = _wigner.wigner_d_table_factors(2)
        start = time.perf_counter()
        factors = _wigner.wigner_d_table_factors(bandwidth)
        record_property(
            f"wigner_d_table_factors_seconds_bw{bandwidth}",
            f"{time.perf_counter() - start:.4f}",
        )
        _wigner.wigner_d_table_pre(2, t, False, *_wigner.wigner_d_table_factors(2))
        buffer = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
        start = time.perf_counter()
        _wigner.wigner_d_table_pre(bandwidth, t, False, *factors, out=buffer)
        record_property(
            f"wigner_d_table_pre_seconds_bw{bandwidth}",
            f"{time.perf_counter() - start:.4f}",
        )
        peak = factors[1].nbytes + factors[2].nbytes + buffer.nbytes
        record_property(
            f"wigner_d_table_pre_peak_mb_bw{bandwidth}", f"{peak / 1e6:.1f}"
        )

        _wigner.wigner_d_half_pi_table(2, True)
        start = time.perf_counter()
        half_pi = _wigner.wigner_d_half_pi_table(bandwidth, True)
        record_property(
            f"wigner_d_half_pi_table_seconds_bw{bandwidth}",
            f"{time.perf_counter() - start:.4f}",
        )
        record_property(
            f"wigner_d_half_pi_table_mb_bw{bandwidth}",
            f"{half_pi.nbytes / 1e6:.1f}",
        )
        assert allocating < 60.0

    @pytest.mark.parametrize("bandwidth", [68, 88])
    def test_rotate_harmonics_timing_is_recorded(self, bandwidth, record_property):
        alm = _random_alm(bandwidth)
        zyz = (0.7, 1.1, -2.3)
        _wigner.rotate_harmonics(_random_alm(2), zyz)  # warm the Numba cache
        start = time.perf_counter()
        _wigner.rotate_harmonics(alm, zyz)
        elapsed = time.perf_counter() - start
        record_property(f"rotate_harmonics_seconds_bw{bandwidth}", f"{elapsed:.4f}")
        assert elapsed < 60.0
