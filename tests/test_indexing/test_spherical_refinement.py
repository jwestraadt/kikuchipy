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

"""Tests of the Newton refinement on the sphere.

Covers every named assertion of
``specs/2026-09-02-spherical-refinement/validation.md``, which spans
three modules and is therefore collected here rather than split
across the three coarse-path suites:

- ``_xcorr._derivatives``: the value against an inner-product oracle
  and against an analytic triple sum built from Phase 3's
  ``wigner_d``/``wigner_d_prime``/``wigner_d_prime2``, the Jacobian
  and Hessian against that oracle and against central finite
  differences, the ``beta`` wrap, the pole NaN contract which
  ``refinePeak()`` uses as its degeneracy detector, the ``.py_func``
  parity and the kernel flags (the project's third
  ``error_model="numpy"`` kernel).
- ``_xcorr._refine_peak`` and the two ``refine_zyz`` entries: the
  ported ``sht_xcorr.cpp`` recovery suites at the C++'s own gates,
  the near-degenerate and exact-pole behaviour, the far-start failure
  semantics with the one pinned converged decreaser, the constructed
  saddle and monotone-step tests, the validation which must precede
  the kernel, and the buffer ownership and factor sharing rules.
- ``SphericalIndexer`` and the two signal methods: the ``refine=True``
  default, the per-candidate refinement, the memory model with its
  ``n_correlators`` factor, the information message, ``refine_patterns``
  and ``EBSD.refine_orientation_spherical`` with its compatibility,
  phase-identity and row-alignment contract, and the refined real
  data bounds.

The coarse-path assertions of Phases 4 and 6 stay in
``test_spherical_xcorr.py``, ``test_spherical_indexer.py`` and
``tests/test_signals/test_ebsd_spherical_indexing.py``, where the
calls now pass ``refine=False`` explicitly.
"""

import functools
import inspect
import math
import threading
import time

import dask
import numpy as np
from orix.crystal_map import (
    CrystalMap,
    Phase,
    PhaseList,
    create_coordinate_arrays,
)
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh, _groups
import pytest

import kikuchipy as kp
from kikuchipy.indexing._spherical import (
    _euler,
    _grid,
    _indexer,
    _sht,
    _symmetry,
    _wigner,
    _xcorr,
)
from kikuchipy.indexing._spherical._indexer import SphericalIndexer
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)

# ------------------------- Frozen constants ------------------------- #

# The three gates of ``EMSphInx/test/sht/sht_xcorr.cpp``, in degrees:
# ``eps = cbrt(float eps)`` (line 294) for the symmetry free loop,
# ``epsN = 10 eps`` (line 316) for the symmetry free *normalized*
# loop, and ``sqrt(eps) * 5`` (line 345) for the point group loops --
# which the C++ applies to its normalized point group loop as well
# (lines 371-391), so the wedge suite here is split the same way
CPP_EPS_DEG = float(np.cbrt(np.finfo(np.float32).eps))
CPP_EPS_NORMALIZED_DEG = 10 * CPP_EPS_DEG
CPP_GROUP_DEG = 5 * math.sqrt(CPP_EPS_DEG)

# Point groups of ``sht_xcorr.cpp`` lines 331-340 under their orix
# names
POINT_GROUPS = ["112", "11m", "2/m", "3", "4", "4/m", "6", "6/m"]
GROUPS = {group.name: group for group in _groups}

# The thinned ``runTests`` size list: the padded sizes 54, 57, 60 and
# 64 are kept, the adjacent duplicates 55/56/58/59/62 and the costly
# 158 are dropped and 63 is added (odd ``slP`` 125, the top of the
# C++ point group loop)
DEFAULT_REFINE_SIZES = [53, 54, 57, 60, 63, 64, 68]
WEEKLY_REFINE_SIZES = [88, 113, 123]
REFINE_SIZES = DEFAULT_REFINE_SIZES + [
    pytest.param(bandwidth, marks=pytest.mark.weekly)
    for bandwidth in WEEKLY_REFINE_SIZES
]

# The C++ point group loop range (lines 350 and 373), thinned to an
# odd, an even and an odd-padded side length
GROUP_BANDWIDTHS = [53, 60, 63]

# Measured iteration counts: two in every non-degenerate case, three
# occasionally near a cell edge.  Asserted with slack rather than
# pinned, since the count is the C++ comment's "generally at most 3"
MAX_ITERATIONS_SYNTHETIC = 4

# Measured worst absolute errors of ``_derivatives`` against the two
# analytic oracles, and the frozen bounds, which sit 165 to 676 times
# above them.  **The bounds are frozen for the ``random_alm``
# fixture recipe below**: a fixture which rescales the spectra
# rescales every error linearly, so both oracle tests record the
# value and Hessian scales they saw.  The oracle numbers are the
# ones the fixtures give **including their negated-beta twins**
VALUE_ORACLE_BOUND = 1e-11  # measured 6.05e-14 at ``bw`` 16
JACOBIAN_ORACLE_BOUND = 1e-10  # measured 3.06e-13 at ``bw`` 12
HESSIAN_ORACLE_BOUND = 1e-9  # measured 1.48e-12 at ``bw`` 12
FINITE_DIFFERENCE_STEP = 1e-5
JACOBIAN_DIFFERENCE_BOUND = 1e-5  # measured 2.72e-7, truncation limited
HESSIAN_DIFFERENCE_BOUND = 1e-2  # measured 6.65e-4, truncation limited

# ``.py_func`` parity: not bitwise, since the compiled kernel
# contracts its multiply-adds
PY_FUNC_VALUE_RELATIVE = 1e-12
PY_FUNC_DERIVATIVE_ABSOLUTE = 1e-9

# The three sanctioned ``error_model="numpy"`` kernels of the package
NUMPY_ERROR_MODEL_KERNELS = {
    "_derivatives",
    "_fit_gaussian_1d_kernel",
    "_interpolate_maxima",
}

# The pinned far-start case which *converges* to a stationary point
# below its start: case 32 of the ``default_rng(101)`` sweep at
# ``bw`` 24 (moved 3.642 degrees, un-normalised value -19.936 ->
# -27.293).  It kills any future "refinement can only raise the
# score" claim
FAR_START_SEED = 101
FAR_START_BANDWIDTH = 24
FAR_START_CASE = 32
FAR_START_MOVED_DEG = 3.642
FAR_START_VALUE_BEFORE = -19.936
FAR_START_VALUE_AFTER = -27.293

# The bandwidth of every real data test
NI_BANDWIDTH = 68

# Refined ``nickel_ebsd_small`` bounds at ``bw`` 68 in the default
# configuration (measured median 0.505 / p90 0.601 / p95 0.648 / max
# 0.695 degrees against the stored map, from a coarse 0.599 / 0.838).
# ``all < 1.0`` is the roadmap's own bound; the median pin carries
# the Phase 6 margin convention
SMALL_REFINED_ALL_DEG = 1.0
SMALL_REFINED_MEDIAN_DEG = 0.75

# Refined normalized scores of those nine patterns (measured mean
# 0.5886 over a coarse 0.5701, per-point deltas +0.0108 to +0.0286)
SMALL_REFINED_SCORE_MEAN = 0.589
SMALL_REFINED_SCORE_MEAN_DELTA = 0.005

# The same run with ``normalize=False`` (measured un-normalised mean
# 0.3220 -> 0.3324, coarse [0.2799, 0.3533] -> refined [0.2903,
# 0.3592], deltas +0.0059 to +0.0164, misorientations identical)
SMALL_REFINED_PLAIN_SCORE_MEAN = 0.332

# ``refine_orientation_spherical`` against
# ``spherical_indexing(refine=True)``: equal to tolerance, not
# bitwise, because the stored quaternion hands back the
# glide-equivalent triple (measured 0.0 degrees / 2.92e-14)
EQUIVALENCE_ANGLE_DEG = 1e-4
EQUIVALENCE_SCORE = 1e-10

# ``nickel_ebsd_large`` refined subsets (measured 20-pt 0.478 / max
# 1.115; 165-pt 0.456 / p95 0.913 / max 1.140)
LARGE_REFINED_MEDIAN_DEG = 0.6
LARGE_REFINED_MAX_DEG = 2.0
WEEKLY_REFINED_P95_DEG = 1.2
WEEKLY_REFINED_SCORE_FRACTION = 0.9

# The constitution's hard floor, patterns per second per core, now
# asserted on the refined default path (measured 65-70 at ``bw`` 68)
THROUGHPUT_FLOOR = 2

# ``memory_per_worker_bytes`` at ``bw`` 68: the Phase 6 base of
# 135^2 x 68 x 24 + 135^3 x 8 per correlator plus, on a refining run,
# one ``16 bw^3`` Wigner table **per correlator clone**
MEMORY_BASE_ONE_PHASE = 49_426_200
MEMORY_BASE_TWO_PHASES = 79_169_400
MEMORY_D_BETA_BW68 = 16 * 68**3
MEMORY_REFINED_ONE_PHASE = 54_457_112
MEMORY_REFINED_TWO_PHASES = 89_231_224


# ----------------------------- Helpers ------------------------------ #


def _njit_kernel_names(module):
    """Return the names of the module's own Numba kernels."""
    return sorted(
        name
        for name, value in vars(module).items()
        if type(value).__name__ == "CPUDispatcher"
        and getattr(value, "py_func", None) is not None
        and value.py_func.__module__ == module.__name__
    )


def assert_shares_the_factor_triple(triple, factors):
    """Assert that ``triple`` holds the very same three arrays as
    ``factors``.

    The sharing contract of
    ``kikuchipy.indexing._spherical._xcorr._validated_wigner_d_factors``
    is over the three **arrays**, which is what the 5 MB of memory
    hangs on; its return statement builds a fresh tuple, so an
    ``is`` on the tuple itself would fail against a correct
    implementation and pass against one which copied the arrays.
    """
    assert triple is not None
    assert len(triple) == len(factors) == 3
    assert all(a is b for a, b in zip(triple, factors))


def _spherical_modules():
    """Return every submodule of ``kikuchipy.indexing._spherical``."""
    import importlib
    import pkgutil

    from kikuchipy.indexing import _spherical

    return [
        importlib.import_module(f"{_spherical.__name__}.{info.name}")
        for info in pkgutil.iter_modules(_spherical.__path__)
    ]


def random_alm(bandwidth, rng, n_fold=1, mirror=False):
    """Return a random spectrum of a real function with the given
    symmetry, entries with ``l < m`` zero and the ``m == 0`` row
    real.

    **The recipe is part of the frozen contract** of the two
    ``_derivatives`` oracle tests: it is the ``randomPair()``
    coefficient draw of ``sht_xcorr.cpp``, every defined slot uniform
    in ``[-1, 1]``, and the absolute error bounds those tests assert
    hold for the scales it gives and would have to be re-measured for
    any other draw.
    """
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for order in range(bandwidth):
        if order % n_fold:
            continue
        for degree in range(order, bandwidth):
            if mirror and (degree + order) % 2:
                continue
            real = rng.uniform(-1, 1)
            imaginary = 0.0 if order == 0 else rng.uniform(-1, 1)
            alm[order, degree] = complex(real, imaginary)
    return alm


def random_zyz(rng):
    """Return the ZYZ angles of a random rotation, i.e.
    ``randomRotation()`` of ``sht_xcorr.cpp`` lines 128-134.
    """
    quaternion = rng.uniform(-1, 1, 4)
    quaternion[0] = abs(quaternion[0])
    quaternion /= np.linalg.norm(quaternion)
    return _euler.quaternion_to_zyz(quaternion)


def random_pair_on_grid(bandwidth, n_fold, mirror, rng):
    """Return a Legendre transformer and a band-limited random
    spectrum with the systematic zeros of ``randomPair()``.
    """
    dim = bandwidth + (3 if bandwidth % 2 == 0 else 2)
    transform = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
    north = rng.uniform(-1, 1, (dim, dim))
    if mirror:
        south = north.copy()
    else:
        south = rng.uniform(-1, 1, (dim, dim))
        equator = _grid.ring_number(dim) == _grid.n_rings(dim) - 1
        south[equator] = north[equator]
    flm = transform.analyze(north, south)
    for order in range(bandwidth):
        if order % n_fold:
            flm[order] = 0
    return transform, flm


def wedge_mask(dim):
    """Return the binary wedge mask of ``sht_xcorr.cpp`` lines
    218-232 on a square Legendre grid.
    """
    normals = _grid.legendre_normals(dim)
    theta = np.arctan2(normals[..., 1], normals[..., 0])
    inside = (theta >= math.radians(-30)) & (theta <= math.radians(60))
    north = inside.astype(np.float64)
    south = north * (normals[..., 2] < math.sqrt(0.5))
    return north, south


def masked_case(bandwidth, n_fold, mirror, rng):
    """Return the ingredients of the ``testNCorr()`` recipe."""
    transform, flm = random_pair_on_grid(bandwidth, n_fold, mirror, rng)
    dim = transform.dim
    north, south = transform.synthesize(flm)
    flm2 = transform.analyze(north**2, south**2)
    mask_north, mask_south = wedge_mask(dim)
    mlm = transform.analyze(mask_north, mask_south)
    mask_north, mask_south = transform.synthesize(mlm)
    return transform, flm, flm2, mlm, (mask_north, mask_south)


def masked_pattern(transform, flm, mask, zyz):
    """Return the spectrum of the masked, rotated reference."""
    north, south = transform.synthesize(_wigner.rotate_harmonics(flm, zyz))
    return transform.analyze(north * mask[0], south * mask[1])


def _orientations(zyz, symmetry):
    """Return the sample to crystal orientations of ZYZ triples."""
    angles = np.atleast_2d(np.asarray(zyz, dtype=np.float64))
    rotations = Rotation(_euler.zyz_to_quaternion(angles))
    return Orientation((~rotations).data, symmetry=symmetry)


def misorientation_deg(zyz_a, zyz_b, name="1"):
    """Return the symmetry reduced misorientation in degrees."""
    symmetry = GROUPS[name]
    angles = _orientations(zyz_a, symmetry).angle_with(
        _orientations(zyz_b, symmetry), degrees=True
    )
    return float(np.atleast_1d(angles).ravel()[0])


def inner_product(alm, blm):
    """Return the real inner product of two spectra, i.e. ``4 pi``
    times the mean of the product of the two real functions.
    """
    value = float(np.sum((alm[0] * blm[0].conjugate()).real))
    value += 2 * float(np.sum((alm[1:] * blm[1:].conjugate()).real))
    return value


def _coefficient(alm, degree, order):
    """Return ``a^l_m`` for either sign of the order, using
    ``a^l_(-m) = (-1)^m conj(a^l_m)``.
    """
    if order >= 0:
        return alm[order, degree]
    sign = 1.0 if (-order) % 2 == 0 else -1.0
    return sign * np.conjugate(alm[-order, degree])


def analytic_derivatives(flm, gln, zyz):
    """Return the value, gradient and Hessian from Phase 3's scalar
    ports, an oracle which shares no line of code with the kernel.

    The cross-correlation is
    ``sum_l sum_{m'} sum_{n'} f^l_{m'} conj(g^l_{n'})
    d^l_{n',m'}(beta) exp(i (m' alpha + n' gamma))``, the placement
    Phase 4 pinned against ``wigner_D``, so ``d/dalpha`` brings down
    ``i m'``, ``d/dgamma`` brings down ``i n'`` and ``d/dbeta``
    replaces ``d`` by
    :func:`kikuchipy.indexing._spherical._wigner.wigner_d_prime` and
    then by ``wigner_d_prime2``.  The imaginary parts cancel; the
    caller asserts that they do.
    """
    bandwidth = flm.shape[0]
    alpha, beta, gamma = (float(value) for value in zyz)
    beta = _euler.wrap_beta(beta)
    t = math.cos(beta)
    negative_beta = math.copysign(1.0, beta) < 0.0
    value = 0j
    jacobian = np.zeros(3, dtype=np.complex128)
    hessian = np.zeros((3, 3), dtype=np.complex128)
    for degree in range(bandwidth):
        for order_f in range(-degree, degree + 1):
            f = _coefficient(flm, degree, order_f)
            if f == 0:
                continue
            for order_g in range(-degree, degree + 1):
                g = np.conjugate(_coefficient(gln, degree, order_g))
                if g == 0:
                    continue
                d0 = _wigner.wigner_d(degree, order_g, order_f, t, negative_beta)
                d1 = _wigner.wigner_d_prime(degree, order_g, order_f, t, negative_beta)
                d2 = _wigner.wigner_d_prime2(degree, order_g, order_f, t, negative_beta)
                base = f * g * np.exp(1j * (order_f * alpha + order_g * gamma))
                i_m = 1j * order_f
                i_n = 1j * order_g
                value += base * d0
                jacobian[0] += base * d0 * i_m
                jacobian[1] += base * d1
                jacobian[2] += base * d0 * i_n
                hessian[0, 0] += base * d0 * i_m * i_m
                hessian[0, 1] += base * d1 * i_m
                hessian[0, 2] += base * d0 * i_m * i_n
                hessian[1, 1] += base * d2
                hessian[1, 2] += base * d1 * i_n
                hessian[2, 2] += base * d0 * i_n * i_n
    hessian[1, 0] = hessian[0, 1]
    hessian[2, 0] = hessian[0, 2]
    hessian[2, 1] = hessian[1, 2]
    return value, jacobian, hessian


def _phase7_derivatives(table, m, n, j, t, negative_beta):
    """Return ``(d1P, d1N, d2P, d2N)`` of ``sht_xcorr.hpp``.

    Verbatim transcription of ``Correlator::derivatives()``
    (``EMSphInx/include/sht/sht_xcorr.hpp`` lines 1009-1041), copied
    from ``test_spherical_wigner.py`` as its docstring says Phase 7
    would, so that the kernel inherits a formula this suite pins
    against Phase 3's scalar ports rather than one it defines.
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


class RefineBuffers:
    """The caller-owned buffers a direct ``_derivatives`` or
    ``_refine_peak`` call needs, built the way the correlators build
    them: a NaN filled ``d_beta`` routed once through
    ``wigner_d_table_pre(out=)`` so that the Phase 3 tripwire runs.
    """

    def __init__(self, bandwidth):
        self.bandwidth = int(bandwidth)
        self.side_length = int(_xcorr._fft.fast_size(2 * self.bandwidth - 1))
        self.e_km, self.w_jkm, self.b_jkm = _wigner.wigner_d_table_factors(
            self.bandwidth
        )
        self.d_beta = _wigner.wigner_d_table_pre(
            self.bandwidth,
            1.0,
            False,
            self.e_km,
            self.w_jkm,
            self.b_jkm,
            out=np.full((self.bandwidth,) * 3 + (2,), np.nan),
        )
        self.jac = np.zeros(3)
        self.hes = np.zeros((3, 3))
        self.step = np.zeros(3)

    def derivatives(self, flm, gln, zyz, n_fold=1, mirror=False, der=True):
        """Return the value at ``zyz``, writing ``jac`` and ``hes``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return _xcorr._derivatives(
                flm,
                gln,
                np.asarray(zyz, dtype=np.float64),
                self.jac,
                self.hes,
                self.bandwidth,
                mirror,
                n_fold,
                der,
                self.d_beta,
                self.e_km,
                self.w_jkm,
                self.b_jkm,
            )

    def refine(self, flm, gln, zyz0, n_fold=1, mirror=False, eps=0.01):
        """Return ``(zyz, value, converged)`` of ``_refine_peak``."""
        return _xcorr._refine_peak(
            flm,
            gln,
            np.asarray(zyz0, dtype=np.float64),
            n_fold,
            mirror,
            self.bandwidth,
            self.side_length,
            self.d_beta,
            self.e_km,
            self.w_jkm,
            self.b_jkm,
            self.jac,
            self.hes,
            self.step,
            eps,
        )


class DerivativeSpy:
    """Count the ``_derivatives`` calls of a refinement, so that its
    iteration count can be asserted without widening the frozen
    ``_refine_peak`` return.

    One ``der=True`` call is one Newton iteration; the ``der=False``
    calls are the failure path's analytic value and the normalized
    denominator's two evaluations.

    The counters are taken under a lock: the kernel releases the GIL
    and a chunked run increments them from several dask worker
    threads, where a bare ``+=`` loses increments.
    """

    def __init__(self, monkeypatch):
        original = _xcorr._derivatives
        self.iterations = 0
        self.value_only = 0
        self._lock = threading.Lock()

        def spy(*args, **kwargs):
            der = kwargs["der"] if "der" in kwargs else args[8]
            with self._lock:
                if der:
                    self.iterations += 1
                else:
                    self.value_only += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_xcorr, "_derivatives", spy)


def constant_derivatives(value, jacobian, hessian):
    """Return a ``_derivatives`` stand-in writing fixed arrays.

    The stand-in counts its own calls: ``fake.n_iterations`` is the
    number of ``der=True`` evaluations, i.e. of Newton iterations,
    and ``fake.n_values`` the number of ``der=False`` ones.
    """

    def fake(*args, **kwargs):
        jac = args[3]
        hes = args[4]
        der = kwargs["der"] if "der" in kwargs else args[8]
        if der:
            fake.n_iterations += 1
            jac[:] = jacobian
            hes[:] = hessian
        else:
            fake.n_values += 1
        return value

    fake.n_iterations = 0
    fake.n_values = 0
    return fake


# --------------------------- Real data ------------------------------ #


@functools.lru_cache(maxsize=1)
def _ni_signal_cached():
    """Return the background corrected ``nickel_ebsd_small`` signal."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    return signal


def ni_signal():
    """Return a fresh deep copy of the corrected signal."""
    return _ni_signal_cached().deepcopy()


def ni_detector():
    """Return a fresh Ni detector with one projection centre."""
    detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    detector.pc = detector.pc_average
    return detector


@functools.lru_cache(maxsize=4)
def ni_harmonics(bandwidth=NI_BANDWIDTH):
    """Return the Ni master pattern harmonics built directly at
    ``bandwidth``, cached because the transform costs about a second.
    """
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


def scrambled_harmonics(seed=42):
    """Return the sign-scrambled decoy phase of the Phase 6 suite."""
    harmonics = ni_harmonics()
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], harmonics.alm.shape)
    return MasterPatternHarmonics(
        harmonics.alm * signs, phase=Phase("scrambled", point_group="1")
    )


def ni_indexer(**kwargs):
    """Return a Ni indexer at ``bw`` 68 with the class defaults."""
    return SphericalIndexer(ni_harmonics(), ni_detector(), **kwargs)


def index_ni(signal=None, **kwargs):
    """Return the crystal map of a Ni call, defaults kept."""
    if signal is None:
        signal = ni_signal()
    kwargs.setdefault("verbose", 0)
    harmonics = kwargs.pop("harmonics", ni_harmonics())
    detector = kwargs.pop("detector", ni_detector())
    return signal.spherical_indexing(harmonics, detector, **kwargs)


def stored_rotations():
    """Return the stored orientations of the small map."""
    return _ni_signal_cached().xmap.rotations


def misorientation(rotations, reference):
    """Return the symmetry reduced misorientation in degrees under
    m-3m between two rotation sets.
    """
    angles = Orientation(rotations.data, Oh).angle_with(
        Orientation(reference.data, Oh), degrees=True
    )
    return np.asarray(angles, dtype=np.float64).ravel()


def record_angles(record_property, tag, angles):
    """Record the per-point misorientations and their statistics."""
    record_property(f"{tag}_per_point", ", ".join(f"{a:.3f}" for a in angles))
    record_property(f"{tag}_median", f"{np.median(angles):.4f}")
    record_property(f"{tag}_p95", f"{np.percentile(angles, 95):.4f}")
    record_property(f"{tag}_max", f"{angles.max():.4f}")


def pattern_spectrum(indexer, pattern):
    """Return the harmonic spectrum of one pattern through the
    indexer's own preprocessing, back-projection and analysis, i.e.
    the three stages ``_index_chunk`` walks before it correlates.
    """
    processed = _indexer._preprocess_pattern(
        pattern,
        good_pixels=indexer.good_pixels,
        gaussian_background=indexer.gaussian_background,
        n_regions=indexer.n_regions,
        emsphinx_compatible=indexer.emsphinx_compatible,
    )
    north, south = indexer.projector.unproject(processed)
    return indexer.projector.sht.analyze(north, south)


def analytic_normalized_score(indexer, gln, zyz):
    """Return the normalized score the refinement itself reports at
    ``zyz``, i.e. the analytic correlation there over
    ``denominator(zyz)``.

    This is what a refinement which fails from ``zyz`` hands back,
    and therefore the only score a refined row can be compared
    against when the starting orientation did not come from the
    coarse pipeline.
    """
    correlator = indexer.correlators[0]
    buffers = RefineBuffers(indexer.bandwidth)
    zyz = np.asarray(zyz, dtype=np.float64)
    value = buffers.derivatives(
        correlator.flm, gln, zyz, correlator.n_fold, correlator.mirror, der=False
    )
    return float(value) / correlator._denominator(zyz)


# =================== _derivatives: the kernel (D1/D2) ================ #


class TestDerivatives:
    @pytest.mark.parametrize("der", [True, False])
    def test_value_equals_the_inner_product_oracle(self, der, record_property):
        # the Phase 4 D4 closed form, evaluated with Phase 3's
        # ``rotate_harmonics``: an oracle which shares no line of
        # code with the kernel.  The absolute bound is frozen for
        # the ``random_alm`` fixture, whose scale is recorded
        bandwidth = 16
        rng = np.random.default_rng(7)
        buffers = RefineBuffers(bandwidth)
        worst = 0.0
        scale = 0.0
        for n_fold, mirror in [(1, False), (2, True), (3, False), (4, True)]:
            flm = random_alm(bandwidth, rng, n_fold, mirror)
            gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
            for _ in range(6):
                zyz = random_zyz(rng)
                got = buffers.derivatives(flm, gln, zyz, n_fold, mirror, der=der)
                want = inner_product(_wigner.rotate_harmonics(flm, zyz), gln)
                worst = max(worst, abs(got - want))
                scale = max(scale, abs(want))
        record_property(f"value_oracle_bw16_der{der}", f"{worst:.3e}")
        record_property("value_oracle_bw16_scale", f"{scale:.2f}")
        assert worst <= VALUE_ORACLE_BOUND

    @pytest.mark.weekly
    def test_the_value_oracle_at_a_larger_bandwidth(self, record_property):
        bandwidth = 24
        rng = np.random.default_rng(7)
        buffers = RefineBuffers(bandwidth)
        worst = 0.0
        for n_fold, mirror in [(1, False), (4, True)]:
            flm = random_alm(bandwidth, rng, n_fold, mirror)
            gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
            for _ in range(3):
                zyz = random_zyz(rng)
                got = buffers.derivatives(flm, gln, zyz, n_fold, mirror, der=False)
                want = inner_product(_wigner.rotate_harmonics(flm, zyz), gln)
                worst = max(worst, abs(got - want))
        record_property("value_oracle_bw24", f"{worst:.3e}")
        assert worst <= VALUE_ORACLE_BOUND

    def test_jacobian_and_hessian_equal_the_analytic_oracle(self, record_property):
        # the triple sum over Phase 3's ``wigner_d``,
        # ``wigner_d_prime`` and ``wigner_d_prime2``, which the
        # kernel does not call: it inlines the table based formulas
        # instead, so this is an independent pin of the sign of the
        # gradient, of the Hessian's slot mapping and of the
        # ``csc`` prefactor.  Every drawn triple is evaluated at its
        # **negated beta** as well: ``quaternion_to_zyz`` returns
        # ``beta`` in ``[0, pi]``, so without the twin the sign the
        # kernel gives ``csc`` below the equator -- the one thing a
        # Newton step across a pole depends on -- is never read here
        bandwidth = 12
        rng = np.random.default_rng(7)
        buffers = RefineBuffers(bandwidth)
        worst_value = worst_jac = worst_hes = 0.0
        worst_imaginary = 0.0
        value_scale = hessian_scale = 0.0
        betas = []
        for n_fold, mirror in [(1, False), (2, True)]:
            flm = random_alm(bandwidth, rng, n_fold, mirror)
            gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
            for _ in range(4):
                drawn = np.asarray(random_zyz(rng), dtype=np.float64)
                for zyz in (drawn, drawn * np.array([1.0, -1.0, 1.0])):
                    betas.append(float(zyz[1]))
                    got_value = buffers.derivatives(flm, gln, zyz, n_fold, mirror)
                    got_jac = buffers.jac.copy()
                    got_hes = buffers.hes.copy()
                    value, jacobian, hessian = analytic_derivatives(flm, gln, zyz)
                    worst_imaginary = max(
                        worst_imaginary,
                        abs(value.imag),
                        float(np.abs(jacobian.imag).max()),
                        float(np.abs(hessian.imag).max()),
                    )
                    worst_value = max(worst_value, abs(got_value - value.real))
                    worst_jac = max(
                        worst_jac, float(np.abs(got_jac - jacobian.real).max())
                    )
                    worst_hes = max(
                        worst_hes, float(np.abs(got_hes - hessian.real).max())
                    )
                    value_scale = max(value_scale, abs(got_value))
                    hessian_scale = max(hessian_scale, float(np.abs(got_hes).max()))
        record_property(
            "analytic_oracle_betas", f"{min(betas):+.4f}..{max(betas):+.4f}"
        )
        assert min(betas) < 0.0 < max(betas)
        record_property("analytic_oracle_value", f"{worst_value:.3e}")
        record_property("analytic_oracle_jacobian", f"{worst_jac:.3e}")
        record_property("analytic_oracle_hessian", f"{worst_hes:.3e}")
        record_property(
            "analytic_oracle_scales",
            f"|value| {value_scale:.2f}, |hes| {hessian_scale:.1f}",
        )
        # the correlation of two real functions is real
        assert worst_imaginary <= 1e-11
        assert worst_value <= VALUE_ORACLE_BOUND
        assert worst_jac <= JACOBIAN_ORACLE_BOUND
        assert worst_hes <= HESSIAN_ORACLE_BOUND
        # the Hessian is symmetrised, not merely upper triangular
        assert np.array_equal(buffers.hes, buffers.hes.T)

    def test_jacobian_and_hessian_match_finite_differences(self, record_property):
        # a second oracle which knows nothing about Wigner functions
        # at all, only that the kernel's value is differentiable.
        # As above, every drawn triple is evaluated at its negated
        # ``beta`` too, so the southern ``csc`` sign is differenced
        # rather than assumed
        bandwidth = 16
        step = FINITE_DIFFERENCE_STEP
        rng = np.random.default_rng(19)
        buffers = RefineBuffers(bandwidth)
        worst_jac = worst_hes = 0.0
        betas = []
        for n_fold, mirror in [(1, False), (2, True)]:
            flm = random_alm(bandwidth, rng, n_fold, mirror)
            gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
            for _ in range(3):
                drawn = np.asarray(random_zyz(rng), dtype=np.float64)
                for zyz in (drawn, drawn * np.array([1.0, -1.0, 1.0])):
                    betas.append(float(zyz[1]))
                    value = buffers.derivatives(flm, gln, zyz, n_fold, mirror)
                    jacobian = buffers.jac.copy()
                    hessian = buffers.hes.copy()

                    def evaluate(eu):
                        return buffers.derivatives(
                            flm, gln, eu, n_fold, mirror, der=False
                        )

                    for i in range(3):
                        axis = np.zeros(3)
                        axis[i] = step
                        plus = evaluate(zyz + axis)
                        minus = evaluate(zyz - axis)
                        worst_jac = max(
                            worst_jac, abs((plus - minus) / (2 * step) - jacobian[i])
                        )
                        plus2 = evaluate(zyz + 2 * axis)
                        minus2 = evaluate(zyz - 2 * axis)
                        second = (
                            -plus2 + 16 * plus - 30 * value + 16 * minus - minus2
                        ) / (12 * step * step)
                        worst_hes = max(worst_hes, abs(second - hessian[i, i]))
                    for i in range(3):
                        for k in range(i + 1, 3):
                            first = np.zeros(3)
                            second_axis = np.zeros(3)
                            first[i] = step
                            second_axis[k] = step
                            mixed = (
                                evaluate(zyz + first + second_axis)
                                - evaluate(zyz + first - second_axis)
                                - evaluate(zyz - first + second_axis)
                                + evaluate(zyz - first - second_axis)
                            ) / (4 * step * step)
                            worst_hes = max(worst_hes, abs(mixed - hessian[i, k]))
        record_property(
            "finite_difference_betas", f"{min(betas):+.4f}..{max(betas):+.4f}"
        )
        assert min(betas) < 0.0 < max(betas)
        record_property("finite_difference_jacobian", f"{worst_jac:.3e}")
        record_property("finite_difference_hessian", f"{worst_hes:.3e}")
        assert worst_jac <= JACOBIAN_DIFFERENCE_BOUND
        assert worst_hes <= HESSIAN_DIFFERENCE_BOUND

    @pytest.mark.parametrize("beta", [0.9, -0.9, 2.5, -2.5])
    def test_phase7_formulas_still_pin_the_kernel_inputs(self, beta, record_property):
        # the Phase 3 assertion, re-run against the shipped table
        # builder the kernel itself calls: the four coefficients the
        # kernel inlines agree with the scalar ``wigner_d_prime``
        # and ``wigner_d_prime2`` to 1e-12
        bandwidth = 15
        t = math.cos(beta)
        negative_beta = math.copysign(1.0, beta) < 0.0
        e_km, w_jkm, b_jkm = _wigner.wigner_d_table_factors(bandwidth)
        table = _wigner.wigner_d_table_pre(
            bandwidth, t, negative_beta, e_km, w_jkm, b_jkm
        )
        worst = [0.0, 0.0, 0.0, 0.0]
        for m in range(bandwidth):
            for n in range(bandwidth):
                for j in range(max(m, n), bandwidth):
                    got = _phase7_derivatives(table, m, n, j, t, negative_beta)
                    sign = 1 if (j + m) % 2 == 0 else -1
                    want = (
                        _wigner.wigner_d_prime(j, m, n, t, negative_beta),
                        sign * _wigner.wigner_d_prime(j, m, -n, t, negative_beta),
                        _wigner.wigner_d_prime2(j, m, n, t, negative_beta),
                        sign * _wigner.wigner_d_prime2(j, m, -n, t, negative_beta),
                    )
                    for index, (a, b) in enumerate(zip(got, want)):
                        worst[index] = max(worst[index], abs(a - b))
        record_property(
            f"phase7_formulas_beta{beta}",
            "d1P {:.3e} d1N {:.3e} d2P {:.3e} d2N {:.3e}".format(*worst),
        )
        assert max(worst) <= 1e-12

    @pytest.mark.parametrize("beta", [0.0, -0.0])
    def test_pole_evaluation_produces_the_nan_contract(self, beta, record_property):
        # ``refinePeak()`` detects the ``beta`` degeneracy by a NaN
        # ``hes[1, 1]`` (lines 461 and 468), which only exists
        # because the unguarded ``csc`` divides by zero under the
        # IEEE error model.  Numba's default model would raise
        # instead and break the C++ control flow
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, np.array([0.4, 0.0, -0.7]))
        start = np.array([0.45, beta, -0.75])
        value = buffers.derivatives(flm, gln, start)
        record_property(
            f"pole_beta{beta!r}",
            f"value {value:.6f} jac {np.array2string(buffers.jac)} "
            f"hes11 {buffers.hes[1, 1]!r}",
        )
        assert math.isnan(buffers.hes[1, 1])
        assert math.isnan(buffers.jac[1])
        for finite in (
            buffers.jac[0],
            buffers.jac[2],
            buffers.hes[0, 0],
            buffers.hes[2, 2],
        ):
            assert math.isfinite(finite)
        # and the fallback path runs rather than raising: the start
        # is on the peak, so the 1 x 1 alpha sub-problem converges
        # in a single iteration (measured)
        zyz, refined_value, converged = buffers.refine(flm, gln, start)
        assert zyz.shape == (3,)
        assert converged is True
        record_property(f"pole_beta{beta!r}_refined", f"{float(refined_value):.6f}")

    @pytest.mark.parametrize("beta", [math.pi, -math.pi])
    def test_the_pole_at_pi_is_nan_or_huge(self, beta, record_property):
        # the exact-NaN claim at ``+-pi`` rides on the host libm
        # returning ``cos(+-pi) == -1.0``, which is true here but is
        # not guaranteed: a libm one ulp off would give a finite
        # ``csc`` of about 6.7e7 and a huge finite Hessian, and the
        # ordinary Newton path instead.  The C++ has the identical
        # dependence, so this is parity rather than a defect, and
        # the assertion is weakened accordingly
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, np.array([0.4, 0.0, -0.7]))
        start = np.array([0.45, beta, -0.75])
        buffers.derivatives(flm, gln, start)
        entry = float(buffers.hes[1, 1])
        record_property(
            f"pole_beta_pi_{beta:+.3f}",
            f"cos {math.cos(beta)!r} hes11 {entry!r}",
        )
        assert math.isnan(entry) or abs(entry) > 1e12
        zyz, value, converged = buffers.refine(flm, gln, start)
        assert zyz.shape == (3,)

    @pytest.mark.parametrize("turns", [-1, 1, 2])
    def test_beta_is_wrapped(self, turns):
        # the C++ wraps ``beta`` into ``[-pi, pi]`` before it takes
        # ``cos`` and ``signbit`` (lines 895-899), without which a
        # Newton step across a pole reads the mirrored table
        bandwidth = 16
        rng = np.random.default_rng(5)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        zyz = np.asarray(random_zyz(rng), dtype=np.float64)
        base = buffers.derivatives(flm, gln, zyz, der=False)
        shifted = zyz + np.array([0.0, 2 * math.pi * turns, 0.0])
        other = buffers.derivatives(flm, gln, shifted, der=False)
        assert other == pytest.approx(base, rel=1e-12)

    def test_py_func_matches_the_compiled_kernel(self, record_property):
        # not bitwise: the compiled kernel contracts its
        # multiply-adds.  The inputs are NumPy scalars and the whole
        # comparison runs under ``errstate``, so that the pole
        # evaluation reaches the interpreted branch as well -- which
        # it only can because the ``csc`` chain is written in
        # ``np.cos``/``np.sqrt``: a ``math.*`` twin raises
        # ``ZeroDivisionError`` there whatever ``errstate`` says
        kernel = _xcorr._derivatives
        assert hasattr(kernel, "py_func"), "`_derivatives` must be @njit-decorated"
        bandwidth = 12
        rng = np.random.default_rng(31)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng, 2, True)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        worst_value = worst_derivative = 0.0
        angles = [np.asarray(random_zyz(rng), dtype=np.float64) for _ in range(3)]
        angles.append(np.array([np.float64(0.45), np.float64(0.0), np.float64(-0.75)]))
        for zyz in angles:
            results = []
            for function in (kernel, kernel.py_func):
                jac = np.zeros(3)
                hes = np.zeros((3, 3))
                with np.errstate(divide="ignore", invalid="ignore"):
                    value = function(
                        flm,
                        gln,
                        zyz,
                        jac,
                        hes,
                        bandwidth,
                        True,
                        2,
                        True,
                        buffers.d_beta,
                        buffers.e_km,
                        buffers.w_jkm,
                        buffers.b_jkm,
                    )
                results.append((float(value), jac, hes))
            compiled, interpreted = results
            worst_value = max(
                worst_value,
                abs(compiled[0] - interpreted[0]) / max(1e-30, abs(compiled[0])),
            )
            for a, b in ((compiled[1], interpreted[1]), (compiled[2], interpreted[2])):
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.any():
                    worst_derivative = max(
                        worst_derivative, float(np.abs(a[finite] - b[finite]).max())
                    )
                # A NaN slot in one must be a NaN slot in the other.
                # This is the file's one exact pattern across the
                # compiled and interpreted paths, and it is portable
                # where a bitwise comparison would not be: IEEE 754
                # makes ``0 / 0`` and ``inf - inf`` NaN and every
                # other operation of the chain NaN-propagating, so
                # *which* slots are NaN does not depend on the
                # contraction of a multiply-add or on the rounding
                # of the transcendentals feeding it
                assert np.array_equal(np.isnan(a), np.isnan(b))
        record_property("py_func_value_relative", f"{worst_value:.3e}")
        record_property("py_func_derivative_absolute", f"{worst_derivative:.3e}")
        assert worst_value <= PY_FUNC_VALUE_RELATIVE
        assert worst_derivative <= PY_FUNC_DERIVATIVE_ABSOLUTE

    def test_a_shape_mismatch_never_reaches_the_kernel(self):
        # with bounds checking off a spectrum which disagrees with
        # the kernel's bandwidth is silent garbage rather than an
        # error: the drafting probe fed ``(68, 68)`` spectra to a
        # ``bw`` 88 kernel and read values of 1e225 and NaN
        # Hessians.  Every public entry therefore validates first
        correlator = _xcorr.SphericalCrossCorrelator(24)
        rng = np.random.default_rng(2)
        good = random_alm(24, rng)
        small = random_alm(16, rng)
        zyz = np.zeros(3)
        with pytest.raises(ValueError):
            correlator.refine_zyz(small, good, 1, False, zyz)
        with pytest.raises(ValueError):
            correlator.refine_zyz(good, small, 1, False, zyz)


# ================== Kernel flags and the error model ================= #


class TestRefinementKernels:
    def test_derivatives_is_a_kernel_of_the_module(self):
        assert "_derivatives" in _njit_kernel_names(_xcorr)

    def test_derivatives_is_compiled_with_the_numpy_error_model(self):
        # load bearing: at ``|cos(beta)| == 1`` the unguarded ``csc``
        # must give the IEEE infinity which becomes the NaN
        # ``hes[1, 1]`` the Newton loop reads as its degeneracy flag
        kernel = _xcorr._derivatives
        assert hasattr(kernel, "targetoptions")
        assert kernel.targetoptions.get("error_model") == "numpy"
        assert kernel.targetoptions.get("nogil") is True
        assert type(kernel._cache).__name__ == "FunctionCache"
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    def test_the_package_has_exactly_three_numpy_error_model_kernels(self):
        found = set()
        for module in _spherical_modules():
            for name, value in vars(module).items():
                if type(value).__name__ != "CPUDispatcher":
                    continue
                if getattr(value, "py_func", None) is None:
                    continue
                if value.py_func.__module__ != module.__name__:
                    continue
                if value.targetoptions.get("error_model") == "numpy":
                    found.add(name)
        assert found == NUMPY_ERROR_MODEL_KERNELS

    def test_the_xcorr_module_has_exactly_two_error_model_kernels(self):
        # the module scoped half of the assertion above, which is
        # the one ``test_spherical_back_projection.py`` makes on
        # this module: the coarse interpolation and the refinement,
        # and nothing else.  It lives here so that the surface the
        # refinement adds is policed with the refinement
        found = {
            name
            for name in _njit_kernel_names(_xcorr)
            if getattr(_xcorr, name).targetoptions.get("error_model") == "numpy"
        }
        assert found == {"_interpolate_maxima", "_derivatives"}

    def test_the_newton_loop_is_not_a_kernel(self):
        # decided: a Python loop over two kernels, so that the C++
        # exception control flow maps to statuses and the
        # ``error_model`` surface stays one function wide
        assert "_refine_peak" not in _njit_kernel_names(_xcorr)
        assert not hasattr(_xcorr._refine_peak, "targetoptions")

    def test_the_cholesky_solve_is_imported_and_not_duplicated(self):
        # the C++ calls the same ``solve::cholesky`` from the
        # Gaussian fit and from ``refinePeak``, and that routine's
        # NaN-sensitive comparison directions and kernel flags are
        # pinned in one place.  ``numpy.linalg.solve`` is refused as
        # well: it goes through BLAS, which the Phase 6 ring-weight
        # guard already had to work around
        from kikuchipy.indexing._spherical import _preprocessing

        assert (
            _xcorr._preprocessing._cholesky_solve_3x3
            is _preprocessing._cholesky_solve_3x3
        )
        source = inspect.getsource(_xcorr)
        assert "np.linalg.solve" not in source
        assert "def _cholesky_solve_3x3" not in source
        # and the new edge is one way, so there is no cycle
        assert "_xcorr" not in inspect.getsource(_preprocessing)

    def test_the_newton_loop_makes_no_transform_call(self, monkeypatch):
        # the refinement is a real space evaluation -- rebuild the
        # Wigner table, sum -- and touches neither of the two SciPy
        # transforms this module binds in its own namespace, which
        # is the seam the Phase 4 recording tests patch.  A source
        # text check would not have noticed a transform reached
        # through a collaborator
        bandwidth = 16
        rng = np.random.default_rng(21)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz0, _ = correlator.correlate(flm, gln, 1, False)
        calls = []

        def recorder(name):
            def fake(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f"the refinement called `{name}`")

            return fake

        for name in ("ifft", "irfft"):
            monkeypatch.setattr(_xcorr, name, recorder(name))
        correlator.refine_zyz(flm, gln, 1, False, zyz0)
        assert calls == []


# ================= _refine_peak: the Newton loop (D4) ================ #


class TestRefinePeak:
    def test_an_on_grid_rotation_refines_in_one_iteration(
        self, monkeypatch, record_property
    ):
        # the autocorrelation identity makes the Newton step from an
        # on-grid peak vanish, so the refinement confirms the coarse
        # answer instead of moving it
        bandwidth = 16
        rng = np.random.default_rng(7)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        flm = random_alm(bandwidth, rng)
        zyz0 = _xcorr.index_to_euler((3, 5, 7), correlator.side_length)
        gln = _wigner.rotate_harmonics(flm, zyz0)
        coarse, _ = correlator.correlate(flm, gln, 1, False)
        spy = DerivativeSpy(monkeypatch)
        refined, score = correlator.refine_zyz(flm, gln, 1, False, coarse)
        power = inner_product(flm, flm)
        delta = misorientation_deg(zyz0, refined)
        record_property("on_grid_refined_deg", f"{delta:.3e}")
        record_property("on_grid_iterations", str(spy.iterations))
        assert delta <= 1e-9
        assert spy.iterations == 1
        assert score / power == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("bandwidth", REFINE_SIZES)
    def test_symmetry_free_pairs_are_recovered(
        self, bandwidth, monkeypatch, record_property
    ):
        # the ``runTests`` symmetry free loop of ``sht_xcorr.cpp``,
        # gated at its own ``eps = cbrt(float eps)``
        rng = np.random.default_rng(11 + bandwidth)
        _, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        # one spy for the whole test, snapshotted per rotation: a
        # spy per iteration would wrap the previous one and route
        # every kernel call through a stack of Python wrappers
        spy = DerivativeSpy(monkeypatch)
        seen = 0
        for i in range(3):
            zyz_true = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz_true)
            coarse, _ = correlator.correlate(flm, gln, 1, False)
            before = spy.iterations
            refined, score = correlator.refine_zyz(flm, gln, 1, False, coarse)
            seen = spy.iterations - before
            coarse_delta = misorientation_deg(zyz_true, coarse)
            delta = misorientation_deg(zyz_true, refined)
            record_property(
                f"symmetry_free_bw{bandwidth}_rot{i}",
                f"coarse {coarse_delta:.5f} refined {delta:.3e} deg, "
                f"{seen} iteration(s)",
            )
            assert math.isfinite(score)
            assert delta < CPP_EPS_DEG
            assert delta < coarse_delta
            assert 1 <= seen <= MAX_ITERATIONS_SYNTHETIC

    @pytest.mark.parametrize("name", POINT_GROUPS)
    @pytest.mark.parametrize("bandwidth", GROUP_BANDWIDTHS)
    def test_the_eight_point_groups_are_recovered(
        self, name, bandwidth, record_property
    ):
        # the C++ point group loop, gated at its own loosened
        # ``sqrt(eps) * 5``
        n_fold, mirror = _symmetry.point_group_flags(name)
        rng = np.random.default_rng(100 * bandwidth + POINT_GROUPS.index(name))
        _, flm = random_pair_on_grid(bandwidth, n_fold, mirror, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        worst = 0.0
        for _ in range(3):
            zyz_true = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz_true)
            coarse, _ = correlator.correlate(flm, gln, n_fold, mirror)
            refined, _ = correlator.refine_zyz(flm, gln, n_fold, mirror, coarse)
            worst = max(worst, misorientation_deg(zyz_true, refined, name))
        record_property(f"point_group_{name}_bw{bandwidth}", f"{worst:.3e}")
        assert worst < CPP_GROUP_DEG

    @pytest.mark.parametrize("name", ["1", "4/m"])
    @pytest.mark.parametrize("bandwidth", [53, 68])
    def test_normalized_wedge_masked_pairs_are_recovered(
        self, name, bandwidth, record_property
    ):
        # the ``testNCorr()`` recipe, gated **as the C++ gates it**:
        # ``epsN`` applies to its symmetry free normalized loop
        # (line 316) while its normalized point group loop runs under
        # the loosened point group gate (lines 345 and 371-391), so a
        # single ``epsN`` over both would be this suite's own
        # tightening rather than the ported one
        n_fold, mirror = (
            (1, False) if name == "1" else _symmetry.point_group_flags(name)
        )
        bound = CPP_EPS_NORMALIZED_DEG if name == "1" else CPP_GROUP_DEG
        rng = np.random.default_rng(29 + bandwidth)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, n_fold, mirror, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, n_fold, mirror, mlm
        )
        for i in range(3):
            zyz_true = random_zyz(rng)
            gln = masked_pattern(transform, flm, mask, zyz_true)
            coarse, coarse_score = correlator.correlate(gln)
            refined, refined_score = correlator.refine_zyz(gln, coarse)
            coarse_delta = misorientation_deg(zyz_true, coarse, name)
            delta = misorientation_deg(zyz_true, refined, name)
            record_property(
                f"wedge_{name}_bw{bandwidth}_rot{i}",
                f"coarse {coarse_delta:.5f} refined {delta:.4e} deg, score "
                f"{coarse_score:.5f} -> {refined_score:.5f}",
            )
            assert delta < bound
            # the denominator is applied: without it the score is the
            # raw correlation, an order of magnitude away
            assert refined_score > coarse_score

    @pytest.mark.parametrize("beta", [0.0, 1e-3, -1e-3, math.pi, math.pi - 1e-3])
    def test_near_degenerate_targets_refine_under_a_tenth_degree(
        self, beta, record_property
    ):
        # the Phase 4 D5 defect zone, where the coarse result is a
        # whole cell out: refinement cures it away from the poles and
        # leaves exactly the ``beta`` offset itself on the pole,
        # since the 1 x 1 and 2 x 2 fallbacks freeze the false degree
        # of freedom
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        flm = random_alm(bandwidth, rng)
        zyz_true = np.array([0.4, beta, -0.7])
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        coarse, _ = correlator.correlate(flm, gln, 1, False)
        refined, _ = correlator.refine_zyz(flm, gln, 1, False, coarse)
        coarse_delta = misorientation_deg(zyz_true, coarse)
        delta = misorientation_deg(zyz_true, refined)
        record_property(
            f"near_degenerate_beta{beta:+.4f}",
            f"coarse {coarse_delta:.4f} refined {delta:.4e} deg",
        )
        assert delta < 0.1
        assert delta < coarse_delta

    def test_a_start_exactly_on_the_pole_uses_the_one_by_one_step(
        self, monkeypatch, record_property
    ):
        # ``hes[1, 1]`` is NaN there, so the loop solves the 1 x 1
        # alpha sub-problem and leaves the other two step slots at
        # exactly zero
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        zyz_true = np.array([0.4, 0.0, -0.7])
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        start = np.array([0.45, 0.0, -0.75])
        spy = DerivativeSpy(monkeypatch)
        zyz, value, converged = buffers.refine(flm, gln, start)
        record_property(
            "pole_start_iterations",
            f"{spy.iterations} iteration(s), converged {converged}, "
            f"step {np.array2string(buffers.step)}",
        )
        assert converged is True
        assert buffers.step[1] == 0.0
        assert buffers.step[2] == 0.0
        assert zyz[1] == start[1]
        assert zyz[2] == start[2]

    def test_far_starts_fail_back_to_the_start(self, record_property):
        # the saddle rejection contract working as designed: a start
        # unrelated to the peak is refused and comes back unchanged,
        # with the **analytic** value there rather than any
        # interpolated peak, which is what makes a failed refinement
        # change the score of a coarse result
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        n_failed = 0
        values = []
        for case in range(10):
            zyz_true = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz_true)
            start = np.asarray(random_zyz(rng), dtype=np.float64)
            value_at_start = buffers.derivatives(flm, gln, start, der=False)
            zyz, value, converged = buffers.refine(flm, gln, start)
            values.append(f"case {case} {'ok' if converged else 'failed'} {value:+.3f}")
            if not converged:
                n_failed += 1
                assert np.array_equal(zyz, start)
                assert value == value_at_start
                assert misorientation_deg(start, zyz) == 0.0
        record_property("far_start_failures", f"{n_failed}/10")
        record_property("far_start_values", "; ".join(values))
        # measured 9 of 10, and 36 of 40 over the extended sweep
        assert n_failed >= 8

    def test_a_far_start_can_converge_to_a_lower_value(self, record_property):
        # the pinned counter-example to "refinement can only raise
        # the score": Newton is local, and the 2 x 2 fallback freezes
        # ``step[2]`` and only tests ``det >= euEps``, so a converged
        # fixed point need not be a maximum
        bandwidth = FAR_START_BANDWIDTH
        rng = np.random.default_rng(FAR_START_SEED)
        buffers = RefineBuffers(bandwidth)
        flm = random_alm(bandwidth, rng)
        # wind the generator to the pinned case and then draw it
        # explicitly, so that the assertions below cannot read names
        # left over from some other iteration if ``FAR_START_CASE``
        # ever moves
        for _ in range(FAR_START_CASE):
            random_zyz(rng)
            random_zyz(rng)
        zyz_true = random_zyz(rng)
        start = np.asarray(random_zyz(rng), dtype=np.float64)
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        value_at_start = buffers.derivatives(flm, gln, start, der=False)
        zyz, value, converged = buffers.refine(flm, gln, start)
        moved = misorientation_deg(start, zyz)
        record_property(
            "far_start_decreaser",
            f"moved {moved:.3f} deg, value {value_at_start:+.3f} -> {value:+.3f}, "
            f"converged {converged}",
        )
        assert converged is True
        assert value < value_at_start
        assert moved == pytest.approx(FAR_START_MOVED_DEG, rel=0.1)
        assert value_at_start == pytest.approx(FAR_START_VALUE_BEFORE, rel=0.05)
        assert value == pytest.approx(FAR_START_VALUE_AFTER, rel=0.05)

    def test_a_constructed_saddle_is_rejected(self, monkeypatch, record_property):
        # an indefinite Hessian with a finite gradient: the Cholesky
        # solve throws in the C++ (``linalg.hpp`` line 416) and
        # returns status 1 here, the 2 x 2 sub-problem's determinant
        # is negative, and the whole refinement is a total failure
        # which hands the start back.  A mutant which ignores the
        # status, or substitutes ``numpy.linalg.solve``, walks
        # downhill instead
        bandwidth = 16
        buffers = RefineBuffers(bandwidth)
        rng = np.random.default_rng(3)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        saddle = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        fake = constant_derivatives(-7.25, np.array([0.1, 0.1, 0.1]), saddle)
        monkeypatch.setattr(_xcorr, "_derivatives", fake)
        start = np.array([0.3, 0.6, -0.9])
        zyz, value, converged = buffers.refine(flm, gln, start)
        record_property(
            "constructed_saddle",
            f"value {value} converged {converged}, "
            f"{fake.n_iterations} iteration(s), {fake.n_values} value call(s)",
        )
        assert converged is False
        assert np.array_equal(zyz, start)
        assert value == -7.25
        # the iteration count is the assertion which kills the
        # mutants: the derivatives are constant, so a loop which
        # accepted the saddle step would take the same step over and
        # over and time out at the iteration cap -- and *still*
        # return the start with this value and ``converged False``.
        # Measured 1 here and 15 with ``numpy.linalg.solve``
        # substituted for the status-carrying Cholesky solve
        assert fake.n_iterations == 1

    def test_steps_must_shrink(self, monkeypatch, record_property):
        # the monotone step rule of lines 463-465 and the C++'s
        # **non-update** of ``prevMag2`` on a fallback step.  The
        # third iteration's full step is longer than the first's but
        # shorter than the second's, so a mutant which updated
        # ``prev_mag2`` from the rejected second step accepts it --
        # and its ``gamma`` slot, which the 2 x 2 fallback freezes,
        # gives it away
        bandwidth = 16
        buffers = RefineBuffers(bandwidth)
        rng = np.random.default_rng(4)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        identity = np.eye(3)
        gradients = [
            np.array([0.1, 0.0, 0.0]),  # accepted, mag2 0.01
            np.array([0.5, 0.0, 0.0]),  # rejected, mag2 0.25 -> fallback
            np.array([0.15, 0.0, 0.3]),  # mag2 0.1125: rejected unless prev grew
            np.array([1e-9, 0.0, 0.0]),  # converged
        ]
        calls = {"n": 0}

        def fake(*args, **kwargs):
            jac = args[3]
            hes = args[4]
            der = kwargs["der"] if "der" in kwargs else args[8]
            if not der:
                return -1.0
            index = min(calls["n"], len(gradients) - 1)
            jac[:] = gradients[index]
            hes[:] = identity
            calls["n"] += 1
            return -1.0

        monkeypatch.setattr(_xcorr, "_derivatives", fake)
        start = np.array([0.0, 0.5, 0.0])
        zyz, value, converged = buffers.refine(flm, gln, start)
        record_property(
            "monotone_step",
            f"{calls['n']} iteration(s), zyz {np.array2string(zyz)}, "
            f"converged {converged}",
        )
        assert converged is True
        # the third step went through the 2 x 2 fallback, which
        # freezes gamma; a ``prev_mag2`` updated by the second,
        # rejected step would have let the full solve through and
        # moved gamma by 0.3
        assert zyz[2] == pytest.approx(start[2], abs=1e-12)
        assert zyz[0] == pytest.approx(start[0] - (0.1 + 0.5 + 0.15 + 1e-9), abs=1e-9)

    def test_the_stopping_threshold_is_the_ported_one(self, monkeypatch):
        # ``absEps = eps 2 pi / slP`` (line 446): a step just under
        # it stops the loop and a step just over it does not
        bandwidth = 16
        buffers = RefineBuffers(bandwidth)
        rng = np.random.default_rng(6)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        threshold = 0.01 * 2 * math.pi / buffers.side_length
        for scale, expected in ((0.5, 1), (2.0, _xcorr._REFINE_MAX_ITERATIONS)):
            calls = {"n": 0}

            def fake(*args, **kwargs):
                jac = args[3]
                hes = args[4]
                der = kwargs["der"] if "der" in kwargs else args[8]
                if not der:
                    return 0.0
                calls["n"] += 1
                jac[:] = np.array([scale * threshold, 0.0, 0.0])
                hes[:] = np.eye(3)
                return 0.0

            monkeypatch.setattr(_xcorr, "_derivatives", fake)
            buffers.refine(flm, gln, np.zeros(3))
            assert calls["n"] == expected

    def test_the_iteration_cap_is_fifteen(self):
        assert _xcorr._REFINE_MAX_ITERATIONS == 15
        assert _xcorr._REFINE_EPS == 0.01

    def test_the_convergence_scale_barely_matters(self, record_property):
        # recorded, not asserted tightly: the real-data residual is
        # systematic, so a hundredfold tighter ``eps`` changes
        # nothing.  A public knob would promise a precision it
        # cannot deliver, which is why there is none
        bandwidth = 53
        rng = np.random.default_rng(77)
        _, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz_true = random_zyz(rng)
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        coarse, _ = correlator.correlate(flm, gln, 1, False)
        loose, _ = correlator.refine_zyz(flm, gln, 1, False, coarse)
        tight, _ = correlator.refine_zyz(flm, gln, 1, False, coarse, eps=0.0001)
        delta = misorientation_deg(loose, tight)
        record_property("eps_insensitivity_deg", f"{delta:.3e}")
        assert delta < CPP_EPS_DEG


# ============ refine_zyz, correlate(refine=True), buffers ============ #


class TestCorrelateRefine:
    def test_the_private_correlators_keep_a_false_refine_default(self):
        # a deliberate deviation from the C++ ``ref = true`` of lines
        # 189 and 255: the user facing default lives on the indexer
        # and the signal method, and flipping the private one would
        # silently change every bare ``correlate`` call
        for method in (
            _xcorr.SphericalCrossCorrelator.correlate,
            _xcorr.NormalizedSphericalCrossCorrelator.correlate,
        ):
            parameter = inspect.signature(method).parameters["refine"]
            assert parameter.default is False
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_refine_zyz_signatures_are_frozen(self):
        plain = inspect.signature(_xcorr.SphericalCrossCorrelator.refine_zyz)
        assert list(plain.parameters) == [
            "self",
            "flm",
            "gln",
            "n_fold",
            "mirror",
            "zyz0",
            "eps",
        ]
        assert plain.parameters["eps"].default == 0.01
        normalized = inspect.signature(
            _xcorr.NormalizedSphericalCrossCorrelator.refine_zyz
        )
        assert list(normalized.parameters) == ["self", "gln", "zyz0", "eps"]
        assert normalized.parameters["eps"].default == 0.01

    def test_correlate_refine_true_returns_the_refined_peak(self, record_property):
        bandwidth = 53
        rng = np.random.default_rng(53)
        _, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz_true = random_zyz(rng)
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        coarse, coarse_score = correlator.correlate(flm, gln, 1, False)
        refined, refined_score = correlator.correlate(flm, gln, 1, False, refine=True)
        delta = misorientation_deg(zyz_true, refined)
        record_property(
            "correlate_refine_plain",
            f"coarse {misorientation_deg(zyz_true, coarse):.5f} refined "
            f"{delta:.3e} deg, score {coarse_score:.5f} -> {refined_score:.5f}",
        )
        assert delta < CPP_EPS_DEG

    def test_normalized_correlate_refine_true_returns_the_refined_peak(
        self, record_property
    ):
        bandwidth = 53
        rng = np.random.default_rng(29 + bandwidth)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        zyz_true = random_zyz(rng)
        gln = masked_pattern(transform, flm, mask, zyz_true)
        coarse, coarse_score = correlator.correlate(gln)
        refined, refined_score = correlator.correlate(gln, refine=True)
        delta = misorientation_deg(zyz_true, refined)
        record_property(
            "correlate_refine_normalized",
            f"refined {delta:.4e} deg, score {coarse_score:.5f} -> {refined_score:.5f}",
        )
        assert delta < CPP_EPS_NORMALIZED_DEG

    def test_refine_false_is_the_coarse_result_bitwise(self):
        bandwidth = 24
        rng = np.random.default_rng(9)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        bare_zyz, bare_score = correlator.correlate(flm, gln, 1, False)
        explicit_zyz, explicit_score = correlator.correlate(
            flm, gln, 1, False, refine=False
        )
        assert np.array_equal(bare_zyz, explicit_zyz)
        assert bare_score == explicit_score

    def test_the_emsphinx_keyword_only_moves_the_start(self):
        # the refinement itself has no compatibility branch, so the
        # two settings must agree far more tightly after refining
        # than they do before
        bandwidth = 53
        rng = np.random.default_rng(63)
        _, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz_true = random_zyz(rng)
        gln = _wigner.rotate_harmonics(flm, zyz_true)
        first, _ = correlator.correlate(
            flm, gln, 1, False, refine=True, emsphinx_compatible=True
        )
        second, _ = correlator.correlate(
            flm, gln, 1, False, refine=True, emsphinx_compatible=False
        )
        assert misorientation_deg(first, second) < CPP_EPS_DEG

    @pytest.mark.parametrize(
        "zyz0", [np.zeros(2), np.zeros(4), np.array([0.0, np.nan, 0.0])]
    )
    def test_refine_zyz_refuses_a_bad_start(self, zyz0):
        bandwidth = 16
        rng = np.random.default_rng(12)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        with pytest.raises(ValueError):
            correlator.refine_zyz(flm, gln, 1, False, zyz0)

    def test_a_bandwidth_mismatched_factor_triple_is_refused(self):
        factors = _wigner.wigner_d_table_factors(12)
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(16, wigner_d_factors=factors)
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(16, wigner_d_factors=factors[:2])

    def test_buffers_are_owned_and_reused(self):
        bandwidth = 16
        rng = np.random.default_rng(13)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        assert correlator.wigner_d_factors is None
        zyz0, _ = correlator.correlate(flm, gln, 1, False)

        first_zyz, first_score = correlator.refine_zyz(flm, gln, 1, False, zyz0)
        d_beta = correlator._d_beta
        factors = correlator.wigner_d_factors
        assert d_beta is not None
        assert d_beta.shape == (bandwidth,) * 3 + (2,)
        assert factors is not None

        # reused across calls, and the second call is bitwise the
        # first: a table left half written would not be
        second_zyz, second_score = correlator.refine_zyz(flm, gln, 1, False, zyz0)
        assert correlator._d_beta is d_beta
        assert correlator.wigner_d_factors is factors
        assert np.array_equal(first_zyz, second_zyz)
        assert first_score == second_score

        # a clone shares the read-only triple and owns its table.
        # The sharing is over the three arrays, not over the tuple:
        # the validation which every constructor runs returns a new
        # tuple of the very same arrays
        clone = correlator.clone()
        assert_shares_the_factor_triple(clone.wigner_d_factors, factors)
        clone_zyz, clone_score = clone.refine_zyz(flm, gln, 1, False, zyz0)
        assert clone._d_beta is not d_beta
        assert np.array_equal(clone_zyz, first_zyz)
        assert clone_score == first_score

    def test_the_table_is_allocated_through_the_tripwire(self, monkeypatch):
        # the per-evaluation call is the raw kernel, which cannot
        # check anything, so the Phase 3 ``out=`` contract is
        # enforced at the one moment a wrong buffer could enter: the
        # allocation must go through ``wigner_d_table_pre``, which
        # refuses a buffer whose undefined slots are not NaN.  The
        # seam is that wrapper and never ``numpy.full``, which is
        # the global NumPy every library in the process calls
        bandwidth = 16
        rng = np.random.default_rng(14)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz0, _ = correlator.correlate(flm, gln, 1, False)
        seen = []
        original = _wigner.wigner_d_table_pre

        def spy(*args, **kwargs):
            # copied before the call, which fills the buffer
            out = kwargs.get("out")
            seen.append(None if out is None else np.array(out, copy=True))
            return original(*args, **kwargs)

        monkeypatch.setattr(_wigner, "wigner_d_table_pre", spy)
        correlator.refine_zyz(flm, gln, 1, False, zyz0)
        assert seen, "`d_beta` must be allocated through `wigner_d_table_pre`"
        buffer = seen[0]
        assert buffer is not None, "the buffer must be handed over as `out=`"
        assert buffer.shape == (bandwidth,) * 3 + (2,)
        # an ``np.empty`` allocation dies here, before the wrapper's
        # own tripwire is even reached
        assert np.isnan(buffer).all()

    def test_a_buffer_whose_undefined_slots_are_not_nan_is_refused(self, monkeypatch):
        # the tripwire itself, on the allocation path: a correlator
        # which allocated with ``numpy.empty`` and still routed the
        # buffer through the wrapper is refused there
        bandwidth = 16
        rng = np.random.default_rng(14)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz0, _ = correlator.correlate(flm, gln, 1, False)
        original = _wigner.wigner_d_table_pre

        def zero_filled(*args, **kwargs):
            if kwargs.get("out") is not None:
                kwargs["out"] = np.zeros_like(kwargs["out"])
            return original(*args, **kwargs)

        monkeypatch.setattr(_wigner, "wigner_d_table_pre", zero_filled)
        with pytest.raises(ValueError, match="NaN"):
            correlator.refine_zyz(flm, gln, 1, False, zyz0)

    def test_the_normalized_class_owns_no_refinement_buffer(self):
        # it refines through its inner plain correlator, so its own
        # attribute set is unchanged and ``clone()`` keeps copying
        # exactly the seven attributes it always did
        bandwidth = 17
        rng = np.random.default_rng(15)
        _, flm, flm2, mlm, _ = masked_case(bandwidth, 1, False, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        assert "_d_beta" not in vars(correlator)
        assert vars(correlator.clone()).keys() == vars(correlator).keys()
        clone = correlator.clone()
        assert clone.correlator._d_beta is None

    def test_the_normalized_denominator_is_applied(self):
        # the refined normalized score is the un-normalised
        # correlation over ``denominator(zyz)`` at the refined
        # rotation, so the two differ by exactly that factor
        bandwidth = 53
        rng = np.random.default_rng(82)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        zyz_true = random_zyz(rng)
        gln = masked_pattern(transform, flm, mask, zyz_true)
        coarse, _ = correlator.correlate(gln)
        refined, score = correlator.refine_zyz(gln, coarse)
        plain_zyz, plain_score = correlator.correlator.refine_zyz(
            flm, gln, 1, False, coarse
        )
        denominator = correlator._denominator(refined)
        assert np.array_equal(plain_zyz, refined)
        assert denominator > 0
        assert score == pytest.approx(plain_score / denominator, rel=1e-12)


# ================== The indexer's plumbing (D3/D7/D8) ================ #


class TestIndexerRefinePlumbing:
    def test_the_default_is_true(self):
        parameter = inspect.signature(SphericalIndexer.__init__).parameters["refine"]
        assert parameter.default is True
        signal_parameter = inspect.signature(
            kp.signals.EBSD.spherical_indexing
        ).parameters["refine"]
        assert signal_parameter.default is True

    def test_construction_with_refine_true_no_longer_raises(self):
        indexer = ni_indexer()
        assert indexer.refine is True
        assert ni_indexer(refine=False).refine is False

    def test_the_factor_triple_is_built_once_and_shared(self):
        # laziness alone would not share it: the chunk workers clone
        # *before* their first refinement, so every clone would build
        # its own 5 MB triple
        indexer = SphericalIndexer(
            [ni_harmonics(), scrambled_harmonics()], ni_detector()
        )
        factors = indexer.wigner_d_factors
        assert factors is not None
        for correlator in indexer.correlators:
            assert_shares_the_factor_triple(
                correlator.correlator.wigner_d_factors, factors
            )
            assert_shares_the_factor_triple(
                correlator.clone().correlator.wigner_d_factors, factors
            )

    def test_the_un_normalised_prototype_shares_the_triple(self):
        # without the keyword on the prototype every chunk clone
        # would lazily rebuild the triple, which is the exact failure
        # the eager sharing exists to prevent
        indexer = ni_indexer(normalize=False)
        factors = indexer.wigner_d_factors
        assert_shares_the_factor_triple(indexer.correlator.wigner_d_factors, factors)
        assert_shares_the_factor_triple(
            indexer.correlator.clone().wigner_d_factors, factors
        )

    def test_a_coarse_indexer_builds_no_triple(self):
        indexer = ni_indexer(refine=False)
        assert indexer.wigner_d_factors is None
        assert indexer.correlators[0].correlator.wigner_d_factors is None

    def test_every_phase_is_refined_before_insertion(self, monkeypatch):
        # ``indexImage()`` refines inside each phase's ``correlate``
        # call (line 230), so with two phases there are two
        # refinements per pattern and the top-n ordering uses the
        # refined scores.  "Refine only the winner" would reorder
        # near ties
        calls = []
        original = _xcorr.NormalizedSphericalCrossCorrelator.correlate

        def spy(self, gln, **kwargs):
            calls.append(kwargs.get("refine"))
            return original(self, gln, **kwargs)

        monkeypatch.setattr(_xcorr.NormalizedSphericalCrossCorrelator, "correlate", spy)
        indexer = SphericalIndexer(
            [ni_harmonics(), scrambled_harmonics()], ni_detector()
        )
        patterns = ni_signal().data.reshape((-1, 60, 60))[:2]
        indexer.index_patterns(patterns, n_best=2, progressbar=False)
        assert len(calls) == 4
        assert all(call is True for call in calls)

    def test_a_fill_row_is_never_refined(self):
        # a run with fewer phases than ``n_best`` leaves fill rows,
        # which carry no candidate at all and must not be handed to
        # a refinement
        indexer = ni_indexer()
        patterns = ni_signal().data.reshape((-1, 60, 60))[:3]
        results = indexer.index_patterns(patterns, n_best=3, progressbar=False)
        # the run itself succeeded, so the fill assertions below are
        # about fill rows and not about a failed map
        assert (results["phase_id"][:, 0] == 0).all()
        assert (results["scores"][:, 0] > 0).all()
        assert (results["phase_id"][:, 1:] == -1).all()
        assert (results["scores"][:, 1:] == 0.0).all()
        assert (results["zyz"][:, 1:] == 0.0).all()


class TestRefinedMemoryModel:
    def test_the_single_phase_model(self):
        assert ni_indexer(refine=False).memory_per_worker_bytes == (
            MEMORY_BASE_ONE_PHASE
        )
        assert ni_indexer().memory_per_worker_bytes == MEMORY_REFINED_ONE_PHASE
        assert MEMORY_REFINED_ONE_PHASE - MEMORY_BASE_ONE_PHASE == MEMORY_D_BETA_BW68

    def test_the_two_phase_model_keeps_the_correlator_factor(self):
        # a worker holds one Wigner table per correlator **clone**,
        # so a flat term would understate every multi-phase
        # normalized run by ``(P - 1) 16 bw^3`` and make the 2 GiB
        # warning under-fire on exactly the runs which need it
        two = SphericalIndexer([ni_harmonics(), scrambled_harmonics()], ni_detector())
        assert two.memory_per_worker_bytes == MEMORY_REFINED_TWO_PHASES
        assert (
            MEMORY_REFINED_TWO_PHASES - MEMORY_BASE_TWO_PHASES == 2 * MEMORY_D_BETA_BW68
        )

    def test_an_un_normalised_run_pays_for_one_table(self):
        one = ni_indexer(normalize=False)
        two = SphericalIndexer(
            [ni_harmonics(), scrambled_harmonics()], ni_detector(), normalize=False
        )
        assert two.memory_per_worker_bytes == one.memory_per_worker_bytes
        assert one.memory_per_worker_bytes == MEMORY_BASE_ONE_PHASE + MEMORY_D_BETA_BW68

    def test_a_coarse_indexer_can_still_price_a_refining_run(self):
        # what ``refine_patterns`` prints: it always refines, so its
        # information message must not report the constructor's model
        indexer = ni_indexer(refine=False)
        assert indexer._memory_model(False) == MEMORY_BASE_ONE_PHASE
        assert indexer._memory_model(True) == MEMORY_REFINED_ONE_PHASE


class TestRefinedInfoMessage:
    def test_the_refinement_line(self):
        assert "Refinement: Newton (on)" in ni_indexer().get_info_message(9, 1)
        assert "Refinement: off" in ni_indexer(refine=False).get_info_message(9, 1)

    def test_the_memory_line_prints_the_refined_model(self):
        message = ni_indexer().get_info_message(9, 1)
        assert "Estimated memory per worker: 54 MB" in message

    def test_the_refining_verb(self):
        message = ni_indexer(refine=False).get_info_message(9, 1, refining=True)
        assert "Refining 9 orientation(s) in 9 chunk(s)" in message
        assert "Indexing" not in message
        assert "Refinement: Newton (on)" in message
        assert "Estimated memory per worker: 54 MB" in message

    def test_the_indexing_verb_is_the_default(self):
        message = ni_indexer().get_info_message(9, 1)
        assert "Indexing 9 pattern(s)" in message
        assert "Refining" not in message


# =================== refine_patterns and its chunks ================== #


def _coarse_results(indexer, patterns):
    """Return the coarse indexing results of a stack of patterns."""
    return indexer.index_patterns(patterns, progressbar=False)


class TestRefinePatterns:
    @staticmethod
    def _setup(**kwargs):
        indexer = ni_indexer(refine=False, **kwargs)
        patterns = ni_signal().data.reshape((-1, 60, 60))
        coarse = _coarse_results(indexer, patterns)
        return indexer, patterns, coarse

    def test_the_result_contract(self):
        indexer, patterns, coarse = self._setup()
        results = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            progressbar=False,
        )
        assert set(results) == {"zyz", "scores", "iq", "phase_id"}
        assert results["zyz"].shape == (9, 3)
        assert results["scores"].shape == (9,)
        assert results["iq"].shape == (9,)
        assert results["phase_id"].shape == (9,)
        assert results["phase_id"].dtype == np.int32
        assert np.array_equal(results["phase_id"], coarse["phase_id"][:, 0])
        assert (results["scores"] > 0).all()

    def test_a_coarse_indexer_still_shares_the_triple(self, monkeypatch):
        # the method always refines, so it must arm the sharing
        # itself: ``SphericalIndexer(refine=False).refine_patterns()``
        # is a public, reachable call, and without this every chunk
        # clone rebuilds a 5 MB triple
        indexer, patterns, coarse = self._setup()
        assert indexer.wigner_d_factors is None
        seen = []
        original = _xcorr.NormalizedSphericalCrossCorrelator.clone

        def spy(self):
            clone = original(self)
            seen.append(clone.correlator.wigner_d_factors)
            return clone

        monkeypatch.setattr(_xcorr.NormalizedSphericalCrossCorrelator, "clone", spy)
        indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            chunksize=1,
            progressbar=False,
        )
        assert seen
        assert seen[0] is not None
        for triple in seen:
            assert_shares_the_factor_triple(triple, seen[0])
        # and it is the one the method armed on the indexer itself
        assert_shares_the_factor_triple(seen[0], indexer.wigner_d_factors)

    def test_a_not_indexed_row_passes_through_bitwise(self):
        indexer, patterns, coarse = self._setup()
        zyz = coarse["zyz"][:, 0].copy()
        phase_id = coarse["phase_id"][:, 0].copy()
        phase_id[4] = -1
        results = indexer.refine_patterns(patterns, zyz, phase_id, progressbar=False)
        assert np.array_equal(results["zyz"][4], zyz[4])
        assert results["phase_id"][4] == -1
        # the whole row, not only the two echoed columns: a point
        # which is not indexed carries no score and no image quality
        # of its own, so the caller can leave the input map's values
        # in place there
        assert results["scores"][4] == 0.0
        assert results["iq"][4] == 0.0
        # and the other eight were refined
        assert not np.array_equal(results["zyz"][0], zyz[0])
        assert results["scores"][0] > 0.0

    def test_a_failing_pattern_keeps_its_input_row(self, monkeypatch):
        indexer, patterns, coarse = self._setup()

        def exploding(*args, **kwargs):
            raise RuntimeError("injected refinement failure")

        monkeypatch.setattr(_xcorr, "_refine_peak", exploding)
        results = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            progressbar=False,
        )
        assert np.array_equal(results["zyz"], coarse["zyz"][:, 0])
        assert np.array_equal(results["phase_id"], coarse["phase_id"][:, 0])

    def test_a_phase_index_out_of_range_is_refused(self):
        indexer, patterns, coarse = self._setup()
        phase_id = coarse["phase_id"][:, 0].copy()
        phase_id[2] = 3
        with pytest.raises(ValueError):
            indexer.refine_patterns(
                patterns, coarse["zyz"][:, 0], phase_id, progressbar=False
            )

    @pytest.mark.parametrize("chunksize", [1, 4, 9])
    def test_the_chunk_size_does_not_change_the_result(self, chunksize):
        # ``index_patterns`` maps a single array, so the starting
        # triples and phase indices have to be block aligned with the
        # patterns explicitly: a mis-aligned chunk would silently
        # refine from the wrong starts and still return nine rows
        indexer, patterns, coarse = self._setup()
        reference = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            chunksize=9,
            progressbar=False,
        )
        other = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            chunksize=chunksize,
            progressbar=False,
        )
        assert np.array_equal(other["zyz"], reference["zyz"])
        assert np.array_equal(other["scores"], reference["scores"])
        assert np.array_equal(other["iq"], reference["iq"])

    def test_the_worker_count_does_not_change_the_result(self):
        indexer, patterns, coarse = self._setup()
        with dask.config.set(num_workers=1):
            one = indexer.refine_patterns(
                patterns,
                coarse["zyz"][:, 0],
                coarse["phase_id"][:, 0],
                chunksize=1,
                progressbar=False,
            )
        with dask.config.set(num_workers=4):
            four = indexer.refine_patterns(
                patterns,
                coarse["zyz"][:, 0],
                coarse["phase_id"][:, 0],
                chunksize=1,
                progressbar=False,
            )
        assert np.array_equal(four["zyz"], one["zyz"])
        assert np.array_equal(four["scores"], one["scores"])

    def test_the_rows_are_not_shuffled_by_the_blocks(self):
        # a mis-alignment which is *consistent* across chunk sizes
        # would survive the invariance test above, so the starting
        # triples are also permuted here: every refined row must
        # follow its own pattern
        indexer, patterns, coarse = self._setup()
        order = np.array([4, 5, 6, 7, 8, 0, 1, 2, 3])
        straight = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            chunksize=2,
            progressbar=False,
        )
        permuted = indexer.refine_patterns(
            patterns[order],
            coarse["zyz"][:, 0][order],
            coarse["phase_id"][:, 0][order],
            chunksize=2,
            progressbar=False,
        )
        assert np.allclose(permuted["zyz"], straight["zyz"][order])
        assert np.allclose(permuted["scores"], straight["scores"][order])

    def test_an_un_normalised_indexer_refines_through_the_prototype(self):
        indexer, patterns, coarse = self._setup(normalize=False)
        results = indexer.refine_patterns(
            patterns,
            coarse["zyz"][:, 0],
            coarse["phase_id"][:, 0],
            progressbar=False,
        )
        # the un-normalised score is the analytic correlation, which
        # lies in the recorded un-normalised band
        assert results["scores"].min() > 0.2
        assert results["scores"].max() < 0.5
        assert (results["scores"] > coarse["scores"][:, 0]).all()

    def test_rows_which_do_not_match_the_patterns_are_refused(self):
        indexer, patterns, coarse = self._setup()
        zyz = coarse["zyz"][:, 0]
        phase_id = coarse["phase_id"][:, 0]
        with pytest.raises(ValueError):
            indexer.refine_patterns(patterns, zyz[:5], phase_id, progressbar=False)
        with pytest.raises(ValueError):
            indexer.refine_patterns(patterns, zyz, phase_id[:5], progressbar=False)
        with pytest.raises(ValueError):
            indexer.refine_patterns(patterns, zyz[:, :2], phase_id, progressbar=False)


# ===================== Real data, the small map ====================== #


class TestRefinedNickelSmall:
    def test_the_default_call_is_refined_and_more_accurate(self, record_property):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        refined = index_ni(signal=signal)
        coarse_angles = misorientation(coarse.rotations, stored_rotations())
        angles = misorientation(refined.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_coarse", coarse_angles)
        record_angles(record_property, "small_bw68_refined", angles)
        # the roadmap's own bound, and the measured-then-pinned median
        assert (angles < SMALL_REFINED_ALL_DEG).all()
        assert np.median(angles) < SMALL_REFINED_MEDIAN_DEG
        assert np.median(angles) < np.median(coarse_angles)

    def test_every_score_rises(self, record_property):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        refined = index_ni(signal=signal)
        deltas = refined.scores - coarse.scores
        record_property(
            "small_bw68_score_deltas",
            ", ".join(f"{d:+.4f}" for d in deltas),
        )
        record_property(
            "small_bw68_refined_scores",
            f"{refined.scores.min():.4f}-{refined.scores.max():.4f} mean "
            f"{refined.scores.mean():.4f}",
        )
        assert (deltas > 0).all()
        assert deltas.mean() > SMALL_REFINED_SCORE_MEAN_DELTA
        assert refined.scores.mean() == pytest.approx(
            SMALL_REFINED_SCORE_MEAN, rel=0.05
        )

    def test_the_iteration_counts_are_recorded(self, monkeypatch, record_property):
        # measured eight patterns in two iterations and one in three,
        # with no failures: the C++ comment's "generally at most 3"
        spy = DerivativeSpy(monkeypatch)
        index_ni()
        record_property(
            "small_bw68_derivative_calls",
            f"{spy.iterations} with der=True, {spy.value_only} with der=False",
        )
        # nine patterns, at least one Newton iteration each and at
        # least one value-only evaluation each for the normalized
        # denominator.  Measured 19 and 18, i.e. the eight two- and
        # one three-iteration patterns and the denominator's two
        # evaluations per pattern; the exact figures are recorded
        # rather than asserted, since ``>= 18`` would sit on the
        # measured value and a denominator evaluated once would be
        # a legal implementation of the same contract
        assert spy.iterations >= 9
        assert spy.value_only >= 9

    def test_unnormalized_refinement_raises_the_unnormalized_scores(
        self, record_property
    ):
        signal = ni_signal()
        coarse = index_ni(signal=signal, normalize=False, refine=False)
        refined = index_ni(signal=signal, normalize=False)
        normalized = index_ni(signal=signal)
        angles = misorientation(refined.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_refined_plain", angles)
        deltas = refined.scores - coarse.scores
        record_property(
            "small_bw68_plain_score_deltas",
            ", ".join(f"{d:+.5f}" for d in deltas),
        )
        # the Newton step maximizes the un-normalised value in both
        # paths and both start from the same coarse cell, so the
        # refined orientations agree per pattern
        agreement = misorientation(refined.rotations, normalized.rotations)
        record_property("small_bw68_plain_vs_normalized", f"{agreement.max():.3e}")
        assert agreement.max() < EQUIVALENCE_ANGLE_DEG
        assert (deltas > 0).all()
        assert refined.scores.mean() == pytest.approx(
            SMALL_REFINED_PLAIN_SCORE_MEAN, rel=0.05
        )

    def test_the_refined_result_is_deterministic(self):
        signal = ni_signal()
        reference = index_ni(signal=signal, chunksize=9)
        # the run indexed, so the comparisons below are between two
        # refined maps and not between two all-fill ones
        assert reference.is_indexed.all()
        for chunksize in (1, 4):
            other = index_ni(signal=signal, chunksize=chunksize)
            assert np.array_equal(other.rotations.data, reference.rotations.data)
            assert np.array_equal(other.scores, reference.scores)
        with dask.config.set(num_workers=4):
            four = index_ni(signal=signal, chunksize=1)
        assert np.array_equal(four.rotations.data, reference.rotations.data)
        assert np.array_equal(four.scores, reference.scores)

    def test_the_information_message_names_the_refinement(self, capsys):
        index_ni(verbose=1)
        out = capsys.readouterr().out
        assert "Refinement: Newton (on)" in out
        assert "Estimated memory per worker: 54 MB" in out

    @pytest.mark.weekly
    def test_a_larger_bandwidth_shrinks_the_residual(self, record_property):
        refined = index_ni(harmonics=ni_harmonics(88), bandwidth=88)
        angles = misorientation(refined.rotations, stored_rotations())
        record_angles(record_property, "small_bw88_refined", angles)
        assert np.median(angles) < SMALL_REFINED_MEDIAN_DEG
        assert (angles < SMALL_REFINED_ALL_DEG).all()


# ============== EBSD.refine_orientation_spherical (D9) =============== #


def refine_ni(xmap, signal=None, **kwargs):
    """Return the refined map of a Ni call, defaults kept."""
    if signal is None:
        signal = ni_signal()
    kwargs.setdefault("verbose", 0)
    harmonics = kwargs.pop("harmonics", ni_harmonics())
    detector = kwargs.pop("detector", ni_detector())
    return signal.refine_orientation_spherical(xmap, harmonics, detector, **kwargs)


class TestRefineOrientationSpherical:
    def test_the_signature_is_frozen(self):
        parameters = inspect.signature(
            kp.signals.EBSD.refine_orientation_spherical
        ).parameters
        assert list(parameters)[:4] == ["self", "xmap", "harmonics", "detector"]
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        assert defaults == {
            "bandwidth": 68,
            "navigation_mask": None,
            "signal_mask": None,
            "normalize": True,
            "n_regions": 10,
            "gaussian_background": False,
            "circular_mask": False,
            "emsphinx_compatible": True,
            "chunksize": None,
            "verbose": 1,
        }
        # the eager pipeline offers ``chunksize``, not the lazy
        # knobs of the sibling refinements
        for absent in ("compute", "rechunk", "chunk_kwargs"):
            assert absent not in parameters

    def test_it_sits_next_to_spherical_indexing(self):
        names = [
            name
            for name in vars(kp.signals.EBSD)
            if name in ("spherical_indexing", "refine_orientation_spherical")
        ]
        assert names == ["spherical_indexing", "refine_orientation_spherical"]

    def test_it_equals_a_refining_index_run(self, record_property):
        # the stored quaternion hands back the glide-equivalent
        # triple, whose refinement walks the mirrored Wigner table
        # path, so this is an equality to tolerance and never a
        # bitwise one
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        indexed = index_ni(signal=signal)
        refined = refine_ni(coarse, signal=signal)
        angles = misorientation(refined.rotations, indexed.rotations)
        score_difference = np.abs(refined.scores - indexed.scores)
        record_property("refine_only_vs_indexed_deg", f"{angles.max():.3e}")
        record_property("refine_only_vs_indexed_score", f"{score_difference.max():.3e}")
        assert angles.max() < EQUIVALENCE_ANGLE_DEG
        assert score_difference.max() < EQUIVALENCE_SCORE

    def test_the_image_quality_is_recomputed(self):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        indexed = index_ni(signal=signal)
        refined = refine_ni(coarse, signal=signal)
        assert np.allclose(refined.iq, indexed.iq)

    def test_the_map_structure_is_carried_over(self):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        refined = refine_ni(coarse, signal=signal)
        assert isinstance(refined, CrystalMap)
        assert refined.shape == coarse.shape
        assert refined.scan_unit == coarse.scan_unit
        assert refined.phases.names == coarse.phases.names
        assert np.array_equal(refined.phase_id, coarse.phase_id)
        assert np.array_equal(refined.is_in_data, coarse.is_in_data)
        assert np.array_equal(refined.is_indexed, coarse.is_indexed)
        assert refined.rotations.shape == (9,)
        assert set(refined.prop) >= {"scores", "iq"}

    def test_a_point_masked_at_refine_time_keeps_its_row(self):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        refined = refine_ni(coarse, signal=signal, navigation_mask=mask)
        assert np.array_equal(refined.rotations[4].data, coarse.rotations[4].data)
        assert refined.scores[4] == coarse.scores[4]
        assert refined.iq[4] == coarse.iq[4]
        # and its neighbours moved
        assert not np.array_equal(refined.rotations[0].data, coarse.rotations[0].data)

    def test_a_not_indexed_point_keeps_its_row(self):
        signal = ni_signal()
        signal.data[1, 1] = np.full((60, 60), 37, np.uint8)
        coarse = index_ni(signal=signal, refine=False)
        assert not coarse.is_indexed[4]
        refined = refine_ni(coarse, signal=signal)
        assert not refined.is_indexed[4]
        assert refined.scores[4] == coarse.scores[4]
        assert np.array_equal(refined.rotations[4].data, coarse.rotations[4].data)

    def test_a_failing_pattern_keeps_its_row(self, monkeypatch):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        target = np.array(signal.data[1, 1])
        original = _indexer._preprocess_pattern

        def exploding(pattern, **kwargs):
            if np.array_equal(pattern, target):
                raise RuntimeError("injected per-pattern failure")
            return original(pattern, **kwargs)

        monkeypatch.setattr(_indexer, "_preprocess_pattern", exploding)
        refined = refine_ni(coarse, signal=signal)
        assert np.array_equal(refined.rotations[4].data, coarse.rotations[4].data)
        assert refined.scores[4] == coarse.scores[4]
        assert refined.iq[4] == coarse.iq[4]

    def test_an_incompatible_map_is_refused(self):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        with pytest.raises(ValueError, match="must be the same"):
            refine_ni(coarse, signal=signal.inav[0])

    def test_a_sparse_mask_map_is_refused(self, record_property):
        # orix derives ``xmap.shape`` from the coordinates of the
        # points which are in the data, so a navigation-masked map
        # has a bounding-box shape and fails the compatibility check
        # every kikuchipy refinement applies.  The supported route is
        # to index the full map and mask at refine time
        signal = ni_signal()
        mask = np.ones((3, 3), dtype=bool)
        mask[0] = False
        sparse = index_ni(signal=signal, refine=False, navigation_mask=mask)
        record_property("sparse_mask_xmap_shape", str(sparse.shape))
        assert sparse.shape != (3, 3)
        with pytest.raises(ValueError, match="must be the same"):
            refine_ni(sparse, signal=signal)

    def test_a_phase_index_out_of_range_is_refused(self):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        phase_list = PhaseList(
            [coarse.phases[0].deepcopy(), Phase("other", point_group="1")]
        )
        keys, _ = create_coordinate_arrays((3, 3), (1.5, 1.5))
        keys["rotations"] = coarse.rotations
        keys["phase_id"] = np.ones(9, dtype=np.int32)
        keys["prop"] = {"scores": coarse.scores.copy(), "iq": coarse.iq.copy()}
        foreign = CrystalMap(phase_list=phase_list, **keys)
        with pytest.raises(ValueError):
            refine_ni(foreign, signal=signal)

    def test_a_phase_which_is_not_the_masters_is_refused(self):
        # ids which merely happen to be in range must not silently
        # refine against the wrong master pattern, which is what a
        # Hough or dictionary map with a re-ordered phase list gives
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        phase_list = PhaseList([Phase("copper", point_group="m-3m")])
        keys, _ = create_coordinate_arrays((3, 3), (1.5, 1.5))
        keys["rotations"] = coarse.rotations
        keys["phase_id"] = np.zeros(9, dtype=np.int32)
        keys["prop"] = {"scores": coarse.scores.copy(), "iq": coarse.iq.copy()}
        other = CrystalMap(phase_list=phase_list, **keys)
        with pytest.raises(ValueError) as info:
            refine_ni(other, signal=signal)
        message = str(info.value)
        assert "copper" in message
        assert "ni" in message
        assert "names" in message

    def test_a_foreign_start_is_not_score_monotone(self, record_property):
        # Newton is local, so an orientation which did not come from
        # this pipeline may converge to a stationary point below its
        # start or fail and keep its input.  The disjunction is the
        # assertion; which branch ran is recorded
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        rotations = coarse.rotations.data.copy()
        # a seeded unrelated orientation, drawn the way
        # ``randomRotation()`` of ``sht_xcorr.cpp`` draws one
        foreign_zyz = random_zyz(np.random.default_rng(7))
        rotations[4] = _euler.rotation_from_zyz(foreign_zyz).data[0]
        # the injected row carries the score the refinement itself
        # reports at that start, computed here from the kernel and
        # the phase's denominator.  Leaving the Ni coarse score
        # there would make the comparison below vacuous: no
        # refinement from a random orientation can reach it, so the
        # disjunct would hold for an implementation which never
        # refined at all
        indexer = ni_indexer()
        gln = pattern_spectrum(indexer, np.asarray(signal.data[1, 1]))
        start_zyz = _euler.rotation_to_zyz(Rotation(rotations[4][np.newaxis]))[0]
        scores = coarse.scores.copy()
        scores[4] = analytic_normalized_score(indexer, gln, start_zyz)
        keys, _ = create_coordinate_arrays((3, 3), (1.5, 1.5))
        keys["rotations"] = Rotation(rotations)
        keys["phase_id"] = coarse.phase_id.astype(np.int32)
        keys["prop"] = {"scores": scores, "iq": coarse.iq.copy()}
        foreign = CrystalMap(
            phase_list=PhaseList([coarse.phases[0].deepcopy()]), **keys
        )
        refined = refine_ni(foreign, signal=signal)
        unchanged = np.array_equal(refined.rotations[4].data, rotations[4])
        record_property(
            "foreign_start_branch",
            f"{'unchanged' if unchanged else 'converged'}, score "
            f"{foreign.scores[4]:.6f} -> {refined.scores[4]:.6f}",
        )
        # measured: the refinement fails from this start and returns
        # it with the analytic value there, which is the score put
        # into the input row above
        assert unchanged or refined.scores[4] <= foreign.scores[4] + 1e-12

    def test_the_verbose_wording(self, capsys):
        signal = ni_signal()
        coarse = index_ni(signal=signal, refine=False)
        refine_ni(coarse, signal=signal, verbose=0)
        assert capsys.readouterr().out == ""
        refine_ni(coarse, signal=signal, verbose=1)
        out = capsys.readouterr().out
        assert "Refining 9 orientation(s)" in out
        assert "Refinement speed:" in out
        assert "Indexing" not in out


# ================ Public messages of the new surface ================= #


def _identity_map(phase, scores=None):
    """Return a 3 x 3 crystal map of identity rotations."""
    keys, _ = create_coordinate_arrays((3, 3), (1.5, 1.5))
    keys["rotations"] = Rotation(np.tile([1.0, 0.0, 0.0, 0.0], (9, 1)))
    keys["phase_id"] = np.zeros(9, dtype=np.int32)
    keys["prop"] = {
        "scores": np.zeros(9) if scores is None else scores,
        "iq": np.zeros(9),
    }
    return CrystalMap(phase_list=PhaseList([phase]), **keys)


class TestPublicMessages:
    def test_no_public_refusal_names_a_roadmap_phase(self):
        # the assertion the deleted ``refine=True`` refusal tests
        # carried, re-homed onto the refusals which survive the
        # flip.  The docstring guards of
        # ``test_spherical_indexer.py`` police the prose; this
        # policies the messages a user actually hits
        messages = []
        bandwidth = 16
        rng = np.random.default_rng(41)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        with pytest.raises(ValueError) as info:
            correlator.refine_zyz(flm, gln, 1, False, np.zeros(2))
        messages.append(str(info.value))
        with pytest.raises(ValueError) as info:
            _xcorr.SphericalCrossCorrelator(
                bandwidth, wigner_d_factors=_wigner.wigner_d_table_factors(12)
            )
        messages.append(str(info.value))

        indexer = ni_indexer(refine=False)
        patterns = ni_signal().data.reshape((-1, 60, 60))
        zyz = np.zeros((9, 3))
        phase_id = np.zeros(9, dtype=np.int32)
        phase_id[2] = 3
        with pytest.raises(ValueError) as info:
            indexer.refine_patterns(patterns, zyz, phase_id, progressbar=False)
        messages.append(str(info.value))
        with pytest.raises(ValueError) as info:
            indexer.refine_patterns(
                patterns, zyz[:5], np.zeros(9, dtype=np.int32), progressbar=False
            )
        messages.append(str(info.value))

        foreign = _identity_map(Phase("copper", point_group="m-3m"))
        with pytest.raises(ValueError) as info:
            refine_ni(foreign, signal=ni_signal())
        messages.append(str(info.value))
        with pytest.raises(ValueError) as info:
            refine_ni(foreign, signal=ni_signal().inav[0])
        messages.append(str(info.value))

        assert len(messages) == 6
        for message in messages:
            assert message
            for token in (
                "Phase 5",
                "Phase 6",
                "Phase 7",
                "Phase 8",
                "spherical-indexing",
                "spherical-refinement",
            ):
                assert token not in message, f"{token!r} in {message!r}"


# ================== Real data, the large map (D10) =================== #


class TestRefinedNickelLarge:
    @staticmethod
    def _subset(step, record_property, tag):
        pytest.importorskip("pooch")
        signal = kp.data.nickel_ebsd_large(allow_download=True)
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        detector = signal.detector.deepcopy()
        detector.pc = detector.pc_average

        nav_shape = signal.axes_manager.navigation_shape[::-1]
        mask = np.ones(nav_shape, dtype=bool)
        mask[::step, ::step] = False
        keep = np.flatnonzero(~mask.ravel())

        coarse = signal.spherical_indexing(
            ni_harmonics(),
            detector,
            navigation_mask=mask,
            refine=False,
            verbose=0,
        )
        refined = signal.spherical_indexing(
            ni_harmonics(),
            detector,
            navigation_mask=mask,
            verbose=0,
        )
        reference = signal.xmap.rotations[keep]
        angles = misorientation(refined.rotations, reference)
        record_angles(record_property, tag, angles)
        record_angles(
            record_property,
            f"{tag}_coarse",
            misorientation(coarse.rotations, reference),
        )
        deltas = refined.scores - coarse.scores
        record_property(
            f"{tag}_score_deltas",
            f"min {deltas.min():+.5f} max {deltas.max():+.5f} mean "
            f"{deltas.mean():+.5f}, up {int((deltas > 0).sum())}/{deltas.size}",
        )
        return angles, deltas

    def test_the_twenty_point_subset(self, record_property):
        angles, deltas = self._subset(15, record_property, "large_20pt_refined")
        assert angles.size == 20
        assert np.median(angles) < LARGE_REFINED_MEDIAN_DEG
        assert angles.max() < LARGE_REFINED_MAX_DEG
        assert (deltas > 0).all()

    @pytest.mark.weekly
    def test_the_one_hundred_and_sixty_five_point_subset(self, record_property):
        angles, deltas = self._subset(5, record_property, "large_165pt_refined")
        assert angles.size == 165
        assert np.median(angles) < LARGE_REFINED_MEDIAN_DEG
        assert np.percentile(angles, 95) < WEEKLY_REFINED_P95_DEG
        assert angles.max() < LARGE_REFINED_MAX_DEG
        # the few points where the omitted window chain rule lets the
        # normalized score dip are recorded, not asserted away
        fraction = float((deltas > 0).mean())
        record_property("large_165pt_score_increase_fraction", f"{fraction:.4f}")
        assert fraction >= WEEKLY_REFINED_SCORE_FRACTION
        assert deltas.mean() > SMALL_REFINED_SCORE_MEAN_DELTA


# ==================== Performance of the new default ================= #


class TestRefinedPerformance:
    def test_the_throughput_floor_on_the_refined_path(self, record_property):
        # the constitution's floor, now asserted through the default
        # (refined) path; measured 65-70 patterns per second per core
        # at ``bw`` 68, so about 33 times the floor
        indexer = ni_indexer()
        patterns = ni_signal().data.reshape((-1, 60, 60))
        with dask.config.set(num_workers=1):
            indexer.index_patterns(patterns, chunksize=9, progressbar=False)
            start = time.perf_counter()
            results = indexer.index_patterns(patterns, chunksize=9, progressbar=False)
            elapsed = time.perf_counter() - start
        per_second = 9 / elapsed
        record_property("refined_patterns_per_second_bw68", f"{per_second:.1f}")
        # the timed run refined nine patterns, so the rate is not
        # the rate of failing them
        assert (results["phase_id"][:, 0] == 0).all()
        assert per_second >= THROUGHPUT_FLOOR

    def test_the_refined_to_coarse_ratio_is_recorded(self, record_property):
        # recorded, never asserted: measured 1.05-1.27 times the
        # coarse wall time here against about 1.7 for the compiled
        # C++, which is only relatively cheaper because our coarse
        # path is slower
        patterns = ni_signal().data.reshape((-1, 60, 60))
        elapsed = {}
        for refine in (False, True):
            indexer = ni_indexer(refine=refine)
            with dask.config.set(num_workers=1):
                indexer.index_patterns(patterns, chunksize=9, progressbar=False)
                start = time.perf_counter()
                results = indexer.index_patterns(
                    patterns, chunksize=9, progressbar=False
                )
                elapsed[refine] = time.perf_counter() - start
            assert (results["phase_id"][:, 0] == 0).all()
        record_property(
            "refined_over_coarse_ratio",
            f"{elapsed[True] / elapsed[False]:.2f}x "
            f"({1e3 * elapsed[False] / 9:.2f} -> {1e3 * elapsed[True] / 9:.2f} "
            "ms per pattern)",
        )

    def test_the_per_stage_refine_cost_is_recorded(self, record_property):
        # the other half of the D8 record: what the refinement costs
        # beside the coarse correlation it follows, rather than what
        # the whole run costs.  Recorded, never asserted -- measured
        # 1.39 ms against 13.2 ms warm at ``bw`` 68
        indexer = ni_indexer()
        correlator = indexer.correlators[0].clone()
        gln = pattern_spectrum(indexer, np.asarray(ni_signal().data[1, 1]))
        coarse, coarse_score = correlator.correlate(gln)
        refined, refined_score = correlator.refine_zyz(gln, coarse)
        # both stages did their work, so the times below are the
        # times of a refinement and not of a refusal
        assert refined_score > 0
        assert not np.array_equal(np.asarray(refined), np.asarray(coarse))
        reps = 5
        start = time.perf_counter()
        for _ in range(reps):
            correlator.correlate(gln)
        coarse_ms = 1e3 * (time.perf_counter() - start) / reps
        start = time.perf_counter()
        for _ in range(reps):
            correlator.refine_zyz(gln, coarse)
        refine_ms = 1e3 * (time.perf_counter() - start) / reps
        record_property(
            "per_stage_ms_bw68",
            f"coarse correlate {coarse_ms:.2f} ms, refine and denominator "
            f"{refine_ms:.2f} ms, ratio {1 + refine_ms / coarse_ms:.2f}x, "
            f"score {coarse_score:.5f} -> {refined_score:.5f}",
        )
