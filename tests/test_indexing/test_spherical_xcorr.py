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

"""Tests of ``kikuchipy.indexing._spherical._xcorr``.

Covers every named assertion of
``specs/2026-08-17-spherical-cross-correlation/validation.md``:

- Sizes and layouts: the ``bw -> slP`` table of the ported
  ``fft::fastSize()`` (and a guard against
  :func:`scipy.fft.next_fast_len`), the ``[k, n, m]`` buffers, the
  shared ``pi/2`` Wigner table and every ``ValueError``,
  ``RuntimeError`` and ``NotImplementedError`` path.
- The spectrum kernel against two oracles which do not share a line
  of code with the port: a triple sum over Phase 3's ``wigner_D``
  (which also rules out the three other index and conjugation
  placements) and the inner product with Phase 3's
  ``rotate_harmonics``; plus the systemic zeros, the zero padding,
  buffer reuse, the pad region garbage test, the flag exactness and
  the ``1e300`` table sentinel.
- The separable inverse against :func:`scipy.fft.irfftn` and against
  itself without plane skipping, and the FFT call recording test
  which pins ``workers=1`` and the separable structure.
- The peak search, the two documented C++ ``extractNeighborhood``
  defects in both ``emsphinx_compatible`` settings (including the
  even-``slP`` out-of-row and out-of-slice reads and the clamped
  past-the-end reads), the tri-quadratic interpolation on synthetic
  blocks (including the ``x[2]`` bounds bug, the wrong monomials of
  the C++ ``vPeak`` expression and the flat ``det == 0`` block) and
  the index <-> Euler round trips.
- Ports of ``EMSphInx/test/sht/sht_xcorr.cpp`` lines 108-395
  (``randomSphere``, ``randomPair``, ``testCorr``, ``testNCorr``,
  ``runTests``) restricted to the coarse path, with the
  symmetry-reduced misorientation metric of the spec's D4.
- Real data: the Ni master pattern autocorrelation at ``bw`` 68 and
  64, the rotated master recovery and the Phase 2 D7 gate.
- ``.py_func`` of every kernel, the Numba compilation flags
  (including ``error_model="numpy"`` on ``_interpolate_maxima``
  alone) and recorded timing and memory baselines.
"""

import inspect
import math
import time
import tracemalloc
import warnings

import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import O, _groups
import pytest
import scipy.fft

import kikuchipy as kp
from kikuchipy.indexing._spherical import (
    _euler,
    _fft,
    _grid,
    _sht,
    _symmetry,
    _wigner,
    _xcorr,
)
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)

EPS = float(np.finfo(np.float64).eps)

# Point groups of ``sht_xcorr.cpp`` lines 331-340 under their orix
# names, and the flags they must map to
POINT_GROUPS = ["112", "11m", "2/m", "3", "4", "4/m", "6", "6/m"]
POINT_GROUP_FLAGS = {
    "112": (2, False),
    "11m": (1, True),
    "2/m": (2, True),
    "3": (3, False),
    "4": (4, False),
    "4/m": (4, True),
    "6": (6, False),
    "6/m": (6, True),
}

# Bandwidth -> padded side length, the D1 table
SIDE_LENGTHS = {
    53: 105,
    54: 108,
    55: 110,
    56: 112,
    57: 117,
    58: 117,
    59: 117,
    60: 120,
    61: 121,
    62: 125,
    63: 125,
    64: 128,
    68: 135,
    88: 175,
    113: 225,
    123: 245,
    158: 315,
}

# Bandwidth -> number of stored beta slices ``bwP``, as literals so
# that ``half_side_length`` is pinned by something other than the
# expression it is computed from
HALF_SIDE_LENGTHS = {
    53: 53,
    54: 55,
    55: 56,
    56: 57,
    57: 59,
    58: 59,
    59: 59,
    60: 61,
    61: 61,
    62: 63,
    63: 63,
    64: 65,
    68: 68,
    88: 88,
    113: 113,
    123: 123,
    158: 158,
}

# The ``runTests`` size list (lines 296-299), split into the default
# suite and the weekly one
DEFAULT_SIZES = [53, 54, 55, 56, 57, 58, 59, 60, 62, 64, 68]
WEEKLY_SIZES = [88, 113, 123, 158]
SIZES = DEFAULT_SIZES + [
    pytest.param(bandwidth, marks=pytest.mark.weekly) for bandwidth in WEEKLY_SIZES
]

# The point group sweeps run at three bandwidths by default (odd,
# even and odd padded side length, the last being the worst measured
# case) and over the whole C++ range 53..63 weekly
DEFAULT_GROUP_BANDWIDTHS = [53, 60, 63]
WEEKLY_GROUP_BANDWIDTHS = [54, 55, 56, 57, 58, 59, 61, 62]
GROUP_BANDWIDTHS = DEFAULT_GROUP_BANDWIDTHS + [
    pytest.param(bandwidth, marks=pytest.mark.weekly)
    for bandwidth in WEEKLY_GROUP_BANDWIDTHS
]

# Every Numba kernel of the module, for the flag and py_func tests
KERNEL_NAMES = [
    "_extract_neighborhood",
    "_find_peak",
    "_interpolate_maxima",
    "_scale_and_find_peak",
    "_xcorr_spectrum",
]

# Point groups by orix name, for the symmetry reduced metric
GROUPS = {group.name: group for group in _groups}

# The Ni master pattern bandwidths of the real data tests
NI_BANDWIDTH = 68
NI_BANDWIDTH_EVEN = 64


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
    undecorated stub. Every caller first asserts that the kernel
    does carry a ``py_func``, so that an implementation without
    ``@njit`` fails loudly instead of silently comparing a function
    to itself.
    """
    return getattr(kernel, "py_func", kernel)


def random_alm(bandwidth, rng, n_fold=1, mirror=False):
    """Return a random spectrum of a real function with the given
    symmetry, entries with ``l < m`` zero and the ``m == 0`` row
    real.
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
    """Return the ZYZ angles of a random rotation.

    The quaternion is ``(|u|, u, u, u)`` normalized, i.e.
    ``randomRotation()`` of ``sht_xcorr.cpp`` lines 128-134.
    """
    quaternion = rng.uniform(-1, 1, 4)
    quaternion[0] = abs(quaternion[0])
    quaternion /= np.linalg.norm(quaternion)
    return _euler.quaternion_to_zyz(quaternion)


def random_pair_on_grid(bandwidth, n_fold, mirror, rng):
    """Return a Legendre transformer and a random spectrum built the
    way ``randomSphere()``/``randomPair()`` do.

    Port of ``sht_xcorr.cpp`` lines 108-170 with the grid side
    length of line 152: a uniform ``(-1, 1)`` northern hemisphere,
    either mirrored or an independent southern one whose equator
    ring is copied from the north (``matchEquator()``), and the
    exact zeroing of the rows ``m % n_fold != 0`` of lines 158-166.
    ``MasterPattern::makeNFold()`` (line 124), the real space
    symmetrization which precedes that zeroing, is deliberately not
    ported: any spectrum with the right systematic zeros serves
    every assertion here, so the numbers of the C++ test are not
    reproduced value for value.
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


def cap_mask(dim, center=(0.0, -0.5, 0.85), half_angle_deg=55.0):
    """Return a binary spherical cap mask on a square Legendre
    grid.
    """
    normals = _grid.legendre_normals(dim)
    axis = np.asarray(center, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cosine = math.cos(math.radians(half_angle_deg))
    north = (normals @ axis > cosine).astype(np.float64)
    flipped = normals.copy()
    flipped[..., 2] *= -1
    south = (flipped @ axis > cosine).astype(np.float64)
    return north, south


def _orientations(zyz, symmetry):
    """Return the sample to crystal orientations of ZYZ triples.

    The symmetry of the master sits on the **left** of
    ``~Rotation(zyz_to_quaternion(zyz))``, which is where
    :class:`orix.quaternion.Orientation` puts it, see the spec's D4.
    """
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


def misorientation_deg_many(zyz_a, zyz_b, name="1"):
    """Return the symmetry reduced misorientations in degrees of a
    stack of ZYZ triples against one or as many others.
    """
    symmetry = GROUPS[name]
    angles = _orientations(zyz_a, symmetry).angle_with(
        _orientations(zyz_b, symmetry), degrees=True
    )
    return np.atleast_1d(angles).ravel()


def total_power(alm):
    """Return ``<f, f>``, the real inner product of a spectrum with
    itself with the ``m > 0`` orders doubled.
    """
    return inner_product(alm, alm)


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


def triple_sum_oracles(flm, gln, zyz):
    """Return the four candidate triple sums over Phase 3's
    ``wigner_D``.

    ``"C"`` is the placement the port must match,
    ``sum_l sum_{m'} sum_{n'} f^l_{m'} conj(g^l_{n'})
    D^l_{n',m'}(zyz)``; ``"A"`` swaps the two orders and ``"B"`` and
    ``"D"`` conjugate ``A`` and ``C``. Only the module's own
    ``wigner_D`` is used, which Phase 3 pinned against Mathematica.
    """
    bandwidth = flm.shape[0]
    angles = np.asarray(zyz, dtype=np.float64)
    totals = {"A": 0j, "B": 0j, "C": 0j, "D": 0j}
    for degree in range(bandwidth):
        for order_f in range(-degree, degree + 1):
            f = _coefficient(flm, degree, order_f)
            for order_g in range(-degree, degree + 1):
                g = np.conjugate(_coefficient(gln, degree, order_g))
                swapped = _wigner.wigner_D(degree, order_f, order_g, angles)
                straight = _wigner.wigner_D(degree, order_g, order_f, angles)
                totals["A"] += f * g * swapped
                totals["B"] += f * g * np.conjugate(swapped)
                totals["C"] += f * g * straight
                totals["D"] += f * g * np.conjugate(straight)
    return totals


def full_cube(fxc):
    """Return the whole ``slP ** 3`` cross-correlation cube, the
    oracle of the separable inverse.
    """
    side_length = fxc.shape[0]
    return scipy.fft.irfftn(
        fxc,
        s=(side_length,) * 3,
        axes=(0, 1, 2),
        norm="forward",
        workers=1,
    )


def flat_index(k, n, m, side_length):
    """Return the flat index of a grid point of ``xc``."""
    return (k * side_length + n) * side_length + m


def unflatten(index, side_length):
    """Return the ``(k, n, m)`` of a flat index of ``xc``."""
    k, remainder = divmod(int(index), side_length * side_length)
    n, m = divmod(remainder, side_length)
    return k, n, m


def literal_indices(slp, bwp, k0, n0, m0):
    """Return the three index slots of the C++
    ``extractNeighborhood<1>``.

    Literal transcription of ``sht_xcorr.hpp`` lines 512-533: the
    periodic wrap, then the glide of the slots whose ``k`` lands in
    the unstored half, with the alpha shift of line 530 and the
    gamma shift of line 531 applied **per slot index**.
    """
    inds = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for axis, center in enumerate((k0, n0, m0)):
        inds[axis][1] = center
        inds[axis][0] = slp - 1 if center == 0 else center - 1
        inds[axis][2] = 0 if center + 1 == slp else center + 1
    for i in range(3):
        if inds[0][i] >= bwp:
            inds[2][i] = inds[2][i] + bwp - 1 if inds[2][i] < bwp else inds[2][i] - bwp
            inds[1][i] = inds[1][i] + bwp - 1 if inds[1][i] < bwp else inds[1][i] - bwp
            inds[0][i] = slp - inds[0][i]
    return inds


def literal_offsets(slp, bwp, k0, n0, m0):
    """Return the 27 unclamped flat offsets the C++ would read."""
    inds = literal_indices(slp, bwp, k0, n0, m0)
    return [
        inds[0][k] * slp * slp + inds[1][n] * slp + inds[2][m]
        for k in range(3)
        for n in range(3)
        for m in range(3)
    ]


def literal_neighborhood(xc, k0, n0, m0):
    """Return the faithful neighbourhood, read from the flat buffer
    with every offset bounded by ``size - 1``.
    """
    bwp, slp = xc.shape[0], xc.shape[1]
    flat = xc.reshape(-1)
    size = flat.size
    inds = literal_indices(slp, bwp, k0, n0, m0)
    block = np.empty((3, 3, 3))
    for k in range(3):
        for n in range(3):
            for m in range(3):
                offset = inds[0][k] * slp * slp + inds[1][n] * slp + inds[2][m]
                block[k, n, m] = flat[min(offset, size - 1)]
    return block


def per_plane_neighborhood(xc, k0, n0, m0):
    """Return the neighbourhood with the per-plane glide, i.e. the
    ``emsphinx_compatible=False`` block.
    """
    bwp, slp = xc.shape[0], xc.shape[1]
    shift = slp // 2
    block = np.empty((3, 3, 3))
    for dk in (-1, 0, 1):
        k = (k0 + dk) % slp
        for dn in (-1, 0, 1):
            n = (n0 + dn) % slp
            for dm in (-1, 0, 1):
                m = (m0 + dm) % slp
                if k >= bwp:
                    block[dk + 1, dn + 1, dm + 1] = xc[
                        slp - k, (n + shift) % slp, (m + shift) % slp
                    ]
                else:
                    block[dk + 1, dn + 1, dm + 1] = xc[k, n, m]
    return block


def periodic_block(cube, k0, n0, m0):
    """Return the 3 x 3 x 3 block of a cube with periodic wrap in
    all three directions.
    """
    ks = [(k0 - 1) % cube.shape[0], k0, (k0 + 1) % cube.shape[0]]
    ns = [(n0 - 1) % cube.shape[1], n0, (n0 + 1) % cube.shape[1]]
    ms = [(m0 - 1) % cube.shape[2], m0, (m0 + 1) % cube.shape[2]]
    return cube[np.ix_(ks, ns, ms)]


def quadratic_block(center, peak=5.0, curvature=(-1.0, -1.0, -1.0), cross=0.2):
    """Return the 27 samples of a quadratic whose stationary point
    is exactly at ``center``.

    ``f(z, y, x) = peak + c0 dz^2 + c1 dy^2 + c2 dx^2 + cross dz dx``
    with ``d`` the offsets from ``center``; the cross term is a
    product of the two shifted coordinates, so it does not move the
    stationary point.
    """
    z0, y0, x0 = center
    block = np.empty((3, 3, 3))
    for k in range(3):
        for n in range(3):
            for m in range(3):
                dz, dy, dx = k - 1 - z0, n - 1 - y0, m - 1 - x0
                block[k, n, m] = (
                    peak
                    + curvature[0] * dz * dz
                    + curvature[1] * dy * dy
                    + curvature[2] * dx * dx
                    + cross * dz * dx
                )
    return block


def random_tri_quadratic(rng, scale=0.05):
    """Return the 27 coefficients ``a[k, j, i]`` of
    ``f = a[k, j, i] z^k y^j x^i`` with a negative definite
    quadratic part and small random higher-order terms.
    """
    a = np.zeros((3, 3, 3))
    fixed = {(0, 0, 0): 5.0, (2, 0, 0): -1.0, (0, 2, 0): -1.2, (0, 0, 2): -0.9}
    for k in range(3):
        for j in range(3):
            for i in range(3):
                a[k, j, i] = fixed.get((k, j, i), scale * rng.uniform(-1, 1))
    return a


def evaluate_tri_quadratic(a, x):
    """Return ``f(x)`` of the 27 coefficients, ``x = (z, y, x)``."""
    z, y, xv = float(x[0]), float(x[1]), float(x[2])
    value = 0.0
    for k in range(3):
        for j in range(3):
            for i in range(3):
                value += a[k, j, i] * z**k * y**j * xv**i
    return value


def cpp_vpeak(a, x):
    """Return the C++ ``vPeak`` expression of the 27 coefficients.

    Literal transcription of ``sht_xcorr.hpp`` lines 1354-1364,
    which is **not** ``f(x)`` of the fit: the three mixed cubic
    terms are evaluated at the wrong monomials, ``a112 xy x[2]``,
    ``a211 yz x[0]`` and ``a121 zx x[1]``, which all reduce to
    ``z y x`` instead of ``z y x^2``, ``z^2 y x`` and ``z y^2 x``
    (the local names are the C++ ones: ``x = (z, y, x)``, so
    ``xx = z^2``, ``zz = x^2``, ``xy = z y`` and ``yz = y x``).
    The Hessian and the gradient use the correct monomials, so
    only the returned value is affected.
    """
    z, y, xv = float(x[0]), float(x[1]), float(x[2])
    xx, yy, zz = z * z, y * y, xv * xv
    xy, yz, zx = z * y, y * xv, xv * z
    return (
        a[0, 0, 0]
        + a[1, 1, 1] * z * y * xv
        + a[2, 2, 2] * xx * yy * zz
        + a[1, 0, 0] * z
        + a[0, 1, 0] * y
        + a[0, 0, 1] * xv
        + a[2, 0, 0] * xx
        + a[0, 2, 0] * yy
        + a[0, 0, 2] * zz
        + a[1, 1, 0] * xy
        + a[0, 1, 1] * yz
        + a[1, 0, 1] * zx
        + a[1, 2, 0] * z * yy
        + a[0, 1, 2] * y * zz
        + a[2, 0, 1] * xv * xx
        + a[2, 1, 0] * xx * y
        + a[0, 2, 1] * yy * xv
        + a[1, 0, 2] * zz * z
        + a[2, 2, 0] * xx * yy
        + a[0, 2, 2] * yy * zz
        + a[2, 0, 2] * zz * xx
        + a[1, 1, 2] * xy * xv
        + a[2, 1, 1] * yz * z
        + a[1, 2, 1] * zx * y
        + a[1, 2, 2] * z * yy * zz
        + a[2, 1, 2] * xx * y * zz
        + a[2, 2, 1] * xx * yy * xv
    )


def tri_quadratic_gradient(a, x):
    """Return the analytic gradient of the 27 coefficient fit."""
    z, y, xv = float(x[0]), float(x[1]), float(x[2])
    gradient = np.zeros(3)
    for k in range(3):
        for j in range(3):
            for i in range(3):
                c = a[k, j, i]
                if k:
                    gradient[0] += c * k * z ** (k - 1) * y**j * xv**i
                if j:
                    gradient[1] += c * j * z**k * y ** (j - 1) * xv**i
                if i:
                    gradient[2] += c * i * z**k * y**j * xv ** (i - 1)
    return gradient


def sample_tri_quadratic(a):
    """Return the 3 x 3 x 3 samples of the 27 coefficient fit."""
    block = np.empty((3, 3, 3))
    for k in range(3):
        for n in range(3):
            for m in range(3):
                block[k, n, m] = evaluate_tri_quadratic(a, (k - 1, n - 1, m - 1))
    return block


def correlator_with_rotated_pair(bandwidth, seed=5, n_fold=1, mirror=False):
    """Return a correlator whose ``xc`` holds the cross-correlation
    of a random spectrum with a rotated copy of itself.

    The recipe and the seed are those of the drafting probe, so that
    the recorded neighbourhood numbers are reproducible.
    """
    rng = np.random.default_rng(seed)
    correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
    flm = random_alm(bandwidth, rng, n_fold, mirror)
    gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
    correlator.compute(flm, gln, n_fold, mirror)
    return correlator


def masked_case(bandwidth, n_fold, mirror, rng, mask="wedge"):
    """Return the ingredients of the ``testNCorr()`` recipe.

    Port of ``sht_xcorr.cpp`` lines 212-260: a band-limited random
    reference, the transform of its square, and a band-limited
    binary mask.
    """
    transform, flm = random_pair_on_grid(bandwidth, n_fold, mirror, rng)
    dim = transform.dim
    north, south = transform.synthesize(flm)
    flm2 = transform.analyze(north**2, south**2)
    mask_north, mask_south = wedge_mask(dim) if mask == "wedge" else cap_mask(dim)
    mlm = transform.analyze(mask_north, mask_south)
    mask_north, mask_south = transform.synthesize(mlm)
    return transform, flm, flm2, mlm, (mask_north, mask_south)


def masked_pattern(transform, flm, mask, zyz):
    """Return the spectrum of the masked, rotated reference."""
    north, south = transform.synthesize(_wigner.rotate_harmonics(flm, zyz))
    return transform.analyze(north * mask[0], south * mask[1])


def cell_deg(side_length):
    """Return one grid cell in degrees."""
    return 360.0 / side_length


def tier_tolerance_deg(beta, side_length):
    """Return the tiered coarse tolerance of the spec's D10.

    Half a cell in general, a whole cell when the applied ``beta``
    is within one cell of ``0`` or of ``pi``, where the two C++
    neighbourhood defects bite.
    """
    cell = 2 * math.pi / side_length
    wrapped = abs(_euler.wrap_beta(float(beta)))
    if min(wrapped, abs(wrapped - math.pi)) < cell:
        return cell_deg(side_length)
    return cell_deg(side_length) / 2


def assert_zyz_in_range(zyz, side_length, emsphinx_compatible):
    """Assert the returned angles lie in the intervals of D7."""
    cell = 2 * math.pi / side_length
    slack = 1e-9
    alpha, beta, gamma = (float(value) for value in zyz)
    assert -math.pi - cell - slack <= beta <= cell + slack
    assert -math.pi / 2 - cell - slack <= gamma <= 3 * math.pi / 2 + slack
    if emsphinx_compatible:
        # the x[2] bug leaves alpha unchecked by interpPeak()
        assert math.isfinite(alpha)
    else:
        assert -math.pi / 2 - cell - slack <= alpha <= 3 * math.pi / 2 + slack


def oracle_grid_points(slp, bwp, rng, count=60):
    """Return grid points including every corner of the stored half
    the oracles must cover.
    """
    points = [
        (0, 0, 0),
        (0, 0, slp - 1),
        (0, slp - 1, 0),
        (0, slp - 1, slp - 1),
        (bwp - 1, 0, 0),
        (bwp - 1, 0, slp - 1),
        (bwp - 1, slp - 1, 0),
        (bwp - 1, slp - 1, slp - 1),
    ]
    while len(points) < count:
        point = (
            int(rng.integers(bwp)),
            int(rng.integers(slp)),
            int(rng.integers(slp)),
        )
        if point not in points:
            points.append(point)
    return points


# ----------------------------- Fixtures ----------------------------- #


@pytest.fixture(scope="module")
def nickel_master_pattern():
    """Return the shipped Ni master pattern in both hemispheres."""
    return kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )


@pytest.fixture(scope="module")
def nickel_harmonics(nickel_master_pattern):
    """Return the ``bw`` 68 harmonics of the Ni master pattern."""
    return MasterPatternHarmonics.from_master_pattern(
        nickel_master_pattern, bandwidth=NI_BANDWIDTH
    )


@pytest.fixture(scope="module")
def nickel_harmonics_even(nickel_master_pattern):
    """Return the ``bw`` 64 harmonics of the Ni master pattern, i.e.
    the even ``slP`` 128 grid on which all 24 proper cubic operators
    are exact grid points.
    """
    return MasterPatternHarmonics.from_master_pattern(
        nickel_master_pattern, bandwidth=NI_BANDWIDTH_EVEN
    )


@pytest.fixture(scope="module")
def nickel_harmonics_plain(nickel_master_pattern):
    """Return the ``bw`` 68 harmonics without EMSphInx'
    normalization quirks, i.e. the second half of the D7 gate.
    """
    return MasterPatternHarmonics.from_master_pattern(
        nickel_master_pattern, bandwidth=NI_BANDWIDTH, emsphinx_compatible=False
    )


# ------------------------------- Tests ------------------------------ #


class TestSizes:
    @pytest.mark.parametrize("bandwidth", [b for b in SIDE_LENGTHS if b <= 68])
    def test_side_lengths_of_a_constructed_correlator(self, bandwidth):
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        assert correlator.bandwidth == bandwidth
        assert correlator.side_length_unpadded == 2 * bandwidth - 1
        assert correlator.side_length == SIDE_LENGTHS[bandwidth]
        assert correlator.side_length == _fft.fast_size(2 * bandwidth - 1)
        assert correlator.half_side_length == correlator.side_length // 2 + 1
        assert correlator.half_side_length == HALF_SIDE_LENGTHS[bandwidth]

    @pytest.mark.parametrize("bandwidth", [88, 113, 123, 158])
    def test_side_lengths_of_the_large_bandwidths(self, bandwidth):
        # the arithmetic alone, so that the table is pinned without
        # paying for a 43-92 MB spectrum buffer; both sides are
        # literals, since ``slP // 2 + 1 == slP // 2 + 1`` pins
        # nothing
        side_length = _fft.fast_size(2 * bandwidth - 1)
        assert side_length == SIDE_LENGTHS[bandwidth]
        assert side_length // 2 + 1 == HALF_SIDE_LENGTHS[bandwidth]

    def test_scipy_next_fast_len_is_not_the_rule(self):
        # a "simplification" to scipy's rule must fail here: it is
        # 11-smooth for real transforms and 7-smooth otherwise
        assert scipy.fft.next_fast_len(105, real=True) == 108
        assert _fft.fast_size(105) == 105
        assert scipy.fft.next_fast_len(117) == 120
        assert _fft.fast_size(117) == 117

    def test_buffer_layouts(self):
        bandwidth = 16
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        slp, bwp = correlator.side_length, correlator.half_side_length
        assert correlator.fxc.shape == (slp, slp, bwp)
        assert correlator.fxc.dtype == np.complex128
        assert correlator.fxc.flags.c_contiguous
        assert np.all(correlator.fxc == 0)
        assert correlator.xc is None

        rng = np.random.default_rng(0)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        xc = correlator.compute(flm, gln, 1, False)
        assert xc.shape == (bwp, slp, slp)
        assert xc.dtype == np.float64
        assert xc.flags.c_contiguous
        assert correlator.xc is xc

    def test_wigner_table_is_the_transposed_phase_three_table(self):
        bandwidth = 12
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        table = correlator.wigner_d_half_pi
        expected = _wigner.wigner_d_half_pi_table(bandwidth, True)
        assert table.shape == (bandwidth,) * 3
        nan = np.isnan(table)
        assert np.array_equal(nan, np.isnan(expected))
        index = np.arange(bandwidth)
        undefined = index[np.newaxis, np.newaxis, :] < np.maximum(
            index[:, np.newaxis, np.newaxis], index[np.newaxis, :, np.newaxis]
        )
        assert np.array_equal(nan, undefined)
        assert np.array_equal(table[~nan], expected[~nan])

    def test_bandwidth_below_one_raises(self):
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(0)

    @pytest.mark.parametrize("shape", [(4, 4, 3), (5, 5, 5), (4, 4)])
    def test_wigner_table_of_the_wrong_shape_raises(self, shape):
        table = np.full(shape, np.nan)
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(4, wigner_d_half_pi=table)

    def test_wigner_table_of_the_wrong_dtype_raises(self):
        table = _wigner.wigner_d_half_pi_table(4, True).astype(np.float32)
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(4, wigner_d_half_pi=table)

    def test_non_contiguous_wigner_table_raises(self):
        table = np.asfortranarray(_wigner.wigner_d_half_pi_table(4, True))
        assert not table.flags.c_contiguous
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(4, wigner_d_half_pi=table)

    @pytest.mark.parametrize("fill", [0.0, 1.0])
    def test_wigner_table_without_the_nan_tripwire_raises(self, fill):
        # Phase 3's out= idiom: slot [0, bw - 1, 0] must be NaN, so
        # that a np.zeros or np.empty buffer is refused
        table = np.full((4, 4, 4), fill)
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(4, wigner_d_half_pi=table)

    def test_untransposed_wigner_table_raises(self):
        # the two layouts differ by (-1)^(k - m) in half of their
        # slots and would silently give a wrong correlation; slot
        # [1, 0, 1] is d^1_{0,1}(pi/2) = -1/sqrt(2) transposed and
        # d^1_{1,0}(pi/2) = +1/sqrt(2) untransposed
        table = _wigner.wigner_d_half_pi_table(4, False)
        assert table[1, 0, 1] > 0
        assert _wigner.wigner_d_half_pi_table(4, True)[1, 0, 1] < 0
        with pytest.raises(ValueError):
            _xcorr.SphericalCrossCorrelator(4, wigner_d_half_pi=table)

    def test_a_table_of_another_instance_is_shared(self):
        first = _xcorr.SphericalCrossCorrelator(8)
        second = _xcorr.SphericalCrossCorrelator(
            8, wigner_d_half_pi=first.wigner_d_half_pi
        )
        assert second.wigner_d_half_pi is first.wigner_d_half_pi

    def test_no_warning_for_a_side_length_with_a_factor_thirteen(self):
        # a regression guard for constitution amendment 0.2: the
        # measured pocketfft penalty of slP 117 = 9 x 13 is recorded
        # in the class docstring, not warned about, and bw 59 is a
        # recommended fast_bandwidths() value with fast_size(117) 117
        _xcorr.SphericalCrossCorrelator(4)  # warm the Numba cache
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            correlator = _xcorr.SphericalCrossCorrelator(57)
        assert correlator.side_length == 117
        # only warnings about the transform size count: "always"
        # records every warning raised inside the block, and the
        # suite's third party imports (diffpy.structure) or a
        # lazily built table elsewhere would otherwise fail this
        assert [
            str(caught_warning.message)
            for caught_warning in caught
            if any(
                word in str(caught_warning.message).lower()
                for word in ("13", "radix", "slp", "side length", "fft size")
            )
        ] == []

    @pytest.mark.parametrize("bandwidth", [57, 58, 59])
    def test_the_factor_thirteen_side_length_is_the_ported_rule(self, bandwidth):
        assert _fft.fast_size(2 * bandwidth - 1) == 117

    def test_interp_peak_before_compute_raises(self):
        correlator = _xcorr.SphericalCrossCorrelator(8)
        with pytest.raises(RuntimeError):
            correlator.interp_peak(0)

    def test_refine_raises_for_the_plain_correlator(self):
        rng = np.random.default_rng(0)
        flm = random_alm(8, rng)
        gln = random_alm(8, rng)
        correlator = _xcorr.SphericalCrossCorrelator(8)
        with pytest.raises(NotImplementedError, match="Phase 7"):
            correlator.correlate(flm, gln, 1, False, refine=True)

    def test_refine_raises_for_the_normalized_correlator(self):
        rng = np.random.default_rng(0)
        transform, flm, flm2, mlm, mask = masked_case(17, 1, False, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            17, flm, flm2, 1, False, mlm
        )
        gln = masked_pattern(transform, flm, mask, random_zyz(rng))
        with pytest.raises(NotImplementedError, match="Phase 7"):
            correlator.correlate(gln, refine=True)

    @pytest.mark.parametrize(
        "n_fold, mirror",
        [(True, False), (False, False), (0, False), (-1, False), (1, 0), (1, 1)],
    )
    def test_bad_flags_raise(self, n_fold, mirror):
        rng = np.random.default_rng(0)
        flm = random_alm(8, rng)
        gln = random_alm(8, rng)
        correlator = _xcorr.SphericalCrossCorrelator(8)
        with pytest.raises(ValueError):
            correlator.compute(flm, gln, n_fold, mirror)

    @pytest.mark.parametrize("shape", [(8, 7), (7, 7), (8, 8, 8), (8,)])
    def test_spectra_of_the_wrong_shape_raise(self, shape):
        rng = np.random.default_rng(0)
        good = random_alm(8, rng)
        bad = np.zeros(shape, dtype=np.complex128)
        correlator = _xcorr.SphericalCrossCorrelator(8)
        with pytest.raises(ValueError):
            correlator.compute(bad, good, 1, False)
        with pytest.raises(ValueError):
            correlator.compute(good, bad, 1, False)

    @pytest.mark.parametrize("shape", [(8, 7), (7, 7), (8, 8, 8), (8,)])
    @pytest.mark.parametrize("position", ["flm", "flm2", "mlm"])
    def test_normalized_spectra_of_the_wrong_shape_raise(self, shape, position):
        bandwidth = 8
        rng = np.random.default_rng(0)
        good = random_alm(bandwidth, rng)
        spectra = {name: good for name in ("flm", "flm2", "mlm")}
        spectra[position] = np.zeros(shape, dtype=np.complex128)
        with pytest.raises(ValueError):
            _xcorr.NormalizedSphericalCrossCorrelator(
                bandwidth,
                spectra["flm"],
                spectra["flm2"],
                1,
                False,
                spectra["mlm"],
            )

    @pytest.mark.parametrize("shape", [(17, 16), (16, 16), (17, 17, 17), (17,)])
    def test_a_normalized_pattern_of_the_wrong_shape_raises(self, shape):
        bandwidth = 17
        rng = np.random.default_rng(0)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        with pytest.raises(ValueError):
            correlator.correlate(np.zeros(shape, dtype=np.complex128))


class TestSpectrumKernel:
    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror",
        [(6, 1, False), (8, 1, False), (8, 2, True), (8, 3, False)],
    )
    def test_triple_sum_oracle(self, bandwidth, n_fold, mirror, record_property):
        # an unrelated gln, so that a transposed table read, a
        # missing conjugation or swapped alpha/gamma axes cannot
        # hide behind a peak
        rng = np.random.default_rng(1)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        xc = correlator.compute(flm, gln, n_fold, mirror)
        points = oracle_grid_points(
            correlator.side_length,
            correlator.half_side_length,
            np.random.default_rng(11),
        )
        worst = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        worst_imaginary = 0.0
        for k, n, m in points:
            zyz = _xcorr.index_to_euler((k, n, m), correlator.side_length)
            totals = triple_sum_oracles(flm, gln, zyz)
            for name, value in totals.items():
                worst[name] = max(worst[name], abs(value.real - xc[k, n, m]))
            worst_imaginary = max(worst_imaginary, abs(totals["C"].imag))
        record_property(f"triple_sum_bw{bandwidth}_nf{n_fold}", f"{worst['C']:.3e}")
        record_property(f"triple_sum_scale_bw{bandwidth}", f"{np.abs(xc).max():.3f}")
        assert worst["C"] <= 1e-11
        assert worst_imaginary <= 1e-13
        for name in ("A", "B", "D"):
            assert worst[name] > 1.0, f"placement {name} must be ruled out"

    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror",
        [
            (16, 1, False),
            (24, 4, True),
            (32, 3, False),
            pytest.param(68, 4, True, marks=pytest.mark.weekly),
        ],
    )
    def test_inner_product_oracle(self, bandwidth, n_fold, mirror, record_property):
        rng = np.random.default_rng(2)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        xc = correlator.compute(flm, gln, n_fold, mirror)
        scale = float(np.abs(xc).max())
        points = oracle_grid_points(
            correlator.side_length,
            correlator.half_side_length,
            np.random.default_rng(12),
            count=40,
        )
        worst = 0.0
        for k, n, m in points:
            zyz = _xcorr.index_to_euler((k, n, m), correlator.side_length)
            value = inner_product(_wigner.rotate_harmonics(flm, zyz), gln)
            worst = max(worst, abs(value - xc[k, n, m]))
        record_property(f"inner_product_bw{bandwidth}", f"{worst:.3e}")
        record_property(f"inner_product_scale_bw{bandwidth}", f"{scale:.3f}")
        assert worst <= 1e-12 * scale

    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror",
        [(6, 1, False), (8, 1, False), (8, 2, True)],
    )
    def test_peak_identity(self, bandwidth, n_fold, mirror, record_property):
        # pins norm="forward" and the sign of the transform: a
        # rotated copy correlates with the reference at exactly the
        # total power of the spectrum
        rng = np.random.default_rng(3)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        zyz0 = np.array([0.7, 1.1, -2.3])
        gln = _wigner.rotate_harmonics(flm, zyz0)
        power = total_power(flm)
        # a Phase 3 precondition, not an assertion about _xcorr:
        # <g, g> is the power of flm because rotate_harmonics is
        # unitary.  It is checked here because every bound below is
        # stated as a fraction of that power
        assert abs(inner_product(gln, gln) - power) <= 1e-13 * abs(power)
        record_property(f"peak_identity_bw{bandwidth}_power", f"{power:.6f}")

        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        zyz, score = correlator.correlate(flm, gln, n_fold, mirror)
        record_property(f"peak_identity_bw{bandwidth}_ratio", f"{score / power:.6f}")
        assert float(correlator.xc.max()) <= power * (1 + 1e-12)
        # the tri-quadratic under-estimates a peak which is one cell
        # wide, and the coarser the grid the more so: measured
        # 0.7897 at slP 11 (bw 6, whose argmax cell alone holds only
        # 0.710) against 0.928 and 0.951 at slP 15 and the 0.85-0.92
        # of validation.md, which was measured at bw >= 16
        lower = 0.75 if correlator.side_length < 15 else 0.8
        assert lower * power <= score <= 1.01 * power

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_an_on_grid_rotation_is_recovered_exactly(self, seed):
        bandwidth = 8
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        rng = np.random.default_rng(seed)
        flm = np.triu(
            rng.normal(size=(bandwidth, bandwidth))
            + 1j * rng.normal(size=(bandwidth, bandwidth))
        )
        flm[0] = flm[0].real
        zyz0 = _xcorr.index_to_euler((3, 5, 7), side_length)
        gln = _wigner.rotate_harmonics(flm, zyz0)
        zyz, score = correlator.correlate(flm, gln, 1, False)
        assert _xcorr.euler_to_index(zyz, side_length) == (3, 5, 7)
        assert np.abs(zyz - zyz0).max() <= 1e-12
        assert abs(score / total_power(flm) - 1) <= 1e-12

    @pytest.mark.parametrize("bandwidth", [16, 57])
    def test_systemic_zeros_are_written(self, bandwidth):
        rng = np.random.default_rng(4)
        flm = random_alm(bandwidth, rng, 4, True)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        correlator.compute(flm, gln, 4, True)
        slp = correlator.side_length
        bwp = correlator.half_side_length

        def assert_the_zeros(fxc):
            for m in range(bwp):
                if m % 4:
                    assert np.all(fxc[:, :, m] == 0), f"column {m}"
            # the zero pad slices and rows, inclusive at the upper
            # end
            assert np.all(fxc[bandwidth : slp - bandwidth + 1] == 0)
            assert np.all(fxc[:, bandwidth : slp - bandwidth + 1] == 0)
            assert np.all(fxc[slp - bandwidth] == 0)
            assert np.all(fxc[:, slp - bandwidth] == 0)

        assert_the_zeros(correlator.fxc)

        # a fresh instance starts from np.zeros, so the assertions
        # above hold whether the kernel writes those zeros or never
        # touches those slots: poison exactly the slots it is meant
        # to write with zero and compute again.  The columns
        # m in [bw, bwP) of the rows n < bw are left alone, since
        # nothing writes them (see
        # test_pad_region_garbage_is_overwritten)
        poison = 1e3 + 1e3j
        for m in range(bandwidth):
            if m % 4:
                correlator.fxc[:, :, m] = poison
        correlator.fxc[:, bandwidth : slp - bandwidth + 1] = poison
        correlator.fxc[bandwidth : slp - bandwidth + 1] = poison
        correlator.compute(flm, gln, 4, True)
        assert_the_zeros(correlator.fxc)

    @pytest.mark.parametrize("bandwidth", [9, 16])
    def test_buffers_may_be_reused_across_flags(self, bandwidth):
        rng = np.random.default_rng(5)
        first = (random_alm(bandwidth, rng, 3, True), random_alm(bandwidth, rng))
        second = (random_alm(bandwidth, rng), random_alm(bandwidth, rng))
        for order in ((3, True, 1, False), (1, False, 3, True)):
            n_fold_a, mirror_a, n_fold_b, mirror_b = order
            spectra_a = first if n_fold_a == 3 else second
            spectra_b = second if n_fold_a == 3 else first
            reused = _xcorr.SphericalCrossCorrelator(bandwidth)
            reused.compute(*spectra_a, n_fold_a, mirror_a)
            xc_reused = reused.compute(*spectra_b, n_fold_b, mirror_b)
            fresh = _xcorr.SphericalCrossCorrelator(bandwidth)
            xc_fresh = fresh.compute(*spectra_b, n_fold_b, mirror_b)
            assert np.array_equal(reused.fxc, fresh.fxc)
            assert np.array_equal(xc_reused, xc_fresh)

    @pytest.mark.parametrize("bandwidth", [16, 57])
    @pytest.mark.parametrize("n_fold, mirror", [(1, False), (4, True)])
    def test_pad_region_garbage_is_overwritten(self, bandwidth, n_fold, mirror):
        # the only test which sees the zero pad write of line 854
        # and the zero row fill of lines 833-836: on a np.zeros
        # buffer both are no-ops that no other test can distinguish.
        # bw 16 has a single pad slice (slP 32) and bw 57 four
        # (slP 117, slices 57-60); at bw 17 slP is 2 bw - 1 = 33,
        # so every slice below would be an empty selection
        rng = np.random.default_rng(6)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        fresh = _xcorr.SphericalCrossCorrelator(bandwidth)
        xc_fresh = fresh.compute(flm, gln, n_fold, mirror)
        slp, bwp = fresh.side_length, fresh.half_side_length
        # a guard against a future parametrisation which degenerates
        # into empty slices, as bw 17 (slP 33) did
        assert slp - 2 * bandwidth + 1 > 0
        assert bwp - bandwidth > 0
        for garbage in ("slices", "rows"):
            correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
            correlator.compute(flm, gln, n_fold, mirror)
            if garbage == "slices":
                correlator.fxc[bandwidth : slp - bandwidth + 1] = 1e3 + 1e3j
            else:
                correlator.fxc[:, bandwidth : slp - bandwidth + 1] = 1e3 + 1e3j
            xc = correlator.compute(flm, gln, n_fold, mirror)
            assert np.array_equal(correlator.fxc, fresh.fxc), garbage
            assert np.array_equal(xc, xc_fresh), garbage
            # the columns m in [bw, bwP) of the rows n < bw are
            # never written by anything and stay zero
            assert np.all(correlator.fxc[:, :bandwidth, bandwidth:bwp] == 0)

    def test_compute_returns_a_fresh_array_every_call(self):
        bandwidth = 12
        rng = np.random.default_rng(7)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        first = correlator.compute(
            random_alm(bandwidth, rng), random_alm(bandwidth, rng), 1, False
        )
        kept = first.copy()
        assert first.base is None
        second = correlator.compute(
            random_alm(bandwidth, rng), random_alm(bandwidth, rng), 1, False
        )
        assert first is not second
        assert correlator.xc is second
        assert np.array_equal(first, kept)

    def test_flags_are_exact_and_act_on_the_alpha_axis(self):
        bandwidth = 16
        rng = np.random.default_rng(8)
        conforming = random_alm(bandwidth, rng, 4, True)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        skipped = correlator.compute(conforming, gln, 4, True).copy()
        full = correlator.compute(conforming, gln, 1, False)
        scale = float(np.abs(full).max())
        assert np.abs(skipped - full).max() <= 1e-12 * scale

        violating = random_alm(bandwidth, rng)
        masked = violating.copy()
        for order in range(bandwidth):
            if order % 4:
                masked[order] = 0
        degree = np.arange(bandwidth)[np.newaxis, :]
        order = np.arange(bandwidth)[:, np.newaxis]
        masked[(degree + order) % 2 == 1] = 0
        skipped = correlator.compute(violating, gln, 4, True).copy()
        expected = correlator.compute(masked, gln, 1, False).copy()
        unmasked = correlator.compute(violating, gln, 1, False)
        scale = float(np.abs(expected).max())
        assert np.abs(skipped - expected).max() <= 1e-12 * scale
        assert np.abs(skipped - unmasked).max() > 1e-3 * scale

    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror",
        [(8, 1, False), (9, 3, True), (12, 2, False)],
    )
    def test_nan_table_slots_are_never_read(self, bandwidth, n_fold, mirror):
        assert hasattr(_xcorr._xcorr_spectrum, "py_func"), (
            "_xcorr_spectrum must be @njit-decorated"
        )
        rng = np.random.default_rng(9)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        table = _wigner.wigner_d_half_pi_table(bandwidth, True)
        poisoned = table.copy()
        poisoned[np.isnan(poisoned)] = 1e300
        slp = _fft.fast_size(2 * bandwidth - 1)
        bwp = slp // 2 + 1
        results = []
        for used in (table, poisoned):
            fxc = np.zeros((slp, slp, bwp), dtype=np.complex128)
            fm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
            gn = np.zeros(bandwidth, dtype=np.complex128)
            _xcorr._xcorr_spectrum(flm, gln, used, n_fold, mirror, fxc, fm, gn)
            results.append(fxc)
        assert np.array_equal(results[0], results[1])

    def test_fft_calls_are_recorded(self, monkeypatch):
        names = [
            "ifft",
            "irfft",
            "fft",
            "rfft",
            "ifftn",
            "irfftn",
            "fftn",
            "rfftn",
        ]
        recorded = []

        def wrap(name, function):
            def wrapper(*args, **kwargs):
                recorded.append((name, kwargs.get("workers")))
                return function(*args, **kwargs)

            return wrapper

        for name in names:
            # the module binds its transforms, but patch scipy.fft
            # too so that a scipy.fft.xxx(...) call style is caught
            if hasattr(_xcorr, name):
                monkeypatch.setattr(_xcorr, name, wrap(name, getattr(_xcorr, name)))
            if hasattr(scipy.fft, name):
                monkeypatch.setattr(
                    scipy.fft, name, wrap(name, getattr(scipy.fft, name))
                )

        bandwidth = 17
        rng = np.random.default_rng(10)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        for n_fold in (1, 3):
            flm = random_alm(bandwidth, rng, n_fold, False)
            gln = random_alm(bandwidth, rng)
            recorded.clear()
            correlator.compute(flm, gln, n_fold, False)
            assert len(recorded) >= 3, "the patch target must be the one used"
            assert all(workers == 1 for _, workers in recorded)
            assert {name for name, _ in recorded} <= {"ifft", "irfft"}
            assert sum(name == "ifft" for name, _ in recorded) == 2
            assert sum(name == "irfft" for name, _ in recorded) == 1


class TestInverseFFT:
    @pytest.mark.parametrize("bandwidth, n_fold", [(16, 1), (17, 3), (24, 6), (53, 4)])
    def test_separable_inverse_equals_the_full_transform(
        self, bandwidth, n_fold, record_property
    ):
        rng = np.random.default_rng(13)
        flm = random_alm(bandwidth, rng, n_fold, False)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        xc = correlator.compute(flm, gln, n_fold, False)
        oracle = full_cube(correlator.fxc)[: correlator.half_side_length]
        scale = float(np.abs(xc).max())
        worst = float(np.abs(xc - oracle).max())
        record_property(f"separable_vs_irfftn_bw{bandwidth}", f"{worst / scale:.3e}")
        assert worst <= 1e-12 * scale

    @pytest.mark.parametrize("bandwidth, n_fold", [(16, 4), (17, 3), (24, 6)])
    def test_plane_skipping_equals_no_skipping(self, bandwidth, n_fold):
        # the skipped planes are the systemic zeros the kernel
        # wrote, so the skipping is exact; the two are different
        # pocketfft call shapes, hence a few eps and not bitwise
        rng = np.random.default_rng(14)
        flm = random_alm(bandwidth, rng, n_fold, False)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        skipped = correlator.compute(flm, gln, n_fold, False)
        plain = _xcorr._inverse_fft(correlator.fxc, 1)
        scale = float(np.abs(skipped).max())
        assert np.abs(skipped - plain).max() <= 4 * EPS * scale


class TestGlide:
    def test_find_peak_equals_argmax_on_random_cubes(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            cube = rng.normal(size=(4, 5, 5))
            assert _xcorr._find_peak(cube) == int(np.argmax(cube))

    def test_find_peak_skips_nan(self):
        rng = np.random.default_rng(2)
        cube = rng.normal(size=(4, 5, 5))
        cube[0, 0, 0] = -np.inf
        cube[1, 2, 3] = np.nan
        finite = np.where(np.isfinite(cube), cube, -np.inf)
        assert _xcorr._find_peak(cube) == int(np.argmax(finite))
        assert int(np.argmax(cube)) == flat_index(1, 2, 3, 5)

    def test_find_peak_returns_the_first_of_tied_maxima(self):
        # a >= mutation returns 90 and would survive every random
        # cube; the Ni ridge cannot see it either, since every tie
        # there reduces to 0.000 degrees
        rng = np.random.default_rng(2)
        cube = rng.normal(size=(4, 5, 5))
        flat = cube.reshape(-1)
        flat[[7, 41, 90]] = flat.max() + 1
        assert _xcorr._find_peak(cube) == 7

    def test_scale_and_find_peak_returns_the_first_of_tied_maxima(self):
        rng = np.random.default_rng(2)
        cube = rng.normal(size=(4, 5, 5))
        flat = cube.reshape(-1)
        flat[[7, 41, 90]] = flat.max() + 1
        r_den = np.ones_like(cube)
        assert _xcorr._scale_and_find_peak(cube, r_den) == 7

    def test_scale_and_find_peak_scales_in_place(self):
        rng = np.random.default_rng(3)
        cube = rng.normal(size=(4, 5, 5))
        r_den = rng.uniform(0.5, 2.0, cube.shape)
        expected = cube * r_den
        index = _xcorr._scale_and_find_peak(cube, r_den)
        assert index == int(np.argmax(expected))
        assert np.array_equal(cube, expected)

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_full_cube_glide_identity_for_even_side_length(
        self, bandwidth, record_property
    ):
        correlator = correlator_with_rotated_pair(bandwidth)
        slp = correlator.side_length
        assert slp % 2 == 0
        full = full_cube(correlator.fxc)
        shift = slp // 2
        k, n, m = np.ogrid[:slp, :slp, :slp]
        glided = full[(-k) % slp, (n + shift) % slp, (m + shift) % slp]
        scale = float(np.abs(full).max())
        worst = float(np.abs(glided - full).max()) / scale
        record_property(f"glide_identity_bw{bandwidth}", f"{worst:.3e}")
        assert worst <= 1e-13

    @pytest.mark.parametrize("bandwidth", [17, 63])
    def test_no_on_grid_glide_for_odd_side_length(self, bandwidth, record_property):
        # recorded so that nobody "fixes" the odd case to an on-grid
        # glide: none exists, and the half cell shift is what the
        # emsphinx_compatible=False neighbourhood uses
        correlator = correlator_with_rotated_pair(bandwidth)
        slp = correlator.side_length
        assert slp % 2 == 1
        full = full_cube(correlator.fxc)
        scale = float(np.abs(full).max())
        k, n, m = np.ogrid[:slp, :slp, :slp]
        for shift in ((slp - 1) // 2, (slp + 1) // 2):
            glided = full[(-k) % slp, (n + shift) % slp, (m + shift) % slp]
            worst = float(np.abs(glided - full).max()) / scale
            record_property(f"glide_odd_bw{bandwidth}_shift{shift}", f"{worst:.3e}")
            assert worst > 0.1

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_per_plane_glide_matches_the_full_cube_for_even_side_length(
        self, bandwidth, record_property
    ):
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        full = full_cube(correlator.fxc)
        scale = float(np.abs(full).max())
        rng = np.random.default_rng(5)
        block = np.empty((3, 3, 3))
        worst = 0.0
        for k0 in (0, bwp - 1):
            for _ in range(20):
                n0 = int(rng.integers(slp))
                m0 = int(rng.integers(slp))
                _xcorr._extract_neighborhood(flat, slp, bwp, k0, n0, m0, False, block)
                truth = periodic_block(full, k0, n0, m0)
                worst = max(worst, float(np.abs(block - truth).max()) / scale)
                assert np.array_equal(block, per_plane_neighborhood(xc, k0, n0, m0))
        record_property(f"per_plane_vs_full_bw{bandwidth}", f"{worst:.3e}")
        assert worst <= 1e-13

    @pytest.mark.parametrize("bandwidth", [17, 63])
    def test_per_plane_glide_is_structural_for_odd_side_length(
        self, bandwidth, record_property
    ):
        # no on-grid glide exists for odd slP, and the deviation
        # from the full cube is a property of the data, not of the
        # port, so only the structure is asserted
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        full = full_cube(correlator.fxc)
        scale = float(np.abs(full).max())
        rng = np.random.default_rng(5)
        block = np.empty((3, 3, 3))
        for k0 in (0, bwp - 1):
            worst = 0.0
            for _ in range(20):
                n0 = int(rng.integers(slp))
                m0 = int(rng.integers(slp))
                _xcorr._extract_neighborhood(flat, slp, bwp, k0, n0, m0, False, block)
                assert np.array_equal(block, per_plane_neighborhood(xc, k0, n0, m0))
                truth = periodic_block(full, k0, n0, m0)
                worst = max(worst, float(np.abs(block - truth).max()) / scale)
            record_property(f"per_plane_odd_bw{bandwidth}_k{k0}", f"{worst:.3e}")

    @pytest.mark.parametrize("bandwidth", [16, 17, 60, 63])
    def test_faithful_glide_reproduces_the_cpp_and_differs_from_the_truth(
        self, bandwidth, record_property
    ):
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        full = full_cube(correlator.fxc)
        scale = float(np.abs(full).max())
        rng = np.random.default_rng(5)
        block = np.empty((3, 3, 3))
        worst = 0.0
        for k0 in (0, bwp - 1):
            for _ in range(20):
                n0 = int(rng.integers(slp))
                m0 = int(rng.integers(slp))
                _xcorr._extract_neighborhood(flat, slp, bwp, k0, n0, m0, True, block)
                assert np.array_equal(block, literal_neighborhood(xc, k0, n0, m0))
                truth = periodic_block(full, k0, n0, m0)
                worst = max(worst, float(np.abs(block - truth).max()) / scale)
        record_property(f"faithful_vs_full_bw{bandwidth}", f"{worst:.3e}")
        # measured 0.382 / 0.345 / 0.091 / 0.084 at bw 16 / 17 / 60
        # / 63 with this seed; the number is a property of the
        # random data, so the bound is only there to show that the
        # faithful block is not the truth
        assert worst > 0.02

    @pytest.mark.parametrize("bandwidth", [16, 17, 60, 63])
    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_an_interior_neighborhood_has_no_glide(
        self, bandwidth, emsphinx_compatible
    ):
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        block = np.empty((3, 3, 3))
        for n0, m0 in ((3, slp - 1), (0, 0), (slp - 1, 4)):
            _xcorr._extract_neighborhood(
                flat, slp, bwp, bwp // 2, n0, m0, emsphinx_compatible, block
            )
            assert np.array_equal(block, periodic_block(xc, bwp // 2, n0, m0))

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_the_even_glide_reads_past_a_row(self, bandwidth):
        # the shift is applied per slot index, so with m0 = bwP the
        # alpha slot 0 is slP for all three k planes and reads the
        # first element of the next row
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        block = np.empty((3, 3, 3))
        for n0 in (1, 7, bwp + 3):
            inds = literal_indices(slp, bwp, 0, n0, bwp)
            assert inds[2][0] == slp
            _xcorr._extract_neighborhood(flat, slp, bwp, 0, n0, bwp, True, block)
            for k in range(3):
                for n in range(3):
                    row = inds[1][n] + 1
                    expected = xc[inds[0][k] + row // slp, row % slp, 0]
                    assert block[k, n, 0] == expected
            assert np.array_equal(block, literal_neighborhood(xc, 0, n0, bwp))

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_the_even_glide_reads_past_a_beta_slice(self, bandwidth):
        # likewise the gamma slot of line 531: with n0 = bwP the
        # slot 0 is slP and reads row 0 of the next beta slice
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        block = np.empty((3, 3, 3))
        for m0 in (2, 9, bwp + 5):
            inds = literal_indices(slp, bwp, 0, bwp, m0)
            assert inds[1][0] == slp
            _xcorr._extract_neighborhood(flat, slp, bwp, 0, bwp, m0, True, block)
            for k in range(3):
                for m in range(3):
                    expected = xc[inds[0][k] + 1, 0, inds[2][m]]
                    assert block[k, 0, m] == expected
            assert np.array_equal(block, literal_neighborhood(xc, 0, bwp, m0))

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_past_the_end_reads_are_clamped(self, bandwidth):
        # undefined behaviour in the C++, so the clamped values are
        # never compared with it, only their positions are pinned
        correlator = correlator_with_rotated_pair(bandwidth)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        size = flat.size
        block = np.empty((3, 3, 3))

        offsets = literal_offsets(slp, bwp, bwp - 1, bwp - 2, 0)
        assert sum(offset >= size for offset in offsets) == 3
        _xcorr._extract_neighborhood(flat, slp, bwp, bwp - 1, bwp - 2, 0, True, block)
        for m in range(3):
            assert block[1, 2, m] == flat[-1]

        offsets = literal_offsets(slp, bwp, bwp - 1, 0, bwp - 2)
        assert sum(offset >= size for offset in offsets) == 1
        _xcorr._extract_neighborhood(flat, slp, bwp, bwp - 1, 0, bwp - 2, True, block)
        assert block[1, 0, 2] == flat[-1]

    @pytest.mark.parametrize("bandwidth", [16, 60])
    def test_the_past_the_end_set_is_exactly_the_documented_one(self, bandwidth):
        slp = _fft.fast_size(2 * bandwidth - 1)
        bwp = slp // 2 + 1
        size = bwp * slp * slp
        found = set()
        for n0 in range(slp):
            for m0 in range(slp):
                offsets = literal_offsets(slp, bwp, bwp - 1, n0, m0)
                if any(offset >= size for offset in offsets):
                    found.add((n0, m0))
        expected = {(bwp - 2, m0) for m0 in range(slp)}
        expected |= {(n0, bwp - 2) for n0 in (0, bwp - 3, slp - 1)}
        assert found == expected
        assert len(found) == slp + 3

    @pytest.mark.parametrize("bandwidth", [17, 63])
    def test_odd_side_lengths_never_read_past_the_end(self, bandwidth):
        slp = _fft.fast_size(2 * bandwidth - 1)
        bwp = slp // 2 + 1
        size = bwp * slp * slp
        for k0 in (0, bwp - 1):
            for n0 in range(slp):
                for m0 in range(slp):
                    offsets = literal_offsets(slp, bwp, k0, n0, m0)
                    assert max(offsets) < size


class TestInterpolation:
    @pytest.mark.parametrize(
        "center",
        [(0.3, -0.2, 0.45), (0.0, 0.0, 0.0), (0.49, 0.49, -0.49), (-0.7, 0.6, 0.1)],
    )
    def test_synthetic_quadratics_are_recovered(self, center):
        block = quadratic_block(center)
        x = np.zeros(3)
        value = _xcorr._interpolate_maxima(block, x)
        assert np.abs(x - np.asarray(center)).max() <= 1e-10
        assert abs(value - 5.0) <= 1e-10

    @pytest.mark.parametrize("seed", [15, 0, 1, 2])
    def test_a_random_tri_quadratic_is_recovered(self, seed, record_property):
        # x is the stationary point of the full 27 coefficient fit
        # (the Hessian and the gradient of the C++ use the correct
        # monomials), but the returned value is the C++ vPeak
        # expression, which is a third defect of sht_xcorr.hpp: it
        # deviates from f(x) by 1.2e-7 to 6.1e-7 here, so the value
        # is pinned against a literal transcription instead and the
        # deviation only recorded
        rng = np.random.default_rng(seed)
        a = random_tri_quadratic(rng)
        block = sample_tri_quadratic(a)
        x = np.zeros(3)
        value = _xcorr._interpolate_maxima(block, x)
        gradient = tri_quadratic_gradient(a, x)
        expected = cpp_vpeak(a, x)
        deviation = abs(value - evaluate_tri_quadratic(a, x))
        record_property(
            f"tri_quadratic_gradient_{seed}", f"{np.abs(gradient).max():.3e}"
        )
        record_property(f"tri_quadratic_vpeak_deviation_{seed}", f"{deviation:.3e}")
        assert np.abs(gradient).max() <= 1e-8
        assert abs(value - expected) <= 1e-12 * abs(expected)

    def test_a_saddle_leaves_the_step_finite(self):
        block = quadratic_block((0.1, -0.1, 0.2), curvature=(-1.0, 1.0, -1.0))
        x = np.zeros(3)
        value = _xcorr._interpolate_maxima(block, x)
        assert np.all(np.isfinite(x))
        assert math.isfinite(value)

    def test_a_flat_block_returns_the_centre_without_raising(self):
        # det == 0, so the C++ divides by zero, never converges and
        # resets at line 1350; this needs error_model="numpy", the
        # default model raises ZeroDivisionError instead
        block = np.full((3, 3, 3), 2.5)
        x = np.zeros(3)
        value = _xcorr._interpolate_maxima(block, x)
        assert value == 2.5
        assert np.array_equal(x, np.zeros(3))

    def test_a_flat_block_in_the_py_func_returns_the_centre(self):
        kernel = _xcorr._interpolate_maxima
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        block = np.full((3, 3, 3), 2.5)
        x = np.zeros(3)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = _py_func(kernel)(block, x)
        assert value == 2.5
        assert np.array_equal(x, np.zeros(3))

    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_correlate_with_an_all_zero_pattern(self, emsphinx_compatible):
        rng = np.random.default_rng(16)
        flm = random_alm(8, rng)
        gln = np.zeros((8, 8), dtype=np.complex128)
        correlator = _xcorr.SphericalCrossCorrelator(8)
        zyz, score = correlator.correlate(
            flm, gln, 1, False, emsphinx_compatible=emsphinx_compatible
        )
        assert np.all(np.isfinite(zyz))
        assert score == 0.0

    @staticmethod
    def _interp_peak_of_block(block, emsphinx_compatible):
        """Return ``interp_peak`` of a block embedded at an interior
        beta slice, where no glide runs.
        """
        correlator = _xcorr.SphericalCrossCorrelator(8)
        slp, bwp = correlator.side_length, correlator.half_side_length
        cube = np.zeros((bwp, slp, slp))
        cube[2:5, 4:7, 6:9] = block
        correlator.xc = cube
        return correlator.interp_peak(
            flat_index(3, 5, 7, slp), emsphinx_compatible=emsphinx_compatible
        )

    def test_the_x2_bounds_bug_lets_an_alpha_over_step_through(self):
        block = quadratic_block((0.2, 0.1, 1.5))
        _, peak, x = self._interp_peak_of_block(block, True)
        assert np.abs(x - np.array([0.2, 0.1, 1.5])).max() <= 1e-10
        assert abs(peak - 5.0) <= 1e-10

    def test_the_exact_bound_rejects_an_alpha_over_step(self):
        block = quadratic_block((0.2, 0.1, 1.5))
        _, peak, x = self._interp_peak_of_block(block, False)
        assert np.array_equal(x, np.zeros(3))
        assert peak == block[1, 1, 1]

    @pytest.mark.parametrize("center", [(1.5, 0.1, 0.2), (0.1, 1.5, 0.2)])
    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_beta_and_gamma_over_steps_are_rejected_in_both_settings(
        self, center, emsphinx_compatible
    ):
        # the C++ expression checks |x[0]| twice and |x[1]| once, so
        # a mutation of either surviving term is killed here
        block = quadratic_block(center)
        _, peak, x = self._interp_peak_of_block(block, emsphinx_compatible)
        assert np.array_equal(x, np.zeros(3))
        assert peak == block[1, 1, 1]

    def test_emsphinx_compatible_is_the_only_knob(self):
        for function in (
            _xcorr.SphericalCrossCorrelator.interp_peak,
            _xcorr.SphericalCrossCorrelator.correlate,
            _xcorr.NormalizedSphericalCrossCorrelator.correlate,
        ):
            parameter = inspect.signature(function).parameters["emsphinx_compatible"]
            assert parameter.default is True
        assert not [
            name for name in vars(_xcorr) if "compat" in name.lower() and name.isupper()
        ]


class TestIndexEuler:
    @pytest.mark.parametrize("side_length", [32, 33, 105, 135])
    def test_index_to_euler_matches_the_constitution_formulas(self, side_length):
        bwp = side_length // 2 + 1
        for k in range(bwp):
            for n in (0, 1, side_length // 3, side_length - 1):
                for m in (0, 1, side_length // 3, side_length - 1):
                    zyz = _xcorr.index_to_euler((k, n, m), side_length)
                    expected = np.array(
                        [
                            2 * math.pi * m / side_length - math.pi / 2,
                            2 * math.pi * k / side_length - math.pi,
                            2 * math.pi * n / side_length - math.pi / 2,
                        ]
                    )
                    assert np.abs(zyz - expected).max() <= 1e-15

    @pytest.mark.parametrize("side_length", [32, 33, 105, 120, 135])
    def test_index_round_trip_on_the_grid(self, side_length):
        bwp = side_length // 2 + 1
        for k in range(bwp):
            for n in (0, 1, side_length // 3, side_length - 1):
                for m in (0, 1, side_length // 3, side_length - 1):
                    zyz = _xcorr.index_to_euler((k, n, m), side_length)
                    assert _xcorr.euler_to_index(zyz, side_length) == (k, n, m)

    @pytest.mark.parametrize("side_length", [32, 33, 105, 120, 135])
    def test_random_rotations_round_trip_within_a_cell(
        self, side_length, record_property
    ):
        rng = np.random.default_rng(17)
        bwp = side_length // 2 + 1
        applied = []
        recovered = []
        for _ in range(200):
            zyz = random_zyz(rng)
            k, n, m = _xcorr.euler_to_index(zyz, side_length)
            assert 0 <= k < bwp
            assert 0 <= n < side_length
            assert 0 <= m < side_length
            applied.append(zyz)
            recovered.append(_xcorr.index_to_euler((k, n, m), side_length))
        deltas = misorientation_deg_many(np.array(applied), np.array(recovered))
        bound = 1.5 * math.sqrt(3) * 180.0 / side_length
        record_property(f"index_round_trip_slp{side_length}", f"{deltas.max():.3f}")
        assert deltas.max() <= bound

    @pytest.mark.parametrize("alpha, gamma", [(0.3, -1.2), (-1.5, 2.0), (2.9, 0.05)])
    def test_an_unwrapped_beta_gives_the_wrapped_index(self, alpha, gamma):
        # the C++ formula on beta = pi + 0.1 at slP 135 gives
        # kR = 137.15, slP - kR = -2.15 and a size_t wrap
        side_length = 135
        unwrapped = (alpha, math.pi + 0.1, gamma)
        wrapped = (alpha, -math.pi + 0.1, gamma)
        assert _xcorr.euler_to_index(unwrapped, side_length) == _xcorr.euler_to_index(
            wrapped, side_length
        )

    @pytest.mark.parametrize("side_length", [33, 135])
    def test_unwrapped_triples_stay_inside_the_grid(self, side_length):
        rng = np.random.default_rng(18)
        bwp = side_length // 2 + 1
        for _ in range(200):
            alpha = float(rng.uniform(-3 * math.pi, 3 * math.pi))
            beta = float(rng.uniform(-4 * math.pi, 4 * math.pi))
            gamma = float(rng.uniform(-3 * math.pi, 3 * math.pi))
            knm = _xcorr.euler_to_index((alpha, beta, gamma), side_length)
            assert 0 <= knm[0] < bwp
            assert 0 <= knm[1] < side_length
            assert 0 <= knm[2] < side_length
            reduced = (
                (alpha + math.pi / 2) % (2 * math.pi) - math.pi / 2,
                _euler.wrap_beta(beta),
                (gamma + math.pi / 2) % (2 * math.pi) - math.pi / 2,
            )
            assert knm == _xcorr.euler_to_index(reduced, side_length)

    def test_a_positive_beta_glides_into_the_stored_half(self):
        side_length = 135
        rng = np.random.default_rng(19)
        for _ in range(20):
            zyz = random_zyz(rng)
            assert zyz[1] >= 0  # quaternion_to_zyz returns beta in [0, pi]
            knm = _xcorr.euler_to_index(zyz, side_length)
            assert 0 <= knm[0] < side_length // 2 + 1
            recovered = _xcorr.index_to_euler(knm, side_length)
            assert recovered[1] <= 0
            bound = 1.5 * math.sqrt(3) * 180.0 / side_length
            assert misorientation_deg(zyz, recovered) <= bound

    def test_rounding_is_floor_of_x_plus_a_half(self):
        # C++ std::round on a non-negative argument, not Python's
        # banker's rounding, which gives 66 for 66.5
        side_length = 135
        gamma = (66.5 * 4 - side_length) * math.pi / (2 * side_length)
        assert ((gamma * 2 * side_length) / math.pi + side_length) / 4 == 66.5
        assert round(66.5) == 66
        knm = _xcorr.euler_to_index((0.0, -math.pi / 2, gamma), side_length)
        assert knm[1] == 67

    @pytest.mark.parametrize(
        "side_length, expected", [(135, (67, 34, 34)), (128, (64, 32, 32))]
    )
    def test_the_beta_zero_corner(self, side_length, expected):
        # for odd slP the fractional k is exactly slP / 2, which the
        # strict > of the glide leaves alone and which rounds to
        # bwP, one slice outside the stored half: the port clamps
        assert _xcorr.euler_to_index((0.0, 0.0, 0.0), side_length) == expected


class TestCorrelate:
    @pytest.mark.parametrize("bandwidth", SIZES)
    def test_symmetry_free_pairs(self, bandwidth, record_property):
        rng = np.random.default_rng(bandwidth)
        transform, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        for i in range(3):
            zyz = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz)
            found, score = correlator.correlate(flm, gln, 1, False)
            delta = misorientation_deg(zyz, found)
            record_property(f"coarse_bw{bandwidth}_rot{i}", f"{delta:.4f}")
            assert math.isfinite(score)
            assert_zyz_in_range(found, side_length, True)
            assert delta < tier_tolerance_deg(zyz[1], side_length)

            raw = _xcorr.index_to_euler(
                unflatten(_xcorr._find_peak(correlator.xc), side_length), side_length
            )
            raw_delta = misorientation_deg(zyz, raw)
            record_property(f"coarse_bw{bandwidth}_rot{i}_argmax", f"{raw_delta:.4f}")
            assert raw_delta < 2.5 * cell_deg(side_length) / 2

    def test_the_point_group_flags_are_the_cpp_ones(self):
        for name in POINT_GROUPS:
            assert _symmetry.point_group_flags(name) == POINT_GROUP_FLAGS[name]

    @pytest.mark.parametrize("name", POINT_GROUPS)
    @pytest.mark.parametrize("bandwidth", GROUP_BANDWIDTHS)
    def test_point_groups(self, name, bandwidth, record_property):
        n_fold, mirror = _symmetry.point_group_flags(name)
        rng = np.random.default_rng(100 * bandwidth + POINT_GROUPS.index(name))
        transform, flm = random_pair_on_grid(bandwidth, n_fold, mirror, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        worst_unreduced = 0.0
        for i in range(3):
            zyz = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz)
            found, _ = correlator.correlate(flm, gln, n_fold, mirror)
            assert_zyz_in_range(found, side_length, True)
            delta = misorientation_deg(zyz, found, name)
            unreduced = misorientation_deg(zyz, found)
            worst_unreduced = max(worst_unreduced, unreduced)
            record_property(f"group_{name}_bw{bandwidth}_rot{i}", f"{delta:.4f}")
            assert delta < tier_tolerance_deg(zyz[1], side_length)
        record_property(
            f"group_{name}_bw{bandwidth}_unreduced", f"{worst_unreduced:.3f}"
        )
        if n_fold > 1:
            # the reduction must be doing work, and on the correct
            # side: without it these are 60-180 degrees apart
            assert worst_unreduced > 30.0

    @pytest.mark.parametrize("name", POINT_GROUPS)
    def test_mirror_symmetrization_and_zeroing_are_exact(self, name):
        bandwidth = 24
        n_fold, mirror = _symmetry.point_group_flags(name)
        rng = np.random.default_rng(20)
        transform, flm = random_pair_on_grid(bandwidth, n_fold, mirror, rng)
        degree = np.arange(bandwidth)[np.newaxis, :]
        order = np.arange(bandwidth)[:, np.newaxis]
        if mirror:
            assert np.all(flm[(degree + order) % 2 == 1] == 0)
        for m in range(bandwidth):
            if m % n_fold:
                assert np.all(flm[m] == 0)

    @pytest.mark.parametrize("bandwidth", [53, 60, 63])
    @pytest.mark.parametrize(
        "label", ["0.15_cell", "pi_minus_0.15_cell", "1e-9", "0.45_cell"]
    )
    def test_near_degenerate_beta(self, bandwidth, label, record_property):
        rng = np.random.default_rng(bandwidth)
        transform, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        cell = 2 * math.pi / side_length
        beta = {
            "0.15_cell": 0.15 * cell,
            "pi_minus_0.15_cell": math.pi - 0.15 * cell,
            "1e-9": 1e-9,
            "0.45_cell": 0.45 * cell,
        }[label]
        zyz = np.array([0.7, beta, -1.9])
        gln = _wigner.rotate_harmonics(flm, zyz)
        deltas = {}
        for emsphinx_compatible in (True, False):
            found, _ = correlator.correlate(
                flm, gln, 1, False, emsphinx_compatible=emsphinx_compatible
            )
            assert_zyz_in_range(found, side_length, emsphinx_compatible)
            deltas[emsphinx_compatible] = misorientation_deg(zyz, found)
            record_property(
                f"degenerate_bw{bandwidth}_{label}_{emsphinx_compatible}",
                f"{deltas[emsphinx_compatible]:.4f}",
            )
            assert deltas[emsphinx_compatible] < cell_deg(side_length)
        if side_length % 2 == 0:
            # the per-plane glide is exact on the grid for even slP
            assert deltas[False] < cell_deg(side_length) / 2

    @pytest.mark.parametrize("bandwidth", [53, 63])
    def test_the_alpha_step_is_recorded(self, bandwidth, record_property):
        # with emsphinx_compatible=True alpha is not bounds checked,
        # so |x[2]| is recorded rather than asserted
        rng = np.random.default_rng(21)
        transform, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        worst = 0.0
        for _ in range(6):
            zyz = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz)
            correlator.compute(flm, gln, 1, False)
            index = _xcorr._find_peak(correlator.xc)
            found, score, x = correlator.interp_peak(index)
            worst = max(worst, abs(float(x[2])))
            assert math.isfinite(score)
            assert_zyz_in_range(found, side_length, True)
        record_property(f"max_abs_x2_bw{bandwidth}", f"{worst:.3f}")

    def test_clone_shares_the_table_and_reproduces_the_result(self):
        bandwidth = 24
        rng = np.random.default_rng(22)
        flm = random_alm(bandwidth, rng)
        gln = _wigner.rotate_harmonics(flm, random_zyz(rng))
        other = random_alm(bandwidth, rng)
        original = _xcorr.SphericalCrossCorrelator(bandwidth)
        clone = original.clone()
        assert clone is not original
        assert clone.wigner_d_half_pi is original.wigner_d_half_pi
        assert clone.fxc is not original.fxc
        # the two scratch buffers of the spectrum kernel too, or a
        # clone would not be usable in another thread
        assert clone._fm is not original._fm
        assert clone._gn is not original._gn
        assert clone.bandwidth == original.bandwidth

        zyz_a, score_a = original.correlate(flm, gln, 1, False)
        zyz_b, score_b = clone.correlate(flm, gln, 1, False)
        assert np.array_equal(zyz_a, zyz_b)
        assert score_a == score_b

        # alternating calls with different inputs must not
        # cross-contaminate
        original.correlate(flm, other, 1, False)
        zyz_c, score_c = clone.correlate(flm, gln, 1, False)
        assert np.array_equal(zyz_a, zyz_c)
        assert score_a == score_c


class TestNormalized:
    @pytest.mark.parametrize(
        "bandwidth",
        DEFAULT_SIZES + [pytest.param(b, marks=pytest.mark.weekly) for b in (88, 113)],
    )
    def test_wedge_masked_patterns(self, bandwidth, record_property):
        rng = np.random.default_rng(bandwidth)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        normalized = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        plain = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = plain.side_length
        assert np.isfinite(normalized.r_den).all()
        assert (normalized.r_den > 0).all()
        for i in range(3):
            zyz = random_zyz(rng)
            gln = masked_pattern(transform, flm, mask, zyz)
            found, score = normalized.correlate(gln)
            assert_zyz_in_range(found, side_length, True)
            delta = misorientation_deg(zyz, found)
            found_plain, score_plain = plain.correlate(flm, gln, 1, False)
            assert_zyz_in_range(found_plain, side_length, True)
            delta_plain = misorientation_deg(zyz, found_plain)
            record_property(f"wedge_bw{bandwidth}_rot{i}", f"{delta:.4f}")
            record_property(f"wedge_bw{bandwidth}_rot{i}_score", f"{score:.4f}")
            record_property(f"wedge_bw{bandwidth}_rot{i}_plain", f"{delta_plain:.4f}")
            assert delta < cell_deg(side_length)
            assert delta_plain < cell_deg(side_length)
            assert math.isfinite(score) and score > 0

    @pytest.mark.parametrize("name", POINT_GROUPS)
    @pytest.mark.parametrize("bandwidth", GROUP_BANDWIDTHS)
    def test_wedge_masked_point_groups(self, name, bandwidth, record_property):
        n_fold, mirror = _symmetry.point_group_flags(name)
        rng = np.random.default_rng(200 * bandwidth + POINT_GROUPS.index(name))
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, n_fold, mirror, rng)
        normalized = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, n_fold, mirror, mlm
        )
        side_length = normalized.side_length
        zyz = random_zyz(rng)
        gln = masked_pattern(transform, flm, mask, zyz)
        found, score = normalized.correlate(gln)
        assert_zyz_in_range(found, side_length, True)
        delta = misorientation_deg(zyz, found, name)
        record_property(f"wedge_group_{name}_bw{bandwidth}", f"{delta:.4f}")
        record_property(f"wedge_group_{name}_bw{bandwidth}_score", f"{score:.4f}")
        assert delta < cell_deg(side_length)
        assert math.isfinite(score) and score > 0

    @pytest.mark.parametrize("mask", ["wedge", "cap"])
    def test_r_den_is_finite_and_positive(self, mask, record_property):
        # the wedge is covered at every size of the sweep above too
        for bandwidth in (17, 24, 53):
            rng = np.random.default_rng(23)
            transform, flm, flm2, mlm, masks = masked_case(
                bandwidth, 1, False, rng, mask=mask
            )
            fraction = float(np.mean(np.concatenate([m.ravel() for m in masks])))
            record_property(f"{mask}_fraction_bw{bandwidth}", f"{fraction:.3f}")
            normalized = _xcorr.NormalizedSphericalCrossCorrelator(
                bandwidth, flm, flm2, 1, False, mlm
            )
            assert np.isfinite(normalized.r_den).all()
            assert (normalized.r_den > 0).all()

    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror", [(17, 1, False), (24, 3, True)]
    )
    def test_r_den_matches_the_huhle_expression(
        self, bandwidth, n_fold, mirror, record_property
    ):
        rng = np.random.default_rng(24)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, n_fold, mirror, rng)
        normalized = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, n_fold, mirror, mlm
        )
        plain = _xcorr.SphericalCrossCorrelator(bandwidth)
        mrf = plain.compute(flm, mlm, n_fold, mirror).copy()
        mrf2 = plain.compute(flm2, mlm, n_fold, mirror)
        s2m = float(mlm[0, 0].real) * math.sqrt(4 * math.pi)
        f_wbar = mrf / s2m
        expected = 1.0 / np.sqrt(mrf2 - 2 * f_wbar * mrf + f_wbar * f_wbar * s2m)
        assert np.isfinite(expected).all()
        assert (
            np.abs(normalized.r_den - expected).max() <= 1e-13 * np.abs(expected).max()
        )

        # a s2m without the sqrt(4 pi) is a different denominator.
        # How different depends on the window mean: measured 23.9 %
        # of the largest relative change at bw 17 but only 8.1 % at
        # bw 24 with the 3-fold mirrored reference, whose window
        # mean mrf is small against mrf2, so the guard is pinned at
        # 5 % and the value recorded
        wrong_s2m = float(mlm[0, 0].real)
        wrong_bar = mrf / wrong_s2m
        with np.errstate(invalid="ignore"):
            wrong = 1.0 / np.sqrt(
                mrf2 - 2 * wrong_bar * mrf + wrong_bar * wrong_bar * wrong_s2m
            )
            relative = np.where(
                np.isfinite(wrong),
                np.abs(wrong - expected) / np.abs(expected),
                np.inf,
            )
        worst = float(np.max(relative))
        record_property(f"wrong_s2m_relative_bw{bandwidth}", f"{worst:.4f}")
        assert worst > 0.05
        # and the guard must bite on the denominator the
        # implementation built, not only on the two test-local
        # expressions above (measured ratio 0.203 / 0.072)
        difference = float(np.nanmax(np.abs(normalized.r_den - wrong)))
        assert difference > 0.05 * np.abs(expected).max()

    def test_the_score_semantics_are_documented(self):
        doc = " ".join(_xcorr.NormalizedSphericalCrossCorrelator.__doc__.split())
        assert "not divided by the standard deviation of the pattern function" in doc

    def test_clone_shares_the_denominator_and_reproduces_the_result(self):
        bandwidth = 17
        rng = np.random.default_rng(25)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        original = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth, flm, flm2, 1, False, mlm
        )
        clone = original.clone()
        assert clone is not original
        assert clone.r_den is original.r_den
        assert clone.flm is original.flm
        assert clone.flm2 is original.flm2
        assert clone.mlm is original.mlm
        assert clone.correlator.wigner_d_half_pi is original.correlator.wigner_d_half_pi
        assert clone.correlator.fxc is not original.correlator.fxc
        assert clone.correlator._fm is not original.correlator._fm
        assert clone.correlator._gn is not original.correlator._gn

        gln = masked_pattern(transform, flm, mask, random_zyz(rng))
        zyz_a, score_a = original.correlate(gln)
        zyz_b, score_b = clone.correlate(gln)
        assert np.array_equal(zyz_a, zyz_b)
        assert score_a == score_b


class TestNickel:
    def test_autocorrelation_at_bandwidth_68(self, nickel_harmonics, record_property):
        flm = nickel_harmonics.alm
        n_fold = nickel_harmonics.n_fold
        mirror = nickel_harmonics.has_equatorial_mirror
        assert (n_fold, mirror) == (4, True)
        correlator = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH)
        side_length = correlator.side_length
        xc = correlator.compute(flm, flm, n_fold, mirror)
        power = total_power(flm)
        record_property("ni_power_bw68", f"{power:.4f}")
        record_property("ni_max_over_power", f"{float(xc.max()) / power:.6f}")
        record_property("ni_min_over_power", f"{float(xc.min()) / power:.4f}")
        assert abs(float(xc.max()) / power - 1) <= 1e-6

        index = _xcorr._find_peak(xc)
        zyz = _xcorr.index_to_euler(unflatten(index, side_length), side_length)
        delta = misorientation_deg(zyz, (0.0, 0.0, 0.0), "m-3m")
        record_property("ni_argmax_to_identity", f"{delta:.4f}")
        assert delta < cell_deg(side_length) / 2

        for threshold in (0.999, 0.99, 0.95, 0.9):
            count = int(np.count_nonzero(xc >= threshold * power))
            record_property(f"ni_count_above_{threshold}", str(count))

    def test_the_24_operators_at_bandwidth_68(self, nickel_harmonics, record_property):
        flm = nickel_harmonics.alm
        n_fold = nickel_harmonics.n_fold
        mirror = nickel_harmonics.has_equatorial_mirror
        correlator = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH)
        side_length = correlator.side_length
        xc = correlator.compute(flm, flm, n_fold, mirror)
        power = total_power(flm)
        inverse = (~O).data
        for i in range(24):
            zyz_operator = _euler.quaternion_to_zyz(inverse[i])
            k, n, m = _xcorr.euler_to_index(zyz_operator, side_length)
            value = float(xc[k, n, m])
            found, peak, _ = correlator.interp_peak(flat_index(k, n, m, side_length))
            delta = misorientation_deg(found, (0.0, 0.0, 0.0), "m-3m")
            record_property(f"ni_operator{i}_cell", f"{value / power:.4f}")
            record_property(f"ni_operator{i}_peak", f"{peak / power:.4f}")
            record_property(f"ni_operator{i}_angle", f"{delta:.4f}")
            assert value >= 0.9 * power
            assert peak >= 0.9 * power
            assert delta < cell_deg(side_length)

    def test_high_correlation_grid_points_are_near_a_symmetry(
        self, nickel_harmonics, record_property
    ):
        flm = nickel_harmonics.alm
        n_fold = nickel_harmonics.n_fold
        mirror = nickel_harmonics.has_equatorial_mirror
        correlator = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH)
        side_length = correlator.side_length
        xc = correlator.compute(flm, flm, n_fold, mirror)
        power = total_power(flm)
        cell = cell_deg(side_length)

        indices = np.argwhere(xc >= 0.95 * power)
        angles = np.array(
            [
                _xcorr.index_to_euler(tuple(int(v) for v in knm), side_length)
                for knm in indices
            ]
        )
        deltas = misorientation_deg_many(angles, (0.0, 0.0, 0.0), "m-3m")
        record_property("ni_worst_angle_above_0.95", f"{deltas.max():.4f}")
        assert deltas.max() < 2 * cell

        indices = np.argwhere(xc >= 0.9 * power)
        angles = np.array(
            [
                _xcorr.index_to_euler(tuple(int(v) for v in knm), side_length)
                for knm in indices
            ]
        )
        inverse = (~O).data
        worst = 0.0
        for i in range(24):
            zyz_operator = _euler.quaternion_to_zyz(inverse[i])
            nearest = misorientation_deg_many(angles, zyz_operator).min()
            worst = max(worst, float(nearest))
        record_property("ni_worst_coverage_above_0.9", f"{worst:.4f}")
        assert worst < 1.5 * cell

    def test_autocorrelation_at_bandwidth_64(
        self, nickel_harmonics_even, record_property
    ):
        flm = nickel_harmonics_even.alm
        n_fold = nickel_harmonics_even.n_fold
        mirror = nickel_harmonics_even.has_equatorial_mirror
        correlator = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH_EVEN)
        side_length = correlator.side_length
        assert side_length == 128
        xc = correlator.compute(flm, flm, n_fold, mirror)
        power = total_power(flm)
        inverse = (~O).data
        exact = 0
        close = 0
        half_cell = 0
        for i in range(24):
            zyz_operator = _euler.quaternion_to_zyz(inverse[i])
            k, n, m = _xcorr.euler_to_index(zyz_operator, side_length)
            ratio = float(xc[k, n, m]) / power
            record_property(f"ni64_operator{i}", f"{ratio:.5f}")
            if abs(math.sin(zyz_operator[1])) < 1e-9:
                assert abs(ratio - 1) <= 1e-6
                exact += 1
            else:
                assert ratio >= 0.97
                close += 1
            found, _, _ = correlator.interp_peak(flat_index(k, n, m, side_length))
            delta = misorientation_deg(found, (0.0, 0.0, 0.0), "m-3m")
            record_property(f"ni64_operator{i}_angle", f"{delta:.4f}")
            assert delta < cell_deg(side_length)
            if delta < cell_deg(side_length) / 2:
                half_cell += 1
        assert exact == 8
        assert close == 16
        record_property("ni64_within_half_a_cell", str(half_cell))
        # all 24, not 22: the worst is 1.2682 deg at Rz(90)/Rz(180),
        # the even-slP glide defect, against a half cell of 1.40625
        assert half_cell == 24

    def test_a_rotated_master_is_recovered(self, nickel_harmonics, record_property):
        flm = nickel_harmonics.alm
        n_fold = nickel_harmonics.n_fold
        mirror = nickel_harmonics.has_equatorial_mirror
        correlator = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH)
        side_length = correlator.side_length
        power = total_power(flm)
        rng = np.random.default_rng(26)
        for i in range(5):
            zyz = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz)
            found, score = correlator.correlate(flm, gln, n_fold, mirror)
            delta = misorientation_deg(zyz, found, "m-3m")
            record_property(f"ni_rotated_{i}", f"{delta:.4f}")
            record_property(f"ni_rotated_{i}_ratio", f"{score / power:.5f}")
            assert delta < cell_deg(side_length) / 2
            assert score / power >= 0.98

    @pytest.mark.parametrize("mask", ["wedge", "cap"])
    def test_the_d7_gate(
        self,
        mask,
        nickel_harmonics,
        nickel_harmonics_plain,
        record_property,
    ):
        dim = NI_BANDWIDTH + 3
        transform = _sht.SphericalHarmonicTransform(NI_BANDWIDTH, "legendre", dim)
        mask_north, mask_south = wedge_mask(dim) if mask == "wedge" else cap_mask(dim)
        record_property(
            f"{mask}_fraction",
            f"{float(np.mean([mask_north.mean(), mask_south.mean()])):.3f}",
        )
        mlm = transform.analyze(mask_north, mask_south)
        mask_north, mask_south = transform.synthesize(mlm)
        masks = (mask_north, mask_south)

        # the two settings are the *master* normalization quirk of
        # Phase 2, not the correlator's own keyword, which keeps its
        # default here: the gate asks whether the master's DC term
        # moves the argmax, not only the score
        harmonics = {True: nickel_harmonics, False: nickel_harmonics_plain}
        side_length = _fft.fast_size(2 * NI_BANDWIDTH - 1)
        deltas = {}
        for compatible, harmonic in harmonics.items():
            flm = harmonic.alm
            n_fold = harmonic.n_fold
            mirror = harmonic.has_equatorial_mirror
            record_property(f"ni_a00_{compatible}", f"{float(flm[0, 0].real):.4f}")
            north, south = transform.synthesize(flm)
            flm2 = transform.analyze(north**2, south**2)
            normalized = _xcorr.NormalizedSphericalCrossCorrelator(
                NI_BANDWIDTH, flm, flm2, n_fold, mirror, mlm
            )
            plain = _xcorr.SphericalCrossCorrelator(NI_BANDWIDTH)
            rng = np.random.default_rng(27)
            values = []
            for i in range(4):
                zyz = random_zyz(rng)
                gln = masked_pattern(transform, flm, masks, zyz)
                found, score = normalized.correlate(gln)
                assert_zyz_in_range(found, side_length, True)
                delta = misorientation_deg(zyz, found, "m-3m")
                found_plain, score_plain = plain.correlate(flm, gln, n_fold, mirror)
                assert_zyz_in_range(found_plain, side_length, True)
                record_property(f"d7_{mask}_{compatible}_{i}", f"{delta:.4f}")
                record_property(f"d7_{mask}_{compatible}_{i}_score", f"{score:.4f}")
                record_property(
                    f"d7_{mask}_{compatible}_{i}_plain_score", f"{score_plain:.4f}"
                )
                record_property(
                    f"d7_{mask}_{compatible}_{i}_plain",
                    f"{misorientation_deg(zyz, found_plain, 'm-3m'):.4f}",
                )
                assert delta < cell_deg(side_length)
                values.append((delta, score))
            deltas[compatible] = values

        for i in range(4):
            difference = abs(deltas[True][i][0] - deltas[False][i][0])
            ratio = deltas[True][i][1] / deltas[False][i][1]
            record_property(f"d7_{mask}_difference_{i}", f"{difference:.4f}")
            record_property(f"d7_{mask}_score_ratio_{i}", f"{ratio:.3f}")
            assert difference < cell_deg(side_length) / 2
            assert 1.5 < ratio < 4.0


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # the flag and py_func tests are parametrised over the
        # literal list above, so a kernel added during the
        # implementation would silently escape both of them
        assert _njit_kernel_names(_xcorr) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _xcorr"
        )

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        # dropping either option leaves every other test passing, so
        # the private Numba attributes are read directly
        kernel = getattr(_xcorr, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_only_the_interpolation_uses_the_numpy_error_model(self, name):
        # the C++ divides by an unguarded Hessian determinant and
        # relies on IEEE semantics, which is a correctness fix
        # there and nowhere else
        kernel = getattr(_xcorr, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        expected = "numpy" if name == "_interpolate_maxima" else None
        assert kernel.targetoptions.get("error_model") == expected

    @pytest.mark.parametrize(
        "bandwidth, n_fold, mirror",
        [(8, 1, False), (9, 3, True), (12, 2, False)],
    )
    def test_spectrum_kernel_py_func(self, bandwidth, n_fold, mirror):
        kernel = _xcorr._xcorr_spectrum
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(28)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        table = _wigner.wigner_d_half_pi_table(bandwidth, True)
        slp = _fft.fast_size(2 * bandwidth - 1)
        bwp = slp // 2 + 1
        results = []
        for function in (kernel, _py_func(kernel)):
            fxc = np.zeros((slp, slp, bwp), dtype=np.complex128)
            fm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
            gn = np.zeros(bandwidth, dtype=np.complex128)
            function(flm, gln, table, n_fold, mirror, fxc, fm, gn)
            results.append(fxc)
        scale = float(np.abs(results[0]).max())
        assert np.abs(results[0] - results[1]).max() <= 4 * EPS * scale

    def test_interpolate_maxima_py_func(self, record_property):
        kernel = _xcorr._interpolate_maxima
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(29)
        blocks = [rng.normal(size=(3, 3, 3)) for _ in range(50)]
        blocks += [
            quadratic_block(center)
            for center in ((0.3, -0.2, 0.45), (0.0, 0.0, 0.0), (0.49, 0.49, -0.49))
        ]
        worst_x = 0.0
        worst_value = 0.0
        bitwise = True
        for block in blocks:
            x_compiled = np.zeros(3)
            x_interpreted = np.zeros(3)
            with np.errstate(divide="ignore", invalid="ignore"):
                compiled = kernel(block, x_compiled)
                interpreted = _py_func(kernel)(block, x_interpreted)
            if not np.isfinite(compiled) or not np.isfinite(interpreted):
                continue
            worst_x = max(worst_x, float(np.abs(x_compiled - x_interpreted).max()))
            worst_value = max(
                worst_value,
                abs(compiled - interpreted) / max(abs(interpreted), 1.0),
            )
            bitwise = bitwise and compiled == interpreted
        record_property("interpolate_py_func_worst_x", f"{worst_x:.3e}")
        record_property("interpolate_py_func_bitwise", str(bitwise))
        assert worst_x <= 1e-9
        assert worst_value <= 1e-12

    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_extract_neighborhood_py_func(self, emsphinx_compatible):
        kernel = _xcorr._extract_neighborhood
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        correlator = correlator_with_rotated_pair(16)
        xc = correlator.xc
        slp, bwp = correlator.side_length, correlator.half_side_length
        flat = xc.reshape(-1)
        rng = np.random.default_rng(30)
        for k0 in (0, bwp // 2, bwp - 1):
            for _ in range(5):
                n0 = int(rng.integers(slp))
                m0 = int(rng.integers(slp))
                compiled = np.empty((3, 3, 3))
                interpreted = np.empty((3, 3, 3))
                kernel(flat, slp, bwp, k0, n0, m0, emsphinx_compatible, compiled)
                _py_func(kernel)(
                    flat, slp, bwp, k0, n0, m0, emsphinx_compatible, interpreted
                )
                assert np.array_equal(compiled, interpreted)

    def test_find_peak_py_func(self):
        kernel = _xcorr._find_peak
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(31)
        for _ in range(10):
            cube = rng.normal(size=(4, 5, 5))
            assert kernel(cube) == _py_func(kernel)(cube)

    def test_scale_and_find_peak_py_func(self):
        kernel = _xcorr._scale_and_find_peak
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        rng = np.random.default_rng(32)
        for _ in range(10):
            cube = rng.normal(size=(4, 5, 5))
            r_den = rng.uniform(0.5, 2.0, cube.shape)
            compiled = cube.copy()
            interpreted = cube.copy()
            assert kernel(compiled, r_den) == _py_func(kernel)(interpreted, r_den)
            assert np.array_equal(compiled, interpreted)


class TestBaselines:
    @pytest.mark.parametrize("bandwidth", [53, 68, 88])
    @pytest.mark.parametrize("n_fold, mirror", [(1, False), (4, True)])
    def test_correlate_timing_is_recorded(
        self, bandwidth, n_fold, mirror, record_property
    ):
        rng = np.random.default_rng(33)
        flm = random_alm(bandwidth, rng, n_fold, mirror)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        correlator.correlate(flm, gln, n_fold, mirror)  # warm the Numba cache
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            correlator.correlate(flm, gln, n_fold, mirror)
            best = min(best, time.perf_counter() - start)
        record_property(
            f"correlate_seconds_bw{bandwidth}_nf{n_fold}_mir{mirror}", f"{best:.4f}"
        )
        assert best < 5.0

    def test_stage_timings_are_recorded(self, record_property):
        bandwidth = 68
        rng = np.random.default_rng(34)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        correlator.compute(flm, gln, 1, False)  # warm the Numba cache

        start = time.perf_counter()
        _xcorr._xcorr_spectrum(
            flm,
            gln,
            correlator.wigner_d_half_pi,
            1,
            False,
            correlator.fxc,
            np.zeros((bandwidth, bandwidth), dtype=np.complex128),
            np.zeros(bandwidth, dtype=np.complex128),
        )
        record_property("stage_kernel_seconds", f"{time.perf_counter() - start:.4f}")

        start = time.perf_counter()
        xc = _xcorr._inverse_fft(correlator.fxc, 1)
        record_property("stage_inverse_seconds", f"{time.perf_counter() - start:.4f}")

        start = time.perf_counter()
        index = _xcorr._find_peak(xc)
        correlator.xc = xc
        correlator.interp_peak(index)
        record_property("stage_peak_seconds", f"{time.perf_counter() - start:.4f}")

    @pytest.mark.parametrize(
        "bandwidth", [63, 68, 88, pytest.param(113, marks=pytest.mark.weekly)]
    )
    def test_memory_of_the_plain_correlator_is_recorded(
        self, bandwidth, record_property
    ):
        rng = np.random.default_rng(35)
        flm = random_alm(bandwidth, rng)
        gln = random_alm(bandwidth, rng)
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
            correlator.correlate(flm, gln, 1, False)
            construct_peak = tracemalloc.get_traced_memory()[1]
            resident = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            correlator.correlate(flm, gln, 1, False)
            recurring = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        analytic = (
            correlator.fxc.nbytes
            + correlator.xc.nbytes
            + correlator.wigner_d_half_pi.nbytes
        )
        for label, value in (
            ("construct", construct_peak),
            ("resident", resident),
            ("recurring", recurring),
            ("analytic", analytic),
        ):
            record_property(f"memory_bw{bandwidth}_{label}_mb", f"{value / 1e6:.1f}")
        slp, bwp = correlator.side_length, correlator.half_side_length
        assert correlator.xc.base is None
        assert correlator.xc.nbytes == 8 * bwp * slp * slp
        assert recurring < 3 * analytic
        assert resident < 1.2 * analytic

    @pytest.mark.parametrize(
        "bandwidth", [63, 68, 88, pytest.param(113, marks=pytest.mark.weekly)]
    )
    def test_memory_of_the_normalized_correlator_is_recorded(
        self, bandwidth, record_property
    ):
        rng = np.random.default_rng(36)
        transform, flm, flm2, mlm, mask = masked_case(bandwidth, 1, False, rng)
        gln = masked_pattern(transform, flm, mask, random_zyz(rng))
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            correlator = _xcorr.NormalizedSphericalCrossCorrelator(
                bandwidth, flm, flm2, 1, False, mlm
            )
            correlator.correlate(gln)
            construct_peak = tracemalloc.get_traced_memory()[1]
            resident = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            correlator.correlate(gln)
            recurring = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        plain = correlator.correlator
        analytic = (
            plain.fxc.nbytes
            + plain.xc.nbytes
            + plain.wigner_d_half_pi.nbytes
            + correlator.r_den.nbytes
        )
        for label, value in (
            ("construct", construct_peak),
            ("resident", resident),
            ("recurring", recurring),
            ("analytic", analytic),
        ):
            record_property(
                f"memory_normalized_bw{bandwidth}_{label}_mb", f"{value / 1e6:.1f}"
            )
        assert recurring < 3 * analytic
        assert resident < 1.2 * analytic


@pytest.mark.weekly
class TestWeeklyStatistics:
    def test_near_degenerate_beta_statistics_at_bandwidth_68(self, record_property):
        bandwidth = 68
        rng = np.random.default_rng(37)
        transform, flm = random_pair_on_grid(bandwidth, 1, False, rng)
        correlator = _xcorr.SphericalCrossCorrelator(bandwidth)
        side_length = correlator.side_length
        cell = 2 * math.pi / side_length
        near = 0
        worst_near = 0.0
        worst_far = 0.0
        for _ in range(30):
            zyz = random_zyz(rng)
            gln = _wigner.rotate_harmonics(flm, zyz)
            found, _ = correlator.correlate(flm, gln, 1, False)
            delta = misorientation_deg(zyz, found)
            beta = abs(_euler.wrap_beta(float(zyz[1])))
            if min(beta, abs(beta - math.pi)) < cell:
                near += 1
                worst_near = max(worst_near, delta)
            else:
                worst_far = max(worst_far, delta)
            assert delta < tier_tolerance_deg(zyz[1], side_length)
        record_property("weekly_near_fraction", f"{near / 30:.3f}")
        record_property("weekly_worst_near", f"{worst_near:.4f}")
        record_property("weekly_worst_far", f"{worst_far:.4f}")
