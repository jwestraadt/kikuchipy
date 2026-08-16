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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517), all from
# ``include/sht/square_sht.hpp``:
# - ``square::lambert::sphereToSquare()`` (lines 591-606)
# - ``square::lambert::squareToSphere()`` (lines 614-642)
# - ``square::lambert::cosLats()`` (lines 648-660)
# - ``square::lambert::normals()`` (lines 665-675)
# - ``square::lambert::solidAngles()`` (lines 681-736)
# - ``square::legendre::roots()`` (lines 746-818)
# - ``square::legendre::normals()`` (lines 823-869)
# - ``square::readRing()`` and ``square::writeRing()`` (lines 942-1014)
# - ``square::computeWeightsSkip()`` (lines 1022-1063)
# - ``square::cosLats()`` (lines 1069-1081)
# - ``square::normals()`` (lines 1087-1099)
# - ``square::solidAngles()`` (lines 1105-1138)
# - ``square::ringNum()`` (lines 1144-1160)
# - The grid and bandwidth limits of
#   ``square::DiscreteSHT::Constants::Constants()`` (lines 337-345)

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Modified by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""Square grids on the unit sphere for the discrete spherical harmonic
transform.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

A square grid has an odd side length ``dim``, and a hemisphere is a
``(dim, dim)`` row-major array with ``X = i / (dim - 1)`` along axis 1
(columns) and ``Y = j / (dim - 1)`` along axis 0 (rows). The sphere is
the northern hemisphere followed by the southern hemisphere, which
share the equator ("double cover"). Ring ``y`` is the set of pixels at
Chebyshev distance ``y`` from the centre pixel and holds
``N_phi(y) = max(1, 8 * y)`` points, equally spaced in azimuth,
starting at azimuth zero and walking counter-clockwise. There are
``n_rings(dim) = (dim + 1) // 2`` rings, from the north pole (ring 0)
to the equator (ring ``n_rings(dim) - 1``).

Two layouts place the rings at different latitudes:

``"lambert"``
    The Roşca equal-area (square Lambert) projection
    :cite:`rosca2010new`, where every pixel covers the same solid
    angle and ``cos(theta_y) = 1 - (2 * y / (dim - 1)) ** 2``.
``"legendre"``
    Iso-latitude rings at the non-negative roots of the Legendre
    polynomial ``P_(dim - 2)``, plus an explicit pole at
    ``cos(theta) = 1`` :cite:`lenthe2019spherical`. Gauss-Legendre
    quadrature is exact, so this layout supports roughly twice the
    bandwidth of ``"lambert"`` for the same ``dim``.
"""

import math

from numba import njit
import numpy as np
from numpy.polynomial.legendre import leggauss

LAYOUTS = ("lambert", "legendre")

# Machine epsilon and pi / 4, both compile time constants in the Numba
# kernels below, as EMSphInx' numeric_limits<Real>::epsilon() and kPi_4
_EPS = float(np.finfo(np.float64).eps)
_PI_4 = math.pi / 4

# Largest tolerated |sum(w_hat) - 1| in _ring_weights_skip(), as in
# EMSphInx (``square_sht.hpp``, line 1057)
_WEIGHT_SUM_TOLERANCE = float(np.cbrt(_EPS) / 64)


# ----------------------------- Helpers ------------------------------ #


def validate_dim(dim: int) -> None:
    """Raise if ``dim`` is not a valid square grid side length.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Raises
    ------
    ValueError
        If ``dim`` is smaller than three or even. EMSphInx supports
        odd side lengths only, because only then does every ring
        start at azimuth zero.
    """
    if dim < 3:
        raise ValueError(f"Square grid side length {dim} must be at least three")
    if dim % 2 == 0:
        raise ValueError(f"Square grid side length {dim} must be odd")


def _validate_layout(layout: str) -> None:
    """Raise if ``layout`` is not a known square grid layout.

    Parameters
    ----------
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Raises
    ------
    ValueError
        If ``layout`` is not in :data:`LAYOUTS`.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Square grid layout {layout!r} must be one of {LAYOUTS}")


def n_rings(dim: int) -> int:
    """Return the number of rings from the north pole to the equator.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    n
        ``(dim + 1) // 2``, i.e. ``Nt`` in EMSphInx.

    Notes
    -----
    This is unchecked arithmetic, unlike its siblings: ``dim`` is not
    validated, since every caller validates it first.
    """
    return (dim + 1) // 2


def n_grid_points(dim: int) -> int:
    """Return the number of unique points on the sphere.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    n
        ``2 * dim * dim - 4 * (dim - 1)``, i.e. the two hemispheres
        minus the equator ring which they share.

    Notes
    -----
    This is unchecked arithmetic, unlike its siblings: ``dim`` is not
    validated, since every caller validates it first.
    """
    return 2 * dim * dim - 4 * (dim - 1)


def default_dim(bandwidth: int, layout: str) -> int:
    """Return the smallest usual grid side length for a bandwidth.

    Parameters
    ----------
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    dim
        ``2 * bandwidth + 1`` for ``"lambert"`` and ``bandwidth + 2``
        (odd bandwidth) or ``bandwidth + 3`` (even bandwidth) for
        ``"legendre"``, as in EMSphInx' round trip test
        (``test/sht/square_sht.cpp``, lines 110-111).

    Raises
    ------
    ValueError
        If ``layout`` is unknown or ``bandwidth`` is smaller than one.
    """
    _validate_layout(layout)
    if bandwidth < 1:
        raise ValueError(f"Bandwidth {bandwidth} must be at least one")
    if layout == "lambert":
        return 2 * bandwidth + 1
    return bandwidth + (2 if bandwidth % 2 == 1 else 3)


def max_bandwidth(dim: int, layout: str) -> int:
    """Return the largest bandwidth a grid supports.

    Parameters
    ----------
    dim
        Side length of the square grid.
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    bandwidth
        ``dim - 2`` for ``"legendre"`` and ``(dim - 1) // 2`` for
        ``"lambert"``.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or ``layout`` is unknown.
    """
    validate_dim(dim)
    _validate_layout(layout)
    if layout == "lambert":
        return (dim - 1) // 2
    return dim - 2


# ------------------ Square Lambert (Roşca) mapping ------------------ #


@njit(cache=True, nogil=True)
def _square_to_sphere_kernel(xy: np.ndarray) -> np.ndarray:
    """Return unit vectors from square Lambert coordinates.

    Parameters
    ----------
    xy
        Square coordinates ``(X, Y)`` in an array of shape ``(n, 2)``
        and 64-bit floating point data type, both in the closed
        interval [0, 1].

    Returns
    -------
    v
        Unit vectors on the northern hemisphere in an array of shape
        ``(n, 3)`` and 64-bit floating point data type.

    Raises
    ------
    ValueError
        If any point lies outside the unit square, i.e. if
        ``max(|2 * X - 1|, |2 * Y - 1|) > 1 + eps``.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = xy.shape[0]
    v = np.empty((n, 3), dtype=np.float64)
    for k in range(n):
        s_x = 2 * xy[k, 0] - 1
        s_y = 2 * xy[k, 1] - 1
        a_x = abs(s_x)
        a_y = abs(s_y)
        v_max = max(a_x, a_y)
        if v_max <= _EPS:
            v[k, 0] = 0.0
            v[k, 1] = 0.0
            v[k, 2] = 1.0
        else:
            if v_max > 1 + _EPS:
                raise ValueError("Point does not lie in the unit square")
            if a_x <= a_y:
                q = s_y * math.sqrt(2 - s_y * s_y)
                qq = _PI_4 * s_x / s_y
                x = q * math.sin(qq)
                y = q * math.cos(qq)
            else:
                q = s_x * math.sqrt(2 - s_x * s_x)
                qq = _PI_4 * s_y / s_x
                x = q * math.cos(qq)
                y = q * math.sin(qq)
            z = 1 - v_max * v_max
            magnitude = math.sqrt(x * x + y * y + z * z)
            v[k, 0] = x / magnitude
            v[k, 1] = y / magnitude
            v[k, 2] = z / magnitude
    return v


@njit(cache=True, nogil=True)
def _sphere_to_square_kernel(v: np.ndarray) -> np.ndarray:
    """Return square Lambert coordinates from unit vectors.

    Parameters
    ----------
    v
        Unit vectors in an array of shape ``(n, 3)`` and 64-bit
        floating point data type.

    Returns
    -------
    xy
        Square coordinates ``(X, Y)`` in an array of shape ``(n, 2)``
        and 64-bit floating point data type, both in the closed
        interval [0, 1].

    Notes
    -----
    The projection uses ``|z|``, so both hemispheres map onto the same
    square, as in EMSphInx.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = v.shape[0]
    xy = np.empty((n, 2), dtype=np.float64)
    for k in range(n):
        x = v[k, 0]
        y = v[k, 1]
        f_z = abs(v[k, 2])
        if f_z == 1.0:
            xy[k, 0] = 0.5
            xy[k, 1] = 0.5
        elif abs(y) <= abs(x):
            half_x = math.copysign(math.sqrt(1 - f_z), x) * 0.5
            xy[k, 1] = half_x * math.atan(y / x) / _PI_4 + 0.5
            xy[k, 0] = half_x + 0.5
        else:
            half_y = math.copysign(math.sqrt(1 - f_z), y) * 0.5
            xy[k, 0] = half_y * math.atan(x / y) / _PI_4 + 0.5
            xy[k, 1] = half_y + 0.5
    return xy


def square_to_sphere(xy: np.ndarray) -> np.ndarray:
    """Return unit vectors from square Lambert coordinates.

    Parameters
    ----------
    xy
        Square coordinates ``(X, Y)`` in an array of shape
        ``(..., 2)``, both in the closed interval [0, 1].

    Returns
    -------
    v
        Unit vectors on the northern hemisphere in an array of shape
        ``(..., 3)`` and 64-bit floating point data type.

    Raises
    ------
    ValueError
        If any point lies outside the unit square.

    See Also
    --------
    sphere_to_square
    """
    xy = np.asarray(xy, dtype=np.float64)
    flat = np.ascontiguousarray(xy.reshape(-1, 2))
    return _square_to_sphere_kernel(flat).reshape(xy.shape[:-1] + (3,))


def sphere_to_square(v: np.ndarray) -> np.ndarray:
    """Return square Lambert coordinates from unit vectors.

    Parameters
    ----------
    v
        Vectors in an array of shape ``(..., 3)``.

    Returns
    -------
    xy
        Square coordinates ``(X, Y)`` in an array of shape
        ``(..., 2)`` and 64-bit floating point data type, both in the
        closed interval [0, 1]. The north pole maps to ``(0.5, 0.5)``.

    See Also
    --------
    square_to_sphere
    """
    v = np.asarray(v, dtype=np.float64)
    flat = np.ascontiguousarray(v.reshape(-1, 3))
    return _sphere_to_square_kernel(flat).reshape(v.shape[:-1] + (2,))


# --------------------------- Ring latitudes ------------------------- #


def lambert_cos_latitudes(dim: int) -> np.ndarray:
    """Return the cosines of the ring latitudes of a Lambert grid.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    cos_lats
        Cosines of the ring latitudes from the north pole to the
        equator in an array of shape ``(n_rings(dim),)`` and 64-bit
        floating point data type. Entry ``y`` equals
        ``1 - (2 * y / (dim - 1)) ** 2``.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.

    Notes
    -----
    The values are accumulated with the integer recursion of EMSphInx
    (``numer -= delta; delta += 8``), which rounds once, and not with
    the closed form, which rounds twice. The Chebyshev-Vandermonde
    system solved in :func:`quadrature_weights` is ill-conditioned
    enough for the difference to matter at large ``dim``.
    """
    validate_dim(dim)
    count = n_rings(dim)
    cos_lats = np.empty(count, dtype=np.float64)
    denominator = (dim - 1) * (dim - 1)
    numerator = denominator
    delta = 4
    for i in range(count):
        cos_lats[i] = numerator / denominator
        numerator -= delta
        delta += 8
    return cos_lats


def legendre_roots(n: int) -> np.ndarray:
    """Return the non-negative roots of the Legendre polynomial
    ``P_n``.

    Parameters
    ----------
    n
        Degree of the Legendre polynomial. Only odd ``n``, which is
        what a square Legendre grid of odd side length needs,
        reproduces EMSphInx: it always writes ``n // 2 + n % 2``
        values and sets the middle (zero) root explicitly for odd
        ``n`` only.

    Returns
    -------
    roots
        Non-negative roots in descending order in an array of shape
        ``(n // 2 + n % 2,)`` and 64-bit floating point data type. For
        odd ``n`` the last entry is exactly ``0.0``.

    Notes
    -----
    The roots come from :func:`numpy.polynomial.legendre.leggauss`
    (Golub-Welsch), while EMSphInx bisects the Sturm sequence of the
    Jacobi matrix (Barth, Martin and Wilkinson). The two agree to
    1e-13.
    """
    roots = np.ascontiguousarray(leggauss(n)[0][::-1][: n // 2 + n % 2])
    if n % 2 == 1:
        roots[-1] = 0.0
    return roots


def legendre_cos_latitudes(dim: int) -> np.ndarray:
    """Return the cosines of the ring latitudes of a Legendre grid.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    cos_lats
        Cosines of the ring latitudes from the north pole to the
        equator in an array of shape ``(n_rings(dim),)`` and 64-bit
        floating point data type. The first entry is the explicit
        pole ``1.0`` and the remaining entries are
        ``legendre_roots(dim - 2)``, so the last entry is exactly
        ``0.0`` (the equator).

    Raises
    ------
    ValueError
        If ``dim`` is invalid.
    """
    validate_dim(dim)
    cos_lats = np.empty(n_rings(dim), dtype=np.float64)
    cos_lats[0] = 1.0
    cos_lats[1:] = legendre_roots(dim - 2)
    return cos_lats


def cos_latitudes(dim: int, layout: str) -> np.ndarray:
    """Return the cosines of the ring latitudes of a square grid.

    Parameters
    ----------
    dim
        Side length of the square grid.
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    cos_lats
        Cosines of the ring latitudes from the north pole to the
        equator in an array of shape ``(n_rings(dim),)`` and 64-bit
        floating point data type.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or ``layout`` is unknown.

    See Also
    --------
    lambert_cos_latitudes
    legendre_cos_latitudes
    """
    validate_dim(dim)
    _validate_layout(layout)
    if layout == "lambert":
        return lambert_cos_latitudes(dim)
    return legendre_cos_latitudes(dim)


# ---------------------------- Grid normals -------------------------- #


def lambert_normals(dim: int) -> np.ndarray:
    """Return the unit vectors of a square Lambert grid.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    v
        Unit vectors of the northern hemisphere in an array of shape
        ``(dim, dim, 3)`` and 64-bit floating point data type, where
        ``v[j, i]`` is ``square_to_sphere([i / (dim - 1),
        j / (dim - 1)])``.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.
    """
    validate_dim(dim)
    fractions = np.arange(dim, dtype=np.float64) / (dim - 1)
    xy = np.empty((dim * dim, 2), dtype=np.float64)
    xy[:, 0] = np.tile(fractions, dim)
    xy[:, 1] = np.repeat(fractions, dim)
    return _square_to_sphere_kernel(xy).reshape(dim, dim, 3)


def legendre_normals(dim: int) -> np.ndarray:
    """Return the unit vectors of a square Legendre grid.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    v
        Unit vectors of the northern hemisphere in an array of shape
        ``(dim, dim, 3)`` and 64-bit floating point data type.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.

    Notes
    -----
    The azimuths are those of the square rings, exactly as in
    :func:`lambert_normals`, while the polar angles come from
    :func:`legendre_cos_latitudes`, i.e. ``v[j, i, 2]`` is
    ``cos_latitudes(dim, "legendre")[ring_number(dim)[j, i]]``.
    """
    validate_dim(dim)
    cos_lats = legendre_cos_latitudes(dim)
    half = dim // 2
    offsets = np.arange(dim, dtype=np.int64) - half
    r_i = np.broadcast_to(offsets, (dim, dim))
    r_j = np.broadcast_to(offsets[:, np.newaxis], (dim, dim))
    a_i = np.abs(r_i)
    a_j = np.abs(r_j)
    a_r = np.maximum(a_i, a_j)
    # The pole has a_r == 0 and is overwritten below
    ring = np.where(a_r == 0, 1, a_r).astype(np.float64)
    s_x = r_i / ring
    s_y = r_j / ring
    # The two products of EMSphInx round differently
    qq_x = _PI_4 * s_x * s_y
    qq_y = _PI_4 * s_y * s_x
    along_y = a_i <= a_j
    x = np.where(along_y, s_y * np.sin(qq_x), s_x * np.cos(qq_y))
    y = np.where(along_y, s_y * np.cos(qq_x), s_x * np.sin(qq_y))
    hypotenuse = np.where(a_r == 0, 1.0, np.hypot(x, y))
    z = cos_lats[a_r]
    sin_theta = np.sqrt(1 - z * z)
    v = np.empty((dim, dim, 3), dtype=np.float64)
    v[..., 0] = sin_theta * x / hypotenuse
    v[..., 1] = sin_theta * y / hypotenuse
    v[..., 2] = z
    v[half, half] = (0.0, 0.0, 1.0)
    return v


def normals(dim: int, layout: str) -> np.ndarray:
    """Return the unit vectors of a square grid.

    Parameters
    ----------
    dim
        Side length of the square grid.
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    v
        Unit vectors of the northern hemisphere in an array of shape
        ``(dim, dim, 3)`` and 64-bit floating point data type. The
        southern hemisphere vectors are the same with the sign of the
        third component flipped.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or ``layout`` is unknown.

    See Also
    --------
    lambert_normals
    legendre_normals
    """
    validate_dim(dim)
    _validate_layout(layout)
    if layout == "lambert":
        return lambert_normals(dim)
    return legendre_normals(dim)


# ------------------------------- Rings ------------------------------ #


def ring_number(dim: int) -> np.ndarray:
    """Return the ring number of every pixel of a hemisphere.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    rings
        Ring numbers in an array of shape ``(dim, dim)`` and 64-bit
        integer data type, i.e. the Chebyshev distance
        ``max(|i - dim // 2|, |j - dim // 2|)`` of pixel ``[j, i]``
        from the centre pixel.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.
    """
    validate_dim(dim)
    distance = np.abs(np.arange(dim, dtype=np.int64) - dim // 2)
    return np.maximum(distance[:, np.newaxis], distance[np.newaxis, :])


def ring_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the flat pixel indices of every ring, in azimuth order.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    offsets
        Start of each ring in ``flat``, in an array of shape
        ``(n_rings(dim) + 1,)`` and 64-bit integer data type.
        ``offsets[0]`` is 0, ``offsets[y + 1] - offsets[y]`` is
        ``max(1, 8 * y)`` and ``offsets[-1]`` is ``dim * dim``.
    flat
        Flat (row-major) pixel indices in an array of shape
        ``(dim * dim,)`` and 64-bit integer data type, where ring
        ``y`` occupies ``flat[offsets[y]:offsets[y + 1]]``. Every
        pixel of the hemisphere appears exactly once.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.

    Notes
    -----
    This replaces EMSphInx' ``readRing()``/``writeRing()`` with a
    precomputed index array: slot 0 of ring ``y`` is the pixel
    ``(row=dim // 2, col=dim // 2 + y)`` at azimuth zero and slot 1 is
    ``(row=dim // 2 + 1, col=dim // 2 + y)``, so the ring is walked
    counter-clockwise and slot ``p`` sits at azimuth
    ``2 * pi * p / (8 * y)``.
    """
    validate_dim(dim)
    n_ring = n_rings(dim)
    half = dim // 2
    counts = np.maximum(1, 8 * np.arange(n_ring, dtype=np.int64))
    offsets = np.zeros(n_ring + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    flat = np.empty(dim * dim, dtype=np.int64)
    flat[0] = half * dim + half
    for ring in range(1, n_ring):
        edge = np.arange(ring, dtype=np.int64)
        side = np.arange(2 * ring + 1, dtype=np.int64)
        inner = np.arange(1, ring, dtype=np.int64)
        buffer = np.empty(8 * ring, dtype=np.int64)
        # +x edge from the azimuth zero point to the quadrant 1 corner
        buffer[:ring] = (half + edge) * dim + (half + ring)
        # +y edge, quadrant 1 corner to quadrant 2 corner
        buffer[ring : 3 * ring + 1] = (half + ring) * dim + (half + ring - side)
        # -x edge, quadrant 2 corner to y == 0 and on to quadrant 3
        buffer[4 * ring - edge] = (half + edge) * dim + (half - ring)
        buffer[5 * ring - inner] = (half - ring + inner) * dim + (half - ring)
        # -y edge, quadrant 3 corner to quadrant 4 corner
        buffer[5 * ring : 7 * ring + 1] = (half - ring) * dim + (half - ring + side)
        # +x edge, quadrant 4 corner back towards y == 0
        buffer[7 * ring + inner] = (half - ring + inner) * dim + (half + ring)
        flat[offsets[ring] : offsets[ring + 1]] = buffer
    return offsets, flat


# ---------------------------- Solid angles -------------------------- #


def ring_solid_angles(dim: int, layout: str) -> np.ndarray:
    """Return the solid angle of a pixel of each ring, relative to the
    average pixel solid angle.

    Parameters
    ----------
    dim
        Side length of the square grid.
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    solid_angles
        Ratio of the actual to the average pixel solid angle for each
        ring from the north pole to the equator, in an array of shape
        ``(n_rings(dim),)`` and 64-bit floating point data type.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or ``layout`` is unknown.

    Notes
    -----
    The last (equatorial) entry covers a full band straddling the
    equator, so a caller summing over one hemisphere only must halve
    it.
    """
    cos_lats = cos_latitudes(dim, layout)
    # Average pixel solid angle, with the factor 2 pi divided out
    average = 2 / n_grid_points(dim)
    cos_a = cos_lats[:-1]
    cos_b = cos_lats[1:]
    # cos((a + b) / 2) with a = acos(cos_a) and b = acos(cos_b), i.e.
    # the ring latitudes half way between two rings
    splits = (
        np.sqrt((1 + cos_a) * (1 + cos_b)) - np.sqrt((1 - cos_a) * (1 - cos_b))
    ) / 2
    # The equatorial band is symmetric about the equator
    splits = np.append(splits, -splits[-1])
    # Spherical cap areas, then ring band areas
    caps = 1 - splits
    bands = np.empty_like(caps)
    bands[0] = caps[0]
    bands[1:] = caps[1:] - caps[:-1]
    n_phi = np.maximum(1, 8 * np.arange(caps.size))
    return bands / (average * n_phi)


def lambert_solid_angles(dim: int) -> np.ndarray:
    """Return the solid angle of every pixel of a square Lambert grid,
    relative to the average pixel solid angle.

    Parameters
    ----------
    dim
        Side length of the square grid.

    Returns
    -------
    solid_angles
        Ratio of the actual to the average pixel solid angle in an
        array of shape ``(dim, dim)`` and 64-bit floating point data
        type.

    Raises
    ------
    ValueError
        If ``dim`` is invalid.

    Notes
    -----
    Each value is the exact solid angle of the spherical quadrilateral
    spanned by the four pixel corners, computed with equation 25 of
    Mazonka, O.: "Solid angle of conical surfaces, polyhedral cones,
    and intersecting spherical caps", arXiv:1205.1396 (2012), and
    normalized by ``4 * pi / n_grid_points(dim)``. Pixels on the edge
    of the square straddle the equator, so their solid angle is that
    of the full pixel (both halves): a caller summing over one
    hemisphere only must halve edge pixels and quarter corner pixels.

    Even though a Lambert grid is equal-area in the continuum limit,
    the pixels are not: the pole pixel converges to ``2 / pi`` from
    above.
    """
    validate_dim(dim)
    inverse_average = n_grid_points(dim) / (4 * np.pi)
    mid = dim // 2
    delta = 0.5 / (dim - 1)
    # One eighth of the grid: rows from the pole to the equator and
    # columns from the diagonal outwards, the rest follows by symmetry
    row, col = np.triu_indices(dim - mid)
    row += mid
    col += mid
    at_last_row = row == dim - 1
    at_last_col = col == dim - 1
    y = row / (dim - 1)
    x = col / (dim - 1)
    # Pixel extents, without crossing the equator
    y_minus = y - delta
    y_plus = y + np.where(at_last_row, 0.0, delta)
    x_minus = x - delta
    x_plus = x + np.where(at_last_col, 0.0, delta)
    corners = np.stack(
        [
            square_to_sphere(np.column_stack((x_minus, y_minus))),
            square_to_sphere(np.column_stack((x_plus, y_minus))),
            square_to_sphere(np.column_stack((x_plus, y_plus))),
            square_to_sphere(np.column_stack((x_minus, y_plus))),
        ]
    )
    # Mazonka's equation 25: the solid angle is arg(product)
    product = np.ones(row.size, dtype=np.complex128)
    for j in range(4):
        s_previous = corners[(j + 3) % 4]
        s_this = corners[j]
        s_next = corners[(j + 1) % 4]
        a_j = np.sum(s_previous * s_next, axis=-1)
        b_j = np.sum(s_previous * s_this, axis=-1)
        c_j = np.sum(s_this * s_next, axis=-1)
        d_j = np.sum(s_previous * np.cross(s_this, s_next), axis=-1)
        product = product * (b_j * c_j - a_j + 1j * d_j)
    # Pixels on an edge of the grid are half below the equator
    factor = np.where(at_last_row, 2.0, 1.0) * np.where(at_last_col, 2.0, 1.0)
    values = -np.arctan2(product.imag, product.real) * factor * inverse_average
    solid_angles = np.empty((dim, dim), dtype=np.float64)
    solid_angles[row, col] = values
    solid_angles[col, row] = values
    solid_angles[mid:, :mid] = solid_angles[mid:, dim - 1 : mid : -1]
    solid_angles[:mid] = solid_angles[dim - 1 : mid : -1]
    return solid_angles


# ------------------------- Quadrature weights ----------------------- #


def _ring_weights_skip(dim: int, cos_lats: np.ndarray, skip: int) -> np.ndarray:
    """Return the unscaled Sneeuw ring quadrature weights with one
    ring excluded.

    Parameters
    ----------
    dim
        Side length of the square grid.
    cos_lats
        Cosines of the ring latitudes from the north pole to the
        equator, of shape ``(n_rings(dim),)``.
    skip
        Ring to exclude from the system, e.g. 0 to skip the poles.

    Returns
    -------
    w_hat
        Unscaled ring weights in an array of shape
        ``(n_rings(dim),)`` and 64-bit floating point data type, with
        ``w_hat[skip] == 0`` and ``sum(w_hat) == 1``.

    Raises
    ------
    ValueError
        If the weights are too imprecise, i.e. if
        ``|sum(w_hat) - 1| > cbrt(eps) / 64``.

    Notes
    -----
    The weights solve Sneeuw's linear system :cite:`sneeuw1994global`
    ``A @ w_hat = b`` with ``A[j, i] = T_j(2 * cos_lats[i] ** 2 - 1)``
    and ``b = [1, -1 / (4 * 1 ** 2 - 1), ...]``, i.e. a
    Chebyshev-Vandermonde system, which is ill-conditioned: for the
    Lambert layout the precision guard trips around ``dim`` 277-301,
    and ``dim = 401`` fails by orders of magnitude.

    Ring ``y`` has ``8 * y`` real samples, so bin ``m = 4 * y`` of its
    real FFT is a structurally real Nyquist bin. That ring is
    therefore excluded from the weights used for orders
    ``4 * y <= m < 4 * (y + 1)``.

    EMSphInx tests the signed residual ``sum(w_hat) - 1`` only. Here
    the absolute residual is tested, because a large negative
    residual is equally unusable. This is a deliberate deviation.
    """
    n_matrix = n_rings(dim) - 1
    kept = np.delete(np.asarray(cos_lats, dtype=np.float64), skip)
    # Chebyshev recursion T_n(x) = 2 x T_(n-1)(x) - T_(n-2)(x) with
    # x = cos(2 theta) = 2 cos(theta) ** 2 - 1, so that
    # a[j, i] = cos(2 j theta_i)
    x = kept * kept * 2 - 1
    a = np.empty((n_matrix, n_matrix), dtype=np.float64)
    a[0] = 1
    if n_matrix > 1:
        a[1] = x
    for j in range(2, n_matrix):
        a[j] = x * a[j - 1] * 2 - a[j - 2]
    b = np.empty(n_matrix, dtype=np.float64)
    b[0] = 1
    j = np.arange(1, n_matrix)
    b[1:] = -1 / (4 * j * j - 1)
    w_hat = np.linalg.solve(a, b)
    residual = np.sum(w_hat) - 1
    if abs(residual) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Insufficient precision to compute the ring weights of the square "
            f"grid of side length {dim} skipping ring {skip}: they sum to "
            f"1 + {residual:.3e}, while at most 1 + {_WEIGHT_SUM_TOLERANCE:.3e} "
            "is tolerated. Use the 'legendre' layout instead"
        )
    return np.insert(w_hat, skip, 0.0)


def quadrature_weights(dim: int, layout: str) -> np.ndarray:
    """Return the ring quadrature weights of a square grid.

    Parameters
    ----------
    dim
        Side length of the square grid.
    layout
        Either ``"lambert"`` or ``"legendre"``.

    Returns
    -------
    weights
        Weights in an array of shape ``(n_weights, n_rings(dim))``
        and 64-bit floating point data type, with
        ``n_weights = (dim - 2) // 4 + 1``. Row ``k`` is the weight
        set with ring ``k`` excluded and is used for the harmonic
        orders ``4 * k <= m < 4 * (k + 1)``. Entry ``[k, y]`` is
        ``4 * pi * w_hat[k, y] / max(1, 8 * y)``, i.e. the
        ``1 / N_phi(y)`` normalization of the forward ring transform
        is already folded in.

    Raises
    ------
    ValueError
        If ``dim`` is invalid, ``layout`` is unknown or the weights
        are too imprecise (see :func:`_ring_weights_skip`).

    Notes
    -----
    For the ``"legendre"`` layout only the ``skip = 0`` set is solved
    and replicated to all rows, because Gauss-Legendre quadrature is
    exact: the solution is the Gauss-Legendre weight set with the
    equator weight halved.
    """
    cos_lats = cos_latitudes(dim, layout)
    n_ring = n_rings(dim)
    n_weights = (dim - 2) // 4 + 1
    w_hat = np.empty((n_weights, n_ring), dtype=np.float64)
    if layout == "legendre":
        w_hat[:] = _ring_weights_skip(dim, cos_lats, 0)
    else:
        for skip in range(n_weights):
            w_hat[skip] = _ring_weights_skip(dim, cos_lats, skip)
    n_phi = np.maximum(1, 8 * np.arange(n_ring))
    return 4 * np.pi * w_hat / n_phi
