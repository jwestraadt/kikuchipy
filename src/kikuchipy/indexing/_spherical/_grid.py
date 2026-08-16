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
# Python/NumPy/Numba for kikuchipy and conveyed under
# GPL-3.0-or-later
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

import numpy as np

LAYOUTS = ("lambert", "legendre")


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# ------------------ Square Lambert (Roşca) mapping ------------------ #


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    Notes
    -----
    The values are accumulated with the integer recursion of EMSphInx
    (``numer -= delta; delta += 8``), which rounds once, and not with
    the closed form, which rounds twice. The Chebyshev-Vandermonde
    system solved in :func:`quadrature_weights` is ill-conditioned
    enough for the difference to matter at large ``dim``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


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

    Notes
    -----
    The azimuths are those of the square rings, exactly as in
    :func:`lambert_normals`, while the polar angles come from
    :func:`legendre_cos_latitudes`, i.e. ``v[j, i, 2]`` is
    ``cos_latitudes(dim, "legendre")[ring_number(dim)[j, i]]``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


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

    Notes
    -----
    This replaces EMSphInx' ``readRing()``/``writeRing()`` with a
    precomputed index array: slot 0 of ring ``y`` is the pixel
    ``(row=dim // 2, col=dim // 2 + y)`` at azimuth zero and slot 1 is
    ``(row=dim // 2 + 1, col=dim // 2 + y)``, so the ring is walked
    counter-clockwise and slot ``p`` sits at azimuth
    ``2 * pi * p / (8 * y)``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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

    Notes
    -----
    Each value is the exact solid angle of the spherical quadrilateral
    spanned by the four pixel corners, computed with Mazonka's formula
    :cite:`mazonka2012solid` and normalized by
    ``4 * pi / n_grid_points(dim)``. Pixels on the edge of the square
    straddle the equator, so their solid angle is that of the full
    pixel (both halves): a caller summing over one hemisphere only
    must halve edge pixels and quarter corner pixels.

    Even though a Lambert grid is equal-area in the continuum limit,
    the pixels are not: the pole pixel converges to ``2 / pi`` from
    above.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
