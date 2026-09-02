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
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517):
# - ZYZ Euler angles to quaternion (``xtal::zyz2qu()`` in
#   ``include/xtal/rotations.hpp``, lines 973-989)
# - Quaternion to ZYZ Euler angles (``xtal::qu2zyz()`` in the same
#   file, lines 996-1022)
# - The pi rotation axis convention (``xtal::detail::orientAxis()``
#   in the same file, lines 247-260)
# - The numerical constants ``rEps`` and ``thr``
#   (``include/xtal/constants.hpp``, lines 95-96)
# - The wrap of the middle Euler angle into [-pi, pi]
#   (``Correlator::derivatives()`` in ``include/sht/sht_xcorr.hpp``,
#   lines 895-899)
# - The crystal to sample conjugation of the indexed orientation
#   (``Indexer::indexImage()`` in ``include/idx/indexer.hpp``, lines
#   264-268)
#
# EMSphInx' own ZYZ to Bunge helpers ``xtal::zyz2eu()`` and
# ``xtal::eu2zyz()`` (``include/xtal/rotations.hpp``, lines 1025-1039)
# are deliberately NOT ported as written: their offsets are the
# reverse of the ones ``zyz2qu()`` implements, see the Notes of
# :func:`zyz_to_bunge`.

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
# Changed by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""ZYZ Euler angle conversions for spherical indexing.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

EMSphInx describes rotations of the sphere with *passive* ZYZ Euler
angles ``(alpha, beta, gamma)``, i.e. a rotation by ``gamma`` about
z, then by ``beta`` about the new y, then by ``alpha`` about the new
z. The Wigner (uppercase) D function of
:mod:`kikuchipy.indexing._spherical._wigner` and the spherical cross
correlation are both written in these angles, while kikuchipy, orix
and EMsoft use Bunge ZXZ angles ``(phi1, Phi, phi2)``.

The two are related by a pure affine shift of the outer angles,

.. code-block::

    (phi1, Phi, phi2) = (alpha + pi/2, beta, gamma - pi/2)
    (alpha, beta, gamma) = (phi1 - pi/2, Phi, phi2 + pi/2)

which is the relation EMSphInx' own test asserts
(``test/xtal/rotations.cpp``, lines 288-318) and which reproduces
``zyz2qu()`` to 8.6e-16 through :meth:`orix.quaternion.Rotation.
from_euler`. The ``zyz2eu()``/``eu2zyz()`` helpers of
``rotations.hpp`` state the offsets the other way round and are not
ported, see :func:`zyz_to_bunge`.

Quaternions are ``(w, x, y, z)`` with ``pijk = +1``
(``include/constants.hpp``, line 66), which is orix'
``(a, b, c, d)`` convention, and are restricted to ``w >= 0``.
"""

import math

from numba import njit
import numpy as np
from orix.quaternion import Rotation

# Machine epsilon derived constants of EMSphInx'
# xtal::Constants<Real> (``include/xtal/constants.hpp``, lines 95-96):
# rEps is the largest |w| still treated as a rotation by pi and thr
# the degeneracy threshold of qu2zyz()
_EPS = float(np.finfo(np.float64).eps)
_R_EPS = math.sqrt(_EPS)
_THR = 10 * _EPS

# Full and quarter turn, EMSphInx' xtal::Constants<Real>::pi2 and
# ::pi_2 (``include/xtal/constants.hpp``, lines 81-82); the quarter
# turn is the ZYZ to Bunge offset of zyz_to_bunge()
_TWO_PI = 2 * math.pi
_PI_2 = math.pi / 2


# ------------------------------ Helpers ----------------------------- #


def _as_float64(values: np.ndarray, size: int, name: str) -> np.ndarray:
    """Return an array-like as 64-bit floats with a checked shape.

    Parameters
    ----------
    values
        Array-like expected to have shape ``(..., size)``.
    size
        Required length of the last dimension, three for Euler angle
        triples and four for quaternions.
    name
        Parameter name to use in the error message.

    Returns
    -------
    array
        ``values`` as an array of 64-bit floating point data type.

    Raises
    ------
    ValueError
        If ``values`` is zero-dimensional or its last dimension is
        not ``size``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != size:
        raise ValueError(
            f"{name} must have shape (..., {size}), but has shape {array.shape}"
        )
    return array


# --------------------------- Angle utility -------------------------- #


@njit(cache=True, nogil=True)
def _wrap_beta(beta: float) -> float:
    """Return the middle ZYZ Euler angle wrapped into [-pi, pi].

    Parameters
    ----------
    beta
        Middle ZYZ Euler angle in radians.

    Returns
    -------
    wrapped
        ``fmod(beta, 2 * pi)``, then shifted by one full turn if the
        result left the closed interval [-pi, pi]. Negative zero is
        preserved, since ``fmod(-0.0, x)`` is ``-0.0``.

    Notes
    -----
    This is the wrap EMSphInx applies to the middle angle before
    every Wigner d evaluation (``Correlator::derivatives()`` in
    ``include/sht/sht_xcorr.hpp``, lines 895-899). It is needed
    because the Wigner functions take ``cos(beta)`` and
    ``signbit(beta)`` separately, and the sign bit only identifies
    the branch inside [-pi, pi].

    Numba has no ``math.fmod`` in nopython mode (``Unknown attribute
    'fmod' of type Module(math)``, checked on 0.65 and on the 0.67
    installed here), so this kernel must use :func:`numpy.fmod`. It
    is the C library ``fmod``, so it takes the sign of the dividend
    and returns ``-0.0`` for ``-0.0``, which is the behaviour the
    negative zero test pins.
    """
    # numpy's fmod, since Numba has no math.fmod, see the Notes
    wrapped = np.fmod(beta, _TWO_PI)
    if wrapped > math.pi:
        wrapped -= _TWO_PI
    elif wrapped < -math.pi:
        wrapped += _TWO_PI
    return wrapped


def wrap_beta(beta: float) -> float:
    """Return the middle ZYZ Euler angle wrapped into [-pi, pi].

    Parameters
    ----------
    beta
        Middle ZYZ Euler angle in radians.

    Returns
    -------
    wrapped
        ``beta`` shifted by a whole number of turns into the closed
        interval [-pi, pi], with negative zero preserved.

    Notes
    -----
    Thin wrapper of the Numba kernel :func:`_wrap_beta`, which is the
    one the compiled kernels of
    :mod:`kikuchipy.indexing._spherical._wigner` call, so that there
    is a single implementation of the wrap.
    """
    return float(_wrap_beta(float(beta)))


# ------------------------ ZYZ and quaternions ----------------------- #


@njit(cache=True, nogil=True)
def _orient_axis(ax: np.ndarray) -> None:
    """Canonicalize the axis of a rotation by pi, in place.

    Parameters
    ----------
    ax
        Rotation axis ``(x, y, z)`` in an array of shape ``(3,)`` and
        64-bit floating point data type, typically a view of the
        vector part of a quaternion. It is modified in place.

    Notes
    -----
    Port of ``xtal::detail::orientAxis()``
    (``include/xtal/rotations.hpp``, lines 247-260). A rotation by pi
    about ``ax`` and about ``-ax`` are the same rotation, so one of
    the two is chosen: the northern hemisphere when ``|z| >= rEps``,
    the ``+y`` half of the equator when ``z`` is zero but ``y`` is
    not (in which case ``x`` and ``y`` are renormalized, since ``z``
    was zeroed), and ``(1, 0, 0)`` when both ``y`` and ``z`` are
    zero.
    """
    if abs(ax[2]) < _R_EPS:
        # z is zero, so the axis is on the equator
        ax[2] = 0.0
        if abs(ax[1]) < _R_EPS:
            # y and z are zero, so use (1, 0, 0), not (-1, 0, 0)
            ax[1] = 0.0
            ax[0] = 1.0
        else:
            # renormalize, since z was zeroed, and keep the +y half
            mag = math.copysign(math.sqrt(ax[0] * ax[0] + ax[1] * ax[1]), ax[1])
            ax[0] /= mag
            ax[1] /= mag
    elif math.copysign(1.0, ax[2]) < 0.0:
        # z is non-zero, so use the northern hemisphere
        ax[0] = -ax[0]
        ax[1] = -ax[1]
        ax[2] = -ax[2]


@njit(cache=True, nogil=True)
def _zyz_to_quaternion_single(zyz: np.ndarray, qu: np.ndarray) -> None:
    """Write one ZYZ Euler triple as a quaternion, in place.

    Parameters
    ----------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array of shape ``(3,)`` and 64-bit floating point data type.
    qu
        Array of shape ``(4,)`` and 64-bit floating point data type
        to write ``(w, x, y, z)`` into.

    Notes
    -----
    Port of ``xtal::zyz2qu()`` (``include/xtal/rotations.hpp``, lines
    973-989) with ``pijk = +1``: with ``c = cos(beta / 2)``,
    ``s = sin(beta / 2)``, ``sigma = (gamma + alpha) / 2`` and
    ``delta = (gamma - alpha) / 2``, the quaternion is
    ``(c cos(sigma), -s sin(delta), -s cos(delta), -c sin(sigma))``,
    negated when ``signbit(w)`` so that ``w >= 0`` and negative zero
    flips too, and, when ``|w| <= rEps``, given an exactly zero ``w``
    and an axis canonicalized by :func:`_orient_axis`.

    Note the operand order of ``sigma`` and ``delta``: they are built
    from ``gamma +- alpha``, which is the reverse of the ZXZ
    ``eu2qu()`` at lines 410-429. Swapping them is a silent 180
    degree error.

    The result is not normalized, as in EMSphInx (``eu2qu()``
    normalizes, ``zyz2qu()`` does not; the trigonometric form is
    already unit length to rounding).
    """
    c = math.cos(zyz[1] / 2)
    s = math.sin(zyz[1] / 2)
    # gamma +- alpha, the reverse of the ZXZ eu2qu() operand order
    sigma = (zyz[2] + zyz[0]) / 2
    delta = (zyz[2] - zyz[0]) / 2
    qu[0] = c * math.cos(sigma)
    qu[1] = -(s * math.sin(delta))
    qu[2] = -(s * math.cos(delta))
    qu[3] = -(c * math.sin(sigma))
    if math.copysign(1.0, qu[0]) < 0.0:
        # signbit(w), so that -0.0 flips too, restricting the
        # rotation angle to [0, pi]
        qu[0] = -qu[0]
        qu[1] = -qu[1]
        qu[2] = -qu[2]
        qu[3] = -qu[3]
    if abs(qu[0]) <= _R_EPS:
        # the rotation is by pi, whose axis is ambiguous
        qu[0] = 0.0
        _orient_axis(qu[1:])


@njit(cache=True, nogil=True)
def _zyz_to_quaternion_2d(zyz2d: np.ndarray) -> np.ndarray:
    """Return quaternions of a two-dimensional array of ZYZ triples.

    Parameters
    ----------
    zyz2d
        ZYZ Euler angles in radians in an array of shape ``(n, 3)``
        and 64-bit floating point data type.

    Returns
    -------
    qu2d
        Quaternions ``(w, x, y, z)`` in an array of shape ``(n, 4)``
        and 64-bit floating point data type.

    Notes
    -----
    Calls :func:`_zyz_to_quaternion_single` per row.
    """
    n = zyz2d.shape[0]
    qu2d = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        _zyz_to_quaternion_single(zyz2d[i], qu2d[i])
    return qu2d


@njit(cache=True, nogil=True)
def _quaternion_to_zyz_single(qu: np.ndarray, eu: np.ndarray) -> None:
    """Write one quaternion as ZYZ Euler angles, in place.

    Parameters
    ----------
    qu
        Quaternion ``(w, x, y, z)`` in an array of shape ``(4,)`` and
        64-bit floating point data type.
    eu
        Array of shape ``(3,)`` and 64-bit floating point data type
        to write ``(alpha, beta, gamma)`` in radians into.

    Notes
    -----
    Port of ``xtal::qu2zyz()`` (``include/xtal/rotations.hpp``, lines
    996-1022) with ``pijk = +1`` and ``thr = 10 eps``. With
    ``q03 = w^2 + z^2``, ``q12 = x^2 + y^2`` and
    ``chi = sqrt(q03 q12)``:

    - ``chi <= thr`` and ``q12 <= thr``: ``beta = 0``,
      ``alpha = atan2(-2 w z, w^2 - z^2)``, ``gamma = 0``.
    - ``chi <= thr`` otherwise: ``beta = pi``,
      ``alpha = atan2(-2 x y, y^2 - x^2)``, ``gamma = 0``.
    - Otherwise ``alpha = atan2(y z + x w, -y w + x z)``,
      ``beta = atan2(2 chi, q03 - q12)`` and
      ``gamma = atan2(y z - x w, -y w - x z)``.

    Every negative angle is finally shifted by ``+2 pi``, so the
    ranges are ``[0, 2 pi) x [0, pi] x [0, 2 pi)``. On the degenerate
    branches only ``alpha +- gamma`` is determined, and the value
    returned is the Python modulo ``(alpha + gamma) % (2 pi)``
    (``beta = 0``) or ``(alpha - gamma) % (2 pi)`` (``beta = pi``),
    not the C ``fmod``, which is negative whenever the difference is.
    """
    w = qu[0]
    x = qu[1]
    y = qu[2]
    z = qu[3]
    q03 = w * w + z * z
    q12 = x * x + y * y
    chi = math.sqrt(q03 * q12)
    if chi <= _THR:
        if q12 <= _THR:
            # a rotation about z alone
            eu[0] = math.atan2(-2.0 * w * z, w * w - z * z)
            eu[1] = 0.0
        else:
            # a rotation by pi about an axis on the equator
            eu[0] = math.atan2(-2.0 * x * y, y * y - x * x)
            eu[1] = math.pi
        eu[2] = 0.0
    else:
        # atan2 is magnitude independent, so chi is not divided out
        y1 = y * z
        y2 = -(x * w)
        x1 = -(y * w)
        x2 = -(x * z)
        eu[0] = math.atan2(y1 - y2, x1 - x2)
        eu[1] = math.atan2(2.0 * chi, q03 - q12)
        eu[2] = math.atan2(y1 + y2, x1 + x2)
    for i in range(3):
        if eu[i] < 0.0:
            eu[i] += _TWO_PI


@njit(cache=True, nogil=True)
def _quaternion_to_zyz_2d(qu2d: np.ndarray) -> np.ndarray:
    """Return ZYZ triples of a two-dimensional array of quaternions.

    Parameters
    ----------
    qu2d
        Quaternions ``(w, x, y, z)`` in an array of shape ``(n, 4)``
        and 64-bit floating point data type.

    Returns
    -------
    zyz2d
        ZYZ Euler angles in radians in an array of shape ``(n, 3)``
        and 64-bit floating point data type.

    Notes
    -----
    Calls :func:`_quaternion_to_zyz_single` per row.
    """
    n = qu2d.shape[0]
    zyz2d = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        _quaternion_to_zyz_single(qu2d[i], zyz2d[i])
    return zyz2d


def zyz_to_quaternion(zyz: np.ndarray) -> np.ndarray:
    """Return quaternions of ZYZ Euler angles.

    This is the single source of truth for the ZYZ Euler angle to
    orientation conversion of the spherical indexing code.

    Parameters
    ----------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array-like of shape ``(..., 3)``. It is cast to 64-bit
        floating point.

    Returns
    -------
    qu
        Quaternions ``(w, x, y, z)`` in an array of shape
        ``(..., 4)`` and 64-bit floating point data type, all with
        ``w >= 0``. Rotations by pi get an exactly zero ``w`` and the
        canonical axis of :func:`_orient_axis`.

    Raises
    ------
    ValueError
        If the last dimension of ``zyz`` is not three.

    Notes
    -----
    Port of ``xtal::zyz2qu()`` (``include/xtal/rotations.hpp``, lines
    973-989). The quaternion equals
    ``orix.quaternion.Rotation.from_euler(zyz_to_bunge(zyz)).data``
    to 8.6e-16, which is the evidence for the Bunge offsets of
    :func:`zyz_to_bunge`.
    """
    zyz = _as_float64(zyz, 3, "zyz")
    zyz2d = np.ascontiguousarray(zyz.reshape(-1, 3))
    return _zyz_to_quaternion_2d(zyz2d).reshape(zyz.shape[:-1] + (4,))


def quaternion_to_zyz(qu: np.ndarray) -> np.ndarray:
    """Return ZYZ Euler angles of quaternions.

    Parameters
    ----------
    qu
        Quaternions ``(w, x, y, z)`` in an array-like of shape
        ``(..., 4)``. It is cast to 64-bit floating point. A
        quaternion and its negative give the same angles.

    Returns
    -------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array of shape ``(..., 3)`` and 64-bit floating point data
        type, in ``[0, 2 pi) x [0, pi] x [0, 2 pi)``.

    Raises
    ------
    ValueError
        If the last dimension of ``qu`` is not four.

    Notes
    -----
    Port of ``xtal::qu2zyz()`` (``include/xtal/rotations.hpp``, lines
    996-1022), not :meth:`orix.quaternion.Rotation.to_euler`, whose
    degeneracy threshold is 1e-9 rather than ``10 eps``. See
    :func:`_quaternion_to_zyz_single` for the branches and what they
    return when only ``alpha +- gamma`` is determined.
    """
    qu = _as_float64(qu, 4, "qu")
    qu2d = np.ascontiguousarray(qu.reshape(-1, 4))
    return _quaternion_to_zyz_2d(qu2d).reshape(qu.shape[:-1] + (3,))


# --------------------------- ZYZ and Bunge -------------------------- #


def zyz_to_bunge(zyz: np.ndarray) -> np.ndarray:
    """Return Bunge ZXZ Euler angles of ZYZ Euler angles.

    Parameters
    ----------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array-like of shape ``(..., 3)``. It is cast to 64-bit
        floating point.

    Returns
    -------
    eu
        Bunge ZXZ Euler angles ``(alpha + pi/2, beta, gamma - pi/2)``
        in radians in an array of shape ``(..., 3)`` and 64-bit
        floating point data type. The shift is affine and the angles
        are not wrapped.

    Raises
    ------
    ValueError
        If the last dimension of ``zyz`` is not three.

    Notes
    -----
    The offsets are the ones EMSphInx' own round trip test asserts
    (``test/xtal/rotations.cpp``, lines 296-310, which builds
    ``zyz = (phi1 - pi/2, Phi, phi2 + pi/2)`` and requires
    ``zyz2qu(zyz) == eu2qu(eu)`` to ``10 eps``) and the ones for
    which :func:`zyz_to_quaternion` agrees with
    :meth:`orix.quaternion.Rotation.from_euler` (8.6e-16 on 1000
    random triples; the opposite sign convention differs by 2.0, a
    180 degree rotation about z).

    EMSphInx' own ``xtal::zyz2eu()``/``xtal::eu2zyz()``
    (``include/xtal/rotations.hpp``, lines 1025-1039) and the notes
    at lines 122, 129, 971 and 995 state these offsets reversed and
    are inconsistent with ``zyz2qu()`` in the same file. They are
    never used on the EMSphInx indexing path and are not ported.
    """
    zyz = _as_float64(zyz, 3, "zyz")
    return zyz + np.array([_PI_2, 0.0, -_PI_2])


def bunge_to_zyz(eu: np.ndarray) -> np.ndarray:
    """Return ZYZ Euler angles of Bunge ZXZ Euler angles.

    Parameters
    ----------
    eu
        Bunge ZXZ Euler angles ``(phi1, Phi, phi2)`` in radians in an
        array-like of shape ``(..., 3)``. It is cast to 64-bit
        floating point.

    Returns
    -------
    zyz
        ZYZ Euler angles ``(phi1 - pi/2, Phi, phi2 + pi/2)`` in
        radians in an array of shape ``(..., 3)`` and 64-bit floating
        point data type. The shift is affine and the angles are not
        wrapped, so this inverts :func:`zyz_to_bunge` up to the
        rounding of the two shifts: ``(x + pi/2) - pi/2`` is one unit
        in the last place away from ``x`` for a large fraction of
        doubles (38138 of 100000 random triples, worst 4.44e-16),
        while ``beta`` is untouched by both maps and comes back
        bitwise.

    Raises
    ------
    ValueError
        If the last dimension of ``eu`` is not three.
    """
    eu = _as_float64(eu, 3, "eu")
    return eu + np.array([-_PI_2, 0.0, _PI_2])


# -------------------------- ZYZ and orix ---------------------------- #


def rotation_from_zyz(zyz: np.ndarray) -> Rotation:
    """Return an orix rotation of ZYZ Euler angles.

    Parameters
    ----------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array-like of shape ``(..., 3)``. It is cast to 64-bit
        floating point.

    Returns
    -------
    rotation
        ``~Rotation(zyz_to_quaternion(zyz))`` of shape ``(...)``, so
        a single triple of shape ``(3,)`` gives a rotation of shape
        ``(1,)``.

    Raises
    ------
    ValueError
        If the last dimension of ``zyz`` is not three.

    Notes
    -----
    The conjugation is the "crystal to sample to sample to crystal"
    step EMSphInx performs on the correlated ZYZ angles
    (``Indexer::indexImage()`` in ``include/idx/indexer.hpp``, lines
    264-268). The detector quaternion which multiplies it there is
    the identity in EMSphInx as shipped, since the tilt dependent
    quaternion of ``include/modality/ebsd/detector.hpp`` line 457 is
    commented out upstream and the abstract default
    (``include/idx/base.hpp``, line 133) is the identity too.

    **The sign of this conjugation is frozen.** It is the faithful
    chain, it is consistent with the direction of
    :func:`kikuchipy.indexing._spherical._wigner.rotate_harmonics`,
    and Phase 5 (``spherical-back-projection``, D8) measured it
    against kikuchipy's forward projection: over 27 rotations of
    :meth:`~kikuchipy.signals.EBSDMasterPattern.get_patterns`
    back-projected and correlated at bandwidth 68, this conjugation
    is 0.34 degrees from the true orientation in the median and 0.72
    at worst, while the other sign is 35 degrees out.
    """
    return ~Rotation(zyz_to_quaternion(zyz))


def rotation_to_zyz(rotation: Rotation) -> np.ndarray:
    """Return ZYZ Euler angles of an orix rotation.

    Parameters
    ----------
    rotation
        Rotation of any shape ``(...)``, as returned by
        :func:`rotation_from_zyz`.

    Returns
    -------
    zyz
        ZYZ Euler angles ``(alpha, beta, gamma)`` in radians in an
        array of shape ``(..., 3)`` and 64-bit floating point data
        type, i.e. ``quaternion_to_zyz((~rotation).data)``.

    Notes
    -----
    The inverse of :func:`rotation_from_zyz` up to the ``2 pi``
    periodicity of the angles and the degeneracy of ``beta = 0`` and
    ``beta = pi``, where only ``alpha +- gamma`` is determined.  Its
    conjugation is the frozen one of :func:`rotation_from_zyz`, see
    the note there and Phase 5's D8.
    """
    return quaternion_to_zyz((~rotation).data)
