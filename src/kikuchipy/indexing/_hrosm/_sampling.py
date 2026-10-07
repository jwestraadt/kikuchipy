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
# functionality in this file is derived and adapted from EMsoftOO:
# - The filled cubochoric grid of a misorientation ball
#   sample_isoCubeFilled_ (EMsoftOOLib/mod_so3.f90)
# - The cubochoric to homochoric conversion
#   Lambert3DCubeForwardDouble with GetPyramidDouble
#   (EMsoftOOLib/mod_Lambert.f90), as called by ch_
#   (EMsoftOOLib/mod_rotations.f90)
# - The composition of the ball with a centre orientation
#   SampleIsoMisorientation_ (EMsoftOOLib/mod_so3.f90)
# - The single precision Rodrigues storage of the sampled
#   orientations (EMOpenCLLib/program_mods/mod_DI.f90)

# #####################################################################
# Copyright (c) 2013-2026, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  - Neither the names of Marc De Graef, Carnegie Mellon University nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################

# Changes by the kikuchipy developers, 2026-10-06: ported from Fortran
# to NumPy; defects reproduced only behind ``emsoft_compatible``.

"""Sampling of orientations in a misorientation ball around a centre
orientation.
"""

from __future__ import annotations

import numpy as np
from orix.quaternion import Rotation
from scipy.spatial import cKDTree

# EMsoft's Lambert parameters of the cube to ball map, as literal
# constants: pi, sqrt(pi), sqrt(6 / pi), pi^(2 / 3), the grid ratio,
# sqrt(2), pi / 12, the curved square prefactor and sqrt(24)
_LAMBERT_PI = 3.141592653589793
_LAMBERT_SPI = 1.772453850905516
_LAMBERT_PREF = 1.381976597885342
_LAMBERT_AP = 2.145029397111025
_LAMBERT_SC = 0.897772786961286
_LAMBERT_R2 = 1.414213562373095
_LAMBERT_PI12 = 0.261799387799149
_LAMBERT_PREK = 1.643456402972504
_LAMBERT_R24 = 4.898979485566356
# Tolerance of the cube bounds and of the cube origin
_LAMBERT_EPS = 1.0e-12


def misorientation_ball(
    center: Rotation | None = None,
    *,
    max_angle: float = 5.0,
    n_steps: int = 20,
    emsoft_compatible: bool = False,
) -> Rotation:
    """Return rotations sampling a ball of misorientations around a
    centre rotation.

    A cube of ``(2 n_steps + 1)**3`` points of the cubochoric grid is
    mapped onto the homochoric ball whose surface has the rotation
    angle ``max_angle`` exactly, as EMsoft's misorientation sampling
    does. Shell ``m = 1, ..., n_steps`` (the points with the largest
    absolute grid index ``m``) holds ``24 m**2 + 2`` points at one
    rotation angle.

    Parameters
    ----------
    center
        One rotation to centre the ball on. If not given (default), the
        ball is centred on the identity.
    max_angle
        Radius of the ball in degrees, the largest rotation angle away
        from ``center``. Must be in (0, 180). Default is 5.0.
    n_steps
        Number of grid steps from the centre to the surface along each
        cube axis, at least 1. Default is 20.
    emsoft_compatible
        Whether to round every rotation through EMsoft's single
        precision Rodrigues storage (axis and ``tan(omega / 2)`` as
        float32) and back to float64, as EMsoft stores the orientations
        it simulates. Default is False.

    Returns
    -------
    ball
        Rotations of shape ``((2 n_steps + 1)**3,)``. With ``N =
        n_steps``, element ``(i + N)(2N + 1)**2 + (j + N)(2N + 1) + (k
        + N)`` has the grid index ``(i, j, k)``, each from ``-N`` to
        ``N``. Element ``n`` is ``~ball_identity[n] * center``, with
        ``ball_identity`` the ball centred on the identity.

    Raises
    ------
    ValueError
        If ``max_angle`` is not in (0, 180), ``n_steps`` is not an
        integer of at least 1, or ``center`` is not one rotation.

    See Also
    --------
    misorientation_ball_spacing

    Examples
    --------
    A ball of radius 2 degrees with 3 steps from the centre to the
    surface, centred on a rotation of 30 degrees about [001]

    >>> import numpy as np
    >>> from orix.quaternion import Rotation
    >>> import kikuchipy as kp
    >>> center = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(30))
    >>> ball = kp.indexing.misorientation_ball(
    ...     center, max_angle=2, n_steps=3
    ... )
    >>> ball.size
    343
    >>> angle = np.rad2deg((ball * ~center).angle)
    >>> print(round(float(angle.max()), 6))
    2.0
    """
    max_angle, n_steps = _validate_ball_arguments(max_angle, n_steps)
    if center is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q0 = np.asarray(center.data, dtype=np.float64)
        if q0.size != 4:
            raise ValueError(
                f"center must be one rotation, not {q0.size // 4} rotations"
            )
        q0 = q0.reshape(4)

    cu = _cubochoric_grid(np.deg2rad(max_angle), n_steps)
    q_v = Rotation.from_homochoric(_cubochoric_to_homochoric(cu)).data
    q_v = np.asarray(q_v, dtype=np.float64).reshape(-1, 4)
    # Element n is conj(Q_v[n]) * center, EMsoft's Rodrigues
    # composition of the grid point with the centre
    q = _multiply(q_v * np.array([1.0, -1.0, -1.0, -1.0]), q0[None, :])
    if emsoft_compatible:
        q = _emsoft_rodrigues_round_trip(q)
    # Keep the quaternions bit for bit, without orix' renormalisation
    ball = Rotation(q)
    ball.data = q
    return ball


def misorientation_ball_spacing(max_angle: float = 5.0, n_steps: int = 20) -> float:
    """Return the mean angular spacing of the rotations sampling a
    misorientation ball.

    The spacing is the mean, over all ``(2 n_steps + 1)**3`` rotations
    of the ball, of the rotation angle to the nearest other rotation of
    the ball. It does not depend on the ball's centre.

    Parameters
    ----------
    max_angle
        Radius of the ball in degrees. Must be in (0, 180). Default is
        5.0.
    n_steps
        Number of grid steps from the centre to the surface, at least
        1. Default is 20.

    Returns
    -------
    spacing
        Mean nearest-neighbour angle in degrees.

    Raises
    ------
    ValueError
        If ``max_angle`` is not in (0, 180) or ``n_steps`` is not an
        integer of at least 1.

    See Also
    --------
    misorientation_ball

    Notes
    -----
    The mean spacing, 0.15918 degrees at the defaults, is smaller than
    the radial step between shells, ``max_angle / n_steps`` (0.25
    degrees at the defaults), since neighbours within a shell are
    closer than neighbours across shells.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> spacing = kp.indexing.misorientation_ball_spacing(5, 2)
    >>> print(round(spacing, 4))
    1.6013
    """
    max_angle, n_steps = _validate_ball_arguments(max_angle, n_steps)
    ball = misorientation_ball(None, max_angle=max_angle, n_steps=n_steps)
    q = ball.data.reshape(-1, 4)
    # Every ball point has a positive scalar part, so the chord d
    # between two unit quaternions is monotone in their angle,
    # 4 arcsin(d / 2); column 0 is the point itself
    d, _ = cKDTree(q).query(q, k=2)
    angle = 4 * np.arcsin(d[:, 1] / 2)
    return float(np.rad2deg(angle).mean())


def _validate_ball_arguments(max_angle: float, n_steps: int) -> tuple[float, int]:
    """Return ``max_angle`` as a float and ``n_steps`` as an int, or
    raise a ValueError naming the invalid argument.
    """
    if (
        isinstance(max_angle, bool)
        or not isinstance(max_angle, (int, float, np.integer, np.floating))
        or not 0 < max_angle < 180
    ):
        raise ValueError(f"max_angle {max_angle!r} must be a number in (0, 180)")
    if (
        isinstance(n_steps, bool)
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 1
    ):
        raise ValueError(f"n_steps {n_steps!r} must be an integer of at least 1")
    return float(max_angle), int(n_steps)


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a * b`` of quaternions of shape
    (..., 4), broadcast against each other.
    """
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 + a2 * b0 + a3 * b1 - a1 * b3,
            a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1,
        ],
        axis=-1,
    )


def _cubochoric_grid(max_angle_rad: float, n_steps: int) -> np.ndarray:
    """Return the cubochoric grid of a misorientation ball, shape
    ((2 n_steps + 1)**3, 3) of float64.

    The cube's half edge is ``0.5 * (pi * (w - sin(w)))**(1 / 3)`` for
    ``w = max_angle_rad``, the step is the half edge divided by
    ``n_steps``, and the first grid index varies slowest.
    """
    w = float(max_angle_rad)
    edge = (np.pi * (w - np.sin(w))) ** (1 / 3) * 0.5
    dx = edge / float(n_steps)
    idx = np.arange(-n_steps, n_steps + 1, dtype=np.float64)
    i, j, k = np.meshgrid(idx, idx, idx, indexing="ij")
    grid = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    return grid * dx


def _pyramid(xyz: np.ndarray) -> np.ndarray:
    """Return EMsoft's pyramid number 1-6 of cube points of shape (n,
    3), the first matching pyramid in EMsoft's test order.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    conditions = [
        (ax <= z) & (ay <= z),
        (ax <= -z) & (ay <= -z),
        (az <= x) & (ay <= x),
        (az <= -x) & (ay <= -x),
        (ax <= y) & (az <= y),
        (ax <= -y) & (az <= -y),
    ]
    return np.select(conditions, [1, 2, 3, 4, 5, 6], default=0)


def _cubochoric_to_homochoric(cu: np.ndarray) -> np.ndarray:
    """Return homochoric vectors of cubochoric vectors of shape (n, 3),
    in EMsoft's operation order of the forward Lambert map of the cube
    with its pyramid selection.
    """
    cu = np.asarray(cu, dtype=np.float64).reshape(-1, 3)
    n = cu.shape[0]
    ho = np.zeros((n, 3), dtype=np.float64)

    # Points outside the cube and at its origin map to zero
    max_abs = np.abs(cu).max(axis=1) if n else np.zeros(0)
    inside = (max_abs <= _LAMBERT_AP / 2 + _LAMBERT_EPS) & (max_abs >= _LAMBERT_EPS)

    p = _pyramid(cu)
    # Coordinates in the order of the pyramid pair: (1, 2) as is,
    # (3, 4) as (y, z, x), (5, 6) as (z, x, y)
    s = cu.copy()
    p34 = (p == 3) | (p == 4)
    p56 = (p == 5) | (p == 6)
    s[p34] = cu[p34][:, [1, 2, 0]]
    s[p56] = cu[p56][:, [2, 0, 1]]

    xyz = _LAMBERT_SC * s
    lam = np.zeros((n, 3), dtype=np.float64)
    nonzero = inside & (np.abs(xyz).max(axis=1) != 0)
    on_axis = nonzero & (np.abs(xyz[:, :2]).max(axis=1) == 0)
    lam[on_axis, 2] = _LAMBERT_PREF * xyz[on_axis, 2]

    general = nonzero & ~on_axis
    xg, yg, zg = xyz[general, 0], xyz[general, 1], xyz[general, 2]
    first = np.abs(yg) <= np.abs(xg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(first, yg / xg, xg / yg)
        q = _LAMBERT_PI12 * ratio
        c = np.cos(q)
        sn = np.sin(q)
        q = _LAMBERT_PREK * np.where(first, xg, yg) / np.sqrt(_LAMBERT_R2 - c)
        t1 = np.where(first, (_LAMBERT_R2 * c - 1.0) * q, _LAMBERT_R2 * sn * q)
        t2 = np.where(first, _LAMBERT_R2 * sn * q, (_LAMBERT_R2 * c - 1.0) * q)

        # Inverse Lambert projection onto the ball
        c = t1**2 + t2**2
        sn = _LAMBERT_PI * c / (24.0 * zg**2)
        c = _LAMBERT_SPI * c / _LAMBERT_R24 / zg
        q = np.sqrt(1.0 - sn)
    lam[general] = np.column_stack([t1 * q, t2 * q, _LAMBERT_PREF * zg - c])

    # Back to the original coordinate order of the pyramid
    ho[:] = lam
    ho[p34] = lam[p34][:, [2, 0, 1]]
    ho[p56] = lam[p56][:, [1, 2, 0]]
    ho[~inside] = 0.0
    return ho


def _emsoft_rodrigues_round_trip(q: np.ndarray) -> np.ndarray:
    """Return quaternions rounded through EMsoft's float32 Rodrigues
    storage.

    Each quaternion of shape (n, 4) is stored as the four-vector (unit
    axis, ``tan(omega / 2)``) in float32, ``(0, 0, 1, 0)`` for a zero
    rotation, and converted back to a quaternion in float64 as
    EMsoft's Rodrigues to quaternion conversion does.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    q = np.where(q[:, :1] < 0, -q, q)
    s = np.sqrt(np.sum(q[:, 1:] * q[:, 1:], axis=1))
    zero = s == 0
    rod = np.zeros((q.shape[0], 4), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rod[:, :3] = q[:, 1:] / s[:, None]
        rod[:, 3] = s / q[:, 0]
    rod[zero] = (0.0, 0.0, 1.0, 0.0)
    rod = rod.astype(np.float32).astype(np.float64)

    # Rodrigues to axis-angle: angle 2 arctan(t), the axis scaled by
    # its reciprocal norm; axis-angle to quaternion; normalisation
    t = rod[:, 3]
    is_zero = t == 0
    angle = 2.0 * np.arctan(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.sqrt(np.sum(rod[:, :3] * rod[:, :3], axis=1))
    axis = rod[:, :3] * inv[:, None]
    out = np.empty_like(rod)
    out[:, 0] = np.cos(angle * 0.5)
    out[:, 1:] = axis * np.sin(angle * 0.5)[:, None]
    out[is_zero] = (1.0, 0.0, 0.0, 0.0)
    norm = np.sqrt(np.sum(out**2, axis=1))
    return out / norm[:, None]
