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
    """
    # TODO: add a runnable Examples section (a small ball, its size and
    # largest angle) once the implementation exists
    raise NotImplementedError


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
    """
    # TODO: add a runnable Examples section (a small ball) once the
    # implementation exists
    raise NotImplementedError


def _cubochoric_grid(max_angle_rad: float, n_steps: int) -> np.ndarray:
    """Return the cubochoric grid of a misorientation ball, shape
    ((2 n_steps + 1)**3, 3) of float64.

    The cube's half edge is ``0.5 * (pi * (w - sin(w)))**(1 / 3)`` for
    ``w = max_angle_rad``, the step is the half edge divided by
    ``n_steps``, and the first grid index varies slowest.
    """
    raise NotImplementedError


def _cubochoric_to_homochoric(cu: np.ndarray) -> np.ndarray:
    """Return homochoric vectors of cubochoric vectors of shape (n, 3),
    in EMsoft's operation order of the forward Lambert map of the cube
    with its pyramid selection.
    """
    raise NotImplementedError


def _emsoft_rodrigues_round_trip(q: np.ndarray) -> np.ndarray:
    """Return quaternions rounded through EMsoft's float32 Rodrigues
    storage.

    Each quaternion of shape (n, 4) is stored as the four-vector (unit
    axis, ``tan(omega / 2)``) in float32, ``(0, 0, 1, 0)`` for a zero
    rotation, and converted back to a quaternion in float64 as
    EMsoft's Rodrigues to quaternion conversion does.
    """
    raise NotImplementedError
