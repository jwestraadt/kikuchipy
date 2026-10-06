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
# - The kernel average misorientation map getKAMMap, its neighbour
#   bookkeeping and its edge multipliers
#   (EMsoftOOLib/mod_DIsupport.f90)

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

"""Kernel average misorientation (KAM) maps of crystal maps, in a
correct mode and in an EMsoft compatible mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import CrystalMap

# Radians to degrees in float64, EMsoft's rtod
RTOD = 180 / np.pi


def kernel_average_misorientation_map(
    xmap: CrystalMap,
    *,
    degrees: bool = True,
    emsoft_compatible: bool = False,
) -> np.ndarray:
    """Return the kernel average misorientation (KAM) map of a crystal
    map.

    The KAM of a map point is the mean disorientation angle to its four
    nearest neighbours (left, right, up, down).

    Parameters
    ----------
    xmap
        Crystal map with a 2D grid of points. One-row, one-column and
        one-point maps are accepted. If there are several rotations per
        point, the first is used.
    degrees
        Whether to return the angles in degrees (default) or radians.
    emsoft_compatible
        Whether to reproduce EMsoft's KAM map (``getKAMMap``) instead
        of the correct mean. Default is False. EMsoft accumulates the
        pair angles along a one-dimensional loop over the map: it
        credits the vertical pair of two rows to the pixel one column
        to the right of the upper pixel, compares the last pixel of the
        first row with the identity, divides with fixed multipliers per
        edge and corner, and computes the pair angles with its own
        operator table without clipping. This mode requires every map
        point to be in the data, one phase, and a point group whose
        EMsoft operators equal orix' proper subgroup.

    Returns
    -------
    kam
        KAM map of shape (n rows, n columns) of float32. In the correct
        mode, a point gets the mean over the neighbours that exist, are
        in the data and share its phase, and NaN if there is none;
        points not in the data are NaN.

    Raises
    ------
    ValueError
        If ``emsoft_compatible=True`` and a map point is not in the
        data, the map has several phases, or the point group has no
        matching EMsoft operator set.

    See Also
    --------
    segment_grains_kam

    Notes
    -----
    The correct pair angle is ``2 arccos(d)``, with ``d`` the largest
    absolute dot product of one orientation with the other one
    multiplied from the left by the operators of the phase's proper
    point group, computed in float64. A ``d`` within four machine
    epsilons of one is set to one, so identical orientations give an
    angle of exactly 0.
    """
    # TODO: add a runnable Examples section on a small synthetic map
    # once the implementation exists
    raise NotImplementedError


def _dot_to_angle(d: np.ndarray) -> np.ndarray:
    """Return the rotation angle ``2 arccos(d)`` in radians of
    symmetry-reduced absolute dot products ``d``, with ``d >= 1 - 4
    eps`` set to 1 first (``eps`` of float64).

    Shared by the correct KAM and the grain reference orientation
    deviation.
    """
    raise NotImplementedError


def _correct_kam(
    q: np.ndarray,
    present: np.ndarray,
    phase_id: np.ndarray,
    operators_by_phase: dict[int, np.ndarray],
) -> np.ndarray:
    """Return the correct KAM in radians (float64).

    Parameters
    ----------
    q
        Unit quaternions of shape (ny, nx, 4) of float64.
    present
        Boolean array of shape (ny, nx), True where the point is in the
        data.
    phase_id
        Phase IDs of shape (ny, nx).
    operators_by_phase
        Proper point group operators of shape (m, 4) per phase ID.

    Returns
    -------
    kam
        Mean pair angle over existing, present, same-phase 4-neighbours
        of shape (ny, nx) of float64; NaN without such a neighbour or
        where not present.
    """
    raise NotImplementedError


def _emsoft_pair_angles(
    q: np.ndarray, operators: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return EMsoft's horizontal and vertical pair angles and the
    spurious angle of the last pixel of the first row to the identity.

    Parameters
    ----------
    q
        Quaternions of shape (H, W, 4) of float64, from EMsoft's Euler
        angle conversion.
    operators
        EMsoft operators of shape (m, 4).

    Returns
    -------
    pair_h
        Angles between (y, x) and (y, x + 1), shape (H, W - 1).
    pair_v
        Angles between (y, x) and (y + 1, x), shape (H - 1, W).
    spurious
        Angle between the identity quaternion and ``q[0, W - 1]``.
    """
    raise NotImplementedError


def _emsoft_neighbour_sum(
    pair_h: np.ndarray,
    pair_v: np.ndarray,
    spurious: float,
    H: int,
    W: int,
    dtype: type | np.dtype,
) -> np.ndarray:
    """Return EMsoft's per-pixel sums of credited pair values, flat in
    raster order, shape (H * W,).

    The order of accumulation per pixel is left, up, right, down, where
    "down" is the vertical pair credited from flat index ``p - 1``, and
    the spurious value goes to the first pixel and to the last pixel of
    the first row. For W == 1, pixel t gets twice the pair value of
    pixels t - 1 and t, with pixel -1 the identity (KAM) or the empty
    list (OSM).
    """
    raise NotImplementedError


def _emsoft_edge_multipliers(
    acc: np.ndarray, H: int, W: int, third: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Return EMsoft's edge-multiplied sums, flat, in EMsoft's order of
    multiplications.

    All values are multiplied by 0.25, then the top interior, the
    bottom interior, the left column interior and the right column
    interior by ``third``, then the first pixel by 4, the
    last pixel of the first row and the last pixel by 2, and the first
    pixel of the last row by ``third``.

    Parameters
    ----------
    acc
        Flat sums of shape (H * W,), see :func:`_emsoft_neighbour_sum`.
    H, W
        Number of rows and columns.
    third
        The function ``x -> (x * 4) / 3`` in the precision of the map:
        float64 for KAM, float32 for the orientation similarity map.
    """
    raise NotImplementedError
