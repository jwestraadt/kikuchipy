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
from orix.quaternion import Rotation
from orix.quaternion.symmetry import C1

from kikuchipy.indexing._hrosm._emsoft_quaternions import (
    emsoft_disorientation_angle,
    emsoft_euler_to_quaternion,
    emsoft_point_group_number,
    emsoft_quaternion_multiply,
    emsoft_symmetry_operators,
)
from kikuchipy.indexing._hrosm._grains import _map_grid

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

    Examples
    --------
    A map of 3 x 4 points rotated about [001] by 0.5 degrees more per
    column, so that every horizontal neighbour pair is disoriented by
    0.5 degrees and every vertical pair by 0 degrees

    >>> import numpy as np
    >>> from orix.crystal_map import (
    ...     CrystalMap, Phase, PhaseList, create_coordinate_arrays
    ... )
    >>> from orix.quaternion import Rotation
    >>> import kikuchipy as kp
    >>> coords, n = create_coordinate_arrays((3, 4), step_sizes=(1, 1))
    >>> rot = Rotation.from_axes_angles(
    ...     [0, 0, 1], np.deg2rad(0.5 * coords["x"])
    ... )
    >>> xmap = CrystalMap(
    ...     rot,
    ...     x=coords["x"],
    ...     y=coords["y"],
    ...     phase_list=PhaseList(Phase(point_group="m-3m")),
    ... )
    >>> kam = kp.indexing.kernel_average_misorientation_map(xmap)
    >>> kam.shape
    (3, 4)
    >>> print(kam.round(2))
    [[0.25 0.33 0.33 0.25]
     [0.17 0.25 0.25 0.17]
     [0.25 0.33 0.33 0.25]]
    """
    grid, (ny, nx) = _map_grid(xmap)
    in_data = grid >= 0

    # The first rotation per point, of the points in the data
    data = np.asarray(xmap.rotations.data, dtype=np.float64)
    if data.ndim > 2:
        data = data[:, 0]
    data = data.reshape(-1, 4)

    phase_id = np.full((ny, nx), -1, dtype=np.int64)
    phase_id[in_data] = np.asarray(xmap.phase_id).ravel()[grid[in_data]]
    # Points not indexed (phase ID -1) count as not in the data
    present = in_data & (phase_id >= 0)

    if emsoft_compatible:
        if not np.all(present):
            raise ValueError(
                "emsoft_compatible requires every map point to be in the "
                "data and indexed"
            )
        ids = np.unique(phase_id)
        if ids.size > 1:
            raise ValueError(
                "emsoft_compatible requires one phase, but the map has "
                f"{ids.size} phases"
            )
        # A phase without a point group has no EMsoft number either
        point_group = xmap.phases[int(ids[0])].point_group
        operators = emsoft_symmetry_operators(emsoft_point_group_number(point_group))
        # EMsoft reads float32 Euler angles and promotes them to
        # float64 before its own conversion to quaternions
        euler = Rotation(data[grid.ravel()]).to_euler().astype(np.float32)
        q = emsoft_euler_to_quaternion(euler.astype(np.float64))
        q = q.reshape(ny, nx, 4)
        pair_h, pair_v, spurious = _emsoft_pair_angles(q, operators)
        acc = _emsoft_neighbour_sum(pair_h, pair_v, spurious, ny, nx, np.float64)
        acc = _emsoft_edge_multipliers(acc, ny, nx, _third_float64)
        kam = acc.astype(np.float32).reshape(ny, nx)
    else:
        q = np.zeros((ny, nx, 4), dtype=np.float64)
        q[..., 0] = 1
        q[in_data] = data[grid[in_data]]
        operators_by_phase = {}
        for i in np.unique(phase_id[present]):
            point_group = xmap.phases[int(i)].point_group
            # A phase without a point group has only the identity
            point_group = C1 if point_group is None else point_group
            operators = point_group.proper_subgroup.data.reshape(-1, 4)
            operators_by_phase[int(i)] = np.asarray(operators, dtype=np.float64)
        kam = _correct_kam(q, present, phase_id, operators_by_phase)
        kam = kam.astype(np.float32)

    if degrees:
        kam = (kam.astype(np.float64) * RTOD).astype(np.float32)
    return kam


def _dot_to_angle(d: np.ndarray) -> np.ndarray:
    """Return the rotation angle ``2 arccos(d)`` in radians of
    symmetry-reduced absolute dot products ``d``, with ``d >= 1 - 4
    eps`` set to 1 first (``eps`` of float64).

    Shared by the correct KAM and the grain reference orientation
    deviation.
    """
    d = np.asarray(d, dtype=np.float64)
    d = np.where(d >= 1 - 4 * np.finfo(np.float64).eps, 1.0, d)
    return 2 * np.arccos(d)


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
    ny, nx = present.shape
    total = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int64)
    every = slice(None)
    # Vertical pairs, then horizontal pairs
    for a, b in [
        ((slice(None, -1), every), (slice(1, None), every)),
        ((every, slice(None, -1)), (every, slice(1, None))),
    ]:
        valid = present[a] & present[b] & (phase_id[a] == phase_id[b])
        qa = q[a][valid]
        qb = q[b][valid]
        pair_phase = phase_id[a][valid]
        angle_valid = np.zeros(qa.shape[0], dtype=np.float64)
        for i, operators in operators_by_phase.items():
            is_phase = pair_phase == i
            angle_valid[is_phase] = _pair_angle(qa[is_phase], qb[is_phase], operators)
        angle = np.zeros(valid.shape, dtype=np.float64)
        angle[valid] = angle_valid
        total[a] += angle
        total[b] += angle
        count[a] += valid
        count[b] += valid
    kam = np.full((ny, nx), np.nan, dtype=np.float64)
    has_neighbour = present & (count > 0)
    kam[has_neighbour] = total[has_neighbour] / count[has_neighbour]
    return kam


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
    H, W = q.shape[:2]
    identity = emsoft_euler_to_quaternion(np.zeros(3)).reshape(1, 4)
    # All pairs in one call: horizontal, vertical, then the last pixel
    # of the first row against the identity
    a = np.concatenate([q[:, :-1].reshape(-1, 4), q[:-1].reshape(-1, 4), identity])
    b = np.concatenate(
        [q[:, 1:].reshape(-1, 4), q[1:].reshape(-1, 4), q[0, W - 1].reshape(1, 4)]
    )
    angle = emsoft_disorientation_angle(a, b, operators)
    n_h = H * (W - 1)
    n_v = (H - 1) * W
    pair_h = angle[:n_h].reshape(H, W - 1)
    pair_v = angle[n_h : n_h + n_v].reshape(H - 1, W)
    spurious = float(angle[-1])
    return pair_h, pair_v, spurious


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
    n = H * W
    if W == 1:
        # Every pixel is the first of its row: it gets its pair with
        # the pixel above twice, the first pixel its pair with the
        # identity (the spurious value) twice
        pair = np.zeros(n, dtype)
        pair[0] = spurious
        pair[1:] = np.asarray(pair_v).ravel()
        return pair + pair
    left = np.zeros((H, W), dtype)
    left[:, 1:] = pair_h
    up = np.zeros((H, W), dtype)
    up[1:, :] = pair_v
    up[0, W - 1] = spurious
    right = np.zeros((H, W), dtype)
    right[:, :-1] = pair_h
    # The vertical pair of a pixel and the one below it goes to the
    # next pixel in raster order
    v = np.zeros(n, dtype)
    v[: (H - 1) * W] = np.asarray(pair_v).ravel()
    down = np.zeros(n, dtype)
    down[1:] = v[:-1]
    down[0] = spurious
    # EMsoft's order of accumulation per pixel: left, up, right, down
    return ((left + up) + right).ravel() + down


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
    acc = np.array(acc, copy=True)
    n = H * W
    acc *= 0.25
    acc[1 : W - 1] = third(acc[1 : W - 1])
    acc[n - W + 1 : n - 1] = third(acc[n - W + 1 : n - 1])
    left = W * np.arange(1, H - 1)
    acc[left] = third(acc[left])
    right = W * np.arange(2, H) - 1
    acc[right] = third(acc[right])
    acc[0] *= 4.0
    acc[W - 1] *= 2.0
    acc[n - 1] *= 2.0
    acc[n - W] = third(acc[n - W])
    return acc


def _pair_angle(a: np.ndarray, b: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Return the symmetry-reduced angles in radians between pairs of
    unit quaternions ``a`` and ``b`` of shape (n, 4): the largest
    absolute dot product of ``b`` with ``a`` multiplied from the left
    by each operator of shape (m, 4), through :func:`_dot_to_angle`.
    """
    d = np.zeros(a.shape[0], dtype=np.float64)
    for s in operators:
        sa = emsoft_quaternion_multiply(s, a)
        d = np.maximum(d, np.abs(np.sum(sa * b, axis=-1)))
    return _dot_to_angle(d)


def _third_float64(x: np.ndarray) -> np.ndarray:
    """Return ``(x * 4) / 3`` in float64, EMsoft's edge multiplier."""
    return (x * 4.0) / 3.0
