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
# - The grain orientation averaging of the cluster stage, its centre
#   pixel choice and its concentration gate
#   (EMOpenCLLib/program_mods/mod_cluster.f90)

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

"""Grain averaging of orientations, the grain reference orientation
deviation (GROD) and the misorientation ball coverage check.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal
import warnings

import numpy as np
from orix.quaternion import Rotation
from orix.quaternion.symmetry import C1

from kikuchipy.indexing._hrosm._directional_statistics import (
    _em_correct,
    _em_emsoft,
    _kappa_from_y,
)
from kikuchipy.indexing._hrosm._emsoft_quaternions import (
    emsoft_euler_to_quaternion,
    emsoft_point_group_number,
    emsoft_quaternion_multiply,
    emsoft_symmetry_operators,
)
from kikuchipy.indexing._hrosm._grains import (
    GrainTable,
    _map_grid,
    _rotation_from_data,
)
from kikuchipy.indexing._hrosm._kam import RTOD, _dot_to_angle
from kikuchipy.indexing._hrosm._segmentation import grain_bounding_boxes

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import CrystalMap

_METHODS = ("mean", "center", "vmf", "watson")
# Largest number of grains listed in the coverage warning
_N_LISTED = 10


def average_grain_orientations(
    xmap: CrystalMap,
    grain_id: np.ndarray,
    *,
    method: Literal["mean", "center", "vmf", "watson"] = "mean",
    max_angle: float = 5.0,
    n_em: int = 25,
    n_iter: int = 40,
    min_kappa: float = 5.0,
    seed: int | np.random.Generator | None = None,
    emsoft_compatible: bool = False,
) -> GrainTable:
    """Return the reference orientation of every grain of a crystal
    map, with the grains' largest orientation deviations from it.

    Parameters
    ----------
    xmap
        Crystal map with a 2D grid of points.
    grain_id
        Grain labels of the map's grid shape (n rows, n columns): 0
        outside grains, 1, 2, ... for the grains, as returned by
        :func:`segment_grains_kam`.
    method
        Averaging method. "mean" (default) aligns every point's
        symmetry variant with the grain's centre point and takes the
        normalised arithmetic mean. "center" takes the grain point
        nearest the grain's centroid (EMsoft compatible: the point at
        the centre of the bounding box, which may lie outside the
        grain). "vmf" and "watson" estimate the mean of a von
        Mises-Fisher or Watson mixture over the symmetry variants by
        expectation maximisation :cite:`chen2015parameter`. Since a
        quaternion and its negative are one orientation, the von
        Mises-Fisher mixture runs over the symmetry operators and
        their negatives (EMsoft compatible: over the operators only,
        as in EMsoft).
    max_angle
        Radius in degrees of the misorientation ball the averages will
        be centred on. A warning is emitted if a valid grain has a
        largest grain reference orientation deviation above it.
        Default is 5.0.
    n_em
        Number of initial guesses of "vmf" and "watson". Default is
        25.
    n_iter
        Largest number of iterations per initial guess of "vmf" and
        "watson". Default is 40.
    min_kappa
        A "vmf" or "watson" grain is kept only if its concentration is
        above this value; otherwise it gets the identity rotation,
        ``kappa = -1.0`` and ``valid = False``. Default is 5.0.
    seed
        Seed or random number generator of the initial guesses of
        "vmf" and "watson". One generator is used across the grains in
        label order. If not given, the result is not reproducible.
    emsoft_compatible
        Whether to reproduce EMsoft's averaging: the bounding box
        centre point for "center", and EMsoft's expectation
        maximisation with the symmetry operators applied from the
        right for "vmf" and "watson". "mean" is the same in both
        modes. Default is False. This mode requires every map point to
        be in the data and one phase.

    Returns
    -------
    grains
        Grain table of labels 1 to the largest label of ``grain_id``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``max_angle`` is not in (0, 180),
        ``n_em`` or ``n_iter`` is not an integer of at least 1,
        ``min_kappa`` is negative, ``grain_id`` does not have the map's
        grid shape, or the EMsoft compatible input requirements are
        not met.

    Warns
    -----
    UserWarning
        If a valid grain has a largest grain reference orientation
        deviation above ``max_angle`` (the message names the
        misorientation ball and the max GROD per grain).

    See Also
    --------
    GrainTable, grain_reference_orientation_deviation_map,
    misorientation_ball

    Examples
    --------
    A 2 x 4 map of two grains, each with a small orientation gradient
    about [001] of 0.5 degrees per column

    >>> import numpy as np
    >>> from orix.crystal_map import (
    ...     CrystalMap, Phase, PhaseList, create_coordinate_arrays
    ... )
    >>> from orix.quaternion import Rotation
    >>> import kikuchipy as kp
    >>> coords, n = create_coordinate_arrays((2, 4), step_sizes=(1, 1))
    >>> grain_id = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    >>> x = coords["x"]
    >>> rot = Rotation.from_axes_angles(
    ...     [0, 0, 1], np.deg2rad(0.5 * x + 20 * (x > 1))
    ... )
    >>> xmap = CrystalMap(
    ...     rot,
    ...     x=coords["x"],
    ...     y=coords["y"],
    ...     phase_list=PhaseList(Phase(point_group="m-3m")),
    ... )
    >>> grains = kp.indexing.average_grain_orientations(xmap, grain_id)
    >>> grains.n_grains
    2
    >>> grains.n_pixels
    array([4, 4])
    >>> print(np.rad2deg(grains.rotation.angle).round(2))
    [ 0.25 21.25]
    >>> print(grains.max_grod.round(2))
    [0.25 0.25]
    """
    table = _average_grains(
        xmap,
        grain_id,
        method=method,
        max_angle=max_angle,
        n_em=n_em,
        n_iter=n_iter,
        min_kappa=min_kappa,
        seed=seed,
        emsoft_compatible=emsoft_compatible,
    )
    labels = np.arange(1, table.n_grains + 1)
    valid = np.asarray(table.valid, dtype=bool)
    message = _coverage_warning_message(
        labels[valid], table.max_grod[valid], table.n_pixels[valid], max_angle
    )
    if message is not None:
        warnings.warn(message, UserWarning, stacklevel=2)
    return table


def grain_reference_orientation_deviation_map(
    xmap: CrystalMap, grain_id: np.ndarray, grains: GrainTable
) -> np.ndarray:
    """Return the grain reference orientation deviation (GROD) map.

    The GROD of a point is its disorientation angle to its grain's
    reference orientation (the rotation in ``grains``), reduced over
    the proper point group operators of the phase.

    Parameters
    ----------
    xmap
        Crystal map with a 2D grid of points.
    grain_id
        Grain labels of the map's grid shape.
    grains
        Grain table of the labels, as returned by
        :func:`average_grain_orientations`.

    Returns
    -------
    grod
        GROD in degrees of the map's grid shape (n rows, n columns) of
        float32; NaN outside grains, in invalid grains and at points
        not in the data.

    Raises
    ------
    ValueError
        If ``grain_id`` does not have the map's grid shape.

    See Also
    --------
    average_grain_orientations, GrainTable

    Examples
    --------
    A 1 x 4 map of one grain with an orientation gradient about [001]
    of 1 degree per column, referred to its centre point

    >>> import numpy as np
    >>> from orix.crystal_map import (
    ...     CrystalMap, Phase, PhaseList, create_coordinate_arrays
    ... )
    >>> from orix.quaternion import Rotation
    >>> import kikuchipy as kp
    >>> coords, n = create_coordinate_arrays((1, 4), step_sizes=(1, 1))
    >>> rot = Rotation.from_axes_angles(
    ...     [0, 0, 1], np.deg2rad(coords["x"])
    ... )
    >>> xmap = CrystalMap(
    ...     rot,
    ...     x=coords["x"],
    ...     y=coords["y"],
    ...     phase_list=PhaseList(Phase(point_group="m-3m")),
    ... )
    >>> grain_id = np.ones((1, 4), dtype=int)
    >>> grains = kp.indexing.average_grain_orientations(
    ...     xmap, grain_id, method="center"
    ... )
    >>> grod = kp.indexing.grain_reference_orientation_deviation_map(
    ...     xmap, grain_id, grains
    ... )
    >>> print(grod.round(3))
    [[1. 0. 1. 2.]]
    """
    _, shape = _map_grid(xmap)
    grain_id = _check_grain_id(grain_id, shape)
    rotation = grains.rotation
    valid = np.asarray(grains.valid, dtype=bool)
    return _grod_map(xmap, grain_id, rotation, valid, _orix_operators_by_phase(xmap))


def _check_grain_id(grain_id: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return the grain labels as an int64 array, or raise a ValueError
    if their shape differs from the map's grid shape.
    """
    grain_id = np.asarray(grain_id)
    if grain_id.shape != tuple(shape):
        raise ValueError(
            f"grain_id shape {grain_id.shape} must equal the map's grid shape "
            f"{tuple(shape)}"
        )
    return grain_id.astype(np.int64)


def _orix_operators_by_phase(xmap: CrystalMap) -> dict[int, np.ndarray]:
    """Return the proper point group operators of every phase of a
    crystal map, the identity for a phase without a point group.
    """
    operators = {}
    for i, phase in xmap.phases:
        if i < 0:
            continue
        point_group = phase.point_group
        point_group = C1 if point_group is None else point_group
        data = np.asarray(point_group.proper_subgroup.data, dtype=np.float64)
        operators[int(i)] = data.reshape(-1, 4)
    return operators


def _first_rotations(xmap: CrystalMap) -> np.ndarray:
    """Return the first rotation per in-data point as unit quaternions
    of shape (n, 4) of float64.
    """
    data = np.asarray(xmap.rotations.data, dtype=np.float64)
    if data.ndim > 2:
        data = data[:, 0]
    return data.reshape(-1, 4)


def _check_positive_int(value: int, name: str):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} {value!r} must be an integer of at least 1")


def _average_grains(
    xmap: CrystalMap,
    grain_id: np.ndarray,
    *,
    method: Literal["mean", "center", "vmf", "watson"] = "mean",
    max_angle: float = 5.0,
    n_em: int = 25,
    n_iter: int = 40,
    min_kappa: float = 5.0,
    seed: int | np.random.Generator | None = None,
    emsoft_compatible: bool = False,
) -> GrainTable:
    """Return the grain table of :func:`average_grain_orientations`
    without ever warning (the private core the driver calls).
    """
    if method not in _METHODS:
        raise ValueError(f"method {method!r} must be one of {_METHODS}")
    if isinstance(max_angle, bool) or not 0 < max_angle < 180:
        raise ValueError(f"max_angle {max_angle!r} must be a number in (0, 180)")
    _check_positive_int(n_em, "n_em")
    _check_positive_int(n_iter, "n_iter")
    if isinstance(min_kappa, bool) or not min_kappa >= 0:
        raise ValueError(f"min_kappa {min_kappa!r} must be at least 0")

    grid, (ny, nx) = _map_grid(xmap)
    grain_id = _check_grain_id(grain_id, (ny, nx))
    flat_grid = grid.ravel()
    in_data = flat_grid >= 0
    phase_id = np.full(ny * nx, -1, dtype=np.int64)
    phase_id[in_data] = np.asarray(xmap.phase_id).ravel()[flat_grid[in_data]]
    # Points not indexed (phase ID -1) count as not in the data
    present = in_data & (phase_id >= 0)

    data = _first_rotations(xmap)
    q = np.zeros((ny * nx, 4), dtype=np.float64)
    q[:, 0] = 1
    q[present] = data[flat_grid[present]]
    q = np.where(q[:, :1] < 0, -q, q)
    operators_by_phase = _orix_operators_by_phase(xmap)

    if emsoft_compatible:
        if not np.all(present):
            raise ValueError(
                "emsoft_compatible requires every map point to be in the data "
                "and indexed"
            )
        ids = np.unique(phase_id)
        if ids.size > 1:
            raise ValueError(
                "emsoft_compatible requires one phase, but the map has "
                f"{ids.size} phases"
            )
        point_group = xmap.phases[int(ids[0])].point_group
        emsoft_operators = emsoft_symmetry_operators(
            emsoft_point_group_number(point_group)
        )
        # EMsoft reads float32 Euler angles and promotes them to
        # float64 before its own conversion to quaternions
        euler = Rotation(data[flat_grid]).to_euler().astype(np.float32)
        q_avg = emsoft_euler_to_quaternion(euler.astype(np.float64)).reshape(-1, 4)
    else:
        q_avg = q

    labels_all = np.where(grain_id > 0, grain_id, 0)
    labels = np.where(present.reshape(ny, nx), labels_all, 0)
    n = int(labels_all.max(initial=0))
    boxes = grain_bounding_boxes(labels)
    if boxes.shape[0] < n:
        boxes = np.concatenate([boxes, np.zeros((n - boxes.shape[0], 4), np.int64)])

    flat_labels = labels.ravel()
    n_pixels = np.bincount(flat_labels, minlength=n + 1)[1 : n + 1].astype(np.int64)
    order = np.argsort(flat_labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(flat_labels, minlength=n + 1))])

    rotation = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    grain_phase = np.full(n, -1, dtype=np.int32)
    kappa = np.full(n, -1.0, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    rng = None
    if method in ("vmf", "watson"):
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

    for g in range(1, n + 1):
        # The grain's points in raster order
        points = order[starts[g] : starts[g + 1]]
        if points.size == 0:
            continue
        phase = int(phase_id[points[0]])
        grain_phase[g - 1] = phase
        # "mean" has no EMsoft counterpart and is identical in both
        # modes: orix quaternions and operators
        if emsoft_compatible and method != "mean":
            operators = emsoft_operators
            x = q_avg[points]
        else:
            operators = operators_by_phase[phase]
            x = q[points]

        if method == "center":
            row, col = _center_pixel(labels, g, boxes[g - 1], emsoft_compatible)
            mu = q_avg[row * nx + col]
            kappa_g = 1.0
            keep = True
        elif method == "mean":
            row, col = _center_pixel(labels, g, boxes[g - 1], False)
            mu, kappa_g = _mean_orientation(x, operators, q[row * nx + col])
            keep = True
        else:
            em = _em_emsoft if emsoft_compatible else _em_correct
            result = em(x, operators, method, int(n_em), int(n_iter), rng)
            mu = result.mu
            kappa_g = result.kappa
            keep = _apply_kappa_gate(kappa_g, float(min_kappa))

        if keep:
            mu = np.asarray(mu, dtype=np.float64)
            rotation[g - 1] = -mu if mu[0] < 0 else mu
            kappa[g - 1] = kappa_g
            valid[g - 1] = True

    rotation = _rotation_from_data(rotation)
    grod = _grod_map(xmap, grain_id, rotation, valid, operators_by_phase)
    max_grod = _max_per_grain(grod, labels_all, n)

    return GrainTable(
        n_pixels=n_pixels,
        bounding_box=boxes,
        rotation=rotation,
        phase_id=grain_phase,
        kappa=kappa,
        max_grod=max_grod,
        valid=valid,
        method=method,
    )


def _max_per_grain(grod: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Return the largest finite GROD per grain label 1 to ``n``,
    float32, NaN for a grain without a finite value.
    """
    grod = grod.ravel()
    labels = labels.ravel()
    finite = np.isfinite(grod) & (labels > 0) & (labels <= n)
    out = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(out, labels[finite] - 1, grod[finite])
    out[np.isneginf(out)] = np.nan
    return out


def _center_pixel(
    grain_id: np.ndarray,
    label: int,
    bounding_box: np.ndarray,
    emsoft_compatible: bool,
) -> tuple[int, int]:
    """Return the (row, column) of a grain's centre point.

    Correct: the grain point nearest the grain's centroid, lowest flat
    index on ties. EMsoft compatible: the bounding box centre
    ``(row0 + height // 2, col0 + width // 2)``, possibly outside the
    grain.
    """
    if emsoft_compatible:
        row0, col0, height, width = (int(v) for v in bounding_box)
        return row0 + height // 2, col0 + width // 2
    points = np.argwhere(np.asarray(grain_id) == label)
    centroid = points.mean(axis=0)
    d2 = np.sum((points - centroid) ** 2, axis=1)
    row, col = points[int(np.argmin(d2))]
    return int(row), int(col)


def _mean_orientation(
    q: np.ndarray, operators: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the symmetry-aligned arithmetic mean quaternion of a
    grain and its von Mises-Fisher concentration.

    Every quaternion of shape (n, 4) is replaced by its variant ``S_j
    q`` with the largest absolute dot product with ``reference``, sign
    fixed to a non-negative dot product; the mean is normalised; the
    concentration comes from the mean resultant length.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    operators = np.asarray(operators, dtype=np.float64).reshape(-1, 4)
    reference = np.asarray(reference, dtype=np.float64).reshape(4)
    n = q.shape[0]
    variants = emsoft_quaternion_multiply(operators[None, :, :], q[:, None, :])
    dots = variants @ reference
    j = np.argmax(np.abs(dots), axis=1)
    rows = np.arange(n)
    aligned = variants[rows, j]
    aligned = np.where(dots[rows, j][:, None] < 0, -aligned, aligned)
    total = aligned.sum(axis=0)
    norm = float(np.sqrt(np.sum(total**2)))
    mu = total / norm
    # A mean resultant length above 1 is rounding only
    y = min(norm / n, 1.0)
    return mu, _kappa_from_y(y, "vmf")


def _apply_kappa_gate(kappa: float, min_kappa: float) -> bool:
    """Return whether a "vmf" or "watson" grain is kept: ``kappa >
    min_kappa``, strictly (NaN fails, infinity passes).
    """
    return bool(kappa > min_kappa)


def _grod_map(
    xmap: CrystalMap,
    grain_id: np.ndarray,
    rotation: Rotation,
    valid: np.ndarray,
    operators_by_phase: dict[int, np.ndarray],
) -> np.ndarray:
    """Return the GROD map in degrees, shape (ny, nx) of float32.

    The angle of every grain point to ``rotation[label - 1]`` is
    reduced over the phase's operators from the left and computed by
    the KAM pair angle with its near-one snap, so that a grain's own
    reference point gets exactly 0.0; NaN at label 0, invalid grains
    and points not in the data. This is the only GROD implementation.
    """
    grid, (ny, nx) = _map_grid(xmap)
    flat_grid = grid.ravel()
    in_data = flat_grid >= 0
    phase_id = np.full(ny * nx, -1, dtype=np.int64)
    phase_id[in_data] = np.asarray(xmap.phase_id).ravel()[flat_grid[in_data]]

    labels = np.asarray(grain_id).astype(np.int64).ravel()
    valid = np.asarray(valid, dtype=bool).ravel()
    reference = np.asarray(rotation.data, dtype=np.float64).reshape(-1, 4)
    n = valid.size
    in_grain = (labels > 0) & (labels <= n)
    in_grain[in_grain] = valid[labels[in_grain] - 1]
    use = in_data & (phase_id >= 0) & in_grain

    data = _first_rotations(xmap)
    grod = np.full(ny * nx, np.nan, dtype=np.float32)
    for phase in np.unique(phase_id[use]):
        sel = np.flatnonzero(use & (phase_id == phase))
        a = data[flat_grid[sel]]
        b = reference[labels[sel] - 1]
        d = np.zeros(sel.size, dtype=np.float64)
        for s in operators_by_phase[int(phase)]:
            sa = emsoft_quaternion_multiply(s, a)
            d = np.maximum(d, np.abs(np.sum(sa * b, axis=-1)))
        grod[sel] = (_dot_to_angle(d) * RTOD).astype(np.float32)
    return grod.reshape(ny, nx)


def _coverage_warning_message(
    labels: np.ndarray,
    max_grod: np.ndarray,
    n_pixels: np.ndarray,
    max_angle: float,
    spacing: float | None = None,
) -> str | None:
    """Return the misorientation ball coverage warning, or None.

    None unless some ``max_grod > max_angle``, compared in float64.
    Otherwise the message names the "misorientation ball" and the "max
    GROD", the number of grains concerned, ``max_angle``, the mean
    spacing if given, the largest max GROD, up to ten (label, max
    GROD, number of points) entries in descending max GROD (ties by
    ascending label), and the hint ``max_angle >= <the largest max
    GROD rounded up to 0.5 degrees>``.
    """
    labels = np.asarray(labels).astype(np.int64).ravel()
    max_grod = np.asarray(max_grod).astype(np.float64).ravel()
    n_pixels = np.asarray(n_pixels).astype(np.int64).ravel()
    above = np.flatnonzero(max_grod > float(max_angle))
    if above.size == 0:
        return None

    # Descending max GROD, ties by ascending label
    order = above[np.lexsort((labels[above], -max_grod[above]))]
    largest = float(max_grod[order[0]])
    hint = math.ceil(largest * 2) / 2
    entries = ", ".join(
        f"grain {labels[i]} (max GROD {max_grod[i]:.2f} deg, {n_pixels[i]} points)"
        for i in order[:_N_LISTED]
    )
    radius = f"radius max_angle = {float(max_angle):g} deg"
    if spacing is not None:
        radius += f" (mean spacing {float(spacing):.4f} deg)"
    shown = "Largest" if order.size > _N_LISTED else "All"
    return (
        f"{above.size} grain(s) have a max GROD (grain reference orientation "
        f"deviation) above the misorientation ball {radius}; the largest max "
        f"GROD is {largest:.2f} deg. Points beyond the radius can only match "
        "orientations on the ball surface. "
        f"{shown} of these grains, by descending max GROD: {entries}. "
        f"Setting max_angle >= {hint:.1f} covers every grain; the cost grows "
        "with the cube of the number of steps across the ball, and keeping "
        "the spacing fixed means increasing n_steps with max_angle."
    )
