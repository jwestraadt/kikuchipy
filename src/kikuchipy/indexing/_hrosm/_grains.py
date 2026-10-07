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

"""The grain table of the HROSM tools and the 2D grid of a crystal
map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from orix.quaternion import Rotation

from kikuchipy.indexing._hrosm._segmentation import grain_bounding_boxes

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import CrystalMap


def _map_grid(xmap: CrystalMap) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the 2D grid of a crystal map over all its points, in the
    data or not.

    The shape ``(ny, nx)`` is orix' shape over all points,
    ``xmap._original_shape``: a 2D shape as is; a 1D shape of ``n``
    points ``(n, 1)`` if ``xmap.x`` is None, else ``(1, n)``; an empty
    shape ``(1, 1)``.

    Returns
    -------
    grid
        Index into the in-data arrays (``xmap.rotations``,
        ``xmap.phase_id``, ``xmap.prop``) per grid point, shape (ny,
        nx) of int64, -1 where the point is not in the data.
    shape
        ``(ny, nx)``.
    """
    # orix' shape over all points, set from the coordinates at
    # construction; xmap.shape counts only the points in the data
    orig_shape = tuple(int(i) for i in xmap._original_shape)
    if len(orig_shape) == 2:
        ny, nx = orig_shape
    elif len(orig_shape) == 1:
        n = orig_shape[0]
        # orix drops a constant coordinate: no x means one column
        ny, nx = (n, 1) if xmap.x is None else (1, n)
    else:
        ny, nx = 1, 1
    grid = np.full(ny * nx, -1, dtype=np.int64)
    grid[np.asarray(xmap.is_in_data, dtype=bool)] = np.arange(xmap.size, dtype=np.int64)
    return grid.reshape(ny, nx), (ny, nx)


def _rotation_from_data(data: np.ndarray) -> Rotation:
    """Return rotations holding the quaternions ``data`` of shape (n,
    4) bit for bit, without orix' renormalisation.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1, 4)
    rotation = Rotation(data)
    rotation.data = data
    return rotation


@dataclass(frozen=True, eq=False)
class GrainTable:
    """Per-grain results of grain averaging, in label order (index
    ``i`` is grain label ``i + 1``).

    Parameters
    ----------
    n_pixels
        Number of map points per grain after any dilation, shape (n,)
        of int64.
    bounding_box
        Bounding box per grain, shape (n, 4) of int64, with columns
        (first row, first column, height, width), 0-based.
    rotation
        Reference rotation per grain, shape (n,): the grain average, or
        the centre point's rotation for the "center" method; the
        identity where not ``valid``.
    phase_id
        Phase ID of the grain's points, shape (n,) of int32; -1 for a
        label without points.
    kappa
        Concentration per grain, shape (n,) of float64: the von
        Mises-Fisher concentration of the mean resultant length for
        the "mean" method, the expectation maximisation estimate for
        "vmf" and "watson", 1.0 for "center", and -1.0 where rejected.
    max_grod
        Largest grain reference orientation deviation (GROD) per grain
        in degrees, shape (n,) of float32; NaN where not ``valid``.
    valid
        Whether the grain average was accepted, shape (n,) of bool
        (``kappa != -1``).
    method
        Averaging method, or None for a table rebuilt from a crystal
        map.

    See Also
    --------
    average_grain_orientations,
    grain_reference_orientation_deviation_map
    """

    n_pixels: np.ndarray
    bounding_box: np.ndarray
    rotation: Rotation
    phase_id: np.ndarray
    kappa: np.ndarray
    max_grod: np.ndarray
    valid: np.ndarray
    method: str | None

    @property
    def n_grains(self) -> int:
        """Return the number of grain labels in the table."""
        return int(np.asarray(self.n_pixels).shape[0])

    @classmethod
    def from_crystal_map(cls, xmap: CrystalMap) -> GrainTable:
        """Return the grain table stored in a crystal map's properties.

        Parameters
        ----------
        xmap
            Crystal map with the properties ``"grain_id"``,
            ``"grain_orientation"``, ``"grain_kappa"`` and
            ``"grain_max_grod"``: each point's grain label, and its
            grain's orientation (quaternion), concentration and
            largest GROD in degrees.

        Returns
        -------
        grains
            Grain table with ``method`` None. The number of grains is
            the largest label; the number of points and the bounding
            boxes are recomputed from the labels; the other fields are
            read at the first point of each label. A label without
            points gets 0 points, the box (0, 0, 0, 0), the identity
            rotation, phase ID -1, ``kappa`` -1.0, ``max_grod`` NaN and
            ``valid`` False.

        Examples
        --------
        A table rebuilt from the grain properties of a 2 x 3 map of two
        grains

        >>> import numpy as np
        >>> from orix.crystal_map import (
        ...     CrystalMap, Phase, PhaseList, create_coordinate_arrays
        ... )
        >>> from orix.quaternion import Rotation
        >>> from kikuchipy.indexing import GrainTable
        >>> coords, n = create_coordinate_arrays((2, 3), (1, 1))
        >>> grain_id = np.array([1, 1, 2, 1, 2, 2], dtype=np.int32)
        >>> orientation = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        >>> xmap = CrystalMap(
        ...     Rotation.identity((n,)),
        ...     x=coords["x"],
        ...     y=coords["y"],
        ...     phase_list=PhaseList(Phase(point_group="m-3m")),
        ...     prop={
        ...         "grain_id": grain_id,
        ...         "grain_orientation": orientation,
        ...         "grain_kappa": np.ones(n),
        ...         "grain_max_grod": np.zeros(n, dtype=np.float32),
        ...     },
        ... )
        >>> grains = GrainTable.from_crystal_map(xmap)
        >>> grains.n_grains
        2
        >>> grains.n_pixels
        array([3, 3])
        >>> grains.bounding_box
        array([[0, 0, 2, 2],
               [0, 1, 2, 2]])
        """
        grid, shape = _map_grid(xmap)
        in_data = grid >= 0
        labels_in_data = np.asarray(xmap.prop["grain_id"]).astype(np.int64).ravel()
        labels = np.zeros(shape, dtype=np.int64)
        labels[in_data] = labels_in_data[grid[in_data]]
        labels = np.where(labels > 0, labels, 0)
        n = int(labels.max(initial=0))

        orientation = np.asarray(xmap.prop["grain_orientation"], dtype=np.float64)
        orientation = orientation.reshape(-1, 4)
        kappa_in_data = np.asarray(xmap.prop["grain_kappa"], dtype=np.float64)
        max_grod_in_data = np.asarray(xmap.prop["grain_max_grod"], dtype=np.float32)
        phase_id_in_data = np.asarray(xmap.phase_id).ravel()

        n_pixels = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
        rotation = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        phase_id = np.full(n, -1, dtype=np.int32)
        kappa = np.full(n, -1.0, dtype=np.float64)
        max_grod = np.full(n, np.nan, dtype=np.float32)

        # The first point of each label in raster order carries the
        # grain's values
        flat = labels.ravel()
        present = np.flatnonzero(flat > 0)
        first_label, first_index = np.unique(flat[present], return_index=True)
        first_point = grid.ravel()[present[first_index]]
        g = first_label - 1
        rotation[g] = orientation[first_point]
        phase_id[g] = phase_id_in_data[first_point]
        kappa[g] = kappa_in_data[first_point]
        max_grod[g] = max_grod_in_data[first_point]

        return cls(
            n_pixels=n_pixels,
            bounding_box=grain_bounding_boxes(labels),
            rotation=_rotation_from_data(rotation),
            phase_id=phase_id,
            kappa=kappa,
            max_grod=max_grod,
            valid=kappa != -1,
            method=None,
        )


def _broadcast_grain_props(
    table: GrainTable, grain_id: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the per-point properties ``"grain_orientation"`` ((n, 4)
    float64), ``"grain_kappa"`` ((n,) float64) and ``"grain_max_grod"``
    ((n,) float32) of a grain table, NaN outside grains, flat over the
    labels ``grain_id`` of the in-data points.
    """
    grain_id = np.asarray(grain_id).astype(np.int64).ravel()
    n = grain_id.size
    orientation = np.full((n, 4), np.nan, dtype=np.float64)
    kappa = np.full(n, np.nan, dtype=np.float64)
    max_grod = np.full(n, np.nan, dtype=np.float32)
    inside = (grain_id > 0) & (grain_id <= table.n_grains)
    g = grain_id[inside] - 1
    orientation[inside] = np.asarray(table.rotation.data, dtype=np.float64).reshape(
        -1, 4
    )[g]
    kappa[inside] = np.asarray(table.kappa, dtype=np.float64)[g]
    max_grod[inside] = np.asarray(table.max_grod, dtype=np.float32)[g]
    return {
        "grain_orientation": orientation,
        "grain_kappa": kappa,
        "grain_max_grod": max_grod,
    }
