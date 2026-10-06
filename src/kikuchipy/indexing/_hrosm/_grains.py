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
    raise NotImplementedError


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
        Concentration per grain, shape (n,) of float64: -1.0 where
        rejected, 1.0 for the "center" method.
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
        raise NotImplementedError

    @classmethod
    def from_crystal_map(cls, xmap: CrystalMap) -> GrainTable:
        """Return the grain table stored in a crystal map's properties.

        Parameters
        ----------
        xmap
            Crystal map with the properties ``"grain_id"``,
            ``"grain_orientation"``, ``"grain_kappa"`` and
            ``"grain_max_grod"``, as returned by ``EBSD.hrosm()``.

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
        """
        # TODO: add a runnable Examples section once the implementation
        # exists
        raise NotImplementedError


def _broadcast_grain_props(
    table: GrainTable, grain_id: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the per-point properties ``"grain_orientation"`` ((n, 4)
    float64), ``"grain_kappa"`` ((n,) float64) and ``"grain_max_grod"``
    ((n,) float32) of a grain table, NaN outside grains, flat over the
    labels ``grain_id`` of the in-data points.
    """
    raise NotImplementedError
