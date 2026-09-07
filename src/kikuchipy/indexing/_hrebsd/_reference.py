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

"""Reference pattern resolution for the HREBSD-DIC engine.

Everything HREBSD-DIC reports is relative: each point's homography is
measured against ITS grain's reference pattern, and both the grain
identifier and the flat index of that reference are stored per point
(requirements D11.4).  Cross-grain absolute comparison is out of
scope for version one and is deferred together with simulated
references.

Stage A implements the EXPLICIT modes only: one global reference
given as ``(row, col)``, or an array of flat indices paired with a
user supplied grain map.  ``reference="auto"`` raises
:class:`NotImplementedError` naming Stage B until the grain
segmentation and the image-quality based per-grain selection of
requirements D11.1 and D11.2 land.
"""

import numpy as np

# The ``reference`` string modes accepted by the engine, frozen
AUTO_REFERENCE: str = "auto"

# The grain label of a point which is not indexed (D11.1)
UNLABELLED: int = -1


def resolve_reference(
    reference: str | tuple[int, int] | np.ndarray,
    grain_labels: np.ndarray | None,
    navigation_shape: tuple[int, int],
    *,
    misorientation_threshold: float = 5.0,
    xmap=None,
    patterns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-point grain identifier and reference index.

    Parameters
    ----------
    reference
        ``"auto"``, a ``(row, col)`` tuple, or an integer array of
        flat indices with one entry per grain label.
    grain_labels
        User supplied grain map of shape *navigation_shape* with
        0-based labels and :data:`UNLABELLED` for points outside any
        grain, or ``None``.
    navigation_shape
        Map shape ``(ny, nx)``.
    misorientation_threshold
        Grain boundary misorientation angle in degrees, used only by
        the ``"auto"`` mode of Stage B. Default is 5.0.
    xmap
        :class:`~orix.crystal_map.CrystalMap` of the map, used only
        by the ``"auto"`` mode of Stage B.
    patterns
        Patterns of shape ``(n, nrows, ncols)``, used only by the
        ``"auto"`` mode of Stage B, whose per-grain selection
        maximizes the image quality of the raw patterns.

    Returns
    -------
    grain_id
        Array of shape ``(ny * nx,)`` and 32-bit integer data type,
        in map order, with :data:`UNLABELLED` where no reference
        applies.
    reference_index
        Array of shape ``(ny * nx,)`` and 32-bit integer data type
        holding the FLAT map index of each point's reference, with
        :data:`UNLABELLED` where no reference applies.

    Raises
    ------
    NotImplementedError
        If *reference* is ``"auto"``. The message names
        ``reference="auto"`` and states that automatic grain
        segmentation and per-grain reference selection arrive in
        Stage B of ``specs/2026-09-07-hrebsd-dic/``, so the caller
        knows to pass an explicit reference meanwhile.
    ValueError
        If *reference* is an unknown string; if a ``(row, col)``
        tuple is outside the map; if an index array is given without
        *grain_labels* or does not hold one index per grain label; or
        if *grain_labels* does not match *navigation_shape*.
    """
    raise NotImplementedError
