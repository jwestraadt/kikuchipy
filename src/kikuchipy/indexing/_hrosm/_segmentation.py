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
# - The KAM difference region growing grow_region_driver_ and
#   grow_region_, as a closed form over connected components
#   (EMOpenCLLib/program_mods/mod_cluster.f90)
# - The grain dilation grain_dilate_ and the bounding boxes getROI_
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

"""Segmentation of a kernel average misorientation (KAM) map into
grains, grain dilation and grain bounding boxes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Offsets of the 8-neighbour graph, each undirected edge once
_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def segment_grains_kam(
    kam: np.ndarray,
    *,
    threshold: float = 5.0,
    dilate: bool = False,
    phase_id: np.ndarray | None = None,
    emsoft_compatible: bool = False,
) -> np.ndarray:
    """Return grain labels of a kernel average misorientation (KAM)
    map, segmented by EMsoft's KAM difference rule.

    Two 8-neighbours belong to the same grain if their KAM values
    differ by at most ``threshold``. A grain is a connected set of at
    least two such points that contains a point with a KAM of at most
    ``threshold``. Grains are labelled 1, 2, ... in increasing raster
    order (row outer, column inner) of their first point with a KAM of
    at most ``threshold``; all other points get 0.

    Parameters
    ----------
    kam
        KAM map of shape (n rows, n columns), in the unit of
        ``threshold``. Points with a non-finite KAM are never in a
        grain before dilation. The comparisons are made in the KAM's
        floating dtype, float64 for an integer KAM map.
    threshold
        Largest KAM difference between neighbours in a grain, and
        largest KAM of a grain's seed point. Default is 5.0 (degrees
        for a KAM map in degrees).
    dilate
        Whether to dilate the grains after segmentation. Default is
        False. In the correct mode, unassigned present points take the
        largest label among their 8-neighbours of the same phase;
        labels are never overwritten. Present points are those with a
        phase ID of at least 0 if ``phase_id`` is given, else those
        with a finite KAM. In the EMsoft compatible mode, every point
        except those in the first row and the first column takes the
        largest label in its 3 x 3 neighbourhood if that is not 0,
        which may overwrite labels and make a small grain vanish.
    phase_id
        Phase IDs of shape (n rows, n columns). If given, neighbours of
        different phases are never in the same grain, and points with
        a negative phase ID (absent points) are never in a grain.
    emsoft_compatible
        Whether to reproduce EMsoft's grain dilation. Default is False.
        The segmentation rule is EMsoft's in both modes.

    Returns
    -------
    grain_id
        Grain labels of the map's shape of int32: 0 outside grains,
        1, 2, ... for the grains.

    Raises
    ------
    ValueError
        If ``kam`` is not 2D, ``threshold`` is not finite and above 0,
        or the shape of ``phase_id`` differs from that of ``kam``.

    See Also
    --------
    kernel_average_misorientation_map, grain_bounding_boxes,
    kikuchipy.indexing.segment_grains :
        HREBSD's orientation-based grain segmentation (fork branch
        ``hrebsd-dic`` only).

    Notes
    -----
    The rule compares KAM values, not orientations. The KAM is about a
    quarter of the boundary angle on both sides of a straight boundary,
    so a boundary of less than about four times ``threshold`` may be
    crossed by a chain of neighbours.

    Examples
    --------
    Two grains of low KAM separated by a column of high KAM, which is
    in no grain, and which joins the larger label with dilation

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> kam = np.array(
    ...     [
    ...         [0.2, 0.3, 12.0, 0.5, 0.4],
    ...         [0.3, 0.2, 12.0, 0.4, 0.6],
    ...         [0.2, 0.4, 12.0, 0.3, 0.5],
    ...     ],
    ...     dtype=np.float32,
    ... )
    >>> kp.indexing.segment_grains_kam(kam, threshold=5.0)
    array([[1, 1, 0, 2, 2],
           [1, 1, 0, 2, 2],
           [1, 1, 0, 2, 2]], dtype=int32)
    >>> kp.indexing.segment_grains_kam(kam, threshold=5.0, dilate=True)
    array([[1, 1, 2, 2, 2],
           [1, 1, 2, 2, 2],
           [1, 1, 2, 2, 2]], dtype=int32)
    """
    kam = np.asarray(kam)
    if kam.ndim != 2:
        raise ValueError(f"kam must be a 2D array, not of shape {kam.shape}")
    if not np.issubdtype(kam.dtype, np.floating):
        kam = kam.astype(np.float64)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold {threshold} must be finite and > 0")
    if phase_id is not None:
        phase_id = np.asarray(phase_id)
        if phase_id.shape != kam.shape:
            raise ValueError(
                f"phase_id shape {phase_id.shape} must equal the kam shape {kam.shape}"
            )

    labels = _kam_difference_labels(kam, threshold, phase_id)
    if dilate:
        if emsoft_compatible:
            labels = _dilate_emsoft(labels)
        else:
            # A present point may have a non-finite KAM (no present
            # same-phase 4-neighbour), so the phase IDs decide
            if phase_id is not None:
                present = phase_id >= 0
            else:
                present = np.isfinite(kam)
            labels = _dilate_correct(labels, present, phase_id)
    return labels.astype(np.int32, copy=False)


def grain_bounding_boxes(grain_id: np.ndarray) -> np.ndarray:
    """Return the bounding box of every grain label.

    Parameters
    ----------
    grain_id
        Grain labels of shape (n rows, n columns): 0 outside grains,
        1, 2, ... for the grains.

    Returns
    -------
    boxes
        Boxes of labels 1 to the largest label, of shape (n, 4) of
        int64, with columns (first row, first column, height, width),
        0-based. A label without points gets the box (0, 0, 0, 0).

    Raises
    ------
    ValueError
        If ``grain_id`` is not 2D.

    See Also
    --------
    segment_grains_kam

    Examples
    --------
    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> grain_id = np.array(
    ...     [
    ...         [1, 1, 0, 0],
    ...         [1, 0, 0, 3],
    ...         [0, 0, 3, 3],
    ...     ]
    ... )
    >>> kp.indexing.grain_bounding_boxes(grain_id)
    array([[0, 0, 2, 2],
           [0, 0, 0, 0],
           [1, 2, 2, 2]])
    """
    grain_id = np.asarray(grain_id)
    if grain_id.ndim != 2:
        raise ValueError(f"grain_id must be a 2D array, not of shape {grain_id.shape}")
    labels = np.where(grain_id > 0, grain_id, 0).astype(np.int64)
    n = int(labels.max(initial=0))
    boxes = np.zeros((n, 4), dtype=np.int64)
    for i, box in enumerate(find_objects(labels, max_label=n)):
        if box is None:
            continue
        rows, cols = box
        boxes[i] = (
            rows.start,
            cols.start,
            rows.stop - rows.start,
            cols.stop - cols.start,
        )
    return boxes


def _kam_difference_labels(
    kam: np.ndarray, threshold: float, phase_id: np.ndarray | None
) -> np.ndarray:
    """Return labels of the KAM difference rule as int32, without
    dilation.

    The graph has edges over :data:`_OFFSETS` between finite, same-phase
    points with ``|kam_a - kam_b| <= threshold`` in the KAM's dtype;
    components of at least two points containing a point with ``kam
    <= threshold`` are labelled 1..n by raster order of each
    component's first such point.
    """
    H, W = kam.shape
    n = H * W
    thr = kam.dtype.type(threshold)
    valid = np.isfinite(kam)
    if phase_id is not None:
        valid &= phase_id >= 0
    index = np.arange(n).reshape(H, W)

    rows = []
    cols = []
    for dy, dx in _OFFSETS:
        # Points a = (y, x) and b = (y + dy, x + dx) inside the map
        a = (slice(0, H - dy), slice(max(0, -dx), W - max(0, dx)))
        b = (slice(dy, H), slice(max(0, dx), W - max(0, -dx)))
        edge = valid[a] & valid[b]
        if phase_id is not None:
            edge &= phase_id[a] == phase_id[b]
        # Compared in the KAM's dtype; NaN never compares True
        edge &= np.abs(kam[b] - kam[a]) <= thr
        rows.append(index[a][edge])
        cols.append(index[b][edge])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, component = connected_components(graph, directed=False)

    size = np.bincount(component)
    is_seed = (valid & (kam <= thr)).ravel()
    seeds = np.flatnonzero(is_seed)
    # The first seed of each component in raster order
    seed_component, first = np.unique(component[seeds], return_index=True)
    keep = size[seed_component] >= 2
    seed_component = seed_component[keep]
    first_seed = seeds[first[keep]]
    order = np.argsort(first_seed, kind="stable")

    component_label = np.zeros(size.size, dtype=np.int32)
    component_label[seed_component[order]] = np.arange(
        1, order.size + 1, dtype=np.int32
    )
    return component_label[component].reshape(H, W)


def _dilate_emsoft(labels: np.ndarray) -> np.ndarray:
    """Return labels dilated as EMsoft does: simultaneously, every
    point outside the first row and column takes the 3 x 3 maximum
    label (zero padded) if it is not 0, else keeps its label.
    """
    g = np.asarray(labels)
    m = maximum_filter(g, size=3, mode="constant", cval=0)
    out = g.copy()
    out[1:, 1:] = np.where(m[1:, 1:] != 0, m[1:, 1:], g[1:, 1:])
    return out


def _dilate_correct(
    labels: np.ndarray, present: np.ndarray, phase_id: np.ndarray | None
) -> np.ndarray:
    """Return labels where every unassigned present point takes the
    largest label among its same-phase 8-neighbours, simultaneously;
    assigned labels are never overwritten.
    """
    labels = np.asarray(labels)
    H, W = labels.shape
    if phase_id is None:
        phase_id = np.zeros((H, W), dtype=np.int64)
    # The largest same-phase 8-neighbour label per point, padded with 0
    largest = np.zeros_like(labels)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            # Point (y, x) and its neighbour (y + dy, x + dx)
            p = (slice(max(0, -dy), H - max(0, dy)), slice(max(0, -dx), W - max(0, dx)))
            q = (slice(max(0, dy), H - max(0, -dy)), slice(max(0, dx), W - max(0, -dx)))
            same = phase_id[p] == phase_id[q]
            candidate = np.where(same, labels[q], 0)
            largest[p] = np.maximum(largest[p], candidate)
    fill = (labels == 0) & present
    return np.where(fill, largest, labels)
