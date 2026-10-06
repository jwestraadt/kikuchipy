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
        grain. The comparisons are made in the KAM's dtype.
    threshold
        Largest KAM difference between neighbours in a grain, and
        largest KAM of a grain's seed point. Default is 5.0 (degrees
        for a KAM map in degrees).
    dilate
        Whether to dilate the grains after segmentation. Default is
        False. In the correct mode, unassigned points take the largest
        label among their 8-neighbours of the same phase; labels are
        never overwritten. In the EMsoft compatible mode, every point
        except those in the first row and the first column takes the
        largest label in its 3 x 3 neighbourhood if that is not 0,
        which may overwrite labels and make a small grain vanish.
    phase_id
        Phase IDs of shape (n rows, n columns). If given, neighbours of
        different phases are never in the same grain, and points with
        a negative phase ID are never in a grain.
    emsoft_compatible
        Whether to reproduce EMsoft's grain dilation. Default is False.
        The segmentation rule is EMsoft's in both modes.

    Returns
    -------
    grain_id
        Grain labels of the map's shape of int32: 0 outside grains,
        1, 2, ... for the grains.

    See Also
    --------
    kernel_average_misorientation_map, grain_bounding_boxes

    Notes
    -----
    The rule compares KAM values, not orientations. The KAM is about a
    quarter of the boundary angle on both sides of a straight boundary,
    so a boundary of less than about four times ``threshold`` may be
    crossed by a chain of neighbours.
    """
    # TODO: add a runnable Examples section on a small synthetic KAM
    # map once the implementation exists
    raise NotImplementedError


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

    See Also
    --------
    segment_grains_kam
    """
    # TODO: add a runnable Examples section once the implementation
    # exists
    raise NotImplementedError


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
    raise NotImplementedError


def _dilate_emsoft(labels: np.ndarray) -> np.ndarray:
    """Return labels dilated as EMsoft does: simultaneously, every
    point outside the first row and column takes the 3 x 3 maximum
    label (zero padded) if it is not 0, else keeps its label.
    """
    raise NotImplementedError


def _dilate_correct(
    labels: np.ndarray, present: np.ndarray, phase_id: np.ndarray | None
) -> np.ndarray:
    """Return labels where every unassigned present point takes the
    largest label among its same-phase 8-neighbours, simultaneously;
    assigned labels are never overwritten.
    """
    raise NotImplementedError
