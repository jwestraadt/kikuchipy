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

from typing import TYPE_CHECKING, Literal

import numpy as np

from kikuchipy.indexing._hrosm._grains import GrainTable

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation


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
        expectation maximisation. Since a quaternion and its negative
        are one orientation, the von Mises-Fisher mixture runs over
        the symmetry operators and their negatives (EMsoft compatible:
        over the operators only, as in EMsoft).
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
        right for "vmf" and "watson". Default is False. This mode
        requires every map point to be in the data and one phase.

    Returns
    -------
    grains
        Grain table of labels 1 to the largest label of ``grain_id``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``grain_id`` does not have the map's
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
    """
    # TODO: add a runnable Examples section on a small synthetic
    # two-grain map once the implementation exists
    raise NotImplementedError


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

    See Also
    --------
    average_grain_orientations, GrainTable
    """
    # TODO: add a runnable Examples section once the implementation
    # exists
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def _apply_kappa_gate(kappa: float, min_kappa: float) -> bool:
    """Return whether a "vmf" or "watson" grain is kept: ``kappa >
    min_kappa``, strictly (NaN fails, infinity passes).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
