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

"""Grain segmentation and per-grain reference selection.

No grain segmentation exists in orix or in kikuchipy, so this is
written here (requirements D11).  It is deliberately small: connected
components on the map grid, where an edge links two neighbours whose
symmetry-reduced misorientation angle is BELOW the threshold.

The comparison side is part of the contract and not an accident: an
edge exists when::

    numpy.rad2deg(angle_between_neighbours) < misorientation_threshold

so a pair sitting exactly AT the threshold is a boundary.  The
docstring says which side because the ``>=`` mutant of plan section
3.4 is invisible to any test that only sweeps the threshold from far
below to far above, and
``test_hrebsd_segmentation.py::TestThreshold`` pins the strict form
by feeding back an angle it measures with the same
:meth:`orix.quaternion.Orientation.angle_with` call.

Misorientation angles come from
:meth:`~orix.quaternion.Orientation.angle_with` on neighbour pairs,
which is symmetry-reduced and available at the orix 0.12.1 floor;
the newer ``degrees`` keyword is not used and the conversion is
:func:`numpy.rad2deg` (the constitution's floor rule).  A pair whose
two points carry different PHASES has no such angle, and none is
invented: a phase boundary is a grain boundary and the edge simply
never exists, which lets a multi-phase map through the frozen
``reference="auto"`` default of requirements D11.3 instead of
refusing it (recorded 2026-09-07, Stage B failing-tests review).

The per-grain reference is the point of highest pattern image
quality, computed internally with kikuchipy's own
:func:`~kikuchipy.pattern.get_image_quality` on the RAW patterns, so
that the choice is deterministic and does not depend on which
properties the input map happens to carry.  Ties go to the lowest
flat index (requirements D11.2, frozen).

References
----------
The 5 degree grain boundary threshold is the standard HREBSD and KAM
practice of OpenXY and MTEX; the image quality kernel is Krieger
Lassen's, as kikuchipy already implements it.
"""

import numpy as np

# The label of a point which is not indexed, and so belongs to no
# grain.  The same value as
# :data:`~kikuchipy.indexing._hrebsd._reference.UNLABELLED`, which is
# what the reference resolution reports for such a point; the two are
# pinned equal by ``test_hrebsd_segmentation.py`` rather than shared
# by an import, which would be circular
UNINDEXED: int = -1

# The neighbourhoods of requirements D11.1: 4-neighbour edges by
# default, 8-neighbour edges on request
SUPPORTED_CONNECTIVITIES: tuple[int, ...] = (1, 2)


def segment_grains(
    xmap,
    *,
    misorientation_threshold: float = 5.0,
    connectivity: int = 1,
) -> np.ndarray:
    """Segment a crystal map into grains by neighbour misorientation.

    Connected components on the map grid: two neighbouring points
    belong to the same grain when the symmetry-reduced misorientation
    angle between their orientations is BELOW
    *misorientation_threshold*.

    Parameters
    ----------
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map, one or two dimensional. Points which are not
        indexed get the label ``-1`` and never join a grain. A map of
        several indexed phases is segmented PER PHASE: an edge whose
        two points carry different phase identifiers never links, and
        the angles of each phase use that phase's own point group.
    misorientation_threshold : float, optional
        Grain boundary misorientation angle in DEGREES. Default is
        5.0, the standard HREBSD and KAM value. A pair exactly at the
        threshold is a boundary: the comparison is strictly less
        than.
    connectivity : int, optional
        1 (default) links the four nearest neighbours of a point, 2
        links all eight.

    Returns
    -------
    labels : numpy.ndarray
        Grain labels of the map's navigation shape ``(ny, nx)`` and
        32-bit integer data type. Labels are 0-based and numbered in
        the row-major order in which the grains are first met, so the
        result is deterministic; points which are not indexed carry
        :data:`UNINDEXED`.

    Raises
    ------
    ValueError
        If *xmap* is not a :class:`~orix.crystal_map.CrystalMap`; if
        *misorientation_threshold* is not positive; or if
        *connectivity* is not one of
        :data:`SUPPORTED_CONNECTIVITIES`.

    Notes
    -----
    The returned shape is ALWAYS two dimensional and comes from the
    map's :attr:`~orix.crystal_map.CrystalMap.row` and
    :attr:`~orix.crystal_map.CrystalMap.col` grids, never from
    :attr:`~orix.crystal_map.CrystalMap.shape`, which orix flattens to
    ``(n,)`` for a map one point wide or one point tall. So a map
    whose points all lie in one row returns ``(1, nx)`` and one whose
    points all lie in one column returns ``(ny, 1)``: which of the two
    a flattened map is is decided by its own grids and not by a
    convention here.
    :func:`~kikuchipy.indexing._hrebsd._reference.resolve_reference`
    compares this shape with the navigation shape EXACTLY, and
    :func:`~kikuchipy.indexing.hrebsd_kam` returns the same shape.

    The angles are symmetry-reduced, so two points related by a
    symmetry operation of the phase sit in the same grain whatever
    their raw rotation angle is.

    Several phases are segmented per phase (recorded 2026-09-07 at the
    Stage B failing-tests review). An earlier draft refused a
    multi-phase map outright, which no frozen requirement asks for and
    which would have narrowed the FROZEN DEFAULT ``reference="auto"``
    of :meth:`~kikuchipy.signals.EBSD.hrebsd_dic` to single-phase maps
    only, against requirements D9.6 ("the engine itself is
    phase-agnostic per grain"): a map the explicit reference modes
    index happily would have raised on the default path. A phase
    boundary is a grain boundary, which is the natural extension of
    the connected-components rule, so no edge crosses one. The
    single-phase restriction of requirements D9.6 stays where it
    belongs, on the STRESS path of
    :func:`~kikuchipy.indexing.hrebsd_strain_stress`.

    This is the segmentation ``reference="auto"`` of
    :meth:`~kikuchipy.signals.EBSD.hrebsd_dic` runs when no grain map
    is supplied, and it accepts the same
    *misorientation_threshold*. It is a plain grid segmentation: no
    grain size filter, no boundary smoothing and no distance-to-
    boundary refinement, which are recorded possible additions and
    not part of version one.

    See Also
    --------
    kikuchipy.signals.EBSD.hrebsd_dic
    """
    raise NotImplementedError(
        "segment_grains arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


def image_quality(patterns: np.ndarray) -> np.ndarray:
    """Return the image quality of every pattern of a map.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)``, raw as the engine
        receives them. A lazy array is computed here, since the
        selection needs every value.

    Returns
    -------
    quality
        Array of shape ``(n,)`` and 64-bit float data type, the
        :func:`~kikuchipy.pattern.get_image_quality` of each pattern
        with the frequency vectors and the maximum inertia computed
        once for the shared pattern shape.

    Raises
    ------
    ValueError
        If *patterns* is not three dimensional.
    """
    raise NotImplementedError(
        "image_quality arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


def select_references(patterns: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the reference pattern of every grain.

    The per-grain selection of requirements D11.2: the point of
    highest :func:`image_quality`, ties broken by the LOWEST flat
    index.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)`` in map order.
    labels
        Flat grain labels of shape ``(n,)``, 0-based, with
        :data:`UNINDEXED` outside any grain.

    Returns
    -------
    reference_index
        Array of shape ``(n_grains,)`` and 32-bit integer data type
        holding one FLAT map index per grain, paired POSITIONALLY
        with the sorted unique labels, which is the pairing
        :func:`~kikuchipy.indexing._hrebsd._reference.resolve_reference`
        expects: with labels ``(0, 3, 7)`` the second entry is the
        reference of label 3.

    Raises
    ------
    ValueError
        If *patterns* is not three dimensional; if *labels* does not
        have one entry per pattern; or if no point carries a label,
        which leaves no grain to pick a reference for.
    """
    raise NotImplementedError(
        "select_references arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )
