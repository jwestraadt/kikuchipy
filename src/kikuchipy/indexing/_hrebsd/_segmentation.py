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

import warnings

import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation

from kikuchipy.pattern._pattern import fft_frequency_vectors, get_image_quality

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

# One ``(drow, dcol)`` offset per UNDIRECTED edge of each
# neighbourhood, so that every pair is visited exactly once.  They are
# grid offsets and never flat-index ones: a ``+nx - 1`` "down left"
# step on the flat index wraps around a row end, which
# ``test_hrebsd_segmentation.py::TestSegmentGrains::test_eight
# _neighbours_do_not_wrap_around_a_row`` catches
_EDGE_OFFSETS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((0, 1), (1, 0)),
    2: ((0, 1), (1, 0), (1, 1), (1, -1)),
}

# Patterns are read from the map in blocks of about this many bytes
# when the image quality of a whole stack is wanted, so that a lazy
# stack is never held in memory whole (added 2026-09-08, Stage B
# adversarial review; requirements D16).  The value is a memory budget
# and changes no number: the kernel is evaluated one pattern at a time
# either way
IMAGE_QUALITY_BLOCK_BYTES: int = 64 * 1024 * 1024


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
    labels
        Grain labels of the map's navigation shape ``(ny, nx)`` and
        32-bit integer data type. Labels are 0-based and numbered in
        the row-major order in which the grains are first met, so the
        result is deterministic; points which are not indexed carry
        :data:`UNINDEXED`.

    Raises
    ------
    ValueError
        If *xmap* is not a :class:`~orix.crystal_map.CrystalMap`; if
        it has no map grid, which is what orix reports as the shape
        ``()``; if *misorientation_threshold* is not positive; or if
        *connectivity* is not one of
        :data:`SUPPORTED_CONNECTIVITIES`.

    Warns
    -----
    UserWarning
        If an indexed phase carries no point group, since its angles
        are then the raw ones and are NOT symmetry-reduced.

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
    their raw rotation angle is -- PROVIDED the phase carries a point
    group. A phase without one has no symmetry to reduce by, orix
    leaves it at ``C1``, and the angles are the raw ones; that is a
    different segmentation and it is warned about rather than left
    silent (recorded 2026-09-08, Stage B adversarial review).

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
    if not isinstance(xmap, CrystalMap):
        raise ValueError(f"xmap must be an orix CrystalMap, not {type(xmap).__name__}")
    threshold = float(misorientation_threshold)
    if not threshold > 0:
        raise ValueError(
            f"misorientation_threshold {misorientation_threshold} must be a "
            "positive misorientation angle in degrees"
        )
    connectivity = int(connectivity)
    if connectivity not in SUPPORTED_CONNECTIVITIES:
        raise ValueError(
            f"connectivity {connectivity} must be one of "
            f"{list(SUPPORTED_CONNECTIVITIES)}, where 1 links the four nearest "
            "neighbours of a point and 2 links all eight"
        )

    grid, _ = _map_grid(xmap)
    size = int(np.asarray(xmap.phase_id).size)
    phase_id = np.asarray(xmap.phase_id).ravel().astype(np.int64)
    indexed = np.asarray(xmap.is_indexed).ravel().astype(bool)
    rotations = np.asarray(xmap.rotations.data, dtype=np.float64)
    if rotations.ndim > 2:
        # Several rotations per point: the best one is the map's own
        rotations = rotations.reshape(size, -1, 4)[:, 0]

    first, second = _candidate_edges(grid, connectivity)
    keep = indexed[first] & indexed[second] & (phase_id[first] == phase_id[second])
    first, second = first[keep], second[keep]

    linked_first = []
    linked_second = []
    for identifier in np.unique(phase_id[first]):
        selection = phase_id[first] == identifier
        phase = xmap.phases[int(identifier)]
        point_group = phase.point_group
        if point_group is None:
            # orix leaves the symmetry at C1 for a phase with no point
            # group, so ``angle_with`` returns the RAW angle and the
            # symmetry reduction this function documents silently does
            # not happen.  MEASURED 2026-09-08 (Stage B adversarial
            # review): a 2 by 2 map of 0/90/0/90 degrees about z is ONE
            # grain with ``point_group="m-3m"`` and TWO without it, so
            # the difference reaches the reference pattern of every
            # point and every strain measured against it
            warnings.warn(
                f"phase {phase.name!r} carries no point group, so ITS "
                "misorientation angles are the raw ones and are NOT "
                "symmetry-reduced; two points related by a symmetry operation "
                "may therefore land in different grains. Give the phase a "
                "space group or a point group to reduce them",
                UserWarning,
            )
        orientations = Orientation(rotations, symmetry=point_group)
        left, right = first[selection], second[selection]
        # ``angle_with`` is symmetry-reduced and exists at the orix
        # 0.12.1 floor; the newer ``degrees`` keyword does not, so the
        # conversion is ``numpy.rad2deg``, which is the pair the
        # threshold pin of the test suite measures with
        angles = np.rad2deg(
            np.asarray(orientations[left].angle_with(orientations[right]))
        )
        # STRICTLY less than: a pair exactly AT the threshold is a
        # boundary (the ``>=`` mutant of plan section 3.4)
        below = angles < threshold
        linked_first.append(left[below])
        linked_second.append(right[below])

    roots = _connected_components(
        size,
        np.concatenate(linked_first) if linked_first else np.empty(0, dtype=np.int64),
        np.concatenate(linked_second) if linked_second else np.empty(0, dtype=np.int64),
    )
    return _label_row_major(grid, indexed, roots)


def _map_grid(xmap) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the map's point indices arranged on its own grid.

    Parameters
    ----------
    xmap
        :class:`~orix.crystal_map.CrystalMap` whose
        :attr:`~orix.crystal_map.CrystalMap.row` and
        :attr:`~orix.crystal_map.CrystalMap.col` grids give the two
        dimensional shape, which
        :attr:`~orix.crystal_map.CrystalMap.shape` does not: orix
        flattens both a one row and a one column map to ``(n,)``.

    Returns
    -------
    grid
        Array of shape ``(ny, nx)`` and 64-bit integer data type
        holding the flat point index at each grid position, and
        ``-1`` where the map has no point.
    navigation_shape
        The two dimensional shape ``(ny, nx)``.
    """
    rows, cols = map_grids(xmap)
    ny = int(rows.max()) + 1
    nx = int(cols.max()) + 1
    grid = np.full((ny, nx), -1, dtype=np.int64)
    grid[rows, cols] = np.arange(rows.size, dtype=np.int64)
    return grid, (ny, nx)


def map_grids(xmap) -> tuple[np.ndarray, np.ndarray]:
    """Return the row and column grids of a crystal map.

    The one place the three grid-shaped functions of Stage B --
    :func:`segment_grains`,
    :func:`~kikuchipy.indexing.hrebsd_kam` and
    :func:`~kikuchipy.indexing.hrebsd_pc_shift` -- read
    :attr:`~orix.crystal_map.CrystalMap.row` and
    :attr:`~orix.crystal_map.CrystalMap.col`, so that a map with no
    grid is named here rather than surfacing as an orix unpacking
    error (added 2026-09-08, Stage B adversarial review).

    Parameters
    ----------
    xmap
        :class:`~orix.crystal_map.CrystalMap` whose points lie on a
        scan grid of at least two positions.

    Returns
    -------
    rows, cols
        Flat arrays of shape ``(n,)`` and 64-bit integer data type.

    Raises
    ------
    ValueError
        If the map has no grid, which is what orix reports as the
        shape ``()``: a one-point map and a map whose points all sit
        at one scan position both land there, and reading the row
        grid of either raises ``not enough values to unpack``, a
        message naming nothing the caller passed.
    """
    if len(tuple(xmap.shape)) == 0:
        raise ValueError(
            "xmap has no map grid: orix reports its shape as (), which is what a "
            "one-point map and a map whose points all sit at one scan position "
            "both give. Pass a crystal map spanning at least two scan positions, "
            "in a row, in a column or on a two dimensional grid"
        )
    rows = np.asarray(xmap.row, dtype=np.int64).ravel()
    cols = np.asarray(xmap.col, dtype=np.int64).ravel()
    return rows, cols


def _candidate_edges(
    grid: np.ndarray, connectivity: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return every neighbouring pair of the grid, once each.

    Parameters
    ----------
    grid
        Point indices on the map grid, as :func:`_map_grid` returns
        them.
    connectivity
        1 for the four nearest neighbours, 2 for all eight.

    Returns
    -------
    first, second
        Flat point indices of shape ``(n_edges,)`` and 64-bit integer
        data type. Positions the map has no point at are dropped.

    Notes
    -----
    There is no empty-list branch here, and that is deliberate: every
    offset of :data:`_EDGE_OFFSETS` is skipped only on a one by one
    grid, which :func:`map_grids` refuses before this function is
    reached. The branch was pruned rather than tested at the Stage B
    adversarial review, on the precedent of the Stage A gate
    (validation entry 23) which pruned an unreachable non-finite
    guard instead of writing a test that could never run.
    """
    ny, nx = grid.shape
    first = []
    second = []
    for drow, dcol in _EDGE_OFFSETS[connectivity]:
        row0, row1 = max(0, -drow), ny - max(0, drow)
        col0, col1 = max(0, -dcol), nx - max(0, dcol)
        if row1 <= row0 or col1 <= col0:
            continue
        here = grid[row0:row1, col0:col1].ravel()
        there = grid[row0 + drow : row1 + drow, col0 + dcol : col1 + dcol].ravel()
        first.append(here)
        second.append(there)
    here = np.concatenate(first)
    there = np.concatenate(second)
    inside = (here >= 0) & (there >= 0)
    return here[inside], there[inside]


def _connected_components(
    size: int, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Return the component representative of every point.

    A union-find walk over the surviving edges, which is what makes
    the segmentation a connected-components one rather than a
    threshold on a single neighbour.

    Parameters
    ----------
    size
        Number of map points.
    first, second
        Flat point indices of the linked pairs.

    Returns
    -------
    roots
        Array of shape ``(size,)`` and 64-bit integer data type
        holding one representative point index per component.
    """
    parent = list(range(size))

    def find(point: int) -> int:
        root = point
        while parent[root] != root:
            root = parent[root]
        while parent[point] != root:
            parent[point], point = root, parent[point]
        return root

    for left, right in zip(first.tolist(), second.tolist()):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    return np.array([find(point) for point in range(size)], dtype=np.int64)


def _label_row_major(
    grid: np.ndarray, indexed: np.ndarray, roots: np.ndarray
) -> np.ndarray:
    """Return grain labels numbered in row-major first-seen order.

    Parameters
    ----------
    grid
        Point indices on the map grid, as :func:`_map_grid` returns
        them.
    indexed
        Boolean array of shape ``(n,)``, ``False`` where a point is
        not indexed.
    roots
        Component representatives, as :func:`_connected_components`
        returns them.

    Returns
    -------
    labels
        Array of the grid shape and 32-bit integer data type, 0-based
        and :data:`UNINDEXED` outside any grain.
    """
    order = grid.ravel()
    order = order[order >= 0]
    order = order[indexed[order]]
    labels = np.full(grid.shape, UNINDEXED, dtype=np.int32)
    sequence = roots[order]
    # ``numpy.unique`` sorts, so the first appearances are re-sorted
    # into the row-major walk order the numbering is defined by
    _, first_seen = np.unique(sequence, return_index=True)
    ordered = sequence[np.sort(first_seen)]
    lookup = {int(root): index for index, root in enumerate(ordered.tolist())}
    flat = np.full(roots.size, UNINDEXED, dtype=np.int32)
    flat[order] = np.array(
        [lookup[int(root)] for root in sequence.tolist()], dtype=np.int32
    )
    rows, cols = np.nonzero(grid >= 0)
    labels[rows, cols] = flat[grid[rows, cols]]
    return labels


def image_quality(patterns, indices: np.ndarray | None = None) -> np.ndarray:
    """Return the image quality of patterns of a map.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)``, raw as the engine
        receives them, eager or lazy. A lazy stack is read in blocks
        of about :data:`IMAGE_QUALITY_BLOCK_BYTES` and is never held
        in memory whole.
    indices
        Flat indices of the patterns to evaluate, or ``None`` (the
        default) for every pattern. Passing only the candidates keeps
        the patterns nothing can be selected from off the memory
        budget as well as out of the arithmetic.

    Returns
    -------
    quality
        Array of shape ``(n,)``, or of *indices* shape when those are
        given, and 64-bit float data type, the
        :func:`~kikuchipy.pattern.get_image_quality` of each pattern
        with the frequency vectors and the maximum inertia computed
        once for the shared pattern shape.

    Raises
    ------
    ValueError
        If *patterns* is not three dimensional, or if *indices* is
        not a one dimensional integer array inside the stack.

    Notes
    -----
    The kernel is evaluated one pattern at a time whatever the block
    size is, so the block size is a memory budget and changes no
    returned number (added 2026-09-08, Stage B adversarial review:
    the whole stack was materialised here, 576 MB on the shipped Si
    wafer and about 57 GB on a 500 by 500 map of 480 by 480 patterns,
    on the FROZEN DEFAULT ``reference="auto"`` path).
    """
    shape = tuple(int(i) for i in patterns.shape)
    if len(shape) != 3:
        raise ValueError(
            f"patterns of shape {shape} must be three dimensional, (n, nrows, ncols)"
        )
    size, nrows, ncols = shape
    if indices is None:
        indices = np.arange(size, dtype=np.int64)
    else:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"indices of shape {indices.shape} and data type {indices.dtype} "
                "must be a one dimensional integer array of flat pattern indices"
            )
        if indices.size and (indices.min() < 0 or indices.max() >= size):
            raise ValueError(
                f"every index must be a flat index within the {size} pattern(s) of "
                "the stack"
            )
        indices = indices.astype(np.int64)
    # Both depend on the pattern SHAPE alone, so they are computed once
    # here; the values are the ones
    # :func:`~kikuchipy.pattern.get_image_quality` computes for itself,
    # which keeps the result bitwise the public kernel's
    frequency_vectors = fft_frequency_vectors((nrows, ncols))
    inertia_max = np.sum(frequency_vectors) / (nrows * ncols)
    itemsize = int(np.dtype(patterns.dtype).itemsize)
    block = max(1, IMAGE_QUALITY_BLOCK_BYTES // max(1, nrows * ncols * itemsize))
    quality = np.empty(indices.size, dtype=np.float64)
    for start in range(0, indices.size, block):
        selection = indices[start : start + block]
        # ONE block is materialised at a time, which is the whole point
        # of the loop: ``patterns`` may be a Dask array of the entire
        # data set
        gathered = np.asarray(patterns[selection])
        for offset, pattern in enumerate(gathered):
            quality[start + offset] = get_image_quality(
                pattern, frequency_vectors=frequency_vectors, inertia_max=inertia_max
            )
    return quality


def select_references(
    patterns, labels: np.ndarray, selectable: np.ndarray | None = None
) -> np.ndarray:
    """Return the reference pattern of every grain.

    The per-grain selection of requirements D11.2: the point of
    highest :func:`image_quality`, ties broken by the LOWEST flat
    index.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)`` in map order, eager
        or lazy. Only the candidates are ever read.
    labels
        Flat grain labels of shape ``(n,)``, 0-based, with
        :data:`UNINDEXED` outside any grain.
    selectable
        Boolean array of shape ``(n,)``, ``False`` at a point which
        may NOT be chosen, or ``None`` (the default) for no
        restriction. This is where a navigation mask enters: a point
        the caller masked out is not a measurement and must not
        become the origin every strain in its grain is measured
        against (added 2026-09-08, Stage B adversarial review).

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
        If *patterns* is not three dimensional; if *labels* or
        *selectable* does not have one entry per pattern; or if no
        point carries a label, which leaves no grain to pick a
        reference pattern for.

    Notes
    -----
    A grain in which NOTHING is selectable falls back to its own
    points, unrestricted: every point of such a grain is masked out,
    so no pattern of it is correlated and the index is never used --
    but the pairing of
    :func:`~kikuchipy.indexing._hrebsd._reference.resolve_reference`
    still wants one index per label.
    """
    shape = tuple(int(i) for i in patterns.shape)
    if len(shape) != 3:
        raise ValueError(
            f"patterns of shape {shape} must be three dimensional, (n, nrows, ncols)"
        )
    labels = np.asarray(labels).ravel()
    if labels.size != shape[0]:
        raise ValueError(
            f"labels holds {labels.size} entry/entries but patterns holds "
            f"{shape[0]}; there must be one grain label per pattern"
        )
    if selectable is not None:
        selectable = np.asarray(selectable).ravel().astype(bool)
        if selectable.size != shape[0]:
            raise ValueError(
                f"selectable holds {selectable.size} entry/entries but patterns "
                f"holds {shape[0]}; there must be one flag per pattern"
            )
    unique = np.unique(labels[labels >= 0])
    if unique.size == 0:
        raise ValueError(
            "no map point carries a grain label, so there is no grain to pick a "
            "reference pattern for"
        )
    groups = []
    for label in unique:
        inside = np.flatnonzero(labels == label)
        if selectable is not None:
            allowed = inside[selectable[inside]]
            # A grain with nothing selectable is a grain nothing is
            # correlated in, so its reference is never read; it keeps
            # the unrestricted choice rather than leaving the pairing
            # one index short
            inside = allowed if allowed.size else inside
        groups.append(inside)
    # The image quality of the CANDIDATES alone, which is both the
    # arithmetic and, on a lazy stack, the memory (requirements D16):
    # a point outside every grain, or one the caller masked out, is
    # never read
    candidates = np.unique(np.concatenate(groups))
    quality = np.full(shape[0], -np.inf, dtype=np.float64)
    quality[candidates] = image_quality(patterns, candidates)
    reference_index = np.empty(unique.size, dtype=np.int32)
    for position, inside in enumerate(groups):
        # ``argmax`` returns the FIRST maximum, which IS the frozen tie
        # rule of requirements D11.2: ties go to the lowest flat index
        reference_index[position] = inside[np.argmax(quality[inside])]
    return reference_index
