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

"""Tests of ``kikuchipy.indexing._hrebsd._segmentation`` and of the
``reference="auto"`` wiring in ``._reference``.

Covers requirements D11 of ``specs/2026-09-07-hrebsd-dic/``: grain
segmentation by neighbour misorientation (two grains, a one-point
grain, unindexed points, 4 against 8 neighbours, a threshold sweep
and the comparison SIDE), the per-grain image-quality reference
selection with its tie rule, and the ``"auto"`` mode which is those
two functions followed by the same positional pairing and same-grain
guard the explicit index array goes through.

Every map here is synthetic and every expectation is built in the
test: the labels by hand, the misorientation angles from rotations
about ONE axis, where the pair angle is the difference of the two
angles and needs no orix algebra, and the reference indices from
``kikuchipy.pattern.get_image_quality`` on the shipped nickel
patterns, which is the kernel requirements D11.2 freezes the
selection on.

The threshold pin deserves its own note.  A sweep from far below to
far above the boundary cannot tell ``<`` from ``<=``, which is the
plan section 3.4 mutant, so ``TestThreshold`` measures the pair angle
with the same
:meth:`~orix.quaternion.Orientation.angle_with` call and the same
:func:`numpy.rad2deg` conversion the module's docstring commits it
to, feeds that angle back as the threshold, and asserts the pair is
SPLIT.  The neighbouring float above it must join.

Two shape cases and one phase case were added or corrected at the
Stage B failing-tests review (2026-09-07).  Shape: on the installed
orix BOTH a ``(1, n)`` and an ``(n, 1)`` navigation shape flatten to
``xmap.shape == (n,)``, so which of the two a flattened map is has to
come from the row and column grids; ``_reference._flatten_labels``
compares the returned shape with the navigation shape EXACTLY, so a
"one dimensional means one row" rule would break a column map.
Phase: the drafted suite pinned a ValueError on a multi-phase map,
which no frozen requirement asks for and which would have narrowed
the FROZEN DEFAULT ``reference="auto"`` against requirements D9.6;
a phase boundary is a grain boundary instead.

Written failing before the implementation, at the Stage B
failing-tests gate: every test which calls the module failed with
``NotImplementedError`` until the segmentation landed, and passed
unchanged after it, while the signature freeze and the constant pins
passed from the start (narration corrected to the past tense
2026-09-08, Stage B adversarial review).
"""

import inspect

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Orientation, Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._reference import UNLABELLED, resolve_reference
from kikuchipy.indexing._hrebsd._segmentation import (
    SUPPORTED_CONNECTIVITIES,
    UNINDEXED,
    image_quality,
    segment_grains,
    select_references,
)

# ------------------------- Frozen constants ------------------------- #

# The frozen default of requirements D11.1, in degrees
DEFAULT_THRESHOLD = 5.0

# An angle far above the default threshold, so that the two grains of
# a synthetic map are unambiguously two grains, and one far below it
BIG_ANGLE = 20.0
SMALL_ANGLE = 0.5

# The band of a machine-precision-class identity, FROZEN
ALGEBRA_TOL = 1e-12


# ------------- The plan 3.4 mutation list, mapped ------------------- #
#
#  segmentation threshold compared with >= .. TestThreshold::test_a
#                                             _pair_at_the_threshold_is
#                                             _a_boundary
#  per-grain reference off by one in the      TestSelectReferences::
#  flat index ............................... test_the_pairing_is
#                                             _positional_in_the_sorted
#                                             _labels, and the same-grain
#                                             guard of TestAutoReference


# ----------------------------- Helpers ------------------------------ #


def crystal_map(angles_deg, navigation_shape, unindexed=(), phase_ids=None):
    """Return a crystal map whose orientations are rotations about
    ONE axis by the given angles, in map order.

    One axis on purpose: the misorientation of two such rotations is
    the DIFFERENCE of their angles, so every expectation below is
    arithmetic rather than a second call to the same orix machinery
    the module under test uses.
    """
    arrays, size = create_coordinate_arrays(navigation_shape, (1.0, 1.0))
    angles = np.asarray(angles_deg, dtype=np.float64).ravel()
    assert angles.size == size
    arrays["rotations"] = Rotation.from_axes_angles(
        np.tile([0.0, 0.0, 1.0], (size, 1)), np.deg2rad(angles)
    )
    if phase_ids is None:
        phase_ids = np.zeros(size, dtype=int)
        phase_list = PhaseList(Phase(name="ni", space_group=225))
    else:
        phase_ids = np.asarray(phase_ids, dtype=int)
        phase_list = PhaseList(
            [Phase(name="ni", space_group=225), Phase(name="ti", space_group=194)]
        )
    phase_ids = np.asarray(phase_ids, dtype=int).copy()
    for index in unindexed:
        phase_ids[index] = -1
    arrays["phase_id"] = phase_ids
    arrays["phase_list"] = phase_list
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    return xmap


def ni_patterns(n=9):
    """Return the first *n* shipped nickel patterns, a cheap 60 by 60
    stack with genuinely different image qualities."""
    signal = kp.data.nickel_ebsd_small()
    return np.asarray(signal.data, dtype=np.float64).reshape(-1, 60, 60)[:n]


def expected_quality(patterns):
    """Return the image quality of every pattern, computed with
    kikuchipy's own public kernel, which is what requirements D11.2
    freezes the selection on."""
    return np.array([kp.pattern.get_image_quality(pattern) for pattern in patterns])


def expected_reference(patterns, labels):
    """Return the argmax-quality flat index of every grain, ties to
    the lowest flat index, assembled here."""
    quality = expected_quality(patterns)
    labels = np.asarray(labels).ravel()
    references = []
    for label in np.unique(labels[labels >= 0]):
        inside = np.flatnonzero(labels == label)
        # ``argmax`` returns the FIRST maximum, which is the frozen
        # tie rule of requirements D11.2 written out
        references.append(int(inside[np.argmax(quality[inside])]))
    return np.array(references, dtype=np.int32)


# ============== D11.1 -- the segmentation itself ==================== #


class TestSegmentGrains:
    """Connected components on the map grid.  [D11.1]"""

    def test_two_grains(self):
        angles = [[0.0, 0.2, BIG_ANGLE, BIG_ANGLE + 0.1]] * 2
        xmap = crystal_map(angles, (2, 4))
        labels = segment_grains(xmap)
        assert labels.shape == (2, 4)
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels, [[0, 0, 1, 1], [0, 0, 1, 1]])

    def test_a_one_point_grain(self):
        # a single point of its own, surrounded by one grain: the
        # smallest grain a map can hold, and the one an off-by-one in
        # the component walk loses
        angles = [[0.0, 0.1, 0.2], [0.1, BIG_ANGLE, 0.2]]
        xmap = crystal_map(angles, (2, 3))
        labels = segment_grains(xmap)
        np.testing.assert_array_equal(labels, [[0, 0, 0], [0, 1, 0]])

    def test_unindexed_points_get_minus_one_and_do_not_bridge(self):
        # every orientation here is identical, so the ONLY reason the
        # map holds two grains is that the point between them is not
        # indexed and cannot carry an edge
        xmap = crystal_map([0.0, 0.0, 0.0], (1, 3), unindexed=(1,))
        labels = segment_grains(xmap)
        assert labels.shape == (1, 3)
        np.testing.assert_array_equal(labels, [[0, UNINDEXED, 1]])

    def test_the_unindexed_label_is_the_reference_module_one(self):
        # the two constants are defined apart to keep the import
        # one-directional, so their equality is pinned rather than
        # assumed
        assert UNINDEXED == UNLABELLED == -1

    def test_four_against_eight_neighbours(self):
        # the two orientations sit on the two diagonals, so the map
        # holds four one-point grains with 4-neighbour edges and two
        # diagonal grains with 8-neighbour ones.  It also catches a
        # 4-connectivity row wraparound: the flat pair (1, 2) both
        # carry BIG_ANGLE, so a flat-index neighbour walk would merge
        # them into [[0, 1], [1, 2]]
        angles = [[0.0, BIG_ANGLE], [BIG_ANGLE, 0.0]]
        xmap = crystal_map(angles, (2, 2))
        np.testing.assert_array_equal(
            segment_grains(xmap, connectivity=1), [[0, 1], [2, 3]]
        )
        np.testing.assert_array_equal(
            segment_grains(xmap, connectivity=2), [[0, 1], [1, 0]]
        )

    def test_eight_neighbours_do_not_wrap_around_a_row(self):
        # ADDED 2026-09-07 at the Stage B failing-tests review.  The
        # test above is the only other connectivity=2 case and it sits
        # on a 2 by 2 map, where every point is a neighbour of every
        # other and its expectation [[0, 1], [1, 0]] is reached with or
        # without a wraparound diagonal: a ``numpy.roll`` style
        # 8-neighbour walk survives it.  Here the map is three columns
        # wide, and the only same-orientation pair, (0, 0) and (0, 2),
        # is NOT a true 8-neighbour pair -- it is two columns apart in
        # one row.  A flat-index walk with the ``+nx - 1`` "down left"
        # offset links flat 0 to flat 2 and would merge them into
        # [[0, 1, 0], [1, 1, 1]]
        angles = [[0.0, BIG_ANGLE, 0.0], [BIG_ANGLE, BIG_ANGLE, BIG_ANGLE]]
        xmap = crystal_map(angles, (2, 3))
        np.testing.assert_array_equal(
            segment_grains(xmap, connectivity=2), [[0, 1, 2], [1, 1, 1]]
        )
        # the true 8-neighbourhood of each end really does exclude the
        # other end, and both ends really are one point: with the
        # wraparound they would be one grain of two
        labels = segment_grains(xmap, connectivity=2)
        assert labels[0, 0] != labels[0, 2]
        assert int(np.sum(labels == labels[0, 0])) == 1

    def test_connectivity_one_is_the_default(self):
        angles = [[0.0, BIG_ANGLE], [BIG_ANGLE, 0.0]]
        xmap = crystal_map(angles, (2, 2))
        np.testing.assert_array_equal(
            segment_grains(xmap), segment_grains(xmap, connectivity=1)
        )
        assert SUPPORTED_CONNECTIVITIES == (1, 2)

    def test_labels_are_numbered_row_major_first_seen(self):
        # NOT by grain size, and NOT by orientation: the numbering is
        # the order in which a row-major walk first meets each grain,
        # which is what makes a run reproducible
        angles = [[BIG_ANGLE, BIG_ANGLE, BIG_ANGLE, 0.0]]
        xmap = crystal_map(angles, (1, 4))
        np.testing.assert_array_equal(segment_grains(xmap), [[0, 0, 0, 1]])

    def test_the_angles_are_symmetry_reduced(self):
        # 90 degrees about z is a symmetry operation of m-3m, so the
        # two points are ONE grain although their raw rotation angle
        # is eighteen times the threshold
        xmap = crystal_map([0.0, 90.0], (1, 2))
        raw = np.rad2deg((~xmap.rotations[0] * xmap.rotations[1]).angle)
        assert float(np.atleast_1d(raw)[0]) > 80.0
        np.testing.assert_array_equal(segment_grains(xmap), [[0, 0]])

    def test_a_one_dimensional_map_is_a_single_row(self):
        # the frozen shape contract: labels are always two
        # dimensional, so that the engine's ``(1, n)`` navigation
        # shape and the public return agree
        arrays, size = create_coordinate_arrays((4,), (1.0,))
        arrays["rotations"] = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)),
            np.deg2rad([0.0, 0.1, BIG_ANGLE, BIG_ANGLE]),
        )
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        labels = segment_grains(CrystalMap(**arrays))
        assert labels.shape == (1, 4)
        np.testing.assert_array_equal(labels, [[0, 0, 1, 1]])

    def test_a_single_column_map_is_a_single_column(self):
        # ADDED 2026-09-07 at the Stage B failing-tests review.  On the
        # installed orix a ``(3, 1)`` navigation shape ALSO flattens to
        # ``xmap.shape == (3,)`` and ``ndim == 1``, so a rule of the
        # form "a one dimensional map is a single row" would return
        # ``(1, 3)`` here and
        # ``_reference._flatten_labels`` -- which compares the shape
        # with the navigation shape EXACTLY -- would reject it.  The
        # row and column grids settle it: this map's rows vary and its
        # columns do not, so it is a COLUMN
        arrays, size = create_coordinate_arrays((3, 1), (1.0, 1.0))
        arrays["rotations"] = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)),
            np.deg2rad([0.0, 0.1, BIG_ANGLE]),
        )
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        xmap = CrystalMap(**arrays)
        # the library measurement the pin rests on
        assert xmap.shape == (3,)
        np.testing.assert_array_equal(np.asarray(xmap.row), [0, 1, 2])
        np.testing.assert_array_equal(np.asarray(xmap.col), [0, 0, 0])
        labels = segment_grains(xmap)
        assert labels.shape == (3, 1)
        np.testing.assert_array_equal(labels, [[0], [0], [1]])

    def test_two_runs_are_identical(self):
        xmap = crystal_map([[0.0, 0.2, BIG_ANGLE]] * 2, (2, 3))
        np.testing.assert_array_equal(segment_grains(xmap), segment_grains(xmap))

    def test_signature_is_frozen(self):
        parameters = inspect.signature(segment_grains).parameters
        assert list(parameters) == [
            "xmap",
            "misorientation_threshold",
            "connectivity",
        ]
        assert parameters["xmap"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["misorientation_threshold"].default == DEFAULT_THRESHOLD
        assert parameters["connectivity"].default == 1
        for name in ("misorientation_threshold", "connectivity"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name

    def test_a_phase_boundary_is_a_grain_boundary(self):
        # CORRECTED 2026-09-07 at the Stage B failing-tests review.
        # The drafted suite pinned a ValueError on a multi-phase map,
        # which no frozen requirement asks for and which would have
        # narrowed the FROZEN DEFAULT ``reference="auto"`` to
        # single-phase maps: requirements D9.6 says the opposite for
        # the engine ("the engine itself is phase-agnostic per
        # grain"), so a multi-phase map that Stage A indexes with an
        # explicit reference would have raised on the default path.
        # A phase boundary is simply a grain boundary -- the edge
        # never exists -- which is the natural extension of the
        # connected-components rule and needs no new decision.  The
        # single-phase restriction of D9.6 stays on the STRESS path.
        #
        # Both points here carry the SAME rotation, so the only reason
        # the map holds two grains is the phase change
        multi = crystal_map([0.0, 0.0], (1, 2), phase_ids=[0, 1])
        np.testing.assert_array_equal(segment_grains(multi), [[0, 1]])
        # and the same rotations in ONE phase are one grain, so the
        # split above is the phase and nothing else
        single = crystal_map([0.0, 0.0], (1, 2))
        np.testing.assert_array_equal(segment_grains(single), [[0, 0]])

    def test_each_phase_segments_with_its_own_symmetry(self):
        # 90 degrees about z is a symmetry operation of m-3m but not
        # of 6/mmm, so the same angle pair is one grain inside the
        # cubic phase and two inside the hexagonal one.  A single
        # point group applied to the whole map cannot give both
        cubic = crystal_map([0.0, 90.0], (1, 2))
        np.testing.assert_array_equal(segment_grains(cubic), [[0, 0]])
        hexagonal = crystal_map([0.0, 90.0], (1, 2), phase_ids=[1, 1])
        np.testing.assert_array_equal(segment_grains(hexagonal), [[0, 1]])

    def test_guards(self):
        xmap = crystal_map([0.0, 0.0], (1, 2))
        with pytest.raises(ValueError):
            segment_grains(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="threshold"):
            segment_grains(xmap, misorientation_threshold=0.0)
        with pytest.raises(ValueError, match="connectivity"):
            segment_grains(xmap, connectivity=3)


class TestThreshold:
    """The threshold sweep and THE comparison-side pin.  [D11.1]"""

    def test_a_sweep_moves_the_boundary(self):
        xmap = crystal_map([0.0, 2.0], (1, 2))
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=5.0), [[0, 0]]
        )
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=1.0), [[0, 1]]
        )

    def test_a_pair_at_the_threshold_is_a_boundary(self):
        # THE ``>=`` mutant of plan section 3.4.  A sweep alone cannot
        # see it, so the angle is MEASURED with the call and the
        # conversion the module docstring commits to
        # (``numpy.rad2deg`` of ``Orientation.angle_with``, no
        # ``degrees`` keyword, which does not exist at the orix 0.12.1
        # floor) and fed back as the threshold
        xmap = crystal_map([0.0, 2.0], (1, 2))
        orientations = Orientation(
            xmap.rotations.data, symmetry=xmap.phases[0].point_group
        )
        angle = float(
            np.rad2deg(np.atleast_1d(orientations[0].angle_with(orientations[1]))[0])
        )
        assert angle == pytest.approx(2.0, abs=1e-9)
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=angle), [[0, 1]]
        )
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=np.nextafter(angle, np.inf)),
            [[0, 0]],
        )

    def test_a_threshold_just_below_and_just_above_a_pair(self):
        # the robust half of the same pin, which holds whatever the
        # last bit of the angle does
        xmap = crystal_map([0.0, 2.0], (1, 2))
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=2.0 - 1e-6), [[0, 1]]
        )
        np.testing.assert_array_equal(
            segment_grains(xmap, misorientation_threshold=2.0 + 1e-6), [[0, 0]]
        )


# ======== D11.2 -- image quality and the reference selection ======== #


class TestImageQuality:
    """The selection kernel of requirements D11.2.  [D11.2]"""

    def test_it_is_the_kikuchipy_kernel(self):
        patterns = ni_patterns()
        got = image_quality(patterns)
        assert got.shape == (9,)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, expected_quality(patterns), rtol=0, atol=1e-12)

    def test_the_qualities_really_differ(self):
        # the guard on the oracle: a selection test means nothing if
        # every candidate scores the same
        quality = image_quality(ni_patterns())
        assert quality.std() > 0

    def test_only_the_asked_for_patterns_are_scored(self):
        # ADDED 2026-09-08 (Stage B adversarial review): the selection
        # scores the CANDIDATES, not the whole map, which is where the
        # memory of the frozen default route was being spent
        patterns = ni_patterns()
        indices = np.array([7, 2, 2], dtype=np.int64)
        got = image_quality(patterns, indices)
        assert got.shape == (3,)
        expected = expected_quality(patterns)
        np.testing.assert_allclose(got, expected[indices], rtol=0, atol=1e-12)

    def test_a_lazy_stack_is_read_in_blocks_and_never_whole(self):
        # the memory claim of the public Memory note, MEASURED: the
        # Dask graph is asked for a few patterns at a time, so the peak
        # is a block and not the data set.  A 576 MB wafer, or a 57 GB
        # map, used to be materialised here before the first fit
        dask = pytest.importorskip("dask.array")
        patterns = ni_patterns()
        loaded = []

        def counted(block, block_info=None):
            loaded.append(int(block.shape[0]))
            return block

        lazy = dask.from_array(patterns, chunks=(1, 60, 60)).map_blocks(
            counted, dtype=patterns.dtype
        )
        got = image_quality(lazy, np.array([0, 4], dtype=np.int64))
        np.testing.assert_allclose(
            got, expected_quality(patterns)[[0, 4]], rtol=0, atol=1e-12
        )
        # only the two asked-for patterns were ever read
        assert sum(loaded) == 2

    def test_guards(self):
        with pytest.raises(ValueError):
            image_quality(np.zeros((60, 60)))
        patterns = ni_patterns(3)
        with pytest.raises(ValueError, match="one dimensional integer"):
            image_quality(patterns, np.zeros((2, 2), dtype=np.int64))
        with pytest.raises(ValueError, match="one dimensional integer"):
            image_quality(patterns, np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="flat index"):
            image_quality(patterns, np.array([0, 3], dtype=np.int64))


class TestSelectReferences:
    """Argmax quality per grain, ties to the lowest flat index.
    [D11.2]"""

    def test_one_grain_takes_the_best_pattern(self):
        patterns = ni_patterns()
        labels = np.zeros(9, dtype=np.int32)
        got = select_references(patterns, labels)
        assert got.shape == (1,)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, expected_reference(patterns, labels))

    def test_each_grain_takes_its_own_best_pattern(self):
        patterns = ni_patterns()
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        got = select_references(patterns, labels)
        np.testing.assert_array_equal(got, expected_reference(patterns, labels))
        # and the two references really are different points, or the
        # per-grain part of the contract is untested
        assert got[0] != got[1]

    def test_a_tie_goes_to_the_lowest_flat_index(self):
        # two bitwise identical patterns score bitwise identical
        # image qualities, which is the only way to build an exact tie
        patterns = ni_patterns(3)
        patterns[1] = patterns[0]
        quality = image_quality(patterns)
        best = int(np.argmax(quality))
        if best not in (0, 1):
            # the shipped pattern which wins outright is excluded so
            # that the tie is the decision
            patterns[best] = patterns[0]
        labels = np.zeros(3, dtype=np.int32)
        assert select_references(patterns, labels)[0] == 0

    def test_the_pairing_is_positional_in_the_sorted_labels(self):
        # requirements D11.3: labels may have GAPS, and entry ``i``
        # belongs to the ``i``-th sorted unique label, not to label
        # ``i``.  This is the "per-grain reference off by one in the
        # flat index" mutant of plan section 3.4
        patterns = ni_patterns(6)
        labels = np.array([7, 7, 0, 0, 3, 3], dtype=np.int32)
        got = select_references(patterns, labels)
        assert got.shape == (3,)
        expected = expected_reference(patterns, labels)
        np.testing.assert_array_equal(got, expected)
        # the first entry serves label 0, which lives at flat indices
        # 2 and 3, so a positional error is visible without any
        # tolerance
        assert got[0] in (2, 3)
        assert got[1] in (4, 5)
        assert got[2] in (0, 1)

    def test_unindexed_points_are_never_selected(self):
        patterns = ni_patterns(4)
        quality = image_quality(patterns)
        best = int(np.argmax(quality))
        labels = np.zeros(4, dtype=np.int32)
        labels[best] = UNINDEXED
        got = select_references(patterns, labels)
        assert got.shape == (1,)
        assert got[0] != best

    def test_an_unselectable_point_is_never_selected(self):
        # the same rule as the unindexed one, for a point the caller
        # masked out rather than one orix left unlabelled (added
        # 2026-09-08, Stage B adversarial review)
        patterns = ni_patterns(4)
        quality = expected_quality(patterns)
        best = int(np.argmax(quality))
        labels = np.zeros(4, dtype=np.int32)
        selectable = np.ones(4, dtype=bool)
        selectable[best] = False
        got = select_references(patterns, labels, selectable)
        assert got.shape == (1,)
        assert got[0] != best
        allowed = np.array([i for i in range(4) if i != best])
        assert got[0] == allowed[int(np.argmax(quality[allowed]))]

    def test_a_grain_with_nothing_selectable_falls_back_to_its_own_points(self):
        # such a grain is correlated nowhere, so its index is never
        # read; leaving it out instead would break the one-index-per-
        # label pairing
        patterns = ni_patterns(4)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        selectable = np.array([True, True, False, False])
        got = select_references(patterns, labels, selectable)
        assert got.shape == (2,)
        assert got[0] in (0, 1)
        assert got[1] in (2, 3)

    def test_guards(self):
        patterns = ni_patterns(3)
        with pytest.raises(ValueError):
            select_references(patterns, np.zeros(2, dtype=np.int32))
        with pytest.raises(ValueError):
            select_references(patterns[0], np.zeros(1, dtype=np.int32))
        # a map with no labelled point has no grain to serve
        with pytest.raises(ValueError):
            select_references(patterns, np.full(3, UNINDEXED, dtype=np.int32))
        with pytest.raises(ValueError, match="selectable"):
            select_references(
                patterns, np.zeros(3, dtype=np.int32), np.ones(2, dtype=bool)
            )


# ============ D11.3 -- the ``reference="auto"`` wiring ============== #


class TestAutoReference:
    """``"auto"`` is the segmentation, then the selection, then the
    explicit per-grain pairing.  The Stage A ``NotImplementedError``
    pin of this mode is replaced by these.  [D11.3]"""

    def test_auto_segments_and_selects(self):
        patterns = ni_patterns(6)
        angles = [[0.0, 0.2, 0.1], [BIG_ANGLE, BIG_ANGLE, BIG_ANGLE + 0.2]]
        xmap = crystal_map(angles, (2, 3))
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        grain_id, reference_index = resolve_reference(
            "auto", None, (2, 3), xmap=xmap, patterns=patterns
        )
        np.testing.assert_array_equal(grain_id, labels)
        references = expected_reference(patterns, labels)
        np.testing.assert_array_equal(
            reference_index, np.where(labels == 0, references[0], references[1])
        )
        assert grain_id.dtype == np.int32
        assert reference_index.dtype == np.int32

    def test_auto_uses_a_supplied_grain_map_unchanged(self):
        # every orientation is identical, so the segmentation would
        # give ONE grain: two grains in the result can only come from
        # the supplied map, which is what requirements D11.3 asks for
        patterns = ni_patterns(6)
        xmap = crystal_map([0.0] * 6, (2, 3))
        labels = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        grain_id, reference_index = resolve_reference(
            "auto", labels, (2, 3), xmap=xmap, patterns=patterns
        )
        np.testing.assert_array_equal(grain_id, labels.ravel())
        references = expected_reference(patterns, labels)
        assert references[0] in (0, 1, 2)
        assert references[1] in (3, 4, 5)
        np.testing.assert_array_equal(
            reference_index,
            np.where(labels.ravel() == 0, references[0], references[1]),
        )

    def test_auto_follows_the_threshold(self):
        patterns = ni_patterns(4)
        xmap = crystal_map([0.0, 2.0, 2.1, 2.2], (1, 4))
        # at the default threshold the whole row is one grain
        grain_id, _ = resolve_reference(
            "auto", None, (1, 4), xmap=xmap, patterns=patterns
        )
        np.testing.assert_array_equal(grain_id, np.zeros(4, dtype=np.int32))
        # and below the first step it is two
        grain_id, _ = resolve_reference(
            "auto",
            None,
            (1, 4),
            misorientation_threshold=1.0,
            xmap=xmap,
            patterns=patterns,
        )
        np.testing.assert_array_equal(grain_id, [0, 1, 1, 1])

    def test_auto_reports_unindexed_points(self):
        patterns = ni_patterns(3)
        xmap = crystal_map([0.0, 0.0, 0.0], (1, 3), unindexed=(1,))
        grain_id, reference_index = resolve_reference(
            "auto", None, (1, 3), xmap=xmap, patterns=patterns
        )
        assert grain_id[1] == UNLABELLED
        assert reference_index[1] == UNLABELLED
        # the unindexed point splits the row, so the two ends have
        # their own references
        assert reference_index[0] == 0
        assert reference_index[2] == 2

    def test_auto_agrees_with_the_explicit_per_grain_mode(self):
        # the two modes share the pairing and the same-grain guard, so
        # feeding the selection back explicitly must give the same
        # answer: that is what makes "auto" a wiring point and not a
        # second implementation
        patterns = ni_patterns(6)
        angles = [[0.0, 0.2, 0.1], [BIG_ANGLE, BIG_ANGLE, BIG_ANGLE + 0.2]]
        xmap = crystal_map(angles, (2, 3))
        labels = np.asarray(segment_grains(xmap))
        indices = select_references(patterns, labels.ravel())
        automatic = resolve_reference(
            "auto", None, (2, 3), xmap=xmap, patterns=patterns
        )
        explicit = resolve_reference(indices, labels, (2, 3))
        np.testing.assert_array_equal(automatic[0], explicit[0])
        np.testing.assert_array_equal(automatic[1], explicit[1])

    def test_a_reference_lies_inside_its_own_grain(self):
        # the guard of requirements D11.3 holds for the automatic
        # pairing too, which is the point of routing it through the
        # same helper
        patterns = ni_patterns(6)
        angles = [[0.0, 0.2, 0.1], [BIG_ANGLE, BIG_ANGLE, BIG_ANGLE + 0.2]]
        xmap = crystal_map(angles, (2, 3))
        grain_id, reference_index = resolve_reference(
            "auto", None, (2, 3), xmap=xmap, patterns=patterns
        )
        for point in range(6):
            assert grain_id[point] == grain_id[reference_index[point]]

    def test_an_unknown_string_is_still_a_value_error(self):
        with pytest.raises(ValueError, match="auto"):
            resolve_reference("best", None, (1, 2))

    def test_a_masked_out_point_is_never_a_reference(self):
        # ADDED 2026-09-08 at the Stage B adversarial review, which
        # found the navigation mask reaching the fit list and NOTHING
        # else: the best-quality pattern of a grain was selected as its
        # reference even when the caller had masked it out, so every
        # strain in that grain was measured against a pattern the
        # caller had said not to trust -- and the reference's own
        # homography came back NaN
        patterns = ni_patterns(6)
        xmap = crystal_map([0.0] * 6, (2, 3))
        unmasked, unmasked_reference = resolve_reference(
            "auto", None, (2, 3), xmap=xmap, patterns=patterns
        )
        best = int(unmasked_reference[0])
        mask = np.zeros((2, 3), dtype=bool)
        mask.ravel()[best] = True
        grain_id, reference_index = resolve_reference(
            "auto",
            None,
            (2, 3),
            xmap=xmap,
            patterns=patterns,
            navigation_mask=mask,
        )
        assert np.all(reference_index != best)
        # the second best of the same grain, computed here
        quality = expected_quality(patterns)
        allowed = np.array([i for i in range(6) if i != best])
        assert reference_index[0] == allowed[int(np.argmax(quality[allowed]))]
        # and the grain identifiers are untouched: the mask says which
        # patterns are correlated, not which points belong to a grain
        np.testing.assert_array_equal(grain_id, unmasked)

    def test_a_grain_masked_out_entirely_still_gets_an_index(self):
        # the pairing wants one index per label whatever the mask
        # does; the index is never read, because no point of such a
        # grain is correlated
        patterns = ni_patterns(6)
        angles = [[0.0, 0.2, 0.1], [BIG_ANGLE, BIG_ANGLE, BIG_ANGLE + 0.2]]
        xmap = crystal_map(angles, (2, 3))
        mask = np.zeros((2, 3), dtype=bool)
        mask[1] = True
        grain_id, reference_index = resolve_reference(
            "auto",
            None,
            (2, 3),
            xmap=xmap,
            patterns=patterns,
            navigation_mask=mask,
        )
        np.testing.assert_array_equal(grain_id, [0, 0, 0, 1, 1, 1])
        assert np.all(reference_index[:3] < 3)
        assert np.all(reference_index[3:] >= 3)


# ====== The grid, the symmetry and the lazy read of the selection === #


class TestMapGrid:
    """Every grid-shaped function reads the map's own row and column
    grids, and names a map which has none.  [D11.1]"""

    def test_a_map_without_a_grid_is_named(self):
        # ADDED 2026-09-08 (Stage B adversarial review): orix gives a
        # map whose points share one scan position the shape ``()``,
        # and reading its row grid then raises "not enough values to
        # unpack", which names nothing the caller passed
        xmap = CrystalMap(
            rotations=Rotation.identity((2,)),
            phase_id=np.zeros(2, dtype=int),
            phase_list=PhaseList(Phase(name="ni", space_group=225)),
            x=np.zeros(2),
            y=np.zeros(2),
        )
        assert xmap.shape == ()
        with pytest.raises(ValueError, match="no map grid"):
            segment_grains(xmap)

    def test_a_column_map_segments_to_a_column(self):
        # requirements D11.1(b), the half a row-shaped assumption
        # breaks: orix flattens both a row and a column map to ``(n,)``
        arrays, size = create_coordinate_arrays((3, 1), (1.0, 1.0))
        arrays["rotations"] = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)), np.deg2rad([0.0, 0.1, BIG_ANGLE])
        )
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        xmap = CrystalMap(**arrays)
        assert xmap.shape == (3,)
        np.testing.assert_array_equal(xmap.col, [0, 0, 0])
        np.testing.assert_array_equal(segment_grains(xmap), [[0], [0], [1]])


class TestPointGroup:
    """A phase with no point group has no symmetry to reduce by, and
    says so.  [D11.1]"""

    @staticmethod
    def quarter_turn_map(point_group):
        arrays, size = create_coordinate_arrays((2, 2), (1.0, 1.0))
        arrays["rotations"] = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)), np.deg2rad([0.0, 90.0, 0.0, 90.0])
        )
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase("a", point_group=point_group))
        return CrystalMap(**arrays)

    def test_a_cubic_phase_reduces_a_quarter_turn_away(self):
        # the guard on the oracle: with the symmetry, the four points
        # are ONE grain although two of them are 90 degrees from the
        # other two
        labels = segment_grains(self.quarter_turn_map("m-3m"))
        np.testing.assert_array_equal(labels, [[0, 0], [0, 0]])

    def test_a_phase_without_a_point_group_warns_and_does_not_reduce(self):
        # ADDED 2026-09-08 at the Stage B adversarial review.  The
        # docstring promises symmetry-reduced angles; orix leaves the
        # symmetry at C1 when there is no point group, so the angles
        # are the RAW ones and the segmentation is a different one --
        # two grains here instead of one.  That is not a defect that
        # can be fixed without inventing a symmetry, so it is WARNED
        # about instead of being left silent
        with pytest.warns(UserWarning, match="no point group"):
            labels = segment_grains(self.quarter_turn_map(None))
        np.testing.assert_array_equal(labels, [[0, 1], [0, 1]])


class TestSeveralRotationsPerPoint:
    """A map from dictionary indexing with ``n_best > 1`` is read at
    its best rotation.  [D11.1]"""

    def test_the_best_rotation_is_the_maps_own(self):
        # ADDED 2026-09-08 (Stage B adversarial review, the coverage
        # gate): the reshape had no test, so a map carrying several
        # rotations per point could have been segmented on the WORST
        # of them without anything noticing
        arrays, size = create_coordinate_arrays((1, 3), (1.0, 1.0))
        best = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)), np.deg2rad([0.0, 0.1, BIG_ANGLE])
        )
        worst = Rotation.from_axes_angles(
            np.tile([0.0, 0.0, 1.0], (size, 1)), np.deg2rad([0.0, BIG_ANGLE, 0.0])
        )
        arrays["rotations"] = Rotation(np.stack([best.data, worst.data], axis=1))
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        xmap = CrystalMap(**arrays)
        assert xmap.rotations.data.ndim == 3
        np.testing.assert_array_equal(segment_grains(xmap), [[0, 0, 1]])
        # the second rotation would give [[0, 1, 1]], so the choice is
        # visible rather than assumed
        assert not np.array_equal(best.data, worst.data)
