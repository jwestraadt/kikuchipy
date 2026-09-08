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

Three modes (requirements D11.3).  ``(row, col)`` is one global
reference and one implicit grain; an array of flat indices is the
per-grain mode and needs a user supplied grain map; and ``"auto"``,
the default, segments the map with
:func:`~kikuchipy.indexing.segment_grains` unless a grain map is
given, then picks each grain's highest image quality pattern
(requirements D11.1 and D11.2).  The ``"auto"`` path is wired HERE
and does nothing else itself: it is exactly the two Stage B
functions of :mod:`~kikuchipy.indexing._hrebsd._segmentation`
followed by the same per-grain pairing the explicit index array
takes, so the two modes cannot drift apart.
"""

import numpy as np

from kikuchipy.indexing._hrebsd._segmentation import segment_grains, select_references

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
        the ``"auto"`` mode, and there only when *grain_labels* is
        not given. Default is 5.0.
    xmap
        :class:`~orix.crystal_map.CrystalMap` of the map, used only
        by the ``"auto"`` mode, and there only when *grain_labels*
        is not given.
    patterns
        Patterns of shape ``(n, nrows, ncols)``, used only by the
        ``"auto"`` mode, whose per-grain selection maximizes the
        image quality of the raw patterns.

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
    ValueError
        If *reference* is an unknown string; if a ``(row, col)``
        tuple is outside the map; if an index array is given without
        *grain_labels*, does not hold one index per grain label, or
        holds an index which is outside the map or outside the grain
        it is paired with; if *grain_labels* does not match
        *navigation_shape* or does not hold integers; or, in the
        ``"auto"`` mode, whatever
        :func:`~kikuchipy.indexing.segment_grains` and the
        per-grain selection raise for a missing or unusable *xmap*
        or *patterns*.

    Notes
    -----
    A ``(row, col)`` reference is ONE global reference: with
    *grain_labels* it still serves every label, and only the reported
    ``grain_id`` comes from the labels, which is what requirements
    D11.3 freezes.  An index array is the per-grain mode, and there
    each index must lie inside its own grain.

    ``"auto"`` is the per-grain mode with the two choices made for
    the caller: the labels come from
    :func:`~kikuchipy.indexing.segment_grains` at
    *misorientation_threshold* and the frozen default connectivity of
    requirements D11.1 (the public signature carries no connectivity
    knob), unless *grain_labels* is given, in which case the supplied
    map is used unchanged; and the index of each grain is the point
    of highest image quality in it.  The resulting pairs go through
    the SAME positional pairing and same-grain guard the explicit
    index array does.
    """
    navigation_shape = tuple(int(i) for i in navigation_shape)
    if len(navigation_shape) != 2:
        raise ValueError(
            f"navigation_shape {navigation_shape} must hold exactly two entries "
            "(ny, nx)"
        )
    ny, nx = navigation_shape
    map_size = ny * nx

    labels = None
    if grain_labels is not None:
        labels = np.asarray(grain_labels)
        if labels.shape != navigation_shape:
            raise ValueError(
                f"grain_labels of shape {labels.shape} must have the navigation "
                f"shape {navigation_shape}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                f"grain_labels must hold integer labels, not {labels.dtype}"
            )
        labels = labels.ravel().astype(np.int32)

    if isinstance(reference, str):
        if reference == AUTO_REFERENCE:
            if labels is None:
                # The frozen connectivity default of D11.1: the public
                # signature carries no connectivity knob
                segmented = segment_grains(
                    xmap, misorientation_threshold=misorientation_threshold
                )
                labels = _flatten_labels(segmented, navigation_shape)
            indices = np.asarray(select_references(patterns, labels))
            return _resolve_index_array(indices, labels, navigation_shape)
        raise ValueError(
            f"reference {reference!r} must be {AUTO_REFERENCE!r}, a (row, col) "
            "tuple, or an integer array of flat indices with one entry per "
            "grain label"
        )

    if isinstance(reference, np.ndarray):
        return _resolve_index_array(reference, labels, navigation_shape)

    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        row, col = (int(i) for i in reference)
        if not (0 <= row < ny and 0 <= col < nx):
            raise ValueError(
                f"reference (row, col) = ({row}, {col}) is outside the map of "
                f"navigation shape {navigation_shape}"
            )
        flat = row * nx + col
        if labels is None:
            # One global reference IS one implicit grain (D11.3)
            grain_id = np.zeros(map_size, dtype=np.int32)
            reference_index = np.full(map_size, flat, dtype=np.int32)
            return grain_id, reference_index
        grain_id = np.where(labels < 0, UNLABELLED, labels).astype(np.int32)
        reference_index = np.where(grain_id < 0, UNLABELLED, flat).astype(np.int32)
        return grain_id, reference_index

    raise ValueError(
        f"reference {reference!r} must be {AUTO_REFERENCE!r}, a (row, col) tuple, "
        "or an integer array of flat indices with one entry per grain label"
    )


def _flatten_labels(
    labels: np.ndarray, navigation_shape: tuple[int, int]
) -> np.ndarray:
    """Return the grain labels of the ``"auto"`` segmentation as one
    flat 32-bit integer array.

    Parameters
    ----------
    labels
        Labels as :func:`~kikuchipy.indexing.segment_grains` returns
        them, of shape *navigation_shape*.
    navigation_shape
        Map shape ``(ny, nx)``.

    Returns
    -------
    flat
        Array of shape ``(ny * nx,)`` and 32-bit integer data type.

    Raises
    ------
    ValueError
        If the segmentation did not return an integer array of the
        navigation shape, which names the offending shape or data
        type: the ``"auto"`` mode is the only caller and a silent
        reshape here would pair references with the wrong points.
    """
    labels = np.asarray(labels)
    if labels.shape != tuple(navigation_shape):
        raise ValueError(
            f"segment_grains returned labels of shape {labels.shape}, which must "
            f"be the navigation shape {tuple(navigation_shape)}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"segment_grains returned labels of data type {labels.dtype}, which "
            "must be an integer type"
        )
    return labels.ravel().astype(np.int32)


def _resolve_index_array(
    reference: np.ndarray,
    labels: np.ndarray | None,
    navigation_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the grain identifier and reference index of the explicit
    index-array mode of requirements D11.3.

    Parameters
    ----------
    reference
        Integer array of flat map indices, one per grain label.
    labels
        Flattened 32-bit integer grain map, or ``None``.
    navigation_shape
        Map shape ``(ny, nx)``.

    Returns
    -------
    grain_id, reference_index
        Arrays of shape ``(ny * nx,)`` and 32-bit integer data type.

    Raises
    ------
    ValueError
        If *reference* is not a one dimensional non-empty integer
        array; if *labels* is ``None``; if the number of indices and
        the number of grain labels differ; if an index is outside the
        map; or if an index does not lie inside the grain it is the
        reference of.

    Notes
    -----
    The pairing is POSITIONAL in the SORTED unique labels, not by
    label value: with labels ``(0, 3, 7)``, ``reference[1]`` is the
    reference of label 3.  Labels may therefore have gaps.  Each
    index must carry the label of the grain it serves, which is
    checked (added 2026-09-07, Stage A adversarial review): without
    it a transposed or off-by-one index array silently correlates one
    grain's patterns against another grain's reference while
    ``grain_id`` truthfully reports different grains.
    """
    ny, nx = navigation_shape
    map_size = ny * nx
    indices = np.asarray(reference)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            f"reference must be a one dimensional array of flat indices, not one "
            f"of shape {indices.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(
            f"reference must hold integer flat map indices, not {indices.dtype}"
        )
    if labels is None:
        raise ValueError(
            "an array of reference indices requires grain_labels, one 0-based "
            "label map of the navigation shape, since each index is the "
            "reference of one grain label"
        )
    unique = np.unique(labels[labels >= 0])
    if indices.size != unique.size:
        raise ValueError(
            f"reference holds {indices.size} index/indices but grain_labels holds "
            f"{unique.size} grain label(s); there must be exactly one reference "
            "index per grain label"
        )
    if np.any(indices < 0) or np.any(indices >= map_size):
        raise ValueError(
            f"every reference index must be a flat map index within the map size "
            f"{map_size}, got {indices.tolist()}"
        )
    grain_id = np.where(labels < 0, UNLABELLED, labels).astype(np.int32)
    reference_index = np.full(map_size, UNLABELLED, dtype=np.int32)
    for position, label in enumerate(unique):
        index = int(indices[position])
        # Every reference must belong to the grain it serves: the
        # pairing is positional in the sorted unique labels, so a
        # transposed or off-by-one array is otherwise silent
        if int(labels[index]) != int(label):
            raise ValueError(
                f"reference index {index} carries grain label {int(labels[index])} "
                f"but is paired with grain label {int(label)}; reference[i] must be "
                "a flat map index INSIDE the i-th grain of the sorted unique labels "
                "of grain_labels"
            )
        reference_index[labels == label] = np.int32(index)
    return grain_id, reference_index
