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

"""High angular resolution kernel average misorientation (HR-KAM).

There is no separately standardized "HR-KAM" in the literature, so
requirements D12 freezes one HERE: the standard KAM evaluated on the
HR ROTATION FIELD, that is on the ``rotation_vector`` property the
tensor chain reports, rather than on the Hough orientations.

Everything about it is frozen, and each clause exists to kill one
way of getting it wrong:

- **Unit: milliradians.** The HR noise floor is 5e-5 to 1e-4 rad,
  two orders below what conventional KAM is quoted in, so degrees
  would print zeros. A radians-in-a-millradian-name mutant dies at
  ``test_hrebsd_kam.py::TestUnits``.
- **Kernel: every point within Chebyshev distance <= order** of the
  centre on the square grid, the MTEX-style "all within order"
  convention rather than the OIM perimeter-only one. ``order=1``
  (default) is therefore the 8 surrounding points, not 4.
- **Disorientation per pair:** ``angle(R_p R_q^T)`` of the two HR
  rotations. They are measured relative to a COMMON grain reference,
  so symmetry operators are unnecessary and are never applied.
  **That angle must NOT be computed as ``arccos((tr - 1) / 2)``**
  (measured 2026-09-07, Stage B failing-tests review):
  ``arccos`` has a square-root singularity at the identity and loses
  about eight significant digits at the 1e-4 rad scale the HR
  rotation field lives at, which is three orders above the band the
  constant-curvature identity is asserted in. Use
  ``arctan2(||skew(M)|| / 2, (tr(M) - 1) / 2)``, which reproduces the
  closed form to 3.7e-16 relative. The oracle of
  ``test_hrebsd_kam.py`` carries the same note and pins its own
  conditioning.
- **Masking:** a pair must share ``grain_id``, always -- the HR
  field is only defined within a grain -- and, if *psi_max* is
  given, must sit at or below it. Points left with no valid
  neighbour are NaN, never zero.
- **Mean over the surviving pairs**, not median.

The constant-curvature identity the oracle checks (requirements D12,
validation V7) carries a fixed kernel factor which is geometry and
not discretization error: on a field ``omega_3 = kappa * x1`` that
varies along one grid axis, six of the eight order-1 neighbours
differ by ``kappa * step`` and the two along the other axis by zero,
so the kernel mean is ``(6/8) * kappa * step`` EXACTLY.  The test
computes that expectation from the kernel offsets and never asserts
``kappa * step``.
"""

import numpy as np

from kikuchipy.indexing._hrebsd._segmentation import map_grids

# The unit of the returned map and of *psi_max*, frozen
KAM_UNIT: str = "mrad"

# Radians to milliradians
RADIANS_TO_MRAD: float = 1e3

# The properties a map must carry to have a HR-KAM: the rotation
# field of the tensor chain and the grain identifiers of the engine
REQUIRED_PROP_NAMES: tuple[str, ...] = ("rotation_vector", "grain_id")

# Entries of one stored rotation vector, ``omega = axis * angle``
ROTATION_VECTOR_SIZE: int = 3

# The grain identifier of a point outside every grain, the value
# :data:`~kikuchipy.indexing._hrebsd._reference.UNLABELLED` carries
UNLABELLED: int = -1


def kernel_offsets(order: int) -> np.ndarray:
    """Return the ``(row, col)`` offsets of the KAM kernel.

    Every point within Chebyshev distance *order* of the centre,
    excluding the centre itself: 8 offsets at ``order=1``, 24 at
    ``order=2`` (requirements D12).

    Parameters
    ----------
    order
        Kernel order, a positive integer.

    Returns
    -------
    offsets
        Array of shape ``(4 * order * (order + 1), 2)`` and 64-bit
        integer data type, in row-major order of the offsets.

    Raises
    ------
    ValueError
        If *order* is not a positive integer.
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError(f"order {order!r} must be a positive integer")
    order = int(order)
    if order < 1:
        raise ValueError(f"order {order} must be a positive integer")
    offsets = [
        (drow, dcol)
        for drow in range(-order, order + 1)
        for dcol in range(-order, order + 1)
        if (drow, dcol) != (0, 0)
    ]
    return np.asarray(offsets, dtype=np.int64)


def hrebsd_kam(xmap, *, order: int = 1, psi_max: float | None = None) -> np.ndarray:
    """Return the high angular resolution kernel average
    misorientation.

    The standard KAM evaluated on the high angular resolution
    rotation field of
    :func:`~kikuchipy.indexing.hrebsd_strain_stress`, in
    milliradians: each point is averaged over the disorientation to
    every neighbour of its kernel which lies in the SAME grain.

    Parameters
    ----------
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map returned by
        :func:`~kikuchipy.indexing.hrebsd_strain_stress`, carrying
        its ``"rotation_vector"`` property and the ``"grain_id"``
        property of :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`.
    order : int, optional
        Kernel order. Every point within Chebyshev distance *order*
        of the centre is a neighbour, which is 8 points at the
        default 1 and 24 at 2, the MTEX-style convention rather than
        the perimeter-only one.
    psi_max : float, optional
        Upper bound on a single pair's disorientation in
        MILLIRADIANS. Pairs ABOVE it are dropped, which guards
        against sub-grain boundaries inside the kernel; a pair
        sitting exactly AT it is kept, the comparison being ``<=``
        (requirements D12). If not given, no pair is dropped for its
        size.

    Returns
    -------
    kam
        Array of the map's navigation shape ``(ny, nx)`` and 64-bit
        float data type in MILLIRADIANS. Points with no surviving
        neighbour pair, points outside any grain and points whose own
        rotation is NaN are NaN. The shape is ALWAYS two dimensional
        and comes from the map's row and column grids, not from
        :attr:`~orix.crystal_map.CrystalMap.shape`, which orix
        flattens to ``(n,)`` for a map one point wide or one point
        tall: a map of one row returns ``(1, nx)`` and a map of one
        column returns ``(ny, 1)``, as
        :func:`~kikuchipy.indexing.segment_grains` does.

    Raises
    ------
    ValueError
        If *xmap* does not carry both ``"rotation_vector"`` and
        ``"grain_id"``, naming
        :func:`~kikuchipy.indexing.hrebsd_strain_stress` and
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; if *xmap* has no
        map grid, which is what orix reports as the shape ``()``; if
        *order* is not a positive integer; or if *psi_max* is not
        positive.

    Notes
    -----
    The disorientation of a pair is the rotation angle of
    ``R_p R_q^T``, with both rotations reconstructed from the stored
    rotation vectors. Because HR rotations are measured against a
    common reference INSIDE one grain, no symmetry operator is
    applied anywhere: symmetry-equivalent descriptions cannot arise
    (requirements D12).

    The unit is milliradians because the HR rotation noise floor is
    5e-5 to 1e-4 radians, two orders below the degrees conventional
    KAM is quoted in.

    Pairs are dropped, never fabricated: a pair touching a
    non-converged point, whose rotation vector is NaN, contributes
    nothing rather than a zero.

    See Also
    --------
    kikuchipy.indexing.hrebsd_strain_stress
    kikuchipy.signals.EBSD.hrebsd_dic
    """
    properties = xmap.prop
    if "rotation_vector" not in properties:
        raise ValueError(
            "xmap does not carry the 'rotation_vector' property this function "
            "averages; pass the crystal map "
            "kikuchipy.indexing.hrebsd_strain_stress returned"
        )
    if "grain_id" not in properties:
        raise ValueError(
            "xmap does not carry the 'grain_id' property, so no pair can be "
            "checked for sharing a grain; pass the crystal map "
            "EBSD.hrebsd_dic returned"
        )
    offsets = kernel_offsets(order)
    if psi_max is not None:
        psi_max = float(psi_max)
        if not psi_max > 0:
            raise ValueError(
                f"psi_max {psi_max} must be a positive disorientation in milliradians"
            )

    # The two dimensional shape comes from the grids, never from
    # ``CrystalMap.shape``, which orix flattens to ``(n,)`` for a map
    # one point wide or one point tall
    rows, cols = map_grids(xmap)
    ny = int(rows.max()) + 1
    nx = int(cols.max()) + 1
    size = rows.size

    field = np.full((ny, nx, ROTATION_VECTOR_SIZE), np.nan, dtype=np.float64)
    field[rows, cols] = np.asarray(
        properties["rotation_vector"], dtype=np.float64
    ).reshape(size, ROTATION_VECTOR_SIZE)
    grain = np.full((ny, nx), -1, dtype=np.int64)
    # A NaN in a FLOAT ``grain_id`` casts to an undefined integer, so
    # it is turned into the unlabelled -1 first rather than left to the
    # platform (2026-09-08, Stage B adversarial review); the documented
    # property is int32 and takes this path unchanged
    identifiers = np.asarray(properties["grain_id"]).ravel()
    grain[rows, cols] = np.where(
        np.isfinite(identifiers), identifiers, UNLABELLED
    ).astype(np.int64)

    usable = np.all(np.isfinite(field), axis=-1) & (grain >= 0)
    matrices = rotation_matrices(field)

    total = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int64)
    for drow, dcol in offsets.tolist():
        row0, row1 = max(0, -drow), ny - max(0, drow)
        col0, col1 = max(0, -dcol), nx - max(0, dcol)
        if row1 <= row0 or col1 <= col0:
            continue
        here = (slice(row0, row1), slice(col0, col1))
        there = (
            slice(row0 + drow, row1 + drow),
            slice(col0 + dcol, col1 + dcol),
        )
        valid = usable[here] & usable[there] & (grain[here] == grain[there])
        if not valid.any():
            continue
        angles = pair_angles(matrices[here], matrices[there])
        if psi_max is not None:
            # A NaN angle compares ``False`` here, which is the same
            # exclusion ``valid`` already carries.
            #
            # ``<=`` and not ``<``: requirements D12 drops the pairs
            # ABOVE the threshold, so a pair sitting exactly AT it is
            # KEPT -- the KAM analogue of the segmentation threshold
            # side, and pinned the same way, by feeding back an angle
            # the module itself reports
            # (``test_hrebsd_kam.py::TestPsiMax::test_a_pair_exactly
            # _at_psi_max_is_kept``)
            valid = valid & (angles <= psi_max)
        total[here] += np.where(valid, angles, 0.0)
        count[here] += valid

    kam = np.full((ny, nx), np.nan, dtype=np.float64)
    averaged = count > 0
    kam[averaged] = total[averaged] / count[averaged]
    return kam


def rotation_matrices(vectors: np.ndarray) -> np.ndarray:
    """Return the rotation matrices of a field of rotation vectors.

    Rodrigues' formula, ``R = I + sin(t) K + (1 - cos(t)) K K`` with
    ``t`` the vector's length and ``K`` the cross-product matrix of
    its direction.  A zero vector gives the identity exactly, and a
    non-finite one gives a non-finite matrix, which is how a
    non-converged point drops its pairs rather than contributing a
    zero.

    Parameters
    ----------
    vectors
        Rotation vectors of shape ``(..., 3)`` in radians.

    Returns
    -------
    matrices
        Array of shape ``(..., 3, 3)`` and 64-bit float data type.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.sqrt(np.sum(vectors * vectors, axis=-1))
    # The division is only a direction, so the zero-angle points are
    # divided by one and overwritten with the identity below
    safe = np.where(angle == 0.0, 1.0, angle)
    axis = vectors / safe[..., np.newaxis]
    zero = np.zeros(angle.shape, dtype=np.float64)
    cross = np.stack(
        [
            np.stack([zero, -axis[..., 2], axis[..., 1]], axis=-1),
            np.stack([axis[..., 2], zero, -axis[..., 0]], axis=-1),
            np.stack([-axis[..., 1], axis[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )
    identity = np.broadcast_to(np.eye(3), cross.shape)
    matrices = (
        identity
        + np.sin(angle)[..., np.newaxis, np.newaxis] * cross
        + (1.0 - np.cos(angle))[..., np.newaxis, np.newaxis] * (cross @ cross)
    )
    return np.where((angle == 0.0)[..., np.newaxis, np.newaxis], identity, matrices)


def pair_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the disorientation of two fields of rotations, in
    milliradians.

    The angle of ``R_p R_q^T`` (requirements D12), evaluated as
    ``arctan2(||skew(M)|| / 2, (tr(M) - 1) / 2)`` and **never** as
    ``arccos((tr(M) - 1) / 2)``: ``arccos`` has a square-root
    singularity at the identity and loses about eight significant
    digits at the 1e-4 radian scale the high angular resolution
    rotation field lives at (measured 2026-09-07, requirements D12).

    Parameters
    ----------
    first, second
        Rotation matrices of shape ``(..., 3, 3)``.

    Returns
    -------
    angles
        Array of the broadcast leading shape and 64-bit float data
        type, in MILLIRADIANS.
    """
    matrix = first @ np.swapaxes(second, -1, -2)
    axis_x = matrix[..., 2, 1] - matrix[..., 1, 2]
    axis_y = matrix[..., 0, 2] - matrix[..., 2, 0]
    axis_z = matrix[..., 1, 0] - matrix[..., 0, 1]
    sine = 0.5 * np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cosine = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    return np.arctan2(sine, cosine) * RADIANS_TO_MRAD
