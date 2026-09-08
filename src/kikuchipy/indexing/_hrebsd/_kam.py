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

# The unit of the returned map and of *psi_max*, frozen
KAM_UNIT: str = "mrad"

# Radians to milliradians
RADIANS_TO_MRAD: float = 1e3

# The properties a map must carry to have a HR-KAM: the rotation
# field of the tensor chain and the grain identifiers of the engine
REQUIRED_PROP_NAMES: tuple[str, ...] = ("rotation_vector", "grain_id")


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
    raise NotImplementedError(
        "kernel_offsets arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


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
        MILLIRADIANS. Pairs above it are dropped, which guards
        against sub-grain boundaries inside the kernel. If not given,
        no pair is dropped for its size.

    Returns
    -------
    kam : numpy.ndarray
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
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; if *order* is not
        a positive integer; or if *psi_max* is not positive.

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
    raise NotImplementedError(
        "hrebsd_kam arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )
