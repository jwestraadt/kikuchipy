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

"""Elastic stiffness handling for HREBSD-DIC.

One convention, stated in every docstring that touches a stiffness
(requirements D9.4): a **6 by 6 Voigt matrix in GPa, given in the
CRYSTAL frame, in the Voigt order (11, 22, 33, 23, 13, 12), with the
ENGINEERING shear convention in the Hooke product**::

    sigma = C @ [e11, e22, e33, 2*e23, 2*e13, 2*e12]

The engineering factor of two lives ONLY inside
:func:`hooke_product`.  Every strain this feature reports or consumes
elsewhere carries TENSOR shears (requirements D15.6), so a factor
dropped or doubled here is a factor-of-two error in every shear
stress, which :func:`hooke_product`'s own oracle pins.

No elastic-constant database exists anywhere in the dependency set
and none is added: the stiffness is supplied per phase by the user.
:func:`voigt_stiffness` is a convenience builder for the two
symmetries version one covers, and it is a builder only -- it holds
no material data.

The crystal to sample rotation is the explicit fourth-order rotation
of the ``(3, 3, 3, 3)`` tensor (requirements D9.5), which is
unambiguous, rather than a hand written Bond 6 by 6 matrix::

    C_sample_ijkl = g_ai g_bj g_ck g_dl C_crystal_abcd

with ``g`` the point's orientation matrix in the kikuchipy sense,
``v_crystal = g @ v_sample`` -- the matrix
:meth:`orix.quaternion.Rotation.to_matrix` returns and the one
``kikuchipy._utils.numba.rotate_vector`` applies (MEASURED, not
assumed: ``test_hrebsd_stiffness.py``,
``TestRotationDirection`` and its
``test_the_matrix_is_the_kikuchipy_orientation``).
Because ``g`` maps sample to crystal, the crystal to sample matrix is
``g^T``, and the transformation above is that matrix applied four
times.  The direction is transpose-SENSITIVE and is pinned by the
22.5 degree ``C16'`` sign oracle of requirements D9.5, not by the
45 degree textbook pins, which are even in the angle and survive a
transposition unchanged.

OpenXY's ``CalcF.m:324-405`` and EMsoftOO's ``StiffnessRotation``
(``mod_HREBSD.f90:3336-3396``) are equation-level cross-references
only; no code is ported from either.

References
----------
Ruggles et al., Ultramicroscopy 195 (2018) 85; Ruggles et al.,
Ultramicroscopy 210 (2020) 112927; Simmons and Wang, Single Crystal
Elastic Constants and Calculated Aggregate Properties (1971), the
source of the Ni and Ti constants the unit tests build with.
"""

import numpy as np

# The Voigt order of requirements D9.4, ``(11, 22, 33, 23, 13, 12)``,
# as zero-based tensor index pairs
VOIGT_INDICES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

# Second axis length of a Voigt matrix or vector
VOIGT_SIZE: int = 6

# The crystal symmetries :func:`voigt_stiffness` builds, frozen
SUPPORTED_SYMMETRIES: tuple[str, ...] = ("cubic", "hexagonal")

# The Voigt components which carry an engineering shear factor of two
# in the Hooke product, the last three of the frozen order
SHEAR_COMPONENTS: tuple[int, ...] = (3, 4, 5)


def voigt_stiffness(
    symmetry: str,
    *,
    c11: float,
    c12: float,
    c44: float,
    c13: float | None = None,
    c33: float | None = None,
) -> np.ndarray:
    """Return a 6 by 6 Voigt stiffness matrix of a crystal symmetry.

    The convenience builder of requirements D9.4.  It builds a matrix
    from constants the caller supplies and holds no material data of
    its own: kikuchipy ships no elastic-constant database and this
    function does not become one.

    Parameters
    ----------
    symmetry
        One of :data:`SUPPORTED_SYMMETRIES`. ``"cubic"`` consumes
        *c11*, *c12* and *c44*; ``"hexagonal"`` additionally consumes
        *c13* and *c33*.
    c11, c12, c44
        Elastic constants in GPa, required for every symmetry.
    c13, c33
        Elastic constants in GPa, required for ``"hexagonal"`` and
        refused for ``"cubic"``.

    Returns
    -------
    stiffness
        Array of shape ``(6, 6)`` and 64-bit float data type in GPa,
        in the CRYSTAL frame, in the Voigt order
        ``(11, 22, 33, 23, 13, 12)`` and in the engineering-shear
        convention of :func:`hooke_product`.

    Raises
    ------
    ValueError
        If *symmetry* is not one of :data:`SUPPORTED_SYMMETRIES`; if
        *c13* or *c33* is missing for ``"hexagonal"``; or if either
        is given for ``"cubic"``, where they are not independent
        constants and a value would be silently ignored.

    Notes
    -----
    Cubic: ``C11 = C22 = C33``, ``C12 = C13 = C23``,
    ``C44 = C55 = C66``.  Hexagonal (transversely isotropic about
    ``c``): ``C11 = C22``, ``C13 = C23``, ``C44 = C55`` and
    ``C66 = (C11 - C12)/2``, which is DERIVED and never taken as an
    input.  Everything else is zero in both.
    """
    if symmetry not in SUPPORTED_SYMMETRIES:
        raise ValueError(
            f"symmetry {symmetry!r} is not one of {list(SUPPORTED_SYMMETRIES)}; no "
            "elastic constant database ships with kikuchipy, so the stiffness of any "
            "other symmetry is supplied as a 6 by 6 Voigt matrix directly"
        )
    stiffness = np.zeros((VOIGT_SIZE, VOIGT_SIZE), dtype=np.float64)
    if symmetry == "cubic":
        given = [
            name for name, value in (("c13", c13), ("c33", c33)) if value is not None
        ]
        if given:
            raise ValueError(
                f"{given} is not an independent constant of a cubic crystal, where "
                "C13 = C12 and C33 = C11; leave it out rather than have it silently "
                "ignored"
            )
        stiffness[:3, :3] = c12
        stiffness[np.diag_indices(3)] = c11
        stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = c44
    else:
        missing = [
            name for name, value in (("c13", c13), ("c33", c33)) if value is None
        ]
        if missing:
            raise ValueError(
                f"a hexagonal stiffness needs {missing} in addition to c11, c12 and "
                "c44; C66 is DERIVED as (C11 - C12)/2 and is never an input"
            )
        stiffness[0, 0] = stiffness[1, 1] = c11
        stiffness[2, 2] = c33
        stiffness[0, 1] = stiffness[1, 0] = c12
        stiffness[0, 2] = stiffness[2, 0] = c13
        stiffness[1, 2] = stiffness[2, 1] = c13
        stiffness[3, 3] = stiffness[4, 4] = c44
        stiffness[5, 5] = 0.5 * (c11 - c12)
    return stiffness


def voigt_to_tensor(stiffness: np.ndarray) -> np.ndarray:
    """Return the fourth-order form of a Voigt stiffness matrix.

    Parameters
    ----------
    stiffness
        Voigt matrix of shape ``(6, 6)`` or ``(n, 6, 6)``.

    Returns
    -------
    tensor
        Array of shape ``(3, 3, 3, 3)`` or ``(n, 3, 3, 3, 3)`` and
        64-bit float data type, with the minor symmetries
        ``C_ijkl = C_jikl = C_ijlk`` filled in from the Voigt entries
        of :data:`VOIGT_INDICES`.

    Raises
    ------
    ValueError
        If *stiffness* is not a 6 by 6 matrix, or a stack of them.
    """
    matrix = np.asarray(stiffness, dtype=np.float64)
    if matrix.ndim not in (2, 3) or matrix.shape[-2:] != (VOIGT_SIZE, VOIGT_SIZE):
        raise ValueError(
            "stiffness must be a 6 by 6 Voigt matrix, or a stack of them of shape "
            f"(n, 6, 6), but has shape {matrix.shape}"
        )
    single = matrix.ndim == 2
    flat = matrix.reshape(-1, VOIGT_SIZE, VOIGT_SIZE)
    tensor = np.zeros((flat.shape[0], 3, 3, 3, 3), dtype=np.float64)
    for p, (i, j) in enumerate(VOIGT_INDICES):
        for q, (k, m) in enumerate(VOIGT_INDICES):
            value = flat[:, p, q]
            for a, b in ((i, j), (j, i)):
                for e, f in ((k, m), (m, k)):
                    tensor[:, a, b, e, f] = value
    return tensor[0] if single else tensor


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Return the Voigt form of a fourth-order stiffness tensor.

    The inverse of :func:`voigt_to_tensor` for any tensor with the
    minor symmetries, and its exact left inverse for every tensor
    that function returns.

    Parameters
    ----------
    tensor
        Array of shape ``(3, 3, 3, 3)`` or ``(n, 3, 3, 3, 3)``.

    Returns
    -------
    stiffness
        Voigt matrix of shape ``(6, 6)`` or ``(n, 6, 6)`` and 64-bit
        float data type, in the order of :data:`VOIGT_INDICES`.

    Raises
    ------
    ValueError
        If *tensor* does not have four trailing axes of length three.
    """
    array = np.asarray(tensor, dtype=np.float64)
    if array.ndim not in (4, 5) or array.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(
            "tensor must have four trailing axes of length three, that is shape "
            f"(3, 3, 3, 3) or (n, 3, 3, 3, 3), but has shape {array.shape}"
        )
    single = array.ndim == 4
    flat = array.reshape(-1, 3, 3, 3, 3)
    matrix = np.zeros((flat.shape[0], VOIGT_SIZE, VOIGT_SIZE), dtype=np.float64)
    for p, (i, j) in enumerate(VOIGT_INDICES):
        for q, (k, m) in enumerate(VOIGT_INDICES):
            matrix[:, p, q] = flat[:, i, j, k, m]
    return matrix[0] if single else matrix


def rotate_stiffness(
    stiffness: np.ndarray, orientation_matrix: np.ndarray
) -> np.ndarray:
    """Return a crystal-frame stiffness rotated into the sample
    frame.

    The explicit fourth-order rotation of requirements D9.5::

        C_sample_ijkl = g_ai g_bj g_ck g_dl C_crystal_abcd

    Parameters
    ----------
    stiffness
        Voigt matrix of shape ``(6, 6)`` in the CRYSTAL frame, in the
        convention of this module.
    orientation_matrix
        The point's orientation ``g`` of shape ``(3, 3)``, or one per
        point of shape ``(n, 3, 3)``, in the kikuchipy sense
        ``v_crystal = g @ v_sample``. This is what
        :meth:`orix.quaternion.Rotation.to_matrix` returns for the
        rotations of a
        :class:`~orix.crystal_map.CrystalMap`, so the per-point
        orientations of the INPUT map enter here unchanged.

    Returns
    -------
    rotated
        Voigt matrix of shape ``(6, 6)`` or ``(n, 6, 6)`` and 64-bit
        float data type, in the SAMPLE frame.

    Raises
    ------
    ValueError
        If *stiffness* is not a 6 by 6 matrix or
        *orientation_matrix* does not have two trailing axes of
        length three.

    Notes
    -----
    Transpose-sensitive, and deliberately so: rotating a cubic matrix
    by a symmetry operation and the 45 degree textbook forms are both
    EVEN in the rotation angle and pass with either convention, so the
    direction is pinned instead by the sign of ``C16'`` at 22.5
    degrees, which is odd in the angle (requirements D9.5)::

        C16'(theta) = (1/4) * sin(4*theta) * (C12 + 2*C44 - C11)

    for ``g`` a rotation of ``theta`` about the crystal ``z`` axis.
    """
    crystal = np.asarray(stiffness, dtype=np.float64)
    if crystal.shape != (VOIGT_SIZE, VOIGT_SIZE):
        raise ValueError(
            "stiffness must be a 6 by 6 Voigt matrix in the crystal frame, but has "
            f"shape {crystal.shape}"
        )
    matrices = np.asarray(orientation_matrix, dtype=np.float64)
    if matrices.ndim not in (2, 3) or matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "orientation_matrix must have two trailing axes of length three, that is "
            f"shape (3, 3) or (n, 3, 3), but has shape {matrices.shape}"
        )
    single = matrices.ndim == 2
    tensor = voigt_to_tensor(crystal)
    # one point at a time, so that a stacked call is the loop over the
    # single-matrix one to the LAST BIT and not merely to a tolerance
    rotated = np.stack(
        [
            np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, tensor)
            for g in matrices.reshape(-1, 3, 3)
        ]
    )
    sample = tensor_to_voigt(rotated)
    return sample[0] if single else sample


def hooke_product(stiffness: np.ndarray, strain: np.ndarray) -> np.ndarray:
    """Return the stress of a strain, in the frame both are given in.

    THE one place the engineering shear factor of requirements D9.4
    lives::

        sigma = C @ [e11, e22, e33, 2*e23, 2*e13, 2*e12]

    Parameters
    ----------
    stiffness
        Voigt matrix of shape ``(6, 6)`` or ``(n, 6, 6)`` in GPa, in
        the SAME frame as *strain*.
    strain
        Strain of shape ``(6,)`` or ``(n, 6)`` in the Voigt order
        ``(11, 22, 33, 23, 13, 12)`` with TENSOR shears, which is the
        convention of the ``strain`` property of requirements D15.6.
        The engineering factor is applied here and nowhere else.

    Returns
    -------
    stress
        Array of shape ``(6,)`` and 64-bit float data type in GPa, in
        the Voigt order and in the frame of the inputs, when BOTH
        arguments are single; ``(n, 6)`` when either is a stack. One
        strain against a stack of stiffnesses is therefore the stress
        that strain costs at every one of them, which is what asking
        the question means (corrected 2026-09-08, Stage B adversarial
        review: the broadcast was built and then all but its first
        row silently discarded).

    Raises
    ------
    ValueError
        If the trailing axis of *strain* is not
        :data:`VOIGT_SIZE`, if *stiffness* is not 6 by 6 in its two
        trailing axes, or if the two stacks have different lengths.
    """
    matrix = np.asarray(stiffness, dtype=np.float64)
    if matrix.ndim not in (2, 3) or matrix.shape[-2:] != (VOIGT_SIZE, VOIGT_SIZE):
        raise ValueError(
            "stiffness must be a 6 by 6 Voigt matrix, or a stack of them of shape "
            f"(n, 6, 6), but has shape {matrix.shape}"
        )
    vector = np.asarray(strain, dtype=np.float64)
    if vector.ndim not in (1, 2) or vector.shape[-1] != VOIGT_SIZE:
        raise ValueError(
            "strain must be a Voigt vector of shape (6,), or a stack of them of shape "
            f"(n, 6), but has shape {vector.shape}"
        )
    # A single result needs BOTH sides single: a stack of stiffnesses
    # against one strain is n stresses, not the first of them
    single = vector.ndim == 1 and matrix.ndim == 2
    # THE engineering shear convention of requirements D9.4, applied
    # here and in no other place in this feature
    engineering = np.atleast_2d(vector).copy()
    engineering[:, list(SHEAR_COMPONENTS)] *= 2.0
    if matrix.ndim == 2:
        stress = engineering @ matrix.T
    else:
        if engineering.shape[0] == 1 and matrix.shape[0] != 1:
            engineering = np.tile(engineering, (matrix.shape[0], 1))
        if matrix.shape[0] != engineering.shape[0]:
            raise ValueError(
                f"a stack of {matrix.shape[0]} stiffnesses cannot be paired with "
                f"{engineering.shape[0]} strains: the two stacks are indexed point by "
                "point and must have the same length"
            )
        stress = np.einsum("nij,nj->ni", matrix, engineering)
    return stress[0] if single else stress
