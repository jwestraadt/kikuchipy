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
    raise NotImplementedError(
        "voigt_stiffness arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


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
    raise NotImplementedError(
        "voigt_to_tensor arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


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
    raise NotImplementedError(
        "tensor_to_voigt arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


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
    raise NotImplementedError(
        "rotate_stiffness arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


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
        Array of shape ``(6,)`` or ``(n, 6)`` and 64-bit float data
        type in GPa, in the Voigt order, in the frame of the inputs.

    Raises
    ------
    ValueError
        If the trailing axis of *strain* is not
        :data:`VOIGT_SIZE`, if *stiffness* is not 6 by 6 in its two
        trailing axes, or if the two stacks have different lengths.
    """
    raise NotImplementedError(
        "hooke_product arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )
