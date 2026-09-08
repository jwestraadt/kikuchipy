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

"""The HREBSD-DIC tensor chain: frames, closure, strain and stress.

ONE private chain, :func:`tensor_chain`, carries every tensor result
of this feature (plan section 3.2), and both
:func:`hrebsd_strain_stress` and the scalar GND densities of Stage C
call it rather than repeating it:

1. The stored ``Fe`` property is the REDUCED tensor of the DETECTOR
   frame and is ALREADY corrected for the beam-scan projection centre
   shift by the engine (requirements D6.2, D15.6).  **It is never
   corrected again here**: convert-then-correct is the exact
   confusion requirements D6.2 refuses, and a second removal of the
   phantom shows up as a spurious isotropic strain of order
   ``alpha_s - 1``, which the validation V6 phantom test of
   ``test_hrebsd_pc_shift.py`` kills.
2. ``beta_det = Fe - I`` is rotated to the SAMPLE frame,
   ``beta_s = M^T beta_det M`` (requirements D7).
3. The ninth degree of freedom, which the projection cannot see, is
   closed in the sample frame: traction-free ``sigma33 = 0`` through
   the linear 3 by 3 solve of requirements D9.2 when a stiffness is
   given, deviatoric ``tr(beta) = 0`` otherwise (D9.1, D9.3).
4. ``F_s = I + beta_closed`` is split by the right polar
   decomposition ``F_s = R_hr U`` (requirements D8): the strain is
   Biot, ``e = U - I``, by default and Green-Lagrange
   ``(U^T U - I)/2`` on request, and the rotation is reported as the
   rotation vector ``omega = axis * angle`` in radians.
5. The stress is ``sigma_s = C_s : e`` in GPa with the crystal to
   sample rotated stiffness, and the derived von Mises, hydrostatic
   and principal stresses follow (requirements D10).

**The detector frame the ``Fe`` property lives in, made explicit.**
Requirements D7 writes the rotation as ``R^T beta_det R`` with
``R = detector.sample_to_detector.to_matrix()``.  That matrix maps
the sample frame into kikuchipy's gnomonic detector frame, whose
``y`` points UP, while the engine's own frame is the numpy array
frame of requirements D1.1, whose ``y`` points DOWN: the two differ
by :data:`DETECTOR_Y_FLIP`, exactly as the Stage A half-pixel
measurement recorded for the coordinates
(``test_hrebsd_engine.py``,
``TestPcCentredFrame`` and its
``test_pc_centred_frame_matches_kikuchipy_geometry``,
1.8e-14 px over a whole detector).  The matrix used here is therefore
``M = DETECTOR_Y_FLIP @ R``, which :func:`sample_to_detector_matrix`
returns and which
``test_hrebsd_tensors.py::TestFrameChain`` MEASURES from the detector
geometry itself rather than assuming.  Dropping the flip mirrors
``beta13``, ``beta23``, ``beta31`` and ``beta32`` and flips two
components of every reported rotation vector, so it is not cosmetic;
requirements D7 gains this as a dated correction at the Stage B
implementation gate, the D1.1 precedent.

References
----------
Ernould et al., AIEP 223 (2022) Ch. 2; Ruggles et al.,
Ultramicroscopy 195 (2018) 85; Ruggles et al., Ultramicroscopy 210
(2020) 112927; Wilkinson, Meaden and Dingley, Ultramicroscopy 106
(2006) 307.  OpenXY's ``CalcF.m:406-542`` is the equation-level
cross-reference for the traction-free solve; no code is ported.
"""

import numpy as np

from kikuchipy.indexing._hrebsd._stiffness import (
    VOIGT_INDICES,
    VOIGT_SIZE,
    hooke_product,
    rotate_stiffness,
)

# The Stage B crystal map properties of requirements D15.6, in the
# order :func:`hrebsd_strain_stress` adds them
STAGE_B_PROP_NAMES: tuple[str, ...] = (
    "strain",
    "rotation_vector",
    "beta",
    "stress",
    "stress_von_mises",
    "stress_hydrostatic",
    "stress_principal",
)

# Second axis lengths of the two dimensional Stage B properties
STRAIN_PROP_SIZE: int = VOIGT_SIZE
STRESS_PROP_SIZE: int = VOIGT_SIZE
BETA_PROP_SIZE: int = 9
ROTATION_VECTOR_PROP_SIZE: int = 3
PRINCIPAL_PROP_SIZE: int = 3

# The ninth degree of freedom closures of requirements D9.1
SUPPORTED_CLOSURES: tuple[str, ...] = ("auto", "traction_free", "deviatoric")

# The strain measures of requirements D8, Biot by default
SUPPORTED_STRAIN_MEASURES: tuple[str, ...] = ("biot", "green-lagrange")

# The spec's detector frame is the numpy array frame, x right and y
# DOWN, while kikuchipy's gnomonic detector frame has y UP, so the
# matrix of ``EBSDDetector.sample_to_detector`` reaches this frame
# through this flip.  MEASURED at the Stage A gate over a whole
# detector (see the module docstring); the same constant appears in
# ``test_hrebsd_engine.py``, where it was measured
DETECTOR_Y_FLIP: np.ndarray = np.diag([1.0, -1.0, 1.0])


def sample_to_detector_matrix(detector) -> np.ndarray:
    """Return the matrix mapping sample vectors to the spec's
    detector frame.

    Parameters
    ----------
    detector
        :class:`~kikuchipy.detectors.EBSDDetector` whose sample tilt,
        tilt, azimuthal and twist angles build
        :attr:`~kikuchipy.detectors.EBSDDetector.sample_to_detector`.
        Nothing is hardcoded here: EMsoftOO's 70 degree literal
        (``mod_HREBSDDIC.f90:925``) is a recorded deviation.

    Returns
    -------
    matrix
        Array of shape ``(3, 3)`` and 64-bit float data type, the
        matrix ``M`` with ``v_detector = M @ v_sample`` in the
        engine's own y-down detector frame, that is
        :data:`DETECTOR_Y_FLIP` composed with the kikuchipy rotation
        (requirements D7, corrected; see the module docstring).
    """
    rotation = np.asarray(
        detector.sample_to_detector.to_matrix(), dtype=np.float64
    ).squeeze()
    return DETECTOR_Y_FLIP @ rotation


def _as_stack(tensor: np.ndarray, name: str, size: int = 3) -> tuple[np.ndarray, bool]:
    """Return a ``(n, size, size)`` view of a tensor and whether it
    arrived unstacked.

    Parameters
    ----------
    tensor
        Array of shape ``(size, size)`` or ``(n, size, size)``.
    name
        Argument name, so that the guard below names what the caller
        passed rather than a local.
    size
        Length the two trailing axes must have. Default is 3.

    Returns
    -------
    stacked
        Array of shape ``(n, size, size)`` and 64-bit float data type.
    single
        Whether *tensor* had exactly two axes, in which case the
        caller returns ``stacked[0]``.

    Raises
    ------
    ValueError
        If the two trailing axes do not both have length *size*.
    """
    array = np.asarray(tensor, dtype=np.float64)
    if array.ndim not in (2, 3) or array.shape[-2:] != (size, size):
        raise ValueError(
            f"{name} must have two trailing axes of length {size}, that is shape "
            f"({size}, {size}) or (n, {size}, {size}), but has shape {array.shape}"
        )
    return array.reshape(-1, size, size), array.ndim == 2


def _as_voigt_stack(vector: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Return a ``(n, 6)`` view of a Voigt vector and whether it
    arrived unstacked.

    Parameters
    ----------
    vector
        Array of shape ``(6,)`` or ``(n, 6)``.
    name
        Argument name for the guard message.

    Returns
    -------
    stacked
        Array of shape ``(n, 6)`` and 64-bit float data type.
    single
        Whether *vector* had exactly one axis.

    Raises
    ------
    ValueError
        If the trailing axis of *vector* is not :data:`VOIGT_SIZE`.
    """
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[-1] != VOIGT_SIZE:
        raise ValueError(
            f"{name} must be a Voigt vector of shape ({VOIGT_SIZE},), or a stack of "
            f"them of shape (n, {VOIGT_SIZE}), but has shape {array.shape}"
        )
    return np.atleast_2d(array), array.ndim == 1


def _voigt_to_matrix(stress: np.ndarray) -> np.ndarray:
    """Return the ``(n, 3, 3)`` symmetric tensors of ``(n, 6)`` Voigt
    vectors.

    Parameters
    ----------
    stress
        Array of shape ``(n, 6)`` in the order of
        :data:`~kikuchipy.indexing._hrebsd._stiffness.VOIGT_INDICES`.

    Returns
    -------
    matrix
        Array of shape ``(n, 3, 3)`` and 64-bit float data type.
    """
    matrix = np.zeros((stress.shape[0], 3, 3), dtype=np.float64)
    for p, (i, j) in enumerate(VOIGT_INDICES):
        matrix[:, i, j] = stress[:, p]
        matrix[:, j, i] = stress[:, p]
    return matrix


def beta_to_sample_frame(beta_detector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return a detector-frame tensor rotated into the sample frame.

    ``beta_s = M^T beta_det M`` (requirements D7).

    Parameters
    ----------
    beta_detector
        Array of shape ``(3, 3)`` or ``(n, 3, 3)`` in the detector
        frame.
    matrix
        The ``(3, 3)`` matrix ``M`` of
        :func:`sample_to_detector_matrix`.

    Returns
    -------
    beta_sample
        Array of the same shape and 64-bit float data type, in the
        sample frame.

    Raises
    ------
    ValueError
        If either argument does not have two trailing axes of length
        three.
    """
    stacked, single = _as_stack(beta_detector, "beta_detector")
    frame = np.asarray(matrix, dtype=np.float64)
    if frame.shape != (3, 3):
        raise ValueError(
            "matrix must be the (3, 3) frame matrix of sample_to_detector_matrix, but "
            f"has shape {frame.shape}"
        )
    sample = frame.T @ stacked @ frame
    return sample[0] if single else sample


def close_beta(
    beta_sample: np.ndarray,
    *,
    stiffness_sample: np.ndarray | None = None,
    closure: str = "auto",
) -> np.ndarray:
    """Return the measured tensor with its ninth degree of freedom
    closed.

    The projection loses the isotropic "radial" degree of freedom:
    measurement determines every off-diagonal entry plus the two
    DIFFERENCES ``b7 = beta11 - beta33`` and ``b8 = beta22 - beta33``
    (requirements D9).  Closing therefore means adding one unknown
    multiple of the identity, ``beta_closed = beta_sample + t * I``,
    and the two closures differ only in how ``t`` is fixed:

    - ``"traction_free"``, the OpenXY post-hoc linear solve of
      requirements D9.2, solves the 3 by 3 system::

          [C3311 C3322 C3333][e11]   [-C3312*(b12+b21)
          [  1     0    -1  ][e22] =   -C3313*(b13+b31)
          [  0     1    -1  ][e33]     -C3323*(b23+b32)]
                                     [ b7 ]
                                     [ b8 ]

      for the diagonal strains and sets ``t = e33 - beta_sample33``.
      EMsoftOO's in-fit NLopt constraint variant
      (``mod_HREBSD.f90:3118-3332``) is NOT used: the post-hoc solve
      is exact for the linearized problem and testable in isolation.
    - ``"deviatoric"`` sets ``t = -tr(beta_sample)/3``, that is
      ``tr(beta_closed) = 0`` (requirements D9.3, Ruggles 2018's own
      choice for the IC-GN benchmark).

    Parameters
    ----------
    beta_sample
        Measured tensor of shape ``(3, 3)`` or ``(n, 3, 3)`` in the
        sample frame. Only its off-diagonals and the two diagonal
        differences are read, so any isotropic part it carries is
        replaced rather than trusted.
    stiffness_sample
        Voigt stiffness of shape ``(6, 6)`` or ``(n, 6, 6)`` in GPa,
        already rotated into the SAMPLE frame, or ``None``.
    closure
        One of :data:`SUPPORTED_CLOSURES`. ``"auto"`` is traction
        free when *stiffness_sample* is given and deviatoric
        otherwise (requirements D9.1). Default is ``"auto"``.

    Returns
    -------
    beta_closed
        Array of the shape of *beta_sample* and 64-bit float data
        type.

    Raises
    ------
    ValueError
        If *closure* is not one of :data:`SUPPORTED_CLOSURES`; or if
        it is ``"traction_free"`` without a *stiffness_sample*, which
        names the missing argument rather than falling back silently.

    Notes
    -----
    The system's second and third rows are the definitions of ``b7``
    and ``b8``, so the closed diagonal reproduces them exactly and
    the solve is a one-unknown problem written as three: swapping the
    rows or misassigning ``b7`` and ``b8`` changes the answer, which
    is what ``test_hrebsd_tensors.py::TestClosure`` pins against an
    independently built tensor.
    """
    resolved = _resolve_closure(closure, stiffness_sample)
    stacked, single = _as_stack(beta_sample, "beta_sample")
    if resolved == "deviatoric":
        offset = -np.trace(stacked, axis1=1, axis2=2) / 3.0
    else:
        offset = _traction_free_offset(stacked, stiffness_sample)
    closed = stacked + offset[:, None, None] * np.eye(3)
    return closed[0] if single else closed


def _resolve_closure(closure: str, stiffness_sample: np.ndarray | None) -> str:
    """Return which closure of requirements D9.1 a request resolves
    to.

    Parameters
    ----------
    closure
        One of :data:`SUPPORTED_CLOSURES`.
    stiffness_sample
        The sample-frame stiffness, or ``None``.

    Returns
    -------
    resolved
        Either ``"traction_free"`` or ``"deviatoric"``.

    Raises
    ------
    ValueError
        If *closure* is unknown, or if it is ``"traction_free"``
        without a stiffness, which the message names rather than
        falling back silently.
    """
    if closure not in SUPPORTED_CLOSURES:
        raise ValueError(
            f"closure {closure!r} is not one of {list(SUPPORTED_CLOSURES)}"
        )
    if closure == "auto":
        return "deviatoric" if stiffness_sample is None else "traction_free"
    if closure == "traction_free" and stiffness_sample is None:
        raise ValueError(
            "closure='traction_free' imposes sigma33 = 0 and so needs a stiffness; "
            "pass one, or ask for closure='deviatoric' explicitly"
        )
    return closure


def _traction_free_offset(
    beta_sample: np.ndarray, stiffness_sample: np.ndarray
) -> np.ndarray:
    """Return the isotropic offset the traction-free closure adds.

    The three by three solve of requirements D9.2, per point.

    Parameters
    ----------
    beta_sample
        Measured tensors of shape ``(n, 3, 3)`` in the sample frame.
    stiffness_sample
        Voigt stiffness of shape ``(6, 6)`` or ``(n, 6, 6)`` in GPa,
        already rotated into the sample frame.

    Returns
    -------
    offset
        Array of shape ``(n,)``, the ``t`` of
        ``beta_closed = beta_sample + t * I``.

    Raises
    ------
    ValueError
        If *stiffness_sample* is not a 6 by 6 matrix or a stack of
        them of the length of *beta_sample*.
    """
    stiffness, _ = _as_stack(stiffness_sample, "stiffness_sample", size=VOIGT_SIZE)
    size = beta_sample.shape[0]
    # one stiffness for the whole map, or one per point, and nothing
    # in between: numpy refuses any other pairing here
    stiffness = np.broadcast_to(stiffness, (size, VOIGT_SIZE, VOIGT_SIZE))
    # The third Voigt row of the sample-frame stiffness, that is
    # C3311, C3322, C3333 and the three shear terms C3323, C3313,
    # C3312 (requirements D9.2).  The shear terms are identically zero
    # for an UNROTATED cubic matrix, which is why the pins of this
    # closure are all built at a generic orientation
    third = stiffness[:, 2, :]
    system = np.zeros((size, 3, 3), dtype=np.float64)
    system[:, 0, :] = third[:, :3]
    system[:, 1, 0] = 1.0
    system[:, 1, 2] = -1.0
    system[:, 2, 1] = 1.0
    system[:, 2, 2] = -1.0
    right = np.zeros((size, 3), dtype=np.float64)
    right[:, 0] = -(
        third[:, 5] * (beta_sample[:, 0, 1] + beta_sample[:, 1, 0])
        + third[:, 4] * (beta_sample[:, 0, 2] + beta_sample[:, 2, 0])
        + third[:, 3] * (beta_sample[:, 1, 2] + beta_sample[:, 2, 1])
    )
    right[:, 1] = beta_sample[:, 0, 0] - beta_sample[:, 2, 2]
    right[:, 2] = beta_sample[:, 1, 1] - beta_sample[:, 2, 2]
    # the right-hand side is passed as a stack of COLUMNS, which is
    # the one reading numpy 1 and numpy 2 agree on
    diagonal_strain = np.linalg.solve(system, right[:, :, None])[:, :, 0]
    return diagonal_strain[:, 2] - beta_sample[:, 2, 2]


def polar_decomposition(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the RIGHT polar decomposition ``F = R U``.

    Parameters
    ----------
    f
        Deformation gradient of shape ``(3, 3)`` or ``(n, 3, 3)``.

    Returns
    -------
    rotation
        The orthogonal factor ``R`` of the same shape and 64-bit
        float data type.
    stretch
        The symmetric positive definite RIGHT stretch ``U``, of the
        same shape. The LEFT stretch ``V = R U R^T`` is a different
        tensor whenever the rotation and the stretch do not commute,
        and requirements D8 reports the right one: a Biot strain
        built from ``V`` is the ``V - I`` mutant of plan section 3.4.

    Raises
    ------
    ValueError
        If *f* does not have two trailing axes of length three.

    Notes
    -----
    Computed from the singular value decomposition per point, which
    is closed form for a 3 by 3 matrix and vectorizes over the map.
    """
    stacked, single = _as_stack(f, "f")
    left, values, right = np.linalg.svd(stacked)
    rotation = left @ right
    stretch = np.swapaxes(right, 1, 2) @ (values[:, :, None] * right)
    if single:
        return rotation[0], stretch[0]
    return rotation, stretch


def strain_from_stretch(
    stretch: np.ndarray, strain_measure: str = "biot"
) -> np.ndarray:
    """Return the strain tensor of a right stretch.

    Parameters
    ----------
    stretch
        Right stretch ``U`` of shape ``(3, 3)`` or ``(n, 3, 3)``.
    strain_measure
        One of :data:`SUPPORTED_STRAIN_MEASURES`. ``"biot"``
        (default, and OpenXY compatible) is ``U - I``;
        ``"green-lagrange"`` is ``(U^T U - I)/2``.

    Returns
    -------
    strain
        Array of the shape of *stretch* and 64-bit float data type,
        with TENSOR shears.

    Raises
    ------
    ValueError
        If *strain_measure* is not one of
        :data:`SUPPORTED_STRAIN_MEASURES`.
    """
    if strain_measure not in SUPPORTED_STRAIN_MEASURES:
        raise ValueError(
            f"strain_measure {strain_measure!r} is not one of "
            f"{list(SUPPORTED_STRAIN_MEASURES)}"
        )
    stacked, single = _as_stack(stretch, "stretch")
    if strain_measure == "biot":
        strain = stacked - np.eye(3)
    else:
        strain = 0.5 * (np.swapaxes(stacked, 1, 2) @ stacked - np.eye(3))
    return strain[0] if single else strain


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the rotation vector ``axis * angle`` of a rotation
    matrix.

    Parameters
    ----------
    rotation
        Orthogonal matrix of shape ``(3, 3)`` or ``(n, 3, 3)``, the
        ``R`` of :func:`polar_decomposition`.

    Returns
    -------
    rotation_vector
        Array of shape ``(3,)`` or ``(n, 3)`` and 64-bit float data
        type, in RADIANS, in the frame *rotation* is given in. At
        small angles its entries are the small-rotation components
        ``(w32, w13, w21)`` of the skew part, which is why they are
        not stored separately (requirements D8, D15.6).

    Raises
    ------
    ValueError
        If *rotation* does not have two trailing axes of length
        three.
    """
    stacked, single = _as_stack(rotation, "rotation")
    skew = 0.5 * (stacked - np.swapaxes(stacked, 1, 2))
    dual = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=-1)
    sine = np.linalg.norm(dual, axis=-1)
    cosine = 0.5 * (np.trace(stacked, axis1=1, axis2=2) - 1.0)
    # ``arctan2`` and not ``arccos``: the square-root singularity of
    # ``arccos`` at the identity costs about eight significant digits
    # at the 1e-4 rad scale this field lives at, which requirements
    # D12 records for the same reason
    angle = np.arctan2(sine, cosine)
    # the limit of ``angle / sin(angle)`` at the identity is one, and
    # it is taken literally where the dual vector vanishes exactly
    scale = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 0.0)
    vector = dual * scale[:, None]
    return vector[0] if single else vector


def small_strain_split(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the small-strain split ``sym(beta)`` and ``skew(beta)``.

    The private fast path of requirements D8.  It is NOT enabled for
    any public result: the Stage B gate first has to measure it
    identical to the polar path within 1e-6 strain over the
    validation V4 rotation sweep, and the enabling angle and band are
    recorded in requirements D8 with a date before anything switches
    to it.  Until then every public result goes through
    :func:`polar_decomposition`.

    Parameters
    ----------
    beta
        Displacement gradient of shape ``(3, 3)`` or ``(n, 3, 3)``.

    Returns
    -------
    strain
        The symmetric part, of the same shape.
    rotation
        The antisymmetric part, of the same shape. Its dual vector
        ``(w32, w13, w21)`` is the small-angle limit of
        :func:`rotation_vector_from_matrix`.

    Raises
    ------
    ValueError
        If *beta* does not have two trailing axes of length three.

    Notes
    -----
    The error against the polar path is second order in the rotation,
    of order 3e-4 strain at one degree, which is why the switch is a
    measurement and not a preference.
    """
    stacked, single = _as_stack(beta, "beta")
    transposed = np.swapaxes(stacked, 1, 2)
    strain = 0.5 * (stacked + transposed)
    rotation = 0.5 * (stacked - transposed)
    if single:
        return strain[0], rotation[0]
    return strain, rotation


def tensor_to_voigt_vector(tensor: np.ndarray) -> np.ndarray:
    """Return the six Voigt components of a symmetric tensor.

    Parameters
    ----------
    tensor
        Array of shape ``(3, 3)`` or ``(n, 3, 3)``. Its symmetric
        part is taken, so a tensor which is symmetric only to
        rounding is not rejected.

    Returns
    -------
    voigt
        Array of shape ``(6,)`` or ``(n, 6)`` and 64-bit float data
        type in the order ``(11, 22, 33, 23, 13, 12)`` of
        :data:`~kikuchipy.indexing._hrebsd._stiffness.VOIGT_INDICES`,
        with TENSOR shears: no factor of two is applied anywhere but
        inside
        :func:`~kikuchipy.indexing._hrebsd._stiffness.hooke_product`.

    Raises
    ------
    ValueError
        If *tensor* does not have two trailing axes of length three.
    """
    stacked, single = _as_stack(tensor, "tensor")
    symmetric = 0.5 * (stacked + np.swapaxes(stacked, 1, 2))
    voigt = np.stack([symmetric[:, i, j] for (i, j) in VOIGT_INDICES], axis=-1)
    return voigt[0] if single else voigt


def von_mises_stress(stress: np.ndarray) -> np.ndarray:
    """Return the von Mises equivalent stress.

    ``sqrt(3/2 * s:s)`` with ``s = sigma - tr(sigma)/3 * I``
    (requirements D10).

    Parameters
    ----------
    stress
        Array of shape ``(6,)`` or ``(n, 6)`` in the Voigt order, in
        GPa.

    Returns
    -------
    von_mises
        Scalar or array of shape ``(n,)`` and 64-bit float data type
        in GPa.

    Raises
    ------
    ValueError
        If the trailing axis of *stress* is not
        :data:`~kikuchipy.indexing._hrebsd._stiffness.VOIGT_SIZE`.
    """
    stacked, single = _as_voigt_stack(stress, "stress")
    deviatoric = stacked.copy()
    deviatoric[:, :3] -= stacked[:, :3].sum(axis=-1)[:, None] / 3.0
    # the last three Voigt slots are off-diagonal tensor entries, each
    # of which appears twice in the double contraction
    contraction = (deviatoric[:, :3] ** 2).sum(axis=-1) + 2.0 * (
        deviatoric[:, 3:] ** 2
    ).sum(axis=-1)
    von_mises = np.sqrt(1.5 * contraction)
    return von_mises[0] if single else von_mises


def hydrostatic_stress(stress: np.ndarray) -> np.ndarray:
    """Return the hydrostatic stress ``tr(sigma)/3``.

    Parameters
    ----------
    stress
        Array of shape ``(6,)`` or ``(n, 6)`` in the Voigt order, in
        GPa.

    Returns
    -------
    hydrostatic
        Scalar or array of shape ``(n,)`` and 64-bit float data type
        in GPa. It is CLOSURE DERIVED, which the public docstrings
        state: the isotropic part of the tensor is what the ninth
        degree of freedom closure supplies, never what the projection
        measures (requirements D10).

    Raises
    ------
    ValueError
        If the trailing axis of *stress* is not
        :data:`~kikuchipy.indexing._hrebsd._stiffness.VOIGT_SIZE`.
    """
    stacked, single = _as_voigt_stack(stress, "stress")
    hydrostatic = stacked[:, :3].sum(axis=-1) / 3.0
    return hydrostatic[0] if single else hydrostatic


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    """Return the three principal stresses in DESCENDING order.

    Parameters
    ----------
    stress
        Array of shape ``(6,)`` or ``(n, 6)`` in the Voigt order, in
        GPa.

    Returns
    -------
    principal
        Array of shape ``(3,)`` or ``(n, 3)`` and 64-bit float data
        type in GPa, sorted DESCENDING (requirements D10), which is
        the opposite of what :func:`numpy.linalg.eigvalsh` returns
        and is the ascending mutant of plan section 3.4.

    Raises
    ------
    ValueError
        If the trailing axis of *stress* is not
        :data:`~kikuchipy.indexing._hrebsd._stiffness.VOIGT_SIZE`.
    """
    stacked, single = _as_voigt_stack(stress, "stress")
    # ``eigvalsh`` returns ASCENDING eigenvalues, which is the plan
    # section 3.4 mutant: requirements D10 reports them descending
    principal = np.linalg.eigvalsh(_voigt_to_matrix(stacked))[:, ::-1]
    return principal[0] if single else principal


def tensor_chain(
    fe: np.ndarray,
    orientation_matrices: np.ndarray,
    detector,
    *,
    stiffness: np.ndarray | None = None,
    closure: str = "auto",
    strain_measure: str = "biot",
) -> dict:
    """Return every Stage B tensor quantity of a map.

    THE one chain of plan section 3.2, shared by
    :func:`hrebsd_strain_stress` and by the scalar GND densities of
    Stage C, so that the frame, the closure and the split have a
    single definition.  The five steps are listed in the module
    docstring.

    Parameters
    ----------
    fe
        Reduced DETECTOR-frame deformation gradients of shape
        ``(n, 9)`` row-major or ``(n, 3, 3)``, the ``Fe`` property of
        requirements D15.6, ALREADY corrected for the beam-scan
        projection centre shift by the engine. Rows holding NaN, the
        non-converged and masked points of requirements D2.6, give
        NaN in every returned quantity.
    orientation_matrices
        Per-point orientation matrices of shape ``(n, 3, 3)`` in the
        kikuchipy sense ``v_crystal = g @ v_sample``, used ONLY to
        rotate the stiffness into the sample frame (requirements D7:
        orientations are never modified by this feature).
    detector
        :class:`~kikuchipy.detectors.EBSDDetector` supplying the
        frame chain through :func:`sample_to_detector_matrix`.
    stiffness
        Voigt stiffness of shape ``(6, 6)`` in GPa in the CRYSTAL
        frame, in the convention of
        :mod:`~kikuchipy.indexing._hrebsd._stiffness`, or ``None``
        for the deviatoric closure and no stress.
    closure
        One of :data:`SUPPORTED_CLOSURES`. Default is ``"auto"``.
    strain_measure
        One of :data:`SUPPORTED_STRAIN_MEASURES`. Default is
        ``"biot"``.

    Returns
    -------
    properties
        Dictionary keyed by :data:`STAGE_B_PROP_NAMES`, every value
        an array whose first axis is ``n`` and which holds NaN where
        the input does. The four stress entries are all NaN when
        *stiffness* is not given, so that the property set does not
        depend on the arguments.

    Raises
    ------
    ValueError
        If *fe* is not ``(n, 9)`` or ``(n, 3, 3)``; if
        *orientation_matrices* is not ``(n, 3, 3)`` of the same
        length; if *stiffness* is not a 6 by 6 matrix; or for any
        invalid *closure* or *strain_measure*, each naming the
        offending argument.

    Notes
    -----
    The beam-scan phantom is NOT removed here.  The stored ``Fe`` is
    corrected once, in the engine, before the conversion
    (requirements D6.2); removing it again is the plan section 3.4
    mutant that ``test_hrebsd_pc_shift.py``'s phantom-through-the-
    chain test kills.
    """
    gradients = _fe_as_matrices(fe)
    size = gradients.shape[0]
    matrices = np.asarray(orientation_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape != (size, 3, 3):
        raise ValueError(
            "orientation_matrices must have shape (n, 3, 3) with n the length of fe, "
            f"that is ({size}, 3, 3), but has shape {matrices.shape}"
        )
    if stiffness is not None:
        stiffness = np.asarray(stiffness, dtype=np.float64)
        if stiffness.shape != (VOIGT_SIZE, VOIGT_SIZE):
            raise ValueError(
                "stiffness must be a 6 by 6 Voigt matrix in GPa in the crystal frame, "
                f"but has shape {stiffness.shape}"
            )
    _resolve_closure(closure, stiffness)
    if strain_measure not in SUPPORTED_STRAIN_MEASURES:
        raise ValueError(
            f"strain_measure {strain_measure!r} is not one of "
            f"{list(SUPPORTED_STRAIN_MEASURES)}"
        )

    properties = {
        "strain": np.full((size, STRAIN_PROP_SIZE), np.nan),
        "rotation_vector": np.full((size, ROTATION_VECTOR_PROP_SIZE), np.nan),
        "beta": np.full((size, BETA_PROP_SIZE), np.nan),
        "stress": np.full((size, STRESS_PROP_SIZE), np.nan),
        "stress_von_mises": np.full(size, np.nan),
        "stress_hydrostatic": np.full(size, np.nan),
        "stress_principal": np.full((size, PRINCIPAL_PROP_SIZE), np.nan),
    }
    # requirements D2.6: a point which did not converge, failed or was
    # masked out keeps NaN in every derived property and is NEVER
    # zeroed, so it is taken out before the linear algebra rather than
    # allowed to poison it
    good = np.isfinite(gradients).all(axis=(1, 2))
    if not good.any():
        return properties
    gradients = gradients[good]
    matrices = matrices[good]

    # 1-2: the stored Fe is already D6.2 corrected, so the phantom is
    # NOT removed again here; only the frame changes
    beta_detector = gradients - np.eye(3)
    beta_sample = beta_to_sample_frame(
        beta_detector, sample_to_detector_matrix(detector)
    )
    # 3: the ninth degree of freedom, closed in the sample frame
    stiffness_sample = None
    if stiffness is not None:
        stiffness_sample = rotate_stiffness(stiffness, matrices)
    closed = close_beta(beta_sample, stiffness_sample=stiffness_sample, closure=closure)
    # 4: the split, through the polar decomposition and never through
    # the small-strain fast path, which requirements D8 leaves private
    rotation, stretch = polar_decomposition(np.eye(3) + closed)
    strain = tensor_to_voigt_vector(strain_from_stretch(stretch, strain_measure))
    properties["strain"][good] = strain
    properties["rotation_vector"][good] = rotation_vector_from_matrix(rotation)
    properties["beta"][good] = closed.reshape(-1, BETA_PROP_SIZE)
    # 5: the stress, in the GPa the stiffness is given in
    if stiffness_sample is not None:
        stress = hooke_product(stiffness_sample, strain)
        properties["stress"][good] = stress
        properties["stress_von_mises"][good] = von_mises_stress(stress)
        properties["stress_hydrostatic"][good] = hydrostatic_stress(stress)
        properties["stress_principal"][good] = principal_stresses(stress)
    return properties


def _fe_as_matrices(fe: np.ndarray) -> np.ndarray:
    """Return the stored ``Fe`` property as ``(n, 3, 3)`` matrices.

    Parameters
    ----------
    fe
        Array of shape ``(n, 9)`` row-major or ``(n, 3, 3)``.

    Returns
    -------
    matrices
        Array of shape ``(n, 3, 3)`` and 64-bit float data type.

    Raises
    ------
    ValueError
        If *fe* has neither layout.
    """
    array = np.asarray(fe, dtype=np.float64)
    if array.ndim == 2 and array.shape[-1] == BETA_PROP_SIZE:
        return array.reshape(-1, 3, 3)
    if array.ndim == 3 and array.shape[-2:] == (3, 3):
        return array
    raise ValueError(
        "fe must be the stored deformation gradients of shape (n, 9), row-major, or "
        f"(n, 3, 3), but has shape {array.shape}"
    )


def single_phase_stiffness_guard(xmap, stiffness: np.ndarray | None) -> None:
    """Refuse a stiffness given for a map holding several phases.

    The requirements D9.6 limitation, in ONE place because both public
    entry points which take a stiffness read it: this one and
    :func:`~kikuchipy.indexing.hrebsd_gnd`, whose stiffness selects the
    traction-free closure of the same chain (2026-09-08, Stage C
    implementation gate).

    Parameters
    ----------
    xmap
        :class:`~orix.crystal_map.CrystalMap` whose
        :attr:`~orix.crystal_map.CrystalMap.phase_id` is counted.
    stiffness
        The crystal-frame Voigt stiffness, or ``None``, in which case
        nothing is checked: without one there is no per-phase quantity
        to get wrong.

    Raises
    ------
    ValueError
        If *stiffness* is given and *xmap* holds more than one indexed
        phase, which version one does not support and which the
        message names.
    """
    if stiffness is None:
        return
    phase_ids = np.unique(np.asarray(xmap.phase_id))
    phase_ids = phase_ids[phase_ids >= 0]
    if phase_ids.size > 1:
        raise ValueError(
            f"a stiffness is ONE crystal-frame matrix and xmap holds "
            f"{phase_ids.size} indexed phases; version one does not support a "
            "per-phase stiffness, so index one phase at a time, or leave "
            "`stiffness` out for the strain and the deviatoric closure alone"
        )


def best_orientation_matrices(xmap) -> np.ndarray:
    """Return one orientation matrix per map point.

    Parameters
    ----------
    xmap
        :class:`~orix.crystal_map.CrystalMap` whose
        :attr:`~orix.crystal_map.CrystalMap.rotations` may carry
        several rotations per point, as a dictionary-indexing map with
        ``n_best > 1`` does.

    Returns
    -------
    matrices
        Array of shape ``(n, 3, 3)`` and 64-bit float data type. A map
        carrying several rotations per point is read at its BEST one,
        which is the convention
        :func:`~kikuchipy.indexing.segment_grains` already follows on
        the same input (2026-09-08, Stage B adversarial review).
    """
    matrices = np.asarray(xmap.rotations.to_matrix(), dtype=np.float64)
    if matrices.ndim > 3:
        matrices = matrices.reshape(matrices.shape[0], -1, 3, 3)[:, 0]
    return matrices


def hrebsd_strain_stress(
    xmap,
    detector,
    *,
    stiffness: np.ndarray | None = None,
    closure: str = "auto",
    strain_measure: str = "biot",
):
    """Split HREBSD-DIC deformation gradients into strain, rotation
    and stress.

    The tensor half of high angular resolution EBSD: the reduced
    detector-frame deformation gradients
    :meth:`~kikuchipy.signals.EBSD.hrebsd_dic` measured are rotated
    into the sample frame, closed in their unobservable ninth degree
    of freedom, and split by a polar decomposition into an elastic
    strain and a lattice rotation, with the stress following from a
    user supplied elastic stiffness.

    Parameters
    ----------
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map returned by
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`, carrying at least
        its ``"Fe"`` property. Its orientations, phases and other
        properties are carried through untouched.
    detector : ~kikuchipy.detectors.EBSDDetector
        The detector the patterns were measured with, whose sample
        tilt, tilt, azimuthal and twist angles define the detector to
        sample frame chain. Nothing is hardcoded.
    stiffness : numpy.ndarray, optional
        Elastic stiffness as a 6 by 6 Voigt matrix in GPa, in the
        CRYSTAL frame, in the Voigt order ``(11, 22, 33, 23, 13,
        12)`` and in the engineering shear convention
        ``sigma = C @ [e11, e22, e33, 2*e23, 2*e13, 2*e12]``. Build
        one with :func:`~kikuchipy.indexing.voigt_stiffness`. If not
        given, the closure falls back to deviatoric and no stress is
        computed.
    closure : str, optional
        Which assumption closes the ninth degree of freedom the
        projection cannot see. ``"auto"`` (default) is
        ``"traction_free"`` when *stiffness* is given and
        ``"deviatoric"`` otherwise. ``"traction_free"`` imposes
        ``sigma33 = 0`` in the sample frame and requires *stiffness*;
        ``"deviatoric"`` imposes a traceless tensor. The traction-free
        assumption is the standard one for a free surface and is
        robust except very close to a localized stress source, and
        its error grows quadratically with the angular misalignment
        of the surface :cite:`hardin2015analysis`.
    strain_measure : str, optional
        ``"biot"`` (default), the ``U - I`` of the right stretch,
        which is what OpenXY reports, or ``"green-lagrange"``,
        ``(U^T U - I)/2``.

    Returns
    -------
    xmap_out
        A copy of *xmap* with the properties ``"strain"`` (n, 6),
        ``"rotation_vector"`` (n, 3), ``"beta"`` (n, 9), ``"stress"``
        (n, 6), ``"stress_von_mises"`` (n,),
        ``"stress_hydrostatic"`` (n,) and ``"stress_principal"``
        (n, 3) added. The input map is not modified.

    Raises
    ------
    ValueError
        If *xmap* does not carry the ``"Fe"`` property, naming
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; if *closure* or
        *strain_measure* is unknown; if ``closure="traction_free"``
        is asked for without a *stiffness*; if *stiffness* is not a
        6 by 6 matrix; or if a *stiffness* is given for a map holding
        more than one phase, which version one does not support and
        which the message names.

    Notes
    -----
    Frames and units. ``"strain"`` and ``"stress"`` are in the SAMPLE
    frame in the Voigt order ``(11, 22, 33, 23, 13, 12)``; the strain
    carries TENSOR shears (``e23``, not ``2*e23``) while the
    engineering factor of two lives only inside the Hooke product;
    the stress is in the GPa the stiffness is given in.
    ``"rotation_vector"`` is ``axis * angle`` in RADIANS in the
    sample frame, and ``"beta"`` is the closed sample-frame
    displacement gradient, row-major.

    Everything is RELATIVE to each point's grain reference pattern,
    which is what ``"reference_index"`` records: an absolute strain
    would need a simulated reference, which version one does not
    provide.

    The hydrostatic stress and ``e33`` are CLOSURE DERIVED. The
    projection cannot see the isotropic part of the deformation, so
    those two numbers are as good as the closure assumption and no
    better.

    Two dimensional properties are retrieved by reshaping, for
    instance ``xmap_out.prop["strain"].reshape(ny, nx, 6)``;
    :meth:`~orix.crystal_map.CrystalMap.get_map_data` is for the
    scalar properties only (requirements D15.7).

    Points which did not converge, failed or were masked out carry
    NaN in every property here, and are never zeroed.

    A map carrying SEVERAL rotations per point, as dictionary indexing
    with ``n_best > 1`` returns, is read at its best rotation, which is
    the convention :func:`~kikuchipy.indexing.segment_grains` follows
    on the same input. The orientations only rotate the stiffness
    (requirements D7), so the choice matters on the stress path alone.

    See Also
    --------
    kikuchipy.signals.EBSD.hrebsd_dic
    kikuchipy.indexing.voigt_stiffness
    kikuchipy.indexing.hrebsd_kam
    kikuchipy.indexing.hrebsd_pc_shift
    """
    if "Fe" not in xmap.prop:
        raise ValueError(
            "xmap does not carry the 'Fe' property this function splits; pass the "
            "crystal map EBSD.hrebsd_dic returned"
        )
    single_phase_stiffness_guard(xmap, stiffness)
    properties = tensor_chain(
        np.asarray(xmap.prop["Fe"]),
        best_orientation_matrices(xmap),
        detector,
        stiffness=stiffness,
        closure=closure,
        strain_measure=strain_measure,
    )
    # The input orientations, phases and scan unit are carried through
    # untouched: this feature never modifies an orientation (D7)
    xmap_out = xmap.deepcopy()
    for name in STAGE_B_PROP_NAMES:
        xmap_out.prop[name] = properties[name]
    return xmap_out
