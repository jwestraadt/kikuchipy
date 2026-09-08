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

"""Projection centre geometry of the HREBSD-DIC engine.

One unit system, binned detector pixels, everywhere inside the engine
(requirements D1).  The mixed pixel and normalized units of EMsoftOO
(``mod_HREBSDDIC.f90:916-924``) are the single most damaging defect
class of the reference implementation and are never reproduced.

From the stored Bruker fractions (D1.2)::

    PCx_px = pcx * ncols
    PCy_px = pcy * nrows
    DD_px  = pcz * nrows

with no sign flips: Bruker's y runs from the top, exactly as the
numpy array frame does.  EMsoftOO's ``patcenty = 0.5 - ypc``
(``mod_HREBSDDIC.f90:917``) is an EMsoft convention artifact and does
not appear here.

The DIC coordinates are PC-centred on the GRAIN REFERENCE's PC and
that one frame carries the reference and every target of the grain
(D1.3).  Per-point projection centres therefore enter only through
the phantom correction of this module and the conversion of
:mod:`~kikuchipy.indexing._hrebsd._homography`.

The signs of the phantom are pinned by the PC-shift phantom oracle
(validation V6), never asserted from reading EMsoftOO
(``mod_HREBSDDIC.f90:925-949``, which sits squarely in its own unit
trap and additionally hardcodes a 70 degree sample tilt, a recorded
deviation: every angle here comes from the
:class:`~kikuchipy.detectors.EBSDDetector`).

References
----------
Ernould et al., AIEP 223 (2022) Ch. 2 section 3.3.2; Singh and De
Graef, Microsc. Microanal. 23 (2017) 1 appendix A (the beam-scan PC
model kikuchipy already implements as
:meth:`~kikuchipy.detectors.EBSDDetector.extrapolate_pc`).
"""

import numpy as np

from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    compose,
    homography_to_fe,
    invert,
)

# Navigation-axis units the scan step sizes are converted FROM, into
# the micrometres of :attr:`EBSDDetector.px_size`.  Requirements D14.5
# sets the precedent this follows: convert the units that are known
# and raise on the ones that are not, since silently guessing units is
# how prefactor bugs hide
STEP_SIZE_UNIT_FACTORS: dict[str, float] = {
    "m": 1e6,
    "mm": 1e3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
}


def step_sizes_in_micrometres(
    scales: tuple[float, ...], units: tuple[str, ...]
) -> tuple[float, ...]:
    """Return scan step sizes converted to micrometres.

    Parameters
    ----------
    scales
        Navigation axis scales, in the units of *units*.
    units
        Navigation axis units, one per entry of *scales*.

    Returns
    -------
    converted
        The scales in micrometres, the unit of
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size`.

    Raises
    ------
    ValueError
        If any unit is missing or is not one of
        :data:`STEP_SIZE_UNIT_FACTORS`, naming the axis and listing
        what is accepted. The ``"px"`` fallback HyperSpy uses for an
        unscaled axis raises here, exactly as requirements D14.5
        makes it raise for ``CrystalMap.scan_unit``.
    """
    converted = []
    for axis, (scale, unit) in enumerate(zip(scales, units)):
        key = str(unit).strip().lower()
        if key not in STEP_SIZE_UNIT_FACTORS:
            raise ValueError(
                f"navigation axis {axis} carries the scan step unit {unit!r}, which "
                "cannot be converted to the micrometres of detector.px_size; set the "
                "axis' `units` to one of "
                f"{sorted(set(STEP_SIZE_UNIT_FACTORS))}, or pass a detector which "
                "already carries one projection centre per map point"
            )
        converted.append(float(scale) * STEP_SIZE_UNIT_FACTORS[key])
    return tuple(converted)


def pc_to_pixels(pc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return projection centres in binned detector pixels.

    Parameters
    ----------
    pc
        Bruker projection centres of shape ``(n, 3)`` or ``(3,)``,
        each ``(pcx, pcy, pcz)`` in fractions.
    shape
        Detector (pattern) shape ``(nrows, ncols)`` in binned pixels.

    Returns
    -------
    pc_px
        Array of shape ``(n, 3)`` and 64-bit float data type holding
        ``(PCx_px, PCy_px, DD_px)``, always two dimensional even for
        one projection centre.

    Raises
    ------
    ValueError
        If *pc* does not have three entries along its last axis.
    """
    pc = np.atleast_2d(np.asarray(pc, dtype=np.float64))
    if pc.shape[-1] != 3:
        raise ValueError(
            f"pc must hold three entries (pcx, pcy, pcz) along its last axis, not "
            f"{pc.shape[-1]}"
        )
    pc = pc.reshape(-1, 3)
    nrows, ncols = shape
    pc_px = np.empty(pc.shape, dtype=np.float64)
    pc_px[:, 0] = pc[:, 0] * ncols
    pc_px[:, 1] = pc[:, 1] * nrows
    # Bruker's ``pcz`` is the detector distance in fractions of the
    # number of detector ROWS (requirements D1.2)
    pc_px[:, 2] = pc[:, 2] * nrows
    return pc_px


def per_point_pc_pixels(
    detector,
    navigation_shape: tuple[int, int],
    step_sizes: tuple[float, float],
    *,
    reference_index: int = 0,
) -> np.ndarray:
    """Return one projection centre per map point, in binned pixels.

    Per-point projection centres are first class (requirements D6.1).
    When the detector carries one per map point, its
    :attr:`~kikuchipy.detectors.EBSDDetector.pc_flattened` is
    consumed directly, which is exactly what the fork's ``fit_pc``,
    ``extrapolate_pc`` and
    ``refine_orientation_projection_center`` workflows produce.  When
    it carries a single projection centre, the beam-scan model of
    :meth:`~kikuchipy.detectors.EBSDDetector.extrapolate_pc` derives
    the map internally, anchored at the reference pattern's scan
    position.

    Parameters
    ----------
    detector
        :class:`~kikuchipy.detectors.EBSDDetector` of the patterns.
    navigation_shape
        Map shape ``(ny, nx)``.
    step_sizes
        Vertical and horizontal step sizes ``(dy, dx)`` **in the unit
        of** :attr:`~kikuchipy.detectors.EBSDDetector.px_size`, which
        is micrometres by the kikuchipy convention. Used only when
        the detector carries a single projection centre.
    reference_index
        Flat map index of the pattern the extrapolation is anchored
        at. Default is 0. Ignored when the detector already carries
        one projection centre per point.

    Returns
    -------
    pc_px
        Array of shape ``(ny * nx, 3)`` and 64-bit float data type
        holding ``(PCx_px, PCy_px, DD_px)`` per map point, in map
        order.

    Raises
    ------
    ValueError
        If the detector's navigation size is neither one nor the map
        size.

    Notes
    -----
    The beam-scan model divides every step by
    ``px_size * binning`` (``_ebsd_detector.py:1360-1379``), so
    *step_sizes*, :attr:`px_size` and :attr:`binning` must agree in
    their units or the derived projection centre map, and with it the
    D6.2 correction removed from every homography, is wrong by their
    ratio.  This function cannot check that: a detector carries no
    unit for :attr:`px_size`, and its default 1.0 is a placeholder
    (the shipped ``nickel_ebsd_small`` detector carries it beside a
    1.5 micrometre scan step).  The navigation-axis UNITS are checked
    and converted one level up, in
    :meth:`kikuchipy.signals.EBSD.hrebsd_dic`, which is where they
    live; passing a wrong :attr:`px_size` stays the caller's
    responsibility and is documented there.
    """
    ny, nx = navigation_shape
    map_size = int(ny) * int(nx)
    navigation_size = int(detector.navigation_size)
    if navigation_size == map_size:
        return pc_to_pixels(detector.pc_flattened, detector.shape)
    if navigation_size != 1:
        raise ValueError(
            f"detector.navigation_size {navigation_size} must be either one or the "
            f"map size {map_size} of navigation_shape {navigation_shape}"
        )
    # The beam-scan model IS ``EBSDDetector.extrapolate_pc`` (Singh and
    # De Graef appendix A), anchored at the reference pattern's scan
    # position so that point keeps the detector's own projection centre
    row, col = divmod(int(reference_index), int(nx))
    extrapolated = detector.extrapolate_pc(
        pc_indices=[row, col],
        navigation_shape=navigation_shape,
        step_sizes=step_sizes,
    )
    return pc_to_pixels(extrapolated.pc_flattened, detector.shape)


def phantom_homography(pc_reference: np.ndarray, pc_target: np.ndarray) -> np.ndarray:
    """Return the homography a pure PC and DD change induces.

    Closed form in the spec's own reference-PC-centred pixel frame
    (requirements D6.2): a pure projection centre change maps
    ``xi' = alpha_s * xi + gamma`` with ``gamma = PC_target -
    PC_reference`` in binned pixels exactly, and ``alpha_s`` the
    detector distance ratio, so the phantom factor is the affinity::

        W_phantom = [ alpha_s  0        gamma_x ]
                    [ 0        alpha_s  gamma_y ]
                    [ 0        0        1       ]

    EMsoftOO's ``gamma_i = delta_i + (delta_i - patcent_i) *
    (alpha_s - 1)`` (``mod_HREBSDDIC.f90:955-970``) is written for
    its absolute-PC coordinates and is NOT transplanted: the extra
    term vanishes exactly when the projection centre is the
    coordinate origin, which is this frame.

    Parameters
    ----------
    pc_reference
        ``(PCx_px, PCy_px, DD_px)`` of the grain reference.
    pc_target
        ``(PCx_px, PCy_px, DD_px)`` of the target pattern.

    Returns
    -------
    h
        Homography parameters of shape ``(8,)`` and 64-bit float data
        type. The drafted orientation of the detector distance ratio
        is pinned by validation V6 and recorded in requirements D6.3
        with its date.
    """
    pc_reference = np.asarray(pc_reference, dtype=np.float64)
    pc_target = np.asarray(pc_target, dtype=np.float64)
    # DRAFTED orientation of the detector distance ratio (D6.2), pinned
    # by the pattern oracle of validation V6
    alpha_s = pc_target[2] / pc_reference[2]
    gamma = pc_target[:2] - pc_reference[:2]
    h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    h[0] = alpha_s - 1.0
    h[4] = alpha_s - 1.0
    h[2] = gamma[0]
    h[5] = gamma[1]
    return h


def correct_homography(
    h: np.ndarray, pc_reference: np.ndarray, pc_target: np.ndarray
) -> np.ndarray:
    """Return *h* with the beam-scan phantom removed.

    The correction of requirements D6.2, applied analytically BEFORE
    the conversion to a deformation gradient.  EMsoftOO computes the
    corrected homographies and then converts the UNCORRECTED ones
    (``mod_HREBSDDIC.f90:978-982``), a recorded deviation.

    Parameters
    ----------
    h
        Raw fitted homography parameters of shape ``(8,)``.
    pc_reference
        ``(PCx_px, PCy_px, DD_px)`` of the grain reference.
    pc_target
        ``(PCx_px, PCy_px, DD_px)`` of the target pattern.

    Returns
    -------
    h_corrected
        Parameters of shape ``(8,)``. The composition side is pinned
        by validation V6 and recorded in requirements D6.3.
    """
    phantom = phantom_homography(pc_reference, pc_target)
    # DRAFTED side ``W_corr = W_phantom**-1 . W`` (D6.2)
    return compose(invert(phantom), h)


def fe_from_homography(
    h: np.ndarray,
    pc_reference: np.ndarray,
    pc_target: np.ndarray,
    *,
    correct: bool = True,
) -> np.ndarray:
    """Return the reduced deformation gradient of a fitted *h*.

    The whole D6 path in one call: remove the beam-scan phantom, then
    convert in the frame the corrected homography lives in.

    Parameters
    ----------
    h
        Raw fitted homography parameters of shape ``(8,)``.
    pc_reference
        ``(PCx_px, PCy_px, DD_px)`` of the grain reference.
    pc_target
        ``(PCx_px, PCy_px, DD_px)`` of the target pattern.
    correct
        Whether to remove the phantom first. Default is ``True``.
        The private switch exists for the phantom oracle of
        validation V6, which needs the uncorrected homographies to
        pin the signs, and is never exposed in the public API.

    Returns
    -------
    fe
        Array of shape ``(3, 3)`` and 64-bit float data type, the
        reduced tensor in the DETECTOR frame (requirements D7). The
        rotation into the sample frame, the ninth degree of freedom
        closure and the strain and stress split are Stage B.

    Notes
    -----
    **CORRECTED 2026-09-07 (Stage A adversarial review, requirements
    D6.2 amended with that date).**  A raw fit in the D1.3 frame is
    ``W = T(delta) . diag(1, 1, 1/DD_t) . Fe . diag(1, 1, DD_r)`` with
    ``delta = PC_t - PC_ref``, whose ``Fe = I`` case is exactly the
    closed-form phantom of :func:`phantom_homography`.  Removing that
    phantom on the D6.2 side therefore leaves
    ``W_corr = diag(1, 1, DD_r)**-1 . Fe . diag(1, 1, DD_r)``, a PURE
    REFERENCE-FRAME homography in which both the projection centre
    offset and the detector distance ratio have already cancelled.
    The exact conversion of the corrected homography is consequently
    ``pc_rel = (0, 0)`` and ``dd = DD_reference``, at which
    :func:`~kikuchipy.indexing._hrebsd._homography.homography_to_fe`
    IS that conjugation.  The earlier
    ``pc_rel = PC_t - PC_ref``/``dd = DD_t`` route was recorded as
    "first-order equivalent", which measurement refutes: on the
    validation V6 phantom geometry it injects a spurious isotropic
    strain of 3.9e-04 into ``Fe11``/``Fe22`` at a one degree
    out-of-plane tilt, thirteen times the pinned
    ``DEFORMED_MASTER_FE_TOL``, while the route above is exact to
    1e-17.  The ``correct=False`` diagnostic path keeps the target
    projection centre, which is what validation V6's uncorrected arm
    reads.
    """
    pc_reference = np.asarray(pc_reference, dtype=np.float64)
    pc_target = np.asarray(pc_target, dtype=np.float64)
    if correct:
        # The corrected homography already lives in the reference-PC
        # frame, so the frame origin IS its conversion centre
        h = correct_homography(h, pc_reference, pc_target)
        return homography_to_fe(h, np.zeros(2), pc_reference[2])
    # The uncorrected diagnostic path still sees the TARGET's own
    # projection centre expressed in the reference-centred frame
    pc_rel = pc_target[:2] - pc_reference[:2]
    return homography_to_fe(h, pc_rel, pc_target[2])
