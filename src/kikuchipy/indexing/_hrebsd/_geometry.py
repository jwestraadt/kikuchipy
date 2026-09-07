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
    raise NotImplementedError


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
        Vertical and horizontal step sizes ``(dy, dx)`` from the
        signal's navigation axes, in the axes' own units.
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
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def fe_from_homography(
    h: np.ndarray,
    pc_reference: np.ndarray,
    pc_target: np.ndarray,
    *,
    correct: bool = True,
) -> np.ndarray:
    """Return the reduced deformation gradient of a fitted *h*.

    The whole D6 path in one call: remove the beam-scan phantom, then
    convert with the target's own projection centre expressed in the
    reference-centred frame.

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
    """
    raise NotImplementedError
