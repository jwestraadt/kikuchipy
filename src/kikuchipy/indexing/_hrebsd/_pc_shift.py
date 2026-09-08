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

"""Projection centre shift analysis, a diagnostic.

The beam scan moves the projection centre from point to point, and
that motion enters every measured homography as a rigid translation
plus an isotropic scaling -- the phantom requirements D6.2 removes
before converting to a deformation gradient.  This module reports the
three quantities side by side (requirements D13): what the fit
MEASURED, what the beam-scan geometry MODELS, and the residual
between them, which is the probe for a miscalibrated projection
centre (Ruggles 2020: residuals of nonzero mean indicate a
``delta-P`` miscalibration).

It reads the RAW stored ``homography`` property, which is exactly why
requirements D15.6 stores the fit uncorrected: with a corrected
homography stored, every residual here would be identically zero and
the diagnostic would be blind by construction.

The return type is a dictionary of named maps and NOT a crystal map,
frozen: these are diagnostics of a run, not per-point results of it.
"""

import numpy as np

# The frozen keys of the returned dictionary (requirements D13), in
# the order they are built
PC_SHIFT_KEYS: tuple[str, ...] = (
    "translation_x",
    "translation_y",
    "translation_x_model",
    "translation_y_model",
    "scaling_model",
    "residual_x",
    "residual_y",
)

# The properties a map must carry: the RAW homography of the engine
# and each point's reference, since every translation is measured
# against THAT point's reference projection centre
REQUIRED_PROP_NAMES: tuple[str, ...] = ("homography", "reference_index")

# Slots of the two translation components in the eight homography
# parameters ``(h11, h12, h13, h21, h22, h23, h31, h32)``
TRANSLATION_SLOTS: tuple[int, int] = (2, 5)


def hrebsd_pc_shift(xmap, detector) -> dict:
    """Compare measured and modelled beam-scan projection centre
    shifts.

    A diagnostic of an HREBSD-DIC run: the translation each fit
    measured, the translation and isotropic scaling the beam-scan
    geometry predicts from the detector's per-point projection
    centres, and the residual between them, which is the probe for a
    miscalibrated projection centre.

    Parameters
    ----------
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map returned by
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`, carrying its RAW
        ``"homography"`` property and its ``"reference_index"``
        property.
    detector : ~kikuchipy.detectors.EBSDDetector
        Detector carrying ONE PROJECTION CENTRE PER MAP POINT, whose
        shape gives the binned pixel unit every number here is in.
        A detector with a single projection centre is refused rather
        than extrapolated: the beam-scan model needs the scan step
        sizes in the unit of
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size`, and
        guessing a unit is how the errors this function exists to
        find are made. Build one with
        :meth:`~kikuchipy.detectors.EBSDDetector.extrapolate_pc`, the
        same model the engine uses internally.

    Returns
    -------
    maps : dict of numpy.ndarray
        Seven arrays of the map's navigation shape ``(ny, nx)`` and
        64-bit float data type, keyed
        ``"translation_x"``, ``"translation_y"`` (the measured
        ``h13`` and ``h23`` of the raw homography, in binned pixels),
        ``"translation_x_model"``, ``"translation_y_model"`` (the
        modelled ``PC_target - PC_reference``, in binned pixels),
        ``"scaling_model"`` (the modelled detector distance ratio
        ``DD_target / DD_reference``, dimensionless and one where
        nothing moves), and ``"residual_x"``, ``"residual_y"``
        (measured minus modelled, in binned pixels). Points which did
        not converge, failed or were masked out are NaN.

    Raises
    ------
    ValueError
        If *xmap* does not carry both ``"homography"`` and
        ``"reference_index"``, naming
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; or if the
        detector's navigation size is neither the map size nor,
        with the message above, one.

    Notes
    -----
    Everything is in BINNED detector pixels, the one unit system of
    this feature.

    The measured translations are read from the RAW homography, which
    is what :meth:`~kikuchipy.signals.EBSD.hrebsd_dic` stores: the
    beam-scan correction is applied transiently on the ``"Fe"`` path
    only, so this diagnostic sees the real translations.

    Refining a projection centre with the residuals is a documented
    workflow and not an automatic loop: fit a plane to
    ``"residual_x"`` and ``"residual_y"`` over a strain-free region
    and feed the result back through
    :meth:`~kikuchipy.detectors.EBSDDetector.fit_pc`. Version one
    wires no feedback of its own.

    See Also
    --------
    kikuchipy.signals.EBSD.hrebsd_dic
    kikuchipy.indexing.hrebsd_strain_stress
    """
    raise NotImplementedError(
        "hrebsd_pc_shift arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )


def beam_scan_model(pc_pixels: np.ndarray, reference_index: np.ndarray) -> np.ndarray:
    """Return the modelled translation and scaling of every point.

    The closed form of requirements D6.2 in the reference-PC-centred
    pixel frame: a pure projection centre change maps
    ``xi' = alpha_s * xi + gamma`` with
    ``gamma = PC_target - PC_reference`` in binned pixels exactly and
    ``alpha_s = DD_target / DD_reference``.

    Parameters
    ----------
    pc_pixels
        Projection centres of shape ``(n, 3)`` in binned pixels,
        ``(PCx_px, PCy_px, DD_px)`` per map point.
    reference_index
        Flat map index of each point's reference, of shape ``(n,)``,
        with a negative entry where no reference applies.

    Returns
    -------
    model
        Array of shape ``(n, 3)`` and 64-bit float data type holding
        ``(gamma_x, gamma_y, alpha_s)`` per point, NaN where the
        reference index is negative.

    Raises
    ------
    ValueError
        If *pc_pixels* is not ``(n, 3)``, if *reference_index* does
        not have one entry per point, or if a non-negative reference
        index is outside the map.
    """
    raise NotImplementedError(
        "beam_scan_model arrives with Stage B of specs/2026-09-07-hrebsd-dic/"
    )
