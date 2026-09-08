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

from kikuchipy.indexing._hrebsd._geometry import pc_to_pixels
from kikuchipy.indexing._hrebsd._segmentation import map_grids

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

# The properties a map must carry: the RAW homography of the engine,
# each point's reference, since every translation is measured against
# THAT point's reference projection centre, and the convergence flag.
#
# ``"converged"`` ADDED 2026-09-08 (Stage B adversarial review).  A
# point which ran out of iterations KEEPS a finite last iterate in
# ``homography`` -- that is requirements D2.6, and it is what makes the
# finiteness of the homography alone the WRONG test for "is there a
# measurement here".  Without this property the residual maps mixed
# converged and non-converged fits: MEASURED on the Si wafer route of
# validation entry 43, 87 of 100 points converged and the mean of
# ``residual_x`` came out 9.9739 px over all 100 against 10.9956 px
# over the 87, a 10 per cent contamination of the one number
# requirements D13 says the diagnostic is read on.
REQUIRED_PROP_NAMES: tuple[str, ...] = ("homography", "reference_index", "converged")

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
        ``"homography"`` property, its ``"reference_index"`` property
        and its ``"converged"`` property.
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
        same model the engine uses internally, and build it the way
        the engine does or the model here describes a different
        geometry than the run did: anchor it at
        ``pc_indices = divmod(reference_index.min(), nx)``, the scan
        position of the FIRST grain reference and not ``[0, 0]``, and
        give it the scan steps in the MICROMETRES
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size` is measured
        in. The translation half of the model is a difference and so
        does not feel the anchor, but ``"scaling_model"`` does.

    Returns
    -------
    maps
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
        If *xmap* does not carry all of ``"homography"``,
        ``"reference_index"`` and ``"converged"``, naming
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; if *xmap* has no
        map grid, which is what orix reports as the shape ``()``; or
        if the detector's navigation size is neither the map size
        nor, with the message above, one.

    Notes
    -----
    Everything is in BINNED detector pixels, the one unit system of
    this feature.

    The measured translations are read from the RAW homography, which
    is what :meth:`~kikuchipy.signals.EBSD.hrebsd_dic` stores: the
    beam-scan correction is applied transiently on the ``"Fe"`` path
    only, so this diagnostic sees the real translations.

    A point which did not converge is NOT a measurement here, even
    though its ``"homography"`` is finite: requirements D2.6 keeps the
    last iterate rather than zeroing it, so the convergence flag and
    not the finiteness of the homography is what separates a fit from
    an abandoned one, and every map here is NaN at such a point
    (recorded 2026-09-08, Stage B adversarial review).

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
    properties = xmap.prop
    missing = [name for name in REQUIRED_PROP_NAMES if name not in properties]
    if missing:
        raise ValueError(
            f"xmap does not carry the {missing} property/properties this "
            "diagnostic reads; pass the crystal map EBSD.hrebsd_dic returned, "
            "whose 'homography' is the RAW fit"
        )

    # The two dimensional shape comes from the grids, as
    # :func:`~kikuchipy.indexing.segment_grains` and
    # :func:`~kikuchipy.indexing.hrebsd_kam` take it
    rows, cols = map_grids(xmap)
    ny = int(rows.max()) + 1
    nx = int(cols.max()) + 1
    size = rows.size

    navigation_size = int(detector.navigation_size)
    if navigation_size == 1:
        raise ValueError(
            "detector carries a single projection centre, and this diagnostic "
            "refuses to extrapolate one map from it: the beam-scan model needs "
            "the scan step sizes in the unit of detector.px_size, and guessing "
            "a unit is the error this function exists to find. Build a per-point "
            "detector with EBSDDetector.extrapolate_pc, the same model the "
            "engine uses internally"
        )
    if navigation_size != size:
        raise ValueError(
            f"detector.navigation_size {navigation_size} must be the map size "
            f"{size}, one projection centre per map point"
        )

    pc_pixels = pc_to_pixels(detector.pc_flattened, detector.shape)
    reference_index = np.asarray(properties["reference_index"]).ravel().astype(np.int64)
    model = beam_scan_model(pc_pixels, reference_index)

    homography = np.asarray(properties["homography"], dtype=np.float64).reshape(
        size, -1
    )
    measured = np.full((size, 2), np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(homography), axis=1)
    for position, slot in enumerate(TRANSLATION_SLOTS):
        measured[finite, position] = homography[finite, slot]

    # A point which never converged, failed or was masked out carries
    # NaN in EVERY map here, the model included: a modelled translation
    # beside a missing measurement is a residual nobody can read.
    #
    # ``converged`` is read and not merely the finiteness of the
    # homography, because requirements D2.6 keeps a NON-CONVERGED
    # point's last iterate: it is finite, and taking it for a
    # measurement is what contaminated the D13 residual mean by 10 per
    # cent before 2026-09-08
    converged = np.asarray(properties["converged"]).ravel().astype(bool)
    dropped = ~finite | ~converged | ~np.all(np.isfinite(model), axis=1)
    measured[dropped] = np.nan
    model = np.where(dropped[:, np.newaxis], np.nan, model)

    values = {
        "translation_x": measured[:, 0],
        "translation_y": measured[:, 1],
        "translation_x_model": model[:, 0],
        "translation_y_model": model[:, 1],
        "scaling_model": model[:, 2],
        "residual_x": measured[:, 0] - model[:, 0],
        "residual_y": measured[:, 1] - model[:, 1],
    }
    maps = {}
    for key in PC_SHIFT_KEYS:
        array = np.full((ny, nx), np.nan, dtype=np.float64)
        array[rows, cols] = values[key]
        maps[key] = array
    return maps


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
    pc_pixels = np.asarray(pc_pixels, dtype=np.float64)
    if pc_pixels.ndim != 2 or pc_pixels.shape[1] != 3:
        raise ValueError(
            f"pc_pixels of shape {pc_pixels.shape} must be (n, 3), one "
            "(PCx_px, PCy_px, DD_px) per map point"
        )
    size = pc_pixels.shape[0]
    reference_index = np.asarray(reference_index).ravel()
    if reference_index.size != size:
        raise ValueError(
            f"reference_index holds {reference_index.size} entry/entries but "
            f"pc_pixels holds {size} projection centre(s); there must be one "
            "reference per point"
        )
    reference_index = reference_index.astype(np.int64)
    resolved = reference_index >= 0
    if np.any(reference_index[resolved] >= size):
        raise ValueError(
            f"every reference index must be a flat map index within the map size "
            f"{size}, got {reference_index.tolist()}"
        )
    model = np.full((size, 3), np.nan, dtype=np.float64)
    of_reference = pc_pixels[reference_index[resolved]]
    # The closed form of requirements D6.2 in the reference-PC-centred
    # pixel frame: the translation is the projection centre difference
    # exactly and the isotropic scaling is the detector distance ratio
    model[resolved, 0] = pc_pixels[resolved, 0] - of_reference[:, 0]
    model[resolved, 1] = pc_pixels[resolved, 1] - of_reference[:, 1]
    model[resolved, 2] = pc_pixels[resolved, 2] / of_reference[:, 2]
    return model
