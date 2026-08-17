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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517), from
# ``include/modality/ebsd/detector.hpp`` unless stated otherwise:
# - ``BackProjector<Real>::Constants`` (lines 502-569), as
#   :func:`_build_lut`
# - ``BackProjector<Real>::unproject()`` (lines 589-623), as
#   :func:`_unproject_kernel` and
#   :meth:`SphericalBackProjector.unproject`
# - ``BackProjector<Real>::mask()`` (lines 628-630), as
#   :meth:`SphericalBackProjector.window_mask`
# - ``Geometry<Real>::solidAngle()`` (lines 401-415), as
#   :func:`_solid_angle_fraction`
# - ``Geometry<Real>::scaleFactor()`` (lines 465-469) and the size
#   arithmetic of ``Geometry<Real>::rescale(wNew, hNew)`` (lines
#   432-450), as :func:`_rescaled_shape`
# - the "cannot rescale detector to less than 1 pixel" guard of the
#   one argument ``Geometry<Real>::rescale()`` (line 424), extended
#   to the two argument overload which ``Constants`` uses and to the
#   empty window it cannot see
# - ``image::Rescaler<Real>`` (``include/util/image.hpp``, lines
#   143-186 and 564-619), as :func:`_dct_rescale`
# - ``image::imageQuality()`` (``include/util/image.hpp``, lines
#   489-507), as :func:`_image_quality_from_spectrum` and
#   :func:`_dct_image_quality`
# - ``image::BiPix<Real>::bilinearCoeff()`` and
#   ``image::BiPix<Real>::interpolate()`` (``include/util/image.hpp``,
#   lines 513-553), as the weights of :func:`_build_lut` and the
#   gather of :func:`_unproject_kernel`
#
# ``Geometry<Real>::interpolatePixel()`` (lines 334-373), the
# direction to pixel map, is **replaced** by kikuchipy's own detector
# geometry, the exact inverse of ``_get_direction_cosines_for_fixed_pc``
# of ``kikuchipy.signals.util._master_pattern``, see the module
# documentation.
#
# The following are deliberately **not** ported here:
# - the south hemisphere loop of ``Constants`` (lines 539-557).
#   ``interpolatePixel()`` rejects every direction with a negative
#   z component at line 336 before anything else, so that loop can
#   never insert a point and its colliding ``p.idx = i`` is
#   unreachable
# - ``Geometry<Real>::northPoleQuat()`` (lines 455-459), the identity
#   in EMSphInx as shipped, and ``Geometry<Real>::ecp()``,
#   ``Geometry<Real>::readEMsoft()`` and ``Geometry<Real>::flip``
# - ``BackProjector<Real>::clone()`` (which shares ``pLut`` and
#   allocates work space): a projector here is immutable after
#   construction and is shared between threads instead
# - the high pass filter of ``Rescaler<Real>::scale()`` (``flt``,
#   ``include/util/image.hpp`` lines 605-616), which ``IndexEBSD``
#   never enables (line 591 of ``detector.hpp`` passes ``flt = 0``)

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Changed by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""Back-projection of detector patterns onto the square Legendre
grid through kikuchipy's detector geometry.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**The direction to pixel chain** (frozen).  For a sample frame unit
vector ``n``, with ``M = detector.sample_to_detector.to_matrix()``
whose rows are the detector basis in sample coordinates,

.. code-block::

    v     = M n                       (v[2] > 0: in front)
    x_g   = v[0] / v[2]
    y_g   = v[1] / v[2]
    col   = (x_g - gb[0]) / x_scale - 0.5
    row   = (gb[3] - y_g) / y_scale - 0.5

with ``gb = detector.gnomonic_bounds.squeeze()``,
``x_scale = (gb[1] - gb[0]) / ncols`` and
``y_scale = (gb[3] - gb[2]) / nrows`` -- the **kernel's** pixel
pitch of ``_get_direction_cosines_for_fixed_pc`` (lines 170-171 of
``kikuchipy.signals.util._master_pattern``), not the
``EBSDDetector.x_scale`` property, which divides by ``ncols - 1``.
Row 0 holds ``y_max``.  This is the exact inverse of kikuchipy's
forward projection: applied to
``_get_direction_cosines_from_detector(detector)`` it returns the
pixel centres to 1.6e-14 (col) and 1.4e-14 (row) pixels, measured on
the Ni detector.  Only ``pc``, ``shape``, ``sample_tilt`` and
``tilt`` enter; ``binning`` and ``px_size`` never do, since the
gnomonic bounds are fractions of the binned shape.

**The sample frame is the sphere frame.**
``_get_direction_cosines_from_detector`` returns sample frame
directions and the forward projection reads the master pattern at
``rotate_vector(q, dc)``, so ``pattern(pixel) = f_master(O n)`` with
``O`` the sample to crystal orientation.  Consequently ``z_s >= 0``
is the physical guard (the C++ ``if (signbit(n[2])) return false``,
line 336, "can't project through sample") and **the south hemisphere
is never gathered**: :meth:`SphericalBackProjector.unproject` always
returns a south grid of zeros, which
:meth:`~kikuchipy.indexing._spherical._sht.
SphericalHarmonicTransform.analyze` and the correlator read as "no
data".  A footprint reaching below the sample plane is clipped, not
wrapped: at ``sample_tilt`` 0 78 % of the pixels have ``z < 0`` and
60 % of the window is lost.

EMSphInx' frame coincides with kikuchipy's: its
``alpha = 90 - sTlt + dTlt`` degrees puts the pattern centre at
``(sin alpha, 0, cos alpha)``, and kikuchipy's detector normal at
``sample_tilt`` 70 with ``tilt`` 0 is measured
``(sin 20, 0, cos 20)``, with ``X_d = +Y_s``, ``Y_d = +Z_s`` before
the tilts and both tilts about ``Y_s``.

**Pixel centre convention** (a deviation, measured).  EMSphInx'
``bilinearCoeff()`` maps the fractional physical position
``X in [0, 1]`` to ``x = X (w - 1)``, i.e. it stretches the pixel
centres over the whole physical width, displacing pixel ``c`` by
``0.5 - (c + 0.5) / w`` -- up to 0.49 px at either edge of the
resampled image.  The port keeps the pixel centre convention above
and maps it into the ``(h_out, w_out)`` image with the DCT's own
sampling convention, ``x = (col + 0.5) w_out / ncols - 0.5``, which
preserves the physical extent.  The direction oracle cannot see a
stretch applied at that step and neither can the forward-projection
lock, so the resample map is pinned structurally by the tests.

**Sizes** (``Geometry::solidAngle()``, ``Geometry::scaleFactor()``,
``image::Rescaler``).  With ``sa`` the fraction of the sphere the
detector covers, evaluated on a ``502 x 502`` square Lambert grid,

.. code-block::

    scale_factor = sqrt(sa (2 dim^2 - 4 (dim - 1)) / (ncols nrows))
    w_out        = floor(scale_factor oversampling ncols + 0.5)
    h_out        = floor(scale_factor oversampling nrows + 0.5)

with ``oversampling = sqrt(2)``, the C++ ``fct``, and ``round``
half away from zero -- never Python's banker's :func:`round`.  The
divisor of ``solidAngle(501)`` is the literal C++
``501^2 + 499^2 = 500002`` although the loop evaluates
``502^2 = 252004`` northern points, so the fraction is biased 0.4 %
high; the quirk is kept for size parity with ``IndexEBSD``.
Measured for the 60 x 60 Ni detector, circle / no circle:

.. code-block::

    bw (dim)      scale_factor       w_out    LUT points
     53 (55)   0.4489 / 0.4859     38 / 41    667 / 788
     63 (65)   0.5320 / 0.5759     45 / 49    934 / 1093
     68 (71)   0.5819 / 0.6299     49 / 53   1117 / 1317
     88 (91)   0.7481 / 0.8098     63 / 69   1844 / 2157
    113 (115)  0.9476 / 1.0257     80 / 87   2955 / 3474

so the resampled pattern is *smaller* than the detector for every
indexing bandwidth up to about 113: ``oversampling`` is relative to
the average spherical grid pixel, not to the detector.

**Resample and normalisation.**  A pattern is resampled with an
unnormalised type-2 discrete cosine transform, a low frequency
corner copy, a zeroed DC term and an unnormalised type-3 transform
(never :func:`scipy.fft.idctn`); the round trip factor
``4 h_in w_in`` cancels in the unit variance normalisation.  The
gather is bilinear with the clamped weights of ``BiPix``, and the
window values are made zero mean and unit variance with the ring
solid angles as weights.  pocketfft's type-2 transform of a constant
image is **not** exactly a delta (AC terms up to 1.1e-11 for the
constant 37), so a constant pattern is detected with ``ptp == 0``
*before* the transform and takes the window mask branch, with an
image quality of ``1.0`` for a non-zero constant and ``0.0`` for the
all-zero pattern -- the exact values of the literal
``imageQuality()``.

**``signal_mask``** is in kikuchipy polarity, ``True`` = ignore the
pixel, as in :meth:`~kikuchipy.signals.EBSD.dictionary_indexing`.  A
grid point is excluded when its **nearest** pixel is masked, and the
masked pixels are filled with the mean of the unmasked ones before
the transform, since the resample is a global spectral operation
(measured: a masked block set to 255 drops the windowed correlation
of a recovered ``Y_4^2`` from 0.997 to 0.932 without the fill).
``circular_mask`` is ``False`` by default, as ``IndexEBSD``'s
namelist ``circmask = -1``; ``True`` is the largest circle inscribed
in the *physical* detector, and an explicit radius is expressed
through ``signal_mask`` instead.

**Who owns what.**  The projector provides the two spherical inputs
which depend on *its* grid, :attr:`SphericalBackProjector.
window_harmonics` (the C++ ``mlm``, computed at construction) and
:meth:`SphericalBackProjector.squared_harmonics` (``flm2``).  The
Huhle denominator ``rDen`` stays in
:class:`~kikuchipy.indexing._spherical._xcorr.
NormalizedSphericalCrossCorrelator`, which builds it from
``(flm, flm2, mlm)``.

**Convention lock.**  Measured over 27 ``get_patterns()`` rotations
at ``bw`` 68 with both correlators, the boundary orientation of a
correlated pattern is ``_euler.rotation_from_zyz(zyz)``, i.e.
``~Rotation(zyz_to_quaternion(zyz))``: 0.34 degrees median and 0.72
maximum misorientation, against 35 degrees median for the other
sign.  The provisional sign of Phases 3 and 4 is frozen by that
measurement.

**Threads, speed and memory.**  Everything is set once in
``__init__``; :meth:`SphericalBackProjector.unproject` reads the
lookup table and writes only the caller's buffers and per-call
temporaries, so **one projector is shared by all threads** and no
``clone()`` is offered.  Measured single thread with the kernel
warm: ``unproject`` including the transform and the image quality
0.055 / 0.058 / 0.069 / 0.077 / 0.080 ms at ``bw`` 53 / 63 / 68 /
88 / 113, against 0.19 / 0.30 / 0.40 / 0.83 / 1.83 ms for the
spherical harmonic analysis of the result; construction 48-95 ms
with a 30 MB transient for the solid angle grid; resident tables
2.7-23 MB (the transform's) and under 0.3 MB (the lookup table's).

References
----------
:cite:`lenthe2019spherical`
"""

import math

from numba import njit
import numpy as np
from scipy.fft import dctn

from kikuchipy.detectors import EBSDDetector
from kikuchipy.indexing._spherical import _grid
from kikuchipy.indexing._spherical._sht import SphericalHarmonicTransform

# :func:`scipy.fft.dctn` is bound in this namespace, as Phases 1-4
# bind their transforms, so that the call recording test can patch
# ``_back_projection.dctn`` and see every transform this phase
# makes.  ``_preprocessing`` deliberately has no ``scipy.fft``
# import: the discrete cosine image quality lives here, and here
# only.

# Side length minus one of the square Lambert grid of
# ``Geometry::solidAngle()`` (``detector.hpp`` line 403, the C++
# ``gridRes``).  The loop evaluates ``(gridRes + 1)^2`` northern
# points
_SOLID_ANGLE_GRID_RES = 501

# Linear oversampling of the resampled detector relative to the
# average spherical grid pixel, the C++ ``fct`` of ``idx.hpp`` line
# 259
_DEFAULT_OVERSAMPLING = math.sqrt(2)

# --------------------------- Numba kernels -------------------------- #


@njit(cache=True, nogil=True)
def _unproject_kernel(
    rescaled_flat: np.ndarray,
    pixel_index: np.ndarray,
    weights: np.ndarray,
    solid_angles: np.ndarray,
    window_solid_angle: float,
    sphere_index: np.ndarray,
    north_flat: np.ndarray,
) -> float:
    """Gather a resampled pattern onto the window of the north grid
    and normalise it in place.

    Parameters
    ----------
    rescaled_flat
        Flat view of the ``(h_out, w_out)`` resampled pattern, 64-bit
        float.
    pixel_index
        ``(n_points, 4)`` 64-bit integer flat indices into
        ``rescaled_flat`` of the four bilinear neighbours of every
        window point, in the C++ order ``(j0 i0, j0 i1, j1 i0,
        j1 i1)``.
    weights
        ``(n_points, 4)`` 64-bit float bilinear weights of those
        four neighbours, each row summing to one.
    solid_angles
        ``(n_points,)`` 64-bit float ring solid angles of the window
        points, the C++ ``omeg``.
    window_solid_angle
        Sum of ``solid_angles``, the C++ ``omgW``.
    sphere_index
        ``(n_points,)`` 64-bit integer flat indices into the
        ``(dim, dim)`` north grid.
    north_flat
        Flat view of the ``(dim, dim)`` 64-bit float north grid,
        written in place.  **Only the window points are written**,
        exactly as the C++ does, so whatever the caller left
        elsewhere stays there.

    Returns
    -------
    stdev
        Solid angle weighted standard deviation of the gathered
        values before the normalisation, or ``0.0`` when it was
        exactly zero.

    Notes
    -----
    Port of ``BackProjector<Real>::unproject()``
    (``include/modality/ebsd/detector.hpp``, lines 589-623) after
    the rescale, with ``image::BiPix<Real>::interpolate()``
    (``include/util/image.hpp``, lines 513-517) inlined.

    The literal ``if(Real(0) == stdev)`` branch of line 607 writes
    ``1`` on every window point and returns ``0.0``.  Through the
    real lookup table it is reachable only from an all-zero
    resampled image: 18 of 1117 weight rows sum to one plus or minus
    an ulp at ``bw`` 68, so a non-zero constant image gives a
    non-zero standard deviation (measured 1.5e-14 for the constant
    7).  It must **not** be relaxed to ``stdev <= tiny``: an image
    of amplitude 1e-14 is normalised, not masked.

    The C++ return value ``var = sqrt(omgW / omgS 4 pi)`` (line 619)
    is a per-geometry constant feeding the dead ``Indexer::sum2()``
    and is not returned; the standard deviation is, so that a test
    can see the ``stdev == 0`` branch which
    :meth:`SphericalBackProjector.unproject` hides.
    """
    n_points = pixel_index.shape[0]
    values = np.empty(n_points)

    # ``BiPix::interpolate()`` and the weighted mean of line 598
    mean = 0.0
    for i in range(n_points):
        value = 0.0
        for k in range(4):
            value += rescaled_flat[pixel_index[i, k]] * weights[i, k]
        values[i] = value
        mean += value * solid_angles[i]
    mean /= window_solid_angle

    # Make the mean zero and take the weighted standard deviation,
    # lines 601-605
    stdev = 0.0
    for i in range(n_points):
        values[i] -= mean
        stdev += values[i] * values[i] * solid_angles[i]
    stdev = np.sqrt(stdev / window_solid_angle)

    # The literal ``if(Real(0) == stdev)`` branch of line 607
    if stdev == 0.0:
        for i in range(n_points):
            north_flat[sphere_index[i]] = 1.0
        return 0.0

    for i in range(n_points):
        north_flat[sphere_index[i]] = values[i] / stdev
    return stdev


# ----------------------------- Geometry ----------------------------- #


def _pixel_map(
    detector: "EBSDDetector",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the geometry of the direction to pixel chain.

    Parameters
    ----------
    detector
        Detector with a single projection centre.

    Returns
    -------
    matrix
        ``(3, 3)`` 64-bit float sample to detector matrix
        ``detector.sample_to_detector.to_matrix().squeeze()``, whose
        rows are the detector basis in sample coordinates.
    gnomonic_bounds
        ``(4,)`` 64-bit float ``(x_min, x_max, y_min, y_max)``.
    x_scale, y_scale
        Gnomonic pixel pitches ``(x_max - x_min) / ncols`` and
        ``(y_max - y_min) / nrows``, i.e. the pitches of
        ``_get_direction_cosines_for_fixed_pc`` and **not** the
        ``EBSDDetector.x_scale`` and ``EBSDDetector.y_scale``
        properties, which divide by ``ncols - 1`` and ``nrows - 1``.

    Notes
    -----
    Replaces the geometry members of
    ``Geometry<Real>::interpolatePixel()``
    (``include/modality/ebsd/detector.hpp``, lines 334-352), see the
    module documentation.
    """
    matrix = np.asarray(
        detector.sample_to_detector.to_matrix(), dtype=np.float64
    ).reshape(3, 3)
    gnomonic_bounds = np.asarray(detector.gnomonic_bounds, dtype=np.float64).reshape(4)
    x_scale = (gnomonic_bounds[1] - gnomonic_bounds[0]) / detector.ncols
    y_scale = (gnomonic_bounds[3] - gnomonic_bounds[2]) / detector.nrows
    return matrix, gnomonic_bounds, float(x_scale), float(y_scale)


def _directions_to_pixels(
    normals: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the continuous detector pixel coordinates of sample
    frame directions.

    Parameters
    ----------
    normals
        ``(n, 3)`` array-like of sample frame directions, not
        necessarily normalised.
    geometry
        The return of :func:`_pixel_map`.

    Returns
    -------
    col, row
        ``(n,)`` 64-bit float continuous pixel centre coordinates,
        with ``col = 0`` the centre of the first column and
        ``row = 0`` the centre of the first row, which holds
        ``y_max``.  Both are NaN or infinite where a direction is
        parallel to the detector plane.
    in_front
        ``(n,)`` boolean, ``True`` where the direction is in front
        of the detector, i.e. where the detector frame ``z``
        component is positive.

    Notes
    -----
    The exact inverse of
    ``kikuchipy.signals.util._master_pattern.
    _get_direction_cosines_for_fixed_pc``, replacing
    ``Geometry<Real>::interpolatePixel()``
    (``include/modality/ebsd/detector.hpp``, lines 353-362).  The
    division by the ``z`` component is left unguarded and the
    resulting non-finite coordinates are rejected by
    :func:`_inside_detector`.
    """
    matrix, gnomonic_bounds, x_scale, y_scale = geometry
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    detector_frame = normals @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        x_gnomonic = detector_frame[:, 0] / detector_frame[:, 2]
        y_gnomonic = detector_frame[:, 1] / detector_frame[:, 2]
    col = (x_gnomonic - gnomonic_bounds[0]) / x_scale - 0.5
    row = (gnomonic_bounds[3] - y_gnomonic) / y_scale - 0.5
    return col, row, detector_frame[:, 2] > 0


def _inside_detector(
    col: np.ndarray,
    row: np.ndarray,
    shape: tuple[int, int],
    circular_mask: bool,
    signal_mask: np.ndarray | None,
) -> np.ndarray:
    """Return whether continuous pixel coordinates fall on the
    unmasked part of the detector.

    Parameters
    ----------
    col, row
        ``(n,)`` continuous pixel centre coordinates of
        :func:`_directions_to_pixels`.
    shape
        Detector shape ``(nrows, ncols)``.
    circular_mask
        Whether to keep only the largest circle inscribed in the
        **physical** detector, ``(col - (ncols - 1) / 2)^2 +
        (row - (nrows - 1) / 2)^2 <= (min(ncols, nrows) / 2)^2``.
    signal_mask
        Optional ``(nrows, ncols)`` boolean mask in kikuchipy
        polarity, ``True`` = ignore the pixel.  A point is excluded
        when its **nearest** pixel is masked.

    Returns
    -------
    inside
        ``(n,)`` boolean.

    Notes
    -----
    The physical extent test is the C++ ``X, Y in [0, 1]`` of
    ``Geometry<Real>::interpolatePixel()``
    (``include/modality/ebsd/detector.hpp``, line 358), closed at
    both ends, so the half pixel rim between the edge pixel centres
    and the physical edge belongs to the detector.  Non-finite
    coordinates are rejected here.

    One circle predicate is shared by :func:`_solid_angle_fraction`
    and :func:`_build_lut`, the physical circle EMSphInx applies in
    ``solidAngle()`` (lines 465-469 through 363-368).  Its own
    lookup table loop applies the circle on the *rescaled* geometry
    instead, which is the same circle for a square detector and
    differs by rounding for a rectangular one (measured: 953 against
    958 points on a ``(48, 60)`` detector at ``bw`` 68).  The port
    uses one circle, the physical one.
    """
    nrows, ncols = shape
    inside = (
        np.isfinite(col)
        & np.isfinite(row)
        & (col >= -0.5)
        & (col <= ncols - 0.5)
        & (row >= -0.5)
        & (row <= nrows - 0.5)
    )
    if circular_mask:
        radius = min(ncols, nrows) / 2
        delta_x = col - (ncols - 1) / 2
        delta_y = row - (nrows - 1) / 2
        inside &= delta_x * delta_x + delta_y * delta_y <= radius * radius
    if signal_mask is not None:
        columns = np.clip(np.floor(np.nan_to_num(col) + 0.5), 0, ncols - 1).astype(
            np.int64
        )
        rows = np.clip(np.floor(np.nan_to_num(row) + 0.5), 0, nrows - 1).astype(
            np.int64
        )
        inside &= ~signal_mask[rows, columns]
    return inside


def _solid_angle_fraction(
    detector: "EBSDDetector",
    circular_mask: bool,
    signal_mask: np.ndarray | None,
    grid_res: int = _SOLID_ANGLE_GRID_RES,
) -> float:
    """Return the fraction of the sphere the detector covers.

    Parameters
    ----------
    detector
        Detector with a single projection centre.
    circular_mask, signal_mask
        As in :func:`_inside_detector`.
    grid_res
        Side length minus one of the square Lambert grid, 501 by
        default, the C++ ``gridRes``.

    Returns
    -------
    fraction
        Number of the ``(grid_res + 1)^2`` northern grid directions
        which are in front of the detector, have a non-negative
        sample frame ``z`` component and fall inside
        :func:`_inside_detector`, divided by the literal C++
        ``grid_res^2 + (grid_res - 2)^2``.

    Notes
    -----
    Port of ``Geometry<Real>::solidAngle()``
    (``include/modality/ebsd/detector.hpp``, lines 401-415) on
    Phase 1's :func:`~kikuchipy.indexing._spherical._grid.
    square_to_sphere`, which is the same Rosca-Lambert map; only the
    count matters.

    **The divisor is a C++ quirk, ported literally.**  The loop
    evaluates ``(gridRes + 1)^2 = 252004`` northern points, so the
    matching two hemisphere count with the equator once would be
    ``502^2 + 500^2 = 502004``, not ``501^2 + 499^2 = 500002``.  The
    fraction is therefore 0.4 % high (measured 0.124350 against the
    consistent 0.123854 for the Ni detector with the circle).  It is
    kept because ``IndexEBSD``'s resampled sizes must match for the
    resample to be the same operation.
    """
    axis = np.arange(grid_res + 1, dtype=np.float64) / grid_res
    square_x, square_y = np.meshgrid(axis, axis, indexing="xy")
    square = np.stack([square_x.ravel(), square_y.ravel()], axis=-1)
    normals = _grid.square_to_sphere(square)
    col, row, in_front = _directions_to_pixels(normals, _pixel_map(detector))
    keep = in_front & (normals[:, 2] >= 0)
    keep &= _inside_detector(col, row, detector.shape, circular_mask, signal_mask)
    # The literal C++ divisor of line 413, 0.4 % small
    divisor = grid_res**2 + (grid_res - 2) ** 2
    return float(np.count_nonzero(keep) / divisor)


def _rescaled_shape(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    """Return the shape of the resampled pattern.

    Parameters
    ----------
    shape
        Detector shape ``(nrows, ncols)``.
    scale
        Product of the scale factor and the oversampling.

    Returns
    -------
    rescaled_shape
        ``(h_out, w_out)`` with ``floor(scale nrows + 0.5)`` and
        ``floor(scale ncols + 0.5)``.

    Notes
    -----
    Size arithmetic of ``image::Rescaler<Real>::Rescaler()``
    (``include/util/image.hpp``, line 169), which rounds with
    ``std::round``, i.e. half away from zero.  Python's :func:`round`
    is banker's rounding and would give a different size for a
    detector whose product ends in ``.5``.
    """
    nrows, ncols = shape
    h_out = int(math.floor(scale * nrows + 0.5))
    w_out = int(math.floor(scale * ncols + 0.5))
    return h_out, w_out


def _build_lut(
    detector: "EBSDDetector",
    dim: int,
    circular_mask: bool,
    signal_mask: np.ndarray | None,
    rescaled_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the gather lookup table of a detector on a square
    Legendre grid.

    Parameters
    ----------
    detector
        Detector with a single projection centre.
    dim
        Side length of the square Legendre grid.
    circular_mask, signal_mask
        As in :func:`_inside_detector`.
    rescaled_shape
        ``(h_out, w_out)`` of :func:`_rescaled_shape`.

    Returns
    -------
    sphere_index
        ``(n_points,)`` 64-bit integer flat indices into the
        ``(dim, dim)`` **north** grid, in the C++ flat order
        ``i = y dim + x``.
    pixel_index
        ``(n_points, 4)`` 64-bit integer flat indices into the
        ``(h_out, w_out)`` resampled image.
    weights
        ``(n_points, 4)`` 64-bit float bilinear weights, rows
        summing to one.
    solid_angles
        ``(n_points,)`` 64-bit float ring solid angles, the C++
        ``omeg``.
    window_solid_angle
        Sum of ``solid_angles``, the C++ ``omgW``.
    sphere_solid_angle
        Solid angle of the whole two hemisphere grid with the
        equator counted once, the C++ ``omgS``, which equals
        ``2 dim^2 - 4 (dim - 1)`` because the ring values are
        relative to the average pixel.

    Notes
    -----
    Port of ``BackProjector<Real>::Constants``
    (``include/modality/ebsd/detector.hpp``, lines 502-569) with
    ``image::BiPix<Real>::bilinearCoeff()``
    (``include/util/image.hpp``, lines 526-551).  A north grid
    normal enters the table when its sample frame ``z`` component is
    non-negative (the physical guard, which rejects nothing on the
    north grid but is what makes the south set empty), when it is in
    front of the detector and when :func:`_inside_detector` accepts
    its pixel coordinates.  The south loop of the C++ is unreachable
    and not ported.

    The continuous position in the resampled image is
    ``x = (col + 0.5) w_out / ncols - 0.5`` and ``y`` likewise, the
    sampling convention of the type-2/type-3 discrete cosine
    transform, and **not** EMSphInx' ``X (w_out - 1)``.  The
    neighbours are ``i0 = clip(floor(x), 0, w_out - 1)``,
    ``i1 = min(i0 + 1, w_out - 1)`` and the same for ``y``, with the
    weights ``wx1 = clip(x - i0, 0, 1)``, ``wx0 = 1 - wx1``: the
    C++'s ``min(., w - 1)`` clamp extended to the left and top rim,
    so a point in the half pixel rim takes the edge pixel's value
    along that axis.  Measured, 1.8-2.8 % of the default window sits
    in that rim.
    """
    nrows, ncols = detector.shape
    h_out, w_out = rescaled_shape

    normals = _grid.legendre_normals(dim).reshape(-1, 3)
    col, row, in_front = _directions_to_pixels(normals, _pixel_map(detector))
    # ``if(std::signbit(n[2])) return false``, line 336, which no
    # north grid normal trips but which empties the south set
    keep = in_front & (normals[:, 2] >= 0)
    keep &= _inside_detector(col, row, detector.shape, circular_mask, signal_mask)
    sphere_index = np.flatnonzero(keep).astype(np.int64)

    # The sampling convention of the type-2/type-3 transform, which
    # preserves the physical extent
    x = (col[sphere_index] + 0.5) * w_out / ncols - 0.5
    y = (row[sphere_index] + 0.5) * h_out / nrows - 0.5
    i0 = np.clip(np.floor(x), 0, w_out - 1).astype(np.int64)
    j0 = np.clip(np.floor(y), 0, h_out - 1).astype(np.int64)
    i1 = np.minimum(i0 + 1, w_out - 1)
    j1 = np.minimum(j0 + 1, h_out - 1)
    wx1 = np.clip(x - i0, 0, 1)
    wy1 = np.clip(y - j0, 0, 1)
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    pixel_index = np.stack(
        [j0 * w_out + i0, j0 * w_out + i1, j1 * w_out + i0, j1 * w_out + i1],
        axis=-1,
    )
    weights = np.stack([wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1], axis=-1)

    rings = _grid.ring_number(dim)
    ring_values = _grid.ring_solid_angles(dim, "legendre")
    solid_angles = ring_values[rings.ravel()[sphere_index]]
    window_solid_angle = float(solid_angles.sum())

    # ``omgS`` of lines 560-568: every grid point once, the interior
    # ones twice, so that the equator is not counted twice
    grid_angles = ring_values[rings]
    equator = np.zeros((dim, dim), dtype=bool)
    equator[0] = equator[-1] = True
    equator[:, 0] = equator[:, -1] = True
    sphere_solid_angle = float(grid_angles.sum() + grid_angles[~equator].sum())

    return (
        sphere_index,
        pixel_index,
        weights,
        solid_angles,
        window_solid_angle,
        sphere_solid_angle,
    )


# ------------------- Resample and image quality --------------------- #


def _mean_fill(pattern: np.ndarray, signal_mask: np.ndarray | None) -> np.ndarray:
    """Return a pattern whose masked pixels hold the mean of the
    unmasked ones.

    Parameters
    ----------
    pattern
        ``(nrows, ncols)`` 64-bit float pattern, modified in place
        and returned.
    signal_mask
        Optional ``(nrows, ncols)`` boolean mask in kikuchipy
        polarity, ``True`` = ignore the pixel.  Nothing is done when
        it is ``None``.

    Returns
    -------
    pattern
        The same array.

    Notes
    -----
    Not in EMSphInx, which has no per-pixel mask in the projector.
    The resample is a global spectral operation, so a dead or
    saturated masked pixel rings into its neighbours: measured on a
    ``12 x 15`` block of a ``Re Y_4^2`` pattern, the windowed
    correlation with the truth is 0.981 with the block zeroed and
    0.932 with it saturated, against 0.997 with the fill.
    """
    if signal_mask is None:
        return pattern
    pattern[signal_mask] = pattern[~signal_mask].mean()
    return pattern


def _image_quality_from_spectrum(spectrum: np.ndarray) -> float:
    """Return the image quality of a discrete cosine spectrum.

    Parameters
    ----------
    spectrum
        ``(h, w)`` unnormalised type-2 discrete cosine spectrum of a
        pattern, of the **input** size.

    Returns
    -------
    image_quality
        ``1 - sum(p r2) / (sum(r2) sum(p) / (h w))`` with
        ``p = |spectrum|`` and ``r2[j, i] = j^2 + i^2``, or ``0.0``
        when ``sum(p)`` is zero.

    Notes
    -----
    Port of ``image::imageQuality()`` (``include/util/image.hpp``,
    lines 489-507).  The magnitudes are unnormalised and the DC term
    is included, so this statistic is offset dependent and is not
    :func:`kikuchipy.pattern.get_image_quality`, the Krieger Lassen
    quality of a normalised pattern (measured correlation 0.62 over
    the nine ``nickel_ebsd_small`` patterns).
    """
    magnitude = np.abs(np.asarray(spectrum, dtype=np.float64))
    height, width = magnitude.shape
    r2 = (
        np.arange(height)[:, np.newaxis] ** 2 + np.arange(width)[np.newaxis, :] ** 2
    ).astype(np.float64)
    total = magnitude.sum()
    # ``if(sumP == Real(0)) vIq = 0``, line 504
    if total == 0:
        return 0.0
    return float(1 - (magnitude * r2).sum() / (r2.sum() * total / (width * height)))


def _dct_image_quality(pattern: np.ndarray) -> float:
    """Return the discrete cosine image quality of a pattern.

    Parameters
    ----------
    pattern
        ``(h, w)`` array-like, cast to 64-bit float.

    Returns
    -------
    image_quality
        :func:`_image_quality_from_spectrum` of
        ``dctn(pattern, type=2)``, short-cut to ``1.0`` for a
        non-zero constant pattern and ``0.0`` for the all-zero one.

    Notes
    -----
    ``image::imageQuality()`` (``include/util/image.hpp``, lines
    489-507) preceded by the ``ptp == 0`` short-cut of
    :meth:`SphericalBackProjector.unproject`, so that the two agree
    by construction on a constant pattern.  The short-cut returns
    the exact values of the literal transcription, which are 1.0 for
    the constants 1, 7, 37 and 200 despite pocketfft's inexact AC
    terms and 0.0 for the all-zero spectrum (its ``sumP == 0``
    branch, line 504).

    **This is the only image quality function of the phase.**
    ``_preprocessing`` defines none and imports no
    :mod:`scipy.fft`, so one call recording test on ``dctn`` in this
    namespace covers every transform the phase makes.
    """
    pattern = np.asarray(pattern, dtype=np.float64)
    if np.ptp(pattern) == 0:
        return 1.0 if pattern.flat[0] != 0 else 0.0
    return _image_quality_from_spectrum(dctn(pattern, type=2, workers=1))


def _dct_rescale(
    pattern: np.ndarray,
    h_out: int,
    w_out: int,
    *,
    zero_mean: bool = True,
    want_iq: bool = False,
) -> tuple[np.ndarray, float]:
    """Return a pattern resampled with the discrete cosine
    transform, and optionally its image quality.

    Parameters
    ----------
    pattern
        ``(h_in, w_in)`` 64-bit float pattern.
    h_out, w_out
        Shape of the resampled pattern, at least one each.
    zero_mean
        Whether to zero the DC term of the truncated spectrum,
        ``True`` by default.
    want_iq
        Whether to evaluate :func:`_image_quality_from_spectrum` on
        the **input size** spectrum, before the truncation, ``False``
        by default.

    Returns
    -------
    rescaled
        Fresh ``(h_out, w_out)`` 64-bit float array.
    image_quality
        The image quality when ``want_iq``, else ``0.0``.

    Notes
    -----
    Port of ``image::Rescaler<Real>::scale()``
    (``include/util/image.hpp``, lines 582-618): an unnormalised
    type-2 transform (FFTW's ``REDFT10``), the low frequency corner
    ``[:min(h_in, h_out), :min(w_in, w_out)]`` copied into a zero
    array, the DC term zeroed and an unnormalised type-3 transform
    (``REDFT01``), **never** :func:`scipy.fft.idctn`, whose
    normalisation would scale the result by ``1 / (4 h_out w_out)``.
    No amplitude correction is applied: the round trip factor
    ``4 h_in w_in`` cancels in the unit variance normalisation of
    :func:`_unproject_kernel`.

    The high pass filter of lines 605-616 is not ported, since
    ``IndexEBSD`` always passes ``flt = 0``.

    Phase 2's :func:`~kikuchipy.indexing._spherical.
    _master_pattern_harmonics._resize_lambert` is the square,
    amplitude carrying sibling of this helper and is deliberately
    left untouched: its caller keeps amplitudes and applies
    EMSphInx' ``0.5 / new_dim^2``.
    """
    pattern = np.asarray(pattern, dtype=np.float64)
    h_in, w_in = pattern.shape
    spectrum = dctn(pattern, type=2, workers=1)
    # On the input size spectrum, before the truncation, lines
    # 584-585
    image_quality = _image_quality_from_spectrum(spectrum) if want_iq else 0.0
    truncated = np.zeros((h_out, w_out))
    h_copy = min(h_in, h_out)
    w_copy = min(w_in, w_out)
    truncated[:h_copy, :w_copy] = spectrum[:h_copy, :w_copy]
    if zero_mean:
        truncated[0, 0] = 0.0
    return dctn(truncated, type=3, workers=1), image_quality


# ----------------------- The back-projector ------------------------- #


class SphericalBackProjector:
    """Back-projection of detector patterns onto a square Legendre
    grid on the unit sphere.

    Parameters
    ----------
    detector
        EBSD detector with exactly one projection centre
        (``detector.navigation_size == 1``) and zero ``azimuthal``
        and ``twist`` angles.  It is deep copied, so later mutations
        of the caller's detector do not reach the projector.
    bandwidth
        Bandwidth of the spherical harmonic transform, at least one.
    signal_mask
        Optional boolean mask of shape ``detector.shape`` in
        kikuchipy polarity, ``True`` = ignore the pixel, as in
        :meth:`~kikuchipy.signals.EBSD.dictionary_indexing`.  It
        excludes grid points by nearest pixel and mean fills the
        masked pixels before the resample.  ``None`` by default.
    circular_mask
        Whether to keep only the largest circle inscribed in the
        physical detector, ``False`` by default, as ``IndexEBSD``'s
        namelist ``circmask = -1``.  An explicit radius is expressed
        through ``signal_mask``, e.g.
        ``~_preprocessing._circular_mask(shape, radius)``.
    oversampling
        Linear oversampling of the resampled detector relative to
        the average spherical grid pixel, ``sqrt(2)`` by default,
        the C++ ``fct``.  Larger than zero.
    dim
        Side length of the square Legendre grid.  Defaults to
        :func:`~kikuchipy.indexing._spherical._grid.default_dim` of
        the bandwidth, i.e. ``bandwidth + 2`` if that is odd and
        ``bandwidth + 3`` otherwise.

    Attributes
    ----------
    detector : kikuchipy.detectors.EBSDDetector
        Deep copy of the detector, with the projection centre
        reshaped to ``(1, 3)`` so that kikuchipy's own geometry takes
        its fixed projection centre path.
    bandwidth : int
        Exclusive maximum harmonic degree.
    dim : int
        Side length of the square Legendre grid.
    sht : kikuchipy.indexing._spherical._sht.SphericalHarmonicTransform
        Transform of that bandwidth, layout and side length.
    signal_mask : numpy.ndarray or None
        Copy of the mask, in kikuchipy polarity.
    circular_mask : bool
        Whether the physical circle is applied.
    oversampling : float
        The C++ ``fct``.
    solid_angle_fraction : float
        Fraction of the sphere the detector covers,
        :func:`_solid_angle_fraction`.
    scale_factor : float
        ``sqrt(solid_angle_fraction (2 dim^2 - 4 (dim - 1)) /
        (ncols nrows))``, the C++ ``Geometry::scaleFactor()``.
    rescaled_shape : tuple of int
        ``(h_out, w_out)`` of the resampled pattern.
    n_points : int
        Number of window points, i.e. the length of the lookup
        table.
    sphere_index : numpy.ndarray
        ``(n_points,)`` 64-bit integer flat indices into the
        ``(dim, dim)`` north grid.
    pixel_index : numpy.ndarray
        ``(n_points, 4)`` 64-bit integer flat indices into the
        resampled image.
    weights : numpy.ndarray
        ``(n_points, 4)`` 64-bit float bilinear weights.
    solid_angles : numpy.ndarray
        ``(n_points,)`` 64-bit float ring solid angles, the C++
        ``omeg``.
    window_solid_angle : float
        Sum of :attr:`solid_angles`, the C++ ``omgW``.
    sphere_solid_angle : float
        Solid angle of the whole grid, the C++ ``omgS``, equal to
        ``2 dim^2 - 4 (dim - 1)``.
    window_fraction : float
        ``window_solid_angle / sphere_solid_angle``.
    window_harmonics : numpy.ndarray
        ``(bw, bw)`` 128-bit complex harmonic coefficients of
        :meth:`window_mask`, the C++ ``mlm``, **computed at
        construction** so that the instance is immutable and
        shareable between threads.

    Raises
    ------
    TypeError
        If ``detector`` is not an
        :class:`~kikuchipy.detectors.EBSDDetector`.
    ValueError
        If ``bandwidth`` is smaller than one; if ``dim`` is rejected
        by :class:`~kikuchipy.indexing._spherical._sht.
        SphericalHarmonicTransform`; if the detector has more than
        one projection centre; if its ``azimuthal`` or ``twist``
        angle is non-zero; if ``signal_mask`` is not boolean of the
        detector shape; if ``oversampling`` is not positive; if the
        resampled pattern would be smaller than one pixel in either
        direction; or if no grid point falls on the detector.

    Notes
    -----
    Port of ``BackProjector<Real>``
    (``include/modality/ebsd/detector.hpp``, lines 502-630) with
    ``Geometry<Real>::solidAngle()``, ``scaleFactor()`` and the size
    arithmetic of ``rescale()``, and of ``image::Rescaler<Real>``,
    ``image::imageQuality()`` and ``image::BiPix<Real>``
    (``include/util/image.hpp``).  See the module documentation for
    the direction to pixel chain, the frames, the sizes, the
    normalisation, the measured convention lock and the speed.

    An instance is **immutable after construction and thread safe**:
    :meth:`unproject` reads the lookup table and writes only the
    caller's buffers and per-call temporaries, so Phase 6 shares one
    projector across its dask threads.  ``BackProjector::clone()``
    is therefore not ported and no ``clone()`` is offered; sharing
    matters, since the transform's tables are 5.5 MB at ``bw`` 68
    and 23.1 MB at ``bw`` 113.

    Deviations from EMSphInx, all measured:

    - EMSphInx' pixel position stretch ``x = X (w - 1)`` is not
      reproduced (up to 0.49 px); the pixel centre convention of
      kikuchipy's forward projection is used with the transform's
      own sampling convention.
    - The south hemisphere is never gathered, since the C++ south
      loop is unreachable behind the physical guard.  ``south`` is
      always zero.
    - There is no ``flip`` parameter: kikuchipy patterns have one
      row convention, row 0 at the top.
    - :meth:`window_mask` is built directly from the lookup table
      rather than by unprojecting a constant pattern, because
      pocketfft's type-2 transform of a constant is inexact, and a
      constant pattern is detected with ``ptp == 0`` before the
      transform (image quality ``1.0`` for a non-zero constant,
      ``0.0`` for the all-zero pattern).
    - ``signal_mask`` pixels are mean filled before the resample.
    - The rim of the window is filled by constant extrapolation off
      the pixel centre box (1.8 % of the default window at ``bw`` 68
      and 2.8 % at ``bw`` 113).
    - The circle is the physical circle of the unrescaled detector,
      while the C++ lookup table loop uses the rescaled geometry: 5
      of 958 points differ on a ``(48, 60)`` detector, none on a
      square one.
    - Two empty window guards are raised where the C++ would fail
      inside FFTW instead.
    - The C++ return value ``var`` of ``unproject()`` is not
      returned; :attr:`window_fraction` exposes ``omgW / omgS``.

    Quirks kept faithfully: the ``stdev == 0`` branch of
    :func:`_unproject_kernel`, the binary mask assumption behind the
    correlator's ``s2m``, the ``omgS`` counting with the equator
    once and the ``solidAngle()`` divisor ``500002``.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> from kikuchipy.indexing._spherical._back_projection import (
    ...     SphericalBackProjector,
    ... )
    >>> detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    >>> detector.pc = detector.pc_average
    >>> projector = SphericalBackProjector(detector, 68)
    >>> projector
    SphericalBackProjector: bw = 68, dim = 71, detector
    (60, 60) -> (53, 53), window 1317 points (14.6 %)
    >>> round(projector.window_fraction, 3)
    0.146
    >>> north, south = projector.unproject(
    ...     kp.data.nickel_ebsd_small().data[0, 0]
    ... )
    >>> north.shape, south.shape
    ((71, 71), (71, 71))
    """

    def __init__(
        self,
        detector: "EBSDDetector",
        bandwidth: int,
        *,
        signal_mask: np.ndarray | None = None,
        circular_mask: bool = False,
        oversampling: float = _DEFAULT_OVERSAMPLING,
        dim: int | None = None,
    ) -> None:
        bandwidth = int(bandwidth)
        if bandwidth < 1:
            raise ValueError(f"Bandwidth {bandwidth} must be at least one")
        if dim is None:
            dim = _grid.default_dim(bandwidth, "legendre")
        dim = int(dim)
        sht = SphericalHarmonicTransform(bandwidth, "legendre", dim)

        if not isinstance(detector, EBSDDetector):
            raise TypeError(
                f"Detector of type {type(detector)} must be an EBSDDetector"
            )
        n_pc = detector.navigation_size
        if n_pc != 1:
            raise ValueError(
                f"The detector has {n_pc} projection centres; "
                "back-projection uses one projection centre per call -- set "
                "`detector.pc = detector.pc_average` on a copy "
                "(`detector.deepcopy()`), or build a detector from a single "
                "projection centre"
            )
        if detector.azimuthal != 0 or detector.twist != 0:
            raise ValueError(
                f"The detector `azimuthal` angle {detector.azimuthal} and "
                f"`twist` angle {detector.twist} must both be zero, since "
                "tilted or twisted detectors are not supported yet"
            )

        if signal_mask is not None:
            signal_mask = np.asarray(signal_mask)
            if signal_mask.dtype != np.bool_:
                raise ValueError(
                    f"Signal mask of data type {signal_mask.dtype} must be "
                    "boolean, since `True` means ignore the pixel"
                )
            if signal_mask.shape != detector.shape:
                raise ValueError(
                    f"Signal mask of shape {signal_mask.shape} must have the "
                    f"detector shape {detector.shape}"
                )
            signal_mask = signal_mask.copy()

        oversampling = float(oversampling)
        if not oversampling > 0:
            raise ValueError(f"Oversampling {oversampling} must be greater than zero")

        # A copy, so that later mutations of the caller's detector do
        # not reach the projector, with the projection centre reshaped
        # so that kikuchipy's own geometry takes its fixed projection
        # centre path
        detector = detector.deepcopy()
        detector.pc = np.asarray(detector.pc, dtype=np.float64).reshape(1, 3)

        self.detector = detector
        self.bandwidth = bandwidth
        self.dim = dim
        self.sht = sht
        self.signal_mask = signal_mask
        self.circular_mask = bool(circular_mask)
        self.oversampling = oversampling

        nrows, ncols = detector.shape
        self.solid_angle_fraction = _solid_angle_fraction(
            detector, self.circular_mask, signal_mask
        )
        # ``Geometry::scaleFactor()``, lines 466-468
        square_points = 2 * dim**2 - 4 * (dim - 1)
        self.scale_factor = math.sqrt(
            self.solid_angle_fraction * square_points / (nrows * ncols)
        )
        self.rescaled_shape = _rescaled_shape(
            detector.shape, self.scale_factor * self.oversampling
        )
        h_out, w_out = self.rescaled_shape
        if h_out < 1 or w_out < 1:
            raise ValueError(
                "The detector footprint covers no part of the northern "
                "hemisphere or `signal_mask` leaves nothing: "
                f"solid_angle_fraction = {self.solid_angle_fraction:.6f}, so "
                f"the resampled pattern would be {w_out} x {h_out} pixels"
            )

        (
            self.sphere_index,
            self.pixel_index,
            self.weights,
            self.solid_angles,
            self.window_solid_angle,
            self.sphere_solid_angle,
        ) = _build_lut(
            detector, dim, self.circular_mask, signal_mask, self.rescaled_shape
        )
        self.n_points = int(self.sphere_index.size)
        if self.n_points == 0:
            raise ValueError(
                "The detector or `signal_mask` leaves an empty window: no "
                f"Legendre grid point of dim {dim} falls on the detector"
            )
        self.window_fraction = self.window_solid_angle / self.sphere_solid_angle
        # Eager, so that the instance holds no mutable state and may
        # be shared read only between threads
        self.window_harmonics = sht.analyze(*self.window_mask())

    def __repr__(self) -> str:
        """Return a string with the bandwidth, the grid side length,
        the detector and resampled shapes and the window, e.g.
        ``"SphericalBackProjector: bw = 68, dim = 71, detector
        (60, 60) -> (53, 53), window 1317 points (14.6 %)"``.
        """
        nrows, ncols = self.detector.shape
        h_out, w_out = self.rescaled_shape
        return (
            f"{type(self).__name__}: bw = {self.bandwidth}, "
            f"dim = {self.dim}, detector ({nrows}, {ncols}) -> "
            f"({h_out}, {w_out}), window {self.n_points} points "
            f"({100 * self.window_fraction:.1f} %)"
        )

    def window_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the binary mask of the window on the two grids.

        Returns
        -------
        north
            Fresh ``(dim, dim)`` 64-bit float array which is ``1``
            on the window points and ``0`` elsewhere.
        south
            Fresh ``(dim, dim)`` 64-bit float array of zeros.

        Notes
        -----
        Port of ``BackProjector<Real>::mask()``
        (``include/modality/ebsd/detector.hpp``, lines 628-630).
        The C++ ``Indexer`` builds this mask by unprojecting a
        constant pattern (``include/idx/idx.hpp``, lines 266-268),
        which pocketfft's inexact type-2 transform of a constant
        makes unusable here, so it is built directly from the lookup
        table.  The ``ptp == 0`` branch of :meth:`unproject` returns
        the same array bitwise.
        """
        north = np.zeros((self.dim, self.dim))
        north.reshape(-1)[self.sphere_index] = 1.0
        south = np.zeros((self.dim, self.dim))
        return north, south

    def squared_harmonics(self, alm: np.ndarray) -> np.ndarray:
        """Return the harmonic coefficients of the square of a
        spherical function on this projector's grid.

        Parameters
        ----------
        alm
            ``(bw, bw)`` 128-bit complex harmonic coefficients of the
            master pattern, already resized to this projector's
            bandwidth.

        Returns
        -------
        flm2
            Fresh ``(bw, bw)`` 128-bit complex coefficients of the
            squared function, i.e.
            ``sht.analyze(north**2, south**2)`` of
            ``sht.synthesize(alm)``.

        Raises
        ------
        ValueError
            If ``alm`` does not have shape ``(bw, bw)``.

        Notes
        -----
        Port of the ``flm2`` lines of ``Indexer::Indexer()``
        (``include/idx/idx.hpp``, lines 276-280).  The result
        depends on the projector's grid, not on the master pattern
        alone, which is why it lives here and not on
        :class:`~kikuchipy.indexing._spherical.
        _master_pattern_harmonics.MasterPatternHarmonics`.  Together
        with :attr:`window_harmonics` it is what
        :class:`~kikuchipy.indexing._spherical._xcorr.
        NormalizedSphericalCrossCorrelator` needs to build ``rDen``.
        """
        alm = np.asarray(alm)
        expected = (self.bandwidth, self.bandwidth)
        if alm.shape != expected:
            raise ValueError(
                f"Harmonic coefficients of shape {alm.shape} must have the "
                f"projector's shape {expected}: resize the master pattern "
                "harmonics to this bandwidth first"
            )
        north, south = self.sht.synthesize(alm)
        return self.sht.analyze(north**2, south**2)

    def image_quality(self, pattern: np.ndarray) -> float:
        """Return the discrete cosine image quality of a pattern.

        Parameters
        ----------
        pattern
            ``(nrows, ncols)`` array-like, cast to 64-bit float.

        Returns
        -------
        image_quality
            :func:`_dct_image_quality` of the pattern, equal bitwise
            to the value :meth:`unproject` returns for the same
            input with ``return_image_quality=True``.

        Raises
        ------
        ValueError
            If ``pattern`` does not have the detector shape.

        Notes
        -----
        The pattern is mean filled exactly as :meth:`unproject` fills
        it when :attr:`signal_mask` is set, so that the two agree
        bitwise for a masked projector as well.
        """
        pattern = np.asarray(pattern)
        if pattern.shape != self.detector.shape:
            raise ValueError(
                f"Pattern of shape {pattern.shape} must have the detector "
                f"shape {self.detector.shape}"
            )
        work = _mean_fill(pattern.astype(np.float64), self.signal_mask)
        return _dct_image_quality(work)

    def unproject(
        self,
        pattern: np.ndarray,
        *,
        out: tuple[np.ndarray, np.ndarray] | None = None,
        return_image_quality: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Return a pattern back-projected onto the square Legendre
        grid, zero mean and unit variance on the window.

        Parameters
        ----------
        pattern
            Real 2-D array-like of the detector shape, cast to
            64-bit float in a copy, so an unsigned 8-bit pattern is
            safe and the input is never modified.
        out
            Optional ``(north, south)`` pair of C-contiguous
            ``(dim, dim)`` 64-bit float buffers to write into, which
            Phase 6 reuses per thread.  Fresh zero arrays are
            allocated when it is not given.
        return_image_quality
            Whether to evaluate the image quality on the input size
            spectrum and return it as a third element, ``False`` by
            default.

        Returns
        -------
        north
            ``(dim, dim)`` 64-bit float north grid.  **Only the
            window points are written**, the C++ contract, so a
            reused buffer keeps its stale values off the window.
        south
            ``(dim, dim)`` 64-bit float south grid, always zero:
            the footprint below the sample plane is clipped, not
            wrapped.
        image_quality
            Only when ``return_image_quality``.

        Raises
        ------
        ValueError
            If ``pattern`` does not have the detector shape, or if
            ``out`` is not a pair of C-contiguous ``(dim, dim)``
            64-bit float arrays.

        Notes
        -----
        Port of ``BackProjector<Real>::unproject()``
        (``include/modality/ebsd/detector.hpp``, lines 589-623): the
        mean fill of a masked pattern, :func:`_dct_rescale` and
        :func:`_unproject_kernel`.  A constant pattern is detected
        with ``ptp == 0`` **before** the transform and returns
        :meth:`window_mask` with an image quality of ``1.0`` when it
        is non-zero and ``0.0`` when it is not.

        NaN in the pattern is not guarded and propagates onto the
        window.  All temporaries are per call, so one projector may
        be shared between threads.
        """
        pattern = np.asarray(pattern)
        if pattern.shape != self.detector.shape:
            raise ValueError(
                f"Pattern of shape {pattern.shape} must have the detector "
                f"shape {self.detector.shape}"
            )

        dim = self.dim
        if out is None:
            north = np.zeros((dim, dim))
            south = np.zeros((dim, dim))
        else:
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                raise ValueError(
                    "Output buffers `out` must be a pair of C-contiguous "
                    f"({dim}, {dim}) 64-bit float arrays"
                )
            north, south = out
            for buffer in (north, south):
                if (
                    not isinstance(buffer, np.ndarray)
                    or buffer.shape != (dim, dim)
                    or buffer.dtype != np.float64
                    or not buffer.flags.c_contiguous
                ):
                    raise ValueError(
                        "Output buffers `out` must be a pair of C-contiguous "
                        f"({dim}, {dim}) 64-bit float arrays, not an array "
                        f"of shape {np.shape(buffer)}"
                    )

        work = _mean_fill(pattern.astype(np.float64), self.signal_mask)
        north_flat = north.reshape(-1)
        # pocketfft's type-2 transform of a constant is not exactly a
        # delta, so the constant is caught before the transform
        if np.ptp(work) == 0:
            north_flat[self.sphere_index] = 1.0
            image_quality = 1.0 if work.flat[0] != 0 else 0.0
        else:
            h_out, w_out = self.rescaled_shape
            rescaled, image_quality = _dct_rescale(
                work,
                h_out,
                w_out,
                zero_mean=True,
                want_iq=return_image_quality,
            )
            _unproject_kernel(
                rescaled.reshape(-1),
                self.pixel_index,
                self.weights,
                self.solid_angles,
                self.window_solid_angle,
                self.sphere_index,
                north_flat,
            )

        if return_image_quality:
            return north, south, image_quality
        return north, south
