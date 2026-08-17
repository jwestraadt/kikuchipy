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

"""Tests of ``kikuchipy.indexing._spherical._back_projection``.

Covers every named assertion of
``specs/2026-08-17-spherical-back-projection/validation.md`` which
belongs to the back-projector:

- Construction and guards: the attribute table at ``bw`` 68 with and
  without the circle, the eager ``window_harmonics``, the single
  projection centre guard on ``navigation_size``, the
  ``azimuthal``/``twist`` guard, the two empty-window guards and the
  ``signal_mask`` validation.
- Geometry: the direction oracle against
  ``_get_direction_cosines_from_detector`` on four detectors, the
  reverse oracle onto the Legendre normals, the frame statement,
  binning neutrality, the EMsoft round trip and the always-zero
  south hemisphere.
- Sizes and the lookup table: the ``bw -> (h_out, w_out)`` table,
  ``floor(x + 0.5)`` against banker's rounding, the ``solidAngle()``
  divisor quirk, ``omgS == 2 dim^2 - 4 (dim - 1)``, the rim
  structure and the **structural pin of the resample map** on three
  geometries -- the Ni detector at ``bw`` 68 and 88 and a
  rectangular one, which between them fill all four rims and give
  the only non-square map -- with EMSphInx' pixel stretch as the
  negative control, the nearest-pixel ``signal_mask`` exclusion and
  the rectangular circle pin.
- ``unproject``: the ``dctn`` convention regression and the
  ``idctn`` negative control first, the constant-pattern short-cut,
  the literal ``stdev == 0`` branch and its tiny-amplitude control,
  the weighted normalisation identities, ``out=`` reuse and the
  transform call recording.
- The discrete cosine image quality, which lives in this module.
- Spherical harmonic recovery, the mean fill, ``mlm``/``flm2`` and
  the effect of ``signal_mask`` on ``rDen``.
- **The forward-projection convention lock**: 27
  ``get_patterns()`` rotations through both correlators and both
  signs, and the asymmetric-blob row/column/flip check.
- Real data: the nine ``nickel_ebsd_small`` patterns and the
  measured mean projection centre error floor on the small map and
  on ``nickel_ebsd_large`` subsets.
- ``py_func`` of the kernel, the Numba compilation flags and
  recorded timing and memory baselines.
"""

import functools
import inspect
import itertools
import math
import time
import tracemalloc

import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest
import scipy.fft

import kikuchipy as kp
from kikuchipy._constants import dependency_version
from kikuchipy.detectors import EBSDDetector
from kikuchipy.indexing._spherical import (
    _back_projection,
    _euler,
    _grid,
    _preprocessing,
    _xcorr,
)
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
)

EPS = float(np.finfo(np.float64).eps)

# The bandwidth of the real data tests and its Legendre side length
NI_BANDWIDTH = 68
NI_DIM = 71

# Every Numba kernel of the module, for the flag and py_func tests
KERNEL_NAMES = ["_unproject_kernel"]

# Measured resampled side length and window size of the Ni detector,
# ``bw -> (circle, no circle)`` (D3)
RESCALED_SIDES = {53: (38, 41), 63: (45, 49), 68: (49, 53), 88: (63, 69), 113: (80, 87)}
WINDOW_POINTS = {
    53: (667, 788),
    63: (934, 1093),
    68: (1117, 1317),
    88: (1844, 2157),
    113: (2955, 3474),
}

# Measured ``Geometry::scaleFactor()`` of the Ni detector with the
# circle, ``bw -> scale_factor``
SCALE_FACTORS = {53: 0.448905, 68: 0.581873, 88: 0.748092}

# Measured ``Geometry::solidAngle(501)`` of the Ni detector, as the
# integer count of accepted grid directions over the literal C++
# divisor ``501^2 + 499^2``
SOLID_ANGLE_COUNTS = {True: 62175, False: 72856}
SOLID_ANGLE_DIVISOR = 501**2 + 499**2

# Measured window fractions of the Ni detector at ``bw`` 68
WINDOW_FRACTIONS = {True: 0.123997, False: 0.145851}
SOLID_ANGLE_FRACTIONS = {True: 0.124350, False: 0.145711}

# Measured ``rDen`` extrema at ``bw`` 68 (D7)
R_DEN_LIMITS = {True: (1.785, 2.150), False: (1.660, 1.951)}

# Measured interpolated peak scores of the 27 rotation lock at
# ``bw`` 68 without the circle (D8)
LOCK_SCORES = {"normalized": (1.087, 1.143), "plain": (0.594, 0.652)}

# The Bunge grid of the forward-projection lock, in degrees (D8)
LOCK_EULER_DEGREES = list(
    itertools.product([20.0, 140.0, 260.0], [30.0, 80.0, 130.0], [10.0, 130.0, 250.0])
)

# The ``12 x 15`` signal mask block of D5 and D7
MASK_BLOCK = (slice(20, 32), slice(25, 40))

# Synthetic detectors of the direction oracle, ``(shape, pc,
# sample_tilt, tilt)``
SYNTHETIC_DETECTORS = [
    ((48, 60), (0.55, 0.65, 0.6), 70.0, 10.0),
    ((60, 60), (0.5, 0.5, 0.5), 70.0, -30.0),
    ((41, 41), (0.3, 0.8, 0.45), 65.0, 0.0),
]

# Geometries of the rim and resample map pins, ``id -> (detector
# case, bandwidth)`` with ``None`` for the Ni detector.  ``bw`` 88
# without the circle is the only one which puts LUT points on the
# **bottom** rim, and the rectangular detector the only one whose
# resample map is not square, so a row/column swap inside it and a
# missing ``j1`` clamp are invisible on the Ni detector at ``bw`` 68
LUT_GEOMETRIES = {
    "ni_bw68": (None, 68),
    "ni_bw88": (None, 88),
    "rect_bw68": (SYNTHETIC_DETECTORS[0], 68),
}

# Measured rim counts ``(left, top, right, bottom)`` in resampled
# coordinates of every geometry above (D3).  Every side is non-empty
# in at least one of them, which
# :meth:`TestLUT.test_every_rim_group_is_exercised` re-asserts
RIM_COUNTS = {
    ("ni_bw68", True): (2, 0, 1, 0),
    ("ni_bw68", False): (10, 10, 11, 0),
    ("ni_bw88", True): (1, 4, 2, 0),
    ("ni_bw88", False): (11, 12, 11, 8),
    ("rect_bw68", True): (0, 1, 0, 0),
    ("rect_bw68", False): (6, 8, 5, 12),
}
RIM_SIDES = ("left", "top", "right", "bottom")

# Measured north window counts of the tilt table (D4), no circle
TILT_TABLE = {
    (70.0, 0.0): 1317,
    (70.0, 10.0): 1375,
    (70.0, -30.0): 1235,
    (0.0, 0.0): 524,
    (20.0, 0.0): 901,
}


# ----------------------------- Helpers ------------------------------ #


def _njit_kernel_names(module):
    """Return the names of the module's own Numba kernels.

    Only dispatchers whose Python function is defined in the module
    itself are returned, so that a kernel imported from another
    module of the package is not counted.
    """
    return sorted(
        name
        for name, value in vars(module).items()
        if type(value).__name__ == "CPUDispatcher"
        and getattr(value, "py_func", None) is not None
        and value.py_func.__module__ == module.__name__
    )


def _py_func(kernel):
    """Return the pure Python function of a Numba kernel.

    Falls back to the function itself while it is still an
    undecorated stub. Every caller first asserts that the kernel does
    carry a ``py_func``, so that an implementation without ``@njit``
    fails loudly instead of silently comparing a function to itself.
    """
    return getattr(kernel, "py_func", kernel)


@functools.lru_cache(maxsize=1)
def _ni_signal_data():
    """Return the nine ``nickel_ebsd_small`` patterns, cached.

    The array is made **read-only**: it is passed straight into the
    module, so an in-place write there would otherwise corrupt every
    later test instead of failing where it happens.
    """
    data = kp.data.nickel_ebsd_small().data
    data.flags.writeable = False
    return data


def ni_detector():
    """Return the Ni detector of the spec: a deep copy of
    ``nickel_ebsd_small().detector`` whose projection centre is the
    average of its nine.
    """
    detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    detector.pc = detector.pc_average
    return detector


def synthetic_detector(shape, pc, sample_tilt, tilt):
    """Return a detector of the given shape, projection centre and
    tilts.
    """
    return EBSDDetector(shape=shape, pc=pc, sample_tilt=sample_tilt, tilt=tilt)


def lut_geometry(name):
    """Return the detector and bandwidth of a :data:`LUT_GEOMETRIES`
    case.
    """
    case, bandwidth = LUT_GEOMETRIES[name]
    detector = ni_detector() if case is None else synthetic_detector(*case)
    return detector, bandwidth


@functools.lru_cache(maxsize=4)
def ni_master_harmonics(bandwidth):
    """Return the harmonics of the small Ni master pattern, cached
    per bandwidth because the transform costs about a second.
    """
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


def direction_cosines(detector):
    """Return the ``(nrows, ncols, 3)`` sample frame direction
    cosines of every detector pixel.
    """
    dc = _get_direction_cosines_from_detector(detector)
    return np.asarray(dc).reshape(detector.nrows, detector.ncols, 3)


def local_pixel_chain(detector, normals):
    """Return the continuous pixel coordinates of sample frame
    directions, computed here and not by the module.

    A second implementation of the spec's D2 chain, so that the
    module's own helpers are checked against something rather than
    against themselves.
    """
    matrix = detector.sample_to_detector.to_matrix().squeeze()
    bounds = np.asarray(detector.gnomonic_bounds).squeeze().astype(np.float64)
    x_scale = (bounds[1] - bounds[0]) / detector.ncols
    y_scale = (bounds[3] - bounds[2]) / detector.nrows
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    v = normals @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        x_gnomonic = v[:, 0] / v[:, 2]
        y_gnomonic = v[:, 1] / v[:, 2]
    col = (x_gnomonic - bounds[0]) / x_scale - 0.5
    row = (bounds[3] - y_gnomonic) / y_scale - 0.5
    return col, row, v[:, 2] > 0


def local_pixels_to_directions(detector, col, row):
    """Return the sample frame directions of continuous pixel
    coordinates.

    The arithmetic of ``_get_direction_cosines_for_fixed_pc``
    written for fractional pixels, i.e. the inverse of
    :func:`local_pixel_chain`.
    """
    bounds = np.asarray(detector.gnomonic_bounds).squeeze().astype(np.float64)
    pcz = float(np.asarray(detector.pcz).squeeze())
    x_scale = (bounds[1] - bounds[0]) / detector.ncols
    y_scale = (bounds[3] - bounds[2]) / detector.nrows
    x_gnomonic = bounds[0] + (np.asarray(col) + 0.5) * x_scale
    y_gnomonic = bounds[3] - (np.asarray(row) + 0.5) * y_scale
    v = np.stack(
        [x_gnomonic * pcz, y_gnomonic * pcz, np.full(np.shape(col), pcz)], axis=-1
    )
    matrix = (~detector.sample_to_detector).to_matrix().squeeze()
    v = v @ matrix.T
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def local_inside(detector, col, row, circular_mask, signal_mask=None):
    """Return whether continuous pixel coordinates fall on the
    unmasked detector, computed here and not by the module.
    """
    nrows, ncols = detector.shape
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
        dx = col - (ncols - 1) / 2
        dy = row - (nrows - 1) / 2
        inside = inside & (dx * dx + dy * dy <= radius * radius)
    if signal_mask is not None:
        columns = np.clip(np.floor(np.nan_to_num(col) + 0.5), 0, ncols - 1).astype(int)
        rows = np.clip(np.floor(np.nan_to_num(row) + 0.5), 0, nrows - 1).astype(int)
        inside = inside & ~signal_mask[rows, columns]
    return inside


def analytic_solid_angle_fraction(detector, circular_mask):
    """Return the pixel grid estimate of the fraction of the sphere
    the detector covers.

    The Riemann sum ``sum x_scale y_scale (1 + x_g^2 + y_g^2)^(-3/2)
    / (4 pi)`` over the pixels of the window, which is independent of
    both the C++ ``solidAngle()`` grid and the Legendre one.
    """
    nrows, ncols = detector.shape
    bounds = np.asarray(detector.gnomonic_bounds).squeeze().astype(np.float64)
    x_scale = (bounds[1] - bounds[0]) / ncols
    y_scale = (bounds[3] - bounds[2]) / nrows
    columns = np.arange(ncols)
    rows = np.arange(nrows)
    x_gnomonic = bounds[0] + (columns + 0.5) * x_scale
    y_gnomonic = bounds[3] - (rows + 0.5) * y_scale
    xg, yg = np.meshgrid(x_gnomonic, y_gnomonic)
    weight = (1 + xg**2 + yg**2) ** -1.5
    if circular_mask:
        col_grid, row_grid = np.meshgrid(columns.astype(float), rows.astype(float))
        keep = local_inside(detector, col_grid.ravel(), row_grid.ravel(), True)
        weight = weight * keep.reshape(nrows, ncols)
    return float(weight.sum() * x_scale * y_scale / (4 * math.pi))


def windowed_truth(projector, values):
    """Return values on the window made zero mean and unit variance
    with the ring solid angles as weights, i.e. what
    :meth:`SphericalBackProjector.unproject` writes.
    """
    omega = projector.solid_angles
    total = projector.window_solid_angle
    values = np.asarray(values, dtype=np.float64)
    mean = float((values * omega).sum() / total)
    centered = values - mean
    stdev = math.sqrt(float((centered * centered * omega).sum() / total))
    return centered / stdev


def weighted_moments(projector, north):
    """Return the solid angle weighted mean and second moment of the
    window values of a north grid.
    """
    values = np.asarray(north).ravel()[projector.sphere_index]
    omega = projector.solid_angles
    total = projector.window_solid_angle
    mean = float((values * omega).sum() / total)
    second = float((values * values * omega).sum() / total)
    return mean, second


def correlation(a, b):
    """Return the Pearson correlation of two flat arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def misorientation_deg(rotation, zyz, sign="inverse"):
    """Return the symmetry reduced misorientation in degrees between
    a rotation and the orientation of a correlated ZYZ triple.

    ``sign="inverse"`` uses ``_euler.rotation_from_zyz(zyz)``, i.e.
    ``~Rotation(zyz_to_quaternion(zyz))``, and ``sign="direct"`` the
    other candidate, which the lock must reject.
    """
    quaternion = Rotation(_euler.zyz_to_quaternion(np.asarray(zyz, dtype=np.float64)))
    candidate = ~quaternion if sign == "inverse" else quaternion
    angles = Orientation(candidate.data, Oh).angle_with(
        Orientation(rotation.data, Oh), degrees=True
    )
    return float(np.atleast_1d(angles).ravel()[0])


def angle_between_deg(a, b):
    """Return the angle in degrees between two unit vectors."""
    cosine = float(np.clip(np.dot(np.asarray(a), np.asarray(b)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def literal_image_quality(spectrum):
    """Return ``image::imageQuality()`` of a spectrum, transcribed
    here from ``include/util/image.hpp`` lines 489-507.

    A second implementation, so that an image quality evaluated on
    the *truncated* spectrum instead of the input size one is caught
    even though it would satisfy the equality between
    ``image_quality`` and ``unproject``.
    """
    magnitude = np.abs(np.asarray(spectrum, dtype=np.float64))
    height, width = magnitude.shape
    r2 = (np.arange(height)[:, None] ** 2 + np.arange(width)[None, :] ** 2).astype(
        np.float64
    )
    total = magnitude.sum()
    if total == 0:
        return 0.0
    return float(1 - (magnitude * r2).sum() / (r2.sum() * total / (width * height)))


def real_spherical_harmonic(normals, degree, order):
    """Return ``Re Y_l^m`` of unit vectors, using SciPy's
    ``sph_harm_y``, which needs SciPy 1.15 or newer.
    """
    from scipy.special import sph_harm_y

    normals = np.asarray(normals, dtype=np.float64)
    polar = np.arccos(np.clip(normals[..., 2], -1.0, 1.0))
    azimuth = np.arctan2(normals[..., 1], normals[..., 0])
    return np.real(sph_harm_y(degree, order, polar, azimuth))


# Positional order of :func:`scipy.fft.dctn`, so that a recorded
# call is read the same way however the module passes its arguments
DCTN_POSITIONAL = ["x", "type", "s", "axes", "norm", "overwrite_x", "workers"]


def recording_dctn(calls, real=None):
    """Return a ``dctn`` wrapper which appends every call to
    ``calls`` as a dictionary of named arguments.
    """
    real = scipy.fft.dctn if real is None else real

    def wrapper(*args, **kwargs):
        recorded = dict(zip(DCTN_POSITIONAL, args))
        recorded.update(kwargs)
        calls.append(recorded)
        return real(*args, **kwargs)

    return wrapper


def refinement_method():
    """Return the refinement method of the error floor tests.

    ``pytest.importorskip`` cannot express a fallback, so the
    optional NLopt dependency is looked up directly.
    """
    if dependency_version["nlopt"] is not None:
        return "LN_NELDERMEAD"
    return "minimize"


# --------------------------- Construction --------------------------- #


class TestConstruction:
    @pytest.mark.parametrize("circular_mask", [False, True])
    def test_attributes_of_the_nickel_projector(self, circular_mask):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=circular_mask
        )
        assert projector.bandwidth == NI_BANDWIDTH
        assert projector.dim == NI_DIM
        assert projector.sht.dim == NI_DIM
        assert projector.sht.bandwidth == NI_BANDWIDTH
        assert projector.sht.layout == "legendre"
        assert projector.circular_mask is circular_mask
        assert projector.signal_mask is None
        assert projector.oversampling == pytest.approx(math.sqrt(2))

        side = RESCALED_SIDES[NI_BANDWIDTH][0 if circular_mask else 1]
        assert projector.rescaled_shape == (side, side)
        assert (
            projector.n_points == WINDOW_POINTS[NI_BANDWIDTH][0 if circular_mask else 1]
        )

        assert projector.sphere_index.dtype == np.int64
        assert projector.sphere_index.shape == (projector.n_points,)
        assert projector.pixel_index.shape == (projector.n_points, 4)
        assert projector.pixel_index.dtype == np.int64
        assert projector.weights.shape == (projector.n_points, 4)
        assert projector.weights.dtype == np.float64
        assert projector.solid_angles.shape == (projector.n_points,)

        assert np.allclose(projector.weights.sum(axis=1), 1.0, atol=1e-12)
        assert projector.weights.min() >= 0.0
        assert projector.weights.max() <= 1.0

        assert projector.window_fraction == pytest.approx(
            WINDOW_FRACTIONS[circular_mask], rel=1e-4
        )
        assert projector.solid_angle_fraction == pytest.approx(
            SOLID_ANGLE_FRACTIONS[circular_mask], rel=1e-4
        )
        assert projector.window_solid_angle == pytest.approx(
            float(projector.solid_angles.sum()), rel=1e-12
        )
        assert projector.sphere_solid_angle == pytest.approx(
            2 * NI_DIM**2 - 4 * (NI_DIM - 1), abs=1e-9
        )
        assert projector.window_fraction == pytest.approx(
            projector.window_solid_angle / projector.sphere_solid_angle, rel=1e-12
        )

    @pytest.mark.parametrize("bandwidth", [53, 68, 88])
    def test_scale_factor_with_the_circle(self, bandwidth):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), bandwidth, circular_mask=True
        )
        assert projector.scale_factor == pytest.approx(
            SCALE_FACTORS[bandwidth], rel=1e-4
        )

    def test_window_harmonics_is_eager_and_shareable(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        # a lazily cached ``mlm`` would be the only mutable state of
        # the instance and would race between Phase 6's threads
        assert "window_harmonics" in vars(projector)
        assert projector.sht._quadrature_weights is not None
        assert projector.window_harmonics is projector.window_harmonics
        assert projector.window_harmonics.shape == (NI_BANDWIDTH, NI_BANDWIDTH)
        assert projector.window_harmonics.dtype == np.complex128

    def test_repr(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        text = repr(projector)
        assert "bw = 68" in text
        assert "dim = 71" in text
        assert "(60, 60) -> (53, 53)" in text
        assert "1317 points" in text

    @pytest.mark.parametrize("bandwidth", [0, -1])
    def test_bandwidth_must_be_at_least_one(self, bandwidth):
        with pytest.raises(ValueError):
            _back_projection.SphericalBackProjector(ni_detector(), bandwidth)

    @pytest.mark.parametrize("oversampling", [0.0, -1.0])
    def test_oversampling_must_be_positive(self, oversampling):
        with pytest.raises(ValueError):
            _back_projection.SphericalBackProjector(
                ni_detector(), NI_BANDWIDTH, oversampling=oversampling
            )

    def test_detector_must_be_an_ebsd_detector(self):
        with pytest.raises(TypeError):
            _back_projection.SphericalBackProjector("not a detector", NI_BANDWIDTH)

    def test_more_than_one_projection_centre_raises(self):
        detector = kp.data.nickel_ebsd_small().detector
        assert detector.navigation_size == 9
        with pytest.raises(ValueError) as info:
            _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        message = str(info.value)
        assert "pc_average" in message
        assert "9" in message
        assert "deepcopy" in message
        # ``EBSDDetector`` is not subscriptable, so the message must
        # not tell the user to index it
        assert "index" not in message

    def test_a_size_one_detector_with_a_1_1_3_pc_is_accepted(self):
        detector = EBSDDetector(
            shape=(60, 60), pc=np.full((1, 1, 3), 0.5), sample_tilt=70
        )
        assert detector.navigation_shape == (1, 1)
        assert detector.navigation_size == 1
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        assert projector.detector.navigation_shape == (1,)
        flat = EBSDDetector(shape=(60, 60), pc=[0.5, 0.5, 0.5], sample_tilt=70)
        reference = _back_projection.SphericalBackProjector(flat, NI_BANDWIDTH)
        assert np.array_equal(projector.sphere_index, reference.sphere_index)
        assert np.array_equal(projector.pixel_index, reference.pixel_index)
        assert np.array_equal(projector.weights, reference.weights)

    def test_empty_window_guard_for_a_geometry_cause(self):
        detector = ni_detector()
        detector.sample_tilt = -40
        with pytest.raises(ValueError) as info:
            _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        assert "solid_angle_fraction" in str(info.value)

    def test_empty_window_guard_for_an_all_true_signal_mask(self):
        mask = np.ones((60, 60), dtype=bool)
        with pytest.raises(ValueError) as info:
            _back_projection.SphericalBackProjector(
                ni_detector(), NI_BANDWIDTH, signal_mask=mask
            )
        assert "solid_angle_fraction" in str(info.value)

    def test_empty_window_guard_for_a_single_open_pixel(self):
        # the 502 x 502 solid angle grid catches this pixel, so the
        # first guard passes with a 1 x 1 resampled pattern, but no
        # Legendre grid point falls on it
        mask = np.ones((60, 60), dtype=bool)
        mask[0, 0] = False
        with pytest.raises(ValueError) as info:
            _back_projection.SphericalBackProjector(
                ni_detector(), NI_BANDWIDTH, signal_mask=mask
            )
        assert "empty window" in str(info.value)

    @pytest.mark.parametrize("name, value", [("azimuthal", 5.0), ("twist", 3.0)])
    def test_azimuthal_and_twist_raise(self, name, value):
        detector = ni_detector()
        setattr(detector, name, value)
        with pytest.raises(ValueError) as info:
            _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        assert name in str(info.value)

    def test_the_azimuthal_guard_is_not_vacuous(self):
        # a non-zero azimuthal angle does change the geometry, so
        # the guard is refusing something real
        detector = ni_detector()
        tilted = ni_detector()
        tilted.azimuthal = 5
        assert not np.allclose(
            detector.sample_to_detector.to_matrix(),
            tilted.sample_to_detector.to_matrix(),
        )

    @pytest.mark.parametrize(
        "mask",
        [np.ones((60, 61), dtype=bool), np.zeros((60, 60), dtype=int)],
    )
    def test_signal_mask_shape_and_dtype_are_validated(self, mask):
        with pytest.raises(ValueError):
            _back_projection.SphericalBackProjector(
                ni_detector(), NI_BANDWIDTH, signal_mask=mask
            )

    @pytest.mark.parametrize("dim", [70, 69])
    def test_dim_is_validated_by_the_transform(self, dim):
        with pytest.raises(ValueError):
            _back_projection.SphericalBackProjector(
                ni_detector(), NI_BANDWIDTH, dim=dim
            )

    def test_the_detector_is_deep_copied(self):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        assert projector.detector is not detector
        before_pc = projector.detector.pc.copy()
        before_index = projector.sphere_index.copy()
        detector.pc = np.array([0.1, 0.9, 0.3])
        assert np.array_equal(projector.detector.pc, before_pc)
        assert np.array_equal(projector.sphere_index, before_index)

    def test_the_signal_mask_attribute_is_a_copy(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[MASK_BLOCK] = True
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, signal_mask=mask
        )
        assert projector.signal_mask is not mask
        assert np.array_equal(projector.signal_mask, mask)


# ----------------------------- Geometry ----------------------------- #


class TestGeometry:
    @pytest.mark.parametrize(
        "case",
        [None] + SYNTHETIC_DETECTORS,
        ids=["nickel", "rect_48_60", "square_tilt_-30", "square_41"],
    )
    def test_direction_oracle(self, case, record_property):
        detector = ni_detector() if case is None else synthetic_detector(*case)
        nrows, ncols = detector.shape
        dc = _get_direction_cosines_from_detector(detector)
        geometry = _back_projection._pixel_map(detector)
        col, row, in_front = _back_projection._directions_to_pixels(dc, geometry)
        expected_row, expected_col = np.divmod(np.arange(nrows * ncols), ncols)
        delta_col = float(np.abs(col - expected_col).max())
        delta_row = float(np.abs(row - expected_row).max())
        record_property(f"oracle_col_{nrows}x{ncols}", f"{delta_col:.3e}")
        record_property(f"oracle_row_{nrows}x{ncols}", f"{delta_row:.3e}")
        assert delta_col <= 1e-10
        assert delta_row <= 1e-10
        assert bool(in_front.all())

    def test_the_module_chain_agrees_with_the_test_local_one(self):
        detector = ni_detector()
        dc = _get_direction_cosines_from_detector(detector)
        geometry = _back_projection._pixel_map(detector)
        col, row, in_front = _back_projection._directions_to_pixels(dc, geometry)
        local_col, local_row, local_front = local_pixel_chain(detector, dc)
        assert np.allclose(col, local_col, atol=1e-12)
        assert np.allclose(row, local_row, atol=1e-12)
        assert np.array_equal(in_front, local_front)

    def test_lut_points_map_back_to_their_legendre_normals(self):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        window_normals = normals[projector.sphere_index]
        # the round trip goes through the **module's** forward map:
        # ``local_pixel_chain`` and ``local_pixels_to_directions``
        # are exact algebraic inverses, so on the local chain alone
        # the assertion could only fail for a point behind the
        # detector, which ``test_lut_membership`` already pins
        geometry = _back_projection._pixel_map(detector)
        col, row, in_front = _back_projection._directions_to_pixels(
            window_normals, geometry
        )
        assert bool(in_front.all())
        local_col, local_row, _ = local_pixel_chain(detector, window_normals)
        assert np.abs(col - local_col).max() <= 1e-12
        assert np.abs(row - local_row).max() <= 1e-12
        recovered = local_pixels_to_directions(detector, col, row)
        assert np.abs(recovered - window_normals).max() <= 1e-10

    def test_the_frame_statement(self):
        detector = ni_detector()
        matrix = detector.sample_to_detector.to_matrix().squeeze()
        alpha = math.radians(90 - detector.sample_tilt + detector.tilt)
        assert np.allclose(
            matrix[2], [math.sin(alpha), 0.0, math.cos(alpha)], atol=1e-12
        )
        assert np.allclose(
            Rotation.from_matrix(matrix).to_matrix().squeeze(), matrix, atol=1e-14
        )
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)

    def test_binning_neutrality(self):
        detector = ni_detector()
        assert detector.binning == 8
        rebinned = EBSDDetector(
            shape=detector.shape,
            pc=detector.pc,
            sample_tilt=detector.sample_tilt,
            tilt=detector.tilt,
            binning=1,
            px_size=8.0,
        )
        a = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        b = _back_projection.SphericalBackProjector(rebinned, NI_BANDWIDTH)
        # same arithmetic on the same numbers, so bitwise is fair
        assert np.array_equal(a.sphere_index, b.sphere_index)
        assert np.array_equal(a.pixel_index, b.pixel_index)
        assert np.abs(a.weights - b.weights).max() == 0.0

    def test_the_emsoft_round_trip(self):
        detector = ni_detector()
        round_tripped = EBSDDetector(
            shape=detector.shape,
            pc=detector.pc_emsoft(),
            convention="emsoft",
            binning=8,
            px_size=1.0,
            sample_tilt=detector.sample_tilt,
            tilt=detector.tilt,
        )
        assert np.allclose(round_tripped.pc, detector.pc, atol=1e-12)
        a = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        b = _back_projection.SphericalBackProjector(round_tripped, NI_BANDWIDTH)
        assert np.array_equal(a.sphere_index, b.sphere_index)
        assert np.abs(a.weights - b.weights).max() <= 1e-12

    @pytest.mark.parametrize("tilts", list(TILT_TABLE))
    def test_the_south_hemisphere_is_never_gathered(self, tilts, record_property):
        sample_tilt, tilt = tilts
        detector = ni_detector()
        detector.sample_tilt = sample_tilt
        detector.tilt = tilt
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        assert projector.n_points == pytest.approx(TILT_TABLE[tilts], abs=2)
        dc = _get_direction_cosines_from_detector(detector)
        record_property(
            f"z_range_{sample_tilt}_{tilt}",
            f"[{dc[:, 2].min():.3f}, {dc[:, 2].max():.3f}]",
        )
        rng = np.random.default_rng(0)
        pattern = rng.uniform(0, 255, (60, 60))
        north, south = projector.unproject(pattern)
        assert np.count_nonzero(south) == 0
        assert np.count_nonzero(north) == projector.n_points

    def test_a_flat_sample_clips_the_window(self, record_property):
        flat = ni_detector()
        flat.sample_tilt = 0
        tilted = ni_detector()
        dc = _get_direction_cosines_from_detector(flat)
        below = float((dc[:, 2] < 0).mean())
        record_property("pixels_below_equator_sample_tilt_0", f"{below:.3f}")
        assert below > 0.5
        a = _back_projection.SphericalBackProjector(flat, NI_BANDWIDTH)
        b = _back_projection.SphericalBackProjector(tilted, NI_BANDWIDTH)
        ratio = a.n_points / b.n_points
        record_property("window_ratio_sample_tilt_0", f"{ratio:.3f}")
        assert ratio < 0.6


# ------------------------------ Sizes ------------------------------- #


class TestSizes:
    @pytest.mark.parametrize("bandwidth", sorted(RESCALED_SIDES))
    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_size_table(self, bandwidth, circular_mask):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), bandwidth, circular_mask=circular_mask
        )
        index = 0 if circular_mask else 1
        side = RESCALED_SIDES[bandwidth][index]
        assert projector.rescaled_shape == (side, side)
        assert projector.n_points == WINDOW_POINTS[bandwidth][index]

    def test_the_size_rounding_is_half_away_from_zero(self):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        target = 48
        oversampling = (target + 0.5) / (projector.scale_factor * detector.ncols)
        product = projector.scale_factor * oversampling * detector.ncols
        assert abs(product - (target + 0.5)) < 1e-9
        # Python's banker's round would give 48 here
        assert round(product) == target
        other = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, oversampling=oversampling
        )
        assert other.rescaled_shape == (target + 1, target + 1)

    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_solid_angle_fraction_and_its_literal_divisor(
        self, circular_mask, record_property
    ):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=circular_mask
        )
        assert projector.solid_angle_fraction == pytest.approx(
            SOLID_ANGLE_FRACTIONS[circular_mask], rel=1e-4
        )
        count = projector.solid_angle_fraction * SOLID_ANGLE_DIVISOR
        assert count == pytest.approx(SOLID_ANGLE_COUNTS[circular_mask], abs=1e-6)

        # a second implementation of ``Geometry::solidAngle(501)``
        grid_res = 501
        axis = np.arange(grid_res + 1) / grid_res
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        normals = _grid.square_to_sphere(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        col, row, in_front = local_pixel_chain(detector, normals)
        keep = in_front & (normals[:, 2] >= 0)
        keep &= local_inside(detector, col, row, circular_mask)
        assert int(keep.sum()) == SOLID_ANGLE_COUNTS[circular_mask]

        consistent = int(keep.sum()) / ((grid_res + 1) ** 2 + (grid_res - 1) ** 2)
        record_property(
            f"solid_angle_consistent_circle_{circular_mask}", f"{consistent:.6f}"
        )

    @pytest.mark.parametrize("bandwidth", [53, 68, 88])
    def test_sphere_solid_angle_counts_the_equator_once(self, bandwidth):
        projector = _back_projection.SphericalBackProjector(ni_detector(), bandwidth)
        dim = projector.dim
        assert projector.sphere_solid_angle == pytest.approx(
            2 * dim**2 - 4 * (dim - 1), abs=1e-9
        )

    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_window_fraction_against_the_analytic_estimate(
        self, circular_mask, record_property
    ):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=circular_mask
        )
        analytic = analytic_solid_angle_fraction(detector, circular_mask)
        ratio = projector.window_fraction / analytic
        record_property(f"window_over_analytic_circle_{circular_mask}", f"{ratio:.4f}")
        assert abs(ratio - 1) < 0.02
        # a plausibility bound only: the divisor quirk and the rim
        # inclusion bias the two in opposite directions
        assert (
            abs(projector.solid_angle_fraction / projector.window_fraction - 1) < 0.01
        )

    def test_solid_angles_match_the_ring_lookup(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        rings = _grid.ring_number(projector.dim).ravel()
        values = _grid.ring_solid_angles(projector.dim, "legendre")
        expected = values[rings[projector.sphere_index]]
        assert np.array_equal(projector.solid_angles, expected)


# ---------------------------- Lookup table -------------------------- #


class TestLUT:
    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_indices_are_in_range_and_weights_sum_to_one(self, circular_mask):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=circular_mask
        )
        h_out, w_out = projector.rescaled_shape
        assert projector.pixel_index.min() >= 0
        assert projector.pixel_index.max() < h_out * w_out
        assert projector.sphere_index.min() >= 0
        assert projector.sphere_index.max() < projector.dim**2
        assert np.abs(projector.weights.sum(axis=1) - 1).max() <= 1e-12

    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_the_clamp_structure_of_the_indices(self, circular_mask):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=circular_mask
        )
        _, w_out = projector.rescaled_shape
        along_x = projector.pixel_index[:, 1] - projector.pixel_index[:, 0]
        along_y = projector.pixel_index[:, 2] - projector.pixel_index[:, 0]
        assert set(np.unique(along_x)).issubset({0, 1})
        assert set(np.unique(along_y)).issubset({0, w_out})
        assert np.array_equal(
            projector.pixel_index[:, 3] - projector.pixel_index[:, 2], along_x
        )

    @pytest.mark.parametrize("geometry", list(LUT_GEOMETRIES))
    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_the_rim_structure(self, geometry, circular_mask, record_property):
        detector, bandwidth = lut_geometry(geometry)
        projector = _back_projection.SphericalBackProjector(
            detector, bandwidth, circular_mask=circular_mask
        )
        h_out, w_out = projector.rescaled_shape
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        col, row, _ = local_pixel_chain(detector, normals[projector.sphere_index])
        x = (col + 0.5) * w_out / detector.ncols - 0.5
        y = (row + 0.5) * h_out / detector.nrows - 0.5

        left = x < 0
        right = x > w_out - 1
        top = y < 0
        bottom = y > h_out - 1
        rim = left | right | top | bottom
        counts = tuple(int(side.sum()) for side in (left, top, right, bottom))
        record_property(
            f"rim_counts_{geometry}_circle_{circular_mask}",
            f"total {int(rim.sum())} left {counts[0]} top {counts[1]} "
            f"right {counts[2]} bottom {counts[3]}",
        )
        # pinned so that facts (b) and (c) below cannot go vacuous
        # unnoticed: the minimum distance of a point to one of the
        # four thresholds is 1.3e-3 resampled pixels over these six
        # cases, so the counts are not knife-edge
        assert counts == RIM_COUNTS[(geometry, circular_mask)]
        detector_rim = (
            (col < 0)
            | (col > detector.ncols - 1)
            | (row < 0)
            | (row > detector.nrows - 1)
        )
        record_property(
            f"rim_detector_{geometry}_circle_{circular_mask}",
            str(int(detector_rim.sum())),
        )
        assert int(rim.sum()) <= 0.05 * projector.n_points

        # (a) every row is a partition of unity
        assert np.abs(projector.weights.sum(axis=1) - 1).max() <= 1e-12
        assert projector.weights.min() >= 0.0
        assert projector.weights.max() <= 1.0
        # (b) the left and top rims clip a weight to exactly zero and
        # keep four distinct neighbours
        assert np.all(projector.weights[left][:, 1] == 0.0)
        assert np.all(projector.weights[left][:, 3] == 0.0)
        assert np.all(
            projector.pixel_index[left][:, 1] == projector.pixel_index[left][:, 0] + 1
        )
        assert np.all(projector.weights[top][:, 2] == 0.0)
        assert np.all(projector.weights[top][:, 3] == 0.0)
        assert np.all(
            projector.pixel_index[top][:, 2] == projector.pixel_index[top][:, 0] + w_out
        )
        # (c) the right and bottom rims collapse the index pair, and
        # the two weights on the pair are not equal
        assert np.all(
            projector.pixel_index[right][:, 0] == projector.pixel_index[right][:, 1]
        )
        assert np.all(
            projector.pixel_index[right][:, 2] == projector.pixel_index[right][:, 3]
        )
        assert np.all(
            projector.pixel_index[bottom][:, 0] == projector.pixel_index[bottom][:, 2]
        )
        inner_x = ~right
        assert np.all(
            projector.pixel_index[inner_x][:, 1]
            == projector.pixel_index[inner_x][:, 0] + 1
        )
        inner_y = ~bottom
        assert np.all(
            projector.pixel_index[inner_y][:, 2]
            == projector.pixel_index[inner_y][:, 0] + w_out
        )
        # no clamp rule can give a rim point a weight of exactly one
        record_property(
            f"max_weight_{geometry}_circle_{circular_mask}",
            f"{projector.weights.max():.3f}",
        )
        assert not np.any(projector.weights == 1.0)

    def test_every_rim_group_is_exercised(self):
        # the counts of :data:`RIM_COUNTS` are asserted case by case
        # above, so this guards the *choice* of geometries: without a
        # case on the bottom rim, fact (c) for ``y > h_out - 1`` is
        # vacuous and a kernel written ``j1 = j0 + 1``, with no
        # ``min(..., h_out - 1)``, passes the whole suite
        covered = {
            side
            for counts in RIM_COUNTS.values()
            for side, count in zip(RIM_SIDES, counts)
            if count > 0
        }
        assert covered == set(RIM_SIDES)

    @pytest.mark.parametrize("geometry", list(LUT_GEOMETRIES))
    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_the_resample_map_is_pinned_structurally(self, geometry, circular_mask):
        # the rectangular case is the discriminating one: on a square
        # detector ``h_out == w_out`` and a row/column swap inside
        # the map, or a transposed corner copy, is invisible here
        detector, bandwidth = lut_geometry(geometry)
        projector = _back_projection.SphericalBackProjector(
            detector, bandwidth, circular_mask=circular_mask
        )
        h_out, w_out = projector.rescaled_shape
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        col, row, _ = local_pixel_chain(detector, normals[projector.sphere_index])
        x = (col + 0.5) * w_out / detector.ncols - 0.5
        y = (row + 0.5) * h_out / detector.nrows - 0.5
        i0 = np.clip(np.floor(x), 0, w_out - 1).astype(np.int64)
        j0 = np.clip(np.floor(y), 0, h_out - 1).astype(np.int64)
        i1 = np.minimum(i0 + 1, w_out - 1)
        j1 = np.minimum(j0 + 1, h_out - 1)
        wx1 = np.clip(x - i0, 0, 1)
        wy1 = np.clip(y - j0, 0, 1)
        wx0 = 1 - wx1
        wy0 = 1 - wy1
        expected_index = np.stack(
            [j0 * w_out + i0, j0 * w_out + i1, j1 * w_out + i0, j1 * w_out + i1],
            axis=-1,
        )
        expected_weights = np.stack(
            [wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1], axis=-1
        )
        assert np.array_equal(projector.pixel_index, expected_index)
        assert np.abs(projector.weights - expected_weights).max() <= 1e-12

    def test_the_emsphinx_pixel_stretch_is_a_negative_control(self, record_property):
        # the direction oracle, the convention lock and a bright
        # pixel argmax are all blind to a stretch applied here
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(detector, NI_BANDWIDTH)
        h_out, w_out = projector.rescaled_shape
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        col, row, _ = local_pixel_chain(detector, normals[projector.sphere_index])
        stretched_x = (col + 0.5) / detector.ncols * (w_out - 1)
        stretched_y = (row + 0.5) / detector.nrows * (h_out - 1)
        x = (col + 0.5) * w_out / detector.ncols - 0.5
        y = (row + 0.5) * h_out / detector.nrows - 0.5
        record_property(
            "stretch_max_delta_px",
            f"dx {np.abs(stretched_x - x).max():.4f} "
            f"dy {np.abs(stretched_y - y).max():.4f}",
        )

        i0_stretched = np.clip(np.floor(stretched_x), 0, w_out - 1).astype(np.int64)
        j0_stretched = np.clip(np.floor(stretched_y), 0, h_out - 1).astype(np.int64)
        i0 = projector.pixel_index[:, 0] % w_out
        j0 = projector.pixel_index[:, 0] // w_out
        moved_i0 = int((i0_stretched != i0).sum())
        moved_j0 = int((j0_stretched != j0).sum())
        record_property("stretch_moves_i0", str(moved_i0))
        record_property("stretch_moves_j0", str(moved_j0))
        assert moved_i0 > 100

        wx1 = np.clip(stretched_x - i0_stretched, 0, 1)
        wy1 = np.clip(stretched_y - j0_stretched, 0, 1)
        stretched_weights = np.stack(
            [
                (1 - wy1) * (1 - wx1),
                (1 - wy1) * wx1,
                wy1 * (1 - wx1),
                wy1 * wx1,
            ],
            axis=-1,
        )
        differing = int(
            (np.abs(projector.weights - stretched_weights).max(axis=1) > 1e-6).sum()
        )
        record_property("stretch_changes_weight_rows", str(differing))
        assert differing > 0.9 * projector.n_points

    def test_signal_mask_excludes_by_nearest_pixel(self):
        detector = ni_detector()
        mask = np.zeros((60, 60), dtype=bool)
        mask[MASK_BLOCK] = True
        plain = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True
        )
        masked = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True, signal_mask=mask
        )
        assert plain.n_points == 1117
        assert masked.n_points == 1020
        assert plain.rescaled_shape == (49, 49)
        assert masked.rescaled_shape == (47, 47)

        normals = _grid.legendre_normals(plain.dim).reshape(-1, 3)
        removed = np.setdiff1d(plain.sphere_index, masked.sphere_index)
        kept = np.intersect1d(plain.sphere_index, masked.sphere_index)
        for indices, expected in ((removed, True), (kept, False)):
            col, row, _ = local_pixel_chain(detector, normals[indices])
            columns = np.clip(np.floor(col + 0.5), 0, 59).astype(int)
            rows = np.clip(np.floor(row + 0.5), 0, 59).astype(int)
            assert np.all(mask[rows, columns] == expected)

    def test_the_circle_is_the_physical_one_on_a_rectangular_detector(self):
        # the C++ lookup table circle, in rescaled pixels, would give
        # 958 points here
        detector = synthetic_detector(*SYNTHETIC_DETECTORS[0])
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True
        )
        assert projector.rescaled_shape == (42, 52)
        assert projector.n_points == 953
        # the control the 953 is a fraction of, and the only
        # non-square resample map of the LUT pins
        plain = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=False
        )
        assert plain.rescaled_shape == (48, 60)
        assert plain.n_points == 1285

    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_lut_membership(self, circular_mask):
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=circular_mask
        )
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        col, row, in_front = local_pixel_chain(detector, normals)
        keep = in_front & (normals[:, 2] >= 0)
        keep &= local_inside(detector, col, row, circular_mask)
        assert np.array_equal(projector.sphere_index, np.flatnonzero(keep))


# ---------------------------- unproject ----------------------------- #


class TestUnproject:
    def test_the_dctn_round_trip_factor_is_four_h_w(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((60, 60))
        round_tripped = scipy.fft.dctn(
            scipy.fft.dctn(x, type=2, workers=1), type=3, workers=1
        )
        assert np.abs(round_tripped / (4 * 60 * 60) - x).max() <= 1e-12

    def test_the_idctn_negative_control(self):
        spectrum = scipy.fft.dctn(np.full((60, 60), 37.0), type=2, workers=1)
        padded = np.zeros((53, 53))
        padded[:53, :53] = spectrum[:53, :53]
        correct = scipy.fft.dctn(padded, type=3, workers=1)
        assert np.allclose(correct, 37.0 * 4 * 60 * 60, rtol=1e-12)
        wrong = scipy.fft.idctn(padded, type=3, workers=1)
        assert np.ptp(wrong) > 1
        assert wrong.max() / correct.max() == pytest.approx(1 / 53**2, rel=1e-6)

    @pytest.mark.parametrize(
        "pattern, expected_iq",
        [
            (np.full((60, 60), 7.0), 1.0),
            (np.full((60, 60), 7, dtype=np.uint8), 1.0),
            (np.zeros((60, 60)), 0.0),
        ],
        ids=["float_7", "uint8_7", "zeros"],
    )
    def test_a_constant_pattern_returns_the_window_mask(self, pattern, expected_iq):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        mask_north, mask_south = projector.window_mask()
        north, south, iq = projector.unproject(pattern, return_image_quality=True)
        assert np.array_equal(north, mask_north)
        assert np.array_equal(south, mask_south)
        assert np.count_nonzero(south) == 0
        assert iq == expected_iq

    def test_the_constant_short_cut_calls_no_transform(self, monkeypatch):
        assert hasattr(_back_projection, "dctn"), (
            "_back_projection must bind scipy.fft.dctn in its namespace"
        )
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        calls = []
        spy = recording_dctn(calls, real=scipy.fft.dctn)
        monkeypatch.setattr(_back_projection, "dctn", spy)
        monkeypatch.setattr(scipy.fft, "dctn", spy)
        projector.unproject(np.full((60, 60), 7.0), return_image_quality=True)
        projector.unproject(np.zeros((60, 60)), return_image_quality=True)
        assert calls == []

    def test_a_random_pattern_calls_dctn_twice_with_one_worker(self, monkeypatch):
        assert hasattr(_back_projection, "dctn"), (
            "_back_projection must bind scipy.fft.dctn in its namespace"
        )
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        rng = np.random.default_rng(1)
        pattern = rng.uniform(0, 255, (60, 60)) + 1000.0
        calls = []
        spy = recording_dctn(calls, real=scipy.fft.dctn)
        monkeypatch.setattr(_back_projection, "dctn", spy)
        monkeypatch.setattr(scipy.fft, "dctn", spy)
        projector.unproject(pattern, return_image_quality=True)
        assert len(calls) == 2
        assert [call.get("type") for call in calls] == [2, 3]
        for call in calls:
            assert call.get("workers") == 1
            assert call.get("norm") is None
        # the DC term of the truncated spectrum must be zeroed, which
        # a large offset makes visible
        second_input = np.asarray(calls[1]["x"])
        assert second_input[0, 0] == 0.0

    def test_the_pocketfft_constant_transform_inexactness_is_recorded(
        self, record_property
    ):
        # ``validation.md`` makes this a recorded guard and not an
        # assertion on purpose: a future exact DCT would make the
        # ``ptp`` short-cut redundant, which is a note for the
        # reviewer, not a regression
        spectrum = scipy.fft.dctn(np.full((60, 60), 37.0), type=2, workers=1)
        alternating = spectrum.copy()
        alternating[0, 0] = 0.0
        record_property("pocketfft_constant_ac_max", f"{np.abs(alternating).max():.3e}")

    def test_the_literal_stdev_zero_branch(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        kernel = _back_projection._unproject_kernel
        h_out, w_out = projector.rescaled_shape
        north = np.zeros(projector.dim**2)
        stdev = kernel(
            np.zeros(h_out * w_out),
            projector.pixel_index,
            projector.weights,
            projector.solid_angles,
            projector.window_solid_angle,
            projector.sphere_index,
            north,
        )
        assert stdev == 0.0
        assert np.all(north[projector.sphere_index] == 1.0)
        assert np.count_nonzero(north) == projector.n_points

    def test_a_tiny_amplitude_image_is_normalised_not_masked(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        kernel = _back_projection._unproject_kernel
        h_out, w_out = projector.rescaled_shape
        rng = np.random.default_rng(2)
        image = rng.standard_normal(h_out * w_out) * 1e-14
        north = np.zeros(projector.dim**2)
        stdev = kernel(
            image,
            projector.pixel_index,
            projector.weights,
            projector.solid_angles,
            projector.window_solid_angle,
            projector.sphere_index,
            north,
        )
        assert stdev > 0
        mean, second = weighted_moments(projector, north)
        assert abs(mean) <= 1e-10
        assert second == pytest.approx(1.0, abs=1e-10)

    def test_the_normalisation_identities(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        rng = np.random.default_rng(3)
        patterns = list(_ni_signal_data().reshape(-1, 60, 60))
        patterns += [rng.uniform(0, 255, (60, 60)) for _ in range(5)]
        for pattern in patterns:
            north, south = projector.unproject(pattern)
            mean, second = weighted_moments(projector, north)
            assert abs(mean) <= 1e-10
            assert second == pytest.approx(1.0, abs=1e-10)
            assert np.count_nonzero(south) == 0
            off_window = np.ones(projector.dim**2, dtype=bool)
            off_window[projector.sphere_index] = False
            assert np.all(north.ravel()[off_window] == 0.0)

    def test_unit_weights_break_the_identity(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        north, _ = projector.unproject(_ni_signal_data()[0, 0])
        values = north.ravel()[projector.sphere_index]
        unweighted = float((values**2).mean())
        assert abs(unweighted - 1) > 1e-3

    def test_the_kernel_returns_a_positive_stdev_for_a_real_pattern(
        self, record_property
    ):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        h_out, w_out = projector.rescaled_shape
        pattern = _ni_signal_data()[0, 0].astype(np.float64)
        rescaled, _ = _back_projection._dct_rescale(pattern, h_out, w_out)
        north = np.zeros(projector.dim**2)
        stdev = _back_projection._unproject_kernel(
            np.ascontiguousarray(rescaled).ravel(),
            projector.pixel_index,
            projector.weights,
            projector.solid_angles,
            projector.window_solid_angle,
            projector.sphere_index,
            north,
        )
        record_property("resampled_stdev", f"{stdev:.4e}")
        assert stdev > 0

    def test_out_buffers_are_written_in_place(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        north = np.full((projector.dim, projector.dim), 5.0)
        south = np.full((projector.dim, projector.dim), 5.0)
        out_north, out_south = projector.unproject(
            _ni_signal_data()[0, 0], out=(north, south)
        )
        assert out_north is north
        assert out_south is south
        off_window = np.ones(projector.dim**2, dtype=bool)
        off_window[projector.sphere_index] = False
        # the C++ contract: only the window points are written
        assert np.all(north.ravel()[off_window] == 5.0)
        assert np.all(south == 5.0)

    @pytest.mark.parametrize(
        "out",
        [
            (np.zeros((70, 70)), np.zeros((71, 71))),
            (np.zeros((71, 71), dtype=np.float32), np.zeros((71, 71))),
            (np.zeros((71, 142))[:, ::2], np.zeros((71, 71))),
            (np.zeros((71, 71)),),
        ],
        ids=["shape", "dtype", "non_contiguous", "not_a_pair"],
    )
    def test_out_is_validated(self, out):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        with pytest.raises(ValueError):
            projector.unproject(_ni_signal_data()[0, 0], out=out)

    @pytest.mark.parametrize("shape", [(60, 61), (3600,), (2, 60, 60)])
    def test_a_pattern_of_the_wrong_shape_raises(self, shape):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        with pytest.raises(ValueError):
            projector.unproject(np.zeros(shape))

    def test_the_input_pattern_is_never_modified(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, signal_mask=np.zeros((60, 60), dtype=bool)
        )
        pattern = _ni_signal_data()[0, 0].copy()
        reference = pattern.copy()
        projector.unproject(pattern)
        assert np.array_equal(pattern, reference)

    def test_nan_is_not_guarded(self):
        # documented: a NaN pixel propagates onto the window rather
        # than raising
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        pattern = _ni_signal_data()[0, 0].astype(np.float64)
        pattern[30, 30] = np.nan
        north, _ = projector.unproject(pattern)
        assert np.isnan(north.ravel()[projector.sphere_index]).any()


# -------------------------- Image quality --------------------------- #


class TestImageQuality:
    def test_image_quality_equals_the_unproject_value(self):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        patterns = list(_ni_signal_data().reshape(-1, 60, 60))
        patterns += [
            np.full((60, 60), 7.0),
            np.full((60, 60), 7, dtype=np.uint8),
            np.zeros((60, 60)),
        ]
        for pattern in patterns:
            direct = projector.image_quality(pattern)
            _, _, from_unproject = projector.unproject(
                pattern, return_image_quality=True
            )
            assert direct == from_unproject

    def test_image_quality_matches_the_literal_transcription(self, record_property):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        values = []
        for pattern in _ni_signal_data().reshape(-1, 60, 60):
            spectrum = scipy.fft.dctn(pattern.astype(np.float64), type=2, workers=1)
            expected = literal_image_quality(spectrum)
            value = projector.image_quality(pattern)
            assert value == pytest.approx(expected, rel=1e-12)
            values.append(value)
        record_property("ni_image_quality", ", ".join(f"{v:.4f}" for v in values))
        assert min(values) >= 0.7
        assert max(values) <= 0.85

        for constant in (1.0, 7.0, 37.0, 200.0):
            spectrum = scipy.fft.dctn(np.full((60, 60), constant), type=2, workers=1)
            assert literal_image_quality(spectrum) == pytest.approx(1.0, abs=1e-12)
            assert _back_projection._dct_image_quality(
                np.full((60, 60), constant)
            ) == pytest.approx(1.0, abs=1e-12)
        assert _back_projection._dct_image_quality(np.zeros((60, 60))) == 0.0

    def test_image_quality_is_scale_invariant_but_offset_dependent(
        self, record_property
    ):
        pattern = _ni_signal_data()[0, 0].astype(np.float64)
        base = _back_projection._dct_image_quality(pattern)
        scaled = _back_projection._dct_image_quality(pattern * 3)
        offset = _back_projection._dct_image_quality(pattern + 100)
        record_property("iq_scale_delta", f"{scaled - base:.3e}")
        record_property("iq_offset_delta", f"{offset - base:.4f}")
        assert abs(scaled - base) <= 1e-12
        assert offset != base

    def test_image_quality_of_uniform_noise(self, record_property):
        rng = np.random.default_rng(4)
        value = _back_projection._dct_image_quality(rng.uniform(0, 255, (60, 60)))
        record_property("iq_uniform_noise", f"{value:.4f}")
        assert value < 0.2

    def test_it_is_not_kikuchipys_image_quality(self, record_property):
        patterns = _ni_signal_data().reshape(-1, 60, 60)
        ours = [_back_projection._dct_image_quality(p) for p in patterns]
        theirs = [
            float(kp.pattern.get_image_quality(p.astype(np.float32))) for p in patterns
        ]
        record_property(
            "kikuchipy_image_quality", ", ".join(f"{v:.4f}" for v in theirs)
        )
        record_property("iq_correlation", f"{correlation(ours, theirs):.3f}")
        assert not np.allclose(ours, theirs, atol=0.1)

    def test_the_preprocessing_module_has_no_transform(self):
        assert not hasattr(_preprocessing, "dctn")
        assert "scipy.fft" not in inspect.getsource(_preprocessing)


# ----------------------- Harmonic recovery -------------------------- #


class TestHarmonicRecovery:
    @pytest.mark.parametrize(
        "degree, order",
        [
            (2, 1),
            (3, 2),
            (4, 0),
            (5, 3),
            (6, 4),
            (8, 2),
            pytest.param(12, 5, marks=pytest.mark.weekly),
        ],
    )
    def test_a_spherical_harmonic_is_recovered(self, degree, order, record_property):
        pytest.importorskip("scipy", minversion="1.15")
        detector = ni_detector()
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True
        )
        dc = direction_cosines(detector)
        pattern = real_spherical_harmonic(dc, degree, order)
        north, south = projector.unproject(pattern)

        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        truth_values = real_spherical_harmonic(
            normals[projector.sphere_index], degree, order
        )
        truth = windowed_truth(projector, truth_values)
        found = north.ravel()[projector.sphere_index]

        windowed = correlation(found, truth)
        rms = math.sqrt(
            float(
                ((found - truth) ** 2 * projector.solid_angles).sum()
                / projector.window_solid_angle
            )
        )
        truth_grid = np.zeros((projector.dim, projector.dim))
        truth_grid.ravel()[projector.sphere_index] = truth
        found_alm = projector.sht.analyze(north, south)
        truth_alm = projector.sht.analyze(
            truth_grid, np.zeros((projector.dim, projector.dim))
        )
        spectral = correlation(
            np.concatenate([found_alm.real.ravel(), found_alm.imag.ravel()]),
            np.concatenate([truth_alm.real.ravel(), truth_alm.imag.ravel()]),
        )
        record_property(
            f"recovery_Y{degree}_{order}",
            f"windowed {windowed:.6f} rms {rms:.4f} spectral {spectral:.6f}",
        )
        assert windowed > 0.999
        assert rms < 0.01
        assert spectral > 0.999

    @pytest.mark.parametrize("fill_value", [0.0, 255.0])
    def test_the_mean_fill_rescues_a_masked_block(self, fill_value, record_property):
        pytest.importorskip("scipy", minversion="1.15")
        detector = ni_detector()
        mask = np.zeros((60, 60), dtype=bool)
        mask[MASK_BLOCK] = True
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True, signal_mask=mask
        )
        dc = direction_cosines(detector)
        pattern = 100 * real_spherical_harmonic(dc, 4, 2) + 50
        pattern[mask] = fill_value

        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        truth = windowed_truth(
            projector, real_spherical_harmonic(normals[projector.sphere_index], 4, 2)
        )

        north, _ = projector.unproject(pattern)
        with_fill = correlation(north.ravel()[projector.sphere_index], truth)

        # the same chain without the fill, so the effect is visible
        h_out, w_out = projector.rescaled_shape
        rescaled, _ = _back_projection._dct_rescale(
            pattern.astype(np.float64), h_out, w_out
        )
        bare = np.zeros(projector.dim**2)
        _back_projection._unproject_kernel(
            np.ascontiguousarray(rescaled).ravel(),
            projector.pixel_index,
            projector.weights,
            projector.solid_angles,
            projector.window_solid_angle,
            projector.sphere_index,
            bare,
        )
        without_fill = correlation(bare[projector.sphere_index], truth)
        record_property(
            f"mean_fill_block_{fill_value:.0f}",
            f"with {with_fill:.5f} without {without_fill:.5f}",
        )
        assert with_fill > 0.99
        if fill_value == 255.0:
            assert without_fill < 0.95


# ------------------- Window harmonics and rDen ---------------------- #


class TestWindowHarmonics:
    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_the_dc_term_of_mlm_is_the_window_fraction(
        self, circular_mask, record_property
    ):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=circular_mask
        )
        mlm = projector.window_harmonics
        s2m_over_four_pi = (
            float(mlm[0, 0].real) * math.sqrt(4 * math.pi) / (4 * math.pi)
        )
        record_property(
            f"mlm00_circle_{circular_mask}",
            f"{mlm[0, 0].real:.7f} -> {s2m_over_four_pi:.7f} vs "
            f"{projector.window_fraction:.7f}",
        )
        assert s2m_over_four_pi == pytest.approx(projector.window_fraction, rel=1e-3)

    def test_window_harmonics_equals_analyze_of_the_mask(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        expected = projector.sht.analyze(*projector.window_mask())
        assert np.array_equal(projector.window_harmonics, expected)

    def test_squared_harmonics_squares_the_synthesis(self):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        alm = ni_master_harmonics(NI_BANDWIDTH).alm
        north, south = projector.sht.synthesize(alm)
        expected = projector.sht.analyze(north**2, south**2)
        assert np.array_equal(projector.squared_harmonics(alm), expected)

    @pytest.mark.parametrize("bandwidth", [67, 69])
    def test_squared_harmonics_validates_the_bandwidth(self, bandwidth):
        projector = _back_projection.SphericalBackProjector(ni_detector(), NI_BANDWIDTH)
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        with pytest.raises(ValueError):
            projector.squared_harmonics(alm)

    @pytest.mark.parametrize("circular_mask", [True, False])
    def test_r_den_is_finite_positive_and_pinned(self, circular_mask, record_property):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=circular_mask
        )
        master = ni_master_harmonics(NI_BANDWIDTH)
        correlator = _xcorr.NormalizedSphericalCrossCorrelator(
            NI_BANDWIDTH,
            master.alm,
            projector.squared_harmonics(master.alm),
            master.n_fold,
            master.has_equatorial_mirror,
            projector.window_harmonics,
        )
        r_den = correlator.r_den
        assert np.isfinite(r_den).all()
        assert r_den.min() > 0
        low, high = R_DEN_LIMITS[circular_mask]
        record_property(
            f"r_den_circle_{circular_mask}", f"{r_den.min():.4f} / {r_den.max():.4f}"
        )
        assert float(r_den.min()) == pytest.approx(low, rel=0.05)
        assert float(r_den.max()) == pytest.approx(high, rel=0.05)

    def test_signal_mask_changes_r_den(self, record_property):
        detector = ni_detector()
        mask = np.zeros((60, 60), dtype=bool)
        mask[MASK_BLOCK] = True
        master = ni_master_harmonics(NI_BANDWIDTH)
        plain = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True
        )
        masked = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=True, signal_mask=mask
        )
        correlators = []
        for projector in (plain, masked):
            correlators.append(
                _xcorr.NormalizedSphericalCrossCorrelator(
                    NI_BANDWIDTH,
                    master.alm,
                    projector.squared_harmonics(master.alm),
                    master.n_fold,
                    master.has_equatorial_mirror,
                    projector.window_harmonics,
                )
            )
        a, b = correlators[0].r_den, correlators[1].r_den
        change = float((np.abs(b - a) / a).max())
        record_property("r_den_relative_change_with_mask", f"{change:.4f}")
        assert change > 0.01
        assert np.isfinite(b).all()
        assert b.min() > 0


# ------------------- Forward-projection convention ------------------ #


class TestForwardProjectionLock:
    @staticmethod
    def _run_lock(bandwidth, record_property):
        detector = ni_detector()
        master = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        harmonics = ni_master_harmonics(bandwidth)
        rotations = Rotation.from_euler(np.deg2rad(np.asarray(LOCK_EULER_DEGREES)))
        simulated = master.get_patterns(
            rotations, detector, energy=20, compute=True, show_progressbar=False
        )
        patterns = simulated.data.reshape(-1, detector.nrows, detector.ncols)

        projector = _back_projection.SphericalBackProjector(detector, bandwidth)
        normalized = _xcorr.NormalizedSphericalCrossCorrelator(
            bandwidth,
            harmonics.alm,
            projector.squared_harmonics(harmonics.alm),
            harmonics.n_fold,
            harmonics.has_equatorial_mirror,
            projector.window_harmonics,
        )
        plain = _xcorr.SphericalCrossCorrelator(bandwidth)

        results = {"normalized": {"inverse": [], "direct": [], "score": []}}
        results["plain"] = {"inverse": [], "direct": [], "score": []}
        for index, pattern in enumerate(patterns):
            north, south = projector.unproject(pattern)
            gln = projector.sht.analyze(north, south)
            zyz_n, score_n = normalized.correlate(gln)
            zyz_p, score_p = plain.correlate(
                harmonics.alm,
                gln,
                harmonics.n_fold,
                harmonics.has_equatorial_mirror,
            )
            for name, zyz, score in (
                ("normalized", zyz_n, score_n),
                ("plain", zyz_p, score_p),
            ):
                results[name]["score"].append(float(score))
                for sign in ("inverse", "direct"):
                    results[name][sign].append(
                        misorientation_deg(rotations[index], zyz, sign)
                    )
        for name, values in results.items():
            for sign in ("inverse", "direct"):
                angles = np.asarray(values[sign])
                record_property(
                    f"lock_bw{bandwidth}_{name}_{sign}",
                    f"median {np.median(angles):.3f} max {angles.max():.3f}",
                )
            scores = np.asarray(values["score"])
            record_property(
                f"lock_bw{bandwidth}_{name}_scores",
                f"{scores.min():.4f} - {scores.max():.4f}",
            )
        return results

    def test_the_boundary_orientation_is_the_conjugated_one(self, record_property):
        results = self._run_lock(NI_BANDWIDTH, record_property)
        for name, values in results.items():
            correct = np.asarray(values["inverse"])
            wrong = np.asarray(values["direct"])
            assert correct.max() < 2.0
            assert float(np.median(correct)) < 1.0
            # a silent conjugation flip must not survive
            assert float(np.median(wrong)) > 10.0
            scores = np.asarray(values["score"])
            low, high = LOCK_SCORES[name]
            assert float(scores.min()) == pytest.approx(low, rel=0.05)
            assert float(scores.max()) == pytest.approx(high, rel=0.05)

    @pytest.mark.weekly
    def test_the_convention_lock_at_bandwidth_53(self, record_property):
        results = self._run_lock(53, record_property)
        for values in results.values():
            correct = np.asarray(values["inverse"])
            assert correct.max() < 2.0
            assert float(np.median(correct)) < 1.0
            assert float(np.median(np.asarray(values["direct"]))) > 10.0

    @pytest.mark.parametrize(
        "case, pixels",
        [
            (None, [(10, 45), (50, 5)]),
            (((60, 60), (0.55, 0.65, 0.6), 70.0, 0.0), [(15, 40), (45, 12)]),
        ],
        ids=["nickel", "pc_below_centre"],
    )
    def test_the_asymmetric_blob_pins_rows_and_columns(
        self, case, pixels, record_property
    ):
        detector = ni_detector() if case is None else synthetic_detector(*case)
        projector = _back_projection.SphericalBackProjector(
            detector, NI_BANDWIDTH, circular_mask=False
        )
        dc = direction_cosines(detector)
        normals = _grid.legendre_normals(projector.dim).reshape(-1, 3)
        tolerance = 2 * 180 / projector.dim
        for r0, c0 in pixels:
            pattern = np.zeros(detector.shape)
            pattern[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2] = 100.0
            north, _ = projector.unproject(pattern)
            found = normals[int(np.argmax(north))]
            true_angle = angle_between_deg(found, dc[r0, c0])
            vertical = angle_between_deg(found, dc[59 - r0, c0])
            horizontal = angle_between_deg(found, dc[r0, 59 - c0])
            transposed = angle_between_deg(found, dc[c0, r0])
            record_property(
                f"blob_{r0}_{c0}",
                f"true {true_angle:.2f} vflip {vertical:.2f} "
                f"hflip {horizontal:.2f} transpose {transposed:.2f}",
            )
            assert true_angle < tolerance
            assert vertical > 20
            assert horizontal > 20
            assert transposed > 20


# ----------------------------- Real data ---------------------------- #


class TestNickelSmall:
    def test_the_nine_patterns(self, record_property):
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        qualities = []
        for pattern in _ni_signal_data().reshape(-1, 60, 60):
            north, south, iq = projector.unproject(pattern, return_image_quality=True)
            assert np.count_nonzero(south) == 0
            assert np.count_nonzero(north) == projector.n_points
            assert np.isfinite(north).all()
            mean, second = weighted_moments(projector, north)
            assert abs(mean) <= 1e-10
            assert second == pytest.approx(1.0, abs=1e-10)
            qualities.append(iq)
        record_property("ni_iq", ", ".join(f"{v:.4f}" for v in qualities))
        assert min(qualities) >= 0.7
        assert max(qualities) <= 0.85


class TestErrorFloor:
    @staticmethod
    def _refine(signal, detector, master, navigation_mask=None):
        return signal.refine_orientation(
            xmap=signal.xmap,
            detector=detector,
            master_pattern=master,
            energy=20,
            navigation_mask=navigation_mask,
            method=refinement_method(),
            trust_region=[2, 2, 2],
            rtol=1e-6,
            compute=True,
        )

    def test_the_mean_pc_floor_on_the_small_map(self, record_property):
        record_property("refinement_method", refinement_method())
        signal = kp.data.nickel_ebsd_small()
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        master = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both", energy=20
        )
        per_point = signal.detector
        averaged = per_point.deepcopy()
        averaged.pc = per_point.pc_average
        spread = np.abs(per_point.pc - per_point.pc_average).max(axis=(0, 1))
        record_property("pc_spread_small", np.array2string(spread, precision=4))

        stored = Orientation(
            signal.xmap.rotations.data, signal.xmap.phases[0].point_group
        )
        refined_pp = self._refine(signal, per_point, master)
        refined_av = self._refine(signal, averaged, master)

        control = refined_pp.orientations.angle_with(stored, degrees=True)
        record_property(
            "small_per_point_vs_stored",
            f"median {np.median(control):.3f} max {control.max():.3f}",
        )
        assert float(np.median(control)) < 0.2
        assert float(control.max()) < 0.5

        against_stored = refined_av.orientations.angle_with(stored, degrees=True)
        record_property(
            "small_average_vs_stored",
            f"median {np.median(against_stored):.3f} max {against_stored.max():.3f}",
        )
        assert float(np.median(against_stored)) < 0.6
        assert float(against_stored.max()) < 1.0

        floor = refined_av.orientations.angle_with(
            refined_pp.orientations, degrees=True
        )
        record_property(
            "small_error_floor",
            f"median {np.median(floor):.3f} max {floor.max():.3f}",
        )
        assert float(np.median(floor)) < 0.6
        assert float(floor.max()) < 1.0

    @staticmethod
    def _large_floor(step, record_property, tag):
        pytest.importorskip("pooch")
        record_property("refinement_method", refinement_method())
        signal = kp.data.nickel_ebsd_large(allow_download=True)
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        master = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both", energy=20
        )
        navigation_mask = np.ones(
            signal.axes_manager.navigation_shape[::-1], dtype=bool
        )
        navigation_mask[::step, ::step] = False
        record_property(f"{tag}_points", str(int((~navigation_mask).sum())))

        per_point = signal.detector
        averaged = per_point.deepcopy()
        averaged.pc = per_point.pc_average
        refined_pp = TestErrorFloor._refine(signal, per_point, master, navigation_mask)
        refined_av = TestErrorFloor._refine(signal, averaged, master, navigation_mask)

        stored = Orientation(
            signal.xmap.rotations[~navigation_mask.ravel()].data,
            signal.xmap.phases[0].point_group,
        )
        control = refined_pp.orientations.angle_with(stored, degrees=True)
        floor = refined_av.orientations.angle_with(
            refined_pp.orientations, degrees=True
        )
        against_stored = refined_av.orientations.angle_with(stored, degrees=True)

        rows, columns = np.nonzero(~navigation_mask)
        deviation = np.linalg.norm(
            (per_point.pc - per_point.pc_average)[rows, columns], axis=-1
        )
        record_property(
            f"{tag}_per_point_vs_stored",
            f"median {np.median(control):.3f} max {control.max():.3f}",
        )
        record_property(
            f"{tag}_average_vs_stored",
            f"median {np.median(against_stored):.3f} "
            f"p95 {np.percentile(against_stored, 95):.3f} "
            f"max {against_stored.max():.3f}",
        )
        record_property(
            f"{tag}_floor",
            f"median {np.median(floor):.3f} p90 {np.percentile(floor, 90):.3f} "
            f"p95 {np.percentile(floor, 95):.3f} max {floor.max():.3f}",
        )
        record_property(
            f"{tag}_correlation",
            f"{np.corrcoef(np.asarray(floor).ravel(), deviation)[0, 1]:.3f}",
        )
        return np.asarray(control).ravel(), np.asarray(floor).ravel(), deviation

    def test_the_mean_pc_floor_on_a_large_map_subset(self, record_property):
        control, floor, _ = self._large_floor(15, record_property, "large20")
        assert float(np.median(control)) < 0.2
        assert float(np.median(floor)) < 0.6
        assert float(floor.max()) < 1.5

    @pytest.mark.weekly
    def test_the_mean_pc_floor_on_the_weekly_large_subset(self, record_property):
        _, floor, deviation = self._large_floor(5, record_property, "large165")
        assert float(np.median(floor)) < 0.6
        assert float(np.percentile(floor, 95)) < 1.2
        assert float(floor.max()) < 1.5
        assert float(np.corrcoef(floor, deviation)[0, 1]) > 0.8


# ------------------------- Kernels and flags ------------------------ #


class TestKernels:
    def test_kernel_names_lists_every_njit_kernel_of_the_module(self):
        # the flag and py_func tests are parametrised over the
        # literal list above, so a kernel added during the
        # implementation would silently escape both of them
        assert _njit_kernel_names(_back_projection) == sorted(KERNEL_NAMES), (
            "KERNEL_NAMES must list exactly the @njit kernels of _back_projection"
        )

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_kernels_are_compiled_with_cache_and_nogil(self, name):
        kernel = getattr(_back_projection, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("nogil") is True, f"{name} needs nogil=True"
        assert type(kernel._cache).__name__ == "FunctionCache", (
            f"{name} needs cache=True"
        )
        assert not kernel.targetoptions.get("parallel", False)
        assert not kernel.targetoptions.get("fastmath", False)

    @pytest.mark.parametrize("name", KERNEL_NAMES)
    def test_no_kernel_of_this_module_uses_the_numpy_error_model(self, name):
        # nothing here divides by a quantity which may be exactly
        # zero outside the guarded branch
        kernel = getattr(_back_projection, name)
        assert hasattr(kernel, "targetoptions"), f"{name} must be decorated with @njit"
        assert kernel.targetoptions.get("error_model") is None

    def test_phase_four_keeps_its_single_error_model(self):
        # regression: the second sanctioned error model of the
        # project arrives in _preprocessing, not here
        for name in _njit_kernel_names(_xcorr):
            kernel = getattr(_xcorr, name)
            expected = "numpy" if name == "_interpolate_maxima" else None
            assert kernel.targetoptions.get("error_model") == expected

    def test_unproject_kernel_py_func(self, record_property):
        kernel = _back_projection._unproject_kernel
        assert hasattr(kernel, "py_func"), "kernel must be @njit-decorated"
        projector = _back_projection.SphericalBackProjector(
            ni_detector(), NI_BANDWIDTH, circular_mask=True
        )
        h_out, w_out = projector.rescaled_shape
        pattern = _ni_signal_data()[0, 0].astype(np.float64)
        rescaled, _ = _back_projection._dct_rescale(pattern, h_out, w_out)
        flat = np.ascontiguousarray(rescaled).ravel()
        results = []
        for function in (kernel, _py_func(kernel)):
            north = np.zeros(projector.dim**2)
            stdev = function(
                flat,
                projector.pixel_index,
                projector.weights,
                projector.solid_angles,
                projector.window_solid_angle,
                projector.sphere_index,
                north,
            )
            results.append((north, stdev))
        difference = float(np.abs(results[0][0] - results[1][0]).max())
        record_property("unproject_py_func_max_difference", f"{difference:.3e}")
        record_property(
            "unproject_py_func_bitwise",
            str(bool(np.array_equal(results[0][0], results[1][0]))),
        )
        assert difference <= 4 * EPS
        assert results[0][1] == pytest.approx(results[1][1], rel=4 * EPS)


class TestBaselines:
    @pytest.mark.parametrize(
        "bandwidth", [53, 68, 88, pytest.param(113, marks=pytest.mark.weekly)]
    )
    def test_timings_are_recorded(self, bandwidth, record_property):
        detector = ni_detector()
        start = time.perf_counter()
        projector = _back_projection.SphericalBackProjector(
            detector, bandwidth, circular_mask=True
        )
        construction = time.perf_counter() - start
        record_property(f"construction_seconds_bw{bandwidth}", f"{construction:.4f}")
        assert construction < 5.0

        pattern = _ni_signal_data()[0, 0]
        north = np.zeros((projector.dim, projector.dim))
        south = np.zeros((projector.dim, projector.dim))
        projector.unproject(pattern, out=(north, south))  # warm the cache

        best = math.inf
        for _ in range(20):
            start = time.perf_counter()
            projector.unproject(pattern, out=(north, south), return_image_quality=True)
            best = min(best, time.perf_counter() - start)
        record_property(f"unproject_seconds_bw{bandwidth}", f"{best:.6f}")
        assert best < 0.05

        start = time.perf_counter()
        projector.sht.analyze(north, south)
        record_property(
            f"analyze_seconds_bw{bandwidth}", f"{time.perf_counter() - start:.6f}"
        )

    @pytest.mark.parametrize(
        "bandwidth", [68, pytest.param(113, marks=pytest.mark.weekly)]
    )
    def test_memory_is_recorded(self, bandwidth, record_property):
        detector = ni_detector()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            projector = _back_projection.SphericalBackProjector(
                detector, bandwidth, circular_mask=True
            )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        lut_bytes = (
            projector.sphere_index.nbytes
            + projector.pixel_index.nbytes
            + projector.weights.nbytes
            + projector.solid_angles.nbytes
        )
        record_property(f"construct_peak_mb_bw{bandwidth}", f"{peak / 1e6:.1f}")
        record_property(f"lut_mb_bw{bandwidth}", f"{lut_bytes / 1e6:.3f}")
        assert lut_bytes < 5e6
