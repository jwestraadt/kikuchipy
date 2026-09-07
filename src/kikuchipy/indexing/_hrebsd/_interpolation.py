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

"""Cubic B-spline interpolation for the HREBSD-DIC engine.

:func:`scipy.ndimage.map_coordinates` cannot be called from a numba
compiled function, so the inner loop of the IC-GN engine evaluates a
hand written kernel over coefficients produced once per pattern by
:func:`scipy.ndimage.spline_filter`.  The kernel is pinned against
``map_coordinates(order=3, prefilter=False, mode="mirror")``
(requirements D3, validation V0).

The interpolation order is a measured-then-pinned decision (D3, plan
open question 1): ``"bicubic"`` is the default in force, the
``interpolation`` keyword stays whatever the measurement says.

References
----------
Ernould et al., Acta Mater. 191 (2020) 131-148; Ruggles et al.,
Ultramicroscopy 195 (2018) 85 (biquintic versus bicubic at 960 px).
"""

from numba import njit
import numpy as np

# The interpolation names accepted by every public entry point, and
# the spline order each maps to (D3; a quintic entry is added here if
# and only if the Stage A measurement re-pins the default)
SUPPORTED_INTERPOLATION: tuple[str, ...] = ("bicubic",)
SPLINE_ORDERS: dict[str, int] = {"bicubic": 3}

# The boundary mode of both the prefilter and the kernel, frozen (D3):
# out-of-frame samples during a warp evaluate through the mirror
# boundary, never through b-spline extrapolation (the EMsoftOO choice,
# a recorded deviation)
BOUNDARY_MODE: str = "mirror"


def spline_coefficients(
    image: np.ndarray,
    *,
    interpolation: str = "bicubic",
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Return the cubic B-spline coefficients of one pattern.

    Parameters
    ----------
    image
        Pattern of shape ``(nrows, ncols)``.
    interpolation
        One of :data:`SUPPORTED_INTERPOLATION`. Default is
        ``"bicubic"``.
    dtype
        Data type of the returned coefficients, 64-bit float by
        default. 32-bit float is the provisional bulk storage type of
        requirements D17, measured then pinned.

    Returns
    -------
    coefficients
        Array of the shape of *image* and data type *dtype*, holding
        the prefiltered coefficients such that evaluating the cubic
        B-spline basis on them reproduces *image* at the pixel
        centres.

    Raises
    ------
    ValueError
        If *interpolation* is not in :data:`SUPPORTED_INTERPOLATION`.

    Notes
    -----
    Equivalent to ``scipy.ndimage.spline_filter(image, order=3,
    mode="mirror")`` up to the output data type, which is the pinned
    reference of validation V0.
    """
    raise NotImplementedError


@njit(cache=True, nogil=True)
def _bicubic_evaluate(
    coefficients: np.ndarray, x: np.ndarray, y: np.ndarray, out: np.ndarray
) -> None:
    """Evaluate the cubic B-spline of *coefficients* at ``(x, y)``.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.
    x
        1D array of column coordinates, right positive, origin at the
        centre of the upper left pixel (requirements D1.1).
    y
        1D array of row coordinates, down positive, of the size of
        *x*.
    out
        1D 64-bit float array of the size of *x*, written in place.

    Notes
    -----
    This function is optimized with numba, so care must be taken with
    array shapes and data types.  Accumulation is 64-bit whatever the
    coefficient data type is (requirements D17).  Coordinates outside
    the frame are folded back through the ``"mirror"`` boundary, the
    exact convention of ``map_coordinates(mode="mirror")``.
    """
    raise NotImplementedError


@njit(cache=True, nogil=True)
def _bicubic_evaluate_gradient(
    coefficients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    out: np.ndarray,
    out_gx: np.ndarray,
    out_gy: np.ndarray,
) -> None:
    """Evaluate the cubic B-spline and its analytic first derivatives.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.
    x
        1D array of column coordinates.
    y
        1D array of row coordinates, of the size of *x*.
    out
        1D 64-bit float array of the size of *x*, written in place
        with the interpolated values.
    out_gx
        1D 64-bit float array written in place with the derivative
        with respect to the column coordinate.
    out_gy
        1D 64-bit float array written in place with the derivative
        with respect to the row coordinate.

    Notes
    -----
    This function is optimized with numba, so care must be taken with
    array shapes and data types.  The derivatives are those of the
    B-spline basis itself, never a finite difference of the sampled
    image.
    """
    raise NotImplementedError


def evaluate(
    coefficients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return the cubic B-spline of *coefficients* at ``(x, y)``.

    The array allocating wrapper of :func:`_bicubic_evaluate`.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.
    x
        Array of column coordinates, of any shape.
    y
        Array of row coordinates, of the shape of *x*.
    out
        Optional 64-bit float array of the shape of *x* to write
        into. A new array is allocated if not given.

    Returns
    -------
    values
        Interpolated values of the shape of *x* and 64-bit float data
        type.
    """
    raise NotImplementedError


def evaluate_gradient(
    coefficients: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values and analytic gradients at ``(x, y)``.

    The array allocating wrapper of
    :func:`_bicubic_evaluate_gradient`.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.
    x
        Array of column coordinates, of any shape.
    y
        Array of row coordinates, of the shape of *x*.

    Returns
    -------
    values
        Interpolated values of the shape of *x*.
    gradient_x
        Derivative with respect to the column coordinate.
    gradient_y
        Derivative with respect to the row coordinate.
    """
    raise NotImplementedError


def gradient_planes(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic spline gradients on the pixel grid.

    The reference gradients of the IC-GN precompute (requirements
    D2.1) are the analytic B-spline derivatives evaluated at the
    integer pixel centres of the whole pattern, not a finite
    difference.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.

    Returns
    -------
    gradient_x
        Derivative with respect to the column coordinate, of the
        shape of *coefficients* and 64-bit float data type.
    gradient_y
        Derivative with respect to the row coordinate.
    """
    raise NotImplementedError
