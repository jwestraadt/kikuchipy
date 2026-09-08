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
from scipy.ndimage import spline_filter

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
    if interpolation not in SUPPORTED_INTERPOLATION:
        raise ValueError(
            f"interpolation must be one of {SUPPORTED_INTERPOLATION}, not "
            f"{interpolation!r}"
        )
    image64 = np.asarray(image, dtype=np.float64)
    # The prefilter is always run in 64-bit floats and only then cast:
    # the 32-bit bulk storage of requirements D17 is a STORAGE
    # decision, never a lower precision prefilter
    coefficients = spline_filter(
        image64, order=SPLINE_ORDERS[interpolation], mode=BOUNDARY_MODE
    )
    return coefficients.astype(dtype, copy=False)


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
    n_rows, n_cols = coefficients.shape
    period_rows = 2 * n_rows - 2
    period_cols = 2 * n_cols - 2
    columns = np.empty(4, dtype=np.int64)
    rows = np.empty(4, dtype=np.int64)
    weights_x = np.empty(4, dtype=np.float64)
    weights_y = np.empty(4, dtype=np.float64)
    for i in range(x.size):
        # Fold the sample into the frame through the whole-sample
        # symmetric mirror boundary, ``d c b | a b c d | c b a``
        cx = x[i]
        if cx < 0.0:
            cx = -cx
        cx = cx % period_cols
        if cx > n_cols - 1:
            cx = period_cols - cx
        cy = y[i]
        if cy < 0.0:
            cy = -cy
        cy = cy % period_rows
        if cy > n_rows - 1:
            cy = period_rows - cy
        ix = int(np.floor(cx))
        iy = int(np.floor(cy))
        tx = cx - ix
        ty = cy - iy
        # The four cubic B-spline basis values of each axis
        one_minus = 1.0 - tx
        tx2 = tx * tx
        tx3 = tx2 * tx
        weights_x[0] = one_minus * one_minus * one_minus / 6.0
        weights_x[1] = (3.0 * tx3 - 6.0 * tx2 + 4.0) / 6.0
        weights_x[2] = (-3.0 * tx3 + 3.0 * tx2 + 3.0 * tx + 1.0) / 6.0
        weights_x[3] = tx3 / 6.0
        one_minus = 1.0 - ty
        ty2 = ty * ty
        ty3 = ty2 * ty
        weights_y[0] = one_minus * one_minus * one_minus / 6.0
        weights_y[1] = (3.0 * ty3 - 6.0 * ty2 + 4.0) / 6.0
        weights_y[2] = (-3.0 * ty3 + 3.0 * ty2 + 3.0 * ty + 1.0) / 6.0
        weights_y[3] = ty3 / 6.0
        # The support of the basis, folded through the same boundary
        for k in range(4):
            column = ix - 1 + k
            if column < 0:
                column = -column
            column = column % period_cols
            if column > n_cols - 1:
                column = period_cols - column
            columns[k] = column
            row = iy - 1 + k
            if row < 0:
                row = -row
            row = row % period_rows
            if row > n_rows - 1:
                row = period_rows - row
            rows[k] = row
        # 64-bit accumulation whatever the coefficient data type is
        value = 0.0
        for a in range(4):
            row = rows[a]
            inner = (
                weights_x[0] * coefficients[row, columns[0]]
                + weights_x[1] * coefficients[row, columns[1]]
                + weights_x[2] * coefficients[row, columns[2]]
                + weights_x[3] * coefficients[row, columns[3]]
            )
            value += weights_y[a] * inner
        out[i] = value


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
    n_rows, n_cols = coefficients.shape
    period_rows = 2 * n_rows - 2
    period_cols = 2 * n_cols - 2
    columns = np.empty(4, dtype=np.int64)
    rows = np.empty(4, dtype=np.int64)
    weights_x = np.empty(4, dtype=np.float64)
    weights_y = np.empty(4, dtype=np.float64)
    derivatives_x = np.empty(4, dtype=np.float64)
    derivatives_y = np.empty(4, dtype=np.float64)
    for i in range(x.size):
        # Fold the sample into the frame, keeping the sign of the
        # derivative of the folding: a reversed axis reverses the
        # gradient with respect to the ORIGINAL coordinate
        cx = x[i]
        sign_x = 1.0
        if cx < 0.0:
            cx = -cx
            sign_x = -1.0
        cx = cx % period_cols
        if cx > n_cols - 1:
            cx = period_cols - cx
            sign_x = -sign_x
        cy = y[i]
        sign_y = 1.0
        if cy < 0.0:
            cy = -cy
            sign_y = -1.0
        cy = cy % period_rows
        if cy > n_rows - 1:
            cy = period_rows - cy
            sign_y = -sign_y
        ix = int(np.floor(cx))
        iy = int(np.floor(cy))
        tx = cx - ix
        ty = cy - iy
        # The basis values and the ANALYTIC derivatives of the basis
        # itself, never a finite difference of the sampled image.  The
        # four derivatives sum to exactly zero in floating point, so a
        # locally constant image has an exactly zero gradient
        one_minus = 1.0 - tx
        tx2 = tx * tx
        tx3 = tx2 * tx
        weights_x[0] = one_minus * one_minus * one_minus / 6.0
        weights_x[1] = (3.0 * tx3 - 6.0 * tx2 + 4.0) / 6.0
        weights_x[2] = (-3.0 * tx3 + 3.0 * tx2 + 3.0 * tx + 1.0) / 6.0
        weights_x[3] = tx3 / 6.0
        derivatives_x[0] = -0.5 * one_minus * one_minus
        derivatives_x[1] = 1.5 * tx2 - 2.0 * tx
        derivatives_x[2] = -1.5 * tx2 + tx + 0.5
        derivatives_x[3] = 0.5 * tx2
        one_minus = 1.0 - ty
        ty2 = ty * ty
        ty3 = ty2 * ty
        weights_y[0] = one_minus * one_minus * one_minus / 6.0
        weights_y[1] = (3.0 * ty3 - 6.0 * ty2 + 4.0) / 6.0
        weights_y[2] = (-3.0 * ty3 + 3.0 * ty2 + 3.0 * ty + 1.0) / 6.0
        weights_y[3] = ty3 / 6.0
        derivatives_y[0] = -0.5 * one_minus * one_minus
        derivatives_y[1] = 1.5 * ty2 - 2.0 * ty
        derivatives_y[2] = -1.5 * ty2 + ty + 0.5
        derivatives_y[3] = 0.5 * ty2
        for k in range(4):
            column = ix - 1 + k
            if column < 0:
                column = -column
            column = column % period_cols
            if column > n_cols - 1:
                column = period_cols - column
            columns[k] = column
            row = iy - 1 + k
            if row < 0:
                row = -row
            row = row % period_rows
            if row > n_rows - 1:
                row = period_rows - row
            rows[k] = row
        value = 0.0
        gradient_x = 0.0
        gradient_y = 0.0
        for a in range(4):
            row = rows[a]
            c0 = coefficients[row, columns[0]]
            c1 = coefficients[row, columns[1]]
            c2 = coefficients[row, columns[2]]
            c3 = coefficients[row, columns[3]]
            inner = (
                weights_x[0] * c0
                + weights_x[1] * c1
                + weights_x[2] * c2
                + weights_x[3] * c3
            )
            inner_derivative = (
                derivatives_x[0] * c0
                + derivatives_x[1] * c1
                + derivatives_x[2] * c2
                + derivatives_x[3] * c3
            )
            value += weights_y[a] * inner
            gradient_x += weights_y[a] * inner_derivative
            gradient_y += derivatives_y[a] * inner
        out[i] = value
        out_gx[i] = sign_x * gradient_x
        out_gy[i] = sign_y * gradient_y


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
    coefficients, x_flat, y_flat, shape = _prepare(coefficients, x, y)
    if out is None:
        out = np.empty(shape, dtype=np.float64)
    writes_through = out.dtype == np.float64 and out.flags.c_contiguous
    out_flat = out.reshape(-1) if writes_through else np.empty(x_flat.size)
    _bicubic_evaluate(coefficients, x_flat, y_flat, out_flat)
    if not writes_through:
        out[...] = out_flat.reshape(shape)
    return out


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
    coefficients, x_flat, y_flat, shape = _prepare(coefficients, x, y)
    values = np.empty(x_flat.size, dtype=np.float64)
    gradient_x = np.empty(x_flat.size, dtype=np.float64)
    gradient_y = np.empty(x_flat.size, dtype=np.float64)
    _bicubic_evaluate_gradient(
        coefficients, x_flat, y_flat, values, gradient_x, gradient_y
    )
    return (
        values.reshape(shape),
        gradient_x.reshape(shape),
        gradient_y.reshape(shape),
    )


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
    shape = np.shape(coefficients)
    rows, cols = np.indices(shape)
    _, gradient_x, gradient_y = evaluate_gradient(
        coefficients,
        cols.ravel().astype(np.float64),
        rows.ravel().astype(np.float64),
    )
    return gradient_x.reshape(shape), gradient_y.reshape(shape)


def _prepare(
    coefficients: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return kernel ready coefficients, flat coordinates and the
    shape the wrappers restore.

    The kernels take contiguous 1D 64-bit float coordinates and a
    contiguous 2D coefficient array of either float data type
    (requirements D17), so every wrapper funnels through here.

    Parameters
    ----------
    coefficients
        2D coefficient array from :func:`spline_coefficients`.
    x
        Array of column coordinates, of any shape.
    y
        Array of row coordinates, broadcastable against *x*.

    Returns
    -------
    coefficients
        The same array made C contiguous.
    x_flat
        Contiguous flat 64-bit float column coordinates.
    y_flat
        Contiguous flat 64-bit float row coordinates.
    shape
        The broadcast shape the wrappers reshape their output to.

    Raises
    ------
    ValueError
        If *x* and *y* do not broadcast against each other. They are
        broadcast FIRST, so a mismatch raises before the kernel can
        read past the end of one of them.
    """
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    return (
        np.ascontiguousarray(coefficients),
        np.ascontiguousarray(x).ravel(),
        np.ascontiguousarray(y).ravel(),
        x.shape,
    )
