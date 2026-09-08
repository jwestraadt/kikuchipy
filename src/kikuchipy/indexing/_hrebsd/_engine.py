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

"""The inverse-compositional Gauss-Newton DIC engine.

Per target pattern, the loop of requirements D2:

1. Per-reference precompute, once per grain: interpolation
   coefficients, subregion gradients from analytic spline
   derivatives, the zero-mean unit-norm reference vector and its
   norm, the steepest-descent images
   ``GJ = (gx*x, gx*y, gx, gy*x, gy*y, gy,
   -(gx*x**2 + gy*x*y), -(gx*x*y + gy*y**2))`` in the standard
   Pan and Ernould signs, the Hessian
   ``H = (2/ref_norm**2) * sum_px GJ GJ^T`` and its Cholesky factor.
2. Initialize the accumulated warp from the phase cross-correlation
   seed.
3. Iterate: warp the ORIGINAL target by the ACCUMULATED warp,
   zero-mean unit-norm the warped subregion, form
   ``residuals = ref_zmn - tar_zmn`` and ``CIC = sum(residuals**2)``,
   the gradient ``g = (2/ref_norm) * GJ^T residuals``, solve
   ``H dp = -g`` through the cached Cholesky factor, update
   ``W <- W . W(dp * step_scale)**-1`` and renormalize ``W[2, 2]``.
4. On exit, evaluate the criterion once more at the accumulated warp,
   so that the reported ``CIC`` belongs to the returned homography
   and not to the iterate before the last composition (D2.7).

The ``2/ref_norm**2`` Hessian scale paired with the ``2/ref_norm``
gradient scale is NOT cosmetic: a scale on the gradient alone does
not cancel in ``H**-1 g``, it makes the step depend on the raw
intensity scale and can spuriously satisfy the exit criterion at the
initial guess.  The matched pairing makes the step invariant under an
affine intensity rescale, which validation V2 pins bitwise with
powers of two.

Two frozen deviations from EMsoftOO, both recorded: the accumulated
warp re-warps the ORIGINAL target every iteration instead of warping
the warped target with re-splining and re-normalization
(``mod_DIC.f90:684-747``), which stops interpolation smoothing from
accumulating; and non-converged points keep their last iterate with
``converged=False`` and NaN in every derived property downstream,
they are NEVER zeroed (``mod_HREBSDDIC.f90:868-870``).  No bit parity
with EMsoftOO is claimed anywhere.

References
----------
Ernould et al., Acta Mater. 191 (2020) 131-148; Ernould et al., AIEP
223 (2022) Ch. 2; Pan, Meas. Sci. Technol. 29 (2018) 082001; Ruggles
et al., Ultramicroscopy 195 (2018) 85.
"""

import math
import time
import warnings

import dask
import dask.array as da
from dask.diagnostics.progress import ProgressBar
from dask.system import CPU_COUNT
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from kikuchipy.indexing._hrebsd._geometry import fe_from_homography, per_point_pc_pixels
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    corner_norm,
    homography_parameters,
    shape_function,
)
from kikuchipy.indexing._hrebsd._interpolation import (
    SUPPORTED_INTERPOLATION,
    evaluate,
    gradient_planes,
    spline_coefficients,
)
from kikuchipy.indexing._hrebsd._preprocessing import (
    band_pass_transfer_function,
    hann_window,
    preprocess,
    subregion_bounds,
    subregion_mask,
    zero_mean_normalize,
)
from kikuchipy.indexing._hrebsd._reference import resolve_reference

# The Stage A crystal map properties, in the order the engine fills
# them (requirements D15.6).  ``homography`` stores the RAW fitted
# parameters, since the PC-shift analysis of Stage B needs the real
# beam-scan translations; the beam-scan correction is applied
# transiently on the ``Fe`` path only
STAGE_A_PROP_NAMES: tuple[str, ...] = (
    "homography",
    "Fe",
    "residual",
    "num_iterations",
    "norm_dp",
    "converged",
    "grain_id",
    "reference_index",
)

# Second axis length of the two dimensional Stage A properties
HOMOGRAPHY_PROP_SIZE: int = 8
FE_PROP_SIZE: int = 9

# Width of one packed result row of the dask chunks: the eight
# homography parameters, the final CIC, the iteration count, the final
# corner norm and the convergence flag.  One 64-bit float array keeps
# the graph's ``chunks=`` metadata truthful and the whole run bitwise
# reproducible whatever the chunking is
_ROW_WIDTH: int = HOMOGRAPHY_PROP_SIZE + 4

# Slots of that row
_SLOT_RESIDUAL: int = HOMOGRAPHY_PROP_SIZE
_SLOT_ITERATIONS: int = HOMOGRAPHY_PROP_SIZE + 1
_SLOT_NORM_DP: int = HOMOGRAPHY_PROP_SIZE + 2
_SLOT_CONVERGED: int = HOMOGRAPHY_PROP_SIZE + 3

# Columns of the steepest-descent block of one reference precompute,
# one per homography degree of freedom.  It equals
# :data:`HOMOGRAPHY_PROP_SIZE` numerically and means something else
# entirely, so the memory model of requirements D16 names it
# separately
_N_STEEPEST_DESCENT_COLUMNS: int = N_HOMOGRAPHY_PARAMETERS

# The 64-bit subregion sized planes a reference precompute holds
# BESIDES the steepest-descent block: the zero-mean unit-norm
# reference, the two coordinate planes and the reference subregion the
# D5 seed is measured on (requirements D16)
_RESIDENT_F64_PLANES: int = 4


class ReferenceState:
    """Per-reference precomputed state of the IC-GN loop.

    One instance per grain reference, reused by every target of that
    grain (requirements D2.1).  Twelve subregion sized 64-bit planes
    are resident, eight of them the steepest-descent columns, plus
    the 32-bit coefficient plane: MEASURED 17.96 MB at 480 by 480
    with the default border (re-measured 2026-09-07 at the Stage A
    adversarial review, which found the earlier 16.54 MB understated
    by the reference subregion the D5 seed is measured on; the
    drafted "about 4.6 MB" was 3.9x too small and is corrected in
    requirements D16 with that date).

    Parameters
    ----------
    pattern
        The reference pattern of shape ``(nrows, ncols)``.
    mask
        Boolean subregion mask of the shape of *pattern*, ``True``
        for the pixels used, from
        :func:`subregion_mask <kikuchipy.indexing._hrebsd\
._preprocessing.subregion_mask>`.
    pc_pixels
        ``(PCx_px, PCy_px, DD_px)`` of this reference, which defines
        the PC-centred coordinate frame shared by the reference and
        every target of the grain (requirements D1.3).
    transfer_function
        Band-pass transfer function applied to the reference and to
        every target alike, or ``None``.
    window
        Window of the shape of *pattern* from
        :func:`~kikuchipy.indexing._hrebsd._preprocessing.hann_window`,
        or ``None``. Its subregion values weight the ZNSSD residual
        and the steepest-descent images IN THE REFERENCE FRAME; it is
        never multiplied into a pattern before that pattern is
        warped (requirements D4.3, corrected 2026-09-07).
    interpolation
        Interpolation name, ``"bicubic"`` by default.
    coefficient_dtype
        Bulk storage data type of the spline coefficients and the
        gradient planes, 32-bit float by default. Every solver
        accumulator stays 64-bit whatever this says, and the 32-bit
        default is PROVISIONAL: requirements D17 pins it only if the
        homography recovery degradation measured by the dtype A/B
        harness of validation V2 stays below ten per cent of the
        64-bit error.

    Attributes
    ----------
    xi_x, xi_y
        1D PC-centred coordinates of the subregion pixels.
    corners
        ``(4, 2)`` subregion corner coordinates, the support of the
        convergence norm.
    reference
        Zero-mean unit-norm reference vector.
    reference_norm
        The 2-norm of the centred reference, the ``ref_norm`` of the
        matched Hessian and gradient scales.
    steepest_descent
        ``(n_pixels, 8)`` steepest-descent images, weighted by the
        window when one is given.
    weights
        The window's subregion values, or ``None``.
    hessian
        ``(8, 8)`` Gauss-Newton Hessian.
    cho_factor
        The cached :func:`scipy.linalg.cho_factor` of *hessian*.

    Raises
    ------
    ValueError
        If *interpolation* is not supported; if *pattern* is not two
        dimensional; if *mask* does not have the pattern shape or
        keeps no pixel at all; or if the reference subregion has no
        contrast to correlate.
    """

    def __init__(
        self,
        pattern: np.ndarray,
        mask: np.ndarray,
        pc_pixels: np.ndarray,
        *,
        transfer_function: np.ndarray | None = None,
        window: np.ndarray | None = None,
        interpolation: str = "bicubic",
        coefficient_dtype: np.dtype | type = np.float32,
    ) -> None:
        if interpolation not in SUPPORTED_INTERPOLATION:
            raise ValueError(
                f"interpolation must be one of {SUPPORTED_INTERPOLATION}, not "
                f"{interpolation!r}"
            )
        pattern = np.asarray(pattern)
        if pattern.ndim != 2:
            raise ValueError(
                f"the reference pattern must be two dimensional, not of shape "
                f"{pattern.shape}"
            )
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != pattern.shape:
            raise ValueError(
                f"the subregion mask of shape {mask.shape} must have the pattern "
                f"shape {pattern.shape}"
            )
        if not mask.any():
            raise ValueError("the subregion mask keeps no pixel at all")

        self.shape: tuple[int, int] = (int(pattern.shape[0]), int(pattern.shape[1]))
        self.mask = mask
        self.pc_pixels = np.asarray(pc_pixels, dtype=np.float64).ravel()
        self.transfer_function = transfer_function
        self.window = window
        self.interpolation = interpolation
        self.coefficient_dtype = np.dtype(coefficient_dtype)

        # The reference and every target of this grain go through ONE
        # identical chain (requirements D4)
        preprocessed = preprocess(pattern, transfer_function=transfer_function)

        # The PC-centred pixel frame of D1.3, with the half-pixel term
        # the Stage A measurement folded into D1.1: a Bruker fraction is
        # measured from the detector EDGE while a column INDEX names a
        # pixel CENTRE
        rows, cols = np.indices(self.shape)
        x_grid = cols + 0.5 - self.pc_pixels[0]
        y_grid = rows + 0.5 - self.pc_pixels[1]
        self.xi_x = np.ascontiguousarray(x_grid[mask])
        self.xi_y = np.ascontiguousarray(y_grid[mask])

        # The bounding box of the subregion, which the phase
        # cross-correlation seed of D5 is measured on
        kept_rows = np.flatnonzero(mask.any(axis=1))
        kept_cols = np.flatnonzero(mask.any(axis=0))
        self.bounds: tuple[int, int, int, int] = (
            int(kept_rows[0]),
            int(kept_rows[-1]) + 1,
            int(kept_cols[0]),
            int(kept_cols[-1]) + 1,
        )
        row_start, row_stop, col_start, col_stop = self.bounds
        self.reference_subregion = np.ascontiguousarray(
            preprocessed[row_start:row_stop, col_start:col_stop]
        )

        # The support of the frozen D2.5 convergence norm
        x0 = float(self.xi_x.min())
        x1 = float(self.xi_x.max())
        y0 = float(self.xi_y.min())
        y1 = float(self.xi_y.max())
        self.corners = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
        )

        self.reference, self.reference_norm = zero_mean_normalize(preprocessed[mask])

        # The D4.3 window is a per-pixel WEIGHT on the ZNSSD residual
        # in this one reference frame, so the zero-mean unit-norm
        # vectors above and below stay UNwindowed and the affine
        # intensity invariance of ZNSSD survives exactly.  Weighting
        # the residual by ``w`` weights the steepest-descent images by
        # the same ``w``, which leaves the D2.1 Hessian and D2.3
        # gradient formulas untouched
        self.weights = None
        if window is not None:
            self.weights = np.ascontiguousarray(
                np.asarray(window, dtype=np.float64)[mask]
            )

        coefficients = spline_coefficients(
            preprocessed, interpolation=interpolation, dtype=self.coefficient_dtype
        )
        self.coefficients = np.ascontiguousarray(coefficients)
        gradient_x, gradient_y = gradient_planes(self.coefficients)
        gx = gradient_x[mask]
        gy = gradient_y[mask]
        if self.weights is not None:
            gx = gx * self.weights
            gy = gy * self.weights

        # The STANDARD Pan and Ernould signs of requirements D2.1;
        # EMsoftOO's sign-structure variant follows its inverted shape
        # function and is a recorded deviation, never reproduced
        xi_x = self.xi_x
        xi_y = self.xi_y
        self.steepest_descent = np.stack(
            [
                gx * xi_x,
                gx * xi_y,
                gx,
                gy * xi_x,
                gy * xi_y,
                gy,
                -(gx * xi_x**2 + gy * xi_x * xi_y),
                -(gx * xi_x * xi_y + gy * xi_y**2),
            ],
            axis=1,
        )
        # The matched D2.1 scale.  It is NOT cosmetic: it must pair with
        # the ``2/ref_norm`` gradient scale of D2.3, or the step depends
        # on the raw intensity scale
        self.hessian = (2.0 / self.reference_norm**2) * (
            self.steepest_descent.T @ self.steepest_descent
        )
        # One Cholesky factorization per reference, reused by every
        # target and every iteration
        self.cho_factor = cho_factor(self.hessian)

    @property
    def n_pixels(self) -> int:
        """Return the number of subregion pixels correlated.

        Returns
        -------
        n_pixels
            The number of ``True`` entries of the subregion mask.
        """
        return int(self.xi_x.size)

    def memory_bytes(self) -> int:
        """Return the resident memory of this precompute in bytes.

        Returns
        -------
        memory
            The number requirements D16 puts in the information
            message: the spline coefficients in their storage data
            type, the 64-bit steepest-descent block, the zero-mean
            unit-norm reference, the two coordinate planes and the
            reference subregion the D5 seed is measured on.

        Notes
        -----
        Every array this instance ALLOCATES is counted (corrected
        2026-09-07, Stage A adversarial review: ``reference_subregion``
        was missing, a ten per cent understatement of a figure
        requirements D16 quotes as measured).  The arrays it merely
        HOLDS ON TO for the caller are not: the subregion mask, the
        band-pass transfer function and the window are built once per
        RUN and shared by every reference, so counting them per
        instance would multiply one allocation by the grain count.
        """
        n_pixels = self.n_pixels
        return int(
            self.coefficients.nbytes
            + self.steepest_descent.nbytes
            + self.reference_subregion.nbytes
            + 3 * n_pixels * np.dtype(np.float64).itemsize
        )


def initial_guess(
    reference: np.ndarray, target: np.ndarray, *, upsample_factor: int = 16
) -> np.ndarray:
    """Return the translation-only seed homography.

    :func:`skimage.registration.phase_cross_correlation` on the
    preprocessed subregions gives a shift which seeds
    ``W0 = W((0, 0, dx, 0, 0, dy, 0, 0))`` (requirements D5).
    EMsoftOO always starts from the identity
    (``mod_HREBSDDIC.f90:836-858``), which fails for large
    translations; this is a recorded deviation in kikuchipy's favour.

    Parameters
    ----------
    reference
        Preprocessed reference subregion of shape ``(nrows, ncols)``.
    target
        Preprocessed target subregion of the shape of *reference*.
    upsample_factor
        Subpixel precision of the seed, 1 over this in pixels.
        Default is 16, far inside the IC-GN basin.

    Returns
    -------
    h
        Homography parameters of shape ``(8,)`` whose only nonzero
        entries are ``h[2] = dx`` and ``h[5] = dy`` in binned pixels,
        the column and row translation carrying the reference onto
        the target.

    Raises
    ------
    ValueError
        If *reference* and *target* do not have the same shape, or if
        either carries no contrast to correlate.

    Notes
    -----
    ``skimage.registration.phase_cross_correlation`` is imported
    inside this function, not at module scope: it does not exist at
    the declared scikit-image floor 0.16.2 (it landed in 0.18), and
    requirements D18 records that the plain :class:`ImportError` at
    first use is the intended behaviour below the tested floor
    0.21.0.

    The seed's rotation capture range is measured on the pure
    rotation sweep of validation V4 and recorded in the
    ``EBSD.hrebsd_dic`` docstring; a Fourier-Mellin pre-rotation
    (Ernould 2020) is deferred.

    Both subregions are zero-mean unit-norm normalized first.  Phase
    correlation is scale invariant in exact arithmetic, but
    scikit-image floors the cross-power spectrum at an ABSOLUTE
    ``100 * eps``, so an unnormalized pair could seed two intensity
    rescalings of one fit differently and break the bitwise
    invariance validation V2 asserts.  The normalization costs
    nothing and removes that path entirely.
    """
    from skimage.registration import phase_cross_correlation

    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape != target.shape:
        raise ValueError(
            f"the reference of shape {reference.shape} and the target of shape "
            f"{target.shape} must have the same shape"
        )
    reference_zmn, _ = zero_mean_normalize(reference)
    target_zmn, _ = zero_mean_normalize(target)
    result = phase_cross_correlation(
        reference_zmn, target_zmn, upsample_factor=int(upsample_factor)
    )
    shift = np.asarray(result[0] if isinstance(result, tuple) else result)
    h = np.zeros(N_HOMOGRAPHY_PARAMETERS, dtype=np.float64)
    # ``phase_cross_correlation`` returns the shift which registers the
    # MOVING image with the reference, so the displacement carrying the
    # reference ONTO the target is its negative.  The row shift seeds
    # ``h23`` and the column shift ``h13``, never the other way round
    h[2] = -float(shift[1])
    h[5] = -float(shift[0])
    return h


def fit_pattern(
    state: ReferenceState,
    target: np.ndarray,
    *,
    h0: np.ndarray | None = None,
    upsample_factor: int = 16,
    max_iterations: int = 50,
    min_step: float = 1e-3,
    step_scale: float = 1.0,
    interpolation: str = "bicubic",
) -> dict:
    """Return the IC-GN fit of one target against one reference.

    Parameters
    ----------
    state
        The reference's precomputed :class:`ReferenceState`.
    target
        The raw target pattern of the shape of the reference. The
        preprocessing of *state* is applied to it here, so that the
        reference and every target go through one identical chain.
    h0
        Seed homography parameters of shape ``(8,)``. If not given,
        :func:`initial_guess` supplies the phase cross-correlation
        seed, which is what every engine call does; passing the
        explicit zero vector forces identity seeding, which is what
        the seed-required test of validation V2 exercises.
    upsample_factor
        Subpixel precision of the seed, passed on to
        :func:`initial_guess`. Default is 16.
    max_iterations
        Iteration cap. Default is 50, EMsoftOO's namelist default.
    min_step
        Convergence threshold on the corner norm of the increment
        warp, in binned pixels. Default is 1e-3, Ernould's criterion.
    step_scale
        Factor on the Gauss-Newton increment. Default is 1.0, plain
        Gauss-Newton; EMsoftOO's 1.25 to 1.5 accelerator is an
        unproven heuristic which is measured before it is ever
        adopted.
    interpolation
        Interpolation name, ``"bicubic"`` by default.

    Returns
    -------
    result
        Dictionary with the keys ``"h"`` (shape ``(8,)``),
        ``"residual"`` (the final ZNSSD ``CIC``), ``"num_iterations"``
        (int), ``"norm_dp"`` (the final corner norm in binned pixels)
        and ``"converged"`` (bool).

    Raises
    ------
    ValueError
        If *interpolation* is not in
        :data:`~kikuchipy.indexing._hrebsd._interpolation\
.SUPPORTED_INTERPOLATION`. Every OTHER failure is caught and
        reported through the NaN result contract below rather than
        raised.

    Notes
    -----
    A non-converged fit keeps its last iterate and reports
    ``converged=False``; it is never zeroed.  A failed pattern (a
    constant subregion, a non-finite ``CIC``, or any per-pattern
    exception) is reported with NaN in ``"h"`` and ``"residual"`` and
    ``converged=False``, following the spherical result contract.

    ``"residual"`` is the ZNSSD criterion OF THE RETURNED
    homography: it is evaluated once more after the loop exits, so
    that the two describe the same iterate (requirements D2.7 asks
    for the final CIC; corrected 2026-09-07 at the Stage A
    adversarial review, which measured the pre-update criterion 26
    per cent high on a point capped by ``max_iterations`` -- exactly
    the points a quality map is read on).
    """
    if interpolation not in SUPPORTED_INTERPOLATION:
        raise ValueError(
            f"interpolation must be one of {SUPPORTED_INTERPOLATION}, not "
            f"{interpolation!r}"
        )
    try:
        return _fit_pattern(
            state,
            target,
            h0=h0,
            upsample_factor=upsample_factor,
            max_iterations=max_iterations,
            min_step=min_step,
            step_scale=step_scale,
        )
    except Exception:
        # One bad pattern never kills the run, the spherical result
        # contract (a constant subregion, a non-finite criterion, a
        # singular increment): NaN properties and ``converged=False``
        return {
            "h": np.full(N_HOMOGRAPHY_PARAMETERS, np.nan),
            "residual": np.nan,
            "num_iterations": 0,
            "norm_dp": np.nan,
            "converged": False,
        }


def _fit_pattern(
    state: ReferenceState,
    target: np.ndarray,
    *,
    h0: np.ndarray | None,
    upsample_factor: int,
    max_iterations: int,
    min_step: float,
    step_scale: float,
) -> dict:
    """Return the IC-GN fit of one target, raising on failure.

    The loop itself, without the per-pattern failure contract of
    :func:`fit_pattern`, which wraps it.

    Parameters
    ----------
    state
        The reference's precomputed :class:`ReferenceState`.
    target
        The raw target pattern of the shape of the reference.
    h0
        Seed homography parameters of shape ``(8,)``, or ``None`` for
        the phase cross-correlation seed.
    upsample_factor
        Subpixel precision of the seed.
    max_iterations
        Iteration cap.
    min_step
        Convergence threshold on the corner norm, in binned pixels.
    step_scale
        Factor on the Gauss-Newton increment.

    Returns
    -------
    result
        The dictionary of :func:`fit_pattern`, whose ``"residual"``
        is the criterion re-evaluated at the RETURNED homography.

    Raises
    ------
    ValueError
        If *target* does not have the reference shape, or if the
        ZNSSD criterion is not finite.
    """
    target = np.asarray(target)
    if target.shape != state.shape:
        raise ValueError(
            f"the target of shape {target.shape} must have the reference shape "
            f"{state.shape}"
        )
    preprocessed = preprocess(target, transfer_function=state.transfer_function)
    row_start, row_stop, col_start, col_stop = state.bounds
    if h0 is None:
        h0 = initial_guess(
            state.reference_subregion,
            preprocessed[row_start:row_stop, col_start:col_stop],
            upsample_factor=upsample_factor,
        )
    matrix = shape_function(h0)

    # The ORIGINAL target is splined ONCE and re-warped by the
    # ACCUMULATED warp every iteration, the frozen deviation from
    # EMsoftOO's progressive warp-of-warp with per-iteration
    # re-splining (``mod_DIC.f90:684-747``): interpolation smoothing
    # never accumulates here
    coefficients = np.ascontiguousarray(
        spline_coefficients(
            preprocessed,
            interpolation=state.interpolation,
            dtype=state.coefficient_dtype,
        )
    )

    xi_x = state.xi_x
    xi_y = state.xi_y
    # The kernel samples the ARRAY frame, whose origin is the centre of
    # the upper left pixel, so the PC-centred coordinates are shifted
    # back by ``PC - 0.5``
    offset_x = state.pc_pixels[0] - 0.5
    offset_y = state.pc_pixels[1] - 0.5
    steepest_descent = state.steepest_descent
    # The matched partner of the D2.1 Hessian scale
    gradient_scale = 2.0 / state.reference_norm

    values = np.empty(xi_x.size, dtype=np.float64)
    weights = state.weights

    def criterion(current):
        """Return the ZNSSD residual vector at a shape function.

        The D2.6 non-finite-criterion case is caught INSIDE, by
        :func:`zero_mean_normalize`, which refuses a warped subregion
        whose centred 2-norm is zero or not finite (pruned from here
        2026-09-07, Stage A adversarial review: with two unit-norm
        vectors the criterion is bounded by four times the pixel
        count and a separate finiteness guard here was unreachable
        dead code).  :func:`fit_pattern` turns that raise into the NaN
        result contract.
        """
        scale = current[2, 0] * xi_x + current[2, 1] * xi_y + current[2, 2]
        warped_x = (current[0, 0] * xi_x + current[0, 1] * xi_y + current[0, 2]) / scale
        warped_y = (current[1, 0] * xi_x + current[1, 1] * xi_y + current[1, 2]) / scale
        evaluate(
            coefficients,
            np.ascontiguousarray(warped_x + offset_x),
            np.ascontiguousarray(warped_y + offset_y),
            out=values,
        )
        warped, _ = zero_mean_normalize(values)
        # The D4.3 window weights the residual in the REFERENCE frame,
        # after the warp, so that it never travels with the target
        difference = state.reference - warped
        return difference if weights is None else difference * weights

    converged = False
    n_iterations = 0
    residual = np.nan
    norm_dp = np.nan
    for _ in range(int(max_iterations)):
        n_iterations += 1
        residuals = criterion(matrix)
        residual = float(residuals @ residuals)
        gradient = gradient_scale * (steepest_descent.T @ residuals)
        step = cho_solve(state.cho_factor, -gradient)
        if step_scale != 1.0:
            step = step * step_scale
        norm_dp = corner_norm(step, state.corners)
        # The inverse-compositional update, on THIS side: the mutant
        # ``W(dp)**-1 . W`` shares the same fixed point and is invisible
        # to every accuracy band, so only the hand-built update oracle
        # separates them
        matrix = matrix @ np.linalg.inv(shape_function(step))
        matrix = matrix / matrix[2, 2]
        if norm_dp < min_step:
            converged = True
            break

    # The FINAL criterion of requirements D2.7, of the homography
    # actually returned rather than of the iterate before the last
    # composition (corrected 2026-09-07, Stage A adversarial review):
    # the two differ by 1.6e-05 relative at convergence and by 26 per
    # cent on a point capped by ``max_iterations``, which is precisely
    # where a quality map is read
    residuals = criterion(matrix)
    residual = float(residuals @ residuals)

    return {
        "h": homography_parameters(matrix),
        "residual": residual,
        "num_iterations": int(n_iterations),
        "norm_dp": float(norm_dp),
        "converged": bool(converged),
    }


def run_hrebsd_dic(
    patterns: np.ndarray,
    navigation_shape: tuple[int, int],
    detector,
    *,
    xmap=None,
    reference: str | tuple[int, int] | np.ndarray = "auto",
    grain_labels: np.ndarray | None = None,
    misorientation_threshold: float = 5.0,
    filter_cutoffs: tuple = (0.05, None),
    window: bool = False,
    border: float = 0.05,
    dead_band: tuple | None = None,
    interpolation: str = "bicubic",
    upsample_factor: int = 16,
    max_iterations: int = 50,
    min_step: float = 1e-3,
    step_scale: float = 1.0,
    navigation_mask: np.ndarray | None = None,
    chunksize: int | None = None,
    verbose: int = 1,
    step_sizes: tuple[float, float] = (1.0, 1.0),
    correct_pc_shift: bool = True,
    coefficient_dtype: np.dtype | type = np.float32,
) -> dict:
    """Run the engine over a whole map and return the properties.

    The orchestration entry point behind
    :meth:`kikuchipy.signals.EBSD.hrebsd_dic`.  Per-pattern arrays
    (patterns, seed shifts, per-point projection centres, reference
    identifiers) are paired with :func:`dask.array.blockwise` exactly
    as the spherical refinement path does, because
    :func:`dask.array.map_blocks` cannot index several arguments
    along one axis.  The engine may order patterns grain by grain so
    that a chunk correlates against one reference's precomputed
    state, and it MUST restore map order in the result; a run is
    deterministic for fixed inputs and chunking (requirements D16).

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)`` in map order.
    navigation_shape
        Map shape ``(ny, nx)``.
    detector
        :class:`~kikuchipy.detectors.EBSDDetector` with one
        projection centre or one per map point.
    xmap
        :class:`~orix.crystal_map.CrystalMap` of the map, consumed by
        the ``reference="auto"`` mode of Stage B only.
    reference
        See :func:`~kikuchipy.indexing._hrebsd._reference\
.resolve_reference`.
    grain_labels
        Optional user supplied grain map of *navigation_shape*.
    misorientation_threshold
        Grain boundary threshold in degrees, Stage B only.
    filter_cutoffs
        ``(high_pass, low_pass)`` band-pass cut-offs as fractions of
        the pattern width. Default is ``(0.05, None)``.
    window
        Whether to weight the ZNSSD residual by a Hann window over
        the subregion. Default is ``False``.
    border
        Border removed from every edge as a fraction of the pattern
        side. Default is 0.05.
    dead_band
        Optional ``(x0, x1, y0, y1)`` cross of dead camera pixels.
    interpolation
        Interpolation name. Default is ``"bicubic"``.
    upsample_factor
        Subpixel precision of the initial guess. Default is 16.
    max_iterations
        Iteration cap. Default is 50.
    min_step
        Convergence threshold in binned pixels. Default is 1e-3.
    step_scale
        Factor on the Gauss-Newton increment. Default is 1.0.
    navigation_mask
        Boolean mask of *navigation_shape* in kikuchipy polarity,
        where only patterns equal to ``False`` are fitted.
    chunksize
        Number of patterns per dask chunk, estimated when not given.
    verbose
        0 for no output, 1 for the information message, progress bar
        and timing.
    step_sizes
        Vertical and horizontal scan step sizes ``(dy, dx)`` **in the
        unit of** :attr:`~kikuchipy.detectors.EBSDDetector.px_size`
        (micrometres by the kikuchipy convention), used by the
        internal beam-scan projection centre model when the detector
        carries a single projection centre.
        :meth:`kikuchipy.signals.EBSD.hrebsd_dic` converts the
        navigation axes into that unit and refuses an axis whose unit
        it cannot read; a caller of this function does the conversion
        itself.
    correct_pc_shift
        Whether to remove the beam-scan phantom before converting a
        homography to a deformation gradient. Default is ``True``.
        The private switch of validation V6, never exposed in the
        public API.
    coefficient_dtype
        Bulk storage data type of the spline coefficients, 32-bit
        float by default, provisional per requirements D17.

    Returns
    -------
    properties
        Dictionary keyed by :data:`STAGE_A_PROP_NAMES`, every value
        an array whose first axis is the full map size in map order,
        with NaN or the not-indexed fill on masked and failed points.

    Raises
    ------
    NotImplementedError
        If *reference* is ``"auto"``, until Stage B.
    ValueError
        For any invalid argument, each naming the offending one.
    """
    if interpolation not in SUPPORTED_INTERPOLATION:
        raise ValueError(
            f"interpolation must be one of {SUPPORTED_INTERPOLATION}, not "
            f"{interpolation!r}"
        )
    navigation_shape = tuple(int(i) for i in navigation_shape)
    if len(navigation_shape) != 2:
        raise ValueError(
            f"navigation_shape {navigation_shape} must hold exactly two entries "
            "(ny, nx)"
        )
    map_size = int(navigation_shape[0]) * int(navigation_shape[1])
    if patterns.ndim != 3 or int(patterns.shape[0]) != map_size:
        raise ValueError(
            f"patterns of shape {patterns.shape} must be (n, nrows, ncols) with n "
            f"the map size {map_size} of navigation_shape {navigation_shape}"
        )
    signal_shape = (int(patterns.shape[1]), int(patterns.shape[2]))

    # The frozen order of ``spherical_indexing``: is-array, data type,
    # shape, all-``True``.  The data type check must precede the
    # all-``True`` one, since an integer mask of ones is all ``True``
    if navigation_mask is not None:
        if not isinstance(navigation_mask, np.ndarray):
            raise ValueError("The navigation mask must be a NumPy array")
        elif navigation_mask.dtype != np.bool_:
            raise ValueError("The navigation mask must be a boolean array")
        elif tuple(navigation_mask.shape) != navigation_shape:
            raise ValueError(
                f"The navigation mask shape {navigation_mask.shape} and the map's "
                f"navigation shape {navigation_shape} must be identical"
            )
        elif navigation_mask.all():
            raise ValueError(
                "The navigation mask must allow for correlation of at least one "
                "pattern (at least one value equal to `False`)"
            )

    grain_id, reference_index = resolve_reference(
        reference,
        grain_labels,
        navigation_shape,
        misorientation_threshold=misorientation_threshold,
        xmap=xmap,
        patterns=patterns,
    )

    keep = np.ones(map_size, dtype=bool)
    if navigation_mask is not None:
        keep = ~navigation_mask.ravel()
    keep &= reference_index >= 0
    fit_indices = np.flatnonzero(keep)
    unique_references = np.unique(reference_index[fit_indices])
    if unique_references.size == 0:
        raise ValueError(
            "no map point has a reference to correlate against; check reference, "
            "grain_labels and navigation_mask"
        )

    # Per-point projection centres, first class (requirements D6.1).
    # A single-PC detector is extrapolated internally with the
    # beam-scan model, anchored at the FIRST grain reference's scan
    # position so that point keeps the detector's own centre
    pc_pixels = per_point_pc_pixels(
        detector,
        navigation_shape,
        step_sizes,
        reference_index=int(unique_references[0]),
    )

    mask = subregion_mask(signal_shape, border=border, dead_band=dead_band)
    transfer_function = band_pass_transfer_function(signal_shape, filter_cutoffs)
    # Over the SUBREGION, as requirements D4.3 asks, not the whole
    # pattern; the engine applies it as a residual weight in the
    # reference frame (corrected 2026-09-07)
    window_array = None
    if window:
        window_array = hann_window(
            signal_shape, bounds=subregion_bounds(signal_shape, border=border)
        )

    reference_patterns = _gather(patterns, unique_references)
    states = []
    for position, index in enumerate(unique_references):
        states.append(
            ReferenceState(
                reference_patterns[position],
                mask,
                pc_pixels[int(index)],
                transfer_function=transfer_function,
                window=window_array,
                interpolation=interpolation,
                coefficient_dtype=coefficient_dtype,
            )
        )

    # Grain by grain, so that a chunk correlates against ONE reference's
    # precomputed state; the map order is restored below (D16)
    state_of_point = np.searchsorted(unique_references, reference_index[fit_indices])
    order = np.argsort(state_of_point, kind="stable")
    fit_indices = fit_indices[order]
    state_of_point = np.ascontiguousarray(state_of_point[order].astype(np.int64))

    n_fit = int(fit_indices.size)
    if chunksize is None:
        chunksize = estimate_chunksize(n_fit)
    chunksize = max(1, min(int(chunksize), n_fit))

    if verbose >= 1:
        print(get_info_message(n_fit, signal_shape, len(states), chunksize=chunksize))

    time_start = time.time()
    packed = _run_chunks(
        patterns,
        fit_indices,
        state_of_point,
        states,
        chunksize,
        progressbar=verbose >= 1,
        options={
            "upsample_factor": upsample_factor,
            "max_iterations": max_iterations,
            "min_step": min_step,
            "step_scale": step_scale,
        },
    )
    total_time = time.time() - time_start

    homography = np.full((map_size, HOMOGRAPHY_PROP_SIZE), np.nan, dtype=np.float64)
    fe = np.full((map_size, FE_PROP_SIZE), np.nan, dtype=np.float64)
    residual = np.full(map_size, np.nan, dtype=np.float64)
    num_iterations = np.zeros(map_size, dtype=np.int32)
    norm_dp = np.full(map_size, np.nan, dtype=np.float64)
    converged = np.zeros(map_size, dtype=bool)

    homography[fit_indices] = packed[:, :HOMOGRAPHY_PROP_SIZE]
    residual[fit_indices] = packed[:, _SLOT_RESIDUAL]
    num_iterations[fit_indices] = packed[:, _SLOT_ITERATIONS].astype(np.int32)
    norm_dp[fit_indices] = packed[:, _SLOT_NORM_DP]
    converged[fit_indices] = packed[:, _SLOT_CONVERGED] > 0.5

    # The D2.6 contract: a non-converged point KEEPS its last iterate in
    # ``homography`` and is never zeroed, while every DERIVED property
    # downstream is NaN.  The beam-scan phantom is removed analytically
    # BEFORE the conversion, and only on this path, which is why the
    # stored homography stays raw (D6.2, D15.6)
    for index in fit_indices:
        if not converged[index]:
            continue
        origin = pc_pixels[int(reference_index[index])]
        fe[index] = fe_from_homography(
            homography[index],
            origin,
            pc_pixels[int(index)],
            correct=correct_pc_shift,
        ).ravel()

    n_bad = int(n_fit - converged[fit_indices].sum())
    if n_bad:
        warnings.warn(
            f"{n_bad} of {n_fit} pattern(s) did not converge or failed, and carry "
            "their last iterate with `converged=False` and NaN in every derived "
            "property; they are never zeroed",
            UserWarning,
        )

    if verbose >= 1 and total_time > 0:
        print(f"  Correlation speed: {n_fit / total_time:.5f} patterns/s")

    return {
        "homography": homography,
        "Fe": fe,
        "residual": residual,
        "num_iterations": num_iterations,
        "norm_dp": norm_dp,
        "converged": converged,
        "grain_id": grain_id,
        "reference_index": reference_index,
    }


def _gather(patterns, indices: np.ndarray) -> np.ndarray:
    """Return the patterns at *indices* as one eager array.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)``, eager or lazy.
    indices
        Flat map indices to gather.

    Returns
    -------
    gathered
        Array of shape ``(indices.size, nrows, ncols)``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if isinstance(patterns, da.Array):
        return np.asarray(patterns[indices].compute())
    return np.asarray(patterns)[indices]


def _fit_chunk(
    patterns_block: np.ndarray,
    state_index_block: np.ndarray,
    states: tuple = (),
    options: dict | None = None,
) -> np.ndarray:
    """Return the packed IC-GN results of one chunk of patterns.

    Parameters
    ----------
    patterns_block
        ``(n, nrows, ncols)`` block of patterns, grain ordered.
    state_index_block
        ``(n,)`` block of indices into *states*, paired with
        *patterns_block* along the same named dask index.
    states
        The per-reference precomputed states.
    options
        Keyword arguments forwarded to :func:`fit_pattern`.

    Returns
    -------
    results
        ``(n, 12)`` 64-bit float array of packed rows.
    """
    options = {} if options is None else options
    n_patterns = int(patterns_block.shape[0])
    results = np.empty((n_patterns, _ROW_WIDTH), dtype=np.float64)
    for i in range(n_patterns):
        state = states[int(state_index_block[i])]
        result = fit_pattern(state, patterns_block[i], **options)
        results[i, :HOMOGRAPHY_PROP_SIZE] = result["h"]
        results[i, _SLOT_RESIDUAL] = result["residual"]
        results[i, _SLOT_ITERATIONS] = result["num_iterations"]
        results[i, _SLOT_NORM_DP] = result["norm_dp"]
        results[i, _SLOT_CONVERGED] = 1.0 if result["converged"] else 0.0
    return results


def _run_chunks(
    patterns,
    fit_indices: np.ndarray,
    state_of_point: np.ndarray,
    states: list,
    chunksize: int,
    *,
    progressbar: bool,
    options: dict,
) -> np.ndarray:
    """Return the packed results of every fitted point, in fit order.

    :func:`dask.array.blockwise` and not
    :func:`dask.array.map_blocks`, which indexes every argument from
    its trailing axis and would hand each block the whole of the state
    index array, so every chunk but the first would correlate against
    the wrong reference -- silently, and with the right number of rows.
    Here the pattern axis is one named index of both arrays, so a block
    mis-alignment is impossible to express.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)``, eager or lazy.
    fit_indices
        Flat map indices to correlate, in grain order.
    state_of_point
        Index into *states* of every entry of *fit_indices*.
    states
        The per-reference precomputed states.
    chunksize
        Number of patterns per chunk.
    progressbar
        Whether to show a progress bar.
    options
        Keyword arguments forwarded to :func:`fit_pattern`.

    Returns
    -------
    packed
        ``(fit_indices.size, 12)`` 64-bit float array.
    """
    if isinstance(patterns, da.Array):
        patterns_da = patterns
    else:
        patterns_da = da.from_array(np.asarray(patterns), chunks=(chunksize, -1, -1))
    ordered = patterns_da[fit_indices].rechunk((chunksize, -1, -1))
    state_da = da.from_array(state_of_point, chunks=(chunksize,))
    results = da.blockwise(
        _fit_chunk,
        "ij",
        ordered,
        "ikl",
        state_da,
        "i",
        states=tuple(states),
        options=options,
        new_axes={"j": _ROW_WIDTH},
        dtype=np.float64,
        concatenate=True,
    )
    # The threaded scheduler, as the spherical path pins it: the states
    # are read-only and never cross a process boundary
    if progressbar:
        with ProgressBar():
            packed = results.compute(scheduler="threads")
    else:
        packed = results.compute(scheduler="threads")
    return np.asarray(packed, dtype=np.float64)


def n_workers() -> int:
    """Return the number of worker threads the chunks are spread over.

    Returns
    -------
    workers
        The active dask configuration's ``num_workers`` when it is set,
        so that an outer :func:`dask.config.set` is honoured as it is
        by the scheduler itself, and the processor count otherwise.
    """
    workers = dask.config.get("num_workers", None)
    if workers is None:
        workers = CPU_COUNT
    return max(1, int(workers))


def estimate_chunksize(n_patterns: int) -> int:
    """Return the number of patterns per chunk when none is given.

    A few chunks per worker, so that the load balances without making
    the graph larger than the work it describes.

    Parameters
    ----------
    n_patterns
        Number of patterns to correlate.

    Returns
    -------
    chunksize
        At least one.
    """
    n_patterns = max(1, int(n_patterns))
    return max(1, math.ceil(n_patterns / (4 * n_workers())))


def get_info_message(
    n_patterns: int,
    signal_shape: tuple[int, int],
    n_grains: int,
    chunksize: int | None = None,
) -> str:
    """Return the information message printed at ``verbose >= 1``.

    Follows the
    :meth:`kikuchipy.indexing.SphericalIndexer.get_info_message`
    precedent, and states the resident memory of the per-grain
    precompute, which requirements D16 makes part of the message.

    Parameters
    ----------
    n_patterns
        Number of patterns to be fitted.
    signal_shape
        Pattern shape ``(nrows, ncols)``.
    n_grains
        Number of grain references.
    chunksize
        Number of patterns per chunk, or ``None`` when estimated.

    Returns
    -------
    message
        A multi-line message naming the number of patterns, the
        subregion, the number of references, the chunking and the
        resident memory per reference in MB.
    """
    n_patterns = int(n_patterns)
    nrows, ncols = (int(i) for i in signal_shape)
    if chunksize is None:
        chunksize = estimate_chunksize(n_patterns)
    chunksize = max(1, int(chunksize))
    n_chunks = math.ceil(n_patterns / chunksize) if n_patterns else 0
    # The model, not a measurement, exactly as the spherical
    # information message states its own: the eight 64-bit
    # steepest-descent columns, the 32-bit coefficient plane and the
    # remaining subregion sized 64-bit planes, at the pattern size,
    # which bounds the subregion from above
    bytes_per_reference = (
        nrows * ncols * (_N_STEEPEST_DESCENT_COLUMNS * 8 + 4 + _RESIDENT_F64_PLANES * 8)
    )
    return "\n".join(
        [
            "HREBSD-DIC information:",
            f"  Correlating {n_patterns} pattern(s) of shape ({nrows}, {ncols}) "
            f"against {int(n_grains)} reference(s)",
            f"  Chunking: {n_chunks} chunk(s) of up to {chunksize} pattern(s)",
            f"  Estimated memory per reference: {bytes_per_reference / 1024**2:.1f} MB",
        ]
    )
