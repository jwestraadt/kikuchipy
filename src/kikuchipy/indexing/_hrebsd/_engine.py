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

from kikuchipy.indexing._hrebsd import _batched as _batched
from kikuchipy.indexing._hrebsd import _fourier_mellin as _fourier_mellin
from kikuchipy.indexing._hrebsd import _gpu as _gpu
from kikuchipy.indexing._hrebsd._geometry import fe_from_homography, per_point_pc_pixels

# The availability gate of requirements D21.2, imported INTO this
# namespace on purpose: ``run_hrebsd_dic`` calls it under
# ``backend="gpu"`` and the tests patch it here (the Phase 12 seam)
from kikuchipy.indexing._hrebsd._gpu import (
    _verify_gpu_or_raise as _verify_gpu_or_raise,
)
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

# ---------------- Stage D, neighbour-seeded propagation ------------- #

# The FROZEN neighbour offset order of requirements D20.2, ``(dy, dx)``
# in row-major reading order of the eight-neighbourhood with the centre
# skipped.  It breaks a residual tie between two equally cheap
# converged neighbours, and it is frozen rather than incidental because
# the tie rule is invisible in a run's result: two neighbours whose
# residuals are bitwise equal have bitwise equal homographies too
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# The iteration budget of PASS 1 of the requirements D20.2 cascade,
# which runs at ``min(max_iterations, PASS1_CAP)``.  An INTERNAL
# constant, never a public knob (D20.1), and MEASURED-THEN-PINNED: the
# measurement is the far-field convergence distribution on the
# Si-indent data, and the cap must sit just above the iteration count
# the EASIEST points need so that an anchor forms wherever easy points
# exist.
#
# PINNED at 50 (measurement-close 2026-09-09, validation.md V8 entry
# 83): on a clean far-field patch of the real Si-indent data every
# point converges by p95 = 10 iterations (median 9), so the shipped cap
# sits 5.0x above that p95 and catches every easy point in pass 1 with
# wide margin.  The measurement KEPT 50; the earlier rim-in-isolation
# cap sweep that appeared to want a higher cap is recorded there as a
# patch artefact (an all-deformed patch with no easy points), not a
# reason to raise the global default.  A cap never truncates a FINAL
# answer: every point pass 1 leaves unconverged is re-fitted at the
# full budget by a cascade round or by the rescue pass.
#
# ADMISSIBLE WINDOW of the frozen ``seed_round`` oracle, recorded
# 2026-09-09 in validation.md V8 so that a re-pin knows what it may not
# cross: ``PASS1_CAP`` in [10, 112].  Below 10 the 1.6 degree ramp
# point of ``test_hrebsd_seeding.py`` drops out of pass 1 and the whole
# ramp shifts by one round; at 113 and above the isolated rescue point
# converges IN pass 1 and the rescue pass stops being exercised
PASS1_CAP: int = 50

# The ``seed_round`` property of requirements D20.5, emitted by a
# SEEDED run only: 0 for a point pass 1 converged (the reference point
# included), ``r >= 1`` for one converted by cascade round ``r``, -2 for
# one converged only in the rescue pass, and -1 for a point which never
# converged or is masked out
SEED_ROUND_PROP_NAME: str = "seed_round"
SEED_ROUND_PASS1: int = 0
SEED_ROUND_RESCUE: int = -2
SEED_ROUND_NONE: int = -1

# The two properties of requirements D22.9, emitted by a run with
# ``fourier_mellin`` other than ``"off"`` only (defined beside the
# Fourier-Mellin stage, re-exported here for ``EBSD.hrebsd_dic``)
FOURIER_MELLIN_PROP_NAMES: tuple[str, ...] = _fourier_mellin.FOURIER_MELLIN_PROP_NAMES

# ------------------ Stage E, the optional GPU backend --------------- #

# The accepted values of ``backend`` (requirements D21.1),
# case-sensitive; there is no "auto" and no silent fallback
SUPPORTED_BACKENDS: tuple[str, ...] = ("cpu", "gpu")

# The frozen message of requirements D21.12, raised under
# ``backend="gpu"`` with ``seed_from_neighbors=True`` before the gate
# and before any work; no roadmap stage letter in public text
SEED_FROM_NEIGHBORS_GPU_MESSAGE: str = (
    "seed_from_neighbors=True is not supported with backend='gpu'; use "
    "backend='cpu' for neighbour-seeded propagation, or "
    "seed_from_neighbors=False with fourier_mellin='auto' for large "
    "rotations about the detector normal"
)

# ------- Stage F, the opt-in Fourier-Mellin rotation initial guess ------- #

# The packed per-point row of a run with ``fourier_mellin`` other than
# ``"off"`` (requirements D22.9): the twelve slots of :data:`_ROW_WIDTH`
# plus the Fourier-Mellin angle (deg, NaN where none was estimated) and
# the applied flag (1.0 where the slot's seed row was the FM row).  An
# ``"off"`` run keeps the twelve-wide row
_SLOT_FM_ANGLE: int = _ROW_WIDTH
_SLOT_FM_APPLIED: int = _ROW_WIDTH + 1
_ROW_WIDTH_FM: int = _ROW_WIDTH + 2

# The ``row_slots`` of the device runner (requirements D21.9.5) on an
# ``"off"`` run, and on a Fourier-Mellin run (D22.9)
_ROW_SLOTS: dict = {
    "width": _ROW_WIDTH,
    "residual": _SLOT_RESIDUAL,
    "iterations": _SLOT_ITERATIONS,
    "norm_dp": _SLOT_NORM_DP,
    "converged": _SLOT_CONVERGED,
}
# The host bytes per pattern pixel a reference with a routed point adds
# on the CPU route (requirements D22.10, the information message's
# memory note of D22.18): the complex128 reference spectrum of its
# numpy seed state, and the 64-bit planes of its box resident (two
# coordinate planes) and of its numpy subregion resident (two
# coordinate planes, the reference, two gradient columns and the
# window weights) plus the 32-bit pixel indices
_FM_HOST_COMPLEX_BYTES: int = 16
_FM_HOST_PLANE_BYTES: int = 2 * 8 + 6 * 8 + 4

_ROW_SLOTS_FM: dict = {
    **_ROW_SLOTS,
    "width": _ROW_WIDTH_FM,
    "fm_angle": _SLOT_FM_ANGLE,
    "fm_applied": _SLOT_FM_APPLIED,
}


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


def choose_seed_indices(
    converged: np.ndarray,
    residual: np.ndarray,
    grain_id: np.ndarray,
    navigation_shape: tuple[int, int],
    *,
    navigation_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the flat index of the neighbour which seeds each point.

    The frozen seed choice of requirements D20.2, as the pure function
    of arrays it is: the seed of a point is the converged, unmasked,
    SAME-grain neighbour of its eight-neighbourhood with the LOWEST
    residual, ties broken by the frozen :data:`NEIGHBOR_OFFSETS`
    order.  Nothing here knows which round it is called from, so the
    cascade's determinism (D20.3) follows from feeding it results of
    strictly earlier rounds only.

    Parameters
    ----------
    converged
        Boolean convergence flag of every map point, flat in map
        order.
    residual
        Final ZNSSD criterion of every map point, flat in map order.
        A NaN entry is never chosen, so a point which failed outright
        cannot seed anything.
    grain_id
        Grain label of every map point, flat in map order. A seed
        never crosses a boundary between two different labels
        (D20.4).
    navigation_shape
        Map shape ``(ny, nx)``, which turns a flat index into a row
        and a column. The map edge is never wrapped around.
    navigation_mask
        Boolean mask of *navigation_shape* in kikuchipy polarity,
        where ``True`` is masked out. A masked point neither seeds
        nor is seeded (D20.4). If not given, nothing is masked.

    Returns
    -------
    seeds
        Flat index of the chosen seed of every map point, ``-1``
        where no eligible neighbour exists.
    """
    nrows, ncols = (int(i) for i in navigation_shape)
    size = nrows * ncols
    converged = np.asarray(converged, dtype=bool).reshape(size)
    residual = np.asarray(residual, dtype=np.float64).reshape(size)
    grain_id = np.asarray(grain_id).reshape(size)
    if navigation_mask is None:
        masked = np.zeros(size, dtype=bool)
    else:
        masked = np.asarray(navigation_mask, dtype=bool).reshape(size)

    rows, cols = np.divmod(np.arange(size, dtype=np.int64), ncols)
    # A seed must have converged and must not be masked out; a
    # candidate must not be masked out either
    can_seed = converged & ~masked
    can_be_seeded = ~masked

    seeds = np.full(size, -1, dtype=np.int64)
    best = np.full(size, np.inf, dtype=np.float64)
    for row_offset, col_offset in NEIGHBOR_OFFSETS:
        neighbor_rows = rows + row_offset
        neighbor_cols = cols + col_offset
        inside = (
            (neighbor_rows >= 0)
            & (neighbor_rows < nrows)
            & (neighbor_cols >= 0)
            & (neighbor_cols < ncols)
        )
        # ``np.where`` keeps the gather in range; ``inside`` throws the
        # out-of-map entries away again below
        neighbors = np.where(inside, neighbor_rows * ncols + neighbor_cols, 0)
        eligible = (
            inside
            & can_be_seeded
            & can_seed[neighbors]
            & (grain_id[neighbors] == grain_id)
        )
        # STRICTLY lower, so an exact tie keeps the neighbour the
        # frozen offset order reached first
        better = eligible & (residual[neighbors] < best)
        best = np.where(better, residual[neighbors], best)
        seeds = np.where(better, neighbors, seeds)
    return seeds


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
    seed_from_neighbors: bool = False,
    fourier_mellin: str = "off",
    navigation_mask: np.ndarray | None = None,
    backend: str = "cpu",
    chunksize: int | None = None,
    verbose: int = 1,
    step_sizes: tuple[float, float] = (1.0, 1.0),
    correct_pc_shift: bool = True,
    coefficient_dtype: np.dtype | type = np.float32,
    device_precision: str = "mixed",
    seed_precision: str = "complex128",
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
        the ``reference="auto"`` mode only, and there only when no
        *grain_labels* are given.
    reference
        See :func:`~kikuchipy.indexing._hrebsd._reference\
.resolve_reference`.
    grain_labels
        Optional user supplied grain map of *navigation_shape*.
    misorientation_threshold
        Grain boundary threshold in degrees, read by the
        ``reference="auto"`` segmentation only. Default is 5.0.
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
    seed_from_neighbors
        Whether to seed every fit which the per-point phase
        cross-correlation misses from an already converged same-grain
        neighbour's homography, the three-phase cascade of
        requirements D20.2. Default is ``False``, which is the
        BITWISE-unchanged behaviour of every release before Stage D
        and emits exactly the pre-Stage-D property set (D20.1, D20.5).
        With ``True`` the returned dictionary carries one more entry,
        ``"seed_round"``.
    fourier_mellin
        ``"off"`` (default), the bitwise-unchanged translation-only
        seed of requirements D5, ``"auto"`` or ``"always"``, the opt-in
        Fourier-Mellin rotation seed of requirements D22. ``"auto"``
        routes the points whose twist about the detector normal from
        their grain reference in *xmap* is at least ``FM_GATE_DEG``
        (or cannot be computed) through the seed, which keeps the
        rotation row only if its criterion beats the translation
        row's (D22.5, D22.7), and needs *xmap*; ``"always"`` evaluates
        the seed on every fitted point. Both then retry once, from the
        forced rotation row, every point whose translation-seeded fit
        did not converge (D22.8). Never combined with
        *seed_from_neighbors* (D22.11).
    navigation_mask
        Boolean mask of *navigation_shape* in kikuchipy polarity,
        where only patterns equal to ``False`` are fitted.
    backend
        ``"cpu"`` (default), the bitwise-unchanged path, or ``"gpu"``,
        the optional CuPy device path of requirements D21.
    chunksize
        Number of patterns per dask chunk, estimated when not given.
        Under ``backend="gpu"`` it is the device batch size.
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
    device_precision
        ``"mixed"`` (default) or ``"float64"``, the device arithmetic
        of requirements D21.4. Read under ``backend="gpu"`` only; an
        engine-only knob the public method never passes.
    seed_precision
        ``"complex128"`` (default) or ``"complex64"``, the precision
        of the device phase cross-correlation seed of requirements
        D21.5. Read under ``backend="gpu"`` only; engine-only.

    Returns
    -------
    properties
        Dictionary keyed by :data:`STAGE_A_PROP_NAMES`, every value
        an array whose first axis is the full map size in map order,
        with NaN or the not-indexed fill on masked and failed points.
        A run with *seed_from_neighbors* carries the one further
        entry :data:`SEED_ROUND_PROP_NAME` of requirements D20.5, an
        ``int32`` array recording how each point was reached:
        :data:`SEED_ROUND_PASS1` (``0``) for a point pass 1 converged,
        the reference included; a positive round number ``r >= 1`` for
        one a cascade round converted; :data:`SEED_ROUND_RESCUE`
        (``-2``) for one converged only in the rescue pass; and
        :data:`SEED_ROUND_NONE` (``-1``) for a point which never
        converged or is masked out. A run with *fourier_mellin* other
        than ``"off"`` carries the two entries of
        :data:`FOURIER_MELLIN_PROP_NAMES` (D22.9) instead:
        ``"fourier_mellin_seed"``, ``int32``, 0 where the stored fit
        was seeded by the translation row, 1 by a rotation row in the
        first pass, 2 by one in the retry pass and -1 on a masked or
        not fitted point; and ``"fourier_mellin_angle"``, ``float64``,
        the most recent rotation estimate in degrees, NaN where none
        was made.

    Raises
    ------
    ValueError
        For any invalid argument, each naming the offending one, an
        unsupported *backend* included (requirements D21.1); for an
        unsupported *fourier_mellin*, for *fourier_mellin* other than
        ``"off"`` with *seed_from_neighbors*, and for
        ``fourier_mellin="auto"`` without *xmap* (D22.1, D22.11).
    NotImplementedError
        With ``backend="gpu"`` and ``seed_from_neighbors=True``
        (requirements D21.12), before the gate and any work.
    ImportError, RuntimeError
        With ``backend="gpu"`` when the availability gate of
        requirements D21.2 fails; there is no CPU fallback.
    MemoryError
        With ``backend="gpu"`` when the device runs out of memory even
        at a device batch size of one (requirements D21.10.4).

    Notes
    -----
    With *seed_from_neighbors* the three phases of requirements D20.2
    run in turn: pass 1 is the ordinary per-point phase
    cross-correlation seeded fit at
    ``min(max_iterations, PASS1_CAP)``; cascade round ``r`` then
    fits, at the full budget, every not-yet-converged unmasked point
    with a converged same-grain eight-neighbour from rounds strictly
    earlier than ``r``, seeded by that neighbour's RAW homography
    through :func:`choose_seed_indices`; and the rescue pass gives
    every point still unconverged one full-budget fit from its best
    available seed, its own pass-1 last iterate or the phase
    cross-correlation seed, in that order.  Every seed is a pure
    function of completed rounds, so the run is deterministic
    whatever the chunking and whatever order the fits of one round
    happen to run in (D20.3).

    On a point a cascade round or the rescue pass re-fitted,
    ``num_iterations`` reports that final fit's OWN iteration count,
    the fit which produced the stored homography, not the sum across
    the phases: the capped pass-1 iterations spent on the point before
    it are not added in.
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

    # Requirements D21.1: the backend checks sit HERE, after every
    # existing argument check and before reference resolution, in the
    # frozen order: the backend string; under "gpu" only the two
    # precisions, the coefficient data type and the batch size; the
    # D21.12 raise; the D21.2 gate.  So a missing GPU costs nothing.
    # The default "cpu" path passes straight through, bitwise
    # unchanged and never touching the gate or cupy
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend!r} not in the list of supported backends "
            f"{list(SUPPORTED_BACKENDS)}"
        )
    # Requirements D22.1: the Fourier-Mellin checks sit here, after the
    # backend string check and before every "gpu"-only check.  The
    # default "off" passes straight through, bitwise unchanged: (1) the
    # value, (2) the D22.11 combination, (3) "auto" without a crystal map
    _check_fourier_mellin(fourier_mellin, seed_from_neighbors, xmap)
    use_fm = fourier_mellin != "off"
    use_gpu = backend == "gpu"
    if use_gpu:
        _check_gpu_arguments(
            device_precision, seed_precision, coefficient_dtype, chunksize
        )
        if seed_from_neighbors:
            raise NotImplementedError(SEED_FROM_NEIGHBORS_GPU_MESSAGE)
        # Looked up in this module's namespace at call time, where the
        # tests patch it (the Phase 12 seam)
        _verify_gpu_or_raise()

    # The mask reaches the reference resolution and not only the fit
    # list: a masked-out point is one the caller does not trust, and
    # ``reference="auto"`` must not pick it as the origin every
    # measurement in its grain is made against (2026-09-08, Stage B
    # adversarial review).  It stays out of the fit list too, below
    grain_id, reference_index = resolve_reference(
        reference,
        grain_labels,
        navigation_shape,
        misorientation_threshold=misorientation_threshold,
        xmap=xmap,
        patterns=patterns,
        navigation_mask=navigation_mask,
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
    # precomputed state; the map order is restored below (D16).  The
    # whole-map identifier is kept as well, because the Stage D cascade
    # fits an arbitrary subset of the map per round and has to order
    # each one the same way
    state_of_map = np.zeros(map_size, dtype=np.int64)
    state_of_map[fit_indices] = np.searchsorted(
        unique_references, reference_index[fit_indices]
    )
    state_of_point = state_of_map[fit_indices]
    order = np.argsort(state_of_point, kind="stable")
    fit_indices = fit_indices[order]
    state_of_point = np.ascontiguousarray(state_of_point[order].astype(np.int64))

    n_fit = int(fit_indices.size)

    # Requirements D22.7: the pre-fit routes of a Fourier-Mellin run,
    # on the host, in fit order, before any fit; ``None`` on "off"
    fm_route = None
    if use_fm:
        fm_route = _fourier_mellin_routes(
            fourier_mellin,
            xmap,
            detector,
            fit_indices,
            reference_index,
            navigation_shape=navigation_shape,
        )
    # D22.18: the device batch choice consults the FM terms only when a
    # route flag of the run is nonzero
    fm_routed = fm_route is not None and bool(fm_route.any())

    if use_gpu:
        # Requirements D21.10.1: ``chunksize`` IS the device batch size,
        # never clamped to the number of fitted points, and the default
        # is the VRAM-model chooser on the queried free VRAM (D21.10.3)
        free_bytes = None
        if chunksize is None or verbose >= 1:
            free_bytes = _gpu._free_device_bytes("cupy")
        if chunksize is None:
            if fm_routed:
                chunksize = _gpu._default_batch_size(
                    free_bytes,
                    signal_shape[0] * signal_shape[1],
                    device_precision,
                    seed_precision,
                    fourier_mellin=True,
                )
            else:
                chunksize = _gpu._default_batch_size(
                    free_bytes,
                    signal_shape[0] * signal_shape[1],
                    device_precision,
                    seed_precision,
                )
        chunksize = int(chunksize)
        if verbose >= 1:
            # The device batches are grain pure (D21.9.3): one grain's
            # points never share a batch with another's, so the count
            # is the sum over grains of ceil(n_g / B)
            message = get_info_message(
                n_fit,
                signal_shape,
                len(states),
                chunksize=chunksize,
                n_chunks=len(_gpu._batch_chunks(state_of_point, chunksize)),
                fourier_mellin=use_fm,
            )
            # D22.18: the device block states the model B was chosen
            # from, the Fourier-Mellin one when a route flag is nonzero
            # (2026-10-08 implementation review, RC-F-CONV-2)
            device_block = _gpu._info_lines(
                chunksize,
                signal_shape,
                device_precision,
                seed_precision,
                free_bytes,
                fourier_mellin=fm_routed,
            )
            print("\n".join([message, *device_block]))
    else:
        if chunksize is None:
            chunksize = estimate_chunksize(n_fit)
        chunksize = max(1, min(int(chunksize), n_fit))

        if verbose >= 1:
            print(
                get_info_message(
                    n_fit,
                    signal_shape,
                    len(states),
                    chunksize=chunksize,
                    fourier_mellin=use_fm,
                )
            )

    homography = np.full((map_size, HOMOGRAPHY_PROP_SIZE), np.nan, dtype=np.float64)
    fe = np.full((map_size, FE_PROP_SIZE), np.nan, dtype=np.float64)
    residual = np.full(map_size, np.nan, dtype=np.float64)
    num_iterations = np.zeros(map_size, dtype=np.int32)
    norm_dp = np.full(map_size, np.nan, dtype=np.float64)
    converged = np.zeros(map_size, dtype=bool)

    fit_options = {
        "upsample_factor": upsample_factor,
        "max_iterations": max_iterations,
        "min_step": min_step,
        "step_scale": step_scale,
    }

    def store(indices: np.ndarray, packed: np.ndarray) -> None:
        """Write one phase's packed rows into the map arrays."""
        homography[indices] = packed[:, :HOMOGRAPHY_PROP_SIZE]
        residual[indices] = packed[:, _SLOT_RESIDUAL]
        num_iterations[indices] = packed[:, _SLOT_ITERATIONS].astype(np.int32)
        norm_dp[indices] = packed[:, _SLOT_NORM_DP]
        converged[indices] = packed[:, _SLOT_CONVERGED] > 0.5

    def fit_phase(
        indices: np.ndarray, budget: int, seeds: np.ndarray | None = None
    ) -> None:
        """Fit one Stage D phase and store it, grain ordered.

        *indices* are flat map indices and *seeds* their per-point
        seed homographies in the SAME order, so the pairing survives
        the grain ordering and the chunking alike.
        """
        phase_order = np.argsort(state_of_map[indices], kind="stable")
        ordered_indices = indices[phase_order]
        options = dict(fit_options)
        options["max_iterations"] = int(budget)
        store(
            ordered_indices,
            _run_chunks(
                patterns,
                ordered_indices,
                np.ascontiguousarray(state_of_map[ordered_indices]),
                states,
                max(1, min(int(chunksize), int(ordered_indices.size))),
                progressbar=verbose >= 1,
                options=options,
                h0=None if seeds is None else np.asarray(seeds)[phase_order],
            ),
        )

    time_start = time.time()
    fm_seed = None
    fm_angle = None
    if seed_from_neighbors:
        # PASS 1 of requirements D20.2: the ordinary phase
        # cross-correlation seeded fit of every point, at the capped
        # budget.  The cap never truncates a final answer -- the
        # cascade rounds and the rescue pass below re-fit every point
        # it leaves unconverged at the FULL budget
        fit_phase(fit_indices, min(int(max_iterations), PASS1_CAP))
    elif use_fm:
        # Requirements D22.8 to D22.10: the first pass with the routes,
        # then one retry pass, both runner changes only
        fm_seed = np.full(map_size, _fourier_mellin.FOURIER_MELLIN_SEED_NONE, np.int32)
        fm_angle = np.full(map_size, np.nan, dtype=np.float64)
        fm_states = [None] * len(states)
        fm_built = [False] * len(states)
        # The device runner writes its final batch size here, per run
        # and never in module state (D22.8; 2026-10-08 implementation
        # review, RF-F1)
        fm_run_state: dict = {}

        def cpu_fm_states(positions) -> tuple:
            """Return the numpy ``SeedState`` of every reference, each
            carrying its ``FourierMellinState``, built ONCE per
            reference in *positions* on the host before the graph
            (D22.10), ``None`` elsewhere and where the build refused
            the reference (D22.6)."""
            for position in np.unique(np.asarray(positions, dtype=np.int64)):
                position = int(position)
                if not fm_built[position]:
                    fm_built[position] = True
                    fm_states[position] = _cpu_fourier_mellin_state(
                        states[position], upsample_factor
                    )
            return tuple(fm_states)

        def fm_pass(indices, state_index, route, batch_size):
            """Run one Fourier-Mellin pass over *indices* (fit order)
            with the per-point *route* and return its packed rows."""
            if use_gpu:
                return _gpu._run_chunks_gpu(
                    patterns,
                    indices,
                    state_index,
                    states,
                    batch_size,
                    progressbar=verbose >= 1,
                    options=fit_options,
                    device_precision=device_precision,
                    seed_precision=seed_precision,
                    row_slots=dict(_ROW_SLOTS_FM),
                    seed_extras={_fourier_mellin.FOURIER_MELLIN_ROUTE_KEY: route},
                    run_state=fm_run_state,
                )
            return _run_chunks(
                patterns,
                indices,
                state_index,
                states,
                max(1, min(int(batch_size), int(indices.size))),
                progressbar=verbose >= 1,
                options=fit_options,
                fm_route=route,
                fm_states=cpu_fm_states(state_index[route != 0]),
            )

        # The first pass: every fitted point, route 1 where routed
        packed = fm_pass(fit_indices, state_of_point, fm_route, chunksize)
        # The retry runs at the first pass's FINAL device batch size
        # (D22.8), which the out-of-memory loop may have halved
        retry_batch_size = int(fm_run_state.get("batch_size", chunksize))
        store(fit_indices, packed)
        applied = packed[:, _SLOT_FM_APPLIED] > 0.5
        fm_seed[fit_indices] = np.where(
            applied,
            _fourier_mellin.FOURIER_MELLIN_SEED_FIRST_PASS,
            _fourier_mellin.FOURIER_MELLIN_SEED_TRANSLATION,
        )
        fm_angle[fit_indices] = packed[:, _SLOT_FM_ANGLE]

        # The retry subset (D22.8): fitted, unmasked, not converged and
        # seeded by the translation row, in a grain with an FM state; a
        # NaN-``h`` first-pass point included.  On the CPU a grain's
        # state is built here if the first pass did not need it; the
        # device builds its own lazily and a grain it refuses leaves
        # its forced slots unapplied, so unfitted
        candidates = ~converged[fit_indices] & (
            fm_seed[fit_indices] == _fourier_mellin.FOURIER_MELLIN_SEED_TRANSLATION
        )
        if not use_gpu and candidates.any():
            cpu_fm_states(state_of_point[candidates])
            has_state = np.array([s is not None for s in fm_states], dtype=bool)
            candidates &= has_state[state_of_point]
        if candidates.any():
            retry_indices = np.ascontiguousarray(fit_indices[candidates])
            retry_states = np.ascontiguousarray(state_of_point[candidates])
            retry_route = np.full(
                retry_indices.size, _fourier_mellin.ROUTE_FORCED, dtype=np.int8
            )
            retried = fm_pass(
                retry_indices, retry_states, retry_route, retry_batch_size
            )
            if verbose >= 1:
                # Only an applied forced slot is fitted (D22.5); the count
                # is the same on both backends (2026-10-08 implementation
                # review, RC-F-CONV-3)
                n_refitted = int((retried[:, _SLOT_FM_APPLIED] > 0.5).sum())
                print(
                    f"  Fourier-Mellin retry: {n_refitted} of {retry_indices.size} "
                    "pattern(s) re-fitted from the rotation seed"
                )
            # The most recent estimate is the retry's (D22.9)
            fm_angle[retry_indices] = retried[:, _SLOT_FM_ANGLE]
            # The retry result replaces the first one wholly if and only
            # if it converged; otherwise the first result stands bitwise
            replace = retried[:, _SLOT_CONVERGED] > 0.5
            store(retry_indices[replace], retried[replace])
            fm_seed[retry_indices[replace]] = np.where(
                retried[replace, _SLOT_FM_APPLIED] > 0.5,
                _fourier_mellin.FOURIER_MELLIN_SEED_RETRY,
                _fourier_mellin.FOURIER_MELLIN_SEED_TRANSLATION,
            )
    elif use_gpu:
        # Requirements D21.9: the dispatch point is ``_run_chunks``;
        # the device runner reproduces its contract, so everything
        # upstream and downstream is shared code.  Reached through the
        # module object at call time, where the tests spy on it
        packed = _gpu._run_chunks_gpu(
            patterns,
            fit_indices,
            state_of_point,
            states,
            chunksize,
            progressbar=verbose >= 1,
            options=fit_options,
            device_precision=device_precision,
            seed_precision=seed_precision,
            row_slots={
                "width": _ROW_WIDTH,
                "residual": _SLOT_RESIDUAL,
                "iterations": _SLOT_ITERATIONS,
                "norm_dp": _SLOT_NORM_DP,
                "converged": _SLOT_CONVERGED,
            },
        )
        store(fit_indices, packed)
    else:
        packed = _run_chunks(
            patterns,
            fit_indices,
            state_of_point,
            states,
            chunksize,
            progressbar=verbose >= 1,
            options=fit_options,
        )
        store(fit_indices, packed)

    seed_round = None
    if seed_from_neighbors:
        seed_round = np.full(map_size, SEED_ROUND_NONE, dtype=np.int32)
        seed_round[converged] = SEED_ROUND_PASS1
        fittable = np.zeros(map_size, dtype=bool)
        fittable[fit_indices] = True

        def current_seeds() -> np.ndarray:
            """Return the seed of every point from EARLIER rounds.

            Looked up through the module global on purpose, so that
            the frozen helper the seed-choice oracles pin is
            demonstrably the one the cascade uses.
            """
            return choose_seed_indices(
                converged,
                residual,
                grain_id,
                navigation_shape,
                navigation_mask=navigation_mask,
            )

        # CASCADE ROUNDS of requirements D20.2: round r fits, in
        # parallel and at the FULL budget, every not-yet-converged
        # unmasked point with at least one converged same-grain
        # neighbour from rounds STRICTLY earlier than r.  Round
        # membership and every seed are computed here, on the host,
        # from completed rounds only, so nothing depends on
        # intra-round scheduling or on the chunking (D20.3)
        round_index = 0
        while True:
            round_index += 1
            seeds = current_seeds()
            candidates = np.flatnonzero(fittable & ~converged & (seeds >= 0))
            if candidates.size == 0:
                break
            if verbose >= 1:
                print(
                    f"  Cascade round {round_index}: "
                    f"{candidates.size} pattern(s) seeded from a neighbour"
                )
            fit_phase(candidates, max_iterations, homography[seeds[candidates]])
            converted = candidates[converged[candidates]]
            if converted.size == 0:
                break
            seed_round[converted] = round_index

        # RESCUE PASS of requirements D20.2: ONE fit at the full budget
        # for every point still unconverged, from the best seed there
        # is by now, else from its own pass-1 last iterate, else from
        # the phase cross-correlation seed.  This is what makes the
        # pass-1 cap safe: no point is ever left with a cap-truncated
        # pass-1 iterate as its final answer
        rescued = np.flatnonzero(fittable & ~converged)
        if rescued.size:
            seeds = current_seeds()[rescued]
            # a row which is not finite throughout tells ``_fit_chunk``
            # to fall back to the phase cross-correlation seed, which
            # is the third branch of the rule above
            rescue_h0 = np.where(
                (seeds >= 0)[:, None],
                homography[np.where(seeds >= 0, seeds, 0)],
                homography[rescued],
            )
            if verbose >= 1:
                print(f"  Rescue pass: {rescued.size} pattern(s) re-fitted")
            fit_phase(rescued, max_iterations, rescue_h0)
            seed_round[rescued[converged[rescued]]] = SEED_ROUND_RESCUE
    total_time = time.time() - time_start

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

    properties = {
        "homography": homography,
        "Fe": fe,
        "residual": residual,
        "num_iterations": num_iterations,
        "norm_dp": norm_dp,
        "converged": converged,
        "grain_id": grain_id,
        "reference_index": reference_index,
    }
    # The D20.5 absence rule: the property exists on a SEEDED run only,
    # so a default-path result carries EXACTLY the pre-Stage-D set
    if seed_round is not None:
        properties[SEED_ROUND_PROP_NAME] = seed_round
    # The D22.9 absence rule, the same precedent: the two properties
    # exist on a Fourier-Mellin run only
    if fm_seed is not None:
        seed_name, angle_name = FOURIER_MELLIN_PROP_NAMES
        properties[seed_name] = fm_seed
        properties[angle_name] = fm_angle
    return properties


def _check_gpu_arguments(
    device_precision: str,
    seed_precision: str,
    coefficient_dtype,
    chunksize: int | None,
) -> None:
    """Check the arguments only ``backend="gpu"`` reads, in the frozen
    order of requirements D21.1: *device_precision* (D21.4),
    *seed_precision* (D21.5), *coefficient_dtype* (the device
    coefficients are 32-bit under both precisions, D21.4), and then
    *chunksize*, the device batch size, which must be at least 1
    (D21.10.1; the CPU path keeps its silent clamp).

    Raises
    ------
    ValueError
        Naming the offending argument.
    """
    device_precisions = _gpu._batched.DEVICE_PRECISIONS
    if device_precision not in device_precisions:
        raise ValueError(
            f"device_precision must be one of {list(device_precisions)} under "
            f"backend='gpu', not {device_precision!r}"
        )
    seed_precisions = _gpu._batched.SEED_PRECISIONS
    if seed_precision not in seed_precisions:
        raise ValueError(
            f"seed_precision must be one of {list(seed_precisions)} under "
            f"backend='gpu', not {seed_precision!r}"
        )
    try:
        is_float32 = np.dtype(coefficient_dtype) == np.float32
    except TypeError:
        is_float32 = False
    if not is_float32:
        raise ValueError(
            "coefficient_dtype must be numpy.float32 under backend='gpu', not "
            f"{coefficient_dtype!r}: the device coefficients are 32-bit floats"
        )
    if chunksize is not None and int(chunksize) < 1:
        raise ValueError(
            f"chunksize, the device batch size under backend='gpu', must be at "
            f"least 1, not {chunksize}"
        )


def _check_fourier_mellin(fourier_mellin, seed_from_neighbors: bool, xmap) -> None:
    """Check *fourier_mellin* in the frozen order of requirements D22.1:
    (1) the value, one of ``"off"``, ``"auto"`` and ``"always"``,
    case-sensitive, a bool included in what is refused; (2) the D22.11
    combination with ``seed_from_neighbors=True``; (3) under ``"auto"``
    an *xmap* of ``None``.  Called after the ``backend`` string check
    and before every ``"gpu"``-only check, the gate of D21.2, reference
    resolution and any pattern read.

    Raises
    ------
    ValueError
        With the frozen message literals of ``_fourier_mellin``.
    """
    values = _fourier_mellin.FOURIER_MELLIN_VALUES
    if not isinstance(fourier_mellin, str) or fourier_mellin not in values:
        raise ValueError(
            _fourier_mellin.FOURIER_MELLIN_VALUE_MESSAGE.format(
                fourier_mellin=fourier_mellin
            )
        )
    if fourier_mellin == "off":
        return
    if seed_from_neighbors:
        raise ValueError(
            _fourier_mellin.FOURIER_MELLIN_NEIGHBORS_MESSAGE.format(
                fourier_mellin=fourier_mellin
            )
        )
    if fourier_mellin == "auto" and xmap is None:
        raise ValueError(_fourier_mellin.FOURIER_MELLIN_XMAP_MESSAGE)


def _fourier_mellin_routes(
    fourier_mellin: str,
    xmap,
    detector,
    fit_indices: np.ndarray,
    reference_index: np.ndarray,
    navigation_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the ``(n_fit,)`` int8 first-pass routes of a
    Fourier-Mellin run in fit order (requirements D22.6, D22.7).

    ``"always"`` routes every fitted point with the acceptance (route
    1).  ``"auto"`` runs the pre-fit gate on the host: the twist about
    the detector normal of every fitted point against its GRAIN
    reference, :func:`~kikuchipy.indexing._hrebsd._fourier_mellin.\
twist_about_detector_normal`, then
    :func:`~kikuchipy.indexing._hrebsd._fourier_mellin.\
fourier_mellin_routes` at ``FM_GATE_DEG``, both reached through the
    module object at call time.  Masked points are not in *fit_indices*
    and so are never routed.  The frozen ``UserWarning`` of D22.7 is
    issued when no twist of a fitted non-reference point is NaN and
    every one is below ``FM_ZERO_TWIST_DEG`` in magnitude (an input
    condition, never a float equality).
    """
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    n_fit = int(fit_indices.size)
    if fourier_mellin == "always":
        return np.full(n_fit, _fourier_mellin.ROUTE_ACCEPT, dtype=np.int8)
    point_reference = np.asarray(reference_index, dtype=np.int64)[fit_indices]
    twist = np.asarray(
        _fourier_mellin.twist_about_detector_normal(
            xmap,
            detector,
            fit_indices,
            point_reference,
            navigation_shape=navigation_shape,
        ),
        dtype=np.float64,
    ).reshape(n_fit)
    route = np.ascontiguousarray(
        np.asarray(
            _fourier_mellin.fourier_mellin_routes(twist, _fourier_mellin.FM_GATE_DEG),
            dtype=np.int8,
        ).reshape(n_fit)
    )
    others = twist[fit_indices != point_reference]
    if not np.isnan(others).any() and np.all(
        np.abs(others) < _fourier_mellin.FM_ZERO_TWIST_DEG
    ):
        warnings.warn(_fourier_mellin.FOURIER_MELLIN_NO_TWIST_WARNING, UserWarning)
    return route


def _cpu_fourier_mellin_state(state: ReferenceState, upsample_factor: int):
    """Return the numpy ``SeedState`` of one reference carrying its
    ``FourierMellinState`` for the CPU route of requirements D22.10, or
    ``None`` when the state builder refuses the reference (that grain
    then runs as under ``"off"``, is never routed and never retried).

    Built once per reference on the host, before the graph: the
    complex128 reference spectrum of the Stage E seed, the numpy
    subregion resident the acceptance criterion gathers through, and
    the box resident and profile of the Fourier-Mellin state (the
    run's look-up table is the builder's, cached by crop shape).  The
    builders are reached through the module objects at call time.
    """
    ctx = _cpu_seed_context()
    seed_state = _batched.build_seed_state(
        ctx, state, precision="complex128", upsample_factor=int(upsample_factor)
    )
    resident = _batched.build_resident(ctx, state)
    fm_state = _fourier_mellin.build_fourier_mellin_state(ctx, state, resident)
    if fm_state is None:
        return None
    seed_state.fourier_mellin = fm_state
    return seed_state


def _cpu_seed_context():
    """Return the numpy seed context of the CPU route of requirements
    D22.10: ``SeedContext(np, np.fft, make_kernel_namespace("numpy",
    "float64"))``."""
    kernels = _batched.make_kernel_namespace("numpy", "float64")
    return _batched.SeedContext(np, np.fft, kernels)


def _fourier_mellin_seed(ctx, state: ReferenceState, seed_state, target, route: int):
    """Return ``(row, angle, applied)`` of the numpy seam at P = 1 for
    one raw *target* with the *route* code (requirements D22.10): the
    target is preprocessed and splined exactly as ``_fit_pattern``
    does, put in a one-slot ``SeedBatch`` carrying the route key, and
    passed to ``_batched.seed_spectra`` and
    ``_batched.seed_homographies`` through the module object.  A seam
    that wrote no outputs reads as angle NaN and not applied."""
    preprocessed = preprocess(
        np.asarray(target), transfer_function=state.transfer_function
    )
    coefficients = np.ascontiguousarray(
        spline_coefficients(
            preprocessed,
            interpolation=state.interpolation,
            dtype=state.coefficient_dtype,
        )
    )
    batch = _batched.SeedBatch(
        np.ascontiguousarray(preprocessed[None], dtype=np.float64),
        coefficients[None],
        np.zeros(1, dtype=np.int64),
        extras={
            _fourier_mellin.FOURIER_MELLIN_ROUTE_KEY: np.array([route], dtype=np.int8)
        },
    )
    spectra = _batched.seed_spectra(ctx, batch, seed_state)
    rows = _batched.seed_homographies(ctx, batch, spectra, seed_state)
    row = np.array(np.asarray(rows)[0], dtype=np.float64)
    outputs = batch.outputs
    angle = np.nan
    applied = False
    if _fourier_mellin.FOURIER_MELLIN_ANGLE_KEY in outputs:
        angle = float(np.asarray(outputs[_fourier_mellin.FOURIER_MELLIN_ANGLE_KEY])[0])
    if _fourier_mellin.FOURIER_MELLIN_APPLIED_KEY in outputs:
        applied = bool(
            np.asarray(outputs[_fourier_mellin.FOURIER_MELLIN_APPLIED_KEY])[0]
        )
    return row, angle, applied


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
    h0_block: np.ndarray | None = None,
    states: tuple = (),
    options: dict | None = None,
    route_block: np.ndarray | None = None,
    fm_states: tuple = (),
) -> np.ndarray:
    """Return the packed IC-GN results of one chunk of patterns.

    With *route_block* (a Fourier-Mellin run, requirements D22.10) a
    point whose route is nonzero and whose reference has a state in
    *fm_states* is seeded through the numpy seam at P = 1 inside this
    function: if the seam applied the Fourier-Mellin row the fit runs
    from it; otherwise a route-1 point runs with the phase
    cross-correlation seed of ``fit_pattern``, exactly as under
    ``"off"``, and a route-2 (forced, the retry of D22.8) point is NOT
    fitted and gets the D2.6 failure row.  A route-0 point is never
    passed to the seam, so it is bitwise the ``"off"`` result.

    Parameters
    ----------
    patterns_block
        ``(n, nrows, ncols)`` block of patterns, grain ordered.
    state_index_block
        ``(n,)`` block of indices into *states*, paired with
        *patterns_block* along the same named dask index.
    h0_block
        ``(n, 8)`` block of seed homographies of the Stage D cascade,
        paired with *patterns_block* along the same named dask index,
        or ``None`` for the phase cross-correlation seed of every
        pattern. A row which is not finite throughout also means the
        phase cross-correlation seed, which is what the rescue pass of
        requirements D20.2 falls back to for a pattern with no seed
        and no usable pass-1 iterate.
    states
        The per-reference precomputed states.
    options
        Keyword arguments forwarded to :func:`fit_pattern`.
    route_block
        ``(n,)`` int8 block of Fourier-Mellin routes (0 not routed, 1
        routed with the acceptance, 2 forced), paired with
        *patterns_block* along the same named dask index as *h0_block*,
        or ``None`` on an ``"off"`` run.
    fm_states
        Aligned with *states*: each reference's numpy ``SeedState``
        carrying its ``FourierMellinState``, or ``None``.

    Returns
    -------
    results
        ``(n, 12)`` 64-bit float array of packed rows when
        *route_block* is ``None``, else ``(n, 14)`` with the
        Fourier-Mellin angle and applied flag (D22.9).
    """
    options = {} if options is None else options
    n_patterns = int(patterns_block.shape[0])
    width = _ROW_WIDTH if route_block is None else _ROW_WIDTH_FM
    results = np.empty((n_patterns, width), dtype=np.float64)
    context = None
    for i in range(n_patterns):
        position = int(state_index_block[i])
        state = states[position]
        h0 = None
        if h0_block is not None:
            row = np.asarray(h0_block[i], dtype=np.float64)
            if np.all(np.isfinite(row)):
                h0 = row
        angle = np.nan
        applied = False
        fit = True
        if route_block is not None:
            route = int(route_block[i])
            seed_state = fm_states[position] if position < len(fm_states) else None
            if route != _fourier_mellin.ROUTE_NONE and seed_state is not None:
                if context is None:
                    context = _cpu_seed_context()
                row, angle, applied = _fourier_mellin_seed(
                    context, state, seed_state, patterns_block[i], route
                )
                if applied:
                    h0 = row
            # A forced slot the seam did not apply is never fitted
            # (D22.5, D22.8): its first-pass result stands
            fit = applied or route != _fourier_mellin.ROUTE_FORCED
        if fit:
            result = fit_pattern(state, patterns_block[i], h0=h0, **options)
        else:
            result = {
                "h": np.full(N_HOMOGRAPHY_PARAMETERS, np.nan),
                "residual": np.nan,
                "num_iterations": 0,
                "norm_dp": np.nan,
                "converged": False,
            }
        results[i, :HOMOGRAPHY_PROP_SIZE] = result["h"]
        results[i, _SLOT_RESIDUAL] = result["residual"]
        results[i, _SLOT_ITERATIONS] = result["num_iterations"]
        results[i, _SLOT_NORM_DP] = result["norm_dp"]
        results[i, _SLOT_CONVERGED] = 1.0 if result["converged"] else 0.0
        if route_block is not None:
            results[i, _SLOT_FM_ANGLE] = angle
            results[i, _SLOT_FM_APPLIED] = 1.0 if applied else 0.0
    return results


def _fit_chunk_routed(
    patterns_block: np.ndarray,
    state_index_block: np.ndarray,
    route_block: np.ndarray,
    *,
    states: tuple,
    options: dict,
    fm_states: tuple,
) -> np.ndarray:
    """The blockwise entry of a Fourier-Mellin run: :func:`_fit_chunk`
    with the route block paired along the pattern index (D22.10),
    reached through the module global at call time."""
    return _fit_chunk(
        patterns_block,
        state_index_block,
        None,
        states,
        options,
        route_block,
        fm_states,
    )


def _run_chunks(
    patterns,
    fit_indices: np.ndarray,
    state_of_point: np.ndarray,
    states: list,
    chunksize: int,
    *,
    progressbar: bool,
    options: dict,
    h0: np.ndarray | None = None,
    fm_route: np.ndarray | None = None,
    fm_states: tuple | None = None,
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
    h0
        ``(fit_indices.size, 8)`` seed homographies of the Stage D
        cascade, in the order of *fit_indices*, or ``None`` for the
        phase cross-correlation seed of every pattern. It is paired
        with the patterns along the SAME named dask index as the
        state identifiers, so a per-point seed cannot be handed to
        the wrong pattern whatever the chunking is.
    fm_route
        ``(fit_indices.size,)`` int8 Fourier-Mellin routes in fit order
        (requirements D22.10), or ``None`` on an ``"off"`` run, which
        keeps the twelve-wide rows. Paired with the patterns along the
        same named dask index as *h0*; a Fourier-Mellin run never
        carries *h0*.
    fm_states
        Aligned with *states*: each reference's numpy ``SeedState``
        with its ``FourierMellinState``, or ``None`` (D22.10).

    Returns
    -------
    packed
        ``(fit_indices.size, 12)`` 64-bit float array, or
        ``(fit_indices.size, 14)`` with *fm_route* (D22.9).
    """
    if isinstance(patterns, da.Array):
        patterns_da = patterns
    else:
        patterns_da = da.from_array(np.asarray(patterns), chunks=(chunksize, -1, -1))
    ordered = patterns_da[fit_indices].rechunk((chunksize, -1, -1))
    state_da = da.from_array(state_of_point, chunks=(chunksize,))
    if fm_route is not None:
        route_da = da.from_array(
            np.ascontiguousarray(fm_route, dtype=np.int8), chunks=(chunksize,)
        )
        results = da.blockwise(
            _fit_chunk_routed,
            "ij",
            ordered,
            "ikl",
            state_da,
            "i",
            route_da,
            "i",
            states=tuple(states),
            options=options,
            fm_states=tuple(() if fm_states is None else fm_states),
            new_axes={"j": _ROW_WIDTH_FM},
            dtype=np.float64,
            concatenate=True,
        )
    else:
        if h0 is None:
            # The default path, untouched: two paired arguments and no
            # seed array in the graph at all
            arguments = (ordered, "ikl", state_da, "i")
        else:
            h0_da = da.from_array(
                np.ascontiguousarray(h0, dtype=np.float64), chunks=(chunksize, -1)
            )
            arguments = (ordered, "ikl", state_da, "i", h0_da, "im")
        results = da.blockwise(
            _fit_chunk,
            "ij",
            *arguments,
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
    n_chunks: int | None = None,
    fourier_mellin: bool = False,
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
    n_chunks
        Number of chunks, or ``None`` for ``ceil(n_patterns /
        chunksize)``.  The GPU backend passes its grain-pure device
        batch count (requirements D21.9.3, D21.10.5).
    fourier_mellin
        Whether the run uses the Fourier-Mellin seed of requirements
        D22; if so a further line states its host memory (D22.10,
        D22.18). Default is ``False``, the message of every earlier
        release.

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
    if n_chunks is None:
        n_chunks = math.ceil(n_patterns / chunksize) if n_patterns else 0
    # The model, not a measurement, exactly as the spherical
    # information message states its own: the eight 64-bit
    # steepest-descent columns, the 32-bit coefficient plane and the
    # remaining subregion sized 64-bit planes, at the pattern size,
    # which bounds the subregion from above
    bytes_per_reference = (
        nrows * ncols * (_N_STEEPEST_DESCENT_COLUMNS * 8 + 4 + _RESIDENT_F64_PLANES * 8)
    )
    lines = [
        "HREBSD-DIC information:",
        f"  Correlating {n_patterns} pattern(s) of shape ({nrows}, {ncols}) "
        f"against {int(n_grains)} reference(s)",
        f"  Chunking: {n_chunks} chunk(s) of up to {chunksize} pattern(s)",
        f"  Estimated memory per reference: {bytes_per_reference / 1024**2:.1f} MB",
    ]
    if fourier_mellin:
        # Requirements D22.10: a reference with a routed point also
        # holds, on the host, the numpy seed state (the complex128
        # reference spectrum), the bounding-box resident (two 64-bit
        # coordinate planes) and the numpy subregion resident; the
        # polar look-up table is held once per run.  The model, bounded
        # by the pattern size as above, not a measurement
        fm_bytes = nrows * ncols * (_FM_HOST_COMPLEX_BYTES + _FM_HOST_PLANE_BYTES)
        side = min(nrows, ncols)
        n_rho = (
            math.floor(
                (_fourier_mellin.FM_RHO_MAX - _fourier_mellin.FM_RHO_MIN) * side + 1e-9
            )
            + 1
        )
        lut_bytes = _fourier_mellin.FM_N_THETA * n_rho * 4 * (8 + 8)
        lines.append(
            "  Fourier-Mellin seed: up to "
            f"{fm_bytes / 1024**2:.1f} MB more per reference with a routed point, "
            f"plus {lut_bytes / 1024**2:.1f} MB once for the polar look-up table"
        )
    return "\n".join(lines)
