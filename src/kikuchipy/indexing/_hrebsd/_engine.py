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

import numpy as np

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


class ReferenceState:
    """Per-reference precomputed state of the IC-GN loop.

    One instance per grain reference, reused by every target of that
    grain (requirements D2.1).  Roughly five subregion sized planes
    are resident, about 4.6 MB at 480 by 480.

    Parameters
    ----------
    pattern
        The reference pattern of shape ``(nrows, ncols)``.
    mask
        Boolean subregion mask of the shape of *pattern*, ``True``
        for the pixels used, from
        :func:`~kikuchipy.indexing._hrebsd._preprocessing.subregion_mask`.
    pc_pixels
        ``(PCx_px, PCy_px, DD_px)`` of this reference, which defines
        the PC-centred coordinate frame shared by the reference and
        every target of the grain (requirements D1.3).
    transfer_function
        Band-pass transfer function applied to the reference and to
        every target alike, or ``None``.
    window
        Window applied to the subregion, or ``None``.
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
        ``(n_pixels, 8)`` steepest-descent images.
    hessian
        ``(8, 8)`` Gauss-Newton Hessian.
    cho_factor
        The cached :func:`scipy.linalg.cho_factor` of *hessian*.
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
        raise NotImplementedError


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
    """
    raise NotImplementedError


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

    Notes
    -----
    A non-converged fit keeps its last iterate and reports
    ``converged=False``; it is never zeroed.  A failed pattern (a
    constant subregion, a non-finite ``CIC``, or any per-pattern
    exception) is reported with NaN in ``"h"`` and ``"residual"`` and
    ``converged=False``, following the spherical result contract.
    """
    raise NotImplementedError


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
        Whether to apply a Hann window over the subregion. Default is
        ``False``.
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
        Vertical and horizontal scan step sizes ``(dy, dx)`` from the
        signal's navigation axes, used by the internal beam-scan
        projection centre model when the detector carries a single
        projection centre.
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
    raise NotImplementedError


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
    raise NotImplementedError
