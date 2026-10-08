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

"""The array-module-agnostic ("xp-agnostic") batched core of the GPU
backend of the HREBSD IC-GN engine: the seed seam, the launch layout,
the per-precision kernel interface and the lockstep IC-GN loop.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Requirements D21.5 (the seed seam), D21.6 (the batched IC-GN
semantics), D21.7.3 (the launch layout) and D21.9.5 (the names)
govern.  The argument lists beyond the frozen names are fixed by the
failing-tests commit and recorded in the docstring of
``tests/test_indexing/test_hrebsd_gpu.py``.

The orchestration and the seed stage are ONE code path, run under
numpy in the default suite and under cupy in the gated one; the
per-pixel and reduction kernels are written twice, as CUDA RawKernels
(``_cuda.py``) and as a numpy twin of the ``"float64"`` build only,
behind the same :class:`KernelNamespace` (D21.14.3).  CuPy is never
imported at module scope (D21.13), and ``_engine`` is never imported at
module scope (D21.9.5).

Call-time seams (frozen by the failing-tests commit, the test module's
docstring): the runner reaches :func:`build_resident`,
:func:`build_seed_state`, :func:`seed_spectra` and
:func:`seed_homographies` through this module's globals at call time,
and every lockstep iteration calls the :class:`KernelNamespace` entry
points as attributes of the session's namespace (never bound to locals
before the loop); any kernel fusion lives inside one entry point,
never across two.

The per-pattern sums of the shifted-data form (D21.4, D21.6.5), shared
by both builds: with ``vp = v - K`` the warped values shifted by the
per-pattern shift K, ``a = (gx_w * w) * vp`` and ``b = (gy_w * w) *
vp`` (``gx_w``, ``gy_w`` the window-weighted gradient columns of the
host steepest-descent block, ``w`` the window weight or 1) and ``p = a
* x + b * y``, the :data:`N_SUMS` sums along the pixel axis are, in
this order, ``sum(vp)``, ``sum(vp**2)``, ``sum(a * x)``, ``sum(a *
y)``, ``sum(a)``, ``sum(b * x)``, ``sum(b * y)``, ``sum(b)``, ``sum(p *
x)`` and ``sum(p * y)``: the last eight are ``SD_w^T (w * vp)`` with
the two perspective columns negated.
"""

import numpy as np

from kikuchipy.indexing._hrebsd._interpolation import _bicubic_evaluate

# The preprocessing sub-batch bound of D21.7.3: every batched FFT and
# seed matmul runs at P = min(SUB_BATCH_SIZE, B) slots, the tail
# sub-batch padded to P
SUB_BATCH_SIZE: int = 32

# The two seed precisions of D21.5, the default first
SEED_PRECISIONS: tuple[str, ...] = ("complex128", "complex64")

# The two device precisions of D21.4, the default first
DEVICE_PRECISIONS: tuple[str, ...] = ("mixed", "float64")

# The number of per-pattern sums :meth:`KernelNamespace.pixel_sums`
# returns, in the order of the module docstring
N_SUMS: int = 10

# The threads per block and the target pixels per thread of the fused
# per-pixel kernels (D21.7.3): the block count follows from the
# subregion pixel count alone
LAUNCH_THREADS: int = 256
LAUNCH_PIXELS_PER_THREAD: int = 16

# The packed-row layout of ``_engine._fit_chunk``, the default of
# :func:`run_lockstep` (the engine passes its own, D21.9.5)
DEFAULT_ROW_SLOTS: dict = {
    "width": 12,
    "residual": 8,
    "iterations": 9,
    "norm_dp": 10,
    "converged": 11,
}

N_PARAMETERS: int = 8


class NumpyOutOfMemoryError(MemoryError):
    """The out-of-memory error of the numpy :class:`KernelNamespace`,
    the module-local stand-in for cupy's ``OutOfMemoryError`` that the
    default-suite twins raise (requirements D21.9.5)."""


class KernelNamespace:
    """The per-precision kernel interface of requirements D21.9.5.

    Implemented once as CUDA RawKernels and once as the numpy twin of
    the ``"float64"`` build (D21.14.3).  A plain mutable container of
    fields: build one with :func:`make_kernel_namespace`.

    Parameters
    ----------
    xp
        The array namespace, :mod:`numpy` or ``cupy``.
    fft
        The namespace's FFT module.
    precision
        ``"mixed"`` or ``"float64"`` (D21.4).
    out_of_memory_error
        The exception type the out-of-memory loop of D21.10.4 catches:
        cupy's ``OutOfMemoryError`` under cupy,
        :class:`NumpyOutOfMemoryError` under numpy.
    gather
        ``gather(resident, coefficients, matrices) -> (values,
        coordinate_ok)``: the batched mirror-folded bicubic evaluation
        of D21.6.3 at the subregion pixels of *resident* warped by the
        carried ``(B, 3, 3)`` float64 *matrices*, on the ``(B, nrows,
        ncols)`` float32 *coefficients*.  ``values`` is ``(B, n)`` at
        the pixel precision (float64 under ``"float64"``, float32
        under ``"mixed"``); ``coordinate_ok`` is ``(B,)`` bool, False
        where any warped coordinate is not finite, flagged BEFORE the
        fold.
    pixel_sums
        ``pixel_sums(resident, values, shifts) -> sums``: the
        :data:`N_SUMS` per-pattern sums of the module docstring over
        the pixels of ``values - shifts[:, None]`` (the shifted-data
        form of D21.4), ``(B, N_SUMS)`` float64.
    reduce_solve_update
        ``reduce_solve_update(resident, sums, lockstep, options) ->
        None``: from *sums*, the gradient, the 8x8 solve on the
        uploaded host Cholesky factor, ``dp * step_scale``, the
        NaN-propagating corner norm, the update with its W33
        renormalisation, the shift update, the status flags and the
        retirement of D21.6, applied IN PLACE to the active slots of
        the :class:`LockstepState` *lockstep*.  *options* carries
        ``min_step``, ``step_scale`` and ``max_iterations``.
    final_criterion
        ``final_criterion(resident, values, shifts) -> residual``: the
        D2.7 criterion ``sum((w * r)**2)`` per pattern at the returned
        homography, ``(B,)`` float64, NaN where the warped values have
        a zero or non-finite centred norm.
    """

    def __init__(
        self,
        xp,
        fft,
        precision: str,
        out_of_memory_error: type,
        gather,
        pixel_sums,
        reduce_solve_update,
        final_criterion,
    ) -> None:
        self.xp = xp
        self.fft = fft
        self.precision = precision
        self.out_of_memory_error = out_of_memory_error
        self.gather = gather
        self.pixel_sums = pixel_sums
        self.reduce_solve_update = reduce_solve_update
        self.final_criterion = final_criterion


def make_kernel_namespace(namespace: str, device_precision: str) -> KernelNamespace:
    """Return the :class:`KernelNamespace` of *namespace* (``"numpy"``
    or ``"cupy"``) at *device_precision* (D21.9.5).

    Under ``"numpy"`` only ``"float64"`` exists (D21.14.3: the numpy
    twin's gather is the numba bicubic kernel, which works in f64), and
    ``"mixed"`` raises ``ValueError``.  Under ``"cupy"`` the RawKernels
    are compiled in ``_cuda.make_cupy_kernel_namespace``, without fast
    math (D21.4), each one CONSTRUCTED through the attribute
    ``cupy.RawKernel`` on every call of this function: no
    ``cupy.RawModule`` and no Python-level cache of kernel objects
    across calls (CuPy's own compile cache is fine), so the gated
    launch spy sees every launch.  The callers validate both strings
    first (``run_hrebsd_dic``, the session).

    Raises
    ------
    ValueError
        For ``("numpy", "mixed")``.
    """
    if namespace == "numpy":
        if device_precision != "float64":
            raise ValueError(
                "the numpy kernel namespace reproduces the 'float64' device "
                f"precision only, not {device_precision!r}"
            )
        return KernelNamespace(
            np,
            np.fft,
            "float64",
            NumpyOutOfMemoryError,
            _numpy_gather,
            _numpy_pixel_sums,
            _numpy_reduce_solve_update,
            _numpy_final_criterion,
        )
    from kikuchipy.indexing._hrebsd import _cuda

    return _cuda.make_cupy_kernel_namespace(device_precision)


# ----------------------------- Seed seam ----------------------------- #


class SeedContext:
    """The execution context of the seed seam, once per run
    (requirements D21.5).

    Parameters
    ----------
    xp
        :mod:`numpy` or ``cupy``.
    fft
        The namespace's FFT module.
    kernels
        The :class:`KernelNamespace`; its ``gather`` is the warp a
        later stage de-rotates through.
    """

    def __init__(self, xp, fft, kernels) -> None:
        self.xp = xp
        self.fft = fft
        self.kernels = kernels


class SeedState:
    """The per-reference seed state of requirements D21.5, built in
    the batch's namespace from a ``ReferenceState`` by
    :func:`build_seed_state`.

    Parameters
    ----------
    bounds
        ``(r0, r1, c0, c1)``, the D5 bounding box of the subregion.
    reference_spectrum
        ``(sr, sc)`` complex, ``fft2`` of the zero-mean unit-norm
        reference crop, at *precision*.
    precision
        ``"complex128"`` or ``"complex64"``.
    upsample_factor
        The public D5 knob.
    fourier_mellin
        ``None`` (the default, and always with ``fourier_mellin="off"``)
        or the reference's
        :class:`~kikuchipy.indexing._hrebsd._fourier_mellin.FourierMellinState`
        (requirements D22.6, a purely additive extension of D21.5).
    """

    def __init__(
        self,
        bounds,
        reference_spectrum,
        precision: str,
        upsample_factor: int,
        fourier_mellin=None,
    ) -> None:
        self.bounds = tuple(int(i) for i in bounds)
        self.reference_spectrum = reference_spectrum
        self.precision = precision
        self.upsample_factor = int(upsample_factor)
        self.fourier_mellin = fourier_mellin


class SeedBatch:
    """One preprocessing sub-batch of P slots (requirements D21.5,
    D21.7.3).

    Parameters
    ----------
    targets
        ``(P, nrows, ncols)`` float64 preprocessed (D4) targets.
    coefficients
        ``(P, nrows, ncols)`` float32 spline coefficients of the same
        targets.
    pattern_index
        ``(P,)`` int64 flat map indices of the slots in fit order,
        ``-1`` on padded slots: a HOST numpy array, so that later
        stages key host data by it.
    extras
        Dictionary of per-slot ``(P, ...)`` INPUT arrays on the host,
        empty unless a Fourier-Mellin run fills the route key
        ``"fourier_mellin_route"`` (requirements D22.6).  ``None``
        gives a new empty dictionary.
    outputs
        Dictionary of per-slot ``(P,)`` OUTPUT arrays in ``ctx.xp``,
        written only by :func:`seed_homographies` (requirements D22.6:
        ``"fourier_mellin_angle"`` and ``"fourier_mellin_applied"``,
        present if and only if the route key has a nonzero entry).
        ``None`` gives a new empty dictionary.
    """

    def __init__(
        self, targets, coefficients, pattern_index, extras=None, outputs=None
    ) -> None:
        self.targets = targets
        self.coefficients = coefficients
        self.pattern_index = pattern_index
        self.extras = {} if extras is None else extras
        self.outputs = {} if outputs is None else outputs


def build_seed_state(
    ctx: SeedContext,
    state,
    *,
    precision: str = "complex128",
    upsample_factor: int = 16,
) -> SeedState:
    """Return the :class:`SeedState` of one ``ReferenceState`` in the
    namespace of *ctx*: the reference crop uploaded, zero-mean
    unit-norm normalised and transformed there, once per reference
    (D21.5).

    *precision* is one of :data:`SEED_PRECISIONS`, validated by
    ``run_hrebsd_dic``.
    """
    xp = ctx.xp
    crop = xp.asarray(np.ascontiguousarray(state.reference_subregion), dtype=xp.float64)
    with np.errstate(all="ignore"):
        zmn = _zero_mean_normalize_crops(xp, crop[None])[0]
        del crop
        zmn = zmn.astype(np.dtype(precision))
        spectrum = ctx.fft.fft2(zmn)
        del zmn
    spectrum = xp.ascontiguousarray(spectrum.astype(np.dtype(precision), copy=False))
    return SeedState(state.bounds, spectrum, precision, int(upsample_factor))


def seed_spectra(ctx, batch, seed_state):
    """Return ``target_spectra``, ``(P, sr, sc)`` complex at
    ``seed_state.precision``: ``fft2`` of each target's zero-mean
    unit-norm crop, computed ONCE per sub-batch (requirements D21.5,
    frozen signature).

    A crop with a zero or non-finite centred norm gives a non-finite
    spectrum, which :func:`seed_homographies` turns into a seed
    failure for that slot only.
    """
    xp = ctx.xp
    r0, r1, c0, c1 = seed_state.bounds
    dtype = np.dtype(seed_state.precision)
    crops = xp.asarray(batch.targets[:, r0:r1, c0:c1], dtype=xp.float64)
    with np.errstate(all="ignore"):
        zmn = _zero_mean_normalize_crops(xp, crops)
        spectra = ctx.fft.fft2(zmn.astype(dtype), axes=(1, 2))
    return xp.ascontiguousarray(spectra.astype(dtype, copy=False))


def seed_homographies(ctx, batch, target_spectra, seed_state):
    """Return ``h0``, ``(P, 8)`` float64 C-contiguous in ``ctx.xp``:
    the FULL starting homography of every slot, in the frame and
    layout of ``fit_pattern``'s ``h0`` (requirements D21.5, frozen
    signature).

    Stage E implements the translation-only phase cross-correlation
    seed of D5 behind this seam, row for row what ``initial_guess``
    returns: the batched transcription of
    ``skimage.registration.phase_cross_correlation`` with
    ``upsample_factor`` on the zero-mean unit-norm crops (the
    ``max(|X|, 100 * eps)`` guard, the inverse FFT, the argmax of the
    modulus, then for ``upsample_factor > 1`` only the rounding to
    ``1 / upsample_factor`` px and the ``ceil(1.5 * upsample_factor)``
    square upsampled DFT by matmuls).  A row which is not finite
    throughout marks a seed FAILURE for that pattern only (a zero or
    non-finite crop norm, a non-finite correlation peak); every row is
    non-finite for ``upsample_factor < 1``, the CPU outcome.  Padded
    slots' rows are discarded by the caller.  The pipeline calls this
    through the module global at call time.
    """
    xp = ctx.xp
    n_slots, n_rows, n_cols = (int(i) for i in target_spectra.shape)
    upsample = int(seed_state.upsample_factor)
    if upsample < 1:
        return xp.full((n_slots, N_PARAMETERS), np.nan, dtype=xp.float64)
    complex_dtype = np.dtype(seed_state.precision)
    real_dtype = np.float64 if complex_dtype == np.complex128 else np.float32
    eps = np.finfo(real_dtype).eps
    shape = np.array([n_rows, n_cols], dtype=real_dtype)
    midpoint = np.trunc(shape / 2)
    rows = xp.arange(n_slots)
    with np.errstate(all="ignore"):
        product = seed_state.reference_spectrum[None] * xp.conj(target_spectra)
        product = product / xp.maximum(xp.abs(product), 100 * eps)
        cross = ctx.fft.ifft2(product, axes=(1, 2))
        cross = cross.reshape(n_slots, n_rows * n_cols)
        peak = xp.argmax(xp.abs(cross), axis=1)
        peak_value = cross[rows, peak]
        # the correlation dies here, before the upsampled DFT's
        # transients (the V9(o) pool calibration, ledger 103)
        del cross
        shift_r = (peak // n_cols).astype(real_dtype)
        shift_c = (peak % n_cols).astype(real_dtype)
        shift_r = xp.where(shift_r > midpoint[0], shift_r - shape[0], shift_r)
        shift_c = xp.where(shift_c > midpoint[1], shift_c - shape[1], shift_c)
        if upsample > 1:
            factor = real_dtype(upsample)
            shift_r = xp.round(shift_r * factor) / factor
            shift_c = xp.round(shift_c * factor) / factor
            region_size = np.ceil(factor * real_dtype(1.5))
            dftshift = np.trunc(region_size / real_dtype(2.0))
            region = int(region_size)
            offset_r = dftshift - shift_r * factor
            offset_c = dftshift - shift_c * factor
            # ``conj(product)`` written straight into its C-contiguous
            # transposed layout, the one copy the matmul reads, and the
            # product released
            data = xp.empty((n_slots, n_cols, n_rows), dtype=product.dtype)
            xp.conjugate(xp.swapaxes(product, 1, 2), out=data)
            del product
            kernel_c = _upsampled_dft_kernel(
                xp, region, n_cols, factor, offset_c, real_dtype, complex_dtype
            )
            # ``tensordot(kernel, data, axes=(1, -1))`` per slot, as
            # skimage contracts the column axis first
            stage = xp.matmul(kernel_c, data)
            del data
            kernel_r = _upsampled_dft_kernel(
                xp, region, n_rows, factor, offset_r, real_dtype, complex_dtype
            )
            upsampled = xp.matmul(
                kernel_r, xp.ascontiguousarray(xp.swapaxes(stage, 1, 2))
            )
            upsampled = xp.conj(upsampled).reshape(n_slots, region * region)
            peak = xp.argmax(xp.abs(upsampled), axis=1)
            peak_value = upsampled[rows, peak]
            maxima_r = (peak // region).astype(real_dtype) - dftshift
            maxima_c = (peak % region).astype(real_dtype) - dftshift
            shift_r = shift_r + maxima_r / factor
            shift_c = shift_c + maxima_c / factor
        failed = xp.isnan(peak_value)
        h0 = xp.zeros((n_slots, N_PARAMETERS), dtype=xp.float64)
        # ``initial_guess``: the displacement carrying the reference
        # onto the target is the negative of the registering shift
        h0[:, 2] = -shift_c.astype(xp.float64)
        h0[:, 5] = -shift_r.astype(xp.float64)
        h0 = xp.where(failed[:, None], xp.nan, h0)
    return xp.ascontiguousarray(h0, dtype=xp.float64)


def _zero_mean_normalize_crops(xp, crops):
    """Return the ``(P, sr, sc)`` *crops* zero-mean unit-norm per slot
    (the D5 normalisation of ``initial_guess``), NaN where the centred
    norm is zero; reductions run per slot over the pixel axes."""
    n_slots = int(crops.shape[0])
    # One private copy, then in place: the transient stays one crop
    flat = xp.array(crops, dtype=xp.float64, order="C", copy=True)
    flat = flat.reshape(n_slots, -1)
    mean = flat.mean(axis=1)
    flat -= mean[:, None]
    norm = xp.sqrt((flat * flat).sum(axis=1))
    flat /= norm[:, None]
    return flat.reshape(crops.shape)


def _upsampled_dft_kernel(xp, region, n, factor, offset, real_dtype, complex_dtype):
    """Return the ``(P, region, n)`` upsampled-DFT kernels of skimage's
    ``_upsampled_dft`` for one axis of length *n*, one per slot of the
    ``(P,)`` sample-region *offset*."""
    frequencies = xp.asarray(np.fft.fftfreq(n, factor).astype(real_dtype))
    samples = xp.arange(region, dtype=real_dtype)
    kernel = (samples[None, :] - offset[:, None])[:, :, None] * frequencies[None, None]
    im2pi = 1j * 2 * np.pi
    kernel = xp.exp(-im2pi * kernel)
    return xp.ascontiguousarray(kernel.astype(complex_dtype, copy=False))


# ----------------------- Per-reference residents --------------------- #


class ReferenceResident:
    """The per-reference device resident set of requirements D21.9.3
    and D21.10.2, built by :func:`build_resident`: the uploaded mirror
    of a ``ReferenceState`` at the device precision.  Opaque to the
    tests; the fields are the contract between the orchestration and
    the two kernel builds.

    Attributes
    ----------
    precision
        The device precision, ``"mixed"`` or ``"float64"``.
    shape
        ``(nrows, ncols)`` of the patterns.
    n_pixels
        The subregion pixel count n.
    xi_x, xi_y
        ``(n,)`` PC-centred coordinates of the subregion pixels at the
        pixel precision (float64 under ``"float64"``, float32 under
        ``"mixed"``; no copy at the other precision is held).
    columns, rows
        ``(n,)`` int32 array-frame column and row of every subregion
        pixel, in the order of *xi_x*, under ``"mixed"`` only (the
        displacement form); ``None`` under ``"float64"``.
    offset
        ``(PCx - 0.5, PCy - 0.5)``, the shift from the PC-centred to the
        array frame, host floats.
    reference
        ``(n,)`` zero-mean unit-norm reference at the pixel precision.
    gradient_x, gradient_y
        ``(n,)`` window-weighted gradient columns ``gx_w``, ``gy_w`` of
        the host steepest-descent block times the window weight (so
        ``gx * w**2`` under a window, ``gx`` without), at the pixel
        precision; the per-pixel factors of the sums.
    weights
        ``(n,)`` window weights at the pixel precision, or ``None``.
    constants
        ``(16,)`` float64 per-reference constants of D21.6.5, built on
        the host in float64 from the WEIGHTED block: ``SD_w^T (w *
        ref)`` then ``SD_w^T w``; host and device copies in
        ``constants_host`` and ``constants``.
    gradient_scale
        ``2 / ref_norm``, a host float.
    cholesky, cholesky_host
        ``(8, 8)`` float64 upper Cholesky factor ``U`` of the host
        Hessian, ``H = U^T U``, on the device and on the host.
    corners, corners_host
        ``(4, 2)`` float64 subregion corners of the D2.5 norm.
    mask_index
        ``(n,)`` int32 flat indices of the subregion in a pattern.
    transfer_function
        The host transfer function ``ifftshift``-ed (spectrum order),
        uploaded as float64, or ``None`` (no band-pass).
    """


def build_resident(ctx: SeedContext, state) -> ReferenceResident:
    """Return the :class:`ReferenceResident` of one ``ReferenceState``
    in the namespace and at the precision of ``ctx.kernels``; an
    upload is a byte copy and never changes a value (D21.9.3).
    Everything is derived on the host in float64 from the state, so
    the Hessian, its Cholesky factor and the reference are the CPU's
    to the bit."""
    xp = ctx.xp
    precision = ctx.kernels.precision
    pixel_dtype = np.float32 if precision == "mixed" else np.float64
    resident = ReferenceResident()
    resident.precision = precision
    resident.shape = tuple(int(i) for i in state.shape)
    resident.n_pixels = int(state.n_pixels)

    rows, columns = np.nonzero(state.mask)
    mask_index = np.ravel_multi_index((rows, columns), resident.shape)
    steepest_descent = np.asarray(state.steepest_descent, dtype=np.float64)
    weights = None
    if state.weights is not None:
        weights = np.asarray(state.weights, dtype=np.float64)
    gradient_x = steepest_descent[:, 2]
    gradient_y = steepest_descent[:, 5]
    if weights is None:
        reference_w = state.reference
        weight_sum = steepest_descent.sum(axis=0)
    else:
        gradient_x = gradient_x * weights
        gradient_y = gradient_y * weights
        reference_w = weights * state.reference
        weight_sum = steepest_descent.T @ weights
    constants = np.concatenate([steepest_descent.T @ reference_w, weight_sum])
    factor, lower = state.cho_factor
    upper = np.tril(factor).T if lower else np.triu(factor)

    def upload(array, dtype):
        return xp.asarray(np.ascontiguousarray(np.asarray(array).astype(dtype)))

    resident.xi_x = upload(state.xi_x, pixel_dtype)
    resident.xi_y = upload(state.xi_y, pixel_dtype)
    resident.reference = upload(state.reference, pixel_dtype)
    resident.gradient_x = upload(gradient_x, pixel_dtype)
    resident.gradient_y = upload(gradient_y, pixel_dtype)
    resident.weights = None if weights is None else upload(weights, pixel_dtype)
    resident.mask_index = upload(mask_index, np.int32)
    if precision == "mixed":
        # The displacement form's array-frame pixel indices; the f64
        # build warps the PC-centred coordinates as the CPU does
        resident.columns = upload(columns, np.int32)
        resident.rows = upload(rows, np.int32)
    else:
        resident.columns = None
        resident.rows = None
    resident.offset = (
        float(state.pc_pixels[0] - 0.5),
        float(state.pc_pixels[1] - 0.5),
    )
    resident.constants_host = np.ascontiguousarray(constants, dtype=np.float64)
    resident.constants = upload(constants, np.float64)
    resident.gradient_scale = float(2.0 / state.reference_norm)
    resident.cholesky_host = np.ascontiguousarray(upper, dtype=np.float64)
    resident.cholesky = upload(upper, np.float64)
    resident.corners_host = np.ascontiguousarray(state.corners, dtype=np.float64)
    resident.corners = upload(state.corners, np.float64)
    resident.transfer_function = None
    if state.transfer_function is not None:
        # The spectrum-ORDER transfer function: multiplying the
        # unshifted spectrum by ``ifftshift(tf)`` is a permutation of
        # the host's ``fftshift``, product, ``ifftshift``
        resident.transfer_function = upload(
            np.fft.ifftshift(np.asarray(state.transfer_function)), np.float64
        )
    return resident


# ---------------------------- Lockstep loop -------------------------- #


class LockstepState:
    """The per-slot state of one batch iterating in LOCKSTEP under
    per-pattern masks (requirements D21.6), all ``(B, ...)`` arrays in
    one namespace.

    Parameters
    ----------
    matrices
        ``(B, 3, 3)`` float64 carried shape-function matrices (D21.6.4).
    shifts
        ``(B,)`` float64 per-pattern shifts K of D21.4, always a value
        the pixel precision represents exactly.
    iterations
        ``(B,)`` int64 per-pattern iteration counts.
    norm_dp
        ``(B,)`` float64 last corner norms, NaN before the first step.
    active
        ``(B,)`` bool, False once a pattern retires.
    converged
        ``(B,)`` bool.
    failed
        ``(B,)`` bool, the status flags of D21.6.1.
    """

    def __init__(
        self, matrices, shifts, iterations, norm_dp, active, converged, failed
    ) -> None:
        self.matrices = matrices
        self.shifts = shifts
        self.iterations = iterations
        self.norm_dp = norm_dp
        self.active = active
        self.converged = converged
        self.failed = failed


def _launch_layout(n_pixels: int) -> tuple[int, int, int]:
    """Return ``(threads, blocks_per_pattern, pixels_per_thread)`` of
    the fused kernels, a function of the subregion pixel count ONLY,
    never of the batch size (requirements D21.7.3).

    :data:`LAUNCH_THREADS` threads per block, as many blocks per
    pattern as give about :data:`LAUNCH_PIXELS_PER_THREAD` pixels per
    thread, and the pixels per thread that then cover the subregion.
    """
    n_pixels = max(int(n_pixels), 1)
    threads = LAUNCH_THREADS
    blocks = max(1, -(-n_pixels // (threads * LAUNCH_PIXELS_PER_THREAD)))
    per_thread = -(-n_pixels // (threads * blocks))
    return threads, blocks, per_thread


def matrices_from_parameters(xp, h):
    """Return the ``(B, 3, 3)`` float64 shape-function matrices of the
    ``(B, 8)`` parameters *h*, the CPU's ``shape_function`` rule."""
    n = int(h.shape[0])
    matrices = xp.empty((n, 3, 3), dtype=xp.float64)
    matrices[:, 0, 0] = 1 + h[:, 0]
    matrices[:, 0, 1] = h[:, 1]
    matrices[:, 0, 2] = h[:, 2]
    matrices[:, 1, 0] = h[:, 3]
    matrices[:, 1, 1] = 1 + h[:, 4]
    matrices[:, 1, 2] = h[:, 5]
    matrices[:, 2, 0] = h[:, 6]
    matrices[:, 2, 1] = h[:, 7]
    matrices[:, 2, 2] = 1.0
    return matrices


def parameters_from_matrices(xp, matrices):
    """Return the ``(B, 8)`` parameters of ``(B, 3, 3)`` *matrices*,
    the CPU's ``homography_parameters`` rule (divide by ``W33``
    first)."""
    with np.errstate(all="ignore"):
        normalised = matrices / matrices[:, 2, 2][:, None, None]
    h = xp.empty((int(matrices.shape[0]), N_PARAMETERS), dtype=xp.float64)
    h[:, 0] = normalised[:, 0, 0] - 1
    h[:, 1] = normalised[:, 0, 1]
    h[:, 2] = normalised[:, 0, 2]
    h[:, 3] = normalised[:, 1, 0]
    h[:, 4] = normalised[:, 1, 1] - 1
    h[:, 5] = normalised[:, 1, 2]
    h[:, 6] = normalised[:, 2, 0]
    h[:, 7] = normalised[:, 2, 1]
    return h


def round_to_pixel_precision(xp, values, precision: str):
    """Return float64 *values* rounded to the pixel precision of
    *precision* (float32 under ``"mixed"``, unchanged under
    ``"float64"``), so that a shift K is exactly representable in the
    pixel pass (D21.4)."""
    if precision == "mixed":
        return values.astype(xp.float32).astype(xp.float64)
    return values


def initial_shifts(ctx, resident, batch):
    """Return the ``(P,)`` float64 initial shifts K0 of one sub-batch
    (requirements D21.4): the mean of each slot's preprocessed
    subregion, a fixed-order reduction along the pixel axis, rounded
    to the pixel precision of ``ctx.kernels`` so the pixel pass
    represents it exactly."""
    xp = ctx.xp
    n_slots = int(batch.targets.shape[0])
    with np.errstate(all="ignore"):
        subregion = xp.asarray(batch.targets, dtype=xp.float64).reshape(n_slots, -1)
        subregion = xp.ascontiguousarray(subregion[:, resident.mask_index])
        mean = subregion.sum(axis=1) / float(resident.n_pixels)
    return round_to_pixel_precision(xp, mean, ctx.kernels.precision)


def run_lockstep(
    kernels, resident, coefficients, h0, shifts, real, options, row_slots=None
):
    """Return the ``(B, width)`` float64 packed rows (in ``kernels.xp``) of
    the batched IC-GN of requirements D21.6 on one batch.

    Lockstep under per-pattern masks: each iteration calls
    ``kernels.gather``, ``.pixel_sums`` and
    ``.reduce_solve_update`` by ATTRIBUTE, until no slot is active or
    ``max_iterations`` lockstep iterations ran; then
    ``kernels.final_criterion`` at the returned homographies.  A
    slot whose seed row is not finite throughout, or which is not
    *real* (a padded slot), never becomes active; a non-finite warped
    coordinate fails that slot before its sums are read.  Failed slots
    get the D2.6 contract (NaN ``h``, ``residual`` and ``norm_dp``, 0
    iterations, not converged); so does a slot whose final criterion is
    not finite (the CPU's raise in the criterion after the loop).

    Parameters
    ----------
    kernels
        The session's :class:`KernelNamespace`; its ``xp`` is the
        batch's namespace.
    resident
        The batch grain's :class:`ReferenceResident`.
    coefficients
        ``(B, nrows, ncols)`` float32 spline coefficients.
    h0
        ``(B, 8)`` float64 starting homographies.
    shifts
        ``(B,)`` float64 initial shifts K0, exact at the pixel
        precision.
    real
        ``(B,)`` bool, host or device, False on padded slots.
    options
        ``max_iterations``, ``min_step`` and ``step_scale``.
    row_slots
        The packed-row layout; :data:`DEFAULT_ROW_SLOTS` if not given.
    """
    xp = kernels.xp
    slots = DEFAULT_ROW_SLOTS if row_slots is None else row_slots
    max_iterations = int(options["max_iterations"])
    step_options = {
        "min_step": float(options["min_step"]),
        "step_scale": float(options["step_scale"]),
        "max_iterations": max_iterations,
    }
    n = int(h0.shape[0])
    h0 = xp.asarray(h0, dtype=xp.float64)
    seed_ok = xp.all(xp.isfinite(h0), axis=1)
    with np.errstate(all="ignore"):
        matrices = matrices_from_parameters(xp, h0)
    active = seed_ok & xp.asarray(real, dtype=bool)
    lockstep = LockstepState(
        matrices,
        xp.asarray(shifts, dtype=xp.float64).copy(),
        xp.zeros(n, dtype=xp.int64),
        xp.full(n, np.nan, dtype=xp.float64),
        active if max_iterations > 0 else xp.zeros(n, dtype=bool),
        xp.zeros(n, dtype=bool),
        ~seed_ok,
    )
    for _ in range(max_iterations):
        if not bool(xp.any(lockstep.active)):
            break
        values, coordinate_ok = kernels.gather(
            resident, coefficients, lockstep.matrices
        )
        bad = lockstep.active & ~coordinate_ok
        lockstep.failed = lockstep.failed | bad
        lockstep.active = lockstep.active & coordinate_ok
        sums = kernels.pixel_sums(resident, values, lockstep.shifts)
        del values
        kernels.reduce_solve_update(resident, sums, lockstep, step_options)
        del sums

    # The FINAL criterion of D2.7 at the returned homography
    values, coordinate_ok = kernels.gather(resident, coefficients, lockstep.matrices)
    residual = xp.asarray(
        kernels.final_criterion(resident, values, lockstep.shifts),
        dtype=xp.float64,
    )
    del values
    failed = lockstep.failed | ~coordinate_ok | ~xp.isfinite(residual)
    h = parameters_from_matrices(xp, lockstep.matrices)
    failed = failed | ~xp.all(xp.isfinite(h), axis=1)

    packed = xp.empty((n, int(slots["width"])), dtype=xp.float64)
    packed[:, :N_PARAMETERS] = xp.where(failed[:, None], xp.nan, h)
    packed[:, slots["residual"]] = xp.where(failed, xp.nan, residual)
    packed[:, slots["iterations"]] = xp.where(
        failed, 0.0, lockstep.iterations.astype(xp.float64)
    )
    packed[:, slots["norm_dp"]] = xp.where(failed, xp.nan, lockstep.norm_dp)
    packed[:, slots["converged"]] = xp.where(lockstep.converged & ~failed, 1.0, 0.0)
    return packed


# --------------------- The numpy twin of "float64" ------------------- #


def _numpy_gather(resident, coefficients, matrices):
    """The numpy twin of ``gather`` (D21.14.3): the carried matrices
    applied to the float64 PC-centred subregion coordinates exactly as
    ``_fit_pattern``'s criterion does, then the numba bicubic kernel on
    each slot's float32 coefficients.  A slot with any non-finite
    warped coordinate is flagged and never evaluated (its values are
    NaN): the fold of a non-finite value is undefined."""
    matrices = np.asarray(matrices, dtype=np.float64)
    coefficients = np.asarray(coefficients)
    xi_x = resident.xi_x
    xi_y = resident.xi_y
    n_slots = int(matrices.shape[0])
    values = np.full((n_slots, resident.n_pixels), np.nan, dtype=np.float64)
    coordinate_ok = np.zeros(n_slots, dtype=bool)
    offset_x, offset_y = resident.offset
    with np.errstate(all="ignore"):
        for b in range(n_slots):
            m = matrices[b]
            scale = m[2, 0] * xi_x + m[2, 1] * xi_y + m[2, 2]
            warped_x = (m[0, 0] * xi_x + m[0, 1] * xi_y + m[0, 2]) / scale
            warped_y = (m[1, 0] * xi_x + m[1, 1] * xi_y + m[1, 2]) / scale
            x = np.ascontiguousarray(warped_x + offset_x)
            y = np.ascontiguousarray(warped_y + offset_y)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            coordinate_ok[b] = True
            _bicubic_evaluate(np.ascontiguousarray(coefficients[b]), x, y, values[b])
    return values, coordinate_ok


def _numpy_pixel_sums(resident, values, shifts):
    """The numpy twin of ``pixel_sums``: the :data:`N_SUMS` sums of the
    module docstring, each a fixed-order reduction along the pixel axis
    of a C-contiguous ``(B, n)`` array (never a BLAS product over the
    batch, D21.14.3)."""
    values = np.asarray(values, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    x = resident.xi_x
    y = resident.xi_y
    with np.errstate(all="ignore"):
        shifted = np.ascontiguousarray(values - shifts[:, None])
        a = resident.gradient_x * shifted
        b = resident.gradient_y * shifted
        p = a * x + b * y
        terms = (
            shifted,
            shifted * shifted,
            a * x,
            a * y,
            a,
            b * x,
            b * y,
            b,
            p * x,
            p * y,
        )
        sums = np.empty((values.shape[0], N_SUMS), dtype=np.float64)
        for k, term in enumerate(terms):
            sums[:, k] = np.add.reduce(np.ascontiguousarray(term), axis=1)
    return sums


def _numpy_final_criterion(resident, values, shifts):
    """The numpy twin of ``final_criterion``: the two-pass D2.7
    criterion ``sum((w * (ref - (vp - mean) / norm))**2)`` on the
    shifted values ``vp``, per pattern along the pixel axis; NaN where
    the centred norm is zero or not finite (the CPU's raise)."""
    values = np.asarray(values, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    n = float(resident.n_pixels)
    with np.errstate(all="ignore"):
        shifted = np.ascontiguousarray(values - shifts[:, None])
        mean = np.add.reduce(shifted, axis=1) / n
        centred = np.ascontiguousarray(shifted - mean[:, None])
        norm = np.sqrt(np.add.reduce(centred * centred, axis=1))
        residual = resident.reference - centred / norm[:, None]
        if resident.weights is not None:
            residual = residual * resident.weights
        criterion = np.add.reduce(np.ascontiguousarray(residual * residual), axis=1)
    bad = ~(np.isfinite(norm) & (norm > 0))
    return np.where(bad, np.nan, criterion)


def _numpy_reduce_solve_update(resident, sums, lockstep, options):
    """The numpy twin of ``reduce_solve_update``: :func:`solve_update`
    on the host constants (D21.14.3)."""
    solve_update(np, resident, np.asarray(sums, dtype=np.float64), lockstep, options)


def solve_update(xp, resident, sums, lockstep, options, *, precision="float64"):
    """Apply one IC-GN step from the per-pattern *sums* to the ACTIVE
    slots of *lockstep*, in place, in float64 (requirements D21.6).

    Per active slot, in the CPU's order (``_engine.py`` loop): the
    mean ``mp = S0 / n`` and centred norm ``sqrt(S2 - S0 * mp)`` (a
    zero or non-finite norm FAILS the slot); the gradient
    ``gscale * (A - (SG - mp * G) / norm)`` from the windowed algebra
    of D21.6.5; the forward and back substitution on the host upper
    Cholesky factor (a non-finite step fails); ``dp * step_scale``;
    the corner norm whose maximum PROPAGATES NaN; ``W <- W . W(dp)**-1``
    by the closed-form 3x3 inverse (a zero or non-finite determinant
    fails) and the ``W33`` renormalisation (a zero or non-finite
    ``W33`` fails); the iteration count; the shift update ``K <-
    round(K + mp)`` to the pixel precision; ``converged`` when
    ``norm_dp < min_step``; retirement when converged or at
    ``max_iterations``.  A failed slot gets ``failed`` set, ``active``
    cleared and a NaN ``norm_dp``; inactive slots never change.

    Every operation is elementwise over the batch with host scalars
    (no BLAS over the batch), so a slot's bits never depend on the
    other slots.
    """
    n = float(resident.n_pixels)
    constants = resident.constants_host
    upper = resident.cholesky_host
    corners = resident.corners_host
    gradient_scale = resident.gradient_scale
    active = lockstep.active
    with np.errstate(all="ignore"):
        mean = sums[:, 0] / n
        variance = sums[:, 1] - sums[:, 0] * mean
        norm = xp.sqrt(variance)
        bad_norm = ~(xp.isfinite(norm) & (norm > 0))
        projected = [
            sums[:, 2],
            sums[:, 3],
            sums[:, 4],
            sums[:, 5],
            sums[:, 6],
            sums[:, 7],
            -sums[:, 8],
            -sums[:, 9],
        ]
        rhs = []
        for k in range(N_PARAMETERS):
            centred = projected[k] - mean * constants[N_PARAMETERS + k]
            gradient = gradient_scale * (constants[k] - centred / norm)
            rhs.append(-gradient)
        # U^T z = rhs (forward), then U dp = z (back), H = U^T U
        z = []
        for i in range(N_PARAMETERS):
            acc = rhs[i]
            for j in range(i):
                acc = acc - upper[j, i] * z[j]
            z.append(acc / upper[i, i])
        dp = [None] * N_PARAMETERS
        for i in range(N_PARAMETERS - 1, -1, -1):
            acc = z[i]
            for j in range(i + 1, N_PARAMETERS):
                acc = acc - upper[i, j] * dp[j]
            dp[i] = acc / upper[i, i]
        step_scale = float(options["step_scale"])
        dp = [d * step_scale for d in dp]
        bad_step = ~xp.isfinite(dp[0])
        for d in dp[1:]:
            bad_step = bad_step | ~xp.isfinite(d)

        # The D2.5 corner norm, ``np.max`` propagating NaN
        distances = []
        for q in range(4):
            x = corners[q, 0]
            y = corners[q, 1]
            scale = dp[6] * x + dp[7] * y + 1.0
            x_warped = ((1 + dp[0]) * x + dp[1] * y + dp[2]) / scale
            y_warped = (dp[3] * x + (1 + dp[4]) * y + dp[5]) / scale
            distances.append(xp.hypot(x_warped - x, y_warped - y))
        norm_dp = xp.max(xp.stack(distances, axis=1), axis=1)

        # W(dp) and its closed-form inverse
        d00, d01, d02 = 1 + dp[0], dp[1], dp[2]
        d10, d11, d12 = dp[3], 1 + dp[4], dp[5]
        d20, d21, d22 = dp[6], dp[7], 1.0
        c00 = d11 * d22 - d12 * d21
        c01 = d12 * d20 - d10 * d22
        c02 = d10 * d21 - d11 * d20
        determinant = d00 * c00 + d01 * c01 + d02 * c02
        singular = ~(xp.isfinite(determinant) & (determinant != 0))
        inverse = [
            [c00, d02 * d21 - d01 * d22, d01 * d12 - d02 * d11],
            [c01, d00 * d22 - d02 * d20, d02 * d10 - d00 * d12],
            [c02, d01 * d20 - d00 * d21, d00 * d11 - d01 * d10],
        ]
        inverse = [[entry / determinant for entry in row] for row in inverse]
        m = lockstep.matrices
        product = xp.empty_like(m)
        for i in range(3):
            for j in range(3):
                product[:, i, j] = (
                    m[:, i, 0] * inverse[0][j]
                    + m[:, i, 1] * inverse[1][j]
                    + m[:, i, 2] * inverse[2][j]
                )
        w33 = product[:, 2, 2]
        bad_w33 = ~(xp.isfinite(w33) & (w33 != 0))
        product = product / w33[:, None, None]
        new_shifts = round_to_pixel_precision(xp, lockstep.shifts + mean, precision)

    failed_now = active & (bad_norm | bad_step | singular | bad_w33)
    ok = active & ~failed_now
    iterations = lockstep.iterations + ok.astype(lockstep.iterations.dtype)
    converged_now = ok & (norm_dp < float(options["min_step"]))
    retire = converged_now | (ok & (iterations >= int(options["max_iterations"])))
    lockstep.matrices = xp.where(ok[:, None, None], product, m)
    lockstep.shifts = xp.where(ok, new_shifts, lockstep.shifts)
    lockstep.norm_dp = xp.where(
        failed_now, np.nan, xp.where(ok, norm_dp, lockstep.norm_dp)
    )
    lockstep.iterations = iterations
    lockstep.converged = lockstep.converged | converged_now
    lockstep.failed = lockstep.failed | failed_now
    lockstep.active = active & ~failed_now & ~retire
