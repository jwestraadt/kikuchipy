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
backend of the HREBSD IC-GN engine: the seed seam, the launch layout
and the per-precision kernel interface.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Requirements D21.5 (the seed seam), D21.6 (the batched IC-GN
semantics), D21.7.3 (the launch layout) and D21.9.5 (the names)
govern.  This is the Stage E failing-tests SKELETON: the containers
are real (fields only) so that tests can build them, and every
function raises ``NotImplementedError`` until the implementation gate.
The argument lists beyond the frozen names are fixed by the
failing-tests commit and recorded in the docstring of
``tests/test_indexing/test_hrebsd_gpu.py``.

The orchestration and the seed stage are ONE code path, run under
numpy in the default suite and under cupy in the gated one; the
per-pixel and reduction kernels are written twice, as CUDA RawKernels
and as a numpy twin of the ``"float64"`` build only, behind the same
:class:`KernelNamespace` (D21.14.3).  CuPy is never imported at module
scope (D21.13), and ``_engine`` is never imported at module scope
(D21.9.5).

Call-time seams (frozen by the failing-tests commit, the test module's
docstring): the runner reaches :func:`build_resident`,
:func:`build_seed_state`, :func:`seed_spectra` and
:func:`seed_homographies` through this module's globals at call time,
and every lockstep iteration calls the :class:`KernelNamespace` entry
points as attributes of the session's namespace (never bound to locals
before the loop); any kernel fusion lives inside one entry point,
never across two.
"""

# The preprocessing sub-batch bound of D21.7.3: every batched FFT and
# seed matmul runs at P = min(SUB_BATCH_SIZE, B) slots, the tail
# sub-batch padded to P
SUB_BATCH_SIZE: int = 32

# The two seed precisions of D21.5, the default first
SEED_PRECISIONS: tuple[str, ...] = ("complex128", "complex64")

# The two device precisions of D21.4, the default first
DEVICE_PRECISIONS: tuple[str, ...] = ("mixed", "float64")

_NOT_IMPLEMENTED = "Stage E: not implemented yet"


class NumpyOutOfMemoryError(MemoryError):
    """The out-of-memory error of the numpy :class:`KernelNamespace`,
    the module-local stand-in for cupy's ``OutOfMemoryError`` that the
    default-suite twins raise (requirements D21.9.5)."""


class KernelNamespace:
    """The per-precision kernel interface of requirements D21.9.5.

    Implemented once as CUDA RawKernels and once as the numpy twin of
    the ``"float64"`` build (D21.14.3).  A container of fields: build
    one with :func:`make_kernel_namespace`.

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
        per-pattern sums over the pixels of ``values - shifts[:, None]``
        (the shifted-data form of D21.4) from which the ZMN, the
        criterion and the gradient follow, ``(B, n_sums)`` float64.
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
        homography, ``(B,)`` float64.
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
    are compiled here, without ``--use_fast_math`` (D21.4), each one
    CONSTRUCTED through the attribute ``cupy.RawKernel`` on every call
    of this function: no ``cupy.RawModule`` and no Python-level cache
    of kernel objects across calls (CuPy's own compile cache is
    fine), so the gated launch spy sees every launch.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    """

    def __init__(
        self, bounds, reference_spectrum, precision: str, upsample_factor: int
    ) -> None:
        self.bounds = tuple(int(i) for i in bounds)
        self.reference_spectrum = reference_spectrum
        self.precision = precision
        self.upsample_factor = int(upsample_factor)


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
        ``-1`` on padded slots.
    extras
        Dictionary of per-slot ``(P, ...)`` arrays reserved for later
        stages; empty in Stage E.  ``None`` gives a new empty
        dictionary.
    """

    def __init__(self, targets, coefficients, pattern_index, extras=None) -> None:
        self.targets = targets
        self.coefficients = coefficients
        self.pattern_index = pattern_index
        self.extras = {} if extras is None else extras


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
    (D21.5)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def seed_spectra(ctx: SeedContext, batch: SeedBatch, seed_state: SeedState):
    """Return ``target_spectra``, ``(P, sr, sc)`` complex at
    ``seed_state.precision``: ``fft2`` of each target's zero-mean
    unit-norm crop, computed ONCE per sub-batch (requirements D21.5,
    frozen signature)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def seed_homographies(
    ctx: SeedContext, batch: SeedBatch, target_spectra, seed_state: SeedState
):
    """Return ``h0``, ``(P, 8)`` float64 C-contiguous in ``ctx.xp``:
    the FULL starting homography of every slot, in the frame and
    layout of ``fit_pattern``'s ``h0`` (requirements D21.5, frozen
    signature).

    Stage E implements the translation-only phase cross-correlation
    seed of D5 behind this seam, row for row what ``initial_guess``
    returns.  A row which is not finite throughout marks a seed
    FAILURE for that pattern only; padded slots' rows are discarded.
    The pipeline calls this through the module global at call time.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


class ReferenceResident:
    """The per-reference device resident set of requirements D21.9.3
    and D21.10.2, built by :func:`build_resident`: the uploaded mirror
    of a ``ReferenceState`` (the reference vector, ``xi_x``, ``xi_y``,
    the window-weighted gradient columns, the weight plane under
    ``window``, the per-reference constants of D21.6.5, the host
    Cholesky factor, the corners and the frame origin) at the device
    precision.  Opaque to the tests; fields are implementation
    freedom."""


def build_resident(ctx: SeedContext, state) -> ReferenceResident:
    """Return the :class:`ReferenceResident` of one ``ReferenceState``
    in the namespace and at the precision of ``ctx.kernels``; an
    upload is a byte copy and never changes a value (D21.9.3)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    never of the batch size (requirements D21.7.3)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
