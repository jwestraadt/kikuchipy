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

"""The optional CuPy GPU backend of the HREBSD IC-GN engine: the
availability gate, the per-call device session, the VRAM model and
batch chooser, and the device runner at the ``_run_chunks`` contract.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Requirements D21 (``specs/2026-09-07-hrebsd-dic/requirements.md``)
govern: the gate of D21.2, the session, the dask topology, the
grain-pure batches and the residency bound of D21.9, and the VRAM
model, the default batch size and the out-of-memory loop of D21.10.
The argument lists beyond the frozen names are fixed by the
failing-tests commit and recorded in the docstring of
``tests/test_indexing/test_hrebsd_gpu.py``.

CuPy is an optional dependency and is NEVER imported at module scope
here (D21.13): every call site imports it inside the function, after
the gate.  This module never imports ``_engine`` at module scope
(D21.9.5), so no import cycle exists: the runner receives the row-slot
constants and the reference states as arguments.

Call-time seams (frozen by the failing-tests commit, the test module's
docstring): the runner calls :func:`_make_session`,
:func:`_free_device_bytes` and :func:`_default_batch_size` through this
module's globals, and the ``_batched`` seam and residency functions
through the ``_batched`` module object, at call time; ``_engine``
calls :func:`_run_chunks_gpu` as ``_gpu._run_chunks_gpu``.  The gate
probe constructs its kernel through the attribute ``cupy.RawKernel``.

The three names below are imported, unchanged, from the spherical
backend (D21.2), so that "one GPU consumer per process" holds across
both backends and a develop-side rename fails loudly at this import.
"""

import gc
import sys

import dask.array as da
from dask.diagnostics.progress import ProgressBar
import numpy as np

from kikuchipy.indexing._hrebsd import _batched
from kikuchipy.indexing._hrebsd._interpolation import spline_coefficients
from kikuchipy.indexing._hrebsd._preprocessing import preprocess
from kikuchipy.indexing._spherical._gpu import (
    _CUPY_MINIMUM_VERSION as _CUPY_MINIMUM_VERSION,
)
from kikuchipy.indexing._spherical._gpu import _DEVICE_LOCK as _DEVICE_LOCK
from kikuchipy.indexing._spherical._gpu import (
    _add_nvidia_dll_directories as _add_nvidia_dll_directories,
)

# The residency bound of D21.9.3 (RECORDED DEFAULT): at most this many
# references' device residents at any time, the least recently used
# evicted before the next grain's upload
R_MAX: int = 2

# The default batch sizes of D21.10.3, largest first: the chooser
# returns the largest whose whole model fits half of free VRAM
_BATCH_SIZES: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)

# The fraction of free VRAM the default batch may claim (D21.10.3)
_FREE_VRAM_FRACTION: float = 0.5

# The cached gate verdict of D21.2: ``None`` until the first
# :func:`_verify_gpu_or_raise` call, then ``True`` or the exception
# instance to re-raise as a fresh copy.  HREBSD's own cache, never the
# spherical one.  Tests reset it by monkeypatching it back to ``None``
_gate_result = None

# The frozen message prefix of every gate stage (D21.2)
_GATE_PREFIX = "EBSD.hrebsd_dic with backend='gpu' requires"

# The stage messages of the gate, each naming its remedy (D21.2)
_GATE_IMPORT_MESSAGE = (
    f"{_GATE_PREFIX} that 'cupy' is installed, which is an optional "
    "dependency of kikuchipy; install it with e.g. 'pip install "
    "cupy-cuda12x' matching your CUDA version (cupy-cuda11x, cupy-cuda13x "
    "and ROCm wheels exist), or use backend='cpu'"
)
_GATE_VERSION_MESSAGE = (
    f"{_GATE_PREFIX} cupy >= {_CUPY_MINIMUM_VERSION}, but version {{version}} "
    "is installed; upgrade with e.g. 'pip install --upgrade cupy-cuda12x' "
    "matching your CUDA version, or use backend='cpu'"
)
_GATE_DEVICE_MESSAGE = (
    f"{_GATE_PREFIX} a CUDA device, but cupy found none ({{detail}}); check "
    "that an NVIDIA GPU is present and that the NVIDIA driver is installed "
    "and current, or use backend='cpu'"
)
_GATE_LIBRARY_MESSAGE = (
    f"{_GATE_PREFIX} that cupy can load cuFFT, cuBLAS and NVRTC, but a probe "
    "of a complex128 and a complex64 FFT, a matrix product and a compiled "
    "kernel failed ({detail}); 'pip install nvidia-cufft-cu12 "
    "nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 "
    "nvidia-nvjitlink-cu12', or install the full CUDA Toolkit and put its "
    "'bin' directory on PATH (on Windows kikuchipy registers the nvidia "
    "wheel DLL directories itself), or use backend='cpu'"
)

# The trivial kernel of the stage-(c) NVRTC probe (D21.2)
_PROBE_KERNEL_NAME = "kikuchipy_hrebsd_gate_probe"
_PROBE_KERNEL_SOURCE = r"""
extern "C" __global__ void kikuchipy_hrebsd_gate_probe(double* out)
{
    out[threadIdx.x] = 2.0 * out[threadIdx.x] + 1.0;
}
"""

# The free device memory the NUMPY namespace reports, bytes: a fixed
# module value, so that the default-suite twin chooses a batch size
# without a device (D21.10.3)
_NUMPY_FREE_BYTES: int = 8 * 2**30

# Bytes of the VRAM model of D21.10.2, per pattern pixel unless stated.
# Every term is an upper bound of what the session allocates, so the
# model never under-states a batch: the calibration against measured
# pool high-water marks is the implementation-gate measurement
# (validation V9(o))
# g, per slot for the whole batch lifetime: the f32 coefficient plane
# and the warped values one ``gather`` returns at the pixel precision
_G_COEFFICIENT_BYTES = 4
# p, per sub-batch slot: the stored-dtype upload counted at its widest
# (f64), the f64 targets, the complex128 band-pass spectrum and its
# inverse, the f64 spline prefilter and its f32 copy, then the seed
# crop's spectrum and cross-power spectrum at the seed precision
_P_UPLOAD_BYTES = 8
_P_TARGET_BYTES = 8
_P_BAND_PASS_BYTES = 32
_P_SPLINE_BYTES = 16
_P_SEED_SPECTRA = 2
# r, per reference: five subregion planes at the pixel precision (the
# reference vector, ``xi_x``, ``xi_y`` and the two gradient columns),
# the weight plane counted always (conservatively), the two int32 pixel
# coordinates and the int64 subregion index, the f64 band-pass transfer
# function counted always, then three reference-crop planes at the seed
# precision: the held seed spectrum, and at its build the normalised
# crop's complex copy and the cuFFT work area (CALIBRATED 2026-10-07 at
# the implementation gate against pool high-water marks, validation.md
# V9 ledger 103: one seed plane under-stated the measured build peak)
_R_PIXEL_PLANES = 6
_R_COORDINATE_BYTES = 16
_R_TRANSFER_BYTES = 8
_R_SEED_SPECTRA = 3
# The fixed per-slot and per-reference overheads: the carried matrix,
# the per-pattern sums, the block partials and flags; the Cholesky
# factor, the corners and the per-reference constants
_SLOT_OVERHEAD_BYTES = 64 * 1024
_REFERENCE_OVERHEAD_BYTES = 64 * 1024

_PIXEL_BYTES = {"mixed": 4, "float64": 8}
_COMPLEX_BYTES = {"complex128": 16, "complex64": 8}

# ------------------------- Availability gate ------------------------ #


def _import_cupy():
    """Return the imported :mod:`cupy`, stage (a) of the gate.

    Raises
    ------
    ImportError
        When cupy cannot be imported, or when the major version read
        from ``cupy.__version__`` is below :data:`_CUPY_MINIMUM_VERSION`.
    """
    try:
        import cupy
    except Exception as error:
        raise ImportError(_GATE_IMPORT_MESSAGE) from error
    version = str(getattr(cupy, "__version__", "0"))
    major_text = version.split(".")[0]
    # a malformed version reads as 0 and fails the floor, actionable
    major = int(major_text) if major_text.isdigit() else 0
    if major < _CUPY_MINIMUM_VERSION:
        raise ImportError(_GATE_VERSION_MESSAGE.format(version=version))
    return cupy


def _device_count(cupy) -> int:
    """Return the CUDA device count, stage (b) of the gate.

    Raises
    ------
    RuntimeError
        When ``cupy.cuda.runtime.getDeviceCount()`` returns zero or
        raises ``CUDARuntimeError`` (no device, no driver).
    """
    runtime = cupy.cuda.runtime
    cuda_runtime_error = getattr(runtime, "CUDARuntimeError", Exception)
    try:
        count = int(runtime.getDeviceCount())
    except cuda_runtime_error as error:
        raise RuntimeError(_GATE_DEVICE_MESSAGE.format(detail=error)) from error
    if count < 1:
        raise RuntimeError(_GATE_DEVICE_MESSAGE.format(detail=f"device count {count}"))
    return count


def _probe_libraries(cupy) -> None:
    """Probe exactly the libraries the HREBSD device path calls, stage
    (c) of the gate: one complex128 and one complex64 ``cupy.fft``
    transform (cuFFT: the band-pass and the seed), one tiny ``matmul``
    (cuBLAS: the seed's upsampled DFT) and the compile and launch of
    one trivial kernel built through ``cupy.RawKernel`` (NVRTC: the
    fused IC-GN kernels).  Never FFT-only (D21.2).

    Raises
    ------
    RuntimeError
        When any probe fails.
    """
    try:
        cupy.fft.fft(cupy.ones(8, dtype=cupy.complex128))
        cupy.fft.fft(cupy.ones(8, dtype=cupy.complex64))
        ones = cupy.ones((2, 2), dtype=cupy.complex128)
        cupy.matmul(ones, ones)
        kernel = cupy.RawKernel(_PROBE_KERNEL_SOURCE, _PROBE_KERNEL_NAME)
        out = cupy.ones(1, dtype=cupy.float64)
        kernel((1,), (1,), (out,))
        synchronize = getattr(cupy.cuda.runtime, "deviceSynchronize", None)
        if synchronize is not None:
            synchronize()
    except Exception as error:
        raise RuntimeError(_GATE_LIBRARY_MESSAGE.format(detail=error)) from error


def _verify_gpu_or_raise() -> None:
    """Verify that the CuPy GPU backend of ``EBSD.hrebsd_dic`` is
    usable here, or raise with an actionable message.

    Requirements D21.2: the three-stage gate in the frozen Phase 12
    order -- (a) ``import cupy`` and the major version floor
    :data:`_CUPY_MINIMUM_VERSION` read from ``cupy.__version__``; (b)
    ``cupy.cuda.runtime.getDeviceCount() >= 1``; then the Windows DLL
    shim :func:`_add_nvidia_dll_directories`; then (c) a probe of one
    complex128 and one complex64 ``cupy.fft`` transform, one ``matmul``
    and the compile and launch of one trivial ``RawKernel``.  Every
    message starts with "EBSD.hrebsd_dic with backend='gpu' requires"
    and names its remedy.  The verdict is cached in
    :data:`_gate_result`; a cached failure re-raises a fresh copy
    chained to the cached instance.

    Imported into ``_engine``'s namespace, where tests patch it.

    Raises
    ------
    ImportError
        At stage (a).
    RuntimeError
        At stages (b) and (c).
    """
    global _gate_result
    if _gate_result is True:
        return
    if _gate_result is not None:
        # A FRESH copy of the cached failure where the type allows it,
        # so its traceback never grows across raises; the cached
        # instance rides along as the cause (the Phase 12 rule)
        cached = _gate_result
        try:
            fresh = type(cached)(*cached.args)
        except Exception:
            raise cached
        raise fresh from cached
    try:
        cupy = _import_cupy()
        _device_count(cupy)
        _add_nvidia_dll_directories()
        _probe_libraries(cupy)
    except Exception as error:
        _gate_result = error
        raise
    _gate_result = True


# ------------------------------ Session ----------------------------- #


class _GpuSession:
    """The per-call device session of requirements D21.9.1.

    Built by :func:`_make_session` after the gate and closed in a
    ``finally``.  It holds the array namespace, the
    :class:`~kikuchipy.indexing._hrebsd._batched.KernelNamespace`, the
    batch size B, the preprocessing sub-batch size P and the device
    residents (at most :data:`R_MAX` references).  Nothing
    device-resident lives on a public object.

    Attributes fixed at the failing-tests gate: ``xp``, ``fft``,
    ``kernels``, ``batch_size``, ``sub_batch_size``,
    ``device_precision``, ``seed_precision`` and ``lock`` (the shared
    :data:`_DEVICE_LOCK`).  Beside them ``namespace`` (the string) and
    ``context``, the one
    :class:`~kikuchipy.indexing._hrebsd._batched.SeedContext` of the
    run, whose ``kernels`` IS :attr:`kernels`.
    """

    def __init__(
        self,
        namespace: str,
        batch_size: int,
        *,
        device_precision: str,
        seed_precision: str,
    ) -> None:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"the device batch size must be >= 1, not {batch_size}")
        if namespace == "cupy":
            # Re-verified (cached per process): a session may be built
            # in a process whose gate never ran, and the gate runs the
            # Windows DLL shim cuFFT needs
            _verify_gpu_or_raise()
            import cupy

            xp = cupy
        elif namespace == "numpy":
            xp = np
        else:
            raise ValueError(f"namespace must be 'cupy' or 'numpy', not {namespace!r}")
        self.namespace = namespace
        self.xp = xp
        self.fft = xp.fft
        self.lock = _DEVICE_LOCK
        self.batch_size = batch_size
        self.sub_batch_size = min(_batched.SUB_BATCH_SIZE, batch_size)
        self.device_precision = device_precision
        self.seed_precision = seed_precision
        self.kernels = _batched.make_kernel_namespace(namespace, device_precision)
        self.context = _batched.SeedContext(xp, self.fft, self.kernels)
        # grain index -> (resident, seed state), least
        # recently used first (D21.9.3)
        self._residents = {}
        self._closed = False

    def residents(self, grain: int, state, upsample_factor: int) -> tuple:
        """Return ``(resident, seed_state)`` of the grain
        *grain*, uploading them on first use (requirements D21.9.3).

        The least recently used reference is evicted BEFORE the next
        upload, so at most ``R_MAX - 1`` references are resident when
        ``_batched.build_resident`` and ``_batched.build_seed_state``
        are entered, and at most :data:`R_MAX` afterwards.  Both are
        reached through the ``_batched`` module object at call time.
        """
        grain = int(grain)
        entry = self._residents.pop(grain, None)
        if entry is not None:
            # re-inserted last: a dict keeps insertion order, so its
            # first key is always the least recently used one
            self._residents[grain] = entry
            return entry
        while len(self._residents) >= R_MAX:
            del self._residents[next(iter(self._residents))]
        resident = _batched.build_resident(self.context, state)
        seed_state = _batched.build_seed_state(
            self.context,
            state,
            precision=self.seed_precision,
            upsample_factor=int(upsample_factor),
        )
        entry = (resident, seed_state)
        self._residents[grain] = entry
        return entry

    def close(self) -> None:
        """Free the residents and the default and pinned pools.
        Idempotent (D21.9.1)."""
        self._residents.clear()
        if self._closed:
            return
        self._closed = True
        if self.namespace == "cupy":
            import cupy

            gc.collect()
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()

    def __dask_tokenize__(self) -> tuple:
        """Return ``("kikuchipy-hrebsd-gpu-session", id(self))``, so no
        device array enters a dask token (D21.9.1)."""
        return ("kikuchipy-hrebsd-gpu-session", id(self))


def _make_session(
    namespace: str,
    batch_size: int,
    *,
    device_precision: str,
    seed_precision: str,
) -> _GpuSession:
    """Return a new :class:`_GpuSession`, the module-global factory
    the runner calls AT CALL TIME (requirements D21.9.5).

    Parameters
    ----------
    namespace
        ``"cupy"`` (what the runner passes) or ``"numpy"`` (the
        default-suite twin, float64 only, D21.14.3).  The module is
        imported inside this function, never at module scope.
    batch_size
        The device batch size B >= 1.
    device_precision
        ``"mixed"`` or ``"float64"`` (D21.4).
    seed_precision
        ``"complex128"`` or ``"complex64"`` (D21.5).

    Raises
    ------
    Exception
        The namespace's out-of-memory error at session build, which
        the runner's halving loop catches (D21.10.4).
    """
    return _GpuSession(
        namespace,
        batch_size,
        device_precision=device_precision,
        seed_precision=seed_precision,
    )


def _free_device_bytes(namespace: str) -> int:
    """Return the free device memory in bytes, the ONE device query
    of the batch choice (D21.10.3: queried by the caller only, never
    inside the model).

    ``cupy.cuda.runtime.memGetInfo()[0]`` under ``"cupy"``; under
    ``"numpy"`` a fixed module value.  Fixed at the failing-tests
    gate so that the default-suite tests can fake free VRAM.
    """
    if namespace == "numpy":
        return int(_NUMPY_FREE_BYTES)
    import cupy

    return int(cupy.cuda.runtime.memGetInfo()[0])


def _device_description(namespace: str) -> tuple[str, int | None]:
    """Return ``(name, total_bytes)`` of the current device for the
    information message, ``("unknown device", None)`` when it cannot
    be read (D21.10.5).  Best effort, never raising."""
    if namespace != "cupy":
        return f"{namespace} namespace", None
    try:
        import cupy

        device = cupy.cuda.Device()
        properties = cupy.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        total = int(cupy.cuda.runtime.memGetInfo()[1])
        return str(name), total
    except Exception:
        return "unknown device", None


# ------------------------- VRAM model (D21.10) ---------------------- #


def _vram_model_terms(
    n_pixels: int, device_precision: str, seed_precision: str
) -> tuple[int, int, int]:
    """Return the three terms ``(g, p, r)`` of the pure-math VRAM model
    of requirements D21.10.2, bytes, for *n_pixels* pattern pixels.

    ``g`` is the per-slot set held for the whole batch lifetime, ``p``
    the preprocessing and seed transient per sub-batch slot and ``r``
    the per-reference resident set (the window's weight plane and
    constants counted always, conservatively).  No device query.

    The subregion and its bounding box are bounded from above by the
    pattern, so every per-pixel term is counted over all *n_pixels*.
    """
    n = int(n_pixels)
    pixel = _PIXEL_BYTES[device_precision]
    complex_bytes = _COMPLEX_BYTES[seed_precision]
    g = n * (_G_COEFFICIENT_BYTES + pixel) + _SLOT_OVERHEAD_BYTES
    p = n * (
        _P_UPLOAD_BYTES
        + _P_TARGET_BYTES
        + _P_BAND_PASS_BYTES
        + _P_SPLINE_BYTES
        + _P_SEED_SPECTRA * complex_bytes
    )
    r = (
        n
        * (
            _R_PIXEL_PLANES * pixel
            + _R_COORDINATE_BYTES
            + _R_TRANSFER_BYTES
            + _R_SEED_SPECTRA * complex_bytes
        )
        + _REFERENCE_OVERHEAD_BYTES
    )
    return int(g), int(p), int(r)


def _vram_model_bytes(
    batch_size: int, n_pixels: int, device_precision: str, seed_precision: str
) -> int:
    """Return ``B * g + P * p + R_MAX * r`` bytes with
    ``P = min(32, B)`` (requirements D21.10.2), the terms of
    :func:`_vram_model_terms`.  Pure math, no device query."""
    batch_size = int(batch_size)
    g, p, r = _vram_model_terms(n_pixels, device_precision, seed_precision)
    sub_batch_size = min(_batched.SUB_BATCH_SIZE, batch_size)
    return int(batch_size * g + sub_batch_size * p + R_MAX * r)


def _default_batch_size(
    free_bytes: int, n_pixels: int, device_precision: str, seed_precision: str
) -> int:
    """Return the largest B in :data:`_BATCH_SIZES` whose WHOLE model
    fits ``0.5 * free_bytes`` (requirements D21.10.3), else 1."""
    budget = _FREE_VRAM_FRACTION * int(free_bytes)
    for batch_size in _BATCH_SIZES:
        model = _vram_model_bytes(
            batch_size, n_pixels, device_precision, seed_precision
        )
        if model <= budget:
            return int(batch_size)
    return 1


def _info_lines(
    batch_size: int,
    signal_shape: tuple[int, int],
    device_precision: str,
    seed_precision: str,
    free_bytes: int,
    namespace: str = "cupy",
) -> list[str]:
    """Return the device block of the information message of
    requirements D21.10.5: the device name, free and total VRAM, B, P,
    the model in MB per term, and a warning line when the model
    exceeds free VRAM.  Called by ``_engine`` with the free VRAM it
    queried through :func:`_free_device_bytes`."""
    n_pixels = int(signal_shape[0]) * int(signal_shape[1])
    batch_size = int(batch_size)
    sub_batch_size = min(_batched.SUB_BATCH_SIZE, batch_size)
    g, p, r = _vram_model_terms(n_pixels, device_precision, seed_precision)
    model = _vram_model_bytes(batch_size, n_pixels, device_precision, seed_precision)
    name, total = _device_description(namespace)
    mb = 1024**2
    total_text = "unknown" if total is None else f"{total / mb:.0f} MB"
    lines = [
        f"  Device: {name}, {int(free_bytes) / mb:.0f} MB free VRAM of "
        f"{total_text} total",
        f"  Device batch: B = {batch_size} pattern(s), preprocessing sub-batch "
        f"P = {sub_batch_size}; device precision {device_precision!r}, seed "
        f"precision {seed_precision!r}",
        f"  VRAM model: {model / mb:.1f} MB = B * {g / mb:.2f} MB + P * "
        f"{p / mb:.2f} MB + {R_MAX} * {r / mb:.2f} MB",
    ]
    if model > int(free_bytes):
        lines.append(
            f"  Warning: the VRAM model ({model / mb:.1f} MB) exceeds the free "
            f"VRAM ({int(free_bytes) / mb:.1f} MB); the run halves B on an "
            "out-of-memory error, or pass a smaller chunksize"
        )
    return lines


# ----------------------------- Runner -------------------------------- #


def _out_of_memory_types(session=None) -> tuple:
    """Return the exception types the out-of-memory loop of D21.10.4
    catches: the numpy namespace's own, cupy's ``OutOfMemoryError``
    when cupy is imported, and the session's
    ``kernels.out_of_memory_error``."""
    types = [_batched.NumpyOutOfMemoryError]
    cupy = sys.modules.get("cupy")
    memory = getattr(getattr(cupy, "cuda", None), "memory", None)
    error = getattr(memory, "OutOfMemoryError", None)
    if isinstance(error, type) and issubclass(error, BaseException):
        types.append(error)
    if session is not None:
        error = getattr(getattr(session, "kernels", None), "out_of_memory_error", None)
        if isinstance(error, type) and issubclass(error, BaseException):
            types.append(error)
    return tuple(types)


def _free_pools() -> None:
    """Collect garbage and free cupy's default pool when cupy is
    imported, before a halved retry (D21.10.4)."""
    gc.collect()
    cupy = sys.modules.get("cupy")
    try:
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _out_of_memory_error(
    signal_shape: tuple[int, int],
    device_precision: str,
    seed_precision: str,
) -> MemoryError:
    """Return the actionable error of a run whose halving bottomed out
    at a device batch of one (D21.10.4): the pattern shape, the model
    MB per term, the free VRAM and the remedies."""
    n_pixels = int(signal_shape[0]) * int(signal_shape[1])
    g, p, r = _vram_model_terms(n_pixels, device_precision, seed_precision)
    model = _vram_model_bytes(1, n_pixels, device_precision, seed_precision)
    try:
        free_text = f"{_free_device_bytes('cupy') / 1024**2:.0f} MB"
    except Exception:
        free_text = "an unknown amount"
    mb = 1024**2
    return MemoryError(
        "EBSD.hrebsd_dic with backend='gpu' ran out of device memory even at a "
        f"device batch size of 1 for patterns of shape {tuple(signal_shape)}: "
        f"the VRAM model needs {model / mb:.1f} MB (per slot {g / mb:.2f} MB, "
        f"preprocessing {p / mb:.2f} MB, per reference {r / mb:.2f} MB times "
        f"{R_MAX}) against {free_text} of free VRAM; free device memory, pass "
        "a smaller chunksize, or use the CPU path (backend='cpu')"
    )


def _run_chunks_gpu(
    patterns,
    fit_indices,
    state_of_point,
    states: list,
    chunksize: int | None,
    *,
    progressbar: bool,
    options: dict,
    device_precision: str,
    seed_precision: str,
    row_slots: dict,
):
    """Return the packed results of every fitted point, in fit order,
    computed on the device: the ``_run_chunks`` contract of
    requirements D21.9.

    The out-of-memory loop of D21.10.4 lives here, in two windows: an
    out-of-memory error at session build halves B and retries; one
    inside the compute disposes the session, frees the pools, rebuilds
    session and graph at B / 2 and re-runs the whole map, so a
    completed run's rows come wholly from its final B.  At B = 1 it
    raises :class:`MemoryError`.  Any other error propagates and fails
    the run (D21.9.4); the session is closed in a ``finally`` either
    way.

    Parameters
    ----------
    patterns
        Patterns of shape ``(n, nrows, ncols)``, eager or lazy, in
        their stored data type.
    fit_indices
        Flat map indices to correlate, in grain order.
    state_of_point
        Index into *states* of every entry of *fit_indices*.
    states
        The per-reference ``ReferenceState`` objects, built on the
        host exactly as for the CPU path.
    chunksize
        The device batch size B (D21.10.1), or ``None`` for
        :func:`_default_batch_size` of :func:`_free_device_bytes`.
    progressbar
        Whether to show a progress bar.
    options
        The engine's fit options: ``upsample_factor``,
        ``max_iterations``, ``min_step`` and ``step_scale``.
    device_precision
        ``"mixed"`` or ``"float64"`` (D21.4).
    seed_precision
        ``"complex128"`` or ``"complex64"`` (D21.5).
    row_slots
        The packed-row layout, passed by ``_engine`` so that this
        module never imports it: ``{"width": 12, "residual": 8,
        "iterations": 9, "norm_dp": 10, "converged": 11}``, the
        homography in columns ``[0, 8)``.

    Returns
    -------
    packed
        ``(fit_indices.size, width)`` 64-bit float array.
    """
    fit_indices = np.ascontiguousarray(np.asarray(fit_indices, dtype=np.int64))
    state_of_point = np.ascontiguousarray(np.asarray(state_of_point, dtype=np.int64))
    signal_shape = (int(patterns.shape[1]), int(patterns.shape[2]))
    n_pixels = signal_shape[0] * signal_shape[1]
    width = int(row_slots["width"])
    if fit_indices.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if chunksize is None:
        batch_size = _default_batch_size(
            _free_device_bytes("cupy"), n_pixels, device_precision, seed_precision
        )
    else:
        batch_size = int(chunksize)
        if batch_size < 1:
            raise ValueError(
                f"chunksize must be >= 1 under backend='gpu', not {chunksize}"
            )

    while True:
        # Window (a): out of memory at session build
        try:
            session = _make_session(
                "cupy",
                batch_size,
                device_precision=device_precision,
                seed_precision=seed_precision,
            )
        except _out_of_memory_types() as error:
            # drop the frames pinning the failed build before the pool
            # is freed (the Phase 12 review fix)
            error.__traceback__ = None
            _free_pools()
            if batch_size <= 1:
                raise _out_of_memory_error(
                    signal_shape, device_precision, seed_precision
                ) from error
            batch_size //= 2
            continue
        # Window (b): out of memory inside the compute
        out_of_memory = _out_of_memory_types(session)
        try:
            try:
                return _compute_map(
                    session,
                    patterns,
                    fit_indices,
                    state_of_point,
                    states,
                    progressbar=progressbar,
                    options=options,
                    row_slots=row_slots,
                )
            finally:
                # Per call (D21.9.1): disposed on success, on an
                # out-of-memory before the rebuild and on any
                # propagating device error
                session.close()
        except out_of_memory as error:
            error.__traceback__ = None
            del session
            _free_pools()
            if batch_size <= 1:
                raise _out_of_memory_error(
                    signal_shape, device_precision, seed_precision
                ) from error
            batch_size //= 2


def _batch_chunks(state_of_point: np.ndarray, batch_size: int) -> tuple[int, ...]:
    """Return the chunk sizes of the grain-pure device batches of
    requirements D21.9.3: every grain's run of the grain-ordered fit
    list split into batches of *batch_size*, its last one short (and
    padded to B on the device, never in the graph)."""
    sizes = []
    n = int(state_of_point.size)
    start = 0
    while start < n:
        grain = state_of_point[start]
        stop = start
        while stop < n and state_of_point[stop] == grain:
            stop += 1
        for s in range(start, stop, batch_size):
            sizes.append(min(batch_size, stop - s))
        start = stop
    return tuple(sizes)


def _compute_map(
    session: _GpuSession,
    patterns,
    fit_indices: np.ndarray,
    state_of_point: np.ndarray,
    states: list,
    *,
    progressbar: bool,
    options: dict,
    row_slots: dict,
) -> np.ndarray:
    """Return the packed rows of the whole map at the session's B: the
    D21.9.2 dask topology, :func:`dask.array.blockwise` over the
    grain-ordered gather with chunk == device batch, computed on the
    threaded scheduler, one device section per chunk under the shared
    device lock.

    The device sections run in BATCH ORDER whatever order the threaded
    scheduler finishes the chunks' reads in: a chunk task hands its
    host block to :class:`_OrderedDevice` and runs, under the lock,
    every batch whose turn has come, never waiting for its own turn
    (so no worker thread ever blocks on another one's read and the
    scheduler cannot deadlock).  Batches are grain ordered, so every
    grain is uploaded once (D21.9.3) and a run's device sequence is
    reproducible.  The chunk tasks return placeholders; the rows come
    from the ordered device sections.
    """
    chunks = _batch_chunks(state_of_point, session.batch_size)
    if isinstance(patterns, da.Array):
        patterns_da = patterns
    else:
        patterns_da = da.from_array(
            np.asarray(patterns), chunks=(session.batch_size, -1, -1)
        )
    # The lazy-stays-lazy gather of D21.11: the stored patterns of one
    # batch are the only host data a chunk reads
    ordered = patterns_da[fit_indices].rechunk((chunks, -1, -1))
    state_da = da.from_array(state_of_point, chunks=(chunks,))
    index_da = da.from_array(fit_indices, chunks=(chunks,))
    batch_number = np.repeat(np.arange(len(chunks), dtype=np.int64), chunks)
    batch_da = da.from_array(batch_number, chunks=(chunks,))
    device = _OrderedDevice(session, tuple(states), options, row_slots)
    results = da.blockwise(
        _device_chunk,
        "ij",
        ordered,
        "ikl",
        state_da,
        "i",
        index_da,
        "i",
        batch_da,
        "i",
        device=device,
        new_axes={"j": int(row_slots["width"])},
        dtype=np.float64,
        concatenate=True,
    )
    try:
        if progressbar:
            with ProgressBar():
                results.compute(scheduler="threads")
        else:
            results.compute(scheduler="threads")
        return device.packed(len(chunks))
    finally:
        device.clear()


class _OrderedDevice:
    """The batch-ordered device sections of one map computation
    (requirements D21.9.2, D21.9.3), shared by its chunk tasks.

    :meth:`submit` stores a chunk's host block under its batch number
    and then runs, under the session's device lock, every stored batch
    whose turn has come, in batch order.  The rows of every batch are
    kept on the host until :meth:`packed` concatenates them.  Never a
    dask token: :meth:`__dask_tokenize__` names the object, not its
    contents (D21.9.1).

    CuPy caches cuFFT plans per THREAD, and the device sections run in
    dask's long-lived worker threads, so the plans of a run (their work
    areas held in the memory pool) would outlive the session in those
    threads' caches.  Under cupy every thread's plan cache that served
    a device section is recorded, and :meth:`clear` empties each one
    once the compute has returned, so the session's
    ``free_all_blocks`` returns the work areas too (the order-dependent
    pool residue the V9(o) calibration found, validation.md ledger
    103).
    """

    def __init__(self, session: _GpuSession, states: tuple, options, row_slots):
        self.session = session
        self.states = states
        self.options = options
        self.row_slots = row_slots
        self.pending = {}
        self.rows = {}
        self.next_batch = 0
        self.plan_caches = {}
        self.closed = False

    def submit(self, number: int, patterns_block, state_index_block, index_block):
        """Store one chunk's host block, then run every batch whose
        turn has come, in order, under the device lock."""
        session = self.session
        with session.lock:
            if self.closed:
                # the compute already failed and returned: a chunk task
                # still in flight in a worker thread must not touch the
                # session being disposed
                return
            self.pending[int(number)] = (
                patterns_block,
                state_index_block,
                index_block,
            )
            if session.namespace == "cupy":
                cache = session.xp.fft.config.get_plan_cache()
                self.plan_caches[id(cache)] = cache
            while self.next_batch in self.pending:
                block = self.pending.pop(self.next_batch)
                self.rows[self.next_batch] = _device_batch(
                    session, self.states, self.options, self.row_slots, *block
                )
                self.next_batch += 1

    def packed(self, n_batches: int) -> np.ndarray:
        """Return the rows of every batch in batch order, which is the
        fit order."""
        if self.next_batch != n_batches:  # pragma: no cover
            raise RuntimeError(
                f"only {self.next_batch} of {n_batches} device batches ran"
            )
        parts = [self.rows[k] for k in range(n_batches)]
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)

    def clear(self) -> None:
        """Drop every stored block and row, and empty the cuFFT plan
        cache of every thread that ran a device section."""
        # Under the device lock: after a failed compute, chunk tasks
        # may still be running in dask's worker threads; any device
        # section in flight finishes first, and later ones return at
        # once (:attr:`closed`)
        with self.session.lock:
            self.closed = True
            self.pending.clear()
            self.rows.clear()
            caches = list(self.plan_caches.values())
            self.plan_caches.clear()
            for cache in caches:
                try:
                    cache.clear()
                except Exception:  # pragma: no cover
                    pass

    def __dask_tokenize__(self) -> tuple:
        return ("kikuchipy-hrebsd-gpu-ordered-device", id(self))


def _device_chunk(
    patterns_block: np.ndarray,
    state_index_block: np.ndarray,
    index_block: np.ndarray,
    batch_block: np.ndarray,
    *,
    device: _OrderedDevice,
) -> np.ndarray:
    """Hand one grain-pure chunk of ``n <= B`` patterns to the ordered
    device sections and return an ``(n, width)`` placeholder; the
    chunk's rows are collected by :meth:`_OrderedDevice.packed`."""
    n = int(patterns_block.shape[0])
    if n:
        device.submit(
            int(batch_block[0]),
            np.asarray(patterns_block),
            np.asarray(state_index_block),
            np.asarray(index_block),
        )
    return np.zeros((n, int(device.row_slots["width"])), dtype=np.float64)


def _device_batch(
    session: _GpuSession,
    states: tuple,
    options: dict,
    row_slots: dict,
    patterns_block: np.ndarray,
    state_index_block: np.ndarray,
    index_block: np.ndarray,
) -> np.ndarray:
    """Return the ``(n, width)`` host rows of one grain-pure batch of
    ``n <= B`` patterns: ONE device section, called with the device lock
    held and outside every per-pattern scope (D21.9.2, D21.9.4)."""
    n = int(patterns_block.shape[0])
    grain = int(state_index_block[0])
    if np.any(state_index_block != grain):  # pragma: no cover
        raise RuntimeError("a device batch straddles grains (D21.9.3)")
    batch_size = session.batch_size
    # Padded to the fixed B on the host, in the stored data type, with
    # zero patterns; the padded slots are computed and discarded
    # (D21.7.3)
    raw = np.zeros((batch_size, *patterns_block.shape[1:]), dtype=patterns_block.dtype)
    raw[:n] = patterns_block
    pattern_index = np.full(batch_size, -1, dtype=np.int64)
    pattern_index[:n] = np.asarray(index_block, dtype=np.int64)
    resident, seed_state = session.residents(
        grain, states[grain], options["upsample_factor"]
    )
    rows = _fit_batch(
        session,
        states[grain],
        resident,
        seed_state,
        raw,
        pattern_index,
        options,
        row_slots,
    )
    del resident, seed_state
    return np.ascontiguousarray(rows[:n], dtype=np.float64)


def _prepare_sub_batch(session: _GpuSession, state, resident, raw: np.ndarray):
    """Return ``(targets, coefficients)`` of one sub-batch of P raw
    patterns in the session's namespace: the D4 preprocessing in f64
    and the f64 spline prefilter stored as f32 (D21.3).

    Under numpy the host functions ``preprocess`` and
    ``spline_coefficients`` run per slot, exactly as the CPU builds a
    target.  Under cupy the patterns are uploaded in their stored data
    type, converted exactly to f64, band-passed by one batched
    complex128 FFT against the resident's spectrum-order (``ifftshift``
    -ed) transfer function -- a permutation of the host's ``fftshift``,
    product and ``ifftshift``, so every product is the host's -- and
    prefiltered along the two pattern axes by the cupy namespace's
    ``spline_prefilter`` (``_cuda.py``), the recursion of
    ``scipy.ndimage.spline_filter`` in mirror mode.
    """
    if session.namespace == "numpy":
        targets = np.stack(
            [preprocess(p, transfer_function=state.transfer_function) for p in raw]
        )
        coefficients = np.stack(
            [
                spline_coefficients(
                    t, interpolation=state.interpolation, dtype=np.float32
                )
                for t in targets
            ]
        )
        return np.ascontiguousarray(targets), np.ascontiguousarray(coefficients)
    xp = session.xp
    fft = session.fft
    targets = xp.asarray(raw).astype(xp.float64)
    if resident.transfer_function is not None:
        spectrum = fft.fft2(targets, axes=(1, 2))
        targets = xp.real(fft.ifft2(spectrum * resident.transfer_function, axes=(1, 2)))
        del spectrum
    targets = xp.ascontiguousarray(targets, dtype=xp.float64)
    filtered = targets.copy()
    # The order of ``scipy.ndimage.spline_filter``: the pattern's axis
    # 0 first, then its axis 1, which are the batch's axes 1 and 2
    session.kernels.spline_prefilter(filtered, 1)
    session.kernels.spline_prefilter(filtered, 2)
    coefficients = xp.ascontiguousarray(filtered.astype(xp.float32))
    return targets, coefficients


def _fit_batch(
    session: _GpuSession,
    state,
    resident,
    seed_state,
    raw: np.ndarray,
    pattern_index: np.ndarray,
    options: dict,
    row_slots: dict,
) -> np.ndarray:
    """Return the ``(B, width)`` host rows of one padded device batch.

    Per sub-batch of P slots (D21.7.3), the tail padded to P with zero
    patterns and ``-1`` indices: the preprocessing of
    :func:`_prepare_sub_batch`; one
    :class:`~kikuchipy.indexing._hrebsd._batched.SeedBatch` whose
    ``pattern_index`` is the HOST int64 index of its slots; the seed
    seam ``_batched.seed_spectra`` then ``_batched.seed_homographies``,
    both through the module object at call time (D21.5); and the
    initial shifts ``_batched.initial_shifts`` (D21.4), while the f64
    targets are alive.  The targets and spectra die with their
    sub-batch (D21.5(ii)).  Then ``_batched.run_lockstep``, the
    lockstep IC-GN of D21.6 over all B slots with its packing, and one
    copy of the ``(B, width)`` rows to the host.
    """
    xp = session.xp
    batch_size = session.batch_size
    sub_batch_size = session.sub_batch_size
    ctx = session.context
    coefficient_parts = []
    seed_parts = []
    shift_parts = []
    for start in range(0, batch_size, sub_batch_size):
        stop = start + sub_batch_size
        part = raw[start:stop]
        index = pattern_index[start:stop]
        if part.shape[0] < sub_batch_size:
            # The tail of a batch whose B is not a multiple of P, padded
            # to P so every FFT runs at one shape per run (D21.7.3)
            pad = sub_batch_size - part.shape[0]
            part = np.concatenate(
                [part, np.zeros((pad, *part.shape[1:]), dtype=part.dtype)]
            )
            index = np.concatenate([index, np.full(pad, -1, dtype=np.int64)])
        targets, coefficients = _prepare_sub_batch(session, state, resident, part)
        batch = _batched.SeedBatch(targets, coefficients, np.ascontiguousarray(index))
        target_spectra = _batched.seed_spectra(ctx, batch, seed_state)
        h0 = _batched.seed_homographies(ctx, batch, target_spectra, seed_state)
        shifts = _batched.initial_shifts(ctx, resident, batch)
        keep = min(sub_batch_size, batch_size - start)
        coefficient_parts.append(coefficients[:keep])
        seed_parts.append(xp.asarray(h0, dtype=xp.float64)[:keep])
        shift_parts.append(shifts[:keep])
        del batch, target_spectra, targets, coefficients, h0, shifts
    coefficients = xp.ascontiguousarray(xp.concatenate(coefficient_parts))
    h0 = xp.ascontiguousarray(xp.concatenate(seed_parts))
    shifts = xp.ascontiguousarray(xp.concatenate(shift_parts))
    del coefficient_parts, seed_parts, shift_parts
    rows = _batched.run_lockstep(
        session.kernels,
        resident,
        coefficients,
        h0,
        shifts,
        pattern_index >= 0,
        options,
        row_slots,
    )
    if xp is not np:
        rows = rows.get()
    return np.asarray(rows, dtype=np.float64)
