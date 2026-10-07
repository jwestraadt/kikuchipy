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
govern.  This is the Stage E failing-tests SKELETON: every frozen name
of D21.2, D21.9.5 and D21.10 exists with its frozen signature, and
every body raises ``NotImplementedError`` until the implementation
gate.  The argument lists beyond the frozen names are fixed by the
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

_NOT_IMPLEMENTED = "Stage E: not implemented yet"


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    :data:`_DEVICE_LOCK`).
    """

    def __init__(
        self,
        namespace: str,
        batch_size: int,
        *,
        device_precision: str,
        seed_precision: str,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Free the residents and the default and pinned pools.
        Idempotent (D21.9.1)."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def __dask_tokenize__(self) -> tuple:
        """Return ``("kikuchipy-hrebsd-gpu-session", id(self))``, so no
        device array enters a dask token (D21.9.1)."""
        raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _free_device_bytes(namespace: str) -> int:
    """Return the free device memory in bytes, the ONE device query
    of the batch choice (D21.10.3: queried by the caller only, never
    inside the model).

    ``cupy.cuda.runtime.memGetInfo()[0]`` under ``"cupy"``; under
    ``"numpy"`` a fixed module value.  Fixed at the failing-tests
    gate so that the default-suite tests can fake free VRAM.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _vram_model_terms(
    n_pixels: int, device_precision: str, seed_precision: str
) -> tuple[int, int, int]:
    """Return the three terms ``(g, p, r)`` of the pure-math VRAM model
    of requirements D21.10.2, bytes, for *n_pixels* pattern pixels.

    ``g`` is the per-slot set held for the whole batch lifetime, ``p``
    the preprocessing and seed transient per sub-batch slot and ``r``
    the per-reference resident set (the window's weight plane and
    constants counted always, conservatively).  No device query.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _vram_model_bytes(
    batch_size: int, n_pixels: int, device_precision: str, seed_precision: str
) -> int:
    """Return ``B * g + P * p + R_MAX * r`` bytes with
    ``P = min(32, B)`` (requirements D21.10.2), the terms of
    :func:`_vram_model_terms`.  Pure math, no device query."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _default_batch_size(
    free_bytes: int, n_pixels: int, device_precision: str, seed_precision: str
) -> int:
    """Return the largest B in :data:`_BATCH_SIZES` whose WHOLE model
    fits ``0.5 * free_bytes`` (requirements D21.10.3), else 1."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)
