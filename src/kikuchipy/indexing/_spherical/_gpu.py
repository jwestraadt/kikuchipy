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

# The following copyright notice is included because the device stage
# split and the pipeline functions in this file are derived and
# adapted from EMSphInx (https://github.com/EMsoft-org/EMSphInx,
# commit 60f3517 and the author's ``feature/GPU`` branch):
# - the device coarse-correlate split -- spectrum product, inverse
#   FFT, peak index and 27-value neighborhood on the device, peak
#   interpolation and Newton refinement on the host -- mirrors
#   ``include/gpu/pipeline.hpp`` and ``include/gpu/corr.hpp`` of the
#   ``feature/GPU`` branch (device refinement,
#   ``refinePeakBatchDevice``, is deliberately not ported)
# - the spectrum, inverse transform, peak and neighborhood
#   semantics reproduced here derive from ``Correlator<Real>`` of
#   ``include/sht/sht_xcorr.hpp`` through their CPU ports in
#   ``_xcorr.py``, whose stage 4-6 numerics this module batches

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Changed by Johan Westraadt, 2026-09: batched, array-module-agnostic
# (NumPy/CuPy) re-expression of the coarse correlation stages for the
# optional GPU backend of kikuchipy's spherical indexing.
# GPL-2.0-or-later, conveyed under GPL-3.0-or-later
# #####################################################################

"""The optional CuPy GPU backend of spherical indexing: availability
gate, device session and the array-module-agnostic ("xp-agnostic")
coarse-correlate pipeline.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**Scope** (spec ``2026-09-07-spherical-gpu``, D2): only stages 4-6 of
the frozen per-pattern pipeline run on the device --

- stage 4, the cross-correlation spectrum
  (``_xcorr._xcorr_spectrum``, ``_xcorr.py`` lines 389-560), as two
  batched complex64 GEMMs against per-phase resident tables plus an
  elementwise four-quadrant mirror fill;
- stage 5, the separable inverse FFT (``_xcorr._inverse_fft``,
  ``_xcorr.py`` lines 1736-1809), ``norm="forward"``, with the exact
  ``m % n_fold`` alpha-plane pruning;
- stage 6, normalise + argmax + 27-neighborhood gather
  (``_xcorr._scale_and_find_peak`` lines 602-639,
  ``_xcorr._find_peak`` lines 563-599 and
  ``_xcorr._extract_neighborhood`` lines 642-763, whose index
  arithmetic the host offsets helper ports).

Everything else -- preprocessing, back-projection, SHT analysis, the
tri-quadratic peak interpolation with its ``emsphinx_compatible``
defects, Newton refinement, positive-score insertion and the
pseudo-symmetry variant loop -- stays on the host, unchanged.  All
three ``emsphinx_compatible`` switch points live in host code or the
host offsets helper: the device stages are compat-neutral.

The device stages run **float32/complex64** (D3); the CPU backend
remains the default, the reference implementation and the parity
oracle for every GPU output (D4).  The pipeline core is xp-agnostic:
every stage function takes an array-module handle ``xp`` (NumPy or
CuPy) and, where it transforms, an FFT namespace ``fft_ns``
(:mod:`numpy.fft`, :mod:`scipy.fft` or ``cupy.fft``), so the logic
runs under NumPy in the default test suite and under CuPy in
production (D11.1).  No ``workers=`` keyword is passed to the FFT
namespace: it is SciPy-only and ``cupy.fft`` rejects it (D7.6).

CuPy is an **optional dependency**: it is never imported at module
scope (a cupy import costs about a second and must not tax
``import kikuchipy``), never registered in
``_constants.deps_for_version_check`` (the ``cupy-cuda12x``
distribution name defeats ``importlib.metadata.version("cupy")``,
measured), and gated at use by the three-stage probe
:func:`_verify_gpu_or_raise` (D6).
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from kikuchipy.indexing._spherical import _fft

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.indexing._spherical._indexer import SphericalIndexer

# ------------------- Gate constants and messages -------------------- #

# Version floor of the three-stage gate's stage (a): the first CuPy
# line built against NumPy 2.  Only 14.2.0 is actually tested
# (machine-specific: RTX 2000 Ada 8 GB, driver 595.71, Windows 11;
# measured 2026-09-07 at the implementation gate)
_CUPY_MINIMUM_VERSION = 13

# The frozen stage messages of the three-stage gate (D6.2).  Each
# stage fails independently in the wild (probe-measured 2026-09-07 on
# this machine) and each message names its remedy.
_GATE_IMPORT_MESSAGE = (
    "Spherical indexing with backend='gpu' requires that 'cupy' is "
    "installed (e.g. 'pip install cupy-cuda12x' matching your CUDA "
    "version; cupy-cuda11x / cupy-cuda13x / ROCm wheels exist), "
    "which is an optional dependency of kikuchipy"
)
_GATE_VERSION_MESSAGE = (
    "Spherical indexing with backend='gpu' requires cupy >= "
    f"{_CUPY_MINIMUM_VERSION}, but version {{version}} is installed; "
    "upgrade with e.g. 'pip install --upgrade cupy-cuda12x' matching "
    "your CUDA version"
)
_GATE_DEVICE_MESSAGE = (
    "Spherical indexing with backend='gpu' requires a CUDA device, "
    "but cupy found none ({detail}); check that an NVIDIA GPU is "
    "present and that the NVIDIA driver is installed and current"
)
_GATE_CUFFT_MESSAGE = (
    "Spherical indexing with backend='gpu' requires that cupy can "
    "load cuFFT, but a probe transform failed ({detail}); install "
    "the full CUDA Toolkit and put its 'bin' directory on PATH, or "
    "'pip install nvidia-cufft-cu12 nvidia-cublas-cu12' (on Windows "
    "kikuchipy registers the nvidia wheel DLL directories for you)"
)

# The cached gate verdict: ``None`` until the first
# :func:`_verify_gpu_or_raise` call, then ``True`` (all three stages
# green) or the exception instance to re-raise.  Tests reset it by
# monkeypatching it back to ``None``.
_gate_result: Any = None

# The process-wide device lock of D7: one GPU consumer per process,
# every device section of every chunk of every concurrent
# ``index_patterns`` call serializes on this one lock.  Module level
# so that concurrent sessions in one process share it.
_DEVICE_LOCK = threading.Lock()

# Element sizes of the device dtypes (D3: the device stages run
# complex64/float32 -- the D3.2 recorded choice (i), all-c64)
_COMPLEX64_BYTES = 8
_COMPLEX128_BYTES = 16
_FLOAT32_BYTES = 4

# ------------------------- Availability gate ------------------------ #


def _add_nvidia_dll_directories() -> None:
    """Best-effort register every ``site-packages/nvidia/*/bin``
    directory with :func:`os.add_dll_directory`, on Windows only.

    Without this every FFT on a Windows pip install dies with
    ``ImportError: DLL load failed while importing cufft`` even with
    the ``nvidia-cufft-cu12``/``nvidia-cublas-cu12`` wheels present
    (probe-measured 2026-09-07: the ``cupy-cuda12x`` wheel bundles no
    cuFFT and CuPy 14.2.0 does not auto-discover the nvidia wheels on
    Windows).  With a full CUDA Toolkit on ``PATH`` the shim is a
    no-op.

    Silent by contract (D6.5): on a non-Windows platform
    (``os.name != "nt"``), with no ``nvidia`` namespace package
    installed, or on any registration failure it returns without
    raising or logging -- the stage-(c) gate message then carries the
    manual remedy.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except Exception:
        return
    bases = [
        base
        for base in list(getattr(nvidia, "__path__", []) or [])
        if os.path.isdir(base)
    ]
    for base in bases:
        try:
            entries = sorted(os.listdir(base))
        except Exception:
            # silent by contract (review-fixed 2026-09-07: the
            # listdir sat outside the try, so an ACL-restricted or
            # concurrently-removed nvidia subtree would escape the
            # shim and poison the cached gate verdict with a raw
            # non-actionable OSError)
            continue
        for entry in entries:
            bin_dir = os.path.join(base, entry, "bin")
            try:
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
            except Exception:
                # silent by contract: the stage-(c) message carries
                # the manual remedy
                pass


def _import_cupy() -> Any:
    """Return the imported :mod:`cupy` module, stage (a) of the gate.

    Raises
    ------
    ImportError
        With :data:`_GATE_IMPORT_MESSAGE` when cupy cannot be
        imported, and with :data:`_GATE_VERSION_MESSAGE` when its
        ``__version__`` is below :data:`_CUPY_MINIMUM_VERSION`.
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


def _device_count(cupy: Any) -> int:
    """Return the CUDA device count, stage (b) of the gate.

    Raises
    ------
    RuntimeError
        With :data:`_GATE_DEVICE_MESSAGE` when
        ``cupy.cuda.runtime.getDeviceCount()`` returns zero or raises
        ``CUDARuntimeError`` (no device, no driver).
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


def _probe_cufft(cupy: Any) -> None:
    """Run one tiny ``cupy.fft`` probe transform and one tiny
    complex64 GEMM, stage (c) of the gate, after
    :func:`_add_nvidia_dll_directories`.

    The GEMM beside the FFT is a review-added hardening (recorded
    2026-09-07, message-frozen-compatible): a Windows install with
    the ``nvidia-cufft-cu12`` wheel but without
    ``nvidia-cublas-cu12`` passes an FFT-only probe and dies
    mid-run at the first spectrum GEMM with a cryptic
    ``ImportError: DLL load failed while importing cublas``
    (measured live); the frozen stage-(c) message already names
    both wheels, so probing cuBLAS here turns that mid-run death
    into the actionable ctor-time gate failure.

    Raises
    ------
    RuntimeError
        With :data:`_GATE_CUFFT_MESSAGE` when the probe transform
        or the probe GEMM fails (probe-measured on Windows:
        ``import cupy`` and the device count succeed while the
        first FFT raises ``ImportError: DLL load failed while
        importing cufft``).
    """
    try:
        cupy.fft.fft(cupy.ones(8, dtype="complex64"))
        ones = cupy.ones((2, 2), dtype="complex64")
        cupy.matmul(ones, ones)
    except Exception as error:
        raise RuntimeError(_GATE_CUFFT_MESSAGE.format(detail=error)) from error


def _verify_gpu_or_raise() -> None:
    """Verify that the CuPy GPU backend is usable, or raise with an
    actionable message.

    The three-stage gate of D6.2, in order: (a) cupy importable with
    ``cupy.__version__ >= 13`` (:func:`_import_cupy`), (b) a CUDA
    device present (:func:`_device_count`), (c) cuFFT loadable via a
    probe transform (:func:`_probe_cufft`), with the Windows DLL shim
    :func:`_add_nvidia_dll_directories` run before stage (c).  Each
    stage fails independently in the wild and each frozen message
    names its remedy.

    The verdict is cached in :data:`_gate_result` after the first
    evaluation, so a passing gate costs one cupy import per process
    and a failing one re-raises the cached exception with no
    re-probe.

    Called from ``SphericalIndexer.__init__`` when
    ``backend="gpu"`` -- fail fast, before any expensive
    construction (D1.3).  Constructing with ``backend="cpu"`` never
    calls this and never touches cupy.
    """
    global _gate_result
    if _gate_result is True:
        return
    if _gate_result is not None:
        # re-raise a FRESH copy of the cached failure where the type
        # allows it (review-fixed 2026-09-07: raising the cached
        # instance itself grows its __traceback__ across raises and
        # can be raised concurrently from two threads); the cached
        # instance rides along as the cause
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
        _probe_cufft(cupy)
    except Exception as error:
        _gate_result = error
        raise
    _gate_result = True


# ----------------- xp-agnostic pipeline core (D11.1) ---------------- #
#
# Pure functions taking an array-module handle ``xp`` (numpy or cupy)
# and, where transforming, an FFT namespace ``fft_ns``.  Under numpy
# these run the identical code path the device runs under cupy, which
# is what buys the default suite coverage of the pipeline LOGIC and a
# numpy-vs-cupy A/B debugging lever.


def _sanitized_table(table: np.ndarray) -> np.ndarray:
    """Return a copy of the transposed ``pi/2`` Wigner d table with
    every NaN slot zeroed.

    Parameters
    ----------
    table
        Transposed ``pi/2`` Wigner d table
        ``table[m, k, j] = d^j_{k,m}(pi/2)`` of shape
        ``(bw, bw, bw)`` 64-bit float, whose undefined slots
        ``j < max(k, m)`` are NaN by validated contract
        (``_xcorr.py`` lines 2213-2215).

    Returns
    -------
    sanitized
        A fresh ``(bw, bw, bw)`` 64-bit float array equal to
        ``table`` with NaN replaced by ``0.0``.

    Notes
    -----
    Load-bearing (D2, review-added): the CPU loops never read the NaN
    slots (their j-loops start at ``max(m, k)`` / ``max(k, n)``,
    ``_xcorr.py`` lines 471 and 489), but the dense GEMMs of
    :func:`_spectrum_batch` read ALL of them and ``0 * NaN = NaN``,
    so a GEMM on the raw table is ~99.5 % NaN.  The zeroed slots plus
    the upper-triangular ``flm``/``gln`` zeros reproduce the CPU
    j-range restriction ``start = max(k, n, m)`` exactly
    (probe-verified at bw 6): the GEMM then equals the CPU loop
    bitwise.
    """
    sanitized = np.array(table, dtype=np.float64, copy=True)
    sanitized[np.isnan(sanitized)] = 0.0
    return sanitized


def _build_a_tables(xp: Any, flm: np.ndarray, table: np.ndarray) -> tuple:
    """Return the per-phase resident ``(A, A2)`` complex64 tables of
    the spectrum GEMMs.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    flm
        ``(bw, bw)`` 128-bit complex host harmonic coefficients of
        the phase's master pattern.
    table
        The NaN-zeroed transposed ``pi/2`` Wigner d table of
        :func:`_sanitized_table`, ``(bw, bw, bw)`` 64-bit float
        (host or already resident on the device).

    Returns
    -------
    a
        ``A[k, j, m] = flm[m, j] * table[m, k, j]``, complex64 of
        shape ``(bw, bw, bw)``, built at complex128 and cast once
        (D3.2).
    a2
        ``A2[k, j, m] = (-1)**(j + m) * A[k, j, m]``, complex64.  The
        ``(-1)**(j + m)`` factor is frozen and review-corrected
        (probe-verified bitwise vs ``_xcorr_spectrum.py_func`` at
        bw 6): the CPU kernel seeds its negated-sum toggle from
        ``(start + m) % 2`` (``_xcorr.py`` lines 504-541), so a
        ``(-1)**j``-only seed flips every odd-``m`` column -- an
        error invisible at even ``n_fold`` and fatal at odd
        ``n_fold`` including 1.
    """
    flm = xp.asarray(flm)
    table = xp.asarray(table)
    bandwidth = int(flm.shape[0])
    # product[k, j, m] = flm[m, j] * table[m, k, j] at complex128
    product = flm.T[None, :, :] * table.transpose(1, 2, 0)
    a = product.astype(np.complex64)
    j = xp.arange(bandwidth).reshape(1, bandwidth, 1)
    m = xp.arange(bandwidth).reshape(1, 1, bandwidth)
    sign = (1 - 2 * ((j + m) % 2)).astype(np.float64)
    a2 = (product * sign).astype(np.complex64)
    return a, a2


def _build_g_batch(xp: Any, gln_batch: Any, table: Any, bandwidth: int) -> tuple:
    """Return the per-batch ``(G, G2)`` factors of the spectrum
    GEMMs.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    gln_batch
        ``(B, bw, bw)`` complex batch of pattern harmonic
        coefficients.  It is cast to complex64 (and the table to
        32-bit float) before the multiply -- the recorded D3.2
        choice (i), all-c64: measured 24 against 113 us/pattern for
        the complex128-multiply-then-cast alternative at bw 68,
        B=32, with indistinguishable cube parity (1.40e-7 against
        1.25e-7 relative to the float64 CPU cube; recorded in
        the implementation-gate measurements of 2026-09-07).
    table
        The NaN-zeroed transposed ``pi/2`` Wigner d table resident on
        the device, ``(bw, bw, bw)``, 32-bit float (any float dtype
        is cast).
    bandwidth
        The bandwidth ``bw``.

    Returns
    -------
    g
        ``G[b, k, n, j] = conj(gln[b, n, j]) * table_T[k, n, j]``,
        complex64.
    g2
        The same product without the conjugation.

    Notes
    -----
    Both are built for ``n < bw`` rows only -- the even-``slP`` guard
    (D2): at even ``slP``, ``bwP = slP // 2 + 1 > bw`` (the bw-16
    oracle case: ``slP`` 32, ``bwP`` 17) and the quadrant rows
    ``n in [bw, bwP)`` stay zero, as does the never-written
    ``m = bw`` column.

    ``table_T[k, n, j]`` is the shared transposed table read at
    ``[k, n, j]``, i.e. ``d^j_{n,k}(pi/2)``, exactly the ``gn``
    factor of the CPU kernel (``_xcorr.py`` line 490).  The arrays
    are materialised contiguous in ``(k, B, n, j)`` order and
    returned as ``(B, k, n, j)`` views, so the
    ``(k, B*n, j)`` GEMM reshape of :func:`_spectrum_batch` is a
    view, never a copy.
    """
    gln = xp.asarray(gln_batch).astype(np.complex64, copy=False)
    table = xp.asarray(table).astype(np.float32, copy=False)
    # (k, B, n, j) contiguous, returned as (B, k, n, j) views; a
    # shape mismatch against ``bandwidth`` fails loudly at the
    # (k, B*n, j) GEMM reshape of _spectrum_batch
    factor = table[:, None, :, :]
    g = xp.conj(gln)[None, :, :, :] * factor
    g2 = gln[None, :, :, :] * factor
    return g.transpose(1, 0, 2, 3), g2.transpose(1, 0, 2, 3)


def _spectrum_batch(
    xp: Any,
    g: Any,
    g2: Any,
    a: Any,
    a2: Any,
    fxc: Any,
    n_fold: int,
    mirror: bool,
) -> None:
    """Fill the batched half-complex cross-correlation spectrum in
    place, stage 4 of the device pipeline.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    g, g2
        The per-batch factors of :func:`_build_g_batch`.
    a, a2
        The per-phase resident tables of :func:`_build_a_tables`.
    fxc
        ``(B, slP, slP, bwP)`` complex64 batch buffer indexed
        ``[b, k, n, m]``, written in place.
    n_fold
        Order of the rotational symmetry of the phase about z, at
        least one.
    mirror
        Whether the phase has an equatorial mirror plane.  The dense
        GEMM needs no mirror branch: under ``mirror`` the CPU's
        stride-2 loop skips the ``(j + m)``-odd coefficients, which
        the symmetry validation guarantees near-zero in ``flm`` (the
        recorded D14.9 deviation below).

    Notes
    -----
    ``value = matmul(G, A)`` and ``negated = matmul(G2, A2)`` as
    batched GEMMs over the k axis (``(k, B*n, j) @ (k, j, m)``;
    einsum measured 1.56x slower, matmul frozen), then the
    four-quadrant mirror fill: the ``(m + n)``-parity signs on the
    two mixed quadrants (``fxc[.., slP-k, n, m]`` and
    ``fxc[.., k, slP-n, m]``), no extra sign on the doubly-negated
    quadrant, and the final ``(-1)**k``, as strided assignments and
    elementwise ops on views.

    **Buffer freshness rule (frozen, D2)**: the batch ``fxc`` buffer
    is re-zeroed (or fully written across all m columns) here on
    every per-phase call, so a buffer cycling between phases with
    different ``n_fold``/``mirror`` never carries stale columns.  The
    CPU kernel WRITES its systematic-zero columns, zero rows and pad
    slices on every call (``_xcorr.py`` lines 492-501); the only
    never-written slots are the ``m = bw`` column at even ``slP``.

    **Recorded systematic deviation** (``mirror=True``, D14.9): the
    dense GEMM also sums the ``(j + m)``-odd coefficients the CPU's
    stride-2 loop skips, which the symmetry validation guarantees
    only <= 1e-8 relative power -- absorbed by the D4 MTP bands.
    """
    bandwidth = int(a.shape[0])
    batch_size = int(fxc.shape[0])
    slp = int(fxc.shape[1])

    # the D2 freshness rule: fully re-zero the batch buffer per phase
    fxc[...] = 0

    # the two batched GEMMs over the k axis; the transposes undo the
    # (B, k, n, j) views of _build_g_batch, so both reshapes are
    # views of the contiguous (k, B, n, j) buffers
    lhs = g.transpose(1, 0, 2, 3).reshape(bandwidth, batch_size * bandwidth, bandwidth)
    lhs2 = g2.transpose(1, 0, 2, 3).reshape(
        bandwidth, batch_size * bandwidth, bandwidth
    )
    value = (
        xp.matmul(lhs, a)
        .reshape(bandwidth, batch_size, bandwidth, bandwidth)
        .transpose(1, 0, 2, 3)
    )
    negated = (
        xp.matmul(lhs2, a2)
        .reshape(bandwidth, batch_size, bandwidth, bandwidth)
        .transpose(1, 0, 2, 3)
    )
    # the final (-1)^k of the negated sum (_xcorr.py lines 540-541)
    negated[:, 1::2] *= -1

    # the systemic-zero columns m % n_fold != 0: the CPU kernel skips
    # those m and writes zeros (the flm rows there are only ASSUMED
    # zero, so the dense GEMM result must not be trusted either)
    if n_fold > 1:
        zero_columns = xp.asarray(np.flatnonzero((np.arange(bandwidth) % n_fold) != 0))
        value[..., zero_columns] = 0
        negated[..., zero_columns] = 0

    # the (m + n)-parity sign of the two mixed quadrants
    # (_xcorr.py lines 546-555), float32 so complex64 survives
    n_index = np.arange(bandwidth).reshape(1, 1, bandwidth, 1)
    m_index = np.arange(bandwidth).reshape(1, 1, 1, bandwidth)
    mixed_sign = xp.asarray((1 - 2 * ((n_index + m_index) % 2)).astype(np.float32))

    # the four quadrants (_xcorr.py lines 542-555): [k, n],
    # [slP-k, slP-n] (negated, no extra sign), [slP-k, n] (value,
    # parity sign) and [k, slP-n] (negated, parity sign); the zero
    # rows n in [bw, bwP), the pad slices k in [bw, slP-bw] and the
    # even-slP m = bw column stay zeroed
    reverse = slice(slp - 1, slp - bandwidth, -1)
    fxc[:, :bandwidth, :bandwidth, :bandwidth] = value
    fxc[:, reverse, reverse, :bandwidth] = negated[:, 1:, 1:, :]
    fxc[:, reverse, :bandwidth, :bandwidth] = value[:, 1:, :, :] * mixed_sign
    fxc[:, :bandwidth, reverse, :bandwidth] = (
        negated[:, :, 1:, :] * mixed_sign[:, :, 1:, :]
    )


def _inverse_fft_batch(xp: Any, fft_ns: Any, fxc: Any, n_fold: int) -> Any:
    """Return the batched stored half of the real cross-correlation
    cubes, stage 5 of the device pipeline.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    fft_ns
        FFT namespace with ``ifft`` and ``irfft`` accepting ``axis``
        and ``norm`` (:mod:`numpy.fft`, :mod:`scipy.fft` or
        ``cupy.fft``).  No ``workers=`` keyword is passed: it is
        SciPy-only (D7.6).
    fxc
        ``(B, slP, slP, bwP)`` complex64 batch spectrum indexed
        ``[b, k, n, m]``.
    n_fold
        Order of the rotational symmetry about z.  Only the alpha
        planes ``m % n_fold == 0`` are transformed, scattered into a
        zeroed half-complex buffer -- the exact CPU skip
        (``_xcorr.py`` lines 1787-1801).

    Returns
    -------
    xc
        ``(B, bwP, slP, slP)`` float32 batch of stored halves,
        mirroring ``_xcorr._inverse_fft`` verbatim: batched ``ifft``
        along k (length ``slP``), slice to ``bwP``, batched ``ifft``
        along n, batched ``irfft`` along m to ``slP``, all
        ``norm="forward"``.  Every intermediate is kept complex64
        (:mod:`numpy.fft` upcasts to complex128 and is cast back;
        ``cupy.fft`` preserves complex64 and the casts are no-ops).
        The NumPy path therefore runs at COMPARABLE, not identical,
        precision to the device: each numpy stage computes at
        complex128 and re-rounds, while cupy accumulates in true
        complex64 -- measured cupy cube error reaches ~3.6e-7
        relative where numpy stays ~1.35e-7 (recorded 2026-09-07),
        so a numpy-measured band must never be reused for a
        device-side cube assert.
    """
    batch_size = int(fxc.shape[0])
    slp = int(fxc.shape[1])
    bwp = int(fxc.shape[3])
    if n_fold == 1:
        # backward along k for every n, then along n for k < bwP
        along_k = fft_ns.ifft(fxc, axis=1, norm="forward").astype(
            np.complex64, copy=False
        )
        planes = fft_ns.ifft(along_k[:, :bwp], axis=2, norm="forward").astype(
            np.complex64, copy=False
        )
    else:
        # the alpha planes m % n_fold != 0 are the systemic zeros
        # the spectrum fill wrote, so skipping them is exact
        along_k = fft_ns.ifft(fxc[:, :, :, ::n_fold], axis=1, norm="forward").astype(
            np.complex64, copy=False
        )
        along_n = fft_ns.ifft(along_k[:, :bwp], axis=2, norm="forward").astype(
            np.complex64, copy=False
        )
        planes = xp.zeros((batch_size, bwp, slp, bwp), dtype=np.complex64)
        planes[:, :, :, ::n_fold] = along_n
    xc = fft_ns.irfft(planes, n=slp, axis=3, norm="forward")
    return xc.astype(np.float32, copy=False)


def _scale_argmax_batch(xp: Any, xc: Any, r_den: Any) -> tuple:
    """Scale the batched cubes in place and return their flat argmax
    indices and peak values, stage 6 of the device pipeline.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    xc
        ``(B, bwP, slP, slP)`` float32 batch of cross-correlation
        cubes, multiplied by ``r_den`` element by element in place
        when ``r_den`` is given.
    r_den
        ``(bwP, slP, slP)`` float32 reciprocal Huhle denominator, or
        ``None`` on the un-normalized path (``normalize=False``,
        D2 stage-6): the multiply is skipped and the argmax runs on
        the raw cube, matching plain ``_find_peak``.

    Returns
    -------
    indices
        ``(B,)`` integer flat per-cube argmax indices.  The argmax is
        the **first occurrence** among ties, matching
        ``_find_peak``'s strict-``>`` first-max semantics
        (``cp.argmax`` and ``np.argmax`` both return the first
        occurrence; re-pinned by planted-tie tests in BOTH the
        default numpy suite and the gated cupy suite).
    values
        ``(B,)`` float32 peak values of the scaled cubes.

    Notes
    -----
    Recorded deviation, not emulated (D4.7): ``_find_peak`` skips NaN
    anywhere except the flat-index-0 seed, while ``argmax`` returns
    the position of any NaN.  NaN in slot 0 agrees (both return 0);
    the divergent case is NaN at any other index, reachable only via
    a degenerate ``r_den`` family.
    """
    batch_size = int(xc.shape[0])
    if r_den is not None:
        xc *= r_den
    flat = xc.reshape(batch_size, -1)
    indices = xp.argmax(flat, axis=1)
    values = xp.take_along_axis(flat, indices[:, None], axis=1)[:, 0]
    return indices, values


def _neighborhood_offsets(
    flat_index: int, bwp: int, slp: int, emsphinx_compatible: bool
) -> np.ndarray:
    """Return the 27 flat offsets of the 3 x 3 x 3 neighborhood
    around a grid point, on the host.

    Parameters
    ----------
    flat_index
        Flat index into a ``(bwP, slP, slP)`` cube, at or near a
        local maximum.
    bwp
        Half side length ``bwP = slP // 2 + 1``.
    slp
        Padded side length ``slP``.
    emsphinx_compatible
        Whether to reproduce the two C++ glide defects of
        ``_extract_neighborhood``: the per-slot alpha/gamma shift and
        the even-``slP`` one-past-the-axis defect, with every flat
        offset clamped to ``bwp * slp * slp - 1`` (``_xcorr.py``
        lines 196-211 and 730-746).  ``False`` uses the per-plane
        glide.

    Returns
    -------
    offsets
        ``(27,)`` 64-bit integer flat offsets in the C order of the
        ``(3, 3, 3)`` neighborhood buffer, such that
        ``cube.reshape(-1)[offsets].reshape(3, 3, 3)`` equals the
        ``nh`` written by ``_extract_neighborhood`` bitwise, in both
        compat settings.

    Notes
    -----
    This ports the *index arithmetic* of ``_extract_neighborhood``
    exactly, including its defects, so the device gather of
    :func:`_gather_neighborhoods` ships back the identical 27 values
    the host interpolator would have read (D2 stage-6).  Pinned by a
    parity unit test against ``_extract_neighborhood`` on random
    cubes in both compat settings at odd and even ``slP``.
    """
    flat_index = int(flat_index)
    bwp = int(bwp)
    slp = int(slp)
    k0, remainder = divmod(flat_index, slp * slp)
    n0, m0 = divmod(remainder, slp)
    # the periodic index arrays (_xcorr.py lines 715-729)
    inds = np.empty((3, 3), dtype=np.int64)
    for axis, center in enumerate((k0, n0, m0)):
        inds[axis, 1] = center
        inds[axis, 0] = slp - 1 if center == 0 else center - 1
        above = center + 1
        inds[axis, 2] = 0 if above == slp else above
    offsets = np.empty(27, dtype=np.int64)
    position = 0
    if emsphinx_compatible:
        # the per-slot glide (_xcorr.py lines 730-738)
        for i in range(3):
            if inds[0, i] >= bwp:
                alpha = inds[2, i]
                inds[2, i] = alpha + bwp - 1 if alpha < bwp else alpha - bwp
                gamma = inds[1, i]
                inds[1, i] = gamma + bwp - 1 if gamma < bwp else gamma - bwp
                inds[0, i] = slp - inds[0, i]
        # every flat offset clamped to the last element
        # (_xcorr.py lines 739-746)
        last = bwp * slp * slp - 1
        for k in range(3):
            for n in range(3):
                for m in range(3):
                    offset = inds[0, k] * slp * slp + inds[1, n] * slp + inds[2, m]
                    if offset > last:
                        offset = last
                    offsets[position] = offset
                    position += 1
    else:
        # the per-plane glide, exact on the grid for even slP
        # (_xcorr.py lines 747-763)
        shift = slp // 2
        for k in range(3):
            beta = int(inds[0, k])
            glided = beta >= bwp
            if glided:
                beta = slp - beta
            for n in range(3):
                gamma = int(inds[1, n])
                if glided:
                    gamma = (gamma + shift) % slp
                for m in range(3):
                    alpha = int(inds[2, m])
                    if glided:
                        alpha = (alpha + shift) % slp
                    offsets[position] = beta * slp * slp + gamma * slp + alpha
                    position += 1
    return offsets


def _gather_neighborhoods(xp: Any, xc: Any, offsets: Any) -> Any:
    """Gather the 27-value neighborhoods of a batch of cubes at
    host-computed offsets, on the device.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    xc
        ``(B, bwP, slP, slP)`` float32 batch of (scaled) cubes.
    offsets
        ``(B, 27)`` integer flat per-cube offsets from
        :func:`_neighborhood_offsets`.  They are per-cube offsets
        into each cube's own flat view, so the gather runs along
        axis 1 of the flattened batch -- never against the flat view
        of the whole batch, which would need a ``b * cube_size``
        base added.

    Returns
    -------
    neighborhoods
        ``(B, 27)`` values gathered from each cube's flat view, to
        be shipped to the host (~120 B/pattern D2H with the indices
        and peak values).
    """
    batch_size = int(xc.shape[0])
    flat = xc.reshape(batch_size, -1)
    offsets = xp.asarray(offsets)
    return xp.take_along_axis(flat, offsets, axis=1)


def _pad_batch(xp: Any, stack: Any, batch_size: int, keep: Any = None) -> tuple:
    """Return a zero-padded fixed-size batch of the kept rows of a
    stack, and the slot map back to pattern indices.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    stack
        ``(n, ...)`` array of per-pattern rows (e.g. the ``gln``
        batch collected by the host stages 0-3).
    batch_size
        The fixed device batch size ``B``.  The kept rows are packed
        into slots ``0..n_kept-1`` and the remaining slots are
        zero-padded (D7.3: one cuFFT/GEMM plan set for the whole run,
        every real pattern transformed at the same batch shape).
    keep
        Optional ``(n,)`` boolean mask; rows with ``False`` (failed
        patterns: ``ptp == 0`` guards, masked points) are excluded
        from the batch -- their slots padded -- and take the
        identical CPU fill-row path (D7.4).  ``None`` keeps every
        row.

    Returns
    -------
    padded
        ``(batch_size, ...)`` array with the kept rows in slots
        ``0..n_kept-1`` and zeros elsewhere.
    slots
        ``(n_kept,)`` 64-bit integer pattern indices, one per
        occupied batch slot.

    Raises
    ------
    ValueError
        If more rows are kept than ``batch_size`` holds.
    """
    stack = xp.asarray(stack)
    batch_size = int(batch_size)
    n_patterns = int(stack.shape[0])
    if keep is None:
        slots = xp.arange(n_patterns, dtype=np.int64)
    else:
        slots = xp.flatnonzero(xp.asarray(keep)).astype(np.int64, copy=False)
    n_kept = int(slots.shape[0])
    if n_kept > batch_size:
        raise ValueError(
            f"cannot pack {n_kept} kept rows into a device batch of size {batch_size}"
        )
    padded = xp.zeros((batch_size,) + tuple(stack.shape[1:]), dtype=stack.dtype)
    if n_kept:
        padded[:n_kept] = stack[slots]
    return padded, slots


def _strip_padding(xp: Any, batch_values: Any, slots: Any, n_patterns: int) -> tuple:
    """Scatter per-slot batch results back to per-pattern rows,
    discarding the padded slots.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    batch_values
        ``(batch_size, ...)`` per-slot results of a device batch.
    slots
        ``(n_kept,)`` integer pattern indices from :func:`_pad_batch`.
    n_patterns
        Number of patterns of the chunk.

    Returns
    -------
    values
        ``(n_patterns, ...)`` array with the slot results scattered
        to their patterns and zeros at rows no slot wrote.
    written
        ``(n_patterns,)`` boolean mask of the rows a slot wrote;
        excluded (failed) patterns are ``False`` and keep the CPU
        fill-row path.  Padded slots (``slots.size <= slot <
        batch_size``) are computed and discarded here, never reaching
        any result row (D7.3).
    """
    slots = xp.asarray(slots)
    n_patterns = int(n_patterns)
    n_kept = int(slots.shape[0])
    values = xp.zeros(
        (n_patterns,) + tuple(batch_values.shape[1:]), dtype=batch_values.dtype
    )
    written = xp.zeros(n_patterns, dtype=bool)
    if n_kept:
        values[slots] = batch_values[:n_kept]
        written[slots] = True
    return values, written


# ------------------- VRAM model and batch chooser ------------------- #


def _grid_lengths(bandwidth: int) -> tuple[int, int]:
    """Return ``(slP, bwP)`` of a bandwidth, the frozen grid math of
    ``SphericalCrossCorrelator`` (``_xcorr.py`` lines 2298-2300)."""
    slp = int(_fft.fast_size(2 * int(bandwidth) - 1))
    return slp, slp // 2 + 1


def _gpu_memory_per_pattern_bytes(bandwidth: int) -> int:
    """Return the modelled per-pattern device working set ``g(bw)``
    in bytes, pure model math (no device query).

    Notes
    -----
    The component sum of the live per-batch device arrays of stages
    4-6, per pattern (D8.2): the uploaded ``gln`` row (complex128),
    the ``G``/``G2`` pair, the two GEMM outputs and the two
    mixed-quadrant sign products of the fill (complex64), the
    ``fxc`` spectrum and the k-axis transform of the same size
    (complex64), the sliced n-axis transform and the pruning scatter
    buffer (``bwP slP bwP`` complex64 each) and the ``xc`` cube
    (float32).  Calibrated at the implementation gate against
    measured pool high-water marks (machine-specific, RTX 2000 Ada
    8 GB: recorded in
    the implementation-gate measurements of 2026-09-07): the model
    returns ~49.9 MB at bw 68 and ~108.3 at 88 against measured
    unpruned/pruned working sets of 52.4/37.6 and (pruned) 82.3
    MB/pattern -- the few percent the unpruned bw-68 run sits above
    the sum is memory-pool block granularity, absorbed by the
    half-free-VRAM headroom of :func:`_default_batch_size`.  The
    drafting anchors were ~50 MB at bw 68 and a ~110 MB scaling
    estimate at bw 88.
    """
    bandwidth = int(bandwidth)
    slp, bwp = _grid_lengths(bandwidth)
    cube = bandwidth**3
    gln = bandwidth * bandwidth * _COMPLEX128_BYTES
    g_pair = 2 * cube * _COMPLEX64_BYTES
    gemm_out = 2 * cube * _COMPLEX64_BYTES
    mirror_fill = 2 * (bandwidth - 1) * bandwidth * bandwidth * _COMPLEX64_BYTES
    fxc = slp * slp * bwp * _COMPLEX64_BYTES
    along_k = slp * slp * bwp * _COMPLEX64_BYTES
    along_n = bwp * slp * bwp * _COMPLEX64_BYTES
    planes = bwp * slp * bwp * _COMPLEX64_BYTES
    xc = bwp * slp * slp * _FLOAT32_BYTES
    return gln + g_pair + gemm_out + mirror_fill + fxc + along_k + along_n + planes + xc


def _gpu_resident_bytes_per_phase(bandwidth: int, normalize: bool) -> int:
    """Return the modelled per-phase resident device bytes: ``A`` and
    ``A2`` complex64, the sanitized transposed table (float32, shared
    in practice but modelled per phase, conservative), and ``r_den``
    float32 only when ``normalize`` (D8.3; the model returns ~11.3 MB
    at bw 68 and ~24.4 at 88 with ``normalize=True``).  Pure model
    math, no device query.
    """
    bandwidth = int(bandwidth)
    slp, bwp = _grid_lengths(bandwidth)
    cube = bandwidth**3
    n_bytes = 2 * cube * _COMPLEX64_BYTES + cube * _FLOAT32_BYTES
    if normalize:
        n_bytes += bwp * slp * slp * _FLOAT32_BYTES
    return n_bytes


def _default_batch_size(bandwidth: int, free_bytes: int) -> int:
    """Return the default device batch size from the VRAM model.

    ``B = clamp(floor(0.5 * free_bytes / g(bw)), 1, 64)`` -- half the
    measured-free VRAM as headroom (WDDM display-attached cards lie
    about "free"), clamped to ``[1, 64]`` (D8.2).  An explicit
    ``chunksize`` overrides this; ``_batch_estimate`` is bypassed for
    the GPU backend (D8.1).  Pure model math: the free-VRAM query
    happens in ``index_patterns``.
    """
    per_pattern = _gpu_memory_per_pattern_bytes(bandwidth)
    return max(1, min(64, int(0.5 * int(free_bytes) // per_pattern)))


# --------------------------- Device session ------------------------- #


class _GpuSession:
    """Per-``index_patterns``-call device session: resident tables,
    batch buffers, the process-wide device lock and the memory pool
    handle (D7.2).

    Parameters
    ----------
    indexer
        The :class:`~kikuchipy.indexing.SphericalIndexer` whose
        phases the session serves.  The indexer itself holds only the
        backend string -- nothing device-resident is ever stored on
        it, so nothing device-resident can be captured in a dask
        graph, pickled or shared across calls.
    batch_size
        The fixed device batch size ``B`` (chunk size == batch size
        under ``backend="gpu"``, D8.1).

    Attributes
    ----------
    xp : module
        The imported :mod:`cupy`, the array-module handle every
        pipeline function takes.
    fft : module
        ``cupy.fft``, the FFT namespace of
        :func:`_inverse_fft_batch`.
    lock : threading.Lock
        The process-wide device lock :data:`_DEVICE_LOCK` (D7.1):
        one GPU consumer per process, every device section
        serializes on it.
    batch_size : int
        The fixed device batch size ``B``.
    bandwidth, side_length, half_side_length : int
        The frozen grid geometry ``bw``, ``slP`` and ``bwP``.
    normalize : bool
        Whether the phases carry ``r_den`` residents.
    table : cupy.ndarray
        The NaN-zeroed transposed ``pi/2`` Wigner d table, 32-bit
        float, resident once and shared by every per-batch
        :func:`_build_g_batch` call (the recorded D3.2 choice (i):
        the ``G`` products multiply all-complex64; the ``A`` tables
        are still built from the 64-bit float table at complex128
        and cast once).
    a_tables : tuple of tuple
        Per phase, in the indexer's phase order, the ``(A, A2)``
        complex64 residents of :func:`_build_a_tables`.
    r_dens : tuple
        Per phase, the float32 reciprocal Huhle denominator resident,
        or ``None`` in every slot on the un-normalized path (D2
        stage-6: no ``r_den`` residents, the stage-6 multiply
        skipped).
    fxc : cupy.ndarray
        The ``(B, slP, slP, bwP)`` complex64 batch spectrum buffer,
        zeroed at build and re-zeroed by every
        :func:`_spectrum_batch` call (the D2 freshness rule); its
        transforms are what the cuFFT plans attach to.

    Notes
    -----
    Holds, per phase: the ``A``/``A2`` complex64 tables (built from
    the host complex128 ``flm`` and the NaN-zeroed table, D2), and
    ``r_den`` float32 only when ``normalize=True`` (no ``r_den``
    residents on the un-normalized path, D2 stage-6).  Holds, per
    session: the sanitized table, the zeroed batch buffer and the
    process-wide lock handle.  The phase symmetry flags
    (``n_fold``, ``mirror``) stay on the indexer's host objects,
    which ``_index_chunk_gpu`` reads directly.

    Built inside ``index_patterns`` when ``backend="gpu"`` and
    disposed at the end of the call (:meth:`close` releases the
    pool); shared read-only by all chunk invocations of that one
    call -- correlator clones do NOT duplicate device tables.  An
    ``OutOfMemoryError`` at session build halves ``batch_size``,
    frees the pool and retries, flooring at ``B = 1`` where it
    re-raises as ``MemoryError`` naming the bandwidth, the model
    bytes, the free VRAM and the remedies (D8.4 window a).  Mid-
    compute OOM is window (b), handled by ``index_patterns``.  Both
    loops live in ``index_patterns``: this class only allocates.
    """

    def __init__(self, indexer: "SphericalIndexer", batch_size: int) -> None:
        # re-verify (cached per process): a session may be built in a
        # process which never constructed the indexer, and the gate
        # runs the Windows DLL shim cuFFT needs (D6.5)
        _verify_gpu_or_raise()
        import cupy

        self.xp = cupy
        self.fft = cupy.fft
        self.lock = _DEVICE_LOCK
        self.batch_size = int(batch_size)
        self.bandwidth = int(indexer.bandwidth)
        slp = int(indexer.side_length)
        bwp = slp // 2 + 1
        self.side_length = slp
        self.half_side_length = bwp
        self.normalize = bool(indexer.normalize)

        # the shared NaN-zeroed table (D2): resident float32 for the
        # per-batch all-complex64 G build (the D3.2 choice (i)); a
        # float64 device copy exists only while the A tables build
        # from it at complex128 (D3.2's "cast once")
        sanitized = _sanitized_table(indexer.wigner_d_half_pi)
        table64 = cupy.asarray(sanitized)
        self.table = table64.astype(np.float32)

        # per-phase residents, in the indexer's phase order
        if self.normalize:
            sources = [(c.flm, c.r_den) for c in indexer.correlators]
        else:
            sources = [(flm, None) for flm, _, _ in indexer.spectra]
        a_tables = []
        r_dens = []
        for flm, r_den in sources:
            a_tables.append(_build_a_tables(cupy, flm, table64))
            if r_den is None:
                r_dens.append(None)
            else:
                # cast on the host, then one float32 upload; the
                # f64 -> f32 inf-overflow family is the recorded
                # D4.7 deviation
                r_dens.append(
                    cupy.asarray(np.ascontiguousarray(r_den, dtype=np.float32))
                )
        self.a_tables = tuple(a_tables)
        self.r_dens = tuple(r_dens)
        # release the float64 table before the batch buffer allocates
        del table64
        cupy.get_default_memory_pool().free_all_blocks()

        # the batch spectrum buffer, zeroed at build (D2 freshness)
        self.fxc = cupy.zeros((self.batch_size, slp, slp, bwp), dtype=np.complex64)

    def close(self) -> None:
        """Dispose every device allocation of this session and free
        the memory pool.  Idempotent: a second call is a no-op."""
        cupy = getattr(self, "xp", None)
        self.a_tables = ()
        self.r_dens = ()
        self.table = None
        self.fxc = None
        self.xp = None
        self.fft = None
        if cupy is not None:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
