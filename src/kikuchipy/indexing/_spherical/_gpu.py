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

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.indexing._spherical._indexer import SphericalIndexer

# ------------------- Gate constants and messages -------------------- #

# Version floor of the three-stage gate's stage (a): the first CuPy
# line built against NumPy 2.  Only 14.2.0 is actually tested
# (machine-specific: RTX 2000 Ada 8 GB, driver 595.71, Windows 11;
# recorded in specs/2026-09-07-spherical-gpu/validation.md)
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

# Skeleton-stage sentinel message (failing-tests gate, spec plan 5):
# every body below raises until the implementation gate fills it in
_NOT_IMPLEMENTED = (
    "spherical-indexing-gpu skeleton: implemented at the "
    "implementation gate of specs/2026-09-07-spherical-gpu"
)

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
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _import_cupy() -> Any:
    """Return the imported :mod:`cupy` module, stage (a) of the gate.

    Raises
    ------
    ImportError
        With :data:`_GATE_IMPORT_MESSAGE` when cupy cannot be
        imported, and with :data:`_GATE_VERSION_MESSAGE` when its
        ``__version__`` is below :data:`_CUPY_MINIMUM_VERSION`.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _device_count(cupy: Any) -> int:
    """Return the CUDA device count, stage (b) of the gate.

    Raises
    ------
    RuntimeError
        With :data:`_GATE_DEVICE_MESSAGE` when
        ``cupy.cuda.runtime.getDeviceCount()`` returns zero or raises
        ``CUDARuntimeError`` (no device, no driver).
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _probe_cufft(cupy: Any) -> None:
    """Run one tiny ``cupy.fft`` probe transform, stage (c) of the
    gate, after :func:`_add_nvidia_dll_directories`.

    Raises
    ------
    RuntimeError
        With :data:`_GATE_CUFFT_MESSAGE` when the probe transform
        fails (probe-measured on Windows: ``import cupy`` and the
        device count succeed while the first FFT raises
        ``ImportError: DLL load failed while importing cufft``).
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    and a failing one re-raises the cached exception.

    Called from ``SphericalIndexer.__init__`` when
    ``backend="gpu"`` -- fail fast, before any expensive
    construction (D1.3).  Constructing with ``backend="cpu"`` never
    calls this and never touches cupy.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
        :func:`_sanitized_table`, ``(bw, bw, bw)`` 64-bit float.

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
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _build_g_batch(
    xp: Any, gln_batch: Any, table: Any, bandwidth: int
) -> tuple:
    """Return the per-batch ``(G, G2)`` factors of the spectrum
    GEMMs.

    Parameters
    ----------
    xp
        Array-module handle, :mod:`numpy` or ``cupy``.
    gln_batch
        ``(B, bw, bw)`` complex batch of pattern harmonic
        coefficients.
    table
        The NaN-zeroed transposed ``pi/2`` Wigner d table resident on
        the device, ``(bw, bw, bw)``.
    bandwidth
        The bandwidth ``bw``.

    Returns
    -------
    g
        ``G[b, k, n, j] = conj(gln[b, n, j]) * table_T[k, n, j]``.
    g2
        The same product without the conjugation.

    Notes
    -----
    Both are built for ``n < bw`` rows only -- the even-``slP`` guard
    (D2): at even ``slP``, ``bwP = slP // 2 + 1 > bw`` (the bw-16
    oracle case: ``slP`` 32, ``bwP`` 17) and the quadrant rows
    ``n in [bw, bwP)`` stay zero, as does the never-written
    ``m = bw`` column.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
        Whether the phase has an equatorial mirror plane.

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
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
        ``norm="forward"``.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
        :func:`_neighborhood_offsets`.

    Returns
    -------
    neighborhoods
        ``(B, 27)`` values gathered from each cube's flat view, to
        be shipped to the host (~120 B/pattern D2H with the indices
        and peak values).
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _strip_padding(
    xp: Any, batch_values: Any, slots: Any, n_patterns: int
) -> tuple:
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
    raise NotImplementedError(_NOT_IMPLEMENTED)


# ------------------- VRAM model and batch chooser ------------------- #


def _gpu_memory_per_pattern_bytes(bandwidth: int) -> int:
    """Return the modelled per-pattern device working set ``g(bw)``
    in bytes, pure model math (no device query).

    Notes
    -----
    Per-pattern device bytes: ``fxc`` complex64 + the separable FFT
    intermediates + ``xc`` float32, with buffer reuse, plus the
    per-batch zero fills (D8.2).  Calibration constants are
    measured-then-pinned at the implementation gate from measured
    pool high-water marks (drafting anchors, machine-specific:
    ~50 MB at bw 68 anchored on the measured <= 1.4 GB pool
    high-water at B=32; ~110 MB at bw 88 is a scaling ESTIMATE).
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _gpu_resident_bytes_per_phase(bandwidth: int, normalize: bool) -> int:
    """Return the modelled per-phase resident device bytes: ``A`` and
    ``A2`` complex64, the sanitized transposed table, and ``r_den``
    float32 only when ``normalize`` (D8.3; ~11 MB at bw 68, ~24 at
    88).  Pure model math, no device query.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _default_batch_size(bandwidth: int, free_bytes: int) -> int:
    """Return the default device batch size from the VRAM model.

    ``B = clamp(floor(0.5 * free_bytes / g(bw)), 1, 64)`` -- half the
    measured-free VRAM as headroom (WDDM display-attached cards lie
    about "free"), clamped to ``[1, 64]`` (D8.2).  An explicit
    ``chunksize`` overrides this; ``_batch_estimate`` is bypassed for
    the GPU backend (D8.1).  Pure model math: the free-VRAM query
    happens in ``index_patterns``.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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

    Notes
    -----
    Holds, per phase: the ``A``/``A2`` complex64 tables (built from
    the host complex128 ``flm`` and the NaN-zeroed table, D2), the
    sanitized transposed table, and ``r_den`` float32 only when
    ``normalize=True`` (no ``r_den`` residents on the un-normalized
    path, D2 stage-6).  Holds, per session: the zeroed batch buffers
    (re-zeroed per phase per the D2 freshness rule) and the
    cuFFT-plan-bearing arrays.

    Built inside ``index_patterns`` when ``backend="gpu"`` and
    disposed at the end of the call (:meth:`close` releases the
    pool); shared read-only by all chunk invocations of that one
    call -- correlator clones do NOT duplicate device tables.  An
    ``OutOfMemoryError`` at session build halves ``batch_size``,
    frees the pool and retries, flooring at ``B = 1`` where it
    re-raises as ``MemoryError`` naming the bandwidth, the model
    bytes, the free VRAM and the remedies (D8.4 window a).  Mid-
    compute OOM is window (b), handled by ``index_patterns``.
    """

    def __init__(self, indexer: "SphericalIndexer", batch_size: int) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Dispose every device allocation of this session and free
        the memory pool."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
