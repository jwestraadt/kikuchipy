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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517), all from
# ``include/sht/square_sht.hpp``:
# - The ``a^m_n``/``b^m_n`` tables of Schaeffer's normalized
#   associated Legendre recursion, built in
#   ``square::DiscreteSHT::Constants::Constants()`` (lines 347-373)
# - The ring weight set assembly of the same constructor, i.e. one
#   set per skipped ring for the Lambert layout and the ``skip = 0``
#   set replicated for the Legendre layout (lines 375-381)
# - ``square::DiscreteSHT::analyze()`` (lines 414-486)
# - ``square::DiscreteSHT::synthesize()`` (lines 495-572)
# - ``square::DiscreteSHT::Legendre()`` and
#   ``square::DiscreteSHT::Lambert()`` (lines 107-112)

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
# Modified by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""Discrete spherical harmonic transform on square grids.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.
"""

from functools import lru_cache
import math

from numba import njit
import numpy as np
from scipy.fft import irfft, rfft

from kikuchipy.indexing._spherical import _grid

# ------------------------- Lookup tables ---------------------------- #


def _alf_recursion_tables(bandwidth: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the coefficients of Schaeffer's normalized associated
    Legendre function recursion.

    Parameters
    ----------
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).

    Returns
    -------
    amn
        Coefficients ``a^m_n`` in an array of shape
        ``(bandwidth, bandwidth)`` and 64-bit floating point data
        type. The diagonal holds ``a^m_m`` of Schaeffer equation 16
        and the entries above it ``a^m_n`` of equation 17.
    bmn
        Coefficients ``b^m_n`` of Schaeffer equation 18 in an array of
        the same shape and data type, non-zero for ``n >= m + 2``.

    Notes
    -----
    The functions evaluated at ``x = cos(theta)`` with
    ``r = sqrt(1 - x ** 2)`` are

    - ``P^m_m = amn[m, m] * r ** m`` (equation 13),
    - ``P^m_(m + 1) = amn[m, m + 1] * x * P^m_m`` (equation 14) and
    - ``P^m_n = amn[m, n] * x * P^m_(n - 1)
      - bmn[m, n] * P^m_(n - 2)`` (equation 15).

    They are fully normalized :cite:`schaeffer2013efficient` and omit
    the Condon-Shortley phase ``(-1) ** m``, which
    :class:`SphericalHarmonicTransform` re-applies when weighting
    (``analyze``) or assembling (``synthesize``) the ring spectrum.
    """
    amn = np.zeros((bandwidth, bandwidth), dtype=np.float64)
    bmn = np.zeros((bandwidth, bandwidth), dtype=np.float64)
    # 1 / (4 pi), the constant of the a^m_m calculation
    k4p = 1 / (np.pi * 4)
    # prod_(k = 1)^|m| (2 k + 1) / (2 k), which is one for m = 0
    kamm = 1.0
    for m in range(bandwidth):
        amn[m, m] = math.sqrt(kamm * k4p)
        kamm *= (2 * m + 3) / (2 * m + 2)
        if m + 1 == bandwidth:
            break
        m2 = m * m
        # n ** 2 and n ** 2 - m ** 2 for n = m + 1
        n2 = (m + 1) * (m + 1)
        n2m2 = n2 - m2
        amn[m, m + 1] = math.sqrt((4 * n2 - 1) / n2m2)
        for n in range(m + 2, bandwidth):
            # Reuse the previous n ** 2 - m ** 2 as the new
            # (n - 1) ** 2 - m ** 2
            n12m2 = float(n2m2)
            n2 = n * n
            n2m2 = n2 - m2
            amn[m, n] = math.sqrt((4 * n2 - 1) / n2m2)
            bmn[m, n] = math.sqrt(((2 * n + 1) / (2 * n - 3)) * (n12m2 / n2m2))
    return amn, bmn


@lru_cache(maxsize=4)
def _ring_dft_tables_cached(
    dim: int, bandwidth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the memoized tables of :func:`_ring_dft_tables`.

    Parameters
    ----------
    dim
        Side length of the square grid.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).

    Returns
    -------
    offsets, cos_table, sin_table
        As documented in :func:`_ring_dft_tables`.
    """
    n_ring = _grid.n_rings(dim)
    ring = np.arange(n_ring, dtype=np.int64)
    n_phi = np.maximum(1, 8 * ring)
    m_lim = np.minimum(bandwidth, 4 * ring + 1)
    offsets = np.zeros(n_ring + 1, dtype=np.int64)
    np.cumsum(m_lim * n_phi, out=offsets[1:])
    cos_table = np.empty(offsets[-1], dtype=np.float64)
    sin_table = np.empty(offsets[-1], dtype=np.float64)
    for y in range(n_ring):
        orders = np.arange(m_lim[y], dtype=np.int64)[:, np.newaxis]
        slots = np.arange(n_phi[y], dtype=np.int64)[np.newaxis, :]
        # The product is reduced modulo N_phi(y) before it is scaled,
        # so that the tabulated angles stay in [0, 2 pi) and are as
        # accurate as the argument reduction of an FFT twiddle factor
        angles = (2 * np.pi / n_phi[y]) * ((orders * slots) % n_phi[y])
        block = slice(offsets[y], offsets[y + 1])
        cos_table[block] = np.cos(angles).ravel()
        sin_table[block] = np.sin(angles).ravel()
    return offsets, cos_table, sin_table


def _ring_dft_tables(
    dim: int, bandwidth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ragged per-ring cosine and sine tables for the direct
    ring discrete Fourier transforms of the Numba path.

    Parameters
    ----------
    dim
        Side length of the square grid.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).

    Returns
    -------
    offsets
        Start of each ring's table in ``cos_table`` and ``sin_table``,
        in an array of shape ``(n_rings(dim) + 1,)`` and 64-bit
        integer data type.
    cos_table
        Flat 64-bit floating point array holding, for ring ``y``,
        ``cos(2 * pi * m * p / N_phi(y))`` in C order over
        ``m < m_lim(y)`` and ``p < N_phi(y)``, where
        ``N_phi(y) = max(1, 8 * y)`` and
        ``m_lim(y) = min(bandwidth, 4 * y + 1)``.
    sin_table
        The corresponding sines, in the same layout.

    Notes
    -----
    The tables are memoised per ``(dim, bandwidth)``, and are used
    only for grids no larger than
    :attr:`SphericalHarmonicTransform.numba_ring_dft_max_dim`, since
    they grow as ``dim ** 3``. At that limit one entry holds about
    32 MB of cosines and sines and up to four entries are retained
    for the lifetime of the process, so call
    ``_ring_dft_tables_cached.cache_clear()`` to release them. Every
    caller with the same key gets the same arrays, which must
    therefore not be modified in place.
    """
    return _ring_dft_tables_cached(int(dim), int(bandwidth))


# ---------------------------- Kernels ------------------------------- #


@njit(cache=True, nogil=True)
def _analyze_ring_kernel(
    alm: np.ndarray,
    g_sym: np.ndarray,
    g_asym: np.ndarray,
    x: float,
    amn: np.ndarray,
    bmn: np.ndarray,
    bandwidth: int,
    m_lim: int,
) -> None:
    """Accumulate one ring's contribution to the harmonic
    coefficients, in place.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[m, l]`` to accumulate into, of
        shape ``(bandwidth, bandwidth)`` and 128-bit complex data
        type.
    g_sym
        Weighted ring spectrum ``(G_north + G_south) / 2`` of the
        symmetric modes, i.e. even ``l + m``, of shape ``(m_lim,)``
        and 128-bit complex data type.
    g_asym
        Weighted ring spectrum ``(G_north - G_south) / 2`` of the
        antisymmetric modes, i.e. odd ``l + m``, of the same shape
        and data type.
    x
        Cosine of the ring latitude.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).
    m_lim
        Number of orders to accumulate, ``min(bandwidth, 4 * y + 1)``
        for ring ``y``.

    Notes
    -----
    This is the Legendre recursion shared by the Numba ring DFT path
    and the :mod:`scipy.fft` path of
    :class:`SphericalHarmonicTransform`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # (1 - x ** 2) ** (|m| / 2), one for m = 0
    kpmm = 1.0
    r1x2 = math.sqrt(1.0 - x * x)
    for m in range(m_lim):
        gs = g_sym[m]
        ga = g_asym[m]
        # P^m_m (Schaeffer equation 13)
        pmn2 = amn[m, m] * kpmm
        kpmm *= r1x2
        alm[m, m] += gs * pmn2
        if m + 1 == bandwidth:  # P^m_(m + 1) does not exist
            break
        # P^m_(m + 1) (Schaeffer equation 14)
        pmn1 = amn[m, m + 1] * x * pmn2
        alm[m, m + 1] += ga * pmn1
        for n in range(m + 2, bandwidth):
            # P^m_n (Schaeffer equation 15)
            pmn = amn[m, n] * x * pmn1 - bmn[m, n] * pmn2
            pmn2 = pmn1
            pmn1 = pmn
            if (n + m) % 2 == 0:
                alm[m, n] += gs * pmn
            else:
                alm[m, n] += ga * pmn


@njit(cache=True, nogil=True)
def _synthesize_ring_kernel(
    alm: np.ndarray,
    x: float,
    amn: np.ndarray,
    bmn: np.ndarray,
    bandwidth: int,
    m_lim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric and antisymmetric spectra of one ring from
    the harmonic coefficients.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type.
    x
        Cosine of the ring latitude.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).
    m_lim
        Number of orders to evaluate, ``min(bandwidth, 4 * y + 1)``
        for ring ``y``.

    Returns
    -------
    f_sym
        Sum over the even ``l + m`` modes, of shape ``(m_lim,)`` and
        128-bit complex data type. The Condon-Shortley phase is *not*
        applied here.
    f_asym
        Sum over the odd ``l + m`` modes, of the same shape and data
        type.

    Notes
    -----
    This is the Legendre recursion shared by the Numba ring DFT path
    and the :mod:`scipy.fft` path of
    :class:`SphericalHarmonicTransform`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    f_sym = np.zeros(m_lim, dtype=np.complex128)
    f_asym = np.zeros(m_lim, dtype=np.complex128)
    # (1 - x ** 2) ** (|m| / 2), one for m = 0
    kpmm = 1.0
    r1x2 = math.sqrt(1.0 - x * x)
    for m in range(m_lim):
        # P^m_m (Schaeffer equation 13)
        pmn2 = amn[m, m] * kpmm
        kpmm *= r1x2
        f_sym[m] += alm[m, m] * pmn2
        if m + 1 == bandwidth:  # P^m_(m + 1) does not exist
            break
        # P^m_(m + 1) (Schaeffer equation 14)
        pmn1 = amn[m, m + 1] * x * pmn2
        f_asym[m] += alm[m, m + 1] * pmn1
        for n in range(m + 2, bandwidth):
            # P^m_n (Schaeffer equation 15)
            pmn = amn[m, n] * x * pmn1 - bmn[m, n] * pmn2
            pmn2 = pmn1
            pmn1 = pmn
            if (n + m) % 2 == 0:
                f_sym[m] += alm[m, n] * pmn
            else:
                f_asym[m] += alm[m, n] * pmn
    return f_sym, f_asym


@njit(cache=True, nogil=True)
def _analyze_numba(
    north: np.ndarray,
    south: np.ndarray,
    bandwidth: int,
    cos_lats: np.ndarray,
    weights: np.ndarray,
    ring_offsets: np.ndarray,
    ring_flat: np.ndarray,
    amn: np.ndarray,
    bmn: np.ndarray,
    dft_offsets: np.ndarray,
    dft_cos: np.ndarray,
    dft_sin: np.ndarray,
) -> np.ndarray:
    """Return harmonic coefficients of a spherical function, using
    tabulated ring discrete Fourier transforms.

    Parameters
    ----------
    north, south
        Function values on the northern and southern hemisphere, both
        of shape ``(dim, dim)`` and 64-bit floating point data type,
        in row-major order.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).
    cos_lats
        Cosines of the ring latitudes, of shape ``(n_rings,)``.
    weights
        Ring quadrature weights of shape ``(n_weights, n_rings)`` from
        :func:`~kikuchipy.indexing._spherical._grid.quadrature_weights`.
    ring_offsets, ring_flat
        Ring index arrays from
        :func:`~kikuchipy.indexing._spherical._grid.ring_indices`.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.
    dft_offsets, dft_cos, dft_sin
        Ring transform tables from :func:`_ring_dft_tables`.

    Returns
    -------
    alm
        Harmonic coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type, with
        ``l < m`` entries zero.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_ring = cos_lats.size
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    north_flat = north.ravel()
    south_flat = south.ravel()
    for y in range(n_ring):
        start = ring_offsets[y]
        n_phi = ring_offsets[y + 1] - start
        # Orders beyond l + 1 are not needed and orders beyond the
        # real transform's N_phi(y) / 2 + 1 bins are zero
        m_lim = min(bandwidth, 4 * y + 1)
        ring_north = np.empty(n_phi, dtype=np.float64)
        ring_south = np.empty(n_phi, dtype=np.float64)
        for p in range(n_phi):
            index = ring_flat[start + p]
            ring_north[p] = north_flat[index]
            ring_south[p] = south_flat[index]
        # G_(m, y) by direct, unnormalized transform (Reinecke
        # equation 10), leveraging the real symmetry
        g_sym = np.empty(m_lim, dtype=np.complex128)
        g_asym = np.empty(m_lim, dtype=np.complex128)
        table = dft_offsets[y]
        for m in range(m_lim):
            base = table + m * n_phi
            north_re = 0.0
            north_im = 0.0
            south_re = 0.0
            south_im = 0.0
            for p in range(n_phi):
                cos_mp = dft_cos[base + p]
                sin_mp = dft_sin[base + p]
                north_re += ring_north[p] * cos_mp
                north_im -= ring_north[p] * sin_mp
                south_re += ring_south[p] * cos_mp
                south_im -= ring_south[p] * sin_mp
            # The mod 4 comes from the rings having 8 y points and the
            # real symmetry of the transform; odd orders are negated
            # to re-apply the Condon-Shortley phase the associated
            # Legendre function recursion omits
            weight = weights[m // 4, y]
            if m % 2 == 1:
                weight = -weight
            north_point = complex(north_re * weight, north_im * weight)
            south_point = complex(south_re * weight, south_im * weight)
            # Even l + m are symmetric and odd l + m antisymmetric
            # across the equator
            g_sym[m] = (north_point + south_point) * 0.5
            g_asym[m] = (north_point - south_point) * 0.5
        _analyze_ring_kernel(
            alm, g_sym, g_asym, cos_lats[y], amn, bmn, bandwidth, m_lim
        )
    return alm


@njit(cache=True, nogil=True)
def _synthesize_numba(
    alm: np.ndarray,
    dim: int,
    cos_lats: np.ndarray,
    ring_offsets: np.ndarray,
    ring_flat: np.ndarray,
    amn: np.ndarray,
    bmn: np.ndarray,
    dft_offsets: np.ndarray,
    dft_cos: np.ndarray,
    dft_sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a spherical function from its harmonic coefficients,
    using tabulated ring discrete Fourier transforms.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type.
    dim
        Side length of the square grid.
    cos_lats
        Cosines of the ring latitudes, of shape ``(n_rings,)``.
    ring_offsets, ring_flat
        Ring index arrays from
        :func:`~kikuchipy.indexing._spherical._grid.ring_indices`.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.
    dft_offsets, dft_cos, dft_sin
        Ring transform tables from :func:`_ring_dft_tables`.

    Returns
    -------
    north, south
        Function values on the northern and southern hemisphere, both
        of shape ``(dim, dim)`` and 64-bit floating point data type,
        in row-major order.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    bandwidth = alm.shape[0]
    n_ring = cos_lats.size
    north = np.zeros(dim * dim, dtype=np.float64)
    south = np.zeros(dim * dim, dtype=np.float64)
    for y in range(n_ring):
        start = ring_offsets[y]
        n_phi = ring_offsets[y + 1] - start
        m_lim = min(bandwidth, 4 * y + 1)
        f_sym, f_asym = _synthesize_ring_kernel(
            alm, cos_lats[y], amn, bmn, bandwidth, m_lim
        )
        # F_(m, y) +- F_(m, Nt - 1 - y) -> F_(m, y) and
        # F_(m, Nt - 1 - y), negating odd orders to re-apply the
        # Condon-Shortley phase
        spectrum_north = np.empty(m_lim, dtype=np.complex128)
        spectrum_south = np.empty(m_lim, dtype=np.complex128)
        for m in range(m_lim):
            phase = -1.0 if m % 2 == 1 else 1.0
            sigma = f_sym[m] * phase
            delta = f_asym[m] * phase
            spectrum_north[m] = sigma + delta
            spectrum_south[m] = sigma - delta
        # Unnormalized complex to real transform (Reinecke equation
        # 7). Bins at and beyond m_lim are zero, the zero bin has no
        # mirror bin and the Nyquist bin of an even length real
        # transform is its own mirror bin and structurally real
        ring_north = np.zeros(n_phi, dtype=np.float64)
        ring_south = np.zeros(n_phi, dtype=np.float64)
        table = dft_offsets[y]
        for m in range(m_lim):
            base = table + m * n_phi
            if m == 0 or 2 * m == n_phi:
                north_re = spectrum_north[m].real
                south_re = spectrum_south[m].real
                for p in range(n_phi):
                    cos_mp = dft_cos[base + p]
                    ring_north[p] += north_re * cos_mp
                    ring_south[p] += south_re * cos_mp
            else:
                north_re = spectrum_north[m].real
                north_im = spectrum_north[m].imag
                south_re = spectrum_south[m].real
                south_im = spectrum_south[m].imag
                for p in range(n_phi):
                    cos_mp = dft_cos[base + p]
                    sin_mp = dft_sin[base + p]
                    ring_north[p] += 2.0 * (north_re * cos_mp - north_im * sin_mp)
                    ring_south[p] += 2.0 * (south_re * cos_mp - south_im * sin_mp)
        for p in range(n_phi):
            index = ring_flat[start + p]
            north[index] = ring_north[p]
            south[index] = ring_south[p]
    return north.reshape(dim, dim), south.reshape(dim, dim)


# ------------------------ scipy.fft helpers ------------------------- #


def _analyze_rfft(
    north: np.ndarray,
    south: np.ndarray,
    bandwidth: int,
    cos_lats: np.ndarray,
    weights: np.ndarray,
    ring_offsets: np.ndarray,
    ring_flat: np.ndarray,
    amn: np.ndarray,
    bmn: np.ndarray,
) -> np.ndarray:
    """Return harmonic coefficients of a spherical function, using
    :func:`scipy.fft.rfft` per ring.

    Parameters
    ----------
    north, south
        Function values on the northern and southern hemisphere, both
        of shape ``(dim, dim)`` and 64-bit floating point data type,
        in row-major order.
    bandwidth
        Bandwidth (exclusive maximum harmonic degree).
    cos_lats
        Cosines of the ring latitudes, of shape ``(n_rings,)``.
    weights
        Ring quadrature weights of shape ``(n_weights, n_rings)``.
    ring_offsets, ring_flat
        Ring index arrays from
        :func:`~kikuchipy.indexing._spherical._grid.ring_indices`.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.

    Returns
    -------
    alm
        Harmonic coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type.

    Notes
    -----
    The forward ring transform is unnormalized, matching FFTW's
    ``r2c``, because the ``1 / N_phi(y)`` factor is already folded
    into the ring weights. Every :mod:`scipy.fft` call passes
    ``workers=1`` so that Dask threads are not oversubscribed.
    """
    n_ring = cos_lats.size
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    north_flat = north.reshape(-1)
    south_flat = south.reshape(-1)
    orders = np.arange(bandwidth)
    # Odd orders are negated to re-apply the Condon-Shortley phase the
    # associated Legendre function recursion omits
    phases = np.where(orders % 2 == 1, -1.0, 1.0)
    for y in range(n_ring):
        start = ring_offsets[y]
        stop = ring_offsets[y + 1]
        index = ring_flat[start:stop]
        m_lim = min(bandwidth, 4 * y + 1)
        spectrum_north = rfft(north_flat[index], workers=1)[:m_lim]
        spectrum_south = rfft(south_flat[index], workers=1)[:m_lim]
        # The mod 4 comes from the rings having 8 y points and the
        # real symmetry of the transform
        weight = weights[orders[:m_lim] // 4, y] * phases[:m_lim]
        north_point = spectrum_north * weight
        south_point = spectrum_south * weight
        _analyze_ring_kernel(
            alm,
            (north_point + south_point) * 0.5,
            (north_point - south_point) * 0.5,
            cos_lats[y],
            amn,
            bmn,
            bandwidth,
            m_lim,
        )
    return alm


def _synthesize_rfft(
    alm: np.ndarray,
    dim: int,
    cos_lats: np.ndarray,
    ring_offsets: np.ndarray,
    ring_flat: np.ndarray,
    amn: np.ndarray,
    bmn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a spherical function from its harmonic coefficients,
    using :func:`scipy.fft.irfft` per ring.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type.
    dim
        Side length of the square grid.
    cos_lats
        Cosines of the ring latitudes, of shape ``(n_rings,)``.
    ring_offsets, ring_flat
        Ring index arrays from
        :func:`~kikuchipy.indexing._spherical._grid.ring_indices`.
    amn, bmn
        Recursion coefficients from :func:`_alf_recursion_tables`.

    Returns
    -------
    north, south
        Function values on the northern and southern hemisphere, both
        of shape ``(dim, dim)`` and 64-bit floating point data type,
        in row-major order.

    Notes
    -----
    The inverse ring transform must be unnormalized, matching FFTW's
    ``c2r``, so ``norm="forward"`` is passed to
    :func:`scipy.fft.irfft` (equivalently, its output is multiplied by
    ``N_phi(y)``). Every :mod:`scipy.fft` call passes ``workers=1`` so
    that Dask threads are not oversubscribed.
    """
    bandwidth = alm.shape[0]
    n_ring = cos_lats.size
    north = np.zeros(dim * dim, dtype=np.float64)
    south = np.zeros(dim * dim, dtype=np.float64)
    orders = np.arange(bandwidth)
    phases = np.where(orders % 2 == 1, -1.0, 1.0)
    for y in range(n_ring):
        start = ring_offsets[y]
        stop = ring_offsets[y + 1]
        index = ring_flat[start:stop]
        n_phi = stop - start
        fft_n = n_phi // 2 + 1
        m_lim = min(bandwidth, fft_n)
        f_sym, f_asym = _synthesize_ring_kernel(
            alm, cos_lats[y], amn, bmn, bandwidth, m_lim
        )
        sigma = f_sym * phases[:m_lim]
        delta = f_asym * phases[:m_lim]
        # Bins at and beyond the bandwidth carry no data
        spectrum_north = np.zeros(fft_n, dtype=np.complex128)
        spectrum_south = np.zeros(fft_n, dtype=np.complex128)
        spectrum_north[:m_lim] = sigma + delta
        spectrum_south[:m_lim] = sigma - delta
        north[index] = irfft(spectrum_north, n_phi, norm="forward", workers=1)
        south[index] = irfft(spectrum_south, n_phi, norm="forward", workers=1)
    return north.reshape(dim, dim), south.reshape(dim, dim)


# --------------------------- Transformer ---------------------------- #


class SphericalHarmonicTransform:
    r"""Discrete spherical harmonic transform on a square grid.

    Parameters
    ----------
    bandwidth
        Bandwidth, i.e. the exclusive maximum harmonic degree: degrees
        ``0 <= l < bandwidth`` and orders ``0 <= m <= l`` are
        transformed.
    layout
        Grid layout, either ``"legendre"`` (default) or
        ``"lambert"``.
    dim
        Side length of the square grid, which must be odd and at
        least three. If not given, the smallest usual side length for
        the bandwidth and layout is used, i.e.
        :func:`~kikuchipy.indexing._spherical._grid.default_dim`.

    Attributes
    ----------
    dim : int
        Side length of the square grid.
    bandwidth : int
        Maximum bandwidth of this transformer.
    layout : str
        Grid layout, ``"lambert"`` or ``"legendre"``.
    n_rings : int
        Number of rings from the north pole to the equator.
    cos_latitudes : numpy.ndarray
        Cosines of the ring latitudes, of shape ``(n_rings,)``.
    quadrature_weights : numpy.ndarray
        Ring quadrature weights, of shape
        ``((dim - 2) // 4 + 1, n_rings)``.
    ring_offsets : numpy.ndarray
        Start of each ring in ``ring_indices``, of shape
        ``(n_rings + 1,)``.
    ring_indices : numpy.ndarray
        Flat pixel index of every ring slot, of shape
        ``(dim * dim,)``.

    Raises
    ------
    ValueError
        If ``layout`` is unknown, if ``dim`` is even or smaller than
        three, or if ``bandwidth`` exceeds
        :func:`~kikuchipy.indexing._spherical._grid.max_bandwidth`.

    Notes
    -----
    Conventions frozen by the tests in
    ``tests/test_indexing/test_spherical_sht.py``:

    - Coefficients are stored as ``alm[m, l]``, m-major and l-minor,
      in a ``(bandwidth, bandwidth)`` array of 128-bit complex data
      type. Entries with ``l < m`` are zero.
    - Only non-negative orders are stored; negative orders follow
      from ``a^l_(-m) = (-1) ** m * conj(a^l_m)``.
    - The harmonics are fully normalized, so a function equal to one
      everywhere gives ``alm[0, 0] == sqrt(4 * pi)``.
    - The stored coefficients match those of
      :func:`scipy.special.sph_harm_y`, Condon-Shortley phase
      included: analysing the complex ``Y_l^m`` gives
      ``alm[m, l] == +1``, and analysing ``Re Y_l^m`` gives
      ``alm[m, l] == +0.5`` for ``m > 0`` and ``+1.0`` for ``m == 0``.
      EMSphInx' Legendre recursion omits the phase and re-applies it
      to odd orders when weighting the ring spectrum in ``analyze``
      and when assembling it in ``synthesize``; the net effect is the
      Condon-Shortley convention.
    - ``analyze`` uses an unnormalized forward ring transform, since
      the ``1 / N_phi(y)`` factor is already folded into the ring
      weights ``w_y = 4 pi w_hat_y / N_phi(y)``.
    - ``synthesize`` uses an unnormalized inverse ring transform,
      since FFTW's ``c2r`` applies no ``1 / N_phi(y)`` either.

    The ``"legendre"`` layout uses the ``skip = 0`` weight set for
    every order. Ring ``y = m / 4`` has ``2 * m`` samples, so bin
    ``m`` of its real transform is a structurally real Nyquist bin
    which is then not excluded for orders ``m % 4 == 0``. This gives
    those orders a systematic error of order ``w_(m / 4)`` which
    shrinks with ``dim``, and is why the round trip tolerance of the
    ``"legendre"`` layout is 5e-3 while ``"lambert"`` reaches 1e-13.

    References
    ----------
    :cite:`reinecke2011libpsht`, :cite:`schaeffer2013efficient`,
    :cite:`sneeuw1994global`, :cite:`rosca2010new`,
    :cite:`lenthe2019spherical`
    """

    numba_ring_dft_max_dim: int = 131
    """Largest ``dim`` for which the Numba ring discrete Fourier
    transform path is used; above it the :mod:`scipy.fft` path is
    used. The crossover is keyed on grid size and not bandwidth,
    because the ring transform tables grow as ``dim ** 3``.

    The constructor reads this class attribute once and freezes the
    decision, which :attr:`uses_numba_ring_dft` reports. Tests patch
    the class attribute before constructing a transformer to force
    either path; patching it afterwards has no effect, since the ring
    transform tables are built by the constructor or not at all.
    """

    def __init__(
        self,
        bandwidth: int,
        layout: str = "legendre",
        dim: int | None = None,
    ) -> None:
        if layout not in _grid.LAYOUTS:
            raise ValueError(
                f"Square grid layout {layout!r} must be one of {_grid.LAYOUTS}"
            )
        bandwidth = int(bandwidth)
        if bandwidth < 1:
            raise ValueError(f"Bandwidth {bandwidth} must be at least one")
        if dim is None:
            dim = _grid.default_dim(bandwidth, layout)
        dim = int(dim)
        max_bandwidth = _grid.max_bandwidth(dim, layout)
        if bandwidth > max_bandwidth:
            raise ValueError(
                f"Bandwidth {bandwidth} cannot exceed the largest bandwidth "
                f"{max_bandwidth} of the {layout!r} square grid of side length "
                f"{dim}"
            )

        self.dim = dim
        self.bandwidth = bandwidth
        self.layout = layout
        self.n_rings = _grid.n_rings(dim)
        self.cos_latitudes = _grid.cos_latitudes(dim, layout)
        self.quadrature_weights = _grid.quadrature_weights(dim, layout)
        self.ring_offsets, self.ring_indices = _grid.ring_indices(dim)
        self._amn, self._bmn = _alf_recursion_tables(bandwidth)

        # Freeze the path at construction, so that patching the class
        # attribute afterwards cannot leave a transformer which claims
        # the Numba path but has no ring transform tables
        self._uses_numba_ring_dft = dim <= type(self).numba_ring_dft_max_dim
        if self._uses_numba_ring_dft:
            tables = _ring_dft_tables(dim, bandwidth)
            self._dft_offsets, self._dft_cos, self._dft_sin = tables

    def __repr__(self) -> str:
        """Return a string with the layout, bandwidth and side length,
        e.g. ``"SphericalHarmonicTransform: legendre, bw = 68,
        dim = 71"``.
        """
        return (
            f"{type(self).__name__}: {self.layout}, bw = {self.bandwidth}, "
            f"dim = {self.dim}"
        )

    @property
    def uses_numba_ring_dft(self) -> bool:
        """Return whether the Numba ring discrete Fourier transform
        path is used, as opposed to the :mod:`scipy.fft` path.

        The decision is made once by the constructor, from
        :attr:`numba_ring_dft_max_dim` and :attr:`dim`, and is frozen
        there.
        """
        return self._uses_numba_ring_dft

    def analyze(
        self,
        north: np.ndarray,
        south: np.ndarray,
        bandwidth: int | None = None,
    ) -> np.ndarray:
        """Return the harmonic coefficients of a spherical function
        (forward transform).

        Parameters
        ----------
        north, south
            Function values on the northern and southern hemisphere,
            both of shape ``(dim, dim)`` in row-major order. They are
            cast to 64-bit floating point.
        bandwidth
            Bandwidth to compute, which must not exceed
            :attr:`bandwidth`. If not given, :attr:`bandwidth` is
            used.

        Returns
        -------
        alm
            Harmonic coefficients ``alm[m, l]`` of shape
            ``(bandwidth, bandwidth)`` and 128-bit complex data type,
            with ``l < m`` entries zero.

        Raises
        ------
        ValueError
            If ``north`` or ``south`` does not have shape
            ``(dim, dim)``, or if ``bandwidth`` exceeds
            :attr:`bandwidth`.

        Notes
        -----
        Calling with a smaller ``bandwidth`` gives exactly the
        upper left block of the full result.

        Examples
        --------
        >>> import numpy as np
        >>> from kikuchipy.indexing._spherical._sht import (
        ...     SphericalHarmonicTransform,
        ... )
        >>> sht = SphericalHarmonicTransform(8)
        >>> sht.dim
        11
        >>> north = np.ones((sht.dim, sht.dim))
        >>> alm = sht.analyze(north, north)
        >>> alm.shape
        (8, 8)
        >>> round(float(alm[0, 0].real), 9)  # sqrt(4 * pi)
        3.544907702
        """
        if bandwidth is None:
            bandwidth = self.bandwidth
        bandwidth = int(bandwidth)
        if bandwidth < 1 or bandwidth > self.bandwidth:
            raise ValueError(
                f"Bandwidth {bandwidth} must be in the closed interval "
                f"[1, {self.bandwidth}]"
            )
        north = np.ascontiguousarray(north, dtype=np.float64)
        south = np.ascontiguousarray(south, dtype=np.float64)
        expected = (self.dim, self.dim)
        if north.shape != expected or south.shape != expected:
            raise ValueError(
                f"Hemisphere shapes {north.shape} and {south.shape} must both "
                f"be {expected}"
            )
        if self.uses_numba_ring_dft:
            return _analyze_numba(
                north,
                south,
                bandwidth,
                self.cos_latitudes,
                self.quadrature_weights,
                self.ring_offsets,
                self.ring_indices,
                self._amn,
                self._bmn,
                self._dft_offsets,
                self._dft_cos,
                self._dft_sin,
            )
        return _analyze_rfft(
            north,
            south,
            bandwidth,
            self.cos_latitudes,
            self.quadrature_weights,
            self.ring_offsets,
            self.ring_indices,
            self._amn,
            self._bmn,
        )

    def synthesize(self, alm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a spherical function from its harmonic coefficients
        (inverse transform).

        Parameters
        ----------
        alm
            Harmonic coefficients ``alm[m, l]`` of shape
            ``(bw, bw)`` with ``bw <= bandwidth``. They are cast to
            128-bit complex.

        Returns
        -------
        north, south
            Function values on the northern and southern hemisphere,
            both of shape ``(dim, dim)`` and 64-bit floating point
            data type, in row-major order.

        Raises
        ------
        ValueError
            If ``alm`` is not square or its side exceeds
            :attr:`bandwidth`.

        Notes
        -----
        The function is assumed real, so only non-negative orders are
        used: order ``-m`` contributes
        ``(-1) ** m * conj(alm[m, l])``.
        """
        alm = np.ascontiguousarray(alm, dtype=np.complex128)
        if alm.ndim != 2 or alm.shape[0] != alm.shape[1]:
            raise ValueError(
                f"Coefficient shape {alm.shape} must be square and 2D, i.e. (bw, bw)"
            )
        if alm.shape[0] < 1 or alm.shape[0] > self.bandwidth:
            raise ValueError(
                f"Coefficient bandwidth {alm.shape[0]} must be in the closed "
                f"interval [1, {self.bandwidth}]"
            )
        if self.uses_numba_ring_dft:
            return _synthesize_numba(
                alm,
                self.dim,
                self.cos_latitudes,
                self.ring_offsets,
                self.ring_indices,
                self._amn,
                self._bmn,
                self._dft_offsets,
                self._dft_cos,
                self._dft_sin,
            )
        return _synthesize_rfft(
            alm,
            self.dim,
            self.cos_latitudes,
            self.ring_offsets,
            self.ring_indices,
            self._amn,
            self._bmn,
        )
