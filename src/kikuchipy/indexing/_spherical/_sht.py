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
#   set replicated for the Legendre layout (lines 375-378)
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
# Python/NumPy/Numba for kikuchipy and conveyed under
# GPL-3.0-or-later
# #####################################################################

"""Discrete spherical harmonic transform on square grids.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.
"""

import numpy as np

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
    raise NotImplementedError


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
    they grow as ``dim ** 3``.
    """
    raise NotImplementedError


# ---------------------------- Kernels ------------------------------- #


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    The constructor copies this class attribute onto the instance and
    decides the path once, which :attr:`uses_numba_ring_dft` reports.
    Tests patch the class attribute before constructing a transformer
    to force either path.
    """

    def __init__(
        self,
        bandwidth: int,
        layout: str = "legendre",
        dim: int | None = None,
    ) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the layout, bandwidth and side length,
        e.g. ``"SphericalHarmonicTransform: legendre, bw = 68,
        dim = 71"``.
        """
        raise NotImplementedError

    @property
    def uses_numba_ring_dft(self) -> bool:
        """Return whether the Numba ring discrete Fourier transform
        path is used, as opposed to the :mod:`scipy.fft` path.

        The decision is made once by the constructor, from
        :attr:`numba_ring_dft_max_dim` and :attr:`dim`.
        """
        return self.dim <= self.numba_ring_dft_max_dim

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
        raise NotImplementedError

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
        raise NotImplementedError
