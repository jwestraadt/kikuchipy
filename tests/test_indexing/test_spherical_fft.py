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

"""Tests of ``kikuchipy.indexing._spherical._fft``.

Covers the "FFT sizing" assertions of
``specs/2026-08-16-sht-square-grid-transform/validation.md``:

- ``fast_size(n) >= max(1, n)`` and ``fast_size(n)`` is 13-smooth for
  ``n`` in ``range(0, 1101)``.
- ``fast_size(n)`` is the *smallest* 13-smooth size not smaller than
  ``n`` on the same range, against a brute force oracle.
- ``fast_size(n) == max(1, n)`` for ``n <= 16``.
- Spot values 105, 175, 245 and 315 map to themselves, 109 to 110, 17
  to 18, 0 to 1 and 16 to 16.
- ``fast_size(-1)`` raises ``ValueError``.
- ``fast_bandwidths(16, 512)`` contains the union of the two lists of
  recommended bandwidths in EMSphInx' ``include/modality/ebsd/nml.hpp``
  (lines 298 and 415) but not 55, and every returned bandwidth ``bw``
  satisfies ``fast_size(2 * bw - 1) == 2 * bw - 1``.
"""

import numpy as np
import pytest

from kikuchipy.indexing._spherical import _fft

# Union of the recommended bandwidths in EMSphInx'
# include/modality/ebsd/nml.hpp lines 298 and 415. The two lists differ
# (298 has 32, 38 and 41 but not 122 and the values above 158), so the
# union is asserted.
EMSPHINX_BANDWIDTHS = (
    32,
    38,
    41,
    53,
    63,
    68,
    74,
    88,
    95,
    113,
    122,
    123,
    158,
    172,
    188,
    203,
    221,
    263,
    284,
    313,
)

# Bandwidth for which 2 * bw - 1 == 109 is prime, so it is not fast
NOT_FAST_BANDWIDTH = 55

SMOOTH_PRIMES = (2, 3, 5, 7, 11, 13)

MAX_N = 1101


def _is_13_smooth(n: int) -> bool:
    """Return whether ``n`` is a positive {2,3,5,7,11,13}-smooth
    integer, by trial division.
    """
    if n < 1:
        return False
    for prime in SMOOTH_PRIMES:
        while n % prime == 0:
            n //= prime
    return n == 1


def _smallest_13_smooth_at_least(n: int) -> int:
    """Return the smallest 13-smooth integer not smaller than
    ``max(1, n)``, by brute force.
    """
    size = max(1, n)
    while not _is_13_smooth(size):
        size += 1
    return size


class TestBruteForceOracle:
    def test_oracle_recognizes_13_smooth_and_non_13_smooth_sizes(self):
        assert _is_13_smooth(1)
        assert _is_13_smooth(16)
        assert _is_13_smooth(2 * 3 * 5 * 7 * 11 * 13)
        assert not _is_13_smooth(0)
        assert not _is_13_smooth(17)
        assert not _is_13_smooth(109)
        assert not _is_13_smooth(2 * 17)

    def test_oracle_returns_smallest_13_smooth_not_smaller_than_n(self):
        assert _smallest_13_smooth_at_least(0) == 1
        assert _smallest_13_smooth_at_least(17) == 18
        assert _smallest_13_smooth_at_least(109) == 110
        assert _smallest_13_smooth_at_least(105) == 105


class TestFastSize:
    def test_fast_size_is_13_smooth_and_not_smaller_than_n(self):
        for n in range(MAX_N):
            size = _fft.fast_size(n)
            assert size >= max(1, n), f"fast_size({n}) = {size}"
            assert _is_13_smooth(size), f"fast_size({n}) = {size}"

    def test_fast_size_is_the_smallest_13_smooth_size_not_smaller_than_n(
        self,
    ):
        for n in range(MAX_N):
            assert _fft.fast_size(n) == _smallest_13_smooth_at_least(n), (
                f"fast_size({n}) is not the smallest 13-smooth size"
            )

    @pytest.mark.parametrize("n", list(range(17)))
    def test_fast_size_returns_max_1_n_for_n_not_greater_than_16(self, n):
        assert _fft.fast_size(n) == max(1, n)

    @pytest.mark.parametrize(
        "n, size",
        [
            (0, 1),
            (16, 16),
            (17, 18),
            (105, 105),
            (109, 110),
            (175, 175),
            (245, 245),
            (315, 315),
        ],
    )
    def test_fast_size_spot_values_match_emsphinx(self, n, size):
        assert _fft.fast_size(n) == size

    @pytest.mark.parametrize("n", [-1, -16, -1000])
    def test_fast_size_raises_value_error_for_negative_size(self, n):
        with pytest.raises(ValueError):
            _fft.fast_size(n)


class TestFastBandwidths:
    def test_fast_bandwidths_contains_emsphinx_recommended_bandwidths(self):
        bandwidths = _fft.fast_bandwidths(16, 512)
        for bandwidth in EMSPHINX_BANDWIDTHS:
            assert bandwidth in bandwidths, f"bw = {bandwidth} is missing"

    def test_fast_bandwidths_excludes_bandwidth_55(self):
        bandwidths = _fft.fast_bandwidths(16, 512)
        assert NOT_FAST_BANDWIDTH not in bandwidths

    def test_fast_bandwidths_all_have_an_unpadded_correlation_size(self):
        bandwidths = _fft.fast_bandwidths(16, 512)
        assert bandwidths.size > 0
        for bandwidth in bandwidths:
            size = 2 * int(bandwidth) - 1
            assert _fft.fast_size(size) == size

    def test_fast_bandwidths_are_sorted_and_within_the_requested_bounds(
        self,
    ):
        bandwidths = _fft.fast_bandwidths(16, 512)
        assert np.all(np.diff(bandwidths) > 0)
        assert bandwidths[0] >= 16
        assert bandwidths[-1] <= 512
