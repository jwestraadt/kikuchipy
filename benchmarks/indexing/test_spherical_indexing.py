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

import numpy as np

import kikuchipy as kp
from kikuchipy.indexing import MasterPatternHarmonics

# Bandwidth of the benchmark, the ``IndexEBSD`` name list default
BANDWIDTH = 68

# Measured mean score of the nine patterns at that bandwidth with the
# default configuration, i.e. Newton refined (the coarse mean is
# 0.570)
MEAN_SCORE = 0.589

# The constitution's hard floor, patterns per second per core
FLOOR = 2


def test_spherical_indexing(benchmark):
    """Benchmark spherical indexing of nine (60, 60) EBSD patterns
    against one master pattern at a bandwidth of 68.

    The benchmarked callable is the **map level, end to end** signal
    method, mirroring the dictionary indexing benchmark and the call a
    user makes, so it includes the per-call construction of the
    indexer: on this nine-pattern map that is roughly one third of the
    wall time (a measured 0.046-0.096 s of construction against
    0.12-0.18 s of indexing) and it amortizes below one per cent from
    a few hundred patterns on. The pure per-pattern throughput floor
    is asserted in the default test suite instead, on a pre-built
    indexer.

    The floor asserted here is therefore a map level one, and it still
    passes with more than an order of magnitude of margin.

    The call keeps every default, so it includes the Newton
    refinement of ``refine=True``, which costs a measured 5-27 % of
    the coarse wall time and lifts the mean score from 0.570 to
    0.589.
    """
    # Load patterns
    s = kp.data.nickel_ebsd_small()
    s.remove_static_background()
    s.remove_dynamic_background()

    # Load master pattern and transform it, outside the benchmarked
    # callable: this costs about a second and is paid once per session
    # by a real user as well
    mp = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    harmonics = MasterPatternHarmonics.from_master_pattern(mp, bandwidth=BANDWIDTH)

    # Define detector with one projection center
    detector = s.detector.deepcopy()
    detector.pc = detector.pc_average

    xmap = benchmark(
        s.spherical_indexing,
        harmonics=harmonics,
        detector=detector,
        verbose=0,
    )

    # Relaxed check of results, just to make sure results are not way
    # off
    assert xmap.rotations.size == 9
    assert np.isclose(xmap.scores.mean(), MEAN_SCORE, atol=0.03)

    # Map level throughput floor
    assert 9 / benchmark.stats["mean"] >= FLOOR
