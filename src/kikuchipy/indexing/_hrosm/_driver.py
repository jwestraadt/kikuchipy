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
# functionality in this file is derived and adapted from EMsoftOO:
# - The per-grain re-indexing of the program EMHROSM: the grain loop,
#   its skip rules, the misorientation ball centred on each grain's
#   average orientation, the bounding box indexing domain, the copy
#   back of each grain's results and the fill of points that are not
#   re-indexed (EMOpenCLLib/program_mods/mod_HROSM.f90)
# - The dictionary indexing of a bounding box against a sampled ball
#   and its orientation similarity map, OSMDIdriver
#   (EMOpenCLLib/program_mods/mod_DI.f90)

# #####################################################################
# Copyright (c) 2013-2026, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  - Neither the names of Marc De Graef, Carnegie Mellon University nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################

# Changes by the kikuchipy developers, 2026-10-07: ported from Fortran
# to NumPy; defects reproduced only behind ``emsoft_compatible``.

"""Per-grain re-indexing of a crystal map against a misorientation
ball centred on each grain's reference orientation, the driver of
:meth:`~kikuchipy.signals.EBSD.hrosm`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from orix.crystal_map import CrystalMap

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors._ebsd_detector import EBSDDetector
    from kikuchipy.signals.ebsd import EBSD
    from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern

# Largest number of bytes of one chunk of simulated float32 dictionary
# patterns, setting the default number of patterns per iteration
_DICTIONARY_CHUNK_BYTES = 256e6


def _hrosm(
    signal: EBSD,
    xmap: CrystalMap,
    master_pattern: EBSDMasterPattern,
    detector: EBSDDetector,
    energy: int | float | None,
    *,
    grains: Sequence[int] | None = None,
    **keywords,
) -> CrystalMap:
    """Re-index the grains of a crystal map against a misorientation
    ball centred on each grain's reference orientation.

    See :meth:`~kikuchipy.signals.EBSD.hrosm`, which validates the
    input before calling this function.

    Parameters
    ----------
    signal
        EBSD signal with two navigation dimensions.
    xmap
        Crystal map of the signal.
    master_pattern
        Master pattern used to simulate each grain's dictionary.
    detector
        Detector with one projection center (PC) or one PC per map
        point.
    energy
        Energy of the master pattern to use.
    grains
        Labels (1..n) of the grains to re-index. All other grains get
        the fill of grains that are not re-indexed. If not given, all
        grains are considered. Only used in tests.
    **keywords
        The validated keyword arguments of
        :meth:`~kikuchipy.signals.EBSD.hrosm`.

    Returns
    -------
    xmap_out
        Crystal map with one rotation per point and the properties
        described in :meth:`~kikuchipy.signals.EBSD.hrosm`.
    """
    raise NotImplementedError


def _default_n_per_iteration(sig_size: int, ball_size: int) -> int:
    """Return the default number of dictionary patterns simulated and
    matched per iteration.

    Parameters
    ----------
    sig_size
        Number of detector pixels per pattern.
    ball_size
        Number of orientations in the misorientation ball.

    Returns
    -------
    n_per_iteration
        Number of float32 patterns fitting in the chunk byte budget,
        clipped to the range [1, ``ball_size``].
    """
    raise NotImplementedError
