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
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517), from
# ``include/idx/indexer.hpp`` unless stated otherwise:
# - ``Result<Real>`` (lines 54-64), the fill values, the descending
#   ``operator<`` and the top-n insertion, as the packed rows of
#   :func:`_index_chunk`.  One deliberate improvement: the fill loop
#   (lines 218-222) sets ``corr``, ``phase`` and ``qu`` but never
#   ``iq``, and ``ebsdWorkItem`` reuses one result vector across the
#   patterns of a batch (``idx.hpp`` line 406), so a not-indexed
#   point inherits the previous pattern's image quality on that
#   thread; the fill row here carries a deterministic ``iq = 0``
# - ``Indexer<Real>`` (lines 68-181), its stored bandwidth, grid,
#   back-projection rotation and per phase correlators, as
#   :meth:`SphericalIndexer.__init__`
# - ``Indexer<Real>::BatchEstimate()`` (lines 189-205), as
#   :func:`_batch_estimate`
# - ``Indexer<Real>::indexImage()`` (lines 216-270), as
#   :func:`_index_chunk`
# - ``Indexer<Real>::computeHarmonics()`` (lines 312-318), as the
#   preprocess, back-project and analyse steps of :func:`_index_chunk`
# - ``Indexer<Real>::correlate()`` (lines 326-331), as the per phase
#   correlate step of :func:`_index_chunk`
# - the correlator and back-projector wiring of
#   ``IndexingData<Real>::initialize()``
#   (``include/modality/ebsd/idx.hpp``, lines 252-296), as
#   :meth:`SphericalIndexer.__init__`
# - the per pattern failure semantics of ``ebsdWorkItem<Real>``
#   (``include/modality/ebsd/idx.hpp``, lines 382-456), as the guards
#   and the exception arm of :func:`_index_chunk`
# - ``Indexer<Real>::refineImage()`` and ``Indexer<Real>::refine()``
#   (lines 277-306 and 337-345) and the refine-only work items
#   ``msk[i] & 0x02`` of ``ebsdWorkItem`` (``idx.hpp`` lines
#   438-450), as :meth:`SphericalIndexer.refine_patterns` and
#   :func:`_refine_chunk`, **with a documented deviation**: the
#   shipped ``refineImage()`` drops the ``Result`` its ``refine()``
#   call returns (line 296) and takes ``eu`` by const reference, so
#   it converts the *unrefined* orientation back and stores a
#   ``corr`` the refine-only branch never assigned.  That score is
#   zero or stale rather than indeterminate, since
#   ``std::vector<Result> res(om.size())`` value-initialises and is
#   hoisted outside the per pattern loop (``idx.hpp`` lines
#   406-407), so a pure ``msk & 0x02`` run stores 0 and a mixed
#   ``0x01``/``0x02`` batch stores the previous pattern's score.
#   This port implements the documented intent instead: it refines
#   the stored orientation and stores the refined score
#
# ``include/idx/base.hpp`` (lines 40-150) declares the abstract
# ``ImageProcessor``, ``BackProjector`` and ``PhaseCorrelator``
# interfaces the indexer holds.  They are **collapsed** here into the
# concrete kikuchipy classes ``_preprocessing._preprocess_pattern``,
# ``_back_projection.SphericalBackProjector`` and the two correlators
# of ``_xcorr``, so no interface hierarchy is ported.
#
# The following are deliberately **not** ported here (the roadmap
# phases are named in this provenance comment only -- decision 6.14
# keeps them out of every public docstring and error message):
# - the pseudo-symmetry loop of ``indexImage()`` (lines 243-261),
#   which needs the pseudo-symmetric operator lists ``pSym`` this
#   release always leaves empty -- **Phase 8**.  The insertion
#   machinery it shares with the phase loop **is** ported
# - ``Geometry<Real>::northPoleQuat()``'s left multiplication (line
#   267), the identity in EMSphInx as shipped, so the conversion of
#   lines 265-269 collapses to ``_euler.rotation_from_zyz``
# - the HDF5, PNG and vendor file output of ``IndexingData``
#   (``idx.hpp`` lines 313-370), the ``roimask`` region of interest
#   grammar and ``ThreadedIqCalc``: a crystal map replaces them,
#   kikuchipy's ``navigation_mask`` replaces the region of interest
#   and the image quality is computed in the indexing pass itself
# - ``ThreadPool``, replaced by dask's threaded scheduler

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
# Changed by Johan Westraadt, 2026-09: translated to
# Python/NumPy/dask for kikuchipy, with the ``refineImage()``
# deviation stated above. GPL-2.0-or-later, conveyed under
# GPL-3.0-or-later
# #####################################################################

"""The spherical indexer: per-pattern pipeline, multi-phase top-n
bookkeeping and dask chunking.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**The per-pattern pipeline** (frozen), in EMSphInx' order:

.. code-block::

    processed = _preprocess_pattern(raw, good_pixels, ...)
    north, south, iq = projector.unproject(
        processed, out=(north, south), return_image_quality=True
    )
    gln = projector.sht.analyze(north, south)
    for every phase p:
        zyz, score = correlator_p.correlate(gln)
    rotation = _euler.rotation_from_zyz(zyz)

The image quality is the discrete cosine image quality of the
**processed** pattern, not of the raw one: measured on the nine
background corrected Ni patterns 0.1727-0.2036 with the default
``n_regions = 10``, against 0.2890-0.3269 with ``n_regions = 0`` and
0.766-0.779 for the raw patterns, three separable bands.

The north and south buffers MUST be **zeroed before their first
use**: ``unproject`` writes only the window points of ``north`` and
never touches ``south``, so uninitialised memory would otherwise
reach the spherical harmonic analysis.  Re-zeroing them between
patterns is defensive only, since every window point is assigned on
every call (measured: zeroing once per chunk and once per pattern
give bitwise equal spectra over the nine patterns).

**The result contract** (frozen).  Every output row starts at the
fill value ``zyz = (0, 0, 0)`` (the identity rotation), ``score =
0``, ``phase_id = -1`` and ``iq = 0``.  A pattern is marked **failed**
-- all of its rows keep the fill -- when

(a) the raw pattern has ``ptp == 0`` (zero variance input),
(b) the processed pattern has ``ptp == 0`` (the pipeline degenerated
    to a constant, so the back-projection would take its mask branch
    and the correlation would see the window mask),
(c) the winning score or any winning Euler angle is not finite,
(d) any exception escapes the per-pattern body, which is caught per
    pattern so that one bad pattern never kills a run, or
(e) no phase scored above zero, which is not a guard but a
    consequence of the insertion rule below.

Guards (a) and (b) are a **deliberate deviation** from EMSphInx for
one input only, and parity for the other.  A constant unsigned 8-bit
pattern does not reach the back-projection's own constant short cut,
because the four term interpolation of the mosaic histogram
equalisation of a uniform image returns ``255 + O(1e-13)``: EMSphInx
normalises that rounding ripple to unit variance and indexes it, at a
measured image quality of ``0.9999999999999996``, a **score of
+0.2301** and a garbage orientation which silently enters the map.
Guard (a) intercepts it -- the recorded deviation.  A constant
*float* pattern with ``n_regions = 0`` does reach the mask branch and
correlates the window mask at a measured **score of -2.6402**, which
EMSphInx' own insertion rule then drops, so it reports that point not
indexed as well: for that input the port fails the pattern earlier on
the same outcome, which is parity.  A NaN *pixel* is not guarded (it
survives the unsigned 8-bit conversion and indexes near normally,
measured score 0.605 against 0.624 for the clean pattern); a NaN that
reaches the scores is caught by (c).

**One candidate per phase** (frozen).  Each phase contributes exactly
one candidate, its global correlation peak, and candidates are
inserted into the descending top-``n_best`` list with C++
``upper_bound`` semantics, so an equal score ranks after the earlier
phase.  Since the rows are seeded with ``score 0`` and ``phase -1``,
a candidate with a **non-positive score is never recorded**: it sorts
after every fill row and is dropped ("only keep something with a
positive phase").  With ``P`` phases there are at most ``P``
candidates, so rows ``>= P`` of an ``n_best > P`` request keep the
fill.  Secondary peaks of one phase are not extracted; EMSphInx has
no such path.

**Threads and determinism.**  Chunks of patterns are mapped over
dask's threaded scheduler.  The indexer instance is immutable and
holds no per-pattern scratch: the projector, its transform and the
preprocessing functions are read only and shared, while the
correlators are cloned **once per chunk invocation** (measured
``clone()`` cost 0.18 / 0.21 / 0.33 ms at ``bw`` 53 / 68 / 88,
against 12.7 ms for one pattern at ``bw`` 68, so under 2 % even at a
chunk size of one) and the north and south buffers are allocated per
chunk.  The arithmetic of a pattern therefore does not depend on how
the patterns are grouped, and results are **bitwise identical**
across chunk sizes, worker counts and lazy against eager input.

**Chunk sizing.**  With no explicit chunk size the ported
``BatchEstimate`` model is used: with ``scl = bw^3 ln(bw^3)`` and
``k = 1e-8`` seconds, ``batch = max(1, int(1 / (scl k) / phi))``
patterns, then the load balancing rule "at least ``n_workers^2``
batches".  It gives 34 / 15 / 6 patterns at ``bw`` 53 / 68 / 88 for a
large map, and a chunk size of one for a nine pattern map on four
workers, so small maps parallelise.  One recorded deviation: the
result is clamped to at least one, since the verbatim load balancing
branch returns zero for zero patterns.

**Memory.**  One worker holds one chunk kit, a correlator clone per
phase plus the north and south pair.  Measured with
:mod:`tracemalloc` (resident after the clone / after the first
correlation / transient peak, MB):

.. code-block::

    bw    resident   after correlate   peak    model
     53      9.4          14.2         21.4     23.3
     63     15.9          23.8         35.7     39.2
     68     20.0          30.0         44.9     49.4
     88     43.4          65.1         97.6    107.6
    113     91.9         137.7        207.2    228.4

:attr:`SphericalIndexer.memory_per_worker_bytes` exposes the model
``(n_phases if normalize else 1) slP^2 bwP 24 + slP^3 8`` bytes,
which is above the measured peak by about 10 % (8.9, 9.8, 10.0, 10.2
and 10.2 % over the five rows of the table); with
``normalize=False`` one shared scratch correlator serves every phase,
which is what the factor expresses.
:meth:`SphericalIndexer.index_patterns` warns when the model times
the worker count exceeds 2 GiB.

A **refining** run adds ``(n_phases if normalize else 1) 16 bw^3``
bytes per worker, one ``(bw, bw, bw, 2)`` Wigner d table per
correlator clone -- 5.03, 10.9 and 23.1 MB each at ``bw`` 68, 88 and
113 -- so the model reads 54,457,112 B at ``bw`` 68 with one phase
and 89,231,224 B with two normalized phases.  The beta independent
factor triple, another ``8 bw^2 + 16 bw^3`` bytes (5.07 MB at ``bw``
68), is read only and shared by every correlator of one process.

**Resizing against direct construction.**  Every phase is stored as
``harmonics.resize(bandwidth)``, exactly as ``IndexEBSD`` resizes the
stored spectra of a ``.sht`` file.  Resizing is **not** the same as
building the harmonics at the target bandwidth, because the weighted
normalisation depends on the analysis bandwidth: measured for the Ni
master, ``resize(120 -> 68)`` differs from a direct construction at
``bw`` 68 by 2.9 % of the largest coefficient, 10.3 % in relative
L2 norm and by up to about 100 % on individual significant
coefficients.  Parity runs must therefore resize from the stored
bandwidth, as ``IndexEBSD`` does.

References
----------
:cite:`lenthe2019spherical`
"""

from __future__ import annotations

# The collaborators are bound in this namespace, as the rest of the
# package binds its own, so that one place is patched in tests and
# one import block changes if a collaborator moves.
import math
import os
from typing import TYPE_CHECKING, Sequence
from warnings import warn

import dask
import dask.array as da
from dask.diagnostics.progress import ProgressBar
import numpy as np

from kikuchipy.indexing._spherical._back_projection import SphericalBackProjector
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._preprocessing import (
    _circular_mask,
    _preprocess_pattern,
)
from kikuchipy.indexing._spherical._wigner import (
    wigner_d_half_pi_table,
    wigner_d_table_factors,
)
from kikuchipy.indexing._spherical._xcorr import (
    NormalizedSphericalCrossCorrelator,
    SphericalCrossCorrelator,
)

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors import EBSDDetector

# The golden ratio reciprocal of ``BatchEstimate()``
# (``indexer.hpp`` line 199), spelled out to the C++ literal's digits
_INVERSE_GOLDEN_RATIO = 0.61803398874989484820458683436564

# Seconds per unit of ``bw^3 ln(bw^3)``, the C++ ``k``
# (``indexer.hpp`` line 195)
_BATCH_TIME_SCALE = 1e-8

# Number of packed columns of one result row: the three ZYZ Euler
# angles, the score, the phase identifier and the image quality
_ROW_WIDTH = 6

# Warn when the estimated memory of all workers together exceeds this
_MEMORY_WARNING_BYTES = 2 * 1024**3

# Smallest and largest bandwidth ``IndexEBSD``'s name list accepts
# (``nml.hpp`` line 635)
_BANDWIDTH_LIMITS = (16, 512)


def _batch_estimate(bandwidth: int, n_workers: int, n_patterns: int) -> int:
    """Return a reasonable number of patterns to index per chunk.

    Parameters
    ----------
    bandwidth
        Bandwidth the patterns are indexed at.
    n_workers
        Number of worker threads the chunks are spread over.
    n_patterns
        Number of patterns to index.

    Returns
    -------
    chunksize
        Number of patterns per chunk, at least one.

    Notes
    -----
    Port of ``Indexer<Real>::BatchEstimate()``
    (``include/idx/indexer.hpp``, lines 189-205): a chunk which takes
    about one over the golden ratio seconds on one thread, at the C++
    complexity model ``bw^3 ln(bw^3)`` and its calibration constant
    ``k = 1e-8`` s, followed by the load balancing rule which shrinks
    the chunk until there are at least ``n_workers^2`` of them.  The
    C++ truncates the first estimate towards zero and rounds the
    second up, which is reproduced.

    One recorded **deviation**: the result is clamped to at least one.
    The verbatim load balancing branch returns ``ceil(0 / nt^2) = 0``
    for zero patterns, which the C++ never meets because its own mask
    fallback guarantees at least one pattern.

    Measured for a large map: 34 / 15 / 6 patterns at ``bw`` 53 / 68 /
    88 on eight workers, and one pattern per chunk for a nine pattern
    map on four workers, so small maps parallelise.
    """
    # An estimate for many more patterns than threads first, at the
    # C++ complexity model and its calibration constant (lines
    # 191-196)
    bandwidth_cubed = float(bandwidth * bandwidth * bandwidth)
    scale = bandwidth_cubed * math.log(bandwidth_cubed)
    patterns_per_second = 1.0 / (scale * _BATCH_TIME_SCALE)
    # A chunk which takes about one over the golden ratio seconds, so
    # that it does not synchronise with the progress updates (line
    # 197).  The C++ truncates towards zero here
    chunksize = max(1, int(patterns_per_second * _INVERSE_GOLDEN_RATIO))

    # Then shrink it until there are enough chunks to balance the
    # load, rounding up this time (lines 200-204)
    if math.ceil(n_patterns / chunksize) < n_workers * n_workers:
        chunksize = math.ceil(n_patterns / (n_workers * n_workers))

    # The recorded deviation: the branch above returns zero for zero
    # patterns, which the C++ never meets
    return max(1, int(chunksize))


def _n_workers() -> int:
    """Return the number of worker threads the chunks are spread over.

    Returns
    -------
    n_workers
        The active dask configuration's ``num_workers`` when it is
        set, so that an outer :func:`dask.config.set` is honoured as
        it is by the scheduler itself, and the processor count
        otherwise.  At least one.
    """
    n_workers = dask.config.get("num_workers", None)
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    return max(1, int(n_workers))


def _phase_name(harmonics: "MasterPatternHarmonics") -> str:
    """Return the name of a phase, ``"?"`` when it carries none."""
    if harmonics.phase is None:
        return "?"
    return str(harmonics.phase.name)


def _phase_description(harmonics: "MasterPatternHarmonics") -> str:
    """Return the name, point group and symmetry flags of a phase,
    e.g. ``"ni (m-3m; 4-fold, mirror)"``.
    """
    phase = harmonics.phase
    point_group = None if phase is None else phase.point_group
    name = "?" if point_group is None else str(point_group.name)
    mirror = "mirror" if harmonics.has_equatorial_mirror else "no mirror"
    return f"{_phase_name(harmonics)} ({name}; {harmonics.n_fold}-fold, {mirror})"


def _insert_candidate(
    rows: np.ndarray,
    zyz: np.ndarray,
    score: float,
    phase_id: int,
    image_quality: float,
) -> None:
    """Insert one candidate into a descending list of result rows.

    Parameters
    ----------
    rows
        ``(n_best, 6)`` packed rows, descending in the score column,
        modified in place.
    zyz
        The candidate's three ZYZ Euler angles.
    score
        The candidate's correlation.
    phase_id
        Index of the phase the candidate came from.
    image_quality
        Image quality of the pattern, which every candidate of one
        pattern carries.

    Notes
    -----
    Port of the ``std::upper_bound`` insertion of
    ``Indexer<Real>::indexImage()`` (``include/idx/indexer.hpp``,
    lines 235-239) under the descending ``Result::operator<``
    (line 63).  The candidate goes where it **strictly beats** an
    existing row, so an equal score ranks after the earlier phase and
    a candidate which beats no row is dropped.  Since the rows are
    seeded with a score of zero, that drops every candidate with a
    non-positive score, which is the C++ "only keep something with a
    positive phase" (line 219).  A NaN score loses every comparison
    and is dropped as well.
    """
    n_best = rows.shape[0]
    index = n_best
    for i in range(n_best):
        if score > rows[i, 3]:
            index = i
            break
    if index >= n_best:
        return
    # Shift the rows below the insertion point down by one, from the
    # back, then write the candidate
    for i in range(n_best - 1, index, -1):
        rows[i] = rows[i - 1]
    rows[index, 0] = zyz[0]
    rows[index, 1] = zyz[1]
    rows[index, 2] = zyz[2]
    rows[index, 3] = score
    rows[index, 4] = phase_id
    rows[index, 5] = image_quality


def _index_chunk(
    patterns_block: np.ndarray, indexer: "SphericalIndexer", n_best: int
) -> np.ndarray:
    """Return the packed indexing results of one chunk of patterns.

    Parameters
    ----------
    patterns_block
        ``(nc, nrows, ncols)`` array of patterns of any real data
        type.
    indexer
        Indexer whose projector, preprocessing configuration and
        correlators to use.  It is only read: the correlators are
        cloned and the buffers allocated in this call, so one indexer
        serves every worker.
    n_best
        Number of candidates to keep per pattern, at least one.

    Returns
    -------
    results
        ``(nc, n_best, 6)`` 64-bit float array packing ``alpha``,
        ``beta``, ``gamma``, ``score``, ``phase_id`` and ``iq`` of
        every candidate, descending in score.  Rows which no
        candidate reached carry the fill values ``(0, 0, 0, 0, -1,
        0)``, and the image quality is repeated over the candidates
        of one pattern.

    Notes
    -----
    Port of ``Indexer<Real>::indexImage()``
    (``include/idx/indexer.hpp``, lines 216-270) without its
    pseudo-symmetry loop, of ``computeHarmonics()`` (lines 312-318)
    and of ``correlate()`` (lines 326-331), with the per pattern
    failure semantics of ``ebsdWorkItem<Real>``
    (``include/modality/ebsd/idx.hpp``, lines 382-456).

    ``indexer.refine`` is handed to **every** phase's ``correlate``,
    which is where ``indexImage()`` puts it (line 230): each phase's
    single candidate is refined *before* insertion, so the top-n
    ordering uses refined scores and a run with ``P`` phases pays
    ``P`` refinements per pattern.  Fill rows carry no candidate and
    are never refined.

    The correlators are cloned and the north and south buffers
    allocated **once per invocation**, with :func:`numpy.zeros` and
    never :func:`numpy.empty`, so that the chunk worker is stateless
    and no buffer is shared between threads.  See the module
    documentation for the pipeline, the fill values, the five failure
    cases and the insertion rule which never records a candidate with
    a non-positive score.
    """
    n_patterns = int(patterns_block.shape[0])
    # Every row starts at the fill value, so a pattern which is
    # failed anywhere below simply keeps it
    results = np.zeros((n_patterns, n_best, _ROW_WIDTH))
    results[:, :, 4] = -1.0

    projector = indexer.projector
    dim = projector.dim
    # Zeroed, never ``numpy.empty``: ``unproject`` writes only the
    # window points of the first buffer and never touches the second
    buffers = (np.zeros((dim, dim)), np.zeros((dim, dim)))

    # One clone per invocation, so that no scratch is shared between
    # threads and the worker itself stays stateless
    if indexer.normalize:
        correlators = [c.clone() for c in indexer.correlators]
        prototype = None
        spectra = None
    else:
        correlators = None
        prototype = indexer.correlator.clone()
        spectra = indexer.spectra

    good_pixels = indexer.good_pixels
    gaussian_background = indexer.gaussian_background
    n_regions = indexer.n_regions
    compatible = indexer.emsphinx_compatible
    refine = indexer.refine

    for i in range(n_patterns):
        # Every guard sits inside the catch, as ``ebsdWorkItem``
        # wraps the whole of ``indexImage()``: a pattern which raises
        # anywhere, the guards included, fails alone
        try:
            pattern = patterns_block[i]
            # (a) a zero variance pattern is failed, not indexed
            if np.ptp(pattern) == 0:
                continue

            processed = _preprocess_pattern(
                pattern,
                good_pixels=good_pixels,
                gaussian_background=gaussian_background,
                n_regions=n_regions,
                emsphinx_compatible=compatible,
            )
            # (b) the pipeline degenerated to a constant, which the
            # back-projection would answer with its window mask
            if np.ptp(processed) == 0:
                continue

            north, south, image_quality = projector.unproject(
                processed, out=buffers, return_image_quality=True
            )
            gln = projector.sht.analyze(north, south)

            rows = np.zeros((n_best, _ROW_WIDTH))
            rows[:, 4] = -1.0
            if correlators is not None:
                for phase_id, correlator in enumerate(correlators):
                    zyz, score = correlator.correlate(
                        gln, refine=refine, emsphinx_compatible=compatible
                    )
                    _insert_candidate(rows, zyz, score, phase_id, image_quality)
            else:
                for phase_id, (alm, n_fold, mirror) in enumerate(spectra):
                    zyz, score = prototype.correlate(
                        alm,
                        gln,
                        n_fold,
                        mirror,
                        refine=refine,
                        emsphinx_compatible=compatible,
                    )
                    _insert_candidate(rows, zyz, score, phase_id, image_quality)

            # (c) a winning score or angle which is not finite
            if not np.isfinite(rows[0, :4]).all():
                continue
            results[i] = rows
        except Exception:
            # (d) one bad pattern never kills the run, the
            # ``ebsdWorkItem`` catch
            continue

    return results


def _map_chunks(
    patterns_da: da.Array, indexer: "SphericalIndexer", n_best: int
) -> da.Array:
    """Return the lazy packed indexing results of a chunked pattern
    array.

    Parameters
    ----------
    patterns_da
        ``(n, nrows, ncols)`` dask array chunked along the first axis
        only.
    indexer
        Indexer to pass to :func:`_index_chunk`.
    n_best
        Number of candidates to keep per pattern.

    Returns
    -------
    results
        ``(n, n_best, 6)`` lazy 64-bit float array with the chunks of
        ``patterns_da`` along the first axis.

    Notes
    -----
    The ``chunks=`` argument is given explicitly rather than left to
    :func:`dask.array.map_blocks` to infer.  Without it the graph
    declares the shape ``(n, 1, 1)`` -- measured -- which computes
    correctly but lies to anything which slices or inspects the array
    before computing it.
    """
    return patterns_da.map_blocks(
        _index_chunk,
        indexer,
        n_best,
        dtype=np.float64,
        drop_axis=(1, 2),
        new_axis=(1, 2),
        chunks=(patterns_da.chunks[0], (n_best,), (_ROW_WIDTH,)),
    )


def _refine_chunk(
    patterns_block: np.ndarray,
    zyz_block: np.ndarray,
    phase_id_block: np.ndarray,
    indexer: "SphericalIndexer",
) -> np.ndarray:
    """Return the packed refinement results of one chunk of patterns
    and their starting orientations.

    Parameters
    ----------
    patterns_block
        ``(nc, nrows, ncols)`` array of patterns of any real data
        type.
    zyz_block
        ``(nc, 3)`` 64-bit float starting passive ZYZ Euler angles,
        **block aligned** with ``patterns_block``.
    phase_id_block
        ``(nc,)`` starting phase indices into
        :attr:`SphericalIndexer.phases`, negative where the point is
        not indexed and must pass through untouched.
    indexer
        Indexer whose projector, preprocessing configuration and
        correlators to use.  It is only read: the correlators are
        cloned and the buffers allocated in this call, so one indexer
        serves every worker.

    Returns
    -------
    results
        ``(nc, 6)`` 64-bit float array packing ``alpha``, ``beta``,
        ``gamma``, ``score``, ``phase_id`` and ``iq`` of every
        pattern.  A row which is not refined -- a negative phase
        index, a failed guard or a raising pattern -- carries its
        input angles and phase with a score and image quality of
        zero, so the caller can leave the input map's values in
        place there.

    Notes
    -----
    Port of the refine-only work item of ``ebsdWorkItem<Real>``
    (``include/modality/ebsd/idx.hpp``, lines 438-450) with the
    **intended** semantics of ``Indexer<Real>::refineImage()``
    (``include/idx/indexer.hpp``, lines 277-306); see the licence
    notice at the top of this module for the defect this deviates
    from.  Per pattern: preprocess, back-project and analyse exactly
    as :func:`_index_chunk` does, with the same two guards and the
    same per pattern exception arm, then refine through **that
    phase's** correlator (line 296) and recompute the image quality
    (lines 280 and 304).

    A non-converged refinement is **not** a failure: it returns the
    input triple with the analytic value there, which is C++ parity.

    ``zyz_block`` and ``phase_id_block`` are block arguments of
    :func:`dask.array.map_blocks` chunked to the pattern blocks, so a
    chunk always refines from the starting orientations of its own
    patterns.
    """
    raise NotImplementedError(
        "`_refine_chunk()` is not implemented yet; it needs the correlators' "
        "`refine_zyz()`"
    )


class SphericalIndexer:
    """Indexing of EBSD patterns by spherical cross-correlation with
    one or more master patterns :cite:`lenthe2019spherical`.

    This is the kikuchipy equivalent of EMSphInx' ``IndexEBSD``
    program, and a call with no keyword arguments reproduces its name
    list defaults.

    Parameters
    ----------
    harmonics
        One
        :class:`~kikuchipy.indexing.MasterPatternHarmonics` or a
        sequence of them, one per phase.  Each is resized to
        ``bandwidth``.  Every phase with both values set must agree
        on ``sample_tilt`` and ``beam_energy``, and the shared
        ``sample_tilt`` must equal ``detector.sample_tilt``.
    detector
        EBSD detector with exactly one projection centre and zero
        ``azimuthal`` and ``twist`` angles.  It is deep copied by the
        back-projector, so later mutations of the caller's detector
        do not reach the indexer.
    bandwidth
        Bandwidth, i.e. the exclusive maximum harmonic degree, to
        index at, between 16 and 512.  Default is 68, the
        ``IndexEBSD`` name list default.  Prefer a value from
        :func:`~kikuchipy.indexing.fast_bandwidths`.
    normalize
        Whether to use the normalized cross-correlation of Huhle,
        which divides by a rotation dependent denominator computed
        from the window.  Default is ``True``, the ``IndexEBSD`` name
        list default.
    refine
        Whether to refine every phase's best orientation of the Euler
        grid with Newton's method on the sphere.  Default is
        ``True``, the ``IndexEBSD`` name list default; pass ``False``
        for the coarse grid result alone.  A refined score is the
        analytic correlation at the refined rotation and is **not**
        comparable with a coarse one, see the ``Notes``.
    signal_mask
        Boolean mask of the detector shape in kikuchipy polarity,
        ``True`` = ignore the pixel, as in
        :meth:`~kikuchipy.signals.EBSD.dictionary_indexing`.  It
        excludes sphere points by nearest pixel and excludes the
        pixels from the histogram equalisation.  ``None`` by default.
    n_regions
        Number of mosaic tiles along each axis of the adaptive
        histogram equalisation, between 0 and the smallest detector
        side.  Default is 10, the ``IndexEBSD`` name list default;
        ``0`` skips the equalisation.
    gaussian_background
        Whether to fit and subtract a separable Gaussian background
        before the equalisation.  Default is ``False``, the
        ``IndexEBSD`` name list default.
    circular_mask
        Whether to keep only the largest circle inscribed in the
        detector, both on the sphere and in the histogram.  Default
        is ``False``, the ``IndexEBSD`` name list default
        ``circmask = -1``.
    emsphinx_compatible
        Whether to reproduce EMSphInx' Gaussian fit off-by-one and
        the two defects of its peak interpolation.  Default is
        ``True``.  Parity also needs
        ``MasterPatternHarmonics.from_master_pattern(...,
        emsphinx_compatible=True)``, see the ``Notes``.

    Attributes
    ----------
    phases : tuple of MasterPatternHarmonics
        The harmonics, each resized to :attr:`bandwidth`.
    n_phases : int
        Number of phases.
    bandwidth : int
        Exclusive maximum harmonic degree.
    normalize : bool
        Whether the normalized correlation is used.
    refine : bool
        Whether every candidate is Newton refined before insertion.
    wigner_d_factors : tuple or None
        The beta independent Wigner d factor triple every correlator
        of a refining indexer shares, and ``None`` when
        :attr:`refine` is ``False`` and
        :meth:`refine_patterns` has not been called.
    projector : kikuchipy.indexing.SphericalBackProjector
        The shared back-projector, whose ``detector`` attribute is
        the isolated deep copy of the detector.
    correlators : tuple or None
        One normalized cross-correlator per phase when
        :attr:`normalize`, and ``None`` otherwise.
    correlator : object or None
        One shared un-normalised correlator prototype when
        :attr:`normalize` is ``False``, and ``None`` otherwise.
    spectra : tuple or None
        One ``(alm, n_fold, mirror)`` triple per phase when
        :attr:`normalize` is ``False``, and ``None`` otherwise.
    wigner_d_half_pi : numpy.ndarray
        The one ``pi/2`` Wigner d table every correlator shares.
    good_pixels : numpy.ndarray or None
        The detector shaped mask in **good pixel** polarity handed to
        the preprocessing, i.e. ``~signal_mask`` intersected with the
        circle, reduced to the terms present and ``None`` when
        neither is given.
    signal_mask : numpy.ndarray or None
        The mask as given, in kikuchipy polarity.
    circular_mask : bool
        Whether the circle is applied.
    n_regions : int
        Number of mosaic tiles along each axis.
    gaussian_background : bool
        Whether the Gaussian background is removed.
    emsphinx_compatible : bool
        Whether the C++ defects are reproduced.
    side_length : int
        Side length ``slP`` of the Euler cube, 135 at ``bw`` 68.
    half_cell_degrees : float
        Half a grid cell of the Euler cube, ``180 / slP`` degrees,
        1.33 at ``bw`` 68, which bounds the coarse orientation error.

    Raises
    ------
    TypeError
        If an entry of ``harmonics`` is not a
        :class:`~kikuchipy.indexing.MasterPatternHarmonics`, or if
        ``detector`` is not an
        :class:`~kikuchipy.detectors.EBSDDetector`.
    ValueError
        If ``bandwidth`` is outside ``[16, 512]``; if ``harmonics``
        is empty; if two phases disagree on ``sample_tilt`` or
        ``beam_energy``; if the phases' ``sample_tilt`` differs from
        the detector's; if ``n_regions`` is negative or larger than
        the smallest detector side; or for any of the
        back-projector's own geometry and mask errors, which
        propagate unchanged.

    Warns
    -----
    UserWarning
        If a phase's own bandwidth is smaller than ``bandwidth``, so
        that its coefficients are zero padded.

    See Also
    --------
    kikuchipy.signals.EBSD.spherical_indexing
    kikuchipy.indexing.MasterPatternHarmonics
    kikuchipy.indexing.SphericalBackProjector
    kikuchipy.indexing.fast_bandwidths

    Notes
    -----
    Port of ``Indexer<Real>`` (``include/idx/indexer.hpp``) and of the
    wiring and failure semantics of the EBSD driver
    (``include/modality/ebsd/idx.hpp``).  See the module
    documentation for the per-pattern pipeline, the memory model and
    the chunk sizing.

    **Validation order** (frozen), so that an expensive construction
    is never paid for a call which cannot work: the bandwidth range,
    the harmonics sequence and its entry types, the shared
    ``sample_tilt`` and ``beam_energy`` of the phases, the binding of
    that ``sample_tilt`` to the detector's, ``n_regions``, and
    finally the back-projector, whose own geometry and mask errors
    propagate unchanged rather than being duplicated here.

    **The tilt binding** deserves its own note.  EMSphInx builds the
    detector geometry *from* the master pattern, so a mismatch is
    structurally impossible there; here the tilt comes from the
    detector, so it is checked instead.  Measured: harmonics computed
    for a 70 degree sample tilt, indexed against a detector set to
    65 degrees, land 4.680 degrees (median) from the stored
    orientations at *higher* scores than the correct run, so no score
    based sanity check could catch it.

    **The scores are not normalized cross-correlations.**  The metric
    is the correlation of the winning rotation, divided by the
    rotation dependent window denominator when ``normalize`` is
    ``True`` but never by the standard deviation of the pattern, so
    it is unbounded and comparable only within one geometry and
    bandwidth.  Measured for the nine Ni patterns at ``bw`` 68 with
    ``refine=False``: 0.4963-0.6239 normalized and 0.2799-0.3533
    un-normalised; with the ``refine=True`` default 0.5143-0.6347
    resp. 0.2903-0.3592.

    **A refined score is not a coarse score.**  With ``refine``, the
    metric is the *analytic* correlation at the Newton point, over
    ``denominator(zyz)`` when ``normalize`` is ``True``, and not the
    tri-quadratic interpolated peak of the coarse grid, so the two
    are not comparable with one another.  The refined normalized
    score can even dip below the coarse one where the window shift
    chain rule is omitted, exactly as it is omitted in EMSphInx
    (measured 4 of 165 points, worst -4.8e-4).

    **Failure semantics**, the result contract of the module
    documentation.  A failed pattern records the identity rotation, a
    score of ``0``, a phase of ``-1`` and an image quality of ``0``
    in every one of its candidate rows, and a pattern fails when

    - the raw pattern has ``ptp == 0``,
    - the processed pattern has ``ptp == 0``,
    - the winning score or Euler angle is not finite,
    - the pattern raises, which is caught so that the rest of the
      run continues, or
    - no phase scored above zero, which the insertion rule gives
      rather than a guard.

    The first case is a recorded deviation from EMSphInx, which would
    index the rounding ripple of the histogram equalisation of a
    uniform image at a measured score of +0.2301 and a garbage
    orientation; the module documentation states the measurements.

    A refinement does not add a failure case of its own -- a Newton
    loop which does not converge returns the coarse triple with the
    analytic value there, as the C++ does -- but that analytic value
    **can be non-positive**, and the insertion rule then drops the
    candidate.  A pattern whose every phase fails that way becomes a
    failed pattern where the coarse path would have kept a positive
    interpolated score.  Measured: zero refinement failures on every
    real data run, so this is a contract statement rather than an
    observed regression.

    **EMSphInx parity** needs ``emsphinx_compatible=True`` **both**
    here and in
    :meth:`~kikuchipy.indexing.MasterPatternHarmonics.from_master_pattern`,
    since the master's normalisation quirk is frozen into the
    coefficients when they are built and is not recorded on them.

    An instance is **immutable after construction and holds no per
    pattern scratch**, so it may be reused across calls and signals
    of the same detector geometry, and
    :meth:`index_patterns` is reentrant: it clones the correlators
    and allocates the buffers per chunk.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> from kikuchipy.indexing import (
    ...     MasterPatternHarmonics,
    ...     SphericalIndexer,
    ... )
    >>> detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    >>> detector.pc = detector.pc_average
    >>> master = kp.data.nickel_ebsd_master_pattern_small(
    ...     projection="lambert", hemisphere="both"
    ... )
    >>> harmonics = MasterPatternHarmonics.from_master_pattern(
    ...     master, bandwidth=68
    ... )
    >>> indexer = SphericalIndexer(harmonics, detector)
    >>> indexer
    SphericalIndexer: 1 phase (ni), bw = 68, sphere window 1317
    points (14.6 %), normalized

    References
    ----------
    :cite:`lenthe2019spherical`
    """

    def __init__(
        self,
        harmonics: "MasterPatternHarmonics | Sequence[MasterPatternHarmonics]",
        detector: "EBSDDetector",
        *,
        bandwidth: int = 68,
        normalize: bool = True,
        refine: bool = True,
        signal_mask: np.ndarray | None = None,
        n_regions: int = 10,
        gaussian_background: bool = False,
        circular_mask: bool = False,
        emsphinx_compatible: bool = True,
    ) -> None:
        bandwidth = int(bandwidth)
        smallest, largest = _BANDWIDTH_LIMITS
        if bandwidth < smallest or bandwidth > largest:
            raise ValueError(
                f"Bandwidth {bandwidth} is an unreasonable bandwidth "
                f"(should be [{smallest}, {largest}])"
            )

        if isinstance(harmonics, MasterPatternHarmonics):
            phases = (harmonics,)
        else:
            phases = tuple(harmonics)
        if len(phases) == 0:
            raise ValueError(
                "`harmonics` must hold the harmonics of at least one "
                "master pattern, but is empty"
            )
        for i, phase in enumerate(phases):
            if not isinstance(phase, MasterPatternHarmonics):
                raise TypeError(
                    f"`harmonics[{i}]` of type {type(phase)} must be a "
                    "MasterPatternHarmonics"
                )

        # All master patterns must have been computed for the same
        # geometry (``idx.hpp`` line 185); a value which is not set
        # skips its comparison, since hand built harmonics may carry
        # neither
        for name in ("sample_tilt", "beam_energy"):
            known = [
                (i, getattr(phase, name))
                for i, phase in enumerate(phases)
                if getattr(phase, name) is not None
            ]
            for i, value in known[1:]:
                first, reference = known[0]
                if abs(value - reference) > 1e-6 * max(1.0, abs(reference)):
                    raise ValueError(
                        f"All master patterns must have the same `{name}`, "
                        f"but harmonics[{first}] has {reference} and "
                        f"harmonics[{i}] has {value}"
                    )

        # EMSphInx builds the detector geometry *from* the master
        # pattern, so the tilt cannot disagree there; here it comes
        # from the detector and is bound to the master's instead
        detector_tilt = getattr(detector, "sample_tilt", None)
        harmonics_tilt = next(
            (p.sample_tilt for p in phases if p.sample_tilt is not None), None
        )
        if detector_tilt is not None and harmonics_tilt is not None:
            detector_tilt = float(detector_tilt)
            tolerance = 1e-6 * max(1.0, abs(detector_tilt))
            if abs(harmonics_tilt - detector_tilt) > tolerance:
                raise ValueError(
                    f"The master patterns' `sample_tilt` {harmonics_tilt} "
                    f"and the `EBSDDetector.sample_tilt` {detector_tilt} "
                    "must be equal, since the harmonics are indexed in "
                    "the geometry they were computed for"
                )

        # The name list rule (``nml.hpp`` line 630), re-validated by
        # the preprocessing, but failing here beats failing in a
        # worker
        n_regions = int(n_regions)
        detector_shape = getattr(detector, "shape", None)
        if detector_shape is not None:
            longest = min(detector_shape)
            if n_regions < 0 or n_regions > longest:
                raise ValueError(
                    f"`n_regions` must be between 0 and {longest}, but is {n_regions}"
                )

        # Last, since it owns every geometry and mask guard and its
        # errors are the ones to maintain
        projector = SphericalBackProjector(
            detector,
            bandwidth,
            signal_mask=signal_mask,
            circular_mask=circular_mask,
        )

        # ``MasterSpectra::resize(nml.bw)`` (``idx.hpp`` line 182):
        # truncation is silent as in the C++, while zero padding gets
        # a warning of ours, since it buys a finer Euler grid but no
        # new signal
        resized = []
        for phase in phases:
            if phase.bandwidth < bandwidth:
                warn(
                    f"The harmonics of bandwidth {phase.bandwidth} are "
                    f"zero padded to the indexing bandwidth {bandwidth}, "
                    "which gives a finer Euler grid but no new signal",
                    UserWarning,
                )
            resized.append(phase.resize(bandwidth))

        # One Wigner table for every correlator (2.5 MB at ``bw`` 68)
        table = wigner_d_half_pi_table(bandwidth, True)
        # The refinement's beta independent factor triple (5.07 MB at
        # ``bw`` 68) is built **eagerly** and shared into every
        # correlator, since the chunk workers clone before their first
        # refinement and a lazily built triple would therefore be
        # rebuilt once per clone.  A coarse-only indexer pays nothing
        factors = wigner_d_table_factors(bandwidth) if refine else None
        if normalize:
            correlators = tuple(
                NormalizedSphericalCrossCorrelator(
                    bandwidth,
                    phase.alm,
                    projector.squared_harmonics(phase.alm),
                    phase.n_fold,
                    phase.has_equatorial_mirror,
                    projector.window_harmonics,
                    wigner_d_half_pi=table,
                    wigner_d_factors=factors,
                )
                for phase in resized
            )
            correlator = None
            spectra = None
        else:
            # The C++ un-normalised correlator holds only a spectrum,
            # so one shared prototype serves every phase
            correlators = None
            correlator = SphericalCrossCorrelator(
                bandwidth, wigner_d_half_pi=table, wigner_d_factors=factors
            )
            spectra = tuple(
                (phase.alm, phase.n_fold, phase.has_equatorial_mirror)
                for phase in resized
            )

        # ``good_pixels`` is in the opposite polarity of
        # ``signal_mask`` and is reduced to the terms present, so that
        # the default is ``IndexEBSD``'s ``circmask = -1``, i.e. no
        # histogram mask at all
        good_pixels = None
        if projector.signal_mask is not None:
            good_pixels = ~projector.signal_mask
        if circular_mask:
            circle = _circular_mask(projector.detector.shape)
            if good_pixels is None:
                good_pixels = circle
            else:
                good_pixels = good_pixels & circle

        self.phases = tuple(resized)
        self.n_phases = len(self.phases)
        self.bandwidth = bandwidth
        self.normalize = bool(normalize)
        self.refine = bool(refine)
        self.projector = projector
        self.correlators = correlators
        self.correlator = correlator
        self.spectra = spectra
        self.wigner_d_half_pi = table
        self.wigner_d_factors = factors
        self.good_pixels = good_pixels
        self.signal_mask = projector.signal_mask
        self.circular_mask = bool(circular_mask)
        self.n_regions = n_regions
        self.gaussian_background = bool(gaussian_background)
        self.emsphinx_compatible = bool(emsphinx_compatible)

        first = correlators[0] if normalize else correlator
        self.side_length = int(first.side_length)
        self._half_side_length = int(first.half_side_length)
        self.half_cell_degrees = 180.0 / self.side_length

    def __repr__(self) -> str:
        """Return a string with the phases, the bandwidth, the sphere
        window and the correlation, e.g. ``"SphericalIndexer: 1 phase
        (ni), bw = 68, sphere window 1317 points (14.6 %),
        normalized"``.

        A phase whose harmonics carry no
        :class:`~orix.crystal_map.Phase` is named ``"?"``.
        """
        names = ", ".join(_phase_name(phase) for phase in self.phases)
        plural = "phase" if self.n_phases == 1 else "phases"
        correlation = "normalized" if self.normalize else "un-normalized"
        projector = self.projector
        return (
            f"{type(self).__name__}: {self.n_phases} {plural} ({names}), "
            f"bw = {self.bandwidth}, sphere window {projector.n_points} "
            f"points ({100 * projector.window_fraction:.1f} %), "
            f"{correlation}"
        )

    def _memory_model(self, refine: bool) -> int:
        """Return the estimated memory one worker needs for a run
        with or without refinement, in bytes.

        Parameters
        ----------
        refine
            Whether the run refines.  :meth:`refine_patterns` always
            does, whatever :attr:`refine` says, so the flag is a
            parameter rather than the attribute.

        Returns
        -------
        n_bytes
            The model of :attr:`memory_per_worker_bytes`, with the
            ``n_correlators * 16 bw^3`` refinement term when
            ``refine``.

        Notes
        -----
        The refinement term is **per correlator clone**, not one per
        worker: every clone owns its own ``(bw, bw, bw, 2)`` Wigner
        d table, and a normalized run clones one correlator per
        phase, so a flat term would understate every multi-phase run
        by ``(P - 1) 16 bw^3`` and make the 2 GiB warning under-fire
        on exactly the runs which need it.
        """
        n_correlators = self.n_phases if self.normalize else 1
        side = self.side_length
        n_bytes = n_correlators * side * side * self._half_side_length * 24
        n_bytes += side**3 * 8
        if refine:
            n_bytes += n_correlators * 16 * self.bandwidth**3
        return n_bytes

    @property
    def memory_per_worker_bytes(self) -> int:
        """Return the estimated memory one worker needs, in bytes.

        Returns
        -------
        n_bytes
            ``(n_phases if normalize else 1) slP^2 bwP 24 + slP^3 8``
            bytes, i.e. one correlation spectrum and cube per
            correlator clone plus the interpolation cube, 49,426,200
            at ``bw`` 68 with one phase; plus
            ``(n_phases if normalize else 1) 16 bw^3`` bytes, one
            refinement Wigner d table per correlator clone, when
            :attr:`refine` is set, i.e. 54,457,112 at ``bw`` 68 with
            one phase.

        Notes
        -----
        The model is above the measured transient peak by about
        10 %, see the table in the module documentation.  With
        ``normalize=False`` one scratch correlator serves every
        phase, so the phase count drops out of both terms.

        :meth:`index_patterns` warns when this times the number of
        dask workers exceeds 2 GiB.
        """
        return self._memory_model(self.refine)

    def get_info_message(
        self,
        n_patterns: int,
        chunksize: int | None = None,
        refining: bool = False,
    ) -> str:
        """Return the information message printed before indexing.

        Parameters
        ----------
        n_patterns
            Number of patterns to index or to refine.
        chunksize
            Number of patterns per chunk.  If not given, the ported
            chunk sizing model resolves it, exactly as
            :meth:`index_patterns` does.
        refining
            Whether the message describes a refinement-only run, i.e.
            :meth:`refine_patterns`.  Default is ``False``.  Such a
            run always refines, so its work line reads ``Refining n
            orientation(s)`` and its memory line prints the refined
            model whatever :attr:`refine` says.

        Returns
        -------
        message
            Multiple lines naming the phases with their point groups
            and symmetry flags, the bandwidth with the Euler side
            length and half cell, the correlation, the refinement,
            the preprocessing, the projection centre, the chunking
            and the estimated memory per worker, e.g.

            .. code-block::

                Spherical indexing information:
                  Phase(s): ni (m-3m; 4-fold, mirror)
                  Bandwidth: 68 (Euler side length 135, half cell
                  1.33 deg)
                  Correlation: normalized
                  Refinement: Newton (on)
                  Preprocessing: n_regions = 10,
                  gaussian_background = False
                  Projection center (Bruker): (0.4251, 0.2134,
                  0.5007)
                  Indexing 9 pattern(s) in 9 chunk(s) of up to 1
                  pattern(s)
                  Estimated memory per worker: 54 MB

        Notes
        -----
        The memory line prints
        :attr:`memory_per_worker_bytes`, the model, not a
        measurement.
        """
        n_patterns = int(n_patterns)
        if chunksize is None:
            chunksize = _batch_estimate(self.bandwidth, _n_workers(), n_patterns)
        chunksize = max(1, int(chunksize))
        n_chunks = -(-n_patterns // chunksize)
        names = ", ".join(_phase_description(phase) for phase in self.phases)
        correlation = "normalized" if self.normalize else "un-normalized"
        refine = self.refine or refining
        refinement = "Newton (on)" if refine else "off"
        pc = tuple(map(float, self.projector.detector.pc.squeeze().round(4)))
        if refining:
            work = (
                f"  Refining {n_patterns} orientation(s) in {n_chunks} "
                f"chunk(s) of up to {chunksize} pattern(s)\n"
            )
        else:
            work = (
                f"  Indexing {n_patterns} pattern(s) in {n_chunks} chunk(s) "
                f"of up to {chunksize} pattern(s)\n"
            )
        return (
            "Spherical indexing information:\n"
            f"  Phase(s): {names}\n"
            f"  Bandwidth: {self.bandwidth} (Euler side length "
            f"{self.side_length}, half cell "
            f"{self.half_cell_degrees:.2f} deg)\n"
            f"  Correlation: {correlation}\n"
            f"  Refinement: {refinement}\n"
            f"  Preprocessing: n_regions = {self.n_regions}, "
            f"gaussian_background = {self.gaussian_background}\n"
            f"  Projection center (Bruker): {pc}\n"
            + work
            + "  Estimated memory per worker: "
            f"{self._memory_model(refine) / 1e6:.0f} MB"
        )

    def index_patterns(
        self,
        patterns: np.ndarray | da.Array,
        *,
        n_best: int = 1,
        chunksize: int | None = None,
        progressbar: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return the best orientations of a stack of patterns.

        Parameters
        ----------
        patterns
            ``(n, nrows, ncols)`` NumPy or dask array of any real
            data type, whose last two axes must match the detector
            shape.  Unsigned 8-bit is EMSphInx' native path; other
            data types are converted inside the preprocessing.
        n_best
            Number of candidates to keep per pattern, at least one.
            Since each phase contributes exactly one candidate, rows
            beyond the number of phases keep the fill values.
        chunksize
            Number of patterns per chunk, at least one.  If not
            given, the ported chunk sizing model sizes the chunks
            from the bandwidth, the number of dask workers and the
            number of patterns.
        progressbar
            Whether to show dask's progress bar, ``True`` by default.

        Returns
        -------
        results
            Dictionary with

            - ``"zyz"``, ``(n, n_best, 3)`` 64-bit float passive ZYZ
              Euler angles in radians, the raw grid quantity, which
              the sample to crystal rotations of
              :meth:`~kikuchipy.signals.EBSD.spherical_indexing` are
              built from,
            - ``"scores"``, ``(n, n_best)`` 64-bit float, descending
              along the second axis,
            - ``"phase_id"``, ``(n, n_best)`` 32-bit integer indices
              into :attr:`phases`, ``-1`` for a row no candidate
              reached,
            - ``"iq"``, ``(n,)`` 64-bit float image quality of the
              processed patterns.

        Raises
        ------
        ValueError
            If the last two axes of ``patterns`` do not match the
            detector shape, if ``n_best`` is smaller than one, or if
            ``chunksize`` is given and smaller than one.

        Warns
        -----
        UserWarning
            If the number of dask workers times
            :attr:`memory_per_worker_bytes` exceeds 2 GiB, or if at
            least one pattern could not be indexed.  A pattern is
            never re-raised on, so this count is the only report of
            the failure cases listed in the class ``Notes``.

        Notes
        -----
        The patterns are always computed eagerly, so the result holds
        NumPy arrays for a lazy input as well.  The default dask
        scheduler for arrays, the threaded one, is used as is, so an
        outer :func:`dask.config.set` is honoured.

        Results are bitwise identical across chunk sizes, worker
        counts and lazy against eager input, see the module
        documentation.
        """
        n_best = int(n_best)
        if n_best < 1:
            raise ValueError(f"`n_best` {n_best} must be at least one")

        detector_shape = self.projector.detector.shape
        shape = tuple(patterns.shape)
        if len(shape) != 3 or shape[1:] != detector_shape:
            raise ValueError(
                f"Patterns of shape {shape[1:]} must have the detector "
                f"shape {detector_shape}"
            )

        if chunksize is not None:
            chunksize = int(chunksize)
            if chunksize < 1:
                raise ValueError(f"`chunksize` {chunksize} must be at least one")

        n_patterns = int(shape[0])
        n_workers = _n_workers()
        if chunksize is None:
            chunksize = _batch_estimate(self.bandwidth, n_workers, n_patterns)

        needed = n_workers * self.memory_per_worker_bytes
        if needed > _MEMORY_WARNING_BYTES:
            warn(
                f"Indexing on {n_workers} worker(s) is estimated to need "
                f"{needed / 1024**3:.2f} GiB of memory, which is more than "
                "2 GiB; reduce the bandwidth, the number of phases or the "
                "number of dask workers",
                UserWarning,
            )

        if isinstance(patterns, da.Array):
            blocks = patterns.rechunk((chunksize, -1, -1))
        else:
            blocks = da.from_array(patterns, chunks=(chunksize, -1, -1))
        results = _map_chunks(blocks, self, n_best)

        # One eager compute, as dictionary indexing does: the graph
        # is an implementation detail and no consumer of a lazy one
        # exists yet
        if progressbar:
            with ProgressBar():
                packed = results.compute()
        else:
            packed = results.compute()

        # A failed pattern is never re-raised on, so without this the
        # only trace of it is the invalid phase of its best row: a run
        # whose every pattern failed would otherwise return in silence
        n_failed = int((packed[:, 0, 4] < 0).sum())
        if n_failed:
            warn(
                f"{n_failed} of {n_patterns} pattern(s) could not be indexed "
                "and carry the fill values (identity rotation, score 0, "
                "phase -1, image quality 0); see the failure cases of "
                "`SphericalIndexer`",
                UserWarning,
            )

        return {
            "zyz": np.ascontiguousarray(packed[:, :, :3]),
            "scores": np.ascontiguousarray(packed[:, :, 3]),
            "phase_id": packed[:, :, 4].astype(np.int32),
            # The image quality belongs to the pattern, not to the
            # candidate, so the best row's column is the pattern's
            "iq": np.ascontiguousarray(packed[:, 0, 5]),
        }

    def refine_patterns(
        self,
        patterns: np.ndarray | da.Array,
        zyz: np.ndarray,
        phase_id: np.ndarray,
        *,
        chunksize: int | None = None,
        progressbar: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return Newton refined orientations of a stack of patterns
        which are already indexed.

        Parameters
        ----------
        patterns
            ``(n, nrows, ncols)`` NumPy or dask array of any real
            data type, whose last two axes must match the detector
            shape.
        zyz
            ``(n, 3)`` array of starting passive ZYZ Euler angles in
            radians, one per pattern, e.g. the ``"zyz"`` of
            :meth:`index_patterns` or
            :func:`kikuchipy.indexing._spherical._euler.
            rotation_to_zyz` of a crystal map's rotations.
        phase_id
            ``(n,)`` array of indices into :attr:`phases`, one per
            pattern.  A **negative** index marks a point which is not
            indexed: its row passes through untouched and is never
            refined.
        chunksize
            Number of patterns per chunk, at least one.  If not
            given, the ported chunk sizing model sizes the chunks.
        progressbar
            Whether to show dask's progress bar, ``True`` by default.

        Returns
        -------
        results
            Dictionary with the same four keys as
            :meth:`index_patterns`, all with a single candidate:
            ``"zyz"`` ``(n, 3)``, ``"scores"`` ``(n,)``, ``"iq"``
            ``(n,)`` and ``"phase_id"`` ``(n,)`` 32-bit integer, the
            input echoed back.  A row which was not refined carries
            its input angles and phase.

        Raises
        ------
        ValueError
            If the last two axes of ``patterns`` do not match the
            detector shape, if ``zyz`` or ``phase_id`` does not have
            one row per pattern, if a non-negative ``phase_id`` does
            not index :attr:`phases`, or if ``chunksize`` is given
            and smaller than one.

        Warns
        -----
        UserWarning
            If the number of dask workers times
            :attr:`memory_per_worker_bytes` exceeds 2 GiB.

        See Also
        --------
        kikuchipy.signals.EBSD.refine_orientation_spherical
        index_patterns

        Notes
        -----
        **This method always refines**, whatever :attr:`refine` says:
        it builds and shares the beta independent Wigner d factor
        triple exactly as a refining construction does, so that the
        chunk workers' clones share one, and its information message
        prints the refined memory model.

        Per pattern the pipeline is the one of
        :meth:`index_patterns` -- preprocess, back-project, analyse
        -- followed by a refinement against **that pattern's own
        phase**, and the image quality is recomputed as EMSphInx
        recomputes it.  A refinement which does not converge is not
        a failure: it returns the starting triple with the analytic
        value there.

        Refinement is score-monotone only for starting orientations
        which came from :meth:`index_patterns` with the same
        configuration.  Newton's method is local, so a foreign
        starting orientation may converge to a stationary point
        whose score is *lower* than the starting one (measured on 3
        of 4 converged unrelated starts).
        """
        raise NotImplementedError(
            "`refine_patterns()` is not implemented yet; it needs the "
            "correlators' `refine_zyz()`"
        )
