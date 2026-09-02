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
#   :func:`_index_chunk`
# - ``Indexer<Real>::Indexer()`` (lines 163-181), as
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
# - ``Indexer<Real>::refineImage()`` and ``Indexer<Real>::refine()``
#   (lines 277-306 and 337-345), the Newton refinement, which
#   ``refine=True`` refuses until it is implemented -- **Phase 7**
#   (spherical-refinement)
# - the pseudo-symmetry loop of ``indexImage()`` (lines 243-261),
#   which needs the pseudo-symmetric operator lists ``pSym`` this
#   release always leaves empty -- **Phase 8**.  The insertion
#   machinery it shares with the phase loop **is** ported
# - ``Geometry<Real>::northPoleQuat()``'s left multiplication (line
#   266), the identity in EMSphInx as shipped, so the conversion of
#   lines 264-269 collapses to ``_euler.rotation_from_zyz``
# - the refine-only work items ``msk[i] & 0x02`` of ``ebsdWorkItem``
#   (``idx.hpp`` lines 438-450)
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
# Python/NumPy/dask for kikuchipy. GPL-2.0-or-later, conveyed under
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
# one import block changes if a collaborator moves.  The ``noqa``
# markers go away with the function bodies -- by hand: ``pyproject``
# selects ``F, E, W, I`` only, so ruff's ``RUF100`` never reports a
# marker left behind (the same holds for the ``SphericalIndexer``
# import of ``signals/ebsd.py``).
import math  # noqa: F401
from typing import TYPE_CHECKING, Sequence

import dask.array as da
import numpy as np

from kikuchipy.indexing._spherical._back_projection import (  # noqa: F401
    SphericalBackProjector,
)
from kikuchipy.indexing._spherical._preprocessing import (  # noqa: F401
    _circular_mask,
    _preprocess_pattern,
)
from kikuchipy.indexing._spherical._wigner import (  # noqa: F401
    wigner_d_half_pi_table,
)
from kikuchipy.indexing._spherical._xcorr import (  # noqa: F401
    NormalizedSphericalCrossCorrelator,
    SphericalCrossCorrelator,
)

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors import EBSDDetector
    from kikuchipy.indexing._spherical._master_pattern_harmonics import (
        MasterPatternHarmonics,
    )

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
    raise NotImplementedError


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

    The correlators are cloned and the north and south buffers
    allocated **once per invocation**, with :func:`numpy.zeros` and
    never :func:`numpy.empty`, so that the chunk worker is stateless
    and no buffer is shared between threads.  See the module
    documentation for the pipeline, the fill values, the five failure
    cases and the insertion rule which never records a candidate with
    a non-positive score.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
        Whether to refine the best orientation of the Euler grid with
        Newton's method.  Only ``False``, the default, is available.
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
    NotImplementedError
        If ``refine`` is ``True``.
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
    is never paid for a call which cannot work: ``refine``, the
    bandwidth range, the harmonics sequence and its entry types, the
    shared ``sample_tilt`` and ``beam_energy`` of the phases, the
    binding of that ``sample_tilt`` to the detector's, ``n_regions``,
    and finally the back-projector, whose own geometry and mask
    errors propagate unchanged rather than being duplicated here.

    **The tilt binding** deserves its own note.  EMSphInx builds the
    detector geometry *from* the master pattern, so a mismatch is
    structurally impossible there; here the tilt comes from the
    detector, so it is checked instead.  Measured: harmonics computed
    for a 70 degree sample tilt, indexed against a detector set to
    65 degrees, land 4.680 degrees (median) from the stored
    orientations at *higher* scores than the correct run, so no score
    based sanity check could catch it.

    **The scores are not normalized cross-correlations.**  The metric
    is the correlation at the interpolated peak, divided by the
    rotation dependent window denominator when ``normalize`` is
    ``True`` but never by the standard deviation of the pattern, so
    it is unbounded and comparable only within one geometry and
    bandwidth.  Measured for the nine Ni patterns at ``bw`` 68:
    0.4963-0.6239 normalized and 0.2799-0.3533 un-normalised.

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
        refine: bool = False,
        signal_mask: np.ndarray | None = None,
        n_regions: int = 10,
        gaussian_background: bool = False,
        circular_mask: bool = False,
        emsphinx_compatible: bool = True,
    ) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the phases, the bandwidth, the sphere
        window and the correlation, e.g. ``"SphericalIndexer: 1 phase
        (ni), bw = 68, sphere window 1317 points (14.6 %),
        normalized"``.

        A phase whose harmonics carry no
        :class:`~orix.crystal_map.Phase` is named ``"?"``.
        """
        raise NotImplementedError

    @property
    def memory_per_worker_bytes(self) -> int:
        """Return the estimated memory one worker needs, in bytes.

        Returns
        -------
        n_bytes
            ``(n_phases if normalize else 1) slP^2 bwP 24 + slP^3 8``
            bytes, i.e. one correlation spectrum and cube per
            correlator clone plus the interpolation cube, 49,426,200
            at ``bw`` 68 with one phase.

        Notes
        -----
        The model is above the measured transient peak by about
        10 %, see the table in the module documentation.  With
        ``normalize=False`` one scratch correlator serves every
        phase, so the phase count drops out.

        :meth:`index_patterns` warns when this times the number of
        dask workers exceeds 2 GiB.
        """
        raise NotImplementedError

    def get_info_message(self, n_patterns: int, chunksize: int | None = None) -> str:
        """Return the information message printed before indexing.

        Parameters
        ----------
        n_patterns
            Number of patterns to index.
        chunksize
            Number of patterns per chunk.  If not given, it is
            resolved with :func:`_batch_estimate` exactly as
            :meth:`index_patterns` resolves it.

        Returns
        -------
        message
            Multiple lines naming the phases with their point groups
            and symmetry flags, the bandwidth with the Euler side
            length and half cell, the correlation, the preprocessing,
            the projection centre, the chunking and the estimated
            memory per worker, e.g.

            .. code-block::

                Spherical indexing information:
                  Phase(s): ni (m-3m; 4-fold, mirror)
                  Bandwidth: 68 (Euler side length 135, half cell
                  1.33 deg)
                  Correlation: normalized
                  Preprocessing: n_regions = 10,
                  gaussian_background = False
                  Projection center (Bruker): (0.4251, 0.2134,
                  0.5007)
                  Indexing 9 pattern(s) in 9 chunk(s) of up to 1
                  pattern(s)
                  Estimated memory per worker: 49 MB

        Notes
        -----
        The memory line prints
        :attr:`memory_per_worker_bytes`, the model, not a
        measurement.
        """
        raise NotImplementedError

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
            given, :func:`_batch_estimate` sizes the chunks from the
            bandwidth, the number of dask workers and the number of
            patterns.
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
            :attr:`memory_per_worker_bytes` exceeds 2 GiB.

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
        raise NotImplementedError
