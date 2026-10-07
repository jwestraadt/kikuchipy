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

from time import time
from typing import TYPE_CHECKING, Sequence
import warnings

import dask
import dask.array as da
from dask.system import CPU_COUNT
import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
from tqdm import tqdm

from kikuchipy.indexing._dictionary_indexing import _dictionary_indexing
from kikuchipy.indexing._hrosm._averaging import (
    _average_grains,
    _coverage_warning_message,
    _first_rotations,
    _grod_map,
    _orix_operators_by_phase,
)
from kikuchipy.indexing._hrosm._grains import (
    GrainTable,
    _broadcast_grain_props,
    _map_grid,
)
from kikuchipy.indexing._hrosm._kam import kernel_average_misorientation_map
from kikuchipy.indexing._hrosm._osm import _osm_emsoft, _osm_grain_aware
from kikuchipy.indexing._hrosm._sampling import (
    _cubochoric_grid,
    _cubochoric_to_homochoric,
    _emsoft_rodrigues_round_trip,
    _multiply,
    misorientation_ball_spacing,
)
from kikuchipy.indexing._hrosm._segmentation import (
    grain_bounding_boxes,
    segment_grains_kam,
)

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors._ebsd_detector import EBSDDetector
    from kikuchipy.signals.ebsd import EBSD
    from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern

# Largest number of bytes of one chunk of simulated float32 dictionary
# patterns, setting the default number of patterns per iteration
_DICTIONARY_CHUNK_BYTES = 256e6

# Smallest number of patterns of one simulation task. Each chunk of
# patterns matched per iteration is simulated as several such tasks,
# which the threaded scheduler runs in parallel, since one task
# projects its patterns on one thread.
_SIMULATION_TASK_MIN = 1024

# Keyword arguments of the driver and their defaults, those of
# EBSD.hrosm
_KEYWORD_DEFAULTS = {
    "threshold": 5.0,
    "max_angle": 5.0,
    "n_steps": 20,
    "keep_n": 20,
    "n_osm": 10,
    "average": "mean",
    "n_em": 25,
    "n_iter": 40,
    "min_kappa": 5.0,
    "min_pixels": 10,
    "dilate": False,
    "metric": "ncc",
    "signal_mask": None,
    "navigation_mask": None,
    "pc": "grain",
    "n_per_iteration": None,
    "seed": None,
    "emsoft_compatible": False,
    "verbose": 1,
}


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
    kw = {**_KEYWORD_DEFAULTS, **keywords}
    compat = bool(kw["emsoft_compatible"])
    max_angle = float(kw["max_angle"])
    n_steps = int(kw["n_steps"])
    keep_n = int(kw["keep_n"])
    n_osm = int(kw["n_osm"])
    verbose = int(kw["verbose"])

    # (a) Points not in the data, not indexed or masked are absent
    grid_in, (ny, nx) = _map_grid(xmap)
    n_map = ny * nx
    flat_in = grid_in.ravel()
    in_data = flat_in >= 0
    work = _without_masked_points(xmap, kw["navigation_mask"])
    grid, _ = _map_grid(work)
    flat_grid = grid.ravel()
    work_in = flat_grid >= 0
    phase_grid = np.full(n_map, -1, dtype=np.int64)
    phase_grid[work_in] = np.asarray(work.phase_id).ravel()[flat_grid[work_in]]
    present = phase_grid >= 0

    # (b) Orientation stage: KAM, grains, boxes, references and GROD
    kam = kernel_average_misorientation_map(work, emsoft_compatible=compat)
    grain_id = segment_grains_kam(
        kam,
        threshold=kw["threshold"],
        dilate=bool(kw["dilate"]),
        phase_id=phase_grid.reshape(ny, nx),
        emsoft_compatible=compat,
    )
    boxes = grain_bounding_boxes(grain_id)
    table = _average_grains(
        work,
        grain_id,
        method=kw["average"],
        max_angle=max_angle,
        n_em=kw["n_em"],
        n_iter=kw["n_iter"],
        min_kappa=kw["min_kappa"],
        seed=kw["seed"],
        emsoft_compatible=compat,
    )
    grod = _grod_map(
        work, grain_id, table.rotation, table.valid, _orix_operators_by_phase(work)
    )

    # (c) Skip decisions
    reindex = _grains_to_reindex(
        table, int(kw["min_pixels"]), work, master_pattern, grains
    )
    labels = np.flatnonzero(reindex) + 1

    # (d) Ball spacing, the one coverage warning and the information
    # line, before the ball is built or anything is simulated
    ball_size = (2 * n_steps + 1) ** 3
    spacing = 0.0
    if labels.size > 0 or verbose >= 1:
        spacing = misorientation_ball_spacing(max_angle, n_steps)
    if labels.size > 0:
        message = _coverage_warning_message(
            labels,
            table.max_grod[labels - 1],
            table.n_pixels[labels - 1],
            max_angle,
            spacing,
        )
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=3)
    if verbose >= 1:
        print(
            f"Misorientation ball: {ball_size} orientations within "
            f"{max_angle:g} deg of each grain reference orientation, mean "
            f"spacing {spacing:.4f} deg; re-indexing {labels.size} grain(s)"
        )

    # Output arrays over the whole grid, holding the fill of points
    # that are not re-indexed
    fill = 0.0 if compat else np.nan
    osm = np.full(n_map, fill, dtype=np.float32)
    scores = np.full((n_map, keep_n), fill, dtype=np.float32)
    simulation_indices = np.full((n_map, keep_n), -1, dtype=np.int32)
    reindexed = np.zeros(n_map, dtype=bool)
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_map, 1))
    if not compat:
        rotations[in_data] = _first_rotations(xmap)[flat_in[in_data]]

    time_start = time()
    if labels.size > 0:
        # (e) The ball about the identity, once
        q_raw = _raw_ball(max_angle, n_steps)
        sig_size = int(np.prod(signal.axes_manager.signal_shape))
        n_per_iteration = kw["n_per_iteration"]
        if n_per_iteration is None:
            n_per_iteration = _default_n_per_iteration(sig_size, ball_size)
        n_per_iteration = int(min(n_per_iteration, ball_size))
        labels_flat = np.asarray(grain_id).ravel()
        multi_pc = detector.navigation_shape != (1,)
        pc_all = np.asarray(detector.pc, dtype=np.float64).reshape(-1, 3)
        centres = np.asarray(table.rotation.data, dtype=np.float64).reshape(-1, 4)

        for label in tqdm(labels, desc="Grains", disable=verbose < 1):
            # The grain's ball, composed with its reference
            q_ball = _multiply(q_raw, centres[label - 1][None, :])
            if compat:
                q_ball = _emsoft_rodrigues_round_trip(q_ball)
            ball = Rotation(q_ball)
            ball.data = q_ball

            # (e2) Domain: the grain's points, or its bounding box in
            # EMsoft compatible mode, flat in raster order
            in_grain = labels_flat == label
            if compat:
                r0, c0, h, w = (int(v) for v in boxes[label - 1])
                rows, cols = np.mgrid[r0 : r0 + h, c0 : c0 + w]
                domain = (rows * nx + cols).ravel()
            else:
                domain = np.flatnonzero(in_grain & present)
            n_domain = int(domain.size)

            # (f) The experimental block, eagerly as float32, and the
            # metric for its size
            block = _experimental_block(signal, domain, nx, sig_size)
            metric = signal._prepare_metric(
                kw["metric"], None, kw["signal_mask"], None, False, ball_size
            )
            metric.n_experimental_patterns = n_domain
            # A metric instance may hold the navigation mask of an
            # earlier call; the block holds only the domain's patterns
            metric.navigation_mask = None

            # (g) The grain's detector
            det_g = detector
            if multi_pc:
                det_g = detector.deepcopy()
                if kw["pc"] == "single":
                    det_g.pc = detector.pc_average
                else:
                    det_g.pc = pc_all[domain].mean(axis=0)

            # (h) The lazy dictionary, chunked along the ball by the
            # patterns per iteration. Each chunk is simulated in
            # parallel tasks tiling it, so no task is simulated in two
            # iterations; patterns are projected one by one, so the
            # rechunked dictionary is bitwise that of one task per chunk
            sim = master_pattern.get_patterns(
                ball,
                det_g,
                energy,
                dtype_out="float32",
                compute=False,
                chunk_shape=_simulation_chunks(ball_size, n_per_iteration),
            )
            sim_data = sim.data.rechunk(
                (n_per_iteration,) + (-1,) * (sim.data.ndim - 1)
            )

            # (i) Dictionary indexing of the block
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                xmap_g = _dictionary_indexing(
                    experimental=block,
                    experimental_nav_shape=(n_domain,),
                    dictionary=sim_data,
                    step_sizes=(1.0,),
                    dictionary_xmap=sim.xmap,
                    metric=metric,
                    keep_n=keep_n,
                    n_per_iteration=n_per_iteration,
                    verbose=verbose >= 2,
                )

            # (j) Results, copied back for the grain's points only
            sim_g = np.asarray(xmap_g.prop["simulation_indices"])
            sim_g = sim_g.reshape(n_domain, -1).astype(np.int32)
            scores_g = np.asarray(xmap_g.prop["scores"])
            scores_g = scores_g.reshape(n_domain, -1).astype(np.float32)
            copy = in_grain[domain]
            target = domain[copy]
            simulation_indices[target] = sim_g[copy]
            scores[target] = scores_g[copy]
            rotations[target] = q_ball[sim_g[copy, 0]]
            reindexed[target] = True
            if compat:
                osm_box = _osm_emsoft(sim_g, h, w, n_osm).ravel()
                osm[target] = osm_box[copy]

        if not compat:
            osm_grid = _osm_grain_aware(
                simulation_indices, grain_id, reindexed.reshape(ny, nx), n_osm
            ).ravel()
            osm[reindexed] = osm_grid[reindexed]

    if verbose >= 1:
        print(f"Re-indexing time: {time() - time_start:.5f} s")

    # (m) Nothing re-indexed
    if not reindexed.any():
        warnings.warn(
            "HROSM: no grain re-indexed, so the orientation similarity map "
            "holds only the fill of points that are not re-indexed",
            UserWarning,
            stacklevel=3,
        )

    # (k) The output map, in the input's frame
    grain_id_flat = np.asarray(grain_id, dtype=np.int32).ravel()
    props = {
        "osm": osm,
        "scores": scores,
        "simulation_indices": simulation_indices,
        "grain_id": grain_id_flat,
        "kam": np.asarray(kam, dtype=np.float32).ravel(),
        "grod": np.asarray(grod, dtype=np.float32).ravel(),
        "reindexed": reindexed,
    }
    props.update(_broadcast_grain_props(table, grain_id_flat))
    return _output_map(xmap, rotations, props)


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
    n = int(_DICTIONARY_CHUNK_BYTES // (4 * int(sig_size)))
    return int(min(max(n, 1), int(ball_size)))


def _simulation_chunks(n_patterns: int, n_per_iteration: int) -> tuple:
    """Return the sizes of the simulation tasks along the dictionary,
    splitting each chunk of patterns matched per iteration into at most
    one task per CPU of at least :data:`_SIMULATION_TASK_MIN` patterns
    each, of near equal size.

    The tasks tile each chunk, so the tasks of one chunk are simulated
    only when that chunk is matched.

    Parameters
    ----------
    n_patterns
        Number of dictionary patterns.
    n_per_iteration
        Number of dictionary patterns matched per iteration.

    Returns
    -------
    chunks
        Number of patterns of each task, summing to ``n_patterns``.
    """
    n_patterns = int(n_patterns)
    n_per_iteration = max(int(n_per_iteration), 1)
    chunks = []
    for start in range(0, n_patterns, n_per_iteration):
        n = min(n_per_iteration, n_patterns - start)
        n_tasks = max(min(CPU_COUNT, n // _SIMULATION_TASK_MIN), 1)
        size, extra = divmod(n, n_tasks)
        chunks += [size + 1] * extra + [size] * (n_tasks - extra)
    return tuple(chunks)


def _without_masked_points(
    xmap: CrystalMap, navigation_mask: np.ndarray | None
) -> CrystalMap:
    """Return a copy of a crystal map with the points where the
    navigation mask is True taken out of the data, or the map itself
    without a mask.
    """
    if navigation_mask is None:
        return xmap
    work = xmap.deepcopy()
    mask = np.asarray(navigation_mask, dtype=bool).ravel()
    work.is_in_data = np.asarray(xmap.is_in_data, dtype=bool) & ~mask
    return work


def _grains_to_reindex(
    table: GrainTable,
    min_pixels: int,
    xmap: CrystalMap,
    master_pattern: EBSDMasterPattern,
    grains: Sequence[int] | None,
) -> np.ndarray:
    """Return whether each grain of a grain table is re-indexed.

    A grain is skipped if it has fewer than ``min_pixels`` points, is
    not valid, has no points, is of a phase without the master
    pattern, or is not in ``grains`` when given. Grains of phases
    without the master pattern give one warning.

    Raises
    ------
    ValueError
        If the map has several phases and none has the name of the
        master pattern's phase.
    """
    n_pixels = np.asarray(table.n_pixels)
    phase_id = np.asarray(table.phase_id)
    has_points = (n_pixels > 0) & (phase_id >= 0)
    reindex = np.asarray(table.valid, dtype=bool) & has_points
    reindex &= n_pixels >= min_pixels

    # One master pattern: a single-phase map uses it whatever the
    # names, a multi-phase map by the phase name
    ids = np.unique(np.asarray(xmap.phase_id))
    ids = ids[ids >= 0]
    if ids.size > 1:
        name = master_pattern.phase.name
        matching = [int(i) for i in ids if xmap.phases[int(i)].name == name]
        if not matching:
            names = [xmap.phases[int(i)].name for i in ids]
            raise ValueError(
                f"The master pattern phase {name!r} is not among the map's phases "
                f"{names}"
            )
        other = has_points & (phase_id != matching[0])
        if other.any():
            warnings.warn(
                f"{int(other.sum())} grain(s) of phases with no master pattern are "
                f"not re-indexed; only grains of the phase {name!r} are",
                UserWarning,
                stacklevel=4,
            )
        reindex &= ~other

    if grains is not None:
        chosen = np.asarray(list(grains), dtype=np.int64).ravel()
        chosen = chosen[(chosen >= 1) & (chosen <= reindex.size)]
        selected = np.zeros(reindex.size, dtype=bool)
        selected[chosen - 1] = True
        reindex &= selected
    return reindex


def _raw_ball(max_angle: float, n_steps: int) -> np.ndarray:
    """Return the conjugated grid quaternions of the misorientation
    ball, shape ((2 n_steps + 1)**3, 4) of float64.

    Their Hamilton product with a centre quaternion is the ball of
    :func:`~kikuchipy.indexing.misorientation_ball` about that centre,
    bit for bit.
    """
    cu = _cubochoric_grid(np.deg2rad(max_angle), n_steps)
    q_v = Rotation.from_homochoric(_cubochoric_to_homochoric(cu)).data
    q_v = np.asarray(q_v, dtype=np.float64).reshape(-1, 4)
    return q_v * np.array([1.0, -1.0, -1.0, -1.0])


def _experimental_block(
    signal: EBSD, domain: np.ndarray, nx: int, sig_size: int
) -> np.ndarray:
    """Return the patterns at the flat map indices ``domain`` as a
    float32 NumPy array of shape (n, number of detector pixels).
    """
    rows, cols = np.divmod(domain, nx)
    data = signal.data
    if isinstance(data, da.Array):
        block = data.vindex[rows, cols].compute()
    else:
        block = data[rows, cols]
    return np.asarray(block, dtype=np.float32).reshape(domain.size, sig_size)


def _output_map(
    xmap: CrystalMap, rotations: np.ndarray, props: dict[str, np.ndarray]
) -> CrystalMap:
    """Return a crystal map with the coordinates, phase IDs, phases,
    points in the data and scan unit of ``xmap``, and the rotations
    and properties given over every point of the grid.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    rotation = Rotation(rotations)
    rotation.data = rotations
    return CrystalMap(
        rotations=rotation,
        phase_id=np.asarray(xmap._phase_id).copy(),
        x=xmap._x,
        y=xmap._y,
        phase_list=xmap.phases,
        prop=props,
        scan_unit=xmap.scan_unit,
        is_in_data=np.asarray(xmap.is_in_data, dtype=bool).copy(),
    )
