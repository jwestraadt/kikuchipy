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

"""Tests of :meth:`~kikuchipy.signals.EBSD.hrosm`, the per-grain
re-indexing of a crystal map against a misorientation ball centred on
each grain's reference orientation.

The physics arms simulate a map with a 0.5 degree sub-grain boundary
that a global dictionary of about 1.4 degree spacing cannot resolve and
check that the re-indexed orientation similarity map shows it. The
contract arms use a noise-free two-grain map whose input orientations
carry an "indexing error" of at most 0.3 degrees, small balls (343
orientations) and the public HROSM functions as oracles: KAM, grain
segmentation, grain averaging, the GROD map, the misorientation ball
and its spacing, and the coverage warning message. Spies on
:meth:`~kikuchipy.signals.EBSDMasterPattern.get_patterns` and on the
dictionary indexing core record what the driver simulates and matches;
the coverage warning arms stop the run at the first simulation.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import time
from types import SimpleNamespace
import warnings

import numpy as np
from orix import io as orix_io
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy.indexing import (
    GrainTable,
    average_grain_orientations,
    grain_bounding_boxes,
    grain_reference_orientation_deviation_map,
    kernel_average_misorientation_map,
    misorientation_ball,
    misorientation_ball_spacing,
    orientation_similarity_map,
    segment_grains_kam,
)
from kikuchipy.indexing._hrosm import _driver, _emsoft_quaternions, _sampling
from kikuchipy.indexing._hrosm._averaging import _coverage_warning_message
from kikuchipy.indexing._hrosm._driver import _default_n_per_iteration
from kikuchipy.indexing._hrosm._grains import _map_grid
from kikuchipy.indexing._hrosm._osm import _osm_emsoft, _osm_grain_aware
from kikuchipy.signals import EBSD, EBSDMasterPattern

# ------------------- Measured pins (sub-grain contrast) -----------------

# Smallest HROSM contrast, the median orientation similarity of points
# one and two columns from the sub-grain boundary minus that of the
# two boundary columns, of 10 (seed 3.0)
SUBGRAIN_CONTRAST_MIN = 3.0
# Smallest ratio of the HROSM contrast over the global dictionary's
# contrast, the latter floored at 0.5 (seed 2.0)
SUBGRAIN_CONTRAST_RATIO = 2.0
# Largest median disorientation in degrees of the re-indexed
# orientations of grain A, away from the grain boundary, to the truth
# (seed 0.15)
SUBGRAIN_MEDIAN_ERROR_DEG = 0.15
# Largest deviation in degrees of the median disorientation across the
# sub-grain boundary from its true 0.5 degrees (seed 0.15)
SUBGRAIN_STEP_TOL_DEG = 0.15
# The same four pins for the full size demonstration (seeds as above)
SUBGRAIN_FULL_CONTRAST_MIN = 3.0
SUBGRAIN_FULL_CONTRAST_RATIO = 2.0
SUBGRAIN_FULL_MEDIAN_ERROR_DEG = 0.15
SUBGRAIN_FULL_STEP_TOL_DEG = 0.15

# Whether results are bitwise identical for any number of dictionary
# patterns per iteration and for lazy and eager signals ("bitwise"),
# or identical in the best match except at exact score ties with
# scores within 2 float32 ulp ("fallback") (seed "bitwise")
CHUNK_INVARIANCE = "bitwise"
# Largest float32 ulp distance of scores in the fallback
CHUNK_FALLBACK_MAX_ULP = 2

# ------------------------------- Constants ------------------------------

# Base map: two grains of 8 x 10 points with an orientation gradient of
# 0.2 degrees per column
BASE_SHAPE = (8, 10)
# Largest angle in degrees and seed of the random "indexing error"
# left-multiplied onto every true orientation of the input map
INDEXING_ERROR_DEG = 0.3
INDEXING_ERROR_SEED = 90
# Ball of the contract arms: (2 * 3 + 1)**3 = 343 orientations
BALL_MAX_ANGLE = 1.0
BALL_N_STEPS = 3
BALL_SIZE = 343
# Largest disorientation in degrees of a re-indexed orientation to the
# truth on the noise-free base map: a fixed bound, not measured, half
# the ball radius
TRUTH_TOLERANCE_DEG = 0.5
# Largest median disorientation in degrees of the re-indexed
# orientations of a grain to the truth on the noise-free base map; the
# nearest ball orientations are 0.10 degrees off in the median, the
# input map 0.18 degrees, the grain average 0.2 degrees or more (seed
# 0.15)
TRUTH_MEDIAN_TOLERANCE_DEG = 0.15
# Number of best matches compared by the orientation similarity map
# and kept per point by default
N_OSM = 10
KEEP_N = 20
# Detector pattern shape and projection center of the synthetic signals
SIG_SHAPE = (32, 32)
PC = (0.42, 0.22, 0.50)
# Largest deviation of the per-point projection centers from PC, and
# their seed
PC_SPREAD = 0.002
PC_SEED = 91
# Points of grain A excluded by the navigation mask, none adjacent to
# grain B
MASKED_POINTS = ((2, 1), (4, 2), (6, 1))
# Seed and threshold of the map whose EMsoft compatible dilation
# removes grain 1 (found by a search over seeds; the precondition is
# asserted in the test)
VANISHING_SEED = 31
VANISHING_THRESHOLD = 0.5

# The ten properties of the output and their dtypes; the second entry
# is True for properties with ``keep_n`` columns and 4 for quaternions
OUTPUT_PROPS = {
    "osm": (np.float32, None),
    "scores": (np.float32, "keep_n"),
    "simulation_indices": (np.int32, "keep_n"),
    "grain_id": (np.int32, None),
    "kam": (np.float32, None),
    "grod": (np.float32, None),
    "reindexed": (np.bool_, None),
    "grain_orientation": (np.float64, 4),
    "grain_kappa": (np.float64, None),
    "grain_max_grod": (np.float32, None),
}

OPERATORS = Oh.proper_subgroup.data

# Results cached per module: building the signals and running the
# re-indexing once keeps the default suite within its budget
_CACHE: dict = {}


class _StopRun(Exception):
    """Raised by the simulation spy to stop a run at the first
    simulation.
    """


# ------------------------------- Helpers --------------------------------


def _axis_angle(axis, angle_deg) -> Rotation:
    """Return rotations about one axis by angles in degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(np.atleast_1d(np.asarray(angle_deg, dtype=np.float64))) / 2
    data = np.zeros(half.shape + (4,))
    data[..., 0] = np.cos(half)
    data[..., 1:] = np.sin(half)[..., None] * axis
    return Rotation(data)


def _crystal_map(
    rotations: Rotation | np.ndarray,
    shape: tuple[int, int],
    phase_id: np.ndarray | None = None,
    is_in_data: np.ndarray | None = None,
    prop: dict | None = None,
) -> CrystalMap:
    """Return a nickel (m-3m) map of step 1 um over all points of the
    grid ``shape``, with the phase "ni2" (ID 1) if ``phase_id`` has
    ones and points not indexed where it has -1.
    """
    coords, n = create_coordinate_arrays(shape, step_sizes=(1, 1))
    data = rotations.data if isinstance(rotations, Rotation) else rotations
    data = np.asarray(data, dtype=np.float64).reshape(n, 4)
    if phase_id is None:
        phase_id = np.zeros(n, dtype=np.int32)
    phase_id = np.asarray(phase_id, dtype=np.int32).ravel()
    phases = [Phase("ni", point_group="m-3m")]
    ids = [0]
    if np.any(phase_id == 1):
        phases.append(Phase("ni2", point_group="m-3m"))
        ids.append(1)
    if is_in_data is None:
        is_in_data = np.ones(n, dtype=bool)
    return CrystalMap(
        rotations=Rotation(data),
        phase_id=phase_id,
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phases=phases, ids=ids),
        is_in_data=np.asarray(is_in_data, dtype=bool).ravel(),
        scan_unit="um",
        prop=prop,
    )


def _with_indexing_error(
    data: np.ndarray, seed: int = INDEXING_ERROR_SEED, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return quaternions (n, 4) left-multiplied by random rotations of
    at most :data:`INDEXING_ERROR_DEG` about random axes, drawn from
    ``default_rng(seed)``, where ``mask`` is True (everywhere if not
    given).
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1, 4)
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = np.deg2rad(rng.uniform(0, INDEXING_ERROR_DEG, size=n)) / 2
    error = np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axes], 1)
    perturbed = (Rotation(error) * Rotation(data)).data
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        perturbed = np.where(mask[:, None], perturbed, data)
    return perturbed


def _quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton products of quaternions broadcast over the
    leading axes.
    """
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 + a2 * b0 + a3 * b1 - a1 * b3,
            a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1,
        ],
        axis=-1,
    )


def _disorientation_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the m-3m disorientation angles in degrees between the
    quaternions ``a`` and ``b`` (n, 4) point by point, the operators
    applied from the left.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    sa = _quaternion_product(OPERATORS[None, :, :], a[:, None, :])
    d = np.abs(np.sum(sa * b[:, None, :], axis=-1)).max(axis=-1)
    return np.rad2deg(2 * np.arccos(np.clip(d, 0, 1)))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation angles in degrees between the quaternions
    ``a`` and ``b`` (n, 4) point by point, without symmetry.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    d = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2 * np.arccos(np.clip(d, 0, 1)))


def _user_warnings(record) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def _ball_warnings(record) -> list[str]:
    return [m for m in _user_warnings(record) if "misorientation ball" in m]


def _assert_tables_equal(a: GrainTable, b: GrainTable):
    np.testing.assert_array_equal(a.n_pixels, b.n_pixels)
    np.testing.assert_array_equal(a.bounding_box, b.bounding_box)
    np.testing.assert_array_equal(a.rotation.data, b.rotation.data)
    np.testing.assert_array_equal(a.phase_id, b.phase_id)
    np.testing.assert_array_equal(a.kappa, b.kappa)
    np.testing.assert_array_equal(a.max_grod, b.max_grod)
    np.testing.assert_array_equal(a.valid, b.valid)


def _assert_outputs_equal(a: CrystalMap, b: CrystalMap):
    """Assert that two outputs are identical, bit for bit."""
    np.testing.assert_array_equal(a.is_in_data, b.is_in_data)
    np.testing.assert_array_equal(a.phase_id, b.phase_id)
    np.testing.assert_array_equal(a.rotations.data, b.rotations.data)
    assert set(a.prop) == set(b.prop)
    for name in a.prop:
        np.testing.assert_array_equal(a.prop[name], b.prop[name], err_msg=name)


def _assert_outputs_invariant(a: CrystalMap, b: CrystalMap):
    """Assert the result invariance of :data:`CHUNK_INVARIANCE`."""
    if CHUNK_INVARIANCE == "bitwise":
        _assert_outputs_equal(a, b)
        return
    # Identical best match except at exact score ties, scores within
    # a few float32 ulp
    sa, sb = a.prop["scores"], b.prop["scores"]
    ia, ib = a.prop["simulation_indices"], b.prop["simulation_indices"]
    finite = np.isfinite(sa[:, 0])
    np.testing.assert_array_equal(finite, np.isfinite(sb[:, 0]))
    ulp = np.abs(
        sa[finite].astype(np.float32).view(np.int32).astype(np.int64)
        - sb[finite].astype(np.float32).view(np.int32).astype(np.int64)
    )
    assert ulp.max(initial=0) <= CHUNK_FALLBACK_MAX_ULP
    differ = ia[:, 0] != ib[:, 0]
    tie = sa[:, 0] == sa[:, 1] if sa.shape[1] > 1 else np.zeros_like(differ)
    assert np.all(tie[differ])
    np.testing.assert_array_equal(a.prop["grain_id"], b.prop["grain_id"])
    np.testing.assert_array_equal(a.prop["reindexed"], b.prop["reindexed"])


def _grid_values(xmap: CrystalMap, values: np.ndarray, fill) -> np.ndarray:
    """Return per-point values on the map's grid, ``fill`` where a
    point is not in the data.
    """
    grid, shape = _map_grid(xmap)
    values = np.asarray(values)
    out = np.full(shape + values.shape[1:], fill, dtype=values.dtype)
    out[grid >= 0] = values[grid[grid >= 0]]
    return out


def _grid_rotations(xmap: CrystalMap) -> np.ndarray:
    """Return the first rotation per point on the map's grid, NaN where
    a point is not in the data, shape (ny, nx, 4).
    """
    data = xmap.rotations.data
    if data.ndim == 3:
        data = data[:, 0]
    return _grid_values(xmap, data.reshape(-1, 4), np.nan)


def _input_labels(xmap: CrystalMap, emsoft_compatible=False, **kwargs) -> np.ndarray:
    """Return the grain labels of the input map, shape (ny, nx)."""
    kam = kernel_average_misorientation_map(xmap, emsoft_compatible=emsoft_compatible)
    return segment_grains_kam(kam, emsoft_compatible=emsoft_compatible, **kwargs)


def _input_table(
    xmap: CrystalMap, grain_id: np.ndarray, emsoft_compatible=False, **kwargs
) -> GrainTable:
    """Return the grain table of the input map without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return average_grain_orientations(
            xmap, grain_id, emsoft_compatible=emsoft_compatible, **kwargs
        )


def _hrosm(signal: EBSD, xmap, master_pattern, detector, **kwargs) -> CrystalMap:
    """Run :meth:`EBSD.hrosm` with the contract defaults: energy 20 kV,
    the 343 orientation ball and no output.
    """
    kwargs.setdefault("energy", 20)
    kwargs.setdefault("max_angle", BALL_MAX_ANGLE)
    kwargs.setdefault("n_steps", BALL_N_STEPS)
    kwargs.setdefault("verbose", 0)
    return signal.hrosm(xmap, master_pattern, detector, **kwargs)


def _cached_run(key: str, signal, xmap, master_pattern, detector, **kwargs):
    """Return a cached :func:`_hrosm` result with warnings ignored.

    What the run writes to standard output and error is kept under
    ``f"{key}_output"``.
    """
    if key not in _CACHE:
        out, err = io.StringIO(), io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                _CACHE[key] = _hrosm(signal, xmap, master_pattern, detector, **kwargs)
        _CACHE[f"{key}_output"] = (out.getvalue(), err.getvalue())
    return _CACHE[key]


def _stop_at_first_simulation(monkeypatch, on_call=None) -> list:
    """Replace :meth:`EBSDMasterPattern.get_patterns` with a spy that
    calls ``on_call(master_pattern, detector)``, if given, and raises
    :class:`_StopRun`; return the list of recorded calls.
    """
    calls = []
    signature = inspect.signature(EBSDMasterPattern.get_patterns)

    def spy(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        calls.append(bound.arguments)
        if on_call is not None:
            on_call(self, bound.arguments["detector"])
        raise _StopRun

    monkeypatch.setattr(EBSDMasterPattern, "get_patterns", spy)
    return calls


def _record_simulations(monkeypatch) -> list:
    """Wrap :meth:`EBSDMasterPattern.get_patterns` in a spy recording
    each call's master pattern and detector projection centers.
    """
    calls = []
    original = EBSDMasterPattern.get_patterns
    signature = inspect.signature(original)

    def spy(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        detector = bound.arguments["detector"]
        calls.append(
            {
                "master_pattern": self,
                "pc": np.array(detector.pc, dtype=np.float64).reshape(-1, 3),
                "n": np.asarray(bound.arguments["rotations"].data)
                .reshape(-1, 4)
                .shape[0],
            }
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EBSDMasterPattern, "get_patterns", spy)
    return calls


def _record_matching(monkeypatch) -> list:
    """Wrap the dictionary indexing core in a spy recording the number
    of experimental patterns of each call and the simulation indices
    it returns, shape (number of patterns, keep_n).
    """
    from kikuchipy.indexing import _dictionary_indexing as di_module

    original = di_module._dictionary_indexing
    signature = inspect.signature(original)
    calls = []

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        experimental = bound.arguments["experimental"]
        nav_shape = bound.arguments["experimental_nav_shape"]
        result = original(*args, **kwargs)
        n_experimental = int(np.asarray(experimental.shape)[0])
        calls.append(
            {
                "n_experimental": n_experimental,
                "nav_size": int(np.prod(nav_shape)),
                "simulation_indices": np.asarray(
                    result.prop["simulation_indices"]
                ).reshape(n_experimental, -1),
            }
        )
        return result

    monkeypatch.setattr(di_module, "_dictionary_indexing", spy)
    monkeypatch.setattr(_driver, "_dictionary_indexing", spy, raising=False)
    return calls


def _zero_signal(shape: tuple[int, ...]) -> EBSD:
    """Return an EBSD signal of zeros with the navigation shape
    ``shape`` and the pattern shape :data:`SIG_SHAPE`.
    """
    return EBSD(np.zeros(tuple(shape) + SIG_SHAPE, dtype=np.float32))


# ------------------------------- Fixtures -------------------------------


@pytest.fixture
def base(hrosm_grain_xmap, hrosm_synthetic_signal) -> SimpleNamespace:
    """Return the two-grain base case, built once per module.

    ``xmap_truth`` holds the true orientations from which the noise-free
    signal ``s`` is simulated; ``xmap`` is the input map, whose
    orientations carry the indexing error and which has one extra
    property ``"input_only"``.
    """
    if "base" not in _CACHE:
        xmap_truth, truth = hrosm_grain_xmap(shape=BASE_SHAPE, n_grains=2, gradient=0.2)
        s, det, mp = hrosm_synthetic_signal(xmap_truth, sig_shape=SIG_SHAPE, pc=PC)
        n = xmap_truth.size
        xmap = _crystal_map(
            _with_indexing_error(xmap_truth.rotations.data),
            BASE_SHAPE,
            prop={"input_only": np.arange(n, dtype=np.float64)},
        )
        grain_id = _input_labels(xmap)
        table = _input_table(xmap, grain_id, max_angle=BALL_MAX_ANGLE)
        label_a = int(grain_id[2, 1])
        label_b = int(grain_id[2, 8])
        _CACHE["base"] = SimpleNamespace(
            xmap_truth=xmap_truth,
            truth=truth,
            xmap=xmap,
            s=s,
            det=det,
            mp=mp,
            grain_id=grain_id,
            table=table,
            label_a=label_a,
            label_b=label_b,
        )
    return _CACHE["base"]


@pytest.fixture
def run_correct(base) -> CrystalMap:
    """Return the cached default run on the base case."""
    return _cached_run("correct", base.s, base.xmap, base.mp, base.det)


@pytest.fixture
def run_compat(base) -> CrystalMap:
    """Return the cached EMsoft compatible run on the base case.

    The dictionary indexing calls of the run, one per bounding box, are
    kept under ``"compat_matching"``.
    """
    if "compat" not in _CACHE:
        with pytest.MonkeyPatch.context() as monkeypatch:
            calls = _record_matching(monkeypatch)
            _cached_run(
                "compat",
                base.s,
                base.xmap,
                base.mp,
                base.det,
                average="center",
                pc="single",
                emsoft_compatible=True,
            )
        _CACHE["compat_matching"] = calls
    return _CACHE["compat"]


@pytest.fixture(params=["correct", "compat"])
def mode_run(request, base) -> tuple[bool, str, CrystalMap]:
    """Return ``(emsoft_compatible, average, output)`` of either mode."""
    if request.param == "correct":
        return False, "mean", request.getfixturevalue("run_correct")
    return True, "center", request.getfixturevalue("run_compat")


def _subgrain_case(
    hrosm_synthetic_signal,
    shape: tuple[int, int],
    sub_col: int,
    b_col: int,
    sig_shape: tuple[int, int],
    max_angle: float,
    n_steps: int,
) -> SimpleNamespace:
    """Return a sub-grain case: grain A of columns ``< b_col`` with
    ``g_A`` left of ``sub_col`` and ``Rz(0.5) * g_A`` from it, grain B
    of the other columns with ``g_B``; a noisy signal; the global
    dictionary indexing result and its orientation similarity map; and
    the HROSM output.
    """
    ny, nx = shape
    n = ny * nx
    _, cols = np.divmod(np.arange(n), nx)
    g_a = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0)))
    g_b = Rotation.from_euler(np.deg2rad((60.0, 45.0, 10.0)))
    assert _disorientation_deg(g_a.data, g_b.data)[0] > 20
    g_a2 = _axis_angle((0, 0, 1), 0.5) * g_a
    data = np.empty((n, 4))
    data[cols < sub_col] = g_a.data
    data[(cols >= sub_col) & (cols < b_col)] = g_a2.data
    data[cols >= b_col] = g_b.data
    xmap_truth = _crystal_map(data, shape)

    s, det, mp = hrosm_synthetic_signal(
        xmap_truth, sig_shape=sig_shape, noise=0.02, seed=50
    )

    # Global dictionary: two balls about centres off every grid point
    # of the truth
    rv = _axis_angle((2, -1, 3), 0.7)
    dictionary_rotations = np.concatenate(
        [
            misorientation_ball(rv * g_a, max_angle=10, n_steps=7).data,
            misorientation_ball(rv * g_b, max_angle=10, n_steps=7).data,
        ]
    )
    dictionary = mp.get_patterns(
        Rotation(dictionary_rotations), det, energy=20, compute=True
    )
    xmap_global = s.dictionary_indexing(dictionary, keep_n=10)
    osm_global = orientation_similarity_map(xmap_global, n_best=10)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = s.hrosm(
            xmap_global,
            mp,
            det,
            energy=20,
            max_angle=max_angle,
            n_steps=n_steps,
            keep_n=10,
            n_osm=10,
            verbose=0,
        )
    runtime = time.perf_counter() - t0
    return SimpleNamespace(
        shape=shape,
        truth=data.reshape(ny, nx, 4),
        xmap_global=xmap_global,
        osm_global=np.asarray(osm_global, dtype=np.float64).reshape(shape),
        out=out,
        runtime=runtime,
    )


@pytest.fixture
def subgrain(hrosm_synthetic_signal) -> SimpleNamespace:
    """Return the small sub-grain case, built once per module."""
    if "subgrain" not in _CACHE:
        _CACHE["subgrain"] = _subgrain_case(
            hrosm_synthetic_signal,
            shape=(10, 16),
            sub_col=4,
            b_col=8,
            sig_shape=(32, 32),
            max_angle=2.0,
            n_steps=8,
        )
    return _CACHE["subgrain"]


def _contrast(osm: np.ndarray, rows: slice, interior, boundary) -> float:
    """Return the median orientation similarity of the interior columns
    minus that of the boundary columns, over ``rows``.
    """
    osm = np.asarray(osm, dtype=np.float64)
    return float(
        np.median(osm[rows][:, list(interior)])
        - np.median(osm[rows][:, list(boundary)])
    )


def _step_deg(rotations: np.ndarray, col_left: int, col_right: int) -> float:
    """Return the median disorientation in degrees between two columns
    of rotations of shape (ny, nx, 4).
    """
    return float(
        np.median(_disorientation_deg(rotations[:, col_left], rotations[:, col_right]))
    )


# ----------------------------- Physics sanity ---------------------------


class TestSubgrainContrast:
    def test_grains_are_segmented_without_splitting_the_subgrain(self, subgrain):
        grain_id = _grid_values(subgrain.out, subgrain.out.prop["grain_id"], -1)
        interior_a = np.unique(grain_id[:, 0:7])
        interior_b = np.unique(grain_id[:, 9:16])
        assert interior_a.size == 1
        assert interior_b.size == 1
        assert interior_a[0] > 0
        assert interior_b[0] > 0
        assert interior_a[0] != interior_b[0]

    def test_hrosm_osm_shows_the_subboundary(self, subgrain, record_property):
        osm = _grid_values(subgrain.out, subgrain.out.prop["osm"], np.nan)
        rows = slice(1, 9)
        c_hrosm = _contrast(osm, rows, (1, 2, 5, 6), (3, 4))
        c_global = _contrast(subgrain.osm_global, rows, (1, 2, 5, 6), (3, 4))
        record_property("subgrain_contrast_hrosm", c_hrosm)
        record_property("subgrain_contrast_global", c_global)
        assert c_hrosm >= SUBGRAIN_CONTRAST_MIN
        assert c_hrosm / max(c_global, 0.5) >= SUBGRAIN_CONTRAST_RATIO

    def test_subgrain_step_is_recovered(self, subgrain, record_property):
        out = subgrain.out
        rotations = _grid_rotations(out)
        reindexed = _grid_values(out, out.prop["reindexed"], False)
        # Grain A away from the grain boundary
        assert np.all(reindexed[:, 0:7])
        error = _disorientation_deg(
            rotations[:, 0:7].reshape(-1, 4), subgrain.truth[:, 0:7].reshape(-1, 4)
        )
        median_error = float(np.median(error))
        step = _step_deg(rotations, 2, 5)
        step_global = _step_deg(_grid_rotations(subgrain.xmap_global), 2, 5)
        record_property("subgrain_median_error_deg", median_error)
        record_property("subgrain_step_deg", step)
        record_property("subgrain_global_step_deg", step_global)
        assert median_error <= SUBGRAIN_MEDIAN_ERROR_DEG
        assert abs(step - 0.5) <= SUBGRAIN_STEP_TOL_DEG

    @pytest.mark.weekly
    def test_full_size_subgrain_demonstration(
        self, hrosm_synthetic_signal, record_property
    ):
        case = _subgrain_case(
            hrosm_synthetic_signal,
            shape=(20, 30),
            sub_col=8,
            b_col=15,
            sig_shape=(60, 60),
            max_angle=5.0,
            n_steps=20,
        )
        out = case.out
        record_property("subgrain_full_runtime_s", case.runtime)

        grain_id = _grid_values(out, out.prop["grain_id"], -1)
        interior_a = np.unique(grain_id[:, 0:14])
        interior_b = np.unique(grain_id[:, 16:30])
        assert interior_a.size == interior_b.size == 1
        assert 0 < interior_a[0] != interior_b[0] > 0

        osm = _grid_values(out, out.prop["osm"], np.nan)
        rows = slice(1, 19)
        c_hrosm = _contrast(osm, rows, (5, 6, 9, 10), (7, 8))
        c_global = _contrast(case.osm_global, rows, (5, 6, 9, 10), (7, 8))
        record_property("subgrain_full_contrast_hrosm", c_hrosm)
        record_property("subgrain_full_contrast_global", c_global)
        assert c_hrosm >= SUBGRAIN_FULL_CONTRAST_MIN
        assert c_hrosm / max(c_global, 0.5) >= SUBGRAIN_FULL_CONTRAST_RATIO

        rotations = _grid_rotations(out)
        error = _disorientation_deg(
            rotations[:, 0:14].reshape(-1, 4), case.truth[:, 0:14].reshape(-1, 4)
        )
        median_error = float(np.median(error))
        step = _step_deg(rotations, 6, 9)
        record_property("subgrain_full_median_error_deg", median_error)
        record_property("subgrain_full_step_deg", step)
        record_property(
            "subgrain_full_global_step_deg",
            _step_deg(_grid_rotations(case.xmap_global), 6, 9),
        )
        assert median_error <= SUBGRAIN_FULL_MEDIAN_ERROR_DEG
        assert abs(step - 0.5) <= SUBGRAIN_FULL_STEP_TOL_DEG


# ------------------------------ Validation ------------------------------

P = inspect.Parameter.POSITIONAL_OR_KEYWORD
K = inspect.Parameter.KEYWORD_ONLY
E = inspect.Parameter.empty

FROZEN_SIGNATURES = {
    "GrainTable": (
        GrainTable,
        [
            ("n_pixels", P, E),
            ("bounding_box", P, E),
            ("rotation", P, E),
            ("phase_id", P, E),
            ("kappa", P, E),
            ("max_grod", P, E),
            ("valid", P, E),
            ("method", P, E),
        ],
    ),
    "average_grain_orientations": (
        average_grain_orientations,
        [
            ("xmap", P, E),
            ("grain_id", P, E),
            ("method", K, "mean"),
            ("max_angle", K, 5.0),
            ("n_em", K, 25),
            ("n_iter", K, 40),
            ("min_kappa", K, 5.0),
            ("seed", K, None),
            ("emsoft_compatible", K, False),
        ],
    ),
    "grain_bounding_boxes": (grain_bounding_boxes, [("grain_id", P, E)]),
    "grain_reference_orientation_deviation_map": (
        grain_reference_orientation_deviation_map,
        [("xmap", P, E), ("grain_id", P, E), ("grains", P, E)],
    ),
    "kernel_average_misorientation_map": (
        kernel_average_misorientation_map,
        [("xmap", P, E), ("degrees", K, True), ("emsoft_compatible", K, False)],
    ),
    "misorientation_ball": (
        misorientation_ball,
        [
            ("center", P, None),
            ("max_angle", K, 5.0),
            ("n_steps", K, 20),
            ("emsoft_compatible", K, False),
        ],
    ),
    "misorientation_ball_spacing": (
        misorientation_ball_spacing,
        [("max_angle", P, 5.0), ("n_steps", P, 20)],
    ),
    "segment_grains_kam": (
        segment_grains_kam,
        [
            ("kam", P, E),
            ("threshold", K, 5.0),
            ("dilate", K, False),
            ("phase_id", K, None),
            ("emsoft_compatible", K, False),
        ],
    ),
    "orientation_similarity_map": (
        orientation_similarity_map,
        [
            ("xmap", P, E),
            ("n_best", P, None),
            ("simulation_indices_prop", P, "simulation_indices"),
            ("normalize", P, False),
            ("from_n_best", P, None),
            ("footprint", P, None),
            ("center_index", P, 2),
            ("grain_id", K, None),
            ("emsoft_compatible", K, False),
        ],
    ),
    "EBSD.hrosm": (
        EBSD.hrosm,
        [
            ("self", P, E),
            ("xmap", P, E),
            ("master_pattern", P, E),
            ("detector", P, E),
            ("energy", P, None),
            ("threshold", K, 5.0),
            ("max_angle", K, 5.0),
            ("n_steps", K, 20),
            ("keep_n", K, 20),
            ("n_osm", K, 10),
            ("average", K, "mean"),
            ("n_em", K, 25),
            ("n_iter", K, 40),
            ("min_kappa", K, 5.0),
            ("min_pixels", K, 10),
            ("dilate", K, False),
            ("metric", K, "ncc"),
            ("signal_mask", K, None),
            ("navigation_mask", K, None),
            ("pc", K, "grain"),
            ("n_per_iteration", K, None),
            ("seed", K, None),
            ("emsoft_compatible", K, False),
            ("verbose", K, 1),
        ],
    ),
}


def _one_phase_map(shape, **kwargs) -> CrystalMap:
    n = int(np.prod(shape))
    return _crystal_map(Rotation.identity((n,)).data, shape, **kwargs)


def _per_point_detector(shape=BASE_SHAPE) -> kp.detectors.EBSDDetector:
    rng = np.random.default_rng(PC_SEED)
    pc = np.asarray(PC) + rng.uniform(-PC_SPREAD, PC_SPREAD, size=tuple(shape) + (3,))
    return kp.detectors.EBSDDetector(SIG_SHAPE, pc=pc, sample_tilt=70)


def _nav_size_8x10_mask(value=False, dtype=bool, shape=BASE_SHAPE) -> np.ndarray:
    return np.full(shape, value, dtype=dtype)


def _absent_one(shape=BASE_SHAPE) -> np.ndarray:
    is_in_data = np.ones(int(np.prod(shape)), dtype=bool)
    is_in_data[11] = False
    return is_in_data


def _not_indexed_one(shape=BASE_SHAPE) -> np.ndarray:
    phase_id = np.zeros(int(np.prod(shape)), dtype=np.int32)
    phase_id[11] = -1
    return phase_id


def _two_phases(shape=BASE_SHAPE) -> np.ndarray:
    phase_id = np.zeros(shape, dtype=np.int32)
    phase_id[:, shape[1] // 2 :] = 1
    return phase_id


# Each case: id, changes to the arguments (a dict of keyword arguments
# and the special keys "signal", "xmap", "detector"), and the expected
# message fragment. The first group violates one check each; the
# second violates two, the earlier of which is reported.
VALIDATION_CASES = [
    (
        "one_nav_dim",
        {"signal": lambda: _zero_signal((80,))},
        "two navigation dimensions",
    ),
    ("xmap_shape", {"xmap": lambda: _one_phase_map((8, 9))}, "xmap shape"),
    ("xmap_shape_transposed", {"xmap": lambda: _one_phase_map((10, 8))}, "xmap shape"),
    (
        "detector_shape",
        {"detector": lambda: kp.detectors.EBSDDetector((16, 16), pc=PC)},
        "detector shape",
    ),
    (
        "detector_navigation_shape",
        {
            "detector": lambda: kp.detectors.EBSDDetector(
                SIG_SHAPE, pc=np.tile(PC, (3, 1))
            )
        },
        "one PC or one PC per map point",
    ),
    ("threshold_zero", {"threshold": 0.0}, "threshold"),
    ("threshold_negative", {"threshold": -1.0}, "threshold"),
    ("max_angle_zero", {"max_angle": 0.0}, "max_angle"),
    ("max_angle_180", {"max_angle": 180.0}, "max_angle"),
    ("n_steps_zero", {"n_steps": 0}, "n_steps"),
    ("n_steps_bool", {"n_steps": True}, "n_steps"),
    ("n_steps_float", {"n_steps": 2.5}, "n_steps"),
    ("keep_n_zero", {"keep_n": 0, "n_osm": 1}, "keep_n"),
    ("keep_n_bool", {"keep_n": True, "n_osm": 1}, "keep_n"),
    ("n_osm_zero", {"n_osm": 0}, "n_osm"),
    ("n_osm_bool", {"n_osm": True}, "n_osm"),
    ("n_osm_above_keep_n", {"keep_n": 5, "n_osm": 6}, "n_osm"),
    ("keep_n_above_ball", {"n_steps": 3, "keep_n": 344}, "keep_n"),
    ("n_em_zero", {"n_em": 0}, "n_em"),
    ("n_em_bool", {"n_em": True}, "n_em"),
    ("n_iter_zero", {"n_iter": 0}, "n_iter"),
    ("n_iter_bool", {"n_iter": True}, "n_iter"),
    ("min_kappa_negative", {"min_kappa": -1.0}, "min_kappa"),
    ("min_pixels_zero", {"min_pixels": 0}, "min_pixels"),
    ("min_pixels_bool", {"min_pixels": True}, "min_pixels"),
    ("n_per_iteration_zero", {"n_per_iteration": 0}, "n_per_iteration"),
    ("n_per_iteration_bool", {"n_per_iteration": True}, "n_per_iteration"),
    ("verbose_three", {"verbose": 3}, "verbose"),
    ("verbose_bool", {"verbose": True}, "verbose"),
    ("average", {"average": "median"}, "average"),
    ("pc", {"pc": "point"}, "pc"),
    (
        "navigation_mask_list",
        {"navigation_mask": [[False] * 10] * 8},
        "navigation mask must be a NumPy array",
    ),
    (
        "signal_mask_list",
        {"signal_mask": [[False] * 32] * 32},
        "signal mask must be a NumPy array",
    ),
    (
        "navigation_mask_dtype",
        {"navigation_mask": _nav_size_8x10_mask(0, dtype=np.uint8)},
        "boolean",
    ),
    ("signal_mask_dtype", {"signal_mask": np.zeros(SIG_SHAPE, np.uint8)}, "boolean"),
    (
        "navigation_mask_shape",
        {"navigation_mask": _nav_size_8x10_mask(shape=(8, 9))},
        "navigation mask shape",
    ),
    (
        "signal_mask_shape",
        {"signal_mask": np.zeros((16, 16), dtype=bool)},
        "signal mask shape",
    ),
    (
        "navigation_mask_all_true",
        {"navigation_mask": _nav_size_8x10_mask(True)},
        "at least one pattern",
    ),
    (
        "compat_absent_point",
        {
            "xmap": lambda: _one_phase_map(BASE_SHAPE, is_in_data=_absent_one()),
            "emsoft_compatible": True,
        },
        "emsoft_compatible requires every map point",
    ),
    (
        "compat_not_indexed_point",
        {
            "xmap": lambda: _one_phase_map(BASE_SHAPE, phase_id=_not_indexed_one()),
            "emsoft_compatible": True,
        },
        "emsoft_compatible requires every map point",
    ),
    (
        "compat_two_phases",
        {
            "xmap": lambda: _one_phase_map(BASE_SHAPE, phase_id=_two_phases()),
            "emsoft_compatible": True,
        },
        "one phase",
    ),
    (
        "compat_per_point_pc",
        {"detector": _per_point_detector, "emsoft_compatible": True},
        "pc='single'",
    ),
    # Two violations: the earlier check is reported
    (
        "nav_dims_before_xmap_shape",
        {"signal": lambda: _zero_signal((80,)), "n_osm": 0},
        "two navigation dimensions",
    ),
    (
        "xmap_shape_before_n_osm",
        {"xmap": lambda: _one_phase_map((8, 9)), "n_osm": 0},
        "xmap shape",
    ),
    (
        "detector_shape_before_average",
        {
            "detector": lambda: kp.detectors.EBSDDetector((16, 16), pc=PC),
            "average": "median",
        },
        "detector shape",
    ),
    ("n_osm_before_average", {"n_osm": 0, "average": "median"}, "n_osm"),
    (
        "average_before_masks",
        {"average": "median", "navigation_mask": [[False] * 10] * 8},
        "average",
    ),
    (
        "masks_before_compat",
        {
            "navigation_mask": [[False] * 10] * 8,
            "xmap": lambda: _one_phase_map(BASE_SHAPE, is_in_data=_absent_one()),
            "emsoft_compatible": True,
        },
        "navigation mask must be a NumPy array",
    ),
    (
        "all_true_mask_before_compat",
        {
            "navigation_mask": _nav_size_8x10_mask(True),
            "detector": _per_point_detector,
            "emsoft_compatible": True,
        },
        "at least one pattern",
    ),
]


class TestValidation:
    def test_signatures_are_frozen(self):
        for name, (obj, expected) in FROZEN_SIGNATURES.items():
            parameters = inspect.signature(obj).parameters.values()
            actual = [(p.name, p.kind, p.default) for p in parameters]
            assert actual == expected, name
            names = [p[0] for p in actual]
            assert "show_progressbar" not in names, name
            assert "max_chunk_bytes" not in names, name

    @pytest.mark.parametrize(
        "changes, fragment",
        [pytest.param(c, f, id=i) for i, c, f in VALIDATION_CASES],
    )
    def test_arguments_are_validated_in_order(self, base, changes, fragment):
        arguments = {
            "signal": _zero_signal(BASE_SHAPE),
            "xmap": base.xmap,
            "detector": base.det,
        }
        kwargs = {"energy": 20, "max_angle": BALL_MAX_ANGLE, "n_steps": BALL_N_STEPS}
        for key, value in changes.items():
            if key in arguments:
                arguments[key] = value()
            else:
                kwargs[key] = value
        signal = arguments["signal"]
        with pytest.raises(ValueError, match=re.escape(fragment)):
            signal.hrosm(arguments["xmap"], base.mp, arguments["detector"], **kwargs)

    def test_compat_point_group_without_emsoft_operators_raises(
        self, base, monkeypatch
    ):
        # EMsoft's operator table with one operator of m-3m dropped
        columns = _emsoft_quaternions._QSYM_INIT_COLUMNS
        monkeypatch.setitem(columns, 30, columns[30][:-1])
        with pytest.raises(ValueError, match="point group"):
            _hrosm(
                _zero_signal(BASE_SHAPE),
                base.xmap,
                base.mp,
                base.det,
                emsoft_compatible=True,
                average="center",
            )

    def test_master_pattern_must_be_in_the_lambert_projection(self, base):
        mp = base.mp.deepcopy()
        mp.projection = "stereographic"
        with pytest.raises(NotImplementedError, match="Lambert projection"):
            _hrosm(_zero_signal(BASE_SHAPE), base.xmap, mp, base.det)
        # Checked before the detector
        with pytest.raises(NotImplementedError, match="Lambert projection"):
            _hrosm(
                _zero_signal(BASE_SHAPE),
                base.xmap,
                mp,
                kp.detectors.EBSDDetector((16, 16), pc=PC),
            )

    @pytest.mark.parametrize("shape", [(1, 6), (6, 1)])
    def test_one_row_or_column_map_passes_the_xmap_shape_check(self, base, shape):
        xmap = _one_phase_map(shape)
        assert len(xmap.shape) == 1
        # The next check fails instead
        with pytest.raises(ValueError, match="n_osm"):
            _hrosm(_zero_signal(shape), xmap, base.mp, base.det, n_osm=0)


# -------------------------------- Output --------------------------------


class TestOutput:
    def test_one_crystal_map_with_the_input_frame(self, base, run_correct):
        out = run_correct
        xmap = base.xmap
        assert isinstance(out, CrystalMap)
        assert out.shape == xmap.shape
        assert out.size == xmap.size
        np.testing.assert_array_equal(out.x, xmap.x)
        np.testing.assert_array_equal(out.y, xmap.y)
        np.testing.assert_array_equal(out.phase_id, xmap.phase_id)
        assert out.phases.names == xmap.phases.names
        assert list(out.phases.ids) == list(xmap.phases.ids)
        np.testing.assert_array_equal(out.is_in_data, xmap.is_in_data)
        assert out.scan_unit == xmap.scan_unit
        assert out.rotations_per_point == 1
        # The input's properties are not carried
        assert "input_only" not in out.prop
        assert set(out.prop) == set(OUTPUT_PROPS)

    def test_properties_dtypes_shapes_and_fills(self, base, mode_run):
        emsoft_compatible, _, out = mode_run
        n = base.xmap.size
        keep_n = KEEP_N
        for name, (dtype, columns) in OUTPUT_PROPS.items():
            value = np.asarray(out.prop[name])
            assert value.dtype == dtype, name
            if columns is None:
                assert value.shape == (n,), name
            elif columns == "keep_n":
                assert value.shape == (n, keep_n), name
            else:
                assert value.shape == (n, columns), name

        grain_id = out.prop["grain_id"]
        reindexed = out.prop["reindexed"]
        # Both grains re-indexed; the two boundary columns unassigned
        expected_id = _input_labels(
            base.xmap, emsoft_compatible=emsoft_compatible
        ).ravel()
        np.testing.assert_array_equal(grain_id, expected_id)
        assert set(np.unique(grain_id)) == {0, 1, 2}
        np.testing.assert_array_equal(reindexed, grain_id > 0)

        kam = kernel_average_misorientation_map(
            base.xmap, emsoft_compatible=emsoft_compatible
        )
        np.testing.assert_array_equal(out.prop["kam"], kam.ravel())

        osm = out.prop["osm"]
        scores = out.prop["scores"]
        sim = out.prop["simulation_indices"]
        rot = out.rotations.data
        assert np.all(np.isfinite(osm[reindexed]))
        assert np.all(np.isfinite(scores[reindexed]))
        assert np.all(sim[reindexed] >= 0)
        assert np.all(sim[reindexed] < BALL_SIZE)
        assert np.all(sim[~reindexed] == -1)
        if emsoft_compatible:
            assert np.all(osm[~reindexed] == 0)
            assert np.all(scores[~reindexed] == 0)
            np.testing.assert_array_equal(
                rot[~reindexed], np.tile([1.0, 0.0, 0.0, 0.0], (np.sum(~reindexed), 1))
            )
        else:
            assert np.all(np.isnan(osm[~reindexed]))
            assert np.all(np.isnan(scores[~reindexed]))
            np.testing.assert_array_equal(
                rot[~reindexed], base.xmap.rotations.data[~reindexed]
            )
        outside = grain_id == 0
        assert np.all(np.isnan(out.prop["grain_orientation"][outside]))
        assert np.all(np.isnan(out.prop["grain_kappa"][outside]))
        assert np.all(np.isnan(out.prop["grain_max_grod"][outside]))
        assert np.all(np.isnan(out.prop["grod"][outside]))
        assert np.all(np.isfinite(out.prop["grod"][~outside]))

    def test_osm_equals_the_osm_of_the_kept_best_matches(self, base, mode_run):
        # The map compares the first n_osm of the keep_n best matches:
        # grain-aware over the re-indexed grain points in correct mode;
        # EMsoft's map over the whole bounding box, copied back for the
        # grain's points only, in EMsoft compatible mode
        emsoft_compatible, _, out = mode_run
        osm = out.prop["osm"]
        grain_id = out.prop["grain_id"]
        reindexed = out.prop["reindexed"]
        assert reindexed.any()
        if not emsoft_compatible:
            sim = out.prop["simulation_indices"]
            labels = grain_id.reshape(BASE_SHAPE)
            mask = reindexed.reshape(BASE_SHAPE)
            expected = {
                n: _osm_grain_aware(sim, labels, mask, n).ravel()
                for n in (N_OSM, KEEP_N)
            }
        else:
            calls = _CACHE["compat_matching"]
            boxes = grain_bounding_boxes(grain_id.reshape(BASE_SHAPE))
            labels = np.unique(grain_id[reindexed])
            assert len(calls) == labels.size
            expected = {
                n: np.full(osm.shape, np.nan, np.float32) for n in (N_OSM, KEEP_N)
            }
            for call, label in zip(calls, labels):
                r0, c0, h, w = boxes[label - 1]
                assert call["nav_size"] == h * w
                rows, cols = np.divmod(np.flatnonzero(grain_id == label), BASE_SHAPE[1])
                for n in (N_OSM, KEEP_N):
                    box_osm = _osm_emsoft(call["simulation_indices"], h, w, n)
                    expected[n][rows * BASE_SHAPE[1] + cols] = box_osm[
                        rows - r0, cols - c0
                    ]
        np.testing.assert_array_equal(osm[reindexed], expected[N_OSM][reindexed])
        # The map of all keep_n best matches differs
        assert not np.array_equal(osm[reindexed], expected[KEEP_N][reindexed])

    def test_grod_props_equal_the_public_deviation_map(self, base, mode_run):
        _, _, out = mode_run
        self._assert_grod_props_equal_the_public_map(base, out)

    def test_grod_props_of_a_grain_skipped_by_min_pixels(self, base):
        n_a = int(base.table.n_pixels[base.label_a - 1])
        out = _cached_run(
            "min_pixels_above", base.s, base.xmap, base.mp, base.det, min_pixels=n_a + 1
        )
        assert not out.prop["reindexed"].any()
        self._assert_grod_props_equal_the_public_map(base, out)
        in_a = out.prop["grain_id"] == base.label_a
        assert in_a.any()
        assert np.all(np.isfinite(out.prop["grod"][in_a]))
        assert np.all(np.isfinite(out.prop["grain_max_grod"][in_a]))

    @staticmethod
    def _assert_grod_props_equal_the_public_map(base, out):
        table = GrainTable.from_crystal_map(out)
        grain_id = out.prop["grain_id"].reshape(BASE_SHAPE)
        expected = grain_reference_orientation_deviation_map(base.xmap, grain_id, table)
        assert expected.dtype == np.float32
        np.testing.assert_array_equal(out.prop["grod"], expected.ravel())
        labels = grain_id.ravel()
        expected_max = np.full(labels.size, np.nan, dtype=np.float32)
        inside = labels > 0
        expected_max[inside] = table.max_grod[labels[inside] - 1]
        np.testing.assert_array_equal(out.prop["grain_max_grod"], expected_max)

    def test_simulation_indices_are_local_to_the_grain_ball(self, mode_run):
        emsoft_compatible, _, out = mode_run
        table = GrainTable.from_crystal_map(out)
        grain_id = out.prop["grain_id"]
        reindexed = out.prop["reindexed"]
        sim = out.prop["simulation_indices"]
        rotations = out.rotations.data
        checked = 0
        for label in np.unique(grain_id[reindexed]):
            ball = misorientation_ball(
                table.rotation[label - 1],
                max_angle=BALL_MAX_ANGLE,
                n_steps=BALL_N_STEPS,
                emsoft_compatible=emsoft_compatible,
            )
            assert ball.size == BALL_SIZE
            points = np.flatnonzero(reindexed & (grain_id == label))
            np.testing.assert_allclose(
                ball.data[sim[points, 0]], rotations[points], atol=1e-12, rtol=0
            )
            checked += points.size
        assert checked == int(np.sum(reindexed)) > 0

    def test_grain_table_round_trips_through_the_crystal_map(
        self, base, mode_run, tmp_path
    ):
        emsoft_compatible, average, out = mode_run
        grain_id = out.prop["grain_id"].reshape(BASE_SHAPE)
        expected = _input_table(
            base.xmap,
            grain_id,
            method=average,
            max_angle=BALL_MAX_ANGLE,
            emsoft_compatible=emsoft_compatible,
        )
        table = GrainTable.from_crystal_map(out)
        assert table.method is None
        _assert_tables_equal(table, expected)

        fname = tmp_path / "hrosm.h5"
        orix_io.save(fname, out)
        loaded = orix_io.load(fname)
        _assert_tables_equal(GrainTable.from_crystal_map(loaded), expected)
        for name in OUTPUT_PROPS:
            np.testing.assert_array_equal(
                np.asarray(loaded.prop[name]).reshape(out.prop[name].shape),
                out.prop[name],
                err_msg=name,
            )

    def test_default_average_is_mean(self, base, run_correct):
        out = run_correct
        mean = _input_table(base.xmap, base.grain_id, method="mean")
        center = _input_table(base.xmap, base.grain_id, method="center")
        labels = out.prop["grain_id"]
        inside = labels > 0
        np.testing.assert_array_equal(
            out.prop["grain_kappa"][inside], mean.kappa[labels[inside] - 1]
        )
        # "center" would give 1.0
        assert not np.array_equal(mean.kappa, center.kappa)


# ------------------------------- Contracts ------------------------------


class TestContracts:
    def test_each_grain_is_matched_against_its_own_ball(self, base, mode_run):
        emsoft_compatible, _, out = mode_run
        reindexed = out.prop["reindexed"]
        assert reindexed.any()
        rotations = out.rotations.data[reindexed]
        centres = out.prop["grain_orientation"][reindexed]
        tolerance = 1e-4 if emsoft_compatible else 1e-9
        to_centre = _angle_deg(rotations, centres)
        assert np.all(to_centre <= BALL_MAX_ANGLE + tolerance)
        truth = base.xmap_truth.rotations.data[reindexed]
        error = _disorientation_deg(rotations, truth)
        assert np.all(error <= TRUTH_TOLERANCE_DEG)

        # Per grain, the patterns pick distinct ball orientations closer
        # to the truth than the input map, not the grain average alone
        grain_id = out.prop["grain_id"][reindexed]
        sim = out.prop["simulation_indices"][reindexed]
        input_error = _disorientation_deg(base.xmap.rotations.data[reindexed], truth)
        for label in np.unique(grain_id):
            in_grain = grain_id == label
            assert np.unique(sim[in_grain, 0]).size > 1
            median = np.median(error[in_grain])
            assert median <= TRUTH_MEDIAN_TOLERANCE_DEG
            assert median < np.median(input_error[in_grain])

        # Best matches first
        scores = out.prop["scores"][reindexed]
        assert np.all(np.diff(scores, axis=1) <= 0)

    def test_navigation_masked_dictionary_indexing_map_keeps_its_grid(self, base):
        dictionary = base.mp.get_patterns(
            base.xmap.rotations, base.det, energy=20, compute=True
        )
        mask = np.zeros(BASE_SHAPE, dtype=bool)
        mask[0] = True
        xmap_di = base.s.dictionary_indexing(dictionary, navigation_mask=mask, keep_n=1)
        np.testing.assert_array_equal(xmap_di.is_in_data, ~mask.ravel())
        assert xmap_di.shape == (7, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _hrosm(base.s, xmap_di, base.mp, base.det)
        np.testing.assert_array_equal(out.is_in_data, xmap_di.is_in_data)
        grid, shape = _map_grid(out)
        assert shape == BASE_SHAPE
        assert np.all(grid[0] == -1)
        grain_id = _grid_values(out, out.prop["grain_id"], -1)
        left = np.unique(grain_id[1:, 0:4])
        right = np.unique(grain_id[1:, 6:10])
        assert left.size == right.size == 1
        assert left[0] > 0
        assert right[0] > 0
        assert left[0] != right[0]
        assert np.all(grain_id[1:, 4:6] == 0)

        reindexed = _grid_values(out, out.prop["reindexed"], False)
        assert np.all(reindexed[1:, 0:4])
        assert np.all(reindexed[1:, 6:10])
        rotations = _grid_rotations(out)[reindexed]
        truth = base.xmap_truth.rotations.data.reshape(BASE_SHAPE + (4,))[reindexed]
        assert np.all(_disorientation_deg(rotations, truth) <= TRUTH_TOLERANCE_DEG)

    @staticmethod
    def _mask() -> np.ndarray:
        mask = np.zeros(BASE_SHAPE, dtype=bool)
        for point in MASKED_POINTS:
            mask[point] = True
        return mask

    @classmethod
    def _masked_run(cls, base) -> CrystalMap:
        if "masked" not in _CACHE:
            mask = cls._mask()
            s = base.s.deepcopy()
            s.data[mask] = 0
            _CACHE["masked_signal"] = s
        return _cached_run(
            "masked",
            _CACHE["masked_signal"],
            base.xmap,
            base.mp,
            base.det,
            navigation_mask=cls._mask(),
        )

    def test_navigation_mask_true_excludes_and_fills(self, base, run_correct):
        out = self._masked_run(base)
        masked = self._mask().ravel()
        assert np.all(np.isnan(out.prop["kam"][masked]))
        assert np.all(out.prop["grain_id"][masked] == 0)
        assert not out.prop["reindexed"][masked].any()
        np.testing.assert_array_equal(
            out.rotations.data[masked], base.xmap.rotations.data[masked]
        )
        assert np.all(np.isnan(out.prop["osm"][masked]))
        assert np.all(np.isnan(out.prop["scores"][masked]))
        assert np.all(out.prop["simulation_indices"][masked] == -1)
        assert np.all(np.isnan(out.prop["grod"][masked]))
        # The rest of grain A is still re-indexed
        in_a = run_correct.prop["grain_id"] == base.label_a
        assert np.all(out.prop["reindexed"][in_a & ~masked])

        # Grain B is unchanged, bit for bit
        in_b = run_correct.prop["grain_id"] == base.label_b
        assert in_b.any()
        np.testing.assert_array_equal(
            out.rotations.data[in_b], run_correct.rotations.data[in_b]
        )
        for name in OUTPUT_PROPS:
            np.testing.assert_array_equal(
                out.prop[name][in_b], run_correct.prop[name][in_b], err_msg=name
            )

    @pytest.mark.parametrize("absent_by", ["is_in_data", "phase_id"])
    def test_absent_points_behave_like_masked_points(self, base, absent_by):
        masked = self._mask().ravel()
        data = base.xmap.rotations.data
        if absent_by == "is_in_data":
            xmap = _crystal_map(data, BASE_SHAPE, is_in_data=~masked)
        else:
            phase_id = np.where(masked, -1, 0).astype(np.int32)
            xmap = _crystal_map(data, BASE_SHAPE, phase_id=phase_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _hrosm(base.s, xmap, base.mp, base.det)
        reference = self._masked_run(base)

        np.testing.assert_array_equal(out.is_in_data, xmap.is_in_data)
        grid, _ = _map_grid(out)
        flat = grid.ravel()
        positions = np.flatnonzero(flat >= 0)
        order = flat[positions]
        np.testing.assert_array_equal(
            out.rotations.data[order], reference.rotations.data[positions]
        )
        for name in OUTPUT_PROPS:
            np.testing.assert_array_equal(
                np.asarray(out.prop[name])[order],
                reference.prop[name][positions],
                err_msg=name,
            )

    def test_min_pixels_boundary_is_inclusive(self, base):
        n_a = int(base.table.n_pixels[base.label_a - 1])
        equal = _cached_run(
            "min_pixels_equal", base.s, base.xmap, base.mp, base.det, min_pixels=n_a
        )
        in_a = equal.prop["grain_id"] == base.label_a
        assert np.all(equal.prop["reindexed"][in_a])

        above = _cached_run(
            "min_pixels_above", base.s, base.xmap, base.mp, base.det, min_pixels=n_a + 1
        )
        in_a = above.prop["grain_id"] == base.label_a
        assert in_a.sum() == n_a
        assert not above.prop["reindexed"][in_a].any()
        assert np.all(np.isnan(above.prop["osm"][in_a]))
        assert np.all(np.isnan(above.prop["scores"][in_a]))
        assert np.all(above.prop["simulation_indices"][in_a] == -1)
        np.testing.assert_array_equal(
            above.rotations.data[in_a], base.xmap.rotations.data[in_a]
        )

    def test_invalid_grains_are_skipped(self, base):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = _hrosm(
                base.s,
                base.xmap,
                base.mp,
                base.det,
                average="vmf",
                min_kappa=1e12,
                n_em=2,
                n_iter=5,
                seed=0,
            )
        assert not out.prop["reindexed"].any()
        assert np.all(np.isnan(out.prop["osm"]))
        assert np.all(np.isnan(out.prop["grod"]))
        table = GrainTable.from_crystal_map(out)
        assert table.n_grains == 2
        assert not table.valid.any()
        messages = _user_warnings(record)
        assert len(messages) == 1
        assert "no grain re-indexed" in messages[0]

    def test_compat_domain_is_the_bounding_box(self, base, monkeypatch):
        grain_compat = _input_labels(base.xmap, emsoft_compatible=True, dilate=True)
        boxes = grain_bounding_boxes(grain_compat)
        n_compat = np.bincount(grain_compat.ravel())[1:]
        box_size = boxes[:, 2] * boxes[:, 3]
        # Compat dilation leaves the first row: the boxes hold more
        # points than the grains
        assert np.all(box_size > n_compat)
        grain_correct = _input_labels(base.xmap, dilate=True)
        n_correct = np.bincount(grain_correct.ravel())[1:]

        calls = _record_matching(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_compat = _hrosm(
                base.s,
                base.xmap,
                base.mp,
                base.det,
                dilate=True,
                average="center",
                emsoft_compatible=True,
            )
        assert [c["n_experimental"] for c in calls] == box_size.tolist()
        assert [c["nav_size"] for c in calls] == box_size.tolist()

        calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _hrosm(base.s, base.xmap, base.mp, base.det, dilate=True)
        assert [c["n_experimental"] for c in calls] == n_correct.tolist()

        # Only the grain's points are copied back
        grain_id = out_compat.prop["grain_id"]
        np.testing.assert_array_equal(grain_id, grain_compat.ravel())
        reindexed = out_compat.prop["reindexed"]
        np.testing.assert_array_equal(reindexed, grain_id > 0)
        outside = grain_id == 0
        assert outside.any()
        assert np.all(out_compat.prop["simulation_indices"][outside] == -1)
        assert np.all(out_compat.prop["osm"][outside] == 0)
        assert np.all(out_compat.prop["scores"][outside] == 0)

    @pytest.mark.parametrize(
        "case", ["one_pc", "per_point_grain", "per_point_single", "compat_grain"]
    )
    def test_pc_policy(self, base, monkeypatch, case):
        detector = base.det if case == "one_pc" else _per_point_detector()
        if case == "compat_grain":
            with pytest.raises(ValueError, match=re.escape("pc='single'")):
                _hrosm(
                    base.s,
                    base.xmap,
                    base.mp,
                    detector,
                    average="center",
                    emsoft_compatible=True,
                )
            return

        pc = "single" if case == "per_point_single" else "grain"
        calls = _record_simulations(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _hrosm(base.s, base.xmap, base.mp, detector, pc=pc)
        labels = np.unique(out.prop["grain_id"][out.prop["reindexed"]])
        assert labels.size == 2
        assert len(calls) == labels.size
        all_pc = np.asarray(detector.pc, dtype=np.float64).reshape(-1, 3)
        for call, label in zip(calls, labels):
            if case == "one_pc":
                expected = all_pc[0]
            elif case == "per_point_single":
                expected = np.asarray(detector.pc_average, dtype=np.float64)
            else:
                domain = out.prop["grain_id"] == label
                expected = all_pc[domain].mean(axis=0)
            assert call["pc"].shape[0] in (1, call["n"])
            np.testing.assert_allclose(
                call["pc"], np.broadcast_to(expected, call["pc"].shape), atol=1e-15
            )

    def test_multi_phase_uses_the_master_by_phase_name(self, base, monkeypatch):
        phase_id = (base.truth == 2).astype(np.int32)
        xmap = _crystal_map(base.xmap.rotations.data, BASE_SHAPE, phase_id=phase_id)
        assert xmap.phases.names == ["ni", "ni2"]
        assert base.mp.phase.name == "ni"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = _hrosm(base.s, xmap, base.mp, base.det)
        messages = [m for m in _user_warnings(record) if "no master pattern" in m]
        assert len(messages) == 1
        in_b = base.truth.ravel() == 2
        in_a = base.truth.ravel() == 1
        assert not out.prop["reindexed"][in_b].any()
        assert np.all(np.isnan(out.prop["osm"][in_b]))
        assert out.prop["reindexed"][in_a].any()
        # Grains of the other phase are still segmented and averaged
        assert np.any(out.prop["grain_id"][in_b] > 0)

        renamed = base.mp.deepcopy()
        renamed.phase.name = "cu"
        assert base.mp.phase.name == "ni"
        with pytest.raises(ValueError, match="master pattern phase"):
            _hrosm(base.s, xmap, renamed, base.det)

        # A single-phase map uses the master whatever its name
        used = []
        _stop_at_first_simulation(monkeypatch, lambda mp, det: used.append(mp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(_StopRun):
                _hrosm(base.s, base.xmap, renamed, base.det)
        assert used[0].phase.name == "cu"

    def test_compat_dilate_vanished_grain_is_not_reindexed(
        self, hrosm_synthetic_signal
    ):
        shape = (6, 7)
        n = shape[0] * shape[1]
        steps = np.random.default_rng(VANISHING_SEED).integers(0, 3, size=shape)
        g0 = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0)))
        rot = _axis_angle((0, 0, 1), steps.ravel() * 1.0) * Rotation(
            np.repeat(g0.data.reshape(1, 4), n, axis=0)
        )
        xmap = _crystal_map(rot.data, shape)
        # Precondition: compat dilation removes grain 1
        undilated = _input_labels(
            xmap, emsoft_compatible=True, threshold=VANISHING_THRESHOLD
        )
        dilated = _input_labels(
            xmap, emsoft_compatible=True, threshold=VANISHING_THRESHOLD, dilate=True
        )
        assert np.any(undilated == 1)
        assert not np.any(dilated == 1)
        assert dilated.max() == 2

        s, det, mp = hrosm_synthetic_signal(xmap, sig_shape=SIG_SHAPE, pc=PC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = s.hrosm(
                xmap,
                mp,
                det,
                energy=20,
                threshold=VANISHING_THRESHOLD,
                max_angle=1.0,
                n_steps=1,
                keep_n=5,
                n_osm=5,
                average="center",
                dilate=True,
                emsoft_compatible=True,
                verbose=0,
            )
        np.testing.assert_array_equal(out.prop["grain_id"], dilated.ravel())
        table = GrainTable.from_crystal_map(out)
        assert table.n_grains == 2
        assert table.n_pixels[0] == 0
        assert not table.valid[0]
        assert table.valid[1]
        assert out.prop["reindexed"][dilated.ravel() == 2].all()


# ------------------------------ Invariance ------------------------------


class TestInvariance:
    @staticmethod
    def _chunked(base, n_per_iteration) -> CrystalMap:
        return _cached_run(
            f"n_per_iteration_{n_per_iteration}",
            base.s,
            base.xmap,
            base.mp,
            base.det,
            n_per_iteration=n_per_iteration,
        )

    def test_results_do_not_depend_on_n_per_iteration(self, base, run_correct):
        # The default at this pattern size is the ball size, one pass
        # over the ball; 50 is a chunked loop with a short last chunk
        assert _default_n_per_iteration(SIG_SHAPE[0] * SIG_SHAPE[1], BALL_SIZE) == (
            BALL_SIZE
        )
        _assert_outputs_invariant(self._chunked(base, 50), run_correct)

    def test_lazy_signal_equals_eager(self, base, run_correct):
        s_lazy = base.s.as_lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _hrosm(s_lazy, base.xmap, base.mp, base.det)
        _assert_outputs_invariant(out, run_correct)

    def test_seeded_runs_are_identical(self, base):
        outputs = []
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs.append(
                    _hrosm(
                        base.s,
                        base.xmap,
                        base.mp,
                        base.det,
                        average="watson",
                        n_em=3,
                        n_iter=10,
                        seed=0,
                    )
                )
        assert outputs[0].prop["reindexed"].any()
        _assert_outputs_equal(outputs[0], outputs[1])

    def test_default_n_per_iteration(self):
        assert _default_n_per_iteration(60 * 60, 68_921) == 17_777
        assert _default_n_per_iteration(480 * 480, 68_921) == 277
        # The ball size when the floor exceeds it
        assert _default_n_per_iteration(32 * 32, BALL_SIZE) == BALL_SIZE
        # At least 1
        assert _default_n_per_iteration(10**9, BALL_SIZE) == 1
        assert isinstance(_default_n_per_iteration(60 * 60, 68_921), int)

    def test_simulation_indices_are_int32_on_both_paths(self, base, run_correct):
        # One pass over the ball and a chunked loop
        for out in (run_correct, self._chunked(base, 50)):
            assert out.prop["simulation_indices"].dtype == np.int32
            assert out.prop["scores"].dtype == np.float32


# ------------------------------- Messages -------------------------------


def _numbers(line: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", line)


class TestMessages:
    def test_verbose_levels(self, base, run_correct, capsys):
        # The cached default run, verbose 0, wrote nothing
        assert _CACHE["correct_output"] == ("", "")
        out0 = run_correct
        n_grains = int(np.unique(out0.prop["grain_id"][out0.prop["reindexed"]]).size)
        assert n_grains == 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _hrosm(base.s, base.xmap, base.mp, base.det, verbose=1)
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        # One information line and one timing line
        assert len(lines) == 2
        info = [line for line in lines if str(BALL_SIZE) in _numbers(line)]
        assert len(info) == 1
        numbers = _numbers(info[0])
        assert any(float(x) == BALL_MAX_ANGLE for x in numbers)
        spacing = misorientation_ball_spacing(BALL_MAX_ANGLE, BALL_N_STEPS)
        printed = [
            x
            for x in numbers
            if "." in x
            and len(x.split(".")[1]) >= 2
            and abs(float(x) - spacing) <= 0.5 * 10 ** -len(x.split(".")[1]) + 1e-12
        ]
        assert printed, info[0]
        # One grain-level progress bar, no dictionary indexing output
        assert f"{n_grains}/{n_grains}" in captured.err
        assert "Dictionary indexing information" not in captured.out
        assert "Indexing speed" not in captured.out

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _hrosm(base.s, base.xmap, base.mp, base.det, verbose=2)
        captured = capsys.readouterr()
        assert captured.out.count("Dictionary indexing information") == n_grains
        assert captured.out.count("Indexing speed") == n_grains

    def test_warnings_are_issued_once_per_call(self, base):
        phase_id = (base.truth == 2).astype(np.int32)
        xmap = _crystal_map(base.xmap.rotations.data, BASE_SHAPE, phase_id=phase_id)
        grain_id = _input_labels(xmap, phase_id=phase_id)
        table = _input_table(xmap, grain_id)
        max_grod_a = float(table.max_grod[int(grain_id[2, 1]) - 1])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _hrosm(base.s, xmap, base.mp, base.det, max_angle=0.5 * max_grod_a)
        messages = _user_warnings(record)
        ball = _ball_warnings(record)
        assert len(ball) == 1
        assert len([m for m in messages if "no master pattern" in m]) == 1
        # Only grain A, of the phase of the master, is listed, though
        # grain B exceeds the radius too
        label_a, label_b = int(grain_id[2, 1]), int(grain_id[2, 8])
        max_angle = 0.5 * max_grod_a
        assert float(table.max_grod[label_b - 1]) > max_angle
        assert ball[0] == _coverage_warning_message(
            np.array([label_a]),
            table.max_grod[[label_a - 1]],
            table.n_pixels[[label_a - 1]],
            max_angle,
            misorientation_ball_spacing(max_angle, BALL_N_STEPS),
        )
        assert "1 grain" in ball[0]
        assert f"grain {label_a} (" in ball[0]
        assert f"grain {label_b} (" not in ball[0]
        assert len([m for m in messages if "no grain re-indexed" in m]) == 0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _hrosm(base.s, base.xmap, base.mp, base.det, min_pixels=1000)
        messages = _user_warnings(record)
        assert len([m for m in messages if "no grain re-indexed" in m]) == 1
        assert len(_ball_warnings(record)) == 0

    # ------------------- Coverage warning before the run ------------------

    @staticmethod
    def _run_until_first_simulation(
        monkeypatch, signal, xmap, mp, det, ball_seen=None, **kwargs
    ):
        """Return the warning messages recorded before the first
        simulation and all recorded messages.

        If ``ball_seen`` is a list, the cubochoric grid of the ball is
        wrapped in a spy appending the messages recorded at each call.
        """
        seen = []
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _stop_at_first_simulation(
                monkeypatch, lambda *_: seen.append(list(_user_warnings(record)))
            )
            if ball_seen is not None:
                original = _sampling._cubochoric_grid

                def spy(*args, **kw):
                    ball_seen.append(list(_user_warnings(record)))
                    return original(*args, **kw)

                monkeypatch.setattr(_sampling, "_cubochoric_grid", spy)
                monkeypatch.setattr(_driver, "_cubochoric_grid", spy, raising=False)
            with pytest.raises(_StopRun):
                _hrosm(signal, xmap, mp, det, **kwargs)
        assert len(seen) == 1
        return seen[0], _user_warnings(record)

    def test_coverage_warning_precedes_the_first_simulation(self, base, monkeypatch):
        max_grod_a = float(base.table.max_grod[base.label_a - 1])
        max_angle = 0.5 * max_grod_a
        ball_seen = []
        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(BASE_SHAPE),
            base.xmap,
            base.mp,
            base.det,
            ball_seen=ball_seen,
            max_angle=max_angle,
        )
        ball = [m for m in before if "misorientation ball" in m]
        assert len(ball) == 1
        assert "max GROD" in ball[0]
        # The mean spacing samples a ball before the warning; the ball
        # of the dictionary, the last one sampled before the first
        # simulation, is built after it
        assert ball_seen
        assert ball[0] in ball_seen[-1]

    def test_coverage_warning_lists_grains_in_descending_max_grod(
        self, hrosm_grain_xmap, base, monkeypatch
    ):
        shape = (8, 12)
        xmap3, truth3 = hrosm_grain_xmap(shape=shape, n_grains=3, gradient=0.2)
        rows = np.arange(xmap3.size) // shape[1]
        # Distinct spreads: an extra gradient along the rows per grain
        extra = np.array([0.0, 0.0, 0.25, 0.5])[truth3.ravel()]
        rot = _axis_angle((1, 0, 0), rows * extra) * xmap3.rotations
        xmap = _crystal_map(rot.data, shape)
        grain_id = _input_labels(xmap)
        table = _input_table(xmap, grain_id)
        assert table.n_grains == 3
        assert np.all(table.n_pixels >= 10)
        assert np.unique(table.max_grod).size == 3
        max_angle = 0.5 * float(table.max_grod.min())
        spacing = misorientation_ball_spacing(max_angle, BALL_N_STEPS)

        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(shape),
            xmap,
            base.mp,
            base.det,
            max_angle=max_angle,
        )
        ball = [m for m in before if "misorientation ball" in m]
        assert len(ball) == 1
        message = ball[0]
        labels = np.arange(1, 4)
        assert message == _coverage_warning_message(
            labels, table.max_grod, table.n_pixels, max_angle, spacing
        )
        assert "3 grain" in message
        assert "max GROD" in message
        assert f"{max_angle:g}" in message
        assert f"{spacing:.4f}" in message
        entries = re.findall(
            r"grain (\d+) \(max GROD ([\d.]+) deg, (\d+) points\)", message
        )
        order = np.argsort(-table.max_grod.astype(np.float64), kind="stable")
        assert [int(e[0]) for e in entries] == (order + 1).tolist()
        assert [int(e[2]) for e in entries] == table.n_pixels[order].tolist()

    def test_coverage_warning_is_silent_at_equality(self, base, monkeypatch):
        m = np.float32(np.max(base.table.max_grod[base.table.valid]))
        assert m.dtype == np.float32
        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(BASE_SHAPE),
            base.xmap,
            base.mp,
            base.det,
            max_angle=float(m),
        )
        assert not [x for x in before if "misorientation ball" in x]

        below = float(np.nextafter(m, np.float32(0)))
        assert below < float(m)
        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(BASE_SHAPE),
            base.xmap,
            base.mp,
            base.det,
            max_angle=below,
        )
        assert len([x for x in before if "misorientation ball" in x]) == 1

        # Compared in float64: a quarter float32 ulp below still warns
        ulp = float(m) - float(np.nextafter(m, np.float32(0)))
        just_below = float(m) - 0.25 * ulp
        assert np.float32(just_below) == m
        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(BASE_SHAPE),
            base.xmap,
            base.mp,
            base.det,
            max_angle=just_below,
        )
        assert len([x for x in before if "misorientation ball" in x]) == 1

    def test_skipped_grains_do_not_trigger_the_coverage_warning(
        self, hrosm_grain_xmap, base, monkeypatch
    ):
        shape = (8, 11)
        xmap_truth, truth = hrosm_grain_xmap(shape=shape, n_grains=2, gradient=0.2)
        in_b = truth.ravel() == 2
        g0 = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0)))
        g_b = (_axis_angle((1, 1, 1), 30.0) * g0).data
        data = xmap_truth.rotations.data.copy()
        data[in_b] = g_b
        data = _with_indexing_error(data, mask=~in_b)
        xmap = _crystal_map(data, shape)
        grain_id = _input_labels(xmap)
        label_a = int(grain_id[2, 1])
        label_b = int(grain_id[2, 9])
        table = _input_table(xmap, grain_id)
        n_a = int(table.n_pixels[label_a - 1])
        assert n_a == 32
        assert int(table.n_pixels[label_b - 1]) == 40
        max_angle = 0.5 * float(table.max_grod[label_a - 1])
        assert float(table.max_grod[label_b - 1]) < max_angle

        before, _ = self._run_until_first_simulation(
            monkeypatch,
            _zero_signal(shape),
            xmap,
            base.mp,
            base.det,
            max_angle=max_angle,
            min_pixels=n_a + 1,
        )
        assert not [x for x in before if "misorientation ball" in x]

        # The free function warns and lists grain A
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            average_grain_orientations(xmap, grain_id, max_angle=max_angle)
        ball = _ball_warnings(record)
        assert len(ball) == 1
        assert f"grain {label_a} (" in ball[0]
