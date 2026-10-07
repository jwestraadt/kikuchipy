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

"""Regression of the HROSM orientation engine against EMsoftOO's own
output, and the discipline of the ``_hrosm`` package.

Every comparison with a file written by an EMsoftOO program lives in
this module:

- ``TestModuleDiscipline``: static checks of the ``_hrosm`` sources
  (no numba, the EMsoftOO notice, nothing LGPL ported, no print, the
  annotation style, the docstrings of the public names).
- ``TestCompatKAMOnEMsoftFiles``: the EMsoft compatible kernel average
  misorientation on the Euler angles of EMsoft's dictionary indexing
  and refinement files, against the KAM maps EMsoft wrote from them,
  on the points where no credited pair is degenerate.
- ``TestCompatOSMOnEMsoftFiles``: the EMsoft compatible orientation
  similarity map on EMsoft's top-match lists.
- ``TestClusterStage``: segmentation, dilation, bounding boxes and
  grain averages against EMHROSM's cluster stage.
- ``TestReferenceFiles``, ``TestEMsoftFileReader``,
  ``TestEMsoftProgramLock`` and ``TestRegenerateReferences``: the
  shipped references, the private EMsoft file reader, the lock shared
  by every process running an EMsoft program, and the regeneration of
  the references with the EMsoftOO programs.

Three gates: the default suite reads only the shipped
``regression_hrosm_*.npz`` files; the local arms need
``KIKUCHIPY_EMSOFT_DATA`` (Johan's historical EMsoft runs, md5
asserted) and the bin arm ``KIKUCHIPY_EMSOFT_BIN`` (the EMsoftOO
programs and a GPU). Both skip with a message naming what is missing.

Constants marked "measured-then-pinned" hold the drafting seed until
the implementation gate measures and pins them.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import importlib
import os
from pathlib import Path
import re
import sys
import time
import tokenize
import warnings

import h5py
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest
from scipy.ndimage import binary_dilation

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.data._registry import _registry_hashes
import kikuchipy.indexing._hrosm as hrosm_package
from kikuchipy.indexing._hrosm._averaging import average_grain_orientations
from kikuchipy.indexing._hrosm._emsoft_file import (
    parse_namelist_text,
    read_emsoft_dot_product_file,
    read_emsoft_hrosm_file,
)
from kikuchipy.indexing._hrosm._kam import kernel_average_misorientation_map
from kikuchipy.indexing._hrosm._osm import _osm_emsoft
from kikuchipy.indexing._hrosm._segmentation import (
    grain_bounding_boxes,
    segment_grains_kam,
)

# ------------------------- Frozen constants ------------------------- #

# The large Ni map of the references: 55 rows, 75 columns
MAP_SHAPE = (55, 75)
N_POINTS = MAP_SHAPE[0] * MAP_SHAPE[1]

# The three EMHROSM runs: the averaging method and whether grains are
# dilated, as written in their namelists
SCENARIOS_HROSM = ("center", "center_dilate", "wat")
SCENARIO_ORAV = {"center": "center", "center_dilate": "center", "wat": "averageWAT"}
SCENARIO_DILATE = {"center": False, "center_dilate": True, "wat": False}

# EMHROSM's namelist of the shipped runs
GANGLE = 5.0
MISORANG = 5.0
NOSM = 10
N_EM = 25
N_ITER = 40
# EMHROSM skips grains with fewer points
EMSOFT_MIN_PIXELS = 10
# EMHROSM keeps a Watson average only if its concentration is above
EMSOFT_MIN_KAPPA = 5.0

# The cached master pattern file every reference run started from
MASTER_MD5 = "8b69c071a036ad3488d465093b67fe4d"

# The programs and libraries every reference names with their md5
EMSOFT_BINARIES = (
    "EMDI.exe",
    "EMFitOrientation.exe",
    "EMHROSM.exe",
    "EMgetOSM.exe",
    "EMsampleRFZ.exe",
    "EMsoftOOLib.dll",
    "EMOpenCLLib.dll",
)

# Per file budget of a shipped reference in bytes
FILE_BUDGET_BYTES = 250_000

# The frozen key table of the shipped references: dtype and shape per
# array, "n" standing for the number of grains of the file
_SCENARIO_ARRAYS = {
    "nGrains": (np.int32, ()),
    "grainID": (np.int32, MAP_SHAPE),
    "npixels": (np.int32, ("n",)),
    "grainROI": (np.int32, ("n", 4)),
    "avor": (np.float64, ("n", 4)),
    "kappa": (np.float64, ("n",)),
    "kam": (np.float32, MAP_SHAPE),
    "newOSM": (np.float32, MAP_SHAPE),
    "newCI": (np.float32, MAP_SHAPE),
    "newEuler": (np.float32, MAP_SHAPE + (3,)),
    "hrosm_namelist": (np.str_, ()),
}
REFERENCE_TABLE = {
    "large_di": {
        "TopMatchIndices": (np.int32, (N_POINTS, 10)),
        "KAM": (np.float32, MAP_SHAPE),
        "OSM": (np.float32, MAP_SHAPE),
        "OSM_05": (np.float32, MAP_SHAPE),
        "CI": (np.float32, (N_POINTS,)),
    },
    "large_refined": {
        "RefinedEulerAngles": (np.float32, (N_POINTS, 3)),
        "RefinedDotProducts": (np.float32, (N_POINTS,)),
        "EulerAngles": (np.float32, (N_POINTS, 3)),
    },
    "large_center": _SCENARIO_ARRAYS,
    "large_center_dilate": _SCENARIO_ARRAYS,
    "large_wat": _SCENARIO_ARRAYS,
    "ball_n6": {
        "qu": (np.float64, (2197, 4)),
        "ro": (np.float64, (2197, 4)),
    },
}
PROVENANCE_KEYS = {
    "program_md5": (np.str_, ()),
    "emsoft_version": (np.str_, ()),
    "emsoft_commit": (np.str_, ()),
    "master_md5": (np.str_, ()),
    "master_run_md5": (np.str_, ()),
    "patterns_md5": (np.str_, ()),
    "pc": (np.float64, (4,)),
    "namelist": (np.str_, ()),
    "gpu_name": (np.str_, ()),
    "numdictsingle": (np.int64, ()),
    "numexptsingle": (np.int64, ()),
    "kikuchipy_version": (np.str_, ()),
}

# Namelist keys holding a path, which must lie below the run directory
NAMELIST_PATH_KEYS = (
    "exptfile",
    "masterfile",
    "datafile",
    "ctffile",
    "angfile",
    "dotproductfile",
    "newdotproductfile",
    "usemasterpatternfile",
    "tiffname",
    "dpfile",
    "OSMfile",
    "OSMtiff",
    "IPFmap",
    "quoutname",
    "euoutname",
    "rooutname",
)

# Keys of the dictionary returned by the private EMsoft file reader
# besides the dataset and namelist names: the map shape (rows,
# columns), the EMsoft version and the namelist file text
READER_SHAPE_KEY = "shape"
READER_VERSION_KEY = "version"
READER_TEXT_KEY = "namelist_text"

# Johan's historical EMsoft runs below KIKUCHIPY_EMSOFT_DATA, with md5
NI6_HROSM_FILE = (
    "DItutorial/Ni/dp-Ni6-refined_HROSM.h5",
    "0d8a77ac950f8d0db9c9a76ab13a84b4",
)
NI6_DI_FILE = ("DItutorial/Ni/dp-Ni6-refined.h5", "dbfa6da1e5e9c08be6864bb1b9b8820b")
GRX810_DI_FILE = (
    "OSM/GRX810_HROSM/dp-GRX-refined.h5",
    "48bd1600f4a417fe4978b8ca4fd5f5f5",
)
AL_DI_FILE = ("Al_HROSM/dp-full.h5", "cdb94c405f9dfa8079c9828769f24075")
# The Ni6 HROSM run: KAM threshold, grain count and the two grains
# EMHROSM skipped for having fewer than ten points
NI6_GANGLE = 10.0
NI6_N_GRAINS = 62
NI6_SMALL_GRAINS = (35, 62)

# A pair of neighbours is degenerate if its symmetry reduced angle is
# below this many radians: EMsoft's value then depends on the last bit
# of 576 symmetry products
DEGENERATE_ANGLE_RAD = 1e-6

# Largest float32 ulp distance of a differing non-degenerate KAM point
# if a count pin is not 0 (the recorded fallback)
KAM_FALLBACK_MAX_ULP = 2

# ------------------- Measured-then-pinned --------------------------- #
# The local KAM and OSM pins (Ni6, GRX810, Al) and the pins on the
# shipped references were measured on 2026-10-06 (the shipped ones on
# the references written that day by the c127868 build).

# Number of non-degenerate KAM points differing from EMsoft's file
SHIPPED_KAM_NONDEGENERATE_DIFF = 0
# measured: 4 of 4,110 non-degenerate points, each within the fallback
# ulp bound
SHIPPED_REFINED_KAM_NONDEGENERATE_DIFF = 4
NI6_KAM_NONDEGENERATE_DIFF = 6
NI6_KAM_MAX_ULP = 2
NI6_DI_KAM_NONDEGENERATE_DIFF = 0
GRX810_KAM_NONDEGENERATE_DIFF = 0
AL_KAM_NONDEGENERATE_DIFF = 16
AL_KAM_MAX_ULP = 2
# Number of OSM points differing from each shipped OSM map (0 if the
# generating build keeps the source order of the edge multiplier; the
# c127868 build folds it, so these are straight-edge points, 1 ulp)
SHIPPED_OSM_DIFF = {"OSM": 79, "OSM_05": 87}
# Number of OSM points of the Ni6 file differing by the folded edge
# multiplier
NI6_OSM_EDGE_PIXELS = 214
# Number of grain labels differing when segmenting our KAM of the
# shipped refined Euler angles
SHIPPED_GRAIN_ID_FROM_EULER_DIFF = 0
# Largest float64 ulp distance of a "center" grain average (measured:
# 2, for EMsoft's eq_ of the centre point and for ours alike)
CENTER_AVOR_MAX_ULP = 2
# Watson averages: largest symmetry reduced angle in degrees to
# EMsoft's average. Measured 2026-10-06: 0.136 degrees (seed 0;
# 0.086-0.136 over seeds 0-7, grain 5 of six points). The
# concentrations are recorded, never banded: the EMsoft compatible
# expectation maximisation lands in seed dependent optima of the
# concentration (grain 35: EMsoft 1,072, ours 2.77e5 at seed 0;
# relative differences 257-671 over seeds 0-7) while the averages
# agree, and EMHROSM seeds its generator from the clock. Only the
# outcome of the concentration gate is compared per grain.
WAT_AVOR_MAX_DEG = 0.3
# Ni6: EMHROSM seeds its generator from the clock, so its Watson
# averages of diffuse grains depend on the run. Grains whose EMsoft
# concentration is at least NI6_WAT_TIGHT_MIN_KAPPA are compared within
# the two bands below (measured 2026-10-06 on those grains: at most
# 0.0086 degrees and 2.2e-4); the others are only required to be kept
# by both, and their count is pinned (measured 2026-10-06: 22 grains
# below the concentration, four of them diffuse with averages 12-54
# degrees apart, plus grain 1, see NI6_WAT_SEED_DEPENDENT_GRAINS)
NI6_WAT_TIGHT_MIN_KAPPA = 50.0
NI6_WAT_AVOR_MAX_DEG = 0.02
NI6_WAT_KAPPA_REL = 5e-4
NI6_WAT_LOOSE_GRAIN_COUNT = 23
# Grain 1 has a concentration of 26,617-37,155 depending on the seed
# (EMsoft's file: 34,896; ours: 37,155, 0.065 apart), outside the
# concentration band, so it is treated like the diffuse grains
NI6_WAT_SEED_DEPENDENT_GRAINS = (1,)
# Total bytes of the six shipped references (measured: 918,408)
REFERENCE_TOTAL_BYTES = 920_000
# Seconds of one regeneration of the references with the local
# EMsoft programs; recorded, never asserted (measured 2026-10-06: one
# complete run of the script, 2,148.8 s)
REGENERATION_RUNTIME_S = 2148.8
# The arrays a regeneration reproduces bitwise, per file. Measured
# 2026-10-06 over four EMDI runs of the c127868 build: the top matches
# and their dot products are identical, but the dictionary Euler
# angles EMDI stores differ on 8,608-12,105 of 333,248 rows between
# any two runs, so the Euler angles of 125 of 4,125 points differ, and
# with them everything computed from them (KAM, the segmentation, 44
# or 45 grains, the grain averages and the re-indexed maps). Those
# arrays are checked for their keys, dtypes and layout only.
REGENERATION_BITWISE_KEYS = {
    "large_di": ("CI", "OSM", "OSM_05", "TopMatchIndices"),
    "ball_n6": ("qu", "ro"),
}
# Data sets of the dot product file of a run compared bitwise with
# those of the run the shipped files came from (the shipped files do
# not hold the dot products)
REGENERATION_BITWISE_DOT_PRODUCT_KEYS = ("TopMatchIndices", "TopDotProductList")
# Misorientation ball files of a run compared bitwise with those of
# the run the shipped files came from
REGENERATION_BALL_FILES = tuple(
    f"ball_n{n}_{kind}.txt" for n in (6, 20) for kind in ("qu", "ro")
)
# Largest difference of the number of grains of a regenerated scenario
# to the shipped one (measured 2026-10-06: 44 vs 45)
REGENERATION_GRAIN_COUNT_TOLERANCE = 2
# Provenance which differs between runs by design: the run directory
# in the namelist paths is normalised, the kikuchipy version recorded
RUN_DIRECTORY_PATTERN = re.compile(r"kikuchipy_hrosm/\d{8}-\d{6}")
RECORDED_ONLY_KEYS = ("kikuchipy_version",)

# ----------------------------- Helpers ------------------------------ #


def reference_directory() -> Path:
    """Return the directory of the shipped references, anchored at the
    installed package so the wheel job finds them.
    """
    return Path(kp.data.__file__).parent / "emsoft_hrosm"


def registry_keys() -> set[str]:
    """Return the registry keys of the shipped references."""
    return {
        key
        for key in _registry_hashes
        if key.startswith("emsoft_hrosm/regression_hrosm_") and key.endswith(".npz")
    }


def reference_name(key_or_path) -> str:
    """Return the scenario name of a registry key or file name."""
    return Path(str(key_or_path)).name[len("regression_hrosm_") : -len(".npz")]


@functools.lru_cache(maxsize=None)
def load_reference(name: str) -> dict[str, np.ndarray]:
    """Return one shipped reference as a dictionary of read only arrays.

    Fetched through the data module, so its registry md5 is verified.
    The file is read out and closed: an open handle keeps the file
    locked on Windows for the whole session.
    """
    fpath = Dataset(f"emsoft_hrosm/regression_hrosm_{name}.npz").fetch_file_path()
    with np.load(fpath, allow_pickle=False) as reference:
        arrays = {key: reference[key] for key in reference.files}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def reference_path(name: str) -> Path:
    """Return the path of one shipped reference."""
    return Path(Dataset(f"emsoft_hrosm/regression_hrosm_{name}.npz").fetch_file_path())


def md5_of_file(fpath) -> str:
    """Return the md5 sum of a file."""
    return hashlib.md5(Path(fpath).read_bytes()).hexdigest()


def crystal_map_from_euler(
    euler: np.ndarray, shape: tuple[int, int], name: str = "ni"
) -> CrystalMap:
    """Return a dense one-phase m-3m crystal map from EMsoft's float32
    Euler angles in radians, raster order with x fastest.
    """
    coords, n = create_coordinate_arrays(shape, step_sizes=(1, 1))
    euler = np.asarray(euler, dtype=np.float32).reshape(n, 3)
    return CrystalMap(
        rotations=Rotation.from_euler(euler.astype(np.float64)),
        phase_id=np.zeros(n, dtype=np.int32),
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phases=[Phase(name, point_group="m-3m")], ids=[0]),
        scan_unit="um",
    )


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the elementwise Hamilton product of quaternions."""
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


def disorientation_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the m-3m symmetry reduced angles in radians between the
    quaternions ``a`` and ``b`` elementwise.
    """
    r = quaternion_product(a, b * np.array([1.0, -1.0, -1.0, -1.0]))
    d = np.abs(r @ Oh.proper_subgroup.data.T).max(axis=-1)
    return 2 * np.arccos(np.clip(d, 0, 1))


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the angles in degrees between rotations elementwise,
    without crystal symmetry.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2 * np.arccos(np.clip(d, 0, 1)))


def degenerate_pixels(euler: np.ndarray, H: int, W: int) -> np.ndarray:
    """Return a mask of shape (H, W) of the points to which EMsoft's
    KAM bookkeeping credits a degenerate pair.

    EMsoft credits a point its left and right horizontal pair, its
    upper vertical pair, the vertical pair one column to the left
    (shifted down in the flat order), and the first row's last point
    compared with the identity is credited to that point and to the
    first point. A pair is degenerate if its symmetry reduced angle is
    below ``DEGENERATE_ANGLE_RAD``.
    """
    q = Rotation.from_euler(np.asarray(euler, np.float64)).data.reshape(H, W, 4)
    pair_h = disorientation_rad(q[:, :-1], q[:, 1:])
    pair_v = disorientation_rad(q[:-1], q[1:])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    spurious = disorientation_rad(identity, q[0, W - 1])

    left = np.full((H, W), np.inf)
    left[:, 1:] = pair_h
    right = np.full((H, W), np.inf)
    right[:, :-1] = pair_h
    up = np.full((H, W), np.inf)
    up[1:] = pair_v
    up[0, W - 1] = spurious
    v = np.full(H * W, np.inf)
    v[: (H - 1) * W] = pair_v.ravel()
    down = np.full(H * W, np.inf)
    down[1:] = v[:-1]
    down[0] = spurious

    smallest = np.minimum(np.minimum(left, right), up).ravel()
    smallest = np.minimum(smallest, down)
    return (smallest < DEGENERATE_ANGLE_RAD).reshape(H, W)


def float32_ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the float32 ulp distance between ``a`` and ``b``."""

    def ordered(x):
        i = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)

    return np.abs(ordered(a) - ordered(b))


def float64_ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the float64 ulp distance between ``a`` and ``b``."""

    def ordered(x):
        i = np.asarray(x, dtype=np.float64).view(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFFFFFFFFFF), i).astype(object)

    return np.abs(ordered(a) - ordered(b)).astype(np.float64)


def emsoft_eq(euler: np.ndarray) -> np.ndarray:
    """Return EMsoft's Euler angle to quaternion conversion ``eq_`` of
    Euler angles in radians, float64, scalar first, with a
    non-negative scalar part.
    """
    euler = np.asarray(euler, dtype=np.float64)
    ee = 0.5 * euler
    c_phi = np.cos(ee[..., 1])
    s_phi = np.sin(ee[..., 1])
    cm = np.cos(ee[..., 0] - ee[..., 2])
    sm = np.sin(ee[..., 0] - ee[..., 2])
    cp = np.cos(ee[..., 0] + ee[..., 2])
    sp = np.sin(ee[..., 0] + ee[..., 2])
    q = np.stack([c_phi * cp, -s_phi * cm, -s_phi * sm, -c_phi * sp], axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def compat_kam_comparison(
    euler: np.ndarray, expected: np.ndarray, record_property, tag: str
) -> tuple[int, int]:
    """Return the number of non-degenerate points where our EMsoft
    compatible KAM differs from EMsoft's map, and their largest float32
    ulp distance; the degenerate counts are recorded.
    """
    H, W = expected.shape
    xmap = crystal_map_from_euler(euler, (H, W))
    start = time.perf_counter()
    kam = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
    record_property(f"{tag}_seconds", f"{time.perf_counter() - start:.3f}")
    assert kam.dtype == np.float32
    assert kam.shape == (H, W)

    degenerate = degenerate_pixels(euler, H, W)
    differ = kam != expected
    ulps = float32_ulps(kam, expected)
    nondegenerate = differ & ~degenerate
    record_property(f"{tag}_nondegenerate_points", int((~degenerate).sum()))
    record_property(f"{tag}_nondegenerate_differ", int(nondegenerate.sum()))
    record_property(f"{tag}_degenerate_points", int(degenerate.sum()))
    record_property(f"{tag}_degenerate_differ", int((differ & degenerate).sum()))
    if np.any(differ & degenerate):
        largest = np.abs(kam - expected)[differ & degenerate].max()
        record_property(f"{tag}_degenerate_max_diff_deg", float(largest))
    max_ulp = int(ulps[nondegenerate].max()) if nondegenerate.any() else 0
    return int(nondegenerate.sum()), max_ulp


def third_pixel_mask(H: int, W: int) -> np.ndarray:
    """Return a mask of shape (H, W) of the points EMsoft's edge
    multiplier sequence passes through ``x -> (x * 4) / 3``: the top
    and bottom row interior, the left and right column interior and
    the first point of the last row.
    """
    n = H * W
    mask = np.zeros(n, dtype=bool)
    mask[1 : W - 1] = True
    mask[n - W + 1 : n - 1] = True
    for jj in range(1, H - 1):
        mask[W * jj] = True
    for jj in range(2, H):
        mask[W * jj - 1] = True
    mask[n - W] = True
    return mask.reshape(H, W)


def emsoft_osm_table(
    top: np.ndarray, n: int, H: int, W: int, folded: bool = False
) -> np.ndarray:
    """Return EMsoft's orientation similarity map of 1-based top-match
    lists (H * W, k) for maps with ``W >= 2``.

    Shared index counts of the 4-neighbour pairs are credited with
    EMsoft's bookkeeping, summed in float32, and the edge multipliers
    applied in float32 with ``(x * 4) / 3``, or ``x * float32(4 / 3)``
    if ``folded`` (a compiler folding the constant).
    """
    lists = np.asarray(top)[:, :n].reshape(H, W, n)

    def shared(a, b):
        return (a[..., :, None] == b[..., None, :]).sum(axis=(-1, -2))

    pair_h = shared(lists[:, :-1], lists[:, 1:]).astype(np.float32)
    pair_v = shared(lists[:-1], lists[1:]).astype(np.float32)
    left = np.zeros((H, W), np.float32)
    left[:, 1:] = pair_h
    up = np.zeros((H, W), np.float32)
    up[1:, :] = pair_v
    right = np.zeros((H, W), np.float32)
    right[:, :-1] = pair_h
    v = np.zeros(H * W, np.float32)
    v[: (H - 1) * W] = pair_v.ravel()
    down = np.zeros(H * W, np.float32)
    down[1:] = v[:-1]
    acc = ((left + up) + right).ravel() + down

    if folded:

        def third(x):
            return x * np.float32(4 / 3)

    else:

        def third(x):
            return (x * np.float32(4)) / np.float32(3)

    total = H * W
    acc = acc * np.float32(0.25)
    acc[1 : W - 1] = third(acc[1 : W - 1])
    acc[total - W + 1 : total - 1] = third(acc[total - W + 1 : total - 1])
    for jj in range(1, H - 1):
        acc[W * jj] = third(acc[W * jj])
    for jj in range(2, H):
        acc[W * jj - 1] = third(acc[W * jj - 1])
    acc[0] = acc[0] * np.float32(4)
    acc[W - 1] = acc[W - 1] * np.float32(2)
    acc[total - 1] = acc[total - 1] * np.float32(2)
    acc[total - W] = third(acc[total - W])
    return acc.reshape(H, W)


def emsoft_grain_roi_to_boxes(grain_roi: np.ndarray) -> np.ndarray:
    """Return EMsoft's 1-based ``(x0, y0, w, h)`` grain boxes as
    0-based ``(row0, col0, height, width)``.
    """
    roi = np.asarray(grain_roi, dtype=np.int64)
    return np.stack([roi[:, 1] - 1, roi[:, 0] - 1, roi[:, 3], roi[:, 2]], axis=1)


def emsoft_center_pixels(grain_roi: np.ndarray) -> np.ndarray:
    """Return the 0-based (row, column) of EMsoft's centre point of
    each 1-based ``(x0, y0, w, h)`` box: ``(x0 + w // 2, y0 + h //
    2)``, 1-based.
    """
    roi = np.asarray(grain_roi, dtype=np.int64)
    x = roi[:, 0] + roi[:, 2] // 2
    y = roi[:, 1] + roi[:, 3] // 2
    return np.stack([y - 1, x - 1], axis=1)


def symmetry_reduced_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the m-3m disorientation angles in degrees elementwise,
    through orix.
    """
    oa = Orientation(a, symmetry=Oh)
    ob = Orientation(b, symmetry=Oh)
    return np.rad2deg(oa.angle_with(ob))


def kappa_relative_difference(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """Return ``|ours / theirs - 1|``, 0 where both are infinite."""
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    both_inf = np.isinf(ours) & np.isinf(theirs)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(ours / theirs - 1)
    return np.where(both_inf, 0.0, rel)


def watson_table(xmap: CrystalMap, grain_id: np.ndarray):
    """Return our EMsoft compatible Watson grain table with EMHROSM's
    25 initial guesses and 40 iterations, seed 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return average_grain_orientations(
            xmap,
            grain_id,
            method="watson",
            n_em=N_EM,
            n_iter=N_ITER,
            seed=0,
            emsoft_compatible=True,
        )


@functools.lru_cache(maxsize=None)
def read_local_dot_product_file(fpath: Path) -> dict:
    """Return the reader's view of a local dot product file, once per
    session.
    """
    return _read_only(read_emsoft_dot_product_file(fpath))


@functools.lru_cache(maxsize=None)
def read_local_hrosm_file(fpath: Path) -> dict:
    """Return the reader's view of a local EMHROSM file, once per
    session.
    """
    return _read_only(read_emsoft_hrosm_file(fpath))


def _read_only(values: dict) -> dict:
    """Return a cached reader dictionary with its arrays made read
    only, so no test can change them for the others.
    """
    for value in values.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        elif isinstance(value, dict):
            _read_only(value)
    return values


def regeneration_differences(
    name: str, regenerated, shipped, record_property=None
) -> list[str]:
    """Return one line per array of a regenerated reference which
    breaks the regeneration contract against the shipped one.

    The arrays of :data:`REGENERATION_BITWISE_KEYS` and the provenance
    (run directory normalised, :data:`RECORDED_ONLY_KEYS` skipped) are
    compared bitwise. Every other array is checked for its key, dtype
    and layout, the number of grains within
    :data:`REGENERATION_GRAIN_COUNT_TOLERANCE`, and the number of
    differing values recorded.
    """
    lines = []
    with (
        np.load(regenerated, allow_pickle=False) as theirs_file,
        np.load(shipped, allow_pickle=False) as ours_file,
    ):
        theirs = {key: theirs_file[key] for key in theirs_file.files}
        ours = {key: ours_file[key] for key in ours_file.files}
    table = {**REFERENCE_TABLE.get(name, {}), **PROVENANCE_KEYS}
    n_grains = int(theirs["nGrains"]) if "nGrains" in theirs else None
    bitwise = REGENERATION_BITWISE_KEYS.get(name, ())
    for key in sorted(set(theirs) | set(ours)):
        if key not in theirs or key not in ours:
            lines.append(f"  {key}: only in one of the two files")
            continue
        one, two = theirs[key], ours[key]
        if one.dtype != two.dtype:
            lines.append(f"  {key}: {one.dtype}{one.shape} vs {two.dtype}{two.shape}")
            continue
        if key in table:
            shape = tuple(n_grains if s == "n" else s for s in table[key][1])
        else:
            shape = two.shape
        if one.shape != shape:
            lines.append(f"  {key}: {one.dtype}{one.shape}, layout {shape}")
            continue
        if key in RECORDED_ONLY_KEYS:
            continue
        if key == "nGrains":
            if abs(int(one) - int(two)) > REGENERATION_GRAIN_COUNT_TOLERANCE:
                lines.append(
                    f"  nGrains: {int(one)} vs {int(two)} grains, more than "
                    f"{REGENERATION_GRAIN_COUNT_TOLERANCE} apart"
                )
            continue
        provenance = key in PROVENANCE_KEYS or one.dtype.kind == "U"
        if one.dtype.kind == "U":
            one = np.asarray(RUN_DIRECTORY_PATTERN.sub("<run>", str(one)))
            two = np.asarray(RUN_DIRECTORY_PATTERN.sub("<run>", str(two)))
        if one.shape == two.shape and np.array_equal(one, two):
            continue
        if provenance:
            lines.append(f"  {key}: differs (provenance)")
        elif key in bitwise:
            n_differ = int(np.count_nonzero(one != two))
            lines.append(f"  {key}: differs in {n_differ} values (bitwise)")
        elif record_property is not None:
            if one.shape == two.shape:
                n_differ = int(np.count_nonzero(one != two))
            else:
                n_differ = f"{one.shape} vs {two.shape}"
            record_property(f"{name}_{key}_differ", n_differ)
    return lines


def regeneration_message(lines: list[str]) -> str:
    """Return the failure message of a regeneration mismatch.

    Suspect number one is a rebuilt EMsoft program or library, which
    ``program_md5`` names outright.
    """
    header = [
        "the regenerated references do not reproduce the shipped ones.",
        "Suspect #1 is a rebuilt EMsoft program or library: program_md5 "
        "pins the md5 of every program and library the shipped bytes came "
        "from, so a differing program_md5 below means another build.",
        "Per-array differences (the arrays EMsoft reproduces run to run "
        "and the provenance: the contract is bitwise; the rest: keys, "
        "dtypes, layout and the number of grains):",
    ]
    return "\n".join(header + lines)


def assert_regenerated(written: dict, record_property=None) -> None:
    """Assert every regenerated file keeps the regeneration contract
    with its shipped twin.
    """
    lines = []
    for name, fpath in sorted(written.items()):
        differences = regeneration_differences(
            name, fpath, reference_path(name), record_property
        )
        if differences:
            lines.append(f"{name}:")
            lines.extend(differences)
    if lines:
        raise AssertionError(regeneration_message(lines))


def shipped_run_directory(data_root: Path) -> Path:
    """Return the EMsoft run directory the shipped files came from,
    as named in their provenance namelist.
    """
    namelist = str(load_reference("large_di")["namelist"])
    match = RUN_DIRECTORY_PATTERN.search(namelist)
    assert match is not None, "the shipped namelist names no run directory"
    return data_root / match.group(0)


def read_raw_top_matches(fpath: Path) -> dict[str, np.ndarray]:
    """Return the padded top match data sets of a dot product file as
    written.
    """
    with h5py.File(fpath, "r") as f:
        group = f["Scan 1/EBSD/Data"]
        return {key: group[key][()] for key in REGENERATION_BITWISE_DOT_PRODUCT_KEYS}


def acid_band_medians(log_path: Path) -> tuple[float, float]:
    """Return the last top-1 and refined median disorientations in
    degrees a reference run logged.
    """
    text = Path(log_path).read_text(encoding="utf-8")
    top1 = re.findall(
        r"acid band: top-1 median disorientation(?: with flipy)? ([0-9.]+) deg",
        text,
    )
    refined = re.findall(
        r"acid band: refined median disorientation ([0-9.]+) deg", text
    )
    assert top1 and refined, f"no acid band lines in {log_path}"
    return float(top1[-1]), float(refined[-1])


def namelist_value_matches(text: str, key: str, value: str) -> bool:
    """Return whether a namelist text sets ``key`` to ``value``."""
    pattern = rf"(^|[\s,]){re.escape(key)}\s*=\s*{re.escape(value)}(\s|,|$)"
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def hrosm_sources() -> list[Path]:
    """Return the source files of the ``_hrosm`` package."""
    return sorted(Path(hrosm_package.__file__).parent.glob("*.py"))


def joined_comment_blocks(source: str) -> str:
    """Return the module's runs of consecutive ``#`` comment lines,
    each joined to one line (the leading ``#`` and one space stripped,
    runs of whitespace collapsed), the runs separated by newlines.
    """
    blocks, block = [], []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            text = stripped[1:]
            if text.startswith(" "):
                text = text[1:]
            block.append(text)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    return "\n".join(re.sub(r"\s+", " ", " ".join(b)).strip() for b in blocks)


# The EMsoftOO derived modules and the year span of their notice
EMSOFT_DERIVED_MODULES = {
    "_emsoft_quaternions": "2013-2026",
    "_kam": "2013-2026",
    "_segmentation": "2013-2026",
    "_averaging": "2013-2026",
    "_directional_statistics": "2014-2026",
    "_sampling": "2013-2026",
    "_osm": "2013-2026",
}

GPL_HEADER = (
    "#\n"
    "# Copyright 2019-2026 the kikuchipy developers\n"
    "#\n"
    "# This file is part of kikuchipy.\n"
    "#\n"
    "# kikuchipy is free software: you can redistribute it and/or modify\n"
    "# it under the terms of the GNU General Public License as published by\n"
    "# the Free Software Foundation, either version 3 of the License, or\n"
    "# (at your option) any later version.\n"
)

# The public names of the orientation engine
PUBLIC_NAMES = (
    "GrainTable",
    "average_grain_orientations",
    "grain_bounding_boxes",
    "grain_reference_orientation_deviation_map",
    "kernel_average_misorientation_map",
    "misorientation_ball",
    "misorientation_ball_spacing",
    "segment_grains_kam",
)

# A Sphinx role and its target
SPHINX_ROLE = re.compile(r":(?:func|class|meth|attr|mod):`~?([\w.]+)`")

# Specification paths, file names and decision identifiers, which no
# docstring may name
CLEAN_REPLAY = re.compile(
    r"spec"
    r"s[/]|requirements\.md|plan\.md|validation\.md|tech-stack\.md"
    r"|mission\.md|roadmap\.md|[^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)"
)

# Routines of LGPL code which must not be ported
LGPL_TOKENS = (
    "r8_normal_01",
    "r8_uniform_01",
    "r8vec_normal_01",
    "BesselI0",
    "BesselI1",
    "BesselIn",
    "SSORT",
    "DSYEV",
)


# ================= Module discipline (static checks) ================ #


class TestModuleDiscipline:
    def test_hrosm_package_imports_no_numba(self):
        for fpath in hrosm_sources():
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("numba"), fpath.name
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("numba"), fpath.name

    @pytest.mark.parametrize("module", sorted(EMSOFT_DERIVED_MODULES))
    def test_emsoft_derived_modules_carry_the_bsd_notice(self, module):
        fpath = Path(hrosm_package.__file__).parent / f"{module}.py"
        source = fpath.read_text(encoding="utf-8")
        assert source.startswith(GPL_HEADER)
        joined = joined_comment_blocks(source)
        assert (
            "The following copyright notice is included because the following "
            "functionality in this file is derived and adapted from EMsoftOO:"
        ) in joined
        assert "Marc De Graef Research Group/Carnegie Mellon University" in joined
        assert "All rights reserved." in joined
        assert "Changes by the kikuchipy developers" in joined
        years = EMSOFT_DERIVED_MODULES[module]
        assert f"Copyright (c) {years}, Marc De Graef" in joined

    @pytest.mark.parametrize("fpath", hrosm_sources(), ids=lambda p: p.stem)
    def test_every_module_starts_with_the_gpl_header(self, fpath):
        source = fpath.read_text(encoding="utf-8")
        assert source.startswith(GPL_HEADER)
        joined = joined_comment_blocks(source)
        assert "GNU General Public License" in joined

    def test_public_docstrings_link_no_private_name_and_no_spec_id(self):
        from kikuchipy.indexing._hrosm._grains import GrainTable

        docstrings = {}
        for name in PUBLIC_NAMES:
            docstrings[name] = getattr(kp.indexing, name).__doc__
        docstrings["GrainTable.n_grains"] = GrainTable.__dict__["n_grains"].__doc__
        docstrings["GrainTable.from_crystal_map"] = GrainTable.from_crystal_map.__doc__
        for name, doc in docstrings.items():
            assert doc, f"{name} has no docstring"
            for target in SPHINX_ROLE.findall(doc):
                private = [p for p in target.split(".") if p.startswith("_")]
                assert not private, f"{name} links the private {target}"
            match = CLEAN_REPLAY.search(doc)
            assert match is None, f"{name} names {match.group(0)!r}"

    def test_no_lgpl_routine_is_ported(self):
        for fpath in hrosm_sources():
            with open(fpath, "rb") as f:
                tokens = list(tokenize.tokenize(f.readline))
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    continue
                for name in LGPL_TOKENS:
                    assert name not in token.string, (fpath.name, name)

            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != "random", fpath.name
                elif isinstance(node, ast.ImportFrom):
                    assert node.module != "random", fpath.name
                    if node.module == "numpy.random":
                        for alias in node.names:
                            assert alias.name in ("default_rng", "Generator"), (
                                fpath.name,
                                alias.name,
                            )
                elif isinstance(node, ast.Attribute):
                    base = node.value
                    if (
                        isinstance(base, ast.Attribute)
                        and base.attr == "random"
                        and isinstance(base.value, ast.Name)
                        and base.value.id in ("np", "numpy")
                    ):
                        assert node.attr in ("default_rng", "Generator"), (
                            fpath.name,
                            node.attr,
                        )
                elif isinstance(node, ast.Name):
                    assert node.id != "RandomState", fpath.name

    def test_print_only_in_the_driver(self):
        for fpath in hrosm_sources():
            if fpath.name == "_driver.py":
                continue
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", fpath.name

    def test_modules_use_postponed_annotations_and_no_optional_union(self):
        for fpath in hrosm_sources():
            source = fpath.read_text(encoding="utf-8")
            assert "Optional[" not in source, fpath.name
            assert "Union[" not in source, fpath.name
            tree = ast.parse(source)
            body = tree.body
            only_docstring = len(body) == 1 and isinstance(body[0], ast.Expr)
            if only_docstring:
                # a package docstring without code has no annotation
                continue
            future = [
                alias.name
                for node in body
                if isinstance(node, ast.ImportFrom) and node.module == "__future__"
                for alias in node.names
            ]
            assert "annotations" in future, fpath.name

    def test_emsoft_compatible_is_never_a_module_global(self):
        for fpath in hrosm_sources():
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                else:
                    continue
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            assert "emsoft_compatible" not in name.id.lower(), (
                                fpath.name
                            )


# =========== EMsoft compatible KAM on EMsoft's own files ============ #


class TestCompatKAMOnEMsoftFiles:
    def test_shipped_di_kam_on_nondegenerate_pixels(self, record_property):
        # EMDI wrote KAM from the top-1 Euler angles it stores
        euler = load_reference("large_refined")["EulerAngles"]
        expected = load_reference("large_di")["KAM"]
        n_differ, max_ulp = compat_kam_comparison(
            euler, expected, record_property, "shipped_di_kam"
        )
        assert n_differ == SHIPPED_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= KAM_FALLBACK_MAX_ULP

    def test_shipped_refined_kam_on_nondegenerate_pixels(self, record_property):
        # EMHROSM's cluster stage wrote kam from the refined angles
        euler = load_reference("large_refined")["RefinedEulerAngles"]
        expected = load_reference("large_center")["kam"]
        n_differ, max_ulp = compat_kam_comparison(
            euler, expected, record_property, "shipped_refined_kam"
        )
        assert n_differ == SHIPPED_REFINED_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= KAM_FALLBACK_MAX_ULP

    def test_ni6_hrosm_kam_on_nondegenerate_pixels(
        self, emsoft_data_file, record_property
    ):
        di = read_local_dot_product_file(emsoft_data_file(*NI6_DI_FILE))
        hrosm = read_local_hrosm_file(emsoft_data_file(*NI6_HROSM_FILE))
        assert tuple(di[READER_SHAPE_KEY]) == (151, 186)
        n_differ, max_ulp = compat_kam_comparison(
            di["RefinedEulerAngles"], hrosm["kam"], record_property, "ni6_hrosm_kam"
        )
        assert n_differ == NI6_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= NI6_KAM_MAX_ULP

    def test_ni6_di_kam_on_nondegenerate_pixels(
        self, emsoft_data_file, record_property
    ):
        di = read_local_dot_product_file(emsoft_data_file(*NI6_DI_FILE))
        n_differ, max_ulp = compat_kam_comparison(
            di["EulerAngles"], di["KAM"], record_property, "ni6_di_kam"
        )
        assert n_differ == NI6_DI_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= KAM_FALLBACK_MAX_ULP

    def test_grx810_di_kam_on_nondegenerate_pixels(
        self, emsoft_data_file, record_property
    ):
        di = read_local_dot_product_file(emsoft_data_file(*GRX810_DI_FILE))
        assert tuple(di[READER_SHAPE_KEY]) == (337, 413)
        n_differ, max_ulp = compat_kam_comparison(
            di["EulerAngles"], di["KAM"], record_property, "grx810_di_kam"
        )
        assert n_differ == GRX810_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= KAM_FALLBACK_MAX_ULP

    @pytest.mark.weekly
    def test_al_di_kam_on_nondegenerate_pixels(self, emsoft_data_file, record_property):
        di = read_local_dot_product_file(emsoft_data_file(*AL_DI_FILE))
        assert tuple(di[READER_SHAPE_KEY]) == (689, 728)
        n_differ, max_ulp = compat_kam_comparison(
            di["EulerAngles"], di["KAM"], record_property, "al_di_kam"
        )
        assert n_differ == AL_KAM_NONDEGENERATE_DIFF
        assert max_ulp <= AL_KAM_MAX_ULP


# =========== EMsoft compatible OSM on EMsoft's own files ============ #


class TestCompatOSMOnEMsoftFiles:
    @pytest.mark.parametrize(
        "key, n", [("OSM", 10), ("OSM_05", 5)], ids=["osm-10", "osm_05-5"]
    )
    def test_shipped_osm_matches(self, key, n, record_property):
        di = load_reference("large_di")
        H, W = MAP_SHAPE
        top = np.asarray(di["TopMatchIndices"])
        osm = _osm_emsoft(top, H, W, n)
        expected = di[key]
        assert osm.dtype == np.float32
        assert osm.shape == MAP_SHAPE
        differ = osm != expected
        record_property(f"shipped_{key}_differ", int(differ.sum()))
        assert int(differ.sum()) == SHIPPED_OSM_DIFF[key]
        if differ.any():
            # a build folding the edge multiplier to x * (4 / 3)
            assert np.all(third_pixel_mask(H, W)[differ])
            assert np.all(float32_ulps(osm, expected)[differ] == 1)
            folded = emsoft_osm_table(top, n, H, W, folded=True)
            assert np.array_equal(folded[differ], expected[differ])

    @pytest.mark.parametrize("key", ["OSM", "OSM_20"])
    def test_grx810_osm_matches(self, key, emsoft_data_file):
        di = read_local_dot_product_file(emsoft_data_file(*GRX810_DI_FILE))
        n = int(di["nosm"]) if key == "OSM" else 20
        H, W = di[READER_SHAPE_KEY]
        osm = _osm_emsoft(np.asarray(di["TopMatchIndices"]), H, W, n)
        assert np.array_equal(osm, di[key])

    @pytest.mark.weekly
    def test_al_osm_matches(self, emsoft_data_file):
        di = read_local_dot_product_file(emsoft_data_file(*AL_DI_FILE))
        H, W = di[READER_SHAPE_KEY]
        osm = _osm_emsoft(np.asarray(di["TopMatchIndices"]), H, W, int(di["nosm"]))
        assert np.array_equal(osm, di["OSM"])

    def test_ni6_osm_differs_only_on_third_pixels_by_one_ulp(
        self, emsoft_data_file, record_property
    ):
        # this Windows build folds x * 4.0 / 3.0 to x * (4.0 / 3.0)
        di = read_local_dot_product_file(emsoft_data_file(*NI6_DI_FILE))
        n = int(di["nosm"])
        assert n == 20
        H, W = di[READER_SHAPE_KEY]
        top = np.asarray(di["TopMatchIndices"])
        osm = _osm_emsoft(top, H, W, n)
        expected = di["OSM"]
        differ = osm != expected
        record_property("ni6_osm_differ", int(differ.sum()))
        assert np.all(third_pixel_mask(H, W)[differ])
        assert np.all(float32_ulps(osm, expected)[differ] == 1)
        folded = emsoft_osm_table(top, n, H, W, folded=True)
        assert np.array_equal(folded[differ], expected[differ])
        assert int(differ.sum()) == NI6_OSM_EDGE_PIXELS


# ================ EMHROSM's cluster stage (shipped) ================= #


class TestClusterStage:
    @pytest.mark.parametrize("scenario", SCENARIOS_HROSM)
    def test_grain_ids_from_the_reference_kam_are_bitwise(self, scenario):
        ref = load_reference(f"large_{scenario}")
        labels = segment_grains_kam(
            np.asarray(ref["kam"]),
            threshold=GANGLE,
            dilate=SCENARIO_DILATE[scenario],
            emsoft_compatible=True,
        )
        assert labels.dtype == np.int32
        assert np.array_equal(labels, ref["grainID"])
        assert int(labels.max()) == int(ref["nGrains"])

    @pytest.mark.parametrize("scenario", SCENARIOS_HROSM)
    def test_grain_ids_from_the_euler_angles(self, scenario, record_property):
        ref = load_reference(f"large_{scenario}")
        euler = load_reference("large_refined")["RefinedEulerAngles"]
        xmap = crystal_map_from_euler(euler, MAP_SHAPE)
        kam = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
        labels = segment_grains_kam(
            kam,
            threshold=GANGLE,
            dilate=SCENARIO_DILATE[scenario],
            emsoft_compatible=True,
        )
        differ = labels != ref["grainID"]
        record_property(f"{scenario}_grain_id_differ", int(differ.sum()))
        assert int(differ.sum()) == SHIPPED_GRAIN_ID_FROM_EULER_DIFF
        # a differing label can only come from a differing KAM value
        # in its 8-neighbourhood
        kam_differ = binary_dilation(kam != ref["kam"], structure=np.ones((3, 3)))
        assert np.all(kam_differ[differ])

    @pytest.mark.parametrize("scenario", SCENARIOS_HROSM)
    def test_pixel_counts_and_boxes(self, scenario):
        ref = load_reference(f"large_{scenario}")
        labels = segment_grains_kam(
            np.asarray(ref["kam"]),
            threshold=GANGLE,
            dilate=SCENARIO_DILATE[scenario],
            emsoft_compatible=True,
        )
        n_grains = int(ref["nGrains"])
        counts = np.bincount(labels.ravel(), minlength=n_grains + 1)[1:]
        assert np.array_equal(counts, ref["npixels"])
        boxes = grain_bounding_boxes(labels)
        assert boxes.dtype == np.int64
        assert boxes.shape == (n_grains, 4)
        assert np.array_equal(boxes, emsoft_grain_roi_to_boxes(ref["grainROI"]))

    @pytest.mark.parametrize("scenario", ["center", "center_dilate"])
    def test_center_average_is_the_box_centre_pixel(self, scenario, record_property):
        ref = load_reference(f"large_{scenario}")
        euler = np.asarray(load_reference("large_refined")["RefinedEulerAngles"])
        xmap = crystal_map_from_euler(euler, MAP_SHAPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            table = average_grain_orientations(
                xmap,
                np.asarray(ref["grainID"]),
                method="center",
                emsoft_compatible=True,
            )

        # EMsoft's centre point of the 1-based box (x0 + w // 2, y0 +
        # h // 2), its float32 Euler angles through eq_, not reduced
        centre = emsoft_center_pixels(ref["grainROI"])
        flat = centre[:, 0] * MAP_SHAPE[1] + centre[:, 1]
        oracle = emsoft_eq(euler[flat].astype(np.float64))
        avor = np.asarray(ref["avor"])
        oracle_ulps = float64_ulps(oracle, avor)
        ours_ulps = float64_ulps(table.rotation.data, avor)
        record_property(f"{scenario}_oracle_max_ulp", float(oracle_ulps.max()))
        record_property(f"{scenario}_avor_max_ulp", float(ours_ulps.max()))
        assert oracle_ulps.max() <= CENTER_AVOR_MAX_ULP
        assert ours_ulps.max() <= CENTER_AVOR_MAX_ULP
        assert np.all(np.asarray(ref["kappa"]) == 1.0)
        assert np.all(np.asarray(table.kappa) == 1.0)

    def test_watson_average_within_bands(self, record_property):
        ref = load_reference("large_wat")
        euler = load_reference("large_refined")["RefinedEulerAngles"]
        xmap = crystal_map_from_euler(euler, MAP_SHAPE)
        table = watson_table(xmap, np.asarray(ref["grainID"]))
        kappa = np.asarray(ref["kappa"])
        ours_kappa = np.asarray(table.kappa)

        # both keep the same grains, and the concentration gate has
        # the same outcome per grain (EMsoft writes -1 for a grain it
        # rejects)
        theirs_valid = kappa != -1
        assert np.array_equal(np.asarray(table.valid), theirs_valid)
        with np.errstate(invalid="ignore"):
            ours_gate = ours_kappa > EMSOFT_MIN_KAPPA
        assert np.array_equal(ours_gate, kappa > EMSOFT_MIN_KAPPA)

        # the averages within the angle band; the concentrations are
        # recorded only (see WAT_AVOR_MAX_DEG)
        ours = table.rotation.data[theirs_valid]
        angles = symmetry_reduced_angle_deg(ours, np.asarray(ref["avor"])[theirs_valid])
        rel = kappa_relative_difference(ours_kappa[theirs_valid], kappa[theirs_valid])
        record_property("wat_avor_max_deg", float(angles.max(initial=0)))
        record_property("wat_kappa_max_rel", float(rel.max(initial=0)))
        record_property(
            "wat_kappa_ratio_range",
            [
                float(np.min(ours_kappa[theirs_valid] / kappa[theirs_valid])),
                float(np.max(ours_kappa[theirs_valid] / kappa[theirs_valid])),
            ],
        )
        assert np.all(angles <= WAT_AVOR_MAX_DEG)

    @pytest.mark.parametrize("scenario", SCENARIOS_HROSM)
    def test_new_euler_lies_in_the_grain_ball(self, scenario):
        # newEuler is in radians: degrees would leave the ball. The
        # ball is a misorientation of the raw average, so no symmetry
        ref = load_reference(f"large_{scenario}")
        reindexed = np.asarray(ref["newCI"]) > 0
        assert reindexed.any()
        labels = np.asarray(ref["grainID"])[reindexed]
        assert np.all(labels > 0)
        new = Rotation.from_euler(
            np.asarray(ref["newEuler"])[reindexed].astype(np.float64)
        ).data
        centre = np.asarray(ref["avor"])[labels - 1]
        assert np.all(rotation_angle_deg(new, centre) <= MISORANG + 1e-3)

    @pytest.mark.parametrize("scenario", SCENARIOS_HROSM)
    def test_unprocessed_pixels_are_zero_filled(self, scenario, record_property):
        ref = load_reference(f"large_{scenario}")
        grain_id = np.asarray(ref["grainID"])
        npixels = np.asarray(ref["npixels"])
        kappa = np.asarray(ref["kappa"])
        skipped = np.flatnonzero((npixels < EMSOFT_MIN_PIXELS) | (kappa == -1)) + 1
        unprocessed = (grain_id == 0) | np.isin(grain_id, skipped)
        record_property(f"{scenario}_unprocessed_points", int(unprocessed.sum()))
        assert np.all(np.asarray(ref["newOSM"])[unprocessed] == 0)
        assert np.all(np.asarray(ref["newEuler"])[unprocessed] == 0)
        assert np.all(np.asarray(ref["newCI"])[unprocessed] == 0)

    def test_ni6_clustering_is_bitwise(self, emsoft_data_file):
        hrosm = read_local_hrosm_file(emsoft_data_file(*NI6_HROSM_FILE))
        assert float(hrosm["gangle"]) == NI6_GANGLE
        assert hrosm["dilate"] is True
        labels = segment_grains_kam(
            np.asarray(hrosm["kam"]),
            threshold=NI6_GANGLE,
            dilate=True,
            emsoft_compatible=True,
        )
        assert np.array_equal(labels, hrosm["grainID"])
        assert int(labels.max()) == NI6_N_GRAINS == int(hrosm["nGrains"])
        counts = np.bincount(labels.ravel(), minlength=NI6_N_GRAINS + 1)[1:]
        assert np.array_equal(counts, hrosm["npixels"])
        assert np.array_equal(
            grain_bounding_boxes(labels), emsoft_grain_roi_to_boxes(hrosm["grainROI"])
        )

    def test_ni6_watson_average_within_bands(self, emsoft_data_file, record_property):
        di = read_local_dot_product_file(emsoft_data_file(*NI6_DI_FILE))
        hrosm = read_local_hrosm_file(emsoft_data_file(*NI6_HROSM_FILE))
        kappa = np.asarray(hrosm["kappa"])
        # none rejected; concentrations from about 8 to about 35,000
        assert np.all(kappa != -1)
        assert 8 < kappa.min() < 8.5
        assert 34_000 < kappa.max() < 35_000
        xmap = crystal_map_from_euler(di["RefinedEulerAngles"], di[READER_SHAPE_KEY])
        table = watson_table(xmap, np.asarray(hrosm["grainID"]))
        # both keep every grain
        assert np.all(np.asarray(table.valid))

        seed_dependent = np.zeros(kappa.size, dtype=bool)
        seed_dependent[np.asarray(NI6_WAT_SEED_DEPENDENT_GRAINS) - 1] = True
        tight = (kappa >= NI6_WAT_TIGHT_MIN_KAPPA) & ~seed_dependent
        assert int((~tight).sum()) == NI6_WAT_LOOSE_GRAIN_COUNT

        ours = table.rotation.data[tight]
        angles = symmetry_reduced_angle_deg(ours, np.asarray(hrosm["avor"])[tight])
        rel = kappa_relative_difference(np.asarray(table.kappa)[tight], kappa[tight])
        record_property("ni6_wat_avor_max_deg", float(angles.max(initial=0)))
        record_property("ni6_wat_kappa_max_rel", float(rel.max(initial=0)))
        assert np.all(angles <= NI6_WAT_AVOR_MAX_DEG)
        assert np.all(rel <= NI6_WAT_KAPPA_REL)

    def test_ni6_small_grains_are_not_reindexed(self, emsoft_data_file):
        hrosm = read_local_hrosm_file(emsoft_data_file(*NI6_HROSM_FILE))
        grain_id = np.asarray(hrosm["grainID"])
        for label in NI6_SMALL_GRAINS:
            assert int(hrosm["npixels"][label - 1]) < EMSOFT_MIN_PIXELS
            mask = grain_id == label
            assert mask.any()
            assert np.all(np.asarray(hrosm["newOSM"])[mask] == 0)
            assert np.all(np.asarray(hrosm["newEuler"])[mask] == 0)
            assert np.all(np.asarray(hrosm["newCI"])[mask] == 0)


# ======================= The reference files ======================== #


class TestReferenceFiles:
    def test_registry_lists_every_reference(self):
        # anchored at the installed package, not at src
        found = {
            path.name for path in reference_directory().glob("regression_hrosm_*.npz")
        }
        keys = registry_keys()
        assert {Path(key).name for key in keys} == found
        assert len(found) == len(REFERENCE_TABLE)
        for key in keys:
            dataset = Dataset(key)
            assert dataset.is_in_package
            assert dataset.has_correct_hash, key

    def test_scenario_set_is_complete(self):
        # the script's own table, the registry, the data directory and
        # this module's frozen table; importing the script is safe
        from kikuchipy.data.emsoft_hrosm import create_hrosm_reference

        script = set(create_hrosm_reference.SCENARIOS)
        registry = {reference_name(key) for key in registry_keys()}
        directory = {
            reference_name(path)
            for path in reference_directory().glob("regression_hrosm_*.npz")
        }
        frozen = set(REFERENCE_TABLE)
        assert script == registry == directory == frozen

    @pytest.mark.parametrize("name", sorted(REFERENCE_TABLE))
    def test_references_load_without_pickle(self, name):
        with np.load(reference_path(name), allow_pickle=False) as reference:
            assert len(reference.files) > 0

    @pytest.mark.parametrize("name", sorted(REFERENCE_TABLE))
    def test_frozen_keys_and_dtypes(self, name):
        ref = load_reference(name)
        table = {**REFERENCE_TABLE[name], **PROVENANCE_KEYS}
        assert set(ref) == set(table)
        n_grains = int(ref["nGrains"]) if "nGrains" in ref else None
        for key, (dtype, shape) in table.items():
            array = ref[key]
            if dtype is np.str_:
                assert array.dtype.kind == "U", key
            else:
                assert array.dtype == dtype, key
            shape = tuple(n_grains if s == "n" else s for s in shape)
            assert array.shape == shape, key
        if "TopMatchIndices" in ref:
            # 1-based dictionary indices
            assert ref["TopMatchIndices"].min() >= 1

    @pytest.mark.parametrize("name", sorted(REFERENCE_TABLE))
    def test_provenance_pins(self, name):
        ref = load_reference(name)
        assert str(ref["master_md5"]) == MASTER_MD5
        master_run_md5 = str(ref["master_run_md5"])
        assert re.fullmatch(r"[0-9a-f]{32}", master_run_md5)
        assert master_run_md5 != MASTER_MD5
        assert int(ref["numdictsingle"]) == int(ref["numexptsingle"]) == 32
        program_md5 = str(ref["program_md5"])
        pairs = dict(pair.split("=") for pair in program_md5.split(";"))
        assert sorted(pairs) == sorted(EMSOFT_BINARIES)
        assert program_md5.split(";") == sorted(program_md5.split(";"))
        for value in pairs.values():
            assert re.fullmatch(r"[0-9a-f]{32}", value)
        assert str(ref["emsoft_version"]).startswith("6_0_")

        # every reference names the same programs and the same master
        for other in REFERENCE_TABLE:
            other_ref = load_reference(other)
            assert str(other_ref["program_md5"]) == program_md5
            assert str(other_ref["master_run_md5"]) == master_run_md5

        text = str(ref["namelist"])
        assert namelist_value_matches(text, "nnk", "20")
        assert namelist_value_matches(text, "nosm", "10")
        assert namelist_value_matches(text, "ncubochoric", "100")
        for key, value in re.findall(r"(\w+)\s*=\s*'([^']*)'", text):
            if key in NAMELIST_PATH_KEYS:
                assert value.startswith("kikuchipy_hrosm/") or value == "undefined", (
                    key,
                    value,
                )
            elif key == "tmpfile":
                assert "/" not in value and "\\" not in value, value

        if "hrosm_namelist" in ref:
            scenario = name[len("large_") :]
            hrosm_text = str(ref["hrosm_namelist"])
            orav = SCENARIO_ORAV[scenario]
            assert namelist_value_matches(hrosm_text, "orav", f"'{orav}'")
            dilate = ".TRUE." if SCENARIO_DILATE[scenario] else ".FALSE."
            if SCENARIO_DILATE[scenario] or re.search(
                r"\bdilate\b", hrosm_text, flags=re.IGNORECASE
            ):
                assert namelist_value_matches(hrosm_text, "dilate", dilate)

    def test_pc_route_is_reproduced(self):
        pytest.importorskip("pooch")
        signal = kp.data.nickel_ebsd_large(allow_download=True, lazy=True)
        det = signal.detector
        det.pc = det.pc_average
        pc = det.pc_emsoft()[0]
        xpc, ypc = pc[:2] / det.binning
        values = (xpc, ypc, pc[2], det.px_size * det.binning)
        expected = np.array([float(f"{value:.6g}") for value in values])
        for name in REFERENCE_TABLE:
            assert np.array_equal(load_reference(name)["pc"], expected), name

    def test_each_file_within_budget(self, record_property):
        sizes = {
            name: reference_path(name).stat().st_size
            for name in sorted(REFERENCE_TABLE)
        }
        record_property(
            "reference_bytes", ", ".join(f"{k} {v}" for k, v in sizes.items())
        )
        total = sum(sizes.values())
        record_property("reference_bytes_total", total)
        for name, size in sizes.items():
            assert size < FILE_BUDGET_BYTES, name
        assert total <= REFERENCE_TOTAL_BYTES

    def test_script_is_import_safe(self, monkeypatch):
        monkeypatch.delenv("KIKUCHIPY_EMSOFT_BIN", raising=False)
        monkeypatch.delenv("KIKUCHIPY_EMSOFT_DATA", raising=False)
        name = "kikuchipy.data.emsoft_hrosm.create_hrosm_reference"
        monkeypatch.delitem(sys.modules, name, raising=False)

        def refuse(*args, **kwargs):
            raise AssertionError("the script opened a file at import")

        monkeypatch.setattr(builtins, "open", refuse)
        module = importlib.import_module(name)
        monkeypatch.undo()
        assert tuple(module.SCENARIOS) == (
            "large_di",
            "large_refined",
            "large_center",
            "large_center_dilate",
            "large_wat",
            "ball_n6",
        )


# ==================== The private EMsoft reader ===================== #


def dot_product_arrays(
    H: int, W: int, nnk: int = 5, n_single: int = 8, seed: int = 60
) -> dict:
    """Return the arrays and namelist values of a minimal dot product
    file of an H x W map, Euler angles in radians.
    """
    rng = np.random.default_rng(seed)
    n = H * W
    return {
        "TopMatchIndices": rng.integers(1, 100, size=(n, nnk)).astype(np.int32),
        "TopDotProductList": rng.random((n, nnk)).astype(np.float32),
        "EulerAngles": (rng.random((n, 3)) * np.pi).astype(np.float32),
        "RefinedEulerAngles": (rng.random((n, 3)) * np.pi).astype(np.float32),
        "KAM": rng.random((H, W)).astype(np.float32),
        "OSM": rng.random((H, W)).astype(np.float32),
        "CI": rng.random(n).astype(np.float32),
        "nnk": nnk,
        "nosm": nnk,
        "ipf_wd": W,
        "ipf_ht": H,
        "ROI": [0, 0, 0, 0],
        "numdictsingle": n_single,
        "numexptsingle": n_single,
        "xpc": 4.6044,
        "ypc": 17.182,
        "L": 240.996,
        "delta": 8.0,
        "thetac": 0.0,
        "energymin": 15.0,
        "energymax": 20.0,
    }


def hrosm_arrays(H: int, W: int, dilate: bool, seed: int = 61) -> dict:
    """Return the arrays and namelist values of a minimal EMHROSM file
    of an H x W map with two grains.
    """
    rng = np.random.default_rng(seed)
    grain_id = np.zeros((H, W), dtype=np.int32)
    grain_id[:, : W // 2] = 1
    grain_id[:, W // 2 :] = 2
    return {
        "nGrains": np.int32(2),
        "grainID": grain_id,
        "npixels": np.array([H * (W // 2), H * (W - W // 2)], dtype=np.int32),
        "grainROI": np.array(
            [[1, 1, W // 2, H], [W // 2 + 1, 1, W - W // 2, H]]
        ).astype(np.int32),
        "avor": rng.normal(size=(2, 4)),
        "kappa": np.array([12.5, -1.0]),
        "kam": rng.random((H, W)).astype(np.float32),
        "newOSM": rng.random((H, W)).astype(np.float32),
        "newEuler": (rng.random((H, W, 3)) * np.pi).astype(np.float32),
        "newCI": rng.random((H, W)).astype(np.float32),
        "gangle": 5.0,
        "misorang": 5.0,
        "nsamples": 20,
        "nosm": 10,
        "orav": "center",
        "numEM": 25,
        "numIter": 40,
        "maxRAMmem": 1.0,
        "dpfile": "kikuchipy_hrosm/run/dp-refined.h5",
        "OSMfile": "kikuchipy_hrosm/run/hrosm_center.h5",
        "OSMtiff": "kikuchipy_hrosm/run/hrosm_center.tiff",
        "IPFmap": "undefined",
        "dilate": dilate,
    }


class TestEMsoftFileReader:
    def test_dot_product_file_shapes_and_padding(
        self, tmp_path, write_emsoft_layout_file
    ):
        H, W = 3, 4
        arrays = dot_product_arrays(H, W, nnk=5, n_single=8)
        fpath = write_emsoft_layout_file(tmp_path / "dp.h5", "dot_product", arrays)
        with h5py.File(fpath, "r") as f:
            # EMsoft pads the lists to a multiple of numexptsingle
            assert f["Scan 1/EBSD/Data/TopMatchIndices"].shape == (16, 5)

        data = read_emsoft_dot_product_file(fpath)
        assert tuple(data[READER_SHAPE_KEY]) == (H, W)
        assert data[READER_VERSION_KEY] == "6_0_20260411_0"
        top = data["TopMatchIndices"]
        assert top.shape == (H * W, 5)
        assert np.array_equal(top, arrays["TopMatchIndices"])
        assert top.min() >= 1
        assert data["EulerAngles"].shape == (H * W, 3)
        assert data["RefinedEulerAngles"].shape == (H * W, 3)
        assert np.array_equal(data["KAM"], arrays["KAM"])
        assert data["KAM"].shape == (H, W)
        assert np.array_equal(data["OSM"], arrays["OSM"])
        assert np.array_equal(data["CI"], arrays["CI"])
        for key in ("nnk", "nosm", "ipf_wd", "ipf_ht", "numexptsingle"):
            assert type(data[key]) is int, key
            assert data[key] == arrays[key], key
        for key in ("xpc", "ypc", "L", "delta", "energymax"):
            assert type(data[key]) is float, key
            assert data[key] == float(np.float32(arrays[key])), key

    def test_roi_sets_the_map_shape(self, tmp_path, write_emsoft_layout_file):
        # ROI is (x0, y0, w, h): a 4 wide, 3 high region of a 10 x 8 map
        H, W = 3, 4
        arrays = dot_product_arrays(H, W, n_single=8)
        arrays["ipf_wd"] = 10
        arrays["ipf_ht"] = 8
        arrays["ROI"] = [2, 3, 4, 3]
        fpath = write_emsoft_layout_file(tmp_path / "dp.h5", "dot_product", arrays)
        data = read_emsoft_dot_product_file(fpath)
        assert tuple(data[READER_SHAPE_KEY]) == (3, 4)
        assert data["TopMatchIndices"].shape == (12, 5)

    def test_euler_datasets_are_returned_in_radians(
        self, tmp_path, write_emsoft_layout_file
    ):
        H, W = 3, 4
        arrays = dot_product_arrays(H, W)
        fpath = write_emsoft_layout_file(tmp_path / "dp.h5", "dot_product", arrays)
        data = read_emsoft_dot_product_file(fpath)
        for key in ("EulerAngles", "RefinedEulerAngles"):
            assert data[key].dtype == np.float32, key
            assert np.array_equal(data[key], arrays[key]), key

        hrosm = hrosm_arrays(H, W, dilate=False)
        fpath = write_emsoft_layout_file(tmp_path / "hrosm.h5", "hrosm", hrosm)
        data = read_emsoft_hrosm_file(fpath)
        assert data["newEuler"].dtype == np.float32
        assert data["newEuler"].shape == (H, W, 3)
        assert np.array_equal(data["newEuler"], hrosm["newEuler"])

    @pytest.mark.parametrize("dilate", [True, False])
    def test_hrosm_file_and_dilate_from_the_namelist_text(
        self, tmp_path, write_emsoft_layout_file, dilate
    ):
        H, W = 3, 4
        arrays = hrosm_arrays(H, W, dilate=dilate)
        fpath = write_emsoft_layout_file(tmp_path / "hrosm.h5", "hrosm", arrays)
        with h5py.File(fpath, "r") as f:
            assert "dilate" not in f["NMLparameters/HROSMNameList"]

        data = read_emsoft_hrosm_file(fpath)
        assert data["dilate"] is dilate
        assert data["newQuat"] is None
        assert data["maxGROD"] is None
        assert int(data["nGrains"]) == 2
        assert np.array_equal(data["grainID"], arrays["grainID"])
        assert data["grainID"].shape == (H, W)
        # grainROI stays 1-based (x0, y0, w, h)
        assert np.array_equal(data["grainROI"], arrays["grainROI"])
        assert np.array_equal(data["npixels"], arrays["npixels"])
        assert np.array_equal(data["avor"], arrays["avor"])
        assert np.array_equal(data["kappa"], arrays["kappa"])
        assert np.array_equal(data["kam"], arrays["kam"])
        assert data["gangle"] == 5.0
        assert data["nosm"] == 10
        assert data["orav"] == "center"
        assert "dilate" in data[READER_TEXT_KEY]
        assert data[READER_VERSION_KEY] == "6_0_20260411_0"

    def test_parse_namelist_text_types_the_values(self):
        text = "\n".join(
            [
                " &HROSMdata",
                "! The line above must not be changed",
                " dilate = .TRUE.,",
                " gangle = 5.0,",
                " nsamples = 20,",
                " orav = 'averageWAT',",
                " ROI = 0 0 0 0,",
                " /",
            ]
        )
        values = parse_namelist_text(text)
        assert values["dilate"] is True
        assert values["gangle"] == 5.0
        assert type(values["gangle"]) is float
        assert values["nsamples"] == 20
        assert type(values["nsamples"]) is int
        assert values["orav"] == "averageWAT"
        assert list(values["ROI"]) == [0, 0, 0, 0]
        assert parse_namelist_text(" dilate = .FALSE.,")["dilate"] is False

    def test_parse_namelist_text_keeps_an_unquoted_word(self):
        values = parse_namelist_text(" indexingmode = dynamic,\n nnk = 5,")
        assert values["indexingmode"] == "dynamic"
        assert values["nnk"] == 5

    def test_dot_product_file_text_strings_and_subgroups(
        self, tmp_path, write_emsoft_layout_file
    ):
        H, W = 3, 4
        arrays = dot_product_arrays(H, W)
        arrays["namelist_text"] = [" &DictIndxOpenCLListData", " nnk = 5,", " /"]
        fpath = write_emsoft_layout_file(tmp_path / "dp.h5", "dot_product", arrays)
        string = h5py.string_dtype("ascii")
        with h5py.File(fpath, "a") as f:
            # A string data set and a group among the data sets and
            # among the namelist values
            f.create_dataset(
                "Scan 1/EBSD/Data/Phase", data=np.array([b"Ni"], dtype=string)
            )
            f.create_group("Scan 1/EBSD/Data/Extra")
            f.create_group("NMLparameters/EMDINameList/Extra")

        data = read_emsoft_dot_product_file(fpath)
        assert data[READER_TEXT_KEY] == "\n".join(arrays["namelist_text"])
        assert data["Phase"] == "Ni"
        assert "Extra" not in data
        assert tuple(data[READER_SHAPE_KEY]) == (H, W)


# ====================== The EMsoft program lock ===================== #


class TestEMsoftProgramLock:
    def test_takes_over_a_stale_lock_heartbeats_times_out_and_releases(
        self, tmp_path, emsoft_program_lock
    ):
        path = tmp_path / "emsoft-program.lock"
        path.write_text("")
        old = time.time() - 1000.0

        os.utime(path, (old, old))

        with emsoft_program_lock(path, timeout=5.0, heartbeat=0.05):
            # the stale file was taken over
            assert path.exists()
            aged = time.time() - 100.0
            os.utime(path, (aged, aged))
            deadline = time.monotonic() + 5.0
            while path.stat().st_mtime <= aged + 50.0:
                assert time.monotonic() < deadline, "the heartbeat did not run"
                time.sleep(0.02)

            with pytest.raises(TimeoutError) as error:
                with emsoft_program_lock(path, timeout=0.2, heartbeat=0.05):
                    pass  # pragma: no cover
            assert path.name in str(error.value)

        assert not path.exists()


# ===================== Regenerating the references ================== #


class TestRegenerateReferences:
    def test_the_mismatch_message_names_the_programs_and_the_arrays(self, tmp_path):
        # the diagnostic of the gated test, exercised on stand-ins:
        # one drops an array, widens one and changes the re-indexed
        # map (not reproducible run to run, so not reported); one
        # changes a bitwise array; one has three grains more
        shipped = load_reference("large_center")
        arrays = {key: shipped[key] for key in shipped if key != "newCI"}
        arrays["kam"] = shipped["kam"].astype(np.float64)
        arrays["newOSM"] = shipped["newOSM"] + np.float32(1)
        center = tmp_path / "regression_hrosm_large_center.npz"
        np.savez(center, **arrays)

        shipped = load_reference("large_di")
        arrays = dict(shipped)
        arrays["CI"] = shipped["CI"].copy()
        arrays["CI"][0] += np.float32(0.5)
        di = tmp_path / "regression_hrosm_large_di.npz"
        np.savez(di, **arrays)

        shipped = load_reference("large_wat")
        n = int(shipped["nGrains"])
        arrays = dict(shipped)
        arrays["nGrains"] = np.int32(n + 3)
        for key in ("npixels", "kappa", "grainROI", "avor"):
            arrays[key] = np.concatenate([shipped[key], shipped[key][:3]])
        wat = tmp_path / "regression_hrosm_large_wat.npz"
        np.savez(wat, **arrays)

        written = {"large_center": center, "large_di": di, "large_wat": wat}
        with pytest.raises(AssertionError) as error:
            assert_regenerated(written)
        message = str(error.value)
        assert "program_md5" in message
        assert "the contract is bitwise" in message
        assert "kam: float64(55, 75) vs float32(55, 75)" in message
        assert "newCI: only in one of the two files" in message
        assert "CI: differs in 1 values (bitwise)" in message
        assert f"nGrains: {n + 3} vs {n} grains, more than 2 apart" in message
        # the arrays which agree, and those outside the bitwise set
        # with the right layout, are not listed
        assert "grainROI" not in message
        assert "newOSM" not in message
        assert "TopMatchIndices" not in message

        # a grain count within the tolerance with a consistent layout
        # passes, and a layout which does not fit the grain count fails
        arrays["nGrains"] = np.int32(n + 2)
        for key in ("npixels", "kappa", "grainROI", "avor"):
            arrays[key] = arrays[key][:-1]
        np.savez(wat, **arrays)
        assert_regenerated({"large_wat": wat})
        arrays["kappa"] = arrays["kappa"][:-1]
        np.savez(wat, **arrays)
        with pytest.raises(AssertionError, match=r"kappa: float64\(\d+,\), layout"):
            assert_regenerated({"large_wat": wat})

    def test_regenerated_references_are_bitwise(
        self,
        emsoft_bin_dir,
        emsoft_program,
        emsoft_program_lock,
        tmp_path,
        record_property,
    ):
        from kikuchipy.data.emsoft_hrosm import create_hrosm_reference

        shipped_directory = reference_directory()
        shipped_before = {
            path.name: md5_of_file(path)
            for path in shipped_directory.glob("regression_hrosm_*.npz")
        }
        data_root = Path(emsoft_program.data_root)
        run_dir, _ = emsoft_program.new_run_dir()
        root_before = {path.name for path in data_root.iterdir()}

        start = time.monotonic()
        # the caller holds the lock; main does not take it again when
        # it is handed the program directory
        with emsoft_program_lock():
            written = create_hrosm_reference.main(
                output_dir=tmp_path, bin_dir=emsoft_bin_dir, run_dir=run_dir
            )
        record_property("REGENERATION_RUNTIME_S", f"{time.monotonic() - start:.1f}")
        record_property("REGENERATION_RUNTIME_S_pinned", REGENERATION_RUNTIME_S)

        # nothing new in EMsoft's data root, the shipped files untouched
        assert {path.name for path in data_root.iterdir()} == root_before
        assert {
            path.name: md5_of_file(path)
            for path in shipped_directory.glob("regression_hrosm_*.npz")
        } == shipped_before
        assert sorted(written) == sorted(REFERENCE_TABLE)
        for fpath in written.values():
            assert Path(fpath).parent == tmp_path

        # the acid bands of the new run (the script aborts before
        # writing outside them; asserted again from its log)
        top1, refined = acid_band_medians(Path(run_dir) / "run.log")
        record_property("regeneration_top1_median_deg", top1)
        record_property("regeneration_refined_median_deg", refined)
        assert top1 <= create_hrosm_reference.TOP1_MEDIAN_MAX_DEG
        assert refined <= create_hrosm_reference.REFINED_MEDIAN_MAX_DEG

        # the top matches, their dot products and the ball lists equal
        # those of the run the shipped files came from (the shipped
        # files hold only part of them)
        shipped_run = shipped_run_directory(data_root)
        assert (shipped_run / "dp.h5").is_file(), shipped_run
        new_top = read_raw_top_matches(Path(run_dir) / "dp.h5")
        shipped_top = read_raw_top_matches(shipped_run / "dp.h5")
        for key in REGENERATION_BITWISE_DOT_PRODUCT_KEYS:
            assert new_top[key].dtype == shipped_top[key].dtype, key
            assert np.array_equal(new_top[key], shipped_top[key]), key
        for fname in REGENERATION_BALL_FILES:
            new_ball = (Path(run_dir) / fname).read_bytes()
            assert new_ball == (shipped_run / fname).read_bytes(), fname

        # newQuat is the float32 eq_ of newEuler on re-indexed points
        for scenario in SCENARIOS_HROSM:
            data = read_emsoft_hrosm_file(Path(run_dir) / f"hrosm_{scenario}.h5")
            assert data["newQuat"] is not None
            reindexed = np.asarray(data["newCI"]) > 0
            new_quat = np.asarray(data["newQuat"]).reshape(MAP_SHAPE + (4,))
            expected = emsoft_eq(
                np.asarray(data["newEuler"])[reindexed].astype(np.float64)
            ).astype(np.float32)
            assert np.array_equal(new_quat[reindexed], expected)

        assert_regenerated(written, record_property)
