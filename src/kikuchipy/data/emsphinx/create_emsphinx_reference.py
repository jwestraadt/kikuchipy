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

r"""Creation of the eight shipped ``regression_*.npz`` EMSphInx
reference files next to this script.

Run once; the output is committed.  This script is excluded from the
doctest job by ``--ignore-glob=src/kikuchipy/data/emsphinx/*.py`` and
from coverage by ``omit = ["src/kikuchipy/data/*/create_*.py"]``, both
in ``pyproject.toml``.

The module is **import safe**: importing it looks up no environment
variable, probes no executable and opens no file, so that the shipped
regression tests can import :data:`SCENARIOS` and compare it with the
registry, the data directory and their own frozen table.  Everything
else happens inside :func:`main`.

What it does, per scenario of :data:`SCENARIOS`, in a temporary run
directory:

1. repacks the background corrected patterns with
   :func:`~kikuchipy.indexing.write_emsphinx_patterns` on the
   **canonical route**, i.e. with every writer default
   (``Manufacturer`` EMsoft, rows unflipped), and guard asserts that
   the written ``/patterns`` data set is ``uint8`` and byte identical
   to ``signal.data.reshape(-1, h, w)``;
2. copies the in package ``ni_small_20kv_bw384.sht`` master beside it;
3. builds the name list with
   :meth:`~kikuchipy.indexing.EMSphInxNamelist.from_kwargs` (name list
   vendor Bruker with the unrounded ``pc_average`` unless the scenario
   says EMsoft, ``bw`` 68, ``normed`` true, ``nthread`` 1,
   ``batchsize`` 1) and keeps its text verbatim;
4. runs ``IndexEBSD`` there and requires exit code 0;
5. reads ``Scan 1/EBSD/Data/{Phi1, Phi, Phi2, Metric, IQ, Phase}`` of
   the output data file -- the payload, five ``float32`` arrays and one
   ``uint8`` one -- and cross checks it against the ``.ang``, whose
   columns are text rounded (Euler at five decimals, ``ci`` = the
   metric at three, ``iq`` = the image quality at one);
6. writes one uncompressed :func:`numpy.savez` file holding those six
   arrays and the provenance the regression tests recompute against,
   the exact name list text and the ``.6g`` round tripped pattern
   centre the program used among them.

Before anything runs, ``git rev-parse HEAD`` of the EMSphInx checkout
is asserted to be commit ``60f351741036c63a59a6061a7ac2fca4f60f2c64``,
and the machine wide program lock file is taken, since the programs
race on one shared FFTW wisdom file.  That wisdom is also why the
reference bytes are only reproducible **given a fixed wisdom state**:
the programs plan with ``FFTW_PATIENT`` and import and export
``<shared data dir>/fftw.wisdom`` at start and exit, and different
plans round differently, so its md5 is printed before and after the
sweep.

The equivalent shell commands, with ``EMSPHINX`` the EMSphInx checkout
built in ``build/Release``::

    export KIKUCHIPY_EMSPHINX_DIR=$EMSPHINX
    uv run python src/kikuchipy/data/emsphinx/create_emsphinx_reference.py
    # which is, per scenario, in a temporary directory:
    #   (a) write patterns.h5, copy ni.sht, write index.nml
    #   (b) $EMSPHINX/build/Release/IndexEBSD.exe index.nml
    #   (c) read out.h5, cross check out.ang
    #   (d) numpy.savez regression_<scenario>.npz
    #   (e) md5sum of each file -> _registry.py

Measured on the machine the shipped files were generated on: about
11 s for the whole sweep, every file bitwise reproducible across full
reruns under an unchanged wisdom.
"""

from collections import namedtuple
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# The EMSphInx commit the references are generated from and the only
# one this script accepts.  The regression test module keeps its own
# copy of this literal: the value stored in every reference file is
# *probed* from the checkout, so the shipped check has power only
# through that independent duplication.
EMSPHINX_COMMIT = "60f351741036c63a59a6061a7ac2fca4f60f2c64"

# The in-package master pattern of every scenario, next to this file
MASTER_SHT = "ni_small_20kv_bw384.sht"

# Everything the scenario matrix holds fixed
SAMPLE_TILT = 70.0
BANDWIDTH = 68
NORMALIZE = True
CIRCULAR_MASK = False
EMSPHINX_COMPATIBLE = True
MANUFACTURER = "EMsoft"
FLIP = False

# The canonical route, recorded in every file so that a writer default
# which changes upstream is visible in the reference itself
ROUTE = (
    "write_emsphinx_patterns defaults (Manufacturer EMsoft, rows unflipped); "
    "EMSphInxNamelist.from_kwargs with the unrounded pc_average; "
    "nthread=1 batchsize=1"
)

# The recipe whose silent change would invalidate every reference
PREPROCESSING = (
    "EBSD.remove_static_background() then EBSD.remove_dynamic_background(), "
    "kikuchipy defaults"
)

# File names inside a run directory
PATTERN_FILE = "patterns.h5"
MASTER_FILE = "ni.sht"
NAMELIST_FILE = "index.nml"
DATA_FILE = "out.h5"
VENDOR_FILE = "out.ang"

# The data file's result data sets, in the order they are stored under
# ``Scan 1/EBSD/Data``
DATA_PATH = "Scan 1/EBSD/Data"
RESULT_KEYS = (("phi1", "Phi1"), ("phi", "Phi"), ("phi2", "Phi2"))

# Generation-time ``.ang`` cross-check tolerances, each exactly twice
# the deterministic half-ULP bound of the column's fixed precision:
# five decimals on the Euler angles, three on ``ci``
ANG_EULER_TOLERANCE = 1e-5
ANG_METRIC_TOLERANCE = 1e-3

# Generation-time acceptance guard: the median misorientation of a
# small-map scenario against the stored crystal map, the Phase 9 acid
# band.  A wrong flip pairing measures about 39.6 degrees and a non
# ``uint8`` repack about 38.9.
ACID_MEDIAN_DEG = 1.2

# Seconds to wait for the machine-wide program lock, and the age at
# which one is assumed to belong to a killed run and taken over.  The
# protocol is the test suite's, so that a manual run of this script
# and a gated test run serialise against one another.
LOCK_TIMEOUT = 600.0
LOCK_STALE = 900.0
LOCK_NAME = "kikuchipy-emsphinx-program.lock"


Scenario = namedtuple(
    "Scenario",
    ["name", "dataset", "vendor", "delta", "refine", "nregions", "gausbckg"],
)
"""One reference: its file is ``regression_<name>.npz``."""


SCENARIOS = (
    Scenario(
        "small_coarse_nr10", "nickel_ebsd_small", "Bruker", 500.0, False, 10, False
    ),
    Scenario(
        "small_refined_nr10", "nickel_ebsd_small", "Bruker", 500.0, True, 10, False
    ),
    Scenario("small_refined_nr0", "nickel_ebsd_small", "Bruker", 500.0, True, 0, False),
    Scenario("small_refined_nr7", "nickel_ebsd_small", "Bruker", 500.0, True, 7, False),
    Scenario(
        "small_refined_nr10_gb", "nickel_ebsd_small", "Bruker", 500.0, True, 10, True
    ),
    Scenario(
        "small_refined_emsoft_d500",
        "nickel_ebsd_small",
        "EMsoft",
        500.0,
        True,
        10,
        False,
    ),
    Scenario(
        "large20_refined_nr10",
        "nickel_ebsd_large_20pt",
        "Bruker",
        500.0,
        True,
        10,
        False,
    ),
    Scenario(
        "large165_refined_nr10",
        "nickel_ebsd_large_165pt",
        "Bruker",
        500.0,
        True,
        10,
        False,
    ),
)
"""The frozen scenario matrix, one factor at a time around the
canonical anchor ``small_refined_nr10``: ``refine`` false and true,
``nregions`` 0, 7 (the mosaic equalisation remainder path, since
60 % 7 = 4) and 10, a Gaussian background scenario, the EMsoft pattern
centre route, and the two ``nickel_ebsd_large`` subsets.

There is deliberately no refine-only scenario, since the shipped
``refineImage`` discards its refinement, and no second ``delta``
scenario, since ``delta`` is inert on every vendor route: the pattern
centre it round trips through is bit identical for 125, 250 and 500,
which the name list unit tests pin instead.
"""

# Subset of each large map, as ``inav`` slices it and as recorded in
# the reference; the small map is unsliced
SUBSET_SLICES = {
    "nickel_ebsd_small": "",
    "nickel_ebsd_large_20pt": "::15,::15",
    "nickel_ebsd_large_165pt": "::5,::5",
}


def main(
    output_dir: str | Path | None = None,
    program: str | Path | None = None,
    scenarios: "list[str] | tuple[str, ...] | None" = None,
) -> dict[str, Path]:
    """Generate the reference files and return their paths.

    Parameters
    ----------
    output_dir
        Directory to write ``regression_<scenario>.npz`` into.  The
        directory of this script, i.e. the shipped location, by
        default.
    program
        Path of the built ``IndexEBSD`` executable.  If not given
        (default), it is resolved from the ``KIKUCHIPY_EMSPHINX_DIR``
        environment variable and the **machine wide program lock is
        taken for the whole sweep**.  A caller which passes the
        program is assumed to hold that lock already, which is what
        the gated regeneration test does through its
        ``emsphinx_program`` fixture; taking it twice from one process
        would dead lock against itself.
    scenarios
        Names of the scenarios to generate, all of :data:`SCENARIOS`
        by default.  The gated regeneration test passes the six small
        map ones, whose regeneration needs neither pooch nor the
        background correction of a full 4125 pattern map.

    Returns
    -------
    written
        Path of every written file, keyed on the scenario name.

    Raises
    ------
    FileNotFoundError
        If the executable is not built, or ``KIKUCHIPY_EMSPHINX_DIR``
        is not set and ``program`` is not given.
    RuntimeError
        If the EMSphInx checkout is not at :data:`EMSPHINX_COMMIT`, if
        ``IndexEBSD`` exits non-zero, or if any generation-time guard
        fails.
    ValueError
        If a scenario name is not one of :data:`SCENARIOS`.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    selected = _select(scenarios)
    if program is None:
        with _program_lock():
            return _sweep(output_dir, _resolve_program(), selected)
    return _sweep(output_dir, Path(program), selected)


# --------------------------- The sweep ------------------------------ #


def _sweep(
    output_dir: Path, program: Path, selected: "tuple[Scenario, ...]"
) -> dict[str, Path]:
    """Run every selected scenario, check the guards and write the
    files.

    Nothing is written before every guard has passed, so a mutated
    route cannot leave a plausible but wrong reference behind.

    The commit is probed **once**, from the checkout holding
    ``program``, and threaded into every reference: probing it again
    per scenario would both re-read ``KIKUCHIPY_EMSPHINX_DIR`` -- which
    a caller passing ``program`` need not have set -- and let the
    stored value come from a different checkout than the one the guard
    validated.
    """
    commit = _check_commit(program)
    wisdom = _wisdom_path()
    print(f"fftw wisdom before: {_wisdom_state(wisdom)}")

    references = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kikuchipy_emsphinx_ref_") as directory:
        for scenario in selected:
            run_dir = Path(directory) / scenario.name
            run_dir.mkdir()
            start = time.monotonic()
            references[scenario.name] = _generate(scenario, program, run_dir, commit)
            print(f"  {scenario.name}: {time.monotonic() - start:.2f} s")

    _check_across_scenarios(references)

    written = {}
    for name, reference in references.items():
        fpath = output_dir / f"regression_{name}.npz"
        np.savez(fpath, **reference)
        written[name] = fpath

    print(f"fftw wisdom after: {_wisdom_state(wisdom)}")
    print("md5 sums, for _registry.py:")
    total = 0
    for name, fpath in written.items():
        total += fpath.stat().st_size
        print(
            f'    "emsphinx/{fpath.name}":'.ljust(56)
            + f' "md5:{_md5_of_file(fpath)}",  # {fpath.stat().st_size} B'
        )
    print(f"total {total} B over {len(written)} files")
    return written


def _generate(scenario: Scenario, program: Path, run_dir: Path, commit: str) -> dict:
    """Return the reference arrays of one scenario, generated in
    ``run_dir``.

    ``commit`` is the sha :func:`_check_commit` probed from the
    checkout holding ``program``, stored as the reference's
    ``emsphinx_commit``.
    """
    from kikuchipy.indexing import EMSphInxNamelist, write_emsphinx_patterns

    signal, detector, scan_shape, scan_steps = _dataset(scenario.dataset)

    # (a) the pattern file, on the canonical route: every writer
    # default, so that ``/patterns`` is the flattened signal verbatim
    pattern_path = run_dir / PATTERN_FILE
    write_emsphinx_patterns(pattern_path, signal, overwrite=True)
    patterns = _read_patterns(pattern_path)
    height, width = signal.axes_manager.signal_shape[::-1]
    expected = np.asarray(signal.data).reshape(-1, height, width)
    if patterns.dtype != np.uint8:
        raise RuntimeError(
            f"{scenario.name}: the repacked patterns are {patterns.dtype} and "
            "not uint8, which EMSphInx reads through a buffered NATIVE_UINT8 "
            "read and corrupts"
        )
    if not np.array_equal(patterns, expected):
        raise RuntimeError(
            f"{scenario.name}: the repacked patterns are not byte identical "
            "to signal.data.reshape(-1, h, w), so the route is not the "
            "canonical one the references are pinned to"
        )

    # (b) the master pattern
    master_source = _master_path()
    shutil.copy(master_source, run_dir / MASTER_FILE)

    # (c) the name list, whose text is stored verbatim
    namelist = EMSphInxNamelist.from_kwargs(
        pattern_file=PATTERN_FILE,
        master_files=[MASTER_FILE],
        detector=detector,
        scan_shape=scan_shape,
        scan_steps=scan_steps,
        data_file=DATA_FILE,
        vendor_file=VENDOR_FILE,
        vendor=scenario.vendor,
        delta=scenario.delta,
        n_thread=1,
        batch_size=1,
        bandwidth=BANDWIDTH,
        normalize=NORMALIZE,
        refine=scenario.refine,
        n_regions=scenario.nregions,
        gaussian_background=scenario.gausbckg,
        circular_mask=CIRCULAR_MASK,
    )
    namelist_text = namelist.to_string()
    namelist.write(run_dir / NAMELIST_FILE, overwrite=True)

    # (d) the run
    result = subprocess.run(
        [str(program), NAMELIST_FILE],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{scenario.name}: IndexEBSD exited with {result.returncode}: "
            f"{result.stdout} {result.stderr}"
        )

    # (e) the payload and its ``.ang`` cross-check
    arrays = _read_data_file(run_dir / DATA_FILE)
    _check_ang(scenario, run_dir / VENDOR_FILE, arrays)

    n_points = int(scan_shape[0]) * int(scan_shape[1])
    for key, array in arrays.items():
        if array.shape != (n_points,):
            raise RuntimeError(
                f"{scenario.name}: the {key} data set has shape "
                f"{array.shape} and not ({n_points},)"
            )

    # (f) the acceptance guard of a small-map scenario
    if scenario.dataset == "nickel_ebsd_small":
        median = _median_against_stored_xmap(arrays)
        print(f"    acid median {median:.4f} deg")
        if not median < ACID_MEDIAN_DEG:
            raise RuntimeError(
                f"{scenario.name}: the median misorientation against the "
                f"stored crystal map is {median:.4f} degrees, which is not "
                f"below the acid band of {ACID_MEDIAN_DEG}"
            )

    # (g) the effective pattern centre: what the program read, i.e.
    # the name list values after ``to_string`` quantised them at six
    # significant digits
    pc = (
        EMSphInxNamelist.from_string(namelist_text)
        .to_detector(sample_tilt=SAMPLE_TILT)
        .pc.astype(np.float64)
        .ravel()
    )

    provenance = {
        "emsphinx_commit": np.str_(commit),
        "bw": np.int64(BANDWIDTH),
        "normed": np.bool_(NORMALIZE),
        "refine": np.bool_(scenario.refine),
        "nregions": np.int64(scenario.nregions),
        "gausbckg": np.bool_(scenario.gausbckg),
        "delta": np.float64(scenario.delta),
        "vendor": np.str_(scenario.vendor),
        "route": np.str_(ROUTE),
        "dataset": np.str_(scenario.dataset),
        "scan_shape": np.asarray(scan_shape, dtype=np.int64),
        "scan_steps": np.asarray(scan_steps, dtype=np.float64),
        "sample_tilt": np.float64(SAMPLE_TILT),
        "pc": pc,
        "namelist": np.str_(namelist_text),
        "master_sht": np.str_(MASTER_SHT),
        "master_md5": np.str_(_md5_of_file(master_source)),
        "patterns_md5": np.str_(_md5_of_array(patterns)),
        "preprocessing": np.str_(PREPROCESSING),
        "subset_slice": np.str_(SUBSET_SLICES[scenario.dataset]),
        "emsphinx_compatible": np.bool_(EMSPHINX_COMPATIBLE),
        "manufacturer": np.str_(MANUFACTURER),
        "flip": np.bool_(FLIP),
        "kikuchipy_version": np.str_(_kikuchipy_version()),
    }
    return {**arrays, **provenance}


def _check_across_scenarios(references: dict[str, dict]) -> None:
    """Check the guards which need two scenarios: the coarse and the
    refined run of the anchor share their preprocessing, so their
    image quality is bitwise equal, and refinement raises the metric
    of every point.
    """
    coarse = references.get("small_coarse_nr10")
    refined = references.get("small_refined_nr10")
    if coarse is None or refined is None:
        return
    if not np.array_equal(coarse["iq"], refined["iq"]):
        raise RuntimeError(
            "the coarse and refined image quality of the anchor are not "
            "bitwise equal, although refinement does not touch the "
            "preprocessing they are computed from"
        )
    deltas = refined["metric"].astype(np.float64) - coarse["metric"].astype(np.float64)
    print(
        f"    refined - coarse metric: min {deltas.min():+.5f} max {deltas.max():+.5f}"
    )
    if not (deltas > 0).all():
        raise RuntimeError(
            f"the refined metric is not above the coarse one at every point: {deltas}"
        )


# ------------------------- The inputs ------------------------------- #


def _dataset(name: str):
    """Return the background corrected signal, its single projection
    centre detector, its scan shape and its scan steps.

    The large maps have their background removed on the **full** map
    and are subset afterwards, and keep the full map's ``pc_average``,
    which is the convention of the existing large map tests.
    """
    import kikuchipy as kp

    if name == "nickel_ebsd_small":
        signal = kp.data.nickel_ebsd_small()
        step = 1
    elif name == "nickel_ebsd_large_20pt":
        signal = kp.data.nickel_ebsd_large(allow_download=True)
        step = 15
    elif name == "nickel_ebsd_large_165pt":
        signal = kp.data.nickel_ebsd_large(allow_download=True)
        step = 5
    else:  # pragma: no cover
        raise ValueError(f"unknown data set {name!r}")

    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)

    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average

    scale = float(signal.axes_manager.navigation_axes[0].scale)
    if step != 1:
        signal = signal.inav[::step, ::step]
    scan_shape = tuple(
        int(value) for value in signal.axes_manager.navigation_shape[::-1]
    )
    scan_steps = (scale * step, scale * step)
    return signal, detector, scan_shape, scan_steps


def _master_path() -> Path:
    """Return the path of the in-package master pattern, checked
    against its registry md5.
    """
    from kikuchipy.data._data import Dataset

    return Path(Dataset(f"emsphinx/{MASTER_SHT}").fetch_file_path())


def _read_patterns(fpath: Path) -> np.ndarray:
    """Return the ``/patterns`` data set of a written pattern file."""
    import h5py

    with h5py.File(fpath, mode="r") as f:
        return np.asarray(f["patterns"])


def _read_data_file(fpath: Path) -> dict[str, np.ndarray]:
    """Return the six result arrays of an ``IndexEBSD`` data file.

    The Euler angles, the metric and the image quality are 32-bit
    floats and the phase is an unsigned 8-bit integer, both measured
    against the program's own output; they are stored exactly as they
    are, one row per scan point in row major scan order.
    """
    import h5py

    arrays = {}
    with h5py.File(fpath, mode="r") as f:
        group = f[DATA_PATH]
        for key, name in RESULT_KEYS:
            arrays[key] = np.asarray(group[name], dtype=np.float32)
        arrays["metric"] = np.asarray(group["Metric"], dtype=np.float32)
        arrays["iq"] = np.asarray(group["IQ"], dtype=np.float32)
        arrays["phase"] = np.asarray(group["Phase"], dtype=np.uint8)
    return arrays


def _check_ang(scenario: Scenario, fpath: Path, arrays: dict) -> None:
    """Cross-check the data file against the ``.ang`` the same run
    wrote.

    The ``.ang`` is text rounded -- ``std::fixed`` with five decimals
    on the Euler angles, one on ``iq`` and three on ``ci`` -- so it is
    a cross-check and not the payload.  Each tolerance is twice the
    deterministic half-ULP bound of its column.
    """
    ang = np.loadtxt(fpath, comments="#")
    euler = np.stack([arrays["phi1"], arrays["phi"], arrays["phi2"]], axis=1)
    worst_euler = float(np.abs(ang[:, :3] - euler.astype(np.float64)).max())
    worst_metric = float(np.abs(ang[:, 6] - arrays["metric"].astype(np.float64)).max())
    if worst_euler > ANG_EULER_TOLERANCE or worst_metric > ANG_METRIC_TOLERANCE:
        raise RuntimeError(
            f"{scenario.name}: the .ang and the data file disagree by "
            f"{worst_euler:.3e} rad on the Euler angles and "
            f"{worst_metric:.3e} on the metric, beyond the "
            f"{ANG_EULER_TOLERANCE:.0e} and {ANG_METRIC_TOLERANCE:.0e} "
            "rounding bounds of the text columns"
        )


def _median_against_stored_xmap(arrays: dict) -> float:
    """Return the median m-3m misorientation in degrees between an
    indexed small map and the stored one.
    """
    from orix.quaternion import Orientation, Rotation
    from orix.quaternion.symmetry import Oh

    import kikuchipy as kp

    euler = np.stack([arrays["phi1"], arrays["phi"], arrays["phi2"]], axis=1).astype(
        np.float64
    )
    theirs = Orientation(Rotation.from_euler(euler).data, Oh)
    stored = kp.data.nickel_ebsd_small().xmap.rotations
    angles = theirs.angle_with(Orientation(stored.data, Oh), degrees=True)
    return float(np.median(np.asarray(angles, dtype=np.float64)))


# ----------------------- The EMSphInx checkout ---------------------- #


def _resolve_program() -> Path:
    """Return the built ``IndexEBSD`` of ``KIKUCHIPY_EMSPHINX_DIR``."""
    value = os.environ.get("KIKUCHIPY_EMSPHINX_DIR")
    if not value:
        raise FileNotFoundError(
            "KIKUCHIPY_EMSPHINX_DIR is not set; set it to an EMSphInx "
            f"checkout at {EMSPHINX_COMMIT} with build/Release/IndexEBSD, or "
            "pass the executable as `program`"
        )
    directory = Path(value) / "build" / "Release"
    for candidate in (directory / "IndexEBSD.exe", directory / "IndexEBSD"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"IndexEBSD is not built in {directory}")


def _checkout(program: Path) -> Path:
    """Return the EMSphInx checkout holding ``program``, i.e. the
    parent of its ``build`` directory.
    """
    program = Path(program).resolve()
    for parent in program.parents:
        if parent.name == "build":
            return parent.parent
    return program.parent


def _check_commit(program: Path) -> str:
    """Assert the checkout is at :data:`EMSPHINX_COMMIT` and return
    the probed sha.

    The stored ``emsphinx_commit`` is this returned value, so a stale
    pin and a wrong checkout both die here rather than in a reference
    nobody can tell apart from a good one, and what is stored is
    provably the sha this guard validated.
    """
    found = _commit(program)
    if found != EMSPHINX_COMMIT:
        raise RuntimeError(
            f"the EMSphInx checkout {_checkout(program)} is at {found} and "
            f"not at the pinned {EMSPHINX_COMMIT}, which the references and "
            "their tests are generated from"
        )
    return found


def _commit(program: Path) -> str:
    """Return ``git rev-parse HEAD`` of the checkout holding
    ``program``.
    """
    directory = _checkout(program)
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read the git commit of {directory}: "
            f"{result.stdout} {result.stderr}"
        )
    return result.stdout.strip()


@contextmanager
def _program_lock():
    """Hold the lock shared by every process running an EMSphInx
    program.

    The same file and protocol the test suite's ``emsphinx_program``
    fixture uses, so that a manual run of this script serialises
    against a concurrently running gated suite: the programs import
    and export one machine wide FFTW wisdom file in a global
    constructor and destructor, and two of them at once race on it.
    """
    path = Path(tempfile.gettempdir()) / LOCK_NAME
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > LOCK_STALE
            except OSError:  # it went away between the two calls
                continue
            if stale:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Waited {LOCK_TIMEOUT} s for the EMSphInx program lock "
                    f"{str(path)!r}. Delete it if no other process is running "
                    "an EMSphInx program"
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _wisdom_path() -> Path:
    """Return the machine wide FFTW wisdom file of the programs.

    ``getSharedDataDir()`` is ``C:\\ProgramData\\`` on Windows,
    ``~/Library/Application Support/`` on macOS and ``~/.local/share/``
    on Linux.
    """
    if sys.platform == "win32":
        directory = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
    elif sys.platform == "darwin":
        directory = Path.home() / "Library" / "Application Support"
    else:
        directory = Path.home() / ".local" / "share"
    return directory / "fftw.wisdom"


def _wisdom_state(fpath: Path) -> str:
    """Return the size and md5 of the wisdom file, or that it is
    missing.

    Printed before and after the sweep: the programs plan with
    ``FFTW_PATIENT``, which picks an algorithm by timing, and
    different algorithms round differently, so the reference bytes are
    a function of this file.  A changed wisdom is suspect number one
    behind a regeneration which is not bitwise.
    """
    if not fpath.is_file():
        return f"{fpath} is missing"
    return f"{fpath} {fpath.stat().st_size} B md5 {_md5_of_file(fpath)}"


# ---------------------------- Utilities ----------------------------- #


def _select(names) -> "tuple[Scenario, ...]":
    """Return the scenarios of ``names``, all of them by default."""
    if names is None:
        return SCENARIOS
    known = {scenario.name: scenario for scenario in SCENARIOS}
    unknown = [name for name in names if name not in known]
    if len(unknown) > 0:
        raise ValueError(f"unknown scenarios {unknown}: must be among {sorted(known)}")
    return tuple(known[name] for name in names)


def _kikuchipy_version() -> str:
    """Return the version of the kikuchipy the references come from."""
    import kikuchipy as kp

    return str(kp.__version__)


def _md5_of_file(fpath: Path) -> str:
    """Return the md5 sum of a file."""
    return hashlib.md5(Path(fpath).read_bytes()).hexdigest()


def _md5_of_array(array: np.ndarray) -> str:
    """Return the md5 sum of an array's bytes."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    main()
