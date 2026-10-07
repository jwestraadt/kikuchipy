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

r"""Creation of the six shipped ``regression_hrosm_*.npz`` EMsoft
reference files next to this script.

Run once by a maintainer with EMsoftOO's programs; the output is
committed.  This script is excluded from the doctest job by
``--ignore-glob=src/kikuchipy/data/emsoft_hrosm/*.py`` and from
coverage by ``omit = ["src/kikuchipy/data/*/create_*.py"]``, both in
``pyproject.toml``.

The module is **import safe**: importing it looks up no environment
variable, probes no executable and opens no file, so that the shipped
regression tests can import :data:`SCENARIOS` and compare it with the
registry, the data directory and their own frozen table.  Everything
else happens inside :func:`main`.

Usage::

    python create_hrosm_reference.py [--out DIR] [--run-dir DIR]
        [--scenarios NAME [NAME ...]]

The programs ``EMDI``, ``EMFitOrientation``, ``EMgetOSM``, ``EMHROSM``
and ``EMsampleRFZ`` run one at a time, under the machine wide lock
file :data:`LOCK_NAME` in the temporary directory, each with its
working directory in a new run directory
``<EMdatapathname>/kikuchipy_hrosm/<YYYYmmdd-HHMMSS>/`` and every path
of every namelist below it, since the programs prepend
``EMdatapathname`` to every path and write some files unconditionally.

What :func:`main` does, in order:

1. Pre-flight checks, each aborting with a message naming it: the
   programs and libraries are present (their md5 sums are recorded);
   EMsoft's configuration is readable and its data and temporary
   directories exist; the nickel crystal file ``Ni.xtal`` in EMsoft's
   crystal folder is space group 225 with ``a = 0.35236`` nm; the
   ``nickel_ebsd_large`` patterns and the master pattern file
   ``ni_mc_mp_20kv.h5`` are cached, the latter with the md5
   :data:`MASTER_MD5`; no EMsoft program is running; every path of
   every namelist lies below the run directory; and a dry ``EMDI``
   run with a coarse dictionary reads the master pattern and the
   crystal and finds an OpenCL device.
2. Inputs: the patterns of ``nickel_ebsd_large`` with the static and
   then the dynamic background removed, written with the NORDIF
   writer; a copy of the master pattern file whose two ``xtalname``
   data sets are rewritten to ``Ni.xtal`` (the cached original names
   ``ni/ni.xtal``, which EMsoft cannot find); the average projection
   centre in EMsoft's convention, rounded through ``.6g``.
3. Runs: dictionary indexing (``EMDI``), whose top match must lie
   within :data:`TOP1_MEDIAN_MAX_DEG` of the stored orientations in
   the median, else it is rerun once with the patterns flipped
   (``flipy``); refinement (``EMFitOrientation``), median within
   :data:`REFINED_MEDIAN_MAX_DEG`; the orientation similarity maps of
   10 and 5 neighbours (``EMgetOSM``), the former bitwise equal to the
   indexing's own map; three ``EMHROSM`` runs; two ``EMsampleRFZ``
   misorientation balls of 6 and 20 steps.  The root of
   ``EMdatapathname`` is listed before the first program and the
   script aborts if a new entry appears there after any program.
4. Output: one uncompressed NPZ file per scenario with fixed zip
   member dates (byte stable), no pickled objects and strings as
   ``numpy.str_`` arrays, each holding the provenance of the run;
   nothing is written before every check has passed.  ``run.log`` in
   the run directory records every command, exit code, output and
   wall time.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Generator
import zipfile

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_logger = logging.getLogger(__name__)

# The machine wide lock shared with the test suite's EMsoft program
# lock: the programs share EMsoft's temporary directory and one GPU.
# Seconds to wait for it, the age of a lock file assumed to belong to a
# killed run (taken over), and the interval at which the holder
# refreshes the lock file's modification time.
LOCK_NAME = "kikuchipy-emsoft-program.lock"
LOCK_TIMEOUT = 3600.0
LOCK_STALE = 300.0
LOCK_HEARTBEAT = 30.0

# The md5 sum of the cached master pattern file ni_mc_mp_20kv.h5
MASTER_MD5 = "8b69c071a036ad3488d465093b67fe4d"

# The programs and libraries whose md5 sums every reference records
PROGRAMS = (
    "EMDI.exe",
    "EMFitOrientation.exe",
    "EMHROSM.exe",
    "EMgetOSM.exe",
    "EMsampleRFZ.exe",
)
LIBRARIES = ("EMsoftOOLib.dll", "EMOpenCLLib.dll")

# The run directory root below EMdatapathname
RUN_ROOT = "kikuchipy_hrosm"

SCENARIOS = (
    "large_di",
    "large_refined",
    "large_center",
    "large_center_dilate",
    "large_wat",
    "ball_n6",
)
"""The shipped references; the file of scenario ``name`` is
``regression_hrosm_<name>.npz``.

``large_di`` and ``large_refined`` hold the dictionary indexing and the
refinement of ``nickel_ebsd_large``; ``large_center``,
``large_center_dilate`` and ``large_wat`` the three EMHROSM runs
(centre pixel averaging without and with grain dilation, Watson
averaging); ``ball_n6`` the EMsampleRFZ misorientation ball of 6
steps.
"""

# The cached data sets the inputs come from
PATTERNS_KEY = "nickel_ebsd_large/patterns.h5"
MASTER_KEY = "ebsd_master_pattern/ni_mc_mp_20kv.h5"
MASTER_NAME = "ni_mc_mp_20kv.h5"

# The two data sets of the master pattern file naming the crystal
# file, and the name they get in the run's copy
XTALNAME_DATASETS = (
    "EMData/EBSDmaster/xtalname",
    "NMLparameters/MCCLNameList/xtalname",
)
XTAL_NAME = "Ni.xtal"
XTAL_SPACE_GROUP = 225
XTAL_LATTICE_NM = 0.35236

# The map of nickel_ebsd_large: rows, columns
MAP_SHAPE = (55, 75)

# Acceptance of the indexing against the orientations stored with the
# patterns: the median disorientation in degrees of the top match and
# of the refined orientation
TOP1_MEDIAN_MAX_DEG = 1.5
REFINED_MEDIAN_MAX_DEG = 0.5

# The dictionary batch sizes of EMDI (equal, multiples of 16, as the
# OpenCL kernel needs) and the dictionary resolution of the dry run
N_SINGLE = 32
PREFLIGHT_NCUBOCHORIC = 10

# Upper size of each written file in bytes
FILE_BUDGET_BYTES = 250_000

# The steps per semi-edge of EMHROSM's misorientation ball.  10, not
# 20: EMHROSM leaks one dictionary transpose buffer per dictionary
# batch, about 1 GB per re-indexed grain at 20 steps, more commit than
# a workstation has for the 31 grains of the larger runs
HROSM_NSAMPLES = 10

# The EMHROSM runs: averaging method and grain dilation
HROSM_RUNS = {
    "center": ("center", False),
    "center_dilate": ("center", True),
    "wat": ("averageWAT", False),
}

# The misorientation balls of EMsampleRFZ: steps, the 6 step one
# shipped; the centre is the Rodrigues vector of the unit axis
# (1, 2, 3) / sqrt(14) and tan(omega / 2) = 0.1
BALL_STEPS = (6, 20)
BALL_AXIS = (1.0, 2.0, 3.0)
BALL_TAN_HALF_ANGLE = 0.1

# Seconds one program may run before it is killed
PROGRAM_TIMEOUT = 7200.0

# Output of EMsoft's fatal error handler ("Progam" is EMsoft's
# spelling) and of the Fortran run time library's severe errors
_ABNORMAL_END_MARKERS = ("ended abnormally", "Fatal error", "forrtl: severe")

# Namelist keys holding a file name: their value must be 'undefined'
# or lie below the run directory, except the temporary file names,
# which EMsoft puts in its temporary directory
_PATH_KEY_SUFFIXES = ("file", "name", "prefix", "map", "tiff")
_TMPFILE_KEY = "tmpfile"

# The frozen arrays of each file: dtype name and shape, "n" standing
# for the number of grains
_HROSM_ARRAYS = {
    "nGrains": ("int32", ()),
    "grainID": ("int32", MAP_SHAPE),
    "npixels": ("int32", ("n",)),
    "grainROI": ("int32", ("n", 4)),
    "avor": ("float64", ("n", 4)),
    "kappa": ("float64", ("n",)),
    "kam": ("float32", MAP_SHAPE),
    "newOSM": ("float32", MAP_SHAPE),
    "newCI": ("float32", MAP_SHAPE),
    "newEuler": ("float32", MAP_SHAPE + (3,)),
}


def main(
    output_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    scenarios: list[str] | tuple[str, ...] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate the reference files and return their paths.

    Parameters
    ----------
    output_dir
        Directory to write ``regression_hrosm_<scenario>.npz`` into.
        The directory of this script, i.e. the shipped location, by
        default.
    bin_dir
        Directory of the EMsoftOO programs and libraries.  If not given
        (default), it is resolved from the ``KIKUCHIPY_EMSOFT_BIN``
        environment variable and the **machine wide program lock is
        taken for the whole run**.  A caller which passes the
        directory is assumed to hold that lock already, which is what
        the gated regeneration test does; taking it twice from one
        process would dead lock against itself.
    scenarios
        Names of the scenarios to write, all of :data:`SCENARIOS` by
        default.
    run_dir
        Run directory below ``<EMdatapathname>/kikuchipy_hrosm/``.  A
        new ``<YYYYmmdd-HHMMSS>`` directory by default.

    Returns
    -------
    written
        Path of every written file, keyed on the scenario name.

    Raises
    ------
    FileNotFoundError
        If a program, library, configuration or input file is missing.
    RuntimeError
        If a pre-flight check, a program run, an acid band or the
        output containment check fails; the message names it.
    TimeoutError
        If ``bin_dir`` is not given and the program lock is held by
        another process for longer than :data:`LOCK_TIMEOUT` seconds.
    ValueError
        If a scenario name is not one of :data:`SCENARIOS`.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    selected = _select(scenarios)
    if bin_dir is None:
        value = os.environ.get("KIKUCHIPY_EMSOFT_BIN")
        if not value:
            raise FileNotFoundError(
                "KIKUCHIPY_EMSOFT_BIN is not set; set it to the EMsoftOO Bin "
                "directory or pass bin_dir"
            )
        with _program_lock():
            return _Run(Path(value), output_dir, selected, run_dir).execute()
    return _Run(Path(bin_dir), output_dir, selected, run_dir).execute()


@contextmanager
def _program_lock() -> Generator[None, None, None]:
    """Hold the lock shared by every process running an EMsoft
    program.

    The same file and protocol as the test suite's EMsoft program
    lock: an exclusive create of :data:`LOCK_NAME` in the temporary
    directory, waiting at most :data:`LOCK_TIMEOUT` seconds, a lock
    file older than :data:`LOCK_STALE` seconds taken over, and the
    lock file's modification time refreshed every
    :data:`LOCK_HEARTBEAT` seconds by a daemon thread while held.
    """
    path = Path(tempfile.gettempdir()) / LOCK_NAME
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:  # it went away between the two calls
                continue
            if age > LOCK_STALE:
                try:
                    path.unlink(missing_ok=True)
                except OSError:  # still open by its holder on Windows
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Waited {LOCK_TIMEOUT} s for the EMsoft program lock "
                    f"{str(path)!r}. Delete it if no process is running an "
                    "EMsoft program"
                )
            time.sleep(0.05)

    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(LOCK_HEARTBEAT):
            try:
                os.utime(path)
            except OSError:
                pass

    thread = threading.Thread(
        target=refresh, name="kikuchipy-emsoft-lock-heartbeat", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> dict:
    """Return the keyword arguments of :func:`main` from the command
    line options ``--out``, ``--run-dir`` and ``--scenarios``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Create the shipped EMsoft HROSM reference files."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: the directory of this script)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="run directory below <EMdatapathname>/kikuchipy_hrosm/ "
        "(default: a new <YYYYmmdd-HHMMSS> directory)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=SCENARIOS,
        help="scenarios to write (default: all)",
    )
    args = parser.parse_args(argv)
    return {
        "output_dir": args.out,
        "run_dir": args.run_dir,
        "scenarios": args.scenarios,
    }


# ------------------------------ The run ----------------------------- #


class _Run:
    """One reference run: pre-flight checks, inputs, program runs,
    acid bands and output.

    Parameters
    ----------
    bin_dir
        Directory of the programs and libraries.
    output_dir
        Directory the files are written into.
    selected
        Scenario names to write.
    run_dir
        Run directory, or None for a new one.
    """

    def __init__(
        self,
        bin_dir: Path,
        output_dir: Path,
        selected: tuple[str, ...],
        run_dir: str | Path | None,
    ) -> None:
        self.bin_dir = Path(bin_dir)
        self.output_dir = Path(output_dir)
        self.selected = selected
        self.requested_run_dir = run_dir
        self.timings: dict[str, float] = {}
        self.namelists: dict[str, str] = {}
        self.log_path: Path | None = None
        self.root_before: set[str] = set()

    # ---------------------------- Driver ---------------------------- #

    def execute(self) -> dict[str, Path]:
        """Run everything and return the written files."""
        start = time.monotonic()
        self._check_programs()
        self._check_configuration()
        self._check_crystal()
        self._check_cached_inputs()
        self._check_no_program_running()
        self._make_run_dir()
        self._log(f"run directory {self.run_dir}")
        self._log(f"programs {self.bin_dir}")
        self._log(f"program md5 {self.program_md5}")
        self._log(f"EMsoft commit {self.emsoft_commit}")
        self._log(f"GPU {self.gpu_name}\n{_nvidia_smi()}")

        need_di = any(name.startswith("large_") for name in self.selected)
        hrosm_runs = [name for name in HROSM_RUNS if f"large_{name}" in self.selected]
        need_ball = "ball_n6" in self.selected

        self._prepare_inputs()
        namelists = self._all_namelists(hrosm_runs, need_di, need_ball)
        for name, text in namelists.items():
            self._check_namelist_paths(name, text)
        self._log("pre-flight: namelist paths below the run directory, ok")

        if need_di:
            self._preflight_indexing(namelists["preflight"])
            self._index(namelists["EMDI"])
            self._refine(namelists["EMFitOrientation"])
            self._similarity_maps(namelists["EMgetOSM"])
        for name in hrosm_runs:
            self._program("EMHROSM", namelists[f"EMHROSM_{name}"], tag=name)
        if need_ball:
            for n_steps in BALL_STEPS:
                self._program(
                    "EMsampleRFZ",
                    namelists[f"EMsampleRFZ_n{n_steps}"],
                    tag=f"n{n_steps}",
                )

        references = {name: self._reference(name, hrosm_runs) for name in self.selected}
        blobs = {}
        for name, arrays in references.items():
            blob = _npz_bytes(arrays)
            if len(blob) >= FILE_BUDGET_BYTES:
                raise RuntimeError(
                    f"output size: regression_hrosm_{name}.npz would be "
                    f"{len(blob)} B, not below {FILE_BUDGET_BYTES} B"
                )
            blobs[name] = blob

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        total = 0
        for name, blob in blobs.items():
            fpath = self.output_dir / f"regression_hrosm_{name}.npz"
            fpath.write_bytes(blob)
            written[name] = fpath
            total += len(blob)
            self._log(
                f'    "emsoft_hrosm/{fpath.name}":'.ljust(67)
                + f' "md5:{_md5_of_bytes(blob)}",  # {len(blob)} B'
            )
        self._log(f"total {total} B over {len(written)} files")
        for name, seconds in self.timings.items():
            self._log(f"wall time {name}: {seconds:.1f} s")
        self._log(f"wall time total: {time.monotonic() - start:.1f} s")
        return written

    # ---------------------- Pre-flight checks ----------------------- #

    def _check_programs(self) -> None:
        """Check the programs and libraries exist and record their md5
        sums, the EMsoft commit and the GPU.
        """
        missing = [
            name for name in PROGRAMS + LIBRARIES if not (self.bin_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"pre-flight check 'programs': {missing} missing from {self.bin_dir}"
            )
        pairs = [
            f"{name}={_md5_of_file(self.bin_dir / name)}"
            for name in PROGRAMS + LIBRARIES
        ]
        self.program_md5 = ";".join(sorted(pairs))
        library = (self.bin_dir / "EMsoftOOLib.dll").read_bytes()
        match = re.search(rb"\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ([0-9a-f]{7,40})", library)
        self.emsoft_commit = match.group(1).decode() if match else "unknown"
        self.gpu_name = _gpu_name()

    def _check_configuration(self) -> None:
        """Check EMsoft's configuration file and its directories."""
        import json

        fpath = Path.home() / ".config" / "EMsoft" / "EMsoftConfig.json"
        try:
            config = json.loads(fpath.read_text())
        except (OSError, ValueError) as error:
            raise FileNotFoundError(
                f"pre-flight check 'configuration': cannot read {fpath}: {error}"
            ) from error
        for key in ("EMdatapathname", "EMtmppathname", "EMXtalFolderpathname"):
            value = config.get(key)
            if not value or not Path(value).is_dir():
                raise FileNotFoundError(
                    f"pre-flight check 'configuration': {key} {value!r} of "
                    f"{fpath} is not an existing directory"
                )
        self.data_root = Path(config["EMdatapathname"])
        self.xtal_dir = Path(config["EMXtalFolderpathname"])
        self.template_dir = Path(config.get("EMsoftpathname", "")) / (
            "NamelistTemplates"
        )
        if not self.template_dir.is_dir():
            raise FileNotFoundError(
                "pre-flight check 'configuration': no NamelistTemplates "
                f"directory {self.template_dir}"
            )

    def _check_crystal(self) -> None:
        """Check the nickel crystal file EMsoft reads."""
        import h5py
        import numpy as np

        fpath = self.xtal_dir / XTAL_NAME
        if not fpath.is_file():
            raise FileNotFoundError(f"pre-flight check 'crystal': no {fpath}")
        with h5py.File(fpath, "r") as f:
            space_group = int(np.ravel(f["CrystalData/SpaceGroupNumber"][()])[0])
            lattice = float(np.ravel(f["CrystalData/LatticeParameters"][()])[0])
        if space_group != XTAL_SPACE_GROUP or abs(lattice - XTAL_LATTICE_NM) > 1e-6:
            raise RuntimeError(
                f"pre-flight check 'crystal': {fpath} has space group "
                f"{space_group} and a = {lattice} nm, not {XTAL_SPACE_GROUP} "
                f"and {XTAL_LATTICE_NM} nm"
            )

    def _check_cached_inputs(self) -> None:
        """Check the patterns and the master pattern file are cached
        and the master has the recorded md5 sum.
        """
        from kikuchipy.data._data import Dataset

        patterns = Dataset(PATTERNS_KEY)
        master = Dataset(MASTER_KEY)
        for dataset in (patterns, master):
            if not dataset.is_in_cache:
                raise FileNotFoundError(
                    f"pre-flight check 'inputs': {dataset.file_relpath} is not "
                    "cached; fetch it with kikuchipy.data first"
                )
        self.master_path = Path(master.fetch_file_path())
        master_md5 = _md5_of_file(self.master_path)
        if master_md5 != MASTER_MD5:
            raise RuntimeError(
                f"pre-flight check 'inputs': {self.master_path} has md5 "
                f"{master_md5}, not {MASTER_MD5}"
            )

    def _check_no_program_running(self) -> None:
        """Check no EMsoft program of the run is running."""
        if sys.platform != "win32":
            return
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True
        )
        running = {
            line.split(",")[0].strip('"').lower()
            for line in result.stdout.splitlines()
            if line
        }
        busy = sorted(name for name in PROGRAMS if name.lower() in running)
        if busy:
            raise RuntimeError(f"pre-flight check 'no program running': {busy} run")

    def _check_namelist_paths(self, name: str, text: str) -> None:
        """Check every file name of a namelist lies below the run
        directory, is 'undefined' or is a temporary file name.
        """
        for key, value in re.findall(r"(\w+)\s*=\s*'([^']*)'", text):
            lower = key.lower()
            if lower.endswith(_TMPFILE_KEY):
                if "/" in value or "\\" in value:
                    raise RuntimeError(
                        f"pre-flight check 'namelist paths': {name} {key} "
                        f"{value!r} is not a bare file name"
                    )
            elif lower.endswith(_PATH_KEY_SUFFIXES):
                if value != "undefined" and not value.startswith(f"{self.relative}/"):
                    raise RuntimeError(
                        f"pre-flight check 'namelist paths': {name} {key} "
                        f"{value!r} is not below {self.relative}/"
                    )

    # --------------------------- Inputs ---------------------------- #

    def _make_run_dir(self) -> None:
        """Create the run directory, then list the data root."""
        root = self.data_root / RUN_ROOT
        if self.requested_run_dir is None:
            while True:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                run_dir = root / stamp
                try:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    time.sleep(1.0)
        else:
            run_dir = Path(self.requested_run_dir)
            if not run_dir.is_absolute():
                run_dir = self.data_root / run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = run_dir.resolve()
        try:
            relative = run_dir.relative_to(root.resolve())
        except ValueError:
            raise RuntimeError(f"run directory {run_dir} is not below {root}") from None
        if len(relative.parts) == 0:
            raise RuntimeError(f"run directory {run_dir} is {root} itself")
        self.run_dir = run_dir
        self.relative = f"{RUN_ROOT}/{relative.as_posix()}"
        self.log_path = run_dir / "run.log"
        self.root_before = _entries(self.data_root)

    def _prepare_inputs(self) -> None:
        """Write the patterns, copy and patch the master pattern file
        and compute the projection centre.
        """
        import h5py
        import numpy as np

        import kikuchipy as kp
        from kikuchipy.io.plugins.nordif._api import file_writer

        signal = kp.data.nickel_ebsd_large(allow_download=False)
        self.xmap = signal.xmap.deepcopy()
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        if signal.data.dtype != np.uint8:
            raise RuntimeError(
                f"inputs: the patterns are {signal.data.dtype}, not uint8"
            )
        if signal.axes_manager.navigation_shape[::-1] != MAP_SHAPE:
            raise RuntimeError(
                f"inputs: the map is {signal.axes_manager.navigation_shape[::-1]}"
                f", not {MAP_SHAPE}"
            )
        (self.run_dir / "patterns").mkdir(exist_ok=True)
        pattern_path = self.run_dir / "patterns" / "Pattern.dat"
        file_writer(str(pattern_path), signal)
        self.patterns_md5 = _md5_of_file(pattern_path)
        self.pattern_shape = signal.axes_manager.signal_shape[::-1]

        master_copy = self.run_dir / MASTER_NAME
        shutil.copyfile(self.master_path, master_copy)
        with h5py.File(master_copy, "r+") as f:
            for key in XTALNAME_DATASETS:
                dataset = f[key]
                before = (dataset.dtype, dataset.shape)
                dataset[0] = XTAL_NAME.encode("ascii")
                if (dataset.dtype, dataset.shape) != before:
                    raise RuntimeError(f"inputs: rewriting {key} changed it")
        with h5py.File(master_copy, "r") as f:
            for key in XTALNAME_DATASETS:
                value = f[key][0]
                value = value.decode() if isinstance(value, bytes) else str(value)
                if value != XTAL_NAME:
                    raise RuntimeError(f"inputs: {key} of the copy is {value!r}")
        if _md5_of_file(self.master_path) != MASTER_MD5:
            raise RuntimeError("inputs: the cached master pattern file changed")
        self.master_run_md5 = _md5_of_file(master_copy)

        det = signal.detector
        det.pc = det.pc_average
        pc = det.pc_emsoft()[0]
        xpc, ypc = pc[:2] / det.binning
        values = (xpc, ypc, pc[2], det.px_size * det.binning)
        self.pc_text = [f"{value:.6g}" for value in values]
        self.pc = np.array([float(text) for text in self.pc_text])
        self._log(
            f"inputs: patterns md5 {self.patterns_md5}, master copy md5 "
            f"{self.master_run_md5}, pc (xpc, ypc, L, delta) {self.pc_text}"
        )

    # -------------------------- Namelists --------------------------- #

    def _all_namelists(
        self, hrosm_runs: list[str], need_di: bool, need_ball: bool
    ) -> dict[str, str]:
        """Return every namelist text of the run, keyed on its name."""
        r = self.relative
        namelists = {}
        if need_di:
            namelists["preflight"] = self._emdi_namelist(
                flipy=False,
                ncubochoric=PREFLIGHT_NCUBOCHORIC,
                datafile=f"{r}/preflight/dp.h5",
            )
            namelists["EMDI"] = self._emdi_namelist(flipy=False)
            namelists["EMFitOrientation"] = _namelist(
                self.template_dir / "EMFitOrientation.template",
                {
                    "nthreads": "20",
                    "dotproductfile": f"'{r}/dp.h5'",
                    "newdotproductfile": f"'{r}/dp-refined.h5'",
                    "usemasterpatternfile": f"'{r}/{MASTER_NAME}'",
                    "ctffile": "'undefined'",
                    "angfile": f"'{r}/dp-refined.ang'",
                    "tmpfile": "'EMFitOrientation_tmp.data'",
                    "inRAM": ".FALSE.",
                    "matchdepth": "1",
                    "method": "'FIT'",
                    "niter": "1",
                    "nmis": "1",
                    "step": "0.03",
                    "PCcorrection": "'off'",
                },
            )
            namelists["EMgetOSM"] = _namelist(
                self.template_dir / "EMgetOSM.template",
                {
                    "nmatch": "10 5 0 0 0",
                    # a logical, quoted in the template, which the
                    # program cannot read
                    "dpweighted": ".FALSE.",
                    "dotproductfile": f"'{r}/dp-refined.h5'",
                    "tiffname": f"'{r}/osm_'",
                },
            )
        for name in hrosm_runs:
            orav, dilate = HROSM_RUNS[name]
            namelists[f"EMHROSM_{name}"] = _namelist(
                self.template_dir / "EMHROSM.template",
                {
                    "gangle": "5.0",
                    "misorang": "5.0",
                    "nsamples": str(HROSM_NSAMPLES),
                    "nosm": "10",
                    "dilate": ".TRUE." if dilate else ".FALSE.",
                    "orav": f"'{orav}'",
                    "numEM": "25",
                    "numIter": "40",
                    "dpfile": f"'{r}/dp-refined.h5'",
                    "OSMfile": f"'{r}/hrosm_{name}.h5'",
                    "OSMtiff": f"'{r}/hrosm_{name}.tiff'",
                    "IPFmap": "'undefined'",
                    "angfile": "'undefined'",
                    "ctffile": "'undefined'",
                    "maxRAMmem": "1.0",
                },
            )
        if need_ball:
            norm = sum(v * v for v in BALL_AXIS) ** 0.5
            axis = ", ".join(f"{v / norm:.16e}" for v in BALL_AXIS)
            for n_steps in BALL_STEPS:
                stem = f"{r}/ball_n{n_steps}"
                namelists[f"EMsampleRFZ_n{n_steps}"] = _namelist(
                    self.template_dir / "EMsampleRFZ.template",
                    {
                        "samplemode": "'MIS'",
                        "pgnum": "32",
                        "maxmisor": "5.0",
                        "nsteps": str(n_steps),
                        "rodrigues": f"{axis}, {BALL_TAN_HALF_ANGLE:.16e}",
                        "quoutname": f"'{stem}_qu.txt'",
                        "euoutname": f"'{stem}_eu.txt'",
                        "rooutname": f"'{stem}_ro.txt'",
                    },
                )
        return namelists

    def _emdi_namelist(
        self,
        flipy: bool,
        ncubochoric: int = 100,
        datafile: str | None = None,
    ) -> str:
        """Return the namelist text of a dictionary indexing run."""
        r = self.relative
        xpc, ypc, L, delta = self.pc_text
        ny, nx = self.pattern_shape
        return _namelist(
            self.template_dir / "EMDI.template",
            {
                "indexingmode": "'dynamic'",
                "ipf_wd": str(MAP_SHAPE[1]),
                "ipf_ht": str(MAP_SHAPE[0]),
                "ROI": "0 0 0 0",
                "nnk": "20",
                "nosm": "10",
                "nism": "5",
                "maskpattern": "'y'",
                "maskradius": "29",
                "hipassw": "0.05",
                "nregions": "4",
                "ncubochoric": str(ncubochoric),
                "scalingmode": "'not'",
                "L": L,
                "thetac": "0.0",
                "delta": delta,
                "xpc": xpc,
                "ypc": ypc,
                "exptnumsx": str(nx),
                "exptnumsy": str(ny),
                "binning": "1",
                "numsx": str(nx),
                "numsy": str(ny),
                "omega": "0.0",
                "energymin": "15.0",
                "energymax": "20.0",
                "exptfile": f"'{r}/patterns/Pattern.dat'",
                "inputtype": "'NORDIF'",
                "flipy": ".TRUE." if flipy else ".FALSE.",
                "tmpfile": "'EMEBSDDict_tmp.data'",
                "keeptmpfile": "'n'",
                "datafile": f"'{datafile or f'{r}/dp.h5'}'",
                "ctffile": "'undefined'",
                "angfile": "'undefined'",
                "masterfile": f"'{r}/{MASTER_NAME}'",
                "numdictsingle": str(N_SINGLE),
                "numexptsingle": str(N_SINGLE),
                "nthreads": "20",
                "platid": "1",
                "devid": "1",
            },
        )

    # ---------------------------- Programs -------------------------- #

    def _program(
        self, name: str, text: str, tag: str = "", cwd: Path | None = None
    ) -> None:
        """Run one program on a namelist text, log it and check its
        exit code and the data root.
        """
        cwd = self.run_dir if cwd is None else cwd
        label = f"{name} {tag}".strip()
        nml = cwd / f"{name}.nml"
        nml.write_text(text, newline="\n")
        command = [str(self.bin_dir / f"{name}.exe"), nml.name]
        _logger.info("running %s", label)
        start = time.monotonic()
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROGRAM_TIMEOUT,
        )
        seconds = time.monotonic() - start
        self.timings[label] = self.timings.get(label, 0.0) + seconds
        self.namelists[label] = text
        # keep the namelist of each run beside the log
        (cwd / f"{name}{'_' + tag if tag else ''}.nml.txt").write_text(
            text, newline="\n"
        )
        self._log(
            f"=== {label}\ncommand: {command} (cwd {cwd})\nexit code: "
            f"{result.returncode}\nwall time: {seconds:.1f} s\n--- stdout\n"
            f"{result.stdout}\n--- stderr\n{result.stderr}"
        )
        new = _entries(self.data_root) - self.root_before
        if new:
            raise RuntimeError(
                f"output containment: {label} created {sorted(new)} in {self.data_root}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} exited with code {result.returncode}; see {self.log_path}"
            )
        # EMsoft's fatal error handler, e.g. after a failed memory
        # allocation, ends the program with exit code 0
        output = result.stdout + result.stderr
        for marker in _ABNORMAL_END_MARKERS:
            if marker in output:
                raise RuntimeError(
                    f"{label} printed {marker!r} although it exited with code "
                    f"0; see {self.log_path}"
                )

    def _preflight_indexing(self, text: str) -> None:
        """Run a coarse dictionary indexing and check its top-match
        list.
        """
        import h5py

        directory = self.run_dir / "preflight"
        directory.mkdir(exist_ok=True)
        try:
            self._program("EMDI", text, tag="preflight", cwd=directory)
        except RuntimeError as error:
            raise RuntimeError(f"pre-flight check 'dry indexing': {error}") from error
        with h5py.File(directory / "dp.h5", "r") as f:
            shape = f["Scan 1/EBSD/Data/TopMatchIndices"].shape
        n_points = MAP_SHAPE[0] * MAP_SHAPE[1]
        if not (
            len(shape) == 2
            and shape[1] == 20
            and shape[0] % N_SINGLE == 0
            and shape[0] >= n_points
        ):
            raise RuntimeError(
                f"pre-flight check 'dry indexing': TopMatchIndices is {shape}"
            )
        self._log(f"pre-flight: dry indexing ok, TopMatchIndices {shape}")

    def _index(self, text: str) -> None:
        """Run the dictionary indexing; rerun once with the patterns
        flipped if the top match misses the stored orientations.
        """
        from kikuchipy.indexing._hrosm._emsoft_file import (
            read_emsoft_dot_product_file,
        )

        self._program("EMDI", text)
        euler = read_emsoft_dot_product_file(self.run_dir / "dp.h5")["EulerAngles"]
        median = self._median_disorientation(euler)
        self._log(f"acid band: top-1 median disorientation {median:.4f} deg")
        self.flipy = False
        if median > TOP1_MEDIAN_MAX_DEG:
            text = self._emdi_namelist(flipy=True)
            self._check_namelist_paths("EMDI flipy", text)
            self._program("EMDI", text, tag="flipy")
            data = read_emsoft_dot_product_file(self.run_dir / "dp.h5")
            median = self._median_disorientation(data["EulerAngles"])
            self._log(
                f"acid band: top-1 median disorientation with flipy {median:.4f} deg"
            )
            if median > TOP1_MEDIAN_MAX_DEG:
                raise RuntimeError(
                    f"acid band: the top-1 median disorientation is {median:.4f}"
                    f" deg with and without flipy, above {TOP1_MEDIAN_MAX_DEG}"
                )
            self.flipy = True
        self.top1_median = median
        self.emdi_namelist = text

    def _refine(self, text: str) -> None:
        """Run the refinement and check its orientations."""
        from kikuchipy.indexing._hrosm._emsoft_file import (
            read_emsoft_dot_product_file,
        )

        self._program("EMFitOrientation", text)
        data = read_emsoft_dot_product_file(self.run_dir / "dp-refined.h5")
        median = self._median_disorientation(data["RefinedEulerAngles"])
        self._log(f"acid band: refined median disorientation {median:.4f} deg")
        if median > REFINED_MEDIAN_MAX_DEG:
            raise RuntimeError(
                f"acid band: the refined median disorientation is {median:.4f}"
                f" deg, above {REFINED_MEDIAN_MAX_DEG}"
            )
        self.refined_median = median

    def _similarity_maps(self, text: str) -> None:
        """Compute the similarity maps of 10 and 5 neighbours and check
        the former equals the indexing's own map bitwise.
        """
        import numpy as np

        from kikuchipy.indexing._hrosm._emsoft_file import (
            read_emsoft_dot_product_file,
        )

        self._program("EMgetOSM", text)
        data = read_emsoft_dot_product_file(self.run_dir / "dp-refined.h5")
        if not np.array_equal(data["OSM_10"], data["OSM"]):
            n = int(np.count_nonzero(data["OSM_10"] != data["OSM"]))
            raise RuntimeError(f"self-check: OSM_10 differs from OSM in {n} points")
        self._log("self-check: OSM_10 == OSM bitwise, ok")

    def _median_disorientation(self, euler: np.ndarray) -> float:
        """Return the median m-3m disorientation in degrees between
        EMsoft's Euler angles and the stored orientations.
        """
        import numpy as np
        from orix.quaternion import Rotation
        from orix.quaternion.symmetry import Oh

        ours = Rotation.from_euler(np.asarray(euler, dtype=np.float64)).data
        theirs = self.xmap.rotations.data.reshape(-1, 4)
        operators = Oh.proper_subgroup.data
        a0, a1, a2, a3 = np.moveaxis(ours, -1, 0)
        best = np.zeros(len(ours))
        for s0, s1, s2, s3 in operators:
            # the quaternion product S * a, dotted with b
            p = np.stack(
                [
                    s0 * a0 - s1 * a1 - s2 * a2 - s3 * a3,
                    s0 * a1 + s1 * a0 + s2 * a3 - s3 * a2,
                    s0 * a2 - s1 * a3 + s2 * a0 + s3 * a1,
                    s0 * a3 + s1 * a2 - s2 * a1 + s3 * a0,
                ],
                axis=-1,
            )
            best = np.maximum(best, np.abs(np.sum(p * theirs, axis=-1)))
        angles = np.rad2deg(2 * np.arccos(np.clip(best, 0, 1)))
        return float(np.median(angles))

    # ---------------------------- Output ---------------------------- #

    def _reference(self, name: str, hrosm_runs: list[str]) -> dict:
        """Return the arrays and provenance of one file."""
        import numpy as np

        from kikuchipy.indexing._hrosm._emsoft_file import (
            read_emsoft_dot_product_file,
            read_emsoft_hrosm_file,
        )

        n_points = MAP_SHAPE[0] * MAP_SHAPE[1]
        if name == "large_di":
            dp = read_emsoft_dot_product_file(self.run_dir / "dp.h5")
            refined = read_emsoft_dot_product_file(self.run_dir / "dp-refined.h5")
            arrays = {
                "TopMatchIndices": dp["TopMatchIndices"][:, :10],
                "KAM": dp["KAM"],
                "OSM": dp["OSM"],
                "OSM_05": refined["OSM_05"],
                "CI": dp["CI"],
            }
            expected = {
                "TopMatchIndices": ("int32", (n_points, 10)),
                "KAM": ("float32", MAP_SHAPE),
                "OSM": ("float32", MAP_SHAPE),
                "OSM_05": ("float32", MAP_SHAPE),
                "CI": ("float32", (n_points,)),
            }
            if arrays["TopMatchIndices"].min() < 1:
                raise RuntimeError("output: TopMatchIndices is not 1-based")
        elif name == "large_refined":
            refined = read_emsoft_dot_product_file(self.run_dir / "dp-refined.h5")
            arrays = {
                key: refined[key]
                for key in ("RefinedEulerAngles", "RefinedDotProducts", "EulerAngles")
            }
            expected = {
                "RefinedEulerAngles": ("float32", (n_points, 3)),
                "RefinedDotProducts": ("float32", (n_points,)),
                "EulerAngles": ("float32", (n_points, 3)),
            }
        elif name == "ball_n6":
            stem = self.run_dir / f"ball_n{BALL_STEPS[0]}"
            arrays = {
                "qu": _read_orientation_file(Path(f"{stem}_qu.txt")),
                "ro": _read_orientation_file(Path(f"{stem}_ro.txt")),
            }
            n = (2 * BALL_STEPS[0] + 1) ** 3
            expected = {"qu": ("float64", (n, 4)), "ro": ("float64", (n, 4))}
        else:
            scenario = name[len("large_") :]
            data = read_emsoft_hrosm_file(self.run_dir / f"hrosm_{scenario}.h5")
            arrays = {key: data[key] for key in _HROSM_ARRAYS}
            n_grains = int(data["nGrains"])
            expected = {
                key: (dtype, tuple(n_grains if s == "n" else s for s in shape))
                for key, (dtype, shape) in _HROSM_ARRAYS.items()
            }
            orav, dilate = HROSM_RUNS[scenario]
            if data["orav"] != orav or data["dilate"] is not dilate:
                raise RuntimeError(
                    f"output: {name} ran orav {data['orav']!r}, dilate {data['dilate']}"
                )
            arrays["hrosm_namelist"] = np.asarray(
                self.namelists[f"EMHROSM {scenario}"], dtype=np.str_
            )

        for key, (dtype, shape) in expected.items():
            array = np.asarray(arrays[key])
            if array.dtype != np.dtype(dtype) or array.shape != shape:
                raise RuntimeError(
                    f"output: {name} {key} is {array.dtype}{array.shape}, not "
                    f"{dtype}{shape}"
                )
            # np.array, not np.ascontiguousarray, which turns a 0-d
            # array into shape (1,)
            arrays[key] = np.array(array, order="C")

        arrays.update(self._provenance())
        return arrays

    def _provenance(self) -> dict:
        """Return the provenance arrays every file holds."""
        import numpy as np

        import kikuchipy as kp
        from kikuchipy.indexing._hrosm._emsoft_file import (
            read_emsoft_dot_product_file,
        )

        dp_path = self.run_dir / "dp.h5"
        if dp_path.is_file():
            version = read_emsoft_dot_product_file(dp_path)["version"]
        else:
            version = "unknown"
        namelist = getattr(self, "emdi_namelist", "")
        return {
            "program_md5": np.asarray(self.program_md5, dtype=np.str_),
            "emsoft_version": np.asarray(version, dtype=np.str_),
            "emsoft_commit": np.asarray(self.emsoft_commit, dtype=np.str_),
            "master_md5": np.asarray(MASTER_MD5, dtype=np.str_),
            "master_run_md5": np.asarray(self.master_run_md5, dtype=np.str_),
            "patterns_md5": np.asarray(self.patterns_md5, dtype=np.str_),
            "pc": np.asarray(self.pc, dtype=np.float64),
            "namelist": np.asarray(namelist, dtype=np.str_),
            "gpu_name": np.asarray(self.gpu_name, dtype=np.str_),
            "numdictsingle": np.asarray(N_SINGLE, dtype=np.int64),
            "numexptsingle": np.asarray(N_SINGLE, dtype=np.int64),
            "kikuchipy_version": np.asarray(str(kp.__version__), dtype=np.str_),
        }

    def _log(self, message: str) -> None:
        """Append a message to the run log and the logger."""
        _logger.info(message)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(message.rstrip("\n") + "\n")


# ---------------------------- Utilities ----------------------------- #


def _select(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Return the scenario names of ``names``, all of them by
    default, in the order of :data:`SCENARIOS`.
    """
    if names is None:
        return SCENARIOS
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}: must be among {SCENARIOS}")
    return tuple(name for name in SCENARIOS if name in names)


def _namelist(template: Path, values: dict[str, str]) -> str:
    """Return a namelist text from an EMsoftOO template with the given
    keys set, comments and blank lines removed.

    Keys absent from the template are added before the closing ``/``.
    """
    lines = template.read_text().splitlines()
    out = []
    done = set()
    lower = {key.lower(): key for key in values}
    for line in lines:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("!"):
            continue
        if stripped == "/":
            for key, value in values.items():
                if key not in done:
                    out.append(f" {key} = {value},")
                    done.add(key)
            out.append(" /")
            break
        match = re.match(r"\s*(\w+)\s*=", line)
        if match and match.group(1).lower() in lower:
            key = lower[match.group(1).lower()]
            out.append(f" {match.group(1)} = {values[key]},")
            done.add(key)
        else:
            out.append(line.rstrip())
    return "\n".join(out) + "\n"


def _read_orientation_file(fpath: Path) -> np.ndarray:
    """Return the rows of an EMsoft orientation text file (first line
    the representation, second line the count).
    """
    import numpy as np

    lines = fpath.read_text().splitlines()
    count = int(lines[1])
    data = np.loadtxt(lines[2:], dtype=np.float64, ndmin=2)
    if data.shape[0] != count:
        raise RuntimeError(f"output: {fpath} holds {data.shape[0]} of {count} rows")
    return data


def _npz_bytes(arrays: dict) -> bytes:
    """Return the bytes of an uncompressed NPZ file of the arrays, its
    zip members dated 1980-01-01 so that equal arrays give equal bytes.
    """
    import numpy as np

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def _entries(directory: Path) -> set[str]:
    """Return the names of a directory's entries."""
    return {path.name for path in directory.iterdir()}


def _gpu_name() -> str:
    """Return the name of the first NVIDIA GPU, or "unknown"."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if result.returncode == 0 and lines else "unknown"


def _nvidia_smi() -> str:
    """Return the output of ``nvidia-smi``, or why there is none."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi failed: {error}"
    return result.stdout


def _md5_of_file(fpath: Path) -> str:
    """Return the md5 sum of a file, read in chunks."""
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _md5_of_bytes(data: bytes) -> str:
    """Return the md5 sum of bytes."""
    return hashlib.md5(data).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(**_parse_args())
