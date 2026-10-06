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
"""

# TODO: implement the pre-flight checks, the pattern and master
# preparation, the five program runs with their acid bands, and the
# writing of the files with their provenance

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

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
    raise NotImplementedError


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
    raise NotImplementedError
    yield  # pragma: no cover


def _parse_args(argv: list[str] | None = None) -> dict:
    """Return the keyword arguments of :func:`main` from the command
    line options ``--out``, ``--run-dir`` and ``--scenarios``.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    main(**_parse_args())
