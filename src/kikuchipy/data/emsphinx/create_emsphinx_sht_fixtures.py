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

r"""Creation of the two shipped EMSphInx *.sht master pattern files
next to this script and the one-off acceptance of the 25 synthetic
files of ``kikuchipy.data._dummy_files.emsphinx_sht``.

Run once; the output is committed.  This script is excluded from the
doctest job by ``--ignore-glob=src/kikuchipy/data/emsphinx/*.py`` in
``pyproject.toml``.

The two files are the only genuinely external oracles of the *.sht
codec, since they come from EMSphInx' own ``mp2sht`` program and not
from kikuchipy:

- ``ni_small_20kv_bw384.sht`` from the in-package
  ``emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5``
  (401 px, one energy bin)
- ``ni_20kv_bw384.sht`` from the cached
  ``ebsd_master_pattern/ni_mc_mp_20kv.h5`` (1001 px, 16 energy bins),
  which ``kikuchipy.data.ebsd_master_pattern("ni")`` downloads

Both are 74 828 B.

Recorded blocker: the built ``mp2sht.exe`` links HDF5 1.8.20 *without*
the deflate filter and fails on the gzip compressed in-package file
with ``H5Z_pipeline(): required filter 'deflate' is not registered``,
so the in-package file is first repacked uncompressed with h5py into a
temporary file which is *not* committed.

The equivalent shell commands, with ``EMSPHINX`` the EMSphInx checkout
and ``TMP`` a scratch directory::

    uv run python src/kikuchipy/data/emsphinx/create_emsphinx_sht_fixtures.py
    # which is
    #   (a) repack the in-package master without compression
    #   (b) $EMSPHINX/build/Release/mp2sht.exe $TMP/ni_small_uncompressed.h5 \
    #           src/kikuchipy/data/emsphinx/ni_small_20kv_bw384.sht
    #       $EMSPHINX/build/Release/mp2sht.exe $CACHE/ni_mc_mp_20kv.h5 \
    #           src/kikuchipy/data/emsphinx/ni_20kv_bw384.sht
    #   (c) write the 25 synthetic files and accept each with
    #       $EMSPHINX/build/Release/sht2png.exe $TMP/synthetic/sg001_bw16.sht \
    #           $TMP/sg001.png
    #   (d) md5sum src/kikuchipy/data/emsphinx/*.sht -> _registry.py

Neither executable writes template files to the current directory.
"""

import argparse
from pathlib import Path

DEFAULT_EMSPHINX_DIR = Path("C:/Users/westraadt.1/Repos/EMSphInx")
"""Default EMSphInx checkout, this machine."""

DEFAULT_CACHE_DIR = Path("C:/Users/westraadt.1/AppData/Local/kikuchipy/kikuchipy/Cache")
"""Default kikuchipy data cache, this machine."""

DEFAULT_OUTPUT_DIR = Path(__file__).parent
"""Directory the two *.sht files are written to, this one."""


def repack_uncompressed(source: Path, destination: Path) -> Path:
    """Copy an HDF5 file without any compression filter.

    Parameters
    ----------
    source
        Path to the gzip compressed file.
    destination
        Path to write the uncompressed copy to.

    Returns
    -------
    destination
        The path written.

    Notes
    -----
    Needed because ``mp2sht.exe`` links an HDF5 build without the
    deflate filter.  The repack of the in-package Ni master is
    1 199 136 B with md5 ``b58bece63152a9b5e4c53f5e8899fef7`` and is
    not committed.
    """
    raise NotImplementedError


def run_mp2sht(emsphinx_dir: Path, source: Path, destination: Path) -> Path:
    """Run EMSphInx' ``mp2sht`` on an EMsoft master pattern.

    Parameters
    ----------
    emsphinx_dir
        EMSphInx checkout with ``build/Release/mp2sht.exe`` (or
        ``build/Release/mp2sht`` on POSIX).
    source
        Path to the uncompressed EMsoft HDF5 master pattern.
    destination
        Path to write the *.sht file to.

    Returns
    -------
    destination
        The path written.

    Raises
    ------
    FileNotFoundError
        If the executable or the source is missing.
    RuntimeError
        If ``mp2sht`` exits non-zero.
    """
    raise NotImplementedError


def run_sht2png(emsphinx_dir: Path, source: Path, destination: Path) -> str:
    """Run EMSphInx' ``sht2png`` on a *.sht file.

    Parameters
    ----------
    emsphinx_dir
        EMSphInx checkout with ``build/Release/sht2png.exe``.
    source
        Path to the *.sht file.
    destination
        Path to write the square Legendre PNG to.

    Returns
    -------
    stdout
        What ``sht2png`` printed, which holds the line
        ``master pattern composed from 1 crystals with effective sg#
        N`` and every header field.

    Raises
    ------
    RuntimeError
        If ``sht2png`` exits non-zero, i.e. it does not accept the
        file.
    """
    raise NotImplementedError


def accept_synthetic_files(emsphinx_dir: Path, directory: Path) -> dict[int, str]:
    """Write the 25 synthetic *.sht files and let ``sht2png`` accept
    them.

    Parameters
    ----------
    emsphinx_dir
        EMSphInx checkout.
    directory
        Scratch directory to write the files and the PNGs into.

    Returns
    -------
    md5_sums
        The md5 sum of every accepted file keyed on its space group,
        to be copied into ``validation.md`` and pinned in
        ``tests/test_indexing/test_spherical_sht_file.py``.

    Raises
    ------
    RuntimeError
        If ``sht2png`` refuses a file or does not print the expected
        effective space group.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Create the shipped fixtures and accept the synthetic ones.

    Parameters
    ----------
    argv
        Command line arguments, :data:`sys.argv` by default.

    Returns
    -------
    exit_code
        0 on success.
    """
    raise NotImplementedError


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Return the parsed command line arguments.

    Parameters
    ----------
    argv
        Command line arguments.

    Returns
    -------
    arguments
        Parsed arguments ``emsphinx_dir``, ``cache_dir``,
        ``in_package_master``, ``full_master``, ``output_dir`` and
        ``tmp_dir``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create the two shipped EMSphInx .sht fixtures with "
            "mp2sht and accept the 25 synthetic ones with sht2png"
        )
    )
    parser.add_argument(
        "--emsphinx-dir",
        type=Path,
        default=DEFAULT_EMSPHINX_DIR,
        help="EMSphInx checkout with build/Release/{mp2sht,sht2png}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="kikuchipy data cache holding the full Ni master pattern",
    )
    parser.add_argument(
        "--in-package-master",
        type=Path,
        default=(
            Path(__file__).parents[1]
            / "emsoft_ebsd_master_pattern"
            / "ni_mc_mp_20kv_uint8_gzip_opts9.h5"
        ),
        help="in-package 401 px Ni master pattern",
    )
    parser.add_argument(
        "--full-master",
        type=Path,
        default=None,
        help=(
            "1001 px Ni master pattern; by default "
            "<cache-dir>/develop/data/ebsd_master_pattern/ni_mc_mp_20kv.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory the two .sht files are written to",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="scratch directory; a temporary one by default",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
