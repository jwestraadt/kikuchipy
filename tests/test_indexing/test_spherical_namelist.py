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

"""Tests of ``kikuchipy.indexing._spherical._namelist``.

Covers the ``test_spherical_namelist.py`` assertions of
``specs/2026-09-02-sht-interop/validation.md``:

- The generic parser: the ported ``test/util/nml.cpp`` suite (scalar
  and vector parsing, partial parsing detection and its eleven error
  cases), the error messages probed through ``IndexEBSD.exe``, the
  column zero comment rule, the unskipped white space only line, the
  two leading spaces message, the white space stripping inside the
  second and later strings of a quoted list, and the strict typing.
- The writer: line parity with the captured ``IndexEBSD -t``
  template, the six significant digit double formatting, the
  terminator quirk and the spaced master file guard.
- The reader: round trips of the defaults, of an acid test namelist
  and of a non-empty ``ipath`` variant, the derived path properties
  with the double ``ipath`` quirk, the optional key set, the always
  optional ``patdset``/``scanname``, the ``scandims`` rules, the
  vendor whitelist, the unused token warning and the thirteen sanity
  checks.
- The conversions: the frozen vendor table on a square and both
  rectangular detectors, the preconditioned ``pc_emsoft(version=4)``
  equality, the deviation from kikuchipy's ``pc_tsl``/``pc_oxford``,
  the ``delta`` invariance and the round trip.
- The arguments: ``to_kwargs`` against the live signatures, the
  circular mask map with its warning, ``to_detector`` and the
  ``from_kwargs`` rules.
- Behind ``KIKUCHIPY_EMSPHINX_DIR``: the live twin of the template
  fixture.
"""

import inspect
import re
import subprocess
import warnings

import h5py
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.detectors import EBSDDetector
from kikuchipy.indexing._spherical._indexer import SphericalIndexer
from kikuchipy.indexing._spherical._namelist import (
    VENDORS,
    EMSphInxNamelist,
    _geometry_to_pctr,
    _NameList,
    _pc_to_pctr,
    _pctr_to_geometry,
    _pctr_to_pc,
)

# ------------------------- Frozen constants ------------------------- #

# Every field of ``ebsd::Namelist`` (``nml.hpp`` lines 54-88) under
# its pythonic name (D5).  The round trip is asserted field for
# field against this tuple, so a new field cannot slip in untested.
FIELDS = (
    "ipath",
    "pat_file",
    "pat_dset",
    "master_files",
    "psym_file",
    "pat_dims",
    "circ_rad",
    "gaus_bckg",
    "n_regions",
    "delta",
    "vendor",
    "pctr",
    "thetac",
    "scan_dims",
    "scan_steps",
    "scan_file",
    "scan_name",
    "roi_mask",
    "bw",
    "normed",
    "refine",
    "n_thread",
    "batch_size",
    "opath",
    "data_file",
    "vendor_file",
    "ipf_name",
    "qual_name",
)

# The good namelist of ``test/util/nml.cpp`` lines 63-96, verbatim,
# with its two comment lines
GOOD_NAMELIST = (
    "! make sure we test a comment\n"
    " vTrue  = .true. ,\n"
    " vFalse = .false.,\n"
    " vInt    =  12345,\n"
    " vIntPos = +12345,\n"
    " vIntNeg = -12345,\n"
    " vDoub    =  1.2345,\n"
    " vDoubPos = +1.2345,\n"
    " vDoubNeg = -1.2345,\n"
    " vDoubSci =  1.23e4,\n"
    "! mix a comment in the middle\n"
    " vStr   = 'str',\n"
    r" vStrSgl = 'str \'with single quotes\''," + "\n"
    " vStrDbl = 'str \"with single quotes\"',\n"
    " vBools   = .true., .false., .false.,\n"
    " vInts    = 1, 2, 3 , 4,\n"
    " vDoubles = 1, 2, 3., 4,\n"
    " vStrs    = 'abc', '123', 'XYZ', '!@#',\n"
)

# The eleven error cases of ``nml.cpp`` lines 227-345, in its order
ERROR_CASES = [
    ("first line key value", " key = 1\n"),
    ("missing comma", "placeholder\n key = 1\n key2 = 2"),
    ("missing leading space", "placeholder\nkey = 1\n"),
    ("duplicate key", "placeholder\n key = 1,\n key = 2"),
    ("missing equals", "placeholder\n key = 1,\n key2 2"),
    ("missing string delimiter", "placeholder\n key = '1' '2'\n"),
    ("bad string opening", "placeholder\n key = '1', 2\n"),
    ("double quoted string", 'placeholder\n key = "1"\n'),
    ("unquoted string", "placeholder\n key = value\n"),
    ("int bool mix", "placeholder\n key = 1, .true.\n"),
    ("double bool mix", "placeholder\n key = 1.0, .true.\n"),
]

# The ``ebsd::Namelist`` parse messages which are not covered by
# ``ERROR_CASES``, verbatim from ``ebsd/nml.hpp`` lines 260-276 and
# ``index_ebsd.cpp`` line 83.  They are quoted here rather than
# matched loosely, as everywhere else in this suite: the C++ typo
# ("dimensinos") and the four spaces of "pctr    must be" are part
# of the frozen wording, and ``IndexEBSD.exe`` was probed for the
# unused token one.
SCANDIMS_COUNT_MESSAGE = (
    "expected a filename or dimensions + resolution for 'scandims' in namelist"
)
SCANDIMS_INTEGER_MESSAGE = "scan dimensinos must be non-negative integers"
PATDIMS_COUNT_MESSAGE = "patdims must be 2 elements"
PCTR_COUNT_MESSAGE = "pctr    must be 3 elements"
UNUSED_TOKEN_MESSAGE = "some namelist parameters weren't used: extrakey"

# A complete namelist for the small nickel map on the canonical
# route, with a full precision pattern centre.  Its values are the
# acid test's, and reading it must give them back exactly.
ACID_NAMELIST = (
    " &EMSphInx\n"
    " patfile    = 'patterns.h5',\n"
    " patdset    = 'patterns',\n"
    " masterfile = 'ni.sht',\n"
    " patdims    = 60, 60,\n"
    " circmask   = -1,\n"
    " gausbckg   = .FALSE.,\n"
    " nregions   = 10,\n"
    " delta      = 500,\n"
    " pctr       = 0.42513885, 0.21336699, 0.50070692,\n"
    " vendor     = 'Bruker',\n"
    " thetac     = 0,\n"
    " scandims   = 3, 3, 1.5,\n"
    " roimask    = '',\n"
    " bw         = 68,\n"
    " normed     = .TRUE.,\n"
    " refine     = .TRUE.,\n"
    " nthread    = 1,\n"
    " batchsize  = 1,\n"
    " datafile   = 'out.h5',\n"
    " vendorfile = 'out.ang'\n"
    " /\n"
)

ACID_PCTR = (0.42513885, 0.21336699, 0.50070692)

# The pattern centre of the nickel detector, rounded to the eight
# decimals of ``ACID_PCTR``, and the geometry triple it converts to
# at ``delta`` 500 on the 60 x 60 detector.  The recorded triple
# derives from the **unrounded** ``pc_average``
# (0.21336699472343632...), so the assertions come from the same
# unrounded source and use ``pytest.approx`` (D6).
NI_GEOMETRY = (-4.49166896, 17.19798032, 15021.20746804)

# The frozen conversion table (D6) on the two rectangular detectors,
# for the Bruker pattern centre (0.4251, 0.2134, 0.5007) at
# ``delta`` 500.  ``shape`` is kikuchipy's (n rows, n columns), so
# ``w = shape[1]`` and ``h = shape[0]``.
RECTANGULAR_PC = (0.4251, 0.2134, 0.5007)
RECTANGULAR_TABLE = {
    (48, 60): {
        "geometry": (-4.494, 13.7568, 12016.8),
        "EMsoft": (-4.494, 13.7568, 12016.8),
        "EDAX": (0.4251, 0.62928, 0.40056),
        "tsl": (0.4251, 0.62928, 0.40056),
        "Oxford": (0.4251, 0.7866, 0.40056),
        "Bruker": (0.4251, 0.2134, 0.5007),
    },
    (60, 48): {
        "geometry": (-3.5952, 17.196, 15021.0),
        "EMsoft": (-3.5952, 17.196, 15021.0),
        "EDAX": (0.4251, 0.98325, 0.625875),
        "tsl": (0.4251, 0.98325, 0.625875),
        "Oxford": (0.4251, 0.7866, 0.625875),
        "Bruker": (0.4251, 0.2134, 0.5007),
    },
}

# Which kikuchipy helper each EMSphInx conversion agrees with, per
# detector shape.  ``pc_oxford`` matches the EMSphInx **TSL** row
# everywhere, and ``pc_tsl`` matches the EMSphInx **Oxford** row
# only where kikuchipy's ``min(nrows, ncols) / nrows`` factor
# happens to reproduce the ``h/w`` scaling: on (60, 48) it does, on
# (48, 60) it is a no-op and the z components differ.
#
# (corrected 2026-09-02 from the spec's "asserted on both (48, 60)
# and (60, 48)": measured here, EMSphInx-Oxford **equals**
# kikuchipy ``pc_tsl()`` on (60, 48).  The deviation row is
# (48, 60) only; see the appended Recorded results of
# validation.md.)
OXFORD_EQUALS_KP_TSL = {(60, 60): True, (48, 60): False, (60, 48): True}

# ``sanityCheck()`` cases, one per bound (D5): thirteen checks, of
# which the three negativity ones are live in the binary
# (``nregions = -5`` and ``nthread = -1`` exit 1, measured)
SANITY_FAILURES = [
    ({"pat_file": ""}, "missing input pattern file"),
    ({"master_files": []}, "no master pattern files"),
    ({"master_files": ["ni.sht", ""]}, "empty master pattern file name"),
    (
        {"pat_dims": (1, 60), "delta": 30000.0},
        "unreasonable pattern dimension",
    ),
    ({"pat_dims": (60, 16385)}, "unreasonable pattern dimension"),
    ({"circ_rad": -2}, "circular mask radius must be >= -1"),
    ({"n_regions": -5}, "unreasonable AHE nregions"),
    ({"n_regions": 61}, "unreasonable AHE nregions"),
    ({"delta": 83.0}, "unreasonable EBSD detector width"),
    ({"delta": 1501.0}, "unreasonable EBSD detector width"),
    ({"thetac": -60.1}, "unreasonable camera tilt"),
    ({"thetac": 60.1}, "unreasonable camera tilt"),
    ({"scan_dims": (0, 3)}, "non-positive scan dimensions"),
    ({"scan_dims": (3, 0)}, "non-positive scan dimensions"),
    ({"bw": 15}, "unreasonable bandwidth"),
    ({"bw": 513}, "unreasonable bandwidth"),
    ({"n_thread": -1}, "negative thread count"),
    ({"batch_size": -1}, "negative batch size"),
    ({"data_file": ""}, "missing output data file"),
]

SANITY_LIMITS = [
    {"pat_dims": (2, 2), "delta": 15000.0, "n_regions": 2},
    {"pat_dims": (16384, 60), "delta": 1.83},
    {"circ_rad": -1},
    {"n_regions": 0},
    {"n_regions": 60},
    {"delta": 84.0},
    {"delta": 1500.0},
    {"thetac": -60.0},
    {"thetac": 60.0},
    {"scan_dims": (1, 1)},
    {"bw": 16},
    {"bw": 512},
    {"n_thread": 0},
    {"batch_size": 0},
]

# The ``IndexEBSD.exe -t`` template, captured on 2026-09-02 from the
# Windows binary built at EMSphInx commit 60f3517.  The capture is
# CRLF (the binary writes in text mode) with an md5 of
# 49ddf0e7d9b2d758d918c20a7f900a6d, so every comparison is line
# wise.  The ``masterfile`` line ends with ", " -- a **trailing
# space**, which ``to_string()`` writes and which
# ``test_the_template_master_line_keeps_its_trailing_space``
# guards.  That one space is written as the sentinel ``<SP>`` and
# substituted below, because a linter (ruff W291) and an editor
# would both eat a real trailing space.
INDEX_EBSD_TEMPLATE = """ &EMSphInx
!#################################################################
! Input Files
!#################################################################

! raw pattern file (relative to ipath) [can be up1, up2, or hdf5]
 patfile    = 'scan.h5',

! h5 path of raw pattern  (ignored for non hdf5 patfile)
 patdset    = 'Scan 1/EBSD/Data/Pattern',

! master pattern with phases to index (relative to ipath)
 masterfile = 'master.h5',<SP>


!#################################################################
! Pattern Processing
!#################################################################

! number of CCD pixels along x and y
 patdims    = 640, 480,

! should a circular mask be applied (-1 for no mask, 0 for largest inscribed circle, >0 to specify radius in pixels)
 circmask   = -1,

! should a 2D gaussian background be subtracted
 gausbckg   = .FALSE.,

! how many regions should be used for adaptive histogram equalization (0 for no AHE)
 nregions   = 10,


!#################################################################
! Camera Calibration
!#################################################################

! CCD pixel size on the scintillator surface [microns]
 delta      = 50,

! pattern center coordinates and vendor
! vendor must be one of the following:
!   EMsoft, EDAX, TSL, Oxford, Bruker
! with pctr interpreted accordingly:
!   EMsoft   - pcx (pixels), pcy (pixels), scintillator distance (microns)
!   EDAX/TSL - x*, y*, z*
!   Oxford   - x*, y*, z*
!   Bruker   - x*, y*, z*
! note that vendors use different x*, y*, and z* : https://doi.org/10.1007/s40192-019-00137-4
 pctr       = 0, 0, 15000,
 vendor     = 'EMsoft',

! tilt angle of the camera (positive below horizontal, [degrees]
 thetac     = 10,


!#################################################################
! Scan Information
!#################################################################

! dimensions of scan to index and pixel size
! x, y, step   for an x by y scan with square pixels of 'step' microns
! x, y, sx, sy for an x by y scan with rectangular pixels of 'sx' by 'sy' microns
! string to read dimensions from a scan file (*.ang, *.ctf, or *.h5)
 scandims   = 256, 256, 1, 1,

! region of interest for indexing
! 0 (or omitted) to index the entire scan
! x0, y0, dx, dy for a (dx, dy) rectangular starting at pixel (x0, y0)
! string for an ROI mask file
 roimask    = '',

!#################################################################
! Indexing Parameters
!#################################################################

! spherical harmonic bandwidth to be used (2*bw-1 should be a product of small primes for speed)
! some reasonable values are: 53, 63, 68, 74, 88, 95, 113, 122, 123, 158, 172, 188, 203, 221, 263, 284, 313
! a nice range for parameter studies is 53, 68, 88, 113, 158, 203, 263, 338 (~a factor of 1.3 between each)
! any value is now pretty fast since the transform is zero padded to the nearest fast FFT size
 bw         = 68,

! should normalized / unnormalized spherical cross correlation be used?
! normalization is more robust for (esp. for lower symmetries) but is slower
 normed     = .TRUE.,

! should newton's method orientation refinement be used?
! normalization is more robust for (esp. for lower symmetries) but is slower
 refine     = .TRUE.,

! number of work threads
! 0 (or omitted) to multithread with an automatic number of threads
! 1 for serial threading
! N to multithread with N threads
 nthread    = 0,

! number of patterns to index per work itme (ignored for single threading)
! should be large enough to make the task significant compared to thread overhead
! should be small enough to enable enough work items for load balancing
! should be small enough so nthread * batchsize patterns can be held in memory
! 0 (or omitted) to estimate a reasonable value based on speed
 batchsize  = 0,


!#################################################################
! Output Files
!#################################################################

! output orientation map name relative to opath [must be hdf5 type]
 datafile   = 'SphInx_Scan.h5',

! output orientation map name relative to opath (or omitted for no vendor output) [can be ang or ctf]
 vendorfile = 'reindexed.ang',

! output ipf map with {0,0,1} reference direction (or omitted for no ipf map) [must be png]
 ipfmap     = 'ipf.png',

! output quality map with (or omitted for no quality map) [must be png]
 qualmap    = 'qual.png'
 /
""".replace("<SP>", " ")


# ----------------------------- Helpers ------------------------------ #


def acid_namelist():
    """Return the acid test namelist, parsed from its file text."""
    return EMSphInxNamelist.from_string(ACID_NAMELIST)


def namelist_with(**overrides):
    """Return the acid namelist with fields replaced."""
    namelist = acid_namelist()
    for name, value in overrides.items():
        setattr(namelist, name, value)
    return namelist


def assert_fields_equal(one, two, pctr_rel=0.0):
    """Assert two namelists agree on every field of :data:`FIELDS`.

    Exactly, except that ``pctr`` may differ by ``pctr_rel``: a
    pattern centre with more than six significant digits does not
    survive the C++ stream precision of ``to_string`` (D5).
    """
    for name in FIELDS:
        first = getattr(one, name)
        second = getattr(two, name)
        if name == "pctr" and pctr_rel:
            assert tuple(first) == pytest.approx(tuple(second), rel=pctr_rel)
        elif (
            isinstance(first, (tuple, list))
            and len(first)
            and not isinstance(first[0], str)
        ):
            assert tuple(first) == tuple(second), name
        else:
            assert first == second, name


def written_keys(text):
    """Return the set of keys a namelist file text writes."""
    return {
        line.split("=")[0].strip()
        for line in text.splitlines()
        if line.startswith(" ") and "=" in line
    }


def without_line(text, key):
    """Return a namelist file text without the line writing ``key``."""
    lines = [line for line in text.splitlines() if not line.startswith(f" {key} ")]
    return "".join(line + "\n" for line in lines)


def line_of(text, key):
    """Return the single line of a namelist file starting with a
    space and ``key``.
    """
    lines = [line for line in text.splitlines() if line.startswith(f" {key} ")]
    assert len(lines) == 1, f"{key} appears {len(lines)} times"
    return lines[0]


def ni_detector(shape=(60, 60), pc=None, delta=500.0):
    """Return a detector whose ``binning`` and ``px_size`` satisfy
    the precondition of the ``pc_emsoft(version=4)`` equality.
    """
    if pc is None:
        pc = RECTANGULAR_PC
    return EBSDDetector(shape=shape, pc=pc, px_size=delta, binning=1)


def width_height(shape):
    """Return the EMSphInx ``(w, h)`` of a kikuchipy shape."""
    return shape[1], shape[0]


# ------------------ The generic parser (D4) ------------------------- #


class TestNameListParser:
    def test_scalar_parsing(self):
        namelist = _NameList.from_string(GOOD_NAMELIST)
        assert namelist.get_bool("vTrue") is True
        assert namelist.get_bool("vFalse") is False
        assert namelist.get_int("vInt") == 12345
        assert namelist.get_int("vIntPos") == 12345
        assert namelist.get_int("vIntNeg") == -12345
        assert namelist.get_double("vDoub") == 1.2345
        assert namelist.get_double("vDoubPos") == 1.2345
        assert namelist.get_double("vDoubNeg") == -1.2345
        assert namelist.get_double("vDoubSci") == 1.23e4
        assert namelist.get_string("vStr") == "str"
        assert namelist.get_string("vStrSgl") == "str 'with single quotes'"
        assert namelist.get_string("vStrDbl") == 'str "with single quotes"'

    def test_vector_parsing(self):
        namelist = _NameList.from_string(GOOD_NAMELIST)
        assert namelist.get_bools("vTrue") == [True]
        assert namelist.get_ints("vInt") == [12345]
        assert namelist.get_doubles("vDoub") == [1.2345]
        assert namelist.get_strings("vStr") == ["str"]
        assert namelist.get_bools("vBools") == [True, False, False]
        assert namelist.get_ints("vInts") == [1, 2, 3, 4]
        # a mix of integers and doubles is promoted to all doubles
        doubles = namelist.get_doubles("vDoubles")
        assert doubles == [1.0, 2.0, 3.0, 4.0]
        assert all(isinstance(value, float) for value in doubles)
        assert namelist.get_strings("vStrs") == ["abc", "123", "XYZ", "!@#"]

    def test_an_int_list_is_not_readable_as_ints_after_promotion(self):
        # ``vDoubles`` holds doubles only, so the integer accessor
        # refuses it (``Variant::getInt()``)
        namelist = _NameList.from_string(GOOD_NAMELIST)
        with pytest.raises(ValueError, match="stored type isn't integer"):
            namelist.get_ints("vDoubles")

    def test_second_string_whitespace_stripped(self):
        # the sticky ``std::skipws`` of ``nml.hpp`` lines 364-368:
        # measured decisively through the binary, which opened
        # ``nismallstripped.sht`` for a spaced second master file
        namelist = _NameList.from_string(
            "placeholder\n key = 'a b', 'c d e', 'f  g',\n"
        )
        assert namelist.get_strings("key") == ["a b", "cde", "fg"]

    def test_empty_string_parses(self):
        namelist = _NameList.from_string("placeholder\n key = '',\n")
        assert namelist.get_string("key") == ""

    def test_partial_parsing(self):
        namelist = _NameList.from_string(GOOD_NAMELIST)
        for key in ("vTrue", "vFalse", "vBools"):
            namelist.get_bools(key)
        for key in ("vInt", "vIntPos", "vIntNeg", "vInts"):
            namelist.get_ints(key)
        for key in ("vDoub", "vDoubPos", "vDoubNeg", "vDoubSci", "vDoubles"):
            namelist.get_doubles(key)
        for key in ("vStr", "vStrSgl", "vStrDbl", "vStrs"):
            namelist.get_strings(key)
        assert namelist.fully_parsed()
        assert namelist.unused_tokens() == ""

        namelist = _NameList.from_string("placeholder\n tokenOne = 1,\n tokenTwo = 2")
        namelist.get_int("tokenOne")
        assert not namelist.fully_parsed()
        assert namelist.unused_tokens() == "tokentwo"

    def test_unused_tokens_are_lower_case_and_sorted(self):
        namelist = _NameList.from_string(
            "placeholder\n Zulu = 1,\n alpha = 2,\n Mike = 3,\n"
        )
        assert namelist.unused_tokens() == "alpha,mike,zulu"

    @pytest.mark.parametrize("name, text", ERROR_CASES, ids=[c[0] for c in ERROR_CASES])
    def test_error_cases(self, name, text):
        with pytest.raises(ValueError):
            _NameList.from_string(text)

    def test_first_line_is_skipped(self):
        # the first line is skipped whatever it holds, as long as it
        # has no '='
        namelist = _NameList.from_string("anything at all\n key = 1,\n")
        assert namelist.get_int("key") == 1

    def test_terminator_and_empty_lines_are_skipped(self):
        namelist = _NameList.from_string("placeholder\n a = 1,\n\n b = 2,\n /\n")
        assert namelist.get_int("a") == 1
        assert namelist.get_int("b") == 2

    def test_column0_comment_is_skipped(self):
        namelist = _NameList.from_string("placeholder\n a = 1,\n! a comment\n b = 2,\n")
        assert namelist.get_int("b") == 2

    def test_indented_comment_is_not_a_comment(self):
        # ``nml.hpp`` line 307 tests ``line.front()`` while its own
        # doc comment at line 291 says "first character after white
        # space"; the code wins, measured through the binary
        text = "placeholder\n a = 1,\n ! an indented comment\n b = 2,\n"
        with pytest.raises(
            ValueError,
            match=re.escape(
                "bad delimeter (expected '=') in namelist line 3 "
                '" ! an indented comment"'
            ),
        ):
            _NameList.from_string(text)

    def test_whitespace_only_line_raises(self):
        # the literal ``line.empty()`` of line 306: the idiomatic
        # ``if not line.strip(): continue`` would diverge, measured
        with pytest.raises(
            ValueError, match=re.escape("error parsing line '   ' from name list")
        ):
            _NameList.from_string("placeholder\n a = 1,\n   \n b = 2,\n")

    @pytest.mark.parametrize("prefix", ["", "\t"])
    def test_missing_leading_space_message(self, prefix):
        text = f"placeholder\n a = 1,\n{prefix}b = 2,\n"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'missing leading space in namelist line 3 "{prefix}b = 2,"'
            ),
        ):
            _NameList.from_string(text)

    def test_two_leading_spaces_give_a_different_message(self):
        # the ``noskipws`` key extraction fails on the second space,
        # so this is *not* the missing leading space error, measured
        text = "placeholder\n a = 1,\n  b = 2,\n"
        with pytest.raises(
            ValueError,
            match=re.escape("error parsing line '  b = 2,' from name list"),
        ):
            _NameList.from_string(text)

    def test_duplicate_key_message(self):
        text = "placeholder\n circmask = -1,\n circmask = 0,\n"
        with pytest.raises(
            ValueError,
            match=re.escape('key "circmask" was defined twice in the name list'),
        ):
            _NameList.from_string(text)

    def test_first_line_key_value_message(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "namelist files cannot have key value pairs in the first line"
            ),
        ):
            _NameList.from_string(" key = 1,\n")

    def test_missing_comma_message(self):
        text = "placeholder\n a = 1\n b = 2,\n"
        with pytest.raises(
            ValueError,
            match=re.escape(
                'missing comma between previous entry and namelist line 3 " b = 2,"'
            ),
        ):
            _NameList.from_string(text)

    def test_unparsable_token_message(self):
        text = "placeholder\n a = value,\n"
        with pytest.raises(
            ValueError,
            match=re.escape(
                'couldn\'t parse token "value" from line 2 " a = value," as '
                "bool, int, or float (strings must be in single quotes, e.g. "
                "key = 'value')"
            ),
        ):
            _NameList.from_string(text)

    def test_missing_key_message(self):
        namelist = _NameList.from_string("placeholder\n a = 1,\n")
        with pytest.raises(
            ValueError, match=re.escape("couldn't find `bw' in namelist")
        ):
            namelist.get_int("bw")

    def test_get_int_rejects_doubles(self):
        namelist = _NameList.from_string("placeholder\n bw = 68.0,\n")
        with pytest.raises(ValueError, match="stored type isn't integer"):
            namelist.get_int("bw")
        assert namelist.get_double("bw") == 68.0

    def test_get_double_accepts_ints(self):
        # ``Variant::getDouble()`` casts integers silently, which is
        # what lets ``delta = 500`` and ``pctr = 0, 0, 15000`` parse
        namelist = _NameList.from_string(
            "placeholder\n delta = 500,\n pctr = 0, 0, 15000,\n"
        )
        assert namelist.get_double("delta") == 500.0
        assert namelist.get_doubles("pctr") == [0.0, 0.0, 15000.0]

    def test_get_bool_and_string_are_strict(self):
        namelist = _NameList.from_string("placeholder\n a = 1,\n b = 'x',\n")
        with pytest.raises(ValueError, match="stored type isn't boolean"):
            namelist.get_bool("a")
        with pytest.raises(ValueError, match="stored type isn't string"):
            namelist.get_string("a")
        with pytest.raises(ValueError, match="stored type isn't integer"):
            namelist.get_int("b")

    def test_keys_are_case_insensitive(self):
        namelist = _NameList.from_string("placeholder\n BW = 68,\n")
        assert namelist.get_int("bw") == 68
        assert namelist.get_int("BW") == 68
        assert namelist.unused_tokens() == ""

    def test_read_from_a_file(self, tmp_path):
        fpath = tmp_path / "test.nml"
        fpath.write_text(GOOD_NAMELIST, encoding="utf-8")
        assert _NameList.read(fpath).get_int("vInt") == 12345


# --------------------- The written template (D5) -------------------- #


class TestNamelistTemplate:
    def test_the_template_master_line_keeps_its_trailing_space(self):
        # ``to_string()`` writes "', " after every master file name,
        # so the line ends in a space; stripping it here would make
        # the parity test pass against a wrong writer
        assert INDEX_EBSD_TEMPLATE.count("\n") == 119
        assert " masterfile = 'master.h5', \n" in INDEX_EBSD_TEMPLATE
        assert "<SP>" not in INDEX_EBSD_TEMPLATE

    def test_defaults_to_string_matches_index_ebsd_template(self):
        text = EMSphInxNamelist.defaults().to_string()
        lines = text.splitlines()
        expected = INDEX_EBSD_TEMPLATE.splitlines()
        assert len(lines) == len(expected) == 119
        for i, (line, want) in enumerate(zip(lines, expected)):
            assert line == want, f"line {i + 1}"
        assert lines[0] == " &EMSphInx"
        assert lines[-1] == " /"

    def test_defaults_values(self):
        namelist = EMSphInxNamelist.defaults()
        assert namelist.ipath == ""
        assert namelist.pat_file == "scan.h5"
        assert namelist.pat_dset == "Scan 1/EBSD/Data/Pattern"
        assert namelist.master_files == ["master.h5"]
        assert namelist.psym_file == ""
        assert tuple(namelist.pat_dims) == (640, 480)
        assert namelist.circ_rad == -1
        assert namelist.gaus_bckg is False
        assert namelist.n_regions == 10
        assert namelist.delta == 50.0
        assert namelist.vendor == "EMsoft"
        assert tuple(namelist.pctr) == (0.0, 0.0, 15000.0)
        assert namelist.thetac == 10.0
        assert tuple(namelist.scan_dims) == (256, 256)
        assert tuple(namelist.scan_steps) == (1.0, 1.0)
        assert namelist.roi_mask == ""
        assert namelist.bw == 68
        assert namelist.normed is True
        assert namelist.refine is True
        assert namelist.n_thread == 0
        assert namelist.batch_size == 0
        assert namelist.opath == ""
        assert namelist.data_file == "SphInx_Scan.h5"
        assert namelist.vendor_file == "reindexed.ang"
        assert namelist.ipf_name == "ipf.png"
        assert namelist.qual_name == "qual.png"

    def test_doubles_format_like_cpp_streams(self):
        # the C++ writes doubles at the default stream precision of
        # six significant digits; ``repr`` would give "50.0"
        namelist = namelist_with(
            delta=50.0,
            thetac=1.5,
            pctr=(0.42513885, 12345678.9, 15000.0),
            scan_steps=(1.0, 0.0001234567),
        )
        text = namelist.to_string()
        assert line_of(text, "delta") == " delta      = 50,"
        assert line_of(text, "thetac") == " thetac     = 1.5,"
        assert line_of(text, "pctr") == " pctr       = 0.425139, 1.23457e+07, 15000,"
        assert line_of(text, "scandims") == " scandims   = 3, 3, 1, 0.000123457,"

    def test_bools_are_fortran_logicals(self):
        text = namelist_with(gaus_bckg=True, normed=False, refine=True).to_string()
        assert line_of(text, "gausbckg") == " gausbckg   = .TRUE.,"
        assert line_of(text, "normed") == " normed     = .FALSE.,"
        assert line_of(text, "refine") == " refine     = .TRUE.,"

    def test_optional_blocks_are_omitted_when_empty(self):
        keys = written_keys(acid_namelist().to_string())
        assert keys.isdisjoint(
            {"ipath", "psymfile", "scanname", "opath", "ipfmap", "qualmap"}
        )
        assert keys == {
            "patfile",
            "patdset",
            "masterfile",
            "patdims",
            "circmask",
            "gausbckg",
            "nregions",
            "delta",
            "pctr",
            "vendor",
            "thetac",
            "scandims",
            "roimask",
            "bw",
            "normed",
            "refine",
            "nthread",
            "batchsize",
            "datafile",
            "vendorfile",
        }

    def test_optional_blocks_are_written_when_set(self):
        namelist = namelist_with(
            ipath="in/",
            psym_file="psym.txt",
            scan_name="Scan 1",
            opath="out/",
            ipf_name="ipf.png",
            qual_name="qual.png",
        )
        text = namelist.to_string()
        assert line_of(text, "ipath") == " ipath      = 'in/',"
        assert line_of(text, "psymfile") == " psymfile   = 'psym.txt',"
        assert line_of(text, "scanname") == " scanname   = 'Scan 1',"
        assert line_of(text, "opath") == " opath      = 'out/',"
        assert line_of(text, "ipfmap") == " ipfmap     = 'ipf.png',"
        assert line_of(text, "qualmap") == " qualmap    = 'qual.png'"

    def test_to_string_without_qualmap_has_no_terminator(self):
        # ``to_string()`` emits " /" only inside the qualmap block
        # (lines 464-468), so an empty quality map name gives a file
        # without a terminator, which the parser accepts
        text = acid_namelist().to_string()
        assert text.splitlines()[-1] != " /"
        assert " /" not in text.splitlines()
        assert_fields_equal(
            EMSphInxNamelist.from_string(text), acid_namelist(), pctr_rel=1e-6
        )

        with_qualmap = namelist_with(qual_name="qual.png").to_string()
        assert with_qualmap.splitlines()[-1] == " /"

    def test_the_master_file_list_is_comma_space_separated(self):
        text = namelist_with(master_files=["a.sht", "b.sht"]).to_string()
        assert line_of(text, "masterfile") == " masterfile = 'a.sht', 'b.sht', "

    def test_write_rejects_spaced_second_master(self, tmp_path):
        # the parser strips all white space inside the second and
        # later strings of a quoted list, so writing one would lose
        # data on read back (D4)
        namelist = namelist_with(master_files=["a.sht", "b c.sht"])
        with pytest.raises(ValueError, match="white space"):
            namelist.to_string()
        with pytest.raises(ValueError, match="white space"):
            namelist.write(tmp_path / "bad.nml", overwrite=True)

    def test_write_accepts_a_spaced_first_master(self, tmp_path):
        namelist = namelist_with(master_files=["ni small.sht"])
        fpath = tmp_path / "spaced.nml"
        namelist.write(fpath, overwrite=True)
        assert EMSphInxNamelist.read(fpath).master_files == ["ni small.sht"]

    def test_write_conventions(self, tmp_path):
        namelist = acid_namelist()
        namelist.write(tmp_path / "sub" / "index", overwrite=True)
        fpath = tmp_path / "sub" / "index.nml"
        assert fpath.is_file()
        assert fpath.read_text(encoding="utf-8") == namelist.to_string()
        assert "\r" not in fpath.read_bytes().decode("utf-8")

        before = fpath.read_bytes()
        namelist_with(bw=88).write(fpath, overwrite=False)
        assert fpath.read_bytes() == before
        namelist_with(bw=88).write(fpath, overwrite=True)
        assert EMSphInxNamelist.read(fpath).bw == 88


# ---------------------- Reading a namelist (D5) --------------------- #


class TestNamelistRoundTrip:
    def test_defaults_round_trip(self):
        namelist = EMSphInxNamelist.defaults()
        again = EMSphInxNamelist.from_string(namelist.to_string())
        assert_fields_equal(again, namelist)
        assert again == namelist

    def test_acid_namelist_round_trip(self):
        # exact on every field which is representable at the C++
        # stream precision of six significant digits, which the eight
        # decimal ``pctr`` is not: it comes back rounded, about 3e-4
        # of a pixel on a 60 pixel detector, and the *second* round
        # trip is then exact (D5, open question 6)
        namelist = acid_namelist()
        again = EMSphInxNamelist.from_string(namelist.to_string())
        assert_fields_equal(again, namelist, pctr_rel=1e-6)
        assert tuple(again.pctr) == (0.425139, 0.213367, 0.500707)
        assert EMSphInxNamelist.from_string(again.to_string()) == again

    def test_non_empty_ipath_round_trips(self):
        # paths are stored raw, so this is exact; the C++ stores the
        # prefixed values and would double prefix here (D5)
        namelist = namelist_with(ipath="data/", psym_file="")
        again = EMSphInxNamelist.from_string(namelist.to_string())
        assert again.ipath == "data/"
        assert again.pat_file == "patterns.h5"
        assert again.master_files == ["ni.sht"]
        assert_fields_equal(again, namelist, pctr_rel=1e-6)

    def test_reading_the_acid_namelist_gives_its_literal_values(self):
        namelist = acid_namelist()
        assert namelist.pat_file == "patterns.h5"
        assert namelist.pat_dset == "patterns"
        assert namelist.master_files == ["ni.sht"]
        assert tuple(namelist.pat_dims) == (60, 60)
        assert namelist.circ_rad == -1
        assert namelist.gaus_bckg is False
        assert namelist.n_regions == 10
        assert namelist.delta == 500.0
        assert namelist.vendor == "Bruker"
        # the eight decimal pattern centre is parsed in full
        assert tuple(namelist.pctr) == ACID_PCTR
        assert namelist.thetac == 0.0
        assert tuple(namelist.scan_dims) == (3, 3)
        assert tuple(namelist.scan_steps) == (1.5, 1.5)
        assert namelist.roi_mask == ""
        assert namelist.bw == 68
        assert namelist.normed is True
        assert namelist.refine is True
        assert namelist.n_thread == 1
        assert namelist.batch_size == 1
        assert namelist.data_file == "out.h5"
        assert namelist.vendor_file == "out.ang"
        assert namelist.ipf_name == ""
        assert namelist.qual_name == ""

    def test_repr_is_one_line_naming_the_files_and_the_bandwidth(self):
        namelist = acid_namelist()
        assert repr(namelist) == (
            "EMSphInxNamelist: 'patterns.h5' (60 x 60), 1 master pattern, "
            "3 x 3 scan, bw = 68"
        )
        assert "\n" not in repr(namelist)

    def test_repr_pluralises_and_follows_the_fields(self):
        # x then y, the namelist's own order, so a rectangular scan
        # and detector catch a transposition
        namelist = namelist_with(
            master_files=["a.sht", "b.sht"],
            pat_dims=(60, 48),
            scan_dims=(4, 3),
            bw=88,
        )
        assert repr(namelist) == (
            "EMSphInxNamelist: 'patterns.h5' (60 x 48), 2 master patterns, "
            "4 x 3 scan, bw = 88"
        )

    def test_derived_paths(self):
        namelist = namelist_with(ipath="data/", master_files=["a.sht", "b.sht"])
        assert namelist.pat_path == "data/patterns.h5"
        assert namelist.master_paths == ["data/a.sht", "data/b.sht"]

    def test_psymfile_double_ipath_quirk_is_ported(self):
        # ``parse_nml()`` line 247 prefixes ``patFile`` a second time
        # when a pseudo symmetry file is given
        namelist = namelist_with(ipath="data/", psym_file="psym.txt")
        assert namelist.pat_path == "data/data/patterns.h5"
        assert namelist.master_paths == ["data/ni.sht"]

    def test_read_from_a_file(self, tmp_path):
        fpath = tmp_path / "acid.nml"
        fpath.write_text(ACID_NAMELIST, encoding="utf-8")
        assert EMSphInxNamelist.read(fpath) == acid_namelist()

    @pytest.mark.parametrize(
        "key, field",
        [
            ("ipath", "ipath"),
            ("psymfile", "psym_file"),
            ("opath", "opath"),
            ("vendorfile", "vendor_file"),
            ("ipfmap", "ipf_name"),
            ("qualmap", "qual_name"),
        ],
    )
    def test_optional_keys(self, key, field):
        # each of the six is read in a try/except by ``parse_nml``,
        # so its absence means "no such input or output"
        full = namelist_with(
            ipath="in/",
            psym_file="psym.txt",
            opath="out/",
            vendor_file="out.ang",
            ipf_name="ipf.png",
            qual_name="qual.png",
        ).to_string()
        assert key in written_keys(full)
        namelist = EMSphInxNamelist.from_string(without_line(full, key))
        assert getattr(namelist, field) == ""

    def test_a_missing_required_key_raises(self):
        text = EMSphInxNamelist.defaults().to_string()
        with pytest.raises(
            ValueError, match=re.escape("couldn't find `bw' in namelist")
        ):
            EMSphInxNamelist.from_string(without_line(text, "bw"))

    def test_patdset_scanname_always_optional(self, tmp_path):
        # the C++ requiredness depends on whether ``patfile`` exists
        # and is HDF5, which a pure parser must not reproduce (D5)
        without = ACID_NAMELIST.replace(" patdset    = 'patterns',\n", "")
        namelist = EMSphInxNamelist.from_string(without)
        assert namelist.pat_dset == ""

        # ... whether or not the pattern file is a real HDF5 file
        fpath = tmp_path / "patterns.h5"
        with h5py.File(fpath, mode="w") as f:
            f.create_dataset("patterns", data=np.zeros((1, 2, 2), np.uint8))
        text = without.replace("'patterns.h5'", f"'{fpath.as_posix()}'")
        assert EMSphInxNamelist.from_string(text).pat_dset == ""

    def test_scanname_is_consumed_with_numeric_scandims(self):
        # always consumed, so it neither warns nor is dropped
        text = ACID_NAMELIST.replace(
            " roimask    = '',\n", " scanname   = 'Scan 1',\n roimask    = '',\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            namelist = EMSphInxNamelist.from_string(text)
        assert namelist.scan_name == "Scan 1"
        again = EMSphInxNamelist.from_string(namelist.to_string())
        assert again.scan_name == "Scan 1"
        assert_fields_equal(again, namelist, pctr_rel=1e-6)

    def test_scandims_three_or_four(self):
        three = acid_namelist()
        assert tuple(three.scan_steps) == (1.5, 1.5)
        four = EMSphInxNamelist.from_string(
            ACID_NAMELIST.replace(
                " scandims   = 3, 3, 1.5,\n", " scandims   = 4, 3, 1.5, 2.5,\n"
            )
        )
        assert tuple(four.scan_dims) == (4, 3)
        assert tuple(four.scan_steps) == (1.5, 2.5)

    @pytest.mark.parametrize(
        "value", [" scandims   = 3, 3,\n", " scandims   = 3, 3, 1, 1, 1,\n"]
    )
    def test_scandims_must_be_three_or_four(self, value):
        text = ACID_NAMELIST.replace(" scandims   = 3, 3, 1.5,\n", value)
        with pytest.raises(ValueError, match=re.escape(SCANDIMS_COUNT_MESSAGE)):
            EMSphInxNamelist.from_string(text)

    def test_non_integer_scandims_raise(self):
        # the C++ typo "dimensinos" is part of the frozen message
        text = ACID_NAMELIST.replace(
            " scandims   = 3, 3, 1.5,\n", " scandims   = 3.5, 3, 1.5,\n"
        )
        with pytest.raises(ValueError, match=re.escape(SCANDIMS_INTEGER_MESSAGE)):
            EMSphInxNamelist.from_string(text)

    @pytest.mark.parametrize(
        "value", [" patdims    = 60,\n", " patdims    = 60, 60, 60,\n"]
    )
    def test_patdims_must_be_two_elements(self, value):
        text = ACID_NAMELIST.replace(" patdims    = 60, 60,\n", value)
        with pytest.raises(ValueError, match=re.escape(PATDIMS_COUNT_MESSAGE)):
            EMSphInxNamelist.from_string(text)

    @pytest.mark.parametrize(
        "value",
        [
            " pctr       = 0.42513885, 0.21336699,\n",
            " pctr       = 0.42513885, 0.21336699, 0.50070692, 1.0,\n",
        ],
    )
    def test_pctr_must_be_three_elements(self, value):
        # the four spaces of "pctr    must be" are the C++'s own
        text = ACID_NAMELIST.replace(
            " pctr       = 0.42513885, 0.21336699, 0.50070692,\n", value
        )
        with pytest.raises(ValueError, match=re.escape(PCTR_COUNT_MESSAGE)):
            EMSphInxNamelist.from_string(text)

    def test_string_scandims_raise_not_implemented(self):
        # the scan file route needs ``OrientationMap``, which is out
        # of scope (open question 8)
        text = ACID_NAMELIST.replace(
            " scandims   = 3, 3, 1.5,\n", " scandims   = 'scan.ang',\n"
        )
        with pytest.raises(NotImplementedError, match="scandims"):
            EMSphInxNamelist.from_string(text)

    @pytest.mark.parametrize("vendor", VENDORS)
    def test_vendor_whitelist_accepts(self, vendor):
        text = ACID_NAMELIST.replace("'Bruker'", f"'{vendor}'")
        assert EMSphInxNamelist.from_string(text).vendor == vendor

    @pytest.mark.parametrize("vendor", ["TSL", "Tsl", "bruker", "EMSoft", "Zeiss"])
    def test_vendor_whitelist_rejects(self, vendor):
        # ``"tsl"`` is lower case in the C++ condition and ``"TSL"``
        # is rejected although the template's comment advertises it
        text = ACID_NAMELIST.replace("'Bruker'", f"'{vendor}'")
        with pytest.raises(ValueError, match="unknown vendor"):
            EMSphInxNamelist.from_string(text)

    def test_extra_key_warns_with_its_name(self):
        # ``IndexEBSD.exe`` still indexes such a file, exit 0, with
        # this wording (``index_ebsd.cpp`` line 83, measured)
        text = ACID_NAMELIST.replace(
            " bw         = 68,\n", " bw = 68,\n extrakey = 42,\n"
        )
        with pytest.warns(UserWarning, match=re.escape(UNUSED_TOKEN_MESSAGE)):
            namelist = EMSphInxNamelist.from_string(text)
        assert namelist.bw == 68

    def test_the_acid_namelist_passes_its_sanity_check(self):
        acid_namelist().sanity_check()

    @pytest.mark.parametrize(
        "overrides, message",
        SANITY_FAILURES,
        ids=[f"{i}-{next(iter(o))}" for i, (o, _) in enumerate(SANITY_FAILURES)],
    )
    def test_sanity_check_bounds(self, overrides, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            namelist_with(**overrides).sanity_check()

    @pytest.mark.parametrize(
        "overrides",
        SANITY_LIMITS,
        ids=[f"{i}-{next(iter(o))}" for i, o in enumerate(SANITY_LIMITS)],
    )
    def test_sanity_check_limits_pass(self, overrides):
        namelist_with(**overrides).sanity_check()

    def test_read_runs_the_sanity_check(self):
        text = ACID_NAMELIST.replace(" bw         = 68,", " bw         = 8,")
        with pytest.raises(ValueError, match="unreasonable bandwidth"):
            EMSphInxNamelist.from_string(text)


# --------------------- Vendor conversions (D6) ---------------------- #


class TestVendorConversions:
    def test_the_nickel_geometry_triple(self):
        # the reference triple derives from the unrounded
        # ``pc_average``, so input and output come from the same
        # source and the comparison is approximate (D6)
        detector = kp.data.nickel_ebsd_small().detector
        pc = detector.pc_average
        geometry = _pctr_to_geometry("Bruker", tuple(pc), 60, 60, 500.0)
        assert geometry == pytest.approx(NI_GEOMETRY, rel=1e-9)
        assert tuple(np.round(pc, 8)) == ACID_PCTR

    @pytest.mark.parametrize("shape", [(48, 60), (60, 48)])
    @pytest.mark.parametrize("vendor", VENDORS)
    def test_conversion_table_rectangular(self, shape, vendor):
        w, h = width_height(shape)
        table = RECTANGULAR_TABLE[shape]
        pctr = table[vendor]
        geometry = _pctr_to_geometry(vendor, pctr, w, h, 500.0)
        assert geometry == pytest.approx(table["geometry"], rel=1e-12)
        back = _geometry_to_pctr(vendor, table["geometry"], w, h, 500.0)
        assert back == pytest.approx(pctr, rel=1e-12)

    @pytest.mark.parametrize("vendor", VENDORS)
    def test_conversion_table_square(self, vendor):
        # on a square detector the TSL and Oxford rows coincide
        table = {
            "EMsoft": (-4.494, 17.196, 15021.0),
            "EDAX": (0.4251, 0.7866, 0.5007),
            "tsl": (0.4251, 0.7866, 0.5007),
            "Oxford": (0.4251, 0.7866, 0.5007),
            "Bruker": RECTANGULAR_PC,
        }
        geometry = _pctr_to_geometry(vendor, table[vendor], 60, 60, 500.0)
        assert geometry == pytest.approx((-4.494, 17.196, 15021.0), rel=1e-12)

    @pytest.mark.parametrize("shape", [(60, 60), (48, 60), (60, 48)])
    def test_every_vendor_describes_the_same_detector(self, shape):
        w, h = width_height(shape)
        reference = _pctr_to_pc("Bruker", RECTANGULAR_PC, w, h, 500.0)
        for vendor in VENDORS:
            pctr = _pc_to_pctr(vendor, reference, w, h, 500.0)
            assert _pctr_to_pc(vendor, pctr, w, h, 500.0) == pytest.approx(
                reference, rel=1e-12
            )

    @pytest.mark.parametrize("shape", [(60, 60), (48, 60), (60, 48)])
    def test_bruker_is_the_identity_on_the_kikuchipy_pc(self, shape):
        # bitwise, which the documented formula composition is not:
        # measured, ``geometry -> pc`` of ``pctr -> geometry`` gives
        # 0.21340000000000003 back for 0.2134 on all three shapes and
        # 0.5006999999999999 for 0.5007 on (60, 48).  ``"Bruker"`` is
        # therefore a short circuit in both directions, as the two
        # helpers' docstrings state (D6)
        w, h = width_height(shape)
        pc = _pctr_to_pc("Bruker", RECTANGULAR_PC, w, h, 500.0)
        assert np.array_equal(pc, np.asarray(RECTANGULAR_PC))
        back = _pc_to_pctr("Bruker", np.asarray(RECTANGULAR_PC), w, h, 500.0)
        assert np.array_equal(np.asarray(back), np.asarray(RECTANGULAR_PC))

    @pytest.mark.parametrize("shape", [(60, 60), (48, 60), (60, 48)])
    def test_emsoft_equals_kikuchipy_pc_emsoft_version_4(self, shape):
        # exactly, and only under the precondition: kikuchipy
        # multiplies by ``binning`` and ``px_size`` while EMSphInx
        # uses the binned ``patdims`` and the namelist ``delta``
        w, h = width_height(shape)
        detector = ni_detector(shape=shape, pc=RECTANGULAR_PC, delta=500.0)
        pctr = _pc_to_pctr("EMsoft", np.asarray(RECTANGULAR_PC), w, h, 500.0)
        assert np.array_equal(np.asarray(pctr), detector.pc_emsoft(version=4).ravel())
        # version 5 has the opposite x sign
        assert not np.array_equal(
            np.asarray(pctr), detector.pc_emsoft(version=5).ravel()
        )

    def test_the_pc_emsoft_equality_needs_its_precondition(self):
        # the nickel fixture carries ``binning=8, px_size=1.0``,
        # which puts ``pc_emsoft(version=4)`` a factor of about eight
        # off the EMSphInx triple (measured, D6)
        detector = kp.data.nickel_ebsd_small().detector
        pc = detector.pc_average
        pctr = _pc_to_pctr("EMsoft", pc, 60, 60, 500.0)
        assert not np.allclose(
            np.asarray(pctr), detector.pc_emsoft(version=4).mean(axis=(0, 1))
        )

    @pytest.mark.parametrize("shape", [(60, 60), (48, 60), (60, 48)])
    def test_tsl_oxford_deviate_from_kikuchipy_on_rectangular(self, shape):
        # the two code bases disagree about which formula belongs to
        # which vendor name, so the port never delegates to the
        # kikuchipy helpers
        w, h = width_height(shape)
        detector = ni_detector(shape=shape, pc=RECTANGULAR_PC, delta=500.0)
        pc = np.asarray(RECTANGULAR_PC)

        emsphinx_tsl = np.asarray(_pc_to_pctr("EDAX", pc, w, h, 500.0))
        assert np.allclose(emsphinx_tsl, detector.pc_oxford().ravel())
        assert np.allclose(
            np.asarray(_pc_to_pctr("tsl", pc, w, h, 500.0)), emsphinx_tsl
        )

        emsphinx_oxford = np.asarray(_pc_to_pctr("Oxford", pc, w, h, 500.0))
        kikuchipy_tsl = detector.pc_tsl().ravel()
        assert np.allclose(emsphinx_oxford[:2], kikuchipy_tsl[:2])
        if OXFORD_EQUALS_KP_TSL[shape]:
            assert np.allclose(emsphinx_oxford[2], kikuchipy_tsl[2])
        else:
            assert not np.allclose(emsphinx_oxford[2], kikuchipy_tsl[2])

    @pytest.mark.parametrize("vendor", ["EDAX", "tsl", "Oxford", "Bruker"])
    def test_delta_invariant_for_fractional_vendors(self, vendor):
        # measured bitwise through the binary: a Bruker namelist at
        # ``delta`` 250 and at 500 gives identical Euler angles
        pcs = [
            _pctr_to_pc(vendor, RECTANGULAR_PC, 48, 60, delta)
            for delta in (250.0, 500.0)
        ]
        assert np.array_equal(pcs[0], pcs[1])

    def test_delta_is_live_for_emsoft(self):
        pcs = [
            _pctr_to_pc("EMsoft", RECTANGULAR_PC, 48, 60, delta)
            for delta in (250.0, 500.0)
        ]
        assert not np.allclose(pcs[0], pcs[1])

    @pytest.mark.parametrize("vendor", VENDORS)
    def test_pctr_pc_round_trip(self, vendor):
        pctr = RECTANGULAR_TABLE[(48, 60)][vendor]
        pc = _pctr_to_pc(vendor, pctr, 60, 48, 500.0)
        assert _pc_to_pctr(vendor, pc, 60, 48, 500.0) == pytest.approx(pctr, rel=1e-12)

    @pytest.mark.parametrize(
        "function",
        [_pctr_to_geometry, _geometry_to_pctr, _pctr_to_pc, _pc_to_pctr],
    )
    def test_unknown_vendor_raises(self, function):
        # all four, including the inverse: an unguarded branch there
        # would fall through to whatever the last ``elif`` left
        with pytest.raises(ValueError, match="vendor"):
            function("Zeiss", np.array([0.5, 0.5, 0.5]), 60, 60, 500.0)


# ------------- Arguments to and from the namelist (D6) -------------- #


class TestToFromKwargs:
    def test_kwargs_keys_are_live_parameters(self):
        kwargs = acid_namelist().to_kwargs()
        method = inspect.signature(kp.signals.EBSD.spherical_indexing).parameters
        indexer = inspect.signature(SphericalIndexer.__init__).parameters
        assert set(kwargs) == {
            "bandwidth",
            "normalize",
            "refine",
            "n_regions",
            "gaussian_background",
            "circular_mask",
            "chunksize",
        }
        for key in kwargs:
            assert key in method, key
            if key != "chunksize":
                assert key in indexer, key

    def test_kwargs_values(self):
        kwargs = acid_namelist().to_kwargs()
        assert kwargs["bandwidth"] == 68
        assert kwargs["normalize"] is True
        assert kwargs["refine"] is True
        assert kwargs["n_regions"] == 10
        assert kwargs["gaussian_background"] is False
        assert kwargs["circular_mask"] is False
        assert kwargs["chunksize"] == 1

    @pytest.mark.parametrize(
        "circ_rad, expected", [(0, True), (-1, False), (5, False), (30, False)]
    )
    def test_positive_circmask_maps_to_false_with_warning(self, circ_rad, expected):
        # a positive radius leaves ``Geometry::circ`` false but still
        # masks in the image processor at radius r, which kikuchipy
        # cannot express, so the mask is lost and named (D6)
        namelist = namelist_with(circ_rad=circ_rad)
        if circ_rad > 0:
            with pytest.warns(UserWarning, match="mask"):
                kwargs = namelist.to_kwargs()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                kwargs = namelist.to_kwargs()
        assert kwargs["circular_mask"] is expected

    def test_batchsize_zero_maps_to_none_chunksize(self):
        assert namelist_with(batch_size=0).to_kwargs()["chunksize"] is None
        assert namelist_with(batch_size=32).to_kwargs()["chunksize"] == 32

    def test_nthread_is_not_smuggled_into_the_kwargs(self):
        kwargs = namelist_with(n_thread=4).to_kwargs()
        assert "n_thread" not in kwargs
        assert "nthread" not in kwargs

    def test_roimask_nonempty_raises(self):
        with pytest.raises(ValueError, match="roimask"):
            namelist_with(roi_mask="0, 0, 2, 2").to_kwargs()

    def test_to_detector_requires_sample_tilt(self):
        # the namelist has no sample tilt; a silent default was
        # measured in Phase 6 to index about five degrees wrong at
        # *higher* scores
        with pytest.raises(TypeError):
            acid_namelist().to_detector()

    def test_to_detector_fields(self):
        namelist = namelist_with(pat_dims=(60, 48), thetac=3.0, delta=500.0)
        detector = namelist.to_detector(sample_tilt=70.0)
        assert isinstance(detector, EBSDDetector)
        assert detector.shape == (48, 60)
        assert detector.pc.shape == (1, 3)
        assert np.allclose(detector.pc_average, ACID_PCTR)
        assert detector.px_size == 500.0
        assert detector.tilt == 3.0
        assert detector.sample_tilt == 70.0
        assert detector.binning == 1

    @pytest.mark.parametrize("vendor", VENDORS)
    def test_to_detector_uses_the_vendor_conversion(self, vendor):
        pctr = RECTANGULAR_TABLE[(48, 60)][vendor]
        namelist = namelist_with(pat_dims=(60, 48), vendor=vendor, pctr=pctr)
        detector = namelist.to_detector(sample_tilt=70.0)
        assert np.allclose(detector.pc_average, RECTANGULAR_PC)

    # ------------------------- from_kwargs -------------------------- #

    def test_from_kwargs_bruker_pctr_is_pc_average_verbatim(self):
        detector = kp.data.nickel_ebsd_small().detector
        namelist = EMSphInxNamelist.from_kwargs(
            pattern_file="patterns.h5",
            master_files=["ni.sht"],
            detector=detector,
            scan_shape=(3, 3),
            scan_steps=(1.5, 1.5),
            data_file="out.h5",
        )
        assert namelist.vendor == "Bruker"
        assert np.array_equal(np.asarray(namelist.pctr), detector.pc_average)
        assert namelist.pat_dset == "patterns"

    def test_from_kwargs_averages_the_pattern_centres(self):
        # the namelist holds one pattern centre and the fixture
        # detector nine, so the averaging is the contract, not an
        # accident of a one pattern centre detector: a detector with
        # more than one is accepted (D6)
        detector = kp.data.nickel_ebsd_small().detector
        assert detector.pc.shape == (3, 3, 3)
        namelist = self.built(detector)
        assert np.array_equal(np.asarray(namelist.pctr), detector.pc_average)
        assert not np.array_equal(np.asarray(namelist.pctr), detector.pc[0, 0])

    def test_from_kwargs_delta_default_is_30mm_detector(self):
        # the detector's own ``px_size`` of 1.0 would be a 0.06 mm
        # detector, which the sanity check rejects (measured)
        detector = kp.data.nickel_ebsd_small().detector
        assert detector.px_size == 1.0
        namelist = self.built(detector)
        assert namelist.delta == 30000 / namelist.pat_dims[0]
        assert namelist.delta == 500.0
        assert namelist.delta * namelist.pat_dims[0] / 1000 == 30.0

    def test_from_kwargs_delta_can_be_given(self):
        namelist = self.built(kp.data.nickel_ebsd_small().detector, delta=250.0)
        assert namelist.delta == 250.0

    def test_from_kwargs_passes_its_own_sanity_check(self):
        self.built(kp.data.nickel_ebsd_small().detector).sanity_check()

    def test_from_kwargs_rejects_azimuthal_twist(self):
        for field in ("azimuthal", "twist"):
            detector = kp.data.nickel_ebsd_small().detector.deepcopy()
            setattr(detector, field, 5.0)
            with pytest.raises(ValueError, match=field):
                self.built(detector)

    def test_from_kwargs_patdims_order(self):
        detector = EBSDDetector(shape=(48, 60), pc=RECTANGULAR_PC)
        namelist = self.built(detector)
        assert tuple(namelist.pat_dims) == (60, 48)

    def test_from_kwargs_scandims_order(self):
        # x then y, the reverse of kikuchipy's navigation shape; a
        # square scan cannot catch the transposition
        namelist = self.built(kp.data.nickel_ebsd_small().detector, scan_shape=(2, 3))
        assert tuple(namelist.scan_dims) == (3, 2)

    def test_from_kwargs_scan_steps_are_x_then_y(self):
        namelist = self.built(
            kp.data.nickel_ebsd_small().detector, scan_steps=(1.5, 2.5)
        )
        assert tuple(namelist.scan_steps) == (1.5, 2.5)

    def test_from_kwargs_thread_batch_passthrough(self):
        namelist = self.built(
            kp.data.nickel_ebsd_small().detector, n_thread=1, batch_size=1
        )
        assert namelist.n_thread == 1
        assert namelist.batch_size == 1
        assert self.built(kp.data.nickel_ebsd_small().detector).n_thread == 0
        assert self.built(kp.data.nickel_ebsd_small().detector).batch_size == 0

    @pytest.mark.parametrize("circular_mask, circ_rad", [(True, 0), (False, -1)])
    def test_from_kwargs_circular_mask_inverse(self, circular_mask, circ_rad):
        namelist = self.built(
            kp.data.nickel_ebsd_small().detector, circular_mask=circular_mask
        )
        assert namelist.circ_rad == circ_rad
        assert namelist.to_kwargs()["circular_mask"] is circular_mask

    def test_from_kwargs_output_names_default_empty(self):
        # the ``defaults()`` names would make every run write PNG
        # maps; an empty quality map also exercises the file without
        # a terminator
        namelist = self.built(kp.data.nickel_ebsd_small().detector)
        assert namelist.vendor_file == ""
        assert namelist.ipf_name == ""
        assert namelist.qual_name == ""
        assert namelist.to_string().splitlines()[-1] != " /"

    def test_from_kwargs_indexing_defaults_are_the_method_defaults(self):
        namelist = self.built(kp.data.nickel_ebsd_small().detector)
        parameters = inspect.signature(kp.signals.EBSD.spherical_indexing).parameters
        assert namelist.bw == parameters["bandwidth"].default
        assert namelist.normed is parameters["normalize"].default
        assert namelist.refine is parameters["refine"].default
        assert namelist.n_regions == parameters["n_regions"].default
        assert namelist.gaus_bckg is parameters["gaussian_background"].default

    def test_from_kwargs_takes_the_indexing_kwargs(self):
        namelist = self.built(
            kp.data.nickel_ebsd_small().detector,
            bandwidth=88,
            normalize=False,
            refine=False,
            n_regions=4,
            gaussian_background=True,
        )
        assert namelist.bw == 88
        assert namelist.normed is False
        assert namelist.refine is False
        assert namelist.n_regions == 4
        assert namelist.gaus_bckg is True

    @pytest.mark.parametrize("vendor", VENDORS)
    def test_from_kwargs_every_vendor_describes_the_detector(self, vendor):
        detector = EBSDDetector(
            shape=(48, 60), pc=RECTANGULAR_PC, px_size=500.0, binning=1
        )
        namelist = self.built(detector, vendor=vendor)
        assert namelist.vendor == vendor
        assert np.allclose(
            namelist.to_detector(sample_tilt=70.0).pc_average, RECTANGULAR_PC
        )

    def test_from_kwargs_rejects_an_unknown_vendor(self):
        with pytest.raises(ValueError, match="vendor"):
            self.built(kp.data.nickel_ebsd_small().detector, vendor="TSL")

    @pytest.mark.parametrize("chunksize, batch_size", [(None, 0), (1, 1), (32, 32)])
    def test_from_kwargs_chunksize_maps_to_batchsize(self, chunksize, batch_size):
        # ``chunksize`` rides in ``**indexing_kwargs`` and is the only
        # one of them which does not name a namelist field of its own;
        # it is the inverse of ``to_kwargs``' ``batch_size or None``
        namelist = self.built(kp.data.nickel_ebsd_small().detector, chunksize=chunksize)
        assert namelist.batch_size == batch_size
        assert namelist.to_kwargs()["chunksize"] == chunksize

    def test_from_kwargs_accepts_its_own_to_kwargs(self):
        # the documented round trip of the two argument converters,
        # which is how Phase 10 rebuilds a namelist from a run
        namelist = self.built(kp.data.nickel_ebsd_small().detector, n_thread=1)
        again = EMSphInxNamelist.from_kwargs(
            pattern_file=namelist.pat_file,
            master_files=namelist.master_files,
            detector=namelist.to_detector(sample_tilt=70.0),
            scan_shape=tuple(namelist.scan_dims)[::-1],
            scan_steps=tuple(namelist.scan_steps),
            data_file=namelist.data_file,
            n_thread=namelist.n_thread,
            **namelist.to_kwargs(),
        )
        assert_fields_equal(again, namelist, pctr_rel=1e-12)

    def test_from_kwargs_round_trips_through_the_file(self):
        # the ``pc_average`` pattern centre carries more than six
        # significant digits, which ``to_string`` rounds (D5)
        namelist = self.built(kp.data.nickel_ebsd_small().detector, n_thread=1)
        again = EMSphInxNamelist.from_string(namelist.to_string())
        assert_fields_equal(again, namelist, pctr_rel=1e-6)

    @staticmethod
    def built(detector, **overrides):
        """Return a namelist built from a detector and the acid
        configuration, with fields replaced.
        """
        parameters = dict(
            pattern_file="patterns.h5",
            master_files=["ni.sht"],
            detector=detector,
            scan_shape=(3, 3),
            scan_steps=(1.5, 1.5),
            data_file="out.h5",
        )
        parameters.update(overrides)
        return EMSphInxNamelist.from_kwargs(**parameters)


# -------------- The EMSphInx binaries (D9, local only) -------------- #


class TestEmsphinxBinaries:
    def test_emsphinx_binaries_index_ebsd_template_matches_ours(
        self, emsphinx_program, tmp_path
    ):
        # the live twin of the frozen template fixture.  ``-t``
        # writes to the hard coded relative "IndexEBSD.nml", so the
        # run must happen in ``tmp_path`` (D9)
        program = emsphinx_program("IndexEBSD")
        result = subprocess.run(
            [str(program), "-t"], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        written = (tmp_path / "IndexEBSD.nml").read_text(encoding="utf-8")
        assert written.splitlines() == INDEX_EBSD_TEMPLATE.splitlines()
        assert EMSphInxNamelist.defaults().to_string().splitlines() == (
            written.splitlines()
        )

    def test_emsphinx_binaries_the_template_parses_back(
        self, emsphinx_program, tmp_path
    ):
        program = emsphinx_program("IndexEBSD")
        subprocess.run([str(program), "-t"], cwd=tmp_path, capture_output=True)
        fpath = tmp_path / "IndexEBSD.nml"
        assert EMSphInxNamelist.read(fpath) == EMSphInxNamelist.defaults()
