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
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517):
# - the typed ``Variant`` accessors and setters, ``NameList::read()``,
#   ``NameList::at()``, the ``get___s()`` vector accessors,
#   ``fullyParsed()`` and ``unusedTokens()`` (``include/util/
#   nml.hpp``, lines 226-520), as :class:`_NameList`
# - the ``ebsd::Namelist`` fields (``include/modality/ebsd/nml.hpp``,
#   lines 52-160), ``clear()``/``defaults()`` (lines 186-218),
#   ``parse_nml()`` (lines 236-315), ``to_string()`` (lines 320-470)
#   and ``sanityCheck()`` (lines 621-639), as
#   :class:`EMSphInxNamelist`
# - ``Geometry<Real>::patternCenter()`` (``include/modality/ebsd/
#   detector.hpp``, line 85), ``patternCenterTSL()`` (lines 249-254),
#   ``patternCenterOxford()`` (lines 261-267) and
#   ``patternCenterBruker()`` (lines 274-279), with the vendor switch
#   of ``include/modality/ebsd/idx.hpp`` lines 221-229, as the
#   pattern centre conversions
# - the geometry and image processor bindings of ``idx.hpp`` lines
#   218-231 and 254 with ``PatternProcessor<Real>::setSize()``
#   (``include/modality/ebsd/imprc.hpp``, lines 108-122), as
#   :meth:`EMSphInxNamelist.to_kwargs` and
#   :meth:`EMSphInxNamelist.to_detector`
#
# **Deliberate deviations from the ported code**, all documented in
# the class:
# - paths are stored exactly as written in the file, and the
#   ``ipath`` prefixed forms are the derived read only properties
#   ``pat_path``/``master_paths``.  The C++ stores the prefixed
#   values (lines 241, 247-248, 252) and therefore double prefixes
#   on its own ``from_string(to_string(x))``; the port keeps the
#   round trip exact and reproduces the double ``ipath`` of line 247
#   on the derived property instead.
# - ``patdset`` and ``scanname`` are always optional and always
#   consumed.  The C++ reads them only when ``H5::H5File::isHdf5()``
#   says the pattern file exists and is HDF5 (lines 242-246,
#   253-254), i.e. its requiredness depends on the file system,
#   which a pure parser must not reproduce.
#
# The following are deliberately **not** ported here:
# - the scan file route of ``parse_nml()`` (a string valued
#   ``scandims`` raises ``NotImplementedError``), i.e.
#   ``readScanFile()``/``findScanFile()`` and
#   ``xtal/orientation_map.hpp``
# - the region of interest grammar of ``idx/roi.h`` (``roimask`` is
#   parsed, stored and written back as an opaque string)
# - ``NML_USE_H5``, i.e. ``NameList::writeParameters()`` and
#   ``writeFile()`` (``nml.hpp`` lines 525-575), which serve the
#   output scan file

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
# Changed by Johan Westraadt, 2026-09: translated to Python for
# kikuchipy. GPL-2.0-or-later, conveyed under GPL-3.0-or-later
# #####################################################################

"""The ``IndexEBSD`` namelist file of EMSphInx: its grammar, its EBSD
field set and the conversions to and from kikuchipy's spherical
indexing arguments.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**The grammar** (``NameList::read()``), ported quirk for quirk:

* the first line is skipped and must not contain ``=``;
* a comment line starts with ``!`` **as the first character of the
  line** (the code tests ``line.front()``, while its own doc comment
  claims the first character after white space: an indented ``!``
  line raises);
* empty lines and the exact line ``" /"`` are skipped, but a white
  space only line is **not** (the test is a literal ``empty()``);
* every key line starts with exactly **one** space: zero spaces or a
  tab raise "missing leading space ...", two or more raise the
  different "error parsing line ...";
* keys are lower cased and may not repeat, ``=`` is the delimiter,
  and every entry line except the last must end with a comma;
* values are either a comma separated list of single quoted strings
  (with ``\\'`` escapes) or a comma separated list of
  ``.true.``/``.false.``, integers and doubles, where a mixed
  integer/double list is promoted to doubles and a boolean mixed
  with a number raises;
* a number token is a **whole** stream extraction, i.e. C's
  ``strtod`` or ``strtol``: ``nan``, ``inf``, a digit separator and
  an overflowing or underflowing value are all "couldn't parse
  token", while a hexadecimal literal is a **double** and so
  refused by an integer field (all measured through the binary);
* from the **second** string of a quoted list onward all white space
  inside the string is stripped, because the C++ sets a sticky
  ``std::skipws`` before the inter-string delimiters.  The writer
  guards against this.

**The types are strict** as the C++ is: ``get_int`` refuses a double
while ``get_double`` silently accepts an integer.

The pattern centre conversions are EMSphInx's own formulas, **not**
kikuchipy's ``pc_tsl``/``pc_oxford`` helpers, which disagree with
them on rectangular detectors.  With ``w = pat_dims[0]``,
``h = pat_dims[1]`` and ``d = delta``, the internal geometry triple
``(cX, cY, sDst)`` is

.. code-block::

    vendor      cX                cY                sDst
    EMsoft      p0                p1                p2
    EDAX, tsl   p0 w - w/2        p1 w - h/2        p2 w d
    Oxford      (p0 - 0.5) w      (p1 - 0.5) h      p2 w d
    Bruker      (p0 - 0.5) w      (0.5 - p1) h      p2 h d

and kikuchipy's (Bruker) pattern centre follows as
``PCx = cX/w + 0.5``, ``PCy = 0.5 - cY/h``, ``PCz = sDst/(h d)``, so
the Bruker row is the identity.  That identity is **algebraic**:
composing the two formulas drifts by an unit in the last place, so
:func:`_pctr_to_pc` and :func:`_pc_to_pctr` short circuit
``"Bruker"`` and return the input unchanged.  ``delta`` cancels
exactly for the three fractional vendors and is live only for
``EMsoft``.
"""

import inspect
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors import EBSDDetector

__all__ = ["EMSphInxNamelist"]

# The vendor whitelist of ``parse_nml()`` (lines 290-295).  ``"tsl"``
# is lower case there and ``"TSL"`` is rejected, although the
# template's own comment advertises it.
VENDORS = ("EMsoft", "EDAX", "tsl", "Oxford", "Bruker")

# ``sanityCheck()`` bounds (lines 621-639)
_PAT_DIM_LIMITS = (2, 16384)
_DETECTOR_WIDTH_LIMITS_MM = (5.0, 90.0)
_THETAC_LIMITS = (-60.0, 60.0)
_BANDWIDTH_LIMITS = (16, 512)

# The section banner of ``to_string()``
_BANNER = "!#################################################################\n"

# ``int32_t`` range of the namelist's integer fields, which decides
# whether ``tryParse<int>()`` succeeds before ``tryParse<double>()``
_INT_LIMITS = (-(2**31), 2**31 - 1)

# The number token grammars of ``detail::tryParse<T>()``, which
# extracts the whole token from a stream: C's ``strtod`` for a double
# (decimal or hexadecimal, with no ``nan``/``inf`` spelling and no
# digit separator) and a decimal ``strtol`` for an integer.  Probed
# with 60 tokens through ``IndexEBSD.exe``.
_DOUBLE_TOKEN = re.compile(
    r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?"
    r"|0x(?:[0-9a-f]+\.?[0-9a-f]*|\.[0-9a-f]+)(?:p[+-]?[0-9]+)?)\Z",
    re.IGNORECASE,
)
_INT_TOKEN = re.compile(r"[+-]?[0-9]+\Z")

# The arguments of :meth:`~kikuchipy.signals.EBSD.spherical_indexing`
# the namelist maps to, in the order :meth:`to_kwargs` returns them
_INDEXING_KEYS = (
    "bandwidth",
    "normalize",
    "refine",
    "n_regions",
    "gaussian_background",
    "circular_mask",
    "chunksize",
)


def _indexing_defaults() -> dict[str, Any]:
    """Return the defaults of the indexing arguments the namelist
    holds, read from the live signature.
    """
    from kikuchipy.signals import EBSD

    parameters = inspect.signature(EBSD.spherical_indexing).parameters
    return {key: parameters[key].default for key in _INDEXING_KEYS}


# ------------------------ The file grammar -------------------------- #


def _file_lines(text: str) -> list[str]:
    """Return the lines ``std::getline()`` reads from ``text``.

    A trailing line feed does not make an extra empty line, as it
    does for :meth:`str.split`.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _extract_key(line: str) -> tuple[str | None, str]:
    """Return the key and delimiter of a key line, or ``(None, "")``
    when the C++ chained extraction fails.

    The extraction is ``iss >> noskipws >> space >> key >> skipws >>
    delim``: the first character is the leading space, the key runs
    to the next white space and must not be empty, and the delimiter
    is the next non white space character.
    """
    start = 1
    end = start
    while end < len(line) and not line[end].isspace():
        end += 1
    key = line[start:end]
    if key == "":
        return None, ""
    while end < len(line) and line[end].isspace():
        end += 1
    if end >= len(line):
        return None, ""
    return key, line[end]


def _rest_of_line(line: str, key: str) -> str:
    """Return what follows the delimiter of a key line, with the
    white space after it skipped.
    """
    index = 1 + len(key)
    while index < len(line) and line[index].isspace():
        index += 1
    return line[index + 1 :].lstrip()


def _next_character(text: str, index: int) -> tuple[str | None, int]:
    """Return the next non white space character and the index after
    it, or ``(None, index)`` at the end, as ``iss >> skipws >> c``.
    """
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None, index
    return text[index], index + 1


def _parse_strings(text: str, number: int, line: str) -> list[str]:
    """Return the single quoted strings of a value, the string branch
    of ``NameList::read()`` (lines 340-370).

    ``text`` starts after the opening quote.  From the second string
    onward all white space inside the string is stripped, because the
    C++ sets a sticky ``std::skipws`` before the inter string
    delimiters.
    """
    values = []
    index = 0
    skip_space = False
    while True:
        token = ""
        found = False
        while index < len(text):
            character = text[index]
            index += 1
            if skip_space and character.isspace():
                continue
            if character == "'":
                if token == "" or token[-1] != "\\":
                    found = True
                    break
                token = token[:-1]
            token += character
        if not found:
            raise ValueError(f'no closing quote for a token in line {number} "{line}"')
        values.append(token)

        skip_space = True
        delimiter, index = _next_character(text, index)
        if delimiter is None:
            break
        if delimiter != ",":
            raise ValueError(
                f'unexpected delimiter between strings in line {number} "{line}"'
            )
        delimiter, index = _next_character(text, index)
        if delimiter is None:
            break
        if delimiter != "'":
            raise ValueError(
                f'unexpected delimiter string opening in line {number} "{line}"'
            )
    return values


def _comma_tokens(text: str) -> list[str]:
    """Return the comma separated tokens ``std::getline(iss, tok,
    ',')`` yields, i.e. without a final empty one.
    """
    tokens = text.split(",")
    if tokens and tokens[-1] == "":
        tokens.pop()
    return tokens


def _try_parse_double(token: str) -> float | None:
    """Return the token parsed as a double, or ``None``.

    ``detail::tryParse<double>()`` (``nml.hpp`` lines 274-278)
    extracts the **whole** token from a stream, which is C's
    ``strtod``: a decimal or a ``0x`` hexadecimal literal and
    nothing else.  :func:`float` is a different grammar, so this
    refuses three kinds of token it takes, as the program refuses
    them (measured): ``nan``/``inf``, a token with a digit separator
    such as ``1_000``, and a value which overflows or underflows the
    double, such as ``1e400`` and ``1e-400``, which sets the
    stream's fail bit.  A subnormal such as ``1e-320`` parses.  It
    also takes one kind :func:`float` refuses, the hexadecimal
    literal, which is why ``bw = 0x44`` is a *double*.
    """
    if _DOUBLE_TOKEN.match(token) is None:
        return None
    mantissa = token.lstrip("+-")
    if mantissa[:2].lower() == "0x":
        try:
            value = float.fromhex(token)
        except (ValueError, OverflowError):
            return None
        mantissa = re.split("[pP]", mantissa[2:])[0]
    else:
        value = float(token)
        mantissa = re.split("[eE]", mantissa)[0]
    if not np.isfinite(value):
        return None
    if value == 0.0 and any(character not in "0." for character in mantissa):
        return None
    return value


def _try_parse_int(token: str) -> int | None:
    """Return the token parsed as a 32-bit integer, or ``None``.

    Decimal only, as the stream extraction of
    ``detail::tryParse<int>()`` is: a hexadecimal literal parses as
    the double instead, so ``bw = 0x44`` is a "stored type isn't
    integer" and not a bandwidth of 68 (measured).  Integers out of
    range fail to parse in the C++ too, which then falls back to the
    double.
    """
    if _INT_TOKEN.match(token) is None:
        return None
    value = int(token)
    smallest, largest = _INT_LIMITS
    if value < smallest or value > largest:
        return None
    return value


def _parse_numbers(text: str, number: int, line: str) -> list:
    """Return the booleans, integers and doubles of a value, the
    number branch of ``NameList::read()`` (lines 371-419).
    """
    values = []
    has_bool = has_int = has_double = False
    for token in _comma_tokens(text):
        token = "".join(c for c in token if not c.isspace()).lower()
        if token == "":
            continue
        if token == ".true.":
            values.append(True)
            has_bool = True
        elif token == ".false.":
            values.append(False)
            has_bool = True
        else:
            as_double = _try_parse_double(token)
            if as_double is None:
                raise ValueError(
                    f'couldn\'t parse token "{token}" from line {number} '
                    f'"{line}" as bool, int, or float (strings must be in '
                    "single quotes, e.g. key = 'value')"
                )
            as_int = _try_parse_int(token)
            if as_int is None:
                values.append(as_double)
                has_double = True
            else:
                values.append(as_int)
                has_int = True

    if (has_int or has_double) and has_bool:
        raise ValueError(f'line {number} "{line}" has a mix of numbers and booleans')
    if has_int and has_double:
        values = [float(value) for value in values]
    return values


class _NameList:
    """Parsed EMSphInx namelist file: lower case keys to lists of
    booleans, integers, doubles or strings, with a used flag each.

    A port of ``nml::NameList`` (``include/util/nml.hpp``).  Not
    public: the public surface is :class:`EMSphInxNamelist`, and no
    other EMSphInx namelist is in scope.

    Parameters
    ----------
    values
        Mapping of lower case key to the list of parsed values.  Not
        normally given: use :meth:`from_string` or :meth:`read`.
    lines
        Raw lines of the parsed file, kept for round trip
        introspection.

    Attributes
    ----------
    lines : list of str
        Raw lines of the parsed file, the first one included.
    """

    def __init__(
        self,
        values: dict[str, list] | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self.lines: list[str] = [] if lines is None else list(lines)
        self._values: dict[str, list] = {}
        self._used: dict[str, list[bool]] = {}
        if values is not None:
            for key, value in values.items():
                self._values[key] = list(value)
                self._used[key] = [False] * len(value)

    # ------------------------- Constructors ------------------------- #

    @classmethod
    def from_string(cls, text: str, comments: str = "!") -> "_NameList":
        """Return a namelist parsed from the contents of a file.

        Parameters
        ----------
        text
            File contents.  The first line is skipped and must not
            contain ``=``.
        comments
            Characters starting a comment line, ``"!"`` by default.
            A line is a comment only when its **first** character is
            one of them.

        Returns
        -------
        namelist

        Raises
        ------
        ValueError
            On any of the grammar violations described in the module
            documentation, with the C++ message.
        """
        lines = _file_lines(text)
        first = lines[0] if lines else ""
        if first != "" and "=" in first:
            raise ValueError(
                "namelist files cannot have key value pairs in the first line"
            )

        values: dict[str, list] = {}
        got_comma = True
        for number, line in enumerate(lines[1:], start=2):
            if line == "":
                continue
            if line[0] in comments:
                continue
            if line == " /":
                continue

            if not got_comma:
                raise ValueError(
                    "missing comma between previous entry and namelist "
                    f'line {number} "{line}"'
                )
            # the C++ scans back over trailing white space for the
            # comma; on a white space only line that scan runs off the
            # front and leaves the stream in a failed state
            stripped = line.rstrip()
            if stripped == "":
                raise ValueError(f"error parsing line '{line}' from name list")
            got_comma = stripped[-1] == ","

            key, delimiter = _extract_key(line)
            if key is None:
                raise ValueError(f"error parsing line '{line}' from name list")
            if line[0] != " ":
                raise ValueError(
                    f'missing leading space in namelist line {number} "{line}"'
                )
            lowered = key.lower()
            if lowered in values:
                raise ValueError(f'key "{lowered}" was defined twice in the name list')
            if delimiter != "=":
                raise ValueError(
                    f"bad delimeter (expected '=') in namelist line {number} \"{line}\""
                )

            rest = _rest_of_line(line, key)
            if rest.startswith("'"):
                values[lowered] = _parse_strings(rest[1:], number, line)
            else:
                values[lowered] = _parse_numbers(rest, number, line)
        return cls(values, lines)

    @classmethod
    def read(cls, filename: str | Path, comments: str = "!") -> "_NameList":
        """Return a namelist parsed from a file.

        Parameters
        ----------
        filename
            Path to the namelist file.
        comments
            Characters starting a comment line, ``"!"`` by default.

        Returns
        -------
        namelist
        """
        text = Path(filename).read_text(encoding="utf-8")
        return cls.from_string(text, comments)

    # -------------------------- Accessors --------------------------- #

    def _at(self, key: str) -> str:
        """Return the stored key of ``key``, which is case
        insensitive.
        """
        lowered = key.lower()
        if lowered not in self._values:
            raise ValueError(f"couldn't find `{key}' in namelist")
        return lowered

    def _get(self, key: str, kind: str, index: int) -> Any:
        """Return one stored value, marking it used, with the C++
        type strictness.

        A key whose value list is empty, which ``" key = ,"`` gives,
        is reported as a missing key.  The C++ indexes its empty
        ``Value`` unchecked, which is undefined behaviour: measured,
        ``IndexEBSD.exe`` exits 3221225477, an access violation.
        """
        lowered = self._at(key)
        values = self._values[lowered]
        if len(values) == 0:
            raise ValueError(f"couldn't find `{key}' in namelist")
        value = values[index]
        is_bool = isinstance(value, bool)
        if kind == "boolean":
            good = is_bool
        elif kind == "integer":
            good = isinstance(value, int) and not is_bool
        elif kind == "double":
            good = isinstance(value, float) or (isinstance(value, int) and not is_bool)
        else:
            good = isinstance(value, str)
        if not good:
            raise ValueError(f"stored type isn't {kind}")
        self._used[lowered][index] = True
        return float(value) if kind == "double" else value

    def _get_all(self, key: str, kind: str) -> list:
        """Return every stored value of ``key``, marking them used."""
        lowered = self._at(key)
        return [
            self._get(lowered, kind, index)
            for index in range(len(self._values[lowered]))
        ]

    def get_bool(self, key: str) -> bool:
        """Return the single boolean stored under ``key``."""
        return self._get(key, "boolean", 0)

    def get_int(self, key: str) -> int:
        """Return the single integer stored under ``key``.

        A double raises, as ``Variant::getInt()`` does.
        """
        return self._get(key, "integer", 0)

    def get_double(self, key: str) -> float:
        """Return the single double stored under ``key``.

        An integer is accepted and cast, as ``Variant::getDouble()``
        does.
        """
        return self._get(key, "double", 0)

    def get_string(self, key: str) -> str:
        """Return the single string stored under ``key``."""
        return self._get(key, "string", 0)

    def get_bools(self, key: str) -> list[bool]:
        """Return the booleans stored under ``key``."""
        return self._get_all(key, "boolean")

    def get_ints(self, key: str) -> list[int]:
        """Return the integers stored under ``key``."""
        return self._get_all(key, "integer")

    def get_doubles(self, key: str) -> list[float]:
        """Return the doubles stored under ``key``, integers cast."""
        return self._get_all(key, "double")

    def get_strings(self, key: str) -> list[str]:
        """Return the strings stored under ``key``."""
        return self._get_all(key, "string")

    # ----------------------- Used flag queries ---------------------- #

    def fully_parsed(self) -> bool:
        """Return whether every stored value has been read."""
        return all(all(flags) for flags in self._used.values())

    def unused_tokens(self) -> str:
        """Return the comma joined keys which have not been read.

        Sorted, as the C++ ``std::map`` order is.
        """
        unused = [key for key, flags in self._used.items() if not all(flags)]
        return ",".join(sorted(unused))


class EMSphInxNamelist:
    """Namelist file of EMSphInx's ``IndexEBSD`` program
    :cite:`lenthe2019spherical`.

    Reads and writes the file, and converts its fields to and from
    the arguments of
    :meth:`~kikuchipy.signals.EBSD.spherical_indexing` and an
    :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    ipath
        Input path prefixed to the pattern, master and scan file
        names by the program.  Empty by default.
    pat_file
        Pattern file name, as written in the file.
    pat_dset
        HDF5 path of the patterns within ``pat_file``.
    master_files
        Master pattern file names, as written in the file.
    psym_file
        Pseudo symmetry file name.
    pat_dims
        Binned pattern ``(width, height)`` in pixels, i.e. the
        reverse of a kikuchipy detector shape.
    circ_rad
        Circular mask radius: ``-1`` for no mask, ``0`` for the
        largest inscribed circle and a positive radius in pixels.
    gaus_bckg
        Whether a two dimensional Gaussian background is subtracted.
    n_regions
        Number of adaptive histogram equalisation regions, ``0`` for
        no equalisation.
    delta
        Unbinned pixel size on the scintillator in microns.
    vendor
        Convention of ``pctr``, one of ``"EMsoft"``, ``"EDAX"``,
        ``"tsl"`` (lower case only), ``"Oxford"`` and ``"Bruker"``.
    pctr
        Pattern centre in the ``vendor`` convention.
    thetac
        Camera tilt in degrees.
    scan_dims
        Scan ``(width, height)`` in pixels, i.e. the reverse of a
        kikuchipy navigation shape.
    scan_steps
        Scan step ``(x, y)`` in microns.
    scan_file
        Scan file to read the dimensions from.  Never populated by
        :meth:`read`, which raises on a string valued ``scandims``.
    scan_name
        HDF5 path of the scan data within ``scan_file``.
    roi_mask
        Region of interest string, stored opaquely.
    bw
        Spherical harmonic bandwidth.
    normed
        Whether the normalised cross correlation is used.
    refine
        Whether Newton refinement is used.
    n_thread
        Number of work threads, ``0`` for automatic.  Has no
        kikuchipy equivalent.
    batch_size
        Number of patterns per work item, ``0`` for automatic.
    opath
        Output path prefix.
    data_file
        Output HDF5 orientation map name.
    vendor_file
        Output ``.ang`` or ``.ctf`` name, empty for none.
    ipf_name
        Output inverse pole figure map name, empty for none.
    qual_name
        Output quality map name, empty for none.

    Attributes
    ----------
    ipath : str
        Input path prefixed to the file names by the program.
    pat_file : str
        Pattern file name, as written in the file.
    pat_dset : str
        HDF5 path of the patterns within :attr:`pat_file`.
    master_files : list of str
        Master pattern file names, as written in the file.
    psym_file : str
        Pseudo symmetry file name.
    pat_dims : tuple of int
        Binned pattern ``(width, height)`` in pixels.
    circ_rad : int
        Circular mask radius: ``-1`` for no mask, ``0`` for the
        largest inscribed circle, positive for a radius in pixels.
    gaus_bckg : bool
        Whether a two dimensional Gaussian background is subtracted.
    n_regions : int
        Number of adaptive histogram equalisation regions.
    delta : float
        Unbinned pixel size on the scintillator in microns.
    vendor : str
        Convention of :attr:`pctr`, one of :data:`VENDORS`.
    pctr : tuple of float
        Pattern centre in the :attr:`vendor` convention.
    thetac : float
        Camera tilt in degrees.
    scan_dims : tuple of int
        Scan ``(width, height)`` in pixels.
    scan_steps : tuple of float
        Scan step ``(x, y)`` in microns.
    scan_file : str
        Scan file to read the dimensions from.
    scan_name : str
        HDF5 path of the scan data within :attr:`scan_file`.
    roi_mask : str
        Region of interest string, stored opaquely.
    bw : int
        Spherical harmonic bandwidth.
    normed : bool
        Whether the normalised cross correlation is used.
    refine : bool
        Whether Newton refinement is used.
    n_thread : int
        Number of work threads, ``0`` for automatic.
    batch_size : int
        Number of patterns per work item, ``0`` for automatic.
    opath : str
        Output path prefix.
    data_file : str
        Output HDF5 orientation map name.
    vendor_file : str
        Output ``.ang`` or ``.ctf`` name, empty for none.
    ipf_name : str
        Output inverse pole figure map name, empty for none.
    qual_name : str
        Output quality map name, empty for none.

    See Also
    --------
    kikuchipy.indexing.write_emsphinx_patterns

    Notes
    -----
    The defaults of this constructor are ``Namelist::clear()``, not
    ``Namelist::defaults()``: use :meth:`defaults` for the latter,
    which is what ``IndexEBSD -t`` writes.

    Paths are stored **exactly as written in the file**.  The
    program prefixes them with ``ipath`` at parse time and writes the
    prefixed values back, so its own read/write round trip doubles
    the prefix; here the prefixed forms are the derived properties
    :attr:`pat_path` and :attr:`master_paths`, on which the double
    prefix of a set ``psym_file`` remains observable.
    :meth:`sanity_check` follows the storage and tests the raw
    :attr:`pat_file`, so an empty one with a non-empty :attr:`ipath`
    raises here and does not in the program, which tests the
    prefixed name.

    ``patdset`` and ``scanname`` are always optional and always
    consumed, unlike the program, whose requiredness for them
    depends on whether the pattern file exists and is HDF5.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> nml = kp.indexing.EMSphInxNamelist.defaults()
    >>> nml.bw
    68
    """

    def __init__(
        self,
        *,
        ipath: str = "",
        pat_file: str = "",
        pat_dset: str = "",
        master_files: list[str] | None = None,
        psym_file: str = "",
        pat_dims: tuple[int, int] = (-1, -1),
        circ_rad: int = -1,
        gaus_bckg: bool = False,
        n_regions: int = 0,
        delta: float = np.nan,
        vendor: str = "EMsoft",
        pctr: tuple[float, float, float] = (np.nan, np.nan, np.nan),
        thetac: float = np.nan,
        scan_dims: tuple[int, int] = (-1, -1),
        scan_steps: tuple[float, float] = (np.nan, np.nan),
        scan_file: str = "",
        scan_name: str = "",
        roi_mask: str = "",
        bw: int = -1,
        normed: bool = False,
        refine: bool = False,
        n_thread: int = 0,
        batch_size: int = 0,
        opath: str = "",
        data_file: str = "",
        vendor_file: str = "",
        ipf_name: str = "",
        qual_name: str = "",
    ) -> None:
        self.ipath = ipath
        self.pat_file = pat_file
        self.pat_dset = pat_dset
        self.master_files = [] if master_files is None else list(master_files)
        self.psym_file = psym_file
        self.pat_dims = tuple(pat_dims)
        self.circ_rad = circ_rad
        self.gaus_bckg = gaus_bckg
        self.n_regions = n_regions
        self.delta = delta
        self.vendor = vendor
        self.pctr = tuple(pctr)
        self.thetac = thetac
        self.scan_dims = tuple(scan_dims)
        self.scan_steps = tuple(scan_steps)
        self.scan_file = scan_file
        self.scan_name = scan_name
        self.roi_mask = roi_mask
        self.bw = bw
        self.normed = normed
        self.refine = refine
        self.n_thread = n_thread
        self.batch_size = batch_size
        self.opath = opath
        self.data_file = data_file
        self.vendor_file = vendor_file
        self.ipf_name = ipf_name
        self.qual_name = qual_name

    # -------------------------- Properties -------------------------- #

    @property
    def pat_path(self) -> str:
        """Return the pattern file name as the program opens it.

        This is ``ipath + pat_file``, with ``ipath`` **twice** when
        :attr:`psym_file` is not empty, a ported quirk of
        ``parse_nml()`` line 247.
        """
        path = self.ipath + self.pat_file
        if self.psym_file != "":
            path = self.ipath + path
        return path

    @property
    def master_paths(self) -> list[str]:
        """Return the master file names as the program opens them,
        i.e. each prefixed with :attr:`ipath`.
        """
        return [self.ipath + name for name in self.master_files]

    # ------------------------- Constructors ------------------------- #

    @classmethod
    def defaults(cls) -> "EMSphInxNamelist":
        """Return the namelist ``Namelist::defaults()`` builds, which
        is what ``IndexEBSD -t`` writes.

        Returns
        -------
        namelist
        """
        return cls(
            ipath="",
            pat_file="scan.h5",
            pat_dset="Scan 1/EBSD/Data/Pattern",
            master_files=["master.h5"],
            psym_file="",
            pat_dims=(640, 480),
            circ_rad=-1,
            gaus_bckg=False,
            n_regions=10,
            delta=50.0,
            vendor="EMsoft",
            pctr=(0.0, 0.0, 15000.0),
            thetac=10.0,
            scan_dims=(256, 256),
            scan_steps=(1.0, 1.0),
            roi_mask="",
            bw=68,
            normed=True,
            refine=True,
            n_thread=0,
            batch_size=0,
            opath="",
            data_file="SphInx_Scan.h5",
            vendor_file="reindexed.ang",
            ipf_name="ipf.png",
            qual_name="qual.png",
        )

    @classmethod
    def from_string(cls, text: str) -> "EMSphInxNamelist":
        """Return a namelist parsed from the contents of a file.

        Parameters
        ----------
        text
            File contents.

        Returns
        -------
        namelist

        Raises
        ------
        ValueError
            On a grammar violation, a missing required key, an
            unknown ``vendor`` or a failed :meth:`sanity_check`.
        NotImplementedError
            If ``scandims`` is a string, i.e. the scan file route,
            which is not ported.

        Warns
        -----
        UserWarning
            If the file has keys the parser does not use, naming
            them as ``IndexEBSD`` does.
        """
        parsed = _NameList.from_string(text)
        namelist = cls()

        # inputs
        namelist.ipath = _optional(parsed, "ipath")
        namelist.psym_file = _optional(parsed, "psymfile")
        namelist.master_files = parsed.get_strings("masterfile")
        namelist.pat_file = parsed.get_string("patfile")
        namelist.pat_dset = _optional(parsed, "patdset")

        # scan dimensions, before the pattern centre as the C++ has it
        try:
            scan_file = parsed.get_string("scandims")
        except ValueError:
            scan_file = None
        if scan_file is not None:
            raise NotImplementedError(
                "A string valued 'scandims' reads the scan dimensions from a "
                f"scan file, here {scan_file!r}, which is not supported. Give "
                "the dimensions and the step size instead"
            )
        dims = parsed.get_doubles("scandims")
        if len(dims) not in (3, 4):
            raise ValueError(
                "expected a filename or dimensions + resolution for "
                "'scandims' in namelist"
            )
        # only the integrality is tested here, as the C++ tests it:
        # a negative whole number passes this and is caught by
        # ``sanity_check()`` as "non-positive scan dimensions"
        # instead, which is what the binary reports (measured)
        if any(value != int(value) for value in dims[:2]):
            raise ValueError("scan dimensinos must be non-negative integers")
        namelist.scan_dims = (int(dims[0]), int(dims[1]))
        namelist.scan_steps = (dims[2], dims[-1])
        namelist.scan_name = _optional(parsed, "scanname")
        namelist.roi_mask = parsed.get_string("roimask")

        # pattern processing
        pat_dims = parsed.get_ints("patdims")
        if len(pat_dims) != 2:
            raise ValueError("patdims must be 2 elements")
        namelist.pat_dims = (pat_dims[0], pat_dims[1])
        namelist.circ_rad = parsed.get_int("circmask")
        namelist.gaus_bckg = parsed.get_bool("gausbckg")
        namelist.n_regions = parsed.get_int("nregions")

        # pattern centre
        namelist.delta = parsed.get_double("delta")
        namelist.thetac = parsed.get_double("thetac")
        namelist.vendor = parsed.get_string("vendor")
        pctr = parsed.get_doubles("pctr")
        if len(pctr) != 3:
            raise ValueError("pctr    must be 3 elements")
        namelist.pctr = (pctr[0], pctr[1], pctr[2])
        if namelist.vendor not in VENDORS:
            raise ValueError(f"unknown vendor for pattern center `{namelist.vendor}'")

        # indexing parameters
        namelist.bw = parsed.get_int("bw")
        namelist.normed = parsed.get_bool("normed")
        namelist.refine = parsed.get_bool("refine")
        namelist.n_thread = parsed.get_int("nthread")
        namelist.batch_size = parsed.get_int("batchsize")

        # outputs
        namelist.opath = _optional(parsed, "opath")
        namelist.data_file = parsed.get_string("datafile")
        namelist.vendor_file = _optional(parsed, "vendorfile")
        namelist.ipf_name = _optional(parsed, "ipfmap")
        namelist.qual_name = _optional(parsed, "qualmap")

        namelist.sanity_check()
        if not parsed.fully_parsed():
            warn(
                f"some namelist parameters weren't used: {parsed.unused_tokens()}",
                UserWarning,
            )
        return namelist

    @classmethod
    def read(cls, filename: str | Path) -> "EMSphInxNamelist":
        """Return a namelist read from a file.

        Parameters
        ----------
        filename
            Path to the namelist file.

        Returns
        -------
        namelist
        """
        return cls.from_string(Path(filename).read_text(encoding="utf-8"))

    @classmethod
    def from_kwargs(
        cls,
        *,
        pattern_file: str,
        master_files: list[str],
        detector: "EBSDDetector",
        scan_shape: tuple[int, int],
        scan_steps: tuple[float, float],
        data_file: str,
        pat_dset: str = "patterns",
        vendor: str = "Bruker",
        delta: float | None = None,
        n_thread: int = 0,
        batch_size: int = 0,
        vendor_file: str = "",
        ipf_name: str = "",
        qual_name: str = "",
        **indexing_kwargs,
    ) -> "EMSphInxNamelist":
        """Return a complete namelist for indexing a written pattern
        file with a kikuchipy detector and indexing arguments.

        Parameters
        ----------
        pattern_file
            Pattern file name, e.g. one written by
            :func:`~kikuchipy.indexing.write_emsphinx_patterns`.
        master_files
            Master pattern ``*.sht`` file names.
        detector
            Detector from which ``pat_dims``, ``pctr`` and
            ``thetac`` are taken.  Any number of projection centres
            is accepted: the namelist has one, so
            :attr:`~kikuchipy.detectors.EBSDDetector.pc_average` is
            converted.
        scan_shape
            Navigation shape ``(n rows, n columns)``, which is
            written reversed as ``scandims``.
        scan_steps
            Scan step ``(x, y)`` in microns.
        data_file
            Output HDF5 orientation map name.
        pat_dset
            HDF5 path of the patterns, ``"patterns"`` by default,
            which is what
            :func:`~kikuchipy.indexing.write_emsphinx_patterns`
            writes.
        vendor
            Convention of the written ``pctr``, ``"Bruker"`` by
            default, for which it is the kikuchipy pattern centre
            verbatim.
        delta
            Pixel size in microns.  If not given (default),
            ``30000 / pat_dims[0]`` is used, a detector exactly 30 mm
            wide, which is always inside the sanity check window.
            The detector's own ``px_size`` is deliberately not used:
            kikuchipy fixtures carry ``px_size=1.0``, which the
            program rejects as a 0.06 mm detector.
        n_thread
            Number of work threads, ``0`` (automatic) by default.
        batch_size
            Number of patterns per work item, ``0`` (automatic) by
            default.  A ``chunksize`` given in ``**indexing_kwargs``
            is the same knob and overrides it.
        vendor_file
            Output ``.ang`` or ``.ctf`` name, empty (no output) by
            default.
        ipf_name
            Output inverse pole figure map name, empty (no output)
            by default.
        qual_name
            Output quality map name, empty (no output) by default.
        **indexing_kwargs
            Arguments of
            :meth:`~kikuchipy.signals.EBSD.spherical_indexing`:
            ``bandwidth``, ``normalize``, ``refine``, ``n_regions``,
            ``gaussian_background``, ``circular_mask`` and
            ``chunksize``, each defaulting to that method's default.
            ``chunksize`` is written to ``batch_size``, ``None``
            becoming ``0`` (automatic), which inverts
            :meth:`to_kwargs` exactly, so
            ``from_kwargs(**namelist.to_kwargs(), ...)`` rebuilds the
            namelist.

        Returns
        -------
        namelist
            Namelist which passes :meth:`sanity_check`.

        Raises
        ------
        ValueError
            If ``vendor`` is unknown, if the detector has a non-zero
            ``azimuthal`` or ``twist``, if ``**indexing_kwargs``
            holds an argument which is not one of the seven, or if
            the resulting namelist fails its sanity check.
        """
        if vendor not in VENDORS:
            raise ValueError(
                f"unknown vendor for pattern center `{vendor}': must be one "
                f"of {list(VENDORS)}"
            )
        for name in ("azimuthal", "twist"):
            angle = float(getattr(detector, name))
            if angle != 0:
                raise ValueError(
                    f"The detector {name} angle must be zero, but is {angle}"
                )

        arguments = _indexing_defaults()
        unknown = set(indexing_kwargs) - set(arguments)
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown indexing arguments {sorted(unknown)}: must be among "
                f"{list(_INDEXING_KEYS)}"
            )
        arguments.update(indexing_kwargs)

        pat_dims = (int(detector.shape[1]), int(detector.shape[0]))
        if delta is None:
            delta = 30000 / pat_dims[0]
        delta = float(delta)
        pctr = _pc_to_pctr(vendor, detector.pc_average, pat_dims[0], pat_dims[1], delta)

        chunksize = arguments["chunksize"]
        if "chunksize" in indexing_kwargs:
            batch_size = 0 if chunksize is None else int(chunksize)

        namelist = cls(
            pat_file=str(pattern_file),
            pat_dset=pat_dset,
            master_files=list(master_files),
            pat_dims=pat_dims,
            circ_rad=0 if arguments["circular_mask"] else -1,
            gaus_bckg=bool(arguments["gaussian_background"]),
            n_regions=int(arguments["n_regions"]),
            delta=delta,
            vendor=vendor,
            pctr=pctr,
            thetac=float(detector.tilt),
            scan_dims=(int(scan_shape[1]), int(scan_shape[0])),
            scan_steps=(float(scan_steps[0]), float(scan_steps[1])),
            roi_mask="",
            bw=int(arguments["bandwidth"]),
            normed=bool(arguments["normalize"]),
            refine=bool(arguments["refine"]),
            n_thread=int(n_thread),
            batch_size=int(batch_size),
            data_file=str(data_file),
            vendor_file=vendor_file,
            ipf_name=ipf_name,
            qual_name=qual_name,
        )
        namelist.sanity_check()
        return namelist

    # --------------------------- Writing ---------------------------- #

    def to_string(self) -> str:
        """Return the namelist file contents.

        The commented template of ``Namelist::to_string()``, with
        doubles at the C++ stream precision of six significant
        digits, line feed line endings, and the optional ``ipath``,
        ``patdset``, ``psymfile``, ``scanname``, ``opath``,
        ``vendorfile``, ``ipfmap`` and ``qualmap`` blocks omitted
        when their value is empty.

        Returns
        -------
        text
            Namelist file contents.

        Raises
        ------
        ValueError
            If any master file name after the first contains white
            space, which the parser would silently strip on read
            back, or if any written string ends with a backslash,
            which would escape its own closing quote.

        Notes
        -----
        The closing ``" /"`` line is emitted only inside the
        ``qualmap`` block, a ported quirk: an empty
        :attr:`qual_name` gives a file without a terminator, which
        the parser accepts.
        """
        for name in self.master_files[1:]:
            if any(character.isspace() for character in name):
                raise ValueError(
                    f"The master pattern file name {name!r} contains white "
                    "space, which the EMSphInx namelist parser strips from "
                    "every quoted string after the first. Only the first "
                    "name may contain white space"
                )
        for field in _WRITTEN_STRING_FIELDS:
            value = getattr(self, field)
            for name in [value] if isinstance(value, str) else value:
                if name.endswith("\\"):
                    raise ValueError(
                        f"The {field} value {name!r} ends with a backslash, "
                        "which escapes the closing quote of the value the "
                        "writer emits, so the EMSphInx namelist parser would "
                        "not read this back. Drop it or use a forward slash: "
                        "the parser has no escape for a backslash"
                    )

        text = " &EMSphInx\n"
        text += _BANNER
        text += "! Input Files\n"
        text += _BANNER
        text += "\n"
        if self.ipath != "":
            text += "! input path, empty for current working directory\n"
            text += f" ipath      = '{self.ipath}',\n"
            text += "\n"
        text += "! raw pattern file (relative to ipath) [can be up1, up2, or hdf5]\n"
        text += f" patfile    = '{self.pat_file}',\n"
        text += "\n"
        if self.pat_dset != "":
            text += "! h5 path of raw pattern  (ignored for non hdf5 patfile)\n"
            text += f" patdset    = '{self.pat_dset}',\n"
            text += "\n"
        text += "! master pattern with phases to index (relative to ipath)\n"
        text += " masterfile = "
        for name in self.master_files:
            text += f"'{name}', "
        text += "\n"
        text += "\n"
        if self.psym_file != "":
            text += (
                "! file with list of pseudo symmetric rotations to check "
                "(or '' for no psym check)\n"
            )
            text += f" psymfile   = '{self.psym_file}',\n"
            text += "\n"
        text += "\n"
        text += _BANNER
        text += "! Pattern Processing\n"
        text += _BANNER
        text += "\n"
        text += "! number of CCD pixels along x and y\n"
        text += f" patdims    = {self.pat_dims[0]}, {self.pat_dims[1]},\n"
        text += "\n"
        text += (
            "! should a circular mask be applied (-1 for no mask, 0 for "
            "largest inscribed circle, >0 to specify radius in pixels)\n"
        )
        text += f" circmask   = {self.circ_rad},\n"
        text += "\n"
        text += "! should a 2D gaussian background be subtracted\n"
        text += f" gausbckg   = {_logical(self.gaus_bckg)},\n"
        text += "\n"
        text += (
            "! how many regions should be used for adaptive histogram "
            "equalization (0 for no AHE)\n"
        )
        text += f" nregions   = {self.n_regions},\n"
        text += "\n"
        text += "\n"
        text += _BANNER
        text += "! Camera Calibration\n"
        text += _BANNER
        text += "\n"
        text += "! CCD pixel size on the scintillator surface [microns]\n"
        text += f" delta      = {_stream(self.delta)},\n"
        text += "\n"
        text += "! pattern center coordinates and vendor\n"
        text += "! vendor must be one of the following:\n"
        text += "!   EMsoft, EDAX, TSL, Oxford, Bruker\n"
        text += "! with pctr interpreted accordingly:\n"
        text += (
            "!   EMsoft   - pcx (pixels), pcy (pixels), scintillator "
            "distance (microns)\n"
        )
        text += "!   EDAX/TSL - x*, y*, z*\n"
        text += "!   Oxford   - x*, y*, z*\n"
        text += "!   Bruker   - x*, y*, z*\n"
        text += (
            "! note that vendors use different x*, y*, and z* : "
            "https://doi.org/10.1007/s40192-019-00137-4\n"
        )
        centre = ", ".join(_stream(value) for value in self.pctr)
        text += f" pctr       = {centre},\n"
        text += f" vendor     = '{self.vendor}',\n"
        text += "\n"
        text += "! tilt angle of the camera (positive below horizontal, [degrees]\n"
        text += f" thetac     = {_stream(self.thetac)},\n"
        text += "\n"
        text += "\n"
        text += _BANNER
        text += "! Scan Information\n"
        text += _BANNER
        text += "\n"
        text += "! dimensions of scan to index and pixel size\n"
        text += (
            "! x, y, step   for an x by y scan with square pixels of 'step' microns\n"
        )
        text += (
            "! x, y, sx, sy for an x by y scan with rectangular pixels of "
            "'sx' by 'sy' microns\n"
        )
        text += "! string to read dimensions from a scan file (*.ang, *.ctf, or *.h5)\n"
        text += (
            f" scandims   = {self.scan_dims[0]}, {self.scan_dims[1]}, "
            f"{_stream(self.scan_steps[0])}, {_stream(self.scan_steps[1])},\n"
        )
        text += "\n"
        if self.scan_name != "":
            text += (
                "! h5 path of scan data folder if scandims is an h5 file "
                "(ignored otherwise)\n"
            )
            text += f" scanname   = '{self.scan_name}',\n"
            text += "\n"
        text += "! region of interest for indexing\n"
        text += "! 0 (or omitted) to index the entire scan\n"
        text += (
            "! x0, y0, dx, dy for a (dx, dy) rectangular starting at pixel (x0, y0)\n"
        )
        text += "! string for an ROI mask file\n"
        text += f" roimask    = '{self.roi_mask}',\n"
        text += "\n"
        text += _BANNER
        text += "! Indexing Parameters\n"
        text += _BANNER
        text += "\n"
        text += (
            "! spherical harmonic bandwidth to be used (2*bw-1 should be a "
            "product of small primes for speed)\n"
        )
        text += (
            "! some reasonable values are: 53, 63, 68, 74, 88, 95, 113, 122, "
            "123, 158, 172, 188, 203, 221, 263, 284, 313\n"
        )
        text += (
            "! a nice range for parameter studies is 53, 68, 88, 113, 158, "
            "203, 263, 338 (~a factor of 1.3 between each)\n"
        )
        text += (
            "! any value is now pretty fast since the transform is zero "
            "padded to the nearest fast FFT size\n"
        )
        text += f" bw         = {self.bw},\n"
        text += "\n"
        text += (
            "! should normalized / unnormalized spherical cross correlation be used?\n"
        )
        text += (
            "! normalization is more robust for (esp. for lower symmetries) "
            "but is slower\n"
        )
        text += f" normed     = {_logical(self.normed)},\n"
        text += "\n"
        text += "! should newton's method orientation refinement be used?\n"
        text += (
            "! normalization is more robust for (esp. for lower symmetries) "
            "but is slower\n"
        )
        text += f" refine     = {_logical(self.refine)},\n"
        text += "\n"
        text += "! number of work threads\n"
        text += "! 0 (or omitted) to multithread with an automatic number of threads\n"
        text += "! 1 for serial threading\n"
        text += "! N to multithread with N threads\n"
        text += f" nthread    = {self.n_thread},\n"
        text += "\n"
        text += (
            "! number of patterns to index per work itme (ignored for single "
            "threading)\n"
        )
        text += (
            "! should be large enough to make the task significant compared "
            "to thread overhead\n"
        )
        text += (
            "! should be small enough to enable enough work items for load balancing\n"
        )
        text += (
            "! should be small enough so nthread * batchsize patterns can be "
            "held in memory\n"
        )
        text += "! 0 (or omitted) to estimate a reasonable value based on speed\n"
        text += f" batchsize  = {self.batch_size},\n"
        text += "\n"
        text += "\n"
        text += _BANNER
        text += "! Output Files\n"
        text += _BANNER
        text += "\n"
        if self.opath != "":
            text += "! output path, empty for current working directory\n"
            text += f" opath      = '{self.opath}',\n"
            text += "\n"
        text += "! output orientation map name relative to opath [must be hdf5 type]\n"
        text += f" datafile   = '{self.data_file}',\n"
        text += "\n"
        if self.vendor_file != "":
            text += (
                "! output orientation map name relative to opath (or omitted "
                "for no vendor output) [can be ang or ctf]\n"
            )
            text += f" vendorfile = '{self.vendor_file}',\n"
            text += "\n"
        if self.ipf_name != "":
            text += (
                "! output ipf map with {0,0,1} reference direction (or "
                "omitted for no ipf map) [must be png]\n"
            )
            text += f" ipfmap     = '{self.ipf_name}',\n"
            text += "\n"
        if self.qual_name != "":
            text += (
                "! output quality map with (or omitted for no quality map) "
                "[must be png]\n"
            )
            text += f" qualmap    = '{self.qual_name}'\n"
            text += " /\n"
        return text

    def write(self, filename: str | Path, overwrite: bool | None = None) -> None:
        """Write the namelist to a file.

        Parameters
        ----------
        filename
            Path to write to.  ``".nml"`` is appended when there is
            no suffix, and the parent directory is created if it
            does not exist.
        overwrite
            Whether to overwrite an existing file.  If not given,
            the user is asked.

        Raises
        ------
        ValueError
            As :meth:`to_string` does, or if ``overwrite`` is not
            ``None``, ``True`` or ``False``.
        """
        from kikuchipy.io._util import _ensure_directory, _overwrite

        text = self.to_string()

        filename = str(filename)
        if os.path.splitext(filename)[1] == "":
            filename += ".nml"

        is_file = os.path.isfile(filename)
        if overwrite is None:
            write = _overwrite(filename)
        elif overwrite is True or (overwrite is False and not is_file):
            write = True
        elif overwrite is False and is_file:
            write = False
        else:
            raise ValueError(
                f"overwrite can only be None, True or False, and not {overwrite}"
            )
        if write:
            # after the decision, so that a refused write leaves no
            # new directory behind
            _ensure_directory(filename)
            with open(filename, mode="w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    # -------------------------- Conversions ------------------------- #

    def sanity_check(self) -> None:
        """Check the field values as ``Namelist::sanityCheck()``
        does, in its order and with its messages.

        Raises
        ------
        ValueError
            If any of the thirteen checks fails.
        """
        if self.pat_file == "":
            raise ValueError("missing input pattern file")
        if len(self.master_files) == 0:
            raise ValueError("no master pattern files")
        for name in self.master_files:
            if name == "":
                raise ValueError("empty master pattern file name")
        smallest, largest = _PAT_DIM_LIMITS
        if (
            self.pat_dims[0] < smallest
            or self.pat_dims[0] > largest
            or self.pat_dims[1] < smallest
            or self.pat_dims[1] > largest
        ):
            raise ValueError(
                f"unreasonable pattern dimension (outside [{smallest}, {largest}] pix)"
            )
        if self.circ_rad < -1:
            raise ValueError("circular mask radius must be >= -1")
        if self.n_regions < 0 or self.n_regions > min(self.pat_dims):
            raise ValueError("unreasonable AHE nregions")
        width = self.delta * self.pat_dims[0] / 1000
        narrowest, widest = _DETECTOR_WIDTH_LIMITS_MM
        if width < narrowest or width > widest:
            raise ValueError("unreasonable EBSD detector width (should be [5, 90] mm)")
        lowest, highest = _THETAC_LIMITS
        if self.thetac < lowest or self.thetac > highest:
            raise ValueError("unreasonable camera tilt (should be [-60, 60] degrees)")
        if self.scan_dims[0] < 1 or self.scan_dims[1] < 1:
            raise ValueError("non-positive scan dimensions")
        smallest, largest = _BANDWIDTH_LIMITS
        if self.bw < smallest or self.bw > largest:
            raise ValueError(
                f"unreasonable bandwidth (should be [{smallest}, {largest}])"
            )
        if self.n_thread < 0:
            raise ValueError("negative thread count")
        if self.batch_size < 0:
            raise ValueError("negative batch size")
        if self.data_file == "":
            raise ValueError("missing output data file")

    def to_detector(self, *, sample_tilt: float) -> "EBSDDetector":
        """Return the detector the namelist describes.

        Parameters
        ----------
        sample_tilt
            Sample tilt in degrees.  Required: the namelist has no
            such field, since ``IndexEBSD`` takes it from the master
            pattern, where it is the ``sample_tilt`` of a
            :class:`~kikuchipy.indexing.MasterPatternHarmonics`.

        Returns
        -------
        detector
            Detector of shape ``(pat_dims[1], pat_dims[0])`` with one
            projection centre, ``px_size`` :attr:`delta` and ``tilt``
            :attr:`thetac`.
        """
        from kikuchipy.detectors import EBSDDetector

        width, height = self.pat_dims
        pc = _pctr_to_pc(self.vendor, self.pctr, width, height, self.delta)
        return EBSDDetector(
            shape=(int(height), int(width)),
            px_size=float(self.delta),
            binning=1,
            tilt=float(self.thetac),
            sample_tilt=float(sample_tilt),
            pc=pc,
            convention="bruker",
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Return the namelist as arguments of
        :meth:`~kikuchipy.signals.EBSD.spherical_indexing`.

        Returns
        -------
        kwargs
            ``bandwidth``, ``normalize``, ``refine``, ``n_regions``,
            ``gaussian_background``, ``circular_mask`` and
            ``chunksize``.

        Raises
        ------
        ValueError
            If :attr:`roi_mask` is neither empty nor ``"0"``, since
            the region of interest grammar is not ported.

        Warns
        -----
        UserWarning
            If :attr:`circ_rad` is positive.  EMSphInx keeps a
            circular mask of that radius in its image processor
            while its geometry flag stays false, and kikuchipy has
            no fixed radius mask, so that mask is lost.

        Notes
        -----
        :attr:`n_thread` has no kikuchipy equivalent, since dask's
        scheduler owns the workers, and is not returned.

        ``"0"`` is the program's own spelling of *no* region of
        interest, as the template's comment advertises and
        ``RoiSelection::from_string`` (``idx/roi.h`` line 592)
        implements, so it is accepted here and indexes the whole
        scan.  It is written back as ``'0'`` rather than as the
        ``''`` the C++ writer normalises it to, since the string is
        stored opaquely.
        """
        if self.roi_mask not in ("", "0"):
            raise ValueError(
                f"The region of interest 'roimask' {self.roi_mask!r} cannot "
                "be converted, since its grammar is not supported. Index the "
                "whole scan with an empty 'roimask' instead"
            )
        if self.circ_rad > 0:
            warn(
                f"The circular mask radius 'circmask' of {self.circ_rad} "
                "pixels is lost: EMSphInx masks its patterns at that radius "
                "in the image processor, and kikuchipy has no fixed radius "
                "mask, so 'circular_mask' is False. Pass a 'signal_mask' to "
                "mask the patterns instead",
                UserWarning,
            )
        return {
            "bandwidth": int(self.bw),
            "normalize": bool(self.normed),
            "refine": bool(self.refine),
            "n_regions": int(self.n_regions),
            "gaussian_background": bool(self.gaus_bckg),
            "circular_mask": self.circ_rad == 0,
            "chunksize": self.batch_size or None,
        }

    # --------------------------- Dunders ---------------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMSphInxNamelist):
            return NotImplemented
        return all(
            _comparable(getattr(self, name)) == _comparable(getattr(other, name))
            for name in _FIELDS
        )

    def __repr__(self) -> str:
        """Return a string with the pattern file, the binned pattern
        dimensions, the number of master pattern files, the scan
        dimensions and the bandwidth, e.g.
        ``"EMSphInxNamelist: 'patterns.h5' (60 x 60), 1 master
        pattern, 3 x 3 scan, bw = 68"``.

        The dimensions are written in the namelist's own x-then-y
        order, as the file has them.
        """
        number = len(self.master_files)
        plural = "" if number == 1 else "s"
        return (
            f"EMSphInxNamelist: {self.pat_file!r} "
            f"({self.pat_dims[0]} x {self.pat_dims[1]}), "
            f"{number} master pattern{plural}, "
            f"{self.scan_dims[0]} x {self.scan_dims[1]} scan, bw = {self.bw}"
        )


# Every field of ``ebsd::Namelist`` (``nml.hpp`` lines 54-88) under
# its pythonic name, which is what :meth:`EMSphInxNamelist.__eq__`
# compares
_FIELDS = (
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


# Every field :meth:`EMSphInxNamelist.to_string` writes inside single
# quotes, which is where a trailing backslash would escape the
# closing quote.  ``scan_file`` is never written.
_WRITTEN_STRING_FIELDS = (
    "ipath",
    "pat_file",
    "pat_dset",
    "master_files",
    "psym_file",
    "vendor",
    "scan_name",
    "roi_mask",
    "opath",
    "data_file",
    "vendor_file",
    "ipf_name",
    "qual_name",
)


def _comparable(value: Any) -> Any:
    """Return a value which compares by equality, sequences as
    tuples.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(value)
    return value


def _optional(parsed: _NameList, key: str) -> str:
    """Return a string valued key which may be absent, empty when it
    is, as the try/except blocks of ``parse_nml()`` do.
    """
    try:
        return parsed.get_string(key)
    except ValueError:
        return ""


def _logical(value: bool) -> str:
    """Return a boolean as the Fortran logical the C++ writes."""
    return ".TRUE." if value else ".FALSE."


def _stream(value: float) -> str:
    """Return a double as ``std::ostream::operator<<`` writes it,
    i.e. at the default precision of six significant digits.
    """
    return format(float(value), ".6g")


# ---------------------- Vendor conversions -------------------------- #


def _pctr_to_geometry(
    vendor: str, pctr: tuple[float, float, float], w: int, h: int, delta: float
) -> tuple[float, float, float]:
    """Return the EMSphInx geometry triple of a vendor pattern
    centre.

    Parameters
    ----------
    vendor
        One of :data:`VENDORS`.
    pctr
        Pattern centre in that vendor's convention.
    w, h
        Pattern width and height in pixels.
    delta
        Pixel size in microns.

    Returns
    -------
    geometry
        ``(cX, cY, sDst)``: the pattern centre in pixels relative to
        the detector centre and the scintillator distance in
        microns.

    Raises
    ------
    ValueError
        If ``vendor`` is unknown.
    """
    p0, p1, p2 = (float(value) for value in pctr)
    if vendor == "EMsoft":
        return p0, p1, p2
    if vendor in ("EDAX", "tsl"):
        return p0 * w - w / 2, p1 * w - h / 2, p2 * w * delta
    if vendor == "Oxford":
        return (p0 - 0.5) * w, (p1 - 0.5) * h, p2 * w * delta
    if vendor == "Bruker":
        return (p0 - 0.5) * w, (0.5 - p1) * h, p2 * h * delta
    raise ValueError(_unknown_vendor(vendor))


def _geometry_to_pctr(
    vendor: str, geometry: tuple[float, float, float], w: int, h: int, delta: float
) -> tuple[float, float, float]:
    """Return the vendor pattern centre of an EMSphInx geometry
    triple, the inverse of :func:`_pctr_to_geometry`.
    """
    cx, cy, distance = (float(value) for value in geometry)
    if vendor == "EMsoft":
        return cx, cy, distance
    if vendor in ("EDAX", "tsl"):
        return (cx + w / 2) / w, (cy + h / 2) / w, distance / (w * delta)
    if vendor == "Oxford":
        return cx / w + 0.5, cy / h + 0.5, distance / (w * delta)
    if vendor == "Bruker":
        return cx / w + 0.5, 0.5 - cy / h, distance / (h * delta)
    raise ValueError(_unknown_vendor(vendor))


def _pctr_to_pc(
    vendor: str, pctr: tuple[float, float, float], w: int, h: int, delta: float
) -> np.ndarray:
    """Return the kikuchipy (Bruker) pattern centre of a vendor
    pattern centre.

    ``"Bruker"`` returns the input unchanged, **bitwise**.  The
    algebraic identity of the module documentation is not the
    identity in floating point: composing
    :func:`_pctr_to_geometry` with the projection back drifts by an
    unit in the last place (measured: ``0.2134`` comes back as
    ``0.21340000000000003``, and ``0.5007`` as
    ``0.5006999999999999`` on a 60 x 48 detector), so the Bruker
    row is a short circuit rather than a composition.

    Raises
    ------
    ValueError
        If ``vendor`` is unknown.
    """
    if vendor == "Bruker":
        return np.asarray(pctr, dtype=float)
    cx, cy, distance = _pctr_to_geometry(vendor, pctr, w, h, delta)
    return np.array([cx / w + 0.5, 0.5 - cy / h, distance / (h * delta)], dtype=float)


def _pc_to_pctr(
    vendor: str, pc: np.ndarray, w: int, h: int, delta: float
) -> tuple[float, float, float]:
    """Return the vendor pattern centre of a kikuchipy (Bruker)
    pattern centre, the inverse of :func:`_pctr_to_pc`.

    ``"Bruker"`` returns the input unchanged, bitwise, for the
    reason given there.

    Raises
    ------
    ValueError
        If ``vendor`` is unknown.
    """
    values = tuple(float(value) for value in np.asarray(pc).ravel())
    if vendor == "Bruker":
        return values
    if vendor not in VENDORS:
        raise ValueError(_unknown_vendor(vendor))
    geometry = (
        (values[0] - 0.5) * w,
        (0.5 - values[1]) * h,
        values[2] * h * delta,
    )
    return _geometry_to_pctr(vendor, geometry, w, h, delta)


def _unknown_vendor(vendor: str) -> str:
    """Return the message of an unknown pattern centre vendor."""
    return (
        f"unknown vendor for pattern center `{vendor}': must be one of {list(VENDORS)}"
    )
