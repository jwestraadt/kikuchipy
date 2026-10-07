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

"""Private reader of EMsoft's dictionary indexing (dot product) and
EMHROSM HDF5 files, for tests and the reference script.

Arrays EMsoft writes in Fortran order are returned in NumPy's order:
an ``(x, y)`` map as ``(H, W)``, a ``(3, N)`` array as ``(N, 3)``, so
the flat index ``y W + x`` is EMsoft's ``(iy - 1) W + ix`` minus one.
h5py already presents a Fortran array with its dimensions reversed,
so the arrays are returned as h5py reads them.
"""

from __future__ import annotations

from pathlib import Path
import re

import h5py
import numpy as np

# Data sets EMsoft writes as one-element arrays holding one number;
# they are returned as zero-dimensional arrays
_SCALAR_DATASETS = (
    "nGrains",
    "PointGroupNumber",
    "FZcnt",
    "Ncubochoric",
    "NumExptPatterns",
    "IndexingSuccessRate",
)

# Top-match lists EMsoft pads to a multiple of numexptsingle rows
_PADDED_DATASETS = ("TopMatchIndices", "TopDotProductList")

# Optional EMHROSM data sets, written only by some builds
_OPTIONAL_HROSM_DATASETS = ("newQuat", "maxGROD")

# A token of a namelist text: a quoted string, "=", "," or a run of
# other characters
_NAMELIST_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|=|,|[^\s,=]+")
_INTEGER = re.compile(r"[+-]?\d+")
_LOGICAL = re.compile(r"\.(true|false|t|f)\.?|\.?(true|false)\.?", re.IGNORECASE)


def read_emsoft_dot_product_file(path: str | Path) -> dict:
    """Return the arrays and namelist values of an EMsoft dot product
    file.

    The map shape is ``(ipf_ht, ipf_wd)``, or ``(ROI[3], ROI[2])``
    when the ROI is not all zeros. ``TopMatchIndices``, padded by
    EMsoft to a multiple of ``numexptsingle`` rows, is sliced to the
    number of map points and kept 1-based. Euler angles are returned
    in radians, unchanged.

    Parameters
    ----------
    path
        Path to the HDF5 file.

    Returns
    -------
    data
        Arrays of ``Scan 1/EBSD/Data``, the namelist values of
        ``NMLparameters/EMDINameList`` as Python scalars, the EMsoft
        version and the map shape.
    """
    with h5py.File(path, "r") as f:
        data = _read_group(f["Scan 1/EBSD/Data"])
        namelist = _read_namelist_group(f["NMLparameters/EMDINameList"])
        version = _read_string(f["EMheader/Version"])
        text = None
        if "NMLfiles/DictionaryIndexingNML" in f:
            text = _read_text(f["NMLfiles/DictionaryIndexingNML"])

    roi = [int(v) for v in np.atleast_1d(namelist.get("ROI", [0, 0, 0, 0]))]
    if sum(roi) != 0:
        shape = (roi[3], roi[2])
    else:
        shape = (int(namelist["ipf_ht"]), int(namelist["ipf_wd"]))
    n_points = shape[0] * shape[1]
    for key in _PADDED_DATASETS:
        if key in data:
            data[key] = data[key][:n_points]

    # A data set and a namelist value of the same name (Ncubochoric)
    # hold the same number; the data set is kept
    for key, value in namelist.items():
        data.setdefault(key, value)
    data["shape"] = shape
    data["version"] = version
    data["namelist_text"] = text
    return data


def read_emsoft_hrosm_file(path: str | Path) -> dict:
    """Return the arrays and namelist values of an EMsoft EMHROSM file.

    ``grainROI`` is returned 1-based as ``(x0, y0, w, h)``, as EMsoft
    writes it. ``dilate`` is parsed from the namelist text
    ``NMLfiles/HROSMNML``, since ``NMLparameters/HROSMNameList`` lacks
    it. The optional ``newQuat`` and ``maxGROD`` are None when absent.

    Parameters
    ----------
    path
        Path to the HDF5 file.

    Returns
    -------
    data
        Arrays of ``EMData/HROSM``, the namelist values, the namelist
        text and the EMsoft version.
    """
    with h5py.File(path, "r") as f:
        data = _read_group(f["EMData/HROSM"])
        namelist = _read_namelist_group(f["NMLparameters/HROSMNameList"])
        version = _read_string(f["EMheader/HROSM/Version"])
        text = _read_text(f["NMLfiles/HROSMNML"])

    for key in _OPTIONAL_HROSM_DATASETS:
        data.setdefault(key, None)
    for key, value in namelist.items():
        data.setdefault(key, value)
    # EMHROSM does not dilate when the namelist text does not say so
    data["dilate"] = bool(parse_namelist_text(text).get("dilate", False))
    data["shape"] = tuple(data["grainID"].shape)
    data["version"] = version
    data["namelist_text"] = text
    return data


def parse_namelist_text(text: str) -> dict:
    """Return the ``key = value`` pairs of a Fortran namelist text as
    Python values (Fortran logicals as bool, quoted strings without
    quotes, numbers as int or float, several values as a list).

    Parameters
    ----------
    text
        Namelist text; comment lines start with ``!``.

    Returns
    -------
    values
        Values keyed on the namelist keys.
    """
    tokens: list[str] = []
    for line in text.splitlines():
        for token in _NAMELIST_TOKEN.findall(_strip_comment(line)):
            if token.startswith("&"):
                continue  # the group name
            tokens.append(token)

    values: dict = {}
    key = None
    items: list = []
    for i, token in enumerate(tokens):
        if token in ("=", ","):
            continue
        quoted = token[0] in "'\""
        if not quoted and i + 1 < len(tokens) and tokens[i + 1] == "=":
            if key is not None:
                values[key] = items[0] if len(items) == 1 else items
            key, items = token, []
        elif token == "/":
            break  # the end of the group
        elif key is not None:
            items.append(_namelist_value(token))
    if key is not None:
        values[key] = items[0] if len(items) == 1 else items
    return values


def _strip_comment(line: str) -> str:
    """Return a namelist line without its ``!`` comment, keeping an
    exclamation mark inside a quoted string.
    """
    quote = None
    for i, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "!":
            return line[:i]
    return line


def _namelist_value(token: str) -> bool | int | float | str:
    """Return one namelist value token as a Python value."""
    if token[0] in "'\"":
        return token[1:-1]
    if _LOGICAL.fullmatch(token):
        return token.strip(".").lower() in ("true", "t")
    if _INTEGER.fullmatch(token):
        return int(token)
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return token


def _is_string(dataset: h5py.Dataset) -> bool:
    """Return whether a data set holds strings."""
    return dataset.dtype.kind in "OSU" or (
        h5py.check_string_dtype(dataset.dtype) is not None
    )


def _read_group(group: h5py.Group) -> dict:
    """Return the data sets of one group, strings decoded and the
    one-number data sets as zero-dimensional arrays.
    """
    data = {}
    for key, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if _is_string(dataset):
            data[key] = _read_string(dataset)
            continue
        value = dataset[()]
        if key in _SCALAR_DATASETS and value.size == 1:
            value = value.reshape(())
        data[key] = value
    return data


def _read_namelist_group(group: h5py.Group) -> dict:
    """Return the namelist values of one group as Python values: one
    value as a scalar, several as a list.
    """
    values = {}
    for key, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if _is_string(dataset):
            strings = _decode(dataset[()])
            values[key] = strings[0] if len(strings) == 1 else strings
            continue
        array = np.atleast_1d(dataset[()])
        values[key] = array[0].item() if array.size == 1 else array.tolist()
    return values


def _decode(values) -> list[str]:
    """Return EMsoft strings (bytes or str) as a list of str."""
    out = []
    for value in np.atleast_1d(values):
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        out.append(str(value))
    return out


def _read_string(dataset: h5py.Dataset) -> str:
    """Return a string data set as one str, lines joined by newlines."""
    return "\n".join(_decode(dataset[()]))


def _read_text(dataset: h5py.Dataset) -> str:
    """Return a namelist file stored line by line as one text."""
    return _read_string(dataset)
