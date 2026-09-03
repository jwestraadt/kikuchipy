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

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    # -------------------------- Accessors --------------------------- #

    def get_bool(self, key: str) -> bool:
        """Return the single boolean stored under ``key``."""
        raise NotImplementedError

    def get_int(self, key: str) -> int:
        """Return the single integer stored under ``key``.

        A double raises, as ``Variant::getInt()`` does.
        """
        raise NotImplementedError

    def get_double(self, key: str) -> float:
        """Return the single double stored under ``key``.

        An integer is accepted and cast, as ``Variant::getDouble()``
        does.
        """
        raise NotImplementedError

    def get_string(self, key: str) -> str:
        """Return the single string stored under ``key``."""
        raise NotImplementedError

    def get_bools(self, key: str) -> list[bool]:
        """Return the booleans stored under ``key``."""
        raise NotImplementedError

    def get_ints(self, key: str) -> list[int]:
        """Return the integers stored under ``key``."""
        raise NotImplementedError

    def get_doubles(self, key: str) -> list[float]:
        """Return the doubles stored under ``key``, integers cast."""
        raise NotImplementedError

    def get_strings(self, key: str) -> list[str]:
        """Return the strings stored under ``key``."""
        raise NotImplementedError

    # ----------------------- Used flag queries ---------------------- #

    def fully_parsed(self) -> bool:
        """Return whether every stored value has been read."""
        raise NotImplementedError

    def unused_tokens(self) -> str:
        """Return the comma joined keys which have not been read.

        Sorted, as the C++ ``std::map`` order is.
        """
        raise NotImplementedError


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
        raise NotImplementedError

    # -------------------------- Properties -------------------------- #

    @property
    def pat_path(self) -> str:
        """Return the pattern file name as the program opens it.

        This is ``ipath + pat_file``, with ``ipath`` **twice** when
        :attr:`psym_file` is not empty, a ported quirk of
        ``parse_nml()`` line 247.
        """
        raise NotImplementedError

    @property
    def master_paths(self) -> list[str]:
        """Return the master file names as the program opens them,
        i.e. each prefixed with :attr:`ipath`.
        """
        raise NotImplementedError

    # ------------------------- Constructors ------------------------- #

    @classmethod
    def defaults(cls) -> "EMSphInxNamelist":
        """Return the namelist ``Namelist::defaults()`` builds, which
        is what ``IndexEBSD -t`` writes.

        Returns
        -------
        namelist
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
            ``azimuthal`` or ``twist``, or if the resulting namelist
            fails its sanity check.
        """
        raise NotImplementedError

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

        Raises
        ------
        ValueError
            If any master file name after the first contains white
            space, which the parser would silently strip on read
            back.

        Notes
        -----
        The closing ``" /"`` line is emitted only inside the
        ``qualmap`` block, a ported quirk: an empty
        :attr:`qual_name` gives a file without a terminator, which
        the parser accepts.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    # -------------------------- Conversions ------------------------- #

    def sanity_check(self) -> None:
        """Check the field values as ``Namelist::sanityCheck()``
        does, in its order and with its messages.

        Raises
        ------
        ValueError
            If any of the thirteen checks fails.
        """
        raise NotImplementedError

    def to_detector(self, *, sample_tilt: float) -> "EBSDDetector":
        """Return the detector the namelist describes.

        Parameters
        ----------
        sample_tilt
            Sample tilt in degrees.  Required: the namelist has no
            such field, since ``IndexEBSD`` takes it from the master
            pattern, where it is
            :attr:`~kikuchipy.indexing.MasterPatternHarmonics.sample_tilt`.

        Returns
        -------
        detector
            Detector of shape ``(pat_dims[1], pat_dims[0])`` with one
            projection centre, ``px_size`` :attr:`delta` and ``tilt``
            :attr:`thetac`.
        """
        raise NotImplementedError

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
            If :attr:`roi_mask` is not empty, since the region of
            interest grammar is not ported.

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
        """
        raise NotImplementedError

    # --------------------------- Dunders ---------------------------- #

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the pattern file, the binned pattern
        dimensions, the number of master pattern files, the scan
        dimensions and the bandwidth, e.g.
        ``"EMSphInxNamelist: 'patterns.h5' (60 x 60), 1 master
        pattern, 3 x 3 scan, bw = 68"``.

        The dimensions are written in the namelist's own x-then-y
        order, as the file has them.
        """
        raise NotImplementedError


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
    raise NotImplementedError


def _geometry_to_pctr(
    vendor: str, geometry: tuple[float, float, float], w: int, h: int, delta: float
) -> tuple[float, float, float]:
    """Return the vendor pattern centre of an EMSphInx geometry
    triple, the inverse of :func:`_pctr_to_geometry`.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
