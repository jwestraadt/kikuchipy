# Copyright 2019-2022 The kikuchipy developers
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

"""Generic, private parent class for all h5ebsd file plugins."""

import abc
from typing import Union, List, Tuple, Optional
import warnings

import dask.array as da
import h5py
import numpy as np


__all__ = ["_dict2hdf5group", "_hdf5group2dict", "H5EBSDReader"]


def _hdf5group2dict(
    group: h5py.Group,
    dictionary: Union[None, dict] = None,
    recursive: bool = False,
    data_dset_names: Optional[list] = None,
) -> dict:
    """Return a dictionary with values from datasets in a group.

    Parameters
    ----------
    group
        HDF5 group object.
    dictionary
        To fill dataset values into.
    recursive
        Whether to add subgroups to ``dictionary`` (default is
        ``False``).
    data_dset_names
        List of names of HDF5 data sets with data to not read.

    Returns
    -------
    dictionary
        Dataset values in group (and subgroups if ``recursive=True``).
    """
    if data_dset_names is None:
        data_dset_names = []
    if dictionary is None:
        dictionary = {}
    for key, val in group.items():
        # Prepare value for entry in dictionary
        if isinstance(val, h5py.Dataset):
            if key not in data_dset_names:
                val = val[()]
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
                key = key.lstrip()  # EDAX has some leading whitespaces
            if isinstance(val, bytes):
                val = val.decode("latin-1")
        # Check whether to extract subgroup or write value to dictionary
        if isinstance(val, h5py.Group) and recursive:
            dictionary[key] = {}
            _hdf5group2dict(
                group=group[key],
                dictionary=dictionary[key],
                data_dset_names=data_dset_names,
                recursive=recursive,
            )
        elif key in data_dset_names:
            pass
        else:
            dictionary[key] = val
    return dictionary


def _dict2hdf5group(dictionary: dict, group: h5py.Group, **kwargs):
    """Write a dictionary to datasets in a new group in an opened HDF5
    file format.

    Parameters
    ----------
    dictionary
        Dictionary with keys as dataset names.
    group
        HDF5 group to write dictionary to.
    **kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        if isinstance(val, dict):
            _dict2hdf5group(val, group.create_group(key), **kwargs)
            continue  # Jump to next item in dictionary
        elif isinstance(val, str):
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif ddtype == np.dtype("O"):
            try:
                if isinstance(val, (np.ndarray, da.Array)):
                    ddtype = val.dtype
                else:
                    ddtype = val[0].dtype
                dshape = np.shape(val)
            except TypeError:
                warnings.warn(
                    "The HDF5 writer could not write the following information to the "
                    f"file '{key} : {val}'"
                )
                break  # or continue?
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val


class H5EBSDReader(abc.ABC):
    """Abstract class implementing a reader of an h5ebsd file in a
    format specific to each manufacturer.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.
    """

    manufacturer_patterns = {
        "bruker nano": "RawPatterns",
        "edax": "Pattern",
        "kikuchipy": "patterns",
        "oxford instruments": "Processed Patterns",
    }

    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.file = h5py.File(filename, **kwargs)
        self.scan_groups = self.get_scan_groups()
        self.manufacturer, self.version = self.get_manufacturer_version()
        self.check_file()
        self.patterns_name = self.manufacturer_patterns[self.manufacturer]

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.version}): {self.filename}"

    @property
    def scan_group_names(self) -> List[str]:
        """Return a list of available scan group names."""
        return [group.name.lstrip("/") for group in self.scan_groups]

    def check_file(self):
        """Check if the file is a valid h5ebsd file by searching for
        datasets containing manufacturer, version and scans in the top
        group.

        Raises
        ------
        IOError
            If there are no datasets in the top group named
            ``"manufacturer"`` and ``"version"``, or if there are no
            groups in the top group containing the datasets
            ``"EBSD/Data"`` and ``"EBSD/Header"``, or if there is no
            reader for the file ``"manufacturer"``.
        """
        error = None
        if self.manufacturer is None or self.version is None:
            error = "manufacturer and/or version could not be read from its top group"
        if not any(
            "EBSD/Data" in group and "EBSD/Header" in group
            for group in self.scan_groups
        ):
            error = (
                "no top groups with subgroup name 'EBSD' with subgroups 'Data' and "
                "'Header' were found"
            )
        man, ver = self.get_manufacturer_version()
        man = man.lower()
        supported_manufacturers = list(self.manufacturer_patterns.keys())
        if man not in supported_manufacturers:
            error = (
                f"'{man}' is not among supported manufacturers "
                f"{supported_manufacturers}"
            )
        if error is not None:
            raise IOError(f"{self.filename} is not a supported h5ebsd file, as {error}")

    def get_manufacturer_version(self) -> Tuple[str, str]:
        """Get manufacturer and version from the top group.

        Returns
        -------
        manufacturer
            File manufacturer.
        version
            File version.
        """
        manufacturer = None
        version = None
        for key, val in _hdf5group2dict(group=self.file["/"]).items():
            if key.lower() == "manufacturer":
                manufacturer = val.lower()
            elif key.lower() in ["version", "format version"]:
                version = val.lower()
        return manufacturer, version

    def get_scan_groups(self) -> List[h5py.Group]:
        """Return a list of the groups with scans.

        Assumes all top groups contain a scan.

        Returns
        -------
        scan_groups
            List of available scan groups.
        """
        scan_groups = []
        for key in self.file.keys():
            if isinstance(self.file[key], h5py.Group):
                scan_groups.append(self.file[key])
        return scan_groups

    def get_desired_scan_groups(
        self, group_names: Union[None, str, List[str]] = None
    ) -> List[h5py.Group]:
        """Return a list of the desired group(s) with scan(s).

        Parameters
        ----------
        group_names
            Name or a list of names of the desired top group(s). If not
            given, the first scan group is returned.

        Returns
        -------
        scan_groups
            A list of the desired scan group(s).
        """
        # Get desired scan groups
        scan_groups = []
        if group_names is None:  # Return the first scan group
            scan_groups.append(self.scan_groups[0])
        else:
            if isinstance(group_names, str):
                group_names = [group_names]
            for desired_name in group_names:
                scan_is_here = False
                for name, scan in zip(self.scan_group_names, self.scan_groups):
                    if desired_name == name:
                        scan_groups.append(scan)
                        scan_is_here = True
                        break
                if not scan_is_here:
                    error_str = (
                        f"Scan '{desired_name}' is not among the available scans "
                        f"{self.scan_group_names} in '{self.filename}'."
                    )
                    if len(group_names) == 1:
                        raise IOError(error_str)
                    else:
                        warnings.warn(error_str)
        return scan_groups

    def read(
        self,
        group_names: Union[None, str, List[str]] = None,
        lazy: bool = False,
    ) -> List[dict]:
        """Return a list of dictionaries which can be used to create
        :class:`~kikuchipy.signals.EBSD` signals.

        Parameters
        ----------
        group_names
            Name or a list of names of the desired top HDF5 group(s). If
            not given, the first scan group is returned.
        lazy
            Read dataset lazily (default is ``False``). If ``False``,
            the file is closed after reading.

        Returns
        -------
        scan_list
            List of dictionaries with keys ``"axes"``, ``"data"``,
            ``"metadata"``, ``"original_metadata"``, ``"detector"``,
            (possibly) ``"static_background"``, and ``"xmap"``.
        """
        scan_dict_list = []
        for scan in self.get_desired_scan_groups(group_names):
            scan_dict_list.append(self.scan2dict(scan, lazy))

        if not lazy:
            self.file.close()

        return scan_dict_list

    @abc.abstractmethod
    def scan2dict(self, group: h5py.Group, lazy: bool = False) -> dict:
        """Read (possibly lazily) patterns from group.

        Parameters
        ----------
        group
            HDF5 group with patterns.
        lazy
            Read dataset lazily (default is ``False``).

        Returns
        -------
        scan_dict
            Dictionary with keys ``"axes"``, ``"data"``, ``"metadata"``,
            ``"original_metadata"``, ``"detector"``,
            ``"static_background"``, and ``"xmap"``. This dictionary can
             be passed as keyword arguments to create an
             :class:`~kikuchipy.signals.EBSD` signal.
        """
        return NotImplemented  # pragma: no cover
