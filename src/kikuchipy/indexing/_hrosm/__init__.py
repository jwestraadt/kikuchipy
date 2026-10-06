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

"""Private tools for high angular resolution orientation similarity
maps (HROSM): kernel average misorientation, grain segmentation, grain
averaging, misorientation ball sampling and orientation similarity
maps, each in a correct mode and in an EMsoft compatible mode.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Submodules
----------
``_averaging``
    Grain averaging ("mean", "center", "vmf", "watson"), the grain
    reference orientation deviation (GROD) and the misorientation
    ball coverage warning.
``_directional_statistics``
    Von Mises-Fisher and Watson mixture estimates of a grain's mean
    orientation over the symmetry variants.
``_emsoft_file``
    Reader of EMsoft's dictionary indexing and EMHROSM HDF5 files,
    for tests and the reference script.
``_emsoft_quaternions``
    EMsoft's quaternion symmetry operators, Euler angle conversion,
    quaternion product and disorientation angle, in EMsoft's
    operation order.
``_grains``
    The grain table and the 2D grid of a crystal map.
``_kam``
    Kernel average misorientation maps.
``_osm``
    Grain-aware and EMsoft compatible orientation similarity maps.
``_sampling``
    Misorientation ball sampling and its mean angular spacing.
``_segmentation``
    Grain segmentation by EMsoft's KAM difference rule, grain
    dilation and bounding boxes.

Nothing is imported here on purpose: each submodule is imported
directly by the code that needs it, keeping the import cost of
:mod:`kikuchipy.indexing` unchanged.
"""
