# Copyright 2019-2024 The kikuchipy developers
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

"""Compute an orientation similarity map, where the ranked list of the
array indices of the best matching simulated patterns in one map point
is compared to the corresponding lists in the nearest neighbour points.
"""

# TODO: Consider moving to orix.

import numpy as np
from orix.crystal_map import CrystalMap
from scipy.ndimage import generic_filter


def orientation_similarity_map(
    xmap: CrystalMap,
    n_best: int | None = None,
    simulation_indices_prop: str = "simulation_indices",
    normalize: bool = False,
    from_n_best: int | None = None,
    footprint: np.ndarray | None = None,
    center_index: int = 2,
    *,
    grain_id: np.ndarray | None = None,
    emsoft_compatible: bool = False,
) -> np.ndarray:
    r"""Compute an orientation similarity map (OSM) where the ranked
    list of the dictionary indices of the best matching simulated
    patterns in one point is compared to the corresponding lists in the
    nearest neighbour points :cite:`marquardt2017quantitative`.

    Parameters
    ----------
    xmap
        A crystal map with a ranked list of the array indices of the
        best matching simulated patterns among its properties.
    n_best
        Number of ranked indices to compare. If not given (default), all
        indices are compared.
    simulation_indices_prop
        Name of simulated indices array in the crystal maps' properties.
        Default is ``"simulation_indices"``.
    normalize
        Whether to normalize the number of equal indices to the range
        [0, 1], by default ``False``.
    from_n_best
        Return an OSM for each n in the range ``[from_n_best, n_best]``.
        If not given (default), the OSM for ``n_best`` indices is
        returned.
    footprint
        Boolean 2D array specifying which neighbouring points to compare
        lists with, by default the four nearest neighbours.
    center_index
        Flat index of central navigation point in the truthy values of
        footprint, by default ``2``.
    grain_id
        Grain labels of the map's grid shape (n rows, n columns): 0
        outside grains, 1, 2, ... for the grains, as returned by
        :func:`~kikuchipy.indexing.segment_grains_kam`. If given, the
        lists of a point are compared only to those of its four
        nearest neighbours in the same grain. Cannot be combined with
        ``from_n_best``, ``footprint`` or ``center_index``.
    emsoft_compatible
        Whether to reproduce EMsoft's orientation similarity map, with
        EMsoft's neighbour bookkeeping and edge multipliers in single
        precision. Requires every map point to be in the data. Default
        is ``False``. Cannot be combined with ``from_n_best``,
        ``footprint`` or ``center_index``.

    Returns
    -------
    osm
        Orientation similarity map(s). If ``from_n_best`` is given, the
        returned array has three dimensions, where ``n_best`` is at
        ``osm[:, :, 0]`` and ``from_n_best`` at ``osm[:, :, -1]``.
        If ``grain_id`` is given or ``emsoft_compatible=True``, the
        map has the grid shape (n rows, n columns) of ``xmap`` and data
        type float32, and is never squeezed.

    Raises
    ------
    ValueError
        If ``n_best`` is greater than the number of ranked indices per
        point; if ``grain_id`` or ``emsoft_compatible=True`` is
        combined with ``from_n_best``, ``footprint`` or
        ``center_index``, or with each other; if
        ``emsoft_compatible=True`` and a map point is not in the data
        or not indexed; or if ``grain_id`` is not a valid label map of
        the grid shape.

    Notes
    -----
    If the set :math:`S_{r,c}` is the ranked list of best matching
    indices for a given point :math:`(r,c)`, then the orientation
    similarity index :math:`\eta_{r,c}` is the average value of the
    cardinalities (\#) of the intersections with the neighbouring sets

    .. math::

        \eta_{r,c} = \frac{1}{4}
            \left(
                \#(S_{r,c} \cap S_{r-1,c}) +
                \#(S_{r,c} \cap S_{r+1,c}) +
                \#(S_{r,c} \cap S_{r,c-1}) +
                \#(S_{r,c} \cap S_{r,c+1})
            \right).

    .. versionchanged:: 0.5
       Default value of ``normalize`` changed to ``False``.

    .. versionchanged:: 0.14
       Keyword-only ``grain_id`` and ``emsoft_compatible`` added.
    """
    if grain_id is not None or emsoft_compatible:
        return _osm_grain_aware_or_emsoft(
            xmap,
            n_best=n_best,
            simulation_indices_prop=simulation_indices_prop,
            normalize=normalize,
            from_n_best=from_n_best,
            footprint=footprint,
            center_index=center_index,
            grain_id=grain_id,
            emsoft_compatible=emsoft_compatible,
        )

    simulation_indices = xmap.prop[simulation_indices_prop]
    nav_size, keep_n = simulation_indices.shape

    if n_best is None:
        n_best = keep_n
    elif n_best > keep_n:
        raise ValueError(f"n_best {n_best} cannot be greater than keep_n {keep_n}")

    data_shape = xmap.shape
    flat_index_map = np.arange(nav_size).reshape(data_shape)

    if from_n_best is None:
        from_n_best = n_best

    osm = np.zeros(data_shape + (n_best - from_n_best + 1,), dtype=np.float32)

    if footprint is None:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    for i, n in enumerate(range(n_best, from_n_best - 1, -1)):
        match_indices = simulation_indices[:, :n]
        osm[:, :, i] = generic_filter(
            flat_index_map,
            lambda v: _orientation_similarity_per_pixel(
                v, center_index, match_indices, n, normalize
            ),
            footprint=footprint,
            mode="constant",
            cval=-1,
            output=np.float32,
        )

    return osm.squeeze()


def _osm_grain_aware_or_emsoft(
    xmap: CrystalMap,
    n_best: int | None,
    simulation_indices_prop: str,
    normalize: bool,
    from_n_best: int | None,
    footprint: np.ndarray | None,
    center_index: int,
    grain_id: np.ndarray | None,
    emsoft_compatible: bool,
) -> np.ndarray:
    """Return the grain-aware or the EMsoft compatible orientation
    similarity map of shape (n rows, n columns) of float32, never
    squeezed.

    See :func:`orientation_similarity_map` for the parameters.
    """
    from kikuchipy.indexing._hrosm._averaging import _check_grain_id
    from kikuchipy.indexing._hrosm._grains import _map_grid
    from kikuchipy.indexing._hrosm._osm import _osm_emsoft, _osm_grain_aware

    keywords = "grain_id" if grain_id is not None else "emsoft_compatible"
    if from_n_best is not None or footprint is not None or center_index != 2:
        raise ValueError(
            f"{keywords} cannot be combined with from_n_best, footprint or center_index"
        )
    if grain_id is not None and emsoft_compatible:
        raise ValueError("grain_id cannot be combined with emsoft_compatible=True")

    simulation_indices = np.asarray(xmap.prop[simulation_indices_prop])
    # A property of one index per point is squeezed by the indexing
    simulation_indices = simulation_indices.reshape(simulation_indices.shape[0], -1)
    keep_n = simulation_indices.shape[1]
    if n_best is None:
        n_best = keep_n
    elif n_best > keep_n:
        raise ValueError(f"n_best {n_best} cannot be greater than keep_n {keep_n}")

    grid, (ny, nx) = _map_grid(xmap)
    in_data = grid >= 0
    phase_id = np.full((ny, nx), -1, dtype=np.int64)
    phase_id[in_data] = np.asarray(xmap.phase_id).ravel()[grid[in_data]]
    present = in_data & (phase_id >= 0)

    # The lists over all grid points in raster order; points not in
    # the data hold -1 and are never compared
    lists = np.full((ny * nx, n_best), -1, dtype=simulation_indices.dtype)
    lists[in_data.ravel()] = simulation_indices[grid[in_data], :n_best]

    if emsoft_compatible:
        if not np.all(present):
            raise ValueError(
                "emsoft_compatible requires every map point to be in the "
                "data and indexed"
            )
        osm = _osm_emsoft(lists, ny, nx, n_best)
    else:
        grain_id = _check_grain_id(grain_id, (ny, nx))
        osm = _osm_grain_aware(lists, grain_id, present, n_best)

    if normalize:
        osm = osm / n_best

    return osm


def _orientation_similarity_per_pixel(
    v: np.ndarray, center_index: int, match_indices: np.ndarray, n: int, normalize: bool
) -> np.ndarray:
    # v are indices picked out with the footprint from flat_index_map
    v = v.astype(int)
    center_value = v[center_index]
    # Filter only true neighbours, -1 out of image and not include
    # itself
    neighbours = v[np.where((v != -1) & (v != center_value))]

    # Cardinality of the intersection between a and b
    number_of_equal_matches_to_its_neighbours = [
        len(np.intersect1d(match_indices[center_value], mi))
        for mi in match_indices[neighbours]
    ]

    os_i = np.nanmean(number_of_equal_matches_to_its_neighbours)

    if normalize:
        os_i /= n

    return os_i
