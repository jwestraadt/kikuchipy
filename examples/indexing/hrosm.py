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
"""
==========================================================
High angular resolution orientation similarity map (HROSM)
==========================================================

This example shows how to re-index each grain of an indexed map against a fine
misorientation ball around the grain's reference orientation using
:meth:`~kikuchipy.signals.EBSD.hrosm`, and how the resulting high angular resolution
orientation similarity map (HROSM) compares with the orientation similarity map (OSM)
of a global dictionary.

The patterns are simulated for a synthetic two-grain nickel map, in which the
orientation in each grain rotates by 0.2 degrees per map column. A global dictionary
with a spacing of 3 degrees cannot resolve such small changes, so its OSM mainly shows
the grain boundary. Re-indexing each grain against a ball of orientations about 0.3
degrees apart recovers the gradient, seen in the grain reference orientation deviation
(GROD) of the re-indexed orientations.

More details are given in the :doc:`HROSM tutorial </tutorials/hrosm>` and the
:doc:`pattern matching tutorial </tutorials/pattern_matching>`.
"""

# %%
# Imports.
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from orix.quaternion import Rotation
from orix.sampling import get_sample_fundamental

import kikuchipy as kp

# Silence progressbars
hs.preferences.General.show_progressbar = False

# %%
# Create a synthetic map of (10, 20) points with two grains. The orientation of
# grain A (left half) rotates by 0.2 degrees per column about the crystal [001]
# axis, and grain B (right half) is grain A rotated by 30 degrees about the crystal
# [111] axis.
n_rows, n_cols = 10, 20
_, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
g0 = Rotation.from_euler(np.deg2rad([10, 20, 30]))
g_a = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(0.2 * cols)) * g0
g_b = Rotation.from_axes_angles([1, 1, 1], np.deg2rad(30)) * g_a
is_b = cols >= n_cols // 2
rotations = Rotation(np.where(is_b[:, None], g_b.data, g_a.data))

# %%
# Simulate patterns of (32, 32) pixels from the small nickel master pattern.
mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
det = kp.detectors.EBSDDetector((32, 32), pc=(0.42, 0.22, 0.50), sample_tilt=70)
s = mp.get_patterns(rotations.reshape(n_rows, n_cols), det, energy=20, compute=True)
print(s)

# %%
# Index the patterns against a global dictionary of orientations with a spacing of
# 3 degrees, keeping the ten best matches per point, and compute the global OSM.
R_global = get_sample_fundamental(resolution=3, point_group=mp.phase.point_group)
sim = mp.get_patterns(R_global, det, energy=20, compute=True)
xmap = s.dictionary_indexing(sim, keep_n=10)
osm_global = kp.indexing.orientation_similarity_map(xmap)
print(xmap)

# %%
# Re-index each grain against a misorientation ball of radius 2 degrees around its
# reference orientation, sampled with four steps along each half axis, giving 9**3 =
# 729 orientations per grain. The points at the grain boundary are not assigned to a
# grain, and are not re-indexed.
xmap_hr = s.hrosm(xmap, mp, det, energy=20, max_angle=2, n_steps=4, verbose=0)
print(xmap_hr.prop.keys())

# %%
# Compute the GROD of the global and of the re-indexed orientations, using the
# grains found by the re-indexing. The global dictionary assigns at most a few
# orientations, 3 degrees apart, to the points of a grain, while the re-indexed
# orientations follow the gradient within each grain.
grain_id = xmap_hr.get_map_data("grain_id").astype(int)
grod = []
for xm in [xmap, xmap_hr]:
    grains = kp.indexing.average_grain_orientations(xm, grain_id)
    grod.append(
        kp.indexing.grain_reference_orientation_deviation_map(xm, grain_id, grains)
    )

# %%
# Plot the global OSM and the HROSM on the same colour scale (top), and the GROD of
# the global and the re-indexed orientations on the same colour scale (bottom).
# Points not re-indexed are shown in grey. Both OSMs are high inside the grains, so
# the gain of the re-indexing shows in the GROD: only the re-indexed orientations
# resolve the gradient of 0.2 degrees per column.
cmap = plt.get_cmap("viridis").with_extremes(bad="0.6")
osm_hr = xmap_hr.get_map_data("osm")
grod_max = np.nanmax(grod)
fig, axes = plt.subplots(2, 2, figsize=(9, 4), layout="constrained")
for ax, values, title, vmax, label in zip(
    axes.ravel(),
    [osm_global, osm_hr, grod[0], grod[1]],
    ["Global OSM", "HROSM", "GROD, global", "GROD, re-indexed"],
    [10, 10, grod_max, grod_max],
    ["OSM", "OSM", "GROD (deg)", "GROD (deg)"],
):
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label=label)
