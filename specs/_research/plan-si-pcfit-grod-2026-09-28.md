# PARKED PLAN: spherical indexing + NCC refinement with a plane-fitted PC, static background from an amorphous area, and GROD maps

**Status (2026-09-28): designed, NOT started.** This is a workflow plan for Johan's own data, built
from existing kikuchipy APIs plus one small GROD helper. It needs no library change.

Johan's questions in this session, in order:
1. "how do run a spherical indexing combined with real DI refinement for a PC fitted with a plane?"
2. "Can you plot the GROD values as well?"
3. "I have collected several patterns from an amorphous area at the same conditions to make a static
   BG. Patterns were recorded without BG correction, so static needs to be calculated from the
   separate dataset, subtracted and dynamicGB applied."
4. He then asked for the workflow to be parked as its own plan.

**To resume:**
- Get the open inputs below from Johan.
- Re-verify the cited code lines.
- Run the steps on `hrebsd-dic`: it has both the spherical indexing code and `segment_grains`.
  `feat-spherical-indexing` and `develop` lack `segment_grains`.
- Related plan: `specs/_research/plan-hrosm-2026-09-28.md`. This workflow's refined map and
  plane-fitted detector are valid inputs to HROSM (its `pc="grain"` mode). Once HROSM lands, its
  `average_grain_orientations(method="mean")` and `grod` property supersede the GROD helper below.

## Constraints found in the code (verified 2026-09-28 on `hrebsd-dic`)

### Spherical indexing and refinement
1. **Spherical indexing (SI) takes exactly one PC** (`src/kikuchipy/signals/ebsd.py:2037-2042`). The
   same holds for `refine_orientation_spherical`. The plane-fitted per-pattern PCs can therefore only
   be used in the NCC step; SI runs on `pc_average` of the fitted detector.
2. **`EBSD.refine_orientation()` takes one PC per map point** (`ebsd.py:3340-3342`). This is where
   the plane fit takes effect.
3. **`EBSDDetector.fit_pc()` overwrites `sample_tilt`** with `90 - x_tilt - tilt`
   (`src/kikuchipy/detectors/_ebsd_detector.py:1575`).
   - SI raises `ValueError` unless `harmonics.sample_tilt == detector.sample_tilt` to a relative
     1e-6 (`src/kikuchipy/indexing/_spherical/_indexer.py:1705-1718`).
   - The harmonics' tilt is header metadata only (`primary_angle`,
     `_master_pattern_harmonics.py:570`), so building them with
     `sample_tilt=det_fit.sample_tilt` is safe.
   - SI then back-projects in the same geometry the NCC refinement uses.
4. **`refine_orientation()` requires the master pattern's phase to equal the xmap's phase**
   (`ebsd.py:4257-4267`). Harmonics built from `mp` carry `mp.phase`, so the check passes. It refines
   one phase per call.
5. **SI `"scores"` are not NCC.** Judge quality by the refined NCC `"scores"`.
6. **Trust region:** ±3° per Euler angle is enough after SI. In the Ånes reproduction, ±1° left
   59.5 % of points stuck at the bound.

### Static background
7. `remove_static_background` requires exactly the same **dtype and shape** as the patterns
   (`ebsd.py:552-563`), so the float mean of the amorphous set must be rounded, clipped and cast.
8. After subtraction, each pattern is rescaled to fill its dtype range
   (`src/kikuchipy/pattern/_pattern.py:393`):
   - a constant offset between the two datasets does not matter;
   - a **gain difference** does, and `scale_bg=True` handles it.

### GROD
9. There is no GROD function in kikuchipy or orix 0.14.2.
   - `kp.indexing.segment_grains` (`src/kikuchipy/indexing/_hrebsd/_segmentation.py:105`) exists
     only on `hrebsd-dic`.
   - orix `Quaternion.mean()` ignores crystal symmetry: tested, the naive mean gave GROD values up to
     36° on symmetry-scrambled points.
   - orix treats `S * o` as equivalent orientations (`Orientation.dot`: `M = other * ~self`
     compared with `S`), so each grain's variants must be aligned before averaging.

## Workflow

```python
import matplotlib.pyplot as plt
import numpy as np
import kikuchipy as kp
from orix.quaternion import Orientation, Quaternion, Rotation

# 0a. Static background from the separate amorphous-area dataset
s = kp.load("patterns.h5")                      # map, NO background correction yet
s_amor = kp.load("amorphous.h5")                # same kV, current, exposure, gain, binning
sig_shape = s.axes_manager.signal_shape[::-1]
assert s_amor.axes_manager.signal_shape[::-1] == sig_shape
bg = np.asarray(s_amor.data, dtype=np.float64).reshape(-1, *sig_shape).mean(axis=0)
# optional if few amorphous patterns: bg = scipy.ndimage.gaussian_filter(bg, 1)
info = np.iinfo(s.data.dtype) if np.issubdtype(s.data.dtype, np.integer) else None
if info is not None:
    bg = np.clip(np.round(bg), info.min, info.max)
bg = bg.astype(s.data.dtype)                    # dtype must match exactly (ebsd.py:552)
s.static_background = bg                        # travels with s and s.extract_grid

# 0b. Subtract static, then dynamic background, on the full map BEFORE extract_grid
s.remove_static_background(operation="subtract")          # scale_bg=True if a gradient remains
s.remove_dynamic_background(operation="subtract")         # std = width / 8 by default

# 0c. Master pattern (Lambert, both hemispheres; the same object feeds SI and NCC)
mp = kp.load("Ni-master-20kV.h5", projection="lambert", hemisphere="both", energy=20)
nav_shape = s.axes_manager.navigation_shape[::-1]
signal_mask = ~kp.filters.Window("circular", s.detector.shape).astype(bool)  # optional

# 1. PC calibration on a grid of the map (skip if calibrated PCs exist)
s_grid, idx = s.extract_grid((5, 5), return_indices=True)
det0 = s.detector.deepcopy(); det0.pc = det0.pc_average
h0 = mp.get_spherical_harmonics(bandwidth=188, sample_tilt=det0.sample_tilt)
xmap_grid = s_grid.spherical_indexing(h0, det0, bandwidth=68)
xmap_grid_ref, det_grid_ref = s_grid.refine_orientation_projection_center(
    xmap_grid, det0, mp, energy=20, signal_mask=signal_mask,
    method="LN_NELDERMEAD", trust_region=[5, 5, 5, 0.05, 0.05, 0.05],
    rtol=1e-5, chunk_kwargs=dict(chunk_shape=1),
)

# 2. Plane fit: one PC per map point (+ fitted sample_tilt)
det_fit = det_grid_ref.fit_pc(
    idx, map_indices=np.indices(nav_shape), transformation="projective",  # or "affine"
)
det_fit.shape = s.detector.shape   # only if the calibration patterns had another shape

# 3. SI over the full map with the single mean PC of the fitted plane
harmonics = mp.get_spherical_harmonics(bandwidth=188, sample_tilt=det_fit.sample_tilt)
det_si = det_fit.deepcopy(); det_si.pc = det_si.pc_average
xmap_si = s.spherical_indexing(harmonics, det_si, bandwidth=68)   # backend="gpu" if CuPy

# 4. NCC refinement with the per-pattern plane PCs (fixed)
xmap_ref = s.refine_orientation(
    xmap_si, det_fit, mp, energy=20,
    navigation_mask=~xmap_si.is_indexed.reshape(xmap_si.shape),  # skip SI failures
    signal_mask=signal_mask,
    method="LN_NELDERMEAD", trust_region=[3, 3, 3], rtol=1e-4,
)

# 5. GROD to the grain mean
def grain_grod(xmap, labels):
    """GROD in degrees of every map point to its grain's mean orientation.

    Returns a (ny, nx) array with NaN outside grains.
    """
    grod = np.full(labels.shape, np.nan)
    rows, cols = xmap.row, xmap.col                  # points in the data only
    lab = labels[rows, cols]
    q = xmap.rotations.data.reshape(-1, 4)
    phase_id = xmap.phase_id
    for g in np.unique(lab[lab >= 0]):
        i = np.flatnonzero(lab == g)
        pg = xmap.phases[int(phase_id[i[0]])].point_group
        # Move every orientation to the symmetry variant S * o closest to
        # the first point's, so that the plain quaternion mean is meaningful
        cand = pg.proper_subgroup.outer(Rotation(q[i])).data   # (n_sym, n, 4)
        k = np.argmax(np.abs(cand @ q[i[0]]), axis=0)
        mean = Orientation(Quaternion(cand[k, np.arange(i.size)]).mean(), pg)
        grod[rows[i], cols[i]] = Orientation(q[i], pg).angle_with(mean, degrees=True)
    return grod

labels = kp.indexing.segment_grains(xmap_ref, misorientation_threshold=5)
grod = grain_grod(xmap_ref, labels)
xmap_ref.plot(
    grod[xmap_ref.row, xmap_ref.col], colorbar=True, colorbar_label="GROD (°)",
    cmap="viridis", vmax=np.nanpercentile(grod, 99),
)
fig, ax = plt.subplots()
ax.hist(grod[np.isfinite(grod)], bins=100)
ax.set(xlabel="GROD (°)", ylabel="Points")
```

**How the GROD helper was checked (2026-09-28)**
- Test map: synthetic 20×30 Ni map, two grains with under 1° of spread, every point scrambled by a
  random `Oh` operator, 40 points removed from the data.
- Result: 2 grains found; the helper matched the true deviation to within 0.03°; the missing points
  came out as NaN; `xmap.plot` accepted the values.
- Without alignment, the mean was up to 36° off.

**Variants**
- **Existing calibration PCs** (e.g. 9 calibration patterns as in `hybrid_indexing.ipynb`): skip
  step 1 and call `det_cal_ref.fit_pc(pc_indices, map_indices=np.indices(nav_shape))`.
- **Multi-phase maps:** pass a list of harmonics to SI. Refine each phase with its own
  `navigation_mask`, then merge with `kp.indexing.merge_crystal_maps`.
- **`extrapolate_pc()` instead of `fit_pc()`** (`pc_extrapolate_plane.ipynb`): it leaves
  `sample_tilt` unchanged, so constraint 3 does not arise.
- **GROD to the best pattern of each grain:** replace `mean` with the orientation of the grain's
  highest-`scores` point.
- **Comparison plot:** add `grain_grod(xmap_si, segment_grains(xmap_si))` next to the refined map
  to show how much refinement lowers the noise floor.

## Open inputs (ask Johan at resume)
**Files and settings**
1. Paths of the map file and of the amorphous-area dataset.
2. Master pattern file and energy.
3. Detector binning, pixel size, sample tilt and camera tilt, if they are not in the file metadata.

**Method choices**
4. PC calibration route: a grid on the map (default above), existing calibration patterns, or an
   earlier calibration.
5. Plane fit: `"projective"` or `"affine"`, or `extrapolate_pc`.

**Deliverable**
6. Where it lives: a script or notebook next to the data (default: outside the repo, nothing
   committed), or a fork tutorial. A tutorial would need its own SDD spec.
7. Whether GROD should become a library function now. Recommended: no, because HROSM's
   `average_grain_orientations` + `grod` will provide it. Until then, keep the helper in the script.

## Verification
**Background**
- `plt.imshow(bg)` shows a smooth, band-free distribution.
- The navigation mean of the corrected map shows bands on a flat background with no gradient;
  otherwise use `scale_bg=True`.

**Plane fit and refinement**
- `det_fit.sample_tilt` is close to the nominal tilt, and `det_fit.plot_pc()` looks sensible.
- `xmap_ref.scores.mean()` is higher than a refinement run with `det_si` (single PC), which isolates
  the effect of the plane.
- The misorientation between SI and refined maps has a median of a few tenths of a degree, with few
  points pinned near 3°; otherwise widen `trust_region`.

**GROD**
- GROD inside grains is mostly sub-degree.
- The refined GROD floor is lower than the SI GROD floor in undeformed grains.

**Dry run**
- Optionally run the whole workflow first on shipped data (`kp.data.ni_gain(10)` +
  `kp.data.ni_gain_calibration(10)`, 20 kV Ni master).
