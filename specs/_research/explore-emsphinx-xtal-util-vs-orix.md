# EMSphInx → kikuchipy-stack capability map

Read-only survey. All paths absolute. EMSphInx = `c:/Users/westraadt.1/Repos/EMSphInx`, kikuchipy = `c:/Users/westraadt.1/Repos/kikuchipy`.

---

## 0. Environment (verified)

Interpreter: `c:/Users/westraadt.1/Repos/kikuchipy/.venv/Scripts/python.exe` — CPython **3.13.12** (MSC v.1944, 64-bit). `pip` is NOT installed in the venv (uv-managed); package list read from `.venv/Lib/site-packages` + `importlib.metadata`.

| package | version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| numba | 0.65.1 (llvmlite 0.47.0) |
| orix | 0.14.2 |
| diffsims | 0.7.0 |
| hyperspy | 2.4.0 |
| rosettasciio | 0.13.0 |
| pyebsdindex | 0.3.10 |
| nlopt | 2.10.0 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.8.0 |
| dask | 2026.3.0 |
| h5py | 3.16.0 |
| matplotlib | 3.10.9 |
| diffpy.structure | 3.4.0 |
| sympy | 1.14.0 |
| transforms3d | 0.4.2 |
| pyvista | 0.48.4 |
| pint | 0.25.3 |
| kikuchipy | 0.13.dev0 (editable, `src/kikuchipy`) |

**Spherical-harmonic libraries: NONE installed.** No `pyshtools`, no `shtns`, no `healpy`, no `pyfftw`, no `s2fft`, no `ducc0`. Available primitives:

| symbol | present |
|---|---|
| `scipy.special.sph_harm_y(n, m, theta, phi)` | ✅ (new API) |
| `scipy.special.sph_harm` | ❌ **removed in scipy 1.17** — do not use |
| `scipy.special.sph_harm_all` | ❌ |
| `scipy.special.sph_legendre_p` | ✅ |
| `scipy.special.assoc_legendre_p_all` | ✅ |
| `scipy.special.lpmv` | ✅ |
| `scipy.special.roots_legendre` / `numpy.polynomial.legendre.leggauss` | ✅ |
| `scipy.fft.dct/idct/rfft/set_workers` | ✅ (pocketfft, multithreaded) |
| `sympy.physics.wigner.wigner_d`, `wigner_d_small`, `wigner_3j`, `gaunt` | ✅ (symbolic, slow — validation only) |
| `sympy.physics.quantum.spin.Rotation.d/.D` | ✅ (symbolic) |

Built EMSphInx binaries at `c:/Users/westraadt.1/Repos/EMSphInx/build/Release/`: `EMSphInxEBSD.exe`, `IndexEBSD.exe`, `MasterXcorr.exe`, `mp2sht.exe`, `sht2png.exe`, `EBSPDims.exe`, `PatternRepack.exe`, `ShtWisdom.exe`.

---

## 1. Rotation conventions — EMSphInx vs orix

### 1.1 Global convention declarations

`c:/Users/westraadt.1/Repos/EMSphInx/include/constants.hpp:52-67`:
```
//crystallographic orientations in this code are described as passive rotations from the sample to the crystal frame
//eu[3] = {alpha, beta, gamma} is a rotation of the reference frame by:
// -first alpha about the z axis
// -next beta about the y' axis        <-- NOTE: comment is stale/wrong, see below
// -finally gamma bout the z'' axis
const int pijk = +1;
```
`c:/Users/westraadt.1/Repos/EMSphInx/include/xtal/rotations.hpp:30-36`:
```
// -passive rotations
// -quaternions as [w, x, y, z]
// -0 <= rotation angle <= pi
// -rotation axis in positive z hemisphere for rotations of pi (+y for z==0, 100 for y==z==0)
// -rotation axis = [0,0,1] for rotations of 0
```
`rotations.hpp:46` and `:137,:160` state unambiguously: **`eu` = ZXZ Euler angles (Bunge convention, radians)**. The "y'" in `constants.hpp` is a leftover; `xtal::Rotation::Euler` is ZXZ/Bunge, and there is a *separate* `xtal::Rotation::EulerZYZ` (`rotations.hpp:167`) for the Wigner ZYZ convention used inside the SHT cross-correlation.

### 1.2 The exact conversion (VERIFIED NUMERICALLY — they are identical)

`EMSphInx::xtal::eu2qu` (`rotations.hpp:409-429`):
```cpp
c=cos(eu[1]/2); s=sin(eu[1]/2); sigma=(eu[0]+eu[2])/2; delta=(eu[0]-eu[2])/2;
qu[0]= c*cos(sigma);
qu[1]= s*cos(delta)*(-pijk);   // pijk = +1
qu[2]= s*sin(delta)*(-pijk);
qu[3]= c*sin(sigma)*(-pijk);
if(qu[0]<0) negate;  // then normalize; if |w|<thr -> orientAxis(qu+1)
```
`orix.quaternion._conversions.eu2qu_single` (`.venv/Lib/site-packages/orix/quaternion/_conversions.py:804-841`) is **byte-for-byte the same formula** (`qu = [c cosσ, −s cosδ, −s sinδ, −c sinσ]`, negate if `a<0`).

Numerical check (4 random triples), max |Δ| = 0 to 10 decimals:
```
eu=(3.216,0.980,3.453) -> orix [ 0.86604205  0.46716181 -0.05569717 -0.16920061]
                       -> EMSphInx [ 0.86604205  0.46716181 -0.05569717 -0.16920061]
```

> **CONVERSION RULE (Bunge / `xtal::Rotation::Euler`)**
> `xtal::eu2qu(eu)` ≡ `orix.quaternion.Rotation.from_euler(eu, direction="lab2crystal", degrees=False).data`
> **No sign flip, no axis permutation, no conjugation, no π/2 offsets.** Both are passive lab→crystal, P = +1, quaternion `[w,x,y,z]` = orix `[a,b,c,d]`, `w ≥ 0`.
> Inverse: `xtal::qu2eu` ≡ `Rotation.to_euler()` (orix ranges φ1∈[0,2π], Φ∈[0,π], φ2∈[0,2π]; EMSphInx `om2eu` wraps identically — `rotations.hpp:435+` adds 2π to negatives).

> **CONVERSION RULE (Wigner ZYZ / `xtal::Rotation::EulerZYZ`)** — this is the representation the SHT cross-correlation peak (`Correlator::correlate(..., Real* eu, ...)`) returns.
> `rotations.hpp:1025-1039`:
> ```
> zyz2eu:  eu = (zyz[0] − π/2,  zyz[1],  zyz[2] + π/2)
> eu2zyz:  zyz = (eu[0] + π/2,  eu[1],  eu[2] − π/2)
> ```
> So in Python: `Rotation.from_euler(np.column_stack([zyz[:,0]-np.pi/2, zyz[:,1], zyz[:,2]+np.pi/2]))`.
> Direct closed form `zyz2qu` at `rotations.hpp:973-989`: `qu = [c cosσ, −s sinδ, −s cosδ, −c sinσ]` with `σ=(eu[2]+eu[0])/2`, `δ=(eu[2]−eu[0])/2` (note the swapped x/y and reversed δ vs ZXZ).
>
> **Erratum (2026-08-16, `specs/2026-08-16-sht-wigner-d/requirements.md`)**: the `zyz2eu`/`eu2zyz` lines above quote EMSphInx' `rotations.hpp:1025-1039`, whose offsets are **reversed** relative to `zyz2qu` (EMSphInx' own `test/xtal/rotations.cpp:296-310` builds `zyz = (phi1 − π/2, Φ, phi2 + π/2)` and asserts `zyz2qu(zyz) == eu2qu(eu)`; probe: `zyz2qu(zyz) == from_euler((α + π/2, β, γ − π/2))` to 8.6e-16 on 1000 triples). The correct relation is `bunge = (α + π/2, β, γ − π/2)` (`_euler.zyz_to_bunge`/`bunge_to_zyz`, Phase 3), so the Python line above must read `Rotation.from_euler(np.column_stack([zyz[:,0]+np.pi/2, zyz[:,1], zyz[:,2]-np.pi/2]))`; `zyz2eu`/`eu2zyz` are never ported.

### 1.3 Output-chain caveat from the indexer

`c:/Users/westraadt.1/Repos/EMSphInx/include/idx/indexer.hpp:264-269`:
```cpp
xtal::zyz2qu(res[i].qu, res[i].qu);                       // ZYZ euler -> quaternion (crystal->sample)
xtal::quat::mul(quNp.data(), res[i].qu, res[i].qu);       // correct for rotated detector frame
for(j=1..3) res[i].qu[j] = -res[i].qu[j];                 // conjugate: crystal->sample  =>  sample->crystal
```
`quNp` = `Geometry::northPoleQuat()` which in the current source is **hard-coded to identity** (`modality/ebsd/detector.hpp:454-459`, the tilt version is commented out). So the net map from the raw SHT peak `zyz` to an orix `Rotation` (sample→crystal, Bunge) is:

```python
eu_bunge = np.stack([zyz[...,0]-np.pi/2, zyz[...,1], zyz[...,2]+np.pi/2], -1)
rot = ~Rotation.from_euler(eu_bunge)      # note the ~ (conjugate)  ... times quNp if ever re-enabled
```

**Erratum (2026-08-16)**: `eu_bunge` above must use `zyz[...,0]+np.pi/2` and `zyz[...,2]-np.pi/2` (see the erratum in 1.2 and `specs/2026-08-16-sht-wigner-d/requirements.md`; Phase 3's `_euler.rotation_from_zyz` is `~Rotation(zyz_to_quaternion(zyz))`); the `~` conjugation is unchanged.

### 1.4 Representation-by-representation map

| EMSphInx (`xtal/rotations.hpp`) | Python equivalent | Notes |
|---|---|---|
| `eu2qu`/`qu2eu` (ZXZ Bunge) `:53-93` | `orix.quaternion.Rotation.from_euler` / `.to_euler` ; numba kernels `orix.quaternion._conversions.eu2qu_single/_2d`, `qu2eu_single/_2d` | identical formula, P=+1 |
| `eu2om`/`om2eu` `:53,:59` | `Rotation.from_matrix` / `.to_matrix` ; `_conversions.om2qu_single/_3d` | row-major 3×3, same |
| `ax2qu`/`qu2ax` `:68,:79` | `Rotation.from_axes_angles` / `.to_axes_angles` ; `_conversions.ax2qu_2d`, `qu2ax_2d` | |
| `ro2*`/`*2ro` (Rodrigues–Frank, `[nx,ny,nz,tan(w/2)]`) | `Rotation.from_rodrigues(..., )` / `.to_rodrigues` ; `_conversions.ax2ro_2d`, `ro2ax_2d` | orix `to_rodrigues` returns 3-vector `n·tan(w/2)`; EMSphInx uses 4-component `[n, tan(w/2)]` with `∞` for w=π. **Split/merge required** (2 lines). |
| `ho2*`/`*2ho` (homochoric) | `Rotation.from_homochoric` / `.to_homochoric` (`orix.vector.Homochoric`) ; `_conversions.ho2ax_2d`, `ho2ro_2d` | radius `hoR = (3π/4)^(1/3)` — `xtal/constants.hpp:100` matches orix |
| `cu2*`/`*2cu` (cubochoric) | `orix.quaternion._conversions.cu2ho`, `cu2ro`, `get_pyramid` (`_conversions.py:31-220,737-800`) | orix has **cu→ho/ro only**; there is **no `ho2cu`/`qu2cu`** in orix. `ho2cu` **must be written** if you need the inverse (invert `cu2ho_single` at `_conversions.py:106-182`; EMSphInx reference `rotations.hpp` `ho2cu`). Cube edge `cuA = π^(2/3)` (`xtal/constants.hpp:102`) matches. |
| `zyz2*`, `*2zyz` `:118-155` | **must be written** — trivial ±π/2 offset wrapper around `from_euler`/`to_euler` (see 1.2). |
| `Rotation` enum + `getConv()` dispatch `:158-184` | not needed — Python duck-typing |
| `xtal::Quat<Real>` class + `quat::` free functions (`quaternion.hpp:36-181`: `conj,inv,neg,cAbs,expl,normalize,norm,norm2,dot,mul,div,rotateVector,to_string`) | `orix.quaternion.Quaternion` (`*`, `~`, `.conj`, `.inv`, `.norm`, `.unit`, `.dot`, `.dot_outer`, `Quaternion * Vector3d`) ; numba `kikuchipy._utils.numba.rotate_vector` (`src/kikuchipy/_utils/numba.py:57-79`) for the hot loop | `quat::expl` ("explement", force w≥0) → `orix` does this in `from_*`; else `np.where(q[...,0]<0, -q, q)` (1 line) |
| `detail::orientAxis` (canonical axis for w=π) `rotations.hpp:247-262` | **must be written** (10 lines) — orix does not canonicalise π-rotation axes. Only matters for exact bit-comparison of unit tests. |
| test harness `test/xtal/rotations.cpp` (round-trip all 7×7 pairs over a Euler grid) | port as a pytest parametrisation over `orix` conversions |

---

## 2. Symmetry / point groups

### 2.1 `xtal::PointGroup` (`include/xtal/symmetry.hpp`)

| EMSphInx member (line) | Python equivalent | verdict |
|---|---|---|
| `PointGroup(uint8 sg, bool alt)` from space group `:58,:402-460` | `orix.quaternion.symmetry.get_point_group(space_group_number, proper=False)`; `orix.quaternion.symmetry.spacegroup2pointgroup_dict` | ✅ reuse |
| `PointGroup(std::string pg)` (HM / Schoenflies / Laue names) `:64` | `orix.quaternion.symmetry` module-level objects `C1…Oh` (39 names incl. `C2x/C2y/C2z`, `Csx/Csy/Csz`, `D3x/D3y`, `C3x/C3y/C3z`, `C4x/C4y/C4z`) + `Symmetry.name` (HM, e.g. `"m-3m"`) | ✅ mostly; **string-parser must be written** (small dict) |
| `Names()` `:74` | `[s.name for s in orix.quaternion.symmetry._groups]` | ✅ |
| `number()` (IUCr 1–32) `:78` | ❌ **must be written** (32-entry lookup keyed on `Symmetry.name`) |
| `name(lng)`, `fullName()`, `schonflies()`, `groth()`, `friedel()` `:88-112` | `Symmetry.name` only | partial; Schoenflies/Groth/Friedel **must be written** (lookup tables) |
| `tslNum()` / `FromTSL()` `:116,:124`, `hklNum()` / `FromHKL()` `:120,:128` | ❌ **must be written**. `pyebsdindex` has its own phase table but not this mapping. Small dict (TSL: 43,62,6,2,20,4,22,42,3,23,32,…; HKL Laue 1–11). |
| `order()` `:132` | `Symmetry.size` (== `Oh.size` → 48) — note `Symmetry.order` also exists | ✅ |
| `laueName()`, `laueGroup()` `:140,:144` | `Symmetry.laue` (returns `Symmetry`), `Symmetry.laue.name` | ✅ |
| `rotationGroup()` `:148` | `Symmetry.proper_subgroup` (and `Symmetry.laue_proper_subgroup`) | ✅ |
| `symmorphic(lat)` `:153` | ❌ **must be written** (32×6 lookup) |
| `symmorphicTrns()` `:158` (45°@z for `222r`, 90°@x for `112`) | `orix` handles settings via distinct objects (`C2x/C2y/C2z`, `D3x/D3y`); `orix.quaternion.symmetry.get_distinguished_points` | partial |
| `inversion()` `:167` | `Symmetry.contains_inversion` | ✅ |
| `enantiomorphism()` `:172` | `Symmetry.is_proper` | ✅ |
| `zMirror()` `:176` — **the `fm` / `fMr` flag** | ❌ **must be written**: `bool(np.any(np.isclose(sym.improper_mirror_normals·z, ±1)))`. Cheap correct form: `Symmetry` has `.diads` and `.get_axis_orders()`; simplest is a 32-entry table (see 2.2). |
| `mmType()` `:182` (0/1/2, mirror azimuth alignment) | ❌ **must be written** (32-entry table) |
| `zRot()` `:186` — **the `fn` / `fNf` flag** | `max(k for axis,k in Symmetry.get_axis_orders().items() if axis ≈ z)`; or `Symmetry.get_highest_order_axis()` when the highest axis is z. Safer: 32-entry table. **must be written (tiny)** |
| `numRotOps()` `:190`, `rotOps<Real>()` `:1618-1690` (closed set of proper ops as `wxyz`) | `Symmetry.proper_subgroup.data` (N×4 `[a,b,c,d]`) | ✅ **verified**: `orix.O` contains the identical 24 quaternions as EMSphInx's `432` block of `cub[]` (up to the irrelevant global sign of each quaternion; EMSphInx forces w≥0, orix does not) |
| `numMirror()`, `mirrors()` `:198,:1748` | ❌ not directly exposed by orix. Derive: `Symmetry[Symmetry.improper].axis` for the improper ops of order 2. **must be written (small)** |
| `numRotAxis()`, `rotAxis()` `:206,:1856` | `Symmetry.get_axis_orders()` → `{Vector3d: order}` ; `Symmetry.diads` | ✅ (sign of order for rotoinversion **must be added**) |
| `roInFz(ro)` `:219` + `FZ1/FZ121/…/FZ432` specialisations `:339-358` | `orix.quaternion.OrientationRegion.from_symmetry(sym)` + `OrientationRegion.__contains__` (`orix/quaternion/orientation_region.py`) | ✅ but the closed-form per-group inequalities are much faster; keep EMSphInx form if hot |
| `fzQu(qu, fz)` `:224,:2040-2060` (`ops * qu`, pick max w) | `orix.quaternion.Orientation(rot, symmetry=sym).reduce()` (`.map_into_symmetry_reduced_zone()` is **deprecated since orix 0.14, removal 0.15**) | ✅ — note the multiplication order: EMSphInx does `O_sym ⊗ q` (left mult, passive convention), which matches orix `Orientation` (symmetry applied on the left for crystal symmetry) |
| `disoQu(qu1,qu2,dis)` `:231`, `DisoQu`, `Diso432` fast path `:368-375` | `orix.quaternion.Misorientation(...).reduce()` ; `Orientation.angle_with(other, degrees=)` ; `Orientation.get_distance_matrix()` | ✅ (no 432 fast-path in orix; add if hot) |
| `nearbyQu(qu1,qu2,q3)` `:237` | ❌ **must be written** (5 lines: `argmax` of `dot(sym*q2, q1)`) |
| `fsDir(n, fs)` (reduce unit direction to fundamental sector) `:243` → dispatches to `xtal::fs::*` | `orix.vector.Vector3d.in_fundamental_sector(symmetry)` | ✅ **verified working** for `Oh`, `D6h` |
| `ipfColor(n, rgb, h2r)` `:248,:1900+` | `orix.plot.IPFColorKeyTSL(symmetry, direction).orientation2color(orientations)` ; `orix.plot.DirectionColorKeyTSL(symmetry).direction2color(v)` | ⚠️ **different colouring** — see §7 |
| `operator==/!=/<` `:253-263` | `Symmetry.name` comparison | ✅ |

### 2.2 `fm` / `fn` — symmetry-reduced FFT flags

These are **not** in `symmetry.hpp` under those names. They are function parameters in `include/sht/sht_xcorr.hpp`:

- `sht_xcorr.hpp:67-68`: `//@param fMr: true/false if there is/isn't a mirror plane in the first function` ; `//@param fNf: rotational symmetry about z axis in first function (1 for no rotational symmetry)`
- Declared in `Correlator<Real>::correlate` `:73`, `refinePeak` `:89`, `derivatives` `:139`, `compute` `:149`
- Stored as members `const bool fMr; const size_t fNf;` in `UnNormalizedCorrelator` `:203-204,:211`, `NormalizedCorrelator` `:245-248`, `Constants::mr` `:315`
- Used in `Correlator::compute` `:701-705`:
  ```cpp
  const size_t flmFold = fNf;   // flm[m*bw+j] == 0 if m % flmFold != 0
  const size_t glnFold = 1;
  const bool   fMir    = fMr;   // flm[m*bw+j] == 0 if (m+j) % 2 != 0
  const bool   gMir    = false;
  ```
  and to skip systematic zeros at `:717-722`, `:759`, `:768-790`, `:1004`. Same logic repeated in `derivatives` `:916-917`.
- **Where they come from**: `include/idx/master.hpp:185,189`
  ```cpp
  size_t nFold()  const {return MasterData<Real>::pointGroup().zRot();}    // -> fNf
  bool   mirror() const {return MasterData<Real>::pointGroup().zMirror();} // -> fMr
  ```

| capability | Python |
|---|---|
| `PointGroup::zRot()` → `fNf` | **must be written** — 32-entry table `{'1':1,'-1':1,'2':2,'m':1,…,'6/mmm':6,'23':2,'m-3':2,'432':4,'-43m':4,'m-3m':4}` keyed on `orix Symmetry.name`. (Note cubic groups report `zRot()==2` or `4`.) |
| `PointGroup::zMirror()` → `fMr` | **must be written** — 32-entry boolean table. |
| the `alm` systematic-zero skipping itself | **must be written** — this is inside your own SHT-correlator; nothing in numpy/scipy does it. |

### 2.3 `xtal/sphere_sector.hpp` — fundamental sector

| EMSphInx | Python |
|---|---|
| `fs::b1, _211, _211r, _121, _112, _m11, _1m1, _11m, _12m1, _112m, _222, _222r, mm2, mm2r, mmm, mmmr, _4, b4, _4m, _422, _4mm, b42m, b4m2, _4mmm, _3, _3r, b3, _321, _312, _3m1, _31m, b3m1, b31m, _6, b6, _6m, _622, _6mm, b6m2, b62m, _6mmm, _23, mb3, _432, b43m, mb3m` (`sphere_sector.hpp:36-81`, impl `:212-…`) — 46 hand-coded in-place direction reductions | `orix.vector.Vector3d.in_fundamental_sector(symmetry)` → `orix.vector.fundamental_sector.FundamentalSector` (`.vertices`, `.edges`, `.center`) and `orix.quaternion.Symmetry.fundamental_sector`. **Verified**: `Oh.fundamental_sector.vertices` = `[[0.577,0.577,0.577],[0.707,0,0.707],[0,0,1]]` (standard m-3m IPF triangle), 378 edge points. | ✅ reuse; orix is generic (built from `SphericalRegion` half-space normals) rather than 46 specialisations — slower but correct. Keep EMSphInx closed forms only if profiling demands. |
| `rot2d`, `mir2d`, `r111` helpers `:180-210` | trivially numpy | ✅ |
| `sph2rgb` (Nolze & Hielscher 2016 phenomenological hsl→rgb) `:88` | ❌ **must be written** (~60 lines). skimage has no HSL; `colorsys.hls_to_rgb` is the plain version. |
| `detail::SphericalPatch<N,Real>` `:94-135` (`toHemi`, `toColor`, fillets, cumulative angles, nonlinear-hue LUT), `SphericalTriangle` `:139`, `SphericalWedge` `:150` | ❌ **must be written** — orix's `DirectionColorKeyTSL` uses a *different* (EDAX/TSL) mapping. See §7. |

---

## 3. Square Lambert projection — grid conventions

### 3.1 orix status

**`orix.projections.LambertProjection` NO LONGER EXISTS in orix 0.14.2.**
`.venv/Lib/site-packages/orix/projections/__init__.pyi` exports only `StereographicProjection` and `InverseStereographicProjection`. `grep -rn "Lambert" orix/` → **zero hits**. Any old code importing it will break.

**`kikuchipy/projections/` NO LONGER EXISTS either** (removed in 0.13.dev0). The Lambert code now lives in:
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/signals/util/_master_pattern.py`
- `c:/Users/westraadt.1/Repos/kikuchipy/src/kikuchipy/_utils/vector.py` (only hemisphere/projection *string* parsing: `poles_from_hemisphere`, `parse_hemisphere`, `parse_projection`; lines 28-60)

### 3.2 The two implementations are the SAME projection up to an affine rescale (verified numerically)

**EMSphInx** `include/sht/square_sht.hpp:585-606` `emsphinx::square::lambert::sphereToSquare(x,y,z, X,Y)`:
```
X,Y ∈ [0,1]      (unit square)
if |y| <= |x|:  X = copysign(sqrt(1-|z|), x)*0.5 ;  Y = X*atan(y/x)/(π/4) + 0.5 ;  X += 0.5
else         :  Y = copysign(sqrt(1-|z|), y)*0.5 ;  X = Y*atan(x/y)/(π/4) + 0.5 ;  Y += 0.5
```
Inverse `squareToSphere(X,Y, x,y,z)` at `:615-642`.

**kikuchipy** `src/kikuchipy/signals/util/_master_pattern.py:543-581` `_vector2lambert(v) -> (n,2)`:
```
X = sign(x)*sqrt(2(1-|z|))*sqrt(π)/2                 (|y|<=|x|)
Y = sign(x)*sqrt(2(1-|z|))*(2/sqrt(π))*atan(y/x)
range: [-sqrt(π/2), +sqrt(π/2)]
```
Inverse `_lambert2vector(x, y)` at `:724-773` (expects x,y already divided by `sqrt(π/2)`, i.e. in [-1,1]).

> **EXACT RELATION (verified to 1e-8 on random unit vectors):**
> ```
> X_EMSphInx = ( _vector2lambert(v)[:,0] / sqrt(pi/2) + 1 ) / 2
> Y_EMSphInx = ( _vector2lambert(v)[:,1] / sqrt(pi/2) + 1 ) / 2
> ```
> i.e. kikuchipy's *normalised* Lambert coordinate `L = _vector2lambert(v)/sqrt(π/2) ∈ [-1,1]²` equals `2·(X,Y)_EMSphInx − 1`. Same square, same wedge assignment (`|y|≤|x|` branch), same handedness, same pole handling.
>
> Sample output:
> ```
> kp_norm [ 0.18938687 -0.1955411 ] -> mapped [0.59469344 0.40222945]  EMSphInx [0.59469344 0.40222945]
> kp_norm [-0.98524901 -0.57412054] -> mapped [0.00737550 0.21293973]  EMSphInx [0.00737550 0.21293973]
> ```

### 3.3 Dimension / indexing conventions

| aspect | EMSphInx | kikuchipy |
|---|---|---|
| side length symbol | `dim` (must be **odd**; `square_sht.hpp:44-52`) | `npx == npy` = signal shape; `= 2*npx_EMsoft + 1` (`src/kikuchipy/io/plugins/_emsoft_master_pattern.py:290`: `data_shape = (npx*2+1,)*2`) — identical (odd) |
| grid coordinate | `X = i/(dim−1)`, `Y = j/(dim−1)`, both ∈[0,1] (`square_sht.hpp:663-676` `normals()`) | `arr = np.linspace(-1, 1, size)` (`src/kikuchipy/signals/_kikuchi_master_pattern.py:168`, `simulations/kikuchi_pattern_simulator.py:184`) — identical after the affine map |
| memory layout | row-major, `xyz + 3*(dim*j + i)` → **row = Y-index, col = X-index** (`square_sht.hpp:668-673`) | `mp[nii, nij]` with `i = xy[:,1]` (Y→row), `j = xy[:,0]` (X→col) (`_master_pattern.py:651-652, 716-721`) — **identical** |
| interpolation scale factor | grid index = `X*(dim−1)` | `scale = (npx − 1) / 2`; `xy = scale * _vector2lambert(v)/sqrt(π/2)`; index = `int(xy + scale)` (`signals/ebsd_master_pattern.py:251-252`, `_master_pattern.py:649, 669-670`) — identical (`scale·L + scale = (npx−1)/2·(2X−1) + (npx−1)/2 = (npx−1)·X`) |
| hemispheres | `nh`, `sh` (`idx/master.hpp` `MasterPattern::nh/sh`), read from EMsoft `mLPNH`/`mLPSH` **with no transpose** (`idx/master.hpp:298-345`) | `"upper"`→`mLPNH`, `"lower"`→`mLPSH` (`io/plugins/_emsoft_master_pattern.py:317`: `{"upper":"NH","lower":"SH"}`); `hemisphere` ∈ `{"upper","lower","both"}` (`_utils/vector.py:24,47-52`); stereographic pole: upper→−1, lower→+1 (`_utils/vector.py:28-44`) | same data, different naming |
| grid *type* | `enum class Layout { Lambert, Legendre }` (`square_sht.hpp:58-61`). **EMSphInx indexes on the square-*Legendre* grid** (`DiscreteSHT::Legendre(dim)` `:107`, bandwidth `dim−2`); `Lambert(dim)` `:112` only reaches bandwidth `(dim−1)/2`. `MasterPattern::toLegendre()/toLambert()` (`idx/master.hpp:112-124`) bilinearly re-grids. | kikuchipy has **only** the Lambert grid | ⚠️ **`square::legendre::roots()` (`square_sht.hpp:739-800`, Barth–Martin–Wilkinson bisection) → replace with `numpy.polynomial.legendre.leggauss(n)` or `scipy.special.roots_legendre(n)`.** The `toLegendre` bilinear regrid **must be written** (~40 lines, or `scipy.interpolate.RegularGridInterpolator`). |

### 3.4 Other square-grid helpers

| EMSphInx (`sht/square_sht.hpp`) | Python |
|---|---|
| `lambert::cosLats(dim, lat)` `:645-660` — ring cos(latitude) = `(dim−1−4k…)/(dim−1)²` | trivial numpy one-liner; **must be written** (3 lines) |
| `lambert::normals(dim, xyz)` `:663-676` | `x,y = np.meshgrid(np.linspace(-1,1,dim), ...)` then `kikuchipy.signals.util._master_pattern._lambert2vector(x.ravel(), y.ravel())` | ✅ reuse |
| `lambert::solidAngles(dim, omg)` `:679-736` (Mazonka 2012 polyhedral-cone solid angle, 8-fold-symmetry-reduced) | ❌ **must be written** (~50 lines). For a *pure* Lambert grid all pixels are equal-area, so `w_y = 1/rings`; the function computes the deviation of the square-grid cells. |
| `legendre::normals` `:802+`, `legendre::boundingInds` `:820+` | ❌ **must be written** |
| `readRing`/`writeRing` `:249,:257`, `ringNum` `:289`, `computeWeightsSkip` `:265` (Reinecke eq. 10 quadrature weights) | ❌ **must be written** — core of the transform |
| `DiscreteSHT<Real>::analyze/synthesize` `:114-172`, `Constants` (`amn`,`bmn` on-the-fly Ylm recursion, `wy` ring weights) `:308-380` | ❌ **must be written** (this *is* the algorithm). Building blocks available: `scipy.fft.rfft` (ring FFTs), `scipy.special.sph_legendre_p` / `assoc_legendre_p_all` (validation), `numba` for the recursion. No installed library does square-grid SHT. |
| `sht/wigner.hpp` `d`, `D`, `dTable`, `dTablePre`, `dTablePreBuild`, `dPrime`, `dPrime2`, `rotateHarmonics` `:65-185` | ❌ **must be written**. `sympy.physics.wigner.wigner_d_small` exists but is symbolic/O(slow) — use for unit-test reference only. |
| `sht/sht_xcorr.hpp` (SO(3) cross-correlation via 3-D FFT of `Σ_{m,n} f̂ ĝ* d^j_{km} d^j_{kn}`, Newton peak refinement with analytic 1st/2nd derivatives) | ❌ **must be written**. Peak refinement could alternatively delegate to `nlopt` (installed, 2.10.0, already used by kikuchipy for `ln_neldermead`) or `scipy.optimize.minimize`. |
| `util/fft.hpp` (FFTW wrapper: `DCT`, `DCT2D`, `RealFFT`, plans, wisdom, `fft::vector` aligned allocator) | `scipy.fft` (`rfft`, `irfft`, `dct`, `idct`, `set_workers(n)`) — pocketfft, no plan/wisdom needed. `pyfftw` is **NOT installed**. | ✅ reuse |

---

## 4. Detector geometry & back-projection

### 4.1 `emsphinx::ebsd::Geometry` (`include/modality/ebsd/detector.hpp:51-160`)

| EMSphInx field / method | kikuchipy equivalent |
|---|---|
| `sTlt` (sample tilt, deg) `:56` / `sampleTilt()` `:69` | `EBSDDetector.sample_tilt` (`src/kikuchipy/detectors/_ebsd_detector.py:320-339`), default 70° |
| `dTlt` (camera tilt, deg) `:55` / `cameraTilt()` `:73` | `EBSDDetector.tilt` (`:341-367`) |
| `dOmg`, `sOmg` (declared, unused in `sampleDir`) `:55-56` | `EBSDDetector.azimuthal` (`:369-394`), `EBSDDetector.twist` (`:396-419`) — kikuchipy is **richer** here |
| `w,h` `:57` / `cameraSize(numsx,numsy,delta)` `:77` | `EBSDDetector.shape` `(nrows, ncols)` (`:585-599`), `.nrows` `:642`, `.ncols` `:637`, `.px_size` `:563-583`, `.binning` `:676-690` |
| `pX,pY` (µm) `:58` | `.px_size`, `.px_size_binned` (`:708`) |
| `cX,cY` (PC offset from image centre, px), `sDst` (µm) `:59-60` / `patternCenter(xpc,ypc,L)` `:82` — **EMsoft convention** | `EBSDDetector.pc_emsoft(version=5)` (`:1669-1718`); internal storage is **Bruker**. Conversions `_pc_emsoft2bruker` (`:2304-2312`) and `_pc_bruker2emsoft` (`:2326-2333`) |
| `patternCenterTSL(x*,y*,z*)` `:~300`: `cX = x*·w − w/2 ; cY = y*·w − h/2 ; sDst = z*·w·pX` | `EBSDDetector(..., convention="tsl"/"edax"/"amatek")`, `_pc_tsl2bruker` (`:2314-2318`), `.pc_tsl()` (`:1731-1768`) |
| `patternCenterOxford` : `cX=(x*−0.5)w ; cY=(y*−0.5)h ; sDst=z*·w·pX` | `convention="oxford"/"aztec"`, `_pc_oxford2bruker` (`:2320-2324`), `.pc_oxford()` (`:1770-1784`) |
| `patternCenterBruker` : `cX=(x*−0.5)w ; cY=(0.5−y*)h ; sDst=z*·h·pX` | native storage; `.pc_bruker()` (`:1720-1729`), `.pc` (`:434-457`), `.pc_average` (`:459-474`), `.pc_flattened` (`:476-486`), `.pcx/.pcy/.pcz` (`:488-561`) |
| `ebsd::Calibration<Real>` + `setVendor()` (`xtal/orientation_map.hpp:42-77`, impl `:165+`) | `EBSDDetector._get_pc_in_convention` (`:2259-2289`), `PC_CONVENTIONS` literal (`:71-91`) and `PC_CONVENTIONS_ALIASES` (`:86-91`) |
| `circ` (circular mask) `:61` / `maskPattern()` `:118` | `kikuchipy.filters.Window(...).make_circular()` (`src/kikuchipy/filters/window.py:249`); or `signal_mask` arg threaded through `_get_direction_cosines_from_detector` (`_master_pattern.py:84,111-113`) |
| `flip` (vertical image flip) `:62` / `flipPattern()` `:122` | not a detector attribute; handled at I/O |
| `bin(n)` `:127` | `EBSDDetector.binning` setter + `EBSD.downsample` (`src/kikuchipy/signals/ebsd.py:1113`) |
| `rescale(scale)` / `rescale(wNew,hNew)` `:145-150` (DCT-based, solid-angle-preserving) | `image::Rescaler` uses DCT — Python: `scipy.fft.dct/idct` (**must be written**, ~30 lines); or `skimage.transform.resize` (different result) |
| `solidAngle(gridRes)` `:141` | ❌ **must be written** (Monte-Carlo/grid count over `square::lambert::normals`) |
| `scaleFactor(dim)` `:159` | ❌ **must be written** (2 lines given `solidAngle`) |
| `northPoleQuat()` `:156`, impl `:454-459` | **currently returns identity** — no port needed |
| `readEMsoft(grp)` `:164` | `kikuchipy.load()` / `io/plugins/emsoft_ebsd_master_pattern` |

### 4.2 `Geometry::sampleDir(X, Y, n)` — the sphere direction for back projection

`include/modality/ebsd/detector.hpp:380-395`:
```cpp
alpha = (90 - sTlt + dTlt) * deg2rad;
sA = sin(alpha); cA = cos(alpha);
fx = (cX - X*w) * pX;
fy = (cY - Y*h) * pY;
den = sqrt(sDst^2 + fx^2 + fy^2);
n[0] = (sDst*sA + fy*cA)/den;
n[1] =           -fx      /den;
n[2] = (sDst*cA - fy*sA)/den;
```
**kikuchipy exact equivalents** — this is the function you asked for by name:

| purpose | function | file:line |
|---|---|---|
| **top-level entry, returns direction cosines in the SAMPLE frame** | `_get_direction_cosines_from_detector(detector, signal_mask=None) -> (n_px, 3)` or `(n_pc, n_px, 3)` | `src/kikuchipy/signals/util/_master_pattern.py:83-124` |
| single-PC numba kernel | `_get_direction_cosines_for_fixed_pc(gnomonic_bounds, pcz, nrows, ncols, om_detector_to_sample, signal_mask)` | `_master_pattern.py:127-212` |
| per-PC numba kernel | `_get_direction_cosines_for_varying_pc(...)` | `_master_pattern.py:215-308` |
| detector↔sample rotation | `EBSDDetector.sample_to_detector -> orix.quaternion.Rotation`; use `(~det.sample_to_detector).to_matrix().squeeze()` for detector→sample | `detectors/_ebsd_detector.py:830-854` |
| the underlying intrinsic-rotation builder (numba) | `_detector_to_sample_intrinsic_matrix(sigma, theta, omega, gamma)` — rows are `(X_d, Y_d, Z_d)` in sample coords; basis starts `[[0,1,0],[0,0,1],[1,0,0]]`, rotated by `(−σ, θ, −ω, −γ)` about axes `[0,0,1,2]` | `detectors/_ebsd_detector.py:94-144` |
| gnomonic extents used by the kernels | `EBSDDetector.x_range/.y_range/.x_scale/.y_scale/.gnomonic_bounds/.r_max` | `detectors/_ebsd_detector.py:743-828` |
| standalone gnomonic bounds (used by refinement objective fns) | `get_gnomonic_bounds(nrows, ncols, pcx, pcy, pcz) -> [x_min,x_max,y_min,y_max]` | `src/kikuchipy/_utils/_gnonomic_bounds.py:25-63` |
| detector Euler angles (Bunge ZXZ, degrees) `(−ω, 90+θ, −γ)` | `EBSDDetector.euler` | `detectors/_ebsd_detector.py:421-431` |
| pixel ↔ gnomonic coordinate conversion | `convert_pixel_to_gnomonic_coords`, `convert_gnomonic_to_pixel_coords`, `get_coordinate_conversion`, numba variants | `src/kikuchipy/detectors/_convert_detector_coordinates.py:56-228` |
| PyEBSDIndex indexer factory | `EBSDDetector.get_indexer(phase_list, reflectors=None, **kw)` → `pyebsdindex.ebsd_index.EBSDIndexer`; `indexer.PC = det.pc_flattened`; passes only `sample_tilt` and `tilt` (warns that azimuthal/twist are ignored) | `detectors/_ebsd_detector.py:1607-1667` + `indexing/_hough_indexing.py:_get_indexer_from_detector` |

> **VERIFIED NUMERICALLY — kikuchipy's direction cosines ARE EMSphInx's `sampleDir` grid.**
> With `pc=(0.5,0.5,0.5)` (PC pixel), varying tilts:
> ```
> sTlt=70 dTlt= 0 : kp=[0.342020 0 0.939693]   ems=(sin20°,0,cos20°)=(0.342020,0,0.939693)
> sTlt=70 dTlt=10 : kp=[0.500000 0 0.866025]   ems=(sin30°,0,cos30°)=(0.500000,0,0.866025)
> sTlt=70 dTlt=-10: kp=[0.173648 0 0.984808]   ems=(sin10°,0,cos10°)=(0.173648,0,0.984808)
> sTlt=60 dTlt= 5 : kp=[0.573576 0 0.819152]   ems=(sin35°,0,cos35°)=(0.573576,0,0.819152)
> ```
> Off-centre pixels also agree: detector +x (column ↑) → sample +y in both (`n[1] = −fx/den`, `fx=(cX−X·w)pX`); detector row ↑ (image up, EMSphInx cartesian `Y` ↑) → `n[0]` ↓ in both. The only difference is EMSphInx's `flip` flag for image-origin (top-left vs bottom-left) — handled inside `Geometry::interpolatePixel(n, pix, flp)` (`detector.hpp:334`).
>
> ⇒ **Reuse `_get_direction_cosines_from_detector` verbatim as the sphere-direction source for back projection.** kikuchipy additionally supports `azimuthal` and `twist`, which EMSphInx's `sampleDir` does not.

### 4.3 Back-projection machinery

| EMSphInx | Python |
|---|---|
| `ebsd::BackProjector<Real>` (`detector.hpp:163-200`) — `unproject(pat, sph, iq)`: rescale pattern → bilinear-interpolate detector pixels onto square-Legendre grid → return `sqrt(∫pat²)` weighted on sphere | ❌ **must be written**. Building blocks: direction cosines above; `image::BiPix` bilinear weights → `kikuchipy.signals.util._master_pattern._get_pixel_from_master_pattern` (`:694-721`) / `_get_lambert_interpolation_parameters` (`:584-691`) show the exact bilinear pattern to mirror, or `scipy.ndimage.map_coordinates(order=1)`. |
| `image::BiPix<Real>::bilinearCoeff(x,y,w,h)`, `::interpolate(pat)` (`util/image.hpp:~130-155`) | `_get_lambert_interpolation_parameters` returns exactly `(nii,nij,niip,nijp,di,dj,dim,djm)` — same 4-point weight scheme | ✅ pattern to copy |
| `emsphinx::BackProjector` abstract base, `ImageSource`, `ImageProcessor` (`idx/base.hpp:45-140`) | replaced by hyperspy lazy signals / dask (`kikuchipy.signals.util._dask.get_chunking`, `get_dask_array`) | ✅ |
| forward projection master-pattern → detector (for validation) | `_project_patterns_from_master_pattern_with_fixed_pc` (`_master_pattern.py:311-385`), `..._with_varying_pc` (`:386-460`), `_project_single_pattern_from_master_pattern` (`:461-540`); public API `EBSDMasterPattern.get_patterns(...)` (`signals/ebsd_master_pattern.py:97-330`) | ✅ reuse |

---

## 5. Pattern preprocessing — EMSphInx `imprc` / `ahe` / `gaussian` vs kikuchipy

`emsphinx::ebsd::PatternProcessor` (`include/modality/ebsd/imprc.hpp:45-89`) does exactly two things, in order:
1. `gaussian::BckgSub2D::fit(im)` then `subtract(im)` — separable 1-D Gaussian least-squares background fit (rows then columns), optional circular mask
2. `AdaptiveHistogramEqualizer::equalize(im, mask)` — 8-bit mosaic AHE

| EMSphInx capability | kikuchipy / stack equivalent | file:line |
|---|---|---|
| `gaussian::Model<Real>` — `f(x)=c·exp(−(x−a)²/b)`, `estimate()`, `fit()` (Gauss-Newton via `linalg`) (`util/gaussian.hpp:43-72`) | `scipy.optimize.curve_fit` / `least_squares`. **Small port** if you need bit-identical background. | — |
| `gaussian::BckgSub2D` (`util/gaussian.hpp:75-114`), `CircMask(w,h,r)` `:95` | **Not the same algorithm**, but the functional equivalent is `kikuchipy.pattern.remove_dynamic_background(pattern, operation="subtract"\|"divide", filter_domain="frequency"\|"spatial", std=w/8, truncate=4.0)` (Gaussian-blurred pattern subtracted/divided) | `src/kikuchipy/pattern/_pattern.py:512-603`; setup `:604-633`; numba kernels `_remove_background_subtract` `:484-497`, `_remove_background_divide` `:498-511`; signal method `EBSD.remove_dynamic_background` `src/kikuchipy/signals/ebsd.py:575-697`; chunked `pattern/chunk.py:33-74` |
| (EMSphInx has **no** static/reference background removal) | `EBSD.remove_static_background(operation="subtract"\|"divide", static_bg=...)` ; numba `_remove_static_background_subtract/_divide` | `signals/ebsd.py:442-574`; `pattern/_pattern.py:392-437` |
| `AdaptiveHistogramEqualizer<Real,uint8_t>` — `setSize(w,h,nx,ny)`, `equalize(im, msk)`, tile CDFs + bilinear interpolation, `hWdth=0.5` ⇒ **mosaic** sampling, 8-bit (256-bin) histograms, NULL-mask support (`util/ahe.hpp:43-104`, impl `:116-…`); free fn `adHistEq(im,w,h,nx,ny)` `:104`; also `image::adHistEq(..., vMin, vMax)` (`util/image.hpp:62-70`) | `EBSD.adaptive_histogram_equalization(kernel_size=None, clip_limit=0.0, nbins=128)` → wraps `skimage.exposure.equalize_adapthist` (CLAHE, 50 % overlap + bilinear, **not** mosaic; `clip_limit=0` ≈ plain AHE) | `src/kikuchipy/signals/_kikuchipy_signal.py:340-460`; kernel `pattern/_pattern.py:810-…` `_adaptive_histogram_equalization`; `EBSDMasterPattern.adaptive_histogram_equalization` `signals/ebsd_master_pattern.py:476` | ⚠️ **close but not identical**: EMSphInx uses mosaic tiles + a validity mask; skimage uses overlapping tiles and no mask. If exact parity matters, **must be written** (~120 lines, easy in numba). |
| `image::to8Bit(im,nPix)` (rescale min/max→[0,255]) (`util/image.hpp:46-58`) | `kikuchipy.pattern.rescale_intensity(pattern, in_range, out_range, dtype_out, percentiles)`; numba `_rescale_with_min_max`, `_rescale_without_min_max`, `_rescale_without_min_max_1d_float32`; signal `EBSD.rescale_intensity` | `pattern/_pattern.py:31-135`; `signals/_kikuchipy_signal.py:88-244` |
| `image::hist(im,w,h,bins,cnts,nBin)` `:72-79` | `numpy.histogram` | ✅ |
| `image::otsu(bins,cnts,nBin)` `:81-88` | `skimage.filters.threshold_otsu` | ✅ |
| `image::imageQuality(dct,w,h)` + `ImageQualityCalc` (**DCT**-based IQ) `:90-113` | `kikuchipy.pattern.get_image_quality(pattern, normalize=True, frequency_vectors=None, inertia_max=None)` — **FFT**-spectrum based (Krieger Lassen), *different definition*; `EBSD.get_image_quality`; numba `_get_image_quality_numba`; `fft_spectrum`, `fft_frequency_vectors` | `pattern/_pattern.py:348-391, 698-775`; `signals/ebsd.py:1312-1376` | ⚠️ different metric. DCT version **must be written** if parity needed (`scipy.fft.dctn`). |
| `image::Rescaler<Real>` (DCT-based image resize, high-pass filter width, zero-mean, optional IQ) `:117-165` | `scipy.fft.dctn/idctn` (**must be written**, ~30 lines); nearest kikuchipy analogue is `EBSD.downsample(factor)` / numba `_bin2d`, `_downsample2d` (integer factors only) | `pattern/_pattern.py:776-809`; `signals/ebsd.py:1113-1220` |
| `image::BlockRowBackground` (row/column DCT background for non-rectangular images) `:167-…` | ❌ **must be written** (or use `remove_dynamic_background` + mask) |
| — | `EBSD.fft_filter(transfer_function, function_domain, shift)` + `kikuchipy.filters.Window`, `modified_hann`, `lowpass_fft_filter`, `highpass_fft_filter`, Barnes-algorithm `filters.fft_barnes.fft_filter` | `signals/ebsd.py:805-942`; `filters/window.py:31-560`; `filters/fft_barnes.py:29-190`; `pattern/_pattern.py:213-347`; `pattern/chunk.py:75-129` | kikuchipy-only extra |
| — | `EBSD.average_neighbour_patterns(window="circular", window_shape=(3,3))`; `pattern/chunk.py:130-176` | `signals/ebsd.py:943-1112` | kikuchipy-only extra |
| — | `EBSD.normalize_intensity(num_std=1, divide_by_square_root, dtype_out)`; numba `_normalize_intensity`, `_zero_mean`, `_normalize`, `_zero_mean_sum_square_1d_float32` | `signals/_kikuchipy_signal.py:245-339`; `pattern/_pattern.py:136-212` | required before NCC; EMSphInx normalises inside `MasterSpectra` (`idx/master.hpp:161-166` `nrm` flag) and via `unproject()`'s returned `sqrt(∫pat²)` |
| — | `EBSD.get_neighbour_dot_product_matrices`, `get_average_neighbour_dot_product_map` | `signals/ebsd.py:1221-1492`; `signals/util/_map_helper.py:35-330` | kikuchipy-only extra |

---

## 6. CrystalMap / result construction

| EMSphInx | Python |
|---|---|
| `xtal::OrientationMap<Real>` (`include/xtal/orientation_map.hpp:79-160`): `width, height, xStep, yStep, calib, owner, name, phsList, phase[], qu[], metric[], imQual[]`, `read/write/writeH5`, `ipfColor` | `orix.crystal_map.CrystalMap` (`x,y,dx,dy,phase_id,phases,phases_in_data,rotations,orientations,prop,is_indexed,is_in_data,scan_unit,shape,get_map_data,plot`) + `orix.crystal_map.create_coordinate_arrays(shape, step_sizes)` | ✅ reuse |
| `xtal::Phase<Real>` (`include/xtal/phase.hpp:47-71`): `lat[6]`, `name`, `pg`; `readMaster` (EMsoft `/CrystalData` → `SpaceGroupNumber`, `LatticeParameters`), `readEBSD`/`writeEBSD` (H5EBSD `Lattice Constant a…gamma`, `Symmetry` = TSL number, `MaterialName`) | `orix.crystal_map.Phase(name, space_group, point_group, structure)` (`.point_group`, `.space_group`, `.structure` = `diffpy.structure.Structure`, `.color`, `.from_cif`, `.expand_asymmetric_unit`), `orix.crystal_map.PhaseList` | ✅ reuse; `PointGroup::tslNum()` mapping still **must be written** for H5EBSD round-trip |
| `ebsd::Calibration` (`orientation_map.hpp:42-77`) | `kikuchipy.detectors.EBSDDetector` (see §4) | ✅ |
| TSL/HKL vendor readers (`xtal/vendor/tsl.hpp`, `hkl.hpp`, `XTAL_USE_TSL/HKL`) | `orix.io.load` (`.ang`, `.ctf`, `.h5`); `kikuchipy.load` for patterns; `rsciio` plugins | ✅ |
| `OrientationMap::metric` (e.g. CI) | see property names below | |

### 6.1 Exact kikuchipy `CrystalMap.prop` key names

| producer | keys | file:line |
|---|---|---|
| **dictionary indexing** (`EBSD.dictionary_indexing`) | `"scores"` (float, shape `(n, keep_n)`), `"simulation_indices"` (int32) | `src/kikuchipy/indexing/_dictionary_indexing.py:160-166` |
| **Hough indexing** (`EBSD.hough_indexing`, PyEBSDIndex) | `"fit"`, `"cm"`, `"pq"`, `"nmatch"` (all from `data_index[...]` of the PyEBSDIndex structured array) | `src/kikuchipy/indexing/_hough_indexing.py:104-118` |
| **refinement** (`refine_orientation`, `refine_projection_center`, `refine_orientation_projection_center`) | `"scores"` (float64), `"num_evals"` (int32), and `"pseudo_symmetry_index"` (int32) when pseudo-symmetry ops are given | `src/kikuchipy/indexing/_refinement/_refinement.py:317-330, 122-128, 286-292` |
| **map merging** | `merge_crystal_maps(crystal_maps, mean_n_best=1, greater_is_better=None, scores_prop="scores", simulation_indices_prop=None, ...)` → props `{scores_prop, f"merged_{scores_prop}"}`, plus `simulation_indices_prop` / `f"merged_{...}"` | `src/kikuchipy/indexing/_merge_crystal_maps.py:28-33, 164-165, 195-298` |
| **OSM** | `orientation_similarity_map(xmap, n_best=None, simulation_indices_prop="simulation_indices", normalize=False)` | `src/kikuchipy/indexing/_orientation_similarity_map.py:30-131` |
| PC per point | `EBSDDetector.pc` array of shape `nav_shape + (3,)`; refinement returns a **new detector**, not a `"pc"` prop (`_refinement.py:196` `return scores, new_detector, num_evals`) | |
| compatibility check | `_xmap_is_compatible_with_signal(xmap, navigation_axes, raise_if_not)`, `_equal_phase(p1,p2)`, `_get_indexed_points_in_data_in_xmap(xmap, navigation_mask)` | `src/kikuchipy/signals/util/_crystal_map.py:28-160` |
| refinement optimizers | `SUPPORTED_OPTIMIZATION_METHODS = {"minimize"(scipy,local), "ln_neldermead"(nlopt,local), "basinhopping","differential_evolution","dual_annealing","shgo"(scipy,global)}` | `src/kikuchipy/indexing/_refinement/__init__.py:32-64`; nlopt solvers `_refinement/_solvers.py:462-600` |

> **Recommendation for an EMSphInx-style result map:** `CrystalMap(rotations=Rotation(...), phase_id=..., phase_list=PhaseList(...), prop={"scores": xc_peak, "iq": image_quality, "simulation_indices": phase_idx}, x=..., y=..., scan_unit="um")`. Use `"scores"` (not `"metric"`) so `merge_crystal_maps` and `orientation_similarity_map` work out of the box. EMSphInx's `metric` → `"scores"`, `imQual` → `"iq"` (new name, no kikuchipy convention exists).

---

## 7. IPF / colour

| EMSphInx | Python |
|---|---|
| `util/colorspace.hpp:41-120` — `rgb2xyz/luv/lab/hsv/hsl`, `xyz2rgb/luv/lab/hsv/hsl`, `luv2*`, `lab2*`, `hsv2*`, `hsl2*` (D65 2° illuminant default, `[0,1]` ranges, hue in `[0,1]`) | `skimage.color`: `rgb2xyz`, `xyz2rgb`, `rgb2lab`, `lab2rgb`, `rgb2luv`, `luv2rgb`, `rgb2hsv`, `hsv2rgb`, `lab2lch`. **HSL is missing from skimage** → `colorsys.hls_to_rgb` / `hls_to_rgb` (stdlib, scalar) or `matplotlib.colors`. `xyz2rgb` gamut-clip flag: skimage clips silently; EMSphInx returns `bool` in-gamut. **Small port** if you need the flag. |
| `detail::inv3x3`, `rgbMat` (`colorspace.hpp:~110-135`) | `numpy.linalg.inv` | ✅ |
| `xtal::sph2rgb` (Nolze & Hielscher 2016 phenomenologically-adjusted HSL→RGB) (`sphere_sector.hpp:88`) | ❌ **must be written** (~60 lines). Not in orix, not in skimage. |
| `detail::SphericalPatch<N>::toHemi/toColor` + `SphericalTriangle`/`SphericalWedge` (`sphere_sector.hpp:94-156`) | ❌ **must be written**. |
| `PointGroup::ipfColor(n, rgb, h2r)` (`symmetry.hpp:248`, impl `:1900+`, 32-case switch over triangles `b3,c2,c2b,c3,c3b,c4,c6,m3,m3m` and wedges `w2,w2b,w2c,w4,w3a,w3b,w6`) | **`orix.plot.IPFColorKeyTSL(symmetry, direction=Vector3d.zvector())`** → `.orientation2color(Orientation)`, `.direction_color_key`, `.plot()`; **`orix.plot.DirectionColorKeyTSL(symmetry)`** → `.direction2color(Vector3d)`; also `orix.plot.EulerColorKey` | ⚠️ **available and working, but the colours differ.** orix implements the EDAX/TSL scheme; EMSphInx implements Nolze–Hielscher. Verified working: `IPFColorKeyTSL(Oh, Vector3d.zvector()).orientation2color(o)` → e.g. `[[1.0,0.858,0.696],[1.0,0.509,0.991],[0.094,1.0,0.42]]`. |
| `OrientationMap::ipfColor(rgb, refDir, h2r, alpha)` (`orientation_map.hpp:150-160`) | `IPFColorKeyTSL(pg, direction).orientation2color(xmap.orientations)` then `xmap.plot(rgb)`; `orix.plot.CrystalMapPlot`, `InversePoleFigurePlot`, `StereographicPlot` | ✅ |
| kikuchipy usage examples | `doc/tutorials/hough_indexing.ipynb`, `pattern_matching.ipynb`, `hybrid_indexing.ipynb`, `pc_orientation_dependence.ipynb`, `esteem2022_diffraction_workshop.ipynb` (all use `IPFColorKeyTSL`) — **no IPF code in `src/kikuchipy/`; it is entirely delegated to orix.** | |
| `xtal/diagram.hpp` (620 lines, stereogram/symmetry diagrams), `util/svg.hpp` (1353), `util/bmp.hpp` (580) | `orix.plot.StereographicPlot`, `orix.plot._symmetry_marker`, `Symmetry.plot()`, matplotlib SVG/PNG backends, `imageio`/`PIL` | ✅ |

---

## 8. `util/linalg.hpp` — dense linear algebra

`include/util/linalg.hpp:27-29` explicitly says *"minimal … prefer a proper library (Eigen/LAPACK) if you need speed"*. Everything maps 1:1 to numpy/scipy (LAPACK).

| EMSphInx | Python |
|---|---|
| `solve::lu(a,x,b,n)` `:68` | `numpy.linalg.solve` / `scipy.linalg.lu_solve` |
| `solve::cholesky(a,x,b,n)` `:77` | `scipy.linalg.cho_solve` |
| `decompose::lu(a,p,n)` `:88` | `scipy.linalg.lu_factor` |
| `decompose::cholesky(a,d,n)` `:98` (returns `neg` flag if negated) | `scipy.linalg.cho_factor` (raises on non-PD; wrap for the `neg` flag) |
| `decompose::qr(a,m,n)` (Givens, Golub & Van Loan 5.2.2) `:108` | `scipy.linalg.qr` (Householder — same R up to signs) |
| `backsolve::lu` `:122`, `backsolve::cholesky` `:134` | `scipy.linalg.lu_solve`, `cho_solve` |
| `qr::applyQ`, `qr::applyQH` `:148,:158` | `scipy.linalg.qr(..., mode='raw')` + `scipy.linalg.lapack.dormqr`, or just `Q @ y` |
| complex support via `detail::is_complex` `:176-181` | numpy dtypes handle it |

**Verdict: reuse entirely; do not port `linalg.hpp`.**

---

## 9. Other EMSphInx modules

| EMSphInx | Python |
|---|---|
| `xtal/position.hpp` (1024 lines) — `GenPos` (packed 4×4 general position: `Mirror`, `Two`, `Three`, `Z`, `getMat3`, `getTrans*`, `det`, `tr`, `order`, `getMat3HexCart`), Wyckoff formatting bitmasks | `diffpy.structure.spacegroups` (`GetSpaceGroup(n).symop_list`, `.iscentrosymmetric`) via `orix.crystal_map.Phase.space_group`; `orix.crystal_map.Phase.expand_asymmetric_unit`. Packed-int representation and Wyckoff string formatting **must be written** if needed. | mostly ✅ |
| `xtal/hm.hpp` (1434 lines) — `HermannMaguin`: `fromNumber(sg, alt)`, `fromString`, `to_string`, `shortSym`, `changeMonoCell`, `changeOrthoAxis`, `generators(xyz, rHex)`, origin choice 1/2, rhombohedral/hexagonal settings | `diffpy.structure.spacegroups.GetSpaceGroup("Fm-3m")` / by number; `orix.quaternion.symmetry.GetSpaceGroup`. **Setting/cell-choice manipulation (`changeMonoCell`, `changeOrthoAxis`, origin choice 2) has no Python equivalent — must be written** if you need it (probably you don't for EBSD indexing). | partial |
| `xtal/diagram.hpp` | see §7 |
| `util/nml.hpp` (EMsoft namelist parser) | `f90nml` not installed; simple regex parser **must be written**, or reuse `kikuchipy/io/plugins/emsoft_*` which read the NML values already stored in the HDF5 (`/NMLparameters/...`) | |
| `util/base64.hpp` | `base64` (stdlib) | ✅ |
| `util/threadpool.hpp` | `dask`, `concurrent.futures`, `numba.prange`, `scipy.fft.set_workers` | ✅ |
| `util/timer.hpp` | `time.perf_counter` | ✅ |
| `util/sysnames.hpp` | `platform`, `socket`, `getpass` | ✅ |
| `miniz/` (zip) | `zipfile` (stdlib) | ✅ |
| `sht_file.hpp` / `.sht` EMsoft spherical-harmonics file format | ❌ **must be written** — kikuchipy has **no** `.sht` reader (`io/plugins/` has only `emsoft_ebsd_master_pattern`, `emsoft_ecp_master_pattern`, `emsoft_tkd_master_pattern`, `bruker_h5ebsd`, `edax_*`, `oxford_*`, `nordif*`, `kikuchipy_h5ebsd`, `ebsd_directory`). Use `h5py` + the layout in `include/sht/sht_file.hpp`, or regenerate spectra from the EMsoft `.h5` master pattern that kikuchipy *can* read. |
| `idx/master.hpp` `MasterPattern::rotate/nFoldSym/mirror/inversion/copyEquator` (`:130-150`, impl `:475-530`) — impose point-group symmetry on the master pattern grid | ❌ **must be written** (numpy slicing/averaging on the square grid, ~80 lines) |
| `idx/master.hpp` `MasterPattern::read(EMsoft .h5)` (`:239-345`): energy-weighted average over `accum_e`, sum over atoms, `mLPNH`/`mLPSH` | `kikuchipy.load("...h5", projection="lambert", hemisphere="both", energy=...)`; energy handling in `io/plugins/emsoft_ebsd_master_pattern/_api.py` and `io/plugins/_emsoft_master_pattern.py:128,141,167-190` | ✅ reuse |
| `wx/` (wxWidgets GUI) | out of scope |

---

## 10. Summary — what you actually have to write

**Reuse as-is (no porting):**
1. All 7 rotation representations + conversions → `orix.quaternion.Rotation` / `orix.quaternion._conversions` (Euler is *bit-identical*).
2. Point-group / Laue-group / proper-subgroup objects, symmetry operators, fundamental zone, fundamental sector, disorientation → `orix.quaternion.symmetry`, `orix.quaternion.Orientation/Misorientation`, `orix.vector.Vector3d.in_fundamental_sector`, `orix.quaternion.OrientationRegion`.
3. Square-Lambert forward/inverse (numba) → `kikuchipy.signals.util._master_pattern._vector2lambert` / `_lambert2vector` / `_get_lambert_interpolation_parameters` / `_get_pixel_from_master_pattern` (affine-equivalent to EMSphInx, verified).
4. Detector geometry + pixel direction cosines in the sample frame → `kikuchipy.detectors.EBSDDetector` + `_get_direction_cosines_from_detector` (numerically identical to `Geometry::sampleDir`, verified; kikuchipy also supports azimuthal & twist).
5. All PC-convention conversions (bruker/tsl/edax/amatek/oxford/aztec/emsoft4/emsoft5).
6. All dense linear algebra (`util/linalg.hpp` → scipy/LAPACK).
7. FFT / DCT (`util/fft.hpp` → `scipy.fft`, no FFTW/pyfftw needed).
8. Colour-space conversions except HSL (`skimage.color` + `colorsys`).
9. Otsu, histogram, image rescaling, CLAHE (skimage), background removal, normalisation, FFT filtering, neighbour averaging (kikuchipy `pattern`/`filters`/`signals`).
10. `CrystalMap` / `Phase` / `PhaseList` / IPF plotting (orix), vendor file I/O (orix.io + rsciio).
11. Gauss–Legendre roots (`numpy.polynomial.legendre.leggauss`) — replaces `square::legendre::roots` bisection outright.
12. Optimisers for peak refinement (`nlopt` 2.10.0 and `scipy.optimize`, both already wired into kikuchipy's refinement).

**Must be written (ranked by size):**

| item | est. size | smallest reasonable implementation |
|---|---|---|
| `DiscreteSHT::analyze/synthesize` on the square Lambert/Legendre grid (`sht/square_sht.hpp`) | **large** | ring-wise `scipy.fft.rfft` + on-the-fly `a^m_n/b^m_n` associated-Legendre recursion in numba; validate against `scipy.special.sph_harm_y` |
| Wigner-d tables + SO(3) cross-correlation + Newton peak refinement (`sht/wigner.hpp`, `sht/sht_xcorr.hpp`) | **large** | numba recursion (Trapani–Navaza / the `u,v,w,a,b` recursions already spelled out at `wigner.hpp:204-290`); validate against `sympy.physics.wigner.wigner_d_small`; peak refine via `nlopt` `LN_NEWUOA`/`LD_LBFGS` instead of hand-coded Newton |
| `BackProjector::unproject` detector→sphere (`modality/ebsd/detector.hpp:163-200`) | medium | precompute `(row,col)→(sphere pixel, 4 bilinear weights)` once from `_get_direction_cosines_from_detector`; `scipy.sparse.csr_matrix @ pattern.ravel()` |
| square-Legendre grid (`legendre::normals`, `boundingInds`, `MasterPattern::toLegendre/toLambert`) | medium | `leggauss` + `scipy.interpolate.RegularGridInterpolator` |
| `square::lambert::solidAngles` (Mazonka polyhedral-cone) | ~50 lines | direct port; only needs the +x/+y/45° octant then mirror (as EMSphInx does) |
| ring quadrature weights `computeWeightsSkip` (Reinecke eq. 10) | ~40 lines | direct port |
| mosaic AHE with validity mask (`util/ahe.hpp`) | ~120 lines | numba; only if skimage CLAHE parity is insufficient |
| `MasterPattern::nFoldSym/mirror/inversion/copyEquator` | ~80 lines | numpy slicing on the square grid |
| `.sht` file reader (`sht_file.hpp`) | ~150 lines | `h5py`/struct; or skip and re-derive spectra from EMsoft `.h5` |
| Nolze–Hielscher `sph2rgb` + `SphericalPatch/Triangle/Wedge` IPF colouring | ~200 lines | only if you need EMSphInx-identical IPF colours; otherwise use `orix.plot.IPFColorKeyTSL` |
| point-group metadata tables: `zRot()`→`fn`, `zMirror()`→`fm`, `mmType()`, `number()`, `tslNum()`/`FromTSL`, `hklNum()`/`FromHKL`, `schonflies()`, `symmorphic()` | ~150 lines total | one 32-row dict keyed on `orix Symmetry.name` |
| `zyz2*` / `*2zyz` wrappers | ~10 lines | `eu ± π/2` on components 0 and 2 |
| `ho2cu` / `qu2cu` (inverse cubochoric) | ~60 lines | invert `orix.quaternion._conversions.cu2ho_single`; EMSphInx `rotations.hpp` `ho2cu` is the reference |
| `nearbyQu`, `detail::orientAxis`, `quat::expl`, Rodrigues 4↔3 adapter | ~30 lines total | trivial numpy |
| DCT-based `image::imageQuality` + `image::Rescaler` | ~60 lines | `scipy.fft.dctn/idctn`; only if EMSphInx-identical IQ is required |
| `gaussian::BckgSub2D` (separable Gaussian background fit) | ~80 lines | `scipy.optimize.curve_fit` per row/column; only if `remove_dynamic_background` parity is insufficient |
| EMsoft `.nml` parser (`util/nml.hpp`) | ~60 lines | regex; or read the mirrored values from `/NMLparameters/` in the HDF5 |

**Traps to avoid**
- `orix.projections.LambertProjection` **does not exist** in orix 0.14.2 — use kikuchipy's numba Lambert functions.
- `kikuchipy.projections` **does not exist** in 0.13.dev0.
- `scipy.special.sph_harm` **was removed** in scipy 1.17 — use `sph_harm_y(n, m, theta, phi)` (note the reordered/renamed args).
- `Orientation.map_into_symmetry_reduced_zone` is **deprecated (orix 0.14, removal 0.15)** → use `Orientation.reduce()`.
- EMSphInx's `constants.hpp:59-62` comment says "beta about the y′ axis" but the code and `rotations.hpp` are ZXZ Bunge; the ZYZ form is a *separate* representation used only inside the SHT correlator.
- `Geometry::northPoleQuat()` currently returns identity (`detector.hpp:454-459`); the tilt-dependent version is commented out — don't port the dead branch.
- EMSphInx `ro` is 4-component `[nx,ny,nz,tan(w/2)]`; orix `to_rodrigues` returns 3-component `n·tan(w/2)`.
- `EBSDDetector.get_indexer` warns that `azimuthal`/`twist` are dropped when handing off to PyEBSDIndex (`_ebsd_detector.py:1657-1659`) — the same caveat applies to EMSphInx's `Geometry`, which has no twist at all.
